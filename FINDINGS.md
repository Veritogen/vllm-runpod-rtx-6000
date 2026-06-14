# Findings: serving Qwen3.6-40B on RunPod (RTX PRO 6000 / Blackwell)

Self-hosting `DavidAU/Qwen3.6-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking` for
personal coding via opencode, on a single RTX PRO 6000 (96GB). Eventual goal: scale-to-zero
serverless. This file records the working config and every issue we hit, so a rebuild "just works."

## Environment
- **GPU:** RTX PRO 6000 Blackwell, 96GB, compute capability **sm_120**, driver 580.x / CUDA 13.0.
- **Base image:** `nvcr.io/nvidia/vllm:26.05.post1-py3` (vLLM 0.21.x dev, CUDA 13.1, FlashInfer
  prebuilt). Anonymously pullable from GHCR-mirrored NGC; chosen because NVIDIA pre-integrates the
  CUDA13/Blackwell stack — but it still needs the env wiring below.
- **Model:** 40B, **hybrid GDN** (48 Gated-DeltaNet linear-attn + 16 full-attn layers), **vision
  tower** (Qwen3-VL based), **reasoning/"thinking"** model. bf16 weights load ~74 GiB into VRAM,
  leaving ~22 GiB for KV cache.

## Working vLLM launch (verified: chat + tool calling)
Required **env** (bake into the image — lost on every SSH reconnect otherwise):
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export CPATH=/usr/local/cuda/include          # -> cuda.h for Triton JIT
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas   # ptxas lives at /usr/local/cuda-13.2/bin
```
Launch:
```bash
HF_TOKEN=hf_xxx vllm serve "DavidAU/Qwen3.6-40B-...-Thinking" \
  --served-model-name qwen3.6 \
  --host 0.0.0.0 --port 8000 \
  --attention-backend flashinfer \
  --max-num-batched-tokens 2096 \      # Qwen3.6 GDN requires this (default 8192 fails)
  --max-model-len 32768 \              # 262144 native -> OOM; keep modest (bf16 leaves ~22GB)
  --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \     # NOT hermes — model uses <function=..><parameter=..> syntax
  --reasoning-parser qwen3 \           # thinking model: separates <think> from output
  --enforce-eager                      # fast startup while iterating; see Performance to remove
```

## Issues hit → fixes (chronological)
1. **`cuda.h: No such file` during GDN Triton JIT** → header exists at `/usr/local/cuda/include`;
   put it on the compiler path: `export CPATH=/usr/local/cuda/include`.
2. **`RuntimeError: Cannot find ptxas`** (Triton needs ptxas to compile vision/GDN kernels; it
   checks `TRITON_PTXAS_PATH`, not `PATH`). ptxas is present at `/usr/local/cuda-13.2/bin/ptxas` →
   `export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`.
3. **Tool calls not parsed** — model emits the **Qwen3-Coder** format
   (`<tool_call><function=get_weather><parameter=city>Berlin</parameter></function></tool_call>`),
   which `--tool-call-parser hermes` cannot read → tool_calls empty, raw XML in `content`, opencode
   loops. Fix: **`--tool-call-parser qwen3_coder`**. Verified: `finish_reason: "tool_calls"` with a
   structured call afterward.
4. **Agent loops / empty returns / "thinks about what to do"** — reasoning model; without a
   reasoning parser the `<think>` block isn't separated. Fix: `--reasoning-parser qwen3`.
5. **OOM after load at 262k context** — native `max_model_len` is 262144; KV cache won't fit after
   74 GiB of weights. Set `--max-model-len` modest (32k–100k depending on KV budget).
6. **Env vars lost after SSH reconnect** — they're per-shell. Bake into image `ENV` (durable) or
   append to `~/.bashrc` (per-pod).
7. **Slow startup** — two compiles: (a) Triton vision/GDN kernels → cache in `~/.cache/triton`;
   (b) `torch.compile`/Inductor (only when `--enforce-eager` is OFF), config-keyed so changing
   flags forces a recompile. The vision-encoder profiling is the slowest step (it's the VL tower).

## Startup caching (the real cold-start fix, pod + serverless)
Attach a network volume and redirect caches to it so compiles/downloads happen **once** and are
reused by every later start, including fresh serverless hosts mounting the same volume:
```bash
export HF_HOME=/runpod-volume/cache/hf            # weights, no re-download
export TRITON_CACHE_DIR=/runpod-volume/cache/triton
export VLLM_CACHE_ROOT=/runpod-volume/cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/runpod-volume/cache/inductor
```
Caveat: caching only helps for steps that **complete**; a stuck compile has nothing to cache.
Mount the pod volume at `/runpod-volume` to match serverless.

## Image notes
- The NGC base needs: `openssh-server` + RunPod start script (seed `$PUBLIC_KEY`, `service ssh
  start`, `sleep infinity`) + the 4 `ENV` lines above. See `docker/Dockerfile.runpod-pod`.
- Built via GitHub Actions → GHCR (`.github/workflows/build-pod-image.yml`); make the package
  Public so RunPod pulls without creds. Image name must be **lowercase**.
- Pull/extract is slow (~tens of GB, one-time per host). **Stop, don't Terminate** to keep it
  cached on the host. This pull cost is the strongest argument against baking weights into the image.

## Performance
- **~12–15 tok/s** for 40B bf16. Decode is **memory-bandwidth-bound**: theoretical ceiling ≈
  GPU bandwidth (~1.8 TB/s) ÷ weights (~80 GB) ≈ ~22 tok/s, so we're near the bf16 limit *and*
  losing more to eager mode.
- **Levers WITHOUT new hardware (biggest first):**
  1. **Quantize.** FP8 (~40 GB) ≈ ~2× decode; GGUF Q6/Q4 even more. Decode is bandwidth-bound, so
     halving the bytes ≈ doubling tok/s. Biggest single win.
  2. **Drop `--enforce-eager`** (enable CUDA graphs + torch.compile) → ~+20–40% decode; pay the
     one-time compile, then cache it (see above).
  3. Speculative decoding does **not** help much here — GDN layers don't parallelize draft
     verification.
- **A faster GPU isn't the first move.** The RTX PRO 6000 is already high-bandwidth; FP8 + CUDA
  graphs on the same card should reach ~30–45 tok/s. (An H200/B200 has more bandwidth but is far
  pricier; only worth it if quantization isn't enough.)
- Perceived slowness is amplified because it's a **thinking model** — it spends tokens reasoning
  before answering.

## Alternatives considered
- **GGUF + llama.cpp** (`ghcr.io/ggml-org/llama.cpp:server-cuda13`, works on sm_120; avoid MXFP4
  quants): tiny image, fast `mmap` load, lower VRAM, no Triton/ptxas/torch.compile/vision-encoder
  pain. A Q6_K GGUF of this model exists (~97% of bf16). **But it's text-only — no vision.** Use it
  if multimodal isn't required. See `docker/Dockerfile.llamacpp`.

## Open / next steps
- Bake the working flags + 4 `ENV` lines into `docker/Dockerfile.runpod-pod` (and cache-dir ENVs).
- Decide **vision vs text-only** — drives vLLM(+VL) vs llama.cpp(GGUF).
- Try **FP8** for ~2× throughput while keeping vLLM + tool calling.
- Serverless (Phase 2): network volume for cache+weights, `min=0`/FlashBoot, idle timeout 2–5 min.

## llama.cpp path — `HauhauCS/Qwen3.6-35B-A3B` Q8 (the working serving setup)
Pivoted here for speed + simplicity. MoE (~3B active) → **fast tokens**; **vision via mmproj**;
**tool calling works with just `--jinja`** (no parser flag needed, unlike vLLM's `qwen3_coder`).
See `docker/Dockerfile.llamacpp-server` + `docker/start-llama.sh`.

- **Base:** `ghcr.io/ggml-org/llama.cpp:server-cuda13` (works on sm_120). Binary at `/app/llama-server`;
  its `.so`s are in `/app` → set **`LD_LIBRARY_PATH=/app`** (and `PATH=/app`) or it fails with
  `command not found` / `libllama-server-impl.so: cannot open`.
- **Networking gotcha:** these pods have **dead IPv6 egress**. huggingface.co resolves IPv6-first, so
  anything not forcing IPv4 crawls. Forced-IPv4 curl hit **~220 MB/s** to the volume.
  - `Dockerfile` adds `precedence ::ffff:0:0/96 100` to `/etc/gai.conf` (prefer IPv4).
  - **llama.cpp's own `-hf` downloader is unreliably slow regardless** (even with token + IPv4 pref) →
    **don't use `-hf`.** Download with `curl -4 -L` + `HF_TOKEN`, then serve local files with `-m`.
- **Token:** HF gives authenticated traffic much better bandwidth → always pass `HF_TOKEN`.
- **Context:** `-c 0` = native max 262144 (kept for the big window) but it makes startup slower +
  more KV VRAM (the `-fit` probe). Lower `-c` for faster cold starts.
- **Storage:** `start-llama.sh` auto-detects the mount (`/runpod-volume` serverless, `/workspace` pod,
  else local). Pre-warm once → reused. Cold load is ~44GB → VRAM (~2–3 min from the volume).

Working manual command (what the image automates):
```bash
cd /workspace/models
B=https://huggingface.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive/resolve/main
curl -4 -L -C - --fail -H "Authorization: Bearer $HF_TOKEN" -o model-Q8_K_P.gguf "$B/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf"
curl -4 -L -C - --fail -H "Authorization: Bearer $HF_TOKEN" -o mmproj-f16.gguf "$B/mmproj-Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-f16.gguf"
LD_LIBRARY_PATH=/app /app/llama-server -m model-Q8_K_P.gguf --mmproj mmproj-f16.gguf \
  --host 0.0.0.0 --port 8000 --alias qwen3.6 -ngl 99 -c 0 --jinja
```
