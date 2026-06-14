# Serverless Qwen3.6 finetune on RunPod (RTX PRO 6000 / Blackwell)

Scale-to-zero hosting of a ~27B/35B Qwen3.6 finetune for personal coding use, served via vLLM with
an OpenAI-compatible API and tool calling. Full rationale and the confirmed CUDA-13/Blackwell
diagnosis live in the plan: `~/.claude/plans/i-want-to-build-replicated-lobster.md`.

## Why the stock RunPod template failed
RTX PRO 6000 is Blackwell (`sm_120`). vLLM/FlashInfer JIT-compile kernels for `sm_120` at load time
and need the **full CUDA 13 toolkit** (`nvcc`, `cccl`, `cudart-dev`), not just the PyPI runtime the
stock template ships → "can't find CUDA 13 things". The NGC vLLM container ships all of it
pre-integrated, so we use that as the base.

## Layout
- `scripts/start-vllm.sh` — launch vLLM with the Qwen3.6-aware flags (run inside the pod).
- `scripts/test_endpoint.py` — verify completion + tool calling against pod or serverless.
- `docker/Dockerfile.pod` — optional pinned image once the working combo is known.
- `.env.example` — config template.

## Phase 1 — bring up on an on-demand pod
1. Rent an **RTX PRO 6000 (96GB)** on-demand pod from `nvcr.io/nvidia/vllm:26.05.post1-py3`
   (anonymously pullable — no NGC login). Attach a **~100GB network volume** (mounts at
   `/runpod-volume`).
2. In the pod, set env and (if the repo is private) authenticate to HF:
   ```bash
   export MODEL_ID=your-org/your-qwen3.6-finetune
   export HF_TOKEN=hf_xxx           # only if private
   ```
3. Launch and watch the pre-flight checks + load:
   ```bash
   ./scripts/start-vllm.sh
   ```
   Troubleshooting is inline in the script comments and the plan:
   - "can't find CUDA 13 / FlashInfer JIT" → toolkit missing (won't happen on the NGC base).
   - "Engine core initialization failed" → Qwen3.6 GDN; keep `--max-num-batched-tokens 2096`.
   - OOM → lower `MAX_MODEL_LEN`/`GPU_MEM_UTIL`, or set `QUANTIZATION=fp8`.
4. From a second shell in the pod, verify:
   ```bash
   pip install openai
   BASE_URL=http://localhost:8000/v1 API_KEY=EMPTY MODEL_ID=$MODEL_ID python scripts/test_endpoint.py
   ```
5. Pin the working versions into `docker/Dockerfile.pod` (and the vLLM args you settled on).

> Note: don't try to build/run this locally — the laptop GPU (GTX 1650, 4GB, `sm_75`, driver
> offloaded) can't run a 27B model or represent Blackwell, and disk is too small for the image.

## Phase 2 — port to serverless (after Phase 1 works)
- Build the serverless image from the same pinned layers; swap the entrypoint for RunPod's
  serverless **handler** (or apply the pins to RunPod `worker-vllm`).
- Create a Serverless Endpoint: RTX PRO 6000, attach the **same network volume**, `min=0`/`max=1`,
  **FlashBoot on**, **idle timeout 120–300s** (stays warm across a coding session, then sleeps to $0).
- Verify against `https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1` with `scripts/test_endpoint.py`,
  then confirm workers scale to 0 between requests.

## Open inputs to confirm at the pod
- Exact HF model id + format (bf16 vs FP8 vs INT4-AutoRound) and whether the repo is private.
- Whether quality holds in FP8 (faster/cheaper) or bf16 is needed (your call after testing).
