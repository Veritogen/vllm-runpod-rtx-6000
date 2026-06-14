#!/usr/bin/env bash
# Launch the vLLM OpenAI-compatible server for the Qwen3.6 finetune on an
# RTX PRO 6000 (Blackwell, sm_120) pod.
#
# Intended to run INSIDE the RunPod pod started from the NGC vLLM container
# (nvcr.io/nvidia/vllm:26.05.post1-py3) or an equivalent CUDA-13 image.
#
# Configure via env vars (see .env.example). Only MODEL_ID is required.
set -euo pipefail

# ---- Configuration (env-overridable) ---------------------------------------
MODEL_ID="${MODEL_ID:?Set MODEL_ID to the HF repo id (or local path) of the finetune}"
# No --download-dir: weights go to the default HF cache on the container disk
# (no network volume attached; re-downloads per fresh pod, fine for testing).
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"     # NOT FLASH_ATTN (+fp8 silently fails on sm_120)
TOOL_PARSER="${TOOL_PARSER:-hermes}"                     # Qwen3 tool-call format
# Qwen3.6 GDN (Gated DeltaNet) layers fail at vLLM's default 8192; 2096 is the known-good value.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2096}"
# Optional precision: leave empty for the checkpoint's native dtype (bf16),
# or set QUANTIZATION=fp8 / auto_round to match an FP8 / INT4-AutoRound finetune.
QUANTIZATION="${QUANTIZATION:-}"
PORT="${PORT:-8000}"
EXTRA_ARGS="${EXTRA_ARGS:-}"                             # anything else, e.g. --reasoning-parser ...

# ---- Pre-flight: confirm the Blackwell/CUDA-13 stack is healthy ------------
echo "==== Pre-flight stack check ===================================="
nvidia-smi || { echo "ERROR: nvidia-smi failed — GPU/driver not visible"; exit 1; }
echo "--- nvcc (CUDA 13 toolkit must be present for FlashInfer JIT) ---"
which nvcc && nvcc --version || echo "WARN: nvcc not found — FlashInfer JIT may fail on sm_120"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cap', torch.cuda.get_device_capability())"
python -c "import vllm; print('vllm', vllm.__version__)"
echo "================================================================"

# ---- Build the argument list ----------------------------------------------
args=(
  "$MODEL_ID"
  --host 0.0.0.0 --port "$PORT"
  --attention-backend "$ATTENTION_BACKEND"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER"
)
[ -n "$QUANTIZATION" ] && args+=(--quantization "$QUANTIZATION")
# shellcheck disable=SC2206
[ -n "$EXTRA_ARGS" ] && args+=($EXTRA_ARGS)

echo "Launching: vllm serve ${args[*]}"
exec vllm serve "${args[@]}"
