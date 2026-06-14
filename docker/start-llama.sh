#!/usr/bin/env bash
# Pod entrypoint: start sshd (for debugging), then autostart llama-server to serve
# the model. The server runs in the background and the container stays alive via
# `sleep infinity`, so an SSH session remains usable even if the server crashes.
set -u

# ---- SSH (debugging) -------------------------------------------------------
mkdir -p ~/.ssh /run/sshd
if [ -n "${PUBLIC_KEY:-}" ]; then
    echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
    chmod 700 ~/.ssh
fi
ssh-keygen -A >/dev/null 2>&1 || true
service ssh start || /usr/sbin/sshd

# ---- Config (all env-overridable; defaults baked in the Dockerfile) --------
: "${MODEL_HF_REPO:=HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive}"
: "${MODEL_QUANT:=Q8_K_P}"
: "${MMPROJ_FILE:=mmproj-Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-f16.gguf}"
: "${ENABLE_VISION:=1}"          # 0 = text-only (skip mmproj download, faster/lighter)
: "${ALIAS:=qwen3.6}"
: "${PORT:=8000}"
: "${CTX:=0}"                    # 0 = model native max (262144 here). Plenty of VRAM at Q8.
: "${NGL:=99}"                   # offload all layers to GPU
: "${TEMP:=1.0}"                 # model card "thinking" defaults
: "${TOP_P:=0.95}"
: "${TOP_K:=20}"
: "${MODELS_DIR:=/models}"
: "${LLAMA_CACHE:=${MODELS_DIR}/cache}"   # llama.cpp -hf download cache (put on a volume to persist)
export LLAMA_CACHE
mkdir -p "$MODELS_DIR" "$LLAMA_CACHE"

# ---- Vision projector (mmproj) download ------------------------------------
mmproj_args=()
if [ "$ENABLE_VISION" = "1" ] && [ -n "$MMPROJ_FILE" ]; then
    mmproj_path="${MODELS_DIR}/${MMPROJ_FILE}"
    if [ ! -f "$mmproj_path" ]; then
        echo "[start] downloading vision projector: $MMPROJ_FILE"
        curl -L -C - --fail -o "$mmproj_path" \
            ${HF_TOKEN:+-H "Authorization: Bearer ${HF_TOKEN}"} \
            "https://huggingface.co/${MODEL_HF_REPO}/resolve/main/${MMPROJ_FILE}?download=true"
    fi
    mmproj_args=(--mmproj "$mmproj_path")
fi

# ---- Serve (background; -hf downloads + caches the Q8 GGUF on first start) --
echo "[start] llama-server  model=${MODEL_HF_REPO}:${MODEL_QUANT}  ctx=${CTX} ngl=${NGL} vision=${ENABLE_VISION}"
llama-server \
    -hf "${MODEL_HF_REPO}:${MODEL_QUANT}" \
    "${mmproj_args[@]}" \
    --host 0.0.0.0 --port "$PORT" \
    --alias "$ALIAS" \
    -ngl "$NGL" -c "$CTX" \
    --jinja \
    --temp "$TEMP" --top-p "$TOP_P" --top-k "$TOP_K" \
    ${EXTRA_ARGS:-} \
    2>&1 | tee /var/log/llama-server.log &

# Keep the container alive (and SSH usable) regardless of server state.
sleep infinity
