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
: "${MODEL_FILE:=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf}"
: "${MMPROJ_FILE:=mmproj-Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-f16.gguf}"
: "${ENABLE_VISION:=1}"          # 0 = text-only (skip mmproj download, faster/lighter)
: "${ALIAS:=qwen3.6}"
: "${PORT:=8000}"
: "${CTX:=0}"                    # 0 = model native max (262144 here). Plenty of VRAM at Q8.
: "${NGL:=99}"                   # offload all layers to GPU
: "${TEMP:=1.0}"                 # model card "thinking" defaults
: "${TOP_P:=0.95}"
: "${TOP_K:=20}"
# Storage: prefer an attached persistent volume, auto-detecting the mount point
# (serverless -> /runpod-volume, pod network volume -> /workspace), else local disk.
# Set MODELS_DIR explicitly to override.
if [ -z "${MODELS_DIR:-}" ]; then
    if   [ -d /runpod-volume ]; then MODELS_DIR=/runpod-volume/models   # serverless
    elif [ -d /workspace ];     then MODELS_DIR=/workspace/models       # pod network volume
    else                             MODELS_DIR=/models                 # local container disk
    fi
fi
: "${LLAMA_CACHE:=${MODELS_DIR}/cache}"   # llama.cpp -hf download cache (lives on the volume)
export LLAMA_CACHE
echo "[start] storage: MODELS_DIR=$MODELS_DIR  LLAMA_CACHE=$LLAMA_CACHE"
mkdir -p "$MODELS_DIR" "$LLAMA_CACHE"

# ---- Wait for network/DNS (RunPod networking can lag container start) -------
for i in $(seq 1 30); do
    getent hosts huggingface.co >/dev/null 2>&1 && break
    echo "[start] waiting for DNS/network ($i)..."
    sleep 2
done

# ---- Download (curl -4 + HF_TOKEN) -----------------------------------------
# llama.cpp's own -hf downloader is unreliably slow on these pods; a plain
# curl -4 (force IPv4) with the token hits full bandwidth (~200+ MB/s). Files
# land on the volume, so this is a one-time cost reused on later starts.
base_url="https://huggingface.co/${MODEL_HF_REPO}/resolve/main"
fetch() {  # $1 = filename
    local dest="${MODELS_DIR}/$1"
    if [ -s "$dest" ]; then
        echo "[start] present: $1"
        return
    fi
    echo "[start] downloading: $1"
    curl -4 -L -C - --fail --retry 5 --retry-delay 3 -o "$dest" \
        ${HF_TOKEN:+-H "Authorization: Bearer ${HF_TOKEN}"} \
        "${base_url}/$1?download=true"
}

fetch "$MODEL_FILE"
serve_args=(-m "${MODELS_DIR}/${MODEL_FILE}")
if [ "$ENABLE_VISION" = "1" ] && [ -n "$MMPROJ_FILE" ]; then
    fetch "$MMPROJ_FILE"
    serve_args+=(--mmproj "${MODELS_DIR}/${MMPROJ_FILE}")
fi

# ---- Serve (background; container stays up via sleep infinity for SSH debug) -
echo "[start] llama-server  model=${MODEL_FILE}  ctx=${CTX} ngl=${NGL} vision=${ENABLE_VISION}"
/app/llama-server \
    "${serve_args[@]}" \
    --host 0.0.0.0 --port "$PORT" \
    --alias "$ALIAS" \
    -ngl "$NGL" -c "$CTX" \
    --jinja \
    --temp "$TEMP" --top-p "$TOP_P" --top-k "$TOP_K" \
    ${EXTRA_ARGS:-} \
    2>&1 | tee /var/log/llama-server.log &

# Keep the container alive (and SSH usable) regardless of server state.
sleep infinity
