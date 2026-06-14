#!/bin/bash
# RunPod pod start script: seed the injected SSH key, start sshd, keep alive.
# Mirrors runpod/containers' container-template/start.sh, minimal (no nginx/jupyter).
set -e

mkdir -p ~/.ssh
if [ -n "${PUBLIC_KEY:-}" ]; then
    echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
    chmod 700 -R ~/.ssh
fi

# Generate host keys if missing, then start sshd.
ssh-keygen -A >/dev/null 2>&1 || true
mkdir -p /run/sshd
service ssh start || /usr/sbin/sshd

# Optional: auto-launch vLLM if MODEL_ID is set; otherwise just hold the pod open
# so you can iterate in a shell (web terminal or SSH).
if [ -n "${MODEL_ID:-}" ] && [ "${AUTOSTART_VLLM:-0}" = "1" ]; then
    /usr/local/bin/start-vllm.sh &
fi

sleep infinity
