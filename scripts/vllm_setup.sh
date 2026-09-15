#!/usr/bin/env bash
# Install vLLM into its own virtualenv, for comparing against sid.
#
#   bash scripts/vllm_setup.sh
#
# Never install vLLM into sid's environment: vLLM wheels pin their own torch/CUDA build and would replace
# the torch that sid's flash-attn-3 was built against.
#
# Overrides (env vars):
#   VLLM_HOME       where the venv goes (default ~/vllm-env). put it on a disk with >= 20 GB free.
#   TORCH_BACKEND   uv torch backend (default auto; picks the right CUDA build for the installed driver).
#                   if torch can't see the GPU afterwards, rerun with TORCH_BACKEND=cu126
set -euo pipefail

VLLM_HOME="${VLLM_HOME:-$HOME/vllm-env}"
TORCH_BACKEND="${TORCH_BACKEND:-auto}"
STATE_DIR="$(dirname "$VLLM_HOME")"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$STATE_DIR/.uv-cache}"    # keep the multi-GB wheel cache next to the venv
export UV_LINK_MODE=copy

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
fail() { printf '\n\033[31mERROR: %s\033[0m\n' "$*"; exit 1; }
warn() { printf '\033[33mWARNING: %s\033[0m\n' "$*"; }

say "GPU"
command -v nvidia-smi >/dev/null || fail "nvidia-smi not found: this shell has no GPU. open the terminal inside the GPU Jupyter session."
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader
used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "${used_mib:-0}" -gt 4000 ]; then
    warn "${used_mib} MiB of GPU memory is in use (a sid server?). setup is fine, but stop it before starting vLLM:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
fi

say "Disk"
mkdir -p "$STATE_DIR"
free_gb=$(df -Pk "$STATE_DIR" | awk 'NR==2 {print int($4 / 1024 / 1024)}')
echo "$STATE_DIR has ${free_gb} GB free"
[ "$free_gb" -ge 20 ] || fail "need >= 20 GB free for the vLLM venv and wheel cache. rerun with VLLM_HOME=/path/on/a/bigger/disk/vllm-env"
hf_cache="${HF_HOME:-$HOME/.cache/huggingface}/hub"
if ls -d "$hf_cache"/models--Qwen--Qwen3-8B >/dev/null 2>&1; then
    echo "Qwen3-8B already in $hf_cache (sid downloaded it); vLLM will reuse it"
else
    echo "Qwen3-8B not in $hf_cache yet; vLLM will download ~16 GB on first start"
fi

say "Environment hygiene"
if [ -n "${CONDA_PREFIX:-}" ]; then
    warn "a conda env is active ($CONDA_PREFIX). that's OK for this script (it uses its own python), but serve with scripts/vllm_serve.sh, which clears conda's CUDA_HOME."
fi
if command -v gcc >/dev/null || command -v cc >/dev/null; then
    echo "C compiler found ($(command -v gcc || command -v cc)); torch.compile/CUDA graphs will work"
else
    warn "no C compiler found; vllm_serve.sh will fall back to --enforce-eager (slower, still correct)"
fi

say "uv"
if ! command -v uv >/dev/null; then
    if [ -x "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    else
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi
uv --version

check_vllm() {
    env -u CUDA_HOME -u PYTHONPATH "$VLLM_HOME/bin/python" - <<'EOF'
import torch, vllm
assert torch.cuda.is_available(), f"torch {torch.__version__} (cuda {torch.version.cuda}) cannot see the GPU"
print(f"vllm {vllm.__version__} | torch {torch.__version__} | cuda {torch.version.cuda} | {torch.cuda.get_device_name(0)}")
EOF
}

if [ -x "$VLLM_HOME/bin/python" ] && check_vllm 2>/dev/null; then
    say "vLLM already installed in $VLLM_HOME"
    check_vllm
    exit 0
fi

say "Installing vLLM into $VLLM_HOME (torch backend: $TORCH_BACKEND); takes a few minutes"
uv venv --python 3.12 --seed "$VLLM_HOME"
env -u CUDA_HOME -u PYTHONPATH VIRTUAL_ENV="$VLLM_HOME" uv pip install --python "$VLLM_HOME/bin/python" vllm --torch-backend="$TORCH_BACKEND"

say "Verifying"
if ! check_vllm; then
    fail "vLLM installed but torch can't use the GPU (usually a driver/CUDA build mismatch). run: rm -rf $VLLM_HOME && TORCH_BACKEND=cu126 bash scripts/vllm_setup.sh"
fi
echo
echo "done. start the server with: bash scripts/vllm_serve.sh"
