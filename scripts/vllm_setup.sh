#!/usr/bin/env bash
# Install vLLM into its own virtualenv, for comparing against sid.
#
#   bash scripts/vllm_setup.sh
#
# Never install vLLM into sid's environment: vLLM wheels pin their own torch/CUDA build and would replace
# the torch that sid's flash-attn-3 was built against.
#
# vLLM on PyPI is built for CUDA 13 (it needs libcudart.so.13 and a driver that supports CUDA 13). Drivers
# that only support CUDA 12 need the +cu129 wheel from vLLM's GitHub release plus a cu129 torch. This script
# reads the driver's supported CUDA version from nvidia-smi and installs the matching pair.
#
# Overrides (env vars):
#   VLLM_HOME      where the venv goes (default ~/vllm-env). put it on a disk with >= 20 GB free.
#   VLLM_VERSION   vLLM version to install (default: latest on PyPI)
#   VLLM_CUDA      force cu129 or cu130 instead of detecting it from the driver
set -euo pipefail

VLLM_HOME="${VLLM_HOME:-$HOME/vllm-env}"
STATE_DIR="$(dirname "$VLLM_HOME")"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$STATE_DIR/.uv-cache}"    # keep the multi-GB wheel cache next to the venv
export UV_LINK_MODE=copy

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
fail() { printf '\n\033[31mERROR: %s\033[0m\n' "$*"; exit 1; }
warn() { printf '\033[33mWARNING: %s\033[0m\n' "$*"; }

say "GPU"
command -v nvidia-smi >/dev/null || fail "nvidia-smi not found: this shell has no GPU. open the terminal inside the GPU Jupyter session."
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader
driver_cuda=$(nvidia-smi | grep -oE 'CUDA Version: *[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)
[ -n "$driver_cuda" ] || fail "could not read the driver's CUDA version from nvidia-smi"
driver_cuda_major=${driver_cuda%%.*}
echo "driver supports CUDA up to $driver_cuda"
used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "${used_mib:-0}" -gt 4000 ]; then
    warn "${used_mib} MiB of GPU memory is in use (a sid server?). setup is fine, but stop it before starting vLLM:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
fi

say "Choosing the vLLM build"
if [ -z "${VLLM_VERSION:-}" ]; then
    VLLM_VERSION=$(curl -sf https://pypi.org/pypi/vllm/json | grep -oE '"version": *"[^"]+"' | head -1 | grep -oE '[0-9][^"]*' || true)
    [ -n "$VLLM_VERSION" ] || fail "could not look up the latest vLLM version on PyPI; set VLLM_VERSION=... and rerun"
fi
if [ -z "${VLLM_CUDA:-}" ]; then
    if [ "$driver_cuda_major" -ge 13 ]; then
        VLLM_CUDA=cu130
    elif [ "$driver_cuda_major" -eq 12 ]; then
        VLLM_CUDA=cu129    # CUDA 12 minor-version compatibility lets 12.9 builds run on any CUDA 12 driver
    else
        fail "driver only supports CUDA $driver_cuda; current vLLM needs CUDA 12 or newer"
    fi
fi
case "$VLLM_CUDA" in
    cu130) vllm_spec="vllm==$VLLM_VERSION" ;;
    cu129) vllm_spec="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu129-cp38-abi3-manylinux_2_28_$(uname -m).whl"
           curl -sfIL -o /dev/null "$vllm_spec" || fail "no cu129 wheel published for vLLM $VLLM_VERSION ($vllm_spec). set VLLM_VERSION to an earlier release and rerun" ;;
    *) fail "VLLM_CUDA must be cu129 or cu130, got $VLLM_CUDA" ;;
esac
echo "vLLM $VLLM_VERSION, $VLLM_CUDA build, with a $VLLM_CUDA torch"

say "Disk"
mkdir -p "$STATE_DIR"
free_gb=$(df -Pk "$STATE_DIR" | awk 'NR==2 {print int($4 / 1024 / 1024)}')
echo "$STATE_DIR has ${free_gb} GB free"
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
    [ -x "$HOME/.local/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
if ! uv pip install --help 2>/dev/null | grep -q "$VLLM_CUDA"; then
    echo "this uv is too old to know the $VLLM_CUDA torch backend; updating uv"
    uv self update 2>/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; hash -r; }
fi
uv --version

check_vllm() {
    env -u CUDA_HOME -u PYTHONPATH EXPECTED_CUDA="$VLLM_CUDA" "$VLLM_HOME/bin/python" - <<'EOF'
import os, torch
expected = os.environ["EXPECTED_CUDA"]    # cu129 -> 12.9, cu130 -> 13.0
want = f"{expected[2:-1]}.{expected[-1]}"
assert torch.version.cuda == want, f"torch is built for CUDA {torch.version.cuda}, expected {want}"
assert torch.cuda.is_available(), f"torch {torch.__version__} (CUDA {torch.version.cuda}) cannot see the GPU"
import vllm, vllm.platforms
vllm.platforms.current_platform    # loads vLLM's compiled CUDA kernels; this is what failed with libcudart.so.13
print(f"vllm {vllm.__version__} | torch {torch.__version__} | CUDA {torch.version.cuda} | {torch.cuda.get_device_name(0)}")
EOF
}

if [ -x "$VLLM_HOME/bin/python" ] && check_vllm 2>/dev/null; then
    say "vLLM already installed and working in $VLLM_HOME"
    check_vllm
    exit 0
fi

if [ -e "$VLLM_HOME" ]; then
    [ -f "$VLLM_HOME/pyvenv.cfg" ] || fail "$VLLM_HOME exists but is not a virtualenv; refusing to delete it. set VLLM_HOME to another path"
    say "Removing the existing non-working venv at $VLLM_HOME"
    rm -rf "$VLLM_HOME"
fi
[ "$free_gb" -ge 20 ] || fail "need >= 20 GB free for the vLLM venv and wheel cache. rerun with VLLM_HOME=/path/on/a/bigger/disk/vllm-env"

say "Installing into $VLLM_HOME (takes a few minutes)"
uv venv --python 3.12 --seed "$VLLM_HOME"
env -u CUDA_HOME -u PYTHONPATH uv pip install --python "$VLLM_HOME/bin/python" "$vllm_spec" --torch-backend="$VLLM_CUDA"

say "Verifying"
if ! check_vllm; then
    echo
    echo "driver CUDA: $driver_cuda | vLLM build: $VLLM_CUDA | vLLM version: $VLLM_VERSION"
    fail "vLLM installed but does not load (see the error above). send that output along with this summary"
fi
echo
echo "done. start the server with: bash scripts/vllm_serve.sh"
