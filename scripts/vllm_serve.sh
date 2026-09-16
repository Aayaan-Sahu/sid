#!/usr/bin/env bash
# Start vLLM in the background with settings that match sid, wait until it is healthy, and smoke-test it.
#
#   bash scripts/vllm_serve.sh            # start (survives closing the Jupyter terminal)
#   bash scripts/vllm_serve.sh stop       # stop it
#
# Overrides (env vars): VLLM_HOME (~/vllm-env), MODEL (Qwen/Qwen3-8B), PORT (8001), GPU_UTIL (0.85),
# MAX_MODEL_LEN (32768), LOG (~/vllm-$PORT.log)
#
# Settings chosen to make the comparison with sid fair:
#   --default-chat-template-kwargs enable_thinking=false   sid disables Qwen3 thinking; vLLM enables it by default
#   --generation-config vllm                               don't silently apply Qwen3's top_k=20/top_p defaults
#   --tool-call-parser hermes                              Qwen3 emits hermes-style <tool_call> blocks
set -euo pipefail

VLLM_HOME="${VLLM_HOME:-$HOME/vllm-env}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
PORT="${PORT:-8001}"
GPU_UTIL="${GPU_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
LOG="${LOG:-$HOME/vllm-$PORT.log}"
PIDFILE="$HOME/.vllm-$PORT.pid"

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
fail() { printf '\n\033[31mERROR: %s\033[0m\n' "$*"; exit 1; }

if [ "${1:-}" = "stop" ]; then
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        pid=$(cat "$PIDFILE")
        kill "$pid"
        for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
        kill -9 "$pid" 2>/dev/null || true
        echo "stopped vLLM (pid $pid)"
    else
        echo "no vLLM server recorded for port $PORT"
    fi
    rm -f "$PIDFILE"
    exit 0
fi

[ -x "$VLLM_HOME/bin/vllm" ] || fail "vLLM not installed in $VLLM_HOME. run: bash scripts/vllm_setup.sh"

if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo "something is already serving on port $PORT:"
    curl -s "http://localhost:$PORT/v1/models" || true
    echo
    echo "stop it with: bash scripts/vllm_serve.sh stop   (or use PORT=... for another port)"
    exit 0
fi

say "GPU memory"
read -r total_mib used_mib < <(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | head -1 | tr -d ',')
need_mib=$(awk -v t="$total_mib" -v u="$GPU_UTIL" 'BEGIN {print int(t * u)}')
free_mib=$((total_mib - used_mib))
echo "total ${total_mib} MiB, in use ${used_mib} MiB, vLLM will claim ${need_mib} MiB (GPU_UTIL=$GPU_UTIL)"
if [ "$free_mib" -lt "$need_mib" ]; then
    # something else (usually a sid server) holds part of the card. vLLM's utilization is a fraction of TOTAL
    # memory, so shrink it to fit what is actually free instead of refusing to start
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
    GPU_UTIL=$(awk -v f="$free_mib" -v t="$total_mib" 'BEGIN {printf "%.2f", int(f * 0.92 / t * 100) / 100}')
    need_mib=$(awk -v t="$total_mib" -v u="$GPU_UTIL" 'BEGIN {print int(t * u)}')
    awk -v u="$GPU_UTIL" 'BEGIN {exit !(u >= 0.20)}' || fail "only ${free_mib} MiB free: not enough for this model. stop the process above (Ctrl-C the sid server, or kill <pid>)"
    printf '\033[33mlowering GPU_UTIL to %s (%s MiB) to fit alongside the process above; smaller kv cache, still correct\033[0m\n' "$GPU_UTIL" "$need_mib"
fi

extra_args=()
if ! command -v gcc >/dev/null && ! command -v cc >/dev/null; then
    echo "no C compiler on PATH; using --enforce-eager"
    extra_args+=(--enforce-eager)
fi

say "Starting vLLM ($MODEL on :$PORT), log: $LOG"
# a clean environment: conda's CUDA_HOME/PYTHONPATH from sid's setup confuse vLLM's kernels and JIT
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$(dirname "$VLLM_HOME")/.vllm-cache}"
nohup env -u CUDA_HOME -u PYTHONPATH -u VIRTUAL_ENV \
    "$VLLM_HOME/bin/vllm" serve "$MODEL" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --generation-config vllm \
    ${extra_args[@]+"${extra_args[@]}"} \
    >"$LOG" 2>&1 < /dev/null &
echo $! >"$PIDFILE"
pid=$(cat "$PIDFILE")

echo -n "waiting for /health (first start compiles kernels; can take a few minutes)"
for i in $(seq 900); do
    if ! kill -0 "$pid" 2>/dev/null; then
        echo
        tail -n 40 "$LOG"
        fail "vLLM exited during startup; the log tail is above (full log: $LOG)"
    fi
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo " up after ${i}s"
        break
    fi
    [ $((i % 10)) -eq 0 ] && echo -n "."
    sleep 1
    [ "$i" -eq 900 ] && fail "still not healthy after 15 minutes; check $LOG"
done

say "Smoke test: tool call with thinking disabled"
response=$(curl -s "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' -d "{
  \"model\": \"$MODEL\",
  \"messages\": [{\"role\": \"user\", \"content\": \"What's the weather in Paris right now?\"}],
  \"tools\": [{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get current weather for a city\",
              \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}],
  \"temperature\": 0,
  \"max_tokens\": 256
}")
"$VLLM_HOME/bin/python" - "$response" <<'EOF'
import json, sys
r = json.loads(sys.argv[1])
if "choices" not in r:
    sys.exit(f"unexpected response: {r}")
message = r["choices"][0]["message"]
calls = message.get("tool_calls") or []
content = message.get("content") or ""
print("content:   ", repr(content[:200]))
print("tool_calls:", [(c["function"]["name"], c["function"]["arguments"]) for c in calls])
if "<think>" in content:
    sys.exit("FAIL: thinking is still on (<think> in content)")
if not calls:
    sys.exit("FAIL: no tool call parsed (check --tool-call-parser)")
print("smoke test passed")
EOF

echo
echo "vLLM is running (pid $pid) at http://localhost:$PORT/v1"
echo "stop it with: bash scripts/vllm_serve.sh stop"
