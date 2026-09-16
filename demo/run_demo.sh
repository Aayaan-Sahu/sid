#!/usr/bin/env bash
# The whole demo, start to finish: same engine, same model, same GPU, verifier on vs off.
#
#   bash demo/run_demo.sh
#
# Runs one server at a time (each takes most of the GPU), and writes results into demo/results/.
# Overrides: MODEL, PORT, RESULTS, REPEATS, BACKGROUND, AGENT_RUNS
set -euo pipefail

cd "$(dirname "$0")/.."
MODEL="${MODEL:-Qwen/Qwen3-8B}"
PORT="${PORT:-8000}"
RESULTS="${RESULTS:-demo/results}"
REPEATS="${REPEATS:-20}"
BACKGROUND="${BACKGROUND:-64}"
AGENT_RUNS="${AGENT_RUNS:-5}"
mkdir -p "$RESULTS"

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
fail() { printf '\n\033[31mERROR: %s\033[0m\n' "$*"; exit 1; }

GPU_UTIL="${GPU_UTIL:-0.9}"
read -r total_mib used_mib < <(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | head -1 | tr -d ',')
free_mib=$((total_mib - used_mib))
echo "GPU: ${free_mib} MiB free of ${total_mib} MiB; sid will use ${GPU_UTIL} of what is free and leave the rest"
if [ "$free_mib" -lt 24000 ]; then
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
    fail "only ${free_mib} MiB free: not enough for an 8B model (~17 GiB of weights plus kv cache). wait for memory, or run a smaller model with MODEL=Qwen/Qwen3-1.7B bash demo/run_demo.sh"
fi

server_pid=""
stop_server() {
    if [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid" 2>/dev/null || true
        for _ in $(seq 60); do kill -0 "$server_pid" 2>/dev/null || break; sleep 1; done
        kill -9 "$server_pid" 2>/dev/null || true
    fi
    server_pid=""
}
trap stop_server EXIT

start_server() {    # $1 = extra flags, $2 = log file
    local extra="$1" log="$2"
    # shellcheck disable=SC2086
    python server.py --model "$MODEL" --port "$PORT" --gpu-memory-utilization "$GPU_UTIL" $extra >"$log" 2>&1 &
    server_pid=$!
    echo -n "  starting server (loading weights, capturing cuda graphs)"
    for i in $(seq 600); do
        kill -0 "$server_pid" 2>/dev/null || { echo; tail -n 25 "$log"; fail "server exited; see $log"; }
        curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && { echo " ready after ${i}s"; return; }
        [ $((i % 10)) -eq 0 ] && echo -n "."
        sleep 1
    done
    fail "server did not become healthy; see $log"
}

run_phase() {       # $1 = label, $2 = server flags, $3 = result stem
    say "$1"
    start_server "$2" "$RESULTS/$3-server.log"
    python demo/side_by_side.py --base-url "http://localhost:$PORT/v1" --label "$1" \
        --repeats "$REPEATS" --background "$BACKGROUND" --out "$RESULTS/$3.json"
    echo "  replaying a multi-step agent $AGENT_RUNS times under load ..."
    python bench/agent_replay.py --base-url "http://localhost:$PORT/v1" --runs "$AGENT_RUNS" \
        --background "$BACKGROUND" --temperature 0.7 --seed 1234 \
        --out "$RESULTS/$3-agent.json" | tail -n 12
    stop_server
}

run_phase "verifier on"  ""                  det
run_phase "verifier off" "--no-determinism"  nodet

say "Result"
python demo/side_by_side.py --report "$RESULTS/det.json" "$RESULTS/nodet.json"
echo "raw results in $RESULTS/"
