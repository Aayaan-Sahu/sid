#!/usr/bin/env bash
# Start both demo servers side by side on one GPU: deterministic on :8000, non-deterministic on :8001.
#
#   bash demo/servers.sh start
#   bash demo/servers.sh stop
#
# They share whatever GPU memory is free. Overrides: MODEL, DET_PORT, NODET_PORT, DET_UTIL, NODET_UTIL, MAX_MODEL_LEN
set -euo pipefail

cd "$(dirname "$0")/.."
MODEL="${MODEL:-Qwen/Qwen3-8B}"
DET_PORT="${DET_PORT:-8000}"
NODET_PORT="${NODET_PORT:-8001}"
DET_UTIL="${DET_UTIL:-0.45}"      # fraction of the memory free when it starts
NODET_UTIL="${NODET_UTIL:-0.8}"   # of what is left after the first server has taken its share
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
LOG_DIR="${LOG_DIR:-$HOME}"

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
fail() { printf '\n\033[31mERROR: %s\033[0m\n' "$*"; exit 1; }

stop_one() {    # $1 = port
    local pidfile="$HOME/.sid-$1.pid"
    if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
        local pid; pid=$(cat "$pidfile")
        kill "$pid" 2>/dev/null || true
        for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
        kill -9 "$pid" 2>/dev/null || true
        echo "stopped server on :$1 (pid $pid)"
    else
        echo "no server recorded on :$1"
    fi
    rm -f "$pidfile"
}

if [ "${1:-start}" = "stop" ]; then
    stop_one "$DET_PORT"
    stop_one "$NODET_PORT"
    exit 0
fi

start_one() {   # $1 = port, $2 = util, $3 = dist port, $4 = label, $5 = extra flags
    local port="$1" util="$2" dist="$3" label="$4" extra="${5:-}"
    local log="$LOG_DIR/sid-$port.log" pidfile="$HOME/.sid-$port.pid"
    if curl -sf "http://localhost:$port/health" >/dev/null 2>&1; then
        echo "already serving on :$port; leaving it alone"
        return
    fi
    say "$label on :$port (log: $log)"
    # shellcheck disable=SC2086
    nohup python server.py --model "$MODEL" --port "$port" --dist-port "$dist" \
        --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$util" $extra \
        >"$log" 2>&1 < /dev/null &
    echo $! >"$pidfile"
    local pid; pid=$(cat "$pidfile")
    echo -n "  loading weights, capturing cuda graphs"
    for i in $(seq 900); do
        kill -0 "$pid" 2>/dev/null || { echo; tail -n 25 "$log"; fail "server on :$port exited; see $log"; }
        curl -sf "http://localhost:$port/health" >/dev/null 2>&1 && { echo " ready after ${i}s"; grep -m1 "kv cache:" "$log" || true; return; }
        [ $((i % 10)) -eq 0 ] && echo -n "."
        sleep 1
    done
    fail "server on :$port never became healthy; see $log"
}

read -r total_mib used_mib < <(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | head -1 | tr -d ',')
echo "GPU: $((total_mib - used_mib)) MiB free of ${total_mib} MiB (two servers will share it)"
[ $((total_mib - used_mib)) -ge 45000 ] || fail "need ~45 GB free to run two 8B servers. use MODEL=Qwen/Qwen3-1.7B, or run one server at a time"

start_one "$DET_PORT"   "$DET_UTIL"   2333 "deterministic"     ""
start_one "$NODET_PORT" "$NODET_UTIL" 2334 "non-deterministic" "--no-determinism"

say "Both up"
echo "  deterministic     http://localhost:$DET_PORT/v1"
echo "  non-deterministic http://localhost:$NODET_PORT/v1"
echo
echo "compare them with:  python demo/compare.py"
echo "stop them with:     bash demo/servers.sh stop"
