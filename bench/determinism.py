"""Determinism benchmark for OpenAI-compatible endpoints (sid, vLLM, SGLang, ...).

For every target conversation it collects outputs under several conditions and counts how many distinct
outputs each endpoint produced:

    isolated-cold   each target alone, one at a time
    isolated-warm   the same again (prefix cache now warm)
    load-c<N>       targets fired at random moments while N concurrent background requests run

Run it against each server with the same arguments:

    python bench/determinism.py --endpoint sid=http://localhost:8000 --endpoint vllm=http://localhost:8001 \
        --targets 8 --repeats 3 --concurrency 32,128 --out results.json

A target is deterministic if every run produced the exact same content and tool calls.
"""
import argparse
import asyncio
import hashlib
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

from bench.workload import filler_conversations, target_conversations

SAMPLING = {
    "greedy": {"temperature": 0.0},
    "sampled": {"temperature": 0.7, "top_p": 0.95, "seed": 1234},
}


async def chat(client: httpx.AsyncClient, base_url: str, model: str, conv: dict, sampling: dict, max_tokens: int, api_key: str | None) -> dict:
    body = {
        "model": model,
        "messages": conv["messages"],
        "tools": conv.get("tools"),
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        **sampling,
    }
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    started = time.perf_counter()
    ttft = None
    content = []
    calls: dict[int, dict] = {}
    result = {"usage": None, "finish_reason": None, "sid": None, "fingerprint": None}
    try:
        async with client.stream("POST", f"{base_url}/v1/chat/completions", json=body, headers=headers) as r:
            if r.status_code != 200:
                return {"error": f"HTTP {r.status_code}: {(await r.aread())[:300]!r}"}
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if "error" in obj:
                    return {"error": str(obj["error"])}
                result["fingerprint"] = obj.get("system_fingerprint") or result["fingerprint"]
                result["usage"] = obj.get("usage") or result["usage"]
                result["sid"] = obj.get("sid") or result["sid"]
                for choice in obj.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        ttft = ttft or time.perf_counter() - started
                        content.append(delta["content"])
                    for tc in delta.get("tool_calls") or []:
                        ttft = ttft or time.perf_counter() - started
                        slot = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                        fn = tc.get("function") or {}
                        slot["name"] += fn.get("name") or ""
                        slot["arguments"] += fn.get("arguments") or ""
                    result["finish_reason"] = choice.get("finish_reason") or result["finish_reason"]
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        return {"error": repr(e)}
    output = {"content": "".join(content), "tool_calls": [calls[i] for i in sorted(calls)]}
    result.update(
        output=output,
        output_key=hashlib.sha256(json.dumps(output, sort_keys=True).encode()).hexdigest(),
        ttft=ttft,
        latency=time.perf_counter() - started,
        completion_tokens=(result["usage"] or {}).get("completion_tokens", 0),
    )
    return result


async def run_background(client, base_url, model, fillers, concurrency, api_key, counters):
    async def worker(worker_id: int):
        i = worker_id * 7919
        while True:
            conv = fillers[i % len(fillers)]
            i += 1
            r = await chat(client, base_url, model, conv, conv["sampling"], conv["max_tokens"], api_key)
            if "error" in r:
                counters["errors"] += 1
                await asyncio.sleep(0.5)
            else:
                counters["requests"] += 1
                counters["tokens"] += r["completion_tokens"]
    return [asyncio.create_task(worker(w)) for w in range(concurrency)]


async def bench_endpoint(name: str, base_url: str, args) -> dict:
    limits = httpx.Limits(max_connections=max(args.concurrency) + args.targets + 16, max_keepalive_connections=64)
    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0), limits=limits) as client:
        headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
        model = args.model or (await client.get(f"{base_url}/v1/models", headers=headers)).json()["data"][0]["id"]
        targets = target_conversations(args.targets, seed=args.seed)
        fillers = filler_conversations(512, seed=args.seed + 1)
        runs = []    # dicts: sampling, condition, target, output_key, output, ttft, latency
        load_stats = []

        def record(sampling_name, condition, index, r):
            if "error" in r:
                print(f"  [{name}] {condition} target {index}: {r['error']}")
                runs.append({"sampling": sampling_name, "condition": condition, "target": index, "error": r["error"]})
                return
            runs.append({"sampling": sampling_name, "condition": condition, "target": index, **{k: r[k] for k in ("output_key", "output", "ttft", "latency", "fingerprint", "sid")}})

        for sampling_name in args.sampling:
            sampling = SAMPLING[sampling_name]
            for condition in ("isolated-cold", "isolated-warm"):
                print(f"[{name}] {sampling_name} {condition}")
                for i, conv in enumerate(targets):
                    record(sampling_name, condition, i, await chat(client, base_url, model, conv, sampling, args.max_tokens, args.api_key))

            for concurrency in args.concurrency:
                condition = f"load-c{concurrency}"
                print(f"[{name}] {sampling_name} {condition}: warming up background load")
                counters = {"requests": 0, "tokens": 0, "errors": 0}
                workers = await run_background(client, base_url, model, fillers, concurrency, args.api_key, counters)
                await asyncio.sleep(args.warmup)
                window_start, tokens_start = time.perf_counter(), counters["tokens"]
                rng = random.Random(f"{args.seed}:{sampling_name}:{concurrency}")
                for repeat in range(args.repeats):
                    async def fire(i, conv):
                        await asyncio.sleep(rng.uniform(0, args.jitter))
                        return i, await chat(client, base_url, model, conv, sampling, args.max_tokens, args.api_key)
                    order = list(enumerate(targets))
                    rng.shuffle(order)
                    for i, r in await asyncio.gather(*[fire(i, conv) for i, conv in order]):
                        record(sampling_name, condition, i, r)
                    print(f"[{name}] {sampling_name} {condition}: repeat {repeat + 1}/{args.repeats} done")
                window = time.perf_counter() - window_start
                for w in workers:
                    w.cancel()
                await asyncio.gather(*workers, return_exceptions=True)
                load_stats.append({
                    "sampling": sampling_name, "concurrency": concurrency,
                    "background_output_tokens_per_second": (counters["tokens"] - tokens_start) / window,
                    "background_requests": counters["requests"], "background_errors": counters["errors"],
                })

        metrics = None
        try:
            metrics = (await client.get(f"{base_url}/metrics")).json()
        except Exception:
            pass
    return {"name": name, "base_url": base_url, "model": model, "runs": runs, "load": load_stats, "server_metrics": metrics}


def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def summarize(result: dict, args) -> str:
    lines = [f"## {result['name']}  ({result['model']} @ {result['base_url']})", ""]
    fingerprints = {r.get("fingerprint") for r in result["runs"] if r.get("fingerprint")}
    if fingerprints:
        lines.append(f"system_fingerprint: {', '.join(sorted(fingerprints))}")
        lines.append("")
    lines.append("| sampling | condition | runs | identical to isolated-cold | errors |")
    lines.append("|---|---|---:|---:|---:|")
    summary_rows = []
    for sampling_name in args.sampling:
        runs = [r for r in result["runs"] if r["sampling"] == sampling_name]
        reference = {r["target"]: r for r in runs if r["condition"] == "isolated-cold" and "error" not in r}
        conditions = list(dict.fromkeys(r["condition"] for r in runs))
        for condition in conditions:
            cond_runs = [r for r in runs if r["condition"] == condition]
            ok = [r for r in cond_runs if "error" not in r]
            same = sum(1 for r in ok if r["target"] in reference and r["output_key"] == reference[r["target"]]["output_key"])
            lines.append(f"| {sampling_name} | {condition} | {len(ok)} | {same}/{len(ok)} ({100 * same / max(len(ok), 1):.0f}%) | {len(cond_runs) - len(ok)} |")
        distinct = {}
        divergence = []
        for r in runs:
            if "error" in r:
                continue
            distinct.setdefault(r["target"], set()).add(r["output_key"])
            ref = reference.get(r["target"])
            if ref and r["output_key"] != ref["output_key"]:
                a = ref["output"]["content"] + json.dumps(ref["output"]["tool_calls"])
                b = r["output"]["content"] + json.dumps(r["output"]["tool_calls"])
                divergence.append(common_prefix_len(a, b))
        fully = sum(1 for keys in distinct.values() if len(keys) == 1)
        summary_rows.append((sampling_name, fully, len(distinct), max((len(k) for k in distinct.values()), default=0), divergence))
    lines.append("")
    for sampling_name, fully, total, worst, divergence in summary_rows:
        line = f"- **{sampling_name}**: {fully}/{total} targets produced exactly one output across all runs; worst target had {worst} distinct outputs"
        if divergence:
            line += f"; divergent runs split from the reference after a median of {int(statistics.median(divergence))} characters"
        lines.append(line)
    ttfts = [r["ttft"] for r in result["runs"] if r.get("ttft") and r["condition"].startswith("load")]
    if ttfts:
        lines.append(f"- target TTFT under load: p50 {statistics.median(ttfts):.2f}s, p90 {sorted(ttfts)[int(0.9 * (len(ttfts) - 1))]:.2f}s")
    for load in result["load"]:
        lines.append(f"- {load['sampling']} c={load['concurrency']}: background throughput {load['background_output_tokens_per_second']:.0f} output tok/s "
                     f"({load['background_requests']} requests, {load['background_errors']} errors)")
    if result.get("server_metrics") and "rollbacks" in result["server_metrics"]:
        m = result["server_metrics"]
        lines.append(f"- sid engine: {m.get('rollbacks', 0)} rollbacks over {m.get('verify_windows', 0)} verified windows")
    return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", action="append", required=True, help="name=base_url, e.g. sid=http://localhost:8000 (repeatable)")
    parser.add_argument("--model", default=None, help="model id to request (default: first entry of /v1/models)")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--targets", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=lambda s: [int(x) for x in s.split(",")], default=[32, 128])
    parser.add_argument("--sampling", type=lambda s: s.split(","), default=["greedy", "sampled"])
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--warmup", type=float, default=10.0, help="seconds of background load before firing targets")
    parser.add_argument("--jitter", type=float, default=3.0, help="targets start at random times within this many seconds")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="determinism_results.json")
    args = parser.parse_args()

    results = []
    for spec in args.endpoint:
        name, _, url = spec.partition("=")
        results.append(await bench_endpoint(name, url.rstrip("/"), args))
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print()
    for result in results:
        print(summarize(result, args))
        print()
    print(f"raw results written to {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
