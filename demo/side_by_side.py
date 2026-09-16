"""Fire the same request many times at a server under load and count how many distinct answers come back.

Run it once per server, then print the comparison:

    python demo/side_by_side.py --base-url http://localhost:8000/v1 --label "verifier on"  --out det.json
    python demo/side_by_side.py --base-url http://localhost:8000/v1 --label "verifier off" --out nodet.json
    python demo/side_by_side.py --report det.json nodet.json

The request is a support-agent question against a ~2.3k token knowledge base, sent with temperature 0.7 and a
fixed seed: settings where a caller already expects the same answer every time.
"""
import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

from bench.determinism import chat, common_prefix_len
from bench.workload import filler_conversations, target_conversations

BOLD, GREEN, RED, DIM, OFF = "\033[1m", "\033[32m", "\033[31m", "\033[2m", "\033[0m"


async def collect(args) -> dict:
    conv = target_conversations(4, seed=args.prompt_seed)[args.prompt_index]
    sampling = {"temperature": args.temperature, "top_p": 0.95, "seed": args.seed} if args.temperature else {"temperature": 0.0}
    fillers = filler_conversations(256, seed=7)
    limits = httpx.Limits(max_connections=args.background + args.repeats + 16)
    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0), limits=limits) as client:
        model = args.model or (await client.get(f"{args.base_url.removesuffix('/v1')}/v1/models")).json()["data"][0]["id"]
        base = args.base_url.removesuffix("/v1")

        print(f"{BOLD}{args.label}{OFF}  ({model} @ {base})")
        print(f"  asking once with the server idle ...", end="", flush=True)
        reference = await chat(client, base, model, conv, sampling, args.max_tokens, args.api_key)
        if "error" in reference:
            sys.exit(f"\n  request failed: {reference['error']}")
        print(f" {len(reference['output']['content'])} characters")

        counters = {"requests": 0, "tokens": 0, "errors": 0}
        stop = asyncio.Event()

        async def background(worker_id):
            i = worker_id * 31
            while not stop.is_set():
                filler = fillers[i % len(fillers)]
                i += 1
                r = await chat(client, base, model, filler, filler["sampling"], filler["max_tokens"], args.api_key)
                counters["errors" if "error" in r else "requests"] += 1
                counters["tokens"] += r.get("completion_tokens", 0)

        workers = [asyncio.create_task(background(w)) for w in range(args.background)]
        print(f"  warming up {args.background} concurrent background requests ...", end="", flush=True)
        await asyncio.sleep(args.warmup)
        print(f" {counters['requests']} done so far")
        print(f"  now sending the same request {args.repeats} times, all at once ...", end="", flush=True)
        started = time.perf_counter()
        runs = await asyncio.gather(*[chat(client, base, model, conv, sampling, args.max_tokens, args.api_key) for _ in range(args.repeats)])
        elapsed = time.perf_counter() - started
        stop.set()
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        print(f" {elapsed:.1f}s")

        try:
            metrics = (await client.get(f"{base}/metrics")).json()
        except Exception:
            metrics = None

    ok = [r for r in runs if "error" not in r]
    result = {
        "label": args.label,
        "model": model,
        "base_url": base,
        "question": conv["messages"][-1]["content"],
        "settings": sampling,
        "background": args.background,
        "background_requests": counters["requests"],
        "reference": reference["output"],
        "reference_key": reference["output_key"],
        "runs": [{"output": r["output"], "output_key": r["output_key"]} for r in ok],
        "errors": len(runs) - len(ok),
        "fingerprint": reference.get("fingerprint"),
        "sid": reference.get("sid"),
        "metrics": metrics,
    }
    summarize(result)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  written to {args.out}")
    return result


def as_text(output: dict) -> str:
    text = output["content"] or ""
    for call in output["tool_calls"]:
        text += f"\n[tool call] {call['name']}({call['arguments']})"
    return text


def stats(result: dict) -> tuple[int, int, int]:
    keys = [r["output_key"] for r in result["runs"]]
    identical = sum(k == result["reference_key"] for k in keys)
    return identical, len(keys), len(set(keys) | {result["reference_key"]})


def summarize(result: dict):
    identical, total, distinct = stats(result)
    colour = GREEN if distinct == 1 else RED
    print(f"  {colour}{identical}/{total} identical to the idle run, {distinct} distinct answer(s) in total{OFF}")
    if result["metrics"] and result["metrics"].get("rollbacks") is not None:
        print(f"  {DIM}engine: {result['metrics']['rollbacks']} rollbacks over {result['metrics']['verify_windows']} verified windows{OFF}")
    print()


def report(paths: list[str]):
    results = []
    for path in paths:
        with open(path) as f:
            results.append(json.load(f))
    first = results[0]
    print(f"\n{BOLD}Same question, same model, same GPU{OFF}")
    print(f"{DIM}{first['model']} | {first['settings']} | {first['background']} concurrent background requests{OFF}")
    print(f"\ncustomer: {first['question'][:160]}\n")
    width = max(len(r["label"]) for r in results)
    for result in results:
        identical, total, distinct = stats(result)
        colour = GREEN if distinct == 1 else RED
        verdict = "one answer, every time" if distinct == 1 else f"{distinct} different answers"
        print(f"  {result['label']:<{width}}  {colour}{identical:>3}/{total} identical   {verdict}{OFF}")
    print()

    for result in results:
        identical, total, distinct = stats(result)
        if distinct == 1:
            continue
        reference = as_text(result["reference"])
        other = next((as_text(r["output"]) for r in result["runs"] if r["output_key"] != result["reference_key"]), None)
        if other is None:
            continue
        cut = common_prefix_len(reference, other)
        print(f"  {BOLD}{result['label']}{OFF}: two runs of the identical request diverge at character {cut}")
        print(f"    {DIM}...{reference[max(0, cut - 60):cut]}{OFF}{RED}{reference[cut:cut + 90]}{OFF}")
        print(f"    {DIM}...{other[max(0, cut - 60):cut]}{OFF}{RED}{other[cut:cut + 90]}{OFF}")
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", nargs="+", help="print a comparison of result files instead of collecting")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--label", default="sid")
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--background", type=int, default=64)
    parser.add_argument("--warmup", type=float, default=8.0)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.7, help="0 for greedy")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--prompt-index", type=int, default=1)
    parser.add_argument("--prompt-seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.report:
        report(args.report)
    else:
        asyncio.run(collect(args))


if __name__ == "__main__":
    main()
