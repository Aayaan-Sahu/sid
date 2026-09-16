"""Ask two servers the same question, once on an idle server and then N times under load, and show what changed.

    python demo/compare.py                        # agent with tool calls, 10 runs per server
    python demo/compare.py --runs 20 --mode chat  # one call per run, no tools

Expects a deterministic sid on :8000 and a non-deterministic one on :8001 (see demo/servers.sh).

Why the load matters: sending N copies of a request all at once is not a test of determinism. They ride in
the same batches, go through identical arithmetic, and agree even on a non-deterministic server. What changes
an answer is the *company a request keeps* - so each server answers once while idle, and then answers the same
question again while other traffic flows through it.
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

from bench.agent_replay import PRODUCT, QUESTION, TOOL_IMPLS
from bench.determinism import chat, common_prefix_len
from bench.workload import TOOLS, filler_conversations

BOLD, GREEN, RED, DIM, OFF = "\033[1m", "\033[32m", "\033[31m", "\033[2m", "\033[0m"

SYSTEM = (
    f"You are the customer support agent for {PRODUCT}. You only know what search_knowledge_base returns, so "
    f"search before answering and search separately for each topic. Use create_ticket when the customer asks for "
    f"follow-up from another team. Finish with a short answer naming the articles you used."
)


async def run_once(client, base, model, args) -> dict:
    """One agent run (or one plain chat call); returns its transcript."""
    sampling = {"temperature": args.temperature, "top_p": 0.95, "seed": args.seed} if args.temperature else {"temperature": 0.0}
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": args.question}]
    tools = TOOLS if args.mode == "agent" else None
    transcript = []
    started = time.perf_counter()
    for _ in range(args.max_steps if args.mode == "agent" else 1):
        r = await chat(client, base, model, {"messages": messages, "tools": tools}, sampling, args.max_tokens, args.api_key)
        if "error" in r:
            return {"error": r["error"]}
        content, calls = r["output"]["content"], r["output"]["tool_calls"]
        transcript.append({"content": content, "tool_calls": calls})
        if not calls:
            break
        messages.append({"role": "assistant", "content": content, "tool_calls": [
            {"id": f"call_{i}", "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
            for i, c in enumerate(calls)
        ]})
        for i, call in enumerate(calls):
            try:
                result = TOOL_IMPLS[call["name"]](**json.loads(call["arguments"]))
            except Exception as e:
                result = {"error": repr(e)}
            messages.append({"role": "tool", "tool_call_id": f"call_{i}", "content": json.dumps(result)})
    return {
        "transcript": transcript,
        "key": hashlib.sha256(json.dumps(transcript, sort_keys=True).encode()).hexdigest(),
        "seconds": time.perf_counter() - started,
    }


async def get_metrics(client, base) -> dict:
    try:
        return (await client.get(f"{base}/metrics")).json()
    except Exception:
        return {}


async def run_server(name, base_url, args) -> dict:
    base = base_url.removesuffix("/v1")
    limits = httpx.Limits(max_connections=args.runs + args.background + 16)
    rng = random.Random(f"{name}:{args.seed}")
    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0), limits=limits) as client:
        try:
            model = args.model or (await client.get(f"{base}/v1/models")).json()["data"][0]["id"]
        except Exception as e:
            return {"name": name, "base": base, "error": f"cannot reach {base}: {e!r}"}

        before = await get_metrics(client, base)
        reference = await run_once(client, base, model, args)    # server idle: nothing else in flight
        if "error" in reference:
            return {"name": name, "base": base, "model": model, "error": f"idle run failed: {reference['error']}"}

        # now the same question again, but sharing the server with other traffic
        fillers = filler_conversations(256, seed=11)
        counters = {"requests": 0, "errors": 0}
        stop = asyncio.Event()

        async def background(worker_id):
            i = worker_id * 31
            while not stop.is_set():
                filler = fillers[i % len(fillers)]
                i += 1
                r = await chat(client, base, model, filler, filler["sampling"], filler["max_tokens"], args.api_key)
                counters["errors" if "error" in r else "requests"] += 1

        async def fire():
            await asyncio.sleep(rng.uniform(0, args.jitter))    # different batches, not lockstep
            return await run_once(client, base, model, args)

        workers = [asyncio.create_task(background(w)) for w in range(args.background)]
        await asyncio.sleep(args.warmup)
        runs = await asyncio.gather(*[fire() for _ in range(args.runs)])
        stop.set()
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        after = await get_metrics(client, base)

    return {
        "name": name, "base": base, "model": model,
        "reference": reference, "runs": runs,
        "background_requests": counters["requests"], "background_errors": counters["errors"],
        "rollbacks": after.get("rollbacks", 0) - before.get("rollbacks", 0) if "rollbacks" in after else None,
    }


def describe_steps(transcript: list[dict]) -> str:
    return " -> ".join("+".join(c["name"] for c in step["tool_calls"]) if step["tool_calls"] else "answer" for step in transcript)


def show_divergence(label: str, reference: list[dict], other: list[dict]):
    for i in range(max(len(reference), len(other))):
        a = reference[i] if i < len(reference) else None
        b = other[i] if i < len(other) else None
        if a == b:
            continue
        if a is None or b is None:
            print(f"  {RED}{label}: {len(reference)} steps when idle, {len(other)} under load{OFF}")
            return
        if a["tool_calls"] != b["tool_calls"]:
            print(f"  {RED}{label}: step {i + 1} called the tool differently{OFF}")
            for tag, step in (("idle      ", a), ("under load", b)):
                calls = ", ".join(f"{c['name']}({c['arguments']})" for c in step["tool_calls"]) or "(no tool call)"
                print(f"    {DIM}{tag}{OFF}  {calls[:150]}")
            return
        text_a, text_b = a["content"] or "", b["content"] or ""
        cut = common_prefix_len(text_a, text_b)
        print(f"  {RED}{label}: step {i + 1} text diverges at character {cut}{OFF}")
        print(f"    {DIM}idle        ...{text_a[max(0, cut - 50):cut]}{OFF}{RED}{text_a[cut:cut + 80]}{OFF}")
        print(f"    {DIM}under load  ...{text_b[max(0, cut - 50):cut]}{OFF}{RED}{text_b[cut:cut + 80]}{OFF}")
        return


def report(results: list[dict], args) -> int:
    print()
    print(f"{BOLD}One idle run, then {args.runs} more of the same question while {args.background} other requests flow through{OFF}")
    settings = f"temperature {args.temperature}, seed {args.seed}" if args.temperature else "temperature 0 (greedy)"
    print(f"{DIM}{args.mode} mode | {settings} | {results[0].get('model', '?')}{OFF}")
    print(f"\ncustomer: {args.question[:150]}\n")

    width = max(len(r["name"]) for r in results)
    diverged = []
    for result in results:
        if result.get("error"):
            print(f"  {result['name']:<{width}}  {RED}{result['error']}{OFF}")
            continue
        ok = [r for r in result["runs"] if "error" not in r]
        failed = len(result["runs"]) - len(ok)
        if not ok:
            print(f"  {result['name']:<{width}}  {RED}every run failed: {result['runs'][0].get('error')}{OFF}")
            continue
        reference = result["reference"]
        identical = sum(r["key"] == reference["key"] for r in ok)
        distinct = len({r["key"] for r in ok} | {reference["key"]})
        colour = GREEN if distinct == 1 else RED
        verdict = "same as idle, every time" if distinct == 1 else f"{distinct} different answers"
        seconds = statistics.mean(r["seconds"] for r in ok)
        note = f"   {failed} failed" if failed else ""
        print(f"  {result['name']:<{width}}  {colour}{identical:>2}/{len(ok)} match the idle run   {verdict:<24}{OFF}"
              f"{DIM}  {describe_steps(reference['transcript'])}   avg {seconds:.0f}s{note}{OFF}")
        if distinct > 1:
            diverged.append((result, reference["transcript"], next(r["transcript"] for r in ok if r["key"] != reference["key"])))
    print()
    for result, reference, other in diverged:
        show_divergence(result["name"], reference, other)
        print()
    for result in results:
        if result.get("rollbacks"):
            print(f"  {DIM}{result['name']}: the verifier caught and corrected {result['rollbacks']} drafted tokens during this run "
                  f"- that is the nondeterminism, fixed before it reached the client{OFF}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  {DIM}transcripts written to {args.out}{OFF}")
    return len(diverged)


async def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--det", default="http://localhost:8000/v1", help="deterministic server")
    parser.add_argument("--nodet", default="http://localhost:8001/v1", help="non-deterministic server")
    parser.add_argument("--runs", type=int, default=10, help="runs under load, per server")
    parser.add_argument("--background", type=int, default=48, help="concurrent background requests during those runs")
    parser.add_argument("--warmup", type=float, default=6.0, help="seconds of background load before the runs start")
    parser.add_argument("--jitter", type=float, default=4.0, help="runs start at random times within this window")
    parser.add_argument("--mode", choices=["agent", "chat"], default="agent")
    parser.add_argument("--question", default=QUESTION)
    parser.add_argument("--temperature", type=float, default=0.7, help="0 for greedy")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    started = time.perf_counter()
    print(f"{DIM}asking each server once while idle, then {args.runs} more under load ...{OFF}")
    results = await asyncio.gather(
        run_server("deterministic", args.det, args),
        run_server("non-deterministic", args.nodet, args),
    )
    report(results, args)
    print(f"{DIM}took {time.perf_counter() - started:.0f}s{OFF}")


if __name__ == "__main__":
    asyncio.run(main())
