"""Ask two servers the same question idle, then again under load, and show what changed.

    python demo/compare.py --questions 8        # sweep: which questions change under load?
    python demo/compare.py                      # agent with tool calls, one question, 10 runs

Expects a deterministic sid on :8000 and a non-deterministic one on :8001 (see demo/servers.sh).

Two things matter for this to mean anything:

  * The comparison is idle vs under load. Firing N copies of a request at once proves nothing: they ride in
    the same batches, hit identical arithmetic, and agree even without a verifier. What changes an answer is
    the company a request keeps.
  * One question is a coin flip. A load-induced wobble only changes the answer when two tokens are nearly
    tied somewhere in it, which is true of roughly one prompt in five. Sweep several questions and report
    which ones moved.
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
from bench.workload import TOOLS, filler_conversations, system_prompt, target_conversations

BOLD, GREEN, RED, DIM, OFF = "\033[1m", "\033[32m", "\033[31m", "\033[2m", "\033[0m"

AGENT_SYSTEM = (
    f"You are the customer support agent for {PRODUCT}. You only know what search_knowledge_base returns, so "
    f"search before answering and search separately for each topic. Use create_ticket when the customer asks for "
    f"follow-up from another team. Finish with a short answer naming the articles you used."
)


def build_conversations(args) -> list[dict]:
    """agent mode: one scripted question with tools. chat mode: N knowledge-base questions, no tools."""
    if args.mode == "agent":
        return [{
            "question": args.question or QUESTION,
            "messages": [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": args.question or QUESTION}],
            "tools": TOOLS,
        }]
    if args.question:
        return [{
            "question": args.question,
            "messages": [{"role": "system", "content": system_prompt(PRODUCT)}, {"role": "user", "content": args.question}],
            "tools": None,
        }]
    return [
        {"question": conv["messages"][-1]["content"], "messages": conv["messages"], "tools": None}
        for conv in target_conversations(args.questions, seed=args.question_seed)
    ]


async def run_once(client, base, model, conv, args) -> dict:
    """One agent run (or one plain chat call); returns its transcript."""
    sampling = {"temperature": args.temperature, "top_p": 0.95, "seed": args.seed} if args.temperature else {"temperature": 0.0}
    messages = list(conv["messages"])
    transcript = []
    started = time.perf_counter()
    for _ in range(args.max_steps if conv["tools"] else 1):
        r = await chat(client, base, model, {"messages": messages, "tools": conv["tools"]}, sampling, args.max_tokens, args.api_key)
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


async def run_server(name, base_url, conversations, args) -> dict:
    base = base_url.removesuffix("/v1")
    limits = httpx.Limits(max_connections=args.runs * len(conversations) + args.background + 16)
    rng = random.Random(f"{name}:{args.seed}")
    async with httpx.AsyncClient(timeout=httpx.Timeout(1800.0), limits=limits) as client:
        try:
            model = args.model or (await client.get(f"{base}/v1/models")).json()["data"][0]["id"]
        except Exception as e:
            return {"name": name, "base": base, "error": f"cannot reach {base}: {e!r}"}

        before = await get_metrics(client, base)
        references = []
        for conv in conversations:    # server idle: nothing else in flight
            references.append(await run_once(client, base, model, conv, args))
        if all("error" in r for r in references):
            return {"name": name, "base": base, "model": model, "error": f"idle runs failed: {references[0]['error']}"}

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

        async def fire(conv):
            await asyncio.sleep(rng.uniform(0, args.jitter))    # different batches, not lockstep
            return await run_once(client, base, model, conv, args)

        workers = [asyncio.create_task(background(w)) for w in range(args.background)]
        await asyncio.sleep(args.warmup)
        loaded = await asyncio.gather(*[fire(conv) for conv in conversations for _ in range(args.runs)])
        stop.set()
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        after = await get_metrics(client, base)

    runs = [loaded[i * args.runs:(i + 1) * args.runs] for i in range(len(conversations))]
    return {
        "name": name, "base": base, "model": model,
        "references": references, "runs": runs,
        "background_requests": counters["requests"], "background_errors": counters["errors"],
        "rollbacks": after.get("rollbacks", 0) - before.get("rollbacks", 0) if "rollbacks" in after else None,
    }


def describe_steps(transcript: list[dict]) -> str:
    return " -> ".join("+".join(c["name"] for c in step["tool_calls"]) if step["tool_calls"] else "answer" for step in transcript)


def tally(result: dict, index: int) -> tuple[int, int, int]:
    """(runs matching the idle answer, usable runs, distinct answers including the idle one)"""
    reference = result["references"][index]
    ok = [r for r in result["runs"][index] if "error" not in r]
    if "error" in reference or not ok:
        return 0, len(ok), 0
    return sum(r["key"] == reference["key"] for r in ok), len(ok), len({r["key"] for r in ok} | {reference["key"]})


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
        print(f"  {RED}{label}: step {i + 1} diverges at character {cut}{OFF}")
        print(f"    {DIM}idle        ...{text_a[max(0, cut - 50):cut]}{OFF}{RED}{text_a[cut:cut + 80]}{OFF}")
        print(f"    {DIM}under load  ...{text_b[max(0, cut - 50):cut]}{OFF}{RED}{text_b[cut:cut + 80]}{OFF}")
        return


def report(results: list[dict], conversations: list[dict], args):
    print()
    print(f"{BOLD}One idle answer per question, then {args.runs} more of each while {args.background} other requests flow through{OFF}")
    settings = f"temperature {args.temperature}, seed {args.seed}" if args.temperature else "temperature 0 (greedy)"
    print(f"{DIM}{args.mode} mode | {settings} | {results[0].get('model', '?')} | {len(conversations)} question(s) x {args.runs} runs{OFF}\n")

    broken = [r for r in results if r.get("error")]
    for result in broken:
        print(f"  {result['name']}: {RED}{result['error']}{OFF}")
    results = [r for r in results if not r.get("error")]
    if not results:
        return

    width = max(len(r["name"]) for r in results)
    if len(conversations) == 1:
        print(f"customer: {conversations[0]['question'][:150]}\n")
        for result in results:
            identical, total, distinct = tally(result, 0)
            colour = GREEN if distinct == 1 else RED
            verdict = "same as idle, every time" if distinct == 1 else f"{distinct} different answers"
            seconds = statistics.mean(r["seconds"] for r in result["runs"][0] if "error" not in r) if total else 0
            steps = describe_steps(result["references"][0]["transcript"])
            print(f"  {result['name']:<{width}}  {colour}{identical:>2}/{total} match the idle run   {verdict:<24}{OFF}{DIM}  {steps}   avg {seconds:.0f}s{OFF}")
    else:
        header = "  " + " " * 4 + f"{'question':<52}" + "".join(f"{r['name']:>20}" for r in results)
        print(header)
        changed = {r["name"]: 0 for r in results}
        for index, conv in enumerate(conversations):
            cells = ""
            for result in results:
                identical, total, distinct = tally(result, index)
                colour = GREEN if distinct == 1 else RED
                if distinct > 1:
                    changed[result["name"]] += 1
                cells += f"{colour}{identical:>13}/{total:<6}{OFF}"
            print(f"  {index + 1:>3} {conv['question'][:50]:<52}{cells}")
        print()
        totals = "".join(
            f"{(RED if changed[r['name']] else GREEN)}{changed[r['name']]:>13}/{len(conversations):<6}{OFF}" for r in results
        )
        print("  " + " " * 4 + f"{'questions whose answer changed under load':<52}" + totals)
    print()

    for result in results:
        for index, conv in enumerate(conversations):
            identical, total, distinct = tally(result, index)
            if distinct <= 1:
                continue
            reference = result["references"][index]
            other = next(r for r in result["runs"][index] if "error" not in r and r["key"] != reference["key"])
            label = result["name"] if len(conversations) == 1 else f"{result['name']}, question {index + 1}"
            show_divergence(label, reference["transcript"], other["transcript"])
            print()
            break

    for result in results:
        if result.get("rollbacks"):
            print(f"  {DIM}{result['name']}: the verifier caught and corrected {result['rollbacks']} drafted tokens during this run "
                  f"- load-dependent arithmetic, fixed before it reached the client{OFF}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"conversations": [c["question"] for c in conversations], "results": results}, f, indent=2)
        print(f"  {DIM}transcripts written to {args.out}{OFF}")


async def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--det", default="http://localhost:8000/v1", help="deterministic server")
    parser.add_argument("--nodet", default="http://localhost:8001/v1", help="non-deterministic server")
    parser.add_argument("--questions", type=int, default=1, help="how many knowledge-base questions to sweep (forces chat mode when > 1)")
    parser.add_argument("--runs", type=int, default=10, help="runs under load, per question, per server")
    parser.add_argument("--background", type=int, default=48, help="concurrent background requests during those runs")
    parser.add_argument("--warmup", type=float, default=6.0, help="seconds of background load before the runs start")
    parser.add_argument("--jitter", type=float, default=4.0, help="runs start at random times within this window")
    parser.add_argument("--mode", choices=["agent", "chat"], default=None, help="default: agent for one question, chat for a sweep")
    parser.add_argument("--question", default=None)
    parser.add_argument("--question-seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7, help="0 for greedy")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.mode is None:
        args.mode = "chat" if args.questions > 1 else "agent"
    if args.questions > 1 and args.mode == "agent":
        parser.error("a sweep over several questions runs in chat mode; drop --mode agent or use --questions 1")

    conversations = build_conversations(args)
    started = time.perf_counter()
    print(f"{DIM}asking each server {len(conversations)} question(s) while idle, then {args.runs} more of each under load ...{OFF}")
    results = await asyncio.gather(
        run_server("deterministic", args.det, conversations, args),
        run_server("non-deterministic", args.nodet, conversations, args),
    )
    report(results, conversations, args)
    print(f"{DIM}took {time.perf_counter() - started:.0f}s{OFF}")


if __name__ == "__main__":
    asyncio.run(main())
