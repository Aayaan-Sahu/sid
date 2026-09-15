"""Replay a multi-step support agent (tool calls included) several times and check the transcripts match.

This is the Brainfish-style demo: an agent searches a knowledge base, may open a ticket, then answers.
Point it at any OpenAI-compatible endpoint; optionally hammer the same endpoint with background traffic
while it runs.

    python bench/agent_replay.py --base-url http://localhost:8000/v1 --runs 5 --background 64
"""
import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from openai import OpenAI

from bench.determinism import chat
from bench.workload import TOOLS, filler_conversations, knowledge_base

PRODUCT = "Ledgerly"
_ARTICLES = knowledge_base(PRODUCT)
_ACTION = next(a["meta"]["action"] for a in _ARTICLES if a["kind"] == "howto")
_INTEGRATION = next(a["meta"]["integration"] for a in _ARTICLES if a["kind"] == "integration")
QUESTION = (
    f"We're on the Growth plan. How do I {_ACTION}, and does {PRODUCT} sync with {_INTEGRATION} (how often)? "
    f"If {_INTEGRATION} isn't available on the Growth plan, open a ticket asking sales to contact us about upgrading."
)
SYSTEM = (
    f"You are the customer support agent for {PRODUCT}. You do not know anything about {PRODUCT} except what "
    f"search_knowledge_base returns, so search before answering and search separately for each topic. "
    f"Open a ticket with create_ticket when the customer asks for follow-up from another team. "
    f"When you have what you need, reply with a short final answer that names the articles you used."
)


def search_knowledge_base(query: str) -> dict:
    terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    scored = []
    for article in knowledge_base(PRODUCT):
        words = re.findall(r"[a-z0-9]+", (article["title"] + " " + article["body"]).lower())
        score = sum(words.count(t) for t in terms) + 5 * sum(t in article["title"].lower() for t in terms)
        scored.append((score, article["title"], article))
    scored.sort(key=lambda s: (-s[0], s[1]))
    return {"results": [{"title": a["title"], "body": a["body"][:900]} for score, _, a in scored[:2] if score > 0]}


def create_ticket(summary: str, priority: str = "normal") -> dict:
    ticket = hashlib.sha256(f"{summary}|{priority}".encode()).hexdigest()[:6].upper()
    return {"ticket_id": f"T-{ticket}", "status": "open", "priority": priority}


def escalate_to_human(reason: str) -> dict:
    return {"status": "handed_off", "queue": "tier-2"}


TOOL_IMPLS = {"search_knowledge_base": search_knowledge_base, "create_ticket": create_ticket, "escalate_to_human": escalate_to_human}


def run_agent(client: OpenAI, model: str, args) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": QUESTION}]
    transcript = []
    for _ in range(args.max_steps):
        response = client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS, temperature=args.temperature, seed=args.seed, max_tokens=args.max_tokens,
        )
        message = response.choices[0].message
        calls = message.tool_calls or []
        transcript.append({
            "content": message.content,
            "tool_calls": [{"name": c.function.name, "arguments": c.function.arguments} for c in calls],
        })
        messages.append(message.model_dump(exclude_none=True))
        if not calls:
            break
        for call in calls:
            try:
                result = TOOL_IMPLS[call.function.name](**json.loads(call.function.arguments))
            except Exception as e:
                result = {"error": repr(e)}
            transcript.append({"tool": call.function.name, "result": result})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
    return transcript


def transcript_hash(transcript: list[dict]) -> str:
    return hashlib.sha256(json.dumps(transcript, sort_keys=True).encode()).hexdigest()[:12]


def start_background(base_url: str, model: str, concurrency: int, api_key: str | None) -> tuple[threading.Event, dict]:
    stop = threading.Event()
    counters = {"requests": 0, "tokens": 0, "errors": 0}
    root = base_url.removesuffix("/v1")

    async def load():
        fillers = filler_conversations(256, seed=99)
        async with httpx.AsyncClient(timeout=900, limits=httpx.Limits(max_connections=concurrency + 8)) as client:
            async def worker(w):
                i = w * 31
                while not stop.is_set():
                    conv = fillers[i % len(fillers)]
                    i += 1
                    r = await chat(client, root, model, conv, conv["sampling"], conv["max_tokens"], api_key)
                    counters["errors" if "error" in r else "requests"] += 1
                    counters["tokens"] += r.get("completion_tokens", 0)
            tasks = [asyncio.create_task(worker(w)) for w in range(concurrency)]
            while not stop.is_set():
                await asyncio.sleep(0.2)
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    threading.Thread(target=lambda: asyncio.run(load()), daemon=True).start()
    return stop, counters


def describe(transcript: list[dict]) -> str:
    parts = []
    for step in transcript:
        if "tool" in step:
            continue
        if step["tool_calls"]:
            parts.append("+".join(c["name"] for c in step["tool_calls"]))
        else:
            parts.append("answer")
    return " -> ".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--background", type=int, default=0, help="concurrent background requests while replaying")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--out", default="agent_transcripts.json")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key or "sid", timeout=900)
    model = args.model or client.models.list().data[0].id
    print(f"model {model} @ {args.base_url}; {args.runs} runs, background load {args.background}")
    print(f"customer: {QUESTION}\n")

    stop, counters = (None, None)
    if args.background:
        stop, counters = start_background(args.base_url, model, args.background, args.api_key)
        time.sleep(5)

    transcripts = []
    try:
        for i in range(args.runs):
            started = time.perf_counter()
            transcript = run_agent(client, model, args)
            transcripts.append(transcript)
            h = transcript_hash(transcript)
            verdict = "reference" if i == 0 else ("identical" if h == transcript_hash(transcripts[0]) else "DIFFERENT")
            load = f", background {counters['requests']} requests done" if counters else ""
            print(f"run {i + 1}: {h}  {verdict:9s}  {describe(transcript)}  ({time.perf_counter() - started:.1f}s{load})")
    finally:
        if stop:
            stop.set()

    hashes = {transcript_hash(t) for t in transcripts}
    print()
    if len(hashes) == 1:
        print(f"all {len(transcripts)} runs produced the identical transcript")
        final = transcripts[0][-1]
        if final.get("content"):
            print(f"\nfinal answer:\n{final['content']}")
    else:
        print(f"{len(hashes)} distinct transcripts across {len(transcripts)} runs")
        reference = transcripts[0]
        for i, transcript in enumerate(transcripts[1:], start=2):
            for step, (a, b) in enumerate(zip(reference, transcript)):
                if a != b:
                    print(f"\nrun {i} first differs from run 1 at step {step}:")
                    print(f"  run 1: {json.dumps(a)[:400]}")
                    print(f"  run {i}: {json.dumps(b)[:400]}")
                    break
    with open(args.out, "w") as f:
        json.dump(transcripts, f, indent=2)


if __name__ == "__main__":
    main()
