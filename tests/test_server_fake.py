"""End-to-end test of server.py on CPU, with a scripted fake model behind the real scheduler.

Covers the HTTP surface (chat + completions, streaming, tool calls, stop strings, errors) and checks that
concurrent requests with noisy drafts return byte-identical results to the same request run alone.

    python tests/test_server_fake.py        # needs fastapi, httpx, transformers, torch (cpu is fine)
"""
import asyncio
import json
import os
import random
import sys
import tempfile
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import torch
import xxhash
from transformers import AutoTokenizer

# the server asks cuda for a device and its name; there is none here
torch.cuda.set_device = lambda *args, **kwargs: None
torch.cuda.get_device_name = lambda *args, **kwargs: "fake-cpu"

import server
from llm_engine import LLMEngine
from scheduler import Scheduler
from sequence import Sequence
from tests.test_scheduler_sim import FakeRunner, make_config

TOKENIZER = os.environ.get("SID_TEST_TOKENIZER", "Qwen/Qwen3-0.6B")
WORDS = "the quick brown fox jumps over a lazy dog while support agents answer tickets about billing plans and integrations".split()
TOOL_CALL = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'


class Script:
    """The fake model's output for a prompt is a fixed token script chosen from the prompt text."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.word_ids = [tokenizer.encode(" " + w, add_special_tokens=False)[0] for w in WORDS]
        self.cache = {}

    def tokens(self, seq) -> list[int]:
        key = tuple(seq.prompt_token_ids)
        if key not in self.cache:
            text = self.tokenizer.decode(key)
            last_user = text.rsplit("<|im_start|>user", 1)[-1]
            if "TOOLCALL" in last_user:
                script = self.tokenizer.encode(TOOL_CALL, add_special_tokens=False)
            elif "STOPTEST" in last_user:
                script = self.tokenizer.encode("alpha beta gamma STOP delta epsilon", add_special_tokens=False)
            else:
                h = xxhash.xxh64(repr(key).encode()).intdigest()
                rng = random.Random(h)
                script = [rng.choice(self.word_ids) for _ in range(20 + h % 60)]
            self.cache[key] = script + [self.eos]
        return self.cache[key]

    def __call__(self, seq, n):
        script = self.tokens(seq)
        i = n - seq.num_prompt_tokens    # negative for non-final prefill chunks, whose token is discarded
        return script[i] if 0 <= i < len(script) else self.eos


def make_engine(tokenizer, noise=0.3, **overrides) -> LLMEngine:
    model_dir = tempfile.mkdtemp()
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        f.write("{}")
    eos_ids = frozenset({tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id})
    cfg = make_config(
        eos_token_ids=eos_ids, kvcache_block_size=16, max_model_len=4096, num_kvcache_blocks=400,
        verify_window=4, verify_batch_size=3, max_num_batched_tokens=64, max_num_seqs=32, **overrides,
    )
    cfg.model = model_dir
    cfg.tensor_parallel_size = 1
    cfg.hf_config = type("HFConfig", (), {"dtype": "bfloat16"})()
    Sequence.block_size = cfg.kvcache_block_size

    engine = LLMEngine.__new__(LLMEngine)    # skip model loading
    engine.config = cfg
    engine.tokenizer = tokenizer
    engine.scheduler = Scheduler(cfg)
    engine.stats = Counter()
    engine.model_runner = FakeRunner(cfg, random.Random(), noise, canonical=Script(tokenizer), vocab_size=len(tokenizer))
    return engine


def chat_body(content, **extra):
    return {"model": "fake", "messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": content}], **extra}


def parse_sse(text: str) -> list:
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            data = line[len("data: "):]
            events.append(data if data == "[DONE]" else json.loads(data))
    return events


async def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    engine = make_engine(tokenizer)
    app = server.build_app(server.AsyncEngine(engine), "fake", default_max_tokens=512, api_key=None)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=60) as client:
        assert (await client.get("/health")).json()["status"] == "ok"
        assert (await client.get("/v1/models")).json()["data"][0]["id"] == "fake"
        info = (await client.get("/v1/sid/info")).json()
        assert info["system_fingerprint"].startswith("sid-")
        print("ok  health, models, info")

        # --- basic chat
        r = (await client.post("/v1/chat/completions", json=chat_body("How do I reset my password?"))).json()
        choice = r["choices"][0]
        assert choice["finish_reason"] == "stop", r
        assert choice["message"]["content"] and "<|im_end|>" not in choice["message"]["content"]
        assert r["system_fingerprint"] == info["system_fingerprint"]
        assert len(r["sid"]["output_sha256"]) == 64
        print("ok  chat completion")

        # --- determinism under concurrency with noisy drafts
        questions = [f"Question {i}: how does feature {i * 7} work on the Growth plan?" for i in range(12)]
        alone = {}
        for q in questions:
            r = (await client.post("/v1/chat/completions", json=chat_body(q))).json()
            alone[q] = (r["choices"][0]["message"]["content"], r["sid"]["output_sha256"])
        rollbacks_before = (await client.get("/metrics")).json().get("rollbacks", 0)
        burst = [q for q in questions for _ in range(4)]
        random.Random(0).shuffle(burst)
        results = await asyncio.gather(*[client.post("/v1/chat/completions", json=chat_body(q)) for q in burst])
        for q, r in zip(burst, results):
            r = r.json()
            assert (r["choices"][0]["message"]["content"], r["sid"]["output_sha256"]) == alone[q], f"diverged under load: {q}"
        rollbacks = (await client.get("/metrics")).json().get("rollbacks", 0) - rollbacks_before
        assert rollbacks > 0, "noise should have caused rollbacks"
        print(f"ok  {len(burst)} concurrent requests identical to isolated runs ({rollbacks} rollbacks along the way)")

        # --- streaming matches non-streaming
        q = questions[3]
        r = await client.post("/v1/chat/completions", json=chat_body(q, stream=True, stream_options={"include_usage": True}))
        events = parse_sse(r.text)
        assert events[-1] == "[DONE]"
        chunks = events[:-1]
        text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
        assert text == alone[q][0], f"stream {text!r} != {alone[q][0]!r}"
        final = [c for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]][-1]
        assert final["choices"][0]["finish_reason"] == "stop" and final["sid"]["output_sha256"] == alone[q][1]
        assert chunks[-1]["usage"]["completion_tokens"] > 0
        print(f"ok  streaming matches non-streaming ({len(chunks)} chunks)")

        # --- tool calls
        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
        r = (await client.post("/v1/chat/completions", json=chat_body("TOOLCALL what's the weather in Paris?", tools=tools))).json()
        choice = r["choices"][0]
        assert choice["finish_reason"] == "tool_calls", choice
        call = choice["message"]["tool_calls"][0]
        assert call["function"]["name"] == "get_weather" and json.loads(call["function"]["arguments"]) == {"city": "Paris"}
        assert choice["message"]["content"] is None
        r2 = (await client.post("/v1/chat/completions", json=chat_body("TOOLCALL what's the weather in Paris?", tools=tools))).json()
        assert r2["choices"][0]["message"]["tool_calls"][0]["id"] == call["id"], "tool call ids must be reproducible"
        events = parse_sse((await client.post("/v1/chat/completions", json=chat_body("TOOLCALL what's the weather in Paris?", tools=tools, stream=True))).text)
        deltas = [c["choices"][0]["delta"] for c in events[:-1] if c["choices"]]
        assert not any("<tool_call" in d.get("content", "") for d in deltas), "raw tool call markup leaked into content"
        streamed_calls = [tc for d in deltas for tc in d.get("tool_calls", [])]
        assert streamed_calls and streamed_calls[0]["function"]["name"] == "get_weather"
        # a follow-up turn with the tool result must render through the chat template
        followup = chat_body("TOOLCALL what's the weather in Paris?", tools=tools)
        followup["messages"] += [choice["message"], {"role": "tool", "tool_call_id": call["id"], "content": '{"temp_c": 21}'}]
        r3 = await client.post("/v1/chat/completions", json=followup)
        assert r3.status_code == 200, r3.text
        print("ok  tool calls (non-streaming, streaming, reproducible ids, tool-result turn)")

        # --- stop strings
        r = (await client.post("/v1/chat/completions", json=chat_body("STOPTEST", stop=["STOP"]))).json()
        content = r["choices"][0]["message"]["content"]
        assert r["choices"][0]["finish_reason"] == "stop" and "STOP" not in content and "delta" not in content and "gamma" in content, content
        events = parse_sse((await client.post("/v1/chat/completions", json=chat_body("STOPTEST", stop=["STOP"], stream=True))).text)
        streamed = "".join(c["choices"][0]["delta"].get("content", "") for c in events[:-1] if c["choices"])
        assert streamed == content, f"{streamed!r} != {content!r}"
        print("ok  stop strings (non-streaming and streaming)")

        # --- completions endpoint and max_tokens
        r = (await client.post("/v1/completions", json={"model": "fake", "prompt": "Once upon a time", "max_tokens": 5})).json()
        assert r["choices"][0]["finish_reason"] == "length" and r["usage"]["completion_tokens"] == 5, r
        print("ok  completions endpoint, max_tokens")

        # --- errors
        r = await client.post("/v1/chat/completions", json=chat_body("hi", n=2))
        assert r.status_code == 400
        r = await client.post("/v1/chat/completions", json=chat_body("word " * 5000))
        assert r.status_code == 400 and "max_model_len" in r.json()["error"]["message"]
        r = await client.post("/v1/chat/completions", json=chat_body("hi", temperature=-1))
        assert r.status_code == 400
        print("ok  invalid requests rejected with 400")

        # --- no blocks leaked once everything finished
        await asyncio.sleep(0.2)
        m = (await client.get("/metrics")).json()
        assert m["running"] == 0 and m["waiting"] == 0 and m["free_kvcache_blocks"] == m["total_kvcache_blocks"], m
        print("ok  kv cache fully released")


if __name__ == "__main__":
    asyncio.run(main())
