"""Rehearse demo/compare.py on CPU: two fake servers (one deterministic, one not) and the real comparison.

    python tests/test_demo_compare.py

Uses the scripted fake model from test_server_fake.py, so it exercises the actual HTTP path, the real
scheduler and the real comparison output without a GPU.
"""
import asyncio
import os
import socket
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

torch.cuda.set_device = lambda *args, **kwargs: None
torch.cuda.get_device_name = lambda *args, **kwargs: "fake-cpu"

import uvicorn
from transformers import AutoTokenizer

import server
from demo import compare
from tests.test_server_fake import make_engine

TOKENIZER = os.environ.get("SID_TEST_TOKENIZER", "Qwen/Qwen3-0.6B")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(tokenizer, deterministic: bool, noise: float) -> int:
    port = free_port()
    engine = make_engine(tokenizer, noise=noise, enable_determinism=deterministic)
    app = server.build_app(server.AsyncEngine(engine), "fake", default_max_tokens=256, api_key=None)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    uvicorn_server = uvicorn.Server(config)
    threading.Thread(target=uvicorn_server.run, daemon=True).start()
    return port


class Args:
    runs = 6
    questions = 1
    question_seed = 0
    background = 4
    warmup = 1.0
    jitter = 0.5
    mode = "chat"
    question = "How do I add a teammate, and what does it cost on the Growth plan?"
    temperature = 0.0
    seed = 1234
    max_tokens = 200
    max_steps = 1
    model = None
    api_key = None
    out = None
    det = nodet = None


async def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    det_port = serve(tokenizer, deterministic=True, noise=0.35)
    nodet_port = serve(tokenizer, deterministic=False, noise=0.35)
    args = Args()
    args.det = f"http://127.0.0.1:{det_port}/v1"
    args.nodet = f"http://127.0.0.1:{nodet_port}/v1"
    await asyncio.sleep(2)    # let uvicorn bind

    conversations = compare.build_conversations(args)
    results = await asyncio.gather(
        compare.run_server("deterministic", args.det, conversations, args),
        compare.run_server("non-deterministic", args.nodet, conversations, args),
    )
    compare.report(results, conversations, args)

    det, nodet = results
    for result in results:
        assert not result.get("error"), result["error"]
        assert all("error" not in r for r in result["references"]), result["references"]
        assert all("error" not in r for runs in result["runs"] for r in runs), result["runs"]
    det_identical, det_total, det_distinct = compare.tally(det, 0)
    nodet_identical, nodet_total, nodet_distinct = compare.tally(nodet, 0)
    assert det_distinct == 1, f"deterministic server returned {det_distinct} distinct answers"
    assert det_identical == det_total, f"only {det_identical}/{det_total} matched the idle run"
    assert nodet_distinct > 1, (
        "non-deterministic server returned one answer; the fake model's noise did not change the output, "
        "so this rehearsal cannot show divergence"
    )
    print(f"ok  deterministic: {det_identical}/{det_total} match the idle run; "
          f"non-deterministic: {nodet_identical}/{nodet_total}, {nodet_distinct} distinct answers")


if __name__ == "__main__":
    asyncio.run(main())
