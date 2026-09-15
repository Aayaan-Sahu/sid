"""OpenAI-compatible HTTP server for sid.

    python server.py --model Qwen/Qwen3-8B --port 8000

Endpoints: /v1/chat/completions (streaming, tools), /v1/completions, /v1/models, /health, /metrics, /v1/sid/info.

Differences from OpenAI worth knowing:
  - temperature defaults to 0 and seed defaults to 0, so requests are reproducible unless you opt out.
  - streamed text only ever contains verified tokens, so it arrives in bursts of up to verify_window tokens
    and is never retracted.
  - every response carries `system_fingerprint` (model + hardware + software + engine config) and a `sid`
    object with the sha256 of the completion token ids: same fingerprint + same request => same hash.
"""
import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import queue
import re
import threading
import time
import traceback
import uuid
from dataclasses import dataclass

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from llm_engine import LLMEngine
from sequence import SamplingParams, Sequence


# ---------------------------------------------------------------------------------------------------------
# engine thread
# ---------------------------------------------------------------------------------------------------------

@dataclass
class Update:
    token_ids: list[int]         # newly verified completion tokens
    finished: bool
    finish_reason: str | None
    num_rollbacks: int


class RequestHandle:
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.queue: asyncio.Queue[Update | Exception] = asyncio.Queue()
        self.seq_id = -1
        self.num_prompt_tokens = 0
        self.num_emitted = 0

    def push(self, item: Update | Exception):
        self.loop.call_soon_threadsafe(self.queue.put_nowait, item)


def _resolve(future: asyncio.Future, result=None, error: Exception | None = None):
    if future.done():
        return
    if error is not None:
        future.set_exception(error)
    else:
        future.set_result(result)


class AsyncEngine:
    """Owns the LLMEngine on a dedicated thread; the event loop talks to it through a command queue."""

    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self.commands: queue.SimpleQueue = queue.SimpleQueue()
        self.active: dict[int, tuple[Sequence, RequestHandle]] = {}
        self.error: str | None = None
        self.thread = threading.Thread(target=self._loop, name="sid-engine", daemon=True)
        self.thread.start()

    async def submit(self, prompt_ids: list[int], params: SamplingParams) -> RequestHandle:
        if self.error:
            raise RuntimeError(f"engine is not healthy: {self.error}")
        loop = asyncio.get_running_loop()
        handle = RequestHandle(loop)
        future = loop.create_future()
        self.commands.put(("add", prompt_ids, params, handle, future))
        await future
        return handle

    def abort(self, handle: RequestHandle):
        self.commands.put(("abort", handle))

    def _loop(self):
        torch.cuda.set_device(0)
        while True:
            block = self.engine.is_finished()    # sleep on the queue only when there is nothing to run
            try:
                while True:
                    self._handle(self.commands.get(block=block))
                    block = False
            except queue.Empty:
                pass
            if self.engine.is_finished():
                continue
            try:
                with torch.inference_mode():
                    seqs, mode = self.engine.step()
            except Exception as e:
                traceback.print_exc()
                self.error = repr(e)
                for _, handle in self.active.values():
                    handle.push(RuntimeError(f"engine failed: {e!r}"))
                self.active.clear()
                return
            if mode == "idle":
                time.sleep(0.001)
            for seq in seqs:
                self._publish(seq)

    def _handle(self, command):
        if command[0] == "add":
            _, prompt_ids, params, handle, future = command
            try:
                seq = self.engine.add_request(prompt_ids, params)
            except (ValueError, AssertionError) as e:
                handle.loop.call_soon_threadsafe(_resolve, future, None, ValueError(str(e)))
                return
            handle.seq_id = seq.seq_id
            handle.num_prompt_tokens = seq.num_prompt_tokens
            self.active[seq.seq_id] = (seq, handle)
            handle.loop.call_soon_threadsafe(_resolve, future, None)
        elif command[0] == "abort":
            entry = self.active.pop(command[1].seq_id, None)
            if entry:
                self.engine.abort(entry[0])

    def _publish(self, seq: Sequence):
        entry = self.active.get(seq.seq_id)
        if entry is None:
            return
        _, handle = entry
        # only verified tokens leave the engine; they are final
        upto = seq.num_tokens if seq.is_finished else min(seq.num_verified_tokens, seq.num_tokens)
        start = seq.num_prompt_tokens + handle.num_emitted
        new_tokens = seq.token_ids[start:upto] if upto > start else []
        handle.num_emitted += len(new_tokens)
        finish_reason = None
        if seq.is_finished:
            del self.active[seq.seq_id]
            eos = not seq.ignore_eos and seq.last_token in self.engine.config.eos_token_ids
            finish_reason = "stop" if eos else "length"
        if new_tokens or seq.is_finished:
            handle.push(Update(new_tokens, seq.is_finished, finish_reason, seq.num_rollbacks))


# ---------------------------------------------------------------------------------------------------------
# request helpers
# ---------------------------------------------------------------------------------------------------------

class RequestError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


class TextState:
    """Incremental detokenization with stop strings and hold-back of partial markers."""

    def __init__(self, tokenizer, eos_token_ids: frozenset[int], stop: list[str], markers: list[str]):
        self.tokenizer = tokenizer
        self.eos_token_ids = eos_token_ids
        self.stop = stop
        self.markers = stop + markers
        self.all_token_ids: list[int] = []
        self.token_ids: list[int] = []
        self.text = ""
        self.num_sent_chars = 0
        self.stopped = False

    def add(self, token_ids: list[int]):
        self.all_token_ids.extend(token_ids)
        self.token_ids.extend(t for t in token_ids if t not in self.eos_token_ids)
        self.text = self.tokenizer.decode(self.token_ids, skip_special_tokens=False)

    def check_stop(self) -> bool:
        hits = [i for i in (self.text.find(s) for s in self.stop) if i != -1]
        if hits:
            self.text = self.text[: min(hits)]
            self.stopped = True
        return self.stopped

    def sendable(self, final: bool) -> str:
        text = self.text
        if final:
            return text
        limit = len(text.rstrip("�"))    # incomplete utf-8 sequence
        for marker in self.markers:
            for k in range(min(len(marker) - 1, limit), 0, -1):
                if text[:limit].endswith(marker[:k]):
                    limit -= k
                    break
        return text[:limit]

    def digest(self) -> str:
        data = b"".join(t.to_bytes(4, "little") for t in self.all_token_ids)
        return hashlib.sha256(data).hexdigest()


def parse_tool_calls(text: str, digest: str) -> tuple[str | None, list[dict]]:
    calls = []
    for match in TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
            continue
        arguments = obj.get("arguments", {})
        calls.append({
            "id": f"call_{digest[:20]}_{len(calls)}",    # derived from the output so replays match exactly
            "type": "function",
            "function": {"name": obj["name"], "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments)},
        })
    if not calls:
        return text, []
    content = text[: text.find(TOOL_CALL_OPEN)].strip()
    return content or None, calls


def normalize_messages(messages) -> list[dict]:
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a non-empty list")
    out = []
    for message in messages:
        message = dict(message)
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        elif content is None:
            message["content"] = ""
        for call in message.get("tool_calls") or []:
            function = call.get("function", {})
            if isinstance(function.get("arguments"), str):
                try:
                    function["arguments"] = json.loads(function["arguments"])
                except json.JSONDecodeError:
                    pass
        out.append(message)
    return out


def parse_stop(body: dict) -> list[str]:
    stop = body.get("stop")
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    if isinstance(stop, list) and all(isinstance(s, str) for s in stop):
        return [s for s in stop if s]
    raise RequestError("stop must be a string or a list of strings")


def sampling_params_from(body: dict, default_max_tokens: int) -> SamplingParams:
    if body.get("n", 1) != 1:
        raise RequestError("only n=1 is supported")
    temperature = body.get("temperature")
    top_p = body.get("top_p")
    seed = body.get("seed")
    max_tokens = body.get("max_completion_tokens") or body.get("max_tokens") or default_max_tokens
    try:
        return SamplingParams(
            temperature=0.0 if temperature is None else float(temperature),
            top_p=1.0 if top_p is None else float(top_p),
            seed=0 if seed is None else int(seed),
            max_tokens=int(max_tokens),
            ignore_eos=bool(body.get("ignore_eos", False)),
        )
    except (AssertionError, TypeError, ValueError) as e:
        raise RequestError(f"invalid sampling parameters: {e}")


def error_response(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": "invalid_request_error", "code": status}}, status_code=status)


def sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def flash_attn_version() -> str:
    for name in ("flash-attn-3", "flash_attn_3", "flash-attn"):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def engine_info(engine: LLMEngine, served_model_name: str) -> dict:
    config = engine.config
    weights = hashlib.sha256()
    with open(os.path.join(config.model, "config.json"), "rb") as f:
        weights.update(f.read())
    for name in sorted(os.listdir(config.model)):
        if name.endswith(".safetensors"):
            weights.update(f"{name}:{os.path.getsize(os.path.join(config.model, name))}".encode())
    info = {
        "model": served_model_name,
        "model_config_digest": weights.hexdigest()[:16],
        "dtype": str(config.hf_config.dtype),
        "gpu": torch.cuda.get_device_name(),
        "tensor_parallel_size": config.tensor_parallel_size,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flash_attn": flash_attn_version(),
        "enable_determinism": config.enable_determinism,
        "verify_window": config.verify_window,
        "verify_batch_size": config.verify_batch_size,
        "kvcache_block_size": config.kvcache_block_size,
        "max_model_len": config.max_model_len,
    }
    info["system_fingerprint"] = "sid-" + hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()[:16]
    return info


# ---------------------------------------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------------------------------------

def build_app(async_engine: AsyncEngine, served_model_name: str, default_max_tokens: int, api_key: str | None) -> FastAPI:
    engine = async_engine.engine
    tokenizer = engine.tokenizer
    eos_token_ids = engine.config.eos_token_ids
    info = engine_info(engine, served_model_name)
    fingerprint = info["system_fingerprint"]
    app = FastAPI(title="sid")

    @app.middleware("http")
    async def check_api_key(request: Request, call_next):
        if api_key and request.url.path.startswith("/v1") and request.headers.get("authorization") != f"Bearer {api_key}":
            return error_response("invalid api key", 401)
        return await call_next(request)

    @app.exception_handler(RequestError)
    async def request_error(_, e: RequestError):
        return error_response(str(e), e.status)

    async def start(prompt_ids: list[int], body: dict) -> RequestHandle:
        params = sampling_params_from(body, default_max_tokens)
        try:
            return await async_engine.submit(prompt_ids, params)
        except ValueError as e:
            raise RequestError(str(e))
        except RuntimeError as e:
            raise RequestError(str(e), 503)

    async def updates(handle: RequestHandle, state: TextState):
        """yields after every engine update; returns the finish reason when done. aborts on early exit."""
        done = False
        try:
            while True:
                update = await handle.queue.get()
                if isinstance(update, Exception):
                    raise update
                state.add(update.token_ids)
                state.num_rollbacks = update.num_rollbacks
                if state.check_stop():
                    async_engine.abort(handle)
                    done = True
                    yield "stop"
                    return
                if update.finished:
                    done = True
                    yield update.finish_reason
                    return
                yield None
        finally:
            if not done:
                async_engine.abort(handle)

    def usage(handle: RequestHandle, state: TextState) -> dict:
        n = len(state.all_token_ids)
        return {"prompt_tokens": handle.num_prompt_tokens, "completion_tokens": n, "total_tokens": handle.num_prompt_tokens + n}

    def receipt(state: TextState) -> dict:
        return {
            "output_sha256": state.digest(),
            "deterministic": engine.config.enable_determinism,
            "rollbacks": getattr(state, "num_rollbacks", 0),
        }

    @app.get("/health")
    async def health():
        if async_engine.error:
            return JSONResponse({"status": "error", "error": async_engine.error}, status_code=503)
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics():
        scheduler = engine.scheduler
        return {
            **engine.stats,
            "running": len(scheduler.running),
            "waiting": len(scheduler.waiting),
            "free_kvcache_blocks": len(scheduler.block_manager.free_block_ids),
            "total_kvcache_blocks": engine.config.num_kvcache_blocks,
        }

    @app.get("/v1/sid/info")
    async def sid_info():
        return info

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": served_model_name, "object": "model", "created": 0, "owned_by": "sid", "max_model_len": engine.config.max_model_len}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        messages = normalize_messages(body.get("messages"))
        tools = body.get("tools") if body.get("tool_choice") != "none" else None
        template_kwargs = {"enable_thinking": False, **(body.get("chat_template_kwargs") or {})}
        try:
            prompt = tokenizer.apply_chat_template(messages, tools=tools or None, add_generation_prompt=True, tokenize=False, **template_kwargs)
        except Exception as e:
            raise RequestError(f"could not apply chat template: {e}")
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        handle = await start(prompt_ids, body)
        state = TextState(tokenizer, eos_token_ids, parse_stop(body), [TOOL_CALL_OPEN] if tools else [])
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        def finalize(finish_reason: str) -> tuple[str | None, list[dict], str]:
            if tools:
                content, calls = parse_tool_calls(state.text, state.digest())
                if calls:
                    return content, calls, "tool_calls"
            return state.text, [], finish_reason

        if not body.get("stream"):
            finish_reason = None
            async for finish_reason in updates(handle, state):
                pass
            content, calls, finish_reason = finalize(finish_reason)
            message = {"role": "assistant", "content": content}
            if calls:
                message["tool_calls"] = calls
            return {
                "id": request_id, "object": "chat.completion", "created": created, "model": served_model_name,
                "system_fingerprint": fingerprint,
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason, "logprobs": None}],
                "usage": usage(handle, state),
                "sid": receipt(state),
            }

        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

        def chunk(delta: dict, finish_reason: str | None = None, **extra) -> str:
            return sse({
                "id": request_id, "object": "chat.completion.chunk", "created": created, "model": served_model_name,
                "system_fingerprint": fingerprint,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason, "logprobs": None}],
                **extra,
            })

        async def stream():
            yield chunk({"role": "assistant", "content": ""})
            try:
                async for finish_reason in updates(handle, state):
                    if finish_reason is None:
                        text = state.sendable(final=False)
                        if tools and TOOL_CALL_OPEN in text:
                            text = text[: text.find(TOOL_CALL_OPEN)]    # tool calls are sent whole at the end
                        if len(text) > state.num_sent_chars:
                            yield chunk({"content": text[state.num_sent_chars:]})
                            state.num_sent_chars = len(text)
                        continue
                    content, calls, finish_reason = finalize(finish_reason)
                    content = content or ""
                    if len(content) > state.num_sent_chars:
                        yield chunk({"content": content[state.num_sent_chars:]})
                    if calls:
                        yield chunk({"tool_calls": [{"index": i, **call} for i, call in enumerate(calls)]})
                    yield chunk({}, finish_reason, sid=receipt(state))
                    if include_usage:
                        yield sse({"id": request_id, "object": "chat.completion.chunk", "created": created, "model": served_model_name,
                                   "system_fingerprint": fingerprint, "choices": [], "usage": usage(handle, state)})
            except Exception as e:
                yield sse({"error": {"message": str(e), "type": "server_error"}})
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        prompt = body.get("prompt")
        if isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], str):
            prompt = prompt[0]
        if isinstance(prompt, str):
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        elif isinstance(prompt, list) and prompt and all(isinstance(t, int) for t in prompt):
            prompt_ids = prompt
        else:
            raise RequestError("prompt must be a string or a list of token ids")
        handle = await start(prompt_ids, body)
        state = TextState(tokenizer, eos_token_ids, parse_stop(body), [])
        request_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        if not body.get("stream"):
            finish_reason = None
            async for finish_reason in updates(handle, state):
                pass
            return {
                "id": request_id, "object": "text_completion", "created": created, "model": served_model_name,
                "system_fingerprint": fingerprint,
                "choices": [{"index": 0, "text": state.text, "finish_reason": finish_reason, "logprobs": None}],
                "usage": usage(handle, state),
                "sid": receipt(state),
            }

        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

        def chunk(text: str, finish_reason: str | None = None, **extra) -> str:
            return sse({
                "id": request_id, "object": "text_completion", "created": created, "model": served_model_name,
                "system_fingerprint": fingerprint,
                "choices": [{"index": 0, "text": text, "finish_reason": finish_reason, "logprobs": None}],
                **extra,
            })

        async def stream():
            try:
                async for finish_reason in updates(handle, state):
                    text = state.sendable(final=finish_reason is not None)
                    if len(text) > state.num_sent_chars:
                        yield chunk(text[state.num_sent_chars:])
                        state.num_sent_chars = len(text)
                    if finish_reason is not None:
                        yield chunk("", finish_reason, sid=receipt(state))
                        if include_usage:
                            yield sse({"id": request_id, "object": "text_completion", "created": created, "model": served_model_name,
                                       "choices": [], "usage": usage(handle, state)})
            except Exception as e:
                yield sse({"error": {"message": str(e), "type": "server_error"}})
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def main():
    parser = argparse.ArgumentParser(description="sid OpenAI-compatible server")
    parser.add_argument("--model", required=True, help="local HF model directory or hub id")
    parser.add_argument("--served-model-name", default=None, help="model id reported by the API (default: --model)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-kvcache-blocks", type=int, default=-1, help="cap the kv cache, e.g. to force preemption in tests")
    parser.add_argument("--default-max-tokens", type=int, default=2048)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--no-prefix-caching", action="store_true")
    parser.add_argument("--no-determinism", action="store_true", help="disable the verifier (plain fast decode, for comparison)")
    parser.add_argument("--verify-window", type=int, default=32)
    parser.add_argument("--verify-batch-size", type=int, default=16)
    parser.add_argument("--dist-port", type=int, default=2333, help="must be unique per engine on a machine")
    args = parser.parse_args()

    engine = LLMEngine(
        args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_kvcache_blocks=args.max_num_kvcache_blocks,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=not args.no_prefix_caching,
        enable_determinism=not args.no_determinism,
        verify_window=args.verify_window,
        verify_batch_size=args.verify_batch_size,
        dist_port=args.dist_port,
        shm_name=f"sid-{args.dist_port}",
    )
    async_engine = AsyncEngine(engine)
    app = build_app(async_engine, args.served_model_name or args.model, args.default_max_tokens, args.api_key)
    print(json.dumps(engine_info(engine, args.served_model_name or args.model), indent=2))
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
