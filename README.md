# sid

sid is an LLM inference engine whose outputs don't depend on server load. The same request with the same model and settings returns the same tokens, whether it runs alone or next to 200 other requests.

It exposes an OpenAI-compatible API, so any agent stack that can take a custom `base_url` works with it: the OpenAI SDKs, LangChain, rig-core, and others.

## Why outputs usually change under load

Temperature 0 is not deterministic on standard inference servers. Many GPU kernels, including matrix multiply, attention and normalization, pick their algorithm based on how many tokens are in the batch. A different algorithm means a different floating-point reduction order, so the logits come out slightly different. Once a single token flips, the rest of the answer diverges.

How many tokens end up in a batch depends on what other users are doing. That makes the output of your request depend on everyone else's traffic.

## How sid removes that

```
prompt           prefilled one sequence at a time, in chunks aligned to KV-cache blocks
                 -> prompt KV and prefix-cache hits are bit-identical under any load
first token      sampled from the last prefill chunk
later tokens     drafted by fast batched decode (CUDA graphs, any batch size), then
                 recomputed by a verifier in windows of W tokens, always at a fixed
                 B x W shape with pinned attention splits and eager kernels
mismatch         roll back to the verifier's token and keep going
streaming        only verified tokens are ever sent, so nothing is retracted
sampling         noise = hash(seed, position, vocab id), so temperature > 0 with a seed
                 is reproducible too
```

Each emitted token is computed by a forward pass whose shape depends only on that request's own content, never on the rest of the batch.

The fast decode path does most of the work. The verifier only confirms its drafts, and most windows match on the first try.

### What is guaranteed

The same output tokens are guaranteed for the same request when all of these hold:
- the same model weights
- the same `system_fingerprint`, which covers GPU type, tensor-parallel size, torch/CUDA/flash-attn versions, and the engine's W, B and block size

This holds regardless of:
- concurrency and arrival order
- prefix-cache state
- preemption

### What is not guaranteed

- **Identical output across different GPU models, TP sizes or library versions.** Those change the kernels.
- **Anything upstream of the model.** If your retrieval step returns different documents, the prompt is different.
- **Hosted APIs.** OpenAI, Anthropic and similar providers can't be made deterministic from the outside.

## Supported models and hardware

- **Model family:** Qwen3 dense models, 0.6B to 32B.
  - A single H200 fits Qwen3-32B in bf16 with room for KV cache.
  - Tensor parallelism works across up to 8 GPUs on one node.
- **GPU:** FlashAttention-3 needs Hopper-class GPUs (H100/H200).

## Setup (H100/H200)

```bash
bash setup.sh
```

## Run the server

```bash
python server.py --model Qwen/Qwen3-8B --port 8000
```

On startup it prints the engine fingerprint. Useful flags:

| Flag | What it does |
|---|---|
| `--no-determinism` | Plain fast decode, for comparison |
| `--max-num-kvcache-blocks N` | Shrinks the KV cache to force preemption |
| `--verify-window` / `--verify-batch-size` | Sets W / B |
| `--api-key` | Requires a key on requests |
| `--dist-port` | Must be unique for each engine on a machine |

```bash
curl http://localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "Qwen/Qwen3-8B",
  "messages": [{"role": "user", "content": "How do I rotate an API key?"}]
}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="sid")
r = client.chat.completions.create(model="Qwen/Qwen3-8B", messages=[...], tools=[...])
print(r.system_fingerprint, r.model_extra["sid"]["output_sha256"])
```

### API notes

- **Endpoints:** `/v1/chat/completions` (streaming, tools, stop strings, seed, top_p), `/v1/completions`, `/v1/models`, `/health`, `/metrics`, `/v1/sid/info`.
- **Defaults:** `temperature` defaults to 0 and `seed` defaults to 0. Requests are reproducible unless you opt out.
- **Streaming:** tokens arrive in bursts of up to W. Each burst is final.
- **Receipts:** every response includes `system_fingerprint` and `sid.output_sha256`. The same fingerprint and the same request always give the same hash, so a stored agent trace can be replayed and checked.
- **Tool calls:** Qwen3's native `<tool_call>` format is parsed into OpenAI `tool_calls`. Tool call IDs are derived from the output, so replays match byte for byte.
- **Thinking mode:** off by default. Pass `"chat_template_kwargs": {"enable_thinking": true}` to turn it on.

## Tests

On CPU (no GPU needed):

```bash
python tests/test_scheduler_sim.py   # scheduler + KV provenance under noise, preemption, aborts, prefix sharing
python tests/test_sampler.py         # seeded sampling is batch-invariant and has the right distribution
python tests/test_server_fake.py     # HTTP API end to end with a scripted fake model
```

On the GPU, a bit-exact check across isolated, batched, preempted and prefix-cache-warm runs, with the verifier off as a control:

```bash
python tests/test_determinism_gpu.py --model Qwen/Qwen3-8B
```

## Benchmark against other servers

Start the servers you want to compare. The model is the same for all of them; use one GPU at a time, or lower their memory settings.

```bash
python server.py --model Qwen/Qwen3-8B --port 8000
vllm serve Qwen/Qwen3-8B --port 8001 --enable-auto-tool-choice --tool-call-parser hermes
```

Include vLLM's batch-invariant mode as well, if your version has it. Check the vLLM docs for the current flag.

Then run both benchmarks:

```bash
python bench/determinism.py --endpoint sid=http://localhost:8000 --endpoint vllm=http://localhost:8001 \
    --targets 8 --repeats 3 --concurrency 32,128
python bench/agent_replay.py --base-url http://localhost:8000/v1 --runs 5 --background 64
```

- **`bench/determinism.py`** sends support-agent requests: a knowledge base of about 2.3k tokens, three tools, and a customer question. It runs them isolated, with the prefix cache warm, and under background load. For each target it counts how many distinct outputs came back, and it also reports throughput and TTFT.
- **`bench/agent_replay.py`** runs a multi-step tool-using agent several times while the server is busy, and checks that the transcripts are identical.
