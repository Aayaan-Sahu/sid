# Demo runbook

One command, about 15 minutes, no second inference server needed:

```bash
bash demo/run_demo.sh
```

It runs the same engine twice on the same GPU with the same model, once with the verifier on and once with it off, and writes everything to `demo/results/`.

To re-print the comparison later, without touching the GPU:

```bash
python demo/side_by_side.py --report demo/results/det.json demo/results/nodet.json
```

## What to show, in order

### 1. The problem (1 minute)

Send one support question 20 times, all at once, while 64 other requests are in flight. Temperature 0.7 with a fixed seed, which is the setting where a caller already expects to get the same answer back.

With the verifier off, the same request returns several different answers. The script prints the exact character where two runs diverge, with the two continuations underneath.

Say this: *the request never changed. Server load did. Kernels pick different algorithms depending on how many tokens are batched together, so the arithmetic changes, one token flips, and the answer goes somewhere else.*

### 2. The fix (1 minute)

Same screen, verifier on: 20/20 identical, one distinct answer.

Say this: *the prompt is processed at a fixed shape, and every generated token is recomputed by a verifier that always runs at the same shape. What the batch happens to contain can no longer change the output. Fast decoding still does the work; the verifier only confirms it.*

### 3. It holds where it usually breaks (1 minute)

```bash
python tests/test_determinism_gpu.py --model Qwen/Qwen3-8B
```

Identical token IDs across: isolated, prefix cache warm, batch of 64, batch of 200, and a deliberately tiny KV cache that forces sequences to be evicted and resumed mid-generation. The `nodet-*` rows are the control.

### 4. It's a real agent, not a single call (1 minute)

The agent replay in the demo output runs a multi-step tool-calling agent (knowledge base search, then answer) five times under load. With the verifier on, all five transcripts are identical, tool arguments included.

Say this: *this is what makes replay and simulation exact. Run the same trace twice and you get the same thing, so a regression test that fails means the change caused it, not the scheduler.*

### 5. How it plugs in (30 seconds)

```bash
curl http://localhost:8000/v1/chat/completions -d '{"model": "Qwen/Qwen3-8B", "messages": [...]}'
```

An OpenAI-compatible endpoint: streaming, tools, seeds. Anything that takes a `base_url` works, including rig-core and the OpenAI SDKs. Every response carries `system_fingerprint` and `sid.output_sha256`, so a stored trace can be replayed and verified.

## Numbers to have ready

From the last run on one H200 with Qwen3-8B:

| | |
|---|---|
| Determinism | 16/16 outputs bit-identical across isolated, batched-64, batched-200, preempted |
| Without the verifier | 3/16 outputs changed between running alone and in a batch of 200 (greedy; far more at temperature 0.7) |
| Rollback rate | 30 of 524 verify windows, about 6% |
| Throughput | 953 output tok/s with the verifier, 2380 without, at batch 200 |

## Questions he will ask

**"What does it cost?"** Right now about 2.5x throughput at batch 200. Most of it isn't the verifier: prompts are currently prefilled one sequence at a time. Batching that at a fixed shape is the next piece of work. Be straight about the number.

**"Why not vLLM's batch-invariant mode?"** Same goal, different trade. That mode replaces kernels with batch-invariant ones and pays on every token. sid keeps the fast kernels for drafting and pays only to verify. Worth measuring both; not yet measured here.

**"Does this fix hallucinations?"** No. It makes behavior reproducible, which is what makes evals, replay and regression tests trustworthy. Brainfish's own framing (knowledge quality) is the other half.

**"Can you do this for our hosted models?"** No. Determinism has to come from inside the server. This is for the parts of the stack that run open models.

**"What breaks the guarantee?"** A different GPU model, a different tensor-parallel size, or different torch/flash-attn versions. Each of those changes the kernels, and the `system_fingerprint` changes with them. Within one fingerprint, load and batching cannot change the output.

## Limits to state up front

- Qwen3 dense models only so far; Llama and Qwen2.5 are small additions.
- One node; on one H200 that means up to 32B.
- Determinism applies to the model server. If retrieval returns different documents, the prompt is different and so is the answer.
