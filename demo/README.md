# Demo runbook

## The short version (what to show live)

Two servers side by side, same question, ten runs each:

```bash
bash demo/servers.sh start           # deterministic on :8000, non-deterministic on :8001
python demo/compare.py --questions 8 # which questions change under load?
bash demo/servers.sh stop
```

Start with the sweep. A load-induced wobble only changes an answer when two tokens are nearly tied somewhere in it, which is true of roughly one prompt in five, so a single question is a coin flip. The sweep tests eight and prints which ones moved:

```
      question                                     deterministic   non-deterministic
   1  If we cancel our annual plan after 20 days…       10/10              10/10
   2  Does Clinico work with Stripe? How often…         10/10               3/10
   ...
      questions whose answer changed under load           0/8                 2/8
```

Then take a question that moved and tell the agent story with it:

```bash
python demo/compare.py --question "Does Clinico work with Stripe? How often does it sync?"
```

`compare.py` takes about two minutes and prints one screen: how many of the ten runs matched the idle run on each server, and the first place the non-deterministic one diverged (a different tool call, or the character where the answer changed). Run it again with a different `--question` to show it live.

**Each server answers once while idle, then answers the same question again while background traffic flows through it.** That contrast is the whole point. Sending ten copies of a request simultaneously proves nothing: they ride in the same batches, hit identical arithmetic, and agree even without the verifier. What changes an answer is the company a request keeps.

The deterministic server also reports how many drafted tokens its verifier caught and corrected during the run. That number is the nondeterminism itself, fixed before it reached the client.

Everything below is the longer version, for when there is time.

---

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
