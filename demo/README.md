# Demo runbook

Measured on one H200 with Qwen3-8B, 8 questions x 10 runs, 96 concurrent background requests:

| | deterministic | non-deterministic |
|---|---|---|
| questions whose answer changed under load | **0 of 8** | **5 of 8** |

Same model, same GPU, same seed. One flag apart.

## Setup (do this before he is watching)

```bash
bash demo/servers.sh start     # deterministic :8000, non-deterministic :8001; ~2 min
```

Then do a dry run of everything below, and keep the saved results as backup:

```bash
python demo/compare.py --questions 8 --runs 10 --background 96 --out demo/results/sweep.json
```

If a live run ever misbehaves, re-print the saved one without touching the GPU:

```bash
python demo/compare.py --from demo/results/sweep.json
```

## The demo

### 1. Frame it (20 seconds)

Two servers, same model, same GPU, same request, same seed. One has the verifier on, the other off. Nothing else differs.

### 2. Run the sweep (about 2.5 minutes, live)

```bash
python demo/compare.py --questions 8 --runs 10 --background 96
```

Each server answers 8 support questions once while idle, then answers each one 10 more times while 96 other requests flow through it. Talk over it while it runs (see beat 3).

The result to land on:

```
      questions whose answer changed under load    0/8            5/8
```

Then read out the divergence it prints. This one is not cosmetic:

```
  non-deterministic, question 1: step 1 diverges at character 158
    idle        ...contact your administrator to check your role and confirm if SSO can be set up.
    under load  ...contact your administrator to check your role or upgrade to a plan that includes
                   SSO (Growth, Scale, or Enterprise).
```

*Same customer, same question, same settings. One version says ask your admin. The other says buy a bigger plan. The only thing that changed is how busy the server was.*

### 3. Explain why, while it runs (40 seconds)

GPU kernels pick their algorithm based on how many tokens are in the batch. A different algorithm means a different floating-point summation order, so the logits shift slightly. Nearly always that changes nothing. Occasionally two candidate tokens are close enough that it flips one, and from there the answer goes somewhere else.

How many tokens are in a batch depends on what everyone else is doing. So your answer depends on other people's traffic.

sid processes the prompt at a fixed shape, then recomputes every generated token in a verifier that always runs at the same shape. Drafting stays fast; the verifier only confirms. What the batch happens to contain can no longer change the output.

Point at the last line of the output:

```
  deterministic: the verifier caught and corrected 36 drafted tokens during this run
```

*That is the same load-dependent arithmetic happening on the deterministic server too. It just gets corrected before the customer sees it.*

### 4. Show it on a real agent (1 minute)

```bash
python demo/compare.py --question "How do I set up SSO? We're on the Starter plan and I'm not sure I have permission."
```

This runs the full agent loop: search the knowledge base, read results, answer. Tool calls included. Ten runs per server under load, compared against the idle answer.

*When a wobble lands in a tool call instead of prose, the agent searches for something different and the whole trajectory changes. That is why replay and simulation are exact here and approximate everywhere else.*

### 5. Show how it plugs in (30 seconds)

```bash
curl http://localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model": "Qwen/Qwen3-8B", "messages": [{"role": "user", "content": "..."}]}'
```

An OpenAI-compatible endpoint with streaming, tools and seeds. Anything that takes a `base_url` works, including rig-core and the OpenAI SDKs. Every response carries `system_fingerprint` and `sid.output_sha256`: store the hash with a trace, replay it later, and prove the model did the same thing.

### 6. Deeper evidence, if he wants it

```bash
python tests/test_determinism_gpu.py --model Qwen/Qwen3-8B
```

Identical token IDs across: isolated, prefix cache warm, batch of 64, batch of 200, and a deliberately tiny KV cache that forces sequences to be evicted and resumed mid-generation. The `nodet-*` rows are the control.

## Numbers to have ready

| | |
|---|---|
| Sweep | 0 of 8 questions changed with the verifier, 5 of 8 without |
| Bit-exact test | 16/16 outputs identical across isolated, batched-64, batched-200, preempted |
| Rollback rate | about 6% of verify windows (30 of 524 in one run) |
| Throughput | 953 output tok/s with the verifier, 2380 without, at batch 200 |
| Latency under load | agent run averaged 73s with the verifier, 34s without |

## Questions he will ask

**"What does it cost?"** About 2.5x throughput today, and he can see it as latency on screen. Most of it is not the verifier: prompts are currently prefilled one sequence at a time. Batching that at a fixed shape is the next piece of work. Give the number before he finds it.

**"Why not vLLM's batch-invariant mode?"** Same goal, different trade. That mode swaps in batch-invariant kernels and pays on every token. sid keeps fast kernels for drafting and pays only to verify. Worth measuring both; not measured yet.

**"Does this fix hallucinations?"** No. It makes behavior reproducible, which is what makes evals, replay and regression tests trustworthy. Knowledge quality is the other half, and that is the half Brainfish already talks about.

**"Can you do this for hosted models?"** No. Determinism has to come from inside the server. This is for the parts of a stack that run open models.

**"What breaks the guarantee?"** A different GPU model, a different tensor-parallel size, or different torch/flash-attn versions. Each changes the kernels, and `system_fingerprint` changes with them. Within one fingerprint, load and batching cannot change the output.

## Limits to state up front

- Qwen3 dense models only so far; Llama and Qwen2.5 are small additions.
- One node; on one H200 that means up to 32B.
- This covers the model server. If retrieval returns different documents, the prompt is different and so is the answer.
- The questions and knowledge base in the demo are synthetic, generated by `bench/workload.py`.

## Teardown

```bash
bash demo/servers.sh stop
```
