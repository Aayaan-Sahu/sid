"""Bit-exact determinism test for the engine on a real GPU.

Runs the same agent prompts in separate engine processes under different conditions and requires identical
completion token ids from every deterministic run:

    alone-cold / alone-warm   each prompt by itself (second pass hits the prefix cache)
    batched-64 / batched-200  prompts mixed into a big batch of other requests, in a different order
    preempt                   a deliberately tiny kv cache, so sequences get preempted and resumed
    nodet-*                   the same with the verifier off, to show the problem being solved

    python tests/test_determinism_gpu.py --model Qwen/Qwen3-8B
"""
import argparse
import json
import os
import random
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WORKERS = {
    "alone": dict(det=True, mode="alone"),
    "batched-64": dict(det=True, mode="batched", fillers=64, order_seed=1),
    "batched-200": dict(det=True, mode="batched", fillers=200, order_seed=2),
    "preempt": dict(det=True, mode="batched", fillers=96, order_seed=3, kv_blocks=48),
    "nodet-alone": dict(det=False, mode="alone"),
    "nodet-batched-200": dict(det=False, mode="batched", fillers=200, order_seed=2),
}


def worker(args):
    from bench.workload import filler_conversations, render, target_conversations
    from llm_engine import LLMEngine
    from sequence import SamplingParams

    spec = WORKERS[args.worker]
    engine = LLMEngine(
        args.model,
        enable_determinism=spec["det"],
        max_model_len=args.max_model_len,
        max_num_kvcache_blocks=spec.get("kv_blocks", -1),
    )
    tokenizer = engine.tokenizer
    targets = [render(tokenizer, conv) for conv in target_conversations(args.targets)]
    samplings = {
        "greedy": SamplingParams(max_tokens=args.max_tokens),
        "sampled": SamplingParams(temperature=0.8, top_p=0.95, seed=7, max_tokens=args.max_tokens),
    }
    outputs = {}
    metrics = {}
    if spec["mode"] == "alone":
        for run in ("cold", "warm"):
            for name, sp in samplings.items():
                for i, prompt in enumerate(targets):
                    outputs.setdefault(f"{args.worker}-{run}", {})[f"{name}/{i}"] = engine.generate([prompt], sp, use_tqdm=False)[0]["token_ids"]
    else:
        fillers = filler_conversations(spec["fillers"], seed=spec["order_seed"])
        prompts, params, keys = [], [], []
        for i, prompt in enumerate(targets):
            for name, sp in samplings.items():
                prompts.append(prompt)
                params.append(sp)
                keys.append(f"{name}/{i}")
        for conv in fillers:
            prompts.append(render(tokenizer, conv))
            params.append(SamplingParams(max_tokens=conv["max_tokens"], **conv["sampling"]))
            keys.append(None)
        order = list(range(len(prompts)))
        random.Random(spec["order_seed"]).shuffle(order)
        results = engine.generate([prompts[j] for j in order], [params[j] for j in order], use_tqdm=True)
        outputs[args.worker] = {keys[j]: results[k]["token_ids"] for k, j in enumerate(order) if keys[j]}
        metrics = engine.last_metrics
    with open(args.out, "w") as f:
        json.dump({"outputs": outputs, "metrics": metrics, "stats": dict(engine.stats)}, f)
    engine.exit()


def driver(args):
    runs, metrics = {}, {}
    for name in args.only or WORKERS:
        out = os.path.join(tempfile.gettempdir(), f"sid-det-{name}.json")
        cmd = [sys.executable, __file__, "--worker", name, "--out", out, "--model", args.model,
               "--targets", str(args.targets), "--max-tokens", str(args.max_tokens), "--max-model-len", str(args.max_model_len)]
        print(f"=== {name}")
        subprocess.run(cmd, check=True)
        with open(out) as f:
            data = json.load(f)
        runs.update(data["outputs"])
        metrics[name] = {**data["metrics"], **data["stats"]}

    reference_name = "alone-cold"
    reference = runs[reference_name]
    failures = 0
    print()
    print(f"{'run':22s} {'identical to ' + reference_name:>28s}   first divergence (token index)")
    for name, outs in runs.items():
        same, divergences = 0, []
        for key, ids in outs.items():
            ref = reference[key]
            if ids == ref:
                same += 1
            else:
                divergences.append(next((i for i, (a, b) in enumerate(zip(ids, ref)) if a != b), min(len(ids), len(ref))))
        deterministic = not name.startswith("nodet")
        if deterministic and same != len(outs):
            failures += 1
        note = f"{sorted(divergences)[:6]}" if divergences else ""
        flag = "" if not deterministic else ("  FAIL" if same != len(outs) else "  ok")
        print(f"{name:22s} {same:>12d}/{len(outs):<15d} {note}{flag}")
    print()
    for name, m in metrics.items():
        if m.get("output_tokens_per_second"):
            print(f"{name:22s} {m['output_tokens_per_second']:8.0f} output tok/s   rollbacks {m.get('rollbacks', 0)}   verify windows {m.get('verify_windows', 0)}")
    nodet_alone = runs.get("nodet-alone-cold")
    if nodet_alone and "nodet-batched-200" in runs:
        diff = sum(runs["nodet-batched-200"][k] != v for k, v in nodet_alone.items())
        print(f"\nwithout the verifier, {diff}/{len(nodet_alone)} outputs changed between running alone and running in a batch of 200")
    if failures:
        print(f"\n{failures} deterministic run(s) did not match")
        sys.exit(1)
    print("\nall deterministic runs are bit-identical")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--targets", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--only", nargs="*", help="subset of workers to run")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--out", help=argparse.SUPPRESS)
    args = parser.parse_args()
    worker(args) if args.worker else driver(args)
