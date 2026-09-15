"""CPU simulation of the deterministic scheduler. no GPU, no torch.

A fake model stands in for the GPU:
  - the "canonical" next token is a hash of the prefix
  - decode drafts are canonical tokens corrupted with probability `noise` (standing in for batch-dependent kernels)
  - prefill and verify return canonical tokens, but only after asserting that every kv slot they read was written
    by a canonical pass with the right token. this catches kv-provenance bugs (shared blocks overwritten,
    misaligned windows, stale kv after preemption or rollback), not just wrong outputs.

Every completion must equal the reference produced by running the prompt alone with no noise, under
random arrival, prefix sharing, multi-turn prompts, preemption (small kv cache) and aborts.

    python tests/test_scheduler_sim.py
"""
import os
import random
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import xxhash

from scheduler import Scheduler
from sequence import Sequence, SamplingParams

VOCAB = 97
EOS = 0


def canonical_token(prefix: list[int]) -> int:
    h = xxhash.xxh64(repr(prefix).encode()).intdigest()
    if h % 41 == 0:
        return EOS
    return 1 + h % (VOCAB - 1)


def reference(prompt: list[int], max_tokens: int, max_model_len: int) -> list[int]:
    tokens = list(prompt)
    while True:
        t = canonical_token(tokens)
        tokens.append(t)
        n_completion = len(tokens) - len(prompt)
        if t == EOS or n_completion >= max_tokens or len(tokens) >= max_model_len:
            return tokens[len(prompt):]


class FakeRunner:
    """Stands in for ModelRunner. `canonical(seq, n)` is the model's token after the first n tokens of seq."""

    def __init__(self, cfg, rng: random.Random, noise: float, canonical=None, vocab_size: int = VOCAB):
        self.cfg = cfg
        self.bs = cfg.kvcache_block_size
        self.rng = rng
        self.noise = noise
        self.canonical = canonical or (lambda seq, n: canonical_token(seq.token_ids[:n]))
        self.vocab_size = vocab_size
        self.kv = {}    # physical slot -> (kind, token)

    def call(self, method_name, *args):
        return getattr(self, method_name)(*args)

    def slot(self, seq, pos):
        assert pos // self.bs < len(seq.block_table), f"seq {seq.seq_id}: no block for position {pos}"
        return seq.block_table[pos // self.bs] * self.bs + pos % self.bs

    def check_canonical(self, seq, upto, what):
        for pos in range(upto):
            got = self.kv.get(self.slot(seq, pos))
            assert got == ("canon", seq[pos]), f"{what}: seq {seq.seq_id} reads non-canonical kv at pos {pos}: {got} (token {seq[pos]})"

    def run(self, seqs, is_prefill):
        out = []
        if is_prefill:
            if self.cfg.enable_determinism:
                assert len(seqs) == 1, "deterministic prefill must be one seq per step"
            for seq in seqs:
                start = seq.num_cached_tokens
                end = start + seq.num_scheduled_tokens
                if self.cfg.enable_determinism:
                    assert start % self.bs == 0, "prefill chunks must start on block boundaries"
                    assert end <= seq.num_prompt_tokens
                    self.check_canonical(seq, start, "prefill")
                for pos in range(start, end):
                    self.kv[self.slot(seq, pos)] = ("canon", seq[pos])
                out.append(self.canonical(seq, end))
        else:
            for seq in seqs:
                pos = seq.num_tokens - 1
                self.kv[self.slot(seq, pos)] = ("draft", seq[pos])
                t = self.canonical(seq, seq.num_tokens)
                if self.rng.random() < self.noise:
                    t = self.rng.randrange(self.vocab_size)
                out.append(t)
        return out

    def run_verify(self, seqs):
        W, B = self.cfg.verify_window, self.cfg.verify_batch_size
        assert len(seqs) <= B
        rows = []
        for seq in seqs:
            start = seq.verify_start(W)
            assert start >= seq.num_prompt_tokens
            assert (start - seq.num_prompt_tokens) % W == 0, "verify windows must stay aligned"
            self.check_canonical(seq, start, "verify")
            for pos in range(start, min(seq.num_tokens, start + W)):
                if pos // self.bs < len(seq.block_table):    # the newest token may not have a block yet
                    self.kv[self.slot(seq, pos)] = ("canon", seq[pos])
                else:
                    assert pos == seq.num_tokens - 1, "only the newest token may lack a block"
            row = []
            for j in range(W):
                if start + j < seq.num_tokens:
                    row.append(self.canonical(seq, start + j + 1))
                else:
                    row.append(self.rng.randrange(self.vocab_size))    # padding rows are garbage
            rows.append(row)
        return rows


def make_config(**overrides):
    cfg = dict(
        max_num_seqs=16,
        max_num_batched_tokens=64,
        max_model_len=200,
        eos_token_ids=frozenset({EOS}),
        kvcache_block_size=16,
        enable_determinism=True,
        verify_window=4,
        verify_batch_size=3,
        num_kvcache_blocks=40,
        enable_prefix_caching=True,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def simulate(seed: int, noise: float = 0.15, abort_rate: float = 0.0, **overrides):
    rng = random.Random(seed)
    cfg = make_config(**overrides)
    Sequence.block_size = cfg.kvcache_block_size
    sched = Scheduler(cfg)
    runner = FakeRunner(cfg, rng, noise)
    preemptions = 0
    preempt = sched.preempt

    def counting_preempt(seq):
        nonlocal preemptions
        preemptions += 1
        preempt(seq)
    sched.preempt = counting_preempt

    base_prompts = [[rng.randrange(1, VOCAB) for _ in range(rng.randrange(1, 60))] for _ in range(4)]
    arrivals = []    # (step, prompt, max_tokens)
    for i in range(30):
        kind = rng.random()
        if kind < 0.4:    # shares a prefix with a base prompt
            prompt = rng.choice(base_prompts) + [rng.randrange(1, VOCAB) for _ in range(rng.randrange(0, 40))]
        elif kind < 0.6:  # exact duplicate
            prompt = list(rng.choice(base_prompts))
        else:
            prompt = [rng.randrange(1, VOCAB) for _ in range(rng.randrange(1, 90))]
        arrivals.append((rng.randrange(0, 150), prompt, rng.randrange(1, 60)))
    arrivals.sort(key=lambda a: a[0])

    live = []            # (seq, max_tokens)
    emitted = {}         # seq_id -> verified tokens seen so far; must only ever grow
    multi_turn_budget = 8
    step = 0
    while arrivals or not sched.is_finished():
        while arrivals and arrivals[0][0] <= step:
            _, prompt, max_tokens = arrivals.pop(0)
            prompt = prompt[: cfg.max_model_len - 1]
            max_tokens = min(max_tokens, cfg.max_model_len - len(prompt))
            seq = Sequence(prompt, SamplingParams(max_tokens=max_tokens))
            sched.add(seq)
            live.append((seq, max_tokens))
        seqs, mode = sched.schedule()
        step += 1
        if mode == "idle":
            assert arrivals, "scheduler idle with work pending (deadlock)"
            continue
        if mode == "verify":
            sched.postprocess_verify(seqs, runner.run_verify(seqs))
        else:
            sched.postprocess(seqs, runner.run(seqs, mode == "prefill"), mode == "prefill")

        for seq, _ in live:
            if abort_rate and not seq.is_finished and rng.random() < abort_rate / 50:
                sched.abort(seq)
                seq.aborted = True
            # once a token has been verified it may be streamed to a client, so it must never change. (after a
            # preemption num_verified_tokens drops back while the verifier rebuilds kv; tokens must still match)
            prev = emitted.get(seq.seq_id, [])
            current = seq.token_ids[seq.num_prompt_tokens: seq.num_prompt_tokens + len(prev)]
            assert current == prev, f"seq {seq.seq_id}: tokens changed after being emitted\n  was {prev}\n  now {current}"
            verified = seq.token_ids[seq.num_prompt_tokens: max(seq.num_verified_tokens, seq.num_prompt_tokens)]
            if len(verified) > len(prev):
                emitted[seq.seq_id] = verified
            if seq.is_finished and not getattr(seq, "followed_up", False) and multi_turn_budget > 0:
                seq.followed_up = True
                multi_turn_budget -= 1
                prompt = seq.token_ids + [rng.randrange(1, VOCAB) for _ in range(rng.randrange(1, 20))]
                if len(prompt) < cfg.max_model_len - 1:
                    arrivals.append((step + rng.randrange(0, 5), prompt, rng.randrange(1, 40)))
                    arrivals.sort(key=lambda a: a[0])
        assert step < 100_000, "simulation did not terminate"

    bm = sched.block_manager
    assert not bm.used_block_ids, "blocks leaked"
    assert len(bm.free_block_ids) == cfg.num_kvcache_blocks
    checked = 0
    for seq, max_tokens in live:
        if getattr(seq, "aborted", False):
            continue
        expected = reference(seq.prompt_token_ids, max_tokens, cfg.max_model_len)
        assert seq.completion_token_ids == expected, (
            f"seed {seed}: seq {seq.seq_id} diverged\n  got      {seq.completion_token_ids}\n  expected {expected}"
        )
        checked += 1
    return checked, sum(seq.num_rollbacks for seq, _ in live), preemptions


def main():
    scenarios = {
        "determinism, roomy kv cache": dict(num_kvcache_blocks=200),
        "determinism, tight kv cache (preemption)": dict(num_kvcache_blocks=24),
        "determinism, heavy noise": dict(noise=0.6),
        "determinism, aborts": dict(abort_rate=1.0),
        "determinism, W=1 B=1": dict(verify_window=1, verify_batch_size=1),
        "determinism, W=16 (== block size)": dict(verify_window=16, verify_batch_size=8),
        "determinism, no prefix caching": dict(enable_prefix_caching=False),
        "no determinism, no noise": dict(enable_determinism=False, noise=0.0),
    }
    for name, overrides in scenarios.items():
        total, rollbacks, preemptions = 0, 0, 0
        for seed in range(60):
            checked, r, p = simulate(seed, **overrides)
            total += checked
            rollbacks += r
            preemptions += p
        print(f"ok  {name:45s} {total:5d} completions match reference, {rollbacks:5d} rollbacks, {preemptions:4d} preemptions")


if __name__ == "__main__":
    main()
