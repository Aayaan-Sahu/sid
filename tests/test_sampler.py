"""CPU tests for the counter-based sampler.

    python tests/test_sampler.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from sampler import Sampler, gumbel_noise, top_p_drop_mask


def test_row_independent_of_batch():
    torch.manual_seed(0)
    vocab = 1000
    sampler = Sampler()
    target = torch.randn(1, vocab) * 3
    params = dict(temperature=0.8, top_p=0.9, seed=1234, position=77)
    alone = sampler(target, [0.8], [0.9], [1234], [77])[0]
    for batch_size in (2, 7, 33, 64, 100):
        for index in (0, batch_size // 2, batch_size - 1):
            logits = torch.randn(batch_size, vocab) * 3
            logits[index] = target[0]
            temps = [float(torch.rand(1)) for _ in range(batch_size)]
            top_ps = [0.5 + 0.5 * float(torch.rand(1)) for _ in range(batch_size)]
            seeds = [int(torch.randint(0, 2**62, (1,))) for _ in range(batch_size)]
            positions = [int(torch.randint(0, 10000, (1,))) for _ in range(batch_size)]
            temps[index], top_ps[index], seeds[index], positions[index] = params["temperature"], params["top_p"], params["seed"], params["position"]
            got = sampler(logits, temps, top_ps, seeds, positions)[index]
            assert got == alone, f"batch {batch_size} index {index}: {got} != {alone}"
    print("ok  sampled token independent of batch size and row position")


def test_top_p_one_never_drops():
    # two tokens at ~0.5 and a tail at ~1e-13: the float32 cumsum hits exactly 1.0 after two tokens, so a naive
    # "mass before >= top_p" mask would drop the whole tail of a top_p == 1 row whenever the mask runs for its chunk
    scaled = torch.full((2, 1000), -30.0)
    scaled[:, :2] = 0.0
    probs = torch.softmax(scaled, -1)
    sorted_probs, _ = probs.sort(-1, descending=True)
    naive = (sorted_probs.cumsum(-1) - sorted_probs) >= 1.0
    assert naive[0].sum() > 0, "test setup no longer triggers early cumsum saturation"
    drop = top_p_drop_mask(scaled, torch.tensor([1.0, 0.4]))
    assert not drop[0].any(), "top_p=1 row dropped tokens"
    assert drop[1].sum() == 999, "top_p=0.4 row should keep only the top token"
    print(f"ok  top_p=1 never drops tokens (a naive mask would drop {int(naive[0].sum())} here)")


def test_greedy_rows():
    torch.manual_seed(1)
    logits = torch.randn(40, 500)
    temps = [0.0 if i % 2 else 1.0 for i in range(40)]
    out = Sampler()(logits, temps, [1.0] * 40, list(range(40)), list(range(40)))
    argmax = logits.argmax(-1).tolist()
    assert all(out[i] == argmax[i] for i in range(1, 40, 2))
    print("ok  temperature 0 rows are greedy")


def test_distribution():
    # sampling many positions of a fixed distribution must match softmax(logits / T)
    vocab, n, temperature = 8, 40000, 0.7
    logits = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -3.0])
    tokens = Sampler()(logits.repeat(n, 1), [temperature] * n, [1.0] * n, [42] * n, list(range(n)))
    freq = torch.bincount(torch.tensor(tokens), minlength=vocab).double() / n
    expected = torch.softmax(logits.double() / temperature, -1)
    err = (freq - expected).abs().max().item()
    assert err < 0.01, f"empirical distribution off by {err}: {freq} vs {expected}"
    print(f"ok  empirical distribution matches softmax (max abs err {err:.4f})")


def test_top_p():
    vocab, n = 8, 20000
    logits = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -3.0])
    probs = torch.softmax(logits, -1)
    top_p = 0.8
    tokens = Sampler()(logits.repeat(n, 1), [1.0] * n, [top_p] * n, [7] * n, list(range(n)))
    cum_before = torch.cumsum(probs, -1) - probs
    allowed = set(torch.nonzero(cum_before < top_p).flatten().tolist())
    assert set(tokens) <= allowed, f"top_p leaked tokens {set(tokens) - allowed}"
    assert set(tokens) == allowed
    print(f"ok  top_p={top_p} samples exactly from the nucleus {sorted(allowed)}")


def test_seeds_differ():
    noise_a = gumbel_noise(torch.tensor([1]), torch.tensor([5]), 1000)
    noise_b = gumbel_noise(torch.tensor([2]), torch.tensor([5]), 1000)
    noise_c = gumbel_noise(torch.tensor([1]), torch.tensor([6]), 1000)
    assert not torch.equal(noise_a, noise_b) and not torch.equal(noise_a, noise_c)
    assert torch.equal(noise_a, gumbel_noise(torch.tensor([1]), torch.tensor([5]), 1000))
    big = gumbel_noise(torch.tensor([2**62 + 3]), torch.tensor([5]), 1000)
    assert torch.isfinite(big).all()
    print("ok  noise is a pure function of (seed, position) and differs across them")


if __name__ == "__main__":
    test_row_independent_of_batch()
    test_top_p_one_never_drops()
    test_greedy_rows()
    test_distribution()
    test_top_p()
    test_seeds_differ()
