import torch

# sampling is made reproducible by replacing the rng with a counter-based hash: the gumbel noise for
# (seed, position, vocab id) is computed with exact integer ops, and everything after that is elementwise
# or a per-row op. so a row's token depends only on its own logits, seed and position, never on the batch.

_M32 = 0xFFFFFFFF
_MUL = 0x45D9F3B        # < 2**31, so (x < 2**32) * _MUL never overflows int64
_ROW_CHUNK = 32         # bounds the [rows, vocab] int64 scratch tensors


def _hash32(x: torch.Tensor) -> torch.Tensor:
    x = x ^ (x >> 16)
    x = (x * _MUL) & _M32
    x = x ^ (x >> 16)
    x = (x * _MUL) & _M32
    return x ^ (x >> 16)


def top_p_drop_mask(scaled: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    # True where a token falls outside the nucleus. a token is dropped once the mass strictly before it already
    # reaches top_p. rows with top_p == 1 never drop anything: a float cumsum can reach 1.0 before the tail, and
    # whether this mask is computed at all depends on the other rows in the chunk
    probs = torch.softmax(scaled, dim=-1)
    sorted_probs, sorted_ids = probs.sort(dim=-1, descending=True)
    drop_sorted = ((sorted_probs.cumsum(dim=-1) - sorted_probs) >= top_p[:, None]) & (top_p < 1)[:, None]
    return torch.empty_like(drop_sorted).scatter_(-1, sorted_ids, drop_sorted)


def gumbel_noise(seeds: torch.Tensor, positions: torch.Tensor, vocab_size: int) -> torch.Tensor:
    # seeds, positions: int64 [rows] -> float64 [rows, vocab]
    key = _hash32(_hash32(seeds & _M32) ^ ((seeds >> 32) & _M32))
    key = _hash32(key ^ (positions & _M32))[:, None]
    vocab = torch.arange(vocab_size, dtype=torch.int64, device=seeds.device)[None, :]
    bits = _hash32(_hash32(key ^ vocab) ^ key)
    u = (bits.double() + 0.5) / 2**32
    return -torch.log(-torch.log(u))


class Sampler:
    def __call__(
        self,
        logits: torch.Tensor,
        temperatures: list[float],
        top_ps: list[float],
        seeds: list[int],
        positions: list[int],
    ) -> list[int]:
        tokens = torch.argmax(logits, dim=-1)
        if max(temperatures) == 0:
            return tokens.tolist()

        device = logits.device
        temperatures = torch.tensor(temperatures, dtype=torch.float32, device=device)
        top_ps = torch.tensor(top_ps, dtype=torch.float32, device=device)
        seeds = torch.tensor(seeds, dtype=torch.int64, device=device)
        positions = torch.tensor(positions, dtype=torch.int64, device=device)
        sampled = torch.empty_like(tokens)
        for i in range(0, logits.size(0), _ROW_CHUNK):
            rows = slice(i, i + _ROW_CHUNK)
            t = temperatures[rows]
            scaled = logits[rows].float() / torch.where(t > 0, t, 1.0)[:, None]
            top_p = top_ps[rows]
            if (top_p < 1).any():
                scaled = scaled.masked_fill(top_p_drop_mask(scaled, top_p), float("-inf"))
            noise = gumbel_noise(seeds[rows], positions[rows], logits.size(-1))
            sampled[rows] = torch.argmax(scaled.double() + noise, dim=-1)
        return torch.where(temperatures > 0, sampled, tokens).tolist()
