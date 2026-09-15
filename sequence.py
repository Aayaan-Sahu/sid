from copy import copy
from enum import Enum, auto
from itertools import count
from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 0.0    # 0 = greedy
    top_p: float = 1.0
    seed: int = 0               # sampling noise is a pure function of (seed, position), so seeded sampling is reproducible
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature >= 0, "temperature must be >= 0"
        assert 0 < self.top_p <= 1, "top_p must be in (0, 1]"
        assert self.max_tokens > 0, "max_tokens must be > 0"
        self.seed %= 2**63


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256 # must be the same as config kvcache_block_size
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams | None = None):
        assert token_ids, "prompt must contain at least one token"
        sampling_params = sampling_params or SamplingParams()
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []

        self.temperature = sampling_params.temperature
        self.top_p = sampling_params.top_p
        self.seed = sampling_params.seed
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

        self.num_verified_tokens = 0
        self.num_rollbacks = 0

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size : (i + 1) * self.block_size]

    def verify_start(self, window: int) -> int:
        # first input position of the verifier window covering the next unverified token. windows sit at
        # fixed offsets from the end of the prompt, so the forward pass a position is recomputed in never
        # depends on where (or whether) the fast decode path happened to diverge.
        return self.num_prompt_tokens + (self.num_verified_tokens - 1 - self.num_prompt_tokens) // window * window

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def rollback(self, num_keep: int, correction_token: int):
        # keep the first num_keep tokens, replace everything after with the verifier's token
        self.token_ids = self.token_ids[:num_keep]
        self.token_ids.append(correction_token)
        self.last_token = correction_token
        self.num_tokens = num_keep + 1
        self.num_cached_tokens = min(self.num_cached_tokens, num_keep)
        self.num_verified_tokens = num_keep + 1
        self.num_rollbacks += 1

    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.num_verified_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.num_verified_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
