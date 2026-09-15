import os
from dataclasses import dataclass, field
from transformers import AutoConfig, GenerationConfig


def resolve_model_path(model: str) -> str:
    # accept either a local HF model directory or a hub id like "Qwen/Qwen3-8B"
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download
    return snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"])


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 32768     # max tokens processed in one batch
    max_num_seqs: int = 256                 # limits num active sequences in one batch
    max_model_len: int = 32768              # max context length
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos_token_ids: frozenset[int] = field(default_factory=frozenset)
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    max_num_kvcache_blocks: int = -1        # cap the kv cache (e.g. to force preemption in tests); -1 = no cap
    enable_prefix_caching: bool = True

    # multi-process plumbing; must be unique per engine instance on a machine
    dist_port: int = 2333
    shm_name: str = "sid"

    # determinism
    enable_determinism: bool = True
    verify_window: int = 32                 # W: tokens per verifier window
    verify_batch_size: int = 16             # B: verifier always runs B*W tokens, padding unused slots

    def __post_init__(self):
        self.model = resolve_model_path(self.model)
        assert self.kvcache_block_size > 0
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.verify_window > 0 and self.verify_batch_size > 0
        assert self.verify_window <= self.kvcache_block_size
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        self.max_num_batched_tokens = max(self.max_num_batched_tokens, self.kvcache_block_size)
        if not self.eos_token_ids:
            self.eos_token_ids = load_eos_token_ids(self.model)

    @property
    def max_num_blocks_per_seq(self) -> int:
        # a verify window may reach W-1 positions past the end of the sequence
        return (self.max_model_len + self.verify_window + self.kvcache_block_size - 1) // self.kvcache_block_size


def load_eos_token_ids(model: str) -> frozenset[int]:
    from transformers import AutoTokenizer
    ids = set()
    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    try:
        eos = GenerationConfig.from_pretrained(model).eos_token_id
        if isinstance(eos, int):
            ids.add(eos)
        elif eos:
            ids.update(eos)
    except OSError:
        pass
    return frozenset(ids)
