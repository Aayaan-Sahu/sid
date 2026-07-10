import torch
from torch import nn
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from context import get_context

from typing import Optional, Sequence


def compute_cu_seqlen(lengths, device):
    lengths = torch.tensor(lengths, dtype=torch.int32, device=device)
    zero = torch.zeros(1, dtype=torch.int32, device=device)
    return torch.cat([zero, torch.cumsum(lengths, dim=0)])


@triton.jit
def store_kv_cache_kernel(key_ptr, key_stride, value_ptr, value_stride, k_cache_ptr, v_cache_ptr, slot_mapping_ptr, D: tl.constexpr):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return

    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)

    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)

    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor
):
    # physical kv cache has a block which can hold n tokens. each token is num_heads * head_dim
    # k_cache is [physical_block_id, token_offset_inside_block, kv_head, head_dim]

    # block table maps logical sequence blocks to physical blocks
    # sequence 0 logical block 0 -> physical block 3
    # sequence 0 logical block 1 -> physical block 1
    # sequence 1 logical block 0 -> physical block 4

    """
    block_table = torch.tensor([
        [3, 1],   # sequence 0
        [4, -1],  # sequence 1
    ], dtype=torch.int32, device="cuda")
    """

    # slot_mapping tells where to write newly computed key and value

    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim

    # check assumptions made by triton kernel
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N

    store_kv_cache_kernel[(N, )](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)



class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, softmax_scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.softmax_scale = softmax_scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        # prefill: bool,
        # q_sequences: Sequence[int],
        # k_sequences: Sequence[int],
        # v_sequences: Sequence[int],
        # cache_seqlens: Optional[torch.tensor],
        # slot_mapping: Optional[torch.tensor],
        # block_table: Optional[torch.tensor]
    ):
        """
        instead of [batch, seq_len, heads, head_dim]
        q  : [total_q_tokens, num_q_heads, head_dim]
        kv : [total_kv_tokens, num_kv_heads, head_dim]

        if batch has 3 sequences with query lengths [3, 5, 2]
        cu_seqlens_q = torch.tensor([0, 3, 8, 10]) --> size is alw batch_size + 1
        q_sequences is something like [3, 5, 2]

        cache_seqlens tells us for each sequence in the batch, how many KV tokens are currently valid in the cache
        cache_seqlens = torch.tensor([6, 4], dtype=torch.int32, device="cuda") means
        sequence 0 has 6 cached KV tokens
        sequence 1 has 4 cached KV tokens
        """

        # max_seqlen_q = max(q_sequences)
        # max_seqlen_k = max(k_sequences)
        # cu_seqlens_q = compute_cu_seqlen(q_sequences, q.device)
        # cu_seqlens_k = compute_cu_seqlen(k_sequences, k.device)

        assert q.shape[0] == sum(q_sequences)
        assert k.shape[0] == sum(k_sequences)
        assert v.shape[0] == sum(k_sequences)
        
        # assert cu_seqlens_q[-1].item() == q.shape[0]
        # assert cu_seqlens_k[-1].item() == k.shape[0]

        assert q.shape[-1] == self.head_dim
        assert k.shape[-1] == self.head_dim
        assert v.shape[-1] == self.head_dim

        assert q.shape[1] == self.num_heads
        assert k.shape[1] == self.num_kv_heads
        assert v.shape[1] == self.num_kv_heads

        assert self.num_heads % self.num_kv_heads == 0

        context = get_context()

        if self.k_cache.numel() and self.v_cache.numel():  # if kv cache is not empty, then paged kv cache
            store_kv_cache(k, v, self.k_cache, self.v_cache, context.slot_mapping)
        
        if prefill:
            o = flash_attn_varlen_func(
                q, k, v,
                max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.softmax_scale,
                causal=True,
                block_table=context.block_tables,
            )
        else:
            o = flash_attn_with_kvcache(
                q.unsqueeze(1), k_cache, v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.softmax_scale,
                causal=True,
            )