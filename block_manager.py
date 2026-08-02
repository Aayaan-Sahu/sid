import xxhash
import numpy as np
from collections import deque

from sequence import Sequence

class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []
    
    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids
    
    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool = True):
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()    # quick access to kv cache block given token ids
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
    
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        # create a hash for a group of tokens
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()
    
    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0    # verify no active sequence is using the block

        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:   # clear old k/v cache data from previous sequence
            del self.hash_to_block_id[block.hash]
        
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id
    
    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        if not self.enable_prefix_caching:
            return 0 if len(self.free_block_ids) >= seq.num_blocks else -1

        h = -1
        num_cached_blocks = 0            # how many blocks can be prefix cached
        num_new_blocks = seq.num_blocks  # how many new blocks need allocation

        # final block is 
        for i in range(seq.num_blocks - 1):
            
            # this part enables prefix caching
            # we're going through token by token and seeing if the hash exists
            # if not, then the block_id is -1 which means new
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)   # compute hash: depends on prev hash
            block_id = self.hash_to_block_id.get(h, -1)

            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
    
        if len(self.free_block_ids) < num_new_blocks: # check if we have enough blocks available
            return -1

        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        # we are populating a sequence's block table with what blocks to use

        assert not seq.block_table

        # first do the blocks that are "prefix cached"
        h = -1
        for i in range(num_cached_blocks):
            # get the right blok
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]

            if block_id in self.used_block_ids:     # someone else is using the block, i am too
                block.ref_count += 1
            else:                                   # first time using the block
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)

            seq.block_table.append(block_id)

        # then allocate rest of blocks
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def _invalidate_hash(self, block: Block):
        if block.hash != -1:
            if self.hash_to_block_id.get(block.hash) == block.block_id:
                del self.hash_to_block_id[block.hash]
            block.hash = -1
            block.token_ids = []

    def rollback(self, seq: Sequence):
        # free blocks past the truncated length; their hashes describe rolled-back tokens,
        # so drop them from the prefix cache before another seq can match them
        while len(seq.block_table) > seq.num_blocks:
            block_id = seq.block_table.pop()
            block = self.blocks[block_id]
            self._invalidate_hash(block)
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        # the boundary block may have been complete and hashed; it is now partial and
        # will be overwritten by new tokens, so its prefix-cache entry is stale
        block = self.blocks[seq.block_table[-1]]
        # TODO: if another seq prefix-matched this block (ref_count > 1), overwriting its
        # kv would corrupt that seq; needs copy-on-write to support prefix sharing + rollback
        assert block.ref_count == 1, "rollback into a shared block is not supported"
        self._invalidate_hash(block)


    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        if not self.enable_prefix_caching:
            return
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
