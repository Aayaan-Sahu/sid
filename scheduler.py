from collections import deque

from config import Config
from block_manager import BlockManager
from sequence import Sequence, SequenceStatus

class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.verify_window = config.verify_window
        self.enable_determinism = config.enable_determinism

        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
            config.enable_prefix_caching,
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
    
    def is_finished(self):
        return not self.waiting and not self.running
    
    def add(self, seq: Sequence):
        self.waiting.append(seq)
    
    def pending_finish(self, seq: Sequence) -> bool:
        # seq has hit its stop condition but cannot be emitted until its tail is verified
        return (not seq.ignore_eos and seq.last_token == self.eos) or seq.num_completion_tokens >= seq.max_tokens

    def needs_verify(self, seq: Sequence) -> bool:
        unverified = seq.num_tokens - seq.num_verified_tokens
        return unverified >= self.verify_window or (unverified > 0 and self.pending_finish(seq))

    def schedule(self) -> tuple[list[Sequence], str]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]

            remaining = self.max_num_batched_tokens - num_batched_tokens  # calculate remaining token budget
            if remaining == 0:
                break
            
            if not seq.block_table:  # empty block table means sequence is not in the kv cache yet
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:  # if no space to process then break
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            # num tokens are the remaining tokens we have to process


            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
           
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        # prefer prefill workloads
        if scheduled_seqs:
            return scheduled_seqs, "prefill"
        
        # verify: one seq per step so the verifier always runs at a fixed shape
        if self.enable_determinism:
            for seq in self.running:
                if self.needs_verify(seq):
                    seq.is_prefill = True    # verify serializes like prefill (tp workers need the token window)
                    return [seq], "verify"


        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))

        return scheduled_seqs, "decode"

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        # this is called after the model generates one new token per scheduled sequence

        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            if is_prefill:
                # prompt tokens are given, so they count as verified; every generated token
                # (including any regenerated after preemption) must go through the verifier
                seq.num_verified_tokens = seq.num_prompt_tokens
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                if self.enable_determinism and seq.num_verified_tokens < seq.num_tokens:
                    continue    # hold the finish until the tail is verified (schedule() will pick it for verify)
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

    def postprocess_verify(self, seqs: list[Sequence], results: list[tuple[int, int]]):
        for seq, (num_matched, correction_token) in zip(seqs, results):
            window_start = seq.num_verified_tokens
            window_len = min(self.verify_window, seq.num_tokens - window_start)
            if num_matched == window_len:
                # every decoded token matched the deterministic replay
                seq.num_verified_tokens = window_start + window_len
            else:
                # keep the matched prefix, take the verifier's token at the mismatch, drop the rest
                seq.rollback(window_start + num_matched, correction_token)
                self.block_manager.rollback(seq)
            # a finish held back in postprocess() can complete once fully verified
            if seq.num_verified_tokens == seq.num_tokens and self.pending_finish(seq):
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
