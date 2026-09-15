from collections import deque
from typing import TYPE_CHECKING

from block_manager import BlockManager
from sequence import Sequence, SequenceStatus

if TYPE_CHECKING:
    from config import Config


# deterministic mode, in one picture:
#
#   prompt kv      written only by canonical prefill: one seq per step, chunked at block boundaries
#   first token    sampled from the last prefill chunk (canonical, so it counts as verified)
#   later tokens   drafted by fast batched decode, then recomputed by the verifier in windows of W inputs
#                  aligned at prompt_len + k*W, always run at a fixed B*W shape. the verifier's tokens
#                  are the output; a draft that disagrees is rolled back to the verifier's token.
#
# every emitted token is a function of the prompt and sampling params alone, not of what else was running.

class Scheduler:
    def __init__(self, config: "Config"):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_model_len = config.max_model_len
        self.eos_token_ids = config.eos_token_ids
        self.block_size = config.kvcache_block_size
        self.enable_determinism = config.enable_determinism
        self.verify_window = config.verify_window
        self.verify_batch_size = config.verify_batch_size

        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
            config.enable_prefix_caching,
            config.enable_determinism,
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def abort(self, seq: Sequence) -> bool:
        if seq.is_finished:
            return False
        if seq in self.waiting:
            self.waiting.remove(seq)
        elif seq in self.running:
            self.running.remove(seq)
        if seq.block_table:
            self.block_manager.deallocate(seq)
        seq.status = SequenceStatus.FINISHED
        return True

    def prefill_target(self, seq: Sequence) -> int:
        # in deterministic mode only the prompt is prefilled; generated positions get their kv from the verifier
        return seq.num_prompt_tokens if self.enable_determinism else seq.num_tokens

    def pending_finish(self, seq: Sequence) -> bool:
        # seq has hit a stop condition (in deterministic mode it cannot be emitted until its tail is verified)
        return seq.num_completion_tokens > 0 and (
            (not seq.ignore_eos and seq.last_token in self.eos_token_ids)
            or seq.num_completion_tokens >= seq.max_tokens
            or seq.num_tokens >= self.max_model_len
        )

    def needs_verify(self, seq: Sequence) -> bool:
        if seq.num_verified_tokens >= seq.num_tokens:
            return False
        window_full = seq.num_tokens > seq.verify_start(self.verify_window) + self.verify_window
        kv_stale = seq.num_cached_tokens < seq.num_tokens - 1    # e.g. resumed after preemption
        return window_full or kv_stale or self.pending_finish(seq)

    def schedule(self) -> tuple[list[Sequence], str]:
        # prefer prefill workloads
        seqs = self._schedule_prefill()
        if seqs:
            return seqs, "prefill"

        if self.enable_determinism:
            seqs = [seq for seq in self.running if self.needs_verify(seq)][:self.verify_batch_size]
            if seqs:
                for seq in seqs:
                    seq.is_prefill = True    # verify serializes like prefill (tp workers need the token window)
                return seqs, "verify"

        seqs = self._schedule_decode()
        return seqs, "decode" if seqs else "idle"

    def _schedule_prefill(self) -> list[Sequence]:
        scheduled_seqs = []
        num_batched_tokens = 0

        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]

            remaining = self.max_num_batched_tokens - num_batched_tokens  # calculate remaining token budget
            if remaining == 0:
                break

            if not seq.block_table:  # empty block table means sequence is not in the kv cache yet
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:  # if no space to process then break
                    break
                num_cached_tokens = num_cached_blocks * self.block_size
            else:
                num_cached_tokens = seq.num_cached_tokens
            num_tokens = self.prefill_target(seq) - num_cached_tokens    # remaining tokens we have to process

            if self.enable_determinism:
                # one seq per step, chunked at block boundaries: every chunk's shape depends only on the
                # prompt, so prompt kv (and prefix-cache hits on it) are bit-identical under any load
                num_tokens = min(num_tokens, self.block_size - num_cached_tokens % self.block_size)
            elif remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break

            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)

            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == self.prefill_target(seq):
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)
            if self.enable_determinism:
                break
        return scheduled_seqs

    def _schedule_decode(self) -> list[Sequence]:
        scheduled_seqs = []
        deferred = []    # running seqs that must not draft further until verified
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            if self.enable_determinism and (self.needs_verify(seq) or self.pending_finish(seq)):
                deferred.append(seq)
                continue
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
        self.running.extendleft(reversed(scheduled_seqs + deferred))
        return scheduled_seqs

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def _finish(self, seq: Sequence):
        seq.status = SequenceStatus.FINISHED
        self.block_manager.deallocate(seq)
        self.running.remove(seq)

    def _maybe_finish(self, seq: Sequence):
        # a finish can only be emitted once the whole sequence is verified
        if self.pending_finish(seq) and seq.num_verified_tokens >= seq.num_tokens:
            self._finish(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        # this is called after the model generates one new token per scheduled sequence
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            if is_prefill:
                seq.num_cached_tokens += seq.num_scheduled_tokens
                seq.num_scheduled_tokens = 0
                if seq.num_cached_tokens < self.prefill_target(seq):
                    continue
                if self.enable_determinism and seq.num_completion_tokens > 0:
                    # resumed after preemption: the canonical prefill reproduces the first generated token.
                    # every later token is re-verified, which also rebuilds its kv
                    if token_id != seq[seq.num_prompt_tokens]:
                        seq.rollback(seq.num_prompt_tokens, token_id)
                        self.block_manager.rollback(seq)
                else:
                    seq.append_token(token_id)
                seq.num_verified_tokens = seq.num_prompt_tokens + 1 if self.enable_determinism else seq.num_tokens
            else:
                seq.num_cached_tokens = seq.num_tokens    # kv now covers every token before the new one
                seq.num_scheduled_tokens = 0
                seq.append_token(token_id)
                if not self.enable_determinism:
                    seq.num_verified_tokens = seq.num_tokens
            self._maybe_finish(seq)

    def postprocess_verify(self, seqs: list[Sequence], rows: list[list[int]]):
        W = self.verify_window
        for seq, row in zip(seqs, rows):
            # row[j] is the canonical token at position start + j + 1
            start = seq.verify_start(W)
            written = min(seq.num_tokens, start + W, len(seq.block_table) * self.block_size)
            seq.num_cached_tokens = max(seq.num_cached_tokens, written)
            last = min(seq.num_tokens - 1, start + W)    # last draft position this window can check
            for pos in range(seq.num_verified_tokens, last + 1):
                if seq[pos] != row[pos - start - 1]:
                    # keep the matched prefix, take the verifier's token at the mismatch, drop the rest
                    seq.rollback(pos, row[pos - start - 1])
                    self.block_manager.rollback(seq)
                    break
            else:
                seq.num_verified_tokens = last + 1
            # a finish held back earlier can complete once fully verified
            self._maybe_finish(seq)
