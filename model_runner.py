import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from config import Config
from sequence import Sequence
from q3 import Qwen3ForCausalLM
from sampler import Sampler
from context import set_context, get_context, reset_context
from loader import load_model

SHM_SIZE = 2**26    # rank 0 -> worker call payloads (pickled seqs); only used with tensor parallelism

# decode compiles RMSNorm/RoPE/SiLU once per captured batch shape; don't let dynamo silently fall back to eager
for _limit in ("recompile_limit", "cache_size_limit"):
    if hasattr(torch._dynamo.config, _limit):
        setattr(torch._dynamo.config, _limit, 64)


class ModelRunner:
    def __init__(self, config: Config, rank: int, event : Event | list[Event]):
        self.config = config
        self.hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # initialize pytorch gpu communication backend
        dist.init_process_group("nccl", f"tcp://localhost:{config.dist_port}", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.hf_config.dtype)
        torch.set_default_device("cuda")

        self.model = Qwen3ForCausalLM(self.hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name=config.shm_name, create=True, size=SHM_SIZE)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name=config.shm_name)
                self.loop()


    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        assert n + 4 <= self.shm.size, f"{method_name} payload is {n} bytes, larger than the {self.shm.size}-byte shared memory buffer"
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        if self.config.enable_determinism:
            self.warmup_verify()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def warmup_verify(self):
        # the verifier materializes and samples logits for all B*W rows; make sure that is in the peak
        W, B = self.config.verify_window, self.config.verify_batch_size
        n = W * B
        input_ids = torch.zeros(n, dtype=torch.int64)
        positions = torch.arange(W, dtype=torch.int64).repeat(B)
        cu_seqlens = torch.arange(0, n + 1, W, dtype=torch.int32)
        set_context(True, cu_seqlens, cu_seqlens, W, W, is_verify=True, canonical=True)
        logits = self.run_model(input_ids, positions, True)
        reset_context()
        if self.rank == 0:
            self.sampler(logits, [1.0] * n, [0.9] * n, [0] * n, [0] * n)

    def allocate_kv_cache(self):
        # every transformer layer has its own kv cache

        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()  # ask gpu for amount of free memory and total memory
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        if config.max_num_kvcache_blocks > 0:
            config.num_kvcache_blocks = min(config.num_kvcache_blocks, config.max_num_kvcache_blocks)
        assert config.num_kvcache_blocks > 0
        # zeros, not empty: verifier padding rows and dummy slots read slots that were never written, and bf16
        # garbage (nan/inf) there could leak into real rows through 0 * nan
        self.kv_cache = torch.zeros(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence], width: int | None = None):
        width = width or max(len(seq.block_table) for seq in seqs)
        # pad with a valid page (0), not -1: some FA3 paged loops dereference page-table entries before masking
        block_tables = [seq.block_table + [0] * (width - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        canonical = self.config.enable_determinism
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        context_lens = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            context_lens.append(seqlen_k)    # kv valid for positions 0..end-1 once store_kv_cache runs
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            # canonical chunks use a fixed page-table width, so kernel heuristics can't see how many
            # blocks the seq happens to own (e.g. more after a preemption)
            block_tables = self.prepare_block_tables(seqs, self.config.max_num_blocks_per_seq if canonical else None)
            context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        else:
            context_lens = None
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, canonical=canonical)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            pos = len(seq) - 1
            input_ids.append(seq.last_token)
            positions.append(pos)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[pos // self.block_size] * self.block_size + pos % self.block_size)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_verify(self, seqs: list[Sequence]):
        # the verifier always runs B slots x W tokens with a fixed page-table width, so every kernel sees
        # the same shapes no matter how many seqs are being verified or what else is running.
        #   slot i (real): inputs at positions start..start+W-1; row j recomputes the token at start+j+1.
        #                  inputs past the end of the seq are padding: token 0, no kv write. causal attention
        #                  means real rows never read them. the newest token may not have a block yet (decode
        #                  allocates lazily); its kv is skipped, which is safe because a window containing it
        #                  is never complete and will run again.
        #   slot i (empty): a dummy window that writes nothing and reads block 0.
        W, B = self.config.verify_window, self.config.verify_batch_size
        width = self.config.max_num_blocks_per_seq
        max_position = self.config.max_model_len - 1
        assert len(seqs) <= B
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        block_tables = []
        for i in range(B):
            if i < len(seqs):
                seq = seqs[i]
                start = seq.verify_start(W)
                for pos in range(start, start + W):
                    if pos < seq.num_tokens:
                        input_ids.append(seq[pos])
                        positions.append(pos)
                        if pos // self.block_size < len(seq.block_table):
                            slot_mapping.append(seq.block_table[pos // self.block_size] * self.block_size + pos % self.block_size)
                        else:
                            slot_mapping.append(-1)
                    else:
                        input_ids.append(0)
                        positions.append(min(pos, max_position))
                        slot_mapping.append(-1)
                context_lens.append(start + W)
                block_tables.append(seq.block_table + [0] * (width - len(seq.block_table)))
            else:
                input_ids.extend([0] * W)
                positions.extend(range(W))
                slot_mapping.extend([-1] * W)
                context_lens.append(W)
                block_tables.append([0] * width)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.arange(0, B * W + 1, W, dtype=torch.int32).pin_memory().cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q=cu_seqlens_q, max_seqlen_q=W, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables, is_verify=True, canonical=True)
        return input_ids, positions

    @torch.inference_mode()
    def run_verify(self, seqs: list[Sequence]) -> list[list[int]] | None:
        input_ids, positions = self.prepare_verify(seqs)
        logits = self.run_model(input_ids, positions, True)    # is_prefill=True: eager, no cuda graphs
        reset_context()
        if self.rank != 0:
            return None
        W, B = self.config.verify_window, self.config.verify_batch_size
        temperatures, top_ps, seeds, sample_positions = [], [], [], []
        for i in range(B):
            if i < len(seqs):
                seq = seqs[i]
                start = seq.verify_start(W)
                temperatures.extend([seq.temperature] * W)
                top_ps.extend([seq.top_p] * W)
                seeds.extend([seq.seed] * W)
                sample_positions.extend(range(start + 1, start + W + 1))
            else:
                temperatures.extend([0.0] * W)
                top_ps.extend([1.0] * W)
                seeds.extend([0] * W)
                sample_positions.extend([0] * W)
        tokens = self.sampler(logits, temperatures, top_ps, seeds, sample_positions)
        return [tokens[i * W : (i + 1) * W] for i in range(len(seqs))]

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = None
        if self.rank == 0:
            # the position of the token being sampled keys its sampling noise
            if is_prefill:
                sample_positions = [seq.num_cached_tokens + seq.num_scheduled_tokens for seq in seqs]
            else:
                sample_positions = [seq.num_tokens for seq in seqs]
            token_ids = self.sampler(
                logits,
                [seq.temperature for seq in seqs],
                [seq.top_p for seq in seqs],
                [seq.seed for seq in seqs],
                sample_positions,
            )
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
