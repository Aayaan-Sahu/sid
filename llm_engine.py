import atexit
from collections import Counter
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch
import torch.multiprocessing as mp

from config import Config
from sequence import Sequence, SamplingParams
from scheduler import Scheduler


class LLMEngine:
    def __init__(self, model, **kwargs):
        from model_runner import ModelRunner    # imported here so the engine module loads without cuda kernels

        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config

        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        self.scheduler = Scheduler(config)
        # every key exists up front so /metrics can read the dict while the engine thread updates it
        self.stats = Counter({key: 0 for key in (
            "steps_prefill", "steps_verify", "steps_decode", "steps_idle",
            "prefill_tokens", "decode_tokens", "verify_windows", "rollbacks", "finished",
        )})
        self.last_metrics = None
        self._exited = False
        atexit.register(self.exit)

    def exit(self):
        if self._exited:
            return
        self._exited = True
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams | None = None) -> Sequence:
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        config = self.config
        if len(prompt) >= config.max_model_len:
            raise ValueError(f"prompt has {len(prompt)} tokens but max_model_len is {config.max_model_len}")
        seq = Sequence(prompt, sampling_params)
        seq.max_tokens = min(seq.max_tokens, config.max_model_len - len(prompt))
        num_blocks = (len(prompt) + seq.max_tokens + config.kvcache_block_size - 1) // config.kvcache_block_size
        if num_blocks > config.num_kvcache_blocks:
            raise ValueError(f"request needs {num_blocks} kv cache blocks but the engine only has {config.num_kvcache_blocks}")
        self.scheduler.add(seq)
        return seq

    def abort(self, seq: Sequence) -> bool:
        return self.scheduler.abort(seq)

    def step(self) -> tuple[list[Sequence], str]:
        seqs, mode = self.scheduler.schedule()
        self.stats[f"steps_{mode}"] += 1
        if mode == "idle":
            return seqs, mode
        if mode == "verify":
            rollbacks = sum(seq.num_rollbacks for seq in seqs)
            rows = self.model_runner.call("run_verify", seqs)
            self.scheduler.postprocess_verify(seqs, rows)
            self.stats["verify_windows"] += len(seqs)
            self.stats["rollbacks"] += sum(seq.num_rollbacks for seq in seqs) - rollbacks
        else:
            is_prefill = mode == "prefill"
            if is_prefill:
                self.stats["prefill_tokens"] += sum(seq.num_scheduled_tokens for seq in seqs)
            else:
                self.stats["decode_tokens"] += len(seqs)
            token_ids = self.model_runner.call("run", seqs, is_prefill)
            self.scheduler.postprocess(seqs, token_ids, is_prefill)
        self.stats["finished"] += sum(seq.is_finished for seq in seqs)
        return seqs, mode

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams] | None = None,
        use_tqdm: bool = True,
    ) -> list[dict]:
        assert prompts, "prompts must not be empty"
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)

        seqs = [self.add_request(prompt, sp) for prompt, sp in zip(prompts, sampling_params)]
        stats_before = self.stats.copy()

        prefill_seconds = 0.
        num_finished = 0
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        while not self.is_finished():
            t = perf_counter()
            _, mode = self.step()
            if mode == "prefill":
                prefill_seconds += perf_counter() - t
            if use_tqdm:
                output_tokens = sum(max(seq.num_verified_tokens - seq.num_prompt_tokens, 0) for seq in seqs)
                pbar.set_postfix({"Output": f"{output_tokens / (perf_counter() - started):.0f} tok/s"})
            finished = sum(seq.is_finished for seq in seqs)
            pbar.update(finished - num_finished)
            num_finished = finished

        torch.cuda.synchronize()
        elapsed = perf_counter() - started
        output_tokens = sum(seq.num_completion_tokens for seq in seqs)
        prefill_tokens = self.stats["prefill_tokens"] - stats_before["prefill_tokens"]
        self.last_metrics = {
            "requests": len(seqs),
            "prompt_tokens": sum(seq.num_prompt_tokens for seq in seqs),
            "output_tokens": output_tokens,
            "elapsed_seconds": elapsed,
            "output_tokens_per_second": output_tokens / elapsed,
            "prefill_tokens_per_second": prefill_tokens / prefill_seconds if prefill_seconds else 0.,
            "rollbacks": self.stats["rollbacks"] - stats_before["rollbacks"],
            "verify_windows": self.stats["verify_windows"] - stats_before["verify_windows"],
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "model": self.config.model,
            "enable_determinism": self.config.enable_determinism,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "gpu": torch.cuda.get_device_name(),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
        }
        pbar.close()
        return [
            {"text": self.tokenizer.decode(seq.completion_token_ids, skip_special_tokens=True), "token_ids": seq.completion_token_ids}
            for seq in seqs
        ]
