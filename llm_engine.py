import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch
import torch.multiprocessing as mp

from config import Config
from sequence import Sequence
from scheduler import Scheduler
from model_runner import ModelRunner


class LLMEngine:
    def __init__(self, model, **kwargs):
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
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self.last_metrics = None
        atexit.register(self.exit)
    
    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()
    
    def add_request(self, prompt: str | list[int], max_tokens: int = 64, ignore_eos: bool = False):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, max_tokens=max_tokens, ignore_eos=ignore_eos)
        self.scheduler.add(seq)
        return seq

    def step(self):
        seqs, mode = self.scheduler.schedule()
        if mode == "verify":
            num_tokens = 0
            results = self.model_runner.call("run_verify", seqs)
            self.scheduler.postprocess_verify(seqs, results)
        else:
            is_prefill = mode == "prefill"
            num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
            token_ids = self.model_runner.call("run", seqs, is_prefill)
            self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        use_tqdm: bool = True,
        max_tokens: int = 64,
        ignore_eos: bool = False,
    ) -> list[dict]:
        assert prompts, "prompts must not be empty"
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)

        seqs = [self.add_request(prompt, max_tokens, ignore_eos) for prompt in prompts]

        outputs = {}
        prefill_tokens = 0
        prefill_seconds = 0.
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_tokens += num_tokens
                prefill_seconds += perf_counter() - t
            elapsed = perf_counter() - started
            output_tokens = sum(seq.num_completion_tokens for seq in seqs)
            if use_tqdm:
                pbar.set_postfix({
                    "Prefill": f"{prefill_tokens / prefill_seconds:.0f} tok/s" if prefill_seconds else "-",
                    "Output": f"{output_tokens / elapsed:.0f} tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)

        torch.cuda.synchronize()
        elapsed = perf_counter() - started
        output_tokens = sum(len(token_ids) for token_ids in outputs.values())
        self.last_metrics = {
            "requests": len(seqs),
            "prompt_tokens": sum(seq.num_prompt_tokens for seq in seqs),
            "output_tokens": output_tokens,
            "elapsed_seconds": elapsed,
            "output_tokens_per_second": output_tokens / elapsed,
            "prefill_tokens_per_second": prefill_tokens / prefill_seconds if prefill_seconds else 0.,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "model": self.config.model,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "gpu": torch.cuda.get_device_name(),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
        }
        if use_tqdm:
            pbar.set_postfix({
                "Prefill": f"{self.last_metrics['prefill_tokens_per_second']:.0f} tok/s",
                "Output": f"{self.last_metrics['output_tokens_per_second']:.0f} tok/s",
            })
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
