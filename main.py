import argparse
from transformers import AutoTokenizer

from llm_engine import LLMEngine
from sequence import SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Qwen3-0.6B", help="local HF model directory or hub id")
    parser.add_argument("--enforce-eager", action="store_true", help="skip CUDA graph capture")
    parser.add_argument("--no-determinism", action="store_true", help="disable the verifier (plain fast decode)")
    args = parser.parse_args()

    prompts = [
        "The capital of France is",
        "The three primary colors are",
        "To make a peanut butter sandwich, first",
        "1 + 1 =",
    ]

    engine = LLMEngine(args.model, enforce_eager=args.enforce_eager, enable_determinism=not args.no_determinism)
    tokenizer = AutoTokenizer.from_pretrained(engine.config.model)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for p in prompts
    ]

    outputs = engine.generate(prompts, SamplingParams(max_tokens=64))

    for prompt, output in zip(prompts, outputs):
        print(f"prompt: {prompt!r}")
        print(f"completion: {output['text']!r}")
        print("-" * 40)
    print(engine.last_metrics)


if __name__ == "__main__":
    main()
