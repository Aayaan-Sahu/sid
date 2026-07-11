import argparse

from llm_engine import LLMEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Qwen3-0.6B", help="path to a local HF model directory")
    parser.add_argument("--enforce-eager", action="store_true", help="skip CUDA graph capture")
    args = parser.parse_args()

    prompts = [
        "The capital of France is",
        "The three primary colors are",
        "To make a peanut butter sandwich, first",
        "1 + 1 =",
    ]

    engine = LLMEngine(args.model, enforce_eager=args.enforce_eager)
    outputs = engine.generate(prompts)

    for prompt, output in zip(prompts, outputs):
        print(f"prompt: {prompt!r}")
        print(f"completion: {output['text']!r}")
        print("-" * 40)


if __name__ == "__main__":
    main()
