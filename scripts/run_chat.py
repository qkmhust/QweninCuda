#!/usr/bin/env python3
import argparse
import re
import subprocess
from pathlib import Path

from transformers import AutoTokenizer


def parse_generated_ids(output: str):
    m = re.search(r"generated_ids=([0-9,\-]+)", output)
    if not m:
        raise RuntimeError("Cannot find generated_ids in engine output")
    return [int(x) for x in m.group(1).split(",") if x]


def main():
    parser = argparse.ArgumentParser(description="Local chat wrapper for qwen_minimal")
    parser.add_argument("--engine", default="./build/qwen_minimal")
    parser.add_argument("--weights", required=True, help="*.qmini")
    parser.add_argument("--model-dir", required=True, help="local qwen model dir for tokenizer")
    parser.add_argument("--prompt", default="你好，请用一句话介绍你自己。")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--temp-decay", type=float, default=1.0)
    parser.add_argument("--greedy-after", type=int, default=-1)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Local model directory not found: {model_dir}. "
            "Please pass an existing local path (absolute path recommended)."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True
    )

    input_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    input_ids_str = ",".join(str(x) for x in input_ids)

    cmd = [
        args.engine,
        "--model",
        args.weights,
        "--input-ids",
        input_ids_str,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--eos-id",
        str(tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 151645),
        "--temperature",
        str(args.temperature),
        "--top-k",
        str(args.top_k),
        "--top-p",
        str(args.top_p),
        "--min-p",
        str(args.min_p),
        "--temp-decay",
        str(args.temp_decay),
        "--greedy-after",
        str(args.greedy_after),
        "--repetition-penalty",
        str(args.repetition_penalty),
    ]

    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    out = proc.stdout + "\n" + proc.stderr
    print(out)

    gen_ids = parse_generated_ids(out)
    new_part = gen_ids[len(input_ids):]
    text = tokenizer.decode(new_part, skip_special_tokens=True)

    print("=== prompt ===")
    print(args.prompt)
    print("=== response ===")
    print(text)


if __name__ == "__main__":
    main()
