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


def remove_think_block(text: str) -> str:
    # 仅做展示层清洗，不影响模型真实输出 token。
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    # 部分模型会输出未闭合的 <think>，此时直接截断到标签前。
    text = re.sub(r"<think>[\s\S]*$", "", text, flags=re.IGNORECASE)
    return text.strip()


def build_input_ids(tokenizer, user_prompt: str, system_prompt: str) -> list[int]:
    # 优先走 chat template，能显著提升指令跟随与回复格式稳定性。
    if hasattr(tokenizer, "apply_chat_template"):
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            ids = tokenizer.apply_chat_template(
                msgs,
                tokenize=True,
                add_generation_prompt=True,
            )
            if isinstance(ids, list) and ids:
                return ids
        except Exception:
            pass

    # 回退：无模板时退化为普通 encode。
    return tokenizer.encode(user_prompt, add_special_tokens=False)


def main():
    parser = argparse.ArgumentParser(description="Local chat wrapper for qwen_minimal")
    parser.add_argument("--engine", default="./build/qwen_minimal")
    parser.add_argument("--weights", required=True, help="*.qmini")
    parser.add_argument("--model-dir", required=True, help="local qwen model dir for tokenizer")
    parser.add_argument("--prompt", default="你好，请用一句话介绍你自己。")
    parser.add_argument(
        "--system-prompt",
        default=(
            "你是一个严谨、友好的中文助手。"
            "请直接给出最终答案，不输出推理过程，不输出<think>标签。"
            "回答尽量简洁、准确、可执行。"
        ),
    )
    parser.add_argument("--keep-think", action="store_true", help="Do not strip <think> block in displayed response")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--temp-decay", type=float, default=1.0)
    parser.add_argument("--greedy-after", type=int, default=-1)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
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

    input_ids = build_input_ids(tokenizer, args.prompt, args.system_prompt)
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
        "--no-repeat-ngram-size",
        str(args.no_repeat_ngram_size),
        "--presence-penalty",
        str(args.presence_penalty),
        "--frequency-penalty",
        str(args.frequency_penalty),
        "--repetition-penalty",
        str(args.repetition_penalty),
    ]

    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    out = proc.stdout + "\n" + proc.stderr
    print(out)

    gen_ids = parse_generated_ids(out)
    new_part = gen_ids[len(input_ids):]
    raw_text = tokenizer.decode(new_part, skip_special_tokens=True)
    text = raw_text
    if not args.keep_think:
        text = remove_think_block(text)
        if not text.strip():
            # 若模型只输出 think 段，给出可操作提示，避免展示污染文本。
            text = "<模型仅输出了思维片段，建议改用更低 temperature 或设置 --greedy-after 0。>"

    print("=== prompt ===")
    print(args.prompt)
    print("=== response ===")
    print(text)


if __name__ == "__main__":
    main()
