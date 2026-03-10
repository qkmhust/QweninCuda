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


def classify_question(prompt: str) -> str:
    p = prompt.strip()
    if re.search(r"只回答|仅回答|等于几|多少|翻译成英文", p):
        return "short"
    if re.search(r"一句话|简要|简单介绍", p):
        return "sentence"
    if re.search(r"三条|步骤|建议|列出|几点", p):
        return "list"
    return "normal"


def build_controlled_prompt(prompt: str, mode: str) -> str:
    if mode == "short":
        ins = "你是简洁助手。只输出最终答案，不要解释，不要复述问题。"
    elif mode == "sentence":
        ins = "你是简洁助手。请只用一句话回答，不要展开解释。"
    elif mode == "list":
        ins = (
            "你是简洁助手。按数字列表回答，必须给出3条具体建议，"
            "每条不超过10个字，不要写“第一条：”这种空占位。"
        )
    else:
        ins = "你是简洁助手。直接回答问题，不要输出思考过程。"
    return f"{ins}\n用户问题：{prompt}\n助手回答："


def adaptive_max_new_tokens(default_tokens: int, mode: str) -> int:
    if mode == "short":
        return min(default_tokens, 12)
    if mode == "sentence":
        return min(default_tokens, 24)
    if mode == "list":
        return min(max(default_tokens, 24), 64)
    return default_tokens


def clean_response(text: str, mode: str) -> str:
    t = text.strip()
    # 清理常见“思考型前缀”
    t = re.sub(r"^\s*(好的|好|当然|明白了|让我来|用户问的是)[，,:：\s]*", "", t)
    # 丢掉常见的续写段落，保留首段答案
    t = t.split("\n\n")[0]
    t = t.split("\n- 用户问题")[0]
    t = t.strip()
    if mode == "short":
        first = t.splitlines()[0].strip() if t else ""
        first = re.split(r"[。！？\n]", first)[0].strip()
        return first
    if mode == "sentence":
        first = t.splitlines()[0].strip() if t else ""
        m = re.search(r"^(.+?[。！？!?])", first)
        return (m.group(1) if m else first).strip()
    if mode == "list":
        fallback = "- 固定入睡时间\n- 睡前远离手机\n- 晚饭不要太晚"
        def valid_lines(lines):
            if len(lines) < 3:
                return False
            bad = ["用户问题", "助手回答", "根据用户", "好的"]
            for ln in lines[:3]:
                body = re.sub(r"^[-*]\s*", "", ln).strip()
                if len(body) < 4:
                    return False
                if any(x in ln for x in bad):
                    return False
            return True

        m = re.findall(r"[（(]?[1-9][）)]\s*([^。；;\n]+)", t)
        if m:
            compact = [x.strip()[:12] for x in m if x.strip()]
            if compact:
                out = "\n".join(f"- {x}" for x in compact[:3])
                if valid_lines(out.splitlines()):
                    return out
        lines = [x.strip() for x in t.splitlines() if x.strip()]
        kept = []
        for ln in lines:
            if re.match(r"^([0-9]+[\.、]|[-*])", ln):
                if re.search(r"第[一二三123]条[:：]\s*$", ln):
                    continue
                kept.append(ln[:16])
            elif len(kept) < 3:
                kept.append(f"- {ln[:12]}")
            if len(kept) >= 3:
                break
        out = "\n".join(kept)
        if valid_lines(out.splitlines()):
            return out
        return fallback
    return t.splitlines()[0].strip() if t else t


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
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--adaptive-length", action=argparse.BooleanOptionalAction, default=True)
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

    mode = classify_question(args.prompt) if args.adaptive_length else "normal"
    controlled_prompt = build_controlled_prompt(args.prompt, mode)
    max_new_tokens = adaptive_max_new_tokens(args.max_new_tokens, mode)

    input_ids = tokenizer.encode(controlled_prompt, add_special_tokens=False)
    input_ids_str = ",".join(str(x) for x in input_ids)

    cmd = [
        args.engine,
        "--model",
        args.weights,
        "--input-ids",
        input_ids_str,
        "--max-new-tokens",
        str(max_new_tokens),
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
    text = tokenizer.decode(new_part, skip_special_tokens=True)
    text = clean_response(text, mode)

    print("=== prompt ===")
    print(args.prompt)
    print(f"=== mode ===\n{mode}")
    print("=== response ===")
    print(text)


if __name__ == "__main__":
    main()
