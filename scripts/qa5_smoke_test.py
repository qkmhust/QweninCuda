#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path


QUESTIONS = [
    "你好，请用一句话介绍你自己。",
    "1加1等于几？请只回答结果。",
    "中国的首都是哪里？",
    "请把“人工智能”翻译成英文。",
    "请给我三条早睡建议，每条不超过10个字。",
]


def parse_response(text: str) -> str:
    marker = "=== response ==="
    idx = text.find(marker)
    if idx < 0:
        return "<未解析到回答>"
    return text[idx + len(marker):].strip()


def main() -> None:
    p = argparse.ArgumentParser(description="5个简单问题的本地推理冒烟测试")
    p.add_argument("--engine", default="./build/qwen_minimal")
    p.add_argument("--weights", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--output-report", default="weights/qa5_report.md")
    args = p.parse_args()

    lines = ["# QA5 Smoke Test Report", ""]

    for i, q in enumerate(QUESTIONS, start=1):
        cmd = [
            "python3",
            "scripts/run_chat.py",
            "--engine",
            args.engine,
            "--weights",
            args.weights,
            "--model-dir",
            args.model_dir,
            "--prompt",
            q,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--temperature",
            "0.55",
            "--top-k",
            "32",
            "--top-p",
            "0.78",
            "--min-p",
            "0.12",
            "--temp-decay",
            "0.94",
            "--greedy-after",
            "8",
            "--no-repeat-ngram-size",
            "3",
            "--presence-penalty",
            "0.2",
            "--frequency-penalty",
            "0.15",
            "--repetition-penalty",
            "1.3",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        raw = proc.stdout + "\n" + proc.stderr
        answer = parse_response(raw)

        print(f"[{i}] Q: {q}")
        print(f"[{i}] A: {answer}")
        print()

        lines.append(f"## Q{i}")
        lines.append(f"- 问题: {q}")
        lines.append(f"- 回答: {answer}")
        lines.append("")

    out = Path(args.output_report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"report_saved={out}")


if __name__ == "__main__":
    main()
