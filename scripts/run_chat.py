#!/usr/bin/env python3
import argparse
import re
import subprocess
import time
from pathlib import Path

from transformers import AutoTokenizer


def parse_generated_ids(output: str):
    m = re.search(r"generated_ids=([0-9,\-]+)", output)
    if not m:
        raise RuntimeError("Cannot find generated_ids in engine output")
    return [int(x) for x in m.group(1).split(",") if x]


def parse_perf_metrics(output: str):
    m = re.search(r"elapsed_ms=([0-9.]+)\s+new_tokens=([0-9]+)\s+tok_per_s=([0-9.]+)", output)
    if not m:
        return None
    return {
        "elapsed_ms": float(m.group(1)),
        "new_tokens": int(m.group(2)),
        "tok_per_s": float(m.group(3)),
    }


def run_engine_once(args, tokenizer, input_ids: list[int], on_token_id=None):
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

    if args.stream:
        cmd.append("--stream-ids")

    if args.stream and on_token_id is not None:
        proc = subprocess.Popen(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )
        stdout_lines = []
        first_token_at = None
        assert proc.stdout is not None
        for line in proc.stdout:
            stdout_lines.append(line)
            m = re.match(r"^stream_token_id=([-0-9]+)\s*$", line.strip())
            if m:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                on_token_id(int(m.group(1)))

        stderr_text = ""
        if proc.stderr is not None:
            stderr_text = proc.stderr.read()
        code = proc.wait()
        out = "".join(stdout_lines) + "\n" + stderr_text
        if code != 0:
            raise subprocess.CalledProcessError(code, cmd, output=out)
        gen_ids = parse_generated_ids(out)
        return out, gen_ids, first_token_at

    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    out = proc.stdout + "\n" + proc.stderr
    gen_ids = parse_generated_ids(out)
    return out, gen_ids, None


def remove_think_block(text: str) -> str:
    # 仅做展示层清洗，不影响模型真实输出 token。
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    # 部分模型会输出未闭合的 <think>，此时直接截断到标签前。
    text = re.sub(r"<think>[\s\S]*$", "", text, flags=re.IGNORECASE)
    return text.strip()


def trim_generated_ids(tokenizer, new_ids: list[int]) -> list[int]:
    stop_ids = set()
    for tok in [tokenizer.eos_token, "<|im_end|>", "<|im_start|>", "<|endoftext|>"]:
        if tok is None:
            continue
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid >= 0:
            stop_ids.add(tid)

    if not stop_ids:
        return new_ids

    for i, tid in enumerate(new_ids):
        if tid in stop_ids:
            return new_ids[:i]
    return new_ids


def truncate_at_role_boundary(text: str) -> str:
    # 某些情况下模型会继续生成下一轮标签（assistant/user/用户/助手），在展示层截断。
    patterns = [
        r"\n\s*(assistant|user|system)\s*[:：]",
        r"\n\s*(助手|用户|系统)\s*[:：]",
    ]
    cut = len(text)
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            cut = min(cut, m.start())

    if cut != len(text):
        text = text[:cut]
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


def build_input_ids_from_messages(tokenizer, messages: list[dict]) -> list[int]:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            if isinstance(ids, list) and ids:
                return ids
        except Exception:
            pass

    # 回退：把多轮消息拼成纯文本上下文。
    lines = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    lines.append("assistant:")
    return tokenizer.encode("\n".join(lines), add_special_tokens=False)


def display_response(tokenizer, input_ids: list[int], gen_ids: list[int], keep_think: bool) -> str:
    new_part = gen_ids[len(input_ids):]
    new_part = trim_generated_ids(tokenizer, new_part)
    raw_text = tokenizer.decode(new_part, skip_special_tokens=True)
    text = truncate_at_role_boundary(raw_text)
    if not keep_think:
        text = remove_think_block(text)
        if not text.strip():
            text = "<模型仅输出了思维片段，建议改用更低 temperature 或设置 --greedy-after 0。>"
    return text


def run_single_turn(args, tokenizer):
    input_ids = build_input_ids(tokenizer, args.prompt, args.system_prompt)
    print("=== prompt ===")
    print(args.prompt)
    if args.stream:
        print("=== response(stream) ===")

    stream_ids: list[int] = []
    stream_text = ""
    t_begin = time.perf_counter()

    def on_token_id(tid: int):
        nonlocal stream_text
        stream_ids.append(tid)
        # 增量显示：每来一个 token id 就重新 decode 并输出新增片段。
        text = tokenizer.decode(stream_ids, skip_special_tokens=True)
        if text.startswith(stream_text):
            delta = text[len(stream_text) :]
        else:
            delta = text
        if delta:
            print(delta, end="", flush=True)
        stream_text = text

    out, gen_ids, first_token_at = run_engine_once(
        args, tokenizer, input_ids, on_token_id=on_token_id if args.stream else None
    )

    if args.show_engine_output:
        print(out)

    text = display_response(tokenizer, input_ids, gen_ids, args.keep_think)
    metrics = parse_perf_metrics(out)
    if args.stream:
        # 流式输出内容已在回调中逐步打印，这里补一个换行保证版式稳定。
        print()
    else:
        print("=== response ===")
        print(text)
    if metrics is not None:
        print(
            f"=== perf === elapsed_ms={metrics['elapsed_ms']:.2f} "
            f"new_tokens={metrics['new_tokens']} tok_per_s={metrics['tok_per_s']:.2f}"
        )
    if args.stream and first_token_at is not None:
        ttft_ms = (first_token_at - t_begin) * 1000.0
        print(f"=== stream === ttft_ms={ttft_ms:.2f}")


def run_interactive(args, tokenizer):
    print("=== QweninCuda Interactive Chat ===")
    print("输入 /exit 退出，输入 /clear 清空历史。")
    history: list[tuple[str, str]] = []
    turn = 0

    while True:
        if args.max_turns > 0 and turn >= args.max_turns:
            print(f"达到最大轮数 {args.max_turns}，会话结束。")
            break

        try:
            user_prompt = input("\n你: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n会话结束。")
            break

        if not user_prompt:
            continue
        if user_prompt.lower() in {"/exit", "exit", "quit", "/quit"}:
            print("会话结束。")
            break
        if user_prompt.lower() == "/clear":
            history.clear()
            print("历史已清空。")
            continue

        messages = [{"role": "system", "content": args.system_prompt}]
        if args.history_turns > 0:
            sliced_history = history[-args.history_turns :]
        else:
            sliced_history = history

        while True:
            messages = [{"role": "system", "content": args.system_prompt}]
            for u, a in sliced_history:
                messages.append({"role": "user", "content": u})
                messages.append({"role": "assistant", "content": a})
            messages.append({"role": "user", "content": user_prompt})

            input_ids = build_input_ids_from_messages(tokenizer, messages)

            if args.max_input_tokens <= 0:
                break
            if len(input_ids) <= args.max_input_tokens:
                break
            if not sliced_history:
                # 保留最近 token，避免超长输入拖慢推理。
                input_ids = input_ids[-args.max_input_tokens :]
                break
            sliced_history = sliced_history[1:]

        stream_ids: list[int] = []
        stream_text = ""

        def on_token_id(tid: int):
            nonlocal stream_text
            stream_ids.append(tid)
            text_now = tokenizer.decode(stream_ids, skip_special_tokens=True)
            if text_now.startswith(stream_text):
                delta = text_now[len(stream_text) :]
            else:
                delta = text_now
            if delta:
                print(delta, end="", flush=True)
            stream_text = text_now

        if args.stream:
            print("助手: ", end="", flush=True)
        out, gen_ids, _ = run_engine_once(
            args, tokenizer, input_ids, on_token_id=on_token_id if args.stream else None
        )

        if args.show_engine_output:
            print(out)

        answer = display_response(tokenizer, input_ids, gen_ids, args.keep_think)
        metrics = parse_perf_metrics(out)
        if args.stream:
            print()
            if metrics is not None:
                print(f"[perf] {metrics['tok_per_s']:.2f} tok/s")
            else:
                print("[perf] n/a")
        else:
            if metrics is not None:
                print(f"助手 [{metrics['tok_per_s']:.2f} tok/s]: {answer}")
            else:
                print(f"助手: {answer}")
        history.append((user_prompt, answer))
        turn += 1


def main():
    parser = argparse.ArgumentParser(description="Local chat wrapper for qwen_minimal")
    parser.add_argument("--engine", default="./build/qwen_minimal")
    parser.add_argument("--weights", required=True, help="*.qmini")
    parser.add_argument("--model-dir", required=True, help="local qwen model dir for tokenizer")
    parser.add_argument("--prompt", default="你好，请用一句话介绍你自己。")
    parser.add_argument("--interactive", action="store_true", help="Enable multi-turn interactive chat")
    parser.add_argument("--max-turns", type=int, default=0, help="0 means unlimited turns")
    parser.add_argument("--history-turns", type=int, default=6, help="How many history turns to keep in context")
    parser.add_argument("--max-input-tokens", type=int, default=1024, help="Hard cap for input context tokens in interactive mode")
    parser.add_argument("--show-engine-output", action="store_true", help="Print full engine raw output")
    parser.add_argument("--stream", action="store_true", help="Stream tokens while engine is generating")
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
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
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

    if args.interactive:
        run_interactive(args, tokenizer)
    else:
        run_single_turn(args, tokenizer)


if __name__ == "__main__":
    main()
