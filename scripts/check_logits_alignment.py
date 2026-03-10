#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TOPK_RE = re.compile(r"^step_topk\[(\d+)\]=(.*)$")
GEN_RE = re.compile(r"generated_ids=([0-9,]+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare HF vs qwen_minimal top-k logits step-by-step")
    p.add_argument("--engine", default="./build/qwen_minimal")
    p.add_argument("--weights", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--eos-id", type=int, default=151645)
    p.add_argument("--save-json", default="", help="Optional path to save detailed report JSON")
    return p.parse_args()


def parse_engine_topk(output: str) -> list[tuple[int, float]]:
    for line in output.splitlines():
        m = TOPK_RE.match(line.strip())
        if not m:
            continue
        payload = m.group(2).strip()
        if not payload:
            return []
        pairs = []
        for item in payload.split(","):
            tok, logit = item.split(":")
            pairs.append((int(tok), float(logit)))
        return pairs
    raise RuntimeError("No step_topk[...] line found in engine output")


def parse_engine_generated(output: str) -> list[int]:
    m = GEN_RE.search(output)
    if not m:
        raise RuntimeError("No generated_ids=... found in engine output")
    return [int(x) for x in m.group(1).split(",") if x]


def run_engine_step(args: argparse.Namespace, prefix_ids: list[int]) -> tuple[list[tuple[int, float]], int, str]:
    cmd = [
        args.engine,
        "--model",
        args.weights,
        "--input-ids",
        ",".join(str(x) for x in prefix_ids),
        "--max-new-tokens",
        "1",
        "--eos-id",
        str(args.eos_id),
        "--temperature",
        "0.0",
        "--top-k",
        str(max(args.top_k, 1)),
        "--top-p",
        "1.0",
        "--min-p",
        "0.0",
        "--temp-decay",
        "1.0",
        "--greedy-after",
        "0",
        "--repetition-penalty",
        "1.0",
        "--dump-topk",
        str(args.top_k),
        "--dump-steps",
        "1",
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    out = proc.stdout
    topk = parse_engine_topk(out)
    generated = parse_engine_generated(out)
    if len(generated) <= len(prefix_ids):
        raise RuntimeError("Engine did not generate next token")
    next_tok = generated[len(prefix_ids)]
    return topk, next_tok, out


def hf_topk(model, prefix_ids: list[int], top_k: int) -> tuple[list[tuple[int, float]], int]:
    x = torch.tensor([prefix_ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(x).logits[0, -1, :].float()
    values, indices = torch.topk(logits, k=top_k)
    top = [(int(indices[i]), float(values[i])) for i in range(top_k)]
    return top, top[0][0]


def token_text(tokenizer, tok_id: int) -> str:
    s = tokenizer.decode([tok_id], clean_up_tokenization_spaces=False)
    s = s.replace("\n", "\\n")
    return s


def main() -> None:
    args = parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model-dir not found: {model_dir}")

    print(f"[1/3] load tokenizer+hf model from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=True,
        device_map="cpu",
    )
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("prompt encodes to empty ids")

    prefix_ids = list(prompt_ids)
    report = []

    print(f"[2/3] compare first {args.steps} steps, top_k={args.top_k}")
    for step in range(args.steps):
        hf_k, hf_next = hf_topk(model, prefix_ids, args.top_k)
        eng_k, eng_next, _ = run_engine_step(args, prefix_ids)

        hf_ids = [x[0] for x in hf_k]
        eng_ids = [x[0] for x in eng_k]
        overlap = len(set(hf_ids) & set(eng_ids))

        hf_map = {tid: logit for tid, logit in hf_k}
        eng_map = {tid: logit for tid, logit in eng_k}
        shared = sorted(set(hf_map.keys()) & set(eng_map.keys()))
        mae = None
        max_abs = None
        if shared:
            diffs = [abs(hf_map[t] - eng_map[t]) for t in shared]
            mae = sum(diffs) / len(diffs)
            max_abs = max(diffs)

        hf_rank = {tid: i for i, tid in enumerate(hf_ids)}
        eng_rank = {tid: i for i, tid in enumerate(eng_ids)}
        rank_delta = None
        if hf_next in eng_rank:
            rank_delta = eng_rank[hf_next] - hf_rank[hf_next]

        item = {
            "step": step,
            "prefix_len": len(prefix_ids),
            "hf_next": hf_next,
            "eng_next": eng_next,
            "same_next": hf_next == eng_next,
            "topk_overlap": overlap,
            "topk_overlap_ratio": overlap / max(1, args.top_k),
            "hf_next_rank_in_engine": eng_rank.get(hf_next, None),
            "rank_delta_for_hf_next": rank_delta,
            "shared_logit_mae": mae,
            "shared_logit_max_abs": max_abs,
            "hf_topk": hf_k,
            "eng_topk": eng_k,
        }
        report.append(item)

        print(
            f"step={step} same_next={item['same_next']} overlap={overlap}/{args.top_k} "
            f"mae={mae if mae is not None else 'NA'} "
            f"hf_next={hf_next}({token_text(tokenizer, hf_next)}) "
            f"eng_next={eng_next}({token_text(tokenizer, eng_next)})"
        )

        # Teacher forcing on HF trajectory so each step compares the same prefix semantics.
        prefix_ids.append(hf_next)

    print("[3/3] summary")
    same = sum(1 for x in report if x["same_next"])
    avg_overlap = sum(x["topk_overlap_ratio"] for x in report) / max(1, len(report))
    mae_vals = [x["shared_logit_mae"] for x in report if x["shared_logit_mae"] is not None]
    avg_mae = (sum(mae_vals) / len(mae_vals)) if mae_vals else 0.0
    first_div = next((x["step"] for x in report if not x["same_next"]), None)
    print(f"same_next_steps={same}/{len(report)}")
    print(f"avg_topk_overlap_ratio={avg_overlap:.3f}")
    print(f"avg_shared_logit_mae={avg_mae:.6f}")
    print(f"first_next_token_divergence_step={first_div}")

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved={out}")


if __name__ == "__main__":
    main()
