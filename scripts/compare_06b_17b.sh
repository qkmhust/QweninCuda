#!/usr/bin/env bash
set -euo pipefail

PROMPT="${1:-你是谁？人工智能是什么？}"

ENGINE="./build/qwen_minimal"
W06="/root/qwen3-cuda-minimal/weights/qwen3_0.6b.qmini"
M06="/root/qwen3-cuda-minimal/ms_models/Qwen/Qwen3-0.6B"
W17="/root/qwen3-cuda-minimal/weights/qwen3_1.7b_seq512.qmini"
M17="/root/qwen3-cuda-minimal/ms_models/Qwen/Qwen3-1.7B"

echo "========== 0.6B =========="
python3 scripts/run_chat.py \
  --engine "$ENGINE" \
  --weights "$W06" \
  --model-dir "$M06" \
  --prompt "$PROMPT" \
  --max-new-tokens 64 \
  --temperature 0.72 \
  --top-k 60 \
  --top-p 0.88 \
  --min-p 0.06 \
  --temp-decay 0.97 \
  --greedy-after 24 \
  --repetition-penalty 1.18

echo
echo "========== 1.7B =========="
python3 scripts/run_chat.py \
  --engine "$ENGINE" \
  --weights "$W17" \
  --model-dir "$M17" \
  --prompt "$PROMPT" \
  --max-new-tokens 64 \
  --temperature 0.72 \
  --top-k 60 \
  --top-p 0.88 \
  --min-p 0.06 \
  --temp-decay 0.97 \
  --greedy-after 24 \
  --repetition-penalty 1.18
