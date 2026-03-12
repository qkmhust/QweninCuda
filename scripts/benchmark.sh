#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <weights.qmini> <model_dir> [engine_path]"
  exit 1
fi

WEIGHTS="$1"
MODEL_DIR="$2"
ENGINE="${3:-./build/qwen_minimal}"

python3 scripts/run_chat.py --engine "$ENGINE" --weights "$WEIGHTS" --model-dir "$MODEL_DIR" --prompt "你好，你是谁？" --max-new-tokens 48 --temperature 0.72 --top-k 60 --top-p 0.88 --min-p 0.06 --temp-decay 0.97 --greedy-after 24 --repetition-penalty 1.18
python3 scripts/run_chat.py --engine "$ENGINE" --weights "$WEIGHTS" --model-dir "$MODEL_DIR" --prompt "请用三句话解释什么是CUDA。" --max-new-tokens 96 --temperature 0.72 --top-k 60 --top-p 0.88 --min-p 0.06 --temp-decay 0.97 --greedy-after 24 --repetition-penalty 1.18
python3 scripts/run_chat.py --engine "$ENGINE" --weights "$WEIGHTS" --model-dir "$MODEL_DIR" --prompt "北京是中国的首都吗？请简短回答。" --max-new-tokens 32 --temperature 0.72 --top-k 60 --top-p 0.88 --min-p 0.06 --temp-decay 0.97 --greedy-after 24 --repetition-penalty 1.18
