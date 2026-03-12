#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda/bin/python}"
ENGINE_PATH="${ENGINE_PATH:-$ROOT_DIR/build/qwen_minimal}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/ms_models/Qwen/Qwen3___5-4B}"
WEIGHTS_PATH="${WEIGHTS_PATH:-$ROOT_DIR/weights/qwen3_5_4b_strict_seq512.qmini}"

if [[ ! -x "$ENGINE_PATH" ]]; then
  echo "[INFO] engine not found, building..."
  cmake -S "$ROOT_DIR" -B "$ROOT_DIR/build" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$ROOT_DIR/build" -j
fi

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "[ERROR] model dir not found: $MODEL_DIR"
  echo "请先下载模型到该目录，或通过 MODEL_DIR 指定正确路径。"
  exit 1
fi

if [[ ! -f "$WEIGHTS_PATH" ]]; then
  echo "[INFO] qmini weights not found, converting..."
  mkdir -p "$ROOT_DIR/weights"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/convert_qwen3_to_qmini.py" \
    --model-dir "$MODEL_DIR" \
    --output "$WEIGHTS_PATH" \
    --max-seq-len 512
fi

echo "[INFO] starting interactive chat..."
"$PYTHON_BIN" "$ROOT_DIR/scripts/run_chat.py" \
  --interactive \
  --engine "$ENGINE_PATH" \
  --weights "$WEIGHTS_PATH" \
  --model-dir "$MODEL_DIR" \
  --max-new-tokens 64 \
  --temperature 0.35 \
  --top-k 24 \
  --top-p 0.68 \
  --min-p 0.08 \
  --temp-decay 0.95 \
  --greedy-after 0 \
  --no-repeat-ngram-size 4 \
  --history-turns 4 \
  --max-input-tokens 768
