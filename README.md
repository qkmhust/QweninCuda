# Qwen3 CUDA Inference (C++20 + CUDA)

这个个项目标是: 在本地离线环境中，目本`C++20 + CUDA` 复现接近官方 地离线环境中的单 batch  `主C+，并把关键实现细节暴露出来+D便学习和二次优化h 推理主流程，并把关键实现细节暴露出来，方便学习和二次优化。

## 1. 当前能力概览

- 支持模型: Qwen3-0.6B, Qwen3-1.7B (本地 ModelScope 下载)
- 推理后端: C++20 + CUDA + cuBLAS
# Qwen3 CUDA Inference (C++20 + CUDA)

这个项目是一个“可读、可改、可验证”的 Qwen3 本地推理学习工程。

核心目标:
- 用 C++20 + CUDA 手写最关键推理链路
- 尽量对齐官方实现语义（尤其是 RoPE / GQA / KV cache）
- 给出可复现的质量对齐流程（HF logits 对齐 + QA 冒烟测试）

## 1. 当前特性

- 模型: Qwen3-0.6B / Qwen3-1.7B（本地 ModelScope）
- 后端: `CUDA + cuBLAS`
- 核心算子: `RMSNorm / q_norm-k_norm / RoPE / FlashAttention / PagedAttention / SwiGLU`
- 采样: `temperature, top-k, top-p, min-p, temp-decay, penalties, no-repeat-ngram`
- 调试: C++ 引擎支持 `--dump-topk --dump-steps` 输出逐步 logits

## 2. 与官方对齐说明

已对齐:
- `Q/K/V/O + MLP + lm_head` 都使用 cuBLAS
- GQA (`num_heads / num_kv_heads`)
- `q_norm / k_norm`
- KV cache 自回归
- RoPE 已按官方 `rotate_half` 语义修复（前半维与后半维配对）
- 手写 FlashAttention / PagedAttention（online softmax）

暂未覆盖:
- continuous batching
- tensor/pipeline parallel
- 量化推理（INT8/FP8）

## 3. 环境准备

要求:
- Linux + NVIDIA GPU
- CUDA 12.x
- CMake >= 3.22
- g++ >= 11
- Python 3.10+

安装依赖:

```bash
python3 -m pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m pip install torch transformers modelscope safetensors accelerate -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 4. 下载模型（ModelScope）

```bash
python3 - << 'PY'
from modelscope.hub.snapshot_download import snapshot_download
print(snapshot_download('Qwen/Qwen3-1.7B', cache_dir='/root/qwen3-cuda-minimal/ms_models'))
PY
```

建议使用绝对路径:
- `/root/qwen3-cuda-minimal/ms_models/Qwen/Qwen3-1.7B`

## 5. 转换权重

`qmini(v3)` 包含 `q_norm/k_norm`。

```bash
cd /root/qwen3-cuda-minimal
python3 scripts/convert_qwen3_to_qmini.py \
  --model-dir /root/qwen3-cuda-minimal/ms_models/Qwen/Qwen3-1.7B \
  --output /root/qwen3-cuda-minimal/weights/qwen3_1.7b_seq512_refactor.qmini \
  --max-seq-len 512
```

## 6. 编译

```bash
cd /root/qwen3-cuda-minimal
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## 7. 快速推理

```bash
cd /root/qwen3-cuda-minimal
python3 scripts/run_chat.py \
  --engine ./build/qwen_minimal \
  --weights /root/qwen3-cuda-minimal/weights/qwen3_1.7b_seq512_refactor.qmini \
  --model-dir /root/qwen3-cuda-minimal/ms_models/Qwen/Qwen3-1.7B \
  --prompt "你好，请用一句话介绍你自己。" \
  --max-new-tokens 20 \
  --temperature 0.55 \
  --top-k 32 \
  --top-p 0.78 \
  --min-p 0.10 \
  --temp-decay 0.94 \
  --greedy-after 8 \
  --no-repeat-ngram-size 3 \
  --presence-penalty 0.2 \
  --frequency-penalty 0.15 \
  --repetition-penalty 1.3
```

## 8. 对齐验证（推荐）

用于定位 HF 与本地引擎剩余偏差。

```bash
cd /root/qwen3-cuda-minimal
python3 scripts/check_logits_alignment.py \
  --engine ./build/qwen_minimal \
  --weights /root/qwen3-cuda-minimal/weights/qwen3_1.7b_seq512_refactor.qmini \
  --model-dir /root/qwen3-cuda-minimal/ms_models/Qwen/Qwen3-1.7B \
  --prompt "你好，请用一句话介绍你自己。" \
  --steps 8 \
  --top-k 10 \
  --save-json /root/qwen3-cuda-minimal/weights/alignment_report_1.7b.json
```

关注指标:
- `same_next_steps`
- `avg_topk_overlap_ratio`
- `avg_shared_logit_mae`
- `first_next_token_divergence_step`

## 9. 五问冒烟测试

```bash
cd /root/qwen3-cuda-minimal
python3 scripts/qa5_smoke_test.py \
  --engine ./build/qwen_minimal \
  --weights /root/qwen3-cuda-minimal/weights/qwen3_1.7b_seq512_refactor.qmini \
  --model-dir /root/qwen3-cuda-minimal/ms_models/Qwen/Qwen3-1.7B \
  --output-report /root/qwen3-cuda-minimal/weights/qa5_report.md
```

测试问题包括:
- 自我介绍
- 简单算术
- 常识问答
- 中英翻译
- 三条短建议

## 10. 代码结构

- `src/main.cpp`: CLI 参数解析与引擎入口
- `src/model.cpp`: 模型加载、前向、采样与调试输出
- `src/kernels.cu`: CUDA 算子（含 Flash/Paged Attention）
- `scripts/convert_qwen3_to_qmini.py`: 权重转换（含严格 shape 校验）
- `scripts/check_logits_alignment.py`: HF logits 对齐检查
- `scripts/qa5_smoke_test.py`: 五问冒烟测试

## 11. 常见问题

1. 路径被识别成 repo_id
- 现象: `Repo id must be in the form ...`
- 处理: 使用存在的本地绝对路径

2. 输出重复或异常
- 优先检查: 是否使用 `qwen3_1.7b_seq512_refactor.qmini`
- 再检查: 采样参数和 `no-repeat-ngram-size`

3. 对齐脚本首步就分叉
- 通常说明算子语义有偏差（RoPE/Attention/权重布局）
- 建议先看 `alignment_report_*.json` 的 step0 细节
- 传入了不存在的本地路径，`transformers` 把它当远程 repo id。
