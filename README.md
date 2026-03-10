# Qwen3 CUDA Inference (C++20 + CUDA)

本项目聚焦一件事：
用 `C++20 + CUDA` 手写 Qwen3-1.7B 的核心推理链路，并把关键实现细节写清楚，方便学习和二次优化。

## 1. 项目范围

当前仅聚焦单卡、单样本、自回归解码，不做训练与服务化。

包含内容：
- Qwen3-1.7B 本地离线推理
- 手写 CUDA 核心算子（RMSNorm / RoPE / FlashAttention / PagedAttention / SwiGLU）
- cuBLAS 线性层（Q/K/V/O、MLP、lm_head）
- 自定义二进制权重格式 `qmini(v3)` 与维度转换脚本
- HF logits 对齐诊断脚本

不包含内容：
- Continuous batching
- 张量并行/流水并行
- 量化（INT8/FP8）

## 2. 目录说明

- `src/main.cpp`: C++ 引擎入口与参数解析
- `src/model.cpp`: 模型加载、前向主流程、采样
- `src/kernels.cu`: 手写 CUDA kernel
- `include/model.hpp`: 模型结构与接口声明
- `include/kernels.cuh`: kernel 启动接口声明
- `scripts/convert_qwen3_to_qmini.py`: 权重转换脚本（含 shape 严格校验）
- `scripts/run_chat.py`: 纯推理调用包装（不做人为回答修正）
- `scripts/check_logits_alignment.py`: HF 与本地 logits 对齐检查
- `scripts/qa5_smoke_test.py`: 五问冒烟测试

## 3. 环境要求

- Linux + NVIDIA GPU
- CUDA 12.x
- CMake >= 3.22
- g++ >= 11
- Python >= 3.10

安装依赖（国内镜像示例）：

```bash
python3 -m pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m pip install torch transformers modelscope safetensors accelerate -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 4. 模型下载与路径

下载 Qwen3-1.7B（ModelScope）：

```bash
python3 - << 'PY'
from modelscope.hub.snapshot_download import snapshot_download
print(snapshot_download('Qwen/Qwen3-1.7B', cache_dir='/root/qwen3-cuda-minimal/ms_models'))
PY
```

推荐本地路径：
- `/root/qwen3-cuda-minimal/ms_models/Qwen/Qwen3-1.7B`

## 5. 数据格式与维度转换

### 5.1 qmini(v3) 头部

按顺序写入：
- magic: `QWENMINI`
- version: `int32`
- vocab_size, hidden_size, intermediate_size
- num_layers, num_heads, num_kv_heads, head_dim
- max_seq_len
- rms_norm_eps, rope_theta

### 5.2 张量布局约定

1. embedding 表
- 形状：`[vocab, hidden]`
- 行主序直接写入
- 不允许转置（embedding lookup 按 `table[token, dim]` 索引）

2. 线性层权重
- PyTorch 常见形状：`[out_dim, in_dim]`
- 为匹配当前 C++ 侧 cuBLAS 调用约定，转换时写 `W.T`
- C++ 读取时按 `out_dim * in_dim` 扁平加载

3. 向量参数
- `input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`, `final_norm`
- 按一维向量原样写入

### 5.3 Qwen3-1.7B 关键维度

- hidden: 2048
- inter: 6144
- num_heads: 16
- num_kv_heads: 8
- head_dim: 128
- q_dim = `num_heads * head_dim` = 2048
- kv_dim = `num_kv_heads * head_dim` = 1024

## 6. 推理主流程（单步 decode）

每生成 1 个 token，执行：

1. `embedding(token)`
2. 对每一层：
- `input_layernorm`
- `Q/K/V` 投影（cuBLAS）
- `q_norm / k_norm`
- RoPE
- 写入 KV cache
- attention（短序列 FlashAttention，长序列 PagedAttention）
- `O` 投影 + 残差
- `post_attention_layernorm`
- `gate/up/down` + SwiGLU + 残差
3. `final_norm`
4. `lm_head`
5. 采样得到 next token

## 7. CUDA 手写内容

`src/kernels.cu` 当前手写实现：
- `embedding_lookup_kernel`
- `rmsnorm_kernel`
- `head_rmsnorm_kernel`
- `rope_inplace_kernel`
- `flash_attention_kernel`（online softmax）
- `paged_attention_kernel`（page table）
- `swiglu_kernel`

说明：
- RoPE 已按 Qwen3 `rotate_half` 语义实现（前半维与后半维配对）
- Flash/Paged Attention 使用在线归一化统计，避免 softmax 数值不稳定

## 8. 使用步骤

### 8.1 转换权重

```bash
cd /root/qwen3-cuda-minimal
python3 scripts/convert_qwen3_to_qmini.py \
  --model-dir /root/qwen3-cuda-minimal/ms_models/Qwen/Qwen3-1.7B \
  --output /root/qwen3-cuda-minimal/weights/qwen3_1.7b_seq512_refactor.qmini \
  --max-seq-len 512
```

### 8.2 编译

```bash
cd /root/qwen3-cuda-minimal
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### 8.3 推理

```bash
cd /root/qwen3-cuda-minimal
python3 scripts/run_chat.py \
  --engine ./build/qwen_minimal \
  --weights /root/qwen3-cuda-minimal/weights/qwen3_1.7b_seq512_refactor.qmini \
  --model-dir /root/qwen3-cuda-minimal/ms_models/Qwen/Qwen3-1.7B \
  --prompt "你好，请用一句话介绍你自己。" \
  --max-new-tokens 32 \
  --temperature 0.55 \
  --top-k 32 \
  --top-p 0.78 \
  --min-p 0.12 \
  --temp-decay 0.94 \
  --greedy-after 8 \
  --no-repeat-ngram-size 3 \
  --presence-penalty 0.2 \
  --frequency-penalty 0.15 \
  --repetition-penalty 1.3
```

## 9. 对齐与验证

### 9.1 HF logits 对齐

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

### 9.2 五问冒烟测试

```bash
cd /root/qwen3-cuda-minimal
python3 scripts/qa5_smoke_test.py \
  --engine ./build/qwen_minimal \
  --weights /root/qwen3-cuda-minimal/weights/qwen3_1.7b_seq512_refactor.qmini \
  --model-dir /root/qwen3-cuda-minimal/ms_models/Qwen/Qwen3-1.7B \
  --output-report /root/qwen3-cuda-minimal/weights/qa5_report.md
```

## 10. 常见问题

1. `Repo id must be in the form ...`
- 原因：传入路径不存在，被当成远程仓库名
- 处理：使用存在的本地绝对路径

2. 输出异常重复
- 先检查权重文件是否为最新转换版本
- 再调采样参数（temperature/top-k/top-p/min-p/penalty）

3. 对齐脚本 step0 就分叉
- 通常是算子语义偏差（RoPE、attention 或权重布局）
- 优先检查 RoPE 配对与转换脚本中的矩阵转置规则
