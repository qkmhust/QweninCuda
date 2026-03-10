# Qwen3 CUDA Inference (C++20 + CUDA)

这个项目目标是: 在本地离线环境中，用 `C++20 + CUDA` 复现接近官方 Qwen3 的单 batch 推理主流程，并把关键实现细节暴露出来，方便学习和二次优化。

## 1. 项目定位

- 重点: 推理过程与 CUDA 实现细节。
- 非重点: 训练、服务化、分布式并行。
- 已对齐的核心路径:
  - `RMSNorm`
  - `Q/K/V + O + MLP` 全部线性层走 `cuBLAS (cublasGemmEx)`
  - `Q/K` 的 head-wise 归一化 (`q_norm/k_norm`)
  - `RoPE`
  - `GQA` attention (num_heads / num_kv_heads)
  - 稳定 softmax (row max + exp + sum reduction)
  - KV cache 自回归解码

## 2. 与官方实现的对应关系

### 2.1 前向主链路

当前实现按层执行:

1. `x = embed(token)`
2. `x_norm = rmsnorm(x, input_layernorm)`
3. `q = x_norm @ Wq, k = x_norm @ Wk, v = x_norm @ Wv` (cuBLAS)
4. `q = head_rmsnorm(q, q_norm), k = head_rmsnorm(k, k_norm)`
5. `q,k -> RoPE(position)`
6. 写入 KV cache
7. `scores = (q * k_cache) / sqrt(head_dim)`
8. `probs = softmax(scores)`
9. `context = probs * v_cache`
10. `attn_out = context @ Wo` (cuBLAS)
11. 残差连接
12. `ffn_norm = rmsnorm(x, post_attention_layernorm)`
13. `gate = ffn_norm @ Wgate, up = ffn_norm @ Wup`
14. `ffn_hidden = swiglu(gate, up)`
15. `ffn_out = ffn_hidden @ Wdown` (cuBLAS)
16. 残差连接
17. 末层 norm + lm_head 得到 logits

### 2.2 当前未覆盖的官方高级优化

- FlashAttention / fused attention kernel
- Paged KV cache
- Continuous batching
- 张量并行/流水并行
- 量化权重与量化 kernel

## 3. 环境准备

- Linux + NVIDIA GPU
- CUDA 12.x
- CMake >= 3.22
- g++ >= 11
- Python 3.10+

安装 Python 依赖 (国内镜像):

```bash
python3 -m pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m pip install torch transformers modelscope safetensors accelerate -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 4. 下载模型 (ModelScope)

```bash
python3 - << 'PY'
from modelscope.hub.snapshot_download import snapshot_download

local_dir = snapshot_download(
    model_id='Qwen/Qwen3-0.6B',
    cache_dir='/root/qwen3-cuda-minimal/ms_models',
)
print(local_dir)
PY
```

建议使用绝对路径:
- `/root/qwen3-cuda-minimal/ms_models/Qwen/Qwen3-0.6B`

## 5. 转换权重到本项目格式

> 当前格式版本为 v3，包含 `q_norm/k_norm`。

```bash
cd /root/qwen3-cuda-minimal
python3 scripts/convert_qwen3_to_qmini.py \
  --model-dir /root/qwen3-cuda-minimal/ms_models/Qwen/Qwen3-0.6B \
  --output /root/qwen3-cuda-minimal/weights/qwen3_0.6b.qmini \
  --max-seq-len 1024
```

## 6. 编译

```bash
cd /root/qwen3-cuda-minimal
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## 7. 推理教程

### 7.1 一条命令跑通

```bash
cd /root/qwen3-cuda-minimal
python3 scripts/run_chat.py \
  --engine ./build/qwen_minimal \
  --weights /root/qwen3-cuda-minimal/weights/qwen3_0.6b.qmini \
  --model-dir /root/qwen3-cuda-minimal/ms_models/Qwen/Qwen3-0.6B \
  --prompt "你好，请用一句话介绍你自己。" \
  --max-new-tokens 64 \
  --temperature 0.72 \
  --top-k 60 \
  --top-p 0.88 \
  --min-p 0.06 \
  --temp-decay 0.97 \
  --greedy-after 24 \
  --repetition-penalty 1.18
```

### 7.2 采样参数流程讲解

每个新 token 生成时:

1. 先按 `repetition_penalty` 调整历史 token 对应 logits
2. 按 `temperature` 缩放
3. 先做 `top-k` 截断
4. 再做 `top-p` nucleus 截断
5. 再做 `min-p` 截断 (阈值 = 当前最大概率 * min_p)
6. 在候选集随机采样
7. 若步数达到 `greedy-after`，改为 argmax
8. 每步后 `temperature *= temp-decay`

### 7.3 常用参数建议

- 更稳: `--temperature 0.65 --top-k 40 --top-p 0.85 --min-p 0.08 --repetition-penalty 1.2`
- 更发散: `--temperature 0.9 --top-k 100 --top-p 0.95 --min-p 0.02`
- 长文本收敛: `--temp-decay 0.97 --greedy-after 24`

## 8. 代码导航

- `src/model.cpp`: 模型加载、cuBLAS 投影、KV cache、采样解码
- `src/kernels.cu`: RMSNorm、head RMSNorm、RoPE、attention score/context、稳定 softmax
- `scripts/convert_qwen3_to_qmini.py`: 从本地 Qwen3 导出 v3 权重格式
- `scripts/run_chat.py`: tokenizer 桥接与推理调用

## 9. 流程图 (单 token decode)

```text
token_id
  -> embedding
  -> for each layer:
       rmsnorm
       qkv projection (cuBLAS)
       q_norm/k_norm
       rope
       kv cache write
       attention score -> softmax -> context
       o projection + residual
       ffn rmsnorm
       gate/up projection + swiglu
       down projection + residual
  -> final rmsnorm
  -> lm_head
  -> sampling
  -> next token
```

## 10. 验证与测速

- 输出关键字段:
  - `generated_ids=...`
  - `elapsed_ms=...`
  - `tok_per_s=...`

批量测试:

```bash
cd /root/qwen3-cuda-minimal
bash scripts/benchmark.sh /root/qwen3-cuda-minimal/weights/qwen3_0.6b.qmini /root/qwen3-cuda-minimal/ms_models/Qwen/Qwen3-0.6B
```

## 11. 下一步可做的严格对齐

1. 接入 FlashAttention (减少显存访问和延迟)
2. 接入 paged KV cache
3. 增加多 batch 连续调度
4. 接入量化权重与 kernel (INT8 / FP8)
