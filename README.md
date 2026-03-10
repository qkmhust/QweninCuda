# Qwen3/Qwen3.5 CUDA 本地离线推理

本项目目标：
- 用 C++20 + CUDA 实现单卡、单样本、自回归 decode。
- 支持 Qwen3（纯 full-attention）与 Qwen3.5（linear + full 混合层）。
- 权重格式统一为 qmini v4，严格保留官方结构，不做截断。

## 1. 项目结构

- src/main.cpp
  - 命令行入口，解析生成参数。
- src/model.cpp
  - 模型加载、缓存分配、单步 decode 主流程。
- src/kernels.cu
  - CUDA 内核：embedding、RMSNorm、RoPE、Attention、SwiGLU 等。
- include/model.hpp
  - 配置结构、权重结构、运行时缓存结构。
- scripts/convert_qwen3_to_qmini.py
  - 从 HF/ModelScope 导出 qmini v4 权重。
- scripts/run_chat.py
  - 本地 tokenizer + C++ 引擎调用。
- scripts/check_logits_alignment.py
  - 与 HF 逐步 top-k/logits 对齐。

## 2. 模型下载（仅国内源）

示例：

python3 - << 'PY'
from modelscope.hub.snapshot_download import snapshot_download
print(snapshot_download('Qwen/Qwen3.5-4B', cache_dir='/root/QweninCuda/ms_models'))
PY

常见本地路径：
- /root/QweninCuda/ms_models/Qwen/Qwen3___5-4B

注意：ModelScope 会把目录中的特殊字符替换为下划线。

## 2.1 提示词优化（提升输出质量）

本仓库已在 `scripts/run_chat.py` 中默认启用：
- `system prompt`：约束“只输出最终答案，不输出推理过程与 `<think>` 标签”。
- `chat template`：优先走 tokenizer 的 `apply_chat_template`，提升对话模型指令遵循能力。
- 展示层清洗：默认清理 `<think>...</think>` 片段，便于直接查看答案。

建议参数（通用问答）：
- `temperature=0.45~0.65`
- `top-k=24~48`
- `top-p=0.75~0.90`
- `min-p=0.05~0.15`
- `repetition-penalty=1.15~1.30`

如果想看模型原始完整输出（包含思维片段），可加：
- `--keep-think`

## 3. qmini v4 权重格式说明

### 3.1 头部字段顺序

1. magic: QWENMINI
2. version: int32（当前为 4）
3. vocab_size, hidden_size, intermediate_size
4. num_layers, num_heads, num_kv_heads, head_dim
5. linear_num_key_heads, linear_num_value_heads
6. linear_key_head_dim, linear_value_head_dim
7. linear_conv_kernel_dim
8. max_seq_len
9. rms_norm_eps, rope_theta
10. layer_types[num_layers]（0=full_attention, 1=linear_attention）

### 3.2 各层权重布局

1. full-attention 层
- input_layernorm: [hidden]
- q_proj: [2*q_dim, hidden]，其中 q_dim = num_heads * head_dim
- k_proj: [kv_dim, hidden]，kv_dim = num_kv_heads * head_dim
- v_proj: [kv_dim, hidden]
- q_norm: [head_dim]
- k_norm: [head_dim]
- o_proj: [hidden, q_dim]

2. linear-attention 层（Qwen3.5 GatedDeltaNet）
- in_proj_qkv: [conv_dim, hidden]
  - conv_dim = 2*linear_key_dim + linear_value_dim
  - linear_key_dim = linear_num_key_heads * linear_key_head_dim
  - linear_value_dim = linear_num_value_heads * linear_value_head_dim
- in_proj_z: [linear_value_dim, hidden]
- in_proj_b: [linear_num_value_heads, hidden]
- in_proj_a: [linear_num_value_heads, hidden]
- conv1d.weight: [conv_dim, linear_conv_kernel_dim]
- dt_bias: [linear_num_value_heads]
- A_log: [linear_num_value_heads]
- norm.weight: [linear_value_head_dim]
- out_proj: [hidden, linear_value_dim]

3. MLP（两类层共用）
- post_attention_layernorm: [hidden]
- gate_proj: [intermediate, hidden]
- up_proj: [intermediate, hidden]
- down_proj: [hidden, intermediate]

4. 模型尾部
- final_norm: [hidden]
- lm_head: [vocab, hidden]

## 4. 推理主流程（逐步维度变化）

以下描述单 batch、单 token 的 decode_next。

### Step 0: 输入 token -> embedding

- 输入：token_id（标量）
- 输出：x，形状 [hidden]

### Step 1: 层前归一化

- 输入：x [hidden]
- 参数：input_layernorm [hidden]
- 输出：x_norm [hidden]

### Step 2A: full-attention 分支（layer_type=0）

1. Q/K/V 投影
- q_cat = q_proj(x_norm): [2*q_dim]
- k = k_proj(x_norm): [kv_dim]
- v = v_proj(x_norm): [kv_dim]

2. q/gate 拆分（按 head 交错）
- 对每个 head，q_cat 的 head 子块为 [q_chunk, gate_chunk]
- 拼接后得到：
  - q: [q_dim]
  - q_gate: [q_dim]

3. q_norm / k_norm
- q 按 head_dim 做 head RMSNorm，形状不变 [q_dim]
- k 按 head_dim 做 head RMSNorm，形状不变 [kv_dim]

4. RoPE
- 对 q、k 就地旋转，形状不变

5. KV cache 追加
- KCache[layer, pos, :] <- k
- VCache[layer, pos, :] <- v

6. 注意力计算
- 短序列走 flash_attention
- 长序列走 paged_attention
- 输出 context: [q_dim]

7. 门控与输出投影
- context <- context * sigmoid(q_gate)
- attn_out = o_proj(context): [hidden]

### Step 2B: linear-attention 分支（layer_type=1）

1. 投影
- mixed_qkv = in_proj_qkv(x_norm): [conv_dim]
- z = in_proj_z(x_norm): [linear_value_dim]
- b = in_proj_b(x_norm): [linear_num_value_heads]
- a = in_proj_a(x_norm): [linear_num_value_heads]

2. depthwise causal conv
- 使用每层 conv_state: [conv_dim, kernel]
- 更新窗口后输出 mixed: [conv_dim]
- mixed = silu(conv1d(...))

3. 切分并重排
- query_raw: [linear_key_dim]
- key_raw: [linear_key_dim]
- value_raw: [linear_value_dim]

4. q/k L2 归一化
- 每个 key head 单独做 L2Norm

5. key head 扩展到 value head
- 当 linear_num_value_heads > linear_num_key_heads 时重复扩展
- q_rep / k_rep: [linear_num_value_heads, linear_key_head_dim]

6. 递推状态更新（每个 value head）
- recurrent_state 形状：
  - [linear_num_value_heads, linear_key_head_dim, linear_value_head_dim]
- 参数变换：
  - beta = sigmoid(b)
  - g = exp(-exp(A_log) * softplus(a + dt_bias))
- 状态更新：
  - state = state * g
  - delta = (v - kv_mem) * beta
  - state = state + k * delta

7. 读出与门控归一化
- core_out = state 与 q 的乘积，形状 [linear_value_dim]
- 按 value_head 做 RMSNorm
- 乘 norm.weight 与 silu(z)
- linear_out = out_proj(core_out): [hidden]

## 4.1 Qwen3.5 新架构讲解（相对 Qwen3）

Qwen3.5 的核心变化是“混合层”而不是“纯 attention 堆叠”：

1. 层类型混合
- Qwen3：通常每层都是 full-attention + MLP。
- Qwen3.5：`layer_types` 显式区分 `linear_attention` 与 `full_attention`。

2. full-attention 的 q_proj 变成双分支
- Qwen3 常见：`q_proj -> [q_dim]`
- Qwen3.5：`q_proj -> [2*q_dim]`，拆为 `query` 与 `gate`。
- 注意力输出会乘 `sigmoid(gate)` 再进入 `o_proj`。

3. linear-attention 的状态化递推
- 每层维护 `conv_state`（卷积窗口状态）和 `recurrent_state`（记忆状态）。
- decode 时每个新 token 仅更新状态，无需回看全部历史 K/V。
- 这种结构在长上下文下可降低部分计算负担。

4. 算子层面新增
- depthwise causal conv
- gated delta rule 递推更新
- RMSNormGated

### Step 3: 残差连接

- x <- residual + mixer_out
- 形状始终 [hidden]

### Step 4: MLP

1. post_attention_layernorm
- x_norm2: [hidden]

2. SwiGLU
- gate = gate_proj(x_norm2): [intermediate]
- up = up_proj(x_norm2): [intermediate]
- hidden_ffn = swiglu(gate, up): [intermediate]

3. 下投影
- mlp_out = down_proj(hidden_ffn): [hidden]

4. 残差
- x <- x + mlp_out

### Step 5: 输出 logits

- x_norm_final = final_norm(x): [hidden]
- logits = lm_head(x_norm_final): [vocab]

### Step 6: 采样参数变化（generate）

每步 i 的温度更新：
- temp_i = max(0.05, temp_{i-1} * temp_decay)
- 当 i >= greedy_after 时直接 argmax。

采样中还会应用：
- top_k / top_p / min_p
- no_repeat_ngram
- presence_penalty / frequency_penalty / repetition_penalty

## 5. 编译与运行

### 5.1 转换

python3 scripts/convert_qwen3_to_qmini.py \
  --model-dir /root/QweninCuda/ms_models/Qwen/Qwen3___5-4B \
  --output /root/QweninCuda/weights/qwen3_5_4b_strict_seq512.qmini \
  --max-seq-len 512

### 5.2 编译

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

### 5.3 运行

python3 scripts/run_chat.py \
  --engine ./build/qwen_minimal \
  --weights /root/QweninCuda/weights/qwen3_5_4b_strict_seq512.qmini \
  --model-dir /root/QweninCuda/ms_models/Qwen/Qwen3___5-4B \
  --prompt "你好，请用一句话介绍你自己。" \
  --max-new-tokens 32 \
  --temperature 0.55 \
  --top-k 32 \
  --top-p 0.78 \
  --min-p 0.12 \
  --temp-decay 0.94 \
  --greedy-after 8

## 6. 对齐与质量验证

### 6.1 与 HF 逐步对齐

python3 scripts/check_logits_alignment.py \
  --engine ./build/qwen_minimal \
  --weights /root/QweninCuda/weights/qwen3_5_4b_strict_seq512.qmini \
  --model-dir /root/QweninCuda/ms_models/Qwen/Qwen3___5-4B \
  --prompt "你好，请用一句话介绍你自己。" \
  --steps 8 \
  --top-k 10

### 6.2 冒烟测试

python3 scripts/qa5_smoke_test.py \
  --engine ./build/qwen_minimal \
  --weights /root/QweninCuda/weights/qwen3_5_4b_strict_seq512.qmini \
  --model-dir /root/QweninCuda/ms_models/Qwen/Qwen3___5-4B \
  --output-report /root/QweninCuda/weights/qa5_report_qwen3_5_4b.md

## 7. 常见问题

1. Repo id must be in the form ...
- 原因：model-dir 不存在，被误当成远程仓库名。
- 处理：改成实际存在的本地绝对路径。

2. 输出质量异常
- 建议先跑逐步对齐脚本。
- 若 step0 分叉，优先检查：
  - q_proj 拆分顺序
  - RMSNorm 权重语义
  - RoPE 配对与头维度

3. 速度偏慢
- 当前 linear-attention 递推为 C++ 标量实现，优先保证结构正确。
- 后续可将 linear 递推与 conv 更新 CUDA 化。

## 8. 不同参数模型差异与显存需求（学习与选型）

### 8.1 显存估算公式

推理显存可粗估为：

$$
	ext{VRAM}_{total} \approx \text{Weights} + \text{KV/State Cache} + \text{Runtime Buffers} + \text{Fragmentation}
$$

其中：

1. 权重显存（FP16）
$$
	ext{Weights(GB)} \approx \frac{\text{ParamCount} \times 2}{1024^3}
$$

2. full-attention KV cache（单 batch）
$$
	ext{KV bytes} \approx 2 \times L_{full} \times T \times (n_{kv} \times d_h) \times 2
$$
- 第一个 `2`：K 和 V 两份
- 最后一个 `2`：FP16 每元素 2 字节
- `L_full`：full-attention 层数
- `T`：上下文长度（token）

3. linear-attention state（单 batch）
- `conv_state` 与 `recurrent_state` 与序列长度弱相关或无关，通常小于权重占用。

### 8.2 常见模型量级（经验值，单卡 FP16 推理）

仅作学习选型参考，实际受实现、batch、上下文长度影响：

1. 1.7B 级
- 权重约 3.4~4.0 GB
- 推荐显存：8~12 GB

2. 4B 级（如 Qwen3.5-4B）
- 权重约 8~10 GB
- 推荐显存：16~24 GB

3. 7B 级
- 权重约 14~16 GB
- 推荐显存：24~32 GB

4. 14B 级
- 权重约 28~32 GB
- 推荐显存：40~48 GB 或多卡

### 8.3 参数量增大时的学习重点

1. hidden/intermediate 增大
- GEMV/GEMM 开销近似按矩阵规模增长。

2. 层数增大
- decode 单步需要遍历更多层，时延近似线性上升。

3. 上下文长度增大
- full-attention 的 KV cache 线性增长。
- linear-attention 更依赖固定状态更新，长上下文更有优势。
