#include "kernels.cuh"

#include "common.hpp"

#include <cuda_runtime.h>
#include <math_constants.h>

namespace {

// -----------------------------------------------------------------------------
// 1) 这个文件只做“单个 CUDA kernel 的实现”，不包含模型调度逻辑。
// 2) 每个 launch_* 函数都在文件末尾，它们只是 kernel 的薄封装。
// 3) 算子可以按用途分成三组：
//    - 基础逐元素/归一化：embedding_lookup / rmsnorm / add / swiglu / sigmoid_mul
//    - 位置编码：rope_inplace
//    - 注意力：flash_attention / paged_attention
//----------------------------------------------------------------------------

// attention kernel 会让一个 block 负责一个 query head。
// 线程数至少覆盖一个 head 的维度，但最多只开到 256，避免 block 过大。
inline int choose_attention_threads(int head_dim) {
  int threads = 1;
  while (threads < head_dim && threads < 256) {
    threads <<= 1;
  }
  return threads;
}

__global__ void embedding_lookup_kernel(const half* table, int vocab_size, int hidden_size, int token_id,
                                        half* out) {
  // 输入：
  // - table: [vocab_size, hidden_size]
  // - token_id: 当前 token
  // 输出：
  // - out: [hidden_size]
  // 一个线程负责拷贝 embedding 向量中的一个元素。
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= hidden_size) {
    return;
  }
  int safe_token = token_id;
  if (safe_token < 0) {
    safe_token = 0;
  }
  if (safe_token >= vocab_size) {
    safe_token = vocab_size - 1;
  }
  out[idx] = table[safe_token * hidden_size + idx];
}

__global__ void rmsnorm_kernel(const half* x, const half* weight, int n, float eps, half* out) {
  // 输入：x/weight 都是长度 n 的向量
  // 输出：out[i] = x[i] * rsqrt(mean(x^2)+eps) * weight[i]
  // 这个 kernel 用一个 block 处理一整个向量：
  // 先算平方和，再得到 inv_rms，最后写回归一化结果。
  __shared__ float ssum;
  if (threadIdx.x == 0) {
    ssum = 0.0f;
  }
  __syncthreads();

  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float v = __half2float(x[i]);
    local_sum += v * v;
  }
  atomicAdd(&ssum, local_sum);
  __syncthreads();

  float inv_rms = rsqrtf(ssum / static_cast<float>(n) + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float xv = __half2float(x[i]);
    float wv = __half2float(weight[i]);
    out[i] = __float2half(xv * inv_rms * wv);
  }
}

__global__ void head_rmsnorm_kernel(const half* x, const half* weight, int num_heads, int head_dim,
                                    float eps, half* out) {
  // 输入：x 视作 [num_heads, head_dim]
  // 输出：out 形状同 x，每个 head 独立做 RMSNorm
  // 每个 block 只处理一个 attention head，
  // 这是 Qwen3.5 在 q/k 上额外使用的 head-wise RMSNorm。
  int h = blockIdx.x;
  if (h >= num_heads) {
    return;
  }

  const half* xh = x + h * head_dim;
  half* oh = out + h * head_dim;

  __shared__ float ssum;
  if (threadIdx.x == 0) {
    ssum = 0.0f;
  }
  __syncthreads();

  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    float v = __half2float(xh[i]);
    local_sum += v * v;
  }
  atomicAdd(&ssum, local_sum);
  __syncthreads();

  float inv_rms = rsqrtf(ssum / static_cast<float>(head_dim) + eps);
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    float xv = __half2float(xh[i]);
    float wv = __half2float(weight[i]);
    oh[i] = __float2half(xv * inv_rms * wv);
  }
}

__global__ void split_q_gate_interleaved_kernel(const half* packed_q_gate, int num_heads, int head_dim,
                                                half* q, half* gate) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_heads * head_dim;
  if (idx >= total) {
    return;
  }

  int h = idx / head_dim;
  int d = idx % head_dim;
  int base = h * (2 * head_dim);
  q[idx] = packed_q_gate[base + d];
  gate[idx] = packed_q_gate[base + head_dim + d];
}

__global__ void add_inplace_kernel(half* x, const half* y, int n) {
  // 输入/输出都在 x 上原地进行，减少额外显存占用。
  // 最简单的残差加法：x = x + y。
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    x[idx] = __float2half(__half2float(x[idx]) + __half2float(y[idx]));
  }
}

__global__ void sigmoid_mul_inplace_kernel(half* x, const half* gate, int n) {
  // 常见于门控结构，把 gate 经过 sigmoid 后作为逐元素缩放系数。
  // full-attention 输出门控：x = x * sigmoid(gate)。
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float g = __half2float(gate[idx]);
    float sig = 1.0f / (1.0f + expf(-g));
    float xv = __half2float(x[idx]);
    x[idx] = __float2half(xv * sig);
  }
}

__global__ void rope_inplace_kernel(half* q_or_k, int num_heads, int head_dim, int position, float rope_theta) {
  // 输入：q_or_k 视作 [num_heads, head_dim]
  // 输出：原地改写，完成 RoPE 旋转
  // RoPE 会把每两个维度视作一个二维平面，然后按当前位置做旋转。
  // 这里的实现按“前半段 + 后半段”两两配对，和当前项目的 qmini 权重布局一致。
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_heads * (head_dim / 2);
  if (idx >= total) {
    return;
  }
  int h = idx / (head_dim / 2);
  int d = idx % (head_dim / 2);

  int base = h * head_dim;
  int half_dim = head_dim / 2;
  int d0 = d;
  int d1 = d + half_dim;

  float x0 = __half2float(q_or_k[base + d0]);
  float x1 = __half2float(q_or_k[base + d1]);

  float inv_freq = powf(rope_theta, -2.0f * static_cast<float>(d) / static_cast<float>(head_dim));
  float angle = static_cast<float>(position) * inv_freq;

  float c = cosf(angle);
  float s = sinf(angle);

  // 旋转后的结果等价于对二维向量乘一个旋转矩阵。
  float y0 = x0 * c - x1 * s;
  float y1 = x1 * c + x0 * s;

  q_or_k[base + d0] = __float2half(y0);
  q_or_k[base + d1] = __float2half(y1);
}


__global__ void swiglu_kernel(const half* gate, const half* up, int n, half* out) {
  // 输入：gate/up 都是长度 n 的向量
  // 输出：out[i] = silu(gate[i]) * up[i]
  // SwiGLU(x, y) = silu(x) * y。
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  float g = __half2float(gate[idx]);
  float u = __half2float(up[idx]);
  float sig = 1.0f / (1.0f + expf(-g));
  out[idx] = __float2half((g * sig) * u);
}

__global__ void flash_attention_kernel(const half* q, const half* k_cache, const half* v_cache,
                                       int num_heads, int num_kv_heads, int head_dim, int seq_len,
                                       half* context) {
  // 张量语义（单 batch）：
  // - q: [num_heads, head_dim]，当前时刻 query
  // - k_cache/v_cache: [seq_len, num_kv_heads, head_dim] 的线性展开
  // - context: [num_heads, head_dim]，输出注意力结果
  // 教学版 flash attention：
  // - 一个 block 负责一个 query head
  // - 顺序扫描整段历史 token
  // - 用 online softmax 避免先存完整分数矩阵
  int h = blockIdx.x;
  if (h >= num_heads) {
    return;
  }

  extern __shared__ float smem[];
  float* dot_buf = smem;
  int tid = threadIdx.x;

  int group = num_heads / num_kv_heads;
  int kv_h = h / group;
  const half* qh = q + h * head_dim;
  // 多 query-head 共享一个 kv-head（GQA）时，
  // group 表示“每个 kv head 对应多少个 query head”。

  // m: 目前看到的最大 logit
  // l: 对应的 softmax 分母累计值
  float m = -CUDART_INF_F;
  float l = 0.0f;
  float inv_scale = rsqrtf(static_cast<float>(head_dim));
  int d0 = tid;
  float acc = 0.0f;

  for (int t = 0; t < seq_len; ++t) {
    // 当前 query head 可能会共享某个 kv head，
    // 所以先算出它映射到哪个 kv head。
    const half* kh = k_cache + (t * num_kv_heads + kv_h) * head_dim;
    const half* vh = v_cache + (t * num_kv_heads + kv_h) * head_dim;

    // 先在 block 内并行做 q·k。
    float partial = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
      partial += __half2float(qh[d]) * __half2float(kh[d]);
    }
    dot_buf[tid] = partial;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        dot_buf[tid] += dot_buf[tid + stride];
      }
      __syncthreads();
    }
    // 经过归约后，dot_buf[0] 就是该 head 对第 t 个 token 的注意力打分（未缩放 softmax）。

    // alpha/beta 是 online softmax 的关键：
    // 历史部分乘 alpha，新 token 贡献乘 beta，二者合并后依然数值稳定。
    float alpha = 0.0f;
    float beta = 0.0f;
    if (tid == 0) {
      float s = dot_buf[0] * inv_scale;
      float new_m = fmaxf(m, s);
      alpha = expf(m - new_m);
      beta = expf(s - new_m);
      m = new_m;
      l = l * alpha + beta;
      dot_buf[0] = alpha;
      dot_buf[1] = beta;
    }
    __syncthreads();
    alpha = dot_buf[0];
    beta = dot_buf[1];

    if (d0 < head_dim) {
      // acc 始终保存“未除以 softmax 分母”的加权和。
      float v = __half2float(vh[d0]);
      acc = acc * alpha + beta * v;
    }
    __syncthreads();
  }

  if (tid == 0) {
    dot_buf[0] = l;
  }
  __syncthreads();
  float inv_l = 1.0f / fmaxf(dot_buf[0], 1e-12f);
  if (d0 < head_dim) {
    context[h * head_dim + d0] = __float2half(acc * inv_l);
  }
}

__global__ void paged_attention_kernel(const half* q, const half* k_cache, const half* v_cache,
                                       const int* page_table, int page_size, int num_heads,
                                       int num_kv_heads, int head_dim, int seq_len, half* context) {
  // 与 flash_attention_kernel 的核心数学完全一致，
  // 差别只是：K/V 不是连续时间步存储，而是按 page 管理。
  // paged 版本和上面的 flash_attention_kernel 几乎一样，
  // 唯一差别是 token 位置要先经过 page table 映射到物理地址。
  int h = blockIdx.x;
  if (h >= num_heads) {
    return;
  }

  extern __shared__ float smem[];
  float* dot_buf = smem;
  int tid = threadIdx.x;

  int group = num_heads / num_kv_heads;
  int kv_h = h / group;
  const half* qh = q + h * head_dim;

  float m = -CUDART_INF_F;
  float l = 0.0f;
  float inv_scale = rsqrtf(static_cast<float>(head_dim));
  int d0 = tid;
  float acc = 0.0f;

  for (int t = 0; t < seq_len; ++t) {
    // 逻辑位置 -> 逻辑页号 + 页内偏移 -> 物理页号 -> 真实物理位置
    int logical_page = t / page_size;
    int page_offset = t % page_size;
    int physical_page = page_table[logical_page];
    int physical_t = physical_page * page_size + page_offset;

    const half* kh = k_cache + (physical_t * num_kv_heads + kv_h) * head_dim;
    const half* vh = v_cache + (physical_t * num_kv_heads + kv_h) * head_dim;

    float partial = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
      partial += __half2float(qh[d]) * __half2float(kh[d]);
    }
    dot_buf[tid] = partial;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        dot_buf[tid] += dot_buf[tid + stride];
      }
      __syncthreads();
    }

    // 下面这段 online softmax 融合逻辑与 flash 版本完全相同。
    float alpha = 0.0f;
    float beta = 0.0f;
    if (tid == 0) {
      float s = dot_buf[0] * inv_scale;
      float new_m = fmaxf(m, s);
      alpha = expf(m - new_m);
      beta = expf(s - new_m);
      m = new_m;
      l = l * alpha + beta;
      dot_buf[0] = alpha;
      dot_buf[1] = beta;
    }
    __syncthreads();
    alpha = dot_buf[0];
    beta = dot_buf[1];

    if (d0 < head_dim) {
      float v = __half2float(vh[d0]);
      acc = acc * alpha + beta * v;
    }
    __syncthreads();
  }

  if (tid == 0) {
    dot_buf[0] = l;
  }
  __syncthreads();
  float inv_l = 1.0f / fmaxf(dot_buf[0], 1e-12f);
  if (d0 < head_dim) {
    context[h * head_dim + d0] = __float2half(acc * inv_l);
  }
}

}  // namespace

void launch_embedding_lookup(const half* table, int vocab_size, int hidden_size, int token_id, half* out) {
  // 这些 launch_* 函数只做两件事：
  // 1. 选择合适的 grid / block 规模
  // 2. 把底层 kernel 封装成更清晰的 C++ 接口
  int threads = 256;
  int blocks = (hidden_size + threads - 1) / threads;
  embedding_lookup_kernel<<<blocks, threads>>>(table, vocab_size, hidden_size, token_id, out);
}

void launch_rmsnorm(const half* x, const half* weight, int n, float eps, half* out) {
  // n 一般是 hidden_size；这里固定 1 个 block，靠 block 内循环覆盖整向量。
  rmsnorm_kernel<<<1, 256>>>(x, weight, n, eps, out);
}

void launch_head_rmsnorm(const half* x, const half* weight, int num_heads, int head_dim, float eps,
                         half* out) {
  // 一头一个 block，便于理解 head-wise 的并行映射。
  head_rmsnorm_kernel<<<num_heads, 256>>>(x, weight, num_heads, head_dim, eps, out);
}

void launch_split_q_gate_interleaved(const half* packed_q_gate, int num_heads, int head_dim,
                                     half* q, half* gate) {
  int total = num_heads * head_dim;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  split_q_gate_interleaved_kernel<<<blocks, threads>>>(packed_q_gate, num_heads, head_dim, q, gate);
}

void launch_add_inplace(half* x, const half* y, int n) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  add_inplace_kernel<<<blocks, threads>>>(x, y, n);
}

void launch_sigmoid_mul_inplace(half* x, const half* gate, int n) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  sigmoid_mul_inplace_kernel<<<blocks, threads>>>(x, gate, n);
}

void launch_rope_inplace(half* q_or_k, int num_heads, int head_dim, int position, float rope_theta) {
  // total = 需要旋转的“二维对”数量 = num_heads * (head_dim/2)
  int total = num_heads * (head_dim / 2);
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  rope_inplace_kernel<<<blocks, threads>>>(q_or_k, num_heads, head_dim, position, rope_theta);
}

void launch_swiglu(const half* gate, const half* up, int n, half* out) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  swiglu_kernel<<<blocks, threads>>>(gate, up, n, out);
}

void launch_flash_attention(const half* q, const half* k_cache, const half* v_cache, int num_heads,
                            int num_kv_heads, int head_dim, int seq_len, half* context) {
  // 动态共享内存大小 = 每个线程一个 float，存 block 内归约临时值。
  int threads = choose_attention_threads(head_dim);
  flash_attention_kernel<<<num_heads, threads, threads * static_cast<int>(sizeof(float))>>>(
    q, k_cache, v_cache, num_heads, num_kv_heads, head_dim, seq_len, context);
}

void launch_paged_attention(const half* q, const half* k_cache, const half* v_cache,
                            const int* page_table, int page_size, int num_heads, int num_kv_heads,
                            int head_dim, int seq_len, half* context) {
  // paged 版本与 flash 版本使用同样的线程配置策略。
  int threads = choose_attention_threads(head_dim);
  paged_attention_kernel<<<num_heads, threads, threads * static_cast<int>(sizeof(float))>>>(
    q, k_cache, v_cache, page_table, page_size, num_heads, num_kv_heads, head_dim, seq_len, context);
}
