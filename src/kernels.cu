#include "kernels.cuh"

#include "common.hpp"

#include <cuda_runtime.h>
#include <math_constants.h>

namespace {

inline int choose_attention_threads(int head_dim) {
  int threads = 1;
  while (threads < head_dim && threads < 256) {
    threads <<= 1;
  }
  return threads;
}

__global__ void embedding_lookup_kernel(const half* table, int vocab_size, int hidden_size, int token_id,
                                        half* out) {
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

__global__ void add_inplace_kernel(half* x, const half* y, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    x[idx] = __float2half(__half2float(x[idx]) + __half2float(y[idx]));
  }
}

__global__ void sigmoid_mul_inplace_kernel(half* x, const half* gate, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float g = __half2float(gate[idx]);
    float sig = 1.0f / (1.0f + expf(-g));
    float xv = __half2float(x[idx]);
    x[idx] = __float2half(xv * sig);
  }
}

__global__ void rope_inplace_kernel(half* q_or_k, int num_heads, int head_dim, int position, float rope_theta) {
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

  // 与 transformers 的 rotate_half 保持一致: [x1, x2] -> [-x2, x1]
  float y0 = x0 * c - x1 * s;
  float y1 = x1 * c + x0 * s;

  q_or_k[base + d0] = __float2half(y0);
  q_or_k[base + d1] = __float2half(y1);
}


__global__ void swiglu_kernel(const half* gate, const half* up, int n, half* out) {
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

  // online softmax 统计量: m 为行最大值, l 为归一化分母
  float m = -CUDART_INF_F;
  float l = 0.0f;
  float inv_scale = rsqrtf(static_cast<float>(head_dim));
  int d0 = tid;
  float acc = 0.0f;

  for (int t = 0; t < seq_len; ++t) {
    const half* kh = k_cache + (t * num_kv_heads + kv_h) * head_dim;
    const half* vh = v_cache + (t * num_kv_heads + kv_h) * head_dim;

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

    // alpha/beta 用于把历史统计与当前 token 增量稳定融合
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

__global__ void paged_attention_kernel(const half* q, const half* k_cache, const half* v_cache,
                                       const int* page_table, int page_size, int num_heads,
                                       int num_kv_heads, int head_dim, int seq_len, half* context) {
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
    // 将逻辑位置映射到物理页，模拟 paged KV cache 访问
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
  int threads = 256;
  int blocks = (hidden_size + threads - 1) / threads;
  embedding_lookup_kernel<<<blocks, threads>>>(table, vocab_size, hidden_size, token_id, out);
}

void launch_rmsnorm(const half* x, const half* weight, int n, float eps, half* out) {
  rmsnorm_kernel<<<1, 256>>>(x, weight, n, eps, out);
}

void launch_head_rmsnorm(const half* x, const half* weight, int num_heads, int head_dim, float eps,
                         half* out) {
  head_rmsnorm_kernel<<<num_heads, 256>>>(x, weight, num_heads, head_dim, eps, out);
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
  int threads = choose_attention_threads(head_dim);
  flash_attention_kernel<<<num_heads, threads, threads * static_cast<int>(sizeof(float))>>>(
    q, k_cache, v_cache, num_heads, num_kv_heads, head_dim, seq_len, context);
}

void launch_paged_attention(const half* q, const half* k_cache, const half* v_cache,
                            const int* page_table, int page_size, int num_heads, int num_kv_heads,
                            int head_dim, int seq_len, half* context) {
  int threads = choose_attention_threads(head_dim);
  paged_attention_kernel<<<num_heads, threads, threads * static_cast<int>(sizeof(float))>>>(
    q, k_cache, v_cache, page_table, page_size, num_heads, num_kv_heads, head_dim, seq_len, context);
}
