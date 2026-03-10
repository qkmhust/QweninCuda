#include "kernels.cuh"

#include "common.hpp"

#include <cuda_runtime.h>
#include <math_constants.h>

namespace {

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

__global__ void rope_inplace_kernel(half* q_or_k, int num_heads, int head_dim, int position, float rope_theta) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_heads * (head_dim / 2);
  if (idx >= total) {
    return;
  }
  int h = idx / (head_dim / 2);
  int d = idx % (head_dim / 2);

  int base = h * head_dim;
  int d0 = 2 * d;
  int d1 = d0 + 1;

  float x0 = __half2float(q_or_k[base + d0]);
  float x1 = __half2float(q_or_k[base + d1]);

  float inv_freq = powf(rope_theta, -2.0f * static_cast<float>(d) / static_cast<float>(head_dim));
  float angle = static_cast<float>(position) * inv_freq;

  float c = cosf(angle);
  float s = sinf(angle);

  float y0 = x0 * c - x1 * s;
  float y1 = x0 * s + x1 * c;

  q_or_k[base + d0] = __float2half(y0);
  q_or_k[base + d1] = __float2half(y1);
}

__global__ void attention_scores_kernel(const half* q, const half* k_cache, int num_heads, int num_kv_heads,
                                        int head_dim, int seq_len, float* scores) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_heads * seq_len;
  if (idx >= total) {
    return;
  }
  int h = idx / seq_len;
  int t = idx % seq_len;

  int group = num_heads / num_kv_heads;
  int kv_h = h / group;

  const half* qh = q + h * head_dim;
  const half* kh = k_cache + (t * num_kv_heads + kv_h) * head_dim;

  float dot = 0.0f;
  for (int i = 0; i < head_dim; ++i) {
    dot += __half2float(qh[i]) * __half2float(kh[i]);
  }
  scores[idx] = dot / sqrtf(static_cast<float>(head_dim));
}

__global__ void softmax_rows_kernel(float* scores, int rows, int cols) {
  int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  __shared__ float sdata[256];
  float local_max = -CUDART_INF_F;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    local_max = fmaxf(local_max, scores[row * cols + i]);
  }
  sdata[threadIdx.x] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  float max_val = sdata[0];

  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    float e = expf(scores[row * cols + i] - max_val);
    scores[row * cols + i] = e;
    local_sum += e;
  }
  sdata[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sdata[threadIdx.x] += sdata[threadIdx.x + stride];
    }
    __syncthreads();
  }
  float denom = sdata[0] + 1e-12f;

  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    scores[row * cols + i] /= denom;
  }
}

__global__ void attention_context_kernel(const float* probs, const half* v_cache, int num_heads,
                                         int num_kv_heads, int head_dim, int seq_len, half* context) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_heads * head_dim;
  if (idx >= total) {
    return;
  }

  int h = idx / head_dim;
  int d = idx % head_dim;
  int group = num_heads / num_kv_heads;
  int kv_h = h / group;

  float acc = 0.0f;
  for (int t = 0; t < seq_len; ++t) {
    float p = probs[h * seq_len + t];
    float v = __half2float(v_cache[(t * num_kv_heads + kv_h) * head_dim + d]);
    acc += p * v;
  }
  context[idx] = __float2half(acc);
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

void launch_rope_inplace(half* q_or_k, int num_heads, int head_dim, int position, float rope_theta) {
  int total = num_heads * (head_dim / 2);
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  rope_inplace_kernel<<<blocks, threads>>>(q_or_k, num_heads, head_dim, position, rope_theta);
}

void launch_attention_scores(const half* q, const half* k_cache, int num_heads, int num_kv_heads,
                             int head_dim, int seq_len, float* scores) {
  int total = num_heads * seq_len;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  attention_scores_kernel<<<blocks, threads>>>(q, k_cache, num_heads, num_kv_heads, head_dim, seq_len,
                                               scores);
}

void launch_softmax_rows(float* scores, int rows, int cols) {
  softmax_rows_kernel<<<rows, 256>>>(scores, rows, cols);
}

void launch_attention_context(const float* probs, const half* v_cache, int num_heads, int num_kv_heads,
                              int head_dim, int seq_len, half* context) {
  int total = num_heads * head_dim;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  attention_context_kernel<<<blocks, threads>>>(probs, v_cache, num_heads, num_kv_heads, head_dim,
                                                seq_len, context);
}

void launch_swiglu(const half* gate, const half* up, int n, half* out) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  swiglu_kernel<<<blocks, threads>>>(gate, up, n, out);
}
