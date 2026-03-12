#include "model.hpp"

#include "common.hpp"
#include "kernels.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace {

constexpr char kMagic[] = "QWENMINI";
constexpr int32_t kLayerTypeFullAttention = 0;
constexpr int32_t kLayerTypeLinearAttention = 1;

inline std::size_t kv_cache_offset(int position, int kv_dim) {
  return static_cast<std::size_t>(position) * static_cast<std::size_t>(kv_dim);
}

void read_exact(std::ifstream& in, char* ptr, std::streamsize n) {
  in.read(ptr, n);
  if (!in) {
    throw std::runtime_error("Failed to read model file");
  }
}

half* read_tensor_to_device(std::ifstream& in, std::size_t expected_elems, const char* name) {
  uint64_t n = 0;
  read_exact(in, reinterpret_cast<char*>(&n), sizeof(n));
  if (expected_elems != 0 && n != expected_elems) {
    throw std::runtime_error(std::string("Tensor shape mismatch: ") + name +
                             " expected=" + std::to_string(expected_elems) +
                             " actual=" + std::to_string(n));
  }

  std::vector<half> host(n);
  read_exact(in, reinterpret_cast<char*>(host.data()), static_cast<std::streamsize>(n * sizeof(half)));

  half* dev = nullptr;
  CUDA_CHECK(cudaMalloc(&dev, n * sizeof(half)));
  CUDA_CHECK(cudaMemcpy(dev, host.data(), n * sizeof(half), cudaMemcpyHostToDevice));
  return dev;
}

int argmax_host(const std::vector<half>& logits) {
  int best = 0;
  float best_v = -1e30f;
  for (int i = 0; i < static_cast<int>(logits.size()); ++i) {
    float v = __half2float(logits[i]);
    if (v > best_v) {
      best_v = v;
      best = i;
    }
  }
  return best;
}

std::vector<std::pair<int, float>> topk_host(const std::vector<half>& logits, int k) {
  if (k <= 0) {
    return {};
  }
  std::vector<std::pair<int, float>> scored;
  scored.reserve(logits.size());
  for (int i = 0; i < static_cast<int>(logits.size()); ++i) {
    scored.emplace_back(i, __half2float(logits[i]));
  }
  int keep = std::min<int>(k, static_cast<int>(scored.size()));
  std::partial_sort(scored.begin(), scored.begin() + keep, scored.end(),
                    [](const auto& a, const auto& b) { return a.second > b.second; });
  scored.resize(keep);
  return scored;
}

void print_step_topk(int step, const std::vector<half>& logits, int k) {
  auto top = topk_host(logits, k);
  std::cout << "step_topk[" << step << "]=";
  for (std::size_t j = 0; j < top.size(); ++j) {
    if (j > 0) {
      std::cout << ',';
    }
    std::cout << top[j].first << ':' << std::fixed << std::setprecision(6) << top[j].second;
  }
  std::cout << "\n";
}

std::unordered_set<int> collect_blocked_ngrams(const std::vector<int>& generated, int ngram_size) {
  std::unordered_set<int> blocked;
  if (ngram_size <= 1 || static_cast<int>(generated.size()) < ngram_size) {
    return blocked;
  }

  int prefix_len = ngram_size - 1;
  int start = static_cast<int>(generated.size()) - prefix_len;
  const int* prefix = generated.data() + start;

  for (int i = 0; i + ngram_size <= static_cast<int>(generated.size()); ++i) {
    bool match = true;
    for (int j = 0; j < prefix_len; ++j) {
      if (generated[i + j] != prefix[j]) {
        match = false;
        break;
      }
    }
    if (match) {
      blocked.insert(generated[i + prefix_len]);
    }
  }
  return blocked;
}

int sample_token(const std::vector<half>& logits, const std::vector<int>& generated, float temperature,
                 int top_k, float top_p, float min_p, int no_repeat_ngram_size,
                 float presence_penalty, float frequency_penalty, float repetition_penalty) {
  if (temperature <= 1e-5f) {
    return argmax_host(logits);
  }

  std::vector<float> adjusted(logits.size());
  for (std::size_t i = 0; i < logits.size(); ++i) {
    adjusted[i] = __half2float(logits[i]);
  }

  if (repetition_penalty > 1.0f) {
    for (int tok : generated) {
      if (tok >= 0 && tok < static_cast<int>(adjusted.size())) {
        if (adjusted[tok] > 0.0f) {
          adjusted[tok] /= repetition_penalty;
        } else {
          adjusted[tok] *= repetition_penalty;
        }
      }
    }
  }

  if (presence_penalty > 0.0f || frequency_penalty > 0.0f) {
    std::unordered_map<int, int> freq;
    for (int tok : generated) {
      if (tok >= 0 && tok < static_cast<int>(adjusted.size())) {
        ++freq[tok];
      }
    }
    for (const auto& kv : freq) {
      float penalty = 0.0f;
      if (presence_penalty > 0.0f) {
        penalty += presence_penalty;
      }
      if (frequency_penalty > 0.0f) {
        penalty += frequency_penalty * static_cast<float>(kv.second);
      }
      adjusted[kv.first] -= penalty;
    }
  }

  if (no_repeat_ngram_size > 0) {
    auto blocked = collect_blocked_ngrams(generated, no_repeat_ngram_size);
    for (int tok : blocked) {
      if (tok >= 0 && tok < static_cast<int>(adjusted.size())) {
        adjusted[tok] = -1e30f;
      }
    }
  }

  std::vector<int> idx(adjusted.size());
  std::iota(idx.begin(), idx.end(), 0);
  int pre_keep = std::min<int>(std::max(1, top_k), static_cast<int>(idx.size()));
  std::partial_sort(idx.begin(), idx.begin() + pre_keep, idx.end(),
                    [&](int a, int b) { return adjusted[a] > adjusted[b]; });

  float max_logit = -1e30f;
  for (int i = 0; i < pre_keep; ++i) {
    max_logit = std::max(max_logit, adjusted[idx[i]] / temperature);
  }

  std::vector<float> probs(pre_keep);
  float sum = 0.0f;
  for (int i = 0; i < pre_keep; ++i) {
    float p = std::exp((adjusted[idx[i]] / temperature) - max_logit);
    probs[i] = p;
    sum += p;
  }
  if (sum <= 0.0f) {
    return idx[0];
  }
  for (float& p : probs) {
    p /= sum;
  }

  float p_cap = std::min(1.0f, std::max(0.0f, top_p));
  if (p_cap < 1.0f) {
    std::vector<int> order(pre_keep);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return probs[a] > probs[b]; });

    float cum = 0.0f;
    int nucleus = 0;
    for (int oi : order) {
      cum += probs[oi];
      ++nucleus;
      if (cum >= p_cap) {
        break;
      }
    }
    nucleus = std::max(1, nucleus);

    std::vector<int> new_idx;
    std::vector<float> new_probs;
    new_idx.reserve(nucleus);
    new_probs.reserve(nucleus);
    float newsum = 0.0f;
    for (int i = 0; i < nucleus; ++i) {
      int oi = order[i];
      new_idx.push_back(idx[oi]);
      new_probs.push_back(probs[oi]);
      newsum += probs[oi];
    }
    for (float& p : new_probs) {
      p /= newsum;
    }
    idx.swap(new_idx);
    probs.swap(new_probs);
  } else {
    idx.resize(pre_keep);
  }

  float min_p_clamped = std::min(1.0f, std::max(0.0f, min_p));
  if (min_p_clamped > 0.0f && !probs.empty()) {
    float pmax = 0.0f;
    for (float p : probs) {
      pmax = std::max(pmax, p);
    }
    float floor = pmax * min_p_clamped;

    std::vector<int> new_idx;
    std::vector<float> new_probs;
    new_idx.reserve(idx.size());
    new_probs.reserve(probs.size());
    float newsum = 0.0f;
    for (std::size_t i = 0; i < probs.size(); ++i) {
      if (probs[i] >= floor) {
        new_idx.push_back(idx[i]);
        new_probs.push_back(probs[i]);
        newsum += probs[i];
      }
    }
    if (new_idx.empty()) {
      return idx[0];
    }
    for (float& p : new_probs) {
      p /= newsum;
    }
    idx.swap(new_idx);
    probs.swap(new_probs);
  }

  static thread_local std::mt19937 rng(std::random_device{}());
  std::discrete_distribution<int> dist(probs.begin(), probs.end());
  int chosen = dist(rng);
  return idx[chosen];
}

inline float silu(float x) {
  return x / (1.0f + std::exp(-x));
}

void l2norm_inplace(std::vector<float>& x, int offset, int n) {
  float ss = 0.0f;
  for (int i = 0; i < n; ++i) {
    float v = x[offset + i];
    ss += v * v;
  }
  float inv = 1.0f / std::sqrt(ss + 1e-6f);
  for (int i = 0; i < n; ++i) {
    x[offset + i] *= inv;
  }
}

}  // namespace

QwenMiniModel::QwenMiniModel() {}

QwenMiniModel::~QwenMiniModel() { free_all(); }

bool QwenMiniModel::load(const std::string& path) {
  free_all();

  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::cerr << "Open model failed: " << path << "\n";
    return false;
  }

  char magic[8] = {};
  read_exact(in, magic, 8);
  if (std::string(magic, 8) != std::string(kMagic, 8)) {
    std::cerr << "Bad model magic\n";
    return false;
  }

  int32_t version = 0;
  read_exact(in, reinterpret_cast<char*>(&version), sizeof(version));
  if (version != 1 && version != 2 && version != 3 && version != 4) {
    std::cerr << "Unsupported model version: " << version << "\n";
    return false;
  }
  cfg_.version = version;

  // qmini 头部采用固定顺序：先公共主干配置，再按版本追加扩展字段。
  // v1/v2/v3 只含 full-attention 所需参数；v4 追加 qwen3.5 linear-attention 参数。 

  read_exact(in, reinterpret_cast<char*>(&cfg_.vocab_size), sizeof(cfg_.vocab_size));
  read_exact(in, reinterpret_cast<char*>(&cfg_.hidden_size), sizeof(cfg_.hidden_size));
  read_exact(in, reinterpret_cast<char*>(&cfg_.intermediate_size), sizeof(cfg_.intermediate_size));
  read_exact(in, reinterpret_cast<char*>(&cfg_.num_layers), sizeof(cfg_.num_layers));
  read_exact(in, reinterpret_cast<char*>(&cfg_.num_heads), sizeof(cfg_.num_heads));
  read_exact(in, reinterpret_cast<char*>(&cfg_.num_kv_heads), sizeof(cfg_.num_kv_heads));
  if (version >= 2) {
    read_exact(in, reinterpret_cast<char*>(&cfg_.head_dim), sizeof(cfg_.head_dim));
  } else {
    cfg_.head_dim = cfg_.hidden_size / cfg_.num_heads;
  }

  if (version >= 4) {
    read_exact(in, reinterpret_cast<char*>(&cfg_.linear_num_key_heads), sizeof(cfg_.linear_num_key_heads));
    read_exact(in, reinterpret_cast<char*>(&cfg_.linear_num_value_heads), sizeof(cfg_.linear_num_value_heads));
    read_exact(in, reinterpret_cast<char*>(&cfg_.linear_key_head_dim), sizeof(cfg_.linear_key_head_dim));
    read_exact(in, reinterpret_cast<char*>(&cfg_.linear_value_head_dim), sizeof(cfg_.linear_value_head_dim));
    read_exact(in, reinterpret_cast<char*>(&cfg_.linear_conv_kernel_dim), sizeof(cfg_.linear_conv_kernel_dim));
  }

  read_exact(in, reinterpret_cast<char*>(&cfg_.max_seq_len), sizeof(cfg_.max_seq_len));
  read_exact(in, reinterpret_cast<char*>(&cfg_.rms_norm_eps), sizeof(cfg_.rms_norm_eps));
  read_exact(in, reinterpret_cast<char*>(&cfg_.rope_theta), sizeof(cfg_.rope_theta));

  if (cfg_.head_dim <= 0) {
    throw std::runtime_error("head_dim must be > 0");
  }

  if (version >= 4) {
    // v4 显式保存每一层类型：0=full_attention, 1=linear_attention。
    cfg_.layer_types.resize(cfg_.num_layers);
    read_exact(in, reinterpret_cast<char*>(cfg_.layer_types.data()),
               static_cast<std::streamsize>(cfg_.num_layers * sizeof(int32_t)));
  } else {
    cfg_.layer_types.assign(cfg_.num_layers, kLayerTypeFullAttention);
  }

  const std::size_t hidden = static_cast<std::size_t>(cfg_.hidden_size);
  const std::size_t inter = static_cast<std::size_t>(cfg_.intermediate_size);
  const std::size_t vocab = static_cast<std::size_t>(cfg_.vocab_size);
  const std::size_t q_dim = static_cast<std::size_t>(cfg_.num_heads) * cfg_.head_dim;
  const std::size_t kv_dim = static_cast<std::size_t>(cfg_.num_kv_heads) * cfg_.head_dim;
  const std::size_t linear_key_dim =
    static_cast<std::size_t>(cfg_.linear_num_key_heads) * cfg_.linear_key_head_dim;
  const std::size_t linear_val_dim =
    static_cast<std::size_t>(cfg_.linear_num_value_heads) * cfg_.linear_value_head_dim;
  const std::size_t linear_conv_dim = linear_key_dim * 2 + linear_val_dim;

  tok_embeddings_ = read_tensor_to_device(in, vocab * hidden, "tok_embeddings");

  layers_.resize(cfg_.num_layers);
  for (int i = 0; i < cfg_.num_layers; ++i) {
    auto& l = layers_[i];
    l.is_linear = (cfg_.layer_types[i] == kLayerTypeLinearAttention);
    l.attn_norm = read_tensor_to_device(in, hidden, "layer.attn_norm");

    if (l.is_linear) {
      // linear-attention 层：完整加载 GatedDeltaNet 权重。
      l.linear_in_qkv = read_tensor_to_device(in, hidden * linear_conv_dim, "layer.linear_in_qkv");
      l.linear_in_z = read_tensor_to_device(in, hidden * linear_val_dim, "layer.linear_in_z");
      l.linear_in_b = read_tensor_to_device(in, hidden * cfg_.linear_num_value_heads, "layer.linear_in_b");
      l.linear_in_a = read_tensor_to_device(in, hidden * cfg_.linear_num_value_heads, "layer.linear_in_a");
      l.linear_conv1d_w = read_tensor_to_device(in, linear_conv_dim * cfg_.linear_conv_kernel_dim,
                                                "layer.linear_conv1d_w");
      l.linear_dt_bias = read_tensor_to_device(in, cfg_.linear_num_value_heads, "layer.linear_dt_bias");
      l.linear_a_log = read_tensor_to_device(in, cfg_.linear_num_value_heads, "layer.linear_a_log");
      l.linear_norm_w = read_tensor_to_device(in, cfg_.linear_value_head_dim, "layer.linear_norm_w");
      l.linear_out_proj = read_tensor_to_device(in, linear_val_dim * hidden, "layer.linear_out_proj");
    } else {
      // full-attention 层：v4 的 q_proj 维度为 2*q_dim（query + gate）。
      std::size_t q_proj_dim = q_dim;
      if (version >= 4) {
        q_proj_dim = q_dim * 2;
      }
      l.wq = read_tensor_to_device(in, hidden * q_proj_dim, "layer.wq");
      l.wk = read_tensor_to_device(in, hidden * kv_dim, "layer.wk");
      l.wv = read_tensor_to_device(in, hidden * kv_dim, "layer.wv");
      if (version >= 3) {
        l.q_norm = read_tensor_to_device(in, cfg_.head_dim, "layer.q_norm");
        l.k_norm = read_tensor_to_device(in, cfg_.head_dim, "layer.k_norm");
      }
      l.wo = read_tensor_to_device(in, q_dim * hidden, "layer.wo");
    }

    l.ffn_norm = read_tensor_to_device(in, hidden, "layer.ffn_norm");
    l.w_gate = read_tensor_to_device(in, hidden * inter, "layer.w_gate");
    l.w_up = read_tensor_to_device(in, hidden * inter, "layer.w_up");
    l.w_down = read_tensor_to_device(in, inter * hidden, "layer.w_down");
  }

  final_norm_ = read_tensor_to_device(in, hidden, "final_norm");
  lm_head_ = read_tensor_to_device(in, hidden * vocab, "lm_head");

  alloc_runtime();
  loaded_ = true;
  return true;
}

bool QwenMiniModel::alloc_runtime() {
  const std::size_t hidden = static_cast<std::size_t>(cfg_.hidden_size);
  const std::size_t inter = static_cast<std::size_t>(cfg_.intermediate_size);
  const std::size_t vocab = static_cast<std::size_t>(cfg_.vocab_size);
  const std::size_t q_dim = static_cast<std::size_t>(cfg_.num_heads) * cfg_.head_dim;
  const std::size_t kv_dim = static_cast<std::size_t>(cfg_.num_kv_heads) * cfg_.head_dim;

  const std::size_t linear_key_dim =
    static_cast<std::size_t>(cfg_.linear_num_key_heads) * std::max(1, cfg_.linear_key_head_dim);
  const std::size_t linear_val_dim =
    static_cast<std::size_t>(cfg_.linear_num_value_heads) * std::max(1, cfg_.linear_value_head_dim);
  const std::size_t linear_conv_dim = linear_key_dim * 2 + linear_val_dim;

  CUBLAS_CHECK(cublasCreate(&cublas_));

  // 主干中间激活缓存（单 token decode）：
  // x/x_norm/x_resid 分别表示层输入、归一化后、残差分支缓存。

  CUDA_CHECK(cudaMalloc(&x_, hidden * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&x_norm_, hidden * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&x_resid_, hidden * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&q_, q_dim * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&q_gate_, q_dim * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&k_, kv_dim * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&v_, kv_dim * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&context_, q_dim * sizeof(half)));

  std::size_t mixed_size = std::max<std::size_t>(linear_conv_dim, q_dim * 2);
  // linear_mixed_qkv_ 在两条路径复用：
  // 1) full-attention v4 临时承接 q_proj(2*q_dim)
  // 2) linear-attention 承接 in_proj_qkv(conv_dim)
  CUDA_CHECK(cudaMalloc(&linear_mixed_qkv_, std::max<std::size_t>(1, mixed_size) * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&linear_z_, std::max<std::size_t>(1, linear_val_dim) * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&linear_b_, std::max(1, cfg_.linear_num_value_heads) * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&linear_a_, std::max(1, cfg_.linear_num_value_heads) * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&linear_out_, std::max<std::size_t>(1, linear_val_dim) * sizeof(half)));

  CUDA_CHECK(cudaMalloc(&ffn_gate_, inter * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&ffn_up_, inter * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&ffn_hidden_, inter * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&logits_, vocab * sizeof(half)));

  caches_.resize(cfg_.num_layers);
  for (int i = 0; i < cfg_.num_layers; ++i) {
    if (cfg_.layer_types[i] == kLayerTypeFullAttention) {
      // full-attention：按 [seq, kv_dim] 线性存储 K/V cache。
      CUDA_CHECK(cudaMalloc(&caches_[i].k, cfg_.max_seq_len * kv_dim * sizeof(half)));
      CUDA_CHECK(cudaMalloc(&caches_[i].v, cfg_.max_seq_len * kv_dim * sizeof(half)));
    } else {
      // linear-attention：
      // conv_state 形状 = [conv_dim, kernel_size]
      // recurrent_state 形状 = [num_v_heads, key_dim_per_head, value_dim_per_head]
      int conv_dim_i = cfg_.linear_num_key_heads * cfg_.linear_key_head_dim * 2 +
                       cfg_.linear_num_value_heads * cfg_.linear_value_head_dim;
      caches_[i].linear_conv_state.assign(conv_dim_i * cfg_.linear_conv_kernel_dim, 0.0f);
      caches_[i].linear_recurrent_state.assign(
        cfg_.linear_num_value_heads * cfg_.linear_key_head_dim * cfg_.linear_value_head_dim, 0.0f);
    }
  }

  int max_pages = (cfg_.max_seq_len + page_size_ - 1) / page_size_;
  page_table_host_.resize(max_pages);
  for (int i = 0; i < max_pages; ++i) {
    page_table_host_[i] = i;
  }
  CUDA_CHECK(cudaMalloc(&page_table_dev_, max_pages * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(page_table_dev_, page_table_host_.data(), max_pages * sizeof(int),
                        cudaMemcpyHostToDevice));
  return true;
}

void QwenMiniModel::free_all() {
  auto safe_free = [](void* p) {
    if (p) {
      cudaFree(p);
    }
  };

  safe_free(tok_embeddings_);
  tok_embeddings_ = nullptr;

  for (auto& l : layers_) {
    safe_free(l.attn_norm);
    safe_free(l.wq);
    safe_free(l.wk);
    safe_free(l.wv);
    safe_free(l.q_norm);
    safe_free(l.k_norm);
    safe_free(l.wo);

    safe_free(l.linear_in_qkv);
    safe_free(l.linear_in_z);
    safe_free(l.linear_in_b);
    safe_free(l.linear_in_a);
    safe_free(l.linear_conv1d_w);
    safe_free(l.linear_dt_bias);
    safe_free(l.linear_a_log);
    safe_free(l.linear_norm_w);
    safe_free(l.linear_out_proj);

    safe_free(l.ffn_norm);
    safe_free(l.w_gate);
    safe_free(l.w_up);
    safe_free(l.w_down);
  }
  layers_.clear();

  safe_free(final_norm_);
  safe_free(lm_head_);
  final_norm_ = nullptr;
  lm_head_ = nullptr;

  for (auto& c : caches_) {
    safe_free(c.k);
    safe_free(c.v);
    c.linear_conv_state.clear();
    c.linear_recurrent_state.clear();
  }
  caches_.clear();

  safe_free(page_table_dev_);
  page_table_dev_ = nullptr;
  page_table_host_.clear();

  safe_free(x_);
  safe_free(x_norm_);
  safe_free(x_resid_);
  safe_free(q_);
  safe_free(q_gate_);
  safe_free(k_);
  safe_free(v_);
  safe_free(context_);

  safe_free(linear_mixed_qkv_);
  safe_free(linear_z_);
  safe_free(linear_b_);
  safe_free(linear_a_);
  safe_free(linear_out_);

  safe_free(ffn_gate_);
  safe_free(ffn_up_);
  safe_free(ffn_hidden_);
  safe_free(logits_);

  x_ = nullptr;
  x_norm_ = nullptr;
  x_resid_ = nullptr;
  q_ = nullptr;
  q_gate_ = nullptr;
  k_ = nullptr;
  v_ = nullptr;
  context_ = nullptr;
  linear_mixed_qkv_ = nullptr;
  linear_z_ = nullptr;
  linear_b_ = nullptr;
  linear_a_ = nullptr;
  linear_out_ = nullptr;
  ffn_gate_ = nullptr;
  ffn_up_ = nullptr;
  ffn_hidden_ = nullptr;
  logits_ = nullptr;

  if (cublas_) {
    cublasDestroy(cublas_);
    cublas_ = nullptr;
  }

  loaded_ = false;
}

void QwenMiniModel::gemv(const half* x, const half* w, int in_dim, int out_dim, half* out) {
  static const float alpha = 1.0f;
  static const float beta = 0.0f;
  CUBLAS_CHECK(cublasGemmEx(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                            out_dim, 1, in_dim,
                            &alpha,
                            w, CUDA_R_16F, out_dim,
                            x, CUDA_R_16F, in_dim,
                            &beta,
                            out, CUDA_R_16F, out_dim,
                            CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

int QwenMiniModel::decode_next(int token_id, int position, std::vector<half>* host_logits) {
  const int hidden = cfg_.hidden_size;
  const int inter = cfg_.intermediate_size;
  const int head_dim = cfg_.head_dim;
  const int q_dim = cfg_.num_heads * head_dim;
  const int kv_dim = cfg_.num_kv_heads * head_dim;

  const int linear_kh = cfg_.linear_num_key_heads;
  const int linear_vh = cfg_.linear_num_value_heads;
  const int linear_kd = cfg_.linear_key_head_dim;
  const int linear_vd = cfg_.linear_value_head_dim;
  const int linear_key_dim = linear_kh * linear_kd;
  const int linear_val_dim = linear_vh * linear_vd;
  const int linear_conv_dim = linear_key_dim * 2 + linear_val_dim;

  launch_embedding_lookup(tok_embeddings_, cfg_.vocab_size, hidden, token_id, x_);

  for (int i = 0; i < cfg_.num_layers; ++i) {
    const auto& w = layers_[i];
    auto& cache = caches_[i];

    CUDA_CHECK(cudaMemcpy(x_resid_, x_, hidden * sizeof(half), cudaMemcpyDeviceToDevice));
    launch_rmsnorm(x_, w.attn_norm, hidden, cfg_.rms_norm_eps, x_norm_);

    if (!w.is_linear) {
      // ---------------- full-attention 路径 ----------------
      // q_proj: [hidden] -> [2*q_dim]，按每个 head 的 [q_chunk, gate_chunk] 交错排列。
      int q_proj_dim = (cfg_.version >= 4) ? (q_dim * 2) : q_dim;
      if (cfg_.version >= 4) {
        gemv(x_norm_, w.wq, hidden, q_proj_dim, linear_mixed_qkv_);
        std::vector<half> qcat(q_proj_dim);
        std::vector<half> qv(q_dim);
        std::vector<half> gv(q_dim);
        CUDA_CHECK(cudaMemcpy(qcat.data(), linear_mixed_qkv_, q_proj_dim * sizeof(half), cudaMemcpyDeviceToHost));
        for (int h = 0; h < cfg_.num_heads; ++h) {
          int src = h * (2 * head_dim);
          int dst = h * head_dim;
          // 逐 head 反交错拆分，严格对齐官方实现。
          for (int d = 0; d < head_dim; ++d) {
            qv[dst + d] = qcat[src + d];
            gv[dst + d] = qcat[src + head_dim + d];
          }
        }
        CUDA_CHECK(cudaMemcpy(q_, qv.data(), q_dim * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(q_gate_, gv.data(), q_dim * sizeof(half), cudaMemcpyHostToDevice));
      } else {
        gemv(x_norm_, w.wq, hidden, q_dim, q_);
      }
      gemv(x_norm_, w.wk, hidden, kv_dim, k_);
      gemv(x_norm_, w.wv, hidden, kv_dim, v_);

      if (w.q_norm) {
        launch_head_rmsnorm(q_, w.q_norm, cfg_.num_heads, head_dim, cfg_.rms_norm_eps, q_);
      }
      if (w.k_norm) {
        launch_head_rmsnorm(k_, w.k_norm, cfg_.num_kv_heads, head_dim, cfg_.rms_norm_eps, k_);
      }

      launch_rope_inplace(q_, cfg_.num_heads, head_dim, position, cfg_.rope_theta);
      launch_rope_inplace(k_, cfg_.num_kv_heads, head_dim, position, cfg_.rope_theta);

      // KV cache 追加当前位置向量。
      std::size_t kv_off = kv_cache_offset(position, kv_dim);
      CUDA_CHECK(cudaMemcpy(cache.k + kv_off, k_, kv_dim * sizeof(half), cudaMemcpyDeviceToDevice));
      CUDA_CHECK(cudaMemcpy(cache.v + kv_off, v_, kv_dim * sizeof(half), cudaMemcpyDeviceToDevice));

      int seq_len = position + 1;
      if (seq_len <= page_size_ * 8) {
        launch_flash_attention(q_, cache.k, cache.v, cfg_.num_heads, cfg_.num_kv_heads, head_dim, seq_len,
                               context_);
      } else {
        launch_paged_attention(q_, cache.k, cache.v, page_table_dev_, page_size_, cfg_.num_heads,
                               cfg_.num_kv_heads, head_dim, seq_len, context_);
      }

      if (cfg_.version >= 4) {
        // 官方 qwen3.5：attention 输出先乘 sigmoid(gate)，再 o_proj。
        launch_sigmoid_mul_inplace(context_, q_gate_, q_dim);
      }
      gemv(context_, w.wo, q_dim, hidden, x_);
    } else {
      // ---------------- linear-attention 路径 ----------------
      // 对齐 Qwen3.5 GatedDeltaNet 的 decode 递推实现。
      gemv(x_norm_, w.linear_in_qkv, hidden, linear_conv_dim, linear_mixed_qkv_);
      gemv(x_norm_, w.linear_in_z, hidden, linear_val_dim, linear_z_);
      gemv(x_norm_, w.linear_in_b, hidden, linear_vh, linear_b_);
      gemv(x_norm_, w.linear_in_a, hidden, linear_vh, linear_a_);

      std::vector<half> h_qkv(linear_conv_dim);
      std::vector<half> h_z(linear_val_dim);
      std::vector<half> h_b(linear_vh);
      std::vector<half> h_a(linear_vh);
      std::vector<half> h_conv_w(linear_conv_dim * cfg_.linear_conv_kernel_dim);
      std::vector<half> h_dt_bias(linear_vh);
      std::vector<half> h_a_log(linear_vh);
      std::vector<half> h_norm_w(linear_vd);

      CUDA_CHECK(cudaMemcpy(h_qkv.data(), linear_mixed_qkv_, linear_conv_dim * sizeof(half), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h_z.data(), linear_z_, linear_val_dim * sizeof(half), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h_b.data(), linear_b_, linear_vh * sizeof(half), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h_a.data(), linear_a_, linear_vh * sizeof(half), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h_conv_w.data(), w.linear_conv1d_w,
                            linear_conv_dim * cfg_.linear_conv_kernel_dim * sizeof(half),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h_dt_bias.data(), w.linear_dt_bias, linear_vh * sizeof(half), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h_a_log.data(), w.linear_a_log, linear_vh * sizeof(half), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h_norm_w.data(), w.linear_norm_w, linear_vd * sizeof(half), cudaMemcpyDeviceToHost));

      std::vector<float> mixed(linear_conv_dim, 0.0f);
      for (int c = 0; c < linear_conv_dim; ++c) {
        // depthwise causal conv 状态左移一位，写入当前 token 投影值。
        for (int t = 0; t < cfg_.linear_conv_kernel_dim - 1; ++t) {
          cache.linear_conv_state[c * cfg_.linear_conv_kernel_dim + t] =
            cache.linear_conv_state[c * cfg_.linear_conv_kernel_dim + t + 1];
        }
        cache.linear_conv_state[c * cfg_.linear_conv_kernel_dim + (cfg_.linear_conv_kernel_dim - 1)] =
          __half2float(h_qkv[c]);

        float acc = 0.0f;
        for (int t = 0; t < cfg_.linear_conv_kernel_dim; ++t) {
          float wv = __half2float(h_conv_w[c * cfg_.linear_conv_kernel_dim + t]);
          float sv = cache.linear_conv_state[c * cfg_.linear_conv_kernel_dim + t];
          acc += wv * sv;
        }
        mixed[c] = silu(acc);
      }

      std::vector<float> q(linear_key_dim);
      std::vector<float> k(linear_key_dim);
      std::vector<float> v(linear_val_dim);
      for (int j = 0; j < linear_key_dim; ++j) {
        q[j] = mixed[j];
        k[j] = mixed[linear_key_dim + j];
      }
      for (int j = 0; j < linear_val_dim; ++j) {
        v[j] = mixed[2 * linear_key_dim + j];
      }

      for (int h = 0; h < linear_kh; ++h) {
        // 与官方 fallback 一致：q/k 在 head 维做 L2Norm。
        l2norm_inplace(q, h * linear_kd, linear_kd);
        l2norm_inplace(k, h * linear_kd, linear_kd);
      }

      int repeat = linear_vh / linear_kh;
      std::vector<float> q_rep(linear_vh * linear_kd);
      std::vector<float> k_rep(linear_vh * linear_kd);
      for (int h = 0; h < linear_vh; ++h) {
        int src_h = h / repeat;
        for (int d = 0; d < linear_kd; ++d) {
          q_rep[h * linear_kd + d] = q[src_h * linear_kd + d] * (1.0f / std::sqrt(static_cast<float>(linear_kd)));
          k_rep[h * linear_kd + d] = k[src_h * linear_kd + d];
        }
      }

      std::vector<float> out(linear_val_dim, 0.0f);
      for (int h = 0; h < linear_vh; ++h) {
        // 递推门控参数：
        // beta = sigmoid(b)
        // g = exp(-exp(A_log) * softplus(a + dt_bias))
        float beta = 1.0f / (1.0f + std::exp(-__half2float(h_b[h])));
        float a = __half2float(h_a[h]) + __half2float(h_dt_bias[h]);
        float softp = std::log1pf(std::exp(a));
        float g = std::exp(-std::exp(__half2float(h_a_log[h])) * softp);

        float* state = cache.linear_recurrent_state.data() + h * linear_kd * linear_vd;

        for (int kd = 0; kd < linear_kd; ++kd) {
          for (int vd = 0; vd < linear_vd; ++vd) {
            // recurrent state 衰减。
            state[kd * linear_vd + vd] *= g;
          }
        }

        std::vector<float> kv_mem(linear_vd, 0.0f);
        for (int vd = 0; vd < linear_vd; ++vd) {
          float m = 0.0f;
          for (int kd = 0; kd < linear_kd; ++kd) {
            m += state[kd * linear_vd + vd] * k_rep[h * linear_kd + kd];
          }
          kv_mem[vd] = m;
        }

        for (int vd = 0; vd < linear_vd; ++vd) {
          // delta rule：state += k * ((v - kv_mem) * beta)
          float delta = (v[h * linear_vd + vd] - kv_mem[vd]) * beta;
          for (int kd = 0; kd < linear_kd; ++kd) {
            state[kd * linear_vd + vd] += k_rep[h * linear_kd + kd] * delta;
          }
        }

        for (int vd = 0; vd < linear_vd; ++vd) {
          float y = 0.0f;
          for (int kd = 0; kd < linear_kd; ++kd) {
            y += state[kd * linear_vd + vd] * q_rep[h * linear_kd + kd];
          }
          out[h * linear_vd + vd] = y;
        }
      }

      std::vector<half> out_h(linear_val_dim);
      for (int h = 0; h < linear_vh; ++h) {
        // RMSNormGated：先按 value head 归一化，再乘 norm_w 与 silu(z)。
        float ss = 0.0f;
        for (int vd = 0; vd < linear_vd; ++vd) {
          float y = out[h * linear_vd + vd];
          ss += y * y;
        }
        float inv = 1.0f / std::sqrt(ss / static_cast<float>(linear_vd) + cfg_.rms_norm_eps);
        for (int vd = 0; vd < linear_vd; ++vd) {
          float normed = out[h * linear_vd + vd] * inv;
          float gate = silu(__half2float(h_z[h * linear_vd + vd]));
          float ww = __half2float(h_norm_w[vd]);
          out_h[h * linear_vd + vd] = __float2half(normed * ww * gate);
        }
      }

      CUDA_CHECK(cudaMemcpy(linear_out_, out_h.data(), linear_val_dim * sizeof(half), cudaMemcpyHostToDevice));
      gemv(linear_out_, w.linear_out_proj, linear_val_dim, hidden, x_);
    }

    // attention/linear mixer 输出与残差相加。
    launch_add_inplace(x_, x_resid_, hidden);

    CUDA_CHECK(cudaMemcpy(x_resid_, x_, hidden * sizeof(half), cudaMemcpyDeviceToDevice));
    launch_rmsnorm(x_, w.ffn_norm, hidden, cfg_.rms_norm_eps, x_norm_);

    gemv(x_norm_, w.w_gate, hidden, inter, ffn_gate_);
    gemv(x_norm_, w.w_up, hidden, inter, ffn_up_);
    launch_swiglu(ffn_gate_, ffn_up_, inter, ffn_hidden_);
    gemv(ffn_hidden_, w.w_down, inter, hidden, x_);
    // MLP 残差。
    launch_add_inplace(x_, x_resid_, hidden);
  }

  // 末层 norm + lm_head，得到当前位置 logits。
  launch_rmsnorm(x_, final_norm_, hidden, cfg_.rms_norm_eps, x_norm_);
  gemv(x_norm_, lm_head_, hidden, cfg_.vocab_size, logits_);

  std::vector<half> local_logits;
  std::vector<half>& out = host_logits ? *host_logits : local_logits;
  out.resize(cfg_.vocab_size);
  CUDA_CHECK(cudaMemcpy(out.data(), logits_, cfg_.vocab_size * sizeof(half), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());
  return argmax_host(out);
}

std::vector<int> QwenMiniModel::generate(const std::vector<int>& input_ids, int max_new_tokens, int eos_id,
                                         float temperature, int top_k, float top_p,
                                         float min_p, float temp_decay, int greedy_after,
                                         int no_repeat_ngram_size,
                                         float presence_penalty, float frequency_penalty,
                                         float repetition_penalty,
                                         int dump_topk, int dump_steps,
                                         float* elapsed_ms) {
  if (!loaded_) {
    throw std::runtime_error("Model not loaded");
  }
  if (input_ids.empty()) {
    throw std::runtime_error("input_ids is empty");
  }

  if (static_cast<int>(input_ids.size()) + max_new_tokens > cfg_.max_seq_len) {
    throw std::runtime_error("input length + max_new_tokens exceeds max_seq_len");
  }

  std::vector<int> all = input_ids;

  auto t0 = std::chrono::high_resolution_clock::now();

  for (int pos = 0; pos < static_cast<int>(input_ids.size()) - 1; ++pos) {
    (void)decode_next(input_ids[pos], pos);
  }

  int cur = input_ids.back();
  int pos = static_cast<int>(input_ids.size()) - 1;
  std::vector<half> step_logits;
  float temp_now = temperature;
  for (int i = 0; i < max_new_tokens; ++i) {
    (void)decode_next(cur, pos, &step_logits);

    if (dump_topk > 0 && (dump_steps <= 0 || i < dump_steps)) {
      print_step_topk(i, step_logits, dump_topk);
    }

    int next = 0;
    if (greedy_after >= 0 && i >= greedy_after) {
      next = argmax_host(step_logits);
    } else {
      next = sample_token(step_logits, all, temp_now, top_k, top_p, min_p,
                          no_repeat_ngram_size, presence_penalty, frequency_penalty,
                          repetition_penalty);
      if (temp_decay > 0.0f) {
        temp_now = std::max(0.05f, temp_now * temp_decay);
      }
    }
    all.push_back(next);
    cur = next;
    ++pos;
    if (next == eos_id) {
      break;
    }
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  if (elapsed_ms) {
    *elapsed_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
  }
  return all;
}
