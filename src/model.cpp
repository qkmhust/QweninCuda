#include "model.hpp"

#include "common.hpp"
#include "kernels.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace {

constexpr char kMagic[] = "QWENMINI";

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

  // Nucleus sampling over top-k candidates.
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

  // Min-p filtering based on the best candidate probability.
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
  if (version != 1 && version != 2 && version != 3) {
    std::cerr << "Unsupported model version: " << version << "\n";
    return false;
  }

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
  read_exact(in, reinterpret_cast<char*>(&cfg_.max_seq_len), sizeof(cfg_.max_seq_len));
  read_exact(in, reinterpret_cast<char*>(&cfg_.rms_norm_eps), sizeof(cfg_.rms_norm_eps));
  read_exact(in, reinterpret_cast<char*>(&cfg_.rope_theta), sizeof(cfg_.rope_theta));

  if (cfg_.head_dim <= 0) {
    throw std::runtime_error("head_dim must be > 0");
  }

  const std::size_t hidden = static_cast<std::size_t>(cfg_.hidden_size);
  const std::size_t inter = static_cast<std::size_t>(cfg_.intermediate_size);
  const std::size_t vocab = static_cast<std::size_t>(cfg_.vocab_size);
  const std::size_t q_dim = static_cast<std::size_t>(cfg_.num_heads) * cfg_.head_dim;
  const std::size_t kv_dim = static_cast<std::size_t>(cfg_.num_kv_heads) * cfg_.head_dim;

  tok_embeddings_ = read_tensor_to_device(in, vocab * hidden, "tok_embeddings");

  layers_.resize(cfg_.num_layers);
  for (int i = 0; i < cfg_.num_layers; ++i) {
    auto& l = layers_[i];
    l.attn_norm = read_tensor_to_device(in, hidden, "layer.attn_norm");
    l.wq = read_tensor_to_device(in, hidden * q_dim, "layer.wq");
    l.wk = read_tensor_to_device(in, hidden * kv_dim, "layer.wk");
    l.wv = read_tensor_to_device(in, hidden * kv_dim, "layer.wv");
    if (version >= 3) {
      l.q_norm = read_tensor_to_device(in, cfg_.head_dim, "layer.q_norm");
      l.k_norm = read_tensor_to_device(in, cfg_.head_dim, "layer.k_norm");
    }
    l.wo = read_tensor_to_device(in, q_dim * hidden, "layer.wo");
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
  const std::size_t head_dim = cfg_.head_dim;
  const std::size_t q_dim = static_cast<std::size_t>(cfg_.num_heads) * head_dim;
  const std::size_t kv_dim = static_cast<std::size_t>(cfg_.num_kv_heads) * head_dim;

  CUBLAS_CHECK(cublasCreate(&cublas_));

  CUDA_CHECK(cudaMalloc(&x_, hidden * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&x_norm_, hidden * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&x_resid_, hidden * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&q_, q_dim * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&k_, kv_dim * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&v_, kv_dim * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&context_, q_dim * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&ffn_gate_, inter * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&ffn_up_, inter * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&ffn_hidden_, inter * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&logits_, vocab * sizeof(half)));

  caches_.resize(cfg_.num_layers);
  for (int i = 0; i < cfg_.num_layers; ++i) {
    CUDA_CHECK(cudaMalloc(&caches_[i].k, cfg_.max_seq_len * kv_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&caches_[i].v, cfg_.max_seq_len * kv_dim * sizeof(half)));
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
  }
  caches_.clear();

  safe_free(page_table_dev_);
  page_table_dev_ = nullptr;
  page_table_host_.clear();

  safe_free(x_);
  safe_free(x_norm_);
  safe_free(x_resid_);
  safe_free(q_);
  safe_free(k_);
  safe_free(v_);
  safe_free(context_);
  safe_free(ffn_gate_);
  safe_free(ffn_up_);
  safe_free(ffn_hidden_);
  safe_free(logits_);

  x_ = nullptr;
  x_norm_ = nullptr;
  x_resid_ = nullptr;
  q_ = nullptr;
  k_ = nullptr;
  v_ = nullptr;
  context_ = nullptr;
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

  // 1) 当前 token 的 embedding -> x
  launch_embedding_lookup(tok_embeddings_, cfg_.vocab_size, hidden, token_id, x_);

  for (int i = 0; i < cfg_.num_layers; ++i) {
    const auto& w = layers_[i];
    auto& cache = caches_[i];

    CUDA_CHECK(cudaMemcpy(x_resid_, x_, hidden * sizeof(half), cudaMemcpyDeviceToDevice));
    launch_rmsnorm(x_, w.attn_norm, hidden, cfg_.rms_norm_eps, x_norm_);

    gemv(x_norm_, w.wq, hidden, q_dim, q_);
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

    // 2) 写入 KV cache，供后续自回归步复用
    CUDA_CHECK(cudaMemcpy(cache.k + static_cast<std::size_t>(position) * kv_dim, k_, kv_dim * sizeof(half),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(cache.v + static_cast<std::size_t>(position) * kv_dim, v_, kv_dim * sizeof(half),
                          cudaMemcpyDeviceToDevice));

    // 3) 短序列走 FlashAttention，长序列走 PagedAttention
    int seq_len = position + 1;
    if (seq_len <= page_size_ * 8) {
      launch_flash_attention(q_, cache.k, cache.v, cfg_.num_heads, cfg_.num_kv_heads, head_dim, seq_len,
                             context_);
    } else {
      launch_paged_attention(q_, cache.k, cache.v, page_table_dev_, page_size_, cfg_.num_heads,
                             cfg_.num_kv_heads, head_dim, seq_len, context_);
    }

    gemv(context_, w.wo, q_dim, hidden, x_);
    launch_add_inplace(x_, x_resid_, hidden);

    CUDA_CHECK(cudaMemcpy(x_resid_, x_, hidden * sizeof(half), cudaMemcpyDeviceToDevice));
    launch_rmsnorm(x_, w.ffn_norm, hidden, cfg_.rms_norm_eps, x_norm_);

    gemv(x_norm_, w.w_gate, hidden, inter, ffn_gate_);
    gemv(x_norm_, w.w_up, hidden, inter, ffn_up_);
    launch_swiglu(ffn_gate_, ffn_up_, inter, ffn_hidden_);
    gemv(ffn_hidden_, w.w_down, inter, hidden, x_);
    launch_add_inplace(x_, x_resid_, hidden);
  }

  // 4) 最终层归一化 + lm_head 输出 logits
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

  // Prefill
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
