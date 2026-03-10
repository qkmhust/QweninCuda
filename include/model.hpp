#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <string>
#include <vector>

struct QwenMiniConfig {
  int32_t vocab_size = 0;
  int32_t hidden_size = 0;
  int32_t intermediate_size = 0;
  int32_t num_layers = 0;
  int32_t num_heads = 0;
  int32_t num_kv_heads = 0;
  int32_t head_dim = 0;
  int32_t max_seq_len = 0;
  float rms_norm_eps = 1e-6f;
  float rope_theta = 1000000.0f;
};

class QwenMiniModel {
 public:
  QwenMiniModel();
  ~QwenMiniModel();

  bool load(const std::string& path);

  std::vector<int> generate(const std::vector<int>& input_ids, int max_new_tokens, int eos_id,
                            float temperature = 0.8f, int top_k = 40,
                            float top_p = 0.95f,
                            float min_p = 0.0f,
                            float temp_decay = 1.0f,
                            int greedy_after = -1,
                            float repetition_penalty = 1.1f,
                            float* elapsed_ms = nullptr);

  const QwenMiniConfig& config() const { return cfg_; }

 private:
  struct LayerWeights {
    half* attn_norm = nullptr;
    half* wq = nullptr;
    half* wk = nullptr;
    half* wv = nullptr;
    half* q_norm = nullptr;
    half* k_norm = nullptr;
    half* wo = nullptr;
    half* ffn_norm = nullptr;
    half* w_gate = nullptr;
    half* w_up = nullptr;
    half* w_down = nullptr;
  };

  struct LayerCache {
    half* k = nullptr;
    half* v = nullptr;
  };

  bool alloc_runtime();
  void free_all();
  int decode_next(int token_id, int position, std::vector<half>* host_logits = nullptr);
  void gemv(const half* x, const half* w, int in_dim, int out_dim, half* out);

  QwenMiniConfig cfg_;
  bool loaded_ = false;
  cublasHandle_t cublas_ = nullptr;

  half* tok_embeddings_ = nullptr;
  std::vector<LayerWeights> layers_;
  half* final_norm_ = nullptr;
  half* lm_head_ = nullptr;

  std::vector<LayerCache> caches_;

  half* x_ = nullptr;
  half* x_norm_ = nullptr;
  half* x_resid_ = nullptr;
  half* q_ = nullptr;
  half* k_ = nullptr;
  half* v_ = nullptr;
  float* attn_scores_ = nullptr;
  half* context_ = nullptr;
  half* ffn_gate_ = nullptr;
  half* ffn_up_ = nullptr;
  half* ffn_hidden_ = nullptr;
  half* logits_ = nullptr;
};
