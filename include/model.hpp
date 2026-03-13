#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <string>
#include <vector>

struct QwenMiniConfig {
  int32_t version = 0;
  int32_t vocab_size = 0;
  int32_t hidden_size = 0;
  int32_t intermediate_size = 0;
  int32_t num_layers = 0;
  int32_t num_heads = 0;
  int32_t num_kv_heads = 0;
  int32_t head_dim = 0;
  int32_t linear_num_key_heads = 0;
  int32_t linear_num_value_heads = 0;
  int32_t linear_key_head_dim = 0;
  int32_t linear_value_head_dim = 0;
  int32_t linear_conv_kernel_dim = 0;
  int32_t max_seq_len = 0;
  float rms_norm_eps = 1e-6f;
  float rope_theta = 1000000.0f;
  std::vector<int32_t> layer_types;
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
                            int no_repeat_ngram_size = 0,
                            float presence_penalty = 0.0f,
                            float frequency_penalty = 0.0f,
                            float repetition_penalty = 1.1f,
                            int dump_topk = 0,
                            int dump_steps = 0,
                            float* elapsed_ms = nullptr);

  const QwenMiniConfig& config() const { return cfg_; }

 private:
  struct LayerWeights {
    bool is_linear = false;

    half* attn_norm = nullptr;
    half* wq = nullptr;
    half* wk = nullptr;
    half* wv = nullptr;
    half* q_norm = nullptr;
    half* k_norm = nullptr;
    half* wo = nullptr;

    half* linear_in_qkv = nullptr;
    half* linear_in_z = nullptr;
    half* linear_in_b = nullptr;
    half* linear_in_a = nullptr;
    half* linear_conv1d_w = nullptr;
    half* linear_dt_bias = nullptr;
    half* linear_a_log = nullptr;
    half* linear_norm_w = nullptr;
    half* linear_out_proj = nullptr;

    half* ffn_norm = nullptr;
    half* w_gate = nullptr;
    half* w_up = nullptr;
    half* w_down = nullptr;
  };

  struct LayerCache {
    half* k = nullptr;
    half* v = nullptr;

    std::vector<float> linear_conv_state;
    std::vector<float> linear_recurrent_state;
  };

  bool alloc_runtime();
  void free_all();
  int decode_next(int token_id, int position, std::vector<half>* host_logits = nullptr);
  void gemv(const half* x, const half* w, int in_dim, int out_dim, half* out);
  void run_full_attention_block(const LayerWeights& weights, LayerCache& cache, int position);
  void run_linear_attention_block(const LayerWeights& weights, LayerCache& cache);
  void run_ffn_block(const LayerWeights& weights);
  int compute_logits(std::vector<half>* host_logits);

  QwenMiniConfig cfg_;
  bool loaded_ = false;
  cublasHandle_t cublas_ = nullptr;

  half* tok_embeddings_ = nullptr;
  std::vector<LayerWeights> layers_;
  half* final_norm_ = nullptr;
  half* lm_head_ = nullptr;

  std::vector<LayerCache> caches_;

  int* page_table_dev_ = nullptr;
  std::vector<int> page_table_host_;
  int page_size_ = 16;

  half* x_ = nullptr;
  half* x_norm_ = nullptr;
  half* x_resid_ = nullptr;
  half* q_ = nullptr;
  half* k_ = nullptr;
  half* v_ = nullptr;
  half* context_ = nullptr;
  half* q_gate_ = nullptr;

  half* linear_mixed_qkv_ = nullptr;
  half* linear_z_ = nullptr;
  half* linear_b_ = nullptr;
  half* linear_a_ = nullptr;
  half* linear_out_ = nullptr;
  half* ffn_gate_ = nullptr;
  half* ffn_up_ = nullptr;
  half* ffn_hidden_ = nullptr;
  half* logits_ = nullptr;
};
