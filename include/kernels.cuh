#pragma once

#include <cuda_fp16.h>

void launch_embedding_lookup(const half* table, int vocab_size, int hidden_size, int token_id, half* out);
void launch_rmsnorm(const half* x, const half* weight, int n, float eps, half* out);
void launch_head_rmsnorm(const half* x, const half* weight, int num_heads, int head_dim, float eps,
                         half* out);
void launch_add_inplace(half* x, const half* y, int n);
void launch_rope_inplace(half* q_or_k, int num_heads, int head_dim, int position, float rope_theta);
void launch_attention_scores(const half* q, const half* k_cache, int num_heads, int num_kv_heads,
                             int head_dim, int seq_len, float* scores);
void launch_softmax_rows(float* scores, int rows, int cols);
void launch_attention_context(const float* probs, const half* v_cache, int num_heads, int num_kv_heads,
                              int head_dim, int seq_len, half* context);
void launch_swiglu(const half* gate, const half* up, int n, half* out);
