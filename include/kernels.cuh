#pragma once

#include <cuda_fp16.h>

void launch_embedding_lookup(const half* table, int vocab_size, int hidden_size, int token_id, half* out);
void launch_rmsnorm(const half* x, const half* weight, int n, float eps, half* out);
void launch_head_rmsnorm(const half* x, const half* weight, int num_heads, int head_dim, float eps,
                         half* out);
void launch_split_q_gate_interleaved(const half* packed_q_gate, int num_heads, int head_dim,
                                     half* q, half* gate);
void launch_add_inplace(half* x, const half* y, int n);
void launch_sigmoid_mul_inplace(half* x, const half* gate, int n);
void launch_rope_inplace(half* q_or_k, int num_heads, int head_dim, int position, float rope_theta);
void launch_swiglu(const half* gate, const half* up, int n, half* out);
void launch_flash_attention(const half* q, const half* k_cache, const half* v_cache, int num_heads,
                            int num_kv_heads, int head_dim, int seq_len, half* context);
void launch_paged_attention(const half* q, const half* k_cache, const half* v_cache,
                            const int* page_table, int page_size, int num_heads, int num_kv_heads,
                            int head_dim, int seq_len, half* context);
