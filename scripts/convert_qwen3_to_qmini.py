#!/usr/bin/env python3
import argparse
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM

MAGIC = b"QWENMINI"
VERSION = 4
LAYER_TYPE_FULL_ATTENTION = 0
LAYER_TYPE_LINEAR_ATTENTION = 1


@dataclass
class ModelDims:
    vocab: int
    hidden: int
    inter: int
    n_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    max_seq_len: int
    rms_eps: float
    rope_theta: float
    layer_types: list[int]

    @property
    def q_dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def linear_key_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def linear_value_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def linear_conv_dim(self) -> int:
        return self.linear_key_dim * 2 + self.linear_value_dim


def ensure_shape(name: str, t: torch.Tensor, expected: tuple[int, ...]) -> None:
    actual = tuple(t.shape)
    if actual != expected:
        raise ValueError(f"Shape mismatch for {name}: expected={expected}, actual={actual}")


def write_raw_tensor(f, t: torch.Tensor) -> None:
    t = t.detach().to(torch.float16).contiguous().cpu()
    f.write(struct.pack("<Q", t.numel()))
    f.write(t.numpy().tobytes())


def write_vector(f, name: str, t: torch.Tensor, size: int) -> None:
    ensure_shape(name, t, (size,))
    write_raw_tensor(f, t)


def write_rmsnorm_vector(f, name: str, t: torch.Tensor, size: int, plus_one: bool) -> None:
    ensure_shape(name, t, (size,))
    tt = t + 1.0 if plus_one else t
    write_raw_tensor(f, tt)


def write_matrix_for_cublas(f, name: str, w: torch.Tensor, out_dim: int, in_dim: int) -> None:
    # C++ 侧 GEMV 采用列主序读取，因此这里把 PyTorch 的 [out, in] 做转置后写盘。
    # 写盘后 C++ 直接按 out_dim * in_dim 扁平读取即可与数学语义对齐。
    ensure_shape(name, w, (out_dim, in_dim))
    write_raw_tensor(f, w.T)


def write_embedding_table(f, name: str, w: torch.Tensor, vocab: int, hidden: int) -> None:
    ensure_shape(name, w, (vocab, hidden))
    write_raw_tensor(f, w)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert local Qwen3/Qwen3.5 weights to qmini v4")
    parser.add_argument("--model-dir", required=True, help="Local model dir from ModelScope")
    parser.add_argument("--output", required=True, help="Output .qmini binary file")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--num-layers", type=int, default=0, help="0 means all layers")
    return parser.parse_args()


def get_text_cfg(cfg: Any) -> Any:
    return getattr(cfg, "text_config", cfg)


def resolve_language_model(model: Any) -> Any:
    if hasattr(model, "model"):
        core = model.model
        if hasattr(core, "layers") and hasattr(core, "embed_tokens"):
            return core
        if hasattr(core, "language_model"):
            return core.language_model
    if hasattr(model, "language_model"):
        return model.language_model
    raise ValueError("Cannot locate language model module")


def build_dims(cfg: Any, req_layers: int, max_seq_len: int) -> ModelDims:
    text_cfg = get_text_cfg(cfg)

    n_layers = int(text_cfg.num_hidden_layers)
    if req_layers > 0:
        n_layers = min(n_layers, req_layers)

    layer_types_cfg = list(getattr(text_cfg, "layer_types", ["full_attention"] * int(text_cfg.num_hidden_layers)))
    layer_types = layer_types_cfg[:n_layers]
    layer_types_encoded = [
        LAYER_TYPE_LINEAR_ATTENTION if t == "linear_attention" else LAYER_TYPE_FULL_ATTENTION
        for t in layer_types
    ]

    rope_theta = getattr(text_cfg, "rope_theta", None)
    if rope_theta is None:
        rope_params = getattr(text_cfg, "rope_parameters", None)
        if rope_params is not None:
            rope_theta = getattr(rope_params, "rope_theta", None)
    if rope_theta is None:
        rope_theta = 1_000_000.0

    return ModelDims(
        vocab=int(text_cfg.vocab_size),
        hidden=int(text_cfg.hidden_size),
        inter=int(text_cfg.intermediate_size),
        n_layers=n_layers,
        num_heads=int(text_cfg.num_attention_heads),
        num_kv_heads=int(getattr(text_cfg, "num_key_value_heads", text_cfg.num_attention_heads)),
        head_dim=int(getattr(text_cfg, "head_dim", text_cfg.hidden_size // text_cfg.num_attention_heads)),
        linear_num_key_heads=int(getattr(text_cfg, "linear_num_key_heads", 0)),
        linear_num_value_heads=int(getattr(text_cfg, "linear_num_value_heads", 0)),
        linear_key_head_dim=int(getattr(text_cfg, "linear_key_head_dim", 0)),
        linear_value_head_dim=int(getattr(text_cfg, "linear_value_head_dim", 0)),
        linear_conv_kernel_dim=int(getattr(text_cfg, "linear_conv_kernel_dim", 0)),
        max_seq_len=int(max_seq_len),
        rms_eps=float(getattr(text_cfg, "rms_norm_eps", 1e-6)),
        rope_theta=float(rope_theta),
        layer_types=layer_types_encoded,
    )


def write_header(f, dims: ModelDims) -> None:
    # qmini v4 头部布局（严格顺序）：
    # magic/version -> 基础维度 -> linear-attention 维度 -> max_seq/rms/rope -> layer_types
    f.write(MAGIC)
    f.write(struct.pack("<i", VERSION))
    f.write(struct.pack("<i", dims.vocab))
    f.write(struct.pack("<i", dims.hidden))
    f.write(struct.pack("<i", dims.inter))
    f.write(struct.pack("<i", dims.n_layers))
    f.write(struct.pack("<i", dims.num_heads))
    f.write(struct.pack("<i", dims.num_kv_heads))
    f.write(struct.pack("<i", dims.head_dim))
    f.write(struct.pack("<i", dims.linear_num_key_heads))
    f.write(struct.pack("<i", dims.linear_num_value_heads))
    f.write(struct.pack("<i", dims.linear_key_head_dim))
    f.write(struct.pack("<i", dims.linear_value_head_dim))
    f.write(struct.pack("<i", dims.linear_conv_kernel_dim))
    f.write(struct.pack("<i", dims.max_seq_len))
    f.write(struct.pack("<f", dims.rms_eps))
    f.write(struct.pack("<f", dims.rope_theta))
    for t in dims.layer_types:
        f.write(struct.pack("<i", int(t)))


def write_layer_weights(f, layer: Any, dims: ModelDims, layer_idx: int, layer_type: int) -> None:
    is_qwen35 = dims.linear_num_value_heads > 0
    write_rmsnorm_vector(
        f,
        f"layers[{layer_idx}].input_layernorm",
        layer.input_layernorm.weight,
        dims.hidden,
        plus_one=is_qwen35,
    )

    if layer_type == LAYER_TYPE_LINEAR_ATTENTION:
        # linear-attention: 导出 Qwen3.5 GatedDeltaNet 全量参数。
        lat = layer.linear_attn
        write_matrix_for_cublas(
            f, f"layers[{layer_idx}].linear.in_proj_qkv", lat.in_proj_qkv.weight, dims.linear_conv_dim, dims.hidden
        )
        write_matrix_for_cublas(
            f, f"layers[{layer_idx}].linear.in_proj_z", lat.in_proj_z.weight, dims.linear_value_dim, dims.hidden
        )
        write_matrix_for_cublas(
            f, f"layers[{layer_idx}].linear.in_proj_b", lat.in_proj_b.weight, dims.linear_num_value_heads, dims.hidden
        )
        write_matrix_for_cublas(
            f, f"layers[{layer_idx}].linear.in_proj_a", lat.in_proj_a.weight, dims.linear_num_value_heads, dims.hidden
        )

        conv_w = lat.conv1d.weight.squeeze(1)
        ensure_shape(
            f"layers[{layer_idx}].linear.conv1d.weight", conv_w, (dims.linear_conv_dim, dims.linear_conv_kernel_dim)
        )
        write_raw_tensor(f, conv_w)

        write_vector(
            f,
            f"layers[{layer_idx}].linear.dt_bias",
            lat.dt_bias,
            dims.linear_num_value_heads,
        )
        write_vector(
            f,
            f"layers[{layer_idx}].linear.A_log",
            lat.A_log,
            dims.linear_num_value_heads,
        )
        write_vector(
            f,
            f"layers[{layer_idx}].linear.norm.weight",
            lat.norm.weight,
            dims.linear_value_head_dim,
        )
        write_matrix_for_cublas(
            f,
            f"layers[{layer_idx}].linear.out_proj",
            lat.out_proj.weight,
            dims.hidden,
            dims.linear_value_dim,
        )
    else:
        # full-attention: q_proj 必须保留 2*q_dim（query + gate），严禁截断。
        attn = layer.self_attn
        write_matrix_for_cublas(
            f,
            f"layers[{layer_idx}].self_attn.q_proj",
            attn.q_proj.weight,
            dims.q_dim * 2,
            dims.hidden,
        )
        write_matrix_for_cublas(
            f,
            f"layers[{layer_idx}].self_attn.k_proj",
            attn.k_proj.weight,
            dims.kv_dim,
            dims.hidden,
        )
        write_matrix_for_cublas(
            f,
            f"layers[{layer_idx}].self_attn.v_proj",
            attn.v_proj.weight,
            dims.kv_dim,
            dims.hidden,
        )
        write_rmsnorm_vector(
            f,
            f"layers[{layer_idx}].self_attn.q_norm",
            attn.q_norm.weight,
            dims.head_dim,
            plus_one=is_qwen35,
        )
        write_rmsnorm_vector(
            f,
            f"layers[{layer_idx}].self_attn.k_norm",
            attn.k_norm.weight,
            dims.head_dim,
            plus_one=is_qwen35,
        )
        write_matrix_for_cublas(
            f,
            f"layers[{layer_idx}].self_attn.o_proj",
            attn.o_proj.weight,
            dims.hidden,
            dims.q_dim,
        )

    write_rmsnorm_vector(
        f,
        f"layers[{layer_idx}].post_attention_layernorm",
        layer.post_attention_layernorm.weight,
        dims.hidden,
        plus_one=is_qwen35,
    )
    write_matrix_for_cublas(
        f,
        f"layers[{layer_idx}].mlp.gate_proj",
        layer.mlp.gate_proj.weight,
        dims.inter,
        dims.hidden,
    )
    write_matrix_for_cublas(
        f,
        f"layers[{layer_idx}].mlp.up_proj",
        layer.mlp.up_proj.weight,
        dims.inter,
        dims.hidden,
    )
    write_matrix_for_cublas(
        f,
        f"layers[{layer_idx}].mlp.down_proj",
        layer.mlp.down_proj.weight,
        dims.hidden,
        dims.inter,
    )


def write_weights(f, model: Any, dims: ModelDims) -> None:
    lm = resolve_language_model(model)
    # embedding 不转置，按 [vocab, hidden] 原布局写入。
    write_embedding_table(f, "embed_tokens", lm.embed_tokens.weight, dims.vocab, dims.hidden)

    for i in range(dims.n_layers):
        layer = lm.layers[i]
        write_layer_weights(f, layer, dims, i, dims.layer_types[i])

    is_qwen35 = dims.linear_num_value_heads > 0
    write_rmsnorm_vector(f, "model.norm", lm.norm.weight, dims.hidden, plus_one=is_qwen35)
    write_matrix_for_cublas(f, "lm_head", model.lm_head.weight, dims.vocab, dims.hidden)


def main() -> None:
    args = parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Local model directory not found: {model_dir}. "
            "Please pass an existing local path (absolute path recommended)."
        )

    out_path = Path(args.output)

    print(f"[1/3] Loading config from {model_dir}")
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)

    print("[2/3] Loading model weights (this may take a while)")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=True,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()

    dims = build_dims(cfg, args.num_layers, args.max_seq_len)
    print(
        "[info] "
        f"vocab={dims.vocab} hidden={dims.hidden} inter={dims.inter} layers={dims.n_layers} "
        f"heads={dims.num_heads} kv_heads={dims.num_kv_heads} head_dim={dims.head_dim} "
        f"linear(k_heads={dims.linear_num_key_heads},v_heads={dims.linear_num_value_heads},"
        f"k_dim={dims.linear_key_head_dim},v_dim={dims.linear_value_head_dim},k={dims.linear_conv_kernel_dim})"
    )

    n_linear = sum(1 for t in dims.layer_types if t == LAYER_TYPE_LINEAR_ATTENTION)
    n_full = dims.n_layers - n_linear
    print(f"[info] layer_types: full_attention={n_full}, linear_attention={n_linear}")

    print("[3/3] Writing qmini file")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        write_header(f, dims)
        write_weights(f, model, dims)

    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
