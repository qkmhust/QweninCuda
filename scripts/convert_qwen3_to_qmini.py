#!/usr/bin/env python3
import argparse
import struct
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

MAGIC = b"QWENMINI"
VERSION = 3


@dataclass
class ModelDims:
    vocab: int
    hidden: int
    inter: int
    n_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    rms_eps: float
    rope_theta: float

    @property
    def q_dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim


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


def write_matrix_for_cublas(f, name: str, w: torch.Tensor, out_dim: int, in_dim: int) -> None:
    # 运行时 GEMV 约定是列主序读取，因此这里把 W 做转置后写盘。
    # 这样 C++ 侧按当前 cuBLAS 调用方式读取时，逻辑上仍等价于 W(out_dim, in_dim)。
    ensure_shape(name, w, (out_dim, in_dim))
    write_raw_tensor(f, w.T)


def write_embedding_table(f, name: str, w: torch.Tensor, vocab: int, hidden: int) -> None:
    # embedding kernel 按 table[token, dim] 行主序索引，不能转置。
    ensure_shape(name, w, (vocab, hidden))
    write_raw_tensor(f, w)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert local Qwen3 weights to qmini binary")
    parser.add_argument("--model-dir", required=True, help="Local model dir from ModelScope")
    parser.add_argument("--output", required=True, help="Output .qmini binary file")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--num-layers", type=int, default=0, help="0 means all layers")
    return parser.parse_args()


def build_dims(cfg, model, max_seq_len: int, req_layers: int) -> ModelDims:
    hidden = int(cfg.hidden_size)
    inter = int(cfg.intermediate_size)
    num_heads = int(cfg.num_attention_heads)
    num_kv_heads = int(getattr(cfg, "num_key_value_heads", num_heads))
    head_dim = int(getattr(cfg, "head_dim", 0))
    if head_dim <= 0:
      sample_layer = model.model.layers[0]
      head_dim = int(sample_layer.self_attn.q_proj.weight.shape[0] // num_heads)

    n_layers = int(cfg.num_hidden_layers)
    if req_layers > 0:
        n_layers = min(n_layers, req_layers)

    return ModelDims(
        vocab=int(cfg.vocab_size),
        hidden=hidden,
        inter=inter,
        n_layers=n_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_seq_len=int(max_seq_len),
        rms_eps=float(getattr(cfg, "rms_norm_eps", 1e-6)),
        rope_theta=float(getattr(cfg, "rope_theta", 1_000_000.0)),
    )


def write_header(f, dims: ModelDims) -> None:
    f.write(MAGIC)
    f.write(struct.pack("<i", VERSION))
    f.write(struct.pack("<i", dims.vocab))
    f.write(struct.pack("<i", dims.hidden))
    f.write(struct.pack("<i", dims.inter))
    f.write(struct.pack("<i", dims.n_layers))
    f.write(struct.pack("<i", dims.num_heads))
    f.write(struct.pack("<i", dims.num_kv_heads))
    f.write(struct.pack("<i", dims.head_dim))
    f.write(struct.pack("<i", dims.max_seq_len))
    f.write(struct.pack("<f", dims.rms_eps))
    f.write(struct.pack("<f", dims.rope_theta))


def write_weights(f, model, dims: ModelDims) -> None:
    write_embedding_table(f, "embed_tokens", model.model.embed_tokens.weight, dims.vocab, dims.hidden)

    for i in range(dims.n_layers):
        layer = model.model.layers[i]
        write_vector(f, f"layers[{i}].input_layernorm", layer.input_layernorm.weight, dims.hidden)
        write_matrix_for_cublas(f, f"layers[{i}].q_proj", layer.self_attn.q_proj.weight, dims.q_dim, dims.hidden)
        write_matrix_for_cublas(f, f"layers[{i}].k_proj", layer.self_attn.k_proj.weight, dims.kv_dim, dims.hidden)
        write_matrix_for_cublas(f, f"layers[{i}].v_proj", layer.self_attn.v_proj.weight, dims.kv_dim, dims.hidden)
        write_vector(f, f"layers[{i}].q_norm", layer.self_attn.q_norm.weight, dims.head_dim)
        write_vector(f, f"layers[{i}].k_norm", layer.self_attn.k_norm.weight, dims.head_dim)
        write_matrix_for_cublas(f, f"layers[{i}].o_proj", layer.self_attn.o_proj.weight, dims.hidden, dims.q_dim)
        write_vector(
            f,
            f"layers[{i}].post_attention_layernorm",
            layer.post_attention_layernorm.weight,
            dims.hidden,
        )
        write_matrix_for_cublas(f, f"layers[{i}].gate_proj", layer.mlp.gate_proj.weight, dims.inter, dims.hidden)
        write_matrix_for_cublas(f, f"layers[{i}].up_proj", layer.mlp.up_proj.weight, dims.inter, dims.hidden)
        write_matrix_for_cublas(f, f"layers[{i}].down_proj", layer.mlp.down_proj.weight, dims.hidden, dims.inter)

    write_vector(f, "model.norm", model.model.norm.weight, dims.hidden)
    write_matrix_for_cublas(f, "lm_head", model.lm_head.weight, dims.vocab, dims.hidden)


def main():
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
    )
    model.eval()

    dims = build_dims(cfg, model, args.max_seq_len, args.num_layers)
    print(
        "[info] "
        f"vocab={dims.vocab} hidden={dims.hidden} inter={dims.inter} "
        f"layers={dims.n_layers} heads={dims.num_heads} kv_heads={dims.num_kv_heads} "
        f"head_dim={dims.head_dim} max_seq_len={dims.max_seq_len}"
    )

    print("[3/3] Writing qmini file")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        write_header(f, dims)
        write_weights(f, model, dims)

    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
