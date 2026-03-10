#!/usr/bin/env python3
import argparse
import struct
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

MAGIC = b"QWENMINI"
VERSION = 3


def write_tensor(f, t: torch.Tensor):
    t = t.detach().to(torch.float16).contiguous().cpu()
    numel = t.numel()
    f.write(struct.pack("<Q", numel))
    f.write(t.numpy().tobytes())


def main():
    parser = argparse.ArgumentParser(description="Convert local Qwen3 weights to qmini binary")
    parser.add_argument("--model-dir", required=True, help="Local model dir from ModelScope")
    parser.add_argument("--output", required=True, help="Output .qmini binary file")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--num-layers", type=int, default=0, help="0 means all layers")
    args = parser.parse_args()

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
        torch_dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=True,
        device_map="cpu",
    )
    model.eval()

    hidden = int(cfg.hidden_size)
    inter = int(cfg.intermediate_size)
    num_heads = int(cfg.num_attention_heads)
    num_kv_heads = int(getattr(cfg, "num_key_value_heads", num_heads))
    head_dim = int(getattr(cfg, "head_dim", 0))
    if head_dim <= 0:
        # Fallback to real tensor shape to avoid config mismatch.
        sample_layer = model.model.layers[0]
        head_dim = int(sample_layer.self_attn.q_proj.weight.shape[0] // num_heads)
    vocab = int(cfg.vocab_size)
    n_layers = int(cfg.num_hidden_layers)
    if args.num_layers > 0:
        n_layers = min(n_layers, args.num_layers)

    rms_eps = float(getattr(cfg, "rms_norm_eps", 1e-6))
    rope_theta = float(getattr(cfg, "rope_theta", 1_000_000.0))

    print("[3/3] Writing qmini file")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<i", VERSION))
        f.write(struct.pack("<i", vocab))
        f.write(struct.pack("<i", hidden))
        f.write(struct.pack("<i", inter))
        f.write(struct.pack("<i", n_layers))
        f.write(struct.pack("<i", num_heads))
        f.write(struct.pack("<i", num_kv_heads))
        f.write(struct.pack("<i", head_dim))
        f.write(struct.pack("<i", int(args.max_seq_len)))
        f.write(struct.pack("<f", rms_eps))
        f.write(struct.pack("<f", rope_theta))

        # embed tokens
        write_tensor(f, model.model.embed_tokens.weight)

        for i in range(n_layers):
            layer = model.model.layers[i]
            write_tensor(f, layer.input_layernorm.weight)
            write_tensor(f, layer.self_attn.q_proj.weight.T)
            write_tensor(f, layer.self_attn.k_proj.weight.T)
            write_tensor(f, layer.self_attn.v_proj.weight.T)
            write_tensor(f, layer.self_attn.q_norm.weight)
            write_tensor(f, layer.self_attn.k_norm.weight)
            write_tensor(f, layer.self_attn.o_proj.weight.T)
            write_tensor(f, layer.post_attention_layernorm.weight)
            write_tensor(f, layer.mlp.gate_proj.weight.T)
            write_tensor(f, layer.mlp.up_proj.weight.T)
            write_tensor(f, layer.mlp.down_proj.weight.T)

        write_tensor(f, model.model.norm.weight)
        write_tensor(f, model.lm_head.weight.T)

    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
