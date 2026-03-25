# Performance Optimization Log

Prompt used:
- 请用一句话解释什么是CUDA。

Decode params:
- max-new-tokens=48
- temperature=0.35
- top-k=24
- top-p=0.72
- min-p=0.08
- temp-decay=0.95
- greedy-after=0

## Baseline (before step1)
Command:
- python scripts/run_chat.py ...

Result:
- elapsed_ms=5929.05
- new_tokens=48
- tok_per_s=8.10

## Step1: device-side q/gate split (remove D2H/H2D split in full-attn)
Code changes:
- add CUDA kernel split_q_gate_interleaved
- replace host split path in run_full_attention_block

Result:
- elapsed_ms=5849.38
- new_tokens=48
- tok_per_s=8.21

Improvement vs baseline:
- +1.36% tok/s

## Step2: streaming display path
Code changes:
- engine adds --stream-ids and prints stream_token_id per generated token
- run_chat.py adds --stream mode and incremental decode display

Non-stream result (for throughput reference):
- elapsed_ms=5816.35
- new_tokens=48
- tok_per_s=8.25

Improvement vs step1:
- +0.49% tok/s (within run-to-run noise range)

Streaming UX metrics:
- non-stream wall_ms=28533.88 (first visible content only at end)
- stream wall_ms=28590.53
- stream ttft_ms=13284.36

Visible-latency improvement:
- first visible output time reduced by 53.45%
- note: total wall time is almost unchanged because model loading + full decode cost is unchanged
