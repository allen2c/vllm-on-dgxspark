#!/usr/bin/env bash
set -euo pipefail

# Qwen3.6-27B (FP8) — hybrid linear-attention (gated delta-net) + full-attention VLM.
# vLLM 0.22.0 supports arch `Qwen3_5ForConditionalGeneration` natively; FP8 is
# auto-detected from the model's quantization_config (no --quantization needed).
# Do NOT set --mamba-cache-mode: default "none" works; "all" is unsupported for
# this model and "align" is rejected by Model Runner V2.

MODEL="${MODEL:-Qwen/Qwen3.6-27B-FP8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"   # config allows up to 262144; cap to start
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"      # ~25 GiB weights on 119.6 GiB unified mem;
                                          # leaves headroom for co-resident services

vllm serve "$MODEL" \
	--host "$HOST" \
	--port "$PORT" \
	--gpu-memory-utilization "$GPU_MEM_UTIL" \
	--max-model-len "$MAX_MODEL_LEN" \
	--max-num-seqs 1 \
	--trust-remote-code
