#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-Omni-3B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

vllm serve "$MODEL" \
	--host "$HOST" \
	--port "$PORT" \
	--gpu-memory-utilization 0.55 \
	--max-model-len 32768 \
	--max-num-seqs 1 \
	--trust-remote-code
