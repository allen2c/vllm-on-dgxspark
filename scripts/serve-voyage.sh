#!/usr/bin/env bash
set -euo pipefail

MODEL="${VOYAGE_MODEL:-voyageai/voyage-4-nano}"
SERVED_NAME="${VOYAGE_SERVED_MODEL_NAME:-voyage-4-nano}"
HOST="${HOST:-0.0.0.0}"
PORT="${VOYAGE_PORT:-8944}"

SYS_TOTAL_GIB=$(python -c "import os; mem=os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES'); print(f'{mem/1024**3:.1f}')")
GPU_MEM_UTIL=$(python -c "total=$SYS_TOTAL_GIB; target=20.0; print(f'{min(target/total, 0.90):.2f}')")
echo "System RAM: ${SYS_TOTAL_GIB} GiB, targeting 20 GiB → utilization: ${GPU_MEM_UTIL}"

vllm serve "$MODEL" \
	--served-model-name "$SERVED_NAME" \
	--host "$HOST" \
	--port "$PORT" \
	--runner pooling \
	--convert embed \
	--trust-remote-code \
	--dtype bfloat16 \
	--max-model-len 32768 \
	--max-num-seqs 256 \
	--enforce-eager \
	--gpu-memory-utilization "$GPU_MEM_UTIL" \
	--hf-overrides '{"architectures": ["VoyageQwen3BidirectionalEmbedModel"]}'
