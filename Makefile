SHELL := /bin/bash
.ONESHELL:

MODEL := Qwen/Qwen2.5-Omni-3B
VOYAGE_MODEL := voyageai/voyage-4-nano
VOYAGE_SERVED_MODEL_NAME := voyage-4-nano
VOYAGE_PORT := 8944
HOST := 0.0.0.0
PORT := 8000
URL := http://localhost:$(PORT)/v1/chat/completions
PYTHON ?= python
AUDIO_FILE := /tmp/vllm-media/cough-16k-mono.wav
VIDEO_URL := https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/draw.mp4

SYS_TOTAL_GIB := $(shell python -c "import os; mem=os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES'); print(f'{mem/1024**3:.1f}')")
VOYAGE_GPU_MEM_UTIL  := $(shell python -c "total=$(SYS_TOTAL_GIB); target=20.0; ratio=target/total; print(f'{min(ratio, 0.90):.2f}')")

.PHONY: format install export-deps mkdocs pytest vllm-serve vllm-serve-voyage-4-nano query-vllm-image query-vllm-audio query-vllm-video download-audio

# Development
format:
	@isort vllm_on_dgxspark tests
	@black vllm_on_dgxspark tests

install:
	pip install -e ".[dev]"

export-deps:
	$(PYTHON) -c 'import tomllib; data = tomllib.load(open("pyproject.toml", "rb")); print("\n".join(data["project"]["dependencies"]))' > requirements.txt

# VLLM Serve
vllm-serve:
	vllm serve $(MODEL) \
		--host $(HOST) \
		--port $(PORT) \
		--gpu-memory-utilization 0.55 \
		--max-model-len 32768 \
		--max-num-seqs 1 \
		--trust-remote-code

vllm-serve-voyage-4-nano:
	@echo "System RAM: $(SYS_TOTAL_GIB) GiB, targeting 20 GiB → utilization: $(VOYAGE_GPU_MEM_UTIL)"
	vllm serve $(VOYAGE_MODEL) \
		--served-model-name $(VOYAGE_SERVED_MODEL_NAME) \
		--host $(HOST) \
		--port $(VOYAGE_PORT) \
		--runner pooling \
		--convert embed \
		--trust-remote-code \
		--dtype bfloat16 \
		--max-model-len 32768 \
		--max-num-seqs 512 \
		--enforce-eager \
		--gpu-memory-utilization $(VOYAGE_GPU_MEM_UTIL) \
		--hf-overrides '{"architectures": ["VoyageQwen3BidirectionalEmbedModel"]}'

# Query VLLM
## Test image input
query-vllm-image:
	curl $(URL) \
		-H "Content-Type: application/json" \
		-d '{
			"model": "$(MODEL)",
			"messages": [{
				"role": "user",
				"content": [
					{"type": "text", "text": "Describe this image briefly."},
					{"type": "image_url", "image_url": {"url": "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/duck.jpg"}}
				]
			}],
			"max_tokens": 128
		}'

## Test audio input
query-vllm-audio:
	AUDIO_B64=$$(base64 -w 0 $(AUDIO_FILE)); \
	curl --fail-with-body $(URL) \
		-H "Content-Type: application/json" \
		-d "$$(AUDIO_B64=$$AUDIO_B64 $(PYTHON) -c 'import json, os; print(json.dumps({"model": "$(MODEL)", "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": [{"type": "text", "text": "What can you hear in this audio? Answer briefly."}, {"type": "input_audio", "input_audio": {"data": os.environ["AUDIO_B64"], "format": "wav"}}]}], "max_completion_tokens": 128}))')"

## Test video input
query-vllm-video:
	curl --fail-with-body $(URL) \
		-H "Content-Type: application/json" \
		-d "$$($(PYTHON) -c 'import json; print(json.dumps({"model": "$(MODEL)", "messages": [{"role": "user", "content": [{"type": "text", "text": "Summarize this video in one sentence."}, {"type": "video_url", "video_url": {"url": "$(VIDEO_URL)"}}]}], "max_completion_tokens": 128}))')"

# Utilities
download-audio:
	mkdir -p /tmp/vllm-media
	curl -L \
		"https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav" \
		-o /tmp/vllm-media/cough-original.wav
	ffmpeg -y \
		-i /tmp/vllm-media/cough-original.wav \
		-ac 1 \
		-ar 16000 \
		-c:a pcm_s16le \
		$(AUDIO_FILE)
	file $(AUDIO_FILE)

# Docs
mkdocs:
	mkdocs serve -a 0.0.0.0:8000

# Tests
pytest:
	$(PYTHON) -m pytest
