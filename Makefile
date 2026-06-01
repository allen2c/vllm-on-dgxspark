SHELL := /bin/bash
.ONESHELL:

PYTHON ?= python

SYS_TOTAL_GIB := $(shell python -c "import os; mem=os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES'); print(f'{mem/1024**3:.1f}')")
GPU_MEM_UTIL  := $(shell python -c "total=$(SYS_TOTAL_GIB); target=20.0; ratio=target/total; print(f'{min(ratio, 0.90):.2f}')")

define INSPECT_PY
from importlib.metadata import version, PackageNotFoundError, distributions
core = ['vllm', 'torch', 'torchvision', 'torchaudio', 'triton', 'flash-attn',
        'xformers', 'transformers', 'tokenizers', 'numpy', 'bitsandbytes']
for p in core:
    try:
        print(f'{p:32} {version(p)}')
    except PackageNotFoundError:
        print(f'{p:32} -')
print()
# Dynamically list every installed NVIDIA package (CUDA 12/13, x86/arm agnostic)
nv = sorted({(d.metadata['Name'], d.version) for d in distributions()
             if (d.metadata['Name'] or '').lower().startswith('nvidia')})
if nv:
    for name, ver in nv:
        print(f'{name:32} {ver}')
else:
    print('(no nvidia-* packages installed)')
print()
try:
    import torch
    print(f'torch.version.cuda               {torch.version.cuda}')
    print(f'torch.cuda.is_available          {torch.cuda.is_available()}')
    print(f'torch.cuda.device_count          {torch.cuda.device_count()}')
except Exception as e:
    print(f'torch CUDA info unavailable: {e}')
endef
export INSPECT_PY

.PHONY: fmt install inspect

# Development
fmt:
	@isort vllm_on_dgxspark tests
	@black vllm_on_dgxspark tests

install:
	pip install -e ".[dev]"

# Inspect environment (read-only, no side effects)
inspect:
	@echo "=== System ==="
	@echo "Host : $$(hostname)"
	@echo "OS   : $$(uname -srm)"
	@echo "RAM  : $(SYS_TOTAL_GIB) GiB"
	@echo "Computed gpu-memory-utilization (target 20 GiB): $(GPU_MEM_UTIL)"
	@echo
	@echo "=== NVIDIA Driver / GPU ==="
	@nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv 2>/dev/null || echo "nvidia-smi not available"
	@echo
	@echo "=== CUDA Toolkit ==="
	@nvcc --version 2>/dev/null | grep -i release || echo "nvcc not available"
	@echo
	@echo "=== Python ==="
	@$(PYTHON) --version
	@echo
	@echo "=== Key packages (NVIDIA / vLLM stack) ==="
	@$(PYTHON) -c "$$INSPECT_PY"
