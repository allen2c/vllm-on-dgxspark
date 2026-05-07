# Development
format:
	@isort vllm_on_dgxspark tests
	@black vllm_on_dgxspark tests

install:
	pip install -e ".[dev]"

# Docs
mkdocs:
	mkdocs serve -a 0.0.0.0:8000

# Tests
pytest:
	python -m pytest
