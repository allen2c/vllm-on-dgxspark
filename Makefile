.PHONY: format install export-deps mkdocs pytest

# Development
format:
	@isort vllm_on_dgxspark tests
	@black vllm_on_dgxspark tests

install:
	pip install -e ".[dev]"

export-deps:
	python -c 'import tomllib; data = tomllib.load(open("pyproject.toml", "rb")); print("\n".join(data["project"]["dependencies"]))' > requirements.txt

# Docs
mkdocs:
	mkdocs serve -a 0.0.0.0:8000

# Tests
pytest:
	python -m pytest
