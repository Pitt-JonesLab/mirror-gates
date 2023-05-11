PYTHON_VERSION = python3.9
PIP = .venv/bin/pip
PYTEST = .venv/bin/pytest
PRE_COMMIT = .venv/bin/pre-commit

init:
	$(PYTHON_VERSION) -m venv .venv
	@$(PIP) install --upgrade pip
	$(PIP) install -e .[dev]
	@$(PRE_COMMIT) install
	@$(PRE_COMMIT) autoupdate

clean:
	@find ./ -type f -name '*.pyc' -exec rm -f {} \; 2>/dev/null || true
	@find ./ -type d -name '__pycache__' -exec rm -rf {} \; 2>/dev/null || true
	@find ./ -type f -name 'Thumbs.db' -exec rm -f {} \; 2>/dev/null || true
	@find ./ -type f -name '*~' -exec rm -f {} \; 2>/dev/null || true
	@rm -rf .cache
	@rm -rf .pytest_cache
	@rm -rf .mypy_cache
	@rm -rf build
	@rm -rf dist
	@rm -rf *.egg-info
	@rm -rf htmlcov
	@rm -rf .tox/
	@rm -rf docs/_build
	@rm -rf .ruff_cache
	@rm -rf src/__pycache__
	@rm -rf src/*.egg-info

test:
	@$(PIP) install -e .[test] --quiet
	$(PYTEST) src/tests

format:
	@$(PIP) install -e .[format] --quiet
	$(PRE_COMMIT) run --all-files

precommit:
	@$(PIP) install -e .[test] --quiet
	$(PYTEST) src/tests
	@$(PIP) install -e .[format] --quiet
	$(PRE_COMMIT) run --all-files

.PHONY: init clean test precommit format
