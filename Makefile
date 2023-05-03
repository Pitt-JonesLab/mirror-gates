PYTHON_VERSION = python3.9
PIP = .venv/bin/pip
PYTEST = .venv/bin/pytest
PRE_COMMIT = .venv/bin/pre-commit

init:
	$(PYTHON_VERSION) -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e .[dev]
	$(PIP) pre-commit install

test:
	$(PIP) install -e .[test] --quiet
	$(PYTEST) src/tests

format:
	$(PIP) install -e .[format] --quiet
	$(PRE_COMMIT) run --all-files

precommit:
	$(PIP) install -e .[test] --quiet
	$(PYTEST) src/tests
	$(PIP) install -e .[format] --quiet
	$(PRE_COMMIT) run --all-files

.PHONY: init test precommit format
