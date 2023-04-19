PYTHON_VERSION = python3.9
PIP = .venv/bin/pip
PYTEST = .venv/bin/pytest

init:
	$(PYTHON_VERSION) -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e .[dev]

test:
	$(PIP) install -e .[test]
	$(PYTEST) src/tests

precommit:
	$(PIP) install -e .[test]
	$(PYTEST) src/tests
	$(PIP) install -e .[format]
	pre-commit run --all-files

.PHONY: init test precommit
