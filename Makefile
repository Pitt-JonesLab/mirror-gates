PYTHON_VERSION = python3.9
PIP = .venv/bin/pip
PYTEST = .venv/bin/pytest
PRE_COMMIT = .venv/bin/pre-commit

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
	$(PRE_COMMIT) run --all-files

.PHONY: init test precommit
