# Variables
PYTHON_VERSION = python3.9
PIP = .venv/bin/pip
PYTEST = .venv/bin/pytest
PRE_COMMIT = .venv/bin/pre-commit

# Targets that do not create any files
.PHONY: venv-setup init dev-init clean test precommit format transpile_benchy monodromy

# Initialize virtual environment
venv-setup:
	rm -rf .venv
	$(PYTHON_VERSION) -m venv .venv
	@$(PIP) install --upgrade pip

# Install main dependencies
# XXX: make init currently unstable due to some package conflicts
# instead use make dev-init
init: venv-setup
	$(PIP) install -e .[core]

# Setup development environment
# NOTE: Instead of pip installing 'core' dependencies directly from git repositories,
# the relevant repositories are cloned into a sibling directory and installed in editable mode.
dev-init: venv-setup install-dev-deps pre-commit-setup transpile_benchy monodromy
	$(PIP) install qiskit==0.43.3
	$(PIP) install git+https://github.com/evmckinney9/qiskit-evmckinney9.git@sqisw-gate

# force using my fork which includes the incomplete sqiswap decomposition PR
install-dev-deps:
	$(PIP) install -e .[dev]

pre-commit-setup:
	@$(PRE_COMMIT) install
	@$(PRE_COMMIT) autoupdate

# Install or update transpile_benchy repo
transpile_benchy:
	if [ -d "../transpile_benchy" ]; then \
		echo "Repository already exists. Updating with latest changes."; \
		cd ../transpile_benchy && git pull && git checkout main; \
	else \
		cd .. && git clone https://github.com/evmckinney9/transpile_benchy.git --recurse-submodules; \
		cd transpile_benchy && git checkout main; \
	fi
	$(PIP) install -e ../transpile_benchy --quiet

# Install or update monodromy repo
monodromy:
	if [ -d "../monodromy" ]; then \
		echo "Repository already exists. Updating with latest changes."; \
		cd ../monodromy && git pull; \
	else \
		cd .. && git clone https://github.com/evmckinney9/monodromy.git; \
		cd monodromy; \
	fi
	$(PIP) install -e ../monodromy --quiet

clean: movefigs
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

movefigs:
	@find ./src/ -type f -name '*.pdf' -exec mv {} ./docs/images/ \; 2>/dev/null || true
	@find ./src/ -type f -name '*.png' -exec mv {} ./docs/images/ \; 2>/dev/null || true
	@find ./src/ -type f -name '*.svg' -exec mv {} ./docs/images/ \; 2>/dev/null || true

test:
	@$(PIP) install -e .[test] --quiet
	$(PYTEST) src/tests

format:
	@$(PIP) install -e .[format] --quiet
	$(PRE_COMMIT) run --all-files

precommit: test format
