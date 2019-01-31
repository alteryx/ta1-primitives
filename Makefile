.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

.PHONY: help
help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)


# CLEAN TARGETS

.PHONY: clean-build
clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

.PHONY: clean-pyc
clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

.PHONY: clean
clean: clean-build clean-pyc clean-test clean-coverage clean-docs ## remove all build, test, coverage, docs and Python artifacts


# INSTALL TARGETS

.PHONY: install
install: clean-build clean-pyc ## install the package to the active Python's site-packages
	pip install .

.PHONY: install-develop
install-develop: clean-build clean-pyc ## install the package in editable mode and dependencies for development
	pip install -e . -r requirements_dev.txt


# LINT TARGETS

.PHONY: lint
lint: ## check style with flake8 and isort
	flake8 featuretools_ta1 scripts
	isort -c --recursive featuretools_ta1 scripts

.PHONY: fix-lint
fix-lint: ## fix lint issues using autoflake, autopep8, and isort
	find featuretools_ta1 -name '*.py' | xargs autoflake --in-place --remove-all-unused-imports --remove-unused-variables
	autopep8 --in-place --recursive --aggressive featuretools_ta1
	isort --apply --atomic --recursive featuretools_ta1

	find scripts -name '*.py' | xargs autoflake --in-place --remove-all-unused-imports --remove-unused-variables
	autopep8 --in-place --recursive --aggressive scripts
	isort --apply --atomic --recursive scripts


# TEST TARGETS

# .PHONY: test
# test: ## run tests quickly with the default Python
# 	python -m pytest

# D3M TARGETS

.PHONY: describe
describe: ## run tests quickly with the default Python
	python scripts/describe_primitives.py
