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
	flake8 featuretools_ta1 scripts/*.py
	isort -c --recursive featuretools_ta1 scripts/*.py

.PHONY: fix-lint
fix-lint: ## fix lint issues using autoflake, autopep8, and isort
	find featuretools_ta1 -name '*.py' | xargs autoflake --in-place --remove-all-unused-imports --remove-unused-variables
	autopep8 --in-place --recursive --aggressive featuretools_ta1
	isort --apply --atomic --recursive featuretools_ta1

	find scripts -name '*.py' | xargs autoflake --in-place --remove-all-unused-imports --remove-unused-variables
	autopep8 --in-place --recursive --aggressive scripts
	isort --apply --atomic --recursive scripts


# TEST TARGETS

.PHONY: test
test: ## run tests quickly with the default Python
	python scripts/run_pipeline.py

# D3M TARGETS

.PHONY: describe
describe: ## Generate the primitive annotations using d3m tools
	rm -rf MIT_FeatureLabs
	python scripts/describe_primitives.py

.PHONY: describe-commit
describe-commit: describe ## Genenrate and commit the primitive annotations
	git add MIT_FeatureLabs
	git commit -m'Add primitive descriptions' MIT_FeatureLabs

# RELEASE TARGETS

.PHONY: bumpversion-release
bumpversion-release: ## Merge master to stable and bumpversion release
	git checkout stable || git checkout -b stable
	git merge --no-ff master -m"make release-tag: Merge branch 'master' into stable"
	bumpversion release
	git push --tags origin stable

.PHONY: bumpversion-minor
bumpversion-minor: ## Bump the version the next minor skipping the release
	bumpversion --no-tag minor

.PHONY: bumpversion-major
bumpversion-major: ## Bump the version the next major skipping the release
	bumpversion --no-tag major

.PHONY: bumpversion-patch
bumpversion-patch: ## Merge stable to master and bumpversion patch
	git checkout master
	git merge stable
	bumpversion --no-tag patch
	git push

.PHONY: release
release: bumpversion-release describe-commit bumpversion-patch

.PHONY: release-minor
release-minor: bumpversion-minor release

.PHONY: release-major
release-major: bumpversion-major release

CURRENT_VERSION := $(shell grep -m1 current_version setup.cfg | cut -d' ' -f3 2>/dev/null)
LATEST_VERSION := $(shell git tag | tail -n1 | cut -c2- 2>/dev/null)

.PHONY: rollback
rollback: ## Rollback the latest release
	sed "s/$(CURRENT_VERSION)/$(LATEST_VERSION)-dev/g" -i setup.py setup.cfg featuretools_ta1/__init__.py
	git add setup.py setup.cfg featuretools_ta1/__init__.py
	git commit -m"Rollback release $(LATEST_VERSION)"
	git tag -d v$(LATEST_VERSION)
	git push --delete origin v$(LATEST_VERSION)
	git push

.PHONY: generate_pipelines
generate_pipelines: # Generate test pipelines
	sh /featuretools_ta1/generate_pipelines.sh

.PHONY: run_pipelines
run_pipelines: # Generate test pipelines
	sh /featuretools_ta1/run_pipelines.sh

.PHONY: do_submission
do_submission: # Generate test pipelines
	sh /featuretools_ta1/do_submission.sh

.PHONY: docker
docker: # Get latest base and build image
	docker pull registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.5.18-20200630-050709
	docker build -t d3mft .
