.ONESHELL:
#* Variables
SHELL := bash
PYTHON := python
PYTHONPATH := `pwd`
CONDA := conda
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

#* Installation
.PHONY: install
install:
	# ! type $(CONDA) &> /dev/null && { echo "Please install conda (https://docs.conda.io/en/latest/miniconda.html)"; exit 1; }
	# ! type pipx &> /dev/null && { echo "Please install pipx (https://github.com/pypa/pipx)"; exit 1; }

	# # TODO ensure version of poetry is correct
	# ! type poetry &> /dev/null && pipx install poetry==1.7.1

	# install reward_learning conda environment
	$(CONDA) create -n reward_learning python=3.10 -y
	$(CONDA_ACTIVATE) reward_learning

	type python
	
	poetry lock -n && poetry export --without-hashes > requirements.txt
	poetry install -n
	poetry run pre-commit install
	# poetry run mypy --install-types --non-interactive ./ 

#* Linting
.PHONY: lint
lint: test codestyle mypy


.PHONY: test
test:
	PYTHONPATH=$(PYTHONPATH) poetry run pytest -c pyproject.toml tests/

.PHONY: mypy
mypy:
	poetry run mypy --config-file pyproject.toml ./

.PHONY: update-dev-deps
update-dev-deps:
	poetry add -D bandit@latest darglint@latest "isort[colors]@latest" mypy@latest pre-commit@latest pydocstyle@latest pylint@latest pytest@latest pyupgrade@latest safety@latest coverage@latest coverage-badge@latest pytest-html@latest pytest-cov@latest
	poetry add -D --allorereleases black@latest

#* Formatters
.PHONY: codestyle
codestyle:
	poetry run pyupgrade --exit-zero-even-if-changed y3lus **/*.py
	poetry run isort --settingath pyproject.toml ./
	poetry run black --config pyproject.toml ./

.PHONY: formatting
formatting: codestyle
