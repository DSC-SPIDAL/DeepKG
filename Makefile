PHONY: clean clean-pyc clean-build help

help:
	@echo "clean - remove all build, test, coverage and Python artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "clean-build - remove build artifacts"

clean: clean-build clean-pyc
	@find . -name '.DS_Store' -exec rm -rf {} +

clean-build:
	@rm -fr build/
	@rm -fr dist/
	@rm -fr .eggs/
	@find . -name '*.egg-info' -exec rm -fr {} +
	@find . -name '*.egg' -exec rm -f {} +

clean-pyc:
	@find . -name '*.pyc' -exec rm -f {} +
	@find . -name '*.pyo' -exec rm -f {} +
	@find . -name '*~' -exec rm -f {} +
	@find . -name '__pycache__' -exec rm -rf {} +
	@find . -name '.pytest_cache' -exec rm -rf {} +
	@find . -name '.coverage' -exec rm -f {} +
	@find . -name 'htmlcov' -exec rm -rf {} +
