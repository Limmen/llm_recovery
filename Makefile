.PHONY: clean

all: install_dev install unit_tests tests release

release: build push

install_dev:
	pip install -r requirements_dev.txt

tests:
	tox

unit_tests:
	pytest --cov=llm_recovery

types:
	mypy .

lint:
	flake8 .

install:
	pip install -e .

build:
	python -m build

push:
	python -m twine upload dist/*

clean:
	rm -r dist