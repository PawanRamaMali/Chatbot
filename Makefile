.PHONY: help install install-dev test lint format clean build docker-build docker-run docs

help:				## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:			## Install the package
	pip install -e .

install-dev:		## Install development dependencies
	pip install -e ".[dev]"
	pre-commit install

test:				## Run tests
	pytest tests/ -v

test-cov:			## Run tests with coverage
	pytest tests/ -v --cov=neural_chatbot --cov-report=html --cov-report=term

lint:				## Run linting
	flake8 src/ tests/
	mypy src/neural_chatbot

format:				## Format code
	black src/ tests/
	isort src/ tests/

clean:				## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build:				## Build package
	python setup.py sdist bdist_wheel

docker-build:		## Build Docker image
	docker build -t neural-chatbot .

docker-run:			## Run Docker container
	docker run -p 5000:5000 neural-chatbot

docker-compose-up:	## Start with docker-compose
	docker-compose up -d

docker-compose-down: ## Stop docker-compose
	docker-compose down

docs:				## Build documentation
	cd docs && make html

train:				## Train the model
	neural-chatbot train --plot

serve:				## Start the web server
	neural-chatbot serve

chat:				## Start interactive chat
	neural-chatbot chat

