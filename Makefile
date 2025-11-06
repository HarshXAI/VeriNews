# Research Project Makefile
# Convenient commands for development and experimentation

.PHONY: help setup install clean test lint format download preprocess train evaluate all

# Default target
help:
	@echo "Fake News GAT - Available Commands"
	@echo "=================================="
	@echo "setup         - Run initial setup (install dependencies)"
	@echo "install       - Install project dependencies with uv"
	@echo "download      - Download FakeNewsNet dataset"
	@echo "preprocess    - Preprocess data"
	@echo "build-graphs  - Build propagation graphs"
	@echo "train         - Train GAT model"
	@echo "evaluate      - Evaluate trained model"
	@echo "test          - Run unit tests"
	@echo "lint          - Run code quality checks"
	@echo "format        - Format code with black and isort"
	@echo "clean         - Clean generated files"
	@echo "all           - Run complete pipeline"
	@echo ""

# Setup
setup:
	@echo "Running setup..."
	chmod +x setup.sh
	./setup.sh

# Install dependencies
install:
	@echo "Installing dependencies with uv..."
	uv pip install -e .
	uv pip install -e ".[dev]"

# Download dataset
download:
	@echo "Downloading FakeNewsNet dataset..."
	python scripts/download_dataset.py

# Preprocess data
preprocess:
	@echo "Preprocessing data..."
	python scripts/preprocess_data.py

# Build graphs
build-graphs:
	@echo "Building propagation graphs..."
	python scripts/build_graphs.py

# Train model
train:
	@echo "Training GAT model..."
	python scripts/train_model.py

# Evaluate model
evaluate:
	@echo "Evaluating model..."
	python scripts/evaluate_model.py

# Run tests
test:
	@echo "Running tests..."
	pytest tests/ -v --cov=src --cov-report=html

# Lint code
lint:
	@echo "Running linters..."
	flake8 src/ tests/ scripts/
	mypy src/

# Format code
format:
	@echo "Formatting code..."
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf **/__pycache__
	rm -rf **/*.pyc
	rm -rf *.egg-info
	rm -rf build/
	rm -rf dist/

# Run complete pipeline
all: download preprocess build-graphs train evaluate
	@echo "Complete pipeline finished!"

# Quick start (after setup)
quickstart: download preprocess
	@echo "Quick start complete! Ready to build graphs and train."
	@echo "Run: make build-graphs && make train"

# Development mode
dev:
	@echo "Starting development environment..."
	jupyter lab

# Check environment
check:
	@echo "Checking environment..."
	@python -c "import torch; import torch_geometric; print('✅ PyTorch:', torch.__version__); print('✅ PyG:', torch_geometric.__version__)"
	@python -c "from src.models import FakeNewsGAT; print('✅ Models imported successfully')"
	@echo "✅ Environment check passed!"
