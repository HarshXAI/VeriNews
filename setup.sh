#!/bin/bash

# Setup script for Fake News GAT project using uv

echo "=========================================="
echo "Fake News GAT Project Setup"
echo "=========================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Source the profile to update PATH
    if [ -f "$HOME/.zshrc" ]; then
        source "$HOME/.zshrc"
    elif [ -f "$HOME/.bashrc" ]; then
        source "$HOME/.bashrc"
    fi
else
    echo "✓ uv is already installed"
fi

echo ""
echo "Creating virtual environment with uv..."
uv venv

echo ""
echo "Activating virtual environment..."
source .venv/bin/activate

echo ""
echo "Installing project dependencies..."
uv pip install -e .

echo ""
echo "Installing development dependencies..."
uv pip install -e ".[dev]"

echo ""
echo "Creating data directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/graphs
mkdir -p data/cache
mkdir -p experiments
mkdir -p outputs
mkdir -p logs

echo ""
echo "Downloading required NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

echo ""
echo "Creating .env file..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✓ Created .env file (please configure as needed)"
else
    echo "✓ .env file already exists"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate the virtual environment:"
echo "     source .venv/bin/activate"
echo ""
echo "  2. Download the FakeNewsNet dataset:"
echo "     python scripts/download_dataset.py"
echo ""
echo "  3. Preprocess the data:"
echo "     python scripts/preprocess_data.py"
echo ""
echo "  4. Train the model:"
echo "     python scripts/train_model.py"
echo ""
echo "For more information, see README.md"
echo ""
