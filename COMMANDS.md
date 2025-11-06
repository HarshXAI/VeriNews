# Command Reference Guide

Quick reference for all available commands in the Fake News GAT project.

---

## 🚀 Environment Management

### Activate Virtual Environment
```bash
source .venv/bin/activate
```

### Deactivate Virtual Environment
```bash
deactivate
```

### Check Environment Status
```bash
make check
# Or manually:
python -c "import torch; import torch_geometric; print('✅ Environment OK')"
```

### Reinstall Dependencies
```bash
uv pip install -e .
uv pip install -e ".[dev]"
```

---

## 📊 Data Pipeline

### Download Dataset
```bash
# Using script
python scripts/download_dataset.py

# Using Makefile
make download

# With custom output directory
python scripts/download_dataset.py --output data/raw/custom_location
```

### Preprocess Data
```bash
# Using script
python scripts/preprocess_data.py

# Using Makefile
make preprocess

# With custom directories
python scripts/preprocess_data.py \
    --input data/raw/fakenewsnet \
    --output data/processed
```

### Build Graphs
```bash
# Using script
python scripts/build_graphs.py

# Using Makefile
make build-graphs

# With options
python scripts/build_graphs.py \
    --input data/processed \
    --output data/graphs \
    --max-samples 1000 \
    --device cuda
```

---

## 🧠 Model Training

### Train Model
```bash
# Using example workflow
python scripts/example_workflow.py

# Using training script
python scripts/train_model.py --config configs/model_config.yaml

# Using Makefile
make train
```

### Resume Training
```python
# In Python
from src.training import GATTrainer

trainer = GATTrainer(model, train_loader, val_loader)
trainer.load_checkpoint("experiments/last_model.pt")
trainer.train(epochs=50)  # Continue training
```

---

## 📈 Evaluation

### Evaluate Model
```bash
# Basic evaluation
python scripts/evaluate_model.py

# With attention analysis
python scripts/evaluate_model.py --analyze-attention

# Custom checkpoint
python scripts/evaluate_model.py \
    --checkpoint experiments/best_model.pt \
    --data data/graphs/test_data.pt \
    --output outputs/results

# Using Makefile
make evaluate
```

---

## 🔬 Analysis & Visualization

### Run Jupyter Notebooks
```bash
# Start Jupyter Lab
jupyter lab

# Start Jupyter Notebook
jupyter notebook

# Open specific notebook
jupyter notebook notebooks/01_data_exploration.md
```

### Generate Visualizations
```python
# In Python
from src.visualization import GraphVisualizer, MetricsVisualizer

# Graph visualizations
graph_viz = GraphVisualizer()
graph_viz.plot_propagation_tree(tree, root_node=0, save_path='outputs/tree.png')

# Metrics visualizations
metrics_viz = MetricsVisualizer()
metrics_viz.plot_confusion_matrix(cm, save_path='outputs/cm.png')
metrics_viz.plot_metrics_comparison(metrics, save_path='outputs/metrics.png')
```

---

## 🧪 Testing

### Run All Tests
```bash
# Using pytest
pytest tests/ -v

# Using Makefile
make test

# With coverage
pytest tests/ --cov=src --cov-report=html

# Open coverage report
open htmlcov/index.html
```

### Run Specific Tests
```bash
# Test data module
pytest tests/test_data.py -v

# Test models module
pytest tests/test_models.py -v

# Test specific function
pytest tests/test_models.py::TestFakeNewsGAT::test_forward_pass -v
```

---

## 🎨 Code Quality

### Format Code
```bash
# Format with black
black src/ tests/ scripts/

# Sort imports
isort src/ tests/ scripts/

# Using Makefile (both)
make format
```

### Lint Code
```bash
# Run flake8
flake8 src/ tests/ scripts/

# Run mypy
mypy src/

# Using Makefile (both)
make lint
```

---

## 🛠️ Utility Commands

### Count Model Parameters
```python
from src.utils import count_parameters
from src.models import FakeNewsGAT

model = FakeNewsGAT(768, 128, 2, 3, 8)
print(f"Parameters: {count_parameters(model):,}")
```

### Get Device
```python
from src.utils import get_device

device = get_device()  # Returns: 'cuda', 'mps', or 'cpu'
```

### Set Random Seed
```python
from src.utils import set_seed

set_seed(42)  # For reproducibility
```

### Load/Save Configuration
```python
from src.utils import load_config, save_config

# Load config
config = load_config('configs/model_config.yaml')

# Modify and save
config['training']['epochs'] = 150
save_config(config, 'configs/custom_config.yaml')
```

### Save/Load Checkpoint
```python
from src.utils import save_checkpoint, load_checkpoint

# Save
save_checkpoint(model, optimizer, epoch, metrics, 'experiments/checkpoint.pt')

# Load
checkpoint = load_checkpoint('experiments/checkpoint.pt', model, optimizer)
```

---

## 📦 Package Management

### List Installed Packages
```bash
uv pip list
```

### Install New Package
```bash
uv pip install package-name
```

### Update Package
```bash
uv pip install --upgrade package-name
```

### Check for Updates
```bash
uv pip list --outdated
```

### Freeze Requirements
```bash
uv pip freeze > requirements.txt
```

---

## 🔄 Complete Pipelines

### Full Pipeline (Makefile)
```bash
# Run everything
make all

# Equivalent to:
make download
make preprocess
make build-graphs
make train
make evaluate
```

### Quick Start Pipeline
```bash
# After environment setup
make quickstart  # Downloads and preprocesses

# Then manually run:
make build-graphs
make train
```

### Development Workflow
```bash
# 1. Make changes to code
vim src/models/gat_model.py

# 2. Format code
make format

# 3. Run tests
make test

# 4. Run lint checks
make lint

# 5. Train with new changes
make train
```

---

## 🧹 Cleanup

### Clean Generated Files
```bash
# Using Makefile
make clean

# Manually remove specific files
rm -rf __pycache__
rm -rf .pytest_cache
rm -rf htmlcov
rm -rf *.egg-info
```

### Clean All Data (CAUTION!)
```bash
# This removes all data and models
rm -rf data/raw/*
rm -rf data/processed/*
rm -rf data/graphs/*
rm -rf experiments/*
rm -rf outputs/*
```

---

## 📊 Monitoring & Logging

### View TensorBoard Logs
```bash
# Start TensorBoard
tensorboard --logdir=experiments/tensorboard

# Open in browser
open http://localhost:6006
```

### View Weights & Biases
```bash
# Login to W&B
wandb login

# Then run training with W&B enabled
# (set USE_WANDB=true in .env)
python scripts/train_model.py
```

### Check Training Logs
```bash
# View logs
tail -f logs/training.log

# View specific experiment
cat experiments/experiment_001/training.log
```

---

## 🔧 Configuration

### View Current Config
```bash
cat configs/model_config.yaml
cat configs/graph_config.yaml
cat configs/preprocessing_config.yaml
```

### Edit Config
```bash
# Using vim
vim configs/model_config.yaml

# Using nano
nano configs/model_config.yaml

# Using VS Code
code configs/model_config.yaml
```

### Validate Config
```python
from src.utils import load_config
import yaml

try:
    config = load_config('configs/model_config.yaml')
    print("✅ Config is valid")
except yaml.YAMLError as e:
    print(f"❌ Config error: {e}")
```

---

## 🐍 Python Interactive

### Load Components in Python
```python
# Start Python
python

# Import modules
from src.data import FakeNewsNetLoader, TextPreprocessor
from src.features import TextEmbedder, PropagationGraphBuilder
from src.models import FakeNewsGAT
from src.training import GATTrainer
from src.evaluation import ModelEvaluator, AttentionAnalyzer
from src.visualization import GraphVisualizer, MetricsVisualizer
from src.utils import *

# Work interactively
loader = FakeNewsNetLoader('data/raw/fakenewsnet')
stats = loader.get_statistics()
print(stats)
```

### IPython Interactive
```bash
# Start IPython with better features
ipython

# With auto-reload for development
ipython
%load_ext autoreload
%autoreload 2
```

---

## 📝 Git Commands (Optional)

### Initialize Git Repository
```bash
git init
git add .
git commit -m "Initial commit: Fake News GAT project"
```

### Create .gitignore (Already included)
```bash
# Already created, but to view:
cat .gitignore
```

### Commit Changes
```bash
git add .
git commit -m "Add new feature"
git push origin main
```

---

## 🆘 Help & Documentation

### View Help for Scripts
```bash
# Any script with --help
python scripts/download_dataset.py --help
python scripts/preprocess_data.py --help
python scripts/build_graphs.py --help
python scripts/train_model.py --help
```

### View Makefile Commands
```bash
make help
```

### View Python Module Help
```python
import src.models.gat_model
help(src.models.gat_model.FakeNewsGAT)
```

### Open Documentation
```bash
# Open README
open README.md

# Open Quick Start
open QUICKSTART.md

# Open Research Guide
open RESEARCH_GUIDE.md
```

---

## 🎯 Quick Reference Table

| Task | Command | Time |
|------|---------|------|
| Setup | `./setup.sh` | 5 mins |
| Download | `make download` | 10 mins |
| Preprocess | `make preprocess` | 20 mins |
| Build Graphs | `make build-graphs` | 30 mins |
| Train | `make train` | 1-4 hours |
| Evaluate | `make evaluate` | 5 mins |
| Test | `make test` | 1 min |
| Format | `make format` | 10 secs |

---

## 💡 Tips & Tricks

### Speed Up Training
```bash
# Use GPU if available
export DEVICE=cuda

# Increase batch size (if memory allows)
# Edit configs/model_config.yaml: batch_size: 64
```

### Debug Mode
```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Profile Performance
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
trainer.train(epochs=10)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

---

**Last Updated:** November 6, 2025  
**Version:** 1.0.0  
**Status:** Complete

For more information, see `README.md` or `RESEARCH_GUIDE.md`.
