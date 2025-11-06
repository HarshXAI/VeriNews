# Fake News Propagation Detection using Graph Attention Networks (GAT)

A research-grade implementation for detecting and analyzing fake news propagation patterns on social media using the FakeNewsNet dataset.

## 🎯 Project Objectives

- **Find Key Spreaders** – Identify influential users using attention weights
- **Map Full Spread** – Build propagation trees for each news item
- **Build Simple, Strong Model** – Efficient GAT without heavy architecture
- **Get Better Results** – Combine user influence, post content, and source credibility
- **Explain Decisions** – Interpret attention weights and propagation paths

## 📊 Dataset

**FakeNewsNet**: https://github.com/KaiDMML/FakeNewsNet

Components:
- News content (source, headline, body text, images/videos)
- Social context (user profiles, content, followers, followees)
- User-user and user-post interaction graphs

## 🏗️ Project Structure

```
majorProject/
├── data/                          # Raw and processed data
│   ├── raw/                       # FakeNewsNet raw JSON files
│   ├── processed/                 # Cleaned and preprocessed data
│   └── graphs/                    # Constructed graph structures
├── src/                           # Source code
│   ├── data/                      # Data ingestion and preprocessing
│   ├── features/                  # Feature engineering
│   ├── models/                    # GAT model definitions
│   ├── training/                  # Training scripts
│   ├── evaluation/                # Evaluation and metrics
│   └── visualization/             # Attention maps and graphs
├── notebooks/                     # Jupyter notebooks for exploration
├── configs/                       # Configuration files
├── experiments/                   # Training logs and checkpoints
├── outputs/                       # Results, predictions, reports
└── tests/                         # Unit tests
```

## 🚀 Setup Instructions

### Prerequisites
- Python 3.10+
- uv package manager

### Installation

1. **Install uv** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Create virtual environment and install dependencies**:
```bash
# Initialize project with uv
uv venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux

# Install dependencies
uv pip install -e .
```

3. **Download FakeNewsNet dataset**:
```bash
python scripts/download_dataset.py
```

## 🔬 Research Pipeline

### Stage 1: Data Ingestion and Preprocessing
```bash
python src/data/preprocess.py --input data/raw --output data/processed
```

### Stage 2: Graph Construction and Feature Engineering
```bash
python src/features/build_graph.py --config configs/graph_config.yaml
```

### Stage 3: Model Training
```bash
python src/training/train_gat.py --config configs/model_config.yaml
```

### Stage 4: Evaluation and Analysis
```bash
python src/evaluation/evaluate.py --checkpoint experiments/best_model.pt
python src/visualization/generate_reports.py
```

## 📈 Expected Results

### Metrics
- Accuracy
- F1-Score
- AUC-ROC
- Explanation Fidelity

### Visualizations
- Propagation graphs
- Attention-based influence maps
- Confusion matrices
- Performance summaries

## 🧪 Usage Examples

### Training a GAT Model
```python
from src.models.gat_model import FakeNewsGAT
from src.training.trainer import GATTrainer

model = FakeNewsGAT(
    in_channels=768,
    hidden_channels=128,
    num_classes=2,
    num_layers=3
)

trainer = GATTrainer(model, train_loader, val_loader)
trainer.train(epochs=100)
```

### Analyzing Key Spreaders
```python
from src.evaluation.explainability import AttentionAnalyzer

analyzer = AttentionAnalyzer(model, graph_data)
key_spreaders = analyzer.identify_influential_users(top_k=20)
analyzer.visualize_propagation_tree(news_id='article_123')
```

## 📚 Tech Stack

- **PyTorch Geometric**: Graph neural networks
- **Transformers (HuggingFace)**: BERT embeddings
- **NetworkX**: Graph construction and analysis
- **scikit-learn**: Classical ML metrics and preprocessing
- **matplotlib/seaborn**: Visualization

## 🔍 Interpretability

All predictions are accompanied by:
- Attention weight distributions
- Top influential users in propagation chains
- Propagation path visualizations
- Feature importance analysis

## 📝 Notes

- **Purpose**: Research and analysis only
- **Focus**: Code efficiency and model accuracy
- **Interpretability**: Attention weights logged and visualized for all predictions

## 📄 License

MIT License - For research purposes only

## 🤝 Contributing

This is a research project. For questions or collaboration, please open an issue.

## 📧 Contact

[Your contact information]
