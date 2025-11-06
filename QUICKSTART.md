# Fake News GAT Project - Quick Start Guide

## ✅ Environment Setup Complete!

Your research environment has been successfully set up with `uv`. All dependencies are installed and ready to use.

## 📋 What's Included

### Core Components
- **PyTorch & PyTorch Geometric**: Deep learning and graph neural networks
- **Transformers**: BERT embeddings for text
- **NetworkX**: Graph construction and analysis
- **scikit-learn**: Classical ML metrics
- **matplotlib/seaborn**: Visualization

### Project Structure
```
majorProject/
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── features/          # Graph construction and embeddings
│   ├── models/            # GAT model definitions
│   ├── training/          # Training utilities
│   ├── evaluation/        # Metrics and explainability
│   └── visualization/     # Plotting utilities
├── configs/               # YAML configuration files
├── scripts/               # Executable scripts
├── notebooks/             # Jupyter notebooks for exploration
├── tests/                 # Unit tests
└── data/                  # Data storage (created)
```

## 🚀 Getting Started

### Step 1: Activate Virtual Environment
```bash
source .venv/bin/activate
```

### Step 2: Download FakeNewsNet Dataset
```bash
python scripts/download_dataset.py
```

This will clone the FakeNewsNet repository into `data/raw/fakenewsnet/`.

**Expected structure:**
```
data/raw/fakenewsnet/
├── politifact/
│   ├── fake/
│   └── real/
└── gossipcop/
    ├── fake/
    └── real/
```

### Step 3: Preprocess Data
```bash
python scripts/preprocess_data.py
```

This will:
- Load news articles and social posts
- Clean and tokenize text
- Extract user features
- Save processed data to `data/processed/`

### Step 4: Explore Data (Optional)
```bash
jupyter notebook notebooks/01_data_exploration.md
```

### Step 5: Build Graphs
Create your graph construction script or use Python:

```python
from src.features import PropagationGraphBuilder, TextEmbedder
from src.data import FakeNewsNetLoader
import pandas as pd

# Load processed data
news_df = pd.read_parquet('data/processed/news_processed.parquet')
social_df = pd.read_parquet('data/processed/social_processed.parquet')

# Build propagation graphs
builder = PropagationGraphBuilder()
graph = builder.build_graph(social_df, social_df)

# Generate embeddings
embedder = TextEmbedder()
embeddings = embedder.embed_texts(news_df['text_clean'].tolist())
```

### Step 6: Train GAT Model
```bash
python scripts/train_model.py --config configs/model_config.yaml
```

Or in a notebook:
```python
from src.models import FakeNewsGAT
from src.training import GATTrainer

model = FakeNewsGAT(
    in_channels=768,
    hidden_channels=128,
    out_channels=2,
    num_layers=3,
    num_heads=8
)

trainer = GATTrainer(model, train_loader, val_loader)
trainer.train(epochs=100)
```

### Step 7: Evaluate and Analyze
```python
from src.evaluation import ModelEvaluator, AttentionAnalyzer
from src.visualization import GraphVisualizer, MetricsVisualizer

# Evaluate
evaluator = ModelEvaluator(model)
metrics = evaluator.evaluate_and_report(test_loader)

# Analyze attention
analyzer = AttentionAnalyzer(model)
influential_users = analyzer.identify_influential_users(attention_weights, top_k=20)

# Visualize
visualizer = GraphVisualizer()
visualizer.plot_influential_users(influential_users)
```

## 📊 Key Features

### 1. Data Preprocessing
- Text cleaning (URLs, mentions, emojis removal)
- Tokenization and stopword removal
- User feature extraction (followers, engagement, etc.)
- Graph construction from retweets, replies, mentions

### 2. Graph Attention Network
- Multi-layer GAT with configurable heads
- Attention weight extraction for interpretability
- Batch normalization and dropout
- Graph-level classification

### 3. Explainability
- **Attention Analysis**: Identify which users/posts the model focuses on
- **Key Spreaders**: Find influential users in propagation
- **Propagation Trees**: Visualize how news spreads
- **Feature Importance**: Understand which features matter

### 4. Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC curve
- Confusion matrix
- Propagation metrics (depth, breadth, viral coefficient)

## 🔧 Configuration

### Model Configuration (`configs/model_config.yaml`)
```yaml
model:
  architecture:
    in_channels: 768        # BERT embedding dimension
    hidden_channels: 128    # Hidden layer size
    num_layers: 3           # Number of GAT layers
    num_heads: 8            # Attention heads per layer
    dropout: 0.3            # Dropout rate

training:
  epochs: 100
  learning_rate: 0.001
  batch_size: 32
```

### Graph Configuration (`configs/graph_config.yaml`)
- Node types (user, post, source)
- Edge types (retweet, reply, mention, follows)
- Feature engineering settings
- Temporal window configuration

### Preprocessing Configuration (`configs/preprocessing_config.yaml`)
- Text cleaning options
- Tokenization method
- User feature selection
- Missing value handling

## 📈 Expected Results

Based on research benchmarks, you can expect:
- **Accuracy**: 85-92%
- **F1-Score**: 0.83-0.90
- **AUC-ROC**: 0.88-0.95

Results depend on:
- Dataset size and quality
- Feature engineering choices
- Model hyperparameters
- Graph construction methodology

## 🔍 Interpretability Features

### 1. Attention Weights
Visualize which nodes (users/posts) receive high attention:
```python
attention_weights = model.get_attention_weights()
visualizer.plot_attention_heatmap(attention_weights)
```

### 2. Key Spreaders Identification
Find the top 20 users who spread fake news:
```python
influential = analyzer.identify_influential_users(attention_weights, top_k=20)
```

### 3. Propagation Tree Visualization
See how a news article spreads through the network:
```python
tree = analyzer.build_propagation_tree(edge_index, attention, root_node=0)
visualizer.plot_propagation_tree(tree, root_node=0)
```

## 🧪 Testing

Run unit tests:
```bash
pytest tests/ -v --cov=src
```

Run specific test:
```bash
pytest tests/test_models.py::TestFakeNewsGAT::test_forward_pass
```

## 📝 Next Steps

1. **Experiment with Hyperparameters**: Try different layer counts, hidden dimensions, attention heads
2. **Feature Engineering**: Add temporal features, network metrics, sentiment analysis
3. **Model Variants**: Try GATv2, heterogeneous graphs, ensemble methods
4. **Deployment**: Export model for inference, create API endpoint
5. **Publication**: Document findings, create visualizations for papers

## 🆘 Troubleshooting

### Out of Memory
- Reduce batch size in `configs/model_config.yaml`
- Use fewer attention heads
- Reduce hidden dimensions

### Poor Performance
- Check data quality and preprocessing
- Increase model capacity (more layers/heads)
- Adjust learning rate
- Add more features

### Slow Training
- Use GPU: Set `device: cuda` in config
- Reduce graph size with sampling
- Use mixed precision training

## 📚 Resources

- **FakeNewsNet Dataset**: https://github.com/KaiDMML/FakeNewsNet
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/
- **GAT Paper**: https://arxiv.org/abs/1710.10903
- **Project Documentation**: See `README.md`

## 🤝 Contributing

To contribute:
1. Create a new branch
2. Make changes
3. Run tests: `pytest tests/`
4. Format code: `black src/ && isort src/`
5. Submit pull request

## 📧 Support

For issues or questions:
- Check documentation in `README.md`
- Review notebook examples
- Check configuration files
- Review error logs in `logs/`

---

**Good luck with your research! 🚀**
