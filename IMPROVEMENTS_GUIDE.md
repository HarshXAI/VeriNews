# GAT Improvements Guide

This guide documents comprehensive improvements to boost F1 score for fake news detection using Graph Attention Networks.

## 🎯 Quick Start

### 1. Guard the Scoreboard (Fix Leakage & Optimization)

#### Temporal Splits

```bash
# Create metadata with timestamps and source IDs
python scripts/enhance_graph.py \
  --data-path data/graphs/graph_data.pt \
  --create-metadata \
  --metadata-path data/graphs/metadata.csv

# Train with temporal splits
python scripts/train_gat_improved.py \
  --data-path data/graphs/graph_data.pt \
  --metadata-path data/graphs/metadata.csv \
  --split-method temporal \
  --tune-threshold \
  --calibrate \
  --output-dir experiments/temporal_split
```

**Expected Impact**: ±1-3 F1 points if leakage existed

#### Source-Grouped Splits

```bash
python scripts/train_gat_improved.py \
  --split-method source_grouped \
  --metadata-path data/graphs/metadata.csv \
  --output-dir experiments/source_grouped
```

#### Threshold Tuning + Calibration

```bash
python scripts/train_gat_improved.py \
  --tune-threshold \
  --calibrate \
  --output-dir experiments/calibrated
```

**Expected Impact**: +0.5-1.0 F1

#### Focal Loss for Class Imbalance

```bash
python scripts/train_gat_improved.py \
  --loss focal \
  --focal-alpha 0.25 \
  --focal-gamma 2.0 \
  --output-dir experiments/focal_loss
```

**Expected Impact**: +0.3-1.0 F1

---

### 2. Add Missing Signal to the Graph

#### Edge Weights (Cosine Similarity)

```bash
# Enhance graph with edge weights
python scripts/enhance_graph.py \
  --data-path data/graphs/graph_data.pt \
  --output-path data/graphs/graph_data_enhanced.pt \
  --add-edge-weights \
  --add-edge-types

# Train with edge attributes
python scripts/train_gat_improved.py \
  --data-path data/graphs/graph_data_enhanced.pt \
  --use-edge-attr \
  --output-dir experiments/edge_weights
```

**Expected Impact**: +0.5-1.5 F1

---

### 3. Model & Training Improvements

#### GATv2 with Residual + LayerNorm

The `ImprovedGATv2` model includes:

- ✅ GATv2Conv (fixes static attention)
- ✅ Residual connections
- ✅ Layer normalization
- ✅ DropEdge regularization
- ✅ Better activation (ELU)

```bash
python scripts/train_gat_improved.py \
  --hidden-dim 192 \
  --num-layers 3 \
  --num-heads 6 \
  --dropout 0.5 \
  --drop-edge-p 0.2 \
  --output-dir experiments/gatv2
```

**Expected Impact**: +0.5-1.5 F1

#### Hyperparameter Optimization

```bash
# Find best hyperparameters with Optuna
python scripts/hyperparameter_optimization.py \
  --data-path data/graphs/graph_data.pt \
  --n-trials 60 \
  --max-epochs 100 \
  --output-dir experiments/optuna_study

# Train with best parameters
python scripts/train_gat_improved.py \
  --hidden-dim 192 \
  --num-layers 3 \
  --num-heads 6 \
  --dropout 0.45 \
  --lr 0.0005 \
  --weight-decay 0.001 \
  --output-dir experiments/best_params
```

**Expected Impact**: +0.5-1.0 F1

---

## 🚀 Complete Pipeline (Maximum Impact)

Run all improvements together:

```bash
# 1. Enhance graph
python scripts/enhance_graph.py \
  --data-path data/graphs/graph_data.pt \
  --output-path data/graphs/graph_data_enhanced.pt \
  --add-edge-weights \
  --add-edge-types \
  --create-metadata \
  --metadata-path data/graphs/metadata.csv

# 2. Hyperparameter optimization
python scripts/hyperparameter_optimization.py \
  --data-path data/graphs/graph_data_enhanced.pt \
  --n-trials 60 \
  --split-method temporal \
  --output-dir experiments/optuna_study

# 3. Train final model with best settings
python scripts/train_gat_improved.py \
  --data-path data/graphs/graph_data_enhanced.pt \
  --metadata-path data/graphs/metadata.csv \
  --split-method temporal \
  --use-edge-attr \
  --loss focal \
  --focal-alpha 0.25 \
  --focal-gamma 2.0 \
  --tune-threshold \
  --calibrate \
  --hidden-dim 192 \
  --num-layers 3 \
  --num-heads 6 \
  --dropout 0.5 \
  --drop-edge-p 0.2 \
  --lr 0.0005 \
  --weight-decay 0.001 \
  --epochs 150 \
  --patience 25 \
  --save-predictions \
  --output-dir experiments/final_model
```

**Total Expected Impact**: +3-7 F1 points

---

## 📊 Feature Comparison

| Feature        | Old Model     | Improved Model               |
| -------------- | ------------- | ---------------------------- |
| Architecture   | GAT           | GATv2 + Residual + LayerNorm |
| Attention      | Static        | Dynamic (GATv2)              |
| Edge Features  | None          | Weights + Types              |
| Regularization | Dropout       | Dropout + DropEdge           |
| Loss           | Cross-Entropy | Focal Loss                   |
| Splits         | Random        | Temporal/Source-Grouped      |
| Threshold      | 0.5           | Tuned on Val Set             |
| Calibration    | None          | Temperature Scaling          |
| Optimization   | Manual        | Optuna (60 trials)           |

---

## 📈 Performance Tracking

Monitor improvements by comparing results:

```bash
# Compare experiments
python scripts/compare_results.py \
  experiments/baseline/results.json \
  experiments/final_model/results.json
```

Track key metrics:

- **F1 Score** (primary)
- **AUC-ROC** (discrimination)
- **Calibration ECE** (reliability)
- **Precision/Recall** (balance)

---

## 🔬 Ablation Studies

To understand which improvements matter most:

### Edge Weight Ablation

```bash
# Without edge weights
python scripts/train_gat_improved.py \
  --data-path data/graphs/graph_data.pt \
  --output-dir experiments/ablation/no_edge_weights

# With edge weights
python scripts/train_gat_improved.py \
  --data-path data/graphs/graph_data_enhanced.pt \
  --use-edge-attr \
  --output-dir experiments/ablation/with_edge_weights
```

### Focal Loss Ablation

```bash
# Cross-entropy
python scripts/train_gat_improved.py --loss ce \
  --output-dir experiments/ablation/ce_loss

# Focal loss
python scripts/train_gat_improved.py --loss focal \
  --output-dir experiments/ablation/focal_loss
```

### Split Method Ablation

```bash
# Random
python scripts/train_gat_improved.py --split-method random \
  --output-dir experiments/ablation/random_split

# Temporal
python scripts/train_gat_improved.py --split-method temporal \
  --metadata-path data/graphs/metadata.csv \
  --output-dir experiments/ablation/temporal_split
```

---

## 🛠️ Advanced: Heterogeneous Graphs

To upgrade to heterogeneous graph (article↔article, article↔source):

```python
# See src/features/hetero_graph_builder.py (to be created)
from torch_geometric.data import HeteroData

# Build hetero graph
data = HeteroData()
data['article'].x = article_features
data['source'].x = source_features
data['article', 'similar_to', 'article'].edge_index = similarity_edges
data['article', 'from', 'source'].edge_index = source_edges

# Use HGT or R-GAT model
from torch_geometric.nn import HGTConv, RGCNConv
```

**Expected Impact**: +1-2 F1 if source/entity signals are strong

---

## 📦 Requirements

Install additional dependencies:

```bash
pip install optuna
pip install scikit-learn
pip install joblib
```

---

## 🎓 Citation

If using these improvements in research:

```bibtex
@article{improved_gat_fake_news,
  title={Enhanced Graph Attention Networks for Fake News Detection},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

---

## 📝 TODO: Future Improvements

High-impact items not yet implemented:

1. **Self-supervised pretraining** (GraphCL/GRACE)
   - Impact: +0.5-1.5 F1
2. **Ensemble methods** (GATv2 + GCNII + SAGE)

   - Impact: +0.5-1.0 F1

3. **Label propagation refinement**

   - Impact: +0.3-0.8 F1

4. **Better text embeddings** (E5, MiniLM-L6 on title+body)

   - Impact: +0.5-1.0 F1

5. **Source credibility embeddings**

   - Impact: +0.5-1.0 F1

6. **Entity-overlap edges** (NER-based)
   - Impact: +0.3-0.8 F1

---

## 🐛 Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
--batch-size 128

# Reduce neighbor sampling
--num-neighbors 10 5 3

# Use gradient checkpointing (to be implemented)
```

### Poor Calibration

```bash
# Always use temperature scaling
--calibrate

# Check ECE metric in results.json
```

### Overfitting

```bash
# Increase regularization
--dropout 0.6 \
--drop-edge-p 0.3 \
--weight-decay 0.002

# More aggressive early stopping
--patience 15
```

---

## 📊 Expected Results Timeline

| Baseline   | After Quick Wins | After Full Pipeline |
| ---------- | ---------------- | ------------------- |
| F1: 0.750  | F1: 0.770-0.800  | F1: 0.820-0.850     |
| AUC: 0.820 | AUC: 0.850-0.870 | AUC: 0.880-0.910    |

Time investment:

- Quick wins (1-2 hours): +2-5 F1 points
- Full pipeline (4-6 hours): +5-10 F1 points
- With hyperopt (12-24 hours): +7-12 F1 points

---

## 📧 Support

For issues or questions:

1. Check experiments/\*/results.json for detailed metrics
2. Review logs in experiments/\*/training.log
3. Compare with baseline using compare_results.py

Good luck! 🚀
