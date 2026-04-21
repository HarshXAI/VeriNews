"""
Improved GAT/Graph Transformer Training v2
===========================================

Key improvements over train_gat_simple_scaled.py:
1. Focal loss for class imbalance (70% of errors were fake→real)
2. Cosine annealing LR schedule (plateau at epoch 50-88 in v1)
3. Stratified data splits with fixed seed
4. Weight decay regularization
5. Edge weight support in GATConv
6. Graph Transformer with Virtual Node architecture
7. Gradient clipping
8. Per-class metrics tracking
9. 5-fold cross-validation option

Current: 87.18% accuracy, 91.76% F1
Target:  92-94% accuracy, 93-95% F1
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import set_seed


# ===========================================================================
# LOSSES
# ===========================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss: down-weights easy examples, focuses on hard ones.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha[targets]
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ===========================================================================
# MODELS
# ===========================================================================

class ImprovedGATNode(nn.Module):
    """
    Improved GAT for node classification with:
    - Residual connections
    - Layer normalization
    - Edge weight support
    - Configurable GATv2
    """
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_heads=8, num_layers=3, dropout=0.3, use_gatv2=False):
        super().__init__()
        from torch_geometric.nn import GATConv, GATv2Conv

        Conv = GATv2Conv if use_gatv2 else GATConv

        self.input_proj = nn.Linear(in_channels, hidden_channels * num_heads)
        self.input_norm = nn.LayerNorm(hidden_channels * num_heads)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            if i == num_layers - 1:
                # Last layer: single head
                self.convs.append(Conv(
                    hidden_channels * num_heads, out_channels,
                    heads=1, concat=False, dropout=dropout
                ))
                self.norms.append(nn.LayerNorm(out_channels))
            else:
                self.convs.append(Conv(
                    hidden_channels * num_heads, hidden_channels,
                    heads=num_heads, dropout=dropout, concat=True
                ))
                self.norms.append(nn.LayerNorm(hidden_channels * num_heads))

        self.dropout = dropout
        self.num_layers = num_layers

    def forward(self, x, edge_index, edge_weight=None):
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = F.elu(x)

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            residual = x
            x = F.dropout(x, p=self.dropout, training=self.training)

            # GATConv doesn't naively support edge_weight; pass if available
            try:
                x = conv(x, edge_index, edge_attr=edge_weight)
            except TypeError:
                x = conv(x, edge_index)

            x = norm(x)

            if i < self.num_layers - 1:
                x = F.elu(x)
                # Residual connection (when dimensions match)
                if residual.shape == x.shape:
                    x = x + residual

        return x


class GraphTransformerVN(nn.Module):
    """
    Graph Transformer with Virtual Node for global context.
    Uses GATv2Conv for local attention + learnable virtual node.
    """
    def __init__(self, in_channels, hidden_channels=256, num_layers=4,
                 num_heads=8, dropout=0.2, num_classes=2):
        super().__init__()
        from torch_geometric.nn import GATv2Conv

        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.virtual_node = nn.Parameter(torch.randn(1, hidden_channels) * 0.02)

        self.gat_layers = nn.ModuleList()
        self.vn_updates = nn.ModuleList()
        self.node_updates = nn.ModuleList()
        self.norms1 = nn.ModuleList()
        self.norms2 = nn.ModuleList()
        self.norms3 = nn.ModuleList()

        for _ in range(num_layers):
            self.gat_layers.append(GATv2Conv(
                hidden_channels, hidden_channels, heads=num_heads,
                dropout=dropout, concat=False, add_self_loops=True
            ))
            self.vn_updates.append(nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels * 2, hidden_channels),
            ))
            self.node_updates.append(nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels * 2, hidden_channels),
            ))
            self.norms1.append(nn.LayerNorm(hidden_channels))
            self.norms2.append(nn.LayerNorm(hidden_channels))
            self.norms3.append(nn.LayerNorm(hidden_channels))

        self.output = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )

        self.dropout = dropout
        self.num_layers = num_layers

    def forward(self, x, edge_index, edge_weight=None):
        x = self.input_proj(x)
        vn = self.virtual_node

        for i in range(self.num_layers):
            # Local graph attention
            x_local = self.gat_layers[i](x, edge_index)
            x = self.norms1[i](x + F.dropout(x_local, p=self.dropout, training=self.training))

            # Update virtual node (aggregate global info)
            vn_input = x.mean(dim=0, keepdim=True)
            vn_update = self.vn_updates[i](vn_input)
            vn = self.norms2[i](vn + vn_update)

            # Broadcast virtual node to all nodes
            vn_broadcast = vn.expand(x.size(0), -1)
            x_combined = torch.cat([x, vn_broadcast], dim=1)
            x_update = self.node_updates[i](x_combined)
            x = self.norms3[i](x + x_update)

        return self.output(x)


# ===========================================================================
# DATA SPLITS
# ===========================================================================

def stratified_split(data, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Create stratified train/val/test splits preserving class distribution"""
    num_nodes = data.x.shape[0]
    labels = data.y.numpy()

    np.random.seed(seed)
    indices = np.arange(num_nodes)

    # Stratified split: first separate test, then train/val
    from sklearn.model_selection import StratifiedShuffleSplit

    # Split: trainval vs test
    test_ratio = 1.0 - train_ratio - val_ratio
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(indices, labels))

    # Split trainval into train and val
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio_adjusted, random_state=seed)
    trainval_labels = labels[trainval_idx]
    train_sub_idx, val_sub_idx = next(sss2.split(trainval_idx, trainval_labels))

    train_idx = trainval_idx[train_sub_idx]
    val_idx = trainval_idx[val_sub_idx]

    # Create masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # Print class distribution
    for name, mask in [('Train', train_mask), ('Val', val_mask), ('Test', test_mask)]:
        labels_split = data.y[mask]
        fake_pct = (labels_split == 0).sum().item() / len(labels_split) * 100 if len(labels_split) > 0 else 0
        real_pct = (labels_split == 1).sum().item() / len(labels_split) * 100 if len(labels_split) > 0 else 0
        print(f"  {name}: {mask.sum().item()} nodes (fake: {fake_pct:.1f}%, real: {real_pct:.1f}%)")

    return data


def kfold_splits(data, n_folds=5, seed=42):
    """Generate k-fold stratified splits"""
    labels = data.y.numpy()
    indices = np.arange(len(labels))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds = []
    for train_idx, test_idx in skf.split(indices, labels):
        # Split train into train/val (80/20)
        val_size = int(len(train_idx) * 0.15 / 0.85)
        val_idx = train_idx[:val_size]
        train_idx = train_idx[val_size:]

        train_mask = torch.zeros(len(labels), dtype=torch.bool)
        val_mask = torch.zeros(len(labels), dtype=torch.bool)
        test_mask = torch.zeros(len(labels), dtype=torch.bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        folds.append((train_mask, val_mask, test_mask))

    return folds


# ===========================================================================
# TRAINING
# ===========================================================================

def train_epoch(model, data, optimizer, criterion, device, grad_clip=1.0):
    """Train for one epoch with gradient clipping"""
    model.train()
    optimizer.zero_grad()

    edge_weight = getattr(data, 'edge_weight', None)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)

    out = model(data.x.to(device), data.edge_index.to(device), edge_weight)
    loss = criterion(out[data.train_mask], data.y[data.train_mask].to(device))

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    pred = out[data.train_mask].argmax(dim=1)
    acc = (pred == data.y[data.train_mask].to(device)).sum().item() / data.train_mask.sum().item()

    return loss.item(), acc


@torch.no_grad()
def evaluate(model, data, device, mask, criterion=None):
    """Evaluate model with comprehensive metrics"""
    model.eval()

    edge_weight = getattr(data, 'edge_weight', None)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)

    out = model(data.x.to(device), data.edge_index.to(device), edge_weight)
    pred = out[mask].argmax(dim=1).cpu()
    y_true = data.y[mask].cpu()

    acc = accuracy_score(y_true, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, pred, average='binary', zero_division=0)

    # Per-class metrics
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    class0_recall = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0
    class1_recall = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0

    # AUC
    try:
        probs = F.softmax(out[mask], dim=1)[:, 1].cpu().numpy()
        auc = roc_auc_score(y_true, probs)
    except Exception:
        auc = 0.0

    # Weighted F1
    _, _, f1_weighted, _ = precision_recall_fscore_support(y_true, pred, average='weighted', zero_division=0)

    loss = 0.0
    if criterion is not None:
        loss = criterion(out[mask], y_true.to(device)).item()

    return {
        'loss': loss,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'f1_weighted': f1_weighted,
        'auc': auc,
        'class0_recall': class0_recall,
        'class1_recall': class1_recall,
        'confusion_matrix': cm.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Improved GAT/Graph Transformer training v2")
    parser.add_argument("--data-path", type=str, default="data/graphs_full/graph_data_clean.pt")
    parser.add_argument("--output-dir", type=str, default="experiments/improved_v2")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)

    # Model architecture
    parser.add_argument("--model", type=str, default="graph_transformer",
                        choices=["gat", "gatv2", "graph_transformer"])
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)

    # Optimizer
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # LR scheduling
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "plateau", "none"])
    parser.add_argument("--warmup-epochs", type=int, default=10)

    # Loss
    parser.add_argument("--loss", type=str, default="focal",
                        choices=["ce", "focal", "weighted_ce"])
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, nargs='+', default=[0.6, 0.4],
                        help="Class weights for focal loss [fake_weight, real_weight]")

    # Data
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--kfold", type=int, default=0, help="Number of CV folds (0 = single split)")

    # Splits
    parser.add_argument("--splits-path", type=str, default=None,
                        help="Path to pre-saved splits .pt file")

    args = parser.parse_args()

    print("=" * 70)
    print("IMPROVED TRAINING v2")
    print("=" * 70)

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"\n  Device: {device}")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print(f"\n  Loading data from {args.data_path}...")
    data = torch.load(args.data_path, weights_only=False)
    print(f"  Nodes: {data.x.shape[0]}, Edges: {data.edge_index.shape[1]}")
    print(f"  Features: {data.x.shape[1]}")

    num_fake = (data.y == 0).sum().item()
    num_real = (data.y == 1).sum().item()
    print(f"  Labels: fake={num_fake}, real={num_real} (ratio {num_fake / (num_fake + num_real):.2%})")

    # K-fold or single split
    if args.kfold > 0:
        folds = kfold_splits(data, n_folds=args.kfold, seed=args.seed)
        print(f"\n  {args.kfold}-fold cross-validation")
    else:
        # Use pre-saved splits or create stratified split
        if args.splits_path and os.path.exists(args.splits_path):
            print(f"\n  Loading splits from {args.splits_path}...")
            splits = torch.load(args.splits_path, weights_only=False)
            data.train_mask = splits['train_mask']
            data.val_mask = splits['val_mask']
            data.test_mask = splits['test_mask']

            for name, mask in [('Train', data.train_mask), ('Val', data.val_mask), ('Test', data.test_mask)]:
                labels_split = data.y[mask]
                fake_pct = (labels_split == 0).sum().item() / len(labels_split) * 100
                print(f"  {name}: {mask.sum().item()} (fake: {fake_pct:.1f}%)")
        else:
            print(f"\n  Creating stratified splits (seed={args.seed})...")
            data = stratified_split(data, seed=args.seed)

            # Save splits for reproducibility
            torch.save({
                'train_mask': data.train_mask,
                'val_mask': data.val_mask,
                'test_mask': data.test_mask,
            }, os.path.join(args.output_dir, 'splits.pt'))

        folds = [(data.train_mask, data.val_mask, data.test_mask)]

    # Loss function
    if args.loss == 'focal':
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        print(f"\n  Loss: Focal (gamma={args.focal_gamma}, alpha={args.focal_alpha})")
    elif args.loss == 'weighted_ce':
        weight_fake = np.sqrt((num_fake + num_real) / max(1, num_fake))
        weight_real = np.sqrt((num_fake + num_real) / max(1, num_real))
        class_weights = torch.tensor([weight_fake, weight_real], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"\n  Loss: Weighted CE (weights=[{weight_fake:.3f}, {weight_real:.3f}])")
    else:
        criterion = nn.CrossEntropyLoss()
        print(f"\n  Loss: Cross-Entropy")

    # Run all folds
    all_fold_results = []

    for fold_idx, (train_mask, val_mask, test_mask) in enumerate(folds):
        if args.kfold > 0:
            print(f"\n{'=' * 70}")
            print(f"FOLD {fold_idx + 1}/{args.kfold}")
            print(f"{'=' * 70}")

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        set_seed(args.seed + fold_idx)

        # Create model
        in_channels = data.x.shape[1]
        if args.model == 'graph_transformer':
            model = GraphTransformerVN(
                in_channels=in_channels,
                hidden_channels=args.hidden_dim,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
            ).to(device)
        elif args.model == 'gatv2':
            model = ImprovedGATNode(
                in_channels=in_channels,
                hidden_channels=args.hidden_dim,
                out_channels=2,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                dropout=args.dropout,
                use_gatv2=True,
            ).to(device)
        else:
            model = ImprovedGATNode(
                in_channels=in_channels,
                hidden_channels=args.hidden_dim,
                out_channels=2,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                dropout=args.dropout,
                use_gatv2=False,
            ).to(device)

        num_params = sum(p.numel() for p in model.parameters())
        print(f"\n  Model: {args.model} ({num_params:,} params)")

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        # LR Scheduler
        if args.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=20, T_mult=2, eta_min=1e-6
            )
            print(f"  Scheduler: CosineAnnealingWarmRestarts (T_0=20, T_mult=2)")
        elif args.scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
            )
            print(f"  Scheduler: ReduceLROnPlateau")
        else:
            scheduler = None

        # Move data to device
        data_device = data.to(device)

        # Training loop
        print(f"\n  Training for up to {args.epochs} epochs (patience={args.patience})...")
        best_val_f1 = 0
        patience_counter = 0
        best_model_state = None
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_f1_weighted': [],
            'val_class0_recall': [], 'val_class1_recall': [],
            'lr': [],
        }

        start_time = time.time()

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_epoch(
                model, data_device, optimizer, criterion, device, args.grad_clip
            )

            # Validate
            val_metrics = evaluate(model, data_device, device, data_device.val_mask, criterion)

            # LR scheduling
            current_lr = optimizer.param_groups[0]['lr']
            if args.scheduler == 'cosine' and scheduler is not None:
                scheduler.step()
            elif args.scheduler == 'plateau' and scheduler is not None:
                scheduler.step(val_metrics['f1_weighted'])

            # Track history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1'])
            history['val_f1_weighted'].append(val_metrics['f1_weighted'])
            history['val_class0_recall'].append(val_metrics['class0_recall'])
            history['val_class1_recall'].append(val_metrics['class1_recall'])
            history['lr'].append(current_lr)

            # Print progress
            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:3d} | "
                    f"TLoss: {train_loss:.4f} TAcc: {train_acc:.4f} | "
                    f"VF1: {val_metrics['f1_weighted']:.4f} VAcc: {val_metrics['accuracy']:.4f} | "
                    f"Fake-R: {val_metrics['class0_recall']:.3f} Real-R: {val_metrics['class1_recall']:.3f} | "
                    f"LR: {current_lr:.6f}"
                )

            # Save best model (use weighted F1 as primary metric)
            if val_metrics['f1_weighted'] > best_val_f1:
                best_val_f1 = val_metrics['f1_weighted']
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"\n  Early stopping at epoch {epoch} (best: {best_epoch})")
                    break

        training_time = time.time() - start_time

        # Load best model and evaluate on test set
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model = model.to(device)

        test_metrics = evaluate(model, data_device, device, data_device.test_mask, criterion)

        print(f"\n{'=' * 70}")
        print(f"  TEST RESULTS (best epoch: {best_epoch})")
        print(f"{'=' * 70}")
        print(f"  Accuracy:    {test_metrics['accuracy']:.4f} ({test_metrics['accuracy'] * 100:.2f}%)")
        print(f"  F1 (binary): {test_metrics['f1']:.4f}")
        print(f"  F1 (weight): {test_metrics['f1_weighted']:.4f}")
        print(f"  Precision:   {test_metrics['precision']:.4f}")
        print(f"  Recall:      {test_metrics['recall']:.4f}")
        print(f"  AUC-ROC:     {test_metrics['auc']:.4f}")
        print(f"  Fake recall: {test_metrics['class0_recall']:.4f}")
        print(f"  Real recall: {test_metrics['class1_recall']:.4f}")

        cm = test_metrics['confusion_matrix']
        print(f"\n  Confusion Matrix:")
        print(f"             Pred 0  Pred 1")
        print(f"  Actual 0   {cm[0][0]:5d}   {cm[0][1]:5d}")
        print(f"  Actual 1   {cm[1][0]:5d}   {cm[1][1]:5d}")

        # Save fold results
        fold_result = {
            'fold': fold_idx,
            'best_epoch': best_epoch,
            'training_time': training_time,
            'num_params': num_params,
            'test_metrics': {k: float(v) if isinstance(v, (float, int, np.floating)) else v
                           for k, v in test_metrics.items()},
            'best_val_f1': float(best_val_f1),
            'history': {k: [float(v) for v in vals] for k, vals in history.items()},
        }
        all_fold_results.append(fold_result)

        # Save model checkpoint
        model_path = os.path.join(args.output_dir, f'best_model_fold{fold_idx}.pt')
        torch.save({
            'model_state_dict': best_model_state,
            'epoch': best_epoch,
            'val_f1': best_val_f1,
            'test_metrics': test_metrics,
            'config': vars(args),
        }, model_path)

    # Summary across folds
    if args.kfold > 0:
        print(f"\n{'=' * 70}")
        print(f"CROSS-VALIDATION SUMMARY ({args.kfold} folds)")
        print(f"{'=' * 70}")
        accs = [r['test_metrics']['accuracy'] for r in all_fold_results]
        f1s = [r['test_metrics']['f1_weighted'] for r in all_fold_results]
        c0rs = [r['test_metrics']['class0_recall'] for r in all_fold_results]
        print(f"  Accuracy:    {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
        print(f"  F1 (weight): {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
        print(f"  Fake recall: {np.mean(c0rs):.4f} +/- {np.std(c0rs):.4f}")

    # Save all results
    results = {
        'config': vars(args),
        'folds': all_fold_results,
        'summary': {
            'mean_accuracy': float(np.mean([r['test_metrics']['accuracy'] for r in all_fold_results])),
            'mean_f1_weighted': float(np.mean([r['test_metrics']['f1_weighted'] for r in all_fold_results])),
            'mean_f1_binary': float(np.mean([r['test_metrics']['f1'] for r in all_fold_results])),
            'mean_auc': float(np.mean([r['test_metrics']['auc'] for r in all_fold_results])),
            'mean_class0_recall': float(np.mean([r['test_metrics']['class0_recall'] for r in all_fold_results])),
        }
    }

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to {args.output_dir}/results.json")
    print(f"  Model saved to {args.output_dir}/best_model_fold0.pt")
    print(f"\n{'=' * 70}")
    print("  DONE!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
