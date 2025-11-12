#!/bin/bash
#
# Quick Start: Run Enhanced GAT Training
# This script demonstrates the complete improved pipeline
#

set -e  # Exit on error

echo "=================================================================="
echo "🚀 ENHANCED GAT TRAINING - QUICK START"
echo "=================================================================="

# Configuration
DATA_PATH="${DATA_PATH:-data/graphs/graph_data.pt}"
OUTPUT_BASE="${OUTPUT_BASE:-experiments}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Quick baseline
echo -e "\n${GREEN}Step 1: Running baseline (random split, default params)${NC}"
python scripts/train_gat_large.py \
  --data-path "$DATA_PATH" \
  --output-dir "$OUTPUT_BASE/baseline" \
  --epochs 100 \
  --patience 20 \
  --hidden-dim 128 \
  --num-heads 8 \
  --dropout 0.3

# Step 2: Enhance graph with edge weights
echo -e "\n${GREEN}Step 2: Enhancing graph with edge weights and metadata${NC}"
python scripts/enhance_graph.py \
  --data-path "$DATA_PATH" \
  --output-path "${DATA_PATH%.pt}_enhanced.pt" \
  --add-edge-weights \
  --add-edge-types \
  --create-metadata \
  --metadata-path "data/graphs/metadata.csv"

# Step 3: Train with temporal split + focal loss
echo -e "\n${GREEN}Step 3: Training with temporal split + focal loss${NC}"
python scripts/train_gat_improved.py \
  --data-path "${DATA_PATH%.pt}_enhanced.pt" \
  --metadata-path "data/graphs/metadata.csv" \
  --split-method temporal \
  --loss focal \
  --focal-alpha 0.25 \
  --focal-gamma 2.0 \
  --output-dir "$OUTPUT_BASE/temporal_focal"

# Step 4: Full improved model
echo -e "\n${GREEN}Step 4: Training full improved model${NC}"
python scripts/train_gat_improved.py \
  --data-path "${DATA_PATH%.pt}_enhanced.pt" \
  --metadata-path "data/graphs/metadata.csv" \
  --split-method temporal \
  --use-edge-attr \
  --loss focal \
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
  --output-dir "$OUTPUT_BASE/improved_full"

# Step 5: Compare results
echo -e "\n${GREEN}Step 5: Comparing all experiments${NC}"
python scripts/compare_results.py \
  "$OUTPUT_BASE/baseline/results.json" \
  "$OUTPUT_BASE/temporal_focal/results.json" \
  "$OUTPUT_BASE/improved_full/results.json" \
  --labels "Baseline" "Temporal+Focal" "Full Improved" \
  --output "$OUTPUT_BASE/comparison.csv"

# Summary
echo -e "\n${GREEN}=================================================================="
echo "✅ QUICK START COMPLETE!"
echo "==================================================================${NC}"
echo ""
echo "📊 Results saved to:"
echo "  • Baseline:        $OUTPUT_BASE/baseline/results.json"
echo "  • Temporal+Focal:  $OUTPUT_BASE/temporal_focal/results.json"
echo "  • Full Improved:   $OUTPUT_BASE/improved_full/results.json"
echo "  • Comparison:      $OUTPUT_BASE/comparison.csv"
echo ""
echo "📈 Next steps:"
echo "  1. Review comparison.csv for improvements"
echo "  2. Run hyperparameter optimization:"
echo "     python scripts/hyperparameter_optimization.py --n-trials 60"
echo "  3. Check IMPROVEMENTS_GUIDE.md for more advanced techniques"
echo ""
echo "🎯 Expected improvements: +3-7 F1 points from baseline"
