#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Repository root
# ============================================================

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ============================================================
# Data
# ============================================================

TRAIN="$ROOT/data/train.csv"
VAL="$ROOT/data/val.csv"
TEST="$ROOT/data/test.csv"

# ============================================================
# Previous-stage outputs
# ============================================================

TEACHER_DIR="$ROOT/outputs/teacher"
PFI_DIR="$ROOT/outputs/pfi"
CAND_DIR="$ROOT/outputs/candidates"

CANDIDATES="$CAND_DIR/candidate_indices.npy"
REP_C0="$CAND_DIR/weighted_representation_class0.npy"
REP_C1="$CAND_DIR/weighted_representation_class1.npy"
BOUNDARY_REP="$CAND_DIR/teacher_input_representation.npy"

PROBS="$TEACHER_DIR/train_probs.npy"
PFI="$PFI_DIR/feature_importance.csv"

# ============================================================
# Python scripts
# ============================================================

GEN="$ROOT/src/bamd_v3.py"
EVAL="$ROOT/src/evaluate_subset.py"

# ============================================================
# Output
# ============================================================

OUT="$ROOT/outputs/bamd_v3"
SUBSET="$OUT/model_subset_bamd_v3.csv"

mkdir -p "$OUT"

# ============================================================
# Configuration summary
# ============================================================

echo
echo "============================================================"
echo "BAMD-v3"
echo "============================================================"
date
echo

echo "ROOT             : $ROOT"
echo "TRAIN            : $TRAIN"
echo "VAL              : $VAL"
echo "TEST             : $TEST"
echo "TEACHER DIR      : $TEACHER_DIR"
echo "BAMD SCRIPT      : $GEN"
echo "EVAL SCRIPT      : $EVAL"
echo "REP C0           : $REP_C0"
echo "REP C1           : $REP_C1"
echo "BOUNDARY REP     : $BOUNDARY_REP"
echo "PROBS            : $PROBS"
echo "PFI              : $PFI"
echo "CANDIDATES       : $CANDIDATES"
echo "OUT              : $OUT"
echo

# ============================================================
# Check required files
# ============================================================

for f in \
    "$GEN" \
    "$EVAL" \
    "$TRAIN" \
    "$VAL" \
    "$TEST" \
    "$CANDIDATES" \
    "$REP_C0" \
    "$REP_C1" \
    "$BOUNDARY_REP" \
    "$PROBS" \
    "$PFI"
do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing required file:"
        echo "  $f"
        exit 1
    fi
done

# ============================================================
# 1. Generate BAMD-v3 subset
# ============================================================

echo
echo "[1/3] Generating BAMD-v3 subset..."
echo

python -u "$GEN" \
    --train "$TRAIN" \
    --candidate-indices "$CANDIDATES" \
    --representation-c0 "$REP_C0" \
    --representation-c1 "$REP_C1" \
    --boundary-representation "$BOUNDARY_REP" \
    --teacher-probs "$PROBS" \
    --pfi "$PFI" \
    --output "$OUT" \
    --budget-c0 389 \
    --budget-c1 244 \
    --alpha 0.8 \
    --k-same 10 \
    --k-opp 10 \
    --seed 42 \
    --n-init 10

# ============================================================
# 2. Validate final subset
# ============================================================

echo
echo "[2/3] Checking final subset..."
echo

if [[ ! -f "$SUBSET" ]]; then
    echo "ERROR: missing final subset:"
    echo "  $SUBSET"
    exit 1
fi

python -u - <<PY
import pandas as pd

path = r"$SUBSET"

df = pd.read_csv(path)

counts = (
    df["label"]
    .value_counts()
    .sort_index()
    .to_dict()
)

print("Shape:", df.shape, flush=True)
print("Class counts:", counts, flush=True)

assert len(df) == 633, (
    f"Expected 633 samples, got {len(df)}"
)

assert counts == {0: 389, 1: 244}, (
    f"Expected class counts {{0: 389, 1: 244}}, got {counts}"
)

print("Subset validation passed.", flush=True)
PY

# ============================================================
# 3. Run downstream MLP evaluation
# ============================================================

echo
echo "[3/3] Running downstream MLP..."
echo

python -u "$EVAL" \
    --train "$TRAIN" \
    --val "$VAL" \
    --test "$TEST" \
    --teacher-dir "$TEACHER_DIR" \
    --output "$OUT/evaluation" \
    --subsets \
        bamd_v3="$SUBSET" \
    --seeds 0 1 2 3 4 \
    --epochs 100 \
    --batch-size 256 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --patience 20

# ============================================================
# Final result
# ============================================================

echo
echo "============================================================"
echo "BAMD-v3 COMPLETE"
echo "============================================================"
date
echo

COMPARISON="$OUT/evaluation/comparison.csv"

if [[ -f "$COMPARISON" ]]; then
    echo "Comparison:"
    echo
    cat "$COMPARISON"
else
    echo "WARNING: comparison.csv not found:"
    echo "  $COMPARISON"
fi