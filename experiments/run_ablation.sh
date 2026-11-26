#!/usr/bin/env bash
# Multi-seed, multi-config ablation harness for LiteDepth-GD.
# Usage:  bash experiments/run_ablation.sh DATA_ROOT VAL_ROOT [OUT_ROOT] [EXTRA ARGS...]
# Produces OUT_ROOT/<config>/s<seed>/metrics.json per run, then summarise with:
#   python experiments/collect_results.py OUT_ROOT
set -euo pipefail

DATA_ROOT="${1:?data root}"
VAL_ROOT="${2:?val root}"
OUT_ROOT="${3:-runs/ablation}"
shift
shift
# any remaining args (e.g. --size 352 1216 --dataset kitti --depth-scale 256
# --max-depth 80) are forwarded verbatim to train.py

SEEDS="42 43 44"
CONFIGS=(
  "full:"
  "no_gcsa:--disable-gated-skips"
  "no_rgr:--disable-recurrence"
  "no_dghead:--disable-detail-head"
  "no_all:--disable-gated-skips --disable-recurrence --disable-detail-head"
)

for entry in "${CONFIGS[@]}"; do
  name="${entry%%:*}"
  flags="${entry#*:}"
  for seed in $SEEDS; do
    out="$OUT_ROOT/$name/s$seed"
    if [ -f "$out/metrics.json" ]; then
      echo "skip $out (exists)"
      continue
    fi
    echo "=== $name seed=$seed ==="
    python train.py --data-root "$DATA_ROOT" --val-root "$VAL_ROOT" \
      --seed "$seed" --out-dir "$out" $flags ${EXTRA_TRAIN_ARGS:+"$EXTRA_TRAIN_ARGS"} "$@"
  done
done
python experiments/collect_results.py "$OUT_ROOT"
