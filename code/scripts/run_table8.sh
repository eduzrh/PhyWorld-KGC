#!/usr/bin/env bash
# Table 8 -- effect of reducing Stage-III agent calls.
#
# Stage III is triggered by candidate-set entropy; `--call-ratio` keeps only the
# highest-entropy share of those triggers.  Each run records
# `stage3.meta_calls`, `stage3.judge_usage` and `stage3.tokens_per_query`, i.e.
# the H@1 / s-per-query / tokens columns of Table 8.
#
#   CKPT=../results/phyicews14/best.pt bash scripts/run_table8.sh
#   RATIOS="1.0 0.5 0.25" DATASET=PhyICEWS-0515 bash scripts/run_table8.sh
set -euo pipefail

cd "$(dirname "$0")/.."

DATA=${DATA:-../datasets}
RULES=${RULES:-rules}
OUT=${OUT:-../results/table8}
DATASET=${DATASET:-PhyICEWS14}
CKPT=${CKPT:-../results/${DATASET,,}/best.pt}
RATIOS=${RATIOS:-"1.0 0.75 0.5 0.25"}
BACKEND=${BACKEND:-offline}
MODEL=${MODEL:-gpt-5.4-mini}

mkdir -p "$OUT"
for R in $RATIOS; do
  echo "=== Table 8: ${DATASET} call-ratio=${R} ==="
  python -m phyworld.train \
    --dataset "$DATASET" --data-dir "$DATA" --rules-dir "$RULES" \
    --setting auto --device auto \
    --llm-backend "$BACKEND" --llm-model "$MODEL" \
    --call-ratio "$R" \
    --eval-only --checkpoint "$CKPT" \
    --output "$OUT/${DATASET,,}_r${R}.json" 2>&1 | tee "$OUT/${DATASET,,}_r${R}.log"
done

echo "wrote $OUT/*.json"