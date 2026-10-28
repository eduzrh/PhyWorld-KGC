#!/usr/bin/env bash
# Table 3 (interpolation) + Table 5 (extrapolation): six benchmarks, tail+head.
# Run from the code/ directory:  bash scripts/run_main.sh
set -e

DATA=${DATA:-../datasets}
OUT=${OUT:-../results/main}
DEVICE=${DEVICE:-auto}
mkdir -p "$OUT" logs

for DS in PhyICEWS14 PhyICEWS14-Ext PhyICEWS18-Ext PhyICEWS-0515 PhyYAGO PhyGDELT; do
  echo "=== $DS ==="
  python -m phyworld.train --dataset "$DS" --data-dir "$DATA" --setting auto \
      --epochs "${EPOCHS:-30}" --batch-size 32 --device "$DEVICE" \
      2>&1 | tee "logs/${DS}.log"
  # --eval-only reuses results/<run_name>/best.pt; run_name defaults to the
  # lower-cased dataset name, so the two invocations line up.
  python -m phyworld.train --dataset "$DS" --data-dir "$DATA" --setting auto \
      --device "$DEVICE" --eval-only --output "$OUT/${DS}.json" \
      2>&1 | tee -a "logs/${DS}.log"
done