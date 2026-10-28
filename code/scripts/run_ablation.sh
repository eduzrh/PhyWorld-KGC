#!/usr/bin/env bash
# Figure 7: the 19 ablation variants (named exactly as in the paper).
# Run from the code/ directory:  bash scripts/run_ablation.sh
set -e

DATA=${DATA:-../datasets}
DS=${DATASET:-PhyICEWS14}
GROUP=${GROUP:-all}
DEVICE=${DEVICE:-auto}
mkdir -p ../results logs

python -m phyworld.ablations --dataset "$DS" --data-dir "$DATA" \
    --group "$GROUP" --epochs "${EPOCHS:-30}" --device "$DEVICE" \
    --output "../results/ablations_${DS}.json" 2>&1 | tee "logs/ablations_${DS}.log"