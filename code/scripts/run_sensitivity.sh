#!/usr/bin/env bash
# Figure 9: |S_q|-1, delta, c_min and lambda sweeps on one benchmark.
# Run from the code/ directory:  bash scripts/run_sensitivity.sh
set -e

DATA=${DATA:-../datasets}
DS=${DATASET:-PhyICEWS14}
OUT=${OUT:-../results/sensitivity}
DEVICE=${DEVICE:-auto}
mkdir -p "$OUT" logs

# |S_q| - 1 in {1, 2, 3, 4, 6}
for R in 1 2 3 4 6; do
  python -m phyworld.train --dataset "$DS" --data-dir "$DATA" --setting auto \
     --spatial-rings "$R" --device "$DEVICE" \
     --run-name "${DS,,}_rings${R}" 2>&1 | tee "logs/${DS}_rings${R}.log"
  python -m phyworld.train --dataset "$DS" --data-dir "$DATA" --setting auto \
     --spatial-rings "$R" --device "$DEVICE" --run-name "${DS,,}_rings${R}" \
     --eval-only --output "$OUT/rings_${R}.json" 2>&1 | tee -a "logs/${DS}_rings${R}.log"
done

# entropy threshold delta in {0.4, 0.6, 0.8, 1.0, 1.2}
for T in 0.4 0.6 0.8 1.0 1.2; do
  python -m phyworld.train --dataset "$DS" --data-dir "$DATA" --setting auto \
     --entropy-threshold "$T" --device "$DEVICE" \
     --run-name "${DS,,}_ent${T}" 2>&1 | tee "logs/${DS}_ent${T}.log"
  python -m phyworld.train --dataset "$DS" --data-dir "$DATA" --setting auto \
     --entropy-threshold "$T" --device "$DEVICE" --run-name "${DS,,}_ent${T}" \
     --eval-only --output "$OUT/entropy_${T}.json" 2>&1 | tee -a "logs/${DS}_ent${T}.log"
done

# pca threshold c_min in {0.05, 0.1, 0.2, 0.4}; each variant re-mines Omega and
# trains against that rule base through --rules-dir.
for C in 0.05 0.1 0.2 0.4; do
  python scripts/mine_rules.py --dataset "$DS" --data-dir "$DATA" \
     --pca-threshold "$C" --out-dir "rules/pca_${C}"
  python -m phyworld.train --dataset "$DS" --data-dir "$DATA" --setting auto \
     --pca-threshold "$C" --rules-dir "rules/pca_${C}" --device "$DEVICE" \
     --run-name "${DS,,}_pca${C}" 2>&1 | tee "logs/${DS}_pca${C}.log"
  python -m phyworld.train --dataset "$DS" --data-dir "$DATA" --setting auto \
     --pca-threshold "$C" --rules-dir "rules/pca_${C}" --device "$DEVICE" \
     --run-name "${DS,,}_pca${C}" --eval-only \
     --output "$OUT/pca_${C}.json" 2>&1 | tee -a "logs/${DS}_pca${C}.log"
done

# fusion weight lambda in {0.5, 0.6, 0.7, 0.8, 0.9}
for L in 0.5 0.6 0.7 0.8 0.9; do
  python -m phyworld.train --dataset "$DS" --data-dir "$DATA" --setting auto \
     --fusion-weight "$L" --device "$DEVICE" \
     --run-name "${DS,,}_lam${L}" 2>&1 | tee "logs/${DS}_lam${L}.log"
  python -m phyworld.train --dataset "$DS" --data-dir "$DATA" --setting auto \
     --fusion-weight "$L" --device "$DEVICE" --run-name "${DS,,}_lam${L}" \
     --eval-only --output "$OUT/lambda_${L}.json" 2>&1 | tee -a "logs/${DS}_lam${L}.log"
done