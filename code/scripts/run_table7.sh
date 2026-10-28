#!/usr/bin/env bash
# Table 7 -- judge-backbone comparison (RQ5).
#
# The Stage-III judge is the only component that talks to an LLM, so switching
# the backbone is a one-flag change.  This driver evaluates the *same* trained
# checkpoint with several OpenAI-compatible model names and writes one JSON per
# backbone, which is exactly the row layout of Table 7.
#
#   export OPENAI_API_KEY=...            # never passed on the command line
#   export OPENAI_BASE_URL=...           # optional, OpenAI-compatible endpoint
#   CKPT=../results/phyicews14/best.pt bash scripts/run_table7.sh
#
# Model names are whatever your endpoint exposes; the defaults mirror the paper.
set -euo pipefail

cd "$(dirname "$0")/.."

DATA=${DATA:-../datasets}
RULES=${RULES:-rules}
OUT=${OUT:-../results/table7}
DATASET=${DATASET:-PhyICEWS14}
CKPT=${CKPT:-../results/${DATASET,,}/best.pt}
MODELS=${MODELS:-"gpt-5.4-mini gpt-4.1-mini DeepSeek-R1 DeepSeek-V3 Llama3-70B Llama3-8B"}
EXTRA=${EXTRA:-}

: "${OPENAI_API_KEY:?export OPENAI_API_KEY before running Table 7}"

mkdir -p "$OUT"
for M in $MODELS; do
  slug=$(printf '%s' "$M" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/_/g')
  echo "=== Table 7: ${DATASET} with ${M} ==="
  python -m phyworld.train \
    --dataset "$DATASET" --data-dir "$DATA" --rules-dir "$RULES" \
    --setting auto --device auto \
    --llm-backend openai --llm-model "$M" \
    --llm-cache "$OUT/judge_cache.json" \
    --call-ratio "${CALL_RATIO:-1.0}" \
    --eval-only --checkpoint "$CKPT" \
    --output "$OUT/${slug}.json" ${EXTRA} 2>&1 | tee "$OUT/${slug}.log"
done

echo "wrote $OUT/*.json"