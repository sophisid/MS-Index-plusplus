#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

JAR_PATH="${JAR_PATH:-target/MS-Index-1.0-jar-with-dependencies.jar}"
DATA_PATH="${DATA_PATH:-data/synthetic}"
N="${N:-100}"
MAX_M="${MAX_M:-100000}"
CHANNELS="${CHANNELS:--1}"
N_QUERY_CHANNELS="${N_QUERY_CHANNELS:--1}"
RUNTIME_MODE="${RUNTIME_MODE:-FULL_NO_STORE}"
PARALLEL="${PARALLEL:-false}"
OUTPUT_DIR="${OUTPUT_DIR:-output/msindexpp_training_grid}"
OUTPUT_CSV="${OUTPUT_CSV:-$OUTPUT_DIR/msindexpp_training_all_q256.csv}"

Q_LEN=256
N_QUERIES=50
K_VALUES=(1 5 10)
SEEDS=(1 5 9)
QUERY_NOISE_VALUES=(0 0.3 0.5 0.9)
NORMALIZE_VALUES=(true false)
QUERY_FROM_INDEX_VALUES=(true false)

if [[ ! -f "$JAR_PATH" ]]; then
  echo "Missing jar at $JAR_PATH"
  echo "Build it first with: mvn -q -DskipTests package"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

experiment_id=1

for k in "${K_VALUES[@]}"; do
  :
done

if [[ -f "$OUTPUT_CSV" ]]; then
  rm "$OUTPUT_CSV"
fi

echo "Writing training rows to $OUTPUT_CSV"

for k in "${K_VALUES[@]}"; do
  for normalize in "${NORMALIZE_VALUES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for query_noise_eps in "${QUERY_NOISE_VALUES[@]}"; do
        for query_from_index in "${QUERY_FROM_INDEX_VALUES[@]}"; do
          echo "Running K=$k qLen=$Q_LEN nQueries=$N_QUERIES seed=$seed noise=$query_noise_eps normalize=$normalize queryFromIndexed=$query_from_index"

          java \
            -Dmsindexpp.trainingCsv="$OUTPUT_CSV" \
            -cp "$JAR_PATH" \
            io/github/Main \
            MSINDEXPP \
            "$DATA_PATH" \
            "$N" \
            "$MAX_M" \
            "$CHANNELS" \
            "$N_QUERY_CHANNELS" \
            "$Q_LEN" \
            "$k" \
            "$normalize" \
            "$N_QUERIES" \
            "$RUNTIME_MODE" \
            "$experiment_id" \
            "$seed" \
            "$PARALLEL" \
            "$query_from_index" \
            "$query_noise_eps"

          experiment_id=$((experiment_id + 1))
        done
      done
    done
  done
done

echo "Training grid completed. CSV is at $OUTPUT_CSV"
