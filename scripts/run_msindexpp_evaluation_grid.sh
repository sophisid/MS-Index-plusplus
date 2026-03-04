#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

JAR_PATH="${JAR_PATH:-target/MS-Index-1.0-jar-with-dependencies.jar}"
MODEL_PATH="${MODEL_PATH:-output/msindexpp_training_grid/msindexpp_gbdt_all_q256.json}"
DATA_PATH="${DATA_PATH:-data/synthetic}"
N="${N:-100}"
MAX_M="${MAX_M:-100000}"
CHANNELS="${CHANNELS:--1}"
N_QUERY_CHANNELS="${N_QUERY_CHANNELS:--1}"
RUNTIME_MODE="${RUNTIME_MODE:-FULL_NO_STORE}"
PARALLEL="${PARALLEL:-false}"
OUTPUT_DIR="${OUTPUT_DIR:-output/msindexpp_evaluation_grid}"

Q_LEN=256
N_QUERIES=50
K_VALUES=(1 5 10)
SEEDS=(13 17 21)
QUERY_NOISE_VALUES=(0 0.3 0.5 0.9)
NORMALIZE_VALUES=(true false)
QUERY_FROM_INDEX_VALUES=(true false)

if [[ ! -f "$JAR_PATH" ]]; then
  echo "Missing jar at $JAR_PATH"
  echo "Build it first with: mvn -q -DskipTests package"
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Missing model at $MODEL_PATH"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

experiment_id=100000

for k in "${K_VALUES[@]}"; do
  for normalize in "${NORMALIZE_VALUES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for query_noise_eps in "${QUERY_NOISE_VALUES[@]}"; do
        for query_from_index in "${QUERY_FROM_INDEX_VALUES[@]}"; do
          run_name="k${k}_norm_${normalize}_seed${seed}_noise${query_noise_eps}_fromidx_${query_from_index}"
          eval_csv="$OUTPUT_DIR/${run_name}.csv"
          log_path="$OUTPUT_DIR/${run_name}.log"

          echo "Evaluating $run_name"

          java \
            -Dmsindexpp.modelPath="$MODEL_PATH" \
            -Dmsindexpp.evaluationCsv="$eval_csv" \
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
            "$query_noise_eps" \
            > "$log_path" 2>&1

          experiment_id=$((experiment_id + 1))
        done
      done
    done
  done
done

echo "Evaluation grid completed. Results are in $OUTPUT_DIR"
