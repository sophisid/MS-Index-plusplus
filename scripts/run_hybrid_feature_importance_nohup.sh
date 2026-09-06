#!/usr/bin/env bash
# Collect current 12-feature Hybrid training data for the five main datasets,
# train raw/normalized/joint routing models, and aggregate feature importance.

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$ROOT/artifacts_hybrid_feature_importance}"
TRAINING_ROOT="${TRAINING_ROOT:-$ARTIFACT_ROOT/training_private}"
RESULT_ROOT="${RESULT_ROOT:-$ARTIFACT_ROOT/results}"
JAR_PATH="${JAR_PATH:-$ROOT/target/MS-Index-1.0-jar-with-dependencies.jar}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.10}"

UEA_BASE_DIR="${UEA_BASE_DIR:-/data/ssideri/Multivariate_msindex}"
SYNTHETIC_DIR="${SYNTHETIC_DIR:-$ROOT/data/synthetic}"
WEATHER_DIR="${WEATHER_DIR:-/data/ssideri/weather}"
STOCKS_DIR="${STOCKS_DIR:-/data/ssideri/stocks}"
WIND_DIR="${WIND_DIR:-/data/ssideri/wind_msindex}"

export JAR_PATH PYTHON_BIN ARTIFACT_ROOT TRAINING_ROOT RESULT_ROOT
export UEA_BASE_DIR SYNTHETIC_DIR WEATHER_DIR STOCKS_DIR WIND_DIR
export TRAIN_QUERIES="${TRAIN_QUERIES:-100}"
export SEED_LIST="${SEED_LIST:-1}"
export NOISE_LIST="${NOISE_LIST:-0.1 0.3 0.5 0.7 0.9}"
export RESUME="${RESUME:-false}"
export CLEANUP_TRAINING_DATA="${CLEANUP_TRAINING_DATA:-true}"
export CLEANUP_TRAIN_RUNS="${CLEANUP_TRAIN_RUNS:-true}"
export TRAIN_ONLY=true

LOG_ROOT="$ARTIFACT_ROOT/logs"
STATUS_CSV="$ARTIFACT_ROOT/status.csv"
mkdir -p "$LOG_ROOT"
printf '%s\n' "dataset,status,log_path" > "$STATUS_CSV"

if [[ ! -f "$JAR_PATH" ]]; then
    echo "Missing JAR: $JAR_PATH" >&2
    exit 1
fi

datasets=(synthetic stocks wind weather DuckDuckGeese)
for dataset in "${datasets[@]}"; do
    log_path="$LOG_ROOT/$dataset.log"
    echo "[$(date -Is)] START dataset=$dataset" | tee "$log_path"
    if bash "$SCRIPT_DIR/HYBRID_jobs.sh" --dataset "$dataset" >> "$log_path" 2>&1; then
        status=0
    else
        status=$?
    fi
    printf '%s,%s,%s\n' "$dataset" "$status" "$log_path" >> "$STATUS_CSV"
    echo "[$(date -Is)] END dataset=$dataset status=$status" | tee -a "$log_path"
    if [[ $status -ne 0 ]]; then
        exit "$status"
    fi
done

"$PYTHON_BIN" "$SCRIPT_DIR/aggregate_hybrid_feature_importance.py" \
    --results-root "$RESULT_ROOT" \
    --output "$RESULT_ROOT/feature_importance_summary.csv" \
    >> "$LOG_ROOT/aggregate.log" 2>&1

echo "[$(date -Is)] Hybrid feature-importance training finished. Summary: $RESULT_ROOT/feature_importance_summary.csv"
