#!/usr/bin/env bash
# Train one Target-Independent Approximate (TIA) model per dataset and
# evaluate it for several requested-recall targets.

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load the shared dataset paths while preserving explicit environment overrides.
UEA_BASE_DIR_OVERRIDE="${UEA_BASE_DIR:-}"
SYNTHETIC_DIR_OVERRIDE="${SYNTHETIC_DIR:-}"
WEATHER_DIR_OVERRIDE="${WEATHER_DIR:-}"
STOCKS_DIR_OVERRIDE="${STOCKS_DIR:-}"
WIND_DIR_OVERRIDE="${WIND_DIR:-}"
TARGET_RECALLS_OVERRIDE="${TARGET_RECALLS:-}"
DATASET_LIST_OVERRIDE="${DATASET_LIST:-}"
NOISE_LIST_OVERRIDE="${NOISE_LIST:-}"
N_QUERIES_OVERRIDE="${N_QUERIES:-}"
MAX_WORKERS_OVERRIDE="${MAX_WORKERS:-}"
source "$SCRIPT_DIR/config.sh"
[[ -n "$UEA_BASE_DIR_OVERRIDE" ]] && UEA_BASE_DIR="$UEA_BASE_DIR_OVERRIDE"
[[ -n "$SYNTHETIC_DIR_OVERRIDE" ]] && SYNTHETIC_DIR="$SYNTHETIC_DIR_OVERRIDE"
[[ -n "$WEATHER_DIR_OVERRIDE" ]] && WEATHER_DIR="$WEATHER_DIR_OVERRIDE"
[[ -n "$STOCKS_DIR_OVERRIDE" ]] && STOCKS_DIR="$STOCKS_DIR_OVERRIDE"
[[ -n "$WIND_DIR_OVERRIDE" ]] && WIND_DIR="$WIND_DIR_OVERRIDE"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/artifacts_target_independent}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.10}"
JAR_PATH="${JAR_PATH:-$ROOT/target/MS-Index-1.0-jar-with-dependencies.jar}"

UEA_BASE_DIR="${UEA_BASE_DIR:-/data/ssideri/Multivariate_msindex}"
SYNTHETIC_DIR="${SYNTHETIC_DIR:-$ROOT/data/synthetic}"
WEATHER_DIR="${WEATHER_DIR:-/data/ssideri/weather}"
STOCKS_DIR="${STOCKS_DIR:-/data/ssideri/stocks}"
WIND_DIR="${WIND_DIR:-/data/ssideri/wind_msindex}"

TARGET_RECALLS="${TARGET_RECALLS_OVERRIDE:-0.80 0.85 0.90 0.95 0.99}"
DATASET_LIST="${DATASET_LIST_OVERRIDE:-synthetic stocks wind weather DuckDuckGeese}"
NOISE_LIST="${NOISE_LIST_OVERRIDE:-0.1 0.3 0.5 0.7 0.9}"
N_QUERIES="${N_QUERIES_OVERRIDE:-100}"
MAX_WORKERS="${MAX_WORKERS_OVERRIDE:-1}"

read -r -a TARGETS <<< "$TARGET_RECALLS"
read -r -a DATASETS <<< "$DATASET_LIST"
read -r -a NOISES <<< "$NOISE_LIST"

STATUS_CSV="$OUTPUT_ROOT/status.csv"
LOG_ROOT="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_ROOT"
if [[ ! -f "$STATUS_CSV" ]]; then
    printf '%s\n' "phase,target_recall,dataset,start_time,end_time,status,log_path" > "$STATUS_CSV"
fi

dataset_spec() {
    case "$1" in
        synthetic) printf '%s\t%s\t%s\t%s\t%s\n' synthetic "$SYNTHETIC_DIR" 1024 100 4096 ;;
        stocks) printf '%s\t%s\t%s\t%s\t%s\n' stocks "$STOCKS_DIR" 730 100 16384 ;;
        wind) printf '%s\t%s\t%s\t%s\t%s\n' wind "$WIND_DIR" 1800 1 32768 ;;
        weather) printf '%s\t%s\t%s\t%s\t%s\n' weather "$WEATHER_DIR" 1488 100 16384 ;;
        DuckDuckGeese) printf '%s\t%s\t%s\t%s\t%s\n' DuckDuckGeese "$UEA_BASE_DIR/DuckDuckGeese" 54 50 10000 ;;
        *) return 1 ;;
    esac
}

run_suite() {
    JAR_PATH="$JAR_PATH" "$PYTHON_BIN" "$SCRIPT_DIR/run_msindexpp_dataset_suite.py" "$@"
}

for dataset in "${DATASETS[@]}"; do
    if ! spec_line="$(dataset_spec "$dataset")"; then
        echo "Unknown dataset: $dataset" >&2
        exit 1
    fi
    IFS=$'\t' read -r name data_path q_len n max_m <<< "$spec_line"
    if [[ ! -d "$data_path" ]]; then
        echo "Missing dataset path for $dataset: $data_path" >&2
        exit 1
    fi

    training_root="$OUTPUT_ROOT/training"
    model_path="$training_root/$dataset/gbdt-model.json"
    training_log="$LOG_ROOT/training-$dataset.log"
    started="$(date -Is)"
    echo "[$started] START phase=training dataset=$dataset" | tee -a "$training_log"
    run_suite \
        --artifact-root "$training_root" \
        --dataset-spec "$name:$data_path:$q_len:$n:$max_m" \
        --max-m "$max_m" \
        --n-queries "$N_QUERIES" \
        --max-workers "$MAX_WORKERS" \
        --seed 1 \
        --checkpoint-interval 64 \
        --accuracy-target 0.90 \
        --target-recall 0.90 \
        --model-score-threshold 0.5 \
        --model-type target_independent_regressor \
        --normalize-values false true \
        --k-values 1 10 100 \
        --noise-values "${NOISES[@]}" \
        --from-index-values true \
        --benchmark-algorithms MSINDEX MASS MSINDEXPP \
        --skip-eval \
        --skip-benchmark \
        >> "$training_log" 2>&1
    status=$?
    ended="$(date -Is)"
    printf '%s\n' "training,,$dataset,$started,$ended,$status,$training_log" >> "$STATUS_CSV"
    if [[ $status -ne 0 || ! -f "$model_path" ]]; then
        echo "[$ended] END phase=training dataset=$dataset status=$status model=$model_path" | tee -a "$training_log"
        exit "$status"
    fi
    echo "[$ended] END phase=training dataset=$dataset status=0 model=$model_path" | tee -a "$training_log"

    for target in "${TARGETS[@]}"; do
        target_tag="${target/./p}"
        started="$(date -Is)"
        artifact_root="$OUTPUT_ROOT/target_${target_tag}"
        log_path="$LOG_ROOT/target-${target_tag}-${dataset}.log"
        echo "[$started] START phase=evaluation target=$target dataset=$dataset" | tee -a "$log_path"
        run_suite \
            --artifact-root "$artifact_root" \
            --model-path "$model_path" \
            --dataset-spec "$name:$data_path:$q_len:$n:$max_m" \
            --max-m "$max_m" \
            --n-queries "$N_QUERIES" \
            --max-workers "$MAX_WORKERS" \
            --seed 1 \
            --checkpoint-interval 64 \
            --accuracy-target 0.90 \
            --target-recall "$target" \
            --model-score-threshold 0.5 \
            --model-type target_independent_regressor \
            --normalize-values false true \
            --k-values 1 10 100 \
            --noise-values "${NOISES[@]}" \
            --from-index-values true \
            --export-evaluation-trace \
            --benchmark-algorithms MSINDEX MASS MSINDEXPP \
            --skip-train \
            >> "$log_path" 2>&1
        status=$?
        ended="$(date -Is)"
        printf '%s\n' "evaluation,$target,$dataset,$started,$ended,$status,$log_path" >> "$STATUS_CSV"
        echo "[$ended] END phase=evaluation target=$target dataset=$dataset status=$status" | tee -a "$log_path"
        if [[ $status -ne 0 ]]; then
            exit "$status"
        fi
    done
done

echo "[$(date -Is)] Target-independent recall sweep finished. Status: $STATUS_CSV"
