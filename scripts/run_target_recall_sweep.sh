#!/usr/bin/env bash
# Train and evaluate Target-Specific Approximate (TSA) AMS-Index independently
# for several requested-recall targets.

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
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/artifacts_target_specific}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.10}"
JAR_PATH="${JAR_PATH:-$ROOT/target/MS-Index-1.0-jar-with-dependencies.jar}"

UEA_BASE_DIR="${UEA_BASE_DIR:-/data/ssideri/Multivariate_msindex}"
SYNTHETIC_DIR="${SYNTHETIC_DIR:-$ROOT/data/synthetic}"
WEATHER_DIR="${WEATHER_DIR:-/data/ssideri/weather}"
STOCKS_DIR="${STOCKS_DIR:-/data/ssideri/stocks}"
WIND_DIR="${WIND_DIR:-/data/ssideri/wind_msindex}"

TARGET_RECALLS="${TARGET_RECALLS_OVERRIDE:-0.90}"
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
    printf '%s\n' "target_recall,dataset,start_time,end_time,status,log_path" > "$STATUS_CSV"
fi

if [[ ! -f "$JAR_PATH" ]]; then
    echo "Missing JAR: $JAR_PATH" >&2
    exit 1
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

for target in "${TARGETS[@]}"; do
    target_tag="${target/./p}"
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

        started="$(date -Is)"
        log_path="$LOG_ROOT/target-${target_tag}-${dataset}.log"
        artifact_root="$OUTPUT_ROOT/target_${target_tag}/$dataset"
        echo "[$started] START target=$target dataset=$dataset" | tee -a "$log_path"

        JAR_PATH="$JAR_PATH" "$PYTHON_BIN" "$SCRIPT_DIR/run_msindexpp_dataset_suite.py" \
            --artifact-root "$artifact_root" \
            --dataset-spec "$name:$data_path:$q_len:$n:$max_m" \
            --max-m "$max_m" \
            --n-queries "$N_QUERIES" \
            --max-workers "$MAX_WORKERS" \
            --seed 1 \
            --checkpoint-interval 64 \
            --accuracy-target "$target" \
            --target-recall "$target" \
            --model-score-threshold 0.5 \
            --model-type target_specific_classifier \
            --normalize-values false true \
            --k-values 1 10 100 \
            --noise-values "${NOISES[@]}" \
            --from-index-values true \
            --benchmark-algorithms MSINDEX MASS MSINDEXPP \
            >> "$log_path" 2>&1
        status=$?
        ended="$(date -Is)"
        printf '%s,%s,%s,%s,%s,%s\n' \
            "$target" "$dataset" "$started" "$ended" "$status" "$log_path" >> "$STATUS_CSV"
        echo "[$ended] END target=$target dataset=$dataset status=$status" | tee -a "$log_path"
        if [[ $status -ne 0 ]]; then
            echo "Target-recall sweep stopped after failure; see $log_path" >&2
            exit "$status"
        fi
    done
done

echo "[$(date -Is)] Target-recall sweep finished. Status: $STATUS_CSV"
