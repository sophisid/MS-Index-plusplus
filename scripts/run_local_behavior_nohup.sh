#!/usr/bin/env bash
# Reduced local HMS/AMS experiment matrix for quickly checking query-time trends.

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/artifacts_local_behavior}"
LOG_ROOT="$OUTPUT_ROOT/logs"
STATUS_CSV="$OUTPUT_ROOT/status.csv"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.10}"
JAR_PATH="${JAR_PATH:-$ROOT/target/MS-Index-1.0-jar-with-dependencies.jar}"

UEA_BASE_DIR="${UEA_BASE_DIR:-/data/ssideri/Multivariate_msindex}"
SYNTHETIC_DIR="${SYNTHETIC_DIR:-$ROOT/data/synthetic}"
WEATHER_DIR="${WEATHER_DIR:-/data/ssideri/weather}"
STOCKS_DIR="${STOCKS_DIR:-/data/ssideri/stocks}"
WIND_DIR="${WIND_DIR:-/data/ssideri/wind_msindex}"

N_QUERIES="${N_QUERIES:-10}"
NOISE_LIST="${NOISE_LIST:-0.1 0.3 0.5 0.7 0.9}"
DATASETS=(wind stocks synthetic weather DuckDuckGeese)

mkdir -p "$LOG_ROOT"
if [[ ! -f "$STATUS_CSV" ]]; then
    printf '%s\n' "pipeline,dataset,start_time,end_time,status,log_path" > "$STATUS_CSV"
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

record_status() {
    local pipeline="$1" dataset="$2" started="$3" status="$4" log_path="$5"
    printf '%s,%s,%s,%s,%s,%s\n' \
        "$pipeline" "$dataset" "$started" "$(date -Is)" "$status" "$log_path" >> "$STATUS_CSV"
}

run_ams() {
    local dataset="$1" name data_path q_len n max_m started log_path status
    IFS=$'\t' read -r name data_path q_len n max_m < <(dataset_spec "$dataset")
    started="$(date -Is)"
    log_path="$LOG_ROOT/ams-${dataset}.log"
    echo "[$started] START AMS dataset=$dataset" | tee -a "$log_path"

    "$PYTHON_BIN" "$SCRIPT_DIR/run_msindexpp_dataset_suite.py" \
        --artifact-root "$OUTPUT_ROOT/ams/$dataset" \
        --dataset-spec "$name:$data_path:$q_len:$n:$max_m" \
        --max-m "$max_m" \
        --n-queries "$N_QUERIES" \
        --max-workers 1 \
        --seed 1 \
        --checkpoint-interval 64 \
        --accuracy-target 0.9 \
        --model-score-threshold 0.5 \
        --normalize-values false true \
        --k-values 1 10 100 \
        --noise-values $NOISE_LIST \
        --from-index-values true \
        --benchmark-algorithms MSINDEX MASS MSINDEXPP \
        >> "$log_path" 2>&1
    status=$?
    record_status ams "$dataset" "$started" "$status" "$log_path"
    echo "[$(date -Is)] END AMS dataset=$dataset status=$status" | tee -a "$log_path"
}

run_hms() {
    local dataset="$1" started log_path status
    started="$(date -Is)"
    log_path="$LOG_ROOT/hms-${dataset}.log"
    echo "[$started] START HMS dataset=$dataset" | tee -a "$log_path"

    env \
        UEA_BASE_DIR="$UEA_BASE_DIR" \
        SYNTHETIC_DIR="$SYNTHETIC_DIR" \
        WEATHER_DIR="$WEATHER_DIR" \
        STOCKS_DIR="$STOCKS_DIR" \
        WIND_DIR="$WIND_DIR" \
        JAR_PATH="$JAR_PATH" \
        PYTHON_BIN="$PYTHON_BIN" \
        ARTIFACT_ROOT="$OUTPUT_ROOT/hms" \
        TRAINING_ROOT="$OUTPUT_ROOT/hms/training_private" \
        RESULT_ROOT="$OUTPUT_ROOT/hms/results" \
        TRAIN_QUERIES="$N_QUERIES" \
        EVAL_QUERIES="$N_QUERIES" \
        SEED_LIST=1 \
        NOISE_LIST="$NOISE_LIST" \
        RESUME=true \
        CLEANUP_TRAINING_DATA=false \
        CLEANUP_TRAIN_RUNS=false \
        bash "$SCRIPT_DIR/HYBRID_jobs.sh" --dataset "$dataset" \
        >> "$log_path" 2>&1
    status=$?
    record_status hms "$dataset" "$started" "$status" "$log_path"
    echo "[$(date -Is)] END HMS dataset=$dataset status=$status" | tee -a "$log_path"
}

for dataset in "${DATASETS[@]}"; do
    run_ams "$dataset"
    run_hms "$dataset"
done

echo "[$(date -Is)] Local behavior matrix finished. Status: $STATUS_CSV"
