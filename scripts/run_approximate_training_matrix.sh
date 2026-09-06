#!/usr/bin/env bash
# Train raw/normalized/joint models for one approximate model family on the
# selected five datasets.  Set MODEL_TYPE to run TSA or TIA:
#
#   MODEL_TYPE=target_specific_classifier       # TSA
#   MODEL_TYPE=target_independent_regressor     # TIA
#
# It is training-only by default.  Set TRAIN_ONLY=false when the same raw,
# normalized, and joint models should also produce the full query-time matrix.

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_TYPE="${MODEL_TYPE:-target_specific_classifier}"
case "$MODEL_TYPE" in
    target_specific_classifier) DEFAULT_ROOT="$ROOT/artifacts_approx_target_specific" ;;
    target_independent_regressor) DEFAULT_ROOT="$ROOT/artifacts_approx_regressor" ;;
    *)
        echo "MODEL_TYPE must be target_specific_classifier or target_independent_regressor" >&2
        exit 1
        ;;
esac

ARTIFACT_ROOT="${ARTIFACT_ROOT:-$DEFAULT_ROOT}"
N_QUERIES="${N_QUERIES:-100}"
K_LIST="${K_LIST:-1 10 100}"
NOISE_LIST="${NOISE_LIST:-0.1 0.3 0.5 0.7 0.9}"
SEED="${SEED:-1}"
TARGET_RECALL="${TARGET_RECALL:-0.9}"
ACCURACY_TARGET="${ACCURACY_TARGET:-0.9}"
MAX_WORKERS="${MAX_WORKERS:-1}"
RESUME="${RESUME:-false}"
TRAIN_ONLY="${TRAIN_ONLY:-true}"

DATASETS=(synthetic stocks wind weather DuckDuckGeese)
LOG_ROOT="$ARTIFACT_ROOT/logs"
mkdir -p "$LOG_ROOT"
STATUS_CSV="$ARTIFACT_ROOT/status.csv"
printf '%s\n' "model_type,dataset,status,log_path" > "$STATUS_CSV"

for dataset in "${DATASETS[@]}"; do
    log_path="$LOG_ROOT/$dataset.log"
    echo "[$(date -Is)] START model_type=$MODEL_TYPE dataset=$dataset" | tee "$log_path"
    if MODEL_TYPE="$MODEL_TYPE" \
        ARTIFACT_ROOT="$ARTIFACT_ROOT" \
        N_QUERIES="$N_QUERIES" \
        K_LIST="$K_LIST" \
        NOISE_LIST="$NOISE_LIST" \
        SEED="$SEED" \
        TARGET_RECALL="$TARGET_RECALL" \
        ACCURACY_TARGET="$ACCURACY_TARGET" \
        MAX_WORKERS="$MAX_WORKERS" \
        RESUME="$RESUME" \
        TRAIN_ONLY="$TRAIN_ONLY" \
        CLEANUP_TRAINING_DATA=true \
        CLEANUP_TRAIN_RUNS=true \
        EXPORT_LOWER_BOUNDS=false \
        bash "$SCRIPT_DIR/EARLY_TERMINATION_jobs.sh" --dataset "$dataset" \
        >> "$log_path" 2>&1; then
        status=0
    else
        status=$?
    fi
    printf '%s,%s,%s,%s\n' "$MODEL_TYPE" "$dataset" "$status" "$log_path" >> "$STATUS_CSV"
    echo "[$(date -Is)] END model_type=$MODEL_TYPE dataset=$dataset status=$status" | tee -a "$log_path"
    if [[ $status -ne 0 ]]; then
        exit "$status"
    fi
done

echo "[$(date -Is)] Approximate training matrix finished. Status: $STATUS_CSV"
