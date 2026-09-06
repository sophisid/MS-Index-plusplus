#!/usr/bin/env bash
# Submit the complete selected-dataset experiment plan to Slurm.
#
# The plan is deliberately serialized by phase:
#   1. Hybrid: one job for each of the five selected datasets
#   2. TSA raw/normalized/joint training and timing matrix
#   3. TIA raw/normalized/joint training and timing matrix
#   4. TSA target-recall sweep at Rt=0.90
#   5. TIA target-recall sweep at Rt in {0.80,0.85,0.90,0.95,0.99}
#
# Hybrid jobs run in parallel within phase 1. The remaining phases are one
# Slurm job each and use afterok dependencies, avoiding a machine-wide burst.
# Every worker reuses this script with --phase, so paths and the five-dataset
# allow-list stay in one place.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/config.sh"

SCRIPT_PATH="$SCRIPT_DIR/$(basename "$0")"
SELECTED_DATASETS=(synthetic stocks wind weather DuckDuckGeese)

# The TIA target-recall sweep is the longest phase (5 targets x 5 datasets).
# Override this if the cluster partition has a different maximum wall time.
SBATCH_TIME="${SBATCH_TIME:-7-00:00:00}"
SBATCH_MEM="${SBATCH_MEM:-32G}"
SBATCH_CPUS_PER_TASK="${SBATCH_CPUS_PER_TASK:-1}"
SBATCH_PARTITION="${SBATCH_PARTITION:-}"
SLURM_LOG_ROOT="${SLURM_LOG_ROOT:-$ROOT/slurm_all_experiments}"
JAR_PATH="${JAR_PATH:-$ROOT/target/MS-Index-1.0-jar-with-dependencies.jar}"

MODE=""
PHASE=""
DATASET=""
JOB_MANIFEST=""

usage() {
    sed -n '1,24p' "$0"
    echo
    echo "Usage:"
    echo "  bash scripts/run_all_selected_experiments_slurm.sh --print-commands"
    echo "  bash scripts/run_all_selected_experiments_slurm.sh --submit"
    echo
    echo "Options:"
    echo "  --print-commands   Show all planned sbatch commands without submitting."
    echo "  --submit           Submit the full dependency chain."
}

quote_shell() {
    printf '%q' "$1"
}

validate_dataset() {
    local candidate="$1"
    local selected
    for selected in "${SELECTED_DATASETS[@]}"; do
        [[ "$candidate" == "$selected" ]] && return 0
    done
    echo "Dataset '$candidate' is outside the selected allow-list" >&2
    exit 1
}

dataset_path() {
    case "$1" in
        synthetic) printf '%s\n' "$SYNTHETIC_DIR" ;;
        stocks) printf '%s\n' "$STOCKS_DIR" ;;
        wind) printf '%s\n' "$WIND_DIR" ;;
        weather) printf '%s\n' "$WEATHER_DIR" ;;
        DuckDuckGeese) printf '%s\n' "$UEA_BASE_DIR/DuckDuckGeese" ;;
        *) return 1 ;;
    esac
}

validate_paths() {
    local dataset path
    for dataset in "${SELECTED_DATASETS[@]}"; do
        path="$(dataset_path "$dataset")"
        if [[ ! -d "$path" ]]; then
            echo "Missing dataset path for $dataset: $path" >&2
            exit 1
        fi
    done
}

run_matrix_family() {
    local model_type="$1"
    MODEL_TYPE="$model_type" TRAIN_ONLY=true RESUME=false \
        bash "$SCRIPT_DIR/run_approximate_training_matrix.sh"
    MODEL_TYPE="$model_type" TRAIN_ONLY=false RESUME=true \
        bash "$SCRIPT_DIR/run_approximate_training_matrix.sh"
}

run_phase() {
    case "$PHASE" in
        hybrid)
            validate_dataset "$DATASET"
            exec bash "$SCRIPT_DIR/HYBRID_jobs.sh" --dataset "$DATASET"
            ;;
        approx-tsa)
            run_matrix_family target_specific_classifier
            ;;
        approx-tia)
            run_matrix_family target_independent_regressor
            ;;
        tsa-sweep)
            exec bash "$SCRIPT_DIR/run_target_recall_sweep.sh"
            ;;
        tia-sweep)
            exec bash "$SCRIPT_DIR/run_target_independent_recall_sweep.sh"
            ;;
        *)
            echo "Unknown phase '$PHASE'" >&2
            exit 1
            ;;
    esac
}

submit_job() {
    local name="$1"
    local dependency="$2"
    local wrap="$3"
    local args=(
        --parsable
        --chdir="$ROOT"
        --job-name="$name"
        --time="$SBATCH_TIME"
        --mem="$SBATCH_MEM"
        --cpus-per-task="$SBATCH_CPUS_PER_TASK"
        --output="$SLURM_LOG_ROOT/%x-%j.out"
        --error="$SLURM_LOG_ROOT/%x-%j.err"
        --export=ALL
    )
    if [[ -n "$SBATCH_PARTITION" ]]; then
        args+=(--partition="$SBATCH_PARTITION")
    fi
    if [[ -n "$dependency" ]]; then
        args+=(--dependency="$dependency")
    fi

    if [[ "$MODE" == "print" ]]; then
        printf 'sbatch'
        printf ' %q' "${args[@]}"
        printf ' --wrap=%q\n' "$wrap"
        return 0
    fi

    mkdir -p "$SLURM_LOG_ROOT"
    local raw_id job_id
    raw_id="$(sbatch "${args[@]}" --wrap="$wrap")"
    job_id="${raw_id%%;*}"
    printf 'Submitted %-12s %s' "$name" "$job_id" >&2
    if [[ -n "$dependency" ]]; then
        printf ' (%s)' "$dependency" >&2
    fi
    printf '\n' >&2
    printf '%s\t%s\t%s\n' "$name" "$job_id" "$dependency" >> "$JOB_MANIFEST"
    printf '%s\n' "$job_id"
}

dependency_for() {
    local dependency_type="$1"
    shift
    local ids=()
    local id
    for id in "$@"; do
        [[ -n "$id" ]] && ids+=("$id")
    done
    if [[ ${#ids[@]} -eq 0 ]]; then
        printf '\n'
        return 0
    fi
    local joined
    joined="$(IFS=:; printf '%s' "${ids[*]}")"
    printf '%s:%s\n' "$dependency_type" "$joined"
}

submit_all() {
    validate_paths
    if [[ ! -f "$JAR_PATH" ]]; then
        echo "Missing JAR: $JAR_PATH" >&2
        echo "Build it first with: mvn -q -DskipTests package" >&2
        exit 1
    fi

    mkdir -p "$SLURM_LOG_ROOT"
    local stamp
    stamp="$(date '+%Y%m%d_%H%M%S')"
    JOB_MANIFEST="$SLURM_LOG_ROOT/submission_${stamp}.tsv"
    printf 'job_name\tjob_id\tdependency\n' > "$JOB_MANIFEST"

    local dataset job_id
    local hybrid_ids=()
    for dataset in "${SELECTED_DATASETS[@]}"; do
        job_id="$(submit_job "hybrid-${dataset//[^A-Za-z0-9_-]/_}" "" \
            "bash $(quote_shell "$SCRIPT_PATH") --phase hybrid --dataset $(quote_shell "$dataset")" | tail -n 1)"
        hybrid_ids+=("$job_id")
    done

    local hybrid_dependency
    hybrid_dependency="$(dependency_for afterok "${hybrid_ids[@]}")"
    local tsa_id tia_id tsa_sweep_id
    tsa_id="$(submit_job approx-tsa "$hybrid_dependency" \
        "bash $(quote_shell "$SCRIPT_PATH") --phase approx-tsa" | tail -n 1)"
    tia_id="$(submit_job approx-tia "$(dependency_for afterok "$tsa_id")" \
        "bash $(quote_shell "$SCRIPT_PATH") --phase approx-tia" | tail -n 1)"
    tsa_sweep_id="$(submit_job tsa-sweep "$(dependency_for afterok "$tia_id")" \
        "bash $(quote_shell "$SCRIPT_PATH") --phase tsa-sweep" | tail -n 1)"
    submit_job tia-sweep "$(dependency_for afterok "$tsa_sweep_id")" \
        "bash $(quote_shell "$SCRIPT_PATH") --phase tia-sweep" >/dev/null

    echo "Submission manifest: $JOB_MANIFEST"
    echo "Monitor with: squeue -u \$USER"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --submit)
            MODE="submit"
            ;;
        --print-commands)
            MODE="print"
            ;;
        --phase)
            [[ $# -ge 2 ]] || { echo "--phase requires a value" >&2; exit 1; }
            PHASE="$2"
            shift
            ;;
        --dataset)
            [[ $# -ge 2 ]] || { echo "--dataset requires a value" >&2; exit 1; }
            DATASET="$2"
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
    shift
done

if [[ -n "$PHASE" ]]; then
    run_phase
fi

case "$MODE" in
    submit)
        submit_all
        ;;
    print)
        echo "# Selected datasets: ${SELECTED_DATASETS[*]}"
        echo "# Hybrid jobs (five independent jobs):"
        for DATASET in "${SELECTED_DATASETS[@]}"; do
            submit_job "hybrid-${DATASET//[^A-Za-z0-9_-]/_}" "" \
                "bash $(quote_shell "$SCRIPT_PATH") --phase hybrid --dataset $(quote_shell "$DATASET")"
        done
        echo "# Remaining phases are submitted automatically with afterok dependencies by --submit:"
        submit_job approx-tsa "afterok:<hybrid-job-ids>" \
            "bash $(quote_shell "$SCRIPT_PATH") --phase approx-tsa"
        submit_job approx-tia "afterok:<approx-tsa-job-id>" \
            "bash $(quote_shell "$SCRIPT_PATH") --phase approx-tia"
        submit_job tsa-sweep "afterok:<approx-tia-job-id>" \
            "bash $(quote_shell "$SCRIPT_PATH") --phase tsa-sweep"
        submit_job tia-sweep "afterok:<tsa-sweep-job-id>" \
            "bash $(quote_shell "$SCRIPT_PATH") --phase tia-sweep"
        ;;
    *)
        usage >&2
        exit 1
        ;;
esac
