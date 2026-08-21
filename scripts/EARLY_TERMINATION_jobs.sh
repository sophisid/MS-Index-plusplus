#!/usr/bin/env bash
# EARLY_TERMINATION_jobs.sh
#
# Run MSINDEXPP early-termination experiments independently per dataset, or
# submit one Slurm job per dataset.  The three model variants are evaluated as:
#   raw model -> raw queries
#   normalized model -> normalized queries
#   joint model -> raw and normalized queries
#
# Training artifacts stay under TRAINING_ROOT.  Models and evaluation outputs
# are copied to RESULT_ROOT and can be shared without the training data.
#
# Prerequisites:
#   1. Edit scripts/config.sh with the dataset paths.
#   2. Build the JAR once: mvn -q -DskipTests package
#
# Examples:
#   bash scripts/EARLY_TERMINATION_jobs.sh --print-commands
#   bash scripts/EARLY_TERMINATION_jobs.sh --submit
#   bash scripts/EARLY_TERMINATION_jobs.sh --dataset synthetic

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

UEA_BASE_DIR_OVERRIDE="${UEA_BASE_DIR:-}"
SYNTHETIC_DIR_OVERRIDE="${SYNTHETIC_DIR:-}"
WEATHER_DIR_OVERRIDE="${WEATHER_DIR:-}"
STOCKS_DIR_OVERRIDE="${STOCKS_DIR:-}"
WIND_DIR_OVERRIDE="${WIND_DIR:-}"
source "$SCRIPT_DIR/config.sh"
[[ -n "$UEA_BASE_DIR_OVERRIDE" ]] && UEA_BASE_DIR="$UEA_BASE_DIR_OVERRIDE"
[[ -n "$SYNTHETIC_DIR_OVERRIDE" ]] && SYNTHETIC_DIR="$SYNTHETIC_DIR_OVERRIDE"
[[ -n "$WEATHER_DIR_OVERRIDE" ]] && WEATHER_DIR="$WEATHER_DIR_OVERRIDE"
[[ -n "$STOCKS_DIR_OVERRIDE" ]] && STOCKS_DIR="$STOCKS_DIR_OVERRIDE"
[[ -n "$WIND_DIR_OVERRIDE" ]] && WIND_DIR="$WIND_DIR_OVERRIDE"

JAR_PATH="${JAR_PATH:-$ROOT/target/MS-Index-1.0-jar-with-dependencies.jar}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.10}"
SUITE="$ROOT/scripts/run_msindexpp_dataset_suite.py"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$ROOT/artifacts/early_termination_all_20260821}"
TRAINING_ROOT="${TRAINING_ROOT:-$ARTIFACT_ROOT/training_private}"
RESULT_ROOT="${RESULT_ROOT:-$ARTIFACT_ROOT/results}"
if [[ "$ARTIFACT_ROOT" != /* ]]; then ARTIFACT_ROOT="$ROOT/$ARTIFACT_ROOT"; fi
if [[ "$TRAINING_ROOT" != /* ]]; then TRAINING_ROOT="$ROOT/$TRAINING_ROOT"; fi
if [[ "$RESULT_ROOT" != /* ]]; then RESULT_ROOT="$ROOT/$RESULT_ROOT"; fi

N_QUERIES="${N_QUERIES:-100}"
N_OVERRIDE="${N_OVERRIDE:-}"
MAX_WORKERS="${MAX_WORKERS:-1}"
SEED="${SEED:-1}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-64}"
ACCURACY_TARGET="${ACCURACY_TARGET:-0.9}"
MODEL_SCORE_THRESHOLD="${MODEL_SCORE_THRESHOLD:-0.5}"
RESUME="${RESUME:-true}"
SBATCH_TIME="${SBATCH_TIME:-12:00:00}"
SBATCH_PARTITION="${SBATCH_PARTITION:-}"
SBATCH_MEM="${SBATCH_MEM:-}"
SBATCH_CPUS_PER_TASK="${SBATCH_CPUS_PER_TASK:-}"

export JAR_PATH PYTHON_BIN TRAINING_ROOT RESULT_ROOT N_QUERIES N_OVERRIDE MAX_WORKERS SEED
export CHECKPOINT_INTERVAL ACCURACY_TARGET MODEL_SCORE_THRESHOLD RESUME

if [[ ! -f "$JAR_PATH" ]]; then
    echo "Missing JAR at $JAR_PATH" >&2
    echo "Build it first with: mvn -q -DskipTests package" >&2
    exit 1
fi
if [[ ! -f "$SUITE" ]]; then
    echo "Missing suite runner at $SUITE" >&2
    exit 1
fi
if ! "$PYTHON_BIN" --version >/dev/null 2>&1; then
    echo "Missing Python interpreter at $PYTHON_BIN" >&2
    exit 1
fi

dataset_config() {
    local dataset="$1"
    case "$dataset" in
        ArticularyWordRecognition) printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$UEA_BASE_DIR/ArticularyWordRecognition" 28 300 10000 ;;
        AtrialFibrillation) printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$UEA_BASE_DIR/AtrialFibrillation" 128 15 10000 ;;
        BasicMotions) printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$UEA_BASE_DIR/BasicMotions" 20 40 10000 ;;
        CharacterTrajectories) printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$UEA_BASE_DIR/CharacterTrajectories" 20 1436 10000 ;;
        Cricket) printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$UEA_BASE_DIR/Cricket" 239 108 10000 ;;
        DuckDuckGeese) printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$UEA_BASE_DIR/DuckDuckGeese" 54 50 10000 ;;
        EigenWorms) printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$UEA_BASE_DIR/EigenWorms" 3596 131 10000 ;;
        Epilepsy) printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$UEA_BASE_DIR/Epilepsy" 41 138 10000 ;;
        ERing) printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$UEA_BASE_DIR/ERing" 20 270 10000 ;;
        EthanolConcentration) printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$UEA_BASE_DIR/EthanolConcentration" 350 263 10000 ;;
        FaceDetection) printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$UEA_BASE_DIR/FaceDetection" 20 5890 10000 ;;
        FingerMovements) printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$UEA_BASE_DIR/FingerMovements" 20 316 10000 ;;
        HandMovementDirection) printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$UEA_BASE_DIR/HandMovementDirection" 80 160 10000 ;;
        Handwriting) printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$UEA_BASE_DIR/Handwriting" 30 850 10000 ;;
        Heartbeat) printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$UEA_BASE_DIR/Heartbeat" 81 205 10000 ;;
        InsectWingbeat) printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$UEA_BASE_DIR/InsectWingbeat" 2 25000 10000 ;;
        JapaneseVowels) printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$UEA_BASE_DIR/JapaneseVowels" 7 370 10000 ;;
        Libras) printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$UEA_BASE_DIR/Libras" 20 180 10000 ;;
        LSST) printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$UEA_BASE_DIR/LSST" 20 2466 10000 ;;
        MotorImagery) printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$UEA_BASE_DIR/MotorImagery" 600 278 10000 ;;
        NATOPS) printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$UEA_BASE_DIR/NATOPS" 20 180 10000 ;;
        PEMS-SF) printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$UEA_BASE_DIR/PEMS-SF" 28 267 10000 ;;
        PenDigits) printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$UEA_BASE_DIR/PenDigits" 8 7494 10000 ;;
        PhonemeSpectra) printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$UEA_BASE_DIR/PhonemeSpectra" 43 3353 10000 ;;
        RacketSports) printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$UEA_BASE_DIR/RacketSports" 20 152 10000 ;;
        SelfRegulationSCP1) printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$UEA_BASE_DIR/SelfRegulationSCP1" 179 293 10000 ;;
        SelfRegulationSCP2) printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$UEA_BASE_DIR/SelfRegulationSCP2" 230 200 10000 ;;
        SpokenArabicDigits) printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$UEA_BASE_DIR/SpokenArabicDigits" 4 6599 10000 ;;
        StandWalkJump) printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$UEA_BASE_DIR/StandWalkJump" 500 15 10000 ;;
        UWaveGestureLibrary) printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$UEA_BASE_DIR/UWaveGestureLibrary" 63 320 10000 ;;
        synthetic) printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$SYNTHETIC_DIR" 1024 100 4096 ;;
        weather) printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$WEATHER_DIR" 1488 100 16384 ;;
        stocks) printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$STOCKS_DIR" 730 100 16384 ;;
        wind) printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$WIND_DIR" 1800 1 32768 ;;
        *) return 1 ;;
    esac
}

all_datasets() {
    printf '%s\n' \
        ArticularyWordRecognition AtrialFibrillation BasicMotions CharacterTrajectories \
        Cricket DuckDuckGeese EigenWorms Epilepsy ERing EthanolConcentration \
        FaceDetection FingerMovements HandMovementDirection Handwriting Heartbeat \
        InsectWingbeat JapaneseVowels Libras LSST MotorImagery NATOPS PEMS-SF \
        PenDigits PhonemeSpectra RacketSports SelfRegulationSCP1 SelfRegulationSCP2 \
        SpokenArabicDigits StandWalkJump UWaveGestureLibrary \
        synthetic weather stocks wind
}

append_runtime() {
    local path="$1" phase="$2" variant="$3" wall_time_s="$4" status="$5" log_path="$6"
    if [[ ! -f "$path" ]]; then
        printf '%s\n' "phase,variant,wall_time_s,status,log_path" > "$path"
    fi
    printf '%s,%s,%s,%s,%s\n' "$phase" "$variant" "$wall_time_s" "$status" "$log_path" >> "$path"
}

run_phase() {
    local phase="$1" variant="$2" private_root="$3" runtime_csv="$4" marker="$5" log_path="$6"
    shift 6
    if [[ "$RESUME" == "true" && -f "$marker" ]]; then
        append_runtime "$runtime_csv" "$phase" "$variant" 0 cached "$log_path"
        return 0
    fi

    mkdir -p "$private_root"
    local start end elapsed
    start="$(date +%s)"
    if "$PYTHON_BIN" "$SUITE" "$@" >"$log_path" 2>&1; then
        end="$(date +%s)"
        elapsed=$((end - start))
        append_runtime "$runtime_csv" "$phase" "$variant" "$elapsed" ok "$log_path"
        printf '%s\n' "$(date -Is)" > "$marker"
    else
        end="$(date +%s)"
        elapsed=$((end - start))
        append_runtime "$runtime_csv" "$phase" "$variant" "$elapsed" failed "$log_path"
        return 1
    fi
}

copy_if_exists() {
    local source="$1"
    local target="$2"
    if [[ -e "$source" ]]; then
        mkdir -p "$(dirname "$target")"
        if [[ -d "$source" ]]; then
            mkdir -p "$target"
            cp -a "$source/." "$target/"
        else
            cp -a "$source" "$target"
        fi
    fi
}

export_variant_results() {
    local dataset="$1" variant="$2" private_root="$3" result_root="$4"
    local variant_result="$result_root/variants/$variant"
    local suite_dataset_root="$private_root/$dataset"
    mkdir -p "$variant_result"

    copy_if_exists "$suite_dataset_root/gbdt-model.json" "$variant_result/gbdt-model.json"
    copy_if_exists "$suite_dataset_root/evaluation_results.jsonl" "$variant_result/evaluation_results.jsonl"
    copy_if_exists "$suite_dataset_root/timing_results.jsonl" "$variant_result/timing_results.jsonl"
    copy_if_exists "$suite_dataset_root/training_runtime.json" "$variant_result/training_runtime.json"
    copy_if_exists "$suite_dataset_root/trainer_metrics.txt" "$variant_result/trainer_metrics.txt"
    copy_if_exists "$suite_dataset_root/eval_runs" "$variant_result/eval_runs"
    copy_if_exists "$suite_dataset_root/timing_runs" "$variant_result/timing_runs"
    copy_if_exists "$suite_dataset_root/index_builds" "$variant_result/index_builds"
}

run_dataset() {
    local dataset="$1" config_line
    if ! config_line="$(dataset_config "$dataset")"; then
        echo "Unknown dataset '$dataset'" >&2
        exit 1
    fi

    local name data_path q_len n max_m
    IFS=$'\t' read -r name data_path q_len n max_m <<< "$config_line"
    if [[ -n "$N_OVERRIDE" ]]; then n="$N_OVERRIDE"; fi
    if [[ ! -d "$data_path" ]]; then
        echo "Dataset path does not exist for $dataset: $data_path" >&2
        exit 1
    fi

    local dataset_training="$TRAINING_ROOT/$dataset"
    local dataset_result="$RESULT_ROOT/$dataset"
    local runtime_csv="$dataset_result/phase_runtime.csv"
    mkdir -p "$dataset_training" "$dataset_result"
    printf 'dataset=%s\ndata_path=%s\nN=%s\nq_len=%s\nmax_m=%s\n' \
        "$dataset" "$data_path" "$n" "$q_len" "$max_m" > "$dataset_result/config.txt"

    local spec="$dataset:$data_path:$q_len:$n:$max_m"
    local total_start="$(date +%s)"
    echo "[$(date -Is)] START dataset=$dataset N=$n qLen=$q_len maxM=$max_m"

    local variant private_root result_variant
    for variant in raw normalized joint; do
        private_root="$dataset_training/$variant"
        case "$variant" in
            raw) normalize_values=(false) ;;
            normalized) normalize_values=(true) ;;
            joint) normalize_values=(false true) ;;
        esac

        local normalize_args=(--normalize-values "${normalize_values[@]}")
        local train_cmd=(
            --artifact-root "$private_root"
            --dataset-spec "$spec"
            --max-m "$max_m"
            --n-queries "$N_QUERIES"
            --max-workers "$MAX_WORKERS"
            --seed "$SEED"
            --checkpoint-interval "$CHECKPOINT_INTERVAL"
            --accuracy-target "$ACCURACY_TARGET"
            --model-score-threshold "$MODEL_SCORE_THRESHOLD"
            "${normalize_args[@]}"
            --benchmark-algorithms MSINDEX MASS MSINDEXPP
            --skip-eval
            --skip-benchmark
        )
        if [[ "$RESUME" != "true" ]]; then train_cmd+=(--force); fi
        run_phase training "$variant" "$private_root" "$runtime_csv" \
            "$private_root/.training.done" "$private_root/training_suite.log" \
            "${train_cmd[@]}"

        local eval_cmd=(
            --artifact-root "$private_root"
            --dataset-spec "$spec"
            --max-m "$max_m"
            --n-queries "$N_QUERIES"
            --max-workers "$MAX_WORKERS"
            --seed "$SEED"
            --checkpoint-interval "$CHECKPOINT_INTERVAL"
            --accuracy-target "$ACCURACY_TARGET"
            --model-score-threshold "$MODEL_SCORE_THRESHOLD"
            "${normalize_args[@]}"
            --benchmark-algorithms MSINDEX MASS MSINDEXPP
            --skip-train
            --skip-benchmark
        )
        if [[ "$RESUME" != "true" ]]; then eval_cmd+=(--force); fi
        run_phase evaluation "$variant" "$private_root" "$runtime_csv" \
            "$private_root/.evaluation.done" "$private_root/evaluation_suite.log" \
            "${eval_cmd[@]}"
        if [[ "$variant" == "joint" ]]; then
            local benchmark_cmd=(
                --artifact-root "$private_root"
                --dataset-spec "$spec"
                --max-m "$max_m"
                --n-queries "$N_QUERIES"
                --max-workers "$MAX_WORKERS"
                --seed "$SEED"
                --checkpoint-interval "$CHECKPOINT_INTERVAL"
                --accuracy-target "$ACCURACY_TARGET"
                --model-score-threshold "$MODEL_SCORE_THRESHOLD"
                "${normalize_args[@]}"
                --benchmark-algorithms MSINDEX MASS MSINDEXPP
                --skip-train
                --skip-eval
            )
            if [[ "$RESUME" != "true" ]]; then benchmark_cmd+=(--force); fi
            run_phase benchmark "$variant" "$private_root" "$runtime_csv" \
                "$private_root/.benchmark.done" "$private_root/benchmark_suite.log" \
                "${benchmark_cmd[@]}"
        fi

        export_variant_results "$dataset" "$variant" "$private_root" "$dataset_result"
    done
    local total_end="$(date +%s)"

    printf '%s\n' "dataset,total_wall_time_s" \
        > "$dataset_result/dataset_runtime.csv"
    printf '%s,%s\n' "$dataset" "$((total_end - total_start))" >> "$dataset_result/dataset_runtime.csv"
    echo "[$(date -Is)] DONE dataset=$dataset results=$dataset_result training_data=$dataset_training"
}

submit_dataset() {
    local dataset="$1"
    mkdir -p "$RESULT_ROOT/slurm"
    local output_path="$RESULT_ROOT/slurm/%x-%j.out"
    local error_path="$RESULT_ROOT/slurm/%x-%j.err"
    local job_name="early-${dataset//[^A-Za-z0-9_-]/_}"
    local args=(--time "$SBATCH_TIME" --job-name "$job_name" --output "$output_path" --error "$error_path" --export ALL)
    if [[ -n "$SBATCH_PARTITION" ]]; then args+=(--partition "$SBATCH_PARTITION"); fi
    if [[ -n "$SBATCH_MEM" ]]; then args+=(--mem "$SBATCH_MEM"); fi
    if [[ -n "$SBATCH_CPUS_PER_TASK" ]]; then args+=(--cpus-per-task "$SBATCH_CPUS_PER_TASK"); fi
    sbatch "${args[@]}" --wrap="bash '$SCRIPT_DIR/EARLY_TERMINATION_jobs.sh' --dataset '$dataset'"
}

print_commands() {
    while IFS= read -r dataset; do
        printf 'sbatch --time=%q --job-name=%q --output=%q --error=%q --export=ALL --wrap=%q\n' \
            "$SBATCH_TIME" "early-${dataset//[^A-Za-z0-9_-]/_}" \
            "$RESULT_ROOT/slurm/%x-%j.out" "$RESULT_ROOT/slurm/%x-%j.err" \
            "bash '$SCRIPT_DIR/EARLY_TERMINATION_jobs.sh' --dataset '$dataset'"
    done < <(all_datasets)
}

usage() {
    sed -n '1,26p' "$0"
    echo
    echo "Options:"
    echo "  --submit              Submit one Slurm job per dataset (12h default)."
    echo "  --print-commands      Print the sbatch commands without submitting."
    echo "  --dataset NAME        Run one dataset inside the current Slurm job."
    echo
    echo "Results:       ${RESULT_ROOT:-<artifact-root>/results}"
    echo "Training data: ${TRAINING_ROOT:-<artifact-root>/training_private}"
}

case "${1:-}" in
    --submit)
        while IFS= read -r dataset; do submit_dataset "$dataset"; done < <(all_datasets)
        ;;
    --print-commands)
        print_commands
        ;;
    --dataset)
        [[ $# -eq 2 ]] || { echo "--dataset requires a name" >&2; exit 1; }
        run_dataset "$2"
        ;;
    --help|-h)
        usage
        ;;
    *)
        usage >&2
        exit 1
        ;;
esac
