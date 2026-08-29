#!/usr/bin/env bash
# HYBRID_jobs.sh
#
# Run the HYBRID routing experiment independently for each dataset, or submit
# one Slurm job per dataset.  The default evaluation matrix is:
#   raw model -> raw
#   normalized model -> normalized
#   joint model -> raw and normalized
# Set EVAL_MATRIX=all to evaluate every model against both query modes.
# Training inputs/logs are written under TRAINING_ROOT.  Models and evaluation
# outputs are written under RESULT_ROOT, so RESULT_ROOT can be shared without
# sending the collected training data.
#
# Prerequisites:
#   1. Edit scripts/config.sh with the dataset paths.
#   2. Build the JAR once: mvn -q -DskipTests package
#
# Examples:
#   bash scripts/HYBRID_jobs.sh --print-commands
#   bash scripts/HYBRID_jobs.sh --submit
#   bash scripts/HYBRID_jobs.sh --dataset synthetic

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Keep environment-provided paths while loading the repository defaults.
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
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$ROOT/artifacts_hybrid}"
TRAINING_ROOT="${TRAINING_ROOT:-$ARTIFACT_ROOT/training_private}"
RESULT_ROOT="${RESULT_ROOT:-$ARTIFACT_ROOT/results}"
if [[ "$ARTIFACT_ROOT" != /* ]]; then ARTIFACT_ROOT="$ROOT/$ARTIFACT_ROOT"; fi
if [[ "$TRAINING_ROOT" != /* ]]; then TRAINING_ROOT="$ROOT/$TRAINING_ROOT"; fi
if [[ "$RESULT_ROOT" != /* ]]; then RESULT_ROOT="$ROOT/$RESULT_ROOT"; fi

N_OVERRIDE="${N_OVERRIDE:-}"
TRAIN_QUERIES="${TRAIN_QUERIES:-100}"
EVAL_QUERIES="${EVAL_QUERIES:-100}"
SEED_LIST="${SEED_LIST:-1 3 5 7}"
NOISE_LIST="${NOISE_LIST:-0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9}"
EVAL_SEED="${EVAL_SEED:-9}"
EVAL_MATRIX="${EVAL_MATRIX:-matched}"
RESUME="${RESUME:-true}"
SBATCH_TIME="${SBATCH_TIME:-12:00:00}"
SBATCH_PARTITION="${SBATCH_PARTITION:-}"
SBATCH_MEM="${SBATCH_MEM:-}"
SBATCH_CPUS_PER_TASK="${SBATCH_CPUS_PER_TASK:-}"

export JAR_PATH PYTHON_BIN ARTIFACT_ROOT TRAINING_ROOT RESULT_ROOT N_OVERRIDE TRAIN_QUERIES EVAL_QUERIES
export SEED_LIST NOISE_LIST EVAL_SEED EVAL_MATRIX RESUME

if [[ ! -f "$JAR_PATH" ]]; then
    echo "Missing JAR at $JAR_PATH" >&2
    echo "Build it first with: mvn -q -DskipTests package" >&2
    exit 1
fi
if ! "$PYTHON_BIN" --version >/dev/null 2>&1; then
    echo "Missing Python interpreter at $PYTHON_BIN" >&2
    exit 1
fi

if [[ "$EVAL_MATRIX" != "matched" && "$EVAL_MATRIX" != "all" ]]; then
    echo "EVAL_MATRIX must be 'matched' or 'all', got '$EVAL_MATRIX'" >&2
    exit 1
fi

IFS=' ' read -r -a SEEDS <<< "$SEED_LIST"
IFS=' ' read -r -a NOISE_VALUES <<< "$NOISE_LIST"

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

csv_header_if_missing() {
    local path="$1"
    local header="$2"
    if [[ ! -f "$path" ]]; then
        printf '%s\n' "$header" > "$path"
    fi
}

append_runtime() {
    local path="$1"
    local phase="$2"
    local model_variant="$3"
    local eval_normalize="$4"
    local k="$5"
    local noise="$6"
    local seed="$7"
    local wall_time_s="$8"
    local status="$9"
    local log_path="${10}"
    csv_header_if_missing "$path" "phase,model_variant,normalize,k,noise,seed,wall_time_s,status,log_path"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$phase" "$model_variant" "$eval_normalize" "$k" "$noise" "$seed" "$wall_time_s" "$status" "$log_path" >> "$path"
}

run_logged() {
    local phase="$1"
    local model_variant="$2"
    local eval_normalize="$3"
    local k="$4"
    local noise="$5"
    local seed="$6"
    local run_dir="$7"
    local log_path="$8"
    shift 8

    local runtime_csv="$1"
    shift
    local cmd=("$@")

    local result_file="$run_dir/evaluation.csv"
    if [[ "$phase" == "training" ]]; then
        result_file="$run_dir/training.csv"
    fi
    if [[ "$RESUME" == "true" && -f "$run_dir/runs.csv" && -f "$result_file" ]]; then
        append_runtime "$runtime_csv" "$phase" "$model_variant" "$eval_normalize" "$k" "$noise" "$seed" 0 cached "$log_path"
        return 0
    fi

    mkdir -p "$run_dir"
    rm -f "$run_dir/runs.csv"
    rm -f "$result_file"
    local start end elapsed
    start="$(date +%s)"
    if "${cmd[@]}" >"$log_path" 2>&1; then
        end="$(date +%s)"
        elapsed=$((end - start))
        append_runtime "$runtime_csv" "$phase" "$model_variant" "$eval_normalize" "$k" "$noise" "$seed" "$elapsed" ok "$log_path"
    else
        end="$(date +%s)"
        elapsed=$((end - start))
        append_runtime "$runtime_csv" "$phase" "$model_variant" "$eval_normalize" "$k" "$noise" "$seed" "$elapsed" failed "$log_path"
        return 1
    fi
}

merge_training_csvs() {
    local output_path="$1"
    shift
    "$PYTHON_BIN" - "$output_path" "$@" <<'PY'
import csv
import sys
from pathlib import Path

output = Path(sys.argv[1])
sources = [Path(value) for value in sys.argv[2:]]
if not sources:
    raise SystemExit("No training CSV sources were provided")

fieldnames = None
next_query_id = 0
rows_written = 0
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", newline="", encoding="utf-8") as target:
    writer = None
    for source in sources:
        if not source.exists():
            raise SystemExit(f"Missing training CSV: {source}")
        with source.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise SystemExit(f"Missing CSV header: {source}")
            if fieldnames is None:
                fieldnames = reader.fieldnames
                writer = csv.DictWriter(target, fieldnames=fieldnames)
                writer.writeheader()
            elif reader.fieldnames != fieldnames:
                raise SystemExit(f"CSV header mismatch in {source}")

            source_max_query = -1
            for row in reader:
                query_id = int(row["query_id"])
                source_max_query = max(source_max_query, query_id)
                row["query_id"] = str(query_id + next_query_id)
                writer.writerow(row)
                rows_written += 1
            next_query_id += source_max_query + 1

if rows_written == 0:
    raise SystemExit(f"No rows found in sources for {output}")
PY
}

train_model() {
    local result_root="$1"
    local variant="$2"
    local input_csv="$3"
    local model_path="$result_root/models/$variant/gbdt-model.json"
    local log_path="$result_root/models/$variant/train.log"
    local runtime_csv="$result_root/model_runtime.csv"

    if [[ "$RESUME" == "true" && -f "$model_path" ]]; then
        return 0
    fi

    mkdir -p "$(dirname "$model_path")"
    local start end elapsed
    start="$(date +%s)"
    if "$PYTHON_BIN" "$ROOT/scripts/train_hybrid_gbdt.py" \
        --input "$input_csv" \
        --output "$model_path" >"$log_path" 2>&1; then
        end="$(date +%s)"
        elapsed=$((end - start))
        append_runtime "$runtime_csv" "model_training" "$variant" "" "" "" "" "$elapsed" ok "$log_path"
    else
        end="$(date +%s)"
        elapsed=$((end - start))
        append_runtime "$runtime_csv" "model_training" "$variant" "" "" "" "" "$elapsed" failed "$log_path"
        return 1
    fi
}

collect_training_variant() {
    local dataset="$1"
    local data_path="$2"
    local n="$3"
    local q_len="$4"
    local max_m="$5"
    local normalize="$6"
    local variant="$7"
    local dataset_root="$TRAINING_ROOT/$dataset"
    local run_root="$dataset_root/training_runs/$variant"
    local runtime_csv="$dataset_root/run_runtime.csv"
    local training_csv="$dataset_root/training/$variant.csv"
    local experiment_id=1
    local run_csvs=()

    mkdir -p "$run_root" "$dataset_root/training"
    for seed in "${SEEDS[@]}"; do
        for k in 1 10 100; do
            for noise in "${NOISE_VALUES[@]}"; do
                local key="seed${seed}_k${k}_noise${noise}"
                local run_dir="$run_root/$key"
                local log_path="$run_dir/train.log"
                local run_training_csv="$run_dir/training.csv"
                run_csvs+=("$run_training_csv")
                local cmd=(
                    java -Xmx4g
                    "-Doutput.path=$run_dir"
                    "-Dhybrid.trainingCsv=$run_training_csv"
                    -cp "$JAR_PATH"
                    io/github/Main HYBRID
                    "$data_path" "$n" "$max_m" -1 -1 "$q_len" "$k" "$normalize"
                    "$TRAIN_QUERIES" FULL_NO_STORE "$experiment_id" "$seed" false true "$noise"
                )
                run_logged training "$variant" "$normalize" "$k" "$noise" "$seed" \
                    "$run_dir" "$log_path" "$runtime_csv" "${cmd[@]}"
                experiment_id=$((experiment_id + 1))
            done
        done
    done

    if [[ "$RESUME" != "true" || ! -f "$training_csv" ]]; then
        merge_training_csvs "$training_csv" "${run_csvs[@]}"
    fi
}

evaluation_modes_for_variant() {
    local variant="$1"
    if [[ "$EVAL_MATRIX" == "all" ]]; then
        printf '%s\n' false true
        return
    fi
    case "$variant" in
        raw) printf '%s\n' false ;;
        normalized) printf '%s\n' true ;;
        joint) printf '%s\n' false true ;;
        *) return 1 ;;
    esac
}

run_evaluation_variant() {
    local dataset="$1"
    local data_path="$2"
    local n="$3"
    local q_len="$4"
    local max_m="$5"
    local variant="$6"
    local eval_normalize="$7"
    local dataset_root="$RESULT_ROOT/$dataset"
    local model_path="$dataset_root/models/$variant/gbdt-model.json"
    local eval_root="$dataset_root/evaluation_runs/$variant/norm${eval_normalize}"
    local result_csv="$dataset_root/evaluation/$variant/norm${eval_normalize}.csv"
    local runtime_csv="$dataset_root/evaluation_runtime.csv"
    local experiment_id=1000
    local eval_csvs=()

    mkdir -p "$eval_root" "$(dirname "$result_csv")"
    for k in 1 10 100; do
        for noise in "${NOISE_VALUES[@]}"; do
            local key="k${k}_noise${noise}"
            local run_dir="$eval_root/$key"
            local log_path="$run_dir/eval.log"
            local run_eval_csv="$run_dir/evaluation.csv"
            eval_csvs+=("$run_eval_csv")
            local cmd=(
                java -Xmx4g
                "-Doutput.path=$run_dir"
                "-Dhybrid.modelPath=$model_path"
                "-Dhybrid.evaluationCsv=$run_eval_csv"
                -cp "$JAR_PATH"
                io/github/Main HYBRID
                "$data_path" "$n" "$max_m" -1 -1 "$q_len" "$k" "$eval_normalize"
                "$EVAL_QUERIES" QUERY "$experiment_id" "$EVAL_SEED" false true "$noise"
            )
            run_logged evaluation "$variant" "$eval_normalize" "$k" "$noise" "$EVAL_SEED" \
                "$run_dir" "$log_path" "$runtime_csv" "${cmd[@]}"
            experiment_id=$((experiment_id + 1))
        done
    done

    if [[ "$RESUME" != "true" || ! -f "$result_csv" ]]; then
        merge_training_csvs "$result_csv" "${eval_csvs[@]}"
    fi
}

run_dataset() {
    local dataset="$1"
    local config_line
    if ! config_line="$(dataset_config "$dataset")"; then
        echo "Unknown dataset '$dataset'" >&2
        exit 1
    fi

    local name data_path q_len n max_m
    IFS=$'\t' read -r name data_path q_len n max_m <<< "$config_line"
    if [[ -n "$N_OVERRIDE" ]]; then
        n="$N_OVERRIDE"
    fi
    if [[ ! -d "$data_path" ]]; then
        echo "Dataset path does not exist for $dataset: $data_path" >&2
        exit 1
    fi

    local training_dataset_root="$TRAINING_ROOT/$dataset"
    local result_dataset_root="$RESULT_ROOT/$dataset"
    mkdir -p "$training_dataset_root/training" "$result_dataset_root/evaluation"
    printf 'dataset=%s\ndata_path=%s\nN=%s\nq_len=%s\nmax_m=%s\n' \
        "$dataset" "$data_path" "$n" "$q_len" "$max_m" > "$result_dataset_root/config.txt"

    local total_start training_start training_end model_start model_end evaluation_start evaluation_end total_end
    total_start="$(date +%s)"
    echo "[$(date -Is)] START dataset=$dataset N=$n qLen=$q_len maxM=$max_m"
    training_start="$(date +%s)"
    collect_training_variant "$dataset" "$data_path" "$n" "$q_len" "$max_m" false raw
    collect_training_variant "$dataset" "$data_path" "$n" "$q_len" "$max_m" true normalized
    training_end="$(date +%s)"

    local raw_training="$training_dataset_root/training/raw.csv"
    local normalized_training="$training_dataset_root/training/normalized.csv"
    local joint_training="$training_dataset_root/training/joint.csv"
    if [[ "$RESUME" != "true" || ! -f "$joint_training" ]]; then
        merge_training_csvs "$joint_training" "$raw_training" "$normalized_training"
    fi

    model_start="$(date +%s)"
    train_model "$result_dataset_root" raw "$raw_training"
    train_model "$result_dataset_root" normalized "$normalized_training"
    train_model "$result_dataset_root" joint "$joint_training"
    model_end="$(date +%s)"

    local variant eval_normalize
    evaluation_start="$(date +%s)"
    while IFS= read -r variant; do
        while IFS= read -r eval_normalize; do
            run_evaluation_variant "$dataset" "$data_path" "$n" "$q_len" "$max_m" "$variant" "$eval_normalize"
        done < <(evaluation_modes_for_variant "$variant")
    done < <(printf '%s\n' raw normalized joint)
    evaluation_end="$(date +%s)"
    total_end="$(date +%s)"

    printf '%s\n' \
        "dataset,training_wall_time_s,model_training_wall_time_s,evaluation_wall_time_s,total_wall_time_s" \
        > "$result_dataset_root/dataset_runtime.csv"
    printf '%s,%s,%s,%s,%s\n' "$dataset" \
        "$((training_end - training_start))" \
        "$((model_end - model_start))" \
        "$((evaluation_end - evaluation_start))" \
        "$((total_end - total_start))" >> "$result_dataset_root/dataset_runtime.csv"

    echo "[$(date -Is)] DONE dataset=$dataset results=$result_dataset_root training_data=$training_dataset_root"
}

submit_dataset() {
    local dataset="$1"
    mkdir -p "$RESULT_ROOT/slurm"
    local output_path="$RESULT_ROOT/slurm/%x-%j.out"
    local error_path="$RESULT_ROOT/slurm/%x-%j.err"
    local job_name="hybrid-${dataset//[^A-Za-z0-9_-]/_}"
    local args=(--time "$SBATCH_TIME" --job-name "$job_name" --output "$output_path" --error "$error_path" --export ALL)
    if [[ -n "$SBATCH_PARTITION" ]]; then
        args+=(--partition "$SBATCH_PARTITION")
    fi
    if [[ -n "$SBATCH_MEM" ]]; then
        args+=(--mem "$SBATCH_MEM")
    fi
    if [[ -n "$SBATCH_CPUS_PER_TASK" ]]; then
        args+=(--cpus-per-task "$SBATCH_CPUS_PER_TASK")
    fi
    sbatch "${args[@]}" --wrap="bash '$SCRIPT_DIR/HYBRID_jobs.sh' --dataset '$dataset'"
}

print_commands() {
    while IFS= read -r dataset; do
        printf 'sbatch --time=%q --job-name=%q --output=%q --error=%q --export=ALL --wrap=%q\n' \
            "$SBATCH_TIME" "hybrid-${dataset//[^A-Za-z0-9_-]/_}" \
            "$RESULT_ROOT/slurm/%x-%j.out" "$RESULT_ROOT/slurm/%x-%j.err" \
            "bash '$SCRIPT_DIR/HYBRID_jobs.sh' --dataset '$dataset'"
    done < <(all_datasets)
}

usage() {
    sed -n '1,24p' "$0"
    echo
    echo "Options:"
    echo "  --submit              Submit one Slurm job per dataset (12h default)."
    echo "  --print-commands      Print the sbatch commands without submitting."
    echo "  --dataset NAME        Run one dataset inside the current Slurm job."
    echo
    echo "Results to share: ${RESULT_ROOT:-<artifact-root>/results}"
    echo "Training data:     ${TRAINING_ROOT:-<artifact-root>/training_private}"
}

case "${1:-}" in
    --submit)
        while IFS= read -r dataset; do
            submit_dataset "$dataset"
        done < <(all_datasets)
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
