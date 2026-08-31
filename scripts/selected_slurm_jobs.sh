#!/usr/bin/env bash
# selected_slurm_jobs.sh
#
# Submit or print Slurm jobs for the selected dataset subset used in the paper
# reruns:
#   synthetic, stocks, wind, weather, DuckDuckGeese
#
# The script is a thin wrapper around:
#   scripts/EARLY_TERMINATION_jobs.sh
#   scripts/HYBRID_jobs.sh
#
# Examples:
#   bash scripts/selected_slurm_jobs.sh --print-commands
#   bash scripts/selected_slurm_jobs.sh --submit
#   bash scripts/selected_slurm_jobs.sh --submit --pipeline early
#   bash scripts/selected_slurm_jobs.sh --submit --pipeline hybrid

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SELECTED_DATASETS=(
    synthetic
    stocks
    wind
    weather
    DuckDuckGeese
)

PIPELINE="both"
MODE=""

usage() {
    sed -n '1,16p' "$0"
    echo
    echo "Options:"
    echo "  --submit                 Submit the selected jobs with sbatch."
    echo "  --print-commands         Print the commands without submitting."
    echo "  --pipeline early         Only submit/print early-termination jobs."
    echo "  --pipeline hybrid        Only submit/print hybrid jobs."
    echo "  --pipeline both          Submit/print both pipelines (default)."
    echo
    echo "Datasets:"
    printf '  %s\n' "${SELECTED_DATASETS[@]}"
}

print_pipeline() {
    local pipeline="$1"
    local dataset
    local print_cmd_output
    local job_name
    case "$pipeline" in
        early)
            print_cmd_output="$(bash "$SCRIPT_DIR/EARLY_TERMINATION_jobs.sh" --print-commands)"
            ;;
        hybrid)
            print_cmd_output="$(bash "$SCRIPT_DIR/HYBRID_jobs.sh" --print-commands)"
            ;;
        *) echo "Unknown pipeline '$pipeline'" >&2; exit 1 ;;
    esac

    for dataset in "${SELECTED_DATASETS[@]}"; do
        job_name="${pipeline}-${dataset//[^A-Za-z0-9_-]/_}"
        printf '%s\n' "$print_cmd_output" | grep -- "--job-name=${job_name} " || {
            echo "Could not build $pipeline command for $dataset" >&2
            exit 1
        }
    done
}

submit_selected() {
    local pipeline="$1"
    local dataset
    local job_name
    local sbatch_output
    for dataset in "${SELECTED_DATASETS[@]}"; do
        job_name="${pipeline}-${dataset//[^A-Za-z0-9_-]/_}"
        case "$pipeline" in
            early)
                sbatch_output="$(bash "$SCRIPT_DIR/EARLY_TERMINATION_jobs.sh" --print-commands | grep -- "--job-name=${job_name} " || true)"
                [[ -n "$sbatch_output" ]] || { echo "Could not build early command for $dataset" >&2; exit 1; }
                eval "$sbatch_output"
                ;;
            hybrid)
                sbatch_output="$(bash "$SCRIPT_DIR/HYBRID_jobs.sh" --print-commands | grep -- "--job-name=${job_name} " || true)"
                [[ -n "$sbatch_output" ]] || { echo "Could not build hybrid command for $dataset" >&2; exit 1; }
                eval "$sbatch_output"
                ;;
            *)
                echo "Unknown pipeline '$pipeline'" >&2
                exit 1
                ;;
        esac
    done
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --submit)
            MODE="submit"
            ;;
        --print-commands)
            MODE="print"
            ;;
        --pipeline)
            [[ $# -ge 2 ]] || { echo "--pipeline requires a value" >&2; exit 1; }
            PIPELINE="$2"
            shift
            ;;
        --pipeline=*)
            PIPELINE="${1#*=}"
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

if [[ -z "$MODE" ]]; then
    usage >&2
    exit 1
fi

case "$PIPELINE" in
    early|hybrid) ;;
    both) ;;
    *)
        echo "Invalid --pipeline value '$PIPELINE'. Use early, hybrid, or both." >&2
        exit 1
        ;;
esac

if [[ "$MODE" == "print" ]]; then
    if [[ "$PIPELINE" == "both" ]]; then
        print_pipeline early
        print_pipeline hybrid
    else
        print_pipeline "$PIPELINE"
    fi
else
    if [[ "$PIPELINE" == "both" ]]; then
        submit_selected early
        submit_selected hybrid
    else
        submit_selected "$PIPELINE"
    fi
fi
