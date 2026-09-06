#!/usr/bin/env bash
# E3_jobs.sh
#
# One command per dataset — run each line as a separate Slurm job for E3 (conformal calibration).
# Must run AFTER the corresponding E1/E2 job has finished for each dataset.
#
# Usage (sequential):
#   bash scripts/E3_jobs.sh
#
# Usage (Slurm — one job per dataset, after E1/E2 jobs complete):
#   while IFS= read -r cmd; do [[ "$cmd" =~ ^python ]] && sbatch --wrap="$cmd"; done < scripts/E3_jobs.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_ROOT="$ROOT/artifacts/msindexpp_suite"
CAL="$SCRIPT_DIR/calibrate_conformal.py"

RUN="/usr/bin/python3.10 $CAL --delta 0.1"

for DATASET in synthetic stocks wind weather DuckDuckGeese; do
    $RUN \
        --model "$ARTIFACT_ROOT/$DATASET/gbdt-model.json" \
        --data  "$ARTIFACT_ROOT/$DATASET/training-data.jsonl"
done
