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

RUN="/sw/arch/RHEL9/EB_production/2025/software/Python/3.13.5-GCCcore-14.3.0/bin/python $CAL --delta 0.1"

for DATASET in \
    ArticularyWordRecognition AtrialFibrillation BasicMotions CharacterTrajectories \
    Cricket DuckDuckGeese EigenWorms Epilepsy ERing EthanolConcentration \
    FaceDetection FingerMovements HandMovementDirection Handwriting Heartbeat \
    InsectWingbeat JapaneseVowels Libras LSST MotorImagery NATOPS "PEMS-SF" \
    PenDigits PhonemeSpectra RacketSports SelfRegulationSCP1 SelfRegulationSCP2 \
    SpokenArabicDigits StandWalkJump UWaveGestureLibrary \
    synthetic weather stocks wind; do
    $RUN \
        --model "$ARTIFACT_ROOT/$DATASET/gbdt-model.json" \
        --data  "$ARTIFACT_ROOT/$DATASET/training-data.jsonl"
done
