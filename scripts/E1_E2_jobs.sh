#!/usr/bin/env bash
# E1_E2_jobs.sh
#
# One command per dataset — run each line as a separate Slurm job for E1 + E2.
#
# Prerequisites:
#   1. Edit scripts/config.sh with your dataset paths.
#   2. Build the JAR once:  mvn -q -DskipTests package
#
# Each command runs independently (no dependency between datasets).
# E3 (calibration) can start for a dataset as soon as its line below finishes.
#
# Usage (sequential, for testing):
#   bash scripts/E1_E2_jobs.sh
#
# Usage (Slurm — one job per dataset):
#   while IFS= read -r cmd; do [[ "$cmd" =~ ^python ]] && sbatch --wrap="$cmd"; done < scripts/E1_E2_jobs.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SUITE="$SCRIPT_DIR/run_msindexpp_dataset_suite.py"
ARTIFACT_ROOT="$ROOT/artifacts/msindexpp_suite"

RUN="/usr/bin/python3.10 $SUITE --artifact-root $ARTIFACT_ROOT --n-queries $N_QUERIES --max-workers $MAX_WORKERS --seed $SEED --benchmark-algorithms MSINDEX MASS MSINDEXPP"

# --- 30 UEA datasets ---
$RUN --dataset-spec "ArticularyWordRecognition:$UEA_BASE_DIR/ArticularyWordRecognition:28:300"   --max-m 10000
$RUN --dataset-spec "AtrialFibrillation:$UEA_BASE_DIR/AtrialFibrillation:128:15"               --max-m 10000
$RUN --dataset-spec "BasicMotions:$UEA_BASE_DIR/BasicMotions:20:40"                             --max-m 10000
$RUN --dataset-spec "CharacterTrajectories:$UEA_BASE_DIR/CharacterTrajectories:20:1436"         --max-m 10000
$RUN --dataset-spec "Cricket:$UEA_BASE_DIR/Cricket:239:108"                                     --max-m 10000
$RUN --dataset-spec "DuckDuckGeese:$UEA_BASE_DIR/DuckDuckGeese:54:50"                          --max-m 10000
$RUN --dataset-spec "EigenWorms:$UEA_BASE_DIR/EigenWorms:3596:131:3596"                        --max-m 10000
$RUN --dataset-spec "Epilepsy:$UEA_BASE_DIR/Epilepsy:41:138"                                   --max-m 10000
$RUN --dataset-spec "ERing:$UEA_BASE_DIR/ERing:20:270"                                         --max-m 10000
$RUN --dataset-spec "EthanolConcentration:$UEA_BASE_DIR/EthanolConcentration:350:263"          --max-m 10000
$RUN --dataset-spec "FaceDetection:$UEA_BASE_DIR/FaceDetection:20:5890"                        --max-m 10000
$RUN --dataset-spec "FingerMovements:$UEA_BASE_DIR/FingerMovements:20:316"                     --max-m 10000
$RUN --dataset-spec "HandMovementDirection:$UEA_BASE_DIR/HandMovementDirection:80:160"          --max-m 10000
$RUN --dataset-spec "Handwriting:$UEA_BASE_DIR/Handwriting:30:850"                             --max-m 10000
$RUN --dataset-spec "Heartbeat:$UEA_BASE_DIR/Heartbeat:81:205"                                 --max-m 10000
$RUN --dataset-spec "InsectWingbeat:$UEA_BASE_DIR/InsectWingbeat:2:25000"                      --max-m 10000
$RUN --dataset-spec "JapaneseVowels:$UEA_BASE_DIR/JapaneseVowels:7:370"                        --max-m 10000
$RUN --dataset-spec "Libras:$UEA_BASE_DIR/Libras:20:180"                                       --max-m 10000
$RUN --dataset-spec "LSST:$UEA_BASE_DIR/LSST:20:2466"                                         --max-m 10000
$RUN --dataset-spec "MotorImagery:$UEA_BASE_DIR/MotorImagery:600:278:600"                      --max-m 10000
$RUN --dataset-spec "NATOPS:$UEA_BASE_DIR/NATOPS:20:180"                                       --max-m 10000
$RUN --dataset-spec "PEMS-SF:$UEA_BASE_DIR/PEMS-SF:28:267"                                    --max-m 10000
$RUN --dataset-spec "PenDigits:$UEA_BASE_DIR/PenDigits:8:7494"                                 --max-m 10000
$RUN --dataset-spec "PhonemeSpectra:$UEA_BASE_DIR/PhonemeSpectra:43:3353"                      --max-m 10000
$RUN --dataset-spec "RacketSports:$UEA_BASE_DIR/RacketSports:20:152"                           --max-m 10000
$RUN --dataset-spec "SelfRegulationSCP1:$UEA_BASE_DIR/SelfRegulationSCP1:179:293"              --max-m 10000
$RUN --dataset-spec "SelfRegulationSCP2:$UEA_BASE_DIR/SelfRegulationSCP2:230:200"              --max-m 10000
$RUN --dataset-spec "SpokenArabicDigits:$UEA_BASE_DIR/SpokenArabicDigits:4:6599"               --max-m 10000
$RUN --dataset-spec "StandWalkJump:$UEA_BASE_DIR/StandWalkJump:500:15:500"                     --max-m 10000
$RUN --dataset-spec "UWaveGestureLibrary:$UEA_BASE_DIR/UWaveGestureLibrary:63:320"             --max-m 10000

# --- 4 custom datasets ---
$RUN --dataset-spec "synthetic:$SYNTHETIC_DIR:1024:100:4096"  --max-m 4096
$RUN --dataset-spec "weather:$WEATHER_DIR:1488:100:8784"      --max-m 16384
$RUN --dataset-spec "stocks:$STOCKS_DIR:730:100:22335"        --max-m 16384
$RUN --dataset-spec "wind:$WIND_DIR:1800:1:432000"            --max-m 32768
