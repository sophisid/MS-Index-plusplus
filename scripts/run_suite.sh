#!/usr/bin/env bash
# run_suite.sh
#
# Runs training, evaluation, and timing benchmark (MSINDEX, MASS, MSINDEXPP)
# for all 34 datasets. Per-dataset results land in:
#   artifacts/msindexpp_suite/<dataset>/
# Combined summary:
#   artifacts/msindexpp_suite/dataset_summary.csv
# Training time per dataset:
#   artifacts/msindexpp_suite/<dataset>/training_runtime.json
#
# Usage: bash run_suite.sh [--datasets all|custom] [--force]
#   --datasets all     Run all 34 datasets (default)
#   --datasets custom  Run only the 4 custom datasets (synthetic, weather, stocks, wind)
#   --force            Clears all cached results and reruns from scratch.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load paths and settings from config.sh
# shellcheck source=config.sh
source "$SCRIPT_DIR/config.sh"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SUITE="$SCRIPT_DIR/run_msindexpp_dataset_suite.py"
ARTIFACT_ROOT="$ROOT/artifacts/msindexpp_suite"
STAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="$ARTIFACT_ROOT/suite_${STAMP}"
mkdir -p "$LOG_DIR"

FORCE_FLAG=""
RUN_MODE="custom"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)            FORCE_FLAG="--force" ;;
        --datasets)         RUN_MODE="$2"; shift ;;
        --datasets=*)       RUN_MODE="${1#*=}" ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
    shift
done
if [[ "$RUN_MODE" != "all" && "$RUN_MODE" != "custom" ]]; then
    echo "Invalid --datasets value '$RUN_MODE'. Use 'all' or 'custom'." >&2
    exit 1
fi

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_DIR/run.log"; }

# Build JAR if not already built
JAR="$ROOT/target/MS-Index-1.0-jar-with-dependencies.jar"
if [[ ! -f "$JAR" ]]; then
    log "Building JAR..."
    (cd "$ROOT" && mvn -q -DskipTests package) | tee -a "$LOG_DIR/build.log"
fi

run_dataset() {
    local name="$1"
    local spec="$2"   # name:path:qLen:defaultN[:maxLen]
    local max_m="$3"  # maximum series length to load
    local log_file="$LOG_DIR/${name}.log"
    log "START $name"
    if /usr/bin/python3.10 "$SUITE" \
        --artifact-root "$ARTIFACT_ROOT" \
        --dataset-spec "$spec" \
        --max-m "$max_m" \
        --n-queries "$N_QUERIES" \
        --max-workers "$MAX_WORKERS" \
        --seed "$SEED" \
        --benchmark-algorithms MSINDEX MASS MSINDEXPP \
        $FORCE_FLAG \
        >"$log_file" 2>&1; then
        log "OK $name"
    else
        log "FAIL $name  (see $log_file)"
    fi
}

# --- 30 UEA Multivariate Datasets ---
if [[ "$RUN_MODE" == "all" ]]; then
# spec format: name:path:qLen:defaultN  (maxLen added where series = query length)
run_dataset "ArticularyWordRecognition" "ArticularyWordRecognition:$UEA_BASE_DIR/ArticularyWordRecognition:28:300"    10000
run_dataset "AtrialFibrillation"        "AtrialFibrillation:$UEA_BASE_DIR/AtrialFibrillation:128:15"                10000
run_dataset "BasicMotions"              "BasicMotions:$UEA_BASE_DIR/BasicMotions:20:40"                              10000
run_dataset "CharacterTrajectories"     "CharacterTrajectories:$UEA_BASE_DIR/CharacterTrajectories:20:1436"          10000
run_dataset "Cricket"                   "Cricket:$UEA_BASE_DIR/Cricket:239:108"                                      10000
run_dataset "DuckDuckGeese"             "DuckDuckGeese:$UEA_BASE_DIR/DuckDuckGeese:54:50"                            10000
run_dataset "EigenWorms"                "EigenWorms:$UEA_BASE_DIR/EigenWorms:3596:131:3596"                          10000
run_dataset "Epilepsy"                  "Epilepsy:$UEA_BASE_DIR/Epilepsy:41:138"                                     10000
run_dataset "ERing"                     "ERing:$UEA_BASE_DIR/ERing:20:270"                                           10000
run_dataset "EthanolConcentration"      "EthanolConcentration:$UEA_BASE_DIR/EthanolConcentration:350:263"            10000
run_dataset "FaceDetection"             "FaceDetection:$UEA_BASE_DIR/FaceDetection:20:5890"                          10000
run_dataset "FingerMovements"           "FingerMovements:$UEA_BASE_DIR/FingerMovements:20:316"                       10000
run_dataset "HandMovementDirection"     "HandMovementDirection:$UEA_BASE_DIR/HandMovementDirection:80:160"            10000
run_dataset "Handwriting"               "Handwriting:$UEA_BASE_DIR/Handwriting:30:850"                               10000
run_dataset "Heartbeat"                 "Heartbeat:$UEA_BASE_DIR/Heartbeat:81:205"                                   10000
run_dataset "InsectWingbeat"            "InsectWingbeat:$UEA_BASE_DIR/InsectWingbeat:2:25000"                        10000
run_dataset "JapaneseVowels"            "JapaneseVowels:$UEA_BASE_DIR/JapaneseVowels:7:370"                          10000
run_dataset "Libras"                    "Libras:$UEA_BASE_DIR/Libras:20:180"                                         10000
run_dataset "LSST"                      "LSST:$UEA_BASE_DIR/LSST:20:2466"                                            10000
run_dataset "MotorImagery"              "MotorImagery:$UEA_BASE_DIR/MotorImagery:600:278:600"                         10000
run_dataset "NATOPS"                    "NATOPS:$UEA_BASE_DIR/NATOPS:20:180"                                          10000
run_dataset "PEMS-SF"                   "PEMS-SF:$UEA_BASE_DIR/PEMS-SF:28:267"                                        10000
run_dataset "PenDigits"                 "PenDigits:$UEA_BASE_DIR/PenDigits:8:7494"                                    10000
run_dataset "PhonemeSpectra"            "PhonemeSpectra:$UEA_BASE_DIR/PhonemeSpectra:43:3353"                         10000
run_dataset "RacketSports"              "RacketSports:$UEA_BASE_DIR/RacketSports:20:152"                              10000
run_dataset "SelfRegulationSCP1"        "SelfRegulationSCP1:$UEA_BASE_DIR/SelfRegulationSCP1:179:293"                10000
run_dataset "SelfRegulationSCP2"        "SelfRegulationSCP2:$UEA_BASE_DIR/SelfRegulationSCP2:230:200"                10000
run_dataset "SpokenArabicDigits"        "SpokenArabicDigits:$UEA_BASE_DIR/SpokenArabicDigits:4:6599"                 10000
run_dataset "StandWalkJump"             "StandWalkJump:$UEA_BASE_DIR/StandWalkJump:500:15:500"                        10000
run_dataset "UWaveGestureLibrary"       "UWaveGestureLibrary:$UEA_BASE_DIR/UWaveGestureLibrary:63:320"               10000
fi # RUN_MODE == all

# --- 4 Custom Datasets ---
run_dataset "synthetic" "synthetic:$SYNTHETIC_DIR:1024:100:4096"  4096
run_dataset "weather"   "weather:$WEATHER_DIR:1488:100:8784"       16384
run_dataset "stocks"    "stocks:$STOCKS_DIR:730:100:22335"         16384
run_dataset "wind"      "wind:$WIND_DIR:1800:1:432000"             32768

# --- Training time summary ---
log "Training time per dataset:"
python3 - "$ARTIFACT_ROOT" <<'PYEOF'
import json, sys
from pathlib import Path

artifact_root = Path(sys.argv[1])
rows = []
for d in sorted(artifact_root.iterdir()):
    rt_file = d / "training_runtime.json"
    if rt_file.exists():
        rt = json.loads(rt_file.read_text())
        rows.append((
            d.name,
            rt.get("total_wall_time_s", 0),
            rt.get("collection_wall_time_s", 0),
            rt.get("trainer_wall_time_s", 0),
        ))

if rows:
    print(f"{'Dataset':<30} {'Total (s)':>10} {'Data coll (s)':>14} {'GBDT fit (s)':>13}")
    print("-" * 70)
    for name, total, coll, fit in rows:
        print(f"{name:<30} {total:>10.1f} {coll:>14.1f} {fit:>13.1f}")
PYEOF

log "Done. Summary CSV: $ARTIFACT_ROOT/dataset_summary.csv"
log "Logs: $LOG_DIR"
