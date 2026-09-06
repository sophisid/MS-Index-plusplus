#!/usr/bin/env bash
# run_suite.sh
#
# Runs training, evaluation, and timing benchmark (MSINDEX, MASS, MSINDEXPP)
# for the selected rerun datasets only. Per-dataset results land in:
#   artifacts/msindexpp_suite/<dataset>/
# Combined summary:
#   artifacts/msindexpp_suite/dataset_summary.csv
# Training time per dataset:
#   artifacts/msindexpp_suite/<dataset>/training_runtime.json
#
# Usage: bash run_suite.sh [--datasets all|custom] [--force]
#   --datasets all|custom  Kept for CLI compatibility; both run the selected
#                          rerun datasets from config.sh.
#   --force            Clears all cached results and reruns from scratch.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load paths and settings from config.sh
# shellcheck source=config.sh
source "$SCRIPT_DIR/config.sh"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SUITE="$SCRIPT_DIR/run_msindexpp_dataset_suite.py"
ARTIFACT_ROOT="$ROOT/artifacts/msindexpp_suite"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.10}"
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
if "$PYTHON_BIN" "$SUITE" \
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

# --- Selected rerun datasets ---
run_dataset "synthetic" "synthetic:$SYNTHETIC_DIR:1024:100:4096"  4096
run_dataset "weather"   "weather:$WEATHER_DIR:1488:100:8784"       16384
run_dataset "stocks"    "stocks:$STOCKS_DIR:730:100:22335"         16384
run_dataset "wind"      "wind:$WIND_DIR:1800:1:432000"             32768
run_dataset "DuckDuckGeese" "DuckDuckGeese:$UEA_BASE_DIR/DuckDuckGeese:54:50:10000" 10000

# --- Training time summary ---
log "Training time per dataset:"
"$PYTHON_BIN" - "$ARTIFACT_ROOT" <<'PYEOF
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
