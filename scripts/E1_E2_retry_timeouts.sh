#!/usr/bin/env bash
# Usage:
#   bash scripts/E1_E2_retry_timeouts.sh              # print sbatch commands (dry run)
#   bash scripts/E1_E2_retry_timeouts.sh | bash        # submit them
#   SUBMIT=1 bash scripts/E1_E2_retry_timeouts.sh      # submit directly
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SUITE="$SCRIPT_DIR/run_msindexpp_dataset_suite.py"
ARTIFACT_ROOT="$ROOT/artifacts/msindexpp_suite"
LOG_DIR="$ARTIFACT_ROOT/retry_timeouts_logs"
mkdir -p "$LOG_DIR"

RETRY_MAX_WORKERS="${RETRY_MAX_WORKERS:-4}"
RETRY_TIME="${RETRY_TIME:-2-00:00:00}"
RETRY_CPUS="${RETRY_CPUS:-4}"
RETRY_MEM="${RETRY_MEM:-32G}"
SUBMIT="${SUBMIT:-0}"

# name:path:qLen:defaultN[:maxLen]  max_m   — same specs as E1_E2_jobs.sh
DATASETS=(
  "weather|weather:$WEATHER_DIR:1488:100:8784|16384"
  "DuckDuckGeese|DuckDuckGeese:$UEA_BASE_DIR/duckduckgeese:54:50|10000"
  "FaceDetection|FaceDetection:$UEA_BASE_DIR/facedetection:20:5890|10000"
  "Heartbeat|Heartbeat:$UEA_BASE_DIR/heartbeat:81:205|10000"
  "InsectWingbeat|InsectWingbeat:$UEA_BASE_DIR/insectwingbeat:2:25000|10000"
  "MotorImagery|MotorImagery:$UEA_BASE_DIR/motorimagery:600:278:600|10000"
  "PEMS-SF|PEMS-SF:$UEA_BASE_DIR/pems-sf:28:267|10000"
  "PhonemeSpectra|PhonemeSpectra:$UEA_BASE_DIR/phonemespectra:43:3353|10000"
  "SelfRegulationSCP1|SelfRegulationSCP1:$UEA_BASE_DIR/selfregulationscp1:179:293|10000"
  "SpokenArabicDigits|SpokenArabicDigits:$UEA_BASE_DIR/spokenarabicdigits:4:6599|10000"
)

quote_join() {
  # Shell-quote each arg and join with spaces -> a single string safe to
  # hand to `sh -c` (used as the --wrap value).
  local out="" a
  for a in "$@"; do
    out+="$(printf '%q' "$a") "
  done
  printf '%s' "${out% }"
}

emit_job() {
  local job_name="$1"
  local extra_args_str="$2"
  local spec="$3"
  local max_m="$4"
  local out_log="$LOG_DIR/${job_name}.out"
  local -a extra_args=($extra_args_str)

  local -a inner_cmd=(
    /sw/arch/RHEL9/EB_production/2025/software/Python/3.13.5-GCCcore-14.3.0/bin/python "$SUITE"
    --artifact-root "$ARTIFACT_ROOT"
    --dataset-spec "$spec"
    --max-m "$max_m"
    --n-queries "$N_QUERIES"
    --max-workers "$RETRY_MAX_WORKERS"
    --seed "$SEED"
    "${extra_args[@]}"
  )
  local wrap_value
  wrap_value="$(quote_join "${inner_cmd[@]}")"

  local -a sbatch_args=(
    sbatch
    --job-name="$job_name"
    --time="$RETRY_TIME"
    --cpus-per-task="$RETRY_CPUS"
    --mem="$RETRY_MEM"
    --output="$out_log"
    --wrap="$wrap_value"
  )

  if [[ "$SUBMIT" == "1" ]]; then
    "${sbatch_args[@]}"
  else
    printf '%q ' "${sbatch_args[@]}"
    printf '\n'
  fi
}

for entry in "${DATASETS[@]}"; do
  IFS='|' read -r name spec max_m <<< "$entry"

  emit_job "${name}_msindexpp" "--benchmark-algorithms MSINDEXPP" "$spec" "$max_m"
  emit_job "${name}_msindex"   "--skip-train --skip-eval --benchmark-algorithms MSINDEX" "$spec" "$max_m"
  emit_job "${name}_mass"      "--skip-train --skip-eval --benchmark-algorithms MASS" "$spec" "$max_m"
done
