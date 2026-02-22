#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MARCO_ROOT="${MARCO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
REPO_ROOT="${REPO_ROOT:-$MARCO_ROOT}"

PARAMS_FILE="${PARAMS_FILE:-$SCRIPT_DIR/array_params.tsv}"
if [[ ! -f "$PARAMS_FILE" ]]; then
  echo "Array params file not found: $PARAMS_FILE"
  echo "Build it first with:"
  echo "  python3 $SCRIPT_DIR/build_manifest.py ..."
  echo "  python3 $SCRIPT_DIR/build_array_params.py ..."
  exit 1
fi

N_ROWS="$(awk 'NF{c++} END{print c+0}' "$PARAMS_FILE")"
if (( N_ROWS <= 0 )); then
  echo "No jobs to submit (rows=$N_ROWS)"
  exit 1
fi

RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"
mkdir -p "$RESULTS_DIR/logs"

CORES="${CORES:-1}"
WALLTIME="${WALLTIME:-2:00:00}"
JOB_NAME="${JOB_NAME:-marco-paper-sat11}"
USE_LOOP="${USE_LOOP:-0}"

echo "Submitting OAR jobs"
echo "  params-file   : $PARAMS_FILE"
echo "  total rows    : $N_ROWS"
echo "  resources     : core=$CORES walltime=$WALLTIME"

COMMON_ENV="MARCO_ROOT=$MARCO_ROOT REPO_ROOT=$REPO_ROOT RESULTS_DIR=$RESULTS_DIR"

if [[ "$USE_LOOP" == "1" ]]; then
  echo "Using loop submission mode (params file + task index)"
  for i in $(seq 1 "${N_ROWS}"); do
    oarsub \
      -n "$JOB_NAME" \
      -l "/nodes=1/core=${CORES},walltime=${WALLTIME}" \
      "export $COMMON_ENV TASK_ID=$i; bash $SCRIPT_DIR/run_array_task.sh $PARAMS_FILE" \
      "$@"
  done
else
  echo "Using OAR array-param-file mode"
  oarsub \
    -n "$JOB_NAME" \
    --array "${N_ROWS}" \
    --array-param-file "$PARAMS_FILE" \
    -l "/nodes=1/core=${CORES},walltime=${WALLTIME}" \
    "export $COMMON_ENV; bash $SCRIPT_DIR/run_array_task.sh" \
    "$@"
fi
