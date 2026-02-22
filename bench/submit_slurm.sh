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

ARRAY_MAX_PARALLEL="${ARRAY_MAX_PARALLEL:-64}"
CPUS_PER_TASK="${CPUS_PER_TASK:-1}"
MEM_PER_TASK="${MEM_PER_TASK:-8G}"
WALLTIME="${WALLTIME:-02:00:00}"
JOB_NAME="${JOB_NAME:-marco-paper-sat11}"

RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"
mkdir -p "$RESULTS_DIR/logs"
LOG_PATTERN="${LOG_PATTERN:-$RESULTS_DIR/logs/%A_%a.out}"

echo "Submitting Slurm array"
echo "  params-file   : $PARAMS_FILE"
echo "  total rows    : $N_ROWS"
echo "  array limit   : $ARRAY_MAX_PARALLEL"
echo "  resources     : cpu=$CPUS_PER_TASK mem=$MEM_PER_TASK time=$WALLTIME"

sbatch \
  --job-name="$JOB_NAME" \
  --array="1-${N_ROWS}%$ARRAY_MAX_PARALLEL" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --mem="$MEM_PER_TASK" \
  --time="$WALLTIME" \
  --output="$LOG_PATTERN" \
  --export=ALL,MARCO_ROOT="$MARCO_ROOT",REPO_ROOT="$REPO_ROOT",RESULTS_DIR="$RESULTS_DIR" \
  "$SCRIPT_DIR/run_array_task.sh" \
  "$PARAMS_FILE" \
  "$@"
