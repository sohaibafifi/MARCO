#!/usr/bin/env bash
set -euo pipefail

# Optional cluster bootstrap.
if [[ -f "/nfs/opt/env/env.sh" ]]; then
  # shellcheck disable=SC1091
  set +u
  . /nfs/opt/env/env.sh
  set -u
fi
if command -v module >/dev/null 2>&1 && [[ -n "${MODULESHOME:-}" ]]; then
  set +u
  module load conda >/dev/null 2>&1 || true
  set -u
fi
if command -v conda >/dev/null 2>&1; then
  conda activate csp >/dev/null 2>&1 || true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MARCO_ROOT="${MARCO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
REPO_ROOT="${REPO_ROOT:-$MARCO_ROOT}"
DATASET_ROOT="${DATASET_ROOT:-}"
if [[ -z "$DATASET_ROOT" ]]; then
  for cand in \
    "$MARCO_ROOT/SAT11-Competition-MUS-SelectedBenchmarks" \
    "$MARCO_ROOT/../SAT11-Competition-MUS-SelectedBenchmarks" \
    "$PWD/SAT11-Competition-MUS-SelectedBenchmarks" \
    "$PWD/benchmarks/SAT11-Competition-MUS-SelectedBenchmarks"; do
    if [[ -d "$cand" ]]; then
      DATASET_ROOT="$cand"
      break
    fi
  done
fi
DATASET_ROOT="${DATASET_ROOT:-$MARCO_ROOT/SAT11-Competition-MUS-SelectedBenchmarks}"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"

TIMEOUT_S="${TIMEOUT_S:-3600}"
MAX_OUTPUTS="${MAX_OUTPUTS:-0}"
RUN_VALIDATE="${RUN_VALIDATE:-0}"
RUN_VERIFY_UNSAT="${RUN_VERIFY_UNSAT:-0}"
RUN_VERBOSE="${RUN_VERBOSE:-0}"

THREADS="${THREADS:-1}"
MUSER_BIN="${MUSER_BIN:-}"
FORCE_MINISAT="${FORCE_MINISAT:-0}"
NO_FEEDBACK="${NO_FEEDBACK:-0}"
CORE_HANDOFF="${CORE_HANDOFF:--1}"
CORE_BASE_RATIO="${CORE_BASE_RATIO:-2}"
CORE_BACKOFF_CAP="${CORE_BACKOFF_CAP:-8}"
CORE_NO_CERTIFY="${CORE_NO_CERTIFY:-0}"
PORTFOLIO_SMART_AFTER_MUS="${PORTFOLIO_SMART_AFTER_MUS:-1}"
PORTFOLIO_SMART_AFTER_OUTPUTS="${PORTFOLIO_SMART_AFTER_OUTPUTS:-0}"

UV_BIN="${UV_BIN:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-${OAR_ARRAY_INDEX:-${OAR_JOBARRAY_INDEX:-${TASK_ID:-}}}}"
TASK_LABEL="${TASK_ID:-param}"

instance_rel=""
method=""
rep_id=""
param_threads=""
param_muser_bin=""
param_force_minisat=""

# Mode A: first argument is a params file (line selected from task index env).
if [[ $# -ge 1 && -f "${1}" ]]; then
  PARAMS_FILE="${1}"
  if [[ -z "${TASK_ID}" ]]; then
    echo "No task index found. Provide SLURM_ARRAY_TASK_ID/OAR_ARRAY_INDEX/OAR_JOBARRAY_INDEX/TASK_ID."
    exit 1
  fi

  line="$(sed -n "${TASK_ID}p" "${PARAMS_FILE}")"
  if [[ -z "${line}" ]]; then
    echo "No params row for task index ${TASK_ID} in ${PARAMS_FILE}"
    exit 1
  fi

  # Parse tab fields explicitly to preserve empty columns.
  instance_rel="$(printf '%s\n' "$line" | awk -F $'\t' '{print $1}')"
  method="$(printf '%s\n' "$line" | awk -F $'\t' '{print $2}')"
  rep_id="$(printf '%s\n' "$line" | awk -F $'\t' '{print $3}')"
  param_threads="$(printf '%s\n' "$line" | awk -F $'\t' '{print $4}')"
  param_muser_bin="$(printf '%s\n' "$line" | awk -F $'\t' '{print $5}')"
  param_force_minisat="$(printf '%s\n' "$line" | awk -F $'\t' '{print $6}')"
  if [[ -z "${instance_rel}" || -z "${method}" || -z "${rep_id}" ]]; then
    read -r instance_rel method rep_id param_threads param_muser_bin param_force_minisat _ <<< "${line}"
  fi
else
  # Mode B: direct row args (for OAR --array-param-file):
  #   run_array_task.sh <instance_rel> <method> <rep_id> [threads] [muser_bin] [force_minisat]
  instance_rel="${1:-}"
  method="${2:-}"
  rep_id="${3:-}"
  param_threads="${4:-}"
  param_muser_bin="${5:-}"
  param_force_minisat="${6:-}"
fi

if [[ -z "${instance_rel}" || -z "${method}" || -z "${rep_id}" ]]; then
  echo "Invalid task parameters. Need: <instance_rel> <method> <rep_id>."
  echo "Either call with a params file + task index env, or pass direct args."
  exit 1
fi

if [[ -n "${param_threads}" ]]; then
  THREADS="${param_threads}"
fi
if [[ -n "${param_muser_bin}" && "${param_muser_bin}" != "-" ]]; then
  MUSER_BIN="${param_muser_bin}"
fi
if [[ -n "${param_force_minisat}" ]]; then
  FORCE_MINISAT="${param_force_minisat}"
fi

if [[ "${THREADS}" -lt 1 ]]; then
  echo "THREADS must be >= 1 (got ${THREADS})"
  exit 1
fi
case "${FORCE_MINISAT}" in
  0|1) ;;
  *)
    echo "FORCE_MINISAT must be 0 or 1 (got ${FORCE_MINISAT})"
    exit 1
    ;;
esac
case "${NO_FEEDBACK}" in
  0|1) ;;
  *)
    echo "NO_FEEDBACK must be 0 or 1 (got ${NO_FEEDBACK})"
    exit 1
    ;;
esac
case "${CORE_NO_CERTIFY}" in
  0|1) ;;
  *)
    echo "CORE_NO_CERTIFY must be 0 or 1 (got ${CORE_NO_CERTIFY})"
    exit 1
    ;;
esac

BASELINE="${method}"

mkdir -p "$RESULTS_DIR/runs" "$RESULTS_DIR/logs"

if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv/bin/activate"
elif [[ -f "$MARCO_ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$MARCO_ROOT/.venv/bin/activate"
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ -z "${UV_BIN}" ]] && command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
fi
if [[ -n "${UV_BIN}" ]]; then
  PY_CMD=("${UV_BIN}" run python)
else
  if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
      PYTHON_BIN="python3"
    elif command -v python >/dev/null 2>&1; then
      PYTHON_BIN="python"
    else
      echo "No Python executable found (tried PYTHON_BIN, python3, python)."
      exit 1
    fi
  fi
  PY_CMD=("${PYTHON_BIN}")
fi

JOB_LABEL="${SLURM_ARRAY_JOB_ID:-${OAR_JOB_ID:-local}}"
instance_tag="$(printf "%s" "$instance_rel" | tr '/.' '__' | tr -cd '[:alnum:]_-' | cut -c1-80)"
CSV_OUT="$RESULTS_DIR/runs/${method}__rep${rep_id}__${instance_tag}__job${JOB_LABEL}__task${TASK_LABEL}.csv"

CMD=(
  "${PY_CMD[@]}"
  "$SCRIPT_DIR/bench_marco_sat11.py"
  --dataset-root "$DATASET_ROOT"
  --marco-root "$MARCO_ROOT"
  --methods "$method"
  --baseline "$BASELINE"
  --instances "$instance_rel"
  --max-vars 0
  --max-clauses 0
  --max-files 1
  --repeats 1
  --warmup 0
  --timeout-s "$TIMEOUT_S"
  --max-outputs "$MAX_OUTPUTS"
  --threads "$THREADS"
  --core-handoff "$CORE_HANDOFF"
  --core-base-ratio "$CORE_BASE_RATIO"
  --core-backoff-cap "$CORE_BACKOFF_CAP"
  --portfolio-smart-after-mus "$PORTFOLIO_SMART_AFTER_MUS"
  --portfolio-smart-after-outputs "$PORTFOLIO_SMART_AFTER_OUTPUTS"
  --output-csv "$CSV_OUT"
)
if [[ -n "${MUSER_BIN}" ]]; then
  CMD+=(--muser-bin "$MUSER_BIN")
fi
if [[ "${FORCE_MINISAT}" == "1" ]]; then
  CMD+=(--force-minisat)
fi
if [[ "${NO_FEEDBACK}" == "1" ]]; then
  CMD+=(--no-feedback)
fi
if [[ "${CORE_NO_CERTIFY}" == "1" ]]; then
  CMD+=(--core-no-certify)
fi
if [[ "${RUN_VALIDATE}" == "1" ]]; then
  CMD+=(--validate)
fi
if [[ "${RUN_VERIFY_UNSAT}" == "1" ]]; then
  CMD+=(--verify-unsat)
fi
if [[ "${RUN_VERBOSE}" == "1" ]]; then
  CMD+=(--verbose)
fi

echo "[task] id=${TASK_LABEL} method=${method} rep=${rep_id}"
echo "[task] instance=${instance_rel}"
echo "[task] threads=${THREADS} force_minisat=${FORCE_MINISAT}"
echo "[task] no_feedback=${NO_FEEDBACK} core_handoff=${CORE_HANDOFF} core_ratio=${CORE_BASE_RATIO} core_backoff=${CORE_BACKOFF_CAP} core_no_certify=${CORE_NO_CERTIFY} portfolio_mus=${PORTFOLIO_SMART_AFTER_MUS} portfolio_out=${PORTFOLIO_SMART_AFTER_OUTPUTS}"
if [[ -n "${MUSER_BIN}" ]]; then
  echo "[task] muser_bin=${MUSER_BIN}"
fi
echo "[task] output=${CSV_OUT}"
printf '[task] command: '
printf '%q ' "${CMD[@]}"
printf '\n'

cd "$MARCO_ROOT"
"${CMD[@]}"
