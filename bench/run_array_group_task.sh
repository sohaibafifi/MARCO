#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PARAMS_FILE="${PARAMS_FILE:-$SCRIPT_DIR/array_params.tsv}"
if [[ ! -f "$PARAMS_FILE" ]]; then
  echo "Params file not found: $PARAMS_FILE"
  exit 1
fi

GROUP_SIZE="${GROUP_SIZE:-20}"
GROUP_PARALLEL="${GROUP_PARALLEL:-1}"
if [[ "$GROUP_SIZE" -lt 1 ]]; then
  echo "GROUP_SIZE must be >= 1 (got $GROUP_SIZE)"
  exit 1
fi
if [[ "$GROUP_PARALLEL" -lt 1 ]]; then
  echo "GROUP_PARALLEL must be >= 1 (got $GROUP_PARALLEL)"
  exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-${OAR_ARRAY_INDEX:-${OAR_JOBARRAY_INDEX:-${TASK_ID:-}}}}"
if [[ -z "${TASK_ID}" ]]; then
  echo "No task index found. Provide SLURM_ARRAY_TASK_ID/OAR_ARRAY_INDEX/OAR_JOBARRAY_INDEX/TASK_ID."
  exit 1
fi

N_ROWS="$(awk 'NF{c++} END{print c+0}' "$PARAMS_FILE")"
if (( N_ROWS <= 0 )); then
  echo "No jobs to run (rows=$N_ROWS)"
  exit 0
fi

START=$(( (TASK_ID - 1) * GROUP_SIZE + 1 ))
if (( START > N_ROWS )); then
  echo "No rows for task $TASK_ID (start=$START > total=$N_ROWS)"
  exit 0
fi
END=$(( START + GROUP_SIZE - 1 ))
if (( END > N_ROWS )); then
  END=$N_ROWS
fi

TMP_CHUNK="$(mktemp "${TMPDIR:-/tmp}/marco_chunk_${TASK_ID}_XXXX.tsv")"
trap 'rm -f "$TMP_CHUNK"' EXIT
sed -n "${START},${END}p" "$PARAMS_FILE" > "$TMP_CHUNK"

N_CHUNK="$(awk 'NF{c++} END{print c+0}' "$TMP_CHUNK")"
echo "[chunk] task=${TASK_ID} rows=${START}-${END} count=${N_CHUNK} group_size=${GROUP_SIZE} parallel=${GROUP_PARALLEL}"

run_line() {
  local line="$1"
  local instance_rel method rep_id param_threads param_muser_bin param_force_minisat
  IFS=$'\t' read -r instance_rel method rep_id param_threads param_muser_bin param_force_minisat _ <<< "$line"
  if [[ -z "${instance_rel}" || -z "${method}" || -z "${rep_id}" ]]; then
    echo "[chunk] skip malformed row: $line"
    return 0
  fi
  bash "$SCRIPT_DIR/run_array_task.sh" \
    "$instance_rel" "$method" "$rep_id" \
    "${param_threads:-}" "${param_muser_bin:-}" "${param_force_minisat:-}"
}

if (( GROUP_PARALLEL == 1 )); then
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" ]] && continue
    run_line "$line"
  done < "$TMP_CHUNK"
  exit 0
fi

if command -v parallel >/dev/null 2>&1; then
  parallel --jobs "$GROUP_PARALLEL" \
    --line-buffer \
    --halt now,fail=1 \
    --colsep '\t' \
    "bash '$SCRIPT_DIR/run_array_task.sh' {1} {2} {3} {4} {5} {6}" \
    :::: "$TMP_CHUNK"
  exit 0
fi

# Fallback when GNU parallel is unavailable: portable bounded background workers.
overall_rc=0
pids=()
while IFS= read -r line || [[ -n "${line}" ]]; do
  [[ -z "${line}" ]] && continue
  run_line "$line" &
  pids+=("$!")

  while (( ${#pids[@]} >= GROUP_PARALLEL )); do
    next=()
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        next+=("$pid")
      else
        wait "$pid" || overall_rc=1
      fi
    done
    pids=("${next[@]}")
    if (( ${#pids[@]} >= GROUP_PARALLEL )); then
      sleep 0.2
    fi
  done
done < "$TMP_CHUNK"

for pid in "${pids[@]}"; do
  wait "$pid" || overall_rc=1
done

exit "$overall_rc"
