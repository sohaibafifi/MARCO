# MARCO Bench Workflow (SAT11)

Standalone benchmark workflow for MARCO code under `MARCO/bench`.

Runner:

- `bench/bench_marco_sat11.py`

Available benchmark methods:

- `marco`: paper default MARCO (`marco.py` default settings)
- `marco_basic`: MARCO without maximization (`marco.py --nomax`)
- `marco_plus`: MARCO+ style (`marco.py --improved-implies`)
- `dual_hs`: non-MARCO implicit hitting-set prototype (`dual_hs.py`)
- `marco_hybrid`: MARCO+ with SAT map-assisted pre-expansion + adaptive MUSer handoff shrink (`marco_hybrid.py`)
- `marco_adaptive`: adaptive variant (`marco_adaptive.py`)
- `marco_smart`: adaptive + smart-core shrink (`marco_smart.py`, enables `--adaptive --smart-core`)
- `marco_portfolio`: adaptive + smart-core with delayed smart activation (`marco_portfolio.py`)

## Generate AAAI'25 SymmetryMUS Dataset (272 CNF Instances)

Generate the synthetic benchmark described in the paper appendix:

- 146 pigeon-hole instances
- 66 n+k-queens instances
- 60 bin-packing instances

The generator writes `.cnf.bz2` directly so it is compatible with the MARCO bench workflow.

```bash
cd /path/to/MARCO

uv run python bench/generate_symmetrymus_aaai25.py \
  --output-root "$PWD/SymmetryMUS-AAAI25-Benchmarks" \
  --manifest-output "$PWD/bench/symmetrymus_aaai25_manifest.tsv" \
  --bin-opt-timeout-s 20
```

Then benchmark on it:

```bash
uv run python bench/bench_marco_sat11.py \
  --dataset-root "$PWD/SymmetryMUS-AAAI25-Benchmarks" \
  --marco-root "$PWD" \
  --methods marco,marco_plus,marco_hybrid,marco_adaptive,marco_smart,marco_portfolio \
  --baseline marco \
  --max-files 272 \
  --repeats 1 \
  --warmup 0 \
  --timeout-s 3600 \
  --verbose
```

Notes:

- The bin-packing default range is `5..24` (not `5..25`) to match the reported total of `60` bin-packing instances.
- Add `--verify-unsat` to `generate_symmetrymus_aaai25.py` if you want SAT-check validation during generation.

## Profiling MARCO+ Bottlenecks

To collect phase timings (`seed/check/shrink/grow/block/hubcomms/...`) per run:

```bash
uv run python bench/bench_marco_sat11.py \
  --dataset-root /path/to/SAT11-Competition-MUS-SelectedBenchmarks \
  --marco-root "$PWD" \
  --methods marco_plus \
  --baseline marco_plus \
  --max-files 10 \
  --repeats 1 \
  --timeout-s 1200 \
  --profile-stats \
  --output-csv bench/results/marco_plus_profile.csv
```

The run CSV will include `phase_*_s` columns and the console output prints a median phase-share summary per method.

## Quick Local Check

```bash
cd /path/to/MARCO

uv run python bench/bench_marco_sat11.py \
  --dataset-root /path/to/SAT11-Competition-MUS-SelectedBenchmarks \
  --marco-root "$PWD" \
  --methods marco_adaptive,marco_smart,marco_portfolio \
  --baseline marco_adaptive \
  --max-files 1 \
  --repeats 1 \
  --warmup 0 \
  --timeout-s 3600 \
  --verbose
```

## 1) Build Manifest

```bash
cd /path/to/MARCO

python3 bench/build_manifest.py \
  --dataset-root /path/to/SAT11-Competition-MUS-SelectedBenchmarks \
  --output bench/sat11_manifest.tsv \
  --max-vars 10000 \
  --max-clauses 100000 \
  --max-files 300
```

## 2) Build Array Params

```bash
python3 bench/build_array_params.py \
  --manifest bench/sat11_manifest.tsv \
  --methods marco,marco_plus,marco_hybrid,marco_adaptive,marco_smart,marco_portfolio \
  --repeats 1 \
  --threads 1 \
  --output bench/array_params.tsv
```

Default row format:

`instance_rel<TAB>method<TAB>repeat_id`

Extended row format (optional):

`instance_rel<TAB>method<TAB>repeat_id<TAB>threads<TAB>muser_bin_or_dash<TAB>force_minisat`

## 3) Submit Array Jobs

```bash
chmod +x bench/run_array_task.sh bench/run_array_group_task.sh bench/submit_slurm.sh bench/submit_oar.sh
N=$(wc -l < bench/array_params.tsv)
```

### Slurm

```bash
sbatch \
  --array=1-"${N}" \
  --export=ALL,MARCO_ROOT="$PWD",RESULTS_DIR="$PWD/bench/results",TIMEOUT_S=3600 \
  bench/run_array_task.sh \
  "$PWD/bench/array_params.tsv"
```

Or helper launcher:

```bash
PARAMS_FILE=bench/array_params.tsv \
RESULTS_DIR="$PWD/bench/results" \
TIMEOUT_S=3600 \
bench/submit_slurm.sh
```

### OAR (`--array-param-file`)

```bash
oarsub \
  --array "${N}" \
  --array-param-file "$PWD/bench/array_params.tsv" \
  -l /nodes=1/core=1,walltime=10:10:00 \
  "MARCO_ROOT=$PWD TIMEOUT_S=3600 RESULTS_DIR=$PWD/bench/results bash $PWD/bench/run_array_task.sh"
```

With explicit MUSer path:

```bash
oarsub \
  --array "${N}" \
  --array-param-file "$PWD/bench/array_params.tsv" \
  -l /nodes=1/core=1,walltime=10:10:00 \
  "MARCO_ROOT=$PWD MUSER2_PATH=muser2-para/src/tools/muser-2/muser-2 TIMEOUT_S=3600 RESULTS_DIR=$PWD/bench/results bash $PWD/bench/run_array_task.sh"
```

For this launcher, `MUSER_BIN` is preferred (it is passed as `--muser-bin` to `bench/bench_marco_sat11.py`), while `MUSER2_PATH` remains a valid MARCO fallback.

Or helper launcher:

```bash
PARAMS_FILE=bench/array_params.tsv \
RESULTS_DIR="$PWD/bench/results" \
TIMEOUT_S=3600 \
bench/submit_oar.sh
```

### OAR Grouped Mode (Recommended for Large Arrays)

Avoid submitting one OAR job per TSV row by batching rows into chunks and running each chunk with local parallel workers.

```bash
PARAMS_FILE=bench/array_params.tsv \
RESULTS_DIR="$PWD/bench/results" \
TIMEOUT_S=3600 \
GROUP_SIZE=20 \
GROUP_PARALLEL=4 \
CORES=4 \
bench/submit_oar.sh
```

Behavior:

- `GROUP_SIZE`: number of TSV rows handled by one OAR array job
- `GROUP_PARALLEL`: max concurrent row workers inside each array job (uses GNU `parallel` when available, otherwise built-in fallback workers)
- OAR array size becomes `ceil(total_rows / GROUP_SIZE)` instead of `total_rows`

## 4) Collect Results

```bash
python3 bench/collect_results.py \
  --runs-dir bench/results/runs \
  --output-all bench/results/all_runs.csv \
  --output-summary bench/results/all_runs_summary.csv
```

## Runtime Environment Variables

`bench/run_array_task.sh` supports:

- `TIMEOUT_S` (default `3600`)
- `RESULTS_DIR`
- `DATASET_ROOT`
- `MARCO_ROOT`
- `THREADS` (default `1`)
- `MUSER_BIN` (optional)
- `FORCE_MINISAT` (`0`/`1`)
- `NO_FEEDBACK` (`0`/`1`, disables adaptive/smart feedback clauses)
- `CORE_HANDOFF` (default `-1`, auto threshold for `marco_smart`)
- `CORE_BASE_RATIO` (default `2`, `marco_smart`)
- `CORE_BACKOFF_CAP` (default `8`, `marco_smart`)
- `CORE_NO_CERTIFY` (`0`/`1`, disables final smart-core certification)
- `PORTFOLIO_SMART_AFTER_MUS` (default `1`, MUS outputs before enabling smart-core in `marco_portfolio`)
- `PORTFOLIO_SMART_AFTER_OUTPUTS` (default `0`, total outputs threshold before enabling smart-core in `marco_portfolio`)
- `SAT_MAP_ASSIST_MIN_GAP` (default `32`, `marco_hybrid` SAT map-assist trigger threshold)
- `HYBRID_SHRINK_HANDOFF_SIZE` (default `256`, `marco_hybrid` MUSer handoff threshold on large cores)
- `HYBRID_SHRINK_HANDOFF_FLOOR` (default `64`, `marco_hybrid` minimum core size for stagnation-based handoff)
- `HYBRID_SHRINK_STAGNATION` (default `64`, `marco_hybrid` failed-deletion streak before handoff)
- `DUALHS_SOLVER` (`implies|hybrid|muser`, configure `dual_hs` backend; default runner arg is `implies`)
- `DUALHS_MAP_MASTER` (`auto|minisat|minicard`, configure `dual_hs` map backend; default `auto`)
- `DUALHS_MUS_QUOTA_EVERY` (default `0`, `dual_hs` MUS-quota period; when `>0`, dual_hs prefers MUS-biased seeds after `N` non-MUS outputs)
- `MAX_OUTPUTS`
- `RUN_VALIDATE` (`0`/`1`, enables cross-method output-signature validation vs baseline in runner)
- `RUN_VERIFY_UNSAT` (`0`/`1`, currently no-op in runner)
- `RUN_VERBOSE` (`0`/`1`)
- `RUN_PROFILE_STATS` (`0`/`1`, enable `--profile-stats` and export phase timings)
- `UV_BIN` or `PYTHON_BIN`
- `GROUP_SIZE` (default `1` in `submit_oar.sh`; set `>1` to enable grouped OAR submission)
- `GROUP_PARALLEL` (default `CORES`; worker concurrency per grouped OAR job)

If `FORCE_MINISAT=0`, MARCO needs a working MUSer2 binary.
Set `MUSER_BIN=/absolute/path/to/muser-2` (preferred) or `MUSER2_PATH=...`.
