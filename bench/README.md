# MARCO Bench Workflow (SAT11)

Standalone benchmark workflow for MARCO paper code (`marco` and `marco_adaptive`) under `MARCO/bench`.

Runner:

- `bench/bench_marco_sat11.py`

## Quick Local Check

```bash
cd /path/to/MARCO

uv run python bench/bench_marco_sat11.py \
  --dataset-root /path/to/SAT11-Competition-MUS-SelectedBenchmarks \
  --marco-root "$PWD" \
  --methods marco,marco_adaptive \
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
  --methods marco,marco_adaptive \
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
chmod +x bench/run_array_task.sh bench/submit_slurm.sh bench/submit_oar.sh
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
- `MAX_OUTPUTS`
- `RUN_VALIDATE` (`0`/`1`, currently no-op in runner)
- `RUN_VERIFY_UNSAT` (`0`/`1`, currently no-op in runner)
- `RUN_VERBOSE` (`0`/`1`)
- `UV_BIN` or `PYTHON_BIN`
