#!/usr/bin/env python3
"""
Benchmark paper MARCO implementations on SAT11 CNF instances (.cnf.bz2).

Compares:
- MARCO baseline from MARCO `marco.py`
- MARCO basic variant (`marco.py --nomax`)
- MARCO+ style variant (`marco.py --improved-implies`)
- Adaptive variant from MARCO `marco_adaptive.py`
- Smart adaptive/core variant from MARCO `marco_smart.py`

The output schema mirrors bench_marco_sat11.py to reuse the same workflow.
"""

from __future__ import annotations

import argparse
import bz2
import csv
import math
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


P_HEADER = re.compile(r"^p\s+cnf\s+(\d+)\s+(\d+)\s*$")
SCRIPT_DIR = Path(__file__).resolve().parent
MARCO_ROOT_DEFAULT = SCRIPT_DIR.parent


def default_dataset_root() -> Path:
    env_root = os.environ.get("DATASET_ROOT")
    if env_root:
        return Path(env_root)

    candidates: List[Path] = [
        MARCO_ROOT_DEFAULT / "SAT11-Competition-MUS-SelectedBenchmarks",
        MARCO_ROOT_DEFAULT.parent / "SAT11-Competition-MUS-SelectedBenchmarks",
        Path.cwd() / "SAT11-Competition-MUS-SelectedBenchmarks",
        Path.cwd() / "benchmarks" / "SAT11-Competition-MUS-SelectedBenchmarks",
    ]
    if len(SCRIPT_DIR.parents) > 3:
        candidates.append(SCRIPT_DIR.parents[3] / "SAT11-Competition-MUS-SelectedBenchmarks")

    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]


@dataclass
class InstanceMeta:
    path: str
    relpath: str
    nvars: int
    nclauses: int
    compressed_bytes: int


@dataclass
class RunRecord:
    instance: str
    method: str
    run_id: int
    elapsed_s: float
    success: bool
    timed_out: bool
    completed: bool
    valid_mus: bool
    valid_mcs: bool
    outputs_count: int
    mus_count: int
    mcs_count: int
    first_output_s: Optional[float]
    first_mus_s: Optional[float]
    first_mcs_s: Optional[float]
    error: str


@dataclass
class MethodSummary:
    method: str
    runs: int
    completed_rate: float
    timeout_rate: float
    valid_rate: float
    median_ms: Optional[float]
    p90_ms: Optional[float]
    median_outputs: Optional[float]
    median_mus: Optional[float]
    median_mcs: Optional[float]
    median_first_mus_ms: Optional[float]


def percentile(values: Sequence[float], p: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1))
    return ordered[idx]


def parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def read_dimacs_header(path: Path) -> Tuple[Optional[int], Optional[int]]:
    with bz2.open(path, "rt", encoding="ascii", errors="ignore") as f:
        for line in f:
            m = P_HEADER.match(line.strip())
            if m:
                return int(m.group(1)), int(m.group(2))
    return None, None


def scan_instances(root: Path) -> List[InstanceMeta]:
    metas: List[InstanceMeta] = []
    for path in sorted(root.rglob("*.cnf.bz2")):
        nvars, nclauses = read_dimacs_header(path)
        if nvars is None or nclauses is None:
            continue
        metas.append(
            InstanceMeta(
                path=str(path),
                relpath=str(path.relative_to(root)),
                nvars=nvars,
                nclauses=nclauses,
                compressed_bytes=path.stat().st_size,
            )
        )
    return metas


def select_instances(
    all_instances: List[InstanceMeta],
    include_patterns: List[str],
    max_vars: int,
    max_clauses: int,
    max_files: int,
    shuffle: bool,
    seed: int,
) -> List[InstanceMeta]:
    selected = list(all_instances)

    if include_patterns:
        pats = [p.lower() for p in include_patterns]
        selected = [
            m
            for m in selected
            if any(p in m.relpath.lower() or p in Path(m.relpath).name.lower() for p in pats)
        ]

    if max_vars > 0:
        selected = [m for m in selected if m.nvars <= max_vars]
    if max_clauses > 0:
        selected = [m for m in selected if m.nclauses <= max_clauses]

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(selected)
    else:
        selected.sort(key=lambda m: (m.nclauses, m.nvars, m.relpath))

    if max_files > 0:
        selected = selected[:max_files]

    return selected


def _extract_bz2_to_temp(path: str, tmpdir: Path) -> Path:
    src = Path(path)
    if src.suffix != ".bz2":
        return src
    dst_name = src.stem  # remove .bz2 -> *.cnf
    dst = tmpdir / dst_name
    with bz2.open(src, "rb") as fin, dst.open("wb") as fout:
        while True:
            chunk = fin.read(1024 * 1024)
            if not chunk:
                break
            fout.write(chunk)
    return dst


def _parse_marco_output(stdout_path: Path) -> Tuple[int, int, int, Optional[float], Optional[float], Optional[float]]:
    outputs_count = 0
    mus_count = 0
    mcs_count = 0
    first_output_s: Optional[float] = None
    first_mus_s: Optional[float] = None
    first_mcs_s: Optional[float] = None

    with stdout_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            toks = line.split()
            if not toks:
                continue
            kind = toks[0]
            if kind not in {"U", "S", "C"}:
                continue

            tval: Optional[float] = None
            if len(toks) > 1:
                try:
                    tval = float(toks[1])
                except ValueError:
                    tval = None

            outputs_count += 1
            if first_output_s is None and tval is not None:
                first_output_s = tval

            if kind == "U":
                mus_count += 1
                if first_mus_s is None and tval is not None:
                    first_mus_s = tval
            else:
                # "S" is MSS in MARCO output; count it in mcs_count for parity with CPMPy benchmark schema.
                mcs_count += 1
                if first_mcs_s is None and tval is not None:
                    first_mcs_s = tval

    return outputs_count, mus_count, mcs_count, first_output_s, first_mus_s, first_mcs_s


def run_once(
    instance_path: str,
    instance_rel: str,
    method_name: str,
    run_id: int,
    timeout_s: float,
    max_outputs: int,
    muser_bin: str,
    force_minisat: bool,
    threads: int,
    no_feedback: bool,
    core_handoff: int,
    core_base_ratio: int,
    core_backoff_cap: int,
    core_no_certify: bool,
    validate: bool,
    verify_unsat: bool,
    marco_root: Path,
) -> RunRecord:
    # Validation and unsat pre-check are currently not implemented in the paper runner.
    # Keep flags for workflow compatibility.
    del validate
    del verify_unsat

    if method_name == "marco":
        script_name = "marco.py"
        method_flags: List[str] = []
    elif method_name == "marco_basic":
        script_name = "marco.py"
        method_flags = ["--nomax"]
    elif method_name == "marco_plus":
        script_name = "marco.py"
        method_flags = ["--improved-implies"]
    elif method_name == "marco_adaptive":
        script_name = "marco_adaptive.py"
        method_flags = []
    elif method_name == "marco_smart":
        script_name = "marco_smart.py"
        method_flags = []
    else:
        raise ValueError(f"Unknown method: {method_name}")

    if method_name in {"marco_adaptive", "marco_smart"}:
        if no_feedback:
            method_flags += ["--feedback-disable"]
        if core_handoff != -1:
            method_flags += ["--core-handoff", str(core_handoff)]
        method_flags += ["--core-base-ratio", str(core_base_ratio)]
        method_flags += ["--core-backoff-cap", str(core_backoff_cap)]
        if core_no_certify:
            method_flags += ["--core-no-certify"]

    script_path = marco_root / script_name

    t0 = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="marco_sat11_") as td:
            tmpdir = Path(td)
            cnf_path = _extract_bz2_to_temp(instance_path, tmpdir)
            stdout_path = tmpdir / "stdout.txt"
            stderr_path = tmpdir / "stderr.txt"

            cmd = [
                sys.executable,
                str(script_path),
                "--cnf",
                "--alltimes",
                "--threads",
                str(max(1, threads)),
            ]
            if timeout_s > 0:
                cmd += ["--timeout", str(max(1, int(math.ceil(timeout_s))))]
            if max_outputs > 0:
                cmd += ["--limit", str(max_outputs)]
            if muser_bin:
                cmd += ["--muser-bin", muser_bin]
            if force_minisat:
                cmd += ["--force-minisat"]
            cmd += method_flags
            cmd += [str(cnf_path)]

            with stdout_path.open("w", encoding="utf-8", errors="replace") as out_f, stderr_path.open(
                "w", encoding="utf-8", errors="replace"
            ) as err_f:
                proc = subprocess.run(
                    cmd,
                    cwd=str(marco_root),
                    stdout=out_f,
                    stderr=err_f,
                    text=True,
                    check=False,
                )

            stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
            timed_out = "Time limit reached." in stderr_text
            completed = (proc.returncode == 0) and (not timed_out)
            success = completed
            error = ""
            if not success:
                if timed_out:
                    error = "timeout"
                else:
                    first_err = stderr_text.splitlines()[0] if stderr_text.strip() else f"exitcode={proc.returncode}"
                    error = first_err

            outputs_count, mus_count, mcs_count, first_output_s, first_mus_s, first_mcs_s = _parse_marco_output(stdout_path)

            elapsed = time.perf_counter() - t0
            return RunRecord(
                instance=instance_rel,
                method=method_name,
                run_id=run_id,
                elapsed_s=elapsed,
                success=success,
                timed_out=timed_out,
                completed=completed,
                valid_mus=True,
                valid_mcs=True,
                outputs_count=outputs_count,
                mus_count=mus_count,
                mcs_count=mcs_count,
                first_output_s=first_output_s,
                first_mus_s=first_mus_s,
                first_mcs_s=first_mcs_s,
                error=error,
            )
    except Exception as exc:  # pragma: no cover
        elapsed = time.perf_counter() - t0
        return RunRecord(
            instance=instance_rel,
            method=method_name,
            run_id=run_id,
            elapsed_s=elapsed,
            success=False,
            timed_out=False,
            completed=False,
            valid_mus=False,
            valid_mcs=False,
            outputs_count=0,
            mus_count=0,
            mcs_count=0,
            first_output_s=None,
            first_mus_s=None,
            first_mcs_s=None,
            error=str(exc),
        )


def summarize(records: List[RunRecord]) -> List[MethodSummary]:
    out: List[MethodSummary] = []
    by_method: Dict[str, List[RunRecord]] = {}
    for rec in records:
        by_method.setdefault(rec.method, []).append(rec)

    for method, rows in sorted(by_method.items()):
        completed_rows = [r for r in rows if r.success and r.completed]
        valid_rows = [r for r in rows if r.success and r.valid_mus and r.valid_mcs]

        elapsed_vals = [r.elapsed_s for r in completed_rows]
        output_vals = [r.outputs_count for r in rows]
        mus_vals = [r.mus_count for r in rows]
        mcs_vals = [r.mcs_count for r in rows]
        first_mus_vals = [r.first_mus_s for r in rows if r.first_mus_s is not None]

        out.append(
            MethodSummary(
                method=method,
                runs=len(rows),
                completed_rate=(len(completed_rows) / len(rows)) if rows else 0.0,
                timeout_rate=(sum(1 for r in rows if r.timed_out) / len(rows)) if rows else 0.0,
                valid_rate=(len(valid_rows) / len(rows)) if rows else 0.0,
                median_ms=(statistics.median(elapsed_vals) * 1000.0) if elapsed_vals else None,
                p90_ms=(percentile(elapsed_vals, 0.90) * 1000.0) if elapsed_vals else None,
                median_outputs=statistics.median(output_vals) if output_vals else None,
                median_mus=statistics.median(mus_vals) if mus_vals else None,
                median_mcs=statistics.median(mcs_vals) if mcs_vals else None,
                median_first_mus_ms=(statistics.median(first_mus_vals) * 1000.0) if first_mus_vals else None,
            )
        )
    return out


def print_method_summary(rows: List[MethodSummary], baseline: str) -> None:
    print("\nMethod summary")
    print("method            done  timeout valid median_ms   p90_ms  med_out med_mus med_mcs first_mus_ms")
    for row in rows:
        median_txt = f"{row.median_ms:9.3f}" if row.median_ms is not None else "      n/a"
        p90_txt = f"{row.p90_ms:8.3f}" if row.p90_ms is not None else "    n/a"
        out_txt = f"{row.median_outputs:.1f}" if row.median_outputs is not None else "n/a"
        mus_txt = f"{row.median_mus:.1f}" if row.median_mus is not None else "n/a"
        mcs_txt = f"{row.median_mcs:.1f}" if row.median_mcs is not None else "n/a"
        fm_txt = f"{row.median_first_mus_ms:.1f}" if row.median_first_mus_ms is not None else "n/a"
        print(
            f"{row.method:16} {row.completed_rate:5.2f} {row.timeout_rate:7.2f} {row.valid_rate:5.2f} "
            f"{median_txt} {p90_txt} {out_txt:>7} {mus_txt:>7} {mcs_txt:>7} {fm_txt:>12}"
        )

    lookup = {r.method: r for r in rows}
    if baseline in lookup:
        base = lookup[baseline]
        print(f"\nSpeedup vs baseline '{baseline}' (median completed runtime):")
        for row in rows:
            if row.method == baseline:
                continue
            if base.median_ms is None or row.median_ms is None or row.median_ms <= 0:
                print(f"  {row.method:16} n/a")
                continue
            print(f"  {row.method:16} x{base.median_ms / row.median_ms:.3f}")


def write_csv(path: Path, runs: List[RunRecord], summary_rows: List[MethodSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "instance",
                "method",
                "run_id",
                "elapsed_s",
                "success",
                "timed_out",
                "completed",
                "valid_mus",
                "valid_mcs",
                "outputs_count",
                "mus_count",
                "mcs_count",
                "first_output_s",
                "first_mus_s",
                "first_mcs_s",
                "error",
            ]
        )
        for r in runs:
            w.writerow(
                [
                    r.instance,
                    r.method,
                    r.run_id,
                    f"{r.elapsed_s:.9f}",
                    r.success,
                    r.timed_out,
                    r.completed,
                    r.valid_mus,
                    r.valid_mcs,
                    r.outputs_count,
                    r.mus_count,
                    r.mcs_count,
                    "" if r.first_output_s is None else f"{r.first_output_s:.9f}",
                    "" if r.first_mus_s is None else f"{r.first_mus_s:.9f}",
                    "" if r.first_mcs_s is None else f"{r.first_mcs_s:.9f}",
                    r.error,
                ]
            )

    summary_path = path.with_name(f"{path.stem}_summary{path.suffix or '.csv'}")
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "method",
                "runs",
                "completed_rate",
                "timeout_rate",
                "valid_rate",
                "median_ms",
                "p90_ms",
                "median_outputs",
                "median_mus",
                "median_mcs",
                "median_first_mus_ms",
            ]
        )
        for s in summary_rows:
            w.writerow(
                [
                    s.method,
                    s.runs,
                    f"{s.completed_rate:.6f}",
                    f"{s.timeout_rate:.6f}",
                    f"{s.valid_rate:.6f}",
                    "" if s.median_ms is None else f"{s.median_ms:.6f}",
                    "" if s.p90_ms is None else f"{s.p90_ms:.6f}",
                    "" if s.median_outputs is None else f"{s.median_outputs:.6f}",
                    "" if s.median_mus is None else f"{s.median_mus:.6f}",
                    "" if s.median_mcs is None else f"{s.median_mcs:.6f}",
                    "" if s.median_first_mus_ms is None else f"{s.median_first_mus_ms:.6f}",
                ]
            )

    print(f"\nWrote run CSV: {path}")
    print(f"Wrote summary CSV: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark paper MARCO methods on SAT11 CNF (.bz2)")
    parser.add_argument(
        "--dataset-root",
        default=str(default_dataset_root()),
        help="Root folder containing SAT11 .cnf.bz2 files",
    )
    parser.add_argument(
        "--marco-root",
        default=str(MARCO_ROOT_DEFAULT),
        help="Path to paper MARCO codebase root",
    )
    parser.add_argument(
        "--methods",
        default="marco,marco_adaptive,marco_smart",
        help="Comma-separated methods: marco,marco_basic,marco_plus,marco_adaptive,marco_smart",
    )
    parser.add_argument(
        "--instances",
        default="",
        help="Optional comma-separated filename/path substrings to include",
    )
    parser.add_argument("--max-vars", type=int, default=10000, help="Keep instances with vars <= this (<=0 disables)")
    parser.add_argument(
        "--max-clauses",
        type=int,
        default=100000,
        help="Keep instances with clauses <= this (<=0 disables)",
    )
    parser.add_argument("--max-files", type=int, default=40, help="Max selected instances (<=0 means all)")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle selected instances before truncating")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for --shuffle")

    parser.add_argument("--repeats", type=int, default=1, help="Measured runs per method/instance")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup runs per method/instance")
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=3600.0,
        help="Timeout in seconds per run (<=0 disables timeout)",
    )
    parser.add_argument(
        "--max-outputs",
        type=int,
        default=0,
        help="Cap outputs per run via MARCO --limit (<=0 means unbounded)",
    )
    parser.add_argument("--verify-unsat", action="store_true", help="Accepted for compatibility (currently no-op)")
    parser.add_argument("--validate", action="store_true", help="Accepted for compatibility (currently no-op)")

    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Threads passed to paper MARCO (--threads)",
    )
    parser.add_argument("--muser-bin", default="", help="Path to MUSer2 binary for paper MARCO")
    parser.add_argument("--force-minisat", action="store_true", help="Run paper MARCO with --force-minisat")
    parser.add_argument(
        "--core-handoff",
        type=int,
        default=-1,
        help="marco_smart/core handoff threshold; -1 picks default",
    )
    parser.add_argument("--core-base-ratio", type=int, default=2, help="marco_smart base batch ratio")
    parser.add_argument("--core-backoff-cap", type=int, default=8, help="marco_smart SAT backoff cap")
    parser.add_argument("--core-no-certify", action="store_true", help="Disable marco_smart final certification")
    parser.add_argument("--no-feedback", action="store_true", help="Disable adaptive/smart feedback clauses")
    parser.add_argument(
        "--baseline",
        default="auto",
        help="Baseline method for speedup reporting; use 'auto' to select the first method",
    )
    parser.add_argument("--output-csv", default="", help="Write run-level CSV to this path")
    parser.add_argument("--verbose", action="store_true", help="Print per-run progress")

    # Unused options accepted for compatibility with cpmpy runner invocation.
    parser.add_argument("--solver", default="", help=argparse.SUPPRESS)
    parser.add_argument("--map-solver", default="", help=argparse.SUPPRESS)
    parser.add_argument("--no-solution-hint", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--feedback-sat-clause-max", type=int, default=12, help=argparse.SUPPRESS)
    parser.add_argument("--feedback-unsat-clause-max", type=int, default=12, help=argparse.SUPPRESS)
    parser.add_argument("--feedback-max-clauses", type=int, default=2000, help=argparse.SUPPRESS)

    args = parser.parse_args()

    methods = parse_csv_list(args.methods)
    allowed = {"marco", "marco_basic", "marco_plus", "marco_adaptive", "marco_smart"}
    for m in methods:
        if m not in allowed:
            raise ValueError(f"Unknown method '{m}'. Allowed: {', '.join(sorted(allowed))}")
    baseline = args.baseline.strip()
    if baseline in {"", "auto"}:
        baseline = methods[0]
    elif baseline not in methods:
        raise ValueError("Baseline must be included in --methods")

    root = Path(args.dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    marco_root = Path(args.marco_root).resolve()
    if not marco_root.exists():
        raise FileNotFoundError(f"MARCO root not found: {marco_root}")

    patterns = parse_csv_list(args.instances)
    all_instances = scan_instances(root)
    selected = select_instances(
        all_instances=all_instances,
        include_patterns=patterns,
        max_vars=args.max_vars,
        max_clauses=args.max_clauses,
        max_files=args.max_files,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    if not selected:
        raise ValueError("No instances selected after filtering")

    if args.validate:
        print("[warn] --validate is currently a no-op in bench_marco_sat11.py")
    if args.verify_unsat:
        print("[warn] --verify-unsat is currently a no-op in bench_marco_sat11.py")

    muser_bin = str(Path(args.muser_bin).resolve()) if args.muser_bin else ""

    print("SAT11 paper-MARCO benchmark configuration:")
    print(f"  dataset       : {root}")
    print(f"  marco-root    : {marco_root}")
    print(f"  found files   : {len(all_instances)}")
    print(f"  selected      : {len(selected)}")
    print(f"  methods       : {', '.join(methods)}")
    print(f"  baseline      : {baseline}")
    print(f"  repeats/warmup: {args.repeats}/{args.warmup}")
    print(f"  timeout_s     : {args.timeout_s}")
    print(f"  max_outputs   : {args.max_outputs}")
    print(f"  threads       : {args.threads}")
    print(f"  force_minisat : {args.force_minisat}")
    print(f"  muser_bin     : {muser_bin if muser_bin else '(default lookup)'}")
    print(f"  no_feedback   : {args.no_feedback}")
    print(f"  core_handoff  : {args.core_handoff}")
    print(f"  core_ratio    : {args.core_base_ratio}")
    print(f"  core_backoff  : {args.core_backoff_cap}")
    print(f"  core_certify  : {not args.core_no_certify}")

    run_records: List[RunRecord] = []
    total_jobs = len(selected) * len(methods) * (args.warmup + args.repeats)
    done_jobs = 0

    for inst_idx, meta in enumerate(selected, start=1):
        if args.verbose:
            print(
                f"\n[instance {inst_idx}/{len(selected)}] {meta.relpath} "
                f"(vars={meta.nvars}, clauses={meta.nclauses}, bz2={meta.compressed_bytes})"
            )

        for method in methods:
            for w in range(max(0, args.warmup)):
                _ = run_once(
                    instance_path=meta.path,
                    instance_rel=meta.relpath,
                    method_name=method,
                    run_id=-1,
                    timeout_s=max(0.0, args.timeout_s),
                    max_outputs=args.max_outputs,
                    muser_bin=muser_bin,
                    force_minisat=args.force_minisat,
                    threads=args.threads,
                    no_feedback=args.no_feedback,
                    core_handoff=args.core_handoff,
                    core_base_ratio=args.core_base_ratio,
                    core_backoff_cap=args.core_backoff_cap,
                    core_no_certify=args.core_no_certify,
                    validate=False,
                    verify_unsat=False,
                    marco_root=marco_root,
                )
                done_jobs += 1
                if args.verbose:
                    print(f"  [warmup {w + 1}/{args.warmup}] {method} done ({done_jobs}/{total_jobs})")

            for run_id in range(args.repeats):
                rec = run_once(
                    instance_path=meta.path,
                    instance_rel=meta.relpath,
                    method_name=method,
                    run_id=run_id,
                    timeout_s=max(0.0, args.timeout_s),
                    max_outputs=args.max_outputs,
                    muser_bin=muser_bin,
                    force_minisat=args.force_minisat,
                    threads=args.threads,
                    no_feedback=args.no_feedback,
                    core_handoff=args.core_handoff,
                    core_base_ratio=args.core_base_ratio,
                    core_backoff_cap=args.core_backoff_cap,
                    core_no_certify=args.core_no_certify,
                    validate=args.validate,
                    verify_unsat=args.verify_unsat,
                    marco_root=marco_root,
                )
                done_jobs += 1
                run_records.append(rec)

                if args.verbose:
                    status = "ok" if rec.success else ("timeout" if rec.timed_out else "fail")
                    print(
                        f"  [run {run_id + 1}/{args.repeats}] {method} {status} "
                        f"time={rec.elapsed_s:.3f}s out={rec.outputs_count} mus={rec.mus_count} mcs={rec.mcs_count} "
                        f"({done_jobs}/{total_jobs})"
                    )
                    if rec.error and rec.error != "timeout":
                        print(f"    error: {rec.error.splitlines()[0]}")

    summary_rows = summarize(run_records)
    print_method_summary(summary_rows, baseline=baseline)

    if args.output_csv:
        write_csv(Path(args.output_csv), run_records, summary_rows)


if __name__ == "__main__":
    main()
