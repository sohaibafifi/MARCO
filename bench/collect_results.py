#!/usr/bin/env python3
"""
Collect per-task CSV files from MARCO-paper SAT11 array runs into consolidated outputs.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

BENCH_DIR = Path(__file__).resolve().parent


@dataclass
class Summary:
    method: str
    instances: int
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
    idx = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * p))))
    return ordered[idx]


def to_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def to_float_opt(v: str) -> Optional[float]:
    s = str(v).strip() if v is not None else ""
    if not s:
        return None
    return float(s)


def to_int(v: str) -> int:
    return int(float(str(v).strip()))


def read_rows(runs_dir: Path) -> List[dict]:
    files = sorted(p for p in runs_dir.rglob("*.csv") if "_summary" not in p.name)
    rows: List[dict] = []
    for p in files:
        with p.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rr = dict(r)
                rr["_source_file"] = str(p)
                rows.append(rr)
    return rows


def summarize(rows: List[dict]) -> List[Summary]:
    by_method: Dict[str, List[dict]] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)

    out: List[Summary] = []
    for method, rs in sorted(by_method.items()):
        unique_instances = len({r.get("instance", "") for r in rs if r.get("instance", "")})
        completed = [r for r in rs if to_bool(r["success"]) and to_bool(r["completed"])]
        valid = [r for r in rs if to_bool(r["success"]) and to_bool(r["valid_mus"]) and to_bool(r["valid_mcs"])]
        elapsed_vals = [float(r["elapsed_s"]) for r in completed if r.get("elapsed_s")]
        outputs_vals = [to_int(r["outputs_count"]) for r in rs if to_bool(r["success"])]
        mus_vals = [to_int(r["mus_count"]) for r in rs if to_bool(r["success"])]
        mcs_vals = [to_int(r["mcs_count"]) for r in rs if to_bool(r["success"])]
        first_mus_vals = [
            to_float_opt(r["first_mus_s"]) for r in rs if to_bool(r["success"]) and to_float_opt(r["first_mus_s"]) is not None
        ]

        out.append(
            Summary(
                method=method,
                instances=unique_instances,
                runs=len(rs),
                completed_rate=(len(completed) / len(rs)) if rs else 0.0,
                timeout_rate=(sum(1 for r in rs if to_bool(r["timed_out"])) / len(rs)) if rs else 0.0,
                valid_rate=(len(valid) / len(rs)) if rs else 0.0,
                median_ms=(statistics.median(elapsed_vals) * 1000.0) if elapsed_vals else None,
                p90_ms=(percentile(elapsed_vals, 0.9) * 1000.0) if elapsed_vals else None,
                median_outputs=statistics.median(outputs_vals) if outputs_vals else None,
                median_mus=statistics.median(mus_vals) if mus_vals else None,
                median_mcs=statistics.median(mcs_vals) if mcs_vals else None,
                median_first_mus_ms=(statistics.median(first_mus_vals) * 1000.0) if first_mus_vals else None,
            )
        )
    return out


def write_all_runs(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = [k for k in rows[0].keys() if not k.startswith("_")]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def write_summary(path: Path, rows: List[Summary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "method",
                "instances",
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
        for r in rows:
            w.writerow(
                [
                    r.method,
                    r.instances,
                    r.runs,
                    f"{r.completed_rate:.6f}",
                    f"{r.timeout_rate:.6f}",
                    f"{r.valid_rate:.6f}",
                    "" if r.median_ms is None else f"{r.median_ms:.6f}",
                    "" if r.p90_ms is None else f"{r.p90_ms:.6f}",
                    "" if r.median_outputs is None else f"{r.median_outputs:.6f}",
                    "" if r.median_mus is None else f"{r.median_mus:.6f}",
                    "" if r.median_mcs is None else f"{r.median_mcs:.6f}",
                    "" if r.median_first_mus_ms is None else f"{r.median_first_mus_ms:.6f}",
                ]
            )


def print_summary(rows: List[Summary]) -> None:
    print("method            inst  done  timeout valid median_ms   p90_ms  med_out med_mus med_mcs first_mus_ms")
    for r in rows:
        median_txt = f"{r.median_ms:9.3f}" if r.median_ms is not None else "      n/a"
        p90_txt = f"{r.p90_ms:8.3f}" if r.p90_ms is not None else "    n/a"
        out_txt = f"{r.median_outputs:.1f}" if r.median_outputs is not None else "n/a"
        mus_txt = f"{r.median_mus:.1f}" if r.median_mus is not None else "n/a"
        mcs_txt = f"{r.median_mcs:.1f}" if r.median_mcs is not None else "n/a"
        fm_txt = f"{r.median_first_mus_ms:.1f}" if r.median_first_mus_ms is not None else "n/a"
        print(
            f"{r.method:16} {r.instances:4d} {r.completed_rate:5.2f} {r.timeout_rate:7.2f} {r.valid_rate:5.2f} "
            f"{median_txt} {p90_txt} {out_txt:>7} {mus_txt:>7} {mcs_txt:>7} {fm_txt:>12}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect per-task SAT11 benchmark CSV files (paper MARCO workflow)")
    parser.add_argument("--runs-dir", default=str(BENCH_DIR / "results" / "runs"), help="Directory containing per-task CSV files")
    parser.add_argument("--output-all", default=str(BENCH_DIR / "results" / "all_runs.csv"), help="Consolidated runs CSV")
    parser.add_argument(
        "--output-summary",
        default=str(BENCH_DIR / "results" / "all_runs_summary.csv"),
        help="Method summary CSV",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"runs-dir not found: {runs_dir}")

    rows = read_rows(runs_dir)
    unique_instances = len({r.get("instance", "") for r in rows if r.get("instance", "")})
    print(f"Found {len(rows)} run rows in {runs_dir}")
    print(f"Found {unique_instances} unique instances")
    write_all_runs(Path(args.output_all), rows)

    summary_rows = summarize(rows)
    write_summary(Path(args.output_summary), summary_rows)
    print_summary(summary_rows)

    print(f"\nWrote: {args.output_all}")
    print(f"Wrote: {args.output_summary}")


if __name__ == "__main__":
    main()
