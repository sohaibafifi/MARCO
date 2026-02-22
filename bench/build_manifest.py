#!/usr/bin/env python3
"""
Build a filtered SAT11 manifest for MARCO-paper array benchmarks.

Output format (TSV, no header):
    <relative_path>\t<nvars>\t<nclauses>\t<compressed_bytes>
"""

from __future__ import annotations

import argparse
import bz2
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

P_HEADER = re.compile(r"^p\s+cnf\s+(\d+)\s+(\d+)\s*$")
BENCH_DIR = Path(__file__).resolve().parent
MARCO_ROOT = BENCH_DIR.parent


@dataclass
class Meta:
    relpath: str
    nvars: int
    nclauses: int
    compressed_bytes: int


def read_header(path: Path) -> Tuple[Optional[int], Optional[int]]:
    with bz2.open(path, "rt", encoding="ascii", errors="ignore") as f:
        for line in f:
            m = P_HEADER.match(line.strip())
            if m:
                return int(m.group(1)), int(m.group(2))
    return None, None


def parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def default_dataset_root() -> Path:
    env_root = os.environ.get("DATASET_ROOT")
    if env_root:
        return Path(env_root)
    candidates: List[Path] = [
        MARCO_ROOT / "SAT11-Competition-MUS-SelectedBenchmarks",
        MARCO_ROOT.parent / "SAT11-Competition-MUS-SelectedBenchmarks",
        Path.cwd() / "SAT11-Competition-MUS-SelectedBenchmarks",
        Path.cwd() / "benchmarks" / "SAT11-Competition-MUS-SelectedBenchmarks",
    ]
    if len(BENCH_DIR.parents) > 3:
        candidates.append(BENCH_DIR.parents[3] / "SAT11-Competition-MUS-SelectedBenchmarks")
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SAT11 manifest TSV for MARCO-paper workflow")
    parser.add_argument(
        "--dataset-root",
        default=str(default_dataset_root()),
        help="Root folder containing .cnf.bz2 files",
    )
    parser.add_argument(
        "--output",
        default=str(BENCH_DIR / "sat11_manifest.tsv"),
        help="Output TSV path",
    )
    parser.add_argument("--contains", default="", help="Comma-separated filename/path substrings to include")
    parser.add_argument("--max-vars", type=int, default=10000, help="Keep instances with vars <= this (<=0 disables)")
    parser.add_argument(
        "--max-clauses",
        type=int,
        default=100000,
        help="Keep instances with clauses <= this (<=0 disables)",
    )
    parser.add_argument("--max-files", type=int, default=40, help="Keep at most this many files (<=0 means all)")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before truncating with --max-files")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for --shuffle")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    patterns = [p.lower() for p in parse_csv_list(args.contains)]
    metas: List[Meta] = []
    total_files = 0
    header_ok = 0

    for p in sorted(root.rglob("*.cnf.bz2")):
        total_files += 1
        nvars, nclauses = read_header(p)
        if nvars is None or nclauses is None:
            continue
        header_ok += 1
        rel = str(p.relative_to(root))

        if patterns and not any(pt in rel.lower() or pt in p.name.lower() for pt in patterns):
            continue
        if args.max_vars > 0 and nvars > args.max_vars:
            continue
        if args.max_clauses > 0 and nclauses > args.max_clauses:
            continue

        metas.append(
            Meta(
                relpath=rel,
                nvars=nvars,
                nclauses=nclauses,
                compressed_bytes=p.stat().st_size,
            )
        )

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(metas)
    else:
        metas.sort(key=lambda m: (m.nclauses, m.nvars, m.relpath))

    if args.max_files > 0:
        metas = metas[: args.max_files]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(f"{m.relpath}\t{m.nvars}\t{m.nclauses}\t{m.compressed_bytes}\n")

    print("Manifest written")
    print(f"  dataset-root : {root}")
    print(f"  scanned      : {total_files}")
    print(f"  header-ok    : {header_ok}")
    print(f"  selected     : {len(metas)}")
    print(f"  output       : {out_path}")
    if metas:
        nvars = [m.nvars for m in metas]
        nclauses = [m.nclauses for m in metas]
        print(f"  vars         : min={min(nvars)} median={sorted(nvars)[len(nvars)//2]} max={max(nvars)}")
        print(f"  clauses      : min={min(nclauses)} median={sorted(nclauses)[len(nclauses)//2]} max={max(nclauses)}")


if __name__ == "__main__":
    main()
