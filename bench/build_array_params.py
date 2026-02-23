#!/usr/bin/env python3
"""
Build an array-params TSV for MARCO-paper SAT11 runs.

Input manifest format (TSV):
    <relative_path>\t<nvars>\t<nclauses>\t<compressed_bytes>

Output array params format (TSV, no header):
    <relative_path>\t<method>\t<repeat_id>
Optional extended format:
    <relative_path>\t<method>\t<repeat_id>\t<threads>\t<muser_bin>\t<force_minisat>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

BENCH_DIR = Path(__file__).resolve().parent


def parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build array params TSV for MARCO-paper cluster execution")
    parser.add_argument(
        "--manifest",
        default=str(BENCH_DIR / "sat11_manifest.tsv"),
        help="Input manifest TSV (from build_manifest.py)",
    )
    parser.add_argument(
        "--methods",
        default="marco,marco_adaptive,marco_smart,marco_portfolio",
        help="Comma-separated methods",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeats per instance/method")
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Optional per-task threads column for MARCO paper runner",
    )
    parser.add_argument(
        "--muser-bin",
        default="",
        help="Optional per-task MUSer2 binary path column",
    )
    parser.add_argument(
        "--force-minisat",
        action="store_true",
        help="Write force-minisat=1 in per-task columns",
    )
    parser.add_argument(
        "--output",
        default=str(BENCH_DIR / "array_params.tsv"),
        help="Output array params TSV",
    )
    args = parser.parse_args()

    manifest = Path(args.manifest)
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.threads < 1:
        raise ValueError("--threads must be >= 1")

    methods = parse_csv_list(args.methods)
    if not methods:
        raise ValueError("No methods specified")
    allowed = {"marco", "marco_basic", "marco_plus", "marco_adaptive", "marco_smart", "marco_portfolio"}
    unknown = [m for m in methods if m not in allowed]
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)} (allowed: {', '.join(sorted(allowed))})")

    instances: List[str] = []
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cols = line.split("\t")
            if cols and cols[0]:
                instances.append(cols[0])

    if not instances:
        raise ValueError(f"No instance rows found in {manifest}")

    muser_col = args.muser_bin.strip()
    muser_col_out = muser_col if muser_col else "-"
    force_col = "1" if args.force_minisat else "0"
    write_extended = (args.threads != 1) or bool(muser_col) or args.force_minisat

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    nrows = 0
    with out.open("w", encoding="utf-8") as f:
        for rep in range(args.repeats):
            for method in methods:
                for inst in instances:
                    if write_extended:
                        f.write(f"{inst}\t{method}\t{rep}\t{args.threads}\t{muser_col_out}\t{force_col}\n")
                    else:
                        f.write(f"{inst}\t{method}\t{rep}\n")
                    nrows += 1

    print("Array params written")
    print(f"  manifest     : {manifest}")
    print(f"  methods      : {', '.join(methods)}")
    print(f"  repeats      : {args.repeats}")
    print(f"  instances    : {len(instances)}")
    print(f"  rows         : {nrows}")
    if write_extended:
        print(f"  threads col  : {args.threads}")
        print(f"  muser col    : {muser_col if muser_col else '- (empty marker)'}")
        print(f"  minisat col  : {force_col}")
    print(f"  output       : {out}")


if __name__ == "__main__":
    main()
