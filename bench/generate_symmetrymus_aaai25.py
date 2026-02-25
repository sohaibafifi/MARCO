#!/usr/bin/env python3
"""
Generate the AAAI'25 SymmetryMUS synthetic benchmark as DIMACS .cnf.bz2 files.

Target composition from the paper appendix:
- 146 pigeon-hole instances
- 66 n+k-queens instances
- 60 bin-packing instances
Total: 272 instances

The output dataset can be consumed directly by MARCO bench scripts that scan
"*.cnf.bz2" files.
"""

from __future__ import annotations

import argparse
import bz2
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from pysat.card import CardEnc, EncType
from pysat.pb import PBEnc
from pysat.solvers import Solver

try:
    from ortools.sat.python import cp_model
except Exception:
    cp_model = None


BENCH_DIR = Path(__file__).resolve().parent
MARCO_ROOT = BENCH_DIR.parent


@dataclass
class GeneratedMeta:
    family: str
    relpath: str
    nvars: int
    nclauses: int
    compressed_bytes: int


class CNFBuilder:
    def __init__(self) -> None:
        self.nvars = 0
        self.clauses: List[List[int]] = []

    def new_var(self) -> int:
        self.nvars += 1
        return self.nvars

    def add_clause(self, clause: Sequence[int]) -> None:
        self.clauses.append(list(clause))

    def add_clauses(self, clauses: Iterable[Sequence[int]]) -> None:
        for c in clauses:
            self.add_clause(c)

    def _merge_encoded(self, enc) -> None:
        self.add_clauses(enc.clauses)
        if enc.nv > self.nvars:
            self.nvars = enc.nv

    def add_atmost(self, lits: Sequence[int], bound: int) -> None:
        lits = list(lits)
        if bound < 0:
            self.add_clause([])
            return
        if len(lits) <= bound:
            return
        enc = CardEnc.atmost(
            lits=lits,
            bound=bound,
            encoding=EncType.seqcounter,
            top_id=self.nvars,
        )
        self._merge_encoded(enc)

    def add_atleast(self, lits: Sequence[int], bound: int) -> None:
        lits = list(lits)
        if bound <= 0:
            return
        if bound > len(lits):
            self.add_clause([])
            return
        enc = CardEnc.atleast(
            lits=lits,
            bound=bound,
            encoding=EncType.seqcounter,
            top_id=self.nvars,
        )
        self._merge_encoded(enc)

    def add_equals(self, lits: Sequence[int], bound: int) -> None:
        lits = list(lits)
        if bound < 0 or bound > len(lits):
            self.add_clause([])
            return
        enc = CardEnc.equals(
            lits=lits,
            bound=bound,
            encoding=EncType.seqcounter,
            top_id=self.nvars,
        )
        self._merge_encoded(enc)

    def add_weighted_atmost(self, lits: Sequence[int], weights: Sequence[int], bound: int) -> None:
        lits = list(lits)
        if not lits:
            if bound < 0:
                self.add_clause([])
            return
        enc = PBEnc.atmost(
            lits=lits,
            weights=list(weights),
            bound=bound,
            top_id=self.nvars,
            encoding=0,
        )
        self._merge_encoded(enc)


def write_cnf_bz2(path: Path, builder: CNFBuilder, compresslevel: int = 9) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with bz2.open(path, "wt", encoding="ascii", compresslevel=compresslevel) as f:
        f.write(f"p cnf {builder.nvars} {len(builder.clauses)}\n")
        for clause in builder.clauses:
            if clause:
                f.write(" ".join(str(x) for x in clause))
                f.write(" 0\n")
            else:
                f.write("0\n")


def check_unsat(builder: CNFBuilder, solver_name: str = "m22") -> bool:
    with Solver(name=solver_name, bootstrap_with=builder.clauses) as s:
        return not s.solve()


def build_php_instance(pigeons: int, holes: int) -> CNFBuilder:
    b = CNFBuilder()
    x = [[b.new_var() for _ in range(holes)] for _ in range(pigeons)]

    # Every pigeon must be in at least one hole.
    for i in range(pigeons):
        b.add_clause(x[i])

    # Every hole can host at most one pigeon.
    for j in range(holes):
        b.add_atmost([x[i][j] for i in range(pigeons)], 1)

    return b


def _line_coords_rows_cols_diags(n: int) -> List[List[Tuple[int, int]]]:
    lines: List[List[Tuple[int, int]]] = []

    # Rows.
    for i in range(n):
        lines.append([(i, j) for j in range(n)])

    # Cols.
    for j in range(n):
        lines.append([(i, j) for i in range(n)])

    # Main diagonals (i-j = const).
    for d in range(-(n - 1), n):
        cells = []
        for i in range(n):
            j = i - d
            if 0 <= j < n:
                cells.append((i, j))
        if len(cells) > 1:
            lines.append(cells)

    # Anti-diagonals (i+j = const).
    for s in range(0, 2 * n - 1):
        cells = []
        for i in range(n):
            j = s - i
            if 0 <= j < n:
                cells.append((i, j))
        if len(cells) > 1:
            lines.append(cells)

    return lines


def build_nk_queens_instance(n: int, k: int) -> CNFBuilder:
    b = CNFBuilder()
    q = [[b.new_var() for _ in range(n)] for _ in range(n)]

    # No attacks in rows, cols, or diagonals.
    for line in _line_coords_rows_cols_diags(n):
        lits = [q[i][j] for (i, j) in line]
        b.add_atmost(lits, 1)

    # Force exactly n+k queens (unsat for k > 0 with row/col at-most-1).
    all_lits = [q[i][j] for i in range(n) for j in range(n)]
    b.add_equals(all_lits, n + k)

    return b


def _solve_binpacking_optimal_bins(weights: List[int], capacity: int, timeout_s: float) -> int:
    """
    Min #bins used for a fixed set of item weights.

    Falls back to capacity lower bound if OR-Tools is unavailable.
    """
    n = len(weights)
    if cp_model is None:
        return int(math.ceil(sum(weights) / float(capacity)))

    model = cp_model.CpModel()
    x = {(i, j): model.NewBoolVar(f"x_{i}_{j}") for i in range(n) for j in range(n)}
    used = {j: model.NewBoolVar(f"u_{j}") for j in range(n)}

    for i in range(n):
        model.Add(sum(x[(i, j)] for j in range(n)) == 1)

    for j in range(n):
        model.Add(sum(weights[i] * x[(i, j)] for i in range(n)) <= capacity)
        for i in range(n):
            model.Add(x[(i, j)] <= used[j])

    model.Minimize(sum(used[j] for j in range(n)))

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = float(max(1.0, timeout_s))
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return int(math.ceil(sum(weights) / float(capacity)))

    return int(round(solver.ObjectiveValue()))


def _binpacking_weights(n_items: int, lb: int, ub: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    return [rng.randint(lb, ub) for _ in range(n_items)]


def build_binpacking_instance(weights: List[int], capacity: int, n_bins: int) -> CNFBuilder:
    """
    Feasibility CNF for bin packing with fixed number of available bins.

    - each item is assigned to exactly one bin
    - weighted capacity per bin
    """
    n_items = len(weights)
    b = CNFBuilder()
    x = [[b.new_var() for _ in range(n_bins)] for _ in range(n_items)]

    for i in range(n_items):
        b.add_equals(x[i], 1)

    for j in range(n_bins):
        b.add_weighted_atmost([x[i][j] for i in range(n_items)], weights, capacity)

    return b


def parse_csv_ints(raw: str) -> List[int]:
    vals: List[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok:
            vals.append(int(tok))
    return vals


def expected_instance_counts(
    php_start: int,
    php_stop: int,
    php_step: int,
    queens_start: int,
    queens_stop: int,
    queens_k_vals: Sequence[int],
    bins_start: int,
    bins_stop: int,
    ratios: Sequence[int],
) -> Dict[str, int]:
    php_n = len(list(range(php_start, php_stop + 1, php_step))) * 2
    queens_n = len(list(range(queens_start, queens_stop + 1))) * len(queens_k_vals)
    bins_n = len(list(range(bins_start, bins_stop + 1))) * len(ratios)
    return {
        "php": php_n,
        "queens": queens_n,
        "binpacking": bins_n,
        "total": php_n + queens_n + bins_n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the AAAI'25 SymmetryMUS 272-instance synthetic benchmark as .cnf.bz2"
    )

    parser.add_argument(
        "--output-root",
        default=str(MARCO_ROOT / "SymmetryMUS-AAAI25-Benchmarks"),
        help="Dataset output directory",
    )
    parser.add_argument(
        "--manifest-output",
        default=str(BENCH_DIR / "symmetrymus_aaai25_manifest.tsv"),
        help="Manifest TSV path (same format as build_manifest.py output). Empty disables writing.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed for bin-packing weights")
    parser.add_argument("--compresslevel", type=int, default=9, help="bzip2 compression level (1-9)")
    parser.add_argument("--verify-unsat", action="store_true", help="Verify each generated CNF is UNSAT")
    parser.add_argument("--solver-name", default="m22", help="PySAT solver for --verify-unsat")

    parser.add_argument("--php-min-pigeons", type=int, default=5)
    parser.add_argument("--php-max-pigeons", type=int, default=149)
    parser.add_argument("--php-step", type=int, default=2)

    parser.add_argument("--queens-min-n", type=int, default=4)
    parser.add_argument("--queens-max-n", type=int, default=25)
    parser.add_argument("--queens-k", default="2,3,4")

    # Paper text says 5..25 bins; to match 60 instances total with 3 ratios,
    # default uses 5..24 (20 values).
    parser.add_argument("--bins-min", type=int, default=5)
    parser.add_argument("--bins-max", type=int, default=24)
    parser.add_argument("--bin-ratios", default="70,80,90")
    parser.add_argument("--capacity", type=int, default=25)
    parser.add_argument("--weight-lb", type=int, default=5)
    parser.add_argument("--weight-ub", type=int, default=10)
    parser.add_argument(
        "--bin-opt-timeout-s",
        type=float,
        default=20.0,
        help="Timeout (seconds) for required-bin optimization in bin-packing generation",
    )

    args = parser.parse_args()

    queens_k_vals = parse_csv_ints(args.queens_k)
    bin_ratios = parse_csv_ints(args.bin_ratios)

    counts = expected_instance_counts(
        php_start=args.php_min_pigeons,
        php_stop=args.php_max_pigeons,
        php_step=args.php_step,
        queens_start=args.queens_min_n,
        queens_stop=args.queens_max_n,
        queens_k_vals=queens_k_vals,
        bins_start=args.bins_min,
        bins_stop=args.bins_max,
        ratios=bin_ratios,
    )

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print("Generating SymmetryMUS synthetic benchmark")
    print(f"  output root : {out_root}")
    print(f"  php         : {counts['php']}")
    print(f"  queens      : {counts['queens']}")
    print(f"  binpacking  : {counts['binpacking']}")
    print(f"  total       : {counts['total']}")

    rows: List[GeneratedMeta] = []

    # 1) Pigeon-hole instances.
    for p in range(args.php_min_pigeons, args.php_max_pigeons + 1, args.php_step):
        for h in (p - 2, p - 3):
            rel = f"php/php_p{p:03d}_h{h:03d}.cnf.bz2"
            dst = out_root / rel
            cnf = build_php_instance(p, h)
            if args.verify_unsat and not check_unsat(cnf, solver_name=args.solver_name):
                raise RuntimeError(f"Expected UNSAT, got SAT for {rel}")
            write_cnf_bz2(dst, cnf, compresslevel=args.compresslevel)
            rows.append(
                GeneratedMeta(
                    family="php",
                    relpath=rel,
                    nvars=cnf.nvars,
                    nclauses=len(cnf.clauses),
                    compressed_bytes=dst.stat().st_size,
                )
            )

    # 2) N+k-queens instances.
    for n in range(args.queens_min_n, args.queens_max_n + 1):
        for k in queens_k_vals:
            rel = f"queens/queens_n{n:02d}_k{k}.cnf.bz2"
            dst = out_root / rel
            cnf = build_nk_queens_instance(n, k)
            if args.verify_unsat and not check_unsat(cnf, solver_name=args.solver_name):
                raise RuntimeError(f"Expected UNSAT, got SAT for {rel}")
            write_cnf_bz2(dst, cnf, compresslevel=args.compresslevel)
            rows.append(
                GeneratedMeta(
                    family="queens",
                    relpath=rel,
                    nvars=cnf.nvars,
                    nclauses=len(cnf.clauses),
                    compressed_bytes=dst.stat().st_size,
                )
            )

    # 3) Bin-packing instances.
    for n_items in range(args.bins_min, args.bins_max + 1):
        # Keep one deterministic weights vector per n_items and derive all ratios from it.
        seed = args.seed + n_items * 1009
        weights = _binpacking_weights(n_items, args.weight_lb, args.weight_ub, seed)

        required = _solve_binpacking_optimal_bins(weights, args.capacity, args.bin_opt_timeout_s)
        if required <= 1:
            # Avoid trivial 0-bin unsat corner cases by re-sampling deterministically.
            for bump in range(1, 128):
                seed2 = seed + bump
                w2 = _binpacking_weights(n_items, args.weight_lb, args.weight_ub, seed2)
                req2 = _solve_binpacking_optimal_bins(w2, args.capacity, args.bin_opt_timeout_s)
                if req2 > 1:
                    weights = w2
                    required = req2
                    seed = seed2
                    break

        for ratio in bin_ratios:
            avail = math.floor((ratio / 100.0) * required)
            if avail >= required:
                avail = required - 1

            rel = (
                f"binpacking/binpack_n{n_items:02d}_r{ratio}_"
                f"req{required:02d}_avail{avail:02d}_seed{seed}.cnf.bz2"
            )
            dst = out_root / rel

            cnf = build_binpacking_instance(weights, args.capacity, avail)
            if args.verify_unsat and not check_unsat(cnf, solver_name=args.solver_name):
                raise RuntimeError(f"Expected UNSAT, got SAT for {rel}")
            write_cnf_bz2(dst, cnf, compresslevel=args.compresslevel)
            rows.append(
                GeneratedMeta(
                    family="binpacking",
                    relpath=rel,
                    nvars=cnf.nvars,
                    nclauses=len(cnf.clauses),
                    compressed_bytes=dst.stat().st_size,
                )
            )

    # Sanity checks.
    by_family: Dict[str, int] = {"php": 0, "queens": 0, "binpacking": 0}
    for r in rows:
        by_family[r.family] += 1

    print("Generated")
    print(f"  php         : {by_family['php']}")
    print(f"  queens      : {by_family['queens']}")
    print(f"  binpacking  : {by_family['binpacking']}")
    print(f"  total       : {len(rows)}")

    if args.manifest_output:
        manifest_path = Path(args.manifest_output)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as f:
            for r in sorted(rows, key=lambda x: x.relpath):
                f.write(f"{r.relpath}\t{r.nvars}\t{r.nclauses}\t{r.compressed_bytes}\n")
        print(f"Manifest written: {manifest_path}")


if __name__ == "__main__":
    main()
