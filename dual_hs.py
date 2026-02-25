#!/usr/bin/env python3

"""
DualHS prototype (non-MARCO): MUS/MSS search via implicit hitting sets.

This is intentionally independent from MARCO seed-map lattice walking.
It uses:
- a hitting-set master (Minicard) over clause-selection variables
- a SAT/UNSAT oracle on selected clauses
- correction-set constraints from SAT (MSS complements)
- MUS blocking clauses from UNSAT (block supersets of discovered MUSes)
"""

import argparse
import os
import sys
import time

from src.marco import CNFsolvers
from src.marco import mapsolvers
from src.marco import utils


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("inputfile", type=str, help="CNF/GCNF input")
    parser.add_argument("--cnf", action="store_true", help="assume CNF/GCNF input")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="print subset indexes")
    parser.add_argument("-a", "--alltimes", action="store_true", help="print time for each output")
    parser.add_argument("-s", "--stats", action="store_true", help="print timing statistics to stderr")
    parser.add_argument("-T", "--timeout", type=int, default=None, help="timeout in seconds")
    parser.add_argument("-l", "--limit", type=int, default=None, help="max number of outputs")
    parser.add_argument("--print-mcses", action="store_true", help="print MCS instead of MSS for SAT outputs")
    parser.add_argument(
        "--solver",
        choices=["implies", "hybrid", "muser"],
        default="implies",
        help="subset solver backend (default: implies)",
    )
    parser.add_argument("--muser-bin", type=str, default=None, help="path to MUSer2 binary")
    parser.add_argument("--hybrid-shrink-handoff-size", type=int, default=256)
    parser.add_argument("--hybrid-shrink-handoff-floor", type=int, default=64)
    parser.add_argument("--hybrid-shrink-stagnation", type=int, default=64)
    parser.add_argument(
        "--map-master",
        choices=["auto", "minisat", "minicard"],
        default="auto",
        help="map solver backend for DualHS seeds (default: auto)",
    )
    parser.add_argument(
        "--mus-quota-every",
        type=int,
        default=0,
        help="prefer MUS seeds when no MUS was output in the last N outputs (<=0 disables)",
    )
    # Compatibility flags accepted from shared benchmark runner.
    parser.add_argument("--threads", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--force-minisat", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--core-handoff", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--core-base-ratio", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--core-backoff-cap", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument("--core-no-certify", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--portfolio-smart-after-mus", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--portfolio-smart-after-outputs", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--sat-map-assist-min-gap", type=int, default=32, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    return args


def print_stats(stats):
    times = stats.get_times()
    counts = stats.get_counts()

    categories = sorted(times, key=times.get)
    if not categories:
        return
    maxlen = max(len(x) for x in categories)
    for category in categories:
        sys.stderr.write("%-*s : %8.3f\n" % (maxlen, category, times[category]))
    for category in sorted(counts):
        sys.stderr.write("%-*s : %8d\n" % (maxlen + 6, category + " count", counts[category]))
        if category in times:
            sys.stderr.write("%-*s : %8.5f\n" % (maxlen + 6, category + " per", times[category] / counts[category]))


def setup_subset_solver(args):
    muser_bin = args.muser_bin
    if muser_bin is None and args.solver in ("hybrid", "muser"):
        candidate = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "muser2-para",
            "src",
            "tools",
            "muser-2",
            "muser-2",
        )
        if os.path.isfile(candidate):
            muser_bin = candidate

    if args.solver == "implies":
        csolver = CNFsolvers.ImprovedImpliesSubsetSolver(args.inputfile)
    elif args.solver == "muser":
        csolver = CNFsolvers.MUSerSubsetSolver(args.inputfile, muser_path=muser_bin)
    else:
        csolver = CNFsolvers.HybridImprovedSubsetSolver(
            args.inputfile,
            muser_path=muser_bin,
            handoff_size=args.hybrid_shrink_handoff_size,
            handoff_floor=args.hybrid_shrink_handoff_floor,
            handoff_stagnation=args.hybrid_shrink_stagnation,
        )
    return csolver


def setup_map_solver(n, backend, bias):
    if backend == "minicard":
        return mapsolvers.MinicardMapSolver(n, bias=bias)
    if backend == "minisat":
        return mapsolvers.MinisatMapSolver(n, bias=bias)
    # auto: prioritize fast unconstrained seed extraction on the main map;
    # keep MUS-biased map strict when quota mode is active.
    if bias:
        return mapsolvers.MinicardMapSolver(n, bias=True)
    return mapsolvers.MinisatMapSolver(n, bias=False)


def format_output(kind, subset, now_s, args, n):
    if kind == "S" and args.print_mcses:
        subset = sorted(set(range(1, n + 1)).difference(set(subset)))
        kind = "C"

    out = kind
    if args.alltimes:
        out = "%s %0.3f" % (out, now_s)
    if args.verbose:
        out = "%s %s" % (out, " ".join(str(x) for x in subset))
    return out


def main():
    args = parse_args()

    if not os.path.isfile(args.inputfile):
        sys.stderr.write("ERROR: input file not found: %s\n" % args.inputfile)
        sys.exit(1)

    if not (args.cnf or args.inputfile.endswith((".cnf", ".cnf.gz", ".gcnf", ".gcnf.gz"))):
        sys.stderr.write("ERROR: DualHS prototype currently supports CNF/GCNF only.\n")
        sys.exit(1)

    if args.hybrid_shrink_handoff_size < 1 or args.hybrid_shrink_handoff_floor < 1 or args.hybrid_shrink_stagnation < 1:
        sys.stderr.write("ERROR: hybrid handoff parameters must be >= 1.\n")
        sys.exit(1)
    if args.mus_quota_every < 0:
        sys.stderr.write("ERROR: --mus-quota-every must be >= 0.\n")
        sys.exit(1)

    stats = utils.Statistics()
    start = time.perf_counter()

    try:
        with stats.time("setup"):
            csolver = setup_subset_solver(args)
            n = csolver.n
            all_constraints = set(range(1, n + 1))
            # Main map for default DualHS enumeration (low-bias, MCS-friendly).
            master = setup_map_solver(n, backend=args.map_master, bias=False)
            # Optional MUS-biased master for quota mode (large seeds first).
            mus_master = (
                setup_map_solver(n, backend=args.map_master, bias=True) if args.mus_quota_every > 0 else None
            )
            csolver.set_msolver(master)
            active_for_csolver = master

        outputs = 0
        outputs_since_mus = 0

        while True:
            if args.timeout and (time.perf_counter() - start) >= args.timeout:
                sys.stderr.write("Time limit reached.\n")
                sys.exit(128)

            use_mus_quota = (
                mus_master is not None and args.mus_quota_every > 0 and outputs_since_mus >= args.mus_quota_every
            )
            active_master = mus_master if use_mus_quota else master
            if active_master is not active_for_csolver:
                csolver.set_msolver(active_master)
                active_for_csolver = active_master
            with stats.time("seed"):
                seed = active_master.next_seed()
                if seed is None and active_master is mus_master:
                    # Safety fallback: if MUS-biased search is exhausted under current
                    # cardinality bound, retry from default map state.
                    seed = master.next_seed()
            if seed is None:
                break

            seed = sorted(set(seed))

            with stats.time("check"):
                seed_is_sat, improved = csolver.check_subset(seed, improve_seed=True)
            if improved is not None:
                seed = sorted(set(improved))

            if seed_is_sat:
                with stats.time("grow"):
                    mss = sorted(set(csolver.grow(seed)))
                correction = sorted(all_constraints.difference(set(mss)))
                # Enforce future candidates to hit this correction set.
                if correction:
                    with stats.time("block"):
                        master.add_clause(correction)
                        if mus_master is not None:
                            mus_master.add_clause(correction)

                try:
                    csolver.increment_MSS()
                except AttributeError:
                    pass

                print(format_output("S", mss, time.perf_counter() - start, args, n), flush=True)
                outputs_since_mus += 1
            else:
                with stats.time("shrink"):
                    mus = csolver.shrink(seed)
                if mus is None:
                    continue
                mus = sorted(set(mus))

                # Block all supersets of this MUS in the hitting-set master.
                with stats.time("block"):
                    master.add_clause([-i for i in mus])
                    if mus_master is not None:
                        mus_master.add_clause([-i for i in mus])

                try:
                    csolver.increment_MUS()
                except AttributeError:
                    pass

                print(format_output("U", mus, time.perf_counter() - start, args, n), flush=True)
                outputs_since_mus = 0

            outputs += 1
            if args.limit and outputs >= args.limit:
                sys.stderr.write("Result limit reached.\n")
                break

        if args.stats:
            print_stats(stats)

    except utils.ExecutableException as exc:
        sys.stderr.write("ERROR: %s\n" % str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
