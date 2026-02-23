#!/usr/bin/env python3
"""
Paper-style analysis for MARCO SAT11 benchmark results.

Inputs:
  - run-level CSV (default: bench/results/all_runs.csv)

Outputs in analysis directory:
  - method_summary.csv
  - pairwise_vs_<baseline>.csv
  - family_summary.csv
  - report.md
  - cactus_solved_vs_time.png
  - performance_profile.png
  - scatter_vs_<baseline>.png
  - solved_by_family.png
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Keep matplotlib cache inside workspace to avoid ~/.matplotlib permission issues.
SCRIPT_DIR = Path(__file__).resolve().parent
TMP_BASE = Path(os.environ.get("TMPDIR", "/tmp")) / "marco_mpl_cache"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(TMP_BASE / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMP_BASE / "xdg_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
import matplotlib.pyplot as plt  # noqa: E402


def safe_geomean(values: np.ndarray) -> Optional[float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return None
    return float(np.exp(np.mean(np.log(vals))))


def df_to_markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"

    def fmt_cell(v: object) -> str:
        if v is None:
            return ""
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return "nan"
            return format(float(v), floatfmt)
        return str(v)

    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(fmt_cell(row[c]) for c in cols) + " |")
    return "\n".join([header, sep] + rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-style analysis + plots for MARCO SAT11 benchmark CSV.")
    parser.add_argument(
        "--input-csv",
        default=str(SCRIPT_DIR / "results" / "all_runs.csv"),
        help="Path to consolidated run-level CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR / "results" / "analysis"),
        help="Directory for analysis artifacts.",
    )
    parser.add_argument(
        "--baseline",
        default="marco",
        help="Baseline method for pairwise analysis.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=-1.0,
        help="Timeout in seconds. If <=0, infer from timed-out rows.",
    )
    parser.add_argument(
        "--par-k",
        type=float,
        default=2.0,
        help="PAR-k penalty multiplier for unsolved instances (default: 2).",
    )
    return parser.parse_args()


def infer_timeout(df: pd.DataFrame) -> float:
    timed = df[df["timed_out"] == True]  # noqa: E712
    if timed.empty:
        raise ValueError("Cannot infer timeout: no timed-out rows in dataset. Pass --timeout-s explicitly.")
    return float(np.median(timed["elapsed_s"].to_numpy()))


def load_runs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "instance",
        "method",
        "elapsed_s",
        "success",
        "timed_out",
        "completed",
        "valid_mus",
        "valid_mcs",
        "outputs_count",
        "mus_count",
        "mcs_count",
        "first_mus_s",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    # Normalize booleans in case CSV has strings.
    for col in ["success", "timed_out", "completed", "valid_mus", "valid_mcs"]:
        if df[col].dtype != bool:
            df[col] = df[col].astype(str).str.lower().isin({"1", "true", "yes", "y"})

    # keep one run per (instance, method) if user accidentally merged repeats
    dup = df.groupby(["instance", "method"]).size()
    if (dup > 1).any():
        # Keep the first run deterministically.
        df = df.sort_values(["instance", "method", "run_id"]).drop_duplicates(["instance", "method"], keep="first")

    return df


def build_method_summary(df: pd.DataFrame, timeout_s: float, par_k: float) -> pd.DataFrame:
    methods = sorted(df["method"].unique())
    n_instances = int(df["instance"].nunique())
    rows: List[Dict[str, object]] = []

    for m in methods:
        d = df[df["method"] == m].copy()
        solved_mask = d["completed"] == True  # noqa: E712
        solved = int(solved_mask.sum())
        timeout = int((d["timed_out"] == True).sum())  # noqa: E712

        capped = np.where(solved_mask, d["elapsed_s"].to_numpy(), timeout_s)
        par = np.where(solved_mask, d["elapsed_s"].to_numpy(), par_k * timeout_s)

        fm = d.loc[solved_mask, "first_mus_s"].dropna().to_numpy(dtype=float)
        rows.append(
            {
                "method": m,
                "instances": n_instances,
                "solved": solved,
                "solved_pct": solved / float(n_instances),
                "timeout": timeout,
                "median_solved_s": float(np.median(d.loc[solved_mask, "elapsed_s"])) if solved > 0 else np.nan,
                "p90_solved_s": float(np.quantile(d.loc[solved_mask, "elapsed_s"], 0.9)) if solved > 0 else np.nan,
                "mean_capped_s": float(np.mean(capped)),
                "par_s": float(np.mean(par)),
                "median_outputs": float(np.median(d.loc[solved_mask, "outputs_count"])) if solved > 0 else np.nan,
                "median_mus": float(np.median(d.loc[solved_mask, "mus_count"])) if solved > 0 else np.nan,
                "median_mcs": float(np.median(d.loc[solved_mask, "mcs_count"])) if solved > 0 else np.nan,
                "median_first_mus_s": float(np.median(fm)) if fm.size > 0 else np.nan,
            }
        )
    out = pd.DataFrame(rows).sort_values(["solved", "par_s"], ascending=[False, True]).reset_index(drop=True)
    return out


def build_pairwise(df: pd.DataFrame, baseline: str, timeout_s: float) -> pd.DataFrame:
    if baseline not in set(df["method"]):
        raise ValueError(f"Baseline '{baseline}' not found in methods: {sorted(df['method'].unique())}")

    methods = sorted(m for m in df["method"].unique() if m != baseline)
    base = df[df["method"] == baseline][["instance", "completed", "elapsed_s"]].rename(
        columns={"completed": "completed_base", "elapsed_s": "time_base"}
    )

    rows: List[Dict[str, object]] = []
    for m in methods:
        cur = df[df["method"] == m][["instance", "completed", "elapsed_s"]].rename(
            columns={"completed": "completed_cur", "elapsed_s": "time_cur"}
        )
        j = base.merge(cur, on="instance", how="inner")

        solved_both = (j["completed_base"] == True) & (j["completed_cur"] == True)  # noqa: E712
        base_only = (j["completed_base"] == True) & (j["completed_cur"] == False)  # noqa: E712
        cur_only = (j["completed_base"] == False) & (j["completed_cur"] == True)  # noqa: E712

        wins = int((j.loc[solved_both, "time_cur"] < j.loc[solved_both, "time_base"]).sum())
        losses = int((j.loc[solved_both, "time_cur"] > j.loc[solved_both, "time_base"]).sum())
        ties = int(solved_both.sum()) - wins - losses

        sp = (j.loc[solved_both, "time_base"] / j.loc[solved_both, "time_cur"]).to_numpy(dtype=float)
        gm = safe_geomean(sp)

        # Censored speedup on all instances.
        t_base = np.where(j["completed_base"], j["time_base"], timeout_s)
        t_cur = np.where(j["completed_cur"], j["time_cur"], timeout_s)
        gm_cens = safe_geomean(t_base / t_cur)

        rows.append(
            {
                "baseline": baseline,
                "method": m,
                "instances": int(len(j)),
                "solved_by_both": int(solved_both.sum()),
                "solved_baseline_only": int(base_only.sum()),
                "solved_method_only": int(cur_only.sum()),
                "wins_on_common": wins,
                "ties_on_common": ties,
                "losses_on_common": losses,
                "geo_speedup_common": gm if gm is not None else np.nan,
                "geo_speedup_capped_all": gm_cens if gm_cens is not None else np.nan,
                "median_speedup_common": float(np.median(sp)) if sp.size > 0 else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("geo_speedup_capped_all", ascending=False).reset_index(drop=True)


def build_family_summary(df: pd.DataFrame, timeout_s: float) -> pd.DataFrame:
    d = df.copy()
    d["family"] = d["instance"].str.split("/").str[0]
    families = sorted(d["family"].unique())
    methods = sorted(d["method"].unique())

    rows: List[Dict[str, object]] = []
    for fam in families:
        fam_df = d[d["family"] == fam]
        n = int(fam_df["instance"].nunique())
        for m in methods:
            mm = fam_df[fam_df["method"] == m]
            solved_mask = mm["completed"] == True  # noqa: E712
            solved = int(solved_mask.sum())
            capped = np.where(solved_mask, mm["elapsed_s"].to_numpy(), timeout_s)
            rows.append(
                {
                    "family": fam,
                    "method": m,
                    "instances": n,
                    "solved": solved,
                    "solved_pct": solved / float(n) if n else np.nan,
                    "median_solved_s": float(np.median(mm.loc[solved_mask, "elapsed_s"])) if solved > 0 else np.nan,
                    "mean_capped_s": float(np.mean(capped)) if len(mm) else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["family", "solved"], ascending=[True, False]).reset_index(drop=True)


def plot_cactus(df: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(8.5, 6))
    for method in sorted(df["method"].unique()):
        vals = np.sort(df.loc[(df["method"] == method) & (df["completed"] == True), "elapsed_s"].to_numpy())  # noqa: E712
        if vals.size == 0:
            continue
        x = np.arange(1, vals.size + 1)
        plt.step(x, vals, where="post", label=method)
    plt.yscale("log")
    plt.xlabel("Solved instances")
    plt.ylabel("Runtime (s, log scale)")
    plt.title("Cactus Plot (SAT11)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_performance_profile(df: pd.DataFrame, timeout_s: float, out_png: Path) -> None:
    methods = sorted(df["method"].unique())
    pivot = df.pivot(index="instance", columns="method", values="elapsed_s")
    solved = df.pivot(index="instance", columns="method", values="completed")

    # Censor unsolved to timeout for profile.
    times = pivot.copy()
    for m in methods:
        times[m] = np.where(solved[m], times[m], timeout_s)

    best = times.min(axis=1)
    ratios = times.div(best, axis=0)
    rmax = float(np.nanmax(ratios.to_numpy()))
    rmax = max(2.0, min(rmax * 1.05, 1e4))
    taus = np.logspace(0.0, math.log10(rmax), 250)

    plt.figure(figsize=(8.5, 6))
    n = ratios.shape[0]
    for m in methods:
        vals = ratios[m].to_numpy(dtype=float)
        prof = [(vals <= t).sum() / float(n) for t in taus]
        plt.plot(taus, prof, label=m)

    plt.xscale("log")
    plt.ylim(0, 1.01)
    plt.xlabel("Slowdown factor vs best on instance (tau, log scale)")
    plt.ylabel("Fraction of instances solved within tau")
    plt.title("Performance Profile (censored at timeout)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_scatter_vs_baseline(df: pd.DataFrame, baseline: str, timeout_s: float, out_png: Path) -> None:
    methods = sorted(m for m in df["method"].unique() if m != baseline)
    base = df[df["method"] == baseline][["instance", "completed", "elapsed_s"]].rename(
        columns={"completed": "completed_base", "elapsed_s": "time_base"}
    )

    n = len(methods)
    cols = 2
    rows = int(math.ceil(n / float(cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4.3 * rows), squeeze=False)

    for idx, method in enumerate(methods):
        ax = axes[idx // cols][idx % cols]
        cur = df[df["method"] == method][["instance", "completed", "elapsed_s"]].rename(
            columns={"completed": "completed_cur", "elapsed_s": "time_cur"}
        )
        j = base.merge(cur, on="instance", how="inner")

        xb = np.where(j["completed_base"], j["time_base"], timeout_s)
        yb = np.where(j["completed_cur"], j["time_cur"], timeout_s)

        both = (j["completed_base"] == True) & (j["completed_cur"] == True)  # noqa: E712
        base_only = (j["completed_base"] == True) & (j["completed_cur"] == False)  # noqa: E712
        cur_only = (j["completed_base"] == False) & (j["completed_cur"] == True)  # noqa: E712
        none = (j["completed_base"] == False) & (j["completed_cur"] == False)  # noqa: E712

        ax.scatter(xb[both], yb[both], s=10, alpha=0.7, label="both solved")
        ax.scatter(xb[base_only], yb[base_only], s=14, alpha=0.8, marker="x", label=f"{baseline} only")
        ax.scatter(xb[cur_only], yb[cur_only], s=14, alpha=0.8, marker="+", label=f"{method} only")
        ax.scatter(xb[none], yb[none], s=8, alpha=0.25, marker=".", label="both timeout")

        lo = min(np.min(xb), np.min(yb), 1e-3)
        hi = max(np.max(xb), np.max(yb), timeout_s)
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color="black")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"{baseline} runtime (s, capped)")
        ax.set_ylabel(f"{method} runtime (s, capped)")
        ax.set_title(f"{method} vs {baseline}")
        ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.4)

    # Hide unused axes.
    total_axes = rows * cols
    for idx in range(n, total_axes):
        axes[idx // cols][idx % cols].axis("off")

    # Shared legend from first populated axis.
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)))
    fig.suptitle(f"Pairwise Scatter vs {baseline} (timeout-capped)", y=0.995)
    fig.tight_layout(rect=[0, 0.05, 1, 0.98])
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_solved_by_family(family_df: pd.DataFrame, out_png: Path) -> None:
    # Restrict to top 10 families by size to keep chart readable.
    fam_sizes = family_df[["family", "instances"]].drop_duplicates().sort_values("instances", ascending=False)
    top_fams = fam_sizes.head(10)["family"].tolist()
    d = family_df[family_df["family"].isin(top_fams)].copy()

    methods = sorted(d["method"].unique())
    x = np.arange(len(top_fams))
    width = 0.14 if len(methods) >= 5 else 0.18

    plt.figure(figsize=(11.5, 6.2))
    for i, m in enumerate(methods):
        mm = d[d["method"] == m].set_index("family").reindex(top_fams)
        plt.bar(x + (i - (len(methods) - 1) / 2.0) * width, mm["solved"].to_numpy(), width=width, label=m)

    plt.xticks(x, top_fams, rotation=35, ha="right")
    plt.ylabel("Solved instances")
    plt.xlabel("Family")
    plt.title("Solved Count by Family (top 10 families by size)")
    plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def build_report(
    method_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    family_df: pd.DataFrame,
    out_md: Path,
    timeout_s: float,
    par_k: float,
) -> None:
    best = method_df.iloc[0]
    lines: List[str] = []
    lines.append("# MARCO SAT11 Paper-Style Analysis")
    lines.append("")
    lines.append(f"- Instances: {int(method_df['instances'].iloc[0])}")
    lines.append(f"- Methods: {', '.join(method_df['method'].tolist())}")
    lines.append(f"- Timeout: {timeout_s:.1f}s")
    lines.append(f"- Penalized score: PAR{par_k:g}")
    lines.append("")
    lines.append("## Method Ranking (by solved count, then PAR)")
    lines.append("")
    lines.append(df_to_markdown_table(method_df, floatfmt=".4f"))
    lines.append("")
    lines.append("## Pairwise vs Baseline")
    lines.append("")
    if not pairwise_df.empty:
        lines.append(df_to_markdown_table(pairwise_df, floatfmt=".4f"))
    else:
        lines.append("(no challenger methods)")
    lines.append("")
    lines.append("## Best Overall")
    lines.append("")
    lines.append(
        f"- Best solved count: `{best['method']}` with `{int(best['solved'])}/{int(best['instances'])}` "
        f"({100.0*float(best['solved_pct']):.2f}%)."
    )
    lines.append(
        f"- Lowest PAR{par_k:g}: `{method_df.sort_values('par_s', ascending=True).iloc[0]['method']}` "
        f"({method_df.sort_values('par_s', ascending=True).iloc[0]['par_s']:.2f}s)."
    )
    lines.append("")
    lines.append("## Family Snapshot (top 20 rows)")
    lines.append("")
    lines.append(df_to_markdown_table(family_df.head(20), floatfmt=".4f"))
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    lines.append("- `cactus_solved_vs_time.png`")
    lines.append("- `performance_profile.png`")
    lines.append("- `scatter_vs_<baseline>.png`")
    lines.append("- `solved_by_family.png`")
    lines.append("")

    out_md.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_runs(input_csv)
    timeout_s = float(args.timeout_s) if args.timeout_s > 0 else infer_timeout(df)

    method_df = build_method_summary(df, timeout_s=timeout_s, par_k=args.par_k)
    pairwise_df = build_pairwise(df, baseline=args.baseline, timeout_s=timeout_s)
    family_df = build_family_summary(df, timeout_s=timeout_s)

    method_csv = out_dir / "method_summary.csv"
    pair_csv = out_dir / f"pairwise_vs_{args.baseline}.csv"
    fam_csv = out_dir / "family_summary.csv"
    method_df.to_csv(method_csv, index=False)
    pairwise_df.to_csv(pair_csv, index=False)
    family_df.to_csv(fam_csv, index=False)

    plot_cactus(df, out_dir / "cactus_solved_vs_time.png")
    plot_performance_profile(df, timeout_s=timeout_s, out_png=out_dir / "performance_profile.png")
    plot_scatter_vs_baseline(df, baseline=args.baseline, timeout_s=timeout_s, out_png=out_dir / f"scatter_vs_{args.baseline}.png")
    plot_solved_by_family(family_df, out_png=out_dir / "solved_by_family.png")

    report_md = out_dir / "paper_style_report.md"
    build_report(
        method_df=method_df,
        pairwise_df=pairwise_df,
        family_df=family_df,
        out_md=report_md,
        timeout_s=timeout_s,
        par_k=args.par_k,
    )

    print(f"Input: {input_csv}")
    print(f"Timeout used: {timeout_s:.3f}s")
    print(f"Wrote: {method_csv}")
    print(f"Wrote: {pair_csv}")
    print(f"Wrote: {fam_csv}")
    print(f"Wrote: {report_md}")
    print(f"Wrote: {out_dir / 'cactus_solved_vs_time.png'}")
    print(f"Wrote: {out_dir / 'performance_profile.png'}")
    print(f"Wrote: {out_dir / f'scatter_vs_{args.baseline}.png'}")
    print(f"Wrote: {out_dir / 'solved_by_family.png'}")


if __name__ == "__main__":
    main()
