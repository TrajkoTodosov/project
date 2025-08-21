#!/usr/bin/env python
# scripts/final/21_plot_results.py
# Per-experiment plots (runs + mean ± std) with best-epoch marker (ρ only),
# and a 4-way comparison plot across experiments.

import os, sys, glob, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- helpers ----------
def load_runs_and_mean(folder: str, stem: str):
    """
    folder: path containing run files and the *_MEAN.xlsx file
    stem:   base name like 'DEAM_cv', 'PMEmo_cv', 'BOTH_FULL_to_DEAM', 'BOTH_FULL_to_PMEmo'
    returns: dict with keys: 'runs' (list of dfs), 'mean' (df or None), 'mean_path'
    """
    pattern_runs = os.path.join(folder, f"{stem}_run*.xlsx")
    run_files = sorted(glob.glob(pattern_runs))
    runs = []
    for rf in run_files:
        try:
            runs.append(pd.read_excel(rf))
        except Exception as e:
            print(f"[WARN] Could not read run file {rf}: {e}")

    mean_path = os.path.join(folder, f"{stem}_MEAN.xlsx")
    mean_df = None
    if os.path.isfile(mean_path):
        try:
            mean_df = pd.read_excel(mean_path)
        except Exception as e:
            print(f"[WARN] Could not read mean file {mean_path}: {e}")

    return {"runs": runs, "mean": mean_df, "mean_path": mean_path}

def plot_single_experiment(folder: str, stem: str, title: str = None, save_path: str = None,
                           mark_best: bool = True):
    data = load_runs_and_mean(folder, stem)
    runs, mean_df = data["runs"], data["mean"]

    if not runs and mean_df is None:
        raise FileNotFoundError(f"No runs or mean file found in {folder} with stem {stem}")

    plt.figure(figsize=(8.5, 5.2), dpi=150)

    # plot individual runs (if any)
    for df in runs:
        if "epoch" in df.columns and "test_rho" in df.columns:
            plt.plot(df["epoch"], df["test_rho"], linewidth=1, alpha=0.25, color="gray")

    # plot mean ± std if available
    best_info = None
    if mean_df is not None and {"epoch","mean_test_rho"}.issubset(mean_df.columns):
        x = mean_df["epoch"].values
        y = mean_df["mean_test_rho"].values
        line, = plt.plot(x, y, linewidth=2.6, label="Mean", zorder=3)
        color = line.get_color()

        if "std_test_rho" in mean_df.columns:
            s = mean_df["std_test_rho"].values
            plt.fill_between(x, y - s, y + s, alpha=0.15, label="±1 std", color=color)

        if mark_best and len(y) > 0:
            i_best = int(np.argmax(y))
            x_best, y_best = x[i_best], y[i_best]
            # marker + annotation
            plt.scatter([x_best], [y_best], color="red", zorder=5)
            label = f"Best epoch = {x_best} (ρ = {y_best:.4f})"
            plt.annotate(label, xy=(x_best, y_best),
                         xytext=(x_best, y_best + 0.015),
                         ha="center", va="bottom",
                         fontsize=9,
                         arrowprops=dict(arrowstyle="->", lw=0.8, color="red"))
            best_info = (x_best, y_best)

    plt.xlabel("Epoch")
    plt.ylabel("Dynamic Pearson ρ")
    if title is None:
        title = stem.replace("_", " ")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="lower right")

    if save_path is None:
        save_path = os.path.join(folder, f"{stem}_plot.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[saved plot] {save_path}")

    if best_info:
        x_best, y_best = best_info
        print(f"Best epoch {x_best}: ρ={y_best:.4f}")

def plot_comparison_four(root_runs_dir: str, save_name: str = "comparison_four.png"):
    """
    Expects a timestamped runs dir like:
      results/runs_YYYYMMDD_HHMMSS/
         DEAM_cv/DEAM_cv_MEAN.xlsx
         PMEmo_cv/PMEmo_cv_MEAN.xlsx
         BOTH_to_DEAM/BOTH_FULL_to_DEAM_MEAN.xlsx
         BOTH_to_PMEmo/BOTH_FULL_to_PMEmo_MEAN.xlsx
    """
    deam_dir  = os.path.join(root_runs_dir, "DEAM_cv")
    pmemo_dir = os.path.join(root_runs_dir, "PMEmo_cv")
    b2d_dir   = os.path.join(root_runs_dir, "BOTH_to_DEAM")
    b2p_dir   = os.path.join(root_runs_dir, "BOTH_to_PMEmo")

    items = [
        ("DEAM_cv",            deam_dir,  "DEAM CV"),
        ("PMEmo_cv",           pmemo_dir, "PMEmo CV"),
        ("BOTH_FULL_to_DEAM",  b2d_dir,   "Both→DEAM"),
        ("BOTH_FULL_to_PMEmo", b2p_dir,   "Both→PMEmo"),
    ]

    plt.figure(figsize=(9.8, 5.6), dpi=150)
    any_plotted = False

    for stem, folder, pretty in items:
        mean_path = os.path.join(folder, f"{stem}_MEAN.xlsx")
        if not os.path.isfile(mean_path):
            print(f"[WARN] Missing mean file: {mean_path} (skipping {pretty})")
            continue

        try:
            df = pd.read_excel(mean_path)
        except Exception as e:
            print(f"[WARN] Could not read {mean_path}: {e}")
            continue

        if not {"epoch","mean_test_rho"}.issubset(df.columns):
            print(f"[WARN] {mean_path} missing required columns (epoch, mean_test_rho)")
            continue

        line, = plt.plot(df["epoch"], df["mean_test_rho"], linewidth=2.6, label=pretty)
        color = line.get_color()
        if "std_test_rho" in df.columns:
            x = df["epoch"].values; y = df["mean_test_rho"].values; s = df["std_test_rho"].values
            plt.fill_between(x, y - s, y + s, alpha=0.12, color=color)
        any_plotted = True

    plt.xlabel("Epoch")
    plt.ylabel("Dynamic Pearson ρ")
    plt.title("Comparison across training regimes")
    plt.grid(True, linestyle="--", alpha=0.3)
    if any_plotted:
        plt.legend(loc="lower right")

    out_path = os.path.join(root_runs_dir, save_name)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[saved comparison] {out_path}")

# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(description="Plot per-experiment curves (best epoch marked) and/or a 4-way comparison.")
    # Single experiment mode
    p.add_argument("--dir", type=str, help="Folder with run*.xlsx + *_MEAN.xlsx (e.g. .../PMEmo_cv)")
    p.add_argument("--stem", type=str, help="Base filename stem (PMEmo_cv, DEAM_cv, BOTH_FULL_to_DEAM, BOTH_FULL_to_PMEmo)")
    p.add_argument("--title", type=str, default=None, help="Optional custom title")
    p.add_argument("--save", type=str, default=None, help="Optional output path for the single plot")
    p.add_argument("--no_mark_best", action="store_true", help="Disable marking best epoch")

    # Comparison mode
    p.add_argument("--compare_root", type=str, help="runs_YYYYMMDD_HHMMSS folder to overlay 4 experiments in one plot")
    p.add_argument("--compare_save", type=str, default="comparison_four.png", help="filename to save comparison plot")

    args = p.parse_args()

    did = False
    if args.dir and args.stem:
        plot_single_experiment(
            args.dir, args.stem, title=args.title, save_path=args.save,
            mark_best=(not args.no_mark_best)
        )
        did = True

    if args.compare_root:
        plot_comparison_four(args.compare_root, save_name=args.compare_save)
        did = True

    if not did:
        print("Nothing to do. Examples:\n"
              "  python scripts/final/21_plot_results.py --dir results/runs_20250818_133711/PMEmo_cv --stem PMEmo_cv --title 'PMEmo CV (10 runs)'\n"
              "  python scripts/final/21_plot_results.py --compare_root results/runs_20250818_133711")

if __name__ == "__main__":
    main()
