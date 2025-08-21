#!/usr/bin/env python
import os, glob, argparse, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="Pattern to match CV run files, e.g. results/*/DEAM_cv/DEAM_cv_run*.xlsx")
    ap.add_argument("--out",  required=False, default="", help="Output path for MEAN.xlsx (optional)")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths: raise FileNotFoundError("No files match pattern")
    dfs = [pd.read_excel(p) for p in paths]
    mean = pd.DataFrame({
        "epoch": dfs[0]["epoch"],
        "mean_test_rho": pd.concat([d["test_rho"] for d in dfs], axis=1).mean(1),
        "std_test_rho":  pd.concat([d["test_rho"] for d in dfs], axis=1).std(1),
    })
    out = args.out or os.path.join(os.path.dirname(paths[0]), "CV_MEAN.xlsx")
    mean.to_excel(out, index=False)
    print(f"Saved mean curve → {out}")

if __name__ == "__main__":
    main()
