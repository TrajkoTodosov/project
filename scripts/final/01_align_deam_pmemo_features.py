#!/usr/bin/env python
import os, glob, pandas as pd

DEAM_DIR         = os.path.join("data", "raw", "features")
PMEMO_FEAT_IN    = os.path.join("PMEmo", "dynamic_features_aligned.csv")  # your current file
PMEMO_FEAT_OUT   = os.path.join("PMEmo", "dynamic_features_aligned_SHARED.csv")
DROP_FROM_PMEMO  = {"musicId", "frameTime"}     # keep for merge only, not as features
DROP_FROM_DEAM   = {"time"}

def main():
    deam_files = glob.glob(os.path.join(DEAM_DIR, "*.csv"))
    if not deam_files: raise FileNotFoundError(f"No DEAM feature files in {DEAM_DIR}")
    deam_df = pd.read_csv(deam_files[0], sep=";")
    deam_cols = [c for c in deam_df.columns if c not in DROP_FROM_DEAM]

    pm_df = pd.read_csv(PMEMO_FEAT_IN)
    pm_cols = [c for c in pm_df.columns if c not in DROP_FROM_PMEMO]

    shared = [c for c in deam_cols if c in pm_cols]
    print(f"DEAM feat cols: {len(deam_cols)}")
    print(f"PMEmo feat cols: {len(pm_cols)}")
    print(f"Shared feature cols: {len(shared)}")

    keep = ["musicId", "frameTime"] + shared
    out = pm_df[keep]
    out.to_csv(PMEMO_FEAT_OUT, index=False)
    print(f"Wrote aligned PMEmo → {PMEMO_FEAT_OUT}")

if __name__ == "__main__":
    main()
