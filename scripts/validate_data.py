#!/usr/bin/env python3
# validate_data.py

import os
import glob
import numpy as np
import pandas as pd

# ─── CONFIG ────────────────────────────────────────────────────────────────
RAW_FEAT_DIR   = os.path.join("data", "raw",       "features")
LABEL_DIR      = os.path.join("data", "processed", "filtered_avg")
METADATA_CSV   = os.path.join("data", "processed", "metadata.csv")
LOOP_FEAT_CSV  = os.path.join("data", "processed", "loop_features.csv")
TEXT_FEAT_CSV  = os.path.join("data", "processed", "text_features.csv")

# ─── HELPERS ───────────────────────────────────────────────────────────────
def report(name, problems):
    if not problems:
        print(f"✓ {name}: no issues found")
    else:
        print(f"✗ {name}:")
        for p in problems:
            print("   ", p)
    print()

# ─── 1) RAW FEATURES ───────────────────────────────────────────────────────
def check_raw_features():
    errs = []
    print("Checking raw CSV features…")
    for path in glob.glob(os.path.join(RAW_FEAT_DIR, "*.csv")):
        sid = os.path.basename(path)
        try:
            df = pd.read_csv(path, sep=";")
        except Exception as e:
            errs.append(f"{sid}: parse error: {e}")
            continue

        if "time" in df.columns:
            df = df.drop(columns=["time"])

        # try converting to float
        try:
            arr = df.values.astype(np.float64)
        except Exception as e:
            errs.append(f"{sid}: cannot cast to float: {e}")
            continue

        nans  = np.isnan(arr).sum()
        infs  = (~np.isfinite(arr)).sum()
        if nans>0:
            errs.append(f"{sid}: {nans} NaNs")
        if infs>0:
            errs.append(f"{sid}: {infs} infinite values")

    report("Raw features", errs)

# ─── 2) LABELS ─────────────────────────────────────────────────────────────
def check_labels():
    errs = []
    print("Checking averaged-arousal labels…")
    for path in glob.glob(os.path.join(LABEL_DIR, "*.csv")):
        sid = os.path.basename(path)
        try:
            # skip header row, take second row of numbers
            vals = pd.read_csv(path, header=None).iloc[1].astype(float).values
        except Exception as e:
            errs.append(f"{sid}: parse error: {e}")
            continue
        nans = np.isnan(vals).sum()
        infs = (~np.isfinite(vals)).sum()
        if nans>0 or infs>0:
            errs.append(f"{sid}: {nans} NaNs, {infs} infs")

    report("Labels", errs)

# ─── 3) METADATA ────────────────────────────────────────────────────────────
def check_metadata():
    errs = []
    print("Checking metadata…")
    try:
        df = pd.read_csv(METADATA_CSV, dtype=str)
    except Exception as e:
        return report("Metadata", [f"parse error: {e}"])

    # missing or non-numeric durations
    for idx, row in df.iterrows():
        sid, dur = row.get("song_id"), row.get("duration")
        if pd.isna(sid) or pd.isna(dur):
            errs.append(f"{sid}: missing song_id or duration")
            continue
        try:
            d = float(dur)
            if not np.isfinite(d):
                errs.append(f"{sid}: duration not finite ({dur})")
        except:
            errs.append(f"{sid}: duration not float ({dur})")

    report("Metadata durations", errs)

# ─── 4) LOOP FEATURES ───────────────────────────────────────────────────────
def check_loops():
    errs = []
    print("Checking loop_features…")
    try:
        df = pd.read_csv(LOOP_FEAT_CSV, dtype=str)
    except Exception as e:
        return report("Loop features", [f"parse error: {e}"])

    for idx, row in df.iterrows():
        sid = row.get("song_id")
        for col in ("loop_period","loop_strength","repeat_count"):
            val = row.get(col)
            if pd.isna(val):
                errs.append(f"{sid}: {col} missing")
                continue
            try:
                v = float(val)
                if not np.isfinite(v):
                    errs.append(f"{sid}: {col} infinite/NaN")
            except:
                errs.append(f"{sid}: {col} not float ({val})")

    report("Loop features", errs)

# ─── 5) TEXT EMBEDDINGS (OPTIONAL) ─────────────────────────────────────────
def check_text_feats():
    if not os.path.isfile(TEXT_FEAT_CSV):
        print("⚠️  text_features.csv not found, skipping")
        print()
        return
    errs = []
    print("Checking text_features.csv…")
    try:
        df = pd.read_csv(TEXT_FEAT_CSV, dtype=str)
    except Exception as e:
        return report("Text embeddings", [f"parse error: {e}"])
    for idx, row in df.iterrows():
        sid = row.get("song_id")
        vals = list(row.drop("song_id").values)
        try:
            arr = np.array(vals, dtype=float)
        except:
            errs.append(f"{sid}: cannot cast to float")
            continue
        if np.isnan(arr).any():
            errs.append(f"{sid}: {np.isnan(arr).sum()} NaNs")
        if not np.isfinite(arr).all():
            errs.append(f"{sid}: {(~np.isfinite(arr)).sum()} infs")
    report("Text embeddings", errs)

# ─── RUN ALL ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n===== DATA VALIDATION =====\n")
    check_raw_features()
    check_labels()
    check_metadata()
    check_loops()
    check_text_feats()
    print("All checks complete.\n")
