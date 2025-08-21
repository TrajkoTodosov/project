#!/usr/bin/env python

import os
import sys
import glob
import numpy as np
import pandas as pd

# ── Make project root importable (../.. from scripts/final/) ─────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

# import dataset loaders from scripts/
from scripts.dataset import DEAMMultimodalDataset
from scripts.pmemo_dataset import PMEmoDataset


def check_exists(path, kind="file"):
    if kind == "file" and not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")
    if kind == "dir" and not os.path.isdir(path):
        raise FileNotFoundError(f"Missing directory: {path}")


def check_mels(folder, expect_first_dim=128):
    npys = sorted(glob.glob(os.path.join(folder, "*.npy")))
    if not npys:
        raise FileNotFoundError(f"No spectrogram .npy files found in {folder}")
    arr = np.load(npys[0])
    print(f"  example mel: {os.path.basename(npys[0])} shape={arr.shape}")
    if arr.ndim != 2 or arr.shape[0] != expect_first_dim:
        print("  [WARN] Expected mel to be [128, T]. Your shape differs.")
    return arr.shape


def main():
    print("=== Sanity checks: paths & basic shapes ===")

    # --- DEAM paths ---
    deam_ids_csv      = os.path.join(PROJECT_ROOT, "data", "processed", "train_ids.csv")
    deam_spec_dir     = os.path.join(PROJECT_ROOT, "data", "processed", "spectrograms")
    deam_feats_dir    = os.path.join(PROJECT_ROOT, "data", "raw", "features")
    deam_labels_dir   = os.path.join(PROJECT_ROOT, "data", "processed", "filtered_avg")

    # --- PMEmo paths ---
    pmemo_feat_csv    = os.path.join(PROJECT_ROOT, "PMEmo", "dynamic_features_aligned_SHARED.csv")
    pmemo_annot_csv   = os.path.join(PROJECT_ROOT, "PMEmo", "dynamic_annotations.csv")
    pmemo_spec_dir    = os.path.join(PROJECT_ROOT, "PMEmo", "spectrograms")

    # Check DEAM existence
    check_exists(deam_ids_csv, "file")
    check_exists(deam_spec_dir, "dir")
    check_exists(deam_feats_dir, "dir")
    check_exists(deam_labels_dir, "dir")

    # Check PMEmo existence
    check_exists(pmemo_feat_csv, "file")
    check_exists(pmemo_annot_csv, "file")
    check_exists(pmemo_spec_dir, "dir")

    print("\n[DEAM] files/folders OK")
    deam_ids = pd.read_csv(deam_ids_csv, dtype=str).iloc[:, 0].str.strip().tolist()
    print(f"  train_ids.csv count: {len(deam_ids)}")
    print("  checking spectrograms:")
    _ = check_mels(deam_spec_dir)  # prints example

    # Peek DEAM feature columns from one CSV
    any_deam_csv = sorted(glob.glob(os.path.join(deam_feats_dir, "*.csv")))
    if any_deam_csv:
        dcols = pd.read_csv(any_deam_csv[0], sep=";").columns.tolist()
        print(f"  DEAM feature columns example ({os.path.basename(any_deam_csv[0])}): {len(dcols)} cols")

    # Try loading a tiny DEAM dataset sample
    try:
        deam_ds = DEAMMultimodalDataset(deam_ids[:5])
        sample = deam_ds[0]
        print("  DEAM sample shapes:")
        print(f"    audio      {tuple(sample['audio'].shape)}  (expect [T,128])")
        print(f"    raw_feats  {tuple(sample['raw_feats'].shape)}")
        print(f"    labels     {tuple(sample['labels'].shape)}")
    except Exception as e:
        print("  [ERROR] Failed to instantiate DEAMMultimodalDataset:", repr(e))

    print("\n[PMEmo] files/folders OK")
    print("  checking spectrograms:")
    _ = check_mels(pmemo_spec_dir)  # prints example

    # Peek PMEmo aligned feature columns
    pcols = pd.read_csv(pmemo_feat_csv).columns.tolist()
    print(f"  PMEmo aligned feature columns: {len(pcols)} cols")

    # Try loading a tiny PMEmo dataset sample
    try:
        pmemo_ds = PMEmoDataset(
            features_csv=pmemo_feat_csv,
            annotations_csv=pmemo_annot_csv,
            target="arousal",
        )
        print(f"  PMEmo songs: {len(pmemo_ds)}")
        ps = pmemo_ds[0]
        print("  PMEmo sample shapes:")
        print(f"    audio      {tuple(ps['audio'].shape)}  (expect [T,128])")
        print(f"    raw_feats  {tuple(ps['raw_feats'].shape)}")
        print(f"    labels     {tuple(ps['labels'].shape)}")
    except Exception as e:
        print("  [ERROR] Failed to instantiate PMEmoDataset:", repr(e))

    print("\nAll basic checks attempted. If there were no [ERROR] lines above, you're good to train.")


if __name__ == "__main__":
    main()
