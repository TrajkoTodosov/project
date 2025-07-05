# scripts/merge_metadata.py

import os
import pandas as pd
import numpy as np

# ─── CONFIG ──────────────────────────────────────────────────────────────
RAW_META_DIR  = os.path.join("data", "raw", "metadata")
OUT_META_PATH = os.path.join("data", "processed", "metadata.csv")

# Bins for segment position
BINS   = [0, 60, 120, 180, np.inf]
LABELS = ["0-1min", "1-2min", "2-3min", "3+min"]

# ─── HELPER TO PARSE "MM.SS" → seconds ────────────────────────────────────
def parse_mmss(x):
    try:
        s = str(x).strip()
        mm, ss = s.split(".")
        return int(mm) * 60 + float(ss)
    except:
        return np.nan

# ─── LOAD & NORMALIZE EACH YEAR'S FILE ───────────────────────────────────
all_dfs = []

for fname in os.listdir(RAW_META_DIR):
    path = os.path.join(RAW_META_DIR, fname)
    if not fname.lower().endswith(".csv"):
        continue

    if "2013" in fname:
        # metadata_2013.csv has 7 cols:
        df = pd.read_csv(
            path,
            usecols=[
                "song_id",
                "file_name",
                "Artist",
                "Song title",
                "start of the segment (min.sec)",
                "end of the segment (min.sec)",
                "Genre",
            ]
        ).rename(columns={
            "song_id": "song_id",
            "file_name": "file_name",
            "Artist": "artist",
            "Song title": "title",
            "start of the segment (min.sec)": "start_sec_raw",
            "end of the segment (min.sec)": "end_sec_raw",
            "Genre": "genre"
        })

    elif "2014" in fname:
        # metadata_2014.csv has 8 cols; we only need these:
        df = pd.read_csv(
            path,
            usecols=["Id", "Artist", "Track", "Genre", "segment start", "segment end"],
            engine="python",
        ).rename(columns={
            "Id": "song_id",
            "Artist": "artist",
            "Track": "title",
            "segment start": "start_sec_raw",
            "segment end": "end_sec_raw",
            "Genre": "genre"
        })
        # derive file_name from song_id
        df["file_name"] = df["song_id"].astype(str) + ".mp3"

    elif "2015" in fname:
        # metadata_2015.csv has no segment times:
        df = pd.read_csv(
            path,
            usecols=["id", "Filename", "title", "artist", "genre"],
            engine="python",
        ).rename(columns={
            "id": "song_id",
            "Filename": "file_name",
            "title": "title",
            "artist": "artist",
            "genre": "genre"
        })
        # add empty segment columns
        df["start_sec_raw"] = np.nan
        df["end_sec_raw"]   = np.nan

    else:
        continue

    all_dfs.append(df)

# ─── CONCAT & PARSE TIMES ────────────────────────────────────────────────
meta = pd.concat(all_dfs, ignore_index=True)

# parse MM.SS → seconds, only where not null
meta["start_sec"] = meta["start_sec_raw"].apply(parse_mmss)
meta["end_sec"]   = meta["end_sec_raw"].apply(parse_mmss)

# compute duration & position bin
meta["duration"] = meta["end_sec"] - meta["start_sec"]
meta["position_bin"] = pd.cut(meta["start_sec"], bins=BINS, labels=LABELS)

# drop the raw columns
meta = meta.drop(columns=["start_sec_raw", "end_sec_raw"])

# optionally, fill missing start/end with defaults:
# meta["start_sec"].fillna(0.0, inplace=True)
# meta["end_sec"].fillna(45.0, inplace=True)
# meta["duration"].fillna(45.0, inplace=True)
# meta["position_bin"].fillna("0-1min", inplace=True)

# ─── SAVE ────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_META_PATH), exist_ok=True)
meta.to_csv(OUT_META_PATH, index=False)
print(" Merged metadata saved to", OUT_META_PATH)
print(meta.head())
