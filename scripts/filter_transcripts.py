# scripts/filter_transcripts.py

import os
import pandas as pd
import shutil

# ─── PATH CONFIG ─────────────────────────────────────────────────────────
LYRICS_INFO_CSV      = os.path.join("data", "processed", "lyrics_info.csv")
RAW_TEXT_DIR         = os.path.join("data", "processed", "lyrics_text")
RAW_TOKEN_DIR        = os.path.join("data", "processed", "lyrics_tokens")
FILTERED_TEXT_DIR    = os.path.join("data", "processed", "filtered_lyrics_text")
FILTERED_TOKEN_DIR   = os.path.join("data", "processed", "filtered_lyrics_tokens")
FILTERED_IDS_CSV     = os.path.join("data", "processed", "filtered_lyrics_ids.csv")

# ─── CREATE OUTPUT DIRECTORIES ────────────────────────────────────────────
os.makedirs(FILTERED_TEXT_DIR, exist_ok=True)
os.makedirs(FILTERED_TOKEN_DIR, exist_ok=True)

# ─── LOAD LYRICS INFO ─────────────────────────────────────────────────────
info = pd.read_csv(LYRICS_INFO_CSV)

# ─── SELECT VALID SONGS ───────────────────────────────────────────────────
# Keep only those with has_lyrics=True and a known language
valid_songs = info[
    (info['has_lyrics'] == True) &
    (info['language'].notna()) &
    (info['language'] != 'unknown') &
    (info['language'] != '')
]['song_id'].astype(str).tolist()

# ─── FILTER TEXT AND TOKEN FILES ──────────────────────────────────────────
for song_id in valid_songs:
    txt_src = os.path.join(RAW_TEXT_DIR, f"{song_id}.txt")
    tok_src = os.path.join(RAW_TOKEN_DIR, f"{song_id}.json")
    if os.path.exists(txt_src):
        shutil.copy(txt_src, os.path.join(FILTERED_TEXT_DIR, f"{song_id}.txt"))
    if os.path.exists(tok_src):
        shutil.copy(tok_src, os.path.join(FILTERED_TOKEN_DIR, f"{song_id}.json"))

# ─── SAVE LIST OF VALID SONG IDs ──────────────────────────────────────────
pd.DataFrame({'song_id': valid_songs}).to_csv(FILTERED_IDS_CSV, index=False)

print(f"✅ Filtered lyrics for {len(valid_songs)} songs.")
print(f"   • Text files in: {FILTERED_TEXT_DIR}")
print(f"   • Token files in: {FILTERED_TOKEN_DIR}")
print(f"   • Song IDs in:  {FILTERED_IDS_CSV}")
