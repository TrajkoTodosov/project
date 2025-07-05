# scripts/filter_lyrics.py

import os
import pandas as pd
from langdetect import detect, DetectorFactory

# ─── CONFIG ──────────────────────────────────────────────────────────────
RAW_TEXT_DIR   = os.path.join("data", "processed", "lyrics_text")
OUT_INFO_PATH  = os.path.join("data", "processed", "lyrics_info.csv")
WORD_THRESHOLD = 5  # min words to count as 'has lyrics'

# reproducible language detection
DetectorFactory.seed = 0

def main():
    records = []
    for fn in os.listdir(RAW_TEXT_DIR):
        if not fn.lower().endswith(".txt"):
            continue
        song_id = os.path.splitext(fn)[0]
        text = open(os.path.join(RAW_TEXT_DIR, fn), encoding="utf-8").read().strip()
        words = text.split()
        wc = len(words)
        has_lyrics = wc >= WORD_THRESHOLD

        if has_lyrics:
            try:
                lang = detect(text)
            except:
                lang = "unknown"
        else:
            lang = ""

        records.append({
            "song_id":    song_id,
            "word_count": wc,
            "has_lyrics": has_lyrics,
            "language":   lang
        })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(OUT_INFO_PATH), exist_ok=True)
    df.to_csv(OUT_INFO_PATH, index=False)
    print(f" Saved lyrics info for {len(df)} songs to {OUT_INFO_PATH}")

if __name__ == "__main__":
    main()
