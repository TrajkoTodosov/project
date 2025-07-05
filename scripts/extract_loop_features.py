# scripts/extract_loop_features.py

import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# ─── CONFIG ───────────────────────────────────────────────────────────────
RAW_AUDIO_DIR     = os.path.join("data", "raw", "audio")
OUT_LOOP_CSV      = os.path.join("data", "processed", "loop_features.csv")

# Librosa parameters
SR                = 22050            # sample rate
HOP_SEC           = 0.5              # analysis hop in seconds
HOP_LENGTH        = int(SR * HOP_SEC)
RECUR_WIDTH       = 3                # recurrence matrix width

# ─── HELPER ───────────────────────────────────────────────────────────────

def detect_loop_properties(y, sr, hop_length, width):
    """
    Returns: (loop_period_sec, loop_strength, repeat_count)
    - loop_period_sec: most prominent recurrence lag in seconds
    - loop_strength: average recurrence affinity along that diagonal
    - repeat_count: number of repeats in clip duration
    """
    # 1) compute chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    # 2) build recurrence matrix
    R = librosa.segment.recurrence_matrix(chroma, width=width, mode='affinity', sym=True)
    # 3) find lag with maximum sum of affinities
    affinity_sums = R.sum(axis=0)
    lag_idx = int(np.argmax(affinity_sums))
    # 4) compute loop period
    loop_period_sec = lag_idx * hop_length / sr
    # 5) compute loop strength as mean affinity on that diagonal
    # diagonal offset=lag_idx: entries R[i, i+lag_idx]
    diag_vals = np.diag(R, k=lag_idx)
    loop_strength = float(diag_vals.mean()) if diag_vals.size > 0 else 0.0
    # 6) clip duration
    clip_duration = y.shape[0] / sr
    # 7) repeat count
    repeat_count = clip_duration / loop_period_sec if loop_period_sec > 0 else 1.0
    return loop_period_sec, loop_strength, repeat_count

# ─── MAIN ─────────────────────────────────────────────────────────────────

def main():
    records = []
    audio_files = [f for f in os.listdir(RAW_AUDIO_DIR)
                   if f.lower().endswith(('.wav', '.mp3'))]
    print(f"Found {len(audio_files)} audio clips for loop extraction.")

    for fname in tqdm(audio_files, desc="Loop features"):
        song_id = os.path.splitext(fname)[0]
        path = os.path.join(RAW_AUDIO_DIR, fname)
        try:
            y, sr = librosa.load(path, sr=SR)
            period, strength, count = detect_loop_properties(y, sr, HOP_LENGTH, RECUR_WIDTH)
            records.append({
                'song_id': song_id,
                'loop_period': round(period, 3),
                'loop_strength': round(strength, 5),
                'repeat_count': round(count, 2)
            })
        except Exception as e:
            print(f"⚠️  Failed on {song_id}: {e}")

    df = pd.DataFrame.from_records(records)
    os.makedirs(os.path.dirname(OUT_LOOP_CSV), exist_ok=True)
    df.to_csv(OUT_LOOP_CSV, index=False)
    print(f"✅ Saved loop features for {len(df)} songs to {OUT_LOOP_CSV}")

if __name__ == '__main__':
    main()
