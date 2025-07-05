# scripts/preprocess_audio.py

import os
import librosa
import numpy as np
from tqdm import tqdm

# ─── CONFIG ──────────────────────────────────────────────────────────────
RAW_AUDIO_DIR = os.path.join("data", "raw", "audio")
OUT_SPEC_DIR  = os.path.join("data", "processed", "spectrograms")

# Audio params
SR         = 22050    # sample rate
N_FFT      = 2048     # FFT window size
N_MELS     = 128      # number of Mel bands
HOP_SEC    = 0.5      # 0.5-second hop
HOP_LENGTH = int(SR * HOP_SEC)

# ─── MAKE SURE OUTPUT DIR EXISTS ─────────────────────────────────────────
os.makedirs(OUT_SPEC_DIR, exist_ok=True)

# ─── PROCESS ONE FILE ────────────────────────────────────────────────────
def process_audio_file(path, out_dir):
    # 1) load (resampled to SR)
    y, sr = librosa.load(path, sr=SR)
    # 2) compute Mel-spectrogram (keyword-only call)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )
    # 3) convert to dB
    log_S = librosa.power_to_db(S, ref=np.max)
    # 4) save as .npy
    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, base + ".npy")
    np.save(out_path, log_S)

# ─── MAIN ────────────────────────────────────────────────────────────────
def main():
    audio_files = [
        f for f in os.listdir(RAW_AUDIO_DIR)
        if f.lower().endswith((".wav", ".mp3"))
    ]

    print(f"Found {len(audio_files)} audio files.")
    for fname in tqdm(audio_files, desc="Spectrograms"):
        src = os.path.join(RAW_AUDIO_DIR, fname)
        process_audio_file(src, OUT_SPEC_DIR)

    print("✅ All spectrograms saved to:", OUT_SPEC_DIR)

if __name__ == "__main__":
    main()
