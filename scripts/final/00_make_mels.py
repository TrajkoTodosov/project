#!/usr/bin/env python
import os, argparse, numpy as np
import librosa
from glob import glob

def make_dir(p): os.makedirs(p, exist_ok=True)

def batch_make_mels(audio_dir, out_dir, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
    make_dir(out_dir)
    paths = sorted(glob(os.path.join(audio_dir, "*.wav")) + glob(os.path.join(audio_dir, "*.mp3")))
    if not paths: raise FileNotFoundError(f"No audio found in {audio_dir}")
    for i, p in enumerate(paths, 1):
        sid = os.path.splitext(os.path.basename(p))[0]
        y, _ = librosa.load(p, sr=sr, mono=True)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        S = librosa.power_to_db(S, ref=np.max).astype("float32")   # [128, T]
        np.save(os.path.join(out_dir, f"{sid}.npy"), S)
        if i % 50 == 0: print(f"  saved {i}/{len(paths)}")
    print(f"Saved {len(paths)} spectrograms to {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Folder with .wav/.mp3")
    ap.add_argument("--out",   required=True, help="Output folder for .npy spectrograms")
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--n_fft", type=int, default=2048)
    ap.add_argument("--hop_length", type=int, default=512)
    args = ap.parse_args()
    batch_make_mels(args.input, args.out, args.sr, args.n_mels, args.n_fft, args.hop_length)
