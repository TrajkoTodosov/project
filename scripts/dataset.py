#!/usr/bin/env python
# scripts/dataset.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ─── CONFIG ─────────────────────────────────────────────────────────────
SPECTRO_DIR      = os.path.join("data", "processed", "spectrograms")
RAW_FEAT_DIR     = os.path.join("data", "raw",       "features")
LABEL_DIR        = os.path.join("data", "processed", "filtered_avg")
TEXT_FEAT_CSV    = os.path.join("data", "processed", "text_features.csv")
LYRICS_IDS_CSV   = os.path.join("data", "processed", "filtered_lyrics_ids.csv")
LOOP_FEAT_CSV    = os.path.join("data", "processed", "loop_features.csv")
LOOP_STRENGTH_THR = 0.1

# ─── SANITY CHECK ────────────────────────────────────────────────────────
for d in (SPECTRO_DIR, RAW_FEAT_DIR, LABEL_DIR):
    if not os.path.isdir(d):
        raise FileNotFoundError(f"Missing directory: {d}")
for f in (LYRICS_IDS_CSV, LOOP_FEAT_CSV):
    if not os.path.isfile(f):
        raise FileNotFoundError(f"Missing file: {f}")

# ─── DATASET DEFINITION ──────────────────────────────────────────────────
class DEAMMultimodalDataset(Dataset):
    """
    Returns a dict with:
      - audio      [T,128]
      - raw_feats  [T,D]
      - labels     [T,1]
      - text_emb   [D_text]
      - loop_feats [3]
      - has_lyrics [ ]    scalar flag
      - has_loop   [ ]    scalar flag
    """
    def __init__(self, song_ids):
        self.ids = [str(s).strip() for s in song_ids]

        # valid lyrics
        lyr = pd.read_csv(LYRICS_IDS_CSV, dtype=str)
        lyr['song_id'] = lyr['song_id'].str.strip()
        self.valid_lyrics = set(lyr['song_id'])

        # optional text embeddings
        if os.path.isfile(TEXT_FEAT_CSV):
            txt = pd.read_csv(TEXT_FEAT_CSV, dtype=str)
            txt['song_id'] = txt['song_id'].str.strip()
            self.text_df = txt.set_index('song_id')
        else:
            self.text_df = None

        # loop features
        lp = pd.read_csv(LOOP_FEAT_CSV, dtype=str)
        lp['song_id'] = lp['song_id'].str.strip()
        self.loop_df = lp.set_index('song_id')

        # filter to IDs with all necessary files
        good = []
        for sid in self.ids:
            if (
                os.path.isfile(f"{SPECTRO_DIR}/{sid}.npy") and
                os.path.isfile(f"{RAW_FEAT_DIR}/{sid}.csv") and
                os.path.isfile(f"{LABEL_DIR}/{sid}.csv") and
                sid in self.loop_df.index
            ):
                good.append(sid)
        if not good:
            raise RuntimeError("No valid song IDs after filtering.")
        self.song_ids = good

    def __len__(self):
        return len(self.song_ids)

    def __getitem__(self, idx):
        sid = self.song_ids[idx]

        # 1) Spectrogram → [T,128]
        spec = np.load(f"{SPECTRO_DIR}/{sid}.npy").T
        spec = torch.from_numpy(spec).float()
        T = spec.size(0)

        # 2) openSMILE features → [T,D]
        df_f = pd.read_csv(f"{RAW_FEAT_DIR}/{sid}.csv", sep=';')
        if 'time' in df_f.columns:
            df_f = df_f.drop(columns=['time'])
        feats = torch.from_numpy(df_f.values.astype(np.float32))
        if feats.size(0) >= T:
            feats = feats[:T]
        else:
            pad = torch.zeros((T - feats.size(0), feats.size(1)))
            feats = torch.cat([feats, pad], dim=0)

        # 3) Labels (arousal) → [T,1]
        lbl = pd.read_csv(f"{LABEL_DIR}/{sid}.csv", header=None)
        arr = lbl.iloc[1].values.astype(np.float32)[:T]
        labels = torch.from_numpy(arr.reshape(-1,1))
        if labels.size(0) < T:
            pad = torch.zeros((T - labels.size(0), 1))
            labels = torch.cat([labels, pad], dim=0)

        # 4) Text embedding → [D_text] or zeros
        if self.text_df is not None and sid in self.valid_lyrics:
            t_emb = torch.from_numpy(self.text_df.loc[sid].values.astype(np.float32))
        else:
            D = self.text_df.shape[1] if self.text_df is not None else 0
            t_emb = torch.zeros(D, dtype=torch.float32)

        # 5) Loop features → [3]
        row = self.loop_df.loc[sid]
        loop_feats = torch.tensor([
            float(row['loop_period']),
            float(row['loop_strength']),
            float(row['repeat_count'])
        ], dtype=torch.float32)

        # flags
        has_lyrics = torch.tensor(float(sid in self.valid_lyrics))
        has_loop   = torch.tensor(float(loop_feats[1] > LOOP_STRENGTH_THR))

        return {
            'audio':      spec,
            'raw_feats':  feats,
            'labels':     labels,
            'text_emb':   t_emb,
            'loop_feats': loop_feats,
            'has_lyrics': has_lyrics,
            'has_loop':   has_loop
        }


# ─── COLLATE FUNCTION ─────────────────────────────────────────────────────
def collate_fn(batch):
    import torch
    B      = len(batch)
    T_max  = max(x['audio'].size(0) for x in batch)
    n_mels = batch[0]['audio'].size(1)
    Df     = batch[0]['raw_feats'].size(1)

    audio     = torch.zeros(B, T_max, n_mels, dtype=torch.float32)
    raw_feats = torch.zeros(B, T_max, Df,     dtype=torch.float32)
    labels    = torch.zeros(B, T_max, 1,      dtype=torch.float32)

    text_embs, loops, has_ly, has_lp = [], [], [], []

    for i, x in enumerate(batch):
        T = x['audio'].size(0)
        audio[i, :T]     = x['audio']
        raw_feats[i, :T] = x['raw_feats']
        labels[i, :T]    = x['labels']

        text_embs.append(x['text_emb'])
        loops.append(x['loop_feats'])
        has_ly.append(x['has_lyrics'])
        has_lp.append(x['has_loop'])

    text_emb   = torch.stack(text_embs, dim=0)  # [B, D_text]
    loop_feats = torch.stack(loops,     dim=0)  # [B,3]
    has_lyrics = torch.stack(has_ly,    dim=0)  # [B]
    has_loop   = torch.stack(has_lp,    dim=0)  # [B]

    return {
        'audio':      audio,
        'raw_feats':  raw_feats,
        'labels':     labels,
        'text_emb':   text_emb,
        'loop_feats': loop_feats,
        'has_lyrics': has_lyrics,
        'has_loop':   has_loop
    }


# ─── SANITY CHECK ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    ids = pd.read_csv(LYRICS_IDS_CSV, dtype=str)['song_id'].tolist()[:5]
    ds  = DEAMMultimodalDataset(ids)
    sample = ds[0]
    for k,v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {tuple(v.shape)}")
        else:
            print(f"{k}: {v}")
