# scripts/pmemo_dataset.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PMEmoDataset(Dataset):
    """
    Dataset for inference on PMEmo using audio + tabular features.
    Only arousal is used as label by default.
    """
    def __init__(self,
                 spectrogram_dir="PMEmo/spectrograms",
                 features_csv="PMEmo/dynamic_features_aligned.csv",  # use aligned file
                 annotations_csv="PMEmo/dynamic_annotations.csv",
                 target="arousal"):
        assert target in ("arousal", "valence")
        self.spectrogram_dir = spectrogram_dir
        self.target = target

        # Load features and annotations
        feats = pd.read_csv(features_csv)
        labels = pd.read_csv(annotations_csv)

        # Drop early time steps to match label availability
        feats = feats[feats["frameTime"] >= 15.5]

        # Select label column
        if target == "arousal":
            labels = labels[["musicId", "frameTime", "Arousal(mean)"]].rename(columns={"Arousal(mean)": "label"})
        else:
            labels = labels[["musicId", "frameTime", "Valence(mean)"]].rename(columns={"Valence(mean)": "label"})

        # Merge on musicId and frameTime
        merged = pd.merge(feats, labels, on=["musicId", "frameTime"])
        self.data = merged
        self.song_ids = merged["musicId"].unique().tolist()

    def __len__(self):
        return len(self.song_ids)

    def __getitem__(self, idx):
        song_id = self.song_ids[idx]

        # Load spectrogram [128, T] → [T, 128]
        mel_path = os.path.join(self.spectrogram_dir, f"{song_id}.npy")
        mel = np.load(mel_path).T  # shape [T, 128]

        # Get tabular rows for this song
        rows = self.data[self.data["musicId"] == song_id]

        # Include frameTime as feature
        feat_cols = [c for c in rows.columns if c not in ("musicId", "label")]
        raw_feats = rows[feat_cols].values.astype(np.float32)  # [T, D]

        # Labels
        labels = rows["label"].values.astype(np.float32)  # [T]

        # Truncate
        T = min(len(mel), len(raw_feats), len(labels))
        mel = mel[:T]
        raw_feats = raw_feats[:T]
        labels = labels[:T]

        return {
            "audio": torch.tensor(mel, dtype=torch.float32),              # [T, 128]
            "raw_feats": torch.tensor(raw_feats, dtype=torch.float32),    # [T, feat_dim]
            "labels": torch.tensor(labels, dtype=torch.float32).unsqueeze(-1),  # [T, 1]
            "text_emb": torch.zeros((1,), dtype=torch.float32),           # dummy
            "loop_feats": torch.zeros((3,), dtype=torch.float32),         # dummy
            "has_lyrics": torch.tensor(0.0, dtype=torch.float32),
            "has_loop": torch.tensor(0.0, dtype=torch.float32)
        }
