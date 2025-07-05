#!/usr/bin/env python
# scripts/train_experts.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.dataset import DEAMMultimodalDataset, collate_fn
from models.experts import (
    AudioExpert,
    TextExpert,
    TabularExpert,
    LoopExpert,
    EXPERT_DIM,
)

def pearson_corr(a, b):
    mask = ~np.isnan(a) & ~np.isnan(b)
    if mask.sum() < 2:
        return 0.0
    return np.corrcoef(a[mask], b[mask])[0,1]

def train_single_expert(name, expert_module, loader, device, lr, epochs):
    print(f"\n=== Training {name} expert ===")
    head      = nn.Linear(EXPERT_DIM, 1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        list(expert_module.parameters()) + list(head.parameters()),
        lr=lr
    )

    for ep in range(1, epochs+1):
        expert_module.train()
        head.train()

        losses, preds, trues = [], [], []
        pbar = tqdm(loader, desc=f"{name} Epoch {ep}", leave=False)

        for batch in pbar:
            # unpack & move to device
            audio      = batch['audio'].to(device)       # [B, T, 128]
            raw_feats  = batch['raw_feats'].to(device)   # [B, T, D_feat]
            labels     = batch['labels'].to(device)      # [B, T, 1]
            text_emb   = batch['text_emb'].to(device)    # [B, D_text]
            loop_feats = batch['loop_feats'].to(device)  # [B, 3]

            # expert forward
            if   name == 'tabular':
                out = expert_module(raw_feats)               # [B, T, H]
            elif name == 'audio':
                out = expert_module(audio)                   # [B, T, H]
            elif name == 'text':
                T   = audio.size(1)
                out = expert_module(text_emb, T)             # [B, T, H]
            elif name == 'loop':
                T   = audio.size(1)
                out = expert_module(loop_feats, T)           # [B, T, H]
            else:
                raise ValueError(f"Unknown expert: {name}")

            # map to scalar arousal
            y_hat = head(out)                                # [B, T, 1]
            loss  = criterion(y_hat, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            preds.append(y_hat.detach().cpu().view(-1).numpy())
            trues.append(labels.detach().cpu().view(-1).numpy())
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # epoch metrics
        mse       = np.mean(losses)
        preds_np  = np.concatenate(preds)
        trues_np  = np.concatenate(trues)
        rho       = pearson_corr(preds_np, trues_np)

        print(f"{name} Epoch {ep} → MSE={mse:.4f}, ρ={rho:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs',     type=int, default=5)
    parser.add_argument('--lr',         type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    proc      = os.path.join("data", "processed")
    train_ids = pd.read_csv(os.path.join(proc, "train_ids.csv"),
                             dtype=str).iloc[:,0].str.strip().tolist()
    test_ids  = pd.read_csv(os.path.join(proc, "test_ids.csv"),
                             dtype=str).iloc[:,0].str.strip().tolist()

    # datasets & loaders
    train_ds      = DEAMMultimodalDataset(train_ids)
    test_ds       = DEAMMultimodalDataset(test_ids)
    train_loader  = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True,  collate_fn=collate_fn)
    val_loader    = DataLoader(test_ds,  batch_size=args.batch_size,
                               shuffle=False, collate_fn=collate_fn)

    # sample to infer dims
    sample = train_ds[0]

    # experts (tabular first)
    experts = {
        'tabular': TabularExpert(feat_dim=sample['raw_feats'].shape[1]),
        'audio':   AudioExpert(n_mels=128),
        'text':    TextExpert(input_dim=sample['text_emb'].shape[0]),
        'loop':    LoopExpert(input_dim=sample['loop_feats'].shape[0]),
    }

    # train each expert
    for name, module in experts.items():
        module.to(device)
        train_single_expert(name, module, train_loader,
                            device, lr=args.lr, epochs=args.epochs)

if __name__ == '__main__':
    main()
