#!/usr/bin/env python
# scripts/train_moe.py

import os, sys
# ensure project root is in path before imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# pull in your dataset and collate function
from scripts.dataset import DEAMMultimodalDataset, collate_fn
from models.moe import MoEModel

# ─── UTILS ────────────────────────────────────────────────────────────────
def pearson_corr(a, b):
    mask = ~np.isnan(a) & ~np.isnan(b)
    if mask.sum() < 2:
        return 0.0
    return np.corrcoef(a[mask], b[mask])[0,1]

# ─── TRAIN / EVAL ─────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses, preds, trues = [], [], []
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        audio     = batch['audio'].to(device)      # [B, T, 128]
        raw_feats = batch['raw_feats'].to(device)  # [B, T, D_feat]
        labels    = batch['labels'].to(device)     # [B, T, 1]
        text_emb  = batch['text_emb'].to(device)   # [B, D_text]
        loop_feats= batch['loop_feats'].to(device) # [B, 3]
        has_ly    = batch['has_lyrics'].to(device) # [B]
        has_lp    = batch['has_loop'].to(device)   # [B]

        # quick NaN/Inf check
        for name,t in (('audio',audio),('raw_feats',raw_feats),('labels',labels)):
            if torch.isnan(t).any() or torch.isinf(t).any():
                raise ValueError(f"{name} contains NaN/Inf")

        optimizer.zero_grad()
        y_hat, _ = model(audio, text_emb, raw_feats, loop_feats, has_ly, has_lp)
        loss = criterion(y_hat, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        preds.append(y_hat.detach().cpu().view(-1).numpy())
        trues.append(labels.detach().cpu().view(-1).numpy())
        pbar.set_postfix({'loss':f"{loss.item():.4f}"})

    avg_loss = np.mean(losses)
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    rho = pearson_corr(preds, trues)
    return avg_loss, rho

def eval_epoch(model, loader, criterion, device):
    model.eval()
    losses, preds, trues = [], [], []
    pbar = tqdm(loader, desc="Validation", leave=False)
    with torch.no_grad():
        for batch in pbar:
            audio     = batch['audio'].to(device)
            raw_feats = batch['raw_feats'].to(device)
            labels    = batch['labels'].to(device)
            text_emb  = batch['text_emb'].to(device)
            loop_feats= batch['loop_feats'].to(device)
            has_ly    = batch['has_lyrics'].to(device)
            has_lp    = batch['has_loop'].to(device)

            y_hat, _ = model(audio, text_emb, raw_feats, loop_feats, has_ly, has_lp)
            loss = criterion(y_hat, labels)

            losses.append(loss.item())
            preds.append(y_hat.cpu().view(-1).numpy())
            trues.append(labels.cpu().view(-1).numpy())
            pbar.set_postfix({'val_loss':f"{loss.item():.4f}"})

    avg_loss = np.mean(losses)
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    rho = pearson_corr(preds, trues)
    return avg_loss, rho

# ─── MAIN ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser("Train MoE on DEAM")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs',     type=int, default=10)
    parser.add_argument('--lr',         type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    proc = os.path.join("data", "processed")
    train_csv = os.path.join(proc, "train_ids.csv")
    test_csv  = os.path.join(proc, "test_ids.csv")

    def read_ids(path):
        df = pd.read_csv(path, dtype=str)
        if 'song_id' in df.columns:
            return df['song_id'].str.strip().tolist()
        return df.iloc[:,0].astype(str).str.strip().tolist()

    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        raise FileNotFoundError("Missing train_ids.csv or test_ids.csv in data/processed")

    train_ids = read_ids(train_csv)
    test_ids  = read_ids(test_csv)

    train_ds = DEAMMultimodalDataset(train_ids)
    test_ds  = DEAMMultimodalDataset(test_ids)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn)

    sample = train_ds[0]
    text_dim = sample['text_emb'].shape[0]
    feat_dim = sample['raw_feats'].shape[1]

    model = MoEModel(
        n_mels = 128,
        feat_dim= feat_dim,
        text_dim= text_dim
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs("models", exist_ok=True)

    for epoch in range(1, args.epochs+1):
        print(f"\n Epoch {epoch}/{args.epochs}")
        tr_loss, tr_rho = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_rho = eval_epoch (model, test_loader,  criterion, device)

        print(f" ▶ Train MSE: {tr_loss:.4f}, ρ: {tr_rho:.4f}")
        print(f" ▶  Test MSE: {va_loss:.4f}, ρ: {va_rho:.4f}")

        ckpt = os.path.join("models", f"moe_epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt)
        print(f" Saved {ckpt}")

if __name__=='__main__':
    main()
