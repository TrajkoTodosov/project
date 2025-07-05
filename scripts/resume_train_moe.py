#!/usr/bin/env python
# scripts/resume_train_moe.py

import os, sys
# put project root on path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.dataset import DEAMMultimodalDataset, collate_fn
from models.moe import MoEModel

def pearson_corr(a, b):
    mask = ~np.isnan(a) & ~np.isnan(b)
    if mask.sum() < 2:
        return 0.0
    return np.corrcoef(a[mask], b[mask])[0,1]

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses, preds, trues = [], [], []
    for batch in tqdm(loader, desc="Training"):
        audio     = batch['audio'].to(device)
        raw_feats = batch['raw_feats'].to(device)
        labels    = batch['labels'].to(device)
        text_emb  = batch['text_emb'].to(device)
        # no duration in this setup
        loop_feats= batch['loop_feats'].to(device)
        has_ly    = batch['has_lyrics'].to(device)
        has_lp    = batch['has_loop'].to(device)

        optimizer.zero_grad()
        y_hat, _ = model(audio, text_emb, raw_feats,
                         None,        # skip duration
                         loop_feats, has_ly, has_lp)
        loss = criterion(y_hat, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        preds.append(y_hat.detach().cpu().view(-1).numpy())
        trues.append(labels.detach().cpu().view(-1).numpy())

    avg_loss = np.mean(losses)
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    rho = pearson_corr(preds, trues)
    return avg_loss, rho

def eval_epoch(model, loader, criterion, device):
    model.eval()
    losses, preds, trues = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            audio     = batch['audio'].to(device)
            raw_feats = batch['raw_feats'].to(device)
            labels    = batch['labels'].to(device)
            text_emb  = batch['text_emb'].to(device)
            loop_feats= batch['loop_feats'].to(device)
            has_ly    = batch['has_lyrics'].to(device)
            has_lp    = batch['has_loop'].to(device)

            y_hat, _ = model(audio, text_emb, raw_feats,
                             None,
                             loop_feats, has_ly, has_lp)
            loss = criterion(y_hat, labels)

            losses.append(loss.item())
            preds.append(y_hat.cpu().view(-1).numpy())
            trues.append(labels.cpu().view(-1).numpy())

    avg_loss = np.mean(losses)
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    rho = pearson_corr(preds, trues)
    return avg_loss, rho

def main():
    parser = argparse.ArgumentParser(
        description="Resume training MoE model from a checkpoint"
    )
    parser.add_argument(
        '--checkpoint', type=str,
        default=os.path.join("models", "moe_epoch10.pt"),
        help="Path to the .pt checkpoint to load"
    )
    parser.add_argument(
        '--start_epoch', type=int, default=11,
        help="Number to label the first resumed epoch (will save moe_epoch<start_epoch>.pt, ...)"
    )
    parser.add_argument(
        '--epochs', type=int, default=10,
        help="How many epochs to run after start_epoch"
    )
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Resuming on device: {device}")

    # load splits
    proc = os.path.join("data", "processed")
    def read_ids(p):
        df = pd.read_csv(p, dtype=str)
        return (df['song_id'] if 'song_id' in df.columns else df.iloc[:,0]
               ).astype(str).str.strip().tolist()

    train_ids = read_ids(os.path.join(proc, "train_ids.csv"))
    test_ids  = read_ids(os.path.join(proc, "test_ids.csv"))

    train_ds = DEAMMultimodalDataset(train_ids)
    test_ds  = DEAMMultimodalDataset(test_ids)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn)

    # build model and load state
    sample = train_ds[0]
    text_dim = sample['text_emb'].shape[0]
    feat_dim = sample['raw_feats'].shape[1]  # includes no duration here
    model = MoEModel(n_mels=128, feat_dim=feat_dim-1, text_dim=text_dim).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        print(f"\n📅 Epoch {epoch}/{args.start_epoch + args.epochs - 1}")
        tr_loss, tr_rho = train_epoch(model, train_loader, criterion,
                                      optimizer, device)
        val_loss, val_rho = eval_epoch(model, val_loader, criterion, device)
        print(f" ▶ Train MSE: {tr_loss:.4f}, ρ: {tr_rho:.4f}")
        print(f" ▶ Test  MSE: {val_loss:.4f}, ρ: {val_rho:.4f}")

        ckpt = os.path.join("models", f"moe_epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt)
        print(f"✅ Saved {ckpt}")

if __name__ == "__main__":
    main()
