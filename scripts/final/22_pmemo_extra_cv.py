#!/usr/bin/env python
"""
Adds 5 more PMEmo cross‑validation runs (numbered run6–run10) using a different fold seed,
then recomputes the epoch‑wise mean±std across ALL found PMEmo CV runs (1–10) under results/.
"""

import os, sys, importlib.util
from datetime import datetime
import numpy as np, pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# ---------- dynamic import helpers ----------
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SCRIPTS  = os.path.join(ROOT, "scripts")
MODELS   = os.path.join(ROOT, "models")
sys.path.append(ROOT)

deam     = load_module("dataset",        os.path.join(SCRIPTS, "dataset.py"))
pmemo    = load_module("pmemo_dataset",  os.path.join(SCRIPTS, "pmemo_dataset.py"))
experts  = load_module("experts",        os.path.join(MODELS,  "experts.py"))

PMEmoDataset  = pmemo.PMEmoDataset
collate_fn    = deam.collate_fn
AudioExpert   = experts.AudioExpert
TabularExpert = experts.TabularExpert
EXPERT_DIM    = experts.EXPERT_DIM

# ---------- utils ----------
def set_seed(s):
    import random
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def pearson(a,b):
    m = ~np.isnan(a) & ~np.isnan(b)
    return pearsonr(a[m], b[m])[0] if m.sum()>=2 else 0.0

def feat_dim_of(ds):
    for i in range(len(ds)):
        rf = ds[i]["raw_feats"]
        if rf.ndim==2: return int(rf.shape[-1])
    raise RuntimeError("feat_dim not found")

# ---------- model ----------
class AblationMoE(nn.Module):
    def __init__(self, feat_dim, lstm_hidden=64, lstm_layers=1, dropout=0.3):
        super().__init__()
        self.experts = nn.ModuleDict({
            "audio":   AudioExpert(n_mels=128, dropout=dropout),
            "tabular": TabularExpert(feat_dim=feat_dim, out_dim=EXPERT_DIM)
        })
        self.gate = nn.Sequential(nn.Linear(EXPERT_DIM+2,32), nn.ReLU(), nn.Linear(32,2), nn.Softmax(-1))
        self.lstm = nn.LSTM(EXPERT_DIM, lstm_hidden, lstm_layers,
                            dropout=dropout if lstm_layers>1 else 0.0,
                            bidirectional=True, batch_first=True)
        self.reg  = nn.Linear(2*lstm_hidden,1)

    def forward(self, mel, text_emb, raw_feats, loop_feats, has_ly, has_lp):
        B,T,_ = mel.shape
        eA = self.experts["audio"](mel)
        eC = self.experts["tabular"](raw_feats)
        w  = self.gate(torch.cat([eA.mean(1), torch.stack([has_ly,has_lp],1)],1))
        fused = (w.view(B,2,1,1)*torch.stack([eA,eC],1)).sum(1)
        y,_ = self.lstm(fused)
        return self.reg(y)

# ---------- train/eval ----------
def train_eval(train_set, test_set, feat_dim, epochs=40, batch_size=1, lr=1e-4, seed=None, device=None):
    if seed is not None: set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = AblationMoE(feat_dim).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    mse    = nn.MSELoss()

    trL = DataLoader(train_set, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    teL = DataLoader(test_set,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    rows=[]
    for ep in range(1,epochs+1):
        print(f"  Epoch {ep}/{epochs}")
        model.train()
        for b in tqdm(trL, leave=False, desc="    Training"):
            audio,raw,lab,txt,loop,hl,lp = b['audio'].to(device), b['raw_feats'].to(device), b['labels'].to(device), b['text_emb'].to(device), b['loop_feats'].to(device), b['has_lyrics'].to(device), b['has_loop'].to(device)
            opt.zero_grad(); y = model(audio,txt,raw,loop,hl,lp); loss = mse(y,lab); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); P,T=[],[]
        with torch.no_grad():
            for b in tqdm(teL, leave=False, desc="    Testing"):
                audio,raw,lab,txt,loop,hl,lp = b['audio'].to(device), b['raw_feats'].to(device), b['labels'].to(device), b['text_emb'].to(device), b['loop_feats'].to(device), b['has_lyrics'].to(device), b['has_loop'].to(device)
                y = model(audio,txt,raw,loop,hl,lp)
                P.append(y.cpu().view(-1).numpy()); T.append(lab.cpu().view(-1).numpy())
        rho = pearson(np.concatenate(P), np.concatenate(T))
        print(f"    Test ρ = {rho:.4f}")
        rows.append({"epoch":ep,"test_rho":rho})
    return pd.DataFrame(rows)

# ---------- main ----------
def main():
    # paths
    pmemo_feat = os.path.join(ROOT, "PMEmo", "dynamic_features_aligned_SHARED.csv")
    pmemo_ann  = os.path.join(ROOT, "PMEmo", "dynamic_annotations.csv")

    # dataset
    pmemo_ds = PMEmoDataset(features_csv=pmemo_feat, annotations_csv=pmemo_ann, target="arousal")
    idx = list(range(len(pmemo_ds)))
    fd  = feat_dim_of(pmemo_ds)

    # hparams
    EPOCHS=40; BATCH=1; LR=1e-4

    # output dirs
    out_root = os.path.join(ROOT, "results"); os.makedirs(out_root, exist_ok=True)
    stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir  = os.path.join(out_root, f"PMEmo_cv_extra_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    # New 5-fold CV with a different random_state so folds differ from the original (42)
    FOLDS = 5
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=123)  # different seed
    start_run_idx = 6  # we will write run6..run10

    print("\n=== PMEmo extra CV (folds 6–10) ===")
    for i,(tr,te) in enumerate(kf.split(idx), start=start_run_idx):
        print(f"\n-- PMEmo CV fold {i}/10 --")
        df = train_eval(Subset(pmemo_ds, tr), Subset(pmemo_ds, te), fd, epochs=EPOCHS, batch_size=BATCH, lr=LR, seed=1000+i)
        p  = os.path.join(out_dir, f"PMEmo_cv_run{i}.xlsx")
        df.to_excel(p, index=False)
        print("[saved]", p)

    # Recompute combined MEAN across ALL found PMEmo_cv_run*.xlsx (1–10) under results/
    print("\nRecomputing MEAN across all found PMEmo_cv_run[1..10].xlsx under results/ ...")
    all_runs = []
    found = 0
    for r in range(1,11):
        target = f"PMEmo_cv_run{r}.xlsx"
        hit = None
        for root, _, files in os.walk(out_root):
            if target in files:
                hit = os.path.join(root, target); break
        if hit:
            all_runs.append(pd.read_excel(hit)); found += 1
        else:
            print(f"  [WARN] {target} not found; skipping in mean")

    if found >= 2:
        mean_df = pd.DataFrame({
            "epoch": all_runs[0]["epoch"],
            "mean_test_rho": pd.concat([d["test_rho"] for d in all_runs], axis=1).mean(1),
            "std_test_rho":  pd.concat([d["test_rho"] for d in all_runs], axis=1).std(1),
        })
        mean_path = os.path.join(out_root, "PMEmo_cv_MEAN.xlsx")
        mean_df.to_excel(mean_path, index=False)
        print("[saved MEAN]", mean_path)
    else:
        print("Not enough runs found to compute a mean (need ≥2).")

    print("\nDone. New runs in:", out_dir)

if __name__ == "__main__":
    main()
