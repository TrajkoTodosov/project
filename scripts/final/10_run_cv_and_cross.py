#!/usr/bin/env python
import os, importlib.util
from datetime import datetime
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from tqdm import tqdm
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset

# — dynamic imports —
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SCRIPTS= os.path.join(ROOT, "scripts"); MODELS = os.path.join(ROOT, "models")
deam   = load_module("dataset", os.path.join(SCRIPTS, "dataset.py"))
pmemo  = load_module("pmemo_dataset", os.path.join(SCRIPTS, "pmemo_dataset.py"))
exp    = load_module("experts", os.path.join(ROOT, "models", "experts.py"))

DEAMMultimodalDataset = deam.DEAMMultimodalDataset
PMEmoDataset          = pmemo.PMEmoDataset
collate_fn            = deam.collate_fn

AudioExpert  = exp.AudioExpert
TabularExpert= exp.TabularExpert
EXPERT_DIM   = exp.EXPERT_DIM

def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def pearson(a,b):
    m = ~np.isnan(a) & ~np.isnan(b)
    return pearsonr(a[m], b[m])[0] if m.sum()>=2 else 0.0

def feat_dim_of(ds):
    for i in range(len(ds)):
        rf = ds[i]["raw_feats"]
        if rf.ndim==2: return int(rf.shape[-1])
    raise RuntimeError("feat_dim not found")

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

def train_eval(train_set, test_set, feat_dim, epochs=40, batch_size=1, lr=1e-4, seed=None, device=None):
    if seed is not None: set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = AblationMoE(feat_dim).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    mse    = nn.MSELoss()

    trL = DataLoader(train_set, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    teL = DataLoader(test_set,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    rec=[]
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
        rec.append({"epoch":ep,"test_rho":rho})
    return pd.DataFrame(rec)

def crossval(dataset, name, out_dir, folds, epochs, batch_size, lr, base_seed=1337):
    os.makedirs(out_dir, exist_ok=True)
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    idx = list(range(len(dataset))); fd = feat_dim_of(dataset)
    fold_dfs=[]; finals=[]
    for i,(tr,te) in enumerate(kf.split(idx),1):
        print(f"\n-- {name} CV fold {i}/{folds} --")
        df = train_eval(Subset(dataset,tr), Subset(dataset,te), fd, epochs, batch_size, lr, seed=base_seed+i)
        p  = os.path.join(out_dir, f"{name}_cv_run{i}.xlsx"); df.to_excel(p, index=False); print("[saved]", p)
        fold_dfs.append(df); finals.append(df["test_rho"].iloc[-1])
    mean = pd.DataFrame({
        "epoch": fold_dfs[0]["epoch"],
        "mean_test_rho": pd.concat([d["test_rho"] for d in fold_dfs], axis=1).mean(1),
        "std_test_rho":  pd.concat([d["test_rho"] for d in fold_dfs], axis=1).std(1),
    })
    mp = os.path.join(out_dir, f"{name}_cv_MEAN.xlsx"); mean.to_excel(mp, index=False); print("[saved]", mp)
    print(f"{name} CV mean(final epoch): {np.mean(finals):.4f}")

def repeated_full(train_set, test_set, name_train, name_test, out_dir, runs, epochs, batch_size, lr, base_seed=2025):
    os.makedirs(out_dir, exist_ok=True); fd = feat_dim_of(train_set)
    run_dfs=[]
    for r in range(1,runs+1):
        print(f"\n-- {name_train}(FULL) → {name_test} run {r}/{runs} --")
        df = train_eval(train_set, test_set, fd, epochs, batch_size, lr, seed=base_seed+r)
        p  = os.path.join(out_dir, f"{name_train}_FULL_to_{name_test}_run{r}.xlsx"); df.to_excel(p, index=False); print("[saved]", p)
        run_dfs.append(df)
    mean = pd.DataFrame({
        "epoch": run_dfs[0]["epoch"],
        "mean_test_rho": pd.concat([d["test_rho"] for d in run_dfs], axis=1).mean(1),
        "std_test_rho":  pd.concat([d["test_rho"] for d in run_dfs], axis=1).std(1),
    })
    mp = os.path.join(out_dir, f"{name_train}_FULL_to_{name_test}_MEAN.xlsx"); mean.to_excel(mp, index=False); print("[saved]", mp)

def main():
    # paths
    deam_ids_csv = os.path.join(ROOT, "data", "processed", "train_ids.csv")
    pmemo_feat   = os.path.join(ROOT, "PMEmo", "dynamic_features_aligned_SHARED.csv")
    pmemo_ann    = os.path.join(ROOT, "PMEmo", "dynamic_annotations.csv")

    # datasets
    deam_ids = pd.read_csv(deam_ids_csv, dtype=str).iloc[:,0].str.strip().tolist()
    deam_ds  = DEAMMultimodalDataset(deam_ids)
    pmemo_ds = PMEmoDataset(features_csv=pmemo_feat, annotations_csv=pmemo_ann, target="arousal")
    both     = ConcatDataset([deam_ds, pmemo_ds])

    # hparams
    EPOCHS=40; BATCH=1; LR=1e-4
    out_root = os.path.join(ROOT, "results"); os.makedirs(out_root, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S"); out = os.path.join(out_root, f"runs_{stamp}"); os.makedirs(out, exist_ok=True)

    # 1) DEAM CV (10)
    crossval(deam_ds, "DEAM", os.path.join(out,"DEAM_cv"), folds=10, epochs=EPOCHS, batch_size=BATCH, lr=LR)
    # 2) PMEmo CV (5)
    crossval(pmemo_ds, "PMEmo", os.path.join(out,"PMEmo_cv"), folds=5, epochs=EPOCHS, batch_size=BATCH, lr=LR)
    # 3) BOTH → DEAM (10 repeated runs)
    repeated_full(both, deam_ds, "BOTH", "DEAM", os.path.join(out,"BOTH_to_DEAM"), runs=10, epochs=EPOCHS, batch_size=BATCH, lr=LR)
    # 4) BOTH → PMEmo (10 repeated runs)
    repeated_full(both, pmemo_ds, "BOTH", "PMEmo", os.path.join(out,"BOTH_to_PMEmo"), runs=10, epochs=EPOCHS, batch_size=BATCH, lr=LR)
    print("\nALL DONE. Results in:", out)

if __name__ == "__main__":
    main()
