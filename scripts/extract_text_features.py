# scripts/extract_text_features.py

import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# ─── CONFIG ─────────────────────────────────────────────────────────────
RAW_TEXT_DIR     = os.path.join("data", "processed", "filtered_lyrics_text")
OUT_FEATURE_PATH = os.path.join("data", "processed", "text_features.csv")
MODEL_NAME       = "bert-base-multilingual-cased"
MIN_WORDS        = 5   # only embed if transcript ≥ this many words

# ─── LOAD MODEL ─────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

# ─── PROCESS EACH FILTERED TRANSCRIPT ──────────────────────────────────
records = []
for fn in sorted(os.listdir(RAW_TEXT_DIR)):
    if not fn.endswith(".txt"):
        continue
    song_id = fn.replace(".txt","")
    text = open(os.path.join(RAW_TEXT_DIR,fn), encoding="utf-8").read().strip()
    if len(text.split()) < MIN_WORDS:
        continue

    # tokenize & embed
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = model(**inputs).last_hidden_state[:,0,:].squeeze().numpy()

    rec = {"song_id": song_id}
    for i, v in enumerate(out):
        rec[f"emb_{i}"] = float(v)
    records.append(rec)
    print(f"Encoded {song_id} → {len(out)}-dim embedding")

# ─── SAVE CSV ───────────────────────────────────────────────────────────
df = pd.DataFrame.from_records(records)
os.makedirs(os.path.dirname(OUT_FEATURE_PATH), exist_ok=True)
df.to_csv(OUT_FEATURE_PATH, index=False)
print(f"\n✅ Wrote {len(df)} embeddings to {OUT_FEATURE_PATH}")
