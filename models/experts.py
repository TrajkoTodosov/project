import torch
import torch.nn as nn

# ─── CONSTANTS ────────────────────────────────────────────────────────────
# Output dimension for each expert (fusion dim = H)
EXPERT_DIM = 128

# ─── AUDIO EXPERT ─────────────────────────────────────────────────────────
class AudioExpert(nn.Module):
    """
    CNN → BiLSTM on log-Mel spectrograms.
    Input:  x [B, T, n_mels]
    Output:    [B, T, EXPERT_DIM]
    """
    def __init__(self, n_mels=128, conv_channels=(64, 128),
                 lstm_hidden=EXPERT_DIM//2, lstm_layers=2, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, conv_channels[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1)
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(dropout)
        # BiLSTM: 2 * lstm_hidden = EXPERT_DIM
        self.lstm  = nn.LSTM(input_size=conv_channels[1],
                             hidden_size=lstm_hidden,
                             num_layers=lstm_layers,
                             dropout=dropout,
                             bidirectional=True,
                             batch_first=True)

    def forward(self, x):
        # x: [B, T, n_mels] → [B, n_mels, T]
        x = x.transpose(1, 2)
        x = self.drop(self.relu(self.conv1(x)))
        x = self.drop(self.relu(self.conv2(x)))
        # → [B, T, conv_channels[1]]
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        return out  # [B, T, EXPERT_DIM]


# ─── TEXT EXPERT ───────────────────────────────────────────────────────────
class TextExpert(nn.Module):
    """
    Projects static text embedding into time-aligned expert features.
    Input:  text_emb [B, D_text], T
    Output:            [B, T, EXPERT_DIM]
    """
    def __init__(self, input_dim, out_dim=EXPERT_DIM):
        super().__init__()
        self.proj = nn.Linear(input_dim, out_dim)

    def forward(self, text_emb, T):
        # text_emb: [B, D_text] → [B, EXPERT_DIM]
        h = self.proj(text_emb)
        # repeat across time
        return h.unsqueeze(1).repeat(1, T, 1)  # [B, T, EXPERT_DIM]


# ─── TABULAR EXPERT ────────────────────────────────────────────────────────
class TabularExpert(nn.Module):
    """
    Projects per-frame openSMILE features into EXPERT_DIM.
    Input:  raw_feats [B, T, D_feat]
    Output:            [B, T, EXPERT_DIM]
    """
    def __init__(self, feat_dim, out_dim=EXPERT_DIM):
        super().__init__()
        # normalize across feature dim to stabilize scale
        self.norm = nn.LayerNorm(feat_dim)
        # linear projection
        self.fc   = nn.Linear(feat_dim, out_dim)

    def forward(self, raw_feats):
        # raw_feats: [B, T, D_feat]
        x = self.norm(raw_feats)       # [B, T, D_feat]
        return self.fc(x)              # [B, T, EXPERT_DIM]


# ─── LOOP EXPERT ───────────────────────────────────────────────────────────
class LoopExpert(nn.Module):
    """
    Projects static loop features into time-aligned expert features.
    Input:  loop_feats [B, 3], T
    Output:            [B, T, EXPERT_DIM]
    """
    def __init__(self, input_dim=3, out_dim=EXPERT_DIM):
        super().__init__()
        self.fc = nn.Linear(input_dim, out_dim)

    def forward(self, loop_feats, T):
        # loop_feats: [B, 3] → [B, EXPERT_DIM]
        h = self.fc(loop_feats)
        return h.unsqueeze(1).repeat(1, T, 1)  # [B, T, EXPERT_DIM]
