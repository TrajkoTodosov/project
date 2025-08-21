# models/experts.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── CONSTANTS ─────────────────────────────────────────────────────────────
EXPERT_DIM = 128  # All experts output this dimension


# ── AUDIO EXPERT (Enhanced; BN→GroupNorm for B=1 stability) ──────────────
class AudioExpert(nn.Module):
    """
    Audio expert with residual/multi-scale 1D convs + BiLSTM.
    Input:  mel [B, T, n_mels]
    Output:     [B, T, EXPERT_DIM]
    """
    def __init__(
        self,
        n_mels: int = 128,
        conv_channels=(64, 64, 64),
        lstm_hidden: int = EXPERT_DIM // 2,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        gn_groups: int = 8,  # GroupNorm groups (must divide channel size)
    ):
        super().__init__()

        def conv_block(in_c, out_c, k):
            # GroupNorm is batch-size agnostic; better than BN for B=1
            return nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=k, padding=k // 2),
                nn.GroupNorm(num_groups=gn_groups, num_channels=out_c),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(out_c, out_c, kernel_size=k, padding=k // 2),
                nn.GroupNorm(num_groups=gn_groups, num_channels=out_c),
            )

        c1, c2, c3 = conv_channels

        # Residual block from input mels
        self.block1 = conv_block(n_mels, c1, 3)
        self.res1   = nn.Conv1d(n_mels, c1, kernel_size=1)

        # Multi-scale convs from c1 → c2 (k=3 and k=5), then project to c3
        self.block2a = conv_block(c1, c2, 3)
        self.block2b = conv_block(c1, c2, 5)
        self.project = nn.Conv1d(c2 * 2, c3, kernel_size=1)

        # Temporal modeling
        self.lstm = nn.LSTM(
            input_size=c3,
            hidden_size=lstm_hidden,     # BiLSTM → 2*lstm_hidden = EXPERT_DIM
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, n_mels] → [B, n_mels, T]
        x = x.transpose(1, 2)

        # Residual conv block
        out = self.block1(x) + self.res1(x)
        out = F.relu(out)

        # Multi-scale and projection
        a = self.block2a(out)
        b = self.block2b(out)
        out = torch.cat([a, b], dim=1)          # [B, 2*c2, T]
        out = F.relu(self.project(out))         # [B, c3, T]

        # Back to [B, T, C]
        out = out.transpose(1, 2)

        # BiLSTM to EXPERT_DIM
        out, _ = self.lstm(out)                 # [B, T, 2*lstm_hidden] = [B,T,EXPERT_DIM]
        return out


# ── TABULAR EXPERT (Enhanced) ────────────────────────────────────────────
class TabularExpert(nn.Module):
    """
    Tabular expert with LN → MLP (residual) → BiLSTM.
    Input:  raw_feats [B, T, D_feat]
    Output:           [B, T, out_dim]  (defaults to EXPERT_DIM)
    """
    def __init__(
        self,
        feat_dim: int,
        out_dim: int = EXPERT_DIM,
        mlp_hidden: int = 256,
        lstm_hidden: int = None,  # if None, use out_dim//2
        lstm_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.out_dim = out_dim
        if lstm_hidden is None:
            if out_dim % 2 != 0:
                raise ValueError("out_dim must be even so BiLSTM outputs out_dim.")
            lstm_hidden = out_dim // 2

        self.norm = nn.LayerNorm(feat_dim)
        self.fc1  = nn.Linear(feat_dim, mlp_hidden)
        self.fc2  = nn.Linear(mlp_hidden, out_dim)
        self.res_proj = nn.Linear(feat_dim, out_dim)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=out_dim,
            hidden_size=lstm_hidden,            # BiLSTM → out_dim
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, raw_feats: torch.Tensor) -> torch.Tensor:
        # raw_feats: [B, T, D_feat]
        x = self.norm(raw_feats)
        h = self.relu(self.fc1(x))
        h = self.drop(h)
        h = self.drop(self.fc2(h))
        h = h + self.res_proj(x)                # residual to out_dim
        out, _ = self.lstm(h)                   # [B, T, out_dim]
        return out


# ── OPTIONAL: TEXT & LOOP EXPERTS (kept for future use; not required) ────
class TextExpert(nn.Module):
    """
    Optional text expert (project static text_emb to time, then Transformer).
    Input:  text_emb [B, D_text], T
    Output:          [B, T, EXPERT_DIM]
    """
    def __init__(self, input_dim: int, out_dim: int = EXPERT_DIM, n_heads: int = 4, ff_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(input_dim, out_dim)
        enc = nn.TransformerEncoderLayer(d_model=out_dim, nhead=n_heads, dim_feedforward=ff_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=1)

    def forward(self, text_emb: torch.Tensor, T: int) -> torch.Tensor:
        h = self.proj(text_emb).unsqueeze(1).repeat(1, T, 1)  # [B, T, out_dim]
        h = self.transformer(h)
        return h


class LoopExpert(nn.Module):
    """
    Optional loop expert: project 3D loop features to time, smooth with conv.
    Input:  loop_feats [B, 3], T
    Output:            [B, T, EXPERT_DIM]
    """
    def __init__(self, input_dim: int = 3, out_dim: int = EXPERT_DIM, hidden_dim: int = 32, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.conv = nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1)

    def forward(self, loop_feats: torch.Tensor, T: int) -> torch.Tensor:
        h = self.relu(self.fc1(loop_feats))
        h = self.drop(self.fc2(h))              # [B, out_dim]
        h = h.unsqueeze(1).repeat(1, T, 1)      # [B, T, out_dim]
        h = h.transpose(1, 2)                   # [B, out_dim, T]
        h = self.relu(self.conv(h))
        h = h.transpose(1, 2)                   # [B, T, out_dim]
        return h
