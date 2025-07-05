import torch
import torch.nn as nn
import torch.nn.functional as F
from .experts import AudioExpert, TextExpert, TabularExpert, LoopExpert

# ─── Mixture-of-Experts with Dynamic Gating ────────────────────────────────
class GatingNetwork(nn.Module):
    """
    Computes frame-wise softmax weights over experts, using audio summary + modality flags.
    Input:
      - audio_summary [B, H]
      - has_lyrics    [B] (0/1)
      - has_loop      [B] (0/1)
    Output:
      - weights       [B, n_experts]
    """
    def __init__(self, audio_dim, n_experts=4, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(audio_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_experts)
        )

    def forward(self, audio_summary, has_lyrics, has_loop):
        flags = torch.stack([has_lyrics, has_loop], dim=1)       # [B,2]
        x = torch.cat([audio_summary, flags], dim=1)            # [B, H+2]
        logits = self.net(x)                                     # [B,4]
        return F.softmax(logits, dim=1)


class MoEModel(nn.Module):
    """
    Mixture-of-Experts model fusing:
      1) AudioExpert
      2) TextExpert
      3) TabularExpert
      4) LoopExpert
    via dynamic gating + BiLSTM→regressor head.

    Forward inputs:
      - mel         [B, T, n_mels]
      - text_emb    [B, D_text]
      - raw_feats   [B, T, D_feat]
      - loop_feats  [B, 3]
      - has_lyrics  [B]
      - has_loop    [B]

    Outputs:
      - y_hat       [B, T, 1]
      - gate_weights[B, T, 4]
    """
    def __init__(
        self,
        n_mels: int   = 128,
        feat_dim: int = 261,
        text_dim: int = 768,
        expert_dim: int=128,
        lstm_hidden:int=64,
        lstm_layers:int=1,
        dropout:  float =0.3
    ):
        super().__init__()
        # Experts
        self.audio_expert   = AudioExpert(n_mels=n_mels, dropout=dropout)
        self.text_expert    = TextExpert(input_dim=text_dim,    out_dim=expert_dim)
        self.tabular_expert = TabularExpert(feat_dim=feat_dim,  out_dim=expert_dim)
        self.loop_expert    = LoopExpert(input_dim=3,           out_dim=expert_dim)

        # Gating network
        self.gate           = GatingNetwork(audio_dim=expert_dim, n_experts=4)

        # Sequence head: BiLSTM on fused features
        self.sequence_head  = nn.LSTM(
            input_size   = expert_dim,
            hidden_size  = lstm_hidden,
            num_layers   = lstm_layers,
            dropout      = dropout if lstm_layers>1 else 0.0,
            bidirectional= True,
            batch_first  = True
        )
        # Final regressor
        self.regressor      = nn.Linear(2 * lstm_hidden, 1)

    def forward(
        self,
        mel: torch.Tensor,
        text_emb: torch.Tensor,
        raw_feats: torch.Tensor,
        loop_feats: torch.Tensor,
        has_lyrics: torch.Tensor,
        has_loop: torch.Tensor
    ):
        B, T, _ = mel.shape

        # ─── Experts ───────────────────────────────────────────────────────
        eA = self.audio_expert(mel)                      # [B, T, E]
        eT = self.text_expert(text_emb, T)               # [B, T, E]
        eC = self.tabular_expert(raw_feats)              # [B, T, E]
        eD = self.loop_expert(loop_feats, T)             # [B, T, E]

        # ─── Gating ───────────────────────────────────────────────────────
        audio_summary = eA.mean(dim=1)                   # [B, E]
        w = self.gate(audio_summary, has_lyrics, has_loop)  # [B, 4]
        w_exp = w.view(B, 4, 1, 1)                       # [B,4,1,1]

        # ─── Fuse ─────────────────────────────────────────────────────────
        experts = torch.stack([eA, eT, eC, eD], dim=1)   # [B,4,T,E]
        fused   = (w_exp * experts).sum(dim=1)           # [B,T,E]

        # ─── Sequence head → regression ─────────────────────────────────
        seq_out, _    = self.sequence_head(fused)        # [B,T,2H]
        y_hat         = self.regressor(seq_out)          # [B,T,1]

        # expand gate weights in time for logging
        gate_weights = w.unsqueeze(1).expand(-1, T, -1)  # [B,T,4]
        return y_hat, gate_weights
