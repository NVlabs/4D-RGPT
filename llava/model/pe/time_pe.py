import math
import re
import warnings

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel


class TimePEConfig(PretrainedConfig):
    model_type = "time_pe"

    def __init__(self, pe_type: str = None, **kwargs):
        super().__init__()
        self.pe_type = pe_type


class TimePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_timescale=1e4):
        super().__init__()
        self.d_model = d_model
        self.max_timescale = max_timescale

    def forward(self, t: torch.Tensor):
        """
        t: shape (batch_size, seq_len), representing timestamps (in seconds or any unit)
        returns: (batch_size, seq_len, d_model)
        """
        device = t.device
        half = self.d_model // 2
        # compute the divisor term for frequencies
        div_term = torch.exp(
            torch.arange(0, half, dtype=torch.float32, device=device)
            * -(math.log(self.max_timescale) / half)
        )  # shape (half,)
        # outer product: (batch, seq_len, half)
        t_unsqueezed = t.unsqueeze(-1)  # (B, L, 1)
        phase = t_unsqueezed * div_term  # (B, L, half)
        pe = torch.zeros(t.shape + (self.d_model,), device=device)
        pe[..., :half] = torch.sin(phase)
        pe[..., half:half*2] = torch.cos(phase)
        if self.d_model % 2 == 1:
            # if odd, zero-pad the last dimension
            pe = torch.cat([pe, torch.zeros_like(pe[..., :1])], dim=-1)
        return pe