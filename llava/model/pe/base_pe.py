import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel


class PEConfig(PretrainedConfig):
    model_type = "spatial_pe"

    def __init__(self, pe_type: str = None, **kwargs):
        super().__init__()
        self.pe_type = pe_type


class Abs3DPositionEmbeddingMLP(nn.Module):
    """Absolute 3D position embedding, NeRF-style.

    Reference: https://github.com/kwea123/nerf_pl/blob/52aeb38/models/nerf.py#L4
    """

    def __init__(self, feature_dim=768, in_channels=3, n_freqs=8, logscale=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_freqs = n_freqs
        self.freq_out_channels = in_channels * (2 * n_freqs + 1)
        if logscale:
            freq_bands = 2 ** torch.linspace(0, n_freqs - 1, n_freqs)
        else:
            freq_bands = torch.linspace(1, 2 ** (n_freqs - 1), n_freqs)

        center = torch.tensor([0.0, 0.0, 2.0]).repeat(in_channels // 3)
        self.register_buffer("freq_bands", freq_bands, persistent=False)
        self.register_buffer("center", center, persistent=False)

        self.position_embedding_head = nn.Sequential(
            nn.Linear(self.freq_out_channels, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        """Small-gain init to keep early training stable."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.01)

    @torch.no_grad()
    def frequency_encoding(self, xyz: torch.Tensor) -> torch.Tensor:
        r"""Embed xyz as (xyz, sin(2^k * xyz), cos(2^k * xyz), ...).

        Coordinate ranges: x ∈ [-2, 2], y ∈ [-2, 2], z ∈ [0, 4].
        Different from the NeRF paper, xyz is also kept in the output.
        See https://github.com/bmild/nerf/issues/12.
        """
        xyz_n = ((xyz - self.center) / 2.0).to(self.freq_bands.dtype)
        xyz_feq = xyz_n.unsqueeze(-1) * self.freq_bands  # (b n m 1)
        sin_xyz, cos_xyz = torch.sin(xyz_feq), torch.cos(xyz_feq)  # (b n m nf)
        encoding = torch.cat([xyz_n.unsqueeze(-1), sin_xyz, cos_xyz], -1).reshape(*xyz.shape[:2], -1)
        return encoding

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """xyz: (B, N, 3 or 6) → (B, N, feature_dim)."""
        freq_encoding = self.frequency_encoding(xyz)
        return self.position_embedding_head(freq_encoding)


class PE(PreTrainedModel):
    config_class = PEConfig

    def __init__(self, pe_cfg: PEConfig, config: PretrainedConfig):
        super().__init__(pe_cfg)
        self.pe_type = pe_cfg.pe_type

        if config.dynamic_s2 and config.image_aspect_ratio == "dynamic_s2":
            feature_dim = config.mm_hidden_size // 3  # block size 3
        else:
            feature_dim = config.mm_hidden_size

        assert self.pe_type == "abs_mlp", f"Unsupported pe_type: {self.pe_type!r}"
        self.layers = Abs3DPositionEmbeddingMLP(feature_dim)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.layers(x)
