import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones import BACKBONES


class MOCLIPBasic(nn.Module):
    """
    A CLIP-like model that embeds metasurface geometry and spectra into a shared embedding space.
    The vision branch is defined by a configurable backbone from the BACKBONES registry.
    """
    def __init__(self,
                 geometry_backbone: str,
                 spectra_backbone: str,
                 geometry_backbone_params: dict,
                 spectra_backbone_params: dict,
                 init_temperature: float):
        super().__init__()
        # Validate backbone
        if geometry_backbone not in BACKBONES:
            raise ValueError(
                f"Unknown geometry backbone: {geometry_backbone}. Available: {list(BACKBONES.keys())}")
        if spectra_backbone not in BACKBONES:
            raise ValueError(
                f"Unknown spectra backbone: {spectra_backbone}. Available: {list(BACKBONES.keys())}")
        # Instantiate the backbones
        self.geometry_encoder = BACKBONES[geometry_backbone](**geometry_backbone_params)
        self.spectra_encoder = BACKBONES[spectra_backbone](**spectra_backbone_params)

        self.logit_scale = nn.Parameter(
            torch.tensor(1.0 / init_temperature).log()
        )

    def forward(self, geom: torch.Tensor, params: torch.Tensor, spectra: torch.Tensor) -> torch.Tensor:
        """
        geom: [B, 1, H, W]
        params: [B, 2]
        spec: [B, spectrum_dim]
        Returns:
            logits: [B, B]
        """
        geometry_output = self.geometry_encoder(geom, params)
        spectra_output = self.spectra_encoder(spectra)

        # Normalize to unit hypersphere
        geometry_normalized = F.normalize(geometry_output, p=2, dim=-1)
        spectra_normalized = F.normalize(spectra_output, p=2, dim=-1)

        # Compute logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * spectra_normalized @ geometry_normalized.T 
        
        return logits