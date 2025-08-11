# src/models/moclip.py
import torch
import torch.nn as nn
from src.models.basic import MOCLIPBasic

VARIANTS = {
    "basic":    MOCLIPBasic,
}

def build_moclip(cfg) -> nn.Module:
    """
    cfg.variant        –– one of the keys in VARIANTS
    cfg.geometry_backbone       –– name of your backbone (from BACKBONES)
    cfg.spectra_backbone       –– name of your backbone (from BACKBONES)
    cfg.geometry_backbone_params
    cfg.spectra_backbone_params
    """
    v = cfg.variant
    if v not in VARIANTS:
        raise ValueError(f"No such variant: {v}. Available: {list(VARIANTS)}")
    ModelCls = VARIANTS[v]
    model = ModelCls(
      geometry_backbone=cfg.geometry_backbone,
      spectra_backbone=cfg.spectra_backbone,
      geometry_backbone_params=cfg.geometry_backbone_params,
      spectra_backbone_params=cfg.spectra_backbone_params,
      init_temperature=cfg.init_temperature,
    )
    # Load the checkpoint
    model_state_dict = torch.load(cfg.weights_path, map_location='cpu')

    # Load the state dict into model
    model.load_state_dict(model_state_dict)
    
    model.eval()
    # Freeze the model parameters
    for param in model.parameters():
        param.requires_grad = False

    return model
        

        

        


