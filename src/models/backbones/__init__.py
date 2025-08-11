# src/models/backbones/__init__.py
from .resnet_custom import ResNetCustom
from .spectra_mlp import SpectraMLP


BACKBONES = {
    "resnet_custom": ResNetCustom,
    "spectra_mlp": SpectraMLP,
}