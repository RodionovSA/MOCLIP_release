# MOCLIP: A Foundation Model for Large-Scale Nanophotonic Inverse Design

## Overview
MOCLIP (Metasurface Optics Contrastive Learning Pretrained) is a foundation model for metasurface optics design, trained using contrastive learning on a large-scale experimentally generated dataset of silicon-on-glass metasurfaces. The model was pretrained on 466,537 unique geometry–spectrum pairs, enabling joint representation learning of metasurface geometries and their corresponding polarization-resolved transmission spectra. 

![MOCLIP overview](assets/overview.png)

Details on the dataset generation and MOCLIP architecture can be found in the following arxiv preprint: 
**[MOCLIP: A Foundation Model for Large-Scale Nanophotonic Inverse Design](https://arxiv.org/abs/2511.18980)**.

Alongside the pretrained model, this repository provides a 10k randomly sampled subset of the full dataset for reproducibility, experimentation, and benchmarking.

## Getting Started

### Hardware Requirements

MOCLIP was trained and evaluated on an NVIDIA RTX 4090 GPU with CUDA support. While the model can technically run on a CPU, performance will be prohibitively slow for most practical use cases. We strongly recommend using an NVIDIA GPU with CUDA support for training and inference.

### Software Requirements
- Ubuntu 24.04 LTS (tested; macOS and Windows are expected to work but are not officially validated)
- Python ≥ 3.9  
- PyTorch ≥ 2.4 (with CUDA 12.1 support)  
- NumPy ≥ 1.26  
- TorchVision ≥ 0.19  
- PyYAML ≥ 0.2.5  

### How to Use
Detailed usage examples are provided in `notebooks/MOCLIP_example.ipynb`. 

## Citation
If you use this repository, model, or dataset in your work, please cite:

```bibtex
@article{rodionov2025moclip,
  title   = {MOCLIP: A Foundation Model for Large-Scale Nanophotonic Inverse Design},
  author  = {Rodionov, S. and Burguete-Lopez, A. and Makarenko, M. and Wang, Q. and Getman, F. and Fratalocchi, A.},
  journal = {arXiv preprint arXiv:2511.18980},
  year    = {2025}
}

