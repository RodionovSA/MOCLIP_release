import torch
import torch.nn as nn

import math
    
class SpectraMLP(nn.Module):
    def __init__(self, output_features: int, 
                 input_features: int = 58, 
                 inter_num_neurons1: int = 100, 
                 inter_num_neurons2: int = 200) -> None:
        
        super(SpectraMLP, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_features, inter_num_neurons1),
            nn.BatchNorm1d(inter_num_neurons1),
            nn.ReLU(inplace=True),
            nn.Linear(inter_num_neurons1, inter_num_neurons2),
            nn.BatchNorm1d(inter_num_neurons2),
            nn.ReLU(inplace=True),
            nn.Linear(inter_num_neurons2, output_features)  
        )

        self.layer_norm = nn.LayerNorm(output_features)
        self.projector = nn.Linear(output_features, output_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        x = self.layer_norm(x)
        x = self.projector(x)
        return x