# src/models/backbones/resnet_custom.py
# A custom ResNet backbone 

import torch
import torch.nn as nn
import torchvision.models as models

class ResNetCustom(nn.Module):
    def __init__(self, output_features: int, backbone: str = 'resnet18') -> None:
        """
        A high-level image processing block that combines a modified ResNet with custom parameter embedding.

        Parameters:
        ----------
        output_features : int
            The number of output neurons in the final fully connected layer.
        backbone : str
            The ResNet backbone to use. Options: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
        """
        super(ResNetCustom, self).__init__()

        # Load pre-trained ResNet model based on user choice
        resnet_models = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152
        }
        
        if backbone not in resnet_models:
            raise ValueError(f"Backbone {backbone} not supported. Choose from: {list(resnet_models.keys())}")
            
        self.resnet = resnet_models[backbone](weights='DEFAULT')

        # Define custom ParametersFCModule for each ResNet layer
        channels = {'resnet18': [64, 128, 256, 512],
                   'resnet34': [64, 128, 256, 512],
                   'resnet50': [256, 512, 1024, 2048],
                   'resnet101': [256, 512, 1024, 2048],
                   'resnet152': [256, 512, 1024, 2048]}[backbone]
                   
        self.parameters_module_dict = nn.ModuleDict({
            'layer1': ParametersFCModule(channels[0]),
            'layer2': ParametersFCModule(channels[1]), 
            'layer3': ParametersFCModule(channels[2]),
            'layer4': ParametersFCModule(channels[3]),
        })
        
        # Move each module inside the parameters_module_dict to the same device
        device = next(self.parameters()).device
        for key, module in self.parameters_module_dict.items():
            self.parameters_module_dict[key] = module.to(device)
        
        # Create the modified ResNet with custom parameter layers
        self.modified_resnet = ModifiedResNet(self.resnet, self.parameters_module_dict,
                                               output_features=output_features)
        self.layer_norm = nn.LayerNorm(output_features)
        self.projector = nn.Linear(output_features, output_features, bias=False)

    def forward(self, geom: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        x = self.modified_resnet(geom, params)
        x = self.layer_norm(x)
        x = self.projector(x)
        return x


class ModifiedResNet(nn.Module):
    def __init__(self, original_resnet: nn.Module, parameters_module_dict: dict, output_features: int) -> None:
        """
        A modified ResNet model that combines standard ResNet layers with custom parameter embedding layers.

        Parameters:
        ----------
        original_resnet : nn.Module
            The pre-trained ResNet model.
        parameters_module_dict : dict
            A dictionary containing ParametersFCModule instances for each ResNet layer.
        output_features : int
            The number of output neurons in the final fully connected layer.
        """
        super(ModifiedResNet, self).__init__()
        
        # Keep the original ResNet layers
        self.original_resnet = original_resnet
        
        # Dictionary containing custom layers for parameter embedding
        self.parameters_module_dict = parameters_module_dict

        # Final output features for the last FC layer
        self.output_features = output_features
        
        # Build the modified ResNet using the original model's layers
        self._build_model()
    
    def _build_model(self) -> None:
        """Builds the modified ResNet model using the original ResNet components."""
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), 
                                stride=(2, 2), padding=(3, 3), bias=False) # one channel only
        
        self.bn1 = self.original_resnet.bn1
        self.relu = self.original_resnet.relu
        self.maxpool = self.original_resnet.maxpool
        self.layer1 = self.original_resnet.layer1
        self.layer2 = self.original_resnet.layer2
        self.layer3 = self.original_resnet.layer3
        self.layer4 = self.original_resnet.layer4
        self.avgpool = self.original_resnet.avgpool

        self.fc = nn.Linear(self.original_resnet.fc.in_features, self.output_features, bias=True)

    def forward(self, x: torch.Tensor, params_input: torch.Tensor) -> torch.Tensor:
        # ResNet input layers
        x = self.conv1(x) 
        x = self.bn1(x) 
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet layers with custom parameter blocks
        x1 = self.layer1(x) # Shape: (1, 64, 32, 32)
        x = self.params_block(x1, params_input, layer=1)

        x2 = self.layer2(x) # Shape: (1, 128, 16, 16)
        x = self.params_block(x2, params_input, layer=2)

        x3 = self.layer3(x) # Shape: (1, 256, 8, 8)
        x = self.params_block(x3, params_input, layer=3)

        x4 = self.layer4(x) # Shape: (1, 512, 4, 4)
        x = self.params_block(x4, params_input, layer=4)

        # Final pooling and FC layer
        x = self.avgpool(x) # Shape: (1, 512, 1, 1)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
    def params_block(self, x: torch.Tensor, params_input: torch.Tensor, layer: int) -> torch.Tensor:
        """
        Combines the output of the ResNet layer with the output from the custom parameter embedding.

        Parameters:
        ----------
        x : torch.Tensor
            The input tensor from the previous ResNet layer.
        params_input : torch.Tensor
            The input data for the parameter embedding.
        layer : int
            The ResNet layer number (used to access the correct ParametersFCModule).

        Returns:
        ----------
        torch.Tensor
            The output tensor after combining the ResNet layer and the parameter embedding output.
        """
        
        # Process through the parameter module
        params_out = self.parameters_module_dict[f'layer{layer}'](params_input)
        params_out = params_out.unsqueeze(-1).unsqueeze(-1)  # Add height and width dimensions
        params_out = params_out.expand(-1, -1, x.size(2), x.size(3))  # Broadcast to match ResNet output
        
        # Combine ResNet output with parameter output in a residual manner
        x = x + params_out

        return x


class ParametersFCModule(nn.Module):
    def __init__(self, output_features: int, input_features: int = 2, 
                 inter_num_neurons1: int = 50, inter_num_neurons2: int = 50) -> None:
        """
        A fully connected neural network module that embeds external parameters with batch normalization and ReLU.

        Parameters:
        ----------
        output_features : int
            The number of output neurons, which should match the ResNet layer's channel size.
        input_features : int, optional
            The number of input neurons, default is 2.
        inter_num_neurons1 : int, optional
            The number of neurons in the first hidden layer, default is 50.
        inter_num_neurons2 : int, optional
            The number of neurons in the second hidden layer, default is 50.
        """
        super(ParametersFCModule, self).__init__()

        # Define the fully connected layers with batch normalization and ReLU activations
        self.network = nn.Sequential(
            nn.Linear(input_features, inter_num_neurons1),
            nn.BatchNorm1d(inter_num_neurons1),
            nn.ReLU(inplace=True),
            nn.Linear(inter_num_neurons1, inter_num_neurons2),
            nn.BatchNorm1d(inter_num_neurons2),
            nn.ReLU(inplace=True),
            nn.Linear(inter_num_neurons2, output_features)  # Output matches ResNet layer channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.network(x)