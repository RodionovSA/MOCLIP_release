import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from omegaconf import DictConfig


class EML_Dataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root #Directory with images
        self.transform = transform #Transformation
        self.Data = np.load(root, allow_pickle=True)

        self.geometries = self.Data['geometries'] 
        self.cell_images = self.Data['cell_images']
        self.spectra_x = self.Data['spectra_x']
        self.spectra_y = self.Data['spectra_y']

        
    def __len__(self):
        return self.spectra_x.shape[1]
    
    def __getitem__(self, index):
        geom = torch.from_numpy(self.cell_images[index]).unsqueeze(0).type(torch.float)
        params = np.array([self.geometries[index][-2] / 1000, 
                        self.geometries[index][-1] / 1000]) #in um
        params = torch.from_numpy(params).type(torch.float)
        spectra_x = torch.from_numpy(self.spectra_x[:, index]).type(torch.float)
        spectra_y = torch.from_numpy(self.spectra_y[:, index]).type(torch.float)
            
        spectra = torch.cat((spectra_x, spectra_y), axis=0)
                                     
        if self.transform:
            geom = self.transform(geom)

        return geom, params, spectra_x, spectra_y, spectra
    

def build_dataset(cfg) -> Dataset:
    if cfg.name == 'EML':
        # Convert transform config to actual transform
        if isinstance(cfg.transform, DictConfig):
            transform = transforms.Compose([
                transforms.Resize(cfg.transform.resize)
            ])
        else:
            transform = cfg.transform
            
        return EML_Dataset(cfg.root, transform)
    else:
        raise ValueError(f"Dataset {cfg.name} not found")
    
def build_loader(dataset: Dataset, 
                 batch_size: int, 
                 shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, 
                      batch_size=batch_size, 
                      shuffle=shuffle,
                      num_workers=2,
                      pin_memory=True)