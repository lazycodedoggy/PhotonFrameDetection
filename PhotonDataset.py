import sys
import logging

import torchvision.transforms as transforms
from torch.utils.data import Dataset

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class PhotonDataset(Dataset):
    def __init__(self, features, labels, winLabels):
        self._features = features
        self._labels = labels
        self._winLabels = winLabels
            
    def __len__(self):
        return len(self._features)

    def __getitem__(self, index):
        return self._features[index], self._labels[index], self._winLabels[index]