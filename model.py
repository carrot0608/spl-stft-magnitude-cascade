"""STFT-CNN and training input augmentation."""
import os
import random
import numpy as np
import torch
from torch import nn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG'):
        torch.use_deterministic_algorithms(True)


class STFTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(9,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32,48,3,padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1,1)))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(48,2))

    def forward(self, x):
        return self.head(self.backbone(x))


class STFTDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, training=False):
        self.x, self.y, self.training = x, y, training

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.x[i].copy()
        if self.training:
            x += np.random.normal(0,0.02,x.shape).astype(np.float32)
            x *= np.random.normal(1,0.05,(x.shape[0],1,1)).astype(np.float32)
            drop = np.random.rand(x.shape[0]) < 0.10
            if drop.all():
                drop[np.random.randint(0,x.shape[0])] = False
            x[drop] = 0
            if np.random.rand() < 0.25 and x.shape[-1] > 8:
                start = np.random.randint(0,x.shape[-1]-4)
                x[:,:,start:start+3] = 0
        return torch.from_numpy(x), torch.tensor(int(self.y[i]), dtype=torch.long)
