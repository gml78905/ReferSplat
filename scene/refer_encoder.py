import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGeoEncoder(nn.Module):
    def __init__(self):
        super(SimpleGeoEncoder, self).__init__()
        self.fc1 = nn.Linear(13, 64)
        self.fc2 = nn.Linear(64, 16)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

