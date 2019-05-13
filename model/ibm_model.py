import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class FullyConnectedIBMBaseline(BaseModel):
    def __init__(self, n_features, n_segments):
        super(FullyConnectedIBMBaseline, self).__init__()
        self.fc1 = nn.Linear(n_features*n_segments, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc2_bn = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, n_features)

    def forward(self, x):
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x))
        return x