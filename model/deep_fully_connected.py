import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from base import BaseModel

class DeepFullyConnectedBaseline(BaseModel):
    def __init__(self, n_features, n_segments):
        super(FullyConnectedBaseline, self).__init__()
        self.fc1 = nn.Linear(n_features * n_segments, 2048)
        self.fc1_bn = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc2_bn = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, n_features)
        self.fc3_bn = nn.BatchNorm1d(2048)
        self.fc4 = nn.Linear(2048, n_features)
        self.fc4_bn = nn.BatchNorm1d(2048)
        self.fc5 = nn.Linear(2048, n_features)
        self.fc5_bn = nn.BatchNorm1d(2048)
        self.fc6 = nn.Linear(2048, n_features)
        self.fc6_bn = nn.BatchNorm1d(2048)
        self.fc7 = nn.Linear(2048, n_features)
        self.fc7_bn = nn.BatchNorm1d(2048)
        self.fc8 = nn.Linear(2048, n_features)
        self.fc8_bn = nn.BatchNorm1d(2048)
        self.fc9 = nn.Linear(2048, n_features)
        self.fc9_bn = nn.BatchNorm1d(2048)
        self.fc10 = nn.Linear(2048, n_features)
        self.fc10_bn = nn.BatchNorm1d(2048)
        self.fc11 = nn.Linear(2048, n_features)
        self.fc11_bn = nn.BatchNorm1d(2048)
        self.fc12 = nn.Linear(2048, n_features)
        self.fc13_bn = nn.BatchNorm1d(2048)
        self.fc14 = nn.Linear(2048, n_features)
        self.fc14_bn = nn.BatchNorm1d(2048)
        self.fc15 = nn.Linear(2048, n_features)
        self.fc15_bn = nn.BatchNorm1d(2048)
        self.fc16 = nn.Linear(2048, n_features)

    def forward(self, x):
        x = F.leaky_relu(self.fc1_bn(self.fc1(x)))
        x = F.leaky_relu(self.fc2_bn(self.fc2(x)))
        x = F.leaky_relu(self.fc3_bn(self.fc3(x)))
        x = F.leaky_relu(self.fc4_bn(self.fc4(x)))
        x = F.leaky_relu(self.fc5_bn(self.fc5(x)))
        x = F.leaky_relu(self.fc6_bn(self.fc6(x)))
        x = F.leaky_relu(self.fc7_bn(self.fc7(x)))
        x = F.leaky_relu(self.fc8_bn(self.fc8(x)))
        x = F.leaky_relu(self.fc9_bn(self.fc9(x)))
        x = F.leaky_relu(self.fc10_bn(self.fc10(x)))
        x = F.leaky_relu(self.fc11_bn(self.fc11(x)))
        x = F.leaky_relu(self.fc12_bn(self.fc12(x)))
        x = F.leaky_relu(self.fc13_bn(self.fc13(x)))
        x = F.leaky_relu(self.fc14_bn(self.fc14(x)))
        x = F.leaky_relu(self.fc15_bn(self.fc15(x)))
        x = self.fc16(x)
        return x
