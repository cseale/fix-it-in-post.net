import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from base import BaseModel


class ConvolutionalShallow_Time(BaseModel):
    def __init__(self, n_features, n_segments):
        super(ConvolutionalShallow_Time, self).__init__()
        self.n_features = n_features
        self.n_segments = n_segments

        self.conv1 = nn.Conv2d(1, 18, kernel_size=(9, 5), padding=(4, 0))
        self.conv2 = nn.Conv2d(18, 30, kernel_size=(9, 3), padding=(4, 0))
        self.conv3 = nn.Conv2d(30, 40, kernel_size=(9, 2), padding=(4, 0))

        self.fc1 = nn.Linear(40 * n_features, 1024)
        self.fc2 = nn.Linear(1024, n_features)

        self.conv1_bn = nn.BatchNorm2d(18)
        self.conv2_bn = nn.BatchNorm2d(30)
        self.conv3_bn = nn.BatchNorm2d(40)
        self.fc1_bn = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, self.n_features, -1)

        x = F.leaky_relu(self.conv1_bn(self.conv1(x)))
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)))
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)))

        x = x.view(-1, 40 * self.n_features)

        x = F.leaky_relu(self.fc1_bn(self.fc1(x)))
        x = self.fc2(x)
        return x
