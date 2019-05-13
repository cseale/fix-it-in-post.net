import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class FullyConnectedBaseline(BaseModel):
    def __init__(self, n_features, n_segments):
        super(FullyConnectedBaseline, self).__init__()
        self.fc1 = nn.Linear(n_features*n_segments, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc2_bn = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, n_features)

    def forward(self, x):
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.fc3(x)
        return x


class ConvolutionalBaseline(BaseModel):
    def __init__(self, n_features, n_segments):
        super(ConvolutionalBaseline, self).__init__()
        self.n_features = n_features
        self.n_segments = n_segments
        self.conv1 = nn.Conv2d(1, 18, kernel_size=(9, 8), padding=(4, 0))
        #self.conv1_bn = nn.BatchNorm2d(18)
        self.conv2 = nn.Conv2d(18, 30, kernel_size=(9, 1), padding=(4, 0))
        #self.conv2_bn = nn.BatchNorm2d(30)
        self.conv3 = nn.Conv2d(30, 40, kernel_size=(9, 1), padding=(4, 0))
        #self.conv3_bn = nn.BatchNorm2d(40)
        self.fc1 = nn.Linear(40 * n_features, 1024)
        #self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, n_features)

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, self.n_features, -1)
        
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))

        #x = F.leaky_relu(self.conv1_bn(self.conv1(x)))
        #x = F.leaky_relu(self.conv2_bn(self.conv2(x)))
        #x = F.leaky_relu(self.conv3_bn(self.conv3(x)))

        x = x.view(-1, 40 * self.n_features)
        
        x = F.leaky_relu(self.fc1(x))
        # x = F.leaky_relu(self.fc1_bn(self.fc1(x)))
        x = self.fc2(x)
        return x