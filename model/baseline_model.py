import torch
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
        x = torch.sigmoid(self.fc3(x))
        return x


class ConvolutionalBaseline(BaseModel):
    def __init__(self, n_features, n_segments):
        super(ConvolutionalBaseline, self).__init__()
        self.n_features = n_features
        self.n_segments = n_segments
        self.conv1 = nn.Conv2d(1, 18, kernel_size=(9, 8), padding=(4, 0))
        self.conv1_bn = nn.BatchNorm2d(18)

        self.conv2 = nn.Conv2d(18, 30, kernel_size=(5, 1), padding=(2, 0))
        self.conv2_bn = nn.BatchNorm2d(30)

        self.conv3 = nn.Conv2d(30, 8, kernel_size=(9, 1), padding=(4, 0))
        self.conv3_bn = nn.BatchNorm2d(8)

        self.conv4 = nn.Conv2d(8, 18, kernel_size=(9, 1), padding=(4, 0))
        self.conv4_bn = nn.BatchNorm2d(18)

        self.conv5 = nn.Conv2d(18, 30, kernel_size=(5, 1), padding=(2, 0))
        self.conv5_bn = nn.BatchNorm2d(30)
        self.conv6 = nn.Conv2d(30, 8, kernel_size=(9, 1), padding=(4, 0))
        self.conv6_bn = nn.BatchNorm2d(8)
        self.conv7 = nn.Conv2d(8, 18, kernel_size=(9, 1), padding=(4, 0))
        self.conv7_bn = nn.BatchNorm2d(18)

        self.conv8 = nn.Conv2d(18, 30, kernel_size=(5, 1), padding=(2, 0))
        self.conv8_bn = nn.BatchNorm2d(30)
        self.conv9 = nn.Conv2d(30, 8, kernel_size=(9, 1), padding=(4, 0))
        self.conv9_bn = nn.BatchNorm2d(8)
        self.conv10 = nn.Conv2d(8, 18, kernel_size=(9, 1), padding=(4, 0))
        self.conv10_bn = nn.BatchNorm2d(18)

        self.conv11 = nn.Conv2d(18, 30, kernel_size=(5, 1), padding=(2, 0))
        self.conv11_bn = nn.BatchNorm2d(30)
        self.conv12 = nn.Conv2d(30, 8, kernel_size=(9, 1), padding=(4, 0))
        self.conv12_bn = nn.BatchNorm2d(8)
        self.conv13 = nn.Conv2d(8, 18, kernel_size=(9, 1), padding=(4, 0))
        self.conv13_bn = nn.BatchNorm2d(18)

        self.conv14 = nn.Conv2d(18, 30, kernel_size=(5, 1), padding=(2, 0))
        self.conv14_bn = nn.BatchNorm2d(30)

        self.conv15 = nn.Conv2d(30, 8, kernel_size=(9, 1), padding=(4, 0))
        self.conv15_bn = nn.BatchNorm2d(8)

        self.fc1 = nn.Linear(8 * n_features, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, n_features)

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, self.n_features, -1)
        
        """x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))

        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))
        x = F.leaky_relu(self.conv7(x))

        x = F.leaky_relu(self.conv8(x))
        x = F.leaky_relu(self.conv9(x))
        x = F.leaky_relu(self.conv10(x))

        x = F.leaky_relu(self.conv11(x))
        x = F.leaky_relu(self.conv12(x))
        x = F.leaky_relu(self.conv13(x))

        x = F.leaky_relu(self.conv14(x))
        x = F.leaky_relu(self.conv15(x)) """

        x = F.leaky_relu(self.conv1_bn(self.conv1(x)))
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)))
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)))
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)))

        x = F.leaky_relu(self.conv5_bn(self.conv5(x)))
        x = F.leaky_relu(self.conv6_bn(self.conv6(x)))
        x = F.leaky_relu(self.conv7_bn(self.conv7(x)))

        x = F.leaky_relu(self.conv8_bn(self.conv8(x)))
        x = F.leaky_relu(self.conv9_bn(self.conv9(x)))
        x = F.leaky_relu(self.conv10_bn(self.conv10(x)))

        x = F.leaky_relu(self.conv11_bn(self.conv11(x)))
        x = F.leaky_relu(self.conv12_bn(self.conv12(x)))
        x = F.leaky_relu(self.conv13_bn(self.conv13(x)))

        x = F.leaky_relu(self.conv14_bn(self.conv14(x)))
        x = F.leaky_relu(self.conv15_bn(self.conv15(x)))

        x = x.view(-1, 8 * self.n_features)
        
        x = F.leaky_relu(self.fc1_bn(self.fc1(x)))
        
        x = torch.sigmoid(self.fc2(x))
        return x


class ConvolutionalBaseline_TimeFiltering(BaseModel):
    def __init__(self, n_features, n_segments):
        super(ConvolutionalBaseline_TimeFiltering, self).__init__()
        self.n_features = n_features
        self.n_segments = n_segments
        self.conv1 = nn.Conv2d(1, 18, kernel_size=(9, 5), padding=(4, 0))
        #self.conv1_bn = nn.BatchNorm2d(18)
        self.conv2 = nn.Conv2d(18, 30, kernel_size=(9, 3), padding=(4, 0))
        #self.conv2_bn = nn.BatchNorm2d(30)
        self.conv3 = nn.Conv2d(30, 40, kernel_size=(9, 2), padding=(4, 0))
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
        x = torch.sigmoid(self.fc2(x))
        return x