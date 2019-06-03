import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from base import BaseModel

class baseLSTM(nn.Module):
    def __init__(self, n_features, n_segments, use_cuda = False):
        super(baseLSTM, self).__init__()
        self.hidden_dim = 128
        self.num_layers = 256
        self.use_cuda = use_cuda

        # Define the LSTM layer
        self.lstm = nn.LSTMCell(n_features, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):

        if self.use_cuda and torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.hidden_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.hidden_dim, x.size(0), self.hidden_dim))

        hn = h0[0, :]

        if self.use_cuda and torch.cuda.is_available():
            c0 = Variable(torch.zeros(self.hidden_dim, x.size(0), self.hidden_dim).cuda())
        else:
            c0 = Variable(torch.zeros(self.hidden_dim, x.size(0), self.hidden_dim))

        cn = c0[0, :]

        hn, cn = self.lstm(x, (hn, cn))
        out = self.linear(hn)
        out = torch.sigmoid(out)
        return out