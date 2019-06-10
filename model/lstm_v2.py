import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from base import BaseModel


class LSTMModel(nn.Module):
    def __init__(self, n_features, batch_size, bidirectional=False, num_layer=3, use_cuda=True):
        super(LSTMModel, self).__init__()
        self.use_cuda = use_cuda
        self.hidden_dim = 129
        self.nb_lstm_layers = num_layer
        self.batch_size = batch_size

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=self.nb_lstm_layers,
            batch_first=True,
            bidirectional=bidirectional)
        self.get_linear_input(bidirectional)
        self.linear = nn.Linear(self.hidden_dim * self.linear_multiplier, n_features)

    def get_linear_input(self, bidirectional):
        self.multiplier = self.nb_lstm_layers
        self.linear_multiplier = 1
        if bidirectional:
            self.multiplier *= 2
            self.linear_multiplier = 2

    def init_hidden(self, batch_size):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        h0 = torch.zeros(self.multiplier, batch_size, self.hidden_dim)
        c0 = torch.zeros(self.multiplier, batch_size, self.hidden_dim)

        if self.use_cuda and torch.cuda.is_available():
            h0 = h0.cuda()
            c0 = c0.cuda()

        h0 = Variable(h0)
        c0 = Variable(c0)

        return (h0, c0)

    def forward(self, X):
        batch_size, segments, num_features = X.size()

        self.hidden = self.init_hidden(batch_size)

        X, cn = self.lstm(X, self.hidden)
        self.hidden = (X, cn)
        out = self.linear(X)
        out = out.view(batch_size, segments, num_features)

        return out
