import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from base import BaseModel

class GRUModel(BaseModel):
    def __init__(self, n_features, n_segments):
        super(GRUModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = 128

        self.gru_cell = nn.GRUCell(n_features, self.hidden_dim)

        self.fc = nn.Linear(128, n_features)

    def forward(self, x):
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        hn = h0[0, :]
        hn = self.gru_cell(x, hn)
        out = hn

        out = self.fc(out)
        return out


