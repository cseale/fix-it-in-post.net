import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from base import BaseModel


class BieberLSTM(nn.Module):
    def __init__(self, n_features, batch_size, use_cuda=True):
        super(BieberLSTM, self).__init__()
        self.use_cuda = use_cuda
        self.nb_lstm_layers = 256
        self.nb_lstm_units = 128
        self.n_features = n_features
        self.batch_size = batch_size

        # design LSTM
        self.lstm = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.nb_lstm_units,
            num_layers=self.nb_lstm_layers,
            batch_first=True
        )

        # output layer which projects back to tag space
        self.linear = nn.Linear(self.nb_lstm_units, self.n_features)

    def init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        h0 = torch.randn(self.nb_lstm_layers, self.batch_size, self.nb_lstm_units)
        c0 = torch.randn(self.nb_lstm_layers, self.batch_size, self.nb_lstm_units)

        if self.use_cuda and torch.cuda.is_available():
            h0 = h0.cuda()
            c0 = c0.cuda()

        h0 = Variable(h0)
        c0 = Variable(c0)

        return (h0, c0)

    def forward(self, X):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        self.hidden = self.init_hidden()

        batch_size, seq_len, _ = X.size()

        # TODO: This is a hack and should be made variable. The problem is that the unpacking returns the variable to the length of the longest sequence.
        # Maybe try this fix: https://github.com/pytorch/pytorch/issues/1591
        total_length = X.shape[1]
        # ---------------------
        # 2. Run through RNN
        # TRICK 2 ********************************
        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, nb_lstm_units)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        X_lengths = (X[:, :, 0].cpu().numpy() == -1).argmax(1)
        # TODO: Hack 1, if sequence has max length, length needs to be set to total length
        # check if equal to zero, convert to binary, multiply by total length, add to original lengths
        X_lengths = ((X_lengths == 0) * 1) * total_length + X_lengths
        print(X_lengths)
        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True, enforce_sorted=False)
        # now run through LSTM

        X, cn = self.lstm(X, self.hidden)
        self.hidden = (X, cn)
        # undo the packing operation
        # TODO: Hack 2, set total_length manually
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True, total_length=total_length)
        # ---------------------
        # 3. Project to tag space
        # Dim transformation: (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)

        # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
        X = X.contiguous()
        X = X.view(-1, X.shape[2])
        # run through actual linear layer
        X = self.linear(X)
        # I like to reshape for mental sanity so we're back to (batch_size, seq_len, nb_tags)
        X = X.view(batch_size, seq_len, self.n_features)
        Y_hat = X
        return Y_hat
