import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from base import BaseModel

# class baseLSTM(nn.Module):
#     def __init__(self, n_features, n_segments):
#         super(baseLSTM, self).__init__()
#         self.hidden_dim = 128
#         self.num_layers = 256

#         # Define the LSTM layer
#         self.lstm = nn.LSTMCell(n_features, self.hidden_dim, self.num_layers)

#         # Define the output layer
#         self.linear = nn.Linear(self.hidden_dim, n_features)

#     def forward(self, x):

#         if torch.cuda.is_available():
#             h0 = Variable(torch.zeros(self.hidden_dim, x.size(0), self.hidden_dim).cuda())
#         else:
#             h0 = Variable(torch.zeros(self.hidden_dim, x.size(0), self.hidden_dim))

#         hn = h0[0, :]

#         if torch.cuda.is_available():
#             c0 = Variable(torch.zeros(self.hidden_dim, x.size(0), self.hidden_dim).cuda())
#         else:
#             c0 = Variable(torch.zeros(self.hidden_dim, x.size(0), self.hidden_dim))

#         cn = c0[0, :]

#         hn, cn = self.lstm(x, (hn, cn))
#         out = self.linear(hn)

#         return out
    
class BieberLSTM(nn.Module):
    def __init__(self, n_features, batch_size):
        super(baseLSTM, self).__init__()
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
        hidden_a = torch.randn(self.nb_lstm_layers, self.batch_size, self.nb_lstm_units)

        if torch.cuda.is_available():
            hidden_a = hidden_a.cuda()

        hidden_a = Variable(hidden_a)

        return hidden_a

    def forward(self, X):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        self.hidden = self.init_hidden()

        batch_size, seq_len, _ = X.size()

        # ---------------------
        # 2. Run through RNN
        # TRICK 2 ********************************
        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, nb_lstm_units)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        X_lengths = (X[:,:,0].numpy() == -1).argmax(1)
        print("X_lengths: " + str(X_lengths))
        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True)

        # now run through LSTM
        X, self.hidden = self.lstm(X, self.hidden)

        # undo the packing operation
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

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