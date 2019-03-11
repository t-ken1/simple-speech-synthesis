import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SimpleRNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_layers=1, bidirectional=True):
        super().__init__()

        self.num_direction = 2 if bidirectional else 1

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                           bidirectional=bidirectional, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, sequence, lengths, *, hidden=None, pad_value=9999):
        output = self.fc1(sequence)
        output = pack_padded_sequence(output, lengths, batch_first=True)
        output, _ = self.rnn(output, hidden)
        output, _ = pad_packed_sequence(output, batch_first=True,
                                        padding_value=pad_value)
        output = self.fc2(output)

        return output
    
