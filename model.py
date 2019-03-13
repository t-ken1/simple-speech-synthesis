import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SimpleRNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_layers=1, bidirectional=True):
        super().__init__()

        self.num_direction = 2 if bidirectional else 1

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers, dropout=0.2,
                           bidirectional=bidirectional, batch_first=True)
        self.fc2 = nn.Sequential(
            nn.Linear(self.num_direction * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def _forward(self, sequences, lengths, hidden=None, pad_value=9999):
        T = sequences.shape[1]
        output = self.fc1(sequences)
        output = pack_padded_sequence(output, lengths, batch_first=True)
        output, _ = self.rnn(output, hidden)
        output, _ = pad_packed_sequence(output, batch_first=True,
                                        padding_value=pad_value,
                                        total_length=T)
        output = self.fc2(output)

        return output

    def forward(self, sequences, lengths, mask, *,
                hidden=None, pad_value=9999):
        output = self._forward(sequences, lengths, hidden, pad_value)
        output[mask] = pad_value

        return output

    def predict(self, sequences, lengths, *, hidden=None, pad_value=9999):
        return self._forward(sequences, lengths, hidden, pad_value)
    
    
