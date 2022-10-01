import torch
import torch.nn as nn
import torch.nn.functional as F


class SeqClassifier(nn.Module):
    def __init__(self, input_dim=4, hidden_dim1=16, hidden_dim2=32, output_dim=10, dropout=0.05):
        super(SeqClassifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.conv = nn.Sequential(nn.Conv1d(input_dim, hidden_dim1, 5, stride=1),
                                  nn.ReLU(),
                                  nn.Conv1d(hidden_dim1, hidden_dim2, 5, stride=1),
                                  nn.ReLU(),
                                  nn.MaxPool1d(3, stride=2))
        self.predict = nn.Sequential(nn.Flatten(),
                                     nn.Linear(31 * hidden_dim2, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 10),
                                     nn.Softmax(dim=1))

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)


    def forward(self, x):
        x = self.transformer_encoder(x)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.predict(x)

        return x
