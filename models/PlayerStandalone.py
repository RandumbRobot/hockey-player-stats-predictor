
from torch import nn
import torch

"""
LSTM model for player data. Accepts variable N. Uses only player data to make predictions.
"""

class PlayerStandalone(nn.Module):
    def __init__(self, input_size, hidden_size, MLP_hidden_size, num_layers=1, dropout=0, device="cpu"):
        super(PlayerStandalone, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.MLP_hidden_size = MLP_hidden_size
        self.num_layers = num_layers
        self.device = device

        # LSTM input
        # Input size: size of each element in time series
        # Hidden size: size of LSTM cell output and hidden output
        # Num layers: number of stacked LSTM cells (vertical, not horizontal). Can also use multiple explicit LSTM layers
        # (https://stackoverflow.com/questions/49224413/difference-between-1-lstm-with-num-layers-2-and-2-lstms-in-pytorch)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)

        # midpoint betweem LSTM and output layer
        self.mid = nn.Sequential(
            nn.Linear(hidden_size, MLP_hidden_size),
            nn.ReLU()
        )
        

        # Output must be same size as input (predictions for all features)
        self.out = nn.Linear(MLP_hidden_size, input_size)


    def forward(self, x):
        #https://www.educative.io/answers/how-to-build-an-lstm-model-using-pytorch
        
        # Initialize 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()

        # Device
        h0 = h0.to(self.device)
        c0 = c0.to(self.device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Select only the output of the last LSTM cycle
        out = out[:, -1, :]

        out = self.mid(out)
        out = self.out(out)
        return out
    
    """
    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden)
        output = self.mid(output)
        output = self.out(output)
        return output, hidden
    """