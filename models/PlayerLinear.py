
from torch import nn
import torch

"""
Baseline linear model for player data. Pads input data with zeros if the number of seasons is less than the initialized N.
"""

class PlayerLinear(nn.Module):
    def __init__(self, input_size, N, device="cpu"):
        super(PlayerLinear, self).__init__()

        self.input_size = input_size*N
        self.output_size = input_size
        self.device = device
        self.N = N

        self.out = nn.Linear(self.input_size, self.output_size)


    def forward(self, x):
        
        #if x has length < N, pad with zeros
        if len(x[0]) < self.N:
            padded_x = []
            pads = [torch.zeros_like(x[0][0]) for i in range(self.N - len(x[0]))]
            #use pre-padding
            for i in range(x.size(0)):
                padded_x.append(torch.cat((torch.stack(pads), x[i]), dim=0))
            x = torch.stack(padded_x)
        
        #if x is a 3D tensor, flatten dim 2 and 3
        if len(x.size()) == 3:
            x = x.view(x.size(0), -1)
        
        out = self.out(x)
        return out
