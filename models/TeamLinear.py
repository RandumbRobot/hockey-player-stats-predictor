
from torch import nn

class TeamLinear(nn.Module):
    def __init__(self, input_size, output_size, N, device="cpu"):
        super(TeamLinear, self).__init__()

        self.input_size = input_size*N
        self.output_size = output_size
        self.device = device

        self.out = nn.Linear(self.input_size, self.output_size)


    def forward(self, x):
        #if x is a 3D tensor, flatten dim 2 and 3
        if len(x.size()) == 3:
            x = x.view(x.size(0), -1)
        
        out = self.out(x)
        return out