
from torch import nn

class PlayerLinear(nn.Module):
    def __init__(self, input_size, N, device="cpu"):
        super(PlayerLinear, self).__init__()

        self.input_size = input_size*N
        self.output_size = input_size
        self.device = device


        # Output must be same size as input (predictions for all features)
        self.out = nn.Linear(self.input_size, self.output_size)


    def forward(self, x):
        #if x is a 3D tensor, flatten dim 2 and 3
        if len(x.size()) == 3:
            x = x.view(x.size(0), -1)
        
        #https://www.educative.io/answers/how-to-build-an-lstm-model-using-pytorch
        out = self.out(x)
        return out
    
    """
    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden)
        output = self.mid(output)
        output = self.out(output)
        return output, hidden
    """