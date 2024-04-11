
from torch import nn
import torch
from .PlayerStandalone import PlayerStandalone
from .TeamStandalone import TeamStandalone

class PlayerCombined(nn.Module):
    def __init__(
        self, 
        player_input_size, player_hidden_size, player_MLP_hidden_size, 
        team_input_size, team_output_size, team_hidden_size,
        hidden_size, num_layers=1, 
        player_num_layers=1, player_dropout=0,
        team_num_layers=1, team_dropout=0, 
        device="cpu"
        ):
        
        super(PlayerCombined, self).__init__()
        
        self.player_input_size = player_input_size
        self.team_output_size = team_output_size
        
        #initialize submodules
        self.player_standalone = PlayerStandalone_Latent(player_input_size, player_hidden_size, player_MLP_hidden_size, player_num_layers, player_dropout, device)
        
        self.team_standalone = TeamStandalone_Latent(team_input_size, team_output_size, team_hidden_size, team_num_layers, team_dropout, device)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        # feed forward layer
        #dimensions: we are using the mid layer of PlayerStandalone of size player_MLP_hidden_size, and the mid layer of TeamStandalone of size 2*team_hidden_size
        self.mid = nn.Sequential(
            nn.Linear(player_MLP_hidden_size + 2*team_hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Output must be same size as input (predictions for all features)
        self.out = nn.Linear(hidden_size, player_input_size)


    def forward(self, x):
        #slice input tensor
        player_x = x[:, :, :self.player_input_size]
        team_x = x[:, :, self.player_input_size:]
        
        #go through submodules
        player_out = self.player_standalone(player_x)
        team_out = self.team_standalone(team_x)
        
        #concatenate player and team outputs
        combined_out = torch.cat((player_out, team_out), dim=1)
        
        out = self.mid(combined_out)
        out = self.out(out)
        
        return out

#override the forward method of PlayerStandalone
class PlayerStandalone_Latent(PlayerStandalone):
    # return mid layer instead of output layer
    def forward(self, x):
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
        # layer removed: out = self.out(out)
        return out

#override the forward method of TeamStandalone
class TeamStandalone_Latent(TeamStandalone):
    # return mid layer instead of output layer
    def forward(self, x):
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
        #layer removed: self.lin = nn.Linear(2*hidden_size, 4*output_size)
        # layer removed: self.out = nn.Linear(4*output_size, output_size)

        return out

        
    
    
