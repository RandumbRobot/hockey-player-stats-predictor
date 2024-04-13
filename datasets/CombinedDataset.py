import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from .TeamDataset import *
from .PlayerDataset import *

"""
This file contains the Dataset class for loading team and player data together, used by the combined models and baselines.
"""

team_name_dict = {
    'AFM': 'Atlanta Flames',
    'ANA': 'Mighty Ducks of Anaheim/Anaheim Ducks',
    'ARI' : 'Arizona Coyotes',
    'ATL' : 'Atlanta Thrashers',
    'BOS' : 'Boston Bruins',
    'BRK' : 'Brooklyn Americans',
    'BUF' : 'Buffalo Sabres',
    'CAR' : 'Carolina Hurricanes',
    'CBJ' : 'Columbus Blue Jackets',
    'CGS' : 'Bay Area Seals/California Golden Seals',
    'CGY' : 'Calgary Flames',
    'CHI' : 'Chicago Black Hawks/Blackhawks',
    'CLE' : 'Cleveland Barons',
    'CLR' : 'Colorado Rockies',
    'COL' : 'Colorado Avalanche',
    'DAL' : 'Dallas Stars',
    'DCG' : 'Detroit Cougars',
    'DET' : 'Detroit Red Wings',
    'DFL' : 'Detroit Falcons',
    'EDM' : 'Edmonton Oilers',
    'FLA' : 'Florida Panthers',
    'HAM' : 'Hamilton Tigers',
    'HFD' : 'Hartford Whalers',
    'KCS' : 'Kansas City Scouts',
    'LAK' : 'Los Angeles Kings',
    'MIN' : 'Minnesota Wild',
    'MMR' : 'Montreal Maroons',
    'MNS' : 'Minnesota North Stars',
    'MTL' : 'Montreal Canadiens',
    'MWN' : 'Montreal Wanderers',
    'NJD' : 'New Jersey Devils',
    'NSH' : 'Nashville Predators',
    'NYA' : 'New York Americans',
    'NYI' : 'New York Islanders',
    'NYR' : 'New York Rangers',
    'OAK' : 'California/Oakland Seals',
    'OTT' : 'Ottawa Senators',
    'PHI' : 'Philadelphia Flyers',
    'PHX' : 'Phoenix Coyotes',
    'PIR' : 'Pittsburgh Pirates',
    'PIT' : 'Pittsburgh Penguins',
    'QBD' : 'Quebec Bulldogs',
    'QUA' : 'Philadelphia Quakers',
    'QUE' : 'Quebec Nordiques',
    'SEA' : 'Seattle Kraken',
    'SEN' : 'Ottawa Senators (original)',
    'SLE' : 'St. Louis Eagles',
    'SJS' : 'San Jose Sharks',
    'STL' : 'St. Louis Blues',
    'TAN' : 'Toronto Hockey Club/Toronto Arenas',
    'TBL' : 'Tampa Bay Lightning',
    'TOR' : 'Toronto Maple Leafs',
    'TSP' : 'Toronto St. Patricks',
    'VAN' : 'Vancouver Canucks',
    'VGK' : 'Vegas Golden Knights',
    'WIN' : 'Winnipeg Jets (original)',
    'WPG' : 'Winnipeg Jets',
    'WSH' : 'Washington Capitals'
}


class CombinedDatasets():
    def __init__(self, dataset, team_dataset, player_dataset, NL=[5], start_season=1990, stop_season=2023):
        """
        Teams dataset

        :param dataset: full dataset
        :param targets: list of target features in dataset
        :param team_dataset: Associated team dataset
        :param player_dataset: Associated player dataset
        :param NL: list of number of consecutive seasons to load per group. Note that the associated label will be the season N+1
        :return: dataset
        """
        self.NL = NL
        self.data = dataset
        self.team_dataset = team_dataset
        self.player_dataset =player_dataset

    
    def random_split(self, ratio):
        """
        Splits the dataset into players for training and players for validation.
        :param ratio: the ratio of validation data over the total data available. Must be between 0 and 1.
        
        """
        if not (ratio > 0 and ratio < 1):
            raise ValueError("Ratio must be between 0 and 1")
            
        
        s = pd.Series(self.data)
        
        #split players into 2 datsaets
        training_data, test_data = [i.to_dict() for i in train_test_split(s, test_size=ratio)]
        
        #split datasets into the N groups
        N_datasets = []
        for N in self.NL:
            train_samples = []
            for player in training_data:
                #check if player has data for N seasons
                if N in training_data[player]:
                    for sample in training_data[player][N]:
                        train_samples.append(training_data[player][N][sample])
                        
            
            test_samples = []
            for player in test_data:
                if N in test_data[player]:
                    for sample in test_data[player][N]:
                        test_samples.append(test_data[player][N][sample])
            
            N_datasets.append((N, train_samples, test_samples))
        
        return N_datasets
    

    def unnormalize(self, old_tensor,feature_name=None):
        tensor = old_tensor.detach().clone()
        if feature_name is not None:
            tensor = tensor * self.stds[self.col_names.index(feature_name)] + self.means[self.col_names.index(feature_name)]
            if feature_name in self.log_scale_feature_names:
                tensor = torch.exp(tensor) + self.mins_for_log[feature_name] - 1
            return tensor
        
        if len(tensor.shape) == 1:
            for feature_name in self.targets:
                tensor[self.targets.index(feature_name)] = tensor[self.targets.index(feature_name)] * self.stds[self.col_names.index(feature_name)] + self.means[self.col_names.index(feature_name)]
                if feature_name in self.log_scale_feature_names:
                    tensor[self.targets.index(feature_name)] = torch.exp(tensor[self.targets.index(feature_name)]) + self.mins_for_log[feature_name] - 1
            return tensor
        
        for feature_name in self.targets:
            tensor[:,self.targets.index(feature_name)] = tensor[:,self.targets.index(feature_name)] * self.stds[self.col_names.index(feature_name)] + self.means[self.col_names.index(feature_name)]
            if feature_name in self.log_scale_feature_names:
                tensor[:,self.targets.index(feature_name)] = torch.exp(tensor[:,self.targets.index(feature_name)]) + self.mins_for_log[feature_name] - 1
            
        return tensor
                


class CombinedDataset(Dataset):
    def __init__(self, dataset, max_N):
        self.data = dataset
        self.max_N = max_N

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_element = self.data[idx]
        if len(data_element[0]) < self.max_N:
            pads = [torch.zeros_like(data_element[0][0]) for i in range(self.max_N - len(data_element[0]))]
            #use pre-padding
            padded_data_element_zero = torch.cat((torch.stack(pads), data_element[0]), dim=0)
        else:
            padded_data_element_zero = data_element[0]
        return padded_data_element_zero, data_element[1]    
    

def get_combined_dataset(NL=[1,2,3,4,5]):
    
    # PLAYER DATASET
    player_dataset = get_player_dataset(NL=NL)
    
    # TEAM DATASET
    team_dataset = get_team_dataset(NL=[1]) # Only need one season (ensures all teams exists in dataset)

    # Append team features to player samples
    player_df = player_dataset.alldata
    team_df = team_dataset.alldata

    # Create deep copy of team dataset
    dataset = player_dataset.data.copy()

    # Perform team/player data linking
    samples_to_drop = []
    for player in dataset:
        for N in dataset[player]:

            # For each group 
            for year in dataset[player][N]:

                # Find the team names for that group
                player_data = player_df[player_df['Player'] == player]
                player_data_year = player_data.loc[player_data['Season'].isin([season for season in range(year, year+N)])]
                team_names = player_data_year['Tm'].values

                # Get the team stats for each season
                drop = 0
                features = []
                for season,team in enumerate(team_names):

                    if team not in team_name_dict:
                        # drop sample and clean up
                        drop = 1
                        samples_to_drop.append((player, N, year))
                        break

                    team_name = team_name_dict[team]

                    # Get normalized team data
                    team_data = team_dataset.all_data_normalized[team_dataset.all_data_normalized['team name'] == team_name]
                    team_data_year = team_data[team_data['Season'] == year+season]

                    if team_data_year.empty:
                        # drop sample and clean up
                        drop = 1
                        samples_to_drop.append((player, N, year))
                        break
                    
                    features.append(torch.tensor(team_data_year.drop(columns=['team name', 'Season'],axis=1).values, dtype=torch.float32))

                if drop == 1:
                    continue

                # Append the team stats to the player stats
                try:
                    team_features = torch.stack(features).reshape((len(features), len(features[0][0]))) #features
                except:
                    print(features)
                    exit(0)
                player_features = dataset[player][N][year][0]
                new_features = torch.hstack((player_features, team_features))
                dataset[player][N][year] = (new_features , dataset[player][N][year][1])

    # cleanup dataset
    for (player, N, year) in samples_to_drop:
        dataset[player][N].pop(year, None)
        if len(dataset[player][N]) == 0: # remove N if empty
            dataset[player].pop(N, None)
            if len(dataset[player]) == 0: # remove player if empty
                dataset.pop(player, None)

    dataset = CombinedDatasets(dataset, team_dataset, player_dataset, NL)
    return dataset

if __name__ == "__main__":

    dataset = get_combined_dataset(NL=[5])
