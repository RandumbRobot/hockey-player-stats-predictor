import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class PlayerDatasets():
    def __init__(self, df, NL=[5],start_season=1990, stop_season=2023):
        """
        Players dataset
        
        param file: path to preprocessed players dataset
        param NL: List of N: number of consecutive seasons to load per group. Note that the associated label will be the season N+1
        return: dataset
        """
        self.NL = NL
        self.start_season = start_season
        self.stop_season = stop_season
        
        # Save the default full data
        self.alldata = df
        self.all_data_normalized = df.copy()
        
        #normalize features
        # features 'GP', get mean zero, unit variance normalization
        # features 'G', 'A', 'PTS', 'PS', 'EV', 'PP', 'S' get ln(x+1) normalization, followed by mean zero, unit variance normalization
        
        print("Normalizing features")
        self.log_scale_feature_names = ['G', 'A', 'PTS', 'PS', 'EV', 'PP', 'S']
        
        self.mins_for_log = self.all_data_normalized[self.log_scale_feature_names].min()
        
        
        for feature_name in self.log_scale_feature_names:
            new_np_values = np.log(self.all_data_normalized[feature_name].values - self.mins_for_log[feature_name] + 1)
            if np.any(np.isnan(new_np_values)):
                print(f'Feature {feature_name} has NaN values.')
                print(self.all_data_normalized[feature_name].values)
                print(new_np_values)
                raise ValueError('NaN values found in features.')
            self.all_data_normalized[feature_name] = new_np_values
        
        self.means = self.all_data_normalized.drop(columns=['Rk','Age', '+/-','Player', 'Season','Tm', 'PIM', 'Pos','SH','GW','EV.1','PP.1','SH.1','S%','TOI','ATOI'],axis=1).values.mean(axis=0)
        self.stds = self.all_data_normalized.drop(columns=['Rk','Age', '+/-','Player', 'Season','Tm', 'PIM', 'Pos','SH','GW','EV.1','PP.1','SH.1','S%','TOI','ATOI'],axis=1).values.std(axis=0)
        
        self.col_names = list(df.drop(columns=['Rk','Age', '+/-','Player', 'Season','Tm', 'PIM', 'Pos','SH','GW','EV.1','PP.1','SH.1','S%','TOI','ATOI'],axis=1).columns)
        print(self.col_names)
        
        
        
        for feature_name in ['GP', 'G', 'A', 'PTS', 'PS', 'EV', 'PP', 'S']:
            self.all_data_normalized[feature_name] = ((self.all_data_normalized[feature_name] - self.means[self.col_names.index(feature_name)]) / self.stds[self.col_names.index(feature_name)])
            
        
        # Get all the possible groups
        print('Loading player data')
        self.data = self.load_player_data(self.all_data_normalized, self.start_season, self.stop_season, self.NL)
    
    def set_seasons_per_group(self, NL):
        """
        Set the number of consecutive seasons (N) to load per group.
        This function also alters self.data by converting the original data to the new list of Ns
        
        param NL: List of N values, where each N is a number of seasons per group
        """
        self.NL = NL
        
        # Get all the possible groups
        self.data = self.load_player_data(self.all_data_normalized, self.start_season, self.stop_season, self.NL)
    
    def load_player_data(self, df: pd.DataFrame, start_season: int, stop_season: int, NL: list):
        """
        load_player_data: loads a dictionary of player data by grouping multiple groups of 
        N seasons together for a range of seasons. 
        NOTE: only players that appear for all seasons for a given group will be added
        
        param data: NHL player dataset for all years
        param start_seasons: First season to consider (inclusive).
        param end_seasons: Last season to consider (inclusive).
        param NL: List of N values, where each N is a number of seasons per group
        return: returns a data structure with the following format (for num_seasons=N=5)
            
        To ensure that separate data is used for training and validation, the split has the
        following levels of complexity:
        dict of players -> dict of N -> samples for N
        
        Therefore, the final dataset must have a form:
            NOTE: a group of seasons for a given player is considered a sample
            Player1:
                N=3:
                    Sample: Group 1990 -> 1990 to 1992 (target is 1993)
                    features: [1990, 1991, 1992]
                    target: 1993
                N=4:
                    ...
            
            Player2:
                N=3:
                    ...
            
            We can then get separate the data into players for training and validation, and then into
        """
        # get player names
        player_names = df['Player'].unique()
        
        #create dictionary with each player as key and an array of their statis as an entry
        # players = {key: pd.DataFrame([row for idx,row in df.iterrows() if row['Player']==key]) for key in player_names}
        
        #create dataset structure
        print('creating dataset structure')
        player_dict_dataset = {}
        for player in player_names:
            player_dict_dataset[player] = {}
            for N in NL:
                player_dict_dataset[player][N] = {}
                # for season in range(start_season, stop_season + 1):
                #     player_dict_dataset[player][N][season] = None
        
        print('creating player dict')
        #for each item
        for player in player_names:
            player_data = df[df['Player'] == player]
            #for each N
            for N in NL:
                #for each season
                for season in range(start_season, stop_season - N + 2):
                    features = []
                    for s in range(season, season + N):
                        season_data = player_data.loc[player_data['Season'] == s]
                        if season_data.shape[0] > 1:
                            break
                        if season_data.empty:
                            break
                        # add normalized features
                        features.append(torch.tensor(season_data.drop(columns=['Rk','Age', '+/-','Player', 'Season','Tm', 'PIM', 'Pos','SH','GW','EV.1','PP.1','SH.1','S%','TOI','ATOI'],axis=1).values, dtype=torch.float32))
                    
                    if len(features) != N:
                        continue
                    
                    target = player_data[player_data['Season'] == season + N]
                    if target.empty:
                        continue
                    
                    #add data to the dataset and normalize
                    player_dict_dataset[player][N][season] = (
                        torch.stack(features).reshape((len(features), len(features[0][0]))), #features
                        (torch.tensor(target.drop(columns=['Rk','Age', '+/-','Player', 'Season','Tm', 'PIM', 'Pos','SH','GW','EV.1','PP.1','SH.1','S%','TOI','ATOI'],axis=1).values, dtype=torch.float32)[0])
                    )
                    
                   

        # remove empty groups, N, and players
        for player in player_names:
            for N in NL:
                if player_dict_dataset[player][N] == {}: 
                    del player_dict_dataset[player][N]
            if player_dict_dataset[player] == {}:
                del player_dict_dataset[player]
        
        return player_dict_dataset
    
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
        tensor = old_tensor.clone()
        if feature_name is not None:
            tensor = tensor * self.stds[self.col_names.index(feature_name)] + self.means[self.col_names.index(feature_name)]
            if feature_name in self.log_scale_feature_names:
                tensor = torch.exp(tensor) + self.mins_for_log[feature_name] - 1
            return tensor
        
        if len(tensor.shape) == 1:
            for feature_name in ['GP', 'G', 'A', 'PTS', 'PS', 'EV', 'PP', 'S']:
                tensor[self.col_names.index(feature_name)] = tensor[self.col_names.index(feature_name)] * self.stds[self.col_names.index(feature_name)] + self.means[self.col_names.index(feature_name)]
                if feature_name in self.log_scale_feature_names:
                    tensor[self.col_names.index(feature_name)] = torch.exp(tensor[self.col_names.index(feature_name)]) + self.mins_for_log[feature_name] - 1
            return tensor
        
        for feature_name in ['GP', 'G', 'A', 'PTS', 'PS', 'EV', 'PP', 'S']:
            tensor[:,self.col_names.index(feature_name)] = tensor[:,self.col_names.index(feature_name)] * self.stds[self.col_names.index(feature_name)] + self.means[self.col_names.index(feature_name)]
            if feature_name in self.log_scale_feature_names:
                tensor[:,self.col_names.index(feature_name)] = torch.exp(tensor[:,self.col_names.index(feature_name)]) + self.mins_for_log[feature_name] - 1
            
        return tensor
                


class PlayerDataset(Dataset):
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

    # def set_seasons_per_group(self, N):
    #     self.N = N
    #     self.data = self.load_player_data(self.alldata, self.start_season, self.stop_season, self.N)

    # def load_player_data(self, df: pd.DataFrame, start_season: int, stop_season: int, N: int):
    #     players = df['Player'].unique()
    #     dataset = []

    #     for player in players:
    #         player_data = df[df['Player'] == player]
    #         for season in range(start_season, stop_season - N + 2):
    #             features = []
    #             for s in range(season, season + N):
    #                 season_data = player_data.loc[player_data['Season'] == s]
    #                 if season_data.shape[0] > 1:
    #                     break # Skip if player has multiple entries for a season
    #                 if season_data.empty:
    #                     break
    #                 features.append(torch.tensor(season_data.drop(columns=['Player', 'Season','Tm', 'Pos','S%','TOI','ATOI'],axis=1).values, dtype=torch.float32))

    #             if len(features) != N:
    #                 continue

    #             target = player_data[player_data['Season'] == season + N]
    #             if target.empty:
    #                 continue

    #             dataset.append(
    #                 [
    #                 torch.stack(features).reshape((len(features), len(features[0][0]))), torch.tensor(target.drop(columns=['Player', 'Season','Tm', 'Pos','S%', 'TOI','ATOI'],axis=1).values, dtype=torch.float32)[0]
    #                 ])
                
    #     #normalize features and targets to zero mean and unit variance using original df
    #     self.means = df.drop(columns=['Player', 'Season','Tm', 'Pos','S%','TOI','ATOI'],axis=1).values.mean(axis=0)
    #     self.stds = df.drop(columns=['Player', 'Season','Tm', 'Pos','S%','TOI','ATOI'],axis=1).values.std(axis=0)
    #     print(self.means)
    #     print(self.stds)
        
    #     for i in range(len(dataset)):
    #         dataset[i][0] = ((dataset[i][0] - self.means) / self.stds).float()
    #         dataset[i][1] = ((dataset[i][1] - self.means) / self.stds).float()
        
               
    #     return dataset
    
    

def get_player_dataset(file='./Data/player/processed/player_data.xlsx', NL=[5]):
    cols = [
        'Rk', 'Player', 'Age', 'Tm', 'Pos', 'GP', 'G', 'A', 'PTS', '+/-', 'PIM',
        'PS', 'EV', 'PP', 'SH', 'GW', 'EV.1', 'PP.1', 'SH.1', 'S', 'S%', 'TOI', 'ATOI', 'Season'
    ]
    print('reading file')
    df = pd.read_excel(file, header=0, usecols=cols)
    df = df.replace('\*', '', regex=True)
    df['original_index'] = df.index
    df = df.sort_values(by='GP', ascending=False)
    df = df.groupby(['Player', 'Age', 'Season']).filter(lambda x: len(x) == 1 or x['GP'].max() > 47)
    df = df.drop_duplicates(subset=['Player', 'Age', 'Season'], keep='first')
    df = df.sort_values(by='original_index')
    df = df.drop(columns=['original_index'])
    print(f'Loaded {df.shape[0]} rows.')
    dataset = PlayerDatasets(df,NL)
    
    return dataset

if __name__ == "__main__":
    print("nothing to run")
