import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np



class TeamDatasets():
    def __init__(self, df, targets, NL=[5], start_season=1990, stop_season=2023):
        """
        Teams dataset

        :param df: dataframe object of dataset
        :param targets: list of target features in dataset
        :param N: number of consecutive seasons to load per group. Note that the associated label will be the season N+1
        :return: dataset
        """

        print("Targets")
        print(targets)

        self.NL=NL
        self.targets=targets
        self.start_season=start_season
        self.stop_season=stop_season

        # Save the default full data
        self.alldata = df
        self.all_data_normalized = df.copy()


        print("Normalizing features")
        self.log_scale_feature_names = []
        print(self.log_scale_feature_names)
        
        self.mins_for_log = self.all_data_normalized[self.log_scale_feature_names].min()
        
        
        for feature_name in self.log_scale_feature_names:
            new_np_values = np.log(self.all_data_normalized[feature_name].values - self.mins_for_log[feature_name] + 1)
            if np.any(np.isnan(new_np_values)):
                print(f'Feature {feature_name} has NaN values.')
                print(self.all_data_normalized[feature_name].values)
                print(new_np_values)
                raise ValueError('NaN values found in features.')
            self.all_data_normalized[feature_name] = new_np_values
        
        self.means = self.all_data_normalized.drop(columns=['team name', 'Season'],axis=1).values.mean(axis=0)
        self.stds = self.all_data_normalized.drop(columns=['team name', 'Season'],axis=1).values.std(axis=0)
        
        print("All features")
        self.col_names = list(df.drop(columns=['team name', 'Season'],axis=1).columns)
        print(self.col_names)
        
        for feature_name in self.col_names:
            self.all_data_normalized[feature_name] = ((self.all_data_normalized[feature_name] - self.means[self.col_names.index(feature_name)]) / self.stds[self.col_names.index(feature_name)])
            
        
        # Get all the possible groups
        print('Loading player data')
        self.data = self.load_team_data(self.all_data_normalized, self.start_season, self.stop_season, self.NL)
    
    def set_seasons_per_group(self, NL):
        """
        Set the number of consecutive seasons (N) to load per group.
        This function also alters self.data by converting the original data to the new list of Ns

        :param NL: List of N values, where each N is a number of seasons per group
        """
        self.NL = NL

        # Get all the possible groups
        self.data = self.load_team_data(self.alldata, self.start_season, self.stop_season, self.NL)


    def load_team_data(self, df: pd.DataFrame, start_season: int, stop_season: int, NL: list):
        """
        load_team_data: loads a dictionary of team data by grouping multiple groups of 
        N seasons together for a range of seasons. 
        NOTE: only teams that appear for all seasons for a given group will be added

        :param data: NHL team dataset for all years
        :param start_seasons: First season to consider (inclusive).
        :param end_seasons: Last season to consider (inclusive).
        :param NL: List of N values, where each N is a number of seasons per group
        :return: returns a data structure with the following format (for num_seasons=N=5)
            
        To ensure that separate data is used for training and validation, the split has the
        following levels of complexity:
        dict of teams -> dict of N -> samples for N

        Therefore, the final dataset must have a form:
            NOTE: a group of seasons for a given team is considered a sample
            Team1:
                N=3:
                    Sample: Group 1990 -> 1990 to 1992 (target is 1993)
                        feature: array of stats for N seasons (1990 to 1990+N-1)
                        label: stats for season N+1
                N=4:
                    ...

            Team2:
                N=3:
                    ...
        
        We can then get separate the data into teams for training and validation, and then into
        batches with the same N value
        """ 


        # Get all the teams names
        team_names = list(set(df.loc[:, 'team name']))

        # Create dictionary with each team as key and an array of all their stats as an entry
        #teams = {key: pd.DataFrame([row for idx,row in df.iterrows() if row['team name']==key]) for key in team_names}


        # Create dataset structure
        print('creating dataset structure')
        team_dict_dataset = {}
        for team in team_names: # Create team level dictionary
            team_dict_dataset[team] = {}
            for N in NL: # Create N level dictionary
                team_dict_dataset[team][N] = {}
                #for season in range(start_season, stop_season+1):
                #    team_dict_dataset[team][N][season] = None # No sample yet

        # For each team
        #for team in team_names:
        #    df = teams[team]
        for team in team_names:
            team_data = df[df['team name'] == team]

            # For each N
            for N in NL:

                # Get groups of N seasons
                for season in range(start_season, stop_season-N+2, 1):
                        
                    # For the current group of seasons
                    features = []
                    for s in range(season, season+N):
                        ss = team_data.loc[team_data['Season']==s]
                        if ss.empty: # abort if inexistant season
                            break
                    
                        features.append(torch.tensor(ss.drop(['team name', 'Season'], axis=1).values, dtype=torch.float32))
                    
                    # Skip if not enough consecutive seasons (inexistant season)
                    if not len(features) == N:
                        # TODO: season+=N for faster iterations?
                        continue

                    # Add target
                    target = team_data.loc[team_data['Season']==season+N]
                    if target.empty: # abort if inexistant season
                        continue

                    # Add data to dataset
                    team_dict_dataset[team][N][season] = (
                        torch.stack(features).reshape((len(features),len(features[0][0]))), # features
                        torch.tensor(target[self.targets].values, dtype=torch.float32)[0] # target
                        )

        # Remove empty groups, N and teams
        for team in team_names: 
            for N in NL: 
                team_dict_dataset[team][N] = {k: v for k, v in team_dict_dataset[team][N].items() if v is not None} # remove empty groups
        #    team_dict_dataset[team] = {k: v for k, v in team_dict_dataset[team].items() if not not v} #remove empty N
        #team_dict_dataset = {k: v for k, v in team_dict_dataset.items() if not not v} #remove empty teams
    
        return team_dict_dataset

    def random_split(self, ratio):
        """
        Splits the dataset into teams for training and teams for validation.

        :param: ratio: ratio of validation data over total data available. Must be between 0 and 1

        :return:
        """

        if not(ratio > 0 and ratio < 1):
            print("RATIO MUST BE BETWEEN 0 and 1")
            return None, None

        s = pd.Series(self.data)

        # Split teams into 2 datasets
        training_data , test_data  = [i.to_dict() for i in train_test_split(s, test_size=ratio)]

        # Split datasets into the N groups
        N_datasets = []
        for N in self.NL:
            train_samples = []
            for team in training_data:
                for sample in training_data[team][N]:
                    train_samples.append(training_data[team][N][sample])

            test_samples = []
            for team in test_data:
                for sample in test_data[team][N]:
                    test_samples.append(test_data[team][N][sample])

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


class TeamDataset(Dataset):
    def __init__(self, dataset, max_N):
        self.data = dataset
        self.max_N = max_N

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # self.data has [features,target] at each ID
        data_element = self.data[idx]

        # no padding
        if self.max_N == 0:
            return data_element[0], data_element[1]

        # padding
        if len(data_element[0]) < self.max_N:
            pads = [torch.zeros_like(data_element[0][0]) for i in range(self.max_N - len(data_element[0]))]
            #use pre-padding
            padded_data_element_zero = torch.cat((torch.stack(pads), data_element[0]), dim=0)
        else:
            padded_data_element_zero = data_element[0]
            
         
        return padded_data_element_zero, data_element[1]


def get_team_dataset(file='./Data/team/processed/team_data.xlsx', NL = [5]):

    # GET DESIRED FIELDS
    # https://www.hockey-reference.com/about/glossary.html#:~:text=Doug%20Drinen%20of%20Pro%2DFootball,great%20explanation%20of%20this%20method.&text=Simple%20Rating%20System%3B%20a%20rating,average%2C%20where%20zero%20is%20average.
    # Season 
    # Name
    # Games played : GP
    # Wins : W
    # Losses : L
    # Win Percentage : W%
    # Loss Percentage : L%
    # Points Percentage : PTS%

    # Goals For Per Game : GF/G
    # Goals Against Per Game : GA/G
    # Shots : S
    # Percentage Shots (shots/goal) : S%
    # Shots against : SA
    # Shots saved percentage : SV%

    # Penalities in minutes per game : PIM/G
    # Opponents penalities in minutes per game : oPIM/G
    # Short-handed goal : SH
    # Short-handed goal against : SHA

    # Simple Rating System : SRS
    # Strength of Schedule : SOS
    cols = ['team name', 'Season', 
            'W%', 'L%', # Win/loss related
            'S','GF/G', 'GA/G', 'S%', 'SV%', # Shots/goals related
            'PIM/G', 'oPIM/G', # Penalty related (USE PP, PPA, SH, SHA but normalized to number of games or something. Might also want to use PPO)
            ]
    targets = ['W%','L%', 'S']
    #cols = ['team name', 'Season', 'W%', 'L%', 'S%', 'PIM/G', 'oPIM/G', 'SV%']
    df = pd.read_excel(file, header=0, usecols=cols)


    # Want to predict
    # Wins % : W%
    # Losses % : L%

    # Remove the asterix from the team names
    df = df.replace('\*', '', regex=True)

    dataset = TeamDatasets(df=df, targets=targets, NL=NL)

    return dataset


if __name__=="__main__":


    dataset = get_team_dataset(NL=[3,5,7])

    # Problematic values
    #print(dataset.data['Seattle Kraken'])
    #print(dataset.data['Minnesota North Stars'])

    N_datasets = dataset.random_split(0.1)


    datasets = []
    for element in N_datasets:
        train_dataset = TeamDataset(element[1],N=element[0])
        test_dataset = TeamDataset(element[2],N=element[0])
        datasets.append((train_dataset, test_dataset))

    print(datasets[0][0].__len__())
    print(datasets[0][0].__getitem__(0))