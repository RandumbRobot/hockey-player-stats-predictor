import pandas as pd
import torch
from torch.utils.data import Dataset



class TeamDataset(Dataset):
    def __init__(self, df, N=5, start_season=1990, stop_season=2023):
        """
        Teams dataset

        :param file: path to preprocessed teams dataset
        :param N: number of consecutive seasons to load per group. Note that the associated label will be the season N+1
        :return: dataset
        """
        self.N=N
        self.start_season=start_season
        self.stop_season=stop_season

        # Save the default full data
        self.alldata = df

        # Get all the possible groups
        self.data = self.load_team_data(self.alldata, self.start_season, self.stop_season, self.N)


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # self.data has [features,target] at each ID
        data_element = self.data[idx]
        return data_element[0], data_element[1]
    
    def set_seasons_per_group(self, N):
        """
        Set the number of consecutive seasons (N) to load per group.
        This function also alters self.data by converting the original data to the new N

        :param N: number of consecutive seasons to load per group. Note that the associated label will be the season N+1
        """
        self.N = N

        # Get all the possible groups
        self.data = self.load_team_data(self.alldata, self.start_season, self.stop_season, self.N)


    def load_team_data(self, df: pd.DataFrame, start_season: int, stop_season: int, N: int):
        """
        load_team_data: loads a dictionary of team data by grouping multiple groups of 
        N seasons together for a range of seasons. 
        NOTE: only teams that appear for all seasons for a given group will be added

        :param data: NHL team dataset for all years
        :param start_seasons: First season to consider (inclusive).
        :param end_seasons: Last season to consider (inclusive).
        :param N: Number of seasons per group
        :return: returns a data structure with the following format (for num_seasons=N=5)
            V2:
            NOTE: a group of seasons for a given team is considered a sample
                Group 1994: 1990 to 1994
                    Team1:
                        feature: array of stats for N seasons (1990 to 1990+N-1)
                        label: stats for season N+1
                    Team2:
                        feature: array of stats for N seasons (1990 to 1990+N-1)
                        label: stats for season N+1
                    ...

                Group 1995: 1991 to 1995
                    Team1
                        feature: array of stats for N seasons (1991 to 1991+N-1)
                        label: stats for season N+1
                    ...
            Then simply concatenate all the teams
        """ 


        # Get all the teams names (10 seconds if full dataset)
        team_names = list(set(df.loc[:, 'team name']))

        # Create dictionary with each team as key and an array of all their stats as an entry
        teams = {key: pd.DataFrame([row for idx,row in df.iterrows() if row['team name']==key]) for key in team_names}

        # Create dataset 
        # A data element has features (array of N seasons of a team, where each row is a season) and target (stats for season N+1)
        dataset = []
        dataset_with_names = []

        # For each team
        for team in teams:
            df = teams[team]

            # Get groups of N seasons
            for season in range(start_season, stop_season-N+2, 1):

                # For the current group of seasons
                features = []
                features_with_names = []
                for s in range(season, season+N):
                    ss = df.loc[df['Season']==s]
                    if ss.empty: # abort if inexistant season
                        break
                
                    features.append(torch.tensor(ss.drop(['team name', 'Season'], axis=1).values, dtype=torch.float32))
                    #features.append(ss)
                    features_with_names.append(ss)
                
                # Skip if not enough consecutive seasons (inexistant season)
                if not len(features) == N:
                    # TODO: season+=N for faster iterations?
                    continue

                # Add target
                target = df.loc[df['Season']==season+N]
                if target.empty: # abort if inexistant season
                    continue

                # Add data to dataset
                dataset.append([torch.stack(features).reshape((len(features),len(features[0][0]))), 
                                torch.tensor(target.drop(['team name', 'Season'], axis=1).values, dtype=torch.float32)[0]])
                #dataset.append([features, target])
                dataset_with_names.append([features_with_names, target])

        self.dataset_with_names = dataset_with_names
        return dataset
        

def get_team_dataset(file='./Data/team/processed/team_data.xlsx'):

    # GET DESIRED FIELDS
    # Season 
    # Name
    # Points Percentage : PTS%
    # Win Percentage : Win%
    # Goals For Per Game : GF/G
    # Goals Against Per Game : GA/G
    cols = ['team name', 'Season', 'PTS%', 'GF/G', 'GA/G']
    df = pd.read_excel(file, header=0, usecols=cols)

    # Remove the asterix from the team names
    df = df.replace('\*', '', regex=True)

    dataset = TeamDataset(df)

    return dataset


if __name__=="__main__":
    print("nothing to run")