import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class PlayerDataset(Dataset):
    def __init__(self, df, N=5, start_season=1990, stop_season=2023):
        self.N = N
        self.start_season = start_season
        self.stop_season = stop_season
        self.alldata = df
        self.data = self.load_player_data(self.alldata, self.start_season, self.stop_season, self.N)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_element = self.data[idx]
        return data_element[0], data_element[1]

    def set_seasons_per_group(self, N):
        self.N = N
        self.data = self.load_player_data(self.alldata, self.start_season, self.stop_season, self.N)

    def load_player_data(self, df: pd.DataFrame, start_season: int, stop_season: int, N: int):
        players = df['Player'].unique()
        dataset = []

        for player in players:
            player_data = df[df['Player'] == player]
            for season in range(start_season, stop_season - N + 2):
                features = []
                for s in range(season, season + N):
                    season_data = player_data[player_data['Season'] == s]
                    if season_data.empty:
                        break
                    features.append(season_data)

                if len(features) != N:
                    continue

                target = player_data[player_data['Season'] == season + N]
                if target.empty:
                    continue

                dataset.append([features, target])

        return dataset

def get_player_dataset(file='./Data/player/processed/player_data.xlsx'):
    cols = [
        'Rk', 'Player', 'Age', 'Tm', 'Pos', 'GP', 'G', 'A', 'PTS', '+/-', 'PIM',
        'PS', 'EV', 'PP', 'SH', 'GW', 'EV.1', 'PP.1', 'SH.1', 'S', 'S%', 'TOI', 'ATOI', 'Season'
    ]
    df = pd.read_excel(file, header=0, usecols=cols)
    df = df.replace('\*', '', regex=True)
    df['original_index'] = df.index
    df = df.sort_values(by='GP', ascending=False)
    df = df.groupby(['Player', 'Age', 'Season']).filter(lambda x: len(x) == 1 or x['GP'].max() > 47)
    df = df.drop_duplicates(subset=['Player', 'Age', 'Season'], keep='first')
    df = df.sort_values(by='original_index')
    df = df.drop(columns=['original_index'])

    dataset = PlayerDataset(df)
    return dataset

if __name__ == "__main__":
    print("nothing to run")
