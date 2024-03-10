# Hockey Player Stats Predictor
Deep Neural Network to predict the stats of NHL players for a season N+1 given the stats of the previous N seasons

## Introduction
TODO

## Model
The model is split into 2 tasks:
* Team Stats Predictor: supervised pretraining is used to get a predictor for the teams stats for season N+1 based on the stats of the N previous seasons. Once trained, the output layer is removed to expose the output of the final layers of the model to get a new data representation.
* Player Stats Predictor: this predictor takes the player stats of the N previous seasons and the new data representation output layer of the Team Stats Predictor as inputs to predict the players stats for the season N+1

### Team Stats Predictor
TODO: arch

### Player Stats Predictor
TODO: arch

## Model Characteristics
### Training Data
* The model was trained on seasons 1990 to 2023

### Limitations
The model has the following limitations:
* The players must have played at least N seasons in the NHL with over 75% of the games played for that season
* The players must have stayed with the same team for the N seasons played
* The goalies are not taken into account, only attackers and defenders
* The model input is fixed to N seasons.

## Using the model for a different league
The model was built with the NHL players and teams in mind. However, it can still be adapted to other leagues using domain adaptation techniques.

## Project Structure
The structure is split as follow:
* web_scrapper.ipynb: scripts performs web scrapping to acquire the stats of players for desired seasons. The data is stored in a */data* folder which is ignored by the repository
* dataloader.ipynb: class that performs the preprocessing of the acquired data and provides the data loading functions. The preprocessed data is cached in the */data* folder which is ignored by the repository
* team_stats_predictor.ipynb: model for the supervised pretraining task to predict the team's stats
* player_stats_predictor.ipynb: model that performs the player stats prediction
* regression.ipnyb: alternative model used as a reference for performance comparison
* results.ipynb: script that performs the training and testing of the models. It also provides graphs for the results.
* /models: folder that contains the weights and biases of the trained models. 
