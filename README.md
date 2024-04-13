# Hockey Player Stats Predictor
Deep Neural Network to predict the stats of NHL players for a season N+1 given the stats of the previous N seasons

## Introduction
This projects aims to use deep learning to predict the season stats of individual players given historical data. We make use of transfer learning, representation learning and multi-task learning to evaluate how these techniques can improve the prediction of player stats.

## Model
The model is split into 2 tasks:
* Team Stats Predictor: supervised pretraining is used to get a predictor for the teams stats for season N+1 based on the stats of the N previous seasons. Once trained, the output layer is removed to expose the output of the final layers of the model to get a new data representation.
* Player Stats Predictor: this predictor takes the player stats of the N previous seasons and the new data representation output layer of the Team Stats Predictor as inputs to predict the players stats for the season N+1


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
* Data: folder contains the preprocessed data used. Raw data was manually downloaded from https://www.hockey-reference.com/ but is not provided in this repository.
* datasets: Dataset classes load the preprocessed data and provide loaders for the model
* models: contains the model classes, inlcuding linear baselines and example checkpoints.
* Diagrams: folder containing the diagrams of the model architecture, and additional figures used for a presentation.
* player_standalone_eval.ipynb: notebook to evaluate the player stats predictor
* team_standalone_eval.ipynb: notebook to evaluate the team stats predictor
* player_combined_eval.ipynb: notebook to evaluate the combined model of player and team stats.


