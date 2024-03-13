# V1 FEEDBACK JACOBY

My initial thoughts:

It looks really good, I am a big fan of all the separate tasks that all come nicely together.


Team-only Stats:
-The LSTM is indeed very useful to capture the time dependency of seasons. A team might be getting worse or better from one season to the next. This would definitely help capture the momentum.


Player-Based Team Stats:
-The idea is good, but might not be flexible enough. We have to be careful with the players chosen because some teams may have changes in their lineup during the same year, which will complicate things A LOT
-However, if we can choose an appropriate number of constant players for each season (8 to 14 would be enough) and allow for a variable number of players into the aggregator (between 8 and 14), it would solve the problem. I assume the aggregator is a simple mathematical operation that is not updated by any backprop. 
-The weight by time on ice makes a lot of sense IMO, because it would allow a player that was traded to still be used. BUT it would require significantly more preprocessing.


Player Stats Predictor:
-Same feedback than for the team-only stats applies.
-I think it lacks the context of the team for all the seasons. I would also give the team stats as an input for each season.

Final Model:
-Looks good to me. I like the idea of "correcting" the prediction of the first player predictor with the predicted performance of the future team of the player.


PROPOSED CHANGES (see V2 in draw.io):
-I really like the Team-as-a-sum-of-players approach, but I don't know if we realisticly have the time to implement this.
-I think we should go with the Team-as-an-entity approach to get a latent representation. I would also make a second model that would only guess the next season stats based on the previous season (naïve) and use its latent representation as an input to the player single season model (see next).
-Similarly, I would modify the Player Stats Predictor to include the latent representation of the team and only attempt to guess the next season from the previous one. The idea is to give context to the player's performance with the team's performance for that season (could be different team at every season). This also allow a building block which can be reused for an arbitray number of seasons.
-In the final model, I would combine the latent representation of the single season player model with the LSTM team-entity latent representation to make a final guess of the next season player's stats.
-The weights of the models trained for their latent representation are freezed when given as inputs.
-NOTE: the single season models have access to way more data than any N to N+1 predictors since they can train on any player that played for two consecutive seasons.

Total number of models:
-One for team-entity single season (naïve, easy to train)
-One for player single season (naïve, easy to train)
-One for team-entity LSTM (complex)
-One final model for player LSTM (complex)

However, my approach might be overcomplicated, but it does allow more flexibility if we want to change N since the "naïve" models can be reused.
Otherwise, we could use the model you made and incorporate the team's stats for every season of the player when learning latent representation.
Total number of models:
-One for player+team stats to player stats LSTM (complex)
-One for team-entity LSTM (complex)
-One final model for player (complex)