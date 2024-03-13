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


PROPOSED CHANGES:
-I really like the Team-as-a-sum-of-players approach, but I don't know if we realisticly have the time to implement this.
-I think we should go with the Team-as-an-entity approach to start to get a latent representation.

Otherwise, we could use the model you made and incorporate the team's stats for every season of the player when learning latent representation.

Total number of models:
-One for player+team stats to player stats LSTM
-One for team-entity LSTM
-One final model for player