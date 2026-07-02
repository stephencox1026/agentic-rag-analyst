# NBA Rebound Model — Notes

The rebound prediction model uses gradient-boosted trees (XGBoost performed best) trained on
roughly five seasons of player-game data.

## Most important features
1. Last-10-game rebound average — the single strongest predictor.
2. Minutes projection — rebounds scale almost linearly with minutes played.
3. Team and opponent pace — more possessions create more rebound opportunities.
4. Opponent field-goal percentage — missed shots are rebound chances.
5. Height/matchup advantage versus the projected primary defender.

## Evaluation
Mean absolute error lands around 1.5 to 2.0 rebounds, with an R-squared near 0.4 to 0.6 on
held-out games. Predictions are converted to win probabilities and compared to the betting line;
only opportunities with a 15 percent or greater edge are flagged as high value.
