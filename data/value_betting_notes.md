# Value Betting — Methodology Notes

A bet has positive expected value (EV) when the model's estimated probability of an outcome is
higher than the probability implied by the offered odds.

## Edge and EV
- Implied probability = 1 / decimal odds (before removing the bookmaker margin).
- Edge = model probability minus implied probability.
- Expected value per $100 = (model probability * profit) - ((1 - model probability) * stake).

## Bankroll discipline
Kelly fraction sizing is recommended but capped (e.g. half-Kelly) to reduce variance.
Closing line value (CLV) — beating the final market price — is the best leading indicator that
an edge is real rather than luck.
