# Member 5 results - 63 matches, 5-fold cross-validation grouped by match

Every number is measured on matches the model did not see during training. Match 10517 (the final) was kept out of training; its separate test score is still to be run (`python -m pff5.holdout`).
`raw` = honest raw ball. `smoothed` = PFF's event-informed ball (leaks label information, see README).
"Baseline" is the simplest sensible rule (most common class, or nearest player).

## 1. Pressure

155802 labelled moments (initial touches + events).

| Metric | raw ball | smoothed ball | Baseline |
|---|---|---|---|
| Pressure type accuracy (N / A / L / P) | 75.0% | 74.6% | 66.9% |
| Pressure type macro-F1 | 48.6% | 48.1% | - |
| - on initial touch: accuracy | 79.7% | 79.7% | 78.3% |
| - on the event itself: accuracy | 70.1% | 69.3% | 55.1% |
| Pressured vs not: best F1 | 72.8% | 72.6% | - |
| Pressured vs not: AUC | 88.9% | 88.8% | - |
| Pressing player correct (top-1) | 80.1% | 75.9% | 73.5% |
| Pressing player in top-2 | 91.8% | 88.9% | 86.8% |
| Type AND player both correct (pressured events) | 46.5% | 44.0% | - |

Per-class F1 (raw ball): N 0.85 (n=104288), A 0.17 (n=5271), L 0.28 (n=6399), P 0.64 (n=39844)

## 2. Finding challenges from tracking alone

11129 challenges. A detection counts if it is within the tolerance of a real challenge.

| Metric | raw ball | smoothed ball | Baseline |
|---|---|---|---|
| F1 within 1s | 49.0% | - | 31.7% |
| - precision / recall within 1s | 43.2% / 56.6% | - / - | - |
| Average precision within 1s | 45.8% | - | 23.2% |
| F1 within 2s | 54.7% | - | 39.2% |
| - precision / recall within 2s | 48.8% / 62.3% | - / - | - |
| Average precision within 2s | 53.9% | - | 32.7% |

## 3. Challenge details - tracking only

Scored on real challenge moments. "End to end" rows use the model's own choice of the two players.

| Metric | raw ball | smoothed ball | Baseline |
|---|---|---|---|
| Involved player correct (per team, top-1) | 71.6% | 77.0% | 67.7% |
| Both involved players correct | 56.4% | 60.6% | - |
| Challenge type (8 classes) | 52.5% | - | 24.6% |
| Challenge type macro-F1 | 34.6% | - | - |
| Tackle/dribble vs duel structure | 84.5% | - | - |
| Which side has the ball | 84.3% | - | 50.4% |
| Which side wins | 69.3% | - | 50.4% |
| Outcome (13 classes) | 58.7% | - | 28.5% |
| Foul given (accuracy) | 89.7% | - | 87.1% |
| Foul given (best F1) | 56.3% | - | - |
| Culprit side, given a foul | 78.1% | - | 50.3% |
| Card, given a foul (AUC) | 68.8% | - | 50.0% |
| Dribble type (dribbles only) | 43.3% | - | 33.2% |
| Tackle attempt type (dribbles only) | 73.0% | - | 72.8% |
| Trick type | 53.4% | - | 41.4% |

Challenge type per-class F1 (raw ball): A 0.89 (n=2740), B 0.48 (n=2437), D 0.46 (n=1877), FIFTY 0.42 (n=1378), H 0.12 (n=371), L 0.02 (n=324), S 0.10 (n=488), T 0.27 (n=1507)

## 3. Challenge details - player in possession known (from Member 1)

Scored on real challenge moments. "End to end" rows use the model's own choice of the two players.

| Metric | raw ball | smoothed ball | Baseline |
|---|---|---|---|
| Involved player correct (per team, top-1) | 85.1% | 81.7% | 67.7% |
| Both involved players correct | 71.0% | 64.8% | - |
| Challenge type (8 classes) | 54.2% | - | 24.6% |
| Challenge type macro-F1 | 38.7% | - | - |
| Tackle/dribble vs duel structure | 84.9% | - | - |
| Which side has the ball | 96.2% | - | 50.4% |
| Which side wins | 79.5% | - | 50.4% |
| Outcome (13 classes) | 61.1% | - | 28.5% |
| Foul given (accuracy) | 90.0% | - | 87.1% |
| Foul given (best F1) | 58.4% | - | - |
| Culprit side, given a foul | 89.2% | - | 50.3% |
| Card, given a foul (AUC) | 69.2% | - | 50.0% |
| Dribble type (dribbles only) | 43.1% | - | 33.2% |
| Tackle attempt type (dribbles only) | 73.1% | - | 72.8% |
| Trick type | 55.7% | - | 41.4% |

Challenge type per-class F1 (raw ball): A 0.89 (n=2740), B 0.50 (n=2437), D 0.50 (n=1877), FIFTY 0.43 (n=1378), H 0.37 (n=371), L 0.05 (n=324), S 0.09 (n=488), T 0.28 (n=1507)

## 4. Tracking only -> complete challenge events

Spot the challenge, then name both players and the type, all with models that never saw the match.

| Metric | raw ball | smoothed ball |
|---|---|---|
| Challenges found within 1 s (recall) | 56.6% | - |
| Detections that are real challenges (precision) | 43.2% | - |
| On found challenges: both players correct | 58.7% | - |
| On found challenges: at least one player correct | 87.3% | - |
| On found challenges: type correct | 57.3% | - |
| On found challenges: winning side correct | 68.9% | - |
| Share of all real challenges fully right (time + both players + type) | 18.7% | - |

## 5. Fouls and offside on events

87244 possession events, 2076 with an infringement.

| Metric | raw ball | smoothed ball | Baseline |
|---|---|---|---|
| Foul given, all events: best F1 | 52.5% | - | - |
| Foul given, all events: AUC | 96.0% | - | 50.0% |
| Foul given, challenges only: best F1 | 57.2% | - | - |
| Foul given, other events: best F1 | 26.2% | - | - |
| Offside: best F1 | 23.8% | - | - |
| Offside: AUC | 92.6% | - | 50.0% |
| Given / advantage / missed (accuracy) | 89.8% | - | 86.3% |
| Offense type (accuracy) | 91.5% | - | 79.3% |
| Card / outcome (accuracy) | 83.6% | - | 83.2% |
| Culprit player (top-1) | 61.1% | - | 10.9% |
| Victim player (top-1) | 86.9% | - | 84.2% |
| VAR used (AUC) | 82.0% | - | 50.0% |

