# Member 4: shot (SH) events from tracking data, version 2

Mohamed Hatem, 11 Nov 2026. Data: PFF FC World Cup 2022, all 64 matches. Code: `apps/api/event_detection/` (branch `feature/event-detection-m4`).

## 1. Status

All of Member 4's part is implemented: the shot event, the 14 fields inside it that tracking can support, and the shared blocker attribution on passes, crosses and clearances. Version 2 improved the fields clearly (outcome, keeper, blockers). Shot detection itself did not improve: three different attempts all ended at F1 0.65, which is the limit of this broadcast tracking. Not implemented: `deflectorPlayerId` and inadvertent shots (see section 6).

Data split: 48 training matches (1,155 PFF shots), 16 unseen matches (404 shots). "Training CV" means each training match is predicted by models that never saw it (6 folds of whole matches).

## 2. Results

### Shot detection

| | Training CV (48) | Unseen (16) |
|---|---|---|
| Precision | 70% | 69% |
| Recall | 61% | 60% |
| F1 | 0.65 | 0.64 |

- Per unseen match, F1 runs from 0.36 to 0.87 (median 0.65).
- 58 shots (43 training, 15 unseen) are penalty shoot-outs after the tracking video ends. Without them, training-CV recall is 63% and F1 0.67.
- If a detection within 1 s (not 0.5 s) of PFF's time counts as correct, training-CV F1 is 0.68.
- The threshold can trade precision for recall (training CV): 0.15 gives P 62% / R 67%; 0.30 gives P 77% / R 56%; 0.50 gives P 88% / R 41%.

### Fields, on the shots we detected (end to end)

| Field | v1 training CV | v2 training CV | v2 unseen | Delivery |
|---|---|---|---|---|
| Shooter | 78% | 79% | 70% | 1 |
| Outcome, 7 PFF types | 61% | 66% | 66% | 1 |
| Outcome, 4 groups | 65% | 68% | 68% | 1 |
| Keeper | 63% | 68% | 66% | 1 |
| Ball moving | 97% | 99% | 99% | 2 |
| Shot nature | 69% | 71% | 68% | 2 |
| Shooter movement | 66% | 67% | 64% | 2 |
| Shot height | 50% | 49% | 49% | 2 |
| Save height | 28% | 28% | 33% | 2 |
| Save rebound zone | 9% | 14% | 12% | 2 |
| Failed blocker | 30% | 43% | 36% | 3 |
| Blocker | 13% | 20% | 31% | 3 |
| Goal-line clearer | 0% | 0% | 0% | 3 |

Goals are now recognised much better: 74% of real goals are labelled G in training CV (45% in v1).

The unseen matches were tested twice (v1, then v2). Every v2 choice was made on training CV only, but the v2 unseen numbers are a second look and should be read with that in mind.

### Fields against guessing (at PFF's shot moments, training CV)

| Field | Most-common-value guess | Model |
|---|---|---|
| Shot nature | 49% | 71% |
| Shot height | 39% | 48% |
| Save rebound zone | 17% | 33% |
| Ball moving | 93% | 96% |
| Shooter movement | 69% | 69% (no skill) |
| Save height | 45% | 45% (no skill) |

### Shared part: who blocked a pass, cross or clearance

Given the event moment and the team on the ball (from Members 2 and 3; here taken from PFF), training CV:

| Event | Blocked events | Nearest-player rule | Ranking model |
|---|---|---|---|
| Pass | 1,043 | 65% | 69% |
| Cross | 208 | 74% | 76% |
| Clearance | 104 | 45% | 48% |

## 3. What changed in version 2

| Change | Idea | Effect |
|---|---|---|
| Kick-off / restart evidence | After a goal, players return to their halves and play stays dead longer. Measured as continuous quantities (share of players in own half, dead time, ball tracked share) instead of one yes/no rule | Outcome 4 groups 66% to 70% at PFF moments; on detected shots, goal recall 45% to 74% |
| Outcome predicted as 7 types directly | The hand-written geometry rules for S/F and B/C were worse than letting the model learn all 7 types | 7-type accuracy 57% to 66% at PFF moments |
| Learned player ranking | Every player is a candidate with distance-to-ball and ball-path features; a classifier picks the best | Blocker 42% to 51%, failed blocker 56% to 65% at PFF moments; shooter 70% to 72% |
| Trajectory context | Ball and nearest-player samples from 0.5 s before to 1.2 s after each kick, following sequence-model papers | Detection average precision 0.626 to 0.639 |
| Sequence stacking (not kept) | Second model that sees the scores of neighbouring kicks | F1 +0.004, not worth the complexity |
| Bigger models, 9 reaction features | | Tiny gains |

## 4. Why detection stops at F1 0.65

Evidence that this is a data limit: 8 times more training data, a deeper model, context features and sequence stacking each moved F1 by less than 0.01.

1. Headers: 39% found (foot shots 63%). The ball is in the air and its position is noisy.
2. Blocked shots: 42% found. The ball hits a defender after 1 to 2 m and never travels toward goal.
3. The shooter is often not at the ball: when our shooter is wrong, the true shooter is on average 10.8 m from the tracked ball and only half of those are marked visible.
4. PFF labels intention ("deliberate attempt to score"); crosses and clearances toward goal move the same way as shots.
5. Ball missing in 22 to 38% of frames; penalty shoot-outs not tracked at all.

## 5. Checks against cheating and overfitting

- PFF's event fields and PFF's smoothed positions are deleted in the reader; `test_leakage.py` passes for all feature modules.
- All splits are by whole match. Thresholds are chosen on training folds only.
- Training-CV and unseen numbers agree for detection and outcome. The shooter drops from 79% to 70% on unseen matches.

## 6. Not done, and why

- `deflectorPlayerId`: PFF defines it as a light touch with only a slight redirection. Such touches are below what this tracking can show; not attempted.
- Inadvertent shots: about 5 per 12 matches; they depend on other members' events.
- Goal-line clearer: 8 examples in 48 matches; a rule exists but scores 0%.
- `shotType`, `keeperTouchType`, `badParry`, `saveable`, grades: need video or human judgement.
- Shooter movement and save height: implemented, but no better than guessing.

## 7. How to run (from `apps/api`)

```
pip install -r event_detection/requirements.txt
python -m event_detection.test_leakage --match 3830
python -m event_detection.cv48 report          (training CV numbers)
python -m event_detection.final48 train        (final model on 48 matches)
```

The final model is `data/interim/m4/member4_models_v2.pkl`. Predicted shots for the unseen matches are `data/interim/m4/pred_shots_<id>.csv` with PFF column names.

## 8. Sources

- Vidal-Codina et al. 2022, Automatic event detection in football using tracking data: https://arxiv.org/abs/2202.00804
- Kim et al. 2025, ELASTIC: https://arxiv.org/abs/2508.09238
- FOOTPASS / TAAD+DST (sequence context for action spotting): https://arxiv.org/pdf/2505.09455v1
- PFF FC Event Data Specification v2.5
