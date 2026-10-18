# Advanced Football Analytics: what each metric means, how it is computed, how far to trust it

Match used for every number below: **PFF World Cup 2022, match 3816, Argentina vs Saudi Arabia** (198,142 tracking frames at 29.97 fps, 912 passes, 24 shots, 3 real goals).
Everything is reproduced by one command: `python validate_metrics.py --match 3816` (about 1.5 minutes). Last run: **31 passed, 5 warnings, 0 failed**.

## 0. Pipeline in one paragraph

Tracking positions are gap-filled (cubic spline, gaps up to 1 s) and low-pass filtered (Butterworth, 6 Hz) to get velocity and acceleration. Every metric is then a function of one frame of 22 players + ball on a 105 x 68 m pitch, laid on a 50 x 32 grid. Events (passes, shots...) are aligned to frames. The API (`apps/api`) computes on demand and caches; the web app (`apps/web/src/components/analytics`) only draws.

## 1. The six core metrics

| # | Metric | The question it answers | How it is computed (plain words) |
|---|---|---|---|
| 1 | **Pitch Control (PC)** | Which team would get the ball first if it were played to this spot right now? | Spearman (2018) model. For each grid cell, every player gets a time-to-reach (0.7 s reaction + distance / 5 m/s, with acceleration), turned into a probability, and the two teams' control is integrated over time until it sums to 1. Offside attackers are ignored; a defending goalkeeper controls 3x faster. Same maths as Laurie Shaw's reference code (checked: max difference 0.000). |
| 2 | **DAS** (Dangerous Accessible Space) | Which space can the attackers reach with a pass before the defenders do, and how dangerous is that space? | The `accessible-space` package (Bischofberger), used unchanged. Passes are simulated from the ball in 30 directions and 15 speeds; where an attacker gets there first counts as accessible; every spot is weighted by a small shot-chance model (distance and goal angle). Result: danger-weighted m2. Needs `pip install accessible-space`; if it is missing the engine falls back to an in-house estimate and says so on screen. |
| 3 | **OBSO** (Off-Ball Scoring Opportunity) | How valuable is each attacker's position if the ball reached him? | As in the C-OBSO repository (Fujii, after Spearman 2018): grid = pitch control of the attackers x transition (a fixed pattern centred on the ball, file from the repository) x EPV / max(EPV). A player's OBSO is the grid value **at the cell where he stands**; the team value is the sum of the grid. The ball carrier and offside attackers are left out. The numbers are not probabilities (EPV is scaled by its maximum, as C-OBSO does). |
| 4 | **EPV** (Expected Possession Value) | How likely is the team with the ball to score from this ball position? | Lookup in the published 32 x 50 EPV grid (Fernández & Bornn; file from Laurie Shaw's repository), mirrored for the attacking direction. Depends on ball position only. |
| 5 | **VAEP** (realised) | How much did this action really change the team's chance to score? | EPV after the action minus EPV before. A completed pass is valued at its end point; a lost pass at minus the opponent's EPV there; a shot is 1 if it is a real goal and 0 otherwise. A goal counts only if play restarts with a kick-off (disallowed goals do not). |
| 6 | **OBV** (expected) | What was the action *worth in expectation*, with pass luck removed? | Same states as VAEP, but a pass is valued as p x (value if completed) + (1 - p) x (value if lost), where p is the predicted completion probability of the intended pass. |

**Pass probability p** (used by OBV and Decision score): a logistic model on six features of each pass (distance, metres gained forward, pitch control at the target, chance the lane is open, space around the receiver, space around the passer), fitted on the 862 real passes of the match. Out-of-sample accuracy (5-fold, folds are whole 30-second blocks): **AUC 0.772**, up from 0.68 for the hand-made version; predicted vs observed completion agrees within 0.057 in every fifth of the data.

**"VAEP above average" (`vaep_oa`)**: VAEP minus the average VAEP of the same kind of action from the same third of the pitch in this match. Use it for averages, rankings and totals. Raw VAEP is kept for the single-action view.

## 2. Player analytics (all use the core metrics)

- **Decision score**: for each pass, the expected EPV gain of every teammate option is ranked; 1.0 = he chose the best option, 0.0 = the worst (needs at least 3 options). Also shown: min-max score, regret (EPV given up), and the old ratio.
- **Fatigue**: pitch-control area (radius) the player controls on his own, sampled every 15 s, per 15-minute block, plus VAEP above average per action and distance. Blocks under 3 minutes are not compared.
- **Chemistry matrix**: EPV gained by completed passes between a pair compared with what the same zone normally gives, shrunk towards zero for small samples; at least 3 passes per pair.
- **Decoy Run Index**: runs at >= 5.5 m/s for >= 1 s where the runner does not receive the ball within 4 s, scored by how much the team's DAS grew during the run, with the runner himself removed as a receiver.
- **Press resistance**: an on-ball action counts as *under pressure* when the nearest opponent is closer than 3 m at that moment. We report the success rate under pressure vs free, and the VAEP-above-average gained per 100 pressured actions. (The first version used the opponent's pitch control on the ball handler > 0.65; that flagged only 2 % of the actions the data provider itself labels as pressured. A 3 m nearest-opponent rule reproduces the provider's pressure rate and separates its labels with AUC 0.86.)
- **Fatigue samples only live play**: a sample is taken only when the ball is tracked and a team is in possession, so every sample compares like with like. Distance per block ignores tracking spikes above 10 m/s.

## 2b. Live player statistics (tab "Live player")

Everything a player had done **up to a chosen moment** of the match, with a plain-English summary. Code: `apps/api/player_live.py`; endpoints `/api/live/players`, `/api/live/player/{id}`, `/api/live/player/{id}/timeline`.

**Counted from the event file (exact, not estimated):** passes and completed passes (accuracy = completed / attempted, crosses counted separately), key passes (the receiver shoots within 8 s) and assists (that shot is a goal), progressive passes (completed, at least 10 m closer to goal), passes into the final third and the box, lines broken, shots split into goal / saved / off target / blocked / goal ruled out / other, tackles, aerials, dribbles and 50-50s (won when `challengeWinnerPlayerId` is the player), interceptions (pass outcome "defended" -> `defenderPlayerId`), blocks (pass or shot "blocked" -> `blockerPlayerId`), clearances, goalkeeper saves and goals conceded, cautions.

**On target** = a real goal or a saved shot (this reproduces the official 6 and 2 on target for Argentina and Saudi Arabia in match 3816). A goal counts only when the next restart is a kick-off, so the three Argentine goals ruled out are shown separately and not counted.

**From the tracking:** minutes, distance (integral of speed), high-speed running (5.5-7 m/s), sprints (>= 7 m/s for >= 1 s), top speed, heat map and average position (attacking left to right). Speed is median-filtered (0.3 s); **top speed is the best speed held for one second** (a 1-second moving average), and values above **10 m/s (36 km/h) are tracking jumps** and are ignored (the raw file contains spikes up to 12.4 m/s that last 1-2 s and agree with the position data, so they are errors in the positions, not in the speed calculation). Even so, about 20 players reach 35 km/h for a full second in match 3816, which is higher than published match records: treat top speed as a way to compare players, not as an exact number.

**Model estimates (labelled as such on screen):** xG of each shot (distance-and-angle model, penalties 0.76), expected assists (xG of the shot that followed a key pass), VAEP above average and OBV summed over his valued actions.

**Ranking against team-mates** uses only outfield players (or only goalkeepers) with at least 10 minutes, and only ranks a percentage if there were enough attempts (10 passes, 5 duels).

**Checks run (match 3816, repeat any time with `python verify_live_stats.py --match 3816`, 34 checks):** passes, completed passes, shots by outcome, crosses, clearances, interceptions, blocks, saves, tackles and cautions were counted a second time directly from the raw events file and match the live numbers for all 208 player-statistic pairs checked; completed <= attempted, shots = goal + saved + off + blocked + other + ruled out, minutes <= 115, top speed <= 36 km/h and distance within 0-15 km hold for every player.

## 3. Validation results (match 3816)

| Metric | Test | Result |
|---|---|---|
| PC | Equals the reference implementation (120 cells) | max diff 0.000 |
| PC | Each player controls his own square | 84.5 % (of the rest: 94 contested within 3 m, 22 offside) |
| PC | Rises as the ball moves toward the opponent goal | rho 0.80 |
| PC | Parameter robustness (reaction 0.5-0.9 s, speed 4-7 m/s, no offside, no acceleration...) | largest change 0.017 |
| EPV | Rises toward goal; peak in front of goal | rho 0.98; peak at (51, -1) m |
| DAS (package) | Higher in the 2-4 s before shots than at normal moments | AUC 0.90 (p < 0.001) |
| OBSO (C-OBSO) | Higher in the 2-4 s before shots than at normal moments | AUC 0.76 (p < 0.001) |
| DAS | The drawn surface adds up to the package's own DAS | max relative difference 7e-7 |
| OBSO | Grid equals the C-OBSO formula recomputed with the repository's own indexing | max difference 0 |
| Pass model | Predicts real completions (cross-validated) | AUC 0.772 |
| VAEP | Real goals get a large value | min 0.83 of 3 goals |
| VAEP | Disallowed goals are not rewarded | max -0.10 |
| VAEP | Shot values unbiased (24 shots) | z = 0.24 |
| OBV | Its "if completed" part agrees with VAEP on completed passes | rho 0.92 |
| OBV | Expected equals realised on average | -0.0013 (+- 0.0015) |
| Decision | Players beat a random teammate choice | 0.63, p < 0.001 |
| Decision | Expected EPV gain predicts the realised one | rho 0.26, p < 0.001 |
| Facts | 3 goals found at the right side and minute | 3 / 3 |

## 4. Limits, stated plainly

1. **Single match.** All checks use one match with 24 shots. Parameters of the pass model are fitted on it; how well they carry over to other matches is not yet tested (the validator can be run on any match).
2. **Only DAS comes close to adding something beyond ball position, and it is not proven.** Cross-validated AUC for 'shot in the next 6 s': EPV alone 0.856, EPV + DAS 0.875 (gain +0.018, 95% interval -0.012 to +0.053), EPV + PC 0.799, EPV + OBSO 0.779. With 28 shots the intervals include zero, so none of them can be called better than ball position. They are best described as *spatial context* consistent with shot build-up.
3. **Raw VAEP is slightly negative on average** (-0.005 per action, because losing the ball costs more than a short pass gains on the EPV grid), so raw totals are not "goals". Use `vaep_oa`, and compare players within the match.
4. **VAEP and OBV are approximations built from an external EPV grid**, not the trained models of the published VAEP (Decroos et al.) and OBV (StatsBomb), which need many matches to fit.
5. **OBV is an expected value of the intended pass** and includes the risk of losing the ball. It is *not* StatsBomb's outcome-based OBV. On open-play shots it is slightly negative (the EPV grid values the spot about 26 % above our xG model: 2.59 vs 2.05 expected goals over the 24 shots). Penalties are valued at the usual conversion rate (0.76), not by spot, so Messi's penalty gets OBV +0.59 (VAEP +0.83).
6. **Event-code meanings were inferred from the data** (no PFF documentation was available): shot codes C and L are treated as 'other' (not on target), and in 50-50 duels only the winner is named, so duel totals are exact for winners and slightly low for losers. **Decision score is noisy for one match**: a player's score from odd passes vs even passes correlates 0.39 (23 players). Read it as a tendency, not a precise rating.
7. **What is reference code and what is not.** Pitch Control and EPV (Laurie Shaw), DAS (accessible-space) and OBSO (C-OBSO) are the supervisors' reference implementations (section 6). VAEP and OBV are approximations (limits 4 and 5). The pass model and the shot-chance surface used by OBV and Decision score are our own and hand-calibrated or fitted on this one match.

## 5. Matches, uploads and speed

- Advanced Analytics follows the match opened in Match Replay (files are sent to the local API automatically, `POST /api/matches/upload`; they are saved in `data/`, which git ignores). Nothing is loaded or computed when the server starts.
- Measured on 3816: server memory at start 148 MB; reading a cached match 0.14 s; one frame (PC + EPV + DAS + OBSO) 63 ms (the DAS package is the largest part: about 35-60 ms of it, most of that is the package's own table handling, not the pass simulation); all player analytics 19 s once per match, then cached on disk.
- **The ball is tracked only while it is in play.** In match 3816 the ball position exists in 47 % of the frames (`/api/metrics/live-segments`: about 26 min of live play in each half). On the other frames (dead ball, set-ups, stoppages) DAS, OBSO and EPV cannot be computed, because they need the ball; the screen shows "Dead ball" and a dash instead of inventing a value. Pitch Control and the back line need only player positions and are still shown. "Possession" is the team of the last on-ball event, so it is shown only while the ball is tracked.
- **Playback.** The screen asks for one frame every 0.4 s of wall-clock time (measured server time 60-130 ms per frame), fetches two frames ahead, keeps the last 160 frames in memory, and slides the players between two frames, so 1x real time looks continuous. Above 2x players jump instead of sliding. "Computing..." appears only if a frame takes longer than 0.35 s. "Skip dead ball" jumps over the stretches without ball.
- Needs `pip install -r apps/api/requirements.txt` (includes `python-multipart` for uploads and `accessible-space` for DAS). `/api/health` shows which DAS is in use.

## 6. Reference repos: what is used and what was measured

| Metric | Reference repo | In the app |
|---|---|---|
| PC | LaurieOnTracking | Same maths (max difference 0.000) |
| EPV | LaurieOnTracking (`EPV_grid.csv`) | Same grid, same lookup |
| OBSO | C-OBSO | **The C-OBSO formula**; `validate_metrics.py` recomputes it with the repository's own indexing (max difference 0) |
| DAS | accessible-space | **The package itself**; the drawn surface adds up to its number (max relative difference 7e-7) |
| VAEP, OBV | papers only, no repository | EPV-grid approximations (limits 4 and 5) |

Why the swap: before it, DAS and OBSO were in-house versions. Measured once on 154 regular frames (one every 20 s of play) and 47 frames 2-4 s before shots (`python check_references.py` rebuilds the old formulas for this comparison):

| Question | Result |
|---|---|
| Old in-house DAS vs the package's DAS: same ordering of moments? | rho 0.83, but a different scale (median 4.7 vs 0.2) |
| Old OBSO team total vs C-OBSO team total | rho 0.82, different scale |
| Per-player OBSO ranking inside a frame, old vs C-OBSO | rho -0.23: a different definition (old: all space he controls; C-OBSO: value at his own spot) |
| Rises before shots (AUC): package DAS / old DAS | 0.80 / 0.75 on that sample (0.90 on the validator's sample) |
| Rises before shots (AUC): C-OBSO / old OBSO | 0.70 / 0.74 (0.76 for C-OBSO on the validator's sample) |
| Package accessible space without the danger weight | AUC 0.53: the danger weighting carries the signal |

Reading: the old versions ranked dangerous moments similarly but were not the same quantity. C-OBSO is not a better shot predictor than the old OBSO on this match; it is used because it is the published definition the supervisors named.

**Decoy Run Index** uses the package DAS: the change in the team's DAS between the start and the end of a run, with the runner himself removed as a receiver (the defenders stay where they were).

## 7. References

- Spearman, W. (2018). *Beyond Expected Goals*. MIT Sloan Sports Analytics Conference. (pitch control, OBSO)
- Fernández, J., Bornn, L. (2018). *Wide Open Spaces: A statistical technique for measuring space creation in professional soccer*. (space value)
- Fernández, J., Bornn, L., Cervone, D. (2021). *A framework for the fine-grained evaluation of the instantaneous expected value of soccer possessions*. Machine Learning 110. (EPV)
- Decroos, T., Bransen, L., Van Haaren, J., Davis, J. (2019). *Actions Speak Louder than Goals: Valuing Player Actions in Soccer*. KDD. https://arxiv.org/abs/1802.07127 (VAEP)
- Hudl StatsBomb. *Introducing On-Ball Value (OBV)*. https://www.hudl.com/blog/introducing-on-ball-value-obv
- Bischofberger, J. *accessible-space* (DAS). https://github.com/jonas-bischofberger/accessible-space
- Fujii, K. *C-OBSO* (OBSO). https://github.com/keisuke198619/C-OBSO
- Shaw, L. *LaurieOnTracking* (reference pitch control code and EPV grid). https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking
