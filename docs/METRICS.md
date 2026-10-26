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

**Model estimates (labelled as such on screen):** xG of each shot (xG model v2, which also sees the goalkeeper and the defenders in the way, see 3c; penalties 0.79), expected assists (xG of the shot that followed a key pass), VAEP above average and OBV summed over his valued actions.

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

## 3b. Accuracy update v2 (what changed, what was tested)

Changes (all small, all tested; nothing was tuned on one match):

| Change | Why | Effect on match 3816 |
|---|---|---|
| EPV is interpolated between the four nearest grid cells (it used the one cell containing the ball) | the grid is ~2 m wide, so the value jumped at cell borders and short passes showed false gains | biggest jump when the ball moves 0.5 m: 0.16 -> 0.04; values at cell centres unchanged |
| Penalty value 0.76 -> 0.79 | published event-data models put penalty conversion at about 0.78-0.79 | tiny: shot values of penalties only |
| Fatigue tab counts live play only (ball in play) for distance, speed and the "enough minutes" rule | the pitch-control samples were live-only, distance and minutes were not | blocks are now comparable; new field `distance_per_live_min` |
| NEW: continuous **pressure score** 0..1 (an opponent 5 m or more away = 0, on top of him = 1, several defenders add up) with success rate for low / medium / high pressure, a 95 % interval and the "pressured" count for 2 / 3 / 4 m | a fixed 3 m cut-off throws information away and hides how sensitive the count is to the radius | success: 81 % free vs 69 % under high pressure (z = 3.5); AUC 0.69 against the provider's own pressure label |
| NEW: `validate_accuracy.py` (see below) | one command that checks the numbers against rules and against what really happened, for ANY match | 25 PASS / 3 WARN |

**Gaps in the metrics (no ball = no metric).** The provider tracks the smoothed ball only while it is in play. On 3816 the ball is tracked 47.8 % of the 110 minutes (about 52.7 min); the rest are 132 stoppages (41 of them longer than 30 s: injuries, VAR, goals; 48 of 10-30 s: throw-ins, free kicks, goal kicks). EPV, DAS and OBSO need the ball, so they are blank there on purpose. Loader v1.3 only repairs what is certainly play: (1) the 84 events whose frame had no ball now take the position of the player who played them (player and ball are 0.00 m apart at the event frame), so 75 more passes and actions get a VAEP/OBV value; (2) holes up to 10 s in the ball track that contain such events are bridged by a straight line through those points (about 1 minute in total). Longer holes stay empty.

**Smooth drawing.** The model grids are 50 x 32 cells of about 2 m. The pitch view now stretches each layer with smooth scaling instead of painting flat squares, DAS spreads every sample of the package over neighbouring cells (the sum is unchanged), and the colour fade has no hard cut-off. The numbers did not change, only the picture.

**Possession after a duel.** A challenge event carries the challenger's team, but the ball goes to whoever won it (in 27 % of the duels of 3816 the winner is the other team). The "team in possession" used by Pitch Control, DAS and OBSO now follows the winner of a duel (201 of 201 duels checked, about 2 % of live frames changed side). Analytics version 1.9.

Tested and deliberately NOT changed: pitch-control parameters (reaction time, speed, sigma, lambda), acceleration use, offside rule, goalkeeper factor, passer exclusion. A pass-outcome test on 862 passes gave AUC 0.72-0.73 for every variant and the differences (0.00-0.01) are far inside the noise (+-0.04 with one match), so the published values stay. Tuning them on a single match would only fit that match. With the 64 matches on Drive this should be repeated with whole matches held out.

Run: `python validate_accuracy.py --match 3816` (add more ids to test other matches, `--quick` to skip the slow checks). Results are saved to `data/interim/accuracy_<match>.json`. The pass model was fitted on 3816, so the pass-model checks on 3816 are in-sample (optimistic); on any other match they are a genuine out-of-sample test.

The three WARNs on 3816 and what they mean:
1. **Pitch control as a probability** (Brier skill +0.02, interval includes 0): it ranks passes well (AUC 0.72, interval 0.68-0.76) but is not a calibrated completion probability. Use it as "who gets there first"; use the pass model for probabilities.
2. **Raw VAEP averages -0.005 per action**: losing the ball costs more than a short pass gains. This is why the app shows VAEP *above average* (mean 0).
3. **Distance totals look low**: outfield players cover a median 8.8 km in 110 tracked minutes from PFF's smoothed positions, against 10.2-11.7 km from raw positions with light smoothing (raw positions are too spiky for sprint figures: acceleration spikes above 35 m/s2 even after 0.75 Hz filtering). Sprint metres are the most affected (about 100 m smoothed vs 400+ m raw-based). The truth is not known without a reference; treat distance, high-speed and sprint totals as a lower bound and use them to compare players, halves and matches. Check against the wearable GPS before quoting kilometres.

## 3c. Top attacks (tab "Top attacks", `apps/api/attacks.py`, version 2.0)

**Attack** = a passage of play by one team (Opta's "sequence"): it starts when the team wins the ball or restarts play and ends when the opponent controls the ball, the ball goes dead (out, foul, substitution), nothing happens for 12 s, or the ball is untracked for more than 3 s. A defender's block, clearance or parry followed within 3 s by the same team keeps the attack alive (rebounds), but never across a dead ball. Possession after a duel goes to its winner. Only attacks that reached the final third or ended in a shot are listed (141 on match 3816).

**Danger** = max(attack xG, peak xT), a chance of a goal:
- **attack xG** = 1 - prod(1 - xG of each shot). Each shot uses **xG model v2** (`assets/xg_model_v2.json`, `MetricsEngine.shot_xg`): a logistic regression fitted on the public StatsBomb open data shots with freeze frames (World Cup 2022 held out), with distance, log distance, goal angle, header, direct free kick, defenders in the shooting triangle (ball-post-post), goalkeeper distance from goal, goalkeeper distance from the shooting line and opponents within 3 m. All inputs are read from the PFF tracking frame of the shot, so the model sees the keeper and the blockers. Penalties 0.79; goals ruled out (offside / VAR) count 0. On the 1,418 held-out World Cup 2022 shots: AUC 0.815, log loss 0.271 (old distance-and-angle model 0.768 / 0.296; StatsBomb's own xG 0.820 / 0.263). No PFF data was used for fitting.
- **peak xT** = 0.852 x Shaw's EPV at the most dangerous spot the team CONTROLLED (where its players played the ball or where its completed passes / carries arrived), not counting the shot spot itself (the shot is already measured by xG). The factor 0.852 puts EPV on the same ruler as xG: over 10,708 public shots, mean model xG / mean EPV at the shot spot = 0.093 / 0.109. Without it the EPV grid (which is higher than xG at most shooting spots) decided almost every attack.

**Danger score 1-100** = the probability on a fixed logarithmic scale, 100 * ln(p / 0.004) / ln(0.8 / 0.004): 2 % -> 30, 5 % -> 48, 10 % -> 61, 20 % -> 74, 40 % -> 87, penalty (79 %) -> 100. Bands: 75+ very dangerous, 55-74 dangerous, 35-54 promising, below 35 low threat (on match 3816: 3, 22, 49, 67 attacks). The scale is fixed, so scores compare across matches.

**Tags** (shown next to the play pattern): Big chance = a shot with xG >= 0.38 or a penalty; Fast break = from the own half into the final third within 12 s; High turnover = ball won from the opponent within 40 m of their goal; Direct attack = starts in the own half, directness (progress / ball path) >= 0.5 and ends in a shot or the box; Build-up = 10+ passes and ends in a shot or the box. The pattern Counter-attack follows StatsBomb's definition: regular play, ball won from the opponent outside the final third, directness >= 0.75 and at least 16.5 m of progress.

**Tab**: ranking "Goals first" (goals ordered by chance quality, then the rest by danger), "Pure danger" or "Best chances" (attacks with a shot, by attack xG); whole match or per team; set pieces can be hidden. Each attack shows a "Why X/100" panel (shot distance, header or foot, defenders in the way, keeper distance; or the best spot, its EPV and how much of it the team controlled), a "Key moment" button (jumps to the best moment and shows pitch control), xT gained, passes, direct speed, defenders goal-side and dangerous space (DAS) at the key moment.

**Tested and not adopted**: weighting the best spot by pitch control (Shaw's "EEPV" = EPV x control). It predicted less well whether an attack would end in a shot (AUC 0.70 vs 0.73 without), so control is shown in the panel but not used in the score. Combining xG and xT as a union 1-(1-xG)(1-xT) instead of the larger would count the same chance twice; it changes 3 of the top 10.

**Checks** (validate_accuracy section 7 and 1, match 3816): every one of the 24 shots is in exactly one attack; the 3 real goals are marked; no overlaps; the score follows the probability (rho 1.00); the penalty scores 100; the best spot reached before any shot predicts that the attack ends in a shot (AUC 0.73, not circular); danger hardly depends on attack length (rho +0.16); xG v2 behaves like football (6 m 0.44, penalty spot 0.18, 17 m 0.07, 25 m 0.02, header 0.09, two blockers 0.11); our shot xG agrees with StatsBomb's xG for the same 16 matched shots (r = 0.91, mean difference 0.05). Team attack xG: Argentina 2.30, Saudi Arabia 0.23 (PFF published 2.34 / 0.17, StatsBomb 1.71 / 0.15, Opta Saudi 0.14). An independent re-implementation (`~/diag/audit_attacks.py`) reproduces shot xG (max difference 0.0004), attack xG and scores (2 of 141 differ by 1 from rounding), peak xT, xT gained, defenders goal-side (135/135), replay positions (8,954 checked, 0 wrong) and possession (1,055/1,055); no dead-ball event lies inside an attack; moving the 12 s / 3 s / 3 s thresholds keeps the top 10 10/10 in all six variants.

**Expert check (to do)**: `python attacks_expert_check.py --match 3816 --make-sheet expert_sheet.csv` writes a blind sheet of 30 attacks spread over the danger range; coaches rate each 1-5 without seeing the scores; `--score expert_sheet.csv` reports agreement between raters (Kendall's W), the rank correlation of the app's danger with the experts (with bootstrap intervals) and the same for shot-xG-only, best-spot-only and "reached the box" baselines (method of Link, Lang & Seidenschwarz 2016).

Ball-track repair (loader v1.4): on 61 events (3 %, mostly clearances, duels and rebounds) the tracked ball was more than 5 m (up to 70 m) from the player who played it. The event now takes the player's position, and the ball track between the two nearest events that agree with their player is rebuilt through the event points (or left empty when no trustworthy neighbour is close). Afterwards no event has the ball more than 5 m from its player, and single-frame ball jumps above 60 m/s fell from 120 to 71.

Limits: the EPV grid is Shaw's published grid, not refitted on this data; xG v2 does not know the shot's height or pass type (cross, through ball) or the shooter's foot; one match only, and no expert ratings yet. StatsBomb open data may be used for non-commercial research with attribution ("Data: StatsBomb", logo in publications).

## 4. Limits, stated plainly

1. **Single match.** All checks use one match with 24 shots. Parameters of the pass model are fitted on it; how well they carry over to other matches is not yet tested (the validator can be run on any match).
2. **Only DAS comes close to adding something beyond ball position, and it is not proven.** Cross-validated AUC for 'shot in the next 6 s': EPV alone 0.856, EPV + DAS 0.875 (gain +0.018, 95% interval -0.012 to +0.053), EPV + PC 0.799, EPV + OBSO 0.779. With 28 shots the intervals include zero, so none of them can be called better than ball position. They are best described as *spatial context* consistent with shot build-up.
3. **Raw VAEP is slightly negative on average** (-0.005 per action, because losing the ball costs more than a short pass gains on the EPV grid), so raw totals are not "goals". Use `vaep_oa`, and compare players within the match.
4. **VAEP and OBV are approximations built from an external EPV grid**, not the trained models of the published VAEP (Decroos et al.) and OBV (StatsBomb), which need many matches to fit.
5. **OBV is an expected value of the intended pass** and includes the risk of losing the ball. It is *not* StatsBomb's outcome-based OBV. On open-play shots it is close to 0 on average (+0.02): the shot's xG v2 replaces the EPV of the spot (3.64 vs 2.59 expected goals over the 24 shots, including the 3 goals ruled out; xG v2 sees the keeper and the blockers, the EPV grid does not). Penalties are valued at the usual conversion rate (0.79), not by spot, so Messi's penalty gets OBV +0.63 (VAEP +0.84).
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
- Hudl StatsBomb. *Open Data* (shots and freeze frames used to fit xG model v2; non-commercial use with attribution). https://github.com/statsbomb/open-data
- Singh, K. (2018). *Introducing Expected Threat (xT)*. https://karun.in/blog/expected-threat.html
- Link, D., Lang, S., Seidenschwarz, P. (2016). *Real Time Quantification of Dangerousity in Football Using Spatiotemporal Tracking Data*. PLOS ONE 11(12). (expert-rating validation)
- Robberechts, P., Davis, J. (2020). *How Data Availability Affects the Ability to Learn Good xG Models*. (freeze-frame features, sample size)
