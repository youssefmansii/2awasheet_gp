# Possession and event detection from tracking

The newer [research-informed implementation](RESEARCH_POSSESSION.md) adds
trajectory smoothing, touch/action intervals, event candidates and optional
reviewed-label control training. This document describes the original baseline.

This local analysis detects actions that have happened. Forecasting future passes
or shots is a different modelling task. Run the baseline with the existing Python
environment, from the repository root:

```powershell
.venv/Scripts/python.exe -B scripts/tracking_possession.py --tracking data/tracking_3816.jsonl --events data/events_3816.json --output output/possession-3816-smoothed --source smoothed
.venv/Scripts/python.exe -B scripts/tracking_possession.py --tracking data/tracking_3816.jsonl --events data/events_3816.json --output output/possession-3816-raw --source raw
.venv/Scripts/python.exe -B -m unittest discover -s apps/api/tests -p test_tracking_possession.py -v
```

The predictor reads only frame number, video time, period, player jerseys,
positions, confidence/visibility and ball positions/visibility. Embedded
`game_event`, `possession_event`, their IDs and event-file snapshots never feed
the prediction. Prediction finishes before the event file opens. That file then
supplies sparse validation labels and the jersey-to-name lookup in `players.csv`.
No fitted parameters or additional dependencies were introduced. This script
does not change the replay dashboard or replace the analytics API.

## Definitions and method

Player control and team possession are separate outputs:

- `carrier`: a confirmed `home:jersey` or `away:jersey`, or empty when uncertain.
  Jerseys alone do not identify players across teams.
- `team`: the controller's team, or a provisional previous team during a short
  moving-ball interval. A ball in flight has no assigned individual carrier.
- `status`: controlled, pending, in flight, contested, ambiguous, missing,
  low quality, loose ball, tracking jump or outside the pitch.
- `nearest`: the unconstrained nearest-player baseline, for comparison. It is
  not an assertion of control.

The default conservative rules are fixed heuristics, not learned parameters:

1. Find the nearest and second-nearest players in pitch metres. Require visible
   ball tracking, a visible player with HIGH/MEDIUM confidence, and a positive
   period. Estimated or missing confidence does not silently become HIGH.
2. Require distance at most 2 m (2.5 m for the previous controller), a 0.5 m
   advantage over the second-nearest player, ball height no greater than 2.5 m
   when supplied, and ball/player relative speed no greater than 15 m/s when
   measurable. Unknown height is not evidence that the ball is on the ground.
3. Confirm a candidate after 0.1 s of continuous evidence. This confirmation is
   causal and adds latency; it does not backdate control to earlier frames.
4. Reject ball speeds above 45 m/s and player speeds above 12.5 m/s as tracking
   jump candidates. Missing data, low quality, contests, period changes and
   gaps longer than 0.2 s reset the relevant state.
5. When control is lost to a moving ball (above 2 m/s), retain only a provisional
   team for up to 3 s since last confirmed control. Clear the team on uncertain
   contests, missing data and out-of-bounds candidates. This provisional label
   can be wrong on an intercepted or unsuccessful pass; it is not confirmed
   continued possession.

`evidence_score` is an uncalibrated distance/separation score, not a probability.
Unknown states remain empty. Do not fill them indefinitely from the last player
or latest event. The existing `MetricsEngine.possession_side` uses the latest
event's team and is unsuitable as a tracking-only estimator or an independent
validation reference.

Durations use `[video_time_i, video_time_(i+1))`, not frame counts or an assumed
frame rate. Exclude period boundaries and gaps over 0.2 s; do not extrapolate
the final frame. Repeated identical frame/time/period keys keep their first
observation. Duplicate counts and conflicting tracking payloads are reported.
This is deterministic and does not select the duplicate closest to an event.

For a team A:

```text
assigned tracked-time share(A) = assigned seconds(A) / (assigned seconds(home) + assigned seconds(away))
assignment coverage = assigned seconds(both teams) / accounted tracking seconds
player control time = sum of valid intervals with that player confirmed
```

**These are not official ball-in-play possession percentages.** A player near
a stationary ball during an in-bounds stoppage can still meet the control rules.
The data lacks a validated continuous ball-in-play label. Report unknown time
and coverage alongside shares, and do not compare player control totals as
complete match totals.

## Events supported by tracking

| Event or state | How to infer it | Baseline status and limits |
| --- | --- | --- |
| Player control / team possession | Distance, height, relative movement, quality and temporal continuity | Implemented; sparse event-actor validation, with abstention |
| Completed pass / reception | Confirmed control changes between teammates, short ball transit and at least 3 m travel | Implemented as candidates; completion becomes observable only at reception; does not separate crosses |
| Recovery / turnover | Confirmed controller changes to the other team | Implemented as candidates; not validated as tackle versus interception |
| Carry | Same controller moves with the ball over a sustained interval | Feasible next step; the provider's nine BC labels do not label every possible carry |
| Dribble / take-on | Carry near an opponent followed by retained control and separation | Candidate classification; needs a definition and duel labels |
| Shot | Release near goal, ball speed/direction towards the goal, subsequent trajectory | Feasible candidates; shots, crosses and clearances can overlap; not implemented |
| Out of bounds | Ball crosses known pitch limits | State candidate implemented; centre coordinates do not establish whole-ball crossing |
| Corner / throw-in / goal-kick / free-kick | Out/stoppage followed by restart position and player configuration | Feasible candidates; last touch, ball-in-play state and referee decisions add ambiguity |
| Pressure | Opponent distance, closing velocity or time to reach the carrier | Feasible proxy, requiring calibration against labelled pressure |
| Offside / goal | Kick time, defender positions or goal-plane crossing | Candidate only; body-part geometry, ball dimensions and referee/VAR decisions are absent |
| Foul, card, official allowed goal, VAR outcome | Referee decisions and incident interpretation | Cannot be established reliably from these positional tracks alone |

Transfer candidates stop across quality failures, missing data, contests,
out-of-bounds intervals and frame gaps. A same-team transfer is not proof of a
deliberate pass; one-touch actions and short passes may be missed. Recovery
candidates are deliberately not advertised as verified interceptions or tackles.

## Validation

Compare event `eventTime` in seconds against tracking `videoTimeMs / 1000`,
within the **same period**, with a maximum alignment error of 67 ms. Do not align
with rounded match clocks or subtract 45 minutes from second-half video time.

Use the appropriate role as the actor: passer for PA, crosser for CR, shooter
for SH, clearer for CL, ball carrier for BC, touch player for TC, rebounder for
RE, and game-event actor for IT. Exclude flagged non-events/missing video,
unmapped identities, and unsupported or ambiguous types such as CH; a duel has
multiple participants and its winner is not necessarily controlling the ball at
the timestamp. Counts of excluded labels and identity conflicts are explicit.
Event actor identity at a kick or header is a **proxy for a touch**, not guaranteed
sustained player control at that instant.

Report correct/total labels with abstentions counted, accuracy on covered labels,
carrier coverage, team agreement, nearest-player baseline, and breakdowns by
event type and period. The separate +/-250 ms diagnostic can find an actor
before/after the label; it is retrospective and not the exact-time accuracy.

Completed-pass validation uses only non-event-filtered PA/CR outcomes `C` with
mapped sender and receiver. Match candidates one-to-one, closest timing error
first, with exact sender/receiver and period and release-time error <=1 s.
Unmatched predictions are false positives; unmatched references are false
negatives. Report precision, recall and F1, not only the share of matched
predictions. Other event classes are not scored by this matcher.

The event file is sparse and comes from the same provider. The provenance of
the supplied smoothed positions is unknown: strong agreement with events does
not establish that the provider's preprocessing was independent of annotation.
The script adds no retrospective smoothing, but the supplied smoothed input is
not proven causal. Neither raw nor smoothed results establish independent sensor
accuracy, full-frame control accuracy or generalization to another match.

To improve beyond this baseline, annotate contiguous clips with player control,
loose/contested ball and ball-in-play/dead-ball intervals. Include missing-ball,
one-touch, aerial, goalkeeper and restart examples. Fit motion/quality features
or a temporal state model on training clips; keep contiguous later clips and
other matches untouched for validation. Evaluate time-weighted team agreement,
carrier identity accuracy versus coverage, transition delay and event precision/
recall. Do not split adjacent frames randomly or train on embedded event fields.

## Outputs

Each output folder contains:

- `frames.csv`: every retained frame, controller/team, reason for uncertainty,
  positions, motion evidence and nearest-player baseline.
- `players.csv`: identity/name lookup and observed confirmed control seconds.
- `event_validation.csv`: every eligible event label, expected/predicted player,
  alignment, uncertainty state and match results.
- `predicted_events.csv`: completed-pass and recovery candidates with release/
  reception times and participant identities.
- `pass_matches.csv`: the one-to-one completed-pass matches and timing errors.
- `report.json`: parameters, source audit, durations, shares, validation counts
  and per-type/per-period metrics.

Generated outputs and private input files stay in ignored directories. The raw
JSONL is streamed, but per-frame prediction records are retained for validation
and export; memory usage grows with match length. This is an offline analysis
command, not the bounded-memory live server.

## Research basis

The separation between individual control, individual actions and team
possession follows the concepts in [Link and Hoernig, Individual ball possession
in soccer (2017)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5503225/).
The control-first, event-second structure is also described in
[Vidal-Codina et al., Automatic event detection in football using tracking data
(2022)](https://arxiv.org/abs/2202.00804). This script is a simple baseline, not a
reproduction of either paper; its thresholds and measured results are its own.
