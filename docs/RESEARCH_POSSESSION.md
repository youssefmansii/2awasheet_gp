# Research-informed possession and event detection

The separate offline command implements trajectory smoothing, interaction starts
and action ends inspired by [Link & Hoernig (2017)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5503225/).
It also adds event candidates derived from geometry and controller transitions.
The old baseline remains available in `scripts/tracking_possession.py`.

## Run locally

From the repository root, using the existing environment and dependencies:

```powershell
.venv/Scripts/python.exe -B scripts/research_possession.py --tracking data/tracking_3816.jsonl --events data/events_3816.json --source smoothed --output output/research-possession-3816-smoothed
.venv/Scripts/python.exe -B scripts/research_possession.py --tracking data/tracking_3816.jsonl --events data/events_3816.json --source raw --output output/research-possession-3816-raw
.venv/Scripts/python.exe -B -m unittest discover -s apps/api/tests -p test_research_possession.py -v
```

This does not alter the API, dashboard, live simulation or default baseline. All
data stays local. The source JSONL is streamed, but the model retains compact
full-match arrays, smoothing states and frame results. It is not a bounded-memory
live implementation. The RTS backward pass explicitly uses later observations.

## What the model does

1. Resolve players by team plus jersey. Remove repeated frame/time/period keys
   by retaining the first observation and audit conflicts. Never use embedded
   event payloads or IDs for prediction.
2. Use estimated coordinates with reduced tracking weight. Fill internal gaps
   of at most 0.3 s between valid observations in the same period, without a
   source-frame gap over 0.2 s. Missing longer sections remain unknown.
3. Quarantine transitions over 15 m/s for players and 60 m/s for the ball.
   Apply constant-velocity Kalman filtering and Rauch–Tung–Striebel smoothing
   independently to xy trajectories. Measurement variance increases as quality
   decreases. The continuous acceleration-noise spectral densities are 16 for
   players and 400 for the ball; measurement SD starts at 0.3 m before quality
   scaling. These are uncalibrated implementation choices, not supplied paper
   parameters.
4. Detect local ball-acceleration peaks >=4 m/s² near players. The nearest
   plausible approach within +/-0.1 s gives the candidate actor. Peaks are
   separated by approximately 0.15 s. A separate stable-proximity onset can
   recognise a slow reception without a sharp kick. Low-ball acceleration peaks
   are labelled `kick_candidate`; no skeletal evidence proves foot contact.
5. Start interaction intervals at those onsets. End them when another player
   interacts, tracking is lost, the period changes, the ball exits, or the
   one-second extrapolated ball position is beyond a 4 m action reach. Contact
   at release can be recognised without sustained control. Extract duration,
   distance, ball speed/acceleration means/variances and opponent counts within
   0.5–5 m for every interval.
6. Estimate physical interaction and heuristic likely control separately.
   Retain a prior interaction owner for provisional team possession for up to
   five seconds while the observed trajectory stays valid. Contested frames
   have no current carrier/team; their prior owner is history for classifying
   the challenge, not an assertion of current control.

The paper's distance thresholds were trained. Our proximity defaults, additional
quality handling and event extensions are adaptations. The provider-smoothed
positions may themselves be annotation-assisted; their provenance is unknown.
Raw input is evaluated separately and produces materially different performance.

## Outputs and meaning

- `REPORT.md`: readable summary, category metrics and limitations.
- `report.json`: configuration, audit, durations, metrics and per-period results.
- `frames.csv`: `carrier` means a physical-interaction estimate. `control_status`
  distinguishes `interaction_only`, `likely_control` and `unknown`. Without a
  reviewed trained model, `control_probability` is blank. `team` and
  `possession_owner` are provisional and can continue through ball travel.
- `predicted_events.csv`: candidate type, time, end, actor, target, evidence and
  optional challenger. Overlapping candidates express ambiguous interpretations,
  such as a goal-directed release that might be a cross.
- `event_matches.csv`: one-to-one prediction/reference joins and timing errors;
  contest rows also compare the eventual winner when both sides name one.
- `actor_validation.csv`: exact-time sparse event-actor comparison, including
  abstentions. This is not dense control ground truth.
- `control_annotations.csv`: interval features and blank reviewed control labels.
- `players.csv`: identity names joined after prediction and partial interaction time.
- `trajectories.npz`: processed positions, motion, quality and reconstruction masks.

## Event coverage

| Candidate | Evidence | Validation scope |
| --- | --- | --- |
| Touch / kick | Proximity with acceleration peak, or stable interaction onset | Sparse actor recovery; touch inventory is not exhaustive, so no touch precision claim |
| Completed pass / reception | Same-team interaction transfer along a valid path | Pass sender, receiver and release time checked; reception time not independently labelled |
| Unsuccessful pass | Ball travels and an opponent subsequently interacts | Compared with non-C PA/CR records; cannot always distinguish clearances/deflections |
| Carry | Sustained heuristic/learned control with ball displacement | Only nine BC labels in the supplied file; sparse labels do not validate all carries |
| Contest and winner | Both teams near the ball; prior controller and subsequent interaction | CH participant pair, time interval and winner checked where available |
| Tackle / interception / recovery | Opponent transfer plus nearby contest or ball flight | Subtypes are heuristic and not verified as official tackle labels |
| Retained challenge / dribble | Same player resumes interaction after a contest; movement for dribble | Candidate only; no verified subtype dictionary or dense labels |
| Pressure | Nearby opponent during an interaction episode | Spatial proxy, not validated tactical pressure |
| Shot | Fast release towards the attacking goal plane | SH actor and timing checked; crosses and shots can be confused |
| Cross | Wide-to-central attacking transfer | CR actor and release time checked; current detector covers completed transfers only |
| Clearance | Fast forward release near the defending goal under pressure | CL actor and timing checked; currently weak |
| Out | Ball centre crosses pitch boundaries | OUT timestamp checked; centre crossing does not prove whole-ball crossing |
| Restart types | Stationary release near centre, touchline, corner or goal area | Candidate only; no validated continuous stoppage/restart state |
| Goal-line crossing | xy crossing between posts with reported height below the crossbar | Geometry candidate; never an official allowed goal or score |

Do not produce official fouls, cards, offside or VAR decisions from these tracks.
The code makes no claim that every listed candidate class is accurate. Inspect
the per-class precision, recall and support before using any class operationally.

## Validation rules

Predictions are frozen before opening the event file. Actor names and provider
IDs are then joined to team/jersey for evaluation only. Exclude flagged non-events,
missing video and unmappable actors. Generic event actors are touch/control proxies.

Use maximum-cardinality, minimum-timing-error one-to-one assignment within each
period and participant identity. Touch/kick/shot/cross/clearance/contest tolerance
is 0.6 s; completed/unsuccessful pass and out tolerance is 1 s. Pass/cross matching
uses release time, not the whole flight interval. Completed passes require the
exact sender and receiver; contest matching requires both participants. Control
frames still report an exact-time comparison separately, with abstentions counted.

Sparse touch/kick and BC references cannot establish precision for all generated
contacts/carries, so precision is null for those categories. A missed BC reference
is reported as missed; no silent redefinition makes it agree. Referee goal outcomes
and tackle/interception subtypes remain unvalidated. This is one-match validation,
not a held-out test of generalization; default thresholds were not tuned to these
labels. Per-period reports expose first/second-half differences.

There is no validated continuous ball-in-play flag. Team time includes possible
in-bounds stoppages. Report the share of *assigned tracked time* and its coverage;
never advertise this as official whole-match possession.

## Optional learned control classifier

The paper uses a trained K2 Bayesian network to distinguish true control from
other interactions. Its trained parameters are not available in this project.
Our optional Gaussian naive Bayes model is a simpler Bayesian network over the
paper's feature family. It is explicitly a different classifier and is not
automatically trained from event-actor pseudo-labels.

Review intervals against video and fill `reviewed=yes`, `label_control=0` or `1`
and `split=train` or `test` in the exported annotation CSV. Keep contiguous test
clips separate from training clips. The trainer requires at least 20 reviewed
training examples and five reviewed test examples of each class and rejects
overlapping/adjacent train/test intervals from the same capture and period.
These minimums establish an executable contract, not statistical adequacy.

```powershell
.venv/Scripts/python.exe -B scripts/research_control.py --annotations output/research-possession-3816-smoothed/control_annotations.csv --output output/research-possession-3816-smoothed/control_model.json
.venv/Scripts/python.exe -B scripts/research_possession.py --tracking data/tracking_3816.jsonl --events data/events_3816.json --source smoothed --control-model output/research-possession-3816-smoothed/control_model.json --output output/research-possession-3816-with-model
```

An untouched annotation template correctly refuses training. Model probabilities
are uncalibrated classifier posteriors. No learned-control accuracy is claimed
until actual reviewed labels train and test the model. Keep training and held-out
control metrics separate from the sparse event detection metrics.
