# Research-informed tracking events: implementation contract

Approved scope: adapt Link & Hoernig (2017) to the local capture for player
interaction/control estimates, team possession, kicks/touches and event candidates.
Run offline, keep the existing baseline and dashboard intact, and validate against
the separate events file only after tracking predictions have been frozen.

The paper is https://pmc.ncbi.nlm.nih.gov/articles/PMC5503225/ (sections 2–4).
Implement trajectory cleaning, RTS smoothing, proximity/acceleration interaction
starts, one-second ball-prediction action ends and interval feature extraction.
The paper's distance thresholds and trained K2 network are not supplied here.
Configurable defaults are adaptations, and optional Gaussian naive Bayes control
classification is a simpler network with the paper's feature family, not its K2
implementation. Until explicitly reviewed interval labels train that model,
publish `likely_control` as a heuristic; never claim learned IBC probabilities.

Outputs: frame-level interaction/control estimates and provisional possession,
touch/kick candidates, action features and annotation template, pass/reception,
carry, recovery/interception/tackle candidates, contested intervals, pressure,
shot/cross/clearance and out/restart/goal-line-crossing candidates where geometry
supports them. These are candidates, not official referee/VAR judgements.
Missing long trajectories stay unknown. Do not infer an official ball-in-play
flag from visibility. State explicitly that inferred possession includes possible
in-bounds stoppages and is not an official match possession percentage.

Prediction inputs: timestamps, periods, player team/jersey, xy, ball xyz and
quality. Embedded annotations/IDs, event snapshots, event actors and outcomes must
not influence tracking prediction. Provider-smoothed source provenance is unknown.
RTS processing uses later frames and is explicitly retrospective.

Validation: report exact-time carrier coverage/agreement and separate touch-event
matching; never equate a touch label with dense control ground truth. Use one-to-one
matches, period and actor/participant identities, report false positives, false
negatives, precision, recall and F1. Event subclasses need their own reference
sets and support counts. Report both periods separately and do not tune on them
while calling them held-out data. Challenge winners are not automatic controls at
the initial timestamp. Event dictionaries and ambiguous semantics remain limits.

Implementation slices: (1) compact tracking loading and trajectory processing,
(2) action/possession and event inference, (3) independent validation and exports,
(4) optional reviewed-label training and documentation. Synthetic regressions
cover missing/estimated input, one-touch contacts, carry/passing, contests/winners,
gaps/period resets, dead-ball boundaries, prediction/event isolation and matching.
Run the new tests, baseline tests and real match commands. No dependencies,
uploads, deployment, event-assisted inference or unrelated changes are required.
