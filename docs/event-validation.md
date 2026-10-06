# Event validation — match 3816

## Score validation

The event parser treats an event as a **goal candidate** only when `possessionEventType` is `SH` and `shotOutcomeType` is `G`. It preserves the provider clock, team, period, shooter, and source timestamp. No candidate is removed or changed to make the result agree with the known score.

| Team | Raw `SH`/`G` count | Provider clock(s) | Known final goals | Result |
|---|---:|---|---:|---|
| Argentina | 4 | 09:14 (Lionel Messi), 21:45 (Lionel Messi), 26:55 (Lautaro Martinez), 34:12 (Lautaro Martinez) | 1 | Mismatch: +3 |
| Saudi Arabia | 2 | 47:39 (Saleh Al-Shehri), 52:43 (Salem Al-Dawsari) | 2 | Match |
| **Total** | **6** | — | **3** | **Mismatch: +3** |

The raw event result is therefore **Argentina 4–2 Saudi Arabia**, while the known final result is **Argentina 1–2 Saudi Arabia**. The three extra Argentina candidates align with the three first-half Argentina goals that were disallowed for offside in this match. The supplied event records still code their shot outcome as `G`; no reliable allowed/disallowed field accompanies those records. This is the likely cause, but remains an inference rather than a provider-dictionary guarantee.

Because validation fails, the application deliberately shows **no score and no possession badge**. It displays a validation warning in the Match view instead. This prevents raw outcome codes from being presented as an official score.

## Possession and ball-in-play limitations

The standalone event array does not provide `home_ball` on its parsed events. The tracking JSONL contains `home_ball` only inside sparse embedded `game_event` records, not as a continuous frame-level possession state. A trustworthy possession percentage cannot therefore be derived from the supplied event file alone.

`OUT` events and restarts exist, but their pairing is not documented well enough to reconstruct every dead-ball interval reliably. Ball visibility is also not equivalent to ball-in-play. The workload model consequently excludes halftime and invalid tracking gaps but does **not** claim a ball-in-play-normalized work rate. This is intentionally skipped rather than guessed.
