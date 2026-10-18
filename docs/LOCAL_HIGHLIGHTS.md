# Local replay: Timeline and Highlights

Local replay offers two views above the existing transport controls:

- **Timeline** keeps the full-file seek bar, frame stepping, five-second jumps, half shortcuts and event timeline.
- **Highlights** builds a chronological reel from a matching local events JSON file. Press Play highlights to play clips continuously, skipping the match time between them. Playback stops at the last clip; pressing Play again restarts the reel. Previous/Next, First/Last, a clip selector and a highlights-only seek bar are available. The existing speed selector applies to both modes.

Load the tracking JSONL and the matching event-array JSON through the existing local file selectors. Choose Highlights, then Play. Missing events, mismatched tracking/event match IDs and no matching actions in the loaded recording produce explanatory empty states rather than invented clips. Replacing tracking/events returns to Timeline and pauses playback. Switching views also pauses, without changing the data files. Live API and its transport are unaffected.

## Clip selection and timing

Selection uses the parsed source `eventTimeMs`, not the formatted match clock. Recognized actions are shots (`SH` possession or `SHOT` game type), provider goal records (`GOAL`), substitutions (`SUB`/`SUBSTITUTION`), and explicit card records. Cards recognize `YC`, `RC`, `2YC`, `YELLOW`, `RED`, `SECONDYELLOW`, `YELLOWCARD`, `REDCARD`, `SECONDYELLOWCARD`, `CARD`, `BOOKING`, including underscore/space/hyphen variants, either as the game type or `payload.gameEvents.cardType`. Ordinary passes and fouls are excluded; a foul with explicit card metadata is included. Unrecognized provider codes are not guessed.

Each action gets five source seconds before and five after its timestamp. Windows are clipped to loaded tracking timestamps and metadata half boundaries when available, then aligned to actual recorded frames. Events outside the recording, outside known halves, or with no recorded frames in their window are omitted. Overlapping windows merge into one clip retaining the action labels, so nearby events are not replayed repeatedly. The UI shows the resulting clip count and the duration of this reel, excluding skipped match time. This is a deterministic event-based selection, not a video/CV highlight detector.

The parser preserves raw goal outcomes that may include disallowed goals; see [event validation](event-validation.md). Consequently `SH` + `G` and `GOAL` records are labelled **Goal candidate**, not asserted as verified goals. Score validation remains unchanged.

## Analytics and privacy

Only the playhead changes. Player positions, match clock, score, event overlay and workload snapshots follow the selected source time. Workload remains the existing full-match-to-current-frame calculation; it is **not** recalculated as if the skipped match intervals never happened or as highlights-only player effort. Automatic clip jumps refresh the analytics playhead immediately; normal continuous playback retains its approximately one-second analytics update cadence.

The streaming JSONL parser, event parser and workload formulas are unchanged. Clip generation uses the already parsed events and binary searches of existing timestamp arrays; it does not reread or upload the full tracking file. No new dependencies, backend endpoints, cloud retention, API keys or persisted highlight files are involved.

## Verification

`pnpm test` covers automatic selection, explicit card metadata, merged windows, recording/half boundaries, unavailable tracking, gap skipping, elapsed-time carry and highlights-only progress mapping. `pnpm test:e2e` covers controls, auto-next, end/restart, seeking, switching back to Timeline, missing/wrong-match events, unchanged Live API and 320/768/1024/1440-pixel layouts. The browser test uses the real local sample if present and a small contract fixture in public checkouts; its test event records are explicit fixtures, not claims about actions in that real footage. Screenshots remain in ignored `test-results/`.

Chrome DevTools MCP is unavailable in this workspace; isolated Playwright browsers provide runtime checks. Rollback is frontend-only: remove/revert this feature's UI/helper changes without changing tracking data, parsers, workload calculations or the deployed Cloudflare API.
