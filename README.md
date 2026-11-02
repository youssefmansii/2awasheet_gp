# Football Tracking Replay

A React replay viewer and local possession-analysis demo for large football tracking JSONL files. Replay streams the selected file through a Web Worker and renders the pitch with Canvas 2D; the possession demo uses a local Python companion API.

## Possession and event demo

Run `npm run demo:possession` to open the local tracking-file demo at
`http://127.0.0.1:5173/?view=possession`. It estimates player interaction,
likely ball control, provisional team possession and 25 types of event candidates.
An optional separate events file provides validation after prediction.

See the [setup and demo guide](docs/POSSESSION_DEMO.md) and the
[complete plain-language PDF](docs/possession_demo_complete_simple_guide.pdf)
for calculations, examples, validation steps, measured results and limitations.
Control is currently a heuristic; possession is not an official match statistic.

## Run locally

Requirements: Node.js 20+ and pnpm 11 (or Corepack).

```bash
pnpm install
pnpm dev
```

Open `http://localhost:5173`, then choose a local tracking `.jsonl`/`.json` file, optional metadata `.json`, and optional event-array `.json`. Match data is not bundled with the public repository and never leaves the browser.

## Verification

```bash
pnpm lint
pnpm typecheck
pnpm test
pnpm test:e2e
pnpm test:performance
pnpm build
```

`test:e2e` starts the Vite app and runs a headless Chromium smoke test against the committed real sample. Install its browser once with `pnpm exec playwright install chromium`. `test:performance` generates an ignored 150,000-frame JSONL file, streams it through the real parser and workload accumulator, and checks the 60-second / 600-MB budgets.

The root scripts also work through npm after dependencies are installed (`npm run test`, etc.).

## Data behavior

- Tracking data is parsed line-by-line with `file.stream()` and `TextDecoderStream` in a Web Worker.
- Full frame objects are never retained. Both raw and smoothed positions are packed into typed arrays and transferred to the main thread without copying.
- Blank or malformed lines are skipped and counted.
- Player identity is always resolved with `team + jerseyNum`; interpolation scans team slots by jersey rather than relying on array order.
- Metadata can be a one-item array or a plain object. It controls team labels, kits, pitch dimensions, and half ranges.
- The event file is a standard JSON array. It is parsed off the main thread and indexed by `eventTime × 1000` against tracking `videoTimeMs`.
- Tracking captures, match metadata and event data are ignored by Git and must stay local.

## Reference artifact

- `docs/reference.png` — supplied visual reference; no match-data JSON is published

Inspect any larger capture without loading it into memory:

```bash
node scripts/scan-jsonl.mjs path/to/tracking.jsonl
```

The scanner reports every distinct key path and event type with counts without loading the full capture into memory.

## Phase 1 scope

Implemented: tracking/metadata/event loading, worker progress, compact raw/smoothed data, metadata validation, responsive pitch, shirts/numbers/ball, confidence fading, real-time playback and scrubbing, playback speeds, frame/5-second/half navigation, interpolation, kit controls, side flipping, halftime handling, selected-player trails, event timeline/filtering, pass/shot overlays, roster identity, playhead-limited technical statistics, a live all-player workload table, five-minute workload charts, and output-decline trends.

### Workload methodology

Workload is derived during the streaming parse from consecutive smoothed positions keyed by `team + jerseyNum`. Defaults reject gaps over 200 ms and speeds above 12.5 m/s. High-speed running is the 5.5–7.0 m/s zone, sprint distance starts at 7.0 m/s, sprint bouts require at least 1.0 continuous second, and acceleration/deceleration uses a 0.5-second speed window with a ±2.5 m/s² threshold held for 0.5 seconds. These values are configurable in the UI and recomputed in a streaming worker. Compact cumulative checkpoints are recorded about once per second so every displayed value is limited to the current replay frame rather than the completed match. Output trends compare the most recent valid five minutes with the preceding valid 15 minutes, require 80% tracking quality by default, and respect parsed substitution states.

Ball-in-play filtering is not applied because the supplied provider events do not contain a documented, continuous ball-in-play state. Adding an inferred filter would risk excluding valid workload. Halftime discontinuities and invalid tracking gaps are excluded, and this limitation is shown here explicitly rather than silently claiming ball-in-play normalization.

Provider event codes are retained in the UI and raw event payloads are preserved. `C` is treated as a completed pass for the completion statistic; goal-like `G` shot outcomes are shown as raw outcomes until the provider dictionary confirms their semantics.

No authentication, backend, Supabase connection, or unverified score is included.

## Environment

Phase 1 requires no environment variables. `.env.example` reserves the Phase 3 server-only Supabase values; never expose the service-role key to the browser.
