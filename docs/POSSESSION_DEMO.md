# Possession demo

The [complete plain-language PDF](possession_demo_complete_simple_guide.pdf)
explains every calculation, threshold, exported field, validation method and
saved benchmark result. Match data and generated analysis outputs stay local.

## First-time setup

Install the workspace JavaScript dependencies using the package manager specified
in the root `package.json` (`pnpm install --frozen-lockfile`). Create the Python
environment and install the existing API requirements. On Windows PowerShell:

```powershell
py -m venv .venv
.venv/Scripts/python.exe -m pip install -r apps/api/requirements.txt
```

On macOS/Linux, use `python3 -m venv .venv` and `.venv/bin/python` instead.
For Python regression tests also install `apps/api/requirements-test.txt`.

## Open the app

```powershell
npm run demo:possession
```

Open [the local demo](http://127.0.0.1:5173/?view=possession). Keep the terminal
running. This starts the existing Vite app and a separate local research API on
127.0.0.1:8010. The existing Python `.venv` and JavaScript dependencies are used;
no extra packages are required. The launcher reuses services already on those
ports and stops only processes it started when you press Ctrl+C.

1. Choose a tracking **.jsonl** file. Choose Raw if the file has no smoothed arrays
   (the picker switches automatically when it can identify this).
2. Optionally choose the provider's **events JSON array** for validation.
3. Press **Analyse locally**. File selection alone never uploads anything.
4. Replay the pitch, change speed, scrub, jump between periods, click players,
   filter/search event candidates and select a row to seek to its source time.
5. Use **Player interaction** for full-recording totals and **Validation & method**
   for calculations, support counts, precision/recall and unvalidated categories.
6. Choose an export to download frames, events, players, control features/labels,
   validation matches or reports.

**Open analysed match** appears only when the owner's previously generated local
match outputs exist. It opens those read-only outputs after an explicit click;
private data is not bundled with the app or automatically opened.

The pitch/side panel follow the current frame. Overview possession shares, player
totals, candidate counts and validation describe the full analysed recording.
Source timestamps are video times, not the match clock. Playback skips long source
gaps between tracked segments. All event labels are candidates. Control remains a
heuristic until a reviewed model is trained; possession is not an official
ball-in-play statistic. Tracking-only results have no invented accuracy scores.

Uploaded copies and derived artifacts live in ignored `output/possession-demo/jobs/`.
A new job replaces the previous demo-owned job. The analysed reference outputs
are preserved. Cancellation terminates only the demo's analysis subprocess.
The service is local-only; it is not a multi-user or remotely deployed API.

## Implementation contract

User-authorized scope: add a local demo to the existing app. Select a tracking
JSONL, optionally select provider events, choose raw/smoothed inputs and explicitly
start local analysis. Replay the result on a pitch with player interaction,
heuristic control, provisional possession, all event candidates, player totals
and honest validation metrics. An explicit button may open the owner's already
analysed match; nothing loads private match data automatically.

Use the existing dark/green visual language. Put replay and frame state first,
then searchable event records, player summaries and validation/method tabs.
Preserve replay/analytics views. Support 320 px–desktop layouts, keyboard-labelled
controls, missing-ball/contested states, progress, cancellation, errors and
tracking-only analyses without invented accuracy percentages.

Architecture: React demo tab plus separate loopback-only Python API on port 8010.
One cancellable subprocess computes research outputs; ready playback reads those
outputs. The source tracking never goes to a remote server. Multipart upload is
limited to the existing local 1 GiB boundary; filenames are metadata only. Allow
only one uploaded job at a time and fixed downloadable artifact names. Cancel
kills the owned worker. New uploads replace only demo-owned job data. Prepared
sample files are read-only. No browser profiles, secrets, deployment or new runtime
dependencies are required.

Verification: API fixtures exercise tracking-only uploads, validation absence,
pagination/filtering, missing frames, cancellation and artifact restrictions.
Run the real Python model on an uploaded synthetic capture. Browser tests verify
file selection does not upload, explicit analysis, progress, pitch/replay, filters,
validation, mobile overflow and cancellation. Inspect screenshots with isolated
Playwright (Chrome DevTools MCP is unavailable). Run TypeScript, lint, Python
regressions and production build. Launch with `npm run demo:possession`.

## Verified

- 12 isolated Chromium browser regressions pass, including three demo tests for
  explicit upload, real analysis/replay, filters, optional events, mobile layout
  and cancellation while job reservation is still in flight.
- All 49 Python regressions pass, including the real tracking-only API pipeline.
- Current shared/API suite: 133 unit tests pass; the existing opt-in performance
  test is not run by the ordinary suite. Use
  `node packages/shared/node_modules/vitest/vitest.mjs run --dir packages/shared/src`.
  The broad `npm test` also discovers old unrelated checkouts under `tmp/`, where
  three copied suites fail to resolve their workspace alias; those copies were
  not modified or deleted.
- TypeScript, ESLint and production build pass. Vite retains its existing large
  Advanced Analytics chunk warning; the new demo is a separate lazy-loaded chunk.
- The full 198,142-frame analysed match was opened in Chromium; replay, validation,
  frame export and desktop/mobile layouts were inspected without browser errors.
- Windows sandbox socket restrictions caused the earlier Python event-loop stall.
  Running local socket tests with the needed permission resolves it; the original
  API-boundary tests now pass.

Implementation references: [FastAPI file uploads](https://fastapi.tiangolo.com/tutorial/request-files/)
and [React effect cleanup](https://react.dev/reference/react/useEffect).
