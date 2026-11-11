# PROJECT_CONTEXT.md — hand-over file for any new chat

> **Purpose.** Paste or attach this file at the start of a new chat so the assistant understands the whole app, the team, the data, what is finished, what is not, and how the owner (Hatem) likes to work.
> **Last updated:** 2026-10-19 · **Branch with the newest work:** `feature/analytics-engine` (merged with `origin/main` at commit `ff2b4e5`, pushed to GitHub).
> Deeper metric details live in [`docs/METRICS.md`](METRICS.md). Friends' backend/viewer docs live in the other `docs/*.md` files (listed in section 6).

---

## 0. 60-second summary

- **Graduation project:** *Multimodal Football Player Load and Injury Risk Monitoring Using Computer Vision and Wearable GPS* (Cairo University, Faculty of Engineering).
- **One monorepo:** `2awasheet_gp` (GitHub `youssefmansii/2awasheet_gp`, local `C:\Users\lenovo\2awasheet_gp`). It contains three big things built by different people:
  1. **Match Replay viewer** (React, browser-only): loads a tracking file and replays the match on a pitch, with workload stats, events, highlights.
  2. **CV diagnostic API** (Node/Cloudflare) + **Live API viewer**: the computer-vision team uploads tracking frames, the viewer plays them back.
  3. **Advanced Football Analytics Engine + Dashboard** (Python FastAPI + React) — **Hatem's part**: tactical/cognitive metrics (Pitch Control, EPV, OBSO, DAS, VAEP, OBV) and live per-player statistics, for any uploaded PFF match.
- **Pilot data:** PFF FC FIFA World Cup 2022, **match 3816 = Argentina 1–2 Saudi Arabia**. The engine must work for *any* match file, not only 3816.
- **State:** Hatem's analytics engine is implemented, verified (34/34 recount checks pass), documented, merged with main and pushed. It still needs: a PR on GitHub, testing by Hatem in the running app, and a decision on re-implementing VAEP/OBV with open-source libraries (section 12).

---

## 1. The owner and how to work with him

**Who:** Mohamed Hatem (university ID 1220099). Role: **data analysis — tactical/cognitive layer**. He is a **beginner**.

**Working rules he gave (follow them):**
1. Explain **every step in simple words, one step at a time**. Do not dump ten steps together.
2. Be **honest about limits** (what is inferred, what is approximate, what is untested). Never claim more than was verified.
3. Use **open-source libraries** when they improve accuracy or save time (he explicitly gave permission).
4. Analytics must be **dynamic per match** — nothing hard-coded to match 3816.
5. **Do not modify teammates' components** beyond minimal integration in `App.tsx` / `main.tsx`. Keep his code isolated in `apps/web/src/components/analytics/` and `apps/api/*.py`.
6. `recharts` is approved for charts.
7. **Do not ask him to paste full raw data in chat** (files are huge). Ask for small excerpts or run things on his PC through the device bridge.
8. Before finishing any feature, **verify it works logically** (recounts from raw data, tests, UI test).
9. Make analytics **understandable for a viewer** (plain wording, "in words" boxes, guides) — not only numbers.
10. He wants to be told about mistakes **before** big edits to his own notebooks.

**Tone:** friendly, short sentences, practical. Give copy-paste terminal commands (Windows PowerShell on his PC).

---

## 2. The team

| Person | Area |
|---|---|
| **Mohamed Hatem** | Data analysis — tactical & cognitive layer (Pitch Control, EPV, OBSO, DAS, VAEP, OBV, cognitive-fatigue proxies, event analysis, this analytics dashboard) |
| **Amr Nasser** | Data analysis — data ingestion, Parquet, physical load, simulated stream |
| **Marwan Mohamed** | Computer vision |
| **Mohamed Hossam (Hoss)** | Computer vision (building on YOLO) |
| **Abdelrahman Osama** | Hardware/software, mobile app (also the funding-proposal contact) |
| **Youssef Mansi** | Software/hardware/middleware; owns the monorepo, Cloudflare/Render CV API, Replay viewer, Live API viewer, highlights |

Supervisors: **Dr Tamer El Basha** and **Dr Hossam Eldin** (weekly meetings). The doctor allowed using AI as an assistant on the project.

**Team plan (Data Analysis Team Plan doc in the Project):** Amr = ingestion/Parquet/physical load/simulated stream. Hatem = tactical & cognitive layer. Together = a **unified risk score** (physical + cognitive/tactical load).

**Other non-code work (separate chats):** ASRT "Mashrouy Bedayati" funding proposal (EGP 100,000, 8 months, one camera + two rotating STATSports GPS pods; draft due Oct 9, final Oct 31, 2026), a 5-minute presentation for Dr Sherif Elgohary (partner clubs WE and El Sekka), and the Barça Innovation Hub × Metrica "More Than A Hack 2027" hackathon (registration closes Nov 10, 2026).

---

## 3. Repository layout

```
2awasheet_gp/
├─ README.md                 Replay viewer overview, run/verify commands, workload method
├─ package.json              root scripts (see section 9); packageManager pnpm@11.19.0
├─ pnpm-workspace.yaml       packages: apps/*, packages/*; allowBuilds (esbuild, workerd true; @scarf/scarf false)
├─ apps/
│  ├─ web/                   React 18 + TS (strict) + Vite + Tailwind  → Replay viewer + Live API viewer + Hatem's analytics module
│  │  └─ src/
│  │     ├─ App.tsx, main.tsx, store.ts, types.ts, styles.css
│  │     ├─ components/      AnalyticsPanel, DropZone, EventTimeline, HighlightControls, LiveApi, PitchCanvas, WorkloadSettings
│  │     ├─ components/analytics/   ← HATEM'S MODULE (section 4.3)
│  │     ├─ live/, replay/, workers/   (Youssef: receiver worker, highlights, parsing workers)
│  ├─ api/                   TWO things share this folder:
│  │  ├─ *.py                ← HATEM'S Python FastAPI analytics backend (section 4.2)
│  │  └─ src/*.ts            Youssef's Node CV diagnostic server (cv-config, cv-service, cv-types, cv-validation, server.ts)
│  └─ cloudflare/            Cloudflare Worker + Durable Object (CvCoordinator) + Swagger docs
├─ packages/shared/          @replay/shared — TS parsing, coordinates, events, workload, playback, highlights, CV validation (+ unit tests)
├─ tests/e2e/                Playwright: replay.spec, live.spec, highlights.spec
├─ scripts/                  run-e2e, run-performance, scan-jsonl, smoke-cloudflare, smoke-cv-api
├─ docs/                     METRICS.md (Hatem), CLOUDFLARE_CV_API, RENDER_CV_API, CV_BACKEND_SECURITY, SPEC_CV_BACKEND_TEST, FRONTEND_LIVE_API, LOCAL_HIGHLIGHTS, event-validation, reference.png, PROJECT_CONTEXT (this file)
└─ data/                     LOCAL ONLY, git-ignored: tracking_3816.jsonl(.bz2), events_3816.json, event_3816.json, metadata_3816.json, roster_3816.json, interim/
```

Match data is **never committed** to Git (private PFF data).

---

## 4. Hatem's Advanced Football Analytics Engine

### 4.1 What it does

A FastAPI backend reads a PFF match (tracking JSONL + events JSON + metadata + roster), computes tactical metrics and per-player statistics, and a React module shows them in **Advanced Analytics** (top-nav toggle in `App.tsx`). Three tabs:

| Tab | What it shows |
|---|---|
| **Core metrics** (Tab1) | Pitch canvas playback of the match with Pitch Control, EPV surface, DAS, OBSO, VAEP/OBV action list, live segments; explanatory guide boxes |
| **Live player** (Tab3, NEW) | Pick any player and any minute (clock + slider + play 10×/30×/60×). Shows that player's stats *up to that moment*: passes & accuracy, shots on/off target, defending, goalkeeping, running, value; maps (now / heat / passes / shots / defending); ranking table within position group; whole-match chart + key moments; both-teams panel |
| **Player analytics** (Tab2) | Whole-match report per player: pressure/press panel, fatigue curve, chemistry, decision score, "In words" explanations and a Guide |

### 4.2 Backend files (`apps/api/`, Python)

| File | Role |
|---|---|
| `data_loader.py` | Loads PFF tracking/events/metadata/roster, maps player ids to a `slot`, normalises columns (events become 224 columns via `json_normalize`), dynamic per-match loading and upload |
| `metrics_engine.py` | Pitch Control (Laurie/Spearman model), EPV grid, DAS (accessible-space 2.1.0), OBSO (C-OBSO), VAEP = ΔEPV (+ `vaep_oa` above-average), OBV expected-outcome variant (pass model AUC 0.77), xG surface (distance/angle logistic), `PENALTY_XG = 0.76`, action table, possession side forward-fill, duel success from `challengeWinnerPlayerId` |
| `player_analytics.py` | `ANALYTICS_VERSION = "1.6"`. Whole-match per-player analytics: press (nearest opponent < `PRESS_RADIUS_M = 3.0`, AUC 0.86 vs provider label), fatigue (15-s samples of pitch-control area, live play only, ignores speed > 10 m/s), chemistry (≥ 3 passes per pair), decision score (1.0 if chosen action is max ΔEPV else 0.0; skipped when max ΔEPV ≤ 0) |
| `player_live.py` (NEW, ~800 lines) | `PhysicalIndex` (cumulative arrays at 1 s stride, 9-frame median filter, `SPEED_CAP = 10.0`, trailing 1-s mean for top speed, HSR/sprint runs) and `LivePlayers` (action table, `recent(idx)` 1-s look-back, `ball_at`, `table(idx)`, `detail(slot, idx)` with heat map/maps/summary, team context ranking, `timeline(slot)`). Self-test: `python player_live.py --match 3816` |
| `main.py` | FastAPI app + `Bundle` (engine + live + analytics) |
| `verify_live_stats.py` (NEW) | `python verify_live_stats.py --match 3816` — recounts every live stat from raw events (34 checks, all PASS) |
| `validate_metrics.py`, `reference_metrics.py`, `check_references.py` | Validation of the 6 core metrics against references (31 PASS / 5 WARN) |
| `requirements.txt` | fastapi, uvicorn[standard], numpy, pandas, scipy, orjson, python-multipart, accessible-space>=2.1.0 |

**Endpoints (`main.py`):**
`GET /api/health` · `GET /api/matches` · `POST /api/matches/upload` · `GET /api/match` · `GET /api/metrics/live-segments` · `GET /api/players` · `GET /api/metrics/frame/{frame_id}` · `GET /api/metrics/epv-timeline` · `GET /api/metrics/actions` · `GET /api/analytics/player/{player_id}` · `GET /api/analytics/status` · `POST /api/analytics/warmup` · **`GET /api/live/players`** (players table + team totals + ball) · **`GET /api/live/player/{player_id}`** (detail: heat, maps, summary) · **`GET /api/live/player/{player_id}/timeline`**.

### 4.3 Frontend files (`apps/web/src/components/analytics/`)

`AnalyticsView.tsx` (tabs: `type TabKey = 'core' | 'live' | 'players'`) · `AppShell.tsx` · `Tab1CoreMetrics.tsx` · `Tab2PlayerAnalytics.tsx` · `Tab3LivePlayer.tsx` · `AnalyticsPitch.tsx` (canvas pitch for Tab1) · `PlayerPitch.tsx` (SVG pitch for Tab3: modes now/heat/passes/shots/defending) · `api.ts` (types + fetch helpers + `clockLabel`) · `bridge.ts` (`announceMatchFile` — tells analytics which match file the replay loaded) · `metricGuide.tsx` · `ui.tsx` (shared UI bits).
Stack: React 18, TS strict, Tailwind with custom colours (`ink`, `panel`, `line`, `lime`, `muted`), `recharts`. Playback uses a frame cache, single-flight requests and gliding so it stays smooth.

### 4.4 Data facts and conventions (important for correctness)

- PFF tracking is **29.97 fps**, indexed by `frameNum`. Events carry `frame_idx`, `frame_num`, `ball_x/y`, `end_ball_x/y`, `side`, `actor_slot`, etc.
- **Ball is tracked in only ~47 % of frames.** Dead-ball frames → DAS/OBSO/EPV are blank there. Possession = side of the last on-ball event (forward-filled).
- Event codes were **inferred from the data** (no PFF documentation available):
  - Shot outcome: `G` goal (only real if the next restart is a kick-off — 3 Argentine "G" shots are disallowed goals), `S` saved, `B` blocked, `O` off target, `C`/`L` other. **On target = real goals + saved** (matches official 6 and 2 for ARG/KSA).
  - Pass outcome: `C` completed, `D` defended (→ `defenderPlayerId`), `B` blocked (→ `blockerPlayerId`), `O` out, `S` stopped.
  - Challenge (duel) types: `T` tackle, `A` aerial, `D` dribble, `B`, `FIFTY`; only the **winner** is named (`challengeWinnerPlayerId`), so losers are partly unknown.
  - Other fields: `linesBrokenType` (string length = lines broken), `keeperPlayerId`, `disciplinePlayerId`.
- Physical: median-filtered speed; > 10 m/s (36 km/h) treated as a tracking jump; HSR 5.5–7 m/s; sprint ≥ 7 m/s for ≥ 1 s; top speed = best 1-s trailing mean (still ~35 km/h for ~20 players → **compare players, do not quote as true top speed**); distance = integral of speed.
- Key pass = completed pass whose receiver shoots within 8 s. Progressive pass = ends ≥ 10 m closer to goal. Rankings are inside position groups (≥ 4 qualifying players, else all outfield); minimum 10 minutes played; ≥ 10 passes / ≥ 5 duels needed for percentage ranks.
- Coordinates: pitch centre = (0,0), metres. The maps draw "attacking left to right".

### 4.5 Verification done

- `verify_live_stats.py`: **34 checks, all PASS** (counts recomputed from raw events; rules; time; ranges). Earlier FAILs were test issues (NaN handling, kick-off pass at frame 0, GKs excluded from 5 km rule) — fixed.
- Metric validation: **31 PASS / 5 WARN** (see `docs/METRICS.md` section on validation).
- UI test: Tab3 rendered with real match data in a headless harness; TypeScript strict compiles.
- Pass-completion, shots on/off target, goals, and tackle/interception counts match the official totals for match 3816.

### 4.6 Known limits (be honest about these)

1. **Top speed inflated by tracking noise** (cap + 1-s mean helps, not perfect).
2. **Event semantics inferred**, no PFF dictionary.
3. **Duel losers partly unknown** (only winner named).
4. **Validated on one match only** (3816) — other matches are untested until their files are provided.
5. **VAEP and OBV are EPV-style approximations**, not the published socceraction/OBV models.
6. Ball visibility ≈ 47 % → many frames have no DAS/OBSO/EPV.
7. Last tracking frame is all-NaN → handled with a 1-s look-back.
8. Analytics warm-up for a match takes ~21 s the first time (cached afterwards).

---

## 5. Match Replay viewer (Youssef)

Client-only React app. Choose a tracking `.jsonl/.json`, optional metadata `.json`, optional events `.json`; **nothing leaves the browser**.
- Streams the file in a Web Worker (`file.stream()` + `TextDecoderStream`), packs raw + smoothed positions into typed arrays, renders with Canvas 2D.
- Player identity is always `team + jerseyNum`. Metadata controls team labels, kits, pitch size, halves.
- Features: playback speeds, frame/5-s/half navigation, interpolation, kit controls, side flip, selected-player trails, event timeline & filter, pass/shot overlays, roster identity, playhead-limited technical stats, live workload table, 5-minute workload charts, output-decline trends.
- **Workload method:** reject gaps > 200 ms and speeds > 12.5 m/s; HSR 5.5–7.0 m/s; sprint ≥ 7.0 m/s for ≥ 1 s; acceleration/deceleration ±2.5 m/s² held 0.5 s over a 0.5-s window; output trend = last valid 5 min vs previous 15 min, needs 80 % tracking quality. No ball-in-play normalisation (not reliably derivable).
- **Score validation:** raw `SH`+`G` count gives ARG 4–2 KSA but real score is 1–2; app shows **no score** when validation fails (see `docs/event-validation.md`). Hatem's engine solves this separately using the kick-off check.
- **Highlights** (`docs/LOCAL_HIGHLIGHTS.md`): event-based reel (shots, goal records, subs, cards), ±5 s windows, merged overlaps; goals labelled "Goal candidate".

## 6. CV diagnostic API + Live API viewer (Youssef)

- Purpose: the CV team (Marwan, Hoss) uploads tracking frames produced by their model; teammates inspect them with GET requests. It does **not** run CV or calculate workload.
- **Cloudflare deployment** (`apps/cloudflare`): Worker → one SQLite Durable Object `CvCoordinator` (fixed name `trusted-team-v1`, four sessions for one trusted team). Live URL base: `https://tracking-cv-api.youssefmansi2.workers.dev` (Swagger at `/docs`, OpenAPI at `/openapi.json`).
- **Alternative Node/Render** server (`apps/api/src/server.ts`, start `node apps/api/dist/server.mjs`, `/health`).
- Auth: shared **Bearer key** (never put it in code, chat, screenshots or URLs). Endpoints: `GET /health`, `POST /api/cv/sessions`, `POST /api/cv/sessions/{id}/frames` (envelope `{sequence, frames[]}`, 1–300 frames, ≤ 1.5 MiB), `GET .../latest`, `GET .../frames?limit=30`, `DELETE /api/cv/sessions/{id}`.
- Rules: `jerseyNum` mandatory per player; sequence starts at 0 and increments; same sequence+content for retries; 409 on conflicts; bounded buffer (300 frames / 1.5 MiB) — not an archive.
- **Live API viewer** (`LiveApi.tsx`, `apps/web/src/live`): enter 32-hex session ID + API key; polls GET frames every 250 ms–1 min (default 1 s); plays at source speed in a queue (max 3,600 unplayed frames), computes workload in a worker, "Waiting for next batch" when empty. Read-only; never creates/deletes sessions.
- Docs: `docs/CLOUDFLARE_CV_API.md`, `RENDER_CV_API.md`, `CV_BACKEND_SECURITY.md`, `SPEC_CV_BACKEND_TEST.md`, `FRONTEND_LIVE_API.md`. Known dependency-audit findings are deferred (review by 2026-10-25).

---

## 7. Data

- **PFF FC FIFA World Cup 2022** (shared Google Drive, 64 matches: event, tracking, metadata, rosters, players.csv). No wearable, injury or training data inside.
- Files for a match: `tracking_<id>.jsonl` (29.97 fps), `events_<id>.json`, `metadata_<id>.json`, `roster_<id>.json`. Pilot match **3816** (Argentina vs Saudi Arabia; first suggestion in the report was 3812).
- Local only in `C:\Users\lenovo\2awasheet_gp\data\` (git-ignored). Other open datasets considered: Metrica, StatsBomb.
- Hatem earlier built Colab notebooks (Metrica sample game 1 → PFF 3816 with kloppy, e.g. `GP_hatem_part_FULL`): Pitch Control, OBSO, EPV-added passes, DAS cross-check, cognitive proxies, 5-minute window tables.

---

## 8. How to run the app on Hatem's PC (Windows PowerShell)

**Terminal 1 — analytics API (Python):**
```powershell
cd C:\Users\lenovo\2awasheet_gp\apps\api
pip install -r requirements.txt
uvicorn main:app --port 8000
```
**Terminal 2 — web app:**
```powershell
cd C:\Users\lenovo\2awasheet_gp
pnpm install
pnpm dev
```
Open `http://localhost:5173`, choose the match files, then use the top toggle **Advanced Analytics** → tabs *Core metrics / Live player / Player analytics*. The first analytics request warms up (~20 s).

Checks: `python verify_live_stats.py --match 3816` and `python player_live.py --match 3816` (from `apps/api`). Frontend: `pnpm lint`, `pnpm typecheck`, `pnpm test`, `pnpm test:e2e`, `pnpm build`.

---

## 9. Root scripts (package.json)

`pnpm dev` (`cd apps/web && vite`) · `build` · `lint` · `typecheck` · `test` · `test:e2e` · `test:performance` · `build:api` · `start:api` · `test:api` · `dev:cloudflare` · `test:cloudflare` (and related smoke scripts).

---

## 10. Git and pnpm notes

- Working branch: **`feature/analytics-engine`**, pushed. Open the PR here: `https://github.com/youssefmansii/2awasheet_gp/pull/new/feature/analytics-engine`.
- Commits: `06eaa12` (live player analytics), merge `ff2b4e5` (origin/main merged; conflict in `apps/web/src/App.tsx` resolved by **keeping both sides' imports**: bridge `announceMatchFile`, HighlightControls, highlight helpers, LiveApi, AnalyticsPanel, WorkloadSettings, EventTimeline, PitchCanvas).
- Local-only changes on his PC that **must NOT be committed**: `pnpm-lock.yaml`, `pnpm-workspace.yaml` (workerd approval), untracked `repomix-output.xml`.
- Past problems and fixes: merge blocked by a dirty `pnpm-lock.yaml` → `git restore pnpm-lock.yaml`; `pnpm` not found → `npm install -g pnpm`; `ERR_PNPM_IGNORED_BUILDS` → `pnpm approve-builds` (approve `workerd`, `esbuild`; keep `@scarf/scarf` off).
- Commit-message attribution for Claude-made commits: end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

---

## 11. Tooling notes for the assistant (cloud session + device bridge)

- The cloud container cannot see his PC. Use `mcp__remote-devices__device_bash` (Linux VM with his folders mounted at `$HOME/mnt/2awasheet_gp`; ≤ 180 s per call; background processes die when a call ends — run long jobs synchronously). Python for the API is under `$HOME/mnt/2awasheet_gp/apps/api`.
- Deliver files with `device_commit_files` (stagedPath under `/mnt/user-data/outputs`, Windows devicePath); commits can race — re-commit with `force` and verify with `grep`. `device_stage_files` brings device files into the container.
- `Projects.project_write` needs a `local_path` inside `/home/claude`.
- Scratch tests (TypeScript harness, `$HOME/diag/*.py`, `data/interim/livetest/*.json`) are not part of the product.

---

## 12. Open decisions and next steps

1. **VAEP/OBV re-implementation (NOT approved yet).** Proposal: use `socceraction`/`kloppy` (open-source) for real VAEP and a closer OBV, instead of the EPV-style approximations. Hatem has not approved it, and has not said whether he/Amr can download other matches' event files. **Do not start until he approves.**
2. **Test other matches** (needs their tracking + event + metadata + roster files) to confirm the engine is truly dynamic.
3. Hatem should **test the running app** (section 8) and report anything that looks wrong; then open the PR for Youssef to merge.
4. Integrate with Amr's physical-load work into the **unified risk score**.
5. Planned later in the team plan: cognitive-fatigue proxies, causality/time-framing analysis, benchmark on Metrica/StatsBomb.
6. Reduce top-speed noise further (e.g., better tracking smoothing) if it matters for the risk score.

---

## 13. Glossary (plain words)

- **Pitch Control (PC):** the probability each team controls each spot on the pitch (who would reach the ball first).
- **EPV (Expected Possession Value):** how likely the possession is to end in a goal, depending on where the ball is.
- **DAS (Dangerous Accessible Space):** dangerous free space the team in possession can actually reach.
- **OBSO (Off-Ball Scoring Opportunity):** value of an off-ball player's position (can he receive and score?).
- **VAEP:** value of an action = change in goal chance caused by it (here: ΔEPV; `vaep_oa` = above average).
- **OBV:** on-ball value — the same idea using an expected-outcome model.
- **xG:** expected goals of a shot (distance/angle).
- **HSR:** high-speed running 5.5–7 m/s; **sprint:** ≥ 7 m/s for ≥ 1 s.
- **Key pass:** pass followed by a shot by the receiver within 8 s.
- **Progressive pass:** moves the ball ≥ 10 m closer to goal.
