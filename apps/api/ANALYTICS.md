# Local Python analytics companion

This is separate from the Node/Cloudflare CV ingestion API. Merging the analytics
branch does not deploy Python, change the live CV endpoints, or connect the new
analytics panels to a CV session.

## Run on Windows (from the repository root)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r apps/api/requirements-test.txt
.\.venv\Scripts\python.exe -m uvicorn main:app --app-dir apps/api --host 127.0.0.1 --port 8000 --no-proxy-headers
```

In a second terminal, run `pnpm dev`. Open Advanced Analytics. Selecting files in
Match Replay only announces their availability: nothing is uploaded until you
press **Send selected files** in Advanced Analytics. The confirmation identifies
the destination and files; a changed tracking, event or metadata selection needs
new approval, even if the Analytics view was opened before and is now hidden.

The default destination is `http://localhost:8000`. `VITE_ANALYTICS_API_URL` is a
build-time destination override, not authentication or permission to expose the
Python service publicly. Do not put API keys in Vite environment variables.

The API stores approved copies in ignored `<repo>/data/` (override with
`MATCH_DATA_DIR`). It caches derived data in `data/interim/`. These copies have
no automatic expiry. Keep this directory local and remove copies/caches through
your file manager when no longer needed. Local Replay and Cloudflare Live API
continue to use their existing, separate data paths.

## Boundary and limits

- This is a **local-only, single-user tool**, not a multi-user hosting service.
  Peer address and Host must be loopback; remote peers, non-local browser origins
  and forwarding headers receive 403. Do not bind publicly, tunnel, reverse-proxy
  or deploy this companion. It has no Internet authentication/authorization.
- Total HTTP request body limit: **1 GiB (1,073,741,824 bytes)**, including multipart
  overhead. Both Content-Length and streamed body bytes are checked; oversize
  requests receive 413. This accommodates the approximately 850 MB capture but
  does not promise constant-memory analytics. Derived full-match arrays consume
  substantial RAM. Cloudflare batch limits are unchanged.
- `POST /api/matches/upload` takes multipart `tracking`, `events`, `metadata` and
  optional numeric `match_id`. Metadata-only updates preserve existing tracking.
  New File objects are not assumed identical because their sizes match.
- Swagger for this local companion: `http://127.0.0.1:8000/docs`.
  Cloudflare's existing Swagger and bearer key are unrelated.

## Frame-time statistics and limitations

- A pass is credited as a key pass / assist (and receives xA) only once the
  associated shot's frame is reached. Earlier tables and pass maps do not expose
  those future labels. Seeking backward recalculates the visible prefix without
  mutating the cached whole-match events.
- UTF-8 BOM-prefixed JSONL preserves its first frame with both Python JSON
  decoders. The owner's real sample contains this encoding prefix.
- High-speed and sprint run counts update at the first frame satisfying the
  configured one-second minimum, not at the end of an ongoing run. A halftime
  boundary always splits runs.
- The new **Live player** panel is retrospective playback over an uploaded match,
  not the Cloudflare real-time feed. It opens at full time. Its whole-match chart
  intentionally includes the complete recording. Offline spline/Butterworth
  processing, fitted models and whole-match value baselines remain retrospective;
  these are not causal online estimates, medical fatigue measurements, or proof
  of CV accuracy. The existing manager workload monitor/Live API are unchanged.
- EPV-derived VAEP/OBV and fatigue/output interpretations retain the limitations
  described in [METRICS](../../docs/METRICS.md). Full-match validation numbers in
  that document are the branch author's historical results, not new verification
  of this merge against the owner's entire capture.

## Verification commands

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s apps/api/tests -v
pnpm install --frozen-lockfile --ignore-scripts
pnpm typecheck
pnpm lint
pnpm test
pnpm test:e2e
pnpm build
```

`pnpm test:analytics` runs the Python regressions using Python on PATH. Browser
regressions use isolated Playwright, not the user's browser profile. Contract
event fixtures are labeled as such; no private tracking or secrets are committed.

Runtime JavaScript audit: no known vulnerabilities reported during this fix.
The full workspace audit still reports **3 critical, 2 high, 6 moderate** in
existing tooling. See [the existing audit disposition](../../docs/CV_BACKEND_SECURITY.md).
Do not expose test/dev servers or describe this as a clean overall security audit.

Implementation references: [Starlette pure ASGI middleware](https://www.starlette.io/middleware/)
for inspecting incoming body chunks, and [Uvicorn settings](https://www.uvicorn.org/settings/)
for loopback binding and disabling trusted proxy headers.
