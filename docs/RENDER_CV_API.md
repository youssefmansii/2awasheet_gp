# Deploy and test the backend-only CV API

This service accepts tracking-coordinate JSON and lets your team inspect it with ordinary GET requests. It does not run the CV model, host the frontend, calculate CV workload, or persist a match to a database. It is for one trusted test team sharing a key.

## 1. Publish code before clicking Deploy

The API source must exist on the GitHub branch selected in Render. The initial public repository commit does not contain this new API. Local implementation is not a deployment. Publish only reviewed source/build/configuration files after authorization; do not include real tracking files, notebooks with private outputs, `.env` files or old local `master` history.

The supplied `render.yaml` can create a backend-only Blueprint. Alternatively use **New > Web Service**, connect the repository and enter:

| Render field | Value |
|---|---|
| Repository | `youssefmansii/2awasheet_gp` |
| Branch | The branch containing the reviewed CV API implementation |
| Language/runtime | Node |
| Root Directory | Leave blank (repository root, not `apps/api`) |
| Instance Type | Free |
| Build Command | See command below |
| Start Command | `node apps/api/dist/server.mjs` |
| Health Check Path | `/health` |
| Environment: `NODE_VERSION` | `24` (Render resolves the current patch in that major) |
| Environment: `NODE_ENV` | `production` |
| Environment: `CV_API_KEY` | Privately generated random key, 32–256 printable non-space characters |

Build command:

```sh
npx --yes --ignore-scripts pnpm@11.19.0 install --frozen-lockfile --prod=false --ignore-scripts && npx --yes --ignore-scripts pnpm@11.19.0 run build:api
```

This pins the repository's package-manager version, preserves the lockfile, installs the build tools even in production mode, and disables dependency lifecycle scripts. It avoids relying on a particular Corepack cache or preinstalled pnpm version. esbuild's platform binary is provided by its optional platform dependency; the clean-install build verifies this path. The service starts only the standalone Node bundle, not Vite/Vitest or frontend tooling.

Render assigns `PORT`; do not override it unless necessary. The backend binds to `0.0.0.0`. TLS is terminated by Render. For a manually created Web Service, generate a random API key in a trusted local terminal or password manager, then paste it privately into Render:

```sh
node -e "console.log(require('node:crypto').randomBytes(32).toString('hex'))"
```

Do not send the key in chat, source code, query parameters, screenshots or notebook output. If using the Blueprint, Render generates the secret; copy it privately from the service environment to the authorized team. After changing a key, redeploy/restart so the process uses it.

Official deployment references: [Render Web Services](https://render.com/docs/web-services), [Node version selection](https://render.com/docs/node-version), [free-tier limits](https://render.com/docs/free).

## 2. Verify public health

After Render reports a successful deploy, use the actual HTTPS URL it provides. The examples below intentionally use a placeholder hostname:

```text
GET https://YOUR-SERVICE.onrender.com/health
```

Expected response:

```json
{"status":"ok","service":"cv-diagnostic-api","storage":"memory"}
```

Health is public; it returns no frames, session IDs or credentials. Other endpoints require the key. Opening the API root in a browser is not a dashboard and is not an authenticated readback test. An authenticated Python/Postman/PowerShell request is required for frame reads.

## 3. Endpoints and authorization

For every CV request:

```http
Authorization: Bearer YOUR_PRIVATE_CV_API_KEY
```

| Method/path | Request | Response |
|---|---|---|
| `POST /api/cv/sessions` | No body | `201`, `sessionId`, creation/expiration and zero counts |
| `POST /api/cv/sessions/{id}/frames` | JSON `{ "sequence": 0, "frames": [...] }` | `200`, acceptance receipt, counts and compatibility warnings |
| `GET /api/cv/sessions/{id}/latest` | No body | Last accepted original frame under `frame`, receipt time and session metadata |
| `GET /api/cv/sessions/{id}/frames?limit=30` | `limit` from 1 to 300 | Recent retained frames, oldest first |
| `DELETE /api/cv/sessions/{id}` | No body | Deletion acknowledgment; subsequent reads return 404 |

The API key is supplied by the owner, **not returned by session creation**. The server returns the session ID. Share the session ID between producer and viewer so both inspect the same data. All key holders are trusted peers with access to all known sessions; there are no individual user roles yet.

Creation is not idempotent: retrying it can create a second empty session if the original response was lost. Delete unwanted sessions or let them expire. Frame uploads are idempotent for the last 32 accepted sequences.

### Frame batch contract

- Content type: `application/json`; no multipart file upload, surrounding JSONL text, gzip or `Content-Encoding: gzip` on this CV endpoint.
- Body: a non-negative integer `sequence` and 1–300 complete consecutive frames. Begin with sequence 0.
- Required frame fields: `frameNum`, `videoTimeMs`, `homePlayers`, `awayPlayers`.
- `frameNum`: non-negative 32-bit integer. Frame numbers and source timestamps must strictly increase inside/across newly accepted batches. They do not reset at upload boundaries.
- `videoTimeMs`: finite, non-negative source milliseconds, not upload time. Preserve measurement intervals even if inference is slow.
- Optional `period` is 0, 1 or 2; optional `periodGameClockTime` is non-negative match-clock seconds; optional `gameRefId` is a bounded string or non-negative safe integer. A supplied match ID cannot change during the session.
- Every included player needs finite pitch-metre `x` and `y` and a **mandatory recognized `jerseyNum`**, either a number 1–255 or its decimal string (prefer `"10"`, matching the replay format). Missing, null, empty, zero or invalid jerseys reject the whole batch with 422, even when `trackId` is provided. Omit a player whose number is not reliably known; never invent a jersey or substitute a tracking ID. Apply this rule to raw and smoothed arrays alike.
- Identity is **team + jersey number**, not `trackId`. Optional `trackId` can be retained as CV metadata; when supplied it must be a non-negative safe integer or non-empty string up to 128 characters. The same jersey on opposing teams is allowed, but duplicates within one team/array are rejected.
- Optional player quality categories: confidence `HIGH`, `MEDIUM`, `LOW`; visibility `VISIBLE`, `ESTIMATED`. Omit unavailable players rather than sending null coordinates or stale invented observations.
- Optional smoothed player arrays use the same rules. Empty smoothed arrays are preserved, not automatically replaced with raw positions.
- `balls` is omitted, empty or contains one finite x/y ball observation. Optional ball z is a non-negative height. Optional ball visibility: `VISIBLE`, `ESTIMATED`, `NOT_VISIBLE`. `ballsSmoothed` is an object or null, never an array.
- Duplicate track IDs or jersey numbers within one team/array are rejected. At most 15 entries per team array.
- Extra provider fields are preserved as JSON, including nested objects, but the entire request is limited to 1.5 MiB, 64 nesting levels and finite numbers. Keeping a field does not mean the app uses it analytically. Send only data you are authorized to share.

The endpoint does not verify pitch calibration, real player identity, actual physical accuracy or whether smoothing used future frames. Those are CV responsibilities to test separately.

### Acceptance versus dashboard compatibility

Readback preserves the supplied recognized jersey number, optional `trackId` metadata and other original fields. Jersey identity matches the current replay app, but readback alone does not implement online CV-to-dashboard integration or certify coordinate accuracy.

Possible warnings for the latest accepted batch:

| Warning code | Meaning |
|---|---|
| `MISSING_MATCH_CLOCK` | Match-clock mapping is still needed for meaningful live time/trend windows |
| `MISSING_PLAYER_QUALITY` | Confidence or visibility is missing; do not assume measured tracking accuracy |

Before the first upload, `/latest` returns `frame: null`, `receivedFrames: 0`, `lastSequence: null`, `receivedAt: null`. These are successful empty reads, not server failures.

## 4. CV team: Python create/upload/read example

Set `CV_API_URL` and `CV_API_KEY` using environment variables or notebook secrets. Do not put real keys in shared notebook cells. `requests` is a client-side prerequisite, not a dependency of this backend.

```python
import os
import requests

base = os.environ["CV_API_URL"].rstrip("/")
http = requests.Session()
http.headers["Authorization"] = f"Bearer {os.environ['CV_API_KEY']}"

# No body on creation. A cold free instance may take roughly a minute to wake.
response = http.post(f"{base}/api/cv/sessions", timeout=90)
response.raise_for_status()
session_id = response.json()["sessionId"]

# Illustrative frame only; replace with the actual output from the CV loop.
frames = [{
    "frameNum": 100,
    "videoTimeMs": 4000,
    "period": 1,
    "periodGameClockTime": 4,
    "homePlayers": [{
        "trackId": 42, "jerseyNum": "10",
        "x": -12.3, "y": 8.5,
        "confidence": "HIGH", "visibility": "VISIBLE"
    }],
    "awayPlayers": [],
    "balls": []
}]
batch = {"sequence": 0, "frames": frames}
response = http.post(
    f"{base}/api/cv/sessions/{session_id}/frames",
    json=batch, timeout=30
)
response.raise_for_status()
receipt = response.json()
assert receipt["sequence"] == batch["sequence"]

response = http.get(f"{base}/api/cv/sessions/{session_id}/latest", timeout=30)
response.raise_for_status()
latest = response.json()
assert latest["frame"] == frames[-1]
print(session_id, latest["receivedFrames"], latest["warnings"])
```

Increment sequence only after acceptance. On a network timeout/uncertain frame-upload response, retry the same batch with the same sequence and key ordering. Hashes use the parsed JSON serialization: whitespace changes are harmless, but changing fields or property order is not a safe retry. A duplicate returns the original receipt with `duplicate: true`; counts in that receipt describe the original acceptance, while GET reports current session counts.

Do not retry permanent 400/401/403/404/409/413/415/422 failures blindly. Fix the key/schema/order or create a new session after expiry. On 429, honor `Retry-After`; after a long pause the session may have expired. On a transient network/5xx failure, retry a bounded number of times with backoff using the identical batch, not a new sequence. Retrying a sequence older than the receipt window returns 409, never double-counts.

For live CV later, use a bounded background sender for approximately 200–500 ms batches. That publisher is not added by this backend change; do not block the inference loop or accumulate an unbounded unsent queue. The diagnostic endpoint accepts immediately; it does not deliberately delay frames to source-time playback.

## 5. Owner: PowerShell GET inspection

Use your real deployed URL and the session ID returned to the CV team. Obtain the secret through your private environment or an interactive prompt; do not hard-code it in a shared script.

```powershell
$cvBaseUrl = 'https://YOUR-SERVICE.onrender.com'
$cvSessionId = 'SESSION_ID_FROM_THE_CV_TEAM'
$cvHeaders = @{ Authorization = "Bearer $env:CV_API_KEY" }
Invoke-RestMethod -Uri "$cvBaseUrl/api/cv/sessions/$cvSessionId/latest" -Headers $cvHeaders |
    ConvertTo-Json -Depth 64

# Optional recent tail, not whole-match history:
Invoke-RestMethod -Uri "$cvBaseUrl/api/cv/sessions/$cvSessionId/frames?limit=30" -Headers $cvHeaders |
    ConvertTo-Json -Depth 64
```

Polling every 0.5–1 second is sufficient for manual testing. This is not yet the web app's SSE integration. No browser CORS is enabled; server-to-server Python, Postman and PowerShell work without it. A future hosted frontend needs an explicitly reviewed origin policy.

## 6. Limits, retention and errors

- At most four sessions; each expires two hours after creation or 15 minutes after the last newly accepted upload, whichever is earlier. Reads and duplicate retries do not extend idle lifetime.
- Keep at most 300 recent frames **and** 1.5 MiB of serialized frame content per session; older frames are discarded when either cap is exceeded. `receivedFrames` is the total accepted count, not the retained count.
- Keep 32 acceptance receipts per session. Original receipt metadata is not a durable checkpoint.
- One in-flight upload per session. A second gets 409; deletion/expiry during upload prevents acceptance.
- A body has a ten-second absolute read deadline. The standalone HTTP server also has header/request limits.
- Single-process global limit: 1,200 non-health requests/minute; failed authentication is limited to 30/minute before returning 429. These limits are for a trusted prototype, not distributed infrastructure. Health remains public and lightweight for Render checks.
- JSON responses never contain the API key. Logs do not intentionally include payloads, headers or query strings. Do not add verbose proxy/request logging that leaks those values.
- Data is memory-only and lost on restart, sleep/redeploy, deletion or expiry. No database, backups, recovery, production accounts or workload accumulation is claimed.

Errors have one consistent shape:

```json
{"error":{"code":"UNAUTHORIZED","message":"A valid Bearer API key is required."}}
```

Status meanings: 400 malformed JSON/query/body usage; 401 missing/wrong key; 403 denied browser origin; 404 unknown/expired session; 405 wrong method; 408 stalled body; 409 sequence/overlap/match/concurrency conflict; 413 body too large; 415 wrong media/encoding; 422 invalid batch/frame/value; 429 rate/session limit; 500 generic internal failure. Failed validation does not partially append frames.

## 7. Local verification and operational checks

For the standalone CV API:

```powershell
cd E:\2awashet
# Set CV_API_KEY privately in this shell first; it must be 32+ random printable characters.
pnpm build:api
pnpm start:api
```

Use `http://localhost:3001` in place of the deployed base URL. The key is required locally too; `.env.example` is not automatically loaded by this command. The local frontend's existing replay/simulation remains available with `pnpm dev`; that Vite development service is separate and does not expose this CV API. The standalone CV deployment deliberately does not expose `/api/live` simulation routes.

After building, `pnpm test:api` starts the actual compiled API briefly on a free local port with a fresh temporary test key, verifies POST/GET/auth/retry/delete, then stops its own child process. It does not contact a cloud service or need private data.

Before sharing the public URL:

1. Confirm health returns 200 and matches the expected service.
2. Confirm an unauthenticated session create/read attempt fails with 401.
3. Create a session with the key, POST a small consented frame batch, and GET it unchanged.
4. Retry the same sequence and verify `duplicate: true` with no increased count.
5. POST malformed/out-of-order data and verify a structured rejection.
6. Delete the session; verify reads return 404.
7. Check Render logs and CPU/memory. Do not treat the absence of logs as proof of successful uploads; verify receipts and readback.

If unexpected errors or disclosure occur, suspend the service first, rotate the key if exposed, inspect non-sensitive error codes and redeploy the last reviewed working revision using Render's deployment controls. Restart/rollback discards memory-only sessions; communicate that loss and recreate IDs. Do not revert to an older standalone revision that publicly exposes unauthenticated simulation creation.

Review known test/build dependency audit findings in `docs/CV_BACKEND_SECURITY.md` before public rollout. This is a limited diagnostic deployment, not a claim of audited production readiness.
