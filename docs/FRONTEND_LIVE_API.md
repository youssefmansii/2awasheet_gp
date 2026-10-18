# Frontend: whole-file replay or live CV API

The app has two top-level choices:

- **Local replay**: select the whole tracking JSONL file (plus optional metadata/events). The existing parser reads it as a stream in a browser worker; the file is not uploaded to Cloudflare. Replay controls, historical stats and settings remain available.
- **Live API**: a read-only viewer of the Cloudflare session produced by the CV team. This replaces the old simulator choice in the UI. Legacy simulator source is retained, not used by this mode.

## Run and connect

From the repo root, run `pnpm dev`, open `http://localhost:5173/` or `http://127.0.0.1:5173/`, then select **Live API**.

1. The CV team creates a session and uploads batches using the [Swagger API contract](https://tracking-cv-api.youssefmansi2.workers.dev/docs).
2. Enter that session's 32-character lowercase hex **Session ID**.
3. Enter the shared **API key without Bearer**. Do not put it in source, screenshots, frontend build variables or URLs.
4. Choose the GET interval (default one second; 250 ms–5 seconds) and click **Connect live API**.
5. Click **Stop receiving** to terminate the receiver worker. This never deletes the producer session. Switching to Local replay also terminates reception. Reconnecting starts a fresh observed-statistics accumulator and requires re-entering the key.

The receiver calls only `GET https://tracking-cv-api.youssefmansi2.workers.dev/api/cv/sessions/{id}/frames?limit=90` with `Authorization: Bearer <key>`. Browser preflight OPTIONS is automatic. It never creates/deletes sessions or POSTs files/frames. The API URL is fixed in the client, not user-supplied, so a mistyped endpoint cannot receive the key. Redirect following is disabled, browser credentials/cookies are omitted, and responses bypass caches.

The key is cleared from the input after connection and retained only by the active receiver worker. No localStorage/sessionStorage persistence. All key holders remain equally trusted peers; this is not per-user authentication and does not make the shared key suitable for embedding in a public app.

## Frames and workload

Network reads, response validation, parsing and incremental workload calculation run in a dedicated Web Worker. Responses are capped at 2 MiB and requests time out after 10 seconds. Snapshots must match the requested session and the frame contract, including mandatory jerseys. Frames are deduplicated by increasing source frame/time and accepted-frame counters before updating statistics.

The pitch uses a 90-frame rolling window; the display follows the latest received source frame, not the computer's wall clock. It updates with polling, not a 60 Hz animated or lossless server stream. The current API's buffer may hold less than 90 frames due to the byte cap. **One second is the delay after a completed request, so network time adds to the polling cycle.** Batch upload cadence also contributes to latency.

Statistics cover **only received observations**, including the server's recent buffer when joining, then newly received frames. They are not server-computed/full-match totals. Historical workload checkpoints accumulate incrementally; raw match frames do not accumulate without bound. Repeated polls do not double-count. Known missing frames break movement continuity rather than inventing distance across the gap. If joining late or a network gap outlasts the server buffer, a partial-history warning and missing-frame count are shown. Missing frame counts come from the API's accepted-frame counter, not assumptions that source frameNum values are consecutive.

Standard workload thresholds apply in Live API mode. Calculations use source videoTimeMs and the supplied period/match clock, never upload time. Metadata selected in Local replay is reused for teams, kit colors and pitch size; defaults are HOME/AWAY and 105 × 68 m. No event/roster feed is read live. Technical event stats therefore remain unavailable in this mode. If any received frame lacks period 1/2 or periodGameClockTime, positions still render but workload/trend display is withheld instead of inventing a clock; correct the producer fields and reconnect. Missing confidence/visibility produces a warning; percentages cannot certify reliability without those fields. Output-decline indicators are not a medical fatigue diagnosis.

The backend does not pace uploaded source frames. If a producer posts a prerecorded match faster than real time, the viewer follows whatever has been accepted, not a simulated clock. Pace the producer to simulate live capture.

## Connection behavior

Empty sessions show Waiting for new frames. A last producer upload older than five seconds is labelled waiting rather than appearing freshly live. 401/403/404 and malformed/inconsistent data stop reception with a generic, actionable message. Network/server failures retry with bounded exponential backoff. 429 respects Retry-After (capped at 60 seconds). No concurrent polling requests or catch-up burst; Stop/unmount terminates the worker and ignores obsolete callbacks.

## Browser access configuration

`CV_ALLOWED_ORIGINS` on the Cloudflare Worker is an exact comma-separated allowlist. The committed non-secret default enables only `http://localhost:5173` and `http://127.0.0.1:5173`, and only session `frames`/`latest` GET reads. Preflight permits Authorization/Content-Type and caches permission for 600 seconds. Returned errors also carry the allowed origin so the UI can explain rejected keys/expired sessions. There is no wildcard or credentialed CORS and no cross-origin session creation, uploads or deletion. Same-origin Swagger and non-browser CV clients keep working.

When deploying the frontend elsewhere, the owner must add its exact HTTPS origin and redeploy the backend. Do not add arbitrary origins automatically. Another local port such as 5174 is not allowed by default. Frontend hosting is separate: deploying the backend's CORS change does not host the frontend.

At the default interval, one viewer makes roughly 7,200 GETs in two hours (slightly fewer once network time is included), plus preflights/retries. Faster polling and additional viewers consume both Worker and Durable Object quotas and SQL row writes. The producer's batch size remains independent and flexible.

## Verification and rollout

`pnpm test` covers overlapping snapshots, frame/response validation, late join, missing history, gap-safe workload and read-only CORS policy. `pnpm test:e2e` covers whole-file replay plus browser live reception, auth errors, deduplication, Stop, missing clocks and layouts at 320/768/1024/1440 pixels. Simulator UI cases were migrated to the requested API mode; its parser/server unit coverage is retained. Chrome DevTools MCP is unavailable; isolated Playwright browsers provide runtime checks. `pnpm test:cloudflare` exercises actual local Worker preflight/GET restrictions and preserves Swagger/API regressions.

Roll back the frontend separately if needed. Backend rollback must preserve the secret and SQLite Durable Object schema; returning to a version without these CORS rules will block cross-origin frontend reception but not same-origin Swagger. Never enable the legacy unauthenticated simulation API publicly as a fallback.
