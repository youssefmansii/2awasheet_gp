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

The receiver calls only `GET https://tracking-cv-api.youssefmansi2.workers.dev/api/cv/sessions/{id}/frames?limit=300` with `Authorization: Bearer <key>`. Browser preflight OPTIONS is automatic. It never creates/deletes sessions or POSTs files/frames. The API URL is fixed in the client, not user-supplied, so a mistyped endpoint cannot receive the key. Redirect following is disabled, browser credentials/cookies are omitted, and responses bypass caches.

The key is cleared from the input after connection and retained only by the active receiver worker. No localStorage/sessionStorage persistence. All key holders remain equally trusted peers; this is not per-user authentication and does not make the shared key suitable for embedding in a public app.

## Frames and workload

Network reads, response validation, parsing and incremental workload calculation run in a dedicated Web Worker. Responses are capped at 2 MiB and requests time out after 10 seconds. Snapshots must match the requested session and the frame contract, including mandatory jerseys. Frames are deduplicated by increasing source frame/time and accepted-frame counters before updating statistics.

Received frames enter a bounded FIFO playback queue, separate from network polling. A 16 ms worker timer advances by original `videoTimeMs` differences: 300 frames at 30 FPS provide approximately ten seconds of playback instead of jumping immediately to the last position. Frames are rendered at source cadence (not artificial 60 Hz interpolation). The pitch retains only 90 already-played frames for drawing/current-speed calculation. The browser holds at most 3,600 unplayed parsed frames (about two minutes at 30 FPS); overflow stops with an explicit error rather than silently dropping queued movement. Historical workload checkpoints accumulate separately.

The API accepts 1–300 frames per batch and up to **1.5 MiB (1,572,864 bytes)** of uncompressed JSON. It retains at most 300 recent frames AND 1.5 MiB of serialized frame content, plus bounded metadata/receipts. This leaves headroom beneath the Cloudflare SQLite 2 MB row limit for the current single-row session snapshot. The real sample's first 300 frames fit these limits; arbitrary larger provider fields may reduce retained history. This is a bounded delivery buffer, not a full-match archive. Connect the viewer before uploading successive batches, and leave at least one polling cycle between uploads to avoid overwriting unseen data.

When the queue empties, playback holds the last position, shows **Waiting for next batch**, and resumes automatically from the next received frame. Waiting for a slow CV model does not invent movement or elapsed match time. **Pause playback** freezes playback/statistics while polling continues; **Resume playback** continues at source speed. Stop/switch modes terminates both playback and networking; it never deletes the producer session. Source gaps longer than one second and known missing-history boundaries are skipped rather than animating nonexistent frames. Browser timer suspension does not fast-forward through the whole queued batch on wake-up.

**The GET interval controls receiving, not playback speed.** It is a delay after a completed request, so network time adds to the polling cycle. A one-second interval is a reasonable starting point for receiving delayed CV batches; a producer can upload a processed segment together, and the viewer plays it at its original speed.

Statistics cover **only played observations**, not frames still queued for future playback, and are not server-computed/full-match totals. The worker adds every played frame to the workload accumulator; checkpoints are sent approximately once per source second and when the queue drains. Growing analytics timelines are not copied with each pitch frame. Receipt, played and queued counts are separate. Repeated polls do not double-count. Known missing frames break movement continuity when playback reaches the gap, without discarding already-queued frames. If joining late or a network gap outlasts the server buffer, a partial-history warning and missing-frame count are shown. Missing frame counts come from the API's accepted-frame counter, not assumptions that source frameNum values are consecutive.

Standard workload thresholds apply in Live API mode. Calculations use source videoTimeMs and the supplied period/match clock, never upload time. Metadata selected in Local replay is reused for teams, kit colors and pitch size; defaults are HOME/AWAY and 105 × 68 m. No event/roster feed is read live. Technical event stats therefore remain unavailable in this mode. If any received frame lacks period 1/2 or periodGameClockTime, positions still render but workload/trend display is withheld instead of inventing a clock; correct the producer fields and reconnect. Missing confidence/visibility produces a warning; percentages cannot certify reliability without those fields. Output-decline indicators are not a medical fatigue diagnosis.

For a demo, process footage in advance and upload a segment; playback uses source time independently of processing/upload duration. If CV processing is slower than playback, the viewer will eventually run out and wait. This is explicitly labelled buffered/delayed playback, not real-time CV inference. Uploading a whole match faster than the viewer can receive it still loses server history; a durable archive/cursor delivery protocol is a separate feature.

## Try the real ten-second sample

Create a fresh session in Swagger, connect the viewer with its ID/key, then run this PowerShell command. Paste the resulting clipboard into Swagger's POST frames body for that session. This is a real upload; use authorized sample data and do not share authenticated cURL snippets or screenshots.

```powershell
$frames = @(Get-Content -LiteralPath 'E:\2awashet\samples\3816-10s.jsonl' -TotalCount 300 |
    ForEach-Object { $_ | ConvertFrom-Json })
@{ sequence = 0; frames = $frames } | ConvertTo-Json -Depth 64 -Compress | Set-Clipboard
```

This uses sequence 0 only for a fresh session. The first 300 real sample frames occupy about 1.33 MB uncompressed and span approximately ten seconds; size varies with included fields and JSON serialization. Subsequent batches must use increasing sequence/frame/time values. The viewer starts at the first available frame, plays the segment, then waits. Sessions still expire after 15 minutes without a new accepted upload or two hours since creation.

## Connection behavior

Empty sessions show **Waiting for next batch**. Nonempty queues show **Playing buffered frames**, even when CV uploads are minutes apart; the last producer upload time remains visible. 401/403/404 and malformed/inconsistent data stop reception with a generic, actionable message. Network/server failures retry with bounded exponential backoff; already-buffered frames can keep playing. 429 respects Retry-After (capped at 60 seconds). No concurrent polling requests or catch-up burst; Stop/unmount terminates the worker and ignores obsolete callbacks.

## Browser access configuration

`CV_ALLOWED_ORIGINS` on the Cloudflare Worker is an exact comma-separated allowlist. The committed non-secret default enables only `http://localhost:5173` and `http://127.0.0.1:5173`, and only session `frames`/`latest` GET reads. Preflight permits Authorization/Content-Type and caches permission for 600 seconds. Returned errors also carry the allowed origin so the UI can explain rejected keys/expired sessions. There is no wildcard or credentialed CORS and no cross-origin session creation, uploads or deletion. Same-origin Swagger and non-browser CV clients keep working.

When deploying the frontend elsewhere, the owner must add its exact HTTPS origin and redeploy the backend. Do not add arbitrary origins automatically. Another local port such as 5174 is not allowed by default. Frontend hosting is separate: deploying the backend's CORS change does not host the frontend.

At the default interval, one viewer makes roughly 7,200 GETs in two hours (slightly fewer once network time is included), plus preflights/retries. Faster polling and additional viewers consume both Worker and Durable Object quotas and SQL row writes. The producer's batch size remains independent and flexible.

## Verification and rollout

`pnpm test` covers source-paced playback, underrun/resume, overlapping snapshots, bounded queue overflow, frame/response validation, late join, missing history, gap-safe workload and read-only CORS policy. `pnpm test:e2e` covers a real-time ten-second 300-frame playback, pause/resume, waiting/next-batch continuation, whole-file replay, auth errors, deduplication, Stop, missing clocks and layouts at 320/768/1024/1440 pixels. Simulator parser/server unit coverage is retained. Chrome DevTools MCP is unavailable; isolated Playwright browsers provide runtime checks. `pnpm test:cloudflare` exercises actual local Worker preflight/GET restrictions, 300-frame upload/retention, rejection of 301 frames/oversized bodies, and Swagger/API regressions.

Roll back the frontend separately if needed. Backend rollback must preserve the secret and SQLite Durable Object schema; returning to a version without these CORS rules will block cross-origin frontend reception but not same-origin Swagger. Never enable the legacy unauthenticated simulation API publicly as a fallback.
