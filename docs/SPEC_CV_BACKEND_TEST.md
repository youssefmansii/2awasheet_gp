# Specification: backend-only CV upload/readback test on Render

Status: implemented and verified locally after user approval. Public deployment/publication remains a separate step.

## Objective

Give the computer-vision team a public HTTPS API where they can POST tracking-frame batches. Give the project owner ordinary GET endpoints to inspect the accepted data before connecting it to the web dashboard.

This is a diagnostic integration prototype, not a deployed service yet. It does not require the user's computer to run after deployment. The CV model runs separately; this service receives tracking JSON, not video/images or model weights.

## Scope and assumptions

- Backend only, using the existing Node.js/TypeScript project.
- Render Web Service, Free instance for initial testing; no paid resources provisioned automatically.
- Temporary in-memory storage with explicit caps and expiration. Restart, sleep/redeploy or expiry loses uploaded data.
- One trusted project team shares a secret API key configured privately in Render. This is not multi-tenant account authentication.
- All CV session creation, upload, read and deletion endpoints require `Authorization: Bearer <CV_API_KEY>`.
- No frontend changes, database, object storage, source fusion or fatigue-model changes.
- No new CV SSE/WebSocket implementation in this first increment. Ordinary GET is the requested inspection mechanism; existing local replay/simulation behavior must remain intact.
- Require a recognized jersey number for every included player, matching the user's clarified requirement. Identity is team + jersey number. Preserve optional `trackId` only as metadata, never as an identity substitute. Reject missing/null jerseys atomically; CV must omit unknown-number players rather than invent numbers.
- Do not expose the older unauthenticated live-session creation route on the public diagnostic deployment. Keep local simulation available separately.

## Proposed API contract

These paths are implemented on the standalone local CV server; no public URL has been deployed yet. JSON successes and structured errors use `Content-Type: application/json` and `Cache-Control: no-store`. Exact usage and limits are documented in `docs/RENDER_CV_API.md`.

| Method/path | Authentication | Purpose |
|---|---|---|
| `GET /health` | Public | Small liveness/readiness result; no payloads, keys or session listing |
| `POST /api/cv/sessions` | API key | Create an isolated temporary session; return `sessionId`, creation time and expiration |
| `POST /api/cv/sessions/:sessionId/frames` | API key | Atomically validate and accept a timestamp-ordered frame batch |
| `GET /api/cv/sessions/:sessionId/latest` | API key | Return the latest accepted frame, sequence/count metadata and compatibility warnings |
| `GET /api/cv/sessions/:sessionId/frames?limit=30` | API key | Return a bounded recent tail, oldest first, not an unlimited archive |
| `DELETE /api/cv/sessions/:sessionId` | API key | Discard the session and all its retained data |

### Upload envelope

```typescript
interface CvFrameBatch {
  sequence: number; // integer, starts at zero, increments by one per accepted batch
  frames: CvTrackingFrame[];
}
```

`CvTrackingFrame` follows the discussed tracking shape:

- Required: non-negative integer `frameNum`, non-negative finite `videoTimeMs`, `homePlayers` array, `awayPlayers` array.
- Optional fields retained and validated when supplied: `gameRefId`, `period`, `periodGameClockTime`, `homePlayersSmoothed`, `awayPlayersSmoothed`, `balls`, `ballsSmoothed`.
- Players: finite pitch-metre `x` and `y`; mandatory `jerseyNum` as a number 1–255 or its decimal string. Missing, null, zero and invalid jersey numbers are rejected, even with `trackId`. No fabricated numbers or coordinates. This applies to both raw and smoothed arrays.
- Preserve bounded additional provider fields as data. Never execute embedded content or fetch URLs in it.
- Missing ball is `[]`; absent players are omitted from the arrays. Smoothed arrays are optional, not fabricated placeholders.
- `videoTimeMs` and `periodGameClockTime` are different clocks: source milliseconds versus match-clock seconds.
- Coordinates are pitch metres, normally centred at the pitch centre; verify source calibration separately. Shape validation cannot certify physical accuracy.

First accepted batch is sequence zero. Timestamps strictly increase inside a batch and across newly accepted batches. A repeated recent sequence with the same content returns the prior acceptance without duplicating frames; different content for that sequence or skipped sequences returns 409.

### Readback semantics

- Return the original accepted frame fields, not just rendered coordinates or packed typed arrays.
- Include session ID, last sequence, received frame count and server receipt time.
- `latest` is the last accepted frame, not a completed whole-match result or a paced simulation snapshot.
- Before any successful upload, return a documented empty result with `frame: null` and zero counts.
- Warnings distinguish missing match clock and absent quality information. Unknown jerseys are validation errors, not warnings. Acceptance is not a claim that all downstream analytics are ready or accurate.
- No workload calculations are performed on this diagnostic CV route. Existing workload processing remains a separate implementation.
- Clients retry the identical batch after an uncertain response. The server retains the last 32 sequence receipts; retries outside that window are rejected rather than double-counted.
- Session IDs are server-generated, validated opaque identifiers. An expired/unknown session returns 404 and must be recreated with a new ID and sequence zero.

## Initial resource and safety limits

- JSON requests only for the new frame endpoint, at most 1 MiB per request and 120 frames per batch.
- At most 15 players per team per raw/smoothed array; reject duplicate identities within a team/frame.
- Retain at most 90 recent frames per session plus 32 small receipt records; enforce a retained-byte cap as well as a frame-count cap.
- Maximum four active sessions; two-hour hard lifetime and 15-minute idle expiration, cleaned up during service operation.
- Bound request read time and concurrent uploads; validation failures must not partially mutate the session.
- Single-process global request limits and stricter failed-authentication limits, sized for small batches and GET testing. Document limits and return 429 with retry guidance. Multi-instance scaling is explicitly out of scope.
- No permissive CORS. Python requests, curl and Postman do not require browser CORS. Cross-origin frontend integration will need a separately reviewed allowlist.
- No API-key query parameters for these JSON routes. Keys stay in headers and private environment variables, not example source or logs.
- Security headers on responses; generic server errors; do not log frame bodies or sensitive provider fields.
- Require a strong configured key and fail startup in public/production mode if it is absent. No default production key.
- Use consented/non-sensitive test captures and minimum necessary player identifiers; no full videos, private match files or personal names need to be committed.

## Render deployment requirements

- Service type: **Web Service**, language: **Node**, plan: **Free** for this prototype.
- Root directory: repository root, because build/shared packages are in the monorepo.
- Bind to `0.0.0.0` and the assigned `PORT`, not only `127.0.0.1`.
- Preserve the pnpm workspace and authoritative `pnpm-lock.yaml`; use a reproducible frozen install and the pinned package-manager version. Verify the actual supported Render build/bootstrap command during implementation.
- Build the backend bundle with `pnpm build:api`; start with `node apps/api/dist/server.mjs`.
- Set the supported Node version, production mode and a private `CV_API_KEY`; configure `/health` as the health-check path.
- Do not run Vite's development server publicly or deploy the frontend in this increment.
- The public API URL only exists after a successful Render deploy. Do not invent a final service hostname.
- Free Render services have cold starts and can restart. Temporary memory storage is suitable for a diagnostic exchange, not durable match history or uninterrupted production service.

Official references used for deployment verification:

- https://render.com/docs/web-services
- https://render.com/docs/node-version
- https://render.com/docs/free

## Commands and project structure

Existing repository commands, to verify again after changes:

```powershell
cd E:\2awashet
pnpm lint
pnpm typecheck
pnpm test
pnpm build:api
pnpm test:e2e
```

Expected implementation locations:

- `apps/api/src/`: new CV validation/session service and production server routing/configuration.
- `packages/shared/src/`: integration tests using the existing Vitest setup, without committing a private fixture.
- `apps/api/README.md` or a linked deployment guide: exact Render settings and Python/PowerShell POST/GET examples.
- `.env.example`: placeholder names only, preserving unrelated entries.
- Optional Render configuration file: backend-only settings with generated/private secrets, never literal key material.

Code style: existing TypeScript conventions, two-space indentation, camelCase fields and explicit interfaces. Use additive changes rather than rewriting the replay pipeline. No new runtime dependency is expected for this small HTTP service.

## Implementation and testing sequence

1. Write failing API tests for authentication, create/POST/GET flow and faithful readback of valid jersey identities with optional CV `trackId` metadata. Reject unknown jerseys in every raw/smoothed team array.
2. Implement the bounded diagnostic service and make that slice pass.
3. Add negative tests for invalid JSON/fields, duplicate identities, sequence retries/conflicts, overlaps, resource limits, expiry and atomic rejection.
4. Add production binding/key checks and health route; verify the standalone bundle through an actual localhost POST/GET smoke test.
5. Run existing lint, type checks, unit/API tests and browser regression tests. Run the package manager's native dependency audit and triage relevant findings before recommending public launch; no forced upgrades.
6. Document exact Render fields and producer/viewer requests. Review the publication file list for private data and unrelated work.
7. After publication/deploy authorization and successful hosting, verify public health, authenticated POST/GET, unauthorized rejection and non-sensitive round-trip values.

The existing working tree has uncommitted live-simulation work and unrelated user files. Preserve it. Use a `codex/` branch if a branch is created; stage exact reviewed files rather than `git add .`. Publishing the deployable backend and required shared/build files to GitHub is a separate externally visible action to confirm before execution.

## Boundaries

- Always: preserve replay behavior and unrelated changes, validate incoming data, protect reads as well as writes, bound storage, document acceptance versus compatibility, test each increment.
- Ask first: publishing source to GitHub, starting paid hosting, adding a database, broadening the endpoint's access model, changing browser CORS, introducing new dependencies or integrating the frontend.
- Never: commit private tracking JSON/JSONL, real secrets or generated captures; push local `master` history; expose arbitrary data publicly; claim deployment before a real URL is verified; silently invent jersey identities, coordinates or match clock.

## Success criteria

1. The CV team can create a session and POST a small frame batch using Python requests and a private API key.
2. The owner can GET the same frame fields, including recognized jersey and optional `trackId` metadata, with counts/timing/warnings. A valid track ID cannot bypass the required jersey number.
3. Missing/wrong keys cannot read, upload, create or delete CV sessions.
4. Invalid or repeated/conflicting batches behave as documented without partial writes or double counting.
5. Limits, expiry and memory-only behavior are explicit and tested.
6. Existing local replay and simulation tests still pass.
7. The backend builds independently and is ready for Render's assigned port/host requirements.
8. Once the user completes/authorizes deployment, the public endpoint passes the agreed POST/GET smoke test. Until then report it as deployment-ready, not deployed.

## Implementation evidence

Implemented: authenticated temporary sessions, validated JSON-batch ingestion, original-frame latest/tail readback, retry receipts, expiry/resource/request limits, Render-compatible standalone startup, public health, deployment configuration and client guide. No frontend integration or database was added.

Verified: 61 unit/API tests passed (including 30 CV/config tests), four replay/simulation browser tests passed, lint/type checks passed, standalone bundle built, pinned pnpm 11.19.0 clean frozen install/build passed in a temporary directory with dependency scripts disabled, and the compiled server passed a real localhost HTTP smoke test. An intentionally inverted auth condition caused the focused test to fail and was immediately restored before final verification.

Subsequent user clarification made jerseys mandatory. The revised contract passed 97 unit/API tests (including 66 CV/config tests), lint, type checks, a rebuilt standalone bundle and compiled-server smoke test including null-jersey rejection. New tests first reproduced acceptance of missing/null jerseys in all four team arrays before the validator was tightened. No frontend code or hosting was changed by this clarification.

The workspace dependency audit is not clean; existing tooling-only findings and their limited-runtime reachability/deferral are recorded in `docs/CV_BACKEND_SECURITY.md`. No forced upgrades were performed. This is readiness for a limited diagnostic test, not a public production-security certification.

No source was pushed and no Render service was created. Publication requires the user's authorization and a reviewed exact file list; deployed public POST/GET verification is still pending.
