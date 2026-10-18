# Cloudflare CV diagnostic API

## Scope and architecture

The backend-only Cloudflare deployment is an alternative to the Node/Render server. It does not deploy the frontend, run the CV model or calculate workload. The frontend now has a read-only Live API viewer described in [FRONTEND_LIVE_API.md](FRONTEND_LIVE_API.md); workload is calculated in its browser worker from received frames. Private tracking files and metadata remain out of Git and the deployment bundle.

A Worker forwards protected HTTP requests to one SQLite-backed Durable Object (`CvCoordinator`, binding `CV_COORDINATOR`). The fixed coordinator name is `trusted-team-v1`: four sessions for one trusted team, not a multi-tenant service. Global request/auth limits, session counts and sequence state are coordinated across Worker locations. A regular Worker global variable would not provide this consistency.

The API uses the existing shared-key Bearer authentication and frame validator. `jerseyNum` is mandatory for each included player in every raw/smoothed team array. `trackId` is optional metadata, never a replacement. Unknown players must be omitted, not assigned invented jerseys. Coordinates are pitch metres, not pixels.

## HTTP contract

Use HTTPS. Every tracking API endpoint except `/health` needs `Authorization: Bearer <CV_API_KEY>`. Documentation and its vendored assets are public; they contain no key, live session IDs or tracking captures.

### Interactive Swagger documentation

Open [Swagger UI](https://tracking-cv-api.youssefmansi2.workers.dev/docs). The [OpenAPI 3.1 specification](https://tracking-cv-api.youssefmansi2.workers.dev/openapi.json) can also be imported into Postman or other OpenAPI clients. The maintained source is `apps/cloudflare/src/openapi.ts`; it describes all six operations, schemas, request/response examples, error statuses, sequence rules and retention.

1. Click **Authorize** and enter only your private key value, without `Bearer `.
2. Expand `POST /api/cv/sessions`, select **Try it out**, then **Execute**. Send no body.
3. Copy the returned session ID into the frame endpoints. Upload a batch with sequence zero, then read latest/recent frames. Use a new increasing sequence for each new batch; reuse the same sequence/content for retries.
4. Delete your test session when finished, and **Logout** in the authorization dialog or close/reload the page.

Try it out performs real operations against the same API origin, including deletion. Swagger request examples/cURL snippets can display the key you enter: do not screenshot or share those authenticated snippets. Authorization is held only in page memory and not persisted across reloads. The API key, shared-key access model and frame validation remain unchanged. Explicit frontend origins can now GET session frames/latest; cross-origin writes remain blocked.

Swagger UI 5.33.3 assets are pinned, copied locally by the package's dev/build/deploy commands, and served from `/docs-assets/`; no CDN, external validator, URL-config overrides or telemetry scripts are used by the page. The package's install scripts, including its Scarf dependency's install-time telemetry, remain disabled. The docs-only CSP allows same-origin scripts/connections and inline styles required by Swagger; API JSON retains its strict `default-src 'none'` policy. Generated `apps/cloudflare/public/` is ignored by Git. Its contents are a build artifact, not a place to add match files. Only three known asset paths are served publicly.

`pnpm test:cloudflare` also opens an isolated headless browser, verifies the six operations, Authorize/Try it out against a random local test key, no external HTTP requests/console errors, and cleared authorization on reload. Chrome DevTools MCP is not configured here; Playwright provides the browser check. A key-free screenshot is saved in ignored `test-results/swagger-docs.png`.

| Method | Path | Body/result |
|---|---|---|
| GET | `/health` | Public readiness/configuration check; `storage: "durable-sqlite"`. Not a database read probe. |
| POST | `/api/cv/sessions` | No body; returns `sessionId`, timestamps, zero counts. Creation is **not idempotent**. |
| POST | `/api/cv/sessions/{id}/frames` | `application/json` envelope `{ "sequence": 0, "frames": [...] }`; returns receipt. |
| GET | `/api/cv/sessions/{id}/latest` | Latest original frame under `frame`, metadata and warnings. |
| GET | `/api/cv/sessions/{id}/frames?limit=30` | 1–90 retained frames, oldest first. |
| DELETE | `/api/cv/sessions/{id}` | Deletes the session; subsequent reads return 404. |

The detailed frame contract and client examples in [RENDER_CV_API.md](RENDER_CV_API.md) also apply; replace the base URL with the deployed Worker URL. No gzip or JSONL-file uploads on this diagnostic interface: send a JSON batch, not multipart or one request per player. The existing local file/chunk simulator is unchanged.

Batch size is **configurable by the producer**, not fixed to five. Send 1–120 frames per request and no more than 1 MiB of UTF-8 JSON, including extra fields. Every frame contains all available players. `frameNum` and `videoTimeMs` must strictly increase within/across new batches. Start `sequence` at zero and increment only after acceptance. Retry uncertain uploads with the exact same sequence/content; the last 32 receipts are retained. JSON property order is part of the parsed-envelope hash; whitespace is not. A different payload with an accepted sequence, skipped sequence, overlap or changed match ID returns 409. Concurrent uploads can return `UPLOAD_IN_PROGRESS`; retry the same batch later.

Responses retain original provider fields and return compatibility warnings for missing match clock/player quality. Missing/invalid jerseys reject the entire batch (422). Malformed JSON is 400; bad media type/encoding 415; oversized bodies 413; expired/missing sessions 404; auth 401; cross-origin browser calls 403; limits 429 with `Retry-After`. No permissive CORS, secrets in URLs, accounts or public frame reads.

## Retention and privacy

Unlike the Node server's memory-only buffer, the Cloudflare adapter stores bounded session snapshots in SQLite, so recent frames and retry receipts survive restarts/eviction. Each session retains at most **90 frames AND 1 MiB serialized frame data**, plus metadata and 32 receipt hashes/responses. This is a live test buffer, not a match archive.

Sessions expire after **15 minutes without a newly accepted batch**, or **two hours after creation**, whichever is first. Reads and duplicate retries never extend expiry. Expiry is checked before reads and again at upload commit; an alarm removes expired rows approximately once a minute while sessions remain. Alarm execution can be delayed by the platform. DELETE removes the active row immediately; Cloudflare may retain platform-managed recovery copies under its retention policies. Do not promise immediate physical erasure from backups. Do not send sensitive player data without consent and a cloud-storage retention/access review.

Writes use parameterized SQL and synchronous transactions. Upload bodies are read with a 1 MiB cap and 10-second deadline, then validated before reloading/committing state. Deleted/expired sessions cannot be resurrected by an in-flight upload. No source captures, payloads or keys are intentionally logged. Wrangler observability ingestion is disabled by default; use platform metrics and scoped live tail for diagnostics without adding payload logging.

The shared key gives all holders read/write/delete access to known sessions. Limits remain 1,200 non-health requests/minute and 30 failed-auth attempts/minute, globally per coordinator. Rate counters survive eviction. Attack traffic can still consume provider quotas; this is not an API gateway or a production privacy/security certification.

## Free-plan budget

At 25 FPS, five-frame batches mean five POSTs/second: **36,000 uploads in two hours**. One viewer GETting once per second adds 7,200: **43,200 Worker requests** plus retries/session operations. Each protected request also invokes the Durable Object, charged against its separate allowance, not as a second public Worker request.

The implementation normally writes one limiter row per protected request and one session row per newly accepted upload. That example is approximately **79,200 SQL row writes**, plus initial schema/session/alarm/deletion overhead. SQLite index accounting and provider metering can add overhead: check actual usage before increasing viewers, test duration or frame rate. GET polling and repeated retries are not free. Expiry alarms add requests and row writes too.

Published free allowances at implementation time: Workers 100,000 requests/day and 10 ms CPU per invocation; Durable Objects 100,000 requests/day, 13,000 GB-s/day; SQLite 5 million rows read/day, 100,000 rows written/day, 5 GB total storage. These are account-wide/shared with other applications and can change. Reaching one allowance can stop service even if another is available. Request count alone does not guarantee a free or low-latency workload. The front Worker only forwards requests; JSON processing runs inside the Durable Object. Cloud CPU/latency still require real deployed measurement. Do not upgrade to paid without owner approval.

Official references: [Workers limits](https://developers.cloudflare.com/workers/platform/limits/), [Durable Objects pricing](https://developers.cloudflare.com/durable-objects/platform/pricing/), [SQLite transactions](https://developers.cloudflare.com/durable-objects/api/sqlite-storage-api/), [Durable Objects setup](https://developers.cloudflare.com/durable-objects/get-started/).

## Local checks

From the repository root, using the declared pnpm version:

```powershell
npx --yes --ignore-scripts pnpm@11.19.0 install --frozen-lockfile --ignore-scripts
pnpm test
pnpm typecheck
pnpm lint
pnpm test:cloudflare
pnpm build:cloudflare
```

The runtime smoke test launches local workerd, uses random test-only credentials and temporary SQLite storage, verifies POST/GET, validation, flexible batches, concurrent retry handling, restart persistence, retention, deletion and limits, then stops its child process. No cloud credentials or private captures are required. `build:cloudflare` is a dry run and uploads nothing.

For interactive local development, create ignored `apps/cloudflare/.dev.vars` with a privately generated `CV_API_KEY` (32–256 printable non-space characters), then `pnpm dev:cloudflare`. Do not commit that file or put a real key into Wrangler `vars`.

## Deploy

1. Use a Cloudflare account with Workers Free and SQLite Durable Objects available. Keep the frontend and any Render service separate.
2. Authenticate privately with Wrangler. Prefer deployment-only scopes:

```powershell
node apps/cloudflare/node_modules/wrangler/bin/wrangler.js login --scopes account:read user:read workers:write workers_scripts:write workers_tail:read
node apps/cloudflare/node_modules/wrangler/bin/wrangler.js whoami
```

3. Confirm the intended account, service name (`tracking-cv-api`) and that no unrelated Worker will be overwritten. For multiple accounts, select the intended account explicitly with the supported Wrangler account setting; never guess.
4. Set the secret using `wrangler secret put CV_API_KEY --config apps/cloudflare/wrangler.jsonc`; paste a privately generated random key into its prompt, not chat or a screenshot. If the Worker does not yet exist, follow the CLI's creation prompt only for the confirmed target. Share the key securely with the authorized CV team; Cloudflare does not return plaintext secrets later.
5. Run `pnpm deploy:cloudflare` (this also prepares the self-hosted Swagger assets). The committed configuration creates the SQLite class with a `v1` migration. Do not delete or rename the class/migration to retry a deploy. On a new Cloudflare account, open Workers & Pages and complete the free `workers.dev` subdomain setup before the first deployment.
6. Verify the actual returned `https://...workers.dev/health` URL, then authenticated session creation, upload and GET round-trip. Remove test sessions. A dry run or printed URL alone is not a successful deployment.

Never set real keys via command-line `--var` or checked-in configuration. `.dev.vars*`, `.env*`, `.wrangler/`, dist, JSONL and compressed captures are ignored. Deployment bundles only the Worker entry point and its imports, not the repo or samples.

If production tests fail, stop producers and inspect status/error codes with payload-free logs. Roll back to a compatible **Cloudflare** Worker version using the dashboard, preserving the Durable Object binding and SQL schema. Do not route back to an unauthenticated legacy simulator. A Worker rollback does not roll back SQLite data or restore deleted sessions. For a compromised key, rotate the secret and redeploy before sharing a replacement.

## Architecture decision (2026-10-18)

Use Worker + SQLite Durable Object rather than a plain Worker, Workers KV or a file server: the prototype needs immediately consistent sequence/retry state and bounded recent-frame reads from different clients. KV propagation is not an atomic upload coordinator; Worker memory can disappear or differ by location. A single coordinator makes four-session caps and rate limits enforceable without separate infrastructure. Trade-offs are a single-region coordinator hop, cloud persistence of a short buffer, SQL write quotas and no horizontal scaling within this trusted-team prototype. Keep the original Node/Render backend available for local diagnostics; do not delete it during this migration.

Workspace audit still reports the existing 3 critical, 2 high and 6 moderate tooling findings documented in [CV_BACKEND_SECURITY.md](CV_BACKEND_SECURITY.md). No newly reported Cloudflare-package advisory was introduced at implementation time. New CLI/runtime tools are pinned and installed with dependency scripts disabled. Existing audit deferrals remain due for review by 2026-10-25; they are not a clean-audit claim.
