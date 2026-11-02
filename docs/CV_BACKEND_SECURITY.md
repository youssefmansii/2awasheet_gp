# CV diagnostic backend: security boundary and audit notes

Scope: one trusted project team uses a shared environment key to exchange consented tracking JSON. The HTTP interface is untrusted. Assets are that key, uploaded tracking observations and service availability. There are no accounts, filesystem uploads, remote URL fetches, database queries, model execution or embedded instruction execution.

Controls: header-only bearer authentication on creation/read/write/delete, timing-safe digest comparison, fail-closed startup key validation, no browser CORS allowlist expansion, bounded JSON/body read time, frame/identity/sequence validation, synchronous atomic commit after all validation, resource caps, expiry, single-process rate/auth limits, no payload/key logging, JSON security headers and no-store responses. The public health route contains no user data. The older unauthenticated simulation API is not routed by this standalone service.

Limitations: shared key holders are equally trusted; no independent per-user authorization, durable recovery, encryption-at-rest service, cluster/distributed limiter, API gateway or guaranteed uptime. Do not use this prototype for sensitive production player data without the relevant consent, access and retention review. No security control can certify CV coordinate accuracy or identity consistency.

## Existing dependency audit findings

`pnpm audit --json` reported **3 critical, 2 high and 6 moderate** advisories in the existing workspace dependency graph during implementation. Dependencies and lockfile were not automatically upgraded. A security upgrade/migration needs separate compatibility testing, not a forced audit fix.

| Component/path | Reported severity/advisories | Reachability for this deployment |
|---|---|---|
| Vitest 2.1.9 | Critical `GHSA-5xrq-8626-4rwp`; moderate `GHSA-82fw-gwwq-j7x9` (also mocker) | Test UI/mocking interfaces; neither runs in the standalone API. Never expose a Vitest/UI server. |
| Tinypool 1.1.1 via Vitest | Critical `GHSA-5gmw-xhrv-c9v3`, `GHSA-85c8-ppgw-ccpr` | Test-worker option/prototype-pollution paths; no Vitest/tinypool in the runtime bundle. Test only trusted repository code/configuration. |
| Vite 5.4.21 via Vitest | High `GHSA-fx2h-pf6j-xcff`; moderate `GHSA-4w7w-66w2-5vf9`, `GHSA-v6wh-96g9-6wx3` | Development-server and Windows editor/filesystem paths; no public Vite server is started. Backend build uses existing Vite 6.4.3 as build tooling. |
| esbuild 0.21.5 via test Vite | Moderate `GHSA-67mh-4wv8-2f99` | Development-server path; not the standalone Node HTTP runtime. |
| braces 3.0.3 via Tailwind | High `GHSA-vfj7-8cjw-p6xm` | Frontend glob-pattern tooling, not invoked by CV server or API build entry. No patched version was reported by the audit. |
| postcss-selector-parser 6.1.4 via Tailwind | Moderate `GHSA-rj75-hqrm-r3gf` | Frontend selector-processing path, not used by the deployed API. |

Disposition: **deferred workspace toolchain remediation, not a clean audit**. The standalone bundle imports only built-in `node:http`/`node:crypto` plus bundled CV source and does not import packages from `node_modules` at runtime. The Render start command must remain `node apps/api/dist/server.mjs`, never a development/test server. Build/install steps still require trusted repository content; isolated builds and disabled dependency scripts reduce but do not eliminate supply-chain risk.

Follow-up owner: project maintainer; review before any broader public rollout or by **2026-10-25**, whichever comes first. Re-run the audit at deployment time and inspect runtime imports after changes. If a new finding is reachable from the standalone API or build with untrusted inputs, reassess before deployment instead of inheriting this deferral.
