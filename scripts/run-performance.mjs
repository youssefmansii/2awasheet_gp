/* global process */
import { spawnSync } from 'node:child_process';

const result = spawnSync(process.execPath, [
  'packages/shared/node_modules/vitest/vitest.mjs',
  'run',
  'packages/shared/src/performance.test.ts'
], {
  cwd: process.cwd(),
  env: { ...process.env, RUN_PERFORMANCE: '1' },
  stdio: 'inherit'
});

process.exit(result.status ?? 1);
