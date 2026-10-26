/* global fetch, process, setTimeout */
import { spawn } from 'node:child_process';

const node = process.execPath;
const server = spawn(node, ['node_modules/vite/bin/vite.js', '--host', '127.0.0.1'], {
  cwd: 'apps/web',
  // Tests mock the default API address; ignore any local VITE_CV_API_BASE override.
  env: { ...process.env, VITE_CV_API_BASE: 'https://tracking-cv-api.youssefmansi2.workers.dev' },
  stdio: 'inherit'
});

let testExit = 1;
try {
  const deadline = Date.now() + 30_000;
  while (Date.now() < deadline) {
    try {
      const response = await fetch('http://127.0.0.1:5173');
      if (response.ok) break;
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 200));
    }
  }
  if (Date.now() >= deadline) throw new Error('Vite did not start within 30 seconds.');

  const tests = spawn(node, ['node_modules/@playwright/test/cli.js', 'test'], { stdio: 'inherit' });
  testExit = await new Promise((resolve) => tests.once('exit', (code) => resolve(code ?? 1)));
} finally {
  server.kill();
}

process.exit(testExit);
