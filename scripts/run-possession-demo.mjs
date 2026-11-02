/* global process, console, fetch, setTimeout */
import { spawn } from 'node:child_process';
import { existsSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const python = path.join(root, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
const children = [];
let closing = false;
function stop(code = 0) {
  if (closing) return;
  closing = true;
  for (const child of children) child.kill();
  process.exitCode = code;
}
process.on('SIGINT', () => stop());
process.on('SIGTERM', () => stop());
function run(command, args, cwd, env = process.env) {
  const child = spawn(command, args, { cwd, env, stdio: 'inherit', windowsHide: true });
  children.push(child);
  child.on('error', (error) => { console.error(error.message); stop(1); });
  child.on('exit', (code) => { if (!closing) stop(code ?? 1); });
}
async function responding(url) {
  try { return await fetch(url, { signal: AbortSignal.timeout(900) }); } catch { return null; }
}

try {
  if (!existsSync(python)) throw new Error('The local Python environment is missing. Follow docs/POSSESSION_DEMO.md to set it up.');
  const spool = path.join(root, 'output', 'possession-demo', 'multipart');
  mkdirSync(spool, { recursive: true });
  const api = await responding('http://127.0.0.1:8010/api/possession/health');
  if (!api?.ok) run(python, ['-B', '-m', 'uvicorn', 'possession_demo:app', '--app-dir', 'apps/api', '--host', '127.0.0.1', '--port', '8010', '--no-proxy-headers'], root, { ...process.env, TEMP: spool, TMP: spool });
  const web = await responding('http://127.0.0.1:5173');
  if (!web?.ok) run(process.execPath, ['node_modules/vite/bin/vite.js', '--host', '127.0.0.1', '--port', '5173', '--strictPort'], path.join(root, 'apps/web'));
  console.log('\nPossession demo: http://127.0.0.1:5173/?view=possession');
  console.log('Tracking stays on this computer. Press Ctrl+C to stop services started by this command.\n');
} catch (error) {
  console.error(error.message);
  stop(1);
}
