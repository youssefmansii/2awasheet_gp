/* global console */
import { mkdir, copyFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, URL } from 'node:url';

const require = createRequire(import.meta.url);
const source = dirname(require.resolve('swagger-ui-dist/package.json'));
const target = fileURLToPath(new URL('../public/docs-assets/', import.meta.url));
await mkdir(target, { recursive: true });
// Copy only vendored documentation assets, never captures, .env files or the frontend.
for (const name of ['swagger-ui-bundle.js', 'swagger-ui.css', 'LICENSE']) {
  await copyFile(resolve(source, name), resolve(target, name));
}
console.info('Prepared pinned, self-hosted Swagger UI assets.');
