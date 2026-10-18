import { defineConfig } from '../web/node_modules/vite';

export default defineConfig({
  build: {
    ssr: 'apps/api/src/server.ts', outDir: 'apps/api/dist', target: 'node22',
    rollupOptions: { output: { entryFileNames: 'server.mjs' } }
  }
});
