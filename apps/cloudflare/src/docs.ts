import { json } from './http';
import { openApi } from './openapi';

const initializer = `window.addEventListener('load', () => {
  window.ui = SwaggerUIBundle({
    url: '/openapi.json', dom_id: '#swagger-ui',
    deepLinking: true, displayRequestDuration: true,
    persistAuthorization: false, queryConfigEnabled: false, validatorUrl: null,
    tryItOutEnabled: false,
    requestInterceptor: (request) => {
      const target = new URL(request.url, window.location.origin);
      if (target.origin !== window.location.origin) throw new Error('Only same-origin API requests are allowed.');
      request.redirect = 'error';
      return request;
    }
  });
});`;
const html = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tracking Studio — API Documentation</title>
<link rel="icon" href="data:,">
<link rel="stylesheet" href="/docs-assets/swagger-ui.css">
</head><body>
<div id="swagger-ui"></div>
<noscript>Enable JavaScript to use Swagger UI. The API specification is available at <a href="/openapi.json">/openapi.json</a>.</noscript>
<script src="/docs-assets/swagger-ui-bundle.js" defer></script>
<script src="/docs/swagger-initializer.js" defer></script>
</body></html>`;

export function documentationResponse(path: string): Response | null {
  if (path === '/openapi.json') return json(200, openApi);
  if (!['/docs', '/docs/', '/docs/swagger-initializer.js'].includes(path)) return null;
  const headers = new Headers(json(200, {}).headers);
  headers.set('Content-Type', path.endsWith('.js') ? 'application/javascript; charset=utf-8' : 'text/html; charset=utf-8');
  // Swagger needs inline styles, never inline/eval scripts. API responses retain their stricter CSP.
  headers.set('Content-Security-Policy', "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'");
  return new Response(path.endsWith('.js') ? initializer : html, { headers });
}
