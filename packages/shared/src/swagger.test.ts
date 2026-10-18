import { describe, expect, it } from 'vitest';
import { openApi } from '../../../apps/cloudflare/src/openapi';
import { documentationResponse } from '../../../apps/cloudflare/src/docs';
import { validateCvBatch } from '../../../apps/api/src/cv-validation';

describe('Swagger documentation contract', () => {
  it('documents all six API operations with same-origin servers and Bearer security', () => {
    expect(openApi.openapi).toBe('3.1.0');
    expect(openApi.servers).toEqual([{ url: '/', description: 'This API instance (same origin)' }]);
    expect(openApi.security).toEqual([{ bearerAuth: [] }]);
    expect(openApi.paths['/health'].get.security).toEqual([]);
    expect(openApi.paths['/api/cv/sessions'].post).not.toHaveProperty('requestBody');
    const operations = Object.values(openApi.paths).flatMap((path) => Object.keys(path).filter((k) => ['get', 'post', 'delete'].includes(k)));
    expect(operations).toHaveLength(6);
  });
  it('uses valid upload examples and preserves mandatory jersey and variable-batch rules', () => {
    const example = openApi.components.schemas.Batch.examples[0];
    expect(() => validateCvBatch(example)).not.toThrow();
    expect(openApi.components.schemas.Player.required).toContain('jerseyNum');
    expect(openApi.components.schemas.Batch.properties.frames.minItems).toBe(1);
    expect(openApi.components.schemas.Batch.properties.frames.maxItems).toBe(120);
  });
  it('resolves every local schema reference and gives every operation response examples', () => {
    const visit = (value: unknown) => {
      if (!value || typeof value !== 'object') return;
      for (const [key, child] of Object.entries(value)) {
        if (key === '$ref') {
          const segments = String(child).split('/').slice(1);
          let current: unknown = openApi;
          for (const segment of segments) current = (current as Record<string, unknown>)[segment];
          expect(current, String(child)).toBeDefined();
        } else visit(child);
      }
    };
    visit(openApi);
    for (const path of Object.values(openApi.paths)) {
      for (const operation of Object.values(path)) {
        for (const response of Object.values(operation.responses)) {
          expect(response).toHaveProperty(['content', 'application/json', 'example']);
        }
      }
    }
  });
  it('serves public HTML with a docs-only CSP and no embedded key or third-party scripts', async () => {
    const docs = documentationResponse('/docs')!;
    expect(docs.status).toBe(200);
    expect(docs.headers.get('Content-Type')).toContain('text/html');
    expect(docs.headers.get('Content-Security-Policy')).toContain("script-src 'self'");
    expect(docs.headers.get('Content-Security-Policy')).not.toContain('unsafe-eval');
    const html = await docs.text();
    expect(html).toContain('/docs-assets/swagger-ui-bundle.js');
    expect(html).not.toMatch(/<script[^>]+src=["']https?:/);
    expect(html).not.toContain('CV_API_KEY=');
    expect(documentationResponse('/docs/')).not.toBeNull();
    expect(documentationResponse('/api/cv/sessions')).toBeNull();
  });
  it('disables validator, query overrides and auth persistence, and restricts requests to same origin', async () => {
    const script = await documentationResponse('/docs/swagger-initializer.js')!.text();
    expect(script).toContain('validatorUrl: null');
    expect(script).toContain('persistAuthorization: false');
    expect(script).toContain('queryConfigEnabled: false');
    expect(script).toContain('target.origin !== window.location.origin');
    expect(script).toContain("redirect = 'error'");
  });
});
