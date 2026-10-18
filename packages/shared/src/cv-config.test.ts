import { describe, expect, it } from 'vitest';
import { cvServerConfig } from '../../../apps/api/src/cv-config';

describe('standalone CV server configuration', () => {
  it('requires a non-placeholder API key and supports Render port binding', () => {
    expect(() => cvServerConfig({})).toThrow(/CV_API_KEY/);
    expect(() => cvServerConfig({ CV_API_KEY: 'short' })).toThrow(/CV_API_KEY/);
    expect(() => cvServerConfig({ CV_API_KEY: ' '.repeat(40) })).toThrow(/CV_API_KEY/);
    const apiKey = 'test-only-not-for-render-0123456789abcdef';
    expect(cvServerConfig({ CV_API_KEY: apiKey, PORT: '10000', NODE_ENV: 'production' })).toEqual({ apiKey, host: '0.0.0.0', port: 10000 });
    expect(cvServerConfig({ CV_API_KEY: apiKey })).toMatchObject({ host: '0.0.0.0', port: 3001 });
    for (const PORT of ['0', '-1', '70000', '1.5', 'not-a-port', '']) expect(() => cvServerConfig({ CV_API_KEY: apiKey, PORT })).toThrow(/PORT/);
  });
});
