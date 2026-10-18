export function cvServerConfig(env: NodeJS.ProcessEnv) {
  const apiKey = env.CV_API_KEY ?? '';
  if (!/^[\x21-\x7e]{32,256}$/.test(apiKey) || /^(change[-_ ]?me|your[-_ ]|replace|example)/i.test(apiKey)) {
    throw new Error('Set CV_API_KEY to a privately generated random key of 32–256 printable characters, with no spaces.');
  }
  const text = env.PORT ?? '3001';
  const port = Number(text);
  if (!/^\d+$/.test(text) || !Number.isInteger(port) || port < 1 || port > 65535) throw new Error('PORT must be an integer from 1 to 65535.');
  return { apiKey, host: '0.0.0.0', port };
}
