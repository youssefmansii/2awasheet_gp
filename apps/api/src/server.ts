import { createServer } from 'node:http';
import { createCvService } from './cv-service';
import { cvServerConfig } from './cv-config';

const config = cvServerConfig(process.env);
const service = createCvService({ apiKey: config.apiKey });
const server = createServer((req, res) => { void service.handle(req, res); });
server.requestTimeout = 30000;
server.headersTimeout = 15000;
server.maxHeadersCount = 64;
server.maxRequestsPerSocket = 500;
server.listen(config.port, config.host, () => console.info(`CV diagnostic API listening on port ${config.port}; memory-only storage.`));
server.on('error', () => { console.error('CV API server could not listen. Check PORT and service configuration.'); service.close(); process.exitCode = 1; });
for (const signal of ['SIGINT', 'SIGTERM'] as const) process.once(signal, () => {
  service.close(); server.close(); server.closeAllConnections();
});
