/// <reference lib="webworker" />
import { parseEventDataset } from '@replay/shared';

declare const self: DedicatedWorkerGlobalScope;

self.onmessage = (event: MessageEvent<{ file: File }>) => {
  void (async () => {
    try {
      self.postMessage({ type: 'progress', progress: 0.1, stage: 'Reading event file' });
      const source = await event.data.file.text();
      self.postMessage({ type: 'progress', progress: 0.45, stage: 'Indexing events and players' });
      const data = parseEventDataset(source);
      self.postMessage({ type: 'progress', progress: 1, stage: 'Ready' });
      self.postMessage({ type: 'complete', data });
    } catch (error) {
      self.postMessage({
        type: 'error',
        message: error instanceof Error ? error.message : 'Unable to parse events file.'
      });
    }
  })();
};
