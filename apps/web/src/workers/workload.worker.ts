/// <reference lib="webworker" />
import {
  WorkloadAccumulator,
  parseTrackingLine,
  resolveWorkloadConfig,
  workloadTransferables,
  type WorkloadConfig
} from '@replay/shared';

declare const self: DedicatedWorkerGlobalScope;

async function recompute(file: File, input: Partial<WorkloadConfig>): Promise<void> {
  const config = resolveWorkloadConfig(input);
  const accumulator = new WorkloadAccumulator(config);
  let bytesRead = 0;
  let parsedFrames = 0;
  let lastProgress = 0;
  const counter = new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, controller) {
      bytesRead += chunk.byteLength;
      controller.enqueue(chunk);
    }
  });
  const reader = (file.stream().pipeThrough(counter) as ReadableStream<BufferSource>)
    .pipeThrough(new TextDecoderStream()).getReader();
  let pending = '';

  const consume = (line: string) => {
    const result = parseTrackingLine(line.replace(/\r$/, ''));
    if (result.kind === 'frame') {
      accumulator.add(result.frame);
      parsedFrames += 1;
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    pending += value;
    let newline = pending.indexOf('\n');
    while (newline !== -1) {
      consume(pending.slice(0, newline));
      pending = pending.slice(newline + 1);
      newline = pending.indexOf('\n');
    }
    if (bytesRead - lastProgress >= 256 * 1024 || bytesRead === file.size) {
      lastProgress = bytesRead;
      self.postMessage({ type: 'progress', bytesRead, totalBytes: file.size, parsedFrames });
    }
  }
  if (pending.length) consume(pending);
  const workloads = accumulator.finish();
  self.postMessage({ type: 'progress', bytesRead: file.size, totalBytes: file.size, parsedFrames });
  self.postMessage({ type: 'complete', workloads, config }, workloadTransferables(workloads));
}

self.onmessage = (event: MessageEvent<{ file: File; config: Partial<WorkloadConfig> }>) => {
  void recompute(event.data.file, event.data.config).catch((error: unknown) => {
    self.postMessage({ type: 'error', message: error instanceof Error ? error.message : 'Unable to recompute workload.' });
  });
};
