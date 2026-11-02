/// <reference lib="webworker" />
import {
  BALL_FLOATS,
  PLAYER_FLOATS,
  TOTAL_SLOTS,
  WorkloadAccumulator,
  parseTrackingLine,
  trackingTransferables,
  type ParsedFrame,
  type TrackingData,
  type WorkloadConfig
} from '@replay/shared';

declare const self: DedicatedWorkerGlobalScope;

class CompactBuilder {
  private capacity: number;
  private length = 0;
  private data: TrackingData;
  private readonly workload: WorkloadAccumulator;

  constructor(capacity: number, config: Partial<WorkloadConfig>) {
    this.capacity = Math.max(128, capacity);
    this.data = this.allocate(this.capacity);
    this.workload = new WorkloadAccumulator(config);
  }

  private allocate(capacity: number): TrackingData {
    return {
      frameCount: 0,
      malformedLines: 0,
      blankLines: 0,
      overflowPlayers: 0,
      gameRefId: null,
      frameNum: new Int32Array(capacity),
      videoTimeMs: new Float64Array(capacity),
      period: new Uint8Array(capacity),
      periodGameClockTime: new Float64Array(capacity),
      rawPlayers: new Float32Array(capacity * PLAYER_FLOATS),
      smoothedPlayers: new Float32Array(capacity * PLAYER_FLOATS),
      rawJerseys: new Uint8Array(capacity * TOTAL_SLOTS),
      smoothedJerseys: new Uint8Array(capacity * TOTAL_SLOTS),
      rawFlags: new Uint8Array(capacity * TOTAL_SLOTS),
      smoothedFlags: new Uint8Array(capacity * TOTAL_SLOTS),
      rawBall: new Float32Array(capacity * BALL_FLOATS),
      smoothedBall: new Float32Array(capacity * BALL_FLOATS),
      rawBallVisible: new Uint8Array(capacity),
      smoothedBallVisible: new Uint8Array(capacity),
      workloads: []
    };
  }

  private grow(): void {
    const nextCapacity = Math.ceil(this.capacity * 1.25);
    const next = this.allocate(nextCapacity);
    for (const key of [
      'frameNum', 'videoTimeMs', 'period', 'periodGameClockTime', 'rawPlayers', 'smoothedPlayers',
      'rawJerseys', 'smoothedJerseys', 'rawFlags', 'smoothedFlags', 'rawBall', 'smoothedBall',
      'rawBallVisible', 'smoothedBallVisible'
    ] as const) {
      next[key].set(this.data[key] as never);
    }
    next.malformedLines = this.data.malformedLines;
    next.blankLines = this.data.blankLines;
    next.overflowPlayers = this.data.overflowPlayers;
    next.gameRefId = this.data.gameRefId;
    this.capacity = nextCapacity;
    this.data = next;
  }

  add(frame: ParsedFrame, overflowPlayers: number): void {
    if (this.length >= this.capacity) this.grow();
    const index = this.length;
    this.data.frameNum[index] = frame.frameNum;
    this.data.videoTimeMs[index] = frame.videoTimeMs;
    this.data.period[index] = frame.period;
    this.data.periodGameClockTime[index] = frame.periodGameClockTime;
    this.data.rawPlayers.set(frame.rawPlayers, index * PLAYER_FLOATS);
    this.data.smoothedPlayers.set(frame.smoothedPlayers, index * PLAYER_FLOATS);
    this.data.rawJerseys.set(frame.rawJerseys, index * TOTAL_SLOTS);
    this.data.smoothedJerseys.set(frame.smoothedJerseys, index * TOTAL_SLOTS);
    this.data.rawFlags.set(frame.rawFlags, index * TOTAL_SLOTS);
    this.data.smoothedFlags.set(frame.smoothedFlags, index * TOTAL_SLOTS);
    this.data.rawBall.set(frame.rawBall, index * BALL_FLOATS);
    this.data.smoothedBall.set(frame.smoothedBall, index * BALL_FLOATS);
    this.data.rawBallVisible[index] = frame.rawBallVisible;
    this.data.smoothedBallVisible[index] = frame.smoothedBallVisible;
    if (!this.data.gameRefId && frame.gameRefId) this.data.gameRefId = frame.gameRefId;
    this.data.overflowPlayers += overflowPlayers;
    this.workload.add(frame);
    this.length += 1;
  }

  blank(): void { this.data.blankLines += 1; }
  malformed(): void { this.data.malformedLines += 1; }
  finish(): TrackingData {
    this.data.frameCount = this.length;
    this.data.workloads = this.workload.finish();
    return this.data;
  }
}

async function parseFile(file: File, config: Partial<WorkloadConfig>): Promise<void> {
  let bytesRead = 0;
  let parsedFrames = 0;
  let lastProgress = 0;
  const samples: string[] = [];
  let sampledBytes = 0;
  let builder: CompactBuilder | null = null;

  const countBytes = new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, controller) {
      bytesRead += chunk.byteLength;
      controller.enqueue(chunk);
    }
  });
  const byteStream = file.stream().pipeThrough(countBytes) as ReadableStream<BufferSource>;
  const reader = byteStream.pipeThrough(new TextDecoderStream()).getReader();
  let pending = '';

  const consume = (line: string): void => {
    if (!builder && samples.length < 128) {
      samples.push(line);
      sampledBytes += new TextEncoder().encode(line).byteLength + 1;
      if (samples.length < 128) return;
      const average = Math.max(256, sampledBytes / samples.length);
      builder = new CompactBuilder(Math.ceil((file.size / average) * 1.04), config);
      for (const sample of samples) consumeIntoBuilder(sample, builder);
      samples.length = 0;
      parsedFrames = builder.finish().frameCount;
      return;
    }
    if (!builder) return;
    consumeIntoBuilder(line, builder);
    parsedFrames += 1;
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    pending += value;
    let newline = pending.indexOf('\n');
    while (newline !== -1) {
      const line = pending.slice(0, newline).replace(/\r$/, '');
      pending = pending.slice(newline + 1);
      consume(line);
      newline = pending.indexOf('\n');
    }
    if (bytesRead - lastProgress >= 256 * 1024 || bytesRead === file.size) {
      lastProgress = bytesRead;
      self.postMessage({ type: 'progress', bytesRead, totalBytes: file.size, parsedFrames });
    }
  }
  if (pending.length > 0) consume(pending.replace(/\r$/, ''));

  if (!builder) {
    const average = Math.max(256, sampledBytes / Math.max(1, samples.length));
    builder = new CompactBuilder(Math.ceil((file.size / average) * 1.04), config);
    for (const sample of samples) consumeIntoBuilder(sample, builder);
  }
  const data = builder.finish();
  self.postMessage({ type: 'progress', bytesRead: file.size, totalBytes: file.size, parsedFrames: data.frameCount });
  self.postMessage({ type: 'complete', data }, trackingTransferables(data));
}

function consumeIntoBuilder(line: string, builder: CompactBuilder): void {
  const result = parseTrackingLine(line);
  if (result.kind === 'blank') builder.blank();
  else if (result.kind === 'malformed') builder.malformed();
  else builder.add(result.frame, result.overflowPlayers);
}

self.onmessage = (event: MessageEvent<{ file: File; config?: Partial<WorkloadConfig> }>) => {
  void parseFile(event.data.file, event.data.config ?? {}).catch((error: unknown) => {
    self.postMessage({ type: 'error', message: error instanceof Error ? error.message : 'Unable to parse tracking file.' });
  });
};
