import type { EventDataset, PlayerWorkload, TrackingData, WorkloadConfig } from '@replay/shared';

export type WorkerResponse =
  | { type: 'progress'; bytesRead: number; totalBytes: number; parsedFrames: number }
  | { type: 'complete'; data: TrackingData }
  | { type: 'error'; message: string };

export type EventWorkerResponse =
  | { type: 'progress'; progress: number; stage: string }
  | { type: 'complete'; data: EventDataset }
  | { type: 'error'; message: string };

export type WorkloadWorkerResponse =
  | { type: 'progress'; bytesRead: number; totalBytes: number; parsedFrames: number }
  | { type: 'complete'; workloads: PlayerWorkload[]; config: WorkloadConfig }
  | { type: 'error'; message: string };
