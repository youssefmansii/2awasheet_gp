export const DEMO_API = 'http://127.0.0.1:8010/api/possession';

export interface Candidate {
  id: number; event_type: string; period: number; time_s: number; end_time_s: number;
  actor: string; target: string; challenger?: string; evidence: string;
}
export interface DemoPlayer {
  key: string; name: string; side: string; jersey: string; interaction_seconds: number; touches: number; passes: number;
}
export interface Metric {
  predictions: number; references: number; matched: number; precision: number | null;
  reference_recall: number | null; f1: number | null; tolerance_s: number; warning?: string;
  winner_among_matched_contests?: { labels: number; assigned: number; correct: number; accuracy_when_assigned: number | null };
}
export interface Manifest {
  source: string; frame_count: number; start: number; end: number;
  segments: { period: number; start: number; end: number }[];
  players: DemoPlayer[]; event_count: number; candidate_counts: Record<string, number>;
  summary: {
    accounted_seconds: number; team_assignment_coverage_percent: number | null;
    assigned_share_percent: { home: number | null; away: number | null };
    team_assigned_seconds?: Record<string, number>; control_estimate_seconds?: Record<string, number>;
  };
  validation: { event_detection: Record<string, Metric>; frame_actor_proxy?: { all: {
    labels: number; covered: number; correct_player: number; carrier_coverage: number | null;
    player_accuracy_when_covered: number | null;
  } }; unvalidated?: string[] } | null;
  control_classifier: string; parameters: Record<string, number>; limitations: string[];
}
export interface DemoJob {
  id: string; name: string; source: string; state: 'uploading' | 'analyzing' | 'loading' | 'ready' | 'error' | 'cancelled';
  stage: string; progress: number; error: string | null; manifest?: Manifest;
}
export interface DemoFrame {
  time_s: number; frame_num: number; period: number; carrier: string; team: string;
  status: string; control_status: string; previous_owner: string; tracking_weight: number | null;
  ball: { x: number; y: number; z: number | null; speed: number | null; reconstructed: boolean } | null;
  players: { key: string; name: string; side: string; jersey: string; x: number; y: number; speed: number | null; ball_distance: number | null }[];
  nearby_events: Candidate[];
}
export interface EventPage { total: number; page: number; page_size: number; events: Candidate[] }
export interface Health { status: string; sample_available: boolean; current_job: DemoJob | null }

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${DEMO_API}${path}`, init);
  const data = await response.json() as { detail?: unknown };
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `Request failed (${response.status}).`);
  return data as T;
}

export function uploadFiles(id: string, tracking: File, events: File | null, signal: AbortSignal, progress: (fraction: number) => void): Promise<DemoJob> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) return reject(new DOMException('Upload cancelled', 'AbortError'));
    const xhr = new XMLHttpRequest();
    const abort = () => xhr.abort();
    const clean = () => signal.removeEventListener('abort', abort);
    signal.addEventListener('abort', abort, { once: true });
    xhr.open('POST', `${DEMO_API}/jobs/${id}/files`);
    xhr.upload.onprogress = (event) => { if (event.lengthComputable) progress(event.loaded / event.total); };
    xhr.onload = () => {
      clean();
      try {
        const result = JSON.parse(xhr.responseText) as DemoJob & { detail?: string };
        if (xhr.status >= 200 && xhr.status < 300) resolve(result);
        else reject(new Error(result.detail ?? `Upload failed (${xhr.status}).`));
      } catch { reject(new Error('The local service returned an unreadable response.')); }
    };
    xhr.onerror = () => { clean(); reject(new Error('Could not reach the local analysis service.')); };
    xhr.onabort = () => { clean(); reject(new DOMException('Upload cancelled', 'AbortError')); };
    const body = new FormData();
    body.append('tracking', tracking);
    if (events) body.append('events', events);
    xhr.send(body);
  });
}

export const isAbort = (reason: unknown) => reason instanceof DOMException && reason.name === 'AbortError';
export const message = (reason: unknown) => reason instanceof Error ? reason.message : 'Something went wrong.';
export const label = (value: string) => value.replace(/_candidate$/, '').replaceAll('_', ' ');
export const percent = (value: number | null | undefined) => value == null ? '—' : `${(value * 100).toFixed(1)}%`;
export const clock = (seconds: number) => {
  const n = Math.max(0, Math.floor(seconds));
  return `${Math.floor(n / 60).toString().padStart(2, '0')}:${(n % 60).toString().padStart(2, '0')}`;
};
