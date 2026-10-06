import { useEffect, useMemo, useState } from 'react';
import { SlidersHorizontal } from 'lucide-react';
import { resolveWorkloadConfig, type WorkloadConfig } from '@replay/shared';

type Props = {
  config: WorkloadConfig;
  available: boolean;
  busy: boolean;
  progress: number;
  frames: number;
  error: string | undefined;
  onApply: (config: WorkloadConfig) => void;
};

const fields: Array<{ key: keyof WorkloadConfig; label: string; unit: string; step: number }> = [
  { key: 'highSpeedThresholdMps', label: 'High-speed starts', unit: 'm/s', step: 0.1 },
  { key: 'sprintThresholdMps', label: 'Sprint starts', unit: 'm/s', step: 0.1 },
  { key: 'maxValidSpeedMps', label: 'Maximum valid speed', unit: 'm/s', step: 0.1 },
  { key: 'sprintMinDurationSeconds', label: 'Minimum sprint', unit: 's', step: 0.1 },
  { key: 'accelerationThresholdMps2', label: 'Acceleration threshold', unit: 'm/s²', step: 0.1 },
  { key: 'accelerationMinDurationSeconds', label: 'Minimum acceleration', unit: 's', step: 0.1 },
  { key: 'accelerationWindowSeconds', label: 'Acceleration window', unit: 's', step: 0.1 },
  { key: 'accelerationResetMps2', label: 'Acceleration reset', unit: 'm/s²', step: 0.1 },
  { key: 'maxFrameGapSeconds', label: 'Maximum frame gap', unit: 's', step: 0.01 },
  { key: 'checkpointIntervalMs', label: 'Live update interval', unit: 'ms', step: 100 },
  { key: 'histogramStepMps', label: 'Speed precision', unit: 'm/s', step: 0.05 },
  { key: 'minimumTrendQualityPercent', label: 'Minimum trend quality', unit: '%', step: 1 }
];

export function WorkloadSettings({ config, available, busy, progress, frames, error, onApply }: Props) {
  const [draft, setDraft] = useState<WorkloadConfig>(config);
  useEffect(() => setDraft(config), [config]);
  const validation = useMemo(() => {
    try { resolveWorkloadConfig(draft); return ''; }
    catch (reason) { return reason instanceof Error ? reason.message : 'Check the workload settings.'; }
  }, [draft]);
  const changed = fields.some(({ key }) => draft[key] !== config[key]);

  return <section className="rounded-3xl border border-line bg-panel p-5 shadow-panel" aria-labelledby="workload-settings-title">
    <div className="flex items-center gap-2 text-lime"><SlidersHorizontal aria-hidden="true" className="h-4 w-4" /><p className="text-xs font-bold uppercase tracking-[0.16em]">Model</p></div>
    <h2 id="workload-settings-title" className="mt-1 font-semibold text-white">Workload settings</h2>
    <p className="mt-2 text-xs leading-5 text-muted">Applied to smoothed player positions. Changing a value streams the tracking file again in a background worker.</p>
    <details className="mt-4" open>
      <summary className="min-h-11 cursor-pointer select-none py-3 text-sm font-semibold text-slate-200">Thresholds and windows</summary>
      <div className="space-y-3 pt-2">
        {fields.map(({ key, label, unit, step }) => <label key={key} className="grid grid-cols-[minmax(0,1fr)_92px] items-center gap-3 text-xs text-slate-200">
          <span>{label} <span className="text-muted">({unit})</span></span>
          <input
            type="number" min={step} step={step} value={draft[key]}
            onChange={(event) => setDraft((current) => ({ ...current, [key]: Number(event.target.value) }))}
            className="h-11 w-full rounded-xl border border-line bg-[#0d1215] px-3 font-mono tabular-nums text-white outline-none focus:border-lime"
          />
        </label>)}
      </div>
    </details>
    {(validation || error) ? <p className="mt-3 text-xs leading-5 text-amber-300" role="alert">{validation || error}</p> : null}
    {busy ? <div className="mt-4" aria-live="polite"><div className="mb-2 flex justify-between text-xs"><span>Recomputing {frames.toLocaleString()} frames</span><span className="font-mono text-lime">{Math.round(progress * 100)}%</span></div><div className="h-2 overflow-hidden rounded-full bg-black/40"><div className="h-full rounded-full bg-lime transition-[width]" style={{ width: `${progress * 100}%` }} /></div></div> : null}
    <button type="button" disabled={!available || busy || !changed || Boolean(validation)} onClick={() => onApply(resolveWorkloadConfig(draft))} className="mt-4 min-h-11 w-full rounded-xl bg-lime px-4 text-sm font-bold text-[#152000] transition-colors hover:bg-[#d6ff72] disabled:cursor-not-allowed disabled:opacity-40">Apply and recompute</button>
  </section>;
}
