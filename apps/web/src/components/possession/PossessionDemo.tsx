import { Activity, ArrowDownToLine, ArrowLeft, ArrowRight, Check, ChevronLeft, ChevronRight, CircleAlert, FileJson, FolderOpen, LoaderCircle, Pause, Play, RotateCcw, Search, ShieldCheck, Upload, X } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { clock, DEMO_API, isAbort, label, message, request, uploadFiles, type Candidate, type DemoFrame, type DemoJob, type EventPage, type Health } from './api';
import { DemoPitch } from './DemoPitch';
import { DemoValidation } from './DemoValidation';
import './possession.css';

type DetailTab = 'events' | 'players' | 'validation';
const humanState: Record<string, string> = {
  likely_control: 'Likely control', interaction: 'Physical interaction', contested: 'Contested ball',
  in_flight_provisional: 'Ball travelling · provisional team', missing_ball: 'Ball tracking unavailable',
  loose_ball: 'No clear interaction', out_of_bounds_candidate: 'Outside pitch · candidate', outside_period: 'Outside a playing period'
};

function FilePicker({ title, description, file, accept, disabled, onChange }: {
  title: string; description: string; file: File | null; accept: string; disabled: boolean; onChange: (file: File | null) => void;
}) {
  return <label className={`pd-file ${file ? 'pd-file-selected' : ''}`}>
    <FileJson size={22} aria-hidden="true" /><span><strong>{file?.name ?? title}</strong><small>{file ? `${(file.size / 1048576).toFixed(1)} MB · ${description}` : description}</small></span>
    {file ? <Check size={16} aria-hidden="true" /> : <Upload size={16} aria-hidden="true" />}
    <input type="file" aria-label={title} accept={accept} disabled={disabled} onChange={(event) => onChange(event.target.files?.[0] ?? null)} />
  </label>;
}

export function PossessionDemo({ active = true }: { active?: boolean }) {
  const [tracking, setTracking] = useState<File | null>(null);
  const [eventsFile, setEventsFile] = useState<File | null>(null);
  const [source, setSource] = useState<'raw' | 'smoothed'>('smoothed');
  const [health, setHealth] = useState<Health | null>(null);
  const [connection, setConnection] = useState<'checking' | 'connected' | 'offline'>('checking');
  const [retry, setRetry] = useState(0);
  const [job, setJob] = useState<DemoJob | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [frame, setFrame] = useState<DemoFrame | null>(null);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [showLabels, setShowLabels] = useState(true);
  const [selected, setSelected] = useState('');
  const [detailTab, setDetailTab] = useState<DetailTab>('events');
  const [kind, setKind] = useState('');
  const [search, setSearch] = useState('');
  const [eventPage, setEventPage] = useState<EventPage | null>(null);
  const [page, setPage] = useState(0);
  const [eventLoading, setEventLoading] = useState(false);
  const [selectedEvent, setSelectedEvent] = useState<number | null>(null);
  const [artifact, setArtifact] = useState('predicted_events.csv');
  const controller = useRef<AbortController | null>(null);
  const currentJob = useRef<DemoJob | null>(null);
  const manifest = job?.state === 'ready' ? job.manifest : undefined;
  const busy = starting || (job != null && ['uploading', 'analyzing', 'loading'].includes(job.state));

  useEffect(() => { currentJob.current = job; }, [job]);
  useEffect(() => {
    const ctrl = new AbortController();
    setConnection('checking');
    request<Health>('/health', { signal: ctrl.signal }).then((data) => { setHealth(data); setConnection('connected'); })
      .catch((e: unknown) => { if (!isAbort(e)) setConnection('offline'); });
    return () => ctrl.abort();
  }, [retry]);
  useEffect(() => () => controller.current?.abort(), []);
  useEffect(() => { if (!active) setPlaying(false); }, [active]);

  useEffect(() => {
    if (!job || !['uploading', 'analyzing', 'loading'].includes(job.state)) return;
    const ctrl = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const next = await request<DemoJob>(`/jobs/${job.id}`, { signal: ctrl.signal });
        if (ctrl.signal.aborted) return;
        setJob(next);
        if (next.state === 'ready' && next.manifest) { setTime(next.manifest.start); setPage(0); setSelected(''); }
        if (next.state === 'error') setError(next.error ?? 'Analysis stopped.');
        if (['uploading', 'analyzing', 'loading'].includes(next.state)) timer = setTimeout(() => { void poll(); }, 650);
      } catch (e) { if (!isAbort(e)) setError(message(e)); }
    };
    timer = setTimeout(() => { void poll(); }, 350);
    return () => { ctrl.abort(); clearTimeout(timer); };
  }, [job?.id, job?.state]);

  useEffect(() => {
    if (!job || !manifest || !active) return;
    const ctrl = new AbortController();
    request<DemoFrame>(`/jobs/${job.id}/frame?time_s=${time}`, { signal: ctrl.signal })
      .then(setFrame).catch((e: unknown) => { if (!isAbort(e)) { setPlaying(false); setError(message(e)); } });
    return () => ctrl.abort();
  }, [job?.id, manifest, time, active]);

  useEffect(() => {
    if (!playing || !manifest) return;
    let last = performance.now();
    const timer = setInterval(() => {
      const now = performance.now();
      const advance = (now - last) / 1000 * speed;
      last = now;
      setTime((previous) => {
        let next = Math.min(previous + advance, manifest.end);
        const gapEnd = manifest.segments.find((segment) => segment.start > previous && next < segment.start);
        const inSegment = manifest.segments.some((segment) => next >= segment.start && next <= segment.end);
        if (!inSegment && gapEnd) next = gapEnd.start;
        return next;
      });
    }, 100);
    return () => clearInterval(timer);
  }, [playing, speed, manifest]);
  useEffect(() => { if (manifest && time >= manifest.end) setPlaying(false); }, [time, manifest]);

  useEffect(() => {
    if (!job || !manifest || detailTab !== 'events') return;
    const ctrl = new AbortController();
    setEventLoading(true);
    setEventPage(null);
    const timer = setTimeout(() => {
      const params = new URLSearchParams({ kind, player: search, page: String(page), size: '40' });
      request<EventPage>(`/jobs/${job.id}/events?${params}`, { signal: ctrl.signal }).then((next) => {
        setEventPage(next); setEventLoading(false);
      }).catch((e: unknown) => { if (!isAbort(e)) { setError(message(e)); setEventLoading(false); } });
    }, 150);
    return () => { ctrl.abort(); clearTimeout(timer); };
  }, [job?.id, manifest, kind, search, page, detailTab]);

  const resetView = () => { setPlaying(false); setFrame(null); setError(null); setEventPage(null); setKind(''); setSearch(''); setPage(0); setSelectedEvent(null); };
  const chooseTracking = (file: File | null) => {
    setTracking(file);
    if (file) void file.slice(0, 65536).text().then((text) => {
      try {
        const first = JSON.parse(text.trimStart().split(/\r?\n/)[0] ?? '') as Record<string, unknown>;
        if (!first.homePlayersSmoothed && !first.awayPlayersSmoothed) setSource('raw');
      } catch { /* The analysis service reports malformed input with context. */ }
    });
  };
  const analyse = async () => {
    if (!tracking) return;
    resetView(); setJob(null); currentJob.current = null; setStarting(true); setUploadProgress(0);
    const ctrl = new AbortController(); controller.current = ctrl;
    let reservedId: string | undefined;
    try {
      const next = await request<DemoJob>('/jobs', { method: 'POST',
        headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: tracking.name, source }) });
      reservedId = next.id;
      currentJob.current = next;
      if (ctrl.signal.aborted) { setJob(await request<DemoJob>(`/jobs/${next.id}`, { method: 'DELETE' })); return; }
      setJob(next);
      const running = await uploadFiles(next.id, tracking, eventsFile, ctrl.signal, setUploadProgress);
      if (!ctrl.signal.aborted) setJob(running);
    } catch (e) {
      if (!isAbort(e)) {
        setError(message(e));
        if (reservedId) void request<DemoJob>(`/jobs/${reservedId}`, { method: 'DELETE' }).then(setJob).catch(() => undefined);
      }
    } finally { setStarting(false); }
  };
  const openSample = async () => {
    resetView(); setJob(null); currentJob.current = null; setStarting(true);
    const ctrl = new AbortController(); controller.current = ctrl;
    try {
      const next = await request<DemoJob>('/sample', { method: 'POST' });
      currentJob.current = next;
      setJob(ctrl.signal.aborted ? await request<DemoJob>(`/jobs/${next.id}`, { method: 'DELETE' }) : next);
    }
    catch (e) { setError(message(e)); }
    finally { setStarting(false); }
  };
  const cancel = async () => {
    controller.current?.abort(); setStarting(false); setPlaying(false);
    const id = currentJob.current?.id;
    if (id) {
      try { setJob(await request<DemoJob>(`/jobs/${id}`, { method: 'DELETE' })); }
      catch (e) { setError(message(e)); }
    }
  };
  const seek = (target: number) => { if (manifest) setTime(Math.max(manifest.start, Math.min(target, manifest.end))); };
  const chooseEvent = (event: Candidate) => { setPlaying(false); seek(event.time_s); setSelectedEvent(event.id); if (event.actor) setSelected(event.actor); };
  const playerName = (key: string) => manifest?.players.find((p) => p.key === key)?.name ?? (key || 'Unassigned');
  const carrier = frame?.players.find((p) => p.key === frame.carrier);
  const inspected = frame?.players.find((p) => p.key === selected) ?? carrier;
  const homeShare = manifest?.summary.assigned_share_percent.home ?? 0;
  const coverage = manifest?.summary.team_assignment_coverage_percent;

  return <main className="possession-demo">
    <header className="pd-heading"><div><div className="pd-eyebrow"><span className="pd-live-dot" /> TRACKING INTELLIGENCE</div>
      <h1>Possession <span>lab.</span></h1><p>Follow the ball. Understand the interaction. Explore the events.</p></div>
      <div className="pd-local-badge"><ShieldCheck size={16} aria-hidden="true" /> Local analysis <span>·</span> Research demo</div></header>

    <section className="pd-upload" aria-label="Analysis files">
      <FilePicker title="Choose tracking file" description="Required · .jsonl" file={tracking} accept=".jsonl" disabled={busy} onChange={chooseTracking} />
      <FilePicker title="Choose events file" description="Optional · validate predictions" file={eventsFile} accept=".json" disabled={busy} onChange={setEventsFile} />
      <div className="pd-source"><label htmlFor="pd-source">Position source</label><select id="pd-source" value={source} disabled={busy} onChange={(e) => setSource(e.target.value as 'raw' | 'smoothed')}><option value="smoothed">Smoothed</option><option value="raw">Raw</option></select></div>
      <button type="button" className="pd-primary" disabled={!tracking || busy || connection !== 'connected'} onClick={() => { void analyse(); }}><Activity size={17} aria-hidden="true" /> Analyse locally</button>
    </section>
    <div className="pd-upload-note"><span>Files are sent only to the local service on this computer when you press Analyse locally. No cloud upload.</span>
      {eventsFile && !busy && <button type="button" className="pd-text-button" onClick={() => setEventsFile(null)}>Remove events file</button>}</div>

    {connection === 'offline' && <div className="pd-alert" role="status"><CircleAlert size={19} aria-hidden="true" /><div><strong>Start the local demo service</strong><p>Run <code>npm run demo:possession</code> from the project folder, then reconnect.</p></div><button type="button" onClick={() => setRetry((v) => v + 1)}>Reconnect</button></div>}
    {error && <div className="pd-alert pd-error" role="alert"><CircleAlert size={19} aria-hidden="true" /><p>{error}</p><button type="button" aria-label="Dismiss error" onClick={() => setError(null)}><X size={17} /></button></div>}
    {busy && <section className="pd-progress" aria-live="polite"><LoaderCircle className="pd-spin" size={23} aria-hidden="true" /><div><strong>{job?.state === 'uploading' ? 'Sending selected files to the local service' : job?.stage ?? 'Preparing analysis'}</strong>
      <p>{job?.state === 'uploading' ? `${Math.round(uploadProgress * 100)}% uploaded` : 'Processing the recording. Large matches can take a few minutes.'}</p>
      <progress aria-label="Analysis progress" max="1" value={job?.state === 'uploading' ? uploadProgress * .15 : job?.progress ?? .02} /></div>
      <button type="button" className="pd-quiet" onClick={() => { void cancel(); }}>Cancel analysis</button></section>}

    {!manifest && !busy && <section className="pd-empty"><div className="pd-empty-pitch"><DemoPitch frame={null} selected="" onSelect={() => undefined} labels={false} /></div>
      <div className="pd-empty-copy"><span className="pd-eyebrow">FROM POSITIONS TO PLAY</span><h2>A closer look at every interaction.</h2><p>Load your tracking file to replay player movement, inspect possession estimates and jump to detected event candidates.</p>
        <ul><li><Check size={16} /> Player and ball replay</li><li><Check size={16} /> Interaction and possession states</li><li><Check size={16} /> Event explorer and validation</li></ul>
        {health?.sample_available && <button type="button" className="pd-primary" onClick={() => { void openSample(); }}><FolderOpen size={17} /> Open analysed match</button>}
        {health?.current_job && ['uploading', 'analyzing', 'loading', 'ready'].includes(health.current_job.state) && <button type="button" className="pd-quiet" onClick={() => { resetView(); setJob(health.current_job); if (health.current_job?.manifest) setTime(health.current_job.manifest.start); }}>Resume latest analysis</button>}
        <small>Candidate events are estimates. Accuracy is measured only when an events file is provided.</small></div></section>}

    {manifest && job && <>
      <div className="pd-recording"><span><span className="pd-live-dot" /> {job.name}</span><small>{manifest.frame_count.toLocaleString()} frames · {manifest.source} source · retrospective analysis</small>
        {health?.sample_available && <button type="button" className="pd-text-button" onClick={() => { void openSample(); }}>Open analysed match</button>}</div>
      <section className="pd-stats" aria-label="Recording overview">
        <div className="pd-stat"><span>Current carrier</span><strong>{carrier ? `#${carrier.jersey}` : '—'} <small>{carrier?.side.toUpperCase() ?? 'UNASSIGNED'}</small></strong><p>{carrier?.name ?? 'No confirmed interaction at this frame'}</p></div>
        <div className="pd-stat pd-share"><span>Share of assigned team time <em>Full recording</em></span><strong>{manifest.summary.assigned_share_percent.home == null ? '—' : homeShare.toFixed(1)}<small>% HOME</small><b>/</b>{manifest.summary.assigned_share_percent.away == null ? '—' : manifest.summary.assigned_share_percent.away.toFixed(1)}<small>% AWAY</small></strong>
          <div className="pd-share-bar" style={manifest.summary.assigned_share_percent.home == null ? { background: '#2d3b30' } : undefined}><span style={{ width: `${homeShare}%` }} /></div><p>Provisional share · not official match possession</p></div>
        <div className="pd-stat"><span>Team assignment coverage</span><strong>{coverage == null ? '—' : `${coverage.toFixed(1)}%`}</strong><p>{coverage == null ? 'Not available' : `${(100 - coverage).toFixed(1)}% of recorded time remains unassigned`}</p></div>
        <div className="pd-stat"><span>Event candidates <em>Full recording</em></span><strong>{manifest.event_count.toLocaleString()}</strong><p>{Object.keys(manifest.candidate_counts).length} detected types · labels can overlap</p></div>
      </section>

      <div className="pd-replay-grid"><section className="pd-pitch-card" aria-label="Tracking replay"><header><div><span className="pd-eyebrow">TRACKING REPLAY</span><h2>Every movement. In context.</h2></div><span className="pd-frame-tag">P{frame?.period ?? 1} <i /> Frame {frame?.frame_num.toLocaleString() ?? '—'}</span></header>
        <DemoPitch frame={frame} selected={selected} onSelect={setSelected} labels={showLabels} />
        <div className="pd-pitch-legend"><span><i className="pd-home-dot" /> Home</span><span><i className="pd-away-dot" /> Away</span><span><i className="pd-ball-dot" /> Ball</span><span><i className="pd-carrier-dot" /> Interaction</span><label><input type="checkbox" checked={showLabels} onChange={(e) => setShowLabels(e.target.checked)} /> Jersey numbers</label></div>
        <div className="pd-playback"><input type="range" aria-label="Seek tracking replay" min={manifest.start} max={Math.max(manifest.start + .001, manifest.end)} step=".033" value={time} onChange={(e) => seek(Number(e.target.value))} />
          <div className="pd-transport"><div><button type="button" aria-label="Restart replay" onClick={() => { setPlaying(false); seek(manifest.start); }}><RotateCcw size={17} /></button>
            <button type="button" aria-label="Back five seconds" onClick={() => seek(time - 5)}><ArrowLeft size={17} /></button>
            <button type="button" className="pd-play" aria-label={playing ? 'Pause replay' : 'Play replay'} onClick={() => { if (time >= manifest.end) seek(manifest.start); setPlaying((v) => !v); }}>{playing ? <Pause size={18} /> : <Play size={18} />}</button>
            <button type="button" aria-label="Forward five seconds" onClick={() => seek(time + 5)}><ArrowRight size={17} /></button><span className="pd-clock">{clock(time - manifest.start)} <small>/ {clock(manifest.end - manifest.start)}</small></span></div>
            <div><select aria-label="Replay speed" value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>{[.5, 1, 2, 4, 8].map((v) => <option key={v} value={v}>{v}×</option>)}</select>
              <select aria-label="Jump to period" value="" onChange={(e) => { if (e.target.value) seek(Number(e.target.value)); }}><option value="">Jump to period</option>{manifest.segments.filter((segment, index, all) => index === all.findIndex((s) => s.period === segment.period)).map((segment) => <option key={segment.start} value={segment.start}>Period {segment.period}</option>)}</select></div>
          </div></div>
      </section>

      <aside className="pd-frame-panel"><span className="pd-eyebrow">AT THIS FRAME</span><div className={`pd-state ${frame?.status === 'contested' ? 'pd-state-warning' : ''}`}><span className="pd-live-dot" />{frame ? humanState[frame.status] ?? label(frame.status) : 'Loading frame…'}</div>
        <div className="pd-player-identity"><span className={`pd-jersey ${inspected?.side === 'away' ? 'pd-jersey-away' : ''}`}>{inspected?.jersey ?? '—'}</span><div><small>{inspected ? `${inspected.side.toUpperCase()} · SELECTED PLAYER` : 'PLAYER INTERACTION'}</small><h3>{inspected?.name ?? 'No player assigned'}</h3><p>{inspected && inspected.key !== frame?.carrier ? 'Selected player · not the estimated carrier' : frame?.control_status === 'likely_control' ? 'Likely control · heuristic estimate' : frame?.control_status === 'interaction_only' ? 'Physical interaction · control not established' : 'Control is uncertain at this moment'}</p></div></div>
        <dl className="pd-frame-values"><div><dt>Team possession</dt><dd>{frame?.team ? `${frame.team.toUpperCase()} · provisional` : 'Unassigned'}</dd></div>
          <div><dt>Ball speed</dt><dd>{frame?.ball?.speed == null ? 'Unavailable' : `${frame.ball.speed.toFixed(1)} m/s`}</dd></div>
          <div><dt>Player speed</dt><dd>{inspected?.speed == null ? '—' : `${inspected.speed.toFixed(1)} m/s`}</dd></div>
          <div><dt>Player → ball</dt><dd>{inspected?.ball_distance == null ? '—' : `${inspected.ball_distance.toFixed(2)} m`}</dd></div>
          <div><dt>Ball tracking</dt><dd>{!frame?.ball ? 'Missing' : frame.ball.reconstructed ? 'Short gap reconstructed' : (frame.tracking_weight ?? 0) < .5 ? 'Estimated position' : 'Observed position'}</dd></div>
          <div><dt>Source timestamp</dt><dd>{frame ? `${frame.time_s.toFixed(3)} s` : '—'}</dd></div></dl>
        {selected && <button type="button" className="pd-text-button" onClick={() => { setSearch(selected); setPage(0); setDetailTab('events'); }}>Show this player’s events <ArrowRight size={14} /></button>}
        <div className="pd-nearby"><span className="pd-eyebrow">NEAR THIS MOMENT</span>{frame?.nearby_events.length ? frame.nearby_events.slice(0, 3).map((event) => <button type="button" key={event.id} onClick={() => chooseEvent(event)}><span className="pd-capitalize">{label(event.event_type)}</span><small>{clock(event.time_s)} <ChevronRight size={13} /></small></button>) : <p>No nearby event candidate.</p>}</div>
        <p className="pd-note">A highlighted player indicates an estimated interaction. The learned control classifier is not trained. Missing and contested frames remain unknown.</p>
      </aside></div>

      <section className="pd-details"><header><div className="pd-tabs" aria-label="Analysis details">{(['events', 'players', 'validation'] as const).map((tab) => <button type="button" key={tab} aria-pressed={detailTab === tab} onClick={() => setDetailTab(tab)}>{tab === 'events' ? 'Event explorer' : tab === 'players' ? 'Player interaction' : 'Validation & method'}{tab === 'events' && <span>{manifest.event_count.toLocaleString()}</span>}</button>)}</div>
        <div className="pd-export"><select aria-label="Choose export data" value={artifact} onChange={(e) => setArtifact(e.target.value)}>
          <option value="predicted_events.csv">Event candidates</option><option value="frames.csv">Frame estimates</option><option value="players.csv">Player totals</option><option value="control_annotations.csv">Control features / labels</option><option value="event_matches.csv">Validation matches</option><option value="report.json">Full report (JSON)</option><option value="REPORT.md">Readable report</option>
        </select><a className="pd-download" href={`${DEMO_API}/jobs/${job.id}/download/${artifact}`} download><ArrowDownToLine size={15} /> Export</a></div></header>
        {detailTab === 'events' && <><div className="pd-event-controls"><div><h2>Explore the event candidates</h2><p>Full recording · select a row to jump to the action</p></div>
          <label className="pd-search"><Search size={16} aria-hidden="true" /><input aria-label="Search event players" placeholder="Search player or jersey…" value={search} onChange={(e) => { setSearch(e.target.value); setPage(0); }} /></label>
          <select aria-label="Filter event type" value={kind} onChange={(e) => { setKind(e.target.value); setPage(0); }}><option value="">All event types</option>{Object.keys(manifest.candidate_counts).sort().map((type) => <option key={type} value={type}>{label(type)} ({manifest.candidate_counts[type]})</option>)}</select></div>
          <div className="pd-table-wrap"><table className="pd-table"><thead><tr><th>Video time</th><th>Period</th><th>Event candidate</th><th>Player</th><th>Target / outcome player</th><th>Evidence</th></tr></thead><tbody>
            {eventPage?.events.map((event) => <tr key={event.id} className={event.id === selectedEvent ? 'pd-selected-row' : ''}>
              <td><button type="button" className="pd-time-button" onClick={() => chooseEvent(event)} aria-label={`Jump to ${label(event.event_type)} at ${clock(event.time_s)}`}><Play size={12} />{clock(event.time_s)}</button></td><td>P{event.period}</td>
              <td><button type="button" className="pd-event-type pd-capitalize" onClick={() => chooseEvent(event)}>{label(event.event_type)}</button></td><td>{playerName(event.actor)}</td><td>{event.target ? playerName(event.target) : '—'}</td><td className="pd-evidence">{label(event.evidence)}</td></tr>)}
          </tbody></table></div>
          {!eventLoading && eventPage?.events.length === 0 && <p className="pd-no-events">No candidates match these filters.</p>}
          <footer className="pd-table-footer"><span aria-live="polite">{eventLoading ? 'Loading events…' : `${(eventPage?.total ?? 0).toLocaleString()} matching candidates`}</span><span>Page {page + 1} <button type="button" aria-label="Previous event page" disabled={page === 0 || eventLoading} onClick={() => setPage((v) => v - 1)}><ChevronLeft size={17} /></button><button type="button" aria-label="Next event page" disabled={eventLoading || (page + 1) * 40 >= (eventPage?.total ?? 0)} onClick={() => setPage((v) => v + 1)}><ChevronRight size={17} /></button></span></footer>
          <p className="pd-note pd-table-note">All rows are candidates, not official event judgements. Several labels can describe the same action. Video times are source timestamps, not the match clock.</p></>}
        {detailTab === 'players' && <><div className="pd-event-controls"><div><h2>Player interaction totals</h2><p>Full recording · partial observed interaction time, not complete control time</p></div><a className="pd-download" href={`${DEMO_API}/jobs/${job.id}/download/players.csv`} download><ArrowDownToLine size={15} /> Export players</a></div>
          <div className="pd-table-wrap"><table className="pd-table"><thead><tr><th>Player</th><th>Team</th><th>Jersey</th><th>Interaction time</th><th>Touch candidates</th><th>Completed-pass candidates</th></tr></thead><tbody>{[...manifest.players].sort((a, b) => b.interaction_seconds - a.interaction_seconds).map((player) => <tr key={player.key}><td><button type="button" className="pd-text-button" onClick={() => { setSelected(player.key); setSearch(player.key); setPage(0); setDetailTab('events'); }}>{player.name}</button></td><td className="pd-capitalize">{player.side}</td><td>#{player.jersey}</td><td>{clock(player.interaction_seconds)}</td><td>{player.touches}</td><td>{player.passes}</td></tr>)}</tbody></table></div></>}
        {detailTab === 'validation' && <DemoValidation manifest={manifest} />}
      </section>
      <footer className="pd-bottom"><span><ShieldCheck size={14} /> Processed on this computer · no cloud upload</span><a href={`${DEMO_API}/jobs/${job.id}/download/report.json`} download>Download full analysis report <ArrowDownToLine size={14} /></a></footer>
    </>}
  </main>;
}
