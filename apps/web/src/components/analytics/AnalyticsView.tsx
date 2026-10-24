import { useEffect, useRef, useState } from 'react';
import { DropZone } from '../DropZone';
import {
  errorMessage,
  getJson,
  isAbort,
  readTrackingMatchId,
  uploadMatch,
  withMatch,
  type MatchInfo,
  type MatchList
} from './api';
import { announceMatchFile, clearMatchFiles, getMatchFiles, subscribeMatchFiles, type MatchFiles } from './bridge';
import { Tab1CoreMetrics } from './Tab1CoreMetrics';
import { Tab2PlayerAnalytics } from './Tab2PlayerAnalytics';
import { Tab3LivePlayer } from './Tab3LivePlayer';
import { Tab4TopAttacks } from './Tab4TopAttacks';
import { ErrorBox, Panel, Spinner } from './ui';

type TabKey = 'core' | 'live' | 'players' | 'attacks';
type Phase = 'idle' | 'checking' | 'uploading' | 'loading' | 'ready' | 'error';
interface Selected {
  id: string;
  origin: 'files' | 'server';
}

const fileSignature = (f: File | null): string => (f ? `${f.name}:${f.size}:${f.lastModified}` : '');
const megabytes = (n: number): string => `${(n / 1_048_576).toFixed(0)} MB`;

/**
 * Advanced analytics always follow the match the user opened (in Match Replay, or by dropping the files here).
 * Nothing is shown until a match is chosen: the analytics tab never falls back to "the default match".
 * Files that the server does not have yet are sent to it automatically; the heavy maths then runs on the server.
 */
export function AnalyticsView() {
  const [files, setFiles] = useState<MatchFiles>(getMatchFiles());
  const [selected, setSelected] = useState<Selected | null>(null);
  const [phase, setPhase] = useState<Phase>('idle');
  const [uploadFraction, setUploadFraction] = useState(0);
  const [info, setInfo] = useState<MatchInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [server, setServer] = useState<MatchList | null>(null);
  const [version, setVersion] = useState(0); // bumps whenever the server received new data -> panels reload
  const [retry, setRetry] = useState(0);
  const [needId, setNeedId] = useState(false);
  const [idDraft, setIdDraft] = useState('');
  const [confirmedId, setConfirmedId] = useState<string | null>(null);
  const [tab, setTab] = useState<TabKey>('core');
  const [visited, setVisited] = useState<Record<TabKey, boolean>>({ core: true, live: false, players: false, attacks: false });
  const [sending, setSending] = useState<string>('');
  const sentRef = useRef<{ id: string; events: string } | null>(null);

  useEffect(() => subscribeMatchFiles(setFiles), []);

  // ---- which matches does the server already have? (cheap call, nothing is loaded) ----
  useEffect(() => {
    const ctrl = new AbortController();
    getJson<MatchList>('/api/matches', ctrl.signal)
      .then(setServer)
      .catch((e: unknown) => {
        if (!isAbort(e)) {
          setError(errorMessage(e));
          setPhase('error');
        }
      });
    return () => ctrl.abort();
  }, [retry]);

  // ---- the files opened in the replay (or dropped here) decide the match; send what the server is missing ----
  useEffect(() => {
    const tracking = files.tracking;
    if (!tracking) {
      setSelected((s) => (s && s.origin === 'files' ? null : s));
      return;
    }
    const ctrl = new AbortController();
    void (async () => {
      try {
        setError(null);
        setPhase('checking');
        const id = confirmedId ?? (await readTrackingMatchId(tracking));
        if (ctrl.signal.aborted) return;
        if (!id) {
          setNeedId(true);
          setPhase('idle');
          return;
        }
        const list = await getJson<MatchList>('/api/matches', ctrl.signal);
        setServer(list);
        const onServer = list.details.find((d) => d.match_id === id);
        const needTracking = !onServer || onServer.tracking_bytes !== tracking.size;
        const eventsSig = fileSignature(files.events);
        const sent = sentRef.current;
        const needEvents = files.events !== null && (needTracking || sent?.id !== id || sent.events !== eventsSig);
        if (needTracking || needEvents) {
          if (!list.upload_enabled) {
            throw new Error('The analytics server cannot receive files yet. In the API terminal run:  pip install python-multipart  then restart uvicorn.');
          }
          setSending([needTracking ? `tracking (${megabytes(tracking.size)})` : '', needEvents ? 'events' : ''].filter(Boolean).join(' + '));
          setUploadFraction(0);
          setPhase('uploading');
          await uploadMatch(
            { tracking: needTracking ? tracking : null, events: needEvents ? files.events : null, metadata: needTracking ? files.metadata : null },
            id,
            setUploadFraction,
            ctrl.signal
          );
          sentRef.current = { id, events: eventsSig };
          setVersion((v) => v + 1);
        }
        if (ctrl.signal.aborted) return;
        setNeedId(false);
        setSelected({ id, origin: 'files' });
      } catch (e) {
        if (isAbort(e) || ctrl.signal.aborted) return;
        setError(errorMessage(e));
        setPhase('error');
      }
    })();
    return () => ctrl.abort();
  }, [files.tracking, files.events, files.metadata, confirmedId, retry]);

  // ---- read the chosen match on the server (first time takes a while, later it is instant) ----
  useEffect(() => {
    if (!selected) {
      setInfo(null);
      return;
    }
    const ctrl = new AbortController();
    setPhase('loading');
    setInfo(null);
    getJson<MatchInfo>(withMatch('/api/match', selected.id), ctrl.signal)
      .then((i) => {
        setInfo(i);
        setPhase('ready');
      })
      .catch((e: unknown) => {
        if (ctrl.signal.aborted) return;
        setError(errorMessage(e));
        setPhase('error');
      });
    return () => ctrl.abort();
  }, [selected, version]);

  const open = (k: TabKey) => {
    setTab(k);
    setVisited((v) => ({ ...v, [k]: true }));
  };

  const changeMatch = () => {
    sentRef.current = null;
    setConfirmedId(null);
    setNeedId(false);
    setSelected(null);
    setInfo(null);
    setError(null);
    setPhase('idle');
    clearMatchFiles();
  };

  const home = info?.teams.home.name ?? 'Home';
  const away = info?.teams.away.name ?? 'Away';
  const ready = phase === 'ready' && info !== null && selected !== null;

  return (
    <div className="min-h-dvh bg-ink pb-[max(2rem,env(safe-area-inset-bottom))] text-slate-50">
      <div className="mx-auto max-w-[1480px] px-4 py-5 sm:px-6 lg:px-8">
        <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
          <div>
            <div className="text-xs font-bold uppercase tracking-[0.2em] text-lime">Advanced analytics</div>
            <h2 className="mt-1 text-lg font-semibold tracking-tight text-white sm:text-xl">
              {ready ? `${home} vs ${away}` : 'Match analytics'}
              {ready && info ? <span className="ml-2 text-sm font-normal text-muted">match {info.match_id} · {info.fps.toFixed(2)} fps</span> : null}
            </h2>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {ready ? (
              <button type="button" onClick={changeMatch} className="min-h-9 rounded-lg border border-line px-3 text-sm font-semibold text-slate-200 hover:bg-white/[0.05]">
                Change match
              </button>
            ) : null}
            <div role="tablist" aria-label="Analytics sections" className="flex gap-1 rounded-xl border border-line bg-panel p-1">
              {([['core', 'Core metrics'], ['live', 'Live player'], ['players', 'Player analytics'], ['attacks', 'Top attacks']] as const).map(([k, label]) => (
                <button
                  key={k}
                  type="button"
                  role="tab"
                  aria-selected={tab === k}
                  disabled={!ready}
                  onClick={() => open(k)}
                  className={`min-h-9 rounded-lg px-4 text-sm font-semibold transition disabled:opacity-40 ${tab === k ? 'bg-lime text-ink' : 'text-muted hover:text-white'}`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {error ? (
          <div className="mb-4 space-y-2">
            <ErrorBox message={error} />
            <button
              type="button"
              onClick={() => {
                setError(null);
                setRetry((r) => r + 1);
                setVersion((v) => v + 1);
              }}
              className="min-h-9 rounded-lg border border-line px-3 text-sm font-semibold text-slate-200 hover:bg-white/[0.05]"
            >
              Try again
            </button>
          </div>
        ) : null}

        {phase === 'checking' ? <Spinner label="Checking which files the analytics server already has…" /> : null}
        {phase === 'uploading' ? (
          <Panel title="Sending the match to the analytics server" subtitle={`Sending ${sending}. The server computes every metric from these files (nothing leaves your computer).`} right={null}>
            <div className="h-2 w-full overflow-hidden rounded bg-ink" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(uploadFraction * 100)}>
              <div className="h-full bg-lime transition-all" style={{ width: `${Math.round(uploadFraction * 100)}%` }} />
            </div>
            <p className="mt-2 text-xs text-muted">{Math.round(uploadFraction * 100)}%</p>
          </Panel>
        ) : null}
        {phase === 'loading' ? <Spinner label="The server is reading the match. The first time this takes about a minute; after that it is instant." /> : null}

        {needId ? (
          <Panel title="Which match is this?" subtitle="The tracking file does not say which match it belongs to (no gameRefId, no number in the file name)." right={null}>
            <form
              className="flex flex-wrap items-center gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                if (/^\d{1,12}$/.test(idDraft.trim())) setConfirmedId(idDraft.trim());
              }}
            >
              <input
                value={idDraft}
                onChange={(e) => setIdDraft(e.target.value)}
                inputMode="numeric"
                placeholder="match number, e.g. 3816"
                aria-label="Match number"
                className="min-h-10 rounded-lg border border-line bg-ink px-3 text-sm text-white"
              />
              <button type="submit" className="min-h-10 rounded-lg bg-lime px-4 text-sm font-semibold text-ink">Use this number</button>
            </form>
          </Panel>
        ) : null}

        {phase === 'idle' && !needId && !selected ? (
          <div className="space-y-4">
            <Panel
              title="No match selected"
              subtitle="Advanced analytics follow the match you open. Load its files in Match Replay (they are picked up automatically), or drop them here."
              right={null}
            >
              <div className="grid gap-3 sm:grid-cols-2">
                <DropZone
                  label="Tracking file"
                  hint="JSONL, one frame per line."
                  accept=".json,.jsonl,.bz2,application/json,application/x-ndjson"
                  required
                  fileName={files.tracking?.name}
                  onFile={(f) => announceMatchFile('tracking', f)}
                />
                <DropZone
                  label="Match events"
                  hint="Needed for player names, passes, shots, VAEP and OBV."
                  accept=".json,application/json"
                  fileName={files.events?.name}
                  onFile={(f) => announceMatchFile('events', f)}
                />
              </div>
              {files.tracking && !files.events ? <p className="mt-3 text-xs text-amber-300">No events file yet: add it to get player names, passes and VAEP/OBV.</p> : null}
            </Panel>
            {server && server.details.length > 0 ? (
              <Panel title="Already on the analytics server" subtitle="Matches that were sent before. Open one without loading its files again." right={null}>
                <div className="flex flex-wrap gap-2">
                  {server.details.map((d) => (
                    <button
                      key={d.match_id}
                      type="button"
                      onClick={() => setSelected({ id: d.match_id, origin: 'server' })}
                      className="min-h-10 rounded-lg border border-line px-3 text-sm font-semibold text-slate-100 hover:bg-white/[0.05]"
                    >
                      Match {d.match_id}
                      <span className="ml-2 text-xs font-normal text-muted">{d.has_events ? 'with events' : 'tracking only'}</span>
                    </button>
                  ))}
                </div>
              </Panel>
            ) : null}
          </div>
        ) : null}

        {ready && selected && info ? (
          <>
            {!server?.details.find((d) => d.match_id === selected.id)?.has_events && !files.events ? (
              <p className="mb-3 text-xs text-amber-300">This match has no events file on the server: player names, passes, VAEP and OBV are limited. Load the events file in Match Replay to add it.</p>
            ) : null}
            <div role="tabpanel" className={tab === 'core' ? '' : 'hidden'}>
              {visited.core ? <Tab1CoreMetrics key={`core-${selected.id}-${version}`} info={info} /> : null}
            </div>
            <div role="tabpanel" className={tab === 'live' ? '' : 'hidden'}>
              {visited.live ? <Tab3LivePlayer key={`live-${selected.id}-${version}`} info={info} /> : null}
            </div>
            <div role="tabpanel" className={tab === 'players' ? '' : 'hidden'}>
              {visited.players ? <Tab2PlayerAnalytics key={`players-${selected.id}-${version}`} info={info} /> : null}
            </div>
            <div role="tabpanel" className={tab === 'attacks' ? '' : 'hidden'}>
              {visited.attacks ? <Tab4TopAttacks key={`attacks-${selected.id}-${version}`} info={info} /> : null}
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
}
