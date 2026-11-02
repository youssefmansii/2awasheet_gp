import { Activity, BarChart3, Play } from 'lucide-react';
import { lazy, Suspense, useState, type ReactNode } from 'react';
import { App } from '../../App';
const AnalyticsView = lazy(() => import('./AnalyticsView').then((module) => ({ default: module.AnalyticsView })));
const PossessionDemo = lazy(() => import('../possession/PossessionDemo').then((module) => ({ default: module.PossessionDemo })));

type View = 'replay' | 'analytics' | 'possession';

/**
 * Wraps the existing <App /> without touching it.
 * Both views stay mounted (the inactive one is just hidden) so the replay keeps the
 * loaded tracking data and the analytics keep their state when you switch back and forth.
 */
export function AppShell() {
  const initialView = new URLSearchParams(window.location.search).get('view') === 'possession' ? 'possession' : 'replay';
  const [view, setView] = useState<View>(initialView);
  const [analyticsOpened, setAnalyticsOpened] = useState(false);
  const [demoOpened, setDemoOpened] = useState(initialView === 'possession');

  const go = (v: View) => {
    setView(v);
    if (v === 'analytics') setAnalyticsOpened(true);
    if (v === 'possession') setDemoOpened(true);
    const url = new URL(window.location.href);
    if (v === 'possession') url.searchParams.set('view', 'possession');
    else url.searchParams.delete('view');
    window.history.replaceState(null, '', url);
  };

  const tab = (v: View, label: string, icon: ReactNode) => (
    <button
      type="button"
      aria-pressed={view === v}
      onClick={() => go(v)}
      className={`flex min-h-10 items-center gap-2 rounded-lg px-4 text-sm font-semibold transition ${
        view === v ? 'bg-lime text-ink' : 'text-muted hover:text-white'
      }`}
    >
      {icon}
      {label}
    </button>
  );

  return (
    <>
      <nav aria-label="Main views" className="sticky top-0 z-40 border-b border-line bg-[#080b0d]/95 backdrop-blur">
        <div className="mx-auto flex max-w-[1480px] flex-wrap items-center gap-2 px-4 py-2 sm:px-6 lg:px-8">
          {tab('replay', 'Match Replay', <Play className="h-4 w-4" aria-hidden="true" />)}
          {tab('analytics', 'Advanced Analytics', <BarChart3 className="h-4 w-4" aria-hidden="true" />)}
          {tab('possession', 'Possession Demo', <Activity className="h-4 w-4" aria-hidden="true" />)}
        </div>
      </nav>
      <div className={view === 'replay' ? '' : 'hidden'}>
        <App />
      </div>
      {analyticsOpened ? (
        <div className={view === 'analytics' ? '' : 'hidden'}>
          <Suspense fallback={<p role="status" className="p-6 text-muted">Loading analytics…</p>}>
            <AnalyticsView />
          </Suspense>
        </div>
      ) : null}
      {demoOpened ? <div className={view === 'possession' ? '' : 'hidden'}>
        <Suspense fallback={<p role="status" className="p-6 text-muted">Loading possession demo…</p>}>
          <PossessionDemo active={view === 'possession'} />
        </Suspense>
      </div> : null}
    </>
  );
}
