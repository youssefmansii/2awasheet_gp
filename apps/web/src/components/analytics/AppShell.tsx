import { BarChart3, Play } from 'lucide-react';
import { useState, type ReactNode } from 'react';
import { App } from '../../App';
import { AnalyticsView } from './AnalyticsView';

type View = 'replay' | 'analytics';

/**
 * Wraps the existing <App /> without touching it.
 * Both views stay mounted (the inactive one is just hidden) so the replay keeps the
 * loaded tracking data and the analytics keep their state when you switch back and forth.
 */
export function AppShell() {
  const [view, setView] = useState<View>('replay');
  const [analyticsOpened, setAnalyticsOpened] = useState(false);

  const go = (v: View) => {
    setView(v);
    if (v === 'analytics') setAnalyticsOpened(true);
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
        <div className="mx-auto flex max-w-[1480px] items-center gap-2 px-4 py-2 sm:px-6 lg:px-8">
          {tab('replay', 'Match Replay', <Play className="h-4 w-4" aria-hidden="true" />)}
          {tab('analytics', 'Advanced Analytics', <BarChart3 className="h-4 w-4" aria-hidden="true" />)}
        </div>
      </nav>
      <div className={view === 'replay' ? '' : 'hidden'}>
        <App />
      </div>
      {analyticsOpened ? (
        <div className={view === 'analytics' ? '' : 'hidden'}>
          <AnalyticsView />
        </div>
      ) : null}
    </>
  );
}
