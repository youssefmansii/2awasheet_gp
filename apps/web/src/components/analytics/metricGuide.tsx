import { useState } from 'react';

export type LayerKey = 'pc' | 'das' | 'obso' | 'epv' | 'vaep' | 'obv';

export interface GuideEntry {
  key: LayerKey;
  name: string;
  dot: string;
  /** which family it belongs to */
  group: 'space' | 'value';
  /** the one question the metric answers, in plain English */
  question: string;
  /** how to read the picture and the number */
  read: string;
  /** what a coach / analyst should look for */
  lookFor: string;
  /** honest limit */
  careful: string;
}

export const GUIDE: GuideEntry[] = [
  {
    key: 'pc',
    name: 'Pitch Control',
    dot: '#ff6b1a',
    group: 'space',
    question: 'Which team would get to the ball first if it were played to this spot?',
    read:
      'Every square of the pitch is coloured by the team that would arrive first (using each player\'s position, speed and direction). A strong colour means clear control, a faint colour means a contested square. The percentage is the share of the pitch one team controls.',
    lookFor: 'A big patch of one colour near the opponent\'s goal means that team owns the space where chances are made. A hole in a defence shows up as the attackers\' colour inside the defenders\' half.',
    careful: 'It says who could get there first, not who will. It is computed from player positions only, so it also works when the ball is not tracked.'
  },
  {
    key: 'das',
    name: 'DAS',
    dot: '#c8f45a',
    group: 'space',
    question: 'How much DANGEROUS space can the attackers really reach with a pass right now?',
    read:
      'The ball carrier\'s possible passes are simulated. Bright squares are places where an attacker would win the ball before any defender, counted more when a shot from there is more likely to score. The number is danger-weighted square metres (higher = more dangerous passes are open).',
    lookFor: 'Bright space behind or between the defenders and close to goal means the defence can be broken now. A dark pitch means the defence has closed everything down.',
    careful: 'Needs the ball position, so it is blank on dead-ball frames. Implemented with the open-source accessible-space package.'
  },
  {
    key: 'obso',
    name: 'OBSO',
    dot: '#facc15',
    group: 'space',
    question: 'Which attackers WITHOUT the ball are standing where a pass could create a goal?',
    read:
      'For each square: can the attackers get there (pitch control) × can the ball realistically get there (transition) × how valuable is that square (EPV). Yellow squares are the valuable-and-reachable ones. A ring around a player shows his own value (the yellow number is shown for the top three); the card shows the team total.',
    lookFor: 'A big ring on a free winger or striker = the best pass option. If the ring is on someone marked by two defenders, the model is telling you the space is valuable but the defenders will close it.',
    careful: 'Not a probability. Compare values within the same match. The ball carrier is left out because OBSO is about off-ball players. Needs the ball. Implemented as C-OBSO.'
  },
  {
    key: 'epv',
    name: 'EPV',
    dot: '#38bdf8',
    group: 'value',
    question: 'How valuable is the ball\'s current location for the team that has it?',
    read:
      'A value map learned from real matches: roughly the chance the team in possession scores next from the ball position. 0.01 means about 1 %; the middle of the pitch is tiny, the area just in front of goal is large. The line chart shows the home team\'s value over the match: above 0 means home has the more valuable possession.',
    lookFor: 'Peaks in the chart are dangerous attacks. Click a peak to jump to it and watch what happened.',
    careful: 'It only looks at where the ball is, not at the defenders. Pair it with Pitch Control / DAS for the full picture. Blank when the ball is not tracked.'
  },
  {
    key: 'vaep',
    name: 'VAEP',
    dot: '#4ade80',
    group: 'value',
    question: 'How much did this action change the team\'s chance of scoring?',
    read:
      'For every pass, dribble, shot and so on: value of the situation after the action minus the value before it. Green = the action made a goal more likely, red = it made it less likely (a lost ball in a good position, for example). The circle size is the size of the change.',
    lookFor: 'Big green circles are the actions that really moved the attack forward. Add them up per player to see who creates value.',
    careful: 'Judges the OUTCOME that happened, so a good pass that was unluckily intercepted counts as bad. That is what OBV below corrects. A goal is worth +1 only when the restart follows.'
  },
  {
    key: 'obv',
    name: 'OBV',
    dot: '#c084fc',
    group: 'value',
    question: 'Was the action a good decision, ignoring luck?',
    read:
      'Same idea as VAEP but uses what was EXPECTED: the chance the pass succeeds × the value if it works + the chance it fails × the value if it fails, minus the value before. So a risky pass that usually fails scores lower than a safe one, even when both happened to work.',
    lookFor: 'Compare OBV with VAEP: when VAEP is much lower than OBV the player was unlucky; much higher means he got away with it.',
    careful: 'Our own expected-outcome variant (a simple pass-success model, AUC 0.77), not the proprietary StatsBomb OBV. Penalties are valued at 0.79 xG.'
  }
];

const BY_KEY: Record<LayerKey, GuideEntry> = Object.fromEntries(GUIDE.map((g) => [g.key, g])) as Record<LayerKey, GuideEntry>;

const STORE_KEY = 'analytics.objective.collapsed';

function readCollapsed(): boolean {
  try {
    return window.localStorage.getItem(STORE_KEY) === '1';
  } catch {
    return false;
  }
}

function writeCollapsed(v: boolean) {
  try {
    window.localStorage.setItem(STORE_KEY, v ? '1' : '0');
  } catch {
    /* storage blocked: the card just opens again next time */
  }
}

/** "What is this page for?" - the objective of the whole page in four lines. */
export function ObjectiveCard() {
  const [collapsed, setCollapsed] = useState<boolean>(() => readCollapsed());
  const toggle = () => {
    setCollapsed((c) => {
      writeCollapsed(!c);
      return !c;
    });
  };
  return (
    <section className="rounded-2xl border border-line bg-panel px-4 py-3" aria-label="What this page is for">
      <button type="button" onClick={toggle} aria-expanded={!collapsed} className="flex w-full items-center justify-between gap-2 text-left">
        <span className="text-sm font-bold uppercase tracking-[0.14em] text-white">What is this page for?</span>
        <span className="text-xs font-semibold text-lime">{collapsed ? 'Show' : 'Hide'}</span>
      </button>
      {collapsed ? null : (
        <div className="mt-2 space-y-3 text-sm leading-relaxed text-slate-200">
          <p>
            Tracking data tells us where all 22 players and the ball are, 30 times a second. These six metrics turn that into the questions a coach asks:
            <span className="text-white"> who owns the space, is that space dangerous, who is free to score, and was each action a good idea?</span>
          </p>
          <div className="grid gap-3 md:grid-cols-2">
            <div className="rounded-xl border border-line bg-ink/50 p-3">
              <div className="text-xs font-bold uppercase tracking-wider text-lime">Space and movement (before the action)</div>
              <ul className="mt-1.5 space-y-1 text-xs text-slate-300">
                {GUIDE.filter((g) => g.group === 'space').map((g) => (
                  <li key={g.key}>
                    <span className="font-semibold" style={{ color: g.dot }}>{g.name}</span> · {g.question}
                  </li>
                ))}
              </ul>
            </div>
            <div className="rounded-xl border border-line bg-ink/50 p-3">
              <div className="text-xs font-bold uppercase tracking-wider text-lime">Value of the ball and of actions (after the action)</div>
              <ul className="mt-1.5 space-y-1 text-xs text-slate-300">
                {GUIDE.filter((g) => g.group === 'value').map((g) => (
                  <li key={g.key}>
                    <span className="font-semibold" style={{ color: g.dot }}>{g.name}</span> · {g.question}
                  </li>
                ))}
              </ul>
            </div>
          </div>
          <p className="text-xs text-muted">
            Switch a metric on above and a short guide for it appears next to the pitch. The ball is tracked only while it is in play (about half of the match), so the metrics that need the ball are blank in between: the green strip under the slider marks the live play.
          </p>
        </div>
      )}
    </section>
  );
}

/** Guide cards for the layers that are switched on. */
export function MetricGuide({ active }: { active: Record<LayerKey, boolean> }) {
  const shown = GUIDE.filter((g) => active[g.key]);
  if (!shown.length) {
    return <p className="rounded-xl border border-dashed border-line px-3 py-2 text-xs text-muted">Switch a metric on above to see what it means and how to read it.</p>;
  }
  return (
    <div className="space-y-2" aria-label="Metric guide">
      {shown.map((g) => {
        const e = BY_KEY[g.key];
        return (
          <details key={g.key} open={shown.length <= 2} className="rounded-xl border border-line bg-panel px-3 py-2">
            <summary className="cursor-pointer list-none">
              <div className="flex items-center gap-2">
                <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: e.dot }} aria-hidden="true" />
                <span className="text-xs font-bold uppercase tracking-wider text-white">{e.name}</span>
              </div>
              <p className="mt-1 text-xs font-semibold leading-snug text-slate-100">{e.question}</p>
            </summary>
            <dl className="mt-2 space-y-1.5 text-[11px] leading-snug text-slate-300">
              <div>
                <dt className="font-bold text-slate-100">How to read it</dt>
                <dd>{e.read}</dd>
              </div>
              <div>
                <dt className="font-bold text-slate-100">What to look for</dt>
                <dd>{e.lookFor}</dd>
              </div>
              <div>
                <dt className="font-bold text-slate-100">Be careful</dt>
                <dd>{e.careful}</dd>
              </div>
            </dl>
          </details>
        );
      })}
    </div>
  );
}
