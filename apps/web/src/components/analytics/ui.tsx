import type { ReactNode } from 'react';

export function Panel({
  title,
  subtitle,
  right,
  children,
  className = ''
}: {
  title: string;
  subtitle: string;
  right: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`rounded-2xl border border-line bg-panel p-4 shadow-panel ${className}`}>
      <div className="mb-3 flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <h3 className="text-sm font-bold uppercase tracking-[0.14em] text-white">{title}</h3>
          {subtitle ? <p className="mt-0.5 text-xs text-muted">{subtitle}</p> : null}
        </div>
        {right}
      </div>
      {children}
    </section>
  );
}

export function Stat({ label, value, hint, help }: { label: string; value: string; hint: string; help?: string }) {
  return (
    <div className="rounded-xl border border-line bg-ink/60 px-3 py-2.5" title={help}>
      <div className="flex items-center justify-between gap-1 text-[11px] font-semibold uppercase tracking-wider text-muted">
        <span>{label}</span>
        {help ? <span aria-hidden="true" className="grid h-3.5 w-3.5 place-items-center rounded-full border border-line text-[9px] normal-case">?</span> : null}
      </div>
      <div className="mt-0.5 font-mono text-lg font-bold tabular-nums text-white">{value}</div>
      {hint ? <div className="mt-0.5 text-[11px] leading-snug text-muted">{hint}</div> : null}
    </div>
  );
}

export function ErrorBox({ message }: { message: string }) {
  return (
    <div role="alert" className="rounded-xl border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-200">
      {message}
    </div>
  );
}

export function Spinner({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-muted" role="status">
      <span className="h-3 w-3 animate-spin rounded-full border-2 border-lime border-t-transparent" aria-hidden="true" />
      {label}
    </div>
  );
}
