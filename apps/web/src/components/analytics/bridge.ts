// A tiny shared "mailbox" between the Match Replay (App) and the Advanced Analytics tab.
//
// The replay parses the files inside the browser and never sends them anywhere, so the Python analytics
// server cannot see them. The replay therefore announces the File objects it was given here (one line per file
// type in App.tsx) and the analytics tab sends them to the server when it needs them.
//
// Isolated on purpose: this module imports nothing from the rest of the app.

export interface MatchFiles {
  tracking: File | null;
  events: File | null;
  metadata: File | null;
}

type Listener = (files: MatchFiles) => void;

const EMPTY: MatchFiles = { tracking: null, events: null, metadata: null };
let current: MatchFiles = EMPTY;
const listeners = new Set<Listener>();

function emit(): void {
  for (const listener of listeners) listener(current);
}

/** Called by the replay (or by the analytics drop zones) whenever the user opens a file. */
export function announceMatchFile(kind: keyof MatchFiles, file: File): void {
  current = { ...current, [kind]: file };
  emit();
}

/** Forget every announced file (the analytics tab's "change match" button). */
export function clearMatchFiles(): void {
  current = EMPTY;
  emit();
}

export function getMatchFiles(): MatchFiles {
  return current;
}

export function subscribeMatchFiles(listener: Listener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}
