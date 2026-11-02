import { create } from 'zustand';

export type PositionMode = 'smoothed' | 'raw';

interface UiState {
  speed: number;
  showNumbers: boolean;
  showBall: boolean;
  interpolate: boolean;
  positionMode: PositionMode;
  homeColor: string;
  homeTextColor: string;
  awayColor: string;
  awayTextColor: string;
  flipSides: boolean;
  includeEverything: boolean;
  trailLength: number;
  selectedPlayer: string | null;
  set: (patch: Partial<Omit<UiState, 'set'>>) => void;
}

export const useUiStore = create<UiState>((set) => ({
  speed: 1,
  showNumbers: true,
  showBall: true,
  interpolate: true,
  positionMode: 'smoothed',
  homeColor: '#ff6b1a',
  homeTextColor: '#111518',
  awayColor: '#f7f8fa',
  awayTextColor: '#0f172a',
  flipSides: false,
  includeEverything: false,
  trailLength: 0,
  selectedPlayer: null,
  set
}));
