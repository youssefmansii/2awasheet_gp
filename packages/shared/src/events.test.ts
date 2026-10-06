import { describe, expect, it } from 'vitest';
import { parseEventDataset, playerOnPitchAt, validateEventScore } from './events';

describe('parseEventDataset', () => {
  it('indexes events, roster identities, technical stats, and raw payloads', () => {
    const source = JSON.stringify([{
      gameId: 3816,
      gameEventId: 1,
      possessionEventId: 2,
      eventTime: 101.969,
      startTime: 101.969,
      endTime: 102.2,
      gameEvents: { gameEventType: 'OTB', period: 1, teamId: 382, teamName: 'Saudi Arabia', homeTeam: false, playerId: 11, playerName: 'Passer' },
      possessionEvents: { possessionEventType: 'PA', formattedGameClock: '00:00', passerPlayerId: 11, passerPlayerName: 'Passer', targetPlayerId: 12, targetPlayerName: 'Target', passOutcomeType: 'C' },
      homePlayers: [],
      awayPlayers: [{ playerId: 11, jerseyNum: 9, positionGroupType: 'CF', speed: 4.2 }],
      customUnknown: { preserved: true }
    }]);
    const result = parseEventDataset(source);
    expect(result.gameId).toBe('3816');
    expect(result.events[0]?.eventTimeMs).toBe(101969);
    expect(result.events[0]?.payload.customUnknown).toEqual({ preserved: true });
    expect(result.roster[0]?.playerName).toBe('Passer');
    expect(result.playerStats[0]?.passes).toBe(1);
    expect(result.playerStats[0]?.completedPasses).toBe(1);
  });

  it('rejects non-array event files', () => {
    expect(() => parseEventDataset('{}')).toThrow('JSON array');
  });

  it('tracks player on-pitch state through substitutions', () => {
    const source = JSON.stringify([{
      gameId: 1,
      eventTime: 600,
      gameEvents: { gameEventType: 'SUB', playerOffId: 10, playerOffName: 'Starter', playerOnId: 20, playerOnName: 'Substitute' },
      possessionEvents: {},
      homePlayers: [{ playerId: 10, jerseyNum: 10 }, { playerId: 20, jerseyNum: 20 }],
      awayPlayers: []
    }]);
    const result = parseEventDataset(source);
    expect(result.events[0]?.playerOffId).toBe('10');
    expect(result.events[0]?.playerOnId).toBe('20');
    expect(playerOnPitchAt(result.events, '10', 599_000)).toBe(true);
    expect(playerOnPitchAt(result.events, '10', 600_000)).toBe(false);
    expect(playerOnPitchAt(result.events, '20', 599_000)).toBe(false);
    expect(playerOnPitchAt(result.events, '20', 600_000)).toBe(true);
  });

  it('validates raw G shot outcomes without forcing the known score', () => {
    const event = (homeTeam: boolean, eventTime: number) => ({
      gameId: 3816, eventTime,
      gameEvents: { gameEventType: 'SHOT', period: 1, homeTeam, teamName: homeTeam ? 'Argentina' : 'Saudi Arabia' },
      possessionEvents: { possessionEventType: 'SH', shotOutcomeType: 'G', formattedGameClock: '10:00', shooterPlayerName: 'Player' },
      homePlayers: [], awayPlayers: []
    });
    const result = validateEventScore(parseEventDataset(JSON.stringify([
      event(true, 600), event(true, 1200), event(false, 1800), event(false, 2400)
    ])));
    expect(result.observedHome).toBe(2);
    expect(result.observedAway).toBe(2);
    expect(result.expectedHome).toBe(1);
    expect(result.expectedAway).toBe(2);
    expect(result.status).toBe('mismatch');
  });
});
