import type {
  EventDataset,
  EventRosterPlayer,
  GoalCandidate,
  MatchEvent,
  PlayerTechnicalStats,
  ScoreValidation
} from './types';

type UnknownRecord = Record<string, unknown>;

function object(value: unknown): UnknownRecord {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as UnknownRecord : {};
}

function text(value: unknown): string | null {
  if (typeof value === 'string' && value.length > 0) return value;
  if (typeof value === 'number' && Number.isFinite(value)) return String(value);
  return null;
}

function number(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function increment(counts: Record<string, number>, key: string | null): void {
  const safe = key ?? '(none)';
  counts[safe] = (counts[safe] ?? 0) + 1;
}

interface MutableRoster extends EventRosterPlayer {
  speedTotal: number;
}

function playerList(record: UnknownRecord, key: 'homePlayers' | 'awayPlayers'): UnknownRecord[] {
  const value = record[key];
  return Array.isArray(value) ? value.map(object) : [];
}

export function parseEventDataset(source: string): EventDataset {
  const parsed: unknown = JSON.parse(source);
  if (!Array.isArray(parsed)) throw new Error('Events file must be a JSON array.');

  const roster = new Map<string, MutableRoster>();
  for (const raw of parsed) {
    const record = object(raw);
    for (const [team, key] of [[0, 'homePlayers'], [1, 'awayPlayers']] as const) {
      for (const player of playerList(record, key)) {
        const playerId = text(player.playerId);
        const jerseyNum = number(player.jerseyNum);
        if (!playerId || !jerseyNum) continue;
        const existing = roster.get(playerId);
        const speed = number(player.speed, Number.NaN);
        if (!existing) {
          roster.set(playerId, {
            playerId,
            playerName: '',
            team,
            teamId: null,
            teamName: null,
            jerseyNum,
            position: text(player.positionGroupType),
            eventSpeedAverageMps: Number.isFinite(speed) ? speed : 0,
            eventSpeedMaxMps: Number.isFinite(speed) ? speed : 0,
            eventSpeedSamples: Number.isFinite(speed) ? 1 : 0,
            speedTotal: Number.isFinite(speed) ? speed : 0
          });
        } else if (Number.isFinite(speed)) {
          existing.speedTotal += speed;
          existing.eventSpeedSamples += 1;
          existing.eventSpeedMaxMps = Math.max(existing.eventSpeedMaxMps, speed);
        }
      }
    }
    const game = object(record.gameEvents);
    const involvedId = text(game.playerId);
    const involved = involvedId ? roster.get(involvedId) : undefined;
    if (involved) {
      involved.playerName = text(game.playerName) ?? involved.playerName;
      involved.teamId = text(game.teamId) ?? involved.teamId;
      involved.teamName = text(game.teamName) ?? involved.teamName;
    }
  }

  const playerStats = new Map<string, PlayerTechnicalStats>();
  const ensureStats = (playerId: string | null, playerName: string | null): PlayerTechnicalStats | null => {
    if (!playerId) return null;
    let stats = playerStats.get(playerId);
    const rosterPlayer = roster.get(playerId);
    if (!stats) {
      stats = {
        playerId,
        playerName: playerName ?? rosterPlayer?.playerName ?? `Player ${playerId}`,
        team: rosterPlayer?.team ?? 0,
        jerseyNum: rosterPlayer?.jerseyNum ?? 0,
        position: rosterPlayer?.position ?? null,
        involvements: 0,
        passes: 0,
        completedPasses: 0,
        shots: 0,
        goalOutcomes: 0,
        crosses: 0,
        challenges: 0,
        clearances: 0,
        rebounds: 0,
        carries: 0,
        touches: 0,
        fouls: 0
      };
      playerStats.set(playerId, stats);
    } else if (!stats.playerName && playerName) {
      stats.playerName = playerName;
    }
    return stats;
  };

  const events: MatchEvent[] = [];
  const gameTypeCounts: Record<string, number> = {};
  const possessionTypeCounts: Record<string, number> = {};
  const teamEventCounts: Record<string, number> = {};
  const goalCandidates: GoalCandidate[] = [];

  for (const raw of parsed) {
    const record = object(raw);
    const game = object(record.gameEvents);
    const possession = object(record.possessionEvents);
    const gameType = text(game.gameEventType);
    const possessionType = text(possession.possessionEventType);
    const teamName = text(game.teamName);
    increment(gameTypeCounts, gameType);
    increment(possessionTypeCounts, possessionType);
    increment(teamEventCounts, teamName);

    const playerId = text(game.playerId);
    const playerName = text(game.playerName);
    const playerOffId = text(game.playerOffId);
    const playerOffName = text(game.playerOffName);
    const playerOnId = text(game.playerOnId);
    const playerOnName = text(game.playerOnName);
    const passerId = text(possession.passerPlayerId);
    const passerName = text(possession.passerPlayerName);
    const targetId = text(possession.targetPlayerId);
    const targetName = text(possession.targetPlayerName);
    const receiverId = text(possession.receiverPlayerId);
    const receiverName = text(possession.receiverPlayerName);
    const mainStats = ensureStats(playerId, playerName);
    if (mainStats) mainStats.involvements += 1;

    const actorId = passerId
      ?? text(possession.shooterPlayerId)
      ?? text(possession.crosserPlayerId)
      ?? text(possession.challengerPlayerId)
      ?? text(possession.clearerPlayerId)
      ?? text(possession.rebounderPlayerId)
      ?? text(possession.ballCarrierPlayerId)
      ?? text(possession.touchPlayerId)
      ?? playerId;
    const actorName = passerName
      ?? text(possession.shooterPlayerName)
      ?? text(possession.crosserPlayerName)
      ?? text(possession.challengerPlayerName)
      ?? text(possession.clearerPlayerName)
      ?? text(possession.rebounderPlayerName)
      ?? text(possession.ballCarrierPlayerName)
      ?? text(possession.touchPlayerName)
      ?? playerName;
    const actor = ensureStats(actorId, actorName);
    if (actor) {
      if (possessionType === 'PA') {
        actor.passes += 1;
        if (text(possession.passOutcomeType) === 'C') actor.completedPasses += 1;
      } else if (possessionType === 'SH') {
        actor.shots += 1;
        if (text(possession.shotOutcomeType) === 'G') actor.goalOutcomes += 1;
      } else if (possessionType === 'CR') actor.crosses += 1;
      else if (possessionType === 'CH') actor.challenges += 1;
      else if (possessionType === 'CL') actor.clearances += 1;
      else if (possessionType === 'RE') actor.rebounds += 1;
      else if (possessionType === 'BC') actor.carries += 1;
      else if (possessionType === 'TC' || possessionType === 'IT') actor.touches += 1;
      else if (possessionType === 'FO') actor.fouls += 1;
    }

    const eventTime = number(record.eventTime, number(record.startTime));
    const startTime = number(record.startTime, eventTime);
    const endTime = number(record.endTime, eventTime);
    const outcome = possessionType === 'PA' ? text(possession.passOutcomeType)
      : possessionType === 'SH' ? text(possession.shotOutcomeType)
      : possessionType === 'CR' ? text(possession.crossOutcomeType)
      : possessionType === 'CH' ? text(possession.challengeOutcomeType)
      : null;
    const eventId = `${text(record.gameEventId) ?? 'event'}:${text(record.possessionEventId) ?? 'none'}:${eventTime}:${events.length}`;
    if (possessionType === 'SH' && outcome === 'G') {
      goalCandidates.push({
        eventId,
        eventTimeMs: eventTime * 1000,
        clock: text(possession.formattedGameClock) ?? text(game.startFormattedGameClock) ?? '--:--',
        period: number(game.period),
        teamId: text(game.teamId),
        teamName,
        homeTeam: typeof game.homeTeam === 'boolean' ? game.homeTeam : null,
        shooterName: actorName
      });
    }
    events.push({
      id: eventId,
      eventTimeMs: eventTime * 1000,
      startTimeMs: startTime * 1000,
      endTimeMs: endTime * 1000,
      clock: text(possession.formattedGameClock) ?? text(game.startFormattedGameClock) ?? '--:--',
      period: number(game.period),
      gameType,
      possessionType,
      teamId: text(game.teamId),
      teamName,
      homeTeam: typeof game.homeTeam === 'boolean' ? game.homeTeam : null,
      homeBall: typeof game.homeBall === 'boolean' ? game.homeBall
        : typeof game.home_ball === 'boolean' ? game.home_ball : null,
      playerId,
      playerName,
      playerOffId,
      playerOffName,
      playerOnId,
      playerOnName,
      passerPlayerId: passerId,
      passerPlayerName: passerName,
      targetPlayerId: targetId,
      targetPlayerName: targetName,
      receiverPlayerId: receiverId,
      receiverPlayerName: receiverName,
      actorPlayerId: actorId,
      actorPlayerName: actorName,
      outcome,
      payload: record
    });
  }

  const rosterResult = [...roster.values()].map(({ speedTotal, ...player }) => ({
    ...player,
    playerName: player.playerName || `Player ${player.playerId}`,
    eventSpeedAverageMps: player.eventSpeedSamples > 0 ? speedTotal / player.eventSpeedSamples : 0
  }));

  return {
    gameId: parsed.length > 0 ? text(object(parsed[0]).gameId) : null,
    events: events.sort((a, b) => a.eventTimeMs - b.eventTimeMs),
    roster: rosterResult.sort((a, b) => a.team - b.team || a.jerseyNum - b.jerseyNum),
    playerStats: [...playerStats.values()].sort((a, b) => b.involvements - a.involvements),
    gameTypeCounts,
    possessionTypeCounts,
    teamEventCounts,
    goalCandidates
  };
}

const KNOWN_FINAL_SCORES: Record<string, readonly [number, number]> = {
  '3816': [1, 2]
};

export function validateEventScore(dataset: EventDataset): ScoreValidation {
  const expected = dataset.gameId ? KNOWN_FINAL_SCORES[dataset.gameId] : undefined;
  const observedHome = dataset.goalCandidates.filter((goal) => goal.homeTeam === true).length;
  const observedAway = dataset.goalCandidates.filter((goal) => goal.homeTeam === false).length;
  return {
    status: expected ? (observedHome === expected[0] && observedAway === expected[1] ? 'matched' : 'mismatch') : 'unknown',
    expectedHome: expected?.[0] ?? null,
    expectedAway: expected?.[1] ?? null,
    observedHome,
    observedAway,
    goalCandidates: dataset.goalCandidates
  };
}

export function playerOnPitchAt(events: MatchEvent[], playerId: string, timeMs: number): boolean {
  const substitutions = events.filter((event) => event.gameType === 'SUB'
    && (event.playerOnId === playerId || event.playerOffId === playerId));
  let onPitch = !substitutions.some((event) => event.playerOnId === playerId);
  for (const event of substitutions) {
    if (event.eventTimeMs > timeMs) break;
    if (event.playerOffId === playerId) onPitch = false;
    if (event.playerOnId === playerId) onPitch = true;
  }
  return onPitch;
}
