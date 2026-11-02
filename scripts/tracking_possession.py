"""Offline, tracking-only possession baseline and sparse event-label validation.

The predictor never reads embedded events, event ids, rosters or event snapshots.
The separate validator reads the event file AFTER all predictions are frozen.
No fitted parameters, gap interpolation or whole-match smoothing is performed.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import orjson
    loads = orjson.loads
except ImportError:
    loads = json.loads


def number(value, default=math.nan):
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


@dataclass
class PossessionTracker:
    source: str = 'smoothed'
    radius_m: float = 2.0
    exit_radius_m: float = 2.5
    margin_m: float = .5
    max_height_m: float = 2.5
    max_relative_speed_mps: float = 15.
    confirm_s: float = .1
    flight_hold_s: float = 3.
    max_gap_s: float = .2
    pitch_length_m: float = 105.
    pitch_width_m: float = 68.

    def __post_init__(self):
        self.previous = None
        self.previous_positions = {}
        self.previous_ball = None
        self.pending = ''
        self.pending_since = 0.
        self.last_carrier = ''
        self.last_control_time = -math.inf

    def clear(self):
        self.pending = ''
        self.last_carrier = ''
        self.last_control_time = -math.inf

    def update(self, frame):
        t = number(frame.get('videoTimeMs')) / 1000
        period = int(frame.get('period') or 0)
        if not math.isfinite(t):
            raise ValueError('Tracking frame requires a finite videoTimeMs')
        dt = t - self.previous[0] if self.previous else math.inf
        continuity = (self.previous is not None and self.previous[1] == period
                      and 0 < dt <= self.max_gap_s)
        if not continuity:
            self.clear()
            self.previous_positions = {}
            self.previous_ball = None
        row = dict(frame_num=int(frame['frameNum']), video_time_s=t, period=period,
                   carrier='', team='', status='missing_ball', nearest='',
                   nearest_distance_m=math.nan, ball_speed_mps=math.nan,
                   relative_speed_mps=math.nan, evidence_score=0.,
                   ball_x=math.nan, ball_y=math.nan, ball_z=math.nan)
        distances, positions, qualities = [], {}, {}
        confirming_candidate = False
        suffix = 'Smoothed' if self.source == 'smoothed' else ''
        ball = frame.get('ballsSmoothed' if suffix else 'balls')
        if isinstance(ball, list):
            ball = ball[0] if len(ball) == 1 else None  # no arbitrary multi-ball selection
        bx, by, bz = [number((ball or {}).get(k)) for k in ['x', 'y', 'z']]
        valid_ball = bool(ball) and math.isfinite(bx) and math.isfinite(by)
        for side in ['home', 'away']:
            for p in frame.get(side + 'Players' + suffix) or []:
                jersey = p.get('jerseyNum')
                x, y = number(p.get('x')), number(p.get('y'))
                if jersey is None or not (math.isfinite(x) and math.isfinite(y)):
                    continue
                key = f'{side}:{jersey}'
                positions[key] = (x, y)
                qualities[key] = (p.get('confidence'), p.get('visibility'))
                if valid_ball:
                    distances.append((math.hypot(x - bx, y - by), key))
        distances.sort()
        if distances:
            row['nearest_distance_m'], row['nearest'] = distances[0]
        if valid_ball:
            row.update(ball_x=bx, ball_y=by, ball_z=bz)
            vx = vy = math.nan
            if continuity and self.previous_ball:
                vx, vy = (bx - self.previous_ball[0]) / dt, (by - self.previous_ball[1]) / dt
                row['ball_speed_mps'] = math.hypot(vx, vy)
            if period <= 0:
                row['status'] = 'outside_period'
            elif abs(bx) > self.pitch_length_m / 2 or abs(by) > self.pitch_width_m / 2:
                row['status'] = 'out_of_bounds_candidate'
            elif (ball or {}).get('visibility') != 'VISIBLE':
                row['status'] = 'low_quality_ball'
            elif row['ball_speed_mps'] > 45:
                row['status'] = 'ball_tracking_jump'
            elif math.isfinite(bz) and bz > self.max_height_m:
                row['status'] = 'airborne'
            elif not distances:
                row['status'] = 'missing_players'
            else:
                dist, candidate = distances[0]
                second = distances[1][0] if len(distances) > 1 else math.inf
                confidence, visibility = qualities[candidate]
                radius = self.exit_radius_m if candidate == self.last_carrier else self.radius_m
                if dist > radius:
                    row['status'] = 'loose_ball'
                elif second - dist < self.margin_m:
                    row['status'] = ('contested' if len(distances) > 1 and
                                     candidate.split(':')[0] != distances[1][1].split(':')[0]
                                     else 'ambiguous_player')
                elif confidence not in ('HIGH', 'MEDIUM') or visibility != 'VISIBLE':
                    row['status'] = 'low_quality_player'
                else:
                    previous_pos = self.previous_positions.get(candidate)
                    if previous_pos and math.isfinite(vx):
                        px, py = positions[candidate]
                        pvx, pvy = (px - previous_pos[0]) / dt, (py - previous_pos[1]) / dt
                        player_speed = math.hypot(pvx, pvy)
                        row['relative_speed_mps'] = math.hypot(vx - pvx, vy - pvy)
                    else:
                        player_speed = 0.
                    if player_speed > 12.5:
                        row['status'] = 'player_tracking_jump'
                    elif row['relative_speed_mps'] > self.max_relative_speed_mps:
                        row['status'] = 'fast_ball'
                    else:
                        confirming_candidate = True
                        if candidate != self.pending:
                            self.pending, self.pending_since = candidate, t
                        if t - self.pending_since + 1e-8 >= self.confirm_s:
                            row.update(carrier=candidate, team=candidate.split(':')[0], status='controlled')
                            # A geometric evidence score, explicitly NOT a probability.
                            row['evidence_score'] = max(0., 1 - dist / (radius + .5)) * min(1., (second - dist) / 2.)
                            self.last_carrier, self.last_control_time = candidate, t
                        else:
                            row['status'] = 'pending_control'
            if row['status'] in ('loose_ball', 'fast_ball', 'airborne', 'pending_control'):
                if (self.last_carrier and t - self.last_control_time <= self.flight_hold_s
                        and row['ball_speed_mps'] > 2.):
                    row.update(team=self.last_carrier.split(':')[0], status='in_flight_provisional')
        if not confirming_candidate:
            self.pending = ''
        if row['status'] in ('missing_ball', 'outside_period', 'out_of_bounds_candidate',
                             'low_quality_ball', 'ball_tracking_jump', 'missing_players',
                             'contested', 'ambiguous_player', 'low_quality_player', 'player_tracking_jump'):
            self.clear()
        self.previous = (t, period)
        self.previous_positions = positions
        self.previous_ball = (bx, by) if valid_ball else None
        return row


def durations(rows, max_gap_s=.2):
    """Attribute [t_i,t_(i+1)) to frame i; no extrapolation or halftime time."""
    return [b['video_time_s'] - a['video_time_s'] if a['period'] == b['period'] and
            0 < b['video_time_s'] - a['video_time_s'] <= max_gap_s else 0.
            for a, b in zip(rows, rows[1:])] + ([0.] if rows else [])


def summarize(rows, max_gap_s=.2):
    states, teams, controlled, players = Counter(), Counter(), Counter(), Counter()
    weights = durations(rows, max_gap_s)
    for r, dt in zip(rows, weights):
        states[r['status']] += dt
        if r['team']:
            teams[r['team']] += dt
        if r['carrier']:
            controlled[r['team']] += dt
            players[r['carrier']] += dt
    total, assigned = sum(weights), sum(teams.values())
    return dict(frames=len(rows), accounted_seconds=total, state_seconds=dict(states),
                team_assigned_seconds=dict(teams), controlled_seconds=dict(controlled),
                player_controlled_seconds=dict(players),
                assigned_share_percent={s: 100 * teams[s] / assigned if assigned else None for s in ['home', 'away']},
                team_assignment_coverage_percent=100 * assigned / total if total else None,
                warning='Assigned tracked-time share includes in-bounds stoppages; not official ball-in-play possession.')


ACTOR_FIELDS = dict(PA='passerPlayerId', CR='crosserPlayerId', SH='shooterPlayerId',
                   CL='clearerPlayerId', BC='ballCarrierPlayerId', TC='touchPlayerId',
                   RE='rebounderPlayerId', IT='playerId')


def event_roster(events):
    """Identity join for validation/reporting only, never called by predictor."""
    roster, conflicts = {}, set()
    for e in events:
        for side in ['home', 'away']:
            for p in e.get(side + 'Players') or []:
                if p.get('playerId') is None or p.get('jerseyNum') is None:
                    continue
                pid, key = str(p['playerId']), f"{side}:{p['jerseyNum']}"
                if pid in roster and roster[pid] != key:
                    conflicts.add(pid)
                roster[pid] = key
    return {pid: key for pid, key in roster.items() if pid not in conflicts}, sorted(conflicts)


def validation_metrics(details):
    n = len(details)
    aligned = [r for r in details if r['aligned']]
    covered = [r for r in aligned if r['predicted_player']]
    correct = sum(r['correct_player'] for r in details)
    return dict(labels=n, aligned=len(aligned), covered=len(covered), correct_player=correct,
                player_accuracy_all_labels=correct / n if n else None,
                player_accuracy_when_covered=correct / len(covered) if covered else None,
                carrier_coverage=len(covered) / n if n else None,
                correct_team=sum(r['correct_team'] for r in details),
                team_accuracy_all_labels=sum(r['correct_team'] for r in details) / n if n else None,
                nearest_player_accuracy_all_labels=sum(r['correct_nearest'] for r in details) / n if n else None,
                player_found_within_250ms=sum(r['found_in_window'] for r in details) / n if n else None)


def validate_events(rows, events):
    roster, conflicts = event_roster(events)
    periods = defaultdict(list)
    for r in rows:
        periods[r['period']].append(r)
    times = {p: [r['video_time_s'] for r in rs] for p, rs in periods.items()}
    details, skipped = [], Counter()
    for index, e in enumerate(events):
        ge, pe = e.get('gameEvents') or {}, e.get('possessionEvents') or {}
        typ = pe.get('possessionEventType')
        if typ not in ACTOR_FIELDS or pe.get('nonEvent') is True or ge.get('videoMissing') is True:
            skipped['unsupported_ambiguous_or_non_event'] += 1
            continue
        pid = ge.get('playerId') if typ == 'IT' else pe.get(ACTOR_FIELDS[typ])
        actor = roster.get(str(pid))
        if not actor:
            skipped['unmapped_actor'] += 1
            continue
        period, t = ge.get('period'), number(e.get('eventTime'))
        ts, rs = times.get(period, []), periods.get(period, [])
        i = bisect.bisect_left(ts, t)
        candidates = [j for j in [i - 1, i] if 0 <= j < len(rs)]
        j = min(candidates, key=lambda j: abs(ts[j] - t)) if candidates else None
        aligned = j is not None and math.isfinite(t) and abs(ts[j] - t) <= .067
        r = rs[j] if aligned else {}
        lo, hi = bisect.bisect_left(ts, t - .25), bisect.bisect_right(ts, t + .25)
        details.append(dict(event_index=index, event_type=typ, event_time_s=t, period=period,
                            expected_player=actor, predicted_player=r.get('carrier', ''),
                            predicted_team=r.get('team', ''), status=r.get('status', 'unaligned'),
                            aligned=aligned, clock_error_s=abs(ts[j] - t) if aligned else '',
                            correct_player=r.get('carrier') == actor,
                            correct_team=r.get('team') == actor.split(':')[0],
                            nearest_player=r.get('nearest', ''), correct_nearest=r.get('nearest') == actor,
                            found_in_window=aligned and any(x['carrier'] == actor for x in rs[lo:hi])))
    report = dict(all=validation_metrics(details), skipped=dict(skipped), identity_conflicts=conflicts)
    report['by_event_type'] = {typ: validation_metrics([r for r in details if r['event_type'] == typ])
                               for typ in sorted({r['event_type'] for r in details})}
    report['by_period'] = {str(p): validation_metrics([r for r in details if r['period'] == p])
                           for p in sorted({r['period'] for r in details})}
    report['warning'] = ('Sparse event actors are proxy labels, not dense control ground truth. '
                         'The +/-250ms result uses future frames and is diagnostic only. '
                         'No parameter fitting or held-out cross-match generalization claim.')
    return report, details


def detect_transfers(rows, max_flight_s=3., min_pass_distance_m=3., max_frame_gap_s=.2):
    """Conservative observed control transfers, not a kick/contact detector."""
    events, last, previous = [], None, None
    bridgeable = {'controlled', 'loose_ball', 'fast_ball', 'airborne',
                  'in_flight_provisional', 'pending_control'}
    for r in rows:
        if previous and r['video_time_s'] - previous['video_time_s'] > max_frame_gap_s:
            last = None
        previous = r
        if (r['status'] not in bridgeable or (last and (r['period'] != last['period'] or
                r['video_time_s'] - last['video_time_s'] > max_flight_s))):
            last = None
        if not r['carrier']:
            continue
        if last and last['carrier'] != r['carrier']:
            distance = math.hypot(r['ball_x'] - last['ball_x'], r['ball_y'] - last['ball_y'])
            same_team = r['team'] == last['team']
            if not same_team or distance >= min_pass_distance_m:
                events.append(dict(event_type='completed_pass_candidate' if same_team else 'recovery_candidate',
                                   period=r['period'], release_time_s=last['video_time_s'],
                                   reception_time_s=r['video_time_s'], from_player=last['carrier'],
                                   to_player=r['carrier'], distance_m=distance))
        last = r
    return events


def player_labels(events):
    roster, _ = event_roster(events)
    labels = {}
    for e in events:
        ge, pe = e.get('gameEvents') or {}, e.get('possessionEvents') or {}
        for record in [ge, pe]:
            for field, pid in record.items():
                if field == 'playerId' or field.endswith('PlayerId'):
                    key = roster.get(str(pid))
                    name = record.get(field[:-2] + 'Name')
                    if key and name:
                        labels[key] = dict(player_id=str(pid), name=name)
    return labels


def validate_transfers(candidates, events, tolerance_s=1.):
    roster, _ = event_roster(events)
    references = []
    for i, e in enumerate(events):
        ge, pe = e.get('gameEvents') or {}, e.get('possessionEvents') or {}
        typ = pe.get('possessionEventType')
        if (typ not in ('PA', 'CR') or pe.get('nonEvent') is True or ge.get('videoMissing') is True
                or pe.get('passOutcomeType' if typ == 'PA' else 'crossOutcomeType') != 'C'):
            continue
        sender = roster.get(str(pe.get('passerPlayerId' if typ == 'PA' else 'crosserPlayerId')))
        receiver = roster.get(str(pe.get('receiverPlayerId')))
        if sender and receiver:
            references.append(dict(event_index=i, period=ge.get('period'), time_s=number(e.get('eventTime')),
                                   sender=sender, receiver=receiver))
    predictions = [c for c in candidates if c['event_type'] == 'completed_pass_candidate']
    edges = []
    for i, c in enumerate(predictions):
        for j, ref in enumerate(references):
            error = abs(c['release_time_s'] - ref['time_s'])
            if (c['period'] == ref['period'] and c['from_player'] == ref['sender']
                    and c['to_player'] == ref['receiver'] and error <= tolerance_s):
                edges.append((error, i, j))
    used_predictions, used_references, matches = set(), set(), []
    for error, i, j in sorted(edges):
        if i not in used_predictions and j not in used_references:
            used_predictions.add(i)
            used_references.add(j)
            matches.append(dict(candidate_index=i, event_index=references[j]['event_index'], timing_error_s=error))
    tp, npred, nref = len(matches), len(predictions), len(references)
    return dict(predictions=npred, references=nref, true_positive=tp, false_positive=npred - tp,
                false_negative=nref - tp, precision=tp / npred if npred else None,
                recall=tp / nref if nref else None, f1=2 * tp / (npred + nref) if npred + nref else None,
                tolerance_s=tolerance_s,
                warning='Greedy closest-time one-to-one match, same period and exact sender/receiver. '
                        'Only completed PA/CR with mapped receiver are references; other event types are not validated.'), matches


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def predict_file(path, tracker):
    rows, audit, previous_obj = [], Counter(), None
    suffix = 'Smoothed' if tracker.source == 'smoothed' else ''
    position_keys = ['homePlayers' + suffix, 'awayPlayers' + suffix,
                     'ballsSmoothed' if suffix else 'balls']
    with path.open('rb') as f:
        for line in f:
            line = line.strip().removeprefix(b'\xef\xbb\xbf')
            if not line:
                audit['blank_lines'] += 1
                continue
            try:
                obj = loads(line)
            except (ValueError, TypeError):
                audit['malformed_json_lines'] += 1
                continue
            audit['input_lines'] += 1
            for key in ['game_event', 'possession_event', 'game_event_id', 'possession_event_id']:
                if obj.get(key) is not None:
                    audit['ignored_' + key] += 1
            if previous_obj and all(obj.get(k) == previous_obj.get(k) for k in ['videoTimeMs', 'frameNum', 'period']):
                audit['duplicate_frames_skipped'] += 1
                if any(obj.get(k) != previous_obj.get(k) for k in position_keys):
                    audit['duplicate_frames_with_conflicting_tracking'] += 1
                continue  # deterministic first observation, never event-guided selection
            if rows and number(obj.get('videoTimeMs')) / 1000 <= rows[-1]['video_time_s']:
                raise ValueError('Tracking must increase in videoTimeMs after duplicate frame removal')
            rows.append(tracker.update(obj))
            previous_obj = obj
            if len(rows) % 50000 == 0:
                print(f'Predicted {len(rows):,} frames', flush=True)
    return rows, audit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tracking', type=Path, required=True)
    ap.add_argument('--events', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--source', choices=['raw', 'smoothed'], default='smoothed')
    args = ap.parse_args()
    tracker = PossessionTracker(source=args.source)
    rows, audit = predict_file(args.tracking, tracker)
    if not rows:
        raise ValueError('No readable tracking frames')
    # All prediction is complete before opening the validation file.
    events = loads(args.events.read_bytes().removeprefix(b'\xef\xbb\xbf'))
    validation, details = validate_events(rows, events)
    transfers = detect_transfers(rows)
    transfer_validation, transfer_matches = validate_transfers(transfers, events)
    report = dict(tracking_file=args.tracking.name, events_file=args.events.name,
                  parameters=asdict(tracker), audit=dict(audit), summary=summarize(rows),
                  validation=validation, transfer_validation=transfer_validation,
                  candidate_counts=dict(Counter(e['event_type'] for e in transfers)), event_type_counts=dict(Counter(
                      (e.get('possessionEvents') or {}).get('possessionEventType') or 'none' for e in events)))
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / 'frames.csv', rows)
    write_csv(args.output / 'event_validation.csv', details)
    write_csv(args.output / 'predicted_events.csv', transfers)
    write_csv(args.output / 'pass_matches.csv', transfer_matches)
    labels = player_labels(events)
    write_csv(args.output / 'players.csv', [dict(player=k, player_id=labels.get(k, {}).get('player_id', ''),
              name=labels.get(k, {}).get('name', ''), controlled_seconds=v)
              for k, v in sorted(report['summary']['player_controlled_seconds'].items(), key=lambda kv: -kv[1])])
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(summary=report['summary'], validation=validation['all']), indent=2))
    print(f'Wrote local analysis to {args.output.resolve()}')


if __name__ == '__main__':
    main()
