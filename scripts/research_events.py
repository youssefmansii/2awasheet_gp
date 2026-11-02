"""Research-informed interaction intervals and explicitly heuristic event extensions."""
from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from research_tracking import time_segments


@dataclass
class ResearchConfig:
    touch_radius_m: float = 2.5
    acceleration_mps2: float = 4.
    minimum_touch_separation_s: float = .15
    reach_m: float = 4.
    prediction_s: float = 1.
    max_possession_hold_s: float = 5.
    contest_radius_m: float = 2.5
    contest_margin_m: float = .6
    max_pass_flight_s: float = 5.
    minimum_pass_distance_m: float = 2.
    pressure_distance_m: float = 3.


def event(kind, period, time, actor='', target='', end=None, evidence='geometry', **extra):
    return dict(event_type=kind, period=int(period), time_s=float(time),
                end_time_s=float(time if end is None else end), actor=actor, target=target,
                evidence=evidence, **extra)


def geometry(c, p, cfg):
    distance = np.linalg.norm(p['xy'] - p['ball'][:, None], axis=-1)
    distance = np.where(np.isfinite(distance), distance, np.inf)
    nearest = np.argmin(distance, axis=1)
    closest = distance[np.arange(len(distance)), nearest]
    nearest[~np.isfinite(closest)] = -1
    home = np.array([k.startswith('home:') for k in c.keys])
    hd = distance[:, home].min(axis=1) if home.any() else np.full(len(distance), np.inf)
    ad = distance[:, ~home].min(axis=1) if (~home).any() else np.full(len(distance), np.inf)
    with np.errstate(invalid='ignore'):
        contested = ((hd <= cfg.contest_radius_m) & (ad <= cfg.contest_radius_m)
                     & (np.abs(hd - ad) < cfg.contest_margin_m))
    return distance, nearest, closest, contested


def contact_candidates(c, p, cfg, distance, nearest, closest, contested):
    t = c.time
    norm = np.linalg.norm(p['acceleration'], axis=-1)
    median_dt = np.median(np.diff(t)) if len(t) > 1 else 1 / 30
    peaks, _ = find_peaks(np.nan_to_num(norm, nan=-1.), height=cfg.acceleration_mps2,
                          distance=max(1, round(cfg.minimum_touch_separation_s / median_dt)))
    contacts = []
    for i in peaks:
        lo, hi = np.searchsorted(t, [t[i] - .1, t[i] + .1])
        js = np.arange(lo, min(hi + 1, len(t)))
        js = js[(c.period[js] == c.period[i]) & (closest[js] <= cfg.touch_radius_m)]
        if not len(js):
            continue
        # Choose a nearby closest-approach instant using tracking only.
        j = int(js[np.argmin(closest[js] + 2 * np.abs(t[js] - t[i]))])
        actor = int(nearest[j])
        if actor < 0:
            continue
        contacts.append(dict(index=j, slot=actor, source='acceleration_peak', strength=float(norm[i])))
    # Slow arrival without a sharp kick is a different source of action onset.
    speed = np.linalg.norm(p['ball_velocity'], axis=-1)
    for actor in range(len(c.keys)):
        mask = (nearest == actor) & (closest <= 1.8) & (speed <= 8.) & ~contested & (c.period > 0)
        for a, b in time_segments(mask, t, c.period):
            if t[b - 1] - t[a] >= .1 and c.period[a] == c.period[b - 1] and np.max(np.diff(t[a:b])) <= .2:
                contacts.append(dict(index=a, slot=actor, source='stable_proximity', strength=0.))
    contacts.sort(key=lambda r: (r['index'], -r['strength']))
    chosen = []
    for contact in contacts:
        i = contact['index']
        if chosen and chosen[-1]['slot'] == contact['slot'] and c.period[chosen[-1]['index']] == c.period[i]:
            if t[i] - t[chosen[-1]['index']] < cfg.minimum_touch_separation_s:
                if contact['strength'] > chosen[-1]['strength']:
                    chosen[-1] = contact
                continue
        chosen.append(contact)
    return chosen


def action_features(c, p, distance, action):
    a, b, slot = action['start_index'], action['end_index'], action['slot']
    b = max(a + 1, b)
    speed = np.linalg.norm(p['ball_velocity'][a:b], axis=-1)
    acceleration = np.linalg.norm(p['acceleration'][a:b], axis=-1)
    valid = np.isfinite(distance[a:b, slot])
    opponents = np.array([k.split(':')[0] != c.keys[slot].split(':')[0] for k in c.keys])
    result = dict(action_id=action['action_id'], player=c.keys[slot], period=int(c.period[a]),
                  start_time_s=float(c.time[a]), end_time_s=float(c.time[min(b, len(c.time) - 1)]),
                  duration_s=float(c.time[min(b, len(c.time) - 1)] - c.time[a]),
                  mean_ball_speed=float(np.nanmean(speed)) if np.isfinite(speed).any() else 0.,
                  var_ball_speed=float(np.nanvar(speed)) if np.isfinite(speed).any() else 0.,
                  mean_ball_acceleration=float(np.nanmean(acceleration)) if np.isfinite(acceleration).any() else 0.,
                  var_ball_acceleration=float(np.nanvar(acceleration)) if np.isfinite(acceleration).any() else 0.,
                  mean_ball_distance=float(np.mean(distance[a:b, slot][valid])) if valid.any() else 10.,
                  mean_tracking_quality=float(np.mean(np.minimum(p['player_quality'][a:b, slot], p['ball_quality'][a:b]))),
                  reconstructed_fraction=float(np.mean(p['reconstructed_ball'][a:b])))
    for radius in np.arange(.5, 5.1, .5):
        result[f'opponents_within_{radius:.1f}m'] = float(np.mean((distance[a:b, opponents] <= radius).sum(axis=1)))
    return result


def infer(c, p, cfg=None):
    cfg = cfg or ResearchConfig()
    distance, nearest, closest, contested = geometry(c, p, cfg)
    contacts = contact_candidates(c, p, cfg, distance, nearest, closest, contested)
    by_index = {}
    touches = []
    for touch in contacts:
        i, slot = touch['index'], touch['slot']
        by_index.setdefault(i, []).append(touch)
        touches.append(event('touch_candidate', c.period[i], c.time[i], c.keys[slot],
                             evidence=touch['source'], frame_index=i,
                             ambiguous=bool(contested[i]), strength=touch['strength']))
    rows, actions, active, owner = [], [], None, -1
    last_influence = -np.inf

    def finish(end):
        nonlocal active
        if active is not None:
            active['end_index'] = max(active['start_index'] + 1, end)
            actions.append(active)
            active = None

    for i, t in enumerate(c.time):
        continuous = i > 0 and c.period[i] == c.period[i - 1] and 0 < t - c.time[i - 1] <= .2
        valid_ball = bool(np.isfinite(p['ball'][i]).all())
        outside = valid_ball and (abs(p['ball'][i, 0]) > 52.5 or abs(p['ball'][i, 1]) > 34.)
        if not continuous or not valid_ball or outside or c.period[i] <= 0:
            finish(i)
            owner, last_influence = -1, -np.inf
        row = dict(frame_num=int(c.frame[i]), video_time_s=float(t), period=int(c.period[i]),
                   carrier='', team='', possession_owner='', previous_owner=c.keys[owner] if owner >= 0 else '',
                   status='loose_ball', control_status='unknown', control_probability='',
                   nearest=c.keys[nearest[i]] if nearest[i] >= 0 else '',
                   ball_x=float(p['ball'][i, 0]), ball_y=float(p['ball'][i, 1]),
                   tracking_quality=float(p['ball_quality'][i]), reconstructed_ball=bool(p['reconstructed_ball'][i]),
                   action_id='')
        if not valid_ball or outside or c.period[i] <= 0:
            row['status'] = 'missing_ball' if not valid_ball else 'out_of_bounds_candidate' if outside else 'outside_period'
            rows.append(row)
            continue
        new = by_index.get(i, [])
        if new:
            touch = max(new, key=lambda r: r['strength'])
            slot = touch['slot']
            if active is None or slot != active['slot']:
                finish(i)
                active = dict(action_id=len(actions), start_index=i, slot=slot, source=touch['source'])
            if not contested[i]:
                owner, last_influence = slot, t
        if contested[i]:
            row.update(status='contested', previous_owner=c.keys[owner] if owner >= 0 else '')
            rows.append(row)
            continue
        if active is not None:
            slot = active['slot']
            ball_distance = distance[i, slot]
            predicted_ball = p['ball'][i] + cfg.prediction_s * p['ball_velocity'][i]
            reachable = np.linalg.norm(predicted_ball - p['xy'][i, slot]) <= cfg.reach_m
            # A contact is recorded even when it immediately releases the ball.
            contact_now = any(touch['slot'] == slot for touch in new)
            if ball_distance <= cfg.touch_radius_m and (reachable or contact_now):
                row.update(carrier=c.keys[slot], status='interaction', control_status='interaction_only',
                           action_id=active['action_id'])
                owner, last_influence = slot, t
                relative = np.linalg.norm(p['ball_velocity'][i] - p['player_velocity'][i, slot])
                if (t - c.time[active['start_index']] >= .15 and relative <= 5.
                        and (not np.isfinite(c.height[i]) or c.height[i] <= 2.)):
                    row.update(status='likely_control', control_status='likely_control')
            if not reachable and not contact_now or not np.isfinite(ball_distance):
                finish(i)
        if owner >= 0 and t - last_influence <= cfg.max_possession_hold_s:
            row.update(possession_owner=c.keys[owner], team=c.keys[owner].split(':')[0])
            if not row['carrier']:
                row['status'] = 'in_flight_provisional'
        else:
            owner = -1
        rows.append(row)
    finish(len(rows))
    for action in actions:
        action['features'] = action_features(c, p, distance, action)
    return rows, touches, actions


def attack_directions(c, p):
    result = {}
    home = np.array([k.startswith('home:') for k in c.keys])
    for period in np.unique(c.period):
        indices = np.flatnonzero(c.period == period)[:150]
        hx, ax = p['xy'][indices][:, home, 0], p['xy'][indices][:, ~home, 0]
        if not np.isfinite(hx).any() or not np.isfinite(ax).any():
            result[int(period)] = None
        else:
            result[int(period)] = 1 if np.nanmedian(hx) < np.nanmedian(ax) else -1
    return result


def movement_events(c, p, rows, touches, actions, cfg):
    distance, _, _, contested = geometry(c, p, cfg)
    directions = attack_directions(c, p)
    output = [dict(touch) for touch in touches]
    previous = None
    for touch in touches:
        i, key, t = touch['frame_index'], touch['actor'], touch['time_s']
        slot = c.keys.index(key)
        side = key.split(':')[0]
        if touch['evidence'] == 'acceleration_peak':
            if np.isfinite(c.height[i]) and c.height[i] <= .5:
                output.append(event('kick_candidate', c.period[i], t, key, evidence='low_ball_acceleration_peak'))
            adir = directions[int(c.period[i])]
            adir = adir if side == 'home' else -adir if adir is not None else None
            k = min(np.searchsorted(c.time, t + .15), len(c.time) - 1)
            vx, vy = p['ball_velocity'][k]
            x, y = p['ball'][i]
            same_segment = c.period[k] == c.period[i] and np.isfinite(p['ball'][i:k + 1]).all() and np.max(np.diff(c.time[i:k + 1]), initial=0) <= .2
            if adir is not None and same_segment and np.isfinite(vx) and vx * adir > 0:
                goal_time = (adir * 52.5 - x) / vx
                goal_y = y + goal_time * vy
                if x * adir > 17.5 and np.hypot(vx, vy) >= 8. and 0 < goal_time <= 4. and abs(goal_y) <= 5.66:
                    output.append(event('shot_candidate', c.period[i], t, key, evidence='goal_directed_release'))
                opponents = np.array([k.split(':')[0] != side for k in c.keys])
                opp_distance = distance[i, opponents].min() if opponents.any() else np.inf
                if x * adir < -17.5 and np.hypot(vx, vy) >= 10. and opp_distance <= 3.:
                    output.append(event('clearance_candidate', c.period[i], t, key, evidence='pressured_defensive_release'))
        if previous and previous['actor'] != key:
            j, sender = previous['frame_index'], previous['actor']
            elapsed = t - previous['time_s']
            valid_path = (c.period[j] == c.period[i] and elapsed <= cfg.max_pass_flight_s
                          and np.isfinite(p['ball'][j:i + 1]).all()
                          and np.max(np.diff(c.time[j:i + 1]), initial=0) <= .2
                          and np.all(np.abs(p['ball'][j:i + 1]) <= [52.5, 34.]))
            if valid_path:
                travel = np.linalg.norm(p['ball'][i] - p['ball'][j])
                same_team = sender.split(':')[0] == side
                if same_team and travel >= cfg.minimum_pass_distance_m:
                    output.append(event('completed_pass_candidate', c.period[j], previous['time_s'], sender, key,
                                        end=t, evidence='teammate_interaction_transfer'))
                    output.append(event('reception_candidate', c.period[i], t, key, sender))
                    adir = directions[int(c.period[j])]
                    adir = adir if side == 'home' else -adir if adir is not None else None
                    if adir is not None and abs(p['ball'][j, 1]) >= 17. and p['ball'][i, 0] * adir >= 30. and abs(p['ball'][i, 1]) < 20.:
                        output.append(event('cross_candidate', c.period[j], previous['time_s'], sender, key, end=t))
                elif not same_team:
                    nearby = np.any(contested[j:i + 1]) and elapsed <= 2.
                    kind = 'tackle_candidate' if nearby else 'interception_candidate' if travel >= 3. and elapsed >= .15 else 'recovery_candidate'
                    output.append(event(kind, c.period[i], t, key, sender, evidence='opponent_interaction_transfer'))
                    if travel >= 3. and elapsed >= .15:
                        output.append(event('unsuccessful_pass_candidate', c.period[j], previous['time_s'], sender,
                                            end=t, evidence='opponent_received_ball_after_travel'))
        previous = touch
    for action in actions:
        a, b, slot = action['start_index'], action['end_index'], action['slot']
        b = min(b, len(c.time))
        f = action['features']
        if f['duration_s'] >= 1. and np.linalg.norm(p['ball'][b - 1] - p['ball'][a]) >= 3.:
            if sum(r['control_status'] in ('likely_control', 'model_control') for r in rows[a:b]) >= .7 * (b - a):
                output.append(event('carry_candidate', c.period[a], c.time[a], c.keys[slot], end=c.time[b - 1], evidence='sustained_moving_control'))
    for a, b in time_segments(contested, c.time, c.period):
        if c.period[a] == c.period[b - 1] and c.time[b - 1] - c.time[a] >= .1:
            before = rows[a]['previous_owner']
            after = ''
            for j in range(b, min(b + 30, len(rows))):
                if (c.period[j] != c.period[a] or c.time[j] - c.time[b - 1] > 1.
                        or c.time[j] - c.time[j - 1] > .2 or not np.isfinite(p['ball'][j]).all()):
                    break
                if rows[j]['carrier']:
                    after = rows[j]['carrier']
                    break
            before_side = before.split(':')[0] if before else ''
            opponents = np.array([k.split(':')[0] != before_side for k in c.keys])
            challenger = c.keys[np.flatnonzero(opponents)[np.argmin(distance[a, opponents])]] if before and opponents.any() else ''
            output.append(event('contest_candidate', c.period[a], c.time[a], before, after,
                                end=c.time[b - 1], evidence='opponents_near_ball', challenger=challenger))
            if np.isfinite(c.height[a:b]).any() and np.nanmax(c.height[a:b]) > 1.5:
                output.append(event('aerial_duel_candidate', c.period[a], c.time[a], before, challenger,
                                    end=c.time[b - 1], evidence='contested_with_reported_ball_height'))
            if before and after:
                kind = 'challenge_retained_candidate' if before == after else 'challenge_turnover_candidate'
                output.append(event(kind, c.period[a], c.time[a], before, after,
                                    end=c.time[b - 1], challenger=challenger))
                if before == after and np.linalg.norm(p['ball'][b - 1] - p['ball'][a]) >= 1.:
                    output.append(event('dribble_candidate', c.period[a], c.time[a], before, challenger,
                                        end=c.time[b - 1], evidence='moving_contest_with_retained_interaction'))
    outside = np.isfinite(p['ball']).all(axis=1) & ((np.abs(p['ball']) > [52.5, 34.]).any(axis=1))
    for a, b in time_segments(outside, c.time, c.period):
        output.append(event('out_candidate', c.period[a], c.time[a], end=c.time[b - 1], evidence='ball_centre_outside_pitch'))
        j = a - 1
        if j >= 0 and c.period[j] == c.period[a] and c.time[a] - c.time[j] <= .2 and np.isfinite(p['ball'][j]).all():
            x0, y0 = p['ball'][j]
            x1, y1 = p['ball'][a]
            if abs(x0) <= 52.5 < abs(x1):
                alpha = (np.sign(x1) * 52.5 - x0) / (x1 - x0)
                crossing_y = y0 + alpha * (y1 - y0)
                if abs(crossing_y) <= 3.66 and np.isfinite(c.height[a]) and c.height[a] <= 2.44:
                    output.append(event('goal_line_crossing_candidate', c.period[a], c.time[a], evidence='xy_crossing_and_reported_height'))
    for slot, key in enumerate(c.keys):
        opponents = np.array([k.split(':')[0] != key.split(':')[0] for k in c.keys])
        if not opponents.any():
            continue
        under_pressure = np.array([r['carrier'] == key for r in rows]) & (distance[:, opponents].min(axis=1) <= cfg.pressure_distance_m)
        opp_indices = np.flatnonzero(opponents)
        for a, b in time_segments(under_pressure, c.time, c.period):
            if c.period[a] == c.period[b - 1] and c.time[b - 1] - c.time[a] >= .2:
                presser = opp_indices[np.argmin(distance[a, opponents])]
                output.append(event('pressure_candidate', c.period[a], c.time[a], key, c.keys[presser], end=c.time[b - 1]))
    # Restart geometry is a separate extension; the in-bounds play state remains unknown.
    for touch in touches:
        i, t = touch['frame_index'], touch['time_s']
        if touch['evidence'] != 'acceleration_peak':
            continue
        j = np.searchsorted(c.time, t - 1.)
        if i - j < 3 or c.period[j] != c.period[i] or not np.isfinite(p['ball'][j:i + 1]).all():
            continue
        if np.max(np.linalg.norm(p['ball'][j:i] - p['ball'][j], axis=1)) > .5:
            continue
        x, y = p['ball'][i]
        if np.hypot(x, y) <= 2.:
            kind = 'kickoff_candidate'
        elif abs(x) >= 50. and abs(y) >= 30.:
            kind = 'corner_candidate'
        elif abs(y) >= 32.:
            kind = 'throw_in_candidate'
        elif abs(x) >= 45. and abs(y) <= 12.:
            kind = 'goal_kick_candidate'
        else:
            kind = 'stationary_restart_candidate'
        output.append(event(kind, c.period[i], t, touch['actor'], evidence='stationary_ball_release_location'))
    output.sort(key=lambda e: (e['time_s'], e['event_type']))
    return output
