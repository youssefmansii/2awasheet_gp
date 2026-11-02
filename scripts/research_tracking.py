"""Compact, annotation-blind tracking input and retrospective trajectory processing."""
from dataclasses import dataclass
from collections import Counter

import numpy as np

from tracking_possession import loads, number


@dataclass
class Capture:
    time: np.ndarray
    period: np.ndarray
    frame: np.ndarray
    keys: list
    xy: np.ndarray
    player_quality: np.ndarray
    ball: np.ndarray
    height: np.ndarray
    ball_quality: np.ndarray
    audit: dict


def quality(record):
    confidence = {'HIGH': 1., 'MEDIUM': .7, 'LOW': .35}.get(record.get('confidence'), .6)
    visibility = 1. if record.get('visibility') == 'VISIBLE' else .5
    return confidence * visibility


def load_capture(path, source='raw'):
    slots, chunks, audit = {}, [], Counter()
    block, capacity = 8192, 64
    xy = np.full((block, capacity, 2), np.nan, dtype=np.float32)
    pq = np.zeros((block, capacity), dtype=np.float32)
    meta = np.full((block, 8), np.nan, dtype=np.float64)
    count, previous = 0, None
    suffix = 'Smoothed' if source == 'smoothed' else ''
    keys = ['homePlayers' + suffix, 'awayPlayers' + suffix, 'ballsSmoothed' if suffix else 'balls']
    with path.open('rb') as f:
        for line in f:
            line = line.strip().removeprefix(b'\xef\xbb\xbf')
            if not line:
                continue
            try:
                obj = loads(line)
            except (ValueError, TypeError):
                audit['malformed_lines'] += 1
                continue
            audit['input_lines'] += 1
            for field in ['game_event', 'possession_event', 'game_event_id', 'possession_event_id']:
                if obj.get(field) is not None:
                    audit['ignored_' + field] += 1
            if previous and all(obj.get(k) == previous.get(k) for k in ['frameNum', 'videoTimeMs', 'period']):
                audit['duplicate_frames_skipped'] += 1
                if any(obj.get(k) != previous.get(k) for k in keys):
                    audit['conflicting_duplicates'] += 1
                continue
            t = number(obj.get('videoTimeMs')) / 1000
            if not np.isfinite(t) or previous and t <= number(previous.get('videoTimeMs')) / 1000:
                raise ValueError('Finite increasing timestamps required after frame deduplication')
            ball = obj.get(keys[2])
            if isinstance(ball, list):
                ball = ball[0] if len(ball) == 1 else None
            ball = ball or {}
            meta[count] = [t, obj.get('period') or 0, obj['frameNum'], number(ball.get('x')),
                           number(ball.get('y')), number(ball.get('z')), quality(ball), bool(ball)]
            for side, key in zip(['home', 'away'], keys[:2]):
                for p in obj.get(key) or []:
                    if p.get('jerseyNum') is None:
                        continue
                    identity = f"{side}:{p['jerseyNum']}"
                    if identity not in slots:
                        if len(slots) >= capacity:
                            raise ValueError('More than 64 player identities in this capture')
                        slots[identity] = len(slots)
                    j = slots[identity]
                    xy[count, j] = number(p.get('x')), number(p.get('y'))
                    pq[count, j] = quality(p)
            previous = obj
            count += 1
            if count == block:
                chunks.append((xy, pq, meta))
                xy = np.full_like(xy, np.nan)
                pq = np.zeros_like(pq)
                meta = np.full_like(meta, np.nan)
                count = 0
    if count:
        chunks.append((xy[:count], pq[:count], meta[:count]))
    if not chunks:
        raise ValueError('No tracking frames')
    p = len(slots)
    if not p:
        raise ValueError('No tracked player identities with jersey numbers')
    xyz = np.concatenate([c[0] for c in chunks])[:, :p].copy()
    scores = np.concatenate([c[1] for c in chunks])[:, :p].copy()
    m = np.concatenate([c[2] for c in chunks])
    return Capture(m[:, 0], m[:, 1].astype(np.int8), m[:, 2].astype(np.int32), list(slots),
                   xyz, scores, m[:, 3:5].astype(np.float32), m[:, 5].astype(np.float32),
                   m[:, 6].astype(np.float32), dict(audit))


def segments(mask):
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    return zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))


def time_segments(mask, time, period):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    breaks = np.flatnonzero((np.diff(indices) != 1) | (np.diff(time[indices]) > .2)
                           | (np.diff(period[indices]) != 0)) + 1
    groups = np.split(indices, breaks)
    return [(int(g[0]), int(g[-1]) + 1) for g in groups]


def fill_short_gaps(t, period, values, max_gap_s=.3):
    out = np.array(values, copy=True)
    flat = out.reshape(len(t), -1)
    filled = np.zeros_like(flat, dtype=bool)
    for j in range(flat.shape[1]):
        for a, b in segments(~np.isfinite(flat[:, j])):
            if (a == 0 or b == len(t) or period[a - 1] != period[b]
                    or t[b] - t[a - 1] > max_gap_s
                    or np.any(np.diff(t[a - 1:b + 1]) > .2)):
                continue
            flat[a:b, j] = np.interp(t[a:b], [t[a - 1], t[b]], flat[[a - 1, b], j])
            filled[a:b, j] = True
    return out, filled.reshape(out.shape)


def rts_smooth(t, period, values, quality_scores, process_noise=16., measurement_sd=.3):
    """Independent xy constant-velocity Kalman filters + RTS backward state pass.

    Measurement variance grows as quality decreases. Missing long runs and period/
    timestamp discontinuities split trajectories. No interpolation occurs here.
    process_noise is continuous white-acceleration spectral density (m²/s³).
    """
    original_shape = values.shape
    obs = np.asarray(values).reshape(len(t), -1)
    scores = np.broadcast_to(quality_scores, original_shape).reshape(obs.shape)
    n, s = obs.shape
    states = np.zeros((n, s, 2), dtype=np.float32)
    cov = np.zeros((n, s, 3), dtype=np.float32)  # p00, p01, p11
    valid = np.isfinite(obs)
    for i in range(n):
        dt = t[i] - t[i - 1] if i else 0.
        continuous = i > 0 and period[i] == period[i - 1] and 0 < dt <= .2
        active = valid[i] & valid[i - 1] if continuous else np.zeros(s, dtype=bool)
        r = (measurement_sd / np.maximum(scores[i], .15)) ** 2
        if continuous:
            x, v = states[i - 1].T
            a, b, c = cov[i - 1].T
            pa = a + 2 * dt * b + dt * dt * c + process_noise * dt ** 3 / 3
            pb = b + dt * c + process_noise * dt * dt / 2
            pc = c + process_noise * dt
            ka, kb = pa / (pa + r), pb / (pa + r)
            error = np.where(valid[i], obs[i] - (x + dt * v), 0.)
            states[i, :, 0] = x + dt * v + ka * error
            states[i, :, 1] = v + kb * error
            cov[i, :, 0] = (1 - ka) * pa
            cov[i, :, 1] = (1 - ka) * pb
            cov[i, :, 2] = pc - kb * pb
        reset = valid[i] & ~active
        states[i, reset, 0] = obs[i, reset]
        states[i, reset, 1] = 0.
        cov[i, reset, 0], cov[i, reset, 1], cov[i, reset, 2] = r[reset], 0., 25.
        states[i, ~valid[i]] = 0.
    smooth = states.copy()
    for i in range(n - 2, -1, -1):
        dt = t[i + 1] - t[i]
        if period[i] != period[i + 1] or not 0 < dt <= .2:
            continue
        active = valid[i] & valid[i + 1]
        a, b, c = cov[i].T
        pa = a + 2 * dt * b + dt * dt * c + process_noise * dt ** 3 / 3
        pb = b + dt * c + process_noise * dt * dt / 2
        pc = c + process_noise * dt
        det = np.maximum(pa * pc - pb * pb, 1e-12)
        g00, g01 = ((a + dt * b) * pc - b * pb) / det, (b * pa - (a + dt * b) * pb) / det
        g10, g11 = ((b + dt * c) * pc - c * pb) / det, (c * pa - (b + dt * c) * pb) / det
        ex = smooth[i + 1, :, 0] - states[i, :, 0] - dt * states[i, :, 1]
        ev = smooth[i + 1, :, 1] - states[i, :, 1]
        smooth[i, active, 0] += (g00 * ex + g01 * ev)[active]
        smooth[i, active, 1] += (g10 * ex + g11 * ev)[active]
    smooth[~valid] = np.nan
    return smooth[:, :, 0].reshape(original_shape), smooth[:, :, 1].reshape(original_shape)


def prepare(capture, max_gap_s=.3):
    t, period = capture.time, capture.period
    xy, filled_players = fill_short_gaps(t, period, capture.xy, max_gap_s)
    ball, filled_ball = fill_short_gaps(t, period, capture.ball, max_gap_s)
    # Quarantine implausible coordinate transitions before differentiating.
    for positions, limit in [(xy, 15.), (ball, 60.)]:
        delta = np.diff(t)
        velocity = np.linalg.norm(np.diff(positions, axis=0), axis=-1) / delta.reshape((-1,) + (1,) * (positions.ndim - 2))
        same_period = (period[1:] == period[:-1]).reshape((-1,) + (1,) * (positions.ndim - 2))
        jumps = (velocity > limit) & same_period
        positions[1:][jumps] = np.nan
    pq = np.where(filled_players.any(axis=-1), .2, capture.player_quality)
    bq = np.where(filled_ball.any(axis=-1), .2, capture.ball_quality)
    print('RTS smoothing player trajectories', flush=True)
    xy, player_velocity = rts_smooth(t, period, xy, pq[:, :, None], process_noise=16.)
    print('RTS smoothing ball trajectory', flush=True)
    ball, ball_velocity = rts_smooth(t, period, ball, bq[:, None], process_noise=400.)
    acceleration = np.full_like(ball_velocity, np.nan)
    if len(t) > 2:
        acceleration[1:-1] = (ball_velocity[2:] - ball_velocity[:-2]) / (t[2:] - t[:-2])[:, None]
        broken = (period[2:] != period[:-2]) | (t[2:] - t[:-2] > .4)
        acceleration[1:-1][broken] = np.nan
    capture.audit['reconstructed_ball_frames'] = int(filled_ball.any(axis=-1).sum())
    return dict(xy=xy, ball=ball, player_velocity=player_velocity,
                ball_velocity=ball_velocity, acceleration=acceleration,
                player_quality=pq, ball_quality=bq, reconstructed_ball=filled_ball.any(axis=-1))
