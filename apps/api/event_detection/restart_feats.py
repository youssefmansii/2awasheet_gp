"""restart_feats.py - how play restarts after a shot (tracking only): continuous kick-off evidence.
After a goal both teams go back to their own halves and the ball is put on the centre spot.  Broadcast
replays hide part of this, so instead of one yes/no rule we measure several quantities in 20-150 s after
the kick and let the outcome model weigh them."""
import numpy as np
import pandas as pd

RS_FEATS = ["ko_own_share_max", "ko_ball_centre_min", "ko_centre_dt", "ko_both", "dead_time_s", "ko_tracked_share"]


def restart_features(tr, ball, kicks) -> pd.DataFrame:
    T = tr.n_frames; fps = tr.fps
    side_home = tr.players.side.values == "home"
    rows = []
    for i in kicks:
        i = int(i); a, b = min(T, i + int(20 * fps)), min(T, i + int(150 * fps))
        same = tr.period[a:b] == tr.period[i]
        r = dict.fromkeys(RS_FEATS, np.nan)
        if same.sum() > fps:
            sl = slice(a, a + int(same.sum()))
            hs = tr.attack_sign("home", int(tr.period[i]))
            sgn = np.where(side_home, hs, -hs)[None, :]
            px = tr.px[sl]; fin = np.isfinite(px)
            own = ((px * sgn <= 1.0) & fin).sum(1) / np.maximum(fin.sum(1), 1)
            own = np.where(fin.sum(1) >= 16, own, np.nan)
            bc = np.hypot(ball["pos"][sl, 0], ball["pos"][sl, 1])
            bc_still = np.where(ball["speed2d"][sl] < 1.5, bc, np.nan)
            r["ko_own_share_max"] = float(np.nanmax(own)) if np.isfinite(own).any() else np.nan
            if np.isfinite(bc_still).any():
                k = int(np.nanargmin(bc_still)); r["ko_ball_centre_min"] = float(bc_still[k]); r["ko_centre_dt"] = 20 + k / fps
                w = own[max(0, k - 60):k + 60]
                r["ko_both"] = float(np.nanmax(w)) if np.isfinite(w).any() and bc_still[k] < 2.5 else 0.0
            r["ko_tracked_share"] = float(np.isfinite(bc).mean())
        # dead time: seconds until the ball is next seen moving faster than 3 m/s for 0.5 s
        e = min(T, i + int(200 * fps)); sp = ball["speed2d"][i + int(3 * fps):e]
        if len(sp) >= 15:
            mv = np.convolve(np.nan_to_num(sp) > 3, np.ones(15), "valid") >= 15
            r["dead_time_s"] = 3 + (np.argmax(mv) / fps if mv.any() else 200)
        rows.append(r)
    return pd.DataFrame(rows)
