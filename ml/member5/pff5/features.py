"""
Hand-crafted geometric features computed from tracking, for every Member-5 task.

Everything here is vectorised numpy over "events" (a frame index + optional focal
players).  No thresholds are hard coded as decisions - distances, closing speeds,
angles etc. are handed to the models, which learn the boundaries from PFF labels.

Tables produced (all as pandas DataFrames, one match at a time):

* ``pressure_tables``        candidate-level + event-level rows for pressure
* ``challenge_participant_table``  every player on the pitch at a challenge frame (who is involved?)
* ``challenge_attr_table``   one row per challenge, features for type/winner/outcome/foul
* ``event_foul_table``       one row per possession event, features for fouls/offside
* ``frame_table``            one row per sampled tracking frame, for challenge spotting
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .data import Match

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
EPS = 1e-6
PE_TYPES = ["PA", "CR", "SH", "CL", "BC", "TC", "RE", "CH", "IT", "FO"]


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _norm(a, axis=-1):
    return np.sqrt((a * a).sum(axis))


def _take(arr, rows, slots):
    """arr (E, P, ...) -> (E, ...) picking slot per row; NaN where slot < 0."""
    out = arr[rows, np.clip(slots, 0, arr.shape[1] - 1)].astype(np.float32, copy=True)
    out[slots < 0] = np.nan
    return out


class Scene:
    """Tracking gathered at several time offsets around E reference frames."""

    def __init__(self, m: Match, idx, offsets):
        self.m = m
        self.idx = np.asarray(idx, np.int64)
        self.offsets = list(offsets)
        self.J = np.stack([m.shift(self.idx, o) for o in self.offsets], 1)  # (E, K)
        self.E = len(self.idx)
        self.rows = np.arange(self.E)

    def k(self, offset):
        return self.offsets.index(offset)

    def pos(self, offset):
        return self.m.pos[self.J[:, self.k(offset)]]

    def vel(self, offset):
        return self.m.vel[self.J[:, self.k(offset)]]

    def ball(self, offset):
        return self.m.ball[self.J[:, self.k(offset)]]

    def bvel(self, offset):
        return self.m.ball_vel[self.J[:, self.k(offset)]]


def _nearest_to_ball(m: Match, j):
    """For frames j: slot of nearest player to the ball, its distance, its team."""
    d = _norm(m.pos[j] - m.ball[j][:, None, :2])
    d = np.where(np.isfinite(d), d, np.inf)
    s = d.argmin(1)
    dist = d[np.arange(len(j)), s]
    none = ~np.isfinite(dist)
    return np.where(none, -1, s), np.where(none, np.nan, dist), np.where(none, -1, m.team[s])


def _out_margin(m: Match, ball_xy):
    """> 0 when the ball is outside the pitch (metres beyond the line)."""
    return np.maximum(np.abs(ball_xy[:, 0]) - m.half_len, np.abs(ball_xy[:, 1]) - m.half_wid)


def _stoppage_feats(m: Match, idx, prefix="post_"):
    """What happens in the seconds after a frame - the footprint of whistles and dead balls."""
    f = {}
    E = len(idx)
    spd_all = []
    for o in (1.0, 2.0, 3.0, 5.0):
        j = m.shift(idx, o)
        sp = _norm(m.vel[j])
        f[f"{prefix}mean_speed_{o:g}"] = np.nanmean(sp, 1)
        f[f"{prefix}ball_speed_{o:g}"] = _norm(m.ball_vel[j][:, :2])
        f[f"{prefix}ball_out_{o:g}"] = _out_margin(m, m.ball[j])
        spd_all.append(f[f"{prefix}mean_speed_{o:g}"])
    j0 = m.shift(idx, 0.0)
    f[f"{prefix}mean_speed_0"] = np.nanmean(_norm(m.vel[j0]), 1)
    f[f"{prefix}slowdown_3"] = f[f"{prefix}mean_speed_0"] - f[f"{prefix}mean_speed_3"]
    b1, b3, b5 = m.ball[m.shift(idx, 1.0)], m.ball[m.shift(idx, 3.0)], m.ball[m.shift(idx, 5.0)]
    f[f"{prefix}ball_move_1_3"] = _norm(b3[:, :2] - b1[:, :2])
    f[f"{prefix}ball_move_3_5"] = _norm(b5[:, :2] - b3[:, :2])
    f[f"{prefix}ball_move_0_3"] = _norm(b3[:, :2] - m.ball[j0][:, :2])
    # how much of the next 5 s has a measured ball, and how still it is
    steps = np.arange(1, 16) / 3.0
    JJ = np.stack([m.shift(idx, float(s)) for s in steps], 1)
    obs = m.ball_obs[JJ]
    f[f"{prefix}ball_missing_frac_5"] = (obs == 0).mean(1)
    bs = _norm(m.ball_vel[JJ][..., :2])
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f[f"{prefix}ball_speed_min_5"] = np.nanmin(bs, 1)
            f[f"{prefix}ball_speed_med_5"] = np.nanmedian(bs, 1)
            f[f"{prefix}ball_still_frac_5"] = np.nanmean(bs < 1.0, 1)
            ms = np.nanmean(_norm(m.vel[JJ]), 2)
            f[f"{prefix}players_speed_min_5"] = np.nanmin(ms, 1)
    return f


def _pitch_feats(m: Match, xy, sign, prefix):
    """Position in attack-normalised coordinates (+x = towards the goal being attacked)."""
    x, y = xy[:, 0] * sign, xy[:, 1] * sign
    gx = m.half_len - x
    f = {prefix + "x": x, prefix + "y_abs": np.abs(y), prefix + "dist_goal": np.hypot(gx, y),
         prefix + "dist_own_goal": np.hypot(m.half_len + x, y),
         prefix + "in_att_box": ((gx < 16.5) & (np.abs(y) < 20.16)).astype(np.float32),
         prefix + "in_own_box": ((m.half_len + x < 16.5) & (np.abs(y) < 20.16)).astype(np.float32),
         prefix + "dist_touchline": m.half_wid - np.abs(y)}
    return f


# --------------------------------------------------------------------------- #
# 1. PRESSURE
# --------------------------------------------------------------------------- #
P_OFFS = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]
P_K = 6  # opponent candidates per event


def pressure_rows(m: Match) -> pd.DataFrame:
    """The (frame, actor) pairs PFF labelled with a pressure type.

    kind 0 = pressure on the initial touch (once per game event, at its start)
    kind 1 = pressure on the possession event itself (pass, cross, shot, ...)
    """
    ev = m.events[m.events["has_trk"]]
    out = []
    # initial touch: one per game event
    it = ev[ev["it_initialPressureType"].notna() & ev["ge_playerId"].notna()]
    it = it.drop_duplicates("gameEventId")
    out.append(pd.DataFrame(dict(
        ev_row=it.index.values, kind=0, idx=it["i_start"].values,
        actor=m.slot_of(it["ge_playerId"].tolist()), pe_type="IT",
        y_type=it["it_initialPressureType"].values,
        presser=m.slot_of(it["it_initialPressurePlayerId"].tolist()), hold=0.0)))
    pe = ev[ev["pe_pressureType"].notna() & ~ev["pe_possessionEventType"].isin(["CH", "IT"])]
    actor_cols = ["pe_passerPlayerId", "pe_crosserPlayerId", "pe_shooterPlayerId", "pe_clearerPlayerId",
                  "pe_ballCarrierPlayerId", "pe_touchPlayerId", "pe_rebounderPlayerId", "ge_playerId"]
    actor = pe[[c for c in actor_cols if c in pe]].bfill(axis=1).iloc[:, 0]
    out.append(pd.DataFrame(dict(
        ev_row=pe.index.values, kind=1, idx=pe["i_event"].values, actor=m.slot_of(actor.tolist()),
        pe_type=pe["pe_possessionEventType"].values, y_type=pe["pe_pressureType"].values,
        presser=m.slot_of(pe["pe_pressurePlayerId"].tolist()),
        hold=(pe["eventTime"] - pe["startTime"]).clip(0, 30).fillna(0).values)))
    rows = pd.concat(out, ignore_index=True)
    rows = rows[rows["actor"] >= 0].reset_index(drop=True)
    rows["gid"] = m.gid
    return rows


def pressure_features(m: Match, idx, actor, pe_type=None, kind=None, hold=None):
    """Features for "is the player ``actor`` at frame ``idx`` being pressured, and by whom".

    Returns (ev_df, cand_df, cand_slots) where ev_df has one row per event and
    cand_df has ``P_K`` rows per event (nearest opponents first).
    """
    idx = np.asarray(idx, np.int64)
    actor = np.asarray(actor, np.int64)
    E = len(idx)
    sc = Scene(m, idx, P_OFFS)
    rows = sc.rows
    team_a = m.team[actor]
    sign = (m.attack[idx] * np.where(team_a == 0, 1, -1)).astype(np.float32)  # actor's team attacks +x*sign
    opp = m.team[None, :] != team_a[:, None]                                    # (E, P)

    # distances actor -> every player at every offset
    D = np.empty((len(P_OFFS), E, m.n_players), np.float32)
    for k, o in enumerate(P_OFFS):
        P = sc.pos(o)
        D[k] = _norm(P - P[rows, actor][:, None, :])
    k0 = sc.k(0.0)
    d0 = np.where(opp & np.isfinite(D[k0]), D[k0], np.inf)
    order = np.argsort(d0, 1)[:, :P_K]                                           # (E, K)
    cand_ok = np.isfinite(np.take_along_axis(d0, order, 1))

    P0, V0 = sc.pos(0.0), sc.vel(0.0)
    pa, va = P0[rows, actor], V0[rows, actor]
    B0, BV0 = sc.ball(0.0), sc.bvel(0.0)
    Bp5, Bp1 = sc.ball(0.5), sc.ball(1.0)

    # ---------------- event level ---------------- #
    f = {}
    f.update(_pitch_feats(m, pa, sign, "act_"))
    f["act_speed"] = _norm(va)
    f["act_vx"] = va[:, 0] * sign
    f["act_is_gk"] = m.is_gk[actor].astype(np.float32)
    f["act_vis"] = (m.vis[idx, actor] == 1).astype(np.float32)
    f["act_conf"] = m.conf[idx, actor].astype(np.float32)
    f["ball_z"] = B0[:, 2]
    f["ball_speed"] = _norm(BV0[:, :2])
    f["ball_speed_m25"] = _norm(sc.bvel(-0.25)[:, :2])
    f["ball_speed_p25"] = _norm(sc.bvel(0.25)[:, :2])
    f["ball_z_m25"] = sc.ball(-0.25)[:, 2]
    f["ball_act_dist"] = _norm(B0[:, :2] - pa)
    f["ball_missing"] = (~np.isfinite(B0[:, 0])).astype(np.float32)
    for r in (2, 3, 5, 10):
        f[f"n_opp_{r}"] = (d0 < r).sum(1).astype(np.float32)
    dmate = np.where(~opp & np.isfinite(D[k0]) & (np.arange(m.n_players)[None] != actor[:, None]), D[k0], np.inf)
    f["mate_d1"] = dmate.min(1)
    f["n_mate_5"] = (dmate < 5).sum(1).astype(np.float32)
    f["kind"] = np.zeros(E, np.float32) if kind is None else np.asarray(kind, np.float32)
    f["hold"] = np.zeros(E, np.float32) if hold is None else np.asarray(hold, np.float32)
    pt = np.array(["IT"] * E if pe_type is None else pe_type, dtype=object)
    f["pe_type"] = np.array([PE_TYPES.index(p) if p in PE_TYPES else -1 for p in pt], np.float32)
    # direction the ball leaves in (where a "passing lane" would be)
    out_vec = Bp1[:, :2] - pa
    out_len = _norm(out_vec)
    f["ball_out_len"] = out_len
    ev_df = pd.DataFrame(f)

    # ---------------- candidate level ---------------- #
    c = {}
    er = np.repeat(rows, P_K)
    cs = order.reshape(-1)
    ok = cand_ok.reshape(-1)
    for k, o in enumerate(P_OFFS):
        c[f"d_{o:g}"] = D[k][er, cs]
    pc, vc = P0[er, cs], V0[er, cs]
    rel = pc - pa[er]
    d = _norm(rel) + EPS
    relv = vc - va[er]
    c["closing"] = -(rel * relv).sum(1) / d                 # > 0: gap is shrinking
    c["cand_speed"] = _norm(vc)
    c["cand_toward"] = -(vc * rel).sum(1) / d               # candidate's own speed towards the actor
    c["ttc"] = np.clip(d / np.maximum(c["closing"], 0.1), 0, 20)
    c["side_goal"] = rel[:, 0] * sign[er] / d               # +1 candidate is goal-side, -1 behind the actor
    c["side_lat"] = np.abs(rel[:, 1]) / d
    spa = _norm(va)[er] + EPS
    c["cos_act_heading"] = (va[er] * rel).sum(1) / (spa * d)
    c["rank"] = np.tile(np.arange(P_K, dtype=np.float32), E)
    dmin = np.minimum.reduce([c[f"d_{o:g}"] for o in (-0.5, -0.25, 0.0, 0.25, 0.5)])
    c["d_min"] = dmin
    c["d_trend"] = c["d_-0.5"] - c["d_0"]
    c["d_trend_post"] = c["d_0"] - c["d_0.5"]
    c["d_ratio_nearest"] = c["d_0"] / (np.take_along_axis(d0, order[:, :1], 1)[er, 0] + EPS)
    c["cand_ball_d"] = _norm(pc - B0[er, :2])
    c["cand_ball_d_p5"] = _norm(sc.pos(0.5)[er, cs] - Bp5[er, :2])
    c["cand_ball_d_p1"] = _norm(sc.pos(1.0)[er, cs] - Bp1[er, :2])
    # passing lane: distance from the candidate to the segment actor -> ball one second later
    u = out_vec[er] / (out_len[er][:, None] + EPS)
    proj = (rel * u).sum(1)
    c["lane_proj"] = proj / (out_len[er] + EPS)
    c["lane_dist"] = _norm(rel - np.clip(proj, 0, out_len[er])[:, None] * u)
    c["cand_is_gk"] = m.is_gk[cs].astype(np.float32)
    c["cand_vis"] = (m.vis[idx[er], cs] == 1).astype(np.float32)
    c["cand_conf"] = m.conf[idx[er], cs].astype(np.float32)
    cand_df = pd.DataFrame(c)
    cand_df.loc[~ok, :] = np.nan
    cand_df["cand_ok"] = ok
    cand_slots = np.where(cand_ok, order, -1)
    return ev_df, cand_df, cand_slots


def pressure_tables(m: Match):
    """Training tables for one match: (event_df, candidate_df)."""
    rows = pressure_rows(m)
    ev_df, cand_df, slots = pressure_features(m, rows["idx"].values, rows["actor"].values,
                                              rows["pe_type"].values, rows["kind"].values, rows["hold"].values)
    E = len(rows)
    key = (m.gid * 100000 + np.arange(E)).astype(np.int64)
    ev_df.insert(0, "key", key)
    ev_df["gid"] = m.gid
    ev_df["ev_row"] = rows["ev_row"].values
    ev_df["idx"] = rows["idx"].values
    ev_df["actor"] = rows["actor"].values
    ev_df["y_type"] = rows["y_type"].values
    ev_df["presser"] = rows["presser"].values
    ev_df["presser_in_cands"] = (slots == rows["presser"].values[:, None]).any(1) & (rows["presser"].values >= 0)
    # flatten the 3 nearest candidates into the event row (for the type model)
    cf = [c for c in cand_df.columns if c != "cand_ok"]
    arr = cand_df[cf].to_numpy(np.float32).reshape(E, P_K, len(cf))
    flat = {f"c{r}_{name}": arr[:, r, i] for r in range(3) for i, name in enumerate(cf) if name != "rank"}
    ev_df = pd.concat([ev_df, pd.DataFrame(flat)], axis=1)
    # candidate table repeats the event-level context
    ctx = ev_df[[c for c in ev_df.columns if not c.startswith(("c0_", "c1_", "c2_")) and c not in
                 ("y_type", "presser", "presser_in_cands", "ev_row", "idx", "actor")]]
    cand = pd.concat([ctx.loc[np.repeat(np.arange(E), P_K)].reset_index(drop=True), cand_df], axis=1)
    cand["cand_slot"] = slots.reshape(-1)
    cand["y"] = (cand["cand_slot"].values == np.repeat(rows["presser"].values, P_K)) & (cand["cand_slot"].values >= 0)
    cand = cand[cand["cand_ok"]].drop(columns="cand_ok").reset_index(drop=True)
    return ev_df, cand


# --------------------------------------------------------------------------- #
# 2. CHALLENGES - participants
# --------------------------------------------------------------------------- #
C_PART_OFFS = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]
DUEL_TYPES = ("A", "FIFTY")


def challenge_rows(m: Match) -> pd.DataFrame:
    """Every PFF challenge with its two primary participants mapped to tracking slots."""
    ev = m.events[m.events["has_trk"]]
    ch = ev[ev["pe_possessionEventType"] == "CH"].copy()
    ge_player = m.slot_of(ch["ge_playerId"].tolist())
    role_cols = ["pe_carrierPlayerId", "pe_dribblerPlayerId", "pe_ballCarrierPlayerId", "pe_challengerPlayerId",
                 "pe_homeDuelPlayerId", "pe_awayDuelPlayerId", "pe_challengeKeeperPlayerId"]
    slots = {c: m.slot_of(ch[c].tolist()) if c in ch else np.full(len(ch), -1) for c in role_cols}
    home = np.full(len(ch), -1, np.int64)
    away = np.full(len(ch), -1, np.int64)
    for c in role_cols:
        s = slots[c]
        t = np.where(s >= 0, m.team[np.clip(s, 0, None)], -1)
        home = np.where((home < 0) & (t == 0), s, home)
        away = np.where((away < 0) & (t == 1), s, away)
    poss = np.maximum.reduce([slots["pe_carrierPlayerId"], slots["pe_dribblerPlayerId"], slots["pe_ballCarrierPlayerId"]])
    win = m.slot_of(ch["pe_challengeWinnerPlayerId"].tolist())
    out = pd.DataFrame(dict(
        ev_row=ch.index.values, gid=m.gid, idx=ch["i_event"].values, home=home, away=away, poss=ge_player,
        ctype=ch["pe_challengeType"].values, outcome=ch["pe_challengeOutcomeType"].values,
        dribble_type=ch["pe_dribbleType"].values, tackle_attempt=ch["pe_tackleAttemptType"].values,
        trick_type=ch["pe_trickType"].values,
        # 1 = the home participant is the one in possession (carrier / dribbler); NaN for duels
        home_has_ball=np.where(poss < 0, np.nan, (m.team[np.clip(poss, 0, None)] == 0).astype(float)),
        home_wins=np.where(win < 0, np.nan, (m.team[np.clip(win, 0, None)] == 0).astype(float)),
        foul_type=ch["foul_foulType"].values if "foul_foulType" in ch else None,
        offense=ch["foul_finalOffenseType"].values if "foul_finalOffenseType" in ch else None,
        foul_outcome=ch["foul_finalFoulOutcomeType"].values if "foul_finalFoulOutcomeType" in ch else None,
        culprit=m.slot_of(ch["foul_finalCulpritPlayerId"].tolist()) if "foul_finalCulpritPlayerId" in ch else -1,
        victim=m.slot_of(ch["foul_victimPlayerId"].tolist()) if "foul_victimPlayerId" in ch else -1,
    ))
    return out.reset_index(drop=True)


def _pair_feats(m: Match, sc: Scene, hs, as_, er):
    """Geometry of (home slot, away slot) pairs. ``er`` maps each pair to its scene row."""
    f = {}
    D = {}
    for o in sc.offsets:
        P = sc.pos(o)
        D[o] = _norm(P[er, hs] - P[er, as_])
        f[f"pd_{o:g}"] = D[o]
    core = [o for o in sc.offsets if -0.5 <= o <= 0.5]
    f["pd_min"] = np.minimum.reduce([D[o] for o in core])
    f["pd_trend_pre"] = D[-0.5] - D[0.0]
    f["pd_trend_post"] = D[0.5] - D[0.0]
    P0, V0, B0 = sc.pos(0.0), sc.vel(0.0), sc.ball(0.0)
    ph, pa, vh, va = P0[er, hs], P0[er, as_], V0[er, hs], V0[er, as_]
    rel = pa - ph
    d = _norm(rel) + EPS
    f["closing"] = -(rel * (va - vh)).sum(1) / d
    f["h_speed"], f["a_speed"] = _norm(vh), _norm(va)
    f["h_toward"] = (vh * rel).sum(1) / d
    f["a_toward"] = -(va * rel).sum(1) / d
    f["cos_vel"] = (vh * va).sum(1) / ((_norm(vh) + EPS) * (_norm(va) + EPS))
    for o in (-0.5, 0.0, 0.5):
        P, B = sc.pos(o), sc.ball(o)
        f[f"h_ball_{o:g}"] = _norm(P[er, hs] - B[er, :2])
        f[f"a_ball_{o:g}"] = _norm(P[er, as_] - B[er, :2])
    mid = (ph + pa) / 2
    f["mid_ball"] = _norm(mid - B0[er, :2])
    f["h_gk"], f["a_gk"] = m.is_gk[hs].astype(np.float32), m.is_gk[as_].astype(np.float32)
    return f, mid


def challenge_participant_table(m: Match, idx, gold_home=None, gold_away=None, poss=None) -> pd.DataFrame:
    """One row per (challenge frame, player on the pitch): is this player one of the two
    primary participants?  Ranking is done per team, so the answer is always one home
    and one away player.

    ``poss`` (optional): slot of the player Member 1's possession model says has the ball
    (-1 / None = unknown).  When given, distance-to-possessor features are added.
    """
    idx = np.asarray(idx, np.int64)
    sc = Scene(m, idx, C_PART_OFFS)
    E, P = sc.E, m.n_players
    rows = sc.rows
    poss = np.full(E, -1, np.int64) if poss is None else np.asarray(poss, np.int64)
    pc = np.clip(poss, 0, None)
    same = m.team[:, None] == m.team[None, :]                    # (P, P)
    f = {}
    DB, DN = {}, {}
    for o in C_PART_OFFS:
        Pp, B = sc.pos(o), sc.ball(o)
        DB[o] = _norm(Pp - B[:, None, :2])                       # (E, P) distance to the ball
        if o in (-0.5, 0.0, 0.5):
            dd = _norm(Pp[:, :, None, :] - Pp[:, None, :, :])    # (E, P, P)
            dd = np.where(same[None] | ~np.isfinite(dd), np.inf, dd)
            DN[o] = dd.min(2)                                    # nearest opponent
            if o == 0.0:
                nn = dd.argmin(2)
    P0, V0, B0 = sc.pos(0.0), sc.vel(0.0), sc.ball(0.0)
    present = np.isfinite(P0[..., 0])
    for o in (-0.5, -0.25, 0.0, 0.25, 0.5):
        f[f"db_{o:g}"] = DB[o]
    f["db_min"] = np.fmin.reduce([DB[o] for o in (-0.5, -0.25, 0.0, 0.25, 0.5)])
    f["db_trend"] = DB[-0.5] - DB[0.0]
    for o in (-0.5, 0.0, 0.5):
        f[f"dn_{o:g}"] = np.where(np.isfinite(DN[o]), DN[o], np.nan)
    f["dn_min"] = np.fmin.reduce([f[f"dn_{o:g}"] for o in (-0.5, 0.0, 0.5)])
    f["nn_ball"] = np.take_along_axis(DB[0.0], nn, 1)            # how close that opponent is to the ball
    relp = np.take_along_axis(P0, nn[..., None], 1) - P0
    relv = np.take_along_axis(V0, nn[..., None], 1) - V0
    f["nn_closing"] = -(relp * relv).sum(2) / (_norm(relp) + EPS)
    f["speed"] = _norm(V0)
    tb = B0[:, None, :2] - P0
    f["toward_ball"] = (V0 * tb).sum(2) / (_norm(tb) + EPS)

    def team_rank(x):                                            # rank within own team (0 = smallest)
        x = np.where(np.isfinite(x) & present, x, np.inf)
        r = np.zeros_like(x)
        for t in (0, 1):
            cols = np.flatnonzero(m.team == t)
            r[:, cols] = np.argsort(np.argsort(x[:, cols], 1), 1)
        return r

    f["rank_db"] = team_rank(DB[0.0])
    f["rank_db_min"] = team_rank(f["db_min"])
    f["rank_dn"] = team_rank(f["dn_min"])
    f["rank_sum"] = team_rank(np.nan_to_num(f["db_min"], nan=8.0) + np.nan_to_num(f["dn_min"], nan=8.0))
    J = sc.J
    f["vis"] = (m.vis[J[:, sc.k(0.0)]] == 1).astype(np.float32)
    f["conf"] = m.conf[J[:, sc.k(0.0)]].astype(np.float32)
    f["vis_frac"] = (m.vis[J] == 1).mean(1).astype(np.float32)
    f["is_gk"] = np.broadcast_to(m.is_gk.astype(np.float32), (E, P))
    # relation to the known possessor (optional)
    known = (poss >= 0)[:, None]
    dposs = _norm(P0 - P0[rows, pc][:, None, :])
    f["poss_known"] = np.broadcast_to(known.astype(np.float32), (E, P))
    f["is_poss"] = np.where(known, (np.arange(P)[None] == poss[:, None]).astype(np.float32), np.nan)
    f["poss_same_team"] = np.where(known, (m.team[None, :] == m.team[pc][:, None]).astype(np.float32), np.nan)
    f["d_poss"] = np.where(known, dposs, np.nan)
    dposs_m = _norm(sc.pos(-0.5) - sc.pos(-0.5)[rows, pc][:, None, :])
    dposs_p = _norm(sc.pos(0.5) - sc.pos(0.5)[rows, pc][:, None, :])
    f["d_poss_min"] = np.where(known, np.fmin.reduce([dposs, dposs_m, dposs_p]), np.nan)
    f["rank_d_poss"] = np.where(known, team_rank(np.where(known, dposs, np.inf)), np.nan)
    # frame-level context, repeated
    f["ball_z"] = np.broadcast_to(B0[:, 2:3], (E, P))
    f["ball_speed"] = np.broadcast_to(_norm(sc.bvel(0.0)[:, :2])[:, None], (E, P))
    f["ball_missing"] = np.broadcast_to((~np.isfinite(B0[:, :1])).astype(np.float32), (E, P))
    f["team"] = np.broadcast_to(m.team.astype(np.float32), (E, P))
    df = pd.DataFrame({k: np.asarray(v, np.float32).reshape(-1) for k, v in f.items()})
    df.insert(0, "key", np.repeat((m.gid * 100000 + rows).astype(np.int64), P) * 2 + np.tile(m.team, E))
    df["ev"] = np.repeat(rows, P)
    df["gid"] = m.gid
    df["slot"] = np.tile(np.arange(P), E)
    if gold_home is not None:
        gold = np.where(m.team[None, :] == 0, np.asarray(gold_home)[:, None], np.asarray(gold_away)[:, None])
        df["y"] = (np.arange(P)[None, :] == gold).reshape(-1)
        df["has_gold"] = np.broadcast_to((gold >= 0), (E, P)).reshape(-1)
    return df[present.reshape(-1)].reset_index(drop=True)


def pick_participants(part_df: pd.DataFrame, score: np.ndarray, n_events: int):
    """Top-scoring home and away player for each event -> (home_slots, away_slots)."""
    d = part_df[["ev", "team", "slot"]].copy()
    d["s"] = score
    best = d.sort_values("s", ascending=False).drop_duplicates(["ev", "team"])
    hs = np.full(n_events, -1, np.int64)
    as_ = np.full(n_events, -1, np.int64)
    h = best[best["team"] == 0]
    a = best[best["team"] == 1]
    hs[h["ev"].values] = h["slot"].values
    as_[a["ev"].values] = a["slot"].values
    return hs, as_


# --------------------------------------------------------------------------- #
# 3. CHALLENGES - attributes (type, who has the ball, winner, outcome, foul)
# --------------------------------------------------------------------------- #
C_ATTR_OFFS = [-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 3.0]


def challenge_attr_table(m: Match, idx, hs, as_, poss=None) -> pd.DataFrame:
    """One row per challenge given its frame and its (home, away) participants.

    ``poss``: optional slot of the player in possession according to Member 1's module."""
    idx = np.asarray(idx, np.int64)
    hs, as_ = np.asarray(hs, np.int64), np.asarray(as_, np.int64)
    sc = Scene(m, idx, C_ATTR_OFFS)
    E, rows = sc.E, sc.rows
    hs_c, as_c = np.clip(hs, 0, None), np.clip(as_, 0, None)
    f, mid = _pair_feats(m, sc, hs_c, as_c, rows)
    sgn = m.attack[idx].astype(np.float32)                      # home attacks +x * sgn
    P0, V0, B0 = sc.pos(0.0), sc.vel(0.0), sc.ball(0.0)
    ph, pa = P0[rows, hs_c], P0[rows, as_c]
    f.update(_pitch_feats(m, mid, sgn, "mid_"))                 # from the home team's point of view
    # who is goal-side of whom, seen from each attacking direction
    f["a_ahead_of_h"] = (pa[:, 0] - ph[:, 0]) * sgn             # > 0: away player is between home player and the away goal
    f["h_vx"] = V0[rows, hs_c][:, 0] * sgn
    f["a_vx"] = V0[rows, as_c][:, 0] * sgn
    # heading alignment before contact: tackles from behind have both moving the same way
    Vm = sc.vel(-0.5)
    vh, va = Vm[rows, hs_c], Vm[rows, as_c]
    rel = pa - ph
    d = _norm(rel) + EPS
    f["h_heading_to_a_pre"] = (vh * rel).sum(1) / ((_norm(vh) + EPS) * d)
    f["a_heading_to_h_pre"] = -(va * rel).sum(1) / ((_norm(va) + EPS) * d)
    f["h_speed_pre"], f["a_speed_pre"] = _norm(vh), _norm(va)
    # ball
    zs = []
    for o in C_ATTR_OFFS:
        B = sc.ball(o)
        if -1.0 <= o <= 1.0:
            f[f"ball_z_{o:g}"] = B[:, 2]
            zs.append(B[:, 2])
        f[f"ball_speed_{o:g}"] = _norm(sc.bvel(o)[:, :2])
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f["ball_z_max"] = np.nanmax(np.stack(zs, 1), 1)
    f["ball_vz"] = sc.bvel(0.0)[:, 2]
    f["ball_missing"] = (~np.isfinite(B0[:, 0])).astype(np.float32)
    f["ball_in_speed_vs_h"] = _norm(sc.bvel(-0.5)[:, :2] - sc.vel(-0.5)[rows, hs_c])
    f["ball_in_speed_vs_a"] = _norm(sc.bvel(-0.5)[:, :2] - sc.vel(-0.5)[rows, as_c])
    # possession before / after: which participant (or team) is nearest the ball
    for o in (-2.0, -1.0, -0.5, 1.0, 2.0, 3.0):
        j = sc.J[:, sc.k(o)]
        s, dist, team = _nearest_to_ball(m, j)
        tag = f"{'pre' if o < 0 else 'post'}{abs(o):g}"
        f[f"near_{tag}_team"] = np.where(team < 0, np.nan, team).astype(np.float32)   # 0 home, 1 away
        f[f"near_{tag}_dist"] = dist
        f[f"near_{tag}_is_h"] = (s == hs).astype(np.float32)
        f[f"near_{tag}_is_a"] = (s == as_).astype(np.float32)
        P, B = sc.pos(o), sc.ball(o)
        f[f"h_ball_{tag}"] = _norm(P[rows, hs_c] - B[:, :2])
        f[f"a_ball_{tag}"] = _norm(P[rows, as_c] - B[:, :2])
        if o > 0:
            f[f"pd_{tag}"] = _norm(P[rows, hs_c] - P[rows, as_c])
            f[f"h_speed_{tag}"] = _norm(sc.vel(o)[rows, hs_c])
            f[f"a_speed_{tag}"] = _norm(sc.vel(o)[rows, as_c])
            f[f"ball_x_adv_{tag}"] = (B[:, 0] - B0[:, 0]) * sgn      # > 0 ball moved towards the away goal
            f[f"ball_travel_{tag}"] = _norm(B[:, :2] - B0[:, :2])
    f.update(_stoppage_feats(m, idx))
    f["period"] = m.period[idx].astype(np.float32)
    dm = _norm(P0 - mid[:, None, :])
    f["n_within_3"] = (dm < 3).sum(1).astype(np.float32)
    f["n_within_6"] = (dm < 6).sum(1).astype(np.float32)
    f["h_vis"] = (m.vis[idx, hs_c] == 1).astype(np.float32)
    f["a_vis"] = (m.vis[idx, as_c] == 1).astype(np.float32)
    f["h_conf"] = m.conf[idx, hs_c].astype(np.float32)
    f["a_conf"] = m.conf[idx, as_c].astype(np.float32)
    if poss is None:
        poss = np.full(E, -1, np.int64)
    poss = np.asarray(poss, np.int64)
    pk = poss >= 0
    f["poss_known"] = pk.astype(np.float32)
    f["poss_is_h"] = np.where(pk, (poss == hs).astype(np.float32), np.nan)
    f["poss_is_a"] = np.where(pk, (poss == as_).astype(np.float32), np.nan)
    f["poss_team"] = np.where(pk, m.team[np.clip(poss, 0, None)].astype(np.float32), np.nan)
    df = pd.DataFrame(f)
    bad = (hs < 0) | (as_ < 0)
    df.loc[bad, [c for c in df.columns if c.startswith(("pd_", "h_", "a_", "closing", "cos_vel", "mid_"))]] = np.nan
    df["pair_missing"] = bad.astype(np.float32)
    df["gid"] = m.gid
    return df


# --------------------------------------------------------------------------- #
# 4. FOULS on any possession event (offside, handball, challenge fouls...)
# --------------------------------------------------------------------------- #
def foul_rows(m: Match) -> pd.DataFrame:
    ev = m.events[m.events["has_trk"]]
    pe = ev[ev["pe_possessionEventType"].notna() & (ev["pe_possessionEventType"] != "IT")]
    actor = m.slot_of(pe["ge_playerId"].tolist())
    g = lambda c: pe[c].values if c in pe else np.full(len(pe), None)
    out = pd.DataFrame(dict(
        ev_row=pe.index.values, gid=m.gid, idx=pe["i_event"].values, actor=actor,
        pe_type=pe["pe_possessionEventType"].values, foul_type=g("foul_foulType"),
        offense=g("foul_finalOffenseType"), foul_outcome=g("foul_finalFoulOutcomeType"),
        var=g("foul_var"), culprit=m.slot_of(pe["foul_finalCulpritPlayerId"].tolist()) if "foul_finalCulpritPlayerId" in pe else -1,
        victim=m.slot_of(pe["foul_victimPlayerId"].tolist()) if "foul_victimPlayerId" in pe else -1,
    ))
    return out.reset_index(drop=True)


def event_foul_table(m: Match, idx, actor, pe_type) -> pd.DataFrame:
    """Per-event features for "was there an infringement here" (incl. offside)."""
    idx = np.asarray(idx, np.int64)
    actor = np.asarray(actor, np.int64)
    E = len(idx)
    rows = np.arange(E)
    ac = np.clip(actor, 0, None)
    team_a = m.team[ac]
    sign = (m.attack[idx] * np.where(team_a == 0, 1, -1)).astype(np.float32)
    P0, V0, B0 = m.pos[idx], m.vel[idx], m.ball[idx]
    pa = P0[rows, ac]
    f = {}
    f.update(_pitch_feats(m, pa, sign, "act_"))
    f["act_speed"] = _norm(V0[rows, ac])
    f["pe_type"] = np.array([PE_TYPES.index(p) if p in PE_TYPES else -1 for p in pe_type], np.float32)
    f["ball_z"] = B0[:, 2]
    f["ball_speed"] = _norm(m.ball_vel[idx][:, :2])
    f["ball_act_dist"] = _norm(B0[:, :2] - pa)
    mate = (m.team[None, :] == team_a[:, None])
    opp = ~mate
    xs = P0[..., 0] * sign[:, None]                             # attack-normalised x of everybody
    d_act = _norm(P0 - pa[:, None, :])
    d_opp = np.where(opp & np.isfinite(d_act), d_act, np.inf)
    f["opp_d1"] = d_opp.min(1)
    f["n_opp_3"] = (d_opp < 3).sum(1).astype(np.float32)
    # ---- offside geometry at the moment of the event ----
    xo = np.where(opp & np.isfinite(xs), xs, -np.inf)
    xo_sorted = np.sort(xo, 1)
    second_last = xo_sorted[:, -2]                               # 2nd deepest defender (deepest is usually the GK)
    ball_x = np.where(np.isfinite(B0[:, 0]), B0[:, 0] * sign, pa[:, 0] * sign)
    line = np.maximum(np.maximum(second_last, ball_x), 0.0)      # offside line: 2nd last defender, ball, halfway
    xm = np.where(mate & np.isfinite(xs) & (np.arange(m.n_players)[None] != ac[:, None]), xs, -np.inf)
    margin = xm - line[:, None]                                  # > 0: that team-mate is in an offside position
    f["off_max_margin"] = np.where(np.isfinite(margin.max(1)), margin.max(1), np.nan)
    f["off_n_beyond"] = (margin > 0).sum(1).astype(np.float32)
    f["off_n_beyond_05"] = (margin > 0.5).sum(1).astype(np.float32)
    f["off_line_x"] = line
    f["off_line_minus_ball"] = line - ball_x
    # where does the ball go: is the player who ends up near it one of those beyond the line?
    for o in (1.0, 2.0, 3.0):
        j = m.shift(idx, o)
        s, dist, team = _nearest_to_ball(m, j)
        sc_ = np.clip(s, 0, None)
        f[f"recv_margin_{o:g}"] = np.where((s >= 0) & (team == team_a), margin[rows, sc_], np.nan)
        f[f"recv_same_team_{o:g}"] = np.where(s < 0, np.nan, (team == team_a)).astype(np.float32)
        f[f"recv_dist_{o:g}"] = dist
        f[f"ball_adv_{o:g}"] = (m.ball[j][:, 0] - B0[:, 0]) * sign
    f.update(_stoppage_feats(m, idx))
    f["period"] = m.period[idx].astype(np.float32)
    df = pd.DataFrame(f)
    df.loc[actor < 0, [c for c in df.columns if c.startswith("act_")]] = np.nan
    df["gid"] = m.gid
    return df


def foul_candidate_table(m: Match, idx, actor, culprit=None, victim=None, K=8) -> pd.DataFrame:
    """Players near an infringement, to rank as culprit / victim."""
    idx = np.asarray(idx, np.int64)
    actor = np.asarray(actor, np.int64)
    E = len(idx)
    rows = np.arange(E)
    ac = np.clip(actor, 0, None)
    team_a = m.team[ac]
    sign = (m.attack[idx] * np.where(team_a == 0, 1, -1)).astype(np.float32)
    P0, V0, B0 = m.pos[idx], m.vel[idx], m.ball[idx]
    ref = np.where(np.isfinite(P0[rows, ac]) & (actor >= 0)[:, None], P0[rows, ac], B0[:, :2])
    d = _norm(P0 - ref[:, None, :])
    xs = P0[..., 0] * sign[:, None]
    opp = m.team[None, :] != team_a[:, None]
    xo = np.sort(np.where(opp & np.isfinite(xs), xs, -np.inf), 1)[:, -2]
    ball_x = np.where(np.isfinite(B0[:, 0]), B0[:, 0] * sign, ref[:, 0] * sign)
    line = np.maximum(np.maximum(xo, ball_x), 0.0)
    margin = np.where(~opp, xs - line[:, None], np.nan)
    # candidates: the K nearest players + (always) the team-mate furthest beyond the offside line
    score = np.where(np.isfinite(d), d, np.inf)
    far = np.nanargmax(np.where(np.isfinite(margin), margin, -np.inf), 1)
    score[rows, far] = np.minimum(score[rows, far], 0.5)
    order = np.argsort(score, 1)[:, :K]
    ok = np.isfinite(np.take_along_axis(score, order, 1)).reshape(-1)
    er = np.repeat(rows, K)
    cs = order.reshape(-1)
    Pp = m.pos[m.shift(idx, 2.0)]
    Vp = m.vel[m.shift(idx, 2.0)]
    f = dict(
        d_ref=d[er, cs], rank=np.tile(np.arange(K, dtype=np.float32), E),
        is_actor=(cs == actor[er]).astype(np.float32), same_team=(~opp)[er, cs].astype(np.float32),
        speed=_norm(V0[er, cs]), speed_post2=_norm(Vp[er, cs]), moved_post2=_norm(Pp[er, cs] - P0[er, cs]),
        off_margin=margin[er, cs], ball_d=_norm(P0[er, cs] - B0[er, :2]), is_gk=m.is_gk[cs].astype(np.float32),
        x=xs[er, cs], toward_ref=((ref[er] - P0[er, cs]) * V0[er, cs]).sum(1) / (d[er, cs] + EPS),
    )
    df = pd.DataFrame(f)
    df.insert(0, "key", np.repeat((m.gid * 100000 + rows).astype(np.int64), K))
    df["gid"] = m.gid
    df["cand_slot"] = cs
    if culprit is not None:
        df["y_culprit"] = cs == np.repeat(np.asarray(culprit), K)
        df["y_victim"] = cs == np.repeat(np.asarray(victim), K)
    return df[ok].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 5. SPOTTING - per-frame features, tracking only
# --------------------------------------------------------------------------- #
F_LAGS = [-1.0, -0.6, -0.3, 0.3, 0.6, 1.0]


def frame_base_features(m: Match, j: np.ndarray) -> dict:
    """Instantaneous "is the ball being contested" descriptors for frames ``j``."""
    P, V, B, BV = m.pos[j], m.vel[j], m.ball[j], m.ball_vel[j]
    n = len(j)
    rows = np.arange(n)
    h = np.flatnonzero(m.team == 0)
    a = np.flatnonzero(m.team == 1)
    db = _norm(P - B[:, None, :2])
    db = np.where(np.isfinite(db), db, np.inf)
    f = {}
    dh, da = np.sort(db[:, h], 1), np.sort(db[:, a], 1)
    f["bh1"], f["ba1"] = dh[:, 0], da[:, 0]
    f["b_first"] = np.minimum(dh[:, 0], da[:, 0])
    f["b_second_team"] = np.maximum(dh[:, 0], da[:, 0])          # nearest player of the *other* team
    f["b_d2_any"] = np.sort(db, 1)[:, 1]
    for r in (1.5, 3.0, 5.0):
        f[f"n_ball_{r:g}"] = (db < r).sum(1).astype(np.float32)
    # the opposing pair that is closest together, favouring pairs near the ball
    pd_ = _norm(P[:, h][:, :, None, :] - P[:, a][:, None, :, :])
    pd_ = np.where(np.isfinite(pd_), pd_, np.inf)
    bt = np.minimum(np.minimum(db[:, h][:, :, None], db[:, a][:, None, :]), 12.0)
    bt = np.where(np.isfinite(B[:, 0])[:, None, None], bt, 3.0)
    flat = (pd_ + 0.25 * bt).reshape(n, -1)
    k = flat.argmin(1)
    hs, as_ = h[k // len(a)], a[k % len(a)]
    ph, pa, vh, va = P[rows, hs], P[rows, as_], V[rows, hs], V[rows, as_]
    rel = pa - ph
    d = _norm(rel)
    f["pair_d"] = d
    f["pair_closing"] = -(rel * (va - vh)).sum(1) / (d + EPS)
    f["pair_speed_max"] = np.maximum(_norm(vh), _norm(va))
    f["pair_speed_min"] = np.minimum(_norm(vh), _norm(va))
    f["pair_ball_min"] = np.minimum(db[rows, hs], db[rows, as_])
    f["pair_ball_max"] = np.maximum(db[rows, hs], db[rows, as_])
    f["pair_cos_vel"] = (vh * va).sum(1) / ((_norm(vh) + EPS) * (_norm(va) + EPS))
    # the player nearest the ball and the opponent nearest to HIM
    s_near = db.argmin(1)
    near_ok = np.isfinite(db[rows, s_near])
    f["near_team"] = np.where(near_ok, m.team[s_near], np.nan).astype(np.float32)
    f["near_speed"] = np.where(near_ok, _norm(V[rows, s_near]), np.nan)
    dn = _norm(P - P[rows, s_near][:, None, :])
    dn = np.where((m.team[None, :] != m.team[s_near][:, None]) & np.isfinite(dn), dn, np.inf)
    s_opp = dn.argmin(1)
    reln = P[rows, s_opp] - P[rows, s_near]
    dno = _norm(reln)
    f["near_opp_d"] = np.where(near_ok & np.isfinite(dno), dno, np.nan)
    f["near_opp_closing"] = np.where(near_ok, -(reln * (V[rows, s_opp] - V[rows, s_near])).sum(1) / (dno + EPS), np.nan)
    f["near_n_opp_3"] = np.where(near_ok, (dn < 3.0).sum(1), np.nan).astype(np.float32)
    f["near_vis"] = np.where(near_ok, m.vis[j, s_near] == 1, np.nan).astype(np.float32)
    f["pair_vis"] = ((m.vis[j, hs] == 1).astype(np.float32) + (m.vis[j, as_] == 1)) / 2
    f["ball_vx"], f["ball_vy"] = BV[:, 0], BV[:, 1]
    flat_pd = pd_.reshape(n, -1)
    f["n_pairs_2"] = (flat_pd < 2.0).sum(1).astype(np.float32)
    f["pair_d_global_min"] = flat_pd.min(1)
    f["ball_z"] = B[:, 2]
    f["ball_speed"] = _norm(BV[:, :2])
    f["ball_vz"] = BV[:, 2]
    f["ball_missing"] = (~np.isfinite(B[:, 0])).astype(np.float32)
    f["ball_interp"] = (m.ball_obs[j] == 4).astype(np.float32)
    sgn = m.attack[j].astype(np.float32)
    mid = (ph + pa) / 2
    f["mid_x_abs"] = np.abs(mid[:, 0])
    f["mid_y_abs"] = np.abs(mid[:, 1])
    f["mid_goal_d"] = np.hypot(m.half_len - np.abs(mid[:, 0]), mid[:, 1])
    f["mean_speed"] = np.nanmean(_norm(V), 1)
    f["out_margin"] = _out_margin(m, np.where(np.isfinite(B[:, :2]), B[:, :2], mid))
    for name in ("pair_d", "b_first", "b_second_team", "bh1", "ba1", "pair_ball_min", "pair_ball_max", "b_d2_any",
                 "pair_d_global_min"):
        f[name] = np.where(np.isfinite(f[name]), f[name], np.nan).astype(np.float32)
    f["_pair_h"], f["_pair_a"] = hs, as_
    return f


def frame_table(m: Match, step: int = 3, chunk: int = 20000) -> pd.DataFrame:
    """Per-frame feature table at ``fps / step`` Hz with lagged copies for temporal context."""
    j_all = np.arange(0, m.n_frames, step, dtype=np.int64)
    parts = []
    for s in range(0, len(j_all), chunk):
        parts.append(pd.DataFrame(frame_base_features(m, j_all[s:s + chunk])))
    base = pd.concat(parts, ignore_index=True)
    hz = m.fps / step
    per = m.period[j_all]
    out = {c: base[c].to_numpy() for c in base.columns}
    lag_cols = ["pair_d", "pair_closing", "b_first", "b_second_team", "ball_speed", "ball_z", "pair_speed_max",
                "pair_ball_min", "n_ball_3", "mean_speed", "ball_missing", "near_team", "near_opp_d", "ball_vx", "ball_vy"]
    n = len(base)
    ar = np.arange(n)
    for lag in F_LAGS:
        sh = int(round(lag * hz))
        src = np.clip(ar + sh, 0, n - 1)
        valid = (per[src] == per) & (ar + sh >= 0) & (ar + sh < n)
        tag = f"{'m' if lag < 0 else 'p'}{abs(lag):g}"
        for c in lag_cols:
            v = base[c].to_numpy()[src].astype(np.float32)
            v[~valid] = np.nan
            out[f"{c}_{tag}"] = v
    df = pd.DataFrame(out)
    # rolling extremes over +-0.5 s
    w = max(1, int(round(0.5 * hz)))
    for c in ("pair_d", "b_second_team"):
        df[f"{c}_rmin"] = df[c].rolling(2 * w + 1, center=True, min_periods=1).min().astype(np.float32)
    df["ball_z_rmax"] = df["ball_z"].rolling(2 * w + 1, center=True, min_periods=1).max().astype(np.float32)
    df["ball_speed_rstd"] = df["ball_speed"].rolling(2 * w + 1, center=True, min_periods=1).std().astype(np.float32)
    df["ball_dspeed"] = (df["ball_speed_p0.3"] - df["ball_speed_m0.3"]).astype(np.float32)
    # did the ball change direction / did the team nearest the ball change around this frame?
    for lag in ("0.3", "0.6", "1"):
        a_, b_ = df[[f"ball_vx_m{lag}", f"ball_vy_m{lag}"]].to_numpy(), df[[f"ball_vx_p{lag}", f"ball_vy_p{lag}"]].to_numpy()
        df[f"ball_turn_{lag}"] = ((a_ * b_).sum(1) / ((_norm(a_) + 0.5) * (_norm(b_) + 0.5))).astype(np.float32)
        df[f"team_switch_{lag}"] = (df[f"near_team_m{lag}"] != df[f"near_team_p{lag}"]).astype(np.float32).where(
            df[f"near_team_m{lag}"].notna() & df[f"near_team_p{lag}"].notna())
    df = df.drop(columns=[c for c in df.columns if c.startswith(("ball_vx", "ball_vy"))])
    df.insert(0, "j", j_all)
    df.insert(0, "gid", m.gid)
    df["period"] = per.astype(np.float32)
    return df
