"""
Step 5 - use the trained models on a match (inference API for the team pipeline).

    from pff5.predict import Member5
    m5 = Member5("runs/raw")                         # folder written by pff5.train
    match = load_match("data/compact/3816.npz", with_events=False, ball_source="raw")

    ch   = m5.challenges_from_tracking(match)        # spot challenges + who / type / winner / outcome / foul
    pr   = m5.pressure(match, frame_idx, player_ids, pe_types)   # for events found by other members
    foul = m5.fouls(match, frame_idx, player_ids, pe_types)

Command line demo (writes PFF-style JSON):

    python -m pff5.predict --run runs/raw --compact data/compact --game 3816 --out pred_3816.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from . import features as F
from .data import Match, game_path, load_match
from .train import (ATTR_POSS, CH_OUTCOMES, CH_TYPES, DRIBBLE_TYPES, FOUL_OUTCOMES, FOUL_TYPES, OFFENSES, PART_POSS,
                    PRESSURE_LABELS, TACKLE_ATTEMPTS, peak_pick)


class Member5:
    def __init__(self, run_dir: str):
        import lightgbm as lgb
        self.dir = os.path.join(run_dir, "models")
        self._lgb = lgb
        self._cache = {}
        thr = os.path.join(self.dir, "spotting_threshold.json")
        self.spot_threshold = json.load(open(thr))["threshold"] if os.path.exists(thr) else 0.5
        self.ball = os.path.basename(os.path.normpath(run_dir))

    # ------------------------------------------------------------------ #
    def _model(self, name):
        if name not in self._cache:
            path = os.path.join(self.dir, f"{name}.txt")
            if not os.path.exists(path):
                self._cache[name] = (None, None)
            else:
                with open(path + ".json") as f:
                    meta = json.load(f)
                self._cache[name] = (self._lgb.Booster(model_file=path), meta)
        return self._cache[name]

    def _predict(self, name, df):
        bst, meta = self._model(name)
        if bst is None:
            return None, None
        X = df.reindex(columns=meta["features"]).to_numpy(np.float32)
        return bst.predict(X), meta

    @staticmethod
    def _pid(m: Match, slots):
        slots = np.asarray(slots)
        return np.where(slots >= 0, m.player_id[np.clip(slots, 0, None)], -1)

    # ------------------------------------------------------------------ #
    # pressure
    # ------------------------------------------------------------------ #
    def pressure(self, m: Match, idx, player_ids, pe_types=None, initial_touch=None, hold_s=None) -> pd.DataFrame:
        """Pressure on ``player_ids`` at frame indices ``idx``.

        pe_types: PFF possession event codes ("PA", "SH", ...) or None for initial touches.
        initial_touch: bool array - True when the frame is the first touch of a possession.
        hold_s: seconds the player has had the ball (0 for initial touches).
        """
        idx = np.asarray(idx, np.int64)
        actor = m.slot_of(list(player_ids))
        E = len(idx)
        kind = np.zeros(E) if initial_touch is None else 1.0 - np.asarray(initial_touch, float)
        if pe_types is None:
            pe_types = ["IT"] * E
        okr = actor >= 0
        out = pd.DataFrame(dict(frame_idx=idx, playerId=list(player_ids), pressureType="N", pressurePlayerId=-1,
                                p_N=1.0, p_A=0.0, p_L=0.0, p_P=0.0))
        if not okr.any():
            return out
        ev, cand, slots = F.pressure_features(m, idx[okr], actor[okr], np.asarray(pe_types, object)[okr], kind[okr],
                                              None if hold_s is None else np.asarray(hold_s, float)[okr])
        n = int(okr.sum())
        cf = [c for c in cand.columns if c != "cand_ok"]
        ctx = ev.loc[np.repeat(np.arange(n), F.P_K)].reset_index(drop=True)
        full = pd.concat([ctx, cand[cf]], axis=1)
        score, _ = self._predict("pressure_presser", full)
        score = np.where(cand["cand_ok"].values, score, -1.0).reshape(n, F.P_K)
        srt = -np.sort(-score, 1)
        arr = cand[cf].to_numpy(np.float32).reshape(n, F.P_K, len(cf))
        for r in range(3):
            for i, name in enumerate(cf):
                if name != "rank":
                    ev[f"c{r}_{name}"] = arr[:, r, i]
        ev["ps1"], ev["ps2"] = srt[:, 0], srt[:, 1]
        proba, _ = self._predict("pressure_type", ev)
        typ = proba.argmax(1)
        best = slots[np.arange(n), score.argmax(1)]
        w = np.flatnonzero(okr)
        out.loc[w, ["p_N", "p_A", "p_L", "p_P"]] = proba
        out.loc[w, "pressureType"] = np.array(PRESSURE_LABELS)[typ]
        out.loc[w, "pressurePlayerId"] = np.where(typ > 0, self._pid(m, best), -1)
        # core answer: pressured yes / no, with a confidence the team can threshold on
        out["pressured"] = (1.0 - out["p_N"].values) > 0.5
        out["pressured_confidence"] = np.maximum(out["p_N"].values, 1.0 - out["p_N"].values)
        return out

    # ------------------------------------------------------------------ #
    # challenges
    # ------------------------------------------------------------------ #
    def spot_challenges(self, m: Match, threshold=None) -> pd.DataFrame:
        """Tracking only -> frames where a challenge happens."""
        fr = F.frame_table(m)
        score, _ = self._predict("spotting", fr)
        dj, ds = peak_pick(score, fr["j"].to_numpy(), fr["period"].to_numpy(), m.fps)
        keep = ds >= (self.spot_threshold if threshold is None else threshold)
        o = np.argsort(dj[keep])
        return pd.DataFrame(dict(frame_idx=dj[keep][o], score=ds[keep][o]))

    def challenges(self, m: Match, idx, possessor_ids=None) -> pd.DataFrame:
        """Full PFF challenge record for each frame index in ``idx``.

        possessor_ids: optional player id in possession at each frame (from Member 1)."""
        idx = np.asarray(idx, np.int64)
        E = len(idx)
        use_poss = possessor_ids is not None
        poss = m.slot_of(list(possessor_ids)) if use_poss else None
        tag = f"poss{int(use_poss)}"
        part = F.challenge_participant_table(m, idx, poss=poss)
        sc, _ = self._predict(f"participants_{tag}", part.drop(columns=[] if use_poss else PART_POSS))
        hs, as_ = F.pick_participants(part, sc, E)
        attr = F.challenge_attr_table(m, idx, hs, as_, poss)
        if not use_poss:
            attr = attr.drop(columns=ATTR_POSS)
        out = pd.DataFrame(dict(frame_idx=idx, period=m.period[idx], videoTime=m.t[idx]))
        p_type, _ = self._predict(f"ch_type_{tag}", attr)
        typ = np.array(CH_TYPES)[p_type.argmax(1)]
        out["challengeType"] = typ
        out["challengeType_confidence"] = p_type.max(1)
        p_grp, mg = self._predict(f"core_group_{tag}", attr)   # core answer: aerial duel / ground duel / dribble / tackle
        if p_grp is not None:
            out["challengeGroup"] = np.array(mg["labels"])[p_grp.argmax(1)]
            out["challengeGroup_confidence"] = p_grp.max(1)
        duel = np.isin(typ, F.DUEL_TYPES)
        hb, _ = self._predict(f"ch_home_has_ball_{tag}", attr)
        hw, _ = self._predict(f"ch_home_wins_{tag}", attr)
        hid, aid = self._pid(m, hs), self._pid(m, as_)
        home_ball = hb > 0.5
        poss_id = np.where(home_ball, hid, aid)
        other_id = np.where(home_ball, aid, hid)
        is_drib = typ == "D"
        out["carrierPlayerId"] = np.where(~duel & ~is_drib, poss_id, -1)
        out["dribblerPlayerId"] = np.where(is_drib, poss_id, -1)
        out["ballCarrierPlayerId"] = np.where(~duel, poss_id, -1)
        out["challengerPlayerId"] = np.where(~duel, other_id, -1)
        out["homeDuelPlayerId"] = np.where(duel, hid, -1)
        out["awayDuelPlayerId"] = np.where(duel, aid, -1)
        out["challengeWinnerPlayerId"] = np.where(hw > 0.5, hid, aid)
        out["winner_confidence"] = np.maximum(hw, 1 - hw)
        out["ballSide_confidence"] = np.maximum(hb, 1 - hb)
        p_out, _ = self._predict(f"ch_outcome_{tag}", attr)
        out["challengeOutcomeType"] = np.array(CH_OUTCOMES)[p_out.argmax(1)]
        for name, labels, col in (("dribble_type", DRIBBLE_TYPES, "dribbleType"),
                                  ("tackle_attempt", TACKLE_ATTEMPTS, "tackleAttemptType")):
            p, _ = self._predict(f"ch_{name}_{tag}", attr)
            out[col] = np.where(is_drib, np.array(labels)[p.argmax(1)], None) if p is not None else None
        pf, _ = self._predict(f"ch_foul_type_{tag}", attr)
        if pf is not None:
            ft = np.array(FOUL_TYPES)[pf.argmax(1)]
            out["foulType"] = np.where(ft == "none", None, ft)
            out["p_foul"] = 1 - pf[:, 0]
            ch, _ = self._predict(f"ch_culprit_is_home_{tag}", attr)
            if ch is not None:
                is_f = ft != "none"
                out["culpritPlayerId"] = np.where(is_f, np.where(ch > 0.5, hid, aid), -1)
                out["victimPlayerId"] = np.where(is_f, np.where(ch > 0.5, aid, hid), -1)
            card, _ = self._predict(f"ch_card_given_foul_{tag}", attr)
            if card is not None:
                out["p_card_if_foul"] = card
        out["pressureType"] = "P"                      # PFF charts every challenge as "player pressured"
        out["pressurePlayerId"] = np.where(~duel, other_id, -1)
        return out

    def challenges_from_tracking(self, m: Match, threshold=None) -> pd.DataFrame:
        det = self.spot_challenges(m, threshold)
        if not len(det):
            return det
        rec = self.challenges(m, det["frame_idx"].values)
        rec.insert(1, "detection_score", det["score"].values)
        return rec

    # ------------------------------------------------------------------ #
    # fouls
    # ------------------------------------------------------------------ #
    def fouls(self, m: Match, idx, player_ids, pe_types) -> pd.DataFrame:
        """Infringement prediction for possession events found by the other members."""
        idx = np.asarray(idx, np.int64)
        actor = m.slot_of(list(player_ids))
        X = F.event_foul_table(m, idx, actor, list(pe_types))
        out = pd.DataFrame(dict(frame_idx=idx))
        p, _ = self._predict("foul_given", X)
        out["p_foul_given"] = p
        p2, _ = self._predict("foul_any", X)
        out["p_any_infringement"] = p2
        po, _ = self._predict("offside", X)
        out["p_offside"] = np.where(np.asarray(pe_types, object) == "CH", 0.0, po)
        for name, labels, col in (("foul_type_I_A_M", FOUL_TYPES[1:], "foulType"), ("offense_type", OFFENSES, "offenseType"),
                                  ("foul_outcome_cards", FOUL_OUTCOMES, "foulOutcomeType")):
            pp, _ = self._predict(f"foul_{name}", X)
            if pp is not None:
                out[col + "_if_infringement"] = np.array(labels)[pp.argmax(1)]
        cand = F.foul_candidate_table(m, idx, actor)
        cand["ev"] = (cand["key"] - m.gid * 100000).astype(int)
        for who in ("culprit", "victim"):
            s, _ = self._predict(f"foul_{who}", cand)
            if s is None:
                continue
            best = cand.assign(s=s).sort_values("s", ascending=False).drop_duplicates("ev").set_index("ev")["cand_slot"]
            slots = pd.Series(np.arange(len(idx))).map(best).fillna(-1).astype(int).values
            out[f"{who}PlayerId_if_infringement"] = self._pid(m, slots)
        return out


def _records(df: pd.DataFrame):
    out = []
    for r in df.to_dict("records"):
        out.append({k: (None if (isinstance(v, float) and np.isnan(v)) or v == -1 else
                        (v.item() if hasattr(v, "item") else v)) for k, v in r.items()})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="e.g. runs/raw")
    ap.add_argument("--compact", required=True)
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    m5 = Member5(a.run)
    m = load_match(game_path(a.compact, a.game), with_events=True, ball_source=m5.ball)
    result = dict(gameId=a.game, ball_source=m5.ball)
    result["challenges_from_tracking_only"] = _records(m5.challenges_from_tracking(m))
    if m.events is not None:
        # demo of the "other members give me events" interface, using PFF's own events as stand-ins
        rows = F.pressure_rows(m)
        pr = m5.pressure(m, rows["idx"].values, m.player_id[rows["actor"].values], rows["pe_type"].values,
                         rows["kind"].values == 0, rows["hold"].values)
        pr["gameEventId"] = m.events.loc[rows["ev_row"].values, "gameEventId"].values
        pr["possessionEventId"] = m.events.loc[rows["ev_row"].values, "possessionEventId"].values
        result["pressure_on_given_events"] = _records(pr)
        fr = F.foul_rows(m)
        fo = m5.fouls(m, fr["idx"].values, m.player_id[np.clip(fr["actor"].values, 0, None)], fr["pe_type"].values)
        fo["possessionEventId"] = m.events.loc[fr["ev_row"].values, "possessionEventId"].values
        result["fouls_on_given_events"] = _records(fo[fo["p_any_infringement"] > 0.2])
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1, default=str)
    print(f"wrote {a.out}: {len(result['challenges_from_tracking_only'])} challenges spotted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
