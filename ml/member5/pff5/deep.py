"""
Step 4 (GPU) - spatio-temporal transformer over raw tracking windows.

Every sample is a short window of tracking (default 25 frames, 4-frame stride
= +-1.6 s) with 23 tokens per frame: 11 home players, 11 away players, the ball.
Layers alternate attention ACROSS PLAYERS inside a frame and ACROSS TIME for
each player, so the network can learn interactions such as "this defender is
closing down that attacker" without any hand-made feature.

Two stages:

    # 1) self-supervised pre-training on ALL tracking frames (no labels needed)
    python -m pff5.deep pretrain --compact data/compact --out runs/deep --ball raw

    # 2) multi-task fine-tuning on PFF labels, validating on one fold of matches
    python -m pff5.deep finetune --compact data/compact --tables data/tables --out runs/deep \
           --ball raw --val-fold 0 --init runs/deep/raw/pretrain.pt

Heads trained together: challenge spotting, challenge participants, challenge
type / outcome / who has the ball / winner / foul, pressure type and presser.
Validation predictions are saved so they can be blended with the LightGBM models
(see ensemble.py).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

from .data import game_path, list_games, load_match
from .models import fold_of
from .train import CH_OUTCOMES, CH_TYPES, PRESSURE_LABELS, codes

T_WIN, STRIDE, N_TOK, N_FEAT = 25, 4, 23, 10


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
class Store:
    """Keeps every match in RAM in a light form and cuts windows on demand."""

    def __init__(self, compact_dir, games, ball="raw", verbose=True):
        self.m = {}
        for i, g in enumerate(games):
            m = load_match(game_path(compact_dir, g), with_events=False, ball_source=ball)
            lo, hi = m._period_bounds()
            self.m[g] = dict(pos=m.pos, ball=m.ball, obs=m.ball_obs, vis=m.vis, conf=m.conf, period=m.period,
                             attack=m.attack, team=m.team, gk=m.is_gk, fps=m.fps, lo=lo, hi=hi,
                             hl=m.half_len, hw=m.half_wid, n=m.n_frames)
            if verbose and (i + 1) % 8 == 0:
                print(f"  loaded {i + 1}/{len(games)} matches", flush=True)

    def windows(self, gid, centers):
        """-> x (B,T,23,F) float32, slots (B,22) int64 (-1 = empty), sign (B,)"""
        d = self.m[gid]
        c = np.asarray(centers, np.int64)
        B = len(c)
        per = d["period"][c]
        J = c[:, None] + (np.arange(T_WIN) - T_WIN // 2)[None, :] * STRIDE
        J = np.clip(J, d["lo"][per][:, None], d["hi"][per][:, None])
        # the 11 + 11 players on the pitch at the centre frame
        present = np.isfinite(d["pos"][c][..., 0])
        slots = np.full((B, 22), -1, np.int64)
        for t in (0, 1):
            cols = np.flatnonzero(d["team"] == t)
            order = np.argsort(~present[:, cols], axis=1, kind="stable")[:, :11]
            chosen = cols[order]
            ok = np.take_along_axis(present[:, cols], order, 1)
            slots[:, t * 11:(t + 1) * 11] = np.where(ok, chosen, -1)
        sc = np.clip(slots, 0, None)
        sign = d["attack"][c].astype(np.float32)
        pos = d["pos"][J[:, :, None], sc[:, None, :]]                       # (B,T,22,2)
        k = 2
        Jp = np.clip(J + k, d["lo"][per][:, None], d["hi"][per][:, None])
        Jm = np.clip(J - k, d["lo"][per][:, None], d["hi"][per][:, None])
        dt = np.maximum((Jp - Jm), 1)[:, :, None, None] / d["fps"]
        vel = (d["pos"][Jp[:, :, None], sc[:, None, :]] - d["pos"][Jm[:, :, None], sc[:, None, :]]) / dt
        x = np.zeros((B, T_WIN, N_TOK, N_FEAT), np.float32)
        s = sign[:, None, None]
        ok = np.isfinite(pos[..., 0]) & (slots >= 0)[:, None, :]
        x[:, :, :22, 0] = np.where(ok, pos[..., 0] * s / d["hl"], 0)
        x[:, :, :22, 1] = np.where(ok, pos[..., 1] * s / d["hw"], 0)
        vok = ok & np.isfinite(vel[..., 0])
        x[:, :, :22, 2] = np.where(vok, vel[..., 0] * s / 8.0, 0)
        x[:, :, :22, 3] = np.where(vok, vel[..., 1] * s / 8.0, 0)
        x[:, :, :22, 4] = np.where(ok, d["vis"][J[:, :, None], sc[:, None, :]] == 1, 0)
        x[:, :, :22, 5] = np.where(ok, d["conf"][J[:, :, None], sc[:, None, :]] / 3.0, 0)
        x[:, :, :22, 6] = ok
        b = d["ball"][J]                                                     # (B,T,3)
        bv = (d["ball"][Jp] - d["ball"][Jm]) / dt[:, :, 0, :]
        bok = np.isfinite(b[..., 0])
        x[:, :, 22, 0] = np.where(bok, b[..., 0] * s[:, :, 0] / d["hl"], 0)
        x[:, :, 22, 1] = np.where(bok, b[..., 1] * s[:, :, 0] / d["hw"], 0)
        bvok = bok & np.isfinite(bv[..., 0])
        x[:, :, 22, 2] = np.where(bvok, bv[..., 0] * s[:, :, 0] / 15.0, 0)
        x[:, :, 22, 3] = np.where(bvok, bv[..., 1] * s[:, :, 0] / 15.0, 0)
        x[:, :, 22, 6] = bok
        x[:, :, 22, 7] = np.where(bok, b[..., 2] / 3.0, 0)
        x[:, :, 22, 8] = np.where(bvok, bv[..., 2] / 10.0, 0)
        o = d["obs"][J]
        x[:, :, 22, 9] = np.where((o == 1) | (o == 2), 1.0, np.where(o == 4, 0.5, 0.0))
        gk = np.where(slots >= 0, d["gk"][sc], False)
        return x, slots, gk


def token_of(slots, slot):
    """Position (0..21) of a tracking slot inside the window's token list, -1 if absent."""
    slot = np.asarray(slot)
    hit = slots == slot[:, None]
    return np.where(hit.any(1) & (slot >= 0), hit.argmax(1), -1)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def build_model(d_model=128, n_layers=4, n_heads=4, dropout=0.1):
    import torch
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            mk = lambda: nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout, batch_first=True,
                                                    norm_first=True, activation="gelu")
            self.space, self.time = mk(), mk()

        def forward(self, h, tok_pad):                       # h (B,T,N,D), tok_pad (B,N) True = empty token
            B, T, N, D = h.shape
            h = self.space(h.reshape(B * T, N, D), src_key_padding_mask=tok_pad.repeat_interleave(T, 0)).reshape(B, T, N, D)
            h = self.time(h.permute(0, 2, 1, 3).reshape(B * N, T, D)).reshape(B, N, T, D).permute(0, 2, 1, 3)
            return h

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(N_FEAT, d_model)
            self.kind = nn.Embedding(3, d_model)              # home / away / ball
            self.gk = nn.Embedding(2, d_model)
            self.tpos = nn.Parameter(torch.zeros(1, T_WIN, 1, d_model))
            self.mask_tok = nn.Parameter(torch.zeros(d_model))
            self.blocks = nn.ModuleList([Block() for _ in range(n_layers)])
            self.norm = nn.LayerNorm(d_model)
            mlp = lambda i, o: nn.Sequential(nn.Linear(i, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, o))
            self.recon = nn.Linear(d_model, 2)
            self.spot = mlp(2 * d_model, 1)
            self.part = mlp(2 * d_model, 1)
            self.ev = nn.ModuleDict(dict(ev_type=mlp(4 * d_model, len(CH_TYPES)), ev_outcome=mlp(4 * d_model, len(CH_OUTCOMES)),
                                         ev_has_ball=mlp(4 * d_model, 1), ev_wins=mlp(4 * d_model, 1), ev_foul=mlp(4 * d_model, 1)))
            self.presser = mlp(3 * d_model, 1)
            self.ptype = mlp(4 * d_model, len(PRESSURE_LABELS))
            nn.init.normal_(self.tpos, std=0.02)
            nn.init.normal_(self.mask_tok, std=0.02)

        def encode(self, x, gk, mask=None):
            """x (B,T,23,F), gk (B,22) bool, mask (B,T,23) bool = hidden entries (pre-training)."""
            B = x.shape[0]
            kind = torch.tensor([0] * 11 + [1] * 11 + [2], device=x.device)
            gk_full = torch.cat([gk.long(), torch.zeros(B, 1, dtype=torch.long, device=x.device)], 1)
            if mask is not None:
                x = x.masked_fill(mask[..., None], 0.0)
            h = self.inp(x) + self.kind(kind)[None, None] + self.gk(gk_full)[:, None] + self.tpos
            if mask is not None:
                h = h + mask[..., None] * self.mask_tok
            pad = x[:, T_WIN // 2, :, 6] < 0.5                # token absent at the centre frame
            pad[:, 22] = False                               # never drop the ball token
            if mask is not None:
                pad = pad & ~mask[:, T_WIN // 2]
            for b in self.blocks:
                h = b(h, pad)
            return self.norm(h), pad

        def center(self, h, pad):
            c = h[:, T_WIN // 2]                              # (B,23,D)
            w = (~pad).float()[..., None]
            g = (c * w).sum(1) / w.sum(1).clamp(min=1)
            return c, torch.cat([g, c[:, 22]], -1)            # tokens, global (2D)

    return Net()


def _gather_tok(c, idx):
    """c (B,23,D), idx (B,) token index (may be -1) -> (B,D), zeros where idx < 0."""
    import torch
    out = c[torch.arange(c.shape[0], device=c.device), idx.clamp(min=0)]
    return out * (idx >= 0).float()[:, None]


def heads_forward(net, h, pad, batch):
    """Run the heads needed for this batch. ``batch`` holds token indices (teacher forcing)."""
    import torch
    c, g = net.center(h, pad)
    out = dict(spot=net.spot(g).squeeze(-1))
    B = c.shape[0]
    part = net.part(torch.cat([c[:, :22], g[:, None, : c.shape[-1]].expand(-1, 22, -1)], -1)).squeeze(-1)
    part = part.masked_fill(pad[:, :22], -1e4)
    out["part_home"], out["part_away"] = part[:, :11], part[:, 11:]
    hp = batch.get("home_tok")
    ap = batch.get("away_tok")
    if hp is None:                                            # inference: use the model's own picks
        hp, ap = part[:, :11].argmax(1), part[:, 11:].argmax(1) + 11
    e = torch.cat([g, _gather_tok(c, hp), _gather_tok(c, ap)], -1)
    for k, head in net.ev.items():
        o = head(e)
        out[k[3:]] = o if o.shape[-1] > 1 else o.squeeze(-1)
    actor = batch.get("actor_tok")
    if actor is not None:
        a = _gather_tok(c, actor)
        pr = net.presser(torch.cat([c[:, :22], a[:, None].expand(-1, 22, -1),
                                    g[:, None, : c.shape[-1]].expand(-1, 22, -1)], -1)).squeeze(-1)
        same = (torch.arange(22, device=c.device)[None] // 11) == (actor.clamp(min=0)[:, None] // 11)
        pr = pr.masked_fill(pad[:, :22] | same, -1e4)
        out["presser"] = pr
        pooled = (torch.softmax(pr, 1)[..., None] * c[:, :22]).sum(1)
        out["ptype"] = net.ptype(torch.cat([g, a, pooled], -1))
    return out


# --------------------------------------------------------------------------- #
# sample lists
# --------------------------------------------------------------------------- #
def make_samples(tables, ball, games, spot_neg_keep=0.2):
    """Three aligned sample tables: spotting frames, challenges, pressure events."""
    from .dataset import load_table
    fr = load_table(tables, ball, "frames", games=games, columns=["gid", "j", "dt_ch"])
    a = fr["dt_ch"].abs().to_numpy()
    rng = np.random.RandomState(0)
    keep = ((a <= 0.5) | ((a > 1.0) & (a <= 2.5)) | ((a > 2.5) & (rng.rand(len(fr)) < spot_neg_keep)))
    spot = pd.DataFrame(dict(gid=fr["gid"].values[keep], idx=fr["j"].values[keep], y=(a[keep] <= 0.5).astype(np.float32)))
    ch = load_table(tables, ball, "ch_rows", games=games)
    ch = ch[(ch["home"] >= 0) & (ch["away"] >= 0)].reset_index(drop=True)
    ch["y_type"] = codes(ch["ctype"].values, CH_TYPES)
    ch["y_outcome"] = codes(ch["outcome"].values, CH_OUTCOMES)
    ch["y_foul"] = (ch["foul_type"] == "I").astype(np.float32)
    pe = load_table(tables, ball, "press_ev", games=games, columns=["gid", "key", "idx", "actor", "y_type", "presser"])
    pe["y"] = codes(pe["y_type"].values, PRESSURE_LABELS)
    pe = pe[pe["y"] >= 0].reset_index(drop=True)
    return spot, ch, pe


def _to_t(a, dev, dtype=None):
    import torch
    t = torch.from_numpy(np.array(a, copy=True)).to(dev)
    return t if dtype is None else t.to(dtype)


def _by_game(df):
    """Rows regrouped match by match: (original index in new order, regrouped frame, list of (gid, part))."""
    orig = df.index.values
    df = df.reset_index(drop=True)
    parts = [(gid, g) for gid, g in df.groupby("gid", sort=False)]
    order = np.concatenate([g.index.values for _, g in parts])
    return orig[order], df.iloc[order], parts


def batch_spot(store, df, dev):
    orig, d, parts = _by_game(df)
    xs, gks = [], []
    for gid, g in parts:
        x, slots, gk = store.windows(gid, g["idx"].values)
        xs.append(x)
        gks.append(gk)
    b = dict(x=_to_t(np.concatenate(xs), dev), gk=_to_t(np.concatenate(gks), dev))
    if "y" in d:
        b["y_spot"] = _to_t(d["y"].values.astype(np.float32), dev)
    b["_order"] = orig
    return b


def batch_ch(store, df, dev):
    orig, d, parts = _by_game(df)
    xs, gks, ht, at = [], [], [], []
    for gid, g in parts:
        x, slots, gk = store.windows(gid, g["idx"].values)
        xs.append(x)
        gks.append(gk)
        ht.append(token_of(slots, g["home"].values))
        at.append(token_of(slots, g["away"].values))
    b = dict(x=_to_t(np.concatenate(xs), dev), gk=_to_t(np.concatenate(gks), dev),
             home_tok=_to_t(np.concatenate(ht), dev), away_tok=_to_t(np.concatenate(at), dev),
             y_type=_to_t(d["y_type"].values, dev), y_outcome=_to_t(d["y_outcome"].values, dev),
             y_has_ball=_to_t(d["home_has_ball"].values.astype(np.float32), dev),
             y_wins=_to_t(d["home_wins"].values.astype(np.float32), dev), y_foul=_to_t(d["y_foul"].values.astype(np.float32), dev))
    b["_order"] = orig
    return b


def batch_press(store, df, dev):
    orig, d, parts = _by_game(df)
    xs, gks, at, pt = [], [], [], []
    for gid, g in parts:
        x, slots, gk = store.windows(gid, g["idx"].values)
        xs.append(x)
        gks.append(gk)
        at.append(token_of(slots, g["actor"].values))
        pt.append(token_of(slots, g["presser"].values))
    b = dict(x=_to_t(np.concatenate(xs), dev), gk=_to_t(np.concatenate(gks), dev),
             actor_tok=_to_t(np.concatenate(at), dev), presser_tok=_to_t(np.concatenate(pt), dev),
             y_ptype=_to_t(d["y"].values, dev))
    b["_order"] = orig
    return b


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #
def loss_ch(out, b):
    import torch
    import torch.nn.functional as Fn
    L = {}
    hv, av = b["home_tok"] >= 0, b["away_tok"] >= 0
    if hv.any():
        L["part"] = Fn.cross_entropy(out["part_home"][hv], b["home_tok"][hv])
    if av.any():
        L["part"] = L.get("part", 0) + Fn.cross_entropy(out["part_away"][av], b["away_tok"][av] - 11)
    for name, key in (("type", "y_type"), ("outcome", "y_outcome")):
        ok = b[key] >= 0
        if ok.any():
            L[name] = Fn.cross_entropy(out[name][ok], b[key][ok], label_smoothing=0.05)
    for name, key in (("has_ball", "y_has_ball"), ("wins", "y_wins"), ("foul", "y_foul")):
        ok = torch.isfinite(b[key])
        if ok.any():
            L[name] = Fn.binary_cross_entropy_with_logits(out[name][ok], b[key][ok])
    return L


def loss_press(out, b):
    import torch.nn.functional as Fn
    L = {}
    ok = b["actor_tok"] >= 0
    if ok.any():
        L["ptype"] = Fn.cross_entropy(out["ptype"][ok], b["y_ptype"][ok], label_smoothing=0.05)
    pk = ok & (b["presser_tok"] >= 0)
    if pk.any():
        L["presser"] = Fn.cross_entropy(out["presser"][pk], b["presser_tok"][pk])
    return L


# --------------------------------------------------------------------------- #
# stage 1: masked-trajectory pre-training
# --------------------------------------------------------------------------- #
def random_mask(x, rng, n_players=4, p_ball=0.5):
    """Hide a few whole trajectories for part of the window."""
    B = x.shape[0]
    mask = np.zeros((B, T_WIN, N_TOK), bool)
    for i in range(B):
        toks = list(rng.choice(22, n_players, replace=False))
        if rng.rand() < p_ball:
            toks.append(22)
        for t in toks:
            L = rng.randint(T_WIN // 3, T_WIN + 1)
            s = rng.randint(0, T_WIN - L + 1)
            mask[i, s:s + L, t] = True
    return mask


def pretrain(a):
    import torch
    import torch.nn.functional as Fn
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    games = [g for g in list_games(a.compact) if str(g) not in a.holdout.split(",")]
    if a.max_games:
        games = games[: a.max_games]
    out_dir = os.path.join(a.out, a.ball)
    os.makedirs(out_dir, exist_ok=True)
    store = Store(a.compact, games, a.ball)
    net = build_model(a.d_model, a.layers).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.05)
    steps = a.epochs * a.steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    rng = np.random.RandomState(0)
    # only sample frames where the ball is tracked (roughly "in play")
    pools = {g: np.flatnonzero(store.m[g]["obs"] > 0) for g in games}
    t0, run = time.time(), 0.0
    for step in range(steps):
        per = max(1, a.batch // 8)
        xs, gks = [], []
        for g in rng.choice(games, 8):
            x, _, gk = store.windows(g, rng.choice(pools[g], per))
            xs.append(x)
            gks.append(gk)
        x = np.concatenate(xs)
        mask = random_mask(x, rng)
        xt, mt, gt = _to_t(x, dev), _to_t(mask, dev), _to_t(np.concatenate(gks), dev)
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            h, _ = net.encode(xt, gt, mt)
            pred = net.recon(h)
            w = (mt & (xt[..., 6] > 0.5)).float()
            loss = (((pred - xt[..., :2]) ** 2).sum(-1) * w).sum() / w.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if (step + 1) % a.log_every == 0 or step == steps - 1:
            # error in metres (x is in half-pitch units)
            print(f"step {step + 1}/{steps}  masked-position loss {run:.5f}  (~{math.sqrt(run) * 43:.2f} m)  "
                  f"{time.time() - t0:.0f}s", flush=True)
            torch.save(dict(model=net.state_dict(), cfg=dict(d_model=a.d_model, layers=a.layers)),
                       os.path.join(out_dir, "pretrain.pt"))
    return 0


# --------------------------------------------------------------------------- #
# stage 2: multi-task fine-tuning
# --------------------------------------------------------------------------- #
def _predict(net, store, df, make_batch, dev, bs, teacher=False):
    import torch
    outs = {}
    net.eval()
    with torch.no_grad():
        for s in range(0, len(df), bs):
            b = make_batch(store, df.iloc[s:s + bs], dev)
            order = b.pop("_order")
            if not teacher:
                b.pop("home_tok", None), b.pop("away_tok", None)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                h, pad = net.encode(b["x"], b["gk"])
                o = heads_forward(net, h, pad, b)
            for k, v in o.items():
                outs.setdefault(k, []).append((order, v.float().cpu().numpy()))
    net.train()
    res = {}
    for k, parts in outs.items():
        idx = np.concatenate([p[0] for p in parts])
        val = np.concatenate([p[1] for p in parts])
        res[k] = pd.DataFrame(val.reshape(len(idx), -1), index=idx).loc[df.index.values].to_numpy()
    return res


def _sig(x):
    return 1 / (1 + np.exp(-x))


def _softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def evaluate(net, store, spot, ch, pe, dev, bs, compact=None, tables=None, ball=None, full_spot=True):
    from .models import binary_report, clf_report
    from .train import match_detections, peak_pick, spotting_metrics
    res, preds = {}, {}
    if len(ch):
        o = _predict(net, store, ch, batch_ch, dev, bs)
        ok = ch["y_type"].values >= 0
        res["ch_type"] = clf_report(ch["y_type"].values, _softmax(o["type"]), CH_TYPES, mask=ok)
        ok = ch["y_outcome"].values >= 0
        res["ch_outcome"] = clf_report(ch["y_outcome"].values, _softmax(o["outcome"]), CH_OUTCOMES, mask=ok)
        for name, col in (("has_ball", "home_has_ball"), ("wins", "home_wins"), ("foul", "y_foul")):
            v = ch[col].values.astype(float)
            m = np.isfinite(v)
            p = _sig(o[name][:, 0])
            r = binary_report(v[m].astype(int), p[m])
            r["accuracy_at_0.5"] = float(((p[m] > 0.5) == (v[m] == 1)).mean())
            res[f"ch_{name}"] = r
        # participants: need the token layout again
        hit_h = hit_a = n = 0
        for gid, g in ch.groupby("gid", sort=False):
            _, slots, _ = store.windows(gid, g["idx"].values)
            loc = ch.index.get_indexer(g.index)
            ph = slots[np.arange(len(g)), o["part_home"][loc].argmax(1)]
            pa = slots[np.arange(len(g)), o["part_away"][loc].argmax(1) + 11]
            hit_h += int((ph == g["home"].values).sum())
            hit_a += int((pa == g["away"].values).sum())
            n += len(g)
        res["participants_top1"] = dict(home=hit_h / max(n, 1), away=hit_a / max(n, 1), n=n)
        preds["ch"] = pd.DataFrame(dict(gid=ch["gid"].values, ev_row=ch["ev_row"].values,
                                        **{f"type_{c}": _softmax(o["type"])[:, i] for i, c in enumerate(CH_TYPES)},
                                        **{f"outcome_{c}": _softmax(o["outcome"])[:, i] for i, c in enumerate(CH_OUTCOMES)},
                                        has_ball=_sig(o["has_ball"][:, 0]), wins=_sig(o["wins"][:, 0]), foul=_sig(o["foul"][:, 0])))
    if len(pe):
        o = _predict(net, store, pe, batch_press, dev, bs)
        p = _softmax(o["ptype"])
        res["pressure_type"] = clf_report(pe["y"].values, p, PRESSURE_LABELS)
        hit = n = 0
        for gid, g in pe.groupby("gid", sort=False):
            _, slots, _ = store.windows(gid, g["idx"].values)
            loc = pe.index.get_indexer(g.index)
            pp = slots[np.arange(len(g)), o["presser"][loc].argmax(1)]
            has = g["presser"].values >= 0
            hit += int((pp[has] == g["presser"].values[has]).sum())
            n += int(has.sum())
        res["presser_top1"] = dict(accuracy=hit / max(n, 1), n=n)
        preds["pressure"] = pd.DataFrame(dict(gid=pe["gid"].values, key=pe["key"].values,
                                              **{f"p_{c}": p[:, i] for i, c in enumerate(PRESSURE_LABELS)}))
    if len(spot):
        o = _predict(net, store, spot, batch_spot, dev, bs)
        res["spot_frame_level"] = binary_report(spot["y"].values.astype(int), _sig(o["spot"][:, 0]))
    if full_spot and tables is not None:
        from .dataset import load_table
        dets, n_gt = [], 0
        for gid in sorted(ch["gid"].unique()) if len(ch) else []:
            fr = load_table(tables, ball, "frames", games=[gid], columns=["gid", "j", "period"])
            fr = fr.rename(columns={"j": "idx"})
            o = _predict(net, store, fr, batch_spot, dev, bs)
            sc = _sig(o["spot"][:, 0])
            fps = store.m[gid]["fps"]
            dj, ds = peak_pick(sc, fr["idx"].to_numpy(), fr["period"].to_numpy(), fps)
            gt = np.sort(load_table(tables, ball, "ch_rows", games=[gid], columns=["idx"])["idx"].to_numpy())
            n_gt += len(gt)
            d = pd.DataFrame(dict(gid=gid, j=dj, score=ds))
            for tol in (1, 2):
                d[f"tp_{tol}s"], _ = match_detections(dj, ds, gt, tol * fps)
            dets.append(d)
        if dets:
            dets = pd.concat(dets, ignore_index=True)
            res["spotting_1s"] = spotting_metrics(dets, n_gt, "1s")
            res["spotting_2s"] = spotting_metrics(dets, n_gt, "2s")
            preds["spotting"] = dets
    return res, preds


def finetune(a):
    import torch
    import torch.nn.functional as Fn
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from .dataset import table_games
    games = [g for g in table_games(a.tables, a.ball) if str(g) not in a.holdout.split(",")]
    if a.max_games:
        games = games[: a.max_games]
    folds = fold_of(np.array(games))
    val_games = [g for g, f in zip(games, folds) if f == a.val_fold] if len(games) > 1 else games
    tr_games = [g for g in games if g not in val_games] or games
    out_dir = os.path.join(a.out, a.ball)
    os.makedirs(out_dir, exist_ok=True)
    print(f"device {dev}; {len(tr_games)} train matches, {len(val_games)} validation matches", flush=True)
    store = Store(a.compact, games, a.ball)
    spot, ch, pe = make_samples(a.tables, a.ball, tr_games)
    vspot, vch, vpe = make_samples(a.tables, a.ball, val_games)
    cfg = dict(d_model=a.d_model, layers=a.layers)
    if a.init and os.path.exists(a.init):
        ck = torch.load(a.init, map_location="cpu")
        cfg = ck.get("cfg", cfg)
    net = build_model(cfg["d_model"], cfg["layers"]).to(dev)
    if a.init and os.path.exists(a.init):
        print("loading pre-trained weights:", net.load_state_dict(ck["model"], strict=False), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.05)
    steps = a.epochs * a.steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    rng = np.random.RandomState(a.seed)
    pos_w = float((1 - spot["y"].mean()) / max(spot["y"].mean(), 1e-3)) ** 0.5
    bs_s, bs_c, bs_p = a.batch, max(8, a.batch // 4), max(8, a.batch // 2)
    t0, hist, best = time.time(), {}, -1.0
    for step in range(steps):
        L = {}
        b = batch_spot(store, spot.iloc[rng.randint(0, len(spot), bs_s)], dev)
        b.pop("_order")
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            h, pad = net.encode(b["x"], b["gk"])
            c_, g_ = net.center(h, pad)
            L["spot"] = Fn.binary_cross_entropy_with_logits(net.spot(g_).squeeze(-1), b["y_spot"],
                                                            pos_weight=torch.tensor(pos_w, device=dev))
            if len(ch):
                bc = batch_ch(store, ch.iloc[rng.randint(0, len(ch), bs_c)], dev)
                bc.pop("_order")
                h, pad = net.encode(bc["x"], bc["gk"])
                L.update(loss_ch(heads_forward(net, h, pad, bc), bc))
            if len(pe):
                bp = batch_press(store, pe.iloc[rng.randint(0, len(pe), bs_p)], dev)
                bp.pop("_order")
                h, pad = net.encode(bp["x"], bp["gk"])
                L.update(loss_press(heads_forward(net, h, pad, bp), bp))
            loss = sum(L.values())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        for k, v in L.items():
            hist[k] = 0.97 * hist.get(k, v.item()) + 0.03 * v.item()
        if (step + 1) % a.log_every == 0:
            print(f"step {step + 1}/{steps} " + " ".join(f"{k}={v:.3f}" for k, v in hist.items()) +
                  f"  {time.time() - t0:.0f}s", flush=True)
        if (step + 1) % a.steps_per_epoch == 0 or step == steps - 1:
            last = step == steps - 1
            res, preds = evaluate(net, store, vspot.iloc[:: (1 if last else 4)], vch, vpe.iloc[:: (1 if last else 4)],
                                  dev, a.batch * 2, a.compact, a.tables, a.ball, full_spot=last)
            brief = dict(type_acc=res.get("ch_type", {}).get("accuracy"), outcome_acc=res.get("ch_outcome", {}).get("accuracy"),
                         part=res.get("participants_top1"), ptype_f1=res.get("pressure_type", {}).get("macro_f1"),
                         presser=res.get("presser_top1", {}).get("accuracy"),
                         spot_ap=res.get("spot_frame_level", {}).get("average_precision"))
            print(f"== epoch {(step + 1) // a.steps_per_epoch}: {json.dumps(brief, default=float)}", flush=True)
            torch.save(dict(model=net.state_dict(), cfg=cfg), os.path.join(out_dir, f"finetune_fold{a.val_fold}.pt"))
            if last:
                with open(os.path.join(out_dir, f"results_fold{a.val_fold}.json"), "w") as f:
                    json.dump(dict(val_games=val_games, **res), f, indent=1, default=float)
                for k, d in preds.items():
                    d.to_parquet(os.path.join(out_dir, f"pred_{k}_fold{a.val_fold}.parquet"), index=False)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["pretrain", "finetune"])
    ap.add_argument("--compact", required=True)
    ap.add_argument("--tables", default=None)
    ap.add_argument("--out", default="runs/deep")
    ap.add_argument("--ball", default="raw", choices=["raw", "smoothed"])
    ap.add_argument("--init", default=None)
    ap.add_argument("--val-fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps-per-epoch", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", default="", help="comma separated match ids never used (final test)")
    a = ap.parse_args(argv)
    return pretrain(a) if a.mode == "pretrain" else finetune(a)


if __name__ == "__main__":
    sys.exit(main())
