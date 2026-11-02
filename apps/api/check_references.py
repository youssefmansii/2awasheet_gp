"""History check: how far were the in-house DAS / OBSO variants (used before the swap) from the supervisors' reference repos?

The engine now uses the references directly (accessible-space for DAS, C-OBSO for OBSO), so this script re-creates the OLD
in-house formulas from their parts (pitch control x xG x line weight for DAS; per-player control x pass chance x xG for OBSO)
and compares them with:
  DAS  : accessible-space (Bischofberger, MIT)   pip install accessible-space
  OBSO : C-OBSO (Fujii, MIT)  PPCF x Transition_gauss x EPV/max  (git clone https://github.com/keisuke198619/C-OBSO)

Run (about 15 seconds):   python check_references.py --api . --cobso /path/to/C-OBSO --step 600
For the CURRENT engine use validate_metrics.py, which checks the engine against the reference formulas exactly.
"""
import sys, time, argparse, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
ap = argparse.ArgumentParser()
ap.add_argument("--api", required=True); ap.add_argument("--cobso", required=True)
ap.add_argument("--step", type=int, default=600); ap.add_argument("--max", type=int, default=400)
ap.add_argument("--shots", type=int, default=1)
a = ap.parse_args()
sys.path.insert(0, a.api)
warnings.filterwarnings("ignore")
from data_loader import load_match
from metrics_engine import MetricsEngine, das_surface, xg_surface, pass_success_surface
import accessible_space as asp

md = load_match("3816"); eng = MetricsEngine(md); g = md.grid
fps = md.fps
T = np.loadtxt(f"{a.cobso}/Transition_gauss.csv", delimiter=","); T = T / T.max()          # 64 x 100
E = np.loadtxt(f"{a.cobso}/EPV_grid.csv", delimiter=",")

def ok(i):
    return eng.possession_side(i) is not None and eng.ball_xy(i) is not None

frames = [i for i in range(0, md.X.shape[0], a.step) if ok(i)][: a.max]
shot_frames = []
if a.shots:
    ev = md.events
    sh = ev[(ev["poss_type"] == "SH") & (ev["frame_idx"] >= 0)]
    for fi in sh["frame_idx"].astype(int):
        for sec in (2.0, 3.0, 4.0):
            j = int(fi - sec * fps)
            if j >= 0 and ok(j): shot_frames.append(j)
all_frames = sorted(set(frames) | set(shot_frames))
print("frames:", len(frames), "baseline +", len(shot_frames), "pre-shot ->", len(all_frames), flush=True)

# ---------------- ours + C-OBSO
rows = []
t0 = time.time()
for i in all_frames:
    fm = eng.frame_metrics(i, include_obso=True)
    pcr = eng.pitch_control(i, per_player=True)
    side = fm["attacking_side"]; adir = fm["attack_dir"]
    ball = eng.ball_xy(i)
    iy, ix = g.cell_index(ball[0], ball[1])
    Tc = T[31 - iy: 63 - iy, 49 - ix: 99 - ix]
    score = (E / E.max()) if adir == 1 else (E / E.max())[::-1, ::-1]
    cobso = pcr["pc_att"] * Tc * score                      # C-OBSO grid, 32 x 50
    # per-player (off-ball attackers): C-OBSO = value of the team grid at the player's own cell
    slots, is_att = pcr["slots"], pcr["is_att"]
    # the OLD in-house formulas, rebuilt from their parts
    xg_s = xg_surface(g, adir); ps_s = pass_success_surface(g, ball); threat = (ps_s / ps_s.sum()) * xg_s
    pos_ = np.column_stack([md.X[i, slots], md.Y[i, slots]]).astype(float)
    att_idx = np.flatnonzero(is_att); carrier = None
    if att_idx.size:
        dd = np.hypot(pos_[att_idx, 0] - ball[0], pos_[att_idx, 1] - ball[1])
        if dd.min() < 3.0: carrier = int(att_idx[dd.argmin()])
    ours_p = {int(slots[n]): float((pcr["ppcf_players"][n] * threat).sum()) for n in att_idx if n != carrier}
    legacy_das = float(das_surface(pcr["pc_att"], g, adir, fm["defensive_line_x"]).sum() * g.cell_area)
    sl, ov, cv = [], [], []
    for n in np.flatnonzero(is_att):
        s = int(slots[n])
        if s in ours_p:
            px, py = md.X[i, s], md.Y[i, s]
            jy, jx = g.cell_index(px, py)
            sl.append(s); ov.append(ours_p[s]); cv.append(cobso[jy, jx])
    prho = spearmanr(ov, cv)[0] if len(sl) >= 4 and np.std(ov) > 0 and np.std(cv) > 0 else np.nan
    rows.append(dict(frame=i, side=side, adir=adir, ours_das=legacy_das, ours_obso=float(sum(ours_p.values())),
                     cobso_total=float(cobso.sum()), cobso_max=float(cobso.max()), player_rho=prho))
res = pd.DataFrame(rows).set_index("frame")
print("ours + C-OBSO done in %.0fs" % (time.time() - t0), flush=True)

# ---------------- accessible-space DAS (one call, many frames)
recs = []
for i in all_frames:
    side = res.loc[i, "side"]; adir = int(res.loc[i, "adir"])
    slots = md.present_slots(i)
    sides = md.players["side"].to_numpy()[slots]
    bx, by = float(md.ball_x[i]), float(md.ball_y[i])
    d = np.hypot(md.X[i, slots] - bx, md.Y[i, slots] - by)
    d_att = np.where(sides == side, d, np.inf); carrier = int(slots[int(np.argmin(d_att))])
    for s, sd in zip(slots, sides):
        recs.append((i, int(md.period[i]), f"{sd}_{s}", sd, float(md.X[i, s]), float(md.Y[i, s]),
                     float(md.VX[i, s]), float(md.VY[i, s]), side, f"{side}_{carrier}", adir))
    j0, j1 = max(i - 1, 0), min(i + 1, md.ball_x.shape[0] - 1)
    bvx = (md.ball_x[j1] - md.ball_x[j0]) / ((j1 - j0) / fps); bvy = (md.ball_y[j1] - md.ball_y[j0]) / ((j1 - j0) / fps)
    recs.append((i, int(md.period[i]), "ball", None, bx, by, float(np.nan_to_num(bvx)), float(np.nan_to_num(bvy)), side, f"{side}_{carrier}", adir))
df = pd.DataFrame(recs, columns=["frame_id", "period_id", "player_id", "team_id", "x", "y", "vx", "vy",
                                 "team_in_possession", "player_in_possession", "adir"])
df = df.dropna(subset=["x", "y"]).copy()
df[["vx", "vy"]] = df[["vx", "vy"]].fillna(0.0)
t0 = time.time()
out = asp.get_dangerous_accessible_space(df, frame_col="frame_id", period_col="period_id", player_col="player_id",
        team_col="team_id", x_col="x", y_col="y", vx_col="vx", vy_col="vy",
        team_in_possession_col="team_in_possession", player_in_possession_col="player_in_possession",
        attacking_direction_col="adir", infer_attacking_direction=False, chunk_size=50, use_progress_bar=False)
tt = time.time() - t0
d1 = pd.DataFrame({"frame": df["frame_id"].to_numpy(), "lib_das": out.das.to_numpy(), "lib_as": out.acc_space.to_numpy()}).drop_duplicates("frame").set_index("frame")
res = res.join(d1)
print("library DAS on %d frames: %.1fs  (%.0f ms / frame)" % (len(res), tt, 1000 * tt / len(res)), flush=True)

# ---------------- comparisons
base = res.loc[[f for f in frames if f in res.index]]
def cmp(name, x, y):
    m = x.notna() & y.notna()
    print("  %-34s spearman %.3f | pearson %.3f | n=%d" % (name, spearmanr(x[m], y[m])[0], pearsonr(x[m], y[m])[0], m.sum()))
print("\n== agreement over %d regular frames (every %.0f s) ==" % (len(base), a.step / fps))
cmp("old DAS  vs  library DAS", base.ours_das, base.lib_das)
cmp("old DAS  vs  library AS (no danger)", base.ours_das, base.lib_as)
cmp("old OBSO total  vs  C-OBSO total", base.ours_obso, base.cobso_total)
print("  per-player OBSO ranks within a frame (ours vs C-OBSO): mean spearman %.3f over %d frames" % (base.player_rho.mean(), base.player_rho.notna().sum()))
print("  scale: median ours_das %.1f | lib_das %.1f | lib_as %.0f" % (base.ours_das.median(), base.lib_das.median(), base.lib_as.median()))
print("  scale: median ours_obso %.5f | C-OBSO total %.5f" % (base.ours_obso.median(), base.cobso_total.median()))

# ---------------- do they rise before shots?
pre = res.loc[sorted(set(shot_frames))]
def auc(col):
    x, y = pre[col].dropna().to_numpy(), base[col].dropna().to_numpy()
    if len(x) < 5 or len(y) < 5: return np.nan, np.nan
    u, p = mannwhitneyu(x, y, alternative="greater"); return u / (len(x) * len(y)), p
print("\n== higher 2-4 s before shots than at regular moments? (AUC 0.5 = no, 1 = always) ==")
for col, nm in [("ours_das", "old in-house DAS"), ("lib_das", "library DAS"), ("lib_as", "library AS (no danger)"), ("ours_obso", "old in-house OBSO"), ("cobso_total", "C-OBSO")]:
    A, p = auc(col); print("  %-24s AUC %.2f  p=%.4f" % (nm, A, p))
