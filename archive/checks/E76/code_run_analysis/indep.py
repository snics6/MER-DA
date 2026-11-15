"""analyze_e76 の 2.4.1 の組の区間を，引き方（Draws）だけ借りて素朴なループで計算し直す（値は出さず差だけ出す）。"""
import sys, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
ROOT = Path("/home/snics/MER-DA"); sys.path.insert(0, str(ROOT/"exp"))
import analyze_e76 as AE
out = Path(sys.argv[1]); B = int(sys.argv[2]); names = sys.argv[3].split(","); pairs = [p.split(":") for p in sys.argv[4].split(",")]
kind = "audio_wavlm"
class C:
    def note_input(self, p): pass
from data import groups_for
groups = groups_for(kind)
tg = {g: AE.Target(C(), kind, g, ROOT/"results/e76", load_npz=True) for g in groups if g in names}
tgts = [tg[g] for g in groups if g in names]
dr = AE.Draws(kind, tgts, B, 76)
P = pd.read_csv(out/"e76a_1_pairs.csv", comment="#", dtype={"K": str})
def wuar(pred, y, w):
    rs = []
    for c in range(4):
        m = y == c
        den = w[m].sum()
        if den > 0: rs.append((w[m] * (pred[m] == c)).sum() / den)
    return np.mean(rs) if rs else np.nan
mx = {}
for T, S in pairs:
    t = tg[T]; z = t.z
    W = dr.W[t.corpus]
    for kk, K in enumerate(AE.KS):
        per_rep_full, per_rep_b = [], []
        jk_rep = []
        persons = dr.persons[t.corpus]
        for r in AE.REPS:
            te = t.te[r]; y = t.y[te]; pi = t.pidx[te]
            C_ = float(z[f"t_C_r{r}"][kk]); pt = z[f"t_logit_r{r}"][kk, AE.C_GRID.index(C_)].argmax(1)
            pm = z[f"m_{S}_r{r}_logit"][kk].argmax(1)
            per_rep_full.append(wuar(pm, y, np.ones(len(y))) - wuar(pt, y, np.ones(len(y))))
            vb = np.array([wuar(pm, y, W[b, pi].astype(float)) - wuar(pt, y, W[b, pi].astype(float)) for b in range(B)])
            per_rep_b.append(vb)
            jj = np.full(len(persons), np.nan)
            for g in range(len(persons)):
                keep = (pi != g).astype(float)
                if keep.sum() == 0: continue
                jj[g] = wuar(pm, y, keep) - wuar(pt, y, keep)
            jk_rep.append(jj)
        theta = np.mean(per_rep_full)
        Vb = np.stack(per_rep_b, 1)                         # (B, R)
        A = dr.A
        ok = np.isfinite(Vb)
        b2 = (A * np.where(ok, Vb, 0)).sum(1) / (A * ok).sum(1)
        lo, hi = np.quantile(b2, [0.025, 0.975])
        p = min(1, 2 * min(((b2 <= 0).sum() + 1) / (B + 1), ((b2 >= 0).sum() + 1) / (B + 1)))
        J = np.stack(jk_rep, 0)                              # (R, G)
        th = np.nanmean(J, 0)
        n_p = np.bincount(np.concatenate([t.pidx[t.te[r]] for r in AE.REPS]), minlength=len(persons)); n_p = n_p[n_p > 0]
        geff = n_p.sum() ** 2 / (n_p ** 2).sum()
        G = len(th); se = np.sqrt((G - 1) / G * ((th - th.mean()) ** 2).sum()); df = max(1, int(np.floor(geff)) - 1)
        tq = stats.t.ppf(0.975, df); jlo, jhi = theta - tq * se, theta + tq * se
        pj = 2 * stats.t.sf(abs(theta) / se, df)
        use = geff < 20
        LO, HI, PP = (min(lo, jlo), max(hi, jhi), max(p, pj)) if use else (lo, hi, p)
        row = P[(P.target == T) & (P.source == S) & (P.K == K)].iloc[0]
        for name, mine, theirs in (("theta", theta * 100, row.gain), ("boot_lo", lo * 100, row.boot_lo), ("boot_hi", hi * 100, row.boot_hi),
                                   ("p_boot", p, row.p_boot), ("jk_lo", jlo * 100, row.jk_lo), ("jk_hi", jhi * 100, row.jk_hi),
                                   ("lo", LO * 100, row.lo), ("hi", HI * 100, row.hi), ("p", PP, row.p), ("geff", geff, row.geff_min)):
            mx[(T, name)] = max(mx.get((T, name), 0), abs(mine - theirs))
        mx[(T, "use_jk_mismatch")] = mx.get((T, "use_jk_mismatch"), 0) + int(bool(row.use_jk) != use)
for k, v in sorted(mx.items()):
    print(k, f"{v:.2e}")
