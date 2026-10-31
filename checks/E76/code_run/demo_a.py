"""gate-code の検算（E76 関門 1）：設計 2.4 の解析が，小さく流した出力だけから CPU で出せるかを実際に計算して確かめる。
値は手順の確かめのためだけに出す（反復 2 つで，結論には使わない）。
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "measure"))
import ablate  # noqa: E402
import shift  # noqa: E402

CR = Path(__file__).resolve().parent
CG = [0.01, 0.1, 1.0, 10.0]
KS = ["1", "5", "10", "25", "50", "all"]
rng = np.random.default_rng(0)


def wuar(y, pred, w):
    """発話の重みつき UAR（重みが 0 の感情は除く）。"""
    rec = []
    for c in range(4):
        m = (y == c)
        tot = w[m].sum()
        if tot > 0:
            rec.append((w[m] * (pred[m] == c)).sum() / tot)
    return np.mean(rec), 4 - len(rec)


# ---- 1. 2 段のブートストラップ（AFEW，video，MOSEI のソース，K=5，③ − ①）
z = np.load(CR / "runA" / "video" / "AFEW.npz")
y, sp, reps = z["y"], z["speaker_group"], list(z["reps"])
spk = np.unique(sp)
k5 = KS.index("5")
per = {}
for r in reps:
    te = z[f"test_idx_r{r}"]
    C = float(z[f"t_C_r{r}"][k5])
    Lt = z[f"t_logit_r{r}"][k5, CG.index(C)]
    Lm = z[f"m_MOSEI_r{r}_logit"][k5]
    per[r] = (te, Lt.argmax(1), Lm.argmax(1))
point = np.mean([shift.uar(y[te], pm) - shift.uar(y[te], pt) for te, pt, pm in per.values()])
vals, dropped = [], 0
for b in range(500):
    wsp = dict(zip(spk, rng.multinomial(len(spk), np.ones(len(spk)) / len(spk))))
    rs = rng.choice(reps, size=len(reps))
    v = []
    for r in rs:
        te, pt, pm = per[r]
        w = np.array([wsp[s] for s in sp[te]], float)
        a, d1 = wuar(y[te], pm, w)
        c, d2 = wuar(y[te], pt, w)
        dropped += d1 + d2
        v.append(a - c)
    vals.append(np.mean(v))
print(f"[1] AFEW←MOSEI video K=5 ③−①：点 {point:+.4f}，2 段ブートストラップ 95% [{np.quantile(vals, .025):+.4f}, "
      f"{np.quantile(vals, .975):+.4f}]（B=500，除いた感情の回数 {dropped}，テストの話者群 {[len(np.unique(sp[per[r][0]])) for r in reps]}）")

# ---- 2. 性別（AFEW の gender を sample_id で貼る）
man = pd.read_csv(ROOT / "data" / "metadata" / "afew_utterance_manifest.csv", low_memory=False).set_index("sample_id")
g = man.gender.reindex(z["sample_id"]).to_numpy()
for r in reps:
    te, pt, pm = per[r]
    out = []
    for gg in ("Female", "Male"):
        m = g[te] == gg
        out.append(f"{gg} n={m.sum()} 話者 {len(np.unique(sp[te][m]))} ③−① {shift.uar(y[te][m], pm[m]) - shift.uar(y[te][m], pt[m]):+.3f}")
    print(f"[2] 反復 {r}：" + "，".join(out) + f"，gender の欠け {pd.isna(g).sum()}")

# ---- 3. 6 対を同時に直せる切片（② の全発話の点数，プールの全ラベルで切片だけを当てる）と l2 の感度（手 4）
for r in reps:
    te = z[f"test_idx_r{r}"]
    pool = np.setdiff1d(np.arange(len(y)), te)
    j = z["s_MOSEI__ver"][reps.index(r)]
    C = float(z[f"s_MOSEI__v{j}_C"])
    Ls = z[f"s_MOSEI__v{j}_logit"][CG.index(C)].astype(float)
    b_pool = ablate.fit_class_bias(Ls[pool], y[pool], l2=0.0)
    ki = z[f"kidx_r{r}_K5"]
    hs = {l2: shift.uar(y[te], (Ls[te] + ablate.fit_class_bias(Ls[ki], y[ki], l2=l2)).argmax(1)) for l2 in (1e-3, 1e-2, 1e-1)}
    print(f"[3] 反復 {r}：② {shift.uar(y[te], Ls[te].argmax(1)):.4f}，切片をプール全ラベルで {shift.uar(y[te], (Ls[te] + b_pool).argmax(1)):.4f}，"
          f"手 4 の l2 別 {', '.join(f'{k:g}:{v:.4f}' for k, v in hs.items())}")

# ---- 4. 感情の対ごとの d と o（E13 の定義をテストの発話で．② と ③）
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def d_o(L, yy):
    out = []
    for k, j in PAIRS:
        zk, zj = L[yy == k, k] - L[yy == k, j], L[yy == j, k] - L[yy == j, j]
        sd = np.sqrt((zk.var(ddof=1) + zj.var(ddof=1)) / 2)
        out.append(((zk.mean() - zj.mean()) / sd, -(zk.mean() + zj.mean()) / 2 / sd))
    return np.array(out)


r = reps[0]
te = z[f"test_idx_r{r}"]
j = z["s_MOSEI__ver"][0]
Ls = z[f"s_MOSEI__v{j}_logit"][CG.index(float(z[f"s_MOSEI__v{j}_C"]))]
print("[4] 反復 0 の ② の 6 対の |o| の平均", np.abs(d_o(Ls[te], y[te])[:, 1]).mean().round(3),
      "，③ K=5", np.abs(d_o(z[f"m_MOSEI_r{r}_logit"][k5], y[te])[:, 1]).mean().round(3),
      "，h2", np.abs(d_o(z[f"h2_MOSEI__v{j}_logit"][te], y[te])[:, 1]).mean().round(3))

# ---- 5. 台詞の重なり（text_roberta，IEMOCAP-script，K 本と同じ転写のテストの発話を除く）
zt = np.load(CR / "runB" / "text_roberta" / "IEMOCAP-script.npz")
mi = pd.read_csv(ROOT / "data" / "metadata" / "iemocap_utterance_manifest.csv", low_memory=False).set_index("sample_id")
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", str(s).lower())).strip()  # noqa: E731
tr = np.array([norm(s) for s in mi.transcript.reindex(zt["sample_id"])])
yt = zt["y"]
for r in list(zt["reps"]):
    te = zt[f"test_idx_r{r}"]
    for K in ("5", "50"):
        kk = KS.index(K)
        ki = zt[f"kidx_r{r}_K{K}"]
        keep = ~np.isin(tr[te], tr[ki])
        Ct = float(zt[f"t_C_r{r}"][kk])
        pt = zt[f"t_logit_r{r}"][kk, CG.index(Ct)].argmax(1)
        pm = zt[f"m_MELD_r{r}_logit"][kk].argmax(1)
        print(f"[5] 反復 {r} K={K}：除いた割合 {1 - keep.mean():.3f}，③−① 全テスト "
              f"{shift.uar(yt[te], pm) - shift.uar(yt[te], pt):+.4f}，重なり無し {shift.uar(yt[te][keep], pm[keep]) - shift.uar(yt[te][keep], pt[keep]):+.4f}")
