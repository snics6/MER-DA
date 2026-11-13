# 第 4 版の both（標準化したコサインの平均）と joint で引いた上位 100 本が，特徴と分類器の答えのそれぞれで表の上位何 % にあるか（ラベルは使わない．CPU．近似は zbalance_real.py と同じ）
import sys, numpy as np, torch
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai1, dai_data as DD, dai16_probe as P16
dai1.ENC["enc"] = DD.ENC1
T, srcs = "AFEW", ["IEMOCAP", "MELD"]
def std(X): return (X - X.mean(0)) / X.std(0).clamp(min=1e-6)
def nrm(E): return torch.nn.functional.normalize(torch.as_tensor(E, dtype=torch.float32), dim=1)
def zrow(C): return (C - C.mean(1, keepdims=True)) / C.std(1, keepdims=True)
def top(a, k=100): return np.argsort(-a, 1)[:, :k]
def pct(C):  # 行の中の上位からの割合（0 = 1 番）
    r = np.argsort(np.argsort(-C, 1), 1); return r / C.shape[1]
tg = dai1.Corpus.get(T)
q = np.random.default_rng(0).choice(len(tg.y), 300, replace=False)
P, Z = {}, {}
for m in "AVT":
    Ft = nrm(std(tg.X[m]).float().cpu())
    Et = np.concatenate([P16.centered(P16.src_probs_on(s, T, m)) for s in srcs], 1)
    Lt = nrm((Et - Et.mean(0)) / Et.std(0))
    Fs, Ls = [], []
    for s in srcs:
        cp = dai1.Corpus.get(s)
        Fs.append(nrm(std(cp.X[m]).float().cpu()))
        E = np.concatenate([P16.centered(P16.src_probs_on(u, s, m)) for u in srcs], 1)
        Ls.append(nrm((E - E.mean(0)) / E.std(0)))
    Fs, Ls = torch.cat(Fs), torch.cat(Ls)
    cf = (Ft[q] @ Fs.T).numpy(); cl = (Lt[q] @ Ls.T).numpy()
    Z[m] = (zrow(cf), zrow(cl)); P[m] = (pct(cf), pct(cl))
    for nm, sc in (("標準化の平均（第 4 版）", (Z[m][0] + Z[m][1]) / 2), ("コサインの平均（第 3 版）", (cf + cl) / 2)):
        t = top(sc)
        pf, pl = np.take_along_axis(P[m][0], t, 1), np.take_along_axis(P[m][1], t, 1)
        print(f"[{m}] {nm}：上位 100 本の位置の中央値 特徴 {100*np.median(pf):.2f}% 答え {100*np.median(pl):.2f}%．"
              f"両方とも上位 5% に入る割合 {np.mean((pf < .05) & (pl < .05)):.2f}，特徴だけ上位 5% {np.mean((pf < .05) & (pl >= .05)):.2f}，答えだけ上位 5% {np.mean((pf >= .05) & (pl < .05)):.2f}．"
              f"答えが上位 20% の外 {np.mean(pl >= .2):.2f}，特徴が上位 20% の外 {np.mean(pf >= .2):.2f}")
zj = sum(Z[m][0] + Z[m][1] for m in "AVT") / 6
t = top(zj)
for m in "AVT":
    pf, pl = np.take_along_axis(P[m][0], t, 1), np.take_along_axis(P[m][1], t, 1)
    print(f"[J の上位 100 本] {m}：位置の中央値 特徴 {100*np.median(pf):.2f}% 答え {100*np.median(pl):.2f}%．上位 20% の外：特徴 {np.mean(pf >= .2):.2f} 答え {np.mean(pl >= .2):.2f}")
