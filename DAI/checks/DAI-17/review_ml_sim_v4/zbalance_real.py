# dai-review-ml 第 4 版の確認：both を「問い合わせごとに標準化したコサインの平均」にしたとき，
# 上位 100 本がどちらの近さで決まるか（ラベルは使わない．CPU．gate-probe 第 3 回の cosscale.py と同じ近似）．
# 近似：ターゲットの標準化は候補でなく全発話，自分のコーパスの分類器の点は交差検証でなく全件の分類器．
import sys, time, numpy as np, torch
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
t0 = time.time()
import dai1, dai_data as DD, dai16_probe as P16
dai1.ENC["enc"] = DD.ENC1
T, srcs = "AFEW", ["IEMOCAP", "MELD"]
def std(X): return (X - X.mean(0)) / X.std(0).clamp(min=1e-6)
def nrm(E): return torch.nn.functional.normalize(torch.as_tensor(E, dtype=torch.float32), dim=1)
def zrow(C): return (C - C.mean(1, keepdims=True)) / C.std(1, keepdims=True)
def top(a, k=100): return np.argsort(-a, 1)[:, :k]
ov = lambda a, b: np.mean([len(np.intersect1d(x, y)) / a.shape[1] for x, y in zip(a, b)])
from scipy.stats import spearmanr
tg = dai1.Corpus.get(T)
rng = np.random.default_rng(0); q = rng.choice(len(tg.y), 300, replace=False)
store = {}
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
    zf, zl = zrow(cf), zrow(cl)
    store[m] = (zf, zl)
    zb, cb = (zf + zl) / 2, (cf + cl) / 2
    tz, tb, tf, tl = top(zb), top(cb), top(cf), top(cl)
    # 上位 100 本の中の 2 つの z の平均（どちらの近さで上位に来たか）
    zf_in = np.take_along_axis(zf, tz, 1).mean(); zl_in = np.take_along_axis(zl, tz, 1).mean()
    # 片方の z の上限（行ごとの最大の中央値）と上位 100 番目の値
    print(f"[{m}] ソース {Fs.shape[0]} 本．行の中の標準偏差の中央値 feat {np.median(cf.std(1)):.3f} logit {np.median(cl.std(1)):.3f}")
    print(f"  z の行ごとの最大の中央値：feat {np.median(zf.max(1)):.2f}，logit {np.median(zl.max(1)):.2f}．100 番目：feat {np.median(np.sort(zf,1)[:, -100]):.2f}，logit {np.median(np.sort(zl,1)[:, -100]):.2f}")
    print(f"  上位 100 本の重なり：z平均と feat {ov(tz, tf):.2f}，z平均と logit {ov(tz, tl):.2f}，feat と logit {ov(tf, tl):.2f}，z平均と旧 both {ov(tz, tb):.2f}")
    print(f"  z平均の上位 100 本の中の z の平均：feat {zf_in:.2f}，logit {zl_in:.2f}．特徴だけの上位100本の feat z 平均 {np.take_along_axis(zf, tf, 1).mean():.2f}，答えだけの上位100本の logit z 平均 {np.take_along_axis(zl, tl, 1).mean():.2f}")
    r = np.array([(spearmanr(zb[i], cf[i])[0], spearmanr(zb[i], cl[i])[0]) for i in range(30)])
    print(f"  行の中の順位相関の中央値：z平均と feat {np.median(r[:,0]):.2f}，z平均と logit {np.median(r[:,1]):.2f}")
    # 上位 100 本の中の順位相関（上位の並びをどちらが決めるか）
    rt = np.array([(spearmanr(zb[i, tz[i]], zf[i, tz[i]])[0], spearmanr(zb[i, tz[i]], zl[i, tz[i]])[0]) for i in range(300)])
    print(f"  z平均の上位 100 本の中の順位相関の中央値：feat {np.nanmedian(rt[:,0]):.2f}，logit {np.nanmedian(rt[:,1]):.2f}")
# joint：6 つ
zj = sum(store[m][0] + store[m][1] for m in "AVT") / 6
tj = top(zj)
for m in "AVT":
    zf, zl = store[m]
    print(f"[J] 上位 100 本の中の z の平均：{m} feat {np.take_along_axis(zf, tj, 1).mean():.2f}，{m} logit {np.take_along_axis(zl, tj, 1).mean():.2f}")
print(f"{time.time()-t0:.0f} 秒")
