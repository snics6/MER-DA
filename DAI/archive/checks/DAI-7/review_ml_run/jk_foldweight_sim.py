"""ML の査読想定の役の第 3 回の検査（DAI-7）：jackknife の各回で，抜いた塊の分割の重みが (塊の数 − 1)/塊の数 に下がることの影響を人工の値で見る．
台本の fold_combine と jk_interval をそのまま呼ぶ．実データは読まない．
各分割の値は「塊ごとの値の発話数の重みつき平均」とし，塊を抜いた回はその塊を除いて作り直す（分割の値の作り直しは台本と同じ考え方）．
比べる 2 つ：(a) 台本どおりの重み，(b) 値のある分割をすべて重み 1 にした重み（点推定＝分割を等しく平均，を減らしたデータに当てた値）．"""
import sys
import numpy as np
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai7_probe_v2 as P

def run(sizes_per_fold, n_sim=2000, seed=0, fold_sd=0.03, cl_sd=0.02):
    rng = np.random.default_rng(seed)
    fold_of = np.concatenate([[f] * len(s) for f, s in enumerate(sizes_per_fold)])
    ncl = len(fold_of)
    n_utt = np.concatenate([np.asarray(s, float) for s in sizes_per_fold])
    n_eff = n_utt.sum() ** 2 / (n_utt ** 2).sum()
    w = 1.0 - np.eye(ncl)
    cover = {"a": 0, "b": 0}
    width = {"a": [], "b": []}
    for _ in range(n_sim):
        mu_f = rng.normal(0, fold_sd, len(sizes_per_fold))            # 分割の水準（真の値は 0）
        v_cl = mu_f[fold_of] + rng.normal(0, cl_sd, ncl) / np.sqrt(n_utt / 50)   # 塊の値（小さい塊ほど揺れる）
        per_fold = {}
        full = {}
        for f in range(len(sizes_per_fold)):
            ix = np.flatnonzero(fold_of == f)
            wf = w[:, ix] * n_utt[ix]
            with np.errstate(invalid="ignore", divide="ignore"):
                per_fold[f] = (wf @ v_cl[ix]) / wf.sum(1)
            full[f] = np.sum(n_utt[ix] * v_cl[ix]) / n_utt[ix].sum()
        est = np.mean(list(full.values()))
        th_a = P.fold_combine(per_fold, w, fold_of)
        V = np.stack([per_fold[f] for f in sorted(per_fold)], 1)
        th_b = np.nanmean(V, 1)                                         # 値のある分割を等しく平均
        for k, th in (("a", th_a), ("b", th_b)):
            lo, hi, _ = P.jk_interval(th, est, n_eff)
            cover[k] += (lo <= 0 <= hi)
            width[k].append(hi - lo)
    return n_eff, {k: cover[k] / n_sim for k in cover}, {k: float(np.median(width[k])) for k in width}

if __name__ == "__main__":
    cases = {
        "AFEW 風（分割 0 は塊 3 つで最大 97%，ほかは 11 塊）": [[154, 2, 2]] + [[20, 18, 16, 15, 14, 14, 13, 13, 12, 12, 11]] * 4,
        "MELD 風（分割 0〜3 は 66 塊で最大 75%，分割 4 は 2 塊で 50% ずつ）": [[1600] + [8] * 65] * 4 + [[1300, 1300]],
        "IEMOCAP 風（分割ごとに塊 1 つ）": [[600]] * 5,
    }
    for nm, s in cases.items():
        n_eff, cov, wid = run(s)
        print(f"{nm}：実効の塊の数 {n_eff:.2f}，含む割合 台本 {cov['a']:.3f} / 重み 1 {cov['b']:.3f}，幅の中央値 台本 {wid['a']:.4f} / 重み 1 {wid['b']:.4f}，比 {wid['a'] / wid['b']:.3f}")

    print("--- 分割の水準の揺れを 0 にした場合（塊の揺れだけ．台本の区間が表すと設計 8 節が書く揺れ）")
    for nm, s in cases.items():
        n_eff, cov, wid = run(s, fold_sd=0.0)
        print(f"{nm}：実効の塊の数 {n_eff:.2f}，含む割合 台本 {cov['a']:.3f} / 重み 1 {cov['b']:.3f}，幅の中央値 台本 {wid['a']:.4f} / 重み 1 {wid['b']:.4f}，比 {wid['a'] / wid['b']:.3f}")
