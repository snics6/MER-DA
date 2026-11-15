"""DAI-7 の正の計算の保存（results/dai7_probe_v2/fits）から，融合で得が足し合わさらない分が
「UAR の頭打ち」か「得の重なり」かを見る探索（関門を通していない．2026-11-08 研究者の依頼）．

量（どれも (c, K, 組) ごとに 30 単位の平均．範囲は 5 分割の値（各 6 単位の平均）の最小と最大）：
  (a) 余地：融合 prod[none] が外した発話のうち，prod[oracle] が当てる割合と prod[joint_a1] が当てる割合．
      3 つの単独の分類器のどれかが当てる発話の割合（感情で平均．UAR と同じ重み）．
  (b) 重なり：単独 m が学び直しで新しく当てた発話（none_m で外れ，joint_m で当たり）のうち，
      prod[none] がもともと当てていた割合．比べる基準は全部の発話のうち prod[none] が当てる割合．
      prod[joint] の発話ごとの出入り（新しく当てた，新しく外した）．
  (c) 判定によらない指標：感情ごとの one-vs-rest の AUC の平均（macro AUC）と，
      正解の感情の確率の対数の感情ごとの平均（balanced log-loss）．単独 3 つと prod．
      得 = joint_a1 − none，足し合わさらない分 S = 単独の得の平均 − prod の得．
  (d) prod の確率の過信：温度 3 で割った prod（3 つの確率の幾何平均．選ばない固定の値）の balanced log-loss，
      prod の最大確率の平均（全部と，外した発話），正解率（感情で重みを付けない）．
  (e) 1 つのほかのモダリティの疑似ラベルだけで学んだ単独の分類器（tt_single_<源>，ρ = 1，a = 1）の UAR．
確率は z_m Hᵀ の softmax（H は台本と同じ）．作り直した判定が保存の判定と一致するかを最初に確かめる．
"""
import argparse, glob, json, os, subprocess, time
import numpy as np, pandas as pd

H = np.linalg.svd(np.eye(4) - 0.25)[0][:, :3]
SETS = ["none", "joint_a1", "oracle"]
KS = [0, 1, 5, 10, 25]


def softmax(a):
    a = a - a.max(1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(1, keepdims=True)


def uar(y, pred):
    return float(np.mean([np.mean(pred[y == c] == c) for c in range(4)]))


def bal_rate(y, ind):
    return float(np.mean([np.mean(ind[y == c]) for c in range(4)]))


def macro_auc(y, P):
    out = []
    for c in range(4):
        pos = y == c
        s = P[:, c]
        r = pd.Series(s).rank().to_numpy()
        n1, n0 = pos.sum(), (~pos).sum()
        out.append((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return float(np.mean(out))


def bal_logloss(y, P):
    lp = -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None))
    return float(np.mean([lp[y == c].mean() for c in range(4)]))


def unit_rows(path):
    f = np.load(path, allow_pickle=False)
    y = f["test_y"].astype(int)
    name = os.path.basename(path)[:-4]
    T, c, fo, r = name.split("_")
    rows, checks = [], []
    for K in KS:
        P, pred = {}, {}
        for s in SETS:
            mods = str(f[f"K{K}_{s}_mods"])
            z = f[f"K{K}_{s}_z"].astype(np.float64)
            Zsum = 0
            for j, m in enumerate(mods):
                Zm = z[:, 3 * j:3 * j + 3] @ H.T
                P[s, m] = softmax(Zm)
                pred[s, m] = P[s, m].argmax(1)
                checks.append(int((pred[s, m] != f[f"K{K}_{s}_pred_{m}"]).sum()))
                Zsum = Zsum + Zm
            P[s, "prod"] = softmax(Zsum)
            pred[s, "prod"] = P[s, "prod"].argmax(1)
            checks.append(int((pred[s, "prod"] != f[f"K{K}_{s}_pred_prod"]).sum()))
            pred[s, "full"] = f[f"K{K}_{s}_pred_full"].astype(int)
        row = dict(T=T, c=c, fold=int(fo[1:]), seed=int(r[1:]), K=K, n_test=len(y))
        for s in SETS:
            for m in ["A", "V", "T", "prod"]:
                row[f"uar_{s}_{m}"] = uar(y, pred[s, m])
                row[f"auc_{s}_{m}"] = macro_auc(y, P[s, m])
                row[f"ll_{s}_{m}"] = bal_logloss(y, P[s, m])
            row[f"uar_{s}_full"] = uar(y, pred[s, "full"])
            anyc = np.zeros(len(y), bool)
            for m in "AVT":
                anyc |= pred[s, m] == y
            row[f"any_{s}"] = bal_rate(y, anyc)
        # (a) 余地
        wrong_none = pred["none", "prod"] != y
        for s in ["joint_a1", "oracle"]:
            row[f"room_{s}"] = float(np.mean(pred[s, "prod"][wrong_none] == y[wrong_none]))
        row["n_wrong_none_prod"] = int(wrong_none.sum())
        # (b) 重なり
        right_none_prod = pred["none", "prod"] == y
        row["base_right_none_prod"] = float(right_none_prod.mean())
        for m in "AVT":
            new = (pred["none", m] != y) & (pred["joint_a1", m] == y)
            row[f"n_new_{m}"] = int(new.sum())
            row[f"already_{m}"] = float(right_none_prod[new].mean()) if new.any() else np.nan
        newp = (pred["none", "prod"] != y) & (pred["joint_a1", "prod"] == y)
        lostp = (pred["none", "prod"] == y) & (pred["joint_a1", "prod"] != y)
        row["prod_new_right"] = float(newp.mean())
        row["prod_new_wrong"] = float(lostp.mean())
        # (d) 過信
        for s in SETS:
            mods = str(f[f"K{K}_{s}_mods"])
            z = f[f"K{K}_{s}_z"].astype(np.float64)
            Zs = sum(z[:, 3 * j:3 * j + 3] @ H.T for j in range(len(mods)))
            P3 = softmax(Zs / 3.0)
            row[f"ll3_{s}_prod"] = bal_logloss(y, P3)
            conf = P[s, "prod"].max(1)
            row[f"conf_{s}_prod"] = float(conf.mean())
            row[f"wrongconf_{s}_prod"] = float(conf[pred[s, "prod"] != y].mean())
            row[f"acc_{s}_prod"] = float(np.mean(pred[s, "prod"] == y))
        # (e) 1 つのほかのモダリティの疑似ラベル
        for src in "AVT":
            for m in "AVT":
                if m == src:
                    continue
                row[f"uar_single_{src}_to_{m}"] = uar(y, f[f"K{K}_tt_single_{src}_pred_{m}"].astype(int))
        rows.append(row)
    return rows, checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", default="results/dai7_probe_v2/fits")
    ap.add_argument("--out", default="results/dai7_explore_ceiling")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    rows, checks = [], []
    files = sorted(glob.glob(os.path.join(a.fits, "*.npz")))
    for p in files:
        r, ch = unit_rows(p)
        rows += r
        checks += ch
    U = pd.DataFrame(rows)
    print(f"単位 {len(files)}，行 {len(U)}，判定の作り直しの不一致 {sum(checks)}（{len(checks)} 個の比べ）", flush=True)
    U.to_csv(os.path.join(a.out, "units.csv"), index=False)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "DAI/exp"], capture_output=True, text=True).stdout.strip())
    json.dump(dict(commit=commit, dirty_DAI_exp=dirty, fits=a.fits, n_units=len(files),
                   n_pred_mismatch=int(sum(checks)), seconds=round(time.time() - t0, 1)),
              open(os.path.join(a.out, "provenance.json"), "w"), indent=1)
    print(f"書いた：{a.out}/units.csv，provenance.json（{time.time() - t0:.0f} 秒）")


if __name__ == "__main__":
    main()
