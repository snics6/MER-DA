"""探索の解析（事前登録の外．2026-11-05）：提案と温度つき蒸留（KD）の得が，学習の目標にした予測（積のモデル）が生徒自身より
当たっているところで出るのかを，評価の回の保存した出力だけで測る（新しい学習はしない）．

研究者の求め「なぜ提案手法と KD が良いのかを探ろう」「AVT に効かない理由を調べる」（2026-11-05）．
単位（T，分割 f，種 r）ごとに，K ∈ {0, 1, 5, 10, 25}，生徒 s ∈ {A, V, T, AVT} について次を数える．

- teacher_uar_pool：提案の主の版（固定のファイルの main[K]）の目標の予測（学び直した A・V・T の識別器の確率の積．one は指数 (1,1,1)，
  sel は固定のファイルの w[K]．dai1.product と同じ式）の，候補（K 本を除く）での UAR．段 2 の保存 p2_K<K>_ret_<m>_pool（float16）から
  作り直し，段 3 の行 `_teacher_stats` の uar_pool と照合する．族 3 の相手の版の目標の予測の UAR も teacher_uar_partner に書く．
  pool の量（teacher_uar_pool，none_uar_pool，agree，fix，harm）は float16 の保存から作るので，約 1% の発話の判定が同値の丸めで
  変わりうる（gate-code の探索の検査．セルの平均で最大 0.05 ポイント程度）．
- none_uar_pool：生徒 s の none[K]（K 本だけ）の候補での UAR（p2_K<K>_none_<s>_pool）．
- adv_pool = teacher_uar_pool − none_uar_pool（目標の予測の先回り）．
- agree：候補で目標の予測と none[K] の判定が同じ割合．fix：none[K] が誤り目標が正しい割合．harm：none[K] が正しく目標が誤る割合．
- none_uar_test：none[K] のテストの UAR．段 1 の保存の int8 の判定（K = 0 は出発点 p1_<s>_start_pred，K ≥ 1 は固定の引き戻しの
  p1_<s>_pred．dai1_stats.UnitPreds.none と同じ）から作る（gate-code の探索の検査の指摘で float16 の p2 から替えた）．
- prop_uar_test：提案（joint．固定のファイルの主の版・選んだ設定）のテストの UAR（段 3 の行）．kd_uar_test：族 3 の相手（固定のファイル）の UAR．
- gain_prop = prop_uar_test − none_uar_test，gain_kd = kd_uar_test − none_uar_test．

UAR は 4 感情の再現率の平均（30 本の規則は当てない．探索の記述）．ラベルは dai1_stats.label_maps（manifest の正本）から引く．
出力：<out>/teacher_adv_units.csv（単位ごと）と teacher_adv_cells.csv（(T, 生徒, K) の単位の平均）．
使い方：python DAI/exp/explore_teacher_adv.py <out> <T> [<T> ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dai1_stats as ST  # noqa: E402

ROOT = HERE.parents[1]
KS = (0, 1, 5, 10, 25)
STUDENTS = ("A", "V", "T", "AVT")


def uar(y, p):
    return 100.0 * np.mean([np.mean(p[y == k] == k) for k in range(4) if (y == k).any()])


def product_w(z, K, w):
    L = sum(float(wi) * np.log(np.clip(z[f"p2_K{K}_ret_{m}_pool"].astype(np.float64), 1e-12, None)) for wi, m in zip(w, "AVT"))
    L -= L.max(1, keepdims=True)
    P = np.exp(L)
    return P / P.sum(1, keepdims=True)


def pick(d, params):
    for k, v in params.items():
        if k == "pseudo_uar" or k not in d:
            continue
        d = d[np.isclose(pd.to_numeric(d[k], errors="coerce"), float(v))]
    return d


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    targets = sys.argv[2:]
    fz = json.load(open(ROOT / "DAI" / "frozen" / "dai1_frozen.json"))
    U = ROOT / "results" / "dai1" / "eval" / "units"
    rows = []
    for T in targets:
        y_of, _ = ST.label_maps(T)
        F = fz[T]
        for f in range(5):
            for r in range(int(F["n_seeds_eval"])):
                base = U / f"{T}_ALL_f{f}_r{r}"
                if not Path(f"{base}_p3.csv").exists():                  # 評価の回の途中（MELD・MOSEI）は終わった単位だけを読む．使った単位の数を出す
                    continue
                z1, z2 = np.load(f"{base}_p1.npz"), np.load(f"{base}_p2.npz")
                p3 = pd.read_csv(f"{base}_p3.csv", low_memory=False)
                ypool, ytest = y_of(z1["sample_pool"].astype(str)), y_of(z1["sample_test"].astype(str))
                pool_ids = z1["sample_pool"].astype(str)
                for K in KS:
                    nk = np.ones(len(pool_ids), bool)
                    if K:
                        nk &= ~np.isin(pool_ids, z1[f"sample_K{K}"].astype(str))
                    ver = F["main"][str(K)]
                    wv = {"one": (1, 1, 1), "sel": tuple(F["w"][str(K)])}
                    tqs = {v: product_w(z2, K, w).argmax(1) for v, w in wv.items()}
                    tq = tqs[ver]
                    t_uar = uar(ypool[nk], tq[nk])
                    ts = p3[(p3.method == "_teacher_stats") & (p3.K == K) & (p3.teacher == ver)]
                    t_saved = float(ts.uar_pool.iloc[0]) if len(ts) else np.nan
                    for s in STUDENTS:
                        sp = z2[f"p2_K{K}_none_{s}_pool"].astype(np.float32).argmax(1)
                        if K == 0:
                            st = z1[f"p1_{s}_start_pred"]
                        else:
                            cw, cb = F["tether"][f"{s}|{K}"]
                            items = [str(x) for x in z1[f"p1_{s}_items"]]
                            st = z1[f"p1_{s}_pred"][items.index(f"{K}|{cw}|{cb}")]
                        n_uar = uar(ypool[nk], sp[nk])
                        right_t, right_s = tq[nk] == ypool[nk], sp[nk] == ypool[nk]
                        d = pick(p3[(p3.method == "joint") & (p3.student == s) & (p3.K == K) & (p3.teacher == ver)],
                                 F["select"][f"{s}|{K}|joint|{ver}"])
                        part = F["partner"][f"{s}|{K}"]
                        dk = p3[(p3.method == part["method"]) & (p3.student == s) & (p3.K == K)]
                        if part["method"] not in ("coreg", "cotrain"):
                            dk = dk[dk.teacher.astype(str) == str(part["version"])]
                        dk = pick(dk, F["select"].get(f"{s}|{K}|{part['method']}|{part['version']}", {}))
                        pv = str(part["version"])
                        tp = tqs.get(pv, tq)
                        rows.append(dict(T=T, fold=f, seed=r, K=K, student=s, prop_version=ver, partner_version=pv,
                                         teacher_uar_pool=t_uar, teacher_uar_saved=t_saved,
                                         teacher_uar_partner=uar(ypool[nk], tp[nk]), unit_commit=str(z1["commit"]),
                                         none_uar_pool=n_uar, adv_pool=t_uar - n_uar, agree=float(np.mean(tq[nk] == sp[nk])),
                                         fix=float(np.mean(right_t & ~right_s)), harm=float(np.mean(~right_t & right_s)),
                                         none_uar_test=uar(ytest, st), prop_uar_test=float(d.uar.mean()) if len(d) else np.nan,
                                         n_prop_rows=len(d), partner=f"{part['method']}|{part['version']}",
                                         kd_uar_test=float(dk.uar.mean()) if len(dk) else np.nan, n_kd_rows=len(dk)))
    D = pd.DataFrame(rows)
    D["gain_prop"] = D.prop_uar_test - D.none_uar_test
    D["gain_kd"] = D.kd_uar_test - D.none_uar_test
    D.to_csv(out / "teacher_adv_units.csv", index=False)
    C = D.groupby(["T", "student", "K"]).agg(n_units=("seed", "size"), teacher_uar_pool=("teacher_uar_pool", "mean"),
                                              none_uar_pool=("none_uar_pool", "mean"), adv_pool=("adv_pool", "mean"),
                                              agree=("agree", "mean"), fix=("fix", "mean"), harm=("harm", "mean"), teacher_uar_partner=("teacher_uar_partner", "mean"),
                                              prop_version=("prop_version", "first"), partner_version=("partner_version", "first"),
                                              none_uar_test=("none_uar_test", "mean"), gain_prop=("gain_prop", "mean"),
                                              gain_kd=("gain_kd", "mean")).reset_index()
    C.to_csv(out / "teacher_adv_cells.csv", index=False)
    chk = (D.teacher_uar_pool - D.teacher_uar_saved).abs()
    assert D.unit_commit.nunique() == 1, D.unit_commit.unique()
    print(f"単位 {D.groupby(['T','fold','seed']).size().groupby(level=0).size().to_dict()}，入力のコミット {D.unit_commit.iloc[0][:10]}，行 {len(D)}，目標の予測の UAR の照合の最大差 {chk.max():.2e}，"
          f"提案の行が 1 つでない {int((D.n_prop_rows != 1).sum())}，相手の行が 1 つでない {int((D.n_kd_rows != 1).sum())}")


if __name__ == "__main__":
    main()
