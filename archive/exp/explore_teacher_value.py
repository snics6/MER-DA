"""E65 の下調べ：別のモダリティの先生は「ターゲットのラベル何本ぶん」か（ラベル換算の価値）を，仮のターゲットのデータだけで測る．

問い（2026-10-28．手法を「ラベル換算の価値で資源の重みを決める」形にできるかの下調べ）
  1. 先生が足す本数 n_mod（K'(先生あり) − K'(先生なし)．K' は UAR をターゲットのみの学習曲線の上で読んだ本数）は，K によらずほぼ一定か
  2. 本数が一定なら，ラベルの項は K に比例して重くなるので，弱いラベルの比重 m は K によらない 1 つの値でよい．
     仮のターゲットを 1 つ抜いて残りで選ぶと，「m と先生を K ごとに選ぶ」今のやり方と，「K によらず 1 つ選ぶ」やり方はどちらが当たるか

使うもの（評価に使うターゲットのテストの結果は使わない）
  - results/multimodal_distill_<世界>/<T>_pseudo.csv：E60 の仮のターゲットの UAR（列 pseudo, rep, student, belief, meff, K, uar）．
    仮のターゲット c はターゲット T のコーパスの外のソースのコーパスで，c の手法は T と c のコーパスの外の群から作る
  - results/map_<種類>/<c>.csv の tgt_only：c をターゲットにしたターゲットのみの UAR（K = 1, 5, 10, 25, 50, 全件．反復の平均）．
    E32（exp/explore_source_value.py）と同じく，K について単調に整え（累積の最大），log K について線形に補う．全件の点は n_tgt/4 本に置く
  - 注意：E60 の生徒は世界のモダリティがそろった発話だけを使うが，学習曲線はその種類の全部の発話で作る（音声の生徒では発話の集合が少し違う）

    ../.venv/bin/python exp/explore_teacher_value.py      （MER-DA の直下で．数十秒）
出力：results/explore_relation/e65_teacher_value.csv（行 = 世界 × T × c × 反復 × 生徒 × 先生 × m × K），
      results/explore_relation/e65_summary.csv，e65_selection.csv
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "explore_relation"
WORLDS = ["fuse-audio-video", "fuse-video-text", "fuse-audio-text", "fuse-audio-video-text"]
SINGLE = {"audio_wavlm", "video", "text_roberta"}
CAND_SINGLE = {"pF", "poe", "agree"}
CAND_FUSED = {"poe0", "ref100", "ref500", "refdirect100", "refdirect500"}
KS = ["0", "1", "5", "10", "25"]


def curve(kind: str, corpus: str) -> tuple[np.ndarray, np.ndarray]:
    """ターゲットのみの学習曲線（本数 k と UAR）．K について単調に整える．"""
    d = pd.read_csv(RES / f"map_{kind}" / f"{corpus}.csv", comment="#", dtype={"K": str, "source": str},
                    usecols=["method", "K", "rep", "n_tgt", "uar"])
    t = d[d.method == "tgt_only"]
    ks, us = [], []
    for K in ["1", "5", "10", "25", "50"]:
        ks.append(float(K))
        us.append(float(t[t.K == K].uar.mean()))
    a = t[t.K == "all"]
    ks.append(float(a.n_tgt.mean()) / 4)
    us.append(float(a.uar.mean()))
    return np.array(ks), np.maximum.accumulate(np.array(us))


def k_equiv(ks: np.ndarray, us: np.ndarray, u: float) -> tuple[float, int]:
    """UAR u が曲線に当たる本数．上を超えたら (nan, 1)，下を割ったら (nan, -1)．"""
    if not np.isfinite(u):
        return np.nan, 0
    if u > us[-1]:
        return np.nan, 1
    if u < us[0]:
        return np.nan, -1
    lk = np.log(ks)
    for i in range(len(us) - 1):
        if us[i] <= u <= us[i + 1]:
            f = 0.0 if us[i + 1] == us[i] else (u - us[i]) / (us[i + 1] - us[i])
            return float(np.exp(lk[i] + f * (lk[i + 1] - lk[i]))), 0
    return np.nan, 1


def load() -> pd.DataFrame:
    curves: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    rows = []
    for w in WORLDS:
        for f in sorted((RES / f"multimodal_distill_{w}").glob("*_pseudo.csv")):
            T = f.name[: -len("_pseudo.csv")]
            d = pd.read_csv(f, comment="#", dtype={"K": str, "meff": str})
            for (c, r, s, K), g in d.groupby(["pseudo", "rep", "student", "K"]):
                key = (s, c)
                if key not in curves:
                    curves[key] = curve(s, c)
                ks, us = curves[key]
                none = g[g.belief == "none"].uar
                if len(none) != 1:
                    continue
                u0 = float(none.iloc[0])
                k0, f0 = k_equiv(ks, us, u0)
                cand = CAND_SINGLE if s in SINGLE else CAND_FUSED
                for _, x in g[g.belief.isin(cand)].iterrows():
                    k1, f1 = k_equiv(ks, us, float(x.uar))
                    rows.append(dict(world=w, target=T, pseudo=c, rep=int(r), student=s, single=s in SINGLE,
                                     belief=x.belief, meff=x.meff, K=K, uar_none=u0, uar=float(x.uar),
                                     k_none=k0, flag_none=f0, k_with=k1, flag_with=f1,
                                     n_start=(k0 - float(K)) if f0 == 0 else np.nan,
                                     n_mod=(k1 - k0) if (f0 == 0 and f1 == 0) else np.nan))
    return pd.DataFrame(rows)


def selection(D: pd.DataFrame) -> pd.DataFrame:
    """仮のターゲットを 1 つ抜いて残りで選び，抜いた仮のターゲットの UAR で当たりを比べる．
    A：（先生, m）を K ごとに選ぶ（E60 と同じ規則）．B：（先生, m）を K によらず 1 つ選ぶ（K で平均した UAR が最大）．
    どちらも「弱いラベルを使わない」（none）を候補に含む．"""
    out = []
    for (w, T, s), g in D.groupby(["world", "target", "student"]):
        g = g.copy()
        g["cand"] = g.belief + "|" + g.meff
        base = g.drop_duplicates(["pseudo", "rep", "K"])[["pseudo", "rep", "K", "uar_none"]]
        none = base.assign(cand="none", uar=base.uar_none)[["pseudo", "rep", "K", "cand", "uar"]]
        h = pd.concat([g[["pseudo", "rep", "K", "cand", "uar"]], none], ignore_index=True)
        cs = sorted(h.pseudo.unique())
        if len(cs) < 2:
            continue
        for c in cs:
            tr, te = h[h.pseudo != c], h[h.pseudo == c]
            m_tr = tr.groupby(["K", "cand"]).uar.mean()
            pick_a = {K: m_tr.loc[K].idxmax() for K in KS if K in m_tr.index.get_level_values(0)}
            pick_b = tr.groupby("cand").uar.mean().idxmax()
            te_m = te.groupby(["K", "cand"]).uar.mean()
            for K in KS:
                if K not in pick_a:
                    continue
                ua = te_m.get((K, pick_a[K]), np.nan)
                ub = te_m.get((K, pick_b), np.nan)
                un = te_m.get((K, "none"), np.nan)
                out.append(dict(world=w, target=T, student=s, single=s in SINGLE, held_out=c, K=K,
                                pick_a=pick_a[K], pick_b=pick_b, uar_a=ua, uar_b=ub, uar_none=un))
    return pd.DataFrame(out)


def main() -> None:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    meta = json.dumps({"script": "exp/explore_teacher_value.py", "commit": commit, "worlds": WORLDS})
    D = load()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "e65_teacher_value.csv").write_text(f"# {meta}\n" + D.to_csv(index=False))
    S = D.groupby(["single", "student", "belief", "meff", "K"]).agg(
        n=("n_mod", "size"), n_mod_med=("n_mod", "median"), n_mod_q25=("n_mod", lambda x: x.quantile(0.25)),
        n_mod_q75=("n_mod", lambda x: x.quantile(0.75)), cens_hi=("flag_with", lambda x: float((x == 1).mean())),
        n_start_med=("n_start", "median"), gain_med=("uar", "median")).reset_index()
    (OUT / "e65_summary.csv").write_text(f"# {meta}\n" + S.to_csv(index=False))
    L = selection(D)
    (OUT / "e65_selection.csv").write_text(f"# {meta}\n" + L.to_csv(index=False))
    print(f"e65: 行 {len(D)}，仮のターゲットを抜く確かめ {len(L)} 行（commit {commit}）")


if __name__ == "__main__":
    main()
