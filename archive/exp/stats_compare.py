"""【探索的】E64（CPU）：手法の比較（E61・E62・E63）の検定と効果量（論文の実験要求 R2．`paper/requests.md`）。

規則は `docs/exploratory-analyses.md` E64．学び直しはしない．保存済みの反復ごとの結果から，対応のある差を作って検定するだけ．

入力（反復ごとの生の結果．保存済みの判定表 `results/explore_relation/e61_*.csv` などは使わない）
  results/baselines_<世界>/<T>.csv，<T>_pseudo.csv，<T>_meta.csv      E61・E62 の相手と oursx（反復 0〜9．仮のターゲットは反復 0〜4）
  results/multimodal_distill_<世界>/<T>.csv，<T>_pseudo.csv，<T>_meta.csv  E60 の選んだ形 ours
  results/finetune_check/<T>.csv，<T>_c.csv，teacher_<T>.csv            E63・E63c（4 ターゲット，反復 0〜2）
  data/cache/ の特徴（話者の列だけ使う）                                反復ごとのテストの話者の集合（`exp/map.py` の test_mask）

選択の作り直し（`paper/scripts/ledger/comparison.py` と同じ規則．書き方は別に書いて照らし合わせる）
  調整版：相手 m のつまみを，K ごとに仮のターゲットの UAR の平均（ターゲットのソース × 反復 0〜4）が最大の候補にする（none も候補．
          `exp/baselines_distill.py` の choose()）．ours：E60 の (見込み, m_eff) を同じく選ぶ（`exp/multimodal_distill.py` の choose()）
  最も強い相手：oursx 以外の相手の調整版のうち，ターゲットの平均（反復 0〜9 の平均をターゲットごとに取って平均）が最大のもの
          （世界 × 生徒 × K ごと．同点は名前の順で最初）

比べる量（ターゲット × 反復の対の差．UAR のポイント）
  (a) oursx − 最も強い相手の調整版，(b) oursx − E39，(c) ours − E39，(d) 最も強い相手の調整版 − E39
  E63・E63c：微調整版の私たちの形（λ=0.01）− 凍結版の私たちの形，K 本だけの微調整（K=0 は出発点のまま）− 凍結の E39，
             弱いラベルの経路の上乗せ（K=0 は ours − start，K=5 は ours − anchor（λ=0.01））

統計（対ごと）：n，平均，標準偏差（ddof=1），d_z = 平均 ÷ 標準偏差，正の対の割合，ターゲットごとの平均が正の数，
  ウィルコクソンの符号順位検定（両側．差が 0 の対は落とす．scipy の method="auto"），Holm の補正（族は下の FAMILIES）．
  補助：同じテストの話者の集合を持つ反復を平均した単位での検定（p_dedup），ターゲットの平均の符号での符号検定（p_sign_tgt）

    ./.venv/bin/python exp/stats_compare.py

出力：results/stats_compare/ に tests.csv（検定 1 行），pairs.csv（対の差），table.csv（ターゲットの平均の UAR），best.csv（最も強い相手），
  testsets.csv（反復ごとのテストの話者の集合），limits.csv（限界），provenance.json
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")                     # GPU は使わない（map の読み込みで torch が入るため）

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import map as M  # noqa: E402
from data import load_any  # noqa: E402
from pair_table import holm  # noqa: E402

RES = ROOT / "results"
OUT = RES / "stats_compare"

WORLDS = {"avt": "fuse-audio-video-text", "av": "fuse-audio-video", "vt": "fuse-video-text", "at": "fuse-audio-text",
          "w2": "fuse2-audio-video-text"}
EXPERIMENT = {"avt": "E61", "av": "E61", "vt": "E61", "at": "E61", "w2": "E62"}
MODS = {"audio_wavlm": "a", "video": "v", "text_roberta": "t", "audio_hubert": "a", "video_clip": "v", "text_deberta": "t"}
KS = ("0", "1", "5", "10", "25")
REPS = tuple(range(10))
PSEUDO_REPS = tuple(range(5))
E60_CANDS = {"mod": {"pF", "poe", "agree"}, "fused": {"poe0", "ref100", "ref500", "refdirect100", "refdirect500"}}
STR = {"K": str, "meff": str, "param": str}

FT_DIR = RES / "finetune_check"
FT_TARGETS = ("IEMOCAP-impro", "IEMOCAP-script", "MELD", "MOSEI")
FT_REPS = (0, 1, 2)
FT_LAM = 0.01
FT_WORLD = "fuse-audio-video-text"                                      # exp/finetune_check.py の WORLD（分割はこの世界の話者の列）
FT_COLS = ["target", "rep", "K", "method", "lam", "uar", "n_test", "eval_sec", "gpu_gb", "n_train", "steps", "train_sec",
           "loss_first", "loss_last"]                                   # exp/finetune_check.py の RUN_COLS（MOSEI.csv は見出しが 9 列）

CMP = {"a": ("oursx-best", "oursx の調整版 − 最も強い相手の調整版"),
       "b": ("oursx-e39", "oursx の調整版 − E39（none）"),
       "c": ("ours-e39", "ours（E60 の選んだ形）− E39（none）"),
       "d": ("best-e39", "最も強い相手の調整版 − E39（none）")}
FT_CMP = {"ours-vs-frozen": "微調整版の私たちの形（λ=0.01）− 凍結版の私たちの形（frozen_ours）",
          "konly-vs-e39": "K 本だけの微調整（K=0 は出発点のまま start，K=5 は k_only）− 凍結の E39（frozen_e39）",
          "weak-gain": "弱いラベルの経路の上乗せ（K=0 は ours − start，K=5 は ours − anchor．λ=0.01）"}
FAMILIES = {
    "F-a": "比較 (a) の全セル：世界 {AVT, AV, VT, AT, w2} × 生徒（AVT・w2 は音声・映像・言葉・融合，AV・VT・AT は融合）× K ∈ {0,1,5,10,25} の 55 検定",
    "F-b": "比較 (b) の全セル（55 検定．セルは F-a と同じ）",
    "F-c": "比較 (c) の全セル（55 検定．セルは F-a と同じ）",
    "F-d": "比較 (d) の全セル（55 検定．セルは F-a と同じ）",
    "F-ft": "E63（中心化なしの微調整）：3 つの比較 × K ∈ {0,5} の 6 検定",
    "F-ftc": "E63c（中心化ありの微調整）：3 つの比較 × K ∈ {0,5} の 6 検定",
}


# ---------------------------------------------------------------- 読み込み

def read(path, **kw):
    return pd.read_csv(path, comment="#", keep_default_na=False, **kw)


def per_rep(df, key, reps, what):
    """key ＋ rep → UAR（％）．重複と反復の欠けがあれば止める。"""
    df = df[df.rep.isin(reps)]
    if df.duplicated(key + ["rep"]).any():
        raise ValueError(f"{what}：同じ条件・同じ反復の行が重複している")
    cnt = df.groupby(key).size()
    if (cnt != len(reps)).any():
        bad = cnt[cnt != len(reps)]
        raise ValueError(f"{what}：反復が {len(reps)} 回そろっていない（例 {bad.index[0]}）")
    return {tuple(k): v * 100 for k, v in zip(df[key + ["rep"]].itertuples(index=False, name=None), df.uar)}


def first_max(cands, score):
    """並べた順で最初の最大（pandas の idxmax と同じ）。"""
    best = cands[0]
    for c in cands[1:]:
        if score[c] > score[best]:
            best = c
    return best


def world_rows(wk):
    """世界 wk の (target, rep, student, K) ごとの値．列 e39，ours，<m>（相手と oursx の調整版）．戻りは (行, 選択, 照合)。"""
    w = WORLDS[wk]
    d, md = RES / f"baselines_{w}", RES / f"multimodal_distill_{w}"
    targets = sorted(f.stem for f in d.glob("*.csv") if not f.stem.endswith(("_pseudo", "_meta")))
    rows, picks = [], []
    chk = {"picks": 0, "meta_mismatch": 0, "e60_picks": 0, "e60_mismatch": 0}
    key = ["student", "method", "param", "meff", "K"]
    ekey = ["student", "belief", "meff", "K"]
    for T in targets:
        a = read(d / f"{T}.csv", dtype=STR)
        val = per_rep(a, key, REPS, f"baselines_{w}/{T}.csv")
        p = read(d / f"{T}_pseudo.csv", dtype=STR)
        ps = p[p.rep.isin(PSEUDO_REPS)].groupby(key).uar.mean().to_dict()
        meta = read(d / f"{T}_meta.csv", dtype=STR)
        meta = meta[meta.chosen.astype(str) == "True"]
        meta_ch = {(r.student, r.method, r.K): (r.cand, r.param, r.meff) for r in meta.itertuples()}
        e = read(md / f"{T}.csv", dtype={"K": str, "meff": str})
        evals = per_rep(e, ekey, REPS, f"multimodal_distill_{w}/{T}.csv")
        ep = read(md / f"{T}_pseudo.csv", dtype={"K": str, "meff": str})
        eps = ep[ep.rep.isin(PSEUDO_REPS)].groupby(ekey).uar.mean().to_dict()
        em = read(md / f"{T}_meta.csv", dtype={"K": str, "meff": str})
        em = em[em.chosen.astype(str) == "True"]
        em_ch = {(r.student, r.K): (r.belief, r.meff) for r in em.itertuples()}

        students = sorted(set(a.student[a.method == "none"]))          # --fused-only の世界ではモダリティの生徒に E39 の行が無い
        for s in students:
            fused = s == w
            methods = sorted(set(a.method[a.student == s]) - {"none"})
            for K in KS:
                pick = {}
                for m in methods:
                    cands = sorted(k for k in ps if k[0] == s and k[4] == K and k[1] in (m, "none"))
                    b = first_max(cands, ps)
                    pick[m] = (b[1], b[2], b[3])
                    chk["picks"] += 1
                    chk["meta_mismatch"] += int(meta_ch.get((s, m, K)) != pick[m])
                cb = E60_CANDS["fused" if fused else "mod"]
                ec = sorted(k for k in eps if k[0] == s and k[3] == K and (k[1] == "none" or k[1] in cb))
                b = first_max(ec, eps)
                epick = (b[1], b[2])
                chk["e60_picks"] += 1
                chk["e60_mismatch"] += int(em_ch.get((s, K)) != epick)
                picks.append({"world": wk, "target": T, "student": s, "K": K, "ours": "|".join(epick),
                              **{m: "|".join(pick[m]) for m in methods}})
                for r in REPS:
                    row = {"world": wk, "target": T, "rep": r, "student": s, "K": K,
                           "e39": val[(s, "none", "-", "0", K, r)], "ours": evals[(s, *epick, K, r)]}
                    for m in methods:
                        row[m] = val[(s, *pick[m], K, r)]
                    rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(picks), chk


def tok(wk, s):
    return "f" if s == WORLDS[wk] else MODS[s]


# ---------------------------------------------------------------- テストの話者の集合

def test_sets(wk_kind: str, targets, reps):
    """(target, rep) → テストの話者の集合の番号（ターゲットの中で最初に現れた反復の順）．`exp/map.py` の test_mask を通す。"""
    out = []
    for T in targets:
        df, _ = load_any(T, wk_kind)
        sp = df["speaker_group"].astype(str).reset_index(drop=True)
        seen = {}
        for r in reps:
            te = M.test_mask(SimpleNamespace(speaker=sp), r)
            s = frozenset(sp[te].unique())
            if s not in seen:
                seen[s] = len(seen)
            out.append({"kind": wk_kind, "target": T, "rep": r, "testset": seen[s], "n_test_speakers": len(s), "n_test_rows": int(te.sum())})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 検定

def wilcoxon(d):
    """両側のウィルコクソンの符号順位検定．戻りは (W, p, 方法の名前)．差が 0 の対は落とす（zero_method="wilcox"）。"""
    d = np.round(np.asarray(d, float), 10)                               # 同じ UAR から作った差の浮動小数の揺れを同点にそろえる
    nz = d[d != 0]
    if len(nz) == 0:
        return np.nan, 1.0, "none（差がすべて 0）"
    ranks = stats.rankdata(np.abs(nz))
    ties = len(np.unique(np.abs(nz))) < len(nz)
    zeros = len(nz) < len(d)
    if len(d) > 50:
        how = "asymptotic（正規近似．同点の補正と連続補正）"
    elif not (ties or zeros):
        how = "exact"
    elif len(d) <= 13:
        how = "permutation（符号の全通り．決定的）"
    else:
        how = "asymptotic（正規近似．同点の補正と連続補正）"
    res = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided", method="auto")
    rp = ranks[nz > 0].sum()
    return float(min(rp, ranks.sum() - rp)), float(res.pvalue), how


def sign_p(npos, nneg):
    n = npos + nneg
    if n == 0:
        return 1.0
    return float(stats.binomtest(npos, n, 0.5).pvalue)


def summarize(d: pd.DataFrame):
    """d：列 target，rep，testset，diff．戻りは統計の dict。"""
    x = d["diff"].to_numpy(float)
    n = len(x)
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    W, p, how = wilcoxon(x)
    tg = d.groupby("target")["diff"].mean()
    un = d.groupby(["target", "testset"])["diff"].mean()
    _, pu, howu = wilcoxon(un.to_numpy())
    xr = np.round(x, 10)
    return {"n": n, "n_targets": int(tg.size), "mean": mean, "sd": sd,
            "dz": mean / sd if sd > 0 else (0.0 if mean == 0 else np.nan),
            "n_pos": int((xr > 0).sum()), "n_zero": int((xr == 0).sum()), "n_neg": int((xr < 0).sum()),
            "pos_pct": 100 * float((xr > 0).mean()),
            "tgt_pos": int((tg.round(10) > 0).sum()), "tgt_neg": int((tg.round(10) < 0).sum()),
            "p_sign_tgt": sign_p(int((tg.round(10) > 0).sum()), int((tg.round(10) < 0).sum())),
            "W": W, "p": p, "p_method": how,
            "n_units_dedup": int(un.size), "mean_dedup": float(un.mean()), "p_dedup": pu, "p_dedup_method": howu,
            "tgt_means": "；".join(f"{t} {v:+.2f}" for t, v in tg.items())}


# ---------------------------------------------------------------- 本体

def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def header(extra=None):
    h = {"script": "exp/stats_compare.py", "commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
         "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if extra:
        h.update(extra)
    return "# " + json.dumps(h, ensure_ascii=False) + "\n"


def write(df, name, extra=None):
    for c in df.columns:
        if df[c].dtype == object and df[c].astype(str).str.contains("#").any():
            raise ValueError(f"{name} の列 {c} に # がある（comment='#' で読めなくなる）")
    with open(OUT / name, "w", encoding="utf-8") as fh:
        fh.write(header(extra))
        df.to_csv(fh, index=False)


def inputs():
    files = []
    for w in WORLDS.values():
        files += sorted((RES / f"baselines_{w}").glob("*.csv"))
        files += sorted((RES / f"multimodal_distill_{w}").glob("*.csv"))
    for T in FT_TARGETS:
        files += [FT_DIR / f"{T}.csv", FT_DIR / f"{T}_c.csv", FT_DIR / f"teacher_{T}.csv"]
    return files


def ft_runs(path):
    a = pd.read_csv(path, comment="#", header=None, skiprows=1, names=FT_COLS, dtype=str, keep_default_na=False)
    a = a[a.target != "target"]
    a = a.assign(rep=a.rep.astype(int), lam=a.lam.astype(float), uar=a.uar.astype(float) * 100)
    return a[["target", "rep", "K", "method", "lam", "uar"]]


def ft_ser(df, m, K, lam=None):
    x = df[(df.method == m) & (df.K == K)]
    if lam is not None:
        x = x[np.isclose(x.lam, lam)]
    x = x.set_index(["target", "rep"]).uar.sort_index()
    if x.index.duplicated().any() or len(x) != len(FT_TARGETS) * len(FT_REPS):
        raise ValueError(f"E63 の {m}（K={K}，λ={lam}）が {len(x)} 行（12 行のはず）")
    return x


def main() -> int:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- 世界ごとの値
    R, P, chks = {}, {}, {}
    for wk in WORLDS:
        R[wk], P[wk], chks[wk] = world_rows(wk)
        print(f"{wk}: {len(R[wk])} 行，選択の照合 {chks[wk]}（{time.perf_counter() - t0:.0f} 秒）", flush=True)

    # ---- テストの話者の集合
    ts = pd.concat([test_sets(WORLDS[wk], sorted(R[wk].target.unique()), REPS).assign(world=wk) for wk in WORLDS]
                   + [test_sets(FT_WORLD, FT_TARGETS, FT_REPS).assign(world="ft")], ignore_index=True)
    print(f"テストの話者の集合：{time.perf_counter() - t0:.0f} 秒", flush=True)

    # ---- ターゲットの平均の表と最も強い相手
    table, best_rows = [], []
    best = {}
    for wk in WORLDS:
        df = R[wk]
        cols = [c for c in df.columns if c not in ("world", "target", "rep", "student", "K")]
        tm = df.groupby(["student", "K", "target"])[cols].mean().groupby(["student", "K"]).mean()   # 反復の平均をターゲットごとに取り，ターゲットで平均
        for (s, K), r in tm.iterrows():
            for c in cols:
                if pd.notna(r[c]):
                    table.append({"world": wk, "world_name": WORLDS[wk], "student": s, "student_key": tok(wk, s), "K": K,
                                  "method": c, "uar": r[c], "n_targets": df.target.nunique()})
            comp = {c: r[c] for c in cols if c not in ("e39", "ours", "oursx") and pd.notna(r[c])}
            b = max(sorted(comp), key=lambda m: comp[m])
            ties = [m for m in sorted(comp) if m != b and abs(comp[m] - comp[b]) < 1e-9]
            order = sorted(comp, key=lambda m: -comp[m])
            best[(wk, s, K)] = b
            best_rows.append({"world": wk, "student": s, "student_key": tok(wk, s), "K": K, "best": b, "best_uar": comp[b],
                              "best_minus_e39": comp[b] - r.e39, "second": order[1], "second_uar": comp[order[1]],
                              "ties": "・".join(ties), "opponents": "・".join(sorted(comp))})
    table = pd.DataFrame(table)
    best_df = pd.DataFrame(best_rows)

    # ---- 対の差と検定（E61・E62）
    tests, pairs = [], []
    for wk in WORLDS:
        df = R[wk].merge(ts[ts.world == wk][["target", "rep", "testset"]], on=["target", "rep"], how="left")
        if df.testset.isna().any():
            raise ValueError(f"{wk}：テストの話者の集合が無い行がある")
        students = sorted(df.student.unique(), key=lambda s: "atvf".index(tok(wk, s)))
        for s in students:
            for K in KS:
                g = df[(df.student == s) & (df.K == K)].sort_values(["target", "rep"])
                b = best[(wk, s, K)]
                for c, (x, y) in {"a": ("oursx", b), "b": ("oursx", "e39"), "c": ("ours", "e39"), "d": (b, "e39")}.items():
                    dd = g[["target", "rep", "testset"]].assign(x=g[x].to_numpy(), y=g[y].to_numpy())
                    dd["diff"] = dd.x - dd.y
                    st = summarize(dd)
                    tests.append({"experiment": EXPERIMENT[wk], "family": f"F-{c}", "comparison": c, "comparison_name": CMP[c][0],
                                  "what": CMP[c][1], "world": wk, "world_name": WORLDS[wk], "student": s, "student_key": tok(wk, s),
                                  "K": K, "x": x, "y": y, "best": b if c in "ad" else "", **st})
                    pairs.append(dd.assign(experiment=EXPERIMENT[wk], comparison=c, world=wk, student=s, K=K, x_name=x, y_name=y))

    # ---- E63・E63c
    ft = pd.concat([ft_runs(FT_DIR / f"{t}.csv") for t in FT_TARGETS])
    fc = pd.concat([ft_runs(FT_DIR / f"{t}_c.csv") for t in FT_TARGETS])
    fr = pd.concat([read(FT_DIR / f"teacher_{t}.csv", dtype={"K": str}) for t in FT_TARGETS])
    fr = fr.assign(uar=fr.uar * 100, lam=0.0)
    E = {K: ft_ser(fr, "frozen_e39", K) for K in ("0", "5")}
    O = {K: ft_ser(fr, "frozen_ours", K) for K in ("0", "5")}
    tsf = ts[ts.world == "ft"].set_index(["target", "rep"]).testset
    for tag, df, exp_name in (("ft", ft, "E63"), ("ftc", fc, "E63c")):
        start, konly = ft_ser(df, "start", "0"), ft_ser(df, "k_only", "5")
        anchor = ft_ser(df, "anchor", "5", FT_LAM)
        ours = {K: ft_ser(df, "ours", K, FT_LAM) for K in ("0", "5")}
        spec = {("ours-vs-frozen", "0"): (ours["0"], O["0"], "ours", "frozen_ours"),
                ("ours-vs-frozen", "5"): (ours["5"], O["5"], "ours", "frozen_ours"),
                ("konly-vs-e39", "0"): (start, E["0"], "start", "frozen_e39"),
                ("konly-vs-e39", "5"): (konly, E["5"], "k_only", "frozen_e39"),
                ("weak-gain", "0"): (ours["0"], start, "ours", "start"),
                ("weak-gain", "5"): (ours["5"], anchor, "ours", "anchor")}
        for (c, K), (x, y, xn, yn) in spec.items():
            dd = pd.DataFrame({"x": x, "y": y}).reset_index()
            dd["testset"] = [tsf[(t, r)] for t, r in zip(dd.target, dd.rep)]
            dd["diff"] = dd.x - dd.y
            st = summarize(dd)
            tests.append({"experiment": exp_name, "family": f"F-{tag}", "comparison": c, "comparison_name": c, "what": FT_CMP[c],
                          "world": tag, "world_name": FT_WORLD + "（音声の生徒 WavLM の LoRA 微調整）", "student": "audio_wavlm",
                          "student_key": "a", "K": K, "x": xn, "y": yn, "best": "", **st})
            pairs.append(dd.assign(experiment=exp_name, comparison=c, world=tag, student="audio_wavlm", K=K, x_name=xn, y_name=yn))

    tests = pd.DataFrame(tests)
    for fam, g in tests.groupby("family"):
        tests.loc[g.index, "family_size"] = len(g)
        tests.loc[g.index, "p_holm"] = holm(g.p.to_numpy(float))
        tests.loc[g.index, "p_dedup_holm"] = holm(g.p_dedup.to_numpy(float))
    tests["family_size"] = tests.family_size.astype(int)
    tests["family_def"] = tests.family.map(FAMILIES)
    lim = ("対は独立でない：同じターゲットの反復はテストの話者の袋を共有しうる（列 n_units_dedup が異なる袋の数）．"
           "K=0 では同じ袋の反復は同じ値になる．p は目安で，p_dedup（同じ袋の反復を平均した単位）と tgt_pos（ターゲットの数）を並べて読む")
    tests["limitation"] = lim
    tests = tests[["experiment", "family", "family_size", "comparison", "comparison_name", "what", "world", "world_name", "student",
                   "student_key", "K", "x", "y", "best", "n", "n_targets", "mean", "sd", "dz", "n_pos", "n_zero", "n_neg", "pos_pct",
                   "tgt_pos", "tgt_neg", "p_sign_tgt", "W", "p", "p_method", "p_holm", "n_units_dedup", "mean_dedup", "p_dedup",
                   "p_dedup_method", "p_dedup_holm", "tgt_means", "family_def", "limitation"]]
    pairs = pd.concat(pairs, ignore_index=True)[["experiment", "comparison", "world", "student", "K", "target", "rep", "testset",
                                                  "x_name", "y_name", "x", "y", "diff"]]

    # ---- 限界
    tsum = (ts.groupby(["world", "kind", "target"])
            .agg(reps=("rep", lambda r: f"{min(r)}〜{max(r)}"), n_reps=("rep", "size"), n_testsets=("testset", "nunique"),
                 groups=("testset", lambda s: "，".join("・".join(str(r) for r in ts.loc[s.index][s == v].rep) for v in sorted(s.unique()))))
            .reset_index())
    lim_rows = [{"item": "testsets", "world": r.world, "target": r.target, "value": r.n_testsets,
                 "text": f"反復 {r.reps}（{r.n_reps} 回）のテストの話者の集合は {r.n_testsets} 通り（同じ集合の反復：{r.groups}）"}
                for r in tsum.itertuples()]
    p0 = pairs[pairs.experiment.isin(["E61", "E62"]) & (pairs.K == "0")]
    v0 = pd.concat([p0[["world", "student", "target", "rep", "testset", "x_name", "x"]].set_axis(
                        ["world", "student", "target", "rep", "testset", "method", "uar"], axis=1),
                    p0[["world", "student", "target", "rep", "testset", "y_name", "y"]].set_axis(
                        ["world", "student", "target", "rep", "testset", "method", "uar"], axis=1)]).drop_duplicates(
        ["world", "student", "target", "rep", "method"])
    spread = v0.groupby(["world", "student", "target", "testset", "method"]).uar.agg(lambda s: s.max() - s.min())
    tw = ts[ts.world != "ft"].groupby(["world", "target"])
    n_dup = int((tw.size() - tw.testset.nunique()).sum())
    lim_rows += [
        {"item": "dependence", "world": "all", "target": "", "value": float(spread.max()),
         "text": "ターゲット × 反復の対は独立でない．同じターゲットの反復は同じコーパスの話者を使い，IEMOCAP（話者 10 人）では反復 r と r+5 のテストがおおむね同じ話者の集合になる"
                 f"（即興 5 通り，台本 6 通り，MELD 9 通り．全世界で {n_dup} の反復が先の反復と同じ集合）．"
                 f"K=0 では同じ集合の反復の値は事実上同じになる（ラベル付きの本数が 0．値の差は最大 {spread.max():.3f} ポイント，0 でないのは {int((spread > 0).sum())} 組）．"
                 "このため対の検定の p は小さく出うる．列 p_dedup は同じ集合の反復を平均した単位での検定"},
        {"item": "targets", "world": "all", "target": "", "value": np.nan,
         "text": "ターゲットは 4 つ（AV は 6 つ）しかない．ターゲットの平均の符号で符号検定をすると，4 つすべてが正でも両側 p = 0.125（6 つで 0.031）にしかならない（列 p_sign_tgt）"},
        {"item": "selection", "world": "all", "target": "", "value": np.nan,
         "text": "最も強い相手はテストのターゲットの平均で選んでいる（比較 (a)・(d)）．相手に有利な選び方なので (a) は控えめに出るが，(d) は大きく出うる"},
        {"item": "dedup", "world": "all", "target": "", "value": np.nan,
         "text": "p_dedup は単位の数が少ない（AVT・VT・AT・w2 で 30，AV で 50）が，同点が無ければ正確な分布を使う．"
                 "p は同点（K=0 の重複）があると正規近似になるので，p_dedup が p より小さいこともある．どちらも単位の独立を仮定している"},
        {"item": "families", "world": "all", "target": "", "value": np.nan,
         "text": "Holm の族は比較ごとに 1 つ（(a)〜(d) は各 55 検定，E63・E63c は各 6 検定）．比較をまたいでは補正していない"},
    ]
    limits = pd.DataFrame(lim_rows)

    # ---- 書き出し
    write(tests, "tests.csv", {"families": FAMILIES})
    write(pairs, "pairs.csv")
    write(table, "table.csv", {"unit": "UAR（％）．反復 0〜9 の平均をターゲットごとに取り，ターゲットで平均．列 method の e39 は none，ours は E60 の選んだ形，ほかは調整版"})
    write(best_df, "best.csv", {"rule": "oursx 以外の相手の調整版のうちターゲットの平均が最大（同点は名前の順で最初）"})
    write(ts, "testsets.csv", {"rule": "exp/map.py の test_mask（speaker_folds(k=5, seed=r) の r mod 5 番の袋）"})
    write(limits, "limits.csv")
    write(pd.concat(P, ignore_index=True), "picks.csv", {"rule": "調整版と ours の選んだ (method|param|meff) と (belief|meff)"})
    files = inputs()
    prov = {"script": "exp/stats_compare.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "dirty_files": git("status", "--porcelain").splitlines(),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": round(time.perf_counter() - t0, 1),
            "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "wilcoxon": "scipy.stats.wilcoxon(d, zero_method='wilcox', alternative='two-sided', method='auto')．差は 10 桁に丸めてから．"
                        "差がすべて 0 なら p=1．auto は n>50 で正規近似，n≤50 で同点も 0 も無ければ正確，あれば n≤13 で符号の全通り，ほかは正規近似",
            "holm": "exp/pair_table.py の holm（族の中で）", "families": FAMILIES,
            "dz": "差の平均 ÷ 差の標準偏差（ddof=1）．差がすべて 0 なら 0",
            "reps": list(REPS), "pseudo_reps": list(PSEUDO_REPS), "ks": list(KS), "ft_reps": list(FT_REPS), "ft_lam": FT_LAM,
            "selection_check": chks,
            "inputs": [{"path": str(f.relative_to(ROOT)), "bytes": f.stat().st_size,
                        "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(f.stat().st_mtime)),
                        "sha256": hashlib.sha256(f.read_bytes()).hexdigest()} for f in files],
            "inputs_other": "data/cache/ の特徴（measure/data.py の load_any．列 speaker_group だけ使う）",
            "outputs": ["tests.csv", "pairs.csv", "table.csv", "best.csv", "testsets.csv", "limits.csv", "picks.csv"]}
    (OUT / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    bad = {wk: c for wk, c in chks.items() if c["meta_mismatch"] or c["e60_mismatch"]}
    print(f"選択の食い違い：{bad or 'なし'}")
    show = tests[tests.K.isin(["0", "5"]) & (tests.comparison == "a")]
    print(show[["world", "student_key", "K", "best", "n", "mean", "dz", "tgt_pos", "n_targets", "p", "p_holm", "p_dedup"]].to_string(index=False))
    print(f"終わり：{time.perf_counter() - t0:.0f} 秒．{OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
