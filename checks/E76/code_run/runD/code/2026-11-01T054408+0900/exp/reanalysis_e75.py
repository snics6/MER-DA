"""【再解析】E75（DAI 論文のステップ 1）：TAC 版の模擬査読 1 回目の指摘のうち，筋 1〜3 について保存済みの結果から
学び直しをせずに CPU で出せるものをまとめて計算する。新しい学習はしない（GPU は使わない）。

設計は `DAI_paper/steps/step1_design.md`（関門 1 の後に `docs/exploratory-analyses.md` の E75 の節へ写す）。
項目（括弧は答える指摘．ML = `paper/reviews/round1/review-ml.md`，感 = `review-affect.md`）

  1  地図の区間を依存に合わせて作り直す：補正した resampled t と IEMOCAP のテストの組を単位にした t（ML Major-1，感 Minor-1）
  2  言葉を含む 4 種類で IEMOCAP 台本をターゲットから除いた集計（ML Minor-2）
  3  加法モデルの調整した説明率と，名前を並べ替えた基準（ML Minor-5）
  4  ターゲットのみの C の規則を変えたときの利得（ML Minor-6）
  5  合併の得を o の分と d の分に分ける順序（ML Major-3 (3)，感 Minor-5）
  6  4 つの切片で 6 対の o を同時に直せる量（ML Major-3 (1)(2)）
  7  式 (2) の閾値の損と測った閾値の損を K ごとに（ML Major-3 (6)）
  8  向きの損が負の升の数と扱い，縮小 LDA の縮小の強さ（ML Major-3 (5)，ML Major-6）
  9  中心化の得とクラスの割合の関係（ML Minor-15）
  10 E12 の α を選んだ合併（反復 3 回）の区間（ML Minor-7）
  11 q の当てはめの診断：残差，区間，別の関数形（ML Major-4，感 Minor-6）
  12 λ の予言と実測の順位相関を K ごと・ターゲットの中で（ML Major-4 (5)）
  13 ソースの向きの効率 e を，ほかのコーパスを仮のターゲットにして推す（ML Optional-1，感 Optional-2）

    ./.venv/bin/python exp/reanalysis_e75.py --out results/e75                     # 全部
    ./.venv/bin/python exp/reanalysis_e75.py --out <dir> --kinds audio_wavlm --items 1 5   # 小さく流す

入力の場所はすべて引数（既定は下の DEFAULTS）で受ける。環境変数 MERDA_OUT_ROOT・MERDA_MAP_ROOT が
与えられていたら止まる（`exp/pair_table.py` の定数がそれで黙って変わるため）。出力先にファイルがあれば止まる
（既存の結果を上書きしない）。出力：`<out>/e75_<項目>_*.csv`，`gaps.csv`（入力の欠けの数），
`checks.csv`（保存済みの値の再現の検算），`provenance.json`（設定，入力，コミット番号，所要時間）。
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # GPU を使わない（ほかの計算が使う）．import より前に置く

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
from scipy import optimize, stats  # noqa: E402
from sklearn.covariance import LedoitWolf, ledoit_wolf_shrinkage  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
for _v in ("MERDA_OUT_ROOT", "MERDA_MAP_ROOT"):
    if os.environ.get(_v):
        raise SystemExit(f"環境変数 {_v} が与えられている．この台本は入力の場所を引数でだけ受ける．外してから流す")
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
sys.path.insert(0, str(ROOT / "features"))
import common  # noqa: E402
import pair_table  # noqa: E402
from ablate import balanced_subset, fit_class_bias  # noqa: E402
from data import GROUP_SPEC, groups_for, load_any  # noqa: E402
from explore_direction_bound import q_fit  # noqa: E402
from explore_geometry import standardize  # noqa: E402
from explore_pairs import ba, measure, own_separation  # noqa: E402
from explore_synergy import PAIRS, d_o  # noqa: E402
from split import speaker_folds  # noqa: E402
from summarize_map import K_ORDER, kstar_table, load as load_map_dir, t_interval  # noqa: E402

KINDS_ALL = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video",
             "text_roberta", "fuse-audio-text", "fuse-video-text", "fuse-audio-video-text"]
TEXT_KINDS = ["text_roberta", "fuse-audio-text", "fuse-video-text", "fuse-audio-video-text"]
MAIN4 = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]
SHORT = {"audio_wavlm": "A", "audio_hubert": "A(HuBERT)", "video": "V", "fuse-audio-video": "AV",
         "text_roberta": "T", "fuse-audio-text": "AT", "fuse-video-text": "VT", "fuse-audio-video-text": "AVT"}
IEMOCAP = ("IEMOCAP-impro", "IEMOCAP-script")
SCRIPT = "IEMOCAP-script"
EMO = ["neutral", "happy", "sad", "angry"]
PAIR_IDX = [f"{a}-{b}" for a, b in PAIRS]                       # explore_synergy.PAIRS の順
PAIR_NAME_TO_IDX = {f"{EMO[a]}-{EMO[b]}": f"{a}-{b}" for a, b in PAIRS}
C_GRID = ("0.01", "0.1", "1", "10")
ITEMS = list(range(1, 14))
DEN_SMALL = 0.5          # D11：割合の分母（点）がこれ未満で 0 より大きい単位は「小さい」と数える
LOC_MIN = 0.1            # D11：項目 6 の割合の分母（位置の損．点）がこれ以下なら割合を出さない
Q_GRID = np.logspace(-1, 3, 400)                                # explore_direction_bound.q_fit と同じ格子
BETA_GRID = np.linspace(0.25, 3.0, 56)
SCALE_GRID = np.linspace(0.5, 2.0, 61)

DEFAULTS = {
    "fixc_root": "results/fixc_map",                           # C を固定した地図（<kind>/<T>.csv）
    "cv_map_prefix": "results/map_",                            # 交差検証の地図（<prefix><kind>/<T>.csv）
    "stageb_main": "results/fixc_analysis/explore_relation",    # 段階 B（主な 4 種類）：e15b_rows.csv・e15b_pairs.csv・e15b_tests.csv・e13_pairs.csv
    "stageb_text": "results/fixc_text/explore_relation",        # 段階 B（言葉系 4 種類）
    "e29_pairs": "results/explore_relation/snapshots/20261011-1524_e29/e29_pairs.csv",
    "e28t9_targets": "results/explore_relation/snapshots/20261011-0139_e28t9/e28t9_targets.csv",
    "decomp_share_main": "results/fixc_analysis/explore_relation/decomp_share.csv",
    "decomp_share_text": "results/text30/explore_relation/decomp_share.csv",
    "kstar_cluster_main": "results/explore_relation/kstar_cluster.csv",
    "kstar_cluster_text": "results/text30/explore_relation/kstar_cluster.csv",
    "e12_wt": "results/fixc_analysis/explore_relation/e12_wt.csv",
    "shift_prefix": "results/shift_",                           # <prefix><kind>/logits/<S>.npz，label.csv，labeling.csv
    "tgt_scores_prefix": "results/tgt_scores_",
    "transfer_scores_prefix": "results/transfer_scores_",
    "mech_prefix": "results/mech_",
}


# ---------------------------------------------------------------- 共通

class Ctx:
    """引数，入力の欠け，検算，項目どうしで共有する表を持つ。"""

    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.out = Path(a.out)
        self.gaps: list[dict] = []
        self.checks: list[dict] = []
        self.inputs: dict[str, dict] = {}
        self._meta: dict[tuple[str, str], pd.DataFrame] = {}
        self._e13: dict[str, pd.DataFrame] = {}
        self.cache: dict[str, object] = {}

    def p(self, key: str) -> Path:
        v = Path(getattr(self.a, key))
        return v if v.is_absolute() else ROOT / v

    def note_input(self, path: Path) -> None:
        path = Path(path)
        if str(path) in self.inputs or not path.exists():
            return
        info = {"bytes": path.stat().st_size if path.is_file() else None,
                "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(path.stat().st_mtime))}
        if path.is_file() and path.suffix == ".csv":
            with open(path, encoding="utf-8") as f:
                head = f.readline().strip()
            if head.startswith("#"):
                info["first_comment"] = head[:400]
        self.inputs[str(path)] = info

    def gap(self, item: int, kind: str, what: str, expected, found, detail: str = "") -> None:
        self.gaps.append({"item": item, "kind": kind, "what": what, "expected": expected,
                          "found": found, "detail": detail})

    def check(self, item: int, kind: str, what: str, value: float, tol: float, detail: str = "") -> None:
        ok = bool(np.isfinite(value) and abs(value) <= tol) if not isinstance(value, str) else False
        self.checks.append({"item": item, "kind": kind, "what": what, "value": value, "tol": tol,
                            "ok": ok, "detail": detail})

    def meta(self, kind: str, group: str) -> pd.DataFrame:
        """発話一覧（sample_id・speaker_group・label_4）．特徴は `load_any` で読んで捨てる（取り決め 1〜6 を通すため）。"""
        key = (kind, group)
        if key not in self._meta:
            df, _ = load_any(group, kind)
            self._meta[key] = df[["sample_id", "speaker_group", "label_4"]].reset_index(drop=True)
        return self._meta[key]

    def e13(self, kind: str) -> pd.DataFrame:
        """E13 の感情の対ごとの表（d，o，own_d，損）．主な 7 種類は C 固定の解析の保存済みの表，
        HuBERT は保存が無いので `explore_pairs.measure` で作る（E28-9 の台本と同じ扱い）。"""
        if kind not in self._e13:
            path = self.p("stageb_main") / "e13_pairs.csv"
            self.note_input(path)
            e = pd.read_csv(path)
            e = e[e.kind == kind]
            if e.empty:
                e = measure(kind).assign(kind=kind)
                self.cache.setdefault("e13_computed", []).append(kind)
            e = e.copy()
            e["pair_idx"] = e.pair.map(PAIR_NAME_TO_IDX)
            self._e13[kind] = e
        return self._e13[kind]

    def write(self, name: str, df: pd.DataFrame) -> None:
        path = self.out / name
        if path.exists():
            raise SystemExit(f"{path} がすでにある．上書きしない")
        df.to_csv(path, index=False)
        self.cache.setdefault("written", []).append({"file": name, "rows": int(len(df))})


def same_corpus(t: str, s: str) -> bool:
    return GROUP_SPEC[t][0] == GROUP_SPEC[s][0]


def read_csv(ctx: Ctx, path: Path, **kw) -> pd.DataFrame:
    ctx.note_input(path)
    return pd.read_csv(path, comment="#", **kw)


def spear(x, y) -> tuple[float, float, int]:
    x, y = pd.Series(np.asarray(x, float)), pd.Series(np.asarray(y, float))
    ok = x.notna() & y.notna()
    if ok.sum() < 3 or x[ok].nunique() < 2 or y[ok].nunique() < 2:
        return np.nan, np.nan, int(ok.sum())
    r, p = stats.spearmanr(x[ok], y[ok])
    return float(r), float(p), int(ok.sum())


def kstar_from(tab: pd.DataFrame, lo_col: str) -> pd.DataFrame:
    """`summarize_map.kstar_table` を，任意の下端の列で呼ぶ（K* ＝ 下端が 0 を上回る最大の K）。"""
    x = tab[["target", "source", "K", "n_rep", lo_col]].rename(columns={lo_col: "diff_lo"}).copy()
    x["K"] = pd.Categorical(x["K"].astype(str), K_ORDER, ordered=True)
    return kstar_table(x)[["target", "source", "K_star"]]


def git_state() -> dict:
    def run(*cmd):
        try:
            return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except Exception as e:                                     # noqa: BLE001
            return f"error: {e}"
    status = run("git", "status", "--porcelain")
    me = Path(__file__).resolve()
    return {"commit": run("git", "rev-parse", "HEAD"), "dirty": bool(status),
            "script_status": run("git", "status", "--porcelain", "--", str(me.relative_to(ROOT))),
            "script_sha256": hashlib.sha256(me.read_bytes()).hexdigest()}


# ---------------------------------------------------------------- 項目 1：区間

def test_clusters(ctx: Ctx, kind: str, target: str, reps) -> dict[int, frozenset]:
    """反復 r のテストの話者群の集合（`exp/map.py` の test_mask と同じ：speaker_folds(seed=r) の r mod 5 番）。"""
    sp = ctx.meta(kind, target).speaker_group.astype(str).reset_index(drop=True)
    out = {}
    for r in reps:
        f = speaker_folds(sp, k=5, seed=int(r))
        out[int(r)] = frozenset(sp[(f == int(r) % 5).to_numpy()].unique())
    return out


def intervals(x: np.ndarray, reps: np.ndarray, rho: float, clus: dict | None) -> dict:
    """1 つの組 × K の反復ごとの差 x から，素朴な t，補正した resampled t，テストの組を単位にした t を作る。"""
    J = len(x)
    out = {"n_rep": J}
    m, lo, hi = t_interval(x)                                  # 反復を独立とみなす（今の地図の区間）
    out.update(mean=m, naive_lo=lo, naive_hi=hi)
    if J >= 2:
        sd = float(np.std(x, ddof=1))
        se_c = sd * np.sqrt(1.0 / J + rho)                     # Nadeau & Bengio (2003)
        tq = stats.t.ppf(0.975, J - 1)
        out.update(corr_lo=m - tq * se_c, corr_hi=m + tq * se_c,
                   corr_p=float(2 * stats.t.sf(abs(m) / se_c, J - 1)) if se_c > 0 else np.nan)
    else:
        out.update(corr_lo=np.nan, corr_hi=np.nan, corr_p=np.nan)
    if clus is not None:
        keys = {c: i for i, c in enumerate(sorted({clus[int(r)] for r in reps}, key=sorted))}
        ids = np.array([keys[clus[int(r)]] for r in reps])
        cm = np.array([x[ids == c].mean() for c in np.unique(ids)])
        n_c = len(cm)
        out["n_clusters"] = n_c
        if n_c >= 2:
            mc, lc, hc = t_interval(cm)
            se = float(np.std(cm, ddof=1) / np.sqrt(n_c))
            out.update(clus_lo=lc, clus_hi=hc,
                       clus_p=float(2 * stats.t.sf(abs(mc) / se, n_c - 1)) if se > 0 else np.nan)
        else:
            out.update(clus_lo=np.nan, clus_hi=np.nan, clus_p=np.nan)
    return out


def gain_table(ctx: Ctx, kind: str, item: int, tgt_col: str = "uar",
               merged_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """反復ごとの得（合併 − ターゲットのみ）．tgt_col でターゲットのみの UAR の列を替える（項目 4）。
    返すのは (得の表, 地図の表)。"""
    d_dir = merged_dir if merged_dir is not None else ctx.p("fixc_root") / kind
    for f in sorted(d_dir.glob("*.csv")):
        ctx.note_input(f)
    df = load_map_dir(d_dir)
    if not np.isfinite(df[df.method != "src_only"]["uar"]).all():
        raise SystemExit(f"{d_dir} の uar に NaN か Inf がある")
    if tgt_col == "uar" and merged_dir is None:
        pair_table.MAP_ROOT = ctx.p("fixc_root")              # 環境変数ではなく引数で地図の場所を決める
        g = pair_table.gains(kind)
    else:
        t = df[df.method == "tgt_only"].set_index(["target", "K", "rep"])[tgt_col]
        m = df[df.method == "merged"].copy()
        m["gain"] = m["uar"].to_numpy() - t.loc[list(zip(m.target, m.K, m.rep))].to_numpy()
        g = m[["target", "source", "K", "rep", "gain"]].reset_index(drop=True)
    if not np.isfinite(g.gain).all():
        raise SystemExit(f"{kind}：得に NaN か Inf がある（{tgt_col}）")
    return g, df


def interval_table(ctx: Ctx, kind: str, g: pd.DataFrame, df: pd.DataFrame, item: int,
                   n_rep_expected: int | None) -> pd.DataFrame:
    tg = df[df.method == "tgt_only"]
    allrow = tg[tg.K == "all"]
    rho_t = (allrow.n_test / allrow.n_tgt).groupby(allrow.target).mean()
    groups = sorted(g.target.unique())
    clus = {t: test_clusters(ctx, kind, t, sorted(g[g.target == t].rep.unique())) for t in groups
            if t in IEMOCAP}
    rows = []
    for (t, s, K), h in g.groupby(["target", "source", "K"]):
        h = h.sort_values("rep")
        x, reps = h.gain.to_numpy(float), h.rep.to_numpy()
        if n_rep_expected is not None and len(x) != n_rep_expected:
            ctx.gap(item, kind, f"反復の数 {t}←{s} K={K}", n_rep_expected, len(x))
        r = intervals(x, reps, float(rho_t.get(t, np.nan)), clus.get(t))
        main = "cluster" if t in IEMOCAP else "corrected"
        pre = "clus" if main == "cluster" else "corr"
        r.update(kind=kind, target=t, source=s, K=K, same_corpus=same_corpus(t, s), rho=float(rho_t.get(t, np.nan)),
                 main_method=main, main_lo=r[f"{pre}_lo"], main_hi=r[f"{pre}_hi"], main_p=r[f"{pre}_p"])
        rows.append(r)
    P = pd.DataFrame(rows)
    # Holm：種類 × K の中で，同じコーパスの組を除いた組を 1 つの族にする
    P["main_p_holm"] = np.nan
    for (_, K), h in P[~P.same_corpus].groupby(["kind", "K"]):
        ok = h.main_p.notna()
        P.loc[h.index[ok], "main_p_holm"] = pair_table.holm(h.main_p[ok].to_numpy())
    return P


def item1(ctx: Ctx) -> None:
    allP, testsets = [], []
    for kind in ctx.a.kinds:
        g, df = gain_table(ctx, kind, 1)
        groups = groups_for(kind)
        exp_pairs = len(groups) * (len(groups) - 1)
        found = g.groupby(["target", "source"]).ngroups
        if found != exp_pairs:
            ctx.gap(1, kind, "組の数", exp_pairs, found)
        P = interval_table(ctx, kind, g, df, 1, n_rep_expected=30)
        allP.append(P)
        # 検算：素朴な区間が地図のまとめ（summary/pairs.csv．summarize_map.pair_table の出力）と一致するか
        sp = ctx.p("fixc_root") / kind / "summary" / "pairs.csv"
        if sp.exists():
            ctx.note_input(sp)
            ref = pd.read_csv(sp, dtype={"K": str}).set_index(["target", "source", "K"])
            mine = P.set_index(["target", "source", "K"])
            j = mine.join(ref[["diff_lo", "diff_hi", "diff_mean"]], how="inner")
            ctx.check(1, kind, "素朴な t 区間の下端・上端・平均と summary/pairs.csv の差の最大",
                      float(np.nanmax(np.abs(np.r_[j.naive_lo - j.diff_lo, j.naive_hi - j.diff_hi, j["mean"] - j.diff_mean]))),
                      1e-9, f"{len(j)} 行")
        # テストの話者の集合の数（S6．感 Minor-1 の「異なるテストの折りの数」）
        for t in groups:
            reps = sorted(df[(df.target == t) & (df.method == "tgt_only")].rep.unique())
            cl = test_clusters(ctx, kind, t, reps)
            testsets.append({"kind": kind, "target": t, "n_reps": len(reps),
                             "n_distinct_test_sets": len(set(cl.values())),
                             "n_speaker_groups": int(ctx.meta(kind, t).speaker_group.nunique())})
    P = pd.concat(allP, ignore_index=True)
    ctx.cache["item1_pairs"] = P
    # K*：素朴，補正，主（IEMOCAP はテストの組，ほかは補正）
    ks = []
    for kind, h in P.groupby("kind"):
        k = None
        for name, col in (("naive", "naive_lo"), ("corrected", "corr_lo"), ("main", "main_lo"), ("cluster", "clus_lo")):
            if col not in h or h[col].notna().sum() == 0:
                continue
            x = kstar_from(h, col).rename(columns={"K_star": f"K_star_{name}"})
            if name == "cluster":                                  # テストの組の区間は IEMOCAP のターゲットだけで作る
                x.loc[~x.target.isin(IEMOCAP), f"K_star_{name}"] = np.nan
            k = x if k is None else k.merge(x, on=["target", "source"], how="outer")
        ks.append(k.assign(kind=kind))
    KS = pd.concat(ks, ignore_index=True)
    KS["same_corpus"] = [same_corpus(t, s) for t, s in zip(KS.target, KS.source)]
    ctx.cache["item1_kstar"] = KS
    # 検算：K* の素朴な値が summary/kstar.csv と，テストの組の値が A6 の kstar_cluster.csv（C 固定の地図）と一致するか
    for kind, h in KS.groupby("kind"):
        kp = ctx.p("fixc_root") / kind / "summary" / "kstar.csv"
        if kp.exists():
            ctx.note_input(kp)
            ref = pd.read_csv(kp, dtype=str).set_index(["target", "source"]).K_star
            j = h.set_index(["target", "source"]).K_star_naive
            ctx.check(1, kind, "素朴な K* が summary/kstar.csv と食い違う組の数", float((j != ref.reindex(j.index)).sum()), 0)
        for key in ("kstar_cluster_main", "kstar_cluster_text"):
            cp = ctx.p(key)
            if cp.exists():
                ctx.note_input(cp)
                ref = pd.read_csv(cp, dtype={"K_star_cluster": str})
                ref = ref[(ref["map"] == "fixc") & (ref.kind == kind)].set_index(["target", "source"]).K_star_cluster
                if len(ref) and "K_star_cluster" in h:
                    j = h.dropna(subset=["K_star_cluster"]).set_index(["target", "source"]).K_star_cluster
                    common_idx = j.index.intersection(ref.index)
                    ctx.check(1, kind, f"テストの組を単位にした K* が {cp.name}（{cp.parent.parent.name}）と食い違う組の数",
                              float((j.loc[common_idx] != ref.loc[common_idx]).sum()), 0, f"{len(common_idx)} 組")
    # 種類 × K の数え上げ（同じコーパスの組を除く）
    cnt = []
    Q = P[~P.same_corpus]
    for (kind, K), h in Q.groupby(["kind", "K"]):
        row = {"kind": kind, "short": SHORT.get(kind, kind), "K": K, "n_pairs": len(h), "mean_gain": h["mean"].mean() * 100}
        for name, lo, hi in (("naive", "naive_lo", "naive_hi"), ("corrected", "corr_lo", "corr_hi"),
                             ("main", "main_lo", "main_hi")):
            row[f"up_{name}"], row[f"down_{name}"] = int((h[lo] > 0).sum()), int((h[hi] < 0).sum())
        sig = h.main_p_holm < 0.05
        row["up_main_holm"], row["down_main_holm"] = int((sig & (h["mean"] > 0)).sum()), int((sig & (h["mean"] < 0)).sum())
        cnt.append(row)
    C = pd.DataFrame(cnt)
    C["K"] = pd.Categorical(C.K, K_ORDER, ordered=True)
    dist = []
    for kind, h in KS[~KS.same_corpus].groupby("kind"):
        for name in ("naive", "corrected", "main"):
            col = f"K_star_{name}"
            if col in h:
                vc = h[col].value_counts()
                dist.append({"kind": kind, "short": SHORT.get(kind, kind), "interval": name, "n_pairs": len(h),
                             **{f"K*={k}": int(vc.get(k, 0)) for k in K_ORDER + ["none"]}})
    ctx.write("e75_1_pairs.csv", P)
    ctx.write("e75_1_counts.csv", C.sort_values(["kind", "K"]))
    ctx.write("e75_1_kstar.csv", KS)
    ctx.write("e75_1_kstar_dist.csv", pd.DataFrame(dist))
    ctx.write("e75_1_testsets.csv", pd.DataFrame(testsets))


def need_item1(ctx: Ctx) -> None:
    if "item1_pairs" not in ctx.cache:
        item1(ctx)


# ---------------------------------------------------------------- 項目 2：IEMOCAP 台本を除いた集計

def item2(ctx: Ctx) -> None:
    need_item1(ctx)
    P, KS = ctx.cache["item1_pairs"], ctx.cache["item1_kstar"]
    kinds = [k for k in ctx.a.kinds if k in TEXT_KINDS]
    rows, tg, kd = [], [], []
    for kind in kinds:
        h = P[(P.kind == kind) & ~P.same_corpus]
        for subset, x in (("all_targets", h), ("excl_script_target", h[h.target != SCRIPT])):
            for K, xk in x.groupby("K"):
                rows.append({"kind": kind, "short": SHORT[kind], "subset": subset, "K": K, "n_pairs": len(xk),
                             "n_targets": xk.target.nunique(), "mean_gain": xk["mean"].mean() * 100,
                             "median_gain": xk["mean"].median() * 100,
                             "up_main": int((xk.main_lo > 0).sum()), "down_main": int((xk.main_hi < 0).sum())})
            k = KS[(KS.kind == kind) & ~KS.same_corpus]
            k = k if subset == "all_targets" else k[k.target != SCRIPT]
            vc = k.K_star_main.value_counts()
            kd.append({"kind": kind, "short": SHORT[kind], "subset": subset, "interval": "main",
                       "n_pairs": len(k), **{f"K*={kk}": int(vc.get(kk, 0)) for kk in K_ORDER + ["none"]}})
        for (t, K), xk in h.groupby(["target", "K"]):
            tg.append({"kind": kind, "short": SHORT[kind], "target": t, "K": K, "n_sources": len(xk),
                       "mean_gain": xk["mean"].mean() * 100})
    if not kinds:
        ctx.gap(2, "-", "言葉を含む種類が --kinds に無い", 4, 0)
        return
    R = pd.DataFrame(rows)
    ctx.write("e75_2_script_excluded.csv", R)
    ctx.write("e75_2_kstar_dist.csv", pd.DataFrame(kd))
    ctx.write("e75_2_targets.csv", pd.DataFrame(tg))


# ---------------------------------------------------------------- 項目 3：加法モデル

def additive(y: np.ndarray, d: pd.DataFrame, n_perm: int, rng: np.random.Generator) -> dict:
    Xt, Xts = pair_table.design(d, True, False), pair_table.design(d, True, True)
    n, k_t, k_ts = len(y), Xt.shape[1] - 1, Xts.shape[1] - 1
    r2t, _ = pair_table.r2(y, Xt)
    r2ts, _ = pair_table.r2(y, Xts)
    adj = lambda r, k: 1 - (1 - r) * (n - 1) / (n - k - 1) if n - k - 1 > 0 else np.nan  # noqa: E731
    null_ts, null_t = np.empty(n_perm), np.empty(n_perm)
    for b in range(n_perm):
        yp = rng.permutation(y)
        null_t[b] = pair_table.r2(yp, Xt)[0]
        null_ts[b] = pair_table.r2(yp, Xts)[0]
    return {"n": n, "n_params": k_ts + 1, "df_resid": n - k_ts - 1, "r2_target": r2t, "r2_both": r2ts,
            "r2_source": r2ts - r2t, "adj_r2_target": adj(r2t, k_t), "adj_r2_both": adj(r2ts, k_ts),
            "null_expect_r2_both": k_ts / (n - 1), "perm_mean_r2_both": float(null_ts.mean()),
            "perm_q95_r2_both": float(np.quantile(null_ts, 0.95)),
            "perm_p_r2_both": float((1 + (null_ts >= r2ts - 1e-12).sum()) / (1 + n_perm)),
            "perm_mean_r2_target": float(null_t.mean()),
            "perm_p_r2_target": float((1 + (null_t >= r2t - 1e-12).sum()) / (1 + n_perm))}


def item3(ctx: Ctx) -> None:
    need_item1(ctx)
    P, KS = ctx.cache["item1_pairs"], ctx.cache["item1_kstar"]
    rank = pair_table.KSTAR_RANK
    rows = []
    for kind in ctx.a.kinds:
        h = P[P.kind == kind]
        piv = h.pivot_table(index=["target", "source"], columns="K", values="mean").reset_index()
        k = KS[KS.kind == kind].set_index(["target", "source"])
        piv["kstar_main"] = [rank.get(k.loc[(t, s), "K_star_main"], np.nan) for t, s in zip(piv.target, piv.source)]
        piv["kstar_naive_nonIEMOCAP"] = [np.nan if t in IEMOCAP else rank.get(k.loc[(t, s), "K_star_naive"], np.nan)
                                         for t, s in zip(piv.target, piv.source)]
        piv["same_corpus"] = [same_corpus(t, s) for t, s in zip(piv.target, piv.source)]
        for pair_set, base in (("excl_same_corpus", piv[~piv.same_corpus]), ("all_pairs", piv)):
            for var in [f"gain_K{K}" for K in K_ORDER] + ["kstar_main", "kstar_naive_nonIEMOCAP"]:
                col = var.replace("gain_K", "") if var.startswith("gain_K") else var
                x = base[["target", "source", col]].dropna()
                if len(x) < 8:
                    ctx.gap(3, kind, f"{pair_set} {var} の組が少ない", 8, len(x))
                    continue
                rng = common.rng_for(f"e75-perm:{kind}:{pair_set}:{var}", ctx.a.seed)
                y = x[col].to_numpy(float) * (100 if var.startswith("gain_K") else 1)
                rows.append({"kind": kind, "short": SHORT[kind], "pair_set": pair_set, "variable": var,
                             **additive(y, x, ctx.a.n_perm, rng)})
    R = pd.DataFrame(rows)
    # 検算：全組の R² が保存済みの decomp_share.csv（C 固定の地図）と一致するか．fixc_analysis の版は言葉系が
    # 反復 5 回の時点（2026-10-09）の値なので主な種類だけと比べ，text30 の版（30 回）はすべての種類と比べる
    for key, only in (("decomp_share_main", MAIN4), ("decomp_share_text", KINDS_ALL)):
        path = ctx.p(key)
        if not path.exists():
            continue
        ctx.note_input(path)
        ref = pd.read_csv(path)
        ref = ref[ref.kind.isin(only)]
        for var, refvar in (("gain_K1", "gain_K1"), ("gain_K5", "gain_K5"), ("kstar_naive_nonIEMOCAP", "kstar")):
            a = R[(R.pair_set == "all_pairs") & (R.variable == var)].set_index("kind")
            b = ref[ref.variable == refvar].set_index("kind")
            j = a.join(b[["r2_target", "r2_both"]], rsuffix="_ref", how="inner")
            if len(j):
                ctx.check(3, ",".join(j.index), f"{var} の R²（全組）と {path.parent.parent.name}/decomp_share.csv の差の最大",
                          float(np.nanmax(np.abs(np.r_[j.r2_target - j.r2_target_ref, j.r2_both - j.r2_both_ref]))), 1e-6)
    ctx.write("e75_3_additive.csv", R)


# ---------------------------------------------------------------- 項目 4：C の規則

def item4(ctx: Ctx) -> None:
    rows, pairs_out, crs = [], [], []
    rules = [("map_rule", "uar", None)] + [(f"C{c}", f"test_uar_C{c}", None) for c in C_GRID] + \
            [("oracle", "uar_oracle", None), ("cv_map", "uar", "cv")]
    for kind in ctx.a.kinds:
        for rule, col, src in rules:
            mdir = (ROOT / f"{ctx.a.cv_map_prefix}{kind}") if src == "cv" else None
            if mdir is not None and not mdir.exists():
                ctx.gap(4, kind, "交差検証の地図が無い", 1, 0, str(mdir))
                continue
            g, df = gain_table(ctx, kind, 4, tgt_col=col, merged_dir=mdir)
            P = interval_table(ctx, kind, g, df, 4, n_rep_expected=None)
            P = P[~P.same_corpus]
            for K, h in P.groupby("K"):
                rows.append({"kind": kind, "short": SHORT[kind], "rule": rule, "K": K, "n_pairs": len(h),
                             "n_rep_min": int(h.n_rep.min()), "mean_gain": h["mean"].mean() * 100,
                             "n_pairs_mean_neg": int((h["mean"] < 0).sum()),
                             "up_main": int((h.main_lo > 0).sum()), "down_main": int((h.main_hi < 0).sum())})
            pairs_out.append(P.assign(rule=rule)[["kind", "rule", "target", "source", "K", "n_rep", "mean",
                                                   "main_lo", "main_hi"]])
            if rule == "map_rule":
                tg = df[df.method == "tgt_only"]
                for K, h in tg.groupby("K"):
                    vc = h.C_rule.value_counts(normalize=True)
                    crs.append({"kind": kind, "short": SHORT[kind], "K": K, "n_rows": len(h),
                                 **{f"share_{r}": float(vc.get(r, 0.0)) for r in ("cv", "tie", "fixed")}})
    R = pd.DataFrame(rows)
    # 得の平均が初めて負になる K（種類 × 規則）
    kneg = []
    for (kind, rule), h in R.groupby(["kind", "rule"]):
        h = h.set_index("K").reindex(K_ORDER)
        neg = [K for K in K_ORDER if h.loc[K, "mean_gain"] < 0]
        kneg.append({"kind": kind, "short": SHORT[kind], "rule": rule, "first_K_mean_gain_negative": neg[0] if neg else "none"})
    ctx.write("e75_4_c_rule.csv", R)
    ctx.write("e75_4_tgt_c_rule_share.csv", pd.DataFrame(crs))
    ctx.write("e75_4_first_negative_K.csv", pd.DataFrame(kneg))
    ctx.write("e75_4_pairs.csv", pd.concat(pairs_out, ignore_index=True))


# ---------------------------------------------------------------- 項目 5：分解の順序

def stageb_rows(ctx: Ctx, kinds: list[str], item: int) -> pd.DataFrame:
    out = []
    for key, ks in (("stageb_main", MAIN4), ("stageb_text", TEXT_KINDS)):
        use = [k for k in kinds if k in ks]
        if not use:
            continue
        r = read_csv(ctx, ctx.p(key) / "e15b_rows.csv", dtype={"K": str, "pair": str})
        r = r[r.kind.isin(use) & (r.rep < 10)]                   # E29 と同じく反復 0〜9
        out.append(r)
    R = pd.concat(out, ignore_index=True)
    R = R[[not same_corpus(t, s) for t, s in zip(R.target, R.source)]].copy()
    for (kind, t, s, K), h in R.groupby(["kind", "target", "source", "K"]):
        if len(h) != 60:
            ctx.gap(item, kind, f"段階 B の行（反復 10 × 対 6）{t}←{s} K={K}", 60, len(h))
    for kind in kinds:
        groups = groups_for(kind)
        exp = sum(1 for t in groups for s in groups if t != s and not same_corpus(t, s)) * 3
        found = R[R.kind == kind].groupby(["target", "source", "K"]).ngroups
        if found != exp:
            ctx.gap(item, kind, "段階 B の組 × K の数", exp, found)
    if not np.isfinite(R[["d_s", "o_s", "d_m", "o_m"]].to_numpy(float)).all():
        raise SystemExit("段階 B の d・o に NaN か Inf がある")
    return R


def shares(pos: pd.Series, tot: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """D11：分母（合計）が 0 以下の単位は割合を定めない（NaN）．0 < 分母 < DEN_SMALL は印を付ける。"""
    sh = pos / tot
    sh[tot <= 0] = np.nan
    return sh, tot <= 0, (tot > 0) & (tot < DEN_SMALL)


def item5(ctx: Ctx) -> None:
    kinds = [k for k in ctx.a.kinds if k in MAIN4 + TEXT_KINDS]
    R = stageb_rows(ctx, kinds, 5)
    R["pos_f"] = (ba(R.d_s, R.o_m) - ba(R.d_s, R.o_s)) * 100           # 今の順序（E15 の 8 節，E29）：位置が先
    R["dir_f"] = (ba(R.d_m, R.o_m) - ba(R.d_s, R.o_m)) * 100
    R["dir_r"] = (ba(R.d_m, R.o_s) - ba(R.d_s, R.o_s)) * 100           # 逆の順序：向きが先
    R["pos_r"] = (ba(R.d_m, R.o_m) - ba(R.d_m, R.o_s)) * 100
    R["neutral_pair"] = R.pair.str.startswith("0-")
    key = ["kind", "target", "source", "K"]
    P = R.groupby(key)[["pos_f", "dir_f", "pos_r", "dir_r"]].mean().reset_index()
    P["pos_sh"], P["dir_sh"] = (P.pos_f + P.pos_r) / 2, (P.dir_f + P.dir_r) / 2
    P["total"] = P.pos_f + P.dir_f
    for o in ("f", "r", "sh"):
        P[f"share_{o}"], P["den_le0"], P["den_small"] = shares(P[f"pos_{o}"], P.total)
    # 検算：今の順序の値が E29 の e29_pairs.csv と一致するか
    path = ctx.p("e29_pairs")
    if path.exists():
        e = read_csv(ctx, path, dtype={"K": str})
        j = P.merge(e[key + ["pos", "dir"]], on=key, how="inner")
        ctx.check(5, "all", "今の順序の位置の分・向きの分と e29_pairs.csv の差の最大",
                  float(np.nanmax(np.abs(np.r_[j.pos_f - j.pos, j.dir_f - j.dir]))), 1e-6, f"{len(j)} 行")
    # 感情の対の種類（平静を含む対とそれ以外）ごとの平均
    E = R.groupby(["kind", "K", "neutral_pair"])[["pos_f", "dir_f", "pos_r", "dir_r"]].mean().reset_index()
    rows = []
    groups_def = [(k, P.kind == k) for k in kinds]
    tk = [k for k in kinds if k in TEXT_KINDS]
    if tk:
        groups_def += [("text4", P.kind.isin(tk)), ("text4_excl_script_target", P.kind.isin(tk) & (P.target != SCRIPT))]
    for name, mask in groups_def:
        for K, h in P[mask].groupby("K"):
            row = {"group": name, "short": SHORT.get(name, name), "K": K, "n_pairs": len(h),
                   "n_den_le0": int(h.den_le0.sum()), "n_den_small": int(h.den_small.sum()),
                   "mean_total": h.total.mean()}
            for o in ("f", "r", "sh"):
                row[f"mean_pos_{o}"], row[f"mean_dir_{o}"] = h[f"pos_{o}"].mean(), h[f"dir_{o}"].mean()
                row[f"median_share_{o}"] = h[f"share_{o}"].median()
                row[f"ratio_of_sums_{o}"] = h[f"pos_{o}"].sum() / h.total.sum() if h.total.sum() > 0 else np.nan
                row[f"n_share_gt_half_{o}"] = int((h[f"share_{o}"] > 0.5).sum())
            rows.append(row)
    ctx.write("e75_5_pairs.csv", P)
    ctx.write("e75_5_summary.csv", pd.DataFrame(rows))
    ctx.write("e75_5_neutral_pairs.csv", E)


# ---------------------------------------------------------------- 項目 6：6 対を同時に直す切片

def pair_sd(L: np.ndarray, y: np.ndarray, base: list[tuple[float, float]]) -> np.ndarray:
    """感情の対ごとの 2 つの山の標準偏差 s．`d_o` は s を返さないので，k 列に 1 を足したときの o の変化
    （o は −(中点)/s なので，k が対の先なら −1/s，後なら +1/s 動く）から逆算する。"""
    sd = np.full(len(PAIRS), np.nan)
    for k in (1, 2, 3):
        L2 = L.copy()
        L2[:, k] += 1.0
        moved = d_o(L2, y)
        for i, (a, b) in enumerate(PAIRS):
            if a == k:
                sd[i] = 1.0 / (base[i][1] - moved[i][1])
            elif b == k:
                sd[i] = 1.0 / (moved[i][1] - base[i][1])
    return sd


def joint_offset(d: np.ndarray, o: np.ndarray, sd: np.ndarray) -> tuple[float, np.ndarray]:
    """2 つの山の模型で，4 つの切片のずらし c（c_0 = 0）を 1 組だけ選び，6 対の位置の損の平均を最小にする。"""
    A = np.zeros((len(PAIRS), 3))
    for i, (a, b) in enumerate(PAIRS):
        if a > 0:
            A[i, a - 1] += 1.0 / sd[i]
        if b > 0:
            A[i, b - 1] -= 1.0 / sd[i]

    def f(c):
        oc = o - A @ c
        return float(np.mean(ba(d, 0) - ba(d, oc)))
    starts = [np.zeros(3), np.linalg.lstsq(A, o, rcond=None)[0]]
    best = None
    for s0 in starts:
        r = optimize.minimize(f, s0, method="L-BFGS-B")
        if best is None or r.fun < best.fun:
            best = r
    return float(best.fun), best.x


def pair_acc(L: np.ndarray, y: np.ndarray) -> np.ndarray:
    """6 対の実際の正解率 ½[P(z > 0 | k) + P(z < 0 | l)]（E13 の acc_actual と同じ定義）。"""
    out = []
    for a, b in PAIRS:
        za, zb = L[y == a, a] - L[y == a, b], L[y == b, a] - L[y == b, b]
        out.append(0.5 * ((za > 0).mean() + (zb < 0).mean()))
    return np.array(out)


def pair_best_threshold(L: np.ndarray, y: np.ndarray) -> np.ndarray:
    """対ごとに境目を最良の位置に置いたときの実際の正解率（対ごとに取り戻せる上限）。"""
    out = []
    for a, b in PAIRS:
        za, zb = L[y == a, a] - L[y == a, b], L[y == b, a] - L[y == b, b]
        cand = np.unique(np.r_[za, zb, -np.inf, np.inf])
        th = np.r_[(cand[:-1] + cand[1:]) / 2]
        th = th[np.isfinite(th)] if np.isfinite(th).any() else np.array([0.0])
        sa, sb = np.sort(za), np.sort(zb)
        pa = 1 - np.searchsorted(sa, th, side="right") / len(sa)     # P(z > t | a)
        pb = np.searchsorted(sb, th, side="left") / len(sb)           # P(z < t | b)
        out.append(float(np.max(0.5 * (pa + pb))))
    return np.array(out)


def uar(L: np.ndarray, y: np.ndarray) -> float:
    pred = L.argmax(1)
    return float(np.mean([(pred[y == c] == c).mean() for c in range(4)]))


def item6(ctx: Ctx) -> None:
    rows = []
    for kind in ctx.a.kinds:
        groups = groups_for(kind)
        lab = ROOT / f"{ctx.a.shift_prefix}{kind}" / "labeling.csv"
        src_uar = read_csv(ctx, lab).set_index(["target", "source"]).uar_t if lab.exists() else None
        e13 = ctx.e13(kind)
        for s in groups:
            path = ROOT / f"{ctx.a.shift_prefix}{kind}" / "logits" / f"{s}.npz"
            if not path.exists():
                ctx.gap(6, kind, f"ソースのみの点数が無い {s}", 1, 0, str(path))
                continue
            ctx.note_input(path)
            z = np.load(path)
            for t in groups:
                if t == s or same_corpus(t, s):
                    continue
                if f"t_{t}" not in z.files:
                    ctx.gap(6, kind, f"点数の行が無い t_{t}（ソース {s}）", 1, 0)
                    continue
                L = z[f"t_{t}"].astype(np.float64)
                m = ctx.meta(kind, t)
                y = m.label_4.to_numpy()
                if len(L) != len(y):
                    raise SystemExit(f"{kind} {s}→{t}：点数 {len(L)} 行と発話 {len(y)} 本が合わない")
                base = d_o(L, y)
                d, o = np.array([b[0] for b in base]), np.array([b[1] for b in base])
                sd = pair_sd(L, y, base)
                loc = float(np.mean(ba(d, 0) - ba(d, o)))
                resid, c = joint_offset(d, o, sd)
                acc0, best = pair_acc(L, y), pair_best_threshold(L, y)
                rng = common.rng_for(f"e75-bias:{kind}:{s}:{t}", 0)
                idx = balanced_subset(y, rng)
                b_all = fit_class_bias(L[idx], y[idx], l2=0.0)
                acc1 = pair_acc(L + b_all, y)
                mod1 = d_o(L + b_all, y)
                loc_after_fit = float(np.mean([ba(dd, 0) - ba(dd, oo) for dd, oo in mod1]))
                # 話者の半分で切片を学び，もう半分で測る（E13 の Δ と同じ分け方：speaker_folds(k=2, seed=0)）
                half = speaker_folds(m.speaker_group, k=2, seed=0).to_numpy()
                cf = {"uar0": [], "uar1": [], "acc0": [], "acc1": []}
                for h in (0, 1):
                    fi, ev = half == h, half == 1 - h
                    if any(np.bincount(y[fi], minlength=4) == 0) or any(np.bincount(y[ev], minlength=4) == 0):
                        ctx.gap(6, kind, f"半分に欠ける感情 {t}", 1, 0, f"half={h}")
                        continue
                    r2 = common.rng_for(f"e75-bias-half:{kind}:{s}:{t}", h)
                    ii = np.flatnonzero(fi)[balanced_subset(y[fi], r2)]
                    bh = fit_class_bias(L[ii], y[ii], l2=0.0)
                    cf["uar0"].append(uar(L[ev], y[ev]))
                    cf["uar1"].append(uar(L[ev] + bh, y[ev]))
                    cf["acc0"].append(pair_acc(L[ev], y[ev]).mean())
                    cf["acc1"].append(pair_acc(L[ev] + bh, y[ev]).mean())
                ref = e13[(e13.target == t) & (e13.source == s)].set_index("pair_idx")
                if len(ref):
                    ctx.check(6, kind, f"位置の損（対ごと）と e13_pairs の差 {t}←{s}",
                              float(np.max(np.abs(ref.loss_offset.reindex(PAIR_IDX).to_numpy() - (ba(d, 0) - ba(d, o))))),
                              1e-6)
                if src_uar is not None and (t, s) in src_uar.index:
                    ctx.check(6, kind, f"ソースのみの UAR と labeling.csv の uar_t の差 {t}←{s}",
                              abs(uar(L, y) - float(src_uar.loc[(t, s)])), 1e-6)
                rows.append({"kind": kind, "target": t, "source": s, "n_target": len(y), "n_fit_balanced": len(idx),
                             "loc_loss_pairwise": loc * 100, "loc_loss_after_joint_model": resid * 100,
                             "loc_recovered_joint_model": (loc - resid) * 100,
                             "frac_recovered_joint_model": (loc - resid) / loc if loc * 100 > LOC_MIN else np.nan,
                             "loc_loss_after_ce_fit": loc_after_fit * 100,
                             "acc_pair_src": acc0.mean() * 100, "acc_pair_best_per_pair": best.mean() * 100,
                             "acc_pair_after_ce_fit": acc1.mean() * 100,
                             "uar_src": uar(L, y) * 100, "uar_after_ce_fit": uar(L + b_all, y) * 100,
                             "cf_uar_src": np.mean(cf["uar0"]) * 100 if cf["uar0"] else np.nan,
                             "cf_uar_after_fit": np.mean(cf["uar1"]) * 100 if cf["uar1"] else np.nan,
                             "cf_acc_pair_src": np.mean(cf["acc0"]) * 100 if cf["acc0"] else np.nan,
                             "cf_acc_pair_after_fit": np.mean(cf["acc1"]) * 100 if cf["acc1"] else np.nan,
                             **{f"c_joint_{k}": v for k, v in zip((1, 2, 3), c)},
                             **{f"b_ce_{k}": v for k, v in enumerate(b_all)}})
        groups_n = sum(1 for t in groups for s in groups if t != s and not same_corpus(t, s))
        found = sum(1 for r in rows if r["kind"] == kind)
        if found != groups_n:
            ctx.gap(6, kind, "組の数", groups_n, found)
    P = pd.DataFrame(rows)
    S = []
    for kind, h in P.groupby("kind"):
        S.append({"kind": kind, "short": SHORT[kind], "n_pairs": len(h),
                  "n_frac_undefined": int(h.frac_recovered_joint_model.isna().sum()),
                  **{f"mean_{c}": h[c].mean() for c in P.columns if c.startswith(("loc_", "acc_", "uar_", "cf_"))},
                  "median_frac_recovered_joint_model": h.frac_recovered_joint_model.median(),
                  "min_frac_recovered_joint_model": h.frac_recovered_joint_model.min()})
    ctx.write("e75_6_pairs.csv", P)
    ctx.write("e75_6_summary.csv", pd.DataFrame(S))


# ---------------------------------------------------------------- 項目 7：式 (2) と測った閾値の損

GH_X, GH_W = np.polynomial.hermite_e.hermegauss(40)              # 標準正規の期待値のための節点


def item7(ctx: Ctx) -> None:
    rows = []
    for kind in ctx.a.kinds:
        for clf, prefix in (("tgt_only", ctx.a.tgt_scores_prefix), ("transfer_h7", ctx.a.transfer_scores_prefix)):
            dpath = ROOT / f"{prefix}{kind}"
            if not dpath.exists():
                ctx.gap(7, kind, f"{clf} の点数が無い", 1, 0, str(dpath))
                continue
            for f in sorted(dpath.glob("*.npz")):
                ctx.note_input(f)
                t = f.stem
                z = np.load(f)
                reps = sorted(int(k.split("_r")[1]) for k in z.files if k.startswith("test_idx_r"))
                if len(reps) != 10:
                    ctx.gap(7, kind, f"{clf} {t} の反復", 10, len(reps))
                if clf == "tgt_only":
                    keys = [(None, K, r, f"t_K{K}_r{r}") for r in reps for K in ("1", "5", "10")]
                else:
                    srcs = sorted({k.split("_")[1] for k in z.files if k.startswith("h7_")})
                    keys = [(s, K, r, f"h7_{s}_K{K}_r{r}") for s in srcs if not same_corpus(t, s)
                            for r in reps for K in ("1", "5", "10")]
                for s, K, r, key in keys:
                    if key not in z.files:
                        ctx.gap(7, kind, f"{clf} の点数の鍵が無い {t} {key}", 1, 0)
                        continue
                    y = z[f"y_test_r{r}"]
                    for (a, b), (dd, oo) in zip(PAIRS, d_o(z[key].astype(np.float64), y)):
                        rows.append({"kind": kind, "clf": clf, "target": t, "source": s or "", "K": K, "rep": r,
                                     "pair": f"{a}-{b}", "d": dd, "o": oo})
    if not rows:
        return
    R = pd.DataFrame(rows)
    Kn = R.K.astype(int).to_numpy()
    d, o = R.d.to_numpy(), R.o.to_numpy()
    R["loss_meas"] = (ba(d, 0) - ba(d, o)) * 100
    R["pred_2nd"] = d / 4 * stats.norm.pdf(d / 2) / (2 * Kn) * 100
    sdv = np.sqrt(1 / (2 * Kn))
    ex = np.zeros(len(R))
    for x, w in zip(GH_X, GH_W):
        ex += w * (ba(d, 0) - ba(d, x * sdv))
    R["pred_exact"] = ex / np.sqrt(2 * np.pi) * 100
    R["o2"] = o ** 2
    # 偏り：同じ（ターゲット，ソース，感情の対，K）の反復の平均の o（プールとテストの話者の平均のずれなど）
    g = R.groupby(["kind", "clf", "target", "source", "K", "pair"]).o
    R["o_cell_mean"] = g.transform("mean")
    S = []
    for (kind, clf, K), h in R.groupby(["kind", "clf", "K"]):
        S.append({"kind": kind, "short": SHORT[kind], "clf": clf, "K": K, "n_rows": len(h),
                  "mean_loss_meas": h.loss_meas.mean(), "mean_pred_2nd": h.pred_2nd.mean(),
                  "mean_pred_exact": h.pred_exact.mean(), "ratio_meas_to_2nd": h.loss_meas.mean() / h.pred_2nd.mean(),
                  "ratio_meas_to_exact": h.loss_meas.mean() / h.pred_exact.mean(),
                  "mean_o2": h.o2.mean(), "theory_o2": 1 / (2 * int(K)),
                  "bias_part_o2": float((h.o_cell_mean ** 2).mean()),
                  "var_part_o2": float(((h.o - h.o_cell_mean) ** 2).mean()),
                  "mean_d": h.d.mean()})
    ctx.write("e75_7_rows.csv", R)
    ctx.write("e75_7_summary.csv", pd.DataFrame(S))


# ---------------------------------------------------------------- 項目 8：向きの損が負の升，縮小の強さ

def item8(ctx: Ctx) -> None:
    rows, prs, shr = [], [], []
    for kind in ctx.a.kinds:
        e = ctx.e13(kind)
        e = e.assign(same_corpus=[same_corpus(t, s) for t, s in zip(e.target, e.source)])
        for subset, h in (("all_pairs", e), ("excl_same_corpus", e[~e.same_corpus])):
            neg = h.loss_direction < 0
            rows.append({"kind": kind, "short": SHORT[kind], "subset": subset, "n_cells": len(h),
                         "n_neg": int(neg.sum()), "frac_neg": float(neg.mean()),
                         "median_neg_points": float(h.loss_direction[neg].median() * 100) if neg.any() else np.nan,
                         "min_points": float(h.loss_direction.min() * 100),
                         **{f"n_neg_{p}": int((neg & (h.pair == p)).sum()) for p in PAIR_NAME_TO_IDX}})
        x = e[~e.same_corpus].copy()
        x["dir_clip"] = x.loss_direction.clip(lower=0)
        pp = x.groupby(["target", "source"]).agg(L_crit=("loss_offset", "mean"), L_disc_raw=("loss_direction", "mean"),
                                                 L_disc_clip=("dir_clip", "mean")).reset_index()
        for c in ("L_crit", "L_disc_raw", "L_disc_clip"):
            pp[c] *= 100
        for v in ("raw", "clip"):
            den = pp.L_crit + pp[f"L_disc_{v}"]
            pp[f"share_loss_{v}"] = (pp.L_crit / den).where(den > 0)
            pp[f"den_le0_{v}"] = den <= 0
        prs.append(pp.assign(kind=kind))
        # 縮小の強さ：E13 の Δ と同じ入力（standardize した特徴，話者の半分）で Ledoit–Wolf の shrinkage_ を取り出す
        groups = groups_for(kind)
        data = standardize({g: load_any(g, kind) for g in groups})
        # Δ の作り直しの検算は最も小さい 1 群だけ（`own_separation` は LedoitWolf.fit で精度行列まで作るので，
        # 2,304 次元で 1 群 40〜55 秒かかる．2026-11-01 の試走）
        small = sorted(data, key=lambda g: len(data[g][0]))[:1]
        for g, (df, X) in data.items():
            yv = df.label_4.to_numpy()
            half = speaker_folds(df.speaker_group, k=2, seed=0).to_numpy()
            if g in small:
                own = own_separation(X, yv, half)
                ref = e[e.target == g].groupby("pair_idx").own_d.mean().reindex(PAIR_IDX).to_numpy()
                if np.isfinite(ref).all():
                    ctx.check(8, kind, f"Δ（own_d）の作り直しと保存済みの差 {g}", float(np.max(np.abs(own - ref))), 1e-6)
            for hh in (0, 1):
                Xa, ya = X[half == hh], yv[half == hh]
                mu = np.stack([Xa[ya == k].mean(0) for k in range(4)])
                # LedoitWolf(assume_centered=True).fit(...) が内部で使う縮小の強さと同じ関数（float64 に直して同じ値）
                sh = float(ledoit_wolf_shrinkage((Xa - mu[ya]).astype(np.float64), assume_centered=True))
                if g == small[0] and hh == 0:
                    ref_sh = LedoitWolf(assume_centered=True).fit(Xa - mu[ya]).shrinkage_
                    ctx.check(8, kind, f"縮小の強さと LedoitWolf.fit の shrinkage_ の差 {g}", abs(sh - ref_sh), 1e-12)
                shr.append({"kind": kind, "group": g, "half": hh, "n": len(ya), "dim": X.shape[1], "shrinkage": sh})
        del data
    P = pd.concat(prs, ignore_index=True)
    # 合併の得に占める o の割合（項目 5 の 3 通りの順序，K=5）と損に占める o の割合の順位相関（P29-3 の形）を，
    # 向きの損の生の値と 0 で切った値で．項目 5 を同じ回に流したときだけ作る
    p293 = []
    if (ctx.out / "e75_5_pairs.csv").exists():
        Q = pd.read_csv(ctx.out / "e75_5_pairs.csv", dtype={"K": str})
        Q = Q[Q.K == "5"].merge(P, on=["kind", "target", "source"])
        for kind, h in list(Q.groupby("kind")) + [("all", Q)]:
            for o in ("f", "r", "sh"):
                for v in ("raw", "clip"):
                    r, pv, n = spear(h[f"share_{o}"], h[f"share_loss_{v}"])
                    p293.append({"kind": kind, "short": SHORT.get(kind, kind), "order": o, "dir_loss": v,
                                 "rho": r, "p": pv, "n": n, "median_share_loss": h[f"share_loss_{v}"].median(),
                                 "median_share_gain": h[f"share_{o}"].median(),
                                 "n_den_le0_loss": int(h[f"den_le0_{v}"].sum())})
    else:
        ctx.gap(8, "-", "項目 5 の出力が無いので P29-3 の行を作らない", 1, 0)
    ctx.write("e75_8_negative_direction.csv", pd.DataFrame(rows))
    if p293:
        ctx.write("e75_8_p29_3.csv", pd.DataFrame(p293))
    ctx.write("e75_8_pairs.csv", P)
    ctx.write("e75_8_shrinkage.csv", pd.DataFrame(shr))


# ---------------------------------------------------------------- 項目 9：中心化の得とクラスの割合

def item9(ctx: Ctx) -> None:
    path = ctx.p("e29_pairs")
    e = read_csv(ctx, path, dtype={"K": str})
    e = e[e.K == "5"]
    rows, prs = [], []
    for kind in ctx.a.kinds:
        lp = ROOT / f"{ctx.a.shift_prefix}{kind}" / "label.csv"
        if not lp.exists():
            ctx.gap(9, kind, "label.csv が無い", 1, 0)
            continue
        lab = read_csv(ctx, lp)
        lab["tv_target_uniform"] = 0.5 * sum((lab[f"p_t_{k}"] - 0.25).abs() for k in range(4))
        lab["neutral_share_target"] = lab.p_t_0
        h = e[e.kind == kind][["kind", "target", "source", "G_center"]].merge(
            lab[["target", "source", "l1_true", "tv_target_uniform", "neutral_share_target"]], on=["target", "source"])
        h = h[[not same_corpus(t, s) for t, s in zip(h.target, h.source)]]
        if h.empty:
            ctx.gap(9, kind, "中心化の得の行が無い", 1, 0)
            continue
        prs.append(h)
        r, p, n = spear(h.G_center, h.l1_true)
        pr = pair_table.partial_rank(h, "G_center", "l1_true")
        dfree = (pr or {}).get("df", 0)
        tgt = h.groupby("target").agg(G=("G_center", "mean"), tv=("tv_target_uniform", "first"),
                                     neu=("neutral_share_target", "first"))
        rt, ptv, nt = spear(tgt.G, tgt.tv)
        rn, pn, _ = spear(tgt.G, tgt.neu)
        rows.append({"kind": kind, "short": SHORT[kind], "n_pairs": n, "rho_Gcenter_l1": r, "p_Gcenter_l1": p,
                     "r_partial_Gcenter_l1": pr["r_partial_rank"] if pr and dfree >= 3 else np.nan,
                     "p_partial_Gcenter_l1": pr["p"] if pr and dfree >= 3 else np.nan, "df_partial": dfree,
                     "n_targets": nt, "rho_target_Gcenter_tv": rt, "rho_target_Gcenter_neutral": rn,
                     "n_Gcenter_neg": int((h.G_center < 0).sum()),
                     "median_l1_Gcenter_neg": float(h.l1_true[h.G_center < 0].median()) if (h.G_center < 0).any() else np.nan,
                     "median_l1_Gcenter_pos": float(h.l1_true[h.G_center >= 0].median())})
    ctx.write("e75_9_centering_prior.csv", pd.DataFrame(rows))
    if prs:
        ctx.write("e75_9_pairs.csv", pd.concat(prs, ignore_index=True))


# ---------------------------------------------------------------- 項目 10：E12 の α を選んだ合併の区間

def item10(ctx: Ctx) -> None:
    rows, prs = [], []
    ref_path = ctx.p("e12_wt")
    ref = read_csv(ctx, ref_path, dtype={"K": str}) if ref_path.exists() else None
    st_path = ref_path.parent / "e12_state.json"                  # e12_wt.csv を作ったときの反復（ターゲットごと）
    state = json.loads(st_path.read_text()) if st_path.exists() else {}
    if st_path.exists():
        ctx.note_input(st_path)
    for kind in ctx.a.kinds:
        mdir = ROOT / f"{ctx.a.mech_prefix}{kind}"
        if not mdir.exists():
            continue
        files = sorted(mdir.glob("*.csv"))
        for f in files:
            ctx.note_input(f)
        mech = pd.concat([pd.read_csv(f, comment="#", dtype={"K": str, "source": str},
                                      usecols=["op", "target", "source", "method", "K", "rep", "uar"]) for f in files])
        w = mech[(mech.op == "wt") & (mech.method == "merged")][["target", "source", "K", "rep", "uar"]]
        mp = load_map_dir(ctx.p("fixc_root") / kind)
        m = mp[mp.method == "merged"][["target", "source", "K", "rep", "uar"]].rename(columns={"uar": "m"})
        t = mp[mp.method == "tgt_only"][["target", "K", "rep", "uar"]].rename(columns={"uar": "t"})
        x = w.merge(m, on=["target", "source", "K", "rep"]).merge(t, on=["target", "K", "rep"])
        if len(x) != len(w):
            ctx.gap(10, kind, "E12 の行のうち地図と突き合わせられた数", len(w), len(x))
        for (tt, s, K), h in x.groupby(["target", "source", "K"]):
            if len(h) != 3:
                ctx.gap(10, kind, f"E12 の反復 {tt}←{s} K={K}", 3, len(h))
        pp = x.groupby(["target", "source", "K"]).agg(wt=("uar", "mean"), m=("m", "mean"), t=("t", "mean"),
                                                      n_rep=("rep", "nunique")).reset_index()
        pp["gain_wt"], pp["gain_nat"] = (pp.wt - pp.t) * 100, (pp.m - pp.t) * 100
        pp["wt_minus_nat"] = pp.gain_wt - pp.gain_nat
        prs.append(pp.assign(kind=kind))
        targets = sorted(pp.target.unique())
        for K, h in pp.groupby("K"):
            rng = common.rng_for(f"e75-e12boot:{kind}:{K}", ctx.a.seed)
            bt = {c: [] for c in ("gain_wt", "gain_nat", "wt_minus_nat")}
            by_t = {tt: h[h.target == tt] for tt in targets}
            for _ in range(ctx.a.n_boot_targets):
                pick = rng.choice(targets, size=len(targets), replace=True)
                hb = pd.concat([by_t[tt] for tt in pick])
                for c in bt:
                    bt[c].append(hb[c].mean())
            row = {"kind": kind, "short": SHORT[kind], "K": K, "n_pairs": len(h), "n_targets": len(targets),
                   "n_rep_min": int(h.n_rep.min())}
            for c in bt:
                row[f"mean_{c}"] = h[c].mean()
                row[f"lo_{c}"], row[f"hi_{c}"] = np.quantile(bt[c], [0.025, 0.975])
            rows.append(row)
            if ref is not None and kind in state:
                rr = ref[(ref.setting == kind) & (ref.K == K)]
                # 検算は e12_wt.csv を作ったときの反復だけで作り直して比べる（HuBERT はそのとき反復 0〜1 の群があった）
                in_state = np.array([int(r) in state[kind].get(tt, []) for tt, r in zip(x.target, x.rep)], dtype=bool)
                xs = x[(x.K == K).to_numpy() & in_state]
                ps = xs.groupby(["target", "source"]).agg(wt=("uar", "mean"), m=("m", "mean"), t=("t", "mean"))
                if len(rr) and len(ps):
                    ctx.check(10, kind, f"E12 の gain_wt・gain_nat（K={K}．e12_state.json の反復）と e12_wt.csv の差の最大",
                              float(max(abs((ps.wt - ps.t).mean() - rr.gain_wt.iloc[0]),
                                        abs((ps.m - ps.t).mean() - rr.gain_nat.iloc[0]))), 1e-9,
                              f"{len(ps)} 組，{len(xs)} 行")
    if rows:
        ctx.write("e75_10_e12_wt.csv", pd.DataFrame(rows))
        ctx.write("e75_10_pairs.csv", pd.concat(prs, ignore_index=True))


# ---------------------------------------------------------------- 項目 11：q の当てはめの診断

def fit_grid(M: np.ndarray, D: np.ndarray, Ks: np.ndarray) -> dict:
    """M[..., 対, K] に 3 つの形を当てる（最小二乗．格子で探す）．先頭の次元はブートストラップの回。"""
    # F0：d = D/√(1 + q/K)（`explore_direction_bound.q_fit` と同じ形・同じ格子）
    f0 = D[None, :, None] / np.sqrt(1 + Q_GRID[:, None, None] / Ks[None, None, :])           # (q, 対, K)
    Mb = M[None] if M.ndim == 2 else M                                                       # (B, 対, K)
    sse0 = ((Mb[:, None, :, :] - f0[None]) ** 2).sum(axis=(-1, -2))                          # (B, q)
    i0 = sse0.argmin(-1)
    q, s0 = Q_GRID[i0], sse0[np.arange(len(i0)), i0]
    out = {"q": q[0] if M.ndim == 2 else q, "sse0": s0[0] if M.ndim == 2 else s0}
    i0 = i0[0] if M.ndim == 2 else i0
    if M.ndim == 2:
        # F1：d = D/√(1 + (q/K)^β)，F2：d = s·D/√(1 + q/K)
        f1 = D[None, None, :, None] / np.sqrt(1 + (Q_GRID[:, None, None, None] / Ks[None, None, None, :])
                                              ** BETA_GRID[None, :, None, None])
        sse1 = ((M[None, None] - f1) ** 2).sum(axis=(-1, -2))
        j1 = np.unravel_index(sse1.argmin(), sse1.shape)
        f2 = SCALE_GRID[None, :, None, None] * f0[:, None]
        sse2 = ((M[None, None] - f2) ** 2).sum(axis=(-1, -2))
        j2 = np.unravel_index(sse2.argmin(), sse2.shape)
        out.update(q1=Q_GRID[j1[0]], beta1=BETA_GRID[j1[1]], sse1=float(sse1[j1]),
                   q2=Q_GRID[j2[0]], scale2=SCALE_GRID[j2[1]], sse2=float(sse2[j2]),
                   fit0=f0[i0])
    return out


def item11(ctx: Ctx) -> None:
    Ks = np.array([1.0, 5.0, 10.0])
    Kstr = ["1", "5", "10"]
    tg_rows, res_rows, boot_rows = [], [], []
    ref_path = ctx.p("e28t9_targets")
    ref = read_csv(ctx, ref_path) if ref_path.exists() else None
    for kind in ctx.a.kinds:
        dpath = ROOT / f"{ctx.a.tgt_scores_prefix}{kind}"
        if not dpath.exists():
            continue
        own_all = ctx.e13(kind)
        for f in sorted(dpath.glob("*.npz")):
            ctx.note_input(f)
            t = f.stem
            z = np.load(f)
            reps = sorted(int(k.split("_r")[1]) for k in z.files if k.startswith("test_idx_r"))
            rec, cube = [], np.full((len(reps), 6, 3), np.nan)
            for ir, r in enumerate(reps):
                y = z[f"y_test_r{r}"]
                for ik, K in enumerate(Kstr):
                    for ip, ((a, b), (dd, oo)) in enumerate(zip(PAIRS, d_o(z[f"t_K{K}_r{r}"].astype(np.float64), y))):
                        rec.append({"rep": r, "K": K, "pair": f"{a}-{b}", "d": dd, "o": oo})
                        cube[ir, ip, ik] = dd
            if not np.isfinite(cube).all():
                ctx.gap(11, kind, f"d が NaN の升 {t}", 0, int((~np.isfinite(cube)).sum()))
                continue
            dT = pd.DataFrame(rec)
            own = own_all[own_all.target == t].groupby("pair_idx").own_d.mean()
            D = own.reindex(PAIR_IDX).to_numpy()
            q_ref = q_fit(dT, own)                                  # 今の台本の関数そのもの
            M = cube.mean(0)
            fg = fit_grid(M, D, Ks)
            ctx.check(11, kind, f"格子の当てはめ（まとめて解く版）と q_fit の差 {t}", abs(float(fg["q"]) - q_ref), 0.0)
            if ref is not None:
                rr = ref[(ref.kind == kind) & (ref.target == t)]
                if len(rr):
                    ctx.check(11, kind, f"q_dir と e28t9_targets.csv の差 {t}", abs(q_ref - float(rr.q_dir.iloc[0])), 1e-9)
            n = M.size
            aic = lambda sse, k: n * np.log(sse / n) + 2 * k  # noqa: E731
            # 反復を単位にしたブートストラップ（反復の間の依存は残る．S1）
            rng = common.rng_for(f"e75-qboot:{kind}:{t}", ctx.a.seed)
            idx = rng.integers(0, len(reps), size=(ctx.a.n_boot, len(reps)))
            Mb = cube[idx].mean(1)                                  # (B, 対, K)
            qb = fit_grid(Mb, D, Ks)["q"]
            boot_rows.append(pd.DataFrame({"kind": kind, "target": t, "b": np.arange(ctx.a.n_boot), "q": qb}))
            tg_rows.append({"kind": kind, "short": SHORT[kind], "target": t, "n_reps": len(reps), "q_dir": q_ref,
                            "q_boot_lo": float(np.quantile(qb, 0.025)), "q_boot_hi": float(np.quantile(qb, 0.975)),
                            "n_boot_at_grid_edge": int(((qb <= Q_GRID[0]) | (qb >= Q_GRID[-1])).sum()),
                            "rms_resid_F0": float(np.sqrt(fg["sse0"] / n)), "aic_F0": aic(fg["sse0"], 1),
                            "q_F1": fg["q1"], "beta_F1": fg["beta1"], "rms_resid_F1": float(np.sqrt(fg["sse1"] / n)),
                            "aic_F1": aic(fg["sse1"], 2), "q_F2": fg["q2"], "scale_F2": fg["scale2"],
                            "rms_resid_F2": float(np.sqrt(fg["sse2"] / n)), "aic_F2": aic(fg["sse2"], 2),
                            "beta_at_grid_edge": bool(fg["beta1"] in (BETA_GRID[0], BETA_GRID[-1])),
                            "scale_at_grid_edge": bool(fg["scale2"] in (SCALE_GRID[0], SCALE_GRID[-1])),
                            "own_d_mean": float(D.mean())})
            for ip, pidx in enumerate(PAIR_IDX):
                for ik, K in enumerate(Kstr):
                    res_rows.append({"kind": kind, "target": t, "pair": pidx, "K": K, "d_meas": M[ip, ik],
                                     "d_fit_F0": float(fg["fit0"][ip, ik]), "resid_F0": M[ip, ik] - float(fg["fit0"][ip, ik]),
                                     "own_d": D[ip], "d_meas_sd_over_reps": float(cube[:, ip, ik].std(ddof=1))})
    if not tg_rows:
        return
    T = pd.DataFrame(tg_rows)
    B = pd.concat(boot_rows, ignore_index=True)
    S = []
    for kind, h in T.groupby("kind"):
        rng = common.rng_for(f"e75-qtgtboot:{kind}", ctx.a.seed)
        tb = [np.median(rng.choice(h.q_dir.to_numpy(), size=len(h), replace=True)) for _ in range(ctx.a.n_boot)]
        rb = B[B.kind == kind].groupby("b").q.median()
        Rk = pd.DataFrame(res_rows)
        Rk = Rk[Rk.kind == kind]
        S.append({"kind": kind, "short": SHORT[kind], "n_targets": len(h), "median_q_dir": h.q_dir.median(),
                  "median_q_rep_boot_lo": float(rb.quantile(0.025)), "median_q_rep_boot_hi": float(rb.quantile(0.975)),
                  "median_q_target_boot_lo": float(np.quantile(tb, 0.025)),
                  "median_q_target_boot_hi": float(np.quantile(tb, 0.975)),
                  "min_q": h.q_dir.min(), "max_q": h.q_dir.max(),
                  "n_targets_F1_better_aic2": int((h.aic_F1 < h.aic_F0 - 2).sum()),
                  "n_targets_F2_better_aic2": int((h.aic_F2 < h.aic_F0 - 2).sum()),
                  "median_beta_F1": h.beta_F1.median(), "median_scale_F2": h.scale_F2.median(),
                  "median_rms_resid_F0": h.rms_resid_F0.median(),
                  **{f"mean_resid_F0_K{K}": float(Rk[Rk.K == K].resid_F0.mean()) for K in Kstr}})
    ctx.write("e75_11_targets.csv", T)
    ctx.write("e75_11_residuals.csv", pd.DataFrame(res_rows))
    ctx.write("e75_11_summary.csv", pd.DataFrame(S))
    ctx.write("e75_11_boot.csv", B)


# ---------------------------------------------------------------- 項目 12：λ の順位相関

def item12(ctx: Ctx) -> None:
    out, win = [], []
    for key, ks in (("stageb_main", MAIN4), ("stageb_text", TEXT_KINDS)):
        use = [k for k in ctx.a.kinds if k in ks]
        if not use:
            continue
        P = read_csv(ctx, ctx.p(key) / "e15b_pairs.csv", dtype={"K": str})
        P_all = P[P.kind.isin(use)]
        P = P_all[[not same_corpus(t, s) for t, s in zip(P_all.target, P_all.source)]]
        tests = ctx.p(key) / "e15b_tests.csv"
        if tests.exists():
            T = read_csv(ctx, tests)
            name = {"audio_wavlm": "W7", "audio_hubert": "Hu7", "video": "V", "fuse-audio-video": "AV"}
            for kind in use:
                if kind in name and (T.setting == name[kind]).any():
                    r, _, _ = spear(P_all[P_all.kind == kind].lam_obs, P_all[P_all.kind == kind].lam_pred_rep09)
                    ctx.check(12, kind, "K をまとめた順位相関と e15b_tests.csv の差",
                              abs(r - float(T[T.setting == name[kind]].spearman.iloc[0])), 1e-9)
        for kind in use:
            h = P[P.kind == kind]
            r, p, n = spear(h.lam_obs, h.lam_pred_rep09)
            out.append({"kind": kind, "short": SHORT[kind], "K": "pooled", "n": n, "rho": r, "p": p,
                        "mean_obs": h.lam_obs.mean(), "mean_pred": h.lam_pred_rep09.mean()})
            for K, hk in h.groupby("K"):
                r, p, n = spear(hk.lam_obs, hk.lam_pred_rep09)
                rows_t = []
                for t, ht in hk.groupby("target"):
                    if len(ht) >= 4:
                        rt, _, nt = spear(ht.lam_obs, ht.lam_pred_rep09)
                        rows_t.append(rt)
                        win.append({"kind": kind, "K": K, "target": t, "n_sources": nt, "rho": rt})
                rt = np.array([v for v in rows_t if np.isfinite(v)])
                out.append({"kind": kind, "short": SHORT[kind], "K": K, "n": n, "rho": r, "p": p,
                            "mean_obs": hk.lam_obs.mean(), "mean_pred": hk.lam_pred_rep09.mean(),
                            "n_targets_within": len(rt), "median_rho_within": float(np.median(rt)) if len(rt) else np.nan,
                            "min_rho_within": float(rt.min()) if len(rt) else np.nan,
                            "max_rho_within": float(rt.max()) if len(rt) else np.nan})
    ctx.write("e75_12_lambda.csv", pd.DataFrame(out))
    ctx.write("e75_12_within_target.csv", pd.DataFrame(win))


# ---------------------------------------------------------------- 項目 13：e をほかのコーパスで推す

def item13(ctx: Ctx) -> None:
    rows, win = [], []
    for kind in ctx.a.kinds:
        e = ctx.e13(kind)
        e = e[[not same_corpus(t, s) for t, s in zip(e.target, e.source)]].copy()
        e["eff"] = (e.d / e.own_d).clip(0.0, 0.999)                   # E28-9 と同じ e = d_s/D（切る範囲も同じ）
        E = e.groupby(["target", "source"]).eff.mean()
        for (t, s), v in E.items():
            others = [tp for (tp, sp) in E.index if sp == s and tp != t and not same_corpus(tp, t)]
            est = float(np.mean([E[(tp, s)] for tp in others])) if others else np.nan
            rows.append({"kind": kind, "target": t, "source": s, "e_measured": v, "e_pseudo": est,
                         "n_pseudo_targets": len(others)})
        R = pd.DataFrame([r for r in rows if r["kind"] == kind])
        for t, h in R.groupby("target"):
            if len(h) >= 4:
                r, _, n = spear(h.e_pseudo, h.e_measured)
                win.append({"kind": kind, "short": SHORT[kind], "target": t, "n_sources": n, "rho": r})
    W = pd.DataFrame(win)
    S = []
    for kind in ctx.a.kinds:
        h = W[W.kind == kind] if len(W) else W
        S.append({"kind": kind, "short": SHORT[kind], "n_targets": len(h),
                  "median_rho": float(h.rho.median()) if len(h) else np.nan,
                  "min_rho": float(h.rho.min()) if len(h) else np.nan, "max_rho": float(h.rho.max()) if len(h) else np.nan})
    ctx.write("e75_13_pairs.csv", pd.DataFrame(rows))
    ctx.write("e75_13_within_target.csv", W)
    ctx.write("e75_13_summary.csv", pd.DataFrame(S))


# ---------------------------------------------------------------- 本体

FUNCS = {1: item1, 2: item2, 3: item3, 4: item4, 5: item5, 6: item6, 7: item7, 8: item8, 9: item9,
         10: item10, 11: item11, 12: item12, 13: item13}


def main() -> int:
    ap = argparse.ArgumentParser(description="E75：筋 1〜3 の保存済みの結果からの再解析（CPU）")
    ap.add_argument("--out", default="results/e75", help="出力先（既定 results/e75）．ファイルがあれば止まる")
    ap.add_argument("--kinds", nargs="+", choices=KINDS_ALL, default=KINDS_ALL)
    ap.add_argument("--items", nargs="+", type=int, choices=ITEMS, default=ITEMS)
    ap.add_argument("--n-perm", type=int, default=2000, help="項目 3 の並べ替えの回数")
    ap.add_argument("--n-boot", type=int, default=1000, help="項目 11 のブートストラップの回数")
    ap.add_argument("--n-boot-targets", type=int, default=2000, help="項目 10 のターゲットを単位にしたブートストラップの回数")
    ap.add_argument("--seed", type=int, default=0)
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", default=v)
    a = ap.parse_args()
    a.kinds = [k for k in KINDS_ALL if k in a.kinds]
    a.items = sorted(set(a.items))
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    a.out = str(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} にすでにファイルがある．既存の結果を上書きしないので，空のフォルダか新しい場所を指定する")
    out.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(a)
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    prov = {"script": "exp/reanalysis_e75.py", "argv": sys.argv[1:], "started": started, **git_state(),
            "settings": {**{k: getattr(a, k) for k in ("kinds", "items", "n_perm", "n_boot", "n_boot_targets", "seed")},
                         **{k: getattr(a, k) for k in DEFAULTS},
                         "den_small_points": DEN_SMALL, "loc_min_points": LOC_MIN, "q_grid": "logspace(-1,3,400)",
                         "beta_grid": "linspace(0.25,3,56)", "scale_grid": "linspace(0.5,2,61)",
                         "interval_main": "IEMOCAP がターゲットの組はテストの話者の集合を単位にした t（A6），ほかは補正した resampled t（rho = n_test/n_pool）",
                         "holm_family": "種類 × K の中の，同じコーパスの組を除いた組",
                         "reps_stage_b": "0-9", "gh_nodes": len(GH_X)},
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                         "scipy": scipy.__version__},
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "elapsed_sec": {}}
    t_all = time.time()
    for it in a.items:
        t0 = time.time()
        print(f"項目 {it} を計算する", flush=True)
        FUNCS[it](ctx)
        prov["elapsed_sec"][str(it)] = round(time.time() - t0, 1)
        print(f"  項目 {it}：{prov['elapsed_sec'][str(it)]} 秒", flush=True)
    prov["elapsed_sec"]["total"] = round(time.time() - t_all, 1)
    prov["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    prov["inputs"] = ctx.inputs
    prov["written"] = ctx.cache.get("written", [])
    prov["e13_computed_on_the_fly"] = ctx.cache.get("e13_computed", [])
    G = pd.DataFrame(ctx.gaps, columns=["item", "kind", "what", "expected", "found", "detail"])
    Ck = pd.DataFrame(ctx.checks, columns=["item", "kind", "what", "value", "tol", "ok", "detail"])
    ctx.write("gaps.csv", G)
    ctx.write("checks.csv", Ck)
    prov["n_gaps"], prov["n_checks"], prov["n_checks_failed"] = len(G), len(Ck), int((~Ck.ok).sum()) if len(Ck) else 0
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1, default=str))
    print(f"欠け {len(G)} 件，検算 {len(Ck)} 件（外れ {prov['n_checks_failed']} 件），計 {prov['elapsed_sec']['total']} 秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
