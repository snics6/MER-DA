"""関門 1（gate-code）の検算．results/ は読むだけ．出力は標準出力だけ。"""
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
ROOT = Path("/home/snics/MER-DA")
sys.path[:0] = [str(ROOT / "exp"), str(ROOT / "measure"), str(ROOT / "features")]
from data import load_any, groups_for, GROUP_SPEC
from summarize_map import load as load_map
OUT = ROOT / "checks/E75/code_run/runA"
KINDS = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video",
         "text_roberta", "fuse-audio-text", "fuse-video-text", "fuse-audio-video-text"]
P1 = pd.read_csv(OUT / "e75_1_pairs.csv", dtype={"K": str})
P4 = pd.read_csv(OUT / "e75_4_pairs.csv", dtype={"K": str})

print("== 1. 地図の重複行（target,source,method,K,rep）")
for pre in ("results/fixc_map/", "results/map_"):
    for k in KINDS:
        d = ROOT / (pre + k) if pre.endswith("_") else ROOT / pre / k
        if not d.exists():
            print(pre, k, "無い"); continue
        df = load_map(d)
        dup = df.duplicated(["target", "source", "method", "K", "rep"]).sum()
        nrep = df[df.method == "merged"].groupby(["target", "source", "K"]).rep.nunique()
        print(f"  {pre}{k}: 重複 {dup}，合併の反復の数 {sorted(nrep.unique())}")

print("== 2. 項目 4：cv_map の得の平均と results/map_<kind>/summary/pairs.csv の diff_mean")
for k in KINDS:
    sp = ROOT / f"results/map_{k}/summary/pairs.csv"
    ref = pd.read_csv(sp, dtype={"K": str}).set_index(["target", "source", "K"]).diff_mean
    mine = P4[(P4.kind == k) & (P4.rule == "cv_map")].set_index(["target", "source", "K"])["mean"]
    j = mine.to_frame().join(ref, how="inner")
    print(f"  {k}: {len(j)} 行，差の最大 {np.abs(j['mean'] - j.diff_mean).max():.2e}")

print("== 3. 項目 4：map_rule が項目 1 の平均と同じか．K=1 で C1 が map_rule と同じか（退化の検算）")
for k in KINDS:
    a = P4[(P4.kind == k) & (P4.rule == "map_rule")].set_index(["target", "source", "K"])["mean"]
    b = P1[(P1.kind == k)].set_index(["target", "source", "K"])["mean"]
    c1 = P4[(P4.kind == k) & (P4.rule == "C1") & (P4.K == "1")].set_index(["target", "source"])["mean"]
    m1 = P4[(P4.kind == k) & (P4.rule == "map_rule") & (P4.K == "1")].set_index(["target", "source"])["mean"]
    df = load_map(ROOT / "results/fixc_map" / k)
    t1 = df[(df.method == "tgt_only") & (df.K == "1")]
    print(f"  {k}: map_rule−項目1 最大 {np.abs(a - b.reindex(a.index)).max():.1e}；K=1 の tgt の C の値 {sorted(t1.C.unique())}；"
          f"C1−map_rule（K=1）最大 {np.abs(c1 - m1).max():.1e}")

print("== 4. 項目 1：補正した区間と，テストの組の数を台本の外で作り直す")
from split import speaker_folds
for k in ("audio_wavlm", "text_roberta"):
    df = load_map(ROOT / "results/fixc_map" / k)
    for t in ("MELD", "IEMOCAP-script", "IEMOCAP-impro"):
        tg = df[(df.method == "tgt_only") & (df.target == t)]
        rho = (tg[tg.K == "all"].n_test / tg[tg.K == "all"].n_tgt).mean()
        s = sorted(set(df[(df.target == t) & (df.method == "merged")].source) - {t})[0]
        mm = df[(df.method == "merged") & (df.target == t) & (df.source == s) & (df.K == "5")].sort_values("rep")
        tt = tg[tg.K == "5"].set_index("rep").uar
        x = mm.uar.to_numpy() - tt.loc[mm.rep].to_numpy()
        J = len(x); se = x.std(ddof=1) * np.sqrt(1 / J + rho); lo = x.mean() - stats.t.ppf(.975, J - 1) * se
        row = P1[(P1.kind == k) & (P1.target == t) & (P1.source == s) & (P1.K == "5")].iloc[0]
        meta, _ = load_any(t, k)
        sp = meta.speaker_group.astype(str).reset_index(drop=True)
        sets = {r: frozenset(sp[(speaker_folds(sp, 5, r) == r % 5).to_numpy()]) for r in range(30)}
        print(f"  {k} {t}←{s} K=5: rho {rho:.3f}（台本 {row.rho:.3f}），補正の下端の差 {abs(lo - row.corr_lo):.1e}，"
              f"異なるテストの集合 {len(set(sets.values()))}（台本 {row.n_clusters}），群の数 {sp.nunique()}，NaN の群 {meta.speaker_group.isna().sum()}")

print("== 5. 項目 7・11：ターゲットのみの d の平均と E28-9 の dT_K*")
e28 = pd.read_csv(ROOT / "results/explore_relation/snapshots/20261011-0139_e28t9/e28t9_targets.csv")
R7 = pd.read_csv(OUT / "e75_7_rows.csv", dtype={"K": str})
g = R7[R7.clf == "tgt_only"].groupby(["kind", "target", "K"]).d.mean().unstack("K")
j = e28.set_index(["kind", "target"]).join(g, how="inner")
print("  行", len(j), "差の最大", max(np.abs(j[f"dT_K{K}"] - j[K]).max() for K in ("1", "5", "10")))
print("== 6. 項目 13：e_measured と E28-9 の e（K に依らない列）")
e28p = pd.read_csv(ROOT / "results/explore_relation/snapshots/20261011-0139_e28t9/e28t9_pairs.csv")
ep = e28p.groupby(["kind", "target", "source"]).e.agg(["min", "max"])
R13 = pd.read_csv(OUT / "e75_13_pairs.csv").set_index(["kind", "target", "source"])
j = R13.join(ep, how="inner")
print("  行", len(j), "e の K による違いの最大", (j["max"] - j["min"]).max(), "差の最大", np.abs(j.e_measured - j["min"]).max())

print("== 7. tgt_scores・transfer_scores の y_test が発話一覧のラベルと同じか（取り決め 1）")
for k in ("audio_wavlm", "video"):
    for pre in ("tgt_scores_", "transfer_scores_"):
        for f in sorted((ROOT / f"results/{pre}{k}").glob("*.npz"))[:3]:
            z = np.load(f); meta, _ = load_any(f.stem, k)
            bad = 0
            for r in range(10):
                idx = z[f"test_idx_r{r}"]
                bad += int((meta.label_4.to_numpy()[idx] != z[f"y_test_r{r}"]).sum())
            print(f"  {pre}{k} {f.stem}: 食い違い {bad}（test_idx の型 {z['test_idx_r0'].dtype}）")

print("== 8. 項目 10：α を選んだ合併の C と地図の合併の C")
for k in ("audio_wavlm", "video"):
    mech = pd.concat([pd.read_csv(f, comment="#", dtype={"K": str, "source": str}) for f in sorted((ROOT / f"results/mech_{k}").glob("*.csv"))])
    w = mech[(mech.op == "wt") & (mech.method == "merged")]
    mp = load_map(ROOT / "results/fixc_map" / k)
    m = mp[mp.method == "merged"][["target", "source", "K", "rep", "C", "uar"]]
    x = w.merge(m, on=["target", "source", "K", "rep"], suffixes=("_wt", "_map"))
    nat = x[x.alpha.astype(str) == "nat"]
    print(f"  {k}: 行 {len(x)}，C が違う行 {int((x.C_wt != x.C_map).sum())}，α=nat の行 {len(nat)}，"
          f"そのうち UAR が地図の合併と違う行 {int((np.abs(nat.uar_wt - nat.uar_map) > 1e-9).sum())}")
    mm = mp[mp.method == "merged"]
    print(f"  {k}: 地図の合併の test_uar_C* の NaN の数 {int(mm[[c for c in mm if c.startswith('test_uar_C')]].isna().sum().sum())}／{len(mm)*4}")
