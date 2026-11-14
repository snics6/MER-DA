"""DAI-20 計算の後の点検（集計の表と，保存からの作り直し）．results/dai20_eval は読むだけ．
UAR と差の値は画面に出さない（数・一致・ずれの大きさだけ）．"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/snics/MER-DA")
sys.path.insert(0, str(ROOT / "DAI" / "exp"))
import dai1_stats as ST  # noqa: E402
import dai16_probe as P16  # noqa: E402

R = ROOT / "results" / "dai20_eval"
U, SM = R / "units", R / "summary"
CORP = ("AFEW", "IEMOCAP", "MOSEI", "MELD")
KS = (5, 10, 25)
GRID = P16.GRID
fz = json.load(open(ROOT / "DAI" / "frozen" / "dai20_frozen.json"))["choice"]

lv = pd.read_csv(SM / "levels.csv")
ef = pd.read_csv(SM / "effects.csv")
pe = pd.read_csv(SM / "per_emotion.csv")
oth = pd.read_csv(SM / "others.csv")
gl = pd.read_csv(SM / "grid_levels.csv")
eg = pd.read_csv(SM / "est_grid.csv")
wt = pd.read_csv(SM / "weights.csv")
cf = pd.read_csv(SM / "confusion.csv")
cfit = pd.read_csv(SM / "competitor_fit.csv")
ass = pd.read_csv(SM / "assumptions.csv")
src = pd.read_csv(SM / "source_tables.csv")
tabs = dict(levels=(lv, ["c", "K", "version", "seeds"]), effects=(ef, ["a", "b", "c", "K", "seeds"]), per_emotion=(pe, ["c", "K", "version", "emotion"]),
            grid_levels=(gl, ["c", "K", "grid"]), est_grid=(eg, ["c", "K", "grid"]), weights=(wt, ["c", "K", "mod", "emotion"]),
            confusion=(cf, ["c", "K", "version", "true", "pred"]), competitor_fit=(cfit, ["c", "K", "version"]), assumptions=(ass, ["c", "K"]),
            source_tables=(src, ["T", "c", "mod"]), others=(oth, ["group", "c", "K", "method"]))
for n, (d, k) in tabs.items():
    allnan = [c for c in d.columns if d[c].isna().all()]
    num = d.select_dtypes(include=[np.number])
    print(f"{n}: 行 {len(d)}，鍵の重複 {int(d.duplicated(k).sum())}，無限大 {int(np.isinf(num.to_numpy()).sum())}，全部 NaN の列 {allnan}")

# levels の中身
print("levels：(c,K) ごとの版の数", lv.groupby(["c", "K"]).version.nunique().unique().tolist(), "seeds", lv.seeds.value_counts().to_dict())
print("levels：NaN のある列", {c: int(lv[c].isna().sum()) for c in lv.columns if lv[c].isna().any()})
print("levels：n_fr の値（seeds=all）", lv[lv.seeds == "all"].n_fr.value_counts().to_dict(), "（0-2）", lv[lv.seeds == "0-2"].n_fr.value_counts().to_dict())
print("levels：role", lv.role.value_counts().to_dict())
bad = []
for c in CORP:
    for K in KS:
        g = int(fz[c][str(K)]["grid"])
        r = lv[(lv.c == c) & (lv.K == K) & (lv.version == "prop")]
        if not (r.grid == g).all():
            bad.append((c, K))
print("levels の prop の grid が固定のファイルと違う (c,K)", bad)
print("levels の uar が 0〜100 の外", int(((lv.uar < 0) | (lv.uar > 100)).sum()), "lo<=uar<=hi でない", int(((lv.lo > lv.uar + 1e-9) | (lv.hi < lv.uar - 1e-9)).sum()))
# effects の中身
pairs = ef[["a", "b"]].drop_duplicates()
print("effects：対の数", len(pairs), "対ごとの行の数", ef.groupby(["a", "b"]).size().value_counts().to_dict(), "seeds", ef.seeds.value_counts().to_dict())
print("effects：NaN のある列", {c: int(ef[c].isna().sum()) for c in ef.columns if ef[c].isna().any()})
print("effects：method", ef.method.value_counts().to_dict(), "mark", ef.mark.fillna("").value_counts().to_dict() and "(数は出さない)")
print("effects：lo<=est<=hi でない", int(((ef.lo > ef.est + 1e-9) | (ef.hi < ef.est - 1e-9)).sum()), "v_seed_all < v_seed", int((ef.v_seed_all < ef.v_seed - 1e-12).sum()))
# 台本の対の一覧と合うか
A_VERS = ("prod", "psel", "avg", "conf", "M_F2_K", "M_F6_K", "cf_D_l05", "tcal", "temp", "tempb", "wf10", "wf100", "wf1000")
ABL = ("prop_noprof", "prop_noprof_sel", "prop_noint", "prop_n0zero", "prop_home", "prop_levelfree")
B_VERS = ("b_kd", "b_agree", "b_mix", "b_cotrain", "b_coreg")
REF, ORC = ("M_F6_ext", "M_F8"), ("M_F6_orc",)
con = ([("prop", b) for b in A_VERS + ABL + ("none_AVT", "joint_AVT") + B_VERS + REF]
       + [(a, "prod") for a in A_VERS[1:] + ("prop_noprof", "prop_noprof_sel", "prop_loto", "oracle_grid") + ORC + REF]
       + [("prop_loto", "prop"), ("oracle_grid", "prop"), ("prop", "prop@raw"), ("prop_levelfree", "prod"), ("prop_noint", "prop_noprof")]
       + [("prop@raw", f"{b}@raw") for b in A_VERS] + [("prop", f"{b}@raw") for b in A_VERS[1:]])
print("effects：台本の対の一覧", len(con), "表の対と同じ集合", set(map(tuple, pairs.to_numpy())) == set(con))
# 設計 5 節の部品を外した版の差（提案 − 部品を外した版，2 つの順序）
need = [("prop", v) for v in ABL] + [("prop_levelfree", "prod"), ("prop_noint", "prop_noprof"), ("prop_noprof", "prod"), ("prop_loto", "prop"), ("oracle_grid", "prop"),
                                     ("prop_loto", "prod"), ("oracle_grid", "prod"), ("M_F6_orc", "prod"), ("prop", "prop@raw")]
print("設計 5 節の差で表に無いもの", [x for x in need if x not in con])
# per_emotion
print("per_emotion：(c,K) ごとの版の数", pe.groupby(["c", "K"]).version.nunique().unique().tolist(), "NaN の recall", int(pe.recall.isna().sum()),
      "n_fr=0", int((pe.n_fr == 0).sum()))
# others
B = oth[oth.group == "B"]
C = oth[oth.group == "C"]
print("others：B", len(B), "uar_here の NaN", int(B.uar_here.isna().sum()), "|uar-uar_here| 最大", float((B.uar - B.uar_here).abs().max()), "B の (c,K,method) の数",
      B.groupby(["c", "K"]).size().unique().tolist())
print("others：C", len(C), "手法", sorted(C.method.unique()), "(c,K) ごとの行", C.groupby(["c", "K"]).size().unique().tolist(), "n_units", C.n_units.value_counts().to_dict())
print("others：C の gradient=True の手法", sorted(C[C.gradient].method.unique()), "seeds", C.seeds.value_counts().to_dict())
# grid_levels を単位の csv から作り直す（値は出さない．一致だけ）
U_ = pd.concat([pd.read_csv(p, low_memory=False, usecols=["T", "K", "version", "uar_adj"]) for p in sorted(U.glob("*.csv"))])
g = U_[U_.version.astype(str).str.fullmatch(r"M_F6_mix\d+")].copy()
g["grid"] = g.version.str.replace("M_F6_mix", "").astype(int)
m = g.groupby(["T", "K", "grid"]).uar_adj.mean()
dev, flag_bad = 0.0, []
for r in gl.itertuples():
    dev = max(dev, abs(m[(r.c, r.K, r.grid)] - r.uar_adj))
    oth_m = np.mean([m[(c2, r.K, r.grid)] for c2 in CORP if c2 != r.c])
    dev = max(dev, abs(oth_m - r.uar_adj_others))
for c in CORP:
    for K in KS:
        s = gl[(gl.c == c) & (gl.K == K)].set_index("grid")
        own = np.array([m[(c, K, i)] for i in range(16)])
        oth_ = np.array([np.mean([m[(c2, K, i)] for c2 in CORP if c2 != c]) for i in range(16)])
        exp = dict(frozen=int(fz[c][str(K)]["grid"]), loto=int(np.argmax(oth_)), oracle=int(np.argmax(own)))
        for col, i in exp.items():
            if list(s.index[s[col]]) != [i]:
                flag_bad.append((c, K, col))
print("grid_levels：作り直しとのずれの最大", dev, "印（frozen・loto・oracle）の食い違い", flag_bad)
print("est_grid：frozen の印が (c,K) ごとに 1 つ", eg.groupby(["c", "K"]).frozen.sum().unique().tolist())
