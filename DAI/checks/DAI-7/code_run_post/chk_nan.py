"""DAI-7 計算の後の点検 (2) の続き：NaN の出どころ．値は出さず，数と場所だけを出す．"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("/home/snics/MER-DA/results/dai7_probe_v2")
units = sorted(p.name[: -len("_units.csv")] for p in (OUT / "rows").glob("*_units.csv"))
U = pd.concat([pd.read_csv(OUT / "rows" / f"{u}_units.csv") for u in units], ignore_index=True)
BOOT = ("none", "joint", "joint_a1", "oracle", "joint_a1_rho0.0625", "joint_a1_rho1.0", "joint_a1_rho4.0", "tt_ownret")
U["cls"] = np.where(U.set.str.startswith("none_tw"), "tw", np.where(U.set.isin(BOOT), "boot", "other"))

print("== units：列の種類ごとの NaN（期待の NaN を除いた残り）")
# 期待：(a) none_tw の行は D・D_raw・D_null・n_emotions・phi・n_emotions_phi・d/d0/n_e* 以外がすべて NaN
tw_cols = {"T", "c", "fold", "seed", "K", "set", "pair", "teacher_version", "D", "D_raw", "D_null", "n_emotions", "phi", "n_emotions_phi",
           *[f"{q}_e{e}" for q in ("d", "d0", "n") for e in range(4)], "commit", "dirty", "cls"}
tw = U[U.cls == "tw"]
print("none_tw の行で，作る列に NaN がある数（d_e*・d0_e* と phi を除く）",
      int(tw[[x for x in tw_cols if x not in {f"{q}_e{e}" for q in ("d", "d0") for e in range(4)} | {"phi"}]].isna().sum().sum()))
print("none_tw の行で，作らない列が NaN でない数", int(tw[[x for x in U.columns if x not in tw_cols]].notna().sum().sum()))
# (b) d_e・d0_e は n_e < 15 の感情で NaN（そのときだけ）
bad = 0
for e in range(4):
    small = U[f"n_e{e}"] < 15
    bad += int((U.loc[small, f"d_e{e}"].notna()).sum() + (U.loc[~small, f"d_e{e}"].isna()).sum())
    bad += int((U.loc[small, f"d0_e{e}"].notna()).sum() + (U.loc[~small, f"d0_e{e}"].isna()).sum())
print("d_e・d0_e の NaN が「感情の本数 < 15」と一致しない数", bad)
print("感情の本数 < 15 の (単位の行, 感情) の数", int(sum((U[f"n_e{e}"] < 15).sum() for e in range(4))),
      "そのコーパス", sorted(U.loc[(U[[f"n_e{e}" for e in range(4)]] < 15).any(axis=1), "c"].unique().tolist()))
print("n_emotions が 0 の行", int((U.n_emotions == 0).sum()), "D が NaN の行", int(U.D.isna().sum()))
print("phi が NaN の行", int(U.phi.isna().sum()), "そのうち n_emotions_phi が 0", int(((U.phi.isna()) & (U.n_emotions_phi == 0)).sum()))
# (c) n_dropped_scen は IEMOCAP だけ
print("n_dropped_scen：IEMOCAP 以外で NaN でない（tw を除く）", int(U[(U.c != "IEMOCAP") & (U.cls != "tw")].n_dropped_scen.notna().sum()),
      "IEMOCAP で NaN（tw を除く）", int(U[(U.c == "IEMOCAP") & (U.cls != "tw")].n_dropped_scen.isna().sum()))
# (d) 変種の列：区間を作る組は自分のコーパスの変種，ほかの組は cent だけ
VAR_OF = {"AFEW": ["cent", "keep", "sub_asr_ok"], "IEMOCAP": ["cent", "keep", "scen", "sub_marker", "sub_nomarker"],
          "MELD": ["cent", "keep", "sub_face_single"], "MOSEI": ["cent", "keep", "chan"]}
allv = sorted({v for vs in VAR_OF.values() for v in vs})
bad_present, bad_absent = 0, 0
for c, vs in VAR_OF.items():
    for cls in ("boot", "other"):
        sub = U[(U.c == c) & (U.cls == cls)]
        should = vs if cls == "boot" else ["cent"]
        for v in allv:
            cols = [f"D_{v}", f"Dnull_{v}"]
            if v in should:
                bad_absent += int(sub[cols].isna().all(axis=1).sum())
            else:
                bad_present += int(sub[cols].notna().sum().sum())
            vcols = [x for x in U.columns if x.startswith(f"v_{v}_")]
            if cls == "boot" and v in should:
                pass
            else:
                bad_present += int(sub[vcols].notna().sum().sum())
print("変種：あるべき (行, 変種) で D と Dnull が両方 NaN の数", bad_absent, "あるべきでない場所に値がある数", bad_present)
# 変種の D が NaN の行（感情が全部 15 本未満）
for c, vs in VAR_OF.items():
    sub = U[(U.c == c) & (U.cls == "boot")]
    for v in vs:
        ne = sum((sub[f"v_{v}_n_e{e}"] >= 15).astype(int) for e in range(4))
        nan_d = sub[f"D_{v}"].isna()
        print(f"{c} {v}：D が NaN の行 {int(nan_d.sum())}，使える感情 0 の行 {int((ne == 0).sum())}，一致しない {int((nan_d != (ne == 0)).sum())}，"
              f"使える感情の数の種類 {sorted(ne.unique().tolist())}")
# D_cent（区間を作らない組）
sub = U[U.cls == "other"]
print("区間を作らない組で D_cent が NaN の行", int(sub.D_cent.isna().sum()))
print("D_pool が NaN の行（tw を除く）", int(U[U.cls != "tw"].D_pool.isna().sum()), "D_partial_t が NaN の行", int(U[U.cls != "tw"].D_partial_t.isna().sum()))

print("== cells の NaN（列ごと，ゼロでない列だけ）")
cells = pd.read_csv(OUT / "cells.csv")
cells["cls"] = np.where(cells.set.str.startswith("none_tw"), "tw", np.where(cells.set.isin(BOOT), "boot", "other"))
nn = cells.isna().sum()
for col in nn[nn > 0].index:
    g = cells[cells[col].isna()].groupby(["c", "cls"]).size().to_dict()
    print(col, int(nn[col]), g)

print("== diffs の NaN（列ごと）")
Df = pd.read_csv(OUT / "diffs.csv")
nn = Df.isna().sum()
print({k: int(v) for k, v in nn[nn > 0].items()})
print("emotion が NaN でない行", int(Df.emotion.notna().sum()), "n_test_5folds が NaN でない行", int(Df.n_test_5folds.notna().sum()))
print("jk 以外で n_eff・df が NaN でない行", int(Df[Df.variant != "jk"][["n_eff", "df"]].notna().sum().sum()),
      "jk で n_eff・df が NaN の行", int(Df[Df.variant == "jk"][["n_eff", "df"]].isna().sum().sum()))
Gr = Df.quantity.str.match(r"^G_")
print("G_ の行以外で G_* の列に値がある数", int(Df.loc[~Gr, ["G_A", "G_V", "G_T", "G_prod", "G_full", "G_cal", "G_max"]].notna().sum().sum()),
      "G_ の行で G_* が NaN の数", int(Df.loc[Gr, ["G_A", "G_V", "G_T", "G_prod", "G_full", "G_cal", "G_max"]].isna().sum().sum()))
print("G_joint の行で lo・hi が NaN（区間を作らないと設計した行）", int(Df[Df.quantity == "G_joint"][["lo", "hi"]].isna().all(axis=1).sum()),
      "G_joint の行の数", int((Df.quantity == "G_joint").sum()))
print("dDt の行で lo が NaN", int(Df[Df.quantity == "dDt_joint_a1_none"].lo.isna().sum()), "の行の数", int((Df.quantity == "dDt_joint_a1_none").sum()))
# est が NaN の行
en = Df[Df.est.isna()]
print("est が NaN の行", len(en), en.groupby(["c", "variant", "quantity"]).size().to_dict() if len(en) else {})
# lo が NaN で区間を作るはずの行
want_ci = ~Df.quantity.isin(["G_joint", "dDt_joint_a1_none"])
ln = Df[want_ci & Df.lo.isna()]
print("区間を作るはずの行で lo が NaN", len(ln), ln.groupby(["c", "variant", "quantity"]).size().to_dict() if len(ln) else {})
print("  そのうち n_boot_valid が 0", int((ln.n_boot_valid == 0).sum()))
print("d_null_diff が NaN の行の量の種類", sorted(Df[Df.d_null_diff.isna()].quantity.unique().tolist())[:40])
dq = Df.quantity.str.startswith("dD_") & (Df.quantity != "dDt_joint_a1_none")
print("ΔD の行で d_null_diff が NaN（感情ごとの行を除く）", int(Df[dq & Df.emotion.isna() & Df.d_null_diff.isna()].shape[0]))
print("区間のある行の n_boot_valid の分布（variant ごとの最小）", Df[Df.lo.notna()].groupby("variant").n_boot_valid.min().astype(int).to_dict())
lowv = Df[Df.lo.notna() & (Df.variant != "jk") & (Df.n_boot_valid < 1000)]
print("jk 以外で有効な回が 1000 未満の行", len(lowv), lowv.groupby(["c", "variant"]).size().to_dict() if len(lowv) else {})
print("jk の行の n_boot_valid（= 塊の数のはず）", Df[Df.variant == "jk"].groupby("c").n_boot_valid.unique().apply(list).to_dict())
share = Df[Df.quantity.str.startswith("share_dep")]
print("share_dep の行で n_boot_valid が 1000 未満", int((share.n_boot_valid < 1000).sum()))

print("== criteria の NaN")
C = pd.read_csv(OUT / "criteria.csv")
h4s = [f"{t}_{q}" for t in ("H4", "H4_rho1") for q in ("largest_by_K", "share_by_K")]
other = [x for x in C.columns if x not in h4s]
nn = C[other].isna().sum()
print("H4 の K ごとの文字列の列を除いた NaN", {k: int(v) for k, v in nn[nn > 0].items()})
print("chan* の列が MOSEI 以外で NaN の数", int(C.loc[C.c != "MOSEI", [x for x in C.columns if x.startswith("chan")]].isna().sum().sum()),
      "MOSEI で NaN", int(C.loc[C.c == "MOSEI", [x for x in C.columns if x.startswith("chan")]].isna().sum().sum()))
print("jk_* の列が MOSEI 以外で NaN の数", int(C.loc[C.c != "MOSEI", [x for x in C.columns if x.startswith("jk_")]].isna().sum().sum()),
      "MOSEI で NaN", int(C.loc[C.c == "MOSEI", [x for x in C.columns if x.startswith("jk_")]].isna().sum().sum()))
# H4 の K ごとの文字列：前提 (iv) を満たす K が 0 のときだけ空（NaN として読まれる）になるはず．一致しない数だけを出す
mis = 0
for tag in ("H4", "H4_rho1"):
    for col in (f"{tag}_largest_by_K", f"{tag}_share_by_K"):
        mis += int((C[col].isna() != (C[f"{tag}_n_K_iv"] == 0)).sum())
print("H4 の K ごとの文字列の NaN が「前提 (iv) の K が 0」と一致しない数", mis, "（H4 の列の NaN の合計は書かない）")
print("H4_wording・H1_wording の NaN", int(C[["H4_wording", "H1_wording"]].isna().sum().sum()))
