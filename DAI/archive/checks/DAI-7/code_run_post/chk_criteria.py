"""DAI-7 計算の後の点検 (5)：criteria.csv を設計 4 節の規則から作り直して突き合わせる．台本の関数は使わない．
値・判定・文の中身と長さは出さない．出すのは列ごとの一致・不一致の数だけである．"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as SS

OUT = Path("/home/snics/MER-DA/results/dai7_probe_v2")
KR = (1, 5, 10, 25)
RHO_SETS = ("joint_a1_rho0.0625", "joint_a1_rho0.25", "joint_a1_rho1.0", "joint_a1_rho4.0")
L_SETS = ("none", "joint", "joint_a1", "joint_a1_rho1.0")
Df = pd.read_csv(OUT / "diffs.csv")
cells = pd.read_csv(OUT / "cells.csv")
ac = pd.read_csv(OUT / "uar_cells.csv")
TW = pd.read_csv(OUT / "tether_weakening_units.csv")
C = pd.read_csv(OUT / "criteria.csv").set_index("c")
PART = {"Scal": "単独の分類器の較正と感情の事前の得（判別分析が none にも与える分）",
        "S1b": "較正をそろえた単独の分類器の得が完全な結合器で足し合わさらない分（情報の重なりか UAR の天井．この 2 つは分けない）",
        "dLdep": "依存を無視した損の増加", "dLcal": "3 つの点数の結合の較正と感情の事前の分の増加"}


def rows(c, q, pair="all", var="main"):
    r = Df[(Df.c == c) & (Df.quantity == q) & (Df.pair == pair) & (Df.variant == var) & Df.emotion.isna()]
    r = r[r.K.isin(KR)].set_index("K").sort_index()
    assert list(r.index) == list(KR), (c, q, pair, var)
    return r


def good_K(c, q, floor, null, pair="all", var="main"):
    r = rows(c, q, pair, var)
    out = []
    for K in KR:
        x = r.loc[K]
        ok = x.lo > 0 and x.est >= floor
        if null:
            ok = ok and abs(x.d_null_diff) < 0.1 * abs(x.est)
        if ok:
            out.append(K)
    return out


R = {}
for c in sorted(Df.c.unique()):
    r = {}
    for nm, q, fl, null, pair in (("H1", "dD_joint_a1_none", 0.02, True, "all"), ("H1_joint", "dD_joint_none", 0.02, True, "all"),
                                  ("H1p", "dD_joint_a1_oracle", 0.02, True, "all"), ("H1_AV", "dD_joint_a1_none", 0.02, True, "AV"),
                                  ("phi", "dphi_joint_a1_none", 0.02, False, "all"), ("share", "dD_rho1_ownret", 0.02, True, "all"),
                                  ("phi_share", "dphi_rho1_ownret", 0.02, False, "all"), ("H3", "dLdep_joint_a1_none", 0.5, False, "all"),
                                  ("H3_joint", "dLdep_joint_none", 0.5, False, "all"), ("H2_size", "dD_rho_hi_lo", 0.02, False, "all")):
        d = rows(c, q, pair)
        gk = good_K(c, q, fl, null, pair)
        r[f"{nm}_n_lo_pos"] = int(sum(d.loc[K, "lo"] > 0 for K in KR))
        r[f"{nm}_n_lo_pos_floor"] = len(gk)
        r[f"{nm}_n_hi_neg"] = int(sum(d.loc[K, "hi"] < 0 for K in KR))
        r[nm] = len(gk) >= 3
    sz = good_K(c, "dD_rho_hi_lo", 0.02, False)
    nH2 = 0
    for K in KR:
        v = [cells[(cells.c == c) & (cells.K == K) & (cells.set == s) & (cells.pair == "all")].D.iloc[0] for s in RHO_SETS]
        rho = SS.spearmanr(v, [0, 1, 2, 3]).statistic
        r[f"H2_spearman_K{K}"] = float(rho)
        nH2 += bool(rho >= 0.8 - 1e-9 and K in sz)
    r["H2"] = nH2 >= 3
    h1K, shK = good_K(c, "dD_joint_a1_none", 0.02, True), good_K(c, "dD_rho1_ownret", 0.02, True)
    for q, Ks, tag in (("dD_joint_a1_none", h1K, ""), ("dD_rho1_ownret", shK, "_share")):
        m = rows(c, q)
        if c == "MOSEI":
            ch = rows(c, q, var="chan")
            r[f"chan{tag}_n_K_robust"] = int(sum(ch.loc[K, "lo"] > 0 and ch.loc[K, "est"] >= 0.5 * m.loc[K, "est"] for K in Ks))
        else:
            r[f"chan{tag}_n_K_robust"] = np.nan
        ce, kp = rows(c, q, var="cent"), rows(c, q, var="keep")
        r[f"cent{tag}_n_K_small"] = int(sum(ce.loc[K, "est"] < 0.5 * kp.loc[K, "est"] for K in Ks))
    tw = TW[(TW.c == c) & TW.K.isin(KR)]
    nr1, nall = int((tw.rho_joint_a1 != 1.0).sum()), len(tw)
    r["n_unit_cells_rho_not1"], r["n_unit_cells"] = nr1, nall
    # H1 の書き方（設計 4 節の 1〜5）
    if not r["H1"]:
        w = "満たさなかった"
    elif not r["H1p"]:
        w = "正解で別々に学んだ分類器の依存の水準の中に収まる"
        if c == "IEMOCAP":
            w += "．IEMOCAP 即興の依存の増加は筋書きの話題の共有と分けられない（筋書きで中心化した D を並べる）"
    else:
        if r["phi"]:
            w = "依存の増加（誤りの相関の増加を伴う）"
        else:
            ph = rows(c, "dphi_joint_a1_none")
            small = sum(ph.loc[K, "lo"] > 0 and ph.loc[K, "est"] < 0.02 for K in KR)
            w = "依存の増加" + ("（誤りの相関の増加は下限 0.02 に届かない）" if small >= 3 else "（誤りの相関の増加は規則を満たさない）")
        if r["share"]:
            w += f"．同じ疑似ラベルを共有したことが依存を増やした（段 3 の共有．段 2 の共有は分けていない．ρ = 1，a = 1 の点で比べた．joint_a1 の ρ が 1 でないセル {nr1} / {nall}）"
            if r["phi_share"]:
                w += "．誤りの相関も共有で増えた"
            if c == "MOSEI" and r["chan_share_n_K_robust"] < 3:
                w += "．共有の比べも，同じ人物の別の動画が候補に残る漏れと分けられない"
            if r["cent_share_n_K_small"] >= 3:
                w += "．共有で増えた依存の多くは話者（塊）の水準の共有である"
        else:
            w += f"．依存の増加は，同じ重みで各モダリティ自身の疑似ラベルで学んだ場合と分けられない（ρ = 1，a = 1 の点で比べた．joint_a1 の ρ が 1 でないセル {nr1} / {nall}）"
        if c == "MOSEI" and r["chan_n_K_robust"] < 3:
            w += "．同じ人物の別の動画が候補に残る漏れと分けられない"
        if r["cent_n_K_small"] >= 3:
            w += "．増えた依存の多くは話者（塊）の水準の共有である"
        if c == "IEMOCAP":
            w += "．IEMOCAP 即興の依存の増加は筋書きの話題の共有と分けられない（筋書きで中心化した D を並べる）"
    r["H1_wording"] = w
    for q, nm in (("dD_joint_a1_none", "H1"), ("dD_joint_a1_oracle", "H1p"), ("dD_rho1_ownret", "share")):
        if c == "MOSEI":
            r[f"jk_{nm}_n_K_flip"] = np.nan; r[f"jk_{nm}_n_lo_pos"] = np.nan
            continue
        jj, mm = rows(c, q, var="jk"), rows(c, q)
        r[f"jk_{nm}_n_K_flip"] = int(sum((mm.loc[K, "lo"] > 0) != (jj.loc[K, "lo"] > 0) for K in KR))
        r[f"jk_{nm}_n_lo_pos"] = int(sum(jj.loc[K, "lo"] > 0 for K in KR))
    g = rows(c, "G_joint_a1")
    r["additivity_n_hi_neg"] = int(sum(g.loc[K, "hi"] < 0 for K in KR))
    r["additivity_fails"] = r["additivity_n_hi_neg"] >= 3
    h = ac[(ac.c == c) & ac.K.isin(KR) & ac.set.isin(L_SETS)]
    r["n_cells_h_excess_ge_0.5"] = int((h.h_excess >= 0.5).sum())
    for s, tag in (("joint_a1", "H4"), ("joint_a1_rho1.0", "H4_rho1")):
        gs = rows(c, f"G_{s}")
        ivs = [K for K in KR if gs.loc[K, "hi"] < 0]
        sh, dl = rows(c, f"share_dep_{s}"), rows(c, f"dLdep_{s}_none")
        ok = [K for K in ivs if dl.loc[K, "lo"] > 0]
        r[f"{tag}_n_K_iv"] = len(ivs)
        r[f"{tag}_n_K_half"] = int(sum(sh.loc[K, "est"] >= 0.5 for K in ok))
        r[f"{tag}_n_K_part"] = int(sum(sh.loc[K, "est"] >= 0.25 for K in ok))
        parts = {"Scal": rows(c, f"Scal_{s}"), "S1b": rows(c, f"S1b_{s}"), "dLdep": dl, "dLcal": rows(c, f"dLcal_{s}")}
        big = {}
        for K in ivs:
            mxv = max(parts[p].loc[K, "est"] for p in PART)
            big[K] = [p for p in PART if parts[p].loc[K, "est"] == mxv]
        r[f"{tag}_largest_by_K"] = ";".join(f"K{K}:" + "+".join(v) for K, v in big.items()) if ivs else np.nan
        r[f"{tag}_share_by_K"] = ";".join(f"K{K}:{sh.loc[K, 'est']:.2f}" for K in ivs) if ivs else np.nan
        if s == "joint_a1":
            if not r["additivity_fails"]:
                w4 = "前提 (iv) を満たさない（足し合わさらない理由を書かない）"
            else:
                if r[f"{tag}_n_K_half"] >= 3:
                    w4 = "依存を無視した損が，足し合わさらない分の半分以上を説明する"
                elif r[f"{tag}_n_K_part"] >= 3:
                    w4 = "依存を無視した損が，足し合わさらない分の一部（4 分の 1 以上）を説明する"
                else:
                    cnt = {p: sum(p in v for v in big.values()) for p in PART}
                    top = [p for p in PART if cnt[p] == max(cnt.values())]
                    w4 = "足し合わさらない分のうち最も大きいのは" + "と".join(f"「{PART[p]}」" for p in top) + \
                        f"である（前提 (iv) を満たした K の {len(ivs)} つのうち {max(cnt.values())} つ）"
                w4 += f"（K ごとの割合 ΔL_dep / S：{r[f'{tag}_share_by_K']}．h − h⁰ ≥ 0.5 のセル {r['n_cells_h_excess_ge_0.5']} 個）"
            r["H4_wording"] = w4
    R[c] = r
G = pd.DataFrame(R).T
for nm in ("H1", "H1_joint", "H1p", "H1_AV", "share", "H2", "H3", "additivity_fails"):
    n_all = int(G[nm].astype(bool).sum()); n_ex = int(G.loc[G.index != "IEMOCAP", nm].astype(bool).sum())
    G[f"{nm}_n_corpora"], G[f"{nm}_n_corpora_excl_IEMOCAP"] = n_all, n_ex
    G[f"{nm}_overall"] = n_all >= 3
    G[f"{nm}_depends_on_IEMOCAP"] = (n_all >= 3) and (n_ex <= 2)

meta = {"commit", "summarize_commit", "summarize_dirty", "boot_seed", "n_boot"}
cols_C = [x for x in C.columns if x not in meta]
print("criteria の列（メタを除く）", len(cols_C), "作り直した列", G.shape[1])
print("criteria にあって作り直しに無い列", sorted(set(cols_C) - set(G.columns)))
print("作り直しにあって criteria に無い列", sorted(set(G.columns) - set(cols_C)))
TEXT = ("H1_wording", "H4_wording", "H4_largest_by_K", "H4_share_by_K", "H4_rho1_largest_by_K", "H4_rho1_share_by_K")
n_ok, n_bad, bad_cols = 0, 0, []
for col in cols_C:
    if col not in G.columns or col in TEXT:
        continue
    for c in C.index:
        a, b = C.loc[c, col], G.loc[c, col]
        both_nan = (pd.isna(a) and pd.isna(b))
        if both_nan:
            ok = True
        elif pd.isna(a) or pd.isna(b):
            ok = False
        elif isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
            ok = bool(a) == bool(b)
        else:
            ok = abs(float(a) - float(b)) <= 1e-12
        n_ok += ok; n_bad += not ok
        if not ok:
            bad_cols.append(col)
print(f"数と真偽の列：一致 {n_ok}，不一致 {n_bad}，不一致の列 {sorted(set(bad_cols))}")
for col in TEXT:
    m = 0
    for c in C.index:
        a, b = C.loc[c, col], G.loc[c, col]
        m += (pd.isna(a) and pd.isna(b)) or (not pd.isna(a) and not pd.isna(b) and str(a) == str(b))
    print(f"文の列 {col}：一致 {m} / {len(C)}")
