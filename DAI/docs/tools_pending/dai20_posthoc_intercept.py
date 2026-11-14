"""実験 20 の計算の後の解析：「版 − 積」の差を，切片の分と向きの分に分ける（単位の保存の判定を読むだけ．学び直さない）．

単位の保存には，版ごとに切片の扱いを変えた 4 通りの判定がある（`dai16_probe.run_unit`）：
  raw：そのまま．adj：ラベルを使わずに切片をずらす．kb：adj の上に K 本で決めた切片を足す．
  ob ：候補（K 本以外）の全ラベルで，対数尤度が最大になるように合わせた切片を足す（参照．テストの UAR が最大になる切片ではない）．
       両方の版の切片を同じ規則で全ラベルに合わせるので，この層の差は重みの向きの違いを表す．
分け方（UAR は正解の数について線形なので，区間は足し合わせた正解の数に `dai1_stats.effect_stats` を当てて作る）：
  向きの分 ＝ 版(ob) − 積(ob)
  切片の分 ＝ [版(L) − 積(L)] − 向きの分 ＝ [版(L) ＋ 積(ob)] − [版(ob) ＋ 積(L)]   （L は adj・raw・kb）
  切片の残り ＝ 版(ob) − 版(L)   （その版の切片が，全ラベルに合わせた切片からどれだけ遠いか）
区間の乱数の鍵は，集計の段（`dai20_eval.stage_summary`）の「版 − 積」の鍵と同じにし，全体・切片・向きと全部の層で共有する．
塊の重みを引き直すコーパスでは，引き直しの 1 回ごとに「全体 ＝ 切片の分 ＋ 向きの分」が成り立ち，adj の全体の区間は集計の段と一致する．

使い方：python dai20_posthoc_intercept.py [--units <units の置き場>] [--out <出力の置き場>] [--key T|c] [--grid <格子の点>] [--allow-partial]
  既定は評価の回（--key T．格子の点は固定のファイル．単位の数・名前・鍵・集計の段との一致を確かめ，合わなければ止まる）．
  --key c・--grid・--allow-partial は検査用（下調べの単位で動作を確かめるとき．確かめを省く）．
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai1_stats as ST  # noqa: E402
import dai12_probe as P12  # noqa: E402
import dai15  # noqa: E402
import dai16_probe as P16  # noqa: E402
import dai_data as DD  # noqa: E402

ROOT = DD.ROOT
KS = (5, 10, 25)
GRID = P16.GRID
LAYERS = ("adj", "raw", "kb")
SEED_SETS = {"all": None, "0-1": (0, 1), "2-9": tuple(range(2, 10))}        # 集計の段と同じ分け方（K 本の引き方 0・1 は下調べと同じ K 本）
FROZEN = ROOT / "DAI" / "frozen" / "dai20_frozen.json"
N_PER_CORPUS = 50                                                           # 評価の回：分割 5 × K 本の引き方 10
TOL = 1e-9
MODULES = ("dai1_stats.py", "dai12_probe.py", "dai15.py", "dai16_probe.py", "dai_data.py", "dai20_eval.py")


def same_but(i: int, n0=None, beta=None) -> int:
    """格子の点 i と同じ設定で，n0 か β だけを替えた点の番号（`dai20_eval.same_but` と同じ）．"""
    a, b, p = GRID[i]
    return GRID.index((a if n0 is None else n0, b if beta is None else beta, p))


def versions_of(i: int, j_np, extra: dict) -> dict:
    """この台本の版の名前（集計の段の名前と同じ）→ 単位の保存の版の名前．`dai20_eval.stage_summary` の pick・names と同じ対応．"""
    v = {"prod": "prod", "prop": f"M_F6_mix{i}", "prop_noprof": f"M_F6_mix{same_but(i, beta=0.0)}", "prop_noint": f"M_F6n_mix{i}",
         "prop_n0zero": f"M_F6_mix{same_but(i, n0=0.0)}",                      # 残り方を掛けない版（prop_home）は ob の層の保存が無いので分けられない
         "prop_levelfree": f"M_F6_mix{GRID.index((float('inf'), 1.0, 'eq'))}", "M_F6_K": "M_F6_K", "M_F2_K": "M_F2_K", "cf_D_l05": "cf_D_l05",
         "M_F6_ext": "M_F6_ext", "M_F8": "M_F8", "M_F6_orc": "M_F6_orc"}
    if j_np is not None:
        v["prop_noprof_sel"] = f"M_F6_mix{j_np}"
    v.update({k_: f"M_F6_mix{g_}" for k_, g_ in extra.items()})
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default=str(ROOT / "results" / "dai20_eval" / "units"))
    ap.add_argument("--out", default=str(ROOT / "results" / "dai20_eval" / "posthoc"))
    ap.add_argument("--key", default="T", choices=("T", "c"))
    ap.add_argument("--grid", type=int, default=None)
    ap.add_argument("--allow-partial", action="store_true", help="検査用：単位の数・名前・鍵・集計の段との一致の確かめを省く")
    args = ap.parse_args()
    strict = not args.allow_partial
    if strict and (args.key != "T" or args.grid is not None):
        raise SystemExit("--key c と --grid は検査用（--allow-partial と一緒に使う）")
    udir, out = Path(args.units), Path(args.out)
    sdir = udir.parent / "summary"
    fz, extra_of = None, {}
    if args.grid is None:
        import dai20_eval as P20                                             # 固定のファイルの元の sha256 と設定を確かめて読む（集計の段と同じ読み方）
        fz = P20.load_frozen()
        if (sdir / "grid_levels.csv").exists():                              # ほかの 3 コーパスで選んだ点と，自分のテストで最良の点（集計の段が決めた点を読む）
            gl = pd.read_csv(sdir / "grid_levels.csv")
            for (c, K), g in gl.groupby(["c", "K"]):
                extra_of[(c, int(K))] = {"prop_loto": int(g[g.loto.astype(bool)].grid.iloc[0]), "oracle_grid": int(g[g.oracle.astype(bool)].grid.iloc[0])}
        elif strict:
            raise SystemExit(f"{sdir / 'grid_levels.csv'} が無い")
    files = sorted(udir.glob("*.npz"))
    cells: dict = {}
    n_of: dict = {}
    for p in files:
        if not p.with_suffix(".csv").exists():                              # csv が済みの印（npz だけの単位は書きかけ）
            raise SystemExit(f"{p.name}：csv が無い")
        head, r = p.stem.rsplit("_r", 1)
        head, f = head.rsplit("_f", 1)
        T, c = head.split("_", 1)
        if strict and c != "ALL":
            raise SystemExit(f"{p.name}：評価の回の単位の名前ではない")
        corp = T if args.key == "T" else c
        n_of[corp] = n_of.get(corp, 0) + 1
        z = np.load(p, allow_pickle=False)
        yt, cl = z["y_test"].astype(int), z["cl_test"]
        for K in KS:
            if f"pred_ob_K{K}_prod" not in z.files:
                if strict:
                    raise SystemExit(f"{p.name}：K = {K} の判定が無い")
                continue
            i, j_np = (fz[corp][K] if fz is not None else (args.grid, None))
            for v, base in versions_of(i, j_np, extra_of.get((corp, K), {})).items():
                for lay in LAYERS + ("ob",):
                    kk = f"pred_{lay}_K{K}_{base}"
                    if kk not in z.files:
                        if strict:
                            raise SystemExit(f"{p.name}：{kk} が無い")
                        continue
                    key = (corp, K, f"{v}@{lay}")
                    if (int(f), int(r)) in cells.get(key, {}) and args.key == "T":
                        raise SystemExit(f"{p.name}：同じ (分割, 引き方) の単位が重なっている")
                    P12._add(cells.setdefault(key, {}), (int(f), int(r)), ST.counts(yt, z[kk].astype(int), cl), f"{p.stem} {v}@{lay}")
    if strict and (sorted(n_of) != sorted(DD.CORPORA) or any(n != N_PER_CORPUS for n in n_of.values())):
        raise SystemExit(f"単位の数が合わない：{n_of}")
    P12._finish(cells)
    geff = {c: ST.g_eff_of(c)[0] for c in DD.CORPORA}

    def sub(A: dict, seeds) -> dict:
        return A if seeds is None else {k_: x for k_, x in A.items() if k_[1] in seeds}

    def comb(c, K, names, seeds):
        """足し合わせた正解の数（件数は 1 つぶん）．UAR の和の区間を作るための入れ物．単位の組と塊の並びがそろわなければ止まる．"""
        cs = [sub(cells[(c, K, n_)], seeds) for n_ in names]
        if any(set(x) != set(cs[0]) for x in cs):
            raise SystemExit(f"差の単位がそろわない：{(c, K, names)}")
        for k_ in cs[0]:
            if any(len(x[k_][0]) != len(cs[0][k_][0]) or not (x[k_][0] == cs[0][k_][0]).all() or not (x[k_][2] == cs[0][k_][2]).all() for x in cs):
                raise SystemExit(f"塊の並びか件数がそろわない：{(c, K, names, k_)}")
        return {k_: (cs[0][k_][0], sum(x[k_][1] for x in cs), cs[0][k_][2]) for k_ in cs[0]}

    def eff(A, B, c, K, v, sn):
        if set(A) != set(B):
            raise SystemExit(f"差の単位がそろわない：{(c, K, v, sn)}")
        es = ST.effect_stats(A, B, geff[c], P16.row_rng("dai20", "A", v, "prod", c, K, sn))      # 集計の段の「v − 積」と同じ鍵
        return dict(est=es["effect"], lo=es["lo"], hi=es["hi"], se=es["se"], df=es.get("df", np.nan), method=es["method"], n_fr=es["n_units"],
                    mark={"上回る": "↑", "下回る": "↓"}.get(P16.verdict(es["lo"], es["hi"], es["effect"]), ""))

    rows, lrows = [], []
    for (c, K) in sorted({(c, K) for c, K, _ in cells}):
        have = sorted({v.split("@")[0] for (c2, K2, v) in cells if c2 == c and K2 == K and v.endswith("@ob")})
        nall = len(cells[(c, K, "prod@ob")])
        for sn, ss in SEED_SETS.items():
            if sn != "all" and len(sub(cells[(c, K, "prod@ob")], ss)) in (0, nall):
                continue
            for v in have:
                for lay in LAYERS + ("ob",):
                    if (c, K, f"{v}@{lay}") not in cells:
                        continue
                    A = comb(c, K, [f"{v}@{lay}"], ss)
                    es = ST.effect_stats(A, {k_: (x[0], np.zeros_like(x[1]), x[2]) for k_, x in A.items()}, geff[c], P16.row_rng("dai20post", "L", c, K, v, lay, sn))
                    lrows.append(dict(c=c, K=K, version=v, layer=lay, seeds=sn, uar=es["effect"], lo=es["lo"], hi=es["hi"], se=es["se"], n_fr=es["n_units"]))
                    if v != "prod" and lay != "ob":
                        rows.append(dict(c=c, K=K, version=v, quantity="total", layer=lay, seeds=sn,
                                         **eff(comb(c, K, [f"{v}@{lay}"], ss), comb(c, K, [f"prod@{lay}"], ss), c, K, v, sn)))
                        rows.append(dict(c=c, K=K, version=v, quantity="intercept", layer=lay, seeds=sn,
                                         **eff(comb(c, K, [f"{v}@{lay}", "prod@ob"], ss), comb(c, K, [f"{v}@ob", f"prod@{lay}"], ss), c, K, v, sn)))
                    if lay != "ob":
                        rows.append(dict(c=c, K=K, version=v, quantity="intercept_gap", layer=lay, seeds=sn,
                                         **eff(comb(c, K, [f"{v}@ob"], ss), comb(c, K, [f"{v}@{lay}"], ss), c, K, v, sn)))
                if v != "prod":
                    rows.append(dict(c=c, K=K, version=v, quantity="direction", layer="ob", seeds=sn,
                                     **eff(comb(c, K, [f"{v}@ob"], ss), comb(c, K, ["prod@ob"], ss), c, K, v, sn)))
    ef, lv = pd.DataFrame(rows), pd.DataFrame(lrows)
    if ef.duplicated(["c", "K", "version", "quantity", "layer", "seeds"]).any() or lv.duplicated(["c", "K", "version", "layer", "seeds"]).any():
        raise SystemExit("行が重なっている")
    # 検算 1：点推定と区間の両端が「全体 ＝ 切片の分 ＋ 向きの分」を満たす（点推定は恒等式に近い．版の取り違えは検算 2・3 で見る）
    piv = ef[ef.quantity.isin(["total", "intercept"])].pivot_table(index=["c", "K", "version", "layer", "seeds"], columns="quantity", values="est").reset_index()
    piv = piv.merge(ef[ef.quantity == "direction"][["c", "K", "version", "seeds", "est"]].rename(columns={"est": "direction"}), on=["c", "K", "version", "seeds"])
    dev_add = float((piv.total - piv.intercept - piv.direction).abs().max()) if len(piv) else float("nan")
    # 検算 2・3：集計の段の水準（adj・raw）と「版 − 積」の差（adj．点推定と区間の両端）に一致する
    dev_lv = dev_ef = dev_ci = float("nan")
    n_lv = n_ef = 0
    sums = {}
    if args.key == "T" and (sdir / "effects.csv").exists() and (sdir / "levels.csv").exists():
        sl = pd.read_csv(sdir / "levels.csv", low_memory=False)
        se_ = pd.read_csv(sdir / "effects.csv", low_memory=False)
        sums = {q.name: dai15.sha(q) for q in (sdir / "levels.csv", sdir / "effects.csv", sdir / "grid_levels.csv") if q.exists()}
        mine = lv[lv.layer.isin(["adj", "raw"])].assign(version=lambda d: np.where(d.layer == "raw", d.version + "@raw", d.version))
        m_ = mine.merge(sl[["c", "K", "version", "seeds", "uar"]].rename(columns={"uar": "uar_summary"}), on=["c", "K", "version", "seeds"])
        n_lv, dev_lv = len(m_), (float((m_.uar - m_.uar_summary).abs().max()) if len(m_) else float("nan"))
        t_ = ef[(ef.quantity == "total") & (ef.layer == "adj")].merge(
            se_[se_.b == "prod"][["a", "c", "K", "seeds", "est", "lo", "hi"]].rename(columns={"a": "version", "est": "est_s", "lo": "lo_s", "hi": "hi_s"}),
            on=["version", "c", "K", "seeds"])
        n_ef = len(t_)
        dev_ef = float((t_.est - t_.est_s).abs().max()) if n_ef else float("nan")
        dev_ci = float(max((t_.lo - t_.lo_s).abs().max(), (t_.hi - t_.hi_s).abs().max())) if n_ef else float("nan")
    msg = (f"単位 {len(files)}（{n_of}），差の行 {len(ef)}，水準の行 {len(lv)}．足し合わせのずれの最大 {dev_add:.2e}．集計の段との照合：水準 {n_lv} 行でずれの最大 {dev_lv:.2e}，"
           f"差 {n_ef} 行で点推定のずれの最大 {dev_ef:.2e}・区間の端のずれの最大 {dev_ci:.2e}")
    print(msg, flush=True)
    if strict:
        if not (dev_add < TOL):
            raise SystemExit("足し合わせの検算が合わない")
        if not (n_lv > 0 and n_ef > 0 and dev_lv < TOL and dev_ef < TOL and dev_ci < 1e-6):
            raise SystemExit("集計の段と合わない（または照合する行が無い）")
    out.mkdir(parents=True, exist_ok=True)
    ef.to_csv(out / "intercept_direction.csv", index=False)
    lv.to_csv(out / "intercept_levels.csv", index=False)
    git = lambda *a: subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    json.dump(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), commit=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain", "--", "DAI/exp", "DAI/frozen", "DAI/docs/tools_pending")),
                   script_sha256=dai15.sha(Path(__file__).resolve()), modules_sha256={n: dai15.sha(ROOT / "DAI" / "exp" / n) for n in MODULES if (ROOT / "DAI" / "exp" / n).exists()},
                   units_dir=str(udir), n_units=len(files), units_per_corpus=n_of, key=args.key, grid=args.grid, allow_partial=args.allow_partial,
                   frozen_sha256=(dai15.sha(FROZEN) if args.grid is None else ""), summary_sha256=sums, additivity_dev_max=dev_add,
                   levels_vs_summary=dict(n=n_lv, dev_max=dev_lv), total_vs_summary=dict(n=n_ef, est_dev_max=dev_ef, ci_dev_max=dev_ci), n_rows=len(ef)),
              open(out / "intercept_provenance.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
