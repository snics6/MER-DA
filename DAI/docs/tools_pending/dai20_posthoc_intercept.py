"""実験 20 の計算の後の解析：「版 − 積」の差を，切片の分と向きの分に分ける（単位の保存の判定を読むだけ．学び直さない）．

単位の保存には，版ごとに切片の扱いを変えた 4 通りの判定がある（`dai16_probe.run_unit`）：
  raw：そのまま．adj：ラベルを使わずに切片をずらす．kb：adj の上に K 本で決めた切片を足す．
  ob ：候補（K 本以外）の全ラベルで決めた切片を足す（参照．切片を最良にそろえて，向きの良さだけを比べる層）．
分け方（UAR は正解の数について線形なので，区間は足し合わせた正解の数に `dai1_stats.effect_stats` を当てて作る）：
  向きの分 ＝ 版(ob) − 積(ob)
  切片の分 ＝ [版(L) − 積(L)] − 向きの分 ＝ [版(L) ＋ 積(ob)] − [版(ob) ＋ 積(L)]   （L は adj・raw・kb）
  切片の残り ＝ 版(ob) − 版(L)   （その版の切片が，全ラベルで決めた切片からどれだけ遠いか）

使い方：python dai20_posthoc_intercept.py [--units <units の置き場>] [--out <出力の置き場>] [--key T|c] [--grid <格子の点>]
  --key c と --grid は検査用（下調べの単位で動作を確かめるとき）．既定は評価の回（--key T．格子の点は固定のファイル）．
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
FROZEN = ROOT / "DAI" / "frozen" / "dai20_frozen.json"


def same_but(i: int, n0=None, beta=None) -> int:
    """格子の点 i と同じ設定で，n0 か β だけを替えた点の番号（`dai20_eval.same_but` と同じ）．"""
    a, b, p = GRID[i]
    return GRID.index((a if n0 is None else n0, b if beta is None else beta, p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default=str(ROOT / "results" / "dai20_eval" / "units"))
    ap.add_argument("--out", default=str(ROOT / "results" / "dai20_eval" / "posthoc"))
    ap.add_argument("--key", default="T", choices=("T", "c"))
    ap.add_argument("--grid", type=int, default=None)
    args = ap.parse_args()
    udir, out = Path(args.units), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fz = json.load(open(FROZEN))["choice"] if args.grid is None else None
    files = sorted(udir.glob("*.npz"))
    cells: dict = {}
    for p in files:
        if not p.with_suffix(".csv").exists():                              # csv が済みの印（npz だけの単位は書きかけ）
            raise SystemExit(f"{p.name}：csv が無い")
        head, r = p.stem.rsplit("_r", 1)
        head, f = head.rsplit("_f", 1)
        T, c = head.split("_", 1)
        corp = T if args.key == "T" else c
        z = np.load(p, allow_pickle=False)
        yt, cl = z["y_test"].astype(int), z["cl_test"]
        for K in KS:
            if f"pred_ob_K{K}_prod" not in z.files:
                continue
            i = int(fz[corp][str(K)]["grid"]) if fz is not None else args.grid
            vers = {"prod": "prod", "prop": f"M_F6_mix{i}", "noprof": f"M_F6_mix{same_but(i, beta=0.0)}", "noint": f"M_F6n_mix{i}",
                    "levelfree": f"M_F6_mix{GRID.index((float('inf'), 1.0, 'eq'))}", "konly": "M_F6_K", "level_only": "M_F2_K", "confusion": "cf_D_l05",
                    "source_only": "M_F6_ext", "all_labels": "M_F6_orc"}
            for v, base in vers.items():
                for lay in LAYERS + ("ob",):
                    kk = f"pred_{lay}_K{K}_{base}"
                    if kk in z.files:
                        P12._add(cells.setdefault((corp, K, f"{v}@{lay}"), {}), (int(f), int(r)), ST.counts(yt, z[kk].astype(int), cl), f"{p.stem} {v}@{lay}")
    P12._finish(cells)
    geff = {c: ST.g_eff_of(c)[0] for c in DD.CORPORA}

    def comb(c, K, names):
        """足し合わせた正解の数（件数は 1 つぶん）．UAR の和の区間を作るための入れ物．"""
        cs = [cells[(c, K, n_)] for n_ in names]
        return {k_: (cs[0][k_][0], sum(x[k_][1] for x in cs), cs[0][k_][2]) for k_ in cs[0]}

    def eff(A, B, c, key):
        es = ST.effect_stats(A, B, geff[c], P16.row_rng("dai20post", *key))
        return dict(est=es["effect"], lo=es["lo"], hi=es["hi"], se=es["se"], df=es.get("df", np.nan), method=es["method"], n_fr=es["n_units"],
                    mark={"上回る": "↑", "下回る": "↓"}.get(P16.verdict(es["lo"], es["hi"], es["effect"]), ""))

    rows, lrows = [], []
    for (c, K) in sorted({(c, K) for c, K, _ in cells}):
        have = {v.split("@")[0] for (c2, K2, v) in cells if c2 == c and K2 == K and v.endswith("@ob")}
        for v in sorted(have):
            for lay in LAYERS + ("ob",):
                if (c, K, f"{v}@{lay}") not in cells:
                    continue
                A = cells[(c, K, f"{v}@{lay}")]
                e_ = eff(A, {k_: (x[0], np.zeros_like(x[1]), x[2]) for k_, x in A.items()}, c, ("L", c, K, v, lay))
                lrows.append(dict(c=c, K=K, version=v, layer=lay, uar=e_["est"], lo=e_["lo"], hi=e_["hi"], se=e_["se"], n_fr=e_["n_fr"]))
            if v == "prod":
                continue
            rows.append(dict(c=c, K=K, version=v, quantity="direction", layer="ob", **eff(comb(c, K, [f"{v}@ob"]), comb(c, K, ["prod@ob"]), c, ("D", c, K, v))))
            for lay in LAYERS:
                if (c, K, f"{v}@{lay}") not in cells:
                    continue
                rows.append(dict(c=c, K=K, version=v, quantity="total", layer=lay,
                                 **eff(comb(c, K, [f"{v}@{lay}"]), comb(c, K, [f"prod@{lay}"]), c, ("T", c, K, v, lay))))
                rows.append(dict(c=c, K=K, version=v, quantity="intercept", layer=lay,
                                 **eff(comb(c, K, [f"{v}@{lay}", "prod@ob"]), comb(c, K, [f"{v}@ob", f"prod@{lay}"]), c, ("I", c, K, v, lay))))
        for v in sorted(have):
            for lay in LAYERS:
                if (c, K, f"{v}@{lay}") in cells:
                    rows.append(dict(c=c, K=K, version=v, quantity="intercept_gap", layer=lay,
                                     **eff(comb(c, K, [f"{v}@ob"]), comb(c, K, [f"{v}@{lay}"]), c, ("G", c, K, v, lay))))
    ef = pd.DataFrame(rows)
    ef.to_csv(out / "intercept_direction.csv", index=False)
    pd.DataFrame(lrows).to_csv(out / "intercept_levels.csv", index=False)
    # 検算：点推定は「全体 ＝ 切片の分 ＋ 向きの分」を満たす
    piv = ef[ef.quantity.isin(["total", "intercept"])].pivot_table(index=["c", "K", "version", "layer"], columns="quantity", values="est").reset_index()
    piv = piv.merge(ef[ef.quantity == "direction"][["c", "K", "version", "est"]].rename(columns={"est": "direction"}), on=["c", "K", "version"])
    dev = float((piv.total - piv.intercept - piv.direction).abs().max()) if len(piv) else float("nan")
    # 検算：集計の段の「提案 − 積」（adj）と同じ値になる
    dev_sum = float("nan")
    sm = udir.parent / "summary" / "effects.csv"
    if args.key == "T" and sm.exists():
        s_ = pd.read_csv(sm, low_memory=False)
        s_ = s_[(s_.a == "prop") & (s_.b == "prod") & (s_.seeds == "all")][["c", "K", "est"]].rename(columns={"est": "est_summary"})
        m_ = ef[(ef.version == "prop") & (ef.quantity == "total") & (ef.layer == "adj")].merge(s_, on=["c", "K"])
        dev_sum = float((m_.est - m_.est_summary).abs().max()) if len(m_) else float("nan")
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    json.dump(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), commit=commit, script_sha256=dai15.sha(Path(__file__).resolve()), units_dir=str(udir), n_units=len(files),
                   key=args.key, grid=args.grid, frozen_sha256=(dai15.sha(FROZEN) if args.grid is None else ""), additivity_dev_max=dev, total_vs_summary_dev_max=dev_sum,
                   n_rows=len(ef)), open(out / "intercept_provenance.json", "w"), indent=1, ensure_ascii=False)
    print(f"単位 {len(files)}，差の行 {len(ef)}，水準の行 {len(lrows)}．足し合わせの検算のずれの最大 {dev:.2e}．集計の段との差の最大 {dev_sum:.2e}", flush=True)


if __name__ == "__main__":
    main()
