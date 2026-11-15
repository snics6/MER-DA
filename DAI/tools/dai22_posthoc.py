"""実験 22 の計算の後に，保存した単位の npz から出すもの（設計 `docs/設計/実験22_設計.md` 13 節．学び直さない）．

出すもの（results/dai22/posthoc/ に書く）：
  start_pseudo_agree.csv   出発点の判定と J_prop の一致（4 つの行．K 本を除いた候補．単位の平均・最小・最大）．8 節の一致の比べに添える
  meld_nonshared_geff.csv  MELD の共有を除いた候補（K 本を除く）の上の発話の数と塊の実効の数（単位ごとの平均・最小）
  check_dai14_imitable.csv 声・顔・言葉の行の start_minus_none_<s> ＋ R0_minus_start_<s> と実験 14 の imitable_R0 の点推定の照合
  resolved_rows.csv        照合が外れて解き直した R0 の行（resolve_kind，予測の違い，保存との差，勾配，止まり）
使い方：python tools/dai22_posthoc.py   （DAI の直下で）
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R22 = ROOT / "results" / "dai22"
R14 = ROOT / "results" / "dai14"
OUT = R22 / "posthoc"
KS = (5, 25)
CORPORA = ("AFEW", "IEMOCAP", "MELD", "MOSEI")
PROP = "prop_one"


def units(T):
    for p in sorted((R22 / "units").glob(f"{T}_f*_r*.npz")):
        a = dict(np.load(p, allow_pickle=False))
        b = dict(np.load(R14 / "units" / p.name, allow_pickle=False))
        if not (a["sid"].astype(str) == b["sid"].astype(str)).all():
            raise SystemExit(f"{p.name}：候補の並びが実験 14 と違う")
        yield p.stem, a, b


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    agree, geff = [], []
    for T in CORPORA:
        acc = {}
        for name, a, b in units(T):
            for K in KS:
                nk = a[f"nk__K{K}"].astype(bool)
                q = a[f"tgt__K{K}"][nk]
                acc.setdefault((K, "AVT"), []).append(float((a["start"][nk] == q).mean()))
                for s in "AVT":
                    acc.setdefault((K, s), []).append(float((b[f"start__{s}"][nk] == q).mean()))
                if T == "MELD":
                    m = nk & a["nonshared"].astype(bool)
                    n = pd.Series(a["cl"][m]).value_counts().to_numpy()
                    geff.append(dict(unit=name, K=K, n_nonshared_nk=int(m.sum()), n_clusters=int(len(n)), g_eff=float(n.sum() ** 2 / (n ** 2).sum())))
        for (K, s), v in acc.items():
            agree.append(dict(T=T, K=K, row=s, agree_start_pseudo_mean=float(np.mean(v)), agree_start_pseudo_min=float(np.min(v)),
                              agree_start_pseudo_max=float(np.max(v)), n_units=len(v)))
    pd.DataFrame(agree).sort_values(["T", "K", "row"]).to_csv(OUT / "start_pseudo_agree.csv", index=False)
    G = pd.DataFrame(geff)
    G.groupby("K").agg(n_nonshared_nk_mean=("n_nonshared_nk", "mean"), n_nonshared_nk_min=("n_nonshared_nk", "min"),
                       n_clusters_min=("n_clusters", "min"), g_eff_mean=("g_eff", "mean"), g_eff_min=("g_eff", "min")).reset_index() \
        .to_csv(OUT / "meld_nonshared_geff.csv", index=False)
    c22 = pd.read_csv(R22 / "summary" / "conditions.csv")
    c14 = pd.read_csv(R14 / "summary" / "conditions.csv")
    chk = []
    for T in CORPORA:
        for K in KS:
            for s in "AVT":
                g = lambda qn: float(c22[(c22["T"] == T) & (c22.K == K) & (c22.subset == "all") & (c22.quantity == qn)].effect.iloc[0])
                ref = float(c14[(c14["T"] == T) & (c14.K == K) & (c14.student == s) & (c14.quantity == "imitable_R0")].effect.iloc[0])
                tot = g(f"start_minus_none_{s}") + g(f"R0_minus_start_{s}")
                chk.append(dict(T=T, K=K, row=s, sum_22=tot, imitable_R0_14=ref, abs_diff=abs(tot - ref)))
    C = pd.DataFrame(chk)
    C.to_csv(OUT / "check_dai14_imitable.csv", index=False)
    rows = []
    for p in sorted((R22 / "units").glob("*.csv")):
        d = pd.read_csv(p)
        d = d[(d.readout == "R0") & (d.abs_diff > 1e-3)]
        rows.append(d)
    keep = ["T", "fold", "seed", "K", "abs_diff", "n_cand", "tol_resolved", "resolve_kind", "n_pred_diff_tight", "n_pred_diff_tol1e-08",
            "n_pred_diff_tol1e-10", "grad_tol1e-08", "grad_tol1e-10", "stalled_tol1e-08", "stalled_tol1e-10", "n_iter_tol1e-08", "n_iter_tol1e-10"]
    R = pd.concat(rows)[keep]
    R.to_csv(OUT / "resolved_rows.csv", index=False)
    print(json.dumps(dict(check_dai14_abs_diff_max=float(C.abs_diff.max()), n_resolved=len(R),
                          kinds=R.resolve_kind.value_counts().to_dict()), ensure_ascii=False))


if __name__ == "__main__":
    main()
