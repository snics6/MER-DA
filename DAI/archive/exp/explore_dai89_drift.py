"""探索（関門を通していない．結果の文では「探索」と書く）：DAI-8・DAI-9 の EM が感情の割り当てから離れる理由の確かめ。

共通の共分散のガウスの混合の最尤は，固い割り当てでは det(W) を小さくする分け方（Friedman & Rubin 1967，Scott & Symons 1971）で，
白色化した計量での平均の間の散らばりが大きい分け方を好む。そこで DAI-9 の r = 0 の 20 単位で，raw と pca64 の空間について，
  (1) 感情の割り当て，EM_truth と EM_rand の最後の割り当て（保存した P_）の上位 3 つの固有値の和（DAI-8 の _top3．A = 0.5T + 0.5I）
  (2) 各割り当てから 1 回の M 段で作ったパラメタでの，候補の発話の混合の対数尤度の平均（1 本あたり）
  (3) EM_truth の最後の割り当てと感情・グループ・対話の AMI（小さい方のエントロピーで正規化）
を出す。テストのラベルは使わない（候補の正解は DAI-8・9 と同じく診断にだけ使う）。

python DAI/exp/explore_dai89_drift.py --out results/dai89_explore_drift
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_mutual_info_score as AMI

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dai1  # noqa: E402
import dai_data as DD  # noqa: E402
import dai8_probe as P8  # noqa: E402
import dai9_probe as P9  # noqa: E402

ROOT = P8.ROOT
DEV, DT = P8.DEV, P8.DT
ALPHA = 0.5
TAUS = ("raw", "pca64")


def mix_loglik(X, lab, XtX):
    """固い割り当て lab から 1 回の M 段で μ，π，Σ = (1 − α)S + αI を作り，混合の対数尤度の 1 本あたりの平均を返す。"""
    R = torch.nn.functional.one_hot(torch.as_tensor(lab, device=DEV, dtype=torch.long), 4).to(DT)
    f = P8.lda_from_R(X, R, XtX, ALPHA)
    L = f["chol"]
    logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
    Z = torch.linalg.solve_triangular(L, X.T.contiguous(), upper=False)        # Σ = L Lᵀ，xᵀΣ⁻¹x = |L⁻¹x|²
    quad = (Z ** 2).sum(0)
    dl = P8.delta(X, f) + torch.log(f["pi"].clamp(min=1e-12))[None, :]
    ll = torch.logsumexp(dl, 1) - 0.5 * quad - 0.5 * logdet - 0.5 * X.shape[1] * math.log(2 * math.pi)
    hard = dl.gather(1, R.argmax(1, keepdim=True)).squeeze(1) - 0.5 * quad - 0.5 * logdet - 0.5 * X.shape[1] * math.log(2 * math.pi)
    return float(ll.mean()), float(hard.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default="results/dai9_probe")
    a = ap.parse_args()
    out = ROOT / a.out
    if dai1.git_commit()[1]:
        raise SystemExit("作業木に未コミットの変更がある（コミットしてから流す）")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for (T, c, f, r) in P9.unit_list():
        if r != 0:
            continue
        z = np.load(ROOT / a.src / "rows" / f"{c}_f{f}_r{r}.npz", allow_pickle=False)
        srcs = [x for x in DD.CORPORA if x not in (T, c)]
        u = dai1.Unit(ROOT / "results" / "dai1", T, c, srcs, f, r)
        yp = u.ypool.astype(int)
        if not (yp == z["pool_y"].astype(int)).all():
            raise SystemExit(f"{c}_f{f}：候補のラベルが保存と違う")
        gcol = "speaker_id" if c == "IEMOCAP" else "cluster"
        gp = u.cp.df[gcol].to_numpy()[u.pool].astype(str)
        gt = u.cp.df[gcol].to_numpy()[u.test].astype(str)
        dlg = z["pool_dialogue"].astype(str)
        X0p, X0t = u.Xp["AVT"].to(DEV, DT), u.Xt["AVT"].to(DEV, DT)
        for tau in TAUS:
            Xp, _ = P9.transform(tau, X0p, X0t, gp, gt)
            XtX = Xp.T @ Xp
            T_ = XtX / Xp.shape[0]
            Ach = torch.linalg.cholesky((1 - ALPHA) * T_ + ALPHA * torch.eye(Xp.shape[1], dtype=DT, device=DEV))
            parts = {"emo": yp, "truth": z[f"P_{tau}_emtruth"].astype(int), "rand": z[f"P_{tau}_emrand"].astype(int)}
            row = dict(c=c, fold=f, tau=tau, n=int(Xp.shape[0]), d=int(Xp.shape[1]))
            for nm, lab in parts.items():
                row[f"top3_{nm}"] = P8._top3(Xp, lab, Ach)
                row[f"ll_{nm}"], row[f"llhard_{nm}"] = mix_loglik(Xp, lab, XtX)
            # 上限：白色化した全体の散らばり A⁻¹T の上位 3 つの固有値の和
            ev = torch.linalg.eigvalsh(torch.cholesky_solve(T_, Ach)).sort(descending=True).values
            row["top3_upper"] = float(ev[:3].sum())
            for nm in ("truth", "rand"):
                for ln, lab in (("emo", yp), ("grp", gp), ("dlg", dlg)):
                    row[f"ami_{nm}_{ln}"] = float(AMI(lab, parts[nm], average_method="min"))
            rows.append(row)
            del Xp, XtX
            torch.cuda.empty_cache()
        del u
        torch.cuda.empty_cache()
        print(f"{c}_f{f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "units.csv", index=False)
    import hashlib
    import sklearn
    commit, dirty = dai1.git_commit()
    (out / "provenance.json").write_text(json.dumps(dict(
        src=a.src, alpha=ALPHA, taus=TAUS, script=str(Path(__file__).name), commit=commit, dirty=dirty,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dai8_sha256=hashlib.sha256((HERE / "dai8_probe.py").read_bytes()).hexdigest(),
        dai9_sha256=hashlib.sha256((HERE / "dai9_probe.py").read_bytes()).hexdigest(),
        torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
        finished=__import__("time").strftime("%F %T")), indent=1, ensure_ascii=False))
    print("書いた", out / "units.csv", len(df), "行")


if __name__ == "__main__":
    main()
