"""【探索的】E27：中心合わせの門．ラベル無しで測れる答えの偏り TV で，中心合わせが効く組と損する組を分けられるか。

目的は `docs/exploratory-analyses.md` E27。新しい学習はしない。
- 中心合わせの得：E25 の手 2（中心合わせ＋ソースのみ）− ソースのみ（`OUT/e25_pairs.csv`，K によらない）
- TV（ラベル無し）：ソースのみの分類器がターゲットの全発話に出した答えの割合と 1/4 との総変動距離
  （`results/shift_<kind>/logits/<ソース>.npz` の `t_<ターゲット>`）
- TV（ラベルで均等化）：E11 と同じ（`explore_mechanism.bias_table`．感情ごとの正解率と同じ重みで見た答えの割合）

    MERDA_MAP_ROOT=results/fixc_map MERDA_OUT_ROOT=results/fixc_analysis ./.venv/bin/python exp/explore_gate.py

出力：`OUT/e27_pairs.csv`（組ごとの TV と中心合わせの得），`OUT/e27_corr.csv`，`OUT/e27_gate.csv`（閾値ごとの門の成績），
`OUT/e27_loko.csv`（種類を 1 つ抜いて閾値を決めたときの成績）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
from data import GROUP_SPEC  # noqa: E402
from explore_mechanism import bias_table  # noqa: E402
from pair_table import OUT  # noqa: E402

KINDS = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video",
         "text_roberta", "fuse-audio-text", "fuse-video-text", "fuse-audio-video-text"]
HARM, HELP = -0.005, 0.005          # 中心合わせで 0.5 点以上下がった／上がった


def tv_free(kind: str, source: str, target: str) -> float:
    L = np.load(ROOT / "results" / f"shift_{kind}" / "logits" / f"{source}.npz")[f"t_{target}"]
    p = np.bincount(L.argmax(1), minlength=4) / len(L)
    return 0.5 * float(np.abs(p - 0.25).sum())


def pairs() -> pd.DataFrame:
    e25 = pd.read_csv(OUT / "e25_pairs.csv", dtype={"K": str})
    e25 = e25[e25.K == "5"][["kind", "target", "source", "src", "t", "h2", "h2_minus_src", "loss_offset"]]
    rows = []
    for kind in KINDS:
        b = bias_table(kind).rename(columns={"tv": "tv_bal"})[["target", "source", "tv_bal", "top_pred"]]
        b["kind"] = kind
        b["tv_free"] = [tv_free(kind, s, t) for s, t in zip(b.source, b.target)]
        rows.append(b)
    P = pd.concat(rows).merge(e25, on=["kind", "target", "source"])
    P["same_corpus"] = [GROUP_SPEC[t][0] == GROUP_SPEC[s][0] for t, s in zip(P.target, P.source)]
    P["gain"] = P.h2_minus_src
    return P


def corr(P: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for kind, g in P.groupby("kind", sort=False):
        for excl in (False, True):
            h = g[~g.same_corpus] if excl else g
            r1, p1 = stats.spearmanr(h.tv_free, h.gain, alternative="greater")
            r2, p2 = stats.spearmanr(h.tv_bal, h.gain, alternative="greater")
            r3 = stats.spearmanr(h.tv_free, h.tv_bal)[0]
            rows.append({"kind": kind, "excl_same_corpus": excl, "n": len(h), "rho_tvfree_gain": r1, "p": p1,
                         "rho_tvbal_gain": r2, "p_bal": p2, "rho_tvfree_tvbal": r3,
                         "n_harm": int((h.gain < HARM).sum()), "n_help": int((h.gain > HELP).sum()),
                         "gain_mean": h.gain.mean()})
    return pd.DataFrame(rows)


def gate_curve(P: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for th in [0.0] + sorted(P[col].unique().tolist()) + [1.01]:
        on = P[col] >= th
        pol = np.where(on, P.gain, 0.0)
        rows.append({"col": col, "th": th, "n_on": int(on.sum()), "harm_kept": int(((P.gain < HARM) & on).sum()),
                     "help_lost": int(((P.gain > HELP) & ~on).sum()), "policy_gain_mean": float(pol.mean()),
                     "policy_safe_rate": float((pol >= HARM).mean())})
    return pd.DataFrame(rows)


def loko(P: pd.DataFrame, col: str) -> pd.DataFrame:
    """種類を 1 つ抜き，残りで（平均の得が最大になる）閾値を決め，抜いた種類に当てる。"""
    rows = []
    for kind in P.kind.unique():
        tr, te = P[P.kind != kind], P[P.kind == kind]
        cv = gate_curve(tr, col)
        th = float(cv.loc[cv.policy_gain_mean.idxmax(), "th"])
        on = te[col] >= th
        pol = np.where(on, te.gain, 0.0)
        rows.append({"kind": kind, "col": col, "th": th, "n": len(te), "n_on": int(on.sum()),
                     "always_gain": te.gain.mean(), "gated_gain": float(pol.mean()),
                     "always_safe": float((te.gain >= HARM).mean()), "gated_safe": float((pol >= HARM).mean()),
                     "harm_kept": int(((te.gain < HARM) & on).sum()), "help_lost": int(((te.gain > HELP) & ~on).sum())})
    return pd.DataFrame(rows)


def main() -> int:
    pd.set_option("display.width", 250)
    P = pairs()
    P.to_csv(OUT / "e27_pairs.csv", index=False)
    C = corr(P)
    C.to_csv(OUT / "e27_corr.csv", index=False)
    print("相関\n", C.round(3).to_string(index=False))
    Q = P[~P.same_corpus]
    print(f"\n同じコーパスの組を除く：{len(Q)} 組．中心合わせで 0.5 点以上下がった組 {int((Q.gain < HARM).sum())}，上がった組 {int((Q.gain > HELP).sum())}")
    print(Q[Q.gain < HARM][["kind", "target", "source", "src", "gain", "tv_free", "tv_bal"]].round(3).to_string(index=False))
    G = pd.concat([gate_curve(Q, "tv_free"), gate_curve(Q, "tv_bal")])
    G.to_csv(OUT / "e27_gate.csv", index=False)
    for col in ("tv_free", "tv_bal"):
        g = G[G.col == col]
        print(f"\n門の曲線（{col}．同じコーパスの組を除く）：")
        print(g.iloc[:: max(1, len(g) // 12)].round(3).to_string(index=False))
    L = pd.concat([loko(Q, "tv_free"), loko(Q, "tv_bal")])
    L.to_csv(OUT / "e27_loko.csv", index=False)
    print("\n種類を 1 つ抜いて閾値を決めたとき\n", L.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
