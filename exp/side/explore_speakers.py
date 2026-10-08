"""【探索的】MH6：位置のずれはコーパス全体で 1 つではなく話者ごとに違うか（E14）。

ソースのみの分類器がターゲットに出した 4 つの点数（`results/shift_<kind>/logits/`）と，ターゲットの話者を使う。
新しい学習はしない。

量（組 × 感情の対 (k, l) ごと．発話が 8 本以上ある話者だけを使う）
- z = 点数_k − 点数_l，r = z − （正解の感情ごとの z の平均）（感情で説明できない残り）
- 話者の割合 S：r のばらつきのうち，話者ごとの平均 u_j で説明できる割合（分散分析の η²）。話者の数と発話の数から
  偶然で出る分 (J − 1)/(N − 1) を差し引いて補正する。6 対で平均する
- 話者ごとに位置をそろえたときの成績（点数の上でそろえる．学び直さない）
  - ラベルを使う版：点数から，その話者の「感情で説明できない残り」の平均を引く（話者のずれだけを取り除く上限）
  - ラベルを使わない版：点数から，その話者の全発話の点数の平均を引き，コーパス全体の平均を足す（実際に使える操作）
- 話者ごとの成績のばらつき：話者ごとの UAR（2 本以上ある感情だけ）の標準偏差

予言（MH6）
1. 悪い組（ソースのみの UAR の組に固有の部分が低い組）ほど，S と話者ごとの成績のばらつきが大きい
2. コーパス全体で中心をそろえた後（E12 の center）に残る損は，S が大きい組ほど大きい
3.（参考）話者ごとに位置をそろえると，点数の上でも成績が上がる

    ./.venv/bin/python exp/side/explore_speakers.py

出力：`results/explore_relation/e14_*.csv`・`fig_e14.png`
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]   # exp/side/ から見たリポジトリの根
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import shift  # noqa: E402
from data import groups_for, load_any  # noqa: E402
from explore_mechanism import partial_rank  # noqa: E402
from pair_table import OUT, SIX, fit  # noqa: E402
from summarize_map import C_MERGED, C_SRC, C_TGT, INK, INK2, MUTED, SURFACE, _style  # noqa: E402

KINDS = ("audio_wavlm", "video", "text_roberta", "fuse-audio-video", "fuse-audio-text",
         "fuse-video-text", "fuse-audio-video-text")
PAIRS = [(k, j) for k in range(4) for j in range(k + 1, 4)]
MIN_UTT = 8
GOOD_SPK = ["IEMOCAP-impro", "IEMOCAP-script", "RAVDESS", "CREMA-D"]   # 話者が十分わかるターゲット


def speaker_share(z: np.ndarray, y: np.ndarray, spk: np.ndarray) -> float:
    r = z.copy()
    for c in np.unique(y):
        r[y == c] -= z[y == c].mean()
    ss = float((r ** 2).sum())
    if ss <= 0:
        return np.nan
    J, N = len(np.unique(spk)), len(r)
    between = sum(float((spk == j).sum()) * r[spk == j].mean() ** 2 for j in np.unique(spk))
    eta = between / ss
    chance = (J - 1) / (N - 1)
    return (eta - chance) / (1 - chance)


def per_speaker_uar(pred: np.ndarray, y: np.ndarray, spk: np.ndarray) -> np.ndarray:
    out = []
    for j in np.unique(spk):
        m = spk == j
        recalls = [float((pred[m & (y == c)] == c).mean()) for c in range(4) if (m & (y == c)).sum() >= 2]
        if len(recalls) >= 2:
            out.append(np.mean(recalls))
    return np.array(out)


def measure(kind: str) -> pd.DataFrame:
    groups = groups_for(kind)
    meta = {g: load_any(g, kind)[0] for g in groups}
    lab = pd.read_csv(ROOT / "results" / f"shift_{kind}" / "labeling.csv", comment="#")
    src_uar = lab.set_index(["target", "source"]).uar_t
    rows = []
    for s in groups:
        z = np.load(ROOT / "results" / f"shift_{kind}" / "logits" / f"{s}.npz")
        for t in groups:
            if t == s:
                continue
            L = z[f"t_{t}"]
            y = meta[t].label_4.to_numpy()
            spk_all = meta[t].speaker_group.astype(str).to_numpy()
            cnt = pd.Series(spk_all).value_counts()
            keep = np.isin(spk_all, cnt.index[cnt >= MIN_UTT])
            Lk, yk, sk = L[keep], y[keep], spk_all[keep]
            shares = []
            for k, j in PAIRS:
                m = (yk == k) | (yk == j)
                c2 = pd.Series(sk[m]).value_counts()
                mm = m & np.isin(sk, c2.index[c2 >= MIN_UTT])
                if mm.sum() >= 2 * MIN_UTT and len(np.unique(sk[mm])) >= 2:
                    shares.append(speaker_share(Lk[mm, k] - Lk[mm, j], yk[mm], sk[mm]))
            # 点数の上で話者ごとに位置をそろえる
            cls_mean = np.stack([Lk[yk == c].mean(0) for c in range(4)])
            R = Lk - cls_mean[yk]
            L_oracle, L_free = Lk.copy(), Lk.copy()
            grand = Lk.mean(0)
            for j in np.unique(sk):
                m = sk == j
                L_oracle[m] -= R[m].mean(0)
                L_free[m] = L_free[m] - Lk[m].mean(0) + grand
            pred = Lk.argmax(1)
            spu = per_speaker_uar(pred, yk, sk)
            rows.append({"kind": kind, "target": t, "source": s, "src_uar": float(src_uar.get((t, s), np.nan)),
                         "n_speakers": int(len(np.unique(sk))), "share_utt_kept": float(keep.mean()),
                         "spk_share": float(np.nanmean(shares)) if shares else np.nan,
                         "uar_kept": shift.uar(yk, pred),
                         "uar_spk_oracle": shift.uar(yk, L_oracle.argmax(1)),
                         "uar_spk_free": shift.uar(yk, L_free.argmax(1)),
                         "spk_uar_sd": float(spu.std(ddof=1)) if len(spu) >= 3 else np.nan,
                         "spk_uar_min": float(spu.min()) if len(spu) else np.nan})
    return pd.DataFrame(rows)


def center_uar(kind: str) -> pd.Series | None:
    fs = sorted((ROOT / "results" / f"mech_{kind}").glob("*.csv"))
    if not fs:
        return None
    m = pd.concat([pd.read_csv(f, comment="#", dtype={"K": str, "source": str}) for f in fs])
    c = m[(m.op == "center") & (m.method == "src_only")]
    return c.groupby(["target", "source"]).uar.mean()


def main() -> int:
    D = pd.concat([measure(k) for k in KINDS], ignore_index=True)
    D["gain_oracle"] = D.uar_spk_oracle - D.uar_kept
    D["gain_free"] = D.uar_spk_free - D.uar_kept
    D.to_csv(OUT / "e14_pairs.csv", index=False)
    rows = []
    for kind, h in D.groupby("kind", sort=False):
        cu = center_uar(kind)
        if cu is not None:
            h = h.assign(center_uar=[cu.get((t, s), np.nan) for t, s in zip(h.target, h.source)])
        settings = [(kind, sorted(h.target.unique())),
                    (f"{kind} speaker-rich targets", [g for g in GOOD_SPK if g in set(h.target)])]
        if kind == "audio_wavlm":
            settings.insert(1, ("audio_wavlm 6 groups", SIX))
        for name, ts in settings:
            x = h[h.target.isin(ts)]
            if name.endswith("6 groups"):
                x = x[x.source.isin(SIX)]
            if x.target.nunique() < 2:
                continue
            row = {"setting": name, "n_pairs": len(x), "n_targets": x.target.nunique(),
                   "mean_spk_share": float(x.spk_share.mean()),
                   "mean_gain_oracle": float(x.gain_oracle.mean()), "mean_gain_free": float(x.gain_free.mean()),
                   "n_up_free": int((x.gain_free > 0).sum())}
            for resp, q in (("src_uar", "spk_share"), ("src_uar", "spk_uar_sd"), ("center_uar", "spk_share"),
                            ("gain_oracle", "spk_share"), ("gain_free", "spk_share")):
                if resp not in x or x[resp].isna().all():
                    continue
                pr = partial_rank(x, resp, q)
                row[f"r_{resp}~{q}"] = pr["r_partial_rank"] if pr else np.nan
                row[f"p_{resp}~{q}"] = pr["p"] if pr else np.nan
                row[f"df_{resp}~{q}"] = pr["df"] if pr else np.nan
            f = fit(x.dropna(subset=["spk_share"]), "spk_share")
            row["spk_share_target"], row["spk_share_source"], row["spk_share_pair"] = f["target"], f["source"], f["pair"]
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "e14_summary.csv", index=False)
    by_t = D.groupby(["kind", "target"]).agg(spk_share=("spk_share", "mean"), n_speakers=("n_speakers", "first"),
                                             share_utt_kept=("share_utt_kept", "first"),
                                             gain_oracle=("gain_oracle", "mean"), gain_free=("gain_free", "mean")).reset_index()
    by_t.to_csv(OUT / "e14_targets.csv", index=False)
    fig(D, OUT / "fig_e14.png")
    print(f"書いた: {OUT}")
    return 0


def fig(D: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), facecolor=SURFACE)
    col = {"audio_wavlm": C_MERGED, "video": C_TGT, "fuse-audio-video": C_SRC}
    ax = axes[0]
    _style(ax)
    for k, c in col.items():
        h = D[D.kind == k]
        ax.scatter(h.spk_share, h.src_uar, s=14, color=c, label=k, zorder=3)
    ax.set_xlabel("share of within-emotion score spread due to speakers", fontsize=8, color=INK2)
    ax.set_ylabel("source-only UAR on target", fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("MH6: speaker offsets vs source-only accuracy", fontsize=8.5, color=INK, loc="left")
    ax = axes[1]
    _style(ax)
    for k, c in col.items():
        h = D[D.kind == k]
        ax.scatter(h.spk_share, h.gain_free, s=14, color=c, zorder=3)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("speaker share", fontsize=8, color=INK2)
    ax.set_ylabel("UAR change from per-speaker centring of scores", fontsize=8, color=INK2)
    ax.set_title("label-free per-speaker centring (scores, no relearning)", fontsize=8.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
