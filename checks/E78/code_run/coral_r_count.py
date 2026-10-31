"""E78 関門 1：CORAL の r の確かめ（設計の文書 7.2 節「各ソースと各ターゲットの候補の発話数が r の 5 倍以上あるか」）。CPU だけ。
台本と同じ規則（n_min = min(候補の数, 最も小さいソースの発話数)，r を外すのは n_min < 5r）で，(種類か世界, 評価のターゲット, 仮のターゲットか評価, 反復) ごとに
使える r を数える。発話一覧とテストの仮面だけを使い，特徴は読まない。"""
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import committee as CM  # noqa: E402
import map as M  # noqa: E402
from data import groups_for, load_any  # noqa: E402

R = (32, 64, 128)
rows = []
for name in ["audio_wavlm", "video", "text_roberta", "fuse-audio-video-text", "fuse-audio-video"]:
    gs = groups_for(name)
    meta = {}
    for g in gs:
        df, _ = load_any(g, name)
        sp = df["speaker_group"].astype(str).reset_index(drop=True)
        person = (M.GROUP_SPEC[g][0] + ":" + sp).to_numpy()
        meta[g] = M.Group(g, np.zeros((len(df), 1)), df["label_4"].to_numpy(), sp, person)
    size = {g: len(meta[g].y) for g in gs}
    for T in gs:
        S = [s for s in gs if CM.corp(s) != CM.corp(T)]
        for r_ in range(10):
            n_c = int((~M.test_mask(meta[T], r_)).sum())
            n_min = min(n_c, min(size[s] for s in S))
            rows.append({"name": name, "target": T, "role": "eval", "unit": T, "rep": r_, "n_cand": n_c, "n_src_min": min(size[s] for s in S),
                         **{f"r{x}": n_min >= 5 * x for x in R}})
        for c in S:
            pool_c = [s for s in S if CM.corp(s) != CM.corp(c)]
            if not pool_c:
                continue
            for r_ in range(5):
                n_c = int((~M.test_mask(meta[c], r_)).sum())
                n_min = min(n_c, min(size[s] for s in pool_c))
                rows.append({"name": name, "target": T, "role": "pseudo", "unit": c, "rep": r_, "n_cand": n_c,
                             "n_src_min": min(size[s] for s in pool_c), **{f"r{x}": n_min >= 5 * x for x in R}})
d = pd.DataFrame(rows)
d.to_csv(Path(__file__).with_suffix(".csv"), index=False)
bad = d[~d.r128 | ~d.r64 | ~d.r32]
print("単位（反復）の数:", len(d), " r=128 を外す単位:", int((~d.r128).sum()), " r=64 を外す:", int((~d.r64).sum()), " r=32 を外す:", int((~d.r32).sum()))
print(bad.groupby(["name", "role", "unit"]).agg(n=("rep", "size"), n_cand_min=("n_cand", "min"), n_src_min=("n_src_min", "min"),
                                                 r128_ok=("r128", "sum"), r64_ok=("r64", "sum")).to_string())
