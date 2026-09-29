"""話者分割が成り立つかを調べる（マニフェストだけで走る）。

**data/README.md 注意点 4「話者は必ず分ける」が実際に守れるかは，コーパスごとに事情が違う。**
このコードは測定ではなく**診断**であり，分割の設計を決めるために実態を出す。

## 漏れを見つける 2 つの規則

1. **合成 id 規則（精密）。**`speaker_id` を `_` で切った語の 1 つが，それ自体
   別の `speaker_id` として存在するなら，2 つは同じ人物を含む。
   例：MELD の `Ross_and_Rachel` には `Ross` が含まれる。**`Ross` を取り置いても
   `Ross_and_Rachel` が学習側に残れば Ross の声と顔が漏れる**
2. **系列規則（粗い・要目視）。**名前の幹（先頭の語）が同じものをまとめる。
   AFEW の `speaker_id` は作品名なので，`HarryPotter_*` のように続編が並ぶと
   **同じ俳優が別の id として両側に出る**。
   コーパス共通の接頭辞（`CRE_` `RAV_`）を拾わないよう，幹は 6 文字以上の
   英字で，かつ全体の半分未満を占めるものに限る

## 分割の見通しとして出すもの

- **80% を占めるのに必要な話者数。**少ないほど偏っており，分割を増やせない
- **4 ラベルが揃う話者の数。**少数事例の取り置きはクラスあたりで数えるので，
  ここが少ないと評価側にクラスが欠ける
- **5 分割したときの評価側の件数の最小〜最大。**開きが大きいほど分散が出る

使い方:

    ./.venv/bin/python measure/check_speakers.py
    ./.venv/bin/python measure/check_speakers.py --corpus MELD --verbose
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "features"))
import common  # noqa: E402

OUT_DIR = common.ROOT / "results"
STEM_MIN_LEN = 6          # 幹として認める最短の長さ（CRE・RAV を落とす）
STEM_MAX_SHARE = 0.5      # 全体の半分以上を占める幹はコーパス共通の接頭辞とみなす


def composite_overlaps(ids: list[str]) -> dict[str, list[str]]:
    """語の 1 つが別の id そのものである合成 id を見つける（精密な規則）。"""
    single = set(ids)
    hit = defaultdict(list)
    for i in ids:
        parts = [p for p in i.split("_") if p]
        if len(parts) < 2:
            continue
        for p in parts:
            if p in single and p != i:
                hit[p].append(i)
    return {k: sorted(v) for k, v in hit.items()}


def franchise_groups(ids: list[str]) -> dict[str, list[str]]:
    """名前の幹が同じものをまとめる（粗い規則．目視で確かめる）。"""
    stem = defaultdict(list)
    for i in ids:
        head = re.split(r"[_\-]", i)[0]
        if len(head) >= STEM_MIN_LEN and head.isalpha():
            stem[head.lower()].append(i)
    return {k: sorted(v) for k, v in stem.items()
            if len(v) > 1 and len(v) < len(ids) * STEM_MAX_SHARE}


def fold_sizes(df: pd.DataFrame, k: int = 5) -> tuple[int, int, int]:
    """話者を件数の多い順に k 個の袋へ配る。評価側の件数の最小・最大・
    4 ラベルが揃わない袋の数を返す。"""
    vc = df["speaker_id"].astype(str).value_counts()
    bags: list[list[str]] = [[] for _ in range(k)]
    load = np.zeros(k)
    for spk, n in vc.items():                 # 大きい順に，いちばん軽い袋へ
        j = int(load.argmin())
        bags[j].append(spk)
        load[j] += n
    sizes, short = [], 0
    for b in bags:
        sub = df[df["speaker_id"].astype(str).isin(b)]
        sizes.append(len(sub))
        if sub["label_4"].nunique() < 4:
            short += 1
    return min(sizes), max(sizes), short


def audit(corpus: str, verbose: bool) -> dict:
    df = common.load_manifest(corpus)
    sp = df["speaker_id"].astype(str)
    vc = sp.value_counts()
    ids = sorted(sp.unique())

    # 80% を占めるのに必要な話者数
    cum = (vc.cumsum() / len(df)).to_numpy()
    n80 = int(np.searchsorted(cum, 0.80) + 1)

    comp = composite_overlaps(ids)
    fr = franchise_groups(ids)
    n_comp = len({x for v in comp.values() for x in v})
    n_comp_utt = int(sp.isin({x for v in comp.values() for x in v}).sum())

    full = sum(1 for _, g in df.groupby("speaker_id") if g["label_4"].nunique() == 4)
    lo, hi, short = fold_sizes(df, 5)

    if verbose and comp:
        print(f"\n[{corpus}] 合成 id（語が別の id そのもの）:")
        for k, v in sorted(comp.items(), key=lambda x: -len(x[1])):
            print(f"  {k} ⊂ {v}")
    if verbose and fr:
        print(f"\n[{corpus}] 幹を共有する id（系列の疑い・要目視）:")
        for k, v in sorted(fr.items(), key=lambda x: -len(x[1])):
            print(f"  {k}: {v}")

    return {
        "corpus": corpus,
        "発話": len(df),
        "話者": len(ids),
        "80%に要る話者": n80,
        "4ラベル揃う話者": full,
        "合成idの数": n_comp,
        "合成idの発話": n_comp_utt,
        "系列の疑い": sum(len(v) for v in fr.values()),
        "5分割の評価側": f"{lo}〜{hi}",
        "ラベル欠けの袋": short,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="all")
    ap.add_argument("--verbose", action="store_true", help="漏れの候補を全部並べる")
    args = ap.parse_args()

    targets = list(common.CORPORA_ALL) if args.corpus == "all" else [args.corpus]
    out = pd.DataFrame([audit(c, args.verbose) for c in targets])

    print("\n=== 話者の実態 ===")
    print(out.to_string(index=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / "speaker_audit.csv"
    out.to_csv(dest, index=False)
    print(f"\n書いた: {dest}")


if __name__ == "__main__":
    main()
