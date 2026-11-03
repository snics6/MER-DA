"""DAI 実験の共通の読み込み：3 モダリティがそろう発話，塊，5 分割，K 本の抜き方，標準化（計画 `DAI/docs/計画_2026-11-04.tex` 8.5 節）。

- コーパス：AFEW，IEMOCAP（即興だけ），MOSEI，MELD。特徴は必ず `measure/data.py` の `load_group` を通して読む（データの取り決め 1〜6）。
- 発話の集合：第 1 エンコーダの音声・顔・言葉の 3 つすべての特徴がある発話（M7）。
- 塊：IEMOCAP はセッション，AFEW は俳優と作品のつながりでまとめた塊，MOSEI と MELD は `speaker_group`（話者の群）。
- 5 分割：塊を壊さずに件数がそろうよう配る（`measure/split.speaker_folds` を塊に当てる．seed 0）。
- MELD は塊で分けたうえで，テストの対話（split と dia の番号）の発話を学習側から除く（M6）。
- K 本：学習側（テスト以外）から感情ごとに K 本を，K について入れ子に抜く。乱数は `common.rng_for` で
  （コーパス，分割，種）から作るので，評価のターゲットに依らず同じ抜き方になる。
- 融合の特徴は，各モダリティの元の特徴をそのままつなぐ（全群の統計で標準化しない．m4）。
  標準化は呼ぶ側が，学習側（候補）の統計で行う（M9）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "measure"))
sys.path.insert(0, str(ROOT / "features"))
import common  # noqa: E402
import data as D  # noqa: E402
from split import speaker_folds  # noqa: E402

CORPORA = ("AFEW", "IEMOCAP", "MOSEI", "MELD")
GROUP = {"AFEW": "AFEW", "IEMOCAP": "IEMOCAP-impro", "MOSEI": "MOSEI", "MELD": "MELD"}
CORPUS_ID = {c: i for i, c in enumerate(CORPORA)}
ENC1 = {"A": "audio_wavlm", "V": "video", "T": "text_roberta"}
ENC2 = {"A": "audio_hubert", "V": "video_clip", "T": "text_deberta"}
KINDS = ("A", "V", "T", "AV", "AT", "VT", "AVT")
STUDENTS = ("A", "V", "T", "AVT")
N_FOLDS = 5
KS = (1, 5, 10, 25)


def others(c: str) -> list[str]:
    return [x for x in CORPORA if x != c]


def _afew_clusters(df: pd.DataFrame) -> pd.Series:
    """AFEW の俳優と作品のつながりでまとめた塊（同じ俳優が出る作品と，同じ作品に出る俳優を 1 つにまとめる）。"""
    man = pd.read_csv(common.META_DIR / "afew_utterance_manifest.csv", dtype=str, keep_default_na=False)
    movie = dict(zip(man["sample_id"], man["movie_title"]))
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for sid, spk in zip(df["sample_id"], df["speaker_id"].astype(str)):
        mv = movie.get(sid, "") or f"nofilm:{sid}"
        parent[find("A:" + spk)] = find("M:" + mv)
    return pd.Series([find("A:" + s) for s in df["speaker_id"].astype(str)], index=df.index)


def clusters(corpus: str, df: pd.DataFrame) -> pd.Series:
    if corpus == "IEMOCAP":
        return df["sample_id"].str.slice(0, 5)
    if corpus == "AFEW":
        return _afew_clusters(df)
    return df["speaker_group"].astype(str)


def dialogue(corpus: str, df: pd.DataFrame) -> pd.Series | None:
    if corpus != "MELD":
        return None
    return df["sample_id"].str.rsplit("_utt", n=1).str[0]


def load_corpus(corpus: str, enc: dict[str, str] = ENC1) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """3 モダリティがそろう発話の表と，モダリティごとの生の特徴（`sample_id` の昇順）を返す。"""
    parts = {m: D.load_group(GROUP[corpus], k) for m, k in enc.items()}
    ids = sorted(set.intersection(*[set(d["sample_id"]) for d, _ in parts.values()]))
    feats = {}
    base = None
    for m, (d, X) in parts.items():
        pos = pd.Series(np.arange(len(d)), index=d["sample_id"].to_numpy())
        ix = pos.loc[ids].to_numpy()
        feats[m] = X[ix].astype(np.float32)
        if base is None:
            base = d.iloc[ix].reset_index(drop=True)
        else:
            assert (d.iloc[ix]["label_4"].to_numpy() == base["label_4"].to_numpy()).all()
    df = base[["sample_id", "speaker_id", "speaker_group", "label_4"]].copy()
    df["cluster"] = clusters(corpus, df).to_numpy()
    dlg = dialogue(corpus, df)
    df["dialogue"] = dlg.to_numpy() if dlg is not None else ""
    df["fold"] = speaker_folds(df["cluster"], k=N_FOLDS, seed=0).to_numpy()
    assert (df["fold"] >= 0).all()
    return df, feats


def kind_matrix(feats: dict[str, np.ndarray], kind: str) -> np.ndarray:
    return np.hstack([feats[m] for m in kind]) if len(kind) > 1 else feats[kind]


def pool_mask(df: pd.DataFrame, fold: int) -> np.ndarray:
    """fold をテストにしたときの学習側（候補）。MELD はテストの対話の発話も除く。"""
    test = df["fold"].to_numpy() == fold
    pool = ~test
    if (df["dialogue"] != "").any():
        bad = set(df.loc[test, "dialogue"])
        pool &= ~df["dialogue"].isin(bad).to_numpy()
    return pool


def k_orders(y: np.ndarray, pool_idx: np.ndarray, corpus: str, fold: int, seed: int) -> dict[int, np.ndarray]:
    """K ごとの K 本の添字（全体の行番号）。感情ごとに学習側を並べ替え，先頭 K 本を取る（K について入れ子）。"""
    rng = common.rng_for("dai-K", CORPUS_ID[corpus], fold, seed)
    perm = {k: rng.permutation(pool_idx[y[pool_idx] == k]) for k in range(4)}
    out = {}
    for K in KS:
        assert all(len(perm[k]) >= K for k in range(4)), (corpus, fold, K)
        out[K] = np.sort(np.concatenate([perm[k][:K] for k in range(4)]))
    return out


def save_splits(out_dir: Path) -> None:
    """塊と 5 分割の表を書き出す（git に入れる）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in CORPORA:
        df, _ = load_corpus(c)
        df[["sample_id", "cluster", "dialogue", "fold", "label_4"]].to_csv(out_dir / f"{c}.csv", index=False)
        print(c, len(df), "塊", df["cluster"].nunique(), "分割ごとの件数", df["fold"].value_counts().sort_index().tolist())


if __name__ == "__main__":
    save_splits(ROOT / "DAI" / "splits")
