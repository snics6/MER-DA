"""AFEW の発話一覧（マニフェスト）を生データだけから作る。

**前身の CSV は参照しない。**`/mnt/d/Databases/AFEW/` の avi と XML だけを読む。

## AFEW の構造

    TRAIN/<感情>/<番号>.avi     TRAIN/Train_6.xml
    VAL/<感情>/<番号>.avi       VAL/Val_6.xml

**ラベルはディレクトリ名である。**Angry / Happy / Neutral / Sad の 4 つを採り，
Disgust・Fear・Surprise は採らない。

## 番号と XML の対応（本リポジトリで突き止めた）

XML の `<Sequence>` には番号が書かれていないが，**`StartTime` から区切り文字を
取り除くとファイル名になる。**

    <StartTime>00:01:23,880</StartTime>   →   000123880.avi

VAL の 383 件すべてでこの対応が成り立つ。これで各 avi に XML の情報が結び付く。

## なぜこれが重要か ── **俳優名が取れる**

XML には `NameOfActor` がある。

    <Person AgeOfActor="38" AgeOfCharacter="35" Gender="Male"
            NameOfActor="Brad Pitt" Pose="Non-Frontal"/>

前身のマニフェストは `speaker_id` に**作品名**を入れていた。作品名では
**続編に同じ俳優が出ると学習側と評価側の両方に現れて漏れる**（data/README.md 注意点 4 の違反）。
**俳優名を使えばこの問題は消える。**作品をまたいで同じ俳優を同一人物として扱える。

## XML の既知の壊れ

一部の行で `AgeOfActor` と `NameOfActor` が入れ替わっている。

    <Person AgeOfActor="Eddie Jemison" ... NameOfActor="40" .../>

**`NameOfActor` が数字なら入れ替える。**

使い方:

    ./.venv/bin/python features/build_manifest_afew.py --check
    ./.venv/bin/python features/build_manifest_afew.py
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

import common

ROOT = Path("/mnt/d/Databases/AFEW")

# 分割名と，ディレクトリ・XML の対応
SPLITS = {"train": ("TRAIN", "Train_6.xml"), "val": ("VAL", "Val_6.xml")}

EMO4 = {"Angry": 3, "Happy": 1, "Neutral": 0, "Sad": 2}


def clip_id(start_time: str) -> str:
    """`00:01:23,880` → `000123880`。区切りを取り除くとファイル名になる。"""
    return re.sub(r"[:,]", "", start_time.strip())


def read_xml(path: Path) -> dict[str, dict]:
    """XML を clip の番号で引けるようにして返す。"""
    out: dict[str, dict] = {}
    root = ET.parse(path).getroot()
    for seq in root.findall("Sequence"):
        st = seq.findtext("StartTime") or ""
        if not st:
            continue
        p = seq.find("Person")
        actor = (p.get("NameOfActor") or "") if p is not None else ""
        age = (p.get("AgeOfActor") or "") if p is not None else ""
        # 一部の行で年齢と名前が入れ替わっている。数字なら取り換える
        if actor.strip().isdigit():
            actor, age = age, actor
        out[clip_id(st)] = {
            "movie_title": seq.get("MovieTitle") or "",
            "actor_name": actor.strip(),
            "actor_age": age.strip(),
            "gender": (p.get("Gender") or "") if p is not None else "",
            "pose": (p.get("Pose") or "") if p is not None else "",
            "length": seq.findtext("Length") or "",
            "start_time": st.strip(),
        }
    return out


def build() -> pd.DataFrame:
    rows = []
    for split, (sub, xml_name) in SPLITS.items():
        meta = read_xml(ROOT / sub / xml_name)
        for emo_dir in sorted((ROOT / sub).iterdir()):
            if not emo_dir.is_dir():
                continue
            emo = emo_dir.name
            for avi in sorted(emo_dir.glob("*.avi")):
                cid = avi.stem
                m = meta.get(cid, {})
                actor = m.get("actor_name", "")
                rows.append({
                    "sample_id": f"{split}_{cid}",
                    "corpus": "AFEW",
                    "split": split,
                    # **俳優名を話者にする。**取れなければ作品名に落とす
                    "speaker_id": (f"AFEW_{actor.replace(' ', '_')}" if actor
                                   else f"AFEW_film_{m.get('movie_title', cid)}"),
                    "speaker_source": "actor" if actor else "film",
                    "label_4": EMO4.get(emo),
                    "label_4_name": common.LABELS[EMO4[emo]] if emo in EMO4 else "",
                    "emotion_raw": emo,
                    "dialog_type": "",
                    "clip_id": cid,
                    "movie_title": m.get("movie_title", ""),
                    "actor_name": actor,
                    "actor_age": m.get("actor_age", ""),
                    "gender": m.get("gender", ""),
                    "pose": m.get("pose", ""),
                    "start_time": m.get("start_time", ""),
                    "marker_side": "",
                    "path_video": str(avi),
                    "path_audio": "",
                    "transcript": "",
                    "exclude": "false",
                })
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> None:
    print(f"\n=== AFEW: {len(df)} 本 ===")
    print(f"sample_id の重複: {int(df['sample_id'].duplicated().sum())} 件")

    n_x = int((df["actor_name"].str.len() > 0).sum())
    print(f"XML と対応がついた: {n_x} / {len(df)}")
    if n_x < len(df):
        miss = df[df["actor_name"] == ""]
        print(f"**対応がつかない: {len(miss)} 本**")
        print(miss[["sample_id", "emotion_raw", "clip_id"]].head(10).to_string(index=False))

    print("\n感情 × 分割:")
    print(pd.crosstab(df["emotion_raw"], df["split"],
                      margins=True, margins_name="計").to_string())

    use = df[df["label_4"].notna()]
    print(f"\n4 クラスに入る: {len(use)}")
    print(pd.crosstab(use["label_4_name"], use["split"],
                      margins=True, margins_name="計").to_string())

    print(f"\n俳優 {df['actor_name'].nunique()} 人 / 作品 {df['movie_title'].nunique()} 本")
    print("\n**作品をまたいで出ている俳優**（作品名を話者にすると漏れる相手）:")
    g = df[df["actor_name"] != ""].groupby("actor_name")["movie_title"].nunique()
    multi = g[g > 1].sort_values(ascending=False)
    print(f"  {len(multi)} 人 / のべ {int(df['actor_name'].isin(multi.index).sum())} 本")
    print(multi.head(10).to_string())

    print("\n1 本しか持たない俳優:",
          int((df[df['actor_name'] != ''].groupby('actor_name').size() == 1).sum()), "人")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if not ROOT.is_dir():
        raise SystemExit(f"生データが無い: {ROOT}")
    df = build()
    report(df)
    if not args.check:
        out = common.META_DIR / "afew_utterance_manifest.csv"
        df.to_csv(out, index=False)
        print(f"\n書いた: {out}")


if __name__ == "__main__":
    main()
