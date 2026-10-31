"""マニフェストの監査。**生データと突き合わせて，全項目を確かめる。**

前身から引き継いだマニフェストには，実際に次の誤りが埋まっていた。

* **IEMOCAP の台本セッションのラベルが芝居の番号で付けられていた**（5,255 件＝57%）
* **IEMOCAP で fru（苛立ち）を Angry に潰していた**（1,849 件）
* **MELD の `sample_id` が 2,576 件重複していた**（分割名が入っていなかった）
* **AFEW の話者が作品名だった**（俳優名が公式注釈にあるのに使っていなかった）

**生成コードが無いと，こうした誤りは永久に見つからない。**本コードは
`features/build_manifest_*.py` が作ったマニフェストを，生データと照合する。

## 確かめること

| | 内容 | 落ちたら |
|---|---|---|
| 1 | `sample_id` が一意か | **致命的。**特徴の突き合わせが壊れる |
| 2 | 必要な列が揃っているか | 致命的 |
| 3 | `label_4` が 0〜3 で，名前と一致するか | 致命的 |
| 4 | 媒体（映像・音声）のファイルが実在するか | 抽出が失敗する |
| 5 | 媒体のパスが重複していないか | **別の発話が同じ動画を指している** |
| 6 | 話者が空でないか・構造はどうか | data/README.md 取り決め 5 に関わる |
| 7 | 書き起こしに文字化けが無いか | 言葉の特徴が壊れる |
| 8 | 生データの件数と一致するか | 取りこぼし |

使い方:

    ./.venv/bin/python measure/audit_manifest.py
    ./.venv/bin/python measure/audit_manifest.py --corpus MELD --full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "features"))
import common  # noqa: E402

OUT_DIR = common.ROOT / "results"

REQUIRED = ("sample_id", "corpus", "split", "speaker_id", "label_4",
            "path_video", "transcript", "exclude")

def raw_count(corpus: str) -> int | None:
    """生データ側の「マニフェストに入るはずの件数」を，独立に数える。

    **マニフェストの総数と比べられる数でなければ意味がない。**たとえば MELD の
    `Raw/` には 15,908 本の動画があるが，注釈が付いているのは 13,708 本だけで，
    残りは注釈ファイルに現れない。**注釈の側を数える。**
    """
    P = Path
    try:
        if corpus == "IEMOCAP":     # 注釈ファイルの発話行の数
            import re
            pat = re.compile(r"^\[[\d.]+ - [\d.]+\]\s+\S+\s+\w+\s+\[")
            n = 0
            for f in P("/mnt/d/Databases/IEMOCAP_full_release").glob(
                    "Session*/dialog/EmoEvaluation/*.txt"):
                n += sum(1 for line in f.read_text(errors="replace").splitlines()
                         if pat.match(line))
            return n
        if corpus == "MELD":        # 注釈 CSV の行数の合計
            return sum(len(pd.read_csv(f)) for f in
                       sorted(P("/mnt/d/Databases/MELD/Labels").glob("*_sent_emo.csv")))
        if corpus == "AFEW":        # 7 感情のディレクトリにある avi
            return sum(1 for _ in P("/mnt/d/Databases/AFEW").glob("*/*/*.avi"))
        if corpus == "MOSEI":       # 発話一覧の行数
            return len(pd.read_csv("/mnt/d/Databases/CMU-MOSEI/label/label.csv"))
        if corpus == "RAVDESS":     # 話し声・音声のみ・全感情
            return sum(1 for f in P("/mnt/d/Databases/RAVDESS/audio_speech").rglob("*.wav")
                       if (q := f.stem.split("-")) and len(q) == 7
                       and q[0] == "03" and q[1] == "01")
        if corpus == "CREMA-D":     # 全感情
            return sum(1 for _ in P("/mnt/d/Databases/CREMA-D/AudioWAV").glob("*.wav"))
    except Exception as e:          # 生データが無い環境でも監査は続ける
        print(f"  ({corpus} の生データを数えられない: {e})")
    return None

# Windows-1252 が壊れて混入した文字（data/README.md 4 節「文字化けの修復（L1）」）
MOJIBAKE = tuple(chr(c) for c in range(0x80, 0xA0))


def check(name: str, ok: bool, detail: str = "") -> dict:
    mark = "○" if ok else "**×**"
    return {"項目": name, "判定": mark, "詳細": detail}


def audit(corpus: str, full: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = common.META_DIR / f"{corpus.lower()}_utterance_manifest.csv"
    if not path.is_file():
        return pd.DataFrame([check(f"{corpus} のマニフェスト", False, "ファイルが無い")]), pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    res = []

    # 1. sample_id の一意性
    dup = int(df["sample_id"].duplicated().sum())
    res.append(check("sample_id が一意", dup == 0, f"重複 {dup} 件"))

    # 2. 必要な列
    miss = [c for c in REQUIRED if c not in df.columns]
    res.append(check("必要な列が揃う", not miss, f"欠け: {miss}" if miss else "全 8 列"))

    # 3. label_4 の整合
    lab = df["label_4"].dropna()
    bad_v = int((~lab.isin([0, 1, 2, 3])).sum())
    if "label_4_name" in df.columns:
        m = df.dropna(subset=["label_4"])
        bad_n = int(sum(common.LABELS[int(a)] != str(b)
                        for a, b in zip(m["label_4"], m["label_4_name"])))
    else:
        bad_n = -1
    res.append(check("label_4 が 0〜3", bad_v == 0, f"範囲外 {bad_v} 件"))
    res.append(check("label_4 と名前が一致", bad_n == 0, f"食い違い {bad_n} 件"))

    # 4・5. 媒体
    for col in ("path_video", "path_audio"):
        if col not in df.columns:
            continue
        p = df[col].fillna("").astype(str)
        has = p.str.len() > 0
        if full:
            miss_f = int(sum(1 for x in p[has] if not Path(x).is_file()))
        else:     # 既定は先頭 300 件だけ確かめる（全件は遅い）
            miss_f = int(sum(1 for x in p[has].head(300) if not Path(x).is_file()))
        res.append(check(f"{col} が実在", miss_f == 0,
                         f"あり {int(has.sum())}/{len(df)}・不在 {miss_f}"
                         + ("" if full else "（先頭 300 件を確認）")))
        d2 = int(p[has].duplicated().sum())
        res.append(check(f"{col} が重複しない", d2 == 0, f"重複 {d2} 件"))

    # 6. 話者
    sp = df["speaker_id"].fillna("").astype(str)
    res.append(check("speaker_id が空でない", int((sp.str.strip() == "").sum()) == 0,
                     f"話者 {sp.nunique()} 人"))

    # 7. 書き起こしの文字化け
    tr = df["transcript"].fillna("").astype(str)
    n_moji = int(sum(any(ch in s for ch in MOJIBAKE) for s in tr))
    # **文字化けは生データの性質であって，マニフェストの誤りではない。**
    # マニフェストは生の写しであるべきなので，ここでは数えるだけにする
    res.append({"項目": "書き起こしの文字化け", "判定": "—",
                "詳細": (f"{n_moji} 件（{n_moji / max(len(df), 1):.1%}）"
                         "・build_text.py が修復する" if n_moji else "0 件")})

    # 8. 生データの件数
    expect = raw_count(corpus)
    if expect:
        res.append(check("生データの件数と一致", len(df) == expect,
                         f"マニフェスト {len(df)} / 生データ {expect}"))

    # 内訳
    use = df[df["label_4"].notna()]
    dist = pd.DataFrame([{
        "corpus": corpus,
        "全行": len(df),
        "4クラス": len(use),
        **{common.LABELS[i]: int((use["label_4"] == i).sum()) for i in range(4)},
        "話者": sp.nunique(),
        "4クラスの話者": use["speaker_id"].nunique(),
    }])
    return pd.DataFrame(res).assign(corpus=corpus), dist


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="all")
    ap.add_argument("--full", action="store_true", help="媒体の実在を全件確かめる（遅い）")
    args = ap.parse_args()

    targets = list(common.CORPORA_ALL) if args.corpus == "all" else [args.corpus]
    checks, dists = [], []
    for c in targets:
        r, d = audit(c, args.full)
        checks.append(r)
        if len(d):
            dists.append(d)

    allc = pd.concat(checks, ignore_index=True)
    print("=== 監査 ===")
    for c in targets:
        sub = allc[allc["corpus"] == c]
        if not len(sub):
            continue
        ng = int((sub["判定"] == "**×**").sum())
        print(f"\n--- {c}  {'**問題 ' + str(ng) + ' 件**' if ng else '問題なし'}")
        print(sub[["項目", "判定", "詳細"]].to_string(index=False))

    if dists:
        print("\n=== 内訳 ===")
        print(pd.concat(dists, ignore_index=True).to_string(index=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    allc.to_csv(OUT_DIR / "manifest_audit.csv", index=False)
    print(f"\n書いた: {OUT_DIR / 'manifest_audit.csv'}")
    ng = int((allc["判定"] == "**×**").sum())
    print(f"**問題の総数: {ng}**")


if __name__ == "__main__":
    main()
