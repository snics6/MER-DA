#!/usr/bin/env python3
"""査読ループの起動漏れを検出する．

**なぜこれが要るか．**
査読→改稿→記録→報告 と進むと，報告を書いた時点で作業が完結した感覚になり，
最後の「次ラウンドの起動」が落ちる．ADOS-SR では規則を3箇所に書いても6回落ちた．
思い出すことに依存する対策は同じ理由で失敗するので，**改稿のたびに必ず走らせているもの
（ビルド確認）に検出を埋め込み，異常終了で目に入るようにする**．

判定：`reviews/.loop-active` があり，`reviews/round<N>/response.md` が存在し，
`reviews/round<N+1>/` に査読レポートが1つも無ければ「未起動」とみなす．

ループを終えるときは `reviews/.loop-active` を消す．そうすればこの検査は黙る．

使い方: python3 scripts/loop_guard.py
終了コード: 未起動なら 1
"""
import pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
REV = ROOT / "reviews"
MARK = REV / ".loop-active"


def reviewed_rounds():
    """査読レポートが1件でもあるラウンド番号．空のディレクトリを先回りで作っても誤認しない."""
    ns = []
    for d in REV.glob("round*"):
        m = re.fullmatch(r"round(\d+)", d.name)
        if m and d.is_dir() and any(d.glob("review-*.md")):
            ns.append(int(m.group(1)))
    return sorted(ns)


def main() -> int:
    if not MARK.exists():
        print("loop_guard: ループは非アクティブ（reviews/.loop-active が無い）．検査を省略する．")
        return 0
    ns = reviewed_rounds()
    if not ns:
        print("loop_guard: 査読レポートのあるラウンドがまだ無い．")
        return 0
    n = max(ns)
    done = (REV / f"round{n}" / "response.md").exists()
    nxt = REV / f"round{n+1}"
    started = nxt.is_dir() and any(nxt.glob("review-*.md"))
    if done and not started:
        bar = "=" * 72
        print(bar)
        print(f"  ラウンド{n} の改稿と記録は終わっているが，ラウンド{n+1} が起動していない．")
        print()
        print("  報告を書く前に，Agent ツールで review-mer と review-novelty を起動すること．")
        print(f"  出力先は reviews/round{n+1}/．引用に関わる \\cite を変えたなら citation-audit も．")
        print()
        print("  ループを終えるなら:  rm reviews/.loop-active")
        print(bar)
        return 1
    if not done:
        print(f"loop_guard: ラウンド{n} は改稿中（response.md 未作成）．")
    else:
        print(f"loop_guard: ラウンド{n+1} は起動済み．")
    return 0


if __name__ == "__main__":
    sys.exit(main())
