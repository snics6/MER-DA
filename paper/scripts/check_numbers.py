#!/usr/bin/env python3
"""本文の実験由来の数値が，台帳（results/numbers.json）を経由しているかを検査する．

**なぜ要るか．**
CCMER は実験がまだ動いており，数値は今後も変わる．本文に生の数値を直書きすると，
実験をやり直したときにどこを直せばよいか分からなくなる（ADOS-SR で最後まで塞げなかった穴）．
そこで **Abstract / Results / Discussion / Conclusion に現れる数値は \\NUM{key} で書く**ことにし，
生の数値が残っていればここで落とす．

例外は scripts/numbers_allowlist.txt に1行1件で登録する（節番号・年・文献値・設定値など）．
書式: `<正規表現>  # 理由`

使い方: python3 scripts/check_numbers.py
終了コード: 台帳を経由しない数値があれば 1
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEX_FILES = [ROOT / "main.tex"] + sorted((ROOT / "sections").glob("*.tex"))
ALLOW = ROOT / "scripts" / "numbers_allowlist.txt"

# 検査する節（ここに書かれた数値だけが主張を担う）
GUARDED = ("abstract", "results", "discussion", "conclusion")

NUMBER = re.compile(r"(?<![\\A-Za-z0-9.])\d+(?:\.\d+)?(?![A-Za-z0-9])")


def load_allow() -> list[re.Pattern]:
    if not ALLOW.exists():
        return []
    pats = []
    for line in ALLOW.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            pats.append(re.compile(line))
    return pats


def guarded_regions(text: str) -> list[tuple[str, int, str]]:
    """(節名, 開始行, 本文) を返す．abstract 環境と \\section を拾う."""
    out = []
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.S)
    if m:
        out.append(("abstract", text[: m.start()].count("\n") + 1, m.group(1)))
    secs = list(re.finditer(r"\\section\{([^}]*)\}", text))
    for i, s in enumerate(secs):
        name = s.group(1).strip().lower()
        if not any(g in name for g in GUARDED):
            continue
        end = secs[i + 1].start() if i + 1 < len(secs) else len(text)
        out.append((name, text[: s.start()].count("\n") + 1, text[s.end(): end]))
    return out


def main() -> int:
    allow = load_allow()
    hits = []
    for tex in TEX_FILES:
        if not tex.exists():
            continue
        text = tex.read_text(encoding="utf-8")
        rel = tex.relative_to(ROOT)
        for name, base, body in guarded_regions(text):
            name = f"{rel}:{name}"
            # コメント行と \NUM{...} / \label / \ref / \cite の中は見ない
            for off, line in enumerate(body.splitlines()):
                line = line.split("%", 1)[0]
                line = re.sub(r"\\NUM\{[^}]*\}", "", line)
                line = re.sub(r"\\(?:label|ref|cite|eqref|autoref)\{[^}]*\}", "", line)
                for m in NUMBER.finditer(line):
                    tok = m.group(0)
                    if any(p.search(tok) for p in allow):
                        continue
                    hits.append((name, base + off, tok, line.strip()))
    if not hits:
        print("check_numbers: 台帳を経由しない数値は無い．")
        return 0
    print(f"check_numbers: 台帳を経由しない数値が {len(hits)} 件ある．")
    for name, ln, tok, line in hits[:40]:
        print(f"  main.tex:{ln} [{name}] {tok}  |  {line[:90]}")
    print("  → results/numbers.json に登録して \\NUM{key} で書くか，")
    print("     主張を担わない数値なら scripts/numbers_allowlist.txt に理由つきで登録する．")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
