#!/usr/bin/env python3
"""本文の実験由来の数値が，台帳（results/numbers.json）を経由しているかを検査する．

**なぜ要るか．**本文に生の数値を直書きすると，結果を作り直したときにどこを直せばよいか分からなくなる
（ADOS-SR で最後まで塞げなかった穴）．そこで主張を担う節の数値は \\NUM{key} で書き，
生の数値が残っていればここで落とす．

検査する範囲：要旨と，見出しに次の語を含む節（main.tex と sections/*.tex）．
  introduction / part i（Part I・Part II）/ comparison / discussion / conclusion
関連研究（他人の数値）と設定の節（設計の定数）は検査しない．ただし設定の節でもデータの件数は \\NUM で書く（規則）．

数式の扱い：
  - 地の文の数はすべて検査する（例外は scripts/numbers_allowlist.txt．年・0・1 など）
  - 数式（$...$，\\(...\\)，equation・align などの環境）の中は，小数（例 0.84）だけを検査する．
    整数は式の係数・指数・設定（$K=5$，$1/(2K)$）として許す．**実験の値を数式の中の整数で書かない**
  - \\NUM{...}，\\label・\\ref・\\cite・\\eqref，図表の寸法の命令（\\includegraphics[...] など）の中は見ない

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

GUARDED = ("introduction", "part i", "comparison", "discussion", "conclusion")

NUMBER = re.compile(r"(?<![\\A-Za-z0-9.])\d+(?:\.\d+)?(?![A-Za-z0-9])")
DECIMAL = re.compile(r"(?<![\\A-Za-z0-9.])\d*\.\d+(?![A-Za-z0-9])")
STRIP = [
    re.compile(r"\\NUM\{[^}]*\}"),
    re.compile(r"\\(?:label|ref|cite|eqref|autoref|cref|Cref|pageref|url|href)\{[^}]*\}"),
    re.compile(r"\\(?:includegraphics|resizebox|scalebox|hspace|vspace|setlength|multicolumn|multirow|cline|cmidrule|addlinespace|renewcommand|arraystretch|IEEEPARstart)\*?(?:\s*(?:\[[^\]]*\]|\([^)]*\)|\{[^}]*\}))*"),
    re.compile(r"\\begin\{(?:tabular|tabularx|table|figure)\*?\}(?:\s*(?:\[[^\]]*\]|\{[^}]*\}))*"),
]
INLINE_MATH = re.compile(r"\$[^$]*\$|\\\(.*?\\\)")
DISPLAY_BEGIN = re.compile(r"\\begin\{(equation|align|gather|multline|eqnarray)\*?\}")
DISPLAY_END = re.compile(r"\\end\{(equation|align|gather|multline|eqnarray)\*?\}")


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
    """(節名, 開始行, 本文) を返す．abstract 環境と，見出しに GUARDED の語を含む \\section を拾う."""
    out = []
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.S)
    if m:
        out.append(("abstract", text[: m.start()].count("\n") + 1, m.group(1)))
    secs = list(re.finditer(r"\\section\*?\{([^}]*)\}", text))
    for i, s in enumerate(secs):
        name = s.group(1).strip().lower()
        if not any(g in name for g in GUARDED):
            continue
        end = secs[i + 1].start() if i + 1 < len(secs) else len(text)
        out.append((name, text[: s.start()].count("\n") + 1, text[s.end(): end]))
    return out


def scan(body: str, base: int, name: str, allow) -> list[tuple]:
    hits = []
    in_display = False
    for off, raw in enumerate(body.splitlines()):
        line = re.sub(r"(?<!\\)%.*", "", raw)
        for p in STRIP:
            line = p.sub(" ", line)
        if DISPLAY_BEGIN.search(line):
            in_display = True
        segments = []  # (is_math, text)
        if in_display:
            segments.append((True, line))
        else:
            pos = 0
            for m in INLINE_MATH.finditer(line):
                segments.append((False, line[pos:m.start()]))
                segments.append((True, m.group(0)))
                pos = m.end()
            segments.append((False, line[pos:]))
        if DISPLAY_END.search(line):
            in_display = False
        for is_math, seg in segments:
            pat = DECIMAL if is_math else NUMBER
            for m in pat.finditer(seg):
                tok = m.group(0)
                if not is_math and any(p.search(tok) for p in allow):
                    continue
                hits.append((name, base + off, tok, raw.strip()))
    return hits


def main() -> int:
    allow = load_allow()
    hits = []
    for tex in TEX_FILES:
        if not tex.exists():
            continue
        text = tex.read_text(encoding="utf-8")
        rel = tex.relative_to(ROOT)
        for name, base, body in guarded_regions(text):
            hits += scan(body, base, f"{rel}:{name}", allow)
    if not hits:
        print("check_numbers: 台帳を経由しない数値は無い．")
        return 0
    print(f"check_numbers: 台帳を経由しない数値が {len(hits)} 件ある．")
    for name, line, tok, ctx in hits:
        print(f"  [{name}] 行 {line}: {tok}  |  {ctx[:100]}")
    print("  → results/numbers.json に登録して \\NUM{key} で書くか，")
    print("     主張を担わない数値なら scripts/numbers_allowlist.txt に理由つきで登録する．")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
