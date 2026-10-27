#!/usr/bin/env python3
"""results/numbers.json から numbers.tex を生成する．

**なぜ台帳を経由させるか．**
ADOS-SR では検査を7本用意したが，最後まで「本文の数値を書き換えても全部通る」穴が残った．
数値と出所を結ぶものが無かったためである．CCMER は実験がまだ動いており数値が変わり続けるので，
この穴は必ず事故になる．そこで **本文の実験由来の数値は \\NUM{key} でしか書けない**ことにし，
key の実体は results/numbers.json（出所つき）に置く．

生成物 numbers.tex は **コミットする**．Overleaf はこのスクリプトを実行できないためである．

使い方: python3 scripts/gen_numbers_tex.py
終了コード: 台帳の形式が壊れていれば 1
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEDGER = ROOT / "results" / "numbers.json"
OUT = ROOT / "numbers.tex"

REQUIRED = ("value", "source", "verified")

HEADER = r"""% 自動生成．手で編集しない．
% 生成元: results/numbers.json ／ 生成器: scripts/gen_numbers_tex.py
% 本文の実験由来の数値は \NUM{key} で書く．未登録の key はビルド時にエラーになる．
\makeatletter
\providecommand{\NUM}[1]{%
  \@ifundefined{NUM@#1}{%
    \GenericError{}{numbers.tex: key '#1' is not registered in results/numbers.json}{}{}%
    \textbf{??}%
  }{\csname NUM@#1\endcsname}%
}
"""

FOOTER = "\\makeatother\n"


def fmt(entry: dict) -> str:
    v = entry["value"]
    f = entry.get("format")
    if f:
        return f % v if "%" in f else f.format(v)
    return str(v)


def main() -> int:
    if not LEDGER.exists():
        OUT.write_text(HEADER + FOOTER, encoding="utf-8")
        print(f"gen_numbers_tex: {LEDGER.relative_to(ROOT)} が無い．空の numbers.tex を書いた．")
        return 0
    try:
        data = json.loads(LEDGER.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"gen_numbers_tex: 台帳が JSON として壊れている: {e}", file=sys.stderr)
        return 1
    lines = [HEADER]
    bad = []
    for key, entry in sorted(data.items()):
        if not isinstance(entry, dict) or any(k not in entry for k in REQUIRED):
            bad.append(key)
            continue
        lines.append(f"\\expandafter\\def\\csname NUM@{key}\\endcsname{{{fmt(entry)}}}\n")
    lines.append(FOOTER)
    if bad:
        print("gen_numbers_tex: 必須項目（value / source / verified）を欠く key: "
              + ", ".join(bad), file=sys.stderr)
        return 1
    OUT.write_text("".join(lines), encoding="utf-8")
    print(f"gen_numbers_tex: {len(data)} 件の数値を numbers.tex に書いた．")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
