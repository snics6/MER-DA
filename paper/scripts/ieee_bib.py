"""references.bib（正本）から，組版用の references_ieee.bib を作る．

IEEE Computer Society の参考文献の書式で要る項目（全著者名・誌名と会議名の正式名・巻・号・月・年・頁．
docs/venue-tac.md）は残し，会議論文（inproceedings）から編者（editor）・出版社（publisher）・
叢書名（series）と叢書の巻（volume），出版社の所在地（Cham）を外す．誌名・会議名は略さない．
生成物なので手で直さない．正本を直してから `python3 scripts/ieee_bib.py` を流す（make が流す）．
"""
from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "references.bib"
DST = HERE / "references_ieee.bib"

DROP_INPROC = ("editor", "publisher", "series")

def entries(src: str) -> list[tuple[str, str, list[tuple[str, str]]]]:
    """@type{key, field = {...}, ...} を括弧を数えて読む（欄が複数行にまたがってもよい）．コメント行は読み飛ばす．"""
    out = []
    i = 0
    while True:
        m = re.compile(r"^@(\w+)\{([^,\s]+),", re.M).search(src, i)
        if not m:
            return out
        typ, key = m.group(1), m.group(2)
        j = m.end()
        fields = []
        while True:
            f = re.compile(r"\s*(\w+)\s*=\s*").match(src, j)
            if not f:
                end = re.compile(r"\s*\}").match(src, j)
                if not end:
                    raise SystemExit(f"ieee_bib: {key} の欄を読めない（{src[j:j+60]!r}）")
                j = end.end()
                break
            name = f.group(1).lower()
            k = f.end()
            if src[k] == "{":
                depth = 0
                for e in range(k, len(src)):
                    depth += {"{": 1, "}": -1}.get(src[e], 0)
                    if depth == 0:
                        break
                val = src[k:e + 1]
                k = e + 1
            elif src[k] == '"':
                e = src.index('"', k + 1)
                val = src[k:e + 1]
                k = e + 1
            else:
                v = re.compile(r"[\w.]+").match(src, k)
                val = v.group(0)
                k = v.end()
            fields.append((name, val))
            c = re.compile(r"\s*,").match(src, k)
            j = c.end() if c else k
        out.append((typ, key, fields))
        i = j


def brace(v: str) -> str:
    return v[1:-1] if v.startswith("{") and v.endswith("}") else v


def convert(typ: str, key: str, fields: list[tuple[str, str]]) -> str:
    t = typ.lower()
    has_series = any(f == "series" for f, _ in fields)
    out = []
    for f, v in fields:
        if t == "inproceedings" and (f in DROP_INPROC or (f == "volume" and has_series) or (f == "address" and brace(v).startswith("Cham"))):
            continue
        out.append(f"  {f} = {v}")
    return f"@{typ}{{{key},\n" + ",\n".join(out) + "\n}\n"


def main() -> None:
    src = SRC.read_text()
    es = entries(src)
    n_src = len(re.findall(r"^@\w+\{", src, re.M))
    if len(es) != n_src:
        raise SystemExit(f"ieee_bib: 項目 {n_src} 件のうち {len(es)} 件しか読めない")
    head = "% 生成物（scripts/ieee_bib.py が references.bib から作る）．手で直さない．\n\n"
    DST.write_text(head + "\n".join(convert(*e) for e in es))
    print(f"ieee_bib: {len(es)} 件を {DST.name} に書いた．")


if __name__ == "__main__":
    main()
