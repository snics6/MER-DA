"""付録の「数値の出所」の表を作る：資料の parts/*.tex の \\NUM{key} を集め，key の頭ごとに台帳の source 欄の結果ファイルを並べる．

    cd slides/2026-10-27-TAC論文 && python3 tools/key_sources.py      （figures/appendix_sources.tex を書く）
"""
from __future__ import annotations

import json
import pathlib
import re

DECK = pathlib.Path(__file__).resolve().parents[1]
LEDGER = json.loads((DECK.parents[1] / "paper" / "results" / "numbers.json").read_text())
ROWS_PER_FRAME = 10


def group(key: str) -> str:
    p = key.split("-")
    n = 2 if p[0] in ("data", "map", "tab", "st") else 3
    return "-".join(p[:n])


def tex(s: str) -> str:
    return s.replace("\\", "/").replace("_", r"\_").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")


def main() -> None:
    used: dict[str, set[str]] = {}
    where: dict[str, set[str]] = {}
    for f in sorted((DECK / "parts").glob("*.tex")):
        for k in re.findall(r"\\NUM\{([^}]+)\}", f.read_text()):
            if k not in LEDGER:
                continue
            g = group(k)
            used.setdefault(g, set()).add(k)
            where.setdefault(g, set()).add(f.stem)
    rows = []
    for g in sorted(used):
        files = []
        for k in sorted(used[g]):
            for part in re.split(r"[，、；]|・(?=results/)| と ", LEDGER[k]["source"].split(" | ")[0]):
                part = re.split(r"（| \(| ←| の | から", part.strip())[0].strip()
                if "/" in part and part not in files:
                    files.append(part)
        src = "；".join(files[:2]) + ("；ほか" if len(files) > 2 else "")
        rows.append((g, len(used[g]), ", ".join(sorted(where[g])), src))
    n_keys = sum(len(v) for v in used.values())
    out = [f"% 生成物（tools/key_sources.py）．資料の \\NUM の key {n_keys} 個を {len(rows)} の頭にまとめた．手で直さない"]
    pages = [rows[i:i + ROWS_PER_FRAME] for i in range(0, len(rows), ROWS_PER_FRAME)]
    for i, page in enumerate(pages, 1):
        out.append(r"\begin{frame}{付録 4：数値の出所（" + f"{i}／{len(pages)}" + r"）}")
        out.append(r"  \rl{数値の出所}" + (r"\label{a:sources}" if i == 1 else ""))
        out.append(r"  \vspace{-0.3em}\centering\tiny\renewcommand{\arraystretch}{1.0}")
        out.append(r"  \begin{tabularx}{\textwidth}{@{}l r l X@{}}")
        out.append(r"    \toprule key の頭 & 数 & 使った part & 台帳の出所（結果ファイル） \\ \midrule")
        for g, n, w, s in page:
            out.append(f"    \\texttt{{{tex(g)}}} & {n} & {tex(w)} & \\texttt{{{tex(s).replace('/', '/\\allowbreak{}').replace('_', '_\\allowbreak{}')}}} \\\\")
        out.append(r"    \bottomrule")
        out.append(r"  \end{tabularx}\par")
        out.append(r"  {\raggedright\tiny key の実体（値・書式・出所・確かめた日）は paper/results/numbers.json．数は資料で使った key の数\par}")
        out.append(r"\end{frame}")
    (DECK / "figures" / "appendix_sources.tex").write_text("\n".join(out) + "\n")
    print(f"key_sources: key {n_keys} 個，頭 {len(rows)} 個，{len(pages)} 枚")


if __name__ == "__main__":
    main()
