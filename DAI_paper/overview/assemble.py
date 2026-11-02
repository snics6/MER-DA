"""Assemble DAI_paper/overview/DAI.md from the hand-written front matter and the five extracted parts.

Usage: python3 DAI_paper/overview/assemble.py <parts_dir>
Each block is cut from a part by heading text (start inclusive, end exclusive) and re-leveled so that
its first heading lands at the requested level. Existing {#id} attributes are removed and fresh ASCII ids
are assigned to every heading of level 1-2 (the table of contents shows those two levels).
"""
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
PARTS = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "parts"
H = re.compile(r"^(#{1,6})\s+(.*?)\s*$")


def load(name):
    (p,) = PARTS.glob(name + "*.md")
    return p.read_text(encoding="utf-8").splitlines()


def find(lines, pat, start=0):
    rx = re.compile(pat)
    for i in range(start, len(lines)):
        m = H.match(lines[i])
        if m and rx.search(m.group(2)):
            return i
    raise SystemExit(f"heading not found: {pat}")


def take(lines, start_pat, end_pat=None, level=1, drop_first=False, ren=()):
    i = find(lines, start_pat)
    j = find(lines, end_pat, i + 1) if end_pat else len(lines)
    block = lines[i + 1:j] if drop_first else lines[i:j]
    levels = [len(H.match(l).group(1)) for l in block if H.match(l)]
    shift = level - min(levels) if levels else 0
    out, fence = [], False
    for l in block:
        if l.startswith("```"):
            fence = not fence
        m = None if fence else H.match(l)
        if m:
            text = re.sub(r"\s*\{#[^}]*\}\s*$", "", m.group(2))
            for a, b in ren:
                text = re.sub(a, b, text)
            lv = max(1, min(6, len(m.group(1)) + shift))
            out.append("#" * lv + " " + text)
        else:
            out.append(l)
    while out and out[-1].strip() in ("", "---"):
        out.pop()
    return out


A, B, C, D, E = (load(f"part_{x}") for x in "ABCDE")
front = (HERE / "front.md").read_text(encoding="utf-8").splitlines()
sec9 = (HERE / "sec9_lead.md").read_text(encoding="utf-8").splitlines()
sec10 = (HERE / "sec10_lead.md").read_text(encoding="utf-8").splitlines()

doc = []
doc += front
# 1, 2
doc += take(A, r"^1 設定と記号", r"^2 測り方", 1)
doc += take(A, r"^2 測り方", r"^3 行 1", 1)
# 3 (prior work moves to 9)
doc += take(A, r"^3 行 1", r"^3\.13 ", 1)
doc += take(A, r"^3\.14 ", r"^未記載", 2, ren=[(r"^3\.14 ", "3.13 ")])
# 4 (prior work moves to 9; 4.8 to the appendix)
doc += take(B, r"^4\. 行 2", r"^4\.6 ", 1)
doc += take(B, r"^4\.7 ", r"^4\.8 ", 2, ren=[(r"^4\.7", "4.6")])
# 5
doc += ["# 5 行 3：同じモダリティのラベル無しのターゲットの発話（Claim 4）", ""]
doc += take(C, r"^5\.0 この節の要約", r"^5\.5 ", 2)
doc += take(C, r"^5\.5 ", r"^8（一部）", 2)
doc += take(C, r"^5\.7 ", r"^未記載", 2, ren=[(r"^5\.7", "5.6")])
# 6, 7
doc += take(D, r"^6\. 行 4", r"^6\.6 ", 1)
doc += take(D, r"^7\. 仮のターゲット", r"^8\.（一部）", 1)
# 8
doc += ["# 8 頑健さ", "",
        "この節は，主な結論が設定を替えても残るかを確かめた実験をまとめる．"
        "第 2 のエンコーダ（E62），エンコーダの微調整（E63），アンカーへの L2 正則化の強さ（E82），"
        "テストと同じセッション・対話を学習側から除いた分割（E81）の 4 つである．", ""]
doc += take(D, r"^8\.（一部）", r"^未記載", 2, drop_first=True)
doc += take(C, r"^8（一部）", r"^5\.6 ", 2, ren=[(r"^8（一部）．頑健さ：", "8.3 "), (r"^8\.E82\.", "8.3.")])
doc += take(E, r"^8 節（E81", r"^10 節", 2, ren=[(r"^8 節（E81 の部分）：", "8.4 E81："), (r"^8\.(\d) (?!E81)", r"8.4.\1 ")])
# 9
doc += sec9
doc += take(A, r"^3\.13 ", r"^3\.14 ", 2, ren=[(r"^3\.13 先行研究との関係（行 1）", "9.1 行 1（K 本）の先行研究")])
doc += take(B, r"^4\.6 ", r"^4\.7 ", 2, ren=[(r"^4\.6 先行研究とこの行の新しさ", "9.2 行 2（ソース）の先行研究とこの行の新しさ"), (r"^4\.6\.", "9.2.")])
doc += take(C, r"^5\.6 ", r"^5\.7 ", 2, ren=[(r"^5\.6 先行研究（この行）", "9.3 行 3（同じモダリティのラベル無し）の先行研究"), (r"^5\.6\.", "9.3.")])
doc += take(D, r"^6\.6 ", r"^7\. 仮のターゲット", 2, ren=[(r"^6\.6 行 4 の先行研究", "9.4 行 4（ほかのモダリティ）の先行研究"), (r"^6\.6\.", "9.4.")])
# 10, 11
doc += sec10
doc += take(E, r"^10 節", r"^11 節", 2, drop_first=True)
doc += take(E, r"^11 節", r"^未記載", 1)
# appendix: notes left by the extractors
doc += ["# 付録 抽出のときに見つかった未記載と食い違い", "",
        "5 つの抽出の担当が，出典を写すときに見つけた欠けと食い違いをそのまま載せる．"
        "どの担当の記録かを小見出しに書いた．", ""]
doc += ["## 1〜3 節の担当（設定・測り方・行 1）", ""] + take(A, r"^未記載", None, 3, drop_first=True)
doc += ["## 4 節の担当（行 2）", ""] + take(B, r"^4\.8 ", None, 3, drop_first=True)
doc += ["## 5 節と E82 の担当（行 3）", ""] + take(C, r"^未記載", None, 3, drop_first=True)
doc += ["## 6・7 節と E62・E63 の担当（行 4）", ""] + take(D, r"^未記載", None, 3, drop_first=True)
doc += ["## 8 節の E81・10・11 節の担当", ""] + take(E, r"^未記載", None, 3, drop_first=True)

# fresh ids: level 1 from the section number ("3 行 1" -> s3), level 2 numbered under it
def fix_table_math(l):
    if not l.lstrip().startswith("|"):
        return l
    def rep(m):
        t = m.group(0)
        return t.replace("\\|", "\\Vert{}").replace("|", "\\vert{}")
    return re.sub(r"\$[^$]+\$", rep, l)

def tidy_math(l):
    """Strip spaces just inside inline $...$ so pandoc accepts the delimiters."""
    if "$$" in l:
        return l
    return re.sub(r"\$([^$]+)\$", lambda m: "$" + m.group(1).strip() + "$", l)


n2 = 0
cur = "s"
out, fence, seen = [], False, set()
for l in doc:
    if l.startswith("```"):
        fence = not fence
    m = None if fence else H.match(l)
    if m and len(m.group(1)) <= 2:
        text = m.group(2)
        if len(m.group(1)) == 1:
            text = re.sub(r"^(\d+)\.?\s*(節：)?", r"\1 ", text)
            num = re.match(r"(\d+)", text)
            cur = f"s{num.group(1)}" if num else {"読み方": "read"}.get(text, "app")
            hid, n2 = cur, 0
        else:
            n2 += 1
            mm = re.match(r"(\d+)\.(\d+)\s", text)
            hid = f"s{mm.group(1)}-{mm.group(2)}" if mm and f"s{mm.group(1)}" == cur else f"{cur}-x{n2}"
        assert hid not in seen, hid
        seen.add(hid)
        l = f"{m.group(1)} {text} {{#{hid}}}"
        out.append("")
    elif not fence:
        l = fix_table_math(tidy_math(l))
    out.append(l)
(HERE / "DAI.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("lines", len(out))
