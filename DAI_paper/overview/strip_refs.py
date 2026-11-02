"""Remove pointers to where a number is recorded (document names, line numbers, checker ids).

The researcher asked (2026-11-03) that the page not show where results are stored.
DAI.md keeps the pointers; only the built page drops them.
"""
import re

CHIP_WORDS = ["照合済み", "関門の導入前", "未照合（下調べ）", "未照合", "設計に使ったターゲット",
              "E80 の決め直しの前", "結果を見た後の探索", "結果を見た後の照合", "関門なし"]
ABBR = (r"(?:EA|RS|RV|PC|RW|SD|PL|VN|V8|MR|RVW|C79|C80p|C80c|LIT|VET|DIAG|G2N|RPT|ML)(?![A-Za-z])")
PATHY = (r"(?:snap/|docs/|data/|paper/|DAI_paper/|research_notes/|research|reports|checks/|exp/|logs/|results/|"
         r"proposal|p\d\.txt|tac-round|step1_|theory_|related-|paper-cl|CLAUDE\.m|audit|explanation/)")
REF_ITEM = re.compile(
    r"^\s*(?:出典[：:]\s*)?(?:同\s|同$|" + ABBR + r"|" + PATHY + r"|"
    r"A\d+-\d+|[A-Z]\d+-\d+|規則-\d+|\d+(?:\.\d+)*\s*節|\d+(?:\.\d+)*\s*結果|"
    r"旧は|HuBERT は EA|HuBERT での E|言葉系の種類ごとの中央値は)"
)
E_ITEM = re.compile(r"^\s*E\d+")


def is_ref(item, strict=False):
    """strict=True is used inside round brackets, where most items are content."""
    plain = re.sub(r"<[^>]+>", "", item).strip()
    if not plain:
        return True
    if any(w == plain or plain.startswith(w + "（") for w in CHIP_WORDS):
        return False
    if re.search(r"\bl\.\s?\d", plain) or re.search(r"\.(?:md|txt|py|log|csv|json|tex)\b", plain):
        return True
    if re.match(r"^\u0003\d+\u0004(?:\s*(?:の\s*)?[\d.]+\s*(?:節|行)[^．]*)?$", plain):
        return True
    if strict:
        return bool(re.match(r"^\s*(?:出典[：:]\s*)?(?:同\s|" + ABBR + r"\s|" + PATHY + r")", plain))
    if REF_ITEM.match(plain):
        return True
    if E_ITEM.match(plain) and not any(w in plain for w in CHIP_WORDS) and "変わる" not in plain:
        return True
    return False


def _filter(inner, strict):
    items = re.split(r"(．|；|;)", inner)
    parts = items[::2]
    kept = [k for k in parts if not is_ref(k, strict)]
    if len(kept) == len(parts):
        return None
    return "．".join(k.strip() for k in kept if k.strip())


def _group(m):
    open_, inner, close = m.group(1), m.group(2), m.group(3)
    r = _filter(inner, strict=False)
    if r is None:
        return m.group(0)
    return open_ + r + close if r else "\u0000"


def _paren(m):
    r = _filter(m.group(1), strict=True)
    if r is None:
        return m.group(0)
    return "（" + r + "）" if r else "\u0000"


def _tables(t):
    lines = t.split("\n")
    out, i = [], 0
    while i < len(lines):
        if not lines[i].lstrip().startswith("|"):
            out.append(lines[i]); i += 1; continue
        j = i
        while j < len(lines) and lines[j].lstrip().startswith("|"):
            j += 1
        block = lines[i:j]
        rows = [r.strip().strip("|").split("|") for r in block]
        head = [c.strip() for c in rows[0]]
        drop = {k for k, c in enumerate(head) if c.startswith("出典") or c in ("出所", "在りか", "所在")}
        refcol = {k for k, c in enumerate(head) if "出典" in c and k not in drop}
        for r_i, r in enumerate(rows):
            cells = []
            for k, c in enumerate(r):
                if k in drop:
                    continue
                if k in refcol and r_i >= 2:
                    f = _filter(c.strip(), strict=False)
                    c = " " + (c.strip() if f is None else f) + " "
                if k in refcol and r_i == 0:
                    c = " " + c.strip().replace("主な出典と状態", "状態").replace("．主な出典", "").replace("と出典", "").replace("・出典", "").replace("出典と", "") + " "
                cells.append(c)
            out.append("|" + "|".join(cells) + "|")
        i = j
    return "\n".join(out)


def strip(text):
    # protect math and code
    keep = []

    def protect(m):
        keep.append(m.group(0))
        return f"\u0001{len(keep) - 1}\u0002"
    t = re.sub(r"\$\$.*?\$\$", protect, text, flags=re.S)
    t = re.sub(r"\$[^$\n]+\$", protect, t)
    def protect_code(m):
        keep.append(m.group(0))
        pathy = "/" in m.group(0) or re.search(r"\.(?:md|py|csv|txt|json|log)`$", m.group(0))
        return (f"\u0003{len(keep) - 1}\u0004" if pathy else f"\u0001{len(keep) - 1}\u0002")
    t = re.sub(r"`[^`\n]+`", protect_code, t)
    t = _tables(t)
    for _ in range(4):  # nested brackets
        t2 = re.sub(r"(\[|［|〔)([^\[\]［］〔〕\n]*)(\]|］|〕)(?!\()", _group, t)
        t2 = re.sub(r"（([^（）\n]*)）", _paren, t2)
        if t2 == t:
            break
        t = t2
    # labels and wrappers left empty
    t = re.sub(r"出典[：:]?\s*\u0000", "\u0000", t)
    t = re.sub(r"（\s*\u0000\s*）", "", t)
    t = re.sub(r"\(\s*\u0000\s*\)", "", t)
    t = re.sub(r"[ \t]*\u0000", "", t)
    t = re.sub(r"^[ \t]*[．，][ \t]*", "", t, flags=re.M)
    t = re.sub(r"^(\s*[-*]\s*)(?:出典[：:]?)?\s*[．。]?\s*$\n", "", t, flags=re.M)
    t = re.sub(r"^\s*出典[：:]?\s*[．。]?\s*", "", t, flags=re.M)
    t = re.sub(r"値は EA [^．\n]*にある．", "", t)
    t = re.sub(r"．\s*．", "．", t)
    t = re.sub(r"，\s*．", "．", t)
    t = re.sub(r"[ \t]+．", "．", t)
    t = re.sub(r"[\u0001\u0003](\d+)[\u0002\u0004]", lambda m: keep[int(m.group(1))], t)
    return t


if __name__ == "__main__":
    import difflib
    import pathlib
    import sys
    src = pathlib.Path(__file__).with_name("DAI.md").read_text(encoding="utf-8")
    out = strip(src)
    for line in difflib.unified_diff(src.splitlines(), out.splitlines(), lineterm="", n=0):
        print(line)
