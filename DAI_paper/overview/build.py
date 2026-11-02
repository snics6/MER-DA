"""Build DAI_paper/overview/DAI.html from DAI_paper/overview/DAI.md with pandoc.

Usage: python3 DAI_paper/overview/build.py
The markdown uses $...$ / $$...$$ for math; pandoc turns them into MathJax spans.
Post-processing: wrap tables for horizontal scroll, mark ☆ and status labels.
"""
import datetime
import pathlib
import re
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "DAI.md"
OUT = HERE / "DAI.html"
TPL = HERE / "template.html"

commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
date = datetime.date.today().isoformat()

html = subprocess.run(
    ["pandoc", str(SRC), "-f", "markdown+tex_math_dollars+pipe_tables+header_attributes-auto_identifiers",
     "-t", "html5", "--mathjax", "--toc", "--toc-depth=2", "--template", str(TPL),
     "-V", f"date={date}", "-V", f"commit={commit}", "--wrap=none"],
    capture_output=True, text=True, check=True,
).stdout

# pandoc's mathjax mode adds its own script tag only with the default template; ours loads MathJax itself.
html = re.sub(r"<table", '<div class="tw"><table', html)
html = re.sub(r"</table>", "</table></div>", html)


def outside_math(text, fn):
    """Apply fn to text outside <span class="math ...">...</span>."""
    parts = re.split(r'(<span class="math (?:inline|display)">.*?</span>)', text, flags=re.S)
    return "".join(p if p.startswith('<span class="math') else fn(p) for p in parts)


CHIPS = [
    ("照合済み", "ok"),
    ("関門の導入前", "pre"),
    ("未照合（下調べ）", "warn"),
    ("未照合", "warn"),
    ("設計に使ったターゲット", "design"),
    ("E80 の決め直しの前", "design"),
]


def mark(seg):
    # Only touch text nodes, not tag attributes.
    def text_fn(m):
        t = m.group(0)
        t = t.replace("☆", '<span class="star">☆</span>')
        for word, cls in CHIPS:
            t = re.sub(r"「" + re.escape(word) + r"」", f'<span class="chip {cls}">{word}</span>', t)

        def in_brackets(b):
            inner = b.group(2)
            for word, cls in CHIPS:
                inner = re.sub(r"(?<![「>])" + re.escape(word) + r"(?![」<（])", f'<span class="chip {cls}">{word}</span>', inner)
            return b.group(1) + inner + b.group(3)
        t = re.sub(r"(〔|［|\[)([^〔〕［］\[\]<>]*?)(〕|］|\])", in_brackets, t)
        return t
    return re.sub(r">[^<]+<", text_fn, seg)


body_start = html.index("<main>")
html = html[:body_start] + outside_math(html[body_start:], mark)
OUT.write_text(html, encoding="utf-8")
print(OUT, len(html))
