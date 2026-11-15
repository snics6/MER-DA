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

import json
import sys
sys.path.insert(0, str(HERE))
import strip_refs  # noqa: E402

md = strip_refs.strip(SRC.read_text(encoding="utf-8"))

html = subprocess.run(
    ["pandoc", "-f", "markdown+tex_math_dollars+pipe_tables+header_attributes-auto_identifiers",
     "-t", "html5", "--mathjax", "--toc", "--toc-depth=2", "--template", str(TPL),
     "-V", f"date={date}", "-V", f"commit={commit}", "--wrap=none"],
    input=md, capture_output=True, text=True, check=True,
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
            if re.fullmatch(r'(?:\s*<span class="chip [a-z]+">[^<]+</span>\s*．?)+', inner):
                return " " + inner.replace("．", "") + " "
            return b.group(1) + inner + b.group(3)
        t = re.sub(r"(〔|［|\[)([^〔〕［］\[\]<>]*?)(〕|］|\])", in_brackets, t)
        return t
    return re.sub(r">[^<]+<", text_fn, seg)


body_start = html.index("<main>")
html = html[:body_start] + outside_math(html[body_start:], mark)

# ---- figure-first sections: lead (figure + a few sentences) on top, the rest in a fold ----
LEADS = HERE / "leads"
FOLD_LABEL = {
    "s0": "詳しく：0.1 表の数値と限定，0.2 Claim と表の対応，0.3 読むときに注意すること",
    "s2": "詳しく：2.1 模型の式と当てはまり，2.2 位置の損と方向の損の測り方，2.3〜2.5 UAR の差の分け方と判定の決まり",
}


def prerender():
    """Draw the default state of the interactive figures with node, so the page is complete at rest."""
    js = (LEADS / "s2.js").as_posix()
    code = ("const h=require(%s);const s=h.state(+h.DEFAULT.theta.toFixed(1),h.DEFAULT.o);"
            "console.log(JSON.stringify({plane:h.svgPlane(s),line:h.svgLine(s),bar:h.svgBar(s)}))") % json.dumps(js)
    out = subprocess.run(["node", "-e", code], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


svgs = prerender()


def lead_html(sid):
    t = (LEADS / f"{sid}.html").read_text(encoding="utf-8")
    for k, v in svgs.items():
        t = t.replace(f"<!--SVG:{k}-->", v)

    def inline_js(m):
        return "<script>\n" + (LEADS / m.group(1)).read_text(encoding="utf-8") + "</script>"
    return re.sub(r"<!--JS:([\w.]+)-->", inline_js, t)


for sid, label in FOLD_LABEL.items():
    m = re.search(r'<h1 id="%s">.*?</h1>' % sid, html, flags=re.S)
    assert m, sid
    nxt = re.search(r'<h1 id=|<footer>', html[m.end():])
    end = m.end() + nxt.start()
    body = html[m.end():end]
    html = (html[:m.end()] + "\n" + lead_html(sid)
            + f'\n<details class="fold" id="{sid}-fold"><summary>{label}</summary><div class="fold-body">'
            + body + "</div></details>\n" + html[end:])
OUT.write_text(html, encoding="utf-8")
print(OUT, len(html))
