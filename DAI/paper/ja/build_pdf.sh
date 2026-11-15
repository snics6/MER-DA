#!/usr/bin/env bash
# 日本語の下書きを 1 本の PDF にする（pandoc で TeX を書き出し，LuaLaTeX で 2 回組む）．
# 出力：DAI/paper/ja/論文_日本語_下書き.pdf（git の外）
set -euo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
FILES=(00_要旨.md 01_序論.md 02_関連研究.md 03_データ.md 04_誤りの分け方.md 05_Q1.md 06_Q2.md
       07_Q3.md 08_考察.md 09_限界.md 10_結論.md 99_申告.md)
pandoc "${FILES[@]}" --standalone \
  --from markdown+tex_math_dollars+pipe_tables \
  --pdf-engine=lualatex \
  --citeproc --bibliography=../references.bib \
  --metadata title="少数ラベルのマルチモーダル感情認識：日本語の下書き 第 2 版" \
  --metadata date="$(date +%Y-%m-%d)" \
  --metadata reference-section-title="参考文献" \
  --metadata link-citations=true \
  -V documentclass=article -V fontsize=10pt -V geometry:margin=20mm \
  -V mainfont="DejaVu Serif" -V monofont="DejaVu Sans Mono" \
  -V CJKmainfont="Noto Serif CJK JP" -V CJKsansfont="Noto Sans CJK JP" -V CJKmonofont="Noto Sans Mono CJK JP" \
  -V colorlinks=true -V linkcolor=blue \
  -H _pdf_header.tex --toc --toc-depth=2 \
  -o "$TMP/ja.tex"
FIGDIR="$(cd ../figures && pwd)"
sed -i "s#\.\./figures/#${FIGDIR}/#g" "$TMP/ja.tex"
cd "$TMP"
for i in 1 2; do lualatex -interaction=nonstopmode ja.tex > lualatex.log 2>&1 || true; done
grep -n -A3 "^!" ja.log | head -20 || true
echo "missing characters: $(grep -c 'Missing character' ja.log || true)"
[ -s ja.pdf ] || { echo "no pdf"; exit 1; }
cp ja.pdf "$HERE/論文_日本語_下書き.pdf"
echo "wrote $HERE/論文_日本語_下書き.pdf"
