#!/usr/bin/env bash
# 日本語の下書きを 1 本の PDF にする（pandoc で TeX を書き出し，LuaLaTeX で 2 回組む）．
# 出力：DAI/paper/ja/論文_日本語_下書き.pdf（git の外）
set -euo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
FILES=(00_要旨.md 01_序論.md 02_設定.md 03_混ぜるgain.md 04_切片.md 05_向き.md 06_ラベル無し.md
       07_本番の評価.md 08_3条件.md 09_中間層.md 10_手法と評価.md 11_考察と限界.md 99_申告.md)
pandoc "${FILES[@]}" --standalone \
  --from markdown+tex_math_dollars+pipe_tables \
  --pdf-engine=lualatex \
  --citeproc --bibliography=../references.bib \
  --metadata title="少数ラベルのマルチモーダル感情認識：日本語の下書き 第 1 版" \
  --metadata date="$(date +%Y-%m-%d)" \
  --metadata reference-section-title="参考文献" \
  --metadata link-citations=true \
  -V documentclass=article -V fontsize=10pt -V geometry:margin=20mm \
  -V mainfont="DejaVu Serif" -V monofont="DejaVu Sans Mono" \
  -V CJKmainfont="Noto Serif CJK JP" -V CJKsansfont="Noto Sans CJK JP" -V CJKmonofont="Noto Sans Mono CJK JP" \
  -V colorlinks=true -V linkcolor=blue \
  -H _pdf_header.tex --toc --toc-depth=2 \
  -o "$TMP/ja.tex"
cd "$TMP"
for i in 1 2; do lualatex -interaction=nonstopmode ja.tex > lualatex.log 2>&1 || true; done
grep -n -A3 "^!" ja.log | head -20 || true
echo "missing characters: $(grep -c 'Missing character' ja.log || true)"
[ -s ja.pdf ] || { echo "no pdf"; exit 1; }
cp ja.pdf "$HERE/論文_日本語_下書き.pdf"
echo "wrote $HERE/論文_日本語_下書き.pdf"
