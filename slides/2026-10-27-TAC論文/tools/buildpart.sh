#!/bin/bash
# 1 つの part だけを試し組みして PNG を出す（ほかの part とビルド用の場所を共有しない）．補足の体裁で組む（\appendix の後）
#   使い方：tools/buildpart.sh <part の名前（parts/ の中のファイル名から .tex を除いたもの）> [PNG を置くディレクトリ]
#   例：tools/buildpart.sh supp_part1 /tmp/xxx/png
#   本編や別の part への \pageref は ?? になる（体裁の確認用）．終わったら _part_<part>.tex と build_part_<part>/ を消してよい
set -euo pipefail
part="$1"
D="$(cd "$(dirname "$0")/.." && pwd)"
cd "$D"
out="${2:-$D/build_part_$part/png}"
src="_part_$part.tex"
sed -n '1,/\\begin{document}/p' main.tex > "$src"
printf '\\appendix\n\\bandheadline\n\\metroset{numbering=counter}\\setbeamertemplate{frame numbering}{\\insertpagenumber/\\insertdocumentendpage}\n\\input{parts/%s}\n\\end{document}\n' "$part" >> "$src"
latexmk -lualatex -interaction=nonstopmode -file-line-error -outdir="build_part_$part" "$src" > "build_part_${part}.log" 2>&1 || true
pdf="build_part_$part/_part_$part.pdf"
if [ ! -f "$pdf" ]; then echo "ビルドに失敗した：build_part_$part/_part_$part.log の ! 行を見ること"; grep -n '^!' "build_part_$part/_part_$part.log" | head; exit 1; fi
grep -n '^!' "build_part_$part/_part_$part.log" | head || true
grep 'Overfull \\vbox' "build_part_$part/_part_$part.log" || true
rm -rf "$out"; mkdir -p "$out"
pdftoppm -png -r 110 "$pdf" "$out/p"
echo "PDF: $D/$pdf（$(pdfinfo "$pdf" | awk '/Pages/{print $2}') ページ）"
echo "PNG: $out"
