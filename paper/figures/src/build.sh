#!/bin/bash
# 模式図（TikZ の standalone 文書）を作り直す．使い方：bash build.sh [名前 ...]（既定は overview method）
# 中間生成物は一時フォルダに置き，figures/<名前>.pdf と確認用の figures/<名前>.png（200 dpi）だけを書く．
set -euo pipefail
cd "$(dirname "$0")"
names=("$@")
[ ${#names[@]} -eq 0 ] && names=(overview method)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
for n in "${names[@]}"; do
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$tmp" "$n.tex" > "$tmp/$n.stdout" \
    || { grep -A6 '^!' "$tmp/$n.log"; exit 1; }
  cp "$tmp/$n.pdf" "../$n.pdf"
  pdftoppm -r 200 -png -singlefile "../$n.pdf" "../$n"
  echo "$n: $(pdfinfo "../$n.pdf" | awk '/Page size/ {printf "%.2f x %.2f in", $3/72, $5/72}')"
done
