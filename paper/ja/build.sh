#!/usr/bin/env bash
# 日本語版の論文（main_ja.pdf）と補足資料（supplementary_ja.pdf）を作る．
#   使い方：bash build.sh            （両方）
#           bash build.sh main_ja    （片方だけ）
# 数値は ../numbers.tex，図は ../figures/，文献は ../references.bib と ../IEEEtran.bst を読む．
# 英語版の数値台帳を更新したときは，先に英語版のフォルダで `make numbers` を流しておく．
set -uo pipefail
cd "$(dirname "$0")"

LUALATEX=(lualatex -interaction=nonstopmode -halt-on-error -file-line-error)
targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=(main_ja supplementary_ja)

run_lualatex() {
  local name=$1
  if ! "${LUALATEX[@]}" "$name.tex" > /dev/null 2>&1; then
    echo "!! $name: lualatex が失敗した．ログの該当箇所："
    grep -nE '^! |:[0-9]+: ' "$name.log" | head -20
    exit 1
  fi
}

build() {
  local name=$1
  echo "--- $name ---"
  run_lualatex "$name"
  if ! bibtex "$name" > "$name.bibtex.out" 2>&1; then
    echo "!! $name: bibtex が失敗した．"
    cat "$name.bibtex.out"
    exit 1
  fi
  run_lualatex "$name"
  run_lualatex "$name"
  # 参照が落ち着くまで（最大 3 回）回す．
  for _ in 1 2 3; do
    grep -qE 'Rerun to get|Label\(s\) may have changed' "$name.log" || break
    run_lualatex "$name"
  done
}

report() {
  local name=$1
  local errors undefined overfull underfull pages
  errors=$(grep -cE '^! |:[0-9]+: ' "$name.log")
  undefined=$(grep -ciE 'undefined (reference|citation)|There were undefined' "$name.log")
  overfull=$(grep -c '^Overfull' "$name.log")
  underfull=$(grep -c '^Underfull' "$name.log")
  pages=$(pdfinfo "$name.pdf" 2>/dev/null | awk '/^Pages/{print $2}')
  echo "$name: errors=$errors undefined=$undefined overfull=$overfull underfull=$underfull pages=$pages"
  grep -E 'Warning--' "$name.blg" | sed 's/^/  bibtex: /' | head -10
}

for t in "${targets[@]}"; do build "$t"; done
echo "=== 結果 ==="
for t in "${targets[@]}"; do report "$t"; done
