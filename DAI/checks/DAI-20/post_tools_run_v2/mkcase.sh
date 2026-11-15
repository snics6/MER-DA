#!/usr/bin/env bash
# 作り物の置き場の写し（単位は記号リンク）を作る：mkcase.sh <名前>
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd)/fake_eval
D=$(cd "$(dirname "$0")" && pwd)/cases/$1
rm -rf "$D"; mkdir -p "$D/units" "$D/summary"
for f in "$B"/units/*; do ln -s "$f" "$D/units/$(basename "$f")"; done
cp "$B"/summary/* "$D/summary/"
echo "$D"
