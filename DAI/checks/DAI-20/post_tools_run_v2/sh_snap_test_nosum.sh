#!/usr/bin/env bash
set -euo pipefail
cd /home/snics/MER-DA
fail() { echo "止まった：$1"; exit 1; }
SMALL="results/dai1/stats results/dai1/source results/dai1/select results/dai18_probe/summary results/dai18_probe/source results/dai18_probe/tether.json
       results/dai18_probe/frozen_sel.json results/dai18_probe/splits_new.csv DAI/frozen"
LARGE="results/dai1/eval/units results/dai16_probe/units results/dai18_probe/units results/dai18_probe/students results/dai18_probe/base results/dai20_eval/units"
for P in $SMALL $LARGE; do [ -e "$P" ] || fail "入力の置き場が無い：$P"; done
snap() { { find $SMALL -type f -print0 | sort -z | xargs -0 sha256sum;
           find $LARGE -type f -printf '%p %s %T@\n' | sort; } | sha256sum | cut -d' ' -f1; }
BEFORE=$(snap)
echo "snap：$BEFORE"
