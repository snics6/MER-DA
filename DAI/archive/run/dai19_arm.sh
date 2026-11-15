#!/usr/bin/env bash
# 実験 19 の下調べ（DAI/docs/実験19_下調べ_設計.md）．1 つの腕（eq・w・wn）の base → tether → units → students を続けて流す．
# 腕ごとに別の計算機（RunPod の Pod）で並べて流せる．置き場所に依らない（作業木の直下に移ってから動く）．
# 前もって weights の段を 1 回だけ流し，results/dai19_probe/pairstats.npz を置いておく（どの腕も同じ表を読む）．
# w・wn の units の段は eq の腕の tether.json を使うので，それができるまで待つ．
# 終わると logs/dai19_<腕>.done（失敗は logs/dai19_<腕>.failed）．検査を通した台本の sha256 を SHA19 で渡す．
# 使い方：SHA19=<sha256> bash DAI/run/dai19_arm.sh <腕>
set -euo pipefail
cd "$(dirname "$0")/../.."
ARM=${1:-}
case "$ARM" in eq|w|wn) ;; *) echo "腕（eq・w・wn）を渡す"; exit 1 ;; esac
mkdir -p logs
trap 'touch "logs/dai19_${ARM}.failed"' ERR
rm -f "logs/dai19_${ARM}.done" "logs/dai19_${ARM}.failed"
fail() { echo "$1"; touch "logs/dai19_${ARM}.failed"; exit 1; }
[ -n "${SHA19:-}" ] || fail "SHA19 が渡されていない"
[ "$(sha256sum DAI/exp/dai19_probe.py | cut -d' ' -f1)" = "$SHA19" ] || fail "dai19_probe.py の sha256 が違う"
[ -e results/dai19_probe/pairstats.npz ] || fail "results/dai19_probe/pairstats.npz が無い（weights の段を先に流す）"
PY=${PY:-.venv/bin/python}
$PY DAI/exp/dai19_probe.py --stage selftest 2>&1 | tee -a "logs/dai19_${ARM}_selftest.log"
for STAGE in base tether; do
  echo "=== dai19 $ARM $STAGE $(date '+%F %T')"
  $PY DAI/exp/dai19_probe.py --stage "$STAGE" --arm "$ARM" 2>&1 | tee -a "logs/dai19_${ARM}_${STAGE}.log"
done
if [ "$ARM" != eq ]; then
  echo "=== dai19 $ARM：eq の腕の tether.json を待つ $(date '+%F %T')"
  until [ -e results/dai19_probe/eq/tether.json ] || [ -e logs/dai19_eq.failed ]; do sleep 30; done
  [ -e results/dai19_probe/eq/tether.json ] || fail "eq の腕が失敗した"
fi
for STAGE in units students; do
  echo "=== dai19 $ARM $STAGE $(date '+%F %T')"
  $PY DAI/exp/dai19_probe.py --stage "$STAGE" --arm "$ARM" 2>&1 | tee -a "logs/dai19_${ARM}_${STAGE}.log"
done
date '+%F %T' > "logs/dai19_${ARM}.done"
echo "EXIT=0"
