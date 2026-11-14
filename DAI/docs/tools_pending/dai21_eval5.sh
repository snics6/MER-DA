#!/usr/bin/env bash
# 実験 21（DAI/docs/実験21_設計.md）：検算（selftest・check3・check18）→ students4 → cremad → summary を続けて流す．
# 終わると logs/dai21_eval5.done（失敗は logs/dai21_eval5.failed）．検査を通した台本の sha256 を SHA21 で渡す．
# 実験 20 の計算が終わってから DAI/run/ に置いて使う（実験 20 の計算のあいだは DAI/docs/tools_pending/ に置く）．
set -euo pipefail
cd /home/snics/MER-DA
mkdir -p logs
trap 'touch logs/dai21_eval5.failed' ERR
rm -f logs/dai21_eval5.done logs/dai21_eval5.failed
fail() { echo "$1"; touch logs/dai21_eval5.failed; exit 1; }
[ -n "${SHA21:-}" ] || fail "SHA21 が渡されていない"
[ "$(sha256sum DAI/exp/dai21_eval5.py | cut -d' ' -f1)" = "${SHA21:?}" ] || fail "dai21_eval5.py の sha256 が違う"
[ -e logs/dai20_eval.done ] || fail "実験 20 の計算が終わっていない"
[ -e DAI/frozen/dai21_frozen.json ] || fail "固定のファイル DAI/frozen/dai21_frozen.json が無い"
PY=.venv/bin/python
# 入力（実験 1 の集計・ソースの分類器・評価の回の保存，実験 18 の単位と集計，実験 20 の単位と集計，固定のファイル）が書き換わらないことを前後で確かめる
snap() { { find results/dai1/stats results/dai1/source results/dai18_probe/summary results/dai18_probe/tether.json results/dai18_probe/frozen_sel.json \
                results/dai20_eval/summary DAI/frozen -type f -print0 | sort -z | xargs -0 sha256sum;
           find results/dai1/eval/units results/dai18_probe/units results/dai18_probe/students results/dai20_eval/units -type f -printf '%p %s %T@\n' | sort; } | sha256sum | cut -d' ' -f1; }
BEFORE=$(snap)
$PY DAI/exp/dai21_eval5.py --stage selftest 2>&1 | tee -a logs/dai21_selftest.log
$PY DAI/exp/dai21_eval5.py --stage check3 2>&1 | tee -a logs/dai21_check3.log
$PY DAI/exp/dai21_eval5.py --stage check18 --out-dir results/dai21_eval5_check18 2>&1 | tee -a logs/dai21_check18.log
for STAGE in students4 cremad summary; do
  echo "=== dai21 $STAGE $(date '+%F %T')"
  $PY DAI/exp/dai21_eval5.py --stage "$STAGE" 2>&1 | tee -a "logs/dai21_${STAGE}.log"
done
AFTER=$(snap)
[ "$BEFORE" = "$AFTER" ] || fail "入力の置き場か固定のファイルが書き換わった"
date '+%F %T' > logs/dai21_eval5.done
echo "EXIT=0"
