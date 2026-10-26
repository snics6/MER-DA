#!/bin/bash
# E63 の追加の確かめ：中心化を足した微調整（docs/exploratory-analyses.md E63「追加の確かめ」．研究者の指示 2026-10-25）．
# E63 の本番（logs/e63.done）を待ってから，出発点 3 組 → 適応 4 ターゲット → 表．済んだものは飛ばす．
#     tmux new-window -t mer-da -n e63c 'cd /home/snics/MER-DA && ./run/e63c.sh 2>&1 | tee -a logs/e63c.log'
cd /home/snics/MER-DA
export MERDA_MAP_ROOT=results/fixc_map
P=./.venv/bin/python
rm -f logs/e63c.done
NFAIL=0
note() { echo "FAILED（続ける）: $*"; NFAIL=$((NFAIL + 1)); }
echo "== $(date +%F_%T) E63 の本番（logs/e63.done）の終了を待つ"
until [ -f logs/e63.done ]; do sleep 120; done
echo "== $(date +%F_%T) 段 source（中心化）"
for s in IEMOCAP MELD MOSEI; do
  $P exp/finetune_check.py --stage source --set "$s" --center > "logs/e63c_source_${s}.log" 2>&1 || note "source_c $s"
done
echo "== $(date +%F_%T) 段 adapt（中心化）"
for T in IEMOCAP-impro IEMOCAP-script MELD MOSEI; do
  $P exp/finetune_check.py --stage adapt --target "$T" --center > "logs/e63c_adapt_${T}.log" 2>&1 || note "adapt_c $T"
done
echo "== $(date +%F_%T) 段 analyze（中心化）"
$P exp/finetune_check.py --stage analyze --center > logs/e63c_analysis.log 2>&1 || note "analyze_c"
[ "$NFAIL" -eq 0 ] && echo "exit=0" > logs/e63c.done || echo "exit=1 fails=$NFAIL" > logs/e63c.done
echo "== $(date +%F_%T) 終わり（落ちた仕事 $NFAIL 本）"
