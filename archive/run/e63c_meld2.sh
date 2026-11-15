#!/bin/bash
# E63 の追加の確かめ（中心化）：MELD の反復 1（残り 1 本）と反復 2 を流し，表を作り直す（2026-10-27）．
# 前の過程は GPU メモリの空きブロックを溜めて上限に触り，評価が 5〜10 倍遅くなったので止めた．exp/finetune_check.py は GPU メモリを 6 割に抑え，評価のたびに空きブロックを返すようにした．
#     tmux new-window -t mer-da -n e63c4 'cd /home/snics/MER-DA && ./run/e63c_meld2.sh 2>&1 | tee -a logs/e63c_meld2.log'
cd /home/snics/MER-DA
export MERDA_MAP_ROOT=results/fixc_map
P=./.venv/bin/python
rm -f logs/e63c_meld2.done
NFAIL=0
note() { echo "FAILED（続ける）: $*"; NFAIL=$((NFAIL + 1)); }
for r in 1 2; do
  echo "== $(date +%F_%T) adapt（中心化）MELD 反復 $r"
  $P exp/finetune_check.py --stage adapt --target MELD --reps "$r" --center >> logs/e63c_adapt_MELD_rest.log 2>&1 || note "adapt_c MELD r$r"
done
echo "== $(date +%F_%T) 表を作り直す"
$P exp/finetune_check.py --stage analyze --center > logs/e63c_analysis.log 2>&1 || note "analyze_c"
[ "$NFAIL" -eq 0 ] && echo "exit=0" > logs/e63c_meld2.done || echo "exit=1 fails=$NFAIL" > logs/e63c_meld2.done
echo "== $(date +%F_%T) 終わり（落ちた仕事 $NFAIL 本）"
