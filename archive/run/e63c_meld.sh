#!/bin/bash
# E63 の追加の確かめ（中心化）：MELD の 3 反復（メモリ不足で落ち，その後 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True が WSL2 で「CUDA driver error: unknown error」を起こして 2 度落ちた．この設定は使わない）を反復ごとに別の過程で流し，表を作り直す（2026-10-26）．
#     tmux new-window -t mer-da -n e63c3 'cd /home/snics/MER-DA && ./run/e63c_meld.sh 2>&1 | tee -a logs/e63c_meld.log'
cd /home/snics/MER-DA
export MERDA_MAP_ROOT=results/fixc_map
P=./.venv/bin/python
rm -f logs/e63c_meld.done
NFAIL=0
note() { echo "FAILED（続ける）: $*"; NFAIL=$((NFAIL + 1)); }
for r in 0 1 2; do
  echo "== $(date +%F_%T) adapt（中心化）MELD 反復 $r"
  $P exp/finetune_check.py --stage adapt --target MELD --reps "$r" --center >> logs/e63c_adapt_MELD_rest.log 2>&1 || note "adapt_c MELD r$r"
done
echo "== $(date +%F_%T) 表を作り直す"
$P exp/finetune_check.py --stage analyze --center > logs/e63c_analysis.log 2>&1 || note "analyze_c"
[ "$NFAIL" -eq 0 ] && echo "exit=0" > logs/e63c_meld.done || echo "exit=1 fails=$NFAIL" > logs/e63c_meld.done
echo "== $(date +%F_%T) 終わり（落ちた仕事 $NFAIL 本）"
