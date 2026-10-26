#!/bin/bash
# E63 の追加の確かめ（中心化）の残り：MELD の適応が GPU のメモリ不足で落ちた（2026-10-26 01:52）ので，e63c の台本が終わってから MELD を反復ごとに別の過程で流し，表を作り直す．
#     tmux new-window -t mer-da -n e63c2 'cd /home/snics/MER-DA && ./run/e63c_rest.sh 2>&1 | tee -a logs/e63c_rest.log'
cd /home/snics/MER-DA
export MERDA_MAP_ROOT=results/fixc_map
P=./.venv/bin/python
rm -f logs/e63c_rest.done
NFAIL=0
note() { echo "FAILED（続ける）: $*"; NFAIL=$((NFAIL + 1)); }
echo "== $(date +%F_%T) e63c（logs/e63c.done）の終了を待つ"
until [ -f logs/e63c.done ]; do sleep 120; done
for T in MELD MOSEI; do                       # 済んだ行は飛ばすので，MOSEI は落ちた分があれば埋まるだけ
  for r in 0 1 2; do
    echo "== $(date +%F_%T) adapt（中心化）$T 反復 $r"
    $P exp/finetune_check.py --stage adapt --target "$T" --reps "$r" --center >> "logs/e63c_adapt_${T}_rest.log" 2>&1 || note "adapt_c $T r$r"
  done
done
echo "== $(date +%F_%T) 表を作り直す"
$P exp/finetune_check.py --stage analyze --center > logs/e63c_analysis.log 2>&1 || note "analyze_c"
[ "$NFAIL" -eq 0 ] && echo "exit=0" > logs/e63c_rest.done || echo "exit=1 fails=$NFAIL" > logs/e63c_rest.done
echo "== $(date +%F_%T) 終わり（落ちた仕事 $NFAIL 本）"
