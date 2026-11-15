#!/bin/bash
# E53（docs/exploratory-analyses.md E53）：条件 2 の主な比べを言葉（T）で行う（exp/text_cond2.py）。
# 4 ターゲットを 2 つずつに分けて同じ GPU で同時に流し，両方終わったら解析。終わったターゲットは台本が飛ばす。計算が失敗したときだけ止める。
#     tmux new-window -t mer-da -n e53 'cd /home/snics/MER-DA && ./run/e53.sh 2>&1 | tee -a logs/e53.log'
cd /home/snics/MER-DA
export MERDA_MAP_ROOT=results/fixc_map
P=./.venv/bin/python
rm -f logs/e53.done
fail() { echo "FAILED: $*"; echo "exit=1" > logs/e53.done; exit 1; }
echo "== $(date +%F_%T) text_roberta（2 つに分けて同時）"
$P exp/text_cond2.py --target MOSEI IEMOCAP-impro > logs/e53_a.log 2>&1 &
PA=$!
$P exp/text_cond2.py --target MELD IEMOCAP-script > logs/e53_b.log 2>&1 &
PB=$!
wait $PA || fail "MOSEI IEMOCAP-impro"
wait $PB || fail "MELD IEMOCAP-script"
echo "== $(date +%F_%T) 解析"
$P exp/text_cond2.py --analyze > logs/e53_analysis.log 2>&1 || fail "解析"
echo "exit=0" > logs/e53.done
echo "== $(date +%F_%T) 終わり"
