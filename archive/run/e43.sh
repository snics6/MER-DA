#!/bin/bash
# E43（docs/exploratory-analyses.md E43）：teach に入れる感情の見込みがどれだけ正確なら得が戻るか（exp/teach_belief.py）。
# W7 → Hu7 → 解析の順に進む。終わったターゲットは台本が飛ばすので，途中で止まったらもう一度起動すれば続きから進む。
# 計算が失敗したときだけ止める。
#     tmux new-window -t mer-da -n e43 'cd /home/snics/MER-DA && ./run/e43.sh 2>&1 | tee -a logs/e43.log'
cd /home/snics/MER-DA
export MERDA_MAP_ROOT=results/fixc_map
P=./.venv/bin/python
KINDS="audio_wavlm audio_hubert"
rm -f logs/e43.done
fail() { echo "FAILED: $*"; echo "exit=1" > logs/e43.done; exit 1; }
for k in $KINDS; do
  echo "== $(date +%F_%T) $k"
  $P exp/teach_belief.py --kind $k || fail "$k"
done
echo "== $(date +%F_%T) 解析"
$P exp/teach_belief.py --analyze > logs/e43_analysis.log 2>&1 || fail "解析"
echo "exit=0" > logs/e43.done
echo "== $(date +%F_%T) 終わり"
