#!/bin/bash
# E45（docs/exploratory-analyses.md E45）：ソースの票の見込み q（E44 の C・B）を言い切る形で teach に入れる．q は直さない（exp/teach_committee.py）。
# W7 → Hu7 → 解析の順に進む。終わったターゲットは台本が飛ばす。計算が失敗したときだけ止める。
#     tmux new-window -t mer-da -n e45 'cd /home/snics/MER-DA && ./run/e45.sh 2>&1 | tee -a logs/e45.log'
cd /home/snics/MER-DA
export MERDA_MAP_ROOT=results/fixc_map
P=./.venv/bin/python
KINDS="audio_wavlm audio_hubert"
rm -f logs/e45.done
fail() { echo "FAILED: $*"; echo "exit=1" > logs/e45.done; exit 1; }
for k in $KINDS; do
  echo "== $(date +%F_%T) $k"
  $P exp/teach_committee.py --kind $k || fail "$k"
done
echo "== $(date +%F_%T) 解析"
$P exp/teach_committee.py --analyze > logs/e45_analysis.log 2>&1 || fail "解析"
echo "exit=0" > logs/e45.done
echo "== $(date +%F_%T) 終わり"
