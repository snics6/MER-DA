#!/bin/bash
# E46（docs/exploratory-analyses.md E46）：ラベル無しで選ぶ既存の方法を同じ手順・同じ学習器で解く（exp/select_baselines.py）。
# W7 → Hu7 → AV → V → 解析の順に進む。終わったターゲットは台本が飛ばす。計算が失敗したときだけ止める。
#     tmux new-window -t mer-da -n e46 'cd /home/snics/MER-DA && ./run/e46.sh 2>&1 | tee -a logs/e46.log'
cd /home/snics/MER-DA
export MERDA_MAP_ROOT=results/fixc_map
P=./.venv/bin/python
KINDS="audio_wavlm audio_hubert fuse-audio-video video"
rm -f logs/e46.done
fail() { echo "FAILED: $*"; echo "exit=1" > logs/e46.done; exit 1; }
for k in $KINDS; do
  echo "== $(date +%F_%T) $k"
  $P exp/select_baselines.py --kind $k || fail "$k"
done
echo "== $(date +%F_%T) 解析"
$P exp/select_baselines.py --analyze > logs/e46_analysis.log 2>&1 || fail "解析"
echo "exit=0" > logs/e46.done
echo "== $(date +%F_%T) 終わり"
