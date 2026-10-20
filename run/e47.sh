#!/bin/bash
# E47（docs/exploratory-analyses.md E47）：ラベルが付くたびに見込み q を直す方法 4 つのオン・オフ 16 通り（exp/belief_correct.py）。
# W7 → Hu7 → AV → V → 解析の順に進む。終わったターゲットは台本が飛ばす。計算が失敗したときだけ止める。
#     tmux new-window -t mer-da -n e47 'cd /home/snics/MER-DA && ./run/e47.sh 2>&1 | tee -a logs/e47.log'
cd /home/snics/MER-DA
export MERDA_MAP_ROOT=results/fixc_map
P=./.venv/bin/python
KINDS="audio_wavlm audio_hubert fuse-audio-video video"
rm -f logs/e47.done
fail() { echo "FAILED: $*"; echo "exit=1" > logs/e47.done; exit 1; }
for k in $KINDS; do
  echo "== $(date +%F_%T) $k"
  $P exp/belief_correct.py --kind $k || fail "$k"
done
echo "== $(date +%F_%T) 解析"
$P exp/belief_correct.py --analyze > logs/e47_analysis.log 2>&1 || fail "解析"
echo "exit=0" > logs/e47.done
echo "== $(date +%F_%T) 終わり"
