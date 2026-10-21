#!/bin/bash
# E50（docs/exploratory-analyses.md E50）：既存の能動学習の方法を E39 の生徒で測り直す（exp/al_student39.py）。
# W7 → Hu7 → AV → V → 解析の順に進む。終わったターゲットは台本が飛ばす。計算が失敗したときだけ止める。
#     tmux new-window -t mer-da -n e50 'cd /home/snics/MER-DA && ./run/e50.sh 2>&1 | tee -a logs/e50.log'
cd /home/snics/MER-DA
export MERDA_MAP_ROOT=results/fixc_map
P=./.venv/bin/python
KINDS="audio_wavlm audio_hubert fuse-audio-video video"
rm -f logs/e50.done
fail() { echo "FAILED: $*"; echo "exit=1" > logs/e50.done; exit 1; }
for k in $KINDS; do
  echo "== $(date +%F_%T) $k"
  $P exp/al_student39.py --kind $k || fail "$k"
done
echo "== $(date +%F_%T) 解析"
$P exp/al_student39.py --analyze > logs/e50_analysis.log 2>&1 || fail "解析"
echo "exit=0" > logs/e50.done
echo "== $(date +%F_%T) 終わり"
