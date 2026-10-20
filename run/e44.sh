#!/bin/bash
# E44（docs/exploratory-analyses.md E44）：ソースごとの分類器の票を癖でまとめた見込み q の正確さ（exp/committee.py）。
# W7 → Hu7 → AV → V → 解析の順に進む。終わったターゲットは台本が飛ばす。計算が失敗したときだけ止める。
#     tmux new-window -t mer-da -n e44 'cd /home/snics/MER-DA && ./run/e44.sh 2>&1 | tee -a logs/e44.log'
cd /home/snics/MER-DA
export MERDA_MAP_ROOT=results/fixc_map
P=./.venv/bin/python
KINDS="audio_wavlm audio_hubert fuse-audio-video video"
rm -f logs/e44.done
fail() { echo "FAILED: $*"; echo "exit=1" > logs/e44.done; exit 1; }
for k in $KINDS; do
  echo "== $(date +%F_%T) $k"
  $P exp/committee.py --kind $k || fail "$k"
done
echo "== $(date +%F_%T) 解析"
$P exp/committee.py --analyze > logs/e44_analysis.log 2>&1 || fail "解析"
echo "exit=0" > logs/e44.done
echo "== $(date +%F_%T) 終わり"
