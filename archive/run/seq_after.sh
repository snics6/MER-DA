#!/usr/bin/env bash
# E34 の追加（本数をそろえながら分けて選ぶ herding）を，E34（logs/labelfree.done）を待ってから主な 4 種類の全ターゲットで流し，解析する。
# 続きから流せる（済んだターゲットは飛ばす）。失敗したら止まる。終わりに logs/seq.done を書く。
#     tmux new-window -t mer-da -n seqall 'cd /home/snics/MER-DA && ./run/seq_after.sh 2>&1 | tee -a logs/seq.log'
set -euo pipefail
cd "$(dirname "$0")/.."
export MERDA_MAP_ROOT=results/fixc_map
until [ -f logs/labelfree.done ]; do sleep 120; done
for k in audio_wavlm audio_hubert video fuse-audio-video; do
  echo "== $(date +%F_%T) select_sequential --kind $k"
  ./.venv/bin/python exp/select_sequential.py --kind "$k"
done
echo "== $(date +%F_%T) explore_sequential"
./.venv/bin/python exp/explore_sequential.py
date +%F_%T > logs/seq.done
