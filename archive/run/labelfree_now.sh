#!/usr/bin/env bash
# E34（ラベル無しで選ぶ herding．予言 L1・L2）を主な 4 種類で流し，解析して logs/labelfree.done を書く。
# E31 と GPU を分け合って流す（2026-10-16 研究者の指示）。続きから流せる。失敗したら止まる。
#     tmux new-window -t mer-da -n lf 'cd /home/snics/MER-DA && ./run/labelfree_now.sh 2>&1 | tee -a logs/labelfree.log'
set -euo pipefail
cd "$(dirname "$0")/.."
export MERDA_MAP_ROOT=results/fixc_map
for k in audio_wavlm audio_hubert video fuse-audio-video; do
  echo "== $(date +%F_%T) select_labelfree --kind $k"
  ./.venv/bin/python exp/select_labelfree.py --kind "$k"
done
echo "== $(date +%F_%T) explore_labelfree"
./.venv/bin/python exp/explore_labelfree.py
date +%F_%T > logs/labelfree.done
