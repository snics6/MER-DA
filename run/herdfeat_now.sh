#!/usr/bin/env bash
# E33 の追加 2（特徴だけを合わせる herding．予言 F1）を主な 4 種類で流し，解析して logs/herdfeat.done を書く。
# 研究者の指示（2026-10-16）で E31 を止めて先に流す。終わったら E31 を続きから再開する（終わったターゲットと段は飛ばす）。
#     tmux new-window -t mer-da -n herdfeat 'cd /home/snics/MER-DA && ./run/herdfeat_now.sh 2>&1 | tee -a logs/herdfeat.log'
set -euo pipefail
cd "$(dirname "$0")/.."
export MERDA_MAP_ROOT=results/fixc_map
for k in audio_wavlm audio_hubert video fuse-audio-video; do
  echo "== $(date +%F_%T) select_herding --feat-only --kind $k"
  ./.venv/bin/python exp/select_herding.py --kind "$k" --feat-only
done
echo "== $(date +%F_%T) explore_herdfeat"
./.venv/bin/python exp/explore_herdfeat.py
date +%F_%T > logs/herdfeat.done
echo "== $(date +%F_%T) E31 を続きから再開"
./run/compare_all.sh 2>&1 | tee -a logs/compare_all.log
