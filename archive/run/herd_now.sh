#!/usr/bin/env bash
# E33 の追加（herding．予言 G1〜G3）を主な 4 種類で流す（研究者の指示：2026-10-16，GPU を E33・E31 と分け合って今すぐ）。
# 続きから流せる（済んだターゲットは飛ばす）。失敗したら止まる。終わりに logs/herd.done を書く。
#     tmux new-window -t mer-da -n herd 'cd /home/snics/MER-DA && ./run/herd_now.sh 2>&1 | tee -a logs/herd.log'
set -euo pipefail
cd "$(dirname "$0")/.."
export MERDA_MAP_ROOT=results/fixc_map
for k in audio_wavlm audio_hubert video fuse-audio-video; do
  echo "== $(date +%F_%T) select_herding --kind $k"
  ./.venv/bin/python exp/select_herding.py --kind "$k"
done
date +%F_%T > logs/herd.done
