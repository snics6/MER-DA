#!/usr/bin/env bash
# 研究者の指示（2026-10-16）：E31 を止めて E33 を先に流し，終わったら E31 を続きから再開する。
# E33：K 本の選び方を変えた仮説転移を主な 4 種類で流し，解析して logs/select.done を書く（続きから流せる）。
# その後 run/compare_all.sh を起動し直す（終わったターゲットと段は飛ばすので，E31 の止まったところから進む）。
#     tmux new-window -t mer-da -n select 'cd /home/snics/MER-DA && ./run/select_now.sh 2>&1 | tee -a logs/select.log'
set -euo pipefail
cd "$(dirname "$0")/.."
export MERDA_MAP_ROOT=results/fixc_map
for k in audio_wavlm audio_hubert video fuse-audio-video; do
  echo "== $(date +%F_%T) select_kshot --kind $k"
  ./.venv/bin/python exp/select_kshot.py --kind "$k"
done
echo "== $(date +%F_%T) explore_select"
./.venv/bin/python exp/explore_select.py
date +%F_%T > logs/select.done
echo "== $(date +%F_%T) E31 を続きから再開"
./run/compare_all.sh 2>&1 | tee -a logs/compare_all.log
