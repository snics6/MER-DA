#!/usr/bin/env bash
# E33：E32 の段 2（logs/combine.done）を待ってから，K 本の選び方を変えた仮説転移を主な 4 種類で流し，解析する。
# 続きから流せる（済んだターゲットは select_kshot.py が飛ばす）。失敗したら止まる。
set -euo pipefail
cd "$(dirname "$0")/.."
export MERDA_MAP_ROOT=results/fixc_map
until [ -f logs/combine.done ]; do sleep 300; done
for k in audio_wavlm audio_hubert video fuse-audio-video; do
  echo "== $(date +%F_%T) select_kshot --kind $k"
  ./.venv/bin/python exp/select_kshot.py --kind "$k"
done
echo "== $(date +%F_%T) explore_select"
./.venv/bin/python exp/explore_select.py
date +%F_%T > logs/select.done
