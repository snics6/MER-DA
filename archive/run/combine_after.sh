#!/usr/bin/env bash
# E32 の 5 節 段 2：E31 の計算（logs/compare_all.done）を待ってから，下書きを取り替えた仮説転移を主な 4 種類で流し，解析する。
# 続きから流せる（済んだターゲットは combine_drafts.py が飛ばす）。失敗したら止まる。
set -euo pipefail
cd "$(dirname "$0")/.."
export MERDA_MAP_ROOT=results/fixc_map
until [ -f logs/compare_all.done ]; do sleep 300; done
for k in audio_wavlm audio_hubert video fuse-audio-video; do
  echo "== $(date +%F_%T) combine_drafts --kind $k"
  ./.venv/bin/python exp/combine_drafts.py --kind "$k"
done
echo "== $(date +%F_%T) explore_combine"
./.venv/bin/python exp/explore_combine.py
date +%F_%T > logs/combine.done
