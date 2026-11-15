#!/usr/bin/env bash
# E69 の下調べ（docs/exploratory-analyses.md E69）：AV → AVT の仮のターゲットを解き，判定を出す．実行中にこの台本を書き換えない
set -u
cd /home/snics/MER-DA
export MERDA_MAP_ROOT=results/fixc_map
for w in fuse-audio-video fuse-audio-video-text; do
  ./.venv/bin/python exp/explore_curvature_split.py --world "$w" || { echo "失敗：$w"; exit 1; }
done
./.venv/bin/python exp/explore_curvature_split.py --analyze > logs/e69_analysis.log 2>&1 || { echo "失敗：判定"; exit 1; }
touch logs/e69.done
echo "E69 終わり：$(date '+%F %T')"
