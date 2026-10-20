#!/bin/bash
# E41（docs/exploratory-analyses.md E41）：teach が選ぶ前に使うラベルを 1 か所ずつ予測に置き換える（exp/teach_ablate.py）。
# W7 → Hu7 → AV → V → 解析の順に進む。終わったターゲットは台本が飛ばすので，途中で止まったらもう一度起動すれば続きから進む。
# 計算が失敗したときだけ止める。
#     tmux new-window -t mer-da -n e41 'cd /home/snics/MER-DA && ./run/e41.sh 2>&1 | tee -a logs/e41.log'
cd /home/snics/MER-DA
export MERDA_MAP_ROOT=results/fixc_map
P=./.venv/bin/python
KINDS="audio_wavlm audio_hubert fuse-audio-video video"
rm -f logs/e41.done
fail() { echo "FAILED: $*"; echo "exit=1" > logs/e41.done; exit 1; }
for k in $KINDS; do
  echo "== $(date +%F_%T) $k"
  $P exp/teach_ablate.py --kind $k || fail "$k"
done
echo "== $(date +%F_%T) 解析"
$P exp/teach_ablate.py --analyze > logs/e41_analysis.log 2>&1 || fail "解析"
echo "exit=0" > logs/e41.done
echo "== $(date +%F_%T) 終わり"
