#!/bin/bash
# P4 を出発点にした実験を反復 10〜29 に延ばす（docs/exploratory-analyses.md「P4 を出発点にした実験を反復 30 回に延ばす」）：
# D1（combine_drafts．equal）→ teach・herding・herding_feat・random（select_teach）→ E39 の 243 通り（meta5）→ 解析（extend30）。
# 出力は *_r30 の新しいフォルダ。終わったターゲットは各台本が飛ばすので，途中で止まったらもう一度起動すれば続きから進む。
# 計算が失敗したときだけ止める。終わりに logs/extend30.done を書く。
#     tmux new-window -t mer-da -n ext30 'cd /home/snics/MER-DA && ./run/extend30.sh 2>&1 | tee -a logs/extend30.log'
cd /home/snics/MER-DA
export MERDA_MAP_ROOT=results/fixc_map
P=./.venv/bin/python
R=$(seq 10 29)
KINDS="audio_wavlm audio_hubert video fuse-audio-video"
rm -f logs/extend30.done
fail() { echo "FAILED: $*"; echo "exit=1" > logs/extend30.done; exit 1; }
echo "== $(date +%F_%T) D1"
for k in $KINDS; do $P exp/combine_drafts.py --kind $k --reps $R --drafts equal --out-dir results/combine_${k}_r30 || fail "D1 $k"; done
echo "== $(date +%F_%T) teach・herding・herding_feat・random"
for k in $KINDS; do $P exp/select_teach.py --kind $k --reps $R --no-pseudo --out-dir results/select_teach_${k}_r30 || fail "teach $k"; done
echo "== $(date +%F_%T) E39"
for k in $KINDS; do $P exp/meta5.py --kind $k --reps $R --no-pseudo --out-dir results/meta5_${k}_r30 || fail "E39 $k"; done
echo "== $(date +%F_%T) 解析"
$P exp/extend30.py > logs/extend30_analysis.log 2>&1 || fail "extend30"
echo "exit=0" > logs/extend30.done
echo "== $(date +%F_%T) 終わり"
