#!/bin/bash
# 言葉系の反復を主な種類にそろえる・列 1（docs/exploratory-analyses.md「言葉系の反復の回数を主な種類にそろえる」）：
# G1 C を固定した地図の K=25・50・全件を反復 5〜29 に → G3 全ソースの合併の C → G4 D1 → G5 teach・herding・random → G6 E39。
# 終わったターゲットは各台本が飛ばすので，止まったらもう一度起動すれば続きから進む。計算が失敗したときだけ止める。終わりに logs/text30_a.done を書く。
#     tmux new-window -t mer-da -n text_a 'cd /home/snics/MER-DA && ./run/text30_a.sh 2>&1 | tee -a logs/text30_a.log'
cd /home/snics/MER-DA
export MERDA_MAP_ROOT=results/fixc_map
P=./.venv/bin/python
TEXT="text_roberta fuse-video-text fuse-audio-text fuse-audio-video-text"
R5=$(seq 5 29); R0=$(seq 0 29)
rm -f logs/text30_a.done
fail() { echo "FAILED: $*"; echo "exit=1" > logs/text30_a.done; exit 1; }
echo "== $(date +%F_%T) G1 C を固定した地図の K=25・50・全件"
for k in $TEXT; do MERDA_OUT_ROOT=results/fixc_text $P exp/fixc_extend.py --kind $k --reps $R5 --K 25 50 all || fail "G1 $k"; done
echo "== $(date +%F_%T) G3 全ソースの合併の C"
for k in $TEXT; do $P exp/multisource_scores.py --kind $k || fail "G3 $k"; done
echo "== $(date +%F_%T) G4 D1"
for k in $TEXT; do $P exp/combine_drafts.py --kind $k --drafts equal --reps $R0 --check || fail "G4 $k"; done
echo "== $(date +%F_%T) G5 teach・herding・herding_feat・random"
for k in $TEXT; do $P exp/select_teach.py --kind $k --reps $R0 || fail "G5 $k"; done
echo "== $(date +%F_%T) G6 E39"
for k in $TEXT; do $P exp/meta5.py --kind $k --reps $R0 || fail "G6 $k"; done
echo "exit=0" > logs/text30_a.done
echo "== $(date +%F_%T) 終わり"
