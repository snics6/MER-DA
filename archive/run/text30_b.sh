#!/bin/bash
# 言葉系の反復を主な種類にそろえる・列 2：G2 E31 の段 3 を言葉系に（反復 10〜29．K ≤ 25 と K=50・全件）。
# 設計は docs/exploratory-analyses.md「言葉系の反復の回数を主な種類にそろえる」。終わったターゲットは台本が飛ばす。終わりに logs/text30_b.done を書く。
#     tmux new-window -t mer-da -n text_b 'cd /home/snics/MER-DA && ./run/text30_b.sh 2>&1 | tee -a logs/text30_b.log'
cd /home/snics/MER-DA
export MERDA_MAP_ROOT=results/fixc_map
P=./.venv/bin/python
TEXT="text_roberta fuse-video-text fuse-audio-text fuse-audio-video-text"
R=$(seq 10 29)
rm -f logs/text30_b.done
fail() { echo "FAILED: $*"; echo "exit=1" > logs/text30_b.done; exit 1; }
for k in $TEXT; do
  echo "== $(date +%F_%T) G2 $k K ≤ 25"
  $P exp/compare_b1.py --kind $k --parts final --reps $R --tag final || fail "G2 $k final"
  echo "== $(date +%F_%T) G2 $k K=50・全件"
  $P exp/compare_b1.py --kind $k --parts final --reps $R --ks 50 all --tag final_k50all || fail "G2 $k final_k50all"
done
echo "exit=0" > logs/text30_b.done
echo "== $(date +%F_%T) 終わり"
