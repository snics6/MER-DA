#!/bin/bash
# E60（docs/exploratory-analyses.md E60）：モダリティをまたぐ蒸留（exp/multimodal_distill.py）。
# 4 つの世界を同じ GPU で同時に流し，全部終わったら解析。終わったターゲットは台本が飛ばす。計算が失敗したときだけ止める。
#     tmux new-window -t mer-da -n e60 'cd /home/snics/MER-DA && ./run/e60.sh 2>&1 | tee -a logs/e60.log'
cd /home/snics/MER-DA
export MERDA_MAP_ROOT=results/fixc_map
P=./.venv/bin/python
rm -f logs/e60.done
fail() { echo "FAILED: $*"; echo "exit=1" > logs/e60.done; exit 1; }
echo "== $(date +%F_%T) AV・VT・AT・AVT（同時）"
$P exp/multimodal_distill.py --world fuse-audio-video > logs/e60_av.log 2>&1 &
P1=$!
$P exp/multimodal_distill.py --world fuse-video-text > logs/e60_vt.log 2>&1 &
P2=$!
$P exp/multimodal_distill.py --world fuse-audio-text > logs/e60_at.log 2>&1 &
P3=$!
$P exp/multimodal_distill.py --world fuse-audio-video-text > logs/e60_avt.log 2>&1 &
P4=$!
wait $P1 || fail "AV"
wait $P2 || fail "VT"
wait $P3 || fail "AT"
wait $P4 || fail "AVT"
echo "== $(date +%F_%T) 解析"
$P exp/multimodal_distill.py --analyze > logs/e60_analysis.log 2>&1 || fail "解析"
echo "exit=0" > logs/e60.done
echo "== $(date +%F_%T) 終わり"
