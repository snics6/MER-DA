#!/bin/bash
# E59（docs/exploratory-analyses.md E59）：見込みの蒸留（exp/distill_belief.py）。
# 3 種類を同じ GPU で同時に流し，全部終わったら解析。終わったターゲットは台本が飛ばす。計算が失敗したときだけ止める。
#     tmux new-window -t mer-da -n e59 'cd /home/snics/MER-DA && ./run/e59.sh 2>&1 | tee -a logs/e59.log'
cd /home/snics/MER-DA
export MERDA_MAP_ROOT=results/fixc_map
P=./.venv/bin/python
rm -f logs/e59.done
fail() { echo "FAILED: $*"; echo "exit=1" > logs/e59.done; exit 1; }
echo "== $(date +%F_%T) audio_wavlm・video・text_roberta（同時）"
$P exp/distill_belief.py --kind audio_wavlm > logs/e59_a.log 2>&1 &
PA=$!
$P exp/distill_belief.py --kind video > logs/e59_v.log 2>&1 &
PV=$!
$P exp/distill_belief.py --kind text_roberta > logs/e59_t.log 2>&1 &
PT=$!
wait $PA || fail "audio_wavlm"
wait $PV || fail "video"
wait $PT || fail "text_roberta"
echo "== $(date +%F_%T) 解析"
$P exp/distill_belief.py --analyze > logs/e59_analysis.log 2>&1 || fail "解析"
echo "exit=0" > logs/e59.done
echo "== $(date +%F_%T) 終わり"
