#!/bin/bash
# E54（docs/exploratory-analyses.md E54）：P4 の生徒で，条件 2 の選び方を A（WavLM）・V・T でそろえて比べる（exp/p4_cond2.py）。
# 3 種類を同じ GPU で同時に流し，全部終わったら解析。終わったターゲットは台本が飛ばす。計算が失敗したときだけ止める。
#     tmux new-window -t mer-da -n e54 'cd /home/snics/MER-DA && ./run/e54.sh 2>&1 | tee -a logs/e54.log'
cd /home/snics/MER-DA
export MERDA_MAP_ROOT=results/fixc_map
P=./.venv/bin/python
rm -f logs/e54.done
fail() { echo "FAILED: $*"; echo "exit=1" > logs/e54.done; exit 1; }
echo "== $(date +%F_%T) audio_wavlm・video・text_roberta（同時）"
$P exp/p4_cond2.py --kind audio_wavlm > logs/e54_a.log 2>&1 &
PA=$!
$P exp/p4_cond2.py --kind video > logs/e54_v.log 2>&1 &
PV=$!
$P exp/p4_cond2.py --kind text_roberta > logs/e54_t.log 2>&1 &
PT=$!
wait $PA || fail "audio_wavlm"
wait $PV || fail "video"
wait $PT || fail "text_roberta"
echo "== $(date +%F_%T) 解析"
$P exp/p4_cond2.py --analyze > logs/e54_analysis.log 2>&1 || fail "解析"
echo "exit=0" > logs/e54.done
echo "== $(date +%F_%T) 終わり"
