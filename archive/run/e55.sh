#!/bin/bash
# E55（docs/exploratory-analyses.md E55）：ソースの発話ごとの信頼度で信頼度の低い発話を外して出発点を作り直す（exp/source_trust.py）。
# 3 種類を同じ GPU で同時に流し，全部終わったら解析。終わったターゲットは台本が飛ばす。計算が失敗したときだけ止める。
#     tmux new-window -t mer-da -n e55 'cd /home/snics/MER-DA && ./run/e55.sh 2>&1 | tee -a logs/e55.log'
cd /home/snics/MER-DA
export MERDA_MAP_ROOT=results/fixc_map
P=./.venv/bin/python
rm -f logs/e55.done
fail() { echo "FAILED: $*"; echo "exit=1" > logs/e55.done; exit 1; }
echo "== $(date +%F_%T) audio_wavlm・video・text_roberta（同時）"
$P exp/source_trust.py --kind audio_wavlm > logs/e55_a.log 2>&1 &
PA=$!
$P exp/source_trust.py --kind video > logs/e55_v.log 2>&1 &
PV=$!
$P exp/source_trust.py --kind text_roberta > logs/e55_t.log 2>&1 &
PT=$!
wait $PA || fail "audio_wavlm"
wait $PV || fail "video"
wait $PT || fail "text_roberta"
echo "== $(date +%F_%T) 解析"
$P exp/source_trust.py --analyze > logs/e55_analysis.log 2>&1 || fail "解析"
echo "exit=0" > logs/e55.done
echo "== $(date +%F_%T) 終わり"
