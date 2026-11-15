#!/bin/bash
# E56（docs/exploratory-analyses.md E56）：信頼度の低いソースの発話を外してから E39 の出発点を作る（exp/source_trust39.py）。
# 3 種類を同じ GPU で同時に流し，全部終わったら解析。終わったターゲットは台本が飛ばす。計算が失敗したときだけ止める。
#     tmux new-window -t mer-da -n e56 'cd /home/snics/MER-DA && ./run/e56.sh 2>&1 | tee -a logs/e56.log'
cd /home/snics/MER-DA
export MERDA_MAP_ROOT=results/fixc_map
P=./.venv/bin/python
rm -f logs/e56.done
fail() { echo "FAILED: $*"; echo "exit=1" > logs/e56.done; exit 1; }
echo "== $(date +%F_%T) audio_wavlm・video・text_roberta（同時）"
$P exp/source_trust39.py --kind audio_wavlm > logs/e56_a.log 2>&1 &
PA=$!
$P exp/source_trust39.py --kind video > logs/e56_v.log 2>&1 &
PV=$!
$P exp/source_trust39.py --kind text_roberta > logs/e56_t.log 2>&1 &
PT=$!
wait $PA || fail "audio_wavlm"
wait $PV || fail "video"
wait $PT || fail "text_roberta"
echo "== $(date +%F_%T) 解析"
$P exp/source_trust39.py --analyze > logs/e56_analysis.log 2>&1 || fail "解析"
echo "exit=0" > logs/e56.done
echo "== $(date +%F_%T) 終わり"
