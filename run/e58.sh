#!/bin/bash
# E58（docs/exploratory-analyses.md E58）：出発点の向きの誤り E_0 の幾何を測る（exp/geom_e0.py）。
# 3 種類を同じ GPU で同時に流し，全部終わったら解析。終わったターゲットは台本が飛ばす。計算が失敗したときだけ止める。
#     tmux new-window -t mer-da -n e58 'cd /home/snics/MER-DA && ./run/e58.sh 2>&1 | tee -a logs/e58.log'
cd /home/snics/MER-DA
export MERDA_MAP_ROOT=results/fixc_map
P=./.venv/bin/python
rm -f logs/e58.done
fail() { echo "FAILED: $*"; echo "exit=1" > logs/e58.done; exit 1; }
echo "== $(date +%F_%T) audio_wavlm・video・text_roberta（同時）"
$P exp/geom_e0.py --kind audio_wavlm > logs/e58_a.log 2>&1 &
PA=$!
$P exp/geom_e0.py --kind video > logs/e58_v.log 2>&1 &
PV=$!
$P exp/geom_e0.py --kind text_roberta > logs/e58_t.log 2>&1 &
PT=$!
wait $PA || fail "audio_wavlm"
wait $PV || fail "video"
wait $PT || fail "text_roberta"
echo "== $(date +%F_%T) 解析"
$P exp/geom_e0.py --analyze > logs/e58_analysis.log 2>&1 || fail "解析"
echo "exit=0" > logs/e58.done
echo "== $(date +%F_%T) 終わり"
