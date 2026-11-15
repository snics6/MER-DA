#!/bin/bash
# E59 の追加（docs/exploratory-analyses.md E59 の追加）：弱いラベルの総重みを K ごとに選ぶ（exp/distill_belief.py --per-k）。
# 3 種類を同じ GPU で同時に流し，全部終わったら解析。終わったターゲットは台本が飛ばす。計算が失敗したときだけ止める。
#     tmux new-window -t mer-da -n e59k 'cd /home/snics/MER-DA && ./run/e59k.sh 2>&1 | tee -a logs/e59k.log'
cd /home/snics/MER-DA
export MERDA_MAP_ROOT=results/fixc_map
P=./.venv/bin/python
rm -f logs/e59k.done
fail() { echo "FAILED: $*"; echo "exit=1" > logs/e59k.done; exit 1; }
echo "== $(date +%F_%T) audio_wavlm・video・text_roberta（同時）"
$P exp/distill_belief.py --per-k --kind audio_wavlm > logs/e59k_a.log 2>&1 &
PA=$!
$P exp/distill_belief.py --per-k --kind video > logs/e59k_v.log 2>&1 &
PV=$!
$P exp/distill_belief.py --per-k --kind text_roberta > logs/e59k_t.log 2>&1 &
PT=$!
wait $PA || fail "audio_wavlm"
wait $PV || fail "video"
wait $PT || fail "text_roberta"
echo "== $(date +%F_%T) 解析"
$P exp/distill_belief.py --analyze --per-k > logs/e59k_analysis.log 2>&1 || fail "解析"
echo "exit=0" > logs/e59k.done
echo "== $(date +%F_%T) 終わり"
