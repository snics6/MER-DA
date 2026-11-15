#!/bin/bash
# E52（docs/exploratory-analyses.md E52）：出発点を仲間に入れた見込みと本数で回す番で教える選び方を動かす（exp/teach_stack.py）。
# A と A(HuBERT) を同じ GPU で同時に流し，両方終わったら解析。終わったターゲットは台本が飛ばす。計算が失敗したときだけ止める。
#     tmux new-window -t mer-da -n e52 'cd /home/snics/MER-DA && ./run/e52.sh 2>&1 | tee -a logs/e52.log'
cd /home/snics/MER-DA
export MERDA_MAP_ROOT=results/fixc_map
P=./.venv/bin/python
rm -f logs/e52.done
fail() { echo "FAILED: $*"; echo "exit=1" > logs/e52.done; exit 1; }
echo "== $(date +%F_%T) audio_wavlm と audio_hubert（同時）"
$P exp/teach_stack.py --kind audio_wavlm > logs/e52_wavlm.log 2>&1 &
PA=$!
$P exp/teach_stack.py --kind audio_hubert > logs/e52_hubert.log 2>&1 &
PB=$!
wait $PA || fail "audio_wavlm"
wait $PB || fail "audio_hubert"
echo "== $(date +%F_%T) 解析"
$P exp/teach_stack.py --analyze > logs/e52_analysis.log 2>&1 || fail "解析"
echo "exit=0" > logs/e52.done
echo "== $(date +%F_%T) 終わり"
