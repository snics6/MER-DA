#!/bin/bash
# E57（docs/exploratory-analyses.md E57）：見本 4 本から始めて残りのどの発話にラベルを付けるかを選ぶ（exp/seeded_cond3.py）。
# A と V は 2 つずつに分け，T と合わせて 5 つを同じ GPU で同時に流し，全部終わったら解析。終わったターゲットは台本が飛ばす。計算が失敗したときだけ止める。
#     tmux new-window -t mer-da -n e57 'cd /home/snics/MER-DA && ./run/e57.sh 2>&1 | tee -a logs/e57.log'
cd /home/snics/MER-DA
export MERDA_MAP_ROOT=results/fixc_map
P=./.venv/bin/python
rm -f logs/e57.done
fail() { echo "FAILED: $*"; echo "exit=1" > logs/e57.done; exit 1; }
echo "== $(date +%F_%T) audio_wavlm×2・video×2・text_roberta（同時）"
$P exp/seeded_cond3.py --kind audio_wavlm --target MOSEI IEMOCAP-impro RAVDESS > logs/e57_a1.log 2>&1 &
P1=$!
$P exp/seeded_cond3.py --kind audio_wavlm --target MELD IEMOCAP-script AFEW CREMA-D > logs/e57_a2.log 2>&1 &
P2=$!
$P exp/seeded_cond3.py --kind video --target MOSEI IEMOCAP-impro AFEW > logs/e57_v1.log 2>&1 &
P3=$!
$P exp/seeded_cond3.py --kind video --target MELD IEMOCAP-script CREMA-D > logs/e57_v2.log 2>&1 &
P4=$!
$P exp/seeded_cond3.py --kind text_roberta > logs/e57_t.log 2>&1 &
P5=$!
for p in $P1 $P2 $P3 $P4 $P5; do wait $p || fail "process $p"; done
echo "== $(date +%F_%T) 解析"
$P exp/seeded_cond3.py --analyze > logs/e57_analysis.log 2>&1 || fail "解析"
echo "exit=0" > logs/e57.done
echo "== $(date +%F_%T) 終わり"
