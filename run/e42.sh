#!/bin/bash
# E42（docs/exploratory-analyses.md E42）：感情の見込みを近所でならすと正確になるか（exp/belief_check.py）。
# E41 と同じ GPU で同時に流す。W7 → Hu7 → AV → V の後，E41 の終わり（logs/e41.done）を待ってから解析する
# （解析で p_K のテストの UAR を E41 の random_pool と照合するため）。計算が失敗したときだけ止める。
#     tmux new-window -t mer-da -n e42 'cd /home/snics/MER-DA && ./run/e42.sh 2>&1 | tee -a logs/e42.log'
cd /home/snics/MER-DA
export MERDA_MAP_ROOT=results/fixc_map
P=./.venv/bin/python
KINDS="audio_wavlm audio_hubert fuse-audio-video video"
rm -f logs/e42.done
fail() { echo "FAILED: $*"; echo "exit=1" > logs/e42.done; exit 1; }
for k in $KINDS; do
  echo "== $(date +%F_%T) $k"
  $P exp/belief_check.py --kind $k || fail "$k"
done
until [ -f logs/e41.done ]; do sleep 60; done
echo "== $(date +%F_%T) E41 の終わりを確認：$(cat logs/e41.done)"
echo "== $(date +%F_%T) 解析"
$P exp/belief_check.py --analyze > logs/e42_analysis.log 2>&1 || fail "解析"
echo "exit=0" > logs/e42.done
echo "== $(date +%F_%T) 終わり"
