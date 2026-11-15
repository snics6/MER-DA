#!/bin/bash
# E31 の全段を通しで流す：B1（段 1）→ B2 の試走を兼ねた WavLM の MOSEI の K=50・全件 → B2 の残り・O2・B3（run/compare_rest.sh）→ 解析。
# 終わったターゲットは exp/compare_b1.py が飛ばすので，途中で止まったら（WSL の再起動など）もう一度起動すれば続きから進む。
# 研究者の指示（2026-10-13）で，試走の時間による停止はせず，計算が失敗したときだけ止める。終わりに logs/compare_all.done を書く。
#     tmux new-window -t mer-da -n all 'cd /home/snics/MER-DA && ./run/compare_all.sh 2>&1 | tee -a logs/compare_all.log'
cd /home/snics/MER-DA
export MERDA_MAP_ROOT=results/fixc_map
P=./.venv/bin/python
rm -f logs/compare_all.done
fail() { echo "FAILED: $*"; echo "exit=1" > logs/compare_all.done; exit 1; }
echo "== $(date +%F_%T) B1"
for k in audio_wavlm audio_hubert video fuse-audio-video; do
    $P exp/compare_b1.py --kind $k || fail "B1 $k"
done
echo "exit=0" > logs/compare_b1.done
echo "== $(date +%F_%T) B2 の試走（WavLM の MOSEI の K=50・全件）"
$P exp/compare_b1.py --kind audio_wavlm --target MOSEI --ks 50 all --tag k50all || fail "B2 trial"
./run/compare_rest.sh || fail "compare_rest"
echo "== $(date +%F_%T) 解析"
$P exp/explore_compare.py > logs/explore_compare.log 2>&1 || fail "explore_compare"
$P exp/explore_compare_mech.py > logs/explore_compare_mech.log 2>&1 || fail "explore_compare_mech"
echo "exit=0" > logs/compare_all.done
echo "== $(date +%F_%T) 終わり"
