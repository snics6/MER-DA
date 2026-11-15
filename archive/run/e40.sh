#!/bin/bash
# E40（docs/exploratory-analyses.md E40）：teach と E39 の学習器の設定を一緒にメタ学習する（exp/teach_joint.py）。
# 反復 30 回への延長（run/extend30.sh）が logs/extend30.done を書くのを待ってから始める。
# W7 の IEMOCAP 即興を先に流して random の重なる組を E39 と照合し（合わなければ止める），W7 → Hu7 → AV → V → 解析の順に進む。
# 終わったターゲットは台本が飛ばすので，途中で止まったらもう一度起動すれば続きから進む。計算が失敗したときだけ止める。
#     tmux new-window -t mer-da -n e40 'cd /home/snics/MER-DA && ./run/e40.sh 2>&1 | tee -a logs/e40.log'
cd /home/snics/MER-DA
export MERDA_MAP_ROOT=results/fixc_map
P=./.venv/bin/python
KINDS="audio_wavlm audio_hubert fuse-audio-video video"
rm -f logs/e40.done
fail() { echo "FAILED: $*"; echo "exit=1" > logs/e40.done; exit 1; }
until [ -f logs/extend30.done ]; do sleep 60; done
echo "== $(date +%F_%T) 延長の終わりを確認：$(cat logs/extend30.done)"
echo "== $(date +%F_%T) W7 の IEMOCAP 即興（E39 との照合）"
$P exp/teach_joint.py --kind audio_wavlm --target IEMOCAP-impro || fail "W7 IEMOCAP-impro"
$P exp/teach_joint.py --check audio_wavlm IEMOCAP-impro || fail "E39 との照合"
for k in $KINDS; do
  echo "== $(date +%F_%T) $k"
  $P exp/teach_joint.py --kind $k || fail "$k"
done
echo "== $(date +%F_%T) 解析"
$P exp/teach_joint.py --analyze > logs/e40_analysis.log 2>&1 || fail "解析"
echo "exit=0" > logs/e40.done
echo "== $(date +%F_%T) 終わり"
