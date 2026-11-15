#!/bin/bash
# E61 の続き（研究者の指示 2026-10-23．docs/exploratory-analyses.md E61 の追記）：AV・VT・AT の 3 世界を，融合の生徒とその相手だけで流す
# （exp/baselines_distill.py --world … --fused-only）。AVT の終了（logs/e61.done）を待ち，3 本を同時に流す（GPU の上限 0.3 ずつ．AVT の最大は 5.6 GB）。
# 全部終わったら 4 世界まとめて解析。終わったターゲットは台本が飛ばす。止めるのは計算の失敗だけ（落ちた仕事は数えて最後に報告）。
#     tmux new-window -t mer-da -n e61f 'cd /home/snics/MER-DA && ./run/e61_fused.sh 2>&1 | tee -a logs/e61_fused.log'
cd /home/snics/MER-DA
export MERDA_MAP_ROOT=results/fixc_map
export MERDA_GPU_FRACTION=0.3
P=./.venv/bin/python
MAXJ=3
rm -f logs/e61_fused.done
NFAIL=0
note() { echo "FAILED（続ける）: $*"; NFAIL=$((NFAIL + 1)); }
echo "== $(date +%F_%T) AVT（logs/e61.done）の終了を待つ"
until [ -f logs/e61.done ]; do sleep 120; done
grep -q "exit=0" logs/e61.done || note "AVT の台本が exit=0 でない（$(cat logs/e61.done)）．3 世界は流す"
WORLDS=("fuse-audio-video" "fuse-audio-text" "fuse-video-text")
for w in "${WORLDS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do wait -n || note "ジョブ（$(date +%T)）"; done
  echo "== $(date +%F_%T) 開始 $w（融合のみ）→ logs/e61_baselines_distill_${w}.log"
  $P exp/baselines_distill.py --world "$w" --fused-only > "logs/e61_baselines_distill_${w}.log" 2>&1 &
done
while [ "$(jobs -rp | wc -l)" -gt 0 ]; do wait -n || note "ジョブ（$(date +%T)）"; done
echo "== $(date +%F_%T) 解析（4 世界まとめて．落ちた仕事 $NFAIL 本）"
$P exp/baselines_distill.py --analyze > logs/e61_fused_analysis.log 2>&1 || note "解析"
[ "$NFAIL" -eq 0 ] && echo "exit=0" > logs/e61_fused.done || echo "exit=1 fails=$NFAIL" > logs/e61_fused.done
echo "== $(date +%F_%T) 終わり"
