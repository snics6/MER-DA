#!/bin/bash
# E61（docs/exploratory-analyses.md E61）：E59・E60 の比べる相手（exp/baselines_distill.py，exp/al_cond2_da.py）。
# E60 の終了（logs/e60.done）を待ち，同じ GPU で 4 本まで同時に流す（長い順：B の 4 世界，A の 3 種類，C の 2 種類）．全部終わったら解析。
# 終わったターゲットは台本が飛ばす。計算が失敗したときだけ止める。
#     tmux new-window -t mer-da -n e61 'cd /home/snics/MER-DA && ./run/e61.sh 2>&1 | tee -a logs/e61.log'
#     ./run/e61.sh avt   … AVT の世界だけを 1 本で流す（研究者の指示 2026-10-22．GPU を独り占めするので上限 0.9）
cd /home/snics/MER-DA
export MERDA_MAP_ROOT=results/fixc_map
export MERDA_GPU_FRACTION=0.22                      # 4 本 × 5.3 GB＋CUDA の文脈で 24 GB に収める（AVT は MOSEI の候補で 5.3 GB を超えて落ちた）
P=./.venv/bin/python
MAXJ=4
ONLY="${1:-}"
rm -f logs/e61.done
NFAIL=0
fail() { echo "FAILED: $*"; echo "exit=1" > logs/e61.done; exit 1; }
note() { echo "FAILED（続ける）: $*"; NFAIL=$((NFAIL + 1)); }
echo "== $(date +%F_%T) E60 の終了を待つ"
until [ -f logs/e60.done ]; do sleep 120; done
grep -q "exit=0" logs/e60.done || fail "E60 が失敗していた"
JOBS=("baselines_distill.py --world fuse-audio-video" "baselines_distill.py --world fuse-audio-video-text" "baselines_distill.py --world fuse-audio-text"
      "baselines_distill.py --world fuse-video-text" "baselines_distill.py --kind audio_wavlm" "baselines_distill.py --kind video"
      "baselines_distill.py --kind text_roberta" "al_cond2_da.py --kind audio_wavlm" "al_cond2_da.py --kind video")
if [ "$ONLY" = "avt" ]; then
  JOBS=("baselines_distill.py --world fuse-audio-video-text"); MAXJ=1; export MERDA_GPU_FRACTION=0.9
fi
# 1 本が落ちても他は続け，終わったターゲットは残す（落ちた仕事は台本をもう一度流せば残りから再開する）
for job in "${JOBS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do wait -n || note "ジョブ（$(date +%T)）"; done
  set -- $job
  tag="${1%.py}_${3}"
  echo "== $(date +%F_%T) 開始 $job → logs/e61_${tag}.log"
  $P exp/$job > "logs/e61_${tag}.log" 2>&1 &
done
while [ "$(jobs -rp | wc -l)" -gt 0 ]; do wait -n || note "ジョブ（$(date +%T)）"; done
echo "== $(date +%F_%T) 解析（落ちた仕事 $NFAIL 本）"
if [ "$ONLY" = "avt" ]; then
  $P exp/baselines_distill.py --analyze --names fuse-audio-video-text > logs/e61_analysis.log 2>&1 || note "解析"
else
  $P exp/baselines_distill.py --analyze > logs/e61_analysis.log 2>&1 || note "解析"
  $P exp/al_cond2_da.py --analyze > logs/e61c_analysis.log 2>&1 || note "解析 C"
fi
[ "$NFAIL" -eq 0 ] && echo "exit=0" > logs/e61.done || echo "exit=1 fails=$NFAIL" > logs/e61.done
echo "== $(date +%F_%T) 終わり"
