#!/bin/bash
# E62：第 2 エンコーダ（音声 HuBERT-large，顔 CLIP ViT-L，言葉 DeBERTa-v3）の AVT の世界で E60（モダリティをまたぐ蒸留）と E61（比べる相手）をやり直す
# （docs/exploratory-analyses.md E62．設計と予言は計算の前に書いた）．RunPod（H100）で流す前提．研究室PCでも同じ命令で流せる．
#   段 1：5 つの値を仮のターゲットだけから選ぶ（meta5 --pseudo-only．video_clip・text_deberta・fuse2 の 3 種類 × 4 ターゲット．audio_hubert は E40 の値）
#   段 2：E60 相当（multimodal_distill --world fuse2-audio-video-text．全部の生徒，反復 0〜9）→ 判定は e60_2nd_*.csv
#   段 3：E61 相当（baselines_distill --world fuse2-audio-video-text．全部の生徒）→ 判定は e61_2nd_*.csv
# 終わったターゲットは台本が飛ばす．止めるのは計算の失敗だけ（落ちた仕事は数えて最後に報告）．
#     tmux new-session -d -s mer-da -n e62 'cd <作業木> && ./run/e62.sh 2>&1 | tee -a logs/e62.log'
cd "$(dirname "$0")/.." || exit 1
export MERDA_MAP_ROOT=results/fixc_map
P=./.venv/bin/python
W=fuse2-audio-video-text
TARGETS=(IEMOCAP-impro IEMOCAP-script MELD MOSEI)
mkdir -p logs
rm -f logs/e62.done
NFAIL=0
note() { echo "FAILED（続ける）: $*"; NFAIL=$((NFAIL + 1)); }
echo "== $(date +%F_%T) 段 1：5 つの値（仮のターゲットだけ）"
for k in video_clip text_deberta "$W"; do
  $P exp/meta5.py --kind "$k" --pseudo-only --target "${TARGETS[@]}" > "logs/e62_meta5_${k}.log" 2>&1 || note "meta5 $k"
done
$P exp/meta5.py --analyze-pseudo --kinds video_clip text_deberta "$W" > logs/e62_meta5_analysis.log 2>&1 || note "meta5 の選択"
[ -f results/explore_relation/e62_targets.csv ] || { echo "e62_targets.csv が無いので止める"; echo "exit=1 fails=$((NFAIL + 1))" > logs/e62.done; exit 1; }
echo "== $(date +%F_%T) 段 2：モダリティをまたぐ蒸留（E60 相当）"
$P exp/multimodal_distill.py --world "$W" > logs/e62_multimodal_distill.log 2>&1 || note "multimodal_distill"
$P exp/multimodal_distill.py --analyze --worlds "$W" --suffix _2nd > logs/e62_e60_analysis.log 2>&1 || note "E60 相当の判定"
echo "== $(date +%F_%T) 段 3：比べる相手（E61 相当）"
$P exp/baselines_distill.py --world "$W" > logs/e62_baselines_distill.log 2>&1 || note "baselines_distill"
$P exp/baselines_distill.py --analyze --names "$W" --suffix _2nd > logs/e62_e61_analysis.log 2>&1 || note "E61 相当の判定"
[ "$NFAIL" -eq 0 ] && echo "exit=0" > logs/e62.done || echo "exit=1 fails=$NFAIL" > logs/e62.done
echo "== $(date +%F_%T) 終わり（落ちた仕事 $NFAIL 本）"
