#!/bin/bash
# E63：微調整の確認（docs/exploratory-analyses.md E63．設計と予言 F1〜F4 は計算の前に書いた）．研究室の A5000 で流す．
#   段 cache（4 群の波形）→ teacher（凍結した特徴の先生と凍結版の値）→ source（出発点 3 組）→ adapt（4 ターゲット × 反復 0〜2 × K ∈ {0,5} × 手法）→ analyze
# 済んだ段・ターゲット・行は台本と exp/finetune_check.py が飛ばす．止めるのは計算の失敗だけ（落ちた仕事は数えて最後に報告）．
#     tmux new-window -t mer-da -n e63 'cd /home/snics/MER-DA && ./run/e63.sh 2>&1 | tee -a logs/e63.log'
cd /home/snics/MER-DA
export MERDA_MAP_ROOT=results/fixc_map
P=./.venv/bin/python
mkdir -p logs results/finetune_check
rm -f logs/e63.done
NFAIL=0
note() { echo "FAILED（続ける）: $*"; NFAIL=$((NFAIL + 1)); }
echo "== $(date +%F_%T) 段 cache"
$P exp/finetune_check.py --stage cache > logs/e63_cache.log 2>&1 || note "cache"
echo "== $(date +%F_%T) 段 teacher"
$P exp/finetune_check.py --stage teacher > logs/e63_teacher.log 2>&1 || note "teacher"
echo "== $(date +%F_%T) 段 source"
for s in IEMOCAP MELD MOSEI; do
  $P exp/finetune_check.py --stage source --set "$s" > "logs/e63_source_${s}.log" 2>&1 || note "source $s"
done
echo "== $(date +%F_%T) 段 adapt"
for T in IEMOCAP-impro IEMOCAP-script MELD MOSEI; do
  $P exp/finetune_check.py --stage adapt --target "$T" > "logs/e63_adapt_${T}.log" 2>&1 || note "adapt $T"
done
echo "== $(date +%F_%T) 段 analyze"
$P exp/finetune_check.py --stage analyze > logs/e63_analysis.log 2>&1 || note "analyze"
[ "$NFAIL" -eq 0 ] && echo "exit=0" > logs/e63.done || echo "exit=1 fails=$NFAIL" > logs/e63.done
echo "== $(date +%F_%T) 終わり（落ちた仕事 $NFAIL 本）"
