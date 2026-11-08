#!/usr/bin/env bash
# DAI-10 の下調べの本計算（設計 DAI/docs/実験10_下調べ_設計.md．台本 DAI/exp/dai10_probe.py）
# 単位の段と集計の段を続けて流す．終わると logs/dai10_probe.done を作る（失敗は logs/dai10_probe.failed）．
set -euo pipefail
cd /home/snics/MER-DA
trap 'touch logs/dai10_probe.failed' ERR
OUT=results/dai10_probe
if [ -e "$OUT" ]; then
  echo "$OUT は既にある（本計算は空の出力先に書く）"
  exit 1
fi
mkdir -p logs
export CUDA_VISIBLE_DEVICES=0
echo "開始 $(date '+%F %T')"
.venv/bin/python DAI/exp/dai10_probe.py --out "$OUT" > logs/dai10_probe_units.log 2>&1
echo "単位の段を終えた $(date '+%F %T')"
.venv/bin/python DAI/exp/dai10_probe.py --out "$OUT" --summarize > logs/dai10_probe_sum.log 2>&1
echo "集計の段を終えた $(date '+%F %T')"
touch logs/dai10_probe.done
