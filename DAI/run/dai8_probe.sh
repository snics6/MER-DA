#!/usr/bin/env bash
# DAI-8 の下調べの本計算（設計 DAI/docs/実験8_下調べ_設計.md 第 3 版．台本 DAI/exp/dai8_probe.py）
# 単位を 2 組に分けて同じ GPU で並べて流し，両方が終わったら集計する．終わると logs/dai8_probe.done を作る．
set -euo pipefail
cd /home/snics/MER-DA
OUT=results/dai8_probe
if [ -e "$OUT" ]; then
  echo "$OUT は既にある（本計算は空の出力先に書く）"
  exit 1
fi
mkdir -p "$OUT" logs
export CUDA_VISIBLE_DEVICES=0
echo "開始 $(date '+%F %T')"
.venv/bin/python DAI/exp/dai8_probe.py --out "$OUT" --shard 0/2 > logs/dai8_probe_s0.log 2>&1 &
P0=$!
sleep 30
.venv/bin/python DAI/exp/dai8_probe.py --out "$OUT" --shard 1/2 > logs/dai8_probe_s1.log 2>&1 &
P1=$!
wait "$P0"
wait "$P1"
echo "単位の段を終えた $(date '+%F %T')"
.venv/bin/python DAI/exp/dai8_probe.py --out "$OUT" --summarize > logs/dai8_probe_sum.log 2>&1
echo "集計の段を終えた $(date '+%F %T')"
touch logs/dai8_probe.done
