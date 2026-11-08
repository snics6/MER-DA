#!/usr/bin/env bash
# DAI-12 の下調べの本計算（設計 DAI/docs/実験12_下調べ_設計.md．台本 DAI/exp/dai12_probe.py）
# 単位の段 → 集計の段 を続けて流す．終わると logs/dai12_probe.done を作る（失敗は logs/dai12_probe.failed）．
# 検査を通した台本の sha256 を DAI12_SHA で必ず渡す．
set -euo pipefail
cd /home/snics/MER-DA
mkdir -p logs
trap 'touch logs/dai12_probe.failed' ERR
OUT=results/dai12_probe
if [ -e "$OUT" ]; then
  echo "$OUT は既にある（本計算は空の出力先に書く）"
  touch logs/dai12_probe.failed
  exit 1
fi
if [ -z "${DAI12_SHA:-}" ]; then
  echo "DAI12_SHA（検査を通した台本の sha256）が無い"
  touch logs/dai12_probe.failed
  exit 1
fi
ACTUAL_SHA=$(sha256sum DAI/exp/dai12_probe.py | cut -d' ' -f1)
if [ "$ACTUAL_SHA" != "$DAI12_SHA" ]; then
  echo "台本の sha256 が検査を通した版と違う：$ACTUAL_SHA"
  touch logs/dai12_probe.failed
  exit 1
fi
export CUDA_VISIBLE_DEVICES=0
echo "開始 $(date '+%F %T')"
.venv/bin/python DAI/exp/dai12_probe.py --out "$OUT" --stage units > logs/dai12_probe_units.log 2>&1
echo "単位の段を終えた $(date '+%F %T')"
.venv/bin/python DAI/exp/dai12_probe.py --out "$OUT" --stage summarize > logs/dai12_probe_sum.log 2>&1
echo "集計の段を終えた $(date '+%F %T')"
touch logs/dai12_probe.done
