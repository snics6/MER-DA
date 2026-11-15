#!/usr/bin/env bash
# DAI-11 の下調べの本計算（設計 docs/設計/実験11_下調べ_設計.md．台本 exp/dai11_probe.py）
# ソースの段 → 段 A（引き戻しの格子）→ 選ぶ段 → 段 B（単位）→ 集計の段 を続けて流す．
# 終わると logs/dai11_probe.done を作る（失敗は logs/dai11_probe.failed）．
# 検査を通した台本の sha256 を DAI11_SHA で必ず渡す（dai-review-ml の m7）．
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
trap 'touch logs/dai11_probe.failed' ERR
OUT=results/dai11_probe
if [ -e "$OUT" ]; then
  echo "$OUT は既にある（本計算は空の出力先に書く）"
  touch logs/dai11_probe.failed
  exit 1
fi
if [ -z "${DAI11_SHA:-}" ]; then
  echo "DAI11_SHA（検査を通した台本の sha256）が無い"
  touch logs/dai11_probe.failed
  exit 1
fi
ACTUAL_SHA=$(sha256sum exp/dai11_probe.py | cut -d' ' -f1)
if [ "$ACTUAL_SHA" != "$DAI11_SHA" ]; then
  echo "台本の sha256 が検査を通した版と違う：$ACTUAL_SHA"
  touch logs/dai11_probe.failed
  exit 1
fi
export CUDA_VISIBLE_DEVICES=0
echo "開始 $(date '+%F %T')"
.venv/bin/python exp/dai11_probe.py --out "$OUT" --stage source2 > logs/dai11_probe_source2.log 2>&1
echo "ソースの段を終えた $(date '+%F %T')"
.venv/bin/python exp/dai11_probe.py --out "$OUT" --stage enc2k > logs/dai11_probe_enc2k.log 2>&1
echo "段 A を終えた $(date '+%F %T')"
.venv/bin/python exp/dai11_probe.py --out "$OUT" --stage select > logs/dai11_probe_select.log 2>&1
echo "選ぶ段を終えた $(date '+%F %T')"
.venv/bin/python exp/dai11_probe.py --out "$OUT" --stage units > logs/dai11_probe_units.log 2>&1
echo "段 B を終えた $(date '+%F %T')"
.venv/bin/python exp/dai11_probe.py --out "$OUT" --stage summarize > logs/dai11_probe_sum.log 2>&1
echo "集計の段を終えた $(date '+%F %T')"
touch logs/dai11_probe.done
