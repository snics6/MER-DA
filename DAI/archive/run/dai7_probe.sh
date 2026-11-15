#!/usr/bin/env bash
# DAI-7 の下調べの本計算（`DAI/docs/実験7_下調べ_設計.md`）．研究者の指示 2026-11-07「通る前にとりあえず本計算してみて．あかんかったらやりなおすからさ」．
# 関門（gate-probe・gate-code・査読想定の 2 役）を通す前の計算である．指摘で直したら出力を消して流し直す．
# 使い方：tmux の mer-da で  bash DAI/run/dai7_probe.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=results/dai7_probe
LOG=logs/dai7_probe.log
mkdir -p logs
echo "開始 $(date '+%F %T') コミット $(git rev-parse --short HEAD)" | tee -a "$LOG"
./.venv/bin/python DAI/exp/dai7_probe.py --out "$OUT" 2>&1 | tee -a "$LOG"
./.venv/bin/python DAI/exp/dai7_probe.py --out "$OUT" --summarize 2>&1 | tee -a "$LOG"
echo "終了 $(date '+%F %T')" | tee -a "$LOG"
touch logs/dai7_probe.done
