#!/usr/bin/env bash
# DAI-6 の下調べの本計算（`DAI/docs/実験6_下調べ_設計.md`．研究者の承認 2026-11-07「よい」）．
# AFEW・IEMOCAP・MELD は並べ替え 1,000 回，MOSEI は 500 回（設計 6 節の決定）．最後に集計する．
# 使い方：tmux の mer-da で  bash DAI/run/dai6_probe.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=results/dai6_probe
LOG=logs/dai6_probe.log
mkdir -p logs
echo "開始 $(date '+%F %T') コミット $(git rev-parse --short HEAD)" | tee -a "$LOG"
./.venv/bin/python DAI/exp/dai6_probe.py --out "$OUT" --corpora AFEW IEMOCAP MELD 2>&1 | tee -a "$LOG"
./.venv/bin/python DAI/exp/dai6_probe.py --out "$OUT" --corpora MOSEI --n-perm 500 2>&1 | tee -a "$LOG"
./.venv/bin/python DAI/exp/dai6_probe.py --out "$OUT" --summarize 2>&1 | tee -a "$LOG"
echo "終了 $(date '+%F %T')" | tee -a "$LOG"
touch logs/dai6_probe.done
