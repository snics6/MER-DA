#!/usr/bin/env bash
# DAI-17 の下調べの本計算（設計 docs/設計/実験17_下調べ_設計.md．台本 exp/dai17_probe.py）
# 単位の段（A）→ 重みの段（B）→ 生徒の段（C）→ 集計の段 を続けて流す．終わると logs/dai17_probe.done を作る（失敗は logs/dai17_probe.failed）．
# 検査を通した台本の sha256 を DAI17_SHA で必ず渡す．途中で止まったら，同じコミットで流し直すと済んだ単位は飛ばす（出力先を消さないこと）．
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
trap 'touch logs/dai17_probe.failed' ERR
if [ -z "${DAI17_SHA:-}" ]; then
  echo "DAI17_SHA（検査を通した台本の sha256）が無い"
  touch logs/dai17_probe.failed
  exit 1
fi
ACTUAL_SHA=$(sha256sum exp/dai17_probe.py | cut -d' ' -f1)
if [ "$ACTUAL_SHA" != "$DAI17_SHA" ]; then
  echo "台本の sha256 が検査を通した版と違う：$ACTUAL_SHA"
  touch logs/dai17_probe.failed
  exit 1
fi
rm -f logs/dai17_probe.done logs/dai17_probe.failed
PY=.venv/bin/python
for STAGE in units reweight students summary; do
  echo "=== $STAGE $(date '+%F %T')"
  $PY exp/dai17_probe.py --stage "$STAGE" 2>&1 | tee -a "logs/dai17_probe_${STAGE}.log"
done
date '+%F %T' > logs/dai17_probe.done
echo "EXIT=0"
