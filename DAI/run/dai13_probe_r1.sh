#!/usr/bin/env bash
# DAI-13 の下調べ：種 r = 1 の 60 単位を足し，2 つの種をそろえた集計（summary_seeds2/）を作る（設計 4.1・5 節．r = 0 の結果によらず無条件に流す）．
# 終わると logs/dai13_probe_r1.done を作る（失敗は .failed）．検査を通した台本の sha256 を DAI13_SHA で必ず渡す（r = 0 と同じ台本）．
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
trap 'touch logs/dai13_probe_r1.failed' ERR
OUT=results/dai13_probe
if [ ! -e "$OUT/summary_seeds1" ]; then
  echo "r = 0 の集計（$OUT/summary_seeds1）が無い"
  touch logs/dai13_probe_r1.failed
  exit 1
fi
if [ -z "${DAI13_SHA:-}" ]; then
  echo "DAI13_SHA（検査を通した台本の sha256）が無い"
  touch logs/dai13_probe_r1.failed
  exit 1
fi
ACTUAL_SHA=$(sha256sum exp/dai13_probe.py | cut -d' ' -f1)
if [ "$ACTUAL_SHA" != "$DAI13_SHA" ]; then
  echo "台本の sha256 が検査を通した版と違う：$ACTUAL_SHA"
  touch logs/dai13_probe_r1.failed
  exit 1
fi
N0=$(ls "$OUT"/rows/*_r0.npz 2>/dev/null | wc -l)
if [ "$N0" != "60" ]; then
  echo "r = 0 の行が $N0 単位（60 のはず）"
  touch logs/dai13_probe_r1.failed
  exit 1
fi
ROWS_SHA=$(.venv/bin/python -c "import numpy as np, glob; print(str(np.load(sorted(glob.glob('$OUT/rows/*_r0.npz'))[0], allow_pickle=False)['script_sha256']))")
if [ "$ROWS_SHA" != "$ACTUAL_SHA" ]; then
  echo "r = 0 の行の台本の sha256（$ROWS_SHA）が今の台本（$ACTUAL_SHA）と違う（gate-code 第 2 回の記録）"
  touch logs/dai13_probe_r1.failed
  exit 1
fi
export CUDA_VISIBLE_DEVICES=0
echo "開始（r = 1）$(date '+%F %T')"
.venv/bin/python exp/dai13_probe.py --out "$OUT" --stage units --units "*_r1" --resume > logs/dai13_probe_r1_units.log 2>&1
echo "単位の段（r = 1）を終えた $(date '+%F %T')"
.venv/bin/python exp/dai13_probe.py --out "$OUT" --stage summarize --n-expected 120 > logs/dai13_probe_r1_sum.log 2>&1
echo "集計の段（2 つの種）を終えた $(date '+%F %T')"
touch logs/dai13_probe_r1.done
