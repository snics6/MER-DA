#!/usr/bin/env bash
# DAI-9 の下調べの本計算（設計 DAI/docs/実験9_下調べ_設計.md．台本 DAI/exp/dai9_probe.py）
# GPU の取り合いを避けるため，DAI-8 の単位の段が終わる（logs/dai8_probe_launch.log に「単位の段を終えた」が出る）のを待ってから流す．
# 終わると logs/dai9_probe.done，失敗は logs/dai9_probe.failed．
set -euo pipefail
cd /home/snics/MER-DA
trap 'touch logs/dai9_probe.failed' ERR
OUT=results/dai9_probe
if [ -e "$OUT" ]; then
  echo "$OUT は既にある"
  exit 1
fi
echo "DAI-8 の単位の段を待つ $(date '+%F %T')"
until grep -q "単位の段を終えた" logs/dai8_probe_launch.log 2>/dev/null || [ -f logs/dai8_probe.failed ]; do sleep 60; done
if [ -f logs/dai8_probe.failed ]; then echo "DAI-8 が失敗したので DAI-9 も止める"; exit 1; fi
# 検査した版（4154f9a）の台本で流すことを確かめる（gate-code の第 2 回の条件）
echo "a57b46f61999dcb9a79d33f8a2bacb4539384b442f03af3b48cc9aa1f043f23c  DAI/exp/dai9_probe.py" | sha256sum -c --quiet
echo "199efb1aefe3ac212061824b6e7def10640bff35cf071767e4734fe201c92610  DAI/exp/dai8_probe.py" | sha256sum -c --quiet
export CUDA_VISIBLE_DEVICES=0
echo "開始 $(date '+%F %T')"
.venv/bin/python DAI/exp/dai9_probe.py --out "$OUT" > logs/dai9_probe_units.log 2>&1
echo "単位の段を終えた $(date '+%F %T')"
.venv/bin/python DAI/exp/dai9_probe.py --out "$OUT" --summarize > logs/dai9_probe_sum.log 2>&1
echo "集計の段を終えた $(date '+%F %T')"
touch logs/dai9_probe.done
