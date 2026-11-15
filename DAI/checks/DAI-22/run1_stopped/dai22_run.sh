#!/bin/bash
# 実験 22 の本番（設計 docs/設計/実験22_設計.md 第 4 版，台本 exp/dai22.py は cc925db から変わらない）．作業木 /home/snics/MER-DA-wt22（10307e3）から流す
L=/home/snics/MER-DA/DAI/logs
cd /home/snics/MER-DA-wt22/DAI || exit 1
.venv/bin/python exp/dai22.py --stage units > $L/dai22_units.log 2>&1 || { echo FAIL_UNITS > $L/dai22.fail; exit 1; }
.venv/bin/python exp/dai22.py --stage summary > $L/dai22_summary.log 2>&1 || { echo FAIL_SUMMARY > $L/dai22.fail; exit 1; }
echo OK > $L/dai22.finished
