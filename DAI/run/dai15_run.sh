#!/bin/bash
cd "$(dirname "$0")/.."
mkdir -p logs
PY=.venv/bin/python
$PY exp/dai15.py --stage pseudo > logs/dai15_pseudo.log 2>&1 || { echo FAIL_PSEUDO > logs/dai15.fail; exit 1; }
$PY exp/dai15.py --stage select > logs/dai15_select.log 2>&1 || { echo FAIL_SELECT > logs/dai15.fail; exit 1; }
git add frozen/dai15_frozen.json results/dai15/provenance_log.jsonl 2>/dev/null; git add frozen/dai15_frozen.json
git commit -q -m "DAI-15 frozen selection (pseudo-target run)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01MWNQHAfAhskFmBuuDn3jwA" || { echo FAIL_COMMIT > logs/dai15.fail; exit 1; }
$PY exp/dai15.py --stage eval > logs/dai15_eval.log 2>&1 || { echo FAIL_EVAL > logs/dai15.fail; exit 1; }
$PY exp/dai15.py --stage summary > logs/dai15_summary.log 2>&1 || { echo FAIL_SUMMARY > logs/dai15.fail; exit 1; }
echo OK > logs/dai15.finished
