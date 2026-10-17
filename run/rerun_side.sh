#!/bin/bash
# 交差検証の地図でしか数値の無かった解析を，2 つの地図で流し直す（docs/exploratory-analyses.md の 2026-10-18 の節）。CPU。
# 本筋の段（rerun_main.py）と端の解析（exp/side/）を，交差検証の地図 → results/rerun_cv/，C を固定した地図 → results/rerun_fixc/ に書く。
#     tmux new-window -t mer-da -n side 'cd /home/snics/MER-DA && ./run/rerun_side.sh 2>&1 | tee -a logs/rerun_side.log'
cd /home/snics/MER-DA
P=./.venv/bin/python
rm -f logs/rerun_side.done
fail() { echo "FAILED: $*"; echo "exit=1" > logs/rerun_side.done; exit 1; }
run_all() {   # $1：出力先，$2：ラベル
  echo "== $(date +%F_%T) $2：本筋の段"
  MERDA_OUT_ROOT=$1 $P exp/rerun_main.py --no-summarize --label $2 || echo "（本筋の段のどこかが失敗．manifest を見る）"
  for s in explore_relation explore_select explore_metrics explore_strategies explore_modality explore_subset check_h7; do
    echo "== $(date +%F_%T) $2：$s"
    MERDA_OUT_ROOT=$1 $P exp/side/$s.py > $1/log_$s.txt 2>&1 || fail "$2 $s"
  done
}
mkdir -p results/rerun_cv results/rerun_fixc
(unset MERDA_MAP_ROOT; run_all results/rerun_cv cv) || fail cv
(export MERDA_MAP_ROOT=results/fixc_map; run_all results/rerun_fixc fixc) || fail fixc
echo "exit=0" > logs/rerun_side.done
echo "== $(date +%F_%T) 終わり"
