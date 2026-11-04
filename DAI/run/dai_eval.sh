#!/bin/bash
# DAI 実験 1〜3 の評価のターゲットの回を，1 つの評価のターゲット T について回す（実験 1 の設計 10 節，実験 2 の設計 4 節，実験 3 の設計 9 節）．
# 使い方：DAI/run/dai_eval.sh <T>．置き場は本番の results/dai1・results/dai2・results/dai3 に固定する．
# 前提：DAI/frozen/eval_gate.json・dai1_frozen.json・dai2_frozen.json・dai3_frozen.json がコミット済みで，作業木が汚れていないこと
# （台本の側の check_eval_gate などが確かめる）．results/dai1/{source,select}，results/dai2/{select,st_cv}，results/dai3/{frozen.json,validate,detect} があること．
# 1. 実験 1 の eval（分割ごとに 5 本）と実験 2 の eval（線形と勾配法を分割ごとに 10 本）を同時に流す．
# 2. 実験 1 の eval がそろったら，実験 3 の eval を T だけで流す（単位ごとの保存 results/dai3/eval/units/<T>_f<f>_r<r>.pkl を作る．
#    4 つの T の判定の規則は，全部の T の保存を 1 か所に集めて --targets を 4 つにして流し直したときに当たる）．
# どれかの処理が失敗したら logs/dai_eval_<T>.failed を書いて止まる．
set -u
cd "$(dirname "$0")/../.."
T=$1
CUT=${CUT:-}
PY=${PY:-python}
mkdir -p logs
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MERDA_GPU_FRACTION=${MERDA_GPU_FRACTION:-0.08}
fail() { echo "$1" > "logs/dai_eval_${T}.failed"; exit 1; }
p1=(); p2=()
for f in 0 1 2 3 4; do
  $PY DAI/exp/dai1.py --stage eval --targets "$T" --folds $f --out-dir results/dai1 > "logs/dai1_eval_${T}_f$f.log" 2>&1 &
  p1+=($!)
  $PY DAI/exp/dai2.py --stage eval --targets "$T" --folds $f --out-dir results/dai2 --dai1-dir results/dai1 > "logs/dai2_eval_${T}_lin_f$f.log" 2>&1 &
  p2+=($!)
  $PY DAI/exp/dai2_grad.py --stage eval --targets "$T" --folds $f --out-dir results/dai2 --dai1-dir results/dai1 --parts map xmuda coral $CUT > "logs/dai2_eval_${T}_grad_f$f.log" 2>&1 &
  p2+=($!)
done
for p in "${p1[@]}"; do wait "$p" || fail "実験 1 の eval の処理 $p が失敗"; done
date > "logs/dai1_eval_${T}.done"
$PY DAI/exp/dai3.py --stage eval --targets "$T" > "logs/dai3_eval_${T}.log" 2>&1 &
p3=$!
for p in "${p2[@]}"; do wait "$p" || fail "実験 2 の eval の処理 $p が失敗"; done
date > "logs/dai2_eval_${T}.done"
wait "$p3" || fail "実験 3 の eval が失敗"
date > "logs/dai3_eval_${T}.done"
date > "logs/dai_eval_${T}.done"
