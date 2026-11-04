#!/bin/bash
# DAI 実験 1〜3 の評価のターゲットの回を，1 つの実験と 1 つの評価のターゲット T について回す
# （実験 1 の設計 10 節，実験 2 の設計 4 節，実験 3 の設計 9 節）．
# 使い方：DAI/run/dai_eval.sh <1|2|3> <T>．置き場は本番の results/dai1・results/dai2・results/dai3 に固定する．
#   1：実験 1 の eval を分割ごとに 5 本並べる（GPU のメモリの上限は 1 本あたり 0.15．仮のターゲットの回と同じ）．
#   2：実験 2 の eval（線形の手法と勾配法の手法）を分割ごとに 10 本並べる（1 本あたり 0.12．仮のターゲットの回と同じ）．
#   3：実験 3 の eval を T だけで 1 本流す（0.5）．results/dai1/eval/units に T の 50 単位がそろっていること．
#      単位ごとの保存 results/dai3/eval/units/<T>_f<f>_r<r>.pkl を作る．4 つの T の判定の規則は，全部の T の保存を
#      1 か所に集め，--targets を 4 つにして流し直したときに当たる（保存を読むだけなので計算はやり直さない）．
# 前提：DAI/frozen/eval_gate.json・dai1_frozen.json・dai2_frozen.json・dai3_frozen.json がコミット済みで，作業木が汚れていないこと
# （台本の側の check_eval_gate などが確かめる）．
# 入力のファイル（特徴・発話一覧・分割の表・固定のファイル）の sha256 を最初にログへ書く（gate-code の計算後の点検の DAI-3 記録 1）．
# 失敗したら logs/dai_eval_<実験>_<T>.failed を書いて止まる．終わったら logs/dai_eval_<実験>_<T>.done を書く．
set -u
cd "$(dirname "$0")/../.."
E=$1
T=$2
PY=${PY:-python}
mkdir -p logs
TAG="${E}_${T}"
fail() { echo "$1" > "logs/dai_eval_${TAG}.failed"; exit 1; }
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
{
  echo "commit $(git rev-parse HEAD)"
  sha256sum data/cache/audio_wavlm_*.npz data/cache/video_*.npz data/cache/text_roberta_*.npz data/metadata/*.csv DAI/splits/* DAI/frozen/*.json
} > "logs/dai_eval_${TAG}.inputs" 2>&1 || fail "入力の sha256 を書けない"
pids=()
case "$E" in
  1)
    export MERDA_GPU_FRACTION=0.15
    for f in 0 1 2 3 4; do
      $PY DAI/exp/dai1.py --stage eval --targets "$T" --folds $f --out-dir results/dai1 > "logs/dai1_eval_${T}_f$f.log" 2>&1 &
      pids+=($!)
    done ;;
  2)
    export MERDA_GPU_FRACTION=0.12
    for f in 0 1 2 3 4; do
      $PY DAI/exp/dai2.py --stage eval --targets "$T" --folds $f --out-dir results/dai2 --dai1-dir results/dai1 > "logs/dai2_eval_${T}_lin_f$f.log" 2>&1 &
      pids+=($!)
      $PY DAI/exp/dai2_grad.py --stage eval --targets "$T" --folds $f --out-dir results/dai2 --dai1-dir results/dai1 --parts map xmuda coral > "logs/dai2_eval_${T}_grad_f$f.log" 2>&1 &
      pids+=($!)
    done ;;
  3)
    export MERDA_GPU_FRACTION=0.5
    n=$(ls results/dai1/eval/units/${T}_ALL_f*_r*_p3.npz 2>/dev/null | wc -l)
    [ "$n" -ge 50 ] || fail "実験 1 の eval の段 3 の単位が $n 個しかない"
    $PY DAI/exp/dai3.py --stage eval --targets "$T" > "logs/dai3_eval_${T}.log" 2>&1 &
    pids+=($!) ;;
  *) fail "実験の番号は 1・2・3 のどれか" ;;
esac
for p in "${pids[@]}"; do wait "$p" || fail "処理 $p が失敗"; done
date > "logs/dai_eval_${TAG}.done"
