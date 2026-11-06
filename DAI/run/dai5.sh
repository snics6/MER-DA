#!/bin/bash
# DAI 実験 5（ソース × モダリティの識別器を K 本の採点で重みづけた疑似ラベル）の仮のターゲットの回と評価の回を，研究室の PC（A5000）で回す．
# 使い方：DAI/run/dai5.sh <pseudo|eval>．置き場は本番の results/dai5 に固定する．評価のターゲット T ごとに 1 本ずつ 4 本を並べる（GPU のメモリの上限は 1 本あたり既定 0.2）．
# 前提：pseudo は DAI/exp・measure・features が汚れていないこと（台本の側が止める）．eval は DAI/frozen/dai5_frozen.json と eval_gate_dai5.json が
# コミット済みで，check_eval_gate5 の条件を満たすこと．入力（特徴・分割の表・段 source・固定のファイル）の sha256，ライブラリの版，GPU を最初に記録する．
# どれかの処理が失敗したら残りを止め，logs/dai5_<段>.failed を書いて止まる．終わったら .done を書く．起動のたびに前の .done・.failed を消す．ログは追記する．
set -u
[ "${BASH_VERSINFO[0]}${BASH_VERSINFO[1]}" -ge 51 ] 2>/dev/null || { echo "bash 5.1 以上が要る（wait -n -p）．今は $BASH_VERSION"; exit 1; }
cd "$(dirname "$0")/../.."
STAGE=$1
PY=${PY:-./.venv/bin/python}
mkdir -p logs
rm -f "logs/dai5_${STAGE}.done" "logs/dai5_${STAGE}.failed"
pids=()
fail() {
  [ ${#pids[@]} -gt 0 ] && kill "${pids[@]}" 2>/dev/null
  wait 2>/dev/null
  echo "$(date) $1" > "logs/dai5_${STAGE}.failed"
  exit 1
}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MERDA_GPU_FRACTION=${FRAC:-0.2}
{
  echo "== $(date) host $(hostname) commit $(git rev-parse HEAD) status $(git status --porcelain DAI/exp DAI/splits DAI/run measure features data/metadata | wc -l) frac $MERDA_GPU_FRACTION"
  $PY -c "import numpy, pandas, sklearn, scipy, torch; print('numpy', numpy.__version__, 'pandas', pandas.__version__, 'sklearn', sklearn.__version__, 'scipy', scipy.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.get_device_name(0))"
  sha256sum data/cache/audio_wavlm_*.npz data/cache/video_*.npz data/cache/text_roberta_*.npz data/metadata/*.csv DAI/splits/* DAI/frozen/*.json
  sha256sum results/dai1/source/*.npz results/dai1/select/frozen.json
} >> "logs/dai5_${STAGE}.inputs" 2>&1 || fail "入力の記録を書けない"
tail -n +"$(grep -n "^== " "logs/dai5_${STAGE}.inputs" | tail -1 | cut -d: -f1)" "logs/dai5_${STAGE}.inputs" | grep -qiE "No such file|cannot|error|Traceback" && fail "入力の記録に誤りの行がある（logs/dai5_${STAGE}.inputs）"
case "$STAGE" in
  pseudo|eval)
    for T in AFEW IEMOCAP MOSEI MELD; do
      $PY DAI/exp/dai5.py --stage "$STAGE" --targets "$T" --out-dir results/dai5 >> "logs/dai5_${STAGE}_${T}.log" 2>&1 &
      pids+=($!)
    done ;;
  *) fail "段は pseudo か eval" ;;
esac
left=${#pids[@]}
while [ "$left" -gt 0 ]; do
  wait -n -p fin; st=$?
  left=$((left - 1))
  [ "$st" -eq 0 ] || fail "処理 ${fin:-?} が失敗（終了コード $st）"
done
date > "logs/dai5_${STAGE}.done"
