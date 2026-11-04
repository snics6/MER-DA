#!/bin/bash
# dai_eval.sh の 25-31 行と 70-76 行の待ち方を写したもの．引数：各処理の「秒:終了コード」
set -u
TAG=test; mkdir -p logs
rm -f logs/t.done logs/t.failed
pids=()
fail() {
  [ ${#pids[@]} -gt 0 ] && kill "${pids[@]}" 2>/dev/null
  wait 2>/dev/null
  echo "$(date +%T) $1" > logs/t.failed
  exit 1
}
for spec in "$@"; do
  s=${spec%%:*}; c=${spec##*:}
  bash -c "sleep $s; exit $c" &
  pids+=($!)
done
sleep "${PRE:-0}"
left=${#pids[@]}
while [ "$left" -gt 0 ]; do
  wait -n -p fin; st=$?
  left=$((left - 1))
  [ "$st" -eq 0 ] || fail "処理 $fin が失敗（終了コード $st）"
done
date +%T > logs/t.done
