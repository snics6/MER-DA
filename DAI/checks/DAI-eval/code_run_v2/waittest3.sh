#!/bin/bash
# 本物の起動の台本と同じく，単純なコマンドを >> で追記して背景に置く形（$! が実行した処理そのものになる）
set -u
mkdir -p logs; rm -f logs/t2.failed logs/t2.done
pids=()
fail() {
  [ ${#pids[@]} -gt 0 ] && kill "${pids[@]}" 2>/dev/null
  wait 2>/dev/null
  echo "$1" > logs/t2.failed; echo "pids ${pids[*]}" > logs/t2.pids
  exit 1
}
sleep 40 >> logs/a.log 2>&1 &
pids+=($!)
python3 -c "import time,sys; time.sleep(1); sys.exit(4)" >> logs/b.log 2>&1 &
pids+=($!)
sleep 41 >> logs/c.log 2>&1 &
pids+=($!)
left=${#pids[@]}
while [ "$left" -gt 0 ]; do
  wait -n -Z fin; st=$?
  left=$((left - 1))
  [ "$st" -eq 0 ] || fail "処理 $fin が失敗（終了コード $st）"
done
date > logs/t2.done
