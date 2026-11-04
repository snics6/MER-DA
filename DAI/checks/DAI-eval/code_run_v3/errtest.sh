#!/bin/bash
# dai_eval.sh:45 の行を写し，記録のファイル $1 で止まるかを返す（止まれば FAIL）
f=$1
tail -n +"$(grep -n "^== " "$f" | tail -1 | cut -d: -f1)" "$f" | grep -qiE "No such file|cannot|error|Traceback" && echo FAIL || echo PASS
