#!/bin/bash
# E31：B2 の試走（run/compare_b2_trial.sh）の後，B2 の残り・O2・B3（run/compare_rest.sh）と解析の台本までを続けて流す。
# 研究者の指示（2026-10-13）で，試走の時間による停止はせず，計算が失敗したときだけ止める。終わりに logs/compare_all.done を書く。
cd /home/snics/MER-DA
until [ -f logs/compare_b2_trial.done ]; do sleep 60; done
grep -q "exit=0" logs/compare_b2_trial.done || { echo "試走が失敗したので止める"; echo "exit=9" > logs/compare_all.done; exit 1; }
echo "== $(date +%F_%T) 試走の時間：$(grep -E '^audio_wavlm MOSEI：' logs/compare_b2.log | tail -1)"
./run/compare_rest.sh || { echo "exit=1" > logs/compare_all.done; exit 1; }
echo "== $(date +%F_%T) 解析"
./.venv/bin/python exp/explore_compare.py > logs/explore_compare.log 2>&1 \
  && ./.venv/bin/python exp/explore_compare_mech.py > logs/explore_compare_mech.log 2>&1
echo "exit=$?" > logs/compare_all.done
echo "== $(date +%F_%T) 終わり"
