#!/bin/bash
# E31 段 2 の試走：B1（logs/compare_b1.done）の終わりを待ち，いちばん重いターゲット（WavLM の MOSEI）の K=50・全件を流す。
# 出力は本番と同じ場所（results/compare_audio_wavlm/MOSEI_k50all.*）なので，本番では飛ばされる。
cd /home/snics/MER-DA
export MERDA_MAP_ROOT=results/fixc_map
until [ -f logs/compare_b1.done ]; do sleep 60; done
grep -q "exit=0" logs/compare_b1.done || { echo "B1 が失敗したので止める"; echo "exit=9" > logs/compare_b2_trial.done; exit 1; }
echo "== $(date +%F_%T) 段 2 の試走"
./.venv/bin/python exp/compare_b1.py --kind audio_wavlm --target MOSEI --ks 50 all --tag k50all
echo "exit=$?" > logs/compare_b2_trial.done
