#!/bin/bash
# E31 段 2（試走の後の残り）と段 3 を順に流す。各段の終わりに logs/compare_<段>.done を書く。
# 設計は docs/exploratory-analyses.md E31（段 2・段 3 の実装の細部）。終わったターゲットは台本が飛ばす。
cd /home/snics/MER-DA
export MERDA_MAP_ROOT=results/fixc_map
P=./.venv/bin/python
MAIN="audio_wavlm audio_hubert video fuse-audio-video"
TEXT="text_roberta fuse-audio-text fuse-video-text fuse-audio-video-text"
run() { echo "== $(date +%F_%T) $*"; $P exp/compare_b1.py "$@" || { echo "FAILED: $*"; return 1; }; }
stage() {   # stage 名 コマンド…（失敗したら .done に exit=1 を書いて止める）
    local name=$1; shift
    if "$@"; then echo "exit=0" > logs/compare_$name.done; else echo "exit=1" > logs/compare_$name.done; exit 1; fi
}
b2_trans() { for k in $MAIN $TEXT; do run --kind $k --parts trans --tag trans || return 1; done; }
b2_text()  { for k in $TEXT; do run --kind $k || return 1; run --kind $k --ks 50 all --tag k50all || return 1; done; }
b2_k50()   { for k in $MAIN; do run --kind $k --ks 50 all --tag k50all || return 1; done; }
b2_o2()    { for k in $MAIN $TEXT; do run --kind $k --ks 1 5 10 25 50 all --parts o2 --tag o2 || return 1; done; }
R="10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29"
b3()       { for k in $MAIN; do run --kind $k --parts final --reps $R --tag final || return 1;
                                run --kind $k --parts final --reps $R --ks 50 all --tag final_k50all || return 1; done; }
stage b2_trans b2_trans
stage b2_text b2_text
stage b2_k50 b2_k50
stage b2_o2 b2_o2
stage b3 b3
