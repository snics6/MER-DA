#!/bin/bash
# 言葉系の反復を主な種類にそろえる・CPU の解析（docs/exploratory-analyses.md「言葉系の反復の回数を主な種類にそろえる」）。
# 列 1（run/text30_a.sh）と列 2（run/text30_b.sh）が両方とも終わる（logs/text30_{a,b}.done が exit=0）のを待ってから，
# 言葉系の地図のまとめ → 手法の解析（E37・E39・30 回の判定・D1 の照合）→ E31・E32 の 30 回の表 →
# 地図の解析（E2・E9・E10・E11・E1・E5・E18・E22〜E24・A6・B4-2．出力は results/text30/）を順に流す。GPU は使わない。
# 計算が失敗したときだけ止める。終わりに logs/text30_cpu.done を書く。
#     tmux new-window -t mer-da -n text_cpu 'cd /home/snics/MER-DA && ./run/text30_cpu.sh 2>&1 | tee -a logs/text30_cpu.log'
cd /home/snics/MER-DA
export MERDA_MAP_ROOT=results/fixc_map CUDA_VISIBLE_DEVICES=""
P="nice -n 10 ./.venv/bin/python"
TEXT="text_roberta fuse-audio-text fuse-video-text fuse-audio-video-text"
ALL7="audio_wavlm video fuse-audio-video $TEXT"
OUTR=results/text30                     # 地図の解析の出力先（MERDA_OUT_ROOT）
REL=$OUTR/explore_relation
rm -f logs/text30_cpu.done
fail() { echo "FAILED: $*"; echo "exit=1" > logs/text30_cpu.done; exit 1; }

# ---- 列 1・列 2 を待つ（失敗の印を見たら，起動し直されるのを待ち続ける）
for c in text30_a text30_b; do
  seen=0
  until grep -qx "exit=0" logs/$c.done 2>/dev/null; do
    if [ $seen = 0 ] && grep -qx "exit=1" logs/$c.done 2>/dev/null; then
      echo "== $(date +%F_%T) logs/$c.done が exit=1．起動し直されるのを待つ"; seen=1
    fi
    sleep 300
  done
  echo "== $(date +%F_%T) $c 済み"
done

# ---- 入力がそろったか（言葉系の C を固定した地図が全ターゲット・全 K で反復 0〜29）
$P - <<'EOF' || fail "言葉系の地図が 30 回そろっていない"
import sys
from pathlib import Path
import pandas as pd
bad = []
for k in "text_roberta fuse-audio-text fuse-video-text fuse-audio-video-text".split():
    fs = sorted(Path("results/fixc_map", k).glob("*.csv"))
    if len(fs) != 4:
        bad.append(f"{k}: ターゲット {len(fs)} 個")
    for f in fs:
        d = pd.read_csv(f, comment="#", dtype={"K": str})
        n = d[d.method != "src_only"].groupby("K").rep.nunique()
        if len(n) != 6 or n.min() < 30:
            bad.append(f"{k}/{f.name}: {n.to_dict()}")
print("\n".join(bad) or "地図：言葉系の 16 ターゲット × 6 つの K で反復 30 回")
sys.exit(1 if bad else 0)
EOF
mkdir -p $REL
{ echo "started $(date +%F_%T)"; echo "commit $(git rev-parse HEAD)"; echo "changed (exp measure run):";
  git status --porcelain -- exp measure run; } > $OUTR/provenance.txt

# ---- 1. 言葉系の地図のまとめ（results/fixc_map/<kind>/summary/ を作り直す．主な種類のまとめは触らない）
echo "== $(date +%F_%T) 地図のまとめ"
for k in $TEXT; do $P exp/summarize_map.py --dir results/fixc_map/$k || fail "summary $k"; done

# ---- 2. 手法の解析（results/explore_relation/ に _text の名前で書く）
echo "== $(date +%F_%T) E37・E39・30 回の判定・D1 の照合"
$P exp/select_teach.py --analyze --kinds $TEXT --suffix _text > logs/text30_cpu_e37.log 2>&1 || fail "E37 analyze"
$P exp/meta5.py --analyze --kinds $TEXT --suffix _text > logs/text30_cpu_e39.log 2>&1 || fail "E39 analyze"
$P exp/extend30.py --kinds $TEXT --suffix _text > logs/text30_cpu_ext30.log 2>&1 || fail "extend30"
$P exp/check_d1_text.py --kinds $TEXT --out $OUTR/d1_check_text.csv > logs/text30_cpu_d1.log 2>&1 || fail "D1 の照合"
echo "== $(date +%F_%T) E31・E32 の 30 回の表"
$P exp/explore_compare_text30.py --kinds $TEXT --suffix _text --require-full > logs/text30_cpu_e31.log 2>&1 || fail "E31 text30"
$P exp/explore_source_value30.py --require-full > logs/text30_cpu_e32.log 2>&1 || fail "E32 kinds30"   # e31_finals30_text.csv を読む

# ---- 3. 地図の解析（MERDA_OUT_ROOT=results/text30．主な種類の行は results/rerun_fixc/ と同じ値になる）
export MERDA_OUT_ROOT=$OUTR
echo "== $(date +%F_%T) E2・E9・E10・E11・E1・E5"
$P exp/explore_decomp.py > $OUTR/log_e2.txt 2>&1 || fail "E2"
$P exp/explore_geometry.py --kinds $ALL7 > $OUTR/log_e9.txt 2>&1 || fail "E9"
$P exp/explore_shape.py > $OUTR/log_e10.txt 2>&1 || fail "E10"
$P exp/explore_mechanism.py > $OUTR/log_e11.txt 2>&1 || fail "E11"
$P exp/side/explore_relation.py > $OUTR/log_e1.txt 2>&1 || fail "E1"
$P exp/side/explore_strategies.py --kinds $TEXT --strategies random leep logme --suffix _text \
  > $OUTR/log_e5.txt 2>&1 || fail "E5"
echo "== $(date +%F_%T) E18・E22・E23・E24・A6・B4"
# 入力を写す：E15 の段階 A は E21（2026-10-09）の言葉系の 30 回（K=1・5・10），E13 の組の表はソースのみの出力から作る（地図に依らない）
for f in e15_pairs.csv e15_lambda.csv; do [ -f $REL/$f ] || cp results/fixc_text/explore_relation/$f $REL/ || fail "copy $f"; done
[ -f $REL/e13_pairs.csv ] || cp results/rerun_fixc/explore_relation/e13_pairs.csv $REL/ || fail "copy e13_pairs.csv"
$P exp/explore_dual.py --kinds $TEXT > $OUTR/log_e18.txt 2>&1 || fail "E18"
$P exp/explore_draws.py --kinds $TEXT > $OUTR/log_e22_e23.txt 2>&1 || fail "E22・E23"
$P exp/explore_select_fixc.py --kinds $TEXT > $OUTR/log_e24.txt 2>&1 || fail "E24"
$P exp/kstar_cluster.py --kinds $TEXT --maps fixc > $OUTR/log_a6.txt 2>&1 || fail "A6"
$P exp/check_room_estimator.py --kinds $TEXT > $OUTR/log_b4.txt 2>&1 || fail "B4"

echo "finished $(date +%F_%T)" >> $OUTR/provenance.txt
echo "exit=0" > logs/text30_cpu.done
echo "== $(date +%F_%T) 終わり"
