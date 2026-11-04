#!/bin/bash
# gate-code（DAI-2 code_v3）：評価の段の守りを，隔離した clone と中身の空の偽の単位で試す（単位の commit・gate_sha256 の確かめの有無）．
# 偽の単位は npz に commit・gate_sha256・frozen_sha256 だけを持つ（守りを通ると Unit の読み込みで KeyError になる）．results/ の本物は読むだけ．
set -u
W=$1; VARIANT=$2    # original | patched
REAL=/home/snics/MER-DA; PY=$REAL/.venv/bin/python
rm -rf "$W"; mkdir -p "$W"; CL=$W/clone
git clone -q "$REAL" "$CL"
if [ "$VARIANT" = patched ]; then cp $REAL/DAI/checks/DAI-2/code_run_v3/patched/dai2_stats_patched.py "$CL/DAI/exp/dai2_stats.py"; else cp $REAL/DAI/exp/dai2_stats.py "$CL/DAI/exp/"; fi
g() { git -C "$CL" -c user.name=gc -c user.email=gc@x "$@"; }
SHA=$(sha256sum "$CL/DAI/exp/dai2_stats.py" | cut -d' ' -f1)
echo "{\"required\": {\"DAI/exp/dai2_stats.py\": \"$SHA\"}}" > "$CL/DAI/frozen/eval_gate.json"
g add DAI/exp/dai2_stats.py DAI/frozen/eval_gate.json; g commit -qm "test: stats + gate"
HEAD0=$(g rev-parse HEAD)
GATE=$(sha256sum "$CL/DAI/frozen/eval_gate.json" | cut -d' ' -f1)
F1=$(sha256sum "$CL/DAI/frozen/dai1_frozen.json" | cut -d' ' -f1); F2=$(sha256sum "$CL/DAI/frozen/dai2_frozen.json" | cut -d' ' -f1)
mkdir -p "$CL/results/dai1/stats" "$CL/results/dai1/eval/units" "$CL/results/dai2/eval/units"
ln -s "$REAL/results/dai1/select" "$CL/results/dai1/select"; ln -s "$REAL/results/dai2/select" "$CL/results/dai2/select"
$PY - "$CL" "$HEAD0" "$GATE" "$F1" "$F2" <<'P'
import sys, json, numpy as np
cl, head, gate, f1, f2 = sys.argv[1:]
fz1 = json.load(open(f"{cl}/DAI/frozen/dai1_frozen.json"))
def mk(path, c, f):
    np.savez(path + ".npz", commit=np.array(c), gate_sha256=np.array(gate), frozen_sha256=np.array(f))
    open(path + ".csv", "w").write("a\n1\n")
for T in ("AFEW", "IEMOCAP", "MOSEI", "MELD"):
    R = int(fz1[T]["n_seeds_eval"])
    for f in range(5):
        for r in range(R):
            for p in ("p1", "p3"): mk(f"{cl}/results/dai1/eval/units/{T}_ALL_f{f}_r{r}_{p}", head, f1)
            for p in ("lin", "coral") + (("map", "xmuda") if r < 3 else ()): mk(f"{cl}/results/dai2/eval/units/{T}_ALL_f{f}_r{r}_{p}", head, f2)
open(f"{cl}/results/dai1/stats/families.csv", "w").write(f"family,T,student,K,effect,lo,hi,p_holm,commit,dirty\n3,AFEW,A,0,0,0,0,1,{head},False\n")
P
run() { (cd "$CL" && $PY DAI/exp/dai2_stats.py eval "$W/out" 2>&1 | tail -2 | tr '\n' ' '); }
echo "[$VARIANT] H1 単位の commit = 今の commit（正しい手順の作業木）: $(run)"
echo '# changed after eval' >> "$CL/DAI/exp/dai1_stats.py"; g commit -qam "test: dai1_stats changed after eval"
echo "[$VARIANT] H2 評価の回の後に dai1_stats.py を変えてコミットした HEAD: $(run)"
g reset -q --hard "$HEAD0"
$PY - "$CL" <<'P'
import sys, numpy as np
p = f"{sys.argv[1]}/results/dai2/eval/units/MELD_ALL_f3_r5_lin.npz"
np.savez(p, commit=np.array("0123456789abcdef"), gate_sha256=np.array("x"), frozen_sha256=np.array(""))
P
echo "[$VARIANT] H3 単位 1 つだけ別のコミットと別の gate_sha256: $(run)"
