"""dai4_stats.guard_eval を，scratch の git clone と偽の評価の保存で流す（コミットの移動・作業木の変更で誤って止まらないか，止まるべきときに止まるか）。"""
import sys, json, subprocess, os
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai1, dai4_stats as D, dai_data as DD
SP = Path("/tmp/claude-1000/-home-snics-MER-DA/b1f27254-32c7-45f0-ac5a-dee1c5e3ec8a/scratchpad")
CL = SP / "clone"
D.ROOT = CL
dai1.committed_clean = lambda p: True
def git(*a, **k): return subprocess.run(["git", *a], cwd=CL, capture_output=True, text=True, **k).stdout.strip()
fz1 = json.load(open(CL / "DAI/frozen/dai1_frozen.json"))
for T in DD.CORPORA: fz1[T]["n_seeds_eval"] = 1
out4 = SP / "fake4" / "dai4"
import shutil; shutil.rmtree(out4.parent, ignore_errors=True)
(out4 / "select").mkdir(parents=True); (out4 / "eval/units").mkdir(parents=True)
(CL / "DAI/frozen/dai4_frozen.json").write_text('{"x":1}'); (out4 / "select/frozen.json").write_text('{"x":1}')
git("add", "-A"); git("commit", "-qm", "eval commit")
E = git("rev-parse", "HEAD")
def units(commit):
    h = dai1.sha(out4 / "select/frozen.json")
    for T in DD.CORPORA:
        for f in range(DD.N_FOLDS):
            b = out4 / "eval/units" / f"{T}_ALL_f{f}_r0"
            np.savez(f"{b}_p4.npz", frozen_sha256=np.array(h), commit=np.array(commit)); Path(f"{b}_p4.csv").write_text("a\n1\n")
def run(label):
    try:
        r = D.guard_eval(out4, fz1); print(f"[{label}] 通った ucommit={r[:7]}")
    except SystemExit as e:
        print(f"[{label}] SystemExit: {str(e)[:150]}")
units(E)
run("評価のコミットそのまま")
(CL / "DAI/docs").mkdir(exist_ok=True); (CL / "DAI/docs/note.md").write_text("n"); 
with open(CL / "DAI/exp/dai4_stats.py", "a") as fh: fh.write("\n# 追加の解析\n")
git("add", "-A"); git("commit", "-qm", "docs + stats script change"); E2 = git("rev-parse", "HEAD")
run("その後のコミット（文書と stats の台本の変更）")
with open(CL / "DAI/exp/dai4_stats.py", "a") as fh: fh.write("# 未コミット\n")
run("stats の台本に未コミットの変更")
git("checkout", "--", "DAI/exp/dai4_stats.py")
with open(CL / "DAI/exp/dai4.py", "a") as fh: fh.write("\n# x\n")
run("dai4.py に未コミットの変更")
git("checkout", "--", "DAI/exp/dai4.py")
with open(CL / "DAI/exp/dai4.py", "a") as fh: fh.write("\n# x\n")
git("add", "-A"); git("commit", "-qm", "dai4 change")
run("dai4.py をコミットで変更")
git("reset", "-q", "--hard", E2)
(CL / "DAI/frozen/dai1_frozen.json").write_text((CL / "DAI/frozen/dai1_frozen.json").read_text() + " ")
git("add", "-A"); git("commit", "-qm", "dai1 frozen change"); run("dai1_frozen.json をコミットで変更")
git("reset", "-q", "--hard", E2)
git("checkout", "-q", "-b", "side", E); (CL / "DAI/docs").mkdir(exist_ok=True); (CL / "DAI/docs/side.md").write_text("s"); git("add", "-A"); git("commit", "-qm", "side"); S = git("rev-parse", "HEAD")
git("checkout", "-q", "-"); git("checkout", "-q", E2) if git("rev-parse", "HEAD") != E2 else None
units(S); run("評価のコミットが HEAD の祖先でない（別の枝）")
units(E)
# 単位のコミットが 2 つ
b = out4 / "eval/units/AFEW_ALL_f0_r0"; np.savez(f"{b}_p4.npz", frozen_sha256=np.array(dai1.sha(out4 / "select/frozen.json")), commit=np.array(S)); run("単位のコミットが 2 つ")
