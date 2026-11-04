"""check_eval_gate4 の経路を，偽の固定ファイルと偽の承認の報告で流す（本物の results/dai4 と DAI/frozen は触らない）。"""
import sys, json, os, shutil, time
from pathlib import Path
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai1, dai4 as D4, dai_data as DD
ROOT = D4.ROOT; GT = Path("/home/snics/MER-DA/DAI/checks/DAI-4/code_run_v2/gate_test")
fz1 = json.load(open(D4.FROZEN1))
# 試験用に n_seeds_eval = 1 の写し
fz1s = json.loads(json.dumps(fz1))
for T in DD.CORPORA: fz1s[T]["n_seeds_eval"] = 1
json.dump(fz1s, open(GT / "dai1_frozen_small.json", "w"))
D4.FROZEN1 = GT / "dai1_frozen_small.json"
# 入力の置き場（実験 1 の評価の回の保存へのリンク．r0 のみ）
ind = GT / "eval_in/eval/units"
for T in DD.CORPORA:
    for f in range(5):
        for w in (1, 2, 3):
            n = f"{T}_ALL_f{f}_r0_p{w}.npz"
            if not (ind / n).exists(): os.symlink(ROOT / "results/dai1/eval/units" / n, ind / n)
D4.IN_DIR = GT / "eval_in"
# 偽の承認の報告
reports = {}
for role in dai1.APPROVAL_ROLES:
    p = GT / "reports" / f"{role}.md"; p.write_text(f"判定：{'OK' if role.startswith('review_') else '通す'}\n"); reports[role] = p.relative_to(ROOT).as_posix()
D4.APPROVAL_FOLDER = GT.relative_to(ROOT).as_posix() + "/reports/"
frozen_rel = "DAI/exp/dai4.py"                       # 固定のファイルの代わりに追跡済みのファイル（コミット時刻の検査用）
data_csv = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "DAI/checks/DAI-4/data").glob("*.csv"))
req_files = [x for x in D4.EVAL_REQUIRED if x != "DAI/frozen/dai4_frozen.json"] + data_csv + list(reports.values()) + [frozen_rel]
def make_gate(**over):
    g = dict(required={r: dai1.sha(ROOT / r) for r in req_files}, approvals={"DAI-4": reports},
             input_sha256={n.name: dai1.sha(n) for n in sorted(ind.glob("*.npz"))})
    g.update(over); json.dump(g, open(GT / "eval_gate_dai4.json", "w"))
D4.GATE4 = GT / "eval_gate_dai4.json"
real_cc = dai1.committed_clean
dai1.committed_clean = lambda p: True if str(p).startswith(str(GT)) else real_cc(p)
D4.EVAL_REQUIRED = [x for x in D4.EVAL_REQUIRED if x != "DAI/frozen/dai4_frozen.json"]
# 偽の dai1 の入力は r0 のみなので，want の集合も r0 のみ（n_seeds_eval=1）
def run(label, **kw):
    try:
        r = D4.check_eval_gate4(frozen_rel, GT / "out"); print(f"[{label}] 通った sha={r[:8]}")
    except SystemExit as e:
        print(f"[{label}] SystemExit: {e}")

import subprocess
HEAD = subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True).stdout.strip()
B018 = subprocess.run(["git","rev-parse","b018ae0"],cwd=ROOT,capture_output=True,text=True).stdout.strip()
pu = GT / "out/pseudo/units"
def make_frozen(pc=HEAD, bad_hash=None, drop_file=None):
    d = {"_meta": {}}
    for T in DD.CORPORA:
        nm = f"{T}_X_f0_r0_p4.npz"; fp = pu / nm; fp.write_bytes(T.encode() * 10)
        h = dai1.sha(fp) if bad_hash != T else "0" * 64
        d[T] = {"pseudo_commit": pc, "pseudo_unit_sha256": {nm: h}}
        if drop_file == T: fp.unlink()
    json.dump(d, open(GT / "out/select/frozen.json", "w"))
make_gate()
make_frozen(); run("正常（pseudo_commit = HEAD，sha256 一致）")
make_frozen(pc=B018); run("pseudo_commit = b018ae0（dai4.py に差）")
make_frozen(bad_hash="MELD"); run("pseudo の保存の sha256 が違う")
make_frozen(drop_file="AFEW"); run("pseudo の保存が無い")
make_frozen(); json.dump({"_meta":{}}, open(GT / "out/select/frozen.json","w")); 
try:
    D4.check_eval_gate4(frozen_rel, GT / "out")
except KeyError as e:
    print("[固定のファイルに T が無い] KeyError", e)
except SystemExit as e:
    print("[固定のファイルに T が無い] SystemExit", e)
