"""check_eval_gate4 の経路を，偽の固定ファイルと偽の承認の報告で流す（本物の results/dai4 と DAI/frozen は触らない）。"""
import sys, json, os, shutil, time
from pathlib import Path
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai1, dai4 as D4, dai_data as DD
ROOT = D4.ROOT; GT = Path("/home/snics/MER-DA/DAI/checks/DAI-4/code_run/gate_test")
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
# 0. 固定のファイルが無い
os.path.exists(D4.GATE4) and os.remove(D4.GATE4); run("gate ファイル無し")
make_gate(); run("正常")
g = json.load(open(D4.GATE4)); g["required"].pop(data_csv[0]); json.dump(g, open(D4.GATE4, "w")); run("診断の表 csv が required に無い")
make_gate(); (GT / "reports/review_ml.md").write_text("判定：条件つき OK\n"); make_gate(); run("査読が条件つき OK")
(GT / "reports/review_ml.md").write_text("判定：OK\n"); make_gate()
(GT / "reports/code_v2.md").write_text("判定：通す\n"); run("承認に古い版を並べている（code_v2 がある）"); os.remove(GT / "reports/code_v2.md")
make_gate(); g = json.load(open(D4.GATE4)); g.pop("input_sha256"); json.dump(g, open(D4.GATE4, "w")); run("input_sha256 無し")
make_gate(); g = json.load(open(D4.GATE4)); k = sorted(g["input_sha256"])[3]; g["input_sha256"][k] = "0" * 64; json.dump(g, open(D4.GATE4, "w")); run("入力の sha256 が違う")
make_gate(); g = json.load(open(D4.GATE4)); g["input_sha256"].pop(sorted(g["input_sha256"])[0]); json.dump(g, open(D4.GATE4, "w")); run("入力が 1 ファイル欠ける")
make_gate()
# 5. コミット時刻の順序：評価の保存が固定のファイルのコミットより前
x = GT / "out/eval/units/AFEW_ALL_f0_r0_p4.npz"; x.write_bytes(b""); os.utime(x, (1, 1)); run("評価の保存がコミットより前（mtime=1）")
os.utime(x, None); run("評価の保存がコミットより後"); os.remove(x)
