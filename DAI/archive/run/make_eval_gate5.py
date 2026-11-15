"""`DAI/frozen/eval_gate_dai5.json` を作る（実験 5 の設計 4 節の評価の段の守り．実験 1 の `eval_gate.json` と実験 4 の守りと同じ形）。

- required：`dai5.EVAL_REQUIRED` と，`DAI/checks/DAI-5/` の 7 役（review_ml，review_affect，stats，leak，compare，define，code）の最新の版の報告の sha256。
- approvals：{"DAI-5": {役: 報告のパス}}。
- input_sha256：実験 1 の評価の回の p1・p2・p3（4 T × 50 単位）の sha256。
使い方：./.venv/bin/python DAI/run/make_eval_gate5.py（作業木が汚れていないこと．作った後にコミットする）。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DAI" / "exp"))
import dai1  # noqa: E402
import dai5  # noqa: E402
import dai_data as DD  # noqa: E402

folder = ROOT / "DAI" / "checks" / "DAI-5"
approvals = {}
for role in dai1.APPROVAL_ROLES:
    name = dai1.latest_report(folder, role)
    if not name:
        sys.exit(f"{role} の報告が {folder} に無い")
    approvals[role] = (folder / name).relative_to(ROOT).as_posix()
required = {}
for rel in dai5.EVAL_REQUIRED + list(approvals.values()):
    fp = ROOT / rel
    if not fp.exists():
        sys.exit(f"無い：{rel}")
    required[rel] = dai1.sha(fp)
fz1 = json.load(open(dai5.FROZEN1))
inputs = {}
for T in DD.CORPORA:
    for f in range(DD.N_FOLDS):
        for r in range(int(fz1[T]["n_seeds_eval"])):
            for w in (1, 2, 3):
                nm = f"{T}_ALL_f{f}_r{r}_p{w}.npz"
                inputs[nm] = dai1.sha(dai5.IN_DIR / "eval" / "units" / nm)
commit, dirty = dai1.git_commit()
gate = {"_meta": {"made_by": "DAI/run/make_eval_gate5.py", "commit": commit, "dirty": dirty},
        "required": required, "approvals": {"DAI-5": approvals}, "input_sha256": inputs}
dai5.GATE5.parent.mkdir(exist_ok=True)
json.dump(gate, open(dai5.GATE5, "w"), indent=1, ensure_ascii=False)
print("書いた:", dai5.GATE5, "required", len(required), "inputs", len(inputs), "approvals", approvals)
