"""gate-code（DAI-5）：check_eval_gate5 を偽の根（必須のファイルの写しと偽の承認の報告）で 11 通り流す。本物の DAI/frozen には書かない。
git の命令は本物の作業場で走らせる。使い方：CUDA_VISIBLE_DEVICES= ./.venv/bin/python DAI/checks/DAI-5/code_run/try_gate.py <偽の根>
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "DAI" / "exp"))
import dai1  # noqa: E402
import dai5  # noqa: E402
import dai_data as DD  # noqa: E402

fake = Path(sys.argv[1])
real_run = subprocess.run


def run_git(cmd, cwd=None, **kw):
    return real_run(cmd, cwd=ROOT, **kw)


dai5.subprocess.run = run_git
dai1.committed_clean = lambda p: True
fz1 = json.load(open(dai5.FROZEN1))
cache = fake.parent / "input_sha.json"
if cache.exists():
    ish = json.load(open(cache))
else:
    ish = {f"{T}_ALL_f{f}_r{r}_p{w}.npz": dai1.sha(dai5.IN_DIR / "eval" / "units" / f"{T}_ALL_f{f}_r{r}_p{w}.npz")
           for T in DD.CORPORA for f in range(DD.N_FOLDS) for r in range(int(fz1[T]["n_seeds_eval"])) for w in (1, 2, 3)}
    json.dump(ish, open(cache, "w"))
HEAD = real_run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def build(case: str) -> Path:
    rt = fake / case
    if rt.exists():
        shutil.rmtree(rt)
    for rel in dai5.EVAL_REQUIRED:
        dst = rt / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if rel == "DAI/frozen/dai5_frozen.json":
            dst.write_text("{}")
        else:
            shutil.copy(ROOT / rel, dst)
    ch = rt / "DAI" / "checks" / "DAI-5"
    ch.mkdir(parents=True, exist_ok=True)
    appr = {}
    for role in dai1.APPROVAL_ROLES:
        word = "OK" if role.startswith("review_") else "通す"
        txt = f"# x\n\n判定：{word}．理由\n"
        if case == "code_not_pass" and role == "code":
            txt = "# x\n\n判定：直してから進む．\n"
        if case == "section_form" and role == "code":
            txt = "# x\n\n## 判定\n通す．理由\n"
        if case == "conditional_ok" and role == "review_ml":
            txt = "# x\n\n判定：条件つき OK\n"
        (ch / f"{role}.md").write_text(txt)
        appr[role] = f"DAI/checks/DAI-5/{role}.md"
    if case == "newer_version":
        (ch / "code_v2.md").write_text("判定：止める\n")
    req = {rel: dai1.sha(rt / rel) for rel in dai5.EVAL_REQUIRED + list(appr.values())}
    if case == "required_sha_bad":
        (rt / "DAI/exp/dai5.py").write_text("changed")
    gate = {"required": req, "approvals": {"DAI-5": appr}, "input_sha256": dict(ish)}
    if case == "input_missing":
        gate["input_sha256"].pop(next(iter(gate["input_sha256"])))
    if case == "input_sha_bad":
        k = next(iter(gate["input_sha256"])); gate["input_sha256"][k] = "0" * 64
    if case == "extra_approval":
        gate["approvals"]["DAI-4"] = {}
    gp = rt / "DAI" / "frozen" / "eval_gate_dai5.json"
    json.dump(gate, open(gp, "w"))
    out = rt / "out"
    (out / "select").mkdir(parents=True)
    (out / "pseudo" / "units").mkdir(parents=True)
    sel = {"_meta": {}}
    pc = "364acce5aa9b80afd11b9a442114a05ee8ee80a3" if case == "pseudo_commit_differs" else HEAD
    for T in DD.CORPORA:
        p = out / "pseudo" / "units" / f"{T}_x_p5.npz"
        p.write_bytes(b"x")
        sel[T] = {"pseudo_commit": pc, "pseudo_unit_sha256": {p.name: dai1.sha(p) if case != "pseudo_sha_bad" else "0" * 64}}
    json.dump(sel, open(out / "select" / "frozen.json", "w"))
    if case == "eval_saved_before_frozen":
        (out / "eval" / "units").mkdir(parents=True)
        e = out / "eval" / "units" / "AFEW_ALL_f0_r0_p5.npz"
        e.write_bytes(b"x")
        os.utime(e, (time.time() - 10 * 365 * 86400,) * 2)
    return rt


cases = ["pass", "section_form", "code_not_pass", "conditional_ok", "newer_version", "required_sha_bad", "input_missing", "input_sha_bad",
         "extra_approval", "pseudo_commit_differs", "pseudo_sha_bad", "eval_saved_before_frozen"]
for case in cases:
    rt = build(case)
    dai5.ROOT, dai5.GATE5 = rt, rt / "DAI" / "frozen" / "eval_gate_dai5.json"
    try:
        h = dai5.check_eval_gate5("DAI/frozen/dai1_frozen.json", rt / "out")     # コミット時刻の照合には本物のコミット済みのファイルを使う
        print(f"{case:26s} -> 通った")
    except SystemExit as e:
        print(f"{case:26s} -> 止まる：{str(e)[:90]}")
