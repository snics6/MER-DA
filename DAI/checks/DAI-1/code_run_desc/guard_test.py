"""gate-code（DAI-1 code_v5）：dai1_desc.guard_eval と main の止まり方を，隔離した複製（clone）と偽の置き場で試す。
results/ の下は読むだけ（偽の置き場は scratchpad）。使い方：python guard_test.py <clone> <偽の置き場の親>"""
import json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np

CL = Path(sys.argv[1]).resolve(); PL = Path(sys.argv[2]).resolve()
REAL = Path("/home/snics/MER-DA/results/dai1")
PY = "/home/snics/MER-DA/.venv/bin/python"


def git(*a, cwd=CL):
    return subprocess.run(["git", "-c", "user.name=gc", "-c", "user.email=gc@x", *a], cwd=cwd, capture_output=True, text=True).stdout.strip()


def build(commit, frozen_path, R=2, skip=None, other_commit_at=None):
    place = PL / "dai1"
    if place.exists():
        shutil.rmtree(place)
    (place / "eval" / "units").mkdir(parents=True)
    (place / "select").mkdir()
    for n in ("source", "pseudo"):
        os.symlink(REAL / n, place / n)
    shutil.copy(frozen_path, place / "select" / "frozen.json")
    h = subprocess.run(["sha256sum", str(place / "select" / "frozen.json")], capture_output=True, text=True).stdout.split()[0]
    for T in ("AFEW", "IEMOCAP", "MOSEI", "MELD"):
        for f in range(5):
            for r in range(R):
                base = place / "eval" / "units" / f"{T}_ALL_f{f}_r{r}"
                for w in (1, 2, 3):
                    for ext in ("npz", "csv"):
                        p = Path(f"{base}_p{w}.{ext}")
                        if skip == p.name:
                            continue
                        if w == 3 and ext == "npz":
                            cm = other_commit_at if (other_commit_at and p.name.startswith("MELD_ALL_f4_r1")) else commit
                            np.savez(p, commit=np.array(cm), frozen_sha256=np.array(h))
                        else:
                            p.write_text("x\n")
    return place, h


def guard(place, cwd_root):
    code = f"""
import sys; sys.path.insert(0, r"{cwd_root}/DAI/exp")
import dai1_desc as D
from pathlib import Path
print("RET", D.guard_eval(Path(r"{place}"), list(D.DD.CORPORA)))
"""
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True, cwd=cwd_root)
    return (r.stdout.strip().splitlines() or [""])[-1] + " | " + (r.stderr.strip().splitlines() or [""])[-1]


def main():
    res = {}
    head = git("rev-parse", "HEAD")
    fzp = CL / "DAI" / "frozen" / "dai1_frozen.json"
    os.makedirs(CL / "results", exist_ok=True) if not (CL / "results").exists() else None
    # 置き場は <clone>/results/dai1 ではなく PL/dai1（out.name = dai1 で写しの名前が合う）
    place, h = build(head, fzp)
    res["G1 そろった置き場・汚れなし・単位のコミット = HEAD"] = guard(place, CL)
    # G2：DAI/exp に未追跡のファイル
    (CL / "DAI/exp/_tmp_untracked.py").write_text("#\n")
    res["G2 DAI/exp に未追跡のファイル"] = guard(place, CL)
    os.remove(CL / "DAI/exp/_tmp_untracked.py")
    # G3：単位の 1 つが欠ける
    place, h = build(head, fzp, skip="MELD_ALL_f4_r1_p2.csv")
    res["G3 単位の段 2 の csv が 1 つ欠ける"] = guard(place, CL)
    # G4：単位のコミットが 2 つ
    place, h = build(head, fzp, other_commit_at="deadbeef")
    res["G4 単位のコミットが 2 つ"] = guard(place, CL)
    # G5：置き場の frozen.json が写しと違う
    place, h = build(head, fzp)
    with open(place / "select" / "frozen.json", "a") as fh:
        fh.write("\n")
    res["G5 置き場の frozen.json が DAI/frozen の写しと違う"] = guard(place, CL)
    # G6：評価の回の後に，集計と関係の無いファイルだけを変えたコミット（台本 2 本は同じ）
    place, h = build(head, fzp)
    (CL / "DAI/docs/_note.md").write_text("x\n"); git("add", "DAI/docs/_note.md"); git("commit", "-qm", "test: doc after eval")
    res["G6 評価の回の後のコミット（台本は同じ）で集計"] = guard(place, CL)
    # G7：評価の回の後に dai1_stats.py を変えたコミット
    with open(CL / "DAI/exp/dai1_stats.py", "a") as fh:
        fh.write("# changed after eval\n")
    git("commit", "-qam", "test: stats changed after eval")
    res["G7 評価の回の後に dai1_stats.py を変えた"] = guard(place, CL)
    git("reset", "-q", "--hard", "HEAD~1")
    # G8：手順の 8 のとおり，評価の回のコミットの別の作業木で集計（results は置き場へのリンク）
    wt = PL.parent / "wt"
    if wt.exists():
        git("worktree", "remove", "--force", str(wt))
    git("worktree", "add", "--detach", str(wt), head)
    os.symlink(PL / "dai1", wt / "results" / "dai1")             # 作業木には追跡された results/ の 3 本があるので，dai1 だけをリンクする
    (wt / "data" / "cache").mkdir(parents=True, exist_ok=True)
    for f in sorted(Path("/home/snics/MER-DA/data/cache").glob("*.npz")):
        os.symlink(f, wt / "data" / "cache" / f.name)
    os.symlink("/home/snics/MER-DA/.venv", wt / ".venv")
    res["G8 評価の回のコミットの別の作業木（results・.venv・npz はリンク）"] = guard(wt / "results" / "dai1", wt)
    st_ = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "DAI/exp", "DAI/splits", "measure", "features", "data/metadata"],
                         cwd=wt, capture_output=True, text=True).stdout
    res["G8 作業木の汚れ（台本が見る範囲）"] = repr(st_)
    # G9：main の出力先にすでに中身がある
    dest = PL / "dest_nonempty"; dest.mkdir(exist_ok=True); (dest / "x.csv").write_text("x\n")
    r = subprocess.run([PY, str(wt / "DAI/exp/dai1_desc.py"), "eval", str(wt / "results" / "dai1"), "--out-dir", str(dest)],
                       capture_output=True, text=True, cwd=wt)
    res["G9 出力先にすでに中身がある"] = (r.stderr.strip().splitlines() or [""])[-1]
    # G10：--targets を絞った eval
    r = subprocess.run([PY, str(wt / "DAI/exp/dai1_desc.py"), "eval", str(wt / "results" / "dai1"), "--targets", "AFEW"],
                       capture_output=True, text=True, cwd=wt)
    res["G10 eval で --targets を絞る"] = (r.stderr.strip().splitlines() or [""])[-1]
    git("worktree", "remove", "--force", str(wt))
    for k, v in res.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
