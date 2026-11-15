"""論文（DAI に出す解析論文）に要る実験コードを paper_code/ に写す．元の置き場の相対パスを保ったまま写すので，
台本の中の import と ROOT（ファイルの位置から決める）がそのまま働く．データと結果は写さない．
入口の台本は ENTRIES（論文の節ごと）．入口から import をたどり，同じリポジトリの部品を全部写す．
MANIFEST.tsv に，写したファイル・論文の節・HEAD のコミット・sha256・結果を計算したコミットとの差の有無を書く．
使い方：.venv/bin/python DAI/evidence/make_paper_code.py"""
import ast, hashlib, json, shutil, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_code"
# 論文の節 → (入口の台本, 結果を計算したコミット（記録から．無ければ None）)
ENTRIES = {
    "data": [("features/build_manifest_iemocap.py", None), ("features/build_manifest_mosei.py", None), ("features/build_manifest_meld.py", None),
             ("features/build_manifest_afew.py", None), ("features/build_manifest_extra.py", None), ("features/build_transcript_afew.py", None),
             ("features/build_audio.py", None), ("features/build_video.py", None), ("features/build_text.py", None),
             ("features/build_audio_layers.py", None), ("features/resnet50_affectnet.py", None), ("run/build_data.sh", None),
             ("measure/test_shift.py", None)],
    "sec3-5_pair_setting(E75,E76)": [("exp/map.py", None), ("exp/fixc_map.py", None), ("exp/summarize_map.py", None),
             ("exp/merged_scores.py", None), ("exp/tgt_scores.py", None), ("exp/transfer_scores.py", None),
             ("exp/reanalysis_e75.py", "c452b38"), ("exp/relearn_e76.py", "a14d436"), ("exp/relearn_e76_finish.py", "a14d436"),
             ("exp/analyze_e76.py", "e39e54a")],
    "sec6_world_setting(E77-E80,E79)": [("exp/compare_dai.py", None), ("exp/compare_dai2.py", None), ("exp/e80_teacher_j.py", "66eacca"),
             ("exp/e80_stage_c.py", None), ("exp/analyze_e79.py", "ec0aeb6")],
    "sec7_loco_eval(DAI-1,2,3)": [("DAI/exp/dai1.py", "ba98974"), ("DAI/exp/dai1_select.py", "ba98974"), ("DAI/exp/dai1_stats.py", "ba98974"),
             ("DAI/exp/dai1_desc.py", "ba98974"), ("DAI/exp/dai2.py", None), ("DAI/exp/dai2_grad.py", None), ("DAI/exp/dai2_select.py", None),
             ("DAI/exp/dai2_stats.py", None), ("DAI/exp/dai3.py", "ba98974"), ("DAI/exp/dai_subsets.py", None),
             ("DAI/run/dai1_pseudo.sh", None), ("DAI/run/dai2_pseudo.sh", None), ("DAI/run/dai_eval.sh", None), ("DAI/run/make_eval_gate5.py", None)],
    "sec8_mechanism(DAI-11,12,14)": [("DAI/exp/dai11_probe.py", "67e3553"), ("DAI/exp/table_dai11_cells.py", None), ("DAI/run/dai11_probe.sh", None),
             ("DAI/exp/dai12_probe.py", None), ("DAI/exp/table_dai12_cells.py", None), ("DAI/run/dai12_probe.sh", None),
             ("DAI/exp/dai14.py", "fbd6c07")],
    "sec9_layers(DAI-13)": [("DAI/exp/dai13_probe.py", None), ("DAI/exp/table_dai13_cells.py", None), ("DAI/run/dai13_probe.sh", None),
             ("DAI/run/dai13_probe_r1.sh", None)],
    "evidence": [("DAI/evidence/make_evidence.py", None), ("DAI/evidence/make_paper_code.py", None)],
}
SEARCH = ["", "exp", "measure", "features", "DAI/exp"]


def local_imports(path: Path) -> set:
    if path.suffix != ".py":
        return set()
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return set()
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            names |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            names.add(n.module)
            names |= {f"{n.module}.{a.name}" for a in n.names}
    found = set()
    for name in names:
        parts = name.split(".")
        for base in [path.parent.relative_to(ROOT).as_posix()] + SEARCH:
            for k in range(len(parts), 0, -1):
                cand = ROOT / base / ("/".join(parts[:k]) + ".py")
                if cand.is_file():
                    found.add(cand.relative_to(ROOT).as_posix()); break
    return found


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def main():
    owner, entry_commit = {}, {}
    for sec, items in ENTRIES.items():
        for rel, c in items:
            if not (ROOT / rel).is_file():
                raise SystemExit(f"入口が無い：{rel}")
            stack = [rel]
            while stack:
                r = stack.pop()
                if r in owner and sec in owner[r]:
                    continue
                owner.setdefault(r, set()).add(sec)
                stack += sorted(local_imports(ROOT / r))
            if c:
                entry_commit[rel] = c
    if OUT.exists():
        shutil.rmtree(OUT / "src", ignore_errors=True)
    head = git("rev-parse", "HEAD")
    rows = []
    for r in sorted(owner):
        dst = OUT / "src" / r
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / r, dst)
        h = hashlib.sha256((ROOT / r).read_bytes()).hexdigest()
        c = entry_commit.get(r, "")
        changed = ""
        if c:
            changed = "変わった" if subprocess.run(["git", "diff", "--quiet", c, "HEAD", "--", r], cwd=ROOT).returncode else "同じ"
        dirty = "未コミットの変更あり" if git("status", "--porcelain", "--", r) else ""
        rows.append((r, ";".join(sorted(owner[r])), h, c, changed, dirty))
    for f in ("requirements.txt",):
        shutil.copy2(ROOT / f, OUT / "src" / f)
    with open(OUT / "MANIFEST.tsv", "w") as fh:
        fh.write("path\tsections\tsha256_at_copy\tresult_commit\tsince_result_commit\tworking_tree\n")
        for row in rows:
            fh.write("\t".join(row) + "\n")
    json.dump(dict(copied_at_head=head, n_files=len(rows)), open(OUT / "copy_info.json", "w"), indent=1)
    print(len(rows), "files")


if __name__ == "__main__":
    main()
