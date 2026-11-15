"""DAI/ の整理（2026-11-15）．案は docs/記録/整理の案_2026-11-14.md．

MER-DA の作業木（この台本を置いた worktree）の中で，DAI/ を「データと大きい結果を除けば単独で再現できる」形に組み替える．
  1. 移す・写す：DAI/ の中は git mv，DAI/ の外（MER-DA/exp・measure・features・data/metadata など）は写す（元は消さない）．
  2. 置き場の行だけを書き換える：対応表 REWRITE_RULES を，論文に使う台本（exp・run・tools）の全文に一度だけ当てる．
     計算の式には触らない．設計の文書・結果の文・検査の報告・固定のファイル（frozen/）は書き換えない（中身の sha256 を保つ）．
  3. 対応表を docs/整理の対応表_2026-11-15.tsv に書く（旧の置き場，新の置き場，種類，旧の sha256，新の sha256）．
使い方：python DAI/tools/reorg_2026-11-15.py <worktree の直下>   （1 回だけ流す．流し直すときは worktree を作り直す）
"""
import hashlib, re, shutil, subprocess, sys
from pathlib import Path

W = Path(sys.argv[1]).resolve()           # MER-DA の worktree の直下
D = W / "DAI"

# ---------------------------------------------------------------- 論文に使う台本と使わない台本
ARCHIVE_EXP = {"dai4.py", "dai4_stats.py", "dai5.py", "dai5_stats.py", "dai5_variant_argmax.py", "dai6_probe.py", "dai7_probe.py",
               "dai7_probe_v2.py", "dai8_probe.py", "dai9_probe.py", "dai10_probe.py", "explore_dai7_ceiling.py", "explore_dai89_drift.py",
               "table_dai89_cells.py", "table_dai10_cells.py", "explore_consensus_emotion.py", "explore_consensus_eval.py",
               "explore_consensus_pseudo.py", "explore_teacher_adv.py", "explore_fusion.py", "explore_fusion_grid.py", "explore_fusion_kshot.py",
               "explore_fusion_select.py", "explore_reliability_loo.py", "explore_depfusion.py"}
ARCHIVE_RUN = {"dai5.sh", "make_eval_gate5.py", "dai6_probe.sh", "dai7_probe.sh", "dai7_probe_v2.sh", "dai8_probe.sh", "dai9_probe.sh",
               "dai10_probe.sh", "dai19_arm.sh", "dai19_post.sh"}
ARCHIVE_FROZEN = {"dai5_frozen.json", "dai5_eval_inputs_sha256.json", "eval_gate_dai5.json"}
ARCHIVE_CHECKS = {"DAI-4", "DAI-5", "DAI-6", "DAI-7", "DAI-8", "DAI-9", "DAI-10", "DAI-19", "DAI-explore"}

# 論文の前半（E75・E76・E77〜E80・E79・付録の E32）の台本．入口から import をたどった 44 本（MER-DA/exp から写す）
FRONT_EXP = """analyze_e76 analyze_e79 baselines_distill belief_check belief_correct check_lll_cues combine_drafts committee compare_b1 compare_dai
compare_dai2 distill_belief e80_stage_c e80_teacher_j explore_compare explore_direction_bound explore_factor explore_geometry explore_mechanism
explore_pairs explore_source_value explore_source_value30 explore_synergy explore_uar_link geom_e0 interventions logreg map meta5
multimodal_distill pair_table reanalysis_e75 relearn_e76 relearn_e76_finish select_baselines select_herding select_kshot select_labelfree
select_teach summarize_map teach_ablate teach_stack teach_student39 text_cond2""".split()
FRONT_RUN = ["e77.sh", "e77b.sh", "e77h.sh", "e78.sh", "e78b.sh", "e78h.sh", "build_data.sh"]
FRONT_CHECKS = ["E75", "E76", "E77", "E78", "E79", "E80"]

# ---------------------------------------------------------------- 文書の行き先（DAI/docs/<名前> → docs/<下位>/<名前>）
DOCS = {}
for n in ["実験1_設計.md", "実験2_設計.md", "実験3_設計.md", "実験11_下調べ_設計.md", "実験12_下調べ_設計.md", "実験13_下調べ_設計.md",
          "実験14_設計.md", "実験15_設計.md", "実験16_下調べ_設計.md", "実験16・17_下調べ_K10_追補.md", "実験17_下調べ_設計.md",
          "実験18_下調べ_設計.md", "実験20_設計.md", "実験21_設計.md"]:
    DOCS[n] = "docs/設計/" + n
for n in ["結果の文の下書き_2026-11-05.md", "実験11_下調べ_全セルの表_2026-11-09.md", "実験11_下調べ_結果の文の下書き_2026-11-09.md",
          "実験12_下調べ_全セルの表_2026-11-09.md", "実験12_下調べ_結果の文の下書き_2026-11-09.md", "実験13_下調べ_全セルの表_2026-11-10.md",
          "実験13_下調べ_結果の文の下書き_2026-11-10.md", "実験14_結果の文の下書き_2026-11-12.md", "実験15_結果の文の下書き_2026-11-13.md",
          "実験16_下調べ_結果の文の下書き_2026-11-14.md", "実験16_下調べ_結果の表_2026-11-14.md",
          "実験16・17_下調べ_K10_結果の文の下書き_2026-11-14.md", "実験16・17_下調べ_結果の表_K5・10・25_2026-11-14.md",
          "実験17_下調べ_結果の文の下書き_2026-11-14.md", "実験18_下調べ_結果の文の下書き_2026-11-14.md", "実験18_下調べ_結果の表_2026-11-14.md",
          "実験20_結果の文の下書き_2026-11-15.md", "実験20_結果の表_2026-11-15.md", "実験21_結果の文の下書き_2026-11-15.md",
          "実験21_結果の表_2026-11-15.md", "資料_2026-11-10_追加の結果の文.md"]:
    DOCS[n] = "docs/結果/" + n
for n in ["主張_2026-11-12.md", "主張_採用した説明_2026-11-13.md"]:
    DOCS[n] = "docs/主張/" + n
for n in ["計画_2026-11-04.tex", "計画_2026-11-04.pdf"]:
    DOCS[n] = "docs/計画/" + n
for n in ["説明_主張_大学1年生向け_2026-11-13.pdf", "説明_主張_大学1年生向け_2026-11-13.tex", "説明_実験16の案_図解_2026-11-13.pdf",
          "説明_実験16の案_図解_2026-11-13.tex", "説明_実験16・17_結果の表_16.tex", "説明_実験16・17_結果の表_17.tex",
          "説明_提案_図表_6コーパス.tex", "説明_提案_図表_まとめ.tex", "説明_提案_図表_下調べ.tex"]:
    DOCS[n] = "docs/説明/" + n
for n in ["論文のセッションへの伝言_2026-11-14.md", "難しさの記録_2026-11-05.md", "整理_棚卸し_コードと結果_2026-11-14.md",
          "整理_棚卸し_文書と検査_2026-11-14.md", "整理の案_2026-11-14.md"]:
    DOCS[n] = "docs/記録/" + n
DOCS["実験の地図.md"] = "docs/実験の地図.md"
for n in ["実験4_設計.md", "実験5_結果の文の下書き_2026-11-07.md", "実験5_設計.md", "実験6_下調べ_設計.md", "実験7_下調べ_結果の文の下書き_2026-11-08.md",
          "実験7_下調べ_設計.md", "実験8_下調べ_設計.md", "実験8・9_下調べ_全セルの表_2026-11-08.md", "実験8・9_下調べ_結果の文の下書き_2026-11-08.md",
          "実験9_下調べ_設計.md", "実験10_下調べ_全セルの表_2026-11-08.md", "実験10_下調べ_結果の文の下書き_2026-11-08.md", "実験10_下調べ_設計.md",
          "実験19_下調べ_設計.md"]:
    DOCS[n] = "archive/docs/" + n
DROP_DOCS = {"実験13_下調べ_結果の文の下書き_2026-11-09.md"}      # 値の無い第 0 版の骨組み（どこからも引かれない．git の履歴に残る）

# MER-DA の直下の文書の写し（docs/参照/）と，論文の前半の設計（docs/設計/前半/）
MER_DOCS = ["methods.md", "seen-data.md", "related-work.md", "exploratory-analyses.md", "results.md", "researchvision.md"]
FRONT_DESIGN = {"DAI_paper/steps/step2_design.md": "docs/設計/前半/step2_design.md",
                "DAI_paper/steps/step4_design.md": "docs/設計/前半/step4_design.md",
                "DAI_paper/reviews/comparison-design/proposal_v8.md": "docs/設計/前半/proposal_v8.md"}

TOOLS_KEEP_NAME = True     # DAI/docs/tools_pending/<名前> → tools/<名前>


# ---------------------------------------------------------------- 置き場の書き換え（台本の全文に 1 回だけ当てる）
def map_dai_path(rest: str) -> tuple[str, int]:
    """'DAI/' の後ろの文字列 rest の先頭のパスを新しい置き場に置き換える．戻り：(新しい頭, 消費した文字数)．"""
    if rest.startswith("docs/tools_pending"):
        return "tools", len("docs/tools_pending")
    if rest.startswith("docs/"):
        for n in sorted(DOCS, key=len, reverse=True):
            if rest.startswith("docs/" + n):
                return DOCS[n], len("docs/" + n)
        return "docs/", len("docs/")
    if rest.startswith("docs"):
        return "docs", 4
    for top, arch in (("exp/", ARCHIVE_EXP), ("run/", ARCHIVE_RUN), ("frozen/", ARCHIVE_FROZEN)):
        if rest.startswith(top):
            for n in sorted(arch, key=len, reverse=True):
                if rest.startswith(top + n):
                    return "archive/" + top + n, len(top + n)
            return top, len(top)
    if rest.startswith("checks/"):
        for n in sorted(ARCHIVE_CHECKS, key=len, reverse=True):
            if rest.startswith("checks/" + n + "/") or rest == "checks/" + n or re.match(re.escape("checks/" + n) + r"(?![\w-])", rest):
                return "archive/checks/" + n, len("checks/" + n)
        return "checks/", len("checks/")
    for top, new in (("exp", "exp"), ("run", "run"), ("frozen", "frozen"), ("splits", "splits"), ("checks", "checks"), ("slides", "slides"),
                     ("paper", "paper"), ("reviews", "reviews"), ("CLAUDE.md", "CLAUDE.md"), (".claude", ".claude"),
                     ("evidence", "archive/evidence")):
        if rest.startswith(top):
            return new, len(top)
    if rest.startswith("reports/"):
        return "docs/記録/", len("reports/")
    return None, 0


MER_DOC_RE = re.compile(r"(?<![\w/.\-])docs/(" + "|".join(re.escape(n) for n in MER_DOCS) + r")")
FRONT_DESIGN_RE = re.compile("|".join(re.escape(k) for k in FRONT_DESIGN))
DAI_RE = re.compile(r"(?<![\w/.\-])DAI/")
SEG_RE = re.compile(r'/ "DAI" / "docs" / "([^"]+)"')


def rewrite(text: str) -> tuple[str, list]:
    log = []
    # (1) ROOT / "DAI" / "docs" / "<名前>" → ROOT / "docs" / "<下位>" / "<名前>"
    def seg_docs(m):
        n = m.group(1)
        new = DOCS.get(n)
        if new is None:
            if n == "tools_pending":
                return '/ "tools"'
            raise SystemExit(f"文書の行き先が無い：{n}")
        parts = new.split("/")
        return "/ " + " / ".join(f'"{p}"' for p in parts)
    text, k = SEG_RE.subn(seg_docs, text); k and log.append(("seg_docs", k))
    for a, b in (('/ "DAI" / "frozen"', '/ "frozen"'), ('/ "DAI" / "splits"', '/ "splits"'), ('/ "DAI" / "exp"', '/ "exp"'),
                 ('/ "DAI" / "checks"', '/ "checks"'), ('/ "DAI" / "run"', '/ "run"')):
        k = text.count(a)
        if k:
            text = text.replace(a, b); log.append((a, k))
    # (2) 絶対パスと ROOT の決め方
    for a, b in (('sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")', 'sys.path.insert(0, str(Path(__file__).resolve().parent))'),
                 ('Path("/home/snics/MER-DA/DAI/exp")', 'Path(__file__).resolve().parent'),
                 ('Path("/home/snics/MER-DA")', 'Path(__file__).resolve().parents[1]'),
                 ('Path(__file__).resolve().parents[2]', 'Path(__file__).resolve().parents[1]'),
                 ('HERE.parents[1]', 'HERE.parent'),
                 ('cd /home/snics/MER-DA\n', 'cd "$(dirname "$0")/.."\n')):
        k = text.count(a)
        if k:
            text = text.replace(a, b); log.append((a, k))
    # (3) git show <commit>:DAI/exp/... は cwd からの相対にする（MER-DA の中でも，DAI だけのリポジトリでも同じに動く）
    k = text.count(':DAI/exp/')
    if k:
        text = text.replace(':DAI/exp/', ':./exp/'); log.append((":DAI/exp/ -> :./exp/", k))
    # (4) 論文の前半の設計（DAI_paper の下）
    text, k = FRONT_DESIGN_RE.subn(lambda m: FRONT_DESIGN[m.group(0)], text); k and log.append(("front_design", k))
    # (5) MER-DA の直下の文書 → docs/参照/
    text, k = MER_DOC_RE.subn(lambda m: "docs/参照/" + m.group(1), text); k and log.append(("mer_docs", k))
    # (6) 残りの DAI/<置き場>
    out, i, n = [], 0, 0
    for m in DAI_RE.finditer(text):
        new, used = map_dai_path(text[m.end():])
        if new is None:
            continue
        out.append(text[i:m.start()]); out.append(new); i = m.end() + used; n += 1
    out.append(text[i:])
    text = "".join(out); n and log.append(("DAI/ prefix", n))
    return text, log


# ---------------------------------------------------------------- 実行
def sh(*a, cwd=W):
    r = subprocess.run(a, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"失敗：{' '.join(map(str, a))}\n{r.stderr}")
    return r.stdout


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else ""


def main():
    rows = []           # (旧, 新, 種類, 旧 sha, 新 sha)

    def gmv(src_rel, dst_rel, kind):
        s, t = W / src_rel, W / dst_rel
        t.parent.mkdir(parents=True, exist_ok=True)
        rows.append([src_rel, dst_rel, kind, sha(s)])
        sh("git", "mv", "-k", src_rel, dst_rel)

    def gcp(src_rel, dst_rel, kind):
        s, t = W / src_rel, W / dst_rel
        t.parent.mkdir(parents=True, exist_ok=True)
        rows.append([src_rel, dst_rel, kind, sha(s)])
        shutil.copy2(s, t)
        sh("git", "add", "-f", dst_rel)

    tracked = set(sh("git", "ls-files", "-z", "DAI").split("\0")) - {""}

    # 台本
    for p in sorted((D / "exp").glob("*.py")):
        if p.name in ARCHIVE_EXP:
            gmv(f"DAI/exp/{p.name}", f"DAI/archive/exp/{p.name}", "archive")
    for n in FRONT_EXP:
        gcp(f"exp/{n}.py", f"DAI/exp/{n}.py", "copy_front")
    for p in sorted((D / "run").iterdir()):
        if p.name in ARCHIVE_RUN:
            gmv(f"DAI/run/{p.name}", f"DAI/archive/run/{p.name}", "archive")
    for n in FRONT_RUN:
        gcp(f"run/{n}", f"DAI/run/{n}", "copy_front")
    for n in ("dai14_run.sh", "dai15_run.sh"):          # git に入っていなかった起動の台本（logs/ にだけあった）
        src = Path("/home/snics/MER-DA/logs") / n
        t = D / "run" / n
        shutil.copy2(src, t); rows.append([f"logs/{n}", f"DAI/run/{n}", "copy_untracked_log", sha(src)]); sh("git", "add", "-f", f"DAI/run/{n}")
    for p in sorted((D / "frozen").iterdir()):
        if p.name in ARCHIVE_FROZEN:
            gmv(f"DAI/frozen/{p.name}", f"DAI/archive/frozen/{p.name}", "archive")
    for p in sorted((D / "docs" / "tools_pending").glob("*.py")):
        gmv(f"DAI/docs/tools_pending/{p.name}", f"DAI/tools/{p.name}", "move")

    # 部品・特徴・データ・環境（MER-DA の直下から写す）
    for n in ("data.py", "split.py", "shift.py", "ablate.py", "audit_manifest.py", "check_speakers.py", "test_shift.py"):
        gcp(f"measure/{n}", f"DAI/measure/{n}", "copy_common")
    for p in sorted((W / "features").iterdir()):
        if p.is_file() and p.suffix in (".py", ".sh"):
            gcp(f"features/{p.name}", f"DAI/features/{p.name}", "copy_common")
    for p in sorted((W / "data" / "metadata").iterdir()):
        gcp(f"data/metadata/{p.name}", f"DAI/data/metadata/{p.name}", "copy_common")
    for p in sorted((W / "data" / "cache").glob("faces_*.csv.gz")):
        gcp(f"data/cache/{p.name}", f"DAI/data/cache/{p.name}", "copy_common")
    for n in ("README.md", "weights.sha256"):
        gcp(f"data/{n}", f"DAI/data/{n}", "copy_common")
    gcp("requirements.txt", "DAI/requirements.txt", "copy_common")

    # 文書
    for p in sorted((D / "docs").iterdir()):
        if not p.is_file():
            continue
        rel = f"DAI/docs/{p.name}"
        if rel not in tracked:
            continue                                   # LaTeX の中間生成物（git の外）は動かさない
        if p.name in DROP_DOCS:
            rows.append([rel, "", "delete", sha(p)]); sh("git", "rm", "-q", rel); continue
        if p.name not in DOCS:
            raise SystemExit(f"行き先の無い文書：{p.name}")
        gmv(rel, "DAI/" + DOCS[p.name], "move")
    for p in sorted((D / "reports").glob("*.md")):
        gmv(f"DAI/reports/{p.name}", f"DAI/docs/記録/{p.name}", "move")
    for n in MER_DOCS:
        gcp(f"docs/{n}", f"DAI/docs/参照/{n}", "copy_snapshot")
    for k, v in FRONT_DESIGN.items():
        gcp(k, "DAI/" + v, "copy_front")
    for p in sorted((D / "evidence").rglob("*")):
        if p.is_file() and f"DAI/{p.relative_to(D).as_posix()}" in tracked:
            r = p.relative_to(D / "evidence").as_posix()
            gmv(f"DAI/evidence/{r}", f"DAI/archive/evidence/{r}", "archive")

    # 検査の報告：試走の出力（大きいデータとリンク）を git から外し，論文に使わない実験の報告は archive へ
    keep_ext = {".md", ".py", ".diff", ".txt", ".sh", ".json", ".csv", ".tsv", ".tex"}
    for rel in sorted(t for t in tracked if t.startswith("DAI/checks/")):
        p = W / rel
        big = p.is_symlink() or p.suffix not in keep_ext or (p.is_file() and p.stat().st_size > 1_000_000)
        if big:
            rows.append([rel, "", "untrack_checks_output", "" if p.is_symlink() else sha(p)])
            sh("git", "rm", "-q", "--cached", rel)
            if p.is_symlink() or p.exists():
                p.unlink()
    for n in sorted(ARCHIVE_CHECKS):
        if (D / "checks" / n).exists():
            for rel in sorted(sh("git", "ls-files", "-z", f"DAI/checks/{n}").split("\0")):
                if rel:
                    gmv(rel, rel.replace(f"DAI/checks/{n}/", f"DAI/archive/checks/{n}/", 1), "archive")
    for e in FRONT_CHECKS:
        for p in sorted((W / "checks" / e).glob("*.md")):
            if f"checks/{e}/{p.name}" in set(sh("git", "ls-files", f"checks/{e}/{p.name}").split("\n")):
                gcp(f"checks/{e}/{p.name}", f"DAI/checks/{e}/{p.name}", "copy_front")

    # 置き場の行の書き換え（論文に使う台本と道具だけ）
    targets = sorted([*(D / "exp").glob("*.py"), *(D / "run").iterdir(), *(D / "tools").glob("*.py")])
    rewrites = []
    for p in targets:
        if not p.is_file() or p.name == "reorg_2026-11-15.py":
            continue
        old = p.read_text(encoding="utf-8")
        new, log = rewrite(old)
        if p.parent.name == "tools":                    # 道具は tools/ にあるので，読み込む台本の置き場は 1 つ上の exp/
            a = 'sys.path.insert(0, str(Path(__file__).resolve().parent))'
            if a in new:
                new = new.replace(a, 'sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp"))')
                log.append(("tools sys.path -> ../exp", 1))
        if new != old:
            p.write_text(new, encoding="utf-8")
            rewrites.append((p.relative_to(W).as_posix(), log))
    # 新の sha256
    for r in rows:
        r.append(sha(W / r[1]) if r[1] else "")
    out = D / "docs" / "整理の対応表_2026-11-15.tsv"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("old_path\tnew_path\tkind\told_sha256\tnew_sha256\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")
    with open(D / "docs" / "整理の書き換えの記録_2026-11-15.tsv", "w", encoding="utf-8") as fh:
        fh.write("file\trule\tcount\n")
        for f, log in rewrites:
            for rule, k in log:
                fh.write(f"{f}\t{rule}\t{k}\n")
    print(f"対応表 {len(rows)} 行，書き換えた台本 {len(rewrites)} 本")


if __name__ == "__main__":
    main()
