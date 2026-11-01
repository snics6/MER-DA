"""【本番】E79（CPU）：DAI 論文の比較実験の判定の解析（ステップ 4）。

設計は `DAI_paper/steps/step4_design.md`（関門 1 の後に `docs/exploratory-analyses.md` の E79 に写す）．正本の比較の設計は
`DAI_paper/reviews/comparison-design/proposal_v8.md`．入力は E77（`exp/compare_dai.py`）と E78（`exp/compare_dai2.py`）の出力で，
この台本は学び直しをしない．GPU を使わない（起動の最初に CUDA_VISIBLE_DEVICES を空にする）。

段（--stage．この順に流す．各段の出力は `--out/<段>/` に書き，既にあれば止まる）
  delta   ：判定の基準 δ_d・δ_u を，E77 の仮のターゲットの none の出力（`<T>_ps_<c>_r<r>_none.npz`）と `<T>_pseudo.csv` だけから計算する．
            入力の場所の中でほかのファイルを開こうとしたら止まる（`sys.addaudithook` で open を見張る．段の最初に，禁じたファイルを
            開けないことを自分で確かめる）．出力の `delta.csv` の sha256 の先頭 16 文字を表示する．これを文書に書いてコミットしてから次の段に進む
  select  ：仮のターゲットの値（`<T>_pseudo.csv`）だけで，(生徒, 選ぶ族, K) ごとに設定を選ぶ．none を含む版（incl）と除く版（excl），
            MOSEI・MELD では IEMOCAP 台本を仮のターゲットから除いた版（ns_incl・ns_excl）．固定版の行と，対照の対応する行も決める．
            評価のターゲットの値は開かない（delta と同じ仕掛け）．δ のコミットを確かめてから動く
  measure ：評価のターゲットのテストの発話ごとの対数の確率から，反復ごとに UAR・6 対の d と o・位置の損・向きの損を全行で測り，
            選んだ行について 2 段のブートストラップ（話者と反復）の複製を作る．D（候補のラベルで学んだ LDA の向きをテストで測った d）は抽出の前に固定する
  judge   ：設計の文書 7.2 節の表を機械的に当てて，P4・P4'・P5-1〜P5-4・除去実験・5.3 節の集計を判定する．感度の行と付録の表も出す

    ./.venv/bin/python exp/analyze_e79.py --stage delta  [--runs …] [--e77 名前=場所 …] [--e78 名前=場所 …] [--out results/e79]
    ./.venv/bin/python exp/analyze_e79.py --stage select
    ./.venv/bin/python exp/analyze_e79.py --stage measure [--boot 1000]
    ./.venv/bin/python exp/analyze_e79.py --stage judge

段の間の照合（関門 1 の指摘 L-2・L-3・I12-2）：select は δ のコミットと delta の台本の版を，measure は δ・`chosen.csv`・`fixed.csv` のコミットと
前の段の台本の版と δ の sha256 を，judge はさらに measure が使った δ・`chosen.csv`・`fixed.csv` の sha256 と 4 段の台本の版を確かめ，外れたら止まる。
`exp/` か `measure/` に未コミットの変更があると止まる（--allow-dirty は試し用）．BLAS の糸の数は既定で 4（環境変数で変えられる．provenance に残す）。

試走（揃っていない出力）では --allow-incomplete（欠けを止めずに記録する），--skip-commit-check（コミットの確認を飛ばし，provenance に残す），
--allow-dirty を使う。
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""                                         # GPU を使わない（読み込むモジュールが torch を使っても CPU に置く）
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):         # E77・E78 と CPU を取り合わないよう糸の数を決める（I10-1）
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse, stats  # noqa: E402
import scipy.linalg  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.special import logsumexp  # noqa: E402
from sklearn.covariance import LedoitWolf  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
from data import GROUP_SPEC, groups_for, load_any  # noqa: E402
from explore_pairs import ba, own_separation  # noqa: E402
from explore_synergy import PAIRS, d_o  # noqa: E402
from multimodal_distill import WORLDS, blocks  # noqa: E402
from shift import uar as shift_uar  # noqa: E402

RUNS = ("audio_wavlm", "video", "text_roberta", "fuse-audio-video-text", "fuse-audio-video")
KS = ("0", "1", "5", "10", "25")
REPS = tuple(range(10))
PSEUDO_REPS = tuple(range(5))
MEFFS = ("100", "500", "2000", "n")
WGTS = ("bal", "raw", "hard0.5", "hard0.7", "hard0.9")
SEEDED = {"dann", "mme", "mapctl", "xmuda", "cot", "cotctl", "cotfix", "cotzero", "cot_g", "cotctl_g", "cotfix_g", "cotzero_g"}
SEEDS = (0, 1, 2)
GROUP_B = ("entmin", "self", "im", "t3a", "lp", "coral", "dann", "mme")       # P4 の 8 手法（T3A は主の版）
P4_KS = ("0", "5")
P5_KS = ("0", "5", "25")
VNORM_MIN = 0.9                                                                # Co-teaching の正規化した重なり ṽ のしきい値（設計の文書 3.2 節）
DISAGREE_MIN = 0.01                                                            # DA の項が予測を変えたとみなす食い違いの割合の下限（K ≥ 1 は種の間の食い違いの最大も超えること）
MODEL_RATIO_MAX = 1 / 3                                                        # 2 つの山の模型：E75 の決まり（模型の誤差 ÷ 損 < 1/3）．誤差は模型が正しいときの標本の揺れを引いてから測る
MODEL_UNIT_FRAC = 0.5                                                          # 単位の印：|g| と |Δe| が δ_d のこの割合を超えたら「模型に依る」
VR_RANGE = (0.5, 2.0)                                                          # 分散の比（記述だけ）
NULL_DRAWS = 100                                                               # 模型が正しいときの誤差の分布を作る抽出の回数
CORAL_FIX_R = 64
CT_TAUS_GRID = (0.2, 0.4, 0.6)
TOL_LOGP, TOL_UAR = 1e-5, 1e-9                                                 # E77 と E78 の同じはずの値の許容幅（I14-1）
SEEN = {"P4": {"massbal", "entmin", "self", "coral"}, "P4'": {"massbal", "entmin", "self", "coral", "im"}}   # 設計の文書 2 節で「UAR を見た後の予言」の手法（L-4）
REFS = ("ref_srconly", "ref_merged", "ref_tgtonly", "ref_wstar", "teacher")
CONSTS = {"VNORM_MIN": VNORM_MIN, "DISAGREE_MIN": DISAGREE_MIN, "MODEL_RATIO_MAX": MODEL_RATIO_MAX, "MODEL_UNIT_FRAC": MODEL_UNIT_FRAC,
          "VR_RANGE": VR_RANGE, "NULL_DRAWS": NULL_DRAWS, "CORAL_FIX_R": CORAL_FIX_R, "CT_TAUS_GRID": CT_TAUS_GRID, "TOL_LOGP": TOL_LOGP,
          "TOL_UAR": TOL_UAR, "P4_KS": P4_KS, "P5_KS": P5_KS, "GROUP_B": GROUP_B, "SEEN": {k: sorted(v) for k, v in SEEN.items()}}
EMPTY = "__none__"
OWN_CHECKED: set = set()

# 入力の場所の中で，段ごとに開いてよいファイル（名前の末尾の正規表現）
ALLOW = {
    "delta": (r"_ps_.+_r\d+_none\.npz$", r"_pseudo\.csv$", r"^provenance\.json$"),
    "select": (r"_pseudo\.csv$", r"_chosen\.csv$", r"_noise\.csv$", r"_settings\.json$", r"_conv\.csv$", r"^provenance\.json$"),
}


# ---------------------------------------------------------------- 道具

def corp(g):
    return GROUP_SPEC[g][0]


def strip_seed(p):
    return re.sub(r"s\d+$", "", p)


def seed_of(method, p):
    if method not in SEEDED:
        return -1
    m = re.search(r"s(\d+)$", p)
    return int(m.group(1)) if m else -1


def role_of(run, student):
    if run not in WORLDS:
        return "kind"
    return "fused" if student == run else "uni"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def majority(n):
    return n // 2 + 1


def install_guard(stage, roots):
    """入力の場所の中で ALLOW[stage] に無いファイルを開こうとしたら PermissionError で止める．入力の場所への書き込みは全段で止める。
    段 (a)・(b) では data/（ラベルの正本と特徴）も開かない（関門 1 の L-9）。"""
    roots = [os.path.realpath(str(r)) for r in roots]
    pats = [re.compile(p) for p in ALLOW.get(stage, ())]
    read_guard = stage in ALLOW
    data_root = os.path.realpath(str(ROOT / "data"))

    def hook(event, args):
        if event != "open":
            return
        p = args[0]
        if not isinstance(p, (str, bytes, os.PathLike)):
            return
        p = os.path.realpath(os.fsdecode(p))
        if read_guard and (p == data_root or p.startswith(data_root + os.sep)):
            raise PermissionError(f"--stage {stage} は data/ を開かない：{p}")
        if not any(p == r or p.startswith(r + os.sep) for r in roots):
            return
        mode = args[1] if len(args) > 1 and isinstance(args[1], str) else "r"
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        if any(c in mode for c in "wax+") or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            raise PermissionError(f"入力の場所に書こうとした：{p}")
        if read_guard and not any(pt.search(os.path.basename(p)) for pt in pats):
            raise PermissionError(f"--stage {stage} は {p} を開かない（評価のターゲットの出力か，この段に要らないファイル）")
    sys.addaudithook(hook)


def guard_selftest(stage, dirs):
    """禁じたファイル（評価のターゲットの `<T>.csv` などがあればそれ，無ければ存在しない名前）を開けないことを確かめる。"""
    probe = None
    for d in dirs:
        for f in sorted(Path(d).glob("*.csv")):
            if not any(re.search(p, f.name) for p in ALLOW[stage]):
                probe = f
                break
        if probe:
            break
    probe = probe or Path(dirs[0]) / "__guard_probe__.csv"
    for pr in (probe, ROOT / "data" / "README.md"):                            # data/ も開けないこと（関門 1 の L-9）
        try:
            open(pr, "rb").close()
        except PermissionError:
            continue
        except FileNotFoundError:
            raise SystemExit(f"見張りが働いていない（{pr} を止めずに開こうとした）")
        raise SystemExit(f"見張りが働いていない（{pr} を開けた）")
    return str(probe)


def prepare_out(out, stage):
    d = Path(out) / stage
    if d.exists() and any(d.iterdir()):
        raise SystemExit(f"{d} は既にある．上書きしない（別の --out を指定するか，消してから流す）")
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_prov(d, stage, args, extra):
    prov = {"script": "exp/analyze_e79.py", "experiment": "E79", "stage": stage, "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "dirty_files": git("status", "--porcelain", "--untracked-files=no").splitlines(),
            "script_sha256": sha256(Path(__file__)), "started": args._t0_str, "seconds": round(time.time() - args._t0, 1),
            "consts": CONSTS, "threads": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "allow_incomplete": args.allow_incomplete, "skip_commit_check": args.skip_commit_check, "allow_dirty": args.allow_dirty, **extra}
    (d / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1, default=str))


def check_dirty(args):
    """exp/ か measure/ に未コミットの変更があれば止める（コードの版を結果から特定できるように．関門 1 の C-7）。"""
    bad = [ln for ln in git("status", "--porcelain", "--untracked-files=no").splitlines() if re.search(r"\s(exp|measure)/", ln)]
    if bad and not args.allow_dirty:
        raise SystemExit(f"exp/ か measure/ に未コミットの変更がある：{bad}．コミットしてから流す（試しは --allow-dirty）")
    return bad


def read_prov(stage, args):
    f = Path(args.out) / stage / "provenance.json"
    if not f.exists():
        raise SystemExit(f"{f} が無い．前の段を先に流す")
    return json.loads(f.read_text())


def check_chain(stage, args):
    """段の間の照合（関門 1 の L-2・L-3・I12-2）．外れたら止める．--skip-commit-check はコミットの確認だけを飛ばす。"""
    me = sha256(Path(__file__))
    out = {}
    dpath, cpath, fpath = (Path(args.out) / "delta" / "delta.csv", Path(args.out) / "select" / "chosen.csv", Path(args.out) / "select" / "fixed.csv")
    out["delta"] = require_committed(dpath, "δ", args)
    prev = ["delta"] + (["select"] if stage in ("measure", "judge") else []) + (["measure"] if stage == "judge" else [])
    for st in prev:
        p = read_prov(st, args)
        if p.get("script_sha256") != me:
            raise SystemExit(f"{st} の段の台本の版（{p.get('script_sha256', '')[:16]}）が今の台本（{me[:16]}）と違う．同じ版で流し直す")
        if not args.skip_commit_check and (p.get("skip_commit_check") or (p.get("delta") or {}).get("skipped")):
            raise SystemExit(f"{st} の段はコミットの確認を飛ばして流されている．確認してから流し直す")
    if stage in ("measure", "judge"):
        out["chosen"] = require_committed(cpath, "選んだ設定（chosen.csv）", args)
        out["fixed"] = require_committed(fpath, "固定版の行（fixed.csv）", args)
        ps = read_prov("select", args)
        if (ps.get("delta") or {}).get("sha16") != out["delta"]["sha16"]:
            raise SystemExit("select の段が使った δ と今の δ が違う")
        if ps.get("chosen_csv_sha256") != sha256(cpath) or ps.get("fixed_csv_sha256") != sha256(fpath):
            raise SystemExit("select の段が書いた chosen.csv・fixed.csv と今のファイルが違う")
    if stage == "judge":
        pm = read_prov("measure", args)
        if (pm.get("delta") or {}).get("sha16") != out["delta"]["sha16"]:
            raise SystemExit("measure の段が使った δ と今の δ が違う")
        if pm.get("select_chosen_sha256") != sha256(cpath) or pm.get("select_fixed_sha256") != sha256(fpath):
            raise SystemExit("measure の段が使った chosen.csv・fixed.csv と今のファイルが違う")
    out["script_sha256"] = me
    return out


def require_committed(path, what, args):
    """`path` の sha256 の先頭 16 文字が，HEAD の docs/ か DAI_paper/ のどれかの追跡しているファイルに書かれているかを確かめる。"""
    if not Path(path).exists():
        raise SystemExit(f"{path} が無い．前の段を先に流す")
    h = sha256(path)[:16]
    hit = git("grep", "-l", "-F", h, "HEAD", "--", "docs", "DAI_paper")
    if not hit:
        if args.skip_commit_check:
            return {"sha16": h, "committed": False, "skipped": True}
        raise SystemExit(f"{what}（{path}，sha256 の先頭 {h}）がコミットされた文書に無い．値を docs/ に書いてコミットしてから流す")
    return {"sha16": h, "committed": True, "files": hit.splitlines()}


def parse_dirs(args):
    """名前 → (E77 の場所, E78 の場所)．指定が無ければ results/e77_<名前>・results/e78_<名前>。"""
    e77 = dict(x.split("=", 1) for x in (args.e77 or []))
    e78 = dict(x.split("=", 1) for x in (args.e78 or []))
    out = {}
    for r in args.runs:
        a = Path(e77.get(r, ROOT / "results" / f"e77_{r}"))
        b = Path(e78.get(r, ROOT / "results" / f"e78_{r}"))
        out[r] = (a, b if b.exists() else None)
    return out


def check_run_name(run, d, wave, issues):
    f = Path(d) / "provenance.json"
    if not f.exists():
        issues.append({"check": "provenance", "run": run, "wave": wave, "note": "provenance.json が無い"})
        return None
    p = json.loads(f.read_text())
    argv = p.get("argv", [])
    nm = argv[argv.index("--kind") + 1] if "--kind" in argv else (argv[argv.index("--world") + 1] if "--world" in argv else None)
    if nm != run:
        raise SystemExit(f"{d} の provenance の生徒の名前 {nm} が {run} と違う")
    return p.get("commit")


def fail_or_note(args, issues, **row):
    issues.append(row)
    if not args.allow_incomplete:
        raise SystemExit(f"欠けか食い違い：{row}")


def read_rows(path):
    d = pd.read_csv(path, dtype={"param": str, "meff": str, "K": str, "student": str, "method": str, "pseudo": str, "target": str})
    if d.empty or "method" not in d.columns:
        return pd.DataFrame(columns=["pseudo", "target", "rep", "row_id", "student", "method", "param", "meff", "K", "uar", "commit"])
    return d


def present_uar(y, pred):
    """現れた感情だけで平均した UAR（全部の感情があれば `shift.uar` と同じ）。"""
    ks = np.unique(y)
    return float(np.mean([(pred[y == k] == k).mean() for k in ks]))


# ---------------------------------------------------------------- 段 (a)：δ

def stage_delta(args, dirs, out):
    issues, units, inputs = [], [], {}
    for run, (d77, _) in dirs.items():
        check_run_name(run, d77, "e77", issues)
        files = sorted(Path(d77).glob("*_ps_*_none.npz"))
        targets = [t for t in (args.targets or groups_for(run)) if t in groups_for(run)]
        for T in targets:
            S = [s for s in groups_for(run) if corp(s) != corp(T)]
            expect = [(c, r) for c in S if any(corp(s2) != corp(c) for s2 in S) for r in args.pseudo_reps]
            have = {}
            for f in files:
                m = re.match(r"^(?P<T>.+)_ps_(?P<c>.+)_r(?P<r>\d+)_none\.npz$", f.name)
                if m and m["T"] == T:
                    have[(m["c"], int(m["r"]))] = f
            for c, r in expect:
                if (c, r) not in have:
                    fail_or_note(args, issues, check="ps_missing", run=run, target=T, pseudo=c, rep=r)
            pfile = Path(d77) / f"{T}_pseudo.csv"
            ps = read_rows(pfile) if pfile.exists() else pd.DataFrame()
            if pfile.exists():
                inputs[str(pfile)] = sha256(pfile)
            ps_none = ps[ps.method == "none"].set_index(["pseudo", "rep", "student", "K"]).uar if len(ps) else pd.Series(dtype=float)
            if ps_none.index.duplicated().any():
                raise SystemExit(f"{pfile}：none の行が重複している")
            for (c, r), f in sorted(have.items()):
                inputs[str(f)] = sha256(f)
                z = np.load(f)
                y = np.asarray(z["y"]).astype(int)
                for i, key in enumerate(z["keys"]):
                    st, K = str(key).split("|")
                    lp = np.asarray(z[f"lp{i}"], dtype=np.float64)
                    if not np.isfinite(lp).all():
                        raise SystemExit(f"NaN/Inf：{f} {key}")
                    pred = lp.argmax(1)
                    u = shift_uar(y, pred)
                    a = float(np.mean([ba(dd, 0.0) for dd, _ in d_o(lp, y)]))
                    ref = ps_none.get((c, r, st, K), np.nan) if len(ps_none) else np.nan
                    if np.isfinite(ref) and abs(ref - u) > 1e-9:
                        fail_or_note(args, issues, check="ps_uar_vs_pseudo_csv", run=run, target=T, pseudo=c, rep=r, student=st, K=K, absdiff=abs(ref - u))
                    units.append({"run": run, "student": st, "role": role_of(run, st), "target": T, "pseudo": c, "rep": r, "K": K,
                                  "uar": 100 * u, "a_d0": 100 * a, "uar_csv_found": bool(np.isfinite(ref))})
    U = pd.DataFrame(units)
    if U.empty:
        raise SystemExit("δ の材料（_ps_ のファイル）が 1 つも無い")
    U.to_csv(out / "delta_reps.csv", index=False)
    g = U.groupby(["run", "student", "role", "target", "pseudo", "K"])
    pu = g.agg(n_reps=("rep", "nunique"), sd_u=("uar", lambda x: x.std(ddof=1)), sd_a=("a_d0", lambda x: x.std(ddof=1))).reset_index()
    pu["s_u"], pu["s_d"] = pu.sd_u / np.sqrt(10), pu.sd_a / np.sqrt(10)
    pu.to_csv(out / "delta_units.csv", index=False)
    full = pu[pu.n_reps >= 2]
    D = full.groupby(["run", "student", "role", "K"]).agg(n_units=("s_u", "size"), n_targets=("target", "nunique"),
                                                          s_u=("s_u", "median"), s_d=("s_d", "median")).reset_index()
    D["delta_u"], D["delta_d"] = 2 * D.s_u, 2 * D.s_d
    # 感度の列（関門 1 の S7）：仮のターゲットのコーパスごとに中央値を取ってから，コーパスの間で中央値を取った δ．判定には使わない
    if len(full):
        full = full.assign(pcorp=full.pseudo.map(corp))
        pc = full.groupby(["run", "student", "K", "pcorp"])[["s_u", "s_d"]].median().groupby(["run", "student", "K"]).median()
        cn = full.groupby(["run", "student", "K", "pcorp"]).size().reset_index(name="n")
        cnt = pd.DataFrame([{"run": a, "student": b, "K": c, "pairs_by_corpus": ";".join(f"{x}:{y}" for x, y in zip(g.pcorp, g.n))}
                            for (a, b, c), g in cn.groupby(["run", "student", "K"])])
        D = D.merge((2 * pc).rename(columns={"s_u": "delta_u_corpus", "s_d": "delta_d_corpus"}).reset_index(), how="left").merge(cnt, how="left")
    else:
        D = D.assign(delta_u_corpus=np.nan, delta_d_corpus=np.nan, pairs_by_corpus="")
    D["K"] = D.K.astype(str)
    D = D.assign(_k=D.K.map(KS.index)).sort_values(["run", "student", "_k"]).drop(columns="_k")
    D.to_csv(out / "delta.csv", index=False)
    pd.DataFrame(issues).to_csv(out / "checks.csv", index=False)
    h = sha256(out / "delta.csv")
    write_prov(out, "delta", args, {"inputs_sha256": inputs, "n_issues": len(issues), "delta_csv_sha256": h,
                                    "rule": "s = 反復 0〜4 の標準偏差（ddof=1）/ √10．δ = 2 × (run, student, K) ごとの (評価のターゲット, 仮のターゲット) の中央値．単位はポイント．"
                                            "delta_*_corpus は仮のターゲットのコーパスごとの中央値のコーパスの間の中央値（感度の列）"})
    print(f"δ：{len(D)} 行（run × 生徒 × K），材料 {len(U)} 行，欠けと食い違い {len(issues)} 件", flush=True)
    print(f"delta.csv の sha256 の先頭 16 文字：{h[:16]}（これを文書に書いてコミットしてから --stage select に進む）", flush=True)


# ---------------------------------------------------------------- 段 (b)：選択

def sel_defs(role):
    """選ぶ族 → [(波, 判定の関数(method, pnorm, meff) → bool)]．ref_*・teacher・*_g の行はどの族にも入らない。"""
    uni, world = role == "uni", role != "kind"
    refs_kd = ("pF", "pFK", "ref") if uni else ("poe0", "ref")
    refs_ox = ("pFK", "ref") if uni else ("ref",)
    ours_ref = "pF" if uni else "poe0"

    def dist(refs, wgts=WGTS, meffs=MEFFS):
        return ("e77", lambda m, p, f: m == "dist" and p.split(":")[0] in refs and p.split(":")[1] in wgts and f in meffs)
    D = {
        "entmin": [("e77", lambda m, p, f: m == "entmin")],
        "self": [("e77", lambda m, p, f: m == "self")],
        "im": [("e77", lambda m, p, f: m == "im" and f in MEFFS)],
        "t3a": [("e77", lambda m, p, f: m == "t3a")],
        "t3a_b": [("e77", lambda m, p, f: m == "t3a_b")],
        "lp": [("e77", lambda m, p, f: m == "lp")],
        "coral": [("e78", lambda m, p, f: m == "coral")],                      # E78 が単位ごとに足した E39 の C′・C_b の点も候補に入れる（関門 1 の C-1）
        "coralctl": [("e78", lambda m, p, f: m == "coralctl")],
    }
    if role != "fused":                                                       # 融合の生徒では DANN・MME を回さない（E78）
        for m_ in ("dann", "mme", "mapctl"):
            D[m_] = [("e78", (lambda mm: lambda m, p, f: m == mm)(m_))]
    D["groupB"] = [x for k in GROUP_B if k in D for x in D[k]]
    if world:
        D["kd"] = [("e77", lambda m, p, f: m in ("kd", "kdh") and re.sub(r"T\d+$", "", p) in refs_kd)]
        D["oursx"] = [dist(refs_ox)]
        D["oursx_small"] = [dist(refs_ox, ("bal", "raw", "hard0.7"), ("500", "n"))]
        D["ours"] = [("e77", lambda m, p, f: m == "dist" and p == f"{ours_ref}:bal")]
        if uni:
            D["abl_refa"], D["abl_refb"], D["abl_pF"] = [dist(("pFK",))], [dist(("ref",))], [dist(("pF",))]
        else:
            D["abl_ref0"], D["abl_poe0"] = [dist(("ref0",))], [dist(("poe0",))]
        D["abl_bal"], D["abl_raw"] = [dist(refs_ox, ("bal",))], [dist(refs_ox, ("raw",))]
        D["abl_hard"] = [dist(refs_ox, ("hard0.5", "hard0.7", "hard0.9"))]
        D["abl_meff500"] = [dist(refs_ox, WGTS, ("500",))]
        D["abl_own"] = [("e77", lambda m, p, f: m == "own")]
        D["abl_ownK"] = [dist(("ownK",))]
        D["cot"] = [("e78", lambda m, p, f: m == "cot" and p.split(":")[0] in refs_kd)]
        D["cotctl"] = [("e78", lambda m, p, f: m == "cotctl" and p.split(":")[0] in refs_kd)]
        if uni:
            D["xmuda"] = [("e78", lambda m, p, f: m == "xmuda")]
            D["cotrain"] = [("e77", lambda m, p, f: m == "cotrain")]
            D["coreg"] = [("e77", lambda m, p, f: m == "coreg")]
        else:
            D["cotrain"] = [("e77", lambda m, p, f: m == "cotrain_fused")]
            D["cotrain_prod"] = [("e77", lambda m, p, f: m == "cotrain_prod")]
            D["coreg"] = [("e77", lambda m, p, f: m == "coreg_sum")]
    return D


def cand_mask(df, defs):
    m = np.zeros(len(df), bool)
    for wave, fn in defs:
        w = (df.wave == wave).to_numpy()
        if w.any():
            sub = df[w]
            m[w] |= np.array([bool(fn(a, b, c)) for a, b, c in zip(sub.method, sub.pnorm, sub.meff)], bool)
    return m


def choose(cs, none_score):
    """cs：候補ごとの (wave, method, pnorm, meff, score)．同点は 'method|pnorm|meff' の文字列の順で先頭（E78 の choose_seeded と同じ）。
    none を含む版は，最良の候補の値が none の値より大きいときだけ候補を選ぶ（同点は none）。"""
    if cs.empty:
        return None, None, 0
    cs = cs.assign(_k=cs.method + "|" + cs.pnorm + "|" + cs.meff).sort_values("_k")
    best = cs.score.max()
    tie = int((cs.score == best).sum())
    ex = cs[cs.score == best].iloc[0]
    inc = ex if best > none_score else None
    return ex, inc, tie


def load_pseudo(run, d77, d78):
    parts = []
    for wave, d in (("e77", d77), ("e78", d78)):
        if d is None:
            continue
        for f in sorted(Path(d).glob("*_pseudo.csv")):
            T = f.name[: -len("_pseudo.csv")]
            x = read_rows(f)
            if x.empty:
                continue
            parts.append(x.assign(wave=wave, eval_target=T))
    if not parts:
        return pd.DataFrame()
    P = pd.concat(parts, ignore_index=True)
    P["pnorm"] = [strip_seed(p) if m in SEEDED else p for m, p in zip(P.method, P.param)]
    P["seed"] = [seed_of(m, p) for m, p in zip(P.method, P.param)]
    return P


def coral_pseudo_units(d78, T, student):
    """仮のターゲットの（仮のターゲット, 反復）のうち，CORAL の r ごとの行が作られた単位（`_conv.csv` の what の列だけを読む）。"""
    f = Path(d78) / f"{T}_conv.csv" if d78 else None
    if f is None or not f.exists():
        return None
    c = pd.read_csv(f, usecols=lambda c_: c_ in ("pseudo", "rep", "student", "what"), dtype=str)
    if "pseudo" not in c.columns:
        return {r: set() for r in (32, 64, 128)}
    c = c[c.pseudo.notna() & (c.student == student)]
    return {r: {(p_, int(x)) for p_, x in zip(c[c.what == f"coral_r{r}"].pseudo, c[c.what == f"coral_r{r}"].rep)} for r in (32, 64, 128)}


def coral_eval_ok(d78, T, student, reps):
    """評価のターゲットの全反復で coral の r ごとの行が作られたか（`_conv.csv` の what の列だけを読む．値は読まない）。"""
    f = Path(d78) / f"{T}_conv.csv" if d78 else None
    if f is None or not f.exists():
        return None
    c = pd.read_csv(f, usecols=lambda c_: c_ in ("target", "rep", "student", "what"), dtype=str)
    if "target" not in c.columns:
        return None
    c = c[c.target.notna() & (c.student == student)]
    out = {}
    for r in (32, 64, 128):
        out[r] = set(reps) <= {int(x) for x in c[c.what == f"coral_r{r}"].rep}
    return out


def expected_pseudo(run, T, pseudo_reps):
    """delta と同じ期待：T と別のコーパスのソースのうち，仮のターゲットになれる群 × 反復。"""
    S = [s_ for s_ in groups_for(run) if corp(s_) != corp(T)]
    return {(c, r) for c in S if any(corp(s2) != corp(c) for s2 in S) for r in pseudo_reps}


def stage_select(args, dirs, out):
    chain = check_chain("select", args)
    dmeta = chain["delta"]
    issues, rows, cand_rows, fixed, inputs = [], [], [], [], {}
    for run, (d77, d78) in dirs.items():
        check_run_name(run, d77, "e77", issues)
        if d78 is not None:
            check_run_name(run, d78, "e78", issues)
        else:
            fail_or_note(args, issues, check="e78_missing", run=run)
        P = load_pseudo(run, d77, d78)
        targets = [t for t in (args.targets or groups_for(run)) if t in groups_for(run)]
        for T in targets:
            for d_ in (d77, d78):
                for nm in (f"{T}_pseudo.csv", f"{T}_chosen.csv", f"{T}_noise.csv", f"{T}_settings.json"):
                    if d_ is not None and (Path(d_) / nm).exists():
                        inputs[str(Path(d_) / nm)] = sha256(Path(d_) / nm)
            PT = P[P.eval_target == T] if len(P) else P
            if PT.empty:
                fail_or_note(args, issues, check="pseudo_missing", run=run, target=T)
                continue
            # 仮のターゲットと反復の組が delta と同じ期待にそろうか（関門 1 の I12-1）
            want = expected_pseudo(run, T, args.pseudo_reps)
            for w in sorted(set(PT.wave)):
                have = set(map(tuple, PT[PT.wave == w][["pseudo", "rep"]].drop_duplicates().itertuples(index=False)))
                if have != want:
                    fail_or_note(args, issues, check="pseudo_units_mismatch", run=run, target=T, wave=w,
                                 missing=sorted(want - have), extra=sorted(have - want))
            # E77 と E78 の none（仮のターゲット）が同じ値か（I14．許容幅を超えたら止める）
            nn = PT[PT.method == "none"].pivot_table(index=["pseudo", "rep", "student", "K"], columns="wave", values="uar")
            if {"e77", "e78"} <= set(nn.columns):
                dmax = float((nn.e77 - nn.e78).abs().max())
                issues.append({"check": "none_e77_vs_e78_pseudo", "run": run, "target": T, "absdiff_max": dmax})
                if not dmax <= TOL_UAR:
                    fail_or_note(args, issues, check="none_e77_vs_e78_pseudo_over_tol", run=run, target=T, absdiff_max=dmax)
            sets = json.loads((Path(d77) / f"{T}_settings.json").read_text()) if (Path(d77) / f"{T}_settings.json").exists() else {}
            chosen78 = None
            if d78 is not None and (Path(d78) / f"{T}_chosen.csv").exists():
                chosen78 = pd.read_csv(Path(d78) / f"{T}_chosen.csv", dtype=str)
            tau = {}                                                          # 固定版の τ：仮のターゲットの候補で測った参照の誤りの割合の中央値（生徒 × 参照 × K．E78 の tau_fix_of と同じ）
            if d78 is not None and (Path(d78) / f"{T}_noise.csv").exists():
                nz = pd.read_csv(Path(d78) / f"{T}_noise.csv", dtype={"K": str})
                if len(nz):
                    tau = nz.groupby(["student", "teacher", "K"]).err.median().to_dict()
            variants = [("incl", None), ("excl", None)]
            if T in ("MOSEI", "MELD"):
                variants += [("ns_incl", "IEMOCAP-script"), ("ns_excl", "IEMOCAP-script")]
            for st in sorted(PT.student.unique()):
                role = role_of(run, st)
                defs = sel_defs(role)
                Ps = PT[PT.student == st]
                ev = coral_eval_ok(d78, T, st, args.reps)                     # CORAL の r が評価のターゲットの全反復で作られたか
                pu = coral_pseudo_units(d78, T, st)                           # CORAL の r が仮のターゲットのどの単位で作られたか
                for K in KS:
                    PK = Ps[Ps.K == K]
                    for vname, drop in variants:
                        PV = PK[PK.pseudo != drop] if drop else PK
                        none = PV[(PV.method == "none") & (PV.wave == "e77")]
                        n_units = none.groupby(["pseudo", "rep"]).ngroups
                        none_score = float(none.uar.mean()) if len(none) else np.nan
                        for sel, dd in defs.items():
                            m = cand_mask(PV, dd)
                            C = PV[m]
                            if C.empty:
                                rows.append({"run": run, "target": T, "student": st, "role": role, "K": K, "selector": sel, "variant": vname,
                                             "wave": None, "method": EMPTY, "param": None, "meff": None, "score": np.nan,
                                             "none_score": none_score, "n_cand": 0, "n_dropped": 0, "n_tie": 0, "chose_none": np.nan})
                                if not (sel == "lp" and K == "0"):                  # 定義から無いのはラベル伝播の K = 0 だけ（I12-1）
                                    fail_or_note(args, issues, check="selector_without_candidates", run=run, target=T, student=st, K=K,
                                                 selector=sel, variant=vname)
                                continue
                            kk = ["wave", "method", "pnorm", "meff"]
                            sc = C.groupby(kk).uar.agg(score="mean", n="size").reset_index()
                            nu = C.drop_duplicates(kk + ["pseudo", "rep"]).groupby(kk).size().rename("nu").reset_index()
                            sc = sc.merge(nu)
                            mult = sc.method.isin(["cot", "cotctl"]).map({True: len(SEEDS), False: 1})
                            ok = (sc.nu == n_units) & (sc.n == n_units * mult)
                            # 仮のターゲットの行が揃わない候補（I12-1）．ただし CORAL の候補で，欠けた（仮のターゲット, 反復）が E78 の `_conv.csv` で
                            # その r の行が作られなかった単位（5r の条件で外れた単位）とちょうど一致するものは，設計どおり外すだけで止めない（関門 1 の再検査の I12-3）
                            units_all = set(map(tuple, none[["pseudo", "rep"]].drop_duplicates().itertuples(index=False)))
                            excused = np.zeros(len(sc), bool)
                            for i in np.flatnonzero(~ok.to_numpy()):
                                rr = sc.iloc[i]
                                m_r = re.match(r"^r(\d+)rho", rr.pnorm) if rr.method == "coral" else None
                                if m_r is None or pu is None:
                                    continue
                                have = set(map(tuple, C[(C.wave == rr.wave) & (C.method == rr.method) & (C.pnorm == rr.pnorm) & (C.meff == rr.meff)][
                                    ["pseudo", "rep"]].drop_duplicates().itertuples(index=False)))
                                want_missing = units_all - pu.get(int(m_r.group(1)), set())
                                if want_missing and (units_all - have) == want_missing and rr.n == rr.nu:
                                    excused[i] = True
                            n_incomplete = int((~ok.to_numpy() & ~excused).sum())
                            if n_incomplete:
                                fail_or_note(args, issues, check="incomplete_candidates", run=run, target=T, student=st, K=K, selector=sel,
                                             variant=vname, n=n_incomplete)
                            if excused.any():
                                issues.append({"check": "coral_r_pseudo_excluded", "run": run, "target": T, "student": st, "K": K, "selector": sel,
                                               "variant": vname, "n": int(excused.sum())})
                            # CORAL の r が 5r の条件で外れた単位：全部の仮のターゲットと評価のターゲットに値のある候補だけ．CORAL を含む全部の族に当てる（I4-1）
                            is_coral = (sc.method == "coral").to_numpy()
                            if is_coral.any():
                                for r_ in (32, 64, 128):
                                    in_r = is_coral & sc.pnorm.str.startswith(f"r{r_}rho").to_numpy()
                                    if not in_r.any():
                                        continue
                                    if ev is not None and not ev[r_]:
                                        ok &= ~in_r
                                    elif ev is None and not args.allow_incomplete:
                                        raise SystemExit(f"{run} {T}：E78 の _conv.csv が無く，CORAL の r の有無を確かめられない")
                            dropped = int((~ok).sum())
                            sc = sc[ok]
                            ex, inc, tie = choose(sc, none_score)
                            pick = ex if vname.endswith("excl") else inc
                            base = {"run": run, "target": T, "student": st, "role": role, "K": K, "selector": sel, "variant": vname,
                                    "none_score": none_score, "n_cand": int(len(sc)), "n_dropped": dropped, "n_tie": tie}
                            if pick is None and ex is None:
                                rows.append({**base, "wave": None, "method": EMPTY, "param": None, "meff": None, "score": np.nan, "chose_none": np.nan})
                            elif pick is None:
                                rows.append({**base, "wave": "e77", "method": "none", "param": "-", "meff": "0", "score": none_score, "chose_none": True})
                            else:
                                rows.append({**base, "wave": pick.wave, "method": pick.method, "param": pick.pnorm, "meff": pick.meff,
                                             "score": float(pick.score), "chose_none": False if vname.endswith("incl") else np.nan})
                            if vname == "incl":
                                cand_rows.append({"run": run, "target": T, "student": st, "role": role, "K": K, "selector": sel,
                                                  "n_cand_with_none": int(len(sc)) + 1, "n_dropped": dropped})
                            # E78 の choose_seeded と同じ選択か（I14）
                            if vname == "excl" and sel in ("dann", "mme", "mapctl", "xmuda") and chosen78 is not None and ex is not None:
                                c78 = chosen78[(chosen78.student == st) & (chosen78.method == sel) & (chosen78.K == K)]
                                same = len(c78) == 1 and c78.param.iloc[0] == ex.pnorm
                                issues.append({"check": "choose_seeded_match", "run": run, "target": T, "student": st, "K": K, "method": sel, "ok": same})
                                if not same and not args.allow_incomplete:
                                    raise SystemExit(f"E78 の _chosen.csv と選択が違う：{run} {T} {st} {sel} K={K}")
                    # 固定版の行（設計の文書の「固定」．選ばない）
                    fixed += fixed_rows(run, T, st, role, K, sets, rows, PK, tau)
    R = pd.DataFrame(rows)
    R.to_csv(out / "chosen.csv", index=False)
    pd.DataFrame(cand_rows).to_csv(out / "candidates.csv", index=False)
    pd.DataFrame(fixed).to_csv(out / "fixed.csv", index=False)
    pd.DataFrame(issues).to_csv(out / "checks.csv", index=False)
    hc, hf = sha256(out / "chosen.csv"), sha256(out / "fixed.csv")
    write_prov(out, "select", args, {"delta": dmeta, "chain": chain, "n_issues": len(issues), "inputs_sha256": inputs,
                                     "rule": "仮のターゲット × 反復（× 種）の全行の UAR の単純平均の最大．同点は 'method|pnorm|meff' の文字列の順で先頭．none を含む版は none と同点なら none",
                                     "chosen_csv_sha256": hc, "fixed_csv_sha256": hf})
    print(f"選択：{len(R)} 行（うち候補なし {int((R.method == EMPTY).sum())}），固定版 {len(fixed)} 行，記録 {len(issues)} 件", flush=True)
    print(f"chosen.csv の sha256 の先頭 16 文字：{hc[:16]}，fixed.csv：{hf[:16]}（δ と同じく文書に書いてコミットしてから --stage measure に進む）", flush=True)


def fixed_rows(run, T, st, role, K, sets, rows, PK=None, tau=None):
    """固定版（文献の既定値）と，対照の対応する行．値は見ずに決まる（設定の文字列だけで作る）。"""
    out = []
    base = {"run": run, "target": T, "student": st, "role": role, "K": K}
    out.append({**base, "fixname": "massbal", "wave": "e77", "method": "massbal", "param": "-", "meff": "0"})
    out.append({**base, "fixname": "saerens", "wave": "e77", "method": "saerens", "param": "-", "meff": "0"})
    out.append({**base, "fixname": "t3a_fix", "wave": "e77", "method": "t3a", "param": "Mall", "meff": "0"})
    if K != "0":
        out.append({**base, "fixname": "im_fix", "wave": "e77", "method": "im", "param": "a0.1", "meff": "tim"})
        out.append({**base, "fixname": "lp_fix", "wave": "e77", "method": "lp", "param": "k50a0.99hard", "meff": "n"})
    if role != "fused":
        out.append({**base, "fixname": "dann_fix", "wave": "e78", "method": "dann", "param": "lr0.001it1000lam0.1mu1.0", "meff": "-", "seedmode": "s0",
                    "note": "原論文の既定値に近い格子の点（λ = 0.1．λ は候補の数 n を掛けた尺度）．種 0"})
        out.append({**base, "fixname": "mme_fix", "wave": "e78", "method": "mme", "param": "lr0.001it1000lam0.1mu1.0sw0.0", "meff": "-", "seedmode": "s0",
                    "note": "原論文の既定値に近い格子の点（λ = 0.1，ソースのラベルを使わない sw = 0）．種 0"})
        out.append({**base, "fixname": "mapctl_fix", "wave": "e78", "method": "mapctl", "param": "lr0.001it1000mu1.0", "meff": "-", "seedmode": "s0"})
        for sel in ("dann", "mme"):                                             # 選んだ DANN・MME と同じ lr・反復・μ の対照（種 0）
            ch = [r for r in rows if r["run"] == run and r["target"] == T and r["student"] == st and r["K"] == K and r["selector"] == sel
                  and r["variant"] == "excl" and r["method"] == sel]
            if ch:
                m = re.match(r"lr(?P<lr>[^i]+)it(?P<it>\d+)lam[^m]+mu(?P<mu>[0-9.]+)", ch[0]["param"])
                if m:
                    out.append({**base, "fixname": f"mapctl_match_{sel}", "wave": "e78", "method": "mapctl",
                                "param": f"lr{m['lr']}it{m['it']}mu{m['mu']}", "meff": "-", "seedmode": "s0"})
    e39 = (sets.get("E39") or {}).get(st)
    if e39:
        cbs = "free" if e39.get("cb") is None else str(float(e39["cb"]))
        p = f"r{CORAL_FIX_R}rho{float(e39['rho_m'])}cp{float(e39['cp'])}cb{cbs}"
        out.append({**base, "fixname": "coral_fix", "wave": "e78", "method": "coral", "param": p, "meff": "-",
                    "note": "r = 64 と E39 の (ρ_M, C′, C_b)．E78 の 957fa35 が E39 の C′・C_b を格子に足した"})
    if role != "kind":
        refs = ("pF", "pFK", "ref") if role == "uni" else ("poe0", "ref")
        # Co-teaching の固定版の参照（E78 の節の規則）：仮のターゲットの cot の行のうち，学習率 1e−3，m_eff ＝ n，τ が固定版の τ に最も近い格子の点の値で選ぶ
        best = None
        for ref in refs:
            tf = (tau or {}).get((st, ref, K))
            if tf is None or PK is None or PK.empty:
                continue
            tg = min(CT_TAUS_GRID, key=lambda t: (abs(t - tf), t))           # 同じ近さなら小さい τ
            x = PK[(PK.wave == "e78") & (PK.method == "cot") & (PK.pnorm == f"{ref}:tau{tg}lr0.001") & (PK.meff == "n")]
            if len(x) and (best is None or x.uar.mean() > best[1] or (x.uar.mean() == best[1] and ref < best[0])):
                best = (ref, float(x.uar.mean()), tf, tg)
        if best is not None:
            out.append({**base, "fixname": "cotfix", "wave": "e78", "method": "cotfix", "param": f"{best[0]}:*", "meff": "n",
                        "note": f"参照 {best[0]}（固定版の τ {best[2]:.4f} に最も近い格子の点 τ = {best[3]} の仮のターゲットの値で選んだ）"})
        ch = [r for r in rows if r["run"] == run and r["target"] == T and r["student"] == st and r["K"] == K and r["selector"] == "cot"
              and r["variant"] == "excl" and r["method"] == "cot"]
        name0 = refs[0]                                                        # cotzero の行の参照の名前（E78 は最初の参照の名前で 1 回だけ回す）
        for lr in ("0.001", "0.01"):
            out.append({**base, "fixname": f"cotzero_lr{lr}", "wave": "e78", "method": "cotzero", "param": f"{name0}:tau0.0lr{lr}", "meff": "0"})
        if ch:
            out.append({**base, "fixname": "cot_g_match", "wave": "e78", "method": "cot_g", "param": ch[0]["param"], "meff": ch[0]["meff"]})
            lr = ch[0]["param"].split("lr")[-1]
            out.append({**base, "fixname": "cotzero_match", "wave": "e78", "method": "cotzero", "param": f"{name0}:tau0.0lr{lr}", "meff": "0",
                        "note": "選んだ Co-teaching と同じ学習率で，候補の項の重みを 0 にした行（確率的な勾配法の届かなさ）"})
        cc = [r for r in rows if r["run"] == run and r["target"] == T and r["student"] == st and r["K"] == K and r["selector"] == "cotctl"
              and r["variant"] == "excl" and r["method"] == "cotctl"]
        if cc:                                                                 # 対照（τ ＝ 0）と同じ学習率の cotzero（E78 の再検査 I13-2）
            lr = cc[0]["param"].split("lr")[-1]
            out.append({**base, "fixname": "cotzero_match_ctl", "wave": "e78", "method": "cotzero", "param": f"{name0}:tau0.0lr{lr}", "meff": "0",
                        "note": "選んだ Co-teaching の対照と同じ学習率の cotzero"})
    return out


# ---------------------------------------------------------------- 段 (c)：測る

def pair_moments(L, y, full=True):
    """L：(R, n, 4) の対数の確率（float64）．6 対の d・o と，full なら 2 感情の実際の正解率・分散の比・z の AUC を返す。"""
    R = L.shape[0]
    d, o, acc, vr, auc = (np.full((R, 6), np.nan) for _ in range(5))
    for p, (k, j) in enumerate(PAIRS):
        mk, mj = y == k, y == j
        if mk.sum() < 2 or mj.sum() < 2:
            continue
        zk = L[:, mk, k] - L[:, mk, j]
        zj = L[:, mj, k] - L[:, mj, j]
        ak, aj = zk.mean(1), zj.mean(1)
        vk, vj = zk.var(1, ddof=1), zj.var(1, ddof=1)
        sd = np.sqrt((vk + vj) / 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            d[:, p] = (ak - aj) / sd
            o[:, p] = -(ak + aj) / 2 / sd
            vr[:, p] = vk / vj
        if not full:
            continue
        acc[:, p] = 0.5 * ((zk > 0).mean(1) + (zj < 0).mean(1))
        both = np.concatenate([zk, zj], 1)
        rk = stats.rankdata(both, axis=1)[:, : zk.shape[1]]
        n1, n2 = zk.shape[1], zj.shape[1]
        auc[:, p] = (rk.sum(1) - n1 * (n1 + 1) / 2) / (n1 * n2)
    return d, o, acc, vr, auc


def lda_sep(Xa, ya, Xb, yb):
    """`explore_pairs.own_separation` の中の向き（LDA．Ledoit–Wolf）を Xa・ya で作り，Xb・yb で 6 対の離れ方を測る（同じ式）。"""
    m = np.stack([Xa[ya == k].mean(0) for k in range(4)])
    S = LedoitWolf(assume_centered=True, store_precision=False).fit(Xa - m[ya]).covariance_   # 共分散は own_separation と同じ（逆行列を作らない）
    U = scipy.linalg.solve(S, np.stack([m[k] - m[j] for k, j in PAIRS]).T, assume_a="pos").T   # 正定値なので Cholesky で解く（同じ解）
    out = []
    for u, (k, j) in zip(U, PAIRS):
        pk, pj = Xb[yb == k] @ u, Xb[yb == j] @ u
        out.append((pk.mean() - pj.mean()) / np.sqrt((pk.var(ddof=1) + pj.var(ddof=1)) / 2))
    return np.array(out)


def student_feats(run, X):
    if run not in WORLDS:
        return {run: X}
    out = {m: X[:, a:b] for m, a, b in blocks(run)}
    out[run] = X
    return out


def uar_bias_fixed(L, y):
    """切片だけを直した UAR：向き（L の差）を変えずに，テストの感情を釣り合わせた交差エントロピーを最小にする 4 つの切片を足した UAR（4 クラスのまま）。"""
    n = len(y)
    w = 1.0 / np.bincount(y, minlength=4)[y]
    oh = np.eye(4)[y]

    def f(b3):
        b = np.r_[0.0, b3]
        Z = L + b
        lse = logsumexp(Z, 1)
        P = np.exp(Z - lse[:, None])
        return float(((lse - Z[np.arange(n), y]) * w).sum()), ((P - oh) * w[:, None]).sum(0)[1:]
    res = minimize(f, np.zeros(3), jac=True, method="L-BFGS-B", options={"maxiter": 200})
    return present_uar(y, (L + np.r_[0.0, res.x]).argmax(1)), bool(res.success)


def eval_table(d, T):
    f = Path(d) / f"{T}.csv"
    if not f.exists():
        return None
    x = read_rows(f)
    x["pnorm"] = [strip_seed(p) if m in SEEDED else p for m, p in zip(x.method, x.param)]
    x["seed"] = [seed_of(m, p) for m, p in zip(x.method, x.param)]
    return x


def key_table(sel, fixed, run, T, tab=None):
    """ブートストラップで測る鍵（生徒, K, 波, 方法, pnorm, meff, 種の扱い）．参照の行（ref_*・teacher）も鍵にする（関門 1 の S5・C-3・R10-1）。"""
    ks = set()
    if tab is not None and tab.get("e77") is not None:
        t = tab["e77"]
        for r in t[t.method.isin(REFS)][["student", "K", "method", "pnorm", "meff"]].drop_duplicates().itertuples(index=False):
            ks.add((r.student, r.K, "e77", r.method, r.pnorm, r.meff, "-"))
    s = sel[(sel.run == run) & (sel.target == T) & sel.method.notna() & (sel.method != EMPTY)]
    for r in s.itertuples():
        modes = ("agg", "s0") if r.method in SEEDED else ("-",)
        for md in modes:
            ks.add((r.student, r.K, r.wave, r.method, r.param, r.meff, md))
    f = fixed[(fixed.run == run) & (fixed.target == T)]
    for r in f.itertuples():
        md = getattr(r, "seedmode", None)
        md = md if isinstance(md, str) else ("agg" if r.method in SEEDED else "-")
        ks.add((r.student, r.K, r.wave, r.method, r.param, r.meff, md))
    for st in s.student.unique():
        for K in KS:
            ks.add((st, K, "e77", "none", "-", "0", "-"))
    return sorted(ks)


def rows_of_key(tab, key):
    st, K, wave, meth, pn, meff, md = key
    t = tab.get(wave)
    if t is None:
        return None
    if meth == "cotfix" and pn.endswith(":*"):
        m = (t.student == st) & (t.K == K) & (t.method == meth) & t.pnorm.str.startswith(pn[:-1]) & (t.meff == meff)
    else:
        m = (t.student == st) & (t.K == K) & (t.method == meth) & (t.pnorm == pn) & (t.meff == meff)
    x = t[m]
    if md == "s0":
        x = x[x.seed == 0]
    return x


def boot_draws(run, corpus, speakers, B, seed, n_reps):
    """コーパスごとの抽出．話者（反復 0〜9 のどれかでテストに出た話者の和集合）と反復を復元抽出する．IEMOCAP の即興と台本は同じ抽出を使う。"""
    rng = np.random.default_rng([seed, zlib.crc32(run.encode()), zlib.crc32(corpus.encode())])
    nS = len(speakers)
    C = rng.multinomial(nS, np.full(nS, 1 / nS), size=B).astype(np.float64)
    A = rng.multinomial(n_reps, np.full(n_reps, 1 / n_reps), size=B).astype(np.float64)
    return C, A


def boot_rep(L, y, sidx, Cb, Drows):
    """1 反復ぶん．L：(R, n, 4)．sidx：発話ごとの話者の番号（0..n_ls−1）．Cb：(B, n_ls) の話者の重み．Drows：(R, 6) の行ごとの D（生徒ごと）。
    戻り：uar・向きの損・位置の損・逆の順序の向きの損 A(D,o) − A(d,o)・逆の順序の位置の損 A(D,0) − A(D,o)（B × R．ポイント），
    空の抽出（B），感情が欠けた抽出（B）。"""
    R, n, _ = L.shape
    nls = Cb.shape[1]
    GT = sparse.csr_matrix((np.ones(n), (sidx, np.arange(n))), shape=(nls, n))   # 話者 × 発話の指示行列
    Yk = np.stack([(y == k).astype(np.float64) for k in range(4)])            # 4 × n
    N = Cb @ (GT @ Yk.T)                                                       # B × 4
    empty = Cb.sum(1) == 0
    present = N > 0
    pred = L.argmax(2)
    feats = []
    for k in range(4):
        feats.append((GT @ ((pred == k) * Yk[k]).T).T)                         # R × nls
    for k, j in PAIRS:
        z = L[:, :, k] - L[:, :, j]
        for c in (k, j):
            feats.append((GT @ (z * Yk[c]).T).T)
            feats.append((GT @ (z * z * Yk[c]).T).T)
    Fm = np.stack(feats, 2)                                                    # R × nls × 28
    M = (Cb @ Fm.transpose(1, 0, 2).reshape(nls, -1)).reshape(len(Cb), R, -1)  # B × R × 28
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = M[:, :, :4] / N[:, None, :]
        u = np.where(present[:, None, :], rec, np.nan)
        uar = np.nanmean(u, 2)
        dirs, poss, dirs_r, poss_r = [], [], [], []
        for p, (k, j) in enumerate(PAIRS):
            b0 = 4 + 4 * p
            nk, nj = N[:, k][:, None], N[:, j][:, None]
            s1k, s2k, s1j, s2j = M[:, :, b0], M[:, :, b0 + 1], M[:, :, b0 + 2], M[:, :, b0 + 3]
            mk, mj = s1k / nk, s1j / nj
            vk = (s2k - nk * mk ** 2) / (nk - 1)
            vj = (s2j - nj * mj ** 2) / (nj - 1)
            sd = np.sqrt(np.clip((vk + vj) / 2, 0, None))
            ok = (nk >= 2) & (nj >= 2) & (sd > 0)
            dd = np.where(ok, (mk - mj) / sd, np.nan)
            oo = np.where(ok, -(mk + mj) / 2 / sd, np.nan)
            a0 = ba(dd, 0.0)
            DD = Drows[:, p][None, :]
            dirs.append(ba(DD, 0.0) - a0)
            poss.append(a0 - ba(dd, oo))
            aDo = ba(DD, oo)
            dirs_r.append(aDo - ba(dd, oo))
            poss_r.append(ba(DD, 0.0) - aDo)
        dirl, posl = np.nanmean(np.stack(dirs, 2), 2), np.nanmean(np.stack(poss, 2), 2)
        dirl_r, posl_r = np.nanmean(np.stack(dirs_r, 2), 2), np.nanmean(np.stack(poss_r, 2), 2)
    for a_ in (uar, dirl, posl, dirl_r, posl_r):
        a_[empty] = np.nan
    return 100 * uar, 100 * dirl, 100 * posl, 100 * dirl_r, 100 * posl_r, empty, (~present).any(1) & ~empty


def stage_measure(args, dirs, out):
    chain = check_chain("measure", args)
    dmeta = chain["delta"]
    sel = pd.read_csv(Path(args.out) / "select" / "chosen.csv", dtype={"K": str, "param": str, "meff": str})
    fixed = pd.read_csv(Path(args.out) / "select" / "fixed.csv", dtype={"K": str, "param": str, "meff": str})
    issues, meta, inputs = [], [], {}
    for run, (d77, d78) in dirs.items():
        c77 = check_run_name(run, d77, "e77", issues)
        c78 = check_run_name(run, d78, "e78", issues) if d78 is not None else None
        want = [t for t in (args.targets or groups_for(run)) if t in groups_for(run)]
        targets = [t for t in want if (Path(d77) / f"{t}.csv").exists()]
        for t in sorted(set(want) - set(targets)):
            fail_or_note(args, issues, check="eval_missing", run=run, target=t)
        by_corpus = {}
        for T in targets:
            by_corpus.setdefault(corp(T), []).append(T)
        rows_all, pair_rows, D_rows, keys_all, diag_rows, null_rows = [], [], [], [], [], []
        args._null_rows = null_rows
        for cp, Ts in by_corpus.items():
            prep = {}
            for T in Ts:
                prep[T] = measure_target(args, run, T, d77, d78, sel, fixed, issues, rows_all, pair_rows, D_rows, diag_rows, c77, c78, inputs)
            speakers = sorted(set().union(*[p["speakers"] for p in prep.values()]))
            C, A = boot_draws(run, cp, speakers, args.boot, args.seed, len(args.reps))
            rep_col = {r: i for i, r in enumerate(args.reps)}
            sp_ix = {s: i for i, s in enumerate(speakers)}
            for T, p in prep.items():
                t0 = time.perf_counter()
                nk = len(p["keys"])
                num = {q: np.zeros((args.boot, nk)) for q in QN}
                den = np.zeros((args.boot, nk))
                n_empty, n_miss, n_draw = 0.0, 0.0, 0.0
                for r in p["reps"]:
                    L, y, spk, Drows, krow, pt = p["load"](r)
                    if L is None:
                        continue
                    gi = np.array([sp_ix[s] for s in spk])
                    ls, sidx = np.unique(gi, return_inverse=True)
                    Cb = C[:, ls]
                    # 検算：重みを全部 1 にすると，全テストの UAR と向きの損（点の値）に戻る
                    u1, d1, *_ = boot_rep(L[:args.chunk].astype(np.float64), y, sidx, np.ones((1, len(ls))), Drows[:args.chunk])
                    e1 = max(float(np.nanmax(np.abs(u1[0] - pt[:args.chunk, 0]))), float(np.nanmax(np.abs(d1[0] - pt[:args.chunk, 1]))))
                    issues.append({"check": "boot_identity", "run": run, "target": T, "rep": r, "absdiff_max": e1})
                    if not e1 <= 1e-6:
                        raise SystemExit(f"ブートストラップの式が点の値に戻らない（{run} {T} 反復 {r}：{e1}）")
                    V = {q: np.full((args.boot, L.shape[0]), np.nan) for q in QN}
                    for a in range(0, L.shape[0], args.chunk):
                        *vals, empty, miss = boot_rep(L[a:a + args.chunk].astype(np.float64), y, sidx, Cb, Drows[a:a + args.chunk])
                        for q, v_ in zip(QN, vals):
                            V[q][:, a:a + args.chunk] = v_
                    ar = A[:, rep_col[r]]
                    w = ar * (~empty)
                    n_empty += float((ar * empty).sum())
                    n_miss += float((ar * miss).sum())
                    n_draw += float(ar.sum())
                    for j, rows_j in enumerate(krow):                          # 種は複製ごとに平均する（合奏ではない）
                        if not rows_j:
                            continue
                        ok = np.isfinite(V["uar"][:, rows_j].mean(1))
                        for q in QN:
                            vq = V[q][:, rows_j].mean(1)
                            num[q][:, j] += np.where(np.isfinite(vq), w * vq, 0)
                        den[:, j] += np.where(ok, w, 0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    B_ = {q: (num[q] / den).astype(np.float32) for q in QN}
                bu = B_["uar"]
                np.savez_compressed(out / f"boot_{run}_{T}.npz", **B_)
                kt = pd.DataFrame(p["keys"], columns=["student", "K", "wave", "method", "param", "meff", "seedmode"]).assign(
                    key_id=range(nk), run=run, target=T, n_seeds=p["nseeds"])
                keys_all.append(kt)
                meta.append({"run": run, "target": T, "corpus": cp, "n_speakers_corpus": len(speakers), "n_speakers_target": p["n_spk"],
                             "max_speaker_share": p["max_share"], "n_keys": nk, "boot": args.boot,
                             "frac_empty_rep_draws": n_empty / max(n_draw, 1), "frac_missing_class_draws": n_miss / max(n_draw, 1),
                             "n_nan_uar_reps": int(np.isnan(bu).sum()), "sec": round(time.perf_counter() - t0, 1)})
                print(f"{run} {T}：鍵 {nk}，ブートストラップ {args.boot} 回，{time.perf_counter() - t0:.0f} 秒", flush=True)
        if rows_all:
            pd.concat(rows_all, ignore_index=True).to_csv(out / f"rows_{run}.csv.gz", index=False)
        if pair_rows:
            pd.DataFrame(pair_rows).to_csv(out / f"pairs_{run}.csv.gz", index=False)
        if D_rows:
            pd.DataFrame(D_rows).to_csv(out / f"D_{run}.csv", index=False)
        if keys_all:
            pd.concat(keys_all, ignore_index=True).to_csv(out / f"keys_{run}.csv", index=False)
        if diag_rows:
            pd.DataFrame(diag_rows).to_csv(out / f"diag_{run}.csv", index=False)
        if null_rows:
            pd.DataFrame(null_rows).to_csv(out / f"model_null_{run}.csv", index=False)
    pd.DataFrame(meta).to_csv(out / "boot_meta.csv", index=False)
    pd.DataFrame(issues).to_csv(out / "checks.csv", index=False)
    write_prov(out, "measure", args, {"delta": dmeta, "chain": chain, "boot": args.boot, "seed": args.seed, "n_issues": len(issues),
                                      "select_chosen_sha256": sha256(Path(args.out) / "select" / "chosen.csv"),
                                      "select_fixed_sha256": sha256(Path(args.out) / "select" / "fixed.csv"), "inputs": inputs})
    print(f"測定：記録 {len(issues)} 件", flush=True)


def null_model_error(y, d6, o6, rng, draws=NULL_DRAWS):
    """模型（等分散の正規）が正しいときの 6 対の平均の |A(d̂, ô) − 実際の 2 感情の正解率|（ポイント）の分布．テストの感情の本数をそのまま使う。"""
    errs = np.zeros(draws)
    used = 0
    for p, (k, j) in enumerate(PAIRS):
        nk, nj = int((y == k).sum()), int((y == j).sum())
        if nk < 2 or nj < 2 or not (np.isfinite(d6[p]) and np.isfinite(o6[p])):
            continue
        mk, mj = d6[p] / 2 - o6[p], -d6[p] / 2 - o6[p]
        zk, zj = rng.normal(mk, 1.0, (draws, nk)), rng.normal(mj, 1.0, (draws, nj))
        ak, aj = zk.mean(1), zj.mean(1)
        sd = np.sqrt((zk.var(1, ddof=1) + zj.var(1, ddof=1)) / 2)
        dh, oh = (ak - aj) / sd, -(ak + aj) / 2 / sd
        acc = 0.5 * ((zk > 0).mean(1) + (zj < 0).mean(1))
        errs += np.abs(ba(dh, oh) - acc)
        used += 1
    return 100 * errs / max(used, 1)


def measure_target(args, run, T, d77, d78, sel, fixed, issues, rows_all, pair_rows, D_rows, diag_rows, c77, c78, inputs=None):
    """1 つの評価のターゲット：全行の反復ごとの点の値，D，選んだ行の 6 対の内訳，診断．ブートストラップに渡す読み込みの関数を返す。"""
    t0 = time.perf_counter()
    tab = {"e77": eval_table(d77, T)}
    if d78 is not None:
        tab["e78"] = eval_table(d78, T)
        if tab["e78"] is None:
            fail_or_note(args, issues, check="e78_eval_missing", run=run, target=T)
            tab.pop("e78")
    if inputs is not None:                                                     # 読んだ評価の出力（関門 1 の I9-1）．対数の確率は大きさと更新の時刻だけ
        for d_ in (d77, d78):
            if d_ is None:
                continue
            for nm in (f"{T}.csv", f"{T}_samples.csv", f"{T}_diag.csv"):
                if (Path(d_) / nm).exists():
                    inputs[str(Path(d_) / nm)] = sha256(Path(d_) / nm)
            for f_ in sorted(Path(d_).glob(f"{T}_r*_logp.npy")):
                st_ = f_.stat()
                inputs[str(f_)] = {"bytes": st_.st_size, "mtime": st_.st_mtime}
    for w, t in tab.items():
        cm = set(t.commit.astype(str))
        if len(cm) != 1:
            fail_or_note(args, issues, check="commit_mixed", run=run, target=T, wave=w, commits=sorted(cm))
        issues.append({"check": "commit_of_eval_csv", "run": run, "target": T, "wave": w, "commits": sorted(cm),
                       "provenance_commit": c77 if w == "e77" else c78})
        reps = set(t.rep)
        if reps != set(args.reps):
            fail_or_note(args, issues, check="reps_incomplete", run=run, target=T, wave=w, have=sorted(reps))
        cnt = t.groupby(["rep"]).size()
        if cnt.nunique() > 1:
            fail_or_note(args, issues, check="rows_per_rep_differ", run=run, target=T, wave=w)
        dup = t.duplicated(["rep", "student", "method", "param", "meff", "K"]).sum()
        if dup:
            fail_or_note(args, issues, check="duplicate_rows", run=run, target=T, wave=w, n=int(dup))
    samp = pd.read_csv(Path(d77) / f"{T}_samples.csv")
    df, X = load_any(T, run)
    if not np.array_equal(df.sample_id.astype(str).to_numpy(), samp.sample_id.astype(str).to_numpy()):
        raise SystemExit(f"{run} {T}：_samples.csv の sample_id の並びが load_any と違う")
    if d78 is not None and (Path(d78) / f"{T}_samples.csv").exists():
        s78 = pd.read_csv(Path(d78) / f"{T}_samples.csv")
        if not np.array_equal(s78.sample_id.astype(str).to_numpy(), samp.sample_id.astype(str).to_numpy()):
            raise SystemExit(f"{run} {T}：E77 と E78 の _samples.csv が違う")
    yall = df.label_4.to_numpy().astype(int)
    spk_all = samp.speaker_group.astype(str).to_numpy()
    feats = student_feats(run, X)
    students = sorted(tab["e77"].student.unique())
    keys = key_table(sel, fixed, run, T, tab)
    reps = [r for r in args.reps if (Path(d77) / f"{T}_r{r}_logp.npy").exists()]
    Dval, te_of = {}, {}
    sel_rows = {}                                                              # (wave, rep) → 鍵ごとの行の番号
    for r in reps:
        te = np.load(Path(d77) / f"{T}_r{r}_te.npy")
        te_of[r] = te
        y = yall[te]
        pool = np.setdiff1d(np.arange(len(yall)), te)
        for w, t in tab.items():
            dw = d77 if w == "e77" else d78
            if not (Path(dw) / f"{T}_r{r}_logp.npy").exists():
                fail_or_note(args, issues, check="logp_missing", run=run, target=T, wave=w, rep=r)
                continue
            if w == "e78" and not np.array_equal(np.load(Path(dw) / f"{T}_r{r}_te.npy"), te):
                raise SystemExit(f"{run} {T} 反復 {r}：E77 と E78 のテストの行が違う")
            Lm = np.load(Path(dw) / f"{T}_r{r}_logp.npy", mmap_mode="r")
            Pm = np.load(Path(dw) / f"{T}_r{r}_pred.npy", mmap_mode="r")
            tr = t[t.rep == r].sort_values("row_id")
            if len(tr) != Lm.shape[0] or Lm.shape[1] != len(te):
                raise SystemExit(f"{run} {T} {w} 反復 {r}：行の数かテストの発話の数が合わない（{len(tr)}・{Lm.shape}・{len(te)}）")
            out_r = []
            for a in range(0, Lm.shape[0], args.chunk):
                Lc = np.asarray(Lm[a:a + args.chunk], dtype=np.float64)
                Pc = np.asarray(Pm[a:a + args.chunk]).astype(int)
                if not np.isfinite(Lc).all():
                    raise SystemExit(f"NaN/Inf：{run} {T} {w} 反復 {r}")
                u = np.array([shift_uar(y, pc) for pc in Pc])
                d, o, _, _, _ = pair_moments(Lc, y, full=False)
                nmis = int((Lc.argmax(2) != Pc).sum())
                out_r.append(pd.DataFrame({"a": np.arange(a, a + len(Lc)), "uar_pred": 100 * u, "argmax_mismatch": nmis,
                                           **{f"d{p}": d[:, p] for p in range(6)}, **{f"o{p}": o[:, p] for p in range(6)}}))
                if a == 0:                                                     # 既存の d_o と同じ値か（I6）
                    for i in range(min(3, len(Lc))):
                        ref = np.array(d_o(Lc[i], y))
                        dm = float(np.nanmax(np.abs(ref - np.stack([d[i], o[i]], 1))))
                        if not dm <= 1e-9:
                            raise SystemExit(f"d・o が explore_synergy.d_o と違う（{dm}）")
            o_r = pd.concat(out_r, ignore_index=True)
            tr = tr.reset_index(drop=True).join(o_r)
            ud = float((tr.uar_pred - 100 * tr.uar).abs().max())
            if ud > 1e-7:
                fail_or_note(args, issues, check="uar_vs_csv", run=run, target=T, wave=w, rep=r, absdiff_max=ud)
            tr["wave"] = w
            if not np.array_equal(tr.row_id.to_numpy(), np.arange(len(tr))):
                raise SystemExit(f"{run} {T} {w} 反復 {r}：row_id が 0 から並んでいない")
            for c_ in ("uar_bfix", "bias_fit_ok", "a_auc", "e_model", "e_model_abs"):
                tr[c_] = np.nan
            rows_all.append(tr)
            sel_rows[(w, r)] = tr
        # D：候補（プール）のラベルで学んだ LDA の向きをテストで測る．プールの平均と標準偏差で標準化する
        for st in students:
            Xs = np.asarray(feats[st], dtype=np.float64)
            mu, sd = Xs[pool].mean(0), Xs[pool].std(0)
            sd = np.where(sd > 1e-12, sd, 1.0)
            Xp, Xt = (Xs[pool] - mu) / sd, (Xs[te] - mu) / sd
            Dv = lda_sep(Xp, yall[pool], Xt, y)
            Dval[(st, r)] = Dv
            for p in range(6):
                D_rows.append({"run": run, "target": T, "student": st, "rep": r, "pair": p, "D": Dv[p]})
            if run not in OWN_CHECKED:                                         # explore_pairs.own_separation と同じ式か（両向きの平均．run ごとに 1 回）
                half = np.r_[np.zeros(len(pool), int), np.ones(len(te), int)]
                Xc = np.vstack([Xp, Xt])
                yc = np.r_[yall[pool], y]
                ref = own_separation(Xc, yc, half)
                mine = (Dv + lda_sep(Xt, y, Xp, yall[pool])) / 2
                if not np.allclose(ref, mine, rtol=1e-8, atol=1e-10):
                    raise SystemExit("D の LDA が explore_pairs.own_separation と違う")
                OWN_CHECKED.add(run)
                issues.append({"check": "lda_vs_own_separation", "run": run, "target": T, "student": st, "rep": r,
                               "absdiff_max": float(np.abs(ref - mine).max())})
    # 鍵ごとの反復の値（点）と，選んだ行の 6 対の内訳・切片だけを直した UAR
    krow, nseeds, krows_rep = [], [], {}
    for key in keys:
        x = rows_of_key(tab, key)
        nseeds.append(0 if x is None or x.empty else int(x.groupby("rep").size().max()))
    for r in reps:
        y = yall[te_of[r]]
        rows_r = []
        for j, key in enumerate(keys):
            wave = key[2]
            tr = sel_rows.get((wave, r))
            x = rows_of_key(tab, key)
            if tr is None or x is None or x.empty:
                rows_r.append([])
                continue
            ids = x[x.rep == r].row_id.tolist()
            rows_r.append([(wave, i) for i in ids])
        krows_rep[r] = rows_r
        flat = sorted({wi for lst in rows_r for wi in lst})
        pos = {wi: i for i, wi in enumerate(flat)}
        if flat:
            arrs = {}
            for w in {wv for wv, _ in flat}:
                dw = d77 if w == "e77" else d78
                ids = [i for wv, i in flat if wv == w]
                arrs[w] = (ids, np.asarray(np.load(Path(dw) / f"{T}_r{r}_logp.npy", mmap_mode="r")[ids], dtype=np.float64))
            L = np.zeros((len(flat), len(y), 4))
            for w, (ids, a) in arrs.items():
                for i, rid in enumerate(ids):
                    L[pos[(w, rid)]] = a[i]
            d, o, acc, vr, auc = pair_moments(L, y)
            st_of = {}
            for j, key in enumerate(keys):
                for wi in rows_r[j]:
                    st_of[wi] = key[0]
            for wi, i in pos.items():                                          # 行ごとに 1 回（鍵との対応は rows_*.csv と keys_*.csv から引ける）
                ub, okb = uar_bias_fixed(L[i], y) if args.bias_fixed else (np.nan, True)
                Dv = Dval[(st_of[wi], r)]
                dauc6 = np.sqrt(2) * stats.norm.ppf(np.clip(auc[i], 1e-12, 1 - 1e-12))
                em = ba(d[i], o[i]) - acc[i]
                trw = sel_rows[(wi[0], r)]                                    # 行の表に，切片だけを直した UAR と模型の量を書く（判定の段が鍵ごとに読む）
                trw.loc[wi[1], ["uar_bfix", "bias_fit_ok", "a_auc", "e_model", "e_model_abs"]] = [
                    100 * ub, float(okb), 100 * np.nanmean(ba(dauc6, 0.0)), 100 * np.nanmean(em), 100 * np.nanmean(np.abs(em))]
                for p in range(6):
                    dauc = np.sqrt(2) * stats.norm.ppf(np.clip(auc[i, p], 1e-12, 1 - 1e-12))
                    pair_rows.append({"run": run, "target": T, "rep": r, "wave": wi[0], "row_id": wi[1], "student": st_of[wi], "pair": p,
                                      "d": d[i, p], "o": o[i, p], "D": Dv[p], "acc_actual": 100 * acc[i, p],
                                      "acc_model": 100 * ba(d[i, p], o[i, p]), "var_ratio": vr[i, p], "d_auc": dauc,
                                      "uar_bias_fixed": 100 * ub, "bias_fit_ok": okb})
    # 向きの損と位置の損（全行．反復ごと）
    for tr in [x for x in rows_all if x.target.iloc[0] == T]:
        r = int(tr.rep.iloc[0])
        dmat = tr[[f"d{p}" for p in range(6)]].to_numpy()
        omat = tr[[f"o{p}" for p in range(6)]].to_numpy()
        Dm = np.stack([Dval.get((st, r), np.full(6, np.nan)) for st in tr.student])
        a0 = ba(dmat, 0.0)
        aD0, aDo, ado = ba(Dm, 0.0), ba(Dm, omat), ba(dmat, omat)
        tr["dirloss"] = 100 * np.nanmean(aD0 - a0, 1)
        tr["posloss"] = 100 * np.nanmean(a0 - ado, 1)
        tr["dirloss_rev"] = 100 * np.nanmean(aDo - ado, 1)                    # 逆の順序（関門 1 の R4-1）：向きを o の下で，位置を D で測る
        tr["posloss_rev"] = 100 * np.nanmean(aD0 - aDo, 1)
        tr["n_neg_dir_pairs"] = (aD0 - a0 < 0).sum(1)
    # 模型が正しいときの誤差の分布（テストの本数と，その反復の none の K=0 の d・o で作る．関門 1 の define の D8）
    rng_null = np.random.default_rng([args.seed, zlib.crc32(run.encode()), zlib.crc32(T.encode())])
    for r in reps:
        tr = sel_rows.get(("e77", r))
        if tr is None:
            continue
        y = yall[te_of[r]]
        for st in students:
            nn = tr[(tr.student == st) & (tr.method == "none") & (tr.K == "0")]
            if nn.empty:
                continue
            e = null_model_error(y, nn[[f"d{p}" for p in range(6)]].to_numpy()[0], nn[[f"o{p}" for p in range(6)]].to_numpy()[0], rng_null)
            args._null_rows.append({"run": run, "target": T, "student": st, "rep": r, "e_null_median": float(np.median(e)),
                                    "e_null_95": float(np.quantile(e, 0.95)), "n_test": int(len(y)),
                                    "n_min_class": int(np.bincount(y, minlength=4).min())})
    # 診断：E77 と E78 の none，K=0 の mapctl と none，教師の確率，DA の項と対照の予測の食い違い，Co-teaching の ṽ
    diagnostics(args, run, T, d77, d78, tab, reps, sel, issues, diag_rows)
    speakers = set(spk_all[np.concatenate([te_of[r] for r in reps])]) if reps else set()
    if reps:                                                                   # 話者群の数と，最大の話者群が占めるテストの発話の割合（関門 1 の S3）
        vc = pd.Series(spk_all[np.concatenate([te_of[r] for r in reps])]).value_counts(normalize=True)
        n_spk, max_share = int(len(vc)), float(vc.iloc[0])
    else:
        n_spk, max_share = 0, float("nan")

    def load(r):
        rows_r = krows_rep.get(r)
        if not rows_r:
            return None, None, None, None, None, None
        flat = sorted({wi for lst in rows_r for wi in lst})
        pos = {wi: i for i, wi in enumerate(flat)}
        te = te_of[r]
        L = np.zeros((len(flat), len(te), 4), np.float32)
        for w in {wv for wv, _ in flat}:
            dw = d77 if w == "e77" else d78
            ids = [i for wv, i in flat if wv == w]
            a = np.load(Path(dw) / f"{T}_r{r}_logp.npy", mmap_mode="r")[ids]
            for i, rid in enumerate(ids):
                L[pos[(w, rid)]] = a[i]
        Drows = np.full((len(flat), 6), np.nan)                                 # 行ごとの D（その行の生徒の D）
        for j, lst in enumerate(rows_r):
            for wi in lst:
                Drows[pos[wi]] = Dval[(keys[j][0], r)]
        pt = np.array([sel_rows[(w, r)].set_index("row_id").loc[rid, ["uar", "dirloss"]].to_numpy(float) * [100, 1] for w, rid in flat])
        return L, yall[te], spk_all[te], Drows, [[pos[wi] for wi in lst] for lst in rows_r], pt
    print(f"{run} {T}：点の値と D（{len(reps)} 反復），{time.perf_counter() - t0:.0f} 秒", flush=True)
    return {"keys": keys, "nseeds": nseeds, "reps": reps, "speakers": speakers, "load": load, "students": students,
            "n_spk": n_spk, "max_share": max_share}


def diagnostics(args, run, T, d77, d78, tab, reps, sel, issues, diag_rows):
    """E77 と E78 の同じはずの値（許容幅を超えたら止める．I14-1），DA の項の効き（種の間の食い違いと比べる），cotzero の届かなさ，
    E78 の `_diag.csv` の DA の項と Co-teaching の診断（選んだ設定）。"""
    t77, t78 = tab.get("e77"), tab.get("e78")

    def chosen(st, K, selname, variant="excl"):
        ch = sel[(sel.run == run) & (sel.target == T) & (sel.student == st) & (sel.K == K) & (sel.selector == selname)
                 & (sel.variant == variant) & (sel.method == selname)]
        return None if ch.empty else ch.param.iloc[0]

    def rows78(a78, st, meth, K, pn):
        x = a78[(a78.student == st) & (a78.method == meth) & (a78.K == K) & (a78.pnorm == pn)]
        return {int(sd): int(rid) for sd, rid in zip(x.seed, x.row_id)}

    def add(r, st, K, what, value, **kw):
        diag_rows.append({"run": run, "target": T, "rep": r, "student": st, "K": K, "what": what, "value": value, **kw})
    for r in reps:
        L77 = np.load(Path(d77) / f"{T}_r{r}_logp.npy", mmap_mode="r")
        a77 = t77[t77.rep == r]
        if t78 is None:
            continue
        L78 = np.load(Path(d78) / f"{T}_r{r}_logp.npy", mmap_mode="r")
        P78 = np.load(Path(d78) / f"{T}_r{r}_pred.npy", mmap_mode="r")
        a78 = t78[t78.rep == r]
        for st in a77.student.unique():
            for K in KS:
                n7 = a77[(a77.student == st) & (a77.method == "none") & (a77.K == K)].row_id
                n8 = a78[(a78.student == st) & (a78.method == "none") & (a78.K == K)].row_id
                if len(n7) and len(n8):
                    v = float(np.abs(L77[int(n7.iloc[0])] - L78[int(n8.iloc[0])]).max())
                    add(r, st, K, "none_e77_vs_e78_logp_absmax", v)
                    if not v <= TOL_LOGP:
                        fail_or_note(args, issues, check="none_e77_vs_e78_over_tol", run=run, target=T, rep=r, student=st, K=K, absdiff_max=v)
                if K == "0" and len(n8):                                     # K=0 の mapctl は none と同じ解のはず
                    for rid in a78[(a78.student == st) & (a78.method == "mapctl") & (a78.K == "0")].row_id:
                        add(r, st, K, "mapctl_K0_vs_none_logp_absmax", float(np.abs(L78[int(rid)] - L78[int(n8.iloc[0])]).max()))
                nid = int(n8.iloc[0]) if len(n8) else None
                # DA の項：a ＝ 選んだ DANN・MME（種 0）と同じ lr・反復・μ の mapctl（種 0）の食い違い，b ＝ 選んだ設定の種 0・1・2 の 3 対の食い違いの最大
                for sm in ("dann", "mme", "mapctl", "xmuda"):
                    pn = chosen(st, K, sm)
                    if pn is None:
                        continue
                    sr = rows78(a78, st, sm, K, pn)
                    pairs_ = [(i, j) for i in sr for j in sr if i < j]
                    if pairs_:
                        add(r, st, K, f"{sm}_seed_disagree_max", float(max((P78[sr[i]] != P78[sr[j]]).mean() for i, j in pairs_)), n_seeds=len(sr))
                    if sm in ("dann", "mme") and 0 in sr:
                        m = re.match(r"lr(?P<lr>[^i]+)it(?P<it>\d+)lam[^m]+mu(?P<mu>[0-9.]+)", pn)
                        cr = rows78(a78, st, "mapctl", K, f"lr{m['lr']}it{m['it']}mu{m['mu']}") if m else {}
                        if 0 in cr:
                            add(r, st, K, f"{sm}_vs_mapctl_disagree", float((P78[sr[0]] != P78[cr[0]]).mean()))
                        if nid is not None:
                            add(r, st, K, f"{sm}_vs_none_disagree", float((P78[sr[0]] != P78[nid]).mean()))
                # cotzero（候補の項の重み 0）の none への届かなさ：学習率と種ごとの予測の食い違いの割合（E78 の再検査 I13-2）
                if nid is not None:
                    cz = a78[(a78.student == st) & (a78.method == "cotzero") & (a78.K == K)]
                    for pn_, sd_, rid in zip(cz.pnorm, cz.seed, cz.row_id):
                        lr = pn_.split("lr")[-1]
                        add(r, st, K, f"cotzero_lr{lr}_vs_none_disagree", float((P78[int(rid)] != P78[nid]).mean()), seed=int(sd_))
        f7, f8 = Path(d77) / f"{T}_r{r}_teachers.npz", Path(d78) / f"{T}_r{r}_teachers.npz"
        if f7.exists() and f8.exists():                                       # 第 2 波の Co-teaching に同じ教師を渡したか（評価のターゲットだけ）
            z7, z8 = np.load(f7), np.load(f8)
            for k in sorted(set(z7.files) & set(z8.files)):
                v = float(np.abs(z7[k] - z8[k]).max())
                add(r, "*", "-", f"teacher_{k}_absmax", v)
                if not v <= TOL_LOGP:
                    fail_or_note(args, issues, check="teacher_e77_vs_e78_over_tol", run=run, target=T, rep=r, teacher=k, absdiff_max=v)
    if d78 is not None and (Path(d78) / f"{T}_diag.csv").exists():
        dg = pd.read_csv(Path(d78) / f"{T}_diag.csv", dtype={"K": str, "param": str, "meff": str})
        if "target" in dg.columns and len(dg):
            dg = dg[dg.target.notna()]
            dg["pnorm"] = dg.param.map(strip_seed)
            # DA の項の記述（選んだ設定．反復と種で平均）．判別器の損失は DANN の行だけ，エントロピーは MME の行だけで読む（E78 の再検査）
            cols = {"dann": ("anorm", "dom_first", "dom_last"), "mme": ("anorm", "ent_first", "ent_last"), "xmuda": ("anorm", "kl_first", "kl_last"),
                    "mapctl": ("anorm",)}
            for (st, K, selname), g in sel[(sel.run == run) & (sel.target == T) & (sel.variant == "excl")
                                            & sel.selector.isin(list(cols))].groupby(["student", "K", "selector"]):
                pn = g.param.iloc[0]
                if g.method.iloc[0] != selname:
                    continue
                x = dg[(dg.student == st) & (dg.method == selname) & (dg.K == K) & (dg.pnorm == pn)]
                for col in cols[selname]:
                    if col in x.columns:
                        add(-1, st, K, f"{selname}_{col}", float(x[col].mean()) if len(x) else np.nan, n=int(len(x)))
            if "v_norm" in dg.columns:                                         # Co-teaching の ṽ と一致の割合（選んだ設定）
                for r_ in sel[(sel.run == run) & (sel.target == T) & (sel.selector == "cot") & (sel.method == "cot")].itertuples():
                    x = dg[(dg.student == r_.student) & (dg.method == "cot") & (dg.K == r_.K) & (dg.meff == r_.meff) & (dg.pnorm == r_.param)]
                    for col in ("v", "v_norm", "agree_sel", "agree_drop"):
                        add(-1, r_.student, r_.K, f"cot_{r_.variant}_{col}", float(x[col].mean()) if len(x) else np.nan, n=int(len(x)))


# ---------------------------------------------------------------- 段 (d)・(e)：判定と感度の行

def judge_unit(x, lo, hi, dl):
    if not (np.isfinite(x) and np.isfinite(lo) and np.isfinite(hi) and np.isfinite(dl)):
        return "missing"
    if lo > 0 and x >= dl:
        return "pos"
    if hi < 0 and x <= -dl:
        return "neg"
    if lo >= -dl and hi <= dl:
        return "same"
    return "undet"


QN = ("uar", "dirloss", "posloss", "dirloss_rev", "posloss_rev")


class Store:
    """measure の出力を読み，(run, target, 鍵) → 点の値（反復の平均）と複製を返す。"""

    def __init__(self, mdir, runs):
        self.mdir, self.keys, self.boot, self.point, self.idx, self.keylist = Path(mdir), {}, {}, {}, {}, {}
        for run in runs:
            f = self.mdir / f"keys_{run}.csv"
            if not f.exists():
                continue
            k = pd.read_csv(f, dtype={"K": str, "param": str, "meff": str})
            self.keylist[run] = k
            self.keys[run] = {tuple(r[:8]): int(r[8]) for r in k[["target", "student", "K", "wave", "method", "param", "meff", "seedmode", "key_id"]]
                              .itertuples(index=False)}
            rows = pd.read_csv(self.mdir / f"rows_{run}.csv.gz", dtype={"K": str, "param": str, "meff": str})
            rows["pnorm"] = [strip_seed(p) if m in SEEDED else p for m, p in zip(rows.method, rows.param)]
            rows["seed"] = [seed_of(m, p) for m, p in zip(rows.method, rows.param)]
            self.point[run] = rows
            self.idx[run] = {g: ix for g, ix in rows.groupby(["target", "student", "K", "wave", "method", "meff"]).groups.items()}

    def rows_of(self, run, T, st, K, wave, meth, pn, meff, md):
        ix = self.idx.get(run, {}).get((T, st, K, wave, meth, meff))
        x = self.point[run].loc[ix] if ix is not None else self.point[run].iloc[:0]
        x = x[x.pnorm.str.startswith(pn[:-1])] if pn.endswith(":*") else x[x.pnorm == pn]
        return x[x.seed == 0] if md == "s0" else x

    def get(self, run, T, st, K, wave, meth, pn, meff, md):
        """点の値（種を反復ごとに平均してから反復で平均）と，ブートストラップの複製（鍵が無ければ None）。"""
        pt = {q: np.nan for q in QN + ("uar_bfix", "uar_bfix_sd", "a_auc", "e_model", "ofix")}
        if run not in self.point:
            return pt, None
        x = self.rows_of(run, T, st, K, wave, meth, pn, meff, md)
        if len(x):
            g = x.groupby("rep")
            for q in QN:
                pt[q] = g[q].mean().mean() * (100 if q == "uar" else 1)
            for q in ("uar_bfix", "a_auc", "e_model"):
                if q in x.columns:
                    pt[q] = g[q].mean().mean()
            if "uar_bfix" in x.columns:
                v = g.uar_bfix.mean()
                pt["uar_bfix_sd"] = v.std(ddof=1) if v.notna().sum() > 1 else np.nan
                pt["ofix"] = pt["uar_bfix"] - pt["uar"]                        # 切片で取り戻せる分（切片だけを直した UAR − UAR．負になりうる）
        j = self.keys.get(run, {}).get((T, st, K, wave, meth, pn, meff, md))
        if j is None:
            return pt, None
        if (run, T) not in self.boot:
            f = self.mdir / f"boot_{run}_{T}.npz"
            z = np.load(f) if f.exists() else None
            self.boot[(run, T)] = None if z is None else {q: z[q] for q in QN if q in z.files}
        z = self.boot[(run, T)]
        return pt, (None if z is None else {q: z[q][:, j] for q in z})


METRIC = {  # 値の向き：+1 なら x − y，−1 なら y − x（向きの損と位置の損は「減らしたら正」にする）．δ の種類．区間の有無
    "uar": ("uar", 1, "u", True), "dir": ("dirloss", -1, "d", True), "pos": ("posloss", -1, "d", True),
    "dir_rev": ("dirloss_rev", -1, "d", True), "pos_rev": ("posloss_rev", -1, "d", True),
    "bfix": ("uar_bfix", 1, "u", False), "ofix": ("ofix", 1, "u", False),
}


def stage_judge(args, dirs, out):
    chain = check_chain("judge", args)
    mdir = Path(args.out) / "measure"
    S = Store(mdir, list(dirs))
    delta = pd.read_csv(Path(args.out) / "delta" / "delta.csv", dtype={"K": str})
    sel = pd.read_csv(Path(args.out) / "select" / "chosen.csv", dtype={"K": str, "param": str, "meff": str})
    fixed = pd.read_csv(Path(args.out) / "select" / "fixed.csv", dtype={"K": str, "param": str, "meff": str})
    units = []
    SEL = {(r.run, r.target, r.student, r.K, r.selector, r.variant): r for r in sel.itertuples(index=False)}
    if "seedmode" not in fixed.columns:
        fixed["seedmode"] = np.nan
    FIX = {(r.run, r.target, r.student, r.K, r.fixname): r for r in fixed.itertuples(index=False)}
    DEL = {(r.run, r.student, r.K): r for r in delta.itertuples(index=False)}

    def pick(run, T, st, K, selector, variant):
        r = SEL.get((run, T, st, K, selector, variant))
        if r is None or pd.isna(r.method) or r.method == EMPTY:
            return None
        return (r.wave, r.method, r.param, r.meff, None)

    def fixkey(run, T, st, K, name):
        r = FIX.get((run, T, st, K, name))
        if r is None:
            return None
        md = r.seedmode if isinstance(r.seedmode, str) else ("agg" if r.method in SEEDED else "-")
        return (r.wave, r.method, r.param, r.meff, md)

    def refkey(meth, pn):
        return ("e77", meth, pn, "0", "-")

    def val(run, T, st, K, k, s0=False):
        if k is None:
            return {q: np.nan for q in QN + ("uar_bfix", "a_auc", "e_model", "ofix")}, None
        wave, meth, pn, meff = k[:4]
        md = k[4] if isinstance(k[4], str) else ("s0" if s0 and meth in SEEDED else ("agg" if meth in SEEDED else "-"))
        if meth == "none":
            md = "-"
        return S.get(run, T, st, K, wave, meth, pn, meff, md)

    NONE = ("e77", "none", "-", "0", "-")

    def dl_of(run, st, K, kind, scale):
        r = DEL.get((run, st, K))
        if r is None:
            return np.nan
        return float(r.delta_d if kind == "d" else r.delta_u) * scale

    def zone(x, dl):
        return "pos" if x >= dl else ("neg" if x <= -dl else "mid")

    def compare(family, comp, run, st, K, kx, ky, metric, variant, sens, s0=False, iem_mode="both"):
        """値 = METRIC の向き（uar は x − y，dir などは y − x）．コーパスごとに点の値と 2 段のブートストラップの区間を作り，判定する。"""
        q, sign, dkind, has_ci = METRIC[metric]
        targets = [t for t in groups_for(run) if (args.targets is None or t in args.targets)]
        byc = {}
        for T in targets:
            if iem_mode == "impro_only" and T == "IEMOCAP-script":
                continue
            byc.setdefault(corp(T), []).append(T)
        for cp, Ts in byc.items():
            pts, reps_, aucs, des = [], [], [], []
            for T in Ts:
                kxx = kx(T) if callable(kx) else kx
                kyy = ky(T) if callable(ky) else ky
                px, bx = val(run, T, st, K, kxx, s0)
                py, by = val(run, T, st, K, kyy, s0)
                pts.append(sign * (px[q] - py[q]))
                aucs.append(sign * -(px["a_auc"] - py["a_auc"]) if metric == "dir" else np.nan)   # 向きの損の差を Φ(d_AUC/2) で測った値
                des.append(px["e_model"] - py["e_model"] if metric == "dir" else np.nan)
                reps_.append(None if (not has_ci or bx is None or by is None or q not in bx) else sign * (bx[q] - by[q]))
            x = float(np.mean(pts)) if pts else np.nan
            if any(r_ is None for r_ in reps_) or not reps_:
                lo = hi = np.nan
                nvalid = 0
            else:
                b = np.mean(np.stack(reps_), 0)
                b = b[np.isfinite(b)]
                nvalid = len(b)
                lo, hi = (np.percentile(b, [2.5, 97.5]) if nvalid else (np.nan, np.nan))
            xa, de = float(np.mean(aucs)), float(np.mean(des))
            for scale, stag in ((1.0, ""), (0.5, "delta_half"), (2.0, "delta_double")):
                dl = dl_of(run, st, K, dkind, scale)
                md = np.nan
                if metric == "dir" and np.isfinite(x) and np.isfinite(xa) and np.isfinite(dl):   # 模型に依る単位の印（3 節の表の 1・2 行目）
                    md = bool(abs(x - xa) > MODEL_UNIT_FRAC * dl or abs(de) > MODEL_UNIT_FRAC * dl or zone(x, dl) != zone(xa, dl))
                units.append({"family": family, "comparison": comp, "run": run, "student": st, "role": role_of(run, st), "K": K,
                              "corpus": cp, "n_groups": len(Ts), "metric": metric, "point": x, "lo": lo, "hi": hi, "n_boot_valid": nvalid,
                              "delta": dl, "judgment": judge_unit(x, lo, hi, dl) if has_ci else "point_only", "variant": variant,
                              "point_auc": xa, "model_err_diff": de, "model_dependent": md,
                              "sensitivity": ",".join([s_ for s_ in (sens, stag) if s_]) or "main",
                              "note": NOTES["oursx_small"] if sens == "oursx_small" else NOTES.get(comp.split("-")[-1], "")})

    for run in dirs:
        world = run in WORLDS
        stus = sorted(set(sel[sel.run == run].student))
        refk = S.keylist.get(run)
        for st in stus:
            role = role_of(run, st)
            fam4 = "P4" if role == "kind" else ("P4'" if role == "fused" else "P4_uni_world")
            for K in KS:
                famK = fam4 + ("" if K in P4_KS or fam4 == "P4_uni_world" else "_otherK")
                for v, s0, sens in (("excl", False, ""), ("ns_excl", True, "noscript_reselect")):
                    def kx_of(mm, vv=v):
                        return lambda T: pick(run, T, st, K, mm, vv if (not vv.startswith("ns") or T in ("MOSEI", "MELD")) else "excl")
                    for meth in GROUP_B + ("t3a_b", "mapctl", "coralctl"):
                        if role == "fused" and meth in ("dann", "mme", "mapctl"):
                            continue
                        fam = famK if meth in GROUP_B else "P4_aux"
                        compare(fam, f"none-{meth}", run, st, K, kx_of(meth), NONE, "dir", v, sens, s0)
                        if not sens:
                            compare(fam, f"none-{meth}", run, st, K, kx_of(meth), NONE, "dir_rev", v, "rev_order", s0)
                            if run in ("text_roberta", "fuse-audio-video-text"):
                                compare(fam, f"none-{meth}", run, st, K, kx_of(meth), NONE, "dir", v, "iemocap_impro_only", s0, "impro_only")
                    # 対照との比較（関門 1 の C-2）：CORAL は coralctl − CORAL，DANN・MME は自分で選んだ mapctl（種 3 つ）− 手法（種 3 つ）
                    ctl = [("coral", "coralctl")] + ([] if role == "fused" else [("dann", "mapctl"), ("mme", "mapctl")])
                    for meth, cm in ctl:
                        fam = (famK if fam4 != "P4_uni_world" else fam4) + "_ctl"
                        compare(fam, f"{cm}-{meth}", run, st, K, kx_of(meth), kx_of(cm), "dir", v, sens, s0)
                    if sens:
                        continue
                    for meth in GROUP_B + ("massbal", "saerens", "t3a_b"):    # 記述（判定の族に入れない）：UAR，位置の損，切片だけを直した UAR，切片で取り戻せる分
                        if role == "fused" and meth in ("dann", "mme"):
                            continue
                        kx = (lambda T, m_=meth: fixkey(run, T, st, K, m_)) if meth in ("massbal", "saerens") else kx_of(meth)
                        for met in ("uar", "pos", "bfix", "ofix"):
                            compare(f"P4_desc_{met}", f"{meth}-none" if met in ("uar", "bfix", "ofix") else f"none-{meth}", run, st, K, kx, NONE,
                                    met, v, "")
                        if meth in ("coral", "dann", "mme"):
                            for cm in (("coralctl",) if meth == "coral" else ("mapctl",)):
                                compare("P4_ctl_uar", f"{meth}-{cm}", run, st, K, kx, kx_of(cm), "uar", v, "")
                    for meth in ("dann", "mme"):                               # 同じ lr・反復・μ の対照（種 0 の 1 回．種の幅と並べて読む．E78 の I5-2）
                        if role == "fused":
                            continue
                        compare("P4_dann_mme_s0", f"mapctl_match-{meth}", run, st, K, kx_of(meth),
                                lambda T, mm=meth: fixkey(run, T, st, K, f"mapctl_match_{mm}"), "dir", v, "", True)
                        compare("P4_dann_mme_s0", f"none-mapctl_match_{meth}", run, st, K, lambda T, mm=meth: fixkey(run, T, st, K, f"mapctl_match_{mm}"),
                                NONE, "dir", v, "", True)
                    for fx in ("im_fix", "t3a_fix", "lp_fix", "dann_fix", "mme_fix", "mapctl_fix", "coral_fix", "massbal", "saerens"):
                        if role == "fused" and fx in ("dann_fix", "mme_fix", "mapctl_fix"):
                            continue
                        compare("fixed", f"none-{fx}", run, st, K, lambda T, f_=fx: fixkey(run, T, st, K, f_), NONE, "dir", "fixed", "")
                        compare("fixed_uar", f"{fx}-none", run, st, K, lambda T, f_=fx: fixkey(run, T, st, K, f_), NONE, "uar", "fixed", "")
                # 参照の行（ref_*・教師）と none の差（区間つき．記述．関門 1 の S5・R10-1）
                if refk is not None:
                    rk = refk[(refk.student == st) & (refk.K == K) & refk.method.isin(REFS)][["method", "param"]].drop_duplicates()
                    for meth, pn in rk.itertuples(index=False):
                        nm = f"{meth}:{pn}" if meth == "teacher" else meth
                        compare("refs", f"{nm}-none", run, st, K, refkey(meth, pn), NONE, "uar", "ref", "")
                        compare("refs_dir", f"none-{nm}", run, st, K, refkey(meth, pn), NONE, "dir", "ref", "")
                if not world or K not in P5_KS:
                    continue
                OX = lambda T, s_="oursx", v_="incl": pick(run, T, st, K, s_, v_)  # noqa: E731
                for oxname in ("oursx", "oursx_small"):
                    sens = "" if oxname == "oursx" else "oursx_small"
                    compare("P5-1", "oursx-kd", run, st, K, lambda T, o_=oxname: OX(T, o_), lambda T: pick(run, T, st, K, "kd", "incl"), "uar", "incl", sens)
                    compare("P5-2", "oursx-groupB", run, st, K, lambda T, o_=oxname: OX(T, o_), lambda T: pick(run, T, st, K, "groupB", "incl"), "uar", "incl", sens)
                compare("P5-3", "none-oursx", run, st, K, OX, NONE, "dir", "incl", "")
                compare("P5-3", "none-oursx", run, st, K, OX, NONE, "dir_rev", "incl", "rev_order")
                compare("P5-3u", "oursx-none", run, st, K, OX, NONE, "uar", "incl", "")
                for met in ("bfix", "ofix", "pos"):                            # P5-3 を 4 クラスのまま分ける（点の値．切片だけを直した UAR）と，模型の位置の損
                    compare(f"P5-3_desc_{met}", "oursx-none" if met != "pos" else "none-oursx", run, st, K, OX, NONE, met, "incl", "")
                compare("P5-4", "oursx-cot", run, st, K, OX, lambda T: pick(run, T, st, K, "cot", "incl"), "uar", "incl", "")
                COT = lambda T: pick(run, T, st, K, "cot", "incl")  # noqa: E731
                CTL = lambda T: pick(run, T, st, K, "cotctl", "incl")  # noqa: E731
                CZ = lambda T: fixkey(run, T, st, K, "cotzero_match")  # noqa: E731
                CZC = lambda T: fixkey(run, T, st, K, "cotzero_match_ctl")  # noqa: E731
                for comp, kx, ky, met in (("cot-cotctl", COT, CTL, "uar"), ("oursx-cotctl", OX, CTL, "uar"),
                                          ("cot-cotzero_match", COT, CZ, "uar"), ("cotctl-cotzero_match_ctl", CTL, CZC, "uar"),
                                          ("cotzero_match-none", CZ, NONE, "uar"), ("none-cotzero_match", CZ, NONE, "dir"),
                                          ("cotfix-none", lambda T: fixkey(run, T, st, K, "cotfix"), NONE, "uar"),
                                          ("cot_g_match-cot", lambda T: fixkey(run, T, st, K, "cot_g_match"), COT, "uar")):
                    compare("P5-4_breakdown" + ("_dir" if met == "dir" else ""), comp, run, st, K, kx, ky, met, "incl", "")
                if run == "fuse-audio-video-text":
                    for fam, comp, kx, ky, met in (("P5-1", "oursx-kd", OX, lambda T: pick(run, T, st, K, "kd", "incl"), "uar"),
                                                   ("P5-2", "oursx-groupB", OX, lambda T: pick(run, T, st, K, "groupB", "incl"), "uar"),
                                                   ("P5-3", "none-oursx", OX, NONE, "dir"),
                                                   ("P5-4", "oursx-cot", OX, COT, "uar")):
                        compare(fam, comp, run, st, K, kx, ky, met, "incl", "iemocap_impro_only", False, "impro_only")
                ns = lambda T: "ns_incl" if T in ("MOSEI", "MELD") else "incl"  # noqa: E731
                for fam, comp, kx, ky in (("P5-1", "oursx-kd", lambda T: pick(run, T, st, K, "oursx", ns(T)), lambda T: pick(run, T, st, K, "kd", ns(T))),
                                          ("P5-2", "oursx-groupB", lambda T: pick(run, T, st, K, "oursx", ns(T)),
                                           lambda T: pick(run, T, st, K, "groupB", ns(T)))):
                    compare(fam, comp, run, st, K, kx, ky, "uar", "ns_incl", "noscript_reselect", True)
                # 配備できる参照と oursx（関門 1 の C-3）：合併（S+T）・ターゲットのみ（K = 5・25），W*（上限としてだけ），融合の生徒の教師
                if refk is not None:
                    rk = refk[(refk.student == st) & (refk.K == K) & refk.method.isin(REFS)][["method", "param"]].drop_duplicates()
                    for meth, pn in rk.itertuples(index=False):
                        if meth in ("ref_srconly",):
                            continue
                        nm = f"{meth}:{pn}" if meth == "teacher" else meth
                        compare("P5_refs", f"oursx-{nm}", run, st, K, OX, refkey(meth, pn), "uar", "incl", "")
                abl = ["abl_refa", "abl_refb", "abl_pF"] if role == "uni" else ["abl_ref0", "abl_poe0"]
                for a in abl + ["abl_bal", "abl_raw", "abl_hard", "abl_meff500", "abl_own", "abl_ownK", "ours", "xmuda", "cotrain", "cotrain_prod", "coreg"]:
                    if (a == "xmuda" and role != "uni") or (a == "cotrain_prod" and role != "fused"):
                        continue
                    fam = "ablation" if a.startswith("abl_") or a == "ours" else "groupC_explore"
                    compare(fam, f"oursx-{a}", run, st, K, OX, lambda T, a_=a: pick(run, T, st, K, a_, "incl"), "uar", "incl", "")
                    compare(fam + "_dir", f"{a}-oursx", run, st, K, lambda T, a_=a: pick(run, T, st, K, a_, "incl"), OX, "dir", "incl", "")
                    if fam == "ablation":
                        compare("ablation_ofix", f"oursx-{a}", run, st, K, OX, lambda T, a_=a: pick(run, T, st, K, a_, "incl"), "ofix", "incl", "")
                if role == "uni":                                              # 5.3 節：K 本で学び直すことの寄与
                    compare("sec5.3", "abl_refa-abl_pF", run, st, K, lambda T: pick(run, T, st, K, "abl_refa", "incl"),
                            lambda T: pick(run, T, st, K, "abl_pF", "incl"), "uar", "incl", "")
                else:
                    compare("sec5.3", "oursx-abl_ref0", run, st, K, OX, lambda T: pick(run, T, st, K, "abl_ref0", "incl"), "uar", "incl", "")
    U = pd.DataFrame(units)
    U["excl_mosei_afew"] = U.corpus.isin(["MOSEI", "AFEW"])
    U.to_csv(out / "units.csv", index=False)
    noneloss = none_levels(S)
    combos, statements = summarize(U, sel, delta, noneloss)
    combos.to_csv(out / "combos.csv", index=False)
    statements.to_csv(out / "statements.csv", index=False)
    tables(args, out, S, sel, fixed, dirs, U, delta)
    write_prov(out, "judge", args, {"chain": chain, "n_units": len(U), "delta_csv_sha256": sha256(Path(args.out) / "delta" / "delta.csv"),
                                    "chosen_csv_sha256": sha256(Path(args.out) / "select" / "chosen.csv"),
                                    "fixed_csv_sha256": sha256(Path(args.out) / "select" / "fixed.csv")})
    print(f"判定：単位 {len(U)} 行，組み合わせ {len(combos)} 行，文の行 {len(statements)} 行", flush=True)


NOTES = {"oursx_small": "格子は E61 の評価の値を見て決めた（独立した確かめではない）", "abl_meff500": "m_eff ＝ 500 は E61 の評価の値を見て決めた（独立した確かめではない）"}


def none_levels(S):
    """(run, 生徒, K) ごとの none の向きの損の水準（コーパスの平均．IEMOCAP は 2 群の平均）。δ_d の大きさを読むために使う。"""
    out = {}
    for run, P in S.point.items():
        x = P[(P.method == "none") & (P.wave == "e77")].assign(c=lambda d: d.target.map(corp))
        g = x.groupby(["student", "K", "target", "c"]).dirloss.mean().groupby(["student", "K", "c"]).mean().groupby(["student", "K"]).mean()
        for (st, K), v in g.items():
            out[(run, st, K)] = float(v)
    return out


def summarize(U, sel, delta, noneloss):
    """7.2 節の表を組み合わせごとに当てる．感度の行（sensitivity）と MOSEI・AFEW を除いた集計も同じ規則で。"""
    combos, st_rows = [], []
    variants = [(s_, False) for s_ in sorted(U.sensitivity.unique())] + [("main", True), ("delta_half", True)]
    DEL = {(r.run, r.student, r.K): r for r in delta.itertuples(index=False)}
    SELX = {(r.run, r.target, r.student, r.K, r.selector, r.variant): r.chose_none for r in sel.itertuples(index=False)}
    for sens, drop_ma in variants:
        V = U[U.sensitivity == sens]
        if drop_ma:
            V = V[~V.excl_mosei_afew]
        tag = sens + ("+excl_mosei_afew" if drop_ma else "")
        for (fam, comp, run, st, K, metric), g in V.groupby(["family", "comparison", "run", "student", "K", "metric"]):
            cps = {corp(t) for t in groups_for(run)} - ({"MOSEI", "AFEW"} if drop_ma else set())
            n_tot = len(cps)
            maj = majority(n_tot)
            c = g.judgment.value_counts()
            npos, nneg, nsame, nund, nmis = (int(c.get(k, 0)) for k in ("pos", "neg", "same", "undet", "missing"))
            dr = DEL.get((run, st, K))
            row = {"tag": tag, "family": fam, "comparison": comp, "metric": metric, "run": run, "student": st, "K": K, "n_corpora": n_tot,
                   "majority": maj, "n_pos": npos, "n_neg": nneg, "n_same": nsame, "n_undet": nund, "n_missing": nmis,
                   "expected_chance_pos": round(0.025 * (n_tot - nmis), 3),
                   "ci_halfwidth_median": float(((g.hi - g.lo) / 2).median()) if g.hi.notna().any() else np.nan,
                   "delta_d": float(dr.delta_d) if dr is not None else np.nan, "none_dirloss": noneloss.get((run, st, K), np.nan)}
            row["delta_d_ratio_to_none_dirloss"] = row["delta_d"] / row["none_dirloss"] if row["none_dirloss"] else np.nan
            md = g.model_dependent.map(lambda v: v is True or str(v) == "True")
            row["n_model_dependent"] = int(md.sum())
            g2 = g[~md]
            c2 = g2.judgment.value_counts()
            np2, nn2, ns2 = (int(c2.get(k, 0)) for k in ("pos", "neg", "same"))
            row.update({"n_pos_nomd": np2, "n_neg_nomd": nn2, "n_same_nomd": ns2})
            pts = g.point.dropna()
            nz = pts[pts != 0]
            row["sign_p"] = stats.binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue if len(nz) else np.nan
            if fam.startswith("P4") and metric in ("dir", "dir_rev") and not fam.startswith(("P4_desc", "P4_dann_mme")):
                meth = comp.split("-", 1)[1] if comp.startswith(("none-", "coralctl-", "mapctl-")) else comp.split("-")[0]
                base = fam.split("_")[0]
                row["method"] = meth
                row["seen_before_prediction"] = meth in SEEN.get(base, set())
                row.update(rule_p4(g, npos, nneg, nsame, nmis, maj, n_tot, SELX, meth, tag))
                r_nomd = rule_p4(g2, np2, nn2, ns2, int((g2.judgment == "missing").sum()), maj, n_tot, SELX, meth, tag)
                row.update({f"{k}_nomd": v for k, v in r_nomd.items() if k in ("r1", "r2", "r2p", "r3ok", "r4")})
            combos.append(row)
    C = pd.DataFrame(combos)
    if C.empty:
        return C, pd.DataFrame()
    C["sign_p_holm"] = np.nan
    for (tag, fam), g in C.groupby(["tag", "family"]):
        ok = g.sign_p.notna()
        if ok.any():
            p = g.sign_p[ok].to_numpy()
            o = np.argsort(p)
            adj = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o]))
            res = np.empty_like(adj)
            res[o] = adj
            C.loc[g.index[ok], "sign_p_holm"] = res
    for tag, g in C.groupby("tag"):
        st_rows += statements_p4(tag, g, C) + statements_p5(tag, g)
    return C, pd.DataFrame(st_rows)


def rule_p4(g, npos, nneg, nsame, nmis, maj, n_tot, SELX, meth, tag):
    out = {"r1": npos >= maj, "r2": npos >= 1 and nneg >= 1, "r2p": 1 <= npos < maj and nneg == 0,
           "r3ok": npos == 0 and nsame + nneg >= maj, "all_missing": nmis == n_tot}
    out["r4"] = (npos == 0 and not out["r3ok"]) and not out["all_missing"]
    rest = n_tot - npos
    out["r2p_rest_same_or_inc"] = (nsame + nneg) >= majority(rest) if rest > 0 else False
    # (2) の一致と (2′) の選択：「減らした」のコーパスで none を含む選択がその手法を選び，「増やした」のコーパスで none を選んだか．
    # IEMOCAP は即興と台本の両方でそうなったときだけ．MOSEI・MELD の選び直しの感度の行では ns_incl の選択で見る
    if g.empty:
        out.update({"r2_selection_agrees": np.nan, "r2p_selection_chose_method": np.nan})
        return out
    run, st, K = g.run.iloc[0], g.student.iloc[0], g.K.iloc[0]
    agree, chose_all = True, True
    for r in g.itertuples():
        if r.judgment not in ("pos", "neg"):
            continue
        Ts = [t for t in groups_for(run) if corp(t) == r.corpus]
        var = ["ns_incl" if ("noscript" in tag and t in ("MOSEI", "MELD")) else "incl" for t in Ts]
        cn = [str(SELX.get((run, t, st, K, meth, v_), "nan")) for t, v_ in zip(Ts, var)]
        chose = bool(cn) and all(v == "False" for v in cn)
        chose_none = bool(cn) and all(v == "True" for v in cn)
        if (r.judgment == "pos" and not chose) or (r.judgment == "neg" and not chose_none):
            agree = False
        if r.judgment == "pos" and not chose:
            chose_all = False
    out["r2_selection_agrees"] = agree if out["r2"] else np.nan
    out["r2p_selection_chose_method"] = chose_all if out["r2p"] else np.nan
    return out


def half_tag(tag):
    if "+" in tag:
        a, b = tag.split("+", 1)
        return None if "delta_" in a else ("delta_half" if a == "main" else a + ",delta_half") + "+" + b
    return "delta_half" if tag == "main" else (None if "delta_" in tag else tag + ",delta_half")


def p4_hit(g, suffix=""):
    for flag, name in (("r1", "(1)"), ("r2", "(2)"), ("r2p", "(2')"), ("r4", "(4)")):
        col = flag + suffix
        if col in g.columns:
            x = g[g[col].map(lambda v: v is True or str(v) == "True")]
            if len(x):
                return name, x
    return None


def p4_is3(g, suffix=""):
    return g[f"r3ok{suffix}"].map(lambda v: v is True or str(v) == "True").all() and (g[f"n_pos{'_nomd' if suffix else ''}"] == 0).all()


def statements_p4(tag, C, Call):
    """筋の 4 の文の鍵．(3) は δ を半分にした行でも (3) のときだけ（関門 1 の S-1）．模型に依る単位，対照との比較，予言の時点も見る。"""
    rows = []
    KEYS = {"(1)": "「ラベルを付ける発話の選び方では直らない」に狭める．直した向きの量を書く",
            "(2)": "選択との一致を見る（r2_selection_agrees）",
            "(2')": "〈コーパス〉では〈手法〉が向きの損を減らした（r2p_rest_same_or_inc で 2 つの書き方．r2p_selection_chose_method で選択を添える）",
            "(4)": "ラベル無しの手法が向きの損を減らした証拠は無かった（区間の幅と判定できない数を添える）"}
    met = "dir_rev" if "rev_order" in tag else "dir"                        # 逆の順序の感度の行は逆の順序の向きの損で文の鍵を作る（関門 1 の再検査の記録）
    for fam in ("P4", "P4'"):
        for subset in ("all", "unseen_only"):
            g = C[(C.family == fam) & (C.metric == met)]
            if subset == "unseen_only" and len(g):
                g = g[~g.seen_before_prediction.map(lambda v: v is True or str(v) == "True")]
            if g.empty:
                continue
            g = g[~g["all_missing"].map(lambda v: v is True or str(v) == "True")]
            if g.empty:
                continue
            hit = p4_hit(g)
            is3 = hit is None and p4_is3(g)
            notes = []
            ht = half_tag(tag)
            if is3 and ht:                                                     # δ を半分にした行でも (3) か（δ を変えた感度の行そのものでは見ない）
                Ch = Call[(Call.tag == ht) & (Call.family == fam) & (Call.metric == met)] if ht else Call.iloc[:0]
                if subset == "unseen_only" and len(Ch):
                    Ch = Ch[~Ch.seen_before_prediction.map(lambda v: v is True or str(v) == "True")]
                if len(Ch):
                    Ch = Ch[~Ch["all_missing"].map(lambda v: v is True or str(v) == "True")]
                if Ch.empty or p4_hit(Ch) is not None or not p4_is3(Ch):
                    is3 = False
                    notes.append("δ を半分にすると (3) に当たらないので (4) の文にする")
                if is3 and not p4_is3(g, "_nomd"):                               # 模型に依る単位を除くと (3) が成り立たない
                    is3 = False
                    notes.append("模型に依る単位を除くと (3) に当たらないので (4) の文にする")
            ctl = Call[(Call.tag == tag) & (Call.family == fam + "_ctl") & (Call.metric == "dir")]
            ctl_hit = p4_hit(ctl[~ctl["all_missing"].map(lambda v: v is True or str(v) == "True")]) if len(ctl) else None
            if ctl_hit is not None and ctl_hit[0] in ("(1)", "(2)", "(2')"):  # 対照より減らした手法がある（関門 1 の C-2）
                notes.append("〈手法〉の DA の項は同じ形の対照より向きの損を減らした（〈コーパス〉）が，none より減らしたとは言えない："
                             + ";".join(f"{a}|{b}|K{c}" for a, b, c in zip(ctl_hit[1].comparison, ctl_hit[1].student, ctl_hit[1].K)))
                if is3:
                    is3 = False
                    notes.append("対照との比較で減らした単位があるので (3) にせず (4) の文にする")
            if hit is not None:
                name, x = hit
                if p4_hit(x, "_nomd") is None:
                    notes.append("この印は模型に依る単位だけで決まった")
            elif is3:
                name, x = "(3)", g
            else:
                name, x = "(4)", g
            key = "筋の 4 を言い換えずに残す（どのコーパスでも減りは δ_d より小さかった，と δ_d の値と none の向きの損に対する割合を添える）" if name == "(3)" else KEYS[name]
            if fam == "P4'" and name != "(3)":
                key = "(5) 筋の 4 を「単モダリティの特徴の空間では」と限る．" + key
            dd = g.groupby(["run", "student", "K"])[["delta_d", "none_dirloss", "delta_d_ratio_to_none_dirloss"]].first().reset_index()
            rows.append({"tag": tag, "family": fam, "subset": subset, "rule": name, "n_combos": len(x),
                         "combos": ";".join(f"{a}|{b}|K{c}" for a, b, c in zip(x.comparison, x.student, x.K)) if name != "(3)" else "",
                         "statement_key": key, "notes": " / ".join(notes),
                         "delta_d": ";".join(f"{a}|{b}|K{c}:{d:.3g}（none の {e:.1%}）" for a, b, c, d, e in
                                             zip(dd.run, dd.student, dd.K, dd.delta_d, dd.delta_d_ratio_to_none_dirloss))})
            units = int((g.n_corpora - g.n_missing).sum())
            rows.append({"tag": tag, "family": fam, "subset": subset, "rule": "chance", "n_combos": len(g), "combos": "",
                         "statement_key": f"単位 {units}，偶然の「減らした」の見込み {0.025 * units:.2f}，実際の「減らした」{int(g.n_pos.sum())}，"
                                          f"模型に依る単位 {int(g.n_model_dependent.sum())}"})
    return rows


def statements_p5(tag, C):
    rows = []
    P5 = C[C.family.str.match(r"^P5-\d$|^ablation$|^sec5\.3$|^P5_refs$") & C.metric.isin(["uar", "dir"])]
    for (fam, comp, run, st, K, metric), g in P5.groupby(["family", "comparison", "run", "student", "K", "metric"]):
        r = g.iloc[0]
        maj = r.majority
        if fam in ("P5-1", "P5-2", "P5-4"):
            if r.n_same + r.n_neg >= maj:
                key = "相手が oursx と同じ水準か上回った"
            elif r.n_pos >= maj:
                key = "oursx が相手を上回った"
                if fam == "P5-4":                                            # 内訳：Co-teaching − 対照が上回った／下回ったときだけ書く
                    b = C[(C.family == "P5-4_breakdown") & (C.comparison == "cot-cotctl") & (C.run == run) & (C.student == st) & (C.K == K)]
                    if len(b) and b.n_pos.iloc[0] >= maj:
                        key += "．損失の小さい候補を選ぶことが得になった"
                    elif len(b) and b.n_neg.iloc[0] >= maj:
                        key += "．損失の小さい候補を選ぶことが損になった"
            else:
                key = "判定できない（受け皿の行）"
        elif fam == "P5-3":
            if metric != "dir":
                continue
            u = C[(C.family == "P5-3u") & (C.run == run) & (C.student == st) & (C.K == K)]
            if r.n_pos >= maj:
                key = "oursx は none より向きの損を減らした"
            elif len(u) and u.n_pos.iloc[0] >= maj:
                key = "向きの損は減らさず UAR は上回った：筋の 5 を書き直し ours を o の補正として書く"
            else:
                key = "判定できない（受け皿の行）"
        elif fam == "ablation":
            if comp != "oursx-abl_hard":
                continue
            key = "5.3 節の筋で書く" if r.n_same >= maj else "（③ だけは oursx と同じ水準ではない）"
        elif fam == "P5_refs":
            if not comp.startswith("oursx-teacher:") or role_of(run, st) != "fused":
                continue
            key = ("融合の生徒では蒸留の得を主張せず，教師と同じ水準か教師が上回ったと書く" if r.n_same + r.n_neg >= maj
                   else ("oursx が教師を上回った" if r.n_pos >= maj else "判定できない（受け皿の行）"))
        else:
            key = "上回った" if r.n_pos >= maj else "上回っていない"
        rows.append({"tag": tag, "family": fam, "rule": comp, "n_combos": 1, "combos": f"{run}|{st}|K{K}", "statement_key": key})
    # 要旨の文は数え上げの形だけ（関門 1 の S-3）：21 の生徒 × K のうち何個で「上回った」か（世界ごとの内訳）．偶然の見込みも並べる
    for fam in ("P5-1", "P5-2", "P5-3", "P5-4"):
        g = C[(C.family == fam) & C.metric.isin(["uar", "dir"])]
        if g.empty:
            continue
        hit = g.n_pos >= g.majority
        by = g.assign(h=hit).groupby("run").h.agg(["sum", "size"])
        units = int((g.n_corpora - g.n_missing).sum())
        rows.append({"tag": tag, "family": fam, "rule": "summary", "n_combos": int(len(g)), "combos": "",
                     "statement_key": f"{len(g)} の生徒 × K のうち {int(hit.sum())} で上回った（"
                                      + "，".join(f"{a} の {b} のうち {c}" for a, c, b in zip(by.index, by["sum"], by["size"])) + "）．"
                                      f"単位 {units}，偶然の「上回った」の見込み {0.025 * units:.2f}，実際の「上回った」{int(g.n_pos.sum())}．"
                                      "特定の生徒と K だけを取り出した文は書かない"})
    s53 = C[(C.family == "sec5.3") & C.K.isin(["5", "25"])]
    if len(s53):
        per = s53.assign(ok=s53.n_pos >= s53.majority).groupby(["run", "student"]).ok.all()
        rows.append({"tag": tag, "family": "sec5.3", "rule": "7 生徒の過半", "n_combos": int(len(per)),
                     "combos": ";".join(f"{a}|{b}:{bool(v)}" for (a, b), v in per.items()),
                     "statement_key": "「K 本で学び直してから」を書く" if int(per.sum()) >= 4 else "その部分を外す（教師の作り方の寄与を 2 段目の参照に限る）"})
    return rows


def tables(args, out, S, sel, fixed, dirs, U, delta):
    """付録の表：none が選ばれた単位の数，選択の取りこぼし（オラクルは上限としてだけ．種の幅つき），参照の行，種の幅，
    模型の当てはまり（E75 の決まり），DA の項の効き，cotzero の届かなさ，Co-teaching の ṽ，定義から同じになる組，向きの損のオラクル，対照との食い違い。"""
    inc = sel[sel.variant == "incl"]
    nn = inc.assign(chose_none=inc.chose_none.map(lambda v: str(v) == "True")).groupby(["run", "role", "selector", "K"]).chose_none.agg(["sum", "size"])
    nn.reset_index().to_csv(out / "none_chosen.csv", index=False)
    CK = ["wave", "method", "pnorm", "meff"]
    rows, orows = [], []
    for run, Pall in S.point.items():
        P = Pall[Pall.seed <= 0]                                               # 種 0 にそろえる（E78 の記録）
        per = (P.groupby(["target", "student", "K"] + CK + ["rep"]).uar.mean() * 100)
        mean = per.groupby(level=list(range(7))).mean()
        nrep = per.groupby(level=list(range(7))).size()
        dmean = P.groupby(["target", "student", "K"] + CK).dirloss.mean()
        members = {}                                                           # (役, 族) → 候補の集合（ターゲットと K に依らない）
        uniq = P[["student"] + CK].drop_duplicates()
        for role in {role_of(run, st) for st in uniq.student}:
            u = uniq[[role_of(run, st) == role for st in uniq.student]].drop_duplicates(CK)
            for selname, dd in sel_defs(role).items():
                members[(role, selname)] = set(map(tuple, u[cand_mask(u, dd)][CK].to_numpy()))
        for r in sel[(sel.run == run) & (sel.variant.isin(["incl", "excl"]))].itertuples():
            if r.method == EMPTY or pd.isna(r.method) or (r.role, r.selector) not in members:
                continue
            try:
                cm = mean.loc[(r.target, r.student, r.K)]
                cn = nrep.loc[(r.target, r.student, r.K)]
            except KeyError:
                continue
            keep = [tuple(i) in members[(r.role, r.selector)] for i in cm.index]
            cm, cn = cm[keep], cn[keep]
            cm = cm[cn == cn.max()] if len(cn) else cm                          # 評価の全反復に値のある候補だけ（CORAL の r の除外と同じ考え）
            if cm.empty:
                continue
            orc, best_key = cm.max(), cm.idxmax()
            if r.variant == "incl":                                             # none を含む版のオラクルは none も候補
                nonev = mean.get((r.target, r.student, r.K, "e77", "none", "-", "0"), np.nan)
                if nonev >= orc:
                    orc, best_key = nonev, ("e77", "none", "-", "0")
            pr = per.loc[(r.target, r.student, r.K)]
            ch = pr.get((r.wave, r.method, r.param, r.meff))
            bs = pr.get(tuple(best_key))
            if ch is None or bs is None:
                continue
            loss = (bs - ch).dropna()
            sd_seed, n_seed = np.nan, 1
            if r.method in SEEDED:                                              # 種の幅（E78 の I5-2：種 0 の 1 回の差は種の幅と並べて読む）
                xs = S.rows_of(run, r.target, r.student, r.K, r.wave, r.method, r.param, r.meff, "agg")
                v = xs.groupby("seed").uar.mean() * 100
                sd_seed, n_seed = (v.std(ddof=1) if len(v) > 1 else np.nan), int(len(v))
            rows.append({"run": run, "target": r.target, "student": r.student, "K": r.K, "selector": r.selector, "variant": r.variant,
                         "n_cand": int(len(cm)), "oracle_upper": orc, "chosen": ch.mean(), "selection_loss": orc - ch.mean(),
                         "loss_sd_over_reps": loss.std(ddof=1) if len(loss) > 1 else np.nan, "n_reps": int(len(loss)),
                         "chosen_seed_sd": sd_seed, "n_seeds": n_seed})
            # 向きの損で見た格子の最良（オラクル．上限としてだけ書く．関門 1 の C-4）
            if r.variant == "excl" and r.selector in GROUP_B:
                dm = dmean.loc[(r.target, r.student, r.K)]
                dm = dm[[tuple(i) in members[(r.role, r.selector)] for i in dm.index]]
                nd = dmean.get((r.target, r.student, r.K, "e77", "none", "-", "0"), np.nan)
                if len(dm):
                    orows.append({"run": run, "target": r.target, "corpus": corp(r.target), "student": r.student, "K": r.K, "method": r.selector,
                                  "none_minus_best_dirloss": nd - dm.min(), "best": "|".join(map(str, dm.idxmin()))})
    pd.DataFrame(rows).to_csv(out / "selection_loss.csv", index=False)
    if orows:
        O = pd.DataFrame(orows)
        O = O.groupby(["run", "corpus", "student", "K", "method"]).none_minus_best_dirloss.mean().reset_index()
        Dd = delta.set_index(["run", "student", "K"]).delta_d
        O["delta_d"] = [Dd.get((a, b, c), np.nan) for a, b, c in zip(O.run, O.student, O.K)]
        O["at_least_delta_d"] = O.none_minus_best_dirloss >= O.delta_d
        O.to_csv(out / "oracle_dir.csv", index=False)
    R = U[U.family.isin(["refs", "refs_dir"]) & (U.sensitivity == "main")]
    if len(R):
        R.to_csv(out / "refs.csv", index=False)
    seeds = []
    for run, P in S.point.items():
        x = P[P.method.isin(SEEDED) & (P.seed >= 0)]
        if len(x):
            g = x.groupby(["target", "student", "K", "method", "pnorm", "meff", "seed"]).agg(uar=("uar", "mean"), dirloss=("dirloss", "mean")).reset_index()
            s_ = g.groupby(["target", "student", "K", "method", "pnorm", "meff"]).agg(n_seeds=("seed", "nunique"), uar_sd=("uar", "std"),
                                                                                      dir_sd=("dirloss", "std")).reset_index()
            seeds.append(s_[s_.n_seeds > 1].assign(run=run))
    if seeds:
        pd.concat(seeds).to_csv(out / "seeds.csv", index=False)
    # 2 つの山の模型（E75 の決まり：誤差 ÷ 損 < 1/3．誤差は模型が正しいときの標本の揺れの中央値を引いてから測る）
    fit = []
    for run, P in S.point.items():
        nf = Path(args.out) / "measure" / f"model_null_{run}.csv"
        if "e_model_abs" not in P.columns or not nf.exists():
            continue
        nul = pd.read_csv(nf).set_index(["target", "student", "rep"])
        x = P[P.e_model_abs.notna()].copy()
        if x.empty:
            continue
        key = list(zip(x.target, x.student, x.rep))
        x["e_null_median"] = [nul.e_null_median.get(k, np.nan) for k in key]
        x["e_null_95"] = [nul.e_null_95.get(k, np.nan) for k in key]
        x["e_adj"] = (x.e_model_abs - x.e_null_median).clip(lower=0)
        for (st, K), g in x.groupby(["student", "K"]):
            md, mp = g.dirloss.abs().median(), g.posloss.abs().median()
            fit.append({"run": run, "student": st, "K": K, "n_rows": len(g), "err_median": g.e_model_abs.median(),
                        "err_null_median": g.e_null_median.median(), "err_adj_median": g.e_adj.median(),
                        "ratio_dir_raw": g.e_model_abs.median() / md if md else np.nan, "ratio_dir_adj": g.e_adj.median() / md if md else np.nan,
                        "ratio_pos_adj": g.e_adj.median() / mp if mp else np.nan,
                        "holds_dir": (g.e_adj.median() / md < MODEL_RATIO_MAX) if md else False,
                        "holds_pos": (g.e_adj.median() / mp < MODEL_RATIO_MAX) if mp else False,
                        "frac_over_null95": float((g.e_model_abs > g.e_null_95).mean())})
    if fit:
        pd.DataFrame(fit).to_csv(out / "model_fit.csv", index=False)
    vr = []                                                                    # 等分散（記述だけ）
    for run in dirs:
        f = Path(args.out) / "measure" / f"pairs_{run}.csv.gz"
        if f.exists():
            p = pd.read_csv(f)
            p["vr_in"] = p.var_ratio.between(*VR_RANGE)
            vr.append(p.groupby(["target", "student"]).vr_in.mean().reset_index().assign(run=run))
    if vr:
        pd.concat(vr).to_csv(out / "model_var_ratio.csv", index=False)
    da, cv, cz = [], [], []
    for run in dirs:
        f = Path(args.out) / "measure" / f"diag_{run}.csv"
        if not f.exists():
            continue
        dg = pd.read_csv(f, dtype={"K": str})
        for sm in ("dann", "mme"):                                             # DA の項：K ≥ 1 は a > b かつ a ≥ 1%，K = 0 は a ≥ 1%
            a = dg[dg.what == f"{sm}_vs_mapctl_disagree"].groupby(["target", "student", "K"]).value.mean().rename("a_vs_mapctl")
            b = dg[dg.what == f"{sm}_seed_disagree_max"].groupby(["target", "student", "K"]).value.mean().rename("b_seed_max")
            nd = dg[dg.what == f"{sm}_vs_none_disagree"].groupby(["target", "student", "K"]).value.mean().rename("vs_none")
            g = pd.concat([a, b, nd], axis=1).reset_index()
            if g.empty:
                continue
            for col in ("anorm", "dom_first", "dom_last", "ent_first", "ent_last"):
                x = dg[dg.what == f"{sm}_{col}"].set_index(["target", "student", "K"]).value
                if len(x):
                    g[col] = [x.get((t, s_, k), np.nan) for t, s_, k in zip(g.target, g.student, g.K)]
            g["da_changed_predictions"] = [(av >= DISAGREE_MIN) if k == "0" else (av >= DISAGREE_MIN and av > bv)
                                           for av, bv, k in zip(g.a_vs_mapctl, g.b_seed_max, g.K)]
            da.append(g.assign(run=run, method=sm))
        x = dg[dg.what.str.match(r"^cot_(incl|excl)_v_norm$")]
        if len(x):
            cv.append(x.assign(run=run, self_training=x.value >= VNORM_MIN))
        x = dg[dg.what.str.match(r"^cotzero_lr.+_vs_none_disagree$")]
        if len(x):
            cz.append(x.groupby(["target", "student", "K", "what"]).value.agg(["mean", "max", "size"]).reset_index().assign(run=run))
    if da:
        pd.concat(da).to_csv(out / "da_effect.csv", index=False)
    if cv:
        pd.concat(cv).to_csv(out / "cot_vnorm.csv", index=False)
    if cz:
        pd.concat(cz).to_csv(out / "cotzero_reach.csv", index=False)
    # 対照との食い違い（関門 1 の C-2）：none との判定と対照との判定がコーパスで食い違う単位
    M = U[(U.sensitivity == "main") & (U.metric == "dir")]
    a = M[M.family.isin(["P4", "P4'", "P4_uni_world"]) & M.comparison.isin(["none-coral", "none-dann", "none-mme"])].assign(
        method=lambda d: d.comparison.str.split("-").str[1])
    b = M[M.family.str.endswith("_ctl")].assign(method=lambda d: d.comparison.str.split("-").str[1])
    if len(a) and len(b):
        j = a.merge(b, on=["run", "student", "K", "corpus", "method"], suffixes=("_none", "_ctl"))
        j["disagree"] = j.judgment_none != j.judgment_ctl
        j[["run", "student", "K", "corpus", "method", "judgment_none", "judgment_ctl", "point_none", "point_ctl", "disagree"]].to_csv(
            out / "ctl_disagree.csv", index=False)
    idc = []                                                                   # 定義から d が同じになる組（massbal・saerens と none，T3A の補助と主）
    dcols = [f"d{p}" for p in range(6)]
    for run, P in S.point.items():
        base = P[P.method.isin(["none", "massbal", "saerens", "t3a", "t3a_b"]) & (P.wave == "e77")]
        for (T, st, K, rp), g in base.groupby(["target", "student", "K", "rep"]):
            nn_ = g[g.method == "none"]
            for m_ in ("massbal", "saerens"):
                x = g[g.method == m_]
                if len(nn_) and len(x):
                    idc.append({"run": run, "target": T, "student": st, "K": K, "rep": rp, "pair": f"{m_}-none",
                                "d_absdiff_max": float(np.nanmax(np.abs(x[dcols].to_numpy() - nn_[dcols].to_numpy())))})
            a_, b_ = g[g.method == "t3a"].set_index("param"), g[g.method == "t3a_b"].set_index("param")
            common = a_.index.intersection(b_.index)
            if len(common):
                idc.append({"run": run, "target": T, "student": st, "K": K, "rep": rp, "pair": "t3a_b-t3a",
                            "d_absdiff_max": float(np.nanmax(np.abs(a_.loc[common, dcols].to_numpy() - b_.loc[common, dcols].to_numpy())))})
    if idc:
        pd.DataFrame(idc).to_csv(out / "identity_checks.csv", index=False)
    bf = []                                                                    # 切片で取り戻せる分（UAR − 切片だけを直した UAR）が負の行の数（関門 1 の define の D7）
    for run, P in S.point.items():
        if "uar_bfix" in P.columns:
            x = P[P.uar_bfix.notna()]
            if len(x):
                o = 100 * x.uar - x.uar_bfix
                bf.append({"run": run, "n_rows": int(len(x)), "n_negative": int((o > 0).sum()), "n_fit_failed": int((x.bias_fit_ok == 0).sum())})
    if bf:
        pd.DataFrame(bf).to_csv(out / "bias_fixed_negative.csv", index=False)


# ---------------------------------------------------------------- 入口

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("delta", "select", "measure", "judge"))
    ap.add_argument("--runs", nargs="+", default=list(RUNS), choices=RUNS)
    ap.add_argument("--e77", nargs="+", help="名前=場所（既定 results/e77_<名前>）")
    ap.add_argument("--e78", nargs="+", help="名前=場所（既定 results/e78_<名前>）")
    ap.add_argument("--targets", nargs="+", help="試走用．評価のターゲットを限る")
    ap.add_argument("--reps", nargs="+", type=int, default=list(REPS))
    ap.add_argument("--pseudo-reps", nargs="+", type=int, default=list(PSEUDO_REPS))
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "e79")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--no-bias-fixed", dest="bias_fixed", action="store_false", help="切片だけを直した UAR を計算しない（試走用）")
    ap.add_argument("--allow-incomplete", action="store_true", help="試走用．欠けと食い違いを止めずに checks.csv に記録する")
    ap.add_argument("--skip-commit-check", action="store_true", help="試走用．δ・chosen.csv・fixed.csv のコミットの確認を飛ばす（provenance に残す）")
    ap.add_argument("--allow-dirty", action="store_true", help="試走用．exp/ か measure/ に未コミットの変更があっても流す（provenance に残す）")
    args = ap.parse_args()
    args._t0 = time.time()
    args._t0_str = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    dirs = parse_dirs(args)
    roots = [d for pair in dirs.values() for d in pair if d is not None]
    check_dirty(args)
    out = prepare_out(args.out, args.stage)
    install_guard(args.stage, roots)
    if args.stage in ALLOW:
        probe = guard_selftest(args.stage, [str(d) for d in roots])
        print(f"見張り：{args.stage} の段は入力の場所で {ALLOW[args.stage]} だけを開く（{Path(probe).name} を開けないことを確かめた）", flush=True)
    {"delta": stage_delta, "select": stage_select, "measure": stage_measure, "judge": stage_judge}[args.stage](args, dirs, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
