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

試走（揃っていない出力）では --allow-incomplete（欠けを止めずに記録する）と --skip-commit-check（δ のコミットの確認を飛ばし，provenance に残す）を使う。
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""                                         # GPU を使わない（読み込むモジュールが torch を使っても CPU に置く）

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
DISAGREE_MAX = 0.01                                                            # DA の項が予測を変えなかったとみなす，対照との予測の食い違いの割合の上限（計算の前に決めた）
FIT_ACC_MAX, FIT_VR, FIT_DAUC = 2.0, (0.5, 2.0), 0.2                           # 2 つの山の模型が成り立つとみなす基準（計算の前に決めた）
CORAL_FIX_R = 64
CORAL_CP_DESIGN, CORAL_CB_DESIGN = {"0.01", "0.03", "0.1"}, {"2.0", "8.0", "free"}   # 設計の文書の格子（E78 は E39 の C′・C_b を足したが，選ぶ候補は設計の 81 通り）
CT_TAUS_GRID = (0.2, 0.4, 0.6)
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
    """入力の場所の中で ALLOW[stage] に無いファイルを開こうとしたら PermissionError で止める．入力の場所への書き込みは全段で止める。"""
    roots = [os.path.realpath(str(r)) for r in roots]
    pats = [re.compile(p) for p in ALLOW.get(stage, ())]
    read_guard = stage in ALLOW

    def hook(event, args):
        if event != "open":
            return
        p = args[0]
        if not isinstance(p, (str, bytes, os.PathLike)):
            return
        p = os.path.realpath(os.fsdecode(p))
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
    try:
        open(probe, "rb").close()
    except PermissionError:
        return str(probe)
    except FileNotFoundError:
        raise SystemExit(f"見張りが働いていない（{probe} を止めずに開こうとした）")
    raise SystemExit(f"見張りが働いていない（{probe} を開けた）")


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
            "script_sha256": sha256(Path(__file__)), "started": args._t0_str, "seconds": round(time.time() - args._t0, 1), **extra}
    (d / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1, default=str))


def require_committed(path, what, args):
    """`path` の sha256 の先頭 16 文字が，HEAD の docs/ か DAI_paper/ のどれかの追跡しているファイルに書かれているかを確かめる。"""
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
    D["K"] = D.K.astype(str)
    D = D.assign(_k=D.K.map(KS.index)).sort_values(["run", "student", "_k"]).drop(columns="_k")
    D.to_csv(out / "delta.csv", index=False)
    pd.DataFrame(issues).to_csv(out / "checks.csv", index=False)
    h = sha256(out / "delta.csv")
    write_prov(out, "delta", args, {"inputs_sha256": inputs, "n_issues": len(issues), "delta_csv_sha256": h,
                                    "rule": "s = 反復 0〜4 の標準偏差（ddof=1）/ √10．δ = 2 × (run, student, K) ごとの (評価のターゲット, 仮のターゲット) の中央値．単位はポイント"})
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
        "coral": [("e78", lambda m, p, f: m == "coral" and coral_in_design(p))],
        "coralctl": [("e78", lambda m, p, f: m == "coralctl" and coral_in_design(p))],
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


def coral_in_design(p):
    m = re.match(r"^(?:r\d+)?rho[0-9.]+cp(?P<cp>[0-9.e-]+)cb(?P<cb>.+)$", p)
    return bool(m) and m["cp"] in CORAL_CP_DESIGN and m["cb"] in CORAL_CB_DESIGN


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


def stage_select(args, dirs, out):
    dmeta = require_committed(Path(args.out) / "delta" / "delta.csv", "δ", args)
    issues, rows, cand_rows, fixed = [], [], [], []
    for run, (d77, d78) in dirs.items():
        check_run_name(run, d77, "e77", issues)
        if d78 is not None:
            check_run_name(run, d78, "e78", issues)
        else:
            fail_or_note(args, issues, check="e78_missing", run=run)
        P = load_pseudo(run, d77, d78)
        targets = [t for t in (args.targets or groups_for(run)) if t in groups_for(run)]
        for T in targets:
            PT = P[P.eval_target == T] if len(P) else P
            if PT.empty:
                fail_or_note(args, issues, check="pseudo_missing", run=run, target=T)
                continue
            # E77 と E78 の none（仮のターゲット）が同じ値か（I14）
            nn = PT[PT.method == "none"].pivot_table(index=["pseudo", "rep", "student", "K"], columns="wave", values="uar")
            if {"e77", "e78"} <= set(nn.columns):
                dmax = float((nn.e77 - nn.e78).abs().max())
                issues.append({"check": "none_e77_vs_e78_pseudo", "run": run, "target": T, "absdiff_max": dmax})
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
                                continue
                            kk = ["wave", "method", "pnorm", "meff"]
                            sc = C.groupby(kk).uar.agg(score="mean", n="size").reset_index()
                            nu = C.drop_duplicates(kk + ["pseudo", "rep"]).groupby(kk).size().rename("nu").reset_index()
                            sc = sc.merge(nu)
                            mult = sc.method.isin(["cot", "cotctl"]).map({True: len(SEEDS), False: 1})
                            ok = (sc.nu == n_units) & (sc.n == n_units * mult)
                            if sel == "coral":                                    # r = 128 などが 5r の条件で外れた単位：全部の仮のターゲットと評価のターゲットに値のある候補だけ
                                ev = coral_eval_ok(d78, T, st, args.reps)
                                if ev is not None:
                                    for r_, okr in ev.items():
                                        if not okr:
                                            ok &= ~sc.pnorm.str.startswith(f"r{r_}rho")
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
    write_prov(out, "select", args, {"delta": dmeta, "n_issues": len(issues),
                                     "rule": "仮のターゲット × 反復（× 種）の全行の UAR の単純平均の最大．同点は 'method|pnorm|meff' の文字列の順で先頭．none を含む版は none と同点なら none",
                                     "chosen_csv_sha256": sha256(out / "chosen.csv")})
    print(f"選択：{len(R)} 行（うち候補なし {int((R.method == EMPTY).sum())}），固定版 {len(fixed)} 行，記録 {len(issues)} 件", flush=True)


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
        out.append({**base, "fixname": "dann_fix", "wave": "e78", "method": "dann", "param": "lr0.001it1000lam0.1mu1.0", "meff": "-", "seedmode": "s0"})
        out.append({**base, "fixname": "mme_fix", "wave": "e78", "method": "mme", "param": "lr0.001it1000lam0.1mu1.0sw0.0", "meff": "-", "seedmode": "s0"})
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
                    "note": "E39 の (ρ_M, C′, C_b) が CORAL の格子に無ければ測れない"})
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


def key_table(sel, fixed, run, T):
    """ブートストラップで測る鍵（生徒, K, 波, 方法, pnorm, meff, 種の扱い）。"""
    ks = set()
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
    戻り：uar・向きの損・位置の損（B × R．ポイント），空の抽出（B），感情が欠けた抽出（B）。"""
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
        dirs, poss = [], []
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
            dirs.append(ba(Drows[:, p], 0.0)[None, :] - a0)
            poss.append(a0 - ba(dd, oo))
        dirl = np.nanmean(np.stack(dirs, 2), 2)
        posl = np.nanmean(np.stack(poss, 2), 2)
    uar[empty] = np.nan
    dirl[empty] = np.nan
    posl[empty] = np.nan
    return 100 * uar, 100 * dirl, 100 * posl, empty, (~present).any(1) & ~empty


def stage_measure(args, dirs, out):
    dmeta = require_committed(Path(args.out) / "delta" / "delta.csv", "δ", args)
    sel = pd.read_csv(Path(args.out) / "select" / "chosen.csv", dtype={"K": str, "param": str, "meff": str})
    fixed = pd.read_csv(Path(args.out) / "select" / "fixed.csv", dtype={"K": str, "param": str, "meff": str})
    issues, meta = [], []
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
        rows_all, pair_rows, D_rows, keys_all, diag_rows = [], [], [], [], []
        for cp, Ts in by_corpus.items():
            prep = {}
            for T in Ts:
                prep[T] = measure_target(args, run, T, d77, d78, sel, fixed, issues, rows_all, pair_rows, D_rows, diag_rows, c77, c78)
            speakers = sorted(set().union(*[p["speakers"] for p in prep.values()]))
            C, A = boot_draws(run, cp, speakers, args.boot, args.seed, len(args.reps))
            rep_col = {r: i for i, r in enumerate(args.reps)}
            sp_ix = {s: i for i, s in enumerate(speakers)}
            for T, p in prep.items():
                t0 = time.perf_counter()
                nk = len(p["keys"])
                num_u, num_d, num_p = (np.zeros((args.boot, nk)) for _ in range(3))
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
                    u1, d1, _, _, _ = boot_rep(L[:args.chunk].astype(np.float64), y, sidx, np.ones((1, len(ls))), Drows[:args.chunk])
                    e1 = max(float(np.nanmax(np.abs(u1[0] - pt[:args.chunk, 0]))), float(np.nanmax(np.abs(d1[0] - pt[:args.chunk, 1]))))
                    issues.append({"check": "boot_identity", "run": run, "target": T, "rep": r, "absdiff_max": e1})
                    if not e1 <= 1e-6:
                        raise SystemExit(f"ブートストラップの式が点の値に戻らない（{run} {T} 反復 {r}：{e1}）")
                    Ur, Dl, Pl = (np.full((args.boot, L.shape[0]), np.nan) for _ in range(3))
                    for a in range(0, L.shape[0], args.chunk):
                        u_, d_, p_, empty, miss = boot_rep(L[a:a + args.chunk].astype(np.float64), y, sidx, Cb, Drows[a:a + args.chunk])
                        Ur[:, a:a + args.chunk], Dl[:, a:a + args.chunk], Pl[:, a:a + args.chunk] = u_, d_, p_
                    ar = A[:, rep_col[r]]
                    w = ar * (~empty)
                    n_empty += float((ar * empty).sum())
                    n_miss += float((ar * miss).sum())
                    n_draw += float(ar.sum())
                    for j, rows_j in enumerate(krow):                          # 種は複製ごとに平均する（合奏ではない）
                        if not rows_j:
                            continue
                        vu, vd, vp = Ur[:, rows_j].mean(1), Dl[:, rows_j].mean(1), Pl[:, rows_j].mean(1)
                        ok = np.isfinite(vu)
                        num_u[:, j] += np.where(ok, w * vu, 0)
                        num_d[:, j] += np.where(np.isfinite(vd), w * vd, 0)
                        num_p[:, j] += np.where(np.isfinite(vp), w * vp, 0)
                        den[:, j] += np.where(ok, w, 0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    bu, bd, bp = num_u / den, num_d / den, num_p / den
                np.savez_compressed(out / f"boot_{run}_{T}.npz", uar=bu.astype(np.float32), dirloss=bd.astype(np.float32),
                                    posloss=bp.astype(np.float32))
                kt = pd.DataFrame(p["keys"], columns=["student", "K", "wave", "method", "param", "meff", "seedmode"]).assign(
                    key_id=range(nk), run=run, target=T, n_seeds=p["nseeds"])
                keys_all.append(kt)
                meta.append({"run": run, "target": T, "corpus": cp, "n_speakers_corpus": len(speakers), "n_keys": nk, "boot": args.boot,
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
    pd.DataFrame(meta).to_csv(out / "boot_meta.csv", index=False)
    pd.DataFrame(issues).to_csv(out / "checks.csv", index=False)
    write_prov(out, "measure", args, {"delta": dmeta, "boot": args.boot, "seed": args.seed, "n_issues": len(issues),
                                      "select_chosen_sha256": sha256(Path(args.out) / "select" / "chosen.csv")})
    print(f"測定：記録 {len(issues)} 件", flush=True)


def measure_target(args, run, T, d77, d78, sel, fixed, issues, rows_all, pair_rows, D_rows, diag_rows, c77, c78):
    """1 つの評価のターゲット：全行の反復ごとの点の値，D，選んだ行の 6 対の内訳，診断．ブートストラップに渡す読み込みの関数を返す。"""
    t0 = time.perf_counter()
    tab = {"e77": eval_table(d77, T)}
    if d78 is not None:
        tab["e78"] = eval_table(d78, T)
        if tab["e78"] is None:
            fail_or_note(args, issues, check="e78_eval_missing", run=run, target=T)
            tab.pop("e78")
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
    keys = key_table(sel, fixed, run, T)
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
        tr["dirloss"] = 100 * np.nanmean(ba(Dm, 0.0) - a0, 1)
        tr["posloss"] = 100 * np.nanmean(a0 - ba(dmat, omat), 1)
        tr["n_neg_dir_pairs"] = (ba(Dm, 0.0) - a0 < 0).sum(1)
    # 診断：E77 と E78 の none，K=0 の mapctl と none，教師の確率，DA の項と対照の予測の食い違い，Co-teaching の ṽ
    diagnostics(args, run, T, d77, d78, tab, reps, sel, issues, diag_rows)
    speakers = set(spk_all[np.concatenate([te_of[r] for r in reps])]) if reps else set()

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
    return {"keys": keys, "nseeds": nseeds, "reps": reps, "speakers": speakers, "load": load, "students": students}


def diagnostics(args, run, T, d77, d78, tab, reps, sel, issues, diag_rows):
    t77, t78 = tab.get("e77"), tab.get("e78")
    for r in reps:
        L77 = np.load(Path(d77) / f"{T}_r{r}_logp.npy", mmap_mode="r")
        a77 = t77[t77.rep == r]
        if t78 is not None:
            L78 = np.load(Path(d78) / f"{T}_r{r}_logp.npy", mmap_mode="r")
            P78 = np.load(Path(d78) / f"{T}_r{r}_pred.npy", mmap_mode="r")
            a78 = t78[t78.rep == r]
            for st in a77.student.unique():
                for K in KS:
                    n7 = a77[(a77.student == st) & (a77.method == "none") & (a77.K == K)].row_id
                    n8 = a78[(a78.student == st) & (a78.method == "none") & (a78.K == K)].row_id
                    if len(n7) and len(n8):
                        diag_rows.append({"run": run, "target": T, "rep": r, "student": st, "K": K, "what": "none_e77_vs_e78_logp_absmax",
                                          "value": float(np.abs(L77[int(n7.iloc[0])] - L78[int(n8.iloc[0])]).max())})
                    if K == "0" and len(n8):                                     # K=0 の mapctl は none と同じ解のはず
                        mc = a78[(a78.student == st) & (a78.method == "mapctl") & (a78.K == "0")].row_id
                        for rid in mc:
                            diag_rows.append({"run": run, "target": T, "rep": r, "student": st, "K": K, "what": "mapctl_K0_vs_none_logp_absmax",
                                              "value": float(np.abs(L78[int(rid)] - L78[int(n8.iloc[0])]).max())})
                # DA の項：選んだ DANN・MME（種 0）と，同じ lr・反復・μ の対照（種 0）の予測の食い違いの割合
                for sm in ("dann", "mme"):
                    for K in KS:
                        ch = sel[(sel.run == run) & (sel.target == T) & (sel.student == st) & (sel.K == K) & (sel.selector == sm)
                                 & (sel.variant == "excl") & (sel.method == sm)]
                        if ch.empty:
                            continue
                        pn = ch.param.iloc[0]
                        m = re.match(r"lr(?P<lr>[^i]+)it(?P<it>\d+)lam[^m]+mu(?P<mu>[0-9.]+)", pn)
                        rid = a78[(a78.student == st) & (a78.method == sm) & (a78.K == K) & (a78.param == pn + "s0")].row_id
                        cid = a78[(a78.student == st) & (a78.method == "mapctl") & (a78.K == K)
                                  & (a78.param == f"lr{m['lr']}it{m['it']}mu{m['mu']}s0")].row_id if m else []
                        nid = a78[(a78.student == st) & (a78.method == "none") & (a78.K == K)].row_id
                        if len(rid) and len(cid):
                            dis = float((P78[int(rid.iloc[0])] != P78[int(cid.iloc[0])]).mean())
                            diag_rows.append({"run": run, "target": T, "rep": r, "student": st, "K": K, "what": f"{sm}_vs_mapctl_disagree", "value": dis})
                        if len(rid) and len(nid):
                            diag_rows.append({"run": run, "target": T, "rep": r, "student": st, "K": K, "what": f"{sm}_vs_none_disagree",
                                              "value": float((P78[int(rid.iloc[0])] != P78[int(nid.iloc[0])]).mean())})
            f7, f8 = Path(d77) / f"{T}_r{r}_teachers.npz", Path(d78) / f"{T}_r{r}_teachers.npz"
            if f7.exists() and f8.exists():                                   # 第 2 波の Co-teaching に同じ教師を渡したか（評価のターゲットだけ）
                z7, z8 = np.load(f7), np.load(f8)
                for k in sorted(set(z7.files) & set(z8.files)):
                    diag_rows.append({"run": run, "target": T, "rep": r, "student": "*", "K": "-", "what": f"teacher_{k}_absmax",
                                      "value": float(np.abs(z7[k] - z8[k]).max())})
    if d78 is not None and (Path(d78) / f"{T}_diag.csv").exists():             # Co-teaching の ṽ と一致の割合（選んだ設定）
        dg = pd.read_csv(Path(d78) / f"{T}_diag.csv", dtype={"K": str, "param": str, "meff": str})
        if "v_norm" in dg.columns and "target" in dg.columns:
            dg = dg[dg.target.notna()]
            for r_ in sel[(sel.run == run) & (sel.target == T) & (sel.selector == "cot") & (sel.method == "cot")].itertuples():
                x = dg[(dg.student == r_.student) & (dg.method == "cot") & (dg.K == r_.K) & (dg.meff == r_.meff)
                       & dg.param.map(strip_seed).eq(r_.param)]
                for col in ("v", "v_norm", "agree_sel", "agree_drop"):
                    diag_rows.append({"run": run, "target": T, "rep": -1, "student": r_.student, "K": r_.K, "what": f"cot_{r_.variant}_{col}",
                                      "value": float(x[col].mean()) if len(x) else np.nan, "n": int(len(x))})


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


class Store:
    """measure の出力を読み，(run, target, 鍵) → 点の値（反復の平均）と複製を返す。"""

    def __init__(self, mdir, runs):
        self.mdir, self.keys, self.boot, self.point, self.idx = Path(mdir), {}, {}, {}, {}
        for run in runs:
            f = self.mdir / f"keys_{run}.csv"
            if not f.exists():
                continue
            k = pd.read_csv(f, dtype={"K": str, "param": str, "meff": str})
            self.keys[run] = {tuple(r[:8]): int(r[8]) for r in k[["target", "student", "K", "wave", "method", "param", "meff", "seedmode", "key_id"]]
                              .itertuples(index=False)}
            rows = pd.read_csv(self.mdir / f"rows_{run}.csv.gz", dtype={"K": str, "param": str, "meff": str})
            rows["pnorm"] = [strip_seed(p) if m in SEEDED else p for m, p in zip(rows.method, rows.param)]
            rows["seed"] = [seed_of(m, p) for m, p in zip(rows.method, rows.param)]
            self.point[run] = rows
            self.idx[run] = {g: ix for g, ix in rows.groupby(["target", "student", "K", "wave", "method", "meff"]).groups.items()}

    def get(self, run, T, st, K, wave, meth, pn, meff, md):
        j = self.keys.get(run, {}).get((T, st, K, wave, meth, pn, meff, md))
        if j is None:
            return np.nan, np.nan, None, None
        if (run, T) not in self.boot:
            f = self.mdir / f"boot_{run}_{T}.npz"
            z = np.load(f) if f.exists() else None
            self.boot[(run, T)] = None if z is None else {"uar": z["uar"], "dirloss": z["dirloss"]}
        z = self.boot[(run, T)]
        ix = self.idx[run].get((T, st, K, wave, meth, meff))
        x = self.point[run].loc[ix] if ix is not None else self.point[run].iloc[:0]
        x = x[x.pnorm.str.startswith(pn[:-1])] if pn.endswith(":*") else x[x.pnorm == pn]
        if md == "s0":
            x = x[x.seed == 0]
        pu = x.groupby("rep").uar.mean().mean() * 100 if len(x) else np.nan   # 種を反復ごとに平均してから反復で平均
        pdl = x.groupby("rep").dirloss.mean().mean() if len(x) else np.nan
        if z is None:
            return pu, pdl, None, None
        return pu, pdl, z["uar"][:, j], z["dirloss"][:, j]


def stage_judge(args, dirs, out):
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
        return (r.wave, r.method, r.param, r.meff, bool(r.chose_none) if pd.notna(r.chose_none) else False)

    def fixkey(run, T, st, K, name):
        r = FIX.get((run, T, st, K, name))
        if r is None:
            return None
        md = r.seedmode if isinstance(r.seedmode, str) else ("agg" if r.method in SEEDED else "-")
        return (r.wave, r.method, r.param, r.meff, md)

    def val(run, T, st, K, k, s0=False):
        if k is None:
            return np.nan, np.nan, None, None
        wave, meth, pn, meff = k[:4]
        md = k[4] if len(k) > 4 and isinstance(k[4], str) else ("s0" if s0 and meth in SEEDED else ("agg" if meth in SEEDED else "-"))
        if meth == "none":
            md = "-"
        return S.get(run, T, st, K, wave, meth, pn, meff, md)

    NONE = ("e77", "none", "-", "0", "-")

    def dl_of(run, st, K, metric, scale):
        r = DEL.get((run, st, K))
        if r is None:
            return np.nan
        return float(r.delta_d if metric == "dir" else r.delta_u) * scale

    def compare(family, comp, run, st, K, kx, ky, metric, variant, sens, s0=False, iem_mode="both"):
        """値 = metric が dir なら dir(y) − dir(x)（正なら x が向きの損を減らした），uar なら uar(x) − uar(y)。"""
        targets = [t for t in groups_for(run) if (args.targets is None or t in args.targets)]
        byc = {}
        for T in targets:
            if iem_mode == "impro_only" and T == "IEMOCAP-script":
                continue
            byc.setdefault(corp(T), []).append(T)
        for cp, Ts in byc.items():
            pts, reps_ = [], []
            for T in Ts:
                kxx = kx(T) if callable(kx) else kx
                kyy = ky(T) if callable(ky) else ky
                ux, dx, bux, bdx = val(run, T, st, K, kxx, s0)
                uy, dy, buy, bdy = val(run, T, st, K, kyy, s0)
                if metric == "dir":
                    pts.append(dy - dx)
                    reps_.append(None if bdx is None or bdy is None else bdy - bdx)
                else:
                    pts.append(ux - uy)
                    reps_.append(None if bux is None or buy is None else bux - buy)
            x = float(np.mean(pts)) if pts else np.nan
            if any(r_ is None for r_ in reps_) or not reps_:
                lo = hi = np.nan
                nvalid = 0
            else:
                b = np.mean(np.stack(reps_), 0)
                b = b[np.isfinite(b)]
                nvalid = len(b)
                lo, hi = (np.percentile(b, [2.5, 97.5]) if nvalid else (np.nan, np.nan))
            for scale, stag in ((1.0, ""), (0.5, "delta_half"), (2.0, "delta_double")):
                dl = dl_of(run, st, K, metric, scale)
                units.append({"family": family, "comparison": comp, "run": run, "student": st, "role": role_of(run, st), "K": K,
                              "corpus": cp, "n_groups": len(Ts), "point": x, "lo": lo, "hi": hi, "n_boot_valid": nvalid, "delta": dl,
                              "judgment": judge_unit(x, lo, hi, dl), "variant": variant,
                              "sensitivity": ",".join([s for s in (sens, stag) if s]) or "main"})

    for run in dirs:
        world = run in WORLDS
        stus = sorted(set(sel[sel.run == run].student))
        for st in stus:
            role = role_of(run, st)
            for K in KS:
                for v, s0, sens in (("excl", False, ""), ("ns_excl", True, "noscript_reselect")):
                    for meth in GROUP_B + ("t3a_b", "mapctl", "coralctl"):
                        if role == "fused" and meth in ("dann", "mme", "mapctl"):
                            continue
                        fam = ("P4" if role == "kind" else ("P4'" if role == "fused" else "P4_uni_world")) if meth in GROUP_B else "P4_aux"
                        if fam in ("P4", "P4'") and K not in P4_KS:
                            fam += "_otherK"                                   # 主の族は K = 0・5 だけ（設計の文書 7.1 節）．ほかの K は記述
                        kx = (lambda mm, vv: lambda T: pick(run, T, st, K, mm, vv) if not vv.startswith("ns") or T in ("MOSEI", "MELD")
                              else pick(run, T, st, K, mm, "excl"))(meth, v)
                        compare(fam, f"none-{meth}", run, st, K, kx, NONE, "dir", v, sens, s0)
                        for iem in (("impro_only",) if (run in ("text_roberta", "fuse-audio-video-text")) and not sens else ()):
                            compare(fam, f"none-{meth}", run, st, K, kx, NONE, "dir", v, "iemocap_impro_only", s0, iem)
                    if sens:
                        continue
                    for meth in ("dann", "mme"):                               # 手法 − 同じ形の対照，対照 − none（種 0）
                        if role == "fused":
                            continue
                        compare("P4_dann_mme", f"mapctl_match-{meth}", run, st, K, lambda T, mm=meth: pick(run, T, st, K, mm, "excl"),
                                lambda T, mm=meth: fixkey(run, T, st, K, f"mapctl_match_{mm}"), "dir", v, "", True)
                        compare("P4_dann_mme", f"none-mapctl_match_{meth}", run, st, K, lambda T, mm=meth: fixkey(run, T, st, K, f"mapctl_match_{mm}"),
                                NONE, "dir", v, "", True)
                    for fx in ("im_fix", "t3a_fix", "lp_fix", "dann_fix", "mme_fix", "mapctl_fix", "coral_fix", "massbal", "saerens"):
                        if role == "fused" and fx in ("dann_fix", "mme_fix", "mapctl_fix"):
                            continue
                        compare("fixed", f"none-{fx}", run, st, K, lambda T, f_=fx: fixkey(run, T, st, K, f_), NONE, "dir", "fixed", "")
                        compare("fixed_uar", f"{fx}-none", run, st, K, lambda T, f_=fx: fixkey(run, T, st, K, f_), NONE, "uar", "fixed", "")
                if not world or K not in P5_KS:
                    continue
                OX = lambda T, s="oursx", v="incl": pick(run, T, st, K, s, v)  # noqa: E731
                for oxname in ("oursx", "oursx_small"):
                    sens = "" if oxname == "oursx" else "oursx_small"
                    compare("P5-1", "oursx-kd", run, st, K, lambda T, o_=oxname: OX(T, o_), lambda T: pick(run, T, st, K, "kd", "incl"), "uar", "incl", sens)
                    compare("P5-2", "oursx-groupB", run, st, K, lambda T, o_=oxname: OX(T, o_), lambda T: pick(run, T, st, K, "groupB", "incl"), "uar", "incl", sens)
                compare("P5-3", "none-oursx", run, st, K, OX, NONE, "dir", "incl", "")
                compare("P5-3u", "oursx-none", run, st, K, OX, NONE, "uar", "incl", "")
                compare("P5-4", "oursx-cot", run, st, K, OX, lambda T: pick(run, T, st, K, "cot", "incl"), "uar", "incl", "")
                compare("P5-4_breakdown", "cot-cotctl", run, st, K, lambda T: pick(run, T, st, K, "cot", "incl"),
                        lambda T: pick(run, T, st, K, "cotctl", "incl"), "uar", "incl", "")
                compare("P5-4_breakdown", "oursx-cotctl", run, st, K, OX, lambda T: pick(run, T, st, K, "cotctl", "incl"), "uar", "incl", "")
                compare("P5-4_breakdown", "cotzero_match-none", run, st, K, lambda T: fixkey(run, T, st, K, "cotzero_match"), NONE, "uar", "fixed", "")
                compare("P5-4_breakdown_dir", "none-cotzero_match", run, st, K, lambda T: fixkey(run, T, st, K, "cotzero_match"), NONE, "dir", "fixed", "")
                compare("P5-4_breakdown", "cotfix-none", run, st, K, lambda T: fixkey(run, T, st, K, "cotfix"), NONE, "uar", "fixed", "")
                if run == "fuse-audio-video-text":
                    for fam, comp, kx, ky, met in (("P5-1", "oursx-kd", OX, lambda T: pick(run, T, st, K, "kd", "incl"), "uar"),
                                                   ("P5-2", "oursx-groupB", OX, lambda T: pick(run, T, st, K, "groupB", "incl"), "uar"),
                                                   ("P5-3", "none-oursx", OX, NONE, "dir"),
                                                   ("P5-4", "oursx-cot", OX, lambda T: pick(run, T, st, K, "cot", "incl"), "uar")):
                        compare(fam, comp, run, st, K, kx, ky, met, "incl", "iemocap_impro_only", False, "impro_only")
                for fam, comp, kx, ky, met in (("P5-1", "oursx-kd", lambda T: pick(run, T, st, K, "oursx", "ns_incl" if T in ("MOSEI", "MELD") else "incl"),
                                                lambda T: pick(run, T, st, K, "kd", "ns_incl" if T in ("MOSEI", "MELD") else "incl"), "uar"),
                                               ("P5-2", "oursx-groupB", lambda T: pick(run, T, st, K, "oursx", "ns_incl" if T in ("MOSEI", "MELD") else "incl"),
                                                lambda T: pick(run, T, st, K, "groupB", "ns_incl" if T in ("MOSEI", "MELD") else "incl"), "uar")):
                    compare(fam, comp, run, st, K, kx, ky, met, "ns_incl", "noscript_reselect", True)
                abl = ["abl_refa", "abl_refb", "abl_pF"] if role == "uni" else ["abl_ref0", "abl_poe0"]
                for a in abl + ["abl_bal", "abl_raw", "abl_hard", "abl_meff500", "abl_own", "abl_ownK", "ours", "xmuda", "cotrain", "cotrain_prod", "coreg"]:
                    if (a == "xmuda" and role != "uni") or (a == "cotrain_prod" and role != "fused"):
                        continue
                    fam = "ablation" if a.startswith("abl_") or a == "ours" else "groupC_explore"
                    compare(fam, f"oursx-{a}", run, st, K, OX, lambda T, a_=a: pick(run, T, st, K, a_, "incl"), "uar", "incl", "")
                    compare(fam + "_dir", f"{a}-oursx", run, st, K, lambda T, a_=a: pick(run, T, st, K, a_, "incl"), OX, "dir", "incl", "")
                if role == "uni":                                              # 5.3 節：K 本で学び直すことの寄与
                    compare("sec5.3", "abl_refa-abl_pF", run, st, K, lambda T: pick(run, T, st, K, "abl_refa", "incl"),
                            lambda T: pick(run, T, st, K, "abl_pF", "incl"), "uar", "incl", "")
                else:
                    compare("sec5.3", "oursx-abl_ref0", run, st, K, OX, lambda T: pick(run, T, st, K, "abl_ref0", "incl"), "uar", "incl", "")
    U = pd.DataFrame(units)
    U["excl_mosei_afew"] = U.corpus.isin(["MOSEI", "AFEW"])
    U.to_csv(out / "units.csv", index=False)
    combos, statements = summarize(U, sel)
    combos.to_csv(out / "combos.csv", index=False)
    statements.to_csv(out / "statements.csv", index=False)
    tables(args, out, S, sel, fixed, dirs)
    write_prov(out, "judge", args, {"n_units": len(U)})
    print(f"判定：単位 {len(U)} 行，組み合わせ {len(combos)} 行，文の行 {len(statements)} 行", flush=True)


def summarize(U, sel):
    """7.2 節の表を組み合わせごとに当てる．感度の行（sensitivity）と MOSEI・AFEW を除いた集計も同じ規則で。"""
    combos, st_rows = [], []
    variants = [(s, False) for s in sorted(U.sensitivity.unique())] + [("main", True)]
    for sens, drop_ma in variants:
        V = U[U.sensitivity == sens]
        if drop_ma:
            V = V[~V.excl_mosei_afew]
        tag = sens + ("+excl_mosei_afew" if drop_ma else "")
        for (fam, comp, run, st, K), g in V.groupby(["family", "comparison", "run", "student", "K"]):
            cps = {corp(t) for t in groups_for(run)} - ({"MOSEI", "AFEW"} if drop_ma else set())
            n_tot = len(cps)
            maj = majority(n_tot)
            c = g.judgment.value_counts()
            npos, nneg, nsame, nund, nmis = (int(c.get(k, 0)) for k in ("pos", "neg", "same", "undet", "missing"))
            row = {"tag": tag, "family": fam, "comparison": comp, "run": run, "student": st, "K": K, "n_corpora": n_tot, "majority": maj,
                   "n_pos": npos, "n_neg": nneg, "n_same": nsame, "n_undet": nund, "n_missing": nmis,
                   "expected_chance_pos": round(0.025 * (n_tot - nmis), 3),
                   "ci_halfwidth_median": float(((g.hi - g.lo) / 2).median()) if g.hi.notna().any() else np.nan}
            pts = g.point.dropna()
            nz = pts[pts != 0]
            row["sign_p"] = stats.binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue if len(nz) else np.nan
            if fam in ("P4", "P4'", "P4_uni_world", "P4_aux"):
                row.update(rule_p4(g, npos, nneg, nsame, nund, nmis, maj, n_tot, sel))
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
        st_rows += statements_p4(tag, g) + statements_p5(tag, g)
    return C, pd.DataFrame(st_rows)


def rule_p4(g, npos, nneg, nsame, nund, nmis, maj, n_tot, sel):
    out = {"r1": npos >= maj, "r2": npos >= 1 and nneg >= 1, "r2p": 1 <= npos < maj and nneg == 0,
           "r3ok": npos == 0 and nsame + nneg >= maj, "all_missing": nmis == n_tot}
    out["r4"] = (npos == 0 and not out["r3ok"]) and not out["all_missing"]
    rest = n_tot - npos
    out["r2p_rest_same_or_inc"] = (nsame + nneg) >= majority(rest) if rest > 0 else False
    # (2) の一致：「減らした」のコーパスで none を含む選択がその手法を選び，「増やした」のコーパスで none を選んだか
    meth = g.comparison.iloc[0].split("-", 1)[1]
    run, st, K = g.run.iloc[0], g.student.iloc[0], g.K.iloc[0]
    agree = True
    for r in g.itertuples():
        if r.judgment not in ("pos", "neg"):
            continue
        Ts = [t for t in groups_for(run) if corp(t) == r.corpus]
        s = sel[(sel.run == run) & (sel.student == st) & (sel.K == K) & (sel.selector == meth) & (sel.variant == "incl") & sel.target.isin(Ts)]
        chose = s.chose_none.map(lambda v: str(v) == "False").all() if len(s) else False
        chose_none = s.chose_none.map(lambda v: str(v) == "True").all() if len(s) else False
        if (r.judgment == "pos" and not chose) or (r.judgment == "neg" and not chose_none):
            agree = False
    out["r2_selection_agrees"] = agree if out["r2"] else np.nan
    return out


def statements_p4(tag, C):
    rows = []
    for fam in ("P4", "P4'"):
        g = C[(C.family == fam)]
        if g.empty:
            continue
        g = g[~g["all_missing"].fillna(False).astype(bool)]
        if g.empty:
            continue
        hit = None
        for flag, name in (("r1", "(1)"), ("r2", "(2)"), ("r2p", "(2')"), ("r4", "(4)")):
            x = g[g[flag].fillna(False).astype(bool)]
            if len(x):
                hit = (name, x)
                break
        if hit is None and g.r3ok.fillna(False).astype(bool).all() and (g.n_pos == 0).all():
            rows.append({"tag": tag, "family": fam, "rule": "(3)", "n_combos": len(g), "combos": "",
                         "statement_key": "筋の 4 を言い換えずに残す（多重性は筋の 4 に厳しい側）"})
        elif hit is not None:
            name, x = hit
            key = {"(1)": "「ラベルを付ける発話の選び方では直らない」に狭める．直した向きの量を書く",
                   "(2)": "選択との一致を見る（r2_selection_agrees）",
                   "(2')": "〈コーパス〉では〈手法〉が向きの損を減らした（r2p_rest_same_or_inc で 2 つの書き方）",
                   "(4)": "ラベル無しの手法が向きの損を減らした証拠は無かった（区間の幅と判定できない数を添える）"}[name]
            if fam == "P4'":
                key = "(5) 筋の 4 を「単モダリティの特徴の空間では」と限る．" + key
            rows.append({"tag": tag, "family": fam, "rule": name, "n_combos": len(x),
                         "combos": ";".join(f"{a}|{b}|K{c}" for a, b, c in zip(x.comparison, x.student, x.K)), "statement_key": key})
        units = int((g.n_corpora - g.n_missing).sum())
        rows.append({"tag": tag, "family": fam, "rule": "chance", "n_combos": len(g), "combos": "",
                     "statement_key": f"単位 {units}，偶然の「減らした」の見込み {0.025 * units:.2f}，実際の「減らした」{int(g.n_pos.sum())}"})
    return rows


def statements_p5(tag, C):
    rows = []
    for (fam, comp, run, st, K), g in C[C.family.str.match(r"^P5-\d$|^ablation$|^sec5\.3$")].groupby(["family", "comparison", "run", "student", "K"]):
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
        else:
            key = "上回った" if r.n_pos >= maj else "上回っていない"
        rows.append({"tag": tag, "family": fam, "rule": comp, "n_combos": 1, "combos": f"{run}|{st}|K{K}", "statement_key": key})
    s53 = C[(C.family == "sec5.3") & C.K.isin(["5", "25"])]
    if len(s53):
        per = s53.assign(ok=s53.n_pos >= s53.majority).groupby(["run", "student"]).ok.all()
        rows.append({"tag": tag, "family": "sec5.3", "rule": "7 生徒の過半", "n_combos": int(len(per)),
                     "combos": ";".join(f"{a}|{b}:{bool(v)}" for (a, b), v in per.items()),
                     "statement_key": "「K 本で学び直してから」を書く" if int(per.sum()) >= 4 else "その部分を外す（教師の作り方の寄与を 2 段目の参照に限る）"})
    return rows


def tables(args, out, S, sel, fixed, dirs):
    """付録の表：none が選ばれた単位の数，選択の取りこぼし（オラクルは上限としてだけ），候補の数，参照の行，種の幅，模型の当てはまり。"""
    inc = sel[sel.variant == "incl"]
    nn = inc.assign(chose_none=inc.chose_none.map(lambda v: str(v) == "True")).groupby(["run", "role", "selector", "K"]).chose_none.agg(["sum", "size"])
    nn.reset_index().to_csv(out / "none_chosen.csv", index=False)
    rows = []
    CK = ["wave", "method", "pnorm", "meff"]
    for run, P in S.point.items():
        P = P[P.seed <= 0]                                                     # 種 0 にそろえる（E78 の記録）
        per = (P.groupby(["target", "student", "K"] + CK + ["rep"]).uar.mean() * 100)
        mean = per.groupby(level=list(range(7))).mean()
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
            except KeyError:
                continue
            cm = cm[[tuple(i) in members[(r.role, r.selector)] for i in cm.index]]
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
            rows.append({"run": run, "target": r.target, "student": r.student, "K": r.K, "selector": r.selector, "variant": r.variant,
                         "n_cand": int(len(cm)), "oracle_upper": orc, "chosen": ch.mean(), "selection_loss": orc - ch.mean(),
                         "loss_sd_over_reps": loss.std(ddof=1) if len(loss) > 1 else np.nan, "n_reps": int(len(loss))})
    pd.DataFrame(rows).to_csv(out / "selection_loss.csv", index=False)
    refs = []
    for run, P in S.point.items():
        x = P[P.method.str.startswith("ref_") | (P.method == "teacher")]
        if len(x):
            refs.append(x.groupby(["target", "student", "method", "param", "K"]).agg(uar=("uar", "mean"), dirloss=("dirloss", "mean"),
                                                                                      n_reps=("rep", "nunique")).reset_index().assign(run=run))
    if refs:
        pd.concat(refs).to_csv(out / "refs.csv", index=False)
    seeds = []
    for run, P in S.point.items():
        x = P[P.method.isin(SEEDED) & (P.seed >= 0)]
        if len(x):
            g = x.groupby(["target", "student", "K", "method", "pnorm", "meff", "seed"]).agg(uar=("uar", "mean"), dirloss=("dirloss", "mean")).reset_index()
            s = g.groupby(["target", "student", "K", "method", "pnorm", "meff"]).agg(n_seeds=("seed", "nunique"), uar_sd=("uar", "std"),
                                                                                     dir_sd=("dirloss", "std")).reset_index()
            seeds.append(s[s.n_seeds > 1].assign(run=run))
    if seeds:
        pd.concat(seeds).to_csv(out / "seeds.csv", index=False)
    fit = []
    for run in dirs:
        f = Path(args.out) / "measure" / f"pairs_{run}.csv.gz"
        if not f.exists():
            continue
        p = pd.read_csv(f)
        p["acc_err"] = (p.acc_model - p.acc_actual).abs()
        p["vr_out"] = ~p.var_ratio.between(*FIT_VR)
        p["dauc_rel"] = (p.d - p.d_auc).abs() / p.d_auc.abs()
        p["neg_dir"] = ba(p.D, 0.0) < ba(p.d, 0.0)
        p = p.drop_duplicates(["target", "rep", "wave", "row_id", "pair"])     # 同じ行が複数の鍵に入るので行ごとに 1 回
        g = p.groupby(["target", "wave", "row_id", "rep"]).agg(acc_err=("acc_err", "mean"), vr_in=("vr_out", lambda v: 1 - v.mean()),
                                                               dauc_rel=("dauc_rel", "mean"), neg_dir=("neg_dir", "sum")).reset_index()
        g["fits"] = (g.acc_err <= FIT_ACC_MAX) & (g.vr_in >= 4 / 6) & (g.dauc_rel <= FIT_DAUC)
        fit.append(g.assign(run=run))
    if fit:
        pd.concat(fit).to_csv(out / "model_fit.csv", index=False)
    da, cv = [], []
    for run in dirs:
        f = Path(args.out) / "measure" / f"diag_{run}.csv"
        if not f.exists():
            continue
        dg = pd.read_csv(f, dtype={"K": str})
        x = dg[dg.what.str.endswith("_vs_mapctl_disagree")]
        if len(x):
            g = x.groupby(["target", "student", "K", "what"]).value.agg(["mean", "size"]).reset_index()
            g["da_term_no_effect"] = g["mean"] < DISAGREE_MAX
            da.append(g.assign(run=run))
        x = dg[dg.what.str.match(r"^cot_(incl|excl)_v_norm$")]
        if len(x):
            cv.append(x.assign(run=run, self_training=x.value >= VNORM_MIN))
    if da:
        pd.concat(da).to_csv(out / "da_effect.csv", index=False)
    idc = []                                                                   # 定義から d が同じになる組（massbal・saerens と none，T3A の補助と主）
    dcols = [f"d{p}" for p in range(6)]
    for run, P in S.point.items():
        base = P[P.method.isin(["none", "massbal", "saerens", "t3a", "t3a_b"]) & (P.wave == "e77")]
        for (T, st, K, rp), g in base.groupby(["target", "student", "K", "rep"]):
            nn = g[g.method == "none"]
            for m_ in ("massbal", "saerens"):
                x = g[g.method == m_]
                if len(nn) and len(x):
                    idc.append({"run": run, "target": T, "student": st, "K": K, "rep": rp, "pair": f"{m_}-none",
                                "d_absdiff_max": float(np.nanmax(np.abs(x[dcols].to_numpy() - nn[dcols].to_numpy())))})
            a, b = g[g.method == "t3a"].set_index("param"), g[g.method == "t3a_b"].set_index("param")
            common = a.index.intersection(b.index)
            if len(common):
                idc.append({"run": run, "target": T, "student": st, "K": K, "rep": rp, "pair": "t3a_b-t3a",
                            "d_absdiff_max": float(np.nanmax(np.abs(a.loc[common, dcols].to_numpy() - b.loc[common, dcols].to_numpy())))})
    if idc:
        pd.DataFrame(idc).to_csv(out / "identity_checks.csv", index=False)
    if cv:
        pd.concat(cv).to_csv(out / "cot_vnorm.csv", index=False)


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
    ap.add_argument("--skip-commit-check", action="store_true", help="試走用．δ のコミットの確認を飛ばす（provenance に残す）")
    args = ap.parse_args()
    args._t0 = time.time()
    args._t0_str = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    dirs = parse_dirs(args)
    roots = [d for pair in dirs.values() for d in pair if d is not None]
    out = prepare_out(args.out, args.stage)
    install_guard(args.stage, roots)
    if args.stage in ALLOW:
        probe = guard_selftest(args.stage, [str(d) for d in roots])
        print(f"見張り：{args.stage} の段は入力の場所で {ALLOW[args.stage]} だけを開く（{Path(probe).name} を開けないことを確かめた）", flush=True)
    {"delta": stage_delta, "select": stage_select, "measure": stage_measure, "judge": stage_judge}[args.stage](args, dirs, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
