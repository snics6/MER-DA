"""E85：配備する分類器は，疑似ラベルを「自分の特徴から予測できる分」だけ学ぶか．表せる分の上限と，手法が受け取った得を並べる（仮のターゲットだけ）．

設計：docs/exploratory-analyses.md の E85（規則は値を見る前に書いた）．下調べ（仮のターゲットの候補とテスト．評価のターゲットのラベルは使わない）．

入力（保存済み）
  - E80 の単位ファイル results/e80/<world>/units/<fused>/<T>/<c>_r<r>.npz：候補の行番号 cand，テストの行番号 te，ラベル y・yte，
    候補の上の各モデルの確率（P__<kind>：ソースだけの出発点，ref__K・ref0__K：学び直して掛けたモデル，poe0，none__<f>__K：K 本で学び直した配備する分類器，
    none_te__<f>__K：同じくテストの上）
  - E77 の仮のターゲットの成績 results/e77_<fused>/<T>_pseudo.csv：行 (pseudo=c, rep, student=f, method, param, meff, K, uar)．
    method=none（出発点＋K 本），teacher（疑似ラベルのモデルそのもの．param=pF・pFK・poe0・ref・ref0），dist（手法．param="<g>:<重み>"，meff）
  - 特徴：measure/data.load_any(c, <fused>) の行を cand・te で引き，配備する分類器のモダリティのブロックだけを使う
新しい学習：候補の上で，疑似ラベル q を自分の特徴 z だけから当てる線形 softmax（柔らかい交差エントロピー，弱い L2）．GPU（torch，L-BFGS）．

量（テストの上）：UAR_anchor（none_te），UAR_g（teacher の行），UAR_proj（当てはめた線形 softmax），UAR_method（dist の行）．
  J = UAR_g − UAR_anchor，表せる分 r_rep = (UAR_proj − UAR_anchor)/J，受け取った分 r_m = (UAR_method − UAR_anchor)/J．
候補の 2 分割（話者）でも同じ量を出す（teacher の無い g にも使える）．対ごとの a（2 感情の正解率．切片を最良に直した値）も出す．

使い方：./.venv/bin/python exp/analyze_projection_e85.py [--out results/e85] [--worlds avt av] [--lam 1e-3] [--smoke]
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
from data import GROUP_SPEC, fused_parts, load_any, load_group  # noqa: E402
from explore_pairs import PAIR_NAMES, PAIRS, ba  # noqa: E402
from split import speaker_folds  # noqa: E402

WORLDS = {"avt": "fuse-audio-video-text", "av": "fuse-audio-video"}
KS = (0, 5, 25)
G_OF_K = {0: ("pF", "ref"), 5: ("pF", "pFK", "ref"), 25: ("pF", "pFK", "ref")}   # teacher と dist（手法）の行がある g（poe0 は dist の行が無いので外す）
WEIGHTS = ("raw", "bal")
MEFFS = ("n", "500", "100", "2000")
TOL = 1e-6
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def corp(g):
    return GROUP_SPEC[g][0]


def uar(pred, y):
    return 100 * float(np.mean([(pred[y == c] == c).mean() for c in range(4) if (y == c).any()]))


def g_key(g, K, fused):
    """単位ファイルの中の，疑似ラベルのモデル g の候補の上の確率の鍵．"""
    return {"pF": f"P__{fused}", "pFK": f"none__{fused}__{K}", "ref": f"ref__{K}", "ref0": f"ref0__{K}", "poe0": "poe0"}[g]


def fit_soft(Z, Qs, lam, Zte_list):
    """柔らかい交差エントロピーの線形 softmax を，同じ特徴 Z の上の P 個の問（疑似ラベル Qs[j]）についてまとめて解く．
    問 j：min (1/n) Σ_i −q_ijᵀ log softmax(W_j z_i + b_j) + lam/2 ‖W_j‖²．損失は問ごとに分かれているので，和を 1 つの L-BFGS で解いても各問の最適解は同じ．
    戻り値：outs[j][k] = 問 j の分類器を Zte_list[k] に当てた確率．GPU では float32（float64 は A5000 で約 30 倍遅い）．"""
    DT = torch.float32 if DEV.type == "cuda" else torch.float64
    P = len(Qs)
    Zt = torch.tensor(Z, dtype=DT, device=DEV)
    Qt = torch.tensor(np.stack(Qs, 1), dtype=DT, device=DEV)          # n × P × 4
    d = Z.shape[1]
    W = torch.zeros(d, 4 * P, dtype=DT, device=DEV, requires_grad=True)
    b = torch.zeros(4 * P, dtype=DT, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=400, tolerance_grad=1e-5, tolerance_change=1e-9, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logp = torch.log_softmax((Zt @ W + b).view(-1, P, 4), 2)
        loss = -(Qt * logp).sum(2).mean(0).sum() + lam / 2 * (W * W).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        outs = []
        for j in range(P):
            Wj, bj = W[:, 4 * j:4 * j + 4], b[4 * j:4 * j + 4]
            outs.append([torch.softmax(torch.tensor(Zte, dtype=DT, device=DEV) @ Wj + bj, 1).double().cpu().numpy() for Zte in Zte_list])
    return outs


def pair_a(P, y):
    """確率 P（n×4）とラベルから，対ごとの a = 100·Φ(d/2)（2 つの山の模型．切片を最良に直した 2 感情の正解率）．"""
    L = np.log(np.clip(P, 1e-12, 1))
    out = []
    for p, (k, j) in enumerate(PAIRS):
        zk, zj = L[y == k, k] - L[y == k, j], L[y == j, k] - L[y == j, j]
        if len(zk) < 2 or len(zj) < 2:
            out.append(np.nan)
            continue
        sd = np.sqrt((zk.var(ddof=1) + zj.var(ddof=1)) / 2)
        out.append(100 * ba((zk.mean() - zj.mean()) / sd, 0) if sd > 0 else np.nan)
    return out


class Unit:
    def __init__(self, world, T, c, r, feats):
        fused = WORLDS[world]
        self.world, self.T, self.c, self.r, self.fused = world, T, c, r, fused
        self.z = np.load(ROOT / "results" / "e80" / world / "units" / fused / T / f"{c}_r{r}.npz")
        self.cand, self.te = self.z["cand"], self.z["te"]
        self.y, self.yte = self.z["y"], self.z["yte"]
        df, X, blocks = feats
        if len(self.cand) != len(self.y) or int(max(self.cand.max(), self.te.max())) >= len(df):
            raise SystemExit(f"{world} {T} {c} r{r}：候補・テストの行番号が発話一覧と合わない")
        if not np.array_equal(df.label_4.to_numpy()[self.cand], self.y) or not np.array_equal(df.label_4.to_numpy()[self.te], self.yte):
            raise SystemExit(f"{world} {T} {c} r{r}：ラベルが発話一覧と合わない")
        self.X, self.blocks, self.df = X, blocks, df
        self.pseudo = pd.read_csv(ROOT / "results" / f"e77_{fused}" / f"{T}_pseudo.csv", comment="#")
        self.pseudo = self.pseudo[(self.pseudo.pseudo == c) & (self.pseudo.rep == r)]

    def feat(self, f, rows):
        lo, hi = self.blocks[f]
        return self.X[rows, lo:hi]

    def ps(self, f, K, method, param, meff=None):
        q = self.pseudo[(self.pseudo.student == f) & (self.pseudo.K.astype(str) == str(K)) & (self.pseudo.method == method) & (self.pseudo.param.astype(str) == param)]
        if meff is not None:
            q = q[q.meff.astype(str) == str(meff)]
        if len(q) != 1:
            return np.nan
        return 100 * float(q.uar.iloc[0])       # pseudo.csv の uar は割合（0〜1）．ポイントにそろえる


def feats_for(c, fused):
    df, X = load_any(c, fused)
    mods, kinds = fused_parts(fused)
    dims = [load_group(c, kd)[1].shape[1] for kd in kinds]
    blocks, lo = {}, 0
    for kd, dm in zip(kinds, dims):
        blocks[kd] = (lo, lo + dm)
        lo += dm
    blocks[fused] = (0, lo)
    if lo != X.shape[1]:
        raise SystemExit(f"{c} {fused}：ブロックの次元の和 {lo} が特徴 {X.shape[1]} と合わない")
    return df, X, blocks


def run_unit(u, singles, lams, rows_u, rows_p, chk):
    fused = u.fused
    fs = tuple(singles) + (fused,)
    sp = pd.read_csv(ROOT / "results" / "e80" / u.world / "units" / fused / f"samples_{u.c}.csv")
    if not np.array_equal(sp.sample_id.to_numpy(), u.df.sample_id.to_numpy()):
        raise SystemExit(f"{u.world} {u.c}：samples の sample_id が発話一覧と合わない")
    grp = sp.speaker_group.iloc[u.cand].reset_index(drop=True)
    fold = speaker_folds(grp, k=2, seed=0).to_numpy()
    for f in singles:                       # 配備する分類器は単モダリティだけ（融合の f に融合の g を当てるのは自明なので外す）
        Zc, Zte = u.feat(f, u.cand), u.feat(f, u.te)
        mu, sd = Zc.mean(0), Zc.std(0) + 1e-8
        Zc, Zte = (Zc - mu) / sd, (Zte - mu) / sd
        # 問の一覧（K, g, λ）を作り，同じ λ の問はまとめて 1 回で解く
        probs = []
        for K in KS:
            anc_te = u.z[f"none_te__{f}__{K}"]
            u_anchor_te = uar(anc_te.argmax(1), u.yte)
            ps_none = u.ps(f, K, "none", "-")
            if np.isfinite(ps_none) and abs(ps_none - u_anchor_te) > 1e-3:
                chk.append(dict(world=u.world, T=u.T, c=u.c, r=u.r, f=f, K=K, what="none_te vs pseudo.csv", diff=ps_none - u_anchor_te))
            for g in G_OF_K[K]:
                key = g_key(g, K, fused)
                if key not in u.z.files:
                    continue
                Q = u.z[key].astype(np.float64)
                Q = Q / Q.sum(1, keepdims=True)
                for lam in lams:
                    if lam != lams[0] and K != 0:
                        continue          # λ の感度は K = 0 だけ
                    probs.append(dict(K=K, g=g, lam=lam, Q=Q, u_anchor_te=u_anchor_te, anc_te=anc_te))
        for lam in lams:
            group = [p for p in probs if p["lam"] == lam]
            if not group:
                continue
            outs = fit_soft(Zc, [p["Q"] for p in group], lam, [Zte])
            for p, o in zip(group, outs):
                p["P_te"] = o[0]
        # 候補の 2 分割（K = 0，主の λ だけ）：fold 0 で当てはめて fold 1 で測る（とその逆）．g ごとの問をまとめて解く
        g0 = [p for p in probs if p["K"] == 0 and p["lam"] == lams[0]]
        for p in g0:
            p["cand_outs"] = []
        for a_, b_ in ((0, 1), (1, 0)):
            outs = fit_soft(Zc[fold == a_], [p["Q"][fold == a_] for p in g0], lams[0], [Zc[fold == b_]])
            anc_c = u.z[f"none__{f}__0"]
            for p, o in zip(g0, outs):
                Pb = o[0]
                p["cand_outs"].append((uar(Pb.argmax(1), u.y[fold == b_]), uar(p["Q"][fold == b_].argmax(1), u.y[fold == b_]), uar(anc_c[fold == b_].argmax(1), u.y[fold == b_])))
        for p in probs:
            K, g, lam, Q = p["K"], p["g"], p["lam"], p["Q"]
            u_anchor_te = p["u_anchor_te"]
            u_g_te = u.ps(fused, K, "teacher", g)        # teacher の行は融合の student の下にだけある（g のテストの UAR は f に依らない）
            u_proj_te = uar(p["P_te"].argmax(1), u.yte)
            if "cand_outs" in p:
                co = p["cand_outs"]
                u_proj_cand, u_g_cand_b, u_anchor_cand_b = (float(np.mean([o[i] for o in co])) for i in range(3))
            else:
                u_proj_cand = u_g_cand_b = u_anchor_cand_b = np.nan
            row = dict(world=u.world, T=u.T, c=u.c, unit=corp(u.c), r=u.r, f=f, K=K, g=g, lam=lam, n_cand=len(u.cand), n_te=len(u.te),
                       uar_anchor_te=u_anchor_te, uar_g_te=u_g_te, uar_proj_te=u_proj_te,
                       J_te=u_g_te - u_anchor_te, gain_proj_te=u_proj_te - u_anchor_te,
                       uar_anchor_cand=u_anchor_cand_b, uar_g_cand=u_g_cand_b, uar_proj_cand=u_proj_cand,
                       J_cand=u_g_cand_b - u_anchor_cand_b, gain_proj_cand=u_proj_cand - u_anchor_cand_b, uar_g_cand_all=uar(Q.argmax(1), u.y))
            for w in WEIGHTS:
                for m in MEFFS:
                    um = u.ps(f, K, "dist", f"{g}:{w}", m)
                    row[f"uar_method_{w}_{m}"] = um
                    row[f"gain_method_{w}_{m}"] = um - u_anchor_te
            rows_u.append(row)
            # 対ごと（テスト）：g の確率はテストに無いので pF・pFK だけ none_te の融合で代用．proj と anchor は全 g で
            a_proj = pair_a(p["P_te"], u.yte)
            a_anchor_te = pair_a(p["anc_te"], u.yte)
            a_g = pair_a(u.z[f"none_te__{fused}__{K}"], u.yte) if g in ("pF", "pFK") else [np.nan] * len(PAIRS)
            for pi, name in enumerate(PAIR_NAMES):
                rows_p.append(dict(world=u.world, T=u.T, c=u.c, unit=corp(u.c), r=u.r, f=f, K=K, g=g, lam=lam, pair=name,
                                   a_anchor=a_anchor_te[pi], a_g=a_g[pi], a_proj=a_proj[pi]))


def summarize(U, lam_main):
    u = U[U.lam == lam_main].copy()
    u["r_rep_te"] = u.gain_proj_te / u.J_te
    u["r_m_te"] = u["gain_method_raw_n"] / u.J_te
    u["r_m_best_te"] = u[[c for c in u.columns if c.startswith("gain_method_")]].max(1) / u.J_te
    u["r_rep_cand"] = u.gain_proj_cand / u.J_cand
    # 単位（c のコーパス）ごとに反復と T を平均してから，kind・K・g ごとに中央値
    per = u.groupby(["world", "f", "K", "g", "unit"]).agg(J_te=("J_te", "mean"), gain_proj_te=("gain_proj_te", "mean"), gain_m=("gain_method_raw_n", "mean"),
                                                         r_rep_te=("r_rep_te", "mean"), r_m_te=("r_m_te", "mean"), r_m_best_te=("r_m_best_te", "mean"),
                                                         J_cand=("J_cand", "mean"), r_rep_cand=("r_rep_cand", "mean"), n=("r", "size")).reset_index()
    out = []
    for key, g in per.groupby(["world", "f", "K", "g"]):
        pos = g[g.J_te > 0]       # J が正の単位だけで割合を読む（J ≤ 0 では割合に意味が無い）
        out.append(dict(world=key[0], f=key[1], K=key[2], g=key[3], n_units=len(g), n_units_Jpos=len(pos),
                        J_te_median=g.J_te.median(), gain_proj_te_median=g.gain_proj_te.median(), gain_m_median=g.gain_m.median(),
                        r_rep_te_median=pos.r_rep_te.median() if len(pos) else np.nan, r_m_te_median=pos.r_m_te.median() if len(pos) else np.nan,
                        r_m_best_te_median=pos.r_m_best_te.median() if len(pos) else np.nan,
                        n_units_proj_ge_method=int((g.gain_proj_te >= g.gain_m).sum()),
                        J_cand_median=g.J_cand.median(), r_rep_cand_median=g[g.J_cand > 0].r_rep_cand.median() if (g.J_cand > 0).any() else np.nan))
    S = pd.DataFrame(out)
    # 判定（値を見る前に書いた規則．E85 4 節）
    S["no_loss_in_learning"] = S.r_m_te_median >= 0.8 * S.r_rep_te_median      # 手法は表せる分をほぼ受け取っている
    S["unrepresentable_dominates"] = S.r_rep_te_median < 0.6                      # 表せない分が主因
    return per, S


def write(df, path, header):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# {header}\n")
        df.to_csv(fh, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/e85")
    ap.add_argument("--worlds", nargs="*", default=list(WORLDS))
    ap.add_argument("--lams", nargs="*", type=float, default=[1e-3, 1e-1])
    ap.add_argument("--lam-main", type=float, default=1e-3)
    ap.add_argument("--reps", type=int, default=3, help="反復 0〜reps−1（既定 3．計算量のため）")
    ap.add_argument("--smoke", action="store_true", help="試走（avt の 1 ターゲット，反復 1，lam 1 つ．出力は --out か checks/E85/code_run/smoke）")
    args = ap.parse_args()
    if args.smoke:
        args.worlds, args.reps, args.lams = ["avt"], 1, [args.lam_main]
        if args.out == "results/e85":
            args.out = "checks/E85/code_run/smoke"
    out = ROOT / args.out
    if out.exists() and any(out.glob("*.csv")):
        raise SystemExit(f"{out} に古い出力がある．消してから流す")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__), out / "script_copy.py")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "exp", "measure", "data/metadata"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    header = f"E85 commit={commit} dirty={bool(dirty)} lams={args.lams} reps={args.reps} device={DEV}"
    prov = {"commit": commit, "dirty": bool(dirty), "script_sha256": sha256(Path(__file__)), "args": vars(args) | {"out": str(out)}, "inputs": {}, "device": str(DEV)}
    rows_u, rows_p, chk = [], [], []
    for world in args.worlds:
        fused = WORLDS[world]
        singles = fused_parts(fused)[1]
        udir = ROOT / "results" / "e80" / world / "units" / fused
        Ts = sorted(p.name for p in udir.iterdir() if p.is_dir())
        if args.smoke:
            Ts = Ts[:1]
        feats = {}
        for T in Ts:
            files = sorted(udir.glob(f"{T}/*_r*.npz"))
            files = [f for f in files if not f.name.endswith("_agg.npz")]
            for fpath in files:
                c, r = fpath.stem.rsplit("_r", 1)
                r = int(r)
                if r >= args.reps:
                    continue
                if c not in feats:
                    feats[c] = feats_for(c, fused)
                prov["inputs"][str(fpath.relative_to(ROOT))] = sha256(fpath)
                u = Unit(world, T, c, r, feats[c])
                run_unit(u, singles, args.lams, rows_u, rows_p, chk)
                print(world, T, c, r, "done", flush=True)
            prov["inputs"][f"e77_{fused}/{T}_pseudo.csv"] = sha256(ROOT / "results" / f"e77_{fused}" / f"{T}_pseudo.csv")
    U, Pp = pd.DataFrame(rows_u), pd.DataFrame(rows_p)
    if chk:
        pd.DataFrame(chk).to_csv(out / "checks.csv", index=False)
        raise SystemExit(f"none_te の UAR が pseudo.csv の none の行と合わない（{len(chk)} 件．checks.csv）")
    write(U, out / "units.csv", header)
    write(Pp, out / "pairs.csv", header)
    per, S = summarize(U, args.lam_main)
    write(per, out / "per_unit.csv", header)
    write(S, out / "summary.csv", header)
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out, len(U), "rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
