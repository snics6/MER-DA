"""【本番の感度】E82（GPU）：P4 の手法（エントロピー最小化・IM・自己学習）の出発点への罰則 C′ を E39 の値の 3 倍・10 倍にした行を，E77 と同じ形で足す。

設計の正本は `docs/exploratory-analyses.md` の「E82．P4 の手法の出発点への罰則を選び直す」．事実の調べは `checks/E81/step7_facts.md` の 5 節。
模擬査読の [S4-3]：E77 は P4 の手法の C′ を none のために選んだ E39 の値に固定した．この台本は C′ だけを倍率 --mults 倍にして，
`compare_dai.run_ab` の self・entmin・im の部分（と評価の単位での種 1・2 の解き直し）を同じ単位で解き直す．倍率 1 の行は E77 の行をそのまま使うので作らない。

何を変えて，何を変えないか：
  変える  ：self・entmin・im の解きで使う C′（`Prob2.solve`・`compare_dai.im_solve` が解くときに読む `pr.s["cp"]`）＝ 倍率 × E39 の C′（`geom_e0.settings`）
  変えない：C_b・ρ_M・罰則の中心（E39 の下書き）・生徒・候補・K 本・テスト．自己学習の出発点 P0・エントロピー最小化と IM の初期値は
            E39 の C′ で解いた none（none は E77 の本番と同じ束（--none-ks．既定 0 5 25）で解く．none の行は E77 の none と同じ値のはず）
  none の対照：倍率ごとに同じ C′ で解いた none の行（param `-C3x`・`-C10x`）も書く（仮のターゲットと評価の両方．関門 1 の define D10・compare）．
            K=0 の none は出発点そのもので C′ に依らないので，E39 の none と W・b がビット単位で同じことを台本の中で確かめ（違えば止まる），`<T>_e82check.csv` に書く
  離れ    ：W・b のある行ごとに，解の出発点（E39 の下書き）からの離れ ‖W − W_出発点‖_F ÷ ‖W_出発点‖_F と ‖b − b_出発点‖ を `<T>_dist.csv` に書く（設計の 2 節の記述の列）
  格子    ：self は τ 3 × m_eff 4，entmin は m_eff 4，IM は α 3 × m_eff 4（TIM は E79 が選ばないので解かない）．評価の単位では
            entmin・IM を揺らした出発点（`compare_dai.shaken`．種 --nc-seeds）からも解く（E77 と同じ）
  範囲    ：--kind audio_wavlm / video / text_roberta（種類の生徒．P4）か --world fuse-audio-video / fuse-audio-video-text（融合の生徒だけ．P4′）
  単位    ：E77 と同じ（仮のターゲット c ∈ S の --pseudo-reps，評価のターゲットの --reps．IEMOCAP-script の仮のターゲットは --copy-pseudo で即興の写し）
  行の鍵  ：param に倍率を付ける（entmin `-C3x`・`-C3xs1`，im `a0.1C3x`・`a0.1C3xs1`，self `0.5C3x`．種の印は最後）．倍率 1（試し専用）は E77 の param のまま

出力（--out．既定 results/e82_<名前>．E77 と同じ形．E79 の読み手（exp/analyze_e82.py）がそのまま読む）：
  `<T>.csv`（target, rep, row_id, student, method, param, meff, K, uar, commit），`<T>_r<r>_logp.npy`・`_pred.npy`・`_rowids.npy`・`_te.npy`・
  `_W_<生徒>.npy`・`_b_<生徒>.npy`・`_wrows_<生徒>.npy`，`<T>_pseudo.csv`，`<T>_ps_<c>_r<r>_none.npz`，`<T>_samples.csv`，`<T>_settings.json`，
  `<T>_conv.csv`（解きごとの所要と収束．what に倍率の印），`<T>_convp.csv`（問ごとの反復の回数・勾配と，解いた C′ の値 cp），`<T>_diag.csv`（自己学習の疑似ラベル），
  `<T>_cp.csv`（単位 × 生徒 × 倍率の E39 の C′ と解きに使った C′），`<T>_dist.csv`（行ごとの出発点からの離れ），`<T>_e82check.csv`（K=0 の倍率の none の検算），
  provenance.json・provenance_log.jsonl
照合（--check E77 の場所．GPU を使わない）：`check.csv`（none の E77 との一致，E77 と鍵が重ならないこと，同じ鍵の行の差．値は出さず差と数だけを出す）．
  GPU の型（E82 と E77 の provenance.json・provenance_log.jsonl の gpu がすべて同じ 1 つの型）も照らす．
  none（E39 の C′）の不一致・相手の無い none・倍率 1 を含まない出力で E77 と重なる鍵（倍率の none を含む）・K=0 の検算の食い違い・GPU の型の違いが
  1 つでもあれば終了の値 1

    ./.venv/bin/python exp/s43_penalty.py --kind video --copy-pseudo                         # 本番（倍率 3・10，K 0・5，種 1・2）
    ./.venv/bin/python exp/s43_penalty.py --world fuse-audio-video-text --copy-pseudo
    ./.venv/bin/python exp/s43_penalty.py --kind video --check results/e77_video              # 照合
    ./.venv/bin/python exp/s43_penalty.py --kind video --targets AFEW --reps 0 --pseudo-targets none --mults 1 --none-ks 0 5 \\
        --out <試し> --allow-dirty                                                           # 恒等の試し（compare_dai.py --ks 0 5 と比べる）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import baselines_distill as BD  # noqa: E402
import committee as CM  # noqa: E402
import compare_b1 as CB  # noqa: E402
import compare_dai as CD  # noqa: E402
import map as M  # noqa: E402
import meta5 as M5  # noqa: E402
import multimodal_distill as MD  # noqa: E402
from baselines_distill import hard_rows, mass  # noqa: E402
from data import groups_for  # noqa: E402
from geom_e0 import settings  # noqa: E402
from teach_student39 import Student  # noqa: E402

KINDS = ("audio_wavlm", "video", "text_roberta")                               # P4 の種類
P4_WORLDS = ("fuse-audio-video", "fuse-audio-video-text")                      # P4′ の融合の生徒
NC_SEEDS = (1, 2)                                                              # --nc-seeds で変える（E77 と同じ既定）
NONE_KS = ("0", "5", "25")                                                     # none を解く束の K（E77 の本番の --ks 0 5 25 と同じ束にして none をそろえる）
TOL_LOGP, TOL_UAR = 1e-5, 1e-9                                                 # --check の許容幅（E79 の none の照合と同じ）
DT, DEV = CD.DT, CD.DEV


def ctag(mult):
    """倍率の印．倍率 1 は E77 の param のまま（試し専用）。"""
    return "" if float(mult) == 1.0 else f"C{float(mult):g}x"


def sha_file(f):
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()


class Prob3(CD.Prob2):
    """`compare_dai.Prob2` と同じ問を同じ手順で解く．問ごとの記録（convp）に解いた C′ を足す（`f.n_iter` の鍵が解きに使った C′ なので，
    鍵が違えば KeyError で止まる）。"""

    def note(self, what, keys, fits):
        cp = self.s["cp"]
        for key, f in zip(keys, fits):
            self.per.append({"what": what, "key": str(key), "n_iter": int(f.n_iter[cp]), "grad": float(f.grad[cp]), "cp": float(cp)})


class Rec2(CD.Rec):
    """`compare_dai.Rec` に 2 つを足す：(1) W・b のある行ごとに，解の出発点からの離れ ‖W − W_出発点‖_F ÷ ‖W_出発点‖_F と ‖b − b_出発点‖
    （生徒の座標 Z の上．出発点は E39 の下書き＝罰則の中心＝K=0 の none．`start[生徒]`）を `dist` に書く．(2) 倍率の C′ の none の行（param が '-' でない）は
    仮のターゲットの none の対数の確率（`_ps_*_none.npz`．E79 の δ の材料と照合）を上書きしない（npz は E39 の C′ の none だけを持つ）。"""

    def __init__(self, tag, yte, keep):
        super().__init__(tag, yte, keep)
        self.start, self.dist, self.check = {}, [], []

    def add(self, student, method, param, meff, K, probs=None, Wb=None, Zte=None, logits=None):
        mult_none = method == "none" and str(param) != "-"
        old = self.none_lp.get((student, K))
        super().add(student, method, param, meff, K, probs=probs, Wb=Wb, Zte=Zte, logits=logits)
        if mult_none and not self.keep:
            if old is None:
                self.none_lp.pop((student, K), None)
            else:
                self.none_lp[(student, K)] = old
        if Wb is not None and student in self.start:
            W0, b0 = self.start[student]
            self.dist.append({**self.tag, "row_id": self.rows[-1]["row_id"], "student": student, "method": method, "param": str(param),
                              "meff": str(meff), "K": K, "dW_rel": float(torch.linalg.norm(Wb[0] - W0) / torch.linalg.norm(W0)),
                              "db": float(torch.linalg.norm(Wb[1] - b0))})


# ---------------------------------------------------------------- 1 つの生徒

def im_timed(pr, starts, combos, what):
    """`compare_dai.im_solve` を呼び，所要と収束を pr.stats に足す（E77 の conv には IM の行が無い．所要の見積もりのため）。"""
    n0 = len(pr.per)
    t0 = time.perf_counter()
    sol = CD.im_solve(pr, starts, combos, what=what)
    per = pr.per[n0:]
    pr.stats.append({"what": what, "n_prob": len(combos), "sec": time.perf_counter() - t0,
                     "n_iter_max": max((x["n_iter"] for x in per), default=-1), "grad_max": max((x["grad"] for x in per), default=float("nan"))})
    return sol


def run_methods(pr, rec, student, none, P0, mult, cp0, cplog):
    """`compare_dai.run_ab` の self・entmin・im（評価の単位では種 --nc-seeds の解き直しも）を C′ = mult × cp0 で解く．
    none・P0 は E39 の C′ で解いた none の解とその候補の上の確率（出発点と初期値）。解き終えたら C′ を戻す。"""
    s, n, Zte = pr.s, pr.n, pr.s["Zte"]
    KS = CD.KS
    tag = ctag(mult)
    if s["cp"] != cp0:
        raise RuntimeError(f"{student}：解く前の C′ が E39 の値でない（{s['cp']} != {cp0}）")
    s["cp"] = float(mult) * cp0
    cplog.append({**rec.tag, "student": student, "mult": float(mult), "cp_e39": float(cp0), "cp_used": float(s["cp"])})
    try:
        if tag:                                                                # none の対照（関門 1 の define D10・compare）：同じ C′ で解いた none（K 本だけ）
            none_m = pr.solve([(K, None, K) for K in KS], what=f"none{tag}")
            for K in KS:
                rec.add(student, "none", f"-{tag}", "0", K, Wb=none_m[K], Zte=Zte)
            if "0" in KS:                                                      # 検算：K=0 の none は出発点そのもので C′ に依らない（E39 の none と同じはず）
                dW = float((none_m["0"][0] - none["0"][0]).abs().max())
                db = float((none_m["0"][1] - none["0"][1]).abs().max())
                rec.check.append({**rec.tag, "student": student, "check": "none_K0_mult_vs_e39", "mult": float(mult), "W_absdiff_max": dW,
                                  "b_absdiff_max": db, "ok": dW == 0.0 and db == 0.0})
                if dW != 0.0 or db != 0.0:
                    raise RuntimeError(f"{rec.tag} {student}：倍率 {mult} の K=0 の none が E39 の none と違う（W {dW}，b {db}）")
        keys = [(tau, meff, K) for tau in CD.TAUS for meff in CD.MEFFS for K in KS]   # 自己学習（run_ab の 397〜410 行と同じ．出発点は E39 の C′ の none）
        cur = {key: P0[key[2]] for key in keys}
        sol = {}
        for rnd in range(BD.ROUNDS_SELF):
            hr = {key: hard_rows(cur[key], key[0], key[1]) for key in keys}
            sol = pr.solve([(key, hr[key], key[2]) for key in keys], what=f"self{tag}")
            for key in keys:
                on = hr[key].view(4, n).sum(0) > 0
                lab_ = hr[key].view(4, n).argmax(0)
                rec.diag.append({**rec.tag, "student": student, "method": "self", "param": f"{key[0]}{tag}", "meff": key[1], "K": key[2],
                                 "round": rnd + 1, "n_pseudo": int(on.sum()),
                                 "acc_pseudo": float((lab_[on] == pr.yt[on]).float().mean()) if on.any() else float("nan"),
                                 "uar_test": float(CB.uar(rec.yte, (Zte @ sol[key][0].T + sol[key][1]).argmax(1).cpu().numpy()))})
            cur = {key: pr.probs(sol[key], s["Z"]) for key in keys}
        for (tau, meff, K), Wb in sol.items():
            rec.add(student, "self", f"{tau}{tag}", meff, K, Wb=Wb, Zte=Zte)
        keys = [(meff, K) for meff in CD.MEFFS for K in KS]                    # エントロピー最小化（411〜415 行．none の解から）
        sol = pr.solve([(key, None, key[1]) for key in keys], u=[mass(meff, n) / n for meff, K in keys], inits=[none[K] for meff, K in keys],
                       what=f"entmin{tag}")
        for (meff, K), Wb in sol.items():
            rec.add(student, "entmin", f"-{tag}", meff, K, Wb=Wb, Zte=Zte)
        im_combos = [(a, meff, K) for a in CD.IM_ALPHAS for meff in CD.MEFFS for K in KS]   # IM（416〜418 行．TIM は E79 が選ばないので解かない）
        for (a, meff, K), Wb in im_timed(pr, none, im_combos, f"im{tag}").items():
            rec.add(student, "im", f"a{a}{tag}", meff, K, Wb=Wb, Zte=Zte)
        if rec.keep and NC_SEEDS:                                              # 揺らした出発点（419〜427 行．評価のターゲットだけ）
            for seed in NC_SEEDS:
                st0 = CD.shaken(none, seed, int(rec.tag["rep"]), student)
                sol = pr.solve([(key, None, key[1]) for key in keys], u=[mass(meff, n) / n for meff, K in keys],
                               inits=[st0[K] for meff, K in keys], what=f"entmin{tag}_s{seed}")
                for (meff, K), Wb in sol.items():
                    rec.add(student, "entmin", f"-{tag}s{seed}", meff, K, Wb=Wb, Zte=Zte)
                for (a, meff, K), Wb in im_timed(pr, st0, im_combos, f"im{tag}_s{seed}").items():
                    rec.add(student, "im", f"a{a}{tag}s{seed}", meff, K, Wb=Wb, Zte=Zte)
    finally:
        s["cp"] = cp0


def solve_student(pr, rec, student, mults, cp0, cplog):
    """none（E39 の C′．E77 の本番と同じ束）を解いて行にし，倍率ごとに self・entmin・im を解く。"""
    s = pr.s
    if s["cp"] != cp0:
        raise RuntimeError(f"{student}：生徒の C′ {s['cp']} が E39 の値 {cp0} と違う")
    none = pr.solve([(K, None, K) for K in NONE_KS], what="none")
    P0 = {K: pr.probs(none[K], s["Z"]) for K in CD.KS}
    for K in CD.KS:
        rec.add(student, "none", "-", "0", K, Wb=none[K], Zte=s["Zte"])
    cplog.append({**rec.tag, "student": student, "mult": 0.0, "cp_e39": float(cp0), "cp_used": float(s["cp"])})   # mult 0 ＝ none の行
    for m in mults:
        run_methods(pr, rec, student, none, P0, m, cp0, cplog)


# ---------------------------------------------------------------- 生徒（E77 と同じ作り方）

def kind_student(kind, ev, Xc, src, lin, n_of, pool_s, C, te, ct, sets):
    """`compare_dai.run_kind` の student_of と同じ（876〜879 行）。"""
    gamma, rho_s, rho_m, cb, cp, _ = sets
    st = Student(ev, Xc, src, lin, n_of, pool_s, C, gamma, rho_s, [rho_m])
    Am = st.mats[rho_m][0]
    return {"st": st, "rho_m": rho_m, "cb": cb, "cp": cp, "Z": st.Xstd @ Am, "Zte": ((M.rows(ev.X, te) - ct - st.src.mu) / st.src.sd) @ Am}


def fused_student(world, T, data_f, S, pool, te, sets, info):
    """`multimodal_distill.make_students` の融合の生徒だけ（81〜90 行の k = world と同じ式．モダリティの生徒は P4′ に使わないので作らない）。
    戻り：(生徒, 候補の平均)。"""
    dat = data_f[T]
    mu = M.rows(dat.X, pool).mean(0)
    Xc = M.rows(dat.X, pool) - mu
    gamma, rho_s, rho_m, cb, cp, _ = sets[world]
    src, lin, n_of = info
    st = Student(dat, Xc, src, lin, n_of, S, MD.C_FIT, gamma, rho_s, [rho_m])
    Am = st.mats[rho_m][0]
    return {"st": st, "rho_m": rho_m, "cb": cb, "cp": cp, "Z": st.Xstd @ Am,
            "Zte": ((M.rows(dat.X, te) - mu - st.src.mu) / st.src.sd) @ Am}, mu


def one_unit(student, s, yq, yte, label_sets, rec, center, mults, cplog):
    """1 つの単位の 1 つの生徒（`compare_dai.kind_unit`・`world_unit` の生徒ごとの部分のうち none・self・entmin・im）。"""
    st = s["st"]
    rec.xform[student] = (st.mats[s["rho_m"]][0], st.src.sd, center + st.src.mu)   # 生の特徴の上の W・b（E77 と同じ写し方）
    t0 = time.perf_counter()
    pr = Prob3(s, yq, label_sets)
    rec.start[student] = (pr.W0, pr.b0)                                       # 解の出発点（E39 の下書き．Z の座標）
    solve_student(pr, rec, student, mults, float(s["cp"]), cplog)
    CD.flush_stats(rec, {student: pr})
    rec.conv.append({**rec.tag, "student": student, "what": "unit_total", "sec": time.perf_counter() - t0})


# ---------------------------------------------------------------- ターゲットごと

def finish(out, T, commit, rows, conv, diag, convp, cplog, dist, check):
    """評価のターゲットまで終えたら記録と行を書く（仮のターゲットの行は評価の前に書いてある）。"""
    pd.DataFrame(dist).assign(commit=commit).to_csv(out / f"{T}_dist.csv", index=False)
    pd.DataFrame(check).assign(commit=commit).to_csv(out / f"{T}_e82check.csv", index=False)
    pd.DataFrame(conv).assign(commit=commit).to_csv(out / f"{T}_conv.csv", index=False)
    pd.DataFrame(diag).assign(commit=commit).to_csv(out / f"{T}_diag.csv", index=False)
    pd.DataFrame(convp).assign(commit=commit).to_csv(out / f"{T}_convp.csv", index=False)
    pd.DataFrame(cplog).assign(commit=commit).to_csv(out / f"{T}_cp.csv", index=False)
    pd.DataFrame(rows).assign(commit=commit).to_csv(out / f"{T}.csv", index=False)  # 最後に書く（check_resume が終わった印に使う）


def copy_extra(out, T, lists):
    """写した仮のターゲットの記録（写し元の `_cp.csv`・`_dist.csv`・`_e82check.csv` の仮のターゲットの行）。lists = {名前: 足す先の list}。"""
    src = CD.COPY_SRC[T]
    for name, lst in lists.items():
        f = out / f"{src}_{name}.csv"
        if f.exists() and f.stat().st_size > 1:
            e = pd.read_csv(f, dtype={"param": str, "meff": str, "K": str})
            if "pseudo" in e.columns:
                lst += e[e.pseudo.notna()].rename(columns={"commit": "src_commit"}).assign(copied_from=src).to_dict("records")


def pseudo_header(T, S, pseudo_targets, pseudo_reps, copy, out, commit, args_now):
    """仮のターゲットの行：写すなら写し，回すなら空で始める。戻り：(prows, conv, diag, convp, 回す群)。"""
    if copy and T in CD.COPY_SRC:
        prows, conv, diag, convp = CD.copy_pseudo(out, T, "E77", CD.KS, commit, CD.pseudo_pairs(S, pseudo_targets, pseudo_reps), args_now)
        return prows, conv, diag, convp, []
    return [], [], [], [], S


def run_kind(kind, T, data, reps, pseudo_reps, out, cache, commit, pseudo_targets, mults, copy, args_now):
    """`compare_dai.run_kind` と同じ単位と生徒（仮のターゲットは合併の C = 0.01，評価は E25 の合併の C）。"""
    S = [s for s in data if CM.corp(s) != CM.corp(T)]
    sets = settings(kind, T)
    C_T = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{kind}" / f"{T}.csv", dtype={"K": str}).C.iloc[0])
    CD.save_settings(out, T, {kind: sets[:5] + (None,)}, {"c_fit_pseudo": CD.C_FIT, "c_eval": C_T, "sources": S,
                                            "e82": {"mults": [float(m) for m in mults], "cp_e39": sets[4],
                                                    "cp": {ctag(m) or "C1x": float(m) * sets[4] for m in mults}}})
    prows, conv, diag, convp, S_loop = pseudo_header(T, S, pseudo_targets, pseudo_reps, copy, out, commit, args_now)
    cplog, dist, check = [], [], []
    if not S_loop and copy and T in CD.COPY_SRC:
        copy_extra(out, T, {"cp": cplog, "dist": dist, "e82check": check})
    for c in S_loop:
        pool_c = [s for s in S if CM.corp(s) != CM.corp(c)]
        if not pool_c or (pseudo_targets and c not in pseudo_targets):
            continue
        src_c, lin_c, n_c = M5.pool_info(data, pool_c, CD.C_FIT, cache)
        evc = data[c]
        ctxc = CB.Ctx(evc, pseudo_reps, list(CD.KS_LAB_ALL), center="ind")
        for r in pseudo_reps:
            pool, te = np.flatnonzero(~ctxc.te[r]), np.flatnonzero(ctxc.te[r])
            s = kind_student(kind, evc, M.rows(evc.X, pool) - ctxc.ct[r], src_c, lin_c, n_c, pool_c, CD.C_FIT, te, ctxc.ct[r], sets)
            rec = Rec2({"pseudo": c, "rep": r}, evc.y[te], keep=False)
            rec.conv, rec.diag, rec.convp, rec.dist, rec.check = conv, diag, convp, dist, check
            one_unit(kind, s, evc.y[pool], evc.y[te], BD.label_sets_of(ctxc, r, pool), rec, ctxc.ct[r], mults, cplog)
            rec.save_pseudo(out, T, c, r, te)
            prows += rec.rows
            del s, rec
            torch.cuda.empty_cache()
    pd.DataFrame(prows).assign(commit=commit).to_csv(out / f"{T}_pseudo.csv", index=False)
    src_pc, lin, n_of = M5.pool_info(data, S, C_T, cache)
    ev = data[T]
    ctx = CB.Ctx(ev, reps, list(CD.KS_LAB_ALL), center="ind")
    rows = []
    for r in reps:
        pool, te = np.flatnonzero(~ctx.te[r]), np.flatnonzero(ctx.te[r])
        s = kind_student(kind, ev, M.rows(ev.X, pool) - ctx.ct[r], src_pc, lin, n_of, S, C_T, te, ctx.ct[r], sets)
        rec = Rec2({"target": T, "rep": r}, ev.y[te], keep=True)
        rec.conv, rec.diag, rec.convp, rec.dist, rec.check = conv, diag, convp, dist, check
        one_unit(kind, s, ev.y[pool], ev.y[te], BD.label_sets_of(ctx, r, pool), rec, ctx.ct[r], mults, cplog)
        rec.save(out, T, r, te)
        rows += rec.rows
        del s, rec
        torch.cuda.empty_cache()
    finish(out, T, commit, rows, conv, diag, convp, cplog, dist, check)


def run_world(world, T, data_f, reps, pseudo_reps, out, cache, commit, pseudo_targets, mults, copy, args_now):
    """`compare_dai.run_world` と同じ単位で，融合の生徒だけ（ソースの合併の C は C_FIT）。"""
    S = [s for s in data_f if CM.corp(s) != CM.corp(T)]
    sets = {world: settings(world, T)}
    CD.save_settings(out, T, sets, {"c_fit": CD.C_FIT, "sources": S,
                                    "e82": {"mults": [float(m) for m in mults], "cp_e39": sets[world][4], "students": [world],
                                            "cp": {ctag(m) or "C1x": float(m) * sets[world][4] for m in mults}}})
    prows, conv, diag, convp, S_loop = pseudo_header(T, S, pseudo_targets, pseudo_reps, copy, out, commit, args_now)
    cplog, dist, check = [], [], []
    if not S_loop and copy and T in CD.COPY_SRC:
        copy_extra(out, T, {"cp": cplog, "dist": dist, "e82check": check})
    for c in S_loop:
        pool_c = [s for s in S if CM.corp(s) != CM.corp(c)]
        if not pool_c or (pseudo_targets and c not in pseudo_targets):
            continue
        info_c = M5.pool_info(data_f, pool_c, CD.C_FIT, cache)
        evc = data_f[c]
        ctxc = CB.Ctx(evc, pseudo_reps, list(CD.KS_LAB_ALL), center="ind")
        for r in pseudo_reps:
            pool, te = np.flatnonzero(~ctxc.te[r]), np.flatnonzero(ctxc.te[r])
            s, mu = fused_student(world, c, data_f, pool_c, pool, te, sets, info_c)
            rec = Rec2({"pseudo": c, "rep": r}, evc.y[te], keep=False)
            rec.conv, rec.diag, rec.convp, rec.dist, rec.check = conv, diag, convp, dist, check
            one_unit(world, s, evc.y[pool], evc.y[te], BD.label_sets_of(ctxc, r, pool), rec, mu, mults, cplog)
            rec.save_pseudo(out, T, c, r, te)
            prows += rec.rows
            del s, rec
            torch.cuda.empty_cache()
    pd.DataFrame(prows).assign(commit=commit).to_csv(out / f"{T}_pseudo.csv", index=False)
    info = M5.pool_info(data_f, S, CD.C_FIT, cache)
    ev = data_f[T]
    ctx = CB.Ctx(ev, reps, list(CD.KS_LAB_ALL), center="ind")
    rows = []
    for r in reps:
        pool, te = np.flatnonzero(~ctx.te[r]), np.flatnonzero(ctx.te[r])
        s, mu = fused_student(world, T, data_f, S, pool, te, sets, info)
        rec = Rec2({"target": T, "rep": r}, ev.y[te], keep=True)
        rec.conv, rec.diag, rec.convp, rec.dist, rec.check = conv, diag, convp, dist, check
        one_unit(world, s, ev.y[pool], ev.y[te], BD.label_sets_of(ctx, r, pool), rec, mu, mults, cplog)
        rec.save(out, T, r, te)
        rows += rec.rows
        del s, rec
        torch.cuda.empty_cache()
    finish(out, T, commit, rows, conv, diag, convp, cplog, dist, check)


# ---------------------------------------------------------------- 照合（GPU を使わない）

def read_tab(f):
    d = pd.read_csv(f, dtype={"param": str, "meff": str, "K": str, "student": str, "method": str})
    return d if "method" in d.columns else None


def run_checks(out, e77):
    """E82 の出力を E77 の出力と照らす（手法ごと）：none の一致（仮のターゲットの UAR・npz の対数の確率，評価のターゲットの対数の確率・予測・UAR），
    テストの行と発話の並び，E77 と鍵が重ならないこと．倍率 1 を含む試しの出力（恒等の試し）では，重なった行の差を測る。値は出さない。"""
    out, e77 = Path(out), Path(e77)
    prov = json.loads((out / "provenance.json").read_text())
    has1 = any(float(m) == 1.0 for m in prov.get("mults", []))
    rows = []
    # GPU の型：E82 の起動の記録すべてと E77 の起動の記録すべてで同じ型か（none を E77 と同じ値にそろえる前提．設計の 4 節．gate-code の I8）
    def gpus(d):
        recs = [json.loads((d / "provenance.json").read_text())] if (d / "provenance.json").exists() else []
        lg = d / "provenance_log.jsonl"
        if lg.exists():
            recs += [json.loads(x) for x in lg.read_text().splitlines() if x.strip()]
        return sorted({str(r.get("gpu", "")) for r in recs})
    g82, g77 = gpus(out), gpus(e77)
    rows.append({"check": "gpu_type", "e82": ";".join(g82), "e77": ";".join(g77), "ok": len(g82) == 1 and g82 == g77 and g82[0] != ""})
    key = ["student", "method", "param", "meff", "K"]

    def ok_of(meth, n, n_m, diff, tol):
        if meth == "none":                                                     # none は全部に相手があり，差が許容幅の中
            return n_m == n and not (diff > tol)
        return has1 or n_m == 0                                                # 倍率 3・10 の行は E77 と重ならない
    for f in sorted(out.glob("*_pseudo.csv")):
        T = f.name[: -len("_pseudo.csv")]
        a = read_tab(f)
        g = e77 / f.name
        if a is None:
            rows.append({"check": "pseudo_empty", "target": T, "ok": True})
            continue
        b = read_tab(g) if g.exists() else None
        if b is None:
            rows.append({"check": "pseudo_e77_missing", "target": T, "ok": False})
            continue
        j = a.merge(b[["pseudo", "rep"] + key + ["uar"]], on=["pseudo", "rep"] + key, suffixes=("", "_e77"), how="left")
        j["mkey"] = np.where((j.method == "none") & (j.param != "-"), "none_mult", j.method)   # 倍率の C′ の none は E77 に相手が無いはず
        for meth, x in j.groupby("mkey"):
            mt = x.uar_e77.notna()
            d = float((x.uar - x.uar_e77).abs()[mt].max()) if mt.any() else np.nan
            rows.append({"check": f"pseudo_{meth}", "target": T, "n": int(len(x)), "n_matched": int(mt.sum()), "uar_max_abs": d,
                         "ok": ok_of(meth, len(x), int(mt.sum()), d, TOL_UAR)})
        for z in sorted(out.glob(f"{T}_ps_*_none.npz")):
            w = e77 / z.name
            if not w.exists():
                rows.append({"check": "ps_npz_e77_missing", "target": T, "file": z.name, "ok": False})
                continue
            za, zb = np.load(z), np.load(w)
            if not np.array_equal(za["te"], zb["te"]) or not np.array_equal(za["y"], zb["y"]):
                rows.append({"check": "ps_npz_te_y", "target": T, "file": z.name, "ok": False})
                continue
            kb = {str(k): i for i, k in enumerate(zb["keys"])}
            v, n_m, miss = 0.0, 0, 0
            for i, k in enumerate(za["keys"]):
                if str(k) not in kb:
                    miss += 1
                    continue
                v = max(v, float(np.abs(za[f"lp{i}"] - zb[f"lp{kb[str(k)]}"]).max()))
                n_m += 1
            rows.append({"check": "ps_npz_none", "target": T, "file": z.name, "n": n_m + miss, "n_matched": n_m, "max_abs": v,
                         "ok": miss == 0 and v <= TOL_LOGP})
    for f in sorted(out.glob("*_samples.csv")):
        T = f.name[: -len("_samples.csv")]
        a = read_tab(out / f"{T}.csv") if (out / f"{T}.csv").exists() else None
        b = read_tab(e77 / f"{T}.csv") if (e77 / f"{T}.csv").exists() else None
        if a is None or b is None:
            rows.append({"check": "eval_missing", "target": T, "e82": a is not None, "e77": b is not None, "ok": False})
            continue
        sb = pd.read_csv(e77 / f"{T}_samples.csv")
        rows.append({"check": "samples", "target": T, "ok": bool(np.array_equal(pd.read_csv(f).sample_id.astype(str), sb.sample_id.astype(str)))})
        for r in sorted(set(a.rep)):
            if not (e77 / f"{T}_r{r}_te.npy").exists():
                rows.append({"check": "eval_rep_e77_missing", "target": T, "rep": int(r), "ok": False})
                continue
            rows.append({"check": "te", "target": T, "rep": int(r),
                         "ok": bool(np.array_equal(np.load(out / f"{T}_r{r}_te.npy"), np.load(e77 / f"{T}_r{r}_te.npy")))})
            La, Lb = np.load(out / f"{T}_r{r}_logp.npy", mmap_mode="r"), np.load(e77 / f"{T}_r{r}_logp.npy", mmap_mode="r")
            Pa, Pb = np.load(out / f"{T}_r{r}_pred.npy", mmap_mode="r"), np.load(e77 / f"{T}_r{r}_pred.npy", mmap_mode="r")
            ar, br = a[a.rep == r], b[b.rep == r]
            rows.append({"check": "eval_rows_vs_logp", "target": T, "rep": int(r), "n": int(len(ar)),
                         "ok": bool(len(ar) == La.shape[0] == Pa.shape[0] and np.array_equal(ar.row_id.to_numpy(), np.arange(len(ar))))})
            j = ar.merge(br[key + ["row_id", "uar"]], on=key, suffixes=("", "_e77"), how="left")
            j["mkey"] = np.where((j.method == "none") & (j.param != "-"), "none_mult", j.method)
            for meth, x in j.groupby("mkey"):
                mt = x[x.row_id_e77.notna()]
                vl = max((float(np.abs(La[int(i)] - Lb[int(k)]).max()) for i, k in zip(mt.row_id, mt.row_id_e77)), default=np.nan)
                npred = int(sum(int((Pa[int(i)] != Pb[int(k)]).sum()) for i, k in zip(mt.row_id, mt.row_id_e77)))
                du = float((mt.uar - mt.uar_e77).abs().max()) if len(mt) else np.nan
                rows.append({"check": f"eval_{meth}", "target": T, "rep": int(r), "n": int(len(x)), "n_matched": int(len(mt)), "max_abs": vl,
                             "n_pred_differ": npred, "uar_max_abs": du, "ok": ok_of(meth, len(x), len(mt), vl, TOL_LOGP)})
    for f in sorted(out.glob("*_e82check.csv")):                              # 台本の中の検算（K=0 の倍率の none ＝ E39 の none）
        e = pd.read_csv(f) if f.stat().st_size > 1 else pd.DataFrame()
        if len(e) and "check" in e.columns:
            for c_, g in e.groupby("check"):
                rows.append({"check": f"inscript_{c_}", "target": f.name[: -len("_e82check.csv")], "n": int(len(g)),
                             "max_abs": float(g[["W_absdiff_max", "b_absdiff_max"]].max().max()), "ok": bool(g.ok.astype(bool).all())})
    d = pd.DataFrame(rows)
    if d.empty:
        print("照合する出力が無い", flush=True)
        return 1
    d.to_csv(out / "check.csv", index=False)
    for c_, g in d.groupby("check", sort=False):                                # 差と数だけを出す（UAR などの値は出さない）
        msg = f"{c_}：{len(g)} 項目，食い違い {int((~g.ok.astype(bool)).sum())}"
        if c_ == "gpu_type":
            msg += f"，E82 {g.e82.iloc[0]} ／ E77 {g.e77.iloc[0]}"
        for col, lab, fmt in (("n", "行", "d"), ("n_matched", "E77 に相手のある行", "d"), ("max_abs", "対数の確率の差の最大", ".3g"),
                              ("n_pred_differ", "予測の食い違い", "d"), ("uar_max_abs", "UAR の差の最大", ".3g")):
            if col in g and g[col].notna().any():
                v = g[col].sum() if fmt == "d" else g[col].max()
                msg += f"，{lab} {v:{'.0f' if fmt == 'd' else fmt}}"
        print(msg, flush=True)
    nbad = int((~d.ok.astype(bool)).sum())
    print(f"照合：食い違い {nbad} 項目（{out / 'check.csv'}）", flush=True)
    return 1 if nbad else 0


# ---------------------------------------------------------------- 入口

def main() -> int:
    global NC_SEEDS, NONE_KS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=KINDS)
    ap.add_argument("--world", choices=P4_WORLDS)
    ap.add_argument("--targets", "--target", dest="targets", nargs="+", help="評価のターゲット（既定は全群．--copy-pseudo では IEMOCAP-script を即興の後に回す）")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--pseudo-reps", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--pseudo-targets", nargs="+", help="試し用．仮のターゲットをこの群だけにする（'none' で仮のターゲットを回さない）")
    ap.add_argument("--ks", nargs="+", default=["0", "5"], help="解く K（E82 は P4 の K 0・5）．K 本の抜き方は全 K で作る")
    ap.add_argument("--none-ks", nargs="+", default=list(NONE_KS), help="none を解く束の K（E77 の本番の --ks と同じにする．恒等の試しは比べる相手の --ks に合わせる）")
    ap.add_argument("--mults", nargs="+", type=float, default=[3.0, 10.0], help="C′ の倍率（E39 の値に掛ける．1 は E77 の param のまま書く試し専用）")
    ap.add_argument("--copy-pseudo", action="store_true", help="IEMOCAP-script の仮のターゲットを同じ出力の場所の IEMOCAP-impro の結果から写す（E77 の短縮 a）")
    ap.add_argument("--nc-seeds", nargs="*", type=int, default=[1, 2], help="entmin・IM を揺らした出発点から解き直す種（評価のターゲットだけ．E77 と同じ）")
    ap.add_argument("--out", "--out-dir", dest="out", type=Path)
    ap.add_argument("--gpu-fraction", type=float, default=0.3)
    ap.add_argument("--allow-dirty", action="store_true", help="試し専用．exp/・measure/ に未コミットの変更があっても，この台本が未追跡でも流す（provenance に残す）")
    ap.add_argument("--check", type=Path, metavar="E77_DIR", help="出力を E77 の同じ実行の出力と照らす（GPU を使わない）")
    args = ap.parse_args()
    if bool(args.kind) == bool(args.world):
        ap.error("--kind か --world のどちらか 1 つを指定する")
    name = args.kind or args.world
    out = args.out or ROOT / "results" / f"e82_{name}"
    if args.check:
        return run_checks(out, args.check)
    if any(m <= 0 for m in args.mults) or len(set(args.mults)) != len(args.mults):
        ap.error(f"--mults は正で重複なし（{args.mults}）")
    CD.set_ks(args.ks)
    ks = CD.KS
    bad = [k for k in args.none_ks if k not in BD.KS]
    if bad or not set(ks) <= set(args.none_ks):
        ap.error(f"--none-ks は {BD.KS} の中から選び，--ks を含む（{args.none_ks}）")
    NONE_KS = tuple(k for k in BD.KS if k in args.none_ks)
    NC_SEEDS = tuple(args.nc_seeds)
    mults = tuple(float(m) for m in args.mults)
    if 1.0 in mults:
        print("注意：倍率 1 は E77 と同じ param の行を書く（恒等の試し専用．E82 の読み手は E77 と重なる鍵で止まる）", flush=True)
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    dirty = [ln for ln in git("status", "--porcelain", "--untracked-files=no").splitlines() if re.search(r"\s(exp|measure)/", ln)]
    tracked = bool(git("ls-files", "exp/s43_penalty.py"))
    if (dirty or not tracked) and not args.allow_dirty:
        raise SystemExit(f"exp/・measure/ に未コミットの変更がある（{dirty}）か，この台本が未追跡（{not tracked}）．コミットしてから流す（試しは --allow-dirty）")
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction)
    out.mkdir(parents=True, exist_ok=True)
    commit = git("rev-parse", "HEAD")
    groups = groups_for(name)
    targets = args.targets or groups
    bad = [t for t in targets if t not in groups]
    if bad:
        ap.error(f"{name} に無い群：{bad}")
    targets = sorted(targets, key=lambda t: (t in CD.COPY_SRC, groups.index(t)))   # 台本は即興の後（写す）
    todo = [t for t in targets if not CD.check_resume(out, t, commit)]
    ptg = ["__none__"] if args.pseudo_targets == ["none"] else args.pseudo_targets
    args_now = {"KS": list(ks), "none_ks": list(NONE_KS), "mults": list(mults), "nc_seeds": list(NC_SEEDS), "nc_scale": CD.NC_SCALE,
                "pseudo_reps": list(args.pseudo_reps), "pseudo_targets": args.pseudo_targets}
    prov = {"script": "exp/s43_penalty.py", "experiment": "E82", "argv": sys.argv[1:], "commit": commit, "dirty": bool(dirty), "dirty_files": dirty,
            "script_tracked": tracked, "allow_dirty": args.allow_dirty, "script_sha256": sha_file(__file__), "compare_dai_sha256": sha_file(CD.__file__),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "run": name, "role": "kind" if args.kind else "fused", "students": [name],
            "targets": todo, "reps": args.reps, **args_now, "copy_pseudo": args.copy_pseudo, "gpu_fraction": args.gpu_fraction,
            "grid": {"taus": list(CD.TAUS), "meffs": list(CD.MEFFS), "im_alphas": list(CD.IM_ALPHAS), "rounds_self": BD.ROUNDS_SELF, "tim": False,
                     "c_fit": CD.C_FIT, "param_tag": "C<倍率>x（種の印 s<種> は最後）"},
            **CD.versions(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", "sec": {}}
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    with open(out / "provenance_log.jsonl", "a") as fh:                         # 写しの照合（compare_dai.copy_pseudo）が読む
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    if not todo:
        print(f"{name}：全ターゲット済み", flush=True)
        return 0
    t_all = time.perf_counter()
    data = {g: M.load(g, name) for g in groups}                               # 世界では融合の特徴（compare_dai の data_f と同じ）
    cache = {}
    for t in todo:
        t0 = time.perf_counter()
        CD.samples_table(name, t).to_csv(out / f"{t}_samples.csv", index_label="row")
        fn = run_world if args.world else run_kind
        fn(name, t, data, args.reps, args.pseudo_reps, out, cache, commit, ptg, mults, args.copy_pseudo, args_now)
        prov["sec"][t] = round(time.perf_counter() - t0, 1)
        (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
        mem = f"　GPU 最大 {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB" if torch.cuda.is_available() else ""
        print(f"{name} {t}：{time.perf_counter() - t0:.0f} 秒{mem}", flush=True)
    prov["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    prov["sec"]["total"] = round(time.perf_counter() - t_all, 1)
    prov["gpu_max_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else None
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
