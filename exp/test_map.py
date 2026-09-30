"""`map.py` の検算。分割・K 本の抽出・C の規則を，実データと人工データで確かめる。

    ./.venv/bin/python exp/test_map.py      # 1 分ほど

`map.py` を変えたら必ず流す。1 つでも落ちたら，その版で実験を回さない。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import logreg  # noqa: E402
import map as M  # noqa: E402

KIND = "audio_wavlm"


def check(name, cond, detail):
    print(("OK   " if cond else "FAIL ") + f"{name}: {detail}")
    return bool(cond)


def main() -> int:
    ok = True
    data = {g: M.load(g, KIND) for g in M.groups_for(KIND)}

    # ---- テストと K 本（全群 × 30 回）
    leak, miss4, notk, nest, outside, redraw = [], [], [], [], [], []
    for g in data.values():
        for r in range(M.N_REPS):
            te = M.test_mask(g, r)
            pool = np.flatnonzero(~te)
            if set(g.person[te]) & set(g.person[pool]):
                leak.append((g.name, r))
            if len(np.unique(g.y[te])) < M.N_CLASSES:
                miss4.append((g.name, r))
            order = M.kshot_order(g, pool, r)
            prev = set()
            for K in M.K_GRID:
                idx = M.take_k(order, K)
                cnt = np.bincount(g.y[idx], minlength=M.N_CLASSES)
                if K != "all" and not (cnt == int(K)).all():
                    notk.append((g.name, r, K))
                if not prev <= set(idx):
                    nest.append((g.name, r, K))
                if te[idx].any():
                    outside.append((g.name, r, K))
                prev = set(idx)
            again = M.kshot_order(g, pool, r)
            if not all(np.array_equal(a, b) for a, b in zip(order, again)):
                redraw.append((g.name, r))
    ok &= check("テストと学習プールで人物が重ならない", not leak, f"違反 {leak[:3]}")
    ok &= check("テストに 4 感情がそろう", not miss4, f"違反 {miss4[:3]}")
    ok &= check("K 本は感情ごとにちょうど K 本", not notk, f"違反 {notk[:3]}")
    ok &= check("K 本は K をまたいで入れ子", not nest, f"違反 {nest[:3]}")
    ok &= check("K 本はテストから取らない", not outside, f"違反 {outside[:3]}")
    ok &= check("同じ (群, r) なら同じ K 本", not redraw, f"違反 {redraw[:3]}")

    g = data["IEMOCAP-impro"]
    times = {}
    for r in range(M.N_REPS):
        for p in np.unique(g.person[M.test_mask(g, r)]):
            times[p] = times.get(p, 0) + 1
    ok &= check("IEMOCAP-impro は 10 人全員が 6 回ずつテストに入る",
                len(times) == 10 and set(times.values()) == {6}, f"{times}")

    # ---- 同じコーパスの群どうし：テストの人物をソースから外す
    tgt, src, other = data["IEMOCAP-impro"], data["IEMOCAP-script"], data["MOSEI"]
    bad, n_ex = [], []
    for r in range(M.N_REPS):
        tp = np.unique(tgt.person[M.test_mask(tgt, r)])
        keep = ~np.isin(src.person, tp)
        n_ex.append(int((~keep).sum()))
        if set(src.person[keep]) & set(tp) or np.isin(other.person, tp).any():
            bad.append(r)
    ok &= check("IEMOCAP 同士ではテストの人物がソースから消える（別コーパスは消えない）",
                not bad and min(n_ex) > 0, f"違反 {bad[:3]}，外した本数 {min(n_ex)}〜{max(n_ex)}")

    # ---- 交差検証の袋：同じ人物は同じ袋（IEMOCAP 同士の合併）
    r = 0
    te = M.test_mask(tgt, r)
    idx = M.take_k(M.kshot_order(tgt, np.flatnonzero(~te), r), "10")
    keep = ~np.isin(src.person, tgt.person[te])
    person = np.r_[tgt.person[idx], src.person[keep]]
    first = np.r_[np.ones(len(idx), bool), np.zeros(int(keep.sum()), bool)]
    folds = M.cv_folds(person, first, seed=r)          # 人物が袋をまたげば中で止まる
    ok &= check("合併の交差検証：全行が袋に入り，ターゲットは 3 袋に散る",
                (folds >= 0).all() and len(np.unique(folds[first])) == M.N_CV_FOLDS,
                f"ターゲットの袋ごとの本数 {np.bincount(folds[first]).tolist()}")

    # ---- C の規則
    C = M.choose_c
    cases = [
        ({0.01: .5, 0.1: .6, 1.0: .4, 10.0: .3}, (0.1, False)),
        ({0.01: .5, 0.1: .5, 1.0: .5, 10.0: .5}, (1.0, True)),
        ({0.01: .5, 0.1: .7, 1.0: .4, 10.0: .7}, (0.1, True)),     # 1.0 から等距離 → 小さい方
        ({0.01: .7, 0.1: .4, 1.0: .4, 10.0: .7}, (10.0, True)),    # 10 の方が 1.0 に近い
        ({0.01: 1 / 3 + 1 / 6, 0.1: .5, 1.0: 0., 10.0: 0.}, (0.1, True)),  # 丸めの差は同点
    ]
    got = [C(s) for s, _ in cases]
    ok &= check("C の同点の規則", got == [w for _, w in cases], f"{got}")

    # ---- K=1 のターゲットのみは交差検証ができず C を固定する
    fixed = []
    for g in data.values():
        te = M.test_mask(g, 0)
        idx = M.take_k(M.kshot_order(g, np.flatnonzero(~te), 0), "1")
        t = M.train(M.rows(g.X, idx), g.y[idx], g.person[idx], np.ones(len(idx), bool), 0)
        fixed.append(t.rule == "fixed" and t.C == M.C_DEFAULT)
    ok &= check("K=1 のターゲットのみは全群で C を固定", all(fixed), f"{fixed}")

    # ---- GPU の学習器が sklearn と同じ問題を解いているか（RAVDESS，話者で分けた保留側で比べる）
    # sklearn は止める基準を 1e-8 に厳しくして比べる。**既定の 1e-4 では C=10 が 25 回で止まり，
    # 保留側の予測が最適解と 5.7% 食い違う**（2026-09-30 実測。目的関数は GPU の方が 8.5e-4 低い）
    g = data["RAVDESS"]
    te = M.test_mask(g, 0)
    Xtr, ytr, Xte = M.rows(g.X, ~te), g.y[~te], M.rows(g.X, te)
    fit = logreg.fit(Xtr, M.labels(ytr), M.C_GRID)
    sc = StandardScaler().fit(Xtr.cpu().numpy())
    Ztr, Zte = sc.transform(Xtr.cpu().numpy()), sc.transform(Xte.cpu().numpy())
    agree, gap = [], []
    for c in M.C_GRID:
        sk = LogisticRegression(C=c, class_weight="balanced", max_iter=100000,
                                tol=1e-8).fit(Ztr, ytr)
        agree.append(float(np.mean(sk.predict(Zte) == fit.predict(Xte, c))))
        Zt, yt = logreg.to_device(Ztr), M.labels(ytr)
        sw = logreg.balanced_weights(yt)
        f_sk = float(logreg.objective(Zt, yt, sw, c, logreg.to_device(sk.coef_),
                                      logreg.to_device(sk.intercept_)))
        f_gpu = float(logreg.objective(Zt, yt, sw, c, fit.W[c], fit.b[c]))
        gap.append(f_gpu - f_sk)
    ok &= check("GPU と sklearn（基準 1e-8）：保留側の予測が一致し，目的関数の差が 1e-6 以下",
                min(agree) >= 0.99 and max(abs(x) for x in gap) <= 1e-6,
                f"一致率 {[round(a, 4) for a in agree]}，目的関数の差 {[f'{x:.1e}' for x in gap]}"
                f"，勾配 {[f'{fit.grad[c]:.1e}' for c in M.C_GRID]}")

    # ---- 束ねて解く（map.py が使う）：1 問ずつ解いたものと同じ答え（交差検証 3 袋＋全体 × C 4 つ）
    yt = M.labels(ytr)
    folds = M.cv_folds(g.person[~te], np.ones(len(ytr), bool), 0)
    masks = [folds != j for j in range(M.N_CV_FOLDS)] + [np.ones(len(ytr), bool)]
    ref = [logreg.fit(M.rows(Xtr, m), M.rows(yt, m), M.C_GRID) for m in masks]
    bat = logreg.fit_batch(Xtr, yt, masks, M.C_GRID)
    agree, gap, grads = [], [], []
    for m, r, b in zip(masks, ref, bat):
        Xm, ym = M.rows(Xtr, m), M.rows(yt, m)
        Z, sw = (Xm - r.mu) / r.sd, logreg.balanced_weights(ym)
        for c in M.C_GRID:
            agree.append(float(np.mean(r.predict(Xte, c) == b.predict(Xte, c))))
            Wb, bb = b.moved_to(r.mu, r.sd, c)
            gap.append(float(logreg.objective(Z, ym, sw, c, Wb, bb))
                       - float(logreg.objective(Z, ym, sw, c, r.W[c], r.b[c])))
            grads.append(b.grad[c])
    ok &= check("束ねて解く：1 問ずつと予測が一致し，目的関数の差が 1e-6 以下，全問が止める基準に届く",
                min(agree) >= 0.99 and max(abs(x) for x in gap) <= 1e-6
                and max(grads) <= logreg.TOL,
                f"一致率の最小 {min(agree):.4f}，目的関数の差の最大 {max(abs(x) for x in gap):.1e}，"
                f"勾配の最大 {max(grads):.1e}")

    # ---- 温め直し：写した出発点は同じ出力を持ち，解き直した結果は冷えた出発点と同じ
    a, b_ = np.flatnonzero(~te)[::2], np.flatnonzero(~te)
    f_a = logreg.fit_batch(M.rows(g.X, a), M.labels(g.y[a]), [np.ones(len(a), bool)],
                           M.C_GRID)[0]
    Xb, yb, all_b = M.rows(g.X, b_), M.labels(g.y[b_]), [np.ones(len(b_), bool)]
    f_b = logreg.fit_batch(Xb, yb, all_b, M.C_GRID)[0]
    W0, b0 = f_a.moved_to(f_b.mu, f_b.sd, 1.0)
    same = torch.allclose(((Xte - f_b.mu) / f_b.sd) @ W0.T + b0, f_a.logits(Xte, 1.0),
                          rtol=0, atol=1e-8)
    f_w = logreg.fit_batch(Xb, yb, all_b, M.C_GRID, [f_a])[0]
    agree = [float(np.mean(f_w.predict(Xte, c) == f_b.predict(Xte, c))) for c in M.C_GRID]
    dW = [float((f_w.W[c] - f_b.W[c]).abs().max()) for c in M.C_GRID]
    ok &= check("温め直し：写した出発点の出力は元と同じ，解は冷えた出発点と同じ",
                same and min(agree) >= 0.99,
                f"予測の一致率 {[round(x, 4) for x in agree]}，W の差の最大 {[f'{x:.1e}' for x in dW]}，"
                f"反復 冷 {sum(f_b.n_iter.values())}／温 {sum(f_w.n_iter.values())}")

    # ---- 人工データで学習の通し：よく分かれた 4 群なら UAR は 1 に近い
    rng = np.random.default_rng(0)
    means = rng.normal(0, 3, size=(M.N_CLASSES, 8))
    y = np.repeat(np.arange(M.N_CLASSES), 60)
    X = logreg.to_device(means[y] + rng.normal(size=(len(y), 8)))
    person = np.array([f"s:{i % 12}" for i in range(len(y))])
    tr = person != "s:0"
    t = M.train(M.rows(X, tr), y[tr], person[tr], np.ones(int(tr.sum()), bool), 0)
    row = M.score(t, M.rows(X, ~tr), y[~tr], cached=False)
    ok &= check("人工データの通し", row["uar"] > 0.95 and row["uar_oracle"] >= row["uar"]
                and t.rule in ("cv", "tie"),
                f"UAR {row['uar']:.3f}，C={row['C']}（{row['C_rule']}）")

    print("\nすべて通った" if ok else "\n落ちたものがある")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
