"""ML の査読者の検算（DAI-21）：4 コーパスの学び直しの道（台本の unit_students4）を，仮のターゲットの回の単位で CPU で通す．

実験 20 の単位の出力と実験 1 の評価の回の保存は開けないので，同じ形の保存を持つ仮のターゲットの回を使う：
  実験 1 の仮のターゲットの回の保存（results/dai1/pseudo/units）＋ 実験 16 の下調べの単位（results/dai16_probe/units．実験 20 と同じ関数 run_unit の出力）．
台本の関数 teacher_probs・fit_students をそのまま呼ぶ．見る量は判定どうしの一致だけで，UAR は計算しない（テストのラベルを使わない）．
 (1) 保存の確率（float32）から作った積の疑似ラベルで学び直した判定が，実験 1 の保存の「今の提案」（pred_raw_K_joint_AVT）と一致する割合（台本の joint_repro と同じ量）．
 (2) メモリの上の float64 の確率（実験 1 の保存の重みから作り直した積）で学び直した判定との一致．疑似ラベルの確率の差の最大．
 (3) 勾配の最大と止まりの数（台本は 1e-5 を超えると止まる）．
 (4) 積の疑似ラベルと提案の疑似ラベルで学び直した 2 つの分類器の判定が違うテストの発話の割合（設定 12 点ごと．ラベルを使わない）．
使い方：CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 .venv/bin/python <この台本>
"""
import importlib.util
import sys
import time

import numpy as np

sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
spec = importlib.util.spec_from_file_location("dai21", "/home/snics/MER-DA/DAI/docs/tools_pending/dai21_eval5.py")
D21 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D21)
dai1, dai3, P16, S, DD, GRID, SGRID = D21.dai1, D21.dai3, D21.P16, D21.S, D21.DD, D21.GRID, D21.SGRID
dai1.ENC["enc"] = DD.ENC1
ROOT, OUT1 = D21.ROOT, D21.OUT1
fz1, _ = dai3.load_fz1(OUT1)
GM = D21.sgrid_index(1.0, 1)
for name in ("MELD_AFEW_f1_r1", "AFEW_IEMOCAP_f0_r0"):
    T, c, f, r = name.split("_")
    f, r = int(f[1:]), int(r[1:])
    sv = dai3.Saved(OUT1, "pseudo", T, c, f, r, fz1[T])
    u = sv.u
    z = np.load(ROOT / "results" / "dai16_probe" / "units" / f"{name}.npz", allow_pickle=False)
    assert (z["sid_test"].astype(str) == sv.sid).all()
    TK = dai1.build_tk(u, sv.p1, sv.p2, sv.teth)
    for K in (5, 25):
        t0 = time.time()
        kpos = u.kpos[K]
        assert (z[f"kpos_K{K}"] == kpos).all()
        nk = np.setdiff1d(np.arange(u.n), kpos)
        i = 2                                                                  # n0 = 0，相性を使う点
        LMp = np.stack([P16.centered(z[f"p_pool_ret_{m}_K{K}"].astype(np.float64)) for m in "AVT"])
        W = P16.to_w(P16.mix_rel(z[f"D_M_K_K{K}"], z["SR_M_ext"], 4 * K, *GRID[i]))
        q64 = dai1.product(TK[K], (1, 1, 1), 0).cpu().numpy().astype(np.float64)
        teach = {"prod": D21.teacher_probs(LMp, None), "prop": D21.teacher_probs(LMp, W), "prod64": q64}
        dq = float(np.abs(teach["prod"] - q64).max())
        cw, cb = sv.teth[("AVT", K)]
        W0, b0 = u.start["AVT"]
        out, gmax, stalled = D21.fit_students(u.Xp["AVT"], u.Xt["AVT"], W0, b0, cw, cb, S.balanced_onehot(u.ypool, kpos, u.n), u.n, teach, f"{name} K={K}")
        saved = z[f"pred_raw_K{K}_joint_AVT"].astype(int)
        rep = float((out[("prod", GM)][0] == saved).mean())
        rep64 = float((out[("prod64", GM)][0] == saved).mean())
        a6432 = min(float((out[("prod", g)][0] == out[("prod64", g)][0]).mean()) for g in range(len(SGRID)))
        dif = [float((out[("prod", g)][0] != out[("prop", g)][0]).mean()) for g in range(len(SGRID))]
        tdif = float((teach["prod"][nk].argmax(1) != teach["prop"][nk].argmax(1)).mean())
        print(f"{name} K={K}：候補 {u.n}，テスト {len(saved)}，引き戻し ({cw}, {cb})．今の提案の再現（float32 の確率から）{rep:.4f}，（float64 から）{rep64:.4f}．"
              f"float32 と float64 の学び直しの判定の一致の最小（12 点）{a6432:.4f}．疑似ラベルの確率の差の最大 {dq:.1e}．勾配の最大 {gmax:.1e}，止まり {stalled}．"
              f"疑似ラベルの判定が違う候補 {100 * tdif:.1f}%．学び直した判定が違うテストの発話：主の点 {100 * dif[GM]:.1f}%，12 点の最小〜最大 {100 * min(dif):.1f}〜{100 * max(dif):.1f}%．{time.time() - t0:.0f} 秒", flush=True)
