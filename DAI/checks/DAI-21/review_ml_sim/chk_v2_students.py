"""ML の査読者の検算（DAI-21 第 2 版）：4 コーパスの学び直しの道（台本の unit_students4 の第 2 版）を，仮のターゲットの回の単位で CPU で通す．

第 2 版で足した計算：疑似ラベルを替えた学び直し 5 つ，実験 1 の 3 手法（温度つき蒸留 kd，一致した発話だけ agree，混ぜる方法 mix）に積と提案の疑似ラベルを渡した問，
重みの保存．実験 20 の出力と実験 1 の評価の回の保存は開けないので，同じ形の保存を持つ仮のターゲットの回を使う
（実験 1 の仮のターゲットの回の保存 ＋ 実験 16 の下調べの単位）．台本の関数 teachers_of・student_specs・fit_students・save_students をそのまま呼ぶ．
見る量は判定どうしの一致だけである（UAR は画面に出さない）．
 (1) 積の疑似ラベルを渡した問が，実験 1 の保存の判定を再現するか：今の提案（joint）と 3 手法（kd・agree・mix）．
 (2) 勾配の最大，止まりの数，問の数，重みの形．
 (3) 温度の較正の疑似ラベルの判定が，実験 20 の add_extra の判定と一致するか（1 単位）．
 (4) 疑似ラベルを替えると学び直した判定がどれだけ変わるか（テストの発話の割合．ラベルを使わない）．
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
dai1, dai3, P16, P20, S, DD, GRID, SGRID = D21.dai1, D21.dai3, D21.P16, D21.P20, D21.S, D21.DD, D21.GRID, D21.SGRID
dai1.ENC["enc"] = DD.ENC1
ROOT, OUT1 = D21.ROOT, D21.OUT1
fz1, _ = dai3.load_fz1(OUT1)
GM = D21.sgrid_index(1.0, 1)
P20.KS = (5, 25)
for name in ("MELD_AFEW_f1_r1", "AFEW_IEMOCAP_f0_r0"):
    T, c, f, r = name.split("_")
    f, r = int(f[1:]), int(r[1:])
    sv = dai3.Saved(OUT1, "pseudo", T, c, f, r, fz1[T])
    u = sv.u
    z = np.load(ROOT / "results" / "dai16_probe" / "units" / f"{name}.npz", allow_pickle=False)
    assert (z["sid_test"].astype(str) == sv.sid).all()
    TK = dai1.build_tk(u, sv.p1, sv.p2, sv.teth)
    a20 = None
    if name == "MELD_AFEW_f1_r1":                                               # 温度の較正の判定と重みを，実験 20 の add_extra で作る（候補が 632 本の単位だけ）
        a20 = {k: z[k] for k in z.files}
        P20.add_extra(a20, [], {K: {} for K in P20.KS}, {K: (1.0, 1.0, 1.0) for K in P20.KS})
    for K in (5, 25):
        t0 = time.time()
        kpos = u.kpos[K]
        assert (z[f"kpos_K{K}"] == kpos).all()
        nk = np.setdiff1d(np.arange(u.n), kpos)
        yk = u.ypool[kpos]
        i = 2
        LMp = np.stack([P16.centered(z[f"p_pool_ret_{m}_K{K}"].astype(np.float64)) for m in "AVT"])
        LMh = np.stack([P16.centered(z[f"pcf_{m}_K{K}"].astype(np.float64)) for m in "AVT"])
        w_tcal = a20[f"W_tcal_K{K}"] if a20 is not None else P20.temp_w(LMh, yk, "sep")
        teach = D21.teachers_of(LMp, z[f"D_M_K_K{K}"], z["SR_M_ext"], K, i, w_tcal)
        ag_tcal = float((teach["tcal"][nk].argmax(1) == a20[f"pred_pool_K{K}_tcal"].astype(int)).mean()) if a20 is not None else float("nan")
        cw, cb = sv.teth[("AVT", K)]
        W0, b0 = u.start["AVT"]
        ex1 = {m_: {k_: v_ for k_, v_ in fz1[T]["select"][f"AVT|{K}|{m_}|one"].items() if k_ != "pseudo_uar"} for m_ in D21.EX1}
        specs = D21.student_specs(GM, ex1)
        out, Wst, bst, gmax, stalled = D21.fit_students(u.Xp["AVT"], u.Xt["AVT"], W0, b0, cw, cb, S.balanced_onehot(u.ypool, kpos, u.n), u.n, TK[K]["none_AVT"][0], teach, specs,
                                                        f"{name} K={K}")
        arr, rows = {}, []
        D21.save_students(arr, rows, K, out, (Wst, bst), specs, teach, u.ytest, u.ypool, nk)
        bp = P20.b_preds(sv, K)                                                 # 実験 1 の保存の判定（仮のターゲットの回．固定した設定の問）
        saved = {"joint": z[f"pred_raw_K{K}_joint_AVT"].astype(int), **{m_: np.asarray(bp[f"b_{m_}"]).astype(int) for m_ in D21.EX1}}
        nt = len(saved["joint"])
        rep = {"joint": int((out[f"stud_prod_g{GM}"][0] != saved["joint"]).sum()), **{m_: int((out[f"{m_}_prod"][0] != saved[m_]).sum()) for m_ in D21.EX1}}
        dif = {t_: 100 * float((out[f"stud_{t_}_g{GM}"][0] != out[f"stud_prod_g{GM}"][0]).mean()) for t_ in ("prop",) + D21.EXTRA_T}
        difm = {m_: 100 * float((out[f"{m_}_prop"][0] != out[f"{m_}_prod"][0]).mean()) for m_ in D21.EX1}
        print(f"{name} K={K}：候補 {u.n}，テスト {nt}，問 {len(specs)}，重み {Wst.shape} {Wst.dtype}，保存の配列 {len(arr)}，行 {len(rows)}．勾配の最大 {gmax:.1e}，止まり {stalled}．"
              f"実験 1 の保存と違うテストの発話の数（積の疑似ラベルを渡した問）{rep}．設定 {ex1}．温度の較正の疑似ラベルの一致 {ag_tcal:.4f}．"
              f"学び直した判定が積の疑似ラベルの版と違うテストの発話（%）{ {k_: round(v_, 1) for k_, v_ in dif.items()} }．3 手法で疑似ラベルを替えると違う発話（%）{ {k_: round(v_, 1) for k_, v_ in difm.items()} }．"
              f"{time.time() - t0:.0f} 秒", flush=True)
