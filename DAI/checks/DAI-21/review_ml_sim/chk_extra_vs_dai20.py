"""ML の査読者の検算（DAI-21）：台本の combine_all（分類器の数に依らない形）が，実験 20 の台本の add_extra と同じ判定を作るか．

台本の check3 が照らすのは，実験 16 の保存にある 38 の版（積，表の版，格子の点）だけである．実験 20 で足した相手
（確率の平均 avg，確信度の平均 conf，温度の較正 tcal，K 本で合わせた重み temp・tempb，重みの表を探す版 wf10・wf100・wf1000，
相性をソースの中の値から取る版 M_F6h）は，実験 16 の保存に無いので check3 に入らない．
ここでは，実験 16 の下調べの単位（3 つの分類器）に，実験 20 の add_extra と台本の combine_all を両方当てて，判定と重みを比べる．
UAR は画面に出さない（判定どうしの一致だけを見る）．
使い方：CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 .venv/bin/python <この台本>
"""
import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
spec = importlib.util.spec_from_file_location("dai21", "/home/snics/MER-DA/DAI/docs/tools_pending/dai21_eval5.py")
D21 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D21)
P20, P16, GRID = D21.P20, D21.P16, D21.GRID
P20.KS = (5, 25)                                    # 実験 16 の下調べの保存は K = 5・25
ROOT = D21.ROOT
VERS = ["avg", "conf", "tcal", "temp", "tempb", "wf10", "wf100", "wf1000"] + [f"M_F6h_mix{i}" for i in range(len(GRID)) if GRID[i][1] == 1.0]
for name in ("AFEW_IEMOCAP_f0_r0", "MELD_AFEW_f1_r1"):
    z = np.load(ROOT / "results" / "dai16_probe" / "units" / f"{name}.npz", allow_pickle=False)
    a20 = {k: z[k] for k in z.files}
    r20 = []
    P20.add_extra(a20, r20, {K: {} for K in P20.KS}, {K: (1.0, 1.0, 1.0) for K in P20.KS})
    yt, yp = z["y_test"].astype(int), z["y_pool"].astype(int)
    for K in P20.KS:
        kpos, fold = z[f"kpos_K{K}"], z[f"kfold_K{K}"].astype(int)
        g = lambda pre: [z[f"{pre}_{m}_K{K}"].astype(np.float64) for m in "AVT"]  # noqa: E731
        a21, r21 = {}, []
        D21.combine_all(a21, r21, K, g("p_test_ret"), g("p_pool_ret"), g("pcf"), g("pcf2"), g("ppj"), kpos, fold, yp[kpos], yt, yp,
                        z[f"D_M_K_K{K}"], z[f"D_M_orc_K{K}"], z["SR_M_ext"], z["SR_M_home"], D21.tables_of(3))
        ag = {v: min(float((a21[f"pred_{w}_K{K}_{v}"] == a20[f"pred_{w}_K{K}_{v}"]).mean()) for w in ("raw", "adj", "pool")) for v in VERS}
        wdev = max(float(np.abs(a21[f"W_{v}_K{K}"] - a20[f"W_{v}_K{K}"]).max()) for v in ("tcal", "temp", "tempb"))
        wfdev = max(float(np.abs(a21[f"W_wf{m}_K{K}"] - a20[f"W_wf{m}_K{K}"]).max()) for m in D21.WF_MS)
        uadev = float(np.abs(a21[f"wf_uar_K{K}"].astype(float) - a20[f"wf_uar_K{K}"].astype(float)).max())
        tied = all((a21[f"wf_tied{m}_K{K}"].shape == a20[f"wf_tied{m}_K{K}"].shape) and bool((a21[f"wf_tied{m}_K{K}"] == a20[f"wf_tied{m}_K{K}"]).all()) for m in D21.WF_MS)
        worst = min(ag, key=ag.get)
        print(f"{name} K={K}：版 {len(VERS)}．判定の一致の最小 {min(ag.values()):.6f}（{worst}）．完全に一致した版 {sum(v == 1.0 for v in ag.values())}．"
              f"合わせた重みの差の最大 {wdev:.1e}．選んだ表の差の最大 {wfdev:.1e}．K 本の UAR の差の最大 {uadev:.1e}．並んだ表の番号が同じ {tied}", flush=True)
