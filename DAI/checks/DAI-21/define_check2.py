"""関門 1（定義）の確かめ 2：読むだけ．学び直しの重み（感情ごとに総量をそろえる）を通したとき，提案の疑似ラベルの切片のずらしがどれだけ残るか．
実験 16 の下調べの保存 2 単位で，(a) ずらしあり と ずらしなし の重みの差，(b) 提案（ずらしあり）と積 の重みの差を，L1 の和で比べる．"""
import sys
import numpy as np
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai16_probe as P16  # noqa: E402

def bal(P):
    P = P / P.sum(0, keepdims=True)
    return P / P.sum()

for name in ("AFEW_IEMOCAP_f0_r0", "MELD_AFEW_f1_r1"):
    z = np.load(f"/home/snics/MER-DA/results/dai16_probe/units/{name}.npz", allow_pickle=False)
    for K in (5, 25):
        LM = np.stack([P16.centered(z[f"p_pool_ret_{m}_K{K}"].astype(np.float64)) for m in "AVT"])
        prod = np.exp(P16.norm_log(LM.sum(0)))
        for i in (2, 6):
            W = P16.to_w(P16.mix_rel(z[f"D_M_K_K{K}"], z["SR_M_ext"], 4 * K, *P16.GRID[i]))
            Sp = P16.norm_log(np.einsum("ky,kny->ny", W, LM))
            dl, _ = P16.match_shift(Sp, prod.mean(0))
            q_adj, q_raw = np.exp(P16.norm_log(Sp + dl)), np.exp(Sp)
            a = np.abs(bal(q_adj) - bal(q_raw)).sum()
            b = np.abs(bal(q_adj) - bal(prod)).sum()
            a0 = np.abs(q_adj / q_adj.sum() - q_raw / q_raw.sum()).sum()
            arg = float((q_adj.argmax(1) != q_raw.argmax(1)).mean())
            print(f"{name} K={K} 点{i}：ずらしの差（そろえる前）{a0:.4f}，（そろえた後）{a:.4f}，提案−積（そろえた後）{b:.4f}，判定が変わる割合 {arg:.4f}，δ={np.round(dl,3)}")
