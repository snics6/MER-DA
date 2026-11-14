"""ML の査読者の再検査（DAI-20 第 3 版）：ソースが 3 つのときの「ソースだけで決まる量」（dai16_probe.source_stats）が，評価の回の 4 通りのソースの組で
途中で止まらずに計算できるかを，評価の回の前に確かめる．CPU だけ．使うのはソースのコーパスの特徴とラベル，実験 1 の段 source の保存（results/dai1/source）だけで，
評価の回の保存（results/dai1/eval）・実験 1・2 の集計・results/dai18_probe は開かない．ターゲットのデータは使わない（source_stats はソースの組だけの関数）．
出すのは，所要，残り方，表 ext（感情認識の査読者の検算 ext3_out.txt の値と見比べる），ソースどうしで学ぶ重み（F8）の当てはめが通ったか，である．
使い方：CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 .venv/bin/python DAI/checks/DAI-20/review_ml_sim/v3_source3.py
"""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import numpy as np
import dai20_eval as E
P16, DD, dai1 = E.P16, E.DD, E.dai1
dai1.ENC["enc"] = DD.ENC1
np.set_printoptions(precision=3, suppress=True, linewidth=160)
for T in E.CORP:
    srcs = DD.others(T)
    t0 = time.time()
    try:
        SR = P16.source_stats(srcs)
    except SystemExit as e:
        print(f"T = {T}：止まった：{e}", flush=True)
        continue
    print(f"T = {T}．ソース {list(srcs)}．{time.time() - t0:.0f} 秒．残り方 {SR['ret']}．F8 の重みの範囲 {SR['F8'].min():.2f}〜{SR['F8'].max():.2f}．有限 {bool(np.isfinite(SR['F8']).all())}", flush=True)
    print("  ext（3 ソース）"); print(SR["M_ext"])
    print("  感情ごとの残り方に欠け", int(np.isnan(SR["ret_emo"]).sum()), "．S_ext の形", SR["S_ext"].shape, flush=True)
