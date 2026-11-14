"""感情認識の査読者の検算（DAI-20．2026-11-14）：ソースが 3 つのときの信頼度の表 ext と，固定した設定で作られる重みの表の例．

読むのは下調べ（仮のターゲットの回）の集計だけである：
  results/dai16_probe/summary/reliability_src.csv（ソースのラベルだけで作った表：コーパスの中の交差検証 home と，別のコーパスに当てた away）
  results/dai16_probe/summary/reliability.csv，results/dai16_probe_k10/summary/reliability.csv（仮のターゲット c の K 本の表 M_K と全ラベルの表 M_orc の単位の平均）
評価の回の出力（results/dai1/eval/，results/dai1/stats/，results/dai2/stats/）は読まない．GPU は使わない．
式は DAI/exp/dai16_probe.py の source_stats・mix_rel・to_w と同じ（ret = 向きの away の合計 ÷ home の合計を 0〜1.5 に切る．ext = home × ret．M_ext はソースの平均．
水準 = (n·L_K + n0·L_0)/(n + n0)，信頼度 = max(0, 水準 + β·相性)，重み = 信頼度 ÷ 12 個の平均）．
使い方：リポジトリの直下で  .venv/bin/python DAI/checks/DAI-20/review_affect_sim/ext3.py
"""
import itertools
import numpy as np
import pandas as pd

C = ["AFEW", "IEMOCAP", "MOSEI", "MELD"]
MODS = "AVT"
EMO = ["Neutral", "Happy", "Sad", "Angry"]
COLS = [f"d_{j}" for j in range(4)]
FIX = {("AFEW", 5): (20, 1, "eq"), ("AFEW", 10): (0, 1, "eq"), ("AFEW", 25): (20, 1, "eq"),
       ("IEMOCAP", 5): (0, 1, "eq"), ("IEMOCAP", 10): (0, 1, "eq"), ("IEMOCAP", 25): (0, 1, "eq"),
       ("MOSEI", 5): (0, 1, "eq"), ("MOSEI", 10): (0, 1, "eq"), ("MOSEI", 25): (0, 1, "eq"),
       ("MELD", 5): (20, 1, "ext"), ("MELD", 10): (20, 1, "ext"), ("MELD", 25): (0, 1, "eq")}   # 設計 3 節の表

r = pd.read_csv("results/dai16_probe/summary/reliability_src.csv")
home = {(s, m): r[(r.source == s) & (r["where"] == "home") & (r.modality == m)][COLS].to_numpy()[0] for s in C for m in MODS}
away = {(s, t, m): r[(r.source == s) & (r["where"] == f"away→{t}") & (r.modality == m)][COLS].to_numpy()[0] for s in C for t in C if s != t for m in MODS}
rel = pd.concat([pd.read_csv("results/dai16_probe/summary/reliability.csv"), pd.read_csv("results/dai16_probe_k10/summary/reliability.csv")], ignore_index=True)


def tab(c, K, src):
    q = rel[(rel.c == c) & (rel.K == K) & (rel.source == src)].set_index("modality")
    return np.stack([q.loc[m, COLS].to_numpy(float) for m in MODS])


def ext(srcs):
    pairs = [(s, t) for s in srcs for t in srcs if s != t]
    ret = {m: float(np.clip(sum(away[(s, t, m)].sum() for s, t in pairs) / sum(home[(s, m)].sum() for s, t in pairs), 0, 1.5)) for m in MODS}
    return np.stack([np.mean([home[(s, m)] * ret[m] for s in srcs], 0) for m in MODS]), ret


def prof(M):
    return M - M.mean(1, keepdims=True)


def order(v):
    return "".join(np.array(list(MODS))[np.argsort(-v)])


np.set_printoptions(precision=3, suppress=True, linewidth=160)
print("行は声 A・顔 V・言葉 T，列は", EMO)
for T in C:
    srcs = [c for c in C if c != T]
    M3, ret3 = ext(srcs)
    print(f"\n=== 評価のターゲット T = {T}．ソース {srcs}．残り方 " + " ".join(f"{m}={ret3[m]:.3f}" for m in MODS))
    print("ext（3 ソース）"); print(M3)
    print("  水準", M3.mean(1), " 相性（行の中の差）"); print(prof(M3))
    print("  どの分類器にも共通の感情ごとの成分（列の平均 − 全体の平均）", M3.mean(0) - M3.mean())
    M2s = [ext(list(p))[0] for p in itertools.combinations(srcs, 2)]
    print("  下調べの 2 ソースの表 3 通りの平均との差の最大", round(float(np.abs(M3 - np.mean(M2s, 0)).max()), 3),
          "．2 ソースの表どうしの相性の差の最大", round(float(max(np.abs(prof(a) - prof(b)).max() for a, b in itertools.combinations(M2s, 2))), 3))
    for drop in srcs:
        M2, _ = ext([s for s in srcs if s != drop])
        d = prof(M3) - prof(M2)
        print(f"  {drop} を除いた表との相性の差：最大の絶対値 {np.abs(d).max():.3f}，Neutral の列 {d[:, 0]}，水準（除いた後）{M2.mean(1)}")
    Th = np.stack([home[(T, m)] for m in MODS])
    print(f"  水準の順（強い順）：ext {order(M3.mean(1))}，{T} の中の交差検証 {order(Th.mean(1))}")
    for K in (5, 10, 25):
        n0, beta, prior = FIX[(T, K)]
        DK, Dorc = tab(T, K, "M_K"), tab(T, K, "M_orc")
        LK = DK.mean(1)
        L0 = np.full(3, LK.mean()) if prior == "eq" else M3.mean(1)
        n = 4 * K
        L = (n * LK + n0 * L0) / (n + n0)
        D = np.maximum(0, L[:, None] + beta * prof(M3))
        W, Wo = D / D.mean(), Dorc / Dorc.mean()
        print(f"  -- K = {K}．設定 n0 = {n0}，β = {beta}，戻し先 {prior}．K 本の水準（下調べの単位の平均）{LK} → 使う水準 {L}．全ラベルの水準 {Dorc.mean(1)}")
        print("     固定した設定の重み"); print(W)
        print("     全ラベルで測った表の重み（参照）"); print(Wo)
        print(f"     重みの差の絶対値の最大 {np.abs(W - Wo).max():.2f}（{MODS[np.unravel_index(np.abs(W - Wo).argmax(), W.shape)[0]]}・{EMO[np.unravel_index(np.abs(W - Wo).argmax(), W.shape)[1]]}），"
              f"相性（行の中の差）の相関 {np.corrcoef(prof(W).ravel(), prof(Wo).ravel())[0, 1]:.2f}，行ごとの相関 " + " ".join(f"{m}={np.corrcoef(prof(W)[i], prof(Wo)[i])[0, 1]:+.2f}" for i, m in enumerate(MODS)))
        a, b = prof(M3), prof(Dorc)
        print(f"     ソースで測った表の相性と全ラベルの表の相性の相関：12 個 {np.corrcoef(a.ravel(), b.ravel())[0, 1]:+.2f}，顔の行を除く 8 個 {np.corrcoef(a[[0, 2]].ravel(), b[[0, 2]].ravel())[0, 1]:+.2f}．"
              f"言葉の行の相性の幅（最大 − 最小）：ソースの表 {np.ptp(a[2]):.2f}，全ラベルの表 {np.ptp(b[2]):.2f}")
