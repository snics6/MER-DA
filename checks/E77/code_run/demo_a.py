"""E77 関門 1 の B：(a) に分けた解析を，試走の出力（audio_wavlm，RAVDESS，反復 0，仮のターゲット AFEW の反復 0）で実際に計算する。"""
import numpy as np, pandas as pd
d, T = "checks/E77/code_run/kind_audio", "RAVDESS"
a = pd.read_csv(f"{d}/{T}.csv", dtype={"K": str, "meff": str, "param": str})
P = np.load(f"{d}/{T}_r0_probs.npy").astype(np.float64); te = np.load(f"{d}/{T}_r0_te.npy")
sm = pd.read_csv(f"{d}/{T}_samples.csv")
man = pd.read_csv("data/metadata/ravdess_utterance_manifest.csv").set_index("sample_id")
y = man.loc[sm.sample_id[te], "label_4"].astype(int).to_numpy()
spk = sm.speaker_group.to_numpy()[te]
def uar(y, p): return np.mean([np.mean(p[y == c] == c) for c in range(4) if (y == c).any()])
rid = lambda m, p, e, K: int(a.query("rep == 0 and method == @m and param == @p and meff == @e and K == @K").row_id.iloc[0])
i0, i1 = rid("none", "-", "0", "5"), rid("entmin", "-", "500", "5")
# 1. 話者を単位にしたブートストラップ（反復 0 だけ．2 段目の反復の抽出は 10 反復がそろえば同じ形）
rng = np.random.default_rng(0); sp = np.unique(spk); D = []
for _ in range(1000):
    pick = rng.choice(sp, len(sp)); idx = np.concatenate([np.flatnonzero(spk == s) for s in pick])
    D.append(uar(y[idx], P[i1][idx].argmax(1)) - uar(y[idx], P[i0][idx].argmax(1)))
print(f"1. entmin(m=500) − none の UAR の差（K=5，反復 0）：点 {uar(y, P[i1].argmax(1)) - uar(y, P[i0].argmax(1)):+.4f}，話者 {len(sp)} 人のブートストラップの 95% 区間 [{np.percentile(D, 2.5):+.4f}, {np.percentile(D, 97.5):+.4f}]")
# 2. 性別ごとの UAR（RAVDESS は俳優番号の奇数が男性．発話一覧に性別の列は無いので俳優番号から作る）
actor = man.loc[sm.sample_id[te], "speaker_id"].str.extract(r"(\d+)")[0].astype(int).to_numpy(); male = actor % 2 == 1
for i, nm in ((i0, "none"), (i1, "entmin")):
    print(f"2. {nm} K=5 の性別ごとの UAR：男性 {uar(y[male], P[i][male].argmax(1)):.4f}（{male.sum()} 発話），女性 {uar(y[~male], P[i][~male].argmax(1)):.4f}（{(~male).sum()} 発話）")
# 3. 感情の対ごとの点数の差（対数確率の差）から d と o を出す．float16 の 0 が −inf を作る数
L = np.log(np.load(f"{d}/{T}_r0_probs.npy").astype(np.float64))
print("3. 対数確率が −inf になる要素の数（float16 で下に溢れた確率）：", int(np.isinf(L).sum()), "／ 行のうち 1 つ以上ある行", int(np.isinf(L).any((1, 2)).sum()), "/", len(L))
for (j, k) in [(0, 1), (2, 3)]:
    s = L[i0][:, j] - L[i0][:, k]; mj, mk = s[y == j].mean(), s[y == k].mean(); sd = np.sqrt((s[y == j].var() + s[y == k].var()) / 2)
    print(f"   none K=5 の対 ({j},{k})：d = {(mj - mk) / sd:.3f}，o = {(mj + mk) / 2 / sd:+.3f}")
# 4. 選択（仮のターゲットの UAR の平均が最大．none も候補）と取りこぼし（テストで最良 − 選んだもの）
p = pd.read_csv(f"{d}/{T}_pseudo.csv", dtype={"K": str, "meff": str, "param": str})
for meth in ("im", "lp", "t3a"):
    for K in ("5",):
        sc = p[(p.K == K) & p.method.isin([meth, "none"])].groupby(["method", "param", "meff"]).uar.mean()
        best = sc.idxmax(); t = a[(a.K == K) & a.method.isin([meth, "none"])].set_index(["method", "param", "meff"]).uar
        print(f"4. {meth} K={K}：選んだ {best}，テストの値 {t[best]:.4f}，テストで最良 {t.max():.4f}，取りこぼし {t.max() - t[best]:.4f}，候補の数 {len(sc)}")
