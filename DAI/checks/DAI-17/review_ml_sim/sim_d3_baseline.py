"""D3 の基準の人工の検算（MER-DA のデータ・結果は読まない．CPU だけ）．
問い：ターゲットの発話の特徴が本当の感情と無関係でも，D3（近い 100 本の当たる割合 − 表の全体からでたらめに引いたときの当たる割合）は 2 点以上になりうるか．
作り：
  ソースの特徴 = 感情と無関係な成分（少数の向きに大きく広がる．話者・録音・話題に当たる）＋ 感情の成分（Neutral 以外の 3 感情だけが別々の向きにずれる）＋ 雑音．
  ターゲットの特徴 = 感情と無関係な成分 ＋ 雑音（感情の成分は無い．本当の感情は特徴と独立に，コーパスの割合で振る）．
  こうすると，Neutral のソースの発話は「感情と無関係な成分」の向きにだけ長さを持つので，どのターゲットの発話からも近くなりやすい（引かれる発話が Neutral に寄る）．
引き方と当たる割合の式は台本（dai17_probe.py の topk_idx，run_unit の rank・chance の行，stage_summary の D1・D3 の式）と同じにした．
"""
import sys
import numpy as np
import torch

sys.path.insert(0, sys.argv[1])
import dai17_probe as P17  # noqa: E402

PROP = {"AFEW": [0.259, 0.267, 0.224, 0.249], "IEMOCAP": [0.373, 0.322, 0.207, 0.098], "MOSEI": [0.464, 0.361, 0.097, 0.078], "MELD": [0.567, 0.203, 0.089, 0.142]}
NSRC = {"AFEW": 790, "IEMOCAP": 2943, "MOSEI": 6000, "MELD": 6000}        # 大きい 2 つは計算を軽くするため 6000 本に切った
D, R_NUIS = 64, 4


def feats(rng, y, load, emo_scale):
    n = len(y)
    X = rng.normal(size=(n, R_NUIS)) @ load * 2.0 + rng.normal(size=(n, D))
    if emo_scale > 0:
        E = np.zeros((n, D))
        for e in (1, 2, 3):
            E[y == e, D - e] = emo_scale                                       # 感情 e は専用の 1 つの向きにずれる
        X = X + E
    return X


def unit(Xs):
    Z = (Xs - Xs.mean(0)) / Xs.std(0)
    return torch.nn.functional.normalize(torch.as_tensor(Z, dtype=torch.float32), dim=1)


def one(c, srcs, emo_scale, seed):
    rng = np.random.default_rng(seed)
    load = rng.normal(size=(R_NUIS, D)); load[:, D - 3:] = 0.0                # 感情と無関係な成分の向き（感情の向きとは重ならない）
    db, ys = [], []
    for s in srcs:
        y = rng.choice(4, NSRC[s], p=np.array(PROP[s]) / sum(PROP[s]))
        db.append(unit(feats(rng, y, load, emo_scale))); ys.append(y)
    DB, yb = torch.cat(db), np.concatenate(ys)
    yt = rng.choice(4, 1500, p=np.array(PROP[c]) / sum(PROP[c]))               # 本当の感情は特徴と独立
    Q = unit(feats(rng, yt, load, 0.0))
    idx = P17.topk_idx(Q, DB, k=100)
    Y = yb[idx]
    hit = Y == yt[:, None]
    pt, pdb = np.bincount(yt, minlength=4) / len(yt), np.bincount(yb, minlength=4) / len(yb)
    q = np.bincount(Y.ravel(), minlength=4) / Y.size                          # 引かれた発話のラベルの割合
    h = lambda a, b: float(hit[:, a:b].mean())
    top100 = (5 * h(0, 5) + 15 * h(5, 20) + 80 * h(20, 100)) / 100
    D3 = 100 * (top100 - float(pt @ pdb))                                     # 台本の D3
    D3_indep = 100 * (top100 - float(pt @ q))                                 # 基準を「引かれた発話のラベルの割合」に替えた版
    bal = 100 * float(np.mean([hit[yt == e].mean() for e in range(4)])) - 25  # 本当の感情ごとの当たる割合の平均 − 25
    D1 = 100 * (h(0, 5) - h(20, 100))
    return D3, D3_indep, bal, D1, q[0], pdb[0]


for emo_scale in (0.0, 3.0, 5.0):
    print(f"\n感情のずれの大きさ（ソースだけ） = {emo_scale}：ターゲットの特徴は本当の感情と独立")
    print("  仮のターゲット  D3（台本の式）  基準を引かれたラベルの割合に替えた D3  感情ごとの当たる割合の平均−25   D1    引かれた Neutral の割合／表の Neutral の割合")
    n_ge2 = 0
    for c in PROP:
        out = []
        for srcs in [[a, b] for i, a in enumerate(x for x in PROP if x != c) for b in [x for x in PROP if x != c][i + 1:]]:
            for seed in range(3):
                out.append(one(c, srcs, emo_scale, 100 * seed + len(out)))
        m = np.mean(out, 0)
        n_ge2 += m[0] >= 2.0
        print(f"  {c:8s}       {m[0]:+6.2f}          {m[1]:+6.2f}                            {m[2]:+6.2f}                {m[3]:+6.2f}   {m[4]:.3f}／{m[5]:.3f}")
    print(f"  台本の D3 が 2 点以上のコーパス：{int(n_ge2)} / 4（3 以上なら「近いソースの発話は，でたらめに引いた発話より当たった」と書く規則）")
