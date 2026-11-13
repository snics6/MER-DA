# 人工の数だけの検算その 2（MER-DA のデータと結果は読まない）
import numpy as np
from a1_sim import table_from, shape_S, shape_D, post, combine, uar, softmax
def hard_combine(Ps, Cs):
    L = sum(np.log(np.clip(post(C)[P.argmax(1)], 1e-300, None)) for P, C in zip(Ps, Cs))
    return L.argmax(1)
def one(seed, biases, seps, n=6000):
    """biases: モダリティごとの切片のずれ (3, 4)．seps: モダリティごと・感情ごとの分かれ方 (3, 4)"""
    rng = np.random.default_rng(seed)
    def gen(n):
        y = rng.integers(0, 4, n)
        Ps = [softmax(np.eye(4)[y] * np.array(s)[y][:, None] + rng.normal(size=(n, 4)) + np.array(b)) for s, b in zip(seps, biases)]
        return y, Ps
    yp, Pp = gen(n); yt, Pt = gen(n)
    I = [np.eye(4)] * 3
    q0p, q0t = combine(Pp, I), combine(Pt, I)
    C = [table_from(P, yp) for P in Pp]
    Ch = [table_from(np.eye(4)[P.argmax(1)], yp) for P in Pp]
    r = {"prod": uar(yt, q0t.argmax(1)),
         "orc_S": uar(yt, combine(Pt, [shape_S(c) for c in C]).argmax(1)),
         "orc_D": uar(yt, combine(Pt, [shape_D(c) for c in C]).argmax(1)),
         "orc_F": uar(yt, combine(Pt, C).argmax(1)),
         "orc_F_hard": uar(yt, hard_combine(Pt, Ch))}
    L = np.log(np.clip(q0p, 1e-300, None)); d = np.zeros(4); grid = np.linspace(-6, 6, 61)
    for _ in range(4):
        for k in range(4):
            d[k] = max(grid, key=lambda g: uar(yp, (L + np.where(np.arange(4) == k, g, d)).argmax(1)))
    r["prod_bias4"] = uar(yt, (np.log(np.clip(q0t, 1e-300, None)) + d).argmax(1))
    # 各モダリティに切片 4 つを足してから掛ける対照は prod_bias4 と同じ族（切片の和）なので省く
    return r
def rep(name, biases, seps):
    acc = {}
    for s in range(6):
        for k, v in one(s, biases, seps).items():
            acc.setdefault(k, []).append(v)
    base = np.mean(acc["prod"])
    print(f"{name}: prod {base:.1f} | " + ", ".join(f"{k} {np.mean(v) - base:+.1f}" for k, v in acc.items() if k != "prod"))
E = [1.2] * 4
rep("1 同じ作り・ずれ無し                  ", [[0, 0, 0, 0]] * 3, [E] * 3)
rep("2 同じ作り・弱い分かれ方（0.6）         ", [[0, 0, 0, 0]] * 3, [[0.6] * 4] * 3)
rep("3 同じ作り・共通の切片のずれ（N −0.5）   ", [[-0.5, 0, 0, 0]] * 3, [E] * 3)
rep("4 同じ作り・共通の切片のずれ（N −1.0）   ", [[-1.0, 0, 0, 0]] * 3, [E] * 3)
rep("5 モダリティで強い感情が違う・ずれ無し    ", [[0, 0, 0, 0]] * 3, [[2.5, 0.3, 0.3, 1.2], [0.3, 2.5, 1.2, 0.3], [1.2, 0.3, 2.5, 0.3]])
rep("6 モダリティの強さだけ違う・ずれ無し      ", [[0, 0, 0, 0]] * 3, [[2.0] * 4, [1.0] * 4, [0.2] * 4])
rep("7 5 に共通の切片のずれ（N −1.0）を足す   ", [[-1.0, 0, 0, 0]] * 3, [[2.5, 0.3, 0.3, 1.2], [0.3, 2.5, 1.2, 0.3], [1.2, 0.3, 2.5, 0.3]])
