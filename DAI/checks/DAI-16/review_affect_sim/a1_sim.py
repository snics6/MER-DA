# 人工の数だけの検算（MER-DA のデータと結果は読まない）．台本 dai16_probe.py の表の形と読み替えの式を書き写した．
import numpy as np
EPS = 1e-9
def table_from(P, y):
    C = np.eye(4)
    for i in range(4):
        C[i] = P[y == i].mean(0)
    return C
def shape_S(C):
    d = np.mean(np.diag(C)); S = np.full((4, 4), (1 - d) / 3); np.fill_diagonal(S, d); return S
def shape_D(C):
    d = np.diag(C); out = np.repeat(((1 - d) / 3)[:, None], 4, axis=1); np.fill_diagonal(out, d); return out
def post(C):
    Ce = C + EPS; Ce = Ce / Ce.sum(1, keepdims=True); return (Ce / Ce.sum(0, keepdims=True)).T
def combine(Ps, Cs):
    L = sum(np.log(np.clip(P @ post(C), 1e-300, None)) for P, C in zip(Ps, Cs))
    L -= L.max(1, keepdims=True); q = np.exp(L); return q / q.sum(1, keepdims=True)
def uar(y, p):
    return 100 * np.mean([(p[y == i] == i).mean() for i in range(4)])
def share(p):
    return np.round(np.bincount(p, minlength=4) / len(p), 3)
def softmax(Z):
    Z = Z - Z.max(1, keepdims=True); E = np.exp(Z); return E / E.sum(1, keepdims=True)

def run(seed, bias, sep=(1.2, 1.2, 1.2)):
    rng = np.random.default_rng(seed)
    out = {}
    def gen(n):
        y = rng.integers(0, 4, n)
        # 3 つのモダリティは同じ作り（同じ分かれ方 sep，同じ切片のずれ bias）．本当の感情を与えると互いに独立
        Ps = [softmax(s * np.eye(4)[y] + rng.normal(size=(n, 4)) + np.array(bias)) for s in sep]
        return y, Ps
    yp, Pp = gen(6000)      # 候補（表を数える）
    yt, Pt = gen(6000)      # テスト
    I = [np.eye(4)] * 3
    q0p, q0t = combine(Pp, I), combine(Pt, I)
    C = [table_from(P, yp) for P in Pp]
    res = {"prod": q0t.argmax(1),
           "orc_S": combine(Pt, [shape_S(c) for c in C]).argmax(1),
           "orc_D": combine(Pt, [shape_D(c) for c in C]).argmax(1),
           "orc_F": combine(Pt, C).argmax(1)}
    # 対照 1：積 1 つの表（掛け合わせた後に読み替える）
    CP = table_from(q0p, yp)
    res["prodtab_D"] = (q0t @ post(shape_D(CP))).argmax(1)
    res["prodtab_F"] = (q0t @ post(CP)).argmax(1)
    # 対照 2：積の対数に感情ごとの切片 4 つだけを足す（候補の全ラベルで，候補の UAR が最大になる切片を座標ごとに探す）
    L = np.log(np.clip(q0p, 1e-300, None)); d = np.zeros(4)
    grid = np.linspace(-6, 6, 61)
    for _ in range(4):
        for k in range(4):
            best = max(grid, key=lambda g: uar(yp, (L + np.where(np.arange(4) == k, g, d)).argmax(1)))
            d[k] = best
    res["prod_bias4"] = (np.log(np.clip(q0t, 1e-300, None)) + d).argmax(1)
    return {k: (uar(yt, v), share(v)) for k, v in res.items()}, [np.round(np.diag(c), 2) for c in C]

def main():
  for name, bias in [("Neutral を答えにくい切片のずれ（3 モダリティ共通）", (-1.5, 0, 0, 0.5)), ("ずれ無し", (0, 0, 0, 0))]:
      print("==", name)
      acc = {}
      for seed in range(5):
          r, diag = run(seed, bias)
          for k, (u, s) in r.items():
              acc.setdefault(k, []).append(u)
          if seed == 0:
              print(" 表の対角（N,H,S,A）モダリティ 1:", diag[0])
              for k, (u, s) in r.items():
                  print(f"  {k:10s} UAR {u:5.1f}  判定の割合 {s}")
      print(" 5 回の平均の差（− prod）:", {k: round(np.mean(v) - np.mean(acc['prod']), 2) for k, v in acc.items() if k != 'prod'})

if __name__ == '__main__':
    main()
