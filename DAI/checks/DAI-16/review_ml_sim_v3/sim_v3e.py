"""ML の査読役の検算 5：第 4 版（コミット 6432e2e）の R0 の量（dcenter の相関と，感情の並べ替えの割合 perm_share）を，
「水準 × 感情の当てやすさ」だけでできた人工の表に当てる．信頼度 d′ は 556a06f と 6432e2e で同じ関数（確かめる）．"""
import sys, importlib.util, inspect
import numpy as np
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
SP = "/tmp/claude-1000/-home-snics-MER-DA/76085a37-4610-4922-b8ae-ced34f163c39/scratchpad/rev_ml_v3"
def load(name, path):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
C = load("p16c", SP + "/dai16_probe_556a06f.py"); V4 = load("p16v4", SP + "/dai16_probe_6432e2e.py")

def draw(rng, n, sep):
    y = rng.integers(0, 4, n)
    Ps = []
    for k in range(sep.shape[0]):
        Z = rng.normal(size=(n, 4)); Z[np.arange(n), y] += sep[k][y]
        Ps.append(C.softmax(Z))
    return y, Ps

def dc_mul(D):
    return D / (D.mean(1, keepdims=True) * D.mean(0, keepdims=True) / D.mean()) - 1.0

def run(name, sepS, sepT, n, reps=40):
    ia, ps, im, ok = [], [], [], []
    for rep in range(reps):
        rng = np.random.default_rng(7000 + rep)
        ys, Ps = draw(rng, n, sepS); yt, Pt = draw(rng, n, sepT)
        Ds = np.stack([C.dprime(p, ys)[0] for p in Ps]); Dt = np.stack([C.dprime(p, yt)[0] for p in Pt])
        ia.append(V4.corr(V4.dcenter(Ds), V4.dcenter(Dt))); ps.append(V4.perm_share(Ds, Dt)); im.append(V4.corr(dc_mul(Ds), dc_mul(Dt)))
        ok.append(ia[-1] >= 0.3 and ps[-1] <= 0.2)
    print(f"{name}（1 つの表を測る発話 {n} 本）：第 4 版の R0 の相関 {np.mean(ia):+.2f}，並べ替えの割合 {np.mean(ps):.3f}，1 回ごとに「0.3 以上かつ 0.2 以下」を満たした割合 {np.mean(ok):.2f}，掛け算の形の相関 {np.mean(im):+.2f}")

e = np.array([0.5, 0.9, 1.4, 2.0]); col = lambda a: np.array(a)[:, None]
for n in (20000, 2000):
    run("場合 A（水準 × 感情の当てやすさ．水準の順は同じ）", col([1.4, 1.0, 0.6]) * e, col([1.3, 1.0, 0.7]) * e, n)
    run("場合 A′（水準の差が小さい：1.1, 1.0, 0.9）", col([1.1, 1.0, 0.9]) * e, col([1.1, 1.0, 0.9]) * e, n)
    run("場合 B（水準の順が逆）", col([1.4, 1.0, 0.6]) * e, col([0.6, 1.0, 1.4]) * e, n)
    run("場合 C（水準 ＋ 感情の当てやすさ）", col([0.4, 0.0, -0.4]) + e, col([0.3, 0.0, -0.3]) + e, n)
    inter = np.array([[2.5, 0.3, 0.3, 0.8], [0.3, 2.5, 0.3, 0.8], [0.3, 0.3, 2.5, 0.8]])
    run("場合 D（モダリティごとに得意な感情が違う）", inter, inter, n)
print("d′ の関数は 2 つのコミットで同じか：", inspect.getsource(C.auc_pair) == inspect.getsource(V4.auc_pair), "（dprime の本体の違いは下）")
import difflib
print("".join(list(difflib.unified_diff(inspect.getsource(C.dprime).splitlines(1), inspect.getsource(V4.dprime).splitlines(1), n=0))[:14]))
con = {(a, b): ws for a, b, ws in V4.contrasts()}
for k in [("MS_F6_mixsel", "prod"), ("MS_F6_mixsel", "MS_eq_l0"), ("MS_F6_mixsel", "MS_eq_l1"), ("MS_F6_orc_l1", "MS_eq_l1"), ("M_F6_mixsel", "M_F6n_mixsel"), ("M_F3_mixsel", "M_F3n_mixsel"),
          ("M_F3_orc", "M_F3n_orc"), ("M_F6_orc", "M_F6n_orc"), ("prod_ob", "prod"), ("sel_M_F6_K", "M_F6_K"), ("M_F6_K", "M_F3_K"), ("M_F6_orc", "prod")]:
    print(k, con.get(k, "無い"))
print("第 4 版：版", len(V4.BASE), "差の定義", len(con), "CRITERIA", len(V4.CRITERIA))
print([c for c in V4.CRITERIA if c[0] == "R3"][:3], [c for c in V4.CRITERIA if "MS" in c[2]])
