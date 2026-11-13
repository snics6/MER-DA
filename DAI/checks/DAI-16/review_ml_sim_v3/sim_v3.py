"""ML の査読役の検算（DAI-16 第 3 版）．人工の数だけ．MER-DA のデータと結果は読まない．
台本 dai16_probe.py の関数（dprime・to_w・combine・centered・match_shift・fit_wb・norm_log・softmax）をそのまま呼ぶ．"""
import sys
import numpy as np
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import importlib.util
_sp = importlib.util.spec_from_file_location("p16c", "/tmp/claude-1000/-home-snics-MER-DA/76085a37-4610-4922-b8ae-ced34f163c39/scratchpad/rev_ml_v3/dai16_probe_556a06f.py")
P = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(P)

def uar(y, pred):
    return 100 * float(np.mean([(pred[y == i] == i).mean() for i in range(4)]))

def draw(rng, n, sep, prior=None):
    """sep：(分類器, 4)．点 Z_k = 標準正規 + 本当の感情の成分に sep_k(y)．確率は softmax(Z_k)．"""
    y = rng.choice(4, size=n, p=prior) if prior is not None else rng.integers(0, 4, n)
    Ps = []
    for k in range(sep.shape[0]):
        Z = rng.normal(size=(n, 4)); Z[np.arange(n), y] += sep[k][y]
        Ps.append(P.softmax(Z))
    return y, Ps

def layers(Ls_pool, Ls_test, yt, form, W, q1p, q1t, target):
    Sp, St = P.combine(form, Ls_pool, W, q1p), P.combine(form, Ls_test, W, q1t)
    dl, _ = P.match_shift(Sp, target)
    return uar(yt, St.argmax(1)), uar(yt, (St + dl).argmax(1))

def prof(D):
    return D - D.mean(1, keepdims=True)

def dbl(D):
    return D - D.mean(1, keepdims=True) - D.mean(0, keepdims=True) + D.mean()

def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])

def run_case(name, sepS, sepT, n=20000, reps=20, prior=None):
    """sepS：信頼度を測る側（ソースに当たる）の sep．sepT：候補とテスト（ターゲットに当たる）の sep．
    重みは (a) ターゲットの候補の全ラベル（orc に当たる），(b) ソースの側（ext に当たる）で測った d′ から作る．"""
    out = {}
    for rep in range(reps):
        rng = np.random.default_rng(1000 + rep)
        ys, Ps = draw(rng, n, sepS, prior)
        yp, Pp = draw(rng, n, sepT, prior)
        yt, Pt = draw(rng, n, sepT, prior)
        Lp, Lt = np.stack([P.centered(p) for p in Pp]), np.stack([P.centered(p) for p in Pt])
        q1p, q1t = P.softmax(Lp.sum(0)), P.softmax(Lt.sum(0))
        target = q1p.mean(0)
        Dorc = np.stack([P.dprime(p, yp)[0] for p in Pp])
        Dsrc = np.stack([P.dprime(p, ys)[0] for p in Ps])
        r = {"prod": uar(yt, Lt.sum(0).argmax(1))}
        for tag, D in (("orc", Dorc), ("src", Dsrc)):
            W = P.to_w(D)
            We = np.repeat(W.mean(0, keepdims=True), W.shape[0], 0)      # 感情ごとの重みを分類器で共通にした対照（設計に無い）
            for f in ("F6", "F3", "F2"):
                r[f"{f}_{tag}_raw"], r[f"{f}_{tag}_adj"] = layers(Lp, Lt, yt, f, W, q1p, q1t, target)
            r[f"Fe_{tag}_raw"], r[f"Fe_{tag}_adj"] = layers(Lp, Lt, yt, "F6", We, q1p, q1t, target)
            r[f"F3e_{tag}_raw"], r[f"F3e_{tag}_adj"] = layers(Lp, Lt, yt, "F3", We, q1p, q1t, target)
        b = P.fit_wb(P.norm_log(Lp.sum(0))[None], yp, np.ones(1), 0.0, fit_w=False, fit_b=True)[1]
        Sp_b, St_b = P.norm_log(P.norm_log(Lp.sum(0)) + b), P.norm_log(P.norm_log(Lt.sum(0)) + b)
        dl, _ = P.match_shift(Sp_b, target)
        r["bias_raw"] = uar(yt, St_b.argmax(1)); r["bias_adj"] = uar(yt, (St_b + dl).argmax(1))
        r["bias_adj_same_as_prod"] = float(((St_b + dl).argmax(1) == Lt.sum(0).argmax(1)).mean())
        r["R0_profcorr"] = corr(prof(Dsrc), prof(Dorc)); r["R0_dblcorr"] = corr(dbl(Dsrc), dbl(Dorc))
        for k, v in r.items():
            out.setdefault(k, []).append(v)
    m = {k: (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v)))) for k, v in out.items()}
    d = lambda a, b: (float(np.mean(np.array(out[a]) - np.array(out[b]))), float(np.std(np.array(out[a]) - np.array(out[b]), ddof=1) / np.sqrt(reps)))
    print(f"\n== {name}")
    print(f"  積 {m['prod'][0]:.1f}   R0 の相関（設計の定義） {m['R0_profcorr'][0]:+.2f}   感情の平均も引いた相関 {m['R0_dblcorr'][0]:+.2f}")
    for tag in ("orc", "src"):
        print(f"  [{tag}] 層 adj：F6−積 {d(f'F6_{tag}_adj','prod')[0]:+.2f}±{d(f'F6_{tag}_adj','prod')[1]:.2f}  F3−積 {d(f'F3_{tag}_adj','prod')[0]:+.2f}±{d(f'F3_{tag}_adj','prod')[1]:.2f}"
              f"  F2−積 {d(f'F2_{tag}_adj','prod')[0]:+.2f}  F6−F2 {d(f'F6_{tag}_adj',f'F2_{tag}_adj')[0]:+.2f}  F6−F3 {d(f'F6_{tag}_adj',f'F3_{tag}_adj')[0]:+.2f}"
              f"  共通(F6 の形)−積 {d(f'Fe_{tag}_adj','prod')[0]:+.2f}  F6−共通 {d(f'F6_{tag}_adj',f'Fe_{tag}_adj')[0]:+.2f}±{d(f'F6_{tag}_adj',f'Fe_{tag}_adj')[1]:.2f}"
              f"  共通(F3 の形)−積 {d(f'F3e_{tag}_adj','prod')[0]:+.2f}")
        print(f"  [{tag}] 層 raw：F6−積 {d(f'F6_{tag}_raw','prod')[0]:+.2f}  F3−積 {d(f'F3_{tag}_raw','prod')[0]:+.2f}  共通(F6 の形)−積 {d(f'Fe_{tag}_raw','prod')[0]:+.2f}  F6−共通 {d(f'F6_{tag}_raw',f'Fe_{tag}_raw')[0]:+.2f}")
    print(f"  切片だけ：raw−積 {d('bias_raw','prod')[0]:+.2f}  adj−積 {d('bias_adj','prod')[0]:+.4f}  adj の判定が積と同じ割合 {m['bias_adj_same_as_prod'][0]:.6f}")

e = np.array([0.3, 0.8, 1.5, 2.5])
one = np.ones((3, 1))
# 場合 1：感情ごとの当てやすさだけが違う（3 つの分類器は同じ作り．モダリティ × 感情の違いは無い）
run_case("場合 1：感情の当てやすさだけ（分類器は 3 つとも同じ）", one * e, one * e)
# 場合 2：場合 1 に，モダリティごとの水準の違いを足す（相性の形はどの分類器も同じ）
lvS, lvT = np.array([[1.3], [1.0], [0.7]]), np.array([[0.7], [1.0], [1.3]])
run_case("場合 2：感情の当てやすさ × 水準（ソースとターゲットで水準の順が逆．モダリティ × 感情の違いは無い）", lvS * e, lvT * e)
# 場合 3：モダリティごとに得意な感情が違う（感情の当てやすさの違いは小さい）．ソースとターゲットで同じ
inter = np.array([[2.5, 0.3, 0.3, 0.8], [0.3, 2.5, 0.3, 0.8], [0.3, 0.3, 2.5, 0.8]])
run_case("場合 3：モダリティごとに得意な感情が違う（ソースとターゲットで同じ）", inter, inter)
# 場合 4：感情の当てやすさは共通．モダリティ × 感情の違いはソースとターゲットで別（持ち越せない）
rs = np.array([[0.6, -0.6, 0.0, 0.0], [-0.6, 0.6, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
rt = np.array([[-0.6, 0.6, 0.0, 0.0], [0.6, -0.6, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
e4 = np.array([0.8, 0.8, 1.5, 2.5])
run_case("場合 4：感情の当てやすさは共通，モダリティ × 感情の違いはソースとターゲットで逆", one * e4 + rs, one * e4 + rt)
