"""dai3.py の部品の検算（人工データ．CPU）．"""
import sys
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import numpy as np, torch
from scipy.optimize import minimize
import dai3 as D3, dai_solver as S

rng = np.random.default_rng(0)
n = 400
y = rng.integers(0, 4, n)
Z = rng.normal(size=(n, 4)) * 2 + np.eye(4)[y] * 1.5
cl = np.array([f"c{i % 13:02d}" for i in range(n)])
sid = np.array([f"s{i:04d}" for i in range(n)])
Zt, b0 = S.t(Z), S.t(np.array([0.3, -0.2, 0.1, 0.0]))

# 1. b* が直接の最小化と一致する（4 感情と対）
for cls in [(0, 1, 2, 3), (1, 3)]:
    m = np.isin(y, cls) & (rng.random(n) < 0.6)
    yl = D3.local_labels(y, cls)
    bs, conv = D3.solve_b(Zt, b0, [(cls, m, yl)])
    sel = np.flatnonzero(m & (yl >= 0)); k = len(cls)
    cnt = np.bincount(yl[sel], minlength=k); w = len(sel) / (k * cnt[yl[sel]])
    def obj(b):
        zz = Z[sel][:, list(cls)] + b
        lse = np.log(np.exp(zz - zz.max(1, keepdims=True)).sum(1)) + zz.max(1)
        return float((w * (lse - zz[np.arange(len(sel)), yl[sel]])).sum() / w.sum())
    r = minimize(obj, np.zeros(k), method="BFGS", options={"gtol": 1e-10})
    d = (bs[0] - bs[0].mean()) - (r.x - r.x.mean())
    print("1. b* と直接の最小化の差", cls, f"{np.abs(d).max():.1e}", "勾配", f"{conv['grad_max']:.1e}")
    assert np.abs(d).max() < 1e-5

# 2. 全体で解いた b* を元の切片にすると fullb = orig（差が無いときの位置の誤りの水準の定義の検算）
grp = D3.test_halves(y, cl, sid)
keys, keep = D3.rep_keep(cl, "cluster", None)
yl = D3.local_labels(y, (0, 1, 2, 3))
bfull, _ = D3.solve_b(Zt, b0, [((0, 1, 2, 3), np.ones(n, bool), yl)])
R, _ = D3.refit_item(Zt, S.t(bfull[0]), y, grp, keep)
print("2. fullb − orig（全体の b* のとき）", f"{np.nanmax(np.abs(R['fullb'][0] - R['orig'][0])):.1e}", "valid", int(R["valid"].sum()), "/", R["valid"].size)
assert np.allclose(R["fullb"][0, 0], R["orig"][0, 0])
# 切片に定数を足しても変わらない
R2, _ = D3.refit_item(Zt, b0 + 5.0, y, grp, keep)
R3, _ = D3.refit_item(Zt, b0, y, grp, keep)
assert np.allclose(np.nan_to_num(R2["refit"]), np.nan_to_num(R3["refit"])) and np.allclose(np.nan_to_num(R2["orig"]), np.nan_to_num(R3["orig"]))
print("2b. 切片の共通の定数で変わらない：通った")

# 3. jackknife：θ_(−g) が残りの塊の値の平均なら，V_jk = s²/G（塊の数 G）
x = rng.normal(size=13)
vals = {(0, 0): {"full": float(x.mean()), **{f"c{g:02d}": float(np.delete(x, g).mean()) for g in range(13)}}}
sizes = {0: {f"c{g:02d}": 30 for g in range(13)}}
e = D3.jk_stats(vals, "cluster", sizes)
print("3. jackknife の分散", f"{e['v_jk']:.6f}", "式", f"{x.var(ddof=1)/13:.6f}", "自由度", e["df"])
assert abs(e["v_jk"] - x.var(ddof=1) / 13) < 1e-12 and e["df"] == 12
x = rng.normal(size=50)
vals = {(0, 0): {"full": float(x.mean()), **{g: float(np.delete(x, g).mean()) for g in range(50)}}}
e = D3.jk_stats(vals, "group", {0: {"a": 1}})
assert abs(e["v_jk"] - x.var(ddof=1) / 50) < 1e-12 and e["df"] == 49
print("3b. 50 群：通った")
# 種の項と，外れる分割の扱い
vals = {(0, 0): {"full": 1.0, "a": 1.0}, (0, 1): {"full": 3.0, "a": 3.0}, (1, 0): {"full": 0.0, "b": np.nan}, (1, 1): {"full": 2.0, "b": np.nan}}
e = D3.jk_stats(vals, "cluster", {0: {"a": 10}, 1: {"b": 10}})
print("3c. θ", e["effect"], "V_seed（分割 1 は塊 b を除くと外れるので足さない）", e["v_seed"], "全部", e["v_seed_all"], "外れた回", e["rep_fold_drops"], e["folds_D"])
assert e["effect"] == 1.5 and abs(e["v_seed"] - (2.0 / 2) / 4) < 1e-12 and e["folds_D"] == "1"

# 4. 座標の上昇：学ぶ組の UAR は b* 以上
yl = D3.local_labels(y, (0, 1, 2, 3)); Yk = D3.onehot(yl, 4)
bstar = np.array([0.5, -0.4, 0.2, 0.0])
bopt = D3.coord_ascent(Z, yl, Yk, bstar)
u = lambda b: D3.uar_of((np.argmax(Z + b, 1) == yl)[None], np.ones((1, n), bool), Yk)[0]
print("4. 座標の上昇の UAR の増え（学ぶ組）", f"{u(bopt) - u(bstar):.2f}", "≥ 0", "切片の差の動き", np.round(bopt - bstar, 2))
assert u(bopt) >= u(bstar) - 1e-9 and abs(bopt[0] - bstar[0]) == 0

# 5. 回す：0 度は元のまま，角度は指定どおり，ノルムはそろう
W = rng.normal(size=(4, 30)); W -= W.mean(0)
for ang in (0.0, 30.0):
    Wr = D3.rotate_rows(W, ang, np.random.default_rng(1))
    cos = (W * Wr).sum(1) / np.linalg.norm(W, axis=1) / np.linalg.norm(Wr, axis=1)
    print("5. 角度", ang, "各行の角度", np.round(np.degrees(np.arccos(np.clip(cos, -1, 1))), 6), "ノルムの差", f"{abs(np.linalg.norm(Wr) - np.linalg.norm(W)):.1e}")
    assert np.allclose(np.degrees(np.arccos(np.clip(cos, -1, 1))), ang, atol=1e-5)

# 6. 2 組：感情ごとに件数の差は 1 以内．候補の 2 組は塊をまたがない
for k in range(4):
    assert abs((grp[y == k] == 0).sum() - (grp[y == k] == 1).sum()) <= 1
h = D3.pool_halves(cl)
assert all(len(set(h[cl == g])) == 1 for g in set(cl))
print("6. 2 組の配り方：通った", "候補の組の大きさ", (h == 0).sum(), (h == 1).sum())

# 7. Freedman–Lane：得と再現できる度合いが当たり方を通してだけつながるとき，p は偏らない（100 回の人工の繰り返しで p_upper ≤ 0.05 の割合）
hits = 0
for it in range(100):
    rr = np.random.default_rng(100 + it)
    nn = 17
    acc = rr.normal(size=nn); g = 2 * acc + rr.normal(size=nn); rp = -acc + rr.normal(size=nn)
    X = np.column_stack([np.ones(nn), acc]); M = np.eye(nn) - X @ np.linalg.pinv(X)
    e_, f_ = M @ g, M @ rp
    obs = D3.spearman_rows(e_[None], f_)[0]
    perms = np.stack([rr.permutation(nn) for _ in range(500)])
    null = D3.spearman_rows((M @ e_[perms].T).T, f_)
    hits += int((1 + (null >= obs - 1e-12).sum()) / 501 <= 0.05)
print("7. 帰無のもとで上側 5% に入った割合", hits / 100)
print("検算：すべて通った")
