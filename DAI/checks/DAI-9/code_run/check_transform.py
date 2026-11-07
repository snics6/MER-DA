"""gate-code（DAI-9）：transform の 4 種の性質を実データ（AFEW，CPU）で確かめる。値は出さない（差の大きさと数だけ）。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path("/home/snics/MER-DA")
sys.path.insert(0, str(ROOT / "DAI" / "exp"))
import dai1, dai_data as DD, dai9_probe as P9  # noqa: E402
T, c, f, r = "IEMOCAP", "AFEW", 0, 0
u = dai1.Unit(ROOT / "results" / "dai1", T, c, [x for x in DD.CORPORA if x not in (T, c)], f, r)
Xp, Xt = u.Xp["AVT"].double(), u.Xt["AVT"].double()
gp = u.cp.df["cluster"].to_numpy()[u.pool].astype(str); gt = u.cp.df["cluster"].to_numpy()[u.test].astype(str)
print("候補", Xp.shape, "テスト", Xt.shape, "機器", Xp.device)
# gc：引いた後（割り直す前）の大きいグループの平均
for nm, X, g in (("候補", Xp, gp), ("テスト", Xt, gt)):
    Z = P9._center_groups(X, g)
    _, gi, cnt = np.unique(g, return_inverse=True, return_counts=True)
    big = [k for k in range(len(cnt)) if cnt[k] >= P9.MIN_GROUP]
    mx = max(float(Z[torch.as_tensor(gi == k)].mean(0).abs().max()) for k in big)
    small = int(sum(cnt[k] for k in range(len(cnt)) if cnt[k] < P9.MIN_GROUP))
    print(f"gc {nm}：3 本以上のグループ {len(big)}／{len(cnt)}，その平均の最大の絶対値 {mx:.1e}，3 本未満のグループの発話 {small}／{len(g)}，全体の平均の最大 {float(Z.mean(0).abs().max()):.1e}")
a, b = P9.transform("gc", Xp, Xt, gp, gt)
print(f"gc の後：候補の標準偏差の 1 からの最大のずれ {float((a.std(0) - 1).abs().max()):.1e}，候補の平均の最大 {float(a.mean(0).abs().max()):.1e}")
# nap：割り直す前に VᵀX が 0 か
for rr in (10, 50):
    Xc = Xp - Xp.mean(0)
    _, gi, cnt = np.unique(gp, return_inverse=True, return_counts=True)
    M = torch.zeros((len(cnt), Xp.shape[1]), dtype=Xp.dtype).index_add_(0, torch.as_tensor(gi), Xc) / torch.as_tensor(cnt, dtype=Xp.dtype)[:, None]
    Sg = (M.T * torch.as_tensor(cnt / cnt.sum())) @ M
    ev, V = torch.linalg.eigh((Sg + Sg.T) / 2); V = V[:, -rr:]
    print(f"nap{rr}：射影の後の Vᵀx の最大 候補 {float(((Xp - Xp @ V @ V.T) @ V).abs().max()):.1e}，テスト {float(((Xt - Xt @ V @ V.T) @ V).abs().max()):.1e}，"
          f"消す向きの固有値の和の割合 {float(ev[-rr:].sum() / ev.sum()):.3f}，グループの数 {len(cnt)}（ランクの上限 {len(cnt) - 1}）")
    a, b = P9.transform(f"nap{rr}", Xp, Xt, gp, gt)
    # 割り直した後のグループの間の散らばりの上位 rr 個の割合（割り直しで戻る分）
    Mc = torch.zeros_like(M).index_add_(0, torch.as_tensor(gi), a - a.mean(0)) / torch.as_tensor(cnt, dtype=Xp.dtype)[:, None]
    Sg2 = (Mc.T * torch.as_tensor(cnt / cnt.sum())) @ Mc
    print(f"   割り直した後：元の V の向きのグループの間の散らばり {float(torch.trace(V.T @ Sg2 @ V)):.2e}（割り直す前は 0），標準偏差のずれ {float((a.std(0) - 1).abs().max()):.1e}，"
          f"標準偏差がほぼ 0 の列 {int((((Xp - Xp @ V @ V.T)).std(0) < 1e-5).sum())}")
for p in (64, 256):
    a, b = P9.transform(f"pca{p}", Xp, Xt, gp, gt)
    print(f"pca{p}：形 {tuple(a.shape)} {tuple(b.shape)}，候補の標準偏差のずれ {float((a.std(0) - 1).abs().max()):.1e}，候補の成分の相関の最大 {float((torch.corrcoef(a.T) - torch.eye(p, dtype=a.dtype)).abs().max()):.1e}")
# コーパスごと・分割ごとの 3 本未満のグループの発話の割合（splits の cluster．IEMOCAP の話者は 10 人で対象外）
for cc in ("AFEW", "MOSEI", "MELD"):
    d = pd.read_csv(ROOT / "DAI" / "splits" / f"{cc}.csv", keep_default_na=False)
    for side in ("候補", "テスト"):
        fr = []
        for ff in range(5):
            s = d[(d.fold != ff)] if side == "候補" else d[d.fold == ff]
            n = s.cluster.map(s.cluster.value_counts())
            fr.append(float((n < 3).mean()))
        print(f"{cc} {side}：3 本未満のグループの発話の割合（5 分割の最小〜最大）{min(fr):.3f}〜{max(fr):.3f}")
