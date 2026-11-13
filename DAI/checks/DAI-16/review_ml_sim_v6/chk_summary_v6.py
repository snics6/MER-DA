"""ML の査読役の検算（DAI-16 第 6 版．コミット 0039bf0）：集計の段に足した R0 の順位の量の行を，台本の行をそのまま取り出して人工の表に当てる．
人工の単位（名前だけ本番と同じ形．MER-DA のデータと結果は読まない）に，信頼度の表を置く：
  仮のターゲット IEMOCAP・MELD・MOSEI は「掛け算の形 ＋ 得意な感情の違い」（ソースとターゲットで同じ違い），AFEW は「2 乗の形」（最も強い分類器はどの感情でも同じ）．
期待：IEMOCAP・MELD・MOSEI は hit と hit_rank の両方が True，AFEW は hit_rank が False．"""
import sys, importlib.util, tempfile, textwrap
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
SP = "/tmp/claude-1000/-home-snics-MER-DA/76085a37-4610-4922-b8ae-ced34f163c39/scratchpad/rev_ml_v6"
sp = importlib.util.spec_from_file_location("p16v6", SP + "/dai16_probe_0039bf0.py"); P = importlib.util.module_from_spec(sp); sp.loader.exec_module(P)
src = open(SP + "/dai16_probe_0039bf0.py").read().splitlines()
i0 = next(i for i, l in enumerate(src) if l.startswith("    rx = {}"))
i1 = next(i for i, l in enumerate(src) if "include_groups=False).reset_index().to_csv(sdir / \"r0_count.csv\"" in l)
block = textwrap.dedent("\n".join(src[i0:i1 + 1]))
rng = np.random.default_rng(3)

def table(n, sep):
    y = rng.integers(0, 4, n); out = []
    for k in range(sep.shape[0]):
        Z = rng.normal(size=(n, 4)); Z[np.arange(n), y] += sep[k][y]; out.append(P.dprime(P.softmax(Z), y)[0])
    return np.stack(out)

col = lambda a: np.array(a)[:, None]
e = np.array([0.5, 0.9, 1.4, 2.0]); es = np.array([0.3, 0.5, 0.8, 1.2])
small = np.array([[0.25, -0.25, 0.0, 0.0], [-0.25, 0.25, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
F = lambda a: col(a) * e + small
E3 = lambda r: (col(r) + np.sqrt(es)) ** 2
levels = {"F": [[1.4, 1.0, 0.6], [1.2, 1.1, 0.7], [1.0, 1.3, 0.7]], "E3": [[0.4, 0.0, -0.4], [0.35, 0.05, -0.4], [0.3, 0.0, -0.3]]}
tmp = Path(tempfile.mkdtemp(prefix="r0chk_")); (tmp / "units").mkdir(); sdir = tmp / "summary"; sdir.mkdir()
rows, files = [], []
corp = ["AFEW", "IEMOCAP", "MOSEI", "MELD"]
for c in corp:
    kind = "E3" if c == "AFEW" else "F"
    for ti, T in enumerate([x for x in corp if x != c]):
        for f in range(2):
            mk = (lambda a: E3(a)) if kind == "E3" else (lambda a: F(a))
            tgt = mk([0.35, 0.0, -0.35] if kind == "E3" else [1.3, 1.0, 0.7])
            arr = {"SR_ret_emo": rng.uniform(0.3, 1.2, size=(3, 4))}
            ext = table(20000, mk(levels[kind][ti]))
            for fam, rep in (("M", 1), ("S", 2)):
                E_ = np.tile(ext, (rep, 1)) + (0 if rep == 1 else rng.normal(scale=0.01, size=(3 * rep, 4)))
                for nm in ("home", "ext", "away"):
                    arr[f"SR_{fam}_{nm}"] = E_ * (1.0 if nm != "ext" else 0.8)
                for K in P.KS:
                    arr[f"D_{fam}_orc_K{K}"] = np.tile(table(20000, tgt), (rep, 1)) + (0 if rep == 1 else rng.normal(scale=0.01, size=(3 * rep, 4)))
                    arr[f"D_{fam}_K_K{K}"] = np.tile(table(4 * K, tgt), (rep, 1))
            p = tmp / "units" / f"{T}_{c}_f{f}_r0.npz"; np.savez(p, **arr); files.append(p)
            for K in P.KS:
                row = dict(c=c, K=K, version="_check")
                for fam in ("M", "S"):
                    for r in ("home", "ext", "away", "K"):
                        Ds = arr[f"D_{fam}_K_K{K}"] if r == "K" else arr[f"SR_{fam}_{r}"]; Do = arr[f"D_{fam}_orc_K{K}"]
                        row[f"intcorr_{fam}_{r}"] = P.corr(P.dcenter(Ds), P.dcenter(Do)); row[f"intperm_{fam}_{r}"] = P.perm_share(Ds, Do)
                        for q in ("profcorr", "emocorr", "levelcorr", "topok"):
                            row[f"{q}_{fam}_{r}"] = 0.0
                rows.append(row)
chk = pd.DataFrame(rows)
ns = dict(np=np, pd=pd, files=sorted(files), chk=chk, sdir=sdir, KS=P.KS, MODS=P.MODS, dcenter=P.dcenter)
exec(block, ns)
r0 = pd.read_csv(sdir / "r0.csv"); rc = pd.read_csv(sdir / "r0_count.csv")
m = r0[(r0.fam == "M") & (r0.source == "ext")]
print(m[["c", "K", "n_units", "intcorr", "intperm", "hit", "resid_src", "resid_orc", "emo_agree", "level_agree", "agree_gain", "hit_rank"]].round(3).to_string(index=False))
print(rc[(rc.fam == "M") & (rc.source == "ext")].to_string(index=False))
# 同じ量を別の書き方で出して照らす
z0 = np.load(sorted(files)[0]); c0 = sorted(files)[0].stem.rsplit("_", 3)[1]
alt = []
for p in sorted(files):
    if p.stem.rsplit("_", 3)[1] != "IEMOCAP":
        continue
    z = np.load(p); Ds, Do = z["SR_M_ext"], z["D_M_orc_K5"]
    alt.append([np.abs(Ds - P.tukey_fit(Ds)).mean(), np.abs(Do - P.tukey_fit(Do)).mean(),
                np.mean([Ds[:, i].argmax() == Do[:, i].argmax() for i in range(4)]), np.mean([Ds.mean(1).argmax() == Do[:, i].argmax() for i in range(4)])])
alt = np.array(alt).mean(0); got = m[(m.c == "IEMOCAP") & (m.K == 5)][["resid_src", "resid_orc", "emo_agree", "level_agree"]].to_numpy()[0]
print("別の書き方との差の最大（IEMOCAP，K=5）", float(np.abs(alt - got).max()))
print("範囲：一致の割合", float(r0[["emo_agree", "level_agree"]].min().min()), "〜", float(r0[["emo_agree", "level_agree"]].max().max()), " 残りの最小", float(r0[["resid_src", "resid_orc"]].min().min()),
      " hit_rank は hit の内側か", bool((~r0.hit_rank | r0.hit).all()), " NaN の数", int(r0.select_dtypes("number").isna().sum().sum()))
