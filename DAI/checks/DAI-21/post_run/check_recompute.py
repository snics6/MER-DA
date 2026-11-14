"""DAI-21 計算の後の点検 3：保存した W・b（float32）と特徴から，学び直した分類器のテストと候補の判定を作り直し，保存の判定と照らす（読むだけ）．
CREMA-D は K 本だけの分類器の W・b も照らす．数と一致の割合だけを出す．"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/snics/MER-DA")
sys.path.insert(0, str(ROOT / "DAI" / "exp"))
import dai1  # noqa: E402
import dai3  # noqa: E402
import dai18_probe as P18  # noqa: E402
import dai_data as DD  # noqa: E402
import dai_solver as S  # noqa: E402

dai1.ENC["enc"] = DD.ENC1
P18.OUTDIR["out"] = ROOT / "results" / "dai18_probe"
R = ROOT / "results" / "dai21_eval5"
OUT1 = ROOT / "results" / "dai1"
KS = (5, 10, 25)
fz1, _ = dai3.load_fz1(OUT1)
res = []


def cmp(Xt, Xp, z, K, items, Wkey, bkey, pt_of, pp_of):
    W = torch.tensor(z[Wkey].astype(np.float64)); b = torch.tensor(z[bkey].astype(np.float64))
    Xt64, Xp64 = S.t(Xt).double().cpu(), S.t(Xp).double().cpu()
    if W.dim() == 2:
        W, b = W[None], b[None]
    zt = (torch.einsum("nd,pkd->pnk", Xt64, W) + b[:, None, :]).numpy()
    zp = (torch.einsum("nd,pkd->pnk", Xp64, W) + b[:, None, :]).numpy()
    out = []
    for q, it in enumerate(items):
        st, sp = zt[q].argmax(1), zp[q].argmax(1)
        srt = np.sort(zt[q], 1)
        out.append(dict(item=it, test_agree=float((st == z[pt_of(it)].astype(int)).mean()),
                        pool_agree=(float((sp == z[pp_of(it)].astype(int)).mean()) if pp_of(it) in z.files else np.nan),
                        min_margin_test=float((srt[:, -1] - srt[:, -2]).min())))
    return out


t0 = time.time()
for T, f, r in (("AFEW", 0, 0), ("IEMOCAP", 2, 5), ("MOSEI", 4, 4), ("MELD", 1, 4)):
    sv = dai3.Saved(OUT1, "eval", T, T, f, r, fz1[T])
    u = sv.u
    z = np.load(R / "students4" / f"{T}_ALL_f{f}_r{r}.npz")
    same_test = bool((z["sid_test"].astype(str) == sv.sid).all() and (z["y_test"].astype(int) == u.ytest).all())
    for K in KS:
        items = [str(x) for x in z[f"stud_items_K{K}"]]
        o = cmp(u.Xt["AVT"], u.Xp["AVT"], z, K, items, f"studW_K{K}", f"studb_K{K}", lambda it: f"pred_K{K}_{it}", lambda it: f"predpool_K{K}_{it}")
        res.append(dict(unit=f"{T}_f{f}_r{r}", K=K, kind="students", n_items=len(o), same_test=same_test, test_agree_min=min(x["test_agree"] for x in o),
                        pool_agree_min=min(x["pool_agree"] for x in o), n_items_exact=sum(x["test_agree"] == 1.0 and x["pool_agree"] == 1.0 for x in o)))
    del sv, u
    torch.cuda.empty_cache()
for f, r in ((0, 0), (3, 7)):
    u = P18.Unit6("RAVDESS", "CREMA-D", f, r)
    z = np.load(R / "cremad" / f"CREMA-D_ALL_f{f}_r{r}.npz")
    same_test = bool((z["sid_test"].astype(str) == u.sid).all() and (z["y_test"].astype(int) == u.ytest).all()) if hasattr(u, "sid") else bool((z["y_test"].astype(int) == u.ytest).all())
    for K in KS:
        items = [str(x) for x in z[f"stud_items_K{K}"]]
        o = cmp(u.Xt[u.fus], u.Xp[u.fus], z, K, items, f"studW_K{K}", f"studb_K{K}", lambda it: f"pred_K{K}_{it}", lambda it: f"predpool_K{K}_{it}")
        res.append(dict(unit=f"CREMA-D_f{f}_r{r}", K=K, kind="students", n_items=len(o), same_test=same_test, test_agree_min=min(x["test_agree"] for x in o),
                        pool_agree_min=min(x["pool_agree"] for x in o), n_items_exact=sum(x["test_agree"] == 1.0 and x["pool_agree"] == 1.0 for x in o)))
        for kind in u.kinds:
            o = cmp(u.Xt[kind], u.Xp[kind], z, K, [kind], f"W_none_{kind}_K{K}", f"b_none_{kind}_K{K}", lambda it: f"pred_raw_K{K}_none_{it}", lambda it: f"predpool_K{K}_none_{it}")
            res.append(dict(unit=f"CREMA-D_f{f}_r{r}", K=K, kind=f"none_{kind}", n_items=1, same_test=same_test, test_agree_min=o[0]["test_agree"],
                            pool_agree_min=o[0]["pool_agree"], n_items_exact=int(o[0]["test_agree"] == 1.0 and o[0]["pool_agree"] == 1.0)))
    del u
    torch.cuda.empty_cache()
for x in res:
    print(json.dumps(x, ensure_ascii=False))
print(f"所要 {time.time() - t0:.0f} 秒．GPU の最大の割り当て {torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MB")
