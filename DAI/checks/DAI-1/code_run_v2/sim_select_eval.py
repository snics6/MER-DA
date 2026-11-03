"""選ぶ段（自分の置き場），detect の部品（分割 0 だけ），評価の回の段 3 の固定の点（frozen_points）を，仮のターゲットの単位で模擬する。
UAR の値は出さない．出すのは行の数，照合，形，例外の有無だけ。"""
import sys, json, shutil, subprocess
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "DAI/exp")
import dai1, dai1_select, dai1_stats as ST, dai_data as DD
O = Path(sys.argv[1]); step = sys.argv[2]
T = "MOSEI"
dai1.RUN["commit"], dai1.RUN["dirty"] = dai1.git_commit()
if step == "select":
    before = subprocess.run(["git", "status", "--porcelain", "DAI/frozen"], capture_output=True, text=True).stdout
    _orig = dai1.select_w   # 分割 0 だけの試走では，分割を 1 つ除くと空になる（指数の安定さの数え上げ）．空のときだけ一致 0 として通す
    dai1.select_w = lambda r: _orig(r) if len(r) else {K: (-1.0, -1.0, -1.0) for K in dai1.KS_ALL}
    dai1_select.main(O, [T])
    after = subprocess.run(["git", "status", "--porcelain", "DAI/frozen"], capture_output=True, text=True).stdout
    print("DAI/frozen の変化", before != after, repr(after))
    fz = json.load(open(O / "select" / "frozen.json"))[T]
    print("select の鍵の数", len(fz["select"]), "partner", len(fz["partner"]), "main", sorted(fz["main"]), "lofo", fz["w_lofo_agree_of5"])
    meths = sorted({k.split("|")[2] for k in fz["select"]})
    print("手法", meths)
    print("partner の手法", sorted({v["method"] + "|" + v["version"] for v in fz["partner"].values()}))
if step == "detect":
    frozen = json.load(open(O / "select" / "frozen.json")); fz = frozen[T]
    rng = np.random.default_rng(0)
    for c in (sys.argv[3].split(",") if len(sys.argv) > 3 else DD.others(T)):
        srcs = [s for s in DD.others(T) if s != c]
        g_eff, _ = ST.g_eff_of(c)
        y_of, cl_of = ST.label_maps(c)
        cells, mism, total = ST.collect(O, "pseudo", T, c, srcs, fz, [(0, 0), (0, 1)], y_of, cl_of)
        ok = 0; methods = set()
        for fam, (A, B, ks) in ST.FAMS.items():
            for s in DD.STUDENTS:
                for K in ks:
                    e = ST.effect_stats(cells[(s, K, A)], cells[(s, K, B)], g_eff, rng)
                    e2 = ST.effect_stats(cells[(s, K, A)], cells[(s, K, B)], 25.0, rng)   # 2 段の引き直しの経路を強制
                    methods |= {e["method"], e2["method"]}
                    ok += int(np.isfinite(e["se"]) and np.isfinite(e2["se"]) and np.isfinite(e2["lo"]))
                    if K == 0:
                        assert e["v_seed"] == 0.0 and not e["seed_varies"], (c, fam, s)
        print(f"c={c} G_eff={g_eff:.2f} 照合の食い違い {mism}/{total}，有限の区間 {ok}/44，経路 {sorted(methods)}")
if step == "evalsim":
    dai1.KS_ALL = (0, 5)          # 所要を抑えるため K を 2 つに絞る（10 分の内）
    frozen = json.load(open(O / "select" / "frozen.json")); sel = frozen[T]
    teth = {tuple([k.split("|")[0], int(k.split("|")[1])]): tuple(v) for k, v in sel["tether"].items()}
    w = {int(k): tuple(v) for k, v in sel["w"].items()}
    fsel = dai1.frozen_points(sel)
    c = "AFEW"; srcs = [s for s in DD.others(T) if s != c]
    dst = O / "evalsim" / "units"; dst.mkdir(parents=True, exist_ok=True)
    for p in (1, 2):
        for ext in ("npz", "csv"):
            shutil.copy(O / "pseudo" / "units" / f"{T}_{c}_f0_r0_p{p}.{ext}", dst / f"{T}_{c}_f0_r0_p{p}.{ext}")
    df = dai1.run_pass(O, "evalsim", T, c, srcs, 0, 0, 3, tether=teth, wsel=w, fsel=fsel)
    d = df[df.stage == 3]
    print("段 3 の行", len(d), "tag の行", d.groupby(["method", "tag"]).size().to_dict() if "tag" in d else "無し")
    # 固定の点の引き戻しの行のうち，選んだ引き戻しの点が，格子の回の同じ点の行と同じ判定になるか
    z = np.load(dst / f"{T}_{c}_f0_r0_p3.npz")
    mis = tot = 0
    for s in DD.STUDENTS:
        for K in dai1.KS_ALL:
            cw, cb = teth[(s, max(K, 1))]
            it_t = [json.loads(x) for x in z[f"p3t_{s}_K{K}_items"]]
            it_g = [json.loads(x) for x in z[f"p3_{s}_K{K}_items"]]
            for j, it in enumerate(it_t):
                if not (np.isclose(it["Cw"], cw) and np.isclose(it["Cb"], cb)):
                    continue
                base_m = it["method"].replace("_tether", "")
                base_m = "joint" if base_m == "joint" else base_m
                want = {k: v for k, v in it.items() if k not in ("Cw", "Cb", "tag", "n_kept", "method")}
                hits = [i for i, g in enumerate(it_g) if g["method"] == base_m and all(ST.norm(g.get(k)) == ST.norm(v) for k, v in want.items())]
                if len(hits) != 1:
                    print("対応が無い", s, K, it, len(hits)); continue
                tot += 1
                mis += int((z[f"p3t_{s}_K{K}_pred"][j] != z[f"p3_{s}_K{K}_pred"][hits[0]]).sum() > 0)
    print(f"選んだ引き戻しの点で，引き戻しの行と格子の行の判定が 1 発話でも違う問 {mis}/{tot}")
    # 再開の照合：評価の回は段 1・2 にも tether・wsel を渡す（dai1.py:717）．段 1 の保存に tether が無いので再開で止まるか
    for which in (1, 2):
        try:
            dai1.run_pass(O, "evalsim", T, c, srcs, 0, 0, which, tether=teth, wsel=w, fsel=fsel)
            print(f"段 {which} の再開：通った")
        except SystemExit as e:
            print(f"段 {which} の再開：止まった（{e}）")
