"""dai3.py の MOSEI・MELD に固有の経路の試走：評価の回に見立てた AFEW の単位を，T = "MELD"（顔の一覧は作り物）と
T = "MOSEI"（チャンネルの一覧は作り物，50 群の delete-a-group jackknife は AFEW の塊を名前の順に配る）として流す．
動くことと出力の形だけを見る（値は使わない）．"""
import sys
from pathlib import Path
TR = Path(__file__).resolve().parent
src = open(TR / "run_trial.py").read().split("sys.argv = [")[0]
exec(src)                                                             # run_trial.py の定数の絞り込みと import dai3
import json, time
import numpy as np, pandas as pd
import dai3 as D3, dai1, dai_data as DD

D3.RUN["commit"], D3.RUN["dirty"] = dai1.git_commit()
DD.others = lambda c: ["IEMOCAP"]                                     # 見立てた単位のソースは IEMOCAP の 1 つ
d1e = TR / "d1e"
fz1 = json.load(open(d1e / "select" / "frozen.json"))
fz3 = json.load(open(TR / "oute" / "frozen.json"))
afew = D3.corpus_df("AFEW")
rng = np.random.default_rng(0)
fake_face = {s: ("single" if rng.random() < 0.55 else "compete") for s in afew.sample_id.astype(str)}
names = sorted(afew.cluster.astype(str).unique())
gmap = {g: i % D3.N_GROUPS_BIG for i, g in enumerate(names)}
orig_sm, orig_jk, orig_df = D3.subset_map, D3.jk_mode, D3.corpus_df
D3.subset_map = lambda corpus, name: fake_face if (corpus, name) == ("MELD", "face_compete") else orig_sm(corpus, name)
for T in ("MELD", "MOSEI"):
    fz3["oracle_tether"][T] = fz3["oracle_tether"]["AFEW"]
    fz3["detect"][T] = fz3["detect"]["AFEW"]
    fz3["validity"]["check1"].update({k.replace("AFEW|", f"{T}|"): v for k, v in list(fz3["validity"]["check1"].items()) if k.startswith("AFEW|")})
    fz3["validity"]["check2"].update({k.replace("AFEW|", f"{T}|"): v for k, v in list(fz3["validity"]["check2"].items()) if k.startswith("AFEW|")})
    if T == "MOSEI":
        D3.jk_mode = lambda c: ("group", gmap)
        D3.mosei_channels = lambda: set(names[::3])
        D3.corpus_df = lambda c: afew if c == "MOSEI" else orig_df(c)
    U = {}
    t0 = time.time()
    for f in (0, 1):
        for r in (0, 1):
            sv = D3.Saved(d1e, "eval", "AFEW", "AFEW", f, r, fz1["AFEW"])
            sv.T = T
            U[(f, r)] = D3.eval_unit(sv, fz3, first_seed=(r == 0))
    tabs = {k: [] for k in ("decomp", "twogroup_vs_full", "family5", "family5_subsets", "expl1", "expl1_tt", "expl2", "repro",
                            "expl3_types", "expl3_cells", "expl3", "expl4", "expl5", "expl6", "recall_content", "assumptions", "null_level", "repro_units")}
    summ = D3.aggregate(T, U, fz1["AFEW"], fz3, np.random.default_rng(1), tabs)
    R0 = U[(0, 0)]
    print(T, f"{time.time()-t0:.0f} 秒", "複製の鍵", len(R0["keys"]), type(R0["keys"][1]).__name__, "部分集合", list(R0["sub"]),
          {k: v["n"] for k, v in R0["sub"].items()})
    print("  表の行", {k: len(v) for k, v in tabs.items() if v})
    d = pd.DataFrame(tabs["decomp"]); print("  方法", d.method.unique().tolist(), "自由度", sorted(d.df.dropna().unique().tolist()))
    if T == "MELD":
        e4 = pd.DataFrame(tabs["expl4"]); print("  説明 4 の部分集合", e4.subset.unique().tolist(), "欠けた値", int(e4[["A", "V", "T"]].isna().sum().sum()))
        rp = pd.DataFrame(tabs["repro"]); print("  再現できる度合いの部分集合", rp.subset.unique().tolist(), "共有する対話の平均", rp.n_shared_dialogues_mean.unique()[:3])
    else:
        fs = pd.DataFrame(tabs["family5_subsets"]); print("  族 5 の部分集合", fs.subset.unique().tolist(), len(fs), "自由度", fs.df.unique().tolist())
        print("  ラベルの中身ごとの再現率の行", len(tabs["recall_content"]))
    D3.jk_mode = orig_jk
