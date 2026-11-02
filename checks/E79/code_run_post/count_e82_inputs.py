"""E82 の結果フォルダの行の格子（評価の T.csv と仮のターゲットの _pseudo.csv）を数える（読むだけ．UAR の値は使わない）。"""
import re
from pathlib import Path
import pandas as pd
RUNS = ["audio_wavlm", "video", "text_roberta", "fuse-audio-video-text", "fuse-audio-video"]
MR = re.compile(r"C([0-9.]+)x(s\d)?$")
out = []
for run in RUNS:
    d = Path(f"results/e82_{run}")
    for f in sorted(d.glob("*.csv")):
        if "_" in f.stem or f.stem == "check":
            continue
        T = f.stem
        x = pd.read_csv(f, dtype={"K": str, "param": str, "meff": str}, usecols=lambda c: c in ("rep", "student", "method", "param", "meff", "K", "commit"))
        m = x.param.str.extract(MR)
        x = x.assign(mult=m[0], seed=m[1].fillna("s0"))
        g = x.groupby(["student", "K", "method", "mult", "seed"]).agg(n=("rep", "size"), nrep=("rep", "nunique")).reset_index()
        dup = int(x.duplicated(["rep", "student", "method", "param", "meff", "K"]).sum())
        out.append(g.assign(run=run, target=T, dup=dup, ncommit=x.commit.nunique()))
        p = pd.read_csv(d / f"{T}_pseudo.csv", dtype={"K": str, "param": str, "meff": str}, usecols=lambda c: c in ("pseudo", "rep", "student", "method", "param", "meff", "K", "seed"))
        pp = p.groupby(["pseudo"]).rep.nunique().to_dict()
        pdup = int(p.duplicated([c for c in ("pseudo", "rep", "student", "method", "param", "meff", "K", "seed") if c in p.columns]).sum())
        print(run, T, "eval rows", len(x), "dup", dup, "commits", x.commit.nunique(), "| pseudo rows", len(p), "dup", pdup, "reps per pseudo", pp)
G = pd.concat(out)
print(G.groupby(["run", "method", "mult", "seed", "K"]).agg(n_targets=("target", "nunique"), n=("n", "sum"), nrep_min=("nrep", "min")).to_string())
