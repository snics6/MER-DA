"""DAI-8 の計算の後の点検（gate-code）．値は出さず，数・欠け・重複・一致だけを出す．"""
import json, hashlib, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/home/snics/MER-DA")
OUT = ROOT / "results/dai8_probe"
UD = ROOT / "results/dai1/pseudo/units"
SP = ("A", "V", "T", "AVT"); KS = (0, 1, 5, 10, 25); AL = (0.1, 0.5, 0.9); BE = (0.0, 1.0)

def expected_keys():
    pred, D, C, P, G = set(), set(), set(), set(), set()
    for s in SP:
        pred.add(f"{s}_oracle")
        for a in AL: pred.add(f"{s}_ldaO_a{a}")
        D.add(f"{s}_ldaO_a0.5"); C.add(f"{s}_ldaO_a0.5"); G.add(f"{s}_ldaO_a0.5")
        if s == "AVT": pred.add("AVT_ldaOblk_a0.5"); D.add("AVT_ldaOblk_a0.5")
        for t in ("truth", "rand"):
            pred |= {f"{s}_em{t}_a0.5_b0.0", f"{s}_em{t}conv_a0.5_b0.0"}
            P.add(f"{s}_em{t}conv_a0.5_b0.0"); G.add(f"{s}_em{t}_a0.5_b0.0")
        for K in KS:
            p = f"K{K}_{s}_"
            pred |= {p + "none", p + "joint", p + "ttfused"}
            P |= {p + "none", p + "em10_a0.5_b1.0", p + "emconv_a0.5_b1.0"}
            if K >= 1:
                for a in AL: pred.add(p + f"ldaK_a{a}")
                pred |= {p + "emfromK_a0.5_b0.0", p + "noneT", p + "emT_a0.5_b1.0"}
                D |= {p + "ldaK_a0.5", p + "emfromK_a0.5_b0.0", p + "emT_a0.5_b1.0"}
                C |= {p + "ldaK_a0.5", p + "emfromK_a0.5_b0.0"}
                G |= {p + "emfromK_a0.5_b0.0", p + "emT_a0.5_b1.0"}
            for a in AL:
                for b in BE:
                    pred.add(p + f"em_a{a}_b{b}"); G.add(p + f"em_a{a}_b{b}")
            for t in (0, 10, 30): pred.add(p + f"emit{t}_a0.5_b1.0")
            pred.add(p + "emconv_a0.5_b1.0")
            D |= {p + "em_a0.5_b1.0", p + "emconv_a0.5_b1.0", p + "emmu_a0.5_b1.0"}
            C |= {p + "em_a0.5_b1.0", p + "emconv_a0.5_b1.0", p + "emmu_a0.5_b1.0"}
            pred.add(p + "emmu_a0.5_b1.0"); G.add(p + "emmu_a0.5_b1.0")
            if s == "AVT":
                pred |= {p + "emblk_a0.5_b1.0", p + "emblkconv_a0.5_b1.0"}
                D.add(p + "emblk_a0.5_b1.0"); C.add(p + "emblk_a0.5_b1.0"); G.add(p + "emblk_a0.5_b1.0")
    return pred, D, C, P, G

EP, ED, EC, EPP, EG = expected_keys()
print("予定の鍵の数：判定", len(EP), "δ", len(ED), "係数", len(EC), "×4", "候補の判定", len(EPP), "診断", len(EG))
expected_units = sorted(p.name[:-len("_p3.npz")] for p in UD.glob("*_p3.npz"))
files = sorted((OUT / "rows").glob("*.npz"))
names = [p.name[:-4] for p in files]
print("入力の単位", len(expected_units), "rows の npz", len(files), "重複", len(names) - len(set(names)),
      "欠け", len(set(expected_units) - set(names)), "余分", len(set(names) - set(expected_units)))
tup = Counter(tuple(n.split("_")) for n in names)
print("(T,c,f,r) の重複", sum(v > 1 for v in tup.values()), "T≠c 違反", sum(n.split("_")[0] == n.split("_")[1] for n in names))
print("c ごとの単位", dict(Counter(n.split("_")[1] for n in names)), "T ごと", dict(Counter(n.split("_")[0] for n in names)))
print("(c,f) ごと", sorted(Counter((n.split("_")[1], n.split("_")[2]) for n in names).values()) [:1], "〜",
      sorted(Counter((n.split("_")[1], n.split("_")[2]) for n in names).values())[-1])
prov = json.loads((OUT / "provenance.json").read_text())
sh = [json.loads((OUT / f"provenance_shard{i}.json").read_text()) for i in (0, 1)]
u0, u1 = set(sh[0]["units"]), set(sh[1]["units"])
print("組の単位", len(u0), len(u1), "重なり", len(u0 & u1), "和と入力の一致", (u0 | u1) == set(expected_units))
print("組の runs", [len(x.get("runs", [])) for x in sh], "finished", [x.get("finished") for x in sh])
print("組の commit", [x["commit"] for x in sh], "dirty", [x["dirty"] for x in sh], "partial", [x["partial"] for x in sh], "shard", [x["shard"] for x in sh])
print("組の script_sha256", [x["script_sha256"] for x in sh])
print("組の frozen_sha256 一致", sh[0]["frozen_sha256"] == sh[1]["frozen_sha256"])
# 入力の sha の照合（いまのファイルと比べる）
bad = 0
for x in sh:
    for un, d in x["inputs_sha256"].items():
        for i, h in d.items():
            if hashlib.sha256((UD / f"{un}_p{i}.npz").read_bytes()).hexdigest() != h: bad += 1
print("入力の sha の不一致（今のファイルと）", bad, "照合した数", sum(len(d) for x in sh for d in x["inputs_sha256"].values()))
print("組の定数の一致", all(sh[0][k] == sh[1][k] for k in sh[0] if k not in ("units", "inputs_sha256", "runs", "started", "finished", "shard")))
print("runs の照合の不一致の和", sum(r["none_mismatch"] for x in sh for r in x["runs"]), "比べた数", sum(r["none_compared"] for x in sh for r in x["runs"]))
print("集計の provenance：", {k: prov.get(k) for k in ("summarize_commit", "summarize_dirty", "rows_commits", "n_units_found", "same_cf_lda_mismatch_max", "boot_seed", "n_boot")},
      "shards の数", len(prov.get("shards", [])))
# 単位ごとの鍵
fixed_meta = {"commit", "test_sample_id", "test_y", "test_cluster", "test_group", "pool_prop", "pool_y", "pool_group", "pool_dialogue", "diag", "checks", "n_pool"}
probs = Counter(); commits = set(); nonfinite = Counter(); diagbad = Counter()
for p in files:
    z = np.load(p, allow_pickle=False)
    ks = set(z.files)
    commits.add(str(z["commit"]))
    nt, npool = len(z["test_y"]), int(z["n_pool"])
    pr = {k for k in ks if k not in fixed_meta and not k.startswith(("D_", "P_", "C_"))}
    dk = {k[2:] for k in ks if k.startswith("D_")}; pk = {k[2:] for k in ks if k.startswith("P_")}
    ck = Counter(k.split("_", 2)[1] for k in ks if k.startswith("C_"))
    cks = {k.split("_", 2)[2] for k in ks if k.startswith("C_")}
    if pr != EP: probs["判定の鍵"] += 1; print(p.name, "判定 欠け", sorted(EP - pr)[:5], "余分", sorted(pr - EP)[:5])
    if dk != ED: probs["δの鍵"] += 1
    if pk != EPP: probs["候補の判定の鍵"] += 1
    if cks != EC or any(v != len(EC) for v in ck.values()) or set(ck) != {"A", "mu", "c", "pi"}: probs["係数の鍵"] += 1
    if not fixed_meta <= ks: probs["付帯の鍵"] += 1
    for k in pr:
        a = z[k]
        if a.shape != (nt,) or a.dtype != np.int8 or a.min() < 0 or a.max() > 3: probs["判定の形・値域"] += 1
    for k in dk:
        a = z["D_" + k]
        if a.shape != (nt, 4): probs["δの形"] += 1
        if not np.isfinite(a).all(): nonfinite["D"] += 1
    for k in pk:
        a = z["P_" + k]
        if a.shape != (npool,) or a.min() < 0 or a.max() > 3: probs["候補の判定の形・値域"] += 1
    for k in cks:
        for part in ("A", "mu", "c", "pi"):
            if not np.isfinite(z[f"C_{part}_{k}"].astype(np.float64)).all(): nonfinite["C_" + part] += 1
        if abs(float(z[f"C_pi_{k}"].sum()) - 1) > 1e-4: probs["π の和"] += 1
    for nm in ("test_sample_id", "test_cluster", "test_group"):
        if len(z[nm]) != nt: probs[nm + " の長さ"] += 1
    if len(set(z["test_sample_id"].astype(str))) != nt: probs["テストの sample_id の重複"] += 1
    for nm in ("pool_y", "pool_group", "pool_dialogue"):
        if len(z[nm]) != npool: probs[nm + " の長さ"] += 1
    if abs(float(z["pool_prop"].sum()) - 1) > 1e-9: probs["pool_prop の和"] += 1
    dg = json.loads(str(z["diag"]))
    if set(dg) != EG: probs["診断の鍵"] += 1; print(p.name, "診断 欠け", sorted(EG - set(dg))[:5], "余分", sorted(set(dg) - EG)[:5])
    for k, d in dg.items():
        h = d.get("hist") or []
        ph = d.get("pool_hist") or []
        if not all(np.isfinite(h)): diagbad["hist の非有限"] += 1
        if "_em_a0.5_b1.0" in k or "emtruth" in k or "emrand" in k or "emblk" in k:
            if len(ph) != d["n_iter_done"] + 1: diagbad["収束の版の pool_hist の長さ"] += 1
        elif "ldaO" not in k:
            if d.get("n_iter_done") != 10: diagbad["10 回の版の回の数"] += 1
            if len(ph) != 11: diagbad["10 回の版の pool_hist の長さ"] += 1
        for nm in ("pool_uar", "entropy", "moved", "ratio"):
            if nm in d and d[nm] is not None and not np.isfinite(d[nm]): diagbad[f"{nm} の非有限 ({k.split('_')[-3] if k.startswith('K') else k})"] += 1
        if "pool_uar" in d and not (0 <= d["pool_uar"] <= 1): diagbad["pool_uar の値域"] += 1
    ch = json.loads(str(z["checks"]))
    if ch["none_mismatch"] != 0: probs["none の照合"] += 1
    kn = OUT / "rows" / f"{p.name[:-4]}_knn.csv"
print("rows の commit", commits)
print("単位の鍵と形の問題", dict(probs) or "なし")
print("非有限（単位の数）", dict(nonfinite) or "なし")
print("診断の問題（診断の数）", dict(diagbad) or "なし")
knnf = sorted((OUT / "rows").glob("*_knn.csv"))
print("knn の csv", len(knnf), "c×f の重複", len(knnf) - len({tuple(p.name.split("_")[1:3]) for p in knnf}),
      "対応する npz が無い", sum(not (OUT / "rows" / (p.name[:-8] + ".npz")).exists() for p in knnf))
print("knn の単位の規則（T が辞書順で最初，r0）に合わない", sum(not (p.name.split("_")[3] == "r0" and p.name.split("_")[0] == sorted(x for x in ("AFEW","IEMOCAP","MELD","MOSEI") if x != p.name.split("_")[1])[0]) for p in knnf))
kc = {str(c) for p in knnf for c in pd.read_csv(p).commit.unique()}
print("knn の commit", kc)
