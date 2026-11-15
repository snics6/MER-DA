"""表の文書の 5 節（標準の手法 C）を others.csv・levels.csv と照らす．"""
import re
import pandas as pd
R = "/home/snics/MER-DA/results/dai20_eval/summary"
ot = pd.read_csv(f"{R}/others.csv", low_memory=False); lv = pd.read_csv(f"{R}/levels.csv", low_memory=False)
oc = ot[ot.group == "C"]
doc = open("/home/snics/MER-DA/DAI/docs/実験20_結果の表_2026-11-15.md").read().splitlines()
C4 = ["AFEW", "IEMOCAP", "MOSEI", "MELD"]
LAB = {"自分の答えを疑似ラベルにする学び直し": "own", "自己学習": "self", "エントロピー最小化": "entmin", "情報最大化": "im", "T3A": "t3a", "T3A（切片つき．記述）": "t3a_b",
       "ラベル伝播": "lp", "ソースの発話と K 本を合わせて学ぶ": "st", "感情の割合の補正（Saerens の EM．付録）": "saerens", "CORAL": "coral", "DANN": "dann", "MME": "mme",
       "勾配法の対照（DA の項なし）": "mapctl", "xMUDA 型": "xmuda"}
on, K, nu, n_ok, n_bad, seen = False, None, None, 0, [], set()
for ln in doc:
    if ln.startswith("## 5"): on = True; continue
    if ln.startswith("## 6"): on = False
    if not on: continue
    m = re.match(r"\*\*K = (\d+)：(\d+) 単位", ln)
    if m: K, nu = int(m.group(1)), int(m.group(2)); continue
    if not ln.startswith("| ") or ln.startswith("| 手法") or ln.startswith("|---"): continue
    cs = [x.strip() for x in ln.strip("|").split("|")]
    lab, vals = cs[0], cs[1:]
    for j, c in enumerate(C4):
        u, d = vals[2 * j], vals[2 * j + 1]
        if lab.startswith("**提案の手法**") or lab.startswith("積（"):
            v = "prop" if lab.startswith("**") else "prod"
            q = lv[(lv.c == c) & (lv.K == K) & (lv.version == v) & (lv.seeds == ("all" if nu == 50 else "0-2"))]
            ok = abs(float(u) - q.uar.iloc[0]) <= 0.005 + 1e-12 and d == "—"
        elif lab.startswith("今の提案"):
            q = oc[(oc.c == c) & (oc.K == K) & (oc.n_units == nu)]
            vals_ = sorted(set(round(float(x), 6) for x in q.uar_exp1prop_same_units))
            ok = all(abs(float(s) - x) <= 0.005 + 1e-12 for s, x in zip(u.split("／"), vals_)) and len(u.split("／")) == len(vals_)
        else:
            mth = LAB[lab]; seen.add((K, nu, mth))
            q = oc[(oc.c == c) & (oc.K == K) & (oc.method == mth)]
            assert len(q) == 1 and int(q.n_units.iloc[0]) == nu, (lab, c, K, nu, len(q))
            r = q.iloc[0]
            mm = re.match(r"([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]$", d)
            ok = abs(float(u) - r.uar) <= 0.005 + 1e-12 and all(abs(float(s) - x) <= 0.5 * 10 ** -len(s.split(".")[1]) + 1e-12
                                                               for s, x in zip(mm.groups(), (r.exp1prop_minus_method, r.exp1prop_minus_method_lo, r.exp1prop_minus_method_hi)))
        n_ok += ok
        if not ok: n_bad.append((lab, K, nu, c, u, d))
want = {(int(r.K), int(r.n_units), r.method) for r in oc.itertuples()}
print("照らしたセル", n_ok, "合わない", n_bad[:5], "表に無い (K, 単位, 手法)", sorted(want - seen), "表にだけある", sorted(seen - want))
