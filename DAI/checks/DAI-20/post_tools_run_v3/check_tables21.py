"""md_tables_dai21.py の出力（作り物の集計から書いたもの）の値を，作り物の集計から取り直して突き合わせる（表の台本は使わない）．
差のセル：同じ (コーパス, K, 引き方) の effects.csv の全行から，丸めた値が合う (a, b) の候補を集め，行ごとに共通部分を取る．
水準のセル：levels.csv（引き方 all）の版の候補．表に出ない差の対と，"—" のセルも数える．"""
import re
import sys
from pathlib import Path

import pandas as pd

V = Path("/home/snics/MER-DA/DAI/checks/DAI-20/post_tools_run_v3")
ef = pd.read_csv(V / "fake_summary21/effects.csv"); lv = pd.read_csv(V / "fake_summary21/levels.csv")
ef20 = pd.read_csv("/home/snics/MER-DA/results/dai20_eval/summary/effects.csv", low_memory=False)
doc = open(V / "fake_tables21.md").read().splitlines()
C5 = ["AFEW", "IEMOCAP", "MOSEI", "MELD", "CREMA-D"]
C4 = C5[:4]
cre = re.compile(r"^([+-]\d+\.\d+) \[([+-]\d+\.\d+), ([+-]\d+\.\d+)\]([↑↓]?)(（同じ判定）)?$")


def okn(s, x):
    return abs(float(s) - x) <= 0.5 * 10 ** (-len(s.split(".")[1])) + 1e-12


def cand_diff(src, c, K, sn, v):
    m = cre.match(v)
    if not m:
        return None, f"読めない {v}"
    q = src[(src.c == c) & (src.K == K) & (src.seeds == sn)]
    cs = {(r.a, r.b) for r in q.itertuples() if okn(m.group(1), r.est) and okn(m.group(2), r.lo) and okn(m.group(3), r.hi)}
    if m.group(5):
        cs = {ab for ab in cs if bool(q[(q.a == ab[0]) & (q.b == ab[1])].same_pred.iloc[0])}
    return cs, None


def cand_lv(c, K, v):
    q = lv[(lv.c == c) & (lv.K == K) & (lv.seeds == "all")]
    return {r.version for r in q.itertuples() if okn(v, r.uar)}


sec = sub = ""; Kc = None; hdr = []
res, bad, dash = {}, [], 0
for ln in doc:
    if ln.startswith("## "): sec = ln
    if ln.startswith("### "): sub = ln; Kc = None
    m = re.match(r"\*\*K = (\d+)", ln)
    if m: Kc = int(m.group(1))
    if ln.startswith("| ") and not ln.startswith("|---"):
        cs = [x.strip() for x in ln.strip("|").split("|")]
        if cs[0] in ("差", "版", "引き方", "K", "コーパス", "学び直した 1 つの分類器・参照"):
            hdr = cs; continue
    else:
        continue
    if sec.startswith("## 5") and sub.startswith("### 設定の 12 点"):
        continue
    lab = cs[0]
    if hdr[:2] == ["差", "K"]:                                           # 差 × K × コーパス
        K = int(cs[1]); corp = hdr[2:]
        for c, v in zip(corp, cs[2:]):
            if v == "—": dash += 1; continue
            cc, e = cand_diff(ef, c, K, "all", v)
            if e: bad.append((sub, lab, c, K, e)); continue
            res.setdefault((sub, lab, c == "CREMA-D"), []).append(cc)
    elif hdr[:2] == ["引き方", "K"]:
        sn = {"0・1": "0-1", "2〜9": "2-9"}[lab]; K = int(cs[1])
        for c, v in zip(hdr[2:], cs[2:]):
            if v == "—": dash += 1; continue
            cc, e = cand_diff(ef, c, K, sn, v)
            res.setdefault((sub, "引き方 " + sn, c == "CREMA-D"), []).append(cc)
    elif hdr[:1] == ["K"]:                                              # 実験 20 と合わせた表
        K = int(cs[0])
        for c, v in zip(hdr[1:], cs[1:]):
            cc, e = cand_diff(ef20 if c in C4 else ef, c, K, "all", v)
            if e: bad.append((sub, c, K, e)); continue
            res.setdefault((sub, "実験20と合わせた " + ("4" if c in C4 else "CD"), c == "CREMA-D"), []).append(cc)
    elif hdr[0] == "差" and hdr[1].startswith("K = "):                 # CREMA-D の差（列が K）
        for K, v in zip(KS := (5, 10, 25), cs[1:]):
            if v == "—": dash += 1; continue
            cc, e = cand_diff(ef, "CREMA-D", K, "all", v)
            if e: bad.append((sub, lab, K, e)); continue
            res.setdefault((sub, lab, True), []).append(cc)
    elif hdr[0] == "版" and len(hdr) == 7:                              # CREMA-D の水準（adj 3 列・raw 3 列）
        for j, v in enumerate(cs[1:]):
            if v == "—": dash += 1; continue
            res.setdefault((sub, lab + ("（raw）" if j >= 3 else ""), True), []).append(cand_lv("CREMA-D", (5, 10, 25)[j % 3], v))
    elif hdr[0] == "学び直した 1 つの分類器・参照":
        for j, v in enumerate(cs[1:]):
            if v == "—": dash += 1; continue
            res.setdefault((sub, lab, True), []).append(cand_lv("CREMA-D", (5, 10, 25)[j], v))
    elif hdr[0] == "版" and Kc is not None:
        for c, v in zip(hdr[1:], cs[1:]):
            if v == "—": dash += 1; continue
            res.setdefault((sub + f" K={Kc}" if False else sub, lab, c == "CREMA-D"), []).append(cand_lv(c, Kc, v))
print("読めないセル", bad[:5], "「—」のセル", dash)
used = set()
empty = 0
for (s, lab, cd), cl in res.items():
    inter = set.intersection(*cl)
    if not inter: empty += 1
    if all(isinstance(x, tuple) for x in inter): used |= {(x, cd) for x in inter}
    print(f"{s[4:40]} | {lab[:60]} | {'CD' if cd else '4'} | セル {len(cl)} | {sorted(inter)[:3]}")
print("候補が空の行", empty)
allp = {((r.a, r.b), r.c == "CREMA-D") for r in ef.itertuples()}
miss = sorted(allp - used)
print("表に出ない差の対", len(miss), miss)
