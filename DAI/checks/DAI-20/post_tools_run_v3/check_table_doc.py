"""表の文書の値を集計から取り直して突き合わせる（表の台本は使わない）．
差の表：各セルの「差 [下端, 上端]」を，同じ (コーパス, K, 引き方) の effects.csv の全行と，自分で丸め直した文字列で照らし，合う (a, b) の候補を集める．
行ごとに候補の共通部分を取り，行の名前と並べて出す．水準の表も同じく levels.csv の版の候補を出す．"""
import re
import numpy as np, pandas as pd
R = "/home/snics/MER-DA/results/dai20_eval/summary"
ef = pd.read_csv(f"{R}/effects.csv", low_memory=False); lv = pd.read_csv(f"{R}/levels.csv", low_memory=False); ot = pd.read_csv(f"{R}/others.csv", low_memory=False)
doc = open("/home/snics/MER-DA/DAI/docs/実験20_結果の表_2026-11-15.md").read().splitlines()
C4 = ["AFEW", "IEMOCAP", "MOSEI", "MELD"]
cell_re = re.compile(r"^([+-]\d+\.\d+) \[([+-]\d+\.\d+), ([+-]\d+\.\d+)\]([↑↓]?)$")


def ok_num(s, x):
    nd = len(s.split(".")[1])
    return abs(float(s) - x) <= 0.5 * 10 ** (-nd) + 1e-12


def ok_end(s, x):                                    # 端：1 桁．|x|<0.05 なら 0 でなくなるまで桁を足す（その桁で照らす）
    return ok_num(s, x) and ((len(s.split(".")[1]) == 1) == (not (abs(x) < 0.05 and x != 0)) or len(s.split(".")[1]) > 1)


sec, sub, Kcur = "", "", None
rows_diff, rows_lv, bad = {}, {}, []
mark_bad = 0
seeds_map = {"0・1": "0-1", "2〜9": "2-9"}
for ln in doc:
    if ln.startswith("## "): sec = ln
    if ln.startswith("### "): sub = ln; Kcur = None
    m = re.match(r"\*\*K = (\d+)", ln)
    if m: Kcur = int(m.group(1))
    if not ln.startswith("| ") or ln.startswith("|---") or ln.startswith("| 差 ") or ln.startswith("| 版 ") or ln.startswith("| 引き方 ") or ln.startswith("| 手法 ") or ln.startswith("| コーパス"):
        continue
    cs = [x.strip() for x in ln.strip("|").split("|")]
    if sec.startswith("## 5"):
        continue
    if len(cs) == 6 and cs[1].isdigit():                    # 差の表（ラベル, K, 4 コーパス）
        lab, K, vals = cs[0], int(cs[1]), cs[2:]
        sn = "all"
        if lab in seeds_map: sn = seeds_map[lab]; lab = "（引き方）提案 − 積 " + lab
        key = (sec, sub, lab)
        for c, v in zip(C4, vals):
            q = ef[(ef.c == c) & (ef.K == K) & (ef.seeds == sn)]
            if v == "0（同じ数）":
                cand = {(r.a, r.b) for r in q.itertuples() if bool(r.same_counts)}
            else:
                mm = cell_re.match(v)
                if not mm: bad.append((lab, K, c, v)); continue
                cand = {(r.a, r.b) for r in q.itertuples() if ok_num(mm.group(1), r.est) and ok_num(mm.group(2), r.lo) and ok_num(mm.group(3), r.hi)}
                for r in q.itertuples():
                    if (r.a, r.b) in cand:
                        want = "↑" if (r.lo > 0 and r.est >= 1.0) else "↓" if (r.hi < 0 and r.est <= -1.0) else ""
                        if want != mm.group(4): mark_bad += 1
                        if len(mm.group(1).split(".")[1]) == 2 and abs(round(r.est, 2)) == 1.0 and abs(r.est) != 1.0: bad.append(("±1.00 の桁", lab, K, c, v))
                        for s_, x in ((mm.group(2), r.lo), (mm.group(3), r.hi)):
                            if len(s_.split(".")[1]) == 1 and abs(x) < 0.05 and x != 0: bad.append(("端の桁", lab, K, c, v))
            rows_diff.setdefault(key, []).append(cand)
    elif len(cs) == 5 and Kcur is not None:                 # 水準の表（ラベル, 4 コーパス）
        lab, vals = cs[0], cs[1:]
        key = (sec, sub, lab)
        for c, v in zip(C4, vals):
            if v == "—": continue
            mm = re.match(r"^(\d+\.\d+)(?: \[(\d+\.\d+), (\d+\.\d+)\])?(＝?)$", v)
            if not mm: bad.append((lab, Kcur, c, v)); continue
            q = lv[(lv.c == c) & (lv.K == Kcur) & (lv.seeds == "all")]
            cand = {r.version for r in q.itertuples() if ok_num(mm.group(1), r.uar) and (mm.group(2) is None or (ok_num(mm.group(2), r.lo) and ok_num(mm.group(3), r.hi)))}
            rows_lv.setdefault(key, []).append(cand)
print("読めないセル", bad[:10], len(bad), "印の食い違い", mark_bad)
print("=== 差の表：行 → 全セルに共通の (a, b)")
nz = 0
for (s, u, lab), cl in rows_diff.items():
    inter = set.intersection(*cl) if cl else set()
    if not inter: nz += 1
    print(f"{u[4:]} | {lab} | セル {len(cl)} | {sorted(inter)[:3]}")
print("候補が空の行", nz)
print("=== 水準の表：行 → 全セルに共通の版")
nz = 0
for (s, u, lab), cl in rows_lv.items():
    inter = set.intersection(*cl) if cl else set()
    if not inter: nz += 1
    print(f"{s[3:12]} {u[4:]} | {lab} | セル {len(cl)} | {sorted(inter)[:4]}")
print("候補が空の行", nz)
