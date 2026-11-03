"""評価の回を模した単位の再開・非有限の扱い・引き戻しの曲線の行の確かめ（絞った設定．UAR は出さない）。引数：Y の置き場 評価を模した置き場"""
import sys, json, shutil, os
import numpy as np, pandas as pd
from pathlib import Path
import dai1, dai1_select as SEL, dai_data as DD
Y, E = Path(sys.argv[1]), Path(sys.argv[2])
shutil.rmtree(E, ignore_errors=True); (E).mkdir(parents=True)
os.symlink((Y / "source").resolve(), E / "source")
tw = json.load(open(Y / "pseudo" / "MOSEI_tether_w.json"))
p1, p2, p3 = (SEL.load(Y, "MOSEI", w) for w in (1, 2, 3))
units3 = SEL.unit_names(p3)
keep = [d[d.fold == 0] for d in (p1, p2, p3)]
r = SEL.select_T(tw, *keep, n_units=len(units3))
fz = {"select": r["select"], "main": r["main"], "partner": r["partner"]}
print("相手の手法", {k: v["method"] + "|" + v["version"] for k, v in r["partner"].items()})
fsel = dai1.frozen_points(fz)
teth = {(k.split("|")[0], int(k.split("|")[1])): tuple(v) for k, v in tw["tether"].items()}
w = {int(k): tuple(v) for k, v in tw["w"].items()}
dai1.RUN.update(commit=dai1.git_commit()[0], dirty=True, frozen_sha256="sha-A", gate_sha256="gate-A", gpu="test")
args = ("eval", "AFEW", "AFEW", ["IEMOCAP", "MELD"], 0, 0)
for which in (1, 2, 3):
    dai1.run_pass(E, *args, which, tether=teth, wsel=w, fsel=fsel)
z = {wh: np.load(E / "eval" / "units" / f"AFEW_ALL_f0_r0_p{wh}.npz") for wh in (1, 2, 3)}
print("各段の npz に tether・wsel・gate_sha256：", {wh: ("tether" in z[wh].files, "wsel" in z[wh].files, str(z[wh]["gate_sha256"])) for wh in z})
d3 = pd.read_csv(E / "eval" / "units" / "AFEW_ALL_f0_r0_p3.csv", low_memory=False)
print("引き戻しの曲線の行", d3[d3.tag.notna()].groupby(["tag", "method"]).size().to_dict() if "tag" in d3 else "tag 列なし")
# 再開：同じ条件なら読むだけ
for which in (1, 2, 3):
    dai1.run_pass(E, *args, which, tether=teth, wsel=w, fsel=fsel)
print("再開：3 段とも止まらずに読んだ")
# 固定のファイルの sha256 が違えば止まる
dai1.RUN["frozen_sha256"] = "sha-B"
try:
    dai1.run_pass(E, *args, 1, tether=teth, wsel=w, fsel=fsel); print("止まらなかった（誤り）")
except SystemExit as e:
    print("sha256 が違う：止まった", str(e)[:60])
dai1.RUN["frozen_sha256"] = "sha-A"
# 非有限：段 2 の学習器が非有限を返したら保存しない
_fit = dai1.S.fit
def bad_fit(*a, **k):
    res = _fit(*a, **k); res["finite_p"] = np.zeros_like(res["finite_p"]); return res
dai1.run_pass(E, "eval", "AFEW", "AFEW", ["IEMOCAP", "MELD"], 1, 0, 1, tether=teth, wsel=w, fsel=fsel)
dai1.S.fit = bad_fit
try:
    dai1.run_pass(E, "eval", "AFEW", "AFEW", ["IEMOCAP", "MELD"], 1, 0, 2, tether=teth, wsel=w, fsel=fsel); print("止まらなかった（誤り）")
except SystemExit as e:
    print("非有限の段 2：止まった．保存は", (E / "eval" / "units" / "AFEW_ALL_f1_r0_p2.npz").exists(), (E / "eval" / "units" / "AFEW_ALL_f1_r0_p2.csv").exists())
dai1.S.fit = _fit
# 再開の経路の非有限：保存済みの csv に finite=False の行があれば止まる
cp = E / "eval" / "units" / "AFEW_ALL_f1_r0_p1.csv"
d = pd.read_csv(cp, low_memory=False); i = d.index[d.finite.notna()][0]; d.loc[i, "finite"] = False; d.to_csv(cp, index=False)
try:
    dai1.run_pass(E, "eval", "AFEW", "AFEW", ["IEMOCAP", "MELD"], 1, 0, 1, tether=teth, wsel=w, fsel=fsel); print("止まらなかった（誤り）")
except SystemExit as e:
    print("再開の経路の非有限：止まった")
# 前の段のコミットの照合
dai1.RUN["commit"] = "0" * 40
try:
    dai1.run_pass(E, "eval", "AFEW", "AFEW", ["IEMOCAP", "MELD"], 0, 1, 2, tether=teth, wsel=w, fsel=fsel); print("止まらなかった（誤り）")
except (SystemExit, FileNotFoundError) as e:
    print("前の段の保存が無い／コミットが違う：", type(e).__name__, str(e)[:80])
