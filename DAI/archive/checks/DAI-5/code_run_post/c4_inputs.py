"""計算の後の点検 4：仮のターゲットの回の保存・評価の段の守りの記録が，評価の回の後も変わっていないか．影響の割合と _teacher_stats の行の数。"""
import hashlib, json, subprocess
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/home/snics/MER-DA")
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
fz = json.load(open(ROOT / "results/dai5/select/frozen.json")); gate = json.load(open(ROOT / "DAI/frozen/eval_gate_dai5.json"))
n = bad = 0
for T in ["AFEW", "IEMOCAP", "MOSEI", "MELD"]:
    for nm, h in fz[T]["pseudo_unit_sha256"].items():
        n += 1; bad += sha(ROOT / "results/dai5/pseudo/units" / nm) != h
print("pseudo の保存の sha256（固定のファイルの記録）：", n, "件中 食い違い", bad)
pc = set()
for p in sorted((ROOT / "results/dai5/pseudo/units").glob("*_p5.csv")):
    d = pd.read_csv(p, low_memory=False, usecols=["commit", "dirty", "gpu"]); pc |= set(zip(d.commit.astype(str), d.dirty.astype(str), d.gpu.astype(str)))
print("pseudo の行のコミット・dirty・計算機：", {(c[:7], dd, g) for c, dd, g in pc})
bad = [r for r, h in gate["required"].items() if sha(ROOT / r) != h]
print("eval_gate_dai5.json の required", len(gate["required"]), "件のうち，今のファイルと sha256 が違うもの", bad)
bad = [nm for nm, h in gate["input_sha256"].items() if sha(ROOT / "results/dai1/eval/units" / nm) != h]
print("input_sha256", len(gate["input_sha256"]), "件のうち今と違うもの", len(bad))
cp = json.load(open(ROOT / "DAI/frozen/dai5_eval_inputs_sha256.json")); print("dai5_eval_inputs_sha256.json と input_sha256 が同じ", cp == gate["input_sha256"])
print("git diff f186d02 HEAD（DAI/exp DAI/frozen DAI/run DAI/docs/実験5_設計.md）:", subprocess.run(["git", "diff", "--stat", "f186d02", "HEAD", "--", "DAI/exp", "DAI/frozen", "DAI/run", "DAI/docs/実験5_設計.md"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or "差なし")
nts = set(); infl_nan = infl_out = 0
for p in sorted((ROOT / "results/dai5/eval/units").glob("*_p5.csv")):
    d = pd.read_csv(p, low_memory=False); nts.add(int((d.method == "_teacher_stats").sum()))
    z = np.load(str(p).replace(".csv", ".npz"))
    for k in z.files:
        if "_infl_" in k:
            I = z[k]; infl_nan += int(np.isnan(I).sum()); infl_out += int(((I < -1e-12) | (I > 1 + 1e-12)).sum())
print("評価の回の _teacher_stats の行の数（1 単位あたり）", nts, "．影響の割合の NaN", infl_nan, "，0〜1 の外", infl_out)
