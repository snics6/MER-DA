"""gate-code（DAI-9）：1 単位を流し，DAI-8 の試走の同じ単位と raw の判定を照合する。値は出さない。"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path("/home/snics/MER-DA")
sys.path.insert(0, str(ROOT / "DAI" / "exp"))
import dai9_probe as P9  # noqa: E402
out = Path(sys.argv[1]); T, c, f, r = sys.argv[2].split(",")
fz = json.loads(P9.FROZEN.read_text())
torch.cuda.reset_peak_memory_stats(); t0 = time.time()
P9.run_unit(T, c, int(f), int(r), out=out, fz=fz, commit="gatecode-test")
print(f"所要 {time.time()-t0:.0f} 秒，GPU 割り当てのピーク {torch.cuda.max_memory_allocated()/2**20:.0f} MiB")
z9 = np.load(out / "rows" / f"{c}_f{f}_r{r}.npz")
p8 = ROOT / "DAI/checks/DAI-8/code_run_v3/run/rows" / f"{T}_{c}_f{f}_r{r}.npz"
if p8.exists():
    z8 = np.load(p8)
    pairs = [("raw_ldaO", "AVT_ldaO_a0.5"), ("raw_emtruth", "AVT_emtruthconv_a0.5_b0.0"), ("raw_emrand", "AVT_emrandconv_a0.5_b0.0")]
    pairs += [(f"K{K}_raw_ldaK", f"K{K}_AVT_ldaK_a0.5") for K in P9.KS_RULE]
    pairs += [(f"K{K}_raw_emK", f"K{K}_AVT_emfromK_a0.5_b0.0") for K in P9.KS_RULE]
    print("テストの sample_id の一致", bool((z9["test_sample_id"] == z8["test_sample_id"]).all()))
    for a, b in pairs:
        print(f"{a} と DAI-8 の {b}：判定の不一致 {int((z9[a] != z8[b]).sum())}/{len(z9[a])}")
print("判定の鍵", sum(1 for k in z9.files if z9[k].dtype == np.int8 and k != "test_y"), "診断の鍵", len(json.loads(str(z9["diag"]))),
      "大きさ", f"{(out/'rows'/f'{c}_f{f}_r{r}.npz').stat().st_size/2**20:.2f} MiB")
