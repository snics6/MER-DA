import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/home/snics/MER-DA")
sys.path[:0] = [str(ROOT / "exp"), str(ROOT / "measure"), str(ROOT / "features")]
from data import load_any, groups_for, GROUP_SPEC
def uar(L, y):
    p = L.argmax(1); return float(np.mean([(p[y == c] == c).mean() for c in range(4)]))
for kind in ("audio_wavlm", "audio_hubert", "video"):
    rows = []
    gs = groups_for(kind)
    for t in gs:
        y = load_any(t, kind)[0].label_4.to_numpy()
        iv = pd.read_csv(ROOT / f"results/interventions_{kind}/{t}.csv", comment="#", dtype={"K": str, "source": str})
        iv = iv[iv.method == "0 ソースのみ（学び直し）"]
        zt = np.load(ROOT / f"results/transfer_scores_{kind}/{t}.npz")
        for s in gs:
            if s == t or GROUP_SPEC[s][0] == GROUP_SPEC[t][0]:
                continue
            L = np.load(ROOT / f"results/shift_{kind}/logits/{s}.npz")[f"t_{t}"].astype(np.float64)
            h = iv[iv.source == s]
            cmode = h.C.mode().iloc[0]
            for _, r in h.iterrows():
                ti = zt[f"test_idx_r{int(r.rep)}"]
                rows.append({"t": t, "s": s, "rep": r.rep, "C": r.C, "C_is_mode": r.C == cmode,
                             "diff": uar(L[ti], y[ti]) - r.uar, "e25_vs_map": r.uar - r.uar_map})
    D = pd.DataFrame(rows)
    nz = D["diff"].abs() > 1e-6
    print(f"{kind}: 行 {len(D)}，差のある行 {int(nz.sum())}（そのうち C が最頻でない行 {int((nz & ~D.C_is_mode).sum())}），"
          f"C が最頻でない行 {int((~D.C_is_mode).sum())}，差の最大 {D['diff'].abs().max():.4f}，符号つきの平均 {D['diff'].mean():.6f}，"
          f"E25 とソースのみの地図の差の最大 {D.e25_vs_map.abs().max():.2e}")
