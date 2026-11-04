"""gate-code の所要の内訳：評価の回に見立てた 1 単位（AFEW f0 r0 と IEMOCAP f0 r0）の eval_unit を，部品ごとの時間に分ける．値は出さない．"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
exec(open(HERE / "run_trial.py").read().split("sys.argv = [")[0])
import torch
import dai3 as D3, dai1
D3.RUN["commit"], D3.RUN["dirty"] = dai1.git_commit()
fz1 = json.load(open(HERE / "d1e" / "select" / "frozen.json")); fz3 = json.load(open(HERE / "oute" / "frozen.json"))
acc = {}
def wrap(name):
    f = getattr(D3, name)
    def g(*a, **k):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.time(); r = f(*a, **k)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        acc.setdefault(name, [0.0, 0]); acc[name][0] += time.time() - t0; acc[name][1] += 1
        return r
    setattr(D3, name, g)
for n in ("refit_item", "solve_b", "repro_fit", "rebuilt_check", "pred_rows", "eval_subsets"):
    wrap(n)
for T in ("AFEW", "IEMOCAP"):
    acc.clear()
    sv = D3.Saved(HERE / "d1e", "eval", T, T, 0, 0, fz1[T])
    t0 = time.time(); R = D3.eval_unit(sv, fz3, first_seed=True); tot = time.time() - t0
    print(T, "テスト", len(sv.y), "複製", len(R["keys"]), "候補", sv.u.n, f"合計 {tot:.1f} 秒", {k: (round(v[0], 1), v[1]) for k, v in acc.items()},
          "GPU の最大", f"{torch.cuda.max_memory_allocated()/2**30:.2f} GiB" if torch.cuda.is_available() else "")
