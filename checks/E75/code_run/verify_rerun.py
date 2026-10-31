"""関門 1 の再検査（527a4ec）．results/ は読むだけ。"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/home/snics/MER-DA")
sys.path[:0] = [str(ROOT / "exp"), str(ROOT / "measure"), str(ROOT / "features")]
from summarize_map import load as load_map
import map as M
O = ROOT / "checks/E75/code_run/rerun_527a4ec"
p = json.loads((O / "provenance.json").read_text())
print("commit", p["commit"], "dirty", p["dirty"], "script_status", repr(p["script_status"]), "n_checks", p.get("n_checks"), "failed", p.get("n_checks_failed"), "gaps", p.get("n_gaps"))
mods = p.get("modules") or p.get("module_state") or {}
print("部品の状態が clean でないもの", {k: v["status"] for k, v in mods.items() if v["status"] != "clean"} if mods else "（キー無し）", "部品の数", len(mods))
C = pd.read_csv(O / "checks.csv", comment="#"); print("検算", len(C), "外れ", int((~C.ok).sum()))
print(C[C.item == 8][["kind", "what", "value", "ok"]].to_string())
print(pd.read_csv(O / "records.csv", comment="#").to_string())
print(pd.read_csv(O / "gaps.csv", comment="#").to_string())
# 項目 6
P6 = pd.read_csv(O / "e75_6_pairs.csv", comment="#")
print("項目 6：割合の範囲", P6.frac_recovered_joint_model.min(), P6.frac_recovered_joint_model.max(),
      "模型の最小が負", int((P6.loc_loss_after_joint_model < -1e-12).sum()), "d≤0 の対を含む組", int((P6.n_pairs_d_le0 > 0).sum()),
      "success でない", int((~P6.joint_success.astype(bool)).sum()), "CE − 模型の最小の最小", float((P6.loc_loss_after_ce_fit - P6.loc_loss_after_joint_model).min()),
      "CE の勾配の最大", float(P6.ce_grad_max_all.max()), float(P6.ce_grad_max_halves.max()))
# 項目 10：α=nat の行で wt と対照が同じか．対照の C と交差検証の地図の合併の C が同じ行での差
for kind in ("audio_wavlm", "audio_hubert", "video"):
    mech = pd.concat([pd.read_csv(f, comment="#", dtype={"K": str, "source": str}) for f in sorted((ROOT / f"results/mech_{kind}").glob("*.csv"))])
    w = mech[(mech.op == "wt") & (mech.method == "merged")].copy()
    cb, ctrl = [], []
    for _, r in w.iterrows():
        c, _ = M.choose_c({c: float(r[f"cv_anat_C{c:g}"]) for c in M.C_GRID}); cb.append(c); ctrl.append(float(r[f"test_anat_C{c:g}"]))
    w["c_ctrl"], w["ctrl"] = cb, ctrl
    nat = w[w.alpha.astype(str) == "nat"]
    cv = load_map(ROOT / f"results/map_{kind}"); cv = cv[cv.method == "merged"][["target", "source", "K", "rep", "C", "uar"]]
    x = w.merge(cv, on=["target", "source", "K", "rep"], suffixes=("", "_cv"))
    same = x[np.isclose(x.c_ctrl, x.C_cv)]
    d = same.ctrl - same.uar_cv
    print(f"項目 10 {kind}: α=nat の行 {len(nat)}，wt と対照の差の最大 {float((nat.uar - nat.ctrl).abs().max()):.2e}，"
          f"wt の C と対照の C が違う行 {int((~np.isclose(nat.C, nat.c_ctrl)).sum())}；"
          f"対照と交差検証の地図で C が違う行 {int((~np.isclose(x.c_ctrl, x.C_cv)).sum())}／{len(x)}；"
          f"C が同じ行の差：最大 {float(d.abs().max()):.4f}，符号つきの平均 {float(d.mean()):.5f}，差のある行 {int((d.abs() > 1e-6).sum())}")
