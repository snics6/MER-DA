"""gate-code（評価の回の起動の台本）：評価の段が固定のファイルから読む鍵が全部そろうかを，評価のデータに触れずに確かめる．
使い方：./.venv/bin/python DAI/checks/DAI-eval/code_run/m3_frozen_keys.py dai1   （今すぐ流せる）
        ./.venv/bin/python DAI/checks/DAI-eval/code_run/m3_frozen_keys.py dai2   （DAI/frozen/dai2_frozen.json ができてから）
成績の値は出さない．"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "exp"))
import dai1, dai_data as DD  # noqa: E402

ROOT = DD.ROOT
mode = sys.argv[1]
fz1 = json.load(open(ROOT / "DAI/frozen/dai1_frozen.json"))
bad = []
for T in DD.CORPORA:
    s1 = fz1[T]
    teth = {(k.split("|")[0], int(k.split("|")[1])) for k in s1["tether"]}
    need_t = {(kind, K) for kind in DD.KINDS for K in DD.KS}
    if teth != need_t:
        bad.append((T, "tether", sorted(need_t ^ teth)[:5]))
    if sorted(int(k) for k in s1["w"]) != list(dai1.KS_ALL):
        bad.append((T, "w", sorted(s1["w"])))
    if int(s1.get("n_seeds_eval", 0)) not in (10, 20):
        bad.append((T, "n_seeds_eval", s1.get("n_seeds_eval")))
    if mode == "dai1":
        try:
            fp = dai1.frozen_points(s1)
            assert len(fp) == len(DD.STUDENTS) * len(dai1.KS_ALL)
        except Exception as e:                                           # noqa: BLE001
            bad.append((T, "frozen_points", repr(e)))
if mode == "dai2":
    import dai2, dai2_grad as G  # noqa: E402
    fz2 = json.load(open(sys.argv[2] if len(sys.argv) > 2 else ROOT / "DAI/frozen/dai2_frozen.json"))
    m = fz2["_meta"]
    for n in dai2.OWN_SCRIPTS:
        if dai2.sha(dai2.HERE / n) != m["script_sha256"][n]:
            bad.append(("_meta", "script_sha256", n))
    if dai2.sha(ROOT / "DAI/frozen/dai1_frozen.json") != m["dai1_frozen_sha256"]:
        bad.append(("_meta", "dai1_frozen_sha256", ""))
    cut = m["grad_cut"]
    if list(m.get("grad_seeds_eval", [])) != list(G.GRAD_SEEDS_EVAL):
        bad.append(("_meta", "grad_seeds_eval", m.get("grad_seeds_eval")))
    idx = set(G.map_problems(cut["mme_sw01"], cut["iter200"]))
    xidx = set(G.xm_problems(cut["iter200"]))
    for T in DD.CORPORA:
        if T not in fz2:
            bad.append((T, "missing target", "")); continue
        if "partial_targets" in m:
            bad.append((T, "partial_targets", m["partial_targets"]))
        sel = fz2[T]["select"]
        try:
            for s in DD.STUDENTS:
                for K in DD.KS:
                    int(sel[f"{s}|{K}|coral"]["r"])
                for K in dai1.KS_ALL:
                    for m_ in ("dann", "mme", "mapctl"):
                        p = sel[f"{s}|{K}|{m_}"]
                        for c in [G.map_key(m_, p, K)] + ([G.map_key("mapctl", p, K)] if m_ != "mapctl" else []):
                            if c not in idx:
                                bad.append((T, "map point not in grid", c))
                    if s in "AVT" and len(s) == 1:
                        p = sel[f"{s}|{K}|xmuda"]
                        if (float(p["lr"]), int(p["iters"]), float(p["kap"]), float(p["mu"]), K) not in xidx:
                            bad.append((T, "xmuda point not in grid", (s, K)))
                        if G.map_key("mapctl", p, K) not in idx:
                            bad.append((T, "ctl of xmuda not in grid", (s, K)))
        except KeyError as e:
            bad.append((T, "KeyError", str(e)))
print("食い違い：", len(bad))
for b in bad[:30]:
    print(" ", b)
