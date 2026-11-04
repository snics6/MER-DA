import dai1_select as SEL
sel = {"A|5|joint_a1|one": {"rho": 0.25, "a": 1}, "V|5|joint_a1|one": {"rho": 4.0, "a": 1}, "A|5|joint|one": {"rho": 0.25, "a": 16},
       "A|5|kbias|-": {"Cb": 10.0}, "A|5|coreg|cr": {"rho": 0.0}, "V|5|coreg|cr": {"rho": 0.25}, "A|5|cotrain|ct": {"rho": 0.0625}}
edge = {}
for key, ch in sel.items():
    meth = key.split("|")[2]; gm = "joint" if meth == "joint_a1" else meth
    keys = [(k, v) for k, v in SEL.GRIDS[gm] if len(v) > 2 and k in ch and not (meth == "joint_a1" and k == "a")]
    if not keys: continue
    hit = any(float(ch[k]) in (float(min(v)), float(max(v))) for k, v in keys)
    edge.setdefault(meth, [0, 0]); edge[meth][0] += int(hit); edge[meth][1] += 1
print("期待：joint_a1 [1,2]，joint [1,1]，kbias なし，coreg [1,2]，cotrain [1,1]")
print("結果：", edge, "coreg の格子", SEL.GRIDS["coreg"])
import inspect; src = inspect.getsource(SEL.select_T); print("台本の式と同じ" if "len(v) > 2 and k in ch and not (meth == \"joint_a1\" and k == \"a\")" in src else "台本の式と違う")
