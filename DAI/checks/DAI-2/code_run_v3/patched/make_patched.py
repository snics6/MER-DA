"""gate-code (DAI-2 code_v3)：dai2_stats.py の eval の守りに，単位のコミット・gate_sha256・families.csv のコミットの確かめを足した版（検査用の見本．DAI/exp は書き換えない）。"""
import sys
src = open("/home/snics/MER-DA/DAI/exp/dai2_stats.py", encoding="utf-8").read()
def rep(a, b):
    global src
    assert src.count(a) == 1, a
    src = src.replace(a, b)
rep('RUN = {"mode": "", "commit": "", "dirty": True}', 'RUN = {"mode": "", "commit": "", "dirty": True, "gate": ""}')
rep('    out["eval_gate"] = dai1.sha(gp)\n    return out', '    out["eval_gate"] = dai1.sha(gp)\n    RUN["gate"] = out["eval_gate"]\n    return out')
rep('        gseeds = [int(x) for x in fz2["_meta"]["grad_seeds_eval"]]\n',
    '        gseeds = [int(x) for x in fz2["_meta"]["grad_seeds_eval"]]\n'
    '        seen_commit, seen_gate = set(fam.commit.astype(str)), set()          # 単位・families.csv のコミットと gate_sha256（指摘 2-N）\n')
rep('''                    sys.exit(f"{T}: 評価の回の単位 f{f} r{r} の保存がそろっていない")
''', '''                    sys.exit(f"{T}: 評価の回の単位 f{f} r{r} の保存がそろっていない")
                for d_, ps in ((d2, ("lin", "coral") + (("map", "xmuda") if (f, r) in grad_units else ())), (d1, ("p1", "p3"))):
                    for p_ in ps:
                        z_ = np.load(f"{d_ / 'eval' / 'units' / f'{T}_ALL_f{f}_r{r}'}_{p_}.npz")
                        seen_commit.add(str(z_["commit"])); seen_gate.add(str(z_["gate_sha256"]))
                        want_fz = info["dai2_frozen_sha256"] if d_ is d2 else info["dai1_frozen_sha256"]
                        if str(z_["frozen_sha256"]) != want_fz:
                            sys.exit(f"{T}: 単位 f{f} r{r} の {p_} の frozen_sha256 が今の固定のファイルと違う")
''')
rep('''        return jobs, info
    for T in targets:''', '''        if seen_commit != {RUN["commit"]}:
            sys.exit(f"今のコミット {RUN['commit'][:10]} と，評価の回の単位・families.csv のコミット {sorted(x[:10] for x in seen_commit)} が違う．"
                     "評価の回のコミットの作業木で流す（DAI/run/eval_procedure.md の 8）")
        if seen_gate != {RUN["gate"]}:
            sys.exit(f"単位の gate_sha256 {sorted(x[:10] for x in seen_gate)} が今の eval_gate.json {RUN['gate'][:10]} と違う")
        return jobs, info
    for T in targets:''')
open(sys.argv[1], "w", encoding="utf-8").write(src)
