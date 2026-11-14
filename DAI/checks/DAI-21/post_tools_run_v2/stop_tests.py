"""gate-code：対象の台本を書き換えずに，import して部品を差し替え，止まるべき場合に止まるかを見る．出力は checks の下だけ．
使い方：python stop_tests.py <case>．case：plain（差し替えなし．出力の再現）／swap（prod と avg の鍵を入れ替える）／con（差の対を 1 つ替える）／
         dropspk（文 IEO を除くと話者が 1 人欠ける単位を作る）．"""
import importlib.util
import sys
from pathlib import Path

case = sys.argv[1]
OUT = Path("/home/snics/MER-DA/DAI/checks/DAI-21/post_tools_run_v2") / f"out_{case}"
spec = importlib.util.spec_from_file_location("ph", "/home/snics/MER-DA/DAI/docs/tools_pending/dai21_posthoc_cremad_sentence.py")
ph = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ph)
if case == "swap":
    orig = ph.names_of

    def names_of(K, q, fk):
        n = orig(K, q, fk)
        n["prod"], n["avg"] = n["avg"], n["prod"]
        return n
    ph.names_of = names_of
elif case == "con":
    orig_c = ph.con_of

    def con_of(fk):
        c = orig_c(fk)
        c[0] = (c[0][1], c[0][0])                                        # 1 つ目の対の向きを逆にする
        return c
    ph.con_of = con_of
elif case == "dropspk":
    orig_l = ph.load_units

    def load_units(*a, **k):
        units, cs = orig_l(*a, **k)
        u = units[0]
        spk = u["cl"][0]
        m = u["cl"] == spk
        u["sen"] = u["sen"].copy()
        u["sen"][m] = "IEO"                                               # この話者の発話を全部 IEO と見なす（除くと話者が欠ける）
        return units, cs
    ph.load_units = load_units
sys.argv = ["x", "--out", str(OUT)]
ph.main()
