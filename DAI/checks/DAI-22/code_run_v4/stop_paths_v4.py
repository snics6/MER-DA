"""gate-code 再検査（DAI-22 第 4 版）：第 4 版の止める規則が偽の入力で実際に止まるかを確かめる．値は出さない．
mode:
  corp_afew    t_corp に AFEW の分割 0・種 0〜9 を偽の入力なしで流す（解き直しは 0 行のはず）
  corp_iemo    同じ置き場に IEMOCAP の分割 0・種 0〜2 を，実験 3 の保存の一致を 0.01 ずらして流す（コーパスを流し終えた時点で止まるはず）
  corp_restart 同じ置き場で IEMOCAP を偽の入力なしで流し直す（3 単位とも飛ばし，済みの単位の csv を数えて止まるはず）
  corp_sum     同じ置き場で，格子を分割 0 × AFEW 種 0〜9・IEMOCAP 種 0〜2 に縮め，非 partial の summary（コーパスごとの上限で止まるはず）
  corp_sum_ok  corp_sum と同じで RESOLVED_MIN を大きくした対照（最後まで通るはず）
  p2bad / p3bad / bothbad / nonebad
               AFEW f0 r0 で，読み込む p2.csv の product の行・p3.csv の one の行・両方・p2.csv の none_AVT の行の uar_pool に 1 を足す
               （p2bad と p3bad は通り，bothbad と nonebad は止まって単位のファイルを書かないはず）
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai22  # noqa: E402

C = Path("/home/snics/MER-DA/DAI/checks/DAI-22/code_run_v4")
mode = sys.argv[1]


def run(argv):
    sys.argv = argv
    try:
        dai22.main()
        print(f"[{mode}] 最後まで通った")
    except SystemExit as e:
        print(f"[{mode}] 止まった：{e}")


def shift_saved(only_T):
    real_run = dai22.run_unit

    def run_unit(T, f, r, fz, out, repro):
        if T == only_T:
            repro = repro.copy(); repro["agree"] = repro["agree"] + 0.01
        return real_run(T, f, r, fz, out, repro)
    dai22.run_unit = run_unit


def bad_csv(p2_prod=False, p3_one=False, p2_none=False):
    real = pd.read_csv

    def rc(p, *a, **k):
        df = real(p, *a, **k)
        name = Path(str(p)).name
        if name.endswith("_p2.csv"):
            if p2_prod:
                df.loc[(df.method == "product") & (df.teacher == "(1.0, 1.0, 1.0)"), "uar_pool"] += 1.0
            if p2_none:
                df.loc[(df.method == "teacher") & (df.teacher == "none_AVT"), "uar_pool"] += 1.0
        if name.endswith("_p3.csv") and p3_one:
            df.loc[(df.method == "_teacher_stats") & df.student.isna() & (df.teacher == "one"), "uar_pool"] += 1.0
        return df
    dai22.pd.read_csv = rc


T_CORP = str(C / "t_corp")
if mode == "corp_afew":
    run(["dai22.py", "--stage", "units", "--out-dir", T_CORP, "--targets", "AFEW", "--folds", "0", "--seeds"] + [str(i) for i in range(10)])
elif mode == "corp_iemo":
    shift_saved("IEMOCAP")
    run(["dai22.py", "--stage", "units", "--out-dir", T_CORP, "--targets", "IEMOCAP", "--folds", "0", "--seeds", "0", "1", "2"])
elif mode == "corp_restart":
    run(["dai22.py", "--stage", "units", "--out-dir", T_CORP, "--targets", "IEMOCAP", "--folds", "0", "--seeds", "0", "1", "2"])
elif mode in ("corp_sum", "corp_sum_ok"):
    real_load = dai22.dai3.load_fz1

    def load(p):
        fz, sh = real_load(p)
        fz["IEMOCAP"]["n_seeds_eval"] = 3
        return fz, sh
    dai22.dai3.load_fz1 = load
    dai22.DD.N_FOLDS = 1
    dai22.TOL_PSEUDO = 1e9          # 縮めた格子では実験 14 の 50 単位の点推定と合わないので，この照合だけを外す
    if mode == "corp_sum_ok":
        dai22.RESOLVED_MIN = 10 ** 6
    run(["dai22.py", "--targets", "AFEW", "IEMOCAP", "--stage", "summary", "--out-dir", T_CORP])
    print("summary の中身：", sorted(p.name for p in (Path(T_CORP) / "summary").glob("*")) if (Path(T_CORP) / "summary").exists() else "置き場なし")
elif mode in ("p2bad", "p3bad", "bothbad", "nonebad"):
    bad_csv(p2_prod=mode in ("p2bad", "bothbad"), p3_one=mode in ("p3bad", "bothbad"), p2_none=mode == "nonebad")
    od = C / f"t_{mode}"
    run(["dai22.py", "--stage", "units", "--out-dir", str(od), "--targets", "AFEW", "--folds", "0", "--seeds", "0"])
    print("単位のファイル：", sorted(p.name for p in (od / "units").glob("*")))
    for p in sorted((od / "units").glob("*.csv")):
        x = pd.read_csv(p)
        x = x[x.readout == "R0"]
        print("p2 との差 > 0.5 の行", int((x.dai1_prop_p2_abs_diff > 0.5).sum()), "p3 との差 > 0.5 の行", int((x.dai1_prop_p3_abs_diff > 0.5).sum()))
