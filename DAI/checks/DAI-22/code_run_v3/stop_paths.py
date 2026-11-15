"""gate-code 再検査（DAI-22 第 3 版）：止める規則が偽の入力で実際に止まるかを確かめる．値は出さない．
mode:
  fake_fail   resolve_r0 を「予測が 1 本違う」と返す偽物にし，AFEW f0 r0 で units が止まるか（照合の前にファイルは書かれる）
  restart     同じ置き場で元の台本を流し直す（飛ばされるはず）
  sum_partial 同じ置き場で summary --allow-partial（I14b の規則で止まるはず）
  many_units  TOL_REPRO を負にしてすべての R0 の行で解き直し，AFEW f0 の r0..r3 で units が「多すぎる」で止まるか
  many_sum    many_units の置き場で，格子を f0 × r0..r2 に縮め，TOL_PSEUDO を大きくして非 partial の summary が「多すぎる」で止まるか
  ok_sum      many_sum と同じだが RESOLVED_MIN を大きくして，規則を外すと summary が最後まで通るか（対照）
  p2_fail     TOL_DAI1 を負にして units が p2.csv の照合で止まり，単位のファイルを書かないか
"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai22  # noqa: E402

C = Path("/home/snics/MER-DA/DAI/checks/DAI-22/code_run_v3")
mode = sys.argv[1]
base = ["dai22.py", "--targets", "AFEW", "--folds", "0"]


def run(argv):
    sys.argv = argv
    try:
        dai22.main()
        print(f"[{mode}] 最後まで通った")
    except SystemExit as e:
        print(f"[{mode}] 止まった：{e}")


def shift_saved():
    """実験 3 の保存の一致を 0.01 ずらして渡す（照合が本当に 1e-3 を超えて外れる偽の入力）．"""
    real_run = dai22.run_unit

    def run_unit(T, f, r, fz, out, repro):
        repro = repro.copy(); repro["agree"] = repro["agree"] + 0.01
        return real_run(T, f, r, fz, out, repro)
    dai22.run_unit = run_unit


if mode == "fake_fail":
    shift_saved()
    real = dai22.resolve_r0

    def fake(*a, **k):
        o = real(*a, **k)
        for t in dai22.TOL_RESOLVE:
            o[f"n_pred_diff_tol{t:g}"] = 1
        return o
    dai22.resolve_r0 = fake
    run(base + ["--stage", "units", "--out-dir", str(C / "t_fail"), "--seeds", "0"])
elif mode == "restart":
    run(base + ["--stage", "units", "--out-dir", str(C / "t_fail"), "--seeds", "0"])
elif mode == "sum_partial":
    run(base + ["--stage", "summary", "--out-dir", str(C / "t_fail"), "--allow-partial"])
elif mode == "many_units":
    shift_saved()
    run(base + ["--stage", "units", "--out-dir", str(C / "t_many"), "--seeds", "0", "1", "2", "3"])
elif mode in ("many_sum", "ok_sum"):
    real_load = dai22.dai3.load_fz1

    def load(p):
        fz, sh = real_load(p)
        fz["AFEW"]["n_seeds_eval"] = 3
        return fz, sh
    dai22.dai3.load_fz1 = load
    dai22.DD.N_FOLDS = 1
    dai22.TOL_PSEUDO = 1e9          # 3 単位では実験 14 の 50 単位の点推定と合わないので，この照合だけを外す
    if mode == "ok_sum":
        dai22.RESOLVED_MIN = 10 ** 6
    run(["dai22.py", "--targets", "AFEW", "--stage", "summary", "--out-dir", str(C / "t_many")])
elif mode == "p2_fail":
    dai22.TOL_DAI1 = -1.0
    run(base + ["--stage", "units", "--out-dir", str(C / "t_p2"), "--seeds", "0"])
    print("単位のファイル：", sorted(p.name for p in (C / "t_p2" / "units").glob("*")))
