"""補足 2（第 II 部の式と詳細）の図の値を CSV に書き出す（資料 2026-10-27-TAC論文）．

論文の台帳（paper/scripts/ledger/part2.py）と同じ結果ファイル・同じ集計（ターゲットごとに反復を平均した表を，ターゲットで平均）で値を作り，
figures/supp_p2_*.csv に置く．台帳（paper/results/numbers.json）に同じ量の key があるものは照合し，食い違えば止める．
各 CSV の先頭の # 行に出所を書く（pgfplots は # の行を読み飛ばす）．結果ファイルを読むだけで，数秒で終わる．

    cd slides/2026-10-27-TAC論文 && ../../.venv/bin/python scripts/export_supp_part2.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd

DECK = pathlib.Path(__file__).resolve().parents[1]
MERDA = DECK.parents[1]
PAPER = MERDA / "paper"
sys.path.insert(0, str(PAPER / "scripts"))
sys.dont_write_bytecode = True

from ledger import part2 as P2  # noqa: E402

OUT = DECK / "figures"
LEDGER = json.loads((PAPER / "results" / "numbers.json").read_text())
N_CHECKED = [0]
KS = ["0", "1", "5", "10", "25"]


def check(key: str, v: float, tol: float = 1e-6) -> None:
    """台帳に key があれば値を照合する（無ければ止める．照合のために呼ぶ key は台帳にあるはずなので）．"""
    if key not in LEDGER:
        raise SystemExit(f"export_supp_part2: 台帳に {key} が無い")
    w = float(LEDGER[key]["value"])
    if abs(w - float(v)) > tol:
        raise SystemExit(f"export_supp_part2: {key} は図の値 {v} と台帳の値 {w} が食い違う")
    N_CHECKED[0] += 1


def write(name: str, df: pd.DataFrame, src: str) -> None:
    OUT.mkdir(exist_ok=True)
    head = "".join(f"# {line}\n" for line in src.strip().splitlines())
    (OUT / name).write_text(head + df.to_csv(index=False, float_format="%.6g"))
    print(f"  {name}: {len(df)} 行")


def rel(p: pathlib.Path) -> str:
    return str(p.relative_to(MERDA))


# ---- E58：出発点の向きの誤差 E0 の広がりと，捉える割合 ---------------------------

def e58() -> None:
    D = pd.read_csv(P2.F_E58, comment="#")
    cols = ["d", "d_eff", "frac_top20", "frac_top100", "null20", "r_random_true", "r_LLL", "r_UA", "dim_UA", "r_UB", "dim_UB", "r_UAB"]
    R = D.groupby("kind")[cols].mean()                                           # ターゲットの平均（台帳 part2._e58 と同じ）
    rows = []
    for x, (kind, tok) in enumerate((("audio_wavlm", "a"), ("video", "v"), ("text_roberta", "t"))):
        r = R.loc[kind]
        row = dict(x=x, kind=tok, d=r.d, d_eff=r.d_eff, dim_UA=r.dim_UA, dim_UB=r.dim_UB,
                   **{c: 100 * r[c] for c in ("frac_top20", "frac_top100", "null20", "r_random_true", "r_LLL", "r_UA", "r_UB", "r_UAB")})
        check(f"p2-e58-r-random-{tok}", row["r_random_true"])
        check(f"p2-e58-r-ua-{tok}", row["r_UA"])
        check(f"p2-e58-r-ub-{tok}", row["r_UB"])
        check(f"p2-e58-deff-z-{tok}", row["d_eff"])
        check(f"p2-e58-ratio-lll-{tok}", row["r_LLL"] / row["r_random_true"])
        rows.append(row)
    write("supp_p2_e58.csv", pd.DataFrame(rows), f"""出所：{rel(P2.F_E58)}（1 行 = 種類 × ターゲット，反復 0〜9 の平均）を種類ごとにターゲットで平均（台帳 part2._e58 と同じ）．
null20 は 20／d（等方な 20 次元が E0 を捉える割合の期待値），frac_top20・frac_top100 は E0 のうち候補の分散の上位 20・100 方向に入る割合，
r_* は K=5 の 20 本（または部分空間）が E0 を捉える割合 r = ‖P E0ᵀ‖²／‖E0‖²．百分率．d は特徴の次元，d_eff は生徒の座標の実効次元，dim_* は部分空間の次元．
台帳 p2-e58-r-random-*・r-ua-*・r-ub-*・deff-z-*・ratio-lll-* と照合した．A 7，V 6，T 4 ターゲット""")


# ---- E59 の追加：同じモダリティの見込みの蒸留 ------------------------------------

def e59() -> None:
    d = pd.read_csv(P2.F_E59K, dtype={"K": str}, comment="#")
    d = d.assign(gain=d.meta - d.e39, ctrl=d.p_ctrl - d.e39, info=d.meta - d.p_ctrl)
    G = d.groupby(["kind", "K"])[["gain", "ctrl", "info"]].mean()
    rows = []
    for x, K in enumerate(KS):
        r = dict(x=x, K=K)
        for kind, tok in (("audio_wavlm", "a"), ("video", "v"), ("text_roberta", "t")):
            for c in ("gain", "ctrl", "info"):
                r[f"{tok}_{c}"] = float(G.loc[(kind, K), c])
            check(f"p2-e59k-{tok}-k{K}", r[f"{tok}_gain"])
            if tok != "t" and K in ("0", "1", "5"):
                check(f"p2-e59k-ctrl-{tok}-k{K}", r[f"{tok}_ctrl"])
                check(f"p2-e59k-info-{tok}-k{K}", r[f"{tok}_info"])
        rows.append(r)
    write("supp_p2_e59.csv", pd.DataFrame(rows), f"""出所：{rel(P2.F_E59K)}（1 行 = 種類 × ターゲット × K，反復 0〜9 の平均）．台帳 part2._e59k と同じ集計．
<種類>_gain = meta − e39（K ごとに仮のターゲットで選んだ見込みの蒸留 − 5 つの値を選んだ出発点の生徒），<種類>_ctrl = p_ctrl − e39（出発点自身の予測を弱いラベルにした対照＝釣り合い直し），
<種類>_info = meta − p_ctrl（見込みの情報の分）．ポイント．A 7，V 6，T 4 ターゲット．台帳 p2-e59k-* と照合した""")


# ---- E60：モダリティをまたぐ蒸留（基本版）の全生徒・全 K ---------------------------

WORLDS = [("fuse-audio-video", "av"), ("fuse-video-text", "vt"), ("fuse-audio-text", "at"), ("fuse-audio-video-text", "avt")]


def stok(s: str) -> str:
    return P2.STOK.get(s, "f")


def e60() -> None:
    d = pd.read_csv(P2.F_E60, dtype={"K": str}, comment="#")
    d = d.assign(gain=d.meta - d.e39)
    G = d.groupby(["world", "student", "K"]).gain.mean()
    rows = []
    for x, K in enumerate(KS):
        r = dict(x=x, K=K)
        for w, wt in WORLDS:
            for s in [s for s in sorted(set(d[d.world == w].student))]:
                key = f"{wt}_{stok(s)}"
                r[key] = float(G.loc[(w, s, K)])
                check(f"p2-e60-{wt}-{stok(s)}-k{K}", r[key])
        rows.append(r)
    write("supp_p2_e60.csv", pd.DataFrame(rows), f"""出所：{rel(P2.F_E60)}（1 行 = 世界 × ターゲット × 生徒 × K，反復 0〜9 の平均）．台帳 part2._e60 と同じ集計．
列 <世界>_<生徒> は meta − e39（K ごとに仮のターゲットで選んだモダリティをまたぐ弱いラベル − 5 つの値を選んだ出発点の生徒）のターゲットの平均．
生徒 a 音声，v 顔，t 言葉，f 融合．ターゲットは AV 6，VT・AT・AVT 4．ポイント．台帳 p2-e60-<世界>-<生徒>-k<K> の 65 個と照合した""")


def e60_choice() -> None:
    """基本版で K ごとに仮のターゲットが選んだ弱いラベルの比重 m（単独の生徒）と先生（融合の生徒）の数．"""
    d = pd.read_csv(P2.F_E60, dtype={"K": str}, comment="#")
    single = d[d.student.isin(P2.STOK)]
    fused = d[~d.student.isin(P2.STOK)]
    rows = []
    for x, K in enumerate(KS):
        s = single[single.K == K]
        f = fused[fused.K == K]
        r = dict(x=x, K=K, n_single=len(s), n_fused=len(f))
        for m in (0, 100, 500, 2000):
            r[f"s_m{m}"] = int(((s.b_star == "none") if m == 0 else ((s.b_star != "none") & (s.m_star == m))).sum())
        for b in ("pF", "poe", "agree"):
            r[f"s_{b}"] = int((s.b_star == b).sum())
        r["f_none"] = int((f.b_star == "none").sum())
        r["f_poe0"] = int((f.b_star == "poe0").sum())
        r["f_ref"] = int(f.b_star.str.match(r"^ref\d").sum())
        r["f_refdirect"] = int(f.b_star.str.startswith("refdirect").sum())
        if r["s_m0"] + r["s_m100"] + r["s_m500"] + r["s_m2000"] != r["n_single"] or r["s_pF"] + r["s_poe"] + r["s_agree"] + r["s_m0"] != r["n_single"]:
            raise SystemExit("export_supp_part2: 単独の生徒の選択の数が合わない")
        if r["f_none"] + r["f_poe0"] + r["f_ref"] + r["f_refdirect"] != r["n_fused"]:
            raise SystemExit("export_supp_part2: 融合の生徒の選択の数が合わない")
        rows.append(r)
    write("supp_p2_choice.csv", pd.DataFrame(rows), f"""出所：{rel(P2.F_E60)} の b_star（選んだ先生）・m_star（選んだ比重 m）列．1 行 = 世界 × ターゲット × 生徒 × K．
K ごとに，仮のターゲットが選んだものを（世界, ターゲット, 生徒）の数で数えた．単独の生徒は 9 組 × ターゲットで 40，融合の生徒は 4 世界 × ターゲットで 18．
s_m0 は「弱いラベルを使わない」，s_m<m> は比重 m を選んだ数，s_<先生> は先生ごとの数．f_ref は ref100・ref500（融合の出発点が教えた単独の生徒の予測の積），
f_refdirect はその積をそのまま分類器にしたもの，f_poe0 は単独の出発点の予測の積．候補の格子は exp/multimodal_distill.py の BEL1・MEFF1・BEL2・MEFF2""")


def ext_choice() -> None:
    """拡張版（oursx）で仮のターゲットが選んだ先生と重みづけの数（K=0〜25 をまとめる）．"""
    worlds = {"fuse-audio-video": "av", "fuse-video-text": "vt", "fuse-audio-text": "at", "fuse-audio-video-text": "avt"}
    rows = []
    for w, wt in worlds.items():
        d = MERDA / "results" / f"baselines_{w}"
        for f in sorted(d.glob("*_meta.csv")):
            m = pd.read_csv(f, dtype={"K": str, "meff": str, "param": str}, keep_default_na=False)
            m = m[(m.method == "oursx") & (m.chosen == True)]  # noqa: E712
            for r in m.itertuples():
                teacher, wgt = ("none", "none") if r.cand == "none" else r.param.split(":")
                rows.append(dict(world=wt, target=f.stem[:-5], student=stok(r.student), K=r.K, teacher=teacher,
                                 weight="hard" if wgt.startswith("hard") else wgt, meff=r.meff))
    D = pd.DataFrame(rows)
    D["group"] = D.student.where(D.student == "f", "s")
    out = []
    for x, (g, lab) in enumerate((("s", "単独の生徒（AVT）"), ("f", "融合の生徒"))):
        q = D[D.group == g]
        r = dict(x=x, group=g, label=lab, n=len(q), none=int((q.teacher == "none").sum()))
        for t in ("pFK", "ref"):
            for wgt in ("bal", "raw", "hard"):
                r[f"{t}_{wgt}"] = int(((q.teacher == t) & (q.weight == wgt)).sum())
        if sum(v for k, v in r.items() if k.startswith(("pFK_", "ref_"))) + r["none"] != r["n"]:
            raise SystemExit("export_supp_part2: 拡張版の選択の数が合わない")
        out.append(r)
    write("supp_p2_ext.csv", pd.DataFrame(out), """出所：results/baselines_fuse-<世界>/<ターゲット>_meta.csv の method = oursx かつ chosen の行（cand・param）．
拡張版の候補（先生 pFK・ref × 重みづけ bal・raw・hardτ × m）と「使わない」から仮のターゲットが選んだものを，
（世界, ターゲット, 生徒, K）の数で数えた．K = 0・1・5・10・25 をまとめる．単独の生徒は AVT の音声・顔・言葉（4 ターゲット），
融合の生徒は AV（6）・VT・AT・AVT（4）．hard は τ = 0.5・0.7・0.9 をまとめた数．台本 exp/baselines_distill.py の oursx_rows""")


def main() -> None:
    print("export_supp_part2: 書き出す")
    e58()
    e59()
    e60()
    e60_choice()
    ext_choice()
    print(f"export_supp_part2: 台帳と {N_CHECKED[0]} 個の値を照合した（食い違いなし）")


if __name__ == "__main__":
    main()
