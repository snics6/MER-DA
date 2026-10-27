"""台帳のモジュール（stats_tables）：論文 6 節「標準の手法との比較」の検定・効果量（st-）と表の数値（tab-）．実験要求 R2（requests.md）．

扱う実験（記録は docs/exploratory-analyses.md の E61・E62・E63・E64 の節）
- 検定（st-）：exp/stats_compare.py（E64）の出力 results/stats_compare/tests.csv から取る．
  差はターゲット × 反復の対（E61・E62 は反復 0〜9，E63 は反復 0〜2）．比べる量は
    a  oursx-best：oursx の調整版 − 最も強い相手の調整版
    b  oursx-e39 ：oursx の調整版 − E39（none）
    c  ours-e39  ：ours（E60 の選んだ形）− E39（none）
    d  best-e39  ：最も強い相手の調整版 − E39（none）
    E63（ft）・E63c（ftc）：ours-vs-frozen（微調整版の私たちの形 − 凍結版の私たちの形），konly-vs-e39（K 本だけの微調整．K=0 は出発点のまま − 凍結の E39），
                         weak-gain（弱いラベルの経路の上乗せ．K=0 は ours − start，K=5 は ours − anchor．λ=0.01）
  key：st-<比較>-<世界>-<生徒>-k<K>-<統計>，st-ft-<比較>-k<K>-<統計>，st-ftc-<比較>-k<K>-<統計>．<統計> は
    n（対の数），mean（差の平均．ポイント），sd（差の標準偏差．ddof=1），dz（mean ÷ sd），pospct（差が正の対の割合．百分率），
    tgtpos（ターゲットごとの差の平均が正のターゲットの数），p（ウィルコクソンの符号順位検定．両側），pholm（族の中の Holm の補正後），
    plt001・pholmlt001（p・pholm が 0.001 未満なら 1．本文で「< 0.001」と書くため）
  Holm の族：a・b・c・d はそれぞれ 55 検定（世界 × 生徒 × K ∈ {0,1,5,10,25}），ft と ftc はそれぞれ 6 検定（3 つの比較 × K ∈ {0,5}）
- 表（tab-）：tab-<世界>-<生徒>-k<K>-<手法>．K ∈ {0, 5}．UAR（％）を反復 0〜9 の平均でターゲットごとに取り，ターゲットで平均した値．
  手法は e39（none），ours，oursx，各相手の調整版（名前は結果ファイルの method．key に使えない「_」は「-」にする：
  cotrain-prod，cotrain-fused，coreg-sum）．値は comparison.py の _world（結果ファイルから選択を作り直すコード）で計算する

照合（食い違えば止める）
- tab- の値は results/stats_compare/table.csv（exp/stats_compare.py が別に書いたコードで作った値）と一致する
- st- の a〜d の mean は comparison.py の cmp-oursx-vs-best・cmp-oursx-vs-e39・cmp-ours-vs-e39・cmp-best-vs-e39 と一致し，
  a・d の最も強い相手の名前は comparison.py の選び方（_world の値から）と一致する
- st-ft・st-ftc の mean は cmp-ft(c)-ours-below-frozen-ours，-konly-below-frozen-e39-k5，-start-below-frozen-e39-k0，-weak-vs-start-k0，-weak-vs-anchor-k5 と一致する
食い違ったら，結果ファイルが変わった後に exp/stats_compare.py を流し直していない可能性が高い．

入力ファイルの大きさ・更新時刻とこのファイル・comparison.py・common.py の中身を鍵にして .cache/stats_tables.json に残す．
"""
from __future__ import annotations

import hashlib
import json
import pathlib

from . import comparison as C
from .common import entry, rel, RESULTS

HERE = pathlib.Path(__file__).resolve()
CACHE = HERE.parent / ".cache" / "stats_tables.json"
SC = RESULTS / "stats_compare"

CMP = {"a": "oursx-best", "b": "oursx-e39", "c": "ours-e39", "d": "best-e39"}
CMP_LEDGER = {"a": "cmp-oursx-vs-best", "b": "cmp-oursx-vs-e39", "c": "cmp-ours-vs-e39", "d": "cmp-best-vs-e39"}
WHAT = {"a": "oursx の調整版 − 最も強い相手 {b} の調整版", "b": "oursx の調整版 − none（E39）",
        "c": "E60 の選んだ形（ours）− none（E39）", "d": "最も強い相手 {b} の調整版 − none（E39）"}
FT_WHAT = {"ours-vs-frozen": {"0": "ours（λ=0.01．K=0）− frozen_ours", "5": "ours（λ=0.01．K=5）− frozen_ours"},
           "konly-vs-e39": {"0": "start（出発点のまま．K=0）− frozen_e39", "5": "k_only（K 本だけ．K=5）− frozen_e39"},
           "weak-gain": {"0": "ours（λ=0.01）− start（K=0）", "5": "ours（λ=0.01）− anchor（λ=0.01）（K=5）"}}
FT_LEDGER = {("ours-vs-frozen", "0"): ("-{t}-ours-below-frozen-ours-k0", -1), ("ours-vs-frozen", "5"): ("-{t}-ours-below-frozen-ours-k5", -1),
             ("konly-vs-e39", "0"): ("-{t}-start-below-frozen-e39-k0", -1), ("konly-vs-e39", "5"): ("-{t}-konly-below-frozen-e39-k5", -1),
             ("weak-gain", "0"): ("-{t}-weak-vs-start-k0", 1), ("weak-gain", "5"): ("-{t}-weak-vs-anchor-k5", 1)}
TAB_KS = ("0", "5")
TOL = 1e-9


# ---------------------------------------------------------------- 入力と控え

def _inputs() -> list[pathlib.Path]:
    return [SC / f for f in ("tests.csv", "table.csv", "best.csv", "provenance.json")] + C._inputs()


def _signature(files) -> str:
    h = hashlib.sha256()
    for f in (HERE, HERE.parent / "comparison.py", HERE.parent / "common.py"):
        h.update(f.read_bytes())
    for f in files:
        st = f.stat()
        h.update(f"{rel(f)}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def entries() -> dict[str, dict]:
    files = _inputs()
    missing = [rel(f) if f.is_relative_to(C.RESULTS.parent) else str(f) for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"stats_tables: {missing[:3]} が無い．MER-DA で ./.venv/bin/python exp/stats_compare.py を流す")
    sig = _signature(files)
    items = None
    if CACHE.exists():
        try:
            c = json.loads(CACHE.read_text(encoding="utf-8"))
            if c.get("signature") == sig:
                items = c["items"]
        except (json.JSONDecodeError, KeyError):
            items = None
    if items is None:
        items = _compute()
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps({"signature": sig, "items": items}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return {k: entry(v[0], v[1], v[2], v[3]) for k, v in items.items()}


# ---------------------------------------------------------------- 計算

def _compute() -> dict[str, list]:
    import numpy as np
    import pandas as pd

    out: dict[str, list] = {}

    def put(key, value, source, fmt="%.1f", note=None):
        if key in out:
            raise ValueError(f"stats_tables: key '{key}' を 2 度作った")
        if not np.isfinite(float(value)):
            raise ValueError(f"stats_tables: key '{key}' の値が有限でない")
        v = int(value) if fmt == "%d" else float(value)
        out[key] = [v, source, fmt, note]

    def fail(msg):
        raise ValueError(f"stats_tables: {msg}．結果ファイルが変わった後に exp/stats_compare.py を流し直していない可能性がある")

    tests = pd.read_csv(SC / "tests.csv", comment="#", dtype={"K": str}, keep_default_na=False)
    for c in ("mean", "sd", "dz", "pos_pct", "p", "p_holm", "p_dedup", "p_dedup_holm"):
        tests[c] = pd.to_numeric(tests[c])
    table = pd.read_csv(SC / "table.csv", comment="#", dtype={"K": str}, keep_default_na=False)
    prov = json.loads((SC / "provenance.json").read_text(encoding="utf-8"))
    bad = {w: c for w, c in prov["selection_check"].items() if c["meta_mismatch"] or c["e60_mismatch"]}
    if bad:
        fail(f"stats_compare の作り直した選択が保存の chosen と食い違う（{bad}）")
    stamp = f"exp/stats_compare.py，コミット {prov['commit'][:7]}{'（未コミットの変更あり）' if prov['dirty'] else ''}"

    # ---- comparison.py の値（照合用）と世界ごとの表
    cmp = {k: e["value"] for k, e in C.entries().items()}
    T, M = {}, {}
    for wk, w in C.WORLDS.items():
        rows, _ = C._world(w)
        T[wk] = pd.DataFrame(rows)
        num = [c for c in T[wk].columns if c not in ("target", "student", "K") and not c.endswith(":pick")]
        M[wk] = T[wk].groupby(["student", "K"])[num].mean()

    def tok(wk, s):
        return "f" if s == C.WORLDS[wk] else C.MODS[s]

    def students(wk):
        return sorted({s for s, _ in M[wk].index}, key=lambda s: "atvf".index(tok(wk, s)))

    def comps(wk, s, K):
        r = M[wk].loc[(s, K)]
        return {c.split(":")[0]: r[c] for c in M[wk].columns if c.endswith(":tuned") and not c.startswith("oursx") and not pd.isna(r[c])}

    def best_of(wk, s, K):
        c = comps(wk, s, K)
        return max(sorted(c), key=lambda m: c[m])

    def picks(wk, s, K, m):
        g = T[wk][(T[wk].student == s) & (T[wk].K == K)]
        return "，".join(f"{r.target} {r[m + ':pick']}" for _, r in g.iterrows())

    # ---- 検定（E61・E62）
    fmt = {"n": "%d", "mean": "%.1f", "sd": "%.1f", "dz": "%.2f", "pospct": "%.0f", "tgtpos": "%d", "p": "%.3f", "pholm": "%.3f",
           "plt001": "%d", "pholmlt001": "%d"}
    ST = "results/stats_compare/tests.csv"

    def put_stats(prefix, r, cond, what, unit_note):
        fam = f"族 {r.family}（{r.family_size} 検定）"
        base = f"{ST}（{stamp}） | {cond}，{what}"
        zero = r.n_zero == r.n
        put(f"{prefix}-n", r.n, f"{base}．ターゲット × 反復の対の数（{r.n_targets} ターゲット） | 数える", fmt["n"])
        put(f"{prefix}-mean", r["mean"], f"{base} | 列 mean：{r.n} 対の差の平均（UAR のポイント）", fmt["mean"])
        put(f"{prefix}-sd", r.sd, f"{base} | 列 sd：{r.n} 対の差の標準偏差（ddof=1）", fmt["sd"])
        put(f"{prefix}-dz", r.dz, f"{base} | 列 dz：差の平均 ÷ 差の標準偏差（対応のある効果量 d_z）", fmt["dz"],
            note="差がすべて 0（比べる 2 つが全部の対で同じ値．選択が none＝E39 と同じ候補になった）なので d_z は 0 とした．" if zero else None)
        put(f"{prefix}-pospct", r.pos_pct, f"{base} | 列 pos_pct：差が正の対の割合（百分率．{r.n_pos}/{r.n}．0 は {r.n_zero}）", fmt["pospct"])
        put(f"{prefix}-tgtpos", r.tgt_pos, f"{base} | 列 tgt_pos：反復で平均した差が正のターゲットの数（全 {r.n_targets}．{r.tgt_means}）", fmt["tgtpos"])
        pn = "差がすべて 0 なので p=1 とした．" if zero else None
        put(f"{prefix}-p", r.p, f"{base} | 列 p：ウィルコクソンの符号順位検定（両側．差 0 の対は落とす．{r.p_method}）", fmt["p"], note=pn)
        put(f"{prefix}-pholm", r.p_holm, f"{base} | 列 p_holm：{fam}の Holm の補正後の p", fmt["pholm"],
            note=(pn or "") + unit_note.format(n=r.n_units_dedup, p=r.p_dedup_holm) or None)
        put(f"{prefix}-plt001", int(r.p < 0.001), f"{base} | 列 p が 0.001 未満なら 1", fmt["plt001"])
        put(f"{prefix}-pholmlt001", int(r.p_holm < 0.001), f"{base} | 列 p_holm が 0.001 未満なら 1", fmt["pholmlt001"])

    UNIT = ("対の独立を仮定した値．同じテストの話者の集合を持つ反復を平均した {n} 単位での Holm 後の p は {p:.3g}（tests.csv の p_dedup_holm．"
            "限界は results/stats_compare/limits.csv）．")
    main = tests[tests.experiment.isin(["E61", "E62"])]
    for wk in C.WORLDS:
        for s in students(wk):
            st = tok(wk, s)
            for K in C.KS:
                b = best_of(wk, s, K)
                for c, name in CMP.items():
                    rr = main[(main.world == wk) & (main.student == s) & (main.K == K) & (main.comparison == c)]
                    if len(rr) != 1:
                        fail(f"tests.csv に {wk}・{s}・K={K}・比較 {c} の行が {len(rr)} 本")
                    r = rr.iloc[0]
                    if c in "ad" and r.best != b:
                        fail(f"{wk}・{s}・K={K} の最も強い相手が stats_compare では {r.best}，comparison.py の選び方では {b}")
                    ck = f"{CMP_LEDGER[c]}-{wk}-{st}-k{K}"
                    if abs(cmp[ck] - r["mean"]) > TOL:
                        fail(f"{ck} = {cmp[ck]} と stats_compare の平均 {r['mean']} が食い違う")
                    cond = f"世界 {C.WORLDS[wk]}，生徒 {s}，K={K}"
                    put_stats(f"st-{name}-{wk}-{st}-k{K}", r, cond, WHAT[c].format(b=f"{b}（{C.NAMES.get(b, b)}）"), UNIT)

    # ---- 検定（E63・E63c）
    ft = tests[tests.experiment.isin(["E63", "E63c"])]
    for tag, ex in (("ft", "E63"), ("ftc", "E63c")):
        for (c, K), (lk, sign) in FT_LEDGER.items():
            rr = ft[(ft.experiment == ex) & (ft.comparison == c) & (ft.K == K)]
            if len(rr) != 1:
                fail(f"tests.csv に {ex}・{c}・K={K} の行が {len(rr)} 本")
            r = rr.iloc[0]
            ck = "cmp" + lk.format(t=tag)
            if abs(sign * cmp[ck] - r["mean"]) > TOL:
                fail(f"{ck} = {cmp[ck]}（符号 {sign:+d}）と stats_compare の平均 {r['mean']} が食い違う")
            if c == "ours-vs-frozen" and cmp[f"cmp-{tag}-ours-below-frozen-ours-k{K}-pos-n"] != r.n_neg:
                fail(f"cmp-{tag}-ours-below-frozen-ours-k{K}-pos-n と stats_compare の負の対の数 {r.n_neg} が食い違う")
            src = "results/finetune_check/<ターゲット>" + ("_c" if tag == "ftc" else "") + ".csv，teacher_<ターゲット>.csv"
            cond = f"{ex}（{'中心化あり' if tag == 'ftc' else '中心化なし'}の LoRA 微調整．音声の生徒．{src}），4 ターゲット × 反復 0〜2"
            put_stats(f"st-{tag}-{c}-k{K}", r, cond, FT_WHAT[c][K],
                      "テストの話者の集合で数えた単位は {n}（12 対のうち同じ集合を共有する反復は無い．単位で数えた Holm 後の p は {p:.3g}）．"
                      "ターゲットは 4 つだけ．" if r.n_units_dedup == r.n else
                      "同じテストの話者の集合の反復を平均した {n} 単位での Holm 後の p は {p:.3g}．ターゲットは 4 つだけ．")

    # ---- 表（K = 0・5）
    tab = table.set_index(["world", "student", "K", "method"]).uar
    for wk in C.WORLDS:
        for s in students(wk):
            st = tok(wk, s)
            for K in TAB_KS:
                r = M[wk].loc[(s, K)]
                cols = {"e39": "e39", "ours": "ours", "oursx": "oursx:tuned"}
                cols.update({m: f"{m}:tuned" for m in sorted(comps(wk, s, K))})
                mine = {m for (w_, s_, k_, m) in tab.index if (w_, s_, k_) == (wk, s, K)}
                if mine != set(cols):
                    fail(f"{wk}・{s}・K={K} の手法が table.csv（{sorted(mine)}）と comparison.py（{sorted(cols)}）で違う")
                b = best_of(wk, s, K)
                for m, col in cols.items():
                    v = r[col]
                    if abs(tab[(wk, s, K, m)] - v) > TOL:
                        fail(f"tab {wk}・{s}・K={K}・{m}：comparison.py {v} と table.csv {tab[(wk, s, K, m)]} が食い違う")
                    d = C.WORLDS[wk]
                    if m == "e39":
                        f, what = f"results/baselines_{d}/<ターゲット>.csv", "none（E39．K 本だけで学び直した出発点）"
                    elif m == "ours":
                        f = f"results/multimodal_distill_{d}/<ターゲット>.csv（選択は同じ場所の <ターゲット>_pseudo.csv から作り直す）"
                        what = "E60 の選んだ形（ours）"
                    else:
                        f = f"results/baselines_{d}/<ターゲット>.csv（選択は <ターゲット>_pseudo.csv から作り直す）"
                        what = (f"{m}{'' if C.NAMES.get(m, m) == m else '（' + C.NAMES[m] + '）'}の調整版（選んだ候補 method|param|meff：{picks(wk, s, K, m)}）"
                                + ("．この生徒・K の最も強い相手" if m == b else ""))
                    put(f"tab-{wk}-{st}-k{K}-{m.replace('_', '-')}", v,
                        f"{f} | 世界 {d}，生徒 {s}，K={K}，{what} の UAR（％） | 反復 0〜9 の平均をターゲットごとに取り，{T[wk].target.nunique()} ターゲットで平均")
    return out
