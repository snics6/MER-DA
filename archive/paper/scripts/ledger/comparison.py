"""台帳のモジュール（comparison）：論文 6 節「標準の手法との比較」の数値（docs/paper-claim.md 4 節の最小の証拠 3）．

扱う実験（記録は docs/exploratory-analyses.md の同じ名前の節，状態は docs/results.md 3.6・3.9 節）
- E61：E59・E60 の比べる相手．第 1 エンコーダの 4 世界．AVT は全部の生徒（音声・映像・言葉・融合），AV・VT・AT は融合の生徒だけ．反復 0〜9
- E62：第 2 エンコーダ（音声 HuBERT，顔 CLIP，言葉 DeBERTa）の AVT の世界で E60・E61 をやり直したもの．反復 0〜9
- E63・E63c：WavLM を LoRA で微調整した音声の生徒（E63c は中心化を足した版）．4 ターゲット，反復 0〜2

値はすべて反復ごとの生の結果（results/baselines_<世界>/，results/multimodal_distill_<世界>/，results/finetune_check/）から計算する．
保存済みの判定表（results/explore_relation/e61_*.csv など）は使わない．
- 調整版：相手のつまみを，仮のターゲット（世界のソースのコーパス．反復 0〜4）の UAR の平均で選んだもの（「使わない」none も候補）．
  選択は <T>_pseudo.csv から exp/baselines_distill.py の choose() と同じ規則で作り直し，<T>_meta.csv の chosen と照らし合わせる
- 固定版：文献の既定値（exp/baselines_distill.py の FIXED と KD_FIXED_TEACHER）
- オラクル：テストで最良の候補（分析専用）．key に oracle を含むものだけに使う
- ours：E60 の選んだ形（仮のターゲットで選ぶ）．oursx：E61 の私たちの形の拡張（同じ選び方）
- 平均の単位：反復の平均をターゲットごとに取り，ターゲットで平均する（ターゲットの平均）．値は UAR のポイント差
- 「最も強い相手」：oursx 以外の相手の調整版のうち，ターゲットの平均が最大のもの（生徒・K ごと）
- 範囲の key（-lo・-hi）は，符号つきの値の最小と最大．「-below-」の key は「凍結版 − 微調整版」の正の幅

重い計算はないが，make check が毎回流すので，入力ファイルの大きさ・更新時刻とこのファイルの中身を鍵にして .cache/comparison.json に残す．
"""
from __future__ import annotations

import hashlib
import json
import pathlib

from .common import entry, rel, RESULTS

HERE = pathlib.Path(__file__).resolve()
CACHE = HERE.parent / ".cache" / "comparison.json"

WORLDS = {"avt": "fuse-audio-video-text", "av": "fuse-audio-video", "vt": "fuse-video-text", "at": "fuse-audio-text",
          "w2": "fuse2-audio-video-text"}
MODS = {"audio_wavlm": "a", "video": "v", "text_roberta": "t", "audio_hubert": "a", "video_clip": "v", "text_deberta": "t"}
KS = ("0", "1", "5", "10", "25")
REPS = tuple(range(10))
PSEUDO_REPS = tuple(range(5))
STR = {"K": str, "meff": str, "param": str}
# exp/baselines_distill.py の固定版（文献の既定値）
FIXED = {"self": ("0.9", "n"), "entmin": ("-", "n"), "gd": ("pFKT2", "n"), "hard": ("0.9", "n"), "cotrain": ("c2", "n"),
         "cotrain_prod": ("c2", "n"), "cotrain_fused": ("c2", "n"), "coreg": ("-", "n"), "coreg_sum": ("-", "n"),
         "saerens": ("-", "0"), "massbal": ("-", "0")}
KD_FIXED = {"mod": ("pFT2", "n"), "fused": ("poe0T2", "n")}
# exp/multimodal_distill.py の choose() の候補（none も候補）
E60_CANDS = {"mod": {"pF", "poe", "agree"}, "fused": {"poe0", "ref100", "ref500", "refdirect100", "refdirect500"}}
NAMES = {"self": "自己訓練", "entmin": "エントロピー最小化", "kd": "温度付きの蒸留", "gd": "一般化蒸留", "hard": "硬い疑似ラベル",
         "cotrain": "共訓練", "cotrain_prod": "共訓練の積", "cotrain_fused": "共訓練の融合", "coreg": "共正則化", "coreg_sum": "共正則化の和",
         "saerens": "Saerens の EM", "massbal": "出力の釣り合い直し", "oursx": "oursx"}
TGT = {"IEMOCAP-impro": "impro", "IEMOCAP-script": "script", "MELD": "meld", "MOSEI": "mosei", "AFEW": "afew", "CREMA-D": "cremad"}

FT_DIR = RESULTS / "finetune_check"
FT_TARGETS = ("IEMOCAP-impro", "IEMOCAP-script", "MELD", "MOSEI")
FT_REPS = (0, 1, 2)
FT_LAM = 0.01
FT_COLS = ["target", "rep", "K", "method", "lam", "uar", "n_test", "eval_sec", "gpu_gb", "n_train", "steps", "train_sec",
           "loss_first", "loss_last"]                                   # exp/finetune_check.py の RUN_COLS（MOSEI.csv は見出しが 9 列）

TOL = 0.055                                                             # 文書の値（小数 1 桁）との照合の許し幅．2 桁に丸めてから 1 桁にした値（1.449 → 1.45 → 1.5）も一致とみなす


# ---------------------------------------------------------------- 入力と控え

def _inputs() -> list[pathlib.Path]:
    files = []
    for w in WORLDS.values():
        files += sorted((RESULTS / f"baselines_{w}").glob("*.csv"))
        files += sorted((RESULTS / f"multimodal_distill_{w}").glob("*.csv"))
    for T in FT_TARGETS:
        files += [FT_DIR / f"{T}.csv", FT_DIR / f"{T}_c.csv", FT_DIR / f"teacher_{T}.csv"]
    return files


def _signature(files) -> str:
    h = hashlib.sha256()
    for f in (HERE, HERE.parent / "common.py"):
        h.update(f.read_bytes())
    for f in files:
        st = f.stat()
        h.update(f"{rel(f)}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def entries() -> dict[str, dict]:
    files = _inputs()
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
    out = {k: entry(v[0], v[1], v[2], v[3]) for k, v in items.items()}
    # 序論の貢献 4 の範囲（2026-10-27 に足した）：11 の生徒 × K ∈ {0, 5} の oursx − 最も強い相手の調整版の最小と最大
    cells = {f"{ws}-k{k}": out[f"cmp-oursx-vs-best-{ws}-k{k}"]["value"]
             for ws in ("avt-a", "avt-v", "avt-t", "avt-f", "av-f", "vt-f", "at-f", "w2-a", "w2-v", "w2-t", "w2-f")
             for k in (0, 5)}
    lo, hi = min(cells, key=cells.get), max(cells, key=cells.get)
    src = ("paper/scripts/ledger/comparison.py の cmp-oursx-vs-best-<世界>-<生徒>-k<K> と同じ計算 | 11 の生徒（AVT の 4 生徒，AV・VT・AT の融合，"
           "第 2 エンコーダの世界の 4 生徒）× K ∈ {{0, 5}} の 22 セル | 最小（{}）と最大（{}）")
    out["cmp-oursx-vs-best-range-lo"] = entry(cells[lo], src.format(lo, hi), "%.1f")
    out["cmp-oursx-vs-best-range-hi"] = entry(cells[hi], src.format(lo, hi), "%.1f")
    # 基本版 ours − 最も強い相手の調整版：22 セルのうち正のセルの数（2026-10-27．照合の A1）
    studs = ("avt-a", "avt-v", "avt-t", "avt-f", "av-f", "vt-f", "at-f", "w2-a", "w2-v", "w2-t", "w2-f")
    oc = {f"{ws}-k{k}": out[f"cmp-ours-vs-best-{ws}-k{k}"]["value"] for ws in studs for k in (0, 5)}
    s2 = ("paper/scripts/ledger/comparison.py の cmp-ours-vs-best-<世界>-<生徒>-k<K> | 11 の生徒 × K ∈ {0, 5} の 22 セル | ")
    out["cmp-ours-vs-best-pos-n"] = entry(sum(v > 0 for v in oc.values()), s2 + "基本版 ours が最も強い相手の調整版を上回るセルの数", "%d")
    out["cmp-ours-vs-best-cells-n"] = entry(len(oc), s2 + "セルの数", "%d")
    # オラクルとの差が 1 ポイント以上の生徒の数（K=0．反証条件 7 の前半）
    og = {ws: out[f"cmp-oursx-oracle-vs-tuned-{ws}-k0"]["value"] for ws in studs}
    s3 = ("paper/scripts/ledger/comparison.py の cmp-oursx-oracle-vs-tuned-<世界>-<生徒>-k0（オラクル．分析専用） | 11 の生徒，K=0 | ")
    out["cmp-oursx-oracle-gap-ge1-n"] = entry(sum(v >= 1 for v in og.values()), s3 + "テストで最良の候補が選んだ候補を 1 ポイント以上上回る生徒の数", "%d")
    out["cmp-oursx-oracle-gap-students-n"] = entry(len(og), s3 + "生徒の数", "%d")
    return out


# ---------------------------------------------------------------- 読み込み

def _read(path, **kw):
    import pandas as pd
    return pd.read_csv(path, comment="#", keep_default_na=False, **kw)


def _means(df, key, reps, what):
    """key ごとの UAR（％）の反復平均．重複と反復の欠けがあれば止める（complete でない結果は使わない）。"""
    if df.duplicated(key + ["rep"]).any():
        raise ValueError(f"comparison: {what} に同じ条件・同じ反復の行が重複している")
    g = df.groupby(key).uar
    cnt = g.size()
    if (cnt != len(reps)).any():
        bad = cnt[cnt != len(reps)]
        raise ValueError(f"comparison: {what} の反復が {len(reps)} 回そろっていない（例 {bad.index[0]}: {int(bad.iloc[0])} 回）")
    return (g.mean() * 100).to_dict()


def _first_max(cands, score):
    """pandas の idxmax と同じ（並べた順で最初の最大）。"""
    best = cands[0]
    for c in cands[1:]:
        if score[c] > score[best]:
            best = c
    return best


def _world(w: str):
    """世界 w の (target, student, K) ごとの値（ターゲットごとの反復平均）．

    列：e39（none），ours（E60 の選んだ形），<m>:tuned，<m>:pick，<m>:fixed，<m>:oracle（m は相手と oursx）．
    戻りは (行のリスト, 照合の数え上げ)．"""
    d, md = RESULTS / f"baselines_{w}", RESULTS / f"multimodal_distill_{w}"
    targets = sorted(f.stem for f in d.glob("*.csv") if not f.stem.endswith(("_pseudo", "_meta")))
    rows, chk = [], {"picks": 0, "meta_mismatch": 0, "e60_picks": 0, "e60_mismatch": 0}
    key = ["student", "method", "param", "meff", "K"]
    for T in targets:
        a = _read(d / f"{T}.csv", dtype=STR)
        mean = _means(a[a.rep.isin(REPS)], key, REPS, f"baselines_{w}/{T}.csv")
        p = _read(d / f"{T}_pseudo.csv", dtype=STR)
        p = p[p.rep.isin(PSEUDO_REPS)]
        ps = (p.groupby(key).uar.mean()).to_dict()
        by = {}
        for k in sorted(ps):
            by.setdefault((k[0], k[4]), []).append(k)
        meta = _read(d / f"{T}_meta.csv", dtype=STR)
        meta = meta[meta.chosen.astype(str) == "True"]
        meta_ch = {(r.student, r.method, r.K): (r.cand, r.param, r.meff) for r in meta.itertuples()}
        own = {}
        for k, v in mean.items():
            own.setdefault((k[0], k[1], k[4]), []).append(v)

        e = _read(md / f"{T}.csv", dtype={"K": str, "meff": str})
        emean = _means(e[e.rep.isin(REPS)], ["student", "belief", "meff", "K"], REPS, f"multimodal_distill_{w}/{T}.csv")
        ep = _read(md / f"{T}_pseudo.csv", dtype={"K": str, "meff": str})
        ep = ep[ep.rep.isin(PSEUDO_REPS)]
        eps = ep.groupby(["student", "belief", "meff", "K"]).uar.mean().to_dict()
        em = _read(md / f"{T}_meta.csv", dtype={"K": str, "meff": str})
        em = em[em.chosen.astype(str) == "True"]
        em_ch = {(r.student, r.K): (r.belief, r.meff) for r in em.itertuples()}

        students = sorted({k[0] for k in mean if k[1] == "none"})        # --fused-only の世界ではモダリティの生徒に E39 の行が無い
        for s in students:
            fused = s == w
            methods = sorted({k[1] for k in mean if k[0] == s} - {"none"})
            for K in KS:
                row = {"target": T, "student": s, "K": K, "e39": mean[(s, "none", "-", "0", K)]}
                for m in methods:
                    cands = [k for k in by[(s, K)] if k[1] in (m, "none")]
                    b = _first_max(cands, ps)
                    pick = (b[1], b[2], b[3])
                    chk["picks"] += 1
                    chk["meta_mismatch"] += int(meta_ch.get((s, m, K)) != pick)
                    row[f"{m}:tuned"] = mean[(s, *pick, K)]
                    row[f"{m}:pick"] = "|".join(pick)
                    fp, fm = KD_FIXED["fused" if fused else "mod"] if m == "kd" else FIXED.get(m, (None, None))
                    if (s, m, fp, fm, K) in mean:
                        row[f"{m}:fixed"] = mean[(s, m, fp, fm, K)]
                    row[f"{m}:oracle"] = max(own[(s, m, K)])
                cb = E60_CANDS["fused" if fused else "mod"]
                ec = sorted(k for k in eps if k[0] == s and k[3] == K and (k[1] == "none" or k[1] in cb))
                b = _first_max(ec, eps)
                chk["e60_picks"] += 1
                chk["e60_mismatch"] += int(em_ch.get((s, K)) != (b[1], b[2]))
                row["ours"] = emean[(s, b[1], b[2], K)]
                rows.append(row)
    return rows, chk


def _ft_runs(path):
    import pandas as pd
    a = pd.read_csv(path, comment="#", header=None, skiprows=1, names=FT_COLS, dtype=str, keep_default_na=False)
    a = a[a.target != "target"]
    a = a.assign(rep=a.rep.astype(int), lam=a.lam.astype(float), uar=a.uar.astype(float) * 100)
    return a[["target", "rep", "K", "method", "lam", "uar"]]


# ---------------------------------------------------------------- 計算

def _compute() -> dict[str, list]:
    import numpy as np
    import pandas as pd

    out: dict[str, list] = {}

    def put(key, value, source, fmt="%.1f", note=None):
        if key in out:
            raise ValueError(f"comparison: key '{key}' を 2 度作った")
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            raise ValueError(f"comparison: key '{key}' の値が有限でない")
        v = int(value) if fmt == "%d" else float(value)
        out[key] = [v, source, fmt, note]

    def doc(value, docv, where, why="", tol=TOL):
        """文書の値と丸めの範囲を超えて食い違えば note の文を返す。"""
        if abs(value - docv) <= tol:
            return None
        return f"{where} の値 {docv:g}．計算は {value:.2f}．{why}".rstrip("．") + "．"

    # ---- E61・E62：世界ごとの表
    T = {}       # wk → DataFrame（行 = target × student × K）
    chks = {}
    for wk, w in WORLDS.items():
        rows, chk = _world(w)
        T[wk] = pd.DataFrame(rows)
        chks[wk] = chk

    def tmean(wk):
        """(student, K) → 列ごとのターゲット平均。"""
        num = [c for c in T[wk].columns if c not in ("target", "student", "K") and not c.endswith(":pick")]
        return T[wk].groupby(["student", "K"])[num].mean()

    M = {wk: tmean(wk) for wk in WORLDS}

    def tok(wk, s):
        return "f" if s == WORLDS[wk] else MODS[s]

    def students(wk):
        return sorted({s for s, _ in M[wk].index}, key=lambda s: "atvf".index(tok(wk, s)))

    def ntarg(wk):
        return T[wk].target.nunique()

    def files(wk, e60=False):
        w = WORLDS[wk]
        f = f"results/baselines_{w}/<ターゲット>.csv（調整版の選択は <ターゲット>_pseudo.csv から作り直す）"
        if e60:
            f += f"，results/multimodal_distill_{w}/<ターゲット>.csv（ours の選択は同じ場所の <ターゲット>_pseudo.csv から）"
        return f

    def unit(wk):
        ts = "・".join(sorted(T[wk].target.unique()))
        return f"反復 0〜9 の平均をターゲットごとに取り，{ntarg(wk)} ターゲット（{ts}）で平均"

    def comps(wk, s, K):
        r = M[wk].loc[(s, K)]
        return {c.split(":")[0]: r[c] for c in M[wk].columns
                if c.endswith(":tuned") and not c.startswith("oursx") and not pd.isna(r[c])}

    def best_of(wk, s, K):
        c = comps(wk, s, K)
        b = max(sorted(c), key=lambda m: c[m])
        return b, c[b]

    def comp_list(wk, s):
        return "・".join(sorted(comps(wk, s, "0")))

    # 照合：作り直した選択が <T>_meta.csv の chosen と一致するか
    def sel_note(wk):
        c = chks[wk]
        if c["meta_mismatch"] or c["e60_mismatch"]:
            return (f"作り直した選択が保存の chosen と食い違う（E61 {c['meta_mismatch']}/{c['picks']}，E60 {c['e60_mismatch']}/{c['e60_picks']}）．"
                    "値は作り直した選択で計算した．")
        return None

    # ---- 生徒・K ごとの値
    for wk in WORLDS:
        for s in students(wk):
            st = tok(wk, s)
            for K in KS:
                r = M[wk].loc[(s, K)]
                b, bv = best_of(wk, s, K)
                ties = [m for m, x in sorted(comps(wk, s, K).items()) if m != b and abs(x - bv) < 1e-9]
                bname = f"{b}（{NAMES[b]}）" + (f"．{'・'.join(ties)} と同点（同点は名前の順で最初のもの）" if ties else "")
                cond = f"世界 {WORLDS[wk]}，生徒 {s}，K={K}"
                note = sel_note(wk)
                put(f"cmp-oursx-vs-e39-{wk}-{st}-k{K}", r["oursx:tuned"] - r.e39,
                    f"{files(wk)} | {cond}，oursx の調整版 − none（E39） | {unit(wk)}", note=note)
                put(f"cmp-best-vs-e39-{wk}-{st}-k{K}", bv - r.e39,
                    f"{files(wk)} | {cond}，最も強い相手 {bname} の調整版 − none（E39）．相手は {comp_list(wk, s)} の調整版でターゲット平均が最大のもの | {unit(wk)}",
                    note=note)
                put(f"cmp-oursx-vs-best-{wk}-{st}-k{K}", r["oursx:tuned"] - bv,
                    f"{files(wk)} | {cond}，oursx の調整版 − 最も強い相手 {bname} の調整版 | {unit(wk)}", note=note)
                put(f"cmp-ours-vs-e39-{wk}-{st}-k{K}", r.ours - r.e39,
                    f"{files(wk, True)} | {cond}，E60 の選んだ形（ours）− none（E39） | {unit(wk)}", note=note)
                put(f"cmp-ours-vs-best-{wk}-{st}-k{K}", r.ours - bv,
                    f"{files(wk, True)} | {cond}，E60 の選んだ形（ours）− 最も強い相手 {bname} の調整版 | {unit(wk)}", note=note)

    def rng(key, vals, source, docs=None, where="", why="", tol=TOL):
        """範囲の key 2 つ．docs が整数の範囲（「2〜3」など）なら tol=0.5 で照らし合わせる。"""
        vals = [float(v) for v in vals]
        if any(not np.isfinite(v) for v in vals):
            raise ValueError(f"comparison: {key} の範囲に有限でない値がある")
        lo, hi = min(vals), max(vals)
        nlo = doc(lo, docs[0], where, why, tol) if docs else None
        nhi = doc(hi, docs[1], where, why, tol) if docs else None
        put(f"{key}-lo", lo, source + "．その最小", note=nlo)
        put(f"{key}-hi", hi, source + "．その最大", note=nhi)

    def col(wk, s, c, K):
        return M[wk].loc[(s, K), c]

    A = {wk: {tok(wk, s): s for s in students(wk)} for wk in WORLDS}
    W39 = "docs/results.md 3.9 節（E61 の AVT）"

    # ---- E61 の AVT：oursx − E39 と oursx − 最も強い相手の範囲
    for st, Ks, tag, d39 in (("a", KS, "", (3.1, 3.8)), ("t", ("0", "1", "5"), "-kle5", (3.7, 4.6)),
                             ("f", ("1", "5", "10", "25"), "-kge1", (2.1, 3.5)), ("v", KS, "", (0.7, 1.6))):
        s = A["avt"][st]
        kk = "・".join(Ks)
        rng(f"cmp-oursx-vs-e39-avt-{st}{tag}", [col("avt", s, "oursx:tuned", K) - col("avt", s, "e39", K) for K in Ks],
            f"{files('avt')} | 生徒 {s}，K ∈ {{{kk}}}，oursx の調整版 − none（E39）の K ごとの値 | {unit('avt')}", d39, W39)
        rng(f"cmp-oursx-vs-best-avt-{st}{tag}", [col("avt", s, "oursx:tuned", K) - best_of("avt", s, K)[1] for K in Ks],
            f"{files('avt')} | 生徒 {s}，K ∈ {{{kk}}}，oursx の調整版 − 最も強い相手の調整版（K ごとに選ぶ：{'・'.join(best_of('avt', s, K)[0] for K in Ks)}） | {unit('avt')}")
    s = A["avt"]["t"]
    rng("cmp-gd-vs-oursx-avt-t-kge10", [col("avt", s, "gd:tuned", K) - col("avt", s, "oursx:tuned", K) for K in ("10", "25")],
        f"{files('avt')} | 生徒 {s}，K ∈ {{10・25}}，一般化蒸留 gd の調整版 − oursx の調整版 | {unit('avt')}", (0.1, 0.3), W39)
    for st, dv in (("a", (1.1, 2.4)), ("t", (2.0, 4.0))):
        s = A["avt"][st]
        rng(f"cmp-gd-vs-e39-avt-{st}", [col("avt", s, "gd:tuned", K) - col("avt", s, "e39", K) for K in KS],
            f"{files('avt')} | 生徒 {s}，K ∈ {{0・1・5・10・25}}，一般化蒸留 gd の調整版 − none（E39） | {unit('avt')}", dv, W39)
    s = A["avt"]["f"]
    rng("cmp-self-vs-e39-avt-f", [col("avt", s, "self:tuned", K) - col("avt", s, "e39", K) for K in KS],
        f"{files('avt')} | 生徒 {s}，K ∈ {{0・1・5・10・25}}，自己訓練 self の調整版 − none（E39） | {unit('avt')}", (0.2, 0.8), W39)
    for K, dv in (("0", 1.4), ("1", 1.0)):
        v = col("avt", s, "massbal:tuned", K) - col("avt", s, "e39", K)
        put(f"cmp-massbal-vs-e39-avt-f-k{K}", v,
            f"{files('avt')} | 生徒 {s}，K={K}，出力の釣り合い直し massbal の調整版（massbal と none から選ぶ）− none（E39） | {unit('avt')}",
            note=doc(v, dv, W39))

    # 最も強い相手の名前が文書の言い方と違う所
    s = A["avt"]["t"]
    b, _ = best_of("avt", s, "0")
    if b != "gd":
        key = "cmp-best-vs-e39-avt-t-k0"
        out[key][3] = ((out[key][3] or "") + f"文書（{W39}・E61 の節のわかったこと 4）は「最も強い相手はモダリティの生徒では一般化蒸留（言葉 +2.0〜+4.0）」とするが，"
                       f"言葉の K=0 では {b}（{NAMES[b]}）の調整版が gd（{col('avt', s, 'gd:tuned', '0') - col('avt', s, 'e39', '0'):+.2f}）より強い．")
    s = A["avt"]["v"]
    bv_names = sorted({best_of("avt", s, K)[0] for K in KS})
    if "gd" not in bv_names:
        key = "cmp-best-vs-e39-avt-v-k0"
        out[key][3] = ((out[key][3] or "") + f"映像の生徒では最も強い相手は K ごとに {'・'.join(best_of('avt', s, K)[0] for K in KS)} で，"
                       "一般化蒸留ではない（文書の「モダリティの生徒では一般化蒸留」は音声と言葉の K ≥ 1 の話）．")

    # 選択の取りこぼし（オラクル）
    for wk in ("avt", "av", "vt", "at", "w2"):
        for st in (("a", "v", "t", "f") if wk in ("avt", "w2") else ("f",)):
            s = A[wk][st]
            r = M[wk].loc[(s, "0")]
            dv = {("avt", "f"): 1.1, ("av", "f"): 0.9, ("vt", "f"): 0.4, ("at", "f"): 0.2,
                  ("w2", "a"): 0.8, ("w2", "t"): 2.2, ("w2", "f"): 1.0, ("w2", "v"): 1.5}.get((wk, st))
            where = "docs/exploratory-analyses.md E61 の節（わかったこと 6）" if wk != "w2" else "docs/exploratory-analyses.md E62 の節"
            v = r["oursx:oracle"] - r["oursx:tuned"]
            put(f"cmp-oursx-oracle-vs-tuned-{wk}-{st}-k0", v,
                f"{files(wk)} | 世界 {WORLDS[wk]}，生徒 {s}，K=0，oursx のオラクル（ターゲットごとにテストで最良の候補．分析専用）− oursx の調整版 | {unit(wk)}",
                note=doc(v, dv, where) if dv is not None else None)
    for st, dv in (("t", 6.2), ("f", 4.0)):
        s = A["avt"][st]
        r = M["avt"].loc[(s, "0")]
        v = r["oursx:oracle"] - r.e39
        put(f"cmp-oursx-oracle-vs-e39-avt-{st}-k0", v,
            f"{files('avt')} | 生徒 {s}，K=0，oursx のオラクル（ターゲットごとにテストで最良の候補．分析専用）− none（E39） | {unit('avt')}",
            note=doc(v, dv, W39 if st == "t" else "docs/exploratory-analyses.md E61 の節"))

    # ---- E61 の融合の生徒（4 世界）
    f3 = ("avt", "vt", "at")
    WF = "docs/results.md 3.9 節・docs/researchvision.md 更新記録 2026-10-24（E61 の AV・VT・AT）"
    why26 = ("どの組（ours・oursx × AVT・VT・AT）の差も 2.0 を超えない．上限 2.6 は再現できない"
             "（VT の ours − E39 が +2.62 なので，差ではなく E39 との差を写した可能性がある）")
    vals = [col(wk, A[wk]["f"], c, "0") - best_of(wk, A[wk]["f"], "0")[1] for wk in f3 for c in ("ours", "oursx:tuned")]
    bn = "・".join(f"{wk.upper()} {best_of(wk, A[wk]['f'], '0')[0]}" for wk in f3)
    srcf = (f"results/baselines_<世界>/<ターゲット>.csv，results/multimodal_distill_<世界>/<ターゲット>.csv（世界は AVT・VT・AT．選択は各 <ターゲット>_pseudo.csv から） | "
            f"融合の生徒，K=0")
    rng("cmp-ours-oursx-vs-best-avt-vt-at-f-k0", vals,
        f"{srcf}，ours と oursx の調整版のそれぞれ − 最も強い相手の調整版（{bn}）の 6 つ | 各世界で 4 ターゲットの平均（反復 0〜9 の平均をターゲットごとに取る）",
        (1.5, 2.6), WF, why26)
    vals = [col(wk, A[wk]["f"], "oursx:tuned", "0") - best_of(wk, A[wk]["f"], "0")[1] for wk in f3]
    rng("cmp-oursx-vs-best-avt-vt-at-f-k0", vals,
        f"{srcf}，oursx の調整版 − 最も強い相手の調整版（{bn}）の 3 つ | 各世界で 4 ターゲットの平均（反復 0〜9 の平均をターゲットごとに取る）",
        (1.5, 2.6), "docs/paper-claim.md 4 節（証拠 3）", why26)
    vals = [best_of(wk, A[wk]["f"], "0")[1] - col(wk, A[wk]["f"], "e39", "0") for wk in f3]
    rng("cmp-best-vs-e39-avt-vt-at-f-k0", vals,
        f"{srcf}，最も強い相手の調整版（{bn}）− none（E39）の 3 つ | 各世界で 4 ターゲットの平均（反復 0〜9 の平均をターゲットごとに取る）",
        (1.0, 1.4), "docs/exploratory-analyses.md E61 の節（AV・VT・AT のわかったこと 1）")
    s = A["av"]["f"]
    tuned_cols = [c for c in M["av"].columns if c.endswith(":tuned")] + ["ours"]
    v = max(x for x in (col("av", s, c, K) - col("av", s, "e39", K) for K in KS for c in tuned_cols) if np.isfinite(x))
    put("cmp-alltuned-vs-e39-av-f-hi", v,
        f"{files('av', True)} | 生徒 {s}，K ∈ {{0・1・5・10・25}}，全部の方法の調整版と ours（{len(tuned_cols)} 通り）− none（E39）の最大 | {unit('av')}",
        note=doc(v, 1.3, WF))
    r = M["av"].loc[(s, "0")]
    v = r["oursx:oracle"] - r.e39
    put("cmp-oursx-oracle-vs-e39-av-f-k0", v,
        f"{files('av')} | 生徒 {s}，K=0，oursx のオラクル（ターゲットごとにテストで最良の候補．分析専用）− none（E39） | {unit('av')}",
        note=doc(v, 2.0, "docs/exploratory-analyses.md E61 の節（AV・VT・AT のわかったこと 1）"))
    for wk, dv in (("avt", (2.1, 2.4)), ("vt", (1.7, 2.2)), ("at", (1.3, 1.6)), ("av", (0.5, 1.0))):
        s = A[wk]["f"]
        Ks = ("5", "10", "25")
        rng(f"cmp-oursx-vs-e39-{wk}-f-kge5", [col(wk, s, "oursx:tuned", K) - col(wk, s, "e39", K) for K in Ks],
            f"{files(wk)} | 生徒 {s}，K ∈ {{5・10・25}}，oursx の調整版 − none（E39） | {unit(wk)}", dv, WF)
        rng(f"cmp-oursx-vs-best-{wk}-f-kge5", [col(wk, s, "oursx:tuned", K) - best_of(wk, s, K)[1] for K in Ks],
            f"{files(wk)} | 生徒 {s}，K ∈ {{5・10・25}}，oursx の調整版 − 最も強い相手の調整版（K ごとに {'・'.join(best_of(wk, s, K)[0] for K in Ks)}） | {unit(wk)}")

    # ---- E61 のエントロピー最小化・共正則化・Saerens の固定版
    WE = "docs/exploratory-analyses.md E61 の節（AV・VT・AT のわかったこと 5 とターゲットごとの表）"
    ent_doc = {"avt": -2.9, "vt": -10.3, "at": -7.2, "av": -14.3}
    tgt_doc = {("avt", "IEMOCAP-impro"): 4.62, ("avt", "IEMOCAP-script"): -0.79, ("avt", "MELD"): -9.38, ("avt", "MOSEI"): -6.05,
               ("vt", "IEMOCAP-impro"): -19.98, ("vt", "IEMOCAP-script"): -6.40, ("vt", "MELD"): -12.82, ("vt", "MOSEI"): -1.86,
               ("at", "IEMOCAP-impro"): 2.45, ("at", "IEMOCAP-script"): -1.79, ("at", "MELD"): -20.05, ("at", "MOSEI"): -9.38,
               ("av", "IEMOCAP-impro"): -28.35, ("av", "IEMOCAP-script"): -17.39, ("av", "MELD"): -17.31, ("av", "MOSEI"): -7.13,
               ("av", "AFEW"): -7.41, ("av", "CREMA-D"): -8.02}
    vals = []
    for wk in ("avt", "vt", "at", "av"):
        s = A[wk]["f"]
        v = col(wk, s, "entmin:fixed", "0") - col(wk, s, "e39", "0")
        vals.append(v)
        put(f"cmp-entmin-fixed-vs-e39-{wk}-f-k0", v,
            f"{files(wk)} | 生徒 {s}，K=0，エントロピー最小化の固定版（候補 1 本あたり重み 1．entmin|-|n）− none（E39） | {unit(wk)}",
            note=doc(v, ent_doc[wk], WE))
        g = T[wk][(T[wk].student == s) & (T[wk].K == "0")]
        for _, rr in g.iterrows():
            v = rr["entmin:fixed"] - rr.e39
            put(f"cmp-entmin-fixed-vs-e39-{wk}-f-k0-{TGT[rr.target]}", v,
                f"results/baselines_{WORLDS[wk]}/{rr.target}.csv | 生徒 {s}，K=0，エントロピー最小化の固定版 − none（E39） | 反復 0〜9 の平均（このターゲットだけ）",
                note=doc(v, tgt_doc[(wk, rr.target)], WE))
    rng("cmp-entmin-fixed-vs-e39-f-k0", vals,
        "results/baselines_<世界>/<ターゲット>.csv（世界は AVT・VT・AT・AV） | 融合の生徒，K=0，エントロピー最小化の固定版 − none（E39）の 4 世界 | "
        "各世界でターゲットの平均（AV 6，ほか 4．反復 0〜9 の平均をターゲットごとに取る）", (-14.3, -2.9), WE)
    s = A["avt"]["f"]
    g = T["avt"][(T["avt"].student == s) & (T["avt"].K == "0")]
    for _, rr in g.iterrows():
        for c, nm in (("ours", "ours"), ("oursx:tuned", "oursx")):
            put(f"cmp-{nm}-vs-e39-avt-f-k0-{TGT[rr.target]}", rr[c] - rr.e39,
                f"results/baselines_{WORLDS['avt']}/{rr.target}.csv" + ("，results/multimodal_distill_fuse-audio-video-text/" + rr.target + ".csv" if nm == "ours" else "")
                + f" | 生徒 {s}，K=0，{'E60 の選んだ形（ours）' if nm == 'ours' else 'oursx の調整版'} − none（E39） | 反復 0〜9 の平均（このターゲットだけ）")
    vals = []
    for wk in ("avt", "vt", "at", "av"):
        s = A[wk]["f"]
        vals += [col(wk, s, f"{m}:fixed", "0") - col(wk, s, "e39", "0") for m in ("coreg_sum", "saerens")]
    rng("cmp-coreg-saerens-fixed-vs-e39-f-k0", vals,
        "results/baselines_<世界>/<ターゲット>.csv（世界は AVT・VT・AT・AV） | 融合の生徒，K=0，共正則化の和（coreg_sum|-|n）と Saerens の EM の固定版 − none（E39）の 8 つ | "
        "各世界でターゲットの平均（AV 6，ほか 4．反復 0〜9 の平均をターゲットごとに取る）", (-33, -10), WE,
        "文書の「−10〜−33」は整数への切り捨てで，小数 1 桁では範囲の端が少しずれる", tol=0.5)
    vals, parts = [], []
    WC = "docs/results.md 3.9 節（E61 の AVT．「モダリティの生徒で −12〜−23」）"
    for st in ("a", "v", "t"):
        s = A["avt"][st]
        for m in ("entmin", "coreg", "saerens"):
            v = col("avt", s, f"{m}:fixed", "0") - col("avt", s, "e39", "0")
            vals.append(v)
            parts.append((st, m, v))
            put(f"cmp-{m}-fixed-vs-e39-avt-{st}-k0", v,
                f"{files('avt')} | 生徒 {s}，K=0，{NAMES[m]}の固定版（{m}|{FIXED[m][0]}|{FIXED[m][1]}）− none（E39） | {unit('avt')}")
    over = [f"{st} {m} {v:+.1f}" for st, m, v in parts if v > -12 + 0.5]
    rng("cmp-fixed-collapse-avt-mod-k0", vals,
        f"{files('avt')} | モダリティの生徒（音声・映像・言葉），K=0，エントロピー最小化・共正則化・Saerens の EM の固定版 − none（E39）の 9 つ | {unit('avt')}",
        (-23, -12), WC, ("文書の範囲は音声の生徒の 3 つ（−12.4〜−22.7）に合う．−12 より損の小さい組がある：" + "，".join(over)) if over else "", tol=0.5)

    # ---- E62（第 2 エンコーダの AVT）
    W62 = "docs/exploratory-analyses.md E62 の節・docs/results.md 3.6 節"
    ours_doc = {("a", "0"): 3.8, ("a", "5"): 2.8, ("v", "0"): 0.9, ("v", "5"): 1.6, ("t", "0"): 4.1, ("t", "5"): 3.7, ("f", "0"): 3.8, ("f", "5"): 2.4}
    for (st, K), dv in ours_doc.items():
        key = f"cmp-ours-vs-e39-w2-{st}-k{K}"
        out[key][3] = (out[key][3] or "") + (doc(out[key][0], dv, W62) or "") or None
    e62 = {("a", "0"): (4.42, 2.70, "cotrain", -13.2), ("a", "5"): (4.11, 2.65, "gd", -16.2), ("v", "0"): (1.34, 0.35, "cotrain", -11.9),
           ("v", "5"): (2.08, 0.71, "kd", -10.7), ("t", "0"): (3.35, 0.34, "massbal", -14.0), ("t", "5"): (4.03, 2.45, "gd", -9.2),
           ("f", "0"): (3.52, 1.94, "massbal", -2.6), ("f", "5"): (2.46, 0.61, "cotrain_fused", -2.8)}
    for (st, K), (dx, db, dn, dent) in e62.items():
        s = A["w2"][st]
        for key, dv in ((f"cmp-oursx-vs-e39-w2-{st}-k{K}", dx), (f"cmp-best-vs-e39-w2-{st}-k{K}", db)):
            out[key][3] = (out[key][3] or "") + (doc(out[key][0], dv, W62) or "") or None
        b = best_of("w2", s, K)[0]
        if b != dn:
            key = f"cmp-best-vs-e39-w2-{st}-k{K}"
            out[key][3] = (out[key][3] or "") + f"{W62} の最も強い相手は {dn}，計算では {b}．"
        v = col("w2", s, "entmin:fixed", K) - col("w2", s, "e39", K)
        put(f"cmp-entmin-fixed-vs-e39-w2-{st}-k{K}", v,
            f"{files('w2')} | 生徒 {s}，K={K}，エントロピー最小化の固定版（entmin|-|n）− none（E39） | {unit('w2')}", note=doc(v, dent, W62))
    S2 = [A["w2"][st] for st in ("a", "v", "t", "f")]
    rng("cmp-oursx-vs-best-w2-k0k5", [col("w2", s, "oursx:tuned", K) - best_of("w2", s, K)[1] for s in S2 for K in ("0", "5")],
        f"{files('w2')} | 4 生徒（{'・'.join(S2)}）× K ∈ {{0・5}}，oursx の調整版 − 最も強い相手の調整版の 8 つ | {unit('w2')}", (1.0, 3.0), W62)
    rng("cmp-oursx-vs-e39-w2-k25", [col("w2", s, "oursx:tuned", "25") - col("w2", s, "e39", "25") for s in S2],
        f"{files('w2')} | 4 生徒（{'・'.join(S2)}），K=25，oursx の調整版 − none（E39） | {unit('w2')}", (2.1, 4.4), W62,
        "文書の「K=25 でも +2.1〜+4.4」は oursx − E39 で，最も強い相手との差ではない")
    rng("cmp-oursx-vs-best-w2-k25", [col("w2", s, "oursx:tuned", "25") - best_of("w2", s, "25")[1] for s in S2],
        f"{files('w2')} | 4 生徒（{'・'.join(S2)}），K=25，oursx の調整版 − 最も強い相手の調整版（{'・'.join(best_of('w2', s, '25')[0] for s in S2)}） | {unit('w2')}")
    rng("cmp-entmin-fixed-vs-e39-w2-k0k5", [col("w2", s, "entmin:fixed", K) - col("w2", s, "e39", K) for s in S2 for K in ("0", "5")],
        f"{files('w2')} | 4 生徒 × K ∈ {{0・5}}，エントロピー最小化の固定版 − none（E39）の 8 つ | {unit('w2')}", (-16.2, -2.6), W62)

    # ---- 件数
    for wk in WORLDS:
        put(f"cmp-targets-{wk}-n", ntarg(wk), f"results/baselines_{WORLDS[wk]}/ | ターゲットの CSV（_pseudo・_meta を除く）の数 | 数える", "%d")
    put("cmp-reps-n", len(REPS), "results/baselines_<世界>/<ターゲット>.csv，results/multimodal_distill_<世界>/<ターゲット>.csv | 列 rep（全部の条件で 0〜9 がそろうことを確かめた） | 数える", "%d")
    put("cmp-pseudo-reps-n", len(PSEUDO_REPS), "results/baselines_<世界>/<ターゲット>_pseudo.csv | 列 rep（仮のターゲットの反復 0〜4） | 数える", "%d")

    # ---- E63・E63c（微調整の確認）
    ft = pd.concat([_ft_runs(FT_DIR / f"{t}.csv") for t in FT_TARGETS])
    fc = pd.concat([_ft_runs(FT_DIR / f"{t}_c.csv") for t in FT_TARGETS])
    fr = pd.concat([_read(FT_DIR / f"teacher_{t}.csv", dtype={"K": str}) for t in FT_TARGETS])
    fr = fr.assign(uar=fr.uar * 100, lam=0.0)
    for nm, df in (("finetune_check/<T>.csv", ft), ("finetune_check/<T>_c.csv", fc), ("finetune_check/teacher_<T>.csv", fr)):
        if df.duplicated(["target", "rep", "K", "method", "lam"]).any():
            raise ValueError(f"comparison: {nm} に重複した行がある")
        if sorted(df.target.unique()) != sorted(FT_TARGETS) or any(sorted(g.rep.unique()) != list(FT_REPS) for _, g in df.groupby("target")):
            raise ValueError(f"comparison: {nm} のターゲットか反復 0〜2 がそろっていない")

    def ser(df, m, K, lam=None):
        x = df[(df.method == m) & (df.K == K)]
        if lam is not None:
            x = x[np.isclose(x.lam, lam)]
        x = x.set_index(["target", "rep"]).uar.sort_index()
        if len(x) != len(FT_TARGETS) * len(FT_REPS):
            raise ValueError(f"comparison: E63 の {m}（K={K}，λ={lam}）が {len(x)} 行（12 行のはず）")
        return x

    def tm(x):
        return x.groupby(level="target").mean().mean()

    FTU = "反復 0〜2 の平均をターゲットごとに取り，4 ターゲット（IEMOCAP-impro・IEMOCAP-script・MELD・MOSEI）で平均"
    FTP = "ターゲット × 反復で対にした差を，" + FTU
    SF = "results/finetune_check/<ターゲット>.csv"
    SC = "results/finetune_check/<ターゲット>_c.csv"
    SR = "results/finetune_check/teacher_<ターゲット>.csv"
    W63 = "docs/exploratory-analyses.md E63 の節・docs/results.md 3.6 節"
    W63c = "docs/exploratory-analyses.md E63 の節（追加の確かめ）・docs/researchvision.md 更新記録 2026-10-27"
    E0, E5 = ser(fr, "frozen_e39", "0"), ser(fr, "frozen_e39", "5")
    O0, O5 = ser(fr, "frozen_ours", "0"), ser(fr, "frozen_ours", "5")
    lv = {}
    for tag, df, src, dl in (("ft", ft, SF, {"start": 39.7, "konly": 41.3, "anchor": 41.7, "ours-k0": 45.5, "ours-k5": 46.1}),
                             ("ftc", fc, SC, {"start": 42.5, "konly": 44.1, "anchor": 44.0, "ours-k0": 46.0, "ours-k5": 47.4})):
        lv[tag] = {"start": ser(df, "start", "0"), "konly": ser(df, "k_only", "5"), "anchor": ser(df, "anchor", "5", FT_LAM),
                   "ours-k0": ser(df, "ours", "0", FT_LAM), "ours-k5": ser(df, "ours", "5", FT_LAM)}
        what = {"start": "start（出発点のまま．K=0）", "konly": "k_only（K 本だけ．K=5）", "anchor": "anchor（出発点への罰則 λ=0.01．K=5）",
                "ours-k0": "ours（私たちの形 λ=0.01．K=0）", "ours-k5": "ours（私たちの形 λ=0.01．K=5）"}
        kname = {"start": "start-k0", "konly": "konly-k5", "anchor": "anchor-k5", "ours-k0": "ours-k0", "ours-k5": "ours-k5"}
        for k, x in lv[tag].items():
            put(f"cmp-{tag}-uar-{kname[k]}", tm(x),
                f"{src} | method {what[k]} の UAR | {FTU}", note=doc(tm(x), dl[k], W63 if tag == "ft" else W63c))
    for k, x, dv in (("e39-k0", E0, 47.5), ("e39-k5", E5, 48.6), ("ours-k0", O0, 50.8), ("ours-k5", O5, 52.3)):
        put(f"cmp-frozen-uar-{k}", tm(x), f"{SR} | method frozen_{k.split('-')[0]}（凍結した特徴．K={k[-1]}）の UAR | {FTU}", note=doc(tm(x), dv, W63))

    for tag, src, W in (("ft", SF, W63), ("ftc", SC, W63c)):
        L = lv[tag]
        d = E5 - L["konly"]
        put(f"cmp-{tag}-konly-below-frozen-e39-k5", tm(d),
            f"{src}，{SR} | K=5，frozen_e39 − k_only（凍結 E39 が K 本だけの微調整を上回る幅） | {FTP}",
            note=doc(tm(d), 7.3 if tag == "ft" else 4.5, W))
        rng(f"cmp-{tag}-konly-below-frozen-e39-k5-tgt", list(d.groupby(level="target").mean()),
            f"{src}，{SR} | K=5，frozen_e39 − k_only の 4 ターゲット | 反復 0〜2 の平均をターゲットごとに取る",
            (6.3, 9.1) if tag == "ft" else None, W)
        d = E0 - L["start"]
        put(f"cmp-{tag}-start-below-frozen-e39-k0", tm(d),
            f"{src}，{SR} | K=0，frozen_e39 − start（凍結 E39 が出発点のままの微調整を上回る幅） | {FTP}",
            note=doc(tm(d), 7.8, W) if tag == "ft" else None)
        d = L["anchor"] - L["konly"]
        put(f"cmp-{tag}-anchor-vs-konly-k5", tm(d), f"{src} | K=5，anchor（λ=0.01）− k_only（出発点への罰則の上乗せ） | {FTP}",
            note=doc(tm(d), 0.4 if tag == "ft" else -0.1, W))
        d0, d5 = L["ours-k0"] - L["start"], L["ours-k5"] - L["anchor"]
        put(f"cmp-{tag}-weak-vs-start-k0", tm(d0), f"{src} | K=0，ours（λ=0.01）− start（弱いラベルの経路の上乗せ） | {FTP}",
            note=doc(tm(d0), 5.8 if tag == "ft" else 3.4, W))
        put(f"cmp-{tag}-weak-vs-anchor-k5", tm(d5), f"{src} | K=5，ours（λ=0.01）− anchor（λ=0.01）（弱いラベルの経路の上乗せ） | {FTP}",
            note=doc(tm(d5), 4.4 if tag == "ft" else 3.4, W))
        g = list(d0.groupby(level="target").mean()) + list(d5.groupby(level="target").mean())
        rng(f"cmp-{tag}-weak-gain", g,
            f"{src} | 弱いラベルの経路の上乗せ（K=0 は ours − start，K=5 は ours − anchor．λ=0.01）の 4 ターゲット × 2 つの K の 8 つ | 反復 0〜2 の平均をターゲットごとに取る",
            (1.7, 8.9) if tag == "ft" else None, W)
        b0, b5 = O0 - L["ours-k0"], O5 - L["ours-k5"]
        put(f"cmp-{tag}-ours-below-frozen-ours-k0", tm(b0),
            f"{src}，{SR} | K=0，frozen_ours − ours（λ=0.01）（凍結版の私たちの形が微調整版を上回る幅） | {FTP}",
            note=doc(tm(b0), 5.4 if tag == "ft" else 4.9, W))
        put(f"cmp-{tag}-ours-below-frozen-ours-k5", tm(b5),
            f"{src}，{SR} | K=5，frozen_ours − ours（λ=0.01） | {FTP}", note=doc(tm(b5), 6.2 if tag == "ft" else 4.9, W))
        pairs = list(b0) + list(b5)
        rng(f"cmp-{tag}-ours-below-frozen-ours-pair", pairs,
            f"{src}，{SR} | frozen_ours − ours（λ=0.01）のターゲット × 反復の対（K=0 と K=5 で 24 組） | 対ごとの値",
            (2.0, 10.3) if tag == "ft" else None, W)
        for K, b in (("0", b0), ("5", b5)):
            put(f"cmp-{tag}-ours-below-frozen-ours-k{K}-pos-n", int((b > 0).sum()),
                f"{src}，{SR} | K={K}，frozen_ours − ours（λ=0.01）が正（微調整版が下回る）のターゲット × 反復の対の数（全 {len(b)} 組） | 数える", "%d",
                note=(f"{W} は「12 組すべて負」と書くが，12 組は K ごとの数で，K=0 と K=5 を合わせると 24 組（範囲 −2.0〜−10.3 は 24 組の範囲）．"
                      if tag == "ft" and K == "0" else None))
    put("cmp-ft-pairs-n", len(FT_TARGETS) * len(FT_REPS), f"{SF} | K ごとのターゲット × 反復の対の数 | 数える", "%d")
    put("cmp-ft-reps-n", len(FT_REPS), f"{SF}，{SC}，{SR} | 列 rep（0〜2 がそろうことを確かめた） | 数える", "%d")
    for lam, dv, nm in ((0.001, 0.3, "lam0001"), (0.1, -0.3, "lam01")):
        d = ser(ft, "anchor", "5", lam) - lv["ft"]["konly"]
        put(f"cmp-ft-anchor-vs-konly-k5-{nm}", tm(d), f"{SF} | K=5，anchor（λ={lam}）− k_only | {FTP}", note=doc(tm(d), dv, W63))
    a5 = ft[(ft.method == "anchor") & (ft.K == "5")].groupby(["target", "lam"]).uar.mean().unstack()
    ko = lv["ft"]["konly"].groupby(level="target").mean()
    v = (a5.max(axis=1) - ko).mean()
    lam_avg = a5.mean().idxmax()
    put("cmp-ft-anchor-oracle-vs-konly-k5", v,
        f"{SF} | K=5，anchor のオラクル（ターゲットごとに λ ∈ {{0.001・0.01・0.1}} からテストで最良．分析専用）− k_only | 反復 0〜2 の平均をターゲットごとに取り，4 ターゲットで平均",
        note=doc(v, 0.4, W63, f"文書の +0.4 は，4 ターゲットの平均で λ を 1 つ選んだ値（λ={lam_avg:g}，{(a5[lam_avg] - ko).mean():+.2f}）と合う．"
                              "E61 のオラクルと同じくターゲットごとに選ぶと，この値になる"))
    lifts = {}
    for k, dv in (("start", 2.8), ("konly", 2.8), ("anchor", 2.3), ("ours-k0", 0.5), ("ours-k5", 1.3)):
        d = lv["ftc"][k] - lv["ft"][k]
        lifts[k] = tm(d)
        nm = {"start": "start-k0", "konly": "konly-k5", "anchor": "anchor-k5", "ours-k0": "ours-k0", "ours-k5": "ours-k5"}[k]
        put(f"cmp-ftc-lift-{nm}", tm(d), f"{SC}，{SF} | 中心化あり − 中心化なし（{k}．λ=0.01） | {FTP}", note=doc(tm(d), dv, W63c))
    rng("cmp-ftc-lift", [lifts["start"], lifts["konly"], lifts["anchor"]],
        f"{SC}，{SF} | 中心化あり − 中心化なし の start（K=0）・k_only（K=5）・anchor（K=5）の 3 つ | {FTP}", (2, 3), W63c, "", tol=0.5)
    for k, base in (("start-k0", E0 - lv["ft"]["start"]), ("konly-k5", E5 - lv["ft"]["konly"])):
        kk = k.split("-")[0]
        v = 100 * lifts[kk] / tm(base)
        put(f"cmp-ftc-gap-closed-{k}", v,
            f"{SC}，{SF}，{SR} | （中心化あり − なし）÷（frozen_e39 − 中心化なし）の {kk}．百分率 | 分子と分母はそれぞれ {FTP}", "%.0f")
    return out
