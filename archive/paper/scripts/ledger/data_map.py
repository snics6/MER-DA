"""台帳のモジュール（data_map）：3 節（設定）の表の件数と，Part I の地図の数値．

- `data-*`：`measure/data.py` の `load_group`・`load_any` を通した後の件数（群 × エンコーダ，クラス，話者群，
  連結した特徴），MOSEI のチャンネル対応表の割合，地図のテストがターゲットの発話に占める割合
- `map-*`：合併の C をソースのみの C に固定した地図（`results/fixc_map/<kind>/`）の組の数，そろった反復の数，
  得（合併 − ターゲットのみ）の組の平均，区間が 0 を上回る・下回る組の数，E2 のばらつきの内訳

特徴の読み込み（約 15 秒）と地図の CSV の読み込み（約 3 秒）は，入力ファイルの大きさと更新時刻を鍵にして
`.cache/data_map.json` に残し，入力が変わったときだけ計算し直す．特徴の読み込みは別のプロセスで行う
（`measure/data.py` は `sys.path` に `features/` を足して `import common` するので，台帳のほかのモジュールと
名前が混ざらないようにする）．
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import stats

from .common import MERDA, RESULTS, entry, rel

HERE = pathlib.Path(__file__).resolve().parent
CACHE = HERE / ".cache" / "data_map.json"
FIXC = RESULTS / "fixc_map"

# key の群の名前 → measure/data.py の群
GRP = {"impro": "IEMOCAP-impro", "script": "IEMOCAP-script", "mosei": "MOSEI", "meld": "MELD",
       "afew": "AFEW", "ravdess": "RAVDESS", "cremad": "CREMA-D"}
CORPUS = {"IEMOCAP-impro": "IEMOCAP", "IEMOCAP-script": "IEMOCAP", "MOSEI": "MOSEI", "MELD": "MELD",
          "AFEW": "AFEW", "RAVDESS": "RAVDESS", "CREMA-D": "CREMA-D"}
# key のエンコーダの名前 → measure/data.py の種類
ENC = {"wavlm": "audio_wavlm", "hubert": "audio_hubert", "resnet": "video", "clip": "video_clip",
       "roberta": "text_roberta", "deberta": "text_deberta"}
FUSE = {"av": "fuse-audio-video", "avt": "fuse-audio-video-text"}
TOTAL_ENC = ("wavlm", "resnet", "roberta")
CLS = ("neu", "hap", "sad", "ang")                       # label_4 の 0〜3
# 地図の種類（key の名前 → results/fixc_map/<kind>/）
KIND = {"a": "audio_wavlm", "ah": "audio_hubert", "v": "video", "av": "fuse-audio-video",
        "t": "text_roberta", "at": "fuse-audio-text", "vt": "fuse-video-text",
        "avt": "fuse-audio-video-text"}
KS = {"k1": "1", "k5": "5", "k10": "10", "k25": "25", "k50": "50", "kall": "all"}
SAME = {("IEMOCAP-impro", "IEMOCAP-script"), ("IEMOCAP-script", "IEMOCAP-impro")}   # 同じコーパスの組

# E2 のばらつきの内訳（C を固定した地図の版）
SHARE_DECOMP = RESULTS / "fixc_analysis" / "explore_relation" / "snapshots" / "20261009-0105_fixc" / "decomp_share.csv"
SHARE_H7 = RESULTS / "rerun_fixc" / "check_h7" / "share.csv"
# 同じ量の交差検証の地図の版（文書 docs/results.md 3.3 節の値の出所．note にだけ使う）
SHARE_DECOMP_CV = RESULTS / "explore_relation" / "snapshots" / "20261008-1310_rerun" / "decomp_share.csv"
SHARE_H7_CV = RESULTS / "check_h7" / "share.csv"
# 「得と K*」の変数：decomp_share.csv の 4 つと check_h7/share.csv の 2 つ（g_small は K=1・5 の得の平均）
SHARE_VARS_DECOMP = ("gain_K1", "gain_K5", "gain_all_Kall", "kstar")
SHARE_VARS_H7 = ("g_small", "kstar")
SHARE_KINDS = ("a", "ah", "v", "av")
KSTAR_RANK = {"none": 0, "1": 1, "5": 2, "10": 3, "25": 4, "50": 5, "all": 6}   # exp/pair_table.py と同じ


# ---------------------------------------------------------------- キャッシュ

def _sig(paths) -> dict:
    """入力ファイルの鍵：MER-DA からの相対パス → [大きさ, 更新時刻 ns]（無ければ None）．"""
    out = {}
    for p in sorted(set(pathlib.Path(p) for p in paths)):
        try:
            st = p.stat()
            out[rel(p)] = [st.st_size, st.st_mtime_ns]
        except FileNotFoundError:
            out[str(p)] = None
    return out


def _cached(section: str, key: dict, compute):
    store = {}
    if CACHE.exists():
        try:
            store = json.loads(CACHE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            store = {}
    hit = store.get(section)
    if hit and hit.get("key") == key:
        return hit["value"]
    value = compute()
    store[section] = {"key": key, "value": value}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(store, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, CACHE)
    return value


# ---------------------------------------------------------------- 特徴の件数（重い．別のプロセス）

def _data_inputs() -> list[pathlib.Path]:
    cache = MERDA / "data" / "cache"
    files = [p for enc in ENC.values() for p in cache.glob(f"{enc}_*.npz")]
    files += list((MERDA / "data" / "metadata").glob("*.csv"))
    files += [MERDA / "measure" / "data.py", MERDA / "measure" / "split.py", MERDA / "features" / "common.py"]
    return files


def _compute_data() -> dict:
    """特徴を読み込んで件数を数える．`_compute_data_subprocess` が別のプロセスで呼ぶ．"""
    sys.path.insert(0, str(MERDA / "measure"))
    import data as D        # measure/data.py
    import common as C      # features/common.py（measure/data.py が sys.path に足す）

    out = {"n": {}, "spk": {}, "cls": {}, "fused": {}, "mosei": {}}
    for g in D.GROUPS_ALL:
        for enc in ENC.values():
            mod = next(m for m, ks in D.MODALITIES.items() if enc in ks)
            if not (D.has_modality(g, mod) and D.available(enc, g)):
                continue
            df, _ = D.load_group(g, enc)
            out["n"][f"{g}|{enc}"] = int(len(df))
            if enc == "audio_wavlm":
                out["spk"][g] = int(df["speaker_group"].nunique())
                out["cls"][g] = [int((df["label_4"] == c).sum()) for c in range(4)]
    for kind in FUSE.values():
        for g in D.groups_for(kind):
            df, _ = D.load_any(g, kind)
            out["fused"][f"{g}|{kind}"] = int(len(df))

    # MOSEI のチャンネル対応表（data/README.md 3 節：「3,225 本中 1,122 本（35%）」は発話一覧の全行の動画で数える）
    man = pd.read_csv(C.META_DIR / "mosei_utterance_manifest.csv", usecols=["sample_id", "video_id"],
                      dtype=str, low_memory=False)
    cmap = set(pd.read_csv(C.META_DIR / "mosei_channel_map.csv", dtype=str)["video_id"])
    used = set(D.load_group("MOSEI", "audio_wavlm")[0]["sample_id"])
    mu = man[man["sample_id"].isin(used)]
    out["mosei"] = {
        "videos_all": int(man["video_id"].nunique()),
        "videos_all_chan": int(man.loc[man["video_id"].isin(cmap), "video_id"].nunique()),
        "videos_used": int(mu["video_id"].nunique()),
        "videos_used_chan": int(mu.loc[mu["video_id"].isin(cmap), "video_id"].nunique()),
        "utt_used": int(len(mu)),
        "utt_used_chan": int(mu["video_id"].isin(cmap).sum()),
        "map_rows": len(cmap),
    }
    return out


def _compute_data_subprocess() -> dict:
    py = MERDA / ".venv" / "bin" / "python"
    py = str(py) if py.exists() else sys.executable
    code = ("import json, sys; sys.path.insert(0, sys.argv[1]); "
            "from ledger.data_map import _compute_data; print(json.dumps(_compute_data()))")
    r = subprocess.run([py, "-I", "-c", code, str(HERE.parent)], capture_output=True, text=True,
                       cwd=str(MERDA), check=False)
    if r.returncode != 0:
        raise RuntimeError(f"data_map: 特徴の件数を数える子プロセスが失敗した\n{r.stderr}")
    return json.loads(r.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------- 地図の CSV（C を固定した地図）

def _map_inputs() -> list[pathlib.Path]:
    return [p for kind in KIND.values() for p in (FIXC / kind).glob("*.csv")]


def _read_map(kind: str) -> pd.DataFrame:
    fs = sorted((FIXC / kind).glob("*.csv"))
    return pd.concat([pd.read_csv(f, comment="#", dtype={"K": str, "source": str}) for f in fs],
                     ignore_index=True)


def _compute_map() -> dict:
    """組の数，全組・全 K がそろった反復の数，得の組の平均と区間の数（pairs.csv の検算用），テストの本数．"""
    out = {}
    ks = list(KS.values())
    for short, kind in KIND.items():
        df = _read_map(kind)
        df = df[df["uar"].notna()]
        mer, tgt, src = (df[df["method"] == m] for m in ("merged", "tgt_only", "src_only"))
        pairs = sorted(set(zip(mer["target"], mer["source"])))
        targets = sorted(set(mer["target"]))
        need_m = {(t, s, k) for t, s in pairs for k in ks}
        need_t = {(t, k) for t in targets for k in ks}
        reps_full = 0
        for r in sorted(df["rep"].unique()):
            m, t, s = (x[x["rep"] == r] for x in (mer, tgt, src))
            if (need_m <= set(zip(m["target"], m["source"], m["K"]))
                    and need_t <= set(zip(t["target"], t["K"]))
                    and set(pairs) <= set(zip(s["target"], s["source"]))):
                reps_full += 1
        j = mer.merge(tgt[["target", "K", "rep", "uar"]], on=["target", "K", "rep"], suffixes=("_m", "_t"))
        j["d"] = (j["uar_m"] - j["uar_t"]) * 100
        g = j.groupby(["target", "source", "K"])["d"].agg(["mean", "std", "count"]).reset_index()
        h = stats.t.ppf(0.975, g["count"] - 1) * g["std"] / np.sqrt(g["count"])
        g["lo"], g["hi"] = g["mean"] - h, g["mean"] + h
        gain = {}
        for k in ks:
            q = g[g["K"] == k]
            gain[k] = {"mean": float(q["mean"].mean()), "pos": int((q["lo"] > 0).sum()),
                       "neg": int((q["hi"] < 0).sum()), "n_pairs": int(len(q)),
                       "n_rep_min": int(q["count"].min()), "n_rep_max": int(q["count"].max())}
        out[short] = {"pairs": len(pairs), "pairs_xsame": sum(p not in SAME for p in pairs),
                      "reps_full": reps_full, "reps_seen": int(df["rep"].nunique()), "gain": gain}
        if kind == "audio_wavlm":
            nt = tgt.groupby(["target", "rep"])["n_test"].agg(["min", "max"]).reset_index()
            out["n_test"] = {f"{r.target}|{int(r.rep)}": [int(r["min"]), int(r["max"])] for _, r in nt.iterrows()}
    return out


# ---------------------------------------------------------------- E2 のばらつきの内訳

def _r2_both(x: pd.DataFrame, col: str) -> float:
    """v = μ + a_t + b_s + e の決定係数（exp/pair_table.py の decompose と同じ）．"""
    x = x[["target", "source", col]].dropna()
    y = x[col].to_numpy(float)
    cols = [np.ones(len(x))]
    for c in ("target", "source"):
        lv = sorted(x[c].unique())
        cols += [(x[c] == v).to_numpy(float) for v in lv[1:]]
    X = np.column_stack(cols)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    return float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum())


def _share_recompute(kind: str) -> dict:
    """fixc_map の summary から E2 の内訳を作り直す（スナップショットの値の検算用）．"""
    s = FIXC / kind / "summary"
    P = pd.read_csv(s / "pairs.csv", dtype={"K": str, "source": str})
    piv = P.pivot_table(index=["target", "source"], columns="K", values="diff_mean")
    d = pd.DataFrame({"gain_K1": piv["1"], "gain_K5": piv["5"], "gain_all_Kall": piv["all"]})
    d["g_small"] = (d["gain_K1"] + d["gain_K5"]) / 2
    ks = pd.read_csv(s / "kstar.csv", dtype=str).set_index(["target", "source"])
    d["kstar"] = [KSTAR_RANK.get(ks.loc[i, "K_star"], np.nan)
                  if ks.loc[i, "status"] not in ("single_rep", "iemocap_interval_undecided") else np.nan
                  for i in d.index]
    d = d.reset_index()
    return {v: _r2_both(d, v) for v in ("gain_K1", "gain_K5", "gain_all_Kall", "kstar", "g_small")}


def _share_values(kind: str, decomp: pathlib.Path, h7: pathlib.Path) -> dict:
    """（ファイル, 変数）→ r2_both．その種類の行が無いファイルは飛ばす．"""
    out = {}
    for path, vars_ in ((decomp, SHARE_VARS_DECOMP), (h7, SHARE_VARS_H7)):
        if not path.exists():
            continue
        t = pd.read_csv(path)
        for v in vars_:
            r = t[(t["kind"] == kind) & (t["variable"] == v)]
            if len(r):
                out[(rel(path), v)] = float(r["r2_both"].iloc[0])
    return out


# ---------------------------------------------------------------- entries

def entries() -> dict[str, dict]:
    e: dict[str, dict] = {}
    dat = _cached("data", _sig(_data_inputs()), _compute_data_subprocess)
    mp = _cached("map", _sig(_map_inputs()), _compute_map)

    # ---- 群 × エンコーダの件数（data/README.md 2 節「測定に入る件数」）
    for gk, g in GRP.items():
        for ek, enc in ENC.items():
            n = dat["n"].get(f"{g}|{enc}")
            if n is None:
                continue
            e[f"data-n-{gk}-{ek}"] = entry(
                n, f"measure/data.py load_group('{g}', '{enc}') ← data/cache/{enc}_{CORPUS[g]}.npz, "
                   f"data/metadata/{CORPUS[g].lower()}_utterance_manifest.csv | ok が真・有限・label_4 あり・"
                   f"話者群の補正の後の行 | 行数", "{:,}")
    for ek in TOTAL_ENC:
        enc = ENC[ek]
        gs = [g for g in GRP.values() if f"{g}|{enc}" in dat["n"]]
        e[f"data-n-total-{ek}"] = entry(
            sum(dat["n"][f"{g}|{enc}"] for g in gs),
            f"measure/data.py load_group(群, '{enc}') ← data/cache/{enc}_*.npz | {len(gs)} 群 | 行数の和", "{:,}")

    # ---- 話者群の数とクラス別の件数（WavLM）
    for gk, g in GRP.items():
        src = (f"measure/data.py load_group('{g}', 'audio_wavlm') ← data/cache/audio_wavlm_{CORPUS[g]}.npz, "
               f"data/metadata/{CORPUS[g].lower()}_utterance_manifest.csv")
        e[f"data-spk-{gk}"] = entry(dat["spk"][g], f"{src} | speaker_group 列 | 異なる値の数", "{:,}")
        for c, ck in enumerate(CLS):
            e[f"data-{gk}-{ck}"] = entry(dat["cls"][g][c], f"{src} | label_4 = {c} の行 | 行数", "{:,}")

    # ---- 連結した特徴の件数
    for fk, kind in FUSE.items():
        for gk, g in GRP.items():
            n = dat["fused"].get(f"{g}|{kind}")
            if n is None:
                continue
            e[f"data-n-{gk}-{fk}"] = entry(
                n, f"measure/data.py load_any('{g}', '{kind}')（load_fused：各モダリティの load_group の "
                   f"sample_id の共通部分）| 行 | 行数", "{:,}")

    # ---- MOSEI のチャンネル対応表
    ms = dat["mosei"]
    e["data-mosei-chan-pct"] = entry(
        100 * ms["videos_all_chan"] / ms["videos_all"],
        "data/metadata/mosei_channel_map.csv, data/metadata/mosei_utterance_manifest.csv | 発話一覧の全行の "
        "video_id（data/README.md 3 節の数え方）| 対応表にある動画の数 ÷ 動画の数 ×100",
        "%.0f",
        f"動画の割合（{ms['videos_all_chan']:,}／{ms['videos_all']:,}）．測定に入る 18,919 発話に限ると動画で "
        f"{ms['videos_used_chan']:,}／{ms['videos_used']:,}＝{100 * ms['videos_used_chan'] / ms['videos_used']:.1f}%，"
        f"発話で {ms['utt_used_chan']:,}／{ms['utt_used']:,}＝{100 * ms['utt_used_chan'] / ms['utt_used']:.1f}%")
    e["data-mosei-chan-videos"] = entry(
        ms["videos_all_chan"], "data/metadata/mosei_channel_map.csv, data/metadata/mosei_utterance_manifest.csv | "
        "発話一覧の全行の video_id のうち対応表にあるもの | 異なる値の数", "{:,}")
    e["data-mosei-videos"] = entry(
        ms["videos_all"], "data/metadata/mosei_utterance_manifest.csv | 全行の video_id | 異なる値の数", "{:,}")

    # ---- 地図のテストがターゲットの発話に占める割合（群 × 反復）
    pct = []
    for k, (lo, hi) in mp["n_test"].items():
        g, _ = k.split("|")
        assert lo == hi, f"data_map: {k} の n_test が反復の中で一定でない"
        pct.append(100 * lo / dat["n"][f"{g}|audio_wavlm"])
    src_t = (f"results/fixc_map/audio_wavlm/*.csv, measure/data.py load_group(群, 'audio_wavlm') | "
             f"tgt_only の行の n_test ÷ その群の行数，{len(mp['n_test'])} 個（7 群 × 反復）")
    e["data-test-pct-mean"] = entry(np.mean(pct), f"{src_t} | 平均 ×100", "%.1f")
    e["data-test-pct-lo"] = entry(np.min(pct), f"{src_t} | 最小 ×100", "%.1f")
    e["data-test-pct-hi"] = entry(np.max(pct), f"{src_t} | 最大 ×100", "%.1f")

    # ---- 地図：組の数，そろった反復の数
    for sk, kind in KIND.items():
        m = mp[sk]
        base = f"results/fixc_map/{kind}/*.csv"
        e[f"map-pairs-{sk}"] = entry(
            m["pairs"], f"{base} | method = merged の (target, source) | 順序つきの組の数（同じコーパスの組を含む）", "%d")
        e[f"map-pairs-{sk}-xsame"] = entry(
            m["pairs_xsame"], f"{base} | method = merged の (target, source)，IEMOCAP 即興↔台本を除く | 順序つきの組の数",
            "%d")
        note = None
        if m["reps_full"] != m["reps_seen"]:
            note = f"CSV に現れる反復は {m['reps_seen']} 個で，全組・全 K がそろうのはそのうち {m['reps_full']} 個"
        e[f"map-reps-{sk}"] = entry(
            m["reps_full"], f"{base} | 3 つの作り方 × 全組 × K（1，5，10，25，50，全件）の uar がそろう rep | 数", "%d",
            note)

    # ---- 地図：得の組の平均と，区間が 0 を上回る・下回る組の数
    for sk, kind in KIND.items():
        pfile = FIXC / kind / "summary" / "pairs.csv"
        P = pd.read_csv(pfile, dtype={"K": str, "source": str}) if pfile.exists() else None
        for kk, K in KS.items():
            raw = mp[sk]["gain"][K]
            note = None
            if P is not None:
                q = P[P["K"] == K]
                val = (float(q["diff_mean"].mean() * 100), int((q["diff_lo"] > 0).sum()),
                       int((q["diff_hi"] < 0).sum()))
                same = (abs(val[0] - raw["mean"]) < 1e-9 and val[1] == raw["pos"] and val[2] == raw["neg"]
                        and len(q) == raw["n_pairs"])
            if P is not None and same:
                src = (f"results/fixc_map/{kind}/summary/pairs.csv | K={K}，{len(q)} 組（同じコーパスの組を含む），"
                       f"各組 {int(q['n_rep'].min())} 回")
            else:
                if P is not None:
                    note = ("summary/pairs.csv が地図の CSV から作り直した値と食い違う（summary が古い）ので，"
                            "地図の CSV から作った値を使った")
                val = (raw["mean"], raw["pos"], raw["neg"])
                src = (f"results/fixc_map/{kind}/*.csv | merged と同じ (target, K, rep) の tgt_only の対の差，"
                       f"K={K}，{raw['n_pairs']} 組（同じコーパスの組を含む）")
            note_k = note
            if sk == "a" and kk == "k1":
                note_k = "; ".join(x for x in (note, (
                    "docs/results.md 3.9 節と paper/CLAIM.md の「A の K=1 で +9.4〜+10.5」は，交差検証の地図の +9.4 と "
                    "C を固定した地図の +10.5 を並べたもの．C を固定した地図の値はこの key（E17 の表の +10.5 と一致）")) if x)
            e[f"map-gain-{sk}-{kk}"] = entry(
                val[0], f"{src} | 組ごとの diff_mean（合併 − ターゲットのみ，同じ反復の対の差の平均）の組の平均 ×100（ポイント）",
                "%+.1f", note_k)
            e[f"map-gain-pos-{sk}-{kk}"] = entry(
                val[1], f"{src} | 対の差の 95% t 区間の下端 diff_lo > 0 の組の数", "%d", note)
            e[f"map-gain-neg-{sk}-{kk}"] = entry(
                val[2], f"{src} | 対の差の 95% t 区間の上端 diff_hi < 0 の組の数", "%d", note)

    # ---- E2 のばらつきの内訳：得と K* のばらつきのうちターゲットとソースの効果で説明できる割合
    for sk in SHARE_KINDS:
        kind = KIND[sk]
        vals = _share_values(kind, SHARE_DECOMP, SHARE_H7)
        if not vals:
            continue
        chk = _share_recompute(kind)
        bad = [f"{v}（{f.split('/')[-1]}）" for (f, v), r in vals.items() if abs(r - chk[v]) > 1e-6]
        cv = _share_values(kind, SHARE_DECOMP_CV, SHARE_H7_CV)
        files = sorted({f for f, _ in vals})
        used = "，".join(f"{v}（{'check_h7' if 'check_h7' in f else 'decomp_share'}）" for f, v in sorted(vals))
        src = (f"{' と '.join(files)} | kind={kind}，変数 {used} | 列 r2_both（v = μ + a_t + b_s の決定係数）の")
        doc = {"a": "W7 51〜73%", "ah": "Hu7 64〜74%", "v": "V 66〜87%", "av": "AV 73〜88%"}[sk]
        note = f"docs/results.md 3.3 節（主張 1）の「{doc}」は交差検証の地図の値である"
        if cv:
            cv_files = "，".join(sorted({f for f, _ in cv}))
            note += (f"．同じ変数を交差検証の地図の版（{cv_files}）で数えると "
                     f"{100 * min(cv.values()):.0f}〜{100 * max(cv.values()):.0f}% で，文書の値を再現する")
        note += (f"．C を固定した地図では {100 * min(vals.values()):.0f}〜{100 * max(vals.values()):.0f}%"
                 f"（fixc_map/{kind}/summary の pairs.csv・kstar.csv から作り直した値と"
                 f"{'食い違う変数がある：' + '，'.join(bad) if bad else '一致する'}）．"
                 f"g_small は K=1・5 の得の平均で，check_h7 の表には音声の 2 種類だけがある")
        e[f"map-share-{sk}-lo"] = entry(100 * min(vals.values()), src + "最小 ×100", "%.0f", note)
        e[f"map-share-{sk}-hi"] = entry(100 * max(vals.values()), src + "最大 ×100", "%.0f", note)
    return e
