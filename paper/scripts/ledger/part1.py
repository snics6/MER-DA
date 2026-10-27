"""台帳のモジュール（part1）：論文 4 節（Part I：少数のラベルが分類器の何を直すか）の数値．

docs/paper-claim.md 4 節の「最小の証拠 1」（E12，E13，E28 の 9，E29，E30）と「付随する証拠」（E15，E16，E32）．
値は結果ファイルからその場で計算する（docs/ の値は写さない）．文書の値は検算の相手として `doc=` に書き，
丸めの範囲（表示の最後の桁の半分）を超えて食い違うときだけ note に「文書の値と差の理由」を書く．
保存した要約（tests・summary の CSV）と計算し直した値が食い違うときも note に書く．

地図は合併の C をソースのみの C に固定した地図（results/fixc_map/）．E12 の学び直し（results/mech_<kind>/）は
C を交差検証で選んでいるので，地図に依る量（合併 − ソースのみ，合併 − ターゲットのみ）だけが地図で変わる．

入力はどれも小さな CSV で，全体で 1 秒ほどで終わる（.cache は使わない）．
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from scipy import stats

from .common import RESULTS, SNAP, entry

KIND = {"a": "audio_wavlm", "ah": "audio_hubert", "v": "video", "av": "fuse-audio-video",
        "t": "text_roberta", "at": "fuse-audio-text", "vt": "fuse-video-text", "avt": "fuse-audio-video-text"}
TEXT = ("t", "at", "vt", "avt")

FIXC_RERUN = RESULTS / "rerun_fixc" / "explore_relation" / "snapshots" / "20261018-0213_fixc"
STAGEB = RESULTS / "fixc_analysis" / "explore_relation" / "snapshots" / "20261009-0105_fixc" / "stageB"
E28T9 = SNAP / "20261011-0139_e28t9"
E29 = SNAP / "20261011-1524_e29"
E30 = SNAP / "20261013-1346_e30"
E32 = SNAP / "20261016-0954_e32"

R3 = "docs/results.md 3.2.3"
EX = "docs/exploratory-analyses.md"


# ---------------------------------------------------------------- 道具

def _unit(fmt: str) -> float:
    if fmt == "%d":
        return 1.0
    return 10.0 ** -int(fmt.split(".")[1][0])


class Book:
    """key → entry を集める．doc（文書の値）と ref（保存した要約の値）との照合を note にする．"""

    def __init__(self):
        self.out: dict[str, dict] = {}

    def add(self, key, value, source, fmt="%.1f", doc=None, why=None, ref=None, note=None):
        notes = []
        if doc is not None:
            dval, where = doc
            if abs(float(value) - dval) > 0.5 * _unit(fmt) + 1e-9:
                notes.append(f"文書（{where}）の値 {fmt % dval if fmt != '%d' else int(dval)}．" + (why or "差の理由は未確認"))
        if ref is not None:
            rval, rname = ref
            if not (isinstance(rval, float) and math.isnan(rval)) and abs(float(value) - float(rval)) > 1e-6 * max(1.0, abs(float(rval))):
                notes.append(f"保存した要約 {rname} の値 {rval:.6g} と食い違う")
        if note:
            notes.append(note)
        if key in self.out:
            raise KeyError(f"part1: key '{key}' が重複している")
        self.out[key] = entry(int(value) if fmt == "%d" else float(value), source, fmt, "；".join(notes) or None)

    def rng(self, key, values: dict, source, fmt="%.2f", doc=None, why=None):
        """values（ラベル → 値）の最小を key-lo，最大を key-hi に入れる．doc は ((lo, hi), where)．"""
        lo, hi = min(values.values()), max(values.values())
        lab_lo = min(values, key=values.get)
        lab_hi = max(values, key=values.get)
        src = f"{source} | 範囲（最小 {lab_lo}，最大 {lab_hi}．対象 {', '.join(values)}）"
        dlo = dhi = None
        if doc is not None:
            (a, b), where = doc
            dlo, dhi = (a, where), (b, where)
        self.add(f"{key}-lo", lo, src, fmt, doc=dlo, why=why)
        self.add(f"{key}-hi", hi, src, fmt, doc=dhi, why=why)


def _cat(folder, cols) -> pd.DataFrame:
    ds = [pd.read_csv(f, comment="#", dtype={"K": str, "source": str}, usecols=cols) for f in sorted(folder.glob("*.csv"))]
    ds = [d for d in ds if len(d)]
    return pd.concat(ds, ignore_index=True)


def _rel(p) -> str:
    return str(p.relative_to(RESULTS.parent))


def _spearman(a, b) -> float:
    return float(stats.spearmanr(np.asarray(a, float), np.asarray(b, float)).statistic)


# ---------------------------------------------------------------- E12：原因を取り除いて学び直す

E12_DOC = {  # 文書の値（docs/results.md 3.2.3，docs/exploratory-analyses.md E12．どちらも交差検証の地図）
    "a": dict(center=4.9, up=37, pairs=42, d1=4.9, merged=4.7, cd1=0.3, cmerged=0.6, map25=-1.7, wt25=2.0, l1map=21, l1wt=6),
    "ah": dict(center=4.2, up=37, pairs=42, d1=4.1, merged=3.9, cd1=0.2, cmerged=0.1, map25=-1.2, wt25=2.3),
    "v": dict(center=2.5, up=23, pairs=30, d1=1.9, merged=0.2, cd1=-0.5, cmerged=-1.6, map25=-1.6, wt25=0.7, l1map=14, l1wt=8),
}
WHY_CV = ("文書は交差検証の地図の合併との比べ．C を固定した地図では K ≤ 25 で合併が良くなる（E17）ので，"
          "地図の合併を使う量だけが変わる（docs/exploratory-analyses.md「交差検証の地図でしか数値の無かった解析を，C を固定した地図で計算し直す」の E12）")


def _e12(b: Book) -> None:
    ref_src = pd.read_csv(FIXC_RERUN / "e12_src.csv")
    ref_d1 = pd.read_csv(FIXC_RERUN / "e12_d1.csv", dtype={"K": str})
    ref_c = pd.read_csv(FIXC_RERUN / "e12_center.csv", dtype={"K": str})
    ref_w = pd.read_csv(FIXC_RERUN / "e12_wt.csv", dtype={"K": str})
    nreps = set()
    for k in ("a", "ah", "v"):
        kind = KIND[k]
        doc = E12_DOC[k]
        mech = _cat(RESULTS / f"mech_{kind}", ["op", "target", "source", "method", "K", "rep", "uar"])
        reps = mech.groupby("target").rep.agg(lambda r: tuple(sorted(set(r))))
        if reps.nunique() != 1:
            raise ValueError(f"part1 E12: {kind} の学び直しの反復がターゲットでそろっていない: {reps.to_dict()}")
        rr = reps.iloc[0]
        nreps.add(len(rr))
        tr = mech[["target", "rep"]].drop_duplicates()
        mp = _cat(RESULTS / "fixc_map" / kind, ["target", "source", "method", "K", "rep", "uar"]).merge(tr)
        t = mp[mp.method == "tgt_only"][["target", "K", "rep", "uar"]].rename(columns={"uar": "t"})
        s = mp[mp.method == "src_only"][["target", "source", "rep", "uar"]].rename(columns={"uar": "s"})
        m = mp[mp.method == "merged"][["target", "source", "K", "rep", "uar"]].rename(columns={"uar": "m"})

        def mm(op, method, name):
            x = mech[(mech.op == op) & (mech.method == method)]
            return x[["target", "source", "K", "rep", "uar"]].rename(columns={"uar": name})

        reps_txt = f"反復 {rr[0]}〜{rr[-1]}（学び直しの全反復．{len(rr)} 回）"
        mech_p, map_p = f"results/mech_{kind}/*.csv", f"results/fixc_map/{kind}/*.csv"
        cmp_src = f"{_rel(FIXC_RERUN)}/e12_*.csv と照合"

        # 中心をそろえたソースのみ − ソースのみ（ソースのみの行は 2 つの地図で同じ）
        x = mm("center", "src_only", "c").drop(columns="K").merge(s)
        pp = (x.c - x.s).groupby([x.target, x.source]).mean() * 100
        rs = ref_src[(ref_src.setting == kind) & (ref_src.op == "center")].iloc[0]
        src = (f"{mech_p}（op=center, method=src_only）と {map_p}（method=src_only） | {reps_txt}，{len(pp)} 組（同じコーパスの組を含む） | "
               f"(target, source, rep) で対にした UAR の差を反復で平均し，組で平均（×100）．{cmp_src}")
        b.add(f"p1-center-srconly-{k}", pp.mean(), src, doc=(doc["center"], R3), ref=(rs.mean_diff * 100, "e12_src.csv mean_diff"))
        b.add(f"p1-center-srconly-up-{k}-n", int((pp > 0).sum()), src + "．差が正の組の数", "%d", doc=(doc["up"], R3), ref=(rs.n_up, "e12_src.csv n_up"))
        b.add(f"p1-center-srconly-pairs-{k}-n", len(pp), src + "．組の数", "%d", doc=(doc["pairs"], R3))

        # K=5：D1 − ソースのみ，合併 − ソースのみ
        y = mm("none", "src_only+D1", "d1").merge(s).merge(m)
        h = y[y.K == "5"].groupby(["target", "source"])[["d1", "s", "m"]].mean() * 100
        rd = ref_d1[(ref_d1.setting == kind) & (ref_d1.op == "none") & (ref_d1.K == "5")].iloc[0]
        src = (f"{mech_p}（op=none, method=src_only+D1, K=5）と {map_p}（src_only，merged K=5） | {reps_txt}，{len(h)} 組 | "
               f"組ごとに反復で平均した差の組の平均（×100）．{cmp_src}")
        b.add(f"p1-d1-gain-{k}-k5", (h.d1 - h.s).mean(), src + "．D1 − ソースのみ", doc=(doc["d1"], R3),
              ref=(rd.mean_d1_minus_s * 100, "e12_d1.csv mean_d1_minus_s"))
        b.add(f"p1-merged-gain-{k}-k5", (h.m - h.s).mean(), src + "．合併（C 固定の地図）− ソースのみ", doc=(doc["merged"], R3), why=WHY_CV,
              ref=(rd.mean_m_minus_s * 100, "e12_d1.csv mean_m_minus_s"))

        # K=5：中心をそろえた後の D1 と合併（どちらも中心をそろえたソースのみから．学び直しだけで決まる）
        z = mm("center", "src_only+D1", "d1c").merge(mm("center", "src_only", "sc").drop(columns="K")).merge(mm("center", "merged", "mc"))
        h = z[z.K == "5"].groupby(["target", "source"])[["d1c", "sc", "mc"]].mean() * 100
        rd = ref_d1[(ref_d1.setting == kind) & (ref_d1.op == "center") & (ref_d1.K == "5")].iloc[0]
        rc = ref_c[(ref_c.setting == kind) & (ref_c.K == "5")].iloc[0]
        src = (f"{mech_p}（op=center の src_only，src_only+D1，merged．K=5） | {reps_txt}，{len(h)} 組 | "
               f"中心をそろえたソースのみからの差を組ごとに反復で平均し，組の平均（×100）．地図を使わない．学び直しの C は交差検証．{cmp_src}")
        b.add(f"p1-center-d1-gain-{k}-k5", (h.d1c - h.sc).mean(), src + "．中心をそろえた D1 − 中心をそろえたソースのみ",
              doc=(doc["cd1"], R3), why="丸めの差（0.1497 を文書は 0.2 と書いた）" if k == "ah" else None,
              ref=(rd.mean_d1_minus_s * 100, "e12_d1.csv mean_d1_minus_s（op=center）"))
        b.add(f"p1-center-merged-gain-{k}-k5", (h.mc - h.sc).mean(), src + "．中心をそろえた合併 − 中心をそろえたソースのみ",
              doc=(doc["cmerged"], R3), ref=(rc.m_minus_s_center * 100, "e12_center.csv m_minus_s_center"))

        # K=25：ターゲットの重み α を交差検証で選ぶ（op=wt）
        w = mm("wt", "merged", "wt").merge(m).merge(t)
        h = w[w.K == "25"].groupby(["target", "source"])[["wt", "m", "t"]].mean() * 100
        gm, gw = h.m - h.t, h.wt - h.t
        rw = ref_w[(ref_w.setting == kind) & (ref_w.K == "25")].iloc[0]
        src = (f"{mech_p}（op=wt, method=merged, K=25．α は nat・0.25・0.5・0.75 から交差検証）と {map_p}（merged，tgt_only K=25） | "
               f"{reps_txt}，{len(h)} 組 | 組ごとに反復で平均した得（− ターゲットのみ）の組の平均（×100）．{cmp_src}")
        b.add(f"p1-wt-gain-{k}-k25-map", gm.mean(), src + "．地図の合併（C 固定）の得", doc=(doc["map25"], R3), why=WHY_CV,
              ref=(rw.gain_nat * 100, "e12_wt.csv gain_nat"))
        b.add(f"p1-wt-gain-{k}-k25", gw.mean(), src + "．α を選んだ合併の得", doc=(doc["wt25"], R3), ref=(rw.gain_wt * 100, "e12_wt.csv gain_wt"))
        dl = (doc["l1map"], f"{EX} E12 の 3") if "l1map" in doc else None
        b.add(f"p1-wt-loss1-{k}-k25-map-n", int((gm < -1).sum()), src + "．地図の合併で 1 点を超えて損する組（得 < −1 点）の数", "%d",
              doc=dl, why=WHY_CV, note=None if dl else "文書に値なし")
        dl = (doc["l1wt"], f"{EX} E12 の 3") if "l1wt" in doc else None
        b.add(f"p1-wt-loss1-{k}-k25-n", int((gw < -1).sum()), src + "．α を選んだ合併で 1 点を超えて損する組（得 < −1 点）の数", "%d",
              doc=dl, note=None if dl else "文書に値なし")
    if len(nreps) != 1:
        raise ValueError(f"part1 E12: 種類で反復の数が違う {nreps}")
    b.add("p1-e12-reps-n", nreps.pop(), "results/mech_{audio_wavlm,audio_hubert,video}/*.csv | rep 列 | 学び直しの反復の数（3 種類・全ターゲットで同じことを確かめた）", "%d")


# ---------------------------------------------------------------- E13：位置による損と向きによる損

E13_DOC = {"a": (4.5, 8.5), "ah": (4.0, 8.8), "v": (2.4, 1.9), "av": (2.5, 6.0), "t": (0.6, 12.2)}
E13_FIT_DOC = {"a": (0.992, 1.0), "v": (0.991, 1.1), "av": (0.996, 0.8)}


def _e13(b: Book) -> None:
    p = pd.read_csv(FIXC_RERUN / "e13_pairs.csv")
    hub = pd.read_csv(FIXC_RERUN / "check_mh" / "e13.csv").set_index("setting").loc["audio_hubert"]
    summ = pd.read_csv(FIXC_RERUN / "e13_summary.csv").set_index("setting")
    fit = pd.read_csv(FIXC_RERUN / "e13_fit.csv").set_index("setting")
    where = f"{EX} E13 の 2，docs/results.md 3.2.4"
    off, dirl, rr, mae = {}, {}, {}, {}
    for k, kind in KIND.items():
        if k == "ah":
            n = int(hub.n_pairs)
            off[k], dirl[k] = hub.mean_loss_offset * 100, hub.mean_loss_direction * 100
            rr[k], mae[k] = float(hub.r_model_actual), hub.mean_abs_err * 100
            src = (f"{_rel(FIXC_RERUN)}/check_mh/e13.csv | setting=audio_hubert（{n} 組）| 組ごとの 6 対の平均を組で平均した要約をそのまま使う"
                   "（HuBERT は組 × 感情の対の表が保存されていない）．ソースのみの分類器の点数から測るので地図と反復に依らない")
            ref_o = ref_d = ref_r = ref_m = None
        else:
            g = p[p.kind == kind]
            pp = g.groupby(["target", "source"])[["loss_offset", "loss_direction"]].mean() * 100
            n = len(pp)
            off[k], dirl[k] = pp.loss_offset.mean(), pp.loss_direction.mean()
            rr[k] = float(np.corrcoef(g.acc_model, g.acc_actual)[0, 1])
            mae[k] = float((g.acc_model - g.acc_actual).abs().mean() * 100)
            src = (f"{_rel(FIXC_RERUN)}/e13_pairs.csv | kind={kind}，{n} 組 × 6 対 | 組ごとに 6 対で平均し組で平均（×100．2 つの感情の正解率の点）．"
                   "ソースのみの分類器の点数から測るので地図と反復に依らない（snapshots/20261002-1339_e13 と同じ表）")
            sm, ft = summ.loc[kind], fit.loc[kind]
            ref_o, ref_d = (sm.mean_loss_offset * 100, "e13_summary.csv"), (sm.mean_loss_direction * 100, "e13_summary.csv")
            ref_r, ref_m = (ft.r_model_actual, "e13_fit.csv"), (ft.mean_abs_err * 100, "e13_fit.csv")
        d = E13_DOC.get(k)
        b.add(f"p1-e13-loss-offset-{k}", off[k], src + "．位置による損 loss_offset", doc=(d[0], where) if d else None, ref=ref_o)
        b.add(f"p1-e13-loss-dir-{k}", dirl[k], src + "．向きによる損 loss_direction", doc=(d[1], where) if d else None, ref=ref_d)
        fsrc = src.replace("組ごとに 6 対で平均し組で平均（×100．2 つの感情の正解率の点）", "組 × 感情の対のすべての升で，2 つの山の模型の正解率 acc_model と実際の正解率 acc_actual を比べる")
        fd = E13_FIT_DOC.get(k)
        b.add(f"p1-e13-fit-r-{k}", rr[k], fsrc + "．ピアソンの相関", "%.3f", doc=(fd[0], f"{EX} E13 の 1") if fd else None, ref=ref_r)
        b.add(f"p1-e13-fit-mae-{k}", mae[k], fsrc + "．差の絶対値の平均（×100，点）", doc=(fd[1], f"{EX} E13 の 1") if fd else None, ref=ref_m)
    tsrc = f"{_rel(FIXC_RERUN)}/e13_pairs.csv | 言葉を含む 4 種類（各 12 組 × 6 対） | 種類ごとの組の平均（×100）"
    b.rng("p1-e13-loss-offset-text", {k: off[k] for k in TEXT}, tsrc + "．位置による損", "%.1f", doc=((0.6, 1.1), "docs/results.md 3.2.4"))
    b.rng("p1-e13-loss-dir-text", {k: dirl[k] for k in TEXT}, tsrc + "．向きによる損", "%.1f", doc=((7.1, 12.2), "docs/results.md 3.2.4"))
    asrc = (f"{_rel(FIXC_RERUN)}/e13_pairs.csv（7 種類）と check_mh/e13.csv（HuBERT） | 8 種類 | 種類ごとの値")
    b.rng("p1-e13-fit-r", rr, asrc + "．模型と実際の正解率の相関", "%.3f", doc=((0.986, 0.996), EX + " E13 の 1"))
    b.rng("p1-e13-fit-mae", mae, asrc + "．差の絶対値の平均（点）", "%.1f", doc=((0.7, 1.1), EX + " E13 の 1"))


# ---------------------------------------------------------------- E28 の 9：q_dir と，10 本で直らない向きの損

def _e28t9(b: Book) -> None:
    T = pd.read_csv(E28T9 / "e28t9_targets.csv")
    P = pd.read_csv(E28T9 / "e28t9_pairs.csv", dtype={"K": str})
    tests = pd.read_csv(E28T9 / "e28t9_tests.csv")
    prov = json.loads((E28T9 / "provenance_audio_wavlm.json").read_text())
    reps = prov["reps"]
    rtxt = f"反復 {reps[0]}〜{reps[-1]}（{len(reps)} 回．ターゲットのみの点数は {prov['map_dir']} と同じ条件で学び直した）"
    where = f"{EX} E28 の 9"
    qdoc = {"a": 16.1, "ah": 22.2, "v": 1.9, "av": 6.8}
    bound_doc = {"a": 40, "ah": 32, "v": 93, "av": 37}
    ht_doc = {"a": 23, "ah": 22, "v": 10, "av": 37}
    q, bound, remain, ht = {}, {}, {}, {}
    for k in ("a", "ah", "v", "av"):
        kind = KIND[k]
        tt = T[T.kind == kind]
        q[k] = float(tt.q_dir.median())
        b.add(f"p1-qdir-{k}", q[k], f"{_rel(E28T9)}/e28t9_targets.csv | kind={kind}，{len(tt)} ターゲット，{rtxt} | "
              "ターゲットごとの q_dir（ターゲットのみの d を K=1・5・10 で測り d_T(K) = D/√(1 + q/K) を当てた値）の中央値",
              doc=(qdoc[k], where))
        g = P[(P.kind == kind) & (~P.same_corpus) & (P.K == "10")]
        src = f"{_rel(E28T9)}/e28t9_pairs.csv | kind={kind}，K=10，同じコーパスの組を除く {len(g)} 組，{rtxt}，合併は C を固定した段階 B |"
        sh = ((g.d_star_dir - g.d_s) / (g.D - g.d_s)).median() * 100
        rt = tests[(tests.kind == kind) & (tests.test == "U3")].median_share_recoverable_K10.iloc[0] * 100
        bound[k], remain[k] = sh, 100 - sh
        b.add(f"p1-dirloss-recov-bound-k10-{k}", sh, src + " 基準モデル d*(10)（q_dir）が取り戻せる向きの損の割合 (d* − d_s)/(D − d_s) の組の中央値（%）",
              "%.0f", doc=(bound_doc[k], where + " の U3"), why="丸めの差（0.3645 を文書は 0.37 と書いた）" if k == "av" else None,
              ref=(rt, "e28t9_tests.csv U3"))
        b.add(f"p1-dirloss-remain-bound-k10-{k}", 100 - sh, src + " 100 − 上の割合（基準モデルでも 10 本で直らない向きの損の割合，%）", "%.0f")
        hv = ((g.d_h_default - g.d_s_obs) / (g.D - g.d_s)).median() * 100
        ht[k] = hv
        b.add(f"p1-dirloss-recov-ht-k10-{k}", hv, src + " 既定の仮説転移（C′=0.01，C_b=8）が取り戻した向きの損の割合 (d_h − d_s_obs)/(D − d_s) の組の中央値（%）．"
              "分子の d_s_obs は段階 B のテストで測ったソースのみの d，分母の d_s は E13 の d（文書の U4 の値はこの組み合わせで再現する）",
              "%.0f", doc=(ht_doc[k], where + " の U4"))
    asrc = f"{_rel(E28T9)}/e28t9_pairs.csv・e28t9_targets.csv | 音声 2 エンコーダ（a，ah），K=10，{rtxt} | 種類ごとの値"
    b.rng("p1-qdir-audio", {k: q[k] for k in ("a", "ah")}, asrc + "．q_dir の中央値", "%.1f", doc=((16.1, 22.2), where))
    b.rng("p1-dirloss-recov-bound-k10-audio", {k: bound[k] for k in ("a", "ah")}, asrc + "．基準モデルが取り戻せる割合（%）", "%.0f",
          doc=((32, 40), where))
    b.rng("p1-dirloss-remain-bound-k10-audio", {k: remain[k] for k in ("a", "ah")}, asrc + "．基準モデルでも直らない割合（%．文書は「6〜7 割」）", "%.0f")
    b.rng("p1-dirloss-recov-ht-k10-audio", {k: ht[k] for k in ("a", "ah")}, asrc + "．既定の仮説転移が取り戻した割合（%）", "%.0f",
          doc=((22, 23), where))
    b.add("p1-e28t9-reps-n", len(reps), f"{_rel(E28T9)}/provenance_audio_wavlm.json | reps | ターゲットのみを学び直した反復の数", "%d")


# ---------------------------------------------------------------- E29：主張 1 の検定（8 種類）

def _e29(b: Book) -> None:
    p = pd.read_csv(E29 / "e29_pairs.csv", dtype={"K": str})
    tests = pd.read_csv(E29 / "e29_tests.csv")
    summ = pd.read_csv(E29 / "e29_summary.csv").set_index("group")
    g5 = p[p.K == "5"]
    where = f"{EX} E29 の結果"
    base = (f"{_rel(E29)}/e29_pairs.csv | K=5，C を固定した条件（E25 の手と C を固定した段階 B），反復 0〜9（10 回），"
            "同じコーパスの組を除く")
    groups = {k: g5[g5.kind == KIND[k]] for k in KIND} | {"text4": g5[g5.group == "text4"]}
    share_doc = {"a": 0.62, "ah": 0.54, "v": 0.65, "av": 0.47, "text4": 0.36}
    loss_doc = {"a": 0.24, "ah": 0.18, "v": 0.23, "av": 0.27, "text4": 0.10}
    corr_doc = {"a": 0.78, "ah": 0.61, "text4": 0.59, "v": 0.44, "av": 0.70}
    gcrit_doc = {"a": 4.8, "ah": 4.3, "v": 1.9, "av": 2.4, "text4": 0.9}
    lcrit_doc = {"a": 4.7, "ah": 4.2, "v": 2.5, "av": 2.5, "text4": 1.0}
    share, loss, corr = {}, {}, {}
    for k, g in groups.items():
        lab = "言葉を含む 4 種類をまとめた" if k == "text4" else f"kind={KIND[k]}"
        src = f"{base}，{lab} {len(g)} 組 |"
        share[k] = float(g.share_pos.median())
        loss[k] = float(g.share_loss.median())
        corr[k] = _spearman(g.G_crit, g.L_crit)
        rs = summ.loc[k if k == "text4" else KIND[k]]
        tname = k if k == "text4" else KIND[k]
        rt = tests[(tests.group == tname) & tests.test.str.startswith("P29-1")].rho.iloc[0]
        b.add(f"p1-e29-critshare-k5-{k}", share[k], src + " 合併の得に占める基準（位置）の分の割合 share_pos = 位置の分/(位置の分 + 向きの分) の組の中央値",
              "%.2f", doc=(share_doc[k], where) if k in share_doc else None, ref=(rs.share_pos_median, "e29_summary.csv share_pos_median"))
        b.add(f"p1-e29-lossshare-{k}", loss[k], src + " 損に占める基準の損の割合 share_loss = L_crit/(L_crit + L_disc) の組の中央値",
              "%.2f", doc=(loss_doc[k], where) if k in loss_doc else None, ref=(rs.share_loss_median, "e29_summary.csv share_loss_median"))
        b.add(f"p1-e29-critcorr-k5-{k}", corr[k], src + " 基準の直し G_crit（D1 − ソースのみ）と基準の損 L_crit（E13）のスピアマンの順位相関（P29-1）",
              "%.2f", doc=(corr_doc[k], where) if k in corr_doc else None, ref=(rt, "e29_tests.csv P29-1 rho"))
        if k in gcrit_doc:
            b.add(f"p1-e29-gcrit-k5-{k}", g.G_crit.mean(), src + " 基準の直し G_crit = D1 − ソースのみ（UAR の点）の組の平均",
                  doc=(gcrit_doc[k], where), ref=(rs.G_crit, "e29_summary.csv G_crit"))
            b.add(f"p1-e29-lcrit-{k}", g.L_crit.mean(), src + " 基準の損 L_crit（E13 の位置による損．2 つの感情の正解率の点）の組の平均",
                  doc=(lcrit_doc[k], where), ref=(rs.L_crit, "e29_summary.csv L_crit"))
    b.rng("p1-e29-critshare-k5-audio-video", {k: share[k] for k in ("a", "ah", "v")}, f"{base} | 音声 2 エンコーダと映像（a，ah，v）| 種類ごとの share_pos の中央値",
          doc=((0.54, 0.65), where + "（「音声と映像で 5〜6 割」）"))
    b.rng("p1-e29-critshare-k5-text", {k: share[k] for k in TEXT}, f"{base} | 言葉を含む 4 種類 | 種類ごとの share_pos の中央値")
    b.rng("p1-e29-lossshare", {k: loss[k] for k in ("a", "ah", "v", "av", "text4")},
          f"{base} | a，ah，v，av と言葉を含む 4 種類をまとめた 40 組 | 種類ごとの share_loss の中央値", doc=((0.10, 0.27), where))
    b.rng("p1-e29-critcorr-k5", {k: corr[k] for k in ("a", "ah", "text4")},
          f"{base} | 判定に使った 3 つ（a，ah，言葉を含む 4 種類をまとめた 40 組）| G_crit と L_crit の順位相関", doc=((0.59, 0.78), where + "（「+0.6〜+0.8」）"))
    rt = tests[tests.group == "all8"].rho.iloc[0]
    b.add("p1-e29-shares-corr-all8", _spearman(g5.share_pos, g5.share_loss),
          f"{base}，8 種類まとめて {len(g5)} 組 | share_pos と share_loss のスピアマンの順位相関（P29-3）", "%.2f",
          doc=(0.61, where), ref=(rt, "e29_tests.csv all8 P29-3"))


# ---------------------------------------------------------------- E30：2 層のパーセプトロン

def _e30(b: Book) -> None:
    p = pd.read_csv(E30 / "e30_pairs.csv", dtype={"K": str})
    tests = pd.read_csv(E30 / "e30_tests.csv")
    prov = json.loads((E30 / "provenance_audio_wavlm.json").read_text())
    reps = prov["reps"]
    where = f"{EX} E30 の結果（N4）"
    doc = {("a", 1e-4): 0.55, ("a", 1e-2): 0.53, ("ah", 1e-4): 0.45, ("ah", 1e-2): 0.37}
    vals = {}
    for k in ("a", "ah", "v", "av"):
        for wd, tag in ((1e-4, "wd4"), (1e-2, "wd2")):
            g = p[(p.kind == KIND[k]) & (p.K == "5") & np.isclose(p.wd, wd)]
            v = float(g.share_pos.median())
            vals[f"{k}-{tag}"] = v
            rt = tests[(tests.kind == KIND[k]) & np.isclose(tests.wd, wd) & (tests.test == "N4")].share_pos_median.iloc[0]
            b.add(f"p1-e30-critshare-k5-{k}-{tag}", v,
                  f"{_rel(E30)}/e30_pairs.csv | kind={KIND[k]}，wd={wd:g}，K=5，反復 {reps[0]}〜{reps[-1]}（{len(reps)} 回），同じコーパスの組を除く {len(g)} 組 | "
                  "2 層のパーセプトロン（隠れ 256）の合併の得に占める位置の分の割合 share_pos の組の中央値",
                  "%.2f", doc=(doc[(k, wd)], where) if (k, wd) in doc else None, ref=(rt, "e30_tests.csv N4"))
    b.rng("p1-e30-critshare-k5-audio", {x: vals[x] for x in ("a-wd4", "a-wd2", "ah-wd4", "ah-wd2")},
          f"{_rel(E30)}/e30_pairs.csv | 音声 2 エンコーダ × wd 2 通り，K=5 | share_pos の中央値", doc=((0.37, 0.55), where))
    b.add("p1-e30-reps-n", len(reps), f"{_rel(E30)}/provenance_audio_wavlm.json | reps | 2 層のパーセプトロンを学んだ反復の数", "%d")


# ---------------------------------------------------------------- E15・E16：C を固定した段階 B

def _e15_e16(b: Book) -> None:
    e15 = pd.read_csv(STAGEB / "e15b_pairs.csv", dtype={"K": str})
    e16 = pd.read_csv(STAGEB / "e16_pairs.csv", dtype={"K": str})
    t15 = pd.read_csv(STAGEB / "e15b_tests.csv").set_index("setting")
    t16 = pd.read_csv(STAGEB / "e16_tests.csv").set_index("setting")
    prov = json.loads((STAGEB / "provenance_merged_scores_audio_wavlm.json").read_text())
    reps = prov["reps"]
    rtxt = f"反復 {reps[0]}〜{reps[-1]}（{len(reps)} 回．C を固定した段階 B の全反復）"
    setting = {"a": "W7", "ah": "Hu7", "v": "V", "av": "AV"}
    where = f"{EX} E17 の「C を固定した段階 B（合併の d・o の実測）と E16」"
    d15 = {"a": 0.87, "ah": 0.91, "v": 0.87, "av": 0.84}
    d16l = {"a": 0.97, "ah": 0.97, "v": 0.96, "av": 0.98}
    d16d = {"a": 0.48, "ah": 0.35, "v": 0.94, "av": 0.85}
    lam15, lam16, dir16 = {}, {}, {}
    for k in ("a", "ah", "v", "av"):
        g = e15[e15.kind == KIND[k]]
        lam15[k] = _spearman(g.lam_obs, g.lam_pred_rep09)
        b.add(f"p1-e15-lam-rho-{k}", lam15[k],
              f"{_rel(STAGEB)}/e15b_pairs.csv | kind={KIND[k]}，組 × K（K=1・5・10）の {len(g)} 行（同じコーパスの組を含む），{rtxt} | "
              "測った λ（lam_obs）と E15 の式の λ（lam_pred_rep09）のスピアマンの順位相関",
              "%.2f", doc=(d15[k], where), why="丸めの差（0.8649 を文書は 0.87 と書いた）" if k == "a" else None,
              ref=(t15.loc[setting[k]].spearman, "e15b_tests.csv spearman"))
        g = e16[e16.kind == KIND[k]]
        lam16[k] = _spearman(g.lam_obs, g.lam_newton)
        dir16[k] = _spearman(g.dc_obs, g.dc_pred)
        src = f"{_rel(STAGEB)}/e16_pairs.csv | kind={KIND[k]}，組 × K（K=1・5・10）の {len(g)} 行，{rtxt} |"
        b.add(f"p1-e16-lam-rho-{k}", lam16[k], src + " 測った λ とニュートン法の 1 歩の λ（lam_newton）のスピアマンの順位相関",
              "%.2f", doc=(d16l[k], where), ref=(t16.loc[setting[k]].spearman_lam, "e16_tests.csv spearman_lam"))
        b.add(f"p1-e16-dir-rho-{k}", dir16[k], src + " 向きの変化 d_m − d_s の実測（dc_obs）と 1 歩の予言（dc_pred）のスピアマンの順位相関",
              "%.2f", doc=(d16d[k], where), ref=(t16.loc[setting[k]].spearman_dc, "e16_tests.csv spearman_dc"))
    s15 = f"{_rel(STAGEB)}/e15b_pairs.csv | 4 種類，組 × K（K=1・5・10），{rtxt} | 種類ごとの順位相関"
    s16 = f"{_rel(STAGEB)}/e16_pairs.csv | 組 × K（K=1・5・10），{rtxt} | 種類ごとの順位相関"
    b.rng("p1-e15-lam-rho", lam15, s15, doc=((0.84, 0.91), "docs/results.md 3.6 の E15"))
    b.rng("p1-e16-lam-rho", lam16, s16 + "（4 種類．λ）", doc=((0.96, 0.98), "docs/results.md 3.6 の E16"))
    b.rng("p1-e16-dir-rho-video", {k: dir16[k] for k in ("v", "av")}, s16 + "（v，av．向き）", doc=((0.85, 0.94), "docs/results.md 3.6 の E16"))
    b.rng("p1-e16-dir-rho-audio", {k: dir16[k] for k in ("a", "ah")}, s16 + "（a，ah．向き）", doc=((0.35, 0.48), "docs/results.md 3.6 の E16"))
    b.add("p1-stageb-reps-n", len(reps), f"{_rel(STAGEB)}/provenance_merged_scores_audio_wavlm.json | reps | 合併の点数を保存した反復の数（E15・E16）", "%d")


# ---------------------------------------------------------------- E32：ソースの価値 n = q e²/(1 − e²)

KC = (1, 5, 10, 25, 50)
BIG = 1e4


def _curves(kind: str, reps) -> dict:
    """ターゲットのみの学習曲線（反復で平均した UAR ×100，K について累積の最大）．tgt_only の行は 2 つの地図で同じ．"""
    out = {}
    for f in sorted((RESULTS / "fixc_map" / kind).glob("*.csv")):
        d = pd.read_csv(f, comment="#", dtype={"K": str, "source": str}, usecols=["method", "K", "rep", "uar"])
        d = d[(d.method == "tgt_only") & d.rep.isin(reps)]
        if d.rep.nunique() != len(reps):
            raise ValueError(f"part1 E32: {kind} {f.stem} のターゲットのみに反復 {reps} がそろっていない")
        u = d.groupby("K").uar.mean() * 100
        out[f.stem] = np.maximum.accumulate(np.array([u[str(k)] for k in KC]))
    return out


def _k_equiv(us, u):
    """UAR u が曲線 us に当たる本数（K の対数で直線補間）．範囲の上なら (nan, 1)，下なら (nan, −1)．"""
    if u > us[-1]:
        return np.nan, 1
    if u < us[0]:
        return np.nan, -1
    lk = np.log(KC)
    for i in range(len(us) - 1):
        if us[i] <= u <= us[i + 1]:
            f = 0.0 if us[i + 1] == us[i] else (u - us[i]) / (us[i + 1] - us[i])
            return float(np.exp(lk[i] + f * (lk[i + 1] - lk[i]))), 0
    return np.nan, 1


def _rankable(v, flag):
    return np.where(flag == 1, BIG, np.where(flag == -1, -BIG, v))


def _n_test(n_meas, flag_meas, n_pred, flag_pred):
    """打ち切りは順位で最大（最小）として扱い，比からは除く（E32 の 4 節の決まり）．"""
    rho = _spearman(_rankable(n_pred, flag_pred), _rankable(n_meas, flag_meas))
    ok = (flag_meas == 0) & np.isfinite(n_pred)
    return rho, float(np.median(n_meas[ok] / n_pred[ok]))


def _e32(b: Book) -> None:
    reps = list(range(10))  # E32 の設計（exp/explore_compare.py の REPS．E31 の段 1 が反復 0〜9）
    P = pd.read_csv(E32 / "e32_pairs.csv", dtype={"K": str})
    A = pd.read_csv(E32 / "e32_allsrc.csv", dtype={"K": str})
    T = pd.read_csv(E32 / "e32_tests.csv", dtype={"K": str}).set_index(["kind", "K"])
    TA = pd.read_csv(E32 / "e32_allsrc_tests.csv", dtype={"K": str}).set_index(["kind", "K"])
    KT = pd.read_csv(E32 / "e32_kinds.csv", dtype={"K": str}).set_index(["kind", "K", "method"])
    where = f"{EX} E32 の 4"
    rdoc = {("a", "5"): (0.90, 0.87), ("a", "10"): (0.88, 0.87), ("ah", "5"): (0.77, 0.86), ("ah", "10"): (0.87, 0.79),
            ("v", "5"): (0.96, 1.34), ("v", "10"): (0.93, 0.69), ("av", "5"): (0.94, 1.02), ("av", "10"): (0.88, 0.82)}
    adoc = {"a": (0.88, 1.15), "ah": (0.78, 0.65), "av": (0.87, 1.01), "v": (0.75, 1.15)}
    n1doc = {"a": (8.5, 10.3, 11.8), "ah": (8.7, 11.1, 11.5), "v": (5.9, 6.8, 6.6), "av": (7.0, 8.6, 9.4)}
    nadoc = {"a": (20.2, 28.9), "ah": (22.5, 30.5), "v": (10.7, 13.7), "av": (18.9, 25.4)}
    curve_src = "学習曲線は results/fixc_map/<kind>/*.csv の tgt_only（反復 0〜9 の平均，K=1・5・10・25・50 を K の対数で直線補間，累積の最大．tgt_only は 2 つの地図で同じ）"
    rho_audio, ratio_audio, aratio, n1_audio, na_audio = {}, {}, {}, {}, {}
    for k in ("a", "ah", "v", "av"):
        kind = KIND[k]
        cur = _curves(kind, reps)
        g = P[P.kind == kind].copy()
        res = [_k_equiv(cur[t], u) for t, u in zip(g.target, g.uar_P1)]
        g["n2"] = [kp - int(K) if fl == 0 else np.nan for (kp, fl), K in zip(res, g.K)]
        g["f2"] = [fl for _, fl in res]
        bad = (g.f2 != g.flag_meas).sum() + int(np.nanmax(np.abs(g.n2 - g.n_meas)) > 1e-6)
        chk = None if bad == 0 else "学習曲線から読み直した測った n が e32_pairs.csv の n_meas と食い違う"
        for K in ("5", "10"):
            h = g[g.K == K]
            rho, ratio = _n_test(h.n2.to_numpy(), h.f2.to_numpy(), h.n_pred.to_numpy(), h.flag_pred.to_numpy())
            src = (f"{_rel(E32)}/e32_pairs.csv（uar_P1：1 ソースの仮説転移 P1（C′=0.01）の UAR，n_pred：E13 の感情の対ごとの e と E28-9 の q_dir から模型で予言した本数）| "
                   f"kind={kind}，K={K}，同じコーパスの組を除く {len(h)} 組，反復 0〜9（10 回．E31 の段 1）| 測った n = 学習曲線で uar_P1 に当たる本数 − K．{curve_src}．"
                   "打ち切りは順位で最大として扱い比から除く")
            dr, dq = rdoc[(k, K)]
            rt = T.loc[(kind, K)]
            b.add(f"p1-e32-rho-{k}-k{K}", rho, src + "．予言と測った n のスピアマンの順位相関", "%.2f", doc=(dr, where),
                  ref=(rt.rho_n_pred, "e32_tests.csv rho_n_pred"), note=chk)
            b.add(f"p1-e32-ratio-{k}-k{K}", ratio, src + "．測った n ÷ 予言 の組の中央値", "%.2f", doc=(dq, where),
                  ref=(rt.ratio_n_pred, "e32_tests.csv ratio_n_pred"), note=chk)
            if k in ("a", "ah"):
                rho_audio[f"{k}-k{K}"], ratio_audio[f"{k}-k{K}"] = rho, ratio
        # 全ソース（P4）：ターゲット × K（K=5・10）
        a = A[(A.kind == kind) & A.K.isin(["5", "10"])].copy()
        res = [_k_equiv(cur[t], u) for t, u in zip(a.target, a.P4)]
        a["n2"] = [kp - int(K) if fl == 0 else np.nan for (kp, fl), K in zip(res, a.K)]
        a["f2"] = [fl for _, fl in res]
        bad = (a.f2 != a.flag).sum() + int(np.nanmax(np.abs(a.n2 - a.n_meas)) > 1e-6)
        chk = None if bad == 0 else "学習曲線から読み直した測った n が e32_allsrc.csv の n_meas と食い違う"
        rho, ratio = _n_test(a.n2.to_numpy(), a.f2.to_numpy(), a.n_pred.to_numpy(), a.flag_pred.to_numpy())
        aratio[k] = ratio
        src = (f"{_rel(E32)}/e32_allsrc.csv（P4：全ソースの下書きからの仮説転移の UAR，n_pred：全ソースの感情の対ごとの e で同じ模型から予言）| "
               f"kind={kind}，ターゲット × K（K=5・10）の {len(a)} 行，反復 0〜9 | 測った n = 学習曲線で P4 に当たる本数 − K．{curve_src}．打ち切りは順位で最大，比から除く")
        rt = TA.loc[(kind, "5+10")]
        b.add(f"p1-e32-allsrc-rho-{k}", rho, src + "．予言と測った n の順位相関（V3）", "%.2f", doc=(adoc[k][0], where + " の V3"),
              ref=(rt.rho_n_pred, "e32_allsrc_tests.csv rho_n_pred"), note=chk)
        b.add(f"p1-e32-allsrc-ratio-{k}", ratio, src + "．測った n ÷ 予言 の中央値（V3）", "%.2f", doc=(adoc[k][1], where + " の V3"),
              ref=(rt.ratio_n_pred, "e32_allsrc_tests.csv ratio_n_pred"), note=chk)
        # 種類の平均：組の平均の UAR を組の平均の学習曲線で読む（同じコーパスの組を除く）
        g5 = g[g.K == "5"]
        w = g5.target.value_counts()
        us = np.mean([cur[t] for t in g5.target], axis=0)
        ksrc = (f"{_rel(E32)}/e32_pairs.csv・e32_allsrc.csv | kind={kind}，同じコーパスの組を除く {len(g5)} 組，反復 0〜9 | "
                f"組の平均の UAR（P4 はターゲットごとの値を組の数で重みづけて平均）を，組の平均の学習曲線（{curve_src}）で読んだ本数 − K（感情ごとの本数）")
        for i, K in enumerate(("5", "10", "25")):
            u1 = float(g[g.K == K].uar_P1.mean())
            kp, fl = _k_equiv(us, u1)
            n1 = kp - int(K) if fl == 0 else np.nan
            ref = KT.loc[(kind, K, "P1_def")]
            b.add(f"p1-e32-n1src-{k}-k{K}", n1, ksrc + f"．1 ソースの仮説転移 P1，K={K}", doc=(n1doc[k][i], f"{EX} E32 の 2"),
                  ref=(ref.n, "e32_kinds.csv P1_def n"))
            if k in ("a", "ah"):
                n1_audio[f"{k}-k{K}"] = n1
        for i, K in enumerate(("5", "10")):
            x = A[(A.kind == kind) & (A.K == K)].set_index("target").P4
            u4 = float(sum(x[t] * w[t] for t in w.index) / w.sum())
            kp, fl = _k_equiv(us, u4)
            n4 = kp - int(K) if fl == 0 else np.nan
            ref = KT.loc[(kind, K, "P4_def")]
            b.add(f"p1-e32-nall-{k}-k{K}", n4, ksrc + f"．全ソースの下書きからの仮説転移 P4，K={K}", doc=(nadoc[k][i], f"{EX} E32 の 2"),
                  ref=(ref.n, "e32_kinds.csv P4_def n"))
            if k in ("a", "ah"):
                na_audio[f"{k}-k{K}"] = n4
    s = f"{_rel(E32)}/e32_pairs.csv | 音声 2 エンコーダ（a，ah）× K=5・10，反復 0〜9 | 種類 × K ごとの値"
    b.rng("p1-e32-rho-audio", rho_audio, s + "．予言と測った n の順位相関", doc=((0.77, 0.90), where))
    b.rng("p1-e32-ratio-audio", ratio_audio, s + "．測った n ÷ 予言 の中央値", doc=((0.79, 0.87), where))
    b.rng("p1-e32-allsrc-ratio", aratio, f"{_rel(E32)}/e32_allsrc.csv | 4 種類（a，ah，v，av），K=5・10 | 種類ごとの 測った n ÷ 予言 の中央値",
          doc=((0.65, 1.15), where + " の V3"))
    b.rng("p1-e32-n1src-audio", n1_audio, f"{_rel(E32)}/e32_pairs.csv | 音声 2 エンコーダ × K=5・10・25 | 種類の平均の 1 ソースの価値（感情ごとの本数）",
          "%.1f", doc=((8.5, 11.8), f"{EX} E32 の 2（「約 10 本」）"))
    b.rng("p1-e32-nall-audio", na_audio, f"{_rel(E32)}/e32_allsrc.csv | 音声 2 エンコーダ × K=5・10 | 種類の平均の全ソースの価値（感情ごとの本数）",
          "%.1f", doc=((20.2, 30.5), f"{EX} E32 の 2（「20〜30 本」）"))


# ---------------------------------------------------------------- まとめ

def entries() -> dict[str, dict]:
    b = Book()
    _e12(b)
    _e13(b)
    _e28t9(b)
    _e29(b)
    _e30(b)
    _e15_e16(b)
    _e32(b)
    return b.out
