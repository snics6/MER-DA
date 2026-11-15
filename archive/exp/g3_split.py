"""【本番の感度】E81（GPU）：テストと同じセッション（IEMOCAP）・対話（MELD）を候補から除いた分割（G3）で，筋 5 の比べ（P5-1〜P5-3）に使う手法を回し直す。

設計の正本は `docs/exploratory-analyses.md` の E81．台本の調べは `checks/E81/step7_facts.md`．E77（`compare_dai.py`）・E78（`compare_dai2.py`）・
E80 段 c（`e80_stage_c.py`）の関数をそのまま呼び，分割（候補・K 本・中心）だけを替える．この 3 つの台本と `e80_teacher_j.py`・`analyze_e79.py` は書き換えない。

分割（`G3Ctx`）：テストは今と同じ（`map.test_mask`）．塊は IEMOCAP がセッション（発話一覧の `session`），MELD が `split|dialogue_id`．
  学習側 ＝ テスト以外の発話から，テストの発話を 1 本でも含む塊の発話を除いたもの．K 本 ＝ 今の感情ごとの並び（`map.kshot_order`．今の学習側で種 r）から
  除いた発話を抜き，先頭の K 本．生徒の中心・尺度・白色化，出発点 pF，none，教師はすべて新しい学習側で作り直す（`make_students`・`unit_pass1` と同じ作り方）．
  ソースの側は変えない．`--no-exclude` は除かない（今の分割と同じ単位．台本の正しさの検算に使う）。
手法：none（全生徒 × K）と，E79 の段 (b) が仮のターゲットで選んだ設定（`--chosen` の chosen.csv の `--variant` の行．既定は incl の oursx・kd・groupB）．選び直さない．
  m_eff ＝ n は小さくなった候補の本数で解く．entmin・im は種 0（none の解から）と `--nc-seeds`（揺らした出発点．E77 と同じ），
  dann・mme・mapctl は種 0〜2（E78 と同じ）．束の中身は本番にそろえる：none・ref・ref0・CORAL・DANN 等は `--batch-ks`（本番の --ks 0 5 25）で束ね，
  self・entmin・IM は E77 の run_ab と同じ全格子の束で解いて選んだ行だけを残す．DANN・MME・mapctl は E78 の run_map と同じ 30 問（12 問）の束を
  そのまま解く（束の中身で値が変わるため．種 1・2 の束は chosen.csv の excl の行＝E78 の `_chosen.csv` から作る）．
  教師：E77 の波の教師（pF・pFK・ref・ref0・poe0・ownK）は E77 と同じ float64 の配列，
  E80 段 c の波の教師は `unit_pass1` の float32 の配列（段 c と同じ）．poolC・mlp（agg_T）は T の仮のターゲットの全単位（分割は変えない）の生徒の出発点から，
  段 c と同じ λ の選び方と種 crc32(world|T|eval) で学び直し，`--e80c-dir` の `<T>_pool.csv` の値と照合する（あれば．食い違えば止まる）。
照合（台本の中）：(a) IEMOCAP 台本の反復 3・8（AVT・AV）は学習側が変わらないこと（候補と K 本が今と同じ），(b) `label_sets_of` が通ること，
  (c) agg_T が段 c と同じこと，(d) 学習側の大きさが `--counts`（checks/E81/g3_counts_all.csv）と同じこと，(e) chosen.csv の oursx・kd の教師が `--kept` で残した教師であること。

出力（`--out`．既定 results/e81_<世界>/．E77 と同じ形）：
  `<T>.csv`（run, target, rep, row_id, student, method, param, meff, K, uar, wave, selectors, pnorm, seed, split, n_pool, commit．最後に書く＝再開の印），
  `<T>_r<r>_logp.npy`（行 × テストの発話 × 4 の float32 の対数の確率．設計の「float16 の確率」は小さい確率が 0 になり向きの損が出せないので E77 と同じ形にした），`_pred.npy`（uint8），`_rowids.npy`，`_te.npy`，
  `_pool.npy`（学習側＝候補の行番号），`_excl.npy`（除いた行番号），`_kshot.npz`（K → K 本の行番号），`_teachers.npz`（使った教師の候補の上の確率．「波|教師__K<K>」），
  `_W_<生徒>.npy`・`_b_<生徒>.npy`・`_wrows_<生徒>.npy`（none を含む全行の生の特徴の上の W・b），
  `<T>_units.csv`（反復ごとの学習側の大きさと所要），`<T>_plan.csv`（使った chosen.csv の行），`<T>_pool.csv`（agg_T の λ・種・損失と段 c との照合），
  `<T>_conv.csv`・`<T>_convp.csv`・`<T>_diag.csv`，`<T>_samples.csv`，`<T>_settings.json`，provenance.json・provenance_log.jsonl
  UAR・向きの損・位置の損と区間は `exp/analyze_e81.py` が E79 の関数で出す．この台本は値を画面に出さない（行の数と所要だけ）。

    ./.venv/bin/python exp/g3_split.py --world fuse-audio-video-text --chosen results/e79/select/chosen.csv [--targets …] [--reps …] [--ks 0 5]
    ./.venv/bin/python exp/g3_split.py --world fuse-audio-video --chosen … --check          # 分割の照合だけ（解かない）
  H100 の本番は --gpu-fraction 0.3（E80 段 c と同じ）．試しは --allow-dirty と小さい --gpu-fraction
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import baselines_distill as BD  # noqa: E402
import committee as CM  # noqa: E402
import compare_b1 as CB  # noqa: E402
import compare_dai as CD  # noqa: E402
import compare_dai2 as E78  # noqa: E402
import e80_stage_c as ES  # noqa: E402
import e80_teacher_j as EA  # noqa: E402
import logreg  # noqa: E402
import map as M  # noqa: E402
import meta5 as M5  # noqa: E402
from baselines_distill import hard_rows, mass, temp_rows, tens  # noqa: E402
from distill_belief import weak_rows  # noqa: E402
from geom_e0 import settings  # noqa: E402
from multimodal_distill import WORLDS, beliefs1, blocks, make_students, sub_group  # noqa: E402
from split import speaker_folds  # noqa: E402
from teach_student39 import Student  # noqa: E402
from data import groups_for  # noqa: E402

DT, DEV = logreg.DTYPE, logreg.DEVICE
C_FIT = CD.C_FIT
E81_WORLDS = ("fuse-audio-video-text", "fuse-audio-video")
E81_TARGETS = ("IEMOCAP-impro", "IEMOCAP-script", "MELD")
BLOCK_COLS = {"IEMOCAP": ("iemocap_utterance_manifest.csv", ("session",)),      # analyze_e79.CLUSTER_COLS と同じ塊
              "MELD": ("meld_utterance_manifest.csv", ("split", "dialogue_id"))}
SAME_SPLIT = {("IEMOCAP-script", 3), ("IEMOCAP-script", 8)}                     # AVT・AV でテストの 2 人が同じセッション（学習側が変わらない単位．事実の調べ 2.3 節）
EMPTY = "__none__"
SEEDED_NC = ("entmin", "im")                                                 # E77 の非凸の手法（種 0 は末尾の s なし）
SEEDED_MAP = ("dann", "mme", "mapctl")                                       # E78 の確率的な勾配法（param の末尾 s<種>）
SUPPORTED = ("none", "dist", "kd", "kdh", "entmin", "im", "self", "t3a", "lp", "coral", "dann", "mme", "mapctl")
E77_TEACH = {"mod": {"dist": ("pFK", "ref", "pF", "ownK"), "kd": ("pF", "pFK", "ref")},          # compare_dai.world_unit の run_c に渡す教師
             "fused": {"dist": ("ref", "ref0", "poe0", "ownK"), "kd": ("poe0", "ref")}}
AGG = ES.AGG                                                                 # ("poolC", "mlp")
AGG_TOL = 1e-9                                                               # agg_T の MLP の損失の相対の差の許容幅（段 c との照合）
UNCHECKED: list = []                                                         # --allow-incomplete で飛ばした照合（provenance の unchecked．analyze_e81 は結論を書かない）
REF_METHODS = ("ref_srconly", "ref_merged", "ref_tgtonly", "ref_wstar")


def skip_or_stop(msg, allow, notes):
    """照合の入力が無いとき：試しの印（--allow-incomplete）が無ければ止める．あれば記録して進む（gate-code の I8・I14）。"""
    if not allow:
        raise SystemExit(f"{msg}（照合を飛ばさない．試しだけ --allow-incomplete）")
    notes.append(msg)
    UNCHECKED.append(msg)


# ---------------------------------------------------------------- 道具

def sha_file(f):
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def loaded_code():
    """読み込んだ ROOT の下の .py（この台本を含む）→ sha256．provenance と dirty の判定に使う。"""
    fs = {Path(__file__).resolve()}
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if not (isinstance(f, str) and f.endswith(".py") and Path(f).is_absolute() and Path(f).exists()):
            continue
        f = Path(f).resolve()
        if any(f.is_relative_to(ROOT / d) for d in ("exp", "measure", "features")):
            fs.add(f)
    return {str(f.relative_to(ROOT)): sha_file(f) for f in sorted(fs)}


def dirty_code(files):
    """読み込んだコードのうち，未コミットの変更があるか追跡していないもの。"""
    bad = [ln for ln in git("status", "--porcelain", "--", *files).splitlines() if ln.strip()]
    tracked = set(git("ls-files", "--", *files).splitlines())
    bad += [f"?? {f}" for f in files if f not in tracked]
    return sorted(set(bad))


def block_labels(T, sample_ids):
    """発話ごとの塊（IEMOCAP はセッション，MELD は split|dialogue_id）．analyze_e79.cluster_labels と同じ作り方．欠けがあれば止まる。"""
    cp = CM.corp(T)
    fn, cols = BLOCK_COLS[cp]
    m = pd.read_csv(ROOT / "data" / "metadata" / fn, usecols=["sample_id", *cols], dtype=str).drop_duplicates("sample_id").set_index("sample_id")
    lab = m.reindex(pd.Index(sample_ids, dtype=str))[list(cols)]
    if lab.isna().any().any():
        raise SystemExit(f"{T}：発話一覧に塊の無い発話が {int(lab.isna().any(axis=1).sum())} 本ある")
    return lab.astype(str).agg("|".join, axis=1).to_numpy()


# ---------------------------------------------------------------- 分割

class G3Ctx(CB.Ctx):
    """compare_b1.Ctx と同じテスト・K 本の並びから，テストと同じ塊の発話を学習側から除いた分割．pool[r] が候補（学習側）の行番号。
    exclude=False なら今の分割と同じ（pool[r] ＝ テスト以外の全部，K 本・中心も今と同じ）。"""

    def __init__(self, tgt, reps, ks, blocks_, exclude=True):
        super().__init__(tgt, reps, ks, center="ind")                           # te・persons・今の K 本（検算に残す）
        self.exclude = exclude
        self.pool, self.excl, self.cur_idx, self.stats = {}, {}, {}, {}
        for r in self.reps:
            te = self.te[r]
            cur = np.flatnonzero(~te)
            for K in self.ks:
                self.cur_idx[(r, K)] = self.idx[(r, K)].copy()
            ex = np.isin(blocks_, np.unique(blocks_[te])) & ~te if exclude else np.zeros(len(te), bool)
            keep = ~te & ~ex
            pool = np.flatnonzero(keep)
            order = M.kshot_order(tgt, cur, r)                                    # 今の並び（同じ乱数の流れ）から除いた発話を抜く
            order = [o[keep[o]] for o in order]
            for K in self.ks:
                idx = M.take_k(order, K)
                self.idx[(r, K)] = idx
                f = speaker_folds(pd.Series(tgt.person[idx]), k=3, seed=r).to_numpy()   # compare_b1.Ctx と同じ（世界の台本は使わない）
                ok = all(len(np.unique(tgt.y[idx][f != j])) == 4 for j in np.unique(f)) and len(np.unique(f)) >= 2
                self.cvf[(r, K)] = f if ok else None
            self.ct[r] = M.rows(tgt.X, pool).mean(0)
            self.pool[r], self.excl[r] = pool, np.flatnonzero(ex)
            cur5 = self.cur_idx.get((r, "5"))
            cur25 = self.cur_idx.get((r, "25"))
            self.stats[r] = {"n_test": int(te.sum()), "n_pool_cur": len(cur), "n_excl": int(ex.sum()), "n_pool": len(pool),
                             "pool_min_class": int(min((tgt.y[pool] == c).sum() for c in range(4))),
                             "k5_in_excl": int(ex[cur5].sum()) if cur5 is not None else -1, "k25_in_excl": int(ex[cur25].sum()) if cur25 is not None else -1,
                             "n_test_blocks": int(len(np.unique(blocks_[te])))}
            if not ex.any():                                                      # 照合 (a)：除くものが無ければ今の分割と同じ
                assert np.array_equal(pool, cur) and all(np.array_equal(self.idx[(r, K)], self.cur_idx[(r, K)]) for K in self.ks)


def check_counts(ctx, world, T, counts, exclude, notes, allow):
    """照合 (d)：学習側の大きさが g3_counts_all.csv と同じか（構造の数だけ）。"""
    if counts is None:
        skip_or_stop(f"{world} {T}：counts（g3_counts_all.csv）が無いので学習側の大きさを照合できない", allow, notes)
        return
    c = counts[(counts.run == world) & (counts["T"] == T)].set_index("rep")
    for r in ctx.reps:
        s = ctx.stats[r]
        if r not in c.index:
            raise SystemExit(f"{world} {T} r{r}：{T} の行が counts に無い")
        e = c.loc[r]
        want = {"n_test": e.n_test, "n_pool_cur": e.n_pool}
        if exclude:
            want.update({"n_excl": e.n_pool_excluded_block, "n_pool": e.n_pool_after, "pool_min_class": e.pool_after_min_class,
                         "k5_in_excl": e.K5_in_excluded, "k25_in_excl": e.K25_in_excluded, "n_test_blocks": e.n_test_blocks})
        bad = {k: (s[k], int(v)) for k, v in want.items() if s[k] != int(v)}
        if bad:
            raise SystemExit(f"{world} {T} r{r}：学習側の大きさが counts と違う {bad}")


# ---------------------------------------------------------------- chosen.csv

def read_chosen(path, world, targets, ks, selectors, variant, allow_incomplete, notes):
    """(T, 生徒, K) → {(波, 方法, pnorm, m_eff): 選んだ族の集合}．none を選んだ行は none（足す行は無い）．候補の無い行（__none__）は記録して飛ばす。"""
    d = pd.read_csv(path, dtype={"K": str, "param": str, "meff": str, "wave": str, "method": str, "target": str, "student": str})
    need = {"run", "target", "student", "K", "selector", "variant", "wave", "method", "param", "meff"}
    if not need <= set(d.columns):
        raise SystemExit(f"{path} に列 {sorted(need - set(d.columns))} が無い")
    d = d[(d.run == world) & d.target.isin(targets) & d.K.isin(ks) & d.selector.isin(selectors) & (d.variant == variant)].copy()
    if d.duplicated(["target", "student", "K", "selector"]).any():
        raise SystemExit(f"{path}：（T, 生徒, K, 族）の重複がある")
    plan, used = {}, []
    studs = [world] + WORLDS[world]
    for T in targets:
        for st in studs:
            for K in ks:
                plan[(T, st, K)] = {}
                for sel in selectors:
                    x = d[(d.target == T) & (d.student == st) & (d.K == K) & (d.selector == sel)]
                    if x.empty:
                        msg = f"chosen に {world} {T} {st} K={K} {sel} {variant} の行が無い"
                        if not allow_incomplete:
                            raise SystemExit(msg)
                        notes.append(msg)
                        continue
                    r = x.iloc[0]
                    base = {"target": T, "student": st, "K": K, "selector": sel, "variant": variant}
                    if pd.isna(r.method) or r.method == EMPTY:
                        used.append({**base, "wave": None, "method": EMPTY, "pnorm": None, "meff": None, "status": "no_candidate"})
                        notes.append(f"{T} {st} K={K} {sel}：候補が無い（__none__）")
                        continue
                    if r.method not in SUPPORTED:
                        raise SystemExit(f"{T} {st} K={K} {sel}：手法 {r.method} はこの台本で解けない")
                    if r.method == "none":
                        used.append({**base, "wave": "e77", "method": "none", "pnorm": "-", "meff": "0", "status": "none"})
                        continue
                    key = (str(r.wave), str(r.method), str(r.param), str(r.meff))
                    plan[(T, st, K)].setdefault(key, set()).add(sel)
                    used.append({**base, "wave": key[0], "method": key[1], "pnorm": key[2], "meff": key[3], "status": "solve"})
    return plan, pd.DataFrame(used)


def read_chosen78(path, world, targets, bks, methods, plan, notes, allow):
    """E78 が評価のターゲットで種 1・2 を解いた設定（E78 の `<T>_chosen.csv`．E79 は excl の選択がそれと同じことを確かめて止める：
    analyze_e79 の choose_seeded_match）を，chosen.csv の excl の行（族 dann・mme・mapctl）から作る．T → {(生徒, 方法, K): {param}}．
    行が無い（試しの chosen.csv など）ときは，選んだ設定だけを足して記録する（束の中身が本番と違いうる）。"""
    d = pd.read_csv(path, dtype={"K": str, "param": str, "meff": str, "method": str, "target": str, "student": str})
    d = d[(d.run == world) & d.target.isin(targets) & d.K.isin(bks) & d.selector.isin(methods) & (d.variant == "excl")
          & (d.method == d.selector)]
    out = {T: {} for T in targets}
    for r in d.itertuples(index=False):
        out[r.target].setdefault((r.student, r.method, r.K), set()).add(r.param)
    for (T, st, K), sets_ in plan.items():
        for (wave, meth, pn, meff) in sets_:
            if meth in SEEDED_MAP and pn not in out[T].get((st, meth, K), set()):
                skip_or_stop(f"{T} {st} K={K} {meth} {pn}：chosen.csv の excl の行に無い（種 1・2 の束が E78 と違いうる）", allow, notes)
                out[T].setdefault((st, meth, K), set()).add(pn)
    return out


def check_kept(plan, kept_path, world, notes, allow):
    """照合 (e)：oursx・kd の教師（dist・kd・kdh の行）が kept.csv で残した教師か（台本の行が無ければ即興の行．E79 の load_kept と同じ）．
    戻りは (kept.csv の sha256, (T, 生徒, K) → 残した教師の集合)．後者は kept の教師の全部の配列の保存に使う。"""
    if kept_path is None or not Path(kept_path).exists():
        skip_or_stop(f"kept.csv（{kept_path}）が無いので選んだ教師を照合できない", allow, notes)
        return None, {}
    kw = ES.read_kept(kept_path, world)
    kept = {(T, st, K): set(g.teacher[g.keep]) for (T, st, K), g in kw.groupby(["target", "student", "K"])}
    for (T, st, K), sets_ in plan.items():
        for (wave, meth, pn, meff), sels in sets_.items():
            if meth not in ("dist", "kd", "kdh") or not ({"oursx", "kd"} & sels):
                continue
            name = pn.split(":")[0] if meth == "dist" else re.sub(r"T\d+$", "", pn)
            k_ = kept.get((T, st, K))
            if k_ is None and T in ES.SCRIPT_SRC:
                k_ = kept.get((ES.SCRIPT_SRC[T], st, K))
            if k_ is None or name not in k_:
                raise SystemExit(f"{T} {st} K={K}：選んだ教師 {name}（{meth}）が kept で残した教師に無い")
    return sha_file(kept_path), kept


def kept_of(kept, T, st, K):
    """(T, 生徒, K) で残した教師（台本の行が無ければ即興の行）。"""
    k_ = kept.get((T, st, K))
    if k_ is None and T in ES.SCRIPT_SRC:
        k_ = kept.get((ES.SCRIPT_SRC[T], st, K))
    return sorted(k_ or ())


# ---------------------------------------------------------------- 1 回目（E80 段 a の unit_pass1 と同じ．float64 の教師も返す）

def unit_pass1x(world, stu, yq, yte, label_sets, ks):
    """`e80_teacher_j.unit_pass1` の写し（同じ順・同じ束ね方）．戻りは (u：段 c と同じ float32 の配列, prs, x：E77 の world_unit と同じ float64 の教師と none)。
    u が unit_pass1 とビット単位で同じことは試しで確かめた（checks/E81/impl.md）。"""
    mods = WORLDS[world]
    prs = {k: CD.Prob2(stu[k], yq, label_sets) for k in mods + [world]}
    none = {k: prs[k].solve([(K, None, K) for K in ks], what="none") for k in [world] + mods}
    P0 = {k: {K: prs[k].probs(none[k][K], stu[k]["Z"]) for K in ks} for k in none}
    P0_f = P0[world]
    pF = tens(stu[world]["P"])
    b1 = beliefs1(stu, world)
    lg, lg0 = {K: 0 for K in ks}, {K: 0 for K in ks}
    lt, lt0, lpoe_t = {K: 0 for K in ks}, {K: 0 for K in ks}, 0                  # テストの上の教師（E77 の world_unit の teacher の行と同じ）
    for m in mods:
        sol = prs[m].solve([(("ref", K), weak_rows(P0_f[K], CD.REF_MIN), K) for K in ks] + [(("ref0", K), weak_rows(pF, CD.REF_MIN), K) for K in ks],
                           what="ref")
        Zt_m = prs[m].s["Zte"]
        lpoe_t = lpoe_t + torch.log_softmax(Zt_m @ prs[m].W0.T + prs[m].b0, 1)
        for K in ks:
            lg[K] = lg[K] + prs[m].probs(sol[("ref", K)], prs[m].s["Z"]).clamp(min=1e-12).log()
            lg0[K] = lg0[K] + prs[m].probs(sol[("ref0", K)], prs[m].s["Z"]).clamp(min=1e-12).log()
            lt[K] = lt[K] + torch.log_softmax(Zt_m @ sol[("ref", K)][0].T + sol[("ref", K)][1], 1)
            lt0[K] = lt0[K] + torch.log_softmax(Zt_m @ sol[("ref0", K)][0].T + sol[("ref0", K)][1], 1)
    votes = np.stack([stu[m]["P"].argmax(1) for m in mods], 1)
    post, pi, Cs = EA.dawid_skene(votes)
    ref = {K: torch.softmax(lg[K], 1) for K in ks}
    ref0 = {K: torch.softmax(lg0[K], 1) for K in ks}
    f32 = EA.f32
    u = {"y": np.asarray(yq).astype(np.int64), "yte": np.asarray(yte).astype(np.int64), "lab": {K: np.asarray(label_sets[K], int) for K in ks},
         "P": {k: f32(stu[k]["P"]) for k in mods + [world]},
         "none": {k: {K: f32(P0[k][K]) for K in ks} for k in P0},
         "none_te": {k: {K: f32(torch.log_softmax(stu[k]["Zte"] @ none[k][K][0].T + none[k][K][1], 1)) for K in ks} for k in none},
         "poe0": f32(b1["poe"]), "agree": f32(b1["agree"]), "ds": f32(post), "ds_pi": pi, "ds_C": np.stack(Cs),
         "ref": {K: f32(ref[K]) for K in ks}, "ref0": {K: f32(ref0[K]) for K in ks}}
    x = {"none": none, "P0": P0, "pF": pF, "poe0": tens(b1["poe"]), "ref": ref, "ref0": ref0, "lt": lt, "lt0": lt0, "lpoe_t": lpoe_t}
    return u, prs, x


# ---------------------------------------------------------------- agg_T（poolC・mlp）：T の仮のターゲットの全単位から

def agg_units(world, T, data_f, data_m, S, sets, pseudo_reps, cache):
    """e80_stage_c.pseudo_pass_a と同じ単位（c ∈ S，c と同じコーパスでないソース，反復 pseudo_reps，今の分割）．poolC・mlp に要る
    生徒の出発点（float32）・ラベル・人物だけを作る（unit_pass1 の u["P"]・u["y"]・u["person"] と同じ値）。"""
    mods = WORLDS[world]
    data_fT = {g: v for g, v in data_f.items() if CM.corp(g) != CM.corp(T)}
    data_mT = {k: {g: v for g, v in d.items() if CM.corp(g) != CM.corp(T)} for k, d in data_m.items()}
    units = {}
    for c in S:
        pool_c = [s for s in S if CM.corp(s) != CM.corp(c)]
        if not pool_c:
            continue
        assert CM.corp(c) != CM.corp(T) and all(CM.corp(s) != CM.corp(T) for s in pool_c)
        info_c = {k: M5.pool_info(data_mT[k] if k != world else data_fT, pool_c, C_FIT, cache[k]) for k in sets}
        evc = data_fT[c]
        ctxc = CB.Ctx(evc, pseudo_reps, list(EA.KS_LAB_ALL), center="ind")
        for r in pseudo_reps:
            t0 = time.perf_counter()
            pool, te = np.flatnonzero(~ctxc.te[r]), np.flatnonzero(ctxc.te[r])
            stu = make_students(world, c, data_fT, data_mT, pool_c, pool, te, sets, info_c)
            units[(c, r)] = {"P": {k: EA.f32(stu[k]["P"]) for k in mods + [world]}, "y": np.asarray(evc.y[pool]).astype(np.int64),
                             "person": np.asarray(evc.person)[pool]}
            del stu
            torch.cuda.empty_cache()
            print(f"  {world} {T} 仮 {c} r{r}：生徒の出発点（候補 {len(pool)}）{time.perf_counter() - t0:.0f} 秒", flush=True)
    return units


def check_agg(rows, e80c_dir, T, notes, allow):
    """照合 (c)：agg_T の λ・選び方・単位の数・MLP の種と損失が段 c の `<T>_pool.csv`（pseudo ＝ *eval*，what ＝ final）と同じか。"""
    fin = [x for x in rows if x.get("what") == "final"]
    assert len(fin) == 1
    me = fin[0]
    f = None if e80c_dir is None else Path(e80c_dir) / f"{T}_pool.csv"
    if f is None or not f.exists():
        skip_or_stop(f"{T}：段 c の {f} が無いので agg_T を照合できない", allow, notes)
        return {"check": "agg_vs_e80c", "status": "not_checked"}
    e = pd.read_csv(f)
    e = e[(e.pseudo.astype(str) == "*eval*") & (e.what == "final")]
    if len(e) != 1:
        raise SystemExit(f"{f}：*eval* の final の行が {len(e)} 行")
    e = e.iloc[0]
    rel = abs(me["mlp_loss"] - float(e.mlp_loss)) / max(1.0, abs(float(e.mlp_loss)))
    same = {"lam": float(me["lam"]) == float(e.lam), "mode": me["mode"] == e["mode"], "n_units": int(me["n_units"]) == int(e.n_units),
            "n_rows": int(me["n_rows"]) == int(e.n_rows), "mlp_seed": int(me["mlp_seed"]) == int(e.mlp_seed), "members": me["members"] == e.members,
            "mlp_loss": rel <= AGG_TOL}
    if not all(same.values()):
        raise SystemExit(f"{T}：agg_T が段 c と違う {[k for k, v in same.items() if not v]}（損失の相対の差 {rel:.3g}）")
    return {"check": "agg_vs_e80c", "status": "ok", "mlp_loss_rel_diff": rel, "file": str(f), "file_sha256": sha_file(f)}


# ---------------------------------------------------------------- 教師

def role(world, k):
    return "fused" if k == world else "mod"


class Teachers:
    """1 つの評価の単位の教師．波 e77 は E77 の world_unit と同じ float64 の配列，波 e80c は段 c の solve_unit と同じ（float32 → float64）。"""

    def __init__(self, world, u, x, agg):
        self.world, self.u, self.x = world, u, x
        self.mods = WORLDS[world]
        self.qC = self.qM = None
        if agg is not None:
            self.qC, self.qM = ES.agg_pred(agg, EA.feat(u, self.mods))
        self.used = {}

    def get(self, wave, fam, k, name, K):
        rl = role(self.world, k)
        if wave == "e77":
            if name not in E77_TEACH[rl][fam]:
                raise SystemExit(f"E77 の {fam} の教師に {name} は無い（生徒 {k}）")
            x = self.x
            d = {"pF": x["pF"], "pFK": x["P0"][self.world], "ref": x["ref"], "ref0": x["ref0"], "poe0": x["poe0"], "ownK": x["P0"][k]}
            P = CD.tk(d[name], K)
        elif wave == "e80c":
            if name not in ES.r1_names(self.world, k):
                raise SystemExit(f"教師 {name} は生徒 {k} の R1 の候補に無い")
            e77 = (ES.E77_DIST if fam == "dist" else ES.E77_KD)[rl]
            if name in e77:
                raise SystemExit(f"教師 {name} は E77 の {fam} の教師（波 e77 のはず）")
            if name in AGG and self.qC is None:
                raise SystemExit(f"教師 {name} に agg_T が要る")
            P = tens(ES.teacher_array(self.world, self.u, name, K, self.qC, self.qM))
        else:
            raise SystemExit(f"教師の波 {wave} は無い")
        self.used[f"{wave}__{name}__K{K}"] = P.detach().cpu().numpy() if torch.is_tensor(P) else np.asarray(P)
        return P

    def keep_array(self, k, name, K):
        """kept の教師の配列を保存に足す（波は E77 の教師の集合にあれば e77，無ければ e80c．gate-code の R9）．agg_T が無い poolC・mlp は False。"""
        rl = role(self.world, k)
        if name in E77_TEACH[rl]["dist"] or name in E77_TEACH[rl]["kd"]:
            self.get("e77", "dist" if name in E77_TEACH[rl]["dist"] else "kd", k, name, K)
        elif name in AGG and self.qC is None:
            return False
        else:
            self.get("e80c", "kd" if name not in ES.E77_KD[rl] else "dist", k, name, K)
        return True


# ---------------------------------------------------------------- 手法（1 つの生徒の選んだ設定だけを解く）

def alpha_of(pn):
    a = [x for x in CD.IM_ALPHAS if f"a{x}" == pn]
    if len(a) != 1:
        raise SystemExit(f"im の param {pn} が格子に無い")
    return a[0]


def lp_of(pn):
    for k in CD.LP_KS:
        for al in CD.LP_ALPHAS:
            for form in CD.LP_FORMS:
                if f"k{k}a{al}{form}" == pn:
                    return k, al, form
    raise SystemExit(f"lp の param {pn} が格子に無い")


def solve_student(world, k, pr, rec, sets_k, tch, lab, lab_full, rep, nc_seeds, map_cfg, map_src):
    """sets_k：[(K, 波, 方法, pnorm, m_eff)]．行の形と param は E77・E78・E80c と同じ。"""
    s, n, Zte = pr.s, pr.n, pr.s["Zte"]
    none, P0 = tch.x["none"][k], tch.x["P0"][k]
    by = {}
    for K, wave, meth, pn, meff in sets_k:
        by.setdefault(meth, []).append((K, wave, pn, meff))
    if "dist" in by:
        items = []
        for K, wave, pn, meff in by["dist"]:
            name, wgt = pn.split(":")
            if wgt not in CD.WGTS or meff not in CD.MEFFS:
                raise SystemExit(f"dist の param {pn} {meff} が格子に無い")
            items.append(((pn, meff, K), ES.rows_of(tch.get(wave, "dist", k, name, K), wgt, meff, n), K))
        for (pn, meff, K), Wb in pr.solve(items, what="dist").items():
            rec.add(k, "dist", pn, meff, K, Wb=Wb, Zte=Zte)
    for meth in ("kd", "kdh"):
        if meth not in by:
            continue
        items = []
        for K, wave, pn, meff in by[meth]:
            m_ = re.fullmatch(r"(.+)T(\d+)", pn)
            if m_ is None or int(m_.group(2)) not in CD.KD_TEMPS or meff not in CD.MEFFS:
                raise SystemExit(f"{meth} の param {pn} {meff} が格子に無い")
            P, Tm = tch.get(wave, "kd", k, m_.group(1), K), int(m_.group(2))
            items.append(((pn, meff, K), temp_rows(P, Tm, meff), K) if meth == "kd" else ((pn, meff, K), P, Tm, meff, K))
        sol = pr.solve(items, what="kd") if meth == "kd" else CD.kdh_solve(pr, items)
        for (pn, meff, K), Wb in sol.items():
            rec.add(k, meth, pn, meff, K, Wb=Wb, Zte=Zte)
    # 自己学習・entmin・IM は凸でないか巡を重ねるので，E77 の run_ab と同じ束（全部の τ・m_eff・α × 束の K）で解き，選んだ行だけを足す
    if "self" in by:                                                             # compare_dai.run_ab の自己学習（3 巡．none の確率から）
        want = set()
        for K, wave, pn, meff in by["self"]:
            if pn not in CD.TAUS or meff not in CD.MEFFS:
                raise SystemExit(f"self の param {pn} {meff} が格子に無い")
            want.add((pn, meff, K))
        keys = [(tau, meff, K) for tau in CD.TAUS for meff in CD.MEFFS for K in CD.KS]
        cur = {key: P0[key[2]] for key in keys}
        for rnd in range(BD.ROUNDS_SELF):
            hr = {key: hard_rows(cur[key], key[0], key[1]) for key in keys}
            sol = pr.solve([(key, hr[key], key[2]) for key in keys], what="self")
            for key in keys:                                                     # E77 の run_ab と同じ診断（巡ごとの疑似ラベルの本数と正しさ，テストの UAR．gate-code の R9）
                on = hr[key].view(4, n).sum(0) > 0
                lab_ = hr[key].view(4, n).argmax(0)
                rec.diag.append({**rec.tag, "student": k, "method": "self", "param": key[0], "meff": key[1], "K": key[2], "round": rnd + 1,
                                 "n_pseudo": int(on.sum()), "acc_pseudo": float((lab_[on] == pr.yt[on]).float().mean()) if on.any() else float("nan"),
                                 "uar_test": float(CB.uar(rec.yte, (Zte @ sol[key][0].T + sol[key][1]).argmax(1).cpu().numpy()))})
            cur = {key: pr.probs(sol[key], s["Z"]) for key in keys}
        for (tau, meff, K), Wb in sol.items():
            if (tau, meff, K) in want:
                rec.add(k, "self", tau, meff, K, Wb=Wb, Zte=Zte)
    if "entmin" in by or "im" in by:                                            # compare_dai.run_ab の entmin・IM（種 0 は none から，種 1・2 は揺らした出発点）
        ew = set()
        for K, wave, pn, meff in by.get("entmin", []):
            if pn != "-" or meff not in CD.MEFFS:
                raise SystemExit(f"entmin の param {pn} {meff} が格子に無い")
            ew.add((meff, K))
        iw = set()
        for K, wave, pn, meff in by.get("im", []):
            if meff not in CD.MEFFS:
                raise SystemExit(f"im の m_eff {meff} は選ばない（tim）")
            iw.add((alpha_of(pn), meff, K))
        ek = [(meff, K) for meff in CD.MEFFS for K in CD.KS] if ew else []                  # E77 の束（run_ab と同じ並び）
        ic = ([(a, meff, K) for a in CD.IM_ALPHAS for meff in CD.MEFFS for K in CD.KS] + [(0.1, "tim", K) for K in CD.KS_LAB]) if iw else []
        for seed in (0,) + tuple(nc_seeds):
            st0 = none if seed == 0 else CD.shaken(none, seed, rep, k)
            sfx = "" if seed == 0 else f"s{seed}"
            if ek:
                sol = pr.solve([(key, None, key[1]) for key in ek], u=[mass(m_, n) / n for m_, _ in ek], inits=[st0[K] for _, K in ek],
                               what="entmin" if seed == 0 else f"entmin_s{seed}")
                for (meff, K), Wb in sol.items():
                    if (meff, K) in ew:
                        rec.add(k, "entmin", f"-{sfx}", meff, K, Wb=Wb, Zte=Zte)
            if ic:
                for (a, meff, K), Wb in CD.im_solve(pr, st0, ic, what="im" if seed == 0 else f"im_s{seed}").items():
                    if (a, meff, K) in iw:
                        rec.add(k, "im", f"a{a}{sfx}", meff, K, Wb=Wb, Zte=Zte)
    for K, wave, pn, meff in by.get("t3a", []):
        Mv = pn[1:] if pn.startswith("M") else None
        if Mv not in CD.T3A_MS:
            raise SystemExit(f"t3a の param {pn} が格子に無い")
        lm, _ = CD.t3a_probs(pr, none[K], *lab[K], [Mv])[Mv]
        rec.add(k, "t3a", pn, "0", K, logits=lm)
    if "lp" in by:
        items = []
        for K, wave, pn, meff in by["lp"]:
            kk, al, form = lp_of(pn)
            if K == "0" or meff not in CD.MEFFS:
                raise SystemExit(f"lp の K={K}・m_eff {meff} は格子に無い")
            cg = {}
            Fs = CD.lp_scores(s["Z"], *lab[K], kk, al, info=cg)
            on = Fs.sum(1) > 0                                                   # E77 の run_ab と同じ診断（gate-code の R9）
            if len(lab[K][0]):
                on[torch.as_tensor(lab[K][0], device=on.device)] = False
            rec.diag.append({**rec.tag, "student": k, "method": "lp", "param": f"k{kk}a{al}", "meff": "-", "K": K, "round": 0, "n_pseudo": int(on.sum()),
                             "acc_pseudo": float((Fs[on].argmax(1) == pr.yt[on]).float().mean()) if on.any() else float("nan"),
                             "uar_test": float("nan"), "form": form, "lp_meff": meff, **cg})
            items.append(((pn, meff, K), CD.lp_rows(Fs, lab[K][0], form, meff, n), K))
        for (pn, meff, K), Wb in pr.solve(items, what="lp").items():
            rec.add(k, "lp", pn, meff, K, Wb=Wb, Zte=Zte)
    if any(m_ in by for m_ in SEEDED_MAP):                                      # compare_dai2.run_map と同じ束（モダリティの生徒だけ．TF32 は E78 と同じくこの間だけ）
        if k == world:
            raise SystemExit("融合の生徒では DANN・MME・mapctl を回さない（E78）")
        wanted = []
        for meth in SEEDED_MAP:
            for K, wave, pn, meff in by.get(meth, []):
                wanted.append((meth, pn, K))
        Zs, ys = map_src
        tf = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            run_map_sel(pr, Zs, ys, rec, k, lab_full, wanted, map_cfg)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf
    return by


def map_chunks(student, d, cfg):
    """E78 の本番の評価の単位で，この生徒の run_map が解いた束（(学習率, 反復の回数, 種) ごとに 30 問．次元が 1,024 を超えれば 12 問）の並び．
    cfg：methods（E78 の --methods の DANN・MME・mapctl），batch_ks（E78 の --ks），chosen78（(生徒, 方法, K) → 種 1・2 を解いた設定．
    E78 の `_chosen.csv` ＝ E79 の excl の選択）．compare_dai2.problems_for と同じ作り方（種 0 は全候補，種 1・2 は選んだ設定）。"""
    old = E78.KS
    E78.KS = tuple(cfg["batch_ks"])
    try:
        allp = E78.map_problems(cfg["methods"], (E78.SEEDS[0],))
        extra = [c for c in E78.map_problems(cfg["methods"], E78.SEEDS[1:])
                 if E78.strip_seed(E78.map_param(c)) in cfg["chosen78"].get((student, c[0], c[7]), set())]
    finally:
        E78.KS = old
    groups = {}
    for c in allp + extra:
        groups.setdefault((c[1], c[2], c[6]), []).append(c)
    step = 30 if d <= 1024 else 12
    return [probs[a:a + step] for probs in groups.values() for a in range(0, len(probs), step)]


def run_map_sel(pr, Zs, ys, rec, student, lab_full, wanted, cfg):
    """compare_dai2.run_map の写し：選んだ設定（種 0・1・2）を含む束だけを，E78 の本番と同じ中身で解き，選んだ設定の行だけを足す
    （DANN・MME は束の中身で値が変わる．E78 の再検査の I5-2）．束に無い選んだ設定は止める。"""
    Z = pr.s["Z"].to(E78.F32)
    Zte = pr.s["Zte"].to(E78.F32)
    Zs = Zs.to(E78.F32)
    n, d = Z.shape
    W0, b0 = pr.W0.to(E78.F32), pr.b0.to(E78.F32)
    cp, cb = float(pr.s["cp"]), pr.s["cb"]
    old = E78.KS
    E78.KS = tuple(cfg["batch_ks"])                                               # lab_tensors の K（E78 の本番と同じ）
    try:
        pos, ylab, wK = E78.lab_tensors(lab_full)
    finally:
        E78.KS = old
    Zl = Z[pos]
    ys_t = torch.as_tensor(np.asarray(ys), device=DEV, dtype=torch.long)
    I = torch.eye(d, dtype=E78.F32, device=DEV)
    want = {(m_, pn, K, sd) for m_, pn, K in wanted for sd in E78.SEEDS}
    found = set()
    for chunk in map_chunks(student, d, cfg):
        hit = [c for c in chunk if (c[0], E78.strip_seed(E78.map_param(c)), c[7], c[6]) in want]
        if not hit:
            continue
        lr, iters, seed = chunk[0][1], chunk[0][2], chunk[0][6]
        out = E78._train_map(Z, Zl, ylab, wK, Zs, ys_t, W0, b0, cp, cb, [(c[0], c[3], c[4], c[5], c[7]) for c in chunk], lr, iters, seed, I)
        for c, (Weff, bb, dg) in zip(chunk, out):
            key = (c[0], E78.strip_seed(E78.map_param(c)), c[7], c[6])
            if key not in want:
                continue
            rec.add(student, c[0], E78.map_param(c), "-", c[7], Wb=(Weff.to(DT), bb.to(DT)), Zte=Zte.to(DT))
            rec.diag.append({**rec.tag, "student": student, "method": c[0], "param": E78.map_param(c), "K": c[7], "chunk_size": len(chunk), **dg})
            found.add(key)
        del out
        torch.cuda.empty_cache()
    if found != want:
        raise SystemExit(f"{student}：E78 の束に無い設定 {sorted(want - found)}（chosen78 か --e78-methods・--batch-ks を確かめる）")


def coral_chosen(ev, Xu, te_X, src_pc, srcs, n_of, pool_s, C, gamma, rho_s, label_sets, yq, rec, student, c_raw, cp39, cb39, wanted, ks, missing):
    """compare_dai2.run_coral の写し（選んだ (r, ρ_M) だけ．同じ (r, ρ_M) の中は元と同じ束ね方：K × C_b の群と C′ の全部を 1 回で解き，選んだ param の行だけを足す）。"""
    want = {}
    for pn in wanted:
        m_ = re.fullmatch(r"r(\d+)rho(.+?)cp(.+?)cb(.+)", pn)
        if m_ is None or int(m_.group(1)) not in E78.CORAL_R:
            raise SystemExit(f"coral の param {pn} が格子に無い")
        rho = [x for x in E78.CORAL_RHO if f"{x}" == m_.group(2)]
        if len(rho) != 1:
            raise SystemExit(f"coral の param {pn} の ρ_M が格子に無い")
        want.setdefault(int(m_.group(1)), {}).setdefault(rho[0], set()).add(pn)
    n_min = min(len(Xu), min(len(X) for X, _ in srcs.values()))
    svd = {}
    for s_, (X, _) in srcs.items():
        both = torch.cat([X - X.mean(0), Xu])
        _, _, Vt = torch.linalg.svd(both - both.mean(0), full_matrices=False)
        svd[s_] = Vt[:max(E78.CORAL_R)]
        del both, Vt
    cps = tuple(sorted(set(E78.CORAL_CP) | {float(cp39)}))
    cbs_grid = tuple(sorted(set(c for c in E78.CORAL_CB if c is not None) | ({float(cb39)} if cb39 is not None else set()))) + (None,)
    lab_w = {}
    yt = M.labels(yq)
    for K in ks:
        idx = np.asarray(label_sets[K], int)
        w = torch.zeros(len(Xu), dtype=DT, device=DEV)
        if len(idx):
            it = torch.as_tensor(idx, device=DEV)
            w[it] = BD.present_weights(yt[it])
        lab_w[K] = (idx, w)
    got = set()
    for r in sorted(want):
        if n_min < E78.CORAL_MIN_RATIO * r:
            missing.append({"student": student, "method": "coral", "r": r, "why": "n_min < 5r"})
            continue
        t0 = time.perf_counter()
        lin = E78.coral_lin(srcs, Xu, rho_s, r, svd)
        st = Student(ev, Xu, src_pc, lin, n_of, pool_s, C, gamma, rho_s, list(E78.CORAL_RHO))
        Xstd = st.Xstd
        Xte_std = (te_X - st.src.mu) / st.src.sd
        st.Xstd = None
        for rho_m in E78.CORAL_RHO:
            if rho_m not in want[r]:
                continue
            Am = st.mats[rho_m][0]
            Zc = Xstd @ Am
            Zt = Xte_std @ Am
            groups = [(K, cbv) for K in ks for cbv in cbs_grid]
            masks, ws, cbs = [], [], []
            for K, cbv in groups:
                idx, w = lab_w[K]
                masks.append(np.ones(len(Zc), bool))
                ws.append(w)
                cbs.append(cbv)
            fits = logreg.fit_batch(Zc, yt, masks, cps, anchor=st.anchor(rho_m), anchor_c=st.C, stats=(st.zero, st.one), Cb=cbs, weights=ws)
            for gi, (K, cbv) in enumerate(groups):
                f = fits[gi]
                for cpv in cps:
                    cbs_ = "free" if cbv is None else cbv
                    par = f"r{r}rho{rho_m}cp{cpv}cb{cbs_}"
                    if (K, par) not in wanted_k(wanted, ks) or par not in want[r][rho_m]:
                        continue
                    W, b = f.W[cpv], f.b[cpv]
                    if K == "0":
                        W, b = st.W_std @ st.mats[rho_m][1], st.b_std
                    rec.add(student, "coral", par, "-", K, logits=Zt @ W.T + b)
                    got.add((K, par))
                    Wr = (W @ Am) / st.src.sd
                    br = b - Wr @ (c_raw + st.src.mu)
                    rec.W.setdefault(student + "@coral", []).append((len(rec.rows) - 1, Wr.to(torch.float32).cpu().numpy(), br.to(torch.float32).cpu().numpy()))
            del Zc, Zt, fits
        rec.conv.append({**rec.tag, "student": student, "what": f"coral_r{r}", "sec": time.perf_counter() - t0})
        del st, Xstd, Xte_std
        torch.cuda.empty_cache()
    return got


def wanted_k(wanted, ks):
    """coral の選んだ (K, param) の組（wanted は param → K の集合）。"""
    return {(K, p) for p, Ks in wanted.items() for K in Ks if K in ks}


# ---------------------------------------------------------------- 参照・教師のテストの行と kept の教師の配列（gate-code の R9・R10）

def extra_rows(world, T, S, data_f, data_m, pool, te, prs, rec, x, u, tch, agg, label_sets, ks, kept, studs, notes):
    """(1) 参照の行（`compare_dai.ref_rows`：ソースのみ，合併・ターゲットのみ（K=5），候補の全ラベルの上限 W*）を生徒ごとに，
    (2) 教師そのもののテストの行（E77 の world_unit の pF・pFK・poe0・ref・ref0 と，段 c の solve_unit の other・poolC・mlp・ds・agree），
    (3) kept の教師の全部の配列（`_teachers.npz`）．行は --ks の K だけ。"""
    mods = WORLDS[world]
    raw = CD.raw_world(world, data_f, data_m, T, S, pool, te)
    old = (CD.KS, CD.REF_KS)
    CD.KS, CD.REF_KS = tuple(ks), tuple(K for K in CD.REF_KS if K in ks)      # ref_rows は K ごとに独立に解くので，行を残す K だけ
    try:
        for k in mods + [world]:
            CD.ref_rows(rec, k, prs[k], raw[k][:4], label_sets)
    finally:
        CD.KS, CD.REF_KS = old
    del raw
    Zt_f = prs[world].s["Zte"]
    for K in ks:                                                                 # E77 の world_unit の teacher の行
        rec.add(world, "teacher", "pF", "0", K, logits=Zt_f @ prs[world].W0.T + prs[world].b0)
        rec.add(world, "teacher", "pFK", "0", K, Wb=x["none"][world][K], Zte=Zt_f)
        rec.add(world, "teacher", "poe0", "0", K, logits=x["lpoe_t"])
        rec.add(world, "teacher", "ref", "0", K, logits=x["lt"][K])
        rec.add(world, "teacher", "ref0", "0", K, logits=x["lt0"][K])
    names = [n_ for n_ in dict.fromkeys(n_ for k in [world] + mods for n_ in ES.r1_names(world, k)) if n_ not in ES.E77_TEACHER_ROWS]
    Lte = {m: torch.log_softmax(prs[m].s["Zte"] @ prs[m].W0.T + prs[m].b0, 1) for m in mods}   # 段 c の solve_unit の teacher の行と同じ
    P32 = {m: EA.f32(Lte[m].exp()) for m in mods}
    Fte = np.log(np.clip(np.stack([P32[m] for m in mods], 1).astype(np.float64), 1e-12, None))
    votes = np.stack([Lte[m].argmax(1).cpu().numpy() for m in mods], 1)
    pF_te = torch.softmax(Zt_f @ prs[world].W0.T + prs[world].b0, 1).cpu().numpy()
    aggte = ES.agg_pred(agg, Fte) if agg is not None else (None, None)
    for K in ks:
        for n_ in names:
            if n_.startswith("other_"):
                m = n_[len("other_"):]
                rec.add(world, "teacher", n_, "0", K, logits=prs[m].s["Zte"] @ prs[m].W0.T + prs[m].b0)
            elif n_ in AGG:
                if agg is None:
                    continue
                rec.add(world, "teacher", n_, "0", K, probs=aggte[AGG.index(n_)])
            elif n_ == "ds":
                rec.add(world, "teacher", n_, "0", K, probs=ES.ds_apply(votes, u["ds_pi"], list(u["ds_C"])))
            elif n_ == "agree":
                rec.add(world, "teacher", n_, "0", K, probs=pF_te * np.where((votes == votes[:, :1]).all(1), 1.0, ES.AGREE_LOW)[:, None])
            else:
                raise SystemExit(f"teacher の行の作り方が無い教師 {n_}")
    n_skip = 0
    for k in studs:                                                              # kept の教師の全部の配列（教師の疑似ラベルの正しさを後から測れるように）
        for K in ks:
            for n_ in kept_of(kept, T, k, K):
                n_skip += int(not tch.keep_array(k, n_, K))
    if n_skip:
        notes.append(f"{T}：agg_T が無いので poolC・mlp の配列 {n_skip} 個を残していない")


# ---------------------------------------------------------------- 評価のターゲットごと

def run_target(world, T, data_f, data_m, args, ks, bks, out, cache, commit, plan, used, unit_cache, prov, counts, c78, kept):
    mods = WORLDS[world]
    studs = mods + [world]
    notes = []
    S = [s for s in data_f if CM.corp(s) != CM.corp(T)]
    sets = {k: settings(k, T) for k in studs}
    samp = CD.samples_table(world, T)
    samp.to_csv(out / f"{T}_samples.csv", index_label="row")
    blk = block_labels(T, samp.sample_id.astype(str).to_numpy())
    ev = data_f[T]
    assert len(blk) == len(ev.y)
    ctx = G3Ctx(ev, args.reps, list(EA.KS_LAB_ALL), blk, exclude=not args.no_exclude)
    check_counts(ctx, world, T, counts, not args.no_exclude, notes, args.allow_incomplete)
    for r in args.reps:                                                         # 照合 (a)
        if (T, r) in SAME_SPLIT:
            st_ = ctx.stats[r]
            if st_["n_excl"] != 0 or not np.array_equal(ctx.pool[r], np.flatnonzero(~ctx.te[r])):
                raise SystemExit(f"{T} r{r}：テストの 2 人が同じセッションの単位なのに学習側が変わった")
        for K in EA.KS_LAB_ALL:                                                  # K 本は学習側の中（照合 (b) の前に構造で確かめる）
            assert np.isin(ctx.idx[(r, K)], ctx.pool[r]).all()
    tp = {key: v for key, v in plan.items() if key[0] == T}
    used_T = used[used.target == T] if len(used) else used
    used_T.to_csv(out / f"{T}_plan.csv", index=False)
    needs_agg = any(meth in ("dist", "kd", "kdh") and wave == "e80c" and (pn.split(":")[0] if meth == "dist" else re.sub(r"T\d+$", "", pn)) in AGG
                    for sets_ in tp.values() for (wave, meth, pn, meff) in sets_)
    needs_agg |= any(n_ in AGG for st in studs for K in ks for n_ in kept_of(kept, T, st, K))   # kept の教師の全部の配列を残す（gate-code の R9）
    CD.save_settings(out, T, sets, {"c_fit": C_FIT, "sources": S, "split": "cur" if args.no_exclude else "g3", "chosen_sha256": prov["chosen_sha256"],
                                    "kept_sha256": prov.get("kept_sha256"), "needs_agg": needs_agg,
                                    "plan": {f"{a}|{b}|{c}": sorted("|".join(k_) + "#" + ";".join(sorted(v)) for k_, v in sets_.items())
                                             for (a, b, c), sets_ in tp.items()}})
    if args.check:
        rows = [{"target": T, "rep": r, **ctx.stats[r]} for r in args.reps]
        for r in args.reps:
            try:
                BD.label_sets_of(ctx, r, ctx.pool[r])
            except KeyError as e:
                raise SystemExit(f"{T} r{r}：label_sets_of が通らない（{e}）")
        pd.DataFrame(rows).to_csv(out / f"{T}_units.csv", index=False)
        prov.setdefault("notes", {})[T] = notes
        print(f"{world} {T}：分割の照合だけ（{len(rows)} 反復，plan の行 {len(used_T)}）", flush=True)
        return
    agg_T, poolr = None, []
    if needs_agg:                                                                # 段 c と同じ：S の全群の仮のターゲットの全単位で学ぶ（分割は変えない）
        t0 = time.perf_counter()
        ukey = (tuple(S), json.dumps(sets, sort_keys=True, default=str), tuple(args.pseudo_reps))
        units = unit_cache.get(ukey)
        if units is None:
            units = agg_units(world, T, data_f, data_m, S, sets, args.pseudo_reps, cache)
            unit_cache.clear()
            unit_cache[ukey] = units
        keys = list(units)
        assert all(CM.corp(c) != CM.corp(T) for c, _ in keys)
        agg_T, poolr = ES.fit_agg(world, keys, units, {"pseudo": "*eval*"}, zlib.crc32(f"{world}|{T}|eval".encode()) & 0x7FFFFFFF)
        chk = check_agg(poolr, args.e80c_dir, T, notes, args.allow_incomplete)
        poolr = [{**x, **({k_: v for k_, v in chk.items()} if x.get("what") == "final" else {})} for x in poolr]
        print(f"  {world} {T}：agg_T（{len(keys)} 単位，照合 {chk['status']}）{time.perf_counter() - t0:.0f} 秒", flush=True)
    info = {k: M5.pool_info(data_m[k] if k != world else data_f, S, C_FIT, cache[k]) for k in sets}
    dat_of = lambda k, g: data_m[k][g] if k != world else data_f[g]  # noqa: E731
    rows, conv, convp, diag, urows, missing = [], [], [], [], [], []
    for r in args.reps:
        t0 = time.perf_counter()
        miss_r = []
        pool, te = ctx.pool[r], np.flatnonzero(ctx.te[r])
        try:
            label_sets = BD.label_sets_of(ctx, r, pool)                          # 照合 (b)：K 本が候補の中にあること
        except KeyError as e:
            raise SystemExit(f"{T} r{r}：label_sets_of が通らない（{e}）")
        yq = ev.y[pool]
        stu = make_students(world, T, data_f, data_m, S, pool, te, sets, info)
        t_stu = time.perf_counter() - t0
        rec = CD.Rec({"target": T, "rep": r}, ev.y[te], keep=True)
        rec.conv, rec.convp, rec.diag = conv, convp, diag
        for k in studs:                                                          # compare_dai.world_unit と同じ（中心は候補の平均）
            st = stu[k]["st"]
            rec.xform[k] = (st.mats[stu[k]["rho_m"]][0], st.src.sd, M.rows(dat_of(k, T).X, pool).mean(0) + st.src.mu)
        sets_of = {k: [(K, *key) for K in ks for key in tp.get((T, k, K), {})] for k in studs}
        t1 = time.perf_counter()
        for k in studs:                                                          # CORAL（E78 と同じく Prob を作る前．生徒は CORAL の中で作り直す）
            cw = {}
            for K, wave, meth, pn, meff in sets_of[k]:
                if meth == "coral":
                    cw.setdefault(pn, set()).add(K)
            if not cw:
                continue
            dat = dat_of(k, T)
            mu = M.rows(dat.X, pool).mean(0)
            gamma, rho_s, rho_m, cb, cp, _ = sets[k]
            src, lin, n_of = info[k]
            srcs = {s_: (dat_of(k, s_).X, dat_of(k, s_).y) for s_ in S}
            mr = []
            got = coral_chosen(dat, M.rows(dat.X, pool) - mu, M.rows(dat.X, te) - mu, src, srcs, n_of, S, C_FIT, gamma, rho_s, label_sets, yq, rec, k,
                               mu, stu[k]["cp"], stu[k]["cb"], cw, bks, mr)
            miss = wanted_k(cw, ks) - got
            if miss and not mr:
                raise SystemExit(f"{T} r{r} {k}：coral の行 {sorted(miss)} を作れなかった")
            miss_r += [{"rep": r, "student": k, "method": "coral", "K": K_, "param": p_, "why": "n_min < 5r（E78 と同じく作らない）"} for K_, p_ in sorted(miss)]
        t_coral = time.perf_counter() - t1
        t1 = time.perf_counter()
        u, prs, x = unit_pass1x(world, stu, yq, ev.y[te], label_sets, bks)
        tch = Teachers(world, u, x, agg_T)
        for k in [world] + mods:                                                 # none の行（E77 と同じ束ね方の解．E80c の none_rows と同じ形）
            for K in ks:
                rec.add(k, "none", "-", "0", K, Wb=x["none"][k][K], Zte=prs[k].s["Zte"])
        t_p1 = time.perf_counter() - t1
        t1 = time.perf_counter()
        lab = CD.lab_of(label_sets, yq)
        lab_full = {K: (np.asarray(label_sets[K], int), np.asarray(yq)[np.asarray(label_sets[K], int)] if len(label_sets[K]) else np.array([], int))
                    for K in BD.KS}                                              # E78 の lab_tensors は K=25 の和集合を使う
        for k in studs:
            sk = [z for z in sets_of[k] if z[2] != "coral"]
            if not sk:
                continue
            map_src = None
            if any(z[2] in SEEDED_MAP for z in sk):
                map_src = E78.src_feats(stu[k]["st"], stu[k]["rho_m"], {s_: (dat_of(k, s_).X, dat_of(k, s_).y) for s_ in S})
            map_cfg = {"methods": args.e78_methods, "batch_ks": bks, "chosen78": c78.get(T, {})}
            solve_student(world, k, prs[k], rec, sk, tch, lab, lab_full, r, args.nc_seeds, map_cfg, map_src)
        t_solve = time.perf_counter() - t1
        t1 = time.perf_counter()
        n0 = len(rec.rows)
        extra_rows(world, T, S, data_f, data_m, pool, te, prs, rec, x, u, tch, agg_T, label_sets, ks, kept, studs, notes)
        n_extra = len(rec.rows) - n0
        t_extra = time.perf_counter() - t1
        CD.flush_stats(rec, prs)
        sel_of = {}                                                              # 行 → 選んだ族（none は "none"）
        for k in studs:
            for K in ks:
                for (wave, meth, pn, meff), sels in tp.get((T, k, K), {}).items():
                    sel_of[(k, meth, pn, meff, K)] = (wave, ";".join(sorted(sels)))
        for row in rec.rows:
            pn = E78.strip_seed(row["param"]) if row["method"] in SEEDED_NC + SEEDED_MAP else row["param"]
            seed = -1
            if row["method"] in SEEDED_NC + SEEDED_MAP:
                m_ = re.search(r"s(\d+)$", row["param"])
                seed = int(m_.group(1)) if m_ else 0
            if row["method"] == "none":
                wave, sels = "e77", "none"
            elif row["method"] in REF_METHODS:
                wave, sels = "e77", "ref"
            elif row["method"] == "teacher":
                wave, sels = ("e77" if row["param"] in ES.E77_TEACHER_ROWS else "e80c"), "teacher"
            else:
                key = (row["student"], row["method"], pn, row["meff"], row["K"])
                if key not in sel_of:
                    raise SystemExit(f"{T} r{r}：計画に無い行 {key}")
                wave, sels = sel_of[key]
            row.update({"run": world, "wave": wave, "selectors": sels, "pnorm": pn, "seed": seed, "split": "cur" if args.no_exclude else "g3",
                        "n_pool": len(pool)})
        want_rows = sum((len(args.nc_seeds) + 1 if key[1] in SEEDED_NC else (len(E78.SEEDS) if key[1] in SEEDED_MAP else 1))
                        for k in studs for K in ks for key in tp.get((T, k, K), {})) + len(studs) * len(ks)
        n_coral_missing = len(miss_r)
        if len(rec.rows) - n_extra != want_rows - n_coral_missing:
            raise SystemExit(f"{T} r{r}：選んだ設定と none の行の数 {len(rec.rows) - n_extra} が期待 {want_rows - n_coral_missing} と違う")
        missing += miss_r
        rec.teachers = tch.used
        rec.save(out, T, r, te)
        np.save(out / f"{T}_r{r}_pool.npy", pool)
        np.save(out / f"{T}_r{r}_excl.npy", ctx.excl[r])
        np.savez(out / f"{T}_r{r}_kshot.npz", **{f"K{K}": ctx.idx[(r, K)] for K in EA.KS_LAB_ALL})
        rows += rec.rows
        sec = time.perf_counter() - t0
        urows.append({"target": T, "rep": r, **ctx.stats[r], "n_rows": len(rec.rows), "n_rows_expected": want_rows, "n_extra_rows": n_extra, "n_coral_missing": n_coral_missing,
                      "sec": round(sec, 2), "sec_students": round(t_stu, 2), "sec_coral": round(t_coral, 2), "sec_pass1": round(t_p1, 2),
                      "sec_solve": round(t_solve, 2), "sec_extra": round(t_extra, 2), "gpu_max_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else None})
        print(f"  {world} {T} r{r}：{len(rec.rows)} 行，候補 {len(pool)}（除いた {ctx.stats[r]['n_excl']}）　{sec:.0f} 秒", flush=True)
        del stu, prs, rec, u, x, tch
        torch.cuda.empty_cache()
    for name, rows_ in (("units", urows), ("conv", conv), ("convp", convp), ("diag", diag), ("pool", poolr), ("missing", missing)):
        pd.DataFrame(rows_).assign(commit=commit).to_csv(out / f"{T}_{name}.csv", index=False)
    d = pd.DataFrame(rows).assign(commit=commit)
    if d.duplicated(["rep", "student", "method", "param", "meff", "K"]).any():
        raise SystemExit(f"{T}：行の鍵が重なる")
    cols = ["run", "target", "rep", "row_id", "student", "method", "param", "meff", "K", "uar", "wave", "selectors", "pnorm", "seed", "split", "n_pool", "commit"]
    d[cols].to_csv(out / f"{T}.csv", index=False)                               # 最後に書く（再開の印）
    if notes:
        prov.setdefault("notes", {})[T] = notes


# ---------------------------------------------------------------- 入口

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", choices=E81_WORLDS, required=True)
    ap.add_argument("--targets", nargs="+", default=list(E81_TARGETS), choices=E81_TARGETS)
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--ks", nargs="+", default=["0", "5"], help="解く K（K 本の抜き方は全 K で作る）")
    ap.add_argument("--chosen", type=Path, default=ROOT / "results" / "e79" / "select" / "chosen.csv")
    ap.add_argument("--selectors", nargs="+", default=["oursx", "kd", "groupB"])
    ap.add_argument("--variant", default="incl")
    ap.add_argument("--kept", type=Path, default=ROOT / "results" / "e80" / "kept.csv", help="照合 (e)：oursx・kd の教師が残した教師か")
    ap.add_argument("--e80c-dir", type=Path, help="照合 (c)：段 c の出力の場所（既定 results/e80c_<世界>．無ければ照合しない）")
    ap.add_argument("--counts", type=Path, default=ROOT / "checks" / "E81" / "g3_counts_all.csv", help="照合 (d)：学習側の大きさ")
    ap.add_argument("--pseudo-reps", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="agg_T を学ぶ仮のターゲットの反復（段 c と同じ）")
    ap.add_argument("--nc-seeds", nargs="*", type=int, default=[1, 2], help="entmin・IM の揺らした出発点の種（E77 と同じ）")
    ap.add_argument("--batch-ks", nargs="+", default=["0", "5", "25"],
                    help="束に入れる K（E77・E78・E80c の本番の --ks 0 5 25 と束の中身をそろえる：none・教師・CORAL・DANN 等．行は --ks の K だけ残す）")
    ap.add_argument("--e78-methods", nargs="+", default=["dann", "mme", "mapctl"], help="E78 の本番の --methods のうち DANN・MME・mapctl（束の中身）")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--gpu-fraction", type=float, default=0.3)
    ap.add_argument("--no-exclude", action="store_true", help="除かない（今の分割と同じ単位．検算用）")
    ap.add_argument("--check", action="store_true", help="分割と chosen.csv の照合だけ（解かない）")
    ap.add_argument("--allow-incomplete", action="store_true", help="試し用：chosen.csv に無い（T, 生徒, K, 族）を飛ばす")
    ap.add_argument("--allow-dirty", action="store_true", help="試し用：読み込んだコードに未コミットの変更があっても流す")
    args = ap.parse_args()
    world = args.world
    if args.e80c_dir is None:
        args.e80c_dir = ROOT / "results" / f"e80c_{world}"
    if not set(args.ks) <= set(args.batch_ks):
        ap.error("--ks は --batch-ks に含める")
    CD.set_ks(args.batch_ks)                                                    # 束の K（shaken・lab_of・none・教師）
    bks = CD.KS
    ks = tuple(K for K in bks if K in args.ks)                                  # 行を残す K
    code = loaded_code()
    bad = dirty_code(list(code))
    if bad and not args.allow_dirty:
        raise SystemExit(f"読み込んだコードに未コミットの変更がある：{bad}．コミットしてから流す（試しは --allow-dirty）")
    out = args.out or ROOT / "results" / f"e81_{world}"
    out.mkdir(parents=True, exist_ok=True)
    commit = git("rev-parse", "HEAD")
    groups = groups_for(world)
    targets = [t for t in args.targets if t in groups]
    todo = targets if args.check else [t for t in targets if not CD.check_resume(out, t, commit)]
    notes = []
    plan, used = read_chosen(args.chosen, world, targets, ks, args.selectors, args.variant, args.allow_incomplete, notes)
    c78 = read_chosen78(args.chosen, world, targets, bks, args.e78_methods, plan, notes, args.allow_incomplete)
    kept_sha, kept = check_kept(plan, args.kept, world, notes, args.allow_incomplete)
    counts = pd.read_csv(args.counts) if args.counts and Path(args.counts).exists() else None
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction)
    prov = {"script": "exp/g3_split.py", "experiment": "E81", "argv": sys.argv[1:], "commit": commit, "dirty_code": bad, "allow_dirty": args.allow_dirty,
            "code_sha256": code, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "world": world, "targets": todo, "reps": args.reps, "KS": list(ks),
            "split": "cur" if args.no_exclude else "g3", "check_only": args.check, "chosen": str(args.chosen), "chosen_sha256": sha_file(args.chosen),
            "selectors": args.selectors, "variant": args.variant, "kept": str(args.kept), "kept_sha256": kept_sha, "e80c_dir": str(args.e80c_dir),
            "counts": str(args.counts), "counts_sha256": sha_file(args.counts) if counts is not None else None, "pseudo_reps": args.pseudo_reps,
            "nc_seeds": list(args.nc_seeds), "nc_scale": CD.NC_SCALE, "map_seeds": list(E78.SEEDS), "batch_ks": list(bks), "e78_methods": args.e78_methods,
            "chosen78": {T: {"|".join(k_): sorted(v) for k_, v in d_.items()} for T, d_ in c78.items()}, "gpu_fraction": args.gpu_fraction,
            "tf32": "dann・mme・mapctl の run_map の間だけ真（E78 と同じ）．ほかは偽（E77・E80c と同じ）", "agg_tol": AGG_TOL,
            "allow_incomplete": args.allow_incomplete, "unchecked": UNCHECKED,
            "block_cols": {k: list(v[1]) for k, v in BLOCK_COLS.items()}, "same_split_units": sorted(SAME_SPLIT), **CD.versions(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", "sec": {}, "notes": {"chosen": notes}}
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1, default=str))
    if not todo:
        print(f"{world}：全ターゲット済み", flush=True)
        return 0
    t_all = time.perf_counter()
    data_f = {g: M.load(g, world) for g in groups}
    data_m = {m: {g: sub_group(data_f[g], a, b) for g in groups} for m, a, b in blocks(world)}
    cache = {k: {} for k in WORLDS[world] + [world]}
    unit_cache = {}
    for T in todo:
        t0 = time.perf_counter()
        run_target(world, T, data_f, data_m, args, ks, bks, out, cache, commit, plan, used, unit_cache, prov, counts, c78, kept)
        prov["sec"][T] = round(time.perf_counter() - t0, 1)
        (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1, default=str))
        mem = f"　GPU 最大 {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB" if torch.cuda.is_available() else ""
        print(f"{world} {T}：{time.perf_counter() - t0:.0f} 秒{mem}", flush=True)
    prov["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    prov["sec"]["total"] = round(time.perf_counter() - t_all, 1)
    prov["gpu_max_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else None
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1, default=str))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
