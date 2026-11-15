"""実験 17 の下調べ（DAI-17）の，設計で「計算の後に出す」とした量．新しい学習はしない．保存した出力（`results/dai17_probe/`）だけを読む．

1. overlap.csv：両方の近さ（both）で引いた上位 100 本と，特徴だけ（feat）・答えだけ（logit）で引いた上位 100 本の重なり（テスト．声・顔・言葉）．
2. same_actor.csv：テストの発話と引いた発話が同じ俳優である対の数（AFEW と MELD の両方に出る 6 人．対応表は下の ACTORS．sample_id の水準）．
3. by_source_bal.csv：ソースのコーパスごとの，感情ごとの当たる割合の平均（テスト．both の 4 つの枠）．
4. students_cond_sel.csv：生徒の段の 3 つの条件の量を，選び直した点（students_sel.csv）で出したもの．
5. speakers_top20.csv：上位 20 本の別々の話者と塊の数（neighbors.csv から．でたらめに引いたときの値と並べる）．

使い方：CUDA_VISIBLE_DEVICES="" .venv/bin/python exp/dai17_posthoc.py [--out-dir results/dai17_probe]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODS = "AVT"
# AFEW と MELD の両方に出る俳優（`checks/DAI-17/review_affect.md` A2・`review_affect_v3.md` b1．MELD の側は役名と話の番号で決める）
ACTORS = {"Elliott Gould": ("AFEW_Elliott_Gould", ["Mr._Geller"], None), "Ben Stiller": ("AFEW_Ben_Stiller", ["Tommy"], [(3, 22)]),
          "Susan Sarandon": ("AFEW_Susan_Sarandon", ["Cecilia", "Jessica_Lockhart"], [(7, 15)]), "Sean Penn": ("AFEW_Sean_Penn", ["Eric"], [(8, 7)]),
          "Gary Oldman": ("AFEW_Gary_Oldman", ["Richard"], [(7, 23), (7, 24)]), "Jason Alexander": ("AFEW_Jason_Alexander", ["Earl"], [(7, 13)])}
N_ACTOR_DESIGN = {"AFEW": 11, "MELD": 62}                               # 設計 7 節の数（3 モダリティがそろい，表に入る発話）


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def actor_of_sid() -> dict:
    """sample_id → 俳優名（6 人だけ）．"""
    a = pd.read_csv(ROOT / "data/metadata/afew_utterance_manifest.csv", low_memory=False)
    m = pd.read_csv(ROOT / "data/metadata/meld_utterance_manifest.csv", low_memory=False)
    out = {}
    for name, (afew_id, meld_roles, eps) in ACTORS.items():
        for sid in a[a.speaker_id.astype(str).isin([afew_id])].sample_id.astype(str):
            out[sid] = name
        d = m[m.speaker_id.astype(str).isin(meld_roles)]
        if eps is not None:
            d = d[[(int(s_), int(e_)) in eps for s_, e_ in zip(d.season, d.episode)]]
        for sid in d.sample_id.astype(str):
            out[sid] = name
    return out


def dependency_table(out: Path, pdir: Path) -> None:
    """設計 3.9 節「選択を通した依存」の表（gate-code の点検の台本 checks/DAI-17/code_run_post/dependency_table.py と同じ規則）．
    評価するコーパス c の生徒の設定は，c を評価のターゲットにした単位（仮のターゲット c′ ≠ c）で選ぶ．その単位の疑似ラベルの版は，T = c′ の 30 単位
    （仮のターゲットに c を含む）で選んだ点である．c′ の点を，仮のターゲットが c の単位を除いた 20 単位で選び直し，元の点と比べる．"""
    import sys
    sys.path.insert(0, str(ROOT / "exp"))
    import dai17_probe as P
    units = pd.concat([pd.read_csv(q) for q in sorted((out / "units").glob("*.csv"))], ignore_index=True)
    v = units[~units.version.astype(str).str.startswith("_")]
    fz = json.load(open(out / "frozen_sel.json"))["choice"]

    def select(df, cp, K, name):
        g = df[(df["T"] == cp) & (df.K == K)].groupby("version").uar_adj.mean()
        cands = P.SELDEF[name]
        vals = np.array([g.get(x, np.nan) for x in cands])
        return cands[int(np.nanargmax(vals))]

    rows = []
    corp = ("AFEW", "IEMOCAP", "MOSEI", "MELD")
    for c in corp:
        for cp in corp:
            if cp == c:
                continue
            sub = v[(v["T"] == cp) & (v.c != c)]
            for K in tuple(json.load(open(out / "provenance.json"))["KS"]):
                new, orig = {}, {}
                for name in ("MN_eq_sel", "MN_F2K_sel", "MN_F6K_sel", "MS_eq_sel"):
                    new[name], orig[name] = select(sub, cp, K, name), fz[f"{cp}|{K}|{name}"]
                new["MS_eq_tied"], orig["MS_eq_tied"] = f"MS_eq_{new['MN_eq_sel'].rsplit('_', 1)[1]}", fz[f"{cp}|{K}|MS_eq_tied"]
                for t_ in new:
                    rows.append(dict(c=c, c_prime=cp, K=K, teacher=t_, n_units_used=sub.groupby(["c", "fold", "seed"]).ngroups, changed=(new[t_] != orig[t_]),
                                     orig=orig[t_], reselected_without_c=new[t_]))
    d = pd.DataFrame(rows)
    d.to_csv(pdir / "dependency_detail.csv", index=False)
    d.groupby(["c", "K", "teacher"]).agg(n_cprime=("changed", "size"), n_changed=("changed", "sum"),
                                         changed_cprime=("c_prime", lambda q: ",".join(q[d.loc[q.index, "changed"]]))).reset_index().to_csv(pdir / "dependency_count.csv", index=False)


def run(out: Path) -> None:
    pdir = out / "posthoc"
    pdir.mkdir(exist_ok=True)
    act = actor_of_sid()
    afew_sids = set(pd.read_csv(ROOT / "data/metadata/afew_utterance_manifest.csv", low_memory=False).sample_id.astype(str))
    in_db = {"AFEW": set(), "MELD": set()}
    for q in sorted((out / "srcdb").glob("*_A.npz")):
        d_ = np.load(q, allow_pickle=False)
        for x in d_["sid"].astype(str):
            if x in act:
                in_db["AFEW" if x in afew_sids else "MELD"].add(x)
    n_in = {k: len(v) for k, v in in_db.items()}
    if n_in != N_ACTOR_DESIGN:
        raise SystemExit(f"表に入る俳優の発話の数 {n_in} が設計の数 {N_ACTOR_DESIGN} と違う")
    ov, sa, bs = [], [], []
    files = sorted((out / "units").glob("*.npz"))
    for p in files:
        T, c, f, r = p.stem.rsplit("_", 3)
        if r != "r0":                                                    # 近い発話は種に依らない
            continue
        z = np.load(p, allow_pickle=False)
        srcs = list(z["srcs"])
        yt = z["y_test"].astype(int)
        sid_t = z["sid_test"].astype(str)
        for slot in "AVTJ":
            db = np.load(out / "srcdb" / f"{'+'.join(srcs)}_{slot}.npz", allow_pickle=False)
            it = z[f"nn_test_both_{slot}"]
            if slot != "J":
                for sp in ("feat", "logit"):
                    o = np.mean([len(np.intersect1d(a_, b_)) / it.shape[1] for a_, b_ in zip(it, z[f"nn_test_{sp}_{slot}"])])
                    ov.append(dict(T=T, c=c, fold=f, modality=slot, other=sp, overlap=float(o)))
            # 同じ俳優の対（空間ごと．期待値は，表からでたらめに k 本引いたときの同じ俳優の対の数）
            ta = np.array([act.get(x, "") for x in sid_t])
            sa_names = np.array([act.get(x, "") for x in db["sid"].astype(str)])
            n_db = len(sa_names)
            cnt_db = {nm: int((sa_names == nm).sum()) for nm in set(sa_names) - {""}}
            for sp in (("both", "feat", "logit") if slot != "J" else ("both",)):
                it_s = z[f"nn_test_{sp}_{slot}"]
                same = (sa_names[it_s] == ta[:, None]) & (ta[:, None] != "")
                exp100 = float(sum(cnt_db.get(x, 0) for x in ta if x) * it_s.shape[1] / n_db)
                sa.append(dict(T=T, c=c, fold=f, modality=slot, space=sp, possible=bool(any(cnt_db.get(x, 0) for x in ta if x)), n_test_actor=int((ta != "").sum()),
                               n_db_actor=int(sum(cnt_db.values())), n_pairs_same=int(same.sum()), n_pairs_same_top20=int(same[:, :20].sum()),
                               exp_random_100=exp100, exp_random_20=exp100 * 20 / it_s.shape[1], n_pairs=int(it_s.size)))
            # ソースごとの感情ごとの当たる割合の平均（空間ごと）
            for sp in (("both", "feat", "logit") if slot != "J" else ("both",)):
                it_s = z[f"nn_test_{sp}_{slot}"]
                Y, C = db["y"][it_s], db["corp"][it_s]
                hit = Y == yt[:, None]
                for ci, s_ in enumerate(srcs):
                    vals = [hit[yt == e][(C[yt == e] == ci)].mean() for e in range(4) if (yt == e).any() and (C[yt == e] == ci).any()]
                    bs.append(dict(T=T, c=c, fold=f, modality=slot, space=sp, source=s_, hit_bal=float(np.mean(vals)) if vals else np.nan, n_emotions=len(vals),
                                   n_pairs_source=int((C == ci).sum()), share=float((C == ci).mean())))
    pd.DataFrame(ov).groupby(["c", "modality", "other"]).overlap.agg(["mean", "min", "max", "count"]).reset_index().to_csv(pdir / "overlap.csv", index=False)
    sad = pd.DataFrame(sa)
    sad.to_csv(pdir / "same_actor_units.csv", index=False)
    sad[sad.possible].groupby(["c", "modality", "space"]).agg(n_units_possible=("possible", "size"), n_test_actor=("n_test_actor", "sum"), n_pairs_same=("n_pairs_same", "sum"),
        exp_random_100=("exp_random_100", "sum"), n_pairs_same_top20=("n_pairs_same_top20", "sum"), exp_random_20=("exp_random_20", "sum"),
        n_pairs=("n_pairs", "sum")).reset_index().to_csv(pdir / "same_actor.csv", index=False)   # 対が起こりうる単位だけ．n_test_actor はソースの組（T）の数だけ重ねて数える
    pd.DataFrame(bs).groupby(["c", "modality", "space", "source"])[["hit_bal", "share", "n_emotions", "n_pairs_source"]].agg(["mean", "min", "count"]).pipe(
        lambda q: q.set_axis([f"{a}_{b}" for a, b in q.columns], axis=1)).reset_index().to_csv(pdir / "by_source_bal.csv", index=False)
    # 生徒の段の条件の量を，選び直した点で
    sc = pd.concat([pd.read_csv(q) for q in sorted((out / "students").glob("*.csv"))], ignore_index=True)
    sel = pd.read_csv(out / "summary" / "students_sel.csv")
    tt = sc[sc.student == "_teacher"][["T", "c", "fold", "seed", "K", "teacher", "uar_pool", "maxp"]].rename(columns={"uar_pool": "uar_pool_teacher"})
    nn_ = sc[sc.teacher == "_check"][["T", "c", "fold", "seed", "K", "student", "uar_none", "uar_pool_none"]]
    st = sc[(sc.student != "_teacher") & (sc.teacher != "_check")][["T", "c", "fold", "seed", "K", "student", "teacher", "grid", "uar", "uar_pool", "mimic"]].copy()
    st = st.merge(sel[["c", "K", "student", "teacher", "grid"]].rename(columns={"grid": "grid_sel"}), on=["c", "K", "student", "teacher"])
    st = st[st.grid == st.grid_sel]                                      # 仮のターゲット c の判定に使う点（c を評価のターゲットにした単位で選んだ点）
    cond = st.merge(tt, on=["T", "c", "fold", "seed", "K", "teacher"]).merge(nn_, on=["T", "c", "fold", "seed", "K", "student"])
    cond["edge"] = cond.uar_pool_teacher - cond.uar_pool_none
    cond["gain_none"] = cond.uar - cond.uar_none
    cond.groupby(["c", "K", "student", "teacher"])[["edge", "mimic", "gain_none", "maxp"]].mean().reset_index().to_csv(
        pdir / "students_cond_sel.csv", index=False)
    dependency_table(out, pdir)
    nb_ = pd.read_csv(out / "summary" / "neighbors.csv")
    nb_[nb_.by.isin(["distinct_speakers_top20", "distinct_clusters_top20"])].to_csv(pdir / "speakers_top20.csv", index=False)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "exp"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    with open(pdir / "posthoc_provenance.json", "w") as fh:
        json.dump(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), commit=commit, dirty=dirty, script_sha256=sha(Path(__file__).resolve()),
                       n_units_r0=len([q for q in files if q.stem.endswith("_r0")]), actors={k: [v[0], v[1], v[2]] for k, v in ACTORS.items()},
                       n_actor_sids=len(act), n_actor_in_db=n_in,
                       inputs_sha256={q.name: sha(q) for q in [out / "summary" / "students_sel.csv", out / "summary" / "neighbors.csv", out / "frozen_sel.json",
                                                               ROOT / "data/metadata/afew_utterance_manifest.csv", ROOT / "data/metadata/meld_utterance_manifest.csv"]},
                       units_commit=sorted({str(x) for q in files[:1] for x in pd.read_csv(q.with_suffix(".csv")).commit.unique()})), fh, ensure_ascii=False, indent=1)
    print(f"posthoc：単位（種 0）{len([q for q in files if q.stem.endswith('_r0')])}，俳優の発話 {len(act)}，条件の行 {len(cond)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "dai17_probe"))
    run(Path(ap.parse_args().out_dir).resolve())
