"""実験 17 の下調べ（DAI-17）の，設計で「計算の後に出す」とした量．新しい学習はしない．保存した出力（`results/dai17_probe/`）だけを読む．

1. overlap.csv：両方の近さ（both）で引いた上位 100 本と，特徴だけ（feat）・答えだけ（logit）で引いた上位 100 本の重なり（テスト．声・顔・言葉）．
2. same_actor.csv：テストの発話と引いた発話が同じ俳優である対の数（AFEW と MELD の両方に出る 6 人．対応表は下の ACTORS．sample_id の水準）．
3. by_source_bal.csv：ソースのコーパスごとの，感情ごとの当たる割合の平均（テスト．both の 4 つの枠）．
4. students_cond_sel.csv：生徒の段の 3 つの条件の量を，選び直した点（students_sel.csv）で出したもの．
5. speakers_top20.csv：上位 20 本の別々の話者と塊の数（neighbors.csv から．でたらめに引いたときの値と並べる）．

使い方：CUDA_VISIBLE_DEVICES="" .venv/bin/python DAI/exp/dai17_posthoc.py [--out-dir results/dai17_probe]
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

ROOT = Path("/home/snics/MER-DA")
MODS = "AVT"
# AFEW と MELD の両方に出る俳優（`DAI/checks/DAI-17/review_affect.md` A2・`review_affect_v3.md` b1．MELD の側は役名と話の番号で決める）
ACTORS = {"Elliott Gould": ("AFEW_Elliott_Gould", "Mr._Geller", None), "Ben Stiller": ("AFEW_Ben_Stiller", "Tommy", [(3, 22)]),
          "Susan Sarandon": ("AFEW_Susan_Sarandon", "Cecilia", [(7, 15)]), "Sean Penn": ("AFEW_Sean_Penn", "Eric", [(8, 7)]),
          "Gary Oldman": ("AFEW_Gary_Oldman", "Richard", [(7, 23), (7, 24)]), "Jason Alexander": ("AFEW_Jason_Alexander", "Earl", [(7, 13)])}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def actor_of_sid() -> dict:
    """sample_id → 俳優名（6 人だけ）．"""
    a = pd.read_csv(ROOT / "data/metadata/afew_utterance_manifest.csv", low_memory=False)
    m = pd.read_csv(ROOT / "data/metadata/meld_utterance_manifest.csv", low_memory=False)
    out = {}
    for name, (afew_id, meld_role, eps) in ACTORS.items():
        for sid in a[a.speaker_id.astype(str).isin([afew_id])].sample_id.astype(str):
            out[sid] = name
        d = m[m.speaker_id.astype(str) == meld_role]
        if eps is not None:
            d = d[[(int(s_), int(e_)) in eps for s_, e_ in zip(d.season, d.episode)]]
        for sid in d.sample_id.astype(str):
            out[sid] = name
    return out


def run(out: Path) -> None:
    pdir = out / "posthoc"
    pdir.mkdir(exist_ok=True)
    act = actor_of_sid()
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
            # 同じ俳優の対
            ta = np.array([act.get(s, "") for s in sid_t])
            sa_names = np.array([act.get(s, "") for s in db["sid"].astype(str)])
            nb = sa_names[it]
            same = (nb == ta[:, None]) & (ta[:, None] != "")
            sa.append(dict(T=T, c=c, fold=f, modality=slot, n_test_actor=int((ta != "").sum()), n_pairs_same=int(same.sum()),
                           n_pairs_same_top20=int(same[:, :20].sum()), n_pairs=int(it.size)))
            # ソースごとの感情ごとの当たる割合の平均
            Y, C = db["y"][it], db["corp"][it]
            hit = Y == yt[:, None]
            for ci, s in enumerate(srcs):
                vals = [hit[yt == e][(C[yt == e] == ci)].mean() for e in range(4) if (yt == e).any() and (C[yt == e] == ci).any()]
                bs.append(dict(T=T, c=c, fold=f, modality=slot, source=s, hit_bal=float(np.mean(vals)) if vals else np.nan, n_emotions=len(vals),
                               share=float((C == ci).mean())))
    pd.DataFrame(ov).groupby(["c", "modality", "other"]).overlap.agg(["mean", "min", "max", "count"]).reset_index().to_csv(pdir / "overlap.csv", index=False)
    sad = pd.DataFrame(sa)
    sad.to_csv(pdir / "same_actor_units.csv", index=False)
    sad.groupby(["c", "modality"])[["n_test_actor", "n_pairs_same", "n_pairs_same_top20", "n_pairs"]].sum().reset_index().to_csv(pdir / "same_actor.csv", index=False)
    pd.DataFrame(bs).groupby(["c", "modality", "source"])[["hit_bal", "share"]].agg(["mean", "count"]).pipe(
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
    nb_ = pd.read_csv(out / "summary" / "neighbors.csv")
    nb_[nb_.by.isin(["distinct_speakers_top20", "distinct_clusters_top20"])].to_csv(pdir / "speakers_top20.csv", index=False)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "DAI/exp"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    with open(pdir / "posthoc_provenance.json", "w") as fh:
        json.dump(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), commit=commit, dirty=dirty, script_sha256=sha(Path(__file__).resolve()),
                       n_units_r0=len([q for q in files if q.stem.endswith("_r0")]), actors={k: list(map(str, v[:2])) for k, v in ACTORS.items()},
                       n_actor_sids=len(act)), fh, ensure_ascii=False, indent=1)
    print(f"posthoc：単位（種 0）{len([q for q in files if q.stem.endswith('_r0')])}，俳優の発話 {len(act)}，条件の行 {len(cond)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "dai17_probe"))
    run(Path(ap.parse_args().out_dir).resolve())
