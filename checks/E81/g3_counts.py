"""Structure-only counts for G3 (no evaluation values). Replicates map.test_mask / kshot_order on CPU."""
import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = "/home/snics/MER-DA"
sys.path.insert(0, ROOT + "/measure"); sys.path.insert(0, ROOT + "/features")
import numpy as np, pandas as pd
import common
from data import load_group, fused_parts, groups_for
from split import speaker_folds, check_disjoint

def ids_df(T, kind):
    parts = fused_parts(kind)
    if not parts:
        return load_group(T, kind)[0]
    dfs = [load_group(T, k)[0] for k in parts[1]]
    ids = set.intersection(*[set(d.sample_id) for d in dfs])
    d = dfs[0][dfs[0].sample_id.isin(ids)].reset_index(drop=True)
    o = np.argsort(d.sample_id.to_numpy())
    return d.iloc[o].reset_index(drop=True)

man = {"IEMOCAP": pd.read_csv(ROOT + "/data/metadata/iemocap_utterance_manifest.csv", usecols=["sample_id", "session", "dialog_id"]),
       "MELD": pd.read_csv(ROOT + "/data/metadata/meld_utterance_manifest.csv", usecols=["sample_id", "split", "dialogue_id"]),
       "MOSEI": pd.read_csv(ROOT + "/data/metadata/mosei_utterance_manifest.csv", usecols=["sample_id", "video_id", "speaker_source"])}

def block_of(T, df):
    if T.startswith("IEMOCAP"):
        m = df.merge(man["IEMOCAP"], on="sample_id", how="left")
        return m.session.astype(str).to_numpy(), m.dialog_id.astype(str).to_numpy()
    if T == "MELD":
        m = df.merge(man["MELD"], on="sample_id", how="left")
        b = (m.split.astype(str) + "|" + m.dialogue_id.astype(str)).to_numpy()
        return b, b
    if T == "MOSEI":
        m = df.merge(man["MOSEI"], on="sample_id", how="left")
        return m.video_id.astype(str).to_numpy(), m.video_id.astype(str).to_numpy()

rows = []
runs = sys.argv[1:] or ["fuse-audio-video-text", "fuse-audio-video", "audio_wavlm", "video", "text_roberta"]
for run in runs:
    for T in [g for g in groups_for(run) if g in ("IEMOCAP-impro", "IEMOCAP-script", "MELD", "MOSEI")]:
        df = ids_df(T, run)
        sp = df["speaker_group"].astype(str).reset_index(drop=True)
        y = df["label_4"].to_numpy()
        blk, dlg = block_of(T, df)
        assert not pd.isna(blk).any()
        for r in range(10):
            f = speaker_folds(sp, k=5, seed=r); check_disjoint(sp, f)
            te = (f == r % 5).to_numpy()
            pool = np.flatnonzero(~te)
            rng = common.rng_for(f"map-kshot:{T}", r)
            order = [rng.permutation(pool[y[pool] == c]) for c in range(4)]
            tb = set(blk[te]); td = set(dlg[te])
            excl = np.isin(blk, list(tb)) & ~te
            excl_d = np.isin(dlg, list(td)) & ~te
            k5 = np.sort(np.concatenate([o[:5] for o in order])); k25 = np.sort(np.concatenate([o[:25] for o in order]))
            keep = ~te & ~excl
            row = {"run": run, "T": T, "rep": r, "n_total": len(df), "n_test": int(te.sum()), "n_pool": len(pool),
                   "test_speakers": ";".join(sorted(set(sp[te]))) if T.startswith("IEMOCAP") else int(sp[te].nunique()),
                   "n_test_blocks": len(tb), "n_pool_excluded_block": int(excl.sum()), "frac_pool_excluded": round(excl.sum() / len(pool), 3),
                   "n_pool_after": int(keep.sum()), "pool_after_min_class": int(min((y[keep] == c).sum() for c in range(4))),
                   "pool_speakers_after": int(sp[keep].nunique()), "pool_speakers_before": int(sp[~te].nunique()),
                   "K5_in_excluded": int(excl[k5].sum()), "K25_in_excluded": int(excl[k25].sum()),
                   "n_test_in_shared_block_before": int(np.isin(blk[te], list(set(blk[~te]))).sum())}
            if T.startswith("IEMOCAP"):
                row["n_pool_excluded_dialog"] = int(excl_d.sum())
            rows.append(row)
        print(run, T, "done", flush=True)
out = pd.DataFrame(rows)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 500)
print(out.to_string(index=False))
out.to_csv("/tmp/claude-1000/-home-snics-MER-DA/4cd03bf4-8db6-4e9a-a1e4-1871eadc639a/scratchpad/g3_counts_" + "_".join(runs) + ".csv", index=False)
