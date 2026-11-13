import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/snics/MER-DA/measure"); sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import data as D
from split import speaker_folds
for g, mods in [("CREMA-D", {"A":"audio_wavlm","V":"video"}), ("RAVDESS", {"A":"audio_wavlm"})]:
    parts = {m: D.load_group(g, k) for m, k in mods.items()}
    ids = sorted(set.intersection(*[set(d["sample_id"]) for d, _ in parts.values()]))
    d0 = parts["A"][0].set_index("sample_id").loc[ids].reset_index()
    for m,(d,X) in parts.items(): print(g, m, X.shape)
    man = pd.read_csv(f"/home/snics/MER-DA/data/metadata/{g.lower()}_utterance_manifest.csv").set_index("sample_id")
    d0["intensity"] = man.loc[d0.sample_id, "emotion_intensity"].to_numpy()
    d0["sent"] = man.loc[d0.sample_id, "transcript"].to_numpy()
    d0["fold"] = speaker_folds(d0["speaker_group"].astype(str), k=5, seed=0).to_numpy()
    print(g, "n", len(d0), "speakers", d0.speaker_group.nunique(), "speaker==group", (d0.speaker_group.astype(str)==d0.speaker_id.astype(str)).all())
    print("test per fold x emotion\n", pd.crosstab(d0.fold, d0.label_4))
    print("speakers per fold", d0.groupby("fold").speaker_group.nunique().to_dict())
    print("min pool per class", min((d0[(d0.fold!=f)&(d0.label_4==k)].shape[0]) for f in range(5) for k in range(4)))
    print(pd.crosstab([d0.label_4], d0.intensity))
    if g=="CREMA-D":
        d0["ieo"] = d0.sent.str.startswith("It's eleven")
        print("share IEO by class", d0.groupby("label_4").ieo.mean().round(3).to_dict())
