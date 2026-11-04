"""本番の設定（分割 5 つ）で label_tables の数を設計の文の数と照らす（ラベルの件数だけ．成績は読まない）．"""
import sys
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import pandas as pd
import dai3
for X in ("AFEW", "IEMOCAP", "MOSEI", "MELD"):
    gc, drops, ph, sc = dai3.label_tables(X)
    gc, drops, ph = pd.DataFrame(gc), pd.DataFrame(drops), pd.DataFrame(ph)
    t = gc.pivot_table(index=["fold", "emotion"], columns="group", values="n")
    print(X, "組・感情ごとの最少", int(gc.n.min()), "組の差の最大", int((t.A - t.B).abs().max()))
    d4 = drops[drops.version == "four"]
    print("  4 感情で塊を除くと外れる回", d4.n_reps_dropping.tolist(), "（塊の数", d4.n_reps.tolist(), "）")
    if X == "MELD":
        print("  候補の対話の共有", ph[ph.subset == "all"].n_shared_dialogues.tolist(), "対話", ph[ph.subset == "all"].n_dialogues.tolist())
        fs = ph[ph.subset == "face_single"]
        print("  顔が競らない候補", (fs.n_half0 + fs.n_half1).tolist())
    if X == "IEMOCAP":
        print(pd.DataFrame(sc).to_string())
