import sys
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai_data as DD
for c in DD.CORPORA:
    s = {}
    for name, enc in (("enc1", DD.ENC1), ("enc2", DD.ENC2)):
        parts = [set(DD.D.load_group(DD.GROUP[c], k)[0]["sample_id"]) for k in enc.values()]
        s[name] = set.intersection(*parts)
    print(c, "第 1 の集合", len(s["enc1"]), "第 2 の集合", len(s["enc2"]), "共通", len(s["enc1"] & s["enc2"]))
