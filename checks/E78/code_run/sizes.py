"""群の発話数と E39 の設定が CORAL の格子に入るか（E78 関門 1）．CPU だけ。"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "exp")); sys.path.insert(0, str(ROOT / "measure"))
from data import groups_for
from geom_e0 import settings
from multimodal_distill import WORLDS
import compare_dai2 as E
import pandas as pd
from data import load_any
for name in ["audio_wavlm", "video", "text_roberta", "fuse-audio-video-text", "fuse-audio-video"]:
    gs = groups_for(name)
    print(name, {g: len(load_any(g, name)[0]) for g in gs})
    studs = WORLDS[name] + [name] if name in WORLDS else [name]
    for T in gs:
        for k in studs:
            g_, rs, rm, cb, cp, _ = settings(k, T)
            ok = (rm in E.CORAL_RHO, cp in E.CORAL_CP, cb in E.CORAL_CB)
            print(f"  {T:15s} {k:22s} rho_m={rm} cp={cp} cb={cb} in_grid={ok}")
