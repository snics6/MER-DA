"""論文に使う結果の置き場から，git に入れる集計と記録を選び，results/MANIFEST.tsv に一覧を書く（2026-11-15 の整理）．

選び方（RULE）：論文に使う置き場（DIRS）の中で，単位ごとの大きい出力の下位フォルダ（HEAVY）の外にある
集計の表と記録（拡張子 TEXT，1 本 MAX_MB 以下）と，ソースの分類器（下位フォルダ source の npz）を git に入れる．
それ以外（単位ごとの出力など）は git の外に置き，sha256 と大きさだけを results/MANIFEST_outside_git.tsv に書く．
使い方：python tools/collect_results.py [--hash-outside]   （DAI の直下で．results/ は DAI/results）
"""
import argparse, fnmatch, hashlib, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
DIRS = ["dai1", "dai2", "dai3", "dai11_probe", "dai12_probe", "dai13_probe", "dai14", "dai15", "dai16_probe", "dai16_probe_k10",
        "dai17_probe", "dai17_probe_k10", "dai18_probe", "dai20_eval", "dai21_eval5",
        "e75", "e76", "e76_analysis", "e77_audio_wavlm", "e77_video", "e77_text_roberta", "e77_fuse-audio-video", "e77_fuse-audio-video-text",
        "e78_audio_wavlm", "e78_video", "e78_text_roberta", "e78_fuse-audio-video", "e78_fuse-audio-video-text", "e80",
        "e80c_fuse-audio-video", "e80c_fuse-audio-video-text", "e79", "e79_p4",
        # 前半の解析が入力に読む旧版の結果（reanalysis_e75.py の DEFAULTS，relearn_e76.py・analyze_e76.py・compare_dai*.py の既定）
        "fixc_map", "fixc_analysis", "fixc_text", "explore_relation", "text30", "map_*", "shift_*", "tgt_scores_*", "transfer_scores_*",
        "mech_*", "interventions_*", "compare_*", "multisource_scores_*"]
HEAVY = {"units", "rows", "rowsA", "students", "reweight", "srcdb", "base", "measure", "code"}
# 候補ごとの行の大きい表しか無い置き場（E79 の入力）．集計は e79・e79_p4 に入る．git には入れない
OUTSIDE_ONLY = ("e77_", "e78_", "e80c_")
# 前半の入力の旧版の結果（E75 より前の実験の出力）．2 MB 以下の表だけを入れる
OLD_INPUTS = ("fixc_analysis", "fixc_text", "explore_relation", "text30", "map_", "shift_", "tgt_scores_", "transfer_scores_", "mech_",
              "interventions_", "compare_", "multisource_scores_")
EXPLORE_RELATION_KEEP = ("e32", "snapshots/20261011-1524_e29/", "snapshots/20261011-0139_e28t9/", "kstar_cluster.csv", "e39_tests.csv")
TEXT = {".csv", ".json", ".jsonl", ".md", ".txt", ".tsv"}
MAX_MB = 20
# 論文の本文が直接引く表で，MAX_MB を超えるもの（2026-11-15．GitHub の 1 本 100 MB の上限より小さい）
PAPER_CITED_LARGE = {"e79/judge/units.csv", "e79_p4/judge/units.csv"}


def expand():
    out = []
    for pat in DIRS:
        out += sorted(p for p in RES.glob(pat) if p.is_dir() and not p.is_symlink() or (p.is_symlink() and p.resolve().is_dir()))
    return out


def tracked(p: Path, top: Path) -> bool:
    parts = p.relative_to(top).parts[:-1]
    if p.relative_to(RES).as_posix() in PAPER_CITED_LARGE:
        return True
    if any(s in HEAVY for s in parts) or top.name.startswith(OUTSIDE_ONLY):
        return False
    size = p.stat().st_size
    if top.name.startswith(OLD_INPUTS):
        rel = p.relative_to(top).as_posix()
        if top.name == "explore_relation" and not rel.startswith(EXPLORE_RELATION_KEEP):
            return False
        return p.suffix in TEXT and size <= 2e6
    if p.suffix in TEXT and size <= MAX_MB * 1e6:
        return True
    return p.suffix == ".npz" and "source" in parts and size <= MAX_MB * 1e6


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash-outside", action="store_true", help="git の外のファイルの sha256 も計算する（数十分）")
    a = ap.parse_args()
    ins, outs = [], []
    for top in expand():
        for dp, dn, fn in os.walk(top):          # 置き場の中のリンク（dai13_probe の summary_seeds*/rows）はたどらない．同じファイルを 2 回数えないため
            for n in fn:
                p = Path(dp) / n
                (ins if tracked(p, top) else outs).append(p)
    with open(RES / "MANIFEST.tsv", "w", encoding="utf-8") as fh:
        fh.write("path\tbytes\tsha256\n")
        for p in sorted(ins):
            fh.write(f"{p.relative_to(ROOT).as_posix()}\t{p.stat().st_size}\t{sha(p)}\n")
    with open(RES / "MANIFEST_outside_git.tsv", "w", encoding="utf-8") as fh:
        fh.write("path\tbytes\tsha256\n")
        for p in sorted(outs):
            fh.write(f"{p.relative_to(ROOT).as_posix()}\t{p.stat().st_size}\t{sha(p) if a.hash_outside else ''}\n")
    tin, tout = sum(p.stat().st_size for p in ins), sum(p.stat().st_size for p in outs)
    print(f"git に入れる {len(ins)} 本 {tin/1e6:.1f} MB，git の外 {len(outs)} 本 {tout/1e9:.1f} GB")


if __name__ == "__main__":
    main()
