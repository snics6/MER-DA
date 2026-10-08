"""本筋の解析をまとめて流し，その回に作ったファイルだけをスナップショットに写す（2026-10-08 から）。

本筋と端の仕分けは `docs/findings.md` 3 節。端の解析（`exp/side/`）はここでは流さない。

    ./.venv/bin/python exp/rerun_main.py                    # 地図のまとめから全部流し，スナップショットを作る
    ./.venv/bin/python exp/rerun_main.py --no-summarize     # 地図のまとめ（summarize_map.py）を流さない
    ./.venv/bin/python exp/rerun_main.py --label e12hubert  # スナップショットの名前の末尾（既定は main）
    MERDA_OUT_ROOT=<フォルダ> ./.venv/bin/python exp/rerun_main.py --no-summarize
                                                            # 今の結果を上書きせずに試す（<フォルダ>/explore_relation/ などに書く）

段の順番（後の段は前の段の出力を読む）：

| 段 | 台本 | 解析 | 主な出力 |
|---|---|---|---|
| 0 | `summarize_map.py`（地図の種類ごと） | 地図のまとめ | `results/map_<kind>/summary/` |
| 1 | `explore_decomp.py` | E2 のばらつきの内訳 | `per_pair.csv`，`decomp_*.csv` |
| 2 | `explore_factor.py` | E8 | `e8_*.csv` |
| 3 | `explore_geometry.py` | E9（量の定義） | `e9_*.csv`（E10・E11・E12 が読む） |
| 4 | `explore_shape.py` | E10（量の定義） | `e10_*.csv`（E11・E12 が読む） |
| 5 | `explore_mechanism.py` | E11 | `e11_*.csv` |
| 6 | `explore_pairs.py` | E13 | `e13_*.csv` |
| 7 | `explore_relearn.py` | E12 | `e12_*.csv` |
| 8 | `check_mh.py` | HuBERT での MH の確認 | `results/check_mh/` |

スナップショット：`<出力先>/explore_relation/snapshots/<日時>_<label>/` に，この回に書いたファイル
（更新時刻が段を始めた時刻より後のもの）だけを写す。`check_mh` の出力は下の `check_mh/` に写す。
`manifest.json` に，コミット番号と作業ツリーの変更の有無，種類ごとの地図と学び直しの反復の数，
段ごとの台本・かかった秒数・書いたファイル，始めと終わりの時刻を書く。流した台本の写しは `code/` に置く。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
from pair_table import OUT, OUT_ROOT, RESULTS  # noqa: E402

PY = sys.executable
MAP_KINDS = ("audio_wavlm", "audio_hubert", "video", "text_roberta", "fuse-audio-video",
             "fuse-audio-text", "fuse-video-text", "fuse-audio-video-text")
STEPS = [("E2", "explore_decomp.py"), ("E8", "explore_factor.py"), ("E9", "explore_geometry.py"),
         ("E10", "explore_shape.py"), ("E11", "explore_mechanism.py"), ("E13", "explore_pairs.py"),
         ("E12", "explore_relearn.py"), ("MH-HuBERT", "check_mh.py")]
CHECK_MH = OUT_ROOT / "check_mh"


def written_since(d: Path, t0: float) -> list[Path]:
    """d の直下（スナップショットを除く）で，更新時刻が t0 以降のファイル。"""
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir() if p.is_file() and p.stat().st_mtime >= t0)


def rep_counts() -> dict:
    """種類ごとに，ターゲットごとの終わった反復の数の最小と最大（地図と学び直し）。"""
    out = {}
    for prefix in ("map", "mech", "mech2"):
        for d in sorted(RESULTS.glob(f"{prefix}_*")):
            fs = sorted(d.glob("*.csv"))
            if not d.is_dir() or not fs:
                continue
            n = [pd.read_csv(f, comment="#", usecols=["rep"]).rep.nunique() for f in fs]
            out[d.name] = {"targets": len(fs), "reps_min": int(min(n)), "reps_max": int(max(n))}
    return out


def git_state() -> dict:
    run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--", "exp", "measure"))}


def main() -> int:
    ap = argparse.ArgumentParser(description="本筋の解析をまとめて流す")
    ap.add_argument("--no-summarize", action="store_true", help="地図のまとめ（段 0）を流さない")
    ap.add_argument("--no-snapshot", action="store_true", help="スナップショットを作らない")
    ap.add_argument("--label", default="main", help="スナップショットの名前の末尾")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    started = datetime.now().astimezone()
    log = {"started": started.isoformat(timespec="seconds"), "out_root": str(OUT_ROOT),
           "git": git_state(), "reps": rep_counts(), "steps": []}
    steps = ([] if a.no_summarize else
             [("summary", "summarize_map.py", ["--dir", str(RESULTS / f"map_{k}")])
              for k in MAP_KINDS if (RESULTS / f"map_{k}").is_dir()])
    steps += [(name, script, []) for name, script in STEPS]
    written: dict[str, Path] = {}
    ok = True
    for name, script, args in steps:
        t0 = time.time()
        r = subprocess.run([PY, str(ROOT / "exp" / script), *args], cwd=ROOT)
        files = written_since(OUT, t0) + written_since(CHECK_MH, t0)
        if name == "summary":
            files = written_since(Path(args[1]) / "summary", t0)
        for f in files:
            written[str(f)] = f
        log["steps"].append({"step": name, "script": f"exp/{script}", "args": args, "returncode": r.returncode,
                             "seconds": round(time.time() - t0, 1),
                             "wrote": [str(f.relative_to(OUT_ROOT if f.is_relative_to(OUT_ROOT) else RESULTS))
                                       for f in files]})
        print(f"[{name}] exp/{script} {' '.join(args)}：終了コード {r.returncode}，{time.time() - t0:.0f} 秒，"
              f"{len(files)} ファイル", flush=True)
        if r.returncode != 0:
            ok = False
            break
    log["finished"] = datetime.now().astimezone().isoformat(timespec="seconds")
    log["ok"] = ok

    if not a.no_snapshot:
        snap = OUT / "snapshots" / f"{started:%Y%m%d-%H%M}_{a.label}"
        (snap / "code").mkdir(parents=True, exist_ok=False)
        for f in written.values():
            if f.is_relative_to(OUT):
                shutil.copy2(f, snap / f.name)
            elif f.is_relative_to(CHECK_MH):
                (snap / "check_mh").mkdir(exist_ok=True)
                shutil.copy2(f, snap / "check_mh" / f.name)
        for script in ["rerun_main.py", "pair_table.py", "summarize_map.py"] + [s for _, s in STEPS]:
            shutil.copy2(ROOT / "exp" / script, snap / "code" / script)
        (snap / "manifest.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))
        print(f"スナップショット：{snap}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
