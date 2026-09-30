"""結果の進み具合を集めて `results/status.json` と `results/status.md` に書く（`docs/results-index.md` の動的な部分）。

    ./.venv/bin/python exp/results_status.py

`results/` の下の実験フォルダ（`map_<kind>/` など，ターゲットごとの CSV を持つもの）を走査し，
ターゲットごとに終わった反復の数，最後に書かれた時刻，実行の設定行（`#` の JSON）のコミット番号，
いま動いているかを表にする。**UAR などの値は読まない**（列は target と rep だけ読む）。
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
R_FULL = 30
EXPERIMENT_PREFIXES = ("map_", "shift_", "ops_")


def running_kinds() -> dict[str, str]:
    """動いている実験の台本と kind（`--kind` が無ければ既定の audio_wavlm）。"""
    out = subprocess.run(["pgrep", "-af", "python exp/"], capture_output=True, text=True).stdout
    kinds: dict[str, str] = {}
    for line in out.splitlines():
        args = line.split()
        script = next((a for a in args if a.startswith("exp/")), None)
        if script is None or "results_status" in script:
            continue
        kind = args[args.index("--kind") + 1] if "--kind" in args else "audio_wavlm"
        kinds[f"{Path(script).stem}_{kind}"] = line
    return kinds


SHIFT_STAGES = {"gpu": "labeling.csv", "input": "zero.csv", "space": "space_choice.json"}


def read_shift_dir(d: Path, running: dict[str, str]) -> dict:
    """ずれの指標のフォルダ：反復は無いので，段ごとの出力の有無を見る。"""
    done = {st: (d / f).exists() for st, f in SHIFT_STAGES.items()}
    choice = None
    if done["space"]:
        choice = json.loads((d / "space_choice.json").read_text()).get("choice")
    return {"dir": str(d.relative_to(ROOT)), "stages_done": done,
            "state": "complete" if all(done.values()) else "partial",
            "running": any(k.startswith("shift_map_" + d.name[len("shift_"):]) for k in running),
            "space_choice": choice}


def read_dir(d: Path, running: dict[str, str]) -> dict:
    if d.name.startswith("shift_"):
        return read_shift_dir(d, running)
    targets, runs = {}, []
    for p in sorted(d.glob("*.csv")):
        for line in p.read_text().splitlines():
            if line.startswith("# {"):
                runs.append(json.loads(line[2:]))
        reps = pd.read_csv(p, comment="#", usecols=["rep"])["rep"]
        targets[p.stem] = {
            "reps_done": int(reps.nunique()),
            "reps": sorted(int(r) for r in reps.unique()),
            "last_write": time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime)),
        }
    done = [t["reps_done"] for t in targets.values()]
    if done and min(done) >= R_FULL:
        state = "complete"
    elif done and max(done) <= 1:
        state = "quick (R=1)"
    else:
        state = "partial"
    commits = sorted({(r.get("commit", "")[:7] + ("+dirty" if r.get("dirty") else ""))
                      for r in runs})
    return {
        "dir": str(d.relative_to(ROOT)),
        "state": state,
        "running": d.name in running,
        "reps_done_min": min(done) if done else 0,
        "reps_done_max": max(done) if done else 0,
        "reps_target": R_FULL,
        "targets": targets,
        "commits": commits,
        "code_snapshots": sorted(str(p.relative_to(ROOT)) for p in (d / "code").glob("*"))
        if (d / "code").is_dir() else [],
        "summary": str((d / "summary").relative_to(ROOT)) if (d / "summary").is_dir() else None,
    }


def main() -> int:
    running = running_kinds()
    dirs = sorted(d for d in RESULTS.iterdir()
                  if d.is_dir() and d.name.startswith(EXPERIMENT_PREFIXES))
    status = {"generated": time.strftime("%Y-%m-%d %H:%M:%S%z"),
              "experiments": {d.name: read_dir(d, running) for d in dirs}}
    (RESULTS / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=1))

    lines = [f"# 結果の進み具合（{status['generated']} に自動生成．`exp/results_status.py`）", "",
             "意味と使い方は `docs/results-index.md`．この表は値（UAR など）を含まない．", "",
             "| 実験 | 状態 | 実行中 | 終わった反復（最小〜最大／30） | ターゲットごと | コミット |",
             "|---|---|---|---|---|---|"]
    for name, e in status["experiments"].items():
        if "stages_done" in e:                           # ずれの指標
            per = "，".join(f"{st} {'○' if ok else '×'}" for st, ok in e["stages_done"].items())
            if e["space_choice"]:
                per += f"（空間 {e['space_choice']}）"
            lines.append(f"| `{name}` | {e['state']} | {'○' if e['running'] else ''} | − | {per} | |")
            continue
        per = "，".join(f"{t} {v['reps_done']}" for t, v in e["targets"].items())
        lines.append(f"| `{name}` | {e['state']} | {'○' if e['running'] else ''} | "
                     f"{e['reps_done_min']}〜{e['reps_done_max']} | {per} | "
                     f"{'，'.join(e['commits'])} |")
    (RESULTS / "status.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
