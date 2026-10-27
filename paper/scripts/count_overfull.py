#!/usr/bin/env python3
"""はみ出し（Overfull \\hbox）を数える．

IEEE Access の雛形と違い，IEEEtran（journal, compsoc）の素の雛形はこのフォルダでは 0 件だった
（2026-10-27 に確かめた）．除外の規則（BENIGN）は空から始め，雛形由来と確かめたものだけを足す．
**本文の書き方で消せるはみ出しは 0 件を維持する．**

使い方: python3 scripts/count_overfull.py main.log
"""
import pathlib
import re
import sys

BENIGN: tuple[re.Pattern, ...] = ()


def main() -> int:
    log = pathlib.Path(sys.argv[1])
    if not log.exists():
        print("0")
        return 0
    n = 0
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Overfull \\hbox" not in line:
            continue
        if any(p.search(line) for p in BENIGN):
            continue
        n += 1
    print(n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
