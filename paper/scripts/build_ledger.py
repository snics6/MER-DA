#!/usr/bin/env python3
"""MER-DA の結果ファイルから数値台帳 results/numbers.json を作る．

**なぜ計算で作るか．**本文の数値を手で台帳に写すと，写し間違いと，結果を作り直したときの取り残しが起きる．
そこで各数値を「どのファイルのどの列をどう集計したか」というコードとして scripts/ledger/ に置き，
台帳はそのコードの出力にする．コードは git に入るので，数値の出所はコミットで追える．

- 値も出所も変わらない key は，前回の `verified`（検算日）を残す
- 値か出所が変わった key は，今日の日付にして一覧を出す（本文の読み直しが要る）
- `--check`：台帳を書かず，結果ファイルから作り直した値が台帳と食い違えば 1 で終わる（make check が使う）
- `--only MOD`：1 つのモジュールだけを計算して表示する（書かない）

使い方: ../.venv/bin/python scripts/build_ledger.py [--check | --only MOD]
"""
from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEDGER = ROOT / "results" / "numbers.json"
sys.path.insert(0, str(ROOT / "scripts"))

from ledger import MODULES  # noqa: E402

REQUIRED = ("value", "format", "source", "verified")


def collect(names):
    out, errors = {}, []
    for name in names:
        mod = importlib.import_module(f"ledger.{name}")
        for key, ent in mod.entries().items():
            if key in out:
                errors.append(f"key '{key}' が {out[key]['module']} と {name} の両方にある")
                continue
            if any(k not in ent for k in REQUIRED):
                errors.append(f"key '{key}'（{name}）に {REQUIRED} のどれかが無い")
                continue
            ent = dict(ent)
            ent["module"] = name
            out[key] = ent
    return out, errors


def fmt(e):
    """gen_numbers_tex.py と同じ規則で値を文字にする（printf 形式か str.format 形式）．"""
    f, v = e.get("format"), e["value"]
    if not f:
        return str(v)
    return f % v if "%" in f else f.format(v)


def same(a, b):
    return a.get("value") == b.get("value") and a.get("source") == b.get("source") and a.get("format") == b.get("format")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only")
    args = ap.parse_args()

    names = [args.only] if args.only else MODULES
    new, errors = collect(names)
    if errors:
        print("build_ledger: " + "\n  ".join(["誤り:"] + errors), file=sys.stderr)
        return 1
    if args.only:
        for k in sorted(new):
            e = new[k]
            print(f"{k} = {fmt(e)}   ← {e['source']}" + (f"  [note: {e['note']}]" if e.get("note") else ""))
        print(f"build_ledger: {args.only} の {len(new)} 件（書いていない）")
        return 0

    old = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else {}
    changed = [k for k in new if k in old and not same(new[k], old[k])]
    added = [k for k in new if k not in old]
    removed = [k for k in old if k not in new]
    for k in new:
        if k in old and same(new[k], old[k]):
            new[k]["verified"] = old[k]["verified"]

    if args.check:
        if changed or added or removed:
            print(f"build_ledger: 結果ファイルから作り直した台帳が results/numbers.json と食い違う"
                  f"（変わった {len(changed)}，増えた {len(added)}，消えた {len(removed)}）．make ledger で作り直し，本文を読み直す．")
            for k in changed:
                print(f"  変わった: {k}: {old[k]['value']} → {new[k]['value']}")
            for k in added:
                print(f"  増えた: {k}")
            for k in removed:
                print(f"  消えた: {k}")
            return 1
        print(f"build_ledger: 台帳 {len(new)} 件は結果ファイルと一致する．")
        return 0

    LEDGER.write_text(json.dumps(dict(sorted(new.items())), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"build_ledger: {len(new)} 件を書いた（変わった {len(changed)}，増えた {len(added)}，消えた {len(removed)}）．")
    for k in changed:
        print(f"  変わった: {k}: {old[k]['value']} → {new[k]['value']}（本文を読み直す）")
    for k in removed:
        print(f"  消えた: {k}（本文で使っていれば make が落ちる）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
