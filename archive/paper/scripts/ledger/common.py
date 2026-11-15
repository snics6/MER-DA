"""台帳のモジュールが共通に使う道具．"""
from __future__ import annotations

import datetime
import pathlib

# paper/scripts/ledger/common.py → MER-DA
MERDA = pathlib.Path(__file__).resolve().parents[3]
RESULTS = MERDA / "results"
SNAP = RESULTS / "explore_relation" / "snapshots"
TODAY = datetime.date.today().isoformat()


def _py(v):
    """numpy の数を JSON に書ける Python の数にする．"""
    try:
        import numpy as np  # noqa: F401
        if hasattr(v, "item"):
            v = v.item()
    except ImportError:
        pass
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    return float(v)


def entry(value, source: str, fmt: str = "%.1f", note: str | None = None) -> dict:
    """台帳の 1 項目．

    value  : 本文に出す数（単位は本文と同じ．UAR の差なら「ポイント」）
    source : 結果ファイルの場所（MER-DA からの相対パス）と，列・条件・集計の仕方
    fmt    : 表示の書式（printf 形式）．整数は "%d"
    note   : 文書の値との食い違いなど，検算で気づいたこと
    """
    d = {"value": _py(value), "format": fmt, "source": source, "verified": TODAY}
    if note:
        d["note"] = note
    return d


def rel(path: pathlib.Path) -> str:
    """MER-DA からの相対パス（source に書く形）．"""
    return str(pathlib.Path(path).resolve().relative_to(MERDA))
