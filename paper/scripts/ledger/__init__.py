"""数値台帳の部品（results/numbers.json を作るモジュール群）．

各モジュールは `entries() -> dict[str, dict]` を持ち，key ごとに `common.entry(...)` の結果を返す．
値は MER-DA の結果ファイル（../../results/）から**その場で計算する**．文書（docs/）の値を写さない．
どのモジュールを使うかは MODULES で決める．key は全モジュールで重複してはならない．
"""
MODULES = ["data_map", "part1", "part2", "comparison", "stats_tables", "stats_summary"]
