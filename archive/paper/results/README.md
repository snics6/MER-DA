# 数値台帳 `numbers.json`

本文の実験由来の数値は `\NUM{key}` でしか書かない．key の実体はこのファイルに置き，`make numbers` で `numbers.tex` を生成する（生成物もコミットする）．

各 key は次の 4 項目を持つ．

```json
"gain_k1_audio": {"value": 10.5, "format": "%.1f", "source": "results/fixc_map/map_audio_wavlm/summary/pairs.csv, mean of gain_k1 over 42 pairs (iters 0-29)", "verified": "2026-10-27"}
```

- `value`：数値．`format` で表示の桁を決める（省略時は `str(value)`）
- `source`：MER-DA の結果ファイルの場所と，列・条件・集計の仕方．**`docs/results.md` 3 節の値を写すのではなく，結果ファイルから取り直す**（`docs/results.md` 2 節の決まり）
- `verified`：結果ファイルで検算した日付

`docs/results.md` 4 節にある「使ってよい状態」（`complete`）の結果だけを使う．オラクルの列は手法の性能として使わない．
