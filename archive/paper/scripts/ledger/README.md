# 数値台帳の部品（scripts/ledger/）

本文の実験由来の数値は `\NUM{key}` で書き，key の実体は `results/numbers.json` に置く．
`results/numbers.json` は手で書かず，このフォルダのモジュールが MER-DA の結果ファイルから計算した値を `scripts/build_ledger.py` が集めて作る．

## モジュールの約束

- 各モジュール（`data_map.py`，`part1.py`，`part2.py`，`comparison.py`）は `entries() -> dict[str, dict]` を持つ
- 値は `common.entry(value, source, fmt, note)` で作る．`value` は本文に出す数（UAR の差なら「ポイント」．割合を百分率で書くなら 100 倍した値）
- **値は結果ファイル（`../../results/` など）からその場で計算する．`docs/` の文書の値を写さない．**文書の値と食い違ったら `note` に「文書 X の値 Y．差の理由」を書く
- `entries()` は速く（モジュール全体で 30 秒以内）．`make check` が毎回流すためである．特徴の読み込み（`measure/data.py` の `load_group`）など重い計算が要る値は，入力ファイルの大きさと更新時刻を鍵にして `.cache/<モジュール名>.json` に結果を残し，入力が変わったときだけ計算し直す
- 使ってよい結果は `docs/results.md` 2・4 節の決まりに従う．状態が `complete` でない結果（`partial`，`quick (R=1)`）は使わない．オラクルの列は手法の性能として使わない（オラクルと明示した key だけ可）．地図は合併の C をソースのみの C に固定した地図（`results/fixc_map/`）を使う（`CLAUDE.md`）
- 試すとき：`../.venv/bin/python scripts/build_ledger.py --only <モジュール名>`（paper/ で）

## key の付け方

- 小文字の英字・数字・ハイフンだけ．先頭は部分の名前：`data-`・`map-`（data_map），`p1-`（part1），`p2-`（part2），`cmp-`（comparison）
- 種類（特徴の世界）：`a` 音声 WavLM，`ah` 音声 HuBERT，`v` 顔 ResNet-50，`t` 言葉 RoBERTa，`av`，`vt`，`at`，`avt`（融合）．第 2 エンコーダの世界（HuBERT＋CLIP＋DeBERTa）は `w2`
- 世界の中の生徒：`<世界>-<生徒>`．生徒は `a`・`v`・`t`・`f`（融合）．例 `avt-t` は AVT の世界の言葉の生徒，`w2-f` は第 2 の世界の融合の生徒
- K：`k0`・`k1`・`k5`・`k10`・`k25`・`k50`・`kall`
- 範囲：`-lo`・`-hi`．平均 `-mean`．件数 `-n`
- 例：`p1-center-srconly-a`（中心化だけでソースのみの UAR が上がる幅．音声 WavLM），`cmp-oursx-vs-best-avt-f-k0`

## source の書き方

`<MER-DA からの相対パス> | <使った行と列（条件）> | <集計の仕方>`．
例：`results/fixc_map/audio_wavlm/summary/pairs.csv | method 差 diff_mean, K=1, 42 組（同じコーパスの組を含む） | 組の平均`
