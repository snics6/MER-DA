# 計算の後の点検：種 r = 0, 1 の結果ファイルと 2 つの種の集計（DAI-13 の下調べ，2026-11-10 02:3x，単位の段のコミット 4932d6c（r = 0）と 7edef9c（r = 1），集計の段のコミット 494eaf8）

点検した対象は次のとおりである．`results/dai13_probe/rows/` の 480 ファイル（r = 0 と r = 1 の `*.npz`・`*_lw.csv`・`*_lab.csv`・`*_layers.csv`．種ごとに各 60 本），2 つの種の集計 `results/dai13_probe/summary_seeds2/` の 16 ファイル，比較のための `summary_seeds1/`，`provenance.json`，`provenance_log.jsonl`，`units_time_20261110-022521.csv`，ログ `logs/dai13_probe_r1_units.log`・`logs/dai13_probe_r1_sum.log`・`logs/dai13_probe_r1_driver.log`・`logs/dai13_probe_r1.done`．台本 `DAI/exp/dai13_probe.py`（sha256 `19960ba6b4257a27b0e9ef757bda776dd7d9e68229e8a69b133877fd210a7b1f`．今のファイルも同じ），起動の台本 `DAI/run/dai13_probe_r1.sh`，設計 `DAI/docs/実験13_下調べ_設計.md`（第 15 版，`54d8245c…`）．
点検は CPU だけで行い，GPU は使っていない．検算の台本は `/tmp/claude-1000/-home-snics-MER-DA/b1f27254-32c7-45f0-ac5a-dee1c5e3ec8a/scratchpad/dai13_post/check1.py`・`check3.py`〜`check7.py`（と `check5b.py`）に置いた．r = 0 の点検の台本（同じ場所の `dai13_post_r0/`）を，種と集計の出力先だけ替えて写したものと，種が 2 つの集計の値を取り直す台本 `check6.py`（書き直した）である．どの台本も，個数・鍵・形・型・不一致の数・有限性・所要と，取り直した値と表の値の差の絶対値だけを印字し，UAR・差・区間の値と判定の文字列（`verdict`，`shuf_case`，`note_*`）は印字していない．`results/`・`data/`・`docs/`・`DAI/docs/` のファイルは書き換えていない．

## 判定
通す．r = 1 の 60 単位の npz と 3 本の csv がそろって r = 0 と合わせて 120 単位で重複が無く，鍵の集合は c ごとに r = 0 と同じで，有限でない値と int8 の範囲の外は 0 で，台本の sha256 は 1 つで，発話・半分・部分集合は DAI-12 の r = 1 の保存と一致し，照合の整数の鍵の不一致は 0 本で，`pair` の許しを超えた単位はログの警告 1 行と同じ 1 単位であり，格子の全点の判定は選んだ版の判定と全件一致し，`summary_seeds2/` の表 16 本は `summary_seeds1/` と同じ行の構造で，v_init を足さず `lo`・`hi` が `_raw` と一致し，行から独立に取り直した点推定と区間と R1 の数え上げが表と浮動小数の丸めの範囲で一致した．記録の不足が 6 件ある（下の「指摘」．どれも結果の文で直せる）．

## 1．単位の集合，ファイル，重複（`check1.py`，`check3.py`，`check7.py`）

| 確かめたこと | 結果 |
|---|---|
| `rows/` のファイル | 480 本．種 × 種類ごとに 60 本（`r0.npz`，`r0_lw.csv`，`r0_lab.csv`，`r0_layers.csv`，`r1.npz`，`r1_lw.csv`，`r1_lab.csv`，`r1_layers.csv`）．この 8 種類の外の名前のファイルは 0 本 |
| `rows/*_r1.npz` の単位の名前 | 60 個，重複 0．`results/dai12_probe/rows/*_r1.npz`（60）と `results/dai1/pseudo/units/*_r1_p3.npz`（60）の集合と一致 |
| 名前の形 | 全部 `<T>_<c>_f[0-4]_r1` で T ≠ c．c ごとに 15 単位，T ごとに 15 単位 |
| r = 1 の `_lw.csv`・`_lab.csv`・`_layers.csv` | 各 60 本．名前の集合は npz と同じ |
| npz の `unit`・`T`・`c`・`seed`（= 1）と名前 | 不一致 0 |
| 単位のログ `logs/dai13_probe_r1_units.log` | 62 行．「続き：残り 60 単位」1 行，「単位」の行 60（重複 0，npz の集合と同じ），「警告」の行 1．ほかの行 0．Traceback・Error・例外・NaN・収束・メモリ不足の語は 0 件．集計のログ `logs/dai13_probe_r1_sum.log` は pandas の `PerformanceWarning` 3 件と `Pandas4Warning` 1 件（r = 0 の集計と同じ 4 件）だけで，例外は 0 件 |
| 発話の対応（r = 1 の 60 単位） | `pool_sample_id`・`test_sample_id`・`pool_y`・`test_y`・`pool_cluster`・`test_cluster`・`kpos_K{1,5,10,25}`・`pool_half`・`pool_sub_*`・`pool_shuf_src_A`・`pool_none_K*_A` が DAI-12 の r = 1 の保存と一致（不一致 0） |
| 漏れの検算（r = 1） | 60 単位とも，候補とテストで共有する `sample_id` 0 個，塊 0 個．2 つの半分が共有する塊 0 個 |
| `half_counts`（r = 1） | 60 単位とも `pool_half` と `pool_y` から数え直した本数と一致 |

## 2．保存の鍵（`check1.py`）

| 確かめたこと | 結果 |
|---|---|
| 鍵の数 | IEMOCAP・MELD を c にした単位は 810 鍵，AFEW・MOSEI を c にした単位は 809 鍵．c ごとに鍵の集合は 1 通りで，**c ごとの鍵の集合は r = 0 と同じ**（4 つの c とも一致） |
| 浮動小数の配列 | 16,740 本（float16 10,620，float32 6,120）．有限でない値を含む配列は 0 本 |
| int8 の配列 | 30,510 本．判定の鍵で 0〜3 の外の値を含む配列 0 本，`pool_half`・`pool_sub_*` で 0・1 の外 0 本 |
| 層の重みの和 | `alpha_lw_`・`alphapre_lw_`・`alphafix_lw_`・`alpha_kshot_` の 6,120 本で，和と 1 の差の最大 2.4e−7 |
| `script_sha256` | r = 1 の 60 単位で 1 つ（`19960ba6…`）．r = 0 と同じ |
| `commit` | r = 1 は 7edef9c（60 単位で 1 つ）．**r = 0 の 4932d6c と違う**（指摘 1） |
| `commit11`・`commit12` | r = 1 も 67e3553・e9f66c9 で r = 0 と同じ．e9f66c9 は DAI-12 の r = 1 の npz の `commit` と同じ |

## 3．台本の照合（`check3.py`）

| 確かめたこと | r = 1 の結果 |
|---|---|
| `mismatch` の鍵の数 | 60 単位とも 64 鍵．有限でない値 0，`frac` の鍵で 0〜1 の外 0 |
| 整数の鍵（`K*_pool_none` 5，`K*_test_none` 5，`K*_pool_teacher_*` 15，`pool_half`，`pool_shuf_src`） | 1,620 個（60 × 27）．0 でないもの 0 個．許しを超えるもの 0 個 |
| 添字 24 の生の特徴と既存の音声の特徴の差 | 最大 9.2e−4（止める値 1e−2）．標準化した後の差の最大 1.6e−2（記録のみ） |
| `final_vs_dai12_lince_pair_over` が立った単位 | 1 単位（`MOSEI_IEMOCAP_f0_r1`）．r = 1 のログの「警告」の 1 行の単位と一致．r = 0 の 2 単位（ログの 2 行と一致．`code_post_r0.md`）と合わせて 3 単位 |
| `pair` で許しを超えた (単位, K) | UAR の差 > 5.0 が 1 個（`MOSEI_IEMOCAP_f0_r1` の K = 0），判定の不一致の割合 > 0.25 が 0 個．r = 0 の 3 個と合わせて 4 個．`summary_seeds2/criteria.csv` の `final_vs_dai12_lince_uar_over_n_pair` の 4 行の和 4，`…_frac_over_n_pair` の和 0 と一致 |
| `self` の `final` と DAI-12 の `lince` | 判定の不一致の割合の最大 0.082（許し 0.25），候補の上の UAR の差の絶対値の最大 2.04（許し 5.0）．`criteria.csv` の `…_over_n_self`・`…_frac_over_n_self` の和はどちらも 0 |
| `pair_shuf`（記録のみ） | 判定の不一致の割合 > 0.25 が 41 個，UAR の差 > 5.0 が 5 個（r = 1 の 300 個の (単位, K) のうち）．r = 0 と合わせて 95／600 と 12／600 |
| `checks_units.csv`（seeds2） | 7,680 行（120 × 64），(unit, key) の重複 0 |

## 4．格子の全点の判定（`check4.py`）

台本の `check_grid_consistency`（`DAI/exp/dai13_probe.py:326-338`）と同じ確認を，**r = 1 の 60 単位すべて**について保存した `grid_pred_*` と `conv` の選んだ点と候補の判定から CPU で行った．

| 腕 | 比べた判定の本数 | 違った本数 |
|---|---|---|
| `lw`（入れ子の前）・`final`・`unif`・`final_lw`・`lw`（入れ子の後）・入れ子の後と元の版 | 各 5,307,705 | 0 |
| `final_lab`・`unif_lab`・`lw_lab`（入れ子の後） | 各 353,847 | 0 |
| `lw_lab`（入れ子の前．置き換わらなかった半分） | 350,901 | 0 |
| `lw_lab_fixalpha`（K = 0, 25） | 707,694 | 0 |

全部が 0 の切片：8 つの腕の 192,000 切片のうち 47 個で，全部が wd 10 の `pair_shuf`（`lw` 7，`final` 14，`unif` 9，`finallw` 17）である（r = 0 は 76 個で同じ種類）．0 以外の 1 つの番号だけの切片が 1 個ある（`IEMOCAP_MELD_f0_r1` の `unif`，wd 10，最後の保存点，`pair_shuf` の K = 10）．これも wd 10 の `pair_shuf` である（指摘 5）．

## 5．集計の表（`check5.py`，`check5b.py`）

`summary_seeds2/` の 16 ファイル（`cells_lw.csv`，`cells_lab.csv`，`cells_layers.csv`，`diffs.csv`，`criteria.csv`，`v_init.csv`，`alpha_summary.csv`，`se_summary.csv`，`lw_units.csv`，`lab_units.csv`，`layers_units.csv`，`uar_units.csv`，`checks_units.csv`，`halves_units.csv`，`summary_meta.json`，`provenance.json`）と `rows` へのリンクがそろい，ほかのファイルは無い．16 本の csv の列名は `summary_seeds1/` と同じである．

| 表 | 行 | 設計と台本から出る数 | 重複 | 有限でない値 |
|---|---|---|---|---|
| `lw_units.csv` | 1,800 | 120 単位 × 出所 3 × K 5．単位 120（r = 0 60，r = 1 60），`seed` は {0, 1} | (unit, teacher, K) 0 | NaN は Q4・Q5 の 24 列だけで，`pair` かつ K ≥ 1 でない 1,320 行と完全に一致 |
| `lab_units.csv` | 600 | 120 × K 5 | (unit, K) 0 | NaN は 2 列だけで，K ∉ {0, 25} の 360 行と完全に一致 |
| `layers_units.csv` | 15,000 | 120 × 層 25 × K 5 | 0 | 0 |
| `cells_lw.csv` | 60 | c 4 × 出所 3 × K 5．`n_units` は全行 30 | 0 | NaN は 44 行 × 24 列で，seeds1 と同じ場所 |
| `cells_lab.csv` | 20 | c 4 × K 5．`n_units` は全行 30 | 0 | NaN は K ∉ {0, 25} の 12 行 × 2 列 |
| `cells_layers.csv` | 500 | c 4 × 層 25 × K 5 | 0 | 0 |
| `alpha_summary.csv` | 128 | seeds1 と同じ（`lw` 60，`lw_pre` 60，`lw_lab` 4，`lw_pre_lab` 4） | 0 | 0 |
| `v_init.csv` | 60 | c 4 × 出所 3 × K 5．`n_folds` は全行 5 | 0 | 0 |
| `uar_units.csv` | 17,160 | 120 × テストの判定の鍵 143．種ごとに 8,580 | (c, T, fold, seed, key) 0 | 0．0〜100 の外 0 |
| `checks_units.csv` | 7,680 | 120 × 64 | 0 | 0 |
| `halves_units.csv` | 120 | 120（種ごとに 60） | 0 | 0 |
| `diffs.csv` | 1,756 | r = 0 の版と同じ 4 × 349 ＋ 60 × 6 | (c, teacher, K, where, quantity, subset) 0 | 下 |
| `se_summary.csv` | 168 | `diffs.csv` の `subset == all` の (c, quantity) の組の数 168 | 0 | 0 |
| `criteria.csv` | 40 | 規則 10 種 × c 4（`Q3/Q2` の 4 行も seeds1 と同じ） | — | 規則の中で一部だけ NaN の列は 0 |

- **行の構造**：`diffs.csv`・`cells_lw.csv`・`cells_lab.csv`・`cells_layers.csv`・`alpha_summary.csv`・`v_init.csv`・`se_summary.csv`・`criteria.csv` の 8 本で，識別の列の組の集合が seeds1 と完全に同じである．`diffs.csv`・`cells_lw.csv`・`criteria.csv` の NaN の場所も，v_init に依る列（`v_init`・`lo`・`hi`・`se`・`p`）を除いて seeds1 と同じである．
- `diffs.csv` の `where` は `pool` 1,516・`test` 240，`subset` は `all` 1,396，6 つの部分集合が各 60．`n_units` は全行 10（分割 5 × 種 2．seeds1 は 5）．
- `est`・`lo`・`hi`・`se`・`p`・`lo_raw`・`hi_raw`・`se_raw`・`p_raw`・`quantile` の NaN は 0 個，inf は 0 個．NaN は `v_init` の列の 108 行だけで，全部 `teacher == labels`（seeds1 と同じ）．
- **v_init**：`diffs.csv` の `v_init_applied` は全行で偽，`se_adjusted` は 0 行．`lo == lo_raw`・`hi == hi_raw`・`se == se_raw`・`p == p_raw` が全行で厳密に成り立つ．台本 `:944-946` は (c, 出所, K, 分割) ごとの種の数が 1 つでも 2 未満なら v_init を足す．全組で 2 つそろったので足していない（設計 4.1 節のとおり）．`v_init.csv` は記録として 60 行書かれている．
- **`method`**：`jk+seed` 1,347 行（AFEW・IEMOCAP・MELD），`boot_cl_shared+seed` 349 行（MOSEI の候補），`boot_cl+seed` 60 行（MOSEI のテスト）で，全行が `+seed` を含む．ただしこの名前は種が 1 つの seeds1 でも同じである（台本の名前は種の数で変わらない．指摘 6）．種の項が実際に区間に入ることは 6 節で行から確かめた．
- `quantile` は `boot_*` で 1.96，`jk+seed` で 2.16〜3.18（seeds1 と同じ範囲）．
- `se_raw` が 0 の行は 35 行で，全部 `teacher == self`（`delta_uar` 20 行，入れ子の後の `layers_minus_final` 10 行，`lw_vs_final_test` 5 行）．35 行とも `est` は 0．
- `summary_meta.json`：`v_init_applied` 偽，`n_seeds` 2，`n_units` 120，`row_commits` は 4932d6c と 7edef9c の 2 つ，`script_sha_rows` と `script_sha_summarize` はどちらも `19960ba6…`．
- 集計のログの最終行は「集計：単位 120（種 [0, 1]），差の行 1756，規則の行 40，v_init を足した False」．

## 6．集計の値を行から取り直す（`check6.py`）

台本と共通の部品の関数を使わずに，npz の判定・`pool_y`・`test_y`・塊・`kpos`・部分集合の印から数え直した．(分割, 種) ごとに 3 つの T の塊ごとの感情ごとの正解の数と件数を足して 3 で割り（台本の `_add`・`_finish` と同じ），UAR の差を出し，種で平均してから分割で平均した．jackknife は，候補の塊の和集合の 1 つずつについて，その塊を含む (分割, 種) では除いた残りの UAR（30 本未満か感情が欠けると無効），含まない (分割, 種) では全体の差を使い，分散 (G−1)/G Σ(θ_g − θ̄)² に種の分散 Σ_f s²_f／2 ÷ 25 を足した．分位点は表の `quantile` の列を使った．印字したのは表の値との差の絶対値と一致の真偽だけである．

| 取り直したもの | 表 | 結果 |
|---|---|---|
| 主の量 `layers_minus_final`（`pair`，候補，`all`），4 コーパス × K = 10 の点推定 | `diffs.csv` の `est` | 一致（差 ≤ 5.8e−15）．4 コーパスとも，分割の中の 2 つの種の差の分散の和が 0 より大きい（種の項が区間に入る） |
| MELD，`pair_shuf`，K = 5，`lw_minus_unif`（候補） | 同 | 一致（差 8.4e−15） |
| MOSEI，`pair`，K = 25，`layers_minus_final_pre`，部分集合 `channel` | 同 | 一致（差 1.2e−14） |
| IEMOCAP，全ラベル，K = 1，`lab_layers_minus_final` | 同 | 一致（差 0） |
| AFEW，`self`，K = 25，`layers_minus_final_pre`（R1c の量） | 同 | 一致（差 0） |
| MELD，`pair`，K = 5，`lw_vs_final_test`（テスト） | 同 | 一致（差 6.2e−15） |
| AFEW・IEMOCAP・MELD の主の量の K ∈ {1, 5, 10, 25}（12 行）の `se_raw` と `lo` を jackknife と種の分散から作り直す | `diffs.csv` の `se_raw`・`lo` | 一致（`se` の差 ≤ 4.4e−15，`lo` の差 ≤ 1.3e−14）．除かれた分割 0 |
| R1 の `n_K_up`（規則：`lo` > 0 かつ `est` ≥ 1.0，K ∈ {1, 5, 10, 25}，`pair`，`layers_minus_final`，`all`） | `criteria.csv` の R1 の行 | 4 コーパスとも表の `lo`・`est` から数えた数が一致．AFEW・IEMOCAP・MELD は作り直した `lo`・`est` から数えた数も一致 |
| `cells_lw.csv` の 4 列（`uar_lw`，`uar_final`，`layers_minus_final`，`final_vs_dai12_lince_uar_absdiff`）× 60 セルと `lw_units.csv` の 30 単位の平均 | `cells_lw.csv` | 一致（差 ≤ 7.1e−15） |
| `cells_lw.csv` の `uar_lw`・`uar_final` の 3 セル（IEMOCAP `pair` K = 5，MOSEI `self` K = 25，AFEW `pair_shuf` K = 0）を npz の判定から 30 単位ぶん | `cells_lw.csv` | 一致（差 ≤ 7.1e−15） |
| `v_init` の 60 行（`lw_units.csv` の `uar_final` と DAI-12 の `rows/*_nl.csv` の `uar_lince` から．対応の取れない行 0） | `v_init.csv` | 一致（差 ≤ 1.0e−16） |
| AFEW，`lw`，`pair`，K = 10 の 25 層の重みの平均（30 単位 × 半分 2） | `alpha_summary.csv` | 一致（差 ≤ 1.8e−8．float32 の丸め） |

## 7．記録の整合（`provenance_log.jsonl`，`provenance.json`，`summary_seeds1/`）

| 記録 | 内容 |
|---|---|
| `provenance_log.jsonl` | 4 行．1 行目 r = 0 の `units`（4932d6c，20:04:10），2 行目 r = 0 の `summarize`（7edef9c，23:12:25），**3 行目 r = 1 の `units`**（7edef9c，dirty 偽，23:17:37，argv `--units *_r1 --resume`），**4 行目 2 つの種の `summarize`**（494eaf8，dirty 偽，02:25:24，argv `--n-expected 120`） |
| 行どうしの違い | 1 行目と 3 行目の違いは `argv`・`commit`・`design_sha256`（第 14 版 → 第 15 版）・`time` だけ．2 行目と 4 行目の違いは `argv`・`commit`・`time` だけ．台本と共通の部品 8 本，凍結の JSON，全層の特徴 4 本の sha256，定数，ソフトの版は 4 行で同じ |
| コミットの差 | 4932d6c → 7edef9c は設計の文（9 行）と `docs/seen-data.md`（3 行）だけ（`code_post_r0.md` 7 節）．7edef9c → 494eaf8 は `DAI/checks/DAI-13/code_post_r0.md`，結果の文の下書き，新しい台本 `DAI/exp/table_dai13_cells.py`（集計の表を読むだけの別の台本），`docs/seen-data.md` の 4 本を足しただけで，`DAI/exp/dai13_probe.py` と部品は変わっていない．494eaf8 から今の作業木まで，`DAI/exp`・`DAI/run`・`measure`・`features`・`data/metadata`・`DAI/docs` に変更は無い |
| `results/dai13_probe/provenance.json` | `summary_seeds2/provenance.json` と同じ内容で，4 行目に `summary_meta.json` の 6 鍵と `summary_dir`（`…/summary_seeds2`）を足したもの．`summary_seeds1/provenance.json` とは違う |
| `summary_seeds1/` | 中のファイルの時刻は 23:13:37〜23:16:05 で，23:16:06 より後に書かれたファイルは 0 本（リンク `rows` は 23:12:29）．`summary_meta.json` は `n_units` 60，`n_seeds` 1，`v_init_applied` 真のまま．`cells_lw.csv`・`cells_lab.csv` の `n_units` は全行 15（seeds2 は 30） |
| ドライバのログ | 開始 23:17:35，単位の段の終わり 02:25:22，集計の段の終わり 02:31:36．`logs/dai13_probe_r1.done` は 02:31:36 の空のファイル．`.failed` は無い |

## 8．所要と GPU（`check7.py`，ログ）

- r = 1 の単位の段：23:17:35 → 02:25:22（3 時間 7 分 47 秒．`units_time_20261110-022521.csv` の秒の和は 3.13 時間，60 行，単位の集合は r = 1 の npz と同じ）．
- 単位あたりの秒（c ごとの最小〜最大）：AFEW 50〜57，IEMOCAP 91〜97，MELD 160〜194，MOSEI 419〜426（r = 0 とほぼ同じ）．
- 2 つの種の集計の段：02:25:24 → 02:31:36（6 分 12 秒）．設計 6 節の見込み 30 分以内の中．
- GPU：`max_memory_allocated` の最大 11.11 GB，`max_memory_reserved` の最大 12.59 GB（r = 0 と同じ）．6 割の上限 14.7 GB の中．

## 指摘

### [記録] 1．行のコミットが種で違い（r = 0 は 4932d6c，r = 1 は 7edef9c），2 つの種の集計のコミットは 494eaf8 である
- 重さ：記録する
- 該当箇所：各 npz の `commit`，`results/dai13_probe/provenance_log.jsonl` の 1・3・4 行目，`summary_seeds2/summary_meta.json` の `row_commits`
- 問題：台本の sha256 は 3 つの段で `19960ba6…` の 1 つで，コミットの差は文書と別の台本だけである（7 節）．計算は同じ台本でされている．ただし `provenance.json` だけを見ると計算のコミットを 494eaf8 と読み，npz の 1 本だけを見ると 4932d6c か 7edef9c のどちらかを読む．
- 計算の後で直せるか：直せる（記録は全部ある）
- 求める対応：結果の文の冒頭に「単位の段は r = 0 がコミット 4932d6c（設計 第 14 版），r = 1 が 7edef9c（設計 第 15 版），集計の段は 494eaf8．3 つの間の差は文書と別の台本 `table_dai13_cells.py` だけで，`dai13_probe.py` と共通の部品の sha256 は同じ（`provenance_log.jsonl` の 4 行）」と書く．

### [記録] 2．`results/dai13_probe/provenance.json` は seeds2 の写しに置き換わった．`summary_seeds1/rows` は 120 単位の入った共有の `rows/` を指す
- 重さ：記録する
- 該当箇所：`DAI/exp/dai13_probe.py:787`，`:765`，`:1212`
- 問題：`code_post_r0.md` の指摘 2 で予告したとおり，`results/dai13_probe/provenance.json` は 2 つの種の集計の記録になった．r = 0 だけの版の記録は `summary_seeds1/provenance.json` と `provenance_log.jsonl` の 2 行目にある．`summary_seeds1/` の表そのものは書き換えられていない（7 節）が，同じ台本で作り直すには `*_r0*` の行だけを別の `rows/` に置く必要がある．
- 計算の後で直せるか：直せる
- 求める対応：結果の文で seeds2 の値を引くときは `summary_seeds2/provenance.json` を，seeds1 の値を引くときは `summary_seeds1/provenance.json` を書く．`results/dai13_probe/provenance.json` は「最後の集計の写し」と書いて引く．

### [記録] 3．`self` の入れ子の後の量は r = 1 でも「作りの上で 0」になっていない
- 重さ：記録する
- 該当箇所：設計 3.2 節の表 (iv) と 4.2 節，`summary_seeds2/lw_units.csv` の `lw_used_layers_h0`・`_h1`（`teacher == self`），`summary_seeds2/diffs.csv` の `self` の `layers_minus_final`
- 問題：`code_post_r0.md` の指摘 3 と同じ種類である．`self` で入れ子の選択が層の重みを学んだ版を残した (単位, K, 半分) は r = 1 で 65／600，2 つの種で 145／1,200 である．このため seeds2 の `diffs.csv` の `self` の入れ子の後の `layers_minus_final` で `se_raw` が 0 なのは 20 行のうち 10 行だけである．R1c は入れ子の前の量で数える（台本 `:982`）ので規則の数え上げには影響しない．
- 計算の後で直せるか：直せる（文の直しだけ）
- 求める対応：結果の文で，seeds2 を引く箇所の数を 145／1,200 と `se_raw` が 0 の行 10／20 に書き換える．

### [記録] 4．`pair` と `pair_shuf` で `final` と DAI-12 の `lince` の許しを超えた数を 2 つの種で並べる
- 重さ：記録する
- 該当箇所：各 npz の `mismatch`，`summary_seeds2/criteria.csv` の `final_vs_dai12_lince_*_n_pair`・`_n_self`，`summary_seeds2/lw_units.csv` の `final_vs_dai12_lince_*` 列
- 問題：`pair` の UAR の差が 5.0 を超えた (単位, K) は 2 つの種で 4／600（r = 0 3，r = 1 1），判定の不一致の割合が 0.25 を超えたものは 0／600．`pair_shuf` は 0.25 超が 95／600，5.0 超が 12／600 で，`criteria.csv` に `pair_shuf` を数える列は無い（`code_post_r0.md` の指摘 5 と同じ種類）．
- 計算の後で直せるか：直せる（`lw_units.csv` から CPU で数えられる）
- 求める対応：結果の文で R1・R1shuf を読む箇所に，上の 4 つの数を `self` の 0／600 と並べて書く．

### [記録] 5．`pair_shuf` で，選んだ点の判定が候補の全部で 1 つの感情だけになる選択が 27 個ある
- 重さ：記録する
- 該当箇所：各 npz の `grid_pred_{lw,final,unif,finallw}_A` と `conv` の選んだ点（`check4.py` の後に流した追加の確認．台本は本報告の作業の中で流しただけで，ファイルには残していない）
- 問題：4 つの腕（入れ子の前の `lw`，`final`，`unif`，`finallw`）× 2 つの種の 14,400 個の選択（単位 × 半分 × 出所 × K × 腕）のうち 27 個（r = 0 18，r = 1 9）で，選んだ (wd, 保存点) の判定が候補の全部で同じ感情になっている．27 個とも出所は `pair_shuf` で，`pair`・`self`・全ラベルの腕には 0 個である．格子の切片では wd 10 の `pair_shuf` に全部 0 の切片が 2 つの種で 123 個，0 以外の 1 つの番号だけの切片が 1 個ある（4 節）．選び方は設計どおりに動いた結果で，誤った数値ではない．`pair_shuf` の量を読むとき，この選択の UAR は 25% になる．
- 計算の後で直せるか：直せる（保存から CPU で数えられる）
- 求める対応：結果の文で R1shuf を読む箇所に「`pair_shuf` では 14,400 個の選択のうち 27 個で 1 つの感情だけを出す点が選ばれた（`pair`・`self` では 0 個）」と書く．

### [記録] 6．`diffs.csv` の `method` の名前は種の数で変わらず，種の分散と塊の分散の内訳の列が無い
- 重さ：記録する
- 該当箇所：`DAI/exp/dai1_stats.py` の `effect_stats`（`method="jk+seed"`／`"boot_cl+seed"`），`DAI/exp/dai12_probe.py:275-310` の `effect_stats_pool`（`"boot_cl_shared+seed"`），`DAI/exp/dai12_probe.py:667-674` の `_diff`（`v_jk`・`v_seed` を行に残さない）
- 問題：種が 1 つの seeds1 でも `method` は `+seed` を含み，そのときの種の分散は 0 である．seeds2 で種の項が実際に区間に入ることは，6 節で行から作り直して確かめた（主の量の 4 コーパスで種の分散 > 0，jackknife の 3 コーパスで `se_raw` が一致）．表だけからは，区間の幅のうち種の揺れが占める分を読めない．
- 計算の後で直せるか：直せる（保存の判定から CPU で出せる．`check6.py` の `jk_se` がその計算である）
- 求める対応：結果の文で区間の作り方を書くとき，「seeds2 の区間は分割ごとの 2 つの種の差の分散を含む．`method` の名前は種の数によらない」と書く．種の分散の割合を問われたら `check6.py` と同じ計算で出す．

## 目録に無い種類の不備

- 区間の作り方の名前（`method`）が，実際に入った分散の項を表さない（指摘 6）．`docs/methods.md` 9.3 節の I の項に「区間の表には，作り方の名前とともに，分散の項ごとの値（塊，種，初期値の代わり）と種の数を列で残す」を足す案を出す．

## 台帳用：この点検で見たもの（2026-11-10 02:3x，gate-code の計算の後の点検．種 r = 0, 1 と seeds2）

見たのは，`results/dai13_probe/rows/*_r1*` の単位の名前と個数，npz の鍵・型・形・値の範囲の外の個数・有限でない値の個数・`commit`・`script_sha256`，DAI-12 の r = 1 の保存との発話と半分と部分集合の一致，`mismatch` の JSON（整数の鍵の不一致 0，`pair`・`pair_shuf`・`self` の許しを超えた個数，`self` の 2 つの最大 0.082 と 2.04），格子の判定と選んだ版の判定の不一致の本数と全部が 0 の切片の数，`summary_seeds2/` の csv の行数・列名・重複・NaN と inf の個数と場所，seeds1 との行の構造の比較，`provenance_log.jsonl`・`provenance.json`・`summary_meta.json`，所要と GPU．UAR・差・区間の値と判定の文字列（`verdict`，`shuf_case`，`note_*`）は印字していない．取り直した値は表の値との差の絶対値と，R1 の `n_K_up` の一致の真偽だけを印字した（`n_K_up` の数そのものは印字していない）．
**性能や手順の結果から決まる個数を 5 つ見た**：(1) `self` で入れ子の選択が層の重みを学んだ版を残した (単位, K, 半分) の数 145／1,200 と，`self` の `se_raw` が 0 の行の数，(2) 格子の全部が 0 の切片と 1 つの番号だけの切片の数（wd 10 の `pair_shuf`），(3) `pair`・`pair_shuf` で `final` と `lince` の許しを超えた (単位, K) の数，(4) 選んだ点が 1 つの感情だけを出す選択の数 27／14,400，(5) 主の量の 4 コーパスで種の分散が 0 より大きいこと．
