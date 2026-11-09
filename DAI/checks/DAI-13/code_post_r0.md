# 計算の後の点検：種 r = 0 の結果ファイルと記録（DAI-13 の下調べ，2026-11-09 23:3x，単位の段のコミット 4932d6c，集計の段のコミット 7edef9c）

点検した対象は次のとおりである．`results/dai13_probe/rows/` の種 r = 0 のファイル（`*_r0.npz`・`*_r0_lw.csv`・`*_r0_lab.csv`・`*_r0_layers.csv`．各 60 本），集計 `results/dai13_probe/summary_seeds1/` の 16 ファイル，`provenance.json`，`provenance_log.jsonl`，`units_time_20261109-231222.csv`，ログ `logs/dai13_probe_units.log`・`logs/dai13_probe_sum.log`・`logs/dai13_probe_driver.log`．台本 `DAI/exp/dai13_probe.py`（sha256 `19960ba6b4257a27b0e9ef757bda776dd7d9e68229e8a69b133877fd210a7b1f`，コミット 4932d6c），設計 `DAI/docs/実験13_下調べ_設計.md`（第 15 版，`54d8245c…`）．
点検は CPU だけで行い，GPU は使っていない（種 r = 1 の単位の段が GPU で走っている）．`rows/*_r1*` は読んでいない．`provenance_log.jsonl` の 3 行目（r = 1 の単位の段の記録）だけは，r = 0 の記録との違いを確かめるために読んだ．検算の台本は `/tmp/claude-1000/-home-snics-MER-DA/b1f27254-32c7-45f0-ac5a-dee1c5e3ec8a/scratchpad/dai13_post_r0/check1.py`〜`check7.py`（と `check5b.py`）に置いた．どの台本も，個数・鍵・形・型・不一致の数・有限性・所要と，取り直した値と表の値の差の絶対値だけを印字し，UAR・差・区間の値と判定の文字列（`verdict`，`shuf_case`，`note_*`）は印字していない．`results/`・`data/`・`docs/`・`DAI/docs/` のファイルは書き換えていない．

## 判定
通す．60 単位の npz と 3 本の csv がそろって重複が無く，鍵の集合は仮のターゲット c ごとに 1 通りで設計 3.6 節の列挙を満たし，浮動小数の配列に有限でない値が無く，int8 の判定は 0〜3 の中で，台本の照合の整数の鍵の不一致は 0 本，`self` の `final` と DAI-12 の `lince` の不一致は許しの中，`pair` の許しを超えた単位はログの警告 2 行と同じ 2 単位であり，格子の全点の判定は選んだ版の判定と全件一致し，集計の表は行の数が設計から出る数と合い，行から独立に取り直した値が表と浮動小数の丸めの範囲で一致した．記録の不足が 5 件ある（下の「指摘」．どれも結果の文で直せる）．

## 1．単位の集合，ファイル，重複（`check1.py`，`check3.py`，`check7.py`）

| 確かめたこと | 結果 |
|---|---|
| `rows/*_r0.npz` の単位の名前 | 60 個，重複 0．`results/dai12_probe/rows/*_r0.npz`（60）と `results/dai1/pseudo/units/*_r0_p3.npz`（60）の集合と一致 |
| 名前の形 | 全部 `<T>_<c>_f[0-4]_r0` で T ≠ c．c ごとに 15 単位，T ごとに 15 単位（設計 2 節の T 4 × c 3 × f 5） |
| `_lw.csv`・`_lab.csv`・`_layers.csv` | 各 60 本．名前の集合は npz と同じ |
| npz の `unit`・`T`・`c`・`seed` と名前 | 不一致 0 |
| 単位のログ | `logs/dai13_probe_units.log` は 62 行で，「単位」の行が 60（重複 0，npz の集合と同じ），「警告」の行が 2，ほかの行 0．例外・収束しない警告・NaN・メモリ不足の語は無い |
| 書き込み時刻 | r = 0 のファイルは 20:05:47〜23:12:22．単位の段の終わり（23:12:23）より後に書かれた r = 0 のファイルは 0 本 |
| 発話の対応 | 60 単位とも `pool_sample_id`・`test_sample_id`・`pool_y`・`test_y`・`pool_cluster`・`test_cluster`・`kpos_K{1,5,10,25}`・`pool_half`・`pool_sub_*`・`pool_shuf_src_A`・`pool_none_K*_A` が DAI-12 の保存と一致 |
| 漏れの検算 | 60 単位とも，候補とテストで共有する `sample_id` 0 個，塊 0 個．2 つの半分が共有する塊 0 個 |
| `half_counts` | 60 単位とも `pool_half` と `pool_y` から数え直した本数と一致 |

## 2．設計 3.6 節の保存の鍵（`check1.py`，`check2.py`）

鍵の数は c ごとに 1 通りである：IEMOCAP・MELD を c にした単位は 810 鍵，AFEW・MOSEI を c にした単位は 809 鍵．違いは部分集合の印 `pool_sub_*` の数（IEMOCAP `marker`・`nomarker`，MELD `dlgfree`・`face_single`，AFEW `asr_ok`，MOSEI `channel`）だけで，ほかの 808 鍵は 60 単位で共通である．ログの「保存 787」（MELD を c にした単位）に meta の 23 鍵を足すと 810 になり合う．

| 3.6 節の項目 | 鍵（単位あたりの数） | 型と形 | 結果 |
|---|---|---|---|
| 候補の判定（3 出所 × K 5） | `pool_lw` 15，`pool_lwfix` 15，`pool_lwpre` 15，`pool_final` 15，`pool_finalfix` 15，`pool_unif` 15，`pool_uniffix` 15，`pool_finallw` 15，`pool_finallwfix` 15 | int8，(候補,) | そろう |
| 全ラベルの分類器の候補の判定 | `pool_unif_lab_A`，`pool_lwpre_lab_A`，`pool_lw_lab_A`，`pool_final_lab_A`，`pool_{lw,final}_labfix_A`，`pool_lw_lab_fixalpha_K{0,25}_A` 2 | int8，(候補,) | そろう |
| Q4・Q5 | `pool_read_{lw,final}_K{1,5,10,25}_A_wd{0,1,2}` 24，`pool_kshot_{lw,final}_K*_A_wd{0,1}` 16，`alpha_kshot_K*_A_wd*` 8 | int8 (候補,)／float32 (25,) | そろう |
| 層の重み | `alpha_lw_` 30，`alphapre_lw_` 30，`alphafix_lw_` 30（3 出所 × K 5 × 半分 2），`alpha_lw_lab_A_h*` 2，`alphapre_lw_lab_A_h*` 2 | float32 (25,)．和と 1 の差の最大 2.4e−7（6,120 本） | そろう |
| 格子の全点の α | `alpha_grid_lw_A`，`alpha_grid_lw_lab_A` | float16，(2, 15, 5, 8, 25) と (2, 1, 5, 10, 25) | そろう（3.6 節の形と同じ） |
| 格子の全点の判定 | `grid_pred_{lw,final,unif,finallw}_A` 4 | int8，(1, 5, 8, 15, 候補) | そろう |
|  | `grid_pred_{lw_lab,final_lab,unif_lab}_A` 3，`grid_pred_lw_lab_fixalpha_A` 1 | int8，(1, 5, 10, 1, 候補) と (1, 5, 10, 2, 候補) | そろう |
| テストの判定 | `K{K}_A_{lw,lwfix,lwpre,final,finalfix,unif,uniffix,finallw,finallwfix}_{出所}` 135，`K{K}_A_none` 5，`labtest_{lw,final,unif}_A` 3 | int8，(テスト,) | そろう |
| 点数 | 候補 `score_{lw,final,unif,finallw,lwpre}_K*` 75，テスト `score_{lw,final,unif,finallw}_test_K*` 60 | float16，(候補, 4)／(2, テスト, 4) | そろう |
| `pair` の W・c | `w_lw_pair_K*_A_h*_{W,c}` 20，`w_final_pair_…` 20 | float16，(1024, 4)／(4,) | そろう |
| 層ごとのリッジ | `pool_layer{l}_ridge_K{K}_A_pair` 125，`pool_layer{l}_ridgelab_A` 25 | int8，(候補,) | そろう |
| 行の情報と記録 | `pool_sample_id`・`pool_y`・`pool_cluster`・`test_*`・`kpos_K*`・`pool_half`・`half_counts`・`pool_shuf_src_A`・`pool_teacher_K*_A_*` 15・`pool_none_K*_A` 5・`mismatch`（JSON 64 鍵）・`conv`（JSON．`lw`・`final`・`unif`・`finallw`・`lw_lab`・`final_lab`・`unif_lab`・`lw_lab_fixalpha`・`lw_pre_cfg`）・`commit`・`commit11`・`commit12`・`script_sha256` | | そろう |

- 浮動小数の配列は 16,740 本（float16 10,620，float32 6,120）で，有限でない値を含む配列は 0 本．
- int8 の配列は 30,510 本．判定の鍵（`pool_*`，`K*`，`labtest_*`，`grid_pred_*`）で 0〜3 の外の値を含む配列は 0 本，`pool_half`・`pool_sub_*` で 0・1 の外は 0 本．
- 形の食い違いは 0 件．

## 3．台本の照合（`check3.py`，`check5.py`）

| 確かめたこと | 結果 |
|---|---|
| `mismatch` の鍵の数 | 60 単位とも 64 鍵．有限でない値 0，`frac` の鍵で 0〜1 の外 0 |
| 整数の鍵（`K*_pool_none` 5，`K*_test_none` 5，`K*_pool_teacher_*` 15，`pool_half`，`pool_shuf_src`） | 1,620 個（60 × 27）．0 でないものは 0 個．許し（候補・テストの max(1, 0.1%) 本，半分と入れ替えは 0 本）を超えるもの 0 個 |
| 添字 24 の生の特徴と既存の音声の特徴の差 | 最大 9.2e−4（止める値 1e−2）．標準化した後の差の最大 1.6e−2（記録のみ） |
| `final_vs_dai12_lince_pair_over` が立った単位 | 2 単位（`AFEW_IEMOCAP_f4_r0`，`MELD_IEMOCAP_f4_r0`）．ログの「警告」の 2 行の単位と一致 |
| `pair` で許しを超えた (単位, K) | UAR の差 > 5.0 が 3 個（`AFEW_IEMOCAP_f4_r0` の K = 0，`MELD_IEMOCAP_f4_r0` の K = 0 と K = 25），判定の不一致の割合 > 0.25 が 0 個．`criteria.csv` の `final_vs_dai12_lince_uar_over_n_pair` の 4 行の和 3，`…_frac_over_n_pair` の和 0 と一致 |
| `self` の `final` と DAI-12 の `lince` | 判定の不一致の割合の最大 0.108（許し 0.25），候補の上の UAR の差の絶対値の最大 2.35（許し 5.0）．`criteria.csv` の `…_over_n_self`・`…_frac_over_n_self` の和はどちらも 0 |
| `pair_shuf`（記録のみ．設計 3.2 節） | 判定の不一致の割合 > 0.25 が 54 個，UAR の差 > 5.0 が 7 個（(単位, K) の数）．止める条件の外 |
| `checks_units.csv` | 3,840 行（60 × 64），(unit, key) の重複 0 |

## 4．格子の全点の判定（`check4.py`）

台本の `check_grid_consistency`（`DAI/exp/dai13_probe.py:326-338`）と同じ確認を，保存した `grid_pred_*` と `conv` の選んだ点 (wd, 保存点) と候補の判定から，**60 単位すべて**について CPU で行った．半分 h の行について，選んだ点の格子の判定と選んだ版の判定を比べた．

| 腕 | 選んだ点の出どころ | 比べた判定の本数 | 違った本数 |
|---|---|---|---|
| `lw`（入れ子の前） | `conv["lw_pre_cfg"]` と `pool_lwpre_*` | 5,307,705 | 0 |
| `final` | `conv["final"]["cfg"]` と `pool_final_*` | 5,307,705 | 0 |
| `unif` | `conv["unif"]["cfg"]` と `pool_unif_*` | 5,307,705 | 0 |
| `final_lw` | `conv["finallw"]["cfg"]` と `pool_finallw_*` | 5,307,705 | 0 |
| `lw`（入れ子の後） | `conv["lw"]["cfg"]`．`_lw.csv` の `lw_used_layers_h*` が偽の半分は `grid_pred_final_A` から | 5,307,705 | 0 |
| `lw`（入れ子の後）と入れ子の前または `final` の判定 | 同上 | 5,307,705 | 0 |
| `final_lab`・`unif_lab` | `conv[...]["cfg"]["h*_j0"]` | 各 353,847 | 0 |
| `lw_lab`（入れ子の後） | `conv["lw_lab"]["cfg"]`．`lw_lab_used_layers_h*` が偽の半分は `grid_pred_final_lab_A` から | 353,847 | 0 |
| `lw_lab`（入れ子の前．置き換わらなかった半分） | `pool_lwpre_lab_A` | 350,060 | 0 |
| `lw_lab_fixalpha`（K = 0, 25） | `conv["lw_lab_fixalpha"]["cfg"]` | 707,694 | 0 |

全部が 0 の切片（wd × 保存点 × 目標の 1 つ）：8 つの腕の 192,000 切片のうち 76 個で，**全部が wd 10 の `pair_shuf`**（`lw` 12，`final` 30，`unif` 11，`finallw` 23）である．前の検査で既知の種類と同じで，ほかの wd・出所・全ラベルの腕には無い．0 以外の 1 つの番号だけの切片は 0 個．

## 5．集計の表（`check5.py`，`check5b.py`）

`summary_seeds1/` の 16 ファイル（`cells_lw.csv`，`cells_lab.csv`，`cells_layers.csv`，`diffs.csv`，`criteria.csv`，`v_init.csv`，`alpha_summary.csv`，`se_summary.csv`，`lw_units.csv`，`lab_units.csv`，`layers_units.csv`，`uar_units.csv`，`checks_units.csv`，`halves_units.csv`，`summary_meta.json`，`provenance.json`）と `rows` へのリンクがそろい，ほかのファイルは無い．

| 表 | 行 | 設計と台本から出る数 | 重複 | 有限でない値 |
|---|---|---|---|---|
| `lw_units.csv` | 900 | 60 単位 × 出所 3 × K 5 | (unit, teacher, K) 0 | NaN は Q4・Q5 の 24 列だけで，`pair` かつ K ≥ 1 でない 660 行と完全に一致（台本 `:727`） |
| `lab_units.csv` | 300 | 60 × K 5 | (unit, K) 0 | NaN は `uar_lab_fixalpha`・`lab_fixalpha_minus_final` だけで，K ∉ {0, 25} の 180 行と完全に一致（`Q3_KS`） |
| `layers_units.csv` | 7,500 | 60 × 層 25 × K 5 | 0 | 0 |
| `cells_lw.csv` | 60 | c 4 × 出所 3 × K 5．`n_units` は全行 15 | 0 | NaN は上と同じ構造（44 行 × 24 列） |
| `cells_lab.csv` | 20 | c 4 × K 5．`n_units` は全行 15 | 0 | NaN は K ∉ {0, 25} の 12 行 × 2 列 |
| `cells_layers.csv` | 500 | c 4 × 層 25 × K 5 | 0 | 0 |
| `alpha_summary.csv` | 128 | c 4 ×（`lw`・`lw_pre` × 出所 3 × K 5 ＋ `lw_lab`・`lw_pre_lab` 各 1） | 0 | 0 |
| `v_init.csv` | 60 | c 4 × 出所 3 × K 5．`n_folds` は全行 5 | 0 | 0 |
| `uar_units.csv` | 8,580 | 60 × テストの判定の鍵 143（腕 9 × 出所 3 × K 5 ＋ `none` 5 ＋ `labtest` 3） | (c, T, fold, seed, key) 0 | 0．UAR などの 0〜100 の外 0 |
| `checks_units.csv` | 3,840 | 60 × 64 | 0 | 0 |
| `halves_units.csv` | 60 | 60 | 0 | 0 |
| `diffs.csv` | 1,756 | 下の計算 | (c, teacher, K, where, quantity, subset) 0 | 下 |
| `se_summary.csv` | 168 | `diffs.csv` の `subset == all` の (c, quantity) の組の数 168 | 0 | 0 |
| `criteria.csv` | 40 | c ごとに R1・R1pre・R1shuf・R1shufpre・R1c・Q2・Q3・EDGE…・GAP_vs_GAIN の 9 行と，条件つきの Q3/Q2 の行（台本 `:1024`） | — | 規則ごとに列があるか無いかが決まり，規則の中で一部だけ NaN の列は 0 |

`diffs.csv` の行の数：台本 `:867-918` から，c ごとに，候補の上の出所の量 190（出所 3 × 量 12 × K 5 ＋ `pair` の入れ替えの 2 × K 5），部分集合 60 × 部分集合の数，テスト 60，全ラベル 27（5 × K 5 ＋ Q3 の 2），Q4・Q5 72（18 × K 4）で，349 ＋ 60 × 部分集合の数になる．部分集合の数は AFEW 1・IEMOCAP 2・MELD 2・MOSEI 1 なので 4 × 349 ＋ 60 × 6 = 1,756 で一致する．`subset` の内訳は `all` 1,396，6 つの部分集合が各 60．`n_units` は全行 5（分割の数．種は 1 つ）．

- `est`・`lo`・`hi`・`se`・`p`・`lo_raw`・`hi_raw`・`se_raw`・`p_raw`・`quantile` の NaN は 0 個，inf は 0 個．NaN は `v_init` の列の 108 行だけで，全部 `teacher == labels`（全ラベルの分類器の差．v_init を足す対象の外）．
- v_init を足した行（`se_adjusted`）は 840 行で，量は台本 `:940` の 8 つ（`layers_minus_final`，`_pre`，`_fixed`，`lw_minus_unif`，`unif_minus_final`，`finallw_minus_final`，`lw_minus_finallw`，`lwpre_minus_finallw`），`where` は全部 `pool`．数は c ごとに 出所 3 × K 5 ×（8 ＋ 4 × 部分集合の数）の和 840 と一致する．
- 足した行で `se = √(se_raw² + v_init)` の差の最大 2.2e−15，`lo = est − quantile × se`・`hi = est + quantile × se` の差の最大 1.8e−15．足していない行では `lo`・`hi`・`se` が `_raw` と同じ．
- `quantile` は `boot_cl_shared+seed`（MOSEI の候補 349 行）と `boot_cl+seed`（MOSEI のテスト 60 行）で 1.96，`jk+seed`（AFEW・IEMOCAP・MELD 1,347 行）で 2.16〜3.18．設計 4.1 節の枝の分け方と合う．
- `se_raw` が 0 の行は 43 行で，全部 `teacher == self`（`delta_uar` 20 行，入れ子の後の `layers_minus_final` 16 行，`lw_vs_final_test` 7 行）．43 行とも `est` は 0．`self` の `delta_uar` は目標と `none` が同じものなので作りの上で 0 である．
- `summary_meta.json`：`v_init_applied` 真，`n_seeds` 1，`n_units` 60，`row_commits` は 4932d6c の 1 つ，`script_sha_rows` と `script_sha_summarize` はどちらも `19960ba6…` で，行の `script_sha256` と同じ．`diffs.csv` の `v_init_applied` 列も全行で真．v_init を足したことはこの 2 か所と `provenance.json`（`summary_meta.json` を取り込む．台本 `:1210`）に記録されている．
- 集計のログ（`logs/dai13_probe_sum.log`）には pandas の `PerformanceWarning` 3 件と `Pandas4Warning` 1 件（引数の書き方の予告）があるだけで，値には影響しない．最終行は「集計：単位 60（種 [0]），差の行 1756，規則の行 40，v_init を足した True」．

## 6．集計の値を行から取り直す（`check6.py`）

台本と共通の部品の関数を使わずに，npz の判定・`pool_y`・`kpos`・部分集合の印から数え直した（分割ごとに 3 つの T の候補の感情ごとの正解の数と件数を足して UAR を出し，その差を分割で平均する．`dai1_stats.uar_full` と `dai12_probe._add`・`_finish` と同じ定義）．印字したのは表の値との差の絶対値だけである．

| 取り直したもの | 表 | 差の絶対値 |
|---|---|---|
| AFEW，`pair`，K = 5，`layers_minus_final`（候補） | `diffs.csv` の `est` | 3.1e−15 |
| MELD，`pair_shuf`，K = 10，`lw_minus_unif`（候補） | 同 | 4.3e−15 |
| MOSEI，`pair`，K = 25，`layers_minus_final_pre`，部分集合 `channel` | 同 | 8.4e−15 |
| IEMOCAP，全ラベル，K = 1，`lab_layers_minus_final` | 同 | 6.0e−15 |
| IEMOCAP，`self`，K = 25，`layers_minus_final_pre`（R1c の量） | 同 | 5.7e−15 |
| AFEW，`pair`，K = 10，`lw_vs_final_test`（テスト） | 同 | 5.3e−15 |
| `cells_lw.csv` の 4 列 × 60 セル（`uar_lw`，`uar_final`，`layers_minus_final_pre`，`final_vs_dai12_lince_uar_absdiff`）と `lw_units.csv` の平均 | `cells_lw.csv` | 最大 7.1e−15 |
| `lw_units.csv` の `uar_lw`・`uar_final`（2 単位 × 15 行）と npz の判定 | `lw_units.csv` | 最大 7.1e−15 |
| `v_init` の 60 行（`lw_units.csv` の `uar_final` と DAI-12 の `rows/*_nl.csv` の `uar_lince` から．対応の取れない行 0） | `v_init.csv` | 最大 1.5e−16 |
| MELD，`lw_pre`，`pair`，K = 5 の 25 層の重みの平均（15 単位 × 半分 2） | `alpha_summary.csv` | 最大 3.2e−8（float32 の丸め） |

## 7．記録の整合（`provenance_log.jsonl`，`provenance.json`）

| 記録 | 内容 |
|---|---|
| `provenance_log.jsonl` | 3 行．1 行目 `units`，コミット 4932d6c，dirty 偽，20:04:10，argv `--out results/dai13_probe --stage units`．2 行目 `summarize`，コミット 7edef9c，dirty 偽，23:12:25．3 行目は r = 1 の `units`（コミット 7edef9c，`--units *_r1 --resume`，23:17:37） |
| 1 行目と 2 行目の違い | `argv`・`commit`・`stage`・`time`・`design_sha256` の 5 つだけ．台本・`dai12_probe.py`・`dai11_probe.py`・`dai1.py`・`dai_solver.py`・`dai_data.py`・`dai1_stats.py`・`build_audio_layers.py`・凍結の JSON の sha256，全層の特徴 4 本の sha256，定数，ソフトの版は同じ．設計の sha256 は単位の段が第 14 版 `e87b7585…`，集計の段が第 15 版 `54d8245c…` |
| 4932d6c と 7edef9c の差 | `git diff --stat` は設計（9 行）と `docs/seen-data.md`（3 行）だけ．設計の差は版の行，3.2 節の理由の文，6 節の 1 回目の記録，9 節の検査の表の 1 行で，規則・定数・見込みは変わっていない．7edef9c から HEAD までの差は `docs/seen-data.md` の 1 行だけ |
| 今の入力 | 台本と共通の部品 8 本の今の sha256，全層の特徴 4 本の今の sha256 が 1 行目の記録と同じ．`DAI/exp`・`DAI/run`・`measure`・`features`・`data/metadata`・`DAI/docs` に未コミットの変更は無い |
| `provenance.json` と `summary_seeds1/provenance.json` | 2 つは同じで，2 行目（集計の段）に `summary_meta.json` の 6 鍵と `summary_dir` を足したもの．コミットは 7edef9c |
| npz の記録 | 60 単位とも `commit` 4932d6c，`script_sha256` `19960ba6…`（1 つ），`commit11` 67e3553，`commit12` e9f66c9（DAI-12 の単位の段のコミット．記録の `dai12_units_commit` と同じ） |
| ドライバのログ | 設計の sha256 `e87b7585…`，開始 20:04:08，単位の段の終わり 23:12:23，集計の段の終わり 23:16:06．`logs/dai13_probe.done` は空のファイルとしてある．`logs/dai13_probe_r1_driver.log` に r = 1 の開始 23:17:35 |

## 8．所要と GPU（`check7.py`，ログ）

- 単位の段：20:04:08 → 23:12:23（3 時間 8 分．`units_time_20261109-231222.csv` の秒の和は 3.14 時間）．設計 6 節の見込み 3〜4 時間の中．
- 単位あたりの秒（c ごとの最小〜最大）：AFEW 52〜60，IEMOCAP 90〜96，MELD 160〜193，MOSEI 420〜426．
- 集計の段：23:12:25 → 23:16:06（3 分 41 秒）．見込み 30 分以内の中．
- GPU：`max_memory_allocated` の最大 11.1 GB，`max_memory_reserved` の最大 12.6 GB（MOSEI を c にした単位）．6 割の上限 14.7 GB の中．

## 指摘

### [記録] 1．`provenance.json` のコミットは集計の段の 7edef9c で，計算をした単位の段のコミット 4932d6c は `provenance_log.jsonl` の 1 行目と npz にだけある
- 重さ：記録する
- 該当箇所：`DAI/exp/dai13_probe.py:1209-1212`（集計の段だけが `provenance.json` を書く），`results/dai13_probe/provenance.json`
- 問題：DAI-12 の計算の後の点検の指摘 1 と同じ種類である．`DAI/CLAUDE.md` の取り決めで結果の文の冒頭に `provenance.json` の場所を書くと，読み手は計算のコミットを 7edef9c と読む．設計の sha256 も，単位の段の記録は第 14 版 `e87b7585…`，集計の段の記録は第 15 版 `54d8245c…` で違う（差は 7 節のとおり文の書き換えだけ）．
- 計算の後で直せるか：直せる（記録は全部ある）
- 求める対応：結果の文の冒頭に「単位の段はコミット 4932d6c（`provenance_log.jsonl` の 1 行目と各 npz の `commit`．設計は第 14 版），集計の段は 7edef9c（`summary_seeds1/provenance.json`．差は設計の文と台帳だけで，台本と部品の sha256 は同じ）」と書く．

### [記録] 2．`results/dai13_probe/provenance.json` は r = 1 の集計で上書きされる．`summary_seeds1/rows` は r = 1 の行も入る共有のディレクトリを指す
- 重さ：記録する
- 該当箇所：`DAI/exp/dai13_probe.py:787`（`(out / "rows").symlink_to(root / "rows")`），`:765`（`(root / "rows").glob("*.npz")`），`:1212`（最新の集計の写し）
- 問題：r = 1 の集計の段が終わると，`results/dai13_probe/provenance.json` は `summary_seeds2` の記録に置き換わる．r = 0 だけの版の記録は `summary_seeds1/provenance.json` と `provenance_log.jsonl` の 2 行目に残る．また `summary_seeds1/rows` は `results/dai13_probe/rows` へのリンクなので，今は r = 1 の行も見える．`summary_seeds1/` の表を同じ台本で作り直すには `*_r0*` の行だけを別のディレクトリに置く必要がある（`summary_seeds1/` の表そのものは 23:13〜23:16 に r = 0 の 60 単位だけから書かれ，`summary_meta.json` の `n_units` は 60）．
- 計算の後で直せるか：直せる
- 求める対応：r = 0 だけの版を引く結果の文では，記録の場所を `summary_seeds1/provenance.json` と書き，`results/dai13_probe/provenance.json` を引かない．作り直しの手順（`*_r0*` だけを別の `rows/` に置いて `--stage summarize --n-expected 60` を流す）を結果の文の記録の節に 1 文で書く．

### [記録] 3．入れ子の後の `self` の量は「作りの上で 0」になっていない
- 重さ：記録する
- 該当箇所：設計 3.2 節の表 (iv)「入れ子の後は作りの上で 0 になり測らない」と 4.2 節「入れ子の後の値は作りの上で 0 なので使わない」，`summary_seeds1/lw_units.csv` の `lw_used_layers_h0`・`_h1`（`teacher == self`），`diffs.csv` の `self` の `layers_minus_final`
- 問題：`self` の 600 個の (単位, K, 半分) のうち 80 個で，入れ子の選択が層の重みを学んだ版を残した．このため `diffs.csv` の `self` の入れ子の後の `layers_minus_final` で `se_raw` が 0 なのは 20 行のうち 16 行だけである．試走（設計 6 節）では `self` は全件 `final` に置き換わったが，本計算ではそうならなかった．規則 R1c は台本 `:982` で入れ子の前の `layers_minus_final_pre` を数えるので，規則の数え上げには影響しない．
- 計算の後で直せるか：直せる（文の直しだけ）
- 求める対応：結果の文で「`self` の入れ子の後の量は作りの上で 0」と書かず，「`self` では (単位, K, 半分) の 600 個のうち 80 個で層の重みを学んだ版が残った（試走では 0 個）．R1c は入れ子の前の量で数える」と書く．`criteria.csv` の `lw_used_layers_frac_self` を引く．

### [記録] 4．GPU の割り当ての最大が設計 6 節の値より大きい
- 重さ：記録する
- 該当箇所：`results/dai13_probe/units_time_20261109-231222.csv` の `gpu_max_gb`・`gpu_reserved_gb`
- 問題：設計 6 節は本番の束 15 本の MOSEI の単位で GPU 最大 10.0 GB（予約 11.4 GB）と書くが，本計算の最大は 11.1 GB（予約 12.6 GB）である．6 割の上限 14.7 GB の中で，ログにメモリ不足は無い．
- 計算の後で直せるか：直せる
- 求める対応：結果の文の所要の記述に本計算の値（単位の段 3 時間 8 分，集計 3 分 41 秒，割り当ての最大 11.1 GB，予約 12.6 GB）を書く．r = 1 は同じ台本と同じ単位の形なので同じ程度になる見込みである（読んだだけ）．

### [記録] 5．`pair_shuf` の `final` と DAI-12 の `lince` の不一致は件数が多いが，集計に数える列が無い
- 重さ：記録する
- 該当箇所：`DAI/exp/dai13_probe.py:1062-1066`（数えるのは `pair` と `self` だけ），各 npz の `mismatch` の `final_vs_dai12_lince_{frac,uar_absdiff}_pair_shuf_K*`
- 問題：設計 3.2 節は `pair_shuf` を「記録だけ」とし，4.4 節の添え書きは `pair`・`self` の最大と許しを超えた数を挙げる．`pair_shuf` では判定の不一致の割合が 0.25 を超えた (単位, K) が 300 個中 54 個，UAR の差が 5.0 を超えたものが 7 個あり，`criteria.csv` には数える列が無い（値は npz と `checks_units.csv` と `lw_units.csv` の `final_vs_dai12_lince_*` 列にある）．設計 3.2 節は DAI-12 の 8 節の記録（wd を 1 点足しただけで `pair_shuf` の `lince` の判定が AFEW 16%・MELD 11% 変わった）を挙げており，想定の範囲の種類である．
- 計算の後で直せるか：直せる（`lw_units.csv` から CPU で数えられる）
- 求める対応：結果の文で R1shuf の値を読むときに，`pair_shuf` の `final` が DAI-12 の `lince` と 0.25 を超えて違った (単位, K) の数 54／300 と，UAR の差が 5.0 を超えた数 7／300 を `pair`・`self` の数と並べて書く．

## 目録に無い種類の不備

- 種を後から足す台本で，集計の出力先 `summary_seeds{n}/rows` を共有の `rows/` へのリンクにすると，前の版の集計の入力が後から変わる（指摘 2）．`docs/methods.md` 9.3 節の I の項に「版を分けて残す集計は，入力の単位の一覧（名前と sha256）を集計の出力に書くか，入力の行を版ごとに分けて置く」を足す案を出す．

## 台帳用：この点検で見たもの（2026-11-09 23:3x，gate-code の計算の後の点検．種 r = 0）

見たのは，`results/dai13_probe/rows/*_r0*` の単位の名前と個数，npz の鍵・型・形・値の範囲の外の個数・有限でない値の個数・`commit`・`script_sha256`，DAI-12 の保存との発話と半分と部分集合の一致，`mismatch` の JSON（整数の鍵の不一致 0，`pair`・`pair_shuf`・`self` の `final` と `lince` の判定の不一致の割合と UAR の差の許しを超えた個数，`self` の 2 つの最大 0.108 と 2.35），格子の判定と選んだ版の判定の不一致の本数と全部が 0 の切片の数，集計の csv の行数・列名・重複・NaN と inf の個数と場所，`v_init` を足した行の式の検算，`provenance_log.jsonl`・`provenance.json`・`summary_meta.json`，所要と GPU（`units_time_*.csv` とログ）．UAR・差・区間の値と判定の文字列（`verdict`，`shuf_case`，`note_*`）は印字していない．取り直した値は表の値との差の絶対値だけを印字した．
**性能や手順の結果から決まる個数を 4 つ見た**：(1) `criteria.csv` の Q3/Q2 の行が 4 行ある（台本 `:1024` の条件は全ラベルの差の K ∈ {0, 25} の平均が 1.0 以上の c だけに行を作る），(2) `self` で入れ子の選択が層の重みを学んだ版を残した (単位, K, 半分) の数 80／600 と，`self` の `diffs.csv` で `se_raw` が 0 の行の数（`layers_minus_final` 16／20，`lw_vs_final_test` 7／20），(3) 格子の全部が 0 の切片の数（wd 10 の `pair_shuf` に 76 個），(4) `pair`・`pair_shuf` で `final` と `lince` の許しを超えた (単位, K) の数．
