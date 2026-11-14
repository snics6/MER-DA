# 計算の後の点検：結果ファイルと由来（DAI-20，2026-11-15，単位のコミット 341c62b・集計のコミット 423412c）

対象：`results/dai20_eval/`（読むだけ．何も書いていない），`logs/dai20_eval.log`・`dai20_units.log`・`dai20_summary.log`・`dai20_selftest.log`．
検算の台本と出力は `DAI/checks/DAI-20/post_run/`（`check_units.py`・`check_summary.py`・`check_recompute.py` と各 `_out.txt`．`check_misc_out.txt` は端末で直接流した短い検算の出力）．すべて CPU（`CUDA_VISIBLE_DEVICES=""`）で流し，最も長い 1 回は約 65 秒だった．集計の段は流し直していない．
この報告には UAR と差の値を書かない．書くのは数，一致の割合，ずれの大きさ，sha256，時刻だけである．

## 判定

通す．200 単位と集計の 12 ファイルが設計の格子どおりにそろい，欠けと重複は 0 件だった．検算の量は設計 6 節の基準の中にあった．保存した判定から作り直した UAR，水準，差，区間は，単位の csv と集計の表に対して数値の丸めの範囲（最大 7.1e-15）で一致した．由来（台本の sha256，コミット，計算に関わるファイルの差）にも食い違いは無かった．指摘は「記録する」の 5 件だけである．

## 1．段ごとの出力の数

| 項目 | 結果 |
|---|---|
| 単位のファイル | csv 200，npz 200．片方だけの名前は 0．期待する 200 の名前（4 T × 分割 5 × 種 10）との差は欠け 0・余り 0 |
| 単位の csv の行 | 200 単位すべてで 840 行（合計 168,000 行）．K ごとに 280 版．(K, 版) の組は 200 単位で 1 種類だけ |
| 鍵の重複 | (T, c, fold, seed, K, version) の重複 0．ファイル名と (T, fold, seed) の食い違い 0．c ≠ T の行 0．tag は全行 `eval` |
| K | 5・10・25 の 3 つ |
| 版の欠け | 設計 3〜5 節の版（`prod`・`psel`・`avg`・`conf`・`M_F2_K`・`M_F6_K`・`cf_D_l05`・`tcal`・`temp`・`tempb`・`wf10/100/1000`・`M_F6_ext`・`M_F8`・`M_F6_orc`・`none_AVT`・`joint_AVT`・`ret_A/V/T`・`b_*` の 5 つ・`prod_sel`・`M_F6_mix0〜15`・`M_F6n_mix0〜15`・`est_M_F6_mix0〜15`・`_check`・`_check20`）がすべての単位とすべての K にある．`M_F6h_mix` は β = 1 の 8 点（2, 3, 6, 7, 10, 11, 14, 15）にあり，台本 342 行の作りどおり |
| npz の中身 | 1 単位 3,452 個の配列．キーの集合は T ごとに 1 種類で，T どうしの違いはソースの名前の入った 6 個（`p_pool_src_<ソース>_*`・`p_test_src_<ソース>_*`）だけ．`pred_` の配列は 3,294 個（各 K で adj 227・raw 234） |
| NaN と無限大（単位の csv） | 無限大 0．`uar_raw`・`uar_adj`・`uar_pool` の NaN は，判定を持たない行（`_check`・`_check20`・`est_*`・`sel_*`・`_est_prod`）と，その層を持たない版（切片をずらさない B・単独の分類器の `uar_adj`，B の `uar_pool`，`prod_sel`・`prod_ob`・`k_bias` の `uar_adj`）だけ．UAR の値が 0〜100 の外の行は 0．全部 NaN の列は 0 |
| NaN と無限大（npz） | 無限大 0．NaN は 4 種の配列だけ：`SR_S_homeauc`・`SR_S_awayauc`（200 ファイル，各 36 個），`AUC_M_K_K*`・`AUC_M_orc_K*`（600 組，各 12 個）．数は 4 × 4 の感情の対の表の対角（4 個）× 表の数（ソース 3 × モダリティ 3，またはモダリティ 3）にちょうど当たり，対角の AUC は定義されないので意図した NaN である |
| テストの発話 | 200 単位すべてで，npz の `sid_test` が実験 1 の評価の回の保存（`results/dai1/eval/units/<名前>_p1.npz` の `sample_test`）と並びまで同じだった |
| 書いた順序 | 全単位で csv の更新時刻が npz 以後（npz を先に書き csv を済みの印にする作りどおり）．単位のファイルの更新時刻は 2026-11-14 21:26:31〜2026-11-15 02:54:13 で，集計の段（02:54:15 に記録）より後に単位が書き換わった跡は無い |
| 大きさ | `units/` は 1.89 GB（設計 8 節の見込み約 1.8 GB） |

## 2．台本の sha256 とコミット

| 項目 | 結果 |
|---|---|
| 台本の sha256 | `provenance_log.jsonl` の 2 行（units 2026-11-14 21:25:36，summary 2026-11-15 02:54:15）と `provenance.json` の `this_script_sha256` はどちらも 2d1b54806585a2efd9e30d85161f22c2bf97f3b1e6fae11059afaeac3f386474 で，検査を通した値と同じ．今の作業ツリーの `DAI/exp/dai20_eval.py` と，341c62b の中の同じファイルも同じ値 |
| 部品の sha256 | `scripts` の 11 本（`dai20_eval.py`・`dai16_probe.py`・`dai15.py`・`dai14.py`・`dai12_probe.py`・`dai11_probe.py`・`dai3.py`・`dai1.py`・`dai1_stats.py`・`dai_solver.py`・`dai_data.py`）は 2 つの段で同じで，今の作業ツリーとも同じ．入力の sha256（実験 1 の source と分割の 32 個）も今のファイルと全部同じ |
| 段のあいだで違う記録 | `stage`・`argv`・`commit`・`time` の 4 つだけ．`frozen_sha256`（faa2391d…bbe9f1），`design_sha256`（a364e2b6…f032），`GRID`，`settings`，`wf_tables_sha256`（ecae5f91…bc81），定数，ソフトの版は 2 つの段で同じ．`provenance.json` は `provenance_log.jsonl` の 2 行目と同じ中身 |
| 単位の csv の `commit`・`dirty` | 168,000 行すべてで commit 341c62b93b7f…（1 種類），dirty は偽．`dai1_commits` は ba98974 の 1 種類．host は 1 種類，GPU は NVIDIA RTX A5000 の 1 種類 |
| 集計の段のコミット | 423412c．`convergence.json` の `unit_commits` は [341c62b] の 1 つ，`summary_commit` は 423412c |
| 計算の途中のコミット | 341c62b（21:25:18）と 423412c（00:34:05）のあいだの 15 コミット（`git rev-list --count 341c62b..423412c`）は，台本の `WATCH`（台本と部品 11 本，固定のファイル，`DAI/splits`，`measure`，`features`，`data/metadata`，`requirements.txt`，設計）に差 0 行だった（`git diff --stat 341c62b 423412c -- <WATCH>` が空）．変わったのは `DAI/checks/` の 42 ファイル，`DAI/docs/` の 24 ファイル（`tools_pending/` と実験 18・21 の文書など．`実験20_設計.md` は含まない），`docs/seen-data.md` の 1 ファイルだけである |
| `code_v3.md` の指摘 1（固定のファイルと格子） | 単位のコミットから HEAD（07c5d67）まで，`DAI/frozen/dai20_frozen.json`・`DAI/exp/dai16_probe.py`（格子 `GRID`）・`DAI/exp/dai20_eval.py`・`DAI/run/dai20_eval.sh` に差 0 行．集計は単位と同じ固定の点と格子で作られた |
| HEAD の DAI-21 のコミット | 341c62b から HEAD までで計算に関わる場所の変更は `DAI/exp/dai21_eval5.py`・`DAI/frozen/dai21_frozen.json`・`DAI/run/dai21_eval5.sh` の追加だけで，3 つとも 03:08 以後のコミット（86c38f1・07c5d67）である．`dai20_eval.py` はこれらを読み込まない．集計の段の記録（02:54:15）で dirty が偽なので，その時点で `DAI/exp` に未追跡の `dai21_eval5.py` は無かった |
| 計算中に作業ツリーで更新されたファイル | 監視の場所（`DAI/exp`・`DAI/frozen`・`DAI/splits`・`measure`・`features`・`data/metadata`・`data/cache`・`requirements.txt`・設計・`DAI/run`・`results/dai1`・`results/dai2`・`results/dai16_probe*`）で，21:25〜03:08 に更新時刻を持つファイルは 3 つだけだった．設計（21:25:18．341c62b のコミットの時刻で，中身は 341c62b と同じ），`DAI/exp/dai21_eval5.py`（23:15:29）と `DAI/run/dai21_eval5.sh`（23:42:08）．後の 2 つはコミットの説明（「DAI-20 の計算の後に移した．中身は同じ」）と mv が更新時刻を残すことから，計算の後に移したファイルの元の時刻と読める（上の dirty の記録と合う） |
| 起動の台本の前後の照合 | `logs/dai20_eval.log` の最後が `EXIT=0` で，`logs/dai20_eval.done`（2026-11-15 03:07:46）があり，`logs/dai20_eval.failed` は無い．`dai20_eval.sh` は sha256 の照合（`SHA20`）と，実験 1・2 の集計，実験 1 の評価の回の保存（名前・大きさ・時刻），下調べの集計，固定のファイルの前後の照合（`snap`）が通らないと `.done` を作らずに止まるので，両方の照合が通った |
| 実験 1 の単位の保存の sha256 | `dai1_eval_units_sha256.txt` は 600 行で，`results/dai1/eval/units/*.npz` の 600 個と同じ数．抜き取った 10 個は今のファイルの sha256 と 10/10 一致 |
| 固定のファイルの元 | `source_sha256` の 2 つ（`dai16_probe/summary/mixsel.csv` 471c6c91…6b70，`dai16_probe_k10/summary/mixsel.csv` c83eab96…582b）は今のファイルと同じ．固定のファイルの選んだ点（12 組の n0・β・戻し先）は設計 3 節の表と全部同じ |

## 3．検算の量（設計 6 節の基準との比べ）

| 量 | 値 | 基準 | 結果 |
|---|---|---|---|
| 作り直した積と実験 1 の保存の積の差の最大（`teacher_dev_max`） | 8.26e-5（MOSEI・K = 25 の単位）．(T, K) ごとの最大は AFEW・IEMOCAP 0.98〜1.4e-5，MELD 4.7〜5.9e-5，MOSEI 5.0〜8.3e-5．5e-5 を超えた (単位, K) は 600 組のうち 12 組 | ≤ 1e-4（超えたら止める） | 中（下の指摘 3） |
| 勾配の最大（`grad_max`） | 9.9999e-7．1e-6 以上の (単位, K) は 0 組 | ≤ 1e-5 | 中 |
| 止まり（`stalled`） | 0 | — | 0 |
| 切片のずらしの残りの最大（`adj_resid_max`） | 1.0e-8 | — | 小さい |
| 作り直した提案の判定の一致（`recheck_agree_min`） | 1.000000（600 組の最小） | 1 未満なら数を書く | 全部一致 |
| 等しい重みの表と積の一致（`eq_agree_min`） | 1.000000 | 同上 | 全部一致 |
| 指数の積の作り直しの一致（`psel_agree_min`） | 1.000000 | 同上 | 全部一致 |
| 実験 1 の集計から写した水準との差（`others_B_dev_max`） | 7.1e-15．84 行すべて | ≤ 1e-6（違えば止める） | 中 |
| 切片の反復が上限に当たった (組, 表) の数（`wf_n_capped` の合計） | 0 | 記録 | 0 |
| 重みの表を探す版の並んだ表 | 1,800 組（200 単位 × 3 K × 3 候補の数）で，保存した並びの番号の数が csv の `wf_n_tied` と一致し，先頭の番号が `wf_index` と一致した．float32 の `wf_uar_K*` から許し 1e-4 で作り直した並びも保存と 1,800/1,800 一致（`code.md` の記録の項目） | 記録 | 一致 |
| 表で読み替えて掛ける版で使わなかった分類器の数（`cf_D_l05` の `n_unused`） | 全単位・全 K の合計 19 | 数を出す（設計 4 節） | 出ている（内訳は `competitor_fit.csv`） |
| `selftest` | 「selftest：通った」（`logs/dai20_selftest.log`） | — | 通った |

## 4．`summary/` の表

| ファイル | 行の数 | 期待する数 | 鍵の重複 | そのほか |
|---|---|---|---|---|
| `levels.csv` | 1,176 | 49 版 × 12 (T, K) × 種の集合 2（all・0-2） | 0 | `n_fr` は all で 50，0-2 で 15．`role` は proposal 48・A 624・ablation 144・selection 24・oracle 48・reference 48・B 168・single 72．`prop` の行の `grid` は 12 組すべてで固定のファイルの点と同じ．区間の端が値を挟まない行 0．NaN は `df`（MOSEI の 294 行．引き直しの区間は自由度を持たない）と `same_as`（印の無い版．空の文字） |
| `effects.csv` | 2,772 | 77 対 × 12 × 種の集合 3（all・0-1・2-9） | 0 | 対の集合は台本 523-526 行の一覧（77 対）と同じ．設計 5 節が並べる差（部品を外した版との差，2 つの順序の分け方，選び方 (b)(c)，参照 − 積，提案の adj − raw）はすべて一覧にある．`v_seed_all` は 2,772 行すべてにあり，`v_seed` を下回る行 0．区間の端が値を挟まない行 0．`method` は jk+seed 2,079・boot_cl+seed 693．NaN は `df`（MOSEI 693），`folds_dropped`（空の文字），`a_same_as`・`b_same_as`（空の文字），`mark`（印の無い行） |
| `per_emotion.csv` | 2,352 | 49 × 12 × 感情 4 | 0 | `recall` の NaN 0，`n_fr` = 0 の行 0 |
| `grid_levels.csv` | 192 | 16 点 × 12 | 0 | 無限大 48 個は n0 = ∞ の 4 点 × 12 の `n0` の列（意図どおり）．単位の csv から作り直した `uar_adj`・`uar_adj_others` とのずれの最大 2.1e-14．`frozen`・`loto`・`oracle` の印は 12 組すべてで 1 つずつ，作り直した点と全部同じ |
| `est_grid.csv` | 192 | 16 × 12 | 0 | `frozen` の印は 12 組すべてで 1 つ |
| `weights.csv` | 144 | 12 × モダリティ 3 × 感情 4 | 0 | — |
| `confusion.csv` | 384 | 12 × 2 版 × 16 | 0 | — |
| `competitor_fit.csv` | 84 | 12 × 7 版（wf10/100/1000・tcal・temp・tempb・cf_D_l05） | 0 | — |
| `assumptions.csv` | 12 | 12 | 0 | — |
| `source_tables.csv` | 12 | 4 T × 3 モダリティ | 0 | — |
| `others.csv` | 240 | B 84 ＋ C 156 | 0 | B は 4 × 3 × 7 ＝ 84 行で，`uar_here` の NaN 0，`|uar − uar_here|` の最大 0.0（`code.md` の記録の項目）．C は 13 手法（coral・dann・entmin・im・lp・mapctl・mme・own・saerens・self・st・t3a・t3a_b）× 12 ＝ 156 行．`n_units` は 50 が 120 行，15 が 36 行で，15 の行は勾配法の 3 手法（dann・mme・mapctl）で `seeds` が 0-2（設計 4 節の C と合う） |

全部 NaN の列は 11 ファイルのどれにも無かった．無限大は `grid_levels.csv` の `n0` の 48 個だけである．

**同じ数の印**（`same_as`・`same_counts`）：
- `effects.csv` の 2,772 行すべてで，`same_counts` の真偽が `levels.csv` の `same_as` の群の一致と同じだった．`a_same_as`・`b_same_as` は 2,772 行すべてで `levels.csv` の `same_as` と同じ．`same_counts` が真の 75 行は，差と標準誤差がすべて 0 だった．
- 設計 5 節の「作りの上で同じになる版」：n0 = 0 の 8 組で `prop_noprof` と `M_F2_K` が同じ群，`prop_n0zero` と `prop` が同じ群だった．相性を使わない点を選び直した版は，固定のファイルの `noprof_sel` が点 0 の 9 組で `M_F2_K` と同じ群，点 4・5 の 3 組（AFEW K = 5，MELD K = 5・25）で別の群だった．AFEW の `psel` は 3 つの K で `prod` と同じ群で，ほかの T では別の群だった（`code_v2.md` の N3 のとおり）．`prop_home` は 12 組すべてで `prop` と別の群だった．
- npz の判定で直接確かめた 4 組：AFEW K = 10 の `prop_noprof` と `M_F2_K`，IEMOCAP K = 5 の `prop_n0zero` と `prop`，AFEW K = 5 の `psel` と `prod` は 50 単位の全発話で判定が同じ（割合 1.0000）で，3 組とも同じ群の印．MELD K = 5（n0 = 20）の `prop_noprof` と `M_F2_K` は判定が同じ発話の割合が 0.9035 で，別の群の印．印と保存が合う．

## 5．保存からの作り直し

- **UAR の数え直し**：200 単位で，npz に判定がある (単位, K, 版, 層) の 276,600 組について，判定とテストのラベルから UAR を数え直し，単位の csv の `uar_adj`・`uar_raw` と比べた．ずれの最大は 7.1e-15．csv が NaN なのに npz に判定がある組は 0．
- **水準と差の作り直し**：塊ごと・感情ごとの数は自前で数え（`dai1_stats.counts` を使わない），点の値は自前で「分割ごとに種の平均，分割で平均」を取り，区間は `dai1_stats.effect_stats` に集計の段と同じ鍵（`row_rng("dai20", "L", c, K, v, 種の集合)` と `row_rng("dai20", "A", a, b, c, K, 種の集合)`）で渡した．4 コーパス・3 つの K・jackknife と引き直しの両方・種の集合 all・0-2・0-1・2-9 を含めた．
  - 水準 8 個（AFEW K5 `prop`，AFEW K5 `prod` 種 0-2，IEMOCAP K10 `wf100`，MOSEI K25 `prop`・`b_kd`，MELD K5 `prop_noprof`，MELD K10 `prop_loto`，IEMOCAP K25 `tcal@raw`）：値のずれの最大 0，区間の端と標準誤差のずれの最大 4.4e-16．
  - 差 11 個（`prop − prod` を AFEW K5 の all と 0-1・MOSEI K25 の 2-9，`prop − wf100` IEMOCAP K10，`prop − b_kd` MOSEI K25，`prop − prop_noprof` MELD K5，`prop_loto − prop` MELD K10，`prop@raw − tcal@raw` IEMOCAP K25，`prop_levelfree − prod` MELD K25，`prop_noint − prop_noprof` AFEW K25，`prop − joint_AVT` MELD K10 の 0-1）：値のずれの最大 8.3e-17，区間の端と標準誤差のずれの最大 2.2e-16．`same_counts` の食い違い 0．
  - 感情ごとの再現率（AFEW K5 `prop`，MOSEI K10 `prod` の 4 感情ずつ）：全体で合わせた再現率，件数 `n`，区間つきの再現率と下端のずれの最大 7.1e-15．
  - 混同行列（AFEW K5 `prop`，MELD K25 `prod`）：件数のずれ 0．

## 6．ログ

- 例外，Traceback，`SystemExit` の文，メモリ不足，収束しない警告は無かった．「済み（飛ばす）」の行は 0 で，`=== dai20 units` の見出しは 1 回だけなので，続きから流した跡は無い．`logs/dai20_units.log` の単位の行（200）は `logs/dai20_eval.log` の単位の行と同じ中身だった．
- 警告は `RuntimeWarning: Mean of empty slice` の 16 件（各 2 行）だけで，`dai16_probe.py:258`（`HOMEA` の記録）が 12 件，`dai16_probe.py:341`（`S_awayauc` の記録）が 4 件だった．どちらも各 T の最初の単位（ソースの表を T ごとに 1 回作る段）で出ており，感情の対の AUC の表の対角（定義されない値）の平均から出る．npz の NaN の数が対角の数にちょうど当たることを 1 節で確かめた．台本の試走（`code_v3.md`）でも同じ警告が出ていた（指摘 2）．
- 最初の単位（AFEW_ALL_f0_r0）のログ：55 秒，840 行，作り直しの差 6.0e-06，勾配の最大 1.0e-06，作り直した提案の判定の一致 1.0000（`code_v2.md` の記録の項目 5）．
- 所要：起動 2026-11-14 21:25:31（`selftest` の後，単位の段の見出し 21:25:34）．単位の段の終わり 02:54:13（5 時間 28 分 39 秒．単位の `sec` の合計 19,708 秒）．T ごとの 1 単位の平均は AFEW 27 秒，IEMOCAP 49 秒，MELD 115 秒，MOSEI 204 秒（最大 268 秒）で，設計 8 節の見込み（5.3〜5.8 時間）の中だった．集計の段 02:54:13〜03:07:37（13 分 24 秒）．実験 1 の単位の sha256 03:07:37〜03:07:46．`.done` 03:07:46．全体で 5 時間 42 分 15 秒．
- GPU のメモリの最大：記録が無い（指摘 1）．

## 7．計算の前の報告で「計算の後に確かめる」とした項目

| 出どころ | 項目 | 結果 |
|---|---|---|
| `code.md`「B の照合が欠けた行を素通りする」 | `others.csv` の B が 84 行で `uar_here` に NaN が無い | 84 行，NaN 0，差の最大 0.0．台本は 579-582 行で行の数と差も照合して止まる作りに直っており，それも通った |
| `code.md`「並んだ表の番号の一覧を保存しない」 | 後で同点の規則を作るときに並びの数を `wf_n_tied` と照らす | 今の台本は並びの番号（`wf_tied{m}_K{K}`）を保存している．1,800 組で保存と csv が一致し，float32 の UAR からの作り直しも一致 |
| `code.md`「`same_counts` は差の一覧の対の中だけで付く」 | 全部の版の組で同じ数の版を数える | 今の集計は `same_as` を全部の版どうしで付ける．設計 5 節の作りの上で同じになる組がすべて印どおりで，npz の判定とも合った（4 節） |
| `code.md`「由来の記録に実験 1 の単位の sha256 が無い」 | 計算の直後に sha256 を残す | 600 行．抜き取り 10 個が今のファイルと一致 |
| `code.md`・`code_v2.md` 項目 5「3 ソースの単位は試走で流していない」 | 最初の単位のログを見る | 6 節のとおり．基準の中 |
| `code_v2.md` N1 | 単位の csv の `commit` の種類と，`provenance_log.jsonl` の版と `scripts` の sha256 が全部の段で同じか | `commit` は 1 種類，`scripts` の 11 本の sha256 と版は 2 つの段で同じ |
| `code_v2.md` N2 | 切片の反復が上限に当たった数が 0 でなければ `wf1000` の行の和で書く | 0 |
| `code_v2.md` N3 | AFEW の `psel` が積と同じ | 3 つの K で同じ群の印．npz の判定でも全発話で同じ |
| `code_v2.md` N5・`code_v3.md` 指摘 1・設計 8 節「計算のあいだ・後の決まり」 | 計算のあいだに監視対象をコミットしない．固定のファイルと格子を計算の後も変えない．`unit_commits` と `summary_commit` のあいだのこの 2 つの差を見る | 341c62b〜423412c で監視対象の差 0 行．固定のファイルと `dai16_probe.py` は HEAD（07c5d67）まで差 0 行 |
| `review_ml_v3.md` n2 | `eq_agree_min` と `recheck_agree_min` が 1 未満なら数を書く | どちらも 1.000000．`psel_agree_min` も 1.000000 |
| `stats_v4.md` 指摘 1 | `v_seed_all` の列を残す | 2,772 行すべてにあり，`v_seed` 以上．IEMOCAP の `v_seed` はすべての行で 0（設計 5 節の記述どおり） |
| 設計 6 節「固定のファイル」 | 単位の段と集計の段の初めに固定のファイルの設定を照合する | 両方の段が止まらずに進んだので照合は通った．`frozen_sha256` は 2 つの段で同じ |

`define_v4.md`・`compare_v3.md`・`leak.md`・`review_affect_v3.md` の指摘は，設計の文の直しか，結果の文に書く記録で，計算の後に結果ファイルで確かめる項目は無かった．`leak.md` の L1・L2（手法の形を同じテストの上で選んだ）と `stats_v3.md` の区間の注は，結果の文の段で扱う．

## 指摘

### [記録] GPU のメモリの最大が記録されていない
- 重さ：記録する
- 該当箇所：`DAI/exp/dai20_eval.py:379-384`（単位の csv に残すのは `gpu` の名前だけ），`logs/dai20_units.log`
- 問題：単位の段は GPU のメモリの最大を csv にもログにも書かない．`provenance.json` にも無い．設計 8 節の見込み（試走で 6.4 GB）と比べられない．`torch.cuda.empty_cache()` は単位ごとに呼ばれている（387 行）．読んだだけ．
- 計算の後で直せるか：直せない（計算の時の値は残っていない）．結果の数値には影響しない．
- 求める対応：台帳には「GPU のメモリの最大は記録していない（試走で 6.4 GB）」と書く．次の台本では単位ごとに `torch.cuda.max_memory_allocated()` を csv に残す．

### [記録] 各 T の最初の単位で `Mean of empty slice` の警告が 16 件出た
- 重さ：記録する
- 該当箇所：`DAI/exp/dai16_probe.py:258`（12 件），`:341`（4 件）．`logs/dai20_eval.log` の 3〜22 行，73〜80 行，131 行，183 行
- 問題：警告は記録用の配列（感情の対ごとの AUC の表）の対角から出る．npz の NaN の数（`SR_S_homeauc`・`SR_S_awayauc` 各 36，`AUC_M_K_K*`・`AUC_M_orc_K*` 各 12）は対角の数にちょうど当たり，判定と UAR の計算の道には入らない（判定の一致の検算がすべて 1 で，UAR の数え直しのずれも 7.1e-15）．
- 計算の後で直せるか：直せる（記録だけ）
- 求める対応：台帳に「警告は対角の NaN の平均から出るもので，試走と同じ」と書く．

### [記録] 作り直しの差の最大は基準 1e-4 の 83% にあたる
- 重さ：記録する
- 該当箇所：`summary/convergence.json` の `teacher_dev_max`（8.26e-5），単位の csv の `_check` の行
- 問題：基準の中だが，MOSEI（5.0〜8.3e-5）と MELD（4.7〜5.9e-5）は AFEW・IEMOCAP（約 1e-5）より 4〜8 倍大きい．下調べの集計でも同じ大きさだった（`results/dai16_probe/summary/convergence.json` 7.5e-5，`dai16_probe_k10` 7.0e-5）．作り直した提案の判定の一致は全部 1 で，判定には表れていない．
- 計算の後で直せるか：直せる（記録だけ）
- 求める対応：結果の文の由来の節に，最大値と (T, K) を書く．別の GPU や別の版の torch で流し直すときは，この余裕が小さいことに注意する．

### [記録] 集計のコミットは単位のコミットと違う
- 重さ：記録する
- 該当箇所：`summary/convergence.json`（`unit_commits` 341c62b，`summary_commit` 423412c），`provenance.json`（`commit` 423412c）
- 問題：計算のあいだに 15 コミットが入ったため，`provenance.json` が示すコミットは単位を作ったコミットではない．監視対象の差は 0 行なので数値には影響しない．`provenance.json` だけを見て単位のコミットを 423412c と書く誤りが起こりうる．
- 計算の後で直せるか：直せる
- 求める対応：結果の文の冒頭に，単位のコミット 341c62b と集計のコミット 423412c を両方書き，両者のあいだで計算に関わるファイルの差が 0 行であることを書く．

### [記録] AFEW の jackknife は全部の行で分割 0 を外し，IEMOCAP は全部の分割を外す
- 重さ：記録する
- 該当箇所：`summary/effects.csv` の `folds_dropped`（AFEW 693 行すべて「0」，IEMOCAP 693 行すべて「0,1,2,3,4」，MELD・MOSEI は空）
- 問題：AFEW では，ある塊を除くと分割 0 のテストが無効になり，その分割の種の揺れが `v_cluster` の側に入る．設計 5 節は IEMOCAP についてこの仕組みを書いているが，AFEW の分割 0 には触れていない．実験 1 の集計（`results/dai1/stats/desc/levels.csv`）も同じ並び（AFEW「0」，IEMOCAP「0,1,2,3,4」）なので，この実験に固有の不備ではない．
- 計算の後で直せるか：直せる（表の注だけ）
- 求める対応：結果の表の注に，AFEW では分割 0 が jackknife の一部の塊で外れ，種の項の一部が `v_cluster` に入ることを書く．種の項の割合は MOSEI の行でだけ読む（設計 5 節）．

## 目録に無い種類の不備

無し．指摘 1（GPU のメモリの最大を残さない）は目録の I の「時間とメモリを見積もる」の延長にあたる．9.3 節に足すなら「単位ごとに GPU のメモリの最大（`torch.cuda.max_memory_allocated`）を結果ファイルに残す」を R の保存の項目に入れる案がある．
