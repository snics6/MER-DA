# 計算の後の点検：実装と出力（E84，2026-11-03，計算のコミット a0efff6．関門 1 の再検査は 320dd5c）

## 判定
通す．再検査の「直してから進む」2 件と依頼の 7 点は差分に入り，結果ファイル 12 本と凍結した写しの出力 15 本に欠け・重複・Inf は無く，summary の値は保存した表から独立に計算し直した値と 1e-9 以内で一致し，同じ台本を今流し直すと全 27 本の CSV が 1 バイトも違わずに再現した．

残る指摘は「記録する」の 6 件だけである（下の指摘）．どれも数値を変えず，計算のやり直しも要らない．判定の値（G・R・R1・R2・P84 の割合）はこの報告に書かない．流し直した出力は照合の後に消し，値を含まないログだけを残した．

## 点検の範囲と手順

- 読んだもの：`git diff 320dd5c a0efff6 -- exp/analyze_source_pairs_e84.py docs/exploratory-analyses.md docs/seen-data.md`，`exp/analyze_source_pairs_e84.py`（a0efff6），`exp/analyze_source_pairs_e84_cebd64b.py`，`docs/exploratory-analyses.md` の E84 1〜5 節，`checks/E84/code_recheck.md`，`logs/e84.log`，`results/e84/provenance.json`，`results/e84/cebd64b/provenance.json`
- 流したもの（すべて CPU．`results/` には書いていない）
  - 独立な計算し直し：`checks/E84/code_run/post_check.py`（台本の関数を import せず，群の一覧と選び方を書き直した）．ログ `checks/E84/code_run/post_check.log`
  - 入力の sha256 の照合：`checks/E84/code_run/post_prov.py`．ログ `post_prov.log`
  - 再現：同じ台本 2 本を `-W always` で `checks/E84/code_run/post_main` と `post_cebd64b` に流し直し，保存済みの CSV と見出しの 1 行を除いて `cmp` で比べた．ログ `post_main.log`，`post_cebd64b.log`，`post_rerun_compare.log`．出力は比べた後に消した

## (1) 再検査の指摘が差分に入っているか

| 再検査の指摘 | 差分の箇所 | 出力での確認 | 状態 |
|---|---|---|---|
| 指摘 1：凍結した写しを同じ回に流す | `exp/analyze_source_pairs_e84_cebd64b.py`（新規）．設計 4 節の最後の項 | 写しは `git show cebd64b:exp/analyze_source_pairs_e84.py` と比べて冒頭のコメント 2 行だけが違う．`results/e84/cebd64b/provenance.json` の script_sha256 `f3ef7ae2…` は今の写しと一致．`logs/e84.log` の 2 行目が写しの回で，終了コードは 0 | 入った |
| 指摘 1(3)：P84-2 の候補に融合を戻す | `:518`（`mstar_all`），`summarize_world` の `sh`・`sh1`，`p2_chance_share=1/(len(singles)+1)`．設計 4 節 P84-2 | `modality_select.csv` に `m_star_all_A` があり，`modality_cells.csv` から計算し直した m\*all_A と 1,680 行すべてで一致（不一致 0）．単モダリティだけの割合も `p2_singles_*` の列にある | 入った |
| 指摘 1(2)：死んだ行（`if False`）を消す | `summarize_kind` の `delta_g` の行を削除し，戻り値を 1 つにした | `summary_kind.csv` に常に NaN の列は無い | 入った |
| 指摘 2：世界の δ_u の定義 | 設計 2 節の δ_u の項に「(C) の世界では，その版の候補のソースについて，世界の全 kind の…」を足した．台本 `:550-559` | `modality_cells.csv` から計算し直した世界の δ_u（20 行）と δ_u^AUC は台本と最大 3.6e-15，7.1e-15 で一致 | 入った |
| 指摘 2：R の a_AUC・a_emp 版 | `:510-521`（`aucB`，`emp_B`，`r_oos_auc`，`r_oos_emp`），`summarize_world` の `n_units_R_oos_auc_ge_delta`（δ_u^AUC と比べる），`n_units_R_oos_emp_ge_delta`（δ_u と比べる）．設計 4 節「模型の確かめ」 | r_oos_auc は `modality_cells.csv` から計算し直した値と最大 2.8e-14 で一致．r_oos_emp は生の点数が要るので出力からは計算し直せない（再現の照合だけ．下の (5)） | 入った |
| 指摘 2：(B) の AUC 版の数え上げを δ_u^AUC と比べる | `:444`，`gain_for_block` の `delta_u_auc` | `summary_kind.csv` の `n_units_G_all_auc_ge_delta` を `gain_units.csv` から数え直して一致 | 入った |
| 記録 3：合併の行の順の照合の段 | `check_mixture_rows`（`:93-103`），`Block.__init__` の `:216-217` | (A)(B) の 4 kind × 24 ターゲットで照合が走り，失敗なら止まる作り．本番は止まらずに終わった．ただし結果は `provenance.json` に書かれず，(C) の融合の kind では走らない（記録 A） | 入った（一部） |
| 記録 7：試走の出力先 | `--smoke` の既定を `checks/E84/code_run/smoke` に．`:613-614` で `--out` に古い CSV があれば止める | `results/e84_smoke` は無い | 入った |
| 記録 7：台本の写し | `:616` `shutil.copy` | `results/e84/script_copy.py` は今の `exp/analyze_source_pairs_e84.py` と同じで，sha256 `61c00b9c…` は `provenance.json` と `git show a0efff6:` の値と一致 | 入った |
| 記録 4・5：区間の書き方，IEMOCAP の対話の文 | 設計 5 節「半分の共有」 | — | 入った |
| 記録 6：台帳の比べる相手 | `docs/seen-data.md` 26 行 | — | 入った |
| 記録 10：模型の確かめの基準 | 設計 4 節「一致が 8 割以上なら a を主に読む」 | `agree_auc_share` は `best_full.csv` から計算し直して一致 | 入った |
| 記録 11：性別ごとは出せない | 設計 5 節に無い | — | 入っていない（記録 E） |
| 記録 12：単モダリティと融合でソースの発話数が違う | 設計 5 節に無い | — | 入っていない（記録 E） |

## (2) 結果ファイルの欠け・重複・NaN・Inf

期待の行数は設計の群の一覧から台本を使わずに数えた（`post_check.py`）．

| ファイル | 行 | 期待 | 重複 | Inf | NaN |
|---|---|---|---|---|---|
| cells.csv | 11,772 | Σ(ソース＋ALL)×6 対×(全発話 1＋分け方×2)＝11,772 | 0（kind, target, source, split, half, pair） | 0 | `a_emp_in` 10,920 行（半分の行．全発話の行は 0） |
| splits.csv | 160 | IEMOCAP 10，ほか 5 の和 | — | 0 | 0 |
| gain_splits.csv | 320 | 160×2 向き | 0 | 0 | 0 |
| select_pairs.csv | 1,920 | 320×6 | 0 | 0 | 0 |
| gain_units.csv | 24 | 7＋6＋4＋7 | 0 | 0 | 0 |
| best_full.csv | 144 | 24×6 | 0 | 0 | 0 |
| modality_cells.csv | 29,772 | 世界 × 版 × kind × ターゲット ×(候補＋ALL)× 6 ×(1＋分け方×2) | 0 | 0 | `a_emp_in` 27,720 行（半分の行） |
| modality_select.csv | 1,680 | AV 960＋AVT 720 | 0 | 0 | `a_B_text_roberta` 960 行（AV の世界の行．作りから空） |
| modality_units.csv | 20 | (6＋4)×2 版 | 0 | 0 | `n_src_*` の他方の世界の列（作りから空） |
| summary_kind.csv / summary_world.csv | 4 / 4 | 4 kind / 2 世界 ×2 版 | 0 | 0 | 0 |

- (kind, ターゲット) ごとのソースの集合は 24 組すべてで設計の候補＋ALL と同じだった
- 分割の単位は IEMOCAP がセッション，ほかが話者群．異なる分け方の数は最小 5
- 主の単位の数は音声 6，HuBERT 6，顔 5，言葉 3，AV 5，AVT 3 で設計 4 節と同じ
- ブートストラップの回数は (B) の 24 単位と (C) の 20 行すべてで 500，置換は 4 kind すべてで 1,000
- 値の範囲：a と a_AUC が 0〜100 の外にある行は 0．var_ratio ≤ 0 は 0．UAR の列が 0〜100 の外は 0．a − 100Φ(d/2) の最大の差は 2.8e-14
- 判定に使う列に NaN は無い
- 凍結した写しの出力 15 本：cells.csv 7,788 行（cebd64b の作り：ALL 無し，分け方 full＋seed 0〜4 × 2 半分．Σ ソース × 6 × 11 ＝ 7,788 と一致），重複 0，NaN 0，Inf 0

## (3) provenance

- `results/e84/provenance.json`：commit `a0efff6c…`，`dirty_exp_measure_metadata` は偽．入力の sha256 は 76 本（shift の点数 34，合併の点数 34，発話一覧 7，e13_pairs.csv 1）で，76 本すべてが今のファイルの sha256 と一致．`smoke` は偽，seeds 5，boot 500，perm 1000．全 CSV の見出しは同じ 1 行（commit a0efff6，dirty=False）
- `results/e84/cebd64b/provenance.json`：commit は同じ a0efff6 だが **`dirty_exp_measure` が真**．入力は 25 本（4 つの kind の shift の点数）で，25 本すべてが今のファイルと一致．原因と影響は記録 B

## (4) 凍結した写しの出力が cebd64b の台本の出力の形になっているか

なっている．写しは cebd64b の台本とコメント 2 行だけが違うので，ファイルの名前と列は cebd64b の `main()` が書くもの（cells，best，gain，gain_units，summary_kind，source_type_audio，世界ごとの modality_cells・modality_a・modality_best・modality_units，summary_world，provenance）とそろう．15 本と provenance がそろっていた．行数は上の (2) のとおりで，gain_units は音声 6，HuBERT 6，顔 5，言葉 3，modality_units は AV 5，AVT 3 の単位（cebd64b の単位．IEMOCAP は 2 群の平均）である．

## (5) summary の独立な計算し直し

`post_check.py` は台本を import せず，選び方（NaN を除いた最大，同点は辞書順で先）を書き直して計算した．

- `summary_kind.csv`：4 kind × 21 列（数え上げ，中央値，R1_room，二項の確率，AUC・emp 版の数え上げ，G_best の副の数え上げ，P84-1 の全発話と半分の割合，p1_obs_diff，agree_auc_share）を `gain_units.csv`・`best_full.csv`・`select_pairs.csv` から計算し直し，不一致 0（許す差 1e-9）
- `summary_world.csv`：4 行 × 16 列（数え上げ，R2_room，AUC・emp 版の数え上げ，旧の R の数え上げ，P84-2 の融合を含む割合と単モダリティだけの割合，偶然の割合）を `modality_units.csv`・`modality_select.csv` から計算し直し，不一致 0
- 1 段下の照合：`gain_units.csv` を `cells.csv` から計算し直すと，G_all（320 組）最大 9.5e-15，G_best 3.6e-15，G_all_auc 7.1e-15，s\*_A の名前の不一致 0，δ_u・δ_u^AUC（24 単位）7.1e-15，G_full_all 5.9e-15．`modality_units.csv` を `modality_cells.csv` から計算し直すと，r_oos・r_oos_auc・r_old（280 組 × 6 対）最大 2.8e-14，m\*_A の不一致 0
- 出力から計算し直せない列：`G_emp_all`・`r_oos_emp`・`uar_*`（生の点数が要る）と P84-1 の置換の帰無分布（乱数の順序に依る）．これらは再現の照合（下）で確かめた
- 再現：同じ台本 2 本を今（HEAD a6aa898．E84 が読む `exp/analyze_source_pairs_e84*.py`・`exp/explore_pairs.py`・`measure/data.py`・`measure/split.py` は a0efff6 から変わっていない）流し直すと，`results/e84/` の 12 本と `results/e84/cebd64b/` の 15 本の CSV が見出しの 1 行を除いて完全に一致した．`provenance.json` の inputs と checks も一致．`-W always` で警告は 0

## (6) ログ

`logs/e84.log` は 3 行（本番の回の「wrote … E13 check: 468 行，最大の差 4.4e-16」，写しの回の同じ行，`exit=0`）．例外，収束しない警告，メモリ不足からのやり直しは無い．GPU は使わない台本である．

## 指摘

### [I14] 記録 A．合併の行の順の照合の結果が provenance に残らず，(C) の融合の kind では照合が走らない
- 重さ：記録する
- 該当箇所：`exp/analyze_source_pairs_e84.py:216-217`（`if s == ALL and idx is None: check_mixture_rows(...)`），`:655` の `prov["checks"]`（e13 だけ）
- 問題（読んで確かめた）：照合は失敗すると止まる作りなので，本番が最後まで走ったことは (A)(B) の 4 kind × 24 ターゲットで照合が通ったことを示す．ただし `provenance.json` の checks には e13 しか無く，照合が走ったことは出力からは読めない．(C) では `idx` を渡すので融合の 2 kind の合併の点数は照合されない．(C) の判定は合併 ALL を使わない（ALL は `modality_cells.csv` の行にだけ出る）ので判定の値は変わらない．再検査の `recheck_all_order.py` は 6 kind × 34 ターゲット × 反復 10 の 340 組で一致を確かめている
- 計算の後で直せるか：直せる（記録の扱い）
- 求める対応：次に台本を直す機会に，照合した組の数を `prov["checks"]["mixture_rows"]` に書く．今回は報告に「(C) の ALL の行は再検査の外部の照合で確かめた」と書けば足りる

### [I12] 記録 B．凍結した写しの provenance の dirty が真である（原因は E84 が読まないファイル）
- 重さ：記録する
- 該当箇所：`exp/analyze_source_pairs_e84_cebd64b.py:332`（`git status --porcelain exp measure`．追跡していないファイルも数える），`results/e84/cebd64b/provenance.json` の `"dirty_exp_measure": true`
- 問題（流して確かめた）：本番の回（05:46 開始）は dirty が偽で，写しの回（05:52 開始）は真である．その間に作業木の `exp/` には，追跡していない `exp/analyze_projection_e85.py`（更新 05:52:40）と，06:04 に 4ed3e3d でコミットされた `exp/analyze_pairs_e83.py` の編集があった．どちらも E84 の 2 本の台本は import しない．E84 が読むコードは a0efff6 から HEAD まで変わっておらず，今流し直すと写しの 15 本が 1 バイトも違わずに再現した．したがって値は a0efff6 のコードの値である
- 計算の後で直せるか：直せる（記録の扱い）．写しは編集しない決まりなので台本は直さない
- 求める対応：E84 6 節の計算の記録に「写しの回の dirty が真なのは E83・E85 の作業中のファイルによる．E84 が読むファイルは a0efff6 と同じで，再実行で全出力が一致した（`checks/E84/code_run/post_rerun_compare.log`）」と 1 行書く

### [I12] 記録 C．凍結した写しの provenance に融合の点数と発話一覧の sha256 が無い
- 重さ：記録する
- 該当箇所：`exp/analyze_source_pairs_e84_cebd64b.py:342-344`（shift の 4 kind だけを記録．`:344`）
- 問題：写しの (C) は `shift_fuse-*` の点数と発話一覧を読むが，provenance に記録しない．cebd64b の台本の作りであり，写しは編集しない．同じ回の本番の `provenance.json` に同じファイルの sha256 があり，今のファイルとも一致する
- 計算の後で直せるか：直せる（記録の扱い）
- 求める対応：E84 6 節に「写しが読む融合の点数と発話一覧の sha256 は本番の provenance にある」と書く

### [I12] 記録 D．起動のコマンドがログに無い
- 重さ：記録する
- 該当箇所：`logs/e84.log`（3 行）
- 問題：ログには台本の出力の行と終了コードしか無く，2 本をどのコマンドで同じ回に流したかが残っていない．`provenance.json` の args（out，kinds，worlds，seeds，boot，perm，smoke）で引数は分かる
- 計算の後で直せるか：直せる
- 求める対応：E84 6 節に起動のコマンドを書く

### [D・感 Minor-3・C1] 記録 E．再検査の記録 11・12 が設計 5 節に入っていない
- 重さ：記録する
- 該当箇所：`docs/exploratory-analyses.md` E84 5 節（10172〜10183 行）
- 問題：再検査の記録 11（性別ごとの値は MELD・MOSEI・AFEW に性別の列が無いので出せない）と記録 12（単モダリティの分類器は融合の分類器よりわずかに多いソースの発話で学んでいる．例：音声 MOSEI 18,919 本，融合 AV 18,634 本）は 5 節に書かれていない．どちらも計算のやり直しは要らない
- 計算の後で直せるか：直せる
- 求める対応：結果を書く前に 5 節に 1 行ずつ足す．関門 2 の照合の前に入れる

### [D] 記録 F．`a_emp_in` は向きの逆な組で `a` と比べられない（再検査の記録 9 のまま）
- 重さ：記録する
- 該当箇所：`exp/analyze_source_pairs_e84.py:166`（`best_thr_acc`）
- 問題（読んだだけ）：直されていない．`a_emp_in` は全発話の記述の列で，判定に使う `G_emp_all`・`r_oos_emp` は半分 A の中点の閾値で測るこの関数とは別の経路（`thr_acc`）なので，判定は変わらない
- 求める対応：結果の文で `a_emp_in` を使うなら，a が 50 未満のセルを除くか，その旨を書く

## 目録に無い種類の不備

- **I（案）dirty の判定に追跡していないファイルを含めるなら，台本が読むファイルに範囲を絞る．**`git status --porcelain exp measure` は，同じ作業木で並行して書いている別の実験の台本（E84 では E83・E85）でも真になり，本当に台本の入力が変わったときと見分けられない．台本が import するモジュールと入力のパスだけについて `git status --porcelain -- <パス>` を取るか，追跡していないファイルを数えない（`--untracked-files=no`）版も並べて記録する（E84 の写しの回）
