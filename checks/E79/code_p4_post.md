# 計算の後の点検：E79 の `--families p4`（5 つの実行，`results/e79_p4/`，2026-11-02）

対象：`results/e79_p4/` の段 delta・select・measure・judge，台本 `exp/analyze_e79.py`（sha256 先頭 09f3a345b9455249），ログ `logs/e79p4_select.log`・`logs/e79p4_measure.log`・`logs/e79p4_judge.log`・`logs/e79p4_mj.done`，入力 `results/e77_<実行>/`・`results/e78_<実行>/`（と台本が既定で読んだ `results/e80c_<世界>/`）．設計は `docs/exploratory-analyses.md` の E79 の 2〜6 節と 16 節．

評価のターゲットの判定の値は点検に使っていない．数え上げの途中で，P4・P4′ の主の行の判定の種類ごとの数が 1 回画面に出たが，この報告には書かず，判断にも使っていない．検算に使った短い台本は `checks/E79/code_run_post/` に置いた（`count_units.py`，`count_select.py`，`count_rows.py`，`scan_boot.py`，`scan_inputs.py`，`verify_inputs.py`，`diff_units.py`）．

## 判定

通す．設計の範囲の単位・組み合わせ・文の行はすべてそろい，重複は 0 で，欠けはすべて設計で欠けると決めた場合（K=0 のラベル伝播，K=0 の IM とラベル伝播の固定版）か，P4 の判定に入らない参照の行（E80 の段 c の教師）であり，provenance の連鎖も切れていない．指摘は 6 件で，すべて「記録する」である．

## 1．設計の範囲と出力の行の数え上げ

| 段・表 | 設計の範囲 | 出力 | 合っているか |
|---|---|---|---|
| delta `delta.csv` | （実行, 生徒）10 通り × K 3 ＝ 30 行 | 30 行．NaN 0．全 10 通りで K が 3 つ | 合う．`delta.csv`・`delta_units.csv`・`delta_reps.csv` は E82 の reselect の delta とバイト単位で同じ |
| select `chosen.csv` | 実行 × T × 生徒 × K{0,5,25} × 族 11（世界の融合の生徒は DANN・MME・mapctl を除く 8）× 版（incl・excl．MOSEI・MELD は ns_incl・ns_excl も） | 4,434 行．計算した期待（A 594，V 528，T 396，AVT 1,476，AV 1,440）と一致．鍵の重複 0．ns の版は MOSEI・MELD だけ | 合う |
| select の候補なし | ラベル伝播の K=0 だけ（設計 2.2 節） | 142 行．すべて `lp`・K=0（＝ lp の行 426 の 3 分の 1） | 合う（設計で欠ける） |
| select `fixed.csv` | 固定版 11 種（massbal，saerens，t3a_fix，im_fix，lp_fix，dann_fix，mme_fix，mapctl_fix，coral_fix，mapctl_match_dann・_mme）× 単位．IM・LP は K≥1 だけ．融合の生徒は DANN・MME・mapctl 系なし | 1,431 行．種類 17 単位 × 3 K，単モダリティ 24，融合 10 で，im_fix・lp_fix の K=0 が 0 行．鍵（実行, T, 生徒, K, fixname）の重複 0．E82 の `fixed.csv` と同じ内容 | 合う．Co-teaching の固定版（cotfix，cotzero）は --families p4 と Co-teaching の除外（E78 の節，11-01）で無い |
| measure `rows_*.csv.gz` | 実行ごとに T × 反復 0〜9 の全行 | 5 本．すべての T で反復 10，反復ごとの行の数が同じ，鍵の重複 0，Inf 0，UAR の NaN 0，UAR が 0〜1 の外 0．NaN は鍵でない行の `uar_bfix`・`a_auc`・`e_model` だけ（設計どおり鍵の行にだけ書く） | 合う |
| measure `D_*`・`model_null_*` | （T, 生徒, 反復）ごと | A 70，V 60，T 40，AVT 160，AV 180 の組で期待と一致．NaN 0 | 合う |
| measure 複製 | `boot_<実行>_<T>.npz` は T ごと（27），塊の区間と jackknife は該当する T だけ | boot 27，bootcl 15（IEMOCAP の 2 群と MELD × 5 実行），jk 16（同じ 15 と A の RAVDESS），jkcl 10（IEMOCAP の 2 群 × 5）．全ファイルで列の数が `keys_*.csv` の鍵の数と一致．NaN 0，Inf 0 | 合う |
| judge `units.csv` P4 主の行（main，dir，excl） | 8 手法 × A6・V5・T3 コーパス × K{0,5} ＝ 224．うち LP の K=0 の 14 が欠け → 210 単位，45 の組み合わせ（設計 2.4 節） | 224 行（欠け 14 はすべて none-lp・K=0），組み合わせ 48（全部欠けの 3 を除くと 45） | 合う |
| judge `units.csv` P4′ 主の行 | 6 手法 × AVT3・AV5 × K{0,5} ＝ 96．LP の K=0 の 8 が欠け → 88 単位，22 の組み合わせ | 96 行（欠け 8 はすべて none-lp・K=0），組み合わせ 24（全部欠けの 2 を除くと 22） | 合う |
| P4・P4′ の感度の行（2.5 節） | δ 半分・2 倍，IEMOCAP 台本を除く（言葉と AVT），MOSEI・MELD の選び直し（ns_excl），逆の順序（dir_rev），MOSEI と AFEW を除く（combos の tag），種 0（nc_seed0），塊の区間 | P4 は 18 の感度 × 224 行（IEMOCAP 台本を除く 3 つは T の 48 行），P4′ は 18 × 96 行（同 AVT の 36 行）．combos と statements に `main+excl_mosei_afew`・`delta_half+excl_mosei_afew` がある | 合う |
| 対照の族 P4_ctl・P4′_ctl | coralctl−coral，mapctl−dann，mapctl−mme（P4′ は coralctl−coral だけ）× 同じ単位 × K{0,5} | P4_ctl 84 行 × 12 の感度，P4′_ctl 16 行 × 12 | 合う（逆の順序は設計で P4・P4′・P5-3 だけ．IEMOCAP 台本を除く行は対照の族に無い．指摘 [R-ctl-iem]） |
| 記述の族（P4_otherK，P4′_otherK，P4_uni_world，P4_aux，P4_desc_*，fixed，refs） | 2.4 節の表 | P4_otherK 112/感度（K=25），P4_uni_world 456/感度（19 単位 × 8 × 3 K），P4_aux 345/感度（41 単位 × 3 K × 3 − 融合の mapctl 24），fixed 1,035（main など 6 感度） | 合う |
| `units.csv` 全体 | — | 79,518 行，鍵（族, 比べるもの, 実行, 生徒, K, コーパス, 量, 感度, 版）の重複 0．δ の NaN 0．判定ありの行で有効な回が 1,000 未満の行 0 | 合う |
| 欠け（missing）の内訳 | — | 2,508 行：LP の K=0（P4 219，P4′ 129，記述の族），im_fix と lp_fix の K=0（fixed・fixed_uar 各 492 のうちの該当），E80 の段 c の教師の参照の行 372（refs・refs_dir 各 186） | 設計で欠ける場合と，P4 の判定に入らない参照の行だけ（指摘 [L-e80c]） |
| 点の値だけの行（point_only） | 切片だけを直した UAR と切片で取り戻せる分は区間を作らない（2.3 節） | P4_desc_bfix・P4_desc_ofix の 15,660 行．うち点の値が NaN の 492 行はすべて lp-none の K=0 | 合う |
| `combos.csv` | — | 25,480 行，鍵（族, 比べるもの, 実行, 生徒, K, tag, 量）の重複 0．`seen_before_prediction`，`*_nomd`，`sign_p_holm`，`r2_selection_agrees`，`r2p_selection_chose_method` の列がある | 合う |
| `statements.csv` | 22 の tag × 2 族 × subset 2（all，unseen_only）× （文の行＋偶然の見込みの行） | 176 行．（tag, 族, subset）ごとに文の行がちょうど 1 | 合う |
| 付録の表（2.5 節） | none_chosen，selection_loss，oracle_dir，refs，seeds，model_fit，model_var_ratio，da_effect，cotzero_reach，ctl_disagree，bias_fixed_negative，cot_vnorm，identity_checks | cotzero_reach・cot_vnorm 以外はある．oracle_dir 895 行（41 単位 × 3 K × 5 手法 ＋ DANN・MME 33 × 3 × 2 ＋ LP 41 × 2）で期待と一致，重複 0．model_fit 30 ＝ 10 生徒 × 3 K．bias_fixed_negative の当てはめの失敗 0．どの表にも Inf 0 | Co-teaching の 2 表は Co-teaching の除外で無い（設計の 2.5 節の文は古い．指摘 [D-cot]） |

## 2．provenance の連鎖

| 項目 | delta | select | measure | judge |
|---|---|---|---|---|
| 台本の sha256 | 09f3a345… | 09f3a345… | 09f3a345… | 09f3a345… |
| コミット（provenance の記録） | cd838b0 | 5220f52（指摘 [I12-commit]） | c4d05b6 | c4d05b6 |
| dirty・dirty_files | 偽・空 | 偽・空 | 偽・空 | 偽・空 |
| allow_incomplete・skip_commit_check・allow_dirty | 偽 | 偽 | 偽 | 偽 |
| families | p4 | p4 | p4 | p4 |
| 開始 | 14:36:45 | 14:38:04 | 14:42:31 | 15:07:47 |

- 今の `exp/analyze_e79.py` の sha256 は 09f3a345… で，4 段と同じである．台本の最後の変更は d4ce405 で，関門の再検査（ddf22da）を通った版である．
- δ：`delta.csv` の sha256 先頭 fbc4d02298e5f231 は 3727cde（14:37:33）で文書に入り，select の開始（14:38:04）より前である．select・measure・judge の provenance の δ の sha16 は同じ値で，`committed: true`，ファイルは `HEAD:docs/exploratory-analyses.md` である．
- 選択：`chosen.csv` 3bdc21b8b1603fb1，`fixed.csv` c8557bb815d36e19 は 24197c1（14:39:54）で文書に入り，measure の開始（14:42:31）より前である．measure の `select_chosen_sha256`・`select_fixed_sha256` と judge の `chosen_csv_sha256`・`fixed_csv_sha256` は今のファイルの sha256 と全桁で一致した．
- 入力：provenance に書かれた入力（delta 607 件，select 155 件，measure 822 件．measure の対数の確率 640 件は大きさと更新の時刻）を今のファイルと照らし，食い違い 0，無いファイル 0 だった（`verify_inputs.py`）．計算の後に入力は変わっていない．
- 各 `<T>.csv` のコミットは 1 つ（measure の `commit_of_eval_csv` 64 件．E77・E78 は種類の実行が f26fad0，世界の実行が 0e0e4f1）．

## 3．NaN・Inf・あり得ない値

- `rows_*`：Inf 0，UAR の NaN 0，UAR が 0〜1 の外 0．
- 複製（boot・bootcl・jk・jkcl の 68 本）：NaN 0，Inf 0，全部が NaN の鍵 0．
- `units.csv`：点の値の NaN 3,000 行はすべて欠け（2,508）か lp-none の K=0 の point_only（492）．区間の NaN 18,168 行はすべて欠けか point_only．δ の NaN 0．
- 検算の記録（`checks.csv`）：none の E77 と E78 の照合 27 件の差の最大 0.0，段 c との照合も 0.0，ブートストラップの重み 1 の恒等 270 件の差の最大 3.4e−12，LDA と `own_separation` の照合 5 件の差の最大 8.8e−14．

## 4．ログ

- 例外，メモリ不足，やり直しは無い．`logs/e79p4_mj.done` は 0．judge のログの最後の行は「判定：単位 79518 行，組み合わせ 25480 行，文の行 176 行」で，出力の行の数と一致した．
- 警告は 2 種類である．(i) `RuntimeWarning: Mean of empty slice`（`analyze_e79.py:1217,1237,1238`．`boot_rep` の中）が IEMOCAP・MELD・RAVDESS の T で出た．保存した複製に NaN は 1 つも無いので，空の平均は途中の配列で起き，出力には残っていない．(ii) `DtypeWarning`（`analyze_e79.py:2712`．`convergence_tables` が `_convp.csv` の `pseudo`・`copied_from` の列を読むとき）．台本は `copied_from.isna()` と `target.notna()` だけを使い，`pseudo` の値を使わないので，表の数には影響しない（読んだだけ）．
- delta の段のログ（`logs/e79p4_delta.log`）が無い（指摘 [記録-log]）．

## 5．入力のファイル（RunPod から写したもの）

- `results/e77_<実行>/` 5 つ（3,489 ファイル）と `results/e78_<実行>/` 5 つ（4,349 ファイル）で，0 バイトのファイルは 0 本だった．
- 全ファイルを開いた（`scan_inputs.py`）：npy はヘッダの形 × 要素の大きさとファイルの大きさが一致，npz は zip の CRC が一致，csv・json・jsonl は全部読めた．読めないファイルは 0 本．
- ファイルの種類ごとの数は，実行 × T × 反復 10 でそろっている（例：E77 の A は `<T>_r<r>_*.npy` が 7 種類 × 70，仮のターゲットの npz が 200 ＝ 2 × 25 ＋ 5 × 30）．仮のターゲットの npz は A 200，V 140，T 50，AVT 50，AV 140 で，T ごとの仮のターゲットの群 × 反復 5 の期待と一致した．
- 中身が空の csv は E78 の世界の `<T>_noise.csv` 10 本（ヘッダ `commit` だけの 7 バイト）だけである．E78 の世界の実行は `--methods dann mme mapctl xmuda coral` で Co-teaching を回していない（`provenance.json` の methods）．`exp/compare_dai2.py:651` は Co-teaching を回すときだけ誤りの割合を書くので，空は計算の設計どおりで，写しの途中で切れたものではない．P4 は読まない（select の provenance の入力に `_noise.csv` は無い）．
- 同じ場所にある `results/e77_*_aborted_*`・`results/e77_fuse-audio-video-text_a5000_20261101` は，どの段の provenance の入力にも現れない（読まれていない）．

## 指摘

### [L-e80c] P4 だけの流れが E80 の段 c の教師を読み，参照の行に P4 と関係の無い単位 1,680 行（うち欠け 372）が入った
- 重さ：記録する
- 該当箇所：`exp/analyze_e79.py:311`（`results/e80c_<世界>` があれば既定で読む），各段の provenance の `e80c_dirs`
- 問題：16 節は E80 の段 c に触れていないが，台本は `results/e80c_fuse-audio-video-text`・`results/e80c_fuse-audio-video` を読み，`refs`・`refs_dir` の族に教師 agree・ds・mlp・poolC・other_audio_wavlm・other_text_roberta・other_video の行 1,680 を足した（E82 の reselect の `units.csv` との鍵の差がちょうどこの 1,680 行で，ほかの鍵は全部一致した．`diff_units.py`）．このうち 372 行は欠けである（例：teacher:other_audio_wavlm は AV の K=0 で 18 行）．P4・P4′ と対照の族の点の値は E82 の比べの段の検算で全部一致しており（`n_other_method_units_point_differ_fixed_a` 0），判定には影響しない．`refs.csv` の行の数は 1,340（E82 の reselect は 1,060）になる．
- 計算の後で直せるか：直せる（表を作るときに教師の 5 つ以外の行を外すだけ）
- 求める対応：16 節に「--families p4 の流れも E80 の段 c の教師の行を参照の表に含む．表 J5 には設計の 5 つの教師だけを載せる」と 1 行書く．

### [I12-commit] provenance のコミットは段の終わりの HEAD で，段を起動したときのコミットと違うことがある
- 重さ：記録する
- 該当箇所：`exp/analyze_e79.py:227`（`write_prov` が段の最後に `git rev-parse HEAD` を呼ぶ）
- 問題：select の段は 14:38:04 に 3727cde の上で起動し，14:38:35 に 5220f52 がコミットされ，終わり（約 14:39:14）の HEAD の 5220f52 が記録された．3727cde..5220f52 の変更は `DAI_paper/steps/plan.md` の 1 行だけで，台本の sha256 も同じなので，この回の出力には影響しない．ただし，段の途中で `exp/` が変わるコミットが入っても，provenance は新しいコミットを記録する（`check_dirty` は起動時にだけ見る）．
- 計算の後で直せるか：直せる（記録の追加だけ）
- 求める対応：台本を次に直すときに，起動時の HEAD（`commit_at_start`）も provenance に書く．今回の記録としては，16 節の「段 (b) の選択（コミット 5220f52 で流した」は「3727cde で起動し，5220f52 で終えた（間の変更は計画の文書だけ）」と書くのが正確である．

### [D-cot] 設計 2.2・2.5 節の Co-teaching の入力と表の記述が，Co-teaching の除外の後も残っている
- 重さ：記録する
- 該当箇所：`docs/exploratory-analyses.md` の E79 の 2.2 節（`<T>_noise.csv`，Co-teaching の固定版），2.5 節（`cotzero_reach.csv`，`cot_vnorm.csv`）
- 問題：E78 の節（11-01 夜の決定）で Co-teaching は回さないと決まり，P5-4 は E79 から外したと書かれているが，E79 の 2.2・2.5 節には Co-teaching の入力と表が残っている．出力には `cotzero_reach.csv`・`cot_vnorm.csv` が無く，E78 の世界の `_noise.csv` は空である．どちらも除外の結果として正しいが，設計の文書だけを読む検査役は欠けと読む．
- 計算の後で直せるか：直せる（文書の直し）
- 求める対応：E79 の 2.2・2.5 節に「Co-teaching は除外した（E78 の節）．`_noise.csv` は空，`cotzero_reach.csv`・`cot_vnorm.csv` は作らない」と書く．

### [R-ctl-iem] IEMOCAP 台本を除いた集計が，対照の族（P4_ctl・P4′_ctl）に無い
- 重さ：記録する
- 該当箇所：`results/e79_p4/judge/units.csv` の P4_ctl・P4′_ctl（感度は 12 種で `iemocap_impro_only` が無い）．設計 2.5 節は「IEMOCAP 台本を除いた集計：言葉の種類，AVT の世界の 4 生徒」と書き，族を限っていない
- 問題：筋 4 の (3) を書く条件の 1 つは対照の族の判定である（2.4 節）．IEMOCAP 台本を除いた tag の文では，対照の族の判定が無いまま文が決まる．
- 計算の後で直せるか：直せる．measure の段は IEMOCAP の 2 群の複製を別々に保存しており（`boot_<実行>_IEMOCAP-impro.npz`），judge の段を CPU で流し直せば出せる．学び直しは要らない
- 求める対応：IEMOCAP 台本を除いた集計を論文に載せるなら，対照の族を含めない理由を 16 節に書くか，judge を流し直して対照の族にも当てる．

### [記録-empty] select の `d8_table.csv`・`kept_used.csv`・`r6_subsets.csv` が 1 バイトの空のファイルである
- 重さ：記録する
- 該当箇所：`results/e79_p4/select/`
- 問題：3 つは P5 と E80 の教師の族の表で，--families p4 では作らない．中身は改行 1 つで，`pandas.read_csv` は `EmptyDataError` で止まる．後で表をまとめる台本がこの 3 つを読むと止まる．
- 計算の後で直せるか：直せる
- 求める対応：読む側で大きさ 1 バイト以下を飛ばすか，16 節に「p4 の流れでは空」と書く．

### [記録-log] delta の段のログが残っていない
- 重さ：記録する
- 該当箇所：`logs/`（`e79p4_select.log` 以降はあるが `e79p4_delta.log` が無い）
- 問題：delta の段の provenance（`results/e79_p4/delta/provenance.json`）と `checks.csv`（13 件すべて ok）はあり，δ の検算は記録に残っている．画面の出力（欠けと食い違いの件数）だけが残っていない．E82 の reselect の delta のログ（`logs/e82a_delta.log`）は同じ入力で「欠けと食い違い 13 件」（すべて provenance の K の照合の記録）を出している．
- 計算の後で直せるか：直せる（記録だけ）
- 求める対応：16 節の δ の行に「ログは残していない．provenance と checks.csv を見よ」と書くか，次から delta もログに書く．

## 目録に無い種類の不備

- **段の途中のコミット**：provenance に段の終わりの HEAD だけを書くと，段の途中に入ったコミットが記録に混ざる（[I12-commit]）．`docs/methods.md` 9.3 節の I12 に「provenance には起動時と終了時の HEAD の両方を書き，違えば間の変更のファイルを書く」を足す案．
