# 計算の後の点検：結果ファイルの欠けと重複（DAI-8 の下調べ，2026-11-08，単位の段のコミット a90af78，集計の段のコミット fcfe634）

対象は `results/dai8_probe/` の全部のファイルと，`logs/dai8_probe_s0.log`・`dai8_probe_s1.log`・`dai8_probe_sum.log`・`dai8_probe_launch.log` である．点検の台本は `DAI/checks/DAI-8/code_run_post/check_post.py`（rows と provenance）と `check_sum.py`（集計のファイル）で，出力は同じフォルダの `check_post.log`・`check_sum.log` にある．どちらも CPU で読むだけで，`results/` には書いていない．値（UAR，UAR の差，区間，規則の可否，読みの結果，収束の回数の内訳）は画面にも報告にも出していない．

## 判定

**通す．**設計が予定した 120 単位と 20 個の Q1 の行がそろい，重複・欠け・NaN・Inf・値域の外の値は無い．2 組は同じコミット・同じ台本の sha256 で流れた．集計の段は `--allow-partial` なしで完全さの検査を通った．再計算も集計のやり直しも要らない．

## 1．単位と各単位の中の行

| 確かめたこと | 予定 | 実際 |
|---|---|---|
| `rows/*.npz` の数 | 120（T 4 × c 3 × f 5 × r 2．入力 `results/dai1/pseudo/units/*_p3.npz` の 120 個） | 120．入力の名前と比べて欠け 0，余分 0，重複 0．T = c の単位は 0 |
| c ごと・T ごとの単位 | 各 30 | AFEW・IEMOCAP・MELD・MOSEI とも c で 30，T で 30 |
| (c, f) ごとの単位 | 6（T 3 × r 2） | 20 個の (c, f) すべてで 6 |
| `rows/*_knn.csv`（Q1） | 20（(c, f) ごとに 1 個．T が辞書順で最初で r = 0 の単位） | 20．(c, f) の重複 0，選び方の規則に合わない単位 0，対応する npz が無いもの 0 |
| テストの判定（int8）の鍵 | 1 単位 419（各空間の oracle・LDA_oracle 3 つ・EM_truth/EM_rand の t = 10 と収束，AVT のブロック対角の LDA_oracle．K ごとに none・joint・ttfused・格子 6 点・固定の点の t = 0/10/30/収束・EM_μ，K ≥ 1 で LDA_K 3 つ・EM_fromK・noneT・EM_T，AVT でブロック対角の EM とその収束の版） | 120 単位すべてで 419 鍵が予定と一致．形はテストの本数，値は 0〜3 |
| δ_k(x)（`D_`，float32） | 1 単位 118 鍵 | 120 単位すべて一致．形 (テストの本数, 4)，非有限 0 |
| 係数（`C_A_`・`C_mu_`・`C_c_`・`C_pi_`） | 1 単位 101 個の方法 × 4 | 120 単位すべて一致．非有限 0，π の和が 1 から 1e-4 以上ずれるもの 0 |
| 候補の判定（`P_`） | 1 単位 68 鍵（EM_truth・EM_rand の収束の版，K ごとに none・固定の点の t = 10・収束の版） | 120 単位すべて一致．形は候補の本数，値は 0〜3 |
| 診断（npz の `diag`） | 1 単位 189 鍵 | 120 単位すべて一致 |
| 付帯の配列 | テストの sample_id・塊・グループ・感情，候補の感情・グループ・対話，割合，n_pool，checks | すべての単位にある．長さはテストの本数か候補の本数に一致．テストの sample_id の重複 0 |
| EM の回の記録 | 収束の版は pool_hist の長さ = 回の数 + 1，10 回の版は回の数 10，pool_hist の長さ 11 | 予定と違う診断 0 |
| none の判定と保存の照合 | 不一致 0 | 120 単位の和で不一致 0（比べた数 4,036,080）．ログの 120 行も全部「不一致 0」 |

## 2．コミット番号と台本の sha256

- `provenance_shard0.json`・`provenance_shard1.json`：commit はどちらも `a90af78a2bef71a8372fcf52c9ca093853c29661`，dirty は偽，partial は偽，shard は [0, 2] と [1, 2] である．
- script_sha256 はどちらも `199efb1a…c92610` で，`git show a90af78:DAI/exp/dai8_probe.py` の sha256 と，今の作業木の台本の sha256 と一致した．
- 2 組の単位は 60 と 60 で，重なり 0，和は入力の 120 と一致した．runs は 60 と 60，finished はどちらも書かれている．
- frozen_sha256 は 2 組で同じで，今の `DAI/frozen/dai1_frozen.json` と一致した．入力の p1・p3 の sha256 は 240 個とも今のファイルと一致した．定数（α，β，固定の点，回の数，許す差など）は 2 組で同じである．
- 行（npz）と Q1 の csv のコミットは a90af78 の 1 つだけである．`provenance.json` の `rows_commits` も同じで，`n_units_found` は 120，`same_cf_lda_mismatch_max` は 0（同じ (c, f) の 6 単位で LDA_oracle の判定が 1 本も違わない）である．
- 集計の段のコミット（`summarize_commit`）は fcfe634 で，単位の段の a90af78 と違う．差分を読んだ（`git diff a90af78 fcfe634 -- DAI/exp measure DAI/run/dai8_probe.sh DAI/frozen`）．変わったのは新しい `DAI/exp/dai9_probe.py` の追加と，起動の台本の実行の許可の印だけで，`dai8_probe.py` とそれが読み込む `dai1.py`・`dai1_stats.py`・`dai_data.py`・`dai7_probe_v2.py`・`dai3.py` は変わっていない．fcfe634 の時刻（05:04）は集計の段の開始（06:55）より前で，その間にコミットは無い．`summarize_dirty` は偽である．したがって集計は単位の段と同じ版の台本で行われた．

## 3．集計のファイル

| ファイル | 予定 | 実際 |
|---|---|---|
| `cells.csv` | (c, 鍵) ごとに 1 行．4 × 419 = 1,676 行，各行の単位の数 30 | 1,676 行．重複 0，count は全部 30，mean の NaN 0，0〜100 の外 0 |
| `diffs.csv` | 全体の行は c あたり 362（A・V・T は K に依らない比べ 3＋K ≥ 1 で 18 × 4＋K = 0 で 14，AVT は 4＋19 × 4＋15）．部分集合の行は 1 つの部分集合あたり 18（AVT の 4 つの比べ） | 1,538 行 = 362 × 4 ＋ 18 × 5．部分集合は AFEW asr_ok，IEMOCAP marker・nomarker，MELD face_single，MOSEI channel で各 18 行．(c, K, 空間, 比べ, 部分集合) の重複 0．effect・lo・hi・se・n_folds・n_units の NaN 0．lo ≤ effect ≤ hi に反する行 0．n_folds は全部 5，n_units は全部 10．identical が真の行 0 |
| 規則の比べの行 | R1・R2・R3・R3′・R6・R7・R8 の 7 つの比べで，AVT・全体・K ∈ {1, 5, 10, 25} が各コーパス 4 行．R4（ldaO_oracle）と R9 の段 1（emtruth_ldaO）が各コーパス 1 行 | 7 つとも各コーパス 4 行．ldaO_oracle・emtruth_ldaO・emrand_ldaO・ldaOblk_ldaO は AVT で各コーパス 1 行 |
| `diag_units.csv` | 120 単位 × 189 鍵 = 22,680 行 | 22,680 行．重複 0，単位あたり 189．Inf 0．UAR の列は 0〜100，割合の列は 0〜1，エントロピーは 0〜log 4 の中 |
| `diag.csv` | 4 × 189 = 756 行 | 756 行．重複 0 |
| `knn_units.csv` | 20 個の (c, f) × 4 空間 = 80 行 | 80 行．(c, f, 空間) の重複 0，各コーパス 20 行．Inf 0 |
| `ami_units.csv` | 1 単位 18 行（ラベルどうし 1 行＋AVT の候補の割り当て 17 鍵） | 2,160 行．重複 0，単位あたり 18，NaN 0 |
| `cmi_units.csv` | 1 単位 19 行（AVT の候補の割り当て 17 鍵＋参照の割り当て 2 つ） | 2,280 行．重複 0，単位あたり 19．Inf 0 |
| `criteria.csv` | 4 コーパス × 1 行．R1〜R10，not_moved，読み分け，channel の確かめ，3.3 節の基準の数 | 4 行 107 列．規則の列（R1，R2，R3，R3x，R3_harm，R4，R4_effect，R6，R7，R8，not_moved，M1_reading，M1_drifted，R10_top，R10_em_top_ami，R10_ami_sentence_ok，R5_top3_top，R5_nonemo_separates，n_not_conv_at_rule）に欠け 0．`*_n_corpora` の 7 列がある．空の文字列 0．R5_n_fold は全部 5 |

NaN は次の場所にだけあり，どれも設計か台本の定義から決まる位置である．一部の行だけが NaN になる（族，列）は 0 だった．

- `criteria.csv`：`R1_chan_n_K_robust`・`R3_chan_n_K_robust`・`R8_chan_n_K_robust` が MOSEI 以外の 3 行で NaN（台本 `:731` が MOSEI だけで数える）．`R10_ref_emotion_x_D_given_E` と `R10_ref_dialogue_x_E_given_D` が 4 行で NaN（設計 8 節：参照の割り当てでは分母が 0 になる）．
- `cmi_units.csv`：参照 `ref_emotion` の `x_D_given_E`・`x_G_given_E` が 120 行，`ref_dialogue` の `x_E_given_D` が 120 行で NaN．割り当ての行（2,040 行）には NaN が無い．
- `knn_units.csv`：筋書きの 5 列が IEMOCAP 以外の 60 行で NaN．IEMOCAP の 20 行には NaN が無い．
- `diag_units.csv`：記録しない量の列だけが，その族の全部の行で NaN である．LDA_oracle の族は回の記録の列，EM_truth・EM_rand は none との比べの列，EM の格子の α = 0.1・0.9 は `tr_sinv_s`（台本 `:444` が α = 0.5 だけで記録する），固定の点以外の EM は `moved_at10`・`pool_uar_at10`，EM_truth・EM_rand 以外は `keep_init`．固定の点（`_em_a0.5_b1.0`，2,400 行）の `moved_at10`・`pool_uar_at10`・`tr_sinv_s`・`hist_at_rule`・`pool_uar_none`・`agree_none_test`・`ratio`・`entropy` に NaN は無い．EM_truth・EM_rand（960 行）の `keep_init`・`keep_init_at10`・`pool_uar_at10` にも NaN は無い．

集計の段の完全さの検査：
- 起動の台本 `DAI/run/dai8_probe.sh:23` は `--summarize` だけで，`--allow-partial` を付けていない．出力先に `provenance_partial_*.json` は無い．
- したがって，`:603` の単位の数（120），`:606-609` の行のコミット，`:651` の Q1 の行の数（80），`:792-798` の規則の行の数と R4・R9 の段 1 の NaN，`:810` の組のコミットと dirty の検査が，止めずに通った．`dai8_probe_sum.log` は 4 コーパスの「区間を作った」と「書いた」の行で終わる．
- `logs/dai8_probe.done` ができていて，`logs/dai8_probe.failed` は無い．

ログ：4 本のログに Traceback，Warning，メモリ不足，NaN，やり直しの行は無い．2 組とも「60 単位を終えた」で終わる．起動の記録は 04:39:45 に始まり，単位の段は 06:55:03，集計の段は 08:44:58 に終わった．

## 4．後で求められそうな解析のための保存

| 求められそうなもの | 保存の場所 | 学び直さずに CPU で出せるか |
|---|---|---|
| 単位ごとの値（UAR） | `rows/<単位>.npz` のテストの判定 419 鍵と `test_y`・`test_cluster`．`cells.csv` は (c, 鍵) の平均と数だけを持つ | 出せる．単位ごとの UAR の表のファイルは書かれていないが，npz から `ST.counts` と `ST.uar_full` で作り直せる．(T, c) ごとの区間も同じ npz から出せる（設計 4 節の記録のとおり） |
| EM の途中の時点 | 固定の点はテストの判定（t = 0・10・30・収束），δ と係数（t = 10・収束），候補の判定（t = 10・収束）．全部の EM について，npz の `diag` に回ごとの r の変化（`hist`）と候補の上の UAR（`pool_hist`）を全回ぶん持つ | 出せる．`diag_units.csv` は初めと最後と規則の回だけに絞っているので，回ごとの曲線は npz の `diag` から読む．格子の点・EM_fromK・EM_T・EM_μ の途中の回のテストの判定は無い（設計はこれを求めていない） |
| 近傍の一致の単位ごとの値 | `knn_units.csv`（80 行）と `rows/*_knn.csv` | 出せる．Q1 は設計どおり (c, f, 空間) ごとの値で，T と r には依らない |
| 条件つき相互情報量の単位ごとの値 | `cmi_units.csv`（2,280 行．値，帰無，割合） | 出せる．元の割り当て（`P_`）と感情・グループ・対話の印（`pool_y`・`pool_group`・`pool_dialogue`）も npz にある |
| AMI の単位ごとの値 | `ami_units.csv`（2,160 行） | 出せる |
| 3.3 節の仮定の表 | `diag_units.csv` の `n_iter_done`・`hist_at_rule`・`moved`・`moved_at10`・`ratio`・`beta_flip_last`・`entropy`・`tr_sinv_s`・`n_k_min`・`pi_err` | 出せる．上限の 100 回で止まった診断があり，その数は `n_iter_done` から出せる（3.3 節の表に出す値なので，ここには数を書かない） |

## 記録する（判定を変えない）

- **[記録] 区間の作り方の分割の除外．**`diffs.csv` の `folds_dropped` は，IEMOCAP の 398 行で 5 つの分割の全部，AFEW の 380 行で分割 0 である．IEMOCAP は塊が分割と同じなので，`dai1_stats.py:23-28` の説明のとおり種の項 V_seed が 0 になり，揺れは jackknife の項だけに入る．AFEW の分割 0 は，ある塊を除くとテストが 30 本未満になるか感情が欠けるので，その分割の種の項は jackknife の項に含めて二重に数えない（同じ説明）．どちらも定義どおりの動きで欠けではない．結果を書くときに「IEMOCAP と AFEW の分割 0 は種の揺れを別に足していない」と注に書くとよい．
- **[記録] 集計の段の provenance．**`provenance.json` の集計の段の項目には，集計の段の台本の sha256 と scikit-learn の版（AMI と相互情報量に使う）が無い．台本は単位の段と同じ版であることを 2 節で確かめた．scikit-learn の版は `requirements.txt` で固定されている．乱数の種は `boot_seed` = 0 として書かれているが，単位の段の組の provenance には `SEED` の項目が無い（台本の定数 `SEED = 0` が EM_rand の並べ替えと Q1 の帰無に使われる）．再現には足りるが，次の台本では集計の段に台本の sha256 と scikit-learn の版を，単位の段に種を書くとよい．
- **[記録] 候補の発話の sample_id．**`P_` の候補の判定と `pool_*` の配列は，候補の並びの順で保存されているが，候補の sample_id そのものは rows の npz に無い．並びは単位の段で `results/dai1/pseudo/units/<単位>_p1.npz` の `sample_pool` と照合済みなので，そこから引ける．
- **[記録] 第 3 回の記録の残り．**固定の点の `entropy`・`pool_rec`・`beta_flip_last` は収束の版の値である（`code_v3.md` の記録のとおり）．t = 10 の柔らかい割り当て r は保存されていないので，t = 10 のエントロピーを後で求められたら学び直しになる．設計 3.1・3.3 節はこれを求めていない．

## 5．再計算か集計のやり直しが要るか

どちらも要らない．欠け・重複・NaN の誤り・コミットの食い違いは見つからなかった．上の記録の項目は，結果を書くときの注か，次の台本での provenance の書き足しで足りる．
