# 本番の関門 1（5 体）の指摘の直し（E83，2026-11-03）

対象：`checks/E83/stats.md`（止める 4，直してから進む 3，記録 4），`define.md`（止める 1，直してから進む 3，記録 4），`compare.md`（直してから進む 3，記録 4），`leak.md`（直してから進む 1，記録 6），`code_recheck.md`（直してから進む 3，記録 6）．重なる指摘はまとめた．設計は `docs/exploratory-analyses.md` の E83 2〜4 節を書き換え，5 節「本番の流し方と記録の決まり」を足し，旧 5 節「計算と解析の記録」を 6 節にした（内容は触っていない）．E84・E85 の節は触っていない．台本は `exp/analyze_pairs_e83.py`．台帳は `docs/seen-data.md` 27 行．コミットはしていない．

守ったこと：評価のターゲットの値を出す (A)(B) は流していない．動作の確認は (C) の試走（`--out <scratchpad>/e83_fix2 --stages c --targets MELD --boot 20 --smoke`．29 秒，RSS 0.74 GB，警告なし．出力は消した）と，(B) の経路と P3a の関数を results/ を読まない人工データで動かした検算（下の「確かめたこと」）．`python -m py_compile` は通る．`results/` の値は開いていない（列名・鍵・配列の形と確率の和だけ）．

行番号は直した後の版（台本 1,244 行，設計 10070〜10157 行）．

## 1．P3a の帰無・条件・数の基準（stats 1〜4，define 1〜3，code_recheck 1〜2）

| 指摘 | どこをどう直したか |
|---|---|
| stats 1 [S10・D14]，define 1 [D14・D6]：割合の帰無を感情の名前の置換に替え，厳密な p を出す | 台本：`CLASS_PERM_MAPS`（72 行．4 感情の置換 24 通りを対 (k,l) → (π(k),π(l)) に写した (24, 6)），`PAIR_PERM_MAPS`（73 行．720 通り．感度の列にだけ使う），`exact_perm_p`（812〜840 行．コーパスごとに置換の数の分布を作って畳み込み，p ＝ P(数 ≥ 観測)）．`p3a_eval`（861〜918 行）で主の列 `p_class_perm_cond`，感度の列 `p_class_perm_uncond`・`p_pair_perm_uncond`．判定は `p < P_ALPHA`（63 行．0.05）．乱数は使わない．設計 10106〜10107 行 |
| stats 2 [D14・S2]，define 3 [D14・D7]，code_recheck 2 [D14]：(iii)(iv) を効果の出うるセルに限った条件つきの並べ替え 1 つにまとめ，分子も同じ集合で絞る | `p3a_eval` 861〜918 行：`elig`（none の向きの損 ≥ δ_d,対，none を選ばない，欠けでない），分子は `pos & elig`，帰無は各コーパスの分子の行をすべて同じ単位の `elig` に移す置換だけ（`exact_perm_p(..., conditional=True)`）．記述の列：`share_selected_among_pos`（全部の減らした）と `base_share`（固定 0.5／1/6），`base_share_cond`，`n_pos_not_elig`，`n_pos_not_elig_selected`，`n_corpora_with_random_perm`．設計 10106〜10107 行．土台の境界が D の揺れを含まないことは 4 節 10139 行，仮定の表の 5 行目 |
| stats 3 [S10・S11]，define 2 [D6]，code_recheck 1 [S4・S11]：数の基準を複製の帰無の 95% 点に（最低でも独立の上限） | `null_block_counts`（843〜858 行．b[i] − x を中心 0 に寄せ，「x* + (lo − x) > 0 かつ x* ≥ δ」を塊に畳んだ数の分布），`p3a_eval` の `null_count_q95`・`null_count_mean`・`chance_pos_blocks_anyof`・`count_threshold = max(q95, anyof)`・`cond_count`（899 行）．塊の数 × 0.025 は列から外した（`chance_pos` は判定の数 × 0.025 の記述だけ）．設計 10104 行，文の規則 (1) は 10118 行 |
| stats 4 [S2]：広がりは怒りの対の「減らした」が出たコーパスで数える | `n_pos_corpora_selected_elig`（902 行）を `cond_two_corpora` に使う．記述の列 `n_pos_corpora`・`n_pos_corpora_selected`．設計 10105 行 |
| stats 5 [S2 記録]：最小の型と外れたときの書き方 | 設計 10111 行（5 コーパス 1 対ずつ p ＝ 0.031，3 コーパス 2〜3 対 p ＝ 0.016，2 コーパスでは当たらない．「偏らない」とは書かない．1〜2 コーパスはコーパスごとの記述） |
| stats 10 [S4 記録]：2 族は別の予言，感度の行でも P3a | 設計 10110 行，10112 行．台本 1081〜1091 行：`b_p3_checks_sens.csv`（delta_half・delta_double・rev_order・iemocap_impro_only・session_jk・delta_corpus・excl_mosei_afew） |
| leak 1 [L8] の 3：cebd64b の規則での当否も並べる | `P3a_holds_cebd64b`（911 行．数 > 判定の数 × 0.025 かつ全部の減らしたの割合 > 固定の土台）．設計 10113 行で宣言 |
| compare 3 [C4] の規則 | 設計 10114 行：oursx と groupB の両方で 3 条件が成り立てば偏りは手法に特有とは書かない．台本 `b_p3_checks_by_comparison.csv`（1092〜1101 行）に groupB ほかの同じ表 |

## 2．比較相手と直接の差（compare 1〜5，stats 9，define 9）

| 指摘 | どこをどう直したか |
|---|---|
| compare 2 [C2・C1・C3]：abl_ownK を足す，候補の数と none を選んだ単位の数 | `COMPARISONS`（64 行）に abl_ownK（と groupB）．`n_cand_of`（297 行），`b_candidates.csv` と provenance の `candidates`（337〜338 行），単位の行の `n_cand_min`・`n_cand_by_group`・`n_groups_none`．設計 10081 行，文の規則 (3)（10118 行）に abl_own の但し書き |
| compare 3 [C4]：群 B を対に分け，種 0〜2 の平均 | `SEEDED`・`NC_SEEDED`・`strip_seed`・`seed_of`（68〜89 行．E79 と同じ），`row_table` の `pnorm`・`seed` 列（272〜273 行），`pair_diffs_target` 569〜610 行（pnorm で引き，種ごとに点・複製・照合を計算して平均．種が 3 つでなければ `seed_rows` を記録）．設計 10081 行 |
| stats 9 [S4・S1]，define 9 [D1]，compare 1 [C8]：手法どうしの直接の差，jackknife の複製の保存，「同じ対」の定義，文の規則 (3) の書き直し | `DIRECT`（66 行），`direct_comparisons`（935〜970 行．同じ (実行, コーパス) の複製の差．`both_chosen` 列，`same_pos` 列），`b_units_direct.csv`・`b_boot_direct.npz`・`b_judgments_by_pair_oursx-<相手>.csv`，P3a の表の `direct_all`・`direct_both_chosen`（1092〜1099 行）．jackknife の複製 `b_jk_units.npz`（669 行，385 行）．設計 10082 行．文の規則 (3)（10118 行）：kd で減っても教師の情報に帰せないとは言えない／oursx の重みづけの形には帰せない，abl_ownK で減れば教師の情報には帰せない，abl_own の減らないことは証拠に使わない，「帰せる」の基準は oursx − abl_ownK の塊の数が帰無の 95% 点を超えること（両方 none 以外に限っても）．教師が生徒のモダリティを含むことは 4 節 10145 行 |
| compare 4 [C8]：oursx が選んだ教師の J | `TEACHER_G`（71 行），`b_c_p3c_rows.csv` の `teacher`・`teacher_g`・`J_teacher_same_corpus`・`J_dir_teacher_same_corpus`（1140〜1147 行．pFK は欠け）．設計 10116 行 |
| compare 5 [C2 記録]：候補の数の差 | 4 節 10145 行（oursx は kd より約 25% 多い．r6 の部分集合は対に分けない）．表には `n_cand_min` |
| compare 7 [C7]，code_recheck 9 [R1・R7]：主張の範囲 | 4 節 10146 行（凍結した特徴，線形，1 組のエンコーダ．性別と E81 は範囲の外） |

## 3．P3c と (C)（define 4，stats 8，compare 4，leak 6，stats 11，code_recheck 3）

| 指摘 | どこをどう直したか |
|---|---|
| define 4 [D1・D5・D14]，stats 8 [S4]：(B) は向きだけ，J は切片を含む．判定から外す | 設計 10116 行（記述にとどめ当否は判定しない．文は「並べる」まで）．台本 1125〜1162 行：`P3c_holds` を無くし `descriptive_only=True`．向きだけの `J_dir`（`d_from_probs` 702〜712 行，790 行．候補の確率の対数から E13 の式）．土台は全体と（世界，f，K）の層で重みづけたもの（`base_share_J_pos_stratified`）．範囲 all・elig・uni・fused・nomd．(C) の f が (B) の none と別の分類器であることは 4 節 10143 行 |
| define 4 (d) の並べ替えの帰無 | 入れていない．P3c を判定から外したので要らない（本体の指示）．必要なら `b_c_p3c_rows.csv` から出せる |
| leak 6 [L6]：IEMOCAP 台本を仮のターゲットから除いた版 | `c_complementarity_summary_noscript.csv`・`_by_corpus_noscript.csv`（1123 行），P3c の `J_ref_same_corpus_noscript`（1147 行）．4 節 10143 行に偏りの向きと ns_incl を使わない理由 |
| stats 11 [S5]：再現率と φ のまとめの四分位と n_corpora | 1166 行，1172 行．設計 10099 行 |
| code_recheck 3 [R13]：「テストの上の J は保存されていない」を事実に直す | 4 節 10144 行（e77 の teacher の行にテストの対数の確率がある．出さない理由は (C) の設計．後から出せる） |
| define 5 [D8]：仮定の表 | 設計 10119〜10131 行（8 行の表と基準）．台本 `assumptions_table`（973〜1026 行．`assumptions.csv`，`assumption3_level_fit.csv`，`assumption4_var_ratio.csv`，`assumption5_negative_counts.csv`，`assumption8_rank.csv`）．(A) に `none_var_ratio` を足した（326 行）．`c_anchor_pairs.csv`（772 行．8 行目の材料） |

## 4．区間・単位・記録（stats 6〜7，define 6〜8，leak 1〜5・7，compare 6，code_recheck 4〜8）

| 指摘 | どこをどう直したか |
|---|---|
| stats 6 [S3・S1]：IEMOCAP のセッションの jackknife | `session_of`（124 行），`stage_b_run` 490〜493 行（セッションの索引と Wj_s），`pair_diffs_target` の `jk_sess`（605〜607 行），`aggregate_unit` の `jkkind = "jk_sess"`（自由度 セッション数 − 1 ＝ 4．規則 `union_pct_session_jk`，659 行），感度の行 `session_jk`（529 行）．表 `b_judgments_by_pair_session_jk.csv` と感度の P3a．人工データで IEMOCAP の区間の幅が主の 1.0〜1.5 倍になることを確かめた．設計 10084・10088 行．MELD の 8.7 は 4 節 10141 行 |
| stats 7 [S6・S9]，code_recheck 8 [I15・S9]：テストの集合が異なる数 | `checks.csv` の `distinct_test_sets`（468 行）と provenance（469 行）．4 節 10141 行 |
| define 6 [D1]：逆の順序の P3a と限界 | 感度の P3a（1081〜1091 行）．4 節 10140 行 |
| define 7 [D2]：UAR への橋 | `recall_by_emotion`（235 行），`b_recall_rep.csv`（621 行），`b_recall_units.csv`（1110 行．種・反復・群の平均）．設計 10090 行．切片だけを直した版は出さない（設計に書いた） |
| define 8 [D7]：負の対の数と語 | `a_none_pairs.csv` の `n_neg_reps`（370〜374 行），`a_summary_*.csv` の `n_neg_targets`・`n_neg_reps`，`assumption5_negative_counts.csv`．設計 10074 行（「縮小 LDA の向き（最良の向きの推定値）」） |
| leak 1 [L8]，leak 2 [L10]，leak 4 [L1-b]，code_recheck 4 [I8・L8]：台帳 | `docs/seen-data.md` 27 行を書き換えた：関門 1 の途中で流れた (A)(B)（gate-code 1 回目：03:31〜03:47，scratchpad の e83_smoke・e83_code・t_degen・t_perturb・t_boot・t_full・t_full2・t_oursx・t_oursx1000・t_sum．設計と台本のコミット 04:49 より前．本体は 1 つも開いていない．本体が 2026-11-03 06:22 に 10 個を消した），gate-code 2 回目（4ed3e3d，06:08〜06:10，`checks/E83/code_run/full/`，照合後に消した．見た値の範囲），gate-probe は (C) だけ，E75 D8・項目 8 を「設計の前に見た値」に追加，cebd64b の規則での当否を並べる宣言 |
| leak 3 [L2]：探索であること | 4 節 10137 行．文の規則 (2)（10118 行）に「設計に使ったターゲットで観察した偏り」の限定 |
| leak 5 [L5・L4]：MOSEI・AFEW を除いた数え上げ，話者の代用の向き，群 B などの理由 | 感度の P3a `excl_mosei_afew`（`EXCL_MA` 70 行，1090 行）．4 節 10141 行（向き），10145 行（群 B は足した．ほかの除去実験と群 C を分けない理由） |
| leak 7 [L7]：対話の重なり | 4 節 10142 行 |
| compare 6 [C6]，code_recheck 5 [I9]，6 [I8・I12]：記録と出力先 | `write_csv`（113〜118 行．各 CSV の先頭に `#` の行：コミット・台本の sha256・引数・開始時刻），`snapshot_code`（1183〜1192 行．台本と読み込む 4 本と `git diff HEAD` を `results/e83/code/<開始時刻>/`），`main` 1208〜1212 行（試走は results/ の下に書けない．前の本番の出力があれば止まる），1216 行（dirty は台本が読む 5 本の追跡済みの変更だけ．未追跡は `untracked_exp_measure`），provenance の `constants`・`deps_sha256`・`candidates`・`g_eff`・`sessions`・`distinct_test_sets`（1220〜1228 行，338・469・490・499 行）．設計 5 節 10150〜10156 行 |
| code_recheck 7 [R4]：感度ごとの P3a の表 | 1081〜1091 行（`b_p3_checks_sens.csv`．delta_corpus は `delta_d_pair_corpus` で判定し直す） |
| code_recheck 10 [L10・9.1]：4 体の報告 | そろった（この直しの対象） |

## 直せなかったもの・入れなかったもの

- `docs/methods.md` 9.3 節の目録への追加（stats の S12〜S14・D14 の追記，define の D15〜D16・D14 の追記，compare の C10〜C12，leak の L11〜L12，code_recheck の S12・D15・R13 の追記）は入れていない．範囲は設計と台本で，番号の振り直し（E84 の追加と重なる）は本体に任せる．
- P3c の並べ替えの帰無（define 4 (d)）は入れていない（P3c を判定から外した）．
- 切片だけを直した再現率（define 7 の後半），E81 の分割での対ごとの判定，性別ごとの内訳，oursx_r6s0〜19 の対ごとの判定は入れず，4 節に範囲の外と書いた．
- gate-code の確認の台本 `checks/E83/code_run/check_recheck.py`・`check_fix.py` は，今の台本では動かない：`pair_diffs_target` の引数（sess，Wj_s，recall_rows が増えた），`aggregate_unit` の引数（jkkind，n_sess，dmap_c，meta），`p3a_table(rows, rep_lookup, tag)`，`summarize` の引数，`stage_ab` の戻り（6 つ），出力の CSV の先頭に `#` の行（`pd.read_csv(..., comment="#")` が要る），`chosen_setting` の戻りの param が種を除いた形．再検査で流し直す必要がある．

## 確かめたこと（人工データ．`results/` を読まない）

- 厳密な p が gate-stats の例に一致：2 コーパスで怒りの 3 対がそろって減った形で，感情の置換 p ＝ 0.0625，対の置換 p ＝ 0.0025．1 コーパス 6 単位の怒りの 3 対（18 行）＋別コーパスの非怒り 1 行で，感情の置換 p ＝ 0.25，対の置換 p ＝ 0.050．効果の出うるセルが怒りの対だけのとき条件つきの p ＝ 1.0（置換を引けたコーパス 0）．
- (B)：種 3 つ（im の a0.1・a0.1s1・a0.1s2）の行を pnorm で引き，種ごとの点を平均．重み 1 の複製が点に戻る．none を選んだ群は差 0．d をずらす・NaN にすると止まる（種 1 の行で）．IEMOCAP（Ses01_F〜Ses05_M）でセッションの jackknife の行が出て，区間の幅が主の 1.0〜1.5 倍．比較 3 つ × 感度（main・δ 半分・2 倍・rev_order・impro_only・session_jk）の行がそろう．
- 直接の差（oursx − kd，oursx − abl_own）の行と判定，`both_chosen`．`b_p3_checks.csv`・`_sens.csv`（7 つの表）・`_by_comparison.csv`（direct_all・direct_both_chosen を含む）・`b_c_p3c.csv`（25 行，descriptive_only）・`b_c_p3c_rows.csv`（teacher_g）が出る．
- (C) の試走：J_dir の NaN 0，`c_anchor_pairs.csv` 2,430 行，再現率・φ のまとめに q1・q3・n_corpora，`_noscript` の版，`assumptions.csv`，provenance の constants・deps_sha256・untracked_exp_measure・code_snapshot，CSV の先頭行．`results/` への試走は止まる．

## 本番の規模の見込み

gate-code の再検査の 2 分 24 秒（比較 3 つ）に対し，比較が 5 つ，セッションの jackknife，直接の差，厳密な p（置換 24 × 5 コーパス，720 は感度）が増えたので 4〜6 分，RSS 2 GB 以内と見込む．GPU は使わない．

## git diff --stat（直した後．HEAD a8a14e2．コミットはしていない）

```
 docs/exploratory-analyses.md |  87 +++--
 docs/seen-data.md            |   2 +-
 exp/analyze_pairs_e83.py     | 826 +++++++++++++++++++++++++++++++------------
 3 files changed, 659 insertions(+), 256 deletions(-)
```
