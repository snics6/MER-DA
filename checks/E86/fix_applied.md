# E86 関門 1 の直しの記録（`checks/E86/probe.md`・`checks/E86/code.md` の指摘をどこに入れたか．2026-11-03）

担当：台本を書いた担当（サブエージェント）．設計の正本は `docs/exploratory-analyses.md` の「## E86」の節（2〜5 節を書き直した．1 節と 6 節，E83〜E85 の節は触っていない）．下書き `research_notes/研究提案_2026-11-04/段0a_設計案.md` は冒頭に「正本は docs の E86 の節．この写しは c60746e 時点のもの」の 1 行だけ足した．台本は `exp/analyze_pair_routing_e86.py`（1,120 行．sha256 の先頭 `b9c4b73bcf2d34fd`．直す前は c60746e の 722 行）．
判定の値は見ていない（試走の出力は行数・列・NaN・恒等式だけを確かめて消した）．`results/` には書いていない．git のコミットはしていない．

## 1. 指摘ごとの直した場所

### gate-probe（直してから進む 7 件）

| 指摘 | 直した内容 | 設計（`docs/exploratory-analyses.md` E86） | 台本 |
|---|---|---|---|
| 1 [S10・S13・S4・S7] 偶然の見込み n × 0.025 | 保存した複製を点で引いて中心を 0 に寄せた帰無（「(b − x) + (lo − x) > 0 かつ b − x ≥ δ_J」）の数の分布の 95% 点と平均を R86-1・R86-2 の偶然の見込みにした．n × 0.025 は下限の参照として並べるだけ．同じ帰無で K をまたがない (T, c, 対) の塊とコーパスの塊の数も出す．複製（セル × 1,000）を `boot_replicates.npz`＋`boot_index.csv` に保存．副の既定の K の 3 重と同じ c の T の重なりは，コーパスの重みを世界・T・反復・f・K・既定・候補・対で共有することで帰無に保つ | 2 節「分母と偶然の見込み」「塊に畳んだ数」，4 節の読む行の既定と R86-1，文の規則 4 | `null_pos_matrix`，`null_stats`，`counts_table`（`null95_base`・`null_mean_base`・`exceeds_null95_base`・`chance_base`），`corpus_table`（`total_corpus`・`total_Tc` の行に `null95_any_base`），`target_table` |
| 2 [S3・S1・L7] 反復ごとに独立な重み，群の数での印 | 重みはコーパスの話者群の一覧（両世界の `samples_<c>.csv` の和集合．IEMOCAP は即興と台本で同じ役者名）について 1 回だけ引き（種 [86, crc32(コーパス名)]），全世界・T・反復・f・K・既定・候補・対で共有．少なさの印は G_eff ＝ (Σn)²/Σn²（反復 0〜4 の候補を群ごとに足した本数）< 10．G_eff < 10 の単位は jackknife（IEMOCAP はセッション 5 つ，自由度 4；ほかは話者群，自由度 max(1, ⌊G_eff⌋ − 1)）の t 区間との和集合を主の区間にした．試走では MELD の G_eff 7.23（照合の 7.2 と一致），MOSEI 695 | 2 節「区間」「実効的な話者群の数」「G_eff < 10 の単位」，5 節 | `corpus_group_lists`，`main`（`W_corpus`），`UnitAcc`（`W`・`JW`・`jk_df`），`process_unit`（`JW @ num`，`weighted_auc(z, isk, JWn)`），`finalize_unit`（`lo_jk`・`hi_jk`・`interval_rule`），`jk_interval` |
| 3 [D・C] 土台のしきい値が既定ごと | δ_base ＝（世界，f，K）ごとの主の既定（ref）の δ_J．土台の主の規則 `fixed`（δ_base で固定．既定と尺度に依らない）．P86-2 は同じ（単位，対）の組で「両方・主だけ・副だけ・どちらでもない」の数と (T, c, 対) に畳んだ数で比べる | 2 節「δ_base」「土台の条件」「既定の主と副の比べ」，4 節 P86-2 | `delta_table`（`delta_base`），`units_table`（`base_ok_fixed`），`default_compare` |
| 4 [S4・D18] 規則と予言が閉じていない | (a) 文の規則を「進む／捨てる」の 1 つに決まる形に書き直した（進む ＝ avt の音声の主がセルの数でも塊の数でも帰無の 95% 点を超える．顔・言葉・av・副は記述）．(b) R86-1 は K ＝ all・土台 `fixed`・尺度 `main`・候補 s の行で読む．(c) P86-1 の avt と av の比べは同じ f どうしの土台後の割合（`share_pos_base`）．(d) 言葉の f の予言は根拠が無いので外し，記述にした（E13 4 節の表は音声と映像の列だけ）．(e) R86-3 は s の上回りが 3 個以上の行（`readable`）だけで読み，読む行を（avt，音声，主，K ＝ all，土台後）に決めた．R86-4 は単位の中央値と ±2 点を外れた単位の数で読む．(f) P86-3 は 4 つの感情を要にした割合の差の順位（記述．偶然を超えるとは書かない） | 4 節全体 | `counts_table`（`K = all` の行），`hier_table`（`readable`・`majority_h1`），`anger_units`（`diff_vs_s`・`within_2pt_of_s`），`pivot_table_emotions`（`rank`・`anger_first`） |
| 5 [D16・C10] Δ が切片を含む．規則 (2) の帰属 | 対数オッズの 2 感情の AUC の差 Δ^AUC（E79 の `pair_moments` と同じ AUC．同点 ½）を同じ区間（重みつき AUC の複製と jackknife）と判定で並べ，「上回ったが AUC では同じ水準」の数を出す．R86-5 と，主で超えたセルの過半が AUC で上回らないときの文を足した．規則 (2) は「既定を学び直して掛けたモデルに置くと余地が見えない」に限った | 2 節「切片に依らない差 Δ^AUC」，4 節 R86-5，文の規則 2 | `pair_auc`，`weighted_auc`，`auc_bundle`，`process_unit`（`Delta_auc`），`units_table`（`judge_auc_*`），`counts_table`（`n_pos_base_auc_pos`・`n_pos_base_auc_same`） |
| 6 [L1・L3・C9] T をまたいだ数え上げ | 評価のターゲット T ごとに，その T の単位だけで数えた表（R86-1T）を足し，「段 1 で T の候補の集合に振り分けの版を足すかは T の単位の判定で決める．すべての T をまとめた数は記述」と書いた．T ごとの帰無の 95% 点も同じ複製から出す | 2 節「評価のターゲット T ごとの表」，4 節 R86-1T，5 節 | `target_table`（`target_counts.csv`） |
| 7 [D8・D7・D4・L9] 土台の仮定を測る表が無い | 仮定の表（2 つの山の模型：模型の誤差 100·[Φ(d/2) − Φ(d_AUC/2)] と印 `model_dep`，分散の比；候補の半分の D：負の割合と半分の本数 ÷ 次元；IEMOCAP の 2 分割が分けたセッションの数）と，崩れたときの読み（負の割合 > 0.5 なら全セルで読む `exceeds_null95_all`；印が半分以上なら土台の規則 `auc` で読む）を値を見る前に決めた．5 節の「緩く当たる」を「厳しくなり，分母が小さくなる」に直した．IEMOCAP の 2 分割が対話を共有することは限界に書いた | 2 節「土台」「土台の仮定を測る表」，4 節の文の規則 7，5 節 | `pair_d`（`var_ratio`），`d_from_auc`，`base_D`（`n_sessions_split`），`process_unit`（`base.csv` の `d_auc`・`var_ratio`・`model_err`・`dir_err_auc`・`n_half_min_over_dim`），`units_table`（`model_dep`・`base_ok_auc`・`n_half_lt_dim`），`assumptions_table` |

### gate-probe（記録 3 件）

| 指摘 | 直した内容 |
|---|---|
| [L2・台帳] 査読 2 の必須の直し 4 と av の主の比較 | `docs/seen-data.md` の E86 の行に足した：`results/e84/`（05:48〜05:52）と `checks/E84/results_draft.md`（コミット a6aa898，06:06）が案 2 の保存（06:11．`案_2.md` の mtime 06:11:29 で確かめた）より前に存在したこと，案出しの担当に渡していないこと（本体のセッションの確認が要る），E66 と E79 17 節，av の主の比較が E83 (C) の `other_<m>` と `ref` の J の差と同じ量で，規則は c60746e（07:03）で固まっていたこと，gate-probe が E83 (C) の先頭 1 行を見たこと，gate-code が 08:05 ごろに全体の格子を T ごとに 8 つに分けて scratchpad の `e86_parts/` で流し確かめた後に消したこと（起動者は gate-code．`checks/E86/code_run/run_parts.sh`）．5 節にも av の重なりを書いた |
| [D4・I1] 決め事の設計への移動 | D2（端の行の h と分母の計算），D4（最小の本数なし．n_pair を出す），D5（K ごとの分け直し．欠けは土台に入れず数を出す），D9（過半は厳密に半分を超える）を設計 2 節に移した．D6 は指摘 2 で置き換え，D12 は指摘 1 で置き換えた．`script_notes.md` 5 節の「E83 の照合は本番でも skipped」は冒頭の注で訂正した（照合は働く） |
| [S2] δ_J は Δ の揺れから作っていない | 2 節に明記し，`delta.csv` に候補ごとの Δ の反復の SD の中央値（`median_sd_Delta_<model>`）を並べる（`delta_table`） |

### gate-code（直してから進む 4 件）

| 指摘 | 直した内容 | 場所 |
|---|---|---|
| [R6] 自分のモダリティのソースだけのモデルからの対照 | モデルに `own`（`P__<f>`）と `h_own1`（h(own, 1)）を足し，`counts.csv`・`corpus_counts.csv`・`target_counts.csv` ほかに同じ形で出す．設計 2 節の表に「対照．判定には使わない」，4 節に対照の読み方を書いた | `process_unit`（`own`・`h_own1`），`MODEL_ROLE`，`pairs.csv` の `role` 列 |
| [R9] 階層の形の疑似ラベルの 4 クラスでの質 | `anger_share.csv`（反復ごと）の行に 4 感情の再現率・UAR・怒りの適合率を足した（比較の基準 a も並べる）．単位の表 `anger_share_units.csv` を足した | `quality`，`process_unit`（`rows_a`），`anger_units` |
| [S／R1] 区間を作り直す材料 | （単位ファイル，f，K）ごとに話者群の順（`<world>__<T>__<c>__groups`），対ごとの発話の数 `den`，比較の基準の当たり `a_right`，モデルごとの当たり `right`（`right_entries`）を `group_counts.npz` に保存．試走（2 ファイル）で 1.1 MB，本番は約 135 MB の見込み（gate-code の見積もり） | `process_unit`（`gc_store`），`finalize_unit`，`main` |
| [R4] 感度の行で土台のしきい値が一緒に変わる | 土台の規則を `fixed`（主．δ_base で固定）・`scaled`（感度）・`auc`（感度）の 3 つにし，`counts.csv` ほかの行に `base_rule` 列で両方の版を出す．設計 2 節に書いた | `BASE_RULES`，`units_table`，`long_judgments` |

### gate-code（記録 6 件）

| 指摘 | 直した内容 |
|---|---|
| [I17] `none__<f>__K` は確率 | 設計 2 節の入力とモデルの表を「候補の上では確率．log_softmax はテストの `none_te__` だけ」に直した．台本の D1 の分岐は保険として残した |
| [I14] E83 との照合に pF | `compare_e83` に `g = pF` ↔（既定 pF，`model = q0`）を足した（`checks.csv` の detail に ref と pF の行数を出す） |
| [I8] 場所の固定 | `--e80`・`--e83` の引数にし，provenance と CSV の先頭行に書く |
| [I12] (T, c) の照合 | 本番（部分でない流し）では (T, c) の組が avt 8・av 24 と一致しなければ止まる（D19）．`checks.csv` に `n_Tc_pairs` の行 |
| [I9] 依存と写し | `SCRIPT_DEPS` に `exp/explore_factor.py` を足した．`explore_pairs.py`・`measure/split.py`・`measure/data.py`・`features/common.py` を出力の `code/` に写し，sha256 を provenance に書く．CSV の先頭行に台本の sha256（先頭 16 桁）と種を足した．docstring の正本の行を docs に直した |
| [R7] 範囲の外 | 設計 5 節に第 2 のエンコーダ・性別ごとの集計・`ref0__K` を既定に置く版を範囲の外と書いた |

### 直せなかったもの・本体のセッションに残すもの

- `docs/methods.md` 9.3 節の目録への追加（gate-probe の S15〜S17・D19〜D21，gate-code の I10・R6 の案）は，指示の範囲外なので入れていない．
- 台帳の「案出しの担当に渡していない」の本体のセッションによる確認（署名）は，本体が行う．
- `段0a_設計案.md` の本文は c60746e 時点のままで，正本（docs）との差は冒頭の 1 行で示した．

## 2. 検算と試走（値は見ていない）

- `py_compile` 通過．`--selftest` 9 項目すべて通過（J(a, a) ＝ 0・J ＝ 正解率の差，dir_err(D, D) ＝ 0・`base_D`・`pair_d`・分散の比，階層の形 3 項目，`to_prob`，ブートストラップの Δ ＝ 0・判定・jackknife の区間，重みつき AUC（重み 1 で rankdata の AUC と一致，整数の重みで複製した標本の AUC と一致，k の重みが 0 なら NaN），G_eff）．
- `--smoke --out <scratchpad>/e86_smoke`（avt の IEMOCAP-impro，c ＝ MELD・MOSEI，反復 1，B ＝ 50）：出力 14 本の CSV ＋ npz 3 本 ＋ `code/` 4 本．`pairs.csv` 1,728 行 25 列（＝ 2 × f 3 × K 3 × 既定 2 × モデル 8 × 対 6），`base.csv` 108 行 29 列，`units.csv` 1,512 行 71 列，`counts.csv` 3,024 行，`corpus_counts.csv` 6,048 行，`target_counts.csv` 3,024 行，`default_compare.csv` 1,512 行，`pivot_counts.csv` 6,048 行，`assumptions.csv` 9 行，`anger_share.csv`・`anger_share_units.csv` 324 行，`hier_counts.csv` 864 行，`checks.csv` 12 行．NaN は想定の列だけ（反復 1 つなので δ_J・δ_base は無く判定は missing；jackknife の列は MELD の単位にだけ値）．恒等式（J ＝ 100·(acc_model − acc_a)，q⁰ の Δ ＝ Δ^AUC ＝ 0，怒りを含まない対の h1・h05・h_own1 の Δ ＝ 0，lo ≤ hi，lo_auc ≤ hi_auc）が全行で成立．`boot_replicates.npz` は (1,512, 50) × 2，`group_counts.npz` 74 鍵．G_eff：MELD 7.23（few_groups，jackknife は話者群 215），MOSEI 695（none）．コーパスの話者群の数：AFEW 282，CREMA-D 91，IEMOCAP 10，MELD 283，MOSEI 2,614．出力は確かめた後に消した．
- 所要：試走（B ＝ 50）は 2 ファイルで 58 秒（MELD 21 秒，MOSEI 16 秒）．**B ＝ 1,000 の検査用の流し**（`--worlds avt --targets IEMOCAP-impro --reps 1 --boot 1000`．出力は消した）では MELD 65 秒，MOSEI 84 秒で，重みつき AUC の複製（1,000 回 × 6 対 × 12 のモデル × f 3 × K 3）が時間の大半である（土台の LDA は 1 ファイル 10 秒前後）．
- 本番の見込み：候補の多い c（MELD・MOSEI）の単位ファイルは avt で 20 本（約 75 秒），av で 40 本（f が 2 つなので約 50 秒），ほかの 100 本は 10〜20 秒．合計 **60〜70 分**（CPU，4 スレッド）．出力は約 150 MB（`group_counts.npz` が大半）．重みの種はコーパス名で決まるので，T ごとに分けて流しても全体を 1 回で流したときと同じ値になる（gate-code の検査の流し方が使える）．

## 3. `git diff --stat`（HEAD ＝ addda65）

```
 docs/exploratory-analyses.md                       | 136 ++--
 docs/seen-data.md                                  |   2 +-
 exp/analyze_pair_routing_e86.py                    | 862 +++++++++++++++------
 research_notes/研究提案_2026-11-04/段0a_設計案.md  |   2 +
 4 files changed, 720 insertions(+), 282 deletions(-)
```

ほかに `checks/E86/script_notes.md`（追跡済み）の冒頭に注を足し，本ファイル（未追跡）を作った．`checks/E86/probe.md`・`code.md` は検査役が置いたままで未追跡．
