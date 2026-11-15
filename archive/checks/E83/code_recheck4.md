# 関門 1：実装と出力の再検査 4（E83，2026-11-03，コミット 69d1bca）

検査の対象：`git diff 1b775e2 69d1bca -- exp/analyze_pairs_e83.py docs/exploratory-analyses.md docs/seen-data.md` の E83 の部分（台本の `elig_and_keys`・`by_corpus_str`・`p3a_eval`・`p3a_table`・`summarize` の表の列，設計 3 節 (iii)・文の規則 (2)・顔の見込みと最小の型，4 節の印の偏りの文，台帳の (6)）と直しの記録 `checks/E83/fix_applied_4.md`．台本は 69d1bca の `exp/analyze_pairs_e83.py`（sha256 4551e97e…．作業木と同じ）．1b775e2 と 69d1bca の間で `exp/`・`measure/` のほかのファイルは変わっていない．

## 判定
通す．前回の直してから進む 1 件（印を除く版の P3a (iii) の対象の決め方）は設計 3 節の文どおりに台本へ入り，全規模の出力から台本の関数を使わずに計算し直した P3a の表 208 行は食い違い 0 で，人工の 10 配置と無作為の 300 配置でも食い違い 0 だった．残る指摘は記録 2 件で，判定の値を変えない．本体はコミット 69d1bca で本番を流してよい．

## 前回の指摘の入り方

| 出所と番号 | 指摘 | 直した箇所（69d1bca の行） | 確かめ方 | 結果 |
|---|---|---|---|---|
| code_recheck3 1 [D14・S15（案）]（直してから進む），stats_recheck2 1 [S2・S10・D14]（直してから進む） | 印を除く版で (iii) の対象と `p_min_attainable`・`judgeable` が印で決まっていた | `elig_and_keys` 892〜901，`p3a_eval` 909〜991（918 行 `full_units_used = full_units_used & full_units_ref`，929〜935 行で `p_min_ref`・`judgeable_ref` を印を含む行で計算，972 行の `not_judgeable_by_flag`），`p3a_table` 1012〜1016（`m_ref=m_all`），表の列 1188〜1189．設計 10106・10118・10139 行 | (1) 全規模の出力で `check_p3a4.py`（下）が新しい規則を別の書き方で計算し直した．(2) 人工の配置 `check_synth4.py` で設計の文を直接書いた計算と比べた | 直った．全列で食い違い 0．印を含む版では使う対象と印を含む行の対象が全行で同じ，印を除く版で使う対象が印を含む行の対象を超える行は 0 |
| stats_recheck2 2 [S2]（記録） | 顔の P3a は判定できない見込み，顔の最小の型 | 設計 10108・10111 行 | 読んだだけ．4^−c・6^−c と 2 コーパスで 1/36 ＝ 0.028 は人工の配置（V2 の `p_min_attainable_after_flag` ＝ 0.02778）と合う | 入った |
| stats_recheck2 3 [S2・S4]（記録） | 当たりの文を対象の単位に限る | 設計 10118 行，台本 988 行 `n_pos_elig_not_full_selected` | 全規模の計算し直しで列が一致 | 入った（ただし印を除く版での列の意味は下の指摘 1） |
| code_recheck3 3 [L8・L11]（記録） | gate-code の 4 回目の流しと見た値を台帳に | `docs/seen-data.md` 27 行の (6) | 読んだ | 入った．07:30 の作業木の変更と 07:35 に本体が報告を読んだ時刻の順も書かれている |
| code_recheck3 4 [I15]（記録） | 作業木の変更をコミットし，流す版と検査した版を合わせる | 69d1bca | 出力の先頭の行 `commit=69d1bca… dirty_deps=False script_sha256=4551e97eebaff194` | 入った |
| code_recheck3 2 [I1]（記録） | 比較相手の表で，その比較が存在しない族の空の行 24 行 | 直していない（`fix_applied_4.md` に記載なし） | 全規模の出力で `b_p3_checks_by_comparison.csv` の `null_count_q95` の NaN が 24 行のまま | 残る（記録．指摘 2） |

## A．設計と台本の対応表

変わった項目だけを書く．ほかの行は前回の表（`code_recheck2.md`・`code_recheck3.md`）のとおりで，今回の全規模の照合でも合っていた．

| 設計の項目（3 節 (iii)，文の規則 (2)，4 節） | 台本の箇所 | 合っているか |
|---|---|---|
| 対象の単位は印を含む行で族ごとに決める（none の損・none の選択・欠けだけで決まる） | `p3a_table` 1012〜1013（`m_all`），`p3a_eval` 919（`elig_and_keys(m_ref)`） | 合う |
| `p_min_attainable` と `judgeable` は印を含む行の対象で計算する | 929〜931 | 合う |
| 印を除く版は，対象のうち印の付いた行を含まない単位で p を計算する | 918（`full_units(m) & full_units(m_ref)`．印を除いた `m` で 6 行そろう単位＝印の付いた行を持たない単位），933 | 合う（`check_p3a4.py` は「対象から印の付いた行を持つ単位を外す」という別の書き方で計算し，一致した） |
| `n_full_elig_dropped_by_flag`，`p_min_attainable_after_flag`，`judgeable_after_flag`，`n_units_full_elig_used`，`full_elig_units_by_corpus_used` | 934〜937，986〜991 | 合う |
| `P3a_verdict`：not_judgeable（印を含む行で p_min ≥ 0.05）→ not_judgeable_by_flag（印の後で p_min ≥ 0.05）→ holds／fails | 972 | 合う．無作為の 300 配置で，not_judgeable_by_flag は必ず「judgeable 真かつ judgeable_after_flag 偽」の行だけに付いた |
| `cond_p` は両方の judgeable が真のとき | 959 | 合う |
| 当たりの文に `n_pos_elig_not_full` と `n_pos_elig_not_full_selected` を並べる（「対象の外の『減らした』（効果の出うるセルだが単位の 6 対全部ではない）」） | 988 | 印を含む版では合う．印を除く版では，印で外れた単位の印の無い「減らした」もこの列に入る（指摘 1） |
| 要約の表に `n_full_elig_dropped_by_flag`・`judgeable_after_flag` を出す | 1187〜1189 | 合う．比較相手の要約（1213〜1216）には印の後の列が無いが，比較相手の表は判定に使わないので影響しない |

設計の表に載る組（世界 2 × f × K 3 × コーパス，比較 7，感度 6，直接の差 6）は出力にすべてそろった（下の格子）．

## A．小さく流した結果と全規模の照合

依頼のとおり (A)(B)(C) を全規模で流した．出力は `checks/E83/code_run/full/` に置き，照合の後の 07:45:53 にすべて消した．判定の値（減らした／増やしたの数，P3a〜P3c の当否，対ごとの点，厳密な p の値）はこの報告に書かない．

- **コマンド**：`./.venv/bin/python -W default exp/analyze_pairs_e83.py --out checks/E83/code_run/full`（既定の B ＝ 1,000，stages ab c）．出力の開始の刻印 07:37:40，終了 07:41:55，**4 分 34 秒**，最大 RSS 1.73 GB，CPU 100%，終了 0．ログに warning・error・traceback は 0 行．別の台本（scratchpad の `time_small.py`）が同時に CPU を使っていた．GPU は使わない（この環境では `nvidia-smi` が無い）．
- **provenance**：commit 69d1bca（検査したコミットと同じ），dirty_deps False，script_sha256 4551e97eebaff194，deps 5 本，inputs 1,130，コードの写しの git_diff_HEAD.patch の大きさ 0．
- **格子と E79 との照合**（`check_recheck2.py`．前回のまま使えた）：b_units_pairs.csv は 13,716 行で前回と同じ格子，鍵の重複 0，point・lo・hi・delta_d_pair・delta_d_pair_corpus・none_dirloss の NaN・Inf 0，missing 0，n_boot_valid の最小 1,000．E79 の P5-3 との照合は main 81 単位すべて対応し，点の差の最大 2.9e−15，百分位の差の最大 5.1e−7，AUC 版 3.4e−14，model_err_diff 4.0e−15（前回と同じ）．複製は oursx 9.5e−7，groupB 1.2e−6，abl_own 2.6e−15．δ の材料 2.8e−14・4.3e−14．(A) の反復の行 9,180，平均の行 918（すべて 10 反復），(B) の none_dirloss と (A) の差 0.0．(C) 52,560 行，J・J_dir の NaN 0．P3c の teacher_g は ref 300，ref0 162，pF 12，欠け 12．
- **直接の差・session_jk・b_pool_nd**（`check_direct3.py`．前回のまま使えた）：直接の差 2,430 行で点・区間・判定・none_side の食い違い 0．session_jk 756 行で区間と和集合の食い違い 0．b_pool_nd 340 行，重複 0，NaN 0．
- **P3a の表の計算し直し**（新しく書いた `checks/E83/code_run/check_p3a4.py`．`check_p3a3.py` の対象の決め方を 4 回目の規則に替えた．対象は印を含む行の 6 対すべてが効果の出うる単位，印を除く版は「そのうち `model_dependent` が真の行を持つ単位を外す」，厳密な p はコーパスごとの 24 通りの置換の直積を列挙，07:42:08〜07:42:14）：主 8 行，感度 56 行，比較相手の main 48 行，直接の差の direct_all・direct_both_chosen 96 行の計 208 行で，次の列がすべて一致した（食い違い 0）：n_units_full_elig，full_elig_units_by_corpus，n_units_full_elig_used，full_elig_units_by_corpus_used，n_full_elig_dropped_by_flag，n_pos_full_elig，n_pos_elig_not_full，n_pos_elig_not_full_selected，観測の数，p_class_perm_full_elig，p_min_attainable，judgeable，p_min_attainable_after_flag，judgeable_after_flag，cond_p，P3a_verdict，n_tests_observable，n_pos_blocks，独立の平均，帰無の 95% 点，count_threshold，cond_count，n_pos_corpora_selected_elig，cond_two_corpora，cebd64b_chance，P3a_holds_cebd64b，n_near_boundary(_selected)，n_eligible．
  - 構造の確かめ：印を含む版で使う対象と印を含む行の対象が違う行 0，印を除く版で使う対象が印を含む行の対象より多い行 0．主の表の印を含む行の対象の単位の数は，前回の報告の数（音声 5，顔 3）と同じで，印を除く版で外れる数も前回の数（音声 4，顔 3）と同じだった（真偽だけを出した）．
  - 感度の表の母集団（即興だけ，session_jk）は主と族ごとの n_tests の差 0．
- **人工の配置**（新しく書いた `checks/E83/code_run/check_synth4.py`．`results/` を読まない．台本の `p3a_table` を呼び，設計の文をそのまま書いた計算（直積の列挙）と比べた）．10 配置 × 印の有無 2 版で食い違い 0：

| 配置 | 族 | 版 | 対象（印を含む行） | 使う対象 | p_min | 印の後の p_min | verdict |
|---|---|---|---|---|---|---|---|
| V1 3 コーパスで平静–怒りが減らした，印なし | 顔 | 除く | 3 | 3 | 0.00463 | 0.00463 | （(i) で fails） |
| V2 A の喜び–悲しみ（減らしていない対）に印 | 顔 | 除く | 3 | 2 | 0.00463 | 0.0278 | 判定できる |
| V3 A・B の喜び–悲しみに印 | 顔 | 除く | 3 | 1 | 0.00463 | 0.167 | not_judgeable_by_flag |
| V3 同上 | 顔 | 含む | 3 | 3 | 0.00463 | 0.00463 | 判定できる |
| V4 効果の出うるセルが平静–怒りだけ | 顔 | 両方 | 0 | 0 | NaN | NaN | not_judgeable |
| V5 印が減らした行（平静–怒り）そのものに付く | 顔 | 除く | 3 | 2 | 0.00463 | 0.0278 | 判定できる |
| V6 対象の外の単位に印 | 顔 | 除く | 3 | 3 | 0.00463 | 0.00463 | 対象は動かない |
| A7 音声 5 コーパスのうち 2 コーパスに印 | 音声 | 除く | 5 | 3 | 0.00098 | 0.0156 | 判定できる |
| A8 音声 3 → 2 コーパス | 音声 | 除く／含む | 3 | 2／3 | 0.0156 | 0.0625／0.0156 | not_judgeable_by_flag／holds |
| A9 同じコーパスの 2 単位の片方に印 | 音声 | 除く | 6 | 5 | 0.0156 | 0.0156 | コーパスが残るので p_min は不変 |
| A10 none を選んだ単位と欠けのある単位 | 音声 | 両方 | 3 | 3 | 0.0156 | 0.0156 | 印に依らず対象外 |

  V2・V3 は gate-stats の再検査 2 の例（0.00463，0.0278，0.167）と同じ値になった．無作為の 300 配置（1〜5 コーパス × 1〜2 単位，各行に 20% で印）では，印を足しても印を含む行の対象・`full_elig_units_by_corpus`・`p_min_attainable`・`judgeable` が動いた配置 0，台本の p・印の後の p_min・使う対象が直積の列挙と違った配置 0，印を足して使う対象が増えた配置 0，verdict の順序の違反 0 だった．設計 3 節の「対象の単位と到達しうる最小の p と judgeable は，印を含む行で族ごとに決める」「印を除く版でも同じ対象を使い，その p は対象のうち印の付いた行を含まない単位で計算する」「印を含む行では判定できるが印で対象の単位が減って判定できない場合は別の区分 not_judgeable_by_flag」の 3 文どおりに動いた．
- **退化した場合・照合の効き**：今回の差分は P3a の表の作り方だけで，`stage_ab` と `aggregate_unit` は 1b775e2 から変わっていない（差分で確かめた）ので，前回の退化の検算（手法の行を none に替えると全行 same）と照合の効きの検算は流し直していない．

## A．本番の規模の見積もり

(A)(B)(C) を合わせて 4 分 34 秒，最大 RSS 1.73 GB（CPU だけ．別の台本と並走した）．前回（4 分 37 秒，1.72 GB）とほぼ同じで，設計 5 節の「10 分以内，RSS 2 GB 以内」に収まる．

## B．予想される指摘と，出力から答えられるか

前回の表（`code_recheck.md`〜`code_recheck3.md`）の行は引き続き同じ区分である．今回の差分で変わった行だけを書く．学び直しを要する行は無い．

| 表・図 | 予想される指摘 | もとにした項目か例 | (a)／(b)／(c) | (b) なら足すもの | 足す費用 |
|---|---|---|---|---|---|
| P3a (iii) | 印を除く版で，印で外れた単位は何で，外れなければ p はいくつか | S15・S17（案），D8 | (a) `b_p3_checks.csv` の `n_full_elig_dropped_by_flag`・`full_elig_units_by_corpus`・`full_elig_units_by_corpus_used`・`p_min_attainable_after_flag` と，同じ族の印を含む版の行の `p_class_perm_full_elig`．単位の名前は `b_units_pairs.csv` の `model_dependent` から数秒で出る（`check_p3a4.py` の書き方） | — | — |
| P3a (iii) | 印を除く版の「対象の外の減らした」のうち，印で外れた単位の分はいくつか | S2・S4 | (a) `b_units_pairs.csv` の none_dirloss・delta_d_pair・chosen・judgment・model_dependent から CPU 数秒で分けられる（指摘 1） | — | — |
| P3a (iii) | 印の付き方が対ごとに違うことで偏りがどちらへずれたか | S2（stats_recheck2） | (a) 同じ族の印を含む版と除く版の `p_class_perm_full_elig` を並べる．向きは設計 4 節のとおり決まらない | — | — |

## 指摘

### [S2・S4] 印を除く版では，`n_pos_elig_not_full` に印で対象から外れた単位の「減らした」も入り，設計の定義「効果の出うるセルだが単位の 6 対全部ではない」と合わない
- 重さ：記録する
- 該当箇所：`exp/analyze_pairs_e83.py:988`（69d1bca）
  ```python
  "n_pos_full_elig": n_pos_full, "n_pos_elig_not_full": int((pos_elig & ~in_full).sum()), "n_pos_elig_not_full_selected": int((pos_elig & ~in_full & sel[pairs_arr]).sum()),
  ```
  `in_full` は使う対象（印を除く版では印の付いた行を持たない単位）で作る（933 行）．設計 `docs/exploratory-analyses.md:10106`：「対象の外の『減らした』（効果の出うるセルだが単位の 6 対全部ではない）の数も書く（`n_pos_elig_not_full`）」．同 10118 行の文の規則 (2) は当たりの文にこの列を「対象の外の『減らした』の数」として並べる．
- 問題（人工の配置で確かめた）：印を除く版では，6 対すべてが効果の出うるセルなのに印で外れた単位の，印の付いていない「減らした」がこの列に入る．V2（コーパス A の喜び–悲しみに印，平静–怒りが減らした）では印を除く版の `n_pos_elig_not_full` が 1，`n_pos_elig_not_full_selected` が 1 になった．この 1 行は A の単位の平静–怒りで，A の単位は 6 対すべてが効果の出うるセルである．V3 では 2 になった．顔の f は印を除く版が主なので，顔の P3a が当たったときに文の規則 (2) のとおりこの列を「6 対全部ではない単位の減らした」と書くと，印で外れた単位の分を取り違える．判定（(i)〜(iii) と `P3a_verdict`）はこの列を使わないので，当否は変わらない．
- 計算の後で直せるか：直せる（`b_units_pairs.csv` から CPU 数秒．学び直しは要らない）．ただし列の読み方を値を見てから決めたと読まれないよう，計算の前に 1 文で決めておく．
- 求める対応：次のどちらかを計算の前に行う．(1) 設計 10106 行の括弧を「効果の出うるセルで，使う対象の単位の外にある（印を除く版では，印で外れた単位の印の無い行を含む）」に直す．(2) 台本で `n_pos_elig_dropped_by_flag`（印で外れた単位の印の無い「減らした」の数）を別の列に出し，`n_pos_elig_not_full` を印を含む行の対象の外だけで数える．台本を変えない (1) を推す．

### [I1] 比較相手の表に，その比較が存在しない族の空の行が 24 行ある（前回の記録 2 のまま）
- 重さ：記録する
- 該当箇所：`exp/analyze_pairs_e83.py` `p3a_table`（1007〜1017 行）と `summarize` の `byc`（1204〜1212 行）
- 問題（全規模で確かめた）：`b_p3_checks_by_comparison.csv` の `null_count_q95` の NaN が 24 行で，前回と同じ（abl_ref0 の単モダリティの族と abl_refb の融合の族）．n_tests 0 で塊 0 と読まれうる．判定には使わない．
- 計算の後で直せるか：直せる
- 求める対応：前回と同じ．6 節を書くときに「abl_refb・abl_ref0 の行は存在する族だけを読む」と書くか，n_tests 0 の行を書き出さない．

### [L8・L11] gate-code の再検査 4 の全規模の流しと見た値の範囲を台帳に書く
- 重さ：記録する
- 該当箇所：`docs/seen-data.md:27`
- 問題：依頼に従い，2026-11-03 07:37:40〜07:41:55 に (A)(B)(C) を全規模で `checks/E83/code_run/full/` に流し，07:42〜07:45:38 に照合し，07:45:53 に消した．私が見た値は次に限る．
  - provenance，E79 との照合の差，行・鍵の数と格子，比較ごとの same_by_definition の数，片方だけ none の 276 行の point ≠ 0 の数，n_seeds の値，区間の規則の行の数，b_pool_nd の次元と n/d の最小 0.414，P3c の teacher_g の内訳，checks.csv の行の種類
  - P3a の表の構造の列：帰無の 95% 点の範囲（主 1〜4，感度 0〜5，比較 0〜4），独立の平均の範囲，n_blocks，n_tests の値，列ごとの NaN の数（主の表で `full_elig_units_by_corpus_used`・`p_min_attainable_after_flag` が 2 行，`P3a_verdict` が 4 行（判定しない族））
  - 対象の単位の数が前回の数（音声 5・顔 3，外れる数 4・3）と同じかの真偽
  - 台本の値と私の計算し直しの値が一致したかの真偽（列の食い違いの数だけを出した）
  
  減らした／増やしたの数，P3a〜P3c の当否と `P3a_verdict` の値，対ごとの点，厳密な p の値，コーパスごとの対象の単位の内訳は見ていない．
- 計算の後で直せるか：直せる（記録だけ）
- 求める対応：`docs/seen-data.md` の E83 の行に (7) として上の流しと見た値の範囲を書く．

当てた項目：I1，I6〜I12，I14〜I17（I2〜I5 は新しい学習と選択が無いので前回の確かめのまま，I13 は当てはまらない），R1〜R13（変わった行だけ表に書いた），S1〜S4・S10・S15〜S17（案），D8・D14・D19，L8・L11．

## 目録に無い種類の不備

- S18（案）：検定の対象を印（手法の結果から作る量）で減らす版を持つときは，「対象の外」の記述の列も，対象を決めた行（印を含む行）の外と，印で外れた分とに分けて定義する．同じ列の名前のまま，版によって中身が変わるようにしない（E83 の関門 1 の再検査 4）．
