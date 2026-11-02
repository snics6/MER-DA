# 計算の後の点検：実装と出力（E83，2026-11-03，結果のコミット 83cfb0d，関門 1 で検査したコミット 69d1bca）

検査の対象：`results/e83/`（66 ファイル．計算 2026-11-03 07:48:14〜07:52:35），ログ `logs/e83.log`，設計 `docs/exploratory-analyses.md` の「## E83．…」1〜5 節，関門 1 の最後の報告 `checks/E83/code_recheck4.md`．検算の台本と出力は `checks/E83/code_run/` と `checks/E83/code_run/post/` に置いた．`results/` には何も書いていない．

## 判定
通す．結果ファイルは設計の格子をすべて持ち，重複・Inf は 0，NaN はすべて設計か台本の定義で説明でき，P3a の表 208 行・判定の数の表 19 本・区間と判定 13,716 行・直接の差 2,430 行を保存した複製から台本を使わずに計算し直して食い違いは 0 だったので，計算のやり直しは要らない．ただし下書き `checks/E83/results_draft.md` 24 行の仮定 3 の文（「1/3 以上になる…180 のうち 96」）は 96 のうち 42 が比を定義できない行なので，関門 2 の前に直す（指摘 1）．

## 1．台本と出所（provenance）

| 確かめたこと | 結果 |
|---|---|
| 結果のコミット | `provenance.json` の commit ＝ 83cfb0d90f14…，`dirty_deps` ＝ False，`dirty_deps_status` ＝ 空，`untracked_exp_measure` ＝ 空．全 CSV の先頭の行も同じ（`commit=83cfb0d… dirty_deps=False script_sha256=4551e97eebaff194 boot=1000 seed=0 smoke=False targets=None stages=['ab', 'c']`） |
| 69d1bca と 83cfb0d の差 | `git diff --stat 69d1bca 83cfb0d` は `checks/E83/code_recheck4.md`・`docs/exploratory-analyses.md`（4 行）・`docs/seen-data.md`（4 行）だけで，`exp/`・`measure/` は変わっていない |
| 台本の sha256 | `exp/analyze_pairs_e83.py` は 69d1bca・83cfb0d・作業木・`results/e83/code/20261103-074814/` の写しの 4 つとも 4551e97eebaff1948cc86bfb759693e6917a73194ef03d10630cfac424aff802 |
| 依存の 4 本 | `exp/explore_pairs.py`・`exp/explore_synergy.py`・`exp/multimodal_distill.py`・`measure/data.py` は，69d1bca・作業木・写し・`deps_sha256` の 4 つで一致した |
| コードの写しの `git_diff_HEAD.patch` | 大きさ 0 |
| 入力 1,130 本 | すべて今も存在し，大きさが記録と一致した．sha256 を記録した 80 本は sha256 も一致した．入力の更新時刻の最大は 2026-11-02T14:48:52 で，計算の開始（07:48:14）より前 |
| 流し方 | 本体のセッションの記録によれば，`rm -rf results/e83 && (./.venv/bin/python exp/analyze_pairs_e83.py --out results/e83) > logs/e83.log 2>&1; echo "exit=$?" >> logs/e83.log` で流した．標準出力と標準エラーの両方がログに入り，ログは出力先の 1 行と `exit=0` の 2 行だけである．例外・警告・メモリ不足からのやり直しは 0．tmux の外で流したが，所要は 4 分 21 秒で 10 分以内の決まりに収まる |

## 2．古いファイルが混ざっていないか

`results/e83/` の 66 ファイルのうち，計算の出力 60 本と `provenance.json`・`summary.md` の更新時刻は 07:48:14〜07:52:35 の範囲にある．範囲の外は `code/20261103-074814/` の下の台本の写し 5 本だけで，写すときに元の更新時刻（07:30:01 と 10 月の日付）を保ったためである．写しのディレクトリ自体は 07:48:14 に作られ，5 本の sha256 は上の表のとおり一致した．流す命令の先頭で `rm -rf results/e83` をしているので，計算の前からのファイルは残りえない．隠しファイルは 0．

## 3．格子・重複・NaN・Inf

期待する行数は `checks/E83/code_recheck4.md` の全規模の流しの数である．

| ファイル | 期待 | 実際 | 鍵の重複 | Inf | NaN |
|---|---|---|---|---|---|
| `b_units_pairs.csv` | 13,716 | 13,716（比較 7 × 感度 6．none-abl_ref0 は 144 行，none-abl_refb は 342 行，ほかは 486 行，iemocap_impro_only は 216・54・162 行，session_jk は 126・36・90 行） | 0 | 0 | point・lo・hi・delta_d_pair・delta_d_pair_corpus・none_dirloss は 0．下の注 |
| `b_units_direct.csv` | 2,430 | 2,430（6 組） | 0 | 0 | 下の注 |
| (A) `a_none_pairs_rep.csv` | 9,180 | 9,180（平均の行 918，すべて 10 反復） | 0 | 0 | 0 |
| (C) `c_complementarity_units.csv` | 52,560 | 52,560 | 0 | 0 | 0（J・J_dir） |
| `b_p3_checks.csv`／`_sens.csv`／`_by_comparison.csv` | 8／56／144 | 8／56／144 | 0 | 0 | 下の注 |
| `b_c_p3c_rows.csv` | 486 | 486（teacher_g：ref 300，ref0 162，pF 12，missing 12） | 0 | 0 | J_teacher・J_dir_teacher の 12（teacher_g が missing の 12 行） |
| `b_pool_nd.csv` | 340 | 340 | 0 | 0 | 0 |
| 複製 `b_boot_units.npz`／`b_boot_direct.npz`／`b_jk_units.npz` | 鍵 1,314／405／774 | 同じ．B は 1,000，`n_boot_valid` の最小 1,000 | — | 0 | 0 |

ほかの 40 本の CSV も全列の重複行 0，Inf 0 だった（`checks/E83/code_run/check_post_scan.py`，出力 `checks/E83/code_run/post/scan.txt`）．値域：再現率と UAR の列は 0〜100 の中にある．負の値は差と損の列だけである．

NaN の内訳（すべて定義で説明できる）：
- `b_units_pairs.csv`：`lo_jk`・`hi_jk` の 6,048 行は区間の規則が `pct`（jackknife を使わない単位）の行と同数．`lo_session_jk`・`hi_session_jk` の 12,960 行は session_jk 以外の行（13,716 − 756）．`model_dependent`・`point_auc`・`model_err_diff` の 2,916 行はすべて rev_order の行で，設計 2 節（10088 行）の「逆の順序…模型の印は付けない」のとおり
- `b_units_direct.csv`：`model_dependent` は全行 NaN（直接の差には印が無い）．`none_side` の 2,154 行は片方だけ none の 276 行以外の行（2,430 − 276）
- P3a の表：主の表で `P3a_verdict` などの 4 行は判定しない族（言葉と融合）．`full_elig_units_by_corpus_used`・`p_min_attainable_after_flag` などの 2 行は印を除く版で使う対象が 0 になった行．比較相手の表の `null_count_q95` の 24 行は，その比較が存在しない族の空の行（`code_recheck4.md` の記録 2 のまま）
- `assumption3_level_fit.csv` の `level_fit_ratio` の 42 行：none の向きの損の中央値が 0 以下で，台本 1067 行 `lv.med_dirloss.where(lv.med_dirloss > 0)` が比を NaN にした行．`level_fit_ok` は偽になる（指摘 1）
- `assumptions.csv` の `n` の 1 行：仮定 7（名前の置換）の行で，数を持たない記述の行
- `checks.csv`：行の種類（`chosen_missing_expected` 81 行，`distinct_test_sets` 10 行）ごとに使わない列

## 4．独立の計算し直し

関門 1 の検算の台本を今の出力に当てた（`results/e83` を読むだけ）．どれも `./.venv/bin/python -W default` で流し，警告は検算の台本自身の `ResourceWarning`（ファイルを閉じていない）と，下の 1 件の `RuntimeWarning` だけだった．

| 検算 | 台本と出力 | 所要 | 結果 |
|---|---|---|---|
| 格子・provenance・E79 との照合 | `check_recheck2.py` → `post/recheck2.txt` | 60 秒 | main 81 単位すべて E79 の P5-3 と対応．点の差の最大 2.9e−15，百分位の差の最大 5.1e−7，AUC 版 3.4e−14，model_err_diff 4.0e−15．rev_order 2.4e−15，iemocap_impro_only 36 単位 2.9e−15．直接の差 oursx − abl_own・abl_ownK と E79 の ablation_replace_dir 2.9e−15．kd・groupB・abl_ownK・abl_own の none − 相手 2.6e−15 以下．複製 oursx 9.5e−7，groupB 1.2e−6．δ の材料 2.8e−14・4.3e−14．(B) の none_dirloss と (A) の差 0.0．`code_recheck4.md` の値と同じ |
| 直接の差・session_jk・b_pool_nd | `check_direct3.py` → `post/direct3.txt` | 4 秒 | 直接の差 2,430 行で点・lo・hi・判定・none_side の食い違い 0．片方だけ none の 276 行はすべて point ≠ 0 で，same_by_definition は 0．session_jk 756 行で jackknife の区間と和集合の食い違い 0，主より狭い行 0．`RuntimeWarning: invalid value encountered in scalar divide` は主の区間の幅が 0 の 66 行（すべて same_by_definition で，session_jk の幅も 0）で幅の比を割ったため．幅が正の行の比は 1.00〜1.64 |
| P3a の表（主 8 行，感度 56 行，比較相手 48 行，直接の差 96 行） | `check_p3a4.py` → `post/p3a4.txt` | 6 秒 | 208 行で `code_recheck4.md` に並べた全列の食い違い 0．印を含む版で使う対象と印を含む行の対象が違う行 0，印を除く版で使う対象が印を含む行の対象を超える行 0．主の表の対象の単位の数と印で外れる数は関門 1 の数と同じ．iemocap_impro_only・session_jk の族ごとの n_tests は主と差 0 |
| 区間と 4 区分の判定（新しく書いた） | `check_post_judg.py` → `post/judg.txt` | 数秒 | 13,716 行で，`b_boot_units.npz` の 2.5・97.5% 点と `lo_jk`・`lo_session_jk` の和集合が `lo`・`hi` と最大 8.1e−7 の差で一致（百分位の補間の差）．`judgment` を点・区間・δ_d,対 から作り直して食い違い 0．same_by_definition 1,278 行の point はすべて 0 |
| 判定の数の表 19 本（新しく書いた） | `check_post_jtab.py` → `post/jtab.txt` | 数秒 | 主・nomd・δ 半分・δ 2 倍・rev_order・即興だけ（主の行の差し替え）・session_jk（同）・比較相手 6 本・直接の差 6 本の全セルで食い違い 0 |

## 5．関門 1 の B で足すと決めた出力

`code_recheck.md`〜`code_recheck4.md` で足した出力（単位ごとの複製 `b_boot_units.npz`・`b_boot_direct.npz`，jackknife の複製 `b_jk_units.npz` とその索引，反復ごとの行 `a_none_pairs_rep.csv`・`b_pairs_rep.csv`・`b_recall_rep.csv`，感度と比較相手と直接の差の P3a の表，`b_pool_nd.csv`，`b_candidates.csv`，`b_c_p3c_rows.csv`，仮定の表 5 本，`c_*` の noscript 版）はすべて書き出され，上の検算で読めた．

## 指摘

### [D（目録の定義の項）・I1] 仮定 3 の「1/3 以上」の 96 行のうち 42 行は比を定義できない行で，下書きの文と設計の注の文が数と合わない
- 重さ：直してから進む（結果ファイルは直さなくてよい．下書きの文を直す）
- 該当箇所：台本 `exp/analyze_pairs_e83.py:1067-1068`（69d1bca）
  ```python
  lv["level_fit_ratio"] = lv.med_abs_err / lv.med_dirloss.where(lv.med_dirloss > 0)
  lv["level_fit_ok"] = lv.level_fit_ratio < 1 / 3
  ```
  設計の仮定の表（3 の行）：「満たさない（f，対）は (A) の表に『模型の誤差が向きの損の 1/3 以上』と注を付ける（`level_fit_ok` が偽の行）」．下書き `checks/E83/results_draft.md:24`：「|模型の正解率 − 実際の正解率| の中央値が none の向きの損の中央値の 1/3 以上になる（f，K，対）は 180 のうち 96」．
- 問題（計算して確かめた）：`assumption3_level_fit.csv` の 180 行のうち `level_fit_ok` が偽の 96 行は，比が 1/3 以上の 54 行と，none の向きの損の中央値が 0 以下（−3.99〜−0.018）で比が NaN の 42 行に分かれる．NaN の比は `< 1/3` が偽になるので `level_fit_ok` も偽になる．台本の扱い（比を定義できない行にも注を付ける）は注を付ける方向に厳しいので判定の値には影響しない．しかし「1/3 以上になるのは 96」という文は 42 行について事実と違う．`a_summary_K0.csv` では 9 行，`a_summary_by_K.csv` では 42 行が同じ状態である．
- 計算の後で直せるか：直せる（文を直すだけ．学び直しも計算し直しも要らない）
- 求める対応：下書きの文を「1/3 以上の（f，K，対）は 180 のうち 54，none の向きの損の中央値が 0 以下で比を定義できない（f，K，対）は 42．どちらも (A) の表に注を付ける（`level_fit_ok` が偽）」のように 2 つに分けて書く．(A) の表の注も，NaN の行には「none の向きの損の中央値が 0 以下」と書き分ける．

### [R（出力の範囲）] 設計が出すと書いた感度 2 つ（delta_corpus・excl_mosei_afew）の判定の数の表が書き出されていない
- 重さ：記録する
- 該当箇所：設計 `docs/exploratory-analyses.md` 2 節（10088 行）「…コーパスの中央値を先に取った δ での判定し直し（delta_corpus）．MOSEI と AFEW…を除いた数え上げ（excl_mosei_afew…）．どれも判定の数の表と P3a の表（3 節）を出す」．台本は 1200〜1201 行で P3a の表だけを作り，`b_judgments_by_pair_delta_corpus.csv`・`b_judgments_by_pair_excl_mosei_afew.csv` は無い．設計の表の節（93 行目付近の「表」の段落）の感度の列挙にはこの 2 つが無く，設計の中でも記述が揃っていない．
- 問題：P3a の表（`b_p3_checks_sens.csv` の 16 行）はあり，検算で一致した．判定の数の表だけが欠ける．
- 計算の後で直せるか：直せる．`b_units_pairs.csv` の main・none-oursx の行から，delta_corpus は `delta_d_pair_corpus` で 4 区分を作り直し，excl_mosei_afew は corpus が MOSEI・AFEW の行を除いて数える．`checks/E83/code_run/check_post_missing_tabs.py` でこの手順を流し，42 行の表 2 本を `checks/E83/code_run/post/` に書いた（値は見ていない．delta_corpus は 486 行，excl_mosei_afew は 306 行から作った）．学び直しは要らない．
- 求める対応：本文や付録でこの 2 つの判定の数を出すなら，上の手順で作った表であることを記録し，`checks/E83/code_run/post/` の 2 本を `results/e83/` に移すか，台本の後処理として作り直す．出さないなら，設計 10088 行の「どれも判定の数の表と」を P3a の表だけに合わせる．

### [I1] iemocap_impro_only の 30 行で `n_groups_none`（2）が `n_groups`（1）を超える
- 重さ：記録する
- 該当箇所：`exp/analyze_pairs_e83.py:534-537`．`n_groups_none` は即興と台本の両方の選択（`chs`）から数え，`n_groups` は使う群（即興だけ）から数える
- 問題（計算して確かめた）：none-abl_own の 5 単位（AVT の IEMOCAP，audio K=25，video K=5・25，text K=25，fused K=25）× 6 対で，`n_groups` ＝ 1，`n_groups_none` ＝ 2 になる．`same_by_definition` の判定は両方の群が none のときだけ付き，この 5 単位では即興も none なので判定は正しい．逆の場合（即興が none で台本が none でなく，`same_by_definition` が付かない行）は 432 行の中に 0 件だった．
- 計算の後で直せるか：直せる（列の意味の記録だけ）
- 求める対応：iemocap_impro_only の行の `n_groups_none` は台本の群も数えた値であると記録する．この列で「none を選んだ群の割合」を出すときは `chosen_by_group` の即興の値を使う．

### 関門 1 からの残り（記録）
- `code_recheck4.md` の記録 2（比較相手の表の空の行 24 行）は出力にそのまま残る．その比較が存在しない族の行として読む
- `code_recheck4.md` の記録 1（印を除く版の `n_pos_elig_not_full`）は 83cfb0d で設計の文を直した．出力の列は 69d1bca のままで，`check_p3a4.py` の定義と一致した

当てた項目：I1，I6〜I12，I14〜I17，R1〜R13，S10・S15〜S17（案），D8・D14・D19，L8．

## 目録に無い種類の不備

- 比の仮定（分母が「損」のように符号を持つ量）を設計に書くときは，分母が 0 以下の場合の扱い（比を定義しない，注を付けるか）と，その場合の注の文を計算の前に決める．台本が NaN を「満たさない」に落とすと，文書では「比が基準以上」と読まれる（E83 の計算の後の点検．仮定 3）．
