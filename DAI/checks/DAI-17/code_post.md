# 計算の後の点検：結果ファイルの欠けと重複（DAI-17 下調べ，2026-11-14，コミット 48212d3）

## 判定

通す．4 つの段の出力は欠け 0・重複 0・非有限 0 で，各段のコミットは 1 つ（48212d3，dirty は偽）だった．`this_script_sha256` は検査した版と同じだった．集計の 21 ファイルの行数は設計から出る数とすべて一致し，抜き取りの作り直しと数えの作り直しの不一致は 0 だった．残る 3 点は結果の文の書き方にかかわるもので，記録する（下の「指摘」）．とくに設計 3.9 節の依存の表では，K = 5 で 60 組のうち 28 組の c′ の点が選び直しで変わった．R5 の 3 つの疑似ラベルのそれぞれについて，4 コーパスすべてに変わった c′ がある．

- 読んだもの：設計 `DAI/docs/実験17_下調べ_設計.md`（第 5 版．sha256 `6e4b087b…`．48212d3 でコミット済み），台本 `DAI/exp/dai17_probe.py`（48212d3），起動の台本 `DAI/run/dai17_probe.sh`，計算の前の検査 `code_v4.md`・`code_v5.md`，`results/dai17_probe/` の全体，ログ 4 本．
- GPU は使っていない（`CUDA_VISIBLE_DEVICES=""` で CPU だけで流した）．計算はやり直していない．`results/` のファイルは読むだけにした．`results/dai1/eval/` は開いていない．
- 画面に出したのは，数・一致・不一致・差の最大・検算の量だけである．UAR，差，判定，選んだ点は見ていない．
- 検算の台本と出力は `DAI/checks/DAI-17/code_run_post/` に置いた：`check_inventory.py`→`.out`，`check_summary.py`→`.out`，`dependency_table.py`→`.out`・`dependency_count.csv`・`dependency_detail.csv`，`check_extra.out`．

## 1．単位・組の欠けと重複，コミット，台本の sha256

| 項目 | 期待 | 実際 |
|---|---|---|
| `units/`（A） | 120 単位（実験 1 の `pseudo/<T>_tether_w.json` の一覧から作った名前） | npz 120・csv 120．欠け 0，余り 0，ほかのファイル 0 |
| `reweight/`（B） | 60 組 | npz 60・csv 60．欠け 0，余り 0 |
| `students/`（C） | 120 単位 | npz 120・csv 120．欠け 0，余り 0 |
| `srcdb/` | ソースの組 6 × 枠 4 = 24 | 24．欠け 0 |
| 単位のソースが T と c 以外の 2 コーパスか | 120 単位 | 外れ 0 |
| csv の行 | A 1002，B 730，C 596（どのファイルも同じ） | A 120,240 行（全ファイル 1002），B 43,800 行（全ファイル 730），C 71,520 行（全ファイル 596） |
| 同じ識別子の重複した行 | 0 | A 0（識別子：ファイル，K，version，space，modality，by，level，name），B 0（K，modality，arm，seed，grid，source），C 0（K，student，teacher，grid） |
| csv の `commit` | 1 つ | 3 段とも `48212d35f652e61982e3c3b154e20b4ea2facf57` だけ |
| csv の `dirty` | 偽 | 3 段とも False だけ |
| A の `dai1_commits` | T ごとに 1 つ | 4 つの T でそれぞれ 1 種類 |
| `provenance_log.jsonl` | 段ごとに 1 行 | units 03:01，reweight 04:12，students 04:36，summary 05:55 の 4 行．どの行もコミット 48212d35，dirty False，`this_script_sha256` = `edfcc3de9d85be815b3914c1fe5aab4fa4e2160d3f484b25172d96f5a5893ae9`．`scripts` と `inputs_sha256` の中身は 4 行で同じ（各 1 種類）．設計の sha256 は 4 行とも今の設計の文書と同じ |
| `provenance.json` | 集計の段 | stage summary，コミット 48212d35…，dirty False，sha256 は上と同じ |
| 終わりの印 | | `logs/dai17_probe.done`（06:02:47）あり，`logs/dai17_probe.failed` なし．`units.done`・`reweight.done`・`students.done` あり |
| `frozen_sel.json` | (c 4 × K 2 × 版 11) = 88 点 | 88 点．`units_commit` は 48212d3 だけ，`n_units` 120，作られた時刻は生徒の段の初め（04:36:44） |

生徒の段の各単位の疑似ラベルは 6 つずつそろった（全 120 単位で 6．試走で `prod` だけだった形は本計算では無い）．(単位, K, 生徒, 疑似ラベル) は 5,760 組で，格子 12 点の欠けは 0，実験 1 の設定の点（`fixed`）は各組にちょうど 1 つあった．B の格子 14 点は (組, K, モダリティ, 重み, 種) のすべてで欠け 0 だった．

## 2．npz と csv の中身（設計 5 節）

| 出力 | 確かめたこと | 結果 |
|---|---|---|
| `units/*.npz` | 145 版 × 5 層 × K 2 の判定，参照 5 版 × 2 層 × K 2 の判定，近いソースの発話の答えの確率（17 通り × テストと候補），10 通りの表の近い 100 本の番号と値（テスト），`nncos_test_both_*`（4 枠），候補の側の番号と値（`both` の 4 枠．種 0 の単位だけ），微調整した分類器とソースの分類器の確率，K 本の位置と組，K 本で測った信頼度 `D_*`，`pcf_*`，`y_test`・`cl_test`・`sid_test`・`speaker_test`・`y_pool`・`cl_pool`・`sid_pool`・`srcs`・`cfgs` | 120 単位で鍵の欠け 0．浮動小数の配列の非有限 0．テストの長さ・候補の長さ・候補の K 本以外の長さの食い違い 0．近い発話の番号が表の範囲の外 0 |
| `units/*.csv` | 版の行（150 版 × K 2 = 300 行），`_nn`（10 枠 × 59 行 = 590 行），`_single`（110 行），`_check`（2 行） | 全単位で同じ行数．欠けた版 0．`uar_*` の非有限 0，0〜100 の外 0．`_nn` の `hit` の NaN 0．`_single` の `sd_pool`・`agree_src`・`corr_src` の非有限 0 |
| `reweight/*.npz` | 重み 4 × モダリティ 3 × K 2 × 格子 14 × 種 2 の判定，固定の強さの解の判定と確率，3 つの積，K = 0 と 0ob の判定，K = 0 の確率，出発点の W・b，引かれた回数 `hub_*` | 60 組で鍵の欠け 0・余り 0．非有限 0．長さの食い違い 0．テストの `sid` は A の単位（種 0）と同じ．`hub_*` は A の単位の `nn_pool_both_*` の数えと 60 組 × 3 モダリティで完全に一致 |
| `reweight/*.csv` | `_weights` 行 360，`_check` 行 720 | `_weights` の非有限 0 |
| `students/*.npz` | 判定と確率（`pred_`・`prob_` × K 2 × 生徒 4 × 疑似ラベル 6 × 格子 12），固定の点の判定の別名，K 本だけの分類器の判定 | 120 単位で鍵の欠け 0・余り 0．非有限 0．テストの長さ・`sid_test`・`y_test` は A の単位と一致．抜き取り 6 単位で確率の行の和のずれの最大 4.27e-4（float16 の丸め），argmax と保存の判定の食い違いは 942 / 6,439,104 で，すべて float16 で最大の値が同点になった発話（同点でない食い違い 0．`code_v5.md` と同じ性質）．固定の点の別名と `_g{grid_fixed}` の判定の不一致 0 |
| `students/*.csv` | `_teacher` 行 1,440，`_check` 行 960，格子の行 69,120 | `uar`・`uar_pool`・`mimic`・`maxp`・`uar_none`・`uar_pool_none` の非有限 0 |
| `srcdb/*.npz` | 設計 5 節の列 | 24 ファイルで列の長さの食い違い 0，非有限 0，`sizes` の和と本数の一致 |

保存の大きさ：`units/` 2.3 GB，`students/` 871 MB，`reweight/` 101 MB，`srcdb/` 17 MB，`summary/` 1.4 MB．設計 6 節の見込み（A 約 2.9 GB，C 約 0.9 GB，B 約 0.1 GB）の中にある．

## 3．`summary/` の行数と欠け

期待する行数は台本の定数（`BASE` 145，`REFS` 5，`SELDEF` 9，`TIED` 2，`contrasts()` の層の数の和 420，`CRITERIA` 21 組）と設計の格子から作った．

| ファイル | 期待 | 実際 |
|---|---|---|
| `sel.csv` | 4 × 2 × 11 = 88 | 88 |
| `levels.csv` | 4 × 2 × (145 + 5 + 11) = 1,288 | 1,288．欠けた (c, K, 版) 0，余り 0，重複 0，`n_fr` はすべて 10 |
| `effects.csv` | 4 × 2 × 420 = 3,360 | 3,360．重複 0 |
| `criteria.csv` | 規則ごとに層の数 × K の和 = 150 | 150．`main` 21 行（21 規則 × K = 5 × adj），`proceed` 3 行（R2 の 3 つ） |
| `per_emotion.csv` | 4 × 2 × 9 版 × 4 感情 = 288 | 288．`n_units` はすべて 30 |
| `neighbors.csv` | 4 × 10 枠 × 61 = 2,440 | 2,440 |
| `neighbors_by_source.csv` | (T, c) 12 × 10 枠 × ソース 2 = 240 | 240 |
| `neighbors_rules.csv` | 4 × 10 × 量 9 = 360 | 360．`count` はすべて 15 |
| `neighbors_count.csv` | 10 × 9 = 90 | 90 |
| `single.csv` | 4 × 2 × 58 = 464 | 464 |
| `reweight_sel.csv` | 4 × 2 × 3 × 4 = 96 | 96 |
| `reweight_levels.csv` | 4 × 80 = 320 | 320．`n_fr` はすべて 10 |
| `reweight_effects.csv` | 4 × 66 = 264 | 264．重複 0 |
| `reweight_criteria.csv` | 66 | 66．`main` 9 行（重み 3 × モダリティ 3） |
| `reweight_weights.csv` | 4 × 3 × 3 = 36 | 36 |
| `students_sel.csv` | 4 × 2 × 4 × 6 = 192 | 192 |
| `students_levels.csv` | 4 × 2 × 4 × 13 = 416 | 416．`n_fr` はすべて 10，重複 0 |
| `students_per_emotion.csv` | 416 × 4 = 1,664 | 1,664．`n_units` はすべて 30 |
| `students_effects.csv` | 4 × 2 × 4 × 17 = 544 | 544．重複 0 |
| `students_criteria.csv` | 2 × 4 × 17 = 136 | 136．`main` 12 行（生徒 4 × 疑似ラベル 3） |
| `students_conditions.csv` | 4 × 2 × 4 × 6 = 192 | 192 |

- **区間が NaN の行は 0 である**（`effects.csv` 0 / 3,360，`reweight_effects.csv` 0 / 264，`students_effects.csv` 0 / 544）．試走で出た `Mean of empty slice` の警告と区間の NaN は，分割 5 つの本計算では出なかった．点推定・`se` の NaN も 0，下端 > 上端の行 0，点推定が区間の外の行 0．`n_folds` はすべて 5，`n_fr` はすべて 10．
- 数えの表の `main` の行は，どれも `n_corpora` が 4（欠けたコーパスが無い）．
- 残る NaN は構造によるものだけである：`sel.csv` の 56 個（連動する版 `MN_F6orc_sel`・`MS_eq_tied` の 16 行に `k`・`n_tied`・`n_cands` が無い．`N_eq_sel` の 8 行は λ を持たないので `lam` が NaN）と，引き直しの方法の行の `df`（指摘 2）．
- 行の数の列 `hit_count`（`neighbors.csv`）は 15 と 10 の 2 種類である．10 は `by = source` の行で，ソース s は c 以外かつ T 以外のときだけ表に入るので 2 T × 5 分割 = 10 になる．`single.csv` の `uar_raw_count` の 20（`src_<コーパス>` の行）も同じ理由である．

## 4．検算の量とログ

| 量 | 許し | 本計算 |
|---|---|---|
| `convergence.json`：K 本を全部使った作り直しの積と保存の積の差の最大 `teacher_dev_max` | 1e-4（`dai15.CHECK_TEACHER`） | 7.37e-5．5e-5 を超えたのは 6 行で，すべて c = MOSEI の単位（指摘 3） |
| 同：勾配の最大 | 1e-5 | 1.0e-6 |
| 同：止まり | 0 | 0 |
| 同：`adj_resid_max` | — | 1.0e-8 |
| 同：GPU の最大 | 上限 0.6（約 14.7 GB） | 5,524 MB |
| `reweight_checks.json`：`none_agree_min` | 0.995 | 0.99973（1 未満は 720 行のうち 4 行） |
| 同：`start_dev_max` | 1e-3 | 7.95e-5 |
| 同：`base_dev_max` | 1e-3 | 3.58e-5 |
| 同：勾配の最大・止まり | 1e-5・0 | 1.0e-6・0 |
| 同：GPU の最大 | 同上 | 6,527 MB |
| `students_checks.json`：`prop_agree_min` | 0.995 | 0.99946（1 未満は 960 行のうち 5 行．`AFEW_MOSEI_f0_r1` K25 AVT，`MELD_MOSEI_f1_r1` K5 AVT・K25 AVT，`MELD_MOSEI_f3_r0` K5 A，`MELD_MOSEI_f4_r0` K5 AVT．どれも仮のターゲットが MOSEI で，テスト約 3,700 本のうち 1〜2 本） |
| 同：`teacher_agree_min` | 0.995 | 1.0 |
| 同：勾配の最大・止まり | 1e-5・0 | 1.0e-6・0 |
| 同：GPU の最大 | 同上 | 4,071 MB |

ログ：`logs/dai17_probe_units.log` は 120 行，`_reweight.log` は 60 行，`_students.log` は 120 行で，どの行も「名前：秒．行 …」の形だけだった．「済み（飛ばす）」の行は 0（途中で止まって流し直した跡は無い）．例外，警告（収束しない，NaN，型の変換，`Mean of empty slice`），メモリ不足は 0．`_summary.log` は A・B・C の 3 行だけで，差の行 3,360・264・544，数えの行 150・66・136 は上の表と同じ．所要は A 71 分，B 24 分，C 79 分，集計 7 分．

## 5．抜き取りの作り直し

| 対象 | やり方 | 結果 |
|---|---|---|
| `levels.csv` 7 行（adj・kb・raw・pool・ob の層，`prod`・選ぶ版・連動する版・格子の点の版，K 5・25，4 コーパス） | 単位の npz の判定から，同じ (分割, 種) の 3 つの T の数を平均して UAR を出し，種，分割の順に平均する（台本の関数を使わずに numpy で数え直した） | 差の最大 1.4e-14．どの行も 10 (分割, 種) × 3 T |
| `effects.csv` 6 行（R2 の 3 つ，`MN_eq_sel − MS_eq_tied`，候補の層，近さの比べ） | 点推定は独立に数え直した．区間は共通の関数 `dai1_stats.effect_stats`・`dai12_probe.effect_stats_pool` に同じ乱数の鍵 `row_rng("A", c, K, a, b, 層)` を渡して作り直した | 点推定の差の最大 7.6e-17，区間の端の差の最大 2.2e-16 |
| 判定の列（全行） | `lo`・`hi`・`est` から「上回る／下回る／言えない」を作り直した | 不一致 `effects.csv` 0 / 3,360，`reweight_effects.csv` 0 / 264，`students_effects.csv` 0 / 544 |
| `sel.csv`（88 行） | 単位の csv の `uar_adj` を (T, K, 版) で平均し，T = c の行の中で最大の候補（同点は並びで先）を選び直した．連動する 2 版は名前の規則で作った | 選んだ版の不一致 0 / 88，`n_tied` の不一致 0，`at_min_lam`・`at_max_lam` の不一致 0 |
| `frozen_sel.json` | 上の作り直しと突き合わせた | 不一致 0 / 88，余りの鍵 0 |
| 選ぶのに使う単位に c が入らないか（A） | T = c の単位の仮のターゲットとソースを npz の `srcs` から確かめた | c が仮のターゲットかソースに入る単位 0 / 120 |
| `students_sel.csv`（192 行） | 生徒の csv の格子の行の `uar` を (T, K, 生徒, 疑似ラベル, 格子) で平均し，T = c の中の最大（同点は並びで先）を選び直した | 選んだ点の不一致 0 / 192，`n_tied` の不一致 0．各点の平均に入る単位は 30（T = c の単位だけ）．T と仮のターゲットが同じ単位 0 |
| `students_levels.csv` の `_sel` 4 行 | `students_sel.csv` の `grid` で `pred_K{K}_{生徒}_{疑似ラベル}_g{grid}` を引き，独立に数え直した | 差の最大 1.4e-14 |
| 選んだ点の生徒の確率が引けるか | `students_sel.csv` の各行の `grid` で `prob_..._g{grid}` を全 120 単位で引いた | 鍵の欠け 0 |
| `reweight_sel.csv`（96 行） | 重みの csv の格子の行の `uar` を (T, K, モダリティ, 重み, 格子) で平均し，T = c の中の最大を選び直した | 不一致 0 / 96．各点の平均に入る行は 30（T = c の組 15 × 種 2）．T = c となる組 0 |
| `criteria.csv`・`reweight_criteria.csv`・`students_criteria.csv`（全行．`main` の行を含む） | 差の表の判定から `n_up`・`n_down`・`hit`・`down2`・`n_up_noAFEW`・`n_down_noAFEW`・`n_corpora` を数え直した | 不一致 0 / 150，0 / 66，0 / 136．`main` の行は 21・9・12 で，設計 4 節の R1〜R3・対照・近さ・範囲・比べの 21 組，R4 の 9，R5 の 12 と同じ組 |

## 6．生徒の設定の選択を通した依存の表（設計 3.9 節）

- 作り方（`code_run_post/dependency_table.py`）：評価するコーパス c の生徒の設定は，T = c の単位（仮のターゲット c′ ≠ c）で選ぶ．その単位の疑似ラベルの版は，c′ について T = c′ の 30 単位（仮のターゲットに c を含む）で選んだ点である．c′ の点を，T = c′ の単位のうち仮のターゲットが c の単位を除いた 20 単位で同じ規則（平均の最大，同点は並びで先）で選び直し，`frozen_sel.json` の点と比べた．対象は `MN_eq_sel`・`MN_F2K_sel`・`MN_F6K_sel`・`MS_eq_sel`・`MS_eq_tied`（`MN_eq_sel` の λ に連動）．組は c 4 × c′ 3 × K 2 × 疑似ラベル 5 = 120．
- 再現の確かめ：同じ台本で 30 単位を使って選び直すと，120 組すべてで `frozen_sel.json` と同じ点になった．使った単位の数はどの組も 20．
- 出力：`DAI/checks/DAI-17/code_run_post/dependency_count.csv`（(c, K, 疑似ラベル) ごとの変わった c′ の数と名前．選んだ点の名前を含まない），`dependency_detail.csv`（元の点と選び直した点の名前を含む．**選んだ点が書かれているので，結果を開いた後に読む**）．

| K | 変わった組 | 評価するコーパス c ごと | 疑似ラベルごと |
|---|---|---|---|
| 5 | 28 / 60 | AFEW 10，IEMOCAP 7，MOSEI 5，MELD 6 | `MN_eq_sel` 7，`MN_F2K_sel` 9，`MN_F6K_sel` 7，`MS_eq_sel` 1，`MS_eq_tied` 4 |
| 25 | 23 / 60 | AFEW 6，IEMOCAP 7，MOSEI 5，MELD 5 | `MN_eq_sel` 6，`MN_F2K_sel` 3，`MN_F6K_sel` 7，`MS_eq_sel` 2，`MS_eq_tied` 5 |

K = 5 で変わった (c, c′, 疑似ラベル)：
- c = AFEW：(IEMOCAP: `MN_eq_sel`，`MN_F2K_sel`，`MN_F6K_sel`，`MS_eq_tied`)，(MOSEI: `MN_eq_sel`，`MN_F2K_sel`，`MS_eq_tied`)，(MELD: `MN_eq_sel`，`MN_F6K_sel`，`MS_eq_sel`)
- c = IEMOCAP：(AFEW: `MN_eq_sel`，`MN_F2K_sel`，`MN_F6K_sel`，`MS_eq_tied`)，(MOSEI: `MN_F2K_sel`)，(MELD: `MN_F2K_sel`，`MN_F6K_sel`)
- c = MOSEI：(AFEW: `MN_eq_sel`，`MN_F2K_sel`，`MN_F6K_sel`)，(IEMOCAP: `MN_F6K_sel`)，(MELD: `MN_eq_sel`)
- c = MELD：(AFEW: `MN_eq_sel`，`MN_F2K_sel`，`MN_F6K_sel`，`MS_eq_tied`)，(IEMOCAP: `MN_F2K_sel`)，(MOSEI: `MN_F2K_sel`)

K = 25 の組は `dependency_table.out` にある．R5 の 3 つの疑似ラベル（`MN_eq_sel`・`MN_F2K_sel`・`MN_F6K_sel`）の 12 の (c, 疑似ラベル) は，K = 5 ですべて 1 つ以上の c′ が変わった．

## 指摘

### 1. [S6・L3] 依存の表で，R5 の 12 の (コーパス, 疑似ラベル) のすべてに変わった c′ がある
- 重さ：記録する（計算の誤りではない．設計 3.9 節が事前に決めた書き方を当てる）
- 該当箇所：設計 3.9 節「変わった組のあるコーパスで R5 が「上回る」になったら結果の文にそのことを書く」，`dai17_probe.py:705`（単位の疑似ラベルは `choice[(c′, K, t_)]`），1108–1120（c の格子の点は T = c の単位で選ぶ）
- 問題：K = 5 で c′ の点の 28 / 60 が，c を仮のターゲットにした 10 単位を除くと変わる．変わった組は 4 コーパスすべてにあり，R5 の 3 つの疑似ラベルのそれぞれで全コーパスに 1 つ以上ある．したがって設計の規則では，R5 がどのコーパスで「上回る」になっても，結果の文にこの依存を書くことになる．また，依存を除いた R5（選び直した疑似ラベルで T = c の単位の生徒を学び直し，設定を選び直した差）は保存した出力からは出せない．生徒の学び直しが要る．
- 計算の後で直せるか：書き方は直せる（文書だけ）．依存を除いた R5 の値は直せない（C の段の一部の計算のやり直しが要る．C の段は 120 単位 × 疑似ラベル 6 で 79 分かかった．変わった疑似ラベルだけを解き直せば，それより短い）．
- 求める対応：結果の文で R5 を書くときに，依存の表の数（K = 5 で 28 / 60．R5 の 12 の (コーパス, 疑似ラベル) のすべてに変わった c′ がある）を添える．R5 が当たって本番を設計するときは，本番で疑似ラベルの版を計算の前に固定するので，この経路は本番では無くなることを書く．依存を除いた R5 を下調べの中で求めるかは設計者が決める（求めるなら上の費用で計算の追加になる）．

### 2. [設計 4 節] MOSEI の差の行には自由度が無い
- 重さ：記録する
- 該当箇所：設計 4 節「R3・R4・R5 には検出できる差の目安が無いので，差・区間に加えて標準誤差と自由度（`effects.csv`・`reweight_effects.csv` の `se`・`df`）を並べる」
- 問題：MOSEI の行は塊の重みの引き直し（`boot_cl+seed`，候補の層は `boot_cl_shared+seed`）で区間を作るので，`df` が NaN である（`effects.csv` 840 行，`reweight_effects.csv` 66 行，`students_effects.csv` 136 行．どれも MOSEI の行の全部で，ほかのコーパスの行には `df` がある）．`se` は全行にある．
- 計算の後で直せるか：直せる（書き方だけ）
- 求める対応：結果の表で，MOSEI の自由度の欄に「引き直し（自由度なし）」と書く．

### 3. [設計 6 節] 検算の量の最大が，設計 6 節に書いた試走の値より大きい
- 重さ：記録する（許しの中である）
- 該当箇所：設計 6 節「検算の量はどれも許しの中（作り直しの差 ≤ 3.6e-5，…，生徒の提案との一致 1.0000，…）」，GPU の最大 7.7 GB
- 問題：本計算では，作り直しの差の最大が 7.37e-5（許し 1e-4．5e-5 を超えた 6 行はすべて c = MOSEI），生徒の提案との一致の最小が 0.99946（許し 0.995．1 未満の 5 行はすべて仮のターゲットが MOSEI でテストの 1〜2 本），B の `none_agree` の最小が 0.99973 だった．GPU の最大は 6,527 MB で，6 節の 7.7 GB より小さい．6 節の値は試走の 6 単位（c が AFEW・IEMOCAP）の値なので，MOSEI を含む本計算で大きくなるのは矛盾しない．結果の文が 6 節の値を写すと，本計算の値と違う．
- 計算の後で直せるか：直せる（文書だけ）
- 求める対応：結果の文の検算の量は，この報告の 4 節の値（`convergence.json`・`reweight_checks.json`・`students_checks.json`）を書く．

## 計算の後に出すと設計に書かれた量の材料

設計が計算の後に出すとした量の材料が保存されていることを確かめた（量そのものは作っていない）．
- 特徴だけ・答えだけの上位 100 本と `both` の上位 100 本の重なり（3.1 節）：`nn_test_both_{A,V,T}`・`nn_test_feat_*`・`nn_test_logit_*` が 120 単位すべてにある．
- 同じ俳優の対（7 節）：`srcdb/*.npz` の `sid` と単位の `sid_test` がそろっている．
- ソースのコーパスごとの感情ごとの当たる割合の平均（3.7 節）：`nn_test_*` と `srcdb` の `corp`・`y` と単位の `y_test` から作れる．
- 選び直した点の 3 つの条件の量（3.9 節）：生徒の csv の格子の行の `uar`・`uar_pool`・`mimic` と `_check` 行の `uar_none`・`uar_pool_none`，`students_sel.csv` の `grid` がそろっている．

## 目録に無い種類の不備

無し．

## 当てた項目

計算の後の点検の 5 項目（格子のそろい・重複，NaN・Inf・あり得ない値，関門 1 で足した出力（生徒の確率の全点，候補の側の近い発話，`nncos`，W・b，`hub`）の書き出し，コミットと sha256，ログ）．`code_v4.md` の「区間の NaN は本計算で起きないことを計算の後の点検で確かめる」と `code_v5.md` の「`this_script_sha256` が検査した版と同じことを確かめる」も確かめた．S6・L3（指摘 1）．
