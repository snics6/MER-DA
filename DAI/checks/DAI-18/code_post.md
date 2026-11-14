# 計算の後の点検：実装と出力（DAI-18，2026-11-14，検査したコミット 02f356e，計算のコミット e02689c・d952b6e）

点検の対象は `results/dai18_probe/` の全部と `logs/dai18_*.log`・`logs/dai18_probe.done` である．確かめの台本と出力は `DAI/checks/DAI-18/code_run_post/` にある．作り直しはすべて CPU（`CUDA_VISIBLE_DEVICES=""`）で流した．`results/dai18_probe/`・`DAI/exp/`・`DAI/frozen/`・`DAI/run/` のファイルは変えていない（点検の後も units と students は 600 個ずつ，units の最後の更新は 15:40 のまま）．UAR の値，差，区間，当たり・外れ，選ばれた点と λ の値はこの報告に書かない．

## 判定

通す．設計の格子の 300 単位が base・units・students の 3 段ですべて csv と npz の対でそろい，欠け・重複・壊れた npz・非有限の値・解けていない最適化が 0 で，全段が検査を通した台本（sha256 290bf037…）で流れ，`MS_eq_sel` と `MN_eq_sel` の選ぶ単位は 18 組すべてで一致し，集計の 13 ファイルは同じ入力から CPU で作り直すとバイト単位で同じになった．

## 1．段ごとの出力の数（`check_files.py`，ログ `log_check_files.txt`）

単位の名前は `all_units()` と同じ規則（T 6 × c 5 × 分割 5 × 種 2 ＝ 300）で作り，名前ごとに照らした．

| 段 | csv | npz | 欠け | 余分 | 大きさ 0 | 壊れた npz（zip の CRC と全配列の読み出し） | csv の行（合計） | 行の重複（全列／鍵） | 名前と行の (T, c, 分割, 種) の食い違い |
|---|---|---|---|---|---|---|---|---|---|
| source | — | 4（CREMA-D A・V・AV，RAVDESS A） | 0 | 0 | 0 | — | `summary.csv` 1 本 | — | — |
| base | 300 | 300 | 0 | 0 | 0 | 0（配列 1,600） | 44,000 | 0／0 | 0 |
| units | 300 | 300 | 0 | 0 | 0 | 0（配列 388,710） | 122,050 | 0／0 | 0 |
| students | 300 | 300 | 0 | 0 | 0 | 0（配列 256,800） | 259,350 | 0／0 | 0 |
| summary | 11 csv＋2 json | — | 0 | — | 0 | — | 下の 5 節 | 0 | — |

- 鍵は base が（kind, K, grid），units が（K, version, modality），students が（K, student, teacher, grid）に（T, c, fold, seed）を足したもの．
- 浮動小数の配列に NaN・Inf を含む npz は 3 段とも 0．units の確率の配列 13,860 個（`p_*` で最後の軸が 4）は，すべての行で和と 1 との差が 0.01 以下だった．
- csv の `uar*` の列で 0〜100 の外の値は 0．格子の行で `uar` が NaN の行は 0．
- 同じ単位の `y_test`・`cl_test` は 3 段のあいだで 300 単位すべてバイト単位で同じ（同じテストの発話と話者の並び）．
- `tether.json`：T 6 × 18 鍵 ＝ 108．`n_units` 300 ＝ `n_units_expected` 300，`n_rows` 42,000 は base の格子の行（grid ≥ 0）の 42,000 と同じ．`base_commits` は e02689c の 1 つ．18 組すべてで AV の強さは AVT の強さと同じ値（設計 2 節の規則どおり）．AV を元の規則で選んだ値は `AV|K|own` の鍵に残っている．
- units の 300 単位すべてに，種類ごとのソースの一覧（`srcs_*`．T と c を含まない），`sid_test`，`nn_test_*` が保存されている（`log_check_srcs.txt`．設計 8 節の計算の後の確認に要るもの）．
- ログの完了行は base・units・students とも 300．「済み（飛ばす）」の行は 0 で，途中で止まって再開した跡は無い．

## 2．台本の sha256 とコミット（provenance と git）

`provenance_log.jsonl` は 6 行（source，base，tether，units，students，summary）．

| 段 | 時刻 | commit | dirty | `this_script_sha256` |
|---|---|---|---|---|
| source | 13:00:44 | e02689c | False | 290bf037… |
| base | 13:02:01 | e02689c | False | 290bf037… |
| tether | 13:28:10 | e02689c | False | 290bf037… |
| units | 13:28:15 | e02689c | False | 290bf037… |
| students | 15:40:13 | e02689c | False | 290bf037… |
| summary | 20:24:01 | d952b6e | False | 290bf037… |

- `provenance.json` は summary の段の記録（d952b6e，dirty False，290bf037…）である．
- 6 段で `scripts` の 10 本の sha256 は同じで，今の作業木の 10 本とも同じ．`design_sha256` は 6 段とも d021385…で，e493c47 以降の設計の文書（第 3 版の記録を足した版）と同じ．
- `inputs_sha256` の 82 鍵は，段が進むにつれて前の段の出力（`out/source/*`，`out/splits_new.csv`，`out/tether.json`，`out/frozen_sel.json`）が加わるだけで，2 つ以上の段に出る鍵で値が変わったものは 0．実験 1 のソースの分類器・分割・発話一覧・特徴の 74 鍵は今のファイルの sha256 と全部同じ．`out/` の 8 鍵も今のファイルと同じ．
- 3 段の csv の `commit` 列は e02689c の 1 つ，`dirty` 列は False の 1 つ．`frozen_sel.json` の `commit` も e02689c．
- `git diff --stat 02f356e <commit> -- DAI/exp measure features DAI/splits data/metadata DAI/run DAI/frozen`：
  - e02689c まで：空
  - d952b6e まで：`DAI/run/dai19_arm.sh`・`DAI/run/dai19_post.sh` の追加だけ（23e1926，19:44）
  - bfbd939 まで：上の 2 本に加えて `DAI/exp/dai20_eval.py`・`DAI/run/dai20_eval.sh`・`DAI/frozen/dai20_frozen.json` の追加
- 実験 18 に影響しないことの確かめ：
  - `dai19_*.sh` は `DAI/run/` にあり，`dai18_probe.sh` から呼ばれず，`dai1.git_commit` の dirty の判定の対象（`DAI/exp`・`DAI/splits`・`measure`・`features`・`data/metadata`）にも入らない．`dai18_probe.sh` の最後の更新は 12:09:39 で，計算の途中に書き換えられていない．
  - `dai20_eval.py`・`dai20_eval.sh` の作成時刻（birth）は 20:27:15，`dai20_frozen.json` は 20:27:28 で，どれも `.done` の書き込み（20:26:28）より後である．summary の段（20:24:01〜20:26:18）が dirty False と記録したことも，その時点で `DAI/exp` に未追跡のファイルが無かったことと合う．
  - `results/dai18_probe/` の各ファイルの更新時刻は，その段の時間帯（source 13:01〜13:01，base 13:02〜13:28，units 13:29〜15:40，students 15:40〜20:23，summary 20:24〜20:26）に収まり，20:26:29 より後に更新されたファイルは 0．
- 依頼の文の `.done` の時刻 20:26:53 は，ファイルの中身と更新時刻（どちらも 20:26:28）と違う．数値には関わらない．

## 3．計算の前の報告で「計算の後に確かめる」とした項目

### 3a．`MS_eq_sel` と `MN_eq_sel` の選ぶ単位（`code_v3.md` 113 行，`probe_v3.md`．`check_summary.py`，`log_check_summary.txt`）

- `sel.csv` の `MS_eq_sel` と `MN_eq_sel` の組は 18 ずつで，（c, K）の鍵の集合は同じ．
- `n_units` は 18 組すべてで一致した．c が AFEW・IEMOCAP・MOSEI・MELD・CREMA-D の 15 組は 40 と 40，c＝RAVDESS の 3 組は 50 と 50（`code_v3.md` の見込みどおり）．
- `pseudo_targets` も 18 組すべてで一致した．15 組は RAVDESS を含まない 4 コーパス，c＝RAVDESS の 3 組は 5 コーパス．
- 参考：`MN_F2K_sel`・`MN_F6K_sel`・`N_eq_sel` の 18 組と `M_F6_mixsel` の 15 組も，`MN_eq_sel` と同じ `n_units` と `pseudo_targets` だった．

### 3b．`frozen_sel.json` と選び直した点（`review_ml_v2.md` m4，`code_v2.md` 記録 3）

- `frozen_sel.json`（students の段が 15:40:16 に書いた 156 鍵）と `sel.csv` の 156 行の `chosen` は，鍵の集合も値もすべて同じ．
- units の 300 個の csv から `select_points` を CPU で流し直すと，156 鍵すべてが `frozen_sel.json` と同じだった．選び直した表を csv に書くと `summary/sel.csv` とバイト単位で同じだった（`DataFrame.equals` が偽になるのは，読み込んだ表と作った表で列の型が違うためで，csv にすると同じになる）．
- students の段の中でも，台本は選び直した点と `frozen_sel.json` を照らして止める処理（`dai18_probe.py:978-979`）を通っている．
- `students_sel.csv` の 417 行の格子の点を，生徒の 300 個の csv から同じ規則（T＝c の単位の平均が最大，役 × 疑似ラベルごと）で選び直すと，417 行すべてで同じ点になった．

### 3c．生徒の配列の作り直し（一致の基準 0.995．`code_v3.md` 99 行．`rerun_students.py`，ログ `log_rerun_*.txt`）

保存した units の npz，`frozen_sel.json`，`tether.json` を入力に，`run_students` を CPU で 4 単位について流し直し，本番の students の npz・csv と比べた．単位は c が RAVDESS・AFEW・CREMA-D・IEMOCAP のものを 1 つずつ選んだ（声だけ，3 モダリティ，2 モダリティの単位を含む）．

| 単位 | CPU の時間 | 配列（新／旧，片方だけ） | 完全一致 | 0.995 未満 | 最小の一致 | csv の行（新／旧） | 1e-9 を超えて違った csv の値 |
|---|---|---|---|---|---|---|---|
| IEMOCAP_RAVDESS_f3_r0 | 17 秒 | 224／224，0 | 224 | 0 | 1.00000 | 237／237 | 0 |
| MOSEI_AFEW_f1_r0 | 94 秒 | 1,034／1,034，0 | 1,034 | 0 | 1.00000 | 1,041／1,041 | `uar_pool`・`mimic` が 1 行 |
| AFEW_CREMA-D_f2_r1 | 325 秒 | 776／776，0 | 776 | 0 | 1.00000 | 786／786 | `uar_pool`・`mimic` が 2 行 |
| MELD_IEMOCAP_f4_r1 | 217 秒 | 1,034／1,034，0 | 1,033 | 0 | 0.99808 | 1,041／1,041 | `uar_pool`・`mimic` が 1 行 |

- 合わせて 3,068 個の配列のうち 3,067 個がビット単位で一致し，残る 1 個も一致 0.99808 で基準 0.995 を上回った．テストの `uar`（1e-9 を超えた違い 0），`uar_none`・`uar_pool_none`・`agree_saved`・`maxp` は 4 単位とも同じだった．違ったのは候補の上の `uar_pool` と `mimic` の計 4 行だけで，`mimic` の差は最大 1.9×10⁻³（候補の発話の数本ぶん）である．GPU と CPU の計算の違いで止める基準の近くの判定が数本ずれる現象で，`code_v2.md` 記録 3・`code_v3.md` 記録 1 と同じ種類である．
- 作り直しの `agree_saved`（疑似ラベルの作り直しと units の保存の一致）は 4 単位とも 1.0，勾配の最大 1.0×10⁻⁶，止まり 0．

### 3d．そのほかの計算の後の項目

- **区間の標準誤差の NaN（`code.md` 74 行）**：`effects.csv` 846 行と `students_effects.csv` 537 行で，`est`・`lo`・`hi`・`se` の NaN は 0．`df` の NaN は `effects.csv` で 306 行，`students_effects.csv` で 189 行あり，すべて塊の重みの引き直しの行（`boot_cl+seed`・`boot_cl_shared+seed`）である．この方法は自由度を持たないので NaN が正しい．jackknife の行の `df` に NaN は無い．
- **土台の作り直しと実験 1 の一致（設計 5b 節 (d)，`probe.md` の記録．`check_none_vs_dai1.py`，`log_check_none_vs_dai1.txt`）**：T と c がどちらも 4 コーパスの 120 単位で，テストの `sample_id` の並びは 120 単位すべて実験 1 と同じ．（単位，K 5・10・25，種類 T・AVT）の 720 組すべてで，実験 18 が選んだ引き戻しの強さは実験 1 の段 1 の選択と同じ点だった．その点での `none_T`・`none_AVT` のテストの判定は 720 組すべてで実験 1 の段 1 の保存と完全に一致した．

## 4．ログ（警告・エラー・止まり）と，ほかの実験の置き場

- `logs/dai18_*.log` の 8 本に，`warn`・`error`・`traceback`・`stalled`・`nan`・`inf`・「超え」・「止」の文字は 0 行（大文字と小文字を区別しない）．`logs/dai18_probe.log` には段の始まりの行，単位ごとの完了の行，source・tether・summary の段の行，`EXIT=0` のほかの行が無い．`logs/dai18_probe.failed` は無い．
- 勾配の最大：base 1.0×10⁻⁶，students の `_check` 3,000 行で 10⁻⁵ を超えた行 0（`students_checks.json` の `grad_max` 1.0×10⁻⁶）．止まり（`stalled`）は base・students とも合計 0．units の段は勾配の列を持たず，解けないときと有限でないときに止める処理（`dai18_probe.py:328`，`:346`，`:551`）があり，止まらずに 300 単位を書いた．`convergence.json` の `adj_resid_max` は 1.0×10⁻⁸．
- 疑似ラベルの作り直しの一致 `agree_saved`：`_teacher` の 6,150 行で 1.0 未満の行は 0（`students_checks.json` の `teacher_agree_min` 1.0）．
- GPU のメモリの最大：base 2,301 MB，units 8,117 MB，students 2,417 MB で，上限の 6 割（14.4 GB）に収まった．所要の合計は base 26 分，units 2.2 時間，students 4.7 時間，起動から `.done` まで約 7 時間 26 分で，設計 7 節の見込み（約 8〜9 時間）の内である．
- 実験 1 と 4 コーパスの実験 16・17 の置き場：起動の台本は計算の前後の sha256 の照合（`dai18_probe.sh:12-13`，`:19`）を通った後にだけ `.done` を書く．`.done` があり `.failed` が無いので照合は通った．あわせて，`results/dai1/source`・`results/dai1/pseudo`・`results/dai16_probe`・`results/dai17_probe`・`results/dai16_probe_k10`・`results/dai17_probe_k10` の 2,607 ファイルに 12:59 より後に更新されたものは 0．`DAI/frozen` で 12:59 より後にできたのは `dai20_frozen.json`（20:27:28 作成）だけで，`.done` の後である．

## 5．集計の入力と表の行の数（`log_check_summary.txt`，`log_summary_rerun.txt`）

- 集計の段は units 300 単位と students 300 単位を読んだ（ログと `convergence.json`・`students_checks.json` の `n_units`）．
- 表の行の数は，全 (T, c) の組をそろえた `code_v3.md` の試走（30 単位）の数と同じである．鍵の重複はすべての表で 0．

| ファイル | 行 | `code_v3.md` の試走 |
|---|---|---|
| `sel.csv` | 156（選択 105，結びつけた行 51） | 156 |
| `levels.csv` | 309 | 309 |
| `effects.csv` | 846（ログの「差の行 846」と同じ） | 846 |
| `criteria.csv` | 39 | 39 |
| `per_emotion.csv` | 996 | 996 |
| `neighbors.csv` | 20 | — |
| `students_sel.csv` | 417 | 417 |
| `students_levels.csv` | 477 | 477 |
| `students_effects.csv` | 537（ログの「差の行 537」と同じ） | 537 |
| `students_criteria.csv` | 108 | 108 |
| `students_conditions.csv` | 417（NaN 0） | 417 |

- `sel.csv` の版ごとの行：`MN_eq_sel`・`MN_F2K_sel`・`MN_F6K_sel`・`MS_eq_sel`・`N_eq_sel` と結びつけた `MN_F6orc_sel`・`MS_eq_tied` が 18 ずつ，`M_F6_mixsel` と `M_F6n_mixsel` が 15 ずつ（RAVDESS を除く）．`levels.csv` の `uar_adj`・`uar_pool` の NaN 60 行は，raw の層だけを持つ `none_<種類>` の行の数（4 コーパス × 4 種類 × K 3 ＋ CREMA-D 3 種類 × 3 ＋ RAVDESS 1 種類 × 3 ＝ 60）と同じである．
- `effects.csv` の c ごとの行は 4 コーパスと CREMA-D が 153 ずつ，RAVDESS が 81（DAI-16 の版が無い）．
- **集計の作り直し**：units と students の置き場を指す一時の置き場で，`--stage summary` を CPU で流し直した（2 分 55 秒）．書いた 13 ファイルはすべて `results/dai18_probe/summary/` とバイト単位で同じだった．比べた後，一時の置き場の写しは消した．

## 指摘

指摘なし．当てた項目：格子のそろい（欠け・余分・重複・名前と鍵の食い違い・大きさ 0・壊れた npz），NaN・Inf・確率の和・UAR の範囲，台本と入力の sha256，各段のコミットと dirty，02f356e からの差分と後から足したファイルの時刻，`MS_eq_sel` と `MN_eq_sel` の単位の一致，`frozen_sel.json` と選び直しの一致，生徒の格子の点の選び直し，生徒の配列の作り直し（基準 0.995），実験 1 との土台の一致，集計の作り直し，ログの警告と止まり，ほかの実験の置き場の保護，関門 1 で足した出力（`students_conditions.csv`，`pred_pool_K*_<種類>_none`，`splits_new.csv`，単位ごとのソースの一覧）の書き出し．

### [記録] 結果に書かれたコミットは，検査したコミット 02f356e ではなく e02689c と d952b6e である
- 重さ：記録する
- 該当箇所：`results/dai18_probe/provenance_log.jsonl`（6 行），`provenance.json`
- 問題：source〜students の 5 段は e02689c（検査の報告だけのコミット），summary の段は d952b6e で流れた．02f356e からどちらのコミットまでも，台本と共通の部品と入力のパスの差分は空か `DAI/run/dai19_*.sh` の追加だけで，台本の sha256 は 6 段とも 290bf037… である．数値には影響しない．
- 計算の後で直せるか：直せる（直す必要はない）
- 求める対応：結果の文の冒頭で，provenance の場所と，段によってコミットが 2 つあることと，その間に計算に関わる変更が無いことを書く．

### [記録] 計算の後に `DAI/frozen` にファイルが足されたので，起動の台本の照合を今の時点で再現すると一致しない
- 重さ：記録する
- 該当箇所：`DAI/run/dai18_probe.sh:12`（`snap` は `DAI/frozen` を含む）
- 問題：`dai20_frozen.json` が `.done` の 60 秒後に足されたので，今 `snap` を計算し直すと計算の前の値と違う．計算の間に置き場が守られたことは，`.done` があることと更新時刻で確かめた（4 節）．
- 計算の後で直せるか：直せる
- 求める対応：後から置き場の保護を確かめ直すときは，`DAI/frozen` の 20:27 以降のファイルを除いて比べる．

### [記録] CPU で作り直すと，候補の上の値（`uar_pool`・`mimic`）が数本ぶんずれる行がある
- 重さ：記録する
- 該当箇所：`dai18_probe.py:686-690`（候補の上の判定は npz に保存せず，csv の値だけを残す）
- 問題：4 単位の作り直しで，テストの判定は 3,068 個中 3,067 個がビット単位で同じ（残り 1 個は 0.99808）だったが，csv の `uar_pool`・`mimic` は 4 行で違った（`mimic` の差は最大 1.9×10⁻³）．GPU と CPU の計算の違いによる．
- 計算の後で直せるか：直せる（直す必要はない）
- 求める対応：結果の文で生徒の再現の割合（`mimic`）を引くときに，作り直しの一致の基準を 0.995 とし，小数第 3 位の違いを誤りと読まない．

## 目録に無い種類の不備

なし．
