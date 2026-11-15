# 計算の後の点検：実装と出力（計算の後の解析 dai1_emotion_margin，2026-11-16，台本はコミット e3f225e の版，本計算の記録はコミット 1ad7f28）

## 判定
判定：通す．本計算の 6 つの出力に欠け・重複・あり得ない値は無く，台本の sha256 は e3f225e の台本と同じで，試走（`code_run_v2/`）と全欄で差 0，要約の "5+25" の行の hurt の点は teacher_margin.csv の 36 点と同じ集合である．

対象：`results/dai1/stats/emotion_margin.csv`，`emotion_margin_check.csv`，`emotion_margin_summary.csv`，`emotion_margin_units.csv.gz`，`emotion_margin_confusion.csv.gz`，`emotion_margin_provenance.json`（いずれも 2026-11-16 03:39 に書かれた），ログ `logs/dai1_emotion_margin.log`．
点検の台本は `checks/dai1_emotion_margin/code_post_run/post_check.py` と `post_check2.py`，出力は同じ場所の `post_check.log`，`post_check2.log`，`post_check2_teachers.log`．
点検では行の数・集合が同じかの真偽・差の最大・上限を超えた行の数だけを見た．margin_e・gain_e・相関・数えの値は画面に出していない（hurt_points は 36 と等しいかの真偽と，集合の大きさだけを出した）．

## 1．台本と記録の版

| 確かめたこと | 結果 |
|---|---|
| provenance の script_sha256 と，`git show e3f225e:DAI/tools/dai1_emotion_margin.py` の sha256 | 両方とも fe1a383e…7e72 で同じ．作業木の台本も同じ値 |
| provenance の git.commit | 1ad7f28（本計算を起動した時点の HEAD）．dirty = false，script_status = clean．出力の表の commit の欄も全行 1ad7f28，dirty は全行 false |
| e3f225e と 1ad7f28 の差分 | `DAI/exp`・`DAI/tools`・`DAI/measure` に差分は無い（差分は論文の原稿と checks の報告だけ）．台本が読み込む `dai1`・`dai1_desc`・`dai1_stats`・`dai3`・`dai_data`・`dai_solver` も関門の版と同じ |
| provenance の入力 6 本の sha256 | 現在の teacher_types.csv，families_flagged.csv，outside_families.csv，recall.csv，teacher_margin.csv，frozen.json の sha256 と全部同じ |
| 単位のコミット | 4 コーパスとも ba98974（試走の provenance と同じ） |
| 計算の装置 | device = cpu．GPU は使っていない |

## 2．欠けと重複

| 表 | 行 | 鍵の重複 | 数の欄の NaN | Inf |
|---|---|---|---|---|
| emotion_margin.csv（点 × 感情．鍵 T，生徒，K，先生，感情） | 4,368 | 0 | 0 | 0 |
| emotion_margin_units.csv.gz（鍵に分割 f と種 r を足す） | 218,400 | 0 | 0 | 0 |
| emotion_margin_confusion.csv.gz（鍵 T，生徒，K，先生，who，true，pred） | 36,480 | 0 | 0 | 0 |
| emotion_margin_check.csv | 3,148 | 0 | 24（後述，作りどおり） | 0 |
| emotion_margin_summary.csv（鍵 T，K，生徒） | 77 | 0 | 77（後述，作りどおり） | 0 |

- 点の数は 1,092（4 コーパス × 273）で，全点が 4 感情の行を持つ．T は AFEW・IEMOCAP・MELD・MOSEI，生徒は A・V・T・AVT，K は 5・10・25，感情は 0〜3
- (T, 生徒, K) の組は 48．先生の数は A・V・T の生徒で全組 22，AVT の生徒で全組 25
- 先生の集合：48 組の全部で，teacher_types.csv（B = none，K = 5・10・25）の先生（AVT の生徒の fused_AVT を除く）に主の先生 main_q を足した集合と同じ．先生の種類は single，pair，fused，poe，noret，ret，start，src，main_q
- 単位の数：コーパスごとに 50 単位（分割 × 種），全点 × 感情が 50 単位を持つ（n_units も全行 50）．単位の表の (点, 感情) の集合は点の表と同じ
- 取り違えの表：who は teacher_pool 17,472 行，retrained_test 17,472 行，self_pool 768 行，none_test 768 行で，(点, who) ごとに 16 行（4 × 4）がそろう
- 除外の印：same_as_poe_none が 12 点（AFEW の K = 5・10・25 の noret_sel × 4 生徒）だけに付く．K = 5・25 の印の無い先生の集合（720 点）は teacher_margin.csv の印の無い先生の集合（720 点）と同じ
- check の表の NaN 24 個は，予測の照合の 12 行（pred_none_checked，pred_p3:tt_checked，pred_p3:joint_checked の各 4 行）の K と n の欄で，これらの行は K を持たない作りである
- 要約の NaN 77 個は hurt_neg_margin_gain_e_mean（40 個）と hurt_pos_margin_gain_e_mean（37 個）だけで，NaN の行は対応する行の数（hurt_neg_margin_rows，hurt_pos_margin_rows）が 0 の行と完全に一致する（code_v2 の「行が無いときだけ NaN になる作り」のとおり）
- 要約の行は各コーパス 15 行と T = all の 17 行，K は 5，10，25，"5+25"，all

## 3．あり得ない値

- t_rec・s_rec が 0〜100 の外：0 行．t_trueprob・s_trueprob が 0〜1 の外：0 行．count_mean が負：0 行
- gain_e が区間 [gain_e_lo, gain_e_hi] の外：0 行

## 4．照合の表（emotion_margin_check.csv）の合否

| 照合 | 行 | 上限 | 上限を超えた行 |
|---|---|---|---|
| teacher_uar_pool_maxdiff（照合 1） | 1,092 | 0.05 点 | 0 |
| gain_mean_vs_uar_effect（照合 3） | 1,092 | 1e-9 | 0 |
| margin_mean_vs_teacher_margin（照合 4） | 728 | 1e-6 | 0 |
| family1_recall_effect（照合 5） | 48 | 1e-9 | 0 |
| family1_recall_lohi（照合 5） | 48 | 1e-9（MOSEI は照合の外） | 12（MOSEI の 12 組だけ．台本の説明 `:17` と code_v2 のとおり，MOSEI の区間は乱数の列が recall.csv と違う．AFEW・IEMOCAP・MELD は 0） |
| n_teachers（照合 6） | 48 | — | 値は 22 か 25 だけ |
| none_test_equal_units（照合 2） | 48 | — | 値は全組 50 |
| teacher_set_equals_teacher_margin（照合 4） | 32 | — | K = 5・25 の 32 組 |
| pred_none／pred_p3:tt／pred_p3:joint_checked | 各 4 | — | 600，13,050，600 で試走と同じ |

台本は照合が合わないと止まる作りで（`tools/dai1_emotion_margin.py:268-279` ほか），本計算は止まらずに最後まで書いた．

## 5．試走（`checks/dai1_emotion_margin/code_run_v2/` の afew，mosei，iem_meld をつないだもの）との一致

| 表 | 本計算の行 | 試走の行 | 片方だけの行 | 値の差 |
|---|---|---|---|---|
| emotion_margin.csv | 4,368 | 4,368 | 0 | commit・dirty を除く全欄で差 0（数の欄の差の最大 0，文字の欄の食い違い 0） |
| emotion_margin_units.csv.gz | 218,400 | 218,400 | 0 | 全欄で差 0 |
| emotion_margin_confusion.csv.gz | 36,480 | 36,480 | 0 | 全欄で差 0 |
| emotion_margin_check.csv | 3,148 | 3,148 | 0 | 全欄で差 0 |
| emotion_margin_summary.csv | 77 | 77（試走の点の表をつなぎ，台本の `summarize` で作り直した） | 0 | 全欄で差 1e-12 以下，NaN の位置も同じ |

MOSEI の区間（gain_e_lo・gain_e_hi，gain_uar_lo・gain_uar_hi）も試走と差 0 で，乱数の列は走らせ方（コーパスを 1 回で流すか分けて流すか）に依らない．

## 6．要約の "5+25" の行と 36 点

- 要約の T = all，K = "5+25"，生徒 = all の行は 1 行で，hurt_points は 36 と等しい
- 本計算の点の表で「除外の印が無い，K = 5・25，margin_uar > 0，gain_uar_hi < 0」の点は 36 点で，teacher_margin.csv の「除外の印が無い，margin > 0，gain_hi < 0」の 36 点と (T, 生徒, K, 先生) の集合として同じ

## 7．ログ

`logs/dai1_emotion_margin.log`（759 バイト）は 4 コーパスの「候補の上の判定を作り直した」「生徒 A・V・T・AVT 済み」「1092 行」と，最後の「書いた：…（4368 行）ほか 5 つ」だけを持つ．例外，収束しない警告，NaN の警告，メモリ不足からのやり直しの行は無い．ログに標準エラーが入っているかは，起動の命令を見ていないので確かめていない（読んだだけ）．

## 8．関門 1 の B で足すと決めた出力

code_v2 の B の表で (a) にした出力（"5+25" の行，取り違えの表の 4 つの who，teacher_trueprob_pool・self_trueprob_pool，単位の表，(T, K, 生徒) の行）は全部書き出されている．code_v2 の記録 1（単位の表にテストの上の感情ごとの差が無い）は台本を変えていないので今もそのままで，要るときは `results/dai1/eval/units/` の予測から別の台本で出す．

## 指摘
指摘なし．当てた項目：欠けと重複（点 × 感情，単位，先生の集合，(T, 生徒, K) の組），NaN・Inf・範囲の外の値，照合の表の合否，台本の sha256 とコミットの差分，入力の sha256，試走との一致，"5+25" と 36 点の集合，ログ．

## 結果を書くときの注意（code_v2 からの引き継ぎ）
- 出力の commit の欄は 1ad7f28 で，関門で検査したコミット e3f225e とは違う．台本と読み込む部品は e3f225e と同じ（1 節）なので，結果の文で版を書くときは「台本 e3f225e（sha256 fe1a383e…），起動時の HEAD 1ad7f28」と書く
- 照合 5 の lo・hi は MOSEI で recall.csv と合わない作りである（MOSEI の区間は乱数の列が違う）
- 取り違えの表の count_mean は 1 単位あたりの平均で，`exp/dai1_desc.py` の confusion の表と尺度が違う（code_v2 の記録 2）
