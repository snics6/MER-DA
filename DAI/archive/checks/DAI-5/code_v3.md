# 関門 1：実装と出力（DAI-5，第 3 版，2026-11-06，コミット 6674496 と 12bfa1d）

判定：通す．第 2 版の指摘 1 と記録 2〜7 は直っており，足した stats の関数（sensitivity，recall_rows，n_clusters_K，fixed_agree，重みの表の cv_uar_・score_mean_）は偽の評価の回で最後まで動き，保存した判定から numpy で作り直した値と差 1e-15 以下で一致し，select の教師の検算の行の数の守りは偽の入力 3 通りで止まった．

検査の経過：依頼は 6674496 の差分（`git diff 01a07b8 6674496`）の確認である．検査の途中で設計者が 12bfa1d（感情認識の査読の第 2 回の小さい直し）をコミットした．12bfa1d の `dai5_stats.py` の差分は `recall_rows` の混同行列を種の数で割る 1 か所だけで，`dai5.py` と `dai5.sh` は変わっていない．12bfa1d の差分も読み，stats の試走は 12bfa1d で流し直した．12bfa1d には，この検査の途中の版の `code_run_v3/try_select_v3.py`・`try_stats_v3.py` も入っている（その後に `try_stats_v3.py` を 1 か所直し，`sens_drop_v3.py`・`recall_seeds_v3.py` を足した．どれも未コミット）．仮のターゲットの成績の値はこの報告に書かない．

## 判定
通す．第 2 版の指摘と記録はすべて直り，新しく足した部分に誤った数値を出す箇所や計算の後に学び直しを求められる出力の不足は見つからなかった．下の記録 1 つは計算の後の点検で確かめればよい．

## 第 2 版の指摘の照合

| 第 2 版の指摘 | 直した箇所 | 確かめた方法と結果 | 状態 |
|---|---|---|---|
| 1. 引き戻しの感度が `srcw − joint` だけ | 設計 8 節の 7 番目（「引き戻しの強さの感度は `srcw − joint`（選んだ β の 14 点）だけで出す．`srcw_b0`・軸を分けた版・`srcw_top1` は固定の引き戻しだけで解く」と理由） | 読んだ．推した対応のとおりで，計算は足していない | 直った |
| 2. 評価の回の画面の問の数が `srcw` だけ | dai5.py:354–356（5 つの版を数え，収束しない問があれば「（収束しない問あり…）」を画面に出す）．設計 4 節に「評価の回は収束しない問があっても止めず，問の数と収束の数を全部の版で数えて画面と行に残す」 | `--test-tether` の試走 5 単位の画面がすべて「問 276，収束 276」（84 + 32 + 32 + 16 + 112）．行の `grad > 1e-5`・`stalled`・`finite = False` は 5 単位 × 276 問で 0．収束しない問の印の分岐は読んだだけ | 直った |
| 3. 生徒ごとの取りこぼしは負になりうる，K = 0 の identical | 設計 2 節に「生徒ごとの行は負になりうる」．dai5_stats.py:557–558（`if K == 0 or b_sel == D5.DEFAULT_BETA: ident.add((s, K))`） | 偽の評価の回で desc の K = 0 の `srcw − srcw_b0` の 4 行すべてで `identical_by_construction` が真．`ident` の数は 8（K = 0 の 4 と，偽の選択で β = 0 にした K = 10 の 4）．`ident` を読むほかの箇所（dai5_stats.py:570，575–576，635）は K ∈ {5, 25} か K = 25 だけなので，K = 0 を足しても族 7 や (b)(d) の数え上げは変わらない | 直った |
| 4. select が `_teacher_check` の欠けを止めない | dai5.py:455–457（行の数が 単位の数 × 3 × (|others(T)| − 1) × |KS_ALL| = 30 × 3 × 2 × 5 = 900 かを見る） | 下の「select の守り」の表．1 単位ぶんの 30 行を消す，1 行を消す，1 行を重ねるの 3 通りがすべて止まり，何も替えない置き場は通った．試走の 1 単位の `_teacher_check` は 30 行（K ごとに 6）で，期待の式と合う．4 つの T のどれでも `others(T)` は 3 つなので，n_c = 2 は全 T で正しい | 直った（下の記録 1） |
| 5. 固定の点の照合が判定でなく数を比べる | dai5_stats.py:128–130（collect の `fixed_agree`），614–615（desc の `fixed_point_pred_agree`） | 偽の評価の回で 16 行すべてに値があり（NaN 0），判定から独立に作り直した一致率との差の最大 1.1e-16．`p3t` の固定の点の判定を 1 単位（A，K = 5）で 3 本だけ替えた偽の入力では，その行だけが 1 未満になった（`counts_agree` は 0.8）．`(tcw, tcb) == tuple(fz1T["tether"][...])` の浮動小数の完全一致は，固定のファイルの値が `dai1.TETHER` の値と同じ書き方なので成り立った（16/16） | 直った |
| 6. `open` の閉じ忘れ 4 か所 | dai5.py:499–500，539–540，dai5_stats.py:235–236，765–766 | 試走（`-W default`）で pseudo と stats のどちらにも ResourceWarning が出なかった | 直った |
| 7. 設計 8 節の範囲の外の追記 | 設計 8 節（SUR 型の学ぶ重みは範囲の外，軸を分けた版は K ∈ {5, 25} だけ，性別の注釈を使わない，MOSEI の話者の代わりの不完全さ） | 読んだ | 直った |

## A．設計と台本の対応表（6674496 と 12bfa1d で足した部分）

| 設計の項目 | 台本の箇所 | 合っているか |
|---|---|---|
| 部分集合の感度（設計 2 節．AFEW の書き起こしの疑わしい発話を除く・MELD の顔が競らない発話・IEMOCAP のマーカーあり／なしに限った `srcw − joint` と `srcw − srcw_b0`，K ∈ {5, 25}．30 本未満か感情が欠ける分割は外す） | dai5_stats.py:432–463（`SUBSETS`，`sensitivity`），751 | 合う．偽の部分集合 3 つ（40 行）で `effect` を判定から独立に作り直して差の最大 8.9e-16．外す規則は，分割 0・1 に 60 本，分割 2 に 10 本，分割 3・4 に 0 本の偽の部分集合で `folds_dropped` が「2,3,4」，`n_units` 2 になり，作り直した期待と一致した．生徒は AFEW が T・AVT，MELD が V・AVT，IEMOCAP が 4 つで，`subsets.csv` の `corpus`・`subset`・`value` の値（AFEW の asr_suspect 35 本，MELD の face_compete の single 6,019 本，IEMOCAP の marker 1,504 本と nomarker 1,439 本）と名前が合う |
| 感情ごとの再現率と混同行列（設計 2 節．`srcw`・`srcw_b0`・`joint`，K ∈ {5, 25}，5 つのテストを合わせ，種の数で割って各テストの発話を 1 回ずつ数える） | dai5_stats.py:465–512（`recall_rows`）．12bfa1d で 506–511 を「種の数 R で割る」に直した | 合う．混同行列 24 行（生徒 4 × K 2 × 行 3）を独立に作り直して差 0，再現率の差の最大 5.6e-17，`n_test_total` はテストの総数 2,943 と一致．同じ判定を種 2 つに写した偽の回で，種 1 つのときと混同行列・再現率・`n_test_total` の差が 0.0，`n_seeds` は 2 |
| 9 個の識別器のテストでの感情ごとの再現率，採点 1 位の Neutral の再現率と中央値，10 ポイント以上下回る印（設計 9 節） | dai5_stats.py:496–502 | 合う．偽の回では識別器が 6 個（ソース 2 × モダリティ 3）なので列は 24．単位の平均の再現率，1 位（同点は平均）と中央値の Neutral の再現率の作り直しとの差の最大 5.6e-17 |
| 採点の妥当性の表に K 本が属する塊の数（設計 3 節 (a)） | dai5_stats.py:390–396，735 | 合う．`n_clusters_K_mean`・`n_clusters_K_min` を分割の表から独立に作り直して食い違い 0（K 1・5・10・25 の 4 行） |
| 重みの表に交差検証の UAR と K 本の採点の平均（設計 2 節の読み方と注） | dai5_stats.py:524，713 | 合う．`cv_uar_<識別器>` は `results/dai1/source/summary.csv` の `cv_uar` と食い違い 0（名前の形 "<ソース>_<モダリティ>" が summary の `corpus`_`kind` と同じ），`score_mean_` は作り直しとの差 0.0，K = 0 は NaN |
| 固定の点の判定の一致率（設計 4 節） | dai5_stats.py:128–130，614–615 | 合う（上の照合の表の 5） |
| 入力の sha256（ML の査読 m7） | dai5.sh:27–30 | 合う．`bash -n` を通った．glob が当たるファイルは pseudo で p1 の npz 120・p3 の csv 120・tether_w.json 4，eval で p1 の npz 200・p3 の npz 200・p3 の csv 200（約 15 GB）．eval の `results/dai5/select/frozen.json` は select の後にできる．無いファイルは sha256sum が「No such file」を書き，次の行の grep が止める．dai5.py の pseudo が読む p1 の csv（dai5.py:325–327 で有無だけを見る）は記録に入らないが，中身を読むのは npz なので結果に関わらない |

## A．小さく流した結果

環境：RTX A5000（流す前の使用は 1.5 GB で，ほかの計算は無かった），`./.venv/bin/python -W default`．出力は scratchpad に置き，`results/` には書いていない．試走の出力は報告を書く前に消した．検査の台本は `DAI/checks/DAI-5/code_run_v3/` に置いた（成績の値を含む出力は置いていない）．

| 実行 | 設定 | 所要 | 結果 |
|---|---|---|---|
| `dai5.py --stage pseudo --targets AFEW --cs IEMOCAP --folds 0 1 2 --seeds 0 --allow-dirty --selftest-rho0 --test-tether`（6674496，GPU の割合 0.2） | 次の行と並べた | 151・134・97 秒／単位 | 3 単位とも「問 276，収束 276」，警告 0 |
| 同 `--folds 3 4`（ρ = 0 の自己検査なし） | 上と並べた | 137・133 秒／単位 | 2 単位とも「問 276，収束 276」，警告 0 |
| `try_select_v3.py`（CPU） | 1 単位の p5.csv を 30 単位に写し，uar を乱数に替えた偽の置き場 | 3 秒 | 下の表 |
| `try_stats_v3.py`（CPU，12bfa1d） | 試走の 5 単位を AFEW_ALL_f0〜4_r0 と名づけた偽の評価の回（第 2 版と同じ作り方）．テストが IEMOCAP の発話なので，分割の表と部分集合の表は偽の根の下に IEMOCAP のものを AFEW と名づけて置いた | 21 秒 | 上の対応表の数．出力の形は下 |
| `sens_drop_v3.py`（CPU） | 30 本未満と感情が欠ける分割を作った偽の部分集合 | 数秒 | `folds_dropped` が期待と一致 |
| `recall_seeds_v3.py`（CPU，12bfa1d） | 同じ判定を種 2 つに写した偽の回 | 数秒 | 種 1 つと 2 つの差 0.0 |

select の守り（6674496）：

| 偽の入力 | 結果 |
|---|---|
| 何も替えない | 通った |
| 1 単位の `_teacher_check` を全部消す | 止まる：「教師の検算の行が 870 で，期待 900 と違う」 |
| 1 単位の `_teacher_check` を 1 行消す | 止まる：899 |
| 1 単位の `_teacher_check` の 1 行を重ねる | 止まる：901 |
| 一致しない行（前からの守り） | 止まる：「教師の検算が一致しない行が 1 ある」 |
| `srcw_tether` の行の勾配を 1e-4 にする | 通った（select は pseudo の 4 版だけを見る．本番の pseudo に `srcw_tether` の行は無い） |

`try_stats_v3.py` の出力の形（12bfa1d）：families7 8 行，desc 568 行，regret 60 行，weights 5 行，scores 4 行，sensitivity 40 行（asr_excluded 2 生徒，marker 4 生徒，nomarker 4 生徒 × K 2 × 相手 2），recall 24 行，confusion 24 行，src_recall 4 行（K 1・5・10・25）．Inf は 0．NaN は設計どおり空になる列だけである（sensitivity の `folds_dropped` が空の 40，weights の `sd_within_fold` 5・`top2_agree` 1・K = 0 の `score_mean_` 6，families7 の `descriptive`・`detectable_first`（偽の detect が空））．`collect` の照合の食い違いは 0/100．

## A．本番の規模の見積もり

`dai5.py` の計算の部分は 6674496 でも変わっていない（画面の 1 行と select の守りと `with` だけ）ので，第 2 版の見積もり（pseudo は並べて 1〜2 時間，eval は 3〜5 時間，GPU のメモリは 1 本で最大 3.64 GB）がそのまま当てはまる．試走の 1 単位の所要は 2 本を並べて 97〜151 秒で，第 2 版の単独の実測（86〜107 秒）と並べた分の取り合いに見合う．stats は 1 つの T に部分集合の collect が 1〜2 回（IEMOCAP は 2 回）増え，偽の回の 5 単位で全体 21 秒だったので，本番の 200 単位では 10〜20 分の見込みである（GPU を使わない）．

## B．予想される指摘と，出力から答えられるか

第 2 版の表から変わった行だけを書く．

| 表・図 | 予想される指摘 | もとにした項目か例 | (a)／(b)／(c) | (b) なら足すもの | 足す費用 |
|---|---|---|---|---|---|
| 重みづけの差と軸を分けた版 | 引き戻しを替えても重みづけの差は残るか | 第 2 版の指摘 1 | (c)．設計 8 節の 7 番目に範囲の外と理由がある | — | — |
| 族 7・記述 | 部分集合（書き起こしの疑わしい発話，顔の競合，マーカー）に限ると結論は変わるか | 感情認識の査読 A1 | (a)．`sensitivity.csv` の `effect`・`lo`・`hi` と `folds_dropped` | — | — |
| 族 7・記述 | MOSEI のチャンネルが分かる発話に限ると | 感情認識の査読 b3 | (a)．設計 8 節のとおり．`p5_<s>_K<K>_pred` と実験 1 の p3 の判定を `sample_test` で引き，`data/metadata` のチャンネルの対応から作る印で絞って，`sensitivity` と同じ計算（`collect` の `mask_fn`）を CPU で流せる | — | — |
| 記述 | 感情ごとの再現率，特定の感情だけで動いた差か | 感情認識の査読 A2 | (a)．`recall.csv`・`confusion.csv`（種の数で割った数），識別器ごとは `src_recall.csv` | — | — |
| 採点の妥当性 | K 本が少数の塊に偏るのでは | 感情認識の査読 a1 | (a)．`scores.csv` の `n_clusters_K_mean`・`n_clusters_K_min` | — | — |
| 固定の点の照合 | 判定は同じか | 第 2 版の指摘 5 | (a)．desc の `fixed_point_pred_agree` | — | — |

## 指摘

### [I11] 1. select の教師の検算の行の数は T ごとの合計で見るので，1 単位の欠けと別の単位の重なりが同時にあると通る
- 重さ：記録する
- 該当箇所：dai5.py:456（`if len(tc) != len(names) * 3 * n_c * len(KS_ALL):`）
- 問題：守りは T の 30 単位の合計 900 行と比べる．1 単位の 1 行が欠け，別の単位で 1 行が重なると合計は 900 のままで通る（読んだだけ．試走では片方だけの 3 通りが止まった）．`run_unit` は単位ごとに決まった数の行を書くので，起きる見込みは低い．
- 計算の後で直せるか：直せる（行は保存される）
- 求める対応：計算の後の点検で，pseudo の各単位の `_teacher_check` が 30 行で，(K, 識別器) の組に重なりが無いことを数える．`dai5.py` を pseudo の前に直すなら，`tc.groupby(["c", "fold", "seed"]).size()` がすべて 30 かを見る形にする．

指摘はこの 1 つだけである．当てた項目：I1，I3，I7，I9，I11，I12，R3，R4，R6，R10，L4．

## 目録に無い種類の不備
- 提案：9.3 節 I に「種を重ねた集計（混同行列など）は，件数の列が発話の数か，種の数を掛けた数かを列の名前か別の列（`n_seeds`）で示す」を足す（12bfa1d で直した `recall_rows` の型の不備．数の列を発話の数と読んで区間を作ると幅が狭く出る）．

## 計算の後の点検（関門 2 の前）への申し送り
- 第 2 版の申し送りはそのまま有効である（pseudo 120 単位 × 164 問，eval 200 単位 × 276 問．重複 0．eval の全 method の `grad > 1e-5`・`stalled`・`finite = False` の数．`gpu_counts.csv` の `same_as_pseudo`）．
- 足す点検：pseudo の各単位の `_teacher_check` が 30 行で (K, 識別器) が重ならない（指摘 1）．stats の `sensitivity.csv`（AFEW 2 生徒 + MELD 2 生徒 + IEMOCAP の 2 つの部分集合 × 4 生徒 = 12 の (部分集合, 生徒) × K 2 × 相手 2 = 48 行），`recall.csv`・`confusion.csv`（T 4 × 生徒 4 × K 2 × 行 3 = 96 行ずつ，`n_seeds` が 10），`src_recall.csv`（T 4 × K 4 = 16 行，識別器の再現率の列 36），desc の `fixed_point_check` の `fixed_point_pred_agree` が 1.0 未満の行の数と最小．
- 結果のコミットが pseudo と eval で同じで，`dai5.py` が 6674496 と同じか（12bfa1d は `dai5.py` を変えていない）．
