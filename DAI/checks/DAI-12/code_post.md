# 計算の後の点検：結果ファイルと記録（DAI-12 の下調べ，2026-11-09 17:2x〜17:4x，単位の段のコミット e9f66c9，集計の段のコミット d0a2d91）

点検した対象：`results/dai12_probe/`（`rows/` の 360 ファイルと集計 15 ファイル），台本 `DAI/exp/dai12_probe.py`（sha256 `bd287f689ba53839bab20aa4476711c307be28e976bc79c20c58253a99eb5a02`），起動の台本 `DAI/run/dai12_probe.sh`（`b0d9880c…`），設計 `DAI/docs/実験12_下調べ_設計.md`（第 6 版，`fac8a28c7ca0a8511c18c88101bedb6540fb462b9faf9c7b70641c5dfc59d61a`），ログ 3 本，台帳 `docs/seen-data.md` の 213 行と 216 行．
点検は CPU だけで行った（GPU は使っていない）．検算の台本は `DAI/checks/DAI-12/code_run_post/post_check1.py`〜`post_check6.py` と `design_revert.py` に置いた．どの台本も個数・鍵・形・型・不一致の数・有限性・所要・記録だけを印字し，UAR・差・区間・判定の文字列は印字していない．結果ファイルは書き換えていない．`results/dai13_probe/` には触れていない．

## 判定

通す．120 単位がそろって重複が無く，設計 3.9 節の保存の鍵がすべての単位にあり，照合の不一致は 1 単位の 1 鍵の 1 本（許容 4 本）だけで，保存と集計に設計の想定の外の有限でない値が無く，両段の台本と共通の部品の sha256 が同じで，途中の値を見た 15:2x の後に設計・台本・規則が変わっていない．記録の不足が 4 件ある（下の「指摘」．どれも結果の文で直せる）．

## 1．単位の集合，行数，重複（`post_check1.py`，`post_check6.py`）

| 確かめたこと | 結果 |
|---|---|
| `rows/*.npz` の単位の名前 | 120 個，重複 0．`results/dai1/pseudo/units/*_p3.npz`（120）と `results/dai11_probe/rows/*.npz`（120）の集合と一致 |
| 名前の形 | 全部 `<T>_<c>_f[0-4]_r[01]` で T ≠ c．T ごとに 30 単位 |
| `rows/` のほかのファイル | `_nl.csv` 120，`_lab.csv` 120．名前の集合は npz と同じ．ほかのファイルは無い |
| npz の `unit` と名前 | 不一致 0 |
| `_nl.csv` | 全単位で 105 行（3 モダリティ × 出所 7 × K 5），列 140（列の並びは 1 通り）．(student, teacher, K) の重複 0．`unit` の列とファイル名の不一致 0 |
| `_lab.csv` | 全単位で 15 行（3 × K 5），列 60．(student, K) の重複 0 |
| `nl_units.csv` | 12,600 行，(unit, student, teacher, K) の重複 0，単位 120 |
| `lab_units.csv` | 1,800 行，重複 0 |
| `cells_nl.csv` | 420 行（c 4 × 3 × 7 × 5）．`n_units` は 420 行とも 30 |
| `cells_lab.csv` | 60 行．`lab_units.csv` から数えたセルごとの単位は 60 セルとも 30 |
| `uar_units.csv` | 88,200 行（120 単位 × テストの判定の鍵 735），(c, T, fold, seed, key) の重複 0．UAR・WAR・WAF・MF1・感情ごとの再現率で 0〜100 の外の値は 0 個 |
| `halves_units.csv` | 120 行，重複 0．npz の `half_counts` は 120 単位とも `pool_half` と `pool_y` から数え直した本数と一致 |
| `checks_units.csv` | 21,600 行（120 × 180 鍵），(unit, key) の重複 0 |
| `diffs.csv` | 13,800 行，全行の `n_units` が 10（(f, r) の単位．設計 4.1 節）．完全に同じ行 0 |
| ログの単位 | `logs/dai12_probe_units.log` の 120 行の単位名は重複 0 で npz の集合と同じ |
| 発話の対応 | 120 単位とも `test_sample_id`・`test_y`・`test_cluster`・`pool_y`・`pool_cluster`・`kpos_K{1,5,10,25}` が DAI-11 の保存と一致（DAI-11 の npz に `pool_sample_id` は無いので候補の sample_id は比べていない．候補の本数は台本 `:336-337` が照合している） |
| 漏れの検算 | 120 単位とも，候補とテストで共有する `sample_id` 0 個，塊 0 個．2 つの半分（主の分け方と DAI-11 の分け方）が共有する塊 0 個 |

## 2．設計 3.9 節の保存の鍵（`post_check2.py`，`post_check3.py`）

鍵の集合は仮のターゲット c ごとに 1 通り（IEMOCAP・MELD 2,504 鍵，AFEW・MOSEI 2,503 鍵）で，違いは部分集合の印の数だけである（ログの「保存 2482／2483」に meta の 21 鍵を足した数と合う）．

| 3.9 節の項目 | 鍵（単位あたりの数） | 型と形 | 結果 |
|---|---|---|---|
| 候補の判定 | `pool_none` 15，`pool_teacher` 105，`pool_ridge` 105，`pool_lince` 105，`pool_lincefix` 105，`pool_lince0` 105，`pool_nl` 105，`pool_nlfix` 105，`pool_lab` 6，`pool_labfix` 6，`pool_read_nl` 105，`pool_read_lince` 105 | int8，(候補,)，値は 0〜3 | そろう |
| 格子の全点の判定 | `grid_pred_{nl,lince,lince0}_{m}` 9 | int8，(1, 5, 8, 35, 候補) | そろう |
| Q3 の格子の全点の判定 | `grid_pred_lab_{nl,lince}_{m}` 6 | int8，(1, 5, 10, 1, 候補) | そろう（gate-code 第 5 回の指摘の鍵） |
| 格子の全点の UAR | `grid_uar_{nl,lince,lince0}_{m}` 9 | float32，(35, 1, 5, 8)，0〜100 の外 0 個 | そろう |
| テストの判定 | `K{K}_{m}_none` 15，`_noneloco` 15，`_ret11_<出所>` 75，`_nl_` 105，`_lince_` 105，`_lince0_` 105，`_nlfix_` 105，`_lincefix_` 105，`_nlhid_` 105，`labtest_{nl,lince}_{m}` 6，`labtest_nlhid_{m}` 3 | int8，(テスト,)，値は 0〜3 | そろう |
| 点数 | 候補 `score_{ridge,nl,lince}_K*` 315，テスト `score_{nl,lince}_test_K*` 各 105 | float16，候補 (候補, 4)，テスト (2, テスト, 4)（2 つの半分） | そろう |
| `pair` の重み | `w_nl_pair_*` 150（K 5 × m 3 × 半分 2 × W1・c1・W2・c・Wl），`w_lince_pair_*` 60（W・c） | float16 | そろう |
| 半分と部分集合と入れ替え | `pool_half`，`pool_half11`（int8，値 0・1），`half_counts`（int64，(2, 4)），`pool_sub_*`（int8，値 0・1．IEMOCAP `marker`・`nomarker`，MELD `dlgfree`・`face_single`，MOSEI `channel`，AFEW `asr_ok`．30 単位ずつ），`pool_shuf_src_{m}` 3（int32，0〜候補−1） | | そろう |
| 候補とテストの行の情報 | `pool_sample_id`・`pool_y`・`pool_cluster`・`test_sample_id`・`test_y`・`test_cluster`・`kpos_K{1,5,10,25}` | 長さは候補とテストの本数に合う．候補とテストの中の sample_id の重複 0 | そろう |
| 照合と収束の記録 | `mismatch`（JSON．180 鍵，うち `frac` を含む鍵 12），`conv`（JSON．12 鍵 = nl・lince・lince0・lab × 3 モダリティ） | | そろう |

形の食い違い 0 件，判定の値の範囲の外 0 件．

## 3．照合の不一致と有限性（`post_check2.py`，`post_check4.py`，`post_check6.py`）

- `frac` を含まない照合の鍵は 20,160 行で，0 でないのは 1 行だけである：`AFEW_MELD_f4_r0` の `V_K25_pool_proj_enc2_vs_dai11` が 1 本（その単位の許容は候補の側 max(1, 0.1%) = 4 本）．許容を超える行 0，有限でない値 0．`frac` を含む 1,440 行（読み出しの格子の端の割合）は全部 0〜1 の中．
- npz の浮動小数の配列（点数，重み，`grid_uar`）に有限でない値 0 個（120 単位すべて）．
- 集計の csv の有限でない値はすべて設計か台本の構造で決まる場所にある．inf はどのファイルにも 0 個．NaN の場所：
  - `nl_units.csv`・`cells_nl.csv` の `pair_comp_kappa`：出所が `pair` でない行だけ（10,800 = 12,600 − 1,800，360 = 420 − 60）．`pair` の行は全部有限（台本 `:627-629`）
  - `cells_nl.csv` の `share_{lince,nl,ridge}`：155 行．NaN の行は `delta_uar` < 0.5（`MIN_DELTA_SHARE`）の行と完全に一致する（台本 `:753-755`）
  - `diffs.csv` の `war_*`・`waf_*`・`mf1_*`：候補の上の行（6,510）とテストの部分集合の行（4,050）．テストの `subset == all` の行（3,240）では全部有限．NaN の行と「候補の行か，テストの部分集合の行」は完全に一致する．`est`・`lo`・`hi`・`se`・`p` に NaN は 0 個
  - `diffs.csv` の部分集合：`dlgfree` は候補の上だけ（345 行．設計 2 節の定義どおり予測に使ったネットの学習の半分に対する印なので，テストには無い）
  - `criteria.csv`：規則ごとに列があるか無いかが決まる（例：EDGE・NEST・OPT・LOSS・Q2a の行に `est_max` が無い，R3 の行に `n_K` が無い，Q3 の行に `verdict` が無い）．規則の中で一部だけ NaN の列は次の 5 種で，全部台本の条件の分岐と一致する：R1 の入れ替えの対照の列は出所 `pair` の 12 行だけにある（設計 4.2 節），R1・R1c・R1shuf の部分集合の列は MOSEI と MELD の行だけにある（設計 4.2 節の部分集合の頑健さ），R2 の `chan_n_K_*` は MOSEI の `nl_vs_ret11` 以外の行だけにある（台本 `:900`．設計 4.2 節の R2 で DAI-11 の線形の版には規則の語を付けない），R2 の `K_up`・`K_down` は該当の K が無い行で空の文字列が NaN として読まれる，EDGE の読み出しの 16 行は `n_K_up` を持たない（台本 `:930`）
  - `q2_cells.csv` の `sub_robust`：AFEW 3 行と IEMOCAP 14 行で NaN（部分集合の頑健さは MOSEI・MELD だけ．Q2 の 30 セルに MELD は無い）．`status` の NaN 0
  - `cells_lab.csv`，`se_summary.csv`，`uar_units.csv`，`lab_units.csv`，`halves_units.csv`，`checks_units.csv`，`units_time_*.csv`：NaN 0
- Q2 の 30 セルの (c, 出所) の内訳は AFEW 1・1・1，IEMOCAP 5・4・5，MOSEI 4・5・4（fuse1・other・enc2fuse の順に 10 ずつ）で，台帳 197 行の DAI-11 の数え上げ（`fuse1` 10，`other` 10，`enc2fuse` 10）と合う．

## 4．記録の整合（`post_check5.py`）

| 記録 | 内容 | 台帳の説明との関係 |
|---|---|---|
| `provenance_log.jsonl` | 2 行だけ．1 行目 `units`，コミット e9f66c9，dirty 偽，07:40:10．2 行目 `summarize`，コミット d0a2d91，dirty 偽，16:38:42．argv はどちらも `--out results/dai12_probe --stage <段>` だけ（`--allow-partial`・`--allow-dirty` なし） | 216 行の説明と合う．1 回目の集計は台本 `:1082-1084` の未コミットの検査で記録を書く前に止まるので，記録の行が残らないことと合う |
| 2 行の違い | `argv`・`commit`・`stage`・`time` の 4 つだけ．台本・`dai11_probe.py`・`dai1.py`・`dai_solver.py`・`dai_data.py`・`dai1_stats.py`・`dai3.py`・凍結の JSON の sha256，特徴 24 本の sha256，定数，ソフトの版は 2 段で同じ | 台本の sha256 は 2 段とも `bd287f68…` |
| `provenance.json` | 2 行目（集計の段）と完全に同じ．コミットは d0a2d91 | 台本 `:1122` は集計の段だけが `provenance.json` を書くので，単位の段のコミットは `provenance.json` に無い（指摘 1） |
| `rows/*.npz` の `commit` | 120 単位とも e9f66c9．`commit11` は 120 単位とも 67e3553（`results/dai11_probe/provenance.json` の `commit` と一致） | 合う |
| 今の入力 | 台本と共通の部品 8 本の今の sha256 と，特徴 24 本の今の sha256 が単位の段の記録と同じ．DAI-11 の台本の sha256，DAI-1 の `pseudo_commit`（896136c）が凍結の JSON と同じ | 合う |
| e9f66c9 と d0a2d91 の差 | `git diff --stat` は 14 ファイルで，DAI-13 の設計・台本・起動の台本・検査の報告，`features/build_audio_layers.py`，`data/README.md` の 1 行（全層の特徴の説明），`docs/seen-data.md` の 3 行，先行研究の報告．DAI-12 の台本・設計・起動の台本・付録の表の台本・共通の部品・`measure/` は変わっていない | 集計の段の計算はコミットの違いの影響を受けない |
| 特徴の記録 | 集計の段の `features_sha256` に DAI-13 の全層の特徴のファイル（`audio_wavlm_layers_*`）は入っていない | 合う |
| ドライバのログ | 3 行．3 行目「集計の段を終えた 2026-11-09 17:25:53（DAI-13 のファイルのコミット d0a2d91 の後に流し直した）」 | 起動の台本の `echo` には括弧の文が無いので，3 行目は起動の台本の外で書かれた．1 回目の集計の失敗はどのログにも残っていない（指摘 2） |

## 5．所要と GPU（`post_check5.py`，ログ）

- 単位の段：07:40:09 → 16:37:11（8 時間 57 分．`units_time_*.csv` の秒の和は 8.95 時間）．設計 6 節の見込み 8〜9 時間の中．
- 単位あたりの秒（c ごとの最小〜最大）：AFEW 119〜135，IEMOCAP 156〜185，MELD 226〜594，MOSEI 516〜523．MELD の最大 594 秒と 2 番目の 548 秒は `MOSEI_MELD_f0_r0`・`f0_r1`（15:45〜16:04 に書かれた単位）で，台帳 215 行の DAI-13 の特徴の取り出し（15:48〜16:00）と GPU を共有した時間に当たる．ほかの MELD の単位は 226〜257 秒．
- 集計の段：16:38:42 → 17:25:53（47 分）．見込み 1〜1.5 時間より短い．
- GPU：単位ごとに `reset_peak_memory_stats`（台本 `:324`）．`max_memory_allocated` の最大は 9.7 GB（MOSEI を仮のターゲットにした単位），`max_memory_reserved` の最大は 10.7 GB．設計 6 節の試走の値（8.5〜9.1 GB，予約 10.7 GB）より割り当ての最大が 0.6 GB 大きいが，`set_per_process_memory_fraction(0.6)`（`dai_solver.py:29`）の上限 14.7 GB の中．
- ログの警告：単位のログに警告・例外・NaN・メモリ不足の語は 0 件．集計のログには pandas の `PerformanceWarning`（DataFrame の断片化）が 3 件あるだけで，値には影響しない．

## 6．途中の値を見た後に変わっていないこと

- 台本：今の作業木，e9f66c9，d0a2d91 の `DAI/exp/dai12_probe.py` の sha256 は全部 `bd287f68…`．起動の台本 `b0d9880c…` と付録の表の台本 `6b96206e…` も 3 つで同じ．
- 設計：今の作業木，e9f66c9，d0a2d91 の第 6 版の sha256 は全部 `fac8a28c…`．
- 規則：e9f66c9 → d0a2d91 の差と，d0a2d91 の上の未コミットの変更（DAI-13 の 3 ファイルと `docs/seen-data.md`）に，`DAI/CLAUDE.md`・`docs/methods.md`・`DAI/frozen/` と DAI-12 の台本・設計・共通の部品は入っていない．
- `results/dai12_probe/rows/` の npz の書き込み時刻は 07:42:49〜16:37:08 で，ログの単位の順と合う．集計のファイルは 16:51 と 17:25 に書かれた．途中の値を見た 15:2x の集計の出力は scratchpad にあり，`results/` には書かれていない（台帳 213 行）．
- gate-code 第 6 回が検査した設計の sha256 は `7825ef06…` で，コミットした版（`fac8a28c…`）と違う．本体のセッションの記録で，第 6 回の後（07:39）の設計の書き換えが 9 節の 1 文（「差分だけ gate-code に再検査させる（2026-11-09 07:3x）．通れば…」→「…第 6 回「通す」（`code_v6.md`…）を得た（2026-11-09 07:4x）．4 体とも通ったので…」）だけであることを見つけ，e9f66c9 の設計からその 1 文を元に戻した文の sha256 が `7825ef06…` と一致することを確かめた（`design_revert.py`）．したがって第 6 回の後の設計の変更は検査の記録の 1 文だけである．

## 指摘

### [記録] 1．`provenance.json` のコミットは集計の段の d0a2d91 で，計算をした単位の段のコミット e9f66c9 は `provenance_log.jsonl` と npz にだけある
- 重さ：記録する
- 該当箇所：`DAI/exp/dai12_probe.py:1122`（`json.dump(prov, open(out / "provenance.json", "w") …` は集計の段だけが書く），`DAI/docs/実験12_下調べ_結果の文の下書き_2026-11-09.md` の冒頭（「コミット e9f66c9（dirty は偽）．設定とコミットは `results/dai12_probe/provenance.json`」．読んだだけ）
- 問題：結果の文の下書きは，コミット e9f66c9 の出典を `provenance.json` としているが，`provenance.json` の `commit` は d0a2d91 である．`DAI/CLAUDE.md` の取り決めでは結果の文の冒頭に `provenance.json` の場所を書くので，読み手は計算のコミットを d0a2d91 と読む．
- 計算の後で直せるか：直せる（記録は全部ある）
- 求める対応：結果の文の冒頭に「単位の段（判定と点数の計算）はコミット e9f66c9（`provenance_log.jsonl` の 1 行目と各 npz の `commit`），集計の段は d0a2d91（`provenance.json`．差は DAI-13 のファイルだけで，DAI-12 の台本と部品の sha256 は同じ）」と書く．

### [記録] 2．1 回目の集計の失敗がログに残っていない
- 重さ：記録する
- 該当箇所：`logs/dai12_probe_driver.log` 3 行目，`logs/dai12_probe_sum.log`，`DAI/run/dai12_probe.sh:31-32`
- 問題：起動の台本は「集計の段を終えた $(date …)」だけを書くが，ドライバのログの 3 行目には「（DAI-13 のファイルのコミット d0a2d91 の後に流し直した）」があり，起動の台本の外で書かれている．1 回目の集計の失敗の出力は流し直しで `dai12_probe_sum.log` が上書きされて消え，`.failed` も消された．`logs/dai12_probe.done`（17:25）も起動の台本の外で作られた．失敗の理由（未追跡の DAI-13 のファイルによる未コミットの検査）は台帳 216 行にだけある．
- 計算の後で直せるか：直せる（失敗は記録を書く前の検査で起きたので，出力には影響しない．`provenance_log.jsonl` が 2 行だけであることと合う）
- 求める対応：結果の文の記録の節に，台帳 216 行の経緯（1 回目の集計が未コミットの検査で止まった，DAI-13 のファイルをコミットして集計の段だけを手で流し直した，ドライバのログの 3 行目と `.done` は手で書いた）を 1 文で書く．

### [記録] 3．GPU の割り当ての最大が設計 6 節の試走の値より大きい
- 重さ：記録する
- 該当箇所：`results/dai12_probe/units_time_20261109-163708.csv`，`logs/dai12_probe_units.log`（例：`MELD_MOSEI_f4_r0` の「GPU 最大 9.7 GB（予約 10.7 GB）」）
- 問題：設計 6 節は MOSEI の試走で 8.5〜9.1 GB と書くが，本計算の最大は 9.7 GB である．予約 10.7 GB と 6 割の上限 14.7 GB の中なので，計算の進み方には影響していない．
- 計算の後で直せるか：直せる
- 求める対応：結果の文の所要の記述に本計算の値（単位の段 8 時間 57 分，集計 47 分，割り当ての最大 9.7 GB，予約 10.7 GB）を書く．

### [記録] 4．DAI-13 の特徴の取り出しと GPU を共有した 2 単位がある
- 重さ：記録する
- 該当箇所：`logs/dai12_probe_units.log` の `MOSEI_MELD_f0_r0`（594 秒）と `MOSEI_MELD_f0_r1`（548 秒），台帳 215 行
- 問題：2 単位はほかの MELD の単位（226〜257 秒）の 2 倍以上かかった．台本は乱数の種を単位の名前から決め（`:409`），照合の不一致はこの 2 単位でも 0 本なので，判定が変わった証拠は無い．ただし，同じ GPU で別の処理が走ったときに判定が同じになることは確かめていない（読んだだけ）．
- 計算の後で直せるか：直せる（気になるなら，この 2 単位だけを一時の出力先に流し直して int8 の判定の鍵を突き合わせる．1 単位 4 分前後）
- 求める対応：結果の文の記録の節に 2 単位の名前と所要を書く．流し直しの突き合わせは必須ではない．

## 目録に無い種類の不備

- 段を分けて別のコミットで流したとき，`provenance.json` が最後の段の記録だけを持つ（指摘 1）．`docs/methods.md` 9.3 節の I の項に「段ごとにコミットが違いうる台本では，`provenance.json` に段ごとのコミットを持たせるか，結果の文に段ごとのコミットを書く」を足す案を出す．
- 起動の台本が止まった後に段を手で流し直すと，失敗のログが上書きされる（指摘 2）．起動の台本で段ごとのログの名前に時刻を付ける案を出す．

## 台帳用：この点検で見たもの（2026-11-09 17:2x〜17:4x，gate-code の計算の後の点検）

見たのは，`results/dai12_probe/` の単位の名前と個数，npz の鍵・型・形・値の範囲の外の個数・有限でない値の個数，`commit`・`commit11`，候補とテストの sample_id と塊の重なりの数，`half_counts` の数え直し，`checks_units.csv` の照合の不一致（0 でないのは `AFEW_MELD_f4_r0` の `V_K25_pool_proj_enc2_vs_dai11` の 1 本だけ）と `frac` の鍵が 0〜1 の中であること，集計の csv の行数・列名・重複・NaN と inf の列ごとの個数，`provenance_log.jsonl`・`provenance.json`，所要と GPU（`units_time_*.csv` とログ），本体のセッションの記録の中の 07:30〜07:45 の設計の書き換えの命令．UAR・差・区間の値と判定の文字列（`verdict`，`status`，`note_*`）は印字していない．
**性能の値から決まる個数を 2 つ見た**：(1) `cells_nl.csv` の `share_*` が NaN の行の数 155／420（30 単位の平均の Δ が 0.5 未満のセルの数と同じ），(2) `criteria.csv` の R2 の 480 行のうち `K_up` が空の行 411，`K_down` が空の行 412（テストの差で「上回った」K が無い行と「下回った」K が無い行の数．出所・c・モダリティ・量の内訳は見ていない）．ほかに，`q2_cells.csv` の `sub_robust` が MOSEI の 13 行で NaN でないこと（値は見ていない）と，Q2 の 30 セルの (c, 出所) ごとの数を見た．
