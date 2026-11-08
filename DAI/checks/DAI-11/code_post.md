# 計算の後の点検：結果ファイルの欠けと重複（DAI-11 の下調べ，2026-11-09，コミット 67e3553）

点検した対象：
- 結果：`results/dai11_probe/`（`enc2/source/` の 28 個の npz と `summary.csv`，`rowsA/` の 120 個の `*_enc2k.csv`，`select/` の 3 ファイル，`rows/` の 120 個の npz と 120 個の `*_req.csv`，集計の csv 10 本，`provenance.json`，`provenance_log.jsonl`，所要の csv 2 本）
- ログ：`logs/dai11_probe_driver.log`，`_source2.log`，`_enc2k.log`，`_select.log`，`_units.log`，`_sum.log`，`logs/dai11_probe.done`

本計算のコミットは 67e3553 で，今の HEAD と同じである．`git diff --stat 67e3553 HEAD -- DAI/exp DAI/run` は空で，作業木の `DAI/exp/` と `DAI/run/` に未コミットの変更は無い．`DAI/exp/dai11_probe.py` の sha256 は ae18a59b…（関門 1 の第 3 回 `code_v3.md` で検査した版）で，`git show 67e3553:DAI/exp/dai11_probe.py` の sha256 とも一致する．起動の台本 `DAI/run/dai11_probe.sh` の sha256 は 8bf987d3…（`code_v3.md` と同じ）である．

この報告には判定の値（UAR・差・κ・Δ・規則の可否・順位相関の値・選ばれた引き戻しの値）を書かない．書くのは行の数・欠け・重複・一致の有無・NaN の数と位置・所要だけである．点検の台本を流したときに `criteria.csv` の `K_up`・`K_down` の空でない行の分布と `verdict` の値の種類が画面に出たが，報告には書かず，判断にも使っていない．

## 判定

**通す．**再計算も集計のやり直しも要らない．確かめた点は次の 6 つである．
- 予定の 120 単位が 3 つのフォルダ（`rowsA/`，`rows/` の npz，`rows/` の `_req.csv`）にそろい，単位の名前の集合は 3 つとも DAI-1 の `*_p3.npz` の集合と一致する．重複は無い．
- npz の鍵の組は 120 単位で 1 種類であり，設計 5 節と関門 1 の B で足すと決めた鍵がすべてある．
- 照合（`none` の判定，候補の確率，`fuse1` と DAI-1 `ret` の W）は全単位で許しの中にあり，有限でない値と収束しない解は 0 である．
- 集計の csv の行の数は予定と一致し，重複は 0 である．NaN は作りの上で出る位置にだけある．
- provenance のコミットと sha256 は流した版と一致し，5 つの段の記録がそろう．
- ログに例外・警告・メモリ不足の行は無く，`--allow-partial` と `--resume` は使っていない．

記録すべき点が 3 件ある（6 節）．どれも数値には影響しない．

## 1. 単位の集合（確かめること 1）

| 確かめたこと | 結果 |
|---|---|
| 予定の単位 | `results/dai1/pseudo/units/` の `*_p3.npz` が 120 個 |
| `rowsA/*_enc2k.csv` | 120 個．ほかのファイル 0 |
| `rows/*.npz`，`rows/*_req.csv` | 120 個ずつ．npz と `_req.csv` 以外のファイル 0 |
| 名前の集合 | 4 つ（予定，rowsA，rows の npz，rows の `_req.csv`）が一致した．名前の重複 0 |
| 評価のターゲット T ごと | AFEW，IEMOCAP，MELD，MOSEI が 30 ずつ．(T, c) の 12 組が 10 単位ずつ（f 5 × r 2） |
| npz の中の単位の情報 | `unit`・`T`・`c`・`fold`・`seed` がファイル名と食い違う単位 0 |
| テストの並び | 120 単位すべてで `test_sample_id` が DAI-1 の `_p1.npz` の `sample_test` と同じ並び |
| K 本 | 120 単位 × 4 つの K で，`kpos_K{K}` が指す候補の sample_id の集合が DAI-1 の `sample_K{K}` と同じ．本数は 4K で，各感情 K 本ずつ．位置の重複 0 |
| DAI-1 の入力 | `results/dai1/pseudo/units/` の最新の更新時刻は 2026-11-04 で，本計算の開始（11-08 22:27）より前．provenance の `pseudo_commit` は固定のファイルと一致 |

## 2. 行のファイルの鍵・形・値域（確かめること 2）

| 確かめたこと | 結果 |
|---|---|
| 鍵の組 | 120 単位で 1 種類，863 鍵．保存の鍵 844（単位の段のログの「保存 844」と同じ．`code_v3.md` の試走と同じ数）と，単位の情報 19 鍵（commit，unit，T，c，fold，seed，n_pool，n_over_d，test_sample_id，test_y，test_cluster，pool_y，pool_cluster，kpos_K1〜K25，mismatch，conv） |
| 予定の鍵との差 | 台本の `TEACHERS` から作った予定の鍵 863 と比べ，欠け 0，余分 0 |
| テストの判定 154 鍵 | `K{K}_{m}_none`，`_noneloco`，`_enc2none`，学び直した版 `_<出所>`（A・V・T は 5 出所，AVT は 3 出所），`{m}_oracle`．全単位で int8，形は（テストの本数，），値域 0〜3 の外 0 |
| 候補の判定 470 鍵 | `pool_none_K{K}_{m}` 20 と，`pool_teacher`・`pool_proj`・`pool_projbal`・`pool_projbal_t`・`pool_student` の各 90（(m, 出所, K) の組）．全単位で int8，形は（候補の本数，），値域 0〜3 の外 0 |
| W・b 220 鍵 | 学び直した版の `W_K{K}_{m}_<出所>`・`b_…` 180 と，第 2 エンコーダの K 本だけの版の `enc2_W_K{K}_{m}`・`enc2_b_…` 40（K = 0 は出発点）．全単位で float32．W の形は（4，次元），b は（4，）で，次元は第 1 エンコーダ A 1,024・V 512・T 768・AVT 2,304，第 2 エンコーダ A 1,024・V 1,024・T 768・AVT 2,816（設計 2 節と同じ）．W の向きは DAI-1 の `ret` の W と同じ（4，次元）である |
| 有限でない値 | W・b で 0．`enc2/source/` の 28 個の npz の浮動小数の配列でも 0 |
| `test_y`・`pool_y` | int8，値域 0〜3．全単位のテストに 4 つの感情がそろう．`n_pool` は `pool_y` の長さと一致 |
| `n_over_d` | 4 つの分類器の値が全単位で（候補の本数 ÷ 次元）と一致 |
| `mismatch` | 各単位 43 鍵（`none` の判定 20，候補の確率 8，`fuse1` の W の差 15）．有限でない値 0 |
| `conv` | 各単位 8 塊（`enc2_{A,V,T,AVT}`，`student_{A,V,T,AVT}`） |
| 確率の和 | 行のファイルは判定（int8）だけを保存し，確率の配列を持たないので，和の点検は当たらない．候補の確率の作り直しは DAI-1 の保存と照合した（4 節） |

## 3. 集計の csv（確かめること 3）

予定の行の数は，テストの判定の鍵 154，(m, 出所, K) の組 90（A・V・T は 5 出所 × 5 K，AVT は 3 出所 × 5 K），R3 に入る組 70（A・V・T は 4 出所 × 5 K，AVT は `enc2`・`other` × 5 K）から出した．

| ファイル | 行の数（予定） | 識別子の重複 | NaN |
|---|---|---|---|
| `uar_units.csv` | 18,480（120 単位 × 154 鍵） | (c, T, fold, seed, key) 0 | 0．UAR・WAR・WAF・macro-F1 の 0〜100 の外 0 |
| `cells.csv` | 616（4 コーパス × 154 鍵）．全行の `n_units` が 30 | (c, key) 0 | 0．8 つの指標の列の 0〜100 の外 0 |
| `diffs.csv` | 3,015．全体の行 1,340（4 コーパス × 5 K ×（A・V・T の 19 比べ × 3 + AVT の 10 比べ））．部分集合の行 1,675（AFEW の asr_ok，MELD の face_single，MOSEI の channel，IEMOCAP の marker・nomarker が 335 ずつ） | (c, student, K, a, b, subset) 0 | `est`・`lo`・`hi`・`se`・`p` の NaN 0．ほかの列は作りの上の位置だけ（下） |
| `req_units.csv` | 10,800（120 単位 × 90） | (unit, student, teacher, K) 0 | 0．2×2 の 4 つの本数の和は全行で `n_nk` と一致し，`n_nk` は全行で `n_pool` − 4K．`lam_rel` は 6 点の格子の中．一致率の 0〜1 の外 0，UAR の 0〜100 の外 0，R² > 1 は 0，κ の −1〜1 の外 0 |
| `req_cells.csv` | 360（4 コーパス × 90）．全行の `n_units` が 30 | (c, student, teacher, K) 0 | `share_cell` だけ．NaN は Δ の平均が 0.5 未満のセルにだけあり，0.5 以上で NaN のセル 0，0.5 未満で値のあるセル 0（設計 3.4 節の定義どおり） |
| `req_vs_gain.csv` | 280（4 コーパス × 70） | (c, student, teacher, K) 0 | `share_cell` だけ（上と同じ位置） |
| `rank_corr.csv` | 247．部分集合 12 種類（all，K>=1，excl_AVTxother，AVT_students_excl_other，student 4 つ，c 4 つ）× 20 行，ただし student=AVT は 18 行．数え上げ 3 行と D8 の割合 6 行 | (subset, kind, quantity) 0 | 下 |
| `criteria.csv` | 196（4 コーパス ×（A・V・T の 14 規則 × 3 + AVT の 7 規則））．全行の `n_K` が 4 | (c, student, rule) 0 | 下 |
| `checks_units.csv` | 5,160（120 単位 × 43） | (unit, key) 0 | 0 |
| `conv_units.csv` | 12,720（120 単位 × 106．enc2 の 4 × 4 と学び直しの 90） | (unit, block, item) 0 | 0 |

**列が設計 5 節と合うか**：
- `cells.csv`：UAR・WAR・WAF・macro-F1・感情ごとの再現率 4 つ・`n_units` がある．
- `diffs.csv`：UAR の `effect_stats` の列（`est`・`lo`・`hi`・`se`・`p`・`method`・`n_units` と `es_*`），WAR・WAF・macro-F1 の分割の t の列（`*_est`・`*_lo`・`*_hi`），`subset` 列がある．比べは台本の `PAIRS` の 19 種類（AVT は `fuse1`・`enc2fuse` を含む 9 種類が無く 10 種類）．
- `req_cells.csv`：3.4 節の量（Δ と WAR・WAF・macro-F1 の Δ，`delta_uar_bal`，`uar_t`，一致率，`comp2`，`kappa`，`phi`，2×2 の本数，`uar_proj`，`proj_gain`，`lam_rel`，`r2`，D8 の一致率 8 列，`uar_projbal`，`uar_projbal_t`，`uar_student_pool`，`n_over_d`），`n_units`，`share_cell` がある．
- `rank_corr.csv`：中心を引いた順位相関（8 つの量 × 塊 2 通り），偶然の水準 `null95`，部分相関（統制 2 つ × 塊 2 通り），数え上げ，D8 の (i)・(ii)・(i)&(ii) を 2 つのそろえ方で出した行がある．
- `criteria.csv`：14 の規則（R0，R1，R1loco，R1p，R1info，R1f1，R1f，R2a〜R2f，SEL），可否，R4 の WAR・WAF・macro-F1 の符号と区間の数，MOSEI の channel の列，R1 の行の一致率と κ の平均がある．

**`diffs.csv` の NaN が作りの上で出る位置**：
- 部分集合の 1,675 行：`es_*` の列と WAR・WAF・macro-F1 の列がすべて NaN．部分集合は UAR の区間だけを出す作りである（設計 5 節，台本 721〜723 行）．
- `es_df`：全体の行では MOSEI の 335 行だけ（塊の引き直しの区間は自由度を持たない）．
- `es_boot_lo`・`es_boot_hi`：全体の行では AFEW・IEMOCAP・MELD の 1,005 行（jackknife の区間はこの列を持たない）．
- `es_t`：全体の行では MOSEI の 335 行（列が無い）と，AFEW・IEMOCAP・MELD の `none − noneloco` の 28 行だけ．28 行は `se` = 0 である．`se` = 0 の行は全体の行で 44 あり，すべて `none − noneloco` である．K = 0 の 16 行は台本が `noneloco` に `none` の判定をそのまま入れる作り（台本 393 行）による．K ≥ 1 の行は，c を除いて選び直した引き戻しが全単位で同じ点になり判定が一致した場合である．
- `es_folds_dropped`：空の行が NaN として読まれる．空でない行は AFEW の全体の 335 行（値は「0」）と IEMOCAP の全体の 335 行（値は「0,1,2,3,4」）で，DAI-10 の `code_post.md` に書いた位置と同じである．
- 全体の行の WAR・WAF・macro-F1 の列の NaN は 0 である．

**`rank_corr.csv` の NaN**：
- `rho`・`null95` の NaN は 11 行で，数え上げと D8 の 9 行（この 2 列を持たない）と，student=AVT の部分相関 2 行だけである．AVT にとってはどの出所も印が 0 なので，部分相関に入る行が 0 になる（設計 4.3 節．`n_cells` = 0 で，台本は塊の並べ替えの行を作らず 1 行ずつ書く）．このため student=AVT は 18 行になる．
- `value` の NaN は 238 行で，順位相関の行（この列を持たない）だけである．
- 中心を引いた順位相関の `n_cells` は，all 280，K>=1 224，excl_AVTxother 240，AVT_students_excl_other 180，student=A・V・T 各 80，student=AVT 40，c ごと 70 で，予定の組の数と一致する．8 つの量で `n_cells` は同じ 280（all）で，量の NaN で落ちたセルは 0 である．部分相関の `n_cells` は AVT の 40 を除いた数（all 240 など）である．

**`criteria.csv` の NaN**：
- `chan_n_K_hit`・`chan_n_K_robust`：AFEW・IEMOCAP・MELD の 147 行（3 × 49）が NaN．MOSEI だけで数える作りによる．
- `enc2_agree_mean`・`enc2_kappa_mean`：R1 の 16 行（4 コーパス × 4 分類器）だけに値があり，ほかの 180 行は NaN．R1 の行にだけ添える作り（台本 806〜808 行）による．
- `K_up`・`K_down`：条件を満たす K が無い行で空の文字列になり，NaN として読まれる．数は判定の値に依るので書かない．
- ほかの列の NaN は 0 である．

## 4. 照合と収束（確かめること 4）

| 確かめたこと | 結果 |
|---|---|
| `none` の判定の作り直しと DAI-1 の保存 | 2,400 行（120 単位 × 4 分類器 × 5 K）．不一致のある単位 0，不一致の合計 0 本（許しは max(1, テストの 0.1%)） |
| `none` の候補の確率と DAI-1 の保存（float16） | 960 行（120 単位 × 4 分類器 × K = 0, 5）．最大の差 2.4e-4（許し 0.02） |
| `fuse1` で学び直した W と DAI-1 の `ret` の W | 1,800 行（120 単位 × A・V・T × 5 K）．最大の差 2.3e-5（許し 1e-4） |
| 有限 | `conv_units.csv` の `finite` が偽の行 0．W・b の有限でない値 0（2 節）．段 A の K ≥ 1 の行で `finite` が偽の行 0 |
| 収束 | `conv_units.csv` の `stalled` が真の行 0．勾配の最大 1.0e-6．反復の最大は enc2 の塊 122，学び直しの塊 426．段 A の K ≥ 1 の 26,880 行でも `stalled` が真の行 0，勾配の最大 1.0e-6 |

## 5. 選ぶ段（確かめること 5）

| 確かめたこと | 結果 |
|---|---|
| 段 A の行 | 27,360 行（120 単位 × 228．4 種類 ×（出発点 1 + 4 K × 14 点））．(unit, kind, K, Cw, Cb) の重複 0．全行のコミット 67e3553．K ≥ 1 の (unit, kind, K) ごとに 14 点で，点の集合は `dai1.TETHER` の 14 点と同じ．UAR の NaN 0，0〜100 の外 0．NaN は K = 0 の 480 行の `Cw`・`Cb`・`grad`・`finite` だけ（出発点は引き戻しを持たない作り） |
| `select/enc2_tether.json` | コミット 67e3553．T 4 つ × (種類, K) 16 で，選ばれた点はすべて 14 点の格子の中．各 T の `n_units` は 30 で，`units` は段 A の T の単位の名前の集合と一致 |
| `select/none_loco_tether.json` | コミット 67e3553．T 4 つ × c 3 つ = 12 組．各組 28 鍵（DAI-1 の p1 の 7 種類 × 4 K）で，選ばれた点はすべて 14 点の格子の中．各組の `n_units` は 20 |
| `select/enc2_grid.csv` | 896 行（T 4 × 種類 4 × K 4 × 14 点．設計 5 節のとおり T ごとの平均の表なので単位の数は掛からない）．(T, kind, K, Cw, Cb) の重複 0．(T, kind, K) ごとに 14 行で，`selected` が真の行は 1 つずつ．選ばれた行は `enc2_tether.json` と一致．`grid_oracle_gap_per_TkindK` は (T, kind, K) の 14 行で同じ値で，負の値 0．NaN 0 |
| ログ | 「enc2_tether.json（T 4），none_loco_tether.json（T 4），enc2_grid.csv（896 行）」 |

## 6. provenance（確かめること 6）

| 確かめたこと | 結果 |
|---|---|
| `provenance.json` | 集計の段の記録．`commit` 67e3553，`dirty` 偽，`script_sha256` ae18a59b…（作業木の版と `git show 67e3553` の版の両方と一致），GPU は RTX A5000 |
| 共通の部品の sha256 | `dai1.py`，`dai_solver.py`，`dai_data.py`，`dai1_stats.py`，`dai3.py`，固定のファイル `dai1_frozen.json` が今のファイルと一致 |
| 入力 | `pseudo_commit` は固定のファイルと一致．`features_sha256` は 24 ファイル（6 つの特徴 × 4 コーパス）で，すべて今の `data/cache/` のファイルと一致 |
| ソフトの版 | Python 3.12.3，torch 2.11.0+cu126，numpy 2.5.2，pandas 3.0.6，scipy 1.18.1，scikit-learn 1.9.1 |
| 定数 | `KS_ALL`，`KS_RULE`，`LAM_REL`，`FLOOR`，`MISMATCH_FRAC`，`PROB_TOL`，`TEACHERS`，`PAIRS`，`RULES`，`BIG_DELTA`，`SEED`，`N_PERM`，`TETHER` がある |
| `provenance_log.jsonl` | 5 行（source2，enc2k，select，units，summarize）．5 行とも commit 67e3553，`dirty` 偽，`script_sha256` が同じ．引数は 5 行とも `--out results/dai11_probe --stage <段>` だけで，`--allow-partial`・`--resume`・`--allow-dirty`・`--units` は無い |
| 段ごとのコミットの照合 | `enc2/source/summary.csv` の 28 行，段 A の全行，2 つの json，`rows/` の 120 個の npz のコミットがすべて 67e3553 |

## 7. ログと所要（確かめること 7・8）

- 起動の台本のログ：開始 22:27:33，ソースの段の終わり 23:02:21（約 35 分），段 A の終わり 23:20:57（約 19 分），選ぶ段の終わり 23:21:01（4 秒），段 B の終わり 23:52:48（約 32 分），集計の段の終わり 23:57:54（約 5 分）．合計約 1 時間 30 分で，設計と関門 1 の見込み（2〜2.5 時間）の中に収まった．`logs/dai11_probe.done` があり，`.failed` は無い．
- 1 単位の所要：段 A は 3〜19 秒（合計 18.6 分），段 B は 6〜37 秒（合計 31.7 分）．所要の csv は 120 行ずつで，単位の重複 0．
- 段 A と段 B のログは 120 行ずつ．段 B の各行は「判定の不一致の最大 0 本」である．
- 6 本のログ（標準エラーを含む）で，warn，error，traceback，exception，nan，inf，out of memory，cuda，killed，retry を含む行は 0 だった．段の開始と終わり，単位ごとの進み具合，ソースの段の C の選択の行のほかに出力は無い．
- **120 単位すべてで集計したか**：集計の段は `--allow-partial` なしで流れ（`provenance_log.jsonl` の引数），正常に終わった．集計の台本は単位の npz が 120 個でないとき（台本 649〜650 行）と，`cells.csv` に単位の数が 30 でないセルがあるとき（台本 698〜699 行）に止まる作りで，`cells.csv` の 616 行の `n_units` はすべて 30 だった．確かめが通ったことを書いた専用の行はログに無い（下の記録 1）．

## 8. 後で求められそうな解析を保存した出力から出せるか

関門 1 の第 1〜3 回で B に足すと決めた出力は，本計算の 120 単位すべてにある．
- 第 2 エンコーダの W・b（`enc2_W_`・`enc2_b_`．K = 0 は出発点）
- 候補の判定 `pool_none`・`pool_teacher`・`pool_proj`・`pool_projbal`・`pool_projbal_t`・`pool_student`
- テストの判定（int8）と `test_sample_id`・`test_y`・`test_cluster`，候補の `pool_y`・`pool_cluster`，`kpos_K{K}`
- 学び直した版の W・b
- 照合の数（`mismatch`）と収束の記録（`conv`．`finite` を含む）
- 単位ごとの要件の量 `_req.csv`

これらから，ターゲットごとの内訳，話者を単位にした区間（`test_sample_id` から発話一覧の `speaker_id` を引く），混同行列，候補の上でどの感情を直したか，W の向きと切片の分解を，学び直しをせずに出せる．

## 9. 記録（再計算も集計のやり直しも要らない）

1. **n_units の確かめが通った記録がログに無い．**集計の台本（`DAI/exp/dai11_probe.py:698-699`）は単位の数が 30 でないセルがあると止まるが，通ったときに何も書かない．`provenance.json` にも DAI-10 の `n_units_found` や `units` に当たる鍵は無い．通ったことは，集計の段が `--allow-partial` なしで正常に終わったことと，`cells.csv` の `n_units` がすべて 30 であることから確かめた．次の実験では，集計の段が見つけた単位の数と名前を provenance に書くとよい．
2. **DAI-1 の単位のファイルの sha256 は provenance に無い．**設計 5 節は provenance に「入力の sha256」を書くとしている．特徴（24 ファイル）と固定のファイルの sha256 は入っているが，DAI-1 の `_p1/_p2/_p3.npz` の sha256 は入っていない．代わりに，台本は各単位を読むときに 3 つの npz のコミットを固定のファイルの `pseudo_commit` と照合する（台本 214〜219 行）．今回の点検では，DAI-1 の単位のファイルの更新時刻が本計算より前で，テストの並びと K 本が DAI-1 の保存と一致することを確かめた．
3. **`rank_corr.csv` の D8 の行の名前．**「出所の総量」の 3 行は `pool_projbal_t` との一致率から計算しているが（台本 789〜794 行），`subset` の文字列は「射影自身の総量」の行と同じ `agree(student,projbal) > agree(none,projbal)` を含む．区別は頭の「射影自身の総量」「出所の総量」の語だけでつく．結果を書くときは頭の語で読む．
4. 渡された一覧の「`enc2/source/` の 29 npz」は，正しくは 28 個の npz（4 コーパス × 7 種類）と `summary.csv` の 29 ファイルである．DAI-1 の `results/dai1/source/` も同じ 29 ファイルである．

## 10. 再計算か集計のやり直しが要るか

どちらも要らない．

---

点検の台本は `DAI/checks/DAI-11/code_run_post/`（`check_units.py`，`check_csv.py`，`check_select.py`，`check_misc.py`．git は無視する）に置いて流した．どれも数分以内に終わった．`results/` の下のファイルは読んだだけで，書き換えていない（`logs/dai11_probe.done` より後に更新された `results/dai11_probe/` の下のファイルは 0）．
