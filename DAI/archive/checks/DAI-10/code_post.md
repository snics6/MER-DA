# 計算の後の点検：結果ファイルの欠けと重複（DAI-10 の下調べ，2026-11-08，コミット ce52b51）

点検した対象：
- 結果：`results/dai10_probe/`（`rows/` の 120 個の npz，`cells.csv`，`diffs.csv`，`criteria.csv`，`diag.csv`，`diag_units.csv`，`recall_units.csv`，`checks_units.csv`，`provenance.json`）
- ログ：`logs/dai10_probe_launch.log`，`logs/dai10_probe_units.log`，`logs/dai10_probe_sum.log`，`logs/dai10_probe.done`

本計算のコミット ce52b51 と，関門 1 の第 3 回で検査した 764e3da の差分は，検査役の報告 3 本，設計の 1 行，台帳の 1 行だけである．`DAI/exp/dai10_probe.py` と `DAI/run/dai10_probe.sh` は 764e3da から変わっていない（`git diff --stat 764e3da ce52b51`）．

この報告には判定の値（UAR の差，規則の可否，channel の数，分け方の値）を書かない．行の数・欠け・重複・一致の有無だけを書く．

## 判定

**通す．**次の 4 点を確かめ，再計算も集計のやり直しも要らない．
- 予定の 120 単位がそろい，重複は無い．全単位で鍵の組が同じである．
- 照合の不一致は 0 である．
- provenance のコミット・sha256 は流した版と一致する．
- 集計の各ファイルに欠けと重複は無い．NaN は，作りの上で出る位置にだけある．

## 1. 行のファイル（`results/dai10_probe/rows/`）

| 確かめたこと | 結果 |
|---|---|
| 単位の数 | 120．`results/dai1/pseudo/units/` の `*_p3.npz` から作った予定の 120 単位と同じ集合．名前の重複 0．npz 以外のファイル 0 |
| 仮のターゲットごと | AFEW，IEMOCAP，MELD，MOSEI が 30 ずつ．各 (c, f, r) に T が 3 つずつ |
| 行のコミット | 120 単位すべて ce52b51（HEAD と同じ） |
| 鍵の組 | 120 単位で 1 種類．テストの判定 428，候補の割り当て `P_` 32，係数 `C_` 48，diag の鍵 20（関門 1 の第 3 回の試走と同じ数） |
| 値の範囲 | 判定と `P_` の 0〜3 の外 0，`P_` の長さが `pool_y` と違う鍵 0，係数の有限でない値 0，diag の有限でない値 0 |
| テストの並び | DAI-8 の行の `test_sample_id` と違う単位 0 |
| 照合の合計（`checks`） | K 本だけの LDA：不一致 0 / 9,686,592 本．全ラベルの LDA：不一致 0 / 807,216 本 |

## 2. provenance（`results/dai10_probe/provenance.json`）

| 確かめたこと | 結果 |
|---|---|
| `commit` | ce52b51（HEAD と同じ）．`dirty` は偽，`partial` は偽 |
| 集計の段 | `summarize_commit` は ce52b51，`summarize_dirty` は偽．`rows_commits` は `[commit]` と同じ．`n_units_found` は 120 |
| 台本の sha256 | `git show ce52b51:DAI/exp/dai10_probe.py` の sha256 と一致．作業木の版とも一致 |
| `dai8_sha256`，`frozen_sha256` | 今の `DAI/exp/dai8_probe.py` と固定のファイルの sha256 に一致 |
| 入力の sha256 | 120 / 120 単位で，p1 と DAI-8 の行の sha256 が今のファイルと一致．DAI-8 の 120 個の行のコミットはすべて `d8_commit`（a90af78） |
| `units` と `runs` | どちらも 120．`units` は予定の集合と同じ．`runs` の名前の重複 0 |
| 設定 | `alphas` = [0.01, 0.1, 0.5, 0.9]，`a_rule` = 0.5，`min_d` = 0.5，`n_boot` = 2000，`boot_seed` = 0，GPU は RTX A5000．設計の第 6 版と同じ |
| `ldaK_mismatch_total` | 0 |

## 3. 集計のファイル

| ファイル | 行の数 | 重複 | NaN |
|---|---|---|---|
| `cells.csv` | 1,712（4 コーパス × 428 鍵）．各行の単位の数 30 | (c, key) の重複 0 | 0．UAR の 0〜100 の外 0 |
| `diffs.csv` | 1,580．1 コーパスの全体の行 380，部分集合の行は AFEW の asr_ok，MELD の face_single，MOSEI の channel が 20 ずつ．比べ 26 種類．各行の `n_units` 10，`n_folds` 5 | (c, K, space, compare, subset) の重複 0 | effect・lo・hi・se の NaN 0．ほかの列の NaN は作りの上の位置だけ（下） |
| `criteria.csv` | 4 行（4 コーパス）× 97 列 | c の重複 0 | channel の 8 列（`G1`・`G2`・`G3n`・`G3j` の `_chan_n_K_ok`・`_chan_n_K_robust`）が AFEW，IEMOCAP，MELD の行で NaN．MOSEI だけで数えるので作りの上でそうなる．ほかの列の NaN 0 |
| `diag.csv` | 80（4 コーパス × 20 鍵） | (c, key) の重複 0 | fold・seed の列は無い |
| `diag_units.csv` | 2,400（120 単位 × 20 鍵．K ごとの鍵が空間ごとに 480，縮みの係数の鍵が空間ごとに 120） | (c, T, fold, seed, key) の重複 0 | 鍵の種類ごとに値の入るはずの列の NaN 0 |
| `recall_units.csv` | 5,280（120 単位 × 44 行） | (c, T, fold, seed, key) の重複 0 | 0．再現率の 0〜1 の外 0．判定の割合の和が 1 から 1e-9 を超えてずれる行 0 |
| `checks_units.csv` | 120 | name の重複 0 | 0 |

**規則の列の欠け**：G1，G2，G3n，G3j，Aa，Aa2，Ab，Gn の `_n_K`・`_n_hi_neg`・`_pos`・`_neg`，`_pos_n_corpora`・`_neg_n_corpora` とその `_excl_IEMOCAP`，channel の 8 列，`dec_K{1,5,10,25}_{total, metric_first, assign_after, assign_first, metric_after, interaction}` の 96 列がすべてある．余分な列は無い．

**`diffs.csv` の NaN が作りの上で出る位置**（どれも DAI-8 と同じ作りによる）：
- `df`：MOSEI の 400 行．塊の引き直しの区間は自由度を持たない．
- `boot_lo`，`boot_hi`：AFEW，IEMOCAP，MELD の 1,180 行．jackknife の区間はこの列を持たない．
- `t`：MOSEI の 400 行（列が無い）と，jackknife の 12 行．12 行は K = 1 の `ldaK_ncm` の 4 空間 × 3 コーパスで，`se` = 0 である．K = 1 では LDA_K と NCM の判定が作りの上で全く同じになる（関門 1 の第 3 回で確かめた）．MOSEI の同じ比べも `se` = 0 だった．
- `folds_dropped`：空の行が NaN として読まれる．空でない行は IEMOCAP の 380 行（値は「0,1,2,3,4」）と AFEW の 400 行（値は「0」）である．IEMOCAP は 1 つの分割のテストが 1 つのセッションなので，塊を 1 つ除くとその分割が外れる．AFEW は分割 0 で，ある塊を除くと分割 0 の UAR が作れなくなる．DAI-8 の `diffs.csv` も同じ値（AFEW「0」，IEMOCAP「0,1,2,3,4」，MELD 空）だった．

## 4. ログ

- 起動の台本のログ：開始 10:08:03，単位の段の終わり 10:22:10，集計の段の終わり 10:24:35．`logs/dai10_probe.done` があり，`.failed` は無い．
- 単位の段のログ：照合の行が 120．1 単位の所要は 4〜14 秒（provenance の `runs`）．
- 単位の段と集計の段のログ（どちらも標準エラーを含む）で，次の語を含む行は 0 だった：warn，error，traceback，nan，inf，out of memory．照合の行と開始・終わりの行のほかに出力は無い．

## 5. 後で求められそうな解析を保存した出力から出せるか

関門 1 の第 1〜3 回で (a) とした解析は，本計算の出力にもそろっている．
- 鍵の組は 120 単位で同じである．
- `test_sample_id`，`test_cluster`，`pool_y`，`P_`，LDA_T の係数，diag の回ごとの記録が全単位にある．

出せる解析は次のとおりである．
- ターゲットごとの内訳
- 感情ごとの区間（`recall_units.csv` は単位ごとの値．区間は行の判定から塊の数を作って出す）
- 話者を単位にした区間（`test_sample_id` から manifest の `speaker_id` を引く）
- AFEW と IEMOCAP の性別
- 4 つの角の交互作用の区間（2 つの判定の数を足して `effect_stats` に渡す）
- IEMOCAP の A・V の空間の差（`diffs.csv`）
- MOSEI の channel，AFEW の asr_ok，MELD の face_single の部分集合（`diffs.csv` の `subset` 列）
- EM_Kmet・EM_Amet の割り当ての移り方（`P_` と `pool_y`，K 本の位置は `dai_data.k_orders`）

**記録（再計算は要らない）**：部分集合ごとの方法の UAR は `cells.csv` に入らず，`diffs.csv` の差だけがある．部分集合の UAR そのものが要るときは，`rows/*.npz` の判定を `dai3.subset_masks` で絞って数え直す．

## 6. 再計算か集計のやり直しが要るか

どちらも要らない．

---

点検の台本は作業用のフォルダ `/tmp/claude-1000/-home-snics-MER-DA/b1f27254-32c7-45f0-ac5a-dee1c5e3ec8a/scratchpad/dai10post` に置いて流し，点検の後に消した．`results/` の下のファイルは読んだだけで，書き換えていない．
