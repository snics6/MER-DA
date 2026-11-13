# 関門 1（再検査）：実装と出力（DAI-17 の計算の後に出す量の台本 `DAI/exp/dai17_posthoc.py`，2026-11-14，コミット cf2c11f．台本の sha256 2a754ba5…81f5）

## 判定

通す．前回の 5 つの指摘はすべて台本で直っており，流し直した出力は `results/dai17_probe/posthoc/` の 9 つの csv と全セルで一致し，同じ俳優の対と期待値は別の書き方で数え直した値と 600 行すべてで一致し，依存の表は点検の台本の出力と同じだった．残る指摘は記録の水準（期待値の比べ方を 1 つ足せること，説明文と provenance の不足）だけである．

## 検査の範囲と守ったこと

- 差分：`git diff fa21def cf2c11f -- DAI/exp/dai17_posthoc.py`（c2bf52e と cf2c11f の 2 回分）．台本の全文を読み直した．HEAD（3cd1ccd）と cf2c11f の間で台本は変わっていない（sha256 が同じ）．
- 読んだ周辺：`DAI/exp/dai17_probe.py` の `save_srcdb`・`topk_idx`・`topk_z`・`nn_rows`・`run_unit` の近い発話を引く箇所（235〜396 行），`DAI/exp/dai_data.py` の別名の表（44〜51 行），設計 `DAI/docs/実験17_下調べ_設計.md` 238 行（AFEW 11 発話，MELD 62 発話），点検の台本 `DAI/checks/DAI-17/code_run_post/dependency_table.py`．
- GPU は使っていない（`CUDA_VISIBLE_DEVICES=""`）．`results/dai17_probe/` は読むだけにした．流すときは `DAI/checks/DAI-17/code_run_posthoc_v2/out/` に `units`・`srcdb`・`students`・`summary`・`frozen_sel.json`・`provenance.json` への symlink を置き，`posthoc/` だけをそこに書かせた．`results/dai1/eval/` は開いていない．
- この報告には UAR の値，差の符号，選ばれた点を書かない．検算の台本は一致の数と形だけを出し，選ばれた点の名前と UAR を画面に出していない．結果の文の下書きは今回は開いていない．

## 前回の指摘の扱い

| 前回の指摘 | 直した箇所 | 確かめた結果 |
|---|---|---|
| [I] 俳優の対応表から `Jessica_Lockhart` が抜けている（直してから進む） | 26〜29 行（役名を組にし `Jessica_Lockhart` を足した），44 行（`isin(meld_roles)`），94〜103 行（表に入る発話を数え，設計の数と違えば `SystemExit`） | 直った．別名の表 `DD.MELD_ALIAS`・`DD.AFEW_ALIAS` を通して独立に作った対応表（`check_actor.py`）でも，表に入る発話は AFEW 11，MELD 62 で台本と同じ．`Jessica_Lockhart` の 2 発話は発話一覧の上でどちらも 7 期 15 話で，話の絞り込みで落ちない．`srcdb` の `_A`・`_V`・`_T`・`_J` の `sid` の並びは 6 つのソースの組のすべてで同じなので，`_A` だけで数える 96 行は 4 つの枠を代表する |
| [A] 同じ俳優の対に期待値と，対が起こりうる単位だけの分母が無い（直してから進む） | 121〜132 行（`possible`，`exp_random_100`，`exp_random_20`，`n_db_actor`．空間 `both`・`feat`・`logit`），145〜147 行（`possible` の単位だけを足す） | 直った．期待値の式は依頼のとおり（テストの俳優の発話ごとに「表の中の同じ俳優の発話の数 × 引く本数 ÷ 表の発話の数」を足す．129 行）．近い発話は表の全行から引いている（`dai17_probe.py:250`，`:269` の `topk` は表の全行が相手で，除く行が無い）ので，分母を表の全行の数にするのは正しい．非復元で k 本引いても，同じ俳優の本数の期待値は k × 数 ÷ 表の大きさで同じである．発話ごとの二重の輪で別に数え直すと，`n_pairs_same`・`n_pairs_same_top20`・`exp_random_100`・`exp_random_20`・`possible` は 600 行すべてで一致した．`same_actor.csv` の 20 行は単位の表の `possible` の行の和と一致し，`possible` が偽の単位で同じ俳優の対は 0 だった．`n_units_possible` は AFEW 8，MELD 6 で前回の検算と同じ |
| [R] ソースごとの感情ごとの当たる割合が `both` だけ（記録する） | 133〜141 行（空間の輪），148〜149 行（`n_emotions`・`n_pairs_source` を残す） | 直った．`by_source_bal.csv` は 120 行（c 4 × 枠と空間の 10 × ソース 3）で，各行の数は 10，`n_emotions_min` はすべて 4，`n_pairs_source_min` の最小は 540．「対をまとめて数える定義」を文書に 1 行書くことは台本の外で，今回は確かめていない |
| [R] 依存の表を作る台本が `DAI/exp/` に無い（記録する） | 52〜87 行（`dependency_table`），163 行 | 直った．`dependency_count.csv`（40 × 6）は `DAI/checks/DAI-17/code_run_post/dependency_count.csv` と全セルで同じ．`dependency_detail.csv`（120 × 8）も，点検の台本だけにある列 `reproduced_with_30` を除く全セルで同じ．`n_units_used` はすべて 20．点検の台本の `reproduced_with_30` が偽の組は 0 |
| [記録] provenance に入力の sha256 が無い，`n_test_actor` の重なり | 168〜174 行，147 行の注記 | 一部直った．`students_sel.csv`・`neighbors.csv`・`frozen_sel.json`・2 つの発話一覧の sha256 と，単位のコミットが入った．`n_test_actor` の重なりは台本の注記だけで，csv の列名には出ていない（下の指摘 3） |

## A．設計と台本の対応表（変わった箇所だけ）

| 設計の項目 | 台本の箇所 | 合っているか |
|---|---|---|
| 7 節：同じ俳優の対応表（AFEW 11 発話，MELD 62 発話．`sample_id` の水準） | 26〜49 行，94〜103 行 | 合っている．数が違えば止まる |
| 7 節・前回の指摘：同じ俳優の対を期待値と並べる．`feat`・`logit` でも出す | 121〜132 行，145〜147 行 | 合っている．J の枠は `both` だけ（J には `feat`・`logit` の番号が保存されていない．`overlap` と同じ扱い） |
| 3.7 節：ソースごとの感情ごとの当たる割合（枠を限らない） | 133〜141 行，148〜149 行 | 合っている．`summary/neighbors_by_source.csv` と同じ 10 の枠と空間 |
| 3.9 節：c′ の点を c を除いた 20 単位で選び直したときに変わる組 | 52〜87 行 | 合っている．選ぶ規則（単位の平均の最大，同点は候補の並びで先）は点検の台本と同じで，点検の台本はこの規則で 30 単位から `frozen_sel.json` を再現していた |
| DAI の取り決め：provenance | 166〜174 行 | コミット cf2c11f，dirty 偽，台本の sha256 は流した台本と同じ．入力の一部の sha256 が無い（指摘 3） |

## A．小さく流した結果

- コマンド：`CUDA_VISIBLE_DEVICES="" ./.venv/bin/python -W always DAI/exp/dai17_posthoc.py --out-dir DAI/checks/DAI-17/code_run_posthoc_v2/out`（全量）
- 所要：20.6 秒．標準出力は「単位（種 0）60，俳優の発話 99，条件の行 5760」．俳優の発話 99 は発話一覧の上の数（AFEW 17，MELD 82）で，表に入る数（11，62）とは別の数である．警告は 61 行の `open` を閉じない `ResourceWarning` 1 つだけで，値に関係しない．
- 保存済みの値の再現（`check_v2.py`）：`results/dai17_probe/posthoc/` の 9 つの csv（`by_source_bal` 120 × 16，`dependency_count` 40 × 6，`dependency_detail` 120 × 8，`overlap` 24 × 7，`same_actor` 20 × 10，`same_actor_units` 600 × 13，`speakers_top20` 160 × 11，`students_cond_sel` 192 × 8）が，流し直した出力と全セルで一致した（`dependency_count` の `changed_cprime` は空の文字列を NaN として読むと一致の判定が偽になるので，`keep_default_na=False` で読んで一致を確かめた）．NaN と Inf は 0．
- 同じ俳優の対と期待値の数え直し（`check_actor.py`）：上の表のとおり 600 行すべてで一致．
- 行の数：`same_actor_units` 600 = 種 0 の単位 60 × （A・V・T の 3 空間 9 ＋ J 1）．`dependency_detail` 120 = c 4 × c′ 3 × K 2 × 疑似ラベル 5．条件の行 5760 は前回と同じ．
- 単位の csv 120 本のコミットはすべて 48212d35 で，provenance の `units_commit`（先頭の 1 本だけを読む）と同じ．

## A．本番の規模の見積もり

全量で 20.6 秒，CPU だけ．出力は 130 KB．

## B．予想される指摘と，出力から答えられるか

| 表・図 | 予想される指摘 | もとにした項目か例 | (a)／(b)／(c) | (b) なら足すもの | 足す費用 |
|---|---|---|---|---|---|
| 同じ俳優の対 | 期待値は表の全体の割合で出しているが，近い発話は相手のコーパス（俳優の発話がある側）から表の割合と違う割合で引かれる．引いたコーパスの割合で条件づけた期待値と比べよ | 9.3 節 A（基準の値），前回の指摘 2 の続き | (a) | 保存済みの `nn_test_*` と `srcdb` の `corp` から，テストの俳優の発話ごとに「引いた 100 本のうち相手のコーパスの本数 × その俳優の発話の数 ÷ 表の中の相手のコーパスの発話の数」を足す．`check_corpus_adj.py` で出せることを確かめた．表の割合の期待値に対する比は 20 組（c × 枠 × 空間）で 0.94〜2.04 で，1 から離れる組がある（指摘 1） | 数秒 |
| 同じ俳優の対 | 対の数が小さいので，単位の間のばらつきを示せ | 9.3 節 S | (a) | `same_actor_units.csv` に単位ごとの観測と期待値がある | 0 |
| 同じ俳優の対 | 対応表は正しいか | `review_affect_v3.md` b1 | (a) | provenance の `actors` と `n_actor_in_db`，台本の中の数の突き合わせ | 0 |
| ソースごとの当たる割合 | 分母の小さい組はどれか．感情の欠けはあるか | 9.3 節 A | (a) | `by_source_bal.csv` の `n_pairs_source_min`・`n_emotions_min` | 0 |
| 依存の表 | 再現の確かめ（30 単位で選び直すと固定した点に戻るか）を出力に残せ | 9.3 節 I | (a) | 点検の台本の `dependency_detail.csv` の `reproduced_with_30`（偽は 0）．台本の出力には無い（指摘 2） | 0 |

「止める」に当たる行は無い．

## 指摘

### [A（基準の値）] 同じ俳優の期待値を表の全体の割合で出しており，近い発話が相手のコーパスから引かれる割合の違いを含む

- 重さ：記録する
- 該当箇所：`DAI/exp/dai17_posthoc.py:129`
- 問題：`exp100 = float(sum(cnt_db.get(x, 0) for x in ta if x) * it_s.shape[1] / n_db)` は依頼の式どおりで，数え直しとも一致した．ただし表には 2 つのソースのコーパスが入り，俳優の発話は相手のコーパス（AFEW のテストなら MELD）にしか無い．近い発話が相手のコーパスから表の割合より多く（または少なく）引かれると，観測と期待値の比には「同じ人物だから近い」分と「そのコーパスから多く引く」分が混ざる．保存済みの出力から，引いたコーパスの割合で条件づけた期待値を出すと，表の割合の期待値に対する比は 0.94〜2.04 だった（`DAI/checks/DAI-17/code_run_posthoc_v2/check_corpus_adj.py`）．読んだうえで流して確かめた．
- 計算の後で直せるか：直せる（数秒，保存済みの番号だけ）
- 求める対応：`same_actor_units.csv` と `same_actor.csv` に，引いたコーパスの割合で条件づけた期待値の列（例：`exp_random_corpus_100`）を足すか，結果の文で「期待値は表の全体からでたらめに引いたときの数で，コーパスの割合の違いを含む」と書く．同じ人物が近さで上位に来ると書くなら，条件づけた期待値とも比べる．

### [I（検算の欠け）] 依存の表を台本に移したときに，30 単位で選び直して固定した点に戻るかの確かめを落とした

- 重さ：記録する
- 該当箇所：`DAI/exp/dai17_posthoc.py:63-83`（点検の台本 `dependency_table.py` の `rep` と `reproduced_with_30`）
- 問題：台本の `select` が本計算の `select_points` と同じ規則であることは，点検の台本が 30 単位で `frozen_sel.json` を再現した（偽は 0）ことで支えられている．台本の出力にはこの確かめが無く，`dai17_probe.py` の規則が後で変わると，台本は気づかずに違う規則で数える．今回の出力の値は点検の台本と同じなので，今の値には影響しない．
- 計算の後で直せるか：直せる
- 求める対応：30 単位で選び直した点が `frozen_sel.json` と違えば止める 1 行を足すか，`reproduced_with_30` の列を残す．

### [記録] 説明文と provenance の不足

- 重さ：記録する
- 該当箇所：`DAI/exp/dai17_posthoc.py:3-7`，`:147`，`:172-174`，`:61`
- 問題：(1) 冒頭の説明は `by_source_bal.csv` を「both の 4 つの枠」と書いたままで，期待値の列，空間の列，`dependency_detail.csv`・`dependency_count.csv` を書いていない．(2) `same_actor.csv` の `n_test_actor` がソースの組（T）の数だけ重ねた数であることは 147 行の注記にしか無く，csv を読む人には分からない．(3) provenance の入力の sha256 は `students_sel.csv`・`neighbors.csv`・`frozen_sel.json`・発話一覧の 5 つで，実際に読む `units/*.npz`・`units/*.csv`・`srcdb/*.npz`・`students/*.csv` を含まない．`units_commit` は先頭の 1 本だけを読む（今は 120 本が同じコミットなので値は正しい）．(4) 61 行の `open` を閉じない．
- 計算の後で直せるか：直せる
- 求める対応：説明文を出力の 9 つに合わせて書き直す．`n_test_actor` の重なりを結果の文か列名に書く．本計算の `provenance.json` の sha256 を写すか，読んだファイルの sha256 の一覧を足す．

## 目録に無い種類の不備

- 案：「まれな事象の基準の値を『表からでたらめに引く』で作るとき，事象が表の一部（特定のコーパス）にしか無いなら，引いた発話がその部分から来る割合で条件づけた基準も並べる」．今回は表の割合の基準との比が最大で 2 倍ずれた．

## 置き場

- 検算の台本と出力：`DAI/checks/DAI-17/code_run_posthoc_v2/`（`check_v2.py`：保存済みとの一致・依存の表の一致・`srcdb` の並び，`check_actor.py`：対応表と同じ俳優の対と期待値の数え直し，`check_corpus_adj.py`：コーパスの割合で条件づけた期待値．`out/posthoc/` は流し直した出力，`out/` のほかは `results/dai17_probe/` への symlink）
