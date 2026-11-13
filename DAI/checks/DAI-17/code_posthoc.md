# 関門 1：実装と出力（DAI-17 の計算の後に出す量の台本 `DAI/exp/dai17_posthoc.py`，2026-11-14，コミット ee8b329．台本の最終の変更は fa21def，sha256 f665e4c4…b035）

## 判定

直してから通す．台本は保存した出力を設計の定義どおりに読み，選び直した点の結び付けも正しいが，同じ俳優の対応表から Susan Sarandon の MELD の別名 `Jessica_Lockhart`（2 発話）が抜けていて設計の数（MELD 62 発話）と合わず，同じ俳優の対の表には「表からでたらめに引いたときの期待値」が無いので，結果の文の下書きの「引いた対の全体に比べて少ない」という読みを確かめられない．

## 検査の範囲と守ったこと

- 読んだもの：台本の全文，設計 `DAI/docs/実験17_下調べ_設計.md`（全文．とくに 3.1・3.6・3.7・3.9・7 節），本計算の台本 `DAI/exp/dai17_probe.py` の `nn_rows`（302〜341 行），近い発話を引いて保存する箇所（370〜396 行），`select_points`（631〜660 行），集計の段の C（1101〜1168 行），`DAI/exp/dai_data.py` の別名の表（44〜51 行），`DAI/checks/DAI-17/review_affect.md` の A2，`review_affect_v3.md` の b1，`DAI/checks/DAI-1/review_affect_v3.md` の a20，AFEW・MELD の発話一覧．
- GPU は使っていない（`CUDA_VISIBLE_DEVICES=""`．この環境に `nvidia-smi` は無かった）．`results/dai17_probe/` の既存のファイルは書き換えていない．試しに流すときは `DAI/checks/DAI-17/code_run_posthoc/out/` に `units`・`srcdb`・`students`・`summary` への symlink を置き，`posthoc/` だけをそこに書かせた．`results/dai1/eval/` は開いていない．
- この報告には UAR の値，差の符号，選ばれた点を書かない．検算の台本も選ばれた点の名前と UAR を画面に出していない（一致の数だけを出した）．
- 注記：結果の文の下書き（`DAI/docs/実験17_下調べ_結果の文の下書き_2026-11-14.md`）を posthoc の出力の参照箇所を探すために grep し，数値を含む行が画面に出た．その数値はこの報告の判断にも本文にも使っていない（同じ俳優の対の分母の数 237,000 と 3,380,100 は，下の検算で独立に出した値と同じであることだけを確かめた）．

## A．設計と台本の対応表

| 設計の項目 | 台本の箇所 | 合っているか |
|---|---|---|
| 3.1・3.7：`both` の上位 100 本と，`feat`・`logit` の上位 100 本の重なり（テスト．声・顔・言葉） | 65〜71 行（`nn_test_both_{A,V,T}` と `nn_test_{feat,logit}_{A,V,T}` の共通の番号の数 ÷ 100），86 行（c × 枠 × 相手で平均・最小・最大・数） | 合っている．数は各組 15（分割 5 × ソースの組 3）で，3.7 節の「15 個の別々の値」と合う |
| 7 節・`review_affect.md` A2・`review_affect_v3.md` b1：同じ俳優の対の数（`sample_id` の水準の対応表．AFEW 11 発話，MELD 62 発話） | 26〜28 行（`ACTORS`），35〜48 行（`actor_of_sid`），72〜78 行（数え方），87〜89 行（出力） | **合っていない**．MELD の側が 60 発話になる（指摘 1）．数え方そのもの（テストの発話の俳優と，引いた発話の俳優が同じで空でない対）は正しい |
| 3.7：ソースのコーパスごとの感情ごとの当たる割合の平均 | 79〜85 行，90〜91 行 | 定義は合っている（テストの感情 e ごとに，ソース s から引いた対の当たる割合を出して平均）．`both` の 4 つの枠だけで，`feat`・`logit` の 6 つの枠は出さない（指摘 4） |
| 3.9：3 つの条件の量を選び直した点で（条件 2：疑似ラベルの候補の上の UAR − K 本だけの分類器の候補の上の UAR，条件 3：生徒が疑似ラベルの判定を候補の上で再現した割合，生徒の UAR − K 本だけの分類器の UAR） | 93〜104 行 | 合っている．`edge`・`mimic`・`gain_none` の定義は本計算の集計の段（`dai17_probe.py` 1162〜1168 行）と同じで，違いは点の選び方（実験 1 の設定の点 → `students_sel.csv` の点）だけ |
| 3.6・3.9：c の判定に使う点は c を評価のターゲット T にした単位で選んだ点で，その点を仮のターゲットが c の単位に当てる | 98〜99 行（`students_sel.csv` の `c` と単位の `c`（仮のターゲット）で結ぶ） | 合っている．`students_sel.csv` の `c` が「T = c の単位で選んだ」ことを独立に検算した（192 行のすべてで一致）．本計算の集計の段も単位の仮のターゲット c で `sch[(c, K, s, t)]` を引く（1123〜1125 行）ので，同じ結び付けである |
| 3.7：上位 20 本の別々の話者の数と塊の数，でたらめに 20 本引いたときの値 | 105〜106 行（`summary/neighbors.csv` から 2 つの `by` の行を抜く） | 合っている（本計算の `nn_rows` 334〜336 行の値をそのまま写す） |
| 3.7・3.9：近い発話の記述は種に依らない（K 本を使わない） | 59〜60 行（種 0 の単位だけを使う） | 合っている．種 0 と種 1 の 60 組の 600 の鍵（`nn_test_*`）で番号が全部同じだった |
| 3.9：c′ の点を c を除いた 20 単位で選び直したときに変わる組の表 | 台本に無い | 台本の外にある（`DAI/checks/DAI-17/code_run_post/dependency_table.py` とその出力）．指摘 5 |
| DAI の取り決め：出力に設定とコミットを残す | 107〜112 行（`posthoc_provenance.json`） | コミット，dirty，台本の sha256 はある．入力（`units`・`students`・`summary` のファイル）の sha256 は無い（指摘 6） |

## A．小さく流した結果

- コマンド：`CUDA_VISIBLE_DEVICES="" ./.venv/bin/python -W always DAI/exp/dai17_posthoc.py --out-dir DAI/checks/DAI-17/code_run_posthoc/out`（全量．小さく流す必要が無い大きさだった）
- 所要：13.6 秒．警告は 0．標準出力は「単位（種 0）60，俳優の発話 97，条件の行 5760」．
- 出力：`DAI/checks/DAI-17/code_run_posthoc/out/posthoc/`．
- 保存済みの値の再現：`results/dai17_probe/posthoc/` の 6 つの csv（`overlap.csv` 24 行，`same_actor.csv` 16 行，`same_actor_units.csv` 240 行，`by_source_bal.csv` 48 行，`students_cond_sel.csv` 192 行，`speakers_top20.csv` 160 行）と，形と値がすべて一致した．NaN と Inf は 0（`check_outputs.py`）．
- 行の数の検算：条件の行 5760 = 単位 120 × K 2 × 生徒 4 × 疑似ラベル 6 で，結び付けで欠けた行も重なった行も無い．`students_cond_sel.csv` は (c, K) ごとに 24 行（生徒 4 × 疑似ラベル 6）．`overlap.csv` の各行の数は 15．`by_source_bal.csv` の各行の数は 10（分割 5 × そのソースを含むソースの組 2）．
- 結び付けの検算（`check_logic.py`）：`students_sel.csv` の 192 行の点を，生徒の csv から T = c の単位の平均の最大（同点は格子の並びで先）で独立に選び直すと，192 行すべてで一致した．単位の c ではなく T で結ぶと，5760 の組のうち 4380 で点が変わる．したがって結び付けの向きは結果を左右し，台本は設計の向きで結んでいる．
- `by_source_bal` の感情の欠け：480 の（単位，枠，ソース）のすべてで 4 感情がそろう（`n_emotions` は集計で落ちるが，欠けは 0 である）．
- 俳優の対応表の検算（`actor_count.py`）：ソースの表にある発話は AFEW 11（6 人とも設計の数と同じ），MELD 60（Elliott Gould 19，Jason Alexander 14，Sean Penn 11，Gary Oldman 7，Ben Stiller 5，Susan Sarandon 4）．設計の 62 との差の 2 は `Jessica_Lockhart` の `test_dia240_utt1`・`test_dia240_utt3` で，表にあるが対応表に入らない．
- 同じ俳優の対の基準（`actor_baseline.py`→`actor_baseline.out`）：テストに俳優の発話があり，ソースの表にその俳優の発話がある単位は，AFEW のテストで 15 単位のうち 8，MELD のテストで 15 単位のうち 6 である．`same_actor.csv` の `n_pairs` は残りの単位（ソースの表に相手のコーパスが無い単位を含む）の対も足している．

## A．本番の規模の見積もり

全量で 13.6 秒，CPU だけ．GPU は使わない．出力は 1 MB 未満．

## B．予想される指摘と，出力から答えられるか

| 表・図 | 予想される指摘 | もとにした項目か例 | (a)／(b)／(c) | (b) なら足すもの | 足す費用 |
|---|---|---|---|---|---|
| 同じ俳優の対 | 「少ない」と言うなら何と比べたか．表からでたらめに引いたときの期待値と並べよ | `review_affect.md` A2，9.3 節 A（基準の値） | (b) | 単位ごとに，テストの俳優の発話について「表の中の同じ俳優の発話の数 ÷ 表の大きさ × 100」を足した期待値と，対が起こりうる単位だけの分母（`n_pairs_possible`，`n_test_actor_possible`）の列 | 再計算 15 秒，読むだけ．`actor_baseline.py` がその形 |
| 同じ俳優の対 | 特徴だけの近さ（`feat`）では同じ俳優がもっと上位に来るのではないか | 設計 7 節「標準化で特徴の近さの重さが増えたので，同じ俳優…が上位に来やすい」 | (b) | `nn_test_feat_*`・`nn_test_logit_*` でも同じ数を出す（保存済み） | 数秒 |
| 同じ俳優の対 | 6 人の対応表は正しいか．別名を含むか | `review_affect_v3.md` b1，`DAI/checks/DAI-1/review_affect_v3.md` a20 | (b) | `Jessica_Lockhart` を足す（指摘 1） | 0 |
| ソースごとの感情ごとの当たる割合 | 局所の情報と読めるのは `feat` だけ（設計 3.7 節末，`review_affect_v3.md` b4）なので，`feat` の枠でもソースごとに出せ | 設計 3.7 節 137 行（枠を限っていない），`neighbors_by_source.csv` は 10 枠 | (b) | 79〜85 行の枠の輪を `both` の 4 つに加えて `feat`・`logit` の 6 つにも回す | 数秒 |
| ソースごとの感情ごとの当たる割合 | ソース × 感情の分母（対の数）が小さい組はどれか | 9.3 節 A | (a) | 単位の npz の `nn_test_both_*` と `srcdb` の `corp` から数えられる．ただし出力には残らないので，`n_emotions` と対の数の列を足すと読みやすい | 0 |
| 重なり | `both` の上位が特徴の近さに寄るのは枠ごとにどうか．順位 20 本でも同じか | 設計 3.1 節，`probe_v4.md` | (a) | `nn_test_*` は保存済みで，上位 20 本の重なりも同じ手順で出せる | 0 |
| 選び直した点の条件の量 | 選び直した点の値が実験 1 の設定の点の値とどれだけ違うか | 設計 3.9 節 | (a) | `summary/students_conditions.csv`（設定の点）と `posthoc/students_cond_sel.csv`（選び直した点）を (c, K, 生徒, 疑似ラベル) で結ぶ | 0 |
| 選び直した点の条件の量 | c′ の点を c を除いて選び直すと条件の量は変わるか | 設計 3.9 節（選択を通した依存） | (a) | 生徒の csv に格子の全点がある．疑似ラベルの版そのものは `frozen_sel.json` で固定され，選び直した疑似ラベルの生徒は計算していないので，疑似ラベルの版を変えた場合の条件の量は (c)（設計 3.9 節は「変わる組を数える」までと書いている） | 0 |
| 話者の数 | ソースの組ごとに基準が違うので，組ごとに並べよ | `review_affect.md` a2 | (a) | `summary/neighbors.csv` は c ごとの平均で，同じ単位の観測とでたらめの値を並べる．組ごとは単位の npz と `srcdb` の `spk`・`spkid` から出せる | 0 |
| 依存の表 | 3.9 節の表を作った台本はどれか．結果と同じ取り決め（provenance）で残っているか | DAI の取り決め（provenance） | (a)（出力はある） | 指摘 5 | 0 |

(b) の行はどれも保存済みの出力から数秒で出せるので，学び直しは要らない．「止める」に当たる行は無い．

## 指摘

### [I（対応表の誤り）] 同じ俳優の対応表から Susan Sarandon の MELD の別名 `Jessica_Lockhart` が抜けている

- 重さ：直してから進む
- 該当箇所：`DAI/exp/dai17_posthoc.py:27`，`:43`
- 問題：台本は `"Susan Sarandon": ("AFEW_Susan_Sarandon", "Cecilia", [(7, 15)])` と書き，43 行で発話一覧の生の `speaker_id == "Cecilia"` だけを引く．`Jessica_Lockhart` は `Cecilia` が劇中で演じる役（同じ俳優）で，`DAI/exp/dai_data.py:49` の `MELD_ALIAS` は `"Jessica_Lockhart": "Cecilia"` とまとめている．`DAI/checks/DAI-1/review_affect_v3.md` の a20 の MELD 48 発話（設計の 62 = 48 + Earl の 14）は `Jessica_Lockhart` の 2 発話を含む．流して数えると，ソースの表にある MELD の俳優の発話は 60 で，`test_dia240_utt1`・`test_dia240_utt3` が入らない．この 2 発話を足しても，この保存の上では同じ俳優の対の数（`n_pairs_same`）は変わらなかったが，MELD のテストの俳優の発話の数（`n_test_actor`）とでたらめの期待値は変わる（`actor_baseline.out`）．結果の文が設計の「MELD 62 発話」の表で数えたと書くなら，今の出力はその表で数えていない．
- 計算の後で直せるか：直せる（15 秒）
- 求める対応：27 行の `meld_role` を役名の組（例：`("Cecilia", "Jessica_Lockhart")`）にできる形にするか，43 行の前に `m.speaker_id.replace(DD.MELD_ALIAS)` を当てる（後者は `Helena` → `Mr._Bing` なども当たるが，6 人の役名には影響しない）．`posthoc_provenance.json` の `n_actor_sids`（今は発話一覧の全行の 97）を，ソースの表にある AFEW 11・MELD 62 の数にして，設計の数との一致を台本の中で確かめる．

### [A（基準の値）] 同じ俳優の対の表に，でたらめに引いたときの期待値と，対が起こりうる単位だけの分母が無い

- 重さ：直してから進む
- 該当箇所：`DAI/exp/dai17_posthoc.py:77-78`，`:89`
- 問題：`n_pairs=int(it.size)` はテストの全発話 × 100 で，89 行は c ごとに 15 単位を足す．15 単位には，ソースの表に相手のコーパス（AFEW のテストなら MELD）が無い単位が入り，そこでは同じ俳優の対は起こりえない．対が起こりうる単位は AFEW のテストで 8，MELD のテストで 6 だった．また，俳優の発話は表の 0.1〜1.4% しかないので，「引いた対の全体」に比べて対が少ないのは，近さに関係なく当然である．同じ俳優が近さで上位に来るか（設計 7 節の問い）を読むには，テストの俳優の発話ごとに「表の中の同じ俳優の発話の割合 × 100」を足した期待値と比べる必要がある．この期待値を `actor_baseline.py` で出すと，観測した対の数と期待値の比は枠とコーパスで 1.5〜8 倍の幅にあった（対の数そのものは 2〜19 で小さい）．結果の文の下書きの「引いた対の全体に比べて少ない」は，この比べ方を経ていない．
- 計算の後で直せるか：直せる（保存済みの番号から数秒）
- 求める対応：`same_actor_units.csv` に `possible`（その単位の表にテストの俳優の発話があるか），`exp_random`（でたらめに 100 本，20 本引いたときの期待値）の列を足し，`same_actor.csv` で `possible` の単位だけの分母と期待値を並べる．`feat`・`logit` の枠でも同じ数を出す（`nn_test_feat_*`・`nn_test_logit_*` は保存済み）．結果の文は「期待値に比べて」の形で書く．

### [R（範囲の不足）] ソースごとの感情ごとの当たる割合を `both` の枠だけで出している

- 重さ：記録する
- 該当箇所：`DAI/exp/dai17_posthoc.py:65-67`，`:79-85`
- 問題：設計 3.7 節 137 行は枠を限らず「ソースのコーパスごとの当たる割合は，感情ごとの当たる割合の平均でも並べる」と書き，対になる `summary/neighbors_by_source.csv` は 10 枠（`both` 4，`feat` 3，`logit` 3）ある．台本は 67 行で `nn_test_both_{slot}` だけを読む．設計 3.7 節末と `review_affect_v3.md` の b4 は，近傍の局所の情報と読めるのを `feat` の値に限るので，ソースごとの比べを局所の情報として読む表は `feat` の枠である．
- 計算の後で直せるか：直せる（数秒）
- 求める対応：`space` の輪（`both`・`feat`・`logit`）を足し，出力に `space` の列と，(単位，枠，ソース) ごとの対の数・`n_emotions` を残す．あわせて，ソース × 感情の当たる割合を「感情 e のテストの発話から引いた対のうちソース s の対」をまとめて数える定義（発話ごとの割合を平均する定義ではない）であることを設計か結果の文に 1 行書く．

### [R（範囲の不足）] 設計 3.9 節の「c′ の点を c を除いた 20 単位で選び直したときに変わる組」の表を作る台本が `DAI/exp/` に無い

- 重さ：記録する
- 該当箇所：設計 3.9 節 164 行．台本は `DAI/checks/DAI-17/code_run_post/dependency_table.py`（gate-code の点検で作った検算の台本）
- 問題：結果の文の下書きはこの表を `DAI/checks/DAI-17/code_run_post/dependency_count.csv` から引いている．検査役の置き場の台本は `results/dai17_probe/` に出力を残さず，provenance も無い．依頼の「計算の後に出す量」の一覧にもこの表は入っていない．読むだけの指摘で，`dependency_table.py` の中身は今回検査していない．
- 計算の後で直せるか：直せる
- 求める対応：`dependency_table.py` の計算を `dai17_posthoc.py` に移すか，結果の文に「この表は gate-code の点検の台本で作った」と出どころを書く．

### [記録] `posthoc_provenance.json` に入力の sha256 が無く，`same_actor.csv` の `n_test_actor` が T の数だけ重なる

- 重さ：記録する
- 該当箇所：`DAI/exp/dai17_posthoc.py:109-112`，`:89`
- 問題：provenance は台本の sha256 とコミットだけを書き，読んだ `summary/students_sel.csv`・`summary/neighbors.csv`・`units/*.npz` の版を記録しない（DAI の取り決めは入力の sha256 も残す）．89 行は同じテストの発話を 3 つのソースの組（T）で 3 回数える．
- 計算の後で直せるか：直せる
- 求める対応：`students_sel.csv`・`neighbors.csv`・`frozen_sel.json` の sha256 と本計算の `provenance.json` のコミットを書き足す．`n_test_actor` の列名か説明に「T の数だけ重ねた数」と書く．

### 確かめて問題が無かった点

- 選び直した点の結び付け：上の対応表と検算のとおり．`students_sel.csv` の `c` は評価のターゲット（T = c の単位で選ぶ）で，台本はその点を仮のターゲットが c の単位に当てる．本計算の集計の段の `_sel` の判定と同じ結び付けである．
- 近い発話の記述を種 0 の単位だけで出すこと：本計算の台本は近い発話を K 本に依らずに引き（`dai17_probe.py` 370〜396 行．候補とテストの特徴とソースの分類器の点だけを使う），保存の上でも種 0 と種 1 の 60 組 600 鍵の番号が全部同じだった．テストの発話の並びとソースの組も同じ．種 0 だけで出してよい．
- `top20` の切り出し（76〜78 行の `same[:, :20]`）：保存の番号は近い順（`dai17_probe.py:394` の注記，`torch.topk` の既定の並び）である．
- 条件の量の行に欠けと重複は無い（5760 行）．

## 目録に無い種類の不備

- 案：「コーパスをまたぐ同じ人物の対応表は，分割の表で使う別名の表（`dai_data.MELD_ALIAS`・`AFEW_ALIAS`）を通した名前で作り，表の発話の数を設計の数と台本の中で突き合わせる」．生の発話一覧の名前で引くと，分割では 1 人にまとめた別名が漏れる（今回の `Jessica_Lockhart`）．
- 案：「まれな事象の数（同じ人物の対など）を報告するときは，全体の対の数ではなく，でたらめに引いたときの期待値と，事象が起こりうる単位だけの分母を並べる」．

## 置き場

- 検算の台本と出力：`DAI/checks/DAI-17/code_run_posthoc/`（`actor_count.py`，`check_outputs.py`，`check_logic.py`，`actor_baseline.py`→`actor_baseline.out`，`out/posthoc/`．`out/` の `units`・`srcdb`・`students`・`summary` は `results/dai17_probe/` への symlink）
