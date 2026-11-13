# 関門 1：実装と出力・差分の再検査（DAI-16・17 の K = 10 の追補 第 2 版，2026-11-14，コミット b1b4d9e）

## 判定

通す．前回の 4 体の必須の指摘 2 つ（`dai16_posthoc.py` が K = 10 で止まる，起動の命令が固定されていない）は b1b4d9e で直っており，`dai16_posthoc.py` は GPU なしで K = 5・25 の保存の出力を数値の差 0 で再現し，K = 10 の置き場では K = 10 だけを回した．起動の台本 `DAI/run/dai_k10.sh` は全部の段に `--ks 10 --out-dir results/dai1x_probe_k10` を渡し，段の失敗と既存の置き場の書き換えで `logs/dai_k10.failed` を作ることを，python の代わりの台本で 7 通り流して確かめた．

起動の前に満たすこと（どれも計算のやり直しにはつながらない．下の指摘はすべて「記録する」）：

1. 作業木にある追補の未コミットの直し（題の「第 2 版」，20〜21 行の分け，29 行の断り，53 行の記録）をコミットしてから起動する．コミットしないと，`dai16_probe.py` の最初の段が `committed_clean(ADDENDUM)` で止まる（読んだだけ．`dai16_probe.py:1141–1142`）．
2. 起動のコミットで `git diff b1b4d9e <起動のコミット> -- DAI/exp DAI/run` が空であることを確かめる．空でなければ差分を再検査に回す．
3. 起動の命令は `SHA16=62863fa8741277762357e9cc9401bde131c4059e3b1f89ebd3d38daffd390e34 SHA17=1a99e29b40274867a4377263b6a1e31128b1aa1fcbc16cdcc0edcb542f1ee558 bash DAI/run/dai_k10.sh` とする（値は b1b4d9e の 2 本の sha256．前回の検査で流した edc7edb の版と同じ）．

検査の範囲：`git diff edc7edb b1b4d9e`（`DAI/exp/dai16_posthoc.py`，`DAI/run/dai_k10.sh`，追補），検査の途中で見つけた追補の未コミットの差分（`git diff b1b4d9e -- DAI/docs/`）．GPU は使っていない（posthoc は `CUDA_VISIBLE_DEVICES=""` で流した）．`results/` には書いていない（検査の後に `results/dai16_probe`・`results/dai17_probe` の下で検査の台本より新しいファイルは 0 個，`results/dai1x_probe_k10` は存在しない）．`results/dai1/eval/` は開いていない．UAR・判定の語は報告に書いていない．保存の単位の写しは，突き合わせの後に消した．

## A．前回の必須の指摘への対応

| 前回の指摘（出した役） | b1b4d9e での対応 | 確かめ方 | 結果 |
|---|---|---|---|
| `dai16_posthoc.py` が `P16.KS`（5，25）で回し，K = 10 の出力で `KeyError: 'D_M_K_K5'`（code.md 指摘 1，probe.md の I1，review の A2・M1） | `dai16_posthoc.py:52–53` で `KS_RUN = tuple(json.load(open(out / "provenance.json"))["KS"])`，`:67`・`:114` を `for K in KS_RUN:` に替えた．`P16.KS` を読む箇所はモジュールの既定値の 38 行だけになった（grep） | 下の「小さく流した結果」1〜3 | 直った |
| K = 10 の回を走らせる命令が固定されておらず，`--ks` を付け忘れると既存の置き場に書く（probe.md の I8） | `DAI/run/dai_k10.sh` をコミットした．6 段と posthoc 2 本のすべてに `--ks 10` と `--out-dir results/dai1x_probe_k10` を渡す（posthoc は `--out-dir` だけ） | `bash -n`，python の代わりの台本で引数を記録（下の 4） | 直った．付け忘れは 0 |
| 既存の置き場の sha256 を前後で照合する（probe.md の I8 の求め） | `dai_k10.sh:16–17,28–29` の `snap()`（`find results/dai16_probe results/dai17_probe -type f -print0 \| sort -z \| xargs -0 sha256sum \| sha256sum`） | 偽の根で，途中の段に既存の置き場のファイルを書き換えさせた | 書き換えを見つけて `failed` を作った．照合はファイルの名前と中身を含む（足す・消す・名前を替えるも見つかる）．照合の範囲は 2 つの置き場だけで，入力の `results/dai1` は含まない（入力は追補 2 節の `provenance.json` の照合で見る） |

`dai16_posthoc.py` は前回の試走で動いた版と同じか：前回の試走では作業木の版のバイトの sha256 を記録していないので，バイトが同じかは確かめられない．前回の報告が引用した直し（`KS_RUN` の行，64・111 行の置き換え）は b1b4d9e の差分と一致する．そのうえで，今回は b1b4d9e の版（sha256 `243a96a9b5b393da5f1a5c3f72a35d0b5881982da69c9b6687b62cdf33dafd1b`）そのものを流し直したので，前回の試走の版と同じかどうかに判定は依らない．`dai16_probe.py`・`dai17_probe.py`・`dai17_posthoc.py` は edc7edb と b1b4d9e で同じである（sha256 `62863fa8…`，`1a99e29b…`，`4301835e…`．作業木も同じ）．

## A．設計と台本の対応表（差分の範囲）

| 設計（追補 第 2 版）の項目 | 台本の箇所 | 合っているか |
|---|---|---|
| 1 節：`dai16_posthoc.py` は `provenance.json` の K を読む | `dai16_posthoc.py:52–53,67,114` | 合う |
| 1 節：`dai17_posthoc.py` の依存の表は `provenance.json` の K を読む | `dai17_posthoc.py:76`（前回から変わらない） | 合う |
| 1 節 18 行：`dai16_probe.py --stage units` → `summary`，`--ks 10 --out-dir results/dai16_probe_k10` | `dai_k10.sh:18–21` | 合う（記録した引数は `--stage units --ks 10 --out-dir results/dai16_probe_k10`，`--stage summary ...` の順） |
| 1 節 19 行：`dai17_probe.py` の 4 段を順に | `dai_k10.sh:22–25` | 合う（units，reweight，students，summary の順） |
| 1 節 20 行：計算の後に posthoc 2 本 | `dai_k10.sh:26–27` | 合う．どちらも `CUDA_VISIBLE_DEVICES=""` で流す |
| 3 節：DAI-16 と DAI-17 は続けて流し，同時には流さない | `dai_k10.sh` は 1 本の bash で順に流す | 合う |
| 起動の台本の頭の注記：終わると `logs/dai_k10.done`，失敗は `logs/dai_k10.failed` | `dai_k10.sh:7–8,30` | 段の失敗・sha256 の違い・既存の置き場の書き換え・照合の `find` の失敗では合う．`SHA16` を渡し忘れたときと，外から止められたときは印が出ない（指摘 1） |
| 起動の台本の頭の注記：検査を通した台本の sha256 を `SHA16`・`SHA17` で渡す | `dai_k10.sh:9–10` | 合う．ただし追補は渡す値を書いていない（指摘 3） |
| 追補の未コミットの変更で止まる | `dai16_probe.py:1141–1142`，`dai17_probe.py:1280–1281` | 合う（読んだだけ）．いまの作業木では追補が未コミットなので，起動すると最初の段で止まる（判定の 1） |

## A．小さく流した結果

置き場は `DAI/checks/DAI-K10/code_run_v2/`．GPU は使っていない．

1. **K = 5・25 の置き場で，新旧の `dai16_posthoc.py` の出力が同じ．**`make_fixtures.py` で `results/dai16_probe/units` の先頭 3 単位（`AFEW_IEMOCAP_f0_r0`・`_f0_r1`・`_f1_r0`）と `summary/mixsel.csv`・`provenance.json`（KS は [5, 25]）を写した置き場を作り，b1b4d9e の版と edc7edb の版（`dai16_posthoc_edc7edb.py`）を同じ置き場の写しで流した（各約 3〜5 秒）．5 つの表（`additive_units`・`additive_corr`・`additive_levels`・`additive_effects`・`additive_clip`）で，行数が同じ，数の列の差の最大 0，NaN の位置の不一致 0 だった（`compare.out`．`additive_units` の「文字の列の不一致 24」は，pandas の文字の型で NaN どうしを比べたときに不一致と数える検算の台本の側の数え方で，NaN の位置は一致している）．
2. **保存の posthoc の出力を再現した．**1 の新しい版の `additive_units.csv` を，保存の `results/dai16_probe/posthoc/additive_units.csv`（全 120 単位で流したもの）の同じ 3 単位の行と比べ，96 行，数の列の差の最大 0，NaN の位置の不一致 0 だった．保存の置き場（KS は [5, 25]）では，直した版は元の動きと同じ K を回す．
3. **K = 10 の置き場では K = 10 だけを回す．**同じ 3 単位の npz の鍵 `*_K5*` を `*_K10*` に名前だけ替え，`*_K25*` を落とし，`provenance.json` の KS を [10]，`mixsel.csv` の K = 5 の行を K = 10 にした置き場を作った．b1b4d9e の版は止まらずに終わり（`B_new.log`），5 つの表の K はすべて 10 だけだった．`mixsel` 以外の 39 行は，1 の K = 5 の行と数の列の差の最大 0 で一致した（K の数え方以外は同じ計算であることの検算）．`mixsel` の 9 行は `mix_rel(..., 4 * K, ...)` が K に依るので比べず，NaN が 0 であることだけを見た．edc7edb の版は同じ置き場で `KeyError: 'D_M_K_K5 is not a file in the archive'` で止まった（`B_old.log`）．保存の判定との一致が NaN の行は 36 行中 12 行で，保存に無い足し算だけの対照（`M_F6add_*`）の行であり，K = 5・25 の置き場と同じ数え方である．
4. **起動の台本．**`bash -n DAI/run/dai_k10.sh` は通った．`runtest/runcase.sh` で，`cd /home/snics/MER-DA` を偽の根に，`PY=.venv/bin/python` を引数を記録する代わりの台本に替えた写し（ほかの行は同じ．`diff` で確かめた）を流した．

| 場合 | 終わりの値 | `done` | `failed` | 呼ばれた数 |
|---|---|---|---|---|
| すべて成功 | 0 | 有 | 無 | 8（引数は上の表のとおり．`--ks 10` の付け忘れ 0） |
| DAI-17 の students の段で失敗 | 3 | 無 | 有 | 5（後の段は流れない） |
| `dai16_posthoc.py` で失敗 | 3 | 無 | 有 | 7 |
| reweight の段で既存の置き場のファイルを書き換え | 1 | 無 | 有 | 8（最後の照合で見つけた） |
| `SHA16` が違う | 1 | 無 | 有 | 0 |
| `SHA16` を渡さない | 1 | 無 | **無** | 0 |
| `results/dai17_probe` が無い（照合の `find` が失敗） | 1 | 無 | 有 | 0 |
| 1 段目の途中で bash に SIGTERM | 143 | 無 | **無** | 1 |

## A．本番の規模の見積もり

前回（`code.md`）の見積もり（合わせて約 3〜4 時間，GPU の最大は試走で 5345 MB で上限の 6 割の中）から変わらない．`dai16_posthoc.py` は CPU で 3 単位 3〜5 秒なので，120 単位で数分と見込む．既存の置き場の sha256（約 3.9 GB）を前後で 2 回とるのに数分かかる．

## B．予想される指摘と，出力から答えられるか

差分は計算の範囲と出力を変えないので，前回の B の表（`code.md`）がそのまま当てはまる．止めるべき (b) は無い．差分に関わる行だけを足す．

| 表・図 | 予想される指摘 | もとにした項目か例 | (a)／(b)／(c) | (b) なら足すもの | 足す費用 |
|---|---|---|---|---|---|
| 全体 | 各段が検査を通した台本で流れたか | I9 | (a)．各段の `provenance_log.jsonl` の行（DAI-16 は `scripts["dai16_probe.py"]`，DAI-17 は `this_script_sha256` と `scripts`）と `posthoc/posthoc_provenance.json` の `script_sha256` を，判定の 3 の値と `dai16_posthoc.py` の `243a96a9…`，`dai17_posthoc.py` の `4301835e…` に照らす．`commit` と `dirty` も各行にある | — | — |
| 全体 | 既存の K = 5・25 の置き場が書き換わっていないことを示せ | probe.md の I8 | (a)．起動の台本の最後の照合（`logs/dai_k10.done` があれば一致した）．ただし前後の sha256 の値はログに残らない（指摘 4） | — | — |
| DAI-16 の足し算だけの対照 | K = 10 でも K = 5・25 と同じ手順で作ったか | 追補 1 節 | (a)．`posthoc/additive_*.csv` の K 列と，`posthoc_provenance.json` の `script_sha256`．保存の判定との一致（`agree_saved_min`）も残る | — | — |

## 指摘

### [I8] `SHA16` を渡し忘れたときと，外から止められたときに `logs/dai_k10.failed` が出ない
- 重さ：記録する
- 該当箇所：`DAI/run/dai_k10.sh:7`（`trap 'touch logs/dai_k10.failed' ERR`），`:9`（`"${SHA16:?}"`）
- 問題：`${SHA16:?}` の展開の失敗はコマンドの失敗ではないので ERR の trap が働かず，bash は印を作らずに終わる（流して確かめた．終わりの値 1，`failed` 無し）．tmux のセッションが落ちたときや SIGTERM で止めたときも印が出ない（流して確かめた．終わりの値 143，`failed` 無し）．終わりを `done` か `failed` のファイルで待つ運用（memory の「区切りはファイルで待つ」）では，待つ側が終わりに気づかない．誤った数値は出さない．
- 計算の後で直せるか：直せる（印が無いだけで，出力は壊れない）
- 求める対応：起動の前に，ERR の trap を `trap '[ -f logs/dai_k10.done ] || touch logs/dai_k10.failed' EXIT` に替えると，展開の失敗・シグナル・どの段の失敗でも印が出る．替えない場合は，待つ側が `done`・`failed` に加えて bash の処理が生きているかも見る．台本を替えたら差分をもう一度検査に回す．

### [I9] 台本の sha256 の照合は起動のときの 1 回だけで，posthoc の 2 本と起動の台本そのものを含まない
- 重さ：記録する
- 該当箇所：`DAI/run/dai_k10.sh:9–10`，`DAI/exp/dai1.py:70–75`（`git_commit` が見る場所に `DAI/run` が無い）
- 問題：計算は約 3〜4 時間続く．その間に `DAI/exp` の変更がコミットされると，各段の未コミットの確かめ（`dai16_probe.py:1145`，`dai17_probe.py:1284`）は通り，後の段は新しい台本で流れる．`SHA16`・`SHA17` は最初の 1 回しか照合しない．posthoc の 2 本は照合しない．bash は台本を少しずつ読むので，計算中に `dai_k10.sh` を編集すると後の段が変わる（memory の「実行中の台本を編集しない」）．各段の `provenance_log.jsonl` と `posthoc_provenance.json` に台本の sha256 とコミットが残るので，後から照合はできる．
- 計算の後で直せるか：直せる（記録から照合できる．違っていたら該当の段を流し直す）
- 求める対応：計算中に `DAI/exp`・`DAI/run` をコミットしない，`dai_k10.sh` を編集しない．計算の後の点検で，上の B の表の 1 行目の照合をする．

### [I9] コミット b1b4d9e の追補は題が「第 1 版」で，20 行に 2 つの文がつながっていた．直しは作業木にあり未コミット
- 重さ：記録する
- 該当箇所：b1b4d9e の `DAI/docs/実験16・17_下調べ_K10_追補.md:1`，`:20`．作業木の差分（`git diff b1b4d9e -- DAI/docs/`）
- 問題：b1b4d9e の 20 行は「`dai17_posthoc.py --out-dir results/dai17_probe_k10``provenance.json` には追補の sha256 が残らないので…」と，命令と次の項目が 1 行につながっていた．作業木はこれを 20・21 行に分け，題を「第 2 版」にし，29 行に「K どうしの差の区間は出していない」を書く断りを足した（`review_affect_v2.md` の b1〜b3）．どれも台本の動きを変えない．作業木の追補と台本に食い違いは無い．追補が未コミットのままだと，`committed_clean(ADDENDUM)` で最初の段が止まり，`failed` が出る（読んだだけ）．
- 計算の後で直せるか：直せる
- 求める対応：作業木の追補をコミットしてから起動する（判定の 1・2）．

### [I9] 追補に起動の命令の全体（`SHA16`・`SHA17` の値）が書かれておらず，前後の sha256 の値がログに残らない
- 重さ：記録する
- 該当箇所：追補 1 節 17〜20 行，`DAI/run/dai_k10.sh:17,28`
- 問題：追補は段ごとの引数を書くが，起動の台本に渡す環境の変数 `SHA16`・`SHA17` とその値を書いていない．起動する人がその場の sha256 を写すと，照合の意味が無くなる．また `BEFORE`・`AFTER` の値は echo されないので，一致したことは `done` があることからしか分からない．
- 計算の後で直せるか：直せる
- 求める対応：起動の命令を判定の 3 のとおりに書いて流し，結果の文に記録する．前後の値を残したい場合は `echo "snap $BEFORE"` を足す（足したら差分を検査に回す）．

前回から残る記録（`code.md`）：手で 1 段だけ流し直すときも `--ks 10` を付ける．`tee -a` はログに追記するので，流し直すとログに前の回の行が残る．既存の置き場の照合は書き換えを見つけるだけで，書き換えを防がない（防ぐなら起動の前に `chmod -R a-w results/dai16_probe results/dai17_probe`）．

当てた項目：I1，I8，I9，I10，R1．

## 目録に無い種類の不備

**起動の台本の終わりの印が，ERR の trap だけに頼ると出ない場合がある．**bash の ERR の trap は，変数の展開の失敗（`${X:?}`）とシグナルでの終わりでは働かない．`docs/methods.md` 9.3 節の I に足す項目の案：「終わりを印のファイルで待つ起動の台本では，失敗の印を EXIT の trap で作るか（`[ -f done ] || touch failed`），引数の欠け・段の失敗・シグナルの 3 通りで印が出ることを，計算の命令を代わりの台本に替えて流して確かめる」．
