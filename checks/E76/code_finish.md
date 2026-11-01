# 関門 1：実装と出力（E76 の続き `exp/relearn_e76_finish.py`，2026-11-02，コミット 40e58f8（検査の途中で 23b59fe に進んだ．差は `checks/E81/`・`docs/` だけ）＋未コミットの台本 sha256 208e3bbd767cb94d）

検査の対象は `exp/relearn_e76_finish.py`（81 行，未コミット）と，それが呼ぶ `exp/relearn_e76.py`（a14d436 から変わっていない）である．E76 が読み込む 10 本のファイル（`results/e76/audio_hubert/MELD.json` の `code_files`）の中身から今の作業木でハッシュを作り直すと，4 ターゲットの json の `code_hash` 9891f0668848 と一致した．a14d436 から HEAD までの 43 コミットに，この 10 本を変えたものは無い．

## 判定

直してから通す．差し替えた判定は本体の main の中の呼び出しに効き，計算・保存・照合の表・json は止める判定を除いて E76 と同じだった（試しで ② と h2 の係数が E76 の MELD の同じソースの係数とビット単位で一致した）．ただし，コードのハッシュが作りから違うので，再開の照合は「残りの 3 ターゲットを既存の 4 ターゲットと同じコードと設定で計算したか」をもう確かめていない．この照合を台本に足してから本番を流す（指摘 1）．台本はコミットしないと本番のコマンドで止まる（指摘 2）．

## A．設計と台本の対応表

| 設計の項目（E76 の 10 節，依頼の経緯） | 台本の箇所 | 合っているか |
|---|---|---|
| `relearn_e76.verdict` だけを差し替え，`relearn_e76.main` をそのまま呼ぶ | `relearn_e76_finish.py:30`，`:76-77` | 合う．`R.main.__globals__["verdict"] is verdict_finish` が真（main は `verdict` をモジュールの名前空間から呼ぶ．`relearn_e76.py:849`） |
| ①・②・③・h2〜h8 の基準は E76 のまま | `:35`（h9・h10 を除いた summ を元の `verdict` に渡す） | 合う．MELD の check と人工の summ で確かめた（下の「判定の差し替えの検算」） |
| h9・h10 は大きさの基準（tol の 2.5 倍以下）だけで止める | `:36-39` | 合う．元の `relearn_e76.py:723` の大きさの条件と同じ式 |
| h9・h10 の数の基準は記録だけ | 各 json の `check.h9/h10.n_over_tol`・`n_ref`（`relearn_e76.py:682`，`:854`） | 記録は残る．ただし `check_failed` は空になり，数の基準を外したことを json だけからは読めない（指摘 4） |
| 範囲は audio_hubert の AFEW・RAVDESS・CREMA-D | `:27-28`，`:57-60` | 普通の書き方では合う．`--targets AFEW --targets=MELD` のような書き方で抜けられる（指摘 3） |
| MELD の出力は作り直さない | 範囲の制限と，main の再開の照合（`relearn_e76.py:797-808`） | 合う．MELD を指せば別のコードのハッシュで止まる |
| 規則の替え方を `<out>/check_rule_finish.json` に書く | `:63-75` | 書く．ただし main が止まる前に書くので，計算しなかった実行も残る（指摘 2） |
| 反復 0〜9，K 6 つ，手 2〜11，C 4 つ，`--merged-all-c` なし（E76 と同じ設定） | 本体の既定値（`relearn_e76.py:735-737`） | 本番のコマンドは既定値を使うので，E76 の本番（argv は `--gpu-fraction 0.15` だけ）と同じ設定になる．台本はこれを確かめていない（指摘 1） |

**組の数え上げ（L4）**：E76 の `provenance.json` の plan では audio_hubert は 7 ターゲットで，`results/e76/audio_hubert/` には 4 ターゲット（IEMOCAP-impro，IEMOCAP-script，MOSEI，MELD）がある．本番のコマンドは残りの 3 つを計算する．ほかの 7 種類は plan のターゲットがすべてそろっている（audio_wavlm 7，video 6，text_roberta 4，AV 6，AT 4，VT 4，AVT 4）．

**判定の差し替えの検算**（CPU，読み込んで関数だけを呼んだ）：
- MELD の json の `check` に元の判定を当てると h9 で 1 件落ち，差し替えた判定では 0 件
- 人工の summ：h9 の差 ÷ tol を 2.6 にすると落ちる．h10 の超過を 50 件にしても落ちない．h3 の超過を 2/180 にすると落ちる．merged の差 ÷ tol を 2.6 にすると落ちる．src_only の C の不一致 1 件で落ちる
- `results/e76/*/*.json` の 42 個すべてに 2 つの判定を当てると，結果が違うのは audio_hubert の MELD だけだった

## A．小さく流した結果

すべて scratchpad（`/tmp/claude-1000/-home-snics-MER-DA/4cd03bf4-8db6-4e9a-a1e4-1871eadc639a/scratchpad/`）に書かせた．`results/` には書いていない（`results/e76` の下に `logs/e76.log` より新しいファイルは無い）．UAR の値は画面に出していない．

| 試し | コマンド（`./.venv/bin/python exp/relearn_e76_finish.py` に続けて） | 所要 | 結果 |
|---|---|---|---|
| 1 | `--kinds audio_hubert --targets AFEW --reps 0 --out <scratch>/e76_finish_trial --allow-dirty`（GPU は空き．このときは既定の上限 0.6） | 235 秒（① 10，② 89，③ 18，手 116） | 終了コード 0，警告 0，GPU の最大 1.31 GiB，npz 5.2 MiB |
| 2 | 試し 1 に `--gpu-fraction 0.2` を足し，出力を `e76_finish_trial_f02` に | 274 秒 | 終了コード 0．試し 1 と npz の 241 個の配列がすべてビット単位で同じ，csv が同じ（288 行 × 28 列），json の `settings`・`check`・`code_hash` が同じ．GPU の最大も同じ 1.31 GiB |
| 3 | `--allow-dirty` なしで `--out <scratch>/e76_dirty_probe` | 数秒 | 「読み込むコードにコミットしていない変更がある：?? exp/relearn_e76_finish.py」で終了コード 1．`check_rule_finish.json` には 1 件書かれた |
| 4 | `--targets AFEW MELD` | 数秒 | 台本の制限で止まる |
| 5 | `--targets AFEW --targets=MELD --reps 0 --allow-dirty`（既定の上限 0.6 のまま流れたので，約 2 分で止めた） | — | 台本の制限を通り，main は MELD を計算し始めた（`provenance.json` の plan は `{'audio_hubert': ['MELD']}`，`check_rule_finish.json` の targets は `['AFEW']`）．npz は書かれていない |

試し 1 の照合（差 ÷ tol の最大，tol を超えた条件の数）：tgt_only・src_only・merged・h2・h4・h5・h7 は 3.9e-15 で 0 件（C と規則の不一致 0）．h3・h6・h8 は 0.40 で 0 件，h9 は 0.50 で 0/18，h10 は 0.50 で 0/90．h11 は照合の相手が無い（E76 の設計どおり）．収束しない行 0．係数から作り直した点数との差は最大 1.9e-6（argmax の一致 1.0）．前の点数との差（I14）は ① 9.5e-7，② 0.12（argmax の一致 1.0），③ 0.80（argmax の一致 0.994）で，E76 の本番の MELD・MOSEI と同じ桁である．

**出力の形の突き合わせ**（`checks/E76/code_run/finish_keys.py`）：
- npz の鍵をソースの名前・反復・版で正規化すると，試しの 56 種類が既存の audio_hubert の MELD と audio_wavlm の AFEW の 56 種類と一致した．反復 0 の鍵の形は audio_wavlm の AFEW とすべて同じ
- json の鍵，`check` の鍵，csv の列が既存の MELD と同じ．`settings` の違いは `reps` だけ（試しは反復 0）．`code_files` の違いは `exp/relearn_e76_finish.py` が増えたことだけ
- csv の重複した行 0，`uar` の NaN 0，Inf 0．NaN を含む配列は `t_cv_r0` だけで，K=1 の行（規則 fixed で交差検証をしない）．既存の audio_wavlm の AFEW も同じ
- **保存済みの値の再現**：試しの AFEW の ② と h2 の係数（`s_<s>__v0_A`・`_a0`・`_C`・`_rule`，`h2_<s>__v0_A`・`_a0`．ソース MOSEI，IEMOCAP-impro，IEMOCAP-script，RAVDESS，CREMA-D）は，E76 の本番の MELD の npz の同じ鍵と差 0（ビット単位で同じ）だった．同じ関数に同じ入力を渡して同じ値が出たので，差し替えた台本は本体と同じ計算をしている
- **退化した場合の検算**：新しい項は無い．①・②・③・h2・h4・h5・h7 が地図と E25 に 1e-16 の桁で一致した

**`--gpu-fraction` について**（依頼の 0.3 の確認）：`relearn_e76.py:776` の `settings["mem_fraction"]` は定数 `MEM_FRACTION`（0.6）で，引数の値は `provenance.json` の `argv` にだけ残る（試し 2 で確かめた）．`exp/logreg.py`・`exp/map.py`・`exp/interventions.py` には，空きメモリに応じて束の大きさや計算の順序を変える処理が無い（`mem_get_info`・chunk の類を検索して 0 件）．`set_per_process_memory_fraction` は割り当ての上限を決めるだけである．試し 2 で 0.6 と 0.2 の出力がビット単位で同じだったので，0.3 でも同じ値になると判断する．0.3 そのものは試しの上限（0.2）を超えるので流していない．0.3 × 24 GiB = 7.2 GiB は，E76 の audio_hubert の GPU の最大 1.72 GiB（MELD）と audio_wavlm の 3 ターゲットの最大 1.60 GiB より十分大きい．

## A．本番の規模の見積もり

- 試し 1 の AFEW 反復 0 は 235 秒で，そのうち ② の 89 秒は最初のターゲットで 6 ソースを学ぶ分である（E76 の本番では前のターゲットの ② を使い回したので 0 秒だった．使い回しても値は同じ：上の係数の一致）．反復 1 つあたり約 145 秒で，反復を束ねると下がる
- E76 の本番では audio_wavlm の同じ 3 ターゲットが 1,023 + 1,432 + 1,558 = 4,013 秒，audio_hubert は audio_wavlm の 0.5〜1 倍の時間だった．GPU を 1 本で使えば **3 ターゲットで 40〜80 分**とみる．別の試しの担当 2 体と GPU を分け合うと，E76 の設計 4 節の測定（3〜13 倍遅くなる）から数時間になりうる
- GPU のメモリの最大は 1.3〜1.8 GiB の見込みで，上限 0.3（7.2 GiB）に収まる．空きブロックを返す処理（`free_gpu`）は本体のまま
- ディスク：E76 の audio_wavlm の 3 ターゲットの npz は 44.6＋43.7＋76.4 MiB で，同じ程度を足す

## B．予想される指摘と，出力から答えられるか

| 表・図 | 予想される指摘 | もとにした項目か例 | (a)／(b)／(c) | (b) なら足すもの | 足す費用 |
|---|---|---|---|---|---|
| 再現の記録（E76 の 10 節，論文の再現の記録） | 規則を替えた 3 ターゲットで，元の規則なら止まっていたか | R，I7 | (a)：各 json の `check.h9`・`check.h10` の `n_over_tol > 0.01 × n_ref` を数える | — | — |
| 同上 | tol を超えた条件はどこか（ソース，K，反復，C，α） | I7，I14 | (a)：`<T>.csv` の method が h9・h10 の行で `diff > tol` を選び，`source`・`K`・`rep`・`C`・`alpha` を読む（MELD でこの手順で 2 条件と 4 条件を出した） | — | — |
| 同上 | ずれは E31 を RunPod の GPU で解いたせいか | I14 | (b)：E31 の O2・O1 を研究室の GPU で解き直して E76 の h9・h10 と比べる．E76 を学び直さずに計算の後でも出せるので，この計算には足さなくてよい | E31 の O2 の audio_hubert の解き直し | 数十分 |
| 筋 1〜3 の表（A(HuBERT) の行） | MELD と残り 3 ターゲットが違うコードで計算されていないか | I12，L4 | (a) だが手順が要る：新しい json の `code_files` から `exp/relearn_e76_finish.py` を除いた 10 本で作ったハッシュが 9891f0668848 であること．今は記録されないので，計算の後に `results/e76/code/<開始時刻>/` の写しから作り直す．指摘 1 の照合を足せば実行の時点で確かめて記録できる | 指摘 1 | 秒 |
| 同上 | E76 の出力を読む台本が，2 つのコードのハッシュと MELD の `check_failed` をどう扱うか | R | (a)：今 `results/e76` を読む台本は無い（`exp/`・`run/`・`DAI_paper/`・`slides/` を検索して `relearn_e76*.py` だけ）．ステップ 4 の台本を書くときの決まりが要る（指摘 4） | — | — |
| E76 の全体の表と図（話者を単位にした区間，性別，C の規則，中心合わせの手） | E76 の本体の関門 1（`checks/E76/code.md`）で扱った | — | この差分で変わらない（鍵と列が同じ） | — | — |

## 指摘

### [I12] 再開の照合が，残りの 3 ターゲットと既存の 4 ターゲットが同じコードと設定かを確かめなくなる
- 重さ：直してから進む
- 該当箇所：`exp/relearn_e76_finish.py:56-77`，`exp/relearn_e76.py:797-808`
- 問題：本体の再開の照合は，ファイルが既にあるターゲットのハッシュと設定だけを比べる（`same = meta.get("code_hash") == state["code_hash"] and meta.get("settings") == settings`）．続きの台本は自分を読み込むコードに含むので，ハッシュは作りから違う（試しで 52172b4ff5a0．既存は 9891f0668848）．しかも 3 ターゲットのファイルはまだ無いので，比べる相手が無い．このため，本番までに `exp/logreg.py`・`exp/map.py` などが変わっても，また `--reps`・`--K`・`--hands`・`--sources`・`--merged-all-c` を誤って渡しても，台本は止まらずに `results/e76/audio_hubert/` に書く．今の作業木では 10 本の中身は E76 と同じだった（ハッシュを作り直して一致）．台本はこの事実を実行の時点で確かめない
- 計算の後で直せるか：直せる（`code/<開始時刻>/` の写しから確かめられる）．ただし違っていれば 3 ターゲットを学び直す
- 求める対応：`R.main()` を呼ぶ前に，(1) `R.code_files()` から `exp/relearn_e76_finish.py` を除いた一覧で `R.code_hash(...)` を作り，`results/e76/audio_hubert/MELD.json` の `code_hash` と同じでなければ止める．(2) main が作る `settings` を MELD.json の `settings` と比べる．main の中で作るので，簡単な方法は，引数を許した 5 つ（`--kinds`・`--targets`・`--gpu-fraction`・`--out`・`--allow-dirty`）に限り，ほかの引数があれば止めることである（指摘 3 と同じ直し）．(1) の結果（E76 の部分のハッシュ）を `check_rule_finish.json` に書く

### [R] 台本が未コミットのままだと本番のコマンドで止まり，`check_rule_finish.json` に計算しなかった実行が残る
- 重さ：直してから進む
- 該当箇所：`exp/relearn_e76_finish.py:72-77`，`exp/relearn_e76.py:762-765`
- 問題：本番のコマンドには `--allow-dirty` が無い．`exp/relearn_e76_finish.py` は読み込むコードに入り，未追跡なので，main は「読み込むコードにコミットしていない変更がある：?? exp/relearn_e76_finish.py」で止まる（試し 3．終了コード 1）．止まるのは正しいが，続きの台本はその前に `check_rule_finish.json` に 1 件を書く（試し 3 で 1 件書かれた）．`results/e76/check_rule_finish.json` に，計算しなかった実行の記録が残る
- 計算の後で直せるか：直せる（記録だけ）
- 求める対応：台本をコミットして push してから本番を流す．`check_rule_finish.json` には，main が終わった後（`try/finally`）に結果（終わった・止まった・止まった理由）と，main の `started`（`code/<開始時刻>/` と `provenance.json` の run を結ぶ鍵）と `code_hash` を書く

### [I1] 種類とターゲットの制限を，`--targets=…` の書き方や略した引数で抜けられる
- 重さ：記録する（既存のファイルは main の再開の照合で守られる．指摘 1 の直しで一緒に閉じる）
- 該当箇所：`exp/relearn_e76_finish.py:46-60`
- 問題：続きの台本は `argv` から「`--targets` の後の値」を自分で拾い，main は argparse で読み直す．`--targets AFEW --targets=MELD` を渡すと，台本は AFEW と読んで通し，argparse は後の MELD を使う（試し 5：`provenance.json` の plan は MELD，`check_rule_finish.json` は AFEW）．argparse は `--targ` のような略も受け付ける．`results/e76` では MELD の npz が別のハッシュで既にあるので main が止まり，壊れはしない．ほかの 7 種類も全ターゲットのファイルがあるので同じく止まる
- 計算の後で直せるか：直せる
- 求める対応：`argv` の要素を，許した 5 つの引数の名前と値だけに限る（`=` を含む引数と許していない `--` の引数があれば止める）

### [R] 数の基準を外したことと，MELD を受け入れたことを，json だけからは読めない
- 重さ：記録する
- 該当箇所：`exp/relearn_e76_finish.py:33-40`，`:63-75`，`exp/relearn_e76.py:778`，`:855`
- 問題：(1) 新しい 3 ターゲットの json の `check_failed` は，h9・h10 の数の基準を超えても空になる．外したかどうかは `check` の `n_over_tol` と `n_ref` から計算し直さないと分からない．(2) `provenance.json` の run の `script` は本体に書き込まれた `"exp/relearn_e76.py"` のままになる（試し 1 で確かめた）．続きの台本で流したことは `code_files` と `code/<開始時刻>/exp/relearn_e76_finish.py` の写しからしか分からない．(3) `results/e76/audio_hubert/MELD.json` の `check_failed` は空でないまま残る．E76 の 10 節には「MELD の出力はそのまま使う」とあるが，機械で読める記録が無い．ステップ 4 で E76 の出力を読む台本が `check_failed` が空であることや `code_hash` が 1 つであることを求めると，MELD か新しい 3 ターゲットを落とす
- 計算の後で直せるか：直せる（どれも保存した出力から作り直せる）
- 求める対応：`verdict_finish` の中で，元の判定なら落ちていた文を集め，main が終わった後に `check_rule_finish.json` へターゲットごとに `waived` として書く．同じファイルに `accepted_existing: {"audio_hubert/MELD": "h9 の数の基準（2/180）"}` を書く．ステップ 4 の台本の設計に「audio_hubert は 2 つのコードのハッシュ（9891f0668848 と続きの台本のもの）を受け入れ，`check_failed` は `check_rule_finish.json` と合わせて読む」と書く

### [I7] h2〜h8・③ の数の基準で，残りの 3 ターゲットがまた止まりうる
- 重さ：記録する
- 該当箇所：`exp/relearn_e76_finish.py:35`，`exp/relearn_e76.py:720-724`
- 問題：既存の 42 個の json で tol を超えた条件があるのは audio_hubert だけである（IEMOCAP-script の h9 1 件，MOSEI の h3・h9・merged 各 1 件，MELD の h9 2 件・h10 4 件）．ほかの 7 種類は手と ③ のどれも 0 件で，AFEW・RAVDESS・CREMA-D の差 ÷ tol の最大は 0.86 だった．h3・h6・h7・h8 は照合の相手が 180 条件なので，2 件を超えると止まる．audio_hubert だけ照合の相手（地図，E25，E31）との差が大きい理由は確かめていない．続きの試し（AFEW の反復 0）ではすべて tol の 0.5 倍以下だった
- 計算の後で直せるか：止まった場合は，研究者の承認の範囲（h9・h10 だけ）の外になる
- 求める対応：本番の前に，h2〜h8・③ の数の基準で止まったときにどうするか（止めたままにするか，同じ手順で承認を取るか）を E76 の 10 節に書いておく

## 目録に無い種類の不備

**判定や設定を差し替える包み台本（wrapper）**：本体の再開の照合はコードのハッシュの一致で「同じ計算」を保証していたが，包み台本は自分をハッシュに含めるので，作りからこの保証を外す．`docs/methods.md` 9.3 節に次の項目を足す案：「I-wrap：本体の関数を差し替える包み台本で残りを計算するときは，(i) 包み台本を除いた読み込むコードのハッシュが既存の出力のハッシュと同じであること，(ii) 設定が既存の出力の設定と同じであること，を計算の前に台本の中で確かめて記録する．(iii) 引数は本体の引数のうち許したものだけに限り，包み台本と本体が引数を別々に読む作りにしない．(iv) 差し替えた規則で受け入れた出力（既存のものを含む）を機械で読める形で残す」
