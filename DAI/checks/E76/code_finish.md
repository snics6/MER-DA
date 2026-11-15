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

---

# 再検査（2026-11-02，コミット 61bd0df）

## 判定

通す．4 つの直し（[I12] のコードのハッシュの照合，[I1] の引数の制限，[R] の計算の後の記録，[I7] の E76 の 10 節の追記）はどれも意図どおりに動いた．直した台本で流した AFEW の反復 0 の出力は，直す前の試しの出力と，配列・csv・settings・check がビット単位で同じだった．残る 2 件はどちらも本番の出力の値に影響しない（下の「残る指摘」．記録する）．

## 確かめたこと

差分は `exp/relearn_e76_finish.py`（81 行 → 132 行），`docs/exploratory-analyses.md` の E76 の 10 節（2 行），この報告である．`exp/relearn_e76.py` とその依存の 10 本は変わっていない．作業木の未コミットの変更は `exp/explore_view_fusion.py` だけで，読み込むコードに入らない．GPU は試しの間，ほかの計算が無かった（1.2 GiB，使用率 0%）．試しはすべて `--gpu-fraction 0.2` 以下で流し，出力は scratchpad に書かせた．`results/e76` の下に `logs/e76.log` より新しいファイルは無い．

**[I12] コードのハッシュの照合**（`relearn_e76_finish.py:82-88`）
- 今の作業木では，包み台本を除いた `R.code_files()` の一覧が MELD.json の `code_files` と同じで，ハッシュは 9891f0668848 で一致した
- `REF_JSON` を，`code_hash` だけを 0 に替えた偽の json に差し替えると，「包み台本を除いたコードが E76 の本番と違う」で止まった．`R.main` は呼ばれず，`--out` のフォルダも作られなかった（CPU で確かめた）
- `REF_JSON` は `--out` によらず `results/e76/audio_hubert/MELD.json` を読む．試しでも本番の参照と照合するので，これは正しい

**[I1] 引数の制限**（`:51-69`，`:74-79`．どれも計算の前に止まり，`--out` のフォルダは作られなかった）

| 渡した引数（`--kinds audio_hubert` に続けて．最後の行は除く） | 結果 |
|---|---|
| `--targets=MELD` | 「`--opt=値` の書き方は使わない」で止まる |
| `--targets MELD` | 範囲の外で止まる |
| `--targets AFEW --reps 0` | 「`--reps` は受け付けない」で止まる |
| `--targets AFEW --targets CREMA-D` | 「2 回ある」で止まる |
| `--targ AFEW`（argparse なら略として通る） | 受け付けないで止まる |
| `--targets AFEW -h` | `-h` がターゲットの値に入り，範囲の外で止まる |
| `x --kinds audio_hubert --targets AFEW` | 位置引数で止まる |
| `--kinds video --targets AFEW` | 範囲の外で止まる |

前回の試し 5 のように argparse が別の値を読む道筋は無くなった．`--reps` などを渡せないので，設定は本体の既定値に固定される．本体の既定値は，上のハッシュの照合で E76 の本番と同じことが保証される．

**[R] 計算の後の記録と，計算が変わっていないこと**

台本は `--reps` を受け付けなくなったので，小さく流す方法が無い．そこで，ROOT の外に置いた検査用の台本（scratchpad の `harness/run_finish_rep0.py`）から流した．この台本は，`relearn_e76_finish.main` の引数の検査とハッシュの照合を通した後，本体の main を呼ぶときにだけ `--reps 0` を足す．ROOT の外に置いたので，`code_files` には入らない．

- コマンド：`./.venv/bin/python <scratchpad>/harness/run_finish_rep0.py --kinds audio_hubert --targets AFEW --gpu-fraction 0.2 --out <scratchpad>/rechk_trial`
- 結果：329 秒，終了コード 0，GPU の最大 1.31 GiB
- 前回の試し 2（直す前の台本，`--gpu-fraction 0.2`）と突き合わせた（`checks/E76/code_run/finish_fraction.py`）．npz の 241 配列がビット単位で同じで，csv（288 行 × 28 列），settings，check も同じだった．code_hash だけが違う（包み台本の中身が変わったため．17f731311ff0）
- json は `code_dirty` が偽で，`commit` は 61bd0df だった
- `check_rule_finish.json` には 1 件だけ書かれ，次の値が入っていた
  - `result: "ok"`，`started`・`finished`
  - `code_hash_without_wrapper` と `code_hash_e76`：どちらも 9891f06688489b66…
  - `written: {"audio_hubert/AFEW": "17f731311ff0…"}`，`waived: {"audio_hubert/AFEW": []}`
  - `accepted_existing: {"audio_hubert/MELD": [h9 の 1 件]}`
- `written` と `waived` は，json の `started` が包み台本の `started` 以降のファイルだけを数える（`:119`）．前の実行で済んだターゲットは数えない．本体が飛ばしたファイルを今回の実行のものと取り違えないので，これは正しい

**[I7]**：E76 の 10 節に「①・②・③・h2〜h8 で止まったら規則を広げずに研究者に報告する」と，ステップ 4 で A(HuBERT) の 2 つのハッシュを受け入れることが書かれた．

**`--gpu-fraction 0.3`**：上限を変えても出力がビット単位で同じことは，前回 0.6 と 0.2 で確かめた．今回の 0.2 でも同じだった．`settings.mem_fraction` は定数 0.6 のままで，0.3 は `provenance.json` の `argv` にだけ残る．

## 残る指摘

### [R] 本体が `--out` のフォルダを作る前に止まると，finally の書き込みが FileNotFoundError になり，止まった理由の表示を上書きする
- 重さ：記録する（本番の `results/e76` は既にあるので起きない）
- 該当箇所：`relearn_e76_finish.py:125-128`
- 問題：新しい `--out` に `--gpu-fraction abc` か `--allow-dirty foo` を渡すと，argparse のエラー（終了コード 2）の後に，finally の `p.write_text` が `FileNotFoundError` を出す．終了コードは 1 になり，記録は書かれない（流して確かめた）．読み込むコードが dirty で止まった場合も同じ道筋を通る（本体の `relearn_e76.py:763-765` は `out.mkdir` の前にある）
- 求める対応：finally の中で `out.mkdir(parents=True, exist_ok=True)` してから書く

### [R] `waived` は元の規則の判定のすべてで，外した基準の分だけではない
- 重さ：記録する
- 該当箇所：`relearn_e76_finish.py:122`
- 問題：新しい規則でも落ちたターゲット（h9 が tol の 2.5 倍を超えた，h3 の数の基準で落ちた，など）では，同じ文が `waived` と json の `check_failed` の両方に入る．`result` が SystemExit になるので，読み分けることはできる
- 求める対応：`waived` を「元の判定の文のうち `check_failed` に無いもの」にするか，ステップ 4 の台本で読むときにそう読むと書く

`provenance.json` の run の `script` は今も `"exp/relearn_e76.py"` である（本体の 778 行）．包み台本で流した run とは，`check_rule_finish.json` の `started`〜`finished` の時間の範囲と，`written` の code_hash で結べる．記録として足りる．

---

# 再検査 2（2026-11-02，コミット fc4fbd7．検査の間に HEAD は 208fdf5 に進んだが，差は E81 の文書と台本だけで，`exp/relearn_e76_finish.py` と E76 の依存の 10 本は変わっていない）

## 判定

通す．①・② の C と規則の違いを記録だけにする判定は，承認の条件（C の 4 つの値すべてのテストの UAR が tol の中，選ばれた C の UAR の基準も満たす）のときだけ通し，そうでなければ止まる．既存のファイルの判定が変わるのは，承認で受け入れた MELD と RAVDESS だけだった．CREMA-D だけを流すと，AFEW・RAVDESS のファイルには触れず，check_rule_finish.json に 2 件目が正しく足された．

## 確かめたこと

差分は `exp/relearn_e76_finish.py` の 29 行である．`C_WAIVED`，`c_waivable`，`verdict_finish` の先頭（summ を写してから C_mismatch と rule_mismatch を 0 にする），`change_c`，`accepted_existing` の RAVDESS，docstring が変わった．1 回目の本番（`results/e76` の AFEW・RAVDESS）はコミット 2790344 で流れていた．2790344 と 61bd0df の間で `exp/`・`measure/`・`features/` に差は無く，AFEW・RAVDESS の code_hash は前回の試しと同じ 17f731311ff0 だった．

**(1) 判定**（CPU．評価の値は出していない）

既存の 41 個の json に元の判定と新しい判定を当てると，結果が違うのは 2 つだった．
- audio_hubert の MELD：元の判定は 1 件で落ち，新しい判定は 0 件．h9 の数の基準で，前回の承認どおり
- audio_hubert の RAVDESS：元の判定は 1 件で落ち，新しい判定は 0 件．tgt_only の C の不一致
- AFEW はどちらの判定も 0 件で，今までと同じ

RAVDESS で C が違った条件は 1 つ（K＝全件，反復 6）だった．E76 は C=10 を規則 cv で選び，地図は C=1 を規則 tie で選んだ．保存した交差検証の UAR では，上位 2 つの差は 0.0016 だった．

RAVDESS の check の tgt_only を人工的に書き換えて，新しい判定を当てた．

| 書き換え（C_mismatch＝1，rule_mismatch＝1 のまま．最後の 2 行は除く） | 新しい判定 |
|---|---|
| そのまま（max_allC_over_tol ＝ 3.3e-15） | 通る |
| max_allC_over_tol ＝ 1.5（C の 4 つの値のテストの UAR が tol を超える） | 止まる |
| max_allC_over_tol ＝ 1.0（境界） | 通る（`<= 1.0`．設計の「≤ 1」と同じ） |
| max_allC_over_tol ＝ None | 止まる |
| 選ばれた C の差 ÷ tol ＝ 2.6 | 止まる |
| 選ばれた C の tol 超え 2/60 | 止まる |
| C_mismatch ＝ 5，rule_mismatch ＝ 5，テストの UAR は一致 | 通る（下の残る指摘） |
| src_only で C_mismatch ＝ 1，max_allC_over_tol ＝ 1.2 | 止まる |
| C の不一致が無く，max_allC_over_tol ＝ 3.0 | 止まる（元の基準のまま） |

`verdict_finish` は summ を写してから書き換えるので，呼んだ側の summ は変わらない（C_mismatch は 1 のまま）．json の `check` には実際の不一致の数が残る．

**(2)(3) CREMA-D だけを流す道筋**

GPU は試しの前に 1.2 GiB・使用率 0% だった．scratchpad の `rechk2/` に，本番の今の状態を写した．写したのは `results/e76/check_rule_finish.json`（1 件）と，`audio_hubert/` の AFEW・RAVDESS の json・npz・csv で，それぞれの sha256 を控えた．そのうえで，前回と同じ検査用の台本（引数の検査とハッシュの照合を通した後，本体にだけ `--reps 0` を足す）から流した．

- コマンド：`<scratchpad>/harness/run_finish_rep0.py --kinds audio_hubert --targets CREMA-D --gpu-fraction 0.2 --out <scratchpad>/rechk2`
- 結果：219 秒，終了コード 0，GPU の最大 1.38 GiB，npz 10.0 MiB
- 包み台本を除いたハッシュは 9891f0668848 で，E76 の本番と一致した．計算の前の照合を通った
- CREMA-D の code_hash は b5b36158ddd4 で，AFEW・RAVDESS の 17f731311ff0 とも，E76 の 9891f0668848 とも違う．`code_dirty` は偽，`check_failed` は空
- 照合はすべて tol の 1.0 倍以下で，超過は 0 件だった．h7 と merged は差 ÷ tol がちょうど 1.0 で，超過は `diff > tol` で数えるので 0 件になる．C と規則の不一致は 0 件
- AFEW・RAVDESS の json・npz・csv は sha256 が流す前と同じだった．再開の照合は `--targets` に入れたターゲットのファイルしか見ないので，ハッシュの違う AFEW・RAVDESS に触れずに進んだ
- `check_rule_finish.json` は 2 件になった．1 件目は流す前の内容と同じだった．2 件目には次の値が入っていた
  - `script`，`started`・`finished`，`result: "ok"`
  - 2 つのハッシュ（どちらも 9891f066…）
  - `written: {"audio_hubert/CREMA-D": "b5b36158…"}`，`waived: {"audio_hubert/CREMA-D": []}`
  - `change_c`
  - `accepted_existing`：MELD の h9 の 1 件と，RAVDESS の tgt_only の 1 件

本番の 1 回目の記録では，`written` は AFEW・RAVDESS ともに 17f731311ff0 で，`waived` は AFEW 0 件・RAVDESS 1 件（tgt_only）だった．`result` は SystemExit で，止まった理由と合っている．

ステップ 4 で audio_hubert を読む台本は，コードのハッシュを 3 つ受け入れる．9891f0668848 は既存の 4 ターゲット，17f731311ff0 は AFEW・RAVDESS，CREMA-D の値は本番で決まる．E76 の 10 節にある「2 つのハッシュを受け入れる」は「3 つ」に直す．

## 残る指摘

### [I7] C の不一致の数に上限が無い
- 重さ：記録する
- 該当箇所：`relearn_e76_finish.py` の `c_waivable`
- 問題：テストの UAR が C の 4 つの値すべてで地図と合えば，C の不一致が 5/60 でも通る．交差検証の折り方を取り違えると，テストの UAR を変えずに C だけが変わりうる．今回は包み台本を除いたコードのハッシュが E76 の本番と同じことを計算の前に確かめているので，交差検証のコードが違う可能性は無い．数のずれは丸めでしか生じない．また，選ばれた C の UAR の数の基準（超過 1% 以下）が，C が大きく入れ替わった場合を捕まえる
- 求める対応：直さなくてよい．CREMA-D で C の不一致が出たら，その数を E76 の 10 節に書く

### [記録] E76 の 10 節の「2 つのハッシュ」を「3 つ」に直す
- 重さ：記録する
- 該当箇所：`docs/exploratory-analyses.md:8973`（「A(HuBERT) についてこの 2 つのハッシュを受け入れる」，「承認の範囲は h9・h10 だけ」）
- 問題：CREMA-D は AFEW・RAVDESS と違うハッシュになるので，受け入れるハッシュは 3 つになる．承認の範囲には，2 回目の承認で①・② の C と規則が加わった（同じ節の「2 回目の止まり」）
- 求める対応：8973 行を「3 つのハッシュ（9891f0668848，17f731311ff0，CREMA-D の値）」と「承認の範囲は h9・h10 の数の基準と，①・② の C と規則の違い（条件つき）」に直す

前回の残る指摘 2 件（`--out` が無いときの finally，waived の中身）は直っていないが，本番の `results/e76` では影響しない（前回のとおり）．
