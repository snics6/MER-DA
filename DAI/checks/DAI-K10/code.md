# 関門 1：実装と出力（DAI-16・17 の K = 10 の追補，2026-11-14，コミット edc7edb）

## 判定

直してから通す．`--ks 10` の台本 2 本（`dai16_probe.py`・`dai17_probe.py`）は K = 10 だけを流し，既定の置き場を書き換えず，同じ K で流すと保存の値をビット単位で再現した．一方，コミット edc7edb の `dai16_posthoc.py` は K = 10 の出力で `KeyError: 'D_M_K_K5'` を出して止まる（流して確かめた）．作業木にある未コミットの直し（`provenance.json` の K を読む形）は試走で正しく動いたので，これをコミットすれば通す．

検査の途中で作業木が変わった．検査を始めた時点では `DAI/exp` に未コミットの変更が無かったが，途中で次の 3 つが現れた：`DAI/exp/dai16_posthoc.py` の直し（未コミット），追補の文書の第 2 版（未コミット），起動の台本 `DAI/run/dai_k10.sh`（未追跡）．3 つとも読み，`dai16_posthoc.py` の直しは試走で流した．`dai16_probe.py`・`dai17_probe.py`・`dai17_posthoc.py` は検査の終わりの時点で edc7edb と同じである（sha256 はそれぞれ `62863fa8…`，`1a99e29b…`，`4301835e…`）．

検査役が見たもの：行数，列，K の値，鍵の集合，NaN の数，保存との差の最大，所要，GPU のメモリ，提案や K 本だけの分類器との一致の割合．UAR・判定の語・選ばれた点の値は画面に出していない．`results/dai1/eval/` は開いていない．`results/` に書いたものは無い（`results/dai16_probe_k10`・`results/dai17_probe_k10` は検査の後も存在しない）．試走の出力（npz と集計）は，形を数えた後に中を開かずに消した．残したのはログと検算の台本だけである（`DAI/checks/DAI-K10/code_run/`）．

## A．設計と台本の対応表

| 設計（追補）の項目 | 台本の箇所 | 合っているか |
|---|---|---|
| `--ks 10` で K = 10 だけを流す | `dai16_probe.py:1134–1136`，`dai17_probe.py:1273–1275`（`KS = tuple(sorted(set(args.ks)))`．`global KS, K_MAIN`） | 合う．単位の csv の K は 10 だけ（DAI-17 は K に依らない行の −1 も），npz に `K5`・`K25` の鍵は 0 個 |
| 出力は `results/dai16_probe_k10/`・`results/dai17_probe_k10/` | `dai16_probe.py:1137,1140`，`dai17_probe.py:1276,1279`（`out_k = f"{OUT_PROD}_k10"`，`prod = out == out_k`） | 合う．名前は追補と同じ |
| 既存の `results/dai16_probe/`・`results/dai17_probe/` を書き換えない | `dai16_probe.py:1138–1139`，`dai17_probe.py:1277–1278` | 合う．`--ks 10 --out-dir results/dai1{6,7}_probe` は「--ks を渡すときは，既定の置き場ではなく別の置き場に書く」で書く前に止まった（流して確かめた） |
| 追補の文書がコミットされていなければ止まる | `dai16_probe.py:1141–1142`，`dai17_probe.py:1280–1281`（`dai1.committed_clean(ADDENDUM)`：追跡されていて HEAD と差が無い） | 合う．検査の初めには追補はコミット済みで，`--ks 10 --out-dir results/dai1{6,7}_probe_k10 --targets AFEW ...` は次の「本番の置き場では対象を絞らない」で書く前に止まった（本番の置き場と判定され，追補と未コミットの確かめを通ったことを示す）．いまは追補が第 2 版で未コミットなので，コミットするまで本番は止まる |
| 未コミットの変更で止まる | `dai16_probe.py:1145–1146`，`dai17_probe.py:1284–1285`（`dai1.git_commit` は `DAI/exp`・`DAI/splits`・`measure`・`features`・`data/metadata` の未追跡も数える．`dai1.py:70–75`） | 合う（既存の箇所．差分で変わっていない）．いまの作業木では `dai16_posthoc.py` の直しが未コミットなので本番は止まる |
| 規則の主の K は 10 | `K_MAIN = K_MAIN if K_MAIN in KS else KS[0]`．使う箇所は `dai16_probe.py:893–894`，`dai17_probe.py:985–986,1091,1159` | 合う．試走の `criteria.csv` の `main` の行の K はすべて 10（DAI-16 34 行，DAI-17 A 21 行，B 9 行，C 12 行） |
| すべての段が K = 10 を使う（DAI-16：units・summary，DAI-17：units・reweight・students・summary） | `KS` を読む箇所：DAI-16 は 536，809，831，870，887，939，973 行．DAI-17 は 408，595，622，638，700，941，968，981，1041，1068，1078，1112，1129，1146 行．どれも関数の中で実行のときにモジュールの `KS` を読む．既定の引数や，読み込みのときに作る定数で `KS` を使う箇所は無い | 合う．`K5`・`K25`・`== 5`・`== 25` の決め打ちは 4 本の台本（probe 2 本，posthoc 2 本）に無い（grep）．DAI-17 が呼ぶ `dai16_probe` の関数（`teachers_mod`・`View`・`mix_rel` など）は K を引数で受け，`P16.KS` を読まない |
| 呼ぶ部品の K | `dai15.KS = (5, 25)` は `dai15` の中の段の関数（`run_unit`・`stage_summary`）だけが使い，DAI-16・17 が呼ぶ `kfold_of`・`table_from`・`shrink`・`match_prior`・`sha` は使わない．`dai1.build_tk`・`dai1.Unit.kpos` は `DD.KS = (1, 5, 10, 25)` で全 K を作る | 合う |
| 実験 1 の固定のファイルの K = 10 の値（引き戻しの強さ，提案の設定，K 本） | `dai3.Saved`（`tether_of`，`fz["select"]["<s>|10|joint|one"]`，`fz["main"]["10"]`），`u.kpos[10]` | 合う．`results/dai1/select/frozen.json` の 4 ターゲットに `main["10"] = "one"`，`w["10"]`，引き戻しの K = 10，`AVT|10|joint|one`，`partner` の K = 10 がある．試走で，作り直した積と実験 1 の保存の積の差 5.0e-6〜2.6e-5（DAI-16，許しは `CHECK_TEACHER`），B の段で K 本だけの分類器と実験 1 の保存の一致の最小 1.0，C の段で積の疑似ラベルの生徒と実験 1 の保存の提案の一致の最小 1.0000 |
| K 本の入れ子（K = 10 の 40 本は K = 5 の 20 本を含み，K = 25 に含まれる） | `u.kpos[K]`（`dai1.py:171`） | 合う．試走の 9 単位で `kpos_K5 ⊂ kpos_K10 ⊂ kpos_K25`，40 本，感情ごとに 10 本，テストの `sample_id` が保存の単位と同じ |
| K を渡さないときの動きは変わらない | 差分は `if args.ks:` の中と `global` の 1 行だけ | 合う．selftest は DAI-16「版 204，差の定義 288」，DAI-17「版 145，差の定義 195，格子 17」で通った．`--ks 5` で流した単位は保存の K = 5 の値とビット単位で同じ（下） |
| `dai17_posthoc.py` の依存の表は `provenance.json` の K を読む | `dai17_posthoc.py:76` | 合う．試走の `provenance.json` の KS は `[10]`，`frozen_sel.json` の鍵の K は 10 だけで，依存の表が引く鍵（`<c′>|10|<版>`）はすべてあった（`code_run/check_dep_k10.py`） |
| `dai16_posthoc.py` の足し算だけの対照（追補の第 2 版で追加） | edc7edb：`dai16_posthoc.py:64,111` の `for K in P16.KS`．作業木：`KS_RUN = tuple(json.load(open(out / "provenance.json"))["KS"])` | **edc7edb では合わない（指摘 1）**．作業木の直しは合う |
| 起動の台本で 6 段を順に流し，既存の置き場の sha256 を前後で照合する（追補の第 2 版） | `DAI/run/dai_k10.sh`（未追跡） | 合う．どの段にも `--ks 10` と `_k10` の置き場を渡す．`set -euo pipefail` と `trap ERR` で，失敗したら `logs/dai_k10.failed` を作る．注意は指摘 2 |

設計の表の組の数え（9.3 節 L4）：追補は元の設計の格子をそのまま K = 10 にするので，計算する組は元の設計の組の K を 10 にしたものである．計算しない K（50）は追補 0 節に理由がある（実験 1 に K = 50 の保存が無い）．

## A．小さく流した結果

流す前に GPU を見た．使用 1406 MiB，使用率 0%，計算のプロセスは無かった．tmux `mer-da` は bash だけだった．`MERDA_GPU_FRACTION` は既定の 0.6．置き場は `DAI/checks/DAI-K10/code_run/` の下．

| 命令（`--allow-dirty --allow-partial` つき） | 所要 | 結果 |
|---|---|---|
| `dai16_probe.py --stage selftest`，`dai17_probe.py --stage selftest` | 各 2 秒 | 通った（版と差の数は元の検査と同じ） |
| `dai16_probe.py --stage units --ks 10 --targets AFEW --folds 0 --seeds 0`（`-W error::FutureWarning`） | 2 分 5 秒（単位 30・60・32 秒） | 3 単位，各 257 行．作り直しの差の最大 2.6e-5，勾配の最大 1.0e-6，止まり 0，割合をそろえる残り 1.0e-8．警告は `Mean of empty slice`（258・341 行．元の検査の記録する 4 と同じ） |
| `dai16_probe.py --stage summary --ks 10` | 31 秒 | 止まらずに終わった．`effects.csv` 1623 行，`criteria.csv` 81 行，`levels.csv` 654 行，`r0.csv` 24 行．どの表も K は 10 だけ（`reliability.csv` は K に依らない行の 0 と 10）．`provenance.json` の KS は `[10]`．`mixsel_units.csv` は空（試走に c を評価のターゲットにした単位が無いため） |
| `dai17_probe.py --stage units --ks 10 --targets AFEW`，同 `--targets IEMOCAP`（分割 0・種 0） | 97 秒＋約 90 秒（単位 22〜45 秒） | 6 単位，各 796 行（K = 10 の 206 行と K に依らない 590 行）．作り直しの差の最大 2.6e-5，勾配の最大 1.0e-6，止まり 0，GPU の最大 5345 MB |
| `dai17_probe.py --stage reweight --ks 10 --targets AFEW`，同 `IEMOCAP` | 76 秒＋約 70 秒（組 15〜32 秒） | 6 組，各 205 行．K 本だけの分類器と実験 1 の保存の一致の最小 1.0，base の学び直しと保存のソースの分類器の差の最大 3.3e-5，GPU の最大 4962 MB |
| `dai17_probe.py --stage students --ks 10` | 57 秒（単位 5〜14 秒） | 6 単位．提案との一致の最小 1.0000，勾配の最大 1.0e-6，GPU の最大 4018 MB．`frozen_sel.json` の鍵 22 個，K はすべて 10 |
| `dai17_probe.py --stage summary --ks 10` | 24 秒 | A：差 1520 行・数え 75 行，B：差 144 行・数え 42 行（`main` 9 行），C：差 144 行・数え 68 行（`main` 12 行）．どの表も K は 10（B は K に依らない `0`・`0ob` も）．警告は `Mean of empty slice` と `invalid value encountered in scalar divide` の各 3 回（分割 0 だけの試走で jackknife が空になるため．元の検査と同じ） |
| 作業木の `dai16_posthoc.py --out-dir <試走>` | 数十秒 | 3 単位，39 行，差の行 36．5 つの表の K は 10 だけ．保存の判定との一致の最小 1.0000 |
| edc7edb の `dai16_posthoc.py`（`git show` で写した `code_run/dai16_posthoc_edc7edb.py.txt`） | — | `KeyError: 'D_M_K_K5 is not a file in the archive'` で止まった（指摘 1） |
| `dai17_posthoc.py --out-dir <試走>` | — | 重なり・同じ俳優・ソースごと・生徒の条件の 5 表は書けたが，依存の表で `All-NaN slice` で止まった．台本の `corp` は 4 コーパスの固定で，試走に T = MOSEI・MELD の単位が無いためである（試走だけの理由．K とは関係しない）．依存の表の K の扱いは `code_run/check_dep_k10.py` で別に確かめた |

退化した場合の検算と保存の値の再現：

- **`--ks 5` で保存の K = 5 を再現した．**`--ks 5` で AFEW を評価のターゲットにした分割 0・種 0 の 3 単位を DAI-16 と DAI-17 の両方で流し，`results/dai16_probe/units`（コミット 0039bf0）と `results/dai17_probe/units`（コミット 48212d3）の同じ単位と比べた（`code_run/compare_units.py`，出力 `compare_d16.out`・`compare_d17.out`）．npz の鍵の集合は同じ（DAI-16 1104 個，DAI-17 869 個），浮動小数の差の最大 0，整数と文字の不一致 0．csv も K = 5 と K に依らない行で，数の列の差の最大 0，文字の列と NaN の位置の不一致 0．`--ks` の道で流した K ごとの計算は，既定の道の計算と同じである．保存のコミットから edc7edb までに `dai16_probe.py`・`dai17_probe.py` が変わったのは `main` の中だけである（`git diff 0039bf0 edc7edb`，`git diff 48212d3 edc7edb`）．
- **K = 10 の出力の形は K = 5 の出力の K を置き換えたものと同じである．**3 単位ずつで，npz の鍵の集合が `K5 → K10` の置き換えで一致し，csv の行数と版の集合も一致した．DAI-16 の NaN を含む鍵 4 つ（`SR_S_homeauc`・`SR_S_awayauc`・`AUC_M_K_K10`・`AUC_M_orc_K10`）は，K = 5 の保存でも同じ位置に同じ数（12 個）の NaN がある記録用の配列である．
- **K に依らない量は元の回と同じである．**DAI-17 の K に依らない 590 行（近い発話の記述など）と近い発話の番号の配列は，`--ks 5` の試走と保存でビット単位で同じだった．`--ks 10` の回でも同じ計算の道を通る．
- 手法の間で同じテストを使っていること：単位の中で版はすべて同じ `y_test`・`cl_test` を使う（元の検査で確かめた構造が変わっていない）．K = 10 の単位のテストの `sample_id` は保存の K = 5・25 の単位と同じだった．
- **`provenance.json` の照合（追補 2 節の「表を作る前の確かめ」）を試走で先に当てた．**`dai1_frozen_sha256`・`inputs_sha256`・設計の sha256 は保存と同じで，`scripts` の違いは DAI-16 で `dai16_probe.py`，DAI-17 で `dai16_probe.py`・`dai17_probe.py` だけだった．

## A．本番の規模の見積もり

- 同じ単位の所要の比（K = 10 だけ ÷ 保存の K = 5・25）は，DAI-16 の単位で 0.50〜0.72，DAI-17 の A で 0.51〜0.76（1 単位だけ 2.58．短い単位の揺れ），B で 0.39〜0.81（1 組だけ 2.17）だった．保存の合計（DAI-16 の単位 115 分，DAI-17 の A 69 分，B 24 分，C 78 分）から，本番は DAI-16 の単位 60〜85 分，DAI-17 の A 35〜55 分，B 10〜20 分，C 約 40 分，集計 2 回で 20〜40 分，計算の後の量 2 本で数分〜十数分，既存の置き場の sha256（3.9 GB，927 ファイル）2 回で数分，合わせて約 3〜4 時間と見込む．追補 3 節の見積もり（K を 2 つ流したときの値）はこれより長めであり，害は無い．
- GPU のメモリ：試走の最大は A の 5345 MB で，上限の 6 割（約 14.7 GB）の中．各段の単位ごとに `torch.cuda.empty_cache()` を呼ぶ（`dai16_probe.py:725–726`，`dai17_probe.py:788–789,815–816,852–853`）．
- ディスク：保存の大きさ（DAI-16 707 MB，DAI-17 3.2 GB）の約半分と見込む．空きは 525 GB．

## B．予想される指摘と，出力から答えられるか

| 表・図 | 予想される指摘 | もとにした項目か例 | (a)／(b)／(c) | (b) なら足すもの | 足す費用 |
|---|---|---|---|---|---|
| K = 5・10・25 を並べる表 | 3 つの K の点推定と区間を同じ単位で並べよ | 追補 2 節，R1 | (a)．2 つの置き場の `summary/levels.csv`・`effects.csv`・`criteria.csv` を (c, K, a, b, where) で縦につなぐ．単位（T, c, 分割, 種）とテストは同じ（試走で `sid_test` の一致を確かめた） | — | — |
| 同上 | K どうしの差（例：K = 10 と K = 5 の `M_F6_mixsel − prod` の差）の区間を出せ | 追補 2 節「K で増える／減る」，S 系，A 系 | (a)．追補はこの区間を作らないと書いたが，後から求められても学び直しは要らない．両方の置き場の `units/<単位>.npz` に，同じテストの発話ごとの判定 `pred_adj_K<K>_<版>` と塊 `cl_test`・`y_test` があり，選ばれた格子の点は `summary/mixsel.csv`（DAI-17 は `sel.csv`・`students_sel.csv`・`reweight_sel.csv`）で引ける．単位ごとに「(a−b) の K = 10 の値 − K = 5 の値」を塊の数え（`dai1_stats.counts`）から作り，元の区間と同じ jackknife と種の分散の手順に通せば出せる．差の差の統計量の関数は台本に無いので書き足しが要る（CPU だけ） | — | — |
| 同上 | 話者（塊）を単位にした区間か | 過去の例（tac-round1-status） | (a)．`effects.csv` の区間は `cl_test` の塊の数えから作る（元の設計と同じ） | — | — |
| 同上 | 感情ごとの内訳 | R 系，過去の例 | (a)．`per_emotion.csv`・`students_per_emotion.csv`．区間は無い（追補 2 節が「K の間で比べる文を書かない」と限っている） | — | — |
| 同上 | K ごとに選ばれた格子の点が違うことが差を作っていないか | 追補 2 節，C 系 | (a)．`mixsel.csv`・`sel.csv`・`students_sel.csv`・`reweight_sel.csv` を K で並べる．格子の全点の判定（DAI-16 は単位の npz に全版の `pred_adj_K10_*`，DAI-17 の C は `pred_K10_<生徒>_<疑似ラベル>_g<点>` と確率）が保存されるので，K = 5 の点を K = 10 に当てた値も出せる | — | — |
| 同上 | K = 10 の 40 本が K = 5 の 20 本を含むことを示せ | 追補 0・2 節 | (a)．`units/*.npz` の `kpos_K10` と保存の `kpos_K5`・`kpos_K25`．試走の 9 単位で入れ子を確かめた | — | — |
| K に依らない量（DAI-17 の近い発話，`dai17_posthoc.py` の 4 表） | 元の回と同じ値か | 追補 2 節 | (a)．両方の置き場の `summary/neighbors*.csv` と `posthoc/overlap.csv`・`same_actor.csv`・`by_source_bal.csv`・`speakers_top20.csv` を突き合わせる．試走の単位の段では K に依らない行と配列が保存とビット単位で同じだった | — | — |
| 全体 | 2 つの回が同じ入力と同じ部品で流れたか | 追補 2 節，I9 | (a)．`provenance.json` の `dai1_frozen_sha256`・`inputs_sha256`・`scripts`．試走で照合の形を確かめた（違いは DAI-16・17 の台本だけ） | — | — |
| 全体 | 追補のどの版で流したか | R 系，記録 | (a)（間接）．`provenance.json` の `commit` から `git show <commit>:DAI/docs/実験16・17_下調べ_K10_追補.md` で引ける．追補の sha256 そのものは残らない（記録する 3） | — | — |
| 全体 | K = 50 ではどうか | 研究者の最初の指示 | (c)．実験 1 に K = 50 の保存が無い．追補 0 節に研究者の判断が書かれ，K = 10 は実験 1 の K の格子の 1 点として読むと限っている | — | — |
| 全体 | 評価のターゲットの回の K = 10 | 追補 0 節 | (c)．下調べは仮のターゲットの回だけで，本番に進む判断に使わないと追補 2 節が限っている | — | — |

止めるべき (b) は無い．

## 指摘

### [I1・I8] コミット edc7edb の `dai16_posthoc.py` は K = 10 の出力で止まる
- 重さ：直してから進む
- 該当箇所：edc7edb の `DAI/exp/dai16_posthoc.py:64` と `:111`（`for K in P16.KS:`）
- 問題：`P16.KS` は `dai16_probe` を読み込んだときの値 `(5, 25)` のままで，`--ks 10` は `dai16_probe.main` の中でしか書き換えない．K = 10 の置き場の npz には `D_M_K_K5` が無いので，`tables(z, 5)` で止まる．edc7edb の版を試走の置き場に流し，`KeyError: 'D_M_K_K5 is not a file in the archive'` を確かめた．誤った数値は出さず，本番の計算の後に止まるので，直して流し直せば学び直しは要らない．作業木の未コミットの直し（`KS_RUN = tuple(json.load(open(out / "provenance.json"))["KS"])`，64・111 行を `KS_RUN` に）は試走で 3 単位を流し，5 つの表の K がすべて 10，保存の判定との一致の最小 1.0000 だった．既存の置き場の `provenance.json` の KS は `[5, 25]` なので，直した版で元の置き場を読んでも元の動きと同じ K を回す．
- 計算の後で直せるか：直せる（保存を読むだけの段）．
- 求める対応：作業木の直しをコミットしてから起動する．`git_commit` は `DAI/exp` の未コミットの変更で本番を止めるので，コミットしないと単位の段が始まらない．

### [I9] 起動の台本 `DAI/run/dai_k10.sh` は未追跡で，未コミットの確かめの外にある
- 重さ：記録する
- 該当箇所：`DAI/run/dai_k10.sh`（全体），`DAI/exp/dai1.py:70–75`（`git_commit` が見る場所に `DAI/run` が無い）
- 問題：台本の sha256 は `SHA16`・`SHA17` で照合するが，起動の台本そのものと，最後に流す `dai16_posthoc.py`・`dai17_posthoc.py` の sha256 は照合しない．bash は台本を少しずつ読むので，計算中に起動の台本を書き換えると，後の段が変わる．計算の後の 2 本は `posthoc_provenance.json` に `dirty` と台本の sha256 を残すので，後から照合はできる．
- 計算の後で直せるか：直せる．
- 求める対応：起動の台本と追補の第 2 版と `dai16_posthoc.py` の直しを同じコミットに入れ，そのコミットで起動する．計算中に `DAI/run/dai_k10.sh` を編集しない．

### [I9] `provenance.json` に主の K と追補の sha256 が残らない
- 重さ：記録する
- 該当箇所：`dai16_probe.py:1153–1161`，`dai17_probe.py:1293–1302`
- 問題：`KS` は残るが `K_MAIN` と追補の文書の sha256 は残らない．`K_MAIN` は `KS = (10,)` から決まり，`criteria.csv` の `main` の列にも出るので，読み手は再現できる．追補の版は `commit` から引ける．追補の第 2 版 1 節が「結果の文に追補のコミット番号を書く」と決めているので，それで足りる．
- 計算の後で直せるか：直せる．
- 求める対応：結果の文の冒頭に，追補のコミット番号を `provenance.json` の場所と並べて書く．

### [I1] `--ks` を付けずに `_k10` の置き場へ流しても止まらない
- 重さ：記録する（読んだだけ．流していない）
- 該当箇所：`dai16_probe.py:1133–1146`，`dai17_probe.py:1272–1285`
- 問題：`--ks` を付けずに `--out-dir results/dai16_probe_k10` を渡すと，`prod` は偽になる．作業木がきれいなら `--allow-dirty` なしで通り，K = 5・25 の単位をその置き場に書く．起動の台本はどの段にも `--ks 10` を付けるので，台本を通す限りこの道には入らない．手で 1 段だけ流し直すときに起こりうる．集計の段に `--ks` を付け忘れたときは，K = 10 の npz に `kpos_K5` が無いので止まる．
- 計算の後で直せるか：直せる（置き場を消して流し直す）．
- 求める対応：手で流し直すときも `--ks 10` を付ける．

当てた項目：I1，I3，I7，I8，I9，I10，I11，L4，R1，R4，S6．

## 目録に無い種類の不備

**モジュールの定数を `main` の中だけで書き換えると，その定数を読む別の台本には伝わらない．**`dai16_probe.KS` は `--ks` で書き換わるが，`import dai16_probe as P16` で読む `dai16_posthoc.py` は読み込んだときの `(5, 25)` を使い続けた．`docs/methods.md` 9.3 節の I に足す項目の案：「実行のときの引数で定数を変える台本では，その定数を読む別の台本（計算の後の量，起動の台本，検算）が，同じ回の値を出力の記録（`provenance.json` など）から読むかを確かめる．モジュールの定数を直接 import して読む箇所を grep で数える」．
