# 関門 1：実装と出力（DAI-16 の計算の後の台本 `DAI/exp/dai16_posthoc.py`，2026-11-14，コミット c507ee7）

## 判定
通す．台本は元の台本 `dai16_probe.py` の `tukey_fit`・`noint`・`combine`・`match_shift`・`mix_rel`・`to_w` と集計の段の `level`・`effect_stats` をそのまま呼び，作り直した `M_F6_*`・`M_F6n_*` は保存の判定と全単位で一致し，水準と差の点推定は `summary/` と 0.0 の差で一致した．直す必要のある誤りは無く，下の 6 つは記録である．

検査の範囲：台本の全行（138 行）と，呼んでいる `dai16_probe.py`（84〜222 行，406〜418 行，513〜689 行，728〜1005 行），`dai12_probe.py` の `_add`・`_finish`，`dai1_stats.py` の `counts`・`uar_full`・`g_eff_of`・`effect_stats` を読んだ．設計 `DAI/docs/実験16_下調べ_設計.md`（第 6 版）の 3.1・3.3・3.5・4 節と，`probe_v5.md` の記録 3，`review_affect_v5.md` の a1・a2，`claim.md` の A5・A7・A12 と照らした．`results/dai1/eval/` は開いていない．`results/dai16_probe/` の既存のファイルは読むだけで，書き換えていない．UAR の値と差の符号は報告に書かない．

## A．設計と台本の対応表

| 設計の項目 | 台本の箇所 | 合っているか |
|---|---|---|
| 足し算の当てはめ「全体の平均 ＋ 行の差 ＋ 列の差」（設計 139 行，第 4 版の量） | `dai16_posthoc.py:41-43`（`add_fit`） | 合う．行の平均 ＋ 列の平均 − 全体の平均で，残りの行の平均と列の平均は 2 × 10⁻¹⁶ 以下（検算 5） |
| 足し算だけを引いた残りの相関（出どころ `home`・`ext`・`away`・`K` 対 `orc`，M の組） | `:70-75` | 合う．`D_M_K_K{K}`・`D_M_orc_K{K}`・`SR_M_*` は単位の段（`dai16_probe.py:564-571`）が計算に使った表そのものである |
| 掛け算の項まで引いた相関（R0 の主の量）を並べる | `:73`（`P16.dcenter`） | 合う．単位の csv の `intcorr_M_<s>`（`dai16_probe.py:670`）と 960 行すべてで差 0.0，NaN の位置も一致（検算 1） |
| 足し算だけの対照 `M_F6add`：重み＝足し算の当てはめ（負は 0） | `:86`（`np.maximum(0.0, ad)`，`ad = add_fit(W)`，`W = P16.to_w(D[s])`） | 合う．`M_F6n` の作り（`build_specs` の `noint(to_w(D))`，`dai16_probe.py:458-463`）と同じ順序（to_w の後に当てはめ，当てはめた後は割り直さない） |
| まとめ方 F6 | `:87`（`P16.combine("F6", …)`） | 合う．点は `P16.centered` で作り，`View.LM` と同じ（`dai16_probe.py:426`） |
| adj の層：候補の上の確率の平均を積の平均にそろえる | `:78`（`target`），`:88-89`（`match_shift`，`(St + dl).argmax`） | 合う．`target` は `pool.q1.mean(0)`（`dai16_probe.py:581`）と同じ式．保存は float32 だが判定の一致は 1.0000 |
| `mixsel` の点：c を評価のターゲットにした単位で選んだ格子の点を，c を仮のターゲットにした単位に当てる | `:54-55`，`:66-69` | 合う．`mixsel.csv` の `M_F6_mixsel` の 8 行の `chosen` の番号は `n0`・`beta`・`prior` の列と一致し，保存の `grid` の文字列は `P16.GRID` の 16 点と一致した（検算 6）．`mix_rel` に渡す本数は 4K で単位の段（`:569`）と同じ |
| `mixsel` の保存の判定との突き合わせ | `:91` | 合う．`M_F6_mixsel` → `pred_adj_K{K}_M_F6_mix<i>`，`M_F6n_mixsel` → `…_M_F6n_mix<i>`（集計の段の `TIED` と同じ対応）．`M_F6add_*` は保存が無いので NaN |
| 作り直しの検算（保存の判定との一致の割合） | `:92`，`:124`，`:128` | 合う．8 版すべてで最小 1.0000（`posthoc_provenance.json` の `agree_saved_min`） |
| 水準：同じ (分割, 種) の 3 つの T を塊ごとに平均し，分割ごとに平均してから全体を平均 | `:94-96`（`P12._add`・`_finish`），`:105-108` | 合う．集計の段 `dai16_probe.py:846`，`:857`，`:860-861` と同じ関数と同じ式．`summary/levels.csv` の 72 行（M_F6・M_F6n の 4 出どころと prod）と差 0.0，`n_fr` は全行 10（検算 2） |
| 区間：`ST.effect_stats`（AFEW・IEMOCAP・MELD は jackknife ＋ 種，MOSEI は塊の引き直し ＋ 種），`geff` は `ST.g_eff_of(c)` | `:110-119` | 合う．集計の段 `:858`，`:877` と同じ．`summary/effects.csv` と重なる 32 行（`M_F6_s − M_F6n_s`，adj）で点推定の差 0.0，AFEW・IEMOCAP・MELD の端の差 0.0．MOSEI の端は乱数の鍵が違うので最大 0.0039 ずれ，判定の語は 32 行すべて同じ（検算 3．記録 1） |
| 判定の語（下端 > 0 かつ 1.0 点以上 など） | `:119`（`P16.verdict`） | 合う |
| 負を 0 に切った単位の数（`probe_v5.md` 記録 3） | `:83-85`，`:121-123` | 合う．`neg_tukey` は `noint` が切る条件 `tukey_fit(W) < 0` と同じ．出どころは `orc`・`ext`・`K`・`mixsel` の 4 つで，c × K × 出どころの 32 行，分母 `n_units` は全行 30（検算 4）．`M_F3n` の `mixsel` の点は数えない（記録 4） |
| 出力 5 本と provenance | `:97-130` | 合う．下の行の数を参照 |

設計の表に載る組の数え：差は c 4 × K 2 × 出どころ 4 × 対 4 ＝ 128 行，水準は c 4 × K 2 × 版 13（出どころ 4 × 3 版 ＋ prod）＝ 104 行，相関は c 4 × K 2 × 出どころ 4 ＝ 32 行，単位の行は 120 単位 × K 2 × (相関 4 ＋ 版 12) ＝ 3,840 行．台本の出力はすべてこの数で，欠けと重複は無い．設計 139 行は形（F6 か F3 か）と層を書いていない．台本は F6 と adj の層だけを出す（記録 3）．

## A．小さく流した結果

`results/dai16_probe/units` を指す記号リンクと `summary/mixsel.csv` の写しを `/tmp/claude-1000/-home-snics-MER-DA/76085a37-4610-4922-b8ae-ced34f163c39/scratchpad/ph16/` に置き，GPU を見せずに流した．

- `CUDA_VISIBLE_DEVICES="" ./.venv/bin/python -W default DAI/exp/dai16_posthoc.py --out-dir <写し> --limit 6`：5.5 秒，終了 0．
- 同じく 120 単位すべて：1 分 47 秒（CPU），終了 0．警告は `posthoc_provenance.json` を閉じずに書く `ResourceWarning` 1 件だけ（記録 5）．収束しない警告と NaN の警告は無い．
- 出力の行の数：`additive_units.csv` 3,840，`additive_corr.csv` 32，`additive_levels.csv` 104，`additive_effects.csv` 128，`additive_clip.csv` 32．
- NaN：`corr_add` 0，`corr_tukey` 0，`uar_adj` 0，`agree_saved`（M_F6 と M_F6n）0，差の表の `est`・`lo`・`hi` 0．
- **既存の出力との一致**：私の写しで流した 5 本の csv は，`results/dai16_probe/posthoc/` の 5 本とバイト単位で同一だった（`cmp`）．私の流した版の provenance は コミット c507ee7，`dirty` 偽，台本の sha256 `8263b1b1…` である．既存の provenance はコミット be6d370，`dirty` 真で，台本の sha256 は同じ `8263b1b1…` である．
- **退化した場合の検算**（`verify.py` の 5）：足し算の形の人工の表では `add_fit` と `tukey_fit` が表と 4.4 × 10⁻¹⁶ 以内で一致した．保存した単位 1 つの上で，β = 0 の格子の 8 点の重みは足し算の形で，`add_fit`・`tukey_fit` と 10⁻¹² 以内で一致した（このとき `M_F6add` と `M_F6n` と `M_F6` は同じ判定になる）．今回の `M_F6_mixsel` の 8 つの選んだ点はすべて β = 1 なので，この退化は本番の行には現れない．
- **保存済みの値の再現**：作り直した `M_F6_*`・`M_F6n_*`（4 出どころ × 2 版）の判定は保存の判定と全単位・全 K で一致（最小 1.0000）．水準は `summary/levels.csv` と差 0.0，差の点推定は `summary/effects.csv` と差 0.0．
- 同じ単位と同じ (分割, 種) の突き合わせ：差を作る 2 つの版の (分割, 種) の組がそろわないときは行を飛ばす作りだが（記録 2），128 行すべてで `n_fr` は 10 だった．
- 検算の台本と記録：`DAI/checks/DAI-16/code_run_posthoc/verify.py`，`DAI/checks/DAI-16/code_run_posthoc/verify.log`（値は出さず，差と数だけ）．

## A．本番の規模の見積もり

120 単位で 1 分 47 秒（CPU，2 コアぶん），メモリは小さい（単位の npz を 1 本ずつ読む）．GPU の計算は無い．ただし `dai_solver.py:27-29` は読み込んだだけで CUDA があれば `set_per_process_memory_fraction` を呼び，GPU に文脈を作る．GPU を使わないなら `CUDA_VISIBLE_DEVICES=""` を付けて流す（私の流した版は付けた）．

## B．予想される指摘と，出力から答えられるか

| 表・図 | 予想される指摘 | もとにした項目か例 | (a)／(b)／(c) | (b) なら足すもの | 足す費用 |
|---|---|---|---|---|---|
| 足し算だけの対照の UAR | 研究者の形 F3 にも足し算だけの対照を作り，`M_F3 − M_F3add` と `M_F3 − M_F3n` を並べよ | 設計 143 行（F6 と F3 の両方に帰属の規則），`review_affect_v5.md` a1 直し方 4 | (a)．学び直しは要らない．`combine("F3", L, Wv, q1)` の `q1` は `np.exp(norm_log(LM.sum(0)))` を候補とテストで作れば単位の段と同じ（`View.q1`）．保存の `pred_adj_K{K}_M_F3_*`・`M_F3n_*` で作り直しを検算できる．`mixsel` は `M_F3_mixsel` の `chosen`（F6 と違う点が 5 行，β = 0 が 2 行）を使う | 台本に F3 の 3 版を足す | CPU 約 1 分の追加 |
| 同上 | 当てはめた対照の重み（足し算と掛け算の両方）で，最も強い分類器が感情によって替わる単位の数を並べよ | `review_affect_v5.md` a1 直し方 3，`claim.md` A5 | (a)．`W`，`tukey_fit(W)`，`add_fit(W)` は台本の 82-83 行で既に作っている．`(fit.argmax(0) != fit.argmax(0)[0]).any()` を `clip` の行に足せば出る．今の出力には無い | `additive_clip.csv` に 2 列 | ほぼ 0 |
| R0 と足し算だけの相関 | 掛け算の項の λ と，掛け算の項が残りに占める割合を並べよ | `review_affect_v5.md` a2 | (a)．保存した表に `tukey_fit` の式を当てれば出る．今の出力には無い | `additive_units.csv` に λ と割合の列 | ほぼ 0 |
| `additive_corr.csv` | 足し算だけを引いた相関にも，感情の並びを 24 通りに入れ替えた偶然の水準を付けよ | 設計 136 行（R0 の主の量には `intperm` がある） | (a)．`perm_share` の `dcenter` を足し算の残りに替えた関数で出る | 1 列 | 1〜2 分 |
| 足し算だけの対照の差 | adj 以外の層（kb・raw・ob）でも並べよ（層で符号が変わりうる） | `review_ml_v5.md` m16，設計 3.4 節，集計の段は `M_F6 − M_F6n` を ob でも出す | (a)．raw は `St.argmax`．ob は `fit_wb(Sp[nk][None], ynk, …)`（`y_pool`・`kpos` は保存）．kb は `pcf_*` と `kfold` から `held` の点を作り `fit_wb` に渡す．`fit_wb` は numpy の関数で CPU で解ける | 台本に層を足す | 数分 |
| 足し算だけの対照 | 感情ごとの再現率と判定の割合を並べよ | 設計 4 節の `per_emotion.csv` の扱い | (a)．ただし `M_F6add` の発話ごとの判定は保存していないので，台本を流し直す（2 分） | 判定を npz に保存するか，感情ごとの数を csv に足す | 数 MB |
| 足し算だけの対照 | 候補の上の UAR（K 本を除く）でも並べよ | 設計 3.7 節の副の量 | (a)．`(Sp + dl)[nk].argmax` で出る | 1 列 | ほぼ 0 |
| 差の数え | AFEW を除く 3 コーパスで数えよ | 設計 146 行 | (a)．`additive_effects.csv` の `c` と `verdict` から数える | なし | なし |
| 差の区間 | 話者（塊）を単位にした区間か | 9.3 節 S，過去の指摘 | (a)．`effect_stats` は塊の jackknife／引き直しで作る（元の集計と同じ） | なし | なし |
| `mixsel` の対照 | 選んだ点が相性を使わない（β = 0）なら帰属の数えから外したか | 設計 114 行，143 行 | (a)．`mixsel.csv` の `beta` で確かめる．`M_F6_mixsel` の選んだ点は 8 行すべて β = 1 | なし（`beta` を差の表に写すと読みやすい） | なし |

どの行も学び直しを要さず，保存した単位の出力から CPU で数分以内に出せる．「止める」に当たる行は無い．

## 指摘

### [記録 1] 差の区間の乱数の鍵が集計の段と違い，MOSEI の `M_F6_s − M_F6n_s` の端が `summary/effects.csv` と最大 0.0039 ずれる
- 重さ：記録する
- 該当箇所：`dai16_posthoc.py:117`（`P16.row_rng("posthoc", c, K, a, b)`）．集計の段は `dai16_probe.py:877`（`row_rng(c, K, a, b, w)`）
- 問題：同じ対の区間が 2 つのファイルで違う値になる．流して確かめた．MOSEI の 8 行で `lo`・`hi` の差の最大 0.0039，判定の語は 32 行すべて同じ．AFEW・IEMOCAP・MELD は jackknife なので差 0.0．
- 計算の後で直せるか：直せる
- 求める対応：結果の文で `M_F6 − M_F6n` を引くときは `summary/effects.csv` の値を使い，`additive_effects.csv` からは `M_F6add` を含む対だけを引く．台本を直すなら，重なる対は鍵を `(c, K, a, b, "adj")` にして集計の段と同じ値にする．

### [記録 2] 2 つの版の (分割, 種) がそろわないとき，行を黙って飛ばす
- 重さ：記録する
- 該当箇所：`dai16_posthoc.py:115`（`if A is None or B is None or set(A) != set(B): continue`）．集計の段 `dai16_probe.py:875-876` は止める
- 問題：欠けがあっても出力から分からない．今回は 128 行すべてそろい，`n_fr` は全行 10 だった（流して確かめた）．
- 計算の後で直せるか：直せる
- 求める対応：`set(A) != set(B)` のときは `SystemExit` にする．

### [記録 3] 足し算だけの対照は F6 と adj の層だけで，F3，ほかの層，最も強い分類器の入れ替わり，λ を出さない
- 重さ：記録する（設計 139 行は形と層を指定していない．計算の後に CPU で出せる）
- 該当箇所：`dai16_posthoc.py:79-94`
- 問題：設計 143 行の帰属の規則は F6 と F3 の両方に当たる．`review_affect_v5.md` a1 の直し方 3（当てはめた表で最も強い分類器が感情によって替わるか）と a2（λ と掛け算の項の割合）の量はこの台本に無い．`claim.md` A5 は，この量が無いために下書きの文が言える範囲を超えると指摘している．
- 計算の後で直せるか：直せる（B の表の 1〜3 行目の手順．合わせて CPU で数分）
- 求める対応：結果の文で「掛け算の項が順の入れ替わりを吸い込んだか」に触れるなら，その前に B の表の 2・3 行目の列を足して流す．研究者の形 F3 の帰属を書くなら，F3 の足し算だけの対照も足す．触れないなら，文を相関の比べで言える範囲に限る．

### [記録 4] 負を 0 に切った数は F6 の選んだ点についてだけで，`M_F3n_mixsel` の選んだ点の数が無い
- 重さ：記録する
- 該当箇所：`dai16_posthoc.py:66-69`，`:79-85`（`mixsel` は `M_F6_mixsel` の `chosen` だけを読む）
- 問題：`M_F3n_orc`・`M_F3n_ext`・`M_F3n_K` は `M_F6n_*` と同じ `noint(W)` なので数は同じだが，`M_F3_mixsel` の選んだ点は 8 行のうち 5 行で F6 と違う．`probe_v5.md` 記録 3 が求める「得を何に帰すかを書くとき，対照の重みに 0 があった単位の数を添える」を F3 の `mixsel` に当てるには足りない．
- 計算の後で直せるか：直せる
- 求める対応：F3 の `mixsel` の帰属を書くなら，`M_F3_mixsel` の `chosen` でも `clip` の行を作る．

### [記録 5] provenance：既存の出力は dirty の作業木で作られ，依存する台本と入力の sha256 が無い
- 重さ：記録する
- 該当箇所：`results/dai16_probe/posthoc/posthoc_provenance.json`（コミット be6d370，`dirty` 真），`dai16_posthoc.py:125-129`
- 問題：既存の 5 本の csv は，私がコミット c507ee7 の作業木（`dirty` 偽）で流した 5 本とバイト単位で同一だったので，数は変わらない．ただし既存の provenance は dirty のままである．provenance は台本自身の sha256 だけを持ち，式の本体がある `dai16_probe.py`・`dai1_stats.py`・`dai12_probe.py` と入力 `summary/mixsel.csv` の sha256 を持たない．`units_commit` は 1 本目の単位の csv だけから取る．`json.dump(…, open(…))` はファイルを閉じず，`ResourceWarning` が出る（CPython では関数を出たときに閉じるので中身は書かれる）．
- 計算の後で直せるか：直せる
- 求める対応：c507ee7（dirty 偽）で `results/dai16_probe/` に流し直し，provenance の `dirty` が偽になったことを確かめる（所要 2 分．数は変わらないことを確かめ済み）．台本を直すときに，依存する 3 本と `mixsel.csv` の sha256 を足し，`units_commit` を全単位から取り，`with open(…)` で書く．GPU を使わない約束を守るには `CUDA_VISIBLE_DEVICES=""` を付ける（A の見積もり）．

### [記録 6] 使っていない行がある
- 重さ：記録する
- 該当箇所：`dai16_posthoc.py:90`（`name = v if s != "mixsel" else v` は両方とも `v`），`:38`（`SRC` は使われない）
- 問題：結果には影響しない．
- 計算の後で直せるか：直せる
- 求める対応：直すときに消す．

当てた項目：9.3 節の I1〜I11（式の一致，地図や出力の場所の固定，同名の台本の読み込み（`sys.path` の先頭は `DAI/exp` で，`dai16_probe` は同じフォルダのもの），乱数，出力の欠けと重複，保存の値の再現，退化の検算，NaN，provenance），R（B の表）．

## 目録に無い種類の不備

- 案：「R 群：計算の後に出す台本が，元の集計の段と同じ対の差を作り直すときは，乱数の鍵を元と同じにするか，結果の文で引く出どころのファイルを 1 つに決める（DAI-16 の計算の後の台本．MOSEI の区間の端が 2 つのファイルで 0.004 違った）」．
