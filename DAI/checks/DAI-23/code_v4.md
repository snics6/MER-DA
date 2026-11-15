# 関門 1：実装と出力の再検査（DAI-23，2026-11-16，コミット 8c9daf5．差分 2947fc8..8c9daf5）

## 判定
判定：通す．台本の差分（`exp/dai23.py` の `KFIX_DEN_MIN` の 5 行）は設計 7 節の新しい文どおりに働き，本番の全 50 単位では 4 組（IEMOCAP/all，MOSEI/all，MELD/all，MELD/nonshared）とも分母が基準を満たすので M3_fixC の行は 1 本も落ちず，`main.csv` は 254 行のまま変わらない．`tools/collect_results.py` の `LICENSED_TEXT` は `asr_check.csv` を正しく外すが，`dai23` がまだ `DIRS` に無いので，今は dai23 の表が 1 本も取り込まれない（指摘 2．記録）．止める指摘と直してから進む指摘は無い．

検査の対象：`git diff 2947fc8 8c9daf5`．台本の変更は `exp/dai23.py` の 127 行（`KFIX_DEN_MIN = 1.0`）と 411〜415 行（分母が 1.0 点未満なら κ_fixC を NaN にし，M3_fixC を出さない），`tools/collect_results.py` の 35〜36 行と 48〜49 行（`LICENSED_TEXT`）だけである．ほかは設計（`docs/設計/実験23_設計.md`），台帳（`docs/seen-data.md`），検査の報告の 3 本である．検査の始めに `git status --porcelain DAI/exp DAI/features DAI/measure DAI/tools` は空だった（作業木は 8c9daf5 と同じ）．試しは 8c9daf5 を `git archive` で一時の置き場（scratchpad の `at8c9/DAI`．`results/dai1`・`results/dai14`・`data/cache` は読むだけのリンク）に取り出して流した．取り出した `exp/dai23.py` は `git show 8c9daf5:DAI/exp/dai23.py` と同じである．

## A．設計と台本の対応表（差分の行だけ）

| 設計の項目（8c9daf5） | 台本の箇所 | 合っているか |
|---|---|---|
| 7 節：C′ を固定した D(A ← V) が 1 点未満なら κ_fixC が定まらないので M3_fixC を出さない | exp/dai23.py:127，411，414–415 | 合う．D の単位は点（UAR の百分率の差．`RATIO_MIN` の 25.0 と同じ尺度）で，基準 1.0 は「1 点」と同じ．判定は点推定の絶対値で行う（区間ではない．指摘 1） |
| 7 節：M3 の κ は κ_fixC で `kappa` の列に残す | exp/dai23.py:419 | 合う．κ_fixC が NaN のときは M1_fixC と M2_fixC の行の `kappa` が NaN になる．M1・M2 は κ を使わないので値は変わらない（下の NaN の枝の試し） |
| 2 節：`asr_check.csv` を git に入れず公開しない（`tools/collect_results.py` の取り込みの対象から外した） | tools/collect_results.py:35–36，48–49 | `tracked()` は `dai23/asr_check.csv` を外す（偽の置き場で確かめた）．ただし `DIRS` に `dai23` が無いので，集計の 4 本（`asr_check_summary.csv`，`sample_scope.csv`，`f0_check.csv`，`fc.json`）も今は取り込まれない（指摘 2） |
| 2 節・9 節：MELD の 2 語以下の割合 18.3% | — | 前回の報告の指摘 2 のとおりに直った．台本の変更は無い |
| 9 節：「同じ割合で減る」前提を確かめていない | — | 文書だけの変更．台本の量は変わらない |
| 5 節：words3 の判定が候補の全体と違うときの書き方 | exp/dai23.py:311–312（`side_subset_verdicts.words3`） | 合う．判定の比べは `criteria.json` の `side_subset_verdicts.words3` と `main.csv` の subset = all の行から出せる |

設計の表の組の数：主の 32 行，`main.csv` 254 行，`descriptive.csv` 172 行，`assumptions.csv` 72 行は前回と同じである（下の試しと，本番の分母の確かめから）．

## A．小さく流した結果

GPU は検査の始めにほかの処理が約 1.7 GB を使っていた．試しはすべて CPU（`CUDA_VISIBLE_DEVICES=""`）で流した．

| 試し | コマンド | 所要 | 結果 |
|---|---|---|---|
| selftest | 取り出した 8c9daf5 で `exp/dai23.py --stage selftest` | 2 秒 | 「6 項目とも通った」 |
| 退化の試しの集計（8c9daf5） | `code_run_v3/degenerate/units` の 8 単位を `checks/DAI-23/code_run_v4/degenerate/units` に写し，取り出した 8c9daf5 から `exp/dai23.py --stage summary --out-dir checks/DAI-23/code_run_v4/degenerate --targets IEMOCAP MOSEI MELD --allow-partial --l-kind audio_wavlm --b-kind audio_wavlm` | 61 秒 | 正常に終わった．`main.csv` 254 行，`descriptive.csv` 172 行，`assumptions.csv` 72 行．`checks/DAI-23/code_run_v4/cmp_v3_v4.py` で 2947fc8 の集計（`code_run_v3/degenerate/summary`）と突き合わせると，3 つの表の数値の列の差の最大は 0.0，NaN の位置は同じ，`criteria.json` は同じだった．`*_fixC` は 24 行（M3_fixC 8 行），`kappa` はすべて有限，効果と区間の両端の絶対値の最大は 0.0．effect・lo・hi の NaN 0，Inf 0 |
| NaN の枝の試し | 取り出した写しの `KFIX_DEN_MIN` を 1e9 に変えた別の写し（scratchpad の `at8c9_nan`）から，同じ単位（リンク）で集計を `checks/DAI-23/code_run_v4/nan_branch` に流した | 60 秒 | 正常に終わった．`main.csv` は 246 行で，消えたのは M3_L_fixC と M3_B_fixC の 8 行（IEMOCAP/all，MOSEI/all，MELD/all，MELD/nonshared）だけで，足された行は無い．残った `*_fixC` 16 行は `kappa` が NaN，効果と区間に NaN は無い．`criteria.json` は基準 1.0 の集計と同じだった．共通の 246 行の点推定の差は 0.0，判定（verdict）はすべて同じ．ただし MOSEI の nomarker と words3 の 16 行で区間の両端が最大 0.030 点ずれた（指摘 3） |
| 本番の分母の確かめ | `checks/DAI-23/code_run_v4/verify_den.py`（実験 14 の全 50 単位に台本と同じ `fixc_cell` と `combo_stats` を当てる．値は出さず，差と基準の真偽だけを出す） | 約 4 分 | 4 組とも fb − P3b（実験 14 の R1fixC の P3b）が 1.8×10⁻¹⁵ 以下，fa − P3 が 0.0 で，`|fb| ≥ 1.0` は真，fb の区間は 0 を含まず，区間の下端も 1.0 以上だった |
| collect_results | `checks/DAI-23/code_run_v4/collect/test_tracked.py`（偽の置き場 `collect/res/dai23/` に空のファイル 9 本を置き，`RES` を差し替えて `tracked()` を呼ぶ） | 1 秒 | `dai23/asr_check.csv` は False，`asr_check_summary.csv`・`f0_check.csv`・`fc.json`・`sample_scope.csv`・`provenance.json`・`summary/main.csv`・`summary/criteria.json` は True，`units/` の下は False．`"dai23" in DIRS` は False．`py_compile` は通った |

**退化した場合の検算**：L と B に元の声の特徴を入れた 8 単位で，8c9daf5 の集計の words3 と `*_fixC` の量は 2947fc8 と同じく 0.0 である（上の表）．基準 1.0 は試しの 8 単位でも 4 組すべてで満たされた（M3_fixC 8 行がすべて出た）．

**保存済みの値の再現**：本番の全 50 単位で，台本の分母と分子は実験 14 の R1fixC の P3b と P3 に一致した（差 1.8×10⁻¹⁵ 以下）．

**見た値の申告**：画面に出したのは行の数，差，真偽，NaN の数だけにするつもりだったが，`assumptions.csv` の突き合わせの不具合を調べたとき，退化の試しの IEMOCAP（4 単位）の A の項の値（R(A ← y)，D(A ← T)，D(A ← V)．L と B には元の声を入れたので同じ値）の先頭の数行が画面に出た．これは実験 14 の保存と同じ A の予測の一部の単位の値で，言葉を消した音声の特徴から作った値ではない．本番の分母の確かめは真偽と差だけを出し，実験 14 の P3b の値は出していない．

## A．本番の規模の見積もり

単位の段は変わっていないので，前回の見積もり（150 単位で約 2.8 時間，GPU は全体で 4 GB 未満，保存は約 0.52 GB）のままである．集計の段は 8 単位で 61 秒（前回 61 秒）で，`KFIX_DEN_MIN` は所要を変えない．`tools/collect_results.py` は計算に関わらない．

## B．差分で変わった予想される指摘と，出力から答えられるか

前回の表のうち，変わった行と新しい行だけを書く．ほかの行は前回の判定のまま．

| 表・図 | 予想される指摘 | もとにした項目か例 | (a)／(b)／(c) | (b) なら足すもの | 足す費用 |
|---|---|---|---|---|---|
| 主の表（C′ を固定した版） | C′ を固定した D(A ← T) と D(A ← V) の値と区間を見せよ．κ_fixC はどれだけ揺れるか | R4，ML の査読 v2 の m2 | (a)：単位の npz（`results/dai23/units/`）と実験 14 の単位から，`D14.fixc_cell(U, ic_fix, "A", "single_T"/"single_V", "true"/"shufcl", subset)` を `combo_stats` に通せば CPU で数分で出る（`verify_den.py` と同じ手順）．MOSEI の区間は乱数の順が台本と違うので数字の末尾が少し違う．台本の `fa`・`fb` を `main.csv` の行にすれば計算を足さずに保存できる（指摘 1） | — | — |
| 主の表（C′ を固定した版） | 分母の基準を区間（0 を含むか）にしたら M3_fixC の出る組は変わるか | ML の査読 v2 の m2 | (a)：本番の 4 組は基準 1.0 と区間の規則の両方を満たす（`verify_den.py`）ので，どちらの規則でも 4 組とも M3_fixC が出る | — | — |
| 操作の確かめ | 発話ごとの書き起こしを公開せずに，一致率の区間を示せ | 感情認識の査読の a7 | (a)：`asr_check.csv` は git の外に残り，ローカルから発話を単位にした区間を作れる．公開するのは集計の 4 本 | — | — |

止めるべき (b) は無い．

## 指摘

### [R4・記録] 1．分母の基準は点推定の 1 点で判定し，D_fixC(A ← T) と D_fixC(A ← V) を行として保存しない
- 重さ：記録する
- 該当箇所：exp/dai23.py:409–411（`fa = D14.combo_stats(...)`，`fb = ...`，`kfix = fa["effect"] / fb["effect"] if abs(fb["effect"]) >= KFIX_DEN_MIN else np.nan`）
- 問題：流して確かめた．ML の査読 v2 の m2 は「D_fixC(A ← V) の区間が 0 を含むなら M3_fixC を読まない」と「`fa`，`fb` を行に並べる」を求めた．台本は点推定の絶対値と 1.0 点の比べにし，`fa`・`fb` は行に残さない．設計 7 節は「1 点未満なら」と書くので設計とは合う．本番の 4 組は分母が 1.0 点以上で区間も 0 を含まない（区間の下端も 1.0 以上）ので，どちらの規則でも結果は同じであり，基準は本番で働かない．分母の値は実験 14 の P3b と同じで，`docs/seen-data.md` によると設計者は第 2・3 版で P3b の行を見ている．基準が働かないので，見た後に基準を決めたことは M3_fixC の行の有無を変えない．
- 計算の後で直せるか：直せる（B の表の 1 行目．CPU で数分）．
- 求める対応：要るなら 411 行の後に `fa`・`fb` を `DAT_fixC`・`DAV_fixC` の行として `mains` に足す．`combo_stats` はすでに呼んでいるので乱数の順もほかの行の値も変わらない．足さないなら，結果の文で「4 組とも C′ を固定した D(A ← V) は 1 点以上で，M3_fixC はすべて出した」と書けば足りる．

### [記録] 2．`dai23` が `tools/collect_results.py` の `DIRS` に無い
- 重さ：記録する
- 該当箇所：tools/collect_results.py:15–22（`DIRS` の最後は `"dai22"`），35–36
- 問題：流して確かめた．`LICENSED_TEXT` は `dai23` を `DIRS` に足したときに初めて働く．今のままだと，設計 2 節が「入れる」とする集計の 4 本と `summary/` の表も MANIFEST に載らず，git に入らない．`.gitignore` は `results/*` を除外し，MANIFEST の行だけを `git add -f` で入れる運用なので，`asr_check.csv` が今 git に入る経路は，人が `git add -f DAI/results/dai23` のように置き場ごと足す場合だけである．
- 計算の後で直せるか：直せる．
- 求める対応：本番の後に `DIRS` に `"dai23"` を足してから `tools/collect_results.py` を流し，`results/MANIFEST.tsv` に `dai23/asr_check.csv` が無く，`results/MANIFEST_outside_git.tsv` にあることを確かめる．`git add -f` は置き場ごとではなく MANIFEST の行ごとに行う．

### [記録] 3．M3_fixC を出さない組があると，MOSEI の後ろの行の区間の乱数が変わる
- 重さ：記録する
- 該当箇所：exp/dai23.py:414–415（`if q == "M3" and not np.isfinite(kfix): continue`）
- 問題：流して確かめた．M3_fixC を飛ばすと `combo_stats` の呼び出しが減り，同じ `rng` から後に引く MOSEI の nomarker と words3 の 16 行の区間の両端が最大 0.030 点ずれた（NaN の枝の試し）．点推定と判定は変わらず，主の 32 行も変わらない．IEMOCAP と MELD の区間は乱数の順に依らない方式なので変わらない．本番では 4 組とも M3_fixC が出るので，この枝は通らない．
- 求める対応：無し．

### 前回の記録の残り
前回の指摘 3（設計 8 節の退化の試しの置き場が `code_run_v2/degenerate/` のまま．8c9daf5 の集計は `code_run_v4/degenerate/` にある），指摘 5（3 語の基準 `MIN_WORDS_W3` と `MIN_WORDS` を 2 か所に書く），指摘 6（設計 5 節の dV の行に MELD の共有を除いた版を見ることが書かれていない）は 8c9daf5 でも直っていない．どれも記録だけで，結果の値は変えない．

## 目録に無い種類の不備
- 取り込みの台本の除外の一覧（`LICENSED_TEXT`）は，対象の置き場が取り込みの一覧（`DIRS`）に入るまで働かないので，除外を足したことだけを確かめても公開の可否は確かめたことにならない．検査役は，除外を足した差分では偽の置き場で `tracked()` を呼び，あわせて置き場が `DIRS` にあるかを確かめる（R の出力の保存の項に足す案）．

## 確かめた項目
I1，I2，I9，I10，I11，R1，R4，R8．

試しの出力：`checks/DAI-23/code_run_v4/`（degenerate/，nan_branch/，collect/，summary.log，summary_nan_branch.log，検算の台本 cmp_v3_v4.py，verify_den.py，collect/test_tracked.py）．`results/` と `data/cache/` には書いていない．取り出した写しは scratchpad の `at8c9/` と `at8c9_nan/` にある．
