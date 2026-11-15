# 計算の後の点検（DAI-23，2026-11-16，コミット a410eb0）

## 判定
判定：通す．単位 150 個（3 コーパス × 5 分割 × 10 種）は欠けも重複も無く，単位・集計・特徴・fc.json のコミットはすべて a410eb0（dirty は偽）で，特徴の sha256 は単位の段と集計の段の記録と今のファイルで一致し，集計の 3 表と criteria.json・convergence.json は単位から作り直すと差 0.0 で一致した．止める指摘と直してから進む指摘は無く，記録の指摘が 6 つある．

検査の対象：本体の `results/dai23/` の全ファイル（`asr_check.csv` は列名・行数・sha256 と，作り直しの差だけを見た．書き起こしの文は画面に出していない），`logs/dai23_*.log`，計算した作業木 `/home/snics/MER-DA-wt23`（HEAD a410eb0，`exp splits measure features data/metadata` に未コミットの変更なし）．点検の台本と出力は `checks/DAI-23/code_post_run/`（`check_units.py`・`check_units.log`，`check_asr.py`・`check_asr.log`，集計の作り直し `resum/`・`resum.log`）．`results/` と `data/cache/` には書いていない．結果の主の量（M1〜M3d，L3，dV）の値は画面に出しておらず，この報告にも書かない．

## 1．格子・欠け・重複

| 確かめたこと | 結果 |
|---|---|
| 単位の数 | IEMOCAP・MOSEI・MELD とも npz 50，csv 50（見込み = `DD.N_FOLDS` 5 × 実験 1 の固定の `n_seeds_eval` 10）．欠け 0，余り 0．実験 14 の同じ 150 単位がそろう |
| 単位の csv の行 | 合計 4,950 行．r > 0 は 27 行（L・B 各 13 + A の y 1），r = 0 は IEMOCAP 63 行（別の分け方 2 × 2 × 9 = 36 を足す），MOSEI・MELD 99 行（別の分け方 4）．集計の段の行数の照らし（`exp/dai23.py` の 147〜152 行）も通っている |
| 重複 | 識別子（T, fold, seed, split, student, target, version）の重複 0．`main.csv` の（T, subset, quantity）の重複 0，`descriptive.csv` の（T, student, target, subset, quantity）の重複 0，`assumptions.csv` の重複 0．`asr_check.csv` の（corpus, condition, sample_id）の重複 0 |
| 単位の npz の鍵 | r > 0 は予測 27，格子の予測 96（2 操作 × 3 目標 × 2 版 × 格子 8），確率 27，内側の曲線 11．r = 0 は予測が別の分け方の分だけ多い．すべての予測は長さが候補の数で，値は 0〜3．sid・y・cl・half と r = 0 の althalf は実験 14 の単位と同じ．実験 14 の鍵と重なる鍵は無い |
| 集計の表の行 | `main.csv` 254 行（主の 32 行は IEMOCAP/all，MOSEI/all，MELD/all，MELD/nonshared に 8 行ずつ），`descriptive.csv` 172 行，`assumptions.csv` 72 行（C_edge 30，frac_changed 12，label_info 6，retained_share 6，pca_kept 9，eff_rank 9）．関門 1 の見込み（code_v4）と同じ．`*_fixC` は 24 行で，M3_fixC 8 行も出ている |

## 2．NaN・Inf・あり得ない値

- `main.csv`・`descriptive.csv` の effect・lo・hi・se に NaN・Inf は 0．lo ≤ effect ≤ hi はすべての行で成り立つ．`descriptive.csv` の R（R，R_shuf，R_shufcl）はすべて 0〜100 の中．NaN があるのは設計どおりの列だけである（`v_split`・`v_split_indep` は分け方の揺れを足さない記述の行と fixC の行，`df` は MOSEI の引き直しの行，`fixC0`・`fixC1` は fixC 以外の行）．
- 単位の csv の数値の列で NaN があるのは `verify_agree_*` だけで，r > 0 の 135 単位の行（設計どおり照合は r = 0 だけ）．agree_halfmean と agree_pooled はすべて 0〜100 の中．frac_changed の NaN は入れ替えた目標の 2,400 行のうち 0．
- 確率（`prob__*`）はすべて有限で，和は 1（差 1e-4 以内），負の値は無い．argmax と保存した予測は，4,050 本 × 発話のうち MOSEI_f1_r9 の `prob__K5__main__R1__L__single_V__shuf` の 1 発話だけ違う．この発話は上位 2 クラスの確率がちょうど同じ（差 0.0）で，実験 14 の MOSEI の 10 単位にも同じ種類の不一致が 1 件ある（指摘 3）．
- 特徴の npz（lp300・rev × 3 コーパス）の X は 1024 次元で，ok の行に NaN・Inf は 0．`sample_id` の並びと `ok` は元の声の `audio_wavlm_<コーパス>.npz` と同じ（MELD の音声を取り出せない 1 発話は元の声でも ok が偽）．
- 収束（`convergence.json`）：勾配の最大 1.0e-6，内側の勾配の最大 1.0e-6，止まった数 0．単位の csv の stalled・inner_stalled も 0．
- `f0_check.csv` の NaN 6 個は，IEMOCAP の性別ごとの行（F，M）の `utt_share_any_above_0.8x{500,400,300}` で，台本がこの列を性別ごとに作らない（指摘 2）．`asr_check_summary.csv` と `sample_scope.csv` に NaN は無い．

## 3．コミットと sha256

| 確かめたこと | 結果 |
|---|---|
| 単位の csv の commit | 150 単位すべて a410eb03…，dirty は偽．`kinds` は 1 種類（L = `audio_wavlm-lp300`，B = `audio_wavlm-rev`）．`dai14_commit` は fbd6c07 の 1 つ |
| 集計の 3 表の commit | すべて a410eb03…，dirty は偽 |
| `provenance_log.jsonl` | 2 行（units 02:28:53，summary 06:38:53）とも a410eb0，dirty は偽．2 行の `cache_sha256` と `scripts` は同じ |
| 特徴の sha256 | `provenance.json` の `cache_sha256` の 15 本（lp300・rev・元の声・顔・言葉 × 3 コーパス）は今の `data/cache/` のファイルと一致 |
| 台本の sha256 | `scripts` の 10 本（`exp/` の 7 本と `features/` の 3 本）は a410eb0 の作業木のファイルと一致．設計の sha256 も a410eb0 の設計と一致 |
| fc.json | `code_commit` a410eb0，`code_dirty` 偽．sha256 は `provenance.json` の `fc_json_sha256` と一致．`asr_check_sha256` は今の `asr_check.csv` と一致 |
| 特徴の npz の記録 | 6 本とも `code_commit` a410eb0，`code_dirty` 偽，`model_revision` c1423ed…，`op`，`lowpass_fc`・`lowpass_order`・`filter`（lp は `butter4_sosfiltfilt_zero_phase`，rev は空）を発話ごとに持つ．`data/cache/` に試しの名前（`_trial`）や規則で選んでいない fc の特徴は残っていない |
| 関門 1 との差 | 関門 1（code_v4）は 8c9daf5 を検査した．8c9daf5..a410eb0 の差は `tools/collect_results.py` の `DIRS` に `dai23` を足した 1 行と，起動の台本 `run/dai23.sh` と報告だけで，`exp/`・`features/`・`measure/` は同じ |

## 4．集計を単位から作り直せるか

- `exp/dai23.py --stage summary` を a410eb0 の作業木から，単位へのリンクを置いた `checks/DAI-23/code_post_run/resum/` に向けて CPU で流した（`--l-kind audio_wavlm --b-kind audio_wavlm` は本番でない置き場で台本が求める引数で，集計の段は記録にしか使わない）．約 8 分．
- `main.csv`・`descriptive.csv`・`assumptions.csv` は形と列が同じで，数値の列の差の最大は 0.0，NaN の位置は同じ，文字の列も同じ（`assumptions.csv` の student・target の不一致に見えた 42 個は両方とも NaN）．`criteria.json` と `convergence.json` は完全に一致した．
- `criteria.json` の κ の 7 個は設計 3 節の値と小数 3 桁で一致し，`main.csv` の M3 の行の `kappa` と同じ．words3 の κ は 3 コーパスとも有限．
- 操作の確かめ（`checks/DAI-23/code_post_run/check_asr.py`）：`asr_check.csv` の 1,500 行（3 コーパス × 5 条件 × 100 本．5 条件の標本の sample_id は同じ．参照の単語の数の最小は 3）の hit・wer・n_ref・n_hyp を，発話一覧の人の書き起こしと `check_lowpass_asr.norm_words`・`lcs`・`edit_distance` で作り直した差の最大は 7.1e-15 以下．`asr_check_summary.csv` の hit_mean，wer_mean，偶然の水準，超過も差 2.2e-16 以下で作り直せた．fc.json の `fc`・`rule_met`・`rev_ok` は作り直した超過に 2 節の規則を当てた結果と同じ．選んだ fc と逆再生の超過で 0.08〜0.12 に入るものは 0 個なので，2 節の「0.10 に近い」の書き方は使わない．

## 5．設計 6 節の仮定の表が出力に入っているか

| 仮定 | 出力の場所 | 入っているか |
|---|---|---|
| 操作で言葉を聞き取れなくなった | `fc.json`（fc，rule_met，rev_ok，超過の基準），`asr_check_summary.csv`，`criteria.json` の `assumptions` | 入っている．結果の文は 2 節の分かれ（rule_met の値に応じた書き方）に従う |
| 基本周波数が残る | `f0_check.csv`（0.8·fc を超える有声の区間の割合を fc ごと．IEMOCAP は性別ごと） | 入っている．0.10 の判定は `criteria.json` に無く，この表の選んだ fc の列から出す（(a)）．発話の単位の割合は性別ごとに無い（指摘 2） |
| 流れが元の特徴を作り直す | `logs/dai23_identity.log` だけ（64 発話 × 3 コーパスの差の最大 4.19e-5，3.28e-5，6.64e-5．基準 2 × 10⁻³．ok の一致 64/64） | 結果の置き場と `provenance.json` に無い．ログは git の外（指摘 1） |
| 感情の情報が残る | `assumptions.csv` の label_info（R_X_label，R_A_label，ratio，flag）と `criteria.json` の `label_info_flagged` | 入っている |
| 顔の判断の D に減る余地 | 実験 14 の P3b（計算の前に確かめた）．`main.csv` の DAV と dV | 入っている |
| D の取りうる幅 | `assumptions.csv` の frac_changed | 入っている |
| 同じラベルの中の強さ | 測らない（設計どおり） | — |
| R1 の調整がそろう | `assumptions.csv` の C_edge（share_low，share_high，share_low_nonplateau，flag） | 入っている |
| 実験 14 の保存を作り直せる | 入れ替えた目標の作り直しは全 150 単位で台本が照らし，違えば止める（`exp/dai23.py:181–185`．止まらなかった）．元の声で J_T・J_V を学び直した予測の一致の割合は r = 0 の 15 単位の csv の `verify_agree_single_T`・`verify_agree_single_V` にあり，15 単位とも 1.0（基準 0.995） | 入っている．ただし集計の表には無く，単位の csv から数える（(a)．指摘 4） |
| WavLM に言葉の手がかりが残らない | 測らない（設計どおり．限界に書く） | — |

退化の試しとの関係：
- 退化の試しの単位（`checks/DAI-23/code_run_v2/degenerate/units/` の 8 単位．commit 9529e23 と d21b840）の単位の段のコードは，本番の a410eb0 と同じである（1fb89ff..a410eb0 の `exp/dai23.py` の差は `words_of`・`KFIX_DEN_MIN` の定義と集計の段だけで，`run_unit` と共通の部品 `dai14.py`・`dai1.py`・`dai3.py`・`dai_data.py`・`dai_solver.py`・`dai1_stats.py` の差は 0 行）．
- 退化の試しと本番で同じ流れの A の y の予測（`K5__main__R1__A__label__true`），確率，内側の曲線は，IEMOCAP_f0_r0・MOSEI_f0_r0・MELD_f0_r0 で完全に一致した（予測は全一致，確率と曲線の差は 0.0）．単位の段は決定的である．
- 退化の試しの集計は 8c9daf5 で `checks/DAI-23/code_run_v4/degenerate/summary/` に作り直されている（集計の段は 8c9daf5 と a410eb0 で同じ）．設計 8 節はまだ `code_run_v2/degenerate/` を指す．`checks/` の `code_run*` は git に入らない（指摘 5）．

## 6．ログ

- 例外・Traceback・メモリ不足からのやり直しは無い．`logs/dai23.failed` は無く，`logs/dai23.done` と段ごとの `.done` 6 個がある．
- 警告は，WavLM の読み込みの既知の重みの名前の表示と，torch の `key_padding_mask` の型の非推奨の警告（特徴の段で各塊 1 回），Whisper の `forced_decoder_ids` と注意の mask の表示（操作の確かめ）だけで，値に関わるものは無い．
- 特徴の段は lp300・rev とも IEMOCAP 5,531，MOSEI 18,919，MELD 11,352 発話で，失敗は MELD の 1 発話（音声を取り出せない．元の声でも ok が偽）．60 秒で切った発話は MOSEI 8 個，MELD 1 個（両方の操作で同じ数）．
- 所要：単位の段は 150 単位の合計 14,968 秒（約 4.2 時間．1 単位の最大 266 秒）で，関門 1 の見込み 2.8 時間と設計の 2〜3 時間より長い．操作の確かめは約 66 分（見込み 20〜40 分），特徴の 2 つは約 52 分（見込み 1〜1.5 時間），集計は約 8 分．

## 指摘

### [R・記録] 1．流れの照合の結果がログにしか無い
- 重さ：記録する
- 該当箇所：`run/dai23.sh:28–31`（`--check-identity > logs/dai23_identity.log`），`features/build_audio_lowpass.py:115–121`
- 問題：設計 6 節の「流れが元の特徴を作り直す」の数値（差の最大 4.19e-5，3.28e-5，6.64e-5．基準 2 × 10⁻³）は `logs/dai23_identity.log` だけにあり，`results/dai23/` と `provenance.json` に無い．`logs/` は git の外なので，結果の文がこの数値を引くと，git に入る記録から根拠をたどれない．
- 計算の後で直せるか：直せる（ログから写すだけ）．
- 求める対応：結果の文に 3 コーパスの差の最大とログの場所を書く．あわせて，ログの 3 行を `results/dai23/identity_check.txt` のような集計の置き場に写して `tools/collect_results.py` の対象に入れるかを決める．

### [記録] 2．`f0_check.csv` の発話の単位の割合が性別ごとに無い
- 重さ：記録する
- 該当箇所：`features/check_lowpass_asr.py:190–196`（性別ごとの行は `share_above_*` だけを作り，`utt_share_any_above_*` は sex = all の行だけ）
- 問題：IEMOCAP の F・M の行の `utt_share_any_above_0.8x{500,400,300}` の 6 個が NaN である．設計 6 節の基準は区間の割合（`share_above_*`）で，それは性別ごとにもあるので，判定は出せる．
- 計算の後で直せるか：直せる（判定に使わない列）．
- 求める対応：結果の文で性別ごとに書くのは `share_above_*` だけにする．

### [記録] 3．確率の同点の発話が 1 つあり，argmax と予測が違う
- 重さ：記録する
- 該当箇所：`results/dai23/units/MOSEI_f1_r9.npz` の `prob__K5__main__R1__L__single_V__shuf`
- 問題：流して確かめた．1 発話で上位 2 クラスの確率が同じ（差 0.0）で，保存した予測は確率の argmax と違う．実験 14 の保存にも同じ種類の不一致が MOSEI の 10 単位で 1 件ある．集計は保存した予測だけを使うので，結果は変わらない．
- 計算の後で直せるか：直せる（直す必要は無い）．
- 求める対応：確率から予測を作り直す後の解析では，同点を保存の予測に合わせる．

### [記録] 4．実験 14 の作り直しの一致の割合が集計の表に無い
- 重さ：記録する
- 該当箇所：`exp/dai23.py` の `stage_summary`（`assumptions.csv` に `verify_agree` の行を作らない）
- 問題：設計 6 節は一致の割合を単位の csv に残すと書き，台本はそのとおりにしている．集計の表には無いので，結果の文の数値は単位の csv から数える必要がある．15 単位とも J_T・J_V の両方で 1.0 である（`checks/DAI-23/code_post_run/check_units.log`）．
- 計算の後で直せるか：直せる（(a)：`results/dai23/units/*_r0.csv` の `verify_agree_single_T`・`verify_agree_single_V` の最初の行を 15 単位で並べる）．
- 求める対応：結果の文に「r = 0 の 15 単位で 15/15 が 1.0」と書き，M1 の検出に要る大きさ（設計 4 節）と並べる．

### [記録] 5．退化の試しの置き場が設計と食い違い，git の外にある
- 重さ：記録する
- 該当箇所：設計 8 節（`checks/DAI-23/code_run_v2/degenerate/`），`checks/DAI-23/code_run_v4/degenerate/summary/`
- 問題：関門 1 の前回の指摘 3 の残りである．退化の試しの単位は `code_run_v2/degenerate/units/`（9529e23，d21b840），今の集計の段で作った集計は `code_run_v4/degenerate/summary/`（8c9daf5．集計の段は a410eb0 と同じ）にある．どちらも `checks/` の試走の出力で git に入らない．単位の段が本番と同じコードで，A の予測が本番と完全に一致することは 5 節で確かめた．
- 計算の後で直せるか：直せる．
- 求める対応：結果の文で退化の試しの M1〜M3d を引くときは，単位を `code_run_v2/degenerate/units/`（9529e23，d21b840），集計を `code_run_v4/degenerate/summary/`（8c9daf5）と書く．git に残すなら，集計の 3 表と criteria.json を `results/dai23/degenerate/` に写して取り込みの対象に入れる．

### [記録] 6．`tools/collect_results.py` をまだ流していない
- 重さ：記録する
- 該当箇所：`results/MANIFEST.tsv`，`results/MANIFEST_outside_git.tsv`（どちらにも dai23 の行が 0）
- 問題：a410eb0 の `DIRS` には `dai23` が入っており，`tracked()` を今の置き場で呼ぶと，`asr_check.csv` は偽，`units/` の 300 本は偽，`asr_check_summary.csv`・`f0_check.csv`・`fc.json`・`sample_scope.csv`・`provenance.json`・`provenance_log.jsonl`・`summary/` の 5 本は真になった（`.done` の 2 本は偽）．`fc.json` と `provenance.json`（`fc_json` を含む）は集計の値だけで，発話ごとの書き起こしを含まない（942 バイトと 4,421 バイト）．
- 計算の後で直せるか：直せる．
- 求める対応：コミットの前に `tools/collect_results.py` を流し，`results/MANIFEST.tsv` に `dai23/asr_check.csv` が無く，`MANIFEST_outside_git.tsv` にあることを確かめてから，MANIFEST の行ごとに `git add -f` する．

## 目録に無い種類の不備
- 前提の照合（流れの照合など）の数値を起動の台本がログにだけ書くと，結果の置き場の記録と git から根拠をたどれない．9.3 節の R（出力の保存）に「計算の前の照合の段の数値は，結果の置き場の集計のファイルか provenance に残す」を足す案．

## 確かめた項目
欠けと重複，NaN・Inf・範囲，確率の和，コミットと dirty，特徴・台本・設計・fc.json・asr_check.csv の sha256，関門 1 と本番のコミットの差，集計の作り直し，操作の確かめの作り直し，仮定の表の出力，退化の試しとの関係，ログの例外と警告，collect_results の対象．
