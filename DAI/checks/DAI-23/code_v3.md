# 関門 1：実装と出力の再検査（DAI-23，2026-11-16，コミット 2947fc8．差分 1fb89ff..2947fc8）

## 判定
判定：通す．差分の 7 項目（κ_fixC，words3，MELD の顔の判断の再現の増えの数え，norm_words，sample_scope.csv，F0 の性別ごとの行，code_dirty）はどれも設計の第 4 版どおりに計算し，退化の試しで words3 の M1・M2・M3・M3d・dV と `*_fixC` の 24 行はすべて 0.0 になり，κ_fixC の作りは実験 14 の R1fixC の P3 ÷ P3b と 50 単位で差 2.2×10⁻¹⁶ 以下で一致した．止める指摘と直してから進む指摘は無い．ただし，検査の間に別のセッションが `exp/dai23.py` に未コミットの変更（`KFIX_DEN_MIN`）を入れたので，それをコミットしたら差分の再検査が要る（指摘 1）．

経緯：検査の対象は 2947fc8 である．検査の途中で，作業木の `exp/dai23.py` に別のセッションの未コミットの変更（5 行．κ_fixC の分母が 1.0 点未満なら M3_fixC を出さない）が入った．この変更のため，退化の試しの MELD f1 r1 の単位は「未コミットの変更がある」で起動しなかった．そこで 2947fc8 の `exp`・`features`・`measure`・`splits`・`frozen`・`data/metadata`・設計を `git archive` で一時の置き場（scratchpad の `at2947/DAI`．結果の dai1・dai14 と `data/cache` は読むだけのリンク）に取り出し，MELD f1 r1 の単位と集計の段はそこから流した．取り出した `exp/dai23.py` は `git show 2947fc8:DAI/exp/dai23.py` と同じである．ほかの 7 単位は作業木から流し，単位の csv のコミットは 2947fc8，dirty は False である（単位の段の行は未コミットの変更の前後で同じ）．

## 前回の指摘の確かめ

| 前回の指摘（code_v2） | 直した箇所（2947fc8） | 確かめ | 結果 |
|---|---|---|---|
| 1．C′ を固定した版の M3 が R1 の κ を使う | exp/dai23.py:406–418 | 退化の試しの集計と，実験 14 の全 50 単位での照合（`code_run_v3/verify_v3.py`） | 直った．`*_fixC` の行の `kappa` は κ_fixC で，4 組（IEMOCAP/all，MOSEI/all，MELD/all，MELD/nonshared）とも R1 の κ と違う値になった．κ_fixC と同じ作り（`fixc_cell` の A の single_T と single_V の真 − #shufcl の比）を実験 14 の全単位に当てると，実験 14 の predictions.csv の R1fixC の P3 ÷ P3b との差は 0，0，−2.2×10⁻¹⁶，0 で，格子の点（`fixed_c_index`）も一致した |
| 2．別のセッションのコミットで再開と集計が止まる | 設計 8 節「流す場所」 | 設計を読んだ | 設計に「固定のコミットの別の作業木から流す」と書かれた．この検査の間にも同じことが起きた（上の経緯）ので，本番は必ず別の作業木から流す |
| 3．退化の試しの置き場 | 設計 8 節 (1) | 設計を読んだ | `code_run_v2/degenerate/` を名指すように直った．第 4 版の台本の退化の試しは `code_run_v3/degenerate/` にある（指摘 3） |
| 4．特徴の台本が作業木の汚れを記録しない | features/build_audio_lowpass.py:51，111，features/check_lowpass_asr.py:233–235 | 特徴の試し（MELD，64 本）と操作の確かめの試し（`--n 4`） | 直った．特徴の npz に長さ 64 の bool の列 `code_dirty`（False．features はきれい）が入り，`common.load_cache` と `dai23.corpus_audio` で読めた（64 × 1024，NaN 0）．`fc.json` の `code_dirty` は True になった（作業木の `exp/dai23.py` に別のセッションの未コミットの変更があったため．検出は正しく働いた） |
| 5．fixC の行の追加で MOSEI の後ろの行の乱数が変わる | — | 読んだ | 記録だけ（指摘 4 に続きを書く） |

## A．設計と台本の対応表（差分の行だけ）

| 設計の項目（第 4 版） | 台本の箇所（2947fc8） | 合っているか |
|---|---|---|
| 7 節：`*_fixC` の M3 の κ は同じ固定の点で作った A の D の比 κ_fixC で，`kappa` の列に残す | exp/dai23.py:406–418 | 合う |
| 2 節：words3 は人の書き起こしが 3 語以上の発話．M1，M2，M3，M3d，L3，dV を記述として出す．κ はこの部分集合の A の項だけから作る | exp/dai23.py:126–134（`words_of`），356–368，382–386，389，397–400 | 合う．`main.csv` の words3 の行は 1 コーパス 19 行（M1・M2・M3・M3d・dV・L3 の L と B，DAT・DAV・D{L,B}{T,V}，M3_L-B）．`criteria.json` の `side_subset_verdicts.words3` に 3 コーパス × 6 量の判定が空なしで入った．`primary` は偽 |
| 2 節：単語の分け方は参照と出力の両方に，文字化けの直し，引用符の統一，小文字，注記の除去，記号の空白化 | features/check_lowpass_asr.py:58–70 | 合う．7 例の単体の確かめが期待どおり（`\x92`，’，‘ ’，“ ”，ʼ，`\x85`，注記，NaN）．`fc.json` の `norm_rule` も同じ規則を書く |
| 2 節：words3 の単語の分け方は操作の確かめと同じ | exp/dai23.py:129–132（`check_lowpass_asr.norm_words` を import） | 合う．`MIN_WORDS_W3` と `MIN_WORDS` はどちらも 3（同じ定数を 2 か所に書いている．指摘 5） |
| 2 節：`sample_scope.csv` に解析の発話のうち 2 語以下の割合 | features/check_lowpass_asr.py:119–129，221 | 合う．ただし MELD の値は設計の 17.6% でなく 18.3% になった（指摘 2） |
| 6 節：F0 を IEMOCAP で性別ごとにも出す | features/check_lowpass_asr.py:184–192 | 合う．性別は実験 14 の `corpus_meta` と同じ `utterance_speaker` の列 |
| 5 節（12 節の m4）：MELD の顔の判断の再現の増えを共有を除いた版でも見る | exp/dai23.py:309–310 | 合う（all と nonshared のどちらかで dV_X が「下回る」なら `face_increase` に名指す）．5 節の書き方の表の dV の行には nonshared の文が無い（指摘 6） |
| 8 節：特徴の npz と `fc.json` に `code_dirty` | features/build_audio_lowpass.py:51，111，features/check_lowpass_asr.py:233–235 | 合う |

設計の表の組の数：主の 32 行は前回と同じである．`main.csv` は前回の 197 行に words3 の 57 行（19 行 × 3 コーパス）が足されて 254 行になった．`descriptive.csv` 172 行と `assumptions.csv` 72 行は前回と同じである．

## A．小さく流した結果

GPU は検査の始めにほかの処理が約 1.7 GB を使っていた．

| 試し | コマンド | 所要 | 結果 |
|---|---|---|---|
| selftest | `CUDA_VISIBLE_DEVICES="" ./.venv/bin/python -W default exp/dai23.py --stage selftest` | 2 秒 | 「6 項目とも通った」 |
| 退化の試しの単位 | `exp/dai23.py --stage units --out-dir checks/DAI-23/code_run_v3/degenerate --targets <T> ... --l-kind audio_wavlm --b-kind audio_wavlm --verify`（IEMOCAP は f0・f1 × r0・r1，MOSEI と MELD は f0 r0 と f1 r1．3 コーパスを並べて流した） | 並べたので 1 単位 42〜372 秒．MELD f1 r1 は取り出した 2947fc8 から 69 秒 | 8 単位．行は 63，27，63，27，99，27，99，27．照合の一致は 8 単位とも J_T・J_V で 1.0．勾配の最大 1.0×10⁻⁶．ログに例外と収束の警告なし（ResourceWarning だけ） |
| 退化の試しの集計 | 取り出した 2947fc8 から `exp/dai23.py --stage summary --out-dir .../code_run_v3/degenerate --allow-partial --l-kind audio_wavlm --b-kind audio_wavlm` | 61 秒 | `main.csv` 254 行 29 列．effect・lo・hi の NaN 0，Inf 0 |
| 特徴の試し | `features/build_audio_lowpass.py --op lp --fc 500 --corpus MELD --limit 64 --chunk 32`，続けて `--merge` | 7 秒と数秒 | `code_dirty` の列（上の表の 4）．試しの 3 ファイル（`data/cache/audio_wavlm-lp500_trial_MELD.npz` と途中保存 2 つ）は確かめた後に消した |
| 操作の確かめ | `features/check_lowpass_asr.py --n 4 --out-dir checks/DAI-23/code_run_v3/asr` | 3 分 0 秒 | 5 ファイル（asr_check.csv 60 行，asr_check_summary.csv，f0_check.csv，fc.json，sample_scope.csv） |
| 単語の確かめ | `checks/DAI-23/code_run_v3/check_words.py`（CPU） | 1 分以内 | 下の「norm_words と words3」 |

**退化した場合の検算**：L と B に元の声の特徴を入れた 8 単位で，words3 の M1・M2・M3・M3d・dV（L と B）と M3_L-B の 33 行，`*_fixC` の 24 行の効果・区間の両端・標準誤差の絶対値の最大はすべて 0.0 である．words3 の `kappa` の列は各コーパスで 1 種類で，同じ部分集合の DAT ÷ DAV との差は 0.0 である．words3 の DAT は all・nomarker の DAT と違う値になり，words3 が別の発話の集まりで計算されていることを確かめた．

**norm_words と words3**（`check_words.py`）：
- 文字化けを直した後の参照に C1 制御域の文字が残る発話は 3 コーパスとも 0．残る非 ASCII の文字は，’（QUOTES が ' に直す），“ ”，–，—，…，é など．MOSEI に `U+FFFD`（壊れた文字）が 40 字ある（指摘 7）．
- 新しい規則で語数が変わった発話の割合は IEMOCAP 0，MOSEI 4.6%，MELD 27.0%．3 語未満の割合は IEMOCAP 15.97%（旧い規則と同じ），MOSEI 0.92%（同じ），MELD 18.32%（旧い規則 17.58%）．
- `words_of` の index の重複は 0．実験 14 の f0 r0 の候補（IEMOCAP 2,212 本，MOSEI 14,907 本，MELD 5,995 本）のうち発話一覧に無いものは 0（`reindex` で NaN になって words3 に黙って入る発話は無い）．words3 に入る割合は 0.859，0.991，0.813．`mask_of(U_w3, "nomarker")` の本数は words3 の本数と一致した．

**操作の確かめの出力**：
- `sample_scope.csv`：解析の発話 IEMOCAP 2,943，MOSEI 18,634，MELD 11,267．3 語未満の割合 0.1597，0.0092，0.1832．0 語の割合 0.0068，0，0.0001．
- `f0_check.csv`：IEMOCAP に sex = F・M・all の 3 行，MOSEI と MELD に all の 1 行．IEMOCAP の F と M の n_utt の和（2 + 2）と n_voiced の和（287 + 73）は all の行（4，360）と一致した．性別の行は `utt_share_any_above_0.8x*` の 3 列が空である（指摘 8）．
- `asr_check.csv` の n_ref は 3 コーパスとも新しい規則で数え直した参照の語数と一致し，最小は 3．選ばれた発話の先頭 2 本は前回の試し（`code_run_v2/asr`）と同じだった．
- `fc.json` に `code_commit`（2947fc8），`code_dirty`（True．経緯を参照），新しい `norm_rule` が入った．

**見た値の申告**：画面に出したのは，行の数，NaN の数，退化の量が 0 であること，κ と照合の差，C′ の固定の点（試しの 8 単位で IEMOCAP 0.001/0.01，MOSEI 0.001/0.001，MELD 0.0001/0.001），sample_scope.csv の割合，f0_check.csv の n_utt と n_voiced だけである．操作の確かめの要約の表（一致率と超過）と F0 の割合は画面に出していない（`asr.log` に書かせて開いていない）．選ばれた 4 本は sha256 の順の先頭 4 本で，前回までの試しと同じ範囲である．誤りとして，実験 14 の predictions.csv の構造を見るために先頭 600 バイトを出したとき，AFEW の R1・#shufcl・all の P1 の 1 行（effect 4.93，区間 [−2.66, 12.53]）が画面に出た．AFEW はこの実験の対象外で，実験 14 の結果の文に載っている値である．

## A．本番の規模の見積もり

単位の段の台本は 1fb89ff から変わっていないので，前回の見積もり（150 単位で約 2.8 時間，GPU は全体で 4 GB 未満，保存は約 0.52 GB）のままである．今回は 3 コーパスを並べて流したので 1 単位の所要が 1.5〜4 倍になった．本番は 1 本ずつ流す．集計の段は words3 と κ_fixC の分で，8 単位で 50 秒から 61 秒になった．全単位で数分である．操作の確かめは 4 本 × 3 コーパスで 3 分（モデルの読み込みを含む）で，前回の見積もり（約 1 時間）を変えない．

## B．差分で変わった予想される指摘と，出力から答えられるか

前回の表のうち，変わった行と新しい行だけを書く．ほかの行は前回の判定のまま．

| 表・図 | 予想される指摘 | もとにした項目か例 | (a)／(b)／(c) | (b) なら足すもの | 足す費用 |
|---|---|---|---|---|---|
| 主の表 | C′ を固定したら M3 の結論は変わるか | R4 | (a)：`main.csv` の `M3_X_fixC`（κ は `kappa` の列の κ_fixC）．R1 の κ を当てた版が要るなら，単位の npz の格子の予測と `quant(X, κ_R1)` で CPU で作れる | — | — |
| 主の表 | 2 語以下の発話では言葉が消えたか，3 語以上だけで結論は変わるか | 感情認識の査読の A2，L1 | (a)：words3 の行（`main.csv` の subset = words3，`criteria.json` の `side_subset_verdicts.words3`）．2 語以下だけの版も，単位の npz の `sid` と `words_of` で `marker` を「3 語以上」に置き換えて同じ集計を流せば CPU で出せる．2 語以下の発話で Whisper の一致率が下がったかは測っていない（操作の確かめの標本の外．設計 9 節の限界に書かれている） | — | — |
| 操作の確かめ | MELD の文字化けを直したら fc の選び方は変わるか | A1 | (a)：本番はまだ流していないので，直した規則だけで fc が決まる．旧い規則の値が要るなら `asr_check.csv` の `hyp` と発話一覧の参照から CPU で作り直せる | — | — |
| 仮定の表 | 女性の話者で F0 が 0.8·fc を超える割合は | a4 | (a)：`f0_check.csv` の IEMOCAP の sex = F の行の `share_above_0.8x<fc>`．発話の割合（`utt_share_any`）は性別の行に無い（指摘 8）．MELD・MOSEI は性別の注釈が無いので (c) | — | — |
| 主の表 | MELD で共有を除いた版の顔の判断の再現が増えたか | m4 | (a)：`criteria.json` の `face_increase` と `main.csv` の nonshared の dV_X | — | — |

止めるべき (b) は無い．

## 指摘

### [I9・I10] 1．検査の間に別のセッションが `exp/dai23.py` を未コミットで変えた
- 重さ：直してから進む（この変更をコミットしたら台本の再検査を受ける）
- 該当箇所：作業木の exp/dai23.py:127（`KFIX_DEN_MIN = 1.0`），411–415（`kfix = ... if abs(fb["effect"]) >= KFIX_DEN_MIN else np.nan`，`if q == "M3" and not np.isfinite(kfix): continue`）
- 問題：読んだだけ．この変更は 2947fc8 に無く，この検査の対象の外である．変更が入ると，κ_fixC の分母が 1.0 点未満の組で `M3_X_fixC` の行が出なくなるので，`main.csv` の行の数（254）と `*_fixC` の行（24）が変わる．`kappa` の列に NaN が入る行ができうる．退化の試しの集計は 2947fc8 の写しで流したので，この変更を通していない．あわせて，同じ作業木で台本が書き換わると，流している途中の単位の段が「未コミットの変更がある」で止まる（この検査の MELD f1 r1 がそうなった）．
- 計算の後で直せるか：直せる（集計の段だけ）．
- 求める対応：変更をコミットしたら，差分を台本の検査に回す．そのとき，分母の基準 1.0 点が設計 7 節に書かれているか，M3_fixC を出さない組を結果の文でどう書くかを確かめる．退化の試しの集計を新しいコミットで流し直す（単位は `code_run_v3/degenerate/units/` をそのまま使える）．

### [D・記録] 2．MELD の 2 語以下の割合は，直した規則で 18.3% になり，設計の 17.6% と違う
- 重さ：記録する
- 該当箇所：docs/設計/実験23_設計.md 33 行と 131 行（「MELD 17.6%」），features/check_lowpass_asr.py:126–127
- 問題：流して確かめた．`sample_scope.csv` の MELD の `share_lt_min_words` は 0.1832 である．同じ発話を旧い規則（文字化けを直さない）で数えると 0.1758 になる．`don\x92t` が 2 語から 1 語になるので，直した規則のほうが短い発話が増える．IEMOCAP 15.97% と MOSEI 0.92% は設計の値と合う．
- 計算の後で直せるか：直せる．
- 求める対応：結果の文と限界の割合は `sample_scope.csv` の値を使い，設計の 17.6% を引かない．

### [I9] 3．第 4 版の台本の退化の試しは `code_run_v3/degenerate/` にある
- 重さ：記録する
- 該当箇所：docs/設計/実験23_設計.md 8 節の順序 (1)
- 問題：設計は `code_run_v2/degenerate/`（台本 1fb89ff）を名指す．words3 と κ_fixC の 0.0 の検算は `code_run_v3/degenerate/`（8 単位．7 単位はコミット 2947fc8，MELD f1 r1 は 2947fc8 を取り出した写しから流したので `commit` の列が空）にある．
- 求める対応：結果の文で words3 と fixC の退化の検算を引くときは `code_run_v3/degenerate/` を名指し，MELD f1 r1 のコミットが空である理由を書く．

### [記録] 4．words3 の κ と κ_fixC の計算で，MOSEI の引き直しの乱数の順が変わる
- 重さ：記録する
- 該当箇所：exp/dai23.py:365–366，408–409
- 問題：読んだだけ．MOSEI の区間は 1 つの `rng` から順に引く．words3 の κ（2 回）と κ_fixC（2 回）の `combo_stats` も `rng` を使うので，その後ろの行（fixC の行，記述の行）の区間の乱数は前の版と違う．点推定は乱数に依らない．主の 32 行は all の主の量の後ろにある κ_fixC より前に引くので変わらない．本番の値は無いので結果には影響しない．
- 求める対応：無し．

### [記録] 5．3 語の基準を 2 か所に書いている
- 重さ：記録する
- 該当箇所：exp/dai23.py:126（`MIN_WORDS_W3 = 3`），features/check_lowpass_asr.py:51（`MIN_WORDS = 3`）
- 問題：読んだだけ．今は同じ値である．片方だけを直すと，words3 と操作の確かめの標本の範囲がずれる．
- 求める対応：`words_of` と同じく `check_lowpass_asr.MIN_WORDS` を import する．直さないなら記録だけでよい．

### [記録] 6．設計 5 節の書き方の表の dV の行に，MELD の共有を除いた版を見ることが書かれていない
- 重さ：記録する
- 該当箇所：docs/設計/実験23_設計.md 83 行（dV_X の行），12 節の m4 の行，exp/dai23.py:309–310
- 問題：台本は all と nonshared のどちらかで dV_X が「下回る」なら `face_increase` に名指し，`lexical_both_ops_ok` から外す．12 節の変更の一覧にはこのことがあるが，5 節の表は「dV_X の区間が 0 を下回る」だけである．
- 求める対応：5 節の dV の行に「MELD は共有を除いた版も見る（どちらかで下回れば）」と書く．

### [記録] 7．MOSEI の書き起こしに壊れた文字 U+FFFD が 40 字ある
- 重さ：記録する
- 該当箇所：features/check_lowpass_asr.py:62–70
- 問題：流して確かめた．`fix_mojibake` の後も MOSEI の解析の発話の書き起こしに `U+FFFD` が 40 字残る．元がアポストロフィなら `don�t` が 2 語に割れ，一致率の参照の単語が 1 つ増える．18,634 発話のうちの 40 字なので，100 本の標本に入っても影響は小さい．
- 求める対応：無し．結果の文に書く必要も無い．

### [記録] 8．F0 の性別の行に発話の割合の列が無い
- 重さ：記録する
- 該当箇所：features/check_lowpass_asr.py:190–192
- 問題：流して確かめた．性別の行は `share_above_0.8x<fc>`（有声の区間の割合）だけを持ち，`utt_share_any_above_0.8x<fc>` の 3 列は空になる．設計 6 節が求めるのは区間の割合なので設計とは合う．発話ごとの F0 は保存しないので，性別ごとの発話の割合を後から求められたら波形から F0 を推定し直す（学び直しは要らず，100 本の YIN で数分）．
- 求める対応：要るなら性別の行にも同じ 3 列を足す．

## 目録に無い種類の不備
- 共有の作業木で別のセッションが検査中の台本を未コミットで書き換えると，検査役の試しが「未コミットの変更がある」で止まるか，検査していない版で流れる．検査役は，検査の始めと試しの直前に `git status --porcelain exp features measure` を確かめ，変わっていたら検査の対象のコミットを `git archive` で取り出して流す（I9 か I10 への追記の案）．

## 確かめた項目
I1，I2，I3，I4，I6，I7，I9，I10，I11，I12，I14，I18，R1，R4，R8，R11．

試しの出力：`checks/DAI-23/code_run_v3/`（degenerate/，asr/，units_*.log，summary.log，asr.log，feat_trial.log，検算の台本 check_words.py と verify_v3.py）．`results/` と `data/cache/` の本番の名前には書いていない．特徴の試しの 3 ファイルは消した．
