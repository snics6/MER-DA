# 関門 1：実装と出力（DAI-23，2026-11-15，コミット 1376f64）

## 判定
判定：直してから通す．第 3 版の台本は，本番の段の初め（provenance を JSON に書く行）と，特徴の途中保存をまとめる段・読む段の 2 か所で必ず落ちる．どちらも誤った数値は出さないが，直すまで本番を流せない．退化の試しと照合は，この 2 か所を試しの包みで避けて流すと，すべて期待どおりだった．

経緯：91af3db（第 2 版）を静的に検査している途中で，本体から第 3 版（c8d7182）とその直し（1376f64）のコミットの知らせを受けた．GPU の試しは 1376f64 で流した（実験 22 の `units.done` ができた 23:07 の後）．91af3db と c8d7182 で見つけたことのうち，1376f64 で直ったものは末尾の「前の版で見つけて直ったこと」に書いた．

## A．設計と台本の対応表

設計は `docs/設計/実験23_設計.md`（第 3 版．1376f64．検査の時点で作業木では未コミットの変更がある．本番の段は `committed_clean(DESIGN)` で止まるので，流す前にコミットが要る）．

| 設計の項目 | 台本の箇所（1376f64） | 合っているか |
|---|---|---|
| 2 節 波形：`build_audio.decode` → 操作 → `normalize` → WavLM-large の最後の層の時間平均 | build_audio_lowpass.py:39，84–95 | 合う．`build_audio.process_chunk` と操作の 1 段だけが違う |
| 2 節 L：4 次の Butterworth の前後 2 回，fc で −6 dB，約 48 dB/oct | build_audio_lowpass.py:42，51–59 | 合う．合成の正弦波で 0.5·fc −0.03 dB，0.8·fc −1.34 dB，fc −6.02 dB，1.25·fc −16.96 dB，2·fc −48.87 dB |
| 2 節 B：`w[::-1]` | build_audio_lowpass.py:66–67 | 合う |
| 2 節 出力の名前と，既存を書き換えないこと | build_audio_lowpass.py:166–183 | 名前は合う．**まとめる段と読む段が落ちる**（指摘 1） |
| 2 節 L は fc.json の fc だけ本番の名前で作る．試しは `_trial`，`--limit` は 256 以下 | build_audio_lowpass.py:47，168–183 | 合う |
| 2 節 流れの照合：sha256 の順の 64 発話，差の最大 2×10⁻³ 以下 | build_audio_lowpass.py:109–127 | 合う．3 コーパスとも 64/64 件で ok が一致し，差の最大は IEMOCAP 4.19×10⁻⁵，MOSEI 3.28×10⁻⁵，MELD 6.64×10⁻⁵ |
| 2 節 標本：3 モダリティがそろう発話（IEMOCAP は即興，MELD は複数の話者を除く），3 語以上，30 秒以下，sha256 の順に 100 本 | check_lowpass_asr.py:112–129 | 合う．4 本の試しで IEMOCAP は impro だけ，3 コーパスとも `load_corpus` の集合に入る |
| 2 節 本番の置き場は `--n 100` だけ | check_lowpass_asr.py:152–153 | 合う |
| 2 節 Whisper の設定 | check_lowpass_asr.py:132–142，159 | `build_transcript_afew.py:83–85，141` と同じ引数 |
| 2 節 一致率 H/N，偶然の水準（別の発話の参照との全組の平均），超過 = H/N − 偶然，規則は超過 0.10 以下の最も高い fc，逆再生は rev_ok | check_lowpass_asr.py:58–83，182–214 | 合う．試しの `asr_check_summary.csv` に hit_mean，hit_chance，excess，n_hyp_mean の列がある．偶然の水準の計算は 100 本でも 1 条件 3 秒未満（合成の語の列で計った） |
| 6 節 F0：YIN，元の音声，F0 > 0.8·fc の割合 | check_lowpass_asr.py:86–109，167–171 | 合う．正弦波と調波 7 本の波で 80〜400 Hz の推定の誤差 0.5 Hz 以下，450 Hz は 444.4 Hz（τ の整数の刻み），無音と白色雑音は有声 0 区間 |
| 3 節 生徒 L・B を足し，A・V の J_m の予測は実験 14 の保存 | dai23.py:143–144，244–260 | 合う |
| 3 節 `load_group` を通して読む．発話の集合，候補の並び，ラベルが実験 14 と同じか確かめる | dai23.py:115–123，151–161 | 合う |
| 3 節 候補の統計で標準化，`aligned_features` で 512 次元 | dai23.py:126–129，156，161 | 合う（dai1.py:176 と同じ式） |
| 3 節 入れ替えた目標を作り直して照らす（全単位） | dai23.py:165–174 | 合う．試しの 6 単位で止まらなかった |
| 3 節 目標，K = 5，R1 だけ，A は y だけ | dai23.py:54，162–164，197–206 | 合う |
| 3 節 r = 0 の別の分け方 | dai23.py:217–230 | 合う |
| 3 節 M1・M2・M3・L3 と記述の量 | dai23.py:70–90 | 合う．係数の符号を 1 項ずつ確かめた |
| 3 節 部分集合ごとの κ を P3 ÷ P3b で作り直し小数 3 桁で照らす | dai23.py:61–63，263–276 | 合う．7 組（IEMOCAP all・nomarker，MOSEI all・nomarker，MELD all・nonshared・nomarker）とも設計の値と一致 |
| 4 節 M3d：κ̂ の揺れを 1 次で入れた区間 | dai23.py:356–366 | 合う．係数 (D_XV/D_AV, −1, −κ̂·D_XV/D_AV, κ̂) は M3 = D_AT − D_XT − (D_AT/D_AV)(D_AV − D_XV) の偏微分と一致し，点推定は M3 と同じになる |
| 4 節 summary が作り直した D(A←T)/D(A←V) と固定の κ の一致を確かめる | dai23.py:354–355 | 合う．実験 14 の全 50 単位で計算すると差は 7 組とも 4.4×10⁻¹⁶ 以下で，本番で誤って止まらない |
| 記述 M3_L−B | dai23.py:368–369 | 合う（−D_LT + D_BT + κ(D_LV − D_BV)） |
| 5 節 判定と数え（E1，E2 に M3d を含める，lexical_both_ops_ok，face_increase） | dai23.py:287–316 | 合う．selftest の (5)(6) が通る |
| 6 節 C′ の端の割合（実験 14 の 8 節と同じ数え） | dai23.py:398–413 | **数え方が実験 14 と違う**（指摘 3） |
| 6 節 `--verify`：r = 0 の単位は本番でも | dai23.py:556 | 合う |
| 8 節 置き場と記録 | dai23.py:540–548 | **provenance を JSON に書く行で落ちる**（指摘 2） |
| 8 節 退化の試しは `--l-kind` と `--b-kind` の両方に audio_wavlm | dai23.py:530–533 | 合う |

設計の表に載る組：主の表は 3 コーパス × 2 操作 × 量（M1・M2・M3・M3d・L3）× 部分集合（MELD は all と nonshared）．台本は，部分集合ごとに主の量と記述の量の 29 行（nomarker は 19 行）を出す．試しの集計で `main.csv` は IEMOCAP all 29，nomarker 19，MOSEI all 29，nomarker 19，MELD all 29，nonshared 29，nomarker 19 の 173 行，primary の行は 32 行（4 組 × 8 量）で，設計の主の表の組をすべて含む．計算しない組（AFEW，K = 25，R0，R1fixC，#shuffine，#shufage，ほかの fc，ほかのエンコーダ，人の聞き取り，非線形の識別器）は設計 7 節に書かれている．

## A．小さく流した結果

1376f64，作業木の exp・measure・features はきれい（dirty = False）．

| 試し | コマンド | 所要 | 結果 |
|---|---|---|---|
| selftest | `CUDA_VISIBLE_DEVICES="" ./.venv/bin/python -W default exp/dai23.py --stage selftest` | 約 2 秒 | 「6 項目とも通った」 |
| 流れの照合 | `cd features && ../.venv/bin/python -W default build_audio_lowpass.py --op lp --fc 500 --limit 64 --check-identity` | 29 秒 | 上の表の数値．GPU の使用は全体で最大 14.2 GB（ほかの処理の約 1.7 GB を含む） |
| 特徴の試しとまとめ | `--op lp --fc 500 --corpus MELD --limit 64 --chunk 32`，`--op rev` 同じ，続けて `--merge` | 各 10 秒 | 途中保存は 2 つずつでき，NaN 0．操作の後の特徴と元の特徴の cos の中央値は lp500 で 0.60，rev で 0.72（操作が特徴に効いていることの確かめ）．**`--merge` と `common.load_cache` は ValueError で落ちた**（指摘 1）．試しの途中保存 4 つは消した |
| 操作の確かめ | `cd features && ../.venv/bin/python -W default check_lowpass_asr.py --n 4 --out-dir ../checks/DAI-23/code_run/asr` | 3 分 9 秒 | `asr_check.csv` 60 行（3 コーパス × 5 条件 × 4 本），summary 15 行，f0 3 行，fc.json．GPU は全体で最大 13.9 GB．警告：Whisper の「attention mask is not set」（AFEW の書き起こしと同じ呼び方．入力は 30 秒に詰めるので影響は無い），fc.json の開いたままのファイルの ResourceWarning |
| 退化の試しの単位 | `checks/DAI-23/code_run/run_dai23_provfix.py --stage units --out-dir checks/DAI-23/code_run/degenerate --l-kind audio_wavlm --b-kind audio_wavlm --verify`（`--folds 0 --seeds 0` と `--folds 1 --seeds 1`） | f0 r0：IEMOCAP 51 秒，MOSEI 119 秒，MELD 120 秒．f1 r1：30 秒，63 秒，62 秒 | 行は f0 r0 で 63，99，99 行（L・B 各 13 行，A 1 行，n_alt × 2 × 9 行．n_alt は IEMOCAP 2，MOSEI・MELD 4），f1 r1 で各 27 行．`--verify` の一致の割合は 6 単位とも J_T・J_V で 1.0．勾配の最大 1.0×10⁻⁶，stalled 0．GPU は全体で最大 4.0 GB（ほかの処理の約 1.7 GB を含む） |
| 退化の試しの集計 | 同じ包みで `--stage summary --allow-partial` | 47 秒 | `main.csv` 173 行 27 列，`descriptive.csv` 172 行，`assumptions.csv` 78 行，criteria・convergence・provenance．NaN 0，Inf 0 |

包みの台本 `checks/DAI-23/code_run/run_dai23_provfix.py` は，dai23 の中の `json.dumps` と `json.dump` を，鍵のタプルを文字列にしてから書く版に差し替えるだけで，計算には触れない（指摘 2 を避けるため）．

**退化した場合の検算**：L と B に元の声の特徴を入れると，f0 r0 の L の予測は実験 14 の A の予測と，元の分け方と別の分け方の J_T・J_V の真・#shuf・#shufcl のすべての組（IEMOCAP 18 組，MOSEI・MELD 30 組）で一致の割合 1.0，L と B の予測も一致の割合 1.0 になった．確率は L と A で差の最大 4.9×10⁻⁵，確率の和のずれは 1.2×10⁻⁷ 以下．集計では M1・M2・M3・M3d（L と B），dV，M3_L−B，M1R・M2R・M1e の全行で効果と区間の両端が 0.0，標準誤差の最大も 0.0 になった．R(X←y) は A・L・B で同じ値になった（値は報告に書かない）．

**保存済みの値の再現**：上の一致の割合 1.0 と，κ の比が実験 14 の全単位で固定の κ と 4.4×10⁻¹⁶ 以内で一致することが，再現の確かめにあたる．

**見た値の申告**：試しの集計を確かめるときに，元の声の特徴での R(A←y)（6 単位の部分）の値と，退化の試しでの L3 の判定（実験 14 の P4a と同じ量）を画面に出した．操作の確かめの 4 本の試しの一致率も見た（見てよいと指示された範囲）．言葉を消した特徴から作る量は見ていない（特徴が無い）．

## A．本番の規模の見積もり

- 操作の確かめ（100 本 × 3 コーパス × 5 条件 = 1,500 回の書き起こし）：4 本の試しの 60 回が約 2.5 分（読み込みを除く）なので約 1 時間．設計 8 節の 20〜40 分より長い．逆再生や低域通過で Whisper が長い文を作ると延びる（試しの MOSEI の逆再生で出力の語数の平均 109.5）．GPU は全体で 14 GB 前後で，6 割の上限（14.7 GB）の内側．
- 特徴（発話の数は IEMOCAP 5,531，MOSEI 18,919，MELD 11,352 の計 35,802）：試しの 0.043〜0.094 秒/件から 1 操作 26〜56 分，2 操作で約 1〜1.9 時間．GPU は 12〜13 GB．
- 単位（150 単位）：r = 0（照合あり）は IEMOCAP 51 秒，MOSEI 119 秒，MELD 120 秒，r ≠ 0 は照合ありで 30 秒，63 秒，62 秒（本番の r ≠ 0 は照合が無いのでこれより少し短い）．合計は多くて約 2.3 時間．GPU は約 2.3 GB．保存は 1 単位 1〜6.4 MB で，合計約 0.5 GB．
- 集計：全単位で数分（試しの 6 単位で 47 秒）．

## B．予想される指摘と，出力から答えられるか

| 表・図 | 予想される指摘 | もとにした項目か例 | (a)／(b)／(c) | (b) なら足すもの | 足す費用 |
|---|---|---|---|---|---|
| 主の表 | 区間を話者を単位に作れ | R1，ML Major-1 | (a)：`combo_stats` が塊 `cl`（IEMOCAP は話者）で jackknife か重みの引き直しをする | — | — |
| 主の表 | IEMOCAP を性別ごとに出せ | R1，感 Minor-3 | (a)：実験 14 の npz の `sex` と dai23 の npz の予測と `tgt__K5__*` から，`D14.counts` に性別の覆いを掛ければ数が作れる．試しの IEMOCAP_f0_r0 で，F 1,117 件・M 1,095 件，各 4 塊の (4, 4) の数が作れた | — | — |
| 主の表 | 感情ごとの内訳 | R1 | (a)：`y`，予測，目標が候補の行ごとにある | — | — |
| 主の表 | L は A と別の C′ が選ばれ，罰則の違いが減りに混ざらないか．C′ を固定したら結論は変わるか | R4，I18 追記（3），実験 14 の R1fixC | **(b)**：dai23.py:202 は `grid_keys` を渡さず，L・B の格子の全点の予測を保存しない．A・V の格子の予測は実験 14 にある（`K5__grid<i>__R1__A__single_T__true` など）．選ばれた C′ は csv の C0・C1 に残るので，選ばれた C′ の分布の違いは (a) で出せる | `D14.fit_r1(..., grid_keys=[((5, nm), v) for nm in JUDG for v in ("true", "shufcl")])` を L・B で渡し，`K5__grid<i>__R1__<X>__<nm>__<v>` を保存する | 外側の問が 1 単位・1 生徒で 26 → 122 本（内側の 160 本と合わせて約 1.5 倍）．単位の合計は約 2.3 → 約 3 時間．保存は int8 で 96 列 × 候補の数（MOSEI でも 1 単位 2 MB 未満） |
| 主の表 | fc を変えると減りは単調か | R4，R7 | (c)：設計 7 節に「ほかの fc での主の量」とあるが，理由の文が無い | 理由を 1 文書く | — |
| 主の表 | κ の揺れを区間に入れよ | 統計 | (a)：第 3 版で M3d として出る | — | — |
| 主の表 | 部分集合では κ を部分集合で作れ | D | (a)：第 3 版で部分集合ごとの κ を使う | — | — |
| 主の表 | 2 つの操作の差（L と B のどちらがより減らすか） | C | (a)：第 3 版で M3_L−B を記述として出す．M1・M2 の L−B も同じ保存から `combo_stats` の係数を変えれば出せる | — | — |
| 主の表 | 減りは発話の長さで説明できるか | 9 節の限界 | (a)：特徴の npz の `duration_sec` を `sample_id` で貼り，長さの区分ごとに数を作れる．長さだけの識別器を保存の目標と 2 組の分け方で学ぶのは CPU で数分（単位の学び直しは要らない） | — | — |
| 主の表 | 逆再生の WavLM の特徴に言葉の手がかりが残っていないか | 9 節の限界 | (a)：特徴の npz と言葉の特徴が残るので，単位を学び直さずに別に測れる（設計では「測らない」として限界に書く） | — | — |
| 主の表 | 全般の劣化と同じ程度に壊す別の操作（雑音など）と比べよ | R6 | (c)：M2・M3 の帰無と R(X←y) の比で扱う設計．設計 7 節にこの操作の名は無い | 7 節に 1 行足す | — |
| 主の表 | 第 2 のエンコーダ，人の聞き取り，非線形の識別器，K = 25，R0 | R7 | (c)：設計 7 節に理由つきで書かれている | — | — |
| 主の表 | 判断の一致でなく確率で測ったら | R1 | (a)：L・B・A（y）の確率は元の分け方で保存（`prob__K5__main__R1__L__label__true` などがあることを確かめた），A・V の J_m の確率は実験 14 の `prob__`，目標の確信度は実験 14 の `maxp__K5__*` | — | — |
| 主の表 | 同じ反復の集合で比べているか | R8 | (a)：A・V は実験 14 の同じ単位，L・B は同じ単位・同じ 2 組・同じ別の分け方（`load_merged` が `sid`・`y`・`cl`・`half` を照らす） | — | — |
| 操作の確かめ | WER の規則や別の閾値なら fc は変わるか | R4 | (a)：`asr_check.csv` に発話ごとの hit，wer，出力の文がある．偶然の水準も出力の文から作り直せる | — | — |
| 操作の確かめ | 低域通過の後の音声で F0 が残るか | 6 節 | (b) に近い：F0 は元の音声でだけ測る．同じ標本を CPU で読み直せば数分で出せ，学び直しは要らない | 必要なら `f0_check.csv` に濾波後の行を足す | CPU 数分 |
| 仮定の表 | 実験 14 の A と同じ条件で L を学んだか | C，I14 | (a)：r = 0 の単位の `verify_agree_*` の列（本番でも入る）と，この関門の退化の試しの出力 `checks/DAI-23/code_run/degenerate/` | — | — |

止めるべき (b) は無い．C′ を固定した版は設計 7 節に範囲の外と書かれているので「止める」にしないが，費用が小さく I18 追記（3）にも当たるので，足すことを勧める（指摘 4）．

## 指摘

### [I10・I12] 1．特徴の途中保存の 0 次元の配列のため，まとめる段と読む段が必ず落ちる
- 重さ：直してから進む（誤った数値は出さないが，本番の手順の (3) で必ず止まり，単位の段も特徴を読めない）
- 該当箇所：features/build_audio_lowpass.py:104–105，features/common.py:179（`merge_npz`），163（`load_cache`）
- 問題：`process_chunk` の戻り値に `"op": np.array(op or "")`，`"lowpass_fc"`，`"lowpass_order"`，第 3 版で足した `"filter"`，`"model_revision"` の 0 次元の配列が入る．`merge_npz` は `out = {k: np.concatenate([z[k] for z in zs]) for k in zs[0].keys()}`，`load_cache` も同じ結合をする．試しで MELD の 64 件を 32 件ずつの途中保存 2 つにして `--merge` を流すと `ValueError: zero-dimensional arrays cannot be concatenated` で落ち，`common.load_cache('audio_wavlm-rev_trial', 'MELD')` も同じく落ちた．途中保存が 1 つでも 0 次元の配列の結合は落ちる．dai23 は `load_group` → `load_cache` で読むので，units の段も落ちる．
- 計算の後で直せるか：直せる（特徴を作り直さずに途中保存を読み替えればまとめられる）が，本番の流れが止まる．
- 求める対応：5 つの値を発話の数 n の長さの配列にする（`np.full(n, ...)`）か，途中保存に入れず別の json に書く．直した後に `--limit 64 --chunk 32` の試しで `--merge` と `dai_data.load_corpus(T, {"A": "<kind>_trial", ...})` が通ることを確かめる（試しの名前は `FEATURE_KEYS` に無いので，dai23 の `corpus_audio` と同じく鍵を足して読む）．

### [I9・I12] 2．本番の段の初めで provenance を JSON に書けず落ちる
- 重さ：直してから進む（units の段も summary の段も，何も計算せずに落ちる）
- 該当箇所：exp/dai23.py:543（`kappa_design=KAPPA_DESIGN`），548（`json.dumps(prov, ...)`），summary の後の `json.dump(prov, ...)`
- 問題：第 3 版で `KAPPA_DESIGN` の鍵を `("IEMOCAP", "all")` のようなタプルにしたので，`json.dumps` が `TypeError: keys must be str, int, float, bool or None, not tuple` で落ちる．1376f64 で `--stage units --out-dir checks/DAI-23/code_run/degenerate --folds 0 --seeds 0 --l-kind audio_wavlm --b-kind audio_wavlm --verify` を流して確かめた（2.5 秒で落ちた）．selftest はこの行を通らないので見つけられない．
- 計算の後で直せるか：直せる（落ちるだけ）．
- 求める対応：`kappa_design={f"{k[0]}/{k[1]}": v for k, v in KAPPA_DESIGN.items()}` にする．直した後に退化の試しの 1 単位を包みなしで流し，`provenance_log.jsonl` が書けることを確かめる．

### [I1] 3．C′ の端の割合の数え方が実験 14 の 8 節と違う
- 重さ：直してから進む（仮定の表で名指す組が変わる）
- 該当箇所：exp/dai23.py:398–413
- 問題：dai23 は `hi = float(((g.C0 == D14.GRID_C[-1]) | (g.C1 == D14.GRID_C[-1])).mean())` と，単位ごとに 2 組のどちらかが端なら 1 と数え，`flag=bool(hi > D14.EDGE_SHARE)` とする．下端の平らでない分（`C_low_nonplateau`）は目標をまとめて 1 行にする．実験 14（dai14.py:878–884）は単位 × 組ごとに選んだ点を数え，目標ごとに `flag = max(hi, lo_np) > 0.25` とする．dai23 の上端の割合は組ごとの割合の最大 2 倍になり，下端の印は目標ごとに出ない．設計 6 節は「実験 14 の 8 節と同じ数え」と書く．
- 計算の後で直せるか：直せる（単位の npz の `inner__main__<X>__K5__<nm>` と csv の C0・C1 から数え直せる）．
- 求める対応：dai14.py:878–884 と同じ数え（単位 × 組ごと，生徒と目標ごと，`flag = max(hi, lo_np) > 0.25`）にする．

### [R4] 4．L・B の C′ を固定した感度を出す保存が無い
- 重さ：記録する（設計 7 節に範囲の外と理由が書かれている）．ただし足すことを勧める
- 該当箇所：exp/dai23.py:202，227
- 問題：M1 は A（実験 14 で C′ を選んだ）と L（dai23 で C′ を選んだ）の差である．L の特徴で別の C′ が選ばれると，罰則の違いが D の減りに混ざる．実験 14 は R1fixC でこれを確かめたが，dai23 は L・B の格子の予測を保存しない．設計 7 節の理由「格子の全点の予測を保存する必要がある」は費用の理由で，費用は B の表のとおり単位の合計で約 0.7 時間の増え．
- 計算の後で直せるか：直せない（学び直しが要る）．
- 求める対応：`grid_keys` を渡して L・B の J_A・J_V・J_T の真と #shufcl の格子の予測を保存する．足さないなら，結果の文で「C′ の選び方の感度は測っていない」と限界に書く．

### [I9] 5．記録の不足
- 重さ：記録する
- 該当箇所：exp/dai23.py:540–548，summary の最後，features/check_lowpass_asr.py:196–214
- 問題：(1) summary に単位ごとの行の数の照合が無い（実験 14 は dai14.py:705–713 で照合する．1 単位の行は L・B で各 13 行，A で 1 行，r = 0 で n_alt × 2 × 9 行．試しの 6 単位はこの数と合った）．(2) provenance に A・V・T の特徴の sha256 と WavLM のモデル名が無い（L・B の特徴の sha256 はある）．(3) `fc.json` にコミット番号と `asr_check.csv` の sha256 が無い．(4) 特徴の npz にコミット番号が無い．(5) 設計 8 節は退化の試しの置き場を `results/dai23/degenerate/` とするが，この関門では検査役の決まりにより `checks/DAI-23/code_run/degenerate/` に書いた．結果の文の根拠に使うなら，本体がこの置き場を名指すか，同じ試しを設計の置き場で流し直す．
- 計算の後で直せるか：(1)〜(3) は直せる．
- 求める対応：(1) を summary に入れる．ほかは provenance に足す．

### [I10] 6．単位ごとに GPU の空きブロックを返さない
- 重さ：記録する
- 該当箇所：exp/dai23.py:565
- 問題：`torch.cuda.empty_cache()` はコーパスの終わりにだけ呼ぶ．試しの 6 単位で GPU の使用は全体で最大 4.0 GB（ほかの処理の約 1.7 GB を含む）で，6 割の上限の内側なので，本番で困る見込みは小さい．
- 求める対応：単位の終わりに `empty_cache()` を足すとよい．

## 前の版で見つけて直ったこと
- c8d7182 の `kappa_from_dai14` は `x` を `subset == "all"` に絞った後で nomarker・nonshared の行を探し，`IndexError` で落ちた（CPU で確かめた）．1376f64 で直り，7 組とも設計の値と一致する．
- 91af3db では `fc.json` の `n_per_corpus` を誰も見ず，`--n 4` を本番の置き場に書けた．1376f64 は本番の置き場で `--n 100` だけを受け付ける．
- 91af3db では本番でない置き場で L・B に任意の特徴を渡せた．1376f64 は audio_wavlm の両方だけを受け付ける．

## 目録に無い種類の不備
- 途中保存に 0 次元の配列を入れると，共通の結合の関数（`common.merge_npz`，`common.load_cache`）で落ちる．特徴の台本に列を足すときは，途中保存のすべての配列が発話の数の長さを持つかを確かめ，試しで `--merge` と `load_group` まで小さく流す（I10 への追記の案）．
- 自己検査（selftest）は provenance を書く行を通らないので，設定の辞書の型を変えると本番の段の初めで落ちる．台本を直したら，selftest だけでなく小さい単位の段を 1 回流す（I10 への追記の案）．

## 確かめた項目
I1，I2，I3，I4，I5，I6，I7，I8，I9，I10，I11，I12，I14，I17，I18，I20，I21，R1，R2，R3，R4，R6，R7，R8，R11，R13．
