# 感情認識の査読者の検査：DAI-13 の下調べの設計の第 1 版（2026-11-09．第 1 回）
判定：条件つき OK

重大な指摘は 1 件（A1）で，設計の文書の 4.2 節・4.4 節・8 節を直せば解ける．台本を変える必要は無い（台本は A1 に要る数を既に出している）．ほかに，計算を始める前に本体が知っておくべき運用の事実が 2 件ある（a1：全層の特徴の取り出しが止まっている，a2：`final` の照合の止める条件が最初の AFEW の単位で働くおそれ）．これらは感情認識の側の欠陥ではないので軽微に置いた．

検査したもの（sha256 の先頭）：設計 `DAI/docs/実験13_下調べ_設計.md` 第 1 版（`6c6d3d0b…`），台本 `DAI/exp/dai13_probe.py`（`b8c7e3bf…`．787 行），起動の台本 `DAI/run/dai13_probe.sh`（`25b0302c…`），特徴の取り出し `features/build_audio_layers.py`（`c6623181…`）．いずれも未コミット．参考に DAI-12 の設計の第 6 版と，自分の DAI-12 の第 1〜6 回の報告，`data/README.md` の 1・3・4・7 節，`docs/seen-data.md` の 213〜214 行，DAI-11・DAI-12 の部品（`dai11_probe.Unit2`，`dai12_probe.balanced_halves`・`crossfit_nets`・`effect_stats_pool`・`_robust`，`dai3.subset_masks`，`dai_data.clusters`），`DAI/splits/*.csv`，`docs/exploratory-analyses.md` の E63 の節を読んだ．DAI-12・DAI-13 の性能の値は見ていない（下の「この検査で私が見たもの」）．

この報告で使う語：
- **候補**：仮のターゲット c のうちテスト以外の発話（ラベル無しとして扱う．K 本はこの中から抜く）．**塊**：話者を分ける単位（IEMOCAP はセッション，MOSEI はチャンネルか動画，MELD は登場人物，AFEW は俳優と作品のつながり）．
- **2 つの半分と交差の予測**：候補の塊を 2 組に分け，一方で学んだネットでもう一方を予測すること．
- **出所 `pair`**：顔と言葉の分類器の確率の積．音声の分類器がこれを目標に学ぶ．**入れ替えの対照 `pair_shuf`**：`pair` の確率を，同じ感情の発話の間で 2 つの半分をまたいで入れ替えた目標．特徴との結びつきは 4 感情のラベルを通る経路だけになる．
- **層の重みつき和（`lw`）**：WavLM-large の 25 個の隠れ状態（入力 1 ＋ 24 層）の時間平均を，softmax で正規化した 25 個の重みで足し合わせた表現．重みと線形の写像を `pair` を目標にラベル無しで学ぶ．**`final`**：添字 24（最終層）だけを使う同じ計算．主の量 `layers_minus_final` は候補の上の UAR の差．
- **部分集合の頑健さ**：MOSEI はチャンネルが分かる塊の行（`channel`），MELD はその発話の対話が学習の半分に無い行（`dlgfree`）に絞って同じ差を出し，区間が 0 をまたがず，符号が全体と同じで，点推定が全体の半分以上であること．

## 重大な指摘

### [A1] 入れ替えの対照の読み (A)(B)(C) が，「DAI-12 と同じ読み」と書きながら DAI-12 第 6 版の (A)(B)(C) と中身が違い，`pair_shuf` の上積みへの部分集合の頑健さの条件と IEMOCAP の筋書きの常時の添え書きが落ちている．中間層は話者の情報が最も読める層なので，この条件は DAI-12 より DAI-13 でこそ要る

- 箇所：
  - 設計 4.2 節 R1 の 3 つ目の項：「`pair_shuf` の `layers_minus_final`（R1shuf）：`pair` と同じ数え方で記述．読みは DAI-12 の 3 つの場合 (A)(B)(C) と同じ：(A) `pair` が取り戻し，`pair_shuf` も同程度 → 感情を通る分（感情そのもの，または感情と結びついた共有）．(B) `pair` が取り戻し，`pair_shuf` は言えない → 感情を通らない共有を含む．(C) `pair` が取り戻さない → 中間層にも無いか，この目標では取り出せない．」
  - 設計 4.4 節の添え書きの一覧：「格子の端（軸ごと），収束，`lw` の損失が `final` を上回った割合（最適化の不足），層の重みの集中，R1c，部分集合，種が 1 つ，DAI-12 の `lince`・`lab_lince` との判定の不一致の割合．」
  - 設計 8 節：「中間層は話者・録音の情報も多く持つ（SUPERB の話者識別の層の重みは浅い層に寄る）．`pair_shuf` の対照と部分集合で近道の分を分けるが，IEMOCAP の筋書き，MELD のエピソード，MOSEI の人物の共有は除けない（DAI-12 の 8 節と同じ）．」
- 問題：
  1. DAI-12 第 6 版 4.2 節の (A)(B)(C) は，`pair` が「上回った」ときに `pair_shuf` の側を 3 つに分ける：(A) `pair_shuf` も 3 本以上の K で上回り，**MOSEI・MELD では部分集合でも頑健**（有利な読み），(B) `pair_shuf` は上回ったが部分集合で頑健でない（「半分をまたぐ人物・場面ごとの感情の偏りと分けられない．有利とも不利とも書かない」），(C) `pair_shuf` は上回らない（不利な読み）．DAI-13 の (A) には部分集合の条件が無く，DAI-13 の (B) は DAI-12 の (C) に当たり，DAI-13 の (C) は DAI-12 に無い場合（`pair` 自体が取り戻さない）である．同じ記号で違う場合を指すので，結果の文を書くときに DAI-12 の読みと取り違える．
  2. DAI-12 の (B) を落とすと，MOSEI・MELD で `pair_shuf` の上積みが同じ人物の別の動画（MOSEI は候補の 68.5% が動画 ID の塊．`data/README.md` 3 節）や同じ対話・同じエピソード（MELD は一方の半分の発話の対話の 64.0〜72.7% がもう一方の半分にもある．DAI-12 2 節）の感情の偏りから出ていても，「感情を通る分」と書くことになる．DAI-12 の第 3・4 回で私が A3・A4 として直させた点と同じである．
  3. DAI-13 ではこの懸念が DAI-12 より大きい．DAI-12 は最終層の上で非線形のネットを足しただけだったが，DAI-13 は層の重み α を `pair` との損失で選ぶので，表現そのものが話者・場面の情報が読みやすい層へ動きうる．WavLM の原論文は Large について「the top layers contribute most to content and semantic tasks, while the middle layers have a great impact on speaker tasks」と書く（Chen ほか 2022，https://arxiv.org/abs/2110.13900 の SUPERB の層の重みの節）．つまり設計の見込み (f) が重みの山を置く中間層（添字 6〜18）は，WavLM-large では話者の課題に最も効く層である．8 節の「SUPERB の話者識別の層の重みは浅い層に寄る」は Base の記述で，Large には当たらない．
  4. IEMOCAP の即興は 8 種類の筋書きが 2 つの半分の両方にあり，筋書きごとの感情の偏りはラベルを通る経路に入るので，入れ替えの対照でも消えない（DAI-12 第 6 版 3.6 節 (b)・4.4 節．私の第 6 回の a39）．DAI-12 ではこれを R1 の常時の添え書きにしたが，DAI-13 の 4.2・4.4 節には無い．8 節に 1 文あるだけでは，結果の文の IEMOCAP の行に添え書きが付かない．
  5. 4.4 節の添え書きの一覧には，DAI-12 4.4 節のラベルの質の添え書き（MOSEI の Neutral はどの感情の強さも 0.5 未満という消去法で決まる，MELD の注釈は 3 人の多数決で一致の κ は 0.43，IEMOCAP の Happy は hap と exc の合併）も無い．Q2（候補の全ラベルで学んだ分類器）の差を c ごとに記述するので，MOSEI の Q2 には消去法の Neutral の添え書きが要る（DAI-12 の a27・a32 と同じ理由）．
- DAI の基準との関係：報告の誤り．主の問い「最終層に無い感情の情報が中間層にあり，ラベル無しの目標で取り出せる」に対して，人物・場面・筋書きの共有から出た上積みを「感情を通る分」と書く文が出うる．
- 直し方（設計の文書だけ．台本は変えない）：
  1. 4.2 節の入れ替えの対照の項を DAI-12 第 6 版 4.2 節の形に書き直す：`pair` の R1 が「取り戻した」のとき，(A) `pair_shuf` の `layers_minus_final` も 3 本以上の K で上回り，MOSEI・MELD ではその K が部分集合（`channel`，`dlgfree`）でも 3 本以上頑健 → 感情を通る分（有利な読み．`pair` 自身に部分集合の「分けられない」が付くときは有利な読みは入れ替えた目標の側にだけ当たる），(B) `pair_shuf` は 3 本以上で上回ったが部分集合で頑健な K が 2 本以下 → 人物・場面の偏りと分けられない（有利とも不利とも書かない．MOSEI にはチャンネルの印の限界を添える），(C) `pair_shuf` の上回る K が 2 本以下 → 感情を通らない結びつきの疑い（`pair_shuf` の検出力の非対称も添える）．`pair` が取り戻さない場合は (A)(B)(C) の外に別の記号で置く．台本は `criteria.csv` の R1shuf の行に `sub_channel_n_K_robust`・`sub_dlgfree_n_K_robust` を既に出す（台本 653〜655 行．`t_` の全部について `_robust` を呼ぶ）ので，数はそろう．
  2. 4.2 節に「IEMOCAP には常時の添え書きを付ける」と書き，4.4 節に DAI-12 4.4 節の 1 文（「IEMOCAP 即興の結果は，筋書きの話題の共有と分けられない．筋書きは 2 つの半分の両方にあり，入れ替えの対照でも消えない」）を写す．層の重みつき和について 1 句足す：「層の重みは筋書きが読みやすい層へも動きうる」．
  3. 4.4 節にラベルの質の 3 件（MOSEI の消去法の Neutral，MELD の κ 0.43，IEMOCAP の hap+exc）を写し，MOSEI の Q2 の文に消去法の Neutral の添え書きを付けると書く．
  4. 8 節の「SUPERB の話者識別の層の重みは浅い層に寄る」を，WavLM の原文に合わせて「WavLM-large では話者の課題に効くのは中間層である（Chen ほか 2022）」に直し，見込み (f) の中間層と重なることを 1 文で書く．

## 軽微な指摘

- [a1] **全層の特徴の取り出しが動いていない．**設計 6 節と台帳（`docs/seen-data.md` 214 行）は「2026-11-09 15:48 に tmux `mer-da` の窓 `dai13feat` で起動した」と書くが，16:04 の時点で `ps` に `build_audio_layers.py` の過程は無く，tmux `mer-da` の 9 つの窓に `dai13feat` は無く，`data/cache/parts/` に `audio_wavlm_layers_*` は 1 本も無い．`logs/dai13_feat.log`（1,544 バイト，15:48 で更新が止まる）はモデルを載せた行と「=== AFEW: 795 発話 ===」と警告 2 行で終わり，1 つ目の塊（1,000 件）の保存の行が無い．標準エラーが記録に出ていれば落ちた理由が残るはずなので，起動の形（記録先）を確かめてから起動し直す必要がある．同じ GPU で DAI-12 の単位の段が 10.7 GB を予約したまま走っている（16:03 で 112/120 単位）ので，起動し直すなら DAI-12 の終了を待つか `--budget` を下げる．起動の台本 `dai13_probe.sh` は `logs/dai13_feat.done` を要求するが，`build_audio_layers.py` はこのファイルを作らないので，起動の命令の側で終わりに `touch` する必要がある．記録にはもう 1 点，Hugging Face の警告「`encoder.pos_conv_embed.conv.weight_g`・`weight_v` が使われず，`parametrizations.weight.original0/1` が新しく初期化された」が出ている．これは重みの正規化の表し方の版の違いで出る警告で，試走の添字 24 と既存の `X` の最大の差 1.66e−5 は，既存の音声の特徴と同じ重みで計算できていることを示す．設計の 2 節か 6 節にこの警告と照合の結果を 1 行書いておくと，後で「位置の畳み込みが乱数のままではないか」と問われたときに答えられる．
- [a2] **`final` と DAI-12 の `lince` の判定の不一致が 10% を超えたら止める条件（設計 3.2 節，台本 57・401 行）は，最初の AFEW の単位で働くおそれがある．**DAI-12 第 6 版 8 節は「wd を 1 点足しただけで `pair_shuf` の `lince` の判定が AFEW で 16%，MELD で 11% 変わった（初期値は束ねるネットの数で変わる）」と記録する．DAI-13 の `final` は 15 本 × 5 wd ＝ 75 のネットを束ね，DAI-12 の `lince` は 35 本 × 5 wd ＝ 175 を束ねたので，初期値の列が違い，`pair_shuf` では同じ桁の不一致が出ると見込む．`pair_shuf` は目標の行ごとの雑音が大きく，候補の本数（AFEW 632）が次元（1,024）より少ないので，初期値で判定が動きやすい．止まると計算が始まらないので，gate-code と相談して，(i) 止める条件を `pair` と `self` に限って `pair_shuf` は記録だけにする，(ii) または閾値を上げて全部の値を `checks_units.csv` に残す，のどちらかに決め，設計 3.2 節の 1 文もそろえる．これは実装の側の点で，感情認識の判定には効かない．
- [a3] **0 節の先行研究の引用は，原典を確かめてから書き直す必要がある．**確かめた範囲：
  - Pepino・Riera・Ferrer 2021（Interspeech．https://arxiv.org/abs/2104.03502）：確認できた．wav2vec 2.0 の局所エンコーダの出力 f₀ と 12 層の出力に学習する重み αᵢ（初期値 1.0）を置いた重みつき平均を使い，「We can see that the middle layers are given larger weights. This could be because these layers have already contextualized enough information from the local encoder, but they are not yet too specific to the pretraining or finetuning task as the last layers」「the layers of the wav2vec2-FT model closer to the output are less useful for emotion recognition than those of the wav2vec2-PT model」と書く（原文 3.3 節と図 2．一時領域に `pdftotext` で写して読んだ）．設計の「感情の情報は中間層に多く，最終層は文字起こしの側に寄る」のうち，後半の「最終層は文字起こしの側」が当てはまるのは ASR で微調整した版（FT）で，事前学習だけの版（PT）では出力に近い層の重みがむしろ大きい．DAI-13 が使う WavLM-large は事前学習だけの版なので，引用するなら「中間層の重みが大きい」の部分に限る．
  - Li・Mohamied・Bell・Lai 2022（SLT．https://arxiv.org/abs/2210.02595）：設計には無いが，主張に最も近い．wav2vec2-base で「for emotion recognition, the middle layer alone performs about as well as averaging across layers; the final layer is the weakest in some cases」と要旨にある（検索の要約で確認．原文の表は未読）．足す候補である．
  - Diatlova ほか 2024「Adapting WavLM for Speech Emotion Recognition」（https://arxiv.org/abs/2405.04485）：MSP-Podcast で WavLM-large を微調整した論文で，層の重みつき和は背景の選択肢として触れるだけであり，層ごとの分析の根拠にはならないと見込む（検索の要約．原文は未読）．層の主張の出典には使わない．
  - 「Yang ほか 2024」：どの論文か特定できなかった．Yang ほか 2024「A Large-Scale Evaluation of Speech Foundation Models」（TASLP．https://arxiv.org/abs/2404.09385）なら，課題ごとの層の重みの分析が V-B1 節にある．原文で確かめてから引く．
  - EMO-SUPERB（Wu ほか 2024．https://arxiv.org/abs/2402.13018）は，感情認識の層の重みは自己教師ありモデルごとに違い「generally tend to prioritize the shallow layers」と書く（検索の要約）．見込み (f)（添字 24 の重みは 0.3 未満，山は添字 6〜18）は計算の前の予言として残してよいが，結果の文で「文献どおり」と書くなら，この食い違いを添える．
  - WavLM の原論文（Chen ほか 2022．https://arxiv.org/abs/2110.13900）の Large の記述は A1 の 3 に書いた．
- [a4] **候補の部分集合の扱いが設計と台本で違う．**設計 2 節・4.2 節は MOSEI `channel` と MELD `dlgfree` だけを使うと書くが，台本は DAI-12 の保存の `pool_sub_*` を全部写し（325〜327 行），集計で全部の部分集合の差と頑健さの数を出す（534・541〜544・653〜655 行）．IEMOCAP の `marker`・`nomarker`（顔の記述用），MELD の `face_single`（顔の記述用），AFEW の `asr_ok`（Whisper の書き起こしが疑わしくない行）が余分に出る．このうち `asr_ok` は音声の `pair` に意味がある（`pair` の言葉の成分は Whisper の書き起こしから作るので，書き起こしが疑わしい行では目標そのものが崩れている）．設計 4.4 節に「ほかの部分集合は記述として表に出す（`asr_ok` は目標の質の目安）」と 1 文書けば，台本を変えずに済む．
- [a5] **「取り戻した情報」の中身の読みに 2 つの可能性を添える．**(i) 言葉の内容：WavLM の上位の層は音素・単語の情報を持つ（Chen ほか 2022 の「content and semantic tasks」）．`pair` の言葉の成分が使う単語の手がかりを音声の中間層から読めば，候補の UAR は本当に上がるが，それは「言語的な情報」であって「声の調子（パラ言語）の情報」ではない．(ii) 録音の環境：AFEW は映画の音声（音楽・効果音・複数の声），MELD はテレビの音声（笑い声の効果音）で，感情と結びついた環境の差は入れ替えの対照でも消えない．どちらも設計 8 節の「中間層から取り戻せる分しか測れない」の中に入るが，結果の文で「中間層に感情の情報がある」と書くときに，この 2 つを分けていないと 1 文添えるのがよい．言葉の成分を外した目標（顔だけ）は作らないと設計に書いてあるとおりでよい（新しい実験の案は出さない）．
- [a6] **4.1 節に「K は同じ候補を共有するので 3 本以上は 3 回の独立な確かめではない」の 1 文が無い．**DAI-12 4.1 節にはある．種が 1 つで区間が種の揺れを含まない点は書かれているので，この 1 文を足せばそろう．
- [a7] **7 節の「候補のラベルを使うのは…」の列挙は正しい．**付け足すなら，`pair_shuf` の格子の選択は入れ替えた目標との損失で行うので，目標を通してラベルの情報が選択にも間接に入る（DAI-12 7 節の文）．記述だけの違いである．

## 確かめたこと

### コーパスの扱い・ラベルの対応・話者と内容の分離（DAI-12 から変わらない）

- 読み込みは `dai11_probe.Unit2` → `dai_data.load_corpus` で，IEMOCAP は即興だけ（`GROUP["IEMOCAP"] = "IEMOCAP-impro"`），MELD は `All`・`Both`・`The_Guys` と `_and_` の合同発話を落とし別名をまとめ，AFEW は `Bradely` を `Bradley` にまとめる．塊は IEMOCAP がセッション（`DAI/splits` で 5 塊），AFEW が俳優と作品のつながり（51 塊），MELD が登場人物（272 塊），MOSEI がチャンネルか動画（2,614 塊）．テストは塊で分けた分割 f の塊，候補はそれ以外で，標準化の統計は候補だけで作る（`Unit2.__init__`．DAI-13 の全層の標準化も `Xl_pool` の統計だけ．台本 285〜286 行）．
- ラベルは 0=Neutral 1=Happy（IEMOCAP は hap+exc）2=Sad 3=Angry で，fru は入れない（`data/README.md` 3 節）．DAI-13 は新しいラベルを作らない．
- 全層の特徴は `sample_id` で引き（`LayerFeats.rows`），候補・テストの `sample_id` の並びを DAI-12 の保存と照合して違えば止める（275〜276 行）．`ok` が偽の行があれば止める．添字 24 の生の特徴と既存の音声の特徴の最大の差が 1e−2 を超えれば止める（283〜284 行）．
- 2 つの半分は `balanced_halves`（塊で分ける．ラベルを使わない）で，DAI-12 の保存 `pool_half` と 1 本も違わないことを要求する（313 行）．入れ替えの元の行も DAI-12 と同じ乱数で作り 0 本の違いを要求する（323 行）．部分集合の印は DAI-12 の保存を写す（`dlgfree` は同じ半分で定義されているので，そのまま使える）．
- テストに当てた判定（`K{K}_A_lw_pair` など）は保存にだけ使い，選ぶものは無い（選択は予測する半分の損失．`train_lw_batched` の `ce_te`）．Q4 の 4×4 の読み出しは K 本の行（`u.kpos[K]`）だけで学び，K 本を除いた候補で測る（`readout_kshot`，443 行）．
- 候補の本数と塊：`DAI/splits` から数えた（AFEW 790 発話，IEMOCAP 2,943，MELD 11,267，MOSEI 18,634．感情の内訳は AFEW 205/211/177/197，IEMOCAP 1,099/947/608/289，MELD 6,383/2,283/999/1,602，MOSEI 8,637/6,734/1,814/1,449）．IEMOCAP の候補は 4 セッションで半分は 2 セッションずつ，Angry の最小は分割によって 53 本（DAI-12 2 節）．DAI-13 で変わらない．

### コーパス間の評価の慣行

- 主指標は UAR．候補の上の量は K 本を除いた発話で，区間は `effect_stats_pool`（塊を分割をまたいで 1 回だけ引き直す．AFEW・IEMOCAP・MELD は塊の jackknife の枝）．種が 1 つなので `v_seed` は `R[f] > 1` でないため 0 になる（`effect_stats_pool` の `s2`）．設計 4.1 節の添え書きと一致する．
- 下限 1.0，「上回る」は下端 > 0 かつ点推定 ≥ 1.0，3 本以上の K，の規則は DAI-12 と同じ．`_verdict` と `_robust` を DAI-12 からそのまま使う．

### モダリティごとの事情

- 全層の特徴の取り出しは `build_audio.py` と同じ前処理（16 kHz，先頭 60 秒，平均 0・標準偏差 1）で，詰め物を除いた時間平均を 25 個の隠れ状態それぞれに取る（`encode_layers`）．`hidden_states[-1]` が `last_hidden_state` と等しいことを毎バッチ確かめる（59〜60 行）．60 秒で切られるのは MOSEI 8 件・MELD 1 件で，既存の特徴と同じ．
- `pair` の顔・言葉の成分は DAI-12 と同じ（`none_wb`）．AFEW の言葉は Whisper の書き起こしで，a4 の `asr_ok` がその目安になる．

### 感情ごとの解析

- 感情ごとの主張はしない．候補の判定（int8）と `pool_y`・`pool_sample_id`，テストの判定と `test_y` を保存するので，感情ごとの再現率は後から出せる．

### 先行研究との比較・データの申告

- 新しいデータもモデルも無い（同じ 4 コーパス，同じ WavLM-large の重み）．利用条件と倫理の申告は DAI-12 までと同じ．Hugging Face のモデルの版が固定されていない点（`data/README.md` 7 節）は，試走の添字 24 と既存の `X` の差 1.66e−5 で同じ重みと分かる（a1）．
- 0 節の引用の確かめは a3．

### 運用の状態（16:04 JST）

- DAI-12 の単位の段は走行中（`results/dai12_probe/rows/` に 112/120 単位．ログの末尾は MOSEI_MELD の単位の本数・所要・GPU の行で，UAR は含まない）．`DAI/checks/DAI-12/code_post.md` は無い．起動の台本 `dai13_probe.sh` はこの 2 つと `logs/dai13_feat.done` を要求するので，いまは正しく止まる．
- 全層の特徴の取り出しは a1 のとおり止まっている．

### この検査で私が見たもの（台帳のため）

設計・台本・起動の台本・特徴の取り出しの台本の全文と sha256，DAI-12 の設計の第 6 版，自分の DAI-12 の報告（第 1〜6 回の判定と A1〜A4 の題），`data/README.md` の 1・3・4・7 節，`docs/seen-data.md` の 193〜214 行，`dai11_probe.py` 145〜232 行，`dai12_probe.py` 66〜135・175〜315・380〜410・651〜690 行，`dai3.subset_masks`，`dai_data.py` の該当行，`dai1_stats.py` の定数，`docs/exploratory-analyses.md` の E63 の節（表の値を含む．2026-10-25 の公開済みの記録），`DAI/splits/*.csv` の塊・分割・感情の本数．`logs/dai13_feat.log` の全文（警告と見出しだけ），`logs/dai12_probe_units.log` の末尾 3 行（本数・所要・GPU．性能の値は無い），`ps`・`tmux`・`nvidia-smi` の出力．ウェブ：Pepino 2021 の PDF を一時領域に `pdftotext` で写して層の重みの節を読んだ．WavLM の原論文は ar5iv の本文の該当の文．Li 2022・Diatlova 2024・Yang 2024・EMO-SUPERB は検索の要約だけ（原文は未読）．DAI-12・DAI-13 の結果ファイル（csv・npz）は開いていない．
