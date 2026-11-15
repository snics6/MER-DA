# 音声の基盤モデルを使うコーパス間の音声感情認識（SER）とドメイン適応（2022〜2026）

調べた日：2026-11-08．
読み方の注記：arXiv・ISCA・研究室のサイトの PDF を pdftotext で文字にし，手法・設定・結果の節を読んだ．HTML 版を読んだものは EmoBox，emotion2vec，Dong ほか（TTA）の 3 本である．PDF から取り出した表は列がずれることがあり，数値を表から組み立て直した箇所には「表の文字列から再構成」と書いた．要旨や検索結果の抜粋しか読めなかったものは，その都度「要旨のみ」「抜粋のみ」と書いた．
比べる相手（この研究の設定）：音声は凍結した WavLM の特徴，顔は ResNet-50，言葉は RoBERTa の特徴を使い，線形の softmax 分類器を学習する．ターゲットのコーパスにはクラスあたり K 件のラベル付き発話とラベルなしの発話がある．ソースは別のコーパス 2〜3 個（AFEW，IEMOCAP，MOSEI，MELD から選ぶ）で，4 クラス，主指標は UAR（クラスごとの再現率の平均）である．

## 問い 1：どの音声基盤モデルが標準で，どう使われているか

### Takeaway
2024〜2025 年の論文で最もよく使われる音声基盤モデルは WavLM（特に Large）と Whisper の encoder（Large-v3）で，wav2vec 2.0 と HuBERT がこれに続く．使い方は 2 系統に分かれる．ベンチマーク（SUPERB，EMO-SUPERB，EmoBox，emotion2vec の評価）はモデルを凍結して軽い頭部だけを学習する．最高精度を競う論文やチャレンジ（Interspeech 2025 など）は Transformer 層を微調整する．凍結する場合の層の使い方は，全層の重み付き和が多数派で，EmoBox は最終層だけを使う．

### Cited Findings
**ベンチマークでの使い方（凍結）**
- SUPERB（2021 年．2022 年より前の基礎文献）は，事前学習モデルを全タスクで凍結し，全層の隠れ状態を学習可能な重みで足し合わせた（weighted sum）表現を軽い頭部に入れる．発話単位のタスク（感情認識 ER を含む）の頭部は「時間方向の平均プーリング → 線形変換」で，損失は交差エントロピーである — [SUPERB, Yang et al., Interspeech 2021](https://arxiv.org/abs/2105.01051)
- EMO-SUPERB（2024）は 15 個の自己教師ありモデル（論文の表では FBANK を除き 16 個）を凍結し，各層の出力に層ごとの重みを掛けて足し合わせ，層の重みと頭部だけを学習する．AdamW，学習率 0.0001，バッチ 32，クラスの偏りを補正する交差エントロピー（class-balanced CE），ラベルは複数の評定者の分布（soft label）とする — [EMO-SUPERB, Wu et al., 2024](https://arxiv.org/abs/2402.13018)
- EmoBox（Interspeech 2024）は 10 個の事前学習モデル（wav2vec 2.0 base，HuBERT base/large，WavLM base/large，data2vec base/large，data2vec 2.0 base/large，Whisper large v3 の encoder）から特徴を取り出し，**最終の Transformer 層の出力だけ**に層正規化をかけて使う．頭部は「線形の隠れ層 → プーリング → ReLU → 分類層」で，学習率 {1e-3, 1e-4} と隠れ次元 {128, 256} を探索し，100 エポック（最初の 10 エポックで学習率を上げる warm-up）学習する．wav2vec 2.0 large は最終層の特徴でほとんどのコーパスの学習に失敗したため結果から除いた，と脚注に書かれている．著者は「最終層だけを評価した」ことを限界として挙げている — [EmoBox, Ma et al., Interspeech 2024](https://arxiv.org/abs/2406.07162)
- emotion2vec（2024）の下流評価は特徴を凍結し，発話単位のタスクでは 2 層の線形層（間に ReLU，隠れ次元 256）を学習する．感情認識でどの層を使ったかは本文に書かれていない（感情の極性の分析だけは最後の 4 層の平均を使うと書かれている） — [emotion2vec, Ma et al., 2024](https://arxiv.org/abs/2312.15185)

**微調整する使い方**
- Wagner ほか（IEEE TPAMI 2023）は wav2vec 2.0 と HuBERT の CNN 部分を凍結し，Transformer 層を微調整した．同じモデルで Transformer 層を凍結して出力層だけを学習した場合と比べると，覚醒度・支配度・感情価のすべてで微調整のほうが CCC（一致相関係数）が高かった．著者は「微調整は計算の手間に見合う」と結論している — [Wagner et al., 2023](https://arxiv.org/abs/2203.07378)
- Wagner ほかの関連研究の表（IEMOCAP の 4 クラスで wav2vec 2.0 / HuBERT を使った 22 件の結果）から，著者は「事前学習の重みを微調整すると約 10% 上がる」「ASR 向けに追加で微調整したモデルは SER に役立たない（例：−3.2%）」「Large は Base より概ね良い（例：+3.0%，ただし +0.5% のように差が小さい場合もある）」「HuBERT が wav2vec 2.0 より良い（例：+2.1%）」「Transformer 層を微調整するなら平均プーリング＋線形分類器で十分」とまとめている — [Wagner et al., 2023](https://arxiv.org/abs/2203.07378)
- Interspeech 2025 の自然な発話の SER チャレンジ（MSP-Podcast）のベースラインは WavLM-large（24 層，約 3.1 億パラメータ）を微調整し，その上に attentive statistics pooling（フレームに重みを付けて平均と標準偏差を取るプーリング）と全結合層を置く（30 エポック，バッチ 32，Adam，学習率 1e-5）．アンケートに答えた上位チームについて主催者は「ほとんどのチームが音声の基盤モデルを微調整した」と書いている — [Naini et al., Interspeech 2025 Challenge](https://www.isca-archive.org/interspeech_2025/naini25_interspeech.pdf)
- 同チャレンジの上位チーム（カテゴリのタスクに 17 チーム，次元のタスクに 8 チームが回答）では，音声と言葉を組み合わせた構成が 76%，音声の encoder は WavLM が 70%，Whisper が 47%，言葉の encoder は RoBERTa が 53%，LLaMA が 24% だった．アンサンブルは 95%，データ拡張は 60% のチームが使った — [Naini et al., Interspeech 2025 Challenge](https://www.isca-archive.org/interspeech_2025/naini25_interspeech.pdf)
- 同チャレンジの論文の表 2（上位チームの一覧）を数えると，カテゴリのタスクの上位 8 チームでは Whisper が 7 チーム，WavLM が 6 チーム，HuBERT が 4 チーム，wav2vec 2.0 が 1 チーム，SALMONN が 1 チームで使われていた．8 チームすべてが言葉のモデル（RoBERTa，DeBERTa，ModernBERT，BERT/T5，LLaMA-3，GPT-4 のいずれか）も使った（表の文字列から再構成） — [Naini et al., Interspeech 2025 Challenge](https://www.isca-archive.org/interspeech_2025/naini25_interspeech.pdf)
- Odyssey 2024 のチャレンジに出した Adapting WavLM の論文は，WavLM Large を微調整し，全層の重み付き平均 → 射影層 → attention pooling → 線形層，という構成をとる — [Adapting WavLM for SER, 2024](https://arxiv.org/abs/2405.04485)
- Ibrahim ほか（Interspeech 2024）は Whisper base の encoder と decoder の両方を使い，全体を微調整して 4096→2048→1024→512 の全結合層で分類する（学習率 1e-5） — [Ibrahim et al., Interspeech 2024](https://arxiv.org/abs/2406.09933)

**パラメータを少しだけ学習する方法（PEFT）**
- PEFT-SER（Feng & Narayanan，ACII 2023）は事前学習モデルを凍結したまま adapter，並列 adapter，embedding prompt tuning，LoRA を比べた．使ったモデルは Whisper Tiny/Base/Small，wav2vec 2.0 Base，WavLM Base+ で，IEMOCAP，MSP-IMPROV，MSP-Podcast，CREMA-D の 4 つで評価した．著者は「LoRA がすべてのモデルで一貫して最も良い」と書いている — [PEFT-SER, Feng & Narayanan, 2023](https://arxiv.org/abs/2306.05350)．公開パッケージのページでは WavLM Base+ が LoRA なしの 63.06 から LoRA ありの 66.11 に，WavLM Large が 68.54 から 68.66 に上がったとされる（論文本文の表とは照合していない．二次情報） — [peft-ser (PyPI)](https://pypi.org/project/peft-ser/)
- Lashkarashvili ほか（ICASSP 2024）は wav2vec 2.0 Base と HuBERT Large に bottleneck adapter（BA），LoRA，重み付き和（WS），提案の weight-gating（WG）を組み合わせた．IEMOCAP の 4 クラス（happy と excited を統合，5,531 発話），セッションを 1 つ抜く 5 分割の正解率で，凍結（PT）が 51.02 / 60.09，全体の微調整（FT．CNN は凍結）が 66.30 / 68.53，BA+LoRA+WS+WG が 67.00 / 71.88 だった（wav2vec 2.0 Base / HuBERT Large の順．表の文字列から再構成）．学習するパラメータは全体の 2〜3% である — [Lashkarashvili et al., ICASSP 2024](https://arxiv.org/abs/2402.11747)

### Inferences
- 凍結した特徴の上に軽い頭部を置く使い方は，SUPERB 以来のベンチマークの標準として残っている．精度を競う論文とチャレンジでは微調整が主流である．Wagner ほかの表と Lashkarashvili ほかの表では，凍結と微調整の差は IEMOCAP の 4 クラスで数ポイントから 10 ポイント以上ある．
- この研究は凍結した WavLM と線形 softmax を使う．これは SUPERB の ER の頭部（平均プーリング＋線形）とほぼ同じ形であり，ベンチマークの慣習には沿っている．ただし最高精度の比較相手（微調整したモデル）と同じ土俵には立たない．査読で「なぜ微調整しないか」を問われる可能性がある．
- 層の使い方は論文によって違う（重み付き和，最終層，最後の 4 層の平均，平均プーリングのみで層の記載なし）．この研究が WavLM のどの層をどうまとめたかを論文に明記する必要がある．

### Gaps
- Goron ほか「Improving Domain Generalization in SER with Whisper」（ICASSP 2024）は書誌情報しか得られず（オープンアクセスでない），手法と数値は確認できなかった — [dblp](https://dblp.org/rec/conf/icassp/GoronARD24)
- Odyssey 2024 チャレンジの論文本体（Goncalves ほか）は取得できなかった（サーバーの名前解決に失敗）．ベースラインが WavLM で，カテゴリのタスクの Test3 で macro-F1 0.311 という値は Hugging Face のモデルカード（二次情報）でしか見ていない — [HF model card](https://huggingface.co/3loi/SER-Odyssey-Baseline-WavLM-Categorical)
- PEFT-SER の本文の表の数値（UAR か正解率か）は本文から取り出せなかった．

## 問い 2：ベンチマーク（EmoBox，emotion2vec，SUPERB / EMO-SUPERB，その他）

### Takeaway
基盤モデルを使ったコーパス間の評価を正式に持つベンチマークは，調べた範囲では EmoBox だけである．EmoBox は 4 コーパス（IEMOCAP，MELD，RAVDESS，SAVEE）を 1 つずつ学習に使い，残りの 3 つで試す 12 組を，凍結した最終層の特徴で評価した．12 組のうち 9 組で Whisper large v3 の encoder が最も高かった．SUPERB と EMO-SUPERB は同じコーパスの中での評価だけを持ち，emotion2vec の「言語をまたぐ評価」は，英語で事前学習したモデルの特徴を各言語のコーパスの中で学習・評価したもので，コーパスをまたいで分類器を移す評価ではない．

### Cited Findings
**EmoBox（Ma ほか，Interspeech 2024）**
- 14 言語の 32 データセットを集めた．同じコーパス内の評価では，公式の分割があるもの（IEMOCAP，MELD など）はそれを使い，ない場合は話者数や感情の偏りに応じて，話者を分けない分割（感情ごとに 25% を評価用）か，話者を 1 人ずつ抜く交差検証を使う．学習データの 20% を検証に回す．指標は UA（クラスごとの再現率の平均＝UAR），WA，macro-F1 — [EmoBox](https://arxiv.org/abs/2406.07162)
- コーパス間の評価：IEMOCAP，MELD，RAVDESS，SAVEE を選び，1 つで学習して別の 1 つで試す 12 組を「ゼロショット」（ターゲットのデータを学習に一切使わない）で評価する．評価用データは，emotion2vec の微調整版（1 万時間超の音声で繰り返し微調整したもの）で付けた予測ラベルと元のラベルが一致する発話だけを残し，コーパスごとに 4 感情（angry，happy，neutral，sad）× 60 件＝240 件のクラス均衡な集合にした．IEMOCAP は話者 10 人全員，MELD は主要登場人物 6 人，RAVDESS は 20 人，SAVEE は 4 人全員を含む — [EmoBox](https://arxiv.org/abs/2406.07162)
- コーパス間の結果は正解率（%）で報告され，12 組の平均（式 1，n=4）も示される．Whisper large v3 の encoder が 12 組中 9 組で 1 位，HuBERT large，WavLM large，data2vec base が 1 組ずつ 1 位だった．Whisper を除くと，Large の中では WavLM，Base の中では HuBERT base の平均が最も高かった — [EmoBox](https://arxiv.org/abs/2406.07162)
- 組ごとの値（HTML 版の表 5 の文字列から再構成したもので，列の対応は PDF で再確認していない）：IEMOCAP→MELD は WavLM large 48.59（Whisper 46.14），MELD→IEMOCAP は Whisper 51.42，RAVDESS→SAVEE は Whisper 66.91，RAVDESS→MELD は Whisper 40.68，SAVEE→IEMOCAP は Whisper 49.30，MELD→SAVEE は HuBERT large 43.74，IEMOCAP→RAVDESS は data2vec base 40.46．どの組も 4 クラスの偶然の水準 25% を上回るが，多くは 40〜50% にとどまる — [EmoBox (HTML)](https://arxiv.org/html/2406.07162)
- 同じコーパス内の評価では Whisper large v3 が 32 データセット中 23 で 1 位，30 で上位 3 位以内，WavLM large が 31 で上位 3 位以内だった．著者は Whisper の強さを多言語の学習データとモデルの大きさに帰している — [EmoBox](https://arxiv.org/abs/2406.07162)
- 学習側のコーパスを 4 クラスにどう写したか，学習側にコーパス全体を使ったか，ラベルを絞った版を使ったかは，本文に書かれていない — [EmoBox](https://arxiv.org/abs/2406.07162)

**emotion2vec（Ma ほか，2024）**
- data2vec / data2vec 2.0 から初期化し，生徒と教師（教師は生徒の指数移動平均）による自己蒸留で，発話単位とフレーム単位の損失を組み合わせて事前学習した．事前学習データは英語のみの 262 時間で，**IEMOCAP（7.0 時間），MELD（12.2 時間），MEAD（37.3 時間），CMU-MOSEI（91.9 時間），MSP-Podcast（113.5 時間）**からなる（ラベルは使わない） — [emotion2vec](https://arxiv.org/abs/2312.15185)
- IEMOCAP の 4 クラス（excited を happy に統合）でセッションを 1 つ抜く 5 分割と話者を 1 人抜く 10 分割を使う．凍結した特徴の WA は wav2vec 2.0 base 63.43，HuBERT base 64.92，WavLM base 65.94，WavLM base+ 67.98，data2vec 67.38，data2vec 2.0 68.58，HuBERT large 67.62，WavLM large 70.03 で，emotion2vec は 71.79 だった（emotion2vec の行は複数あり，一部は検証と評価に同じ分割を使ったと注記されている） — [emotion2vec](https://arxiv.org/abs/2312.15185)
- 「言語をまたぐ評価」は，英語以外の 9 データセット（M3ED，SUBESCO，CaFE，EmoDB，AESDD，EMOVO，ShEMO，RESD，URDU）のそれぞれで，凍結した特徴に線形の頭部を学習し，10 分割の交差検証（RESD は元の分割）で評価したものである．emotion2vec は 9 つすべてで data2vec 2.0 を上回ったが，差が 1 ポイント未満のもの（RESD 64.75 対 64.08，EmoDB 84.34 対 83.77，EMOVO 61.21 対 60.69）もある — [emotion2vec](https://arxiv.org/abs/2312.15185)
- 著者は RAVDESS と SAVEE は事前学習で見ていないと述べる — [emotion2vec](https://arxiv.org/abs/2312.15185)

**SUPERB と EMO-SUPERB**
- SUPERB の ER は IEMOCAP を使い，「従来の評価プロトコル」として数の少ないクラスを落として neutral，happy，sad，angry の 4 クラスにし，標準の 5 分割で交差検証し，正解率（ACC）で評価する — [SUPERB](https://arxiv.org/abs/2105.01051)
- EMO-SUPERB は 6 つの公開コーパス（IEMOCAP，CREMA-D，MSP-IMPROV，MSP-Podcast，NNIME，台湾華語版の MSP-Podcast＝B-POD）で 9 条件を作り，対話の両方の話者を同じ側（学習か評価）に入れる話者独立の分割を定めた．指標は macro-F1．平均で最も高かったのは XLS-R-1B で，WavLM は 9 条件中 3 条件で最も高かった．コーパスをまたぐ評価は含まない — [EMO-SUPERB](https://arxiv.org/abs/2402.13018)

**その他**
- Ibrahim ほか（Interspeech 2024）は英語の 11 データセットを 4 感情（または 5 感情）に揃えて合併し，Whisper base を微調整した．各データセットから話者 1 人を評価用に取り分け，残りで 5 分割の交差検証をし，正解率（WA）で評価する．合併して学習したほうが，各データセットだけで学習するより評価が高かったと報告している．ソースだけで学習してターゲットのコーパスで試す評価ではない — [Ibrahim et al., Interspeech 2024](https://arxiv.org/abs/2406.09933)
- Interspeech 2025 のチャレンジは MSP-Podcast の中の話者独立な分割で，8 クラス（anger，happiness，sadness，fear，surprise，contempt，disgust，neutral）のカテゴリ分類（クラス均衡な評価集合，macro-F1）と，覚醒度・感情価・支配度の回帰（CCC）を評価する．ベースラインは macro-F1 0.329，CCC 平均 0.579．1 位は macro-F1 0.4316 だった．93 チームが参加した — [Naini et al., Interspeech 2025 Challenge](https://www.isca-archive.org/interspeech_2025/naini25_interspeech.pdf)

### Inferences
- EmoBox の評価集合は 4 クラス × 60 件で均衡しているので，そこでの正解率は UAR と同じ値になる．この研究の UAR と数値の水準を比べることはできる．ただし EmoBox は (1) 音声のみ，(2) ソースは 1 コーパス，(3) ターゲットのラベルもラベルなしデータも使わない，(4) 評価データを emotion2vec と一致する「分かりやすい」発話に絞っている，の 4 点でこの研究と違う．(4) のため，EmoBox の数値はふつうの評価データより高めに出ている可能性がある．
- emotion2vec を音声の特徴に使う場合，その事前学習に IEMOCAP，MELD，MOSEI の音声（ラベルなし）が入っている．この研究のソースとターゲットはこの 3 つを含むので，ターゲットの評価用の音声を特徴の事前学習が見ていることになる．WavLM Large の事前学習データは Libri-Light 60k 時間，GigaSpeech 10k 時間（オーディオブック・ポッドキャスト・YouTube から集めたもの），VoxPopuli 24k 時間の計 94k 時間で，感情のコーパスを名前で含んではいない — [WavLM, Chen et al., 2022](https://arxiv.org/abs/2110.13900)．GigaSpeech の YouTube 部分と MOSEI の動画が重なるかは確認していない．
- AFEW と MOSEI をコーパス間の評価に使った基盤モデルの SER のベンチマークは，調べた範囲では見つからなかった．

### Gaps
- EmoBox の図 1（12 組の平均正解率）の数値は画像の中にあり，取り出せなかった．
- EmoBox の後続版（2025 年以降の更新）があるかは調べていない．

## 問い 3：自己教師ありの特徴の上でのドメイン適応・汎化の手法と，ソースのみに対する得

### Takeaway
基盤モデルを使う SER で報告が多い適応は，(a) ターゲットの音声で事前学習を続ける方法（TAPT），(b) 話者の情報を敵対的に消す方法，(c) テスト時適応（TTA），(d) 疑似ラベル，(e) ラベルの分布を揃える方法，(f) 少数のターゲットラベルでの微調整，である．得の大きさは条件によって大きく違い，TTA の多くの手法はソースのみより下がった．DANN・MMD・CORAL のような分布を揃える古典的な DA を，凍結した WavLM などの特徴の上で使い，ソースのみとの差を報告した 2022〜2026 年の論文は，このセッションでは本文で確認できなかった．調べた範囲の DANN 系のコーパス間 SER（Emo-DNA など）は MFCC と CNN を入力にしていた．

### Cited Findings
**テスト時適応（TTA）**
- Dong ほか（ICASSP 2026）は wav2vec 2.0 base をソースで微調整し（AdamW，50 エポック，学習率 3e-5），頭部は 256 ユニットの全結合層 1 つとした．エントロピー最小化系 6 手法（Tent，SAR，EATA，SUTA，DSUTA，CEA），疑似ラベル系 2 手法（CoTTA，AWMC），誤差逆伝播を使わない 3 手法（T3A，LAME，FOA）を，バッチ 32 で比べた．指標は正解率と macro-F1 — [Dong et al., ICASSP 2026](https://arxiv.org/abs/2601.16240)
- コーパス間（IEMOCAP↔RAVDESS，共通の 4 クラス）では，ソースのみの正解率が RAVDESS 側 37.8（F1 26.7），IEMOCAP 側 50.0（F1 45.7）だった．T3A は RAVDESS 側で 43.8（+6.0），IEMOCAP 側で 50.2（+0.2）と最も良かった．LAME は RAVDESS 側で 28.7（−9.1），SUTA は 31.1（−6.7），CoTTA は 32.7（−5.1）と下がった — [Dong et al., ICASSP 2026](https://arxiv.org/abs/2601.16240)
- IEMOCAP の台本→即興（演技から自然な発話へ）では，ソースのみ 51.3 に対し最良の LAME が 51.5（+0.2），DSUTA は 46.1（−5.2）だった．著者は「エントロピー最小化と疑似ラベルは一般に効果が低く，その理由は 1 つの確信の高いラベルを仮定することが感情の曖昧さと合わないため」「どの手法も一様には勝たない」と結論している — [Dong et al., ICASSP 2026](https://arxiv.org/abs/2601.16240)

**ターゲットの音声で事前学習を続ける方法（TAPT）**
- Chen & Rudnicky（ICASSP 2023）は wav2vec 2.0 base を，(1) そのまま微調整（V-FT），(2) IEMOCAP の音声で wav2vec 2.0 の事前学習を続けてから微調整（TAPT），(3) TAPT の目的を疑似ラベルの予測に置き換えた P-TAPT，の 3 つで比べた．IEMOCAP の 4 クラス，セッションを 1 つ抜く 5 分割の UA の平均は V-FT 69.9，TAPT 73.5，P-TAPT 74.3 だった（表の文字列から再構成）．SAVEE でも TAPT と P-TAPT が V-FT を大きく上回った．これは同じコーパスの中での評価で，ソースからターゲットへ移す評価ではない — [Chen & Rudnicky, ICASSP 2023](https://arxiv.org/abs/2110.06309)
- emotion2vec は感情のコーパスの音声（262 時間）で自己教師あり学習を続けたモデルとみなせ，凍結した特徴の IEMOCAP の WA が初期化元の data2vec 2.0 の 68.58 から 71.79 に上がった — [emotion2vec](https://arxiv.org/abs/2312.15185)

**話者の正規化**
- Gat ほか（ICASSP 2022）は，感情の分類と同時に話者識別（VoxCeleb，1,251 話者）を学習し，話者識別の損失の勾配で上流のモデルから話者の情報を消す方法を提案した．IEMOCAP の話者独立の 5 分割の WA で，HuBERT Large は 71.9 から 74.2（全パラメータを学習する設定），HuBERT Base は 68.9 から 69.9 に上がった（表の文字列から再構成）．凍結した HuBERT Large の上で話者識別器を学習すると正解率は 60.7% で，提案手法を通した後は 45.9% に下がった — [Gat et al., ICASSP 2022](https://arxiv.org/abs/2202.01252)
- Pepino ほか（Interspeech 2021．2022 年より前の基礎文献）は凍結した wav2vec 2.0 の全層の重み付き平均に，話者ごとの平均と標準偏差で正規化をかけた．UAR（論文では average recall）は，話者ごとの正規化で IEMOCAP 67.2，RAVDESS 84.3，全体の統計による正規化で IEMOCAP 65.8，RAVDESS 75.7 だった．著者は「話者ごとの正規化は話者の情報の一部を除き，感情の特徴に集中させる」と説明している — [Pepino et al., Interspeech 2021](https://arxiv.org/abs/2104.03502)

**疑似ラベルと少数のターゲットラベル**
- Agarla ほか（arXiv 2207.06767v2，2023）は英語（RAVDESS，SAVEE，TESS を合併）をソースにし，フランス語（CaFE），ドイツ語（EmoDB），イタリア語（EMOVO），ペルシア語（ShEMO）をターゲットにした．5 クラス（anger，fear，happiness，neutral，sadness），話者独立の 60/20/20 分割，指標は UA（＝UAR）．HuBERT base が encoder の中で最も良かった．ソースのみで学習した HuBERT のターゲットの正解率はドイツ語 78.87%，ペルシア語 33.24% だった．ターゲットのラベル付き発話を **合計で** 0，25，50，100 件とし，残りに硬い疑似ラベル（hard）か柔らかい疑似ラベル（soft）を付けて学習した．硬い疑似ラベルのほうが良く，柔らかい疑似ラベルはソース側の性能も落とした．ペルシア語では下限（ソースのみ）より約 60% 上がった．HuBERT 自体を更新したかは，読んだ箇所からは確認できなかった — [Agarla et al., 2023](https://arxiv.org/abs/2207.06767)
- Chen & Yu（Interspeech 2023）は wav2vec 2.0 base を 4 つの感情コーパス（MSP-IMPROV，MSP-Podcast，Mandarin Affective Speech，IEMOCAP）の全部分集合（16 通り）で先に微調整し（pre-finetuning），ESD の話者ごと・感情ごとの 2 値分類（その感情か否か）を k ∈ {2, 4, 8, 16, 24, 32, 64} 件で微調整した．合計 33,600 回の試行で，指標は macro-F1．7 条件中 6 条件で，1 コーパスだけで先に微調整すると直接微調整より下がることがあり，コーパスの数を増やすと平均が上がり続け，1 個から 2 個に増やしたときの上がり幅が最も大きかった．ComParE 2016 の特徴と全結合層では定数を予測するより良くならなかった — [Chen & Yu, Interspeech 2023](https://arxiv.org/abs/2302.12921)
- Lashkarashvili ほか（ICASSP 2024）は ESD（演技）で学習したモデルを IEMOCAP の即興（自然な発話に近い）の**ラベル付き**学習データで 2 段目に微調整した．HuBERT Large では，ESD のみで学習したときの IEMOCAP 即興の正解率 35.42 が，2 段目の後 68.52〜74.66 に上がった．一方で ESD 側の正解率は 96.08 から 36.16〜48.25 に下がった（表の文字列から再構成）．2 段目はターゲットの学習分割をすべて使っており，少数ラベルの設定ではない — [Lashkarashvili et al., ICASSP 2024](https://arxiv.org/abs/2402.11747)

**ラベルの分布を揃える方法**
- Naini ほか（ICASSP 2025）は WavLM-large を MSP-Podcast で CCC を最適化して微調整したあと凍結し，特徴抽出器として使う．ソースの MSP-Podcast を，ターゲット（VAM，RECOLA，WHiSER）の覚醒度・感情価の分布に合うように間引いて学習する．ターゲットのラベル付きの一部から分布を推定する場合（Case 2）と，疑似ラベルから推定する場合（Case 3）を比べた．覚醒度の CCC は Case 1（間引きなし）に比べて Case 2 で約 4.3%（VAM），約 3.5%（RECOLA），約 3.0%（WHiSER）上がり，感情価は VAM で約 21%，RECOLA で約 3.2%，WHiSER で約 6.3% 上がった．ターゲットのラベル付きデータを何件使ったかは，読んだ箇所に数値がなかった — [Naini et al., ICASSP 2025](http://www.lab-msp.com/MSP/publications/Naini_2025.pdf)
- 同じ論文は Naini ほか（ICASSP 2024）を引いて「教師なしの DA は，ターゲットのデータが学習データと大きく違うときほど相対的な改善が大きい」と書いている — [Naini et al., ICASSP 2025](http://www.lab-msp.com/MSP/publications/Naini_2025.pdf)．Naini ほか（ICASSP 2024）の要旨は，wav2vec2，HuBERT，WavLM，data2vec を比べ，WavLM の特徴がほとんどのコーパス間の条件で最も良かったとする（要旨のみ．本文は取得できなかった） — [Naini et al., ICASSP 2024 (abstract)](https://aminer.org/pub/65f90a1613fb2c6cf67b8a4e/generalization-of-self-supervised-learning-based-representations-for-cross-domain-speech-emotion)

**DANN・MMD などの分布を揃える方法（自己教師ありの特徴でない例）**
- Emo-DNA（ACM MM 2023）は 6 コーパス（CASIA，EmoDB，EMOVO，IEMOCAP，RAVDESS，SAVEE）の 30 組で，ラベル付きのソースとラベルなしのターゲットから学習する教師なし DA を評価した．入力は MFCC で，encoder は 1 次元 CNN と時間方向の畳み込み網である．比較相手は DANN，DAN，JAN，CDAN，BSP と，コーパス間 SER の既存手法 4 つで，指標は WAR と F1 — [Emo-DNA, Ye et al., 2023](https://arxiv.org/abs/2308.02190)
- AKTLR（Zhao ほか，2023）は音響パラメータの特徴集合の上の転移線形回帰で，EmoDB，eNTERFACE，CASIA で評価した．自己教師ありの特徴は使っていない — [Zhao et al., 2023](https://arxiv.org/abs/2312.06466)
- sADDi（Latif ほか，2022）は 2 つの識別器を使う敵対的 DA に，ラベルなしデータでの自己教師ありの事前学習（合成データの生成を前提タスクにする）を組み合わせた．これは wav2vec 2.0 などの公開された基盤モデルを使う方法ではない（要旨のみ） — [Latif et al., 2022](https://arxiv.org/abs/2204.08625)

### Inferences
- ソースのみに対する得は手法と組で大きく違う．数値が確認できたものでは，TTA の T3A が +6.0（RAVDESS 側の正解率），TAPT が +3.6（IEMOCAP 内の UA），話者の正規化が +1.0〜+2.3（IEMOCAP 内の WA），話者ごとの特徴の正規化が +1.4（IEMOCAP）と +8.6（RAVDESS）（UAR）だった．同じ TTA の中でも −9.1 から +6.0 まで幅があった．
- この研究と最も近い設定（ラベル付きのソース，ターゲットの少数のラベル，ターゲットのラベルなしデータ）を持つのは Agarla ほかである．ただし Agarla ほかは言語をまたぐ演技のコーパスで，ラベル数はクラスあたりではなく合計（25〜100 件），モダリティは音声だけである．
- Naini ほか（2025）の「ソースを間引いてターゲットのラベルの分布に揃える」操作は，ラベルの割合のずれ（label shift）への対処であり，この研究が 4 クラスの割合の違うコーパス（例：MELD は neutral が多い）を混ぜるときに比較の対象になる．
- 調べた範囲では，基盤モデルを使う SER の DA は「モデル自体を動かす」方法（微調整，TAPT，TTA，話者の敵対学習）が多い．凍結した特徴と線形分類器のままで行う DA の報告は少ない．この研究の設定（凍結＋線形）で使える手法は，特徴の正規化，ラベル分布の補正，疑似ラベル，ソースの重み付け，T3A のような頭部だけを調整する TTA に限られる．

### Gaps
- DANN・MMD・CORAL を凍結した WavLM / HuBERT / wav2vec 2.0 の特徴に適用し，ソースのみとの差を報告した 2022〜2026 年の論文を本文で確認できなかった．検索で候補（例：HuBERT と話者の埋め込みと韻律の特徴を使う教師なし適応 [Tabriz 大学の論文誌](https://tjee.tabrizu.ac.ir/article_18417.html?lang=en)，kNN 声質変換によるターゲット音声の変換 [Mote et al., 2024](https://ecs.utdallas.edu/research/researchlabs/msp-lab/publications/Mote_2024.pdf)）は出たが，本文を読めていない．
- コーパス単位の正規化（コーパスごとに特徴の平均と分散を揃える）を自己教師ありの特徴で評価した 2022〜2026 年の論文は見つけられなかった．古典的な文献（Schuller ほか 2010 年ごろのコーパス間の研究）もこのセッションでは確認していない．
- Emo-TTA（音声 LLM を使う TTA，2025）は二次情報の要約しか見ておらず，検証していない — [emergentmind の要約](https://www.emergentmind.com/topics/emo-tta)
- クラスあたり K 件のターゲットラベルで K を変えて，基盤モデルの凍結特徴と線形分類器の UAR を報告した論文は見つからなかった．

## 問い 4：標準の評価（コーパス，クラスの対応，指標，話者独立の分割，コーパスを 1 つ抜く評価）

### Takeaway
最も多い評価は IEMOCAP の 4 クラス（neutral，happy＋excited，sad，angry．5,531 発話）を，セッションを 1 つ抜く 5 分割で評価するものである．指標は UA（＝UAR）と WA が並んで報告されることが多く，ベンチマークでは正解率や macro-F1 も使われる．コーパス間の評価は「1 つのソースから 1 つのターゲットへ」の組を総当たりで評価するのが多く，複数のソースを合併して 1 つを抜く評価（leave-one-corpus-out）を基盤モデルで行った論文は，調べた範囲ではなかった．

### Cited Findings
- Wagner ほかは IEMOCAP の 4 クラスを anger 1,103，happiness（＋excitement）1,636，sadness 1,084，neutral 1,708 発話とし，これを「IEMOCAP の典型的なカテゴリの SER の定式化」と呼ぶ．関連研究のほとんどはセッションを 1 つ抜く 5 分割を使い，1 件は話者を 1 人抜く 10 分割を使っていた — [Wagner et al., 2023](https://arxiv.org/abs/2203.07378)
- Lashkarashvili ほか，Dong ほか，emotion2vec，Chen & Rudnicky，Pepino ほか，SUPERB も IEMOCAP の 4 クラス（happy と excited を統合）とセッションを 1 つ抜く 5 分割を使う — [Lashkarashvili et al.](https://arxiv.org/abs/2402.11747)；[Dong et al.](https://arxiv.org/abs/2601.16240)；[emotion2vec](https://arxiv.org/abs/2312.15185)；[Chen & Rudnicky](https://arxiv.org/abs/2110.06309)；[Pepino et al.](https://arxiv.org/abs/2104.03502)；[SUPERB](https://arxiv.org/abs/2105.01051)
- PEFT-SER は IEMOCAP を 5 分割，MSP-IMPROV を 6 分割（各セッションを評価用にする），CREMA-D を話者の 20% を評価用にする 5 分割，MSP-Podcast を標準の分割（test set 1）で評価する — [PEFT-SER](https://arxiv.org/abs/2306.05350)
- EMO-SUPERB は，対話の 2 人の話者を同じ側に入れないと情報が漏れると指摘し，対話の両話者を同じ側に入れる分割を定めた — [EMO-SUPERB](https://arxiv.org/abs/2402.13018)
- EmoBox のコーパス間評価は 4 感情（angry，happy，neutral，sad）× 60 件の均衡な評価集合で正解率を報告する — [EmoBox](https://arxiv.org/abs/2406.07162)
- Dong ほかのコーパス間（IEMOCAP↔RAVDESS）は共通の 4 クラスを使うが，どの 4 クラスかは書かれていない — [Dong et al.](https://arxiv.org/abs/2601.16240)
- Agarla ほかは 5 クラス（fear を含む）で UA を使う — [Agarla et al.](https://arxiv.org/abs/2207.06767)
- MSP-Podcast を使う研究とチャレンジは 8 クラスの macro-F1 か，覚醒度・感情価・支配度の CCC を使う — [Interspeech 2025 Challenge](https://www.isca-archive.org/interspeech_2025/naini25_interspeech.pdf)；[Naini et al., 2025](http://www.lab-msp.com/MSP/publications/Naini_2025.pdf)；[Wagner et al., 2023](https://arxiv.org/abs/2203.07378)
- Wagner ほかは MSP-Podcast で学習したモデルを IEMOCAP 全体（評価だけに使うので話者の分割をしない）と MOSI に当てた．w2v2-L-robust の CCC は MSP-Podcast 内に比べて IEMOCAP で覚醒度 11%，支配度 21%，感情価 30%，MOSI の感情の極性で 15% 下がった．CNN14 では覚醒度 34%，支配度 52% 下がった — [Wagner et al., 2023](https://arxiv.org/abs/2203.07378)
- Chen & Yu の pre-finetuning は 4 コーパスの全部分集合を試すので，ソースの数を 0〜4 に変える評価になっているが，ターゲットは ESD の話者ごとの 2 値分類である — [Chen & Yu, 2023](https://arxiv.org/abs/2302.12921)

### Inferences
- 4 クラスの対応（neutral，happy，sad，angry．IEMOCAP では excited を happy に入れる）は，この研究の 4 クラスと同じで，広く使われている対応である．
- UAR は IEMOCAP の研究では UA として標準的に報告されるが，コーパス間のベンチマーク（EmoBox）は正解率，チャレンジは macro-F1 を使う．この研究の UAR を EmoBox の値と比べるときは，評価集合が均衡しているので同じ値として扱える．macro-F1 の値とは直接比べられない．
- この研究の評価（ソース 2〜3 個，ターゲット 1 個，話者を分ける）は，調べた範囲の基盤モデルの論文にはない組み合わせである．一番近いのは EmoBox の 1 対 1 の組と Chen & Yu の部分集合の総当たりである．
- AFEW（映画の切り抜き）と MOSEI（YouTube）をソースやターゲットに使った基盤モデルの SER は，調べた範囲では見つからなかった．音声のコーパス間 SER の論文は演技のコーパス（RAVDESS，SAVEE，CREMA-D，EmoDB，ESD）と IEMOCAP，MSP-Podcast が中心である．

### Gaps
- 基盤モデルを使い，複数のソースを合併してコーパスを 1 つ抜く評価をした論文を，専用の検索語では探していない．
- MSP-IMPROV と MSP-Podcast の間のコーパス間評価（MSP の研究室の論文に多い）は，本文を読めた論文がなかった．

## 問い 5：どの層に感情の情報があるか，話者の情報が表現を占めるか

### Takeaway
感情の情報は中間の層に多いという報告が多い．wav2vec 2.0 base では中間の層（IEMOCAP で 6 層目，RAVDESS で 5 層目）が最も良く，最終層が最も悪い場合がある．ただし EMO-SUPERB は，モデルごとに重みの山が違い，はっきりした型はないと報告している．話者の情報は WavLM / HuBERT の Base では下の層，Large では中間の層に多く，凍結した HuBERT Large から線形に近い頭部で話者を 60.7% 当てられる．話者の情報を消す（正規化する）と感情の精度が上がる報告が複数ある．

### Cited Findings
- Li ほか（SLT 2022）は wav2vec 2.0 base（事前学習のみ，100 時間の ASR で微調整，960 時間の ASR で微調整の 3 種）の各層の出力で IEMOCAP と RAVDESS の UA を測った．中間の層（IEMOCAP で 6 層目，RAVDESS で 5 層目）までは 3 つのモデルで UA が近く，その後は下がる．中間の 1 層だけで全層平均と同じくらいの UA が出て，最終層が最も悪い場合がある．著者は「wav2vec 2.0 は単語の認識に役立たない言語外の情報（paralinguistic information）を捨てているように見える」と書いている — [Li et al., SLT 2022](https://arxiv.org/abs/2210.02595)
- Pepino ほか（2021．基礎文献）の重み付き平均の学習後の重みは中間の層が大きかった．ASR で微調整したモデルでは出力に近い層の重みが小さかった — [Pepino et al., 2021](https://arxiv.org/abs/2104.03502)
- EMO-SUPERB の層の重みの分析では，W2V2 R（robust）は 17 層目，Data2Vec-A は 3 層目の重みが最も高く，ほかの 3 モデルは全層を同じように使っていた．著者は「モデルによって振る舞いが違い，はっきりした型はない」と結論した．XLS-R-1B は浅い層を重んじ，MSP-Podcast で学習したときだけ 30 層目に山があった — [EMO-SUPERB](https://arxiv.org/abs/2402.13018)
- WavLM の論文の SUPERB での重みの分析では，Base（WavLM と HuBERT）では下の層が話者識別・話者照合・話者ダイアライゼーションに，上の層が音声認識・音素認識・意図分類・スロット埋めに大きく寄与した．Large では上の層が内容と意味のタスクに，中間の層が話者のタスクに寄与した — [WavLM, Chen et al., 2022](https://arxiv.org/abs/2110.13900)
- Gat ほかは，凍結した HuBERT Large の上で学習した話者識別器が 60.7% の正解率を出し，話者の正規化の後に 45.9% に下がったと報告した — [Gat et al., 2022](https://arxiv.org/abs/2202.01252)
- Wagner ほかは，Transformer のモデルが性別の群に対しては公平だが，個々の話者に対しては公平でない（話者によって精度が大きく違う）と報告した．また感情価の高い精度は，Transformer 層の微調整で暗黙に言語の情報を学ぶことによると説明した — [Wagner et al., 2023](https://arxiv.org/abs/2203.07378)
- Pepino ほかでは話者ごとの正規化が全体の正規化より UAR を IEMOCAP で 1.4，RAVDESS で 8.6 上げた（67.2 対 65.8，84.3 対 75.7）．著者は RAVDESS の差が大きい理由を，RAVDESS では感情によって言う文が変わらないため，話者の影響を除くと感情による違いだけが残るからだと説明している — [Pepino et al., 2021](https://arxiv.org/abs/2104.03502)
- EmoBox は最終層だけを使い，wav2vec 2.0 large の最終層ではほとんどのコーパスで SER が学習できなかったと報告している — [EmoBox](https://arxiv.org/abs/2406.07162)

### Inferences
- 凍結した特徴を使う場合，最終層だけより中間の層か全層の重み付き和のほうが安全である，という報告が多い．EmoBox の最終層だけの評価は，凍結特徴の上限を下回っている可能性がある（EmoBox 自身が限界として認めている）．
- WavLM Large では話者の情報が中間の層に多く，感情に役立つ層も中間に多いという 2 つの報告を合わせると，中間の層の特徴は感情と話者の情報が重なっている．コーパス間のずれのうち話者やコーパスに由来する成分が，感情に使う特徴と同じ層に乗っている可能性がある．これは 2 つの別々の論文の結果を組み合わせた推論であり，同じ実験で確かめた報告ではない．

### Gaps
- 「話者の情報が自己教師ありの表現の分散の大部分を占める」ことを分散の割合などの数値で示した論文は，このセッションでは本文で確認できなかった．
- どの層の特徴がコーパス間で最もよく汎化するか（同じコーパス内で最良の層と同じか）を直接比べた論文は見つからなかった．
- WavLM Large の感情認識での層ごとの重みの数値（WavLM の論文の図 2）は図の中にあり，取り出せなかった．

## 問い 6：何が「標準の手法」か，それぞれの選択はどのくらい多いか

### Takeaway
このセッションで本文を読んだ自己教師ありの音声モデルを使う SER の論文 19 本（2021〜2026．基礎文献 2 本を含む．網羅的な標本ではない）では，凍結が 9 本（そのうち 2 本は PEFT を足す），微調整が 8 本，ソースで微調整してから凍結が 1 本，不明が 1 本だった．ソースからターゲットへ移す評価を含むのは 8 本で，ターゲットのラベルなしデータを使うのは 3 本，ターゲットの少数ラベルを使うのは 3 本だった．音声と顔と言葉を組み合わせてコーパス間の転移を測った論文はなかった．

### Cited Findings
次の表は各論文の本文から読んだ設定である（出典は各行の論文）．「移す」はソースのコーパスで学習しターゲットの別のコーパスで評価することを指す．

| 論文 | モデル | 凍結／微調整 | 層 | 頭部 | 移す評価 | ターゲットのラベルなし | ターゲットのラベル | 指標 | この研究との主な違い |
|---|---|---|---|---|---|---|---|---|---|
| [SUPERB 2021](https://arxiv.org/abs/2105.01051)（基礎） | 多数（wav2vec 2.0，HuBERT ほか） | 凍結 | 重み付き和 | 平均プーリング＋線形 | なし | なし | なし | 正解率 | IEMOCAP 内のみ，音声のみ |
| [Pepino 2021](https://arxiv.org/abs/2104.03502)（基礎） | wav2vec 2.0 base | 凍結 | 重み付き平均（CNN 出力を含む） | 全結合 | なし | なし | なし | UAR | 話者ごとの正規化を使う |
| [Gat 2022](https://arxiv.org/abs/2202.01252) | HuBERT Base/Large | 微調整または射影層 | 記載を確認せず | 不明 | なし | なし | なし | WA | 話者の敵対的な正規化，IEMOCAP 内のみ |
| [Li 2022](https://arxiv.org/abs/2210.02595) | wav2vec 2.0 base | 凍結（層ごとの調査） | 1 層ずつ，平均 | 下流モデル | なし | なし | なし | UA | 層の分析のみ |
| [Wagner 2023](https://arxiv.org/abs/2203.07378) | wav2vec 2.0，HuBERT | 微調整（CNN は凍結） | 最終の Transformer 層を時間方向に平均 | 隠れ層 1 つ＋出力層 | MSP-Podcast→IEMOCAP，MOSI | なし | なし | CCC | 次元の回帰，ゼロショット |
| [Chen & Rudnicky 2023](https://arxiv.org/abs/2110.06309) | wav2vec 2.0 base | 微調整＋TAPT | 最終層を時間方向に平均 | ReLU＋線形 1 層 | なし | 同じコーパスの音声（TAPT） | 全部 | UA | コーパス内のみ |
| [PEFT-SER 2023](https://arxiv.org/abs/2306.05350) | Whisper，wav2vec 2.0，WavLM Base+ | 凍結＋PEFT | 未確認 | 未確認 | なし | なし | 全部 | 未確認 | コーパス内のみ |
| [Chen & Yu 2023](https://arxiv.org/abs/2302.12921) | wav2vec 2.0 base | 微調整 | 未確認 | 線形（タスクごと） | 4 コーパスの部分集合→ESD | なし | k = 2〜64（2 値） | macro-F1 | 2 値，話者ごとの適応 |
| [Agarla 2023](https://arxiv.org/abs/2207.06767) | HuBERT base | 未確認 | 未確認 | 線形 | 英語→仏・独・伊・ペルシア語 | あり（疑似ラベル） | 合計 0〜100 件 | UA | 5 クラス，演技のみ，音声のみ |
| [Lashkarashvili 2024](https://arxiv.org/abs/2402.11747) | wav2vec 2.0 Base，HuBERT Large | 凍結／PEFT／微調整を比較 | 重み付き和を含む | 下流層 | ESD→IEMOCAP 即興 | なし | 学習分割の全部 | 正解率，CCC | ターゲットのラベルを全部使う |
| [EMO-SUPERB 2024](https://arxiv.org/abs/2402.13018) | 15 モデル | 凍結 | 重み付き和 | 軽い頭部 | なし | なし | 全部 | macro-F1 | コーパス内のみ |
| [EmoBox 2024](https://arxiv.org/abs/2406.07162) | 10 モデル | 凍結（特徴抽出） | 最終層 | 線形→プーリング→ReLU→線形 | 4 コーパスの 12 組 | なし | なし | 正解率（均衡） | ゼロショット，ソース 1 個，音声のみ |
| [emotion2vec 2024](https://arxiv.org/abs/2312.15185) | data2vec 系を感情の音声で再学習 | 凍結 | 未記載（SER） | 2 層の線形 | なし（各言語の内部） | 事前学習に IEMOCAP・MELD・MOSEI | 全部 | WA，UA | 事前学習とこの研究のコーパスが重なる |
| [Ibrahim 2024](https://arxiv.org/abs/2406.09933) | Whisper base | 微調整 | encoder と decoder | 5 層の全結合 | なし（合併して学習） | なし | 全部 | 正解率 | 11 コーパスの合併 |
| [Adapting WavLM 2024](https://arxiv.org/abs/2405.04485) | WavLM Large | 微調整 | 全層の重み付き平均 | 射影層＋attention pooling＋線形 | なし | なし | 全部 | macro-F1 | MSP-Podcast 内 |
| [Naini 2025](http://www.lab-msp.com/MSP/publications/Naini_2025.pdf) | WavLM-large | ソースで微調整後に凍結 | 未確認 | DNN | MSP-Podcast→VAM，RECOLA，WHiSER | あり（疑似ラベル，Case 3） | 一部（件数未確認） | CCC | 次元の回帰，ラベル分布の補正 |
| [Interspeech 2025 チャレンジ](https://www.isca-archive.org/interspeech_2025/naini25_interspeech.pdf) | ベースラインは WavLM-large，上位は WavLM・Whisper | 微調整が多数 | 未確認 | attentive statistics pooling＋全結合 | なし | なし | 全部 | macro-F1，CCC | MSP-Podcast 内，8 クラス |
| [Tavernor 2025](https://arxiv.org/abs/2509.12295) | WavLM＋BERT | 凍結 | 平均プーリング（層は未確認） | 未確認 | あり（評定者のモデル） | なし | 少数の登録データ | CCC | 評定者ごとの予測 |
| [Dong 2026](https://arxiv.org/abs/2601.16240) | wav2vec 2.0 base | ソースで微調整 | 未記載 | 256 ユニットの全結合 | IEMOCAP↔RAVDESS ほか | あり（テスト時） | なし | 正解率，macro-F1 | TTA，音声のみ |

- Interspeech 2025 のチャレンジの上位チームでは音声と言葉の組み合わせが 76%，WavLM が 70%，Whisper が 47% だった — [Interspeech 2025 Challenge](https://www.isca-archive.org/interspeech_2025/naini25_interspeech.pdf)
- EmoBox のコーパス間評価では Whisper large v3 が 12 組中 9 組で最も高かった — [EmoBox](https://arxiv.org/abs/2406.07162)

### Inferences
- 上の 19 本を数えると次のようになる（この数は論文選びに依存し，分野全体の割合ではない）．
  - 凍結が主（SUPERB，Pepino，Li，PEFT-SER＋PEFT，Lashkarashvili（比較），EMO-SUPERB，EmoBox，emotion2vec，Tavernor）：9 本．微調整が主（Gat，Wagner，Chen & Rudnicky，Chen & Yu，Ibrahim，Adapting WavLM，チャレンジ，Dong）：8 本．ソースで微調整してから凍結（Naini 2025）：1 本．不明（Agarla）：1 本．
  - 凍結のとき層をどうまとめたかが分かる論文は 6 本で，重み付き和か平均を使ったのが 5 本（SUPERB，Pepino，EMO-SUPERB，Lashkarashvili の WS，Li の全層平均），最終層だけを使ったのが 1 本（EmoBox）だった．Li は 1 層ずつの特徴も調べている．微調整する論文では，最終層を時間方向に平均する構成（Wagner，Chen & Rudnicky）と全層の重み付き平均（Adapting WavLM）の両方がある．
  - 使われたモデル（評価の主役として）：wav2vec 2.0 が 11 本，WavLM が 7 本，HuBERT が 7 本，Whisper が 4 本，data2vec 系が 3 本，emotion2vec が 1 本（EmoBox では評価データの選別にも使われた）．2024 年以降の論文に限ると WavLM と Whisper の比重が高い．
  - 移す評価があるのは 8 本（Wagner，Chen & Yu，Agarla，Lashkarashvili，EmoBox，Naini 2025，Tavernor，Dong）．このうちターゲットのラベルなしデータを使うのは 3 本（Agarla，Naini 2025 の Case 3，Dong），ターゲットの少数ラベルを使うのは 3 本（Chen & Yu，Agarla，Tavernor の登録データ）．Naini 2025 の Case 2 もラベル付きのターゲットの一部を使うが件数を確認できなかった．
  - 指標（1 本で複数を使う場合は重ねて数えた）：UA（＝UAR）は Pepino，Li，Chen & Rudnicky，Agarla，emotion2vec，EmoBox（コーパス内）の 6 本．正解率または WA は SUPERB，Gat，Lashkarashvili，EmoBox（コーパス間），Ibrahim，Dong，emotion2vec の 7 本．macro-F1 は EMO-SUPERB，Chen & Yu，Adapting WavLM，チャレンジ，Dong，EmoBox（コーパス内）の 6 本．CCC は Wagner，Lashkarashvili，Naini 2025，チャレンジ，Tavernor の 5 本．
- 「標準の手法」として 2 つの型が言える．
  - ベンチマークの型：凍結した WavLM / HuBERT / wav2vec 2.0（最近は Whisper の encoder も）の全層の重み付き和，平均プーリング，線形か 1 隠れ層の頭部，IEMOCAP の 4 クラスでセッションを 1 つ抜く 5 分割，UA / WA．
  - 精度を競う型：WavLM-large か Whisper large-v3 を微調整し，attentive statistics pooling と全結合層を置き，重み付き交差エントロピーで学習し，言葉のモデル（RoBERTa など）と組み合わせてアンサンブルする．
- この研究（凍結 WavLM＋ResNet-50＋RoBERTa，線形 softmax，ソース 2〜3 個，ターゲットはクラスあたり K 件のラベルとラベルなしデータ，4 クラス，UAR）は，ベンチマークの型の「凍結＋軽い頭部」と 4 クラス・UAR を受け継いでいる．一方で，(1) 3 つのモダリティを使う，(2) ソースが複数，(3) ターゲットのラベルをクラスあたり K 件に絞る，(4) AFEW と MOSEI を含む，の 4 点は，調べた範囲の音声の論文のどれにも同時には当てはまらない．
- 査読者がこの研究に求めそうな比較相手は，調べた結果からは，(a) 同じ凍結特徴でのソースのみ，(b) ターゲットの K 件だけの学習，(c) 話者ごと・コーパスごとの特徴の正規化，(d) 疑似ラベル（Agarla 型），(e) ラベル分布の補正（Naini 2025 型），(f) 頭部だけを調整する TTA（T3A），である．ここでの列挙は，各論文が比較に使った手法から導いたもので，提案ではない．

### Gaps
- この 19 本は網羅的な系統レビューではない．ICASSP・Interspeech の 2024〜2025 年のコーパス間 SER の論文を全部数えたわけではなく，割合は参考値である．
- MSP の研究室（Busso ら）のコーパス間の論文の多く（Naini 2024，Mote 2024，Odyssey 2024 の論文）は，サーバーに接続できず本文を読めなかった．
- 音声の基盤モデルと顔・言葉の特徴を組み合わせてコーパス間の転移を測った研究は，この調査の範囲（音声の SER）の外にあり，探していない．
