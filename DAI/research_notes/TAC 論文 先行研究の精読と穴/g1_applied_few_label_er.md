# 少数のターゲットラベルを使うコーパス間の感情認識 9 本の精読（本研究の軸 A1〜A11 との対応と穴）

作成：2026-10-27．前回の調べ（`research_notes/TAC 論文の先行研究と書き方/applied_few_label_cross_corpus.md`）の URL と抄録の要点を出発点にし，本文を読み切った．

**確かめた水準**：◎ 本文（arXiv の PDF を `pdftotext` で文字にして全節を読んだ），○ 抄録，△ 検索結果だけ．
**書誌の出どころ**：Crossref API，Semantic Scholar API，OpenAlex API（DOI つき）．掲載誌の版（IEEE Xplore）は 1 本も読めていない．読んだのはすべて arXiv 版であり，掲載版との差は未確認である（各論文の「注意」に書く）．
**引用**：原文は英語のまま 1〜2 文．表の値は本文の表を写した．図しか無い数値は「図のみ」と書く．

本研究の軸（表の列）：A1 学習器（凍結エンコーダ＋線形の頭），A2 誤差の切片と向きの分解・K の式，A3 ソースの価値をラベル本数に換算，A4 合併の切片・暗黙のドメイン別切片，A5 K ≤ 10 程度・K を振る・K の単位，A6 コーパス間・ソース数・コーパス数，A7 マルチモーダル・融合の先生，A8 ソースだけの分類器の出力を弱いラベルに・クラス均衡，A9 出発点への L2 罰則，A10 評価（話者分離・UAR・相手の調整・検定・補正・効果量・実用値とオラクル値），A11 ずれの測定と利得の結びつけ・条件の予言．

---

## 1. M4SER（He, Shi, Hu, Mi, Li, Toda，2025）

### Takeaway
音声＋ASR の言葉の 2 モダリティで，IEMOCAP↔MELD の 1 組だけ，クラスあたり K=5・10・20・60 を振った少数ショットの節（V-G）がある．エンコーダ（HuBERT・BERT）は微調整し，モデル選択にはターゲットの検証用 30% のラベルを使う．ずれの測定も，ラベルが何を直すかの分析も無い．

### Cited Findings

**1. 書誌と読んだ範囲**
- 書誌（確かめた）：J. He, X. Shi, C.-H. Hu, J. Mi, X. Li, T. Toda, "M4SER: Multimodal, Multirepresentation, Multitask, and Multistrategy Learning for Speech Emotion Recognition," IEEE Transactions on Audio, Speech and Language Processing, vol. 33, pp. 4055–4070, 2025, DOI 10.1109/TASLPRO.2025.3614428 — [Crossref API](https://api.crossref.org/works?query.bibliographic=M4SER+Multimodal+Multirepresentation+Multitask+Multistrategy+Learning+for+Speech+Emotion+Recognition&rows=3)．Crossref の誌名は「IEEE Transactions on Audio, Speech and Language Processing」であり，「IEEE/ACM」は付かない（2025 年から誌名が変わった可能性がある．引用時は要確認）．著者名は arXiv 版の表紙と一致する — [arXiv 2509.18706v1](https://arxiv.org/abs/2509.18706)
- 読んだ範囲（◎）：arXiv v1（2025-09-23）の全節 I〜VI．特に IV-A（データと指標），IV-B（実装），V-G（Cross-corpus Generalization Ability．表 VIII・IX）．掲載版は読めていない

**2. 設定**（[arXiv 2509.18706v1](https://arxiv.org/pdf/2509.18706) IV-A・IV-B・V-G）
- 課題：4 感情（Neutral・Happy（Excited を含む）・Angry・Sad）の発話単位の分類．コーパス間では MELD の Surprise・Fear・Disgust を捨てて 4 感情に揃える
- データ：IEMOCAP 5,531 発話（Neutral 1,708・Angry 1,103・Happy 1,636・Sad 1,084），MELD 13,708 発話（train 9,989／val 1,109／test 2,610．7 感情：Neutral 6,167・Happy 2,198・Fear 350・Sad 985・Disgust 355・Angry 1,541・Surprise 1,499）
- モダリティ：音声（HuBERT-base-ls960）＋ Whisper の ASR 文（BERT-base-uncased）．WER は IEMOCAP 20.48%，MELD 37.87%
- ターゲットのラベル数：コーパス間（表 VIII）は 0．少数ショット（表 IX）は**クラスあたり** 5・10・20・60 本（原文："sampling 5, 10, 20, and 60 labeled instances per class from the target-domain validation set"）．ただし "The remaining validation data is still used for model selection"．つまり K 本のほかに，ターゲットの検証用 30% の残りのラベルをモデル選択に使う
- 学習器：深層．"Both the HuBERT and BERT models were fine-tuned during the training stage."（IV-B．コーパス内の設定の記述）．コーパス間・少数ショットの節に，エンコーダを凍結したか微調整したかの明記は無い
- 評価：コーパス内は IEMOCAP が 5 分割 LOSO（WA・UA），MELD は公式分割（ACC・W-F1）．コーパス間は「一方のコーパス全体で学習，ターゲットの 30% を検証，70% をテスト」で，話者分離の記述は無い．反復回数・標準偏差・検定の記述は無い．比較相手（SAMS，MF-AED-AEC）は再実装

**3. 示したこと**（[arXiv 2509.18706v1](https://arxiv.org/pdf/2509.18706) 表 VIII・IX）
- 表 VIII（ターゲットのラベル 0）．IE→ME は ACC／W-F1，ME→IE は WA／UA（%）：SAMS 17.0／11.6，36.3／33.9；MF-AED-AEC 56.0／54.2，62.5／60.6；Baseline（HuBERT＋BERT）53.0／52.4，58.9／57.4；＋MSR&MIR 54.7／53.2，60.8／59.9；＋GAN&LCL 55.6／53.9，61.9／60.7；＋AED&AEC（M4SER）57.3／55.5，64.0／62.2
- 表 IX（少数ショット．K=5／10／20／60）：
  - SAMS：IE→ME ACC 51.6／52.5／52.3／59.9，W-F1 49.0／49.2／52.3／56.8；ME→IE WA 39.0／38.2／39.9／44.4，UA 37.2／38.3／35.0／45.3
  - MF-AED-AEC：ACC 56.3／57.8／57.5／60.8，W-F1 54.8／55.7／56.7／60.3；WA 64.4／65.8／67.6／72.5，UA 63.3／65.6／66.6／73.7
  - M4SER：ACC 57.9／58.7／59.4／62.1，W-F1 56.0／58.0／58.8／61.5；WA 66.4／66.7／69.0／74.0，UA 64.4／67.7／68.7／75.9
- 読み取り：ME→IE の UA は K=0 の 62.2 から K=5 で 64.4（＋2.2），K=60 で 75.9（＋13.7）．IE→ME の ACC は 57.3 → 57.9（K=5）→ 62.1（K=60）．著者の記述："with only 5 labeled samples per class, M4SER achieves ACC of 57.9% in the IE→ME setting, already surpassing MF-AED-AEC with 20 shots."

**4. 軸との対応**
- A1 ×：HuBERT と BERT を微調整する深層モデル．A2 ×，A3 △（「K=5 で相手の K=20 を上回る」というショット換算の比較だけ），A4 ×
- A5 ○：クラスあたり 5〜60 で 4 点振る．ただし K ≤ 10 は 5 と 10 の 2 点
- A6 △：コーパス間だがソースは 1 つ，コーパスは 2 つ（1 組 2 方向）
- A7 ○：音声＋言葉の融合．融合の先生から単モーダルの生徒へ使う設計は無い
- A8 ×，A9 ×（ソースで学んだモデルから微調整するだけで罰則は無い）
- A10 △：UA は ME→IE だけ（IE→ME は ACC・W-F1）．話者分離の記述なし．反復・検定・効果量なし．モデル選択にターゲットの検証用ラベル（30% の残り）を使うので，K 本だけの実用値ではない
- A11 ×：t-SNE（図 9・10）はコーパス内の表現の可視化で，ずれの量は測らない

**5. 穴**
- していないこと：K ≤ 4，複数ソース，顔，ずれの測定，ラベルが直すものの分析，統計検定，話者分離の明記
- 限界として自分で書いていること：限界の節は無い．V-F で ASR 文と正解文の差を論じるだけ（"GT text lacks ASR errors, missing additional cues beneficial for emotion recognition."）
- 今後の課題（原文）："In the future, we plan to extend our approach to the visual modality and introduce disentangled representation learning to further enhance emotion recognition performance. We also plan to investigate whether AED and AEC modules remain necessary when using powerful LLM-based ASR systems."

### Inferences（6. 本研究との差の書き方）
- 案：「M4SER [He et al., 2025] は音声と ASR の言葉を融合した深層モデルを IEMOCAP と MELD の間で評価し，クラスあたり 5〜60 本のターゲットラベルで微調整する少数ショットの実験を 1 組のコーパス対で報告した．本研究は凍結したエンコーダの上の線形の頭に限ることで，K 本のラベルが切片と向きのどちらを直すかを式で分け，7 群の多数の組で K を振る．」
- M4SER の K=5→60 の伸び（UA 64.4→75.9）は，微調整したエンコーダかつ検証用ラベルでモデル選択をした値であり，凍結特徴の地図と並べるときはこの 2 点を明記する

### Gaps（7. 注意）
- 掲載版（TASLP 33:4055–4070）と arXiv v1 の差は未確認．誌名に「IEEE/ACM」を付けるかは Crossref では付かない（要確認）
- SAMS の K=0 の ACC 17.0（4 クラスのでたらめ 25% 未満）は再実装の失敗の疑いがあり，比較相手として弱い
- コーパス間で話者を分けたか，エンコーダを凍結したかは本文に無い（要確認．著者への確認が要る）

---

## 2. MDAT（Zaidi, Latif, Qadir，2024）

### Takeaway
音声（XLS-R）＋言葉（RoBERTa）の事前学習特徴の上に，グラフ注意・共注意・Transformer の頭を載せ，4 言語の 4 コーパスの 12 組で K=0・5・10・15 本のターゲットラベルを使う．K の単位（クラスあたりか合計か）は書かれておらず，表 IV と表 V の 0 本の値が 6 か所で食い違う．

### Cited Findings

**1. 書誌と読んだ範囲**
- 書誌（確かめた）：掲載版は S. A. M. Zaidi, S. Latif, J. Qadir, "Enhancing Cross-Language Multimodal Emotion Recognition With Dual Attention Transformers," IEEE Open Journal of the Computer Society, vol. 5, pp. 684–693, 2024, DOI 10.1109/OJCS.2024.3486904 — [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/ojcs.2024.3486904?fields=title,authors,venue,journal,year)．**掲載版の題名は arXiv 版（"Cross-Language Speech Emotion Recognition Using Multimodal Dual Attention Transformers"）と違う**
- 読んだ範囲（◎）：arXiv v2（2023-06-27）の全節 I〜VI — [arXiv 2306.13804v2](https://arxiv.org/abs/2306.13804v2)．arXiv の注記は IEEE TAC 投稿中だが掲載先は OJCS．掲載版は読めておらず，表の値や K の定義が直されている可能性がある

**2. 設定**（[arXiv 2306.13804v2](https://arxiv.org/pdf/2306.13804v2) III・IV・V-C）
- 課題：4 感情（angry・happy・neutral・sad）の分類．言語間（コーパス＝言語）
- データ：IEMOCAP（英．「800 発話の部分集合．ただし実験ごとに数が変わりうる」と明記），EMODB（独．420 発話：angry 127・sad 143・neutral 79・happy 71），EMOVO（伊．336 発話，各 84），URDU（ウルドゥー．400 発話，各 100．文は EmulationAI の API で書き起こし）
- モダリティ：音声＋言葉．XLS-R と RoBERTa を特徴抽出に使う（"utilises pre-trained models for multimodal feature extraction"．微調整の記述は無く凍結と読める）
- ターゲットのラベル数：0・5・10・15．原文："The shots are the number of labelled samples from the target language that are used to fine-tune the models."．**クラスあたりか合計かの記述は無い**
- 学習器：深層の頭（グラフ注意→共注意→Transformer エンコーダ→全結合）．基準は BiLSTM の単純結合．比較相手は SAFRLM と HCAM
- 評価：UA．話者分離・反復・標準偏差・検定の記述は無い．K ショットの微調整の手順（どの層を更新するか，学習率）も書かれていない

**3. 示したこと**（[arXiv 2306.13804v2](https://arxiv.org/pdf/2306.13804v2) 表 IV・V・VI）
- 表 IV（ターゲットのラベル 0．UA%．Baseline／SAFRLM／HCAM／MDAT）：IEMOCAP→EMODB 40.78／39.61／38.63／42.48；→EMOVO 73.46／69.60／73.52／85.51；→URDU 63.18／50.07／63.17／64.43．EMODB→IEMOCAP 49.23／50.77／55.03／55.55；→EMOVO 50.54／52.25／51.13／56.25；→URDU 72.46／70.00／72.98／73.23．URDU→IEMOCAP 47.96／44.75／50.55／58.32；→EMODB 65.35／63.40／72.71／75.31；→EMOVO 63.55／65.36／63.55／67.66．EMOVO→IEMOCAP 57.25／55.17／49.09／59.96；→EMODB 64.60／64.28／75.22／81.60；→URDU 72.46／66.82／58.92／73.23
- 表 V（K ショット．UA%．Baseline／MDAT を K=0，5，10，15 の順に）：
  - IEMOCAP→EMODB 40.78／48.48，73.18／73.70，78.77／83.80，78.98／91.48
  - IEMOCAP→URDU 61.18／64.43，64.44／65.22，70.68／73.86，73.19／77.54
  - IEMOCAP→EMOVO 73.46／85.51，77.86／85.10，78.62／87.77，83.93／92.05
  - EMODB→IEMOCAP 49.23／55.55，48.80／51.35，51.39／59.61，52.11／63.48
  - EMODB→URDU 72.46／74.23，72.41／76.64，75.30／83.12，78.40／84.57
  - EMODB→EMOVO 50.54／56.25，69.99／64.18，74.01／89.36，78.40／93.56
  - URDU→IEMOCAP 47.96／58.32，45.16／42.79，48.61／58.91，46.67／59.52
  - URDU→EMODB 65.35／75.32，73.18／75.42，75.42／76.49，76.51／80.67
  - URDU→EMOVO 65.35／76.21，65.42／77.36，69.95／83.24，70.83／88.64
  - EMOVO→IEMOCAP 57.25／59.96，60.14／64.71，62.38／64.81，68.74／69.09
  - EMOVO→EMODB 65.60／81.60，79.43／82.81，80.73／85.41，84.62／94.23
  - EMOVO→URDU 72.46／73.23，70.12／73.50，78.76／80.45，81.16／82.05
- 表 IV と表 V の K=0 の食い違い（6 か所）：IEMOCAP→EMODB の MDAT 42.48 対 48.48；IEMOCAP→URDU の Baseline 63.18 対 61.18；EMODB→URDU の MDAT 73.23 対 74.23；URDU→EMODB の MDAT 75.31 対 75.32；URDU→EMOVO の Baseline 63.55 対 65.35 と MDAT 67.66 対 76.21；EMOVO→EMODB の Baseline 64.60 対 65.60．本文は "increases its accuracy from 42.48 % with 0 shots to 91.48 % with 15 shots"（表 IV の値）と書き，IEMOCAP→EMOVO については "from 82.51 & with 0 shots to 92.05 %" と書くが両表とも 85.51 である
- 単調でない値：URDU→IEMOCAP の MDAT は 58.32（0）→ 42.79（5）→ 58.91（10）；EMODB→EMOVO の Baseline は 50.54（0）→ 69.99（5）で 5 本で＋19.5；EMODB→IEMOCAP と URDU→IEMOCAP の Baseline は K=5 で K=0 を下回る
- 表 VI（消去実験．K=0，IEMOCAP→EMODB／EMOVO／URDU）：全部 42.48／85.51／64.43；Transformer 無し 39.21／80.68／56.52；グラフ注意だけ 41.58／76.13／45.41；共注意だけ 39.06／41.82／53.14；Transformer だけ 34.80／46.59／36.23

**4. 軸との対応**
- A1 △：特徴は凍結した事前学習モデルだが，頭は深層（グラフ注意＋共注意＋Transformer）
- A2 ×，A3 ×，A4 ×
- A5 △：K=5・10・15 を振るが単位が不明．0〜15 の 4 点
- A6 ○：4 コーパス（4 言語）の 12 組．ソースは 1 つずつ
- A7 ○：音声＋言葉の融合．先生→生徒は無い
- A8 ×，A9 ×
- A10 ×：UA は報告するが，話者分離・反復・標準偏差・検定・相手の調整の記述が無い．表の間で値が食い違う
- A11 ×：ずれは測らない．言語の「情報の豊かさ」で傾向を言葉で説明するだけ（"English has more information and richness than the other languages"）

**5. 穴**
- していないこと：K の単位の定義，反復，検定，話者分離，ずれの測定，ラベルが直すものの分析
- 限界として自分で書いていること：限界の節は無い．結論の (5) "k-shot adaptation techniques demonstrate their usefulness in addressing the scarcity of labeled data in low-resource languages, as they successfully enhance model performance even with a limited number of labeled samples."
- 今後の課題（原文）："These include the exploration of the visual modality to further enhance the performance of MDAT. Furthermore, it is important to evaluate the effectiveness of multimodal frameworks in more realistic and diverse scenarios, such as noisy environments, in the face of adversarial attacks, and with spontaneous speech and mixed emotions."

### Inferences（6. 本研究との差の書き方）
- 案：「MDAT [Zaidi et al., 2024] は XLS-R と RoBERTa の特徴の上に注意機構の頭を載せ，4 言語の 12 組で 0〜15 本のターゲットラベルによる適応を報告したが，ラベルの単位と反復は示されていない．本研究は線形の頭に限り，クラスあたりの本数を定義して K を振り，反復と対応のある検定で利得の有無を判定する．」
- 「少数ラベル × 音声＋言葉 × 言語間」の先例として引けるが，数値の食い違いと K の単位の欠落があるので，値そのものは引かない

### Gaps（7. 注意）
- 掲載版（OJCS 5:684–693）は題名が変わっており，表の値や K の定義が直っている可能性がある（未読．要確認）
- K の単位は著者かコードで確かめるしかない
- IEMOCAP が「800 発話の部分集合」であることは，一般の IEMOCAP 4 感情（5,531 発話）と比べられない理由になる

---

## 3. CTIAL（Xu, Jiang, Wu，2024）

### Takeaway
学習器は本研究と同じ型（凍結 wav2vec2 の平均特徴＋PCA＋ロジスティック回帰）だが，問いは「どの標本にラベルを付けるか」（能動学習）で，ソースとターゲットで課題（分類と次元推定）が違う．コーパス間ではラベルが増えると CTIAL とランダムの差が消えると著者自身が書く．

### Cited Findings

**1. 書誌と読んだ範囲**
- 書誌（確かめた）：Y. Xu, X. Jiang, D. Wu, "Cross-Task Inconsistency Based Active Learning (CTIAL) for Emotion Recognition," IEEE Transactions on Affective Computing, vol. 15, no. 3, pp. 1659–1668, 2024, DOI 10.1109/TAFFC.2024.3366767 — [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/taffc.2024.3366767?fields=title,authors,venue,journal,year)（号は前回の調べの OpenAlex による）
- 読んだ範囲（◎）：arXiv v1（2024-12-02）の全節 1〜6 — [arXiv 2412.01171v1](https://arxiv.org/abs/2412.01171)

**2. 設定**（[arXiv 2412.01171v1](https://arxiv.org/pdf/2412.01171) 2.6・3・4・5）
- 課題：カテゴリ分類（CEC）と次元推定（DEE．valence・arousal・dominance）の間の課題間の能動学習．NRC の感情語彙（Valence-Arousal-Dominance Lexicon）でカテゴリの予測確率を次元空間に写し，2 つの課題の予測の不一致（CTI）で標本を選ぶ
- データ：IEMOCAP（即興の節．5 クラス angry 289・happy 947（excited 込み）・sad 608・frustrated 971・neutral 1,099．MELD からの転移では 4 クラス），MELD の訓練集合（angry 1,109・happy 1,743・sad 683・neutral 4,709），VAM（947 発話，47 話者，次元ラベル）
- モダリティ：音声のみ
- ターゲットのラベル数：初期 20 本（ランダム）＋ 1 本ずつ 200 回の問い合わせ．**合計**で数え，クラスあたりではない
- 学習器：wav2vec 2.0（LibriSpeech 960 時間で事前学習し ASR 用に微調整したもの）の 12 層の平均を時間平均した 768 次元．PCA（分散 90%）．分類はロジスティック回帰，推定はリッジ回帰．正則化の重みは {1, 5, 10, 50, 1e2, 5e2, 1e3, 5e3} から訓練データの 3 分割交差検証で選ぶ．コーパス間では TCA（30 次元）か BDA（40 次元）で特徴を合わせる
- 評価：ターゲット全体（人が付けた分と分類器が付けた分を合わせる）で BCA（クラスごとの正解率の平均＝UAR），RMSE，CC．コーパス内は 2 セッションをソース・3 セッションをターゲットにした 10 通りの分割 × 3 反復（ソースとターゲットの話者は別）．コーパス間は 10 反復．Wilcoxon の符号付き順位検定に Holm の補正を，能動学習の各反復で行う
- **コーパス間ではソースとターゲットの課題が違う**：VAM（DEE）→ IEMOCAP（CEC），MELD（CEC）→ IEMOCAP（DEE）．MELD→IEMOCAP の分類（CEC→CEC）は 5.3 節の BDA の確かめ（図 8）にだけ出る

**3. 示したこと**（[arXiv 2412.01171v1](https://arxiv.org/pdf/2412.01171) 4.2・4.3・5.2・5.3．主結果は図 3・6 のみで表は無い）
- DEE→CEC（4.2 節，図 3・4）：原文 "CTIAL outperformed Rand, the two within-task AL approaches (Ent and LC), and the cross-task AL baseline Source MTiGS when K was small ... However, in cross-corpus transfer, CTIAL and Rand had similar performance when K was large. The reason may be that domain shift limited the performance of the source DEE models on the target dataset, which further resulted in inaccurate CTI calculation and degraded AL performance." 不確かさだけの選び方については "both may not outperform Rand, suggesting that only considering the uncertainty may not be enough." LC-CTIAL が全体で最良
- CEC→DEE（5.2 節，図 6・7）："In within-corpus transfer, CTIAL outperformed both Rand and MTiGS initially, but gradually became inferior to Rand. In cross-corpus transfer, CTIAL performed better than Rand on arousal but worse on valence and dominance."
- BDA の確かめ（5.3 節，図 8）：IEMOCAP 内の 4 クラスの BCA 0.6477，MELD→IEMOCAP の直接転移 0.3904，BDA（45 次元）で 0.5378
- TCA の確かめ（4.3 節，図 5）：IEMOCAP 内の DEE の平均 RMSE 0.6667，CC 0.5659．VAM→IEMOCAP は TCA で RMSE が下がり CC が上がる

**4. 軸との対応**
- A1 ○：凍結した自己教師あり特徴＋PCA＋ロジスティック回帰（本研究と同じ型）
- A2 ×，A3 ×，A4 ×
- A5 △：初期 20 本は少数だが，合計で数え，最大 220 本まで 1 本ずつ増やす．クラスあたりの定義は無い
- A6 △：コーパス間（VAM→IEMOCAP，MELD→IEMOCAP）だが課題が違う．ソースは 1 つ，コーパスは 3 つ
- A7 ×：音声のみ
- A8 △：BDA でソースのモデルがターゲットに擬似ラベルを付け，クラスごとの平均を合わせる（特徴の整列のためで，ターゲットの分類器の学習ラベルではない．クラス均衡の扱いは無い）
- A9 ×
- A10 ○寄り：BCA（＝UAR），反復，Wilcoxon＋Holm の補正，正則化は訓練データの交差検証（実用値）．ただし評価集合にラベル付き標本を含める（能動学習の設定）．効果量は無い
- A11 △：MMD をバランス係数の選択にだけ使う．利得との結びつけや条件の予言は無い

**5. 穴**
- していないこと：ソースを混ぜて学ぶ比較（ソースは選び方の情報源），ずれと利得の対応，K をクラスあたりで振ること
- 限界として自分で書いていること：限界の節は無い．上の "domain shift limited the performance of the source DEE models on the target dataset" が唯一の自己評価
- 今後の課題：書かれていない（結論は "To our knowledge, this is the first work that utilizes prior knowledge on affective norms and data in a different task to facilitate AL for a new task" で終わる）

### Inferences（6. 本研究との差の書き方）
- 案：「CTIAL [Xu et al., 2024] は凍結した wav2vec 2.0 の特徴とロジスティック回帰という本研究と同じ型の学習器で，ターゲットのどの発話にラベルを付けるかを能動学習で選び，コーパス間ではラベルが増えるとランダムとの差が消えることを報告した．本研究は選び方ではなく，ソースを混ぜる価値と K 本のラベルが直す誤差の成分を測る．」
- 本研究の否定結果（強い出発点の上ではラベル無しの選び方がランダムを上回らない）と同じ向きの観察として引ける

### Gaps（7. 注意）
- 主結果は図だけで，K ごとの BCA の値は写せない
- 号（15(3)）は前回の OpenAlex の記録による

---

## 4. Feng & Chaspari（2023）

### Takeaway
台本の音声（4 コーパスを 1 つずつ）から即興の音声（IEMOCAP-SP）へ，話者あたり・感情あたり 1〜10 本のターゲットラベルで転移する．手作り特徴＋小さな網で，微調整は 1 本のとき負の転移を起こす．数値は図だけで，検定は無い．

### Cited Findings

**1. 書誌と読んだ範囲**
- 書誌（確かめた）：K. Feng, T. Chaspari, "Few-Shot Learning in Emotion Recognition of Spontaneous Speech Using a Siamese Neural Network With Adaptive Sample Pair Formation," IEEE Transactions on Affective Computing, vol. 14, no. 2, pp. 1627–1633, 2023（早期公開 2021），DOI 10.1109/TAFFC.2021.3109485 — [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/taffc.2021.3109485?fields=title,authors,venue,journal,year)（号は前回の調べの OpenAlex による）
- 読んだ範囲（◎）：arXiv v1（2021-09-07）の全節 1〜6 — [arXiv 2109.02915v1](https://arxiv.org/abs/2109.02915)

**2. 設定**（[arXiv 2109.02915v1](https://arxiv.org/pdf/2109.02915) 3・4.5・5）
- 課題：3 感情（anger・happiness・sadness）の分類．ソースは台本の音声（IEMOCAP-SC 133 分，RAVDESS 36 分，CREMA-D 159 分，eNTERFACE'05 30 分）を**1 つずつ**，ターゲットは即興の IEMOCAP-SP（92 分）
- モダリティ：音声のみ
- 特徴：openSMILE の IS09 設定 64 次元（手作り）．データセットごとに標準化（IEMOCAP の SC と SP は別々）
- ターゲットのラベル数：**話者あたり・感情あたり** 1 本から 10 本（原文："randomly selecting 1 labelled sample per speaker and per emotion from the target data and using the remaining samples in the test set. We repeat this process 10 times ... Then, we gradually increase the number of samples to 2, . . . , 10 clips per speaker and per emotion"）．IEMOCAP は 10 話者なので，クラスあたり 10〜100 本に当たる
- 学習器：5 層の FNN（64・32・16・16・3）とシャム網（MeL，MeL-S，MeL-S-ASPF）．基準は FNN の微調整（ソースで事前学習し全層を更新）と ADDA（CNN＋対数メルの区画．ターゲットで微調整）
- 評価：UAR（でたらめ 33.3%）．10 反復の平均．検定は無い．**ターゲットの K 本はテストと同じ話者から抜く**（話者をまたぐ分離はしない．コーパス内の LOSO はソース無しの in-domain の基準にだけ使う）

**3. 示したこと**（[arXiv 2109.02915v1](https://arxiv.org/pdf/2109.02915) 5 節．数値は図 3 のみ）
- in-domain（ターゲットだけで LOSO）の UAR 72.5%．out-of-domain（ソースだけ）は 49〜68%．"RAVDESS, CREMA-D, and IEMOCAP-SC provide relatively successful out-of-domain training (i.e., 67.4%)"，eNTERFACE'05 は差が大きい
- 最良は CREMA-D をソースにした MeL-S-ASPF で話者・感情あたり 10 本のとき UAR 74% 超（in-domain を上回る）．要旨："70% unweighted average recall using 2-3 labelled samples per class on a 3-way classification task"
- 相対改善："MeL-S and MeL-S-ASPF yield a relative improvement of 6-23% compared to out-of-domain learning and 1-7% compared to the FNN fine-tuning baseline."
- 負の転移（原文）："When we only use one labelled sample per speaker and emotion, negative transfer occurs for the FNN fine-tuning in many cases (e.g., RAVDESS, IEMOCAP-SC), indicating that the corresponding approach may not be promising for transferring knowledge between scripted and spontaneous speech in emotion recognition."
- MeL（教師なしの距離学習だけ）は FNN 微調整と ADDA を上回らない

**4. 軸との対応**
- A1 ×：手作り特徴＋FNN／シャム網（線形の頭ではない）
- A2 ×，A3 ×，A4 ×
- A5 ○：1〜10 本を 10 点振る．単位は話者あたり・感情あたり（クラスあたりでは 10〜100 本）
- A6 △：コーパス間だがソースは 1 つずつ（混ぜない）．コーパスは 5 つ（ソース 4＋ターゲット 1）
- A7 ×：音声のみ
- A8 ×，A9 ×（微調整の出発点にするだけ）
- A10 △：UAR と 10 反復はあるが，検定・効果量・相手の調整の記述は無い．ターゲットの K 本とテストが同じ話者
- A11 ×：ずれは測らない．eNTERFACE の差を「俳優でない参加者」で言葉で説明するだけ

**5. 穴**
- していないこと：複数ソースの合併，ずれの測定，検定，クラスあたりの K の定義，話者をまたぐ評価
- 限界として自分で書いていること（原文）："First, the spontaneous speech data from IEMOCAP-SP are collected in laboratory conditions involving high quality microphones and low levels of noise. ... the fact that both source and target data have been collected in laboratory conditions has the potential to decrease the domain mismatch between them." "Second, the source and target audio samples are uttered in English"
- 今後の課題（原文）："Finally, we studied each of the source datasets in isolation. Inspired by prior findings [10], [23], it would be beneficial to explore the extent to which the combination of multiple source datasets can benefit few-shot learning. It would be also interesting to investigate the interplay between the properties of the various source datasets (e.g., including complementary information) and their effectiveness in few-shot learning."

### Inferences（6. 本研究との差の書き方）
- 案：「Feng & Chaspari [2023] は台本の音声コーパスを 1 つずつソースにして即興の音声へ話者・感情あたり 1〜10 本のラベルで転移し，微調整が 1 本では負の転移を起こすことを示した．本研究は，この負の転移が線形の頭では切片と向きのどちらの誤差から生じるかを分け，複数ソースの合併の価値をラベルの本数で表す．」
- 著者が今後の課題に挙げた「複数ソースの組み合わせ」と「ソースの性質と効き目の関係」は，本研究の問い（ソースの価値，得が生じる条件）に直接つながる

### Gaps（7. 注意）
- 数値は図 3 だけなので，K ごとの UAR は写せない
- ターゲットの K 本とテストが同じ話者であることは，本研究の話者分離の評価と並べるときに明記する

---

## 5. ADDoG（Gideon, McInnis, Mower Provost，2021）

### Takeaway
感情価の 3 区分だけを対象に，ターゲットのラベルを 0〜3,200 発話（合計）で振り，ラベルが少ないときは一般化（ADDoG）が勝ち，多いときはターゲットだけの特化（SP）が並ぶか勝つことを ANOVA で示した．ハイパーパラメータはターゲットのラベルが無いことを理由に固定した．

### Cited Findings

**1. 書誌と読んだ範囲**
- 書誌（確かめた）：J. Gideon, M. G. McInnis, E. Mower Provost, "Improving Cross-Corpus Speech Emotion Recognition with Adversarial Discriminative Domain Generalization (ADDoG)," IEEE Transactions on Affective Computing, vol. 12, no. 4, pp. 1055–1068, 2021（早期公開 2019），DOI 10.1109/TAFFC.2019.2916092，PMC9173710 — [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/taffc.2019.2916092?fields=title,authors,venue,journal,year)（号は前回の調べの PMC による）
- 読んだ範囲（◎）：arXiv 1903.12094（IEEE 2019 の著作権注記つきの採録版）の全節 1〜7 — [arXiv 1903.12094](https://arxiv.org/abs/1903.12094)

**2. 設定**（[arXiv 1903.12094](https://arxiv.org/pdf/1903.12094) 3・4・5・6）
- 課題：**感情価のみ**．注釈者の評点を低・中・高の 3 区分に数えて柔らかいラベルにする．多数決の区分が無い発話は除く．覚醒度は予備実験で効果が無く除外
- データ：IEMOCAP（同点を除き 6,816 発話：低 3,181・中 1,641・高 1,994），MSP-Improv（7,852：2,160・2,961・2,731），PRIORI Emotion（8,685：2,809・4,779・1,097．双極性障害の患者と対照 12 人の電話，8 kHz）
- モダリティ：音声のみ（40 次元の対数メル）
- ターゲットのラベル数：実験 2〜4 で **0・200・400・800・1,600・3,200 発話（合計）**．ターゲットを発話単位でランダムに半分に割り，片方から K 本を抜いて 80:20 で訓練・検証，もう片方でテスト．入れ替えて繰り返す．残りのラベル無し発話は学習に使う（トランスダクティブ）
- 学習器：CNN（畳み込み＋大域最大＋全結合，128 チャネル），ADDoG（＋WGAN の批評家），MADDoG（複数ソース），SP（ターゲットのラベルだけで学ぶ CNN）
- ハイパーパラメータ固定の理由（原文）："It is difficult to validate multiple sets of hyperparameters when conducting cross-dataset experiments, due to the lack of labelled data in the target domain. For this reason, we select hyperparameters based on those found to be commonly selected in prior work and keep them constant for all experiments."
- 評価：被験者ごとの UAR（でたらめ 0.33）．10 反復（実験 1 は 50）．検定（原文）："Significance is determined using an analysis of variance in R [56] over the matrix of subject UARs output by the compared methods"（p=0.05）．**話者分離は無い**（ターゲットの半分割は発話単位）

**3. 示したこと**（[arXiv 1903.12094](https://arxiv.org/pdf/1903.12094) 6 節）
- 表 2（ラベル 0）：MSP-Improv→IEMOCAP CNN 0.439±0.022，ADDoG 0.474±0.009（有意）；IEMOCAP→MSP-Improv 0.432±0.012 対 0.444±0.007（有意）
- 実験 2（図 6 のみ）：ADDoG は全ての K で CNN より有意に良いが差は K とともに縮む．MSP-Improv→IEMOCAP の 200 本では，ラベル付き ADDoG がラベル無し ADDoG より悪い（検証集合が 40 発話しか無いためと推測）．IEMOCAP→MSP-Improv ではラベルが多いと SP が CNN を上回る
- 実験 3（図 7 のみ．研究室→電話）：ソースとターゲットを単純に合併する CNN は "entirely fails"．ADDoG は 800 本以下で有意に良く，"the advantage of using ADDoG is approximately similar to that attained by doubling the amount of labelled TAR data"．IEMOCAP→PRIORI の 3,200 本では ADDoG が有意に悪い．2 ソース（IEMOCAP＋MSP-Improv）では MADDoG がラベル付きの全条件で最良（200 本の ADDoG を除き有意）
- 実験 4（図 8 のみ．電話→研究室）：CNN が最悪．十分なラベルがあると SP が ADDoG と同等か上

**4. 軸との対応**
- A1 ×：対数メルからの CNN
- A2 ×，A4 ×．A3 △：「ADDoG の利得はターゲットのラベルを 2 倍にするのと同程度」という 1 文の換算だけ（式や系統的な測定は無い）
- A5 ×：最小 200 発話（区分あたり約 67）．K を 6 点振る
- A6 ○：3 コーパス，実験 3c で 2 ソース
- A7 ×：音声のみ
- A8 ×（ラベル無しのターゲットは批評家の学習に使い，擬似ラベルは付けない），A9 ×
- A10 △：UAR，10 反復，ANOVA はある．補正・効果量は無い．ハイパーパラメータは全手法で固定（相手も同じ扱い）．検証はターゲットの K 本の 20% で実用値．話者分離は無い
- A11 ×：Wasserstein 距離は学習の目的関数としてだけ使い，ずれの量は報告しない

**5. 穴**
- していないこと：感情のカテゴリ，少数（K ≤ 10）の領域，話者分離，ずれの測定，ラベルが直すものの分析
- 限界として自分で書いていること（原文）："While ADDoG and MADDoG are significantly the best methods in most cases, SP performs comparably or better when enough TAR labels are available and TAR and SRC are very different. This indicates that generalized models can only go so far and it is important to understand when a domain is so different that a specialized model may be needed."
- 今後の課題（原文）："Once enough labelled data is available in a target domain, it is often better to just specialize with it instead of using information from other domains, as seen in Section 6.3. Because of this, it is important to formulate exactly when to generalize and when to specialize. We plan on investigating a method that fuses both approaches and estimates which is better, given that domain data currently present."

### Inferences（6. 本研究との差の書き方）
- 案：「ADDoG [Gideon et al., 2021] は感情価の 3 区分でターゲットのラベルを 200〜3,200 発話まで振り，ラベルが少ないときはソースを使う一般化が，多いときはターゲットだけの特化が勝つことを示し，『いつ一般化し，いつ特化すべきかを定式化する』ことを今後の課題に挙げた．本研究は線形の頭でこの境目（K*）を切片と向きの誤差の式から予言し，ソースの価値をラベルの本数で表す．」
- 「利得はラベルを 2 倍にするのと同程度」という記述は，ソースの価値をラベル本数に換算する発想の先例として引ける（ただし 1 文の観察であり測定ではない）

### Gaps（7. 注意）
- 実験 2〜4 の数値は図のみ
- 掲載版（TAC 12(4)）と arXiv の採録版の差は未確認（PMC 版は前回の調べで読んでおり内容は一致する）

---

## 6. Chen & Yu（2023）

### Takeaway
4 つの感情コーパスの全部分集合（16 通り）で Wav2Vec2 を事前微調整し，ESD の話者ごとに k=2〜64 本（合計．2 値分類）で微調整する 33,600 回の実験で，ソースの組み合わせの価値を総当たりで測った．ソースが 1 つだと害になりうること，MSP-PODCAST が最も効き IEMOCAP が k=32・64 で害になることを報告した．

### Cited Findings

**1. 書誌と読んだ範囲**
- 書誌（確かめた）：M. Chen, Z. Yu, "Pre-Finetuning for Few-Shot Emotional Speech Recognition," Interspeech 2023, pp. 3602–3606, DOI 10.21437/Interspeech.2023-136 — [Crossref API](https://api.crossref.org/works?query.bibliographic=Pre-Finetuning+for+Few-Shot+Emotional+Speech+Recognition&rows=3)
- 読んだ範囲（◎）：arXiv v3（2024-11-07）の全節 1〜6 — [arXiv 2302.12921v3](https://arxiv.org/abs/2302.12921)

**2. 設定**（[arXiv 2302.12921v3](https://arxiv.org/pdf/2302.12921) 3・4）
- 課題：ESD（英語 10 話者＋北京語 10 話者，5 感情）で，話者ごと・感情ごとの 2 値分類（例：Happy か否か）．「話者適応を少数ショット学習とみなす」
- ソース：MSP-IMPROV（8,438 発話，4 感情），MSP-PODCAST（62,140 発話，9 ラベル），Mandarin Affective Speech（25,636 発話，68 話者，5 ラベル），IEMOCAP（10,039 発話，9 ラベル）．全部分集合 16 通り（空集合＝事前微調整なしを含む）でそれぞれ事前微調整（課題ごとに別の線形の頭，損失を出力数で割って平均）
- モダリティ：音声のみ
- ターゲットのラベル数：k ∈ {2, 4, 8, 16, 24, 32, 64}（**合計**．正例 k/2 ＋ 負例 k/2）．各条件 3 試行．合計 33,600 試行
- 学習器：Wav2Vec2.0 base（94.4M）を全体で微調整（最大 200 エポック，早期終了 30）．ComParE 2016 特徴＋DNN は定数予測を超えられず捨てた
- 評価：Macro-F1．図は平均と標準誤差．検定は無い．**同じ話者の中で訓練と評価**（話者をまたぐ評価ではない）

**3. 示したこと**（[arXiv 2302.12921v3](https://arxiv.org/pdf/2302.12921) 4 節）
- 表 1（各コーパスだけで事前微調整した F1in 対 そのコーパスを除く 3 つで事前微調整した F1ex，Δ）：
  - k=2：IEMOCAP 0.5048／0.6232／−0.1184；Mandarin AS 0.5812／0.6445／−0.0633；MSP-IMPROV 0.5233／0.6299／−0.1066；MSP-PODCAST 0.6150／0.6272／−0.0122
  - k=4：0.5447／0.7054／−0.1607；0.6495／0.7160／−0.0665；0.5623／0.7030／−0.1407；0.7010／0.6990／＋0.0020
  - k=8：0.5816／0.7747／−0.1931；0.7040／0.7862／−0.0822；0.6021／0.7783／−0.1762；0.7640／0.7549／＋0.0091
  - 脚注：k=64 の MSP-PODCAST は F1in 0.9126 対 F1ex 0.9108
- 図 2（ソースの数 n と F1）："In six of the seven few-shot settings, using only one pre-finetuning corpus may hurt performance compared to direct fine-tuning without a pre-finetuning step ... However, in the most extreme case with k = 2, even using just one pre-finetuning corpus still yields performance improvements over the baseline. After n = 1, we witness continuous improvements in average performance as n increases."
- 図 3（コーパスごとの寄与）："models pre-finetuned on combinations of corpora which include MSP-PODCAST results in the most improvements over baseline performance on average, whereas IEMOCAP results in the least improvements. In the k = 32 and k = 64 few-shot data settings, pre-finetuning on IEMOCAP actually hurts performance on average."
- 図 4："the performance gains generally taper off substantially after k = 24." Surprise だけは事前微調整で良くならない（ソースに Surprise が少ない）
- 結論："boosting performance from near random to over 0.600 F1 using k = 2 training examples"

**4. 軸との対応**
- A1 ×：Wav2Vec2 を全体で微調整
- A2 ×，A4 ×．A3 △：ソースの価値を F1 の差（F1in−F1ex，対基準の差）で測る．ラベルの本数には換算しない
- A5 ○：k=2〜64（合計．2 値なのでクラスあたり 1〜32）を 7 点振る
- A6 ○：4 ソースの全部分集合（複数ソース）．ターゲットは別コーパス（ESD）
- A7 ×：音声のみ
- A8 ×，A9 ×（事前微調整したモデルを出発点にするだけ）
- A10 ×：Macro-F1（UAR ではない），3 試行，検定無し，同じ話者の中で評価
- A11 ×：ずれは測らない．MSP-PODCAST が効く理由を「最大で感情の種類が多い」と言葉で説明するだけ

**5. 穴**
- していないこと：話者をまたぐ評価，多クラス，凍結特徴，ずれの測定，ソースの価値のラベル換算，検定
- 限界として自分で書いていること（原文）："In this study we only used four corpora due to licensing and computational constraints, but our findings warrant examining the upside of more corpora." "In this study, we held the model choice fixed, but our workflow is compatible with any pre-trained model."
- 今後の課題：上の 2 文に含まれる（より多くのコーパス，より大きいモデル）

### Inferences（6. 本研究との差の書き方）
- 案：「Chen & Yu [2023] は 4 つの音声コーパスの全部分集合で Wav2Vec2 を事前微調整し，話者ごとの 2 値分類を 2〜64 本で微調整して，ソースの組み合わせが少数ラベルの学習に与える利得を総当たりで測った．本研究はソースの価値を F1 の差ではなくターゲットのラベルの本数に換算し，その値をずれの指標から予言する．」
- 「ソースが 1 つだと害になり，k が増えると IEMOCAP が害になる」という観察は，本研究の「得が消える条件（K*）」と同じ現象を微調整の設定で見たものとして引ける

### Gaps（7. 注意）
- k=16〜64 の表 1 は省略されている（本文の脚注で「同じ傾向」とだけ）
- Interspeech 版（2023）と arXiv v3（2024-11）の差は未確認

---

## 7. Li, Kim, Kuo, Narayanan（2021）

### Takeaway
MSP-IMPROV の中の収集条件（自然な会話・即興・台本の即興・朗読）をドメインとみなし，音声＋顔で，ターゲットの 10% のラベルを使う適応を報告した．コーパス間ではない．台本（TR）へは 10% で 0.50→0.58 に上がるが，内容が重なる組（NI+OI→TI）ではラベルを足しても上がらず，敵対学習が下げた．

### Cited Findings

**1. 書誌と読んだ範囲**
- 書誌（確かめた）：H. Li, Y. Kim, C.-H. Kuo, S. Narayanan, "Acted vs. Improvised: Domain Adaptation for Elicitation Approaches in Audio-Visual Emotion Recognition," Interspeech 2021, pp. 3395–3399, DOI 10.21437/Interspeech.2021-666 — [Crossref API](https://api.crossref.org/works?query.bibliographic=Acted+vs.+Improvised%3A+Domain+Adaptation+for+Elicitation+Approaches+in+Audio-Visual+Emotion+Recognition&rows=3)
- 読んだ範囲（◎）：arXiv v2（2021-06-07）の全節 1〜6 — [arXiv 2104.01978v2](https://arxiv.org/abs/2104.01978)

**2. 設定**（[arXiv 2104.01978v2](https://arxiv.org/pdf/2104.01978) 2・3・4）
- 課題：4 感情（Angry・Sad・Happy・Neutral）の分類．ドメインは MSP-IMPROV の発話の種類：NI（自然な会話 2,640），OI（即興の他の発話 3,940），TI（即興の目標文 640），TR（朗読の目標文 578）．実験は NI+OI→TR，NI+OI→TI，TI+TR→NI，TI+TR→OI
- モダリティ：音声（40 次元メルフィルタバンク＋エネルギー → Conv1D×2＋GRU）＋顔（顔認識で事前学習した inception-ResNet 系の 512 次元埋め込み → GRU）．結合して全結合
- ターゲットのラベル数：**ターゲットの 10%** を訓練，40% を開発，50% を評価（TR なら約 58 発話，TI 約 64，NI 約 264，OI 約 394．クラスあたりの定義は無い）
- 学習器：深層（上記）．損失：感情の交差エントロピー＋最大エントロピーの敵対損失（勾配反転）＋ソフトラベル損失（ソースのモデルのクラスごとの平均出力をラベルにする．τ=2）．Adam，L2 正則化 0.01（重みをゼロに寄せる通常のもの）
- 評価：4 クラスの平均再現率（＝UAR）．排他的な訓練標本で 5 回の平均と標準偏差．検定は無い．話者の扱いは書かれていない

**3. 示したこと**（[arXiv 2104.01978v2](https://arxiv.org/pdf/2104.01978) 表 4・5）
- 表 4（NI+OI→TR／NI+OI→TI．平均（標準偏差））：Source only 0.5005 (0.021)／0.7330 (0.020)；Source＋Target 0.5837 (0.031)／0.7288 (0.034)；敵対損失のみ 0.6225 (0.018)／0.7002 (0.007)；敵対＋ソフトラベル 0.6339 (0.023)／0.7092 (0.022)
- 表 5（TI+TR→NI／TI+TR→OI）：Source only 0.4296 (0.028)／0.5499 (0.010)；Source＋Target 0.4440 (0.032)／0.5886 (0.010)；敵対損失のみ 0.4601 (0.013)／0.5904 (0.024)；敵対＋ソフトラベル 0.4653 (0.023)／0.5920 (0.017)
- 原文："Even though only a few samples (10% of the samples from the target domain) are employed during training, the performance improves from 0.5005 to 0.5837 directly." "However, in the experiment of NI+OI → TI, there is no improvement by adding target training samples. Surprisingly, in contrast to the previous findings, the domain adversarial loss and softlabel loss even worsen the performance." 理由は OI に TI の台本の練習発話が多く含まれるため
- 序論の記述："traditional transfer learning approaches, such as fine-tuning on target domain, often require hundreds of labeled samples"

**4. 軸との対応**
- A1 ×：畳み込み＋GRU を学習（顔の埋め込みは凍結だが頭は深層）
- A2 ×，A3 ×，A4 ×
- A5 △：ターゲットの 10%（固定．振らない．単位は割合）
- A6 ×：1 つのコーパスの中の収集条件の間（コーパス間ではない）
- A7 ○：音声＋顔の融合（先生→生徒は無い）
- A8 ×：ソフトラベルはクラスごとの類似度の目標で，ターゲットの標本への弱いラベルではない．A9 ×（L2 はゼロへの正則化）
- A10 △：UAR と 5 反復の標準偏差はあるが，検定・話者分離の記述は無い
- A11 ×：ずれは測らない．内容の重なりを言葉で説明するだけ

**5. 穴**
- していないこと：コーパス間，K を振ること，ずれの測定，検定
- 限界として自分で書いていること：限界の節は無い．上の NI+OI→TI の失敗の原因（OI と TI の重なり）を書くだけ
- 今後の課題：書かれていない（結論は仮説 H1・H2 の確認で終わる）

### Inferences（6. 本研究との差の書き方）
- 案：「Li et al. [2021] は MSP-IMPROV の収集条件（朗読・即興・自然な会話）をドメインとみなし，音声と顔を融合した深層モデルでターゲットの 10% のラベルを使う適応を報告した．同じコーパスの中の条件間であり，コーパス間ではない．本研究は 7 群のコーパス間で，音声・顔・言葉の 3 モダリティと線形の頭を使う．」
- 「内容が重なるソースではラベルを足しても上がらない」という観察は，本研究の「得が消える条件」の例として引ける

### Gaps（7. 注意）
- 10% がクラスあたり何本かは表 2 から概算するしかない（TR の angry 169 → 約 17，happy 88 → 約 9）
- Interspeech 版と arXiv v2 の差は未確認

---

## 8. Ahn, Rana, Busso, Rajapakse（2025）

### Takeaway
TAC 掲載版は本文が取れず**抄録のみ**（○）．音声＋言葉の別々のマルチタスク Transformer を判定段で融合し，ラベル無しのターゲットを補助課題（対照学習と情報最大化）に使う．「コーパス間でのマルチモーダルの手法は大きく限られる」と書く．前身の arXiv 2023 版（◎．音声のみ）は MSP-IMPROV の 5% のラベルを使い，IEMOCAP→MSP-IMPROV の 2 値の UAR を 59.39（ラベル無し）→69.25（5% のラベル）と報告する．

### Cited Findings

**1. 書誌と読んだ範囲**
- 書誌（確かめた）：C. Ahn, R. Rana, C. Busso, J. C. Rajapakse, "Multitask Transformer for Cross-Corpus Speech Emotion Recognition," IEEE Transactions on Affective Computing, vol. 16, no. 3, pp. 1581–1591, 2025, DOI 10.1109/TAFFC.2025.3526592 — [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/taffc.2025.3526592?fields=title,authors,venue,journal,year,abstract)；号は [OpenAlex](https://api.openalex.org/works/https://doi.org/10.1109/taffc.2025.3526592) による
- 読んだ範囲：掲載版は IEEE Xplore（document 10830494）の本文も PDF も取得できず，**抄録のみ（○）**．前身の C.-S. Ahn, J. C. Rajapakse, R. Rana, "Integrating Contrastive Learning into a Multitask Transformer Model for Effective Domain Adaptation," arXiv 2310.04703v1（2023-10-07，5 ページ）は全節を読んだ（◎） — [arXiv 2310.04703](https://arxiv.org/abs/2310.04703)．TAC 版は Busso が加わり，言葉のモダリティと EMO-DB が加わっている（抄録）．TAC 版がターゲットのラベルを使うかは抄録からは分からない

**2. 設定**
- TAC 版（抄録のみ）："we propose a multitask framework with SER as the primary task and contrastive learning and information maximization as auxiliary tasks. We design the auxiliary tasks innovatively to use the target data without emotional labels" "We use text as the second modality, developing separate multitask transformers for audio and text and conduct decision-level fusion during inference. We use ... the IEMOCAP, MSP-IMPROV and EMO-DB databases." — [Semantic Scholar API（抄録）](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/taffc.2025.3526592?fields=abstract)
- arXiv 2023 版（[arXiv 2310.04703](https://arxiv.org/pdf/2310.04703) 3・4）：wav2vec2-base-960h を微調整（13 層の出力の学習する重み付き和＋softmax）．損失は感情の交差エントロピー＋拡張の種類の分類＋InfoNCE の対照損失＋情報最大化（λ1=0.1，λ2=0.5，λ3=0.5）．課題は **2 値**（neutral・happy・excited を正，sad・angry を負）．ソース IEMOCAP（90% 訓練・10% 検証），ターゲット MSP-IMPROV の **5% をラベル付きの訓練，95% をテスト**（ラベル無しの設定では 30%／70%）．ソースとターゲットを 1 エポックずつ交互に学習し，ソースの検証精度で早期終了．UAR．反復・検定・話者の扱いは書かれていない．原文："Our work is assuming that small fraction of target data with label is available."

**3. 示したこと**（[arXiv 2310.04703](https://arxiv.org/pdf/2310.04703) 表 1・2）
- 表 1（UAR）：CNN-BLSTM 59.52%；CNN-BLSTM＋DANN＋CenterLoss 57.26%；CNN-LSTM 55.73%；DoGAT 59.42%；Ours（ラベル無し）59.39%；Ours（5% のラベル）69.25%
- 表 2（消去．ラベル無し）：w/o Laug 59.10%；w/o LIM & Laug 58.68%；w/o Lcont & Laug 57.37%
- 著者の主張："our model improved 10% above state-of-the-art with only 5% of the target dataset with labels."（比較相手はターゲットのラベルを使わない手法と読めるので，ラベルの予算が揃っていない）

**4. 軸との対応**（TAC 版は抄録，arXiv 版は本文に基づく）
- A1 ×：wav2vec2 を微調整
- A2 ×，A3 ×，A4 ×
- A5 ×：arXiv 版は 5%（MSP-IMPROV 8,438 発話なら約 420 本．2 値でクラスあたり約 200）で振らない．TAC 版は不明
- A6 ○：コーパス間．arXiv 版は IEMOCAP→MSP-IMPROV（1 組），TAC 版は 3 コーパス（組は不明）
- A7 ○（TAC 版，抄録）：音声＋言葉の判定段の融合．arXiv 版は音声のみ
- A8 ×（情報最大化はエントロピーの目的で擬似ラベルではない），A9 ×
- A10 ×：UAR はあるが反復・検定・話者の扱いが無い．ラベルの予算が相手と揃わない
- A11 ×

**5. 穴**
- していないこと：K を振ること，ずれの測定，検定．TAC 版でラベルを使うかは不明
- 限界・今後の課題：arXiv 版には無い．TAC 版は本文が読めず不明
- 抄録の記述（原文）："While transformers have gained attention in SER, their application to cross-corpus scenarios is still limited. Multimodal approaches for cross-corpus scenario is substantially limited as well."

### Inferences（6. 本研究との差の書き方）
- 案：「Ahn et al. [2025] は音声と言葉のマルチタスク Transformer を判定段で融合してコーパス間の音声感情認識を扱い，『コーパス間でのマルチモーダルの手法は大きく限られる』と述べた．ターゲットは補助課題にラベル無しで使う．本研究はターゲットのラベルを少数使う設定で，音声・顔・言葉の 3 モダリティを扱う．」
- 「マルチモーダルのコーパス間は限られる」の TAC 内の根拠として抄録を引く．引くときは「抄録による」と明記する

### Gaps（7. 注意）
- TAC 版の本文は未読（IEEE Xplore の取得は自動取得の制限で止めた）．設定（ターゲットのラベルの有無，コーパスの組，K）は要確認
- arXiv 版と TAC 版は著者も題名も違うので，引用は TAC 版の書誌で行い，数値を引くなら arXiv 版と明記する

---

## 9. Dong, Jia, Dang（2026）

### Takeaway
ターゲットのラベルもソースのデータも使わないテスト時適応（TTA）11 手法を，IEMOCAP と RAVDESS の 3 課題（個人化・台本→即興・コーパス間）で比べた．誤差逆伝播を使わない手法（T3A・LAME・FOA）が最も有望で，エントロピー最小化と擬似ラベルは「1 つの確かな正解ラベル」の前提が感情の曖昧さに合わないためおおむね失敗する．コーパス間では T3A が正解率で平均＋3.1，F1 で＋4.1．台本→即興の利得は最小．

### Cited Findings

**1. 書誌と読んだ範囲**
- 書誌（確かめた）：J. Dong, H. Jia, T. Dang, "Test Time Adaptation for Speech Emotion Recognition," ICASSP 2026, pp. 17157–17161, DOI 10.1109/ICASSP55912.2026.11460722 — [Crossref API](https://api.crossref.org/works?query.bibliographic=Test-Time+Adaptation+for+Speech+Emotion+Recognition+Dong+Jia+Dang&rows=3)
- 読んだ範囲（◎）：arXiv v1（2026-01-21）の全節 1〜5 — [arXiv 2601.16240v1](https://arxiv.org/abs/2601.16240)

**2. 設定**（[arXiv 2601.16240v1](https://arxiv.org/pdf/2601.16240) 2・3）
- 課題：ラベル無しのターゲットだけで推論時にモデルを合わせる TTA．ソースのデータは使わない
- データ：IEMOCAP（4 感情，excited を happy に併合，5,531 発話），RAVDESS の音声（24 俳優，1,440 発話，8 感情．コーパス間では共通の 4 感情）
- モダリティ：音声のみ
- ターゲットのラベル数：0
- 学習器：wav2vec2-base をソースで微調整（AdamW，50 エポック，学習率 3e-5）＋全結合 256 の分類器．ソースの学習は深層．T3A はこの分類器の頭（クラスのプロトタイプ）だけを，確信度の高い擬似ラベルの標本の平均で作り直す（誤差逆伝播なし）．LAME はパラメータを変えず予測分布を「ソースへの忠実さ」と「近傍の一貫性」で直す．FOA はプロンプトの埋め込みを CMA-ES で探し，エントロピーとソース・ターゲットの特徴の平均・標準偏差の差を下げる
- 比べた 11 手法：エントロピー最小化（Tent，SAR，EATA，SUTA，DSUTA，CEA），擬似ラベル（CoTTA，AWMC），誤差逆伝播なし（T3A，LAME，FOA）
- 評価：正解率と macro F1．課題 1（個人化）は IEMOCAP で leave-one-session-out，RAVDESS は 20 人で学習・4 人でテスト．課題 2 は IEMOCAP の台本→即興．課題 3 は IEMOCAP↔RAVDESS．バッチ 32．ハイパーパラメータ（原文）："We follow the original settings of each TTA method. For fairness, hyperparameters were chosen by grid search."（何で選んだかは書かれていない）．反復・検定は無い

**3. 示したこと**（[arXiv 2601.16240v1](https://arxiv.org/pdf/2601.16240) 表 1，4 節）
- 表 1(c)（コーパス間．正解率／F1 %．列はターゲット）：
  - ターゲット RAVDESS（ソース IEMOCAP）：Source 37.8／26.7；Tent 37.4／26.2；SAR 37.8／26.7；EATA 37.8／26.7；SUTA 31.1／16.3；DSUTA 30.5／14.9；CEA 31.6／17.2；CoTTA 32.7／19.0；AWMC 37.8／26.7；T3A 43.8／34.3；LAME 28.7／12.3；FOA 40.9／30.3
  - ターゲット IEMOCAP（ソース RAVDESS）：Source 50.0／45.7；Tent 50.0／45.6；SAR 50.0／45.7；EATA 49.9／45.7；SUTA 48.9／45.5；DSUTA 50.1／45.8；CEA 50.1／46.1；CoTTA 50.0／45.6；AWMC 50.0／45.7；T3A 50.2／46.2；LAME 49.6／45.2；FOA 49.4／45.3
  - 原文："T3A reached the best performance on both datasets, showing a 3.1% and 4.1% average gain for accuracy and F1 score over the source baselines."（RAVDESS 側の＋6.0／＋7.6 と IEMOCAP 側の＋0.2／＋0.5 の平均）
- 表 1(b)（台本→即興，IEMOCAP）：Source の正解率 51.3．最良は LAME で 51.5／51.3．原文："the improvement is minimal, as with most other TTA methods."
- 表 1(a)（個人化）：誤差逆伝播なしが平均で最良，FOA が最良（数値は抽出が乱れたので写さない）
- 失敗の条件（原文）："entropy minimization and pseudo-labeling methods assume a single true class and maximize its probability, which is less suited to SER, where emotions are ambiguous and one-label ground truths may not reflect underlying emotion" "Freezing the backbone helps preserve source knowledge, which is advantageous with subtle distribution shifts, as in intra-corpus personalization." "We hypothesize that the domain shift between acted and natural emotions is too complex and inconsistent for current TTA methods to model effectively. TTA typically excels when domain shifts are uniform and global" "when the source model performance is weak, it generates a high proportion of low-confidence predictions. By recalibrating the prototype using only the most confident samples, the method effectively filters out this noise" "TTA might be more effective at adapting medium to severe distribution shifts compared to minor ones in SER." バッチ 1 ではエントロピー最小化が不安定．RAVDESS では LAME のためにバッチが大きいほど誤差逆伝播なしの平均が下がる
- 要旨："The effectiveness of TTA methods applied with SER is bounded by type and severity of domain shifts."

**4. 軸との対応**
- A1 △：線形の頭（全結合 1 層）だが，エンコーダはソースで微調整（凍結ではない）．T3A は頭だけを作り直す
- A2 ×，A3 ×，A4 ×（T3A はクラスのプロトタイプを作り直すが，切片の解析はしない）
- A5 ×：ラベル 0
- A6 ○：コーパス間（IEMOCAP↔RAVDESS）．ソース 1 つ，コーパス 2 つ
- A7 ×：音声のみ
- A8 △：T3A はソースのモデルの出力をラベル無しのターゲットの擬似ラベルにし，確信度の高い標本だけでクラスごとのプロトタイプを作る（クラスごとに作るので均衡に近いが，均衡の操作は無い．ターゲットのラベルとは混ぜない）
- A9 △：LAME の「ソースへの忠実さ」は予測分布をソースの予測に寄せる項で，重みの出発点への L2 罰則ではない
- A10 ×：正解率と macro F1（UAR ではない），反復・検定なし，ハイパーパラメータの選び方の根拠が不明．個人化の課題は話者を分ける
- A11 △：ずれの「種類と強さ」で効き目が決まると言葉で述べ，台本→即興（小さく不均一）とコーパス間（大きい）を対比する．ずれの量は測らない

**5. 穴**
- していないこと：ターゲットのラベルを使う比較（微調整や少数ショットの基準が無い），ずれの定量化，反復・検定，UAR
- 限界として自分で書いていること：限界の節は無い．上の "no single method universally excels, and its effectiveness is highly dependent on the distributional shifts and tasks."
- 今後の課題（原文）："This study provides a foundational investigation and offers critical insights for future research into more robust and adaptive SER systems."（具体的な項目は無い）

### Inferences（6. 本研究との差の書き方）
- 案：「Dong et al. [2026] はラベル無しのターゲットだけで合わせるテスト時適応 11 手法を比べ，エントロピー最小化と擬似ラベルは感情の曖昧さのためおおむね失敗し，分類器の頭だけを作り直す手法が最も有望だと報告した．本研究はターゲットのラベルを少数使い，ソースだけの分類器の出力を弱いラベルにしてクラスを均衡させる手を，線形の頭の上で理論と結びつける．」
- 「頭（プロトタイプ）だけを直すのが最も有望」「擬似ラベルで全体を更新すると失敗」という観察は，本研究の「切片だけを直す」「弱いラベルは線形の頭に限って使う」という設計の傍証として引ける

### Gaps（7. 注意）
- 表 1 の多くのエントロピー最小化・擬似ラベルの値がソースの値と一致する（例：SAR・EATA・AWMC の 37.8／26.7）．更新が起きていない可能性があり，比較の意味が弱い
- ハイパーパラメータをテストで選んだ可能性がある（記述が無い）
- ICASSP 版と arXiv v1 の差は未確認

---

## 10. 9 本 × 本研究の軸 A1〜A11 の表

○ 扱っている，△ 部分的，× 扱っていない．各欄の根拠は 1 行．

| 論文 | A1 線形の頭 | A2 切片と向き・K の式 | A3 ソースの価値をラベル換算 | A4 合併の切片 | A5 K ≤ 10・振る・単位 | A6 コーパス間・ソース数 | A7 マルチモーダル・先生 | A8 弱いラベル・均衡 | A9 出発点への L2 | A10 評価 | A11 ずれと利得 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M4SER 2025 | × HuBERT・BERT を微調整 | × | △ K=5 で相手の K=20 を上回るという比較だけ | × | ○ クラスあたり 5・10・20・60 | △ 1 ソース，2 コーパス 1 組 | ○ 音声＋ASR 文の融合．先生→生徒なし | × | × 微調整の出発点のみ | △ UA は ME→IE だけ．話者分離・反復・検定なし．検証用ラベルでモデル選択 | × t-SNE のみ |
| MDAT 2024 | △ 凍結特徴の上に深層の頭 | × | × | × | △ 0・5・10・15．単位不明 | ○ 1 ソース，4 コーパス 12 組 | ○ 音声＋言葉の融合 | × | × | × 反復・検定・話者なし．表の値が食い違う | × |
| CTIAL 2024 | ○ 凍結 wav2vec2＋PCA＋LR | × | × | × | △ 初期 20 本＋200 本（合計） | △ コーパス間だが課題が違う．1 ソース，3 コーパス | × 音声のみ | △ BDA の擬似ラベルで特徴を整列 | × | ○寄り UAR，反復，Wilcoxon＋Holm，CV で調整．効果量なし | △ MMD を係数の選択にだけ |
| Feng & Chaspari 2023 | × 手作り特徴＋FNN/シャム | × | × | × | ○ 話者・感情あたり 1〜10 | △ 1 ソースずつ，5 コーパス | × 音声のみ | × | × | △ UAR・10 反復．検定なし．K 本とテストが同じ話者 | × |
| ADDoG 2021 | × 対数メルの CNN | × | △ 「ラベルを 2 倍にするのと同程度」の 1 文 | × | × 200〜3,200 発話（合計） | ○ 3 コーパス，2 ソース（MADDoG） | × 音声のみ | × | × | △ UAR・10 反復・ANOVA．固定の調整．話者分離なし | × Wasserstein は目的関数のみ |
| Chen & Yu 2023 | × Wav2Vec2 を微調整 | × | △ F1 の差で測る | × | ○ k=2〜64（合計，2 値） | ○ 4 ソースの全部分集合 | × 音声のみ | × | × | × Macro-F1・3 試行・検定なし・同じ話者 | × |
| Li et al. 2021 | × CNN＋GRU | × | × | × | △ 10%（固定） | × 1 コーパス内の条件間 | ○ 音声＋顔の融合 | × | × | △ UAR・5 反復の std．検定・話者なし | × |
| Ahn et al. 2025（抄録）／arXiv 2023 | × wav2vec2 を微調整 | × | × | × | × 5%（約 420 本）で振らない | ○ 1〜2 組，3 コーパス | ○ 音声＋言葉の判定段融合（抄録） | × | × | × UAR のみ．反復・検定なし | × |
| Dong et al. 2026 | △ 線形の頭だがエンコーダはソースで微調整 | × | × | × | × ラベル 0 | ○ 1 ソース，2 コーパス | × 音声のみ | △ T3A の確信度つき擬似ラベルで頭を作り直す | △ LAME は予測のソース忠実項 | × Acc・F1．反復・検定なし | △ ずれの種類と強さを言葉で論じる |

- 9 本とも A2・A4 は ×．A3 は ADDoG の 1 文と Chen & Yu の F1 差と M4SER のショット換算が △ で，ラベルの本数に換算した測定は無い
- A1 で ○ は CTIAL だけ（能動学習の論文で，ソースを混ぜる価値は測らない）
- A5 で ○（K ≤ 10 を振る）は M4SER（クラスあたり），Feng & Chaspari（話者・感情あたり），Chen & Yu（合計・2 値）の 3 本．MDAT は単位不明
- A7 で ○ は M4SER・MDAT（音声＋言葉），Li（音声＋顔，コーパス内），Ahn（音声＋言葉，抄録）．顔を含むコーパス間は無い．融合の先生から単モーダルの生徒へ使う設計は無い
- A10 で検定があるのは CTIAL（Wilcoxon＋Holm）と ADDoG（ANOVA）だけ．効果量と「実用値とオラクル値の区別」はどの論文にも無い．ターゲットで話者を分けているのは CTIAL のコーパス内（セッション）と Dong の個人化だけで，Feng・Chen & Yu・ADDoG は K 本とテストが同じ話者である

## 11. この群の先行研究に空いている穴

本研究の主張「MER において線形の頭に限って理論を立て，マルチモーダル性を活かして弱いラベルを作る」の置き場所を，示されたことと示されていないことに分けて書く．

**示されていること（先例があるので「初めて」と書けないこと）**
- 音声＋言葉の 2 モダリティで，少数のターゲットラベルを使うコーパス間（言語間）の手法は 2 本ある（M4SER：クラスあたり 5〜60，MDAT：0〜15 本・単位不明）
- ターゲットのラベル数を振り，少ないときはソースが効き，多いときはターゲットだけが並ぶか勝つ，という形はある（ADDoG：200〜3,200 発話・感情価 3 区分，Chen & Yu：2〜64 本・2 値・話者ごと）
- ソースの組み合わせの価値を総当たりで測った例はある（Chen & Yu．F1 の差で）
- 凍結した自己教師あり特徴＋ロジスティック回帰をコーパス間の少数ラベルの設定で使った例はある（CTIAL．能動学習）
- 少数ラベルの微調整で負の転移が起きる報告はある（Feng & Chaspari：話者・感情あたり 1 本の FNN 微調整）
- ラベル無しの適応では「分類器の頭だけを作り直す」のが最も有望で，擬似ラベルで全体を更新すると失敗するという比較はある（Dong）

**示されていないこと（本研究の差として書けること）**
1. **理論**：9 本とも，ターゲットの K 本のラベルが直す誤差を切片（クラスごとの閾値）と向き（重み）に分けていない．K 本で減る誤差の式も，合併で切片が直る割合も，暗黙のドメイン別切片も無い（A2・A4 が全て ×）
2. **ソースの価値の単位**：ソースの価値を「ターゲットのラベル何本ぶんか」で測った論文は無い．最も近いのは ADDoG の「ラベルを 2 倍にするのと同程度」という 1 文と，Chen & Yu の F1 の差（A3）
3. **K の領域と組の数**：クラスあたり K ≤ 10 を振り，かつ多数のソース・ターゲットの組で測った論文は無い．M4SER は 1 組，Feng は 4 ソース × 1 ターゲット（混ぜない），Chen & Yu は 1 ターゲット（話者ごと）．K の単位を明記しない論文（MDAT）もある
4. **モダリティ**：顔を含む 3 モダリティで少数ラベルのコーパス間を扱った論文は無い．音声＋顔は Li（コーパス内）だけ．融合の先生の出力を単モーダルの生徒の弱いラベルにする設計はどの論文にも無い（A7）
5. **弱いラベルの作り方**：ソースだけで学んだ分類器の出力を，ターゲットのラベル無し全体の弱いラベルとしてクラス均衡で使い，少数の本当のラベルと混ぜる手は無い．近いのは CTIAL の BDA（整列のため）と Dong の T3A（ラベル無し，頭のプロトタイプ）で，どちらもターゲットのラベルと組み合わせない（A8）
6. **出発点への罰則**：ソースで学んだ分類器を出発点にして L2 罰則で少数ラベルから学ぶ手は無い．9 本の「微調整」は出発点にするだけで罰則が無い（A9）
7. **評価**：話者分離・UAR・相手の同じ手順の調整・対応のある検定と補正・効果量・実用値とオラクル値の区別を揃えた論文は無い．検定は 2 本（CTIAL，ADDoG）だけで，効果量とオラクル値の区別は 0 本．M4SER はモデル選択にターゲットの検証用ラベルを使い，K 本だけの実用値ではない（A10）
8. **ずれと利得**：ずれの量（距離や平均のずれ）を測って利得と結びつけ，得が生じる条件と消える条件を予言した論文は無い．CTIAL は MMD を係数の選択に使うだけ，Dong はずれの種類と強さを言葉で論じるだけ（A11）

**関連研究の節での書き方の方針**
- 「マルチモーダルで少数ラベルのコーパス間は初めて」とは書かない（M4SER・MDAT がある）
- 書けるのは次の組み合わせの新しさである：(a) 凍結したエンコーダの上の線形の頭に限ることで，(b) K 本のラベルが切片と向きのどちらを直すかを式で分け，(c) ソースの価値をラベルの本数で表し，(d) 融合の先生の出力を単モーダルの生徒の弱いラベルにし，(e) 顔を含む 3 モダリティ・7 群の多数の組で K を振り，(f) ずれの指標で得が生じる条件と消える条件を予言する
- 先例を悲観も楽観もせずに置く：ADDoG の「いつ一般化し，いつ特化するかの定式化」と Feng & Chaspari の「複数ソースの組み合わせとソースの性質の効き目」は，著者自身が今後の課題に挙げたものであり，本研究はそれに答える位置にある

## 12. 余力：VoxEmo（arXiv 2603.08936）の表と本文の食い違いの確認

### Takeaway
前回の調べの「本文の＋17.7 と表 8 の値（＋17.1・＋22.2・＋11.2）の平均 16.8 が食い違う」は，**前回の読み違い**だった．表 8 の MELD をソースにした Qwen2-Audio の利得は CREMA-D ＋17.1，IEMOCAP ＋22.2，MSP-Podcast ＋13.7 で，平均は 17.67 ≈ 17.7 と本文に一致する．＋11.2 は MELD→MELD（コーパス内の微調整，対角）の値である．

### Cited Findings
- 書誌：H. Zhang, H.-C. Chou, S. Narayanan, T. Hain, "VoxEmo: Benchmarking Speech Emotion Recognition with Speech LLMs," arXiv 2603.08936v1（2026-03-09）— [arXiv 2603.08936](https://arxiv.org/abs/2603.08936)．Interspeech 2026 への投稿かは未確認（◎ 本文 4.4 節と表 8 を読んだ．表は `pdftotext` の列の順が乱れたので，本文の記述（MELD→IEMOCAP 81.6，MELD→CREMA-D 86.6，IEMOCAP→MELD −3.4）と照らして列を復元した）
- 表 8（Macro-F1．行がソース，列がターゲット．括弧はゼロショットの最良に対する差）— [arXiv 2603.08936](https://arxiv.org/pdf/2603.08936)：
  - Qwen2-Audio：ターゲット CREMA-D：CREMA-D 91.1 (+21.7)，IEMOCAP 76.8 (+7.4)，MELD 86.6 (+17.1)，MSP-Podcast 78.6 (+9.1)；ターゲット IEMOCAP：79.9 (+20.5)，82.4 (+23.0)，81.6 (+22.2)，68.1 (+8.7)；ターゲット MELD：30.7 (+2.6)，24.7 (−3.4)，39.3 (+11.2)，30.8 (+2.8)；ターゲット MSP-Podcast：25.8 (+5.5)，31.5 (+11.1)，34.1 (+13.7)，29.2 (+8.8)
  - AudioFlamingo-3：ターゲット CREMA-D：75.2 (+15.4)，63.6 (+3.8)，63.3 (+3.5)，36.6 (−23.2)；ターゲット IEMOCAP：65.1 (+1.1)，65.2 (+1.2)，62.0 (−2.0)，45.4 (−18.6)；ターゲット MELD：21.6 (−8.2)，20.3 (−9.5)，25.7 (−4.1)，18.9 (−10.9)；ターゲット MSP-Podcast：22.9 (−5.7)，19.2 (−9.5)，23.5 (−5.2)，19.1 (−9.6)
- 本文（原文）："For Q2A, fine-tuning on a mismatched English source exceeds the best zero-shot prompt on 11 of the 12 source–target pairs. MELD is the most effective source, with a mean Macro-F1 gain of 17.7 over zero-shot across three targets. On IEMOCAP, it reaches 81.6, and on CREMA-D, 86.6." "the sole degradation in the table occurs when transferring IEMOCAP→MELD (−3.4)" "Under the same LoRA configuration, AF3 exceeds zero-shot on only 3 of 12 pairs, all targeting acted corpora with modest gains." "MSP-Podcast is a particularly harmful source, accounting for the three largest drops in the table."
- 照合：Q2A の MELD をソースにした 3 つの非対角の差は ＋17.1・＋22.2・＋13.7，平均 17.67 で本文の 17.7 と一致．Q2A の非対角 12 組で負は IEMOCAP→MELD の −3.4 だけで「11 of 12」と一致．AF3 の非対角で正は IEMOCAP→CREMA-D ＋3.8，MELD→CREMA-D ＋3.5，CREMA-D→IEMOCAP ＋1.1 の 3 組で一致．AF3 の下げ幅の大きい 3 つは MSP-Podcast をソースにした −23.2・−18.6・−10.9 で一致
- 限界（原文）："this work focuses on corpus-level aggregate metrics and does not examine within-dataset factors such as class imbalance, per-class performance, or speaker effects."

### Inferences
- VoxEmo の表と本文に食い違いは無い．前回のノート（問い 5）の「+17.7 と約 16.8 で食い違う」の記述は直す必要がある
- VoxEmo はターゲットのラベルを使わない（ソースで微調整してターゲットで評価）ので，本研究の K を振る設定とは別である．ソースの価値がモデルに依存する（Q2A では MELD が最良，AF3 では MSP-Podcast が害）という点は，2 つのエンコーダで再現を確かめる理由づけに使える

### Gaps
- 表 8 の列の復元は本文との照合に基づく推定である．掲載版が出たら表を直接確かめる
