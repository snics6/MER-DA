<!-- 2026-11-17：調べの担当（general-purpose）はファイルを書けなかったので，最後の返答の全文を本体が保存した． -->

# 先行研究：Ryumina ほか 2025 の新しさと，蒸留と DA 手法の比べ（2026-11-17）

研究者の問い（研究者の言葉）：「Ryumina も出版時点で彼女の方法が本当に新しいかをまず調べてほしい．その上で彼女たちは新規性を主張しているよね．それは彼女たちの限定された書き方では OK なんだと思う．我々も，結果を見ると蒸留で他の方法よりも良い結果でてるよね，概ね．それを主張するのはあり？」

この報告は材料と範囲の候補だけを示す．主張してよいかの最終の判断は本体のセッションが行う．新しい手法や実験の案は書かない．

確認の水準：◎＝原文（PDF か arXiv の本文）を読んだ，○＝抄録か書誌だけを読んだ，△＝ほかの論文の中の紹介（二次資料）だけ．英語の引用は原文のままである．PDF の行末のハイフンで切れた語（"context-independent" など）は，つないで書いた．

---

## 0. 要点

1. Ryumina ほか 2025 の「初めて」は，要旨の 1 文だけにある．"multi-corpus"，"multimodal"，"evaluated through a leave-one-corpus-out protocol" の 3 つの語を同時に満たすものに付けた「初めて」である．LOCO は，1 つのコーパスを抜いて残りで学び，抜いたコーパスで評価する手順である．今回の検索では，3 つのモダリティの感情認識を LOCO で評価した論文で，2025 年 2 月より前のものは見つからなかった．この狭い読み方なら，主張は先行研究と食い違わない．
2. 一方で，「多コーパスの MER の研究は少ない」「コーパス間の評価は顧みられていない」という背景の文と，融合の部品の新しさの文については，引かれていない先行研究がある．著者ら自身の 2023 年の論文（多コーパスの音声・映像の研究で，cross-modal gated self-attention fusion を提案）と 2024 年の論文，Avots ほか 2018（3 コーパスで学んで AFEW で評価），UniMSE 2022（MOSI・MOSEI・MELD・IEMOCAP の学習集合をまとめて学習），MERBench 2024（3 モダリティのコーパス間の評価），Amanda 2024（IEMOCAP・MELD・MOSEI・MSP-IMPROV の間の多モダリティの教師なし DA）である．どれも参考文献の 31 本に入っていない．
3. 私たちの比べ（ほかのモダリティを含む疑似ラベルの蒸留と，ほかのモダリティを使わない SSDA・UDA・テスト時適応の手法を，コーパス間の感情認識で並べる比べ）と同じ比べをした研究は，今回の検索では見つからなかった．同じ形の比べは，3 次元の意味分割の xMUDA（2020）と xMoSSDA（2022）にある．声の感情認識では Zhang ほか 2021 が，顔の疑似ラベルと声の半教師あり学習 3 手法を比べている．ただしターゲット（IEMOCAP）のラベルは全部使い，DA の手法とは比べていない．
4. 既存の手法どうしの比べを貢献にした論文は，次のもので範囲を付けている．比べた手法の数，データセットの数，設定の名前，「when carefully implemented」「in a realistic cross-domain evaluation setting」「often」「in 8 out of 9 cases」のような語である（4 節に原文 6 本）．
5. 私たちの主張は，次の範囲を付ければ，今回見た先行研究と食い違わない（5 節）．「コーパス間（4 コーパスの LOCO）」「感情ごとに K = 5，10，25 本のラベル」「凍結したエンコーダの上の線形分類器」「声の分類器」「名前を挙げた 11 手法」「UAR の点推定で 132 中 132，区間で 122」．手法そのものの新しさと，「クロスモーダルの手法と単一モダリティの DA 手法の比べは初めて」は書けない．

---

## 1. Ryumina ほか 2025 が新しいと主張したこと

書誌：E. Ryumina, D. Ryumin, A. Axyonov, D. Ivanko, A. Karpov. Multi-corpus emotion recognition method based on cross-modal gated attention fusion. Pattern Recognition Letters 190 (2025) 192–200. https://doi.org/10.1016/j.patrec.2025.02.024（◎．手元の PDF の本文，表 1〜7，参考文献 31 本を読んだ）

### 1.1 日付（PDF の 1 頁目）

"Received 4 July 2024; Received in revised form 31 January 2025; Accepted 17 February 2025; Available online 27 February 2025"

巻号の出版は 2025 年 4 月である（Crossref の published-print）．投稿は 2024 年 7 月 4 日なので，「出版時点で新しいか」は 2 つの時点で見る必要がある．投稿した 2024 年 7 月と，公開された 2025 年 2 月である．

### 1.2 要旨の主張（原文）

| 原文 | 範囲を絞った語 |
|---|---|
| "We propose the first multi-corpus multimodal emotion recognition method with high generalizability evaluated through a leave-one-corpus-out protocol." | "first" は名詞句全体に付く．範囲は "multi-corpus"，"multimodal"，"with high generalizability"，"evaluated through a leave-one-corpus-out protocol" |
| "The proposed method achieves the state-of-the-art results on these corpora and establishes the first baseline for multi-corpus studies." | 最良の成績は "on these corpora"（4 コーパス）に限る．"first baseline" の範囲は "for multi-corpus studies" だけで，コーパスの名は付けていない |
| "We demonstrate that due to the MELD rich emotional expressiveness across three modalities, the models trained on it exhibit the best generalization ability when applied to other corpora used." | "other corpora used"（使った 3 コーパス） |
| "We also reveal that the AFEW annotation better correlates with the annotations of MOSEI, MELD, and IEMOCAP, as well as shows the best cross-corpus performance as it is consistent with the widely-accepted theories of basic emotions." | 比べる相手を 3 コーパスの名で限る．理由づけ（基本感情の理論に合う）は測っていない |

### 1.3 序論の背景の文と貢献の箇条（原文）

背景の文："Despite the abundance of multi-corpus unimodal ER methods [6,7], multi-corpus MER studies are scarce."
序論では "scarce"（少ない）と書き，「無い」とは書いていない．[6] は Kaya & Karpov 2018 の音声のコーパス間の研究，[7] は Latif ほか 2023 の音声のコーパス間・言語間の DA である．

貢献の箇条（4 つ）：

1. "We present a novel method for MER based on the context-independent cross-modal gated fusion. Context independence is achieved through calculating feature statistics and using gated attention to adapt modality informativeness, resulting in ER independent of the recording duration."
   - 範囲：新しさを "context-independent" という性質に付けている．区切りごとの特徴の平均と標準偏差を使い，録音の長さに依らない性質である．gated fusion そのものを新しいとは書いていない．
2. "We perform a large-scale multi-corpus research for MER, enabling the training of models applicable across diverse domains, rather than being limited to a single domain as existing SOTA models."
   - 範囲：比べる相手を "existing SOTA models"（表 1 の 12 手法）に置いている．
3. "We analyze the ability of encoders trained on four benchmark corpora to generalize to other corpora using a leave-one-corpus-out (LOCO) protocol. This analysis highlights the generalizability of these models when applied to different corpora."
   - 範囲："four benchmark corpora"，"LOCO"．
4. "Our method outperforms the SOTA results on all four corpora, highlighting the effectiveness of the proposed method and its components in a single domain research. Additionally, we make the source code publicly available."
   - 範囲："on all four corpora"，"in a single domain research"（1 つのコーパスで学んで同じコーパスで評価する設定）．

### 1.4 関連研究の節の「足りない点」の文（原文）

- 節の冒頭で対象を限っている："This section summarizes the SOTA methods that are developed based on the corpora used in this research and can be compared with our method."
- 足りない点："Moreover, while these methods are evaluated on multiple corpora, cross-corpus evaluation is neglected, limiting their applicability to new data."
- 範囲："these methods" は表 1 の 12 手法を指す．2019〜2024 年に研究の 4 コーパスで評価された手法である．MER の研究全体についての文ではない．

### 1.5 融合の部品の位置づけ（3.4 節の原文）

- "A similar fusion technique was used in bimodal ER [29] and other tasks [30,31]."（[29] は Liu, Li, Meng 2020 の Group gated fusion．Interspeech 2020）
- "The authors of [18] also employ a gated attention mechanism; however, they do not incorporate the step of regulating the feature importance of various modalities."（[18] は Ma ほか 2023．IEEE TMM）
- 読み：似た融合の先例を 2 本引き，[18] との違いを 1 文で書いている．

### 1.6 先行研究との比べの文（4.5 節の原文）

- "A comparison of our single-corpus MER method with other similar methods using the same experimental setup (same data splitting into subsets and single-corpus training protocol) is presented in Table 7."
- "Our method outperforms SOTA on all of the research corpora. Moreover, the performance gains and generalizability of our methods are statistically significant, and STD values does not exceed 1.1%."
- 範囲："using the same experimental setup" と書き，括弧の中で「同じ分割」「単一コーパスの学習」と条件を示している．有意性は，各テスト集合をクラスの割合を保って 5 つに分けたときの平均と標準偏差から出している．

### 1.7 結論の原文

- "In this paper, we proposed the multi-corpus MER method designed to exhibit high generalizability to new unseen data."
- "The method has been evaluated on IEMOCAP, MOSEI, MELD, and AFEW in both single- and multi-corpus setups. The new baselines have been obtained on all four corpora."
- 限界："Despite the results obtained, our method, developed through multi-corpus learning and cross-corpus testing, does not reach the performance levels of models trained on a single corpus."，"whereas in our study we limited ourselves to single-corpus training of the encoders."
- 注：著者の公開した頁（https://smil-spcras.github.io/MER/）の結論は "we first propose the multi-corpus MER method" と書いており，PRL の版（"we proposed"）と 1 語違う（WebFetch の要約で読んだ．○）．論文で引くのは PRL の版である．

### 1.8 範囲の付け方のまとめ

- 「初めて」は要旨の 1 文に限られ，多コーパス，多モダリティ，LOCO での評価の 3 語を全部満たすものに付いている．
- 最良の成績は「この 4 コーパス」と「同じ分割と単一コーパスの学習」に限っている．
- 背景の文は「少ない」（scarce）と書き，「無い」とは書いていない．
- 「足りない点」は表 1 の 12 手法についての文として書いている．
- 融合の部品は，新しさを「文脈に依らない」性質に付け，似た融合を 2 本引いている．
- 範囲が弱いのは "establishes the first baseline for multi-corpus studies" の 1 か所である．コーパスも手順も書いていない．

---

## 2. 出版時点で本当に新しかったか

### 2.1 2025 年 2 月より前の近い研究と，Ryumina ほかが引いているか

参考文献の 31 本を 1 本ずつ見た（手元の PDF）．下の表の「引用」の列は，その 31 本に入っているかを示す．

| 研究（公開日） | 何をしたか（原文の要点） | Ryumina 2025 と重なる点 | 引用 | 水準 |
|---|---|---|---|---|
| Avots, Sapiński, Bachmann, Kamińska. Audiovisual emotion recognition in wild. Machine Vision and Applications 30:975–985（オンライン 2018-07-19） | "A cross-corpus evaluation is done using three different databases as the training set (SAVEE, eNTERFACE'05 and RML) and AFEW (database simulating real-world conditions) as a testing set."．音声と顔を，判断の段で融合する | 多コーパスで学び，学習に使わない AFEW で評価した音声・映像の感情認識．分け方は 1 通りだけで，LOCO のすべての分割は回していない | 無し | ○（抄録） |
| Ryumina, Markitantov, Karpov. Multi-Corpus Learning for Audio–Visual Emotions and Sentiment Recognition. Mathematics 11(16):3519（2023-08-15 公開） | 要旨 "we propose a multi-corpus learned audio–visual approach for emotion and sentiment recognition"．貢献 "We propose a novel Cross-Modal Gated Self-Attention Fusion (CMGSAF) strategy at the feature level"．RAMAS と CMU-MOSEI | 同じ第 1 著者による，多コーパスの音声・映像の感情認識と cross-modal gated の融合．評価は各コーパスの中で行い，コーパス間の評価はしていない | 無し | ◎（MDPI の PDF） |
| Ryumina, Markitantov, Ryumin ほか. Audio-Visual Compound Expression Recognition Method based on Late Modality Fusion and Rule-based Decision. 6th ABAW の論文，arXiv 2403.12687（2024-03-19） | 貢献 "We present a method for audio-visual emotion recognition based on both multi-corpus and cross-corpus research."．要旨 "The method is evaluated in multi-corpus training and cross-corpus validation setups."．音声のモデルを AffWild2 と MELD で学び，AFEW の検証用の部分を追加の検証コーパスに使う | 同じ第 1 著者による，多コーパスで学んで別コーパスで確かめる音声・映像の感情認識．AFEW と MELD を使う | 無し | ◎（arXiv の本文） |
| Hu ほか. UniMSE. EMNLP 2022（arXiv 2211.11256，2022-11-21） | "We integrate the training sets of MOSI, MOSEI, MELD, IEMOCAP to train the model and valid sets to select hyperparameters."．映像の特徴の抽出器は "pre-trained (supervised) on VGGface and AFEW dataset" | 4 つのデータセットをまとめて学ぶ 3 モダリティのモデル（多コーパスの学習）．MOSEI・MELD・IEMOCAP が重なる．評価は各データセットの中 | 無し | ◎（arXiv の本文） |
| Lian ほか. MERBench. arXiv 2401.03429（v1 2024-01-07，v3 2024-04-21．`docs/related-work.md` の記録では掲載は TPAMI 48(5), 2026） | 5.4 節 "Cross-corpus Benchmark"．"Given that IEMOCAP(four), MELD, and MER-MULTI share four discrete labels (i.e., happiness, sadness, anger, neutral), cross-corpus experiments are conducted based on these labels."．同じ節の結果 "multimodal fusion can improve both within- and cross-corpus results. However, its utility is relatively lower in the cross-corpus setting." | 3 モダリティのコーパス間の評価．IEMOCAP と MELD が重なり，4 感情も同じ．1 コーパスで学んで別コーパスで評価する形で，LOCO ではない | 無し | ◎（arXiv v3 の本文） |
| Yin, Huang, Wu, Soleymani. Speaker-Invariant Adversarial Domain Adaptation for Emotion Recognition. ICMI 2020（2020-10-21） | 抄録 "Our experiments with multimodal data (speech, vision, and text) and the cross-domain evaluation indicate that the proposed SIDANN outperforms ... the DANN model"．MSP-Improv と IEMOCAP の間で，覚醒度と感情価をそれぞれ 2 値で判定する | 3 モダリティのコーパス間の評価（DA）．感情の 4 分類ではない | 無し | ○（抄録） |
| Zhang, Sun, Hong, Li. Amanda: Adaptively Modality-Balanced Domain Adaptation for Multimodal Emotion Recognition. Findings of ACL 2024（会議は 2024 年 8 月．Ryumina の投稿の後，改訂の前） | "four widely used benchmark multimodal emotion recognition datasets, IEMOCAP ..., MELD ..., CMU-MOSEI ..., and MSP-IMPROV"．"we select samples in the four classes— neutral, happy, sad and angry"．コーパスの対ごとの教師なし DA | 3 モダリティ，4 感情のコーパス間の評価で，IEMOCAP・MELD・MOSEI の 3 つが重なる．ラベル無しのターゲットを使う DA で，LOCO ではない | 無し | ◎（ACL Anthology の PDF） |
| Li, Kim, Kuo, Narayanan. Acted vs. Improvised: Domain Adaptation for Elicitation Approaches in Audio-Visual Emotion Recognition. Interspeech 2021（arXiv 2104.01978） | "explore domain transfer learning techniques on emotional utterances collected under different emotion elicitation approaches, particularly with limited labeled target samples" | 音声・映像の感情認識のドメイン間の転移．ずれはコーパス間ではなく，演じ方の違いである | 無し | ○（抄録．`docs/related-work.md` には本文を読んだ記録がある） |
| Ananthram, Saravanakumar, Huynh, Beigi. Multi-Modal Emotion Detection with Transfer Learning. arXiv 2011.07065（2020-11-13） | 声と言葉．"Our best variant, trained on only VoxCeleb and Crema-D and evaluated on IEMOCAP, achieves an EER of 38.05%." | 2 モダリティで，学習に使わないコーパスで評価する | 無し | ○（抄録） |
| Arevalo ほか. Gated Multimodal Units for Information Fusion. arXiv 1702.01992（2017-02-07．掲載先は確かめていない） | "The GMU learns to decide how modalities influence the activation of the unit using multiplicative gates." | Ryumina の式（1）H^{A+V} = H^{F1} ⊗ H^{F} ⊕ H^{F2} ⊗ (1 − H^{F}) は，ゲートの重み z と 1 − z で 2 つのモダリティの隠れ状態を足す GMU と同じ形である（こちらの読み） | 無し | ○（抄録．GMU の本文の式は読んでいない） |
| Jiang, Ji. CMGA: Cross-Modality Gated Attention Fusion for Multimodal Sentiment Analysis. arXiv 2208.11893（2022-08-25） | "we propose CMGA, a Cross-Modality Gated Attention fusion model for MSA that tends to make adequate interaction across different modality pairs. CMGA also adds a forget gate to filter the noisy and redundant signals" | モダリティの対ごとのゲートつき注意の融合．Ryumina は A+V，V+T，T+A の対ごとにゲートの層を通しており，構成が近い．課題は感情極性（MOSI，MOSEI） | 無し | ○（抄録） |
| Kumar, Vepa. Gated Mechanism for Attention Based Multimodal Sentiment Analysis. arXiv 2003.01043（掲載先は確かめていない） | "Cross modal interaction learning" と "Fusion of unimodal and cross modal cues" を扱う | ゲートつきのクロスモーダルの注意 | 無し | ○（抄録） |
| Sangwan ほか 2019（multi-task Gated Contextual Cross-Modal Attention，GCCMA） | Mathematics 2023 の 2.2 節の紹介："presented the multi-task Gated Contextual Cross-Modal Attention (GCCMA) approach for sentiment and multi-label emotion recognition together" | ゲートつきのクロスモーダルの注意による感情の認識 | 無し（Mathematics 2023 では引いている） | △（二次資料だけ．原論文は読んでいない） |
| Liu, Li, Meng. Group gated fusion on attention-based bidirectional alignment for multimodal emotion recognition. Interspeech 2020 | — | 2 モダリティのゲートつき融合 | **あり**（[29]．"A similar fusion technique was used in bimodal ER [29]"） | 引用の文だけ確かめた |
| Ma ほか 2023（transformer-based model with self-distillation．IEEE TMM） | — | ゲートつきの注意 | **あり**（[18]．"they do not incorporate the step of regulating the feature importance of various modalities"） | 引用の文だけ確かめた |
| Ryumina, Dresvyanskiy, Karpov. In search of a robust facial expressions recognition model: A large-scale visual cross-corpus study. Neurocomputing 514 (2022) | 顔だけのコーパス間の研究．LOCO を使うことは検索の要約で知っただけ | 顔だけの LOCO | **あり**（[25]．顔のエンコーダの出典として引いている） | 引用はあることを確かめた．LOCO の使用は △ |

### 2.2 要素ごとの判定

| Ryumina 2025 の要素 | 2025 年 2 月より前の先例 | 判定 |
|---|---|---|
| 多コーパスで学ぶ多モダリティの感情認識 | UniMSE 2022（3 モダリティ，4 データセットをまとめて学習），Ryumina 2023 Mathematics（音声・映像），Avots 2018（音声・映像） | 先例がある．背景の文 "multi-corpus MER studies are scarce" は「少ない」なので先例と食い違わない．ただし著者ら自身の 2023 年と 2024 年の研究を引いていない |
| 多モダリティのコーパス間の評価 | Avots 2018（→ AFEW），Yin 2020（3 モダリティ，MSP-Improv ↔ IEMOCAP），MERBench 2024（3 モダリティ，IEMOCAP・MELD・MER-MULTI，4 感情），Amanda 2024（3 モダリティ，4 コーパス，4 感情），Ryumina 2024 ABAW（音声・映像） | 先例がある．"cross-corpus evaluation is neglected" は，表 1 の 12 手法についての文として読めば正しい．MER の研究全体についての文として読むと先例と食い違う |
| 3 モダリティの MER を LOCO で評価する | 今回の検索では見つからなかった（LOCO は顔だけなら Ryumina 2022（△），音声だけなら Kaya & Karpov 2018 などにある） | この狭い読み方なら，要旨の「初めて」は先例と食い違わない．多コーパスで学んで別コーパスで確かめた音声・映像の研究（Avots 2018，Ryumina 2024 ABAW）は LOCO のすべての分割を回していないので，この「初めて」を否定しない |
| 同じ 4 コーパス（IEMOCAP，MOSEI，MELD，AFEW）の組 | 4 つ全部の組は見つからなかった．そのうち 3 つ（IEMOCAP，MELD，MOSEI）は Amanda 2024 と UniMSE 2022 が使っている | 4 つの組の先例は見つけていない |
| cross-modal gated attention の融合 | GMU 2017，Kumar & Vepa 2020，Liu 2020，CMGA 2022，Ma 2023，Ryumina 2023 の CMGSAF | ゲートつきのクロスモーダルの融合そのものは新しくない．著者は新しさを "context-independent"（特徴の統計量を使い，録音の長さに依らない）に付け，似た融合を 2 本（[29]，[18]）引いているので，貢献 1 の文は先例と食い違わない．ただし，最も近い著者ら自身の CMGSAF と，モダリティの対ごとにゲートをかける CMGA は引いていない |
| 「the first baseline for multi-corpus studies」 | MERBench 2024 がコーパス間の基準値を出している（IEMOCAP と MELD が重なる） | この文には範囲の語が無いので，MERBench と重なる．「この 4 コーパスの LOCO で最初の基準値」と読めば先例と食い違わない |

### 2.3 研究者の見立て「限定された書き方では OK」への材料

- 要旨の「初めて」は 3 つの語で範囲を絞っている．今回の検索の範囲では，その範囲で先にした研究は見つからなかった．この 1 文については，研究者の見立てと食い違う材料は無い．
- 背景と関連研究の文は「少ない」「表 1 の手法は」と範囲を付けているので，文としては誤りではない．ただし，多コーパスの音声・映像の研究（著者ら自身の 2 本と Avots 2018）と，3 モダリティのコーパス間の研究（MERBench，Yin 2020，Amanda 2024）を引いていない．これらを引いていれば，位置づけはより正確になった．
- これらの先行研究を引かないまま，論文は PRL の査読を通った．査読でこれらが指摘されたかどうかは，公開の資料からは確かめられなかった．

---

## 3. 私たちの比べと同じ比べをした先行研究はあるか

比べの中身：少数ラベル（感情ごとに K = 5，10，25 本）のコーパス間の MER（4 コーパスの LOCO）で，2 種類の手法を声の分類器と AVT の分類器で比べた．1 つは，ほかのモダリティを含む分類器の確率の積を疑似ラベルにして学び直す蒸留である（Albanie ほか 2018 の型で，温度は使わない）．もう 1 つは，疑似ラベルにほかのモダリティを使わない 11 手法である（自己学習，エントロピー最小化，IM，T3A，ラベル伝播，自分の予測を疑似ラベル，S+T，CORAL，DANN，MME，xMUDA 型）．

### 3.1 結論

今回の検索と手元のノートの範囲では，この比べを先にした研究は見つからなかった．つまり，クロスモーダルの蒸留・疑似ラベルと，同じモダリティの SSDA・UDA・テスト時適応の手法を，音声・顔・言葉のコーパス間の感情認識で並べた研究は無かった．最も近いものは次の 3 群である．

- 同じ形の比べを別の課題でした研究：xMUDA（CVPR 2020）と xMoSSDA（TPAMI 2022）．課題は 3 次元の意味分割である．
- 声の感情認識で，クロスモーダルの疑似ラベルと同じモダリティの半教師あり学習を比べた研究：Zhang ほか 2021．DA の手法とは比べておらず，ターゲットのラベルは全部使っている．
- 感情認識で，多モダリティの DA を DA 手法と比べた研究：Amanda 2024（コーパス間．クロスモーダルの蒸留は比べていない），CMJDA 2025・UF-AMA 2026（脳波と視線，被験者間），MuSACo 2025（被験者ごとの適応）．

### 3.2 1 本ずつ

**Albanie, Nagrani, Vedaldi, Zisserman. Emotion Recognition in Speech using Cross-Modal Transfer in the Wild. ACM MM 2018（arXiv 1808.05561）**（◎．手元の `docs/参照/DAI_paper/参照論文/要点_問い1_別モダリティ蒸留.md` 3 節）
- 顔の教師（FERPlus で学んだもの）から声の生徒へ，ラベル無しの VoxCeleb の上で温度 T = 2 で蒸留した．評価は RML と eNTERFACE で，生徒の 8 次元の出力の上にアフィン写像と softmax を学ばせ，10 分割の交差検証で測った．
- DA の手法や半教師あり学習の手法とは比べていない．比べたのは，教師（顔）の成績と，手作りの音声の特徴を使った先行研究（Noroozi ほか）の値である．

**Zhang, Chen, Chen, Li, Wu, Li, Zhu. Combining cross-modal knowledge transfer and semi-supervised learning for speech emotion recognition. Knowledge-Based Systems 229:107340, 2021**（◎．著者の公開 PDF を本調べで読み直した）
- ターゲットは IEMOCAP（4 感情．8 話者で学び，1 話者で検証，1 話者で評価）と CH-SIMS である．ターゲットのラベルは全部使い，ラベル無しのデータを別のコーパス（EmoVoxCeleb，iQIYI-VID）から足す．
- 比べた手法は，同じモダリティの半教師あり学習 3 つ（FixMatch，π-model，Temporal ensembling）と，Albanie の手をそのまま当てた DCMKT とその微調整である．DA の手法は含まれていない．
- 表 5（IEMOCAP，UA）：ラベルだけ 59.43，FixMatch 57.71，π-model 61.57，Temporal ensembling 58.46，DCMKT 27.99，DCMKT の微調整 54.97，顔と声を混ぜた疑似ラベルの Consistent and random（ε = 0.5）61.20，Weighted fusion（α = 0.2）62.50．
- 著者の文は "We can see that our method outperforms all of the baseline methods." である．ただし UA では，ε = 0.5 の 61.20 が π-model の 61.57 を下回る（F1 と WA では上回る）．著者の「すべてを上回る」は，Weighted fusion については正しく，ε = 0.5 については F1 と WA でだけ正しい．
- 著者は DCMKT の低さの理由を "This is mainly because of the distributional difference between the IEMOCAP and EmoVoxCeleb datasets and the category imbalance problem of the EmoVoxCeleb dataset." と書いた．
- 私たちの比べとの違いは 3 つある．ターゲットのラベルが全部あるので，コーパス間の DA の設定ではない．比べた相手に DA の手法が無い．疑似ラベルの元は，顔だけか，顔と声の混合である．

**Jaritz, Vu, de Charette, Wirbel, Pérez. xMUDA: Cross-Modal Unsupervised Domain Adaptation for 3D Semantic Segmentation. CVPR 2020（arXiv 1911.12676）**（◎．arXiv の本文）
- 比べ方："We evaluate our method on the 3 proposed cross-modal UDA scenarios and compare against uni-modal UDA methods: Deep logCORAL [21], entropy minimization (MinEnt) [29] and pseudo-labeling (PL) [17]."，"The uni-modal UDA baselines [17, 21, 29] are applied separately on each modality."
- 結果の文："xMUDA – using the cross-modal loss but not PL – brings a significant adaptation effect on all 3 UDA scenarios compared to 'Baseline (source only)' and often outperforms the uni-modal UDA baselines. We observe, that xMUDA consistently improves both modalities (2D and 3D), i.e. even the strong modality can learn from the weaker one."
- 比べの形は私たちと同じである．クロスモーダルの学習と，各モダリティに別々に当てた単一モダリティの UDA 手法を，モダリティごとに比べている．課題は 3 次元の意味分割で，感情ではない．

**Jaritz ほか. Cross-modal Learning for Domain Adaptation in 3D Semantic Segmentation（xMoSSDA を含む）. TPAMI 2022（arXiv 2101.07253）**（◎．arXiv の本文）
- SSDA の比べ："We compare xMoSSDA against eight baselines."．単一モダリティの SSDA の相手は "three SSDA baselines (trained on S + Tℓ + Tu) adapted from uni-modal UDA baselines [5], [7], [20]" で，Deep logCORAL，MinEnt，PL にあたる．
- 結果の文："Finally, xMoSSDAPL outperforms all baselines by up to +2.2 in 8 out of 9 cases."
- 私たちの声と AVT の違いに近い観察："We also observe that xMoSSDA improves 2D and 3D more (max. 2.0) than ensemble result 2D+3D (max. 0.4), w.r.t. to Baseline (S + Tℓ)."．単一モダリティの分類器の得は大きく，2 つを合わせた予測の得は小さかった．
- 付録 E の見出しは "SSDA EXAMPLE: NO GAIN FROM SOURCE DATA" である．ラベル付きのターゲットがソースより多いと，ソースを混ぜても得が無かった．

**Zhang, Sun, Hong, Li. Amanda. Findings of ACL 2024**（◎．ACL Anthology の PDF）
- コーパス間の多モダリティの教師なし DA である（IEMOCAP，MELD，CMU-MOSEI，MSP-IMPROV の対．4 感情．重みつき F1）．
- 比べた相手："We compare our model, Amanda, with DANN (Ganin et al., 2016), CDAN (Long et al., 2018), MADA (Pei et al., 2018), DALN (Chen et al., 2022), and CDAN-E"．単一モダリティ用の手法は "we directly conduct domain adaptation for each modality" として当てた．
- クロスモーダルの蒸留や疑似ラベルの手法とは比べていない．評価は融合の分類器だけで，単一モダリティの分類器は評価していない．ターゲットのラベルは使わない．

**Dong, Jia, Dang. Test-Time Adaptation for Speech Emotion Recognition. ICASSP 2026（arXiv 2601.16240．arXiv の注記 "Accepted by ... ICASSP 2026"）**（○．抄録）
- "we present the first systematic evaluation and comparison covering 11 TTA methods across three representative SER tasks."
- "Conversely, entropy minimization and pseudo-labeling generally fail, as their core assumption of a single, confident ground-truth label is incompatible with the inherent ambiguity of emotional expression."
- ほかのモダリティは使わない．私たちの 11 手法のうち，同じモダリティのラベル無しの発話を使う手法は声の分類器を上げにくかった．Dong の結果はこれと同じ向きである（こちらの読み．条件は違う）．

**He ほか. M4SER. IEEE TASLP 33:4055–4070, 2025（arXiv 2509.18706）**（手元のノート `要点_問い3_同じ設定の研究.md` 3 節に全文を読んだ記録がある．本調べでは読み直していない）
- 声と ASR の言葉を使い，IEMOCAP ↔ MELD，4 感情，クラスあたり K = 5，10，20，60 で評価した．ノートの記録では，比べた相手は土台のモデル（HuBERT＋BERT）だけで，DA の手法やクロスモーダルの蒸留とは比べていない．

**Li, Kim, Kuo, Narayanan. Interspeech 2021**（○．抄録）
- 少数のターゲットラベルで，演じ方（即興と台本）の違いを越える音声・映像の感情認識である．"Our emotion recognition model combines the gradient reversal technique with an entropy loss function as well as the softlabel loss"．抄録には，クロスモーダルの蒸留との比べは無い．

**Jiménez-Guarneros, Fuentes-Pineda, Grande-Barreto. CMJDA. IEEE TIM 2025（DOI 10.1109/TIM.2025.3551924）**（○．抄録）
- 脳波と視線の多モダリティの SSDA で，ずれは被験者間である．"correlated features and consistent predictions are produced between modalities"，"only including three labeled target samples per class from the first recorded trial"．
- 感情認識の少数ラベルの SSDA でモダリティ間の一致を使う点は近い．信号の種類とずれの種類（被験者間）は私たちと違う．比べた相手は抄録に書かれていない．

**MuSACo. arXiv 2508.12522（2025-08-17）**（○．抄録）
- 被験者ごとの適応で，多ソースの DA にあたる．"generates pseudo-labels using the dominant modality for class-aware learning"，"MuSACo outperforms UDA (blending) and state-of-the-art MSDA methods."．データは BioVid，StressID，BAH で，ずれは被験者間である．

**UF-AMA. arXiv 2606.00170（2026-05-29）**（○．抄録）
- 脳波と視線の被験者間・セッション間の感情認識で，"applies global consistency alignment and cross-modal distillation"．コーパス間の音声・顔・言葉ではない．

**SSMDG（Li ほか CVPR 2026）と Audio-Adaptive（Zhang ほか CVPR 2022）**（手元の `先行研究_Albanieの後続研究_2026-11-04/report_B_少数ラベルとコーパス間DA.md` の Q4．本調べでは読み直していない）
- 行動認識で，ほかのモダリティの予測を疑似ラベルに使い，同じモダリティの手法（FixMatch など）と比べた．課題は感情ではない．

### 3.3 私たちの結果と先行研究の関係（こちらの読み．検定していない）

| 私たちの結果 | 先行研究の観察 | 関係 |
|---|---|---|
| 声の分類器で，蒸留の点推定が 11 手法より上（132 中 132） | xMUDA "often outperforms the uni-modal UDA baselines"，xMoSSDA "in 8 out of 9 cases"（3 次元の意味分割） | 向きは同じである．課題が違うので，この 2 本は「別の課題でも同じ向き」として引ける．私たちの比べの新しさは否定しないが，「クロスモーダルの学習と単一モダリティの DA の比べは初めて」とは書けない |
| AVT の分類器では差の向きが混ざった | xMoSSDA "improves 2D and 3D more (max. 2.0) than ensemble result 2D+3D (max. 0.4)" | 同じ向きの観察が別の課題にある．AVT の結果を書くときに引ける |
| 同じモダリティのラベル無しの手法（エントロピー最小化，自己学習など）は声の分類器を上げにくい | Dong 2026 "entropy minimization and pseudo-labeling generally fail"（声のテスト時適応） | 向きは同じである．ただし Dong はソースのモデルをテストの時に適応する設定で，ターゲットのラベルを使わない |
| 疑似ラベルの元にほかのモダリティを含む | Zhang 2021 では，顔だけの DCMKT は UA 27.99（ほぼでたらめの水準）で，顔と声の混合が最良（62.50）だった．ただし π-model（61.57）が ε = 0.5 の混合（61.20）を UA で上回った | 「ほかのモダリティの疑似ラベルなら必ず同じモダリティの手法に勝つ」という一般の主張は，Zhang 2021 の表 5 と食い違う．私たちの主張では，疑似ラベルの作り方（どの分類器の積か）を本文の定義どおりに書いて範囲を限る必要がある |

---

## 4. 既存の手法どうしの比べを貢献として書いた例（原文）

手法そのものは新しくない比べの結果を，範囲を付けて貢献に書いた例である．

### 4.1 Dong, Jia, Dang. Test-Time Adaptation for Speech Emotion Recognition. ICASSP 2026（arXiv 2601.16240．抄録，○）

> "In this paper, we present the first systematic evaluation and comparison covering 11 TTA methods across three representative SER tasks. The results indicate that backpropagation-free TTA methods are the most promising. Conversely, entropy minimization and pseudo-labeling generally fail, as their core assumption of a single, confident ground-truth label is incompatible with the inherent ambiguity of emotional expression. Further, no single method universally excels, and its effectiveness is highly dependent on the distributional shifts and tasks."

- 範囲の語：「初めて」を "systematic evaluation and comparison" に付け，数を "11 TTA methods" と "three representative SER tasks" で限っている．"generally" と "no single method universally excels" で例外を残している．声の感情認識で，既存の手法どうしの比べを貢献にした例のうち，私たちに最も近い．

### 4.2 Gulrajani, Lopez-Paz. In Search of Lost Domain Generalization（DomainBed）. arXiv 2007.01434（抄録，○．ICLR 2021 の掲載は `docs/related-work.md` で要確認とされている）

> "Next, we implement DomainBed, a testbed for domain generalization including seven multi-domain datasets, nine baseline algorithms, and three model selection criteria. We conduct extensive experiments using DomainBed and find that, when carefully implemented, empirical risk minimization shows state-of-the-art performance across all datasets."

- 範囲の語：比べの大きさを "seven multi-domain datasets, nine baseline algorithms, and three model selection criteria" と先に書き，結果の文に "when carefully implemented" という条件を付けている．既存の手法（ERM）がほかの既存の手法より良かったことを，主な発見として書いている．

### 4.3 Chen, Liu, Kira, Wang, Huang. A Closer Look at Few-shot Classification. ICLR 2019（arXiv 1904.04232．抄録，○）

> "In this paper, we present 1) a consistent comparative analysis of several representative few-shot classification algorithms, with results showing that deeper backbones significantly reduce the performance differences among methods on datasets with limited domain differences, 2) a modified baseline method that surprisingly achieves competitive performance when compared with the state-of-the-art on both the miniImageNet and the CUB datasets, and 3) a new experimental setting for evaluating the cross-domain generalization ability for few-shot classification algorithms."
> "In a realistic cross-domain evaluation setting, we show that a baseline method with a standard fine-tuning practice compares favorably against other state-of-the-art few-shot learning algorithms."

- 範囲の語："on datasets with limited domain differences"，"on both the miniImageNet and the CUB datasets"，"In a realistic cross-domain evaluation setting"．勝ち方は "competitive" と "compares favorably" で書き，「上回る」とは書いていない．arXiv の抄録にある "\miniI" は LaTeX の命令なので，ここでは miniImageNet と書いた．

### 4.4 Oliver, Odena, Raffel, Cubuk, Goodfellow. Realistic Evaluation of Deep Semi-Supervised Learning Algorithms. NeurIPS 2018（arXiv 1804.09170．抄録，○）

> "After creating a unified reimplementation of various widely-used SSL techniques, we test them in a suite of experiments designed to address these issues. We find that the performance of simple baselines which do not use unlabeled data is often underreported, that SSL methods differ in sensitivity to the amount of labeled and unlabeled data, and that performance can degrade substantially when the unlabeled dataset contains out-of-class examples."

- 範囲の語："a unified reimplementation"（同じ実装で比べたこと），"often"，"can degrade"．発見を条件ごとに分けて書いている（ラベル付きとラベル無しのデータの量，クラス外の例の有無）．

### 4.5 Lian ほか. MERBench. arXiv 2401.03429（v3．本文，◎）

> "Although each method shows its superior performance, different methods lack a fair comparison due to inconsistencies in feature extractors, evaluation manners, and experimental settings. These inconsistencies severely hinder the development of this field. Therefore, we build MERBench, a unified evaluation benchmark for multimodal emotion recognition."（要旨）
> "In general, multimodal fusion can improve both within- and cross-corpus results. However, its utility is relatively lower in the cross-corpus setting."（5.4 節）

- 範囲の語：MER の手法どうしの比べを貢献にした例である．結果の文は "In general" と "relatively lower in the cross-corpus setting" のように，設定の名で範囲を限っている．

### 4.6 参考：手法の論文だが，比べの数え方で範囲を付けた例

- xMoSSDA（TPAMI 2022）："Finally, xMoSSDAPL outperforms all baselines by up to +2.2 in 8 out of 9 cases."．勝った場合の数と全体の数，差の上限を同じ文に書いている．私たちの「132 中 132（区間で 122）」と同じ書き方である．
- xMUDA（CVPR 2020）：要旨では "xMUDA brings large improvements over uni-modal UDA on all tested scenarios"，本文では "often outperforms the uni-modal UDA baselines"．要旨では「試したすべての場面で」と範囲を付け，本文では "often" と弱めている．
- Wagner ほか. Dawn of the transformer era in speech emotion recognition. TPAMI 2023（arXiv 2203.07378．抄録，○）："To the best of our knowledge, we obtain the top performance for valence prediction without use of explicit linguistic information, with a concordance correlation coefficient (CCC) of .638 on MSP-Podcast."．"To the best of our knowledge" に加えて，条件（言葉の情報を明示的に使わない），指標，データセットで範囲を付けている．

### 4.7 例に共通する範囲の付け方

1. 比べの大きさを数で書く（手法の数，データセットの数，設定の数）．
2. 結果の文に設定の名を付ける（"in a realistic cross-domain evaluation setting"，"in the cross-corpus setting"，"across three representative SER tasks"）．
3. 勝った場合を数えて書く（"in 8 out of 9 cases"）か，弱い語を使う（"often"，"generally"，"compares favorably"）．
4. 「初めて」を付けるときは，手法には付けない．比べの仕方（"systematic evaluation and comparison"）か，条件つきの成績（"without use of explicit linguistic information"）に付ける．

---

## 5. 私たちの主張に付けるべき範囲の候補（先行研究と食い違わない範囲）

主張の候補：「少数ラベルのコーパス間の MER で，ほかのモダリティを含む疑似ラベルの蒸留が，同じモダリティのラベル無しの発話やソースのコーパスを使う DA・半教師あり学習の手法より，声の分類器で UAR が高かった」．

### 5.1 付ければ食い違わない範囲（どの先行研究との食い違いを避けるか）

| 範囲の要素 | 付けないと食い違う先行研究・記録 |
|---|---|
| コーパス間（AFEW，IEMOCAP の即興の部，CMU-MOSEI，MELD の 4 コーパスの LOCO） | xMUDA・xMoSSDA（3 次元の意味分割で同じ形の比べをした），CMJDA・UF-AMA・MuSACo（感情認識だが，ずれは被験者間） |
| ターゲットに，感情ごとに K = 5，10，25 本のラベルと，3 モダリティのそろうラベル無しの発話がある | Zhang 2021（ターゲットのラベルは全部あり，ラベル無しのデータは別コーパス），Amanda・Dong 2026（ターゲットのラベルを使わない） |
| 凍結したエンコーダの上の線形分類器．比べた手法はこの形に移した版である（MME は分類器の側の変種，xMUDA は「型」） | 原論文の手法は網を学ぶので，同じ手法だとは言えない（`paper/notes/引用の候補_2026-11-14.md` の 7 節にも同じ注意がある） |
| 声の分類器に限る．AVT では差の向きが混ざったことを，同じ文か次の文で書く | xMoSSDA（単一モダリティの得が融合より大きい）．AVT を含めて「蒸留が DA 手法より良い」と書くと，私たちの結果そのものと食い違う |
| 比べた 11 手法の名を挙げ，「ほかのモダリティを疑似ラベルに使わない手法」と定義する | 「DA 手法より良い」と一般に書くと，比べていない手法（Amanda，xMoSSDA の本体，ほかの多モダリティの DA）まで含めてしまう |
| 疑似ラベルの作り方を本文の定義どおりに書く（どの分類器の確率の積か，温度なし，Albanie ほか 2018 の型） | Zhang 2021（顔だけの疑似ラベルは弱く，π-model が混合の 1 つを UA で上回った） |
| 数え方を書く：UAR の点推定で 132 中 132，差の 95% 区間が 0 を超えたのは 122 | 「すべてで有意に上回った」と書くと，区間が 0 を超えなかった 10 と食い違う |

### 5.2 書けないこと

- 手法の新しさ：蒸留の型は Albanie ほか 2018 のものである．融合から単一モダリティへのまねは xMUDA の Fusion にある（`docs/related-work.md` 1.14 節の (h)）．
- 「クロスモーダルの学習と単一モダリティの DA・SSDA の手法を比べたのは初めて」：xMUDA と xMoSSDA が 3 次元の意味分割で比べている．
- 「声の感情認識で，クロスモーダルの疑似ラベルと同じモダリティの半教師あり学習を比べたのは初めて」：Zhang 2021 が比べている．
- 範囲を付けない「state-of-the-art」「outperforms domain adaptation methods」．

### 5.3 「見つからなかった」と書ける範囲

- 今回の検索と手元のノートの範囲では，次の研究は見つからなかった．音声・顔・言葉のコーパス間の感情認識で，ターゲットのラベルが感情ごとに数本のときに，ほかのモダリティを含む疑似ラベルの蒸留と，同じモダリティの SSDA・UDA・テスト時適応の手法を，同じ条件で並べた研究である．
- 論文にこれを書くなら，Ryumina 2025 や Dong 2026 と同じ型になる．"To the best of our knowledge" を付けるか，比べの仕方に「初めて」を付ける書き方である．その場合は，上の 3 つの条件（コーパス間，感情ごとに数本のラベル，3 モダリティ）を同じ文に入れる．同じ段落で Zhang 2021，xMUDA・xMoSSDA，Amanda を引き，違い（ターゲットのラベルの量，課題，比べた相手）を 1 文ずつ書けば，4 節の例と同じ形になる．書くかどうかは本体のセッションが判断する．

---

## 6. 確かめられなかったこと・調査の限界

- Ryumina 2025 の査読の記録は確かめられなかった．Interspeech 2024 に先に投稿したかどうかも確かめられなかった（公開の頁のフッターにある GitHub の組織名は "INTERSPEECH-2024" である）．
- 次の論文は抄録だけを読み，本文の比べの相手と数値は確かめていない．Avots 2018，Yin 2020，Ananthram 2020，CMGA，GMU，Kumar & Vepa 2020，CMJDA，MuSACo，UF-AMA，Dong 2026，DomainBed，Chen 2019，Oliver 2018，Wagner 2023．GMU と Kumar & Vepa の掲載先，MuSACo の著者も確かめていない．
- Sangwan ほか 2019（GCCMA）は Mathematics 2023 の紹介で知っただけで，原論文は読んでいない．
- 「GMU と Ryumina の式（1）が同じ形である」という判断は，Ryumina の式と GMU の抄録の記述からのこちらの読みである．GMU の本文の式とは照らしていない．
- Ryumina 2022（Neurocomputing）が LOCO を使うことは，検索の要約でしか確かめていない．
- MERBench の掲載先（TPAMI 48(5), 2026）は `docs/related-work.md` の記録で，本調べでは確かめていない．
- Amanda の公開日は，会議の月（2024 年 8 月）までしか確かめていない．
- 検索は英語だけで，米国向けの結果である．中国語とロシア語の論文，学位論文はほとんど見ていない．OpenAlex の API は回数の上限で止まったので，Semantic Scholar，arXiv，Crossref の API で代えた．
- 検索語には一般の語と論文の題名だけを使った：multi-corpus multimodal emotion recognition，cross-corpus multimodal emotion recognition leave-one-corpus-out，leave-one-database-out audiovisual emotion，multimodal emotion recognition domain generalization unseen corpus，cross-corpus speech and text emotion domain adaptation，cross-modal knowledge distillation cross-corpus speech emotion recognition，multimodal unsupervised domain adaptation emotion recognition，test-time adaptation multimodal emotion recognition，semi-supervised domain adaptation speech emotion recognition，multimodal teacher unimodal student domain adaptation emotion．

---

## 7. 出典

◎（原文を読んだ）
- Ryumina ほか 2025，PRL 190:192–200．手元の PDF `paper/references/2025_Ryumina_MultiCorpusEmotionRecognitionCrossModalGated_PRL.pdf`．DOI 10.1016/j.patrec.2025.02.024
- Ryumina, Markitantov, Karpov 2023，Mathematics 11(16):3519．DOI 10.3390/math11163519（MDPI の PDF）
- Ryumina ほか，arXiv 2403.12687（6th ABAW）https://arxiv.org/abs/2403.12687
- Hu ほか，UniMSE，arXiv 2211.11256 https://arxiv.org/abs/2211.11256
- Lian ほか，MERBench，arXiv 2401.03429v3 https://arxiv.org/abs/2401.03429
- Zhang ほか，Amanda，Findings of ACL 2024 https://aclanthology.org/2024.findings-acl.859.pdf
- Jaritz ほか，xMUDA，arXiv 1911.12676 https://arxiv.org/abs/1911.12676
- Jaritz ほか，TPAMI 2022（xMoSSDA），arXiv 2101.07253 https://arxiv.org/abs/2101.07253
- Zhang ほか 2021，KBS 229:107340．DOI 10.1016/j.knosys.2021.107340．本文 https://users.monash.edu/~yli/assets/pdf/2021-KBS-SpeechEmotionRecognition.pdf

○（抄録・書誌）
- Avots ほか，MVA 30:975–985．DOI 10.1007/s00138-018-0960-9（Semantic Scholar の抄録）
- Yin ほか，ICMI 2020．DOI 10.1145/3382507.3418813（Semantic Scholar の抄録）
- Li, Kim, Kuo, Narayanan，Interspeech 2021，arXiv 2104.01978
- Ananthram ほか，arXiv 2011.07065
- Arevalo ほか，GMU，arXiv 1702.01992
- Jiang, Ji，CMGA，arXiv 2208.11893
- Kumar, Vepa，arXiv 2003.01043
- Dong, Jia, Dang，ICASSP 2026，arXiv 2601.16240
- Jiménez-Guarneros ほか，CMJDA，DOI 10.1109/TIM.2025.3551924
- MuSACo，arXiv 2508.12522
- UF-AMA，arXiv 2606.00170
- Gulrajani, Lopez-Paz，arXiv 2007.01434
- Chen ほか，arXiv 1904.04232
- Oliver ほか，arXiv 1804.09170
- Wagner ほか，arXiv 2203.07378
- Ryumina ほかの公開の頁 https://smil-spcras.github.io/MER/（WebFetch の要約）

△（二次資料）
- Sangwan ほか 2019（GCCMA）：Mathematics 2023 の 2.2 節の紹介
- Ryumina, Dresvyanskiy, Karpov 2022（Neurocomputing 514）が LOCO を使うこと：検索の要約

手元で読んだ記録
- `docs/related-work.md`（1.12〜1.15 節，Ryumina の項）
- `research_notes/有名な論文_2026-11-07/README.md`
- `research_notes/先行研究_Albanieの後続研究_2026-11-04/`（README，report_A，report_B）
- `docs/参照/DAI_paper/参照論文/要点_問い1_別モダリティ蒸留.md`，`要点_問い3_同じ設定の研究.md`
- `paper/notes/書き方_Ryumina2025.md`，`paper/notes/引用の候補_2026-11-14.md`