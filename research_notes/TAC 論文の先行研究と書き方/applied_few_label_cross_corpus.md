# 少数のターゲットラベルを使うコーパス間の感情認識（音声・顔・マルチモーダル，2022〜2026）

確かめた水準：◎ 本文を読んだ（arXiv の HTML 版や PMC 版を取得ツールの要約経由で読んだものを含む），○ 抄録を読んだ，△ 検索結果の抜き書きか書誌だけ．
書誌の出どころは OpenAlex と Semantic Scholar の書誌データ（DOI つき）である．IEEE TAC の巻と年は「巻 N ＝ 2009＋N 年」で対応する（例：ADDoG は PMC の記録で巻 12 号 4 が 2021 年 10〜12 月号）．OpenAlex の「年」は早期公開の年なので，巻の年と食い違うことがある．

## 問い1：ターゲットのラベルを数本〜数十本だけ使う感情認識の論文（2022〜2026）

### Takeaway
2022〜2026 年に，ターゲットのラベルをクラスあたり数本〜数十本だけ使うコーパス間（言語間を含む）の感情認識は，音声単独で 6 本前後，音声＋言葉で 2 本（M4SER，MDAT）見つかった．どの論文も手法の提案が目的で，ずれの量を測る論文や，少数ラベルが判断基準（切片）と境目の向きのどちらを直すかを調べた論文は見つからなかった．

### Cited Findings

**一覧表**（既知のものは除く．既知の TAC 掲載分は問い3 で確かめ直した）

| 論文 | 誌・年・巻号 | モダリティ | コーパス | ターゲットのラベル数 K | 手法 | 比較相手 | ずれを測るか | ラベルが何を直すかの分析 | 水準 |
|---|---|---|---|---|---|---|---|---|---|
| M4SER | IEEE/ACM TASLP 33:4055–4070（2025） | 音声＋ASR の言葉 | IEMOCAP ↔ MELD（4 感情 N/H/A/S） | クラスあたり 5・10・20・60 | HuBERT＋BERT の微調整，ASR 誤り検出・訂正の補助課題，敵対学習，ラベル対照学習 | SAMS，MF-AED-AEC | 測らない | なし | ◎ |
| MDAT（Zaidi, Latif, Qadir） | IEEE Open J. Computer Society 5:684–693（2024） | 音声＋言葉 | IEMOCAP・EMODB・EMOVO・URDU（言語間 12 組，4 感情） | 0・5・10・15（クラスあたりか合計かは書かれていない） | XLS-R と RoBERTa の特徴，グラフ注意と共注意 | BiLSTM，SAFRLM，HCAM | 測らない | なし | ◎（arXiv v2） |
| CTIAL（Xu, Jiang, Wu） | IEEE TAC 15(3):1659–1668（2024） | 音声 | IEMOCAP 内，VAM→IEMOCAP，MELD→IEMOCAP | 初期 20 本＋能動学習で最大 200 本（合計） | wav2vec2 の特徴（PCA）＋ロジスティック回帰．次元と分類の予測の不一致で選ぶ | ランダム，エントロピー，最小確信度など | MMD を BDA の調整にだけ使う | なし | ◎ |
| Agarla ほか | Expert Syst. Appl. 237:121368（2024．arXiv 2022） | 音声 | 5 言語（抄録では名前なし） | 「少数」（抄録に数なし） | Transformer＋擬似ラベル（硬・軟） | 既存手法 | 抄録では分布の差を課題として述べるだけ | なし | ○ |
| Yue ほか | APSIPA ASC 2023:1008–1012 | 音声 | 抄録に記載なし | 少数ショット（数は抄録になし） | 次元空間での感情間の距離を事前知識にした距離学習 | 抄録に記載なし | 不明 | なし | ○ |
| Zhang, Jia ほか | Speech Communication 168:103185（2025） | 音声 | 不明 | 不明（題名は semi-supervised DA） | 半教師ありドメイン適応のネットワーク | 不明 | 不明 | 不明 | △ |
| Liu ほか | Applied Soft Computing 147:110766（2023） | 音声 | 不明 | 不明 | メタ転移学習＋DA | 不明 | 不明 | 不明 | △ |
| Castorena, Cobos, Ferri | IEEE SPL 32:2873–2877（2025） | 音声 | 複数コーパス（抄録では名前なし） | 新しい話者ごとに少数 | 潜在空間で k-means を変形して代表例を逐次選ぶ（話者適応） | ランダムな選択とラベル付け | なし | なし | ○ |
| 知識ベースの論文（Wav2Vec2＋韻律の注意融合） | Knowledge-Based Systems（2023） | 音声 | EmoDB ほか | ターゲットの話者の 20% | ソースで学んだモデルの微調整 | 不明 | 不明 | 不明 | △ |

**各論文の要点**

- M4SER は IEMOCAP→MELD と MELD→IEMOCAP の 2 方向で，ターゲットの検証用の 30% からクラスあたり 5・10・20・60 本を抜いて少数ショット学習をした．感情は Neutral・Happy・Angry・Sad の 4 つで，MELD の Surprise・Fear・Disgust は捨てた．エンコーダは全体の設定では HuBERT と BERT を微調整している（コーパス間の節では凍結か微調整かを書いていない）．MELD→IEMOCAP の UA は，ターゲットのラベルなしで 62.2，クラスあたり 5・10・20・60 本で 64.4・67.7・68.7・75.9 である．比較相手の MF-AED-AEC は同じ K で 63.3・65.6・66.6・73.7 だった — [arXiv 2509.18706（HTML 版，表 VIII・IX）](https://arxiv.org/html/2509.18706)；誌名と巻号は [Semantic Scholar API の書誌](https://api.semanticscholar.org/graph/v1/paper/ARXIV:2509.18706?fields=title,venue,journal) による．著者名は確かめていない（要確認）
- MDAT は XLS-R（音声）と RoBERTa（言葉）の事前学習モデルで特徴を取り出し，ターゲット言語の K=0・5・10・15 本で微調整した．IEMOCAP→EMODB の UA は 15 本で 78.98（BiLSTM の基準）対 91.48（MDAT）である．0 本の値は表 IV（42.48）と表 V（48.48）で食い違う．ずれの量も，少数ラベルが何を直すかも調べていない — [arXiv 2306.13804v2（HTML 版）](https://arxiv.org/html/2306.13804v2)．arXiv の注記は「IEEE TAC に投稿中」だが，掲載先は IEEE Open Journal of the Computer Society 5:684–693 である — [DOI 10.1109/OJCS.2024.3486904](https://doi.org/10.1109/ojcs.2024.3486904)
- CTIAL は wav2vec2（LibriSpeech で事前学習し ASR 用に微調整したもの）の 12 層を時間平均した 768 次元に PCA をかけ，分類はロジスティック回帰，次元の推定はリッジ回帰で解く．コーパス間では MELD→IEMOCAP（4 感情）で BDA を使うと BCA（クラスごとの再現率の平均＝UAR）が 0.3904 から 0.5378 に上がった．能動学習は初期 20 本から 1 本ずつ 200 回選ぶ．著者の記述では，コーパス間では K が大きいところで CTIAL とランダムがほぼ同じになり，次元を推定する向きでは CTIAL が効いたのは覚醒度だけだった — [arXiv 2412.01171（HTML 版）](https://arxiv.org/html/2412.01171v1)；書誌は [DOI 10.1109/TAFFC.2024.3366767](https://doi.org/10.1109/taffc.2024.3366767)
- Agarla ほかは，新しい言語にラベル付きの例が少数しかないときに，ラベルのない発話に擬似ラベルを付けて Transformer を適応させる．話者独立で 5 言語を評価し，UA が既存手法より平均 40% 上がったと書く — [arXiv 2207.06767](https://arxiv.org/abs/2207.06767)
- Yue ほかは，連続の感情空間での分布から感情カテゴリ間の距離を求め，それを事前知識にした距離学習で少数ショットのコーパス間音声感情認識を行う — [Semantic Scholar の抄録（DOI 10.1109/APSIPAASC58517.2023.10317289）](https://doi.org/10.1109/apsipaasc58517.2023.10317289)
- Castorena ほかは，新しい話者への適応で，ラベルを付ける発話を潜在空間の代表例として逐次選ぶと，ランダムに選んでラベルを付けるより良いと報告した — [DOI 10.1109/LSP.2025.3584714](https://doi.org/10.1109/lsp.2025.3584714)
- 知識ベースの論文は，ソースで学んだモデルをターゲットの少数の標本で微調整し，EmoDB で話者の 20% を適応に使って正解率 92.45% と報告した（検索結果の抜き書き．著者は未確認） — [ScienceDirect S0950705123005646](https://www.sciencedirect.com/science/article/abs/pii/S0950705123005646)
- 話者単位の少数ショットの個人化（コーパス間ではない）として，音声言語モデルの文脈内学習で目標話者の少数の発話を条件にする Ihori ほか（ASRU 2025）がある — [DOI 10.1109/ASRU65441.2025.11434728](https://doi.org/10.1109/asru65441.2025.11434728)．聴き手ごとの個人化をメタ学習で行う Meta-PerSER（Interspeech 2025:136–140）もある（△） — [DOI 10.21437/Interspeech.2025-832](https://doi.org/10.21437/interspeech.2025-832)
- 顔では「cross-domain few-shot FER」という名の研究群がある（Zou ほか ECCV 2022・AAAI 2022，Chen ほか IEEE TIP 34:6273–6285（2025），IJCV 2026，IEEE TMM 2026）．この群は，ソースの基本表情で学び，ターゲットの**新しい複合表情のクラス**を少数の画像で認識する設定であり，ソースとターゲットでクラスが同じ本研究とは問題が違う — [Chen ほか TIP 2025 の抄録（DOI 10.1109/TIP.2025.3612281）](https://doi.org/10.1109/tip.2025.3612281)；[Zou ほか ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/790_ECCV_2022_paper.php)
- 生理信号では「multimodal semi-supervised domain adaptation」を名乗る論文がある（Jiménez-Guarneros ほか，IEEE TIM 74（2025））．EEG と眼球運動で，ラベル付きのソース・少数のラベル付きターゲット・多数のラベルなしターゲットを使う被験者間の適応であり，音声・顔・言葉ではない — [DOI 10.1109/TIM.2025.3551924](https://doi.org/10.1109/tim.2025.3551924)
- 2022 年以前で特に近いもの：Abdelwahab & Busso 2017 は，IEMOCAP→MSP-IMPROV で能動学習によりターゲットから 200 本を選んで SVM を逐次適応させた．ポスターの数値は，適応なし 45.5%，全標本で一度に適応 46.7% である（検索結果の抜き書き．会議名は未確認） — [MSP 研究室の PDF](https://msp.utdallas.edu/publications/Abdelwahab_2017_2.pdf)

### Inferences
- 少数のターゲットラベルを使うコーパス間の感情認識は 2022〜2026 年も主流ではなく，主流は「ターゲットのラベルを使わない」教師なし DA・領域汎化・ソースなし適応・テスト時適応である（問い3 の TAC の一覧でも同じ傾向）．
- 少数ラベルの実験がある論文でも，K は 1 つのコーパス対で 3〜4 点振るだけで，K を振ったときの「ソースを混ぜる利得」の形（どの K で追い越されるか）を調べた論文は見つからなかった．
- CTIAL は凍結した自己教師ありモデルの特徴＋ロジスティック回帰という学習器が本研究と同じ型で，コーパス間で能動学習の選び方がランダムと差を失う場面を報告している．本研究の否定結果（強い出発点の上ではラベルなしの選び方がランダムを上回らない）と同じ向きの観察として引ける．ただし CTIAL はソースとターゲットで課題（分類と次元）が違う．

### Gaps
- Zhang, Jia ほか（Speech Communication 2025）と Liu ほか（Applied Soft Computing 2023）は抄録を取得できなかった（ScienceDirect が取得を拒否した）．K の値と設定は要確認．
- Yue ほか（APSIPA 2023）は本文を見つけられず，コーパスと K は要確認．
- Agarla ほかの K の値は抄録に無く，本文で要確認．
- OpenAlex の無料枠を使い切り，dblp と Semantic Scholar も回数制限で止まったため，Interspeech 2024・2025 と ICASSP 2024〜2026 の全件の確認はできていない．

## 問い3：IEEE TAC 2023〜2026 のコーパス間・少数ショット・ドメイン適応の論文（音声・顔・マルチモーダル）

### Takeaway
TAC の 2023〜2026 年（巻 14〜17）で，音声・顔・マルチモーダルのコーパス間・ドメイン適応を扱う論文はおよそ 25 本見つかった．そのうちターゲットのラベルを使うのは CTIAL（能動学習）と個人化の数本だけで，コーパス間で少数ラベルを使う論文は見つからなかった．既知の TAC 3 本（Feng & Chaspari，ADDoG，Schuller 2010）の巻号は下のとおり確かめた．

### Cited Findings

**既知の TAC 掲載分の確かめ直し**

- Feng & Chaspari「Few-Shot Learning in Emotion Recognition of Spontaneous Speech Using a Siamese Neural Network With Adaptive Sample Pair Formation」は IEEE TAC 14(2):1627–1633，DOI 10.1109/TAFFC.2021.3109485 である（早期公開 2021，巻 14 は 2023 年） — [OpenAlex の書誌（DOI）](https://doi.org/10.1109/taffc.2021.3109485)．内容は related-work.md 2.2 節の記述（本文を読んだもの）のままでよい
- Gideon, McInnis, Mower Provost「Improving Cross-Corpus Speech Emotion Recognition with Adversarial Discriminative Domain Generalization (ADDoG)」は IEEE TAC 12(4):1055–1068（2021 年 10〜12 月号．早期公開 2019 年 5 月 14 日）である — [PMC9173710](https://pmc.ncbi.nlm.nih.gov/articles/PMC9173710/)（◎）
  - 課題は**感情価だけ**で，低・中・高の 3 区分の柔らかいラベルにしている（覚醒度は予備実験のみ）
  - 実験 1 は IEMOCAP ↔ MSP-Improv．UAR（でたらめなら 0.33）は MSP-Improv→IEMOCAP で CNN 0.439±0.022 対 ADDoG 0.474±0.009，IEMOCAP→MSP-Improv で 0.432±0.012 対 0.444±0.007
  - 実験 2〜4 はターゲットのラベルを **0・200・400・800・1,600・3,200 発話**（発話数）使う．ラベルなしのテスト特徴も学習に使う（トランスダクティブ）
  - 著者の記述では，ラベルが無いか 800 本までなら ADDoG が基準より有意に良く，IEMOCAP→PRIORI の 3,200 本では有意に悪い．ターゲットのラベルが多いと，ターゲットだけで学んだモデルが ADDoG に並ぶか上回る．複数ソースの MADDoG は実験 3c（IEMOCAP＋MSP-Improv→PRIORI）だけで使う
  - 少数ラベルが何を直すかの分析も，ずれの量の報告もない（Wasserstein 距離は学習の目的関数としてだけ使う）
- Schuller, Vlasenko, Eyben, Wöllmer, Stuhlsatz, Wendemuth, Rigoll「Cross-Corpus Acoustic Emotion Recognition: Variances and Strategies」は IEEE TAC 1(2):119–131（2010），DOI 10.1109/T-AFFC.2010.8 である — [OpenAlex の書誌（DOI）](https://doi.org/10.1109/t-affc.2010.8)．著者の 6・7 人目は OpenAlex の先頭 5 人に入らず未確認

**TAC 2023〜2026（巻 14〜17）の一覧**（EEG・生理信号・感情分析（文章）・抑うつは除いた．巻(号):頁．早期公開は「EA」）

| 題名 | 著者（先頭） | 巻(号):頁 | 要点 | 水準 |
|---|---|---|---|---|
| Self Supervised Adversarial Domain Adaptation for Cross-Corpus and Cross-Language Speech Emotion Recognition | Latif, Rana, Khalifa, Jurdak, Schuller | 14(3):1912–1926 | 2 つの識別器を使う敵対学習（ADDi）と自己教師ありの事前学習（sADDi）．**ターゲットのラベルを使わない**．5 コーパス・3 言語 | ○ [DOI](https://doi.org/10.1109/taffc.2022.3167013) |
| Unsupervised Cross-Corpus Speech Emotion Recognition Using a Multi-Source Cycle-GAN | Su, Lee | 14(3):1991–2004 | IEMOCAP と VAM をソース，MSP-Podcast をターゲットにし，コーパスを意識した注意で複数ソースを集めて合成標本を作る．覚醒度・感情価の分類．教師なし | ○ [DOI](https://doi.org/10.1109/taffc.2022.3146325) |
| Few-Shot Learning in Emotion Recognition of Spontaneous Speech Using a Siamese Neural Network With Adaptive Sample Pair Formation | Feng, Chaspari | 14(2):1627–1633 | 既知（上記） | ◎（既存の調べ） |
| An Adversarial Training Based Speech Emotion Classifier With Isolated Gaussian Regularization | Fu, Liu, Ishi, Ishiguro | 14(3):2361–2374 | 「cross-corpus」の検索で出た．内容は未読 | △ [DOI](https://doi.org/10.1109/taffc.2022.3169091) |
| Multitask Learning From Augmented Auxiliary Data for Improving Speech Emotion Recognition | Latif ほか | 14(4):3164–3176 | 「cross-corpus」の検索で出た．内容は未読 | △ [DOI](https://doi.org/10.1109/taffc.2022.3221749) |
| An Enroll-to-Verify Approach for Cross-Task Unseen Emotion Class Recognition | Li, Lee | 14(4):3066–3077 | 未知の感情クラスへの課題間転移．内容は未読 | △ [DOI](https://doi.org/10.1109/taffc.2022.3183166) |
| Domain-Incremental Continual Learning for Mitigating Bias in Facial Expression and Action Unit Recognition | Churamani, Kara, Gunes | 14(4):3191–3206 | 顔．領域を増やしながら学ぶ継続学習．内容は未読 | △ [DOI](https://doi.org/10.1109/taffc.2022.3181033) |
| Audio-Visual Emotion Recognition With Preference Learning Based on Intended and Multi-Modal Perceived Labels | Lei, Cao | 14(4):2954–2969 | 音声＋映像．内容は未読 | △ [DOI](https://doi.org/10.1109/taffc.2023.3234777) |
| Adversarial Domain Generalized Transformer for Cross-Corpus Speech Emotion Recognition | Gao, Wang, Liu, Dang, Okada | 15(2):697–708 | 領域汎化（ADoGT）．4 コーパス．ターゲットのラベルなし | ○ [DOI](https://doi.org/10.1109/taffc.2023.3290795) |
| From the Lab to the Wild: Affect Modeling Via Privileged Information | Makantasis ほか | 15(2):380–392 | 実験室では全モダリティで学び，実環境では映像だけで推定する（特権情報） | ○ [DOI](https://doi.org/10.1109/taffc.2023.3265072) |
| Deep Adaptation of Adult-Child Facial Expressions by Fusing Landmark Features | Witherow ほか | 15(3):847–858 | 顔．成人と子どもの表情を共有の潜在空間でそろえる DA．2 組のデータセット対 | ○ [DOI](https://doi.org/10.1109/taffc.2023.3297075) |
| Cross-Task Inconsistency Based Active Learning (CTIAL) for Emotion Recognition | Xu, Jiang, Wu | 15(3):1659–1668 | 問い1 の表．コーパス間の能動学習，凍結特徴＋ロジスティック回帰 | ◎ [DOI](https://doi.org/10.1109/taffc.2024.3366767) |
| Joint Training on Multiple Datasets With Inconsistent Labeling Criteria for Facial Expression Recognition | Yu, Zhang, Zou, Li | 15(3):1812–1825 | 顔．追加データセット（AffectNet）から部分集合を選び，ターゲット（RAF-DB，CAER-S）の基準でラベルを付け直して合同学習．単純な合併より良い | ○ [DOI](https://doi.org/10.1109/taffc.2024.3382618) |
| Versatile Audio-Visual Learning for Emotion Recognition | Goncalves, Leem, Lin, Sisman, Busso | 16(1):306–318 | 音声＋映像で片方が欠けても動く枠組み．CREMA-D，MSP-IMPROV，MOSEI．コーパス間ではない | ○ [DOI](https://doi.org/10.1109/taffc.2024.3433386) |
| Individual-Aware Attention Modulation for Unseen Speaker Emotion Recognition | Fang ほか | 16(2):1205–1218 | 未知の話者へのメタ学習による個人化 | ○ [DOI](https://doi.org/10.1109/taffc.2024.3498937) |
| ParaLBench: A Large-Scale Benchmark for Computational Paralinguistics Over Acoustic Foundation Models | Zhang ほか | 16(3):1290–1306 | 音声基盤モデルのベンチマーク．「cross-corpus」の検索で出た．コーパス間の設定は未確認 | △ [DOI](https://doi.org/10.1109/taffc.2024.3506554) |
| Multitask Transformer for Cross-Corpus Speech Emotion Recognition | Ahn, Rana, Busso, Rajapakse | 16(3):1581–1591 | 音声と言葉で別々のマルチタスク Transformer を作り判定段で融合．ラベルなしのターゲットで対照学習と情報最大化．IEMOCAP，MSP-IMPROV，EMO-DB．著者は「コーパス間でのマルチモーダルの手法は大きく限られる」と書く | ○ [DOI](https://doi.org/10.1109/taffc.2025.3526592) |
| Unsupervised Cross-Domain Facial Expression Recognition via Class Adaptive Self-Training | Wang, Liu | 16(3):1618–1630 | 顔．教師なし．クラス分布（ラベルの割合）の違いが擬似ラベルに与える悪影響を分析し，クラス単位の表現分布とクラス分布を合わせる | ○ [DOI](https://doi.org/10.1109/taffc.2025.3529947) |
| Phonetically-Anchored Domain Adaptation for Cross-Lingual Speech Emotion Recognition | Upadhyay, Martinez-Lucas, Katz ほか | 16(3):1631–1645 | MSP-Podcast（英）↔ BIIC-Podcast（台湾華語）．感情ごとに共通する母音を錨にした教師なし適応 | ○ [DOI](https://doi.org/10.1109/taffc.2025.3530105) |
| SetPeER: Set-Based Personalized Emotion Recognition With Weak Supervision | Tran, Yin, Soleymani | 16(4):3000–3013 | 音声＋映像の弱ラベルのデータ EmoCeleb（約 1,500 話者・150 時間）を**モダリティをまたいだラベル付け**で作る．話者あたり 8 発話で個人化 | ○ [DOI](https://doi.org/10.1109/taffc.2025.3568024) |
| Generalizing to Unseen Speakers: Multimodal Emotion Recognition in Conversations With Speaker Generalization | Tu ほか | 16(4):3043–3054 | 会話の MER で未知の話者への汎化．内容は未読 | △ [DOI](https://doi.org/10.1109/taffc.2025.3566059) |
| Progressive Multi-Source Domain Adaptation for Personalized Facial Expression Recognition | Zeeshan, Pedersoli, Koerich ほか | 17(1):575–586 | 顔．被験者を 1 ソースとみなし，ターゲットの被験者に似たソースから順に足す（カリキュラム）．似ていないソースによる負の転移を避ける．ターゲットはラベルなし | ○ [DOI](https://doi.org/10.1109/taffc.2025.3627534) |
| Static for Dynamic: Towards a Deeper Understanding of Dynamic Facial Expressions Using Static Expression Data | Chen, Li, Zhang, Hu, Shan, Wang | 17(1):438–451 | 静止画の表情データで動画の表情認識を補う．素朴なマルチタスク学習では**負の転移**が起き，アダプタの専門家の混合で避ける | ○ [DOI](https://doi.org/10.1109/taffc.2025.3623135) |
| Harnessing Multimodal Unlabeled Data for Enhanced Speech Emotion Recognition | Goncalves, Busso | 17(1):1134–1146 | ラベルなし 70.7 時間から音響・顔・言葉の目的を作って音声モデルを鍛える．推論は音声のみ | ○ [DOI](https://doi.org/10.1109/taffc.2025.3647876) |
| Biased Heritage: How Datasets Shape Models in Facial Expression Recognition | Dominguez-Catena ほか | 17(3):2745–2761 | 顔．データセットの偏りがモデルの予測にどう伝わるかを，偏りを人工的に入れて測る | ○ [DOI](https://doi.org/10.1109/taffc.2026.3693403) |
| Strength in Numbers, Power in Subjectivity: Scalable Modeling of Individual Annotators for Emotion Recognition Within and Across Corpora | Tavernor, Mower Provost | 17(3):2840–2853 | 注釈者を 1 人ずつモデル化し，コーパス内とコーパス間で評価．ASRU 2025（既知）と同じ系統 | ○ [DOI](https://doi.org/10.1109/taffc.2026.3687422) |
| Video-Based Cross-Domain Emotion Recognition via Sample-Graph Relations Self-Distillation | Chen, Wen ほか | 17(3):3154–3169 | 動画の表情認識の教師なしコーパス間適応 | ○ [DOI](https://doi.org/10.1109/taffc.2026.3680134) |
| Evaluating Facial Expression Recognition Datasets for Deep Learning: A Benchmark Study With Novel Similarity Metrics | Gaya-Morey ほか | 17(3):3717–3729 | 顔の 24 データセットで，難しさ・汎化・データセット間の移りやすさを測る 3 つの類似度を提案（問い5） | ○ [DOI](https://doi.org/10.1109/taffc.2026.3693316) |
| Enhanced Emotion-Aware Contrastive Adaptation Networks for Source-Free Cross-Corpus Speech Emotion Recognition | Zhao, Wang, Lu, Li, Liu, Schuller | EA（2026） | ソースのデータを使わない（ソースなし）コーパス間適応．抄録は取得できず | △ [DOI](https://doi.org/10.1109/taffc.2026.3746232) |

- TAC で題名に「few-shot」を含む 2022 年以降の論文は 6 本で，感情認識（音声・顔・マルチモーダル）でコーパス間のものは無かった．あったのは，自閉の特性分類（14(2):1110–1124），音声映像のストレス検出の公平性（EA 2026），ソーシャルメディアのマルチモーダル感情分析 FMSA（17(3):3405–3419）などである — [OpenAlex の検索（TAC，2022 年以降，few-shot）](https://api.openalex.org/works?filter=primary_location.source.id:S104780363,from_publication_date:2022-01-01,title_and_abstract.search:few-shot)
- TAC の「cross-domain emotion」の検索 42 件の大半は EEG の被験者間適応だった（同じ検索の結果を数えた） — [OpenAlex の検索](https://api.openalex.org/works?filter=primary_location.source.id:S104780363,from_publication_date:2022-01-01,title_and_abstract.search:cross-domain%20emotion)
- 感情以外で近いものとして，音声映像の嘘の検出の領域間ベンチマーク「Benchmarking Cross-Domain Audio-Visual Deception Detection」（17(3):2625–2642）がある（△） — [DOI](https://doi.org/10.1109/taffc.2026.3699375)

### Inferences
- TAC の 2023〜2026 年のコーパス間の音声・顔の論文は，ほぼすべてターゲットのラベルを使わない設定である．本研究は TAC の中では「少数のターゲットラベル」の側に立つ数少ない論文になり，最も近い TAC の先行は Feng & Chaspari（音声）と CTIAL（音声，能動学習）である．
- Ahn, Rana, Busso, Rajapakse（2025）の「コーパス間でのマルチモーダルの手法は大きく限られる」という記述は，本研究の序論で「マルチモーダルのコーパス間は未開拓」と書くときの TAC 内の根拠として引ける（抄録の記述）．
- Wang & Liu（2025）が「クラス分布の違い」を擬似ラベルの失敗の原因として分析している点は，本研究の「少数ラベルは感情ごとの切片を直す」と近い話題（ラベルの割合のずれ）を顔の教師なし適応で扱った例である．ただし少数ラベルの役割は調べていない．

### Gaps
- OpenAlex は IEEE の抄録を持たないことが多く，題名だけで拾った．題名に「cross-corpus」「adaptation」などを含まない論文は漏れうる．
- 2026 年の早期公開分（巻 17 号 4 以降）は OpenAlex の収録が遅れている可能性があり，網羅できていない．
- Fu ほか，Latif ほか（14(4)），Li & Lee，Churamani ほか，Lei & Cao，ParaLBench，Tu ほか，Zhao ほか（EA 2026）は抄録を読めておらず，コーパス間で少数ラベルを使うかは要確認．

## 問い4：感情認識で「少数ラベルは閾値・事前確率・切片を直す」に当たる分析，ターゲットの少数ラベルで校正だけをする研究

### Takeaway
2022〜2026 年の感情認識で，ターゲットの少数ラベルが判断基準（切片・閾値・クラスの事前確率）を直すのか境目の向きを直すのかを切り分けた研究は見つからなかった．少数ラベルを閾値だけに使う先例は既知の Olah 2021（文章）だけで，音声の校正の研究（Chou 2023）はコーパス内の確信度の校正である．

### Cited Findings
- Chou ほか（Interspeech 2023）は，多ラベルの音声感情分類に事後の温度スケーリングを広げ，正解率を保ったまま校正（ECE）を改善した．目的は確信度の低い予測を棄却することで，コーパス間やターゲットの少数ラベルは扱っていない（検索結果の抜き書き） — [ISCA Archive Interspeech 2023](https://www.isca-archive.org/interspeech_2023/chou23_interspeech.html)
- Dong, Jia, Dang（ICASSP 2026:17157–17161）は，ラベルのないターゲットだけを使うテスト時適応 11 手法を 3 つの音声感情の課題で比べ，誤差逆伝播を使わない手法が最も有望で，エントロピー最小化と擬似ラベルはおおむね失敗すると報告した．著者は失敗の理由を「1 つの確かな正解ラベルがあるという前提が感情の曖昧さと合わない」とする — [DOI 10.1109/ICASSP55912.2026.11460722](https://doi.org/10.1109/icassp55912.2026.11460722)（○）
- Wang & Liu（TAC 2025）は，顔の教師なしコーパス間適応で，クラス分布の違いが擬似ラベルの学習に与える悪影響を「クラス単位の表現分布」と「予測確率」の両面から分析した — [DOI 10.1109/TAFFC.2025.3529947](https://doi.org/10.1109/taffc.2025.3529947)（○）
- Su & Lee（TAC 14(3)）は，ソースとターゲットの間の「データのずれ」と「ラベルのゆがみ（label distortion）」を性能低下の原因に挙げる — [DOI 10.1109/TAFFC.2022.3146325](https://doi.org/10.1109/taffc.2022.3146325)（○）
- Gideon ほか（ADDoG，TAC 12(4)）は，ターゲットのラベルが何を直すかを分析していない — [PMC9173710](https://pmc.ncbi.nlm.nih.gov/articles/PMC9173710/)（◎）
- M4SER と MDAT も，少数ラベルで微調整した結果の数値を示すだけで，何が直ったかを分析していない — [arXiv 2509.18706](https://arxiv.org/html/2509.18706)；[arXiv 2306.13804v2](https://arxiv.org/html/2306.13804v2)（◎）
- 2015 年の EURASIP J. Audio, Speech, and Music Processing の論文（潜在トピックモデルの特徴によるコーパス内・コーパス間の音声感情認識）は，学習とテストのコーパスでクラスの事後確率が違うため，片方で学んだ判断の境目が他方で最適でなくなると論じ，テスト側のコーパスの少数のラベル付き発話でその偏りを補正する，と検索結果の要約にある（本文は取得できなかった．著者と数値は未確認） — [Springer の記事ページ](https://link.springer.com/article/10.1186/s13636-014-0049-y)

### Inferences
- 「少数ラベルは切片を直し，向きは直さない」という本研究の主張に当たる分析は，感情認識の 2022〜2026 年の論文には見つからなかった．校正の研究はコーパス内の確信度の校正（Chou 2023），テスト時適応の研究はラベルなしの適応（Dong 2026）で，どちらも本研究の問いとは別である．
- 先例として引けるのは，Olah 2021（文章，閾値だけを少数ラベルで合わせる．既知）と，上の 2015 年の EURASIP の論文（少数のテスト側ラベルで境目の偏りを補正．未確認）である．後者は本研究の主張に最も近い古い先例になりうるので，本文を確かめる価値がある．

### Gaps
- 2015 年の EURASIP の論文は，著者・補正の方法（切片だけか）・ラベル数・数値をすべて確かめられていない．
- 「prior shift」「label shift」「logit adjustment」を感情認識に使った論文は，今回の検索では見つからなかった．見つからなかったことは，存在しないことの証明ではない．

## 問い5：感情認識で「どのソースコーパスを選ぶか」「ソースの価値を測る」研究（2022〜2026）

### Takeaway
ソースの選び方を扱う研究は，(a) ソースの組み合わせを総当たりで試すもの（Chen & Yu 2023，既知），(b) ソースとターゲットの似ている度合いで選ぶもの（顔の個人化，話者の個人化），(c) データセット間の移りやすさを測る指標を作るもの（顔の 24 データセット），に分かれる．ソースの価値を「ターゲットのラベル何本ぶんか」で測った研究は見つからなかった．

### Cited Findings
- VoxEmo（Zhang, Chou, Narayanan, Hain．arXiv 2603.08936，Interspeech 2026 投稿）は，英語 4 コーパス（CREMA-D，IEMOCAP，MELD，MSP-Podcast）の全組で，音声言語モデルをソースで微調整してターゲットで評価した（Macro-F1，ターゲットのラベルは学習に使わないと読める）．Qwen2-Audio では MELD をソースにしたときに 3 ターゲット平均で利得が最大（本文は +17.7，表 8 の値 +17.1・+22.2・+11.2 の平均は約 16.8 で食い違う）．AudioFlamingo-3 では MSP-Podcast が「特に害のあるソース」で，表の最大の低下 3 つを占めた．IEMOCAP→MELD の低下（−3.4）は，IEMOCAP の 4 感情が MELD の 7 感情のうち 4 つしか覆わないためと説明している．コーパスの似ている度合い・大きさ・演技か自然かによる説明はしていない — [arXiv 2603.08936（HTML 版）](https://arxiv.org/html/2603.08936)（◎）
- Cahyawijaya ほか（Interspeech 2023:3352–3356）は，英語・中国語・広東語と成人・高齢者の間で音声感情の移りやすさを調べ，言語間の推論は不向きだが群をまたいだデータ拡張は正則化として役立ち，**言語間の距離が移りやすさに大きく影響する**と結論した — [DOI 10.21437/Interspeech.2023-327](https://doi.org/10.21437/interspeech.2023-327)（○）
- Braunschweiler ほか（IEEE SPL 29:722–726，2022）は，音声・言葉・両方を 6 コーパスで学習・評価し，言葉を足して良くなるのは感情ごとに文が変わるコーパスだけで，同じ文を違う感情で読むコーパスでは音声だけのほうが良いと報告した．単一コーパスのモデルは条件が合うと複数コーパスのモデルより良く，複数コーパスのモデルは音響の変化に頑健な傾向がある．成績は学習とテストの両方のコーパスの性質に依存する — [DOI 10.1109/LSP.2022.3151551](https://doi.org/10.1109/lsp.2022.3151551)（○）
- Gaya-Morey ほか（TAC 17(3):3717–3729）は顔の 24 データセットを正規化して比べ，Local・Global・Paired の 3 つの類似度で，難しさ・汎化・データセット間の移りやすさを測った．AffectNet や FER2013 のような大きい自動収集のデータセットは，ラベルの雑音や人口の偏りがあっても汎化しやすく，統制されたデータセットは注釈の質が高いが多様性が低い — [DOI 10.1109/TAFFC.2026.3693316](https://doi.org/10.1109/taffc.2026.3693316)（○）
- Zeeshan ほか（TAC 17(1):575–586）は，顔の個人化で被験者を 1 つのソースとみなし，ターゲットの被験者に似たソースから順に足すと，似ていないソースによる負の転移を避けられ，全ソースを一度に使うより計算も軽いと主張する — [DOI 10.1109/TAFFC.2025.3627534](https://doi.org/10.1109/taffc.2025.3627534)（○）
- Sridhar & Busso（TAC 13(4):1959–1972，2022）は，テストの話者と音響の似た学習側の話者を探して適応用の集合を作り，感情価のモデルを教師なしで個人化した（相対改善は最大 13.52%） — [DOI 10.1109/TAFFC.2022.3187336](https://doi.org/10.1109/taffc.2022.3187336)（○）
- Yu ほか（TAC 15(3):1812–1825）は，追加のデータセット（AffectNet）から部分集合を選び，ターゲットのデータセットのラベル基準に合わせてラベルを付け直してから合同学習すると，単純な合併より良いと報告した — [DOI 10.1109/TAFFC.2024.3382618](https://doi.org/10.1109/taffc.2024.3382618)（○）
- Su & Lee（TAC 14(3)）は，複数ソース（IEMOCAP，VAM）をコーパスを意識した注意で重み付けして集める — [DOI 10.1109/TAFFC.2022.3146325](https://doi.org/10.1109/taffc.2022.3146325)（○）
- Chen ほか（TAC 17(1):438–451）は，静止画の表情データを動画の表情認識に足すと，素朴なマルチタスク学習では負の転移が起きたと報告した — [DOI 10.1109/TAFFC.2025.3623135](https://doi.org/10.1109/taffc.2025.3623135)（○）
- Ryumina ほか（Big Data and Cognitive Computing 9(11):285，2025）は，英語の MELD とロシア語の RESD で音声＋言葉のモデルを合同学習し，最適なデータ拡張はコーパスによって違うと報告した — [DOI 10.3390/bdcc9110285](https://doi.org/10.3390/bdcc9110285)（○）

### Inferences
- ソースの価値を測る研究は，顔では Gaya-Morey ほか（類似度の指標），音声では Chen & Yu 2023（組み合わせの総当たり）と VoxEmo（組ごとの利得の表）があるが，いずれもターゲットのラベルを使わないか固定した条件での比較である．K を振ってソースの価値を「ターゲットのラベル何本ぶんか」で表す本研究の測り方は，今回見た範囲には無い．
- 「似ているソースから選ぶ」という発想（Zeeshan 2025，Sridhar & Busso 2022）は話者・被験者の単位で使われており，コーパスの単位で，しかも少数のターゲットラベルがある条件で検証した研究は見つからなかった．
- VoxEmo の「MELD が平均で最も良いソース」「MSP-Podcast が害になるモデルがある」という結果は，ソースの価値がモデル（エンコーダ）に依存することを示しており，本研究が 2 つのエンコーダで再現を確かめる理由づけに使える．

### Gaps
- VoxEmo は投稿中の preprint で，表の数値と本文の数値が食い違う（+17.7 と約 16.8）．
- 顔の「データセット蒸留」「データ価値の評価（data valuation）」を感情認識に使った研究は今回は探していない．

## 問い2（総合判定）：マルチモーダル × 少数ターゲットラベル × コーパス間の組み合わせはあるか，本研究に最も近い 5 本

### Takeaway
「マルチモーダル × 少数ターゲットラベル × コーパス間」は，**音声＋言葉の 2 モダリティでは存在する**（M4SER：IEEE/ACM TASLP 2025，MDAT：IEEE OJCS 2024）．音声・顔・言葉の 3 つ，または顔を含む組み合わせで少数のターゲットラベルを使うコーパス間の研究は見つからなかった．さらに「仕組みの分析」（ずれを測る，少数ラベルが何を直すかを調べる，ソースの価値をラベルの本数で測る）まで含む研究は見つからなかった．したがって本研究と同じ組み合わせは，今回の調べの範囲では存在しない．ただし 2026-09-30 の調べの「マルチモーダル×少数ターゲットラベル×コーパス間の論文は見当たらなかった」という記述は，音声＋言葉については改める必要がある．

### Cited Findings
- M4SER は音声＋ASR の言葉，IEMOCAP ↔ MELD，4 感情（Neutral・Happy・Angry・Sad），クラスあたり K=5・10・20・60 で，MELD→IEMOCAP の UA が 62.2（K=0）から 75.9（K=60）に上がった．少数ショットは論文の主題ではなく，手法の評価の 1 節（V-G）である — [arXiv 2509.18706](https://arxiv.org/html/2509.18706)（◎）
- MDAT は音声（XLS-R）＋言葉（RoBERTa），4 言語の 4 コーパス，4 感情，K=0・5・10・15 で，IEMOCAP→EMODB の UA は K=15 で 91.48 である — [arXiv 2306.13804v2](https://arxiv.org/html/2306.13804v2)（◎）
- Ahn, Rana, Busso, Rajapakse（TAC 2025）は音声＋言葉のコーパス間だが，ターゲットのラベルを使わない — [DOI 10.1109/TAFFC.2025.3526592](https://doi.org/10.1109/taffc.2025.3526592)（○）
- Huang, Zhao, Jin（NLPCC 2023，LNCS:431–443）「Two-Stage Adaptation for Cross-Corpus Multimodal Emotion Recognition」は，事前学習したマルチモーダルの感情モデルを，課題に合わせた追加の事前学習と，クラスタに基づく損失での微調整の 2 段階でラベルなしのターゲットに適応させ，順位に基づく擬似ラベルのふるいでクラスの釣り合った標本を選ぶ．IEMOCAP と MSP-IMPROV で実験 — [Springer（DOI 10.1007/978-3-031-44696-2_34）](https://link.springer.com/10.1007/978-3-031-44696-2_34)（検索結果の抄録の要約．○ と △ の間）
- Shou ほか（arXiv 2603.26840，2026）は言葉・音声・映像の会話 MER で IEMOCAP と MELD の間の領域のずれとラベルの雑音を同時に扱う．抄録ではターゲットのラベル数は書かれていない — [arXiv 2603.26840](https://arxiv.org/abs/2603.26840)（○）
- Tran ほか SAAML（ACM MM 2023:6004–6015）は，MSP-IMPROV と CREMA-D を手がかりに VoxCeleb2 から感情の強い区間を 100 時間以上取り出して AV-HuBERT を追加学習し，微調整後の正解率を 3〜6% 上げた．ターゲットの少数ラベルの設定ではない — [DOI 10.1145/3581783.3612286](https://doi.org/10.1145/3581783.3612286)（○）
- Li, Kim, Kuo, Narayanan 2021（既知）は音声＋映像でターゲットの 10% のラベルを使うが，MSP-IMPROV の中の区分間であり，コーパス間ではない（related-work.md 2.2 節）

**本研究に最も近い 5 本**

1. **M4SER（IEEE/ACM TASLP 33:4055–4070，2025）**：設定が最も近い．4 感情が本研究と同じ（N/H/S/A），IEMOCAP と MELD はどちらも本研究の群で，K をクラスあたりで数え 5〜60 を振り，UA で報告する．違いは，モダリティが 2 つ（顔なし），ソースが 1 つでコーパス対が 1 組，エンコーダを微調整する，ずれを測らない，少数ラベルが何を直すかを調べない，の 5 点である — [arXiv 2509.18706](https://arxiv.org/html/2509.18706)
2. **MDAT（Zaidi, Latif, Qadir．IEEE OJCS 5:684–693，2024）**：音声＋言葉で，事前学習モデルの特徴の上で K=0〜15 のターゲットラベルを使う．4 感情．違いは，言語間（コーパス＝言語）であること，K の単位（クラスあたりか合計か）が書かれていないこと，表の間で数値が食い違うこと，分析がないこと — [arXiv 2306.13804v2](https://arxiv.org/html/2306.13804v2)
3. **CTIAL（Xu, Jiang, Wu．IEEE TAC 15(3):1659–1668，2024）**：学習器が本研究と同じ型（凍結した自己教師ありモデルの特徴＋ロジスティック回帰）で，コーパス間（MELD→IEMOCAP など）で少数のターゲットラベルを能動学習で選ぶ．コーパス間では K が大きいとランダムとの差が消えるという観察は，本研究の否定結果と同じ向きである．違いは，音声のみ，ソースとターゲットで課題（次元と分類）が違う，ずれを測らない — [arXiv 2412.01171](https://arxiv.org/html/2412.01171v1)
4. **Feng & Chaspari（IEEE TAC 14(2):1627–1633，2023．既知）**：コーパスが大きく重なり（IEMOCAP の台本と即興，RAVDESS，CREMA-D），ターゲットの少数ラベル（話者あたり・感情あたり 1〜10 件）を使い，微調整で負の転移が起きる場面を報告した．違いは音声のみ・手作り特徴・3 クラス・ソースを混ぜない — [DOI 10.1109/TAFFC.2021.3109485](https://doi.org/10.1109/taffc.2021.3109485)
5. **Chen & Yu（Interspeech 2023．既知）**：4 つのソースコーパスの全部分集合 16 通り × k=2〜64 で，どのソースが少数ラベル学習を助けるかを総当たりで測った．本研究の「ソースの価値」の問いに最も近い．違いは音声のみ・微調整・2 値判定・ずれを測らない（related-work.md 2.2 節）

次点は ADDoG（TAC 12(4)．ターゲットのラベル 0〜3,200 発話を振り，ラベルが多いとターゲットだけのモデルが並ぶか上回る．感情価のみ）と Li, Kim, Kuo, Narayanan 2021（音声＋映像，ターゲットの 10%，コーパス内の区分）である — [PMC9173710](https://pmc.ncbi.nlm.nih.gov/articles/PMC9173710/)

### Inferences
- **判定**：本研究と同じ組み合わせ（音声・顔・言葉のマルチモーダル × クラスあたり K=0〜50 のターゲットラベル × 7 群のコーパス間 × 仕組みの分析）は，今回の調べの範囲では存在しない．存在するのは，(i) 音声＋言葉で少数ラベルを使うコーパス間（言語間）の手法論文 2 本（M4SER，MDAT），(ii) 凍結特徴＋線形分類器で少数ラベルを能動学習で選ぶ音声の論文 1 本（CTIAL），(iii) 音声のみでソースの組み合わせと k を振る論文 1 本（Chen & Yu 2023），である．
- 論文 1 で新しさを書くときは「マルチモーダルで少数ラベルのコーパス間は初めて」とは書けない（M4SER と MDAT がある）．書けるのは，(a) 顔を含む 3 モダリティで，(b) 7 群の多数のソースとターゲットの組にわたって K を振り，(c) ずれを測って少数ラベルが判断基準（切片）を直すか境目の向きを直すかを切り分け，(d) ソースの価値をターゲットのラベルの本数で表した，という組み合わせの新しさである．
- M4SER の K=5→60 の UA の伸び（MELD→IEMOCAP で 64.4→75.9）は，微調整したエンコーダでの値であり，凍結特徴の本研究の地図と並べるときは学習器の違いを明記する必要がある．

### Gaps
- M4SER の著者名と，コーパス間の実験でエンコーダを凍結したかは本文の該当節で確かめられなかった（要確認）．
- MDAT の K がクラスあたりか合計かは本文に書かれていない（著者への確認か，コードの確認が要る）．
- Huang, Zhao, Jin（NLPCC 2023）と Shou ほか（arXiv 2026）は本文を読めておらず，少数ラベルの実験の有無は要確認．
- Interspeech 2024・2025，ICASSP 2024〜2026，ACM MM 2024・2025 を全件は確認できていない（API の回数制限）．とりわけ音声言語モデルの文脈内学習で新しいコーパスに少数例を与える研究は 2025〜2026 年に増えている可能性があり，追加の検索が要る．
