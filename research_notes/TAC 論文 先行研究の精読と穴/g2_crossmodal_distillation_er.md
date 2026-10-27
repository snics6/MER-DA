# 感情認識（と，その原型）でモダリティをまたぐ蒸留・特権情報を使う論文 10 本の精読と，本研究の軸 A1〜A11 との対応（2026-10-27）

対象は依頼文の 10 本（Gupta 2016，Lopez-Paz 2016，Albanie 2018，Srinivasan 2022，LiSER 2025，Hajavi & Etemad 2023，MKE 2021，UMT 2021，DAMR 2026，G²D 2025）と，余力分の 3 本（PriCon 2025，Guo 2026，KBS 2021）である．
読み方は，arXiv の PDF を curl で取って `pdftotext -layout` で文字にし，本文を全節読んだ（DAMR は MDPI のサイトが curl と WebFetch の両方に 403 を返したので抄録だけ．Guo 2026 と KBS 2021 は Elsevier と著者の頁が 403 で書誌だけ）．
前回のノート（`research_notes/TAC 論文の先行研究と書き方/method_components_precedents.md`）と `docs/related-work.md` 1.11・1.12 節を先に読み，そこに書いてある抄録水準の判定を本文で確かめ直した．

確かめた水準の記号：◎＝本文を全節読んだ，○＝抄録（Crossref・OpenAlex・arXiv の抄録）だけ，△＝書誌の記録か検索結果だけ．
軸の記号：○＝その軸をそのまま持つ，△＝近いものを持つ（根拠に書く），×＝持たない，？＝本文が読めず判定できない．

本研究の主張（確認用）：MER において凍結したエンコーダの上の線形の頭に限って理論を立て，ソースのコーパスだけで学んだ融合の分類器をターゲットのラベル無し発話に当てて弱いラベルを作り，クラスあたり K 本のターゲットのラベルと出発点への L2 罰則で単モーダル・融合の生徒を学ぶ．

---

## 論文 1．Gupta, Hoffman, Malik 2016, "Cross Modal Distillation for Supervision Transfer"

### Takeaway
ラベルのあるモダリティ（RGB）で学んだ CNN の**中間層の特徴**を，対になるラベル無しのモダリティ（深度・光学流）の CNN に L2 で写す．確率ではなく特徴を蒸留し，その後にターゲットの全ラベルで微調整する．少数ラベルの設定・理論・クラス均衡は無い．

### Cited Findings

**1．書誌と読んだ範囲**
- arXiv:1507.00448v2（2015-11-25）．本文 1〜4 節（4.1 深度，4.2 光学流），結論，補足 A・B を読んだ（◎）．CVPR 2016, pp. 2827–2836 という書誌は Albanie 2018 の参考文献 [31] と MKE の [15] の記載で確かめた（会議ページは未確認）— [arXiv](https://arxiv.org/abs/1507.00448)

**2．設定**
- 課題：物体検出（NYUD2，19 クラス，mAP）と行動検出（JHMDB，21 クラス，mAP）— [arXiv](https://arxiv.org/abs/1507.00448)
- 先生：ImageNet で学んだ RGB の CNN（AlexNet か VGG）．生徒：深度（HHA 表現）か光学流の AlexNet を乱数初期化から学ぶ．対になるラベル無し画像：NYUD2 の動画から学習側の場面の枠だけ（4.1 冒頭は「10K」，実験 1C の記述は「5K」と食い違う），光学流は UCF101 の 9K 本の動画から 5 枚ずつ（ラベルは使わない）
- 蒸留する量：「We use the last convolutional layer pool5 for all experiments」，損失は「f(x, y) = ||x − y||²」．確率ではなく特徴．閾値つきのロジスティック版も試したが「resulted in worse performance in initial experiments」
- 蒸留の後にターゲットのラベルで Fast R-CNN を微調整する（深度は NYUD2 の学習集合の全ラベル）．少数ラベルの設定は無い
- 評価：NYUD2 の標準の val/test（場面ごとに分割），JHMDB の 3 分割の平均

**3．示したこと（数値）**
- 表 1（NYUD2 val，mAP）：乱数初期化 22.7，RGB の重みの複製 25.1，蒸留 29.7，検出に適応した RGB 網から蒸留 30.5．pool5 まで凍結して fc だけ学ぶと複製 19.8 に対し蒸留 30.0（「the features learned on depth images at pool5 are discriminative by themselves」）．VGG から蒸留すると凍結 32.2，全層 33.6
- 相補性（表 1 右）：RGB だけ 22.3，＋複製の深度 33.8，＋蒸留の深度 35.6，＋VGG から蒸留 37.0．「the representations are still complementary」
- 表 4（蒸留する層）：pool1 24.4，pool2 28.4，conv3 30.6，conv4 29.9，pool5 30.5，fc6 29.7，fc7 27.7，conv3＋fc7 31.3
- 表 6（NYUD2 test）：深度だけ 34.2 → 41.7（22% 相対改善），RGB＋深度 44.4 → 47.1（AlexNet），46.2 → 49.1（VGG）
- 表 7（JHMDB，光学流）：乱数初期化 31.7，蒸留 35.7，教師あり事前学習の上限 38.4
- 表 5（ゼロショット）：MS COCO で学んだ RGB の検出器の上半分を，蒸留した深度の下半分につなぐ．NYUD2 の注釈を一切使わず RGB 33.0，深度 30.4，両方 37.6（NYUD2 で学ぶと 35.7/45.0/54.4）
- 理論：無い

**5．穴（原文）**
- 高い層での蒸留が落ちる理由：「We believe the drop in performance at higher layers is an artifact of the amount of data used for supervision transfer」
- 複数の層での蒸留は「We defer exploration of other choices in this space for future work」
- ターゲットのラベルの本数を変える実験は無い．クラス均衡・重みの選び方の記述は無い

### Inferences

**4．軸との対応**
- A1 ×：深層 CNN を乱数初期化から学ぶ．凍結した特徴の上の評価（「ft fc only」）はあるが，fc6・fc7 は非線形の層で線形の頭ではない
- A2 ×，A3 ×，A4 ×：理論は無い
- A5 ×：ターゲット（深度）のラベルは NYUD2 の学習集合を全部使う．K の概念は無い
- A6 △：先生は ImageNet，対データは NYUD2 か UCF101，評価は NYUD2 か JHMDB．光学流では対データ（UCF101）と評価（JHMDB）が別のコーパスで，ゼロショットの実験は MS COCO → NYUD2 である．ソースは 1 つ
- A7 △：2 つのモダリティの対データ．単モーダルの先生 → 単モーダルの生徒．融合の先生ではない．運用時は深度だけ（ゼロショット）か RGB＋深度
- A8 △：ソースだけで学んだ先生をターゲットのラベル無しデータに学び直さずに当てる点は同じ．ただし確率ではなく特徴．クラス均衡・K ごとの重みは無い
- A9 ×：出発点への罰則は無い（「copy from RGB」は初期化であって罰則ではない．Aytar & Zisserman の正則化は関連研究で引くだけ）
- A10 ×：mAP．話者・UAR・検定は該当しない
- A11 △：「Is the learned representation complementary to the representation on the source modality?」を融合の得で測った（表 1 右）．理論は無い

**6．本研究との差の書き方（案）**
- Gupta ら [CVPR 2016] は，ラベルのあるモダリティで学んだ網の中間層の特徴を対になるラベル無しのモダリティに写し，その後にターゲットの全ラベルで微調整した．本研究は特徴ではなく確率を弱いラベルにし，ターゲットのラベルをクラスあたり K 本に絞り，先生をソースのコーパスで固定したまま線形の生徒で得の仕組みを解析する．

### Gaps
- 対データの枚数が 4.1 節の「10K」と実験 1C の「5K」で食い違う．引くときはどちらかを明記しない
- CVPR の頁番号は他の論文の参考文献からの転記である

---

## 論文 2．Lopez-Paz, Bottou, Schölkopf, Vapnik 2016, "Unifying Distillation and Privileged Information"

### Takeaway
温度 T と模倣の重み λ をもつ一般化蒸留を定義し，特権情報（学習時だけ使えるモダリティ）の先生の soft label で生徒を学ぶ枠を与えた．理論は定理ではなく，3 つの仮定（生徒は遅い速さ，先生は速い速さ，先生からの模倣は n^−α）の下での上界の書き換えで，「先生が役立つのは少数データの場面」と述べる．実験は線形（ロジスティック回帰）の人工データと小さな網で，λ・T を掃くが K ごとに選ばない．

### Cited Findings

**1．書誌と読んだ範囲**
- arXiv:1511.03643v3（2016-02-26）．PDF の頭書きに「Published as a conference paper at ICLR 2016」．本文 1〜5 節を全部読んだ（◎）— [arXiv](https://arxiv.org/abs/1511.03643)

**2．設定**
- 定義（式 4・5）：f_s = argmin (1/n) Σ [(1−λ) ℓ(y_i, σ(f(x_i))) + λ ℓ(s_i, σ(f(x_i)))]，s_i = σ(f_t(x_i)/T)．「The temperature parameter T > 0 controls how much do we want to soften ... the imitation parameter λ ∈ [0, 1] balances the importance between imitating the soft predictions s_i and predicting the true hard labels y_i」
- 一般化蒸留の 3 段：(1) 先生を (x*_i, y_i) で学ぶ，(2) soft label σ(f_t(x*_i)/T) を計算，(3) 生徒を (x_i, y_i) と (x_i, s_i) で式 4 により学ぶ．x* = x なら Hinton の蒸留，x* が特権なら Vapnik の LUPI
- 半教師ありへの拡張（4.2）：「the semi-supervised extension of distillation allows the teacher to prepare soft labels for all the unlabeled data」
- 多課題・DA への言及（4.2）：「if the multiple tasks share the same labels y_i but differ in their input modalities, the input modalities from the source tasks are privileged information. ... the regular student representation is the input modality from the target task」．**本研究の設定（ラベルは同じ，ソースのコーパスに余分なモダリティ）に最も近い文**
- 実験：人工データ 4 種（ロジスティック回帰，n_tr = 200，d = 50，λ = T = 1，100 回の分割），MNIST（300 か 500 本，28×28 を特権・7×7 を通常，2 層 20 ユニットの網），CIFAR10 の半教師あり（ラベル 300 枚＋ラベル無し 50,000 枚），SARCOS の多課題

**3．示したこと（数値と理論）**
- 4.1 節の速さの議論（定理ではない）：生徒は R(f_s) − R(f) ≤ O(|F_s|_C/√n) + ε_s，先生は O(|F_t|_C/n) + ε_t，先生からの模倣は O(|F_s|_C/n^α) + ε_l（1/2 ≤ α ≤ 1）と**仮定**すると，R(f_s) − R(f) ≤ O((|F_s|_C + |F_t|_C)/n^α) + ε_l + ε_t．得が出る条件は「i) the capacity of the teacher being small, ii) the approximation error of the teacher being smaller than the approximation error of the student, and iii) the coefficient α being greater than 1/2」．「the teacher is most helpful in low data regimes, such as small datasets, ..., domain adaptation, transfer learning」．α > 1/2 は「We believe that the “α > 1/2 case” is a general situation」と推測で書く
- 人工データ（特権／通常／蒸留の正解率）：(1) きれいなラベル 96/88/95，(2) きれいな特徴 90/68/70（「not statistically significant」），(3) 関係のある特徴 98/89/97，(4) 標本ごとに違う特徴 96/55/56（「the privileged features adding information about the target function is not a sufficient condition for improvement」）
- CIFAR10 の半教師あり：「Distillation on the 300 labeled samples did not improve the student performance」，ラベル無し 50,000 枚への soft label で「a significant improvement」．温度によって落ちる理由は「the lack of a proper weighting between labeled and unlabeled data in (4)」
- MNIST：「the benefits of distillation diminished as we further increased the sample size」（300 → 500）

**5．穴（原文）**
- 「the lack of a proper weighting between labeled and unlabeled data in (4)」（ラベル付きとラベル無しの重みづけが無い）
- 速さの議論は仮定の列挙で，α の値も条件 ii) も実データで測っていない
- λ と T は図 1・2 で掃くだけで，ラベルの本数ごとに選ぶ手続きは無い

### Inferences

**4．軸との対応**
- A1 △：人工データの実験はロジスティック回帰（線形）だが，特徴は生の入力で凍結したエンコーダではない．MNIST・CIFAR は 2 層の網
- A2 ×：ラベルの誤差を切片と向きに分けない
- A3 △：速さの議論は n の関数で，「先生は少数データで最も役立つ」と述べるが，ソースの価値をラベルの本数に換算しない
- A4 ×
- A5 △：n = 200〜500（総数）と CIFAR の 300 枚．クラスあたりではなく総数．K を系統的に振っていない（MNIST の 300 と 500 だけ）
- A6 ×：同じ分布の中．DA は応用先として言及するだけ
- A7 △：x* を特権モダリティとみなす．先生は特権だけで学び，融合ではない（4.2 の多課題の文は融合に近い設定を述べる）
- A8 △：半教師ありの拡張で先生の soft label をラベル無し全体に付ける．先生はラベル付きの同じ分布で学ぶ．T・λ は固定か掃く．クラス均衡は無い
- A9 ×
- A10 ×：正解率と 100 回の分割の ± だけ
- A11 △：条件 i)〜iii) と因果の議論（4.3）で「いつ効くか」を述べる．相補性を測る量は無い

**6．本研究との差の書き方（案）**
- Lopez-Paz ら [ICLR 2016] は，特権情報の先生の soft label を温度 T と重み λ で生徒に学ばせる一般化蒸留を定義し，ラベルの少ない場面で先生が役立つと速さの議論で述べた．本研究はその枠を，凍結した特徴の上の線形の頭に限って，ソースのコーパスで学んだ融合の先生の弱いラベルの重みを K ごとに仮のターゲットで選び，得と損を切片と向きの式で説明する．

### Gaps
- 4.1 節の不等式は「Theorem」の形では書かれていないので，論文で「定理」と呼ばない
- 人工実験 4 の「0.56 ± 4%」は原文の表記（56% の誤植と思われる）

---

## 論文 3．Albanie, Nagrani, Vedaldi, Zisserman 2018, "Emotion Recognition in Speech using Cross-Modal Transfer in the Wild"

### Takeaway
顔の先生（FERPlus で学んだ SENet）の温度 2 の確率を，VoxCeleb のラベル無し動画で音声の生徒に交差エントロピーで写す．先生の予測は中立に偏り，均衡サンプリングは効かなかった．評価は RML・eNTERFACE で，生徒の 8 次元の出力の上に**アフィン写像＋softmax**（線形の頭）を 10 分割で学ぶ．蒸留するコーパス（VoxCeleb）と評価のコーパスは別で，ターゲットのラベルは全部使う．

### Cited Findings

**1．書誌と読んだ範囲**
- arXiv:1808.05561v1（2018-08-16）．PDF に「MM ’18, October 22–26, 2018, Seoul」「https://doi.org/10.1145/3240508.3240578」．頁 292–301 は Hajavi & Etemad の参考文献 [1] の記載．本文 1〜6 節を全部読んだ（◎）— [arXiv](https://arxiv.org/abs/1808.05561)

**2．設定**
- 先生：SENet-50 を VGGFace2 で事前学習し FERPlus（8 感情，注釈者 10 人の分布を交差エントロピーで合わせる）で微調整．FERPlus で 88.8 ± 0.3%（表 2）
- 生徒：VGG-M（VGGVox）を乱数初期化から，スペクトログラム（4 秒）で学ぶ．データは VoxCeleb の発話つき顔トラック：Train 118.5k，Heard-Val 4.5k，Unheard-Val 30.5k（表 3）．音声のラベルは一切使わない
- 蒸留する量：先生の枠ごとの予測を max-pool した 8 次元．「we used a cross entropy loss between the outputs of the networks after passing both both sets of predictions through a softmax function with temperature T」，T = 2．logit の回帰（L2）は「did not perform as well」
- クラス均衡：「We experimented with sampling speech segments in a manner that balanced the number of utterance level emotions seen by the student during training. However, in practice, we found that it did not have a significant effect on the quality of the learned student network」
- 先生のラベルの偏り：「the dataset is heavily skewed towards a small number of emotions (particularly neutral」，枠の遷移で中立になる「neutral transition-face phenomenon」（図 2）．生徒も「the distribution of predictions made by the student are dominated by the neutral class so the useful signal is primarily encoded through the relative soft weightings of each emotion」
- 評価：RML（720 標本，8 名，6 感情，6 言語）と eNTERFACE（42 名，6 感情）．「we treat the predictions as 8-dimensional embeddings and ... learning a single affine transformation (comprising a matrix multiply and a bias) followed by a softmax to map the 8 predicted student emotions to the target labels」，Noroozi らの 10 分割の手順

**3．示したこと（数値）**
- 表 5（正解率）：RML：でたらめ 16.7，生徒（音声）49.7 ± 5.4，先生（映像）72.6 ± 3.9，Noroozi ら（音声）65.3．eNTERFACE：生徒 34.3 ± 4.0，先生 48.3 ± 4.9，Noroozi ら 47.1
- 生徒の VoxCeleb 内の AUC：unheard 0.69，heard 0.71（同一性の近道を使っていない根拠）
- AFEW に先生を当て，AFEW の学習集合でアフィン写像を学ぶと 49.3%（主催者の基準 38.81%）
- 理論：無い

**5．穴（原文）**
- 「the teacher is far from perfect on both the RML and eNTERFACE benchmarks」
- 「EmoVoxCeleb used to perform the distillation may lack the distribution of emotions required for the student to fully capture the knowledge of the teacher」
- 「certain emotions are easier to detect from speech than faces, and vice versa [15], suggesting that the degree to which there is a redundant emotional signal across modalities may differ across emotions」
- 「Another drawback is the heavily unbalanced nature of the dataset where some emotions such as contempt and fear occur rarely」
- 今後：「explore unlabelled videos with a greater range of emotional diversity」「explore the use of proximal non-speech facial expressions as a supervisory signal」「scaling up to larger unlabelled datasets」
- ターゲットのラベルの本数を減らす実験は無い．先生の確率をターゲットのコーパス自体に当てる形ではない（第三のコーパス VoxCeleb で蒸留）

### Inferences

**4．軸との対応**
- A1 △：生徒の網は乱数初期化から学ぶが，評価は凍結した生徒の 8 次元出力の上の**アフィン写像＋softmax**（線形の頭）．本研究の「凍結したエンコーダ＋線形の頭」に，評価の段だけ一致する
- A2 ×，A3 ×：理論は無い
- A4 △：ターゲットでアフィン写像（行列＋バイアス）を学ぶので，コーパスごとの切片を暗黙に直しているが，分析はしていない
- A5 ×：RML・eNTERFACE の 10 分割で全ラベルを使う．K は無い
- A6 ○：先生は FERPlus，蒸留は VoxCeleb，評価は RML・eNTERFACE．3 つとも別のコーパスで，「the conditions of the evaluation datasets differ significantly from those on which the student network was trained」．ソースは 1 つ
- A7 △：顔の単モーダルの先生 → 音声の単モーダルの生徒．融合の先生ではない．運用時は音声だけ
- A8 △：学び直さない先生の確率（T = 2）をラベル無しデータ全体に付ける点は同じ．違い：データは第三のコーパスでターゲットではない．均衡サンプリングは試して効果なし．重みを K ごとに選ばない
- A9 ×
- A10 △：正解率（UAR ではない．評価データは「class-balanced」と書く）．10 分割の mean ± std．話者で分けたかは「the evaluation protocol of [51]」としか書かない（VoxCeleb の検証は同一性で分けた）．検定は無い
- A11 △：「our goal is to perform cross-modal learning by exploiting the redundancy of the signal that is common to both modalities」と，融合が使う相補な成分を対比する．混同行列で感情ごとの違いを見る．理論・測定の量は無い

**6．本研究との差の書き方（案）**
- Albanie ら [ACM MM 2018] は，顔の先生の確率を第三のラベル無しコーパスで音声の生徒に写し，評価のコーパスでは線形の頭を全ラベルで学んだ．先生の確率が中立に偏り，均衡サンプリングでは直らなかったと報告している．本研究は先生の確率をターゲットのコーパス自体に当て，列ごとの正規化と総重量 m_eff で偏りを扱い，ターゲットのラベルを K 本に絞る．

### Gaps
- 10 分割が話者で分かれているかは Noroozi ら [51] の手順を見ないと分からない（未確認）
- 「cross-modal distillation methods for speech emotion recognition on two standard datasets」の数値は，後続の論文と比べるときに同じ分割か確かめる

---

## 論文 4．Srinivasan, Huang, Kirchhoff 2022, "Representation Learning Through Cross-Modal Conditional Teacher-Student Training for Speech Emotion Recognition"

### Takeaway
音声（HuBERT）＋テキスト（BERT）の融合の先生の**埋め込み**を，音声だけの生徒（HuBERT の上 6 層を微調整）に L2 で写す．先生と生徒は同じラベル付きコーパスで学び，ラベル無しのターゲットは使わない．先生の予測の質（線形回帰の残差）で蒸留を外す条件つきの版は MSP-Podcast で効き，IEMOCAP では落ちた．

### Cited Findings

**1．書誌と読んだ範囲**
- arXiv:2112.00158．本文 1〜5 節を全部読んだ（◎）．PDF に会議名の記載は無い．ICASSP 2022 は前回のノートの記録で，会議ページは未確認 — [arXiv](https://arxiv.org/abs/2112.00158)

**2．設定**
- 課題：活性・感情価・支配の回帰（CCC）．MSP-Podcast v1.6（学習 34,280，検証 5,958，テスト 10,124 発話）と IEMOCAP（5 分割の話者独立交差検証，1 セッション評価，1 セッション検証，3 セッション学習）
- 先生：HuBERT の GRU と BERT の GRU を連結し線形射影（式 1）．生徒：音声だけの HuBERT（上 6 層を微調整）．蒸留：「minimizing the L2 loss between the two embeddings: L_T/S = ||e_AT − e_A||」（式 6），全体は L_CCC + λ L_T/S（式 7），λ = 30（「relatively insensitive in the range 3-50」，検証集合で調整）
- 条件つき蒸留：先生の予測をラベルに最小二乗で合わせた残差（式 8）が 2 SD を超える発話を蒸留の損失から外す（「∼12.7% of the training utterances were discarded」）．**ラベルを使う**
- 転写：MSP-Podcast は内製 ASR，IEMOCAP は正解の転写
- ターゲットのラベル：学習集合の全ラベル．少数ラベルの設定は無い

**3．示したこと（表 2・3 の CCC，活性／感情価／支配）**
- MSP-Podcast：HuBERT Base 0.733/0.485/0.640，HuBERT Large 0.752/0.547/0.674，BERT 0.317/0.563/0.290，融合の先生 0.765/0.690/0.683，HuBERT Base 微調整 0.730/0.533/0.644，T/S 0.738/0.590/0.650，条件つき T/S 0.757/0.627/0.671
- IEMOCAP：融合の基準 [16] 0.594/0.446/0.485，HuBERT Base 0.663/0.527/0.530，BERT 0.463/0.576/0.442，融合の先生 0.668/0.648/0.537，T/S 0.667/0.582/0.545，条件つき T/S 0.645/0.481/0.528（**悪化**）
- 「there is complementary information between the audio and text modalities, even when using a strong speech representation like the HuBERT model」

**5．穴（原文）**
- 「on IEMOCAP, ... conditioning T/S leads to performance degradation, possibly due to the small size of the dataset」「removing utterances using conditional teacher-student training led to overfitting」
- 今後：「explore using a different layer or a weighted combination of layers」「contrastive learning of the information shared by different modalities」「injecting lexical information into generic speech representations before fine-tuning」
- コーパスをまたぐ評価・ラベル無しデータ・少数ラベルは扱わない

### Inferences

**4．軸との対応**
- A1 ×：HuBERT を微調整．線形の予測網は最上段にあるが，エンコーダは凍結ではない
- A2 ×，A3 ×，A4 ×
- A5 ×：全ラベル
- A6 ×：同じコーパスの中．2 つ目のコーパス（IEMOCAP）も同じコーパス内で先生を学び直す
- A7 ○：融合の先生（音声＋テキスト）→ 音声だけの生徒．運用時は音声だけ．テキストが特権情報
- A8 ×：蒸留するのはラベル付きの学習集合上の埋め込み．ターゲットのラベル無し全体の弱いラベルではない．先生はコーパスごとに学び直す．クラス均衡は無い（回帰）．標本ごとの質の判定はラベルを使う
- A9 ×
- A10 △：IEMOCAP は話者独立の 5 分割．CCC．検定は無い
- A11 △：相補性は「融合すると感情価が上がる」ことで示す．理論・量は無い

**6．本研究との差の書き方（案）**
- Srinivasan ら [ICASSP 2022] は，音声＋テキストの融合の先生の埋め込みを，同じコーパスのラベル付きデータ上で音声の生徒に写し，先生の予測の質で蒸留を外す条件を加えた．本研究は融合の先生をソースのコーパスだけで学び，ターゲットのラベル無し発話に当てた確率を弱いラベルにする．先生の質による重みづけではなく，クラスごとの総量をそろえた上で総重量を K ごとに選ぶ．

### Gaps
- 会議名（ICASSP 2022）と頁は本文に無い．引く前に IEEE Xplore で確かめる
- 「the best reported baseline on IEMOCAP is [11], but their partition ... is not speaker-independent」という脚注があり，IEMOCAP の数値は分割の違いに注意する

---

## 論文 5．Pendyala, Morgado, Sethares 2025, "Leveraging Unlabeled Audio-Visual Data in Speech Emotion Recognition using Knowledge Distillation"（LiSER）

### Takeaway
ラベル無しの第三のコーパス（MSP-Face，46.55 時間）上で，音声の先生（ターゲットのラベルで LoRA 微調整した HuBERT）と顔の先生（FERV39k で学んだ S2D）の softmax 出力を，軽い音声の生徒の別々の頭に MAE で写す．ラベル付きの損失と同時に解き，λ は検証集合で選ぶ．話者分離の 5 分割で UAR・WAR を報告．ラベルの割合（0.15〜1）を振る図がある．

### Cited Findings

**1．書誌と読んだ範囲**
- arXiv:2507.00055v1（2025-06-26）．本文 1〜4 節を全部読んだ（◎）．Interspeech 2025 は前回のノートの記録で，PDF には会議名が無い（書式は Interspeech 形式）— [arXiv](https://arxiv.org/abs/2507.00055)

**2．設定**
- データ：ラベル無しの MSP-Face（46.55 時間，386 話者），ラベル付きの RAVDESS（1,440 本，24 名，8 感情，音声部だけ），CREMA-D（7,442 本，91 名，6 感情，音声部だけ）
- 先生：音声は hubert-base-ls960 を「utilizing labeled speech samples from the same dataset used to train the student model」で LoRA 微調整（80% で学び 20% で選ぶ）．映像は S2D（FERV39k の 7 感情）
- 生徒：10.5 万パラメータの CNN-LSTM，頭は 3 つ（教師あり K_S，音声蒸留 K_S，映像蒸留 K_V）．映像の先生のクラス集合（7）はターゲット（8 か 6）と違い，別の頭で受けるのでラベルの対応づけはしない
- 損失：ラベル付きは交差エントロピー（式 1），ラベル無しは softmax の MAE（式 2・3），ミニバッチ損失（式 4）は両方の和を N_l + N_u で割る．λ_vd, λ_sd ∈ {0.1, 0.5, 1, 5, 10} を検証集合で選ぶ．確信度版（式 5）は先生の最大確率で標本ごとに重みづけ
- 評価：「five-fold cross-validation protocol ... ensuring no overlap in speakers」，UAR と WAR

**3．示したこと（表 2，UAR/WAR）**
- RAVDESS：蒸留なし 0.517/0.535，映像 0.534/0.547，音声 0.545/0.556，両方 0.556/0.570，確信度つき 0.595/0.611（相対で +15.09%）
- CREMA-D：蒸留なし 0.551/0.55，映像 0.576/0.575，音声 0.576/0.575，両方 0.584/0.583，確信度つき 0.578/0.576（+5.99% → +4.9% に減る）
- 表 3（RAVDESS）：MAE を交差エントロピーに替えると落ちる（映像 0.523/0.54，音声 0.54/0.546）．2 段階学習（先にラベル無し，後でラベル付き）は 0.464/0.485・0.511/0.528 と大きく落ちる
- 図 2（CREMA-D，ラベルの割合 0.15/0.3/0.5/1）：「the student model trained with all data from MSP-Face and only half of the labeled data performs better than the model that used the whole labeled dataset in the no-dstl scenario」
- 理論：無い

**5．穴（原文）**
- 限界の節は無い．「no reported literature investigates distillation using unlabeled audio-visual data for SER」と位置づける
- 確信度の重みは RAVDESS で効き CREMA-D では「the improvement slightly decreases from 5.99% to 4.9%」
- ドメインの違い（MSP-Face は実環境，RAVDESS・CREMA-D は演技）は本文で扱わない．映像の先生のラベル集合の違いは別の頭で受け流すだけ

### Inferences

**4．軸との対応**
- A1 ×：小さな CNN-LSTM を乱数初期化から学ぶ
- A2 ×，A3 ×，A4 ×
- A5 △：ラベルの割合（学習集合の 0.15〜1）を振る．単位は割合で，クラスあたりの本数ではない．CREMA-D の 0.15 は 1 クラス 100 本を超える
- A6 △：蒸留のデータは第三のコーパス．音声の先生はターゲットのラベルで学ぶ．評価はコーパス内．映像の先生だけが別コーパス（FERV39k）で学んだもの
- A7 △：単モーダルの先生 2 本（音声・顔）で，融合の先生ではない．運用時は音声だけ
- A8 △：先生の確率をラベル無し全体に別の頭で写す．音声の先生はターゲットで学び直す．λ は検証集合で選ぶ（K ごとではない）．標本ごとの確信度の重みはあるがクラス均衡は無い
- A9 ×
- A10 ○：話者で分けた 5 分割，UAR と WAR．検定は無い
- A11 △：「Utilizing both audio and visual modalities during the distillation process provides greater performance improvements compared to relying solely on one modality」と実証．理論は無い

**6．本研究との差の書き方（案）**
- Pendyala ら [Interspeech 2025] は，ターゲットのラベルで微調整した音声の先生と顔の先生の確率を，第三のラベル無しコーパス上で軽い音声の生徒に写し，ラベルの割合を減らしても蒸留が効くことを示した．本研究は先生をソースのコーパスだけで学んでターゲットのラベル無し発話に当て，クラスあたり K 本という単位で得の曲線を測る．

### Gaps
- 会議名は本文に無い．引く前に ISCA の頁で確かめる
- 図 2 の数値は図からしか読めず，表には無い

---

## 論文 6．Hajavi & Etemad 2023/2024, "Audio Representation Learning by Distilling Video as Privileged Information"

### Takeaway
映像の先生の**埋め込み**を，同じラベル付きデータ上で音声の生徒のエンコーダに写し，生徒の頭は正解ラベルで学ぶ．soft label の蒸留（Albanie 型）と多課題（特権を再構成）を比較相手にし，α（模倣の重み）を増やすと相手は落ちるが提案法は落ちないと示す．IEMOCAP では先生が生徒より弱く，得は 0．少数ラベル・ラベル無し・コーパス間は無い．

### Cited Findings

**1．書誌と読んだ範囲**
- arXiv:2302.02845v1（2023-02-06）．IEEE Transactions on Artificial Intelligence，DOI 10.1109/tai.2023.3243596．LiSER の参考文献 [3] は「vol. 5, no. 1, pp. 446–456, 2024」と記す（冊子は 2024 年）．本文 I〜V 節を全部読んだ（◎）— [arXiv](https://arxiv.org/abs/2302.02845)

**2．設定**
- 課題：話者認識（VoxCeleb の映像つき版，21,819 本，1,211 名，70/10/20）と SER（RAVDESS：24 名，7 感情，発話＋歌，2,452 本，leave-one-subject-out；IEMOCAP：4 感情，5 分割）
- 先生：映像の網（VGG16／ResNet34／SEResNet34，順序版は＋BiLSTM）を同じデータのラベルで学ぶ．「During the training of the student, the teacher network is frozen」
- 生徒：音声のスペクトログラムの CNN（＋BiLSTM）．勾配は ∇_s = (1−α)∇L_Y + α∇L_E（式 8）．L_E は生徒のエンコーダの埋め込みと先生のピーク枠の埋め込みの距離．頭は L_Y（正解ラベル）だけ．順序版は集約層で合わせる版もある（式 13）
- 「While the previous methods proposed for LUPI use soft-labels generated by the teacher, in our proposed method we use embeddings learned by the teacher」
- α = 0.5 で報告．「our method has the highest impact when α is either 0.5 or 0.6」
- 比較相手：「Soft-label distillation」（[1, 30, 32] の形）と「Multitask Learning」（特権情報を復元する分岐）

**3．示したこと（数値）**
- 表 VI（RAVDESS，unweighted accuracy，先生／蒸留前／蒸留後）：VGG 80.5／51.0／52.3，SEResNet 81.0／56.6／59.5（+5.12%），SEResNet+BiLSTM+AGG 85.3／57.6／59.9
- 表 VII（IEMOCAP）：先生 60.8〜65.5 は生徒 68.6〜71.8 より弱く，ΔAcc は全て 0．「despite the teacher having a lower performance than the student, the proposed method does not negatively affect the performance of this network」
- 図 5・6：soft label と多課題の相手は α を上げると落ちる．「they do not improve the performance of deep neural networks when training is done using both modalities but only audio is available at inference」
- 表 IV・V（VoxCeleb）：識別 73.7 → 76.4 など，EER 5.33 → 3.91 など
- 理論：無い

**5．穴（原文）**
- 今後：「study the use of generative models such as Generative Adversarial Networks and Variational Autoencoders for creating the embeddings from the teacher model with better domain adaptation and generalization」
- ラベル無しデータ・少数ラベル・コーパス間は扱わない．先生が弱いと得が出ない（IEMOCAP）

### Inferences

**4．軸との対応**
- A1 ×：深層の網を学ぶ
- A2 ×，A3 ×，A4 ×
- A5 ×：全ラベル
- A6 ×：コーパス内（LOSO か 5 分割）
- A7 △：映像の単モーダルの先生 → 音声の生徒．融合の先生ではない．運用時は音声だけ
- A8 ×：埋め込みをラベル付きの学習データ上で写す．ラベル無しのターゲットも確率も無い．α は 0.5 固定（掃いた図はある）
- A9 ×
- A10 △：RAVDESS は LOSO．IEMOCAP の 5 分割が話者で分かれているかは記述が無い．unweighted accuracy（UAR に相当する語だが定義は書いていない）．検定は無い
- A11 △：先生が生徒より弱いと得が 0 で損も無い，という実証．α の掃引．理論は無い

**6．本研究との差の書き方（案）**
- Hajavi と Etemad [IEEE TAI 2024] は，映像を特権情報として先生の埋め込みを音声の生徒のエンコーダに写し，soft label の蒸留は正解ラベルがある場面では効かないと報告した．本研究は soft label（確率）を使うが，それをラベル付きの標本にではなく，ソースの先生が付けたターゲットのラベル無し発話の弱いラベルとして使う点で設定が違う．

### Gaps
- 表 VII の列の並びが本文の説明と合いにくい（「AccT − AccS」の列に負の値，ΔAccS の列に 0）．ΔAcc = 0 の読みは本文の記述に合わせた
- 掲載年は DOI（2023）と冊子（2024, vol. 5, no. 1）で違う．引くときは冊子の年を使う

---

## 論文 7．Xue, Ren, Gao, Zhao 2021, "Multimodal Knowledge Expansion"（MKE）

### Takeaway
ラベル付きの単モーダルのデータで学んだ先生（学び直さない）が，同じ分布のラベル無し多モーダルデータに疑似ラベルを付け，多モーダルの生徒を一貫性の正則化つきで学ぶと，生徒が先生を超える（RAVDESS で 80.33% → 91.38%）．定理 1 は，条件付き独立と拡張条件（c1c2 > 5）の下で生徒の誤差 ≤ 4·先生の誤差／(c1c2 − 1) + 4μ を与える．ドメインのずれ・少数ラベル・クラス均衡は無い．

### Cited Findings

**1．書誌と読んだ範囲**
- arXiv:2103.14431．本文 1〜5 節を全部読んだ（◎）．補足資料（ラベル無しデータの量，先生，硬い／柔らかいラベルの消去実験）は PDF に無く未読．ICCV 2021 は前回のノートの記録で PDF には無い — [arXiv](https://arxiv.org/abs/2103.14431)

**2．設定**
- アルゴリズム 1：(1) 単モーダルの先生 θ_t* を D_l = {(x^α_i, y_i)} で学ぶ，(2) D_u = {(x^α_i, x^β_i)} に ỹ_i = f_t(x^α_i; θ_t*) を付ける，(3) 多モーダルの生徒を L_pl + γ L_reg（式 3〜5）で学ぶ．実装では「to avoid introducing and tuning one extra hyperparameter γ ... we train the student network with θ_s* = argmin (1/M) Σ l_cls(ỹ_i, T(f_s(x^α_i, x^β_i)))」（変換 T を通した予測を疑似ラベルに合わせる）
- 疑似ラベル：「l_cls: cross entropy loss for hard ỹ_i and KL divergence loss for soft ỹ_i」
- RAVDESS：「randomly split as 2:8 for D_l and D_u and 8:1:1 as train / validation / test for D_u」，画像が α（先生），音声が β．3 層 CNN × 2 ＋ 3 層 MLP．T は dropout 0.5．話者で分けた記述は無い
- その他：TwoMoon（30 ラベル／270 ラベル無し／200 テスト），NYU Depth V2（RGB の先生 795 枚 → RGB-D の生徒，ラベル無し 1.5K 枚），AudioSet 3.7K → VGGSound 3.7K

**3．示したこと（数値と理論）**
- 表 3（RAVDESS，正解率 val/test）：先生 79.67/80.33，単モーダルの生徒 79.01/77.79，NOISY student 82.54/83.09，多モーダルの生徒（正則化なし）88.73/89.28，MKE 90.61/91.38，正解ラベルの上限 97.46/97.35
- 表 2（TwoMoon）：先生 68.00，単モーダルの生徒は最大 72.80，MKE は最大 83.15，上限 86.95〜88.05
- 表 4（NYU，mIoU）：先生 44.15，Gupta ら 45.65，CMKD 45.25，MKE 48.88．表 5（mAP）：先生 0.345，MKE 0.427，上限 0.434
- 定理 1：仮定 1（X^α と X^β がそれぞれ (ā, c1)・(ā, c2) の拡張条件を満たし，1 < min(c1, c2) ≤ max(c1, c2) ≤ 1/ā，c1c2 > 5），仮定 2（「Conditioning on ground truth labels, X^α and X^β are independent」），Wei らの仮定 3.3 の下で，Err(f_s(x^α, x^β)) ≤ 4·Err(f_t(x^α))／(c1c2 − 1) + 4μ．単モーダルの生徒は 4·Err／(c1 − 1) + 4μ．「the role of multimodality is to increase the expansion factor from c1 to c1c2」
- 「multimodal inputs serve as a strong regularization, which helps correct inaccurate pseudo labels」

**5．穴（原文）**
- 限界の節は無い．定理は条件付き独立を仮定し，誤差の上界は先生の誤差の定数倍で，ラベルの本数 n の関数ではない
- 「SSL ... leveraging unlabeled data of the same source」と対比するが，MKE 自身も D_l と D_u は同じデータの無作為分割で，ドメインのずれは無い
- 生徒は運用時に両方のモダリティが要る

### Inferences

**4．軸との対応**
- A1 ×：3 層 CNN を学ぶ
- A2 ×，A3 ×（誤差の上界は先生の誤差の倍率で，n に換算しない），A4 ×
- A5 ×：ラベル付きは RAVDESS の 20%（約 288 本）．クラスあたりの K ではない．ラベル無しの量の消去実験は補足（未読）
- A6 ×：同じコーパスの無作為分割
- A7 △：単モーダルの先生 → 多モーダルの生徒（本研究の Case B の向き）．運用時は多モーダル
- A8 ○：学び直さない先生の確率（硬い／柔らかい）をラベル無し全体に付ける．クラス均衡・重みの選択は無い（γ を避ける）
- A9 ×
- A10 ×：正解率．無作為分割で話者分離の記述が無い．検定は無い
- A11 ○：条件付き独立と拡張係数 c1c2 で，生徒が先生を超える条件を理論で与える．測定はしない

**6．本研究との差の書き方（案）**
- Xue ら [ICCV 2021] は，単モーダルの先生の疑似ラベルで多モーダルの生徒を学ぶと生徒が先生を超えることを，条件付き独立の下の拡張条件で説明した．本研究は向きが逆の Case A（融合の先生 → 単モーダルの生徒）と同じ向きの Case B の両方を，コーパスをまたぐ少数ラベルで測り，得の条件を相補性の量で記述する．MKE の「生徒が疑似ラベルを直す」観察は Case B の説明に引ける．

### Gaps
- 補足資料（硬いラベルと柔らかいラベルの比較，ラベル無しの量）は未読．論文に書く前に ICCV 版の補足を見る
- ICCV 2021 の書誌は本文に無い（要確認）
- RAVDESS の分割が話者をまたぐ可能性が高く，91.38% は話者分離の数値と比べない

---

## 論文 8．Du ほか 2021, "Improving Multi-Modal Learning with Uni-Modal Teachers"（UMT）

### Takeaway
合同学習の融合では弱いモダリティのエンコーダが学ばれない（modality failure）ことを示し，同じラベル付きデータで事前学習した単モーダルのエンコーダの**特徴**を，融合の生徒の各エンコーダに L2（λ = 50）で写す．VGGSound で 49.46 → 53.46．理論は Allen-Zhu & Li の多視点の枠への類推と予想（未証明）．ラベル無し・少数ラベル・ドメインのずれは無い．

### Cited Findings

**1．書誌と読んだ範囲**
- arXiv:2106.11059v1（2021-06-21）．PDF に「Preprint. Under review.」．G²D の参考文献 [7] も会議名なしで引く．正式な掲載先は見つからず，後続の ICML 2023 論文（Du ほか "On Uni-Modal Feature Learning in Supervised Multi-Modal Learning"）が UMT と UME を含む．本文 1〜7 節を全部読んだ（◎）— [arXiv](https://arxiv.org/abs/2106.11059)

**2．設定**
- データ：VGGSound（約 175k 学習，15k テスト，309 クラス），NYU Depth V2（795/654）
- 先生：単モーダルの学習問題（式 2）で学んだエンコーダ φ*_s．生徒：乱数初期化の融合（後期融合，連結）．損失 L_UMT = L(θ, W1, W2) + λ Σ_s ||φ*_s(X_i) − φ_s(W_s, X_i)||²（式 3），MSE，λ = 50
- 「matching the fusion output to soft labels of well-trained uni-modal networks may not help, since both networks have already achieved 100% training accuracy. We instead ... distill the features」

**3．示したこと（数値と理論）**
- 表 1（VGGSound，凍結したエンコーダ＋分類器で評価）：合同学習の映像エンコーダはテスト 15.86%・学習 34.69%（単独学習なら 99.47%）．UMT は音声 46.66，映像 24.84
- 表 2（VGGSound top-1）：合同 49.46 ± 0.28，Dropout 49.83，映像だけ蒸留 49.55，自己蒸留 49.86，事前学習＋微調整 50.81，Modality Dropout 51.37，Gradient-Blending 50.39，UMT 53.46 ± 0.49
- 表 3（NYUv2 mIoU）：ESANet 38.59 → 40.45（乱数初期化），48.48 → 49.14（ImageNet）
- 理論：仮説 1（不均衡な多視点）→ 系 1（弱いモダリティの特徴を学べない）．予想 1「the encoders trained by the UMT method can at least learn all the features learned by the corresponding teacher encoders」，「We leave the proof of this conjecture as a future direction」．「by setting λ → ∞, ... the learning problem reduces to train a linear classifier over pre-trained features」

**5．穴（原文）**
- 「We leave the proof of this conjecture as a future direction」
- 今後：「can we alleviate the modality failure by carefully designing the encoders, and whether a similar problem also exists in uni-modal learning?」
- ラベル無しデータ・少数ラベル・ドメインのずれ・クラス均衡は無い

### Inferences

**4．軸との対応**
- A1 ×：エンコーダを学ぶ．λ → ∞ の極限が「凍結した特徴の上の線形分類器」だと述べるだけ
- A2 ×，A3 ×，A4 ×
- A5 ×，A6 ×
- A7 △：単モーダルの先生 → 融合の生徒．運用時は多モーダル
- A8 ×：ラベル付きの同じデータ上で特徴を写す
- A9 △：生徒のエンコーダの特徴を先生の特徴に L2 で引き寄せる（特徴空間の L2-SP に相当）．線形の頭をソースの頭に寄せる本研究の罰則とは対象が違う
- A10 ×：top-1 と複数回の ± だけ
- A11 △：modality failure を勾配の式で説明し，凍結したエンコーダの評価で測る．相補性の量は無い

**6．本研究との差の書き方（案）**
- Du ら [arXiv 2021] は，合同学習で弱いモダリティが学ばれないことを示し，単モーダルの先生の特徴を融合の生徒に写した．本研究の Case B は，単モーダルの生徒の出力の積を融合の線形の生徒の参照にする点と，先生がソースのコーパスで固定され生徒がターゲットの K 本で学ぶ点で違う．

### Gaps
- UMT 2021 が査読つきで出たかは確かめられなかった．引くなら arXiv か，ICML 2023 の論文に置き換える

---

## 論文 9．Liu, Gu, Zhang, Wang, Li, Ande, Lin 2026, "Dynamic Mutual Refinement for Multimodal Speech Emotion Recognition: A Bidirectional Distillation and Inverse Curriculum Learning Approach"（DAMR）

### Takeaway
多モーダルの模型と単モーダルの模型の間で，EMA で測った能力に応じて重みを変える相互蒸留と，難→易の逆カリキュラムを組む．MELD と IEMOCAP で基準より IEMOCAP で 5.54% 上がると抄録に書く．**本文は読めていない**（MDPI のサイトが 403）．コーパス間か，凍結か，指標は何かは分からない．

### Cited Findings

**1．書誌と読んだ範囲**
- Crossref：Electronics 15(20):4690，2026-10-12 公開，DOI 10.3390/electronics15204690，著者は Linsong Liu, Yu Gu, He Zhang, Shuang Wang, Chenyu Li, Quan Ande, Haixiang Lin．抄録だけ（○）．読めなかった節：全部（MDPI の HTML・PDF は curl と WebFetch で「Access Denied」）— [Crossref](https://api.crossref.org/works/10.3390/electronics15204690)，[DOI](https://doi.org/10.3390/electronics15204690)

**2．設定（抄録から）**
- 「classification ambiguity, where multimodal models may misclassify samples that unimodal models correctly identify, and vice versa」を減らす
- 「(1) dynamic mutual distillation (DMD), which employs an EMA-based capability assessment mechanism and competitive dynamic weighting to adaptively adjust knowledge transfer between models; (2) adaptive inverse curriculum learning (AICL), which implements a hard-to-easy learning schedule ...; (3) an integrated optimization framework」
- 「while preserving their core architectures」（模型の構造は変えない）
- データ：MELD と IEMOCAP

**3．示したこと（抄録から）**
- 「achieving a notable 5.54% improvement over the baseline on IEMOCAP」．指標・分割・エンコーダは不明

**5．穴**
- 本文が読めないので，限界と今後の課題は不明．抄録に「cross-corpus」の語は無い

### Inferences

**4．軸との対応**
- A1 ？，A2 ×（抄録に理論の記述なし），A3 ×，A4 ×，A5 ？（抄録に少数ラベルの記述なし），A6 ？（MELD と IEMOCAP を別々に使う記述で，コーパス間とは書かない），A7 △（多モーダル ↔ 単モーダルの双方向），A8 ？，A9 ？，A10 ？，A11 △（「classification ambiguity」を動機にし，EMA で能力を測る）

**6．本研究との差の書き方（案）**
- Liu ら [Electronics 2026] は，同じ学習の中で多モーダルと単モーダルの模型を相互に蒸留し，能力に応じて重みを変える．本研究の二段構え（先に単モーダル，次に積を参照に融合）は同時の相互蒸留ではなく，先生はソースのコーパスで固定され，ターゲットは K 本のラベルしか持たない．**本文を読んでから引く**．

### Gaps
- 本文未読．コーパス間か否か，話者分離，指標，エンコーダの扱いは全て未確認．引用の前に本文（MDPI の頁をブラウザで開く）で確かめる

---

## 論文 10．Rakib & Bagavathi 2025, "G²D: Boosting Multimodal Learning with Gradient-Guided Distillation"

### Takeaway
同じラベル付きデータで学んだ単モーダルの先生の特徴（L2）と logit（KL，T = 1）を融合の生徒に写し（α = β = 1），先生の確信度（正解ラベルの確率）の低いモダリティから順に単独で学ぶ（SMP）．CREMA-D で合同 67.47 → 85.89．少数ラベル・ラベル無し・ドメインのずれ・理論は無い．

### Cited Findings

**1．書誌と読んだ範囲**
- arXiv:2506.21514v3（2025-10-17）．本文 1〜5 節と付録 A.1〜A.7 を全部読んだ（◎）．ICCV 2025 は前回のノートの記録で，PDF には会議名が無い（A.5 に「In response to reviewer feedback」とある）— [arXiv](https://arxiv.org/abs/2506.21514)

**2．設定**
- データ：CREMA-D（6 感情，6,027/669/745），AV-MNIST，VGGSound，UR-Funny（音声・映像・テキスト），IEMOCAP（音声・映像・テキスト，Transformer エンコーダ），MIS-ME（回帰）
- 損失（式 4）：L_G2D = L_S + α Σ_m L_feat^m + β Σ_m L_logit^m．L_feat は L2，L_logit は KL(σ(l_t^m) || σ(l_s))．「we used a temperature of 1.0 in the KL Divergence without further softening」，α = β = 1
- SMP：先生の「batch-wise average of their softmax」の正解ラベルの確率 ρ（式 5，**ラベルを使う**）で順位を付け，最も弱いモダリティだけ τ_1 エポック学び，最後に全部（式 6〜8）
- 学習：SGD，バッチ 16，乱数の種 999，後期融合（表 5 で最良）

**3．示したこと（表 1，正解率）**
- CREMA-D の融合：合同 67.47，MSES 60.99，MSLR 64.42，AGM 78.48，PMR 59.13，OGM-GE 72.18，MLA 79.70，MMPareto 75.13，ReconBoost 79.82，DLMG 83.62，UMT 67.61，G²D 85.89．先生は音声 61.69，映像 76.48．G²D の中の音声 56.45，映像 72.72
- AV-MNIST 69.77 → 73.03，VGGSound 50.97 → 53.82，UR-Funny（A-V-TXT）62.58 → 65.49，IEMOCAP（表 11，欠損 0%）75.51 → 77.19
- 表 4：SMP だけを合同学習に足すと 80.78，G²D の損失だけで 78.63，両方で 85.89．表 12：(α, β) = (0, 0) 80.78，(1, 1) 85.89．表 7：τ = (0, 150) 78.63 → (150, 150) 85.89
- 理論：無い（仮説 1 を実験で確かめる）

**5．穴（原文）**
- 限界の節は無い．τ_j を手で決める（表 7）．「For all experiments, we set the random seed to 999」で 1 回の学習
- ラベル無しデータ・少数ラベル・ドメインのずれは無い

### Inferences

**4．軸との対応**
- A1 ×，A2 ×，A3 ×，A4 ×，A5 ×，A6 ×
- A7 △：単モーダルの先生 → 融合の生徒．運用時は多モーダル
- A8 ×：ラベル付きの同じデータ上で特徴と logit を写す
- A9 △：UMT と同じく特徴の L2（エンコーダを先生に寄せる）
- A10 ×：正解率，1 つの種，検定なし，話者の記述なし
- A11 △：先生の確信度の比（図 6），モダリティの隙間，特徴の整列を測る．理論は無い

**6．本研究との差の書き方（案）**
- Rakib と Bagavathi [ICCV 2025] は，単モーダルの先生の特徴と logit を融合の生徒に写し，弱いモダリティから順に学ばせることで同じコーパスの融合を良くした．本研究の Case B は，ソースのコーパスで固定した先生と，ターゲットの K 本で学んだ単モーダルの生徒の積を参照にする点で設定が違う．

### Gaps
- ICCV 2025 の書誌は本文に無い（要確認）
- CREMA-D の分割が話者で分かれているかは書いていない

---

## 余力分：PriCon 2025（◎），Guo 2026（△），KBS 2021（△）

### Takeaway
PriCon は「実験室→実環境」を**モダリティの有無**の違いとして扱い（コーパスは同じ），特権の先生の KL と交差エントロピーを α で混ぜる LUPI の生徒を，参加者で分けた 5 分割で評価する．Guo 2026 と KBS 2021 は本文も抄録も取れず，書誌だけ確かめた（KBS の巻の食い違いは解けた）．

### Cited Findings

**PriCon（Pinitas, Makantasis, Yannakakis 2025, "Privileged Contrastive Pretraining for Multimodal Affect Modelling"）**
- 書誌：arXiv:2508.03729v1（2025-07-30），PDF の謝辞に「Proceedings of the 27th International Conference on Multimodal Interaction (ICMI ’25), https://doi.org/10.1145/3716553.3750766」．本文 1〜7 節を全部読んだ（◎）— [arXiv](https://arxiv.org/abs/2508.03729)
- 設定：生徒は枠（映像）だけの CNN．先生は特権の特徴だけ（T_p）か特徴＋枠の融合（T_f）．損失 L_P = (1−α) L_CE(f(x), y) + α L_KL(g(x̃), f(x))（式 1）．α ∈ {0.25, 0.5, 0.75, 1.0} を検証集合で選ぶ．PriCon は先生を教師あり対照学習で事前学習してから LUPI に使う．RECOLA（公開 23 名）と AGAIN（120 名，3 ゲーム）．「we make sure that data from the same participant is either in the training or the test set, not both」，5 分割，2 値（高／低）の正解率，対応のある t 検定
- 「in-vitro」と「in-vivo」はコーパスの違いではなく，学習時に使えるモダリティ（研究室の特徴・生理信号）と運用時のモダリティ（カメラの枠）の違いである．評価はコーパス内
- 表 1（RECOLA 活性 1 秒，LUPI／PC）：基準 E 59.41／60.60，S_p 63.61／65.27，S_f 60.75／63.74，T_p 69.24／71.41，T_f 63.04／64.60．感情価は 60〜65 で動かない
- 「The accuracy of the T_p, T_f and E models in the PC setting is obtained by applying a linear probe on top of the learned representations」（先生と基準は凍結した表現の上の線形の頭．生徒は端から端まで）
- 限界（原文）：「valence results show only marginal gains」「The study employed an exhaustive search protocol to fine tune the α hyperparameter ... it may not generalise well to other datasets」「Tuning was done for 1-second time windows, assuming the optimal α value is consistent across all windows」「the manual selection of privileged features represents another limitation」．今後：「semi-supervised and self-supervised learning paradigms」「combining PriCon with few-shot learning strategies [27, 41] could facilitate robust affect modelling in low-data regimes」
- 軸：A1 △（先生と基準が線形プローブ，生徒は深層），A5 ×（全ラベル），A6 ×（コーパス内），A7 ○（融合の先生 T_f → 単モーダルの生徒．運用時は枠だけ），A8 ×（ラベル付き上の KL），A9 ×，A10 △（参加者分離，t 検定，2 値の正解率），A11 ×

**Guo, Jiao, Guo, Song 2026, "Multimodal Helps Unimodal: Frequency-Aware Distillation for Cross-domain Few-shot Egocentric Action Recognition"**
- Crossref：Information Fusion 137:104636（発行日は 2027-01 と記録），PII S1566253526005129．Elsevier の頁は WebFetch で 403，Crossref と OpenAlex に抄録なし（△）— [Crossref](https://api.crossref.org/works/10.1016/j.inffus.2026.104636)
- 題名から MM-CDFSL（Hatano ほか ECCV 2024）と同じ設定（多モーダル → 単モーダル，ドメインをまたぐ少数ショット）の続きと推定するが，中身は未確認

**Zhang, Chen, Chen, Li, Wu, Li, Zhu 2021, "Combining cross-modal knowledge transfer and semi-supervised learning for speech emotion recognition"**
- Crossref：Knowledge-Based Systems 229:107340，2021-10（△．抄録は Crossref・OpenAlex に無く，著者の頁の PDF と ScienceDirect は 403）— [Crossref](https://api.crossref.org/works/10.1016/j.knosys.2021.107340)
- 前回のノートの「229 巻 107340 と 346 で食い違い」は，346 が datalearner の内部の通し番号で巻ではないと分かった．巻は 229 で確定
- 検索結果の要約（△）：映像から音声への知識の移し替え＋一貫性正則化の半教師あり学習，重みつき融合と「consistent & random」，CH-SIMS と IEMOCAP．ラベルの量・コーパス間か否かは不明

### Inferences
- PriCon の「実験室→実環境」は本研究の「コーパス間」と語が似ているが中身は違う（モダリティの有無）．関連研究で引くときは「学習時だけ使えるモダリティ」の文脈に限る
- PriCon が今後の課題に「few-shot ... low-data regimes」を挙げているので，本研究の K 本の設定はこの群で空いている

### Gaps
- Guo 2026 と KBS 2021 は本文が無い．部品 4（多モーダル → 単モーダルの少数ショット，映像で疑似ラベルを直す）に近い可能性が残るので，引用の前に本文を開く

---

## 10 本 × A1〜A11 の表

○＝そのまま持つ，△＝近いものを持つ，×＝持たない，？＝本文が読めず不明．各欄は記号と 1 行の根拠．

| 論文 | A1 線形の頭 | A2 切片と向き | A3 本数に換算 | A4 合併の切片 | A5 K ≤ 10・K を振る | A6 コーパス間・ソース数 | A7 マルチモーダル・融合→単 | A8 ソース先生の弱いラベル | A9 出発点への L2 | A10 評価 | A11 相補性 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 Gupta 2016 | ×：CNN を乱数初期化から学ぶ | ×：理論なし | ×：理論なし | ×：理論なし | ×：NYUD2 の全ラベルで微調整 | △：先生 ImageNet，対データ UCF101，評価 JHMDB．ソース 1 | △：RGB 先生 → 深度／流の生徒．融合の先生ではない | △：学び直さない先生をラベル無しに当てるが特徴．均衡・K なし | ×：初期化のみ | ×：mAP | △：融合の得で相補性を測る |
| 2 Lopez-Paz 2016 | △：人工データはロジスティック回帰．凍結エンコーダではない | ×：なし | △：n の速さの議論「低データで先生が効く」 | ×：なし | △：n = 200〜500（総数），CIFAR 300 枚 | ×：同じ分布 | △：特権 x* の先生．融合ではない | △：半教師あり拡張でラベル無し全体に soft label．T・λ 固定／掃引．均衡なし | ×：なし | ×：正解率，100 分割の ± | △：条件 i)〜iii) と因果の議論 |
| 3 Albanie 2018 | △：評価は凍結した 8 次元出力のアフィン写像＋softmax | ×：なし | ×：なし | △：ターゲットでアフィン（切片を含む）を学ぶが分析なし | ×：10 分割で全ラベル | ○：FERPlus → VoxCeleb → RML／eNTERFACE．ソース 1 | △：顔の先生 → 音声の生徒．運用は音声だけ | △：先生の確率（T=2）を第三コーパス全体に．均衡サンプリングは効かず | ×：なし | △：正解率，10 分割 ± std，話者分離は不明 | △：冗長性の仮説と混同行列 |
| 4 Srinivasan 2022 | ×：HuBERT 上 6 層を微調整 | ×：なし | ×：なし | ×：なし | ×：全ラベル | ×：コーパス内 | ○：音声＋テキストの融合の先生 → 音声の生徒 | ×：ラベル付き上の埋め込み．ターゲットのラベル無しなし | ×：なし | △：IEMOCAP は話者独立 5 分割，CCC | △：融合の得で相補性を述べる |
| 5 LiSER 2025 | ×：CNN-LSTM を乱数初期化から | ×：なし | ×：なし | ×：なし | △：ラベルの割合 0.15〜1（割合，クラスあたりでない） | △：蒸留は第三コーパス，音声の先生はターゲットのラベル，評価はコーパス内 | △：音声と顔の先生 2 本．融合ではない．運用は音声だけ | △：先生の確率をラベル無し全体に別頭で．λ は検証で．確信度重み．均衡なし | ×：なし | ○：話者分離 5 分割，UAR・WAR | △：2 本の先生 > 1 本の実証 |
| 6 Hajavi 2023 | ×：CNN を学ぶ | ×：なし | ×：なし | ×：なし | ×：全ラベル | ×：コーパス内 | △：映像の先生 → 音声の生徒 | ×：埋め込みをラベル付き上で．α = 0.5 | ×：なし | △：LOSO（RAVDESS），unweighted accuracy | △：弱い先生で得 0・損 0 の実証 |
| 7 MKE 2021 | ×：3 層 CNN | ×：なし | ×：上界は先生の誤差の倍率 | ×：なし | ×：RAVDESS の 20% | ×：同じコーパスの無作為分割 | △：単モーダルの先生 → 多モーダルの生徒 | ○：学び直さない先生の疑似ラベルをラベル無し全体に．均衡・重み選択なし | ×：なし | ×：正解率，話者分離の記述なし | ○：定理 1（条件付き独立，c1c2 > 5） |
| 8 UMT 2021 | ×：エンコーダを学ぶ | ×：なし | ×：なし | ×：なし | ×：全ラベル | ×：コーパス内 | △：単モーダルの先生 → 融合の生徒 | ×：ラベル付き上の特徴 | △：生徒の特徴を先生の特徴に L2（特徴の L2-SP） | ×：top-1 ± | △：modality failure の勾配の説明 |
| 9 DAMR 2026 | ？ | ×：抄録に理論なし | × | × | ？ | ？：MELD・IEMOCAP を別々に | △：多モーダル ↔ 単モーダルの双方向 | ？ | ？ | ？ | △：「classification ambiguity」と EMA の能力 |
| 10 G²D 2025 | ×：ResNet-18／Transformer を学ぶ | ×：なし | ×：なし | ×：なし | ×：全ラベル | ×：コーパス内 | △：単モーダルの先生 → 融合の生徒 | ×：ラベル付き上の特徴＋logit（T = 1） | △：特徴の L2（UMT と同じ） | ×：正解率，種 1 つ | △：確信度の比・隙間・整列を測る |

参考（余力分）：PriCon は A1 △（先生・基準が線形プローブ），A7 ○（融合の先生 → 枠だけの生徒），A10 △（参加者分離，t 検定），他は ×．Guo 2026 と KBS 2021 は判定不能．

---

## この群の先行研究に空いている穴（示されたことと示されていないことの区分け）

### Takeaway
この群が示したのは，(a) 別モダリティの先生の特徴か確率を対になるラベル無しデータで写せば単モーダルの生徒が学べること（Gupta，Albanie，LiSER），(b) 融合の先生から単モーダルの生徒への蒸留がコーパス内で効くこと（Srinivasan，Hajavi，PriCon），(c) 単モーダルの先生が融合の生徒を助け，条件付き独立の下で生徒が先生を超えうること（MKE，UMT，G²D），(d) 特権情報の蒸留を温度と重みで一般化し，少数データで先生が効くと述べたこと（Lopez-Paz）である．示されていないのは，ソースのコーパスだけで学んだ先生をターゲットのコーパスのラベル無し発話に当てること，凍結したエンコーダの上の線形の頭だけで先生と生徒を作ること，クラスあたり K ≤ 10 の単位で得の曲線を測ること，ラベルの誤差を切片と向きに分ける理論，先生の確率のクラス均衡，K ごとの重みの選択，線形の頭をソースの頭に寄せる L2 罰則，である．

### Cited Findings（示されたこと）
- 別モダリティの先生の中間特徴を対データで写すと，ラベル無しのモダリティの表現が学べ，凍結しても 30.0 対 19.8 で複製より良い — [Gupta 2016](https://arxiv.org/abs/1507.00448)
- 顔の先生の温度 2 の確率で音声の生徒が学べ，RML で 49.7%（でたらめ 16.7%）．先生の確率は中立に偏り，均衡サンプリングは「did not have a significant effect」 — [Albanie 2018](https://arxiv.org/abs/1808.05561)
- 音声と顔の先生の確率をラベル無しの第三コーパスで写すと UAR が RAVDESS で 0.517 → 0.556，確信度の重みで 0.595．ラベルを半分にしても蒸留ありが全ラベルの蒸留なしを上回る — [LiSER 2025](https://arxiv.org/abs/2507.00055)
- 融合（音声＋テキスト）の先生の埋め込みで音声の生徒の感情価が 0.485 → 0.627（MSP-Podcast）．IEMOCAP で条件つき版は落ちる — [Srinivasan 2022](https://arxiv.org/abs/2112.00158)
- 映像の埋め込みを特権情報にすると RAVDESS で +1.6〜5.1 点，先生が弱い IEMOCAP では ±0．soft label 型の相手は α を上げると落ちる — [Hajavi 2023](https://arxiv.org/abs/2302.02845)
- 単モーダルの先生の疑似ラベルで多モーダルの生徒が先生を超える（RAVDESS 80.33 → 91.38）．定理 1：Err(f_s) ≤ 4·Err(f_t)／(c1c2 − 1) + 4μ（条件付き独立，c1c2 > 5） — [MKE 2021](https://arxiv.org/abs/2103.14431)
- 合同学習は弱いモダリティを学ばず（映像の学習正解率 34.69%），単モーダルの先生の特徴を写すと 49.46 → 53.46 — [UMT 2021](https://arxiv.org/abs/2106.11059)
- 先生の特徴と logit（T = 1）＋弱いモダリティから順に学ぶと CREMA-D で 67.47 → 85.89 — [G²D 2025](https://arxiv.org/abs/2506.21514)
- 一般化蒸留：T と λ で定義，「the teacher is most helpful in low data regimes」，「the lack of a proper weighting between labeled and unlabeled data」，同じラベルで入力モダリティが違う課題では「the input modalities from the source tasks are privileged information」 — [Lopez-Paz 2016](https://arxiv.org/abs/1511.03643)
- 融合の先生の KL で枠だけの生徒が RECOLA 活性で 59.41 → 60.75〜63.61．α は検証集合で選び「may not generalise well to other datasets」 — [PriCon 2025](https://arxiv.org/abs/2508.03729)
- 多モーダルと単モーダルの相互蒸留で IEMOCAP の基準より 5.54% 上がる（抄録） — [DAMR 2026](https://api.crossref.org/works/10.3390/electronics15204690)

### Cited Findings（示されていないこと＝穴．10 本の本文に該当の記述が無いことを根拠にする）
1. **ソースのコーパスだけで学んだ先生を，ターゲットのコーパスのラベル無し発話に当てて弱いラベルにする研究は無い．** Albanie は第三コーパス（VoxCeleb）で蒸留しターゲットでは全ラベルでアフィン写像を学ぶ（[Albanie 2018](https://arxiv.org/abs/1808.05561)）．LiSER は第三コーパスで蒸留し，音声の先生はターゲットのラベルで微調整する（[LiSER 2025](https://arxiv.org/abs/2507.00055)）．Srinivasan・Hajavi・MKE・UMT・G²D・PriCon は同じコーパスの中（各 arXiv 本文）．Gupta のゼロショット（MS COCO → NYUD2）だけがソースの先生をターゲットに当てるが，特徴の蒸留で感情ではない（[Gupta 2016](https://arxiv.org/abs/1507.00448)）
2. **先生も生徒も凍結したエンコーダの上の線形の頭，という形は無い．** 近いのは，Albanie の評価の段（8 次元出力のアフィン写像＋softmax），Lopez-Paz の人工データ（ロジスティック回帰），PriCon の先生と基準（線形プローブ）だけで，いずれも蒸留の本体は深層の網を学ぶ
3. **クラスあたり K ≤ 10 の単位でラベルを振った研究は無い．** 単位は，Lopez-Paz が総数 n（200〜500），MKE がデータの 20%，LiSER が学習集合の割合（0.15〜1）．CREMA-D の 0.15 はクラスあたり 100 本を超える
4. **理論は 2 本だけで，どちらもラベルの誤差を切片と向きに分けない．** Lopez-Paz は速さの議論（仮定 3 つ，定理ではない），MKE は先生の誤差の倍率の上界（条件付き独立）．ソースの価値をラベルの本数に換算する式，合併の切片の式は無い
5. **先生の確率のクラス均衡は，Albanie が均衡サンプリングを試して効かなかった以外に無い．** LiSER の確信度の重みは確信の高いクラスを強める向きで均衡の逆．Srinivasan の残差による除外はラベルを使う
6. **蒸留の重みをラベルの本数ごとに選ぶ研究は無い．** 固定値：Albanie T = 2，Srinivasan λ = 30，UMT λ = 50，G²D α = β = 1・T = 1，Hajavi α = 0.5．検証集合で選ぶ：LiSER（λ ∈ {0.1, …, 10}），PriCon（α ∈ {0.25, …, 1.0}，1 秒窓で決めて他の窓に流用）．Lopez-Paz は λ・T を掃き，重みづけが無いことを限界に挙げる
7. **線形の頭をソースのコーパスの頭に寄せる L2 罰則は無い．** UMT と G²D の特徴の L2 はエンコーダを先生の特徴に寄せるもので，対象が違う
8. **評価の 4 点（話者分離・UAR・相手の調整・検定）を全部持つ研究は無い．** 話者分離：LiSER，Srinivasan（IEMOCAP），Hajavi（RAVDESS の LOSO），PriCon（参加者）．UAR：LiSER（Hajavi は unweighted accuracy）．検定：PriCon（t 検定）．Albanie・MKE・UMT・G²D は正解率で話者分離の記述が無い
9. **相補性を測った量で得の条件を予言する研究は無い．** MKE の c1c2 と条件付き独立は理論の条件で測っていない．Albanie は冗長性の仮説を述べるだけ．G²D の確信度の比は学習中の診断で，得の予言ではない
10. **融合の先生 → 単モーダルの生徒をコーパス間で，先生を学び直さずに行う研究は無い**（1 と 2 の組み合わせ）．Srinivasan と PriCon は融合の先生を持つがコーパス内

### Inferences（論文の位置づけの書き方）
- 本研究の主張「MER において線形の頭に限って理論を立て，マルチモーダル性を活かして弱いラベルを作る」は，この群に対して次の 3 点で差を書ける．(i) 設定：先生はソースのコーパスだけで学び，ターゲットのコーパスのラベル無し発話に当てる（穴 1・10）．(ii) 学習器：凍結した特徴の上の線形の頭に限る（穴 2）．(iii) 理論と測定：ラベルの誤差を切片と向きに分け，K 本で減る式と，相補性の量で得の条件を書く（穴 4・9）
- 書いてはいけないこと：「別モダリティの先生の確率をラベル無しデータに付けるのは新しい」（Albanie，LiSER，Lopez-Paz の半教師あり拡張），「融合の先生から単モーダルの生徒への蒸留は新しい」（Srinivasan，PriCon），「単モーダルの先生が融合を助けることは新しい」（UMT，MKE，G²D），「蒸留は少数ラベルで効くと初めて述べた」（Lopez-Paz が述べている）
- Lopez-Paz 4.2 節の多課題の文（同じラベル・違う入力モダリティならソースのモダリティが特権情報）は，本研究の設定を一般化蒸留の枠に置く文としてそのまま引ける
- Albanie の「均衡サンプリングは効かなかった」は，本研究の列正規化（クラスごとの総量をそろえる）が別の答えであることを示す導入に使える．ただし Albanie は第三コーパスでの話なので，ターゲットのコーパス上での偏りは本研究で別に示す
- Hajavi の「soft label の蒸留は正解ラベルがある場面では効かない」は，本研究で K が増えると弱いラベルの得が消える観察と向きが同じである．引くときは，Hajavi がラベル付きの標本に soft label を付けた（本研究はラベル無しの発話に付ける）という違いを書く
- Srinivasan の IEMOCAP での悪化と PriCon の「α は他のデータに移らない」は，重みを K ごと・コーパスごとに仮のターゲットで選ぶ本研究の手続きの動機に使える

### Gaps
- DAMR の本文（コーパス間か，凍結か，UAR か）と，Guo 2026・KBS 2021 の本文は読めていない．この 3 本が穴 1・10 を埋める可能性は低いが，部品 5（往復）と部品 4（映像で疑似ラベルを直す半教師あり）に近いので，引用の前に本文を開く
- 会議名が本文に無い 5 本（Gupta の CVPR 頁，Srinivasan の ICASSP，LiSER の Interspeech，MKE の ICCV，G²D の ICCV）は，引く前に会議ページで確かめる．UMT 2021 は査読つきの掲載先が見つからなかった
- MKE の補足資料（硬いラベル対柔らかいラベル，ラベル無しの量）は未読
- この群の判定は 10 本（＋3 本）に限る．感情認識のモダリティ間蒸留の全体（Li 2021 Interspeech，EmoDistill，PKDOT など）は `docs/related-work.md` 1.12 節の抄録水準の判定のまま
