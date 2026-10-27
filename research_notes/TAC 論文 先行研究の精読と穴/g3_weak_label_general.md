# G3. 少数ショット・ドメインをまたぐ設定で，学び直さない先生の出力や疑似ラベルをラベル無しデータ全体に使う一般手法 10 本の精読（2026-10-27）

この記録は，前回のノート（`research_notes/TAC 論文の先行研究と書き方/method_components_precedents.md`）で URL と要点だけを押さえた 10 本（＋余力の xMUDA と fusion-then-distillation の系統）の本文を読み切り，各論文が「どの設定で，何を示し，どこで止まっているか」を本研究の軸 A1〜A11 に沿って書いたものである．目的は，本研究の主張「MER において線形の頭に限って理論を立て，マルチモーダル性を活かして弱いラベルを作る」の置き場所を，先行研究との差として正確に書くことにある．

**調べ方．**arXiv の PDF 11 本を curl で `scratchpad/g3_pdfs/` に取り，`pdftotext -layout` で文字にして全文を読んだ（STARTUP・AdaMatch は "Published as a conference paper at ICLR" の表記つき PDF）．書誌は PDF の表記と arXiv の抄録頁の Comments 欄で確かめ，無いものは WebSearch で補った．fusion-then-distillation の系統は FtD++（arXiv:2410.19446）の HTML を WebFetch の要約で読んだ（直接の精読ではない）．

**確かめた水準の記号．**◎＝本文（PDF 全文）を読んだ，○＝抄録か書誌の頁（arXiv の Comments 欄，学会の頁，他論文の参考文献）で確かめた，△＝検索結果の要約だけ．書誌は作らず，確かめられないものは「要確認」と書く．原文の引用は英語のまま短く．

**本研究の軸（表の列）．**A1 学習器：凍結したエンコーダの特徴の上の線形の頭だけか．A2 理論：ラベルの誤差を切片と向きに分けるか，K 本で減る誤差の式があるか．A3 理論：ソースの価値をターゲットのラベルの本数に換算するか．A4 理論：合併で切片が直る割合や暗黙のドメイン別切片を扱うか．A5 設定：ターゲットのラベルがクラスごとに数本か，K を振るか，K の単位．A6 設定：ドメインをまたぐか，クラスの集合が同じか．A7 設定：マルチモーダルか，融合の先生から単モーダルの生徒へ使うか．A8 手法：学び直さない分類器の出力をターゲットのラベル無し全体の弱いラベルにするか（確率か特徴か，クラス均衡の形，重みを K ごとに選ぶか）．A9 手法：出発点への L2 罰則つきで少数ラベルから学ぶか（強さの決め方）．A10 評価：相手の調整，検証データの扱い，検定．A11 分析：得が生じる条件と消える条件（クラスの比率の偏り，ドメインの差の大きさ）．

## Q1. 10 本（＋余力の 2 系統）の本文の記録：どの設定で，何を示し，どこで止まっているか

### Takeaway
10 本すべての本文を読み切った（◎）．書誌は，ICLR 2021（STARTUP），ICLR 2022（AdaMatch），NeurIPS 2021（Veilleux）が PDF の表記で確かめられ（◎），ECCV 2024（MM-CDFSL），CVPR 2024（CLAP），ECCV 2020 Oral（BD-CSPN），CVPR 2020（Noisy Student，xMUDA）が arXiv の Comments 欄で確かめられ（○），NeurIPS 2024 Spotlight（TransCLIP）は NeurIPS の論文集の頁の検索結果で（○），ICANN 2021（PT-MAP）は DBLP の記録の検索結果で（○）確かめた．UPL は掲載先が見つからず arXiv preprint として扱う．設定の要点は次のとおり：ソースだけで学んだ先生を学び直さずにターゲットのラベル無し全体に当てるのは STARTUP（確率，温度 1，表現の学習にだけ使う），TransCLIP（テキスト側のゼロショット確率への KL，λ=1／0.5 固定），UPL（硬いラベル，上位 16 本），Noisy Student（柔らかいラベル，クラスごと 13 万本）である．出発点への L2 罰則つきの線形プローブは CLAP だけ（クラスごとの λ_c，切片なし，ラベル無しは使わない）．理論を持つのは BD-CSPN（標本数 K+Z の関数の下界）と TransCLIP（最適化の収束）だけで，切片と向きの分解・ソースの価値の換算・合併の切片は 10 本のどれにも無い．クラスの比率の偏りを正面から扱うのは Veilleux（PT-MAP が 18〜20 点，TIM が約 4 点落ちる）だけで，AdaMatch は限界として自分で書き，PT-MAP は比率が分かれば効くと示す．

### Cited Findings
以下，論文ごとに template の順（1 書誌と読んだ範囲，2 設定，3 示したこと，4 軸との対応，5 穴，6 差の書き方，7 注意）で書く．出典の URL は各論文の 1 に置く．

#### 1. STARTUP — Phoo & Hariharan, "Self-training for Few-shot Transfer Across Extreme Task Differences"

**1. 書誌と読んだ範囲**
- 書誌（確かめた）：Cheng Perng Phoo, Bharath Hariharan（Cornell）．arXiv:2010.07734v2（2021-03-17）．PDF の各頁の頭に "Published as a conference paper at ICLR 2021" とある．ICLR 2021 は PDF の表記で確認（◎）．
- 読んだ範囲：◎ 本文全節（1 Introduction，2 Problem Setup，3 Related Work，4 Approach（4.1 Evaluation，4.2 Initialization），5 Experiments（5.1〜5.3），6 Conclusion）と付録 A.1〜A.7（実装，全ショットの表 5〜11）．読めなかった節：なし．
- 出典：[arXiv PDF](https://arxiv.org/pdf/2010.07734)，[ar5iv](https://ar5iv.labs.arxiv.org/html/2010.07734)．

**2. 設定**
- 課題：ドメインとクラスの両方が違う少数ショット転移（cross-domain few-shot learning）．ベース（ソース）は miniImageNet か ImageNet（ILSVRC 2012），ターゲットは BSCD-FSL の 4 つ（CropDiseases，EuroSAT，ISIC2018，ChestX）．2 節で「ベースのデータは完全に別の画像空間 X_B と別のクラス集合 Y_B から引く」と明記（"drawn from a completely disjoint image space X_B and a disjoint set of classes Y_B"）．
- データ：各ターゲットの 20% をラベル無し集合 D_u に，残りを評価の課題の抽出に使う．
- モダリティ：画像のみ．
- ラベル数と単位：5-way k-shot，k ∈ {1, 5}（付録で 20, 50）．単位はクラスあたりの本数．600 課題の平均と 95% 信頼区間．
- 学習器：表現学習の段では ResNet-10（ImageNet の実験は ResNet-18）の全体を微調整（生徒）．評価の段では「表現 φ を凍結し，支持集合で線形分類器を学ぶ」（4.1 節．"we freeze the representation φ after performing STARTUP and train a linear classifier on the support set"）．線形分類器は交差エントロピー＋SGD，100 エポック，学習率 0.01，weight decay 1e-4（A.1.3）．
- 評価：5-way の正解率．太字は「t 検定（有意水準 0.05）で差が無い上位の方法」（表 1 のキャプション）．

**3. 示したこと**
- 手法（4 節）：(1) ベースのデータで先生 θ_0 を交差エントロピーで学ぶ．(2) 先生をターゲットのラベル無し D_u 全体に当て，柔らかいラベル ȳ_i = f_{θ0}(x_i)（確率分布）を付ける（式 1）．(3) 生徒を「ベースの交差エントロピーの平均 ＋ ラベル無し上の先生の確率への KL の平均 ＋ 自己教師の損失（SimCLR）」の和で学ぶ（式 2．3 項に重みは無い）．
- 温度：「ラベル無し集合の柔らかいラベルを推論するときの温度は 1」（A.1.2．"we use a temperature of 1 when inferring the soft label of the unlabeled set"）．SimCLR の温度も「BSCD-FSL には検証集合が無いので」1 に固定．
- 弱いラベルは何の学習に使うか：**表現（埋め込み φ）の学習だけ**．ベースのクラスはターゲットでは意味が無いので，先生の確率は「先生が誘導するグループ分け」として表現に写すためのものである（"our goal is to adapt the feature representation to the target domain, rather than improve the base classifier itself"）．ターゲットの分類器（線形）は，支持集合の K 本だけで学ぶ．
- 生徒の初期化：先生の埋め込み＋ランダムな分類器が既定（4.2 節．3 通りを比べ，決定的な優劣は無い）．
- 主な結果（表 1，miniImageNet → BSCD-FSL，5-way 1-shot）：Transfer（素朴な転移）に対し平均 +2.9 点．CropDisease 1-shot で 69.97 → 75.93（約 6 点）．EuroSAT 1-shot 60.73 → 63.88．ChestX は全手法が同程度（22.71 → 23.09）．5-shot では ISIC 43.08 → 47.22，EuroSAT 80.30 → 82.29．
- SimCLR を抜いた版（STARTUP (no SS)）でも EuroSAT で Transfer と Transfer+SimCLR を上回る（62.90 vs 60.73 / 57.18）ので，得は自己訓練の項にもある．
- ImageNet（ResNet-18）の先生（表 2）：EuroSAT 1-shot 66.08 → 73.83，CropDisease 85.10 vs 74.17．
- 仕組みの分析（5.2 節）：別のドメインのラベル無しデータで自己訓練すると（EuroSAT で学んで CropDisease を評価）Transfer を下回る（70.40 vs 74.17）ので，「ラベル無しデータは単なる雑音ではない」．先生の予測の最頻クラスと真のラベルの AMI（調整相互情報量）は ChestX 0.0075，ISIC 0.0427，EuroSAT 0.3079，CropDisease 0.2969 で，AMI が高い 2 つで得が大きい（表 3）．
- ドメインが近いとき（5.3 節，miniImageNet）：「ほとんどの場合に助けにも害にもならない」．
- 理論：なし．
- ラベル無しデータの量：多いほど良い（A.5，図 3）．ベースのデータ無し（微調整）：ISIC の 5-shot だけ有意に落ちる（A.6，表 10）．

**4. 軸との対応**
- A1 線形の頭だけ：△．評価は凍結した表現の上の線形分類器だが，手法の本体は表現（深層網）の微調整．
- A2 切片と向き／K 本の誤差の式：×．理論なし．
- A3 ソースの価値をラベル本数に換算：×．
- A4 合併の切片：×．
- A5 K 本・K を振る：○．5-way k-shot，k ∈ {1, 5, 20, 50}．単位はクラスあたり．
- A6 ドメインをまたぐ・クラス集合：○（ドメインをまたぐ）／クラス集合は**違う**（disjoint）．本研究（同じ 4 クラス）とは設定が違う．
- A7 マルチモーダル：×．画像のみ．
- A8 学び直さない先生の出力をラベル無し全体の弱いラベルに：○．ソースだけで学んだ先生の**確率**（温度 1）を D_u 全体に付け，KL で生徒に写す．クラス均衡：×（無し）．重みを K ごとに選ぶ：×（3 項の和で重み無し．K は評価の段だけ）．
- A9 出発点への L2 罰則：×．生徒は先生の埋め込みで初期化するが，罰則は無い．線形分類器は weight decay 1e-4 のみ．
- A10 評価：検証は「ラベル無しの 10% とラベル付きの 5%」の内部検証集合で学習率を選ぶ（A.1.2）．検定は t 検定（0.05）．相手の調整：Guo ほかの数値を転載し，Transfer は再実装．
- A11 得が生じる条件：△．ドメインの近さ（5.3 節）と先生のグループ分けの AMI（5.2.2 節）で得の大小を説明．クラスの比率の偏りは扱わない．

**5. 穴**
- 弱いラベルを分類器の学習に使っていない（クラス集合が違うので使えない）．本研究は同じ 4 クラスなので，弱いラベルを分類器そのものに使える．
- 先生の出力のクラスの偏りの補正は無い．
- 理論（なぜ KL の項が効くか）は「仮説」の検証（AMI と t-SNE）にとどまる．"it is not clear why or how it achieves this improvement"（5.2 節の冒頭）．
- 限界として自分で書いていること：ChestX では「すべての方法が同程度」．miniImageNet では「助けにも害にもならない」．
- 今後の課題の節は無い（結論は 1 段落）．

**6. 本研究との差の書き方（案）**
STARTUP（Phoo & Hariharan, ICLR 2021）は，ソースのデータだけで学んだ先生の確率を温度 1 でターゲットのラベル無し画像全体に付け，KL 損失で生徒の表現に写す．ソースとターゲットでクラスの集合が違うため，弱いラベルは表現の学習にしか使えず，ターゲットの分類器は支持集合のラベルだけで学ぶ．本研究はクラスの集合が同じ感情認識で，弱いラベルを線形の分類器そのものの学習に使い，その総重量をラベルの本数 K ごとに選ぶ．

**7. 注意**
- 表 1 の Transfer は Guo ほか（2020）の転載値（Transfer∗）と著者の再実装（Transfer）で数値が違う（例：ChestX 5-shot 25.35 vs 26.71）．批判する側は「80% のテスト集合とバッチサイズの違い」と自分で書いている．
- 「平均 +2.9 点」は 1-shot の miniImageNet の 4 データセットの平均で，5-shot の平均は本文に無い．

#### 2. MM-CDFSL — Hatano, Hachiuma, Fujii, Saito, "Multimodal Cross-Domain Few-Shot Learning for Egocentric Action Recognition"

**1. 書誌と読んだ範囲**
- 書誌（確かめた）：Masashi Hatano, Ryo Hachiuma, Ryo Fujii, Hideo Saito（慶應，NVIDIA）．arXiv:2405.19917v3（2024-07-16）．PDF は LNCS の体裁で会議名の表記が無い．ECCV 2024 の掲載は，arXiv の抄録頁の Comments 欄 "Accepted at ECCV'24" と，前回のノートの ECVA の頁（papers_ECCV/papers/04830.pdf）で確かめた（○）．
- 読んだ範囲：◎ 本文全節（1 Introduction，2 Related Work，3 Method 3.1〜3.4，4 Experiments 4.1〜4.4，5 Conclusion（Limitations and future work を含む））と補足 A〜D（データ，実装，UMAP，損失の重みの ablation）．読めなかった節：なし．
- 出典：[arXiv PDF](https://arxiv.org/pdf/2405.19917)，[arXiv HTML](https://arxiv.org/html/2405.19917v1)．

**2. 設定**
- 課題：一人称視点の行動認識で，ドメインをまたぐ少数ショット（CD-FSL）＋ラベル無しターゲット＋多モーダル入力（学習時だけ）．ソースは Ego4D（204 クラス，15.5k クリップ），ターゲットは EPIC-Kitchens（料理），MECCANO（工業），WEAR（屋外運動）．「ソースとターゲットでクラスの重なりが無いことを保証する」（3.1 節．"Ensure there is no class overlap between the source and target datasets"）．ターゲットは学習用（ラベル無し）と検証用にクラスで分ける（EPIC：58 クラス 3.6k と 228 クラス 17.2k）．
- モダリティ：RGB，光学流（FlowFormer で推定），手の姿勢のヒートマップ（RTMPose で推定，21 関節）．推論は RGB だけ．
- ラベル数と単位：5-way 1-shot／5-shot（クラスあたり）．600 回の平均と 95% 信頼区間．
- 学習器：ViT-S の VideoMAE（Kinetics-400 で事前学習）を各モダリティで微調整（深層）．少数ショットの段は「RGB の生徒のエンコーダを保ち，その上の分類の頭 G' を N-way K-shot の支持集合で学ぶ」（3.4 節）．Random Initialization の基準は「学習していない VideoMAE の上のロジスティック回帰」．
- 評価：top-1 正解率と推論時間（ms），GFLOPs，メモリ．

**3. 示したこと**
- 手法：(I) 事前学習：モダリティごとに VideoMAE を「ソースの再構成 ＋ ターゲットの再構成 ＋ λ_ce × ソースの交差エントロピー」で学ぶ（式 2）．λ_ce は RGB 5e-2，光学流・手の姿勢 1e-2（補足 B，"empirically adapt"）．(II) 多モーダル蒸留：生徒（RGB）はターゲットの**ラベル無し** RGB だけを入力し，射影 M_m で各モダリティの先生の特徴に L2 で合わせる（式 3〜6．和に重み無し）．「先生のエンコーダの重みは蒸留の段で凍結する」（"the weights of teacher encoders are kept frozen during the multimodal distillation stage"）．先生には RGB 自身も含める（忘却と他モダリティへの偏りを防ぐため）．蒸留するのは**特徴**（埋め込み）で確率ではない．(III) 少数ショットの頭の学習．(IV) マスクした入力の集団推論（高速化）．
- 主な結果（表 2．Ego4D → 3 データセット）：1-shot EPIC 41.97 / MEC 28.34 / WEAR 51.25，5-shot 58.70 / 37.80 / 69.57．CDFSL-V（1-shot 38.17 / 26.03 / 39.11，5-shot 53.72 / 35.64 / 58.27）に対し平均 +6.12（1-shot）／+6.10（5-shot）．STARTUP++（VideoMAE 事前学習に置き換えた STARTUP）は 1-shot 35.18 / 26.84 / 39.15．推論は 2.2 倍速い（22.1 → 9.64 ms）．
- Ablation（表 4，EPIC）：再構成のみ 35.42 / 49.82，ソースのみ（ターゲットの再構成無し）40.50 / 56.43，両方 41.97 / 58.70．表 5（WEAR）：蒸留無し 46.17 / 67.19，RGB+姿勢 49.39 / 67.90，3 モダリティ 51.25 / 69.57．
- 「先生がターゲットのドメインの特徴を十分に捉えていなければ，ターゲットのデータでの蒸留は失敗する」（4.4 節．"The distillation process using target domain data fails if the pre-trained models do not adequately capture the target domain characteristics"）→ だからターゲットのラベル無しデータで先生を事前学習する．
- 理論：なし．

**4. 軸との対応**
- A1 線形の頭だけ：△．少数ショットの段は頭だけだが，手法の本体は ViT-S の微調整．
- A2：×．A3：×．A4：×．
- A5：○．5-way 1／5-shot，単位はクラスあたり．K は 2 点のみ．
- A6：○（ドメインをまたぐ）／クラス集合は**違う**（重なり無し）．
- A7：○．多モーダル（RGB・光学流・手の姿勢）．**モダリティごとの先生 → RGB の生徒**．融合の先生ではない．
- A8：△．先生はソースのラベル＋ソースとターゲットのラベル無しデータ（再構成）で学ぶので「ターゲットで学び直さない」ではない．ターゲットのラベル無し**全体**に当てる点は同じ．蒸留は**特徴**（確率ではない）．クラス均衡：×．重みを K ごとに：×（蒸留の重みは全モダリティで 1）．
- A9：×．出発点への罰則は無い（生徒は事前学習の重みで初期化）．
- A10：相手は「公平のため」VideoMAE 事前学習に置き換えた STARTUP++・Dynamic Distillation++．検証データの扱い：λ_ce は EPIC の 5-shot（評価と同じ）で掃いた（補足 D，表 6：λ=1→54.58，0.1→56.68，0.05→57.07，0.01→52.40）．検定：信頼区間のみ．
- A11：×．得が消える条件の分析は無い（λ_ce の高低の両方で落ちる，という ablation のみ）．

**5. 穴**
- 先生がターゲットのラベル無しデータを使って学んでいるので，「ソースだけで学んだ先生」の弱いラベルの効き方は測っていない．
- 蒸留が特徴の回帰なので，先生の分類器の出力（確率）とクラスの偏りの問題は生じない（扱っていない）．
- 限界（原文，5 節）："our approach applies constant loss weights for feature distillation losses during the multimodal distillation process, regardless of the specific target dataset. This strategy does not account for the varying significance of modalities based on the target domain"．今後："Dynamically adjusting distillation weights according to the modality's relevance in the target domain is crucial ... We will leave this for our future efforts."
- 光学流と手の姿勢は既製の推定器に頼る（"the bias and errors from the off-the-shelf detector may still affect the input modality information"）．

**6. 本研究との差の書き方（案）**
MM-CDFSL（Hatano ほか，ECCV 2024）は，一人称視点の行動認識で，モダリティごとの先生をソースのラベルとソース・ターゲットのラベル無しデータで事前学習して凍結し，ターゲットのラベル無し RGB で生徒に特徴を蒸留してから 5-way K-shot の頭を学ぶ．先生がターゲットのデータで学んでいること，蒸留が確率ではなく特徴であること，ソースとターゲットでクラスが違うことが本研究と異なる．本研究は，ソースのコーパスだけで学んだ融合の線形分類器の確率を，同じ 4 クラスのターゲットの発話全体の弱いラベルとして線形の生徒に写し，その効き方を K ごとに測る．

**7. 注意**
- 表 2 で MECCANO 5-shot は Dynamic Distill++ 37.87 が Ours 37.80 を上回る（本文は「一貫して上回る」と書くが，この 1 欄は例外）．
- 4.3 節の「WEAR 5-shot で 58.28 → 69.57」は表 2 の CDFSL-V の値 58.27 と 0.01 ずれる．
- 平均の改善幅を表 2 から計算すると 1-shot +6.08，5-shot +6.15 で，本文の 6.12／6.10 と小数第 2 位で食い違う（丸めの差とみられる）．
- 少数ショットの段でエンコーダを凍結するかは「retained」とだけ書かれており，明示されていない（要確認）．

#### 3. TransCLIP — Zanella, Gérin, Ben Ayed, "Boosting Vision-Language Models with Transduction"

**1. 書誌と読んだ範囲**
- 書誌（確かめた）：Maxime Zanella, Benoît Gérin（UCLouvain），Ismail Ben Ayed（ÉTS Montréal）．arXiv:2406.01837v1（2024-06-03）．PDF v1 には "Preprint. Under review." とあり，arXiv の抄録頁にも Comments 欄は無い．NeurIPS 2024（Main Conference Track，Spotlight）の掲載は，NeurIPS の論文集の頁（proceedings.neurips.cc，hash 71d7dbe2652bd4662d29fa269f059db4）と前回のノートの DOI 10.52202/079017-1988 で確かめた（○）．
- 読んだ範囲：◎ 本文全節（1〜5，3.1 目的関数，3.2 少数ショット，3.3 BMM，3.4 収束，4 実験 4.1〜4.2）と付録 A（収束），B（実装・ハイパーパラメータ），C.1〜C.5 の本文（表 8〜23 の数値は読んでいない），D（限界）．
- 出典：[arXiv PDF](https://arxiv.org/pdf/2406.01837)，[arXiv HTML](https://arxiv.org/html/2406.01837)．

**2. 設定**
- 課題：CLIP のゼロショット・少数ショット分類を，テスト標本全体（ラベル無し）を一緒に推論する転導で良くする．ImageNet ＋ 10 の細粒度データセット，ImageNet の 4 変種（Adversarial，V2，Rendition，Sketch）．
- モダリティ：画像とテキスト（CLIP）．テキスト側のゼロショット予測 ŷ_i が「先生」の役．
- ラベル数と単位：0-shot，1／4／16-shot（表 1・4．クラスあたり）．
- 学習器：**模型は学び直さない**（"black-box setting ... without training the model parameters"）．最適化するのは各ラベル無し標本の割り当て z_i（確率単体），クラスの中心 μ_k，共通の対角共分散 Σ．分類器は混合ガウス（線形の頭ではない）．
- 評価：top-1，ViT-B/16，3 シードの平均．

**3. 示したこと**
- 目的関数（式 2）：「ラベル無し標本の GMM の負の対数尤度の平均 ＋ ラプラシアン項（近い標本は同じ割り当て．親和度 w_ij = max(0, f_i^T f_j)，上位 3 近傍で疎に）＋ Σ_i KL_λ(z_i ‖ ŷ_i)」．KL_λ(z‖ŷ) = z^T log z − λ z^T log ŷ（式 3．λ=1 で通常の KL）．**KL はラベル無し（問い合わせ）の全標本にかかる．**
- 少数ショット（式 4）：支持集合の項 −(γ/|S|) Σ_{i∈S} z_i^T log p_i（z_i は one-hot に固定）を足す．
- 重みの固定値：「ゼロショットの全実験で λ=1，少数ショットの全実験で λ=0.5（テキストの正則化の影響を減らすため）」（4 節 Implementation details）．γ は {0.002, 0.01, 0.02, 0.2} から検証（min(4, #shots) 本の検証ショット，問い合わせの最終割り当てで 1-NN）で選ぶ（付録 B）．
- 更新（式 5〜7）：z の更新は ŷ_i^λ ⊙ exp(log p_i + Σ_j w_ij z_j) を正規化（標本ごとに独立）．μ_k と Σ は閉じた形．10 回の block 更新，z の内側は 5 回．
- 理論：定理 1（BMM の収束．Razaviyayn ほか 2013 からの引用）：「各ブロックの majorizing 関数が準凸で一次の挙動が元の目的関数と局所的に同じ，各ブロックの部分問題の解が一意なら，反復の任意の極限点は座標ごとの最小点」．**汎化誤差や誤差の分解の理論は無い．**
- 主な結果：ゼロショット CLIP-ViT-B/16 の 11 データセット平均 65.3 → 70.3（+5.1）．EuroSAT 48.3 → 65.1（+16.8）．少数ショットの手法の上（1-shot）で平均 +3.7〜+4.8．16-shot では得が小さく，PLOT の上では −1.4（表 1）．ドメイン一般化（表 3）：OOD 平均 57.1 → 59.2．
- 転導的少数ショットとの比較（表 4，1-shot 平均）：TF 42.8，BD-CSPN 50.6，LaplacianShot 50.9，PT-MAP 52.7，TIM 52.8，TransCLIP-FS w/o text（λ=0）50.2，TransCLIP-FS 73.2．**テキストの項（先生）を外すと 73.2 → 50.2** なので，得はほぼ先生の KL から来る．16-shot では w/o text 76.6 vs 78.4 で差が縮む．
- λ の感度（表 6b，ゼロショット）：ImageNet で λ=0.1→56.3，0.5→69.8，1→70.3，2→69.5，5→68.2．EuroSAT で λ=5 だと 51.2（ゼロショット 48.3 に近い）．
- UPL との比較（表 5）：転導版 UPL∗ 69.8（>150 分）vs TransCLIP-ZS 70.3（14.4 秒）．UPL のクラスあたりの疑似ラベル数を再検証しても 16 が平均で最良（C.1）．
- クラス均衡："balanced mixture of multivariate Gaussian distributions"（均等な混合比を仮定）．列正規化や Sinkhorn のような明示の均衡の操作は無い．

**4. 軸との対応**
- A1 線形の頭だけ：×．凍結した埋め込みの上だが，分類器は GMM の割り当て（線形ではない）．
- A2：×．A3：×．A4：×（理論は最適化の収束のみ）．
- A5：○．0／1／4／16-shot（クラスあたり）．
- A6：△．ドメイン一般化（ImageNet の変種）と cross-dataset を扱うが，ソースのラベル付きデータからの転移ではない．クラス集合は同じ（ゼロショットのプロンプトが全クラスを覆う）．
- A7：○（画像＋テキスト）．ただし「融合の先生 → 単モーダルの生徒」ではなく，テキスト側のゼロショット予測が画像側の割り当てを導く．
- A8：○．学び直さない先生（テキストのゼロショット予測）の**確率**（温度は CLIP の τ）に対する KL を**ラベル無しの全標本**にかける．クラス均衡：△（均等な混合を仮定するだけ）．重みを K ごとに：△（λ はラベルの有無で 1 と 0.5 の 2 段階．γ は検証ショットで選ぶ）．
- A9：×．出発点への L2 罰則ではなく，予測分布への KL．
- A10：相手の調整：転導的少数ショットの相手は各データセット・各ショット数で min(4, #shots) 本の検証ショットで格子探索（C.5）．検定：なし（3 シードの平均のみ）．
- A11：△．得が消える条件：ショット数が増えると得が減る（表 1・4，付録 D）．先生（テキスト）が弱いデータセット（EuroSAT，Flowers）では 16-shot で w/o text が勝つ．クラスの比率の偏りは扱わない（均等な混合を仮定）．

**5. 穴**
- 理論は最適化の収束だけで，「先生の KL がいつ得になるか」の誤差の式は無い．
- 均等な混合比を仮定しており，クラスの比率が偏ったときの挙動は調べていない（Veilleux ほかの指摘の対象．同じ Ben Ayed のグループ）．
- 限界（付録 D，原文）："the gain of TransCLIP-ZS on top of few-shot methods tends to decrease when the number of shots is high (e.g., 16 shots)"；"as TransCLIP's performance relies greatly on its text-regularization term, TransCLIP is subject to some biases. One notable bias pertains to the quality of text embeddings within each class."
- 今後の課題（5 節，原文）："future works may consider making it more resilient (e.g., through adaptive class-wise weighting factor) when text prompts are less reliable."
- ソースのドメインで学んだ分類器を先生にする設定（本研究）は無い．

**6. 本研究との差の書き方（案）**
TransCLIP（Zanella ほか，2024）は，CLIP のテキスト側のゼロショット予測を学び直さない先生とし，その確率への KL をラベル無しのテスト標本全体にかけて，混合ガウスの割り当てを少数ショットの項と同時に解く．KL の重みはゼロショットで 1，少数ショットで 0.5 に固定し，理論は最適化の収束にとどまる．本研究は，先生がソースのコーパスで学んだ融合の線形分類器であり，生徒も線形の頭なので，弱いラベルの総重量を K ごとに選び，その得が生じる条件を誤差の式で説明する．

**7. 注意**
- NeurIPS 2024 の掲載は PDF v1（"Preprint. Under review."）には無く，NeurIPS の論文集の頁で確かめた（○）．著者の所属は論文集版では UCLouvain・UMons・ÉTS と検索結果にあり，arXiv 版（UCLouvain・ÉTS）と少し違う．
- 4.1 節の「ゼロショットの top-1 を 5% 以上」は表 1 の平均 +5.1（11 データセット）で，表 7 の平均は +5.0 と書かれている（丸めの差）．
- 付録の表 8〜23 の数値は読んでいない．

#### 4. CLAP — Silva-Rodríguez, Hajimiri, Ben Ayed, Dolz, "A Closer Look at the Few-Shot Adaptation of Large Vision-Language Models"

**1. 書誌と読んだ範囲**
- 書誌（確かめた）：Julio Silva-Rodríguez, Sina Hajimiri, Ismail Ben Ayed, Jose Dolz（ÉTS Montreal）．arXiv:2312.12730v2（2024-03-25）．PDF に会議名の表記は無い．arXiv の抄録頁の Comments 欄に "CVPR 2024. Code: ..." とあり，前回のノートの DOI（10.1109/cvpr52733.2024.02235）と合わせて CVPR 2024 を確かめた（○）．
- 読んだ範囲：◎ 本文全節（1〜6．3 Preliminaries，4 Proposed approach 4.1〜4.3，5 Experiments 5.1〜5.3，6 Limitations）と補足 A.1〜A.5（SoTA アダプタの分析，ハイパーパラメータの選び方），B（ALM の罰則関数の公理），C.1〜C.3（追加の設定，結果，ablation）．補足の表 9〜11 の個別の数値は 1-shot・2-shot の行だけ読んだ．
- 出典：[arXiv PDF](https://arxiv.org/pdf/2312.12730)，[arXiv HTML](https://arxiv.org/html/2312.12730)．

**2. 設定**
- 課題：CLIP の少数ショット適応（efficient transfer learning）．11 データセット（ImageNet，Caltech101，OxfordPets，StanfordCars，Flowers102，Food101，FGVCAircraft，SUN397，DTD，EuroSAT，UCF101）と，ImageNet で適応して ImageNet-V2・Sketch・A・R で測るドメイン一般化．
- モダリティ：画像とテキスト（CLIP）．テキストのプロトタイプ t_c（プロンプト集合の埋め込みの平均）がゼロショット分類器．
- ラベル数と単位：K ∈ {1, 2, 4, 8, 16}（クラスあたり）．検証集合は使わない（"validation-free"）．
- 学習器：**凍結した CLIP の特徴の上の線形プローブ**（クラスのプロトタイプ w_c，式 3．ŷ_c ∝ exp(v·w_c / τ)．**切片は無い**（コサイン類似度，w_c を各更新後に ℓ2 正規化）．ResNet-50 と ViT-B/16．300 エポック，SGD，学習率 0.1 の cosine 減衰，支持標本ごとに 20 のデータ拡張．
- 評価：3 シードの平均の正解率．

**3. 示したこと**
- 観察 1（図 1，A.3）：既存のアダプタ（CLIP-Adapter，TIP-Adapter，TaskRes）はデータセットごとにテスト集合か大きな検証集合でハイパーパラメータを調整しており，「あるデータセット i で最適な設定を別のデータセット j に使う」交差行列（cross-shift model selection matrices）では，ゼロショット初期化の線形プローブ（ZS-LP）に比べて「最大 21% 落ちる」（"dramatic performance degradations, up to 21%, compared to this simple baseline"）．TIP-Adapter の α の探索範囲は ImageNet で [1.17, 7]，Flowers102 で [10, 50] と重ならない（A.3）．TaskRes は「Flowers102 だけ α=1，他は 0.5」と書いており，テストの事前知識を示唆する（A.3）．
- 観察 2：ZS-LP（ゼロショット初期化＋温度 τ＋ℓ2 正規化＋データ拡張）は強い基準．ランダム初期化の LP は 1-shot で 30.42 に対し ZS-LP は 61.28（表 1）．ℓ2 正規化を外すと −12.6，温度を外すと −2.9，拡張を外すと −3.5（表 4，1-shot）．
- 手法（4.2〜4.3）：制約つき問題「支持集合の交差エントロピーを最小化，ただし w_c = t_c」（式 4）を罰則で緩め，クラスごとの罰則 Σ_c λ_c ‖t_c − w_c‖²（式 7）にする．λ_c は拡張ラグランジュ法（PHR 罰則関数，式 13）の外側の反復で学ぶが，「支持標本から乗数を推定すると過学習するので，**外側の反復は 1 回だけ**」．その 1 回の値は「そのクラスの支持標本のゼロショットの softmax 確率の平均」λ*_c（4.3 節）．つまり，ゼロショットが得意なクラスほど出発点に強く縛る．
- 主な結果（表 1，ResNet-50，11 データセット平均）：K=1／2／4／8／16 で ZS-LP 61.28／64.88／67.98／71.43／74.37，CLAP 62.79／66.07／69.13／72.08／74.57．得は 1-shot の +1.5 から 16-shot の +0.2 へ縮む．
- ドメイン一般化（表 2，16-shot ImageNet → 4 変種の平均）：ResNet-50 でゼロショット 40.61，ZS-LP 36.58（−4.03），CLAP 42.91（+2.30）．ViT-B/16 でゼロショット 57.17，TIP-Adapter(f) 35.58（−21.6），ZS-LP 53.65（−3.52），CLAP 60.04（+2.87）．**罰則なしの ZS-LP はドメインのずれでゼロショットを下回る．**
- ablation（C.3，表 8）：外側の反復を最後まで回すと K=4 で −4.2（過学習）．λ=1（一様）は 1-shot 以外で劣る．λ_avg（クラス平均で一様）は 2-shot で最良．λ*_c は 4-shot 以上で最良．
- 理論：無い（ALM の公理の記述のみ）．

**4. 軸との対応**
- A1 線形の頭だけ：○．凍結した CLIP の特徴の上の線形プローブ（切片なし，ℓ2 正規化あり）．
- A2 切片と向き：×．切片は無く，誤差の分解も無い．K で減る誤差の式も無い（経験的に「得は K で縮む」）．
- A3：×．A4：×．
- A5：○．K ∈ {1, 2, 4, 8, 16}（クラスあたり）．
- A6：△．ドメイン一般化は「ImageNet で適応し，変種で評価」で，ソースのラベル付きデータからターゲットへの転移ではない．クラス集合は同じ．
- A7：△．画像＋テキストだが，テキストは出発点（初期値）であって蒸留の先生ではない．ラベル無しデータは使わない．
- A8：×．ラベル無しデータを使わない．
- A9：○．**出発点（ゼロショットのプロトタイプ）への L2 罰則つきで少数ラベルから学ぶ．**罰則の強さは**クラスごと**に，支持標本のゼロショット確率の平均で決める（検証なし，1 回の ALM 更新）．
- A10：○．「検証集合を使わない」流儀を明示し，全データセット・全ショット数・全骨格で同じ設定．相手も同じ流儀で再評価（A.4）．検定：なし（3 シードの平均と標準偏差）．表 5 では「検証に min(K,4) 本を使う相手」に対し，自分は K+min(K,4) 本を学習に使う比較を提示．
- A11：△．得が消える条件：ショット数が増えると消える（6 節）．クラスの比率の偏り：×．ドメインの差：ドメイン一般化で罰則の有無の差を示す（表 2）が，差の大きさの関数としては扱わない．

**5. 穴**
- ラベル無しのターゲットデータを一切使わない．
- 切片を持たない（コサイン分類器）ので，本研究の「切片と向き」の分解は当てはまらない．
- 限界（6 節，原文）："our empirical validation suggests that the benefits of our approach diminish as the number of shots increases, indicating that other strategies might be privileged if the number of adaptation samples is large."
- C.3（原文）："In the special case of 1-shot or 2-shot, adaptive computation of λ may lead to a slight overfitting to the information provided by the few samples"．
- 今後の課題の節は無い．

**6. 本研究との差の書き方（案）**
CLAP（Silva-Rodríguez ほか，CVPR 2024）は，CLIP のテキストから得たゼロショットの分類器を出発点とする線形プローブを，出発点への L2 罰則つきで少数ラベルから学び，罰則の強さをクラスごとに支持標本のゼロショット確率で決める（検証集合は使わない）．本研究の出発点はソースのコーパスで学んだ分類器であり，罰則を重みと切片で分け，ラベル無しのターゲットの発話を弱いラベルとして併用し，罰則と弱いラベルの重みを仮のターゲットで K ごとに選ぶ点が異なる．CLAP の「別のデータセットで調整した設定は最大 21% 落ちる」という報告は，本研究の仮のターゲットでの選択がオラクルにどれだけ近いかを示す必要があることを意味する．

**7. 注意**
- 「最大 21%」は図 1 の交差行列の相対値（ZS-LP 比）で，本文に数値の表は無い．
- 表 2 の TIP-Adapter(f) の ViT-B/16 の OOD は −21.6 と書かれ，これは図 1 の 21% とは別の数値（ゼロショット比の絶対差）．混同しない．
- CVPR 2024 の掲載は PDF には無く，arXiv の Comments 欄で確かめた（○）．

#### 5. PT-MAP — Hu, Gripon, Pateux, "Leveraging the Feature Distribution in Transfer-based Few-Shot Learning"

**1. 書誌と読んだ範囲**
- 書誌（確かめた）：Yuqing Hu（IMT Atlantique，Orange Labs），Vincent Gripon（IMT Atlantique），Stéphane Pateux（Orange Labs）．arXiv:2006.03806v3（2021-01-26）．PDF に会議名の表記は無い（頁の見出しは "Running heading title breaks the line" のまま）．arXiv の抄録頁にも Comments 欄は無い．ICANN 2021（LNCS）の掲載は，DBLP の記録 "ICANN (2) 2021: 487-499"（検索結果．DBLP 本体はボット対策で開けなかった）と前回のノートの DOI 10.1007/978-3-030-86340-1_39 で確かめた（○）．
- 読んだ範囲：◎ 本文全節（1〜5．3.1 問題設定，3.2 特徴抽出，3.3 前処理（PT），3.4 MAP，4.1〜4.4 実験）と補足 6.1〜6.2．図 3（クラス不均衡）の曲線の数値は読めない．
- 出典：[arXiv PDF](https://arxiv.org/pdf/2006.03806)，[ar5iv](https://ar5iv.labs.arxiv.org/html/2006.03806)．

**2. 設定**
- 課題：転移に基づく少数ショット分類（帰納・転導）．miniImageNet，tieredImageNet，CUB，CIFAR-FS．ドメインをまたぐ実験は miniImageNet のベースで学んだ骨格を CUB で評価（表 4）．ベースと新しいクラスは重ならない（D_base ∩ D_novel = ∅）．
- モダリティ：画像のみ．
- ラベル数と単位：w-way s-shot q-query（w=5，s=1 か 5，q=15）．単位はクラスあたり．10,000 回の平均と 95% 信頼区間．
- 学習器：**凍結した骨格（WRN，ResNet12/18，DenseNet121）の特徴の上**で，べき変換（PT）→ クラスの中心の推定（MAP）．線形の頭ではなく，中心までのユークリッド距離による割り当て（NCM の転導版）．
- 評価：正解率．ハイパーパラメータ（β，λ，α，n_steps）は検証クラスで調整（4.4 節）．

**3. 示したこと**
- 前処理（式 1）：特徴（ReLU 後なので非負）を (v+ε)^β（β=0.5）にしてノルムで割る．ゆがんだ分布をガウスに近づける．
- **trans-mean-sub の定義**（4.2 節，原文）："we implement the 'trans-mean-sub' [18] before MAP where we separately subtract inputs by the means of labelled and unlabelled examples, followed by a unit hypersphere projection"．つまり，**ラベル付き集合にはラベル付きの平均を，ラベル無し集合にはラベル無しの平均を，別々に引いてから単位球に射影する．**注意：[18] は TAFSSL（Lichtenstein ほか，arXiv:2003.06670）で，trans-mean-sub の出所は TAFSSL であり PT-MAP の発明ではない．
- MAP（3.4 節）：クラスごとにガウスを仮定し，ラベル無し標本とクラスの中心の間の最適輸送を Sinkhorn で解く（式 3）．**割り当て行列 M* の制約（式 4）：行の和が p（各ラベル無し標本の質量 1），列の和が q（各クラスに割り当てる質量．q = q·1_w，つまり各クラスにちょうど q 本）．**エントロピー正則化 λ=10．中心の更新（式 5）はラベル付きの標本（割り当て 1）とラベル無しの柔らかい割り当ての重みつき平均．慣性 α（0.2〜0.4）で更新．n_steps は 20〜30．
- 均衡の仮定（3.1 節，原文）："In the case of transductive few-shot [20, 18], the prediction is performed considering all wq samples together. In the latter case, most works exploit the information that there are exactly q samples in each class."
- 主な結果（表 2，WRN）：miniImageNet 1-shot で帰納 PT+NCM 65.35，転導 PT+MAP 82.92（BD-CSPN 70.31，Transfer+SGC 76.47）．5-shot 83.87 → 88.82．CUB 1-shot 91.55，CIFAR-FS 87.69．miniImageNet → CUB（表 4）1-shot 62.49，5-shot 76.51（帰納 PT+NCM 48.37／70.22）．
- ablation（表 5）：MAP を K-means（中心をラベル付きで初期化）に替えると落ちる（WRN 1-shot 82.92 → 76.67）．PT を外すと約 6%（1-shot）・4%（5-shot）落ちる（表 6）．
- クラス不均衡（4.4 節，図 3，2-way 1-shot，ラベル無し 100 本）：(1) 帰納版 PT-NCM は比率に依存しない，(2) 比率が分かっているときに q を合わせた転導版，(3) 「各クラスに少なくとも 30 本」とだけ仮定した q=[30,30] の転導版，を比べ，"the transductive setting still outperforms the inductive ones even when the proportion of elements in both classes is only approximately known"．**ただし，比率を誤って一様と仮定した場合（q=[50,50] で実際は偏る）の結果は示していない．**
- ラベル無しの本数 q（図 4(1)）：正解率は q が増えるとすぐに頭打ちになる．
- 理論：無し（MAP と Wasserstein 距離の対応の説明のみ）．

**4. 軸との対応**
- A1 線形の頭だけ：△．凍結した特徴の上だが，分類器はクラスの中心（距離）であって線形の重みと切片ではない．
- A2：×．A3：×．A4：×．
- A5：○．s ∈ {1, 5}（クラスあたり）．K を振るのは 2 点のみ．
- A6：△．miniImageNet → CUB のドメインをまたぐ評価が 1 表ある．ベースと新しいクラスは重ならない（クラス集合は違う）．
- A7：×．
- A8：△．先生は無い（ソースの分類器の出力を使わない）．ラベル無し全体を使う点，**各クラスの総質量をそろえる Sinkhorn の列の制約**（q 本ずつ）は本研究の列正規化と同じ系統．重みを K ごとに：×（α と n_steps を s=1 と 5 で別に調整する程度）．
- A9：×．
- A10：ハイパーパラメータは検証クラスで選び，検証と新しいクラスの曲線が同じ β，λ でピークになることを示す（図 4）．検定：信頼区間のみ．
- A11：△．クラスの比率の偏り（図 3）を扱う．ただし「比率を知っている／おおよそ知っている」場合だけで，「一様と誤って仮定した」場合は示さない．ドメインの差：表 4 の 1 組のみ．

**5. 穴**
- 列の制約（各クラス q 本）が間違っているときの損は示していない（Veilleux ほかが後に指摘）．
- 先生（ソースの分類器）の出力は使わず，距離で割り当てを作るので，本研究の「学び直さない先生の弱いラベル」とは出所が違う．
- 限界の節は無い．結論は「few hyperparameters」で実用的と述べるのみ．
- 今後の課題の記述は無い．

**6. 本研究との差の書き方（案）**
転導的な少数ショット学習の PT-MAP（Hu ほか，2021）は，凍結した特徴の上でラベル無し標本とクラスの中心の最適輸送を Sinkhorn で解き，列の制約で各クラスに同じ本数（q）を割り当てる．また TAFSSL に由来する trans-mean-sub で，ラベル付きとラベル無しの平均を別々に引く．本研究の列正規化は Sinkhorn の列のスケーリングを 1 回だけ行う簡略形にあたり，中心化はソースとターゲットという別のドメインの平均を別々に引く点で対象が異なる．PT-MAP は各クラスの本数が既知であることを前提とし，比率を誤ったときの損は示していない．

**7. 注意**
- 「trans-mean-sub」は PT-MAP の 4.2 節で TAFSSL [18] からの借用と明記されている．関連研究で「PT-MAP の trans-mean-sub」と書くなら「TAFSSL に由来し PT-MAP が採用した」と書く．
- 掲載先（ICANN 2021，LNCS，pp. 487–499）は PDF には無く，DBLP の記録の検索結果と DOI で確かめた（○）．Springer の頁そのものは開いていない．
- 図 3 の数値は図からしか読めず，本文に数字は無い．

#### 6. BD-CSPN — Liu, Song, Qin, "Prototype Rectification for Few-Shot Learning"

**1. 書誌と読んだ範囲**
- 書誌（確かめた）：Jinlu Liu, Liang Song, Yongqiang Qin（AInnovation Technology）．arXiv:1911.10713v4（2020-07-13）．PDF は LNCS の体裁で会議名の表記は無い．ECCV 2020 の掲載は，arXiv の抄録頁の Comments 欄 "ECCV 2020 Oral" と，TransCLIP の参考文献 [36]（"Computer Vision–ECCV 2020 ... Part I 16, pages 741–756"）で確かめた（○）．
- 読んだ範囲：◎ 本文全節（1〜6．3.1〜3.3 手法，4.1〜4.2 理論，5.1〜5.6 実験）と付録 A.1〜A.6（実装，Omniglot・CUB，骨格の ablation，多クラス，ロバスト性，Meta-Dataset）．読めなかった節：なし．
- 出典：[arXiv PDF](https://arxiv.org/pdf/1911.10713)，[ar5iv](https://ar5iv.labs.arxiv.org/html/1911.10713)．

**2. 設定**
- 課題：転導的な少数ショット分類．miniImageNet，tieredImageNet，Meta-Dataset（ILSVRC で学び 10 のテスト源で評価），付録で Omniglot・CUB．ベースのクラスと少数ショットのクラスは別．
- モダリティ：画像のみ．
- ラベル数と単位：N-way K-shot（5-way，K=1 か 5，問い合わせ 15 本／クラス）．600 エピソードの平均と 95% 信頼区間．
- 学習器：ベースのクラスでコサイン分類器つきの特徴抽出器（WRN-28-10 など）を学び，推論ではそれを凍結して**クラスの平均（原型）とのコサイン類似度**で分類．線形の重みの学習は無い（"retraining F_θ and classification weights on the scarce data ... is likely to run into overfitting. To avoid it, we directly compute the basic prototype"）．
- 評価：正解率．Z（疑似ラベルの本数）は検証集合で選ぶ（5.5 節）．

**3. 示したこと**
- 2 つの偏り（3.3 節）：クラス内の偏り B_intra ＝（そのクラスの全データの期待値）−（手元のラベル付きデータの期待値）（式 4）．クラス間の偏り B_cross ＝（支持集合の平均）−（問い合わせ集合の平均）（式 7）．後者は「ドメイン適応で平均を一次の統計量としてデータセットを表す」考え方から来ると明記（"derived from the domain adaptation problem where the mean value is used as a type of the first order statistic"）．ただし支持集合と問い合わせ集合は「同じドメインにあると仮定しているので，両者の距離はドメインの差ではなく分布の偏り」と述べる．
- **平均のずれの補正（式 8）**：正規化した問い合わせの特徴に ξ ＝ (1/|S|) Σ X_s − (1/|Q|) Σ X_q を足す（問い合わせ集合を支持集合の中心に寄せる）．4.2 節で，各特徴を X = P + ε と書き，問い合わせのコサイン類似度を最大にする共通のずれ ξ が −E[ε] になることを導く（式 20〜24）．
- **疑似ラベルで原型を直す形（式 5〜6）**：クラスごとに確信度の上位 Z 本の問い合わせ標本を支持集合に足し，基本の原型とのコサイン類似度の softmax（温度 ε=10）を重みにした重みつき和で原型を作り直す．Z は検証集合で決め，8 を使う．
- 理論（4.1 節）：各次元が独立と仮定して，期待コサイン類似度の下界 ≈ Σ E[x_i]² / sqrt((1/T) Σ Var[x_i] + Σ E[x_i]²)（式 16）．T = K + Z（標本の本数）が増えると下界が上がる．これが疑似ラベルを足す根拠．式 25 で Acc ≈ η / sqrt(λ/(K+Z) + α) とし，Z の理論曲線と実験の曲線の傾向が一致する（図 2c）．
- 主な結果（表 1・2，WRN-28-10）：miniImageNet 1-shot 70.31，5-shot 81.89；tieredImageNet 78.74／86.92．ablation（表 4）：CSPN（補正なし）61.84／78.64 → クラス間の補正だけ 62.54／79.32 → クラス内（疑似ラベル）だけ 69.81／81.58 → 両方 70.31／81.89．**得の大半は疑似ラベル（+8.0 点，1-shot）で，平均のずれの補正は +0.7 点（1-shot miniImageNet），+1.64（1-shot tieredImageNet）．**
- ロバスト性（A.5）：5 クラス以外の標本をラベル無しに混ぜると 1-shot で 3.43〜5.73 点落ちる（表 13）．
- Meta-Dataset（表 3・14）：ILSVRC だけで学び 10 源で評価，5-shot の平均順位 1.9．

**4. 軸との対応**
- A1 線形の頭だけ：△．凍結した特徴の上だが，分類器は原型とのコサイン類似度（線形の重みの学習は無い）．
- A2 切片と向き：△．平均のずれ（支持と問い合わせの差）を補正する点で「切片」に相当する偏りを扱うが，分類器の切片ではなく特徴の平行移動．**K で減る誤差の式はある**（式 16・25：下界は標本数 T=K+Z の関数）．
- A3：×．A4：×．
- A5：○．K ∈ {1, 5}（クラスあたり）．
- A6：△．Meta-Dataset の「ILSVRC で学び別の源で評価」がドメインをまたぐ評価にあたるが，クラス集合は違う．
- A7：×．
- A8：○（素朴な版）．先生（ベースで学んだ特徴＋支持集合の原型）を学び直さず，問い合わせ（ラベル無し）全体に当てて**硬い**疑似ラベルを付け，**クラスごとに確信度の上位 Z 本**を使う（クラス均衡は上位 k 本の形）．柔らかい重みは原型とのコサインの softmax．重みを K ごとに選ぶ：×（Z=8 固定）．
- A9：×．
- A10：Z は検証集合で選ぶ．検定：信頼区間のみ．相手の調整：公表値の転載．
- A11：×．得が消える条件の分析は無い（A.5 の外れクラスの混入のみ）．

**5. 穴**
- 平均のずれの補正は同じドメインの支持集合と問い合わせ集合の間の偏りを対象にし，「ドメインの差ではない」と自分で書いている．本研究のソースとターゲットの中心化とは対象が違う．
- 疑似ラベルは硬く，上位 Z 本だけを使う．先生の出力の確率を全標本に重みつきで使う形ではない．
- 理論は「標本が増えれば下界が上がる」だけで，疑似ラベルが誤っているときの損は式に入っていない（各次元の独立性も仮定）．
- 限界・今後の課題の節は無い．

**6. 本研究との差の書き方（案）**
BD-CSPN（Liu ほか，ECCV 2020）は，転導的な少数ショット学習で，問い合わせの特徴に「支持集合の平均 − 問い合わせの平均」を足して平均のずれを消し，確信度の上位 Z 本の疑似ラベルで原型を直す．ずれの対象は同じドメインの中のラベル付き集合とラベル無し集合であり，理論は標本数が増えると期待コサイン類似度の下界が上がることを示すにとどまる．本研究はソースとターゲットという別のコーパスの平均を別々に引き，疑似ラベルの誤りが切片と向きの誤差にどう入るかを式で扱う点が異なる．

**7. 注意**
- ECCV 2020（Oral）の掲載は PDF には無く，arXiv の Comments 欄と他論文の参考文献で確かめた（○）．
- 結論の「8.47% on 1-shot miniImageNet」は表 4 の 61.84 → 70.31 の差（8.47）と一致．「9.54% on 1-shot tieredImageNet」は 69.20 → 78.74 で一致．

#### 7. UPL — Huang, Chu, Wei, "Unsupervised Prompt Learning for Vision-Language Models"

**1. 書誌と読んだ範囲**
- 書誌（確かめた）：Tony Huang, Jack Chu（北京大学），Fangyun Wei（Microsoft Research Asia）．arXiv:2204.03649v2（2022-08-22）．PDF は AAAI の体裁だが会議名の表記は無い．**掲載先は未確認（要確認）．**TransCLIP の参考文献 [25] も "arXiv preprint arXiv:2204.03649, 2022" と引いている．
- 読んだ範囲：◎ 本文全節（Introduction，Related Work，Method，Experiment（Implementation Details，Main Results，Ablation Study），Conclusion）と補足（More Experiments：他の枠組みへの適用，[CLS] の位置，単一プロンプトとの比較，最近傍の語，データセットとプロンプトの表 10，図 9）．
- 出典：[arXiv PDF](https://arxiv.org/pdf/2204.03649)，[ar5iv](https://ar5iv.labs.arxiv.org/html/2204.03649)．

**2. 設定**
- 課題：ターゲットのデータセットに**ラベルが無い**設定で，CLIP のプロンプト（連続表現 V）を学ぶ．ImageNet ＋ 10 データセット．
- モダリティ：画像とテキスト（CLIP）．
- ラベル数と単位：0（教師なし）．比較相手の CoOp・Tip-Adapter は 2／4／8-shot（クラスあたり）．
- 学習器：CLIP の画像・テキストのエンコーダは凍結し，クラスで共有する長さ L=16 のプロンプト表現だけを学ぶ（線形の頭ではない）．推論用の模型は ResNet-50 の CLIP．
- 評価：正解率（11 データセット）．ハイパーパラメータの選び方の記述は無い（K=16 は「実験的に」）．

**3. 示したこと**
- 手法：(1) CLIP の最も単純なプロンプト（例 "a photo of a [CLS]"）で，ターゲットのラベル無し画像に疑似ラベル（argmax）を付ける．(2) **各クラスで確信度の上位 K=16 本を選ぶ**（閾値ではなく）．(3) 選んだ標本の交差エントロピーでプロンプト表現を学ぶ．
- **上位 K 本を選ぶ理由（原文）**："1) vision-language models have biased preferences for different classes, using a pre-defined threshold to filter out unconfident samples leads to an imbalanced pseudo data distribution; 2) there is no obvious correlation between confidence scores and pseudo label accuracy"．偏りの原因は「事前学習のデータとターゲットのデータのドメインの差」（"which is mainly caused by the domain gap between the pre-training dataset and the target dataset"）．
- 閾値との比較（表 2，UCF101）：確信度 >0.3 で 63.83，>0.9 で 57.32，上位 16 本で 64.84．「高い閾値（0.9）は疑似ラベルのクラスの分布を偏らせ，低い閾値（0.3）は雑音を入れすぎる」．K を 2／4／8／16 と増やすと成績は上がる（図 6）ので 16 にした．
- 主な結果（表 1，11 データセット平均）：CLIP（プロンプト工学あり）59.18 → UPL 63.38（+4.2）→ UPL*（複数の CLIP で疑似ラベルを集団化）68.37．UPL は 2-shot CoOp（62.97）を上回り，UPL* は 8-shot CoOp（69.15）・8-shot Tip-Adapter（68.16）に近い．EuroSAT 38.23 → 54.83 → 71.04．
- 疑似ラベルの雑音への頑健性（図 8）：クラスごとの疑似ラベルの正解率と転移の改善に明確な対応は無く，疑似ラベルの正解率が低いクラスでも改善する．理由は「全クラスが同じプロンプト表現を共有するから」．
- 理論：無し．

**4. 軸との対応**
- A1：×．学ぶのはプロンプト表現（テキスト側の入力）．
- A2：×．A3：×．A4：×．
- A5：×．ラベルは 0 本（少数ショットの設定ではない）．K を振らない．
- A6：△．「事前学習データとターゲットのドメインの差」を疑似ラベルの偏りの原因として述べるが，ソースのラベル付きデータからの転移ではない．クラス集合はプロンプトで指定するので同じ．
- A7：○（画像＋テキスト）．テキスト側のゼロショット予測が先生，学ぶのはテキスト側のプロンプトなので「融合の先生 → 単モーダルの生徒」ではない．
- A8：○．学び直さない先生（CLIP のゼロショット）の出力をターゲットのラベル無し全体に当て，**硬い**疑似ラベルにする．クラス均衡は**上位 k 本（K=16）**．重みを K ごとに：×．
- A9：×．
- A10：△．ハイパーパラメータの選び方は書かれていない（K=16 は実験的）．検定：無し．相手の数値は転載．
- A11：△．先生の出力のクラスの偏り（図 3）を示し，その補正として上位 k 本を提案．ドメインの差の大きさの関数としては扱わない．

**5. 穴**
- 少数のラベルを併用する設定は無い（TransCLIP が CoOp+UPL として補った）．
- 柔らかい疑似ラベル・総重量の選択は無い．
- 掲載先が確認できない．
- 限界・今後の課題の節は無い．

**6. 本研究との差の書き方（案）**
UPL（Huang ほか，2022）は，CLIP のゼロショット予測がターゲットのデータセットでクラスごとに偏ることを示し，閾値ではなく各クラスで確信度の上位 16 本を疑似ラベルに採ってプロンプトを学ぶ．本研究の列正規化も同じ動機（学び直さない先生の出力のクラスの偏り）に基づくが，硬いラベルの本数ではなく柔らかいラベルの各クラスの総量をそろえ，その総重量をラベルの本数 K ごとに選ぶ点が異なる．

**7. 注意**
- 掲載先（会議・雑誌）は PDF でも他論文の引用でも確認できない．引用するなら arXiv preprint として書く．
- 表 1 の UPL* は疑似ラベルの生成に ViT-L/14 など大きい CLIP を使うので，CoOp 8-shot との「競合」は同じ骨格の比較ではない．

#### 8. AdaMatch — Berthelot, Roelofs, Sohn, Carlini, Kurakin, "AdaMatch: A Unified Approach to Semi-Supervised Learning and Domain Adaptation"

**1. 書誌と読んだ範囲**
- 書誌（確かめた）：David Berthelot, Rebecca Roelofs（equal contribution），Kihyuk Sohn, Nicholas Carlini, Alex Kurakin（Google Research）．arXiv:2106.04732v2（2022-03-15）．PDF の各頁の頭に "Published as a conference paper at ICLR 2022" とあり，ICLR 2022 を確認（◎）．
- 読んだ範囲：◎ 本文全節（1〜7．3.1 手法，3.2 ハイパーパラメータ，3.3 SSL・SSDA への拡張，4 設定，5.1〜5.3 結果，6 ablation，Ethics statement）と付録 A（動機と例），B（BaselineBN），C（事前学習の曲線），D.1〜D.2（SSL・SSDA のまとめの表 5・6）．付録 E（データセットの組ごとの表 7 以降．2,420 の実験）は読んでいない．
- 出典：[arXiv PDF](https://arxiv.org/pdf/2106.04732)，[ar5iv](https://ar5iv.labs.arxiv.org/html/2106.04732)．

**2. 設定**
- 課題：UDA（ソースのラベル＋ターゲットのラベル無し），SSL（同じドメイン），SSDA（ソースのラベル＋ターゲットの少数ラベル＋ターゲットのラベル無し）を 1 つの方法で解く．DigitFive（5 ドメイン，20 組）と DomainNet（6 ドメイン，345 クラス，30 組．64×64 と 224×224）．「ソースとターゲットは同じ分類課題で，クラス数 k は同じ」（3 節）．
- モダリティ：画像のみ．
- ラベル数と単位：SSL・SSDA でターゲットのラベル 1／5／10／25 本（表 5・6 の列．**"labels per class" と明記されているのは抄録の "one labeled example per class"**）．
- 学習器：WRN-28-2（32×32），WRN-34-2（64×64），ResNetV2-101（224×224）を**ゼロから**学ぶ（深層．事前学習は「歴史的な比較のため」だけ）．
- 評価：最後の 10 チェックポイントの中央値の正解率を，全ソース→ターゲットの組で平均．検定なし．

**3. 示したこと**
- 手法（3.1 節）：FixMatch を土台に，(1) ランダムな logit 補間（ソースだけのバッチと，ソース＋ターゲットのバッチで BN の統計が違う 2 つの logit を一様乱数で混ぜて，ソースの交差エントロピーを計算），(2) **分布整合**，(3) 相対的な確信度の閾値，を足す．弱い拡張（平行移動＋反転）と強い拡張（＋cutout）を使う．
- **分布整合の形（式 5〜6）**：弱い拡張のターゲットの予測 Ŷ_TU,w に，E[Ŷ_SL,w]／E[Ŷ_TU,w]（ソースの予測の平均をターゲットの予測の平均で割った比）を要素ごとに掛け，正規化して和を 1 にする．結果としてターゲットの疑似ラベルの平均はソースの予測の平均に一致する．ReMixMatch と違い，ソースの**真のラベルの分布ではなく模型の出力の平均**を使う（"the model may not be capable of matching the ground truth source label distribution"）．"If the target label distribution is known, one can simply replace the term E[Ŷ_SL,w] with it"．**自分で書いている限界**："the more the source label distribution differs from the target distribution, the more incorrect the approximation will be, which may cause the model performance to degrade"．付録 A の例：ソースの予測の平均 {0.3, 0.7}，ターゲット {0.6, 0.4} なら，比 {0.5, 1.75} を掛ける．
- 相対的な閾値（式 7〜8）：ソースの弱い拡張の top-1 確信度の平均 × τ（τ=0.9）を超えるターゲットの標本だけを使う（模型の較正がドメインで違うため）．
- 損失（式 9〜11）：ソースの交差エントロピー（弱・強の両方）＋ μ(t) × ターゲットの疑似ラベルの交差エントロピー（強い拡張に対し，stop-gradient，mask つき）．μ(t) は前半で 0 → 1 の warmup．
- ハイパーパラメータ（3.2 節）：τ=0.9 と uratio=3 を「全実験で同じ」．τ は 64×64 の UDA の 1 組（clipart→infograph）で {0.8, 0.9, 0.95} から選んだ（4 節）．
- **SSDA（ターゲットの少数ラベル）の形（3.3 節）**：ソースのラベル付きバッチにターゲットのラベル付きを**連結するだけ**（"feeding the concatenated batch X_STL = {X_SL, X_TL} in place of X_SL"）．出発点への罰則や K ごとの設定の変更は無い．
- 主な結果：UDA（表 2）で DomainNet224 の平均 28.7（NoisyStudent 23.9，FixMatch+ 20.8，MCD 14.9，BaselineBN 18.9，Oracle 65.9）．SSDA（表 6，DomainNet224）：ターゲットのラベル 1／5／10／25 本で AdaMatch 34.8／42.3／46.7／52.8（FixMatch+ 28.3／38.7／44.1／50.8，BaselineBN 25.3／33.8／39.3／47.6）．抄録の「1 本で +6.1，5 本で +13.6」は UDA の 28.7 からの差（34.8−28.7=6.1，42.3−28.7=13.6）．"In very low label regime, AdaMatch further improves its lead over other methods."
- ablation（表 4，DomainNet64 の 6 組）：全部 26.3，logit 補間なし 25.2，**分布整合なし 17.7**，相対閾値なし 23.2．分布整合を外すと最も落ちる．
- 感度（図 4）：高い閾値が要る，uratio が大きいほど良い，weight decay は 1e-4〜1e-3 で安定．
- 事前学習（表 3，付録 C）：ImageNet 事前学習の AdaMatch は早期終了が要る（8M で 28.2，2M で 33.4，最大 35.6）．
- 理論：無し．

**4. 軸との対応**
- A1：×．深層網をゼロから学ぶ．
- A2：×．A3：×．A4：×．
- A5：○．SSDA でターゲットのラベル 1／5／10／25 本を振る（クラスあたり．抄録の表記による）．
- A6：○．ドメインをまたぐ．クラス集合は同じ．
- A7：×．
- A8：△．先生は学び直さない固定の先生ではなく，学習中の模型自身（弱い拡張の予測）．ラベル無し全体に（閾値つきで）当てる．**硬い**疑似ラベル（分布整合の後で argmax）．クラス均衡：**分布整合（列平均の比を掛けて行を正規化．目標はソースの予測の平均）**．重みは μ(t) の warmup のみで K ごとには変えない．
- A9：×．ターゲットのラベルはソースのバッチに連結するだけ．
- A10：○（設定の固定）．τ と uratio を全実験で固定．相手（MCD，FixMatch+，NoisyStudent）は同じコードで再実装．検定なし．検証データ：τ の選択に 1 組の UDA を使った（ターゲットのラベルを使ったかは明記なし）．
- A11：△．得が消える条件：「ソースとターゲットのラベルの分布が違うほど分布整合の近似が悪くなる」と自分で書くが，実験で測っていない．ドメインの差：SSL より SSDA・UDA で得が大きい（5.2 節）．

**5. 穴**
- 弱い拡張と強い拡張（データ拡張）に依存する（凍結した特徴の上の線形分類器では作れない）．
- 分布整合の目標はソースの予測の平均で，ターゲットのクラスの比率がソースと違うときの損は測っていない（限界として自分で書く．上の引用）．
- SSDA の少数ラベルは「連結」だけで，出発点への罰則や K ごとの重みの調整は無い．
- 今後の課題（7 節，原文）："which is more important for future progress, making use of unlabeled data more efficiently, or attempting to better model domain shifts?"．Ethics statement："the confirmation bias of self-training (Arazo et al., 2019) could be a concern for building a fair ML model across major and minor classes. AdaMatch partially resolves the issue with the distribution alignment, but more in-depth investigation ... should be done in the future."

**6. 本研究との差の書き方（案）**
AdaMatch（Berthelot ほか，ICLR 2022）は，ソースのラベルとターゲットのラベル無しデータから深層網をゼロから学ぶ FixMatch の拡張で，ターゲットの疑似ラベルにソースの予測の平均とターゲットの予測の平均の比を掛けて正規化する分布整合を持ち，ターゲットの少数ラベル（クラスあたり 1〜25 本）はソースのバッチに連結して使う．弱い拡張と強い拡張を前提とし，設定は全実験で固定する．本研究は凍結した特徴の上の線形の頭で，学び直さない先生の柔らかいラベルの列を一様にそろえ，その総重量と出発点への罰則を K ごとに選ぶ点が異なる．AdaMatch 自身が「ソースとターゲットのラベルの分布が違うほど分布整合の近似は悪くなる」と述べており，本研究の列正規化にも同じ注意が当てはまる．

**7. 注意**
- 表 5・6 の列見出し（1／5／10／25）が「クラスあたり」か「合計」かは表には書かれていない．抄録の "one labeled example per class" と 5.3 節の "randomly sample a subset of the target dataset and treat it as labeled" から，クラスあたりと読むのが自然だが，表の注記では確認できない（要確認）．
- 抄録の "+6.1" と "+13.6" は UDA の 28.7 を基準にした SSDA の値（表 6）で，表 2・6 から再計算して一致した．
- NoisyStudent の比較相手としての実装は「ソースで評価するだけで分布のずれを使わない」と述べている．

#### 9. Veilleux, Boudiaf, Piantanida, Ben Ayed, "Realistic Evaluation of Transductive Few-Shot Learning"

**1. 書誌と読んだ範囲**
- 書誌（確かめた）：Olivier Veilleux, Malik Boudiaf（ÉTS Montreal，equal contribution），Pablo Piantanida（L2S CentraleSupélec），Ismail Ben Ayed（ÉTS）．arXiv:2204.11181v1（2022-04-24）．PDF の 1 頁目に "35th Conference on Neural Information Processing Systems (NeurIPS 2021)" とあり，NeurIPS 2021 を確認（◎）．
- 読んだ範囲：◎ 本文全節（1〜6，Conclusion．2 設定，3 ディリクレ分布の周辺確率，4 均衡の偏りの分析，5 α ダイバージェンス，6 実験 6.1〜6.2）と付録 A〜G（均衡した設定での α-TIM，DeepEMD との比較，α エントロピーの導出，他の不均衡との比較，各項の ablation（表 4），ハイパーパラメータの検証の図，コード）．図 4（不均衡の強さ a を振った曲線）の数値は読めない．
- 出典：[arXiv PDF](https://arxiv.org/pdf/2204.11181)．

**2. 設定**
- 課題：転導的な少数ショット分類の評価の見直し．mini-ImageNet，tiered-ImageNet，CUB．ベースと検証とテストのクラスは全部別．
- モダリティ：画像のみ．
- ラベル数と単位：5-way 1／5／10／20-shot（クラスあたり）．問い合わせは 75 本に固定し，**クラスの比率をディリクレ分布 Dir(a·1_K) から引く**（テストは a=2，検証は a=1）．10,000 課題の平均．
- 学習器：凍結した特徴抽出器（ResNet-18，WRN28-10．ベースのクラスで交差エントロピー＋ラベル平滑化）の上で，推論時に分類器 f_W を学ぶ．実験では p_ik ∝ exp(−(τ/2)‖w_k − z_i‖²)（正規化した特徴とクラスの重みの距離）．
- 評価：正解率．ハイパーパラメータは各データセットの検証集合（a=1 の不均衡な課題）で調整．「クラス均衡の偏りに直接つながる」と判定したもの（TIM の周辺エントロピーの重み λ）は再調整する．

**3. 示したこと**
- 問題の指摘（1 節，原文）："The current few-shot benchmarks use perfectly class-balanced tasks at inference"；"using perfectly balanced query sets for benchmarking the models assumes exact knowledge of the marginal distributions of the true labels of the testing points, but such labels are unknown. This is, undeniably, an unrealistic assumption"．
- ディリクレ分布での偏らせ方（3 節）：各クラスの比率 p = (p_1,…,p_K) を Dir(a) から引き，n_k = p_k|Q| に最も近い整数にする．a → ∞ が従来の完全に均衡した設定．Gamma 乱数の正規化で生成（定理 3.1）．
- **均衡の仮定への警告（4 節，原文）**：PT-MAP について "Setting c = (1/K)1_K as done in [31] ensures that M defines a valid joint distribution, but also crucially encodes the strong prior that all the classes within the query sets are equally likely. Such a hard constraint is detrimental to the performance in more realistic scenarios where the class distributions of the query sets could be arbitrary"．TIM について "minimizing marginal entropy H(Y_Q) is equivalent to minimizing the KL divergence between the predicted marginal distribution and uniform distribution u_K"（式 2）；"This KL penalty could harm the performances whenever the class distribution of the few-shot task is no longer uniform"．
- **どの方法がどれだけ落ちるか（表 1・2，a=2，括弧は均衡した設定からの変化）**：
  - PT-MAP：mini-ImageNet RN-18 1-shot 60.1（↓16.8），5-shot 67.1（↓18.2）；WRN 5-shot 66.8（↓19.8）；tiered 1-shot 64.1（↓18.8）；CUB 1-shot 65.1（↓20.4）．本文："PT-MAP undergoes a substantial performance drop ... with a consistent decrease in accuracy between 18 and 20 % on all benchmarks, in the 5-ways case"．**帰納的な Simpleshot（63.0／80.1）を大きく下回る．**
  - Entropy-min（転導的微調整）：↓5.1〜8.0（mini），本文「約 7%」．
  - TIM：↓3.7〜4.5（λ を再調整した後．"keeping the same hyper-parameters for TIM as prescribed in the original paper would result in a drastic drop of about 18%"）．
  - LaplacianShot：1-shot で ↓4.7〜5.3，5-shot 以上ではほぼ変わらず（↑0.5 もある）．
  - BD-CSPN：↓1.1〜3.4（最も落ちにくい転導的な方法の 1 つ）．
  - MAML ↓3.6〜5.7，Versa ↓2.2〜4.0，SIB ↓5.3〜8.7，LR+ICI ↓1.7〜8.1．
  - 本文："more than half of the transductive methods evaluated perform overall worse than inductive baselines in our realistic setting"．
- 提案（5 節）：相互情報量の周辺エントロピーを Tsallis の α エントロピーに置き換えた α-TIM（式 5〜6）．α > 1 で一様分布からのずれを許しつつ，自明解（全部同じクラス）は防ぐ．mini-ImageNet RN-18 で 1-shot 67.4，5-shot 82.5，20-shot 87.9（帰納の Simpleshot 86.1 を上回る）．α を外しても急には落ちない（図 3）．均衡した検証集合で α を選ぶと α ≈ 1 になり TIM と同じになる（付録 A）．
- 不均衡の強さを振った結果（6.2 節，図 4）：a を小さくすると（偏りが強いと）TIM と PT-MAP が最も急に落ち，"past a certain imbalance severity, the inductive baseline in [33] becomes more competitive than most transductive methods"．LaplacianShot と α-TIM は極端な不均衡にも耐える．
- 転導的 BN（MAML 系）も不均衡で損なう（6.2 節）．
- 理論：定理 3.1（ディリクレ分布の生成）のみ．汎化の理論は無い．

**4. 軸との対応**
- A1 線形の頭だけ：△．凍結した特徴の上で推論時に分類器（距離に基づく）を学ぶ．線形の重みと切片の形ではない．
- A2：×．A3：×．A4：×．
- A5：○．1／5／10／20-shot（クラスあたり）．
- A6：×．ドメインをまたがない（クラスは別）．
- A7：×．
- A8：△．先生は無いが，転導的な方法（TIM，PT-MAP）がラベル無し全体の予測の周辺分布を一様に寄せる操作を持ち，**その均衡の仮定が偏った比率で損になることを示した**．
- A9：×．
- A10：○．検証集合（a=1 の不均衡な課題）でハイパーパラメータを選び，偏りに直結する項の重みは相手も再調整する流儀．検定：無し（10,000 課題の平均）．
- A11：○（クラスの比率の偏り）．**得が消える条件として「ラベル無し集合のクラスの比率の偏り」を直接扱い，偏りの強さ a の関数として示した．**ドメインの差：×．

**5. 穴**
- 偏りを「ラベル無し（問い合わせ）集合」に入れ，支持集合は均衡のまま．"Future works could study the mixed effect of imbalance on both support and query sets."（結論）．
- 真の比率を推定して均衡の制約に入れる（PT-MAP の q を合わせる）方向は扱っていない（PT-MAP 側は比率が分かれば効くと示している）．
- ドメインのずれは扱わない．
- 限界の節は無く，結論に今後の課題が 1 文．

**6. 本研究との差の書き方（案）**
Veilleux ほか（NeurIPS 2021）は，転導的な少数ショット学習のベンチマークが問い合わせ集合を完全に均衡させており，クラスの比率をディリクレ分布で偏らせると，各クラスに同じ本数を割り当てる PT-MAP が 18〜20 点，周辺エントロピーを一様に寄せる TIM が約 4 点（再調整後．元の設定では約 18 点）落ち，半数以上の転導的な方法が帰納的な方法を下回ると報告した．本研究の列正規化は同じ均衡の仮定を持つため，ターゲットのコーパスのクラスの比率の偏りと列正規化の得の関係を示す必要がある．

**7. 注意**
- TIM の落ち幅は「λ を再調整した後」の値であり，元の設定なら約 18% 落ちる（6.2 節）．引用するときは条件を書く．
- 図 4 の数値は本文に無い．
- arXiv の投稿（2022-04）は NeurIPS 2021 の採録より後なので，年は 2021 で引く．

#### 10. Noisy Student — Xie, Luong, Hovy, Le, "Self-training with Noisy Student improves ImageNet classification"

**1. 書誌と読んだ範囲**
- 書誌（確かめた）：Qizhe Xie, Minh-Thang Luong, Quoc V. Le（Google Brain），Eduard Hovy（CMU）．arXiv:1911.04252v4（2020-06-19）．PDF に会議名の表記は無い（CVPR の体裁）．CVPR 2020 は arXiv の抄録頁の Comments 欄 "CVPR 2020" と，STARTUP・UPL の参考文献（"CVPR, pp. 10687–10698, 2020"）で確かめた（○）．
- 読んだ範囲：◎ 本文全節（1〜6．2 手法，3.1〜3.4 実験，4.1〜4.3 ablation，5 関連研究）と付録 A.2（Study #1〜#8）．A.1（構造），A.3，A.4（YFCC100M）は読んでいない．
- 出典：[arXiv PDF](https://arxiv.org/pdf/1911.04252)，[ar5iv](https://ar5iv.labs.arxiv.org/html/1911.04252)．

**2. 設定**
- 課題：ImageNet の半教師あり学習（ラベル 1.3M ＋ ラベル無し JFT 300M）．評価は ImageNet と頑健性の集合（ImageNet-A，C，P）．
- モダリティ：画像のみ．
- ラベル数と単位：ImageNet 全体（少数ショットではない）．
- 学習器：EfficientNet-B0〜L2 の全体を学ぶ（深層）．生徒は先生と同じか大きい．
- 評価：top-1／top-5．検定なし．

**3. 示したこと**
- 手法（Algorithm 1）：(1) ラベル付きで先生を学ぶ．(2) 雑音を入れない先生で，きれいなラベル無し画像に柔らかいか硬い疑似ラベルを付ける．(3) 生徒を「ラベル付きの交差エントロピーの平均 ＋ ラベル無しの交差エントロピーの平均」で，雑音（RandAugment，dropout，stochastic depth）を入れて学ぶ．(4) 生徒を先生にして繰り返す．
- **クラスごとの件数の均衡化の形（3.1 節，原文）**："we run an EfficientNet-B0 trained on ImageNet over the JFT dataset to predict a label for each image. We then select images that have confidence of the label higher than 0.3. For each class, we select at most 130K images that have the highest confidence. Finally, for classes that have less than 130K images, we duplicate some images at random so that each class can have 130K images."．理由："To ensure that the distribution of the unlabeled images match that of the training set, we also need to balance the number of unlabeled images for each class, as all classes in ImageNet have a similar number of labeled images"．合計 130M（重複を含む．一意は 81M）．"We do not tune these hyperparameters extensively since our method is highly robust to them."
- 均衡化の効き（Study #5，表 12）：均衡化なしとの差は B0 で 77.9 vs 77.6，B1 で 79.9 vs 79.6，B2 で 80.7 vs 80.6，B3 で 82.1 vs 82.1．「小さい模型では効くが，大きい模型では差が縮む」（"more powerful models can learn from unbalanced data effectively"）．
- 柔らかい vs 硬い疑似ラベル（Study #3，図 5）：先生の確信度の区間 [p, p+0.1] ごとに 1.3M 本を取って学ぶと，確信度が高い（in-domain）区間では両方効くが，確信度が低い（out-of-domain）区間では硬いラベルは損ない，柔らかいラベルは頑健．ただし「大きい先生なら硬いラベルが同等かやや良いこともある」．既定は柔らかいラベル．
- 主な結果：EfficientNet-L2 で 88.4%（従来 86.4%）．ImageNet-A 61.0 → 83.7，ImageNet-C mCE 45.7 → 28.3，ImageNet-P mFR 27.8 → 12.2．B0〜B7 で一貫して約 0.8% 改善（図 2）．
- 雑音の役割（4.1 節）：雑音を外すと落ちる（表 6．130M で 85.1 → 84.3）．先生に雑音を入れると落ちる．
- その他の知見（4.3 節）：大きい先生ほど良い（#1），ラベル無しは多いほど良い（#2），生徒は大きいほど良い（#4），同時学習 > 事前学習＋微調整（#6），生徒をゼロから学ぶ方が良いことがある（#8）．
- 理論：無し．

**4. 軸との対応**
- A1：×．深層網の全体．
- A2：×．A3：×．A4：×．
- A5：×．ラベルは ImageNet 全体．
- A6：△．ラベル無し（JFT）はソース（ImageNet）と分布が違うが，評価はソースのドメイン（AdaMatch が「分布のずれを利用せず，ソースでしか評価しない」と指摘）．クラス集合は ImageNet の 1000 に合わせる（確信度で絞る）．
- A7：×．
- A8：○．学び直さない先生の出力（**柔らかい確率が既定**，硬いも可）をラベル無し全体に付ける．クラス均衡：**クラスごとに 130K 本にそろえる（上位 k 本＋複製）**．重みは 1（ラベル付きとラベル無しの交差エントロピーの和．バッチの比で調整）．K ごと：×．
- A9：×（生徒はゼロから学ぶのが既定）．
- A10：△．ハイパーパラメータは「広く調整していない」と述べる．検定なし．
- A11：△．得が消える条件：均衡化は大きい模型では効かない（表 12）；硬いラベルは out-of-domain で損なう（図 5）．クラスの比率の偏り：均衡化の有無の差のみ．

**5. 穴**
- 少数ラベル・ドメインをまたぐ評価・線形の頭はいずれも無い．
- 均衡化は「ラベル付きのクラスの比率が一様だから一様にそろえる」という理由で，ターゲットの比率が違う場合は扱わない．
- 限界・今後の課題の節は無い．

**6. 本研究との差の書き方（案）**
Noisy Student（Xie ほか，2020）は，先生の確信度の低い画像を捨て，各クラスを上位 13 万枚にそろえて（足りないクラスは複製）疑似ラベルの集合を作り，柔らかいラベルを既定にして大きな生徒を学ぶ．均衡化の理由はラベル付きの ImageNet のクラスの比率が一様であることであり，その効果は小さい模型に限られる．本研究の列正規化は同じ「クラスごとの量をそろえる」操作を，本数ではなく柔らかいラベルの総量で行い，ターゲットの比率が一様でないときの損を測る点が異なる．

**7. 注意**
- CVPR 2020 の掲載は PDF には無く，arXiv の Comments 欄と他論文の参考文献で確かめた（○）．
- 均衡化の効き（表 12）は差が 0.1〜0.3 点で小さく，検定も無い．「均衡化が効く」と強く引かない．

#### 11（余力）. xMUDA — Jaritz, Vu, de Charette, Wirbel, Pérez, "xMUDA: Cross-Modal Unsupervised Domain Adaptation for 3D Semantic Segmentation"

**1. 書誌と読んだ範囲**
- 書誌（確かめた）：Maximilian Jaritz（Inria，Valeo DAR，Valeo.ai），Tuan-Hung Vu，Raoul de Charette，Émilie Wirbel，Patrick Pérez．arXiv:1911.12676v2（2020-03-30）．PDF に会議名の表記は無い（CVPR の体裁）．CVPR 2020 は arXiv の抄録頁の Comments 欄 "Accepted at CVPR 2020." と前回のノートの CVF の頁で確かめた（○）．
- 読んだ範囲：◎ 本文全節（1〜6．3.1 構造，3.2 学習（式 1〜5），3.3 議論，4.1〜4.4 実験（融合への拡張を含む），5.1〜5.3 ablation）と付録 A（データの分割）．
- 出典：[arXiv PDF](https://arxiv.org/pdf/1911.12676)．

**2. 設定**
- 課題：3D 意味分割の教師なしドメイン適応（UDA）．ソースはラベル付きの 2D 画像＋3D 点群，ターゲットはラベル無しの両モダリティ．場面は Day→Night，USA→Singapore（nuScenes），A2D2→SemanticKITTI（10 の共通クラス）．
- モダリティ：2D 画像（U-Net＋ResNet34）と 3D 点群（SparseConvNet）．**推論でも両方使う**（"we here assume all modalities are available at train and test time on both source and target domains"）．
- ラベル数と単位：ターゲットのラベルは 0（UDA）．
- 学習器：深層網の 2 系統を同時に学ぶ．
- 評価：mIoU．検証集合で最良のチェックポイントを選ぶ（疑似ラベルの生成だけは最後のチェックポイントを使う）．

**3. 示したこと**
- 手法（3.2 節）：各モダリティに主の頭ともう 1 つの「まね」の頭（mimicry head）を置き，**まねの頭が相手のモダリティの主の予測を KL で推定する**（式 2〜3．D_KL(P_3D‖P_2D→3D) と D_KL(P_2D‖P_3D→2D)）．この損失はラベルが要らないのでソースとターゲットの両方にかける（重み λ_s，λ_t．式 4）．主の頭と分けるのは「強いモダリティが弱いモダリティの誤った予測を取り込まない」ため（"to prevent the stronger modality from adopting false predictions from the weaker one"）．1 つの頭で直接合わせると λ_t が大きいときに急に落ちる（図 5b）．
- 疑似ラベル（xMUDA_PL，式 5）：学んだ xMUDA の模型でターゲットにクラスごとの閾値で疑似ラベルを付け，ゼロから学び直す（Li ほか 2019 に従う）．
- **融合の先生から単モーダルの生徒への蒸留の形（4.4 節，表 2）**：(a) "xMUDA Fusion"：2D と 3D の特徴を連結した融合の頭 P_fuse を置き，各モダリティに P_2D→fuse，P_3D→fuse の頭を足して**融合の出力をまねさせる**（KL(P_fuse‖P_2D→fuse) など．図 4b）．(b) "Distillation"（表 2 の 1 行）："we use the xMUDA_PL model of the main experiments ... to generate pseudo-labels from the softmax average and use those to train the Vanilla Fusion network"．つまり 2D と 3D の softmax の平均（集団）を先生にして融合の網を学ぶ．
- 主な結果（表 1，mIoU）：USA/Singapore で source only 2D 53.4／3D 46.5／平均 61.3 → xMUDA 59.3／52.0／62.7 → xMUDA_PL 61.1／54.1／63.2（PL だけ 55.5／51.8／61.5）．A2D2/SemanticKITTI で source only 36.0／36.6／41.8 → xMUDA_PL 43.7／48.5／49.1．融合（表 2，USA/Singapore）：Vanilla 59.9，PL 65.2，Distillation 65.8，xMUDA Fusion 61.9，xMUDA_PL Fusion 66.6，Oracle 72.2．
- ソースにもまねの損失をかけると 2D で +4.8，3D で +4.4（5.2 節）．教師ありの oracle にも効く（表 3）．
- 理論：無し．

**4. 軸との対応**
- A1：×．A2：×．A3：×．A4：×．
- A5：×（ターゲットのラベル 0）．
- A6：○（ドメインをまたぐ．クラス集合は同じ 10 クラスにそろえる）．
- A7：○．2 つのモダリティ．**融合の出力を各モダリティの頭にまねさせる形（xMUDA Fusion）と，単モーダルの softmax 平均を先生に融合の網を学ぶ形（Distillation）の両方がある．**ただし推論では両モダリティを使う（単モーダルの生徒だけで推論する設定ではない）．
- A8：△．先生は学び直さない固定の先生ではなく，同時に学ぶ相手のモダリティの主の予測（確率）．ターゲットのラベル無し全体にかける．クラス均衡：×（疑似ラベルのクラスごとの閾値と，損失のクラス重みのみ）．
- A9：×．
- A10：検証集合でチェックポイントを選ぶ．相手は「最良のハイパーパラメータを探した」．検定なし．
- A11：△．場面ごとに「どちらのモダリティがずれに弱いか」を述べる（Day/Night では camera が弱い）が，差の大きさの関数としては扱わない．

**5. 穴**
- ターゲットの少数ラベルは扱わない（雑誌版 TPAMI 2022 の xMoSSDA が半教師ありの設定を足したと前回のノートにあるが，今回は本文未読）．
- 融合の先生から**単モーダルだけで推論する**生徒への蒸留ではない（推論は両モダリティ）．
- 線形の頭でも凍結したエンコーダでもない．
- 限界の節は無い．結論に「他のモダリティや教師あり学習にも使えるはず」とだけ書く．

**6. 本研究との差の書き方（案）**
xMUDA（Jaritz ほか，CVPR 2020）は，2D と 3D の深層網をターゲットのラベル無しデータ上で互いの予測に KL で合わせ，融合の出力を各モダリティにまねさせる拡張（xMUDA Fusion）も示した．先生は同時に学ぶ相手の網であり，推論でも両モダリティを使う．本研究は，ソースのコーパスだけで学んで固定した融合の線形分類器を先生とし，推論で単モーダルだけを使う線形の生徒に弱いラベルを渡す点が異なる．

**7. 注意**
- CVPR 2020 の掲載は PDF には無く，arXiv の Comments 欄で確かめた（○）．
- fusion-then-distillation の 3 本（ACM MM 2022・2023，IEEE TCSVT 2025）は本調べでも本文を取れていない（下の「余力の調べ」参照）．

#### 12（余力）. fusion-then-distillation の系統 — Wu, Xing, Zhang, Xie, Qu, "Fusion-then-Distillation: Toward Cross-modal Positive Distillation for Domain Adaptive 3D Semantic Segmentation"（FtD++）

**1. 書誌と読んだ範囲**
- 書誌：Yao Wu, Mingwei Xing, Yachao Zhang, Yuan Xie, Yanyun Qu．arXiv:2410.19446v1（2024-10）．IEEE TCSVT vol. 35, no. 9, pp. 9030–9045（2025）は検索結果（ECNU の機関リポジトリの記録）による（△．出版社の頁は未確認）．会議版は "Cross-modal Unsupervised Domain Adaptation for 3D Semantic Segmentation via Bidirectional Fusion-then-Distillation"（BFtD，ACM MM 2023，DOI 10.1145/3581783.3612013．△，本文未読）．ACM MM 2022 の "Cross-Domain and Cross-Modal Knowledge Distillation in Domain Adaptation for 3D Semantic Segmentation" は arXiv 版が見つからず，本文未読（△）．
- 読んだ範囲：○（arXiv HTML の前半 10 万字を WebFetch の要約で読んだ．式の形と表の数値は要約経由で，直接の精読ではないので ◎ にはしない）．
- 出典：[arXiv HTML](https://arxiv.org/html/2410.19446v1)，[arXiv PDF](https://arxiv.org/pdf/2410.19446)．

**2. 設定**
- 課題：xMUDA と同じ 3D 意味分割のドメイン適応（UDA と半教師あり SSDA）．Day→Night，USA→Singapore，A2D2→SemanticKITTI，vKITTI→SemanticKITTI．
- モダリティ：2D 画像と 3D 点群．推論は 2D のみ，3D のみ，両方の平均（xM）の 3 つを報告（単モーダルだけで推論する設定の明記は要約からは分からない）．
- ラベル数と単位：SSDA はフレーム数で指定（USA→Singapore でラベル付き 2,884 フレーム，ラベル無し 6,781；A2D2→SemanticKITTI で 5,642 と 32,738）．クラスあたりの少数ではない．
- 学習器：深層網（同時学習）．

**3. 示したこと（要約経由）**
- 融合（MFFM）：2D と 3D の特徴を連結しアダプタで融合の特徴を作る．
- 蒸留（xP-Distill）：(a) Modality-Preserving Distillation：融合の特徴を生の 2D の特徴に連結して 2D の分類器に入れ，各モダリティに「まね」の頭を置き，**2D と 3D の予測の対称な KL**（式 14〜15）をソースとターゲットの両方にかける．会議版 BFtD では「まねの頭を融合の分類器の出力に合わせる」（つまり**融合 → 単モーダルの蒸留**）だったと，差として書かれている．(b) Domain-Preserving Distillation：先生は生徒の EMA で，先生の 2D と 3D の予測の**平均**をターゲットの様式に変えたソースのデータで生徒に合わせる．
- 疑似ラベル（xDPL）：1 段目の模型の疑似ラベルを，相手のモダリティの予測とそのまねの頭の KL の指数で重みづけ（動的な閾値）．**クラス均衡の記述は無い．**
- 主な結果（mIoU，xM）：Day→Night で xMUDA 50.6，BFtD 60.9，FtD++ 60.0；USA→Singapore で 63.3／66.8／68.2．SSDA（表 V）：USA→Singapore で FtD++ 79.8 vs ラベル付きターゲットのみ 74.2，xMoSSDA 78.5；A2D2→SemanticKITTI 70.7 vs xMoSSDA 65.9．
- 限界：専用の節は無い．一般の疑似ラベルでは「模型が不安定で退化しやすかった」と述べる．

**4. 軸との対応（要約に基づく暫定）**
- A1：×．A2：×．A3：×．A4：×．A5：×（ラベルはフレーム数で多い）．A6：○．A7：○（融合 → 単モーダルの形は会議版 BFtD に，FtD++ は対称な KL と EMA の先生の平均）．A8：△（先生は EMA の自分自身．クラス均衡なし）．A9：×．A10：△．A11：×．

**5〜6. 穴と本研究との差（案）**
- 先生は同時に学ぶ網（または EMA）であり，ソースだけで学んで固定した融合の分類器ではない．ターゲットのラベルは少数（クラスあたり数本）ではない．線形の頭でも凍結したエンコーダでもない．
- 「融合の出力を単モーダルの頭にまねさせる」形は xMUDA Fusion（2020）と BFtD（2023）にあるので，本研究の部品 5（融合と単モーダルの往復）を新しいとは書けない．残る差は，往復が同時ではなく二段構えであること，参照が単モーダルの積であること，K 本と弱いラベルで学ぶ線形の生徒であること．

**7. 注意**
- 本文を直接は読んでいない（WebFetch の要約）．引用の前に PDF を読むこと．特に SSDA でターゲットのラベルがどう損失に入るかは要約では分からなかった．
- ACM MM 2022 の論文は本文も抄録も取れていない．


### Inferences（Q1）
- 「学び直さない先生の柔らかい出力をターゲットのラベル無し全体に付ける」操作は STARTUP・TransCLIP・Noisy Student に，「クラスごとの量をそろえる」操作は UPL・Noisy Student・PT-MAP・AdaMatch に，「出発点への L2 罰則つきの線形プローブ」は CLAP にある．本研究の部品をどれか 1 つだけ取り出して「新しい」と書くことはできない．
- 一方で，10 本のどれも，(i) 線形の頭に限った誤差の分解（切片と向き）と K の関数の式，(ii) ソースのドメインで学んで固定した先生（STARTUP だけが該当するが，クラス集合が違うので分類器には使えない），(iii) 同じクラス集合のターゲットで弱いラベルを分類器そのものに使うこと，(iv) 弱いラベルの重みを K ごとに選ぶこと，を同時には持たない．
- 先生の出力のクラスの偏りの原因を「ドメインの差」と明記するのは UPL（"caused by the domain gap between the pre-training dataset and the target dataset"）で，列正規化の動機として引ける．ただし均衡の仮定の損は Veilleux が示しているので，動機と警告を並べて書く．

### Gaps（Q1）
- UPL の掲載先は PDF・arXiv・検索のいずれでも確認できなかった．
- MM-CDFSL の少数ショットの段でエンコーダを凍結するかは本文に明記が無い．
- AdaMatch の表 5・6 の列（1／5／10／25）が「クラスあたり」かは表の注記に無く，抄録の "per class" から推定した．
- TransCLIP の付録の表 8〜23，CLAP の表 9〜11 の 4-shot 以降，AdaMatch の付録 E，Noisy Student の A.1・A.3・A.4 は読んでいない．
- FtD++ は WebFetch の要約で読んだだけで，SSDA でターゲットのラベルが損失にどう入るかは分からなかった．ACM MM 2022 の論文は本文も抄録も取れていない．

## Q2. 10 本 × 本研究の軸 A1〜A11 の表

### Takeaway
A2〜A4（理論の 3 列）は 10 本すべて × である（BD-CSPN だけが標本数の関数の下界を持つので A2 を △）．A1（線形の頭だけ）で ○ は CLAP のみ．A9（出発点への L2 罰則）で ○ も CLAP のみ．A8（学び直さない先生の弱いラベル）で ○ は STARTUP・TransCLIP・BD-CSPN・UPL・Noisy Student の 5 本で，確率を使うのは STARTUP・TransCLIP・Noisy Student，硬いラベルは BD-CSPN・UPL．クラスの比率の偏りを得が消える条件として測ったのは Veilleux だけ．

### Cited Findings
各欄は「判定：根拠 1 行」．根拠は Q1 の各論文の記録（出典はそこに書いた）による．

**表 A（A1〜A6）**

| 論文 | A1 線形の頭だけ | A2 切片と向き／K の式 | A3 ソースの価値を本数に換算 | A4 合併の切片 | A5 K 本・K を振る・単位 | A6 ドメインをまたぐ／クラス集合 |
|---|---|---|---|---|---|---|
| 1 STARTUP | △：評価は凍結した表現の上の線形だが，本体は ResNet の微調整 | ×：理論なし | × | × | ○：5-way k ∈ {1,5,20,50}，クラスあたり | ○：miniImageNet／ImageNet → BSCD-FSL．クラスは **disjoint** |
| 2 MM-CDFSL | △：最後の段だけ頭を学ぶ．本体は ViT-S の微調整 | × | × | × | ○：5-way 1／5-shot，クラスあたり | ○：Ego4D → EPIC／MECCANO／WEAR．クラスは**重なり無し** |
| 3 TransCLIP | ×：凍結した埋め込みの上だが GMM の割り当て | ×：理論は BMM の収束（定理 1）のみ | × | × | ○：0／1／4／16-shot，クラスあたり | △：ImageNet の変種と cross-dataset．ソースのラベル付きからの転移ではない．クラスは同じ |
| 4 CLAP | ○：凍結 CLIP の上の線形プローブ（コサイン，**切片なし**） | ×：経験的に「得は K で縮む」のみ | × | × | ○：K ∈ {1,2,4,8,16}，クラスあたり | △：ImageNet で適応し変種で評価．クラスは同じ |
| 5 PT-MAP | △：凍結した特徴の上の中心（距離）．線形の重みではない | × | × | × | ○：s ∈ {1,5}，クラスあたり | △：miniImageNet → CUB が 1 表．クラスは別 |
| 6 BD-CSPN | △：凍結した特徴の上の原型（コサイン） | △：平均のずれ ξ を補正．下界 ∝ 1/sqrt(λ/(K+Z)+α)（式 16・25）．切片と向きの分解は無い | × | × | ○：K ∈ {1,5}，クラスあたり | △：Meta-Dataset（ILSVRC で学び 10 源で評価）．クラスは別 |
| 7 UPL | ×：学ぶのはプロンプト表現 | × | × | × | ×：ラベル 0 本 | △：事前学習データとターゲットの差を偏りの原因と述べる．クラスは同じ |
| 8 AdaMatch | ×：深層網をゼロから | × | × | × | ○：SSDA でターゲットのラベル 1／5／10／25（クラスあたりは抄録の表記．要確認） | ○：DigitFive，DomainNet．クラスは同じ |
| 9 Veilleux | △：凍結した特徴の上で推論時に分類器（距離）を学ぶ | ×：定理 3.1 はディリクレ乱数の生成 | × | × | ○：1／5／10／20-shot，クラスあたり | ×：ドメインはまたがない．クラスは別 |
| 10 Noisy Student | ×：EfficientNet 全体 | × | × | × | ×：ImageNet 全体のラベル | △：JFT は別分布だが評価はソース（ImageNet）．クラスは同じ 1000 |

**表 B（A7〜A11）**

| 論文 | A7 マルチモーダル／融合の先生 → 単モーダルの生徒 | A8 学び直さない先生の弱いラベル（確率か特徴か／均衡の形／K ごとの重み） | A9 出発点への L2 罰則（強さの決め方） | A10 評価（相手の調整／検証データ／検定） | A11 得が生じる・消える条件（比率の偏り／ドメインの差） |
|---|---|---|---|---|---|
| 1 STARTUP | ×：画像のみ | ○：ソースだけの先生の**確率**（温度 1）を D_u 全体に KL で／均衡 ×／重みは 3 項の和で固定 | ×：先生の埋め込みで初期化するだけ（weight decay 1e-4） | ○：学習率を内部検証（ラベル無し 10%＋ラベル 5%）で選ぶ／t 検定 0.05／Transfer は再実装 | △：ドメインが近い miniImageNet では「助けにも害にもならない」．先生の AMI が高いほど得（表 3）．比率 × |
| 2 MM-CDFSL | ○：RGB・光学流・手の姿勢．**モダリティごとの先生 → RGB の生徒**（融合ではない） | △：先生はターゲットのラベル無しでも学ぶ／**特徴**を L2 で／均衡 ×／重み 1 で固定 | × | △：λ_ce を評価と同じ EPIC 5-shot で掃いた（表 6）／信頼区間のみ／相手は VideoMAE 事前学習に置き換え | ×：λ_ce の高低で落ちる ablation のみ |
| 3 TransCLIP | ○：画像＋テキスト．テキストの先生 → 画像の割り当て（融合ではない） | ○：ゼロショットの**確率**への KL をラベル無し全体に／均衡 △（均等な混合を仮定）／λ=1（0-shot）と 0.5（few-shot）の 2 段階固定．γ は min(4,#shots) 本の検証で | ×：KL であって L2 ではない | ○：相手は各データセット・各ショットで検証ショットで格子探索／自分は λ 固定／検定なし | △：ショットが増えると得が減る（表 1・4，付録 D）．テキストが弱いデータセットでは 16-shot で w/o text が勝つ．比率 × |
| 4 CLAP | △：テキストは出発点（先生ではない） | ×：ラベル無しを使わない | ○：Σ_c λ_c‖t_c−w_c‖²．λ_c はクラスの支持標本のゼロショット確率の平均（ALM の外側 1 回） | ○：検証集合を使わず全設定固定／相手も同流儀で再評価／検定なし | △：得は K で縮む（6 節）．罰則なし ZS-LP はドメインのずれでゼロショット以下（表 2）．比率 × |
| 5 PT-MAP | ×：画像のみ | △：先生なし．ラベル無し全体の Sinkhorn で**各クラス q 本**（列の制約）／重みは s で α と n_steps を変える程度 | × | ○：検証クラスで β，λ，α を選ぶ（図 4）／検定なし | △：比率が既知またはおおよそ既知なら帰納版を上回る（図 3）．誤った一様の仮定の損は示さない |
| 6 BD-CSPN | ×：画像のみ | ○（素朴）：先生（原型）を学び直さず問い合わせ全体に**硬い**ラベル／**クラスごと上位 Z=8 本**／重みはコサインの softmax | × | △：Z は検証集合／相手は転載／検定なし | ×：外れクラスの混入（A.5）のみ |
| 7 UPL | ○：画像＋テキスト．テキストの先生 → プロンプト | ○：ゼロショットの**硬い**ラベル／**クラスごと上位 16 本**（閾値だと偏る，表 2）／K ごと × | × | △：選び方の記述なし／検定なし／相手は転載 | △：先生のクラスの偏り（図 3）を示す．原因は "domain gap"．差の大きさの関数 × |
| 8 AdaMatch | ×：画像のみ | △：先生は学習中の自分（弱い拡張）／**硬い**（分布整合後に argmax，閾値つき）／**分布整合**（E[ソースの予測]/E[ターゲットの予測] を掛けて行を正規化）／μ(t) の warmup のみ | ×：ターゲットのラベルはソースのバッチに連結するだけ | ○：τ=0.9，uratio=3 を全実験で固定／相手は同じコードで再実装／検定なし | △：「ソースとターゲットのラベルの分布が違うほど近似が悪い」と自分で書くが測らず．SSL より SSDA・UDA で得が大きい |
| 9 Veilleux | ×：画像のみ | △：先生なし．転導的な方法の均衡の仮定（PT-MAP の列制約，TIM の周辺エントロピー）の損を示す | × | ○：不均衡な検証課題（a=1）で調整／偏りに直結する項は相手も再調整／検定なし | ○：比率を Dir(a) で偏らせ，PT-MAP −18〜20，Entropy-min 約 −7，TIM 約 −4（再調整後）．a の関数（図 4）．ドメインの差 × |
| 10 Noisy Student | ×：画像のみ | ○：先生の**柔らかい確率**（既定）をラベル無し全体に／確信度 >0.3 で絞り**クラスごと 13 万本**（上位＋複製）／重み 1 固定 | ×（生徒はゼロから） | △：「広く調整していない」／検定なし | △：均衡化は小さい模型でだけ効く（表 12）．硬いラベルは out-of-domain で損（図 5）．比率 × |

### Inferences（Q2）
- 表 A の A2〜A4 が全列 × であることが，本研究の理論（線形の頭に限った切片と向きの分解，K の式，ソースの価値の換算，合併の切片）の置き場所である．ただし「理論がある」だけでは弱いので，理論の予言が測定と合うことを示す必要がある．
- 表 B の A8 で ○ の 5 本は，確率（STARTUP・TransCLIP・Noisy Student）と硬いラベル（BD-CSPN・UPL）に分かれ，均衡の形は「本数（上位 k 本・複製）」が 3 本，「列の制約（Sinkhorn）」が PT-MAP，「分布整合（比を掛ける）」が AdaMatch，「均等な混合の仮定」が TransCLIP である．本研究の列正規化（柔らかいラベルの各クラスの総量をそろえ，行は正規化しない）は，この 4 種のどれとも同じではないが，Sinkhorn の列のスケーリング 1 回分として説明できる．
- A9 と A8 を同時に ○ にする論文は無い（CLAP は A8 ×，TransCLIP は A9 ×）．「出発点への L2 罰則」と「学び直さない先生の弱いラベル」を同じ線形の頭で同時に使うのは，この 10 本の中には無い．

### Gaps（Q2）
- 表の判定は本文の記述に基づく．図からしか読めない数値（PT-MAP の図 3，Veilleux の図 4）は判定の根拠にしていない．
- AdaMatch の A5 の単位は要確認．

## Q3. この群の先行研究に空いている穴（本研究の主張の置き場所）

### Takeaway
10 本を合わせても空いているのは次の 6 つである．(1) 線形の頭に限った誤差の理論（切片と向き，K の式，ソースの価値の換算，合併の切片）．(2) ソースのドメインだけで学んで固定した分類器を，同じクラス集合のターゲットのラベル無し全体の弱いラベルとして，線形の分類器そのものの学習に使うこと（STARTUP はクラス集合が違うので表現にしか使えず，TransCLIP の先生は同じ模型のテキスト側，AdaMatch の先生は学習中の自分）．(3) 融合の線形の先生から，単モーダルだけで推論する線形の生徒への弱いラベル（MM-CDFSL はモダリティごとの先生で特徴，xMUDA・BFtD は同時に学ぶ深層網で推論も両モダリティ）．(4) 弱いラベルの重みと出発点への罰則を K ごとに選び，オラクルとの差を K ごとに示すこと（TransCLIP は 0 本と少数の 2 段階，CLAP はクラスごとに支持標本から，他は固定）．(5) ターゲットのクラスの比率の偏りの関数として，均衡の操作の得と損を，ソースで学んだ先生の設定で測ること（Veilleux は先生なしの転導的な方法で，AdaMatch は限界として書くだけ）．(6) 感情認識（音声・顔・言葉の 3 モダリティ，話者を分けた評価，UAR）での測定（10 本はすべて画像か画像＋テキスト）．

### Cited Findings
- **理論の穴**：10 本のうち理論を持つのは BD-CSPN（期待コサイン類似度の下界が標本数 T=K+Z の関数．式 16．各次元の独立を仮定）と TransCLIP（BMM の収束．定理 1）だけで，どちらもドメインのずれ・切片・ソースの価値を扱わない（Q1 の 3 と 6）．
- **先生の出所の穴**：ソースのラベル付きデータだけで学んだ先生を学び直さずにターゲットのラベル無し全体に当てるのは STARTUP だけだが，"we are interested in transferring to novel domains with a wholly different label space"（3 節）であり，弱いラベルは表現にしか使えない（Q1 の 1）．TransCLIP の先生はテキスト側のゼロショット（Q1 の 3），UPL も同じ（Q1 の 7），AdaMatch と Noisy Student の先生は学習中（または前の反復）の自分（Q1 の 8・10），MM-CDFSL の先生はターゲットのラベル無しで学ぶ（Q1 の 2）．
- **線形の頭の穴**：凍結した特徴の上の線形の頭を学ぶのは CLAP だけで，ラベル無しデータを使わず切片も持たない（Q1 の 4）．TransCLIP・PT-MAP・BD-CSPN・Veilleux は凍結した特徴の上だが，分類器は GMM の割り当て・中心・原型（Q1 の 3・5・6・9）．
- **融合の先生の穴**：MM-CDFSL は "Each teacher model is trained independently on source and target data for its respective modality" で特徴を蒸留し（Q1 の 2），xMUDA Fusion は融合の出力を各モダリティの頭にまねさせるが推論は両モダリティ（Q1 の 11），FtD++／BFtD も同時に学ぶ深層網（Q1 の 12）．
- **K ごとの選択の穴**：TransCLIP は "λ = 1 for all our zero-shot experiments ... and λ = 0.5 in all the few-shot experiments"（Q1 の 3），CLAP は λ_c を支持標本から 1 回だけ更新（Q1 の 4），AdaMatch は "τ (set to 0.9 for all experiments)"（Q1 の 8），MM-CDFSL は "constant loss weights ... regardless of the specific target dataset" を限界と書く（Q1 の 2）．K の関数として重みの最適値がどう動くかを示した論文は無い．
- **比率の偏りの穴**：Veilleux は転導的な方法が "perfectly class-balanced tasks" に依存することを示し PT-MAP が 18〜20 点落ちると報告（Q1 の 9）．AdaMatch は "the more the source label distribution differs from the target distribution, the more incorrect the approximation will be" と書くが測っていない（Q1 の 8）．PT-MAP は比率が分かれば転導版が勝つと示す（Q1 の 5）．Noisy Student の均衡化は差が 0.1〜0.3 点で，大きい模型では消える（Q1 の 10）．ソースで学んだ先生の弱いラベルを均衡させたときの得が，ターゲットの比率の偏りでどう変わるかを測った論文は無い．
- **評価の穴**：検定を使うのは STARTUP（t 検定）だけ．検証データの扱いは，検証なし（CLAP），K+min(K,4) 本の検証ショット（TransCLIP の相手），評価と同じベンチマークで掃く（MM-CDFSL の λ_ce），内部検証（STARTUP）と分かれる．CLAP は「別のデータセットで調整した設定は最大 21% 落ちる」（図 1）と報告する（Q1 の 4）．
- **領域の穴**：10 本はすべて画像（STARTUP，MM-CDFSL，PT-MAP，BD-CSPN，AdaMatch，Veilleux，Noisy Student，xMUDA）か画像＋テキスト（TransCLIP，CLAP，UPL）で，音声・顔・言葉の感情認識，話者を分けた分割，UAR は無い．

### Inferences
**関連研究に書いてよい差（この 10 本に対して）**
- 本研究は，凍結したエンコーダの上の線形の頭に限ることで，ラベルの誤差を切片と向きに分け，K 本で減る誤差の式と，ソースの価値をターゲットのラベルの本数に換算する式を立てた（A2〜A4 は 10 本すべて ×）．
- 本研究の先生は，ソースのコーパスだけで学んで固定した融合の線形分類器であり，同じ 4 クラスのターゲットの発話全体に確率の弱いラベルを付け，それを単モーダルの線形の生徒の分類器そのものの学習に使う（STARTUP は表現にしか使えず，TransCLIP の先生は同じ模型のテキスト側，MM-CDFSL の先生はターゲットで学び特徴を蒸留する）．
- 弱いラベルの総重量と出発点への罰則の強さを，ラベルの本数 K ごとに仮のターゲットで選び，オラクルとの差を K ごとに示す（TransCLIP の 2 段階，CLAP の支持標本からの 1 回更新，AdaMatch の固定，との差）．CLAP の「最大 21%」の報告に答えるため，差の数値を出す．
- 列正規化（柔らかいラベルの各クラスの総量を一様にそろえる）は，UPL・Noisy Student の上位 k 本，PT-MAP の Sinkhorn の列の制約，AdaMatch の分布整合と同じ動機の簡略形と書き，Veilleux の警告に答えて，ターゲットのクラスの比率の偏りと得の関係を示す．

**書いてはいけないこと（この 10 本が既に示した）**
- 「学び直さない先生の確率をラベル無し全体に付けるのは新しい」（STARTUP，TransCLIP，Noisy Student）．
- 「クラスごとの量をそろえるのは新しい」（UPL，Noisy Student，PT-MAP，AdaMatch）．
- 「出発点への L2 罰則つきの線形プローブは新しい」（CLAP）．
- 「融合の出力を単モーダルの頭にまねさせるのは新しい」（xMUDA Fusion，BFtD）．
- 「均衡の仮定は安全」（Veilleux）．
- 「ソースとターゲットの平均を別々に引くのは新しい」（PT-MAP が採用した TAFSSL 由来の trans-mean-sub，BD-CSPN の ξ）．

### Gaps
- fusion-then-distillation の 3 本のうち本文を（要約で）読めたのは FtD++ だけで，ACM MM 2022 の論文は未読．部品 5（融合と単モーダルの往復）の差の書き方は，これを読んでから確定する．
- 感情認識での同じ設定の論文（前回のノートの Q5・Q6）は本調べの対象外で，今回は確かめていない．
- 10 本の「限界」の節は MM-CDFSL・TransCLIP（付録 D）・CLAP（6 節）・AdaMatch（Ethics statement）にだけあり，他は結論の 1 文か無い．引用するときは節を明示する．
