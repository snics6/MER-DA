# マルチモーダル学習の理論は，コーパス間のずれのもとで何を予言するか．DA の学習法になっているか（2026-10-28 時点）

確かめた水準の記号：◎ 本文を読んだ（pdftotext で取り出した本文か，HTML の本文．括弧内に読んだ範囲を書く），○ 抄録か学会・出版社の書誌ページ，△ 検索結果の抜き書きか記憶の書誌だけ．`docs/related-work.md` 1.11・1.12・1.14 節と `reports/TAC 論文 先行研究の精読と穴.md` に既にある文献（Lopez-Paz 2016，Gupta 2016，MKE の実験，MFH の実験節，Huang 2021，Huang 2022 の抄録，xMUDA，UMT，Parthasarathy 2020，感情認識の特権情報 Q6 の表）は繰り返さず，そこに無い情報（定理の仮定と式，ずれの扱い，2024〜2026 年の論文）を書く．調べ方はウェブ検索（標準・拡張）と，arXiv・PMLR・ACL Anthology・TTI-C からの本文取得である．中国語圏の論文誌，IEEE Xplore の本文，OpenReview（閲覧の確認画面で止まった）は探せていない．

## 問い 1．部分情報分解（PID）を，コーパス間で「どの情報が移るか」の判断や DA の学習法に使った研究はあるか

### Takeaway
PID（2 つのモダリティが感情について持つ情報を，両方が重ねて持つ分＝冗長，片方だけが持つ分＝固有，組み合わせて初めて出る分＝相乗に分ける考え方）を，マルチモーダルのコーパス間 DA の判断や学習法に使った研究は見つからなかった．近いものは，特徴の集合が違う 2 つの領域のあいだで「どの成分が失われ，どの成分が共有特徴から取り戻せるか」を PID で整理した枠組み（Gianini ほか 2024，人工データだけ）と，グラフの分布外汎化で冗長情報を最大化する学習法（RIG，NeurIPS 2025）の 2 本である．感情・感情極性のデータでは PID の値の推定が行われているが，すべて同じコーパスの中である．

### Cited Findings
- PID の元の定式化は Williams & Beer 2010（"Nonnegative Decomposition of Multivariate Information"）で，冗長の測り方は一意に決まらない．固有情報の測り方の代表は Bertschinger, Rauh, Olbrich, Jost, Ay 2014（Entropy 16(4):2161–2183）である（△ 記憶の書誌．今回は開いていない） — [Williams & Beer, arXiv 1004.2515](https://arxiv.org/abs/1004.2515)；[Bertschinger ほか 2014](https://doi.org/10.3390/e16042161)
- 共通情報に基づく冗長の測り方では，相互情報量を非負の成分に分けられないという指摘がある（△ 検索の抜き書き） — [Synergy, Redundancy and Common Information, arXiv 1509.03706](https://arxiv.org/html/1509.03706v1)
- Liang ほか（NeurIPS 2023，"Quantifying & Modeling Multimodal Interactions: An Information Decomposition Framework"）は，高次元の分布に広がる PID の推定量を 2 つ提案し，データセットの相互作用の定量化，人の注釈との照合，モデル選択に使った．事例は病理，気分（mood）の予測，ロボットの知覚の 3 つである（○ 学会ページと検索の要約） — [NeurIPS 2023 ポスター](https://neurips.cc/virtual/2023/poster/72080)；[arXiv 2302.12247](https://arxiv.org/abs/2302.12247v5)
- Liang ほか（ICLR 2024，"Multimodal Learning Without Labeled Multimodal Data: Guarantees and Applications"）は，ラベル付きの単モダリティのデータと，ラベル無しの対になったマルチモーダルのデータだけから，相乗 S の下限 2 つ（冗長に基づく S_R と，別々に学んだ単モダリティ分類器の不一致に基づく S_U）と上限（最小エントロピー結合の近似）を出した．S_U は単モダリティの分類器がベイズ最適であることを仮定する．用途は学習前の性能の推定（Fano の不等式），マルチモーダルのラベルを集めるかの判断，融合の複雑さの選択である（◎ HTML 本文の前半を要約器で読んだ．式は原典で照合していない） — [arXiv 2306.04539v2](https://arxiv.org/html/2306.04539v2)；[ICLR 2024 予稿集](https://proceedings.iclr.cc/paper_files/paper/2024/hash/63f97950e32e6bdb4c6c9e31568b08c2-Abstract-Conference.html)
- 同論文の解析は，ラベル付きの単モダリティのデータとラベル無しのマルチモーダルのデータがすべて同じ分布 p から来ると仮定し，分布のずれを扱わない（◎ 同上） — [arXiv 2306.04539v2](https://arxiv.org/html/2306.04539v2)
- 同論文の感情・感情極性の値：MOSEI は冗長が支配的（R = 0.26）で相乗が小さく（S ≈ 0.03〜0.04），単純な融合が最良．MOSI は S = 0.24 に対し S_R = 0.01，S_U = 0.03．皮肉の MUStARD は S = 0.44，不一致 0.12，S_U = 0.11 で，複雑な融合が最良（◎ 同上．MOSEI の S は要約器が 2 つの値を返したので原典で確かめる） — [arXiv 2306.04539v2](https://arxiv.org/html/2306.04539v2)
- Gianini, Barsotti, Mio, Lin（MEDES 2023，CCIS vol. 2022, pp. 133–146, 2024）は，ソース固有・ターゲット固有・共有の特徴の集合を持つ異種転移学習で，PID の構造から「ある領域から別の領域へ移るときに失われる情報成分」「共有特徴を橋にして取り戻せる成分」を判定する枠組みを出した．示したのは人工データの橋渡しの例だけである（○ 書誌ページの抄録） — [Khazna の書誌](https://khazna.ku.ac.ae/en/publications/heterogeneous-transfer-learning-fromapartial-information-decompos/)；[DOI 10.1007/978-3-031-51643-6_10](https://doi.org/10.1007/978-3-031-51643-6_10)
- Halder, Dissanayake, Dutta（NeurIPS 2025，RIG：Redundancy-guided Invariant Graph learning）は，不変な部分グラフと見せかけの部分グラフが Y について持つ情報を PID で分け，両者の冗長情報の下限を推定しながら最大化する多段の最適化で，グラフの分布外汎化を改善する．グラフに限った定式化である（○ 学会ページの抄録） — [NeurIPS 2025](https://neurips.cc/virtual/2025/133836)；[arXiv 2512.06154](https://arxiv.org/pdf/2512.06154)
- 情報分解に基づく説明可能なマルチモーダル回帰（PIDReg）は付録でガウスの分布のずれを扱う（△ 検索の抜き書き） — [arXiv 2512.22102](https://arxiv.org/html/2512.22102v1)
- Divide and Refine（Mai, Nguyen, Le，arXiv 2601.14274，2026-01．リポジトリ名から WACV 2026 と推測されるが未確認）は，会話の感情認識で各モダリティを固有・対ごとの冗長・相乗の成分に分ける学習目的を置き，IEMOCAP と MELD で評価した．PID は推定量としてではなく設計の原理として使われ，コーパス間の評価は無い（○ 抄録） — [arXiv 2601.14274](https://arxiv.org/abs/2601.14274)
- Wen ほか（arXiv 2609.14421，2026-09）は相互情報量・情報ボトルネック・PID を 1 つの枠で整理した総説（170 本）で，抄録は分布のずれに触れない（○ 抄録） — [arXiv 2609.14421](https://arxiv.org/abs/2609.14421)
- 高次元データのガウス PID の推定の偏りを補正する論文が NeurIPS 2023 にある（△ 検索結果の題名のみ） — [NeurIPS 2023 PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/ec0bff8bf4b11e36f874790046dfdb65-Paper-Conference.pdf)

### Inferences
- (a) マルチモーダルのコーパス間 DA の学習法：見つからない．(b) 近い定式化済みのもの：RIG（グラフの分布外汎化，冗長の最大化），Gianini 2024（異種転移で失われる成分の判定．学習法ではない）．(d) 感情認識：Liang 2024 の MOSEI・MOSI・MUStARD と DnR の IEMOCAP・MELD はどれもコーパス内．
- (c) 障害は 3 つある．第 1 に，PID は (X1, X2, Y) の同時分布を要するので，ターゲットのラベルがクラスあたり K 本しかないと，ターゲットの PID は推定できない．Liang 2024 の下限はラベル無しの対とラベル付きの単モダリティのデータで済むが，全データが同じ分布という仮定に立つので，ソースのラベルとターゲットのラベル無しの対を混ぜると仮定が崩れる．第 2 に，冗長の定義が一意でなく，連続で高次元の特徴（MER-DA は各エンコーダの出力の発話平均）では離散化か低次元化が要る．第 3 に，「冗長な情報ほどコーパスをまたいで移る」という命題そのものを述べた理論が無い．
- MFH（Xue 2023）の γ（モダリティ共通の決め手の特徴の割合）は PID の冗長の割合に近い量で，E60（融合の先生 → 配備するモダリティだけの生徒）の得の大小を説明する指標の候補になる．ただし MFH は γ を PID で推定していない（問い 3 参照）．

### Gaps
- PID の値（冗長・固有・相乗）がコーパスの組ごとにどう変わるか，それが転移の得を予測するかを実測した論文は見つからなかった．
- Liang 2024 の付録 B.4（不一致の下限の正式な形）と，ICLR 版での MOSEI の S の値は読んでいない．
- 探した検索語：partial information decomposition domain adaptation distribution shift multimodal／redundancy synergy unique information emotion recognition partial information decomposition／partial information decomposition transfer learning out-of-distribution synergy redundancy generalize across datasets／Liang multimodal learning without labeled multimodal data．場所：ウェブ検索（標準・拡張），arXiv，ICLR・NeurIPS の学会ページ．

## 問い 2．多視点・共訓練の理論と，その後の DA の手法．感情認識への適用

### Takeaway
多視点の理論は DA の学習法として定式化済みである．Blitzer, Foster, Kakade は 2 つの視点の CCA（正準相関分析．2 組の特徴のあいだで相関が最大になる方向を探す方法）で作った射影を使い，ターゲットのラベル無しでソースの線形の予測器を移す方法と誤差の上界を出した（TTI-C 技術報告 2009，AISTATS 2011）．共訓練の DA 版（CODA，NIPS 2011）と，expansion（ある感情のクラスの中で，誤りの集合の近傍がその集合より十分大きいという連続性の仮定）に基づくラベル伝播の DA（Cai ほか ICML 2021）もある．MKE（ICCV 2021）は expansion と条件付き独立から「マルチモーダルの生徒は先生の誤りを直す」ことを示した．コーパス間の感情認識で 2 つのモダリティの共訓練を使った研究は見つからなかった．

### Cited Findings
- Blitzer, Foster, Kakade の技術報告 TTI-TR-2009-1（"Zero-Shot Domain Adaptation: A Multi-View Approach"）の仮定 1（Identical Tasks）は，すべての領域 d で E[Y|X, D=d] = βᵀX となる共通の線形写像 β がある，というものである．仮定 2（Dimensionality Reduction）は，各領域に，Y の予測力を失わない射影 Π_d と線形写像 β_d がある，というものである．Π_d は 2 つの視点の CCA でラベル無しのデータから学び，2 つの視点の条件付き独立か冗長の仮定のもとで，射影後の最良の線形予測器が元の最良の線形予測器と一致する（Foster ほか 2008，Kakade & Foster 2007 による）（◎ 本文 1〜3 節） — [TTI-TR-2009-1](https://www.ttic.edu/technical_reports/ttic-tr-2009-1.pdf)
- 同報告の定理 4：P_{s,t} を共有部分空間への射影とすると，転移の制約を満たす推定量は Π_t P_{s,t} の方向で正しく，rank(Π_t P_{s,t}) = dim_t（ターゲットの射影のどの方向も共有方向のどれかと内積が 0 でない）なら完全に移る（perfect transfer）．ターゲットのラベルは不要で，複数ソースと少数のターゲットのラベルの場合にも広がると書かれている（◎ 同上） — [TTI-TR-2009-1](https://www.ttic.edu/technical_reports/ttic-tr-2009-1.pdf)
- 査読版は Blitzer, Kakade, Foster "Domain Adaptation with Coupled Subspaces"（AISTATS 2011，PMLR 15:173–181）で，同じ仮定 1・2 と定理 4（Π_t X_{s,t} = Π_t X_t なら完全に移る）を持ち，視点は特徴を無作為に 2 つに分けて作る．課題は製品レビューの評価予測と品詞付けである（◎ 仮定と定理 4 の文と実験の設定．証明は未読） — [PMLR 15](https://proceedings.mlr.press/v15/blitzer11a.html)
- Chen, Weinberger, Blitzer "Co-Training for Domain Adaptation"（CODA，NIPS 2011, pp. 2456–2464）は，視点への特徴の分け方を事前に決めず，ターゲットの予測器・視点の分け方・使う特徴の部分集合を 1 つの最適化で学び，確信度の高いターゲットの事例と特徴を少しずつ足す．12 領域の評価で 84 の比較のうち 65 で最良，ターゲットのラベルは 0 本から多数まで（○ 学会ページと検索の要約） — [著者 PDF](https://www.cs.cornell.edu/~kilian/papers/coda_nips2011.pdf)；[NeurIPS 2011](https://neurips.cc/virtual/2011/poster/2605)
- Balcan, Blum, Yang "Co-Training and Expansion: Towards Bridging Theory and Practice"（NIPS 2004）の書誌を確かめた（△ 書誌のみ．内容は開いていない） — [ML Anthology](https://mlanthology.org/neurips/2004/balcan2004neurips-cotraining)
- Cai, Gao, Lee, Lei "A Theory of Label Propagation for Subpopulation Shift"（ICML 2021，PMLR 139）は，ソースで学んだ先生から出発し，入力の一貫性の損失によるラベル伝播で，ターゲットに伝わるだけでなく先生を上回る分類器が得られることを，Wei ほか 2021 の expansion の仮定のもとで示し，深層網の有限標本の保証も出した．FixMatch を DA に合わせて BREEDS の ENTITY-30 で改善した．ラベル無しの別のデータを介したソースからターゲットへの転移にも広げている（○ 学会ページと抄録） — [PMLR 139](https://proceedings.mlr.press/v139/cai21b.html)；[arXiv 2102.11203](https://arxiv.org/pdf/2102.11203)
- MKE（Xue, Ren, Gao, Zhao，ICCV 2021）の 3.3 節：仮定 1 は各モダリティの入力空間で (ā, c1)・(ā, c2) の expansion（ā は先生がどれかのクラスで誤る割合の最大値，1 < min(c1, c2)，c1c2 > 5），仮定 2 は真のラベルを条件にした 2 つのモダリティの独立．補題 1 で同時の分布が (ā, c1c2) の expansion を満たし，定理 1 で，一貫性の正則化つきの損失を最小にしたマルチモーダルの生徒の誤り ≤ 4·Err(先生)/(c1c2 − 1) + 4μ となる．単モダリティ α だけの生徒は ≤ 4·Err(先生)/(c1 − 1) + 4μ で，マルチモダリティの役割は expansion の係数を大きくすることだと述べる．証明は Wei ほか 2021 に従う（◎ 3.3 節） — [arXiv 2103.14431](https://arxiv.org/abs/2103.14431)
- Pan & Woodard "Efficient Generalization via Multimodal Co-Training under Data Scarcity and Distribution Shift"（arXiv 2510.07509．本文に "Accepted for ICML 2025 Affinity Workshop: New In ML" とある）の定理 3.7 は，R(h) ≤ 経験リスク + C1√((d_eff ln(N_L/d_eff) + ln(1/δ))/N_L) − Γ(N_U/(N_L+N_U)，一致度，独立性) + C2√(ln(1/δ)/(N_L+N_U)) の形で，ラベル無しの割合・視点の一致・条件付き独立が大きいほど引き算の項 Γ が大きい．題名に分布のずれがあるが，上界にずれの項は無く，ずれは付録 A.4 で言葉で論じるだけである．実験は人工の数値例だけ（◎ 定理 3.7 と付録 A.4） — [arXiv 2510.07509](https://arxiv.org/abs/2510.07509)
- コーパス間の音声感情認識で，ラベル無しのターゲットを分類して学び直しに加える自己訓練では，ラベル無しのデータの得はラベル付きのデータの得のおよそ半分だった（△ 検索の抜き書き．単モダリティで，2 視点の共訓練ではない） — [Unsupervised Learning in Cross-Corpus Acoustic Emotion Recognition](https://mediatum.ub.tum.de/doc/1107343/687488.pdf)

### Inferences
- (a) 定式化済み．(b) 手法名：coupled subspaces（CCA で結んだ部分空間でソースの線形予測器を移す），CODA，ラベル伝播と一貫性の正則化（Cai 2021，FixMatch の DA 版），MKE（同じ分布の設定だが，定理 1 の誤りはラベル無しのデータの分布 P の上で測るので，P をターゲットにすると先生のずれは Err(先生) に入る）．(d) 感情認識：MKE は RAVDESS（同じコーパス内．related-work 1.12 Q2）で，コーパス間の 2 モダリティの共訓練は見つからない．
- MER-DA の E60 の観察「得 ≤ 先生の正解率 − 出発点の正解率」は，MKE と Cai 2021 の理論と矛盾しない．両者で生徒が先生を超えるのは一貫性の正則化（近傍で答えを揃える項）があるときで，E60 の生徒（出発点に L2 で錨づけたロジスティック回帰）にはその項が無い．MKE の式 (14) では，単モダリティの生徒の上界が先生の誤りを下回るには 4/(c1 − 1) < 1，つまり c1 > 5 が要る．
- coupled subspaces の仮定 1（全コーパスで同じ線形の E[Y|X]）は，MER-DA が実測した切片と向きのずれと両立しない可能性が高い．仮定 1 の破れの大きさを測ることが先に要る．

### Gaps
- Dasgupta, Littman, McAllester "PAC generalization bounds for co-training"（NIPS 2001）は，今回の検索で書誌を確かめられなかった（記憶では NIPS 14）．
- Balcan 2004 の expansion の定義と Wei ほか 2021 の定義の違いは，原典を開いて確かめていない．
- 探した検索語：Blitzer Foster Kakade domain adaptation coupled subspaces multi-view／Zero-shot domain adaptation: A multi-view approach／co-training for domain adaptation Chen Weinberger Blitzer NIPS 2011 CODA／expansion assumption self-training domain adaptation theory label propagation subpopulation shift／co-training cross-corpus emotion recognition audio visual unlabeled target domain／Balcan Blum Yang co-training and expansion．場所：ウェブ検索，TTI-C の技術報告，PMLR，arXiv．

## 問い 3．特権情報（LUPI）の学習率の理論，ずれがあるときに特権情報が役立つ条件，別のモダリティの先生からの蒸留の理論

### Takeaway
ずれのもとで特権情報（学習時だけ使える情報）を使う DA の学習法は定式化済みである（DALUPI，Adaptive SVM+，In-N-Out）．In-N-Out の線形の理論は「補助情報を入力に使うと分布内の誤差は下がるが分布外の誤差は上がりうる．補助情報を出力（事前学習の目標）に使うと分布外の誤差が下がる」と予言する．一方で，ソースで学んだ別のモダリティの先生をターゲットの生徒に蒸留する，という MER-DA の形をずれ込みで扱った理論は見つからなかった．別のモダリティの蒸留の理論（MFH，CCH）は同じ分布を仮定し，ずれ込みの蒸留の理論（Li ほか ICML 2026）は同じモダリティの線形回帰である．

### Cited Findings
- Pechyony & Vapnik 2010 は，特殊な問題の構成で，特権情報を使う ERM が O(1/n) の速い率を，使わない ERM が O(1/√n) の遅い率を持つと示した（△ 検索の要約．原典は開いていない） — [Stanford の解説ノート](https://web.stanford.edu/~kedart/files/lupi.pdf)；[DALUPI の関連研究の記述](https://arxiv.org/html/2303.09350)
- Sharoni & Sabato "On the Capacity Limits of Privileged ERM"（AISTATS 2023，PMLR 206:523–534）は，Pechyony & Vapnik 2010 の「特権 ERM の損失の族の VC 次元」の主張に反例を示し，正しい上下界を出した．0-1 損失で，特権情報の容量が通常の情報の容量と同程度か小さくない限り，特権 ERM の最悪の場合の保証は通常の ERM より良くならない（○ 抄録） — [PMLR 206](https://proceedings.mlr.press/v206/sharoni23a.html)
- Vapnik & Izmailov 2015（JMLR 16，"Learning Using Privileged Information: Similarity Control and Knowledge Transfer"）の PDF の所在を確かめた（△ 検索結果のみ） — [JMLR](https://jmlr.org/papers/volume16/vapnik15b/vapnik15b.pdf)
- Xie, Kumar, Jones, Khani, Ma, Liang "In-N-Out"（ICLR 2021）：補助情報を入力の特徴にすると分布内の誤差は下がるが分布外の誤差は上がりうる．補助情報を事前学習の目標にすると分布外の誤差が下がる．理論は多課題の線形回帰．提案法は，補助情報を入力に使って学び，分布内のデータに疑似ラベルを付け，分布外のデータで補助情報を出力にして事前学習し，疑似ラベルで微調整する（○ 抄録） — [arXiv 2012.04550](https://arxiv.org/abs/2012.04550)
- DALUPI（Breitholtz, Matsson, Johansson，arXiv 2303.09350v3．掲載先は arXiv のページに無い）：学習時にソースの (X, W, Y) とターゲットの (X, W) を使い，テスト時は X だけを見る．仮定はW についての共変量シフト（P(Y|W) が両領域で同じ），W についての重なり（ターゲットの W の範囲をソースが覆う），十分性（両領域で Y ⊥ X | W）で，X の重なりは要らない．推定量は f: X → W をターゲットで，g: W → Y をソースで学んで g∘f で予測する 2 段の方法．命題 2 は W の密度比を含むターゲットのリスクの上界．先行の Xie ほか 2020 の理論は線形の分類器だけ，特権情報の DA の先行は SVM（Li ほか 2022，Sarafianos ほか 2017）と書く．実験は CelebA，人工の数字，MS-COCO，胸部 X 線で，マルチモーダルと感情は無い（◎ HTML 本文の前半 10 万字を要約器で読んだ．式は原典で照合していない） — [arXiv 2303.09350](https://arxiv.org/html/2303.09350)
- Sarafianos, Vrigkas, Kakadiaris "Adaptive SVM+"（ICCV 2017 ワークショップ）は特権情報の学習（SVM+）と DA を 1 つの SVM の目的に入れ，主の情報と特権情報が同じ分布から来ないことを課題に挙げる．Animals with Attributes と INTERACT で評価（○ 抄録） — [CVF](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w38/html/Sarafianos_Adaptive_SVM_Learning_ICCV_2017_paper.html)
- MFH（Xue, Gao, Shao, Zhao，ICLR 2023）の定理 1：線形の 2 値分類，勾配流の極限で，2 つのモダリティの特徴のグラム行列の差 ‖Z^{a,T}Z^a − Z^{b,T}Z^b‖ が ε(1 − γ) 以下である確率が 1 − δ 以上なら，訓練の蒸留損失は (1 − γ) を因子に含む量で上から抑えられる（γ が大きいほど上界が小さい）．γ = d0/(d0 + d_sa + d_sb) はモダリティ共通の決め手の特徴の割合で，データは 1 つの分布 P から来る（ずれは無い）（◎ 3 節と定理 1） — [arXiv 2206.06487](https://arxiv.org/abs/2206.06487)
- Xie, Xu, Sanguinetti（arXiv 2510.13182，2025-10）の Cross-modal Complementarity Hypothesis：先生と生徒の表現の相互情報量が，生徒の表現とラベルの相互情報量を上回るとき，別のモダリティの蒸留が効く．同時ガウスのモデルで確かめ，画像・言葉・映像・音声・オミクスで試した．抄録はずれに触れない（○ 抄録） — [arXiv 2510.13182](https://arxiv.org/abs/2510.13182)
- Li, Han, Yang, Tong "Quantifying Cross-Domain Knowledge Distillation in the Presence of Domain Shift"（ICML 2026，PMLR 306:67268–67304）は，線形の先生と生徒の高次元漸近解析で，モデルのずれと共変量のずれを入れた超過リスクを出し，ソースとターゲットが大きく違っても生徒だけの学習より良くなる範囲がありうること，超過リスクが先生と生徒の次元と標本数の比について単調でない（crossed double descent）ことを示した．同じモダリティの回帰である（○ 抄録） — [PMLR 306](https://proceedings.mlr.press/v306/li26bd.html)
- Tran ほか "Cross-Modal Knowledge Distillation without Paired Data: Theoretical Foundation and Algorithm"（arXiv 2606.10504，2026-06）は，先生と生徒の分布の関係を特徴の揃い方とラベルの揃い方の 2 量で表し，対になったデータ無しの蒸留を出した．抄録はずれに触れない（○ 抄録） — [arXiv 2606.10504](https://arxiv.org/abs/2606.10504)

### Inferences
- (a) ずれのもとの特権情報：定式化済み．(b) 手法名：DALUPI（2 段の g∘f と端から端までの版），Adaptive SVM+，In-N-Out．別のモダリティの蒸留の理論：MFH，CCH，Tran 2026（同じ分布）．ずれのある蒸留の理論：Li 2026（同じモダリティ）．(c) ソースで固定した融合の先生 → ターゲットの単モダリティの生徒，の理論が無い障害は，ずれ（ソースとターゲット）とモダリティの違い（先生の入力と生徒の入力）の 2 つを同時に扱う必要があることで，各論文はどちらか一方だけを扱う．(d) 感情認識：related-work 1.12 Q6 のとおり，特権情報の感情認識はほぼすべてコーパス内で，DALUPI・In-N-Out・Li 2026 の型を感情認識に使った例は見つからなかった．
- DALUPI の g∘f の形は E60 に対応する．W を融合の特徴，g をソースだけで学んだ融合の分類器とし，X（配備するモダリティ）から g の出力をまねる生徒を学ぶと，生徒の誤りは g のターゲットでの誤りで下から抑えられる．E60 の「得 ≤ 先生 − 出発点」はこの構造と整合する．DALUPI の十分性 Y ⊥ X | W は W が X を含む（融合が音声を含む）ので自動的に満たされ，残る仮定は W についての P(Y|W) の不変性と重なりである．
- In-N-Out の予言を MER-DA に写すと，顔・言葉を融合の入力に使う（E60 の先生）だけでなく，ターゲットのラベル無しの発話で「音声から顔・言葉の特徴を予測する」写像を学んで出力側に使う手が残っている．In-N-Out の理論は線形回帰なので，凍結特徴の上の線形の頭という MER-DA の設定と型が近い．
- 依頼文の「Huang ほか 2022」は，蒸留の理論の文脈では特定できなかった．ICML 2022 の "Modality Competition"（問い 4）を指す可能性がある．

### Gaps
- Pechyony & Vapnik 2010，Vapnik & Vashist 2009，Vapnik & Izmailov 2015 の本文は開いていない．
- DALUPI の掲載先（TMLR などか）は確かめられなかった．
- Li 2026 の本文（生徒がターゲットのラベルと先生の出力をどの重みで混ぜるか，最適な混合の重みの式があるか）は読んでいない．E60 の m_eff と直接比べられる可能性があるので，最初に読むべき 1 本である．
- 探した検索語：learning using privileged information under domain shift theory when does privileged information help／privileged information domain adaptation SVM+／theory multimodal training unimodal inference provable benefit cross-modal distillation generalization bound／cross-modal knowledge distillation under domain shift teacher source domain unimodal student target theory／Quantifying Cross-Domain Knowledge Distillation．場所：ウェブ検索，arXiv，PMLR，CVF，JMLR．

## 問い 4．モダリティの競合・不均衡の理論は，コーパス間でどう変わるか

### Takeaway
モダリティの競合・不均衡の理論（Huang 2022，Wang 2020，Du 2023，Zhang ほか 2024）はすべて同じ分布の中の学習の動きを扱い，ドメインが変わると競合がどう変わるかを理論で扱った研究は見つからなかった．マルチモーダルの DA・DG で不均衡を直す手法（BOOMDA，A3D2，RNA，MER-DG，MBCD）は 2024〜2026 年に出ているが，転移の誤差の上界を持たない．感情認識では BOOMDA と A3D2 が IEMOCAP・MSP-IMPROV で評価したが，ターゲットは雑音を人工的に足して作ったもので，実際のコーパス間ではない．

### Cited Findings
- Huang ほか "Modality Competition"（ICML 2022，PMLR 162）：滑らかな ReLU の後期融合の網を勾配降下で合同に学ぶと，モダリティが競合し，一部のモダリティだけが符号化器に学ばれる．負けたモダリティが合同学習の最適でなさの原因で，扱うのは同じ分布の性能の差である（○ 検索の要約と学会ページ） — [PMLR 162](https://proceedings.mlr.press/v162/huang22e.html)；[arXiv 2203.12221](https://arxiv.org/pdf/2203.12221)
- Wang, Tran, Feiszli（CVPR 2020，"What Makes Training Multi-Modal Classification Networks Hard?"）：マルチモーダルの網は最良の単モダリティの網に負けることが多く，原因は容量が大きく過学習しやすいことと，モダリティごとに過学習と汎化の速さが違うこと．過学習と汎化の比（OGR）を指標にし，それを小さくするように監督信号を混ぜる Gradient Blending を出した（○ 抄録） — [CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_What_Makes_Training_Multi-Modal_Classification_Networks_Hard_CVPR_2020_paper.html)
- Du ほか "On Uni-Modal Feature Learning in Supervised Multi-Modal Learning"（ICML 2023，PMLR 202:8632–8656）：後期融合の合同学習は単モダリティの特徴を学び足りず，それが汎化を下げることを示し，課題の中の単モダリティの特徴と対の特徴の混じり方に応じて Uni-Modal Ensemble と Uni-Modal Teacher を選ぶ（○ 抄録） — [PMLR 202](https://proceedings.mlr.press/v202/du23e.html)
- Zhang, Latham, Saxe "Understanding Unimodal Bias in Multimodal Deep Linear Networks"（ICML 2024，PMLR 235:59100–59125）：融合する層が深いほど単モダリティだけを学ぶ期間が長く，過剰パラメータの領域では汎化の不足と単モダリティへの偏りが残る．線形の設定では早期融合に単モダリティの偏りは無い（○ 抄録） — [PMLR 235](https://proceedings.mlr.press/v235/zhang24aa.html)
- BOOMDA（Sun, Zhang, Hong, Zhu, Gao，arXiv 2511.08152，2025-11）：モダリティごとの情報ボトルネックの表現，モダリティの投票による疑似ラベル，相関の整列（CORAL）を置き，モダリティごとの整列の損失の重みを，勾配のグラム行列の対角近似から閉じた形 γ = Q̃⁻¹1/(1ᵀQ̃⁻¹1) で毎回決める（MGDA）．定理 1 はこの重みの部分問題の解で，転移の誤差の上界ではない．データは IEMOCAP と MSP-IMPROV の 4 感情（中立・喜び・悲しみ・怒り）で，ターゲットは音声に白色雑音，映像に減光と雑音，言葉の 40% を伏せて作る（◎ HTML 本文を要約器で読んだ） — [arXiv 2511.08152](https://arxiv.org/html/2511.08152)
- A3D2（Sun, Zhang, Hong, Zhu, Zeng，ACL 2025）：一部のモダリティ（anchor，例では音声）はずれず，ほかのモダリティ（drift）がずれる「部分的にずれたモダリティ」の DA．drift どうしをソースとターゲットで敵対的に揃え，ターゲットの中で drift を anchor に最適輸送で寄せる．定理は無い．データは IEMOCAP と MIntRec を半分に分け，片方にガウス雑音を足してターゲットにする（◎ 本文の 1〜3 節と実験の設定） — [ACL 2025](https://aclanthology.org/2025.acl-long.967.pdf)
- Relative Norm Alignment（Planamente ほか）は，モダリティ間の周辺分布の違いが特徴の平均のノルムの違いに現れるとして，ドメイン・モダリティ・クラスをまたいでノルムを揃える（△ 検索の抜き書き） — [IJCV 2024](https://link.springer.com/article/10.1007/s11263-024-01998-9)
- MER-DG（arXiv 2605.01967，抄録では ICML 2026 採録）：合同の最適化で符号化器がモダリティ間の共起を使い，ドメインに依らない特徴を学ばない現象を Fusion Overfitting と呼び，符号化器ごとの特徴の分布のエントロピーを最大にする項を足す．EPIC-Kitchens と HAC で評価．抄録に定理は無い（○ 抄録） — [arXiv 2605.01967](https://arxiv.org/abs/2605.01967)
- MBCD（Wang, Cheng, Zhong, Chen, Zhou，arXiv 2511.20258）：重みの平均化（WA）では収束の速いモダリティに早く過学習し，鋭い極小に寄ると論じ（定式の理論ではない），モダリティを適応的に落とす生徒と，重みの平均の先生から各単モダリティの枝への蒸留を置く（○ 抄録） — [arXiv 2511.20258](https://arxiv.org/abs/2511.20258)

### Inferences
- (a) 「コーパス間で競合がどう変わるか」の理論：見つからない．(b) 不均衡を直す DA・DG の手法：BOOMDA，A3D2，RNA，MER-DG，MBCD（いずれも経験的な手法で，重みを理論の上界から決めるものは無い）．(c) 障害：競合の理論は非凸な網の学習の動き（どのモダリティの神経細胞が先に立ち上がるか）を扱うので，ドメインのずれ（入力の分布の変化）を入れると，勝つモダリティがソースとターゲットで入れ替わるかを追う必要があり，解析がさらに難しくなる．(d) 感情認識：BOOMDA と A3D2 が IEMOCAP 系で評価したが，ずれは人工の雑音である．
- MER-DA の分類器は凍結特徴の連結の上のロジスティック回帰で，目的関数は凸である．Zhang ほか 2024 の「線形の早期融合に単モダリティの偏りは無い」に従えば，学習の動きとしての競合は MER-DA では起きにくい．MER-DA で問うべきは，L2 罰則のもとで連結した特徴のどのブロック（音声・顔・言葉）に重みが乗り，そのブロックがコーパス間でどれだけずれるか，という静的な量である．

### Gaps
- Huang 2022 の本文でデータの分布の仮定（どのモダリティが勝つかを決める量）を読んでいないので，その量がコーパス間で変わるかを推論できなかった．
- RNA の本文は開いていない．
- 探した検索語：modality imbalance under domain shift cross-domain modality competition multimodal／"modality competition" OR "modality imbalance" theory out-of-distribution generalization multimodal feature learning provably／Zhang Latham Saxe unimodal bias／Du on uni-modal feature learning／Wang Tran Feiszli gradient blending．場所：ウェブ検索，arXiv，PMLR，CVF，ACL Anthology．

## 問い 5．マルチモーダル DA の上界（モダリティごとにずれの大きさが違う場合）と，モダリティごとの適応の重みを理論から決める研究

### Takeaway
モダリティごとのずれの項を持つ DA の誤差の上界を述べた定理は見つからなかった．最も近い統計の理論は，全ソースとターゲットで観測される「参照のモダリティ」でターゲットのラベルの分布を推定し，ソースを重みづけてから揃える Wang, Dou, Liu, Cai（arXiv 2608.01282，2026-08）で，ラベルのずれとブロック状に欠けたモダリティを扱う．モダリティごとの重みを決める手法（BOOMDA の MGDA）は最適化の都合で決め，転移の上界からは決めていない．

### Cited Findings
- Wang, Dou, Liu, Cai "Multimodal domain adaptation under label shift and blockwise missing modalities"（arXiv 2608.01282，stat.ME，2026-08-02）：複数のラベル付きのソースがそれぞれ違うモダリティの部分集合を観測し，ターゲットにはラベルが無い．全ソースとターゲットで観測される参照のモダリティでターゲットのラベルの分布を推定してソースを重みづけ，補助のモダリティは CCA でターゲットが決める表現に写し，各ソースではリッジ回帰の写像で再現し，ラベルの情報は密度比のモデルで移す．ソースの正解ラベルが少ない場合の代理ラベルの版もある．理論は CCA とリッジの写像の「回転の一致」の結果と，ターゲットの条件付きの結果の分布の一致性で，誤差の率は抄録に無い．応用は腎細胞がんで，感情は無い（○ 抄録） — [arXiv 2608.01282](https://arxiv.org/abs/2608.01282)
- BOOMDA の重みは各モダリティの整列の損失の勾配から閉じた形で決まり，パレート停留の枠を超える保証は無い（◎ HTML 本文を要約器で読んだ） — [arXiv 2511.08152](https://arxiv.org/html/2511.08152)
- A3D2 は，ずれないモダリティとずれるモダリティを分ける設定を置いたが，上界は無い（◎ 本文 1〜3 節） — [ACL 2025](https://aclanthology.org/2025.acl-long.967.pdf)
- マルチモーダル VQA の DA で，モダリティごとの特徴抽出器にそれぞれ MMD の項を置く例がある（△ 検索の抜き書き） — [arXiv 1911.04058](https://arxiv.org/pdf/1911.04058)
- Dong ほか（arXiv 2501.18592，2025-01，v4 は 2025-09）のマルチモーダルの適応と汎化の総説は 5 つの領域（マルチモーダル DA，テスト時適応，DG，基盤模型の助けを借りる DA・DG，基盤模型の適応）を扱い，抄録は理論を未解決の課題に挙げていない（○ 抄録） — [arXiv 2501.18592](https://arxiv.org/abs/2501.18592)

### Inferences
- (a) モダリティごとのずれの項を持つ上界：見つからない．(b) 近いもの：Wang 2026（参照のモダリティ，ラベルのずれ），BOOMDA（重みの最適化），A3D2（ずれないモダリティを錨にする）．(c) 障害：Ben-David 2010 の上界は 1 つの仮説の族について書かれ，連結した特徴の上の 1 つの線形分類器の H-divergence はモダリティごとの divergence の和に分かれない．分けるには，モダリティのブロックごとに独立な分類器を足す（後期融合）か，ブロック間の独立を仮定する必要がある．(d) 感情認識：BOOMDA と A3D2 は IEMOCAP 系（人工のずれ）．
- MER-DA では音声が 7 群すべてにあるので，Wang 2026 の「参照のモダリティ」の条件を音声が満たす．同論文の「参照のモダリティでターゲットのラベルの分布を推定してから揃える」は，MER-DA の釣り合い直し（自分の予測を感情ごとに総量を揃えて弱いラベルにする，+0.9）と同じ向きの手続きで，比べる相手か理論の枠の候補になる．

### Gaps
- Wang 2026 の本文（仮定，率）は読んでいない．
- 探した検索語：multimodal domain adaptation theoretical bound modality-specific discrepancy／multimodal domain generalization theory generalization bound modality 2025 arXiv．場所：ウェブ検索，arXiv，ACL Anthology．

## 問い 6．学習時は複数のモダリティがあり，配備時は 1 つだけの設定の理論

### Takeaway
理論はある．Lu（arXiv 2309.12458）は，モダリティ間の写像をラベル無しのデータで学び，配備時は単モダリティの入力を写像に通して予測する 2 段の学習で，単モダリティの学習より上界が最大 O(√n) 良くなることを，モダリティ間に「つながり」と「異質さ」の両方がある場合に示した．LUPI，MFH，MKE の式 (14) もこの設定の理論である．ただし，どれも学習時と配備時の分布が同じで，コーパス間のずれを入れたものは DALUPI（問い 3）と Wang 2026（問い 5）だけだった．

### Cited Findings
- Lu "A Theory of Multimodal Learning"（arXiv 2309.12458，2023-09，改訂 2023-12．arXiv のページに掲載先は無い）：モダリティ間のつながり g をラベル無しのデータで学び，配備時は単モダリティの母集団で f̂∘ĝ を使う．学んだ単モダリティの予測器は消える汎化誤差を持ち，単モダリティの学習より上界が最大 O(√n) 良くなる．利点はつながりと異質さの両方があるときに生じる．差の大きさの定量化が無いこと，リプシッツの仮定を限界に挙げる（◎ 序論と限界の節） — [arXiv 2309.12458](https://arxiv.org/abs/2309.12458)
- MKE の式 (14)：単モダリティ α の生徒の誤り ≤ 4·Err(先生)/(c1 − 1) + 4μ（◎ 3.3 節） — [arXiv 2103.14431](https://arxiv.org/abs/2103.14431)
- 欠けたモダリティの深層学習の総説（arXiv 2409.07825）は，多くの研究が学習時は全モダリティがそろい，テスト時だけ欠けると仮定すると述べる（△ 検索の抜き書き） — [arXiv 2409.07825v3](https://arxiv.org/html/2409.07825v3)
- 対ごとの距離学習で，モダリティの部分集合に対応する関数の族の階層と，モダリティの数と細かさを含む汎化の上界を出した論文がある（仮定は射影について閉じていること）（△ 検索の抜き書き） — [arXiv 2605.01424](https://arxiv.org/html/2605.01424)

### Inferences
- (a) 定式化済み（ずれ無し）：Lu 2023 の 2 段の学習，LUPI（Lopez-Paz 2016 の一般化蒸留），MFH，MKE．ずれ込み：DALUPI，Wang 2026．(c) ずれ込みで「配備は 1 モダリティ」の理論が少ない障害は，先生（または写像 g）を学ぶ分布と生徒を使う分布の両方を動かす必要があることである．(d) 感情認識：MFH の RAVDESS と MKE の RAVDESS（コーパス内）．
- Lu 2023 の f̂∘ĝ は，MER-DA で「ターゲットのラベル無しの発話で音声 → 顔・言葉の写像 ĝ を学び，ソースの融合の分類器 f̂ に通す」手に対応する．E60 は f̂ の出力を弱いラベルにして音声の線形の生徒を学ぶので，ĝ を明示的に学ばない．2 つを比べると，写像を経由することの得と損を分けられる．
- 言葉が無い群（AFEW・RAVDESS・CREMA-D）と顔が無い群（RAVDESS）があるので，MER-DA は学習時にもモダリティが欠ける．総説が述べる「学習時は全モダリティがそろう」前提は MER-DA では成り立たない．

### Gaps
- Lu 2023 の査読付きの掲載先（NeurIPS 2023 と記憶するが未確認）．
- 探した検索語：theory multimodal training unimodal inference provable benefit cross-modal distillation generalization bound／missing modality theoretical analysis generalization bound incomplete multimodal learning provable．場所：ウェブ検索，arXiv．

## 問い 7．2024〜2026 年のマルチモーダル DA の理論の論文

### Takeaway
マルチモーダル DA の汎化の上界そのものを主題にした 2024〜2026 年の論文は見つからなかった．見つかったのは，ずれと別の要素を組み合わせた理論（ラベルのずれと欠けたモダリティの統計 Wang 2026，ずれ込みの蒸留 Li ICML 2026，共訓練 Pan 2025 のワークショップ論文，PID のグラフの分布外汎化 RIG 2025）と，理論の無い手法（BOOMDA，A3D2，MER-DG，MBCD）である．

### Cited Findings
- Wang, Dou, Liu, Cai（arXiv 2608.01282，2026-08）：ラベルのずれとブロック状に欠けたモダリティの DA，一致性の結果（○ 抄録） — [arXiv 2608.01282](https://arxiv.org/abs/2608.01282)
- Li, Han, Yang, Tong（ICML 2026）：線形の先生と生徒の，ずれ込みの蒸留の超過リスク（○ 抄録） — [PMLR 306](https://proceedings.mlr.press/v306/li26bd.html)
- Pan & Woodard（ICML 2025 Affinity Workshop）：マルチモーダル共訓練の上界．ずれの項は無い（◎ 定理 3.7） — [arXiv 2510.07509](https://arxiv.org/abs/2510.07509)
- Halder, Dissanayake, Dutta（NeurIPS 2025）：PID の冗長を最大化するグラフの分布外汎化（○ 抄録） — [NeurIPS 2025](https://neurips.cc/virtual/2025/133836)
- Xie, Xu, Sanguinetti（arXiv 2510.13182）と Tran ほか（arXiv 2606.10504）：別のモダリティの蒸留の情報量の基準と，対の無い蒸留の理論．どちらも抄録はずれに触れない（○ 抄録） — [arXiv 2510.13182](https://arxiv.org/abs/2510.13182)；[arXiv 2606.10504](https://arxiv.org/abs/2606.10504)
- BOOMDA（arXiv 2511.08152），A3D2（ACL 2025），MER-DG（arXiv 2605.01967），MBCD（arXiv 2511.20258）は転移の誤差の上界を持たない（◎／○ は問い 4 のとおり） — [BOOMDA](https://arxiv.org/html/2511.08152)；[A3D2](https://aclanthology.org/2025.acl-long.967.pdf)；[MER-DG](https://arxiv.org/abs/2605.01967)；[MBCD](https://arxiv.org/abs/2511.20258)

### Inferences
- 2024〜2026 年の理論は「ずれ」「モダリティ」「蒸留」の 3 つのうち 2 つまでを扱い，3 つを同時に扱うものは見つからない．MER-DA の E60（ソースで固定した融合の先生，ターゲットの単モダリティの生徒，少数ラベル）の理論はこの空いた位置にある．

### Gaps
- OpenReview の ICLR 2026 投稿は閲覧の確認画面で止まり探せなかった．中国語圏の論文誌，IEEE TPAMI・TAC の本文は探していない．
- 探した検索語：問い 3〜6 の検索語に加え，multimodal domain generalization theory generalization bound modality 2025 arXiv．

## 問い 8．MER-DA の設定で学習法にするなら何が要るか（候補ごとに 2〜3 行）

### Takeaway
MER-DA の設定（凍結特徴の連結の上の線形の頭，クラスあたり K 本，ラベル無しのターゲットの発話あり，音声は 7 群すべてにある）に最も近いのは，expansion と一貫性の正則化（MKE，Cai 2021），DALUPI の g∘f，In-N-Out の「補助情報を出力に使う」，Li 2026 の線形の蒸留の漸近解析の 4 つである．とくに一貫性の正則化は，E60 の上限「得 ≤ 先生 − 出発点」を理論の上で超えうる唯一の候補だった．

### Cited Findings
- 生徒が先生を超えるのは，一貫性の正則化と expansion があるとき（MKE 定理 1・式 (14)，Cai 2021） — [arXiv 2103.14431](https://arxiv.org/abs/2103.14431)；[PMLR 139](https://proceedings.mlr.press/v139/cai21b.html)
- 補助情報を入力に使うと分布外で悪化しうるが，出力に使うと改善する（線形回帰の理論） — [arXiv 2012.04550](https://arxiv.org/abs/2012.04550)
- g∘f の 2 段で，X の重なりなしにターゲットで一致する条件（W の共変量シフト，W の重なり，十分性） — [arXiv 2303.09350](https://arxiv.org/html/2303.09350)
- 2 視点の CCA で結んだ部分空間で，ターゲットのラベル無しに線形の予測器を移す（全領域で同じ線形の E[Y|X] を仮定） — [TTI-TR-2009-1](https://www.ttic.edu/technical_reports/ttic-tr-2009-1.pdf)

### Inferences
候補ごとの要るもの（(a)〜(d) は上の各問いの判定）：

1. **一貫性の正則化つきの E60（expansion．MKE・Cai 2021）**：ターゲットのラベル無しの発話の凍結特徴の空間に近傍（k 近傍のグラフなど）を定め，生徒の答えを近傍で揃える項を E60 の目的に足す．要るのは，近傍 N(V) の定義と，expansion の係数 c をターゲットで実測する手続き（MKE の式 (14) で生徒の上界が先生の誤りを下回るのは c1 > 5 のとき）と，L2 の錨との重みの選び方である．
2. **DALUPI の g∘f として E60 を書き直す**：W を融合の特徴，g をソースの融合の分類器とし，E60 の得の上限を「g のターゲットでの誤り」で表す定理にする．要るのは，W についての P(Y|W) の不変性と重なりがコーパスの組ごとにどれだけ破れるかの測り方（MER-DA の切片と向きのずれの分解が使える）と，ロジスティック回帰・UAR への書き直しである．
3. **In-N-Out の「補助情報を出力に」**：ターゲットのラベル無しの発話で，音声の特徴から顔・言葉の特徴を予測する線形の写像を学び（Lu 2023 の ĝ にもあたる），その予測を生徒の表現か事前の目標にしてから K 本と E60 の弱いラベルで学ぶ．要るのは，言葉・顔の無い群（AFEW・RAVDESS・CREMA-D，RAVDESS の顔）でどのソースの写像を使うかの決め方と，In-N-Out の線形回帰の理論を多クラスの分類に写すことである．
4. **Li 2026 の線形の蒸留の漸近解析**：E60 の弱いラベルの総重み m_eff の最適値を，先生と生徒の次元と標本数の比，ソースとターゲットのずれで表す式にする．要るのは，回帰から多クラスのロジスティック回帰への拡張と，先生と生徒の入力の次元が違う（別のモダリティ）場合の扱いで，まず本文で混合の重みの最適値が出ているかを確かめる．
5. **PID の冗長でどの先生が効くかを予測する**：ソースの各組で，音声と顔・言葉の感情についての冗長の割合をガウス PID か Liang 2024 の推定量で（主成分で数十次元に落としてから）測り，E60 の得と比べる．MFH の γ の実測版にあたる．要るのは，連続の高次元特徴の推定量と，ターゲットではラベルが K 本しかないので Liang 2024 の「ラベル無しの対と単モダリティのラベル」から出す下限をずれ込みで成り立たせることである．
6. **coupled subspaces（CCA）で移す**：ターゲットのラベル無しの発話で音声と顔（または言葉）の CCA を学び，共有部分空間に射影してからソースの線形分類器を移す．要るのは，仮定 1（全コーパスで同じ線形の E[Y|X]）の破れの大きさを実測すること（MER-DA の切片と向きのずれは破れを示唆する）と，2 視点がそろう群が限られる（言葉は 4 群，顔は 6 群）ことへの対処である．
7. **参照のモダリティでラベルの分布を推定してから揃える（Wang 2026）**：7 群すべてにある音声を参照にしてターゲットの感情の比率を推定し，ソースを重みづけてから融合の先生を学ぶ．要るのは，ラベルのずれの仮定（P(X|Y) がコーパス間で同じ）が MER-DA で成り立たない分の扱いで，E59 の釣り合い直し（+0.9）との違いを示すことである．
8. **モダリティのブロックごとの重みを上界から決める**：連結した特徴のブロック（音声・顔・言葉）ごとに Ben-David の divergence を測り，ずれの大きいブロックの L2 罰則を強める．要るのは，連結の上の 1 つの線形分類器の上界をブロックごとの項に分ける仮定（ブロック間の独立など）で，凸な目的なので競合の学習の動きの理論（Huang 2022）は使えない．

### Gaps
- 候補 1・3・4 は，MER-DA で理論を立てる前に，それぞれ expansion の係数，音声 → 顔・言葉の写像の当たり具合，Li 2026 の混合の重みの式があるかを確かめる必要がある．どれも本調査では確かめていない．
- 候補 1〜8 のうち，感情認識（コーパス間）で試された例はどれも見つからなかった（各問いの (d)）．
