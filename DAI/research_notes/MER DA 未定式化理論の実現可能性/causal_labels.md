# 因果と不変性に基づく DA の理論と，感情コーパスのラベルの付き方（意図・知覚）との対応

確かめた水準：◎ 本文を読んだ（該当の節を pdftotext で取り出して読んだ），○ 抄録を読んだ，または本文の HTML を WebFetch の要約で確かめた（その旨を書く），△ 書誌か検索結果のみ．「既出」は `docs/related-work.md` に既に記録がある文献で，そこでの水準に従う．調査日は 2026-10-28．

用語（このノートでの意味）：
- 共変量シフト：入力の分布 P(X) だけが変わり，入力からラベルを当てる関係 P(Y|X) は変わらない．
- ラベルシフト（target shift，prior shift）：ラベルの割合 P(Y) だけが変わり，ラベルごとの入力の分布 P(X|Y) は変わらない．
- 条件つきシフト（conditional shift）：P(X|Y) が変わる．P(Y) も変わる場合を Zhang ほか 2013 は一般化ターゲットシフト（GeTarS）と呼ぶ．
- causal 学習：原因 X から結果 Y を当てる．anticausal 学習：結果 X から原因 Y を当てる．
- 意図ラベル：演技者に指示した感情（演技者の意図が声と顔を生むので Y→X）．知覚ラベル：評定者が発話を見聞きして付けた感情（発話が評定を生むので X→Y）．

## 問い 1：因果の向きと不変性の理論は，DA の仮定の選び方と，ラベル無しのデータが役立つ向きについて何を予言するか

### Takeaway
Schölkopf ほか 2012 は，causal 学習では共変量シフトが自然な仮定で，ラベル無しの入力は P(Y|X) の推定に役立たないと予言し，anticausal 学習ではラベルシフト（P(X|Y) が不変）が自然な仮定で，ラベル無しの入力が役立ちうると予言した．後続の Zhang ほか 2013・Gong ほか 2016 は Y→X を前提に，P(Y) と P(X|Y) の両方が変わる場合を「クラスごとの位置と尺度の変換」に制限して，ラベル無しのターゲットだけで補正する学習法として定式化した．

### Cited Findings
- **Schölkopf, Janzing, Peters, Sgouritsa, Zhang, Mooij 2012（ICML．"On causal and anticausal learning"）◎**（2〜5 節を読んだ）— [arXiv 1206.6471](https://arxiv.org/abs/1206.6471)
  - causal（X→Y）で入力の分布だけが変わるとき，「原因の分布と仕組みは独立」という原理から P′(Y|X) = P(Y|X) と結論し，これを共変量シフトと呼ぶ（2.1.1 節）．ただし有限標本では学習した関数が P(X) の濃い所に合うので，予測の規則を直さなくてよいという意味ではないと注意する．
  - causal でラベル無しの入力を足す半教師あり学習は「P(X) contains no information about P(Y|X)」なので「SSL is pointless」と書く（2.1.2 節）．
  - anticausal（Y→X）の例として手書き数字を挙げ，「a person intends to write the digit 7 … this intention causes a motor pattern producing an image」と書く（3 節冒頭）．このとき P(X|Y) が仕組みで P(Y) と独立であり，P(Y|X) は P(Y) の変化に左右されるので，P(X|Y) を先に模型化してベイズの式で P(Y|X) を作る方がよいと述べる．
  - anticausal で出力の分布だけが変わる場合（prior probability shift）は P′(X|Y) = P(X|Y) から P′(Y|X) を計算できる（3.2.1 節）．入力の分布が変わった場合は，P(Y) と P(X|Y) のどちらが変わったかを先に決める必要があり（"Localizing distribution change"），P(Y) が変わったなら P(X|Y) が単射であるという仮定の下で P′(Y) を推定し，P(X|Y) が変わったなら加法雑音模型の下で逆畳み込みにより P′(X|Y) を推定する（3.1.1 節，4 節）．
  - anticausal ではラベル無しの入力を足すと P(X) の推定がよくなり，P(X) と P(Y|X) は独立でないので役立ちうる（3.1.2 節）．
  - 実証：Chapelle ほかの半教師あり学習の比較（8 データ，ラベル 100 件）で，causal の 1 データでは半教師あり学習が精度を有意に上げず，anticausal または交絡のあるデータの多くでは上げた．UCI の 26 データ（自己訓練）でも causal の 3 データで誤り率が有意に下がらなかった．著者はデータが少なく結論は出しにくいと書く（5 節）．ここで「Anticausal/Confounded」は「少なくとも 1 つの特徴がクラスの結果であるか，クラスと観測されない共通原因を持つ」データと定義される．
- **Zhang, Schölkopf, Muandet, Wang 2013（ICML．"Domain adaptation under target and conditional shift"）◎**（1〜4 節を読んだ）— [PMLR v28](https://proceedings.mlr.press/v28/zhang13d.pdf)
  - 全編で Y が X の原因であると仮定する．X がドメインに Y を通してだけ依存するならターゲットシフト（TarS），Y とドメインの両方が X の原因で Y がドメインと独立なら条件つきシフト（ConS），P(Y) と P(X|Y) の両方がドメインに依存するなら GeTarS，という因果の図による対応を示す．
  - ConS は一般には推定できない（ill-posed）ので，P(X|Y) がクラスごとの位置と尺度の変換 x_new = w(y)⊙x + b(y) で変わる場合（LS-ConS，LS-GeTarS）に限る．定理 2 は，クラスごとの分布が位置と尺度の変換の後も線形独立であれば，ラベル無しのターゲットの P(X) からクラスごとの変換が識別できることを示す．著者は次元が高いほどこの条件は破れにくいと書く．
  - 推定はカーネル平均埋め込みで行い，ターゲットのラベルは使わない．ConS 単独は「less realistic」として実験しない．
- **Gong, Zhang, Liu, Tao, Glymour, Schölkopf 2016（ICML．"Domain adaptation with conditional transferable components"．PMLR 48:2839–2848）○** — [PMLR](https://proceedings.mlr.press/v48/gong16.html)
  - Y が X の原因である系で P(X|Y) と P(Y) の両方が変わると仮定し，位置と尺度の変換の後で P(T(X)|Y) が不変になる成分 T(X) を取り出し，P(Y) の変化も推定する．
- **von Kügelgen, Mey, Loog, Schölkopf 2020（UAI．"Semi-supervised learning, causality and the conditional cluster assumption"）○** — [arXiv 1905.12081](https://arxiv.org/abs/1905.12081)
  - Schölkopf 2012 の「原因から当てるときは半教師あり学習は役立たない」を，原因の特徴と結果の特徴が混ざる場合に広げ，ラベル無しのデータは「原因の特徴を与えたときの結果の特徴の条件つき分布」を通して役立つと主張する．
- **Janzing & Schölkopf 2015（JMLR 16(58):1923–1948．"Semi-supervised interpolation in an anticausal learning scenario"）△** — [JMLR](https://jmlr.org/papers/v16/janzing15a.html)
  - 検索結果の抄録によれば，決定論的な関係 Y = f(X) で，ラベル無しのデータが補間を助けるのは直交の条件が破れる場合だけで，それは主に anticausal の向きで起きると論じる．
- **Jin, von Kügelgen, Ni, Vaidhya, Kaushal, Sachan, Schölkopf 2021（EMNLP．"Causal direction of data collection matters"）◎**（1〜6 節を読んだ）— [arXiv 2110.03618](https://arxiv.org/abs/2110.03618)
  - データの集め方で向きが決まると述べ，意図の分類を例に，書き手が先に意図を示して書いたなら intent→text で anticausal，他人が後から注釈したなら text→annotated intent で causal と分ける（3 節）．
  - 予言：半教師あり学習は anticausal で役立ち causal では役立たない．共変量シフトの DA は causal の方が易しい．
  - 先行研究のメタ分析：半教師あり学習の改善は causal 55 課題で平均 +0.04%（標準偏差 4.23），anticausal 50 課題で +1.70%（2.05），Welch の t 検定で p = 0.011．ターゲットに少数のラベルがある DA の改善は causal 22 課題で 5.18（6.57），anticausal 11 課題で 1.26（1.79），p = 0.023（表 5・表 6）．
- **Garg, Wu, Balakrishnan, Lipton 2020（NeurIPS．"A unified view of label shift estimation"）○** — [arXiv 2003.07554](https://arxiv.org/abs/2003.07554)
  - ラベルの割合の推定で，混同行列を使う BBSE は粗い較正を使った最尤法（MLLS，EM）とほぼ同じで，BBSE の効率の悪さは較正が粗いことによる情報の損失だと示す．MLLS の誤差を較正の誤りと推定の誤差に分ける．MLLS の一致性には分類器の較正と混同行列の可逆性が要る．
- **既出（`docs/related-work.md` 2.10・2.11 節）**：Lipton ほか 2018 BBSE（ラベルシフトを anticausal の学習と位置づけ，予測の分布の 2 標本検定で検出する），Alexandari ほか 2020（較正つきの EM が強い），Tachet des Combes ほか 2020（表現 Z で P(Z|Y) が不変という一般化ラベルシフト GLS と IWDAN など），Chen & Bühlmann 2021（anticausal の線形模型で平均だけがずれれば，ラベル無しで平均をそろえる DIP のリスクがずれの大きさに依らない），Ha & Chen 2025/2026（anticausal の線形模型でずれを CA・SC・AW の 3 型に分け，型ごとに最適な作り方と，少数のターゲット検証で選ぶ MASFT を示す），Garg ほか 2023 RLSbench．
- **Zhang, Gong, Stojanov, Huang, Liu, Glymour 2020（"Domain adaptation as a problem of inference on graphical models"．会議名は抄録の頁に無い）○** — [arXiv 2002.03278](https://arxiv.org/abs/2002.03278)
  - 複数のソースから，同時分布のどの部分（モジュール）が変わりどの部分が変わらないかを表すグラフを学び，ラベル無しの DA をそのグラフ上のベイズ推論として解く．どう変わるかを前もって決めずにデータから学ぶ．
- **Liu, Zhang, Gong, Gong, Huang, van den Hengel, Zhang, Shi（"Latent covariate shift"．v4 は 2025-04）○** — [arXiv 2208.14161](https://arxiv.org/abs/2208.14161)
  - 従来の共変量シフト・条件つきシフトの枠は感情の割合などラベルの分布がドメイン間で同じと仮定すると批判し，潜在の内容変数を条件にしたラベルの分布を学ぶことで，ラベルの分布が変わってもよい複数ソースの DA を作る．内容変数はブロック単位で識別できると主張する．掲載先（TMLR 2025 と検索結果は言う）は抄録の頁で確かめられなかった（△）．
- **Ng, Li, Li, Zheng, Chen, Zhang 2026（"A general representation-based approach to multi-source domain adaptation"．2026-04-26 投稿．コメント欄は ICML 2025 と書き，投稿日と合わない）○** — [arXiv 2604.23790](https://arxiv.org/abs/2604.23790)
  - ラベルのマルコフブランケットの表現を，ラベルの親・子・配偶者に分けて学べば識別できると主張する．ラベル付きの複数ソースとラベル無しのターゲットの設定．

### Inferences
- Schölkopf 2012 の予言を感情に当てはめると，意図ラベル（Y_int→X）は手書き数字と同じ anticausal で，ラベルシフトとクラスごとの補正（LS-GeTarS，CTC）が自然な仮定になり，ラベル無しのターゲットが P(Y|X) の推定に情報を持ちうる．知覚ラベル（X→Y_perc）は causal で，共変量シフトが自然な仮定になり，ラベル無しのターゲットは P(Y|X) の推定に情報を持たない．
- ただし知覚ラベルでも，話者の内的な状態が声・顔を生み，評定者は声・顔・前後の文脈を見て判定するので，モデルが見る入力（凍結エンコーダの発話平均）が評定者の見た入力より狭いと，入力とラベルが観測されない共通原因を持つ．Schölkopf 2012 はこの場合を「Anticausal/Confounded」に入れており，ラベル無しのデータが役立たないという予言は弱まる．
- Jin ほか 2021 のメタ分析は，少数のターゲットラベルによる DA の改善が causal の課題で大きいことを示した．感情に当てはめると，知覚ラベルのターゲットの方が少数ラベルの DA の得が大きいという予言になるが，感情認識で確かめた研究は見つからなかった（問い 2）．
- MER-DA の既存の結果（ラベル無しの中心化で平均のずれが消える．EM による割合の補正は効かない．ラベル無しの情報では境目の向きを直せない）は，(1) 平均のずれは位置の変換の一種で LS-ConS の特別な場合として扱える，(2) EM はラベルシフト（P(X|Y) 不変）を仮定するので，P(X|Y) が変わる組では前提が崩れる，(3) 向きのずれは位置と尺度の変換では表せない，と読める．(2) と (3) は仮説で，組ごとに検定して確かめる必要がある（問い 4）．

### Gaps
- Schölkopf 2012 の付録（ウェブ頁 [1] のデータの分類）は読んでいない．
- Gong 2016 がターゲットのラベルを使うかは抄録の頁に書かれていなかった（本文は未読）．
- Janzing & Schölkopf 2015 と Kügelgen 2020 の本文，Garg 2020 の定理の条件は読んでいない．
- Zhang 2020 の掲載先（NeurIPS 2020 と思われるが未確認），Liu の LCS の掲載先は確かめられなかった．

## 問い 2：感情のラベルの付き方（意図か知覚か）による因果の向きの違いを，DA の仮定の選択や学習法に使った研究はあるか

### Takeaway
音声・顔・マルチモーダルの感情認識で，意図ラベルと知覚ラベルの因果の向きの違いを DA の仮定の選択や学習法に使った研究は，探した範囲では見つからなかった（未定式化と判定する）．近い研究は，同じ問いを自然言語処理で論じた Jin ほか 2021，1 つのコーパスの中で意図ラベルと知覚ラベルを比べた Giordano ほか 2026（Emozionalmente），コーパス間の比較でラベルの出所を他の要因と分けられなかった VoxEmo の 3 本である．

### Cited Findings
- **Giordano, Giacomelli, Rinaldi, Graziosi 2026（"Label authority in speech emotion recognition: intended and perceived ground truth in Emozionalmente"．arXiv v1 2026-10-19．掲載先の記載なし）○**（本文 HTML を WebFetch の要約で確かめた）— [arXiv 2610.26446](https://arxiv.org/html/2610.26446)
  - イタリア語のコーパス Emozionalmente（431 話者，6,902 発話，7 感情）で，829 人の聞き手が 1 発話あたりちょうど 5 票を付けた（有効な判定 34,510）．意図ラベル，聞き手の最多票，票の分布の 3 つを「label authority」として同じ音声で比べる．
  - Krippendorff の α は全体で 0.476．発話ごとの最多票が意図ラベルと一致するのは 79.1%，評定者 1 人ごとの一致は 66.4%．満場一致の 1,792 発話はすべて意図と一致した．
  - 自分のラベルで測った UAR の最良値は，意図で学んで意図で測ると 0.825 ± 0.009，最多票で学んで最多票で測ると 0.746 ± 0.006，票の分布で学ぶと 0.749 ± 0.004．最多票で共通に測ると，意図で学んだ模型 0.739，最多票で学んだ模型 0.731，分布で学んだ模型 0.736．票の分布で学ぶと UAR はほぼ変わらず，票の分布との KL が 1.223 から 0.510 に下がった．
  - 因果・DA・ラベルシフトには触れない．Grossi ほかが Emozionalmente の意図ラベルで学んだ模型の外部コーパスへの移りが限られたと報告したことを引く．CREMA-D と RAVDESS は引いていない．
- **VoxEmo（arXiv 2603.08936．既出 ◎）**：ラベルの出所の扱いは WebFetch の要約で確かめた（○）— [arXiv 2603.08936](https://arxiv.org/abs/2603.08936)
  - 「annotations reflect the emotion intended/expressed by the speaker or the emotion perceived by the listeners」と出所を区別し，2.4 節では CREMA-D を意図ラベル，IEMOCAP・MELD・MSP-Podcast を知覚ラベルとする．表 1 は CREMA-D と IEMOCAP を「Expr+Perc」と書き，2.4 節と食い違う．
  - コーパス間の設定では「label inventories, recording conditions, and label source … vary simultaneously」と書き，ラベルの出所の効果を分けて測っていない．
- **Keesing, Watson, Witbrock 2020（ALTA．"Convolutional and recurrent neural networks for spoken emotion recognition"．PDF から取り出した文字で著者の並び順が崩れたので，名前だけを確かめた）◎**（該当段落を読んだ）— [ACL Anthology](https://aclanthology.org/2020.alta-1.13.pdf)
  - CREMA-D のような演技コーパスで意図ラベルと評定の多数決のどちらを正解にするかという問いを立て，MSP-IMPROV と IEMOCAP は評定のラベル，CREMA-D は意図ラベルを使ったと書く．両者の比較はしていない．
- **Jin ほか 2021（問い 1）◎**：意図の分類で「書き手が先に示した意図」は anticausal，「他人の注釈」は causal と分けた．感情は扱わない．— [arXiv 2110.03618](https://arxiv.org/abs/2110.03618)
- **感情の伝達のレンズ模型（Scherer 2003，Speech Communication．"Vocal communication of emotion: a review of research paradigms"）△**：話し手の状態が声の手がかりを生み（encoding），聞き手が手がかりから状態を推し量る（decoding）という Brunswik のレンズ模型を，声の感情研究の土台として提案した．— [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167639302000845)；レンズ模型を音響の手がかりで検証した経路模型の研究 △ — [PLOS ONE 2015](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0136675)；Busso と Narayanan の講義資料は意図した感情から知覚された感情への流れとして説明する △ — [講義資料 2009](https://people.ict.usc.edu/~gratch/CSCI534/Old-Lectures/Lecture8-Busso-2009.pdf)
- **「因果」を名乗る感情認識の研究は，交絡（データセットや文脈が感情と入力の両方に影響する）を取り除く研究で，ラベルの付き方の向きは扱わない △**：画像と文脈の感情認識で文脈を交絡因子としてバックドア調整する研究（[arXiv 2303.11921](https://arxiv.org/abs/2303.11921)，[arXiv 2407.04963](https://arxiv.org/abs/2407.04963)），データセット由来の文脈を交絡因子とする IERN（[Monash の頁](https://research.monash.edu/en/publications/towards-unbiased-visual-emotion-recognition-via-causal-interventi/)），コーパス間の音声感情認識の「causal feature disentanglement」（[ACM DL](https://dl.acm.org/doi/10.1145/3686490.3686527)．本文は 403 で取得できず）．
- **音声感情認識でラベルの割合のずれを測った研究 △**：話者ごとの適応の研究は，話者のラベル分布と学習集合全体の KL を測り，覚醒度で成績との相関が約 −0.50（特徴のずれは −0.71）だったと報告する．補正はしていない．— [arXiv 2309.02418](https://arxiv.org/abs/2309.02418)
- **MSP-IMPROV（Busso ほか．TAC）△**：目標の感情（演技の意図）と評定者の知覚ラベルの両方を持ち，原著に意図と知覚の混同行列がある．— [ResearchGate の頁](https://www.researchgate.net/publication/289571004_MSP-IMPROV_An_Acted_Corpus_of_Dyadic_Interactions_to_Study_Emotion_Perception)

### Inferences
- 「意図と知覚の向きの違いを DA の学習法に使う」ことは，一般の機械学習では部品（ラベルシフトの補正，LS-GeTarS，CTC，GLS，DIP/CIP，MASFT）が定式化済みで，感情認識ではどの部品も「ラベルの出所で仮定を選ぶ」形では使われていない．MER-DA が新しさを置くなら，部品ではなく「出所に応じて仮定を選ぶ規則と，それを 7 群の組で確かめること」に置くことになる．
- コーパス間の比較では，ラベルの出所と「演技か自然な会話か」がほぼ重なる（CREMA-D・RAVDESS は演技で意図ラベル，MELD・MOSEI は自然で知覚ラベル）．IEMOCAP は演技（台本と即興の場面で目標の感情を指示）だが公開ラベルは評定の多数決なので，出所と演技の有無を分けられる唯一の群である．VoxEmo が分けられなかった交絡を，MER-DA は IEMOCAP の 2 群で部分的に分けられる．
- Emozionalmente の結果（意図と最多票の一致 79.1%，学ぶラベルを変えても共通の基準での UAR は 0.731〜0.739）は，1 つのコーパスの中では学ぶラベルの出所が UAR を大きく変えないことを示す．コーパス間で出所が効くかは別の問いで，答えは見つからなかった．

### Gaps
- **探した検索語と場所**（一般のウェブ検索．arXiv・ACL Anthology・PMLR・PMC・IEEE Xplore・ACM DL の頁が結果に出た．Google Scholar と IEEE Xplore・ACM DL の全文検索は使えなかった）："speech emotion recognition anticausal causal learning label shift perceived intended emotion"，"\"anticausal\" emotion recognition domain generalization"，"intended versus perceived emotion labels training speech emotion recognition cross-corpus generalization CREMA-D MSP-IMPROV"，"causal inference backdoor adjustment cross-corpus speech emotion recognition confounder 2024 2025"，"\"label shift\" OR \"prior shift\" speech emotion recognition cross-corpus class prior adaptation"，"expressed versus perceived emotion speech emotion recognition encoder decoder lens model Brunswik label source model training"，"2025 causal domain adaptation multiple sources anticausal \"conditional shift\" \"label shift\" few target labels identifiability arXiv"．どれも感情認識で向きを使った学習法を返さなかった．
- 顔の表情認識だけの文献（FER）で意図と知覚を分けた DA は，専用の検索をしていない．
- Hassan, Damper, Niranjan 2013（IEEE TASLP．"On acoustic emotion recognition: compensating for covariate shift"）は音声感情認識を共変量シフトとして重みづけで補正した先例と思われるが，検索結果の参考文献欄に現れただけで，出所の URL を確保できなかった．
- Grossi ほか（Emozionalmente の意図ラベルからの外部コーパスへの移り）は原典を確かめていない．
- コーパス間の感情認識で「意図ラベルで学ぶか知覚ラベルで学ぶか」を比べた研究は見つからなかった．

## 問い 3：不変性の理論（ICP，不変な部分集合，IRM とその批判）を複数コーパスの感情認識に使った研究と結果

### Takeaway
ICP・不変な部分集合・IRM を複数コーパスの感情認識に使った研究は，探した範囲では見つからなかった．感情認識のコーパス間の研究は話者やコーパスを敵対学習で消す方法が中心である．Rosenfeld ほか 2021 は線形の場合，環境の数 E が環境ごとに変わる特徴の次元 d_e 以下だと IRM の解が変わる特徴に頼ることを示しており，MER-DA の 6 ソースと高次元の凍結特徴では条件を満たしにくい．

### Cited Findings
- **Peters, Bühlmann, Meinshausen 2016（JRSS B 78(5):947–1012．"Causal inference using invariant prediction"）○** — [arXiv 1501.01332](https://arxiv.org/abs/1501.01332)
  - 複数の環境（介入）で予測の精度が変わらない模型をすべて集めると，因果の模型が高い確率でその集合に入るので，因果の関係に信頼区間が付けられる．構造方程式模型で原因の特徴の集合が識別できる条件を示す．
- **Rojas-Carulla, Schölkopf, Turner, Peters 2018（JMLR 19．"Invariant models for causal transfer learning"）○** — [arXiv 1507.05333](https://arxiv.org/abs/1507.05333)
  - 共変量シフトを緩め，特徴のある部分集合を与えたときの目的変数の条件つき分布がすべての課題で不変だと仮定する．敵対的な設定ではこの部分集合を使うのが最適だと証明し，課題が多様なら合併より平均でも良い例を示す．ターゲットの例があるときに全特徴を使う方法も出すが保証は無いと書く．
- **Arjovsky, Bottou, Gulrajani, Lopez-Paz 2019（IRM）△** — [arXiv 1907.02893](https://arxiv.org/abs/1907.02893)
- **Rosenfeld, Ravikumar, Risteski 2021（ICLR．"The risks of invariant risk minimization"）◎**（模型の節，定理 3.1・5.1 を読んだ）— [arXiv 2010.05761](https://arxiv.org/abs/2010.05761)
  - 模型：ラベル y が不変な潜在特徴 z_c ∼ N(y·μ_c, σ_c² I) と，環境ごとに平均 μ_e が変わる潜在特徴 z_e を生み，観測 x = f(z_c, z_e)．著者は y と z の因果の向きは区別できないと注記する．
  - 線形の場合（定理 5.1）：E > d_e なら IRM の解は最適な不変予測器になる．E ≤ d_e で環境の平均が線形独立なら，変わる特徴に頼る線形の予測器が IRM の制約を満たし，訓練の環境では不変予測器より損失が小さい．
  - 非線形の場合：テストのデータが訓練の分布に十分近くないと IRM は大きく失敗しうる．結論として IRM とその後継は経験損失最小化（ERM）を本質的に改善しないと書く．
- **IRM と ERM の標本数の比較（"Empirical or invariant risk minimization? A sample complexity perspective"）△** — [arXiv 2010.16412](https://arxiv.org/abs/2010.16412)
- **感情認識での「不変」の学習は，話者やコーパスを敵対学習で消す方法が中心 △**：話者を勾配反転で消す（IEMOCAP と CMU-MOSEI で評価．[USC の PDF](https://sail.usc.edu/publications/files/Haoqi_icassp_updated.pdf)），コーパスと話者の識別器を足す SIDANN（[ICMI 2020](https://dl.acm.org/doi/10.1145/3382507.3418813)），コーパスに依らない特徴を対照学習で分ける Emo-DNA（[arXiv 2308.02190](https://arxiv.org/abs/2308.02190)），ドメインごとに変わる表現とドメインに依らない分類器を組み合わせる研究（英語と日本語の間．[IEEE Xplore 10832334](https://ieeexplore.ieee.org/document/10832334/)），脳波の感情認識で DG の手法を比べた基準（[arXiv 2204.09016](https://arxiv.org/abs/2204.09016)）．どれも IRM・ICP そのものではない．

### Inferences
- MER-DA はターゲットを 1 群とすると環境（ソース）が 6 つで，凍結特徴の次元はそれよりずっと大きい．コーパスごとに変わる方向の数 d_e が 6 を超えれば，Rosenfeld の定理 5.1 から線形の IRM はコーパスごとに変わる方向に頼る解を選ぶ．IRM を使うなら，先にソースの分類器の出力（4 次元）のような低い次元へ写す必要がある．
- ICP は Y の原因となる特徴の集合を探す方法である．意図ラベルでは声・顔の特徴はすべて Y の結果の側にあるので，原因の集合は空になりうる．知覚ラベルでは特徴が評定の原因になるので ICP の前提に近いが，評定者の基準がコーパスで違えば（IEMOCAP は苛立ちを別のラベルにした，MOSEI の中立は消去法）P(Y|X) 自体が変わり，不変な集合は見つからない．
- Rosenfeld の模型は，ラベルが平均 y·μ をずらすガウスの模型で，MER-DA の「中心化で平均のずれが消える」の解析と同じ形をしている．MER-DA の線形の頭の解析に，定理 5.1 の「環境の数と変わる方向の数」の比較を持ち込める．

### Gaps
- 探した検索語："invariant risk minimization cross-corpus speech emotion recognition"，"invariant risk minimization emotion recognition domain generalization corpora speakers IRM"，"\"invariant risk minimization\" facial expression OR \"speech emotion\" recognition environments"．IRM を感情認識に使った論文は返らなかった．ACII・Interspeech・ICASSP の予稿集の全文検索はしていない．
- Arjovsky 2019 と IRM と ERM の標本数の比較の論文は本文を読んでいない．
- Rosenfeld の非線形の定理（6.1）の条件は読んでいない．

## 問い 4：どの DA の仮定が成り立つかを，ターゲットの少数のラベルとラベル無しのデータで検定する方法はあるか

### Takeaway
ラベル無しのターゲットだけで検定できるのは「ラベルの割合が変わったか」（Lipton ほか 2018 の BBSE の検定）で，ラベルシフトの仮定そのもの（P(X|Y) が不変）はラベル無しでは確かめられない．ターゲットに少数のラベルがあれば，「P(Y|X) が同じか」を調べる 2 標本の条件つき分布の検定（Hu & Lei 2024）や，ドメインを条件つき独立の検定で調べる方法が使えるが，K 本での検出力を示した研究は見つからなかった．

### Cited Findings
- **Lipton, Wang, Smola 2018（既出 ◎）**：ラベルシフトの仮定の下で，分類器の予測の分布 p(ŷ) と q(ŷ) が等しいことと q(y) = p(y) が同値なので，予測の 2 標本検定でラベルの割合の変化を検出できる．— [arXiv 1802.03916](https://arxiv.org/abs/1802.03916)
- **Kouw & Loog（"A review of domain adaptation without target labels"）△**：ドメインの関係についての仮定（共変量シフト，事前確率のずれ）はターゲットのラベル無しでは確かめられないと述べ，ラベルシフトの検定をソースの事前分布と予測したターゲットの事前分布の 2 標本検定として紹介する．— [arXiv 1901.05335](https://arxiv.org/abs/1901.05335)
- **Schölkopf 2012（◎）**：P(Y) と P(X|Y) のどちらが変わったかを決める手順（"Localizing distribution change"）を加法雑音模型の下で大まかに描く．例えば P(E) のフーリエ変換の零点が原因側と雑音側のどちらに由来するかで判定する（4 節）．— [arXiv 1206.6471](https://arxiv.org/abs/1206.6471)
- **Zhang 2013（◎）**：LS-ConS の定理 2 は，条件を満たせばラベル無しのターゲットの P(X) からクラスごとの変換が一意に決まることを示す．— [PMLR v28](https://proceedings.mlr.press/v28/zhang13d.pdf)
- **Hu & Lei 2024（JASA 119(546):1136–1154．"A two-sample conditional distribution test using conformal prediction and weighted rank sum"）△**：2 つの集団で応答の共変量への条件つき分布が同じかを，共形予測の重みつき順位和で検定する．転移学習を動機に挙げる．両方の集団にラベルが要る．— [arXiv 2010.07147](https://arxiv.org/abs/2010.07147)，[IDEAS](https://ideas.repec.org/a/taf/jnlasa/v119y2024i546p1136-1154.html)；続編の Chen & Lei（偏りを除いた U 統計量）△ — [arXiv 2402.00164](https://arxiv.org/abs/2402.00164v1)
- **Zhang, Singh, Ghassemi, Joshi 2023（ICML．"Why did the model fail?"）△**：2 つの環境の成績の差を，データ生成の各モジュールの分布のずれに Shapley 値で割り振る．値は重要度の重みで計算する．— [arXiv 2210.10769](https://arxiv.org/abs/2210.10769)
- **Choi 2026（ICML 2026 の仮説検定の workshop．"Anytime-valid confirmation of label-shift corrections"）○**：前もって決めたラベルシフトの補正が，順に届くラベル付きのターゲットと合うかを，補正後とソースの予測の尤度比を掛けたマルチンゲールで逐次検定する．ガウス過程回帰で閉じた式を持つ．ラベルシフトと条件つきシフトを見分けるとは書いていない．— [arXiv 2606.14028](https://arxiv.org/abs/2606.14028)
- **特徴ごとのずれの場所を条件つき分布の検定で特定する研究 △** — [arXiv 2107.06929](https://arxiv.org/abs/2107.06929)
- **Zhang ほか 2020（○）**：複数ソースから変わるモジュールと変わらないモジュールを学ぶので，「どの仮定が成り立つか」をデータから選ぶ形になっている．— [arXiv 2002.03278](https://arxiv.org/abs/2002.03278)

### Inferences
- MER-DA では，ソースのラベルとターゲットの K 本で「ドメイン D と特徴 X が感情 Y を条件に独立か」（X ⊥ D | Y．成り立てばラベルシフト）を検定できる．高次元の X のままでは検出力が無いので，ソースの分類器の出力（4 次元）やクラスごとの平均の差の方向へ写してから，クラスごとに 2 標本検定を行う形が現実的である．
- この検定が棄却する組では，EM による割合の補正の前提が崩れていたことになる．EM が効かなかった組と棄却された組が重なるかを見れば，「EM が効かない理由はラベルシフトの仮定の破れ」という仮説を組ごとに確かめられる．
- 「P(Y|X) が同じか」（共変量シフト）は Hu & Lei の検定の形で K 本から調べられるが，4 クラスで K=5〜50 の検出力は未知である．Ha & Chen の MASFT のように，検定で仮定を決める代わりに少数のターゲット検証で候補の作り方を選ぶ方法と比べる必要がある．

### Gaps
- 探した検索語："testing label shift assumption hypothesis test \"label shift\" violated detect conditional shift unlabeled target"，"diagnosing distribution shift decompose covariate shift versus concept shift Y|X test method"，"two-sample conditional distribution test conformal weighted rank sum Hu Lei JASA"，"\"Why did the model fail\" attributing model performance changes …"．
- ラベル無しのターゲットだけで条件つきシフトを検出する方法は見つからなかった．
- 少数のラベル（クラスあたり 5〜50）でラベルシフトの仮定を検定したときの検出力を示した研究は見つからなかった．
- Zhang ほか 2023 がターゲットのラベルを必要とするかは確かめていない．

## 問い 5：各コーパスのラベルの出所（原著論文での事実確認）

### Takeaway
MER-DA の 7 群のうち，CREMA-D と RAVDESS はファイル名の意図ラベル（演技者に指示した感情）を使い，IEMOCAP（評定者 3 人の多数決），MELD（3 人の多数決），MOSEI（3 人の群衆評定の強さ）は知覚ラベルを使う．AFEW は映画の字幕のキーワードで候補を選んだ後に人が付けたラベルで，評定者の人数は原著で確かめられなかった．CREMA-D は意図と群衆評定の両方を持ち，評定の多数が意図と一致する割合は音声だけで 41%，音声と映像で 72% である．

### Cited Findings
- **MER-DA でのラベルの決め方**（`data/README.md` の「ラベルの決め方」）：IEMOCAP は注釈ファイルの代表ラベル（多数決）で hap と exc を Happy にまとめ fru は入れない．MOSEI は 6 感情の強さ（0〜3）が 0.5 以上の感情が 1 つならその感情，0 個なら neutral（消去法）．MELD は公式 CSV の感情ラベル．AFEW は公式のフォルダ名．RAVDESS はファイル名の 3 番目の数字（calm は入れない）．CREMA-D はファイル名の 3 番目の項目．発話一覧 `data/metadata/crema-d_utterance_manifest.csv` の列に群衆評定は無く，`iemocap_utterance_manifest.csv` の列にも評定者ごとのラベルは無い（valence・arousal・dominance はある）．
- **CREMA-D（Cao, Cooper, Keutmann, Gur, Nenkova, Verma 2014，IEEE TAC）○**（PMC の本文を WebFetch の要約で 2 回抜き出し，数値は一致した）— [PMC4313618](https://pmc.ncbi.nlm.nih.gov/articles/PMC4313618/)
  - 演技者 91 人（男 48，女 43，20〜74 歳），12 文，6 感情（怒り・嫌悪・恐れ・喜び・中立・悲しみ），7,442 本．1 文目だけ強さ 3 段階．
  - 意図の作り方：演劇の演出家 2 人が場面を示して演技者を導き，「The actors acted out a given sentence in a specific target emotion until the director approved the performance」．
  - 知覚ラベル：群衆の評定者 2,443 人．1 項目あたり目標 10 評定（平均約 9.8，範囲 4〜12，95% 超の本が 8 評定以上）．音声だけ・映像だけ・音声と映像の 3 通りで合計 22,326 項目．
  - 評定者 1 人ずつが意図と一致した割合：音声だけ 40.9%，映像だけ 58.2%，音声と映像 63.6%．
  - 評定の集団のラベル（最も多く選ばれた感情）が意図と一致した本の割合：音声だけ 41%（不一致 46%，あいまい 13%），映像だけ 64%（25%，11%），音声と映像 72%（21%，8%）（表 7）．多数の定義には二項検定で 95% で棄却できる「binomial majority」も使う．
  - Krippendorff の α：感情ラベル 0.42（範囲 0.25〜0.57），強さ 0.47，一致が 80% を超える本に限ると 0.79．
- **RAVDESS（Livingstone & Russo 2018，PLOS ONE 13(5):e0196391）○**（本文を WebFetch の要約で確かめた．前半 10 万字のみ）— [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196391)
  - 演技者 24 人（男女 12 人ずつ，21〜33 歳），7,356 ファイル（話し 4,320，歌 3,036），話しは 8 感情，強さは普通と強い（中立を除く）．
  - 意図の作り方：「actors were told that they could use whatever techniques they were trained in to induce the desired state」．
  - 検証：全 7,356 本をそれぞれ 10 回，感情の妥当性・強さ・本物らしさで評定した（妥当性の評定者 247 人，再評定 72 人）．正答率は話しと歌を合わせて音声と映像 80%，映像だけ 75%，音声だけ 60%．Fleiss の κ は話しの強い表出で 0.62，普通で 0.53．再評定の一致は Cohen の κ の平均 0.73．
  - ファイル名の感情は意図ラベルで，評定で当たらなかった本も全部コーパスに残した（評定は S1 Table にある）．
- **IEMOCAP（Busso ほか 2008，Language Resources and Evaluation 42(4)．頁は未確認）◎**（SAIL の稿 2007-10-15 版の 3 節と 5.2 節を読んだ）— [SAIL の PDF](https://sail.usc.edu/publications/files/BussoLRE2008.pdf)
  - 演技者 10 人の 2 人組で，台本（感情の内容がはっきりした 3 本の劇）と，喜び・怒り・悲しみ・苛立ち・中立を引き出すよう作った場面での即興の 2 通りで録った．
  - 評定者は USC の学生 6 人で，各発話を異なる 3 人が評定した．評定者は動画を見ながら順に評定したので，声・映像・それまでの会話の流れが判定に使えた．9 カテゴリ（基本 6 感情と苛立ち・興奮・中立）から複数を選んでよい．
  - 最多票のカテゴリが 1 つに決まるとき多数決でラベルを付け，その条件で合意したのは 74.6% の発話（台本 66.9%，即興 83.1%）．
  - Fleiss の κ は全体 0.27，合意した発話で 0.40．喜びと興奮をまとめ，恐れ・嫌悪・驚きを「その他」にすると 0.35 と 0.48．次元（valence・activation・dominance）の評定は 2 人（表 III の説明）．
- **MELD（Poria ほか 2019，ACL）◎**（2〜3 節を読んだ）— [arXiv 1810.02508](https://arxiv.org/abs/1810.02508)
  - 元の EmotionLines は Amazon Mechanical Turk の 5 人が文字起こしだけを見て付け，Fleiss の κ は 0.34．
  - MELD は英語に堪能な大学院生 3 人が動画も見て付け直し，多数決で最終ラベルを決めた．3 人とも違った 89 発話（11 対話）は対話ごと落とした．Fleiss の κ は 0.43．ラベルは Ekman の 6 感情と中立．
- **CMU-MOSEI（Zadeh ほか 2018，ACL）◎**（3.2 節を読んだ）— [ACL Anthology P18-1208](https://aclanthology.org/P18-1208.pdf)
  - 23,453 本の文に，極性を [−3, 3]，Ekman の 6 感情それぞれの「有無」を [0, 3]（0：証拠なし，1：弱い，2：ある，3：強い）で付けた．
  - 評定は Amazon Mechanical Turk の群衆 3 人（承認率 98% 超の master worker）で，「to capture the raw perception of the crowd」ため訓練は 5 分の動画だけにした．一致度は補足資料にあると書く．
- **AFEW（Dhall, Goecke, Lucey, Gedeon 2012，IEEE MultiMedia 19(3):34–41）△**（書誌と抄録の検索結果のみ）— [ANU の頁](https://openresearch-repository.anu.edu.au/items/a63944bd-dfad-4020-85db-039c5d72a3e8)
  - 同じ著者の SFEW の稿 ◎ は「AFEW was collected using a semi-automatic closed caption and subtitle for deaf parsing」と書く．— [SFEW の PDF](https://cvhci.iar.kit.edu/download/SFEW.pdf)
  - 総説 △ は，37 本の映画から 957 本を，字幕の「laughs」「smiles」「scared」などのキーワードで探し，人の観察者が 6 基本感情（と中立）と演技者の情報を付けたと書く．— [arXiv 1802.08784](https://arxiv.org/pdf/1802.08784)；957 本と 37 本の映画はキャンベラ大学のデータ記録 △ とも一致する — [researchdata.edu.au](https://researchdata.edu.au/acted-facial-expressions-wild/2734)

### Inferences
- 出所で分けると，意図ラベル（anticausal）は CREMA-D と RAVDESS，知覚ラベル（causal に近い）は IEMOCAP の 2 群・MELD・MOSEI，混ざったもの（映画の脚本と演出が意図を決め，字幕と人の判定がラベルを決める）は AFEW になる．
- CREMA-D の音声だけでは評定の多数が意図と一致する本が 41% しかない．MER-DA の音声の特徴で CREMA-D の意図ラベルを当てる課題は，評定者が音声から当てる課題より「正解が音声に表れていない」本を多く含む．同じ CREMA-D でも，音声だけの組と音声と映像の組でラベルの雑音の量が違う．
- IEMOCAP は即興の方が合意の割合が高い（83.1% 対 66.9%）．MER-DA の IEMOCAP-impro と IEMOCAP-script の違いには，ラベルの一致度の違いも含まれる．
- 評定者の数は IEMOCAP・MELD・MOSEI が 3 人，CREMA-D が約 10 人，RAVDESS の検証が 10 評定で，一致度は IEMOCAP の κ 0.27，MELD の κ 0.43，CREMA-D の α 0.42，RAVDESS の κ 0.53〜0.62 である．指標が違う（Fleiss の κ と Krippendorff の α）ので数値を直接は比べられない．

### Gaps
- AFEW の評定者の人数，一致の確かめ方，音声を聞いたかは原著（IEEE MultiMedia 2012）で確かめられなかった．AFEW の技術報告（ANU，2011）も見つからなかった．
- MOSEI の一致度（補足資料）と，配布された強さが 3 人の平均かどうかは確かめていない．
- RAVDESS の話しだけのモダリティ別の正答率（表 2）は読んでいない．
- CREMA-D の群衆評定の票が配布物のどのファイルにあるか（GitHub の集計ファイル）は確かめていない．IEMOCAP の評定者ごとの注釈が配布物にあるかも確かめていない．
- IEMOCAP の LRE の巻・号・頁は SAIL の稿からは確かめられなかった．

## 問い 6：MER-DA の設定で学習法にするなら何が要るか（候補ごとに 2〜3 行）

### Takeaway
理論から作れる候補は 5 つある．新しい学習を回さずに既存の結果だけで確かめられるのは候補 A（ラベル無しの手法の得をターゲットのラベルの出所で分けて，向きの予言と合うかを見る）だけで，他の候補は CREMA-D の群衆評定の票の取得か，K 本での検定の検出力の見積もりが要る．

### Cited Findings
- 向きの予言（causal ではラベル無しが P(Y|X) に情報を持たない，anticausal では持ちうる）は Schölkopf 2012 ◎ — [arXiv 1206.6471](https://arxiv.org/abs/1206.6471)，少数ラベルの DA は causal の方が得が大きいというメタ分析は Jin 2021 ◎ — [arXiv 2110.03618](https://arxiv.org/abs/2110.03618)．
- クラスごとの位置と尺度の補正（LS-GeTarS）とその識別可能性は Zhang 2013 ◎ — [PMLR v28](https://proceedings.mlr.press/v28/zhang13d.pdf)，Gong 2016 ○ — [PMLR](https://proceedings.mlr.press/v48/gong16.html)．
- CREMA-D の意図と評定の一致は音声だけ 41%，映像だけ 64%，音声と映像 72% — [PMC4313618](https://pmc.ncbi.nlm.nih.gov/articles/PMC4313618/)．
- 線形の IRM は E ≤ d_e で変わる特徴に頼る（Rosenfeld 2021 ◎）— [arXiv 2010.05761](https://arxiv.org/abs/2010.05761)．
- 票の分布で学ぶと UAR はほぼ変わらず確率の当たりがよくなる（Emozionalmente，KL 1.223 → 0.510）○ — [arXiv 2610.26446](https://arxiv.org/html/2610.26446)．

### Inferences
- **候補 A：ラベル無しの手法の得を，ターゲットのラベルの出所で層別して向きの予言を確かめる．**予言は「意図ラベルのターゲット（CREMA-D・RAVDESS）ではラベル無しの手法（中心化，EM，自己訓練など）の得が大きく，知覚ラベルのターゲット（IEMOCAP の 2 群・MELD・MOSEI）では小さい」．要るのは既存の結果の表をターゲットの出所で分け直すことだけで，出所と「演技か自然か」の交絡は，演技だが知覚ラベルの IEMOCAP の 2 群で切り分ける．評定者が見た入力（声・顔・文脈）がモデルの入力より広いので，知覚ラベル側の予言は弱まる点を前提に書く．
- **候補 B：CREMA-D の二重のラベルから「意図 → 知覚」の 4×4 の遷移行列を推定し，意図ラベルのソースを知覚ラベルのターゲットに合わせて損失を補正する．**要るのは CREMA-D の群衆評定の票（MER-DA の発話一覧には無い）と，モダリティごとの行列（音声だけでは一致が 41% なので行列は対角から大きく外れる）．遷移行列による損失の補正はノイズラベル学習の既存の手法で，出典は本調査では確かめていない．行列が CREMA-D 以外のコーパスでも同じかは確かめようが無いのが弱点である．
- **候補 C：意図ラベルのソースにはクラスごとの位置と尺度の補正（LS-GeTarS，CTC）を使い，全体の中心化から一歩進める．**要るのは K 本で 4 クラス×次元のずれを推定するための低次元化か，クラスごとのずれを全体のずれへ縮める推定で，Zhang 2013 の定理 2 の条件（変換後もクラスの分布が線形独立）が凍結特徴で成り立つかの確認である．「ラベル無しでは向きを直せない」という既存の結果は，向きのずれが位置と尺度の変換では表せないことを示唆するので，得は位置と尺度の部分に限られる見込みである．
- **候補 D：組ごとにラベルシフトの仮定を検定し，成り立つ組では EM，棄却された組では条件つきの補正（候補 C）を使う．**検定はソースのラベルとターゲットの K 本で X ⊥ D | Y を，ソースの分類器の出力（4 次元）上のクラスごとの 2 標本検定で行い，多数の組を Holm などで補正する．要るのは K=5〜50 での検出力の見積もりで，比較相手は検定をせず少数のターゲット検証で選ぶ Ha & Chen の MASFT である．
- **候補 E：IRM・ICP を 6 ソースの環境で使う．**Rosenfeld の定理 5.1 から，コーパスごとに変わる方向の数が 6 を超えれば線形の IRM は変わる方向に頼るので，4 次元程度への射影が前提になる．ICP は意図ラベルでは原因の特徴が無く空集合を返しうる．理論の上で効く見込みが最も低い．
- 候補とは別に，評定者の人数と一致度でソースのラベルを軟らかくする方法も考えられるが，発話ごとの票が手元にあるのは CREMA-D と IEMOCAP の配布物だけの可能性が高く（未確認），Emozionalmente の結果から UAR の改善は期待しにくい．

### Gaps
- CREMA-D の評定の票と IEMOCAP の評定者ごとの注釈が配布物にあるかは確かめていない（候補 B の前提）．
- 遷移行列による損失の補正の原典（ノイズラベル学習）は本調査で確かめていない．
- 候補 A の判定に使う「これまでのラベル無しの手法の得のターゲット別の表」が MER-DA の結果のどこにあるかは，本調査の範囲外として確かめていない．
