# 引用の監査（round 1）

## 1. 集計

| 項目 | 数 | 内訳 |
|---|---|---|
| 引用箇所（`\cite` 命令） | 95 | 本文 85，補足 10．主張の単位に分けると 81（付録 A） |
| 引用キー | 81 | 本文 78，補足 11（補足だけは 3） |
| 検証済み（キー単位） | 79 | 本文または該当節で確かめた 65，抄録または二次資料の水準で確かめた 14（7.2） |
| 未検証（キー単位） | 2 | kimeldorf1971results，ryumina2022facial（7.1） |
| 不一致（Major） | 4 | M-1 Raudys の実効次元の帰属，M-2 比較相手の記述と「原典の値」，M-3 交差モダリティ蒸留の区別の文と xMUDA，M-4 IEMOCAP の台本の記述 |
| 要修正（Minor） | 21 | m-1〜m-21（3.2） |
| 引用漏れ | 13 | 5 節 |

- 書誌（著者・年・誌名・題名・巻号頁・DOI）の誤りは見つからなかった．欄の不足が 1 件（scholkopf2001generalized の LNAI の巻 2111）ある．プレプリントを引いている文献のうち，出版版が見つかったものは無い（du2021improving には同じグループの別の査読論文がある．m-19）．
- 設定の台帳（2 節）の感情認識の方法の文献 13 本では，本文が帰している結果が原典の範囲を超えた例は無かった．一部不一致は 2 本（ahn2025multitask の「顔を含む」，pendyala2025leveraging の「音声映像の教師」）で，どちらも結論を変えないので Minor とした．コーパスの文献では IEMOCAP の台本の記述が原典と合わない（M-4）．
- 新しさの文（4 節）は，組み合わせとしてはどれも保たれる．ただし交差モダリティの教師の区別の文は最も近い先例（xMUDA Fusion とその後継）を誤って分類しており（M-3），顔を含むコーパス間のマルチモーダル研究 2 本（MERBench，Ryumina et al. 2025）が引かれていない（m-3）．


## 2. 設定の台帳

感情認識の文献について，原典で確かめた設定と，本文がその文献に帰している内容を照らした．「確かめた水準」の列の「本文」は全文を読んだこと，「抄録」は抄録だけを読んだことを指す．

### 2.1 感情認識の方法の文献

| 引用キー | 文献が実際に使った設定（コーパス／ラベル／評価／ターゲットのラベル数／モダリティ） | 本文が帰している設定・結果 | 判定 | 確かめた水準 |
|---|---|---|---|---|
| schuller2010crosscorpus | 「six standard databases」でのコーパス間評価，音響特徴だけ，正規化を数種比較．抄録："1.8 k individual evaluations in total indicate the crucial performance inferiority of inter to intracorpus testing."．コーパス名・ラベル・評価指標は抄録に無い | I1：コーパスをまたぐと精度が落ちる．R1：大きな低下を示しコーパスの正規化を動機づけた | 一致 | 抄録（全文は取得できず） |
| gideon2021addog | IEMOCAP，MSP-Improv，PRIORI Emotion．ラベルは valence の 3 区分（柔らかいラベル）．UAR（偶然 0.33）を被験者ごとに計算．ターゲットは「randomly in half」に分割（話者で分けていない）．ターゲットのラベルは 0, 200, 400, 800, 1600, 3200 発話（クラスあたりではなく総数）．音声だけ（40 次元 MFB）．ハイパーパラメータは "keep them constant for all experiments"（理由は "due to the lack of labelled data in the target domain"） | I1：精度の低下．R3：ターゲットのラベル量を増やして評価した敵対的ドメイン汎化．R9：「いつ汎化しいつ特化するか」を未解決として残した（原文 "it is important to formulate exactly when to generalize and when to specialize"）．S6：ターゲットのラベルが少ない場合にハイパーパラメータを固定した | 一致 | 本文（arXiv:1903.12094） |
| feng2023fewshot | ソースは IEMOCAP の台本，RAVDESS，CREMA-D，eNTERFACE'05，ターゲットは IEMOCAP の自発（即興）．怒り・喜び・悲しみの 3 クラス．64 次元の韻律・スペクトル特徴．ラベル付きは「1 labelled sample per speaker and per emotion」から 10 まで，残りをテストに使う（ラベルの話者とテストの話者が重なる）．UAR．音声だけ | I1：精度の低下．R2：自発音声のシャム網による少数ショット学習．I3：少数のラベル付き発話で音声モデルを適応．R10：ソースの性質とソースの組み合わせの効果を未解決として残した（原文 "it would be beneficial to explore the extent to which the combination of multiple source datasets can benefit few-shot learning"） | 一致 | 本文（arXiv:2109.02915） |
| chen2023prefinetuning | wav2vec 2.0 を IEMOCAP，MSP-IMPROV，MSP-PODCAST，Mandarin Affective Speech の組み合わせで事前微調整し，ESD の話者ごとに 2 値（ある感情か否か）分類を $k\in\{2,\dots,64\}$ 例（総数）で微調整．指標は Macro F1．音声だけ | R4：ソースのコーパスでの事前微調整 | 一致 | 本文（arXiv:2302.12921） |
| xu2024crosstask | IEMOCAP（セッションで分割する同一コーパス内），VAM→IEMOCAP，MELD→IEMOCAP．4 または 5 クラス（frustrated を含む場合あり）．wav2vec 2.0 の特徴を平均した 768 次元（エンコーダは学習しない）にロジスティック回帰．初期 20 ラベル＋最大 200 件を能動的に選ぶ．BCA（＝UAR）を，ラベルを付けた標本も含むターゲット全体で評価．音声だけ | R5：凍結した音声特徴での能動学習 | 一致 | 本文（arXiv:2412.01171） |
| he2025m4ser | IEMOCAP↔MELD の 4 クラス（neutral, happy, angry, sad）．音声＋ASR の文字（HuBERT＋BERT）．ターゲットの検証用 30% から「5, 10, 20, and 60 labeled instances per class」を取り，残りの検証データでモデルを選ぶ．指標は IE→ME で ACC と W-F1，ME→IE で WA と UA | R6：音声＋文字のモデルをクラスあたり 5〜60 ラベルで適応．I3 | 一致 | 本文（arXiv:2509.18706v1．出版版の該当表は未確認） |
| zaidi2024enhancing | IEMOCAP（英），EMODB（独），EMOVO（伊），URDU の 4 クラス．音声＋文字．UA．0, 5, 10, 15 ショット（"the number of labelled samples from the target language that are used to fine-tune the models"．クラスあたりかは書かれていない） | R7：言語をまたぎコーパスあたり少数のラベルで適応．I3 | 一致 | プレプリント本文（arXiv:2306.13804v3．題名が出版版と違う．OJCS 版は未確認） |
| ahn2025multitask | IEMOCAP，MSP-IMPROV，EMO-DB．音声と文字のマルチタスク変換器を別々に作り判定段で融合．ターゲットのラベル無しデータを補助課題に使う．抄録："Multimodal approaches for cross-corpus scenario is substantially limited as well." | I5：「顔を含む」複数モダリティのクロスコーパス手法は少ない．R8：クロスコーパスのマルチモーダル手法は少ない | 一部不一致（Minor）：原典が述べるのはマルチモーダル一般の少なさで，顔については何も言わず，自身も音声＋文字である．3 節の項目 m-3 | 抄録（全文は購読が必要） |
| dong2026test | IEMOCAP，RAVDESS の 4 クラス．wav2vec2-base．Acc と macro F1．課題は話者への個人化，演技→自然（IEMOCAP の台本→即興），RAVDESS↔IEMOCAP のコーパス間．ラベル無しのテスト時適応．抄録："entropy minimization and pseudo-labeling generally fail" | R11：エントロピー最小化と擬似ラベルはほとんどの設定で失敗する | 一致 | 本文（arXiv:2601.16240） |
| albanie2018emotion | 教師は VGGFace2 で事前学習し FERplus で微調整した顔の表情認識器（第三のコーパス）．ラベル無しの VoxCeleb で音声の生徒に蒸留．評価は RML と eNTERFACE で，生徒の 8 次元出力から目的のラベルへの affine 写像を 10 分割交差検証で学習し正答率を報告 | R32：顔の教師が音声の生徒を学習させる．R36：教師は第三のコーパスで学習 | 一致 | 本文（arXiv:1808.05561） |
| srinivasan2022representation | MSP-Podcast v1.6 と IEMOCAP（5 分割）．activation・valence・dominance の回帰（CCC）．教師は同じ訓練集合で学習した音声＋文字の融合モデル，生徒は音声だけ | R33：音声＋文字の融合教師が音声の生徒を学習させる．R36：教師はターゲットのコーパスで学習 | 一致 | 本文（arXiv:2112.00158） |
| pendyala2025leveraging | RAVDESS（音声部分），CREMA-D．教師は 2 つの単一モダリティのモデル：音声教師（HuBERT を同じデータセットのラベル付き音声で LoRA 微調整）と映像教師（FERV39k で学習した S2D）．ラベル無しの音声映像は MSP-Face．生徒は軽量な音声モデル | R34：「音声映像または映像の教師」が音声の生徒を学習させる．R36：教師はターゲットか第三のコーパスで学習 | 一部不一致（Minor）：融合した音声映像の教師ではなく，音声の教師と映像の教師を別々に使う．R36 の区別は成り立つ．3 節の項目 m-4 | 本文（arXiv:2507.00055） |
| hajavi2024audio | 話者認識（VoxCeleb）と感情認識（RAVDESS は 1 被験者抜きの交差検証，IEMOCAP）．映像の教師を同じデータセットで先に学習して凍結し，音声の生徒を学習 | R34：映像の教師が音声の生徒を学習させる．R36：教師はターゲットのコーパスで学習 | 一致 | 本文（arXiv:2302.02845） |

この表の文献では，本文が帰している結果が原典の範囲を超えた例は無い．ただし，どの文献も本稿と同じ設定（クラスあたり $K$ 本・話者を分けた評価・UAR・3 モダリティ）ではない．特に gideon2021addog と feng2023fewshot はターゲットの分割で話者を分けておらず，gideon2021addog と chen2023prefinetuning のラベル数はクラスあたりではない．本文はこれらの文献の数値を比較に使っていないので，現状の書き方で誤りは生じない．数値を並べて比べる記述を将来足す場合は，この違いを書く必要がある．


### 2.2 コーパスと表情認識モデルの文献

| 引用キー | 原典の仕様（話者／ラベル／書き起こし／言語） | 本文が帰している内容 | 判定 | 確かめた水準 |
|---|---|---|---|---|
| busso2008iemocap | 10 人（"Five female and five male actors"），5 セッション．台本は "Three scripts were selected"，各台本は "a female and a male role"．ラベルに frustration，excited を含む | 3.1 節："all ten speakers read the same lines in the scripted sessions"，happy と excited を合併，frustrated を除外 | 不一致（Major，M-4）：同じ台詞を読むのは同じ役の 5 人ずつ．合併と除外は本稿の選択として正しく書かれている | 本文（SAIL の版） |
| zadeh2018mosei | 6 感情を "annotated on a [0,3] Likert scale ... by 3 crowdsourced judges"．話者は "The definition of a identity is proxy to the number of channels"．"All videos have manual transcription." | 3.1 節：6 つの強度の 1 つだけが 0.5 以上なら感情，どれも無ければ neutral．話者の注記は無い | 一致（閾値 0.5 と評点のまとめ方は本稿の規則であり，原典には無い．m-17） | 本文 |
| poria2019meld | "Joy, Sadness, Fear, Anger, Surprise, and Disgust" と Neutral．話者 260/47/100（train/dev/test）．音声・映像・文字 | 4 ラベルの 1 つを happy と呼ぶ | 一致（Joy を happy と読み替えたことは書かれていない．m-17） | 本文 |
| dhall2012afew | 映画から 1426 本，被写者 330 人，6 基本表情と neutral．台詞の書き起こしの項目は無い．言語の記述は無い | 3.1 節：書き起こしは無い．8 節：英語のコーパス | 書き起こしは一致．英語は原典で確かめられない（m-17） | 著者の草稿 |
| livingstone2018ravdess | 24 人，"Two neutral statements"，話し言葉の 8 感情に calm を含む．"actors needed to have English as their first language"．映像を含む（"face-and-voice, face-only, and voice-only formats"） | 3.1 節：固定の文を読む，calm を除外，顔の特徴は「抽出していない」 | 一致（映像が無いとは書いておらず正確） | 本文 |
| cao2014cremad | "91 actors, 48 male and 43 female"，"12 sentences"，6 感情 | 3.1 節：固定の文を読む | 一致 | 本文（PMC 版） |
| ryumina2022facial | 論文本文は未読．著者の GitHub では VGGFace2 の ResNet-50 を AffectNet で微調整し，512 次元の 'features' 層を出す | 3.2 節と補足 S1：表情認識用の ResNet-50，VGGFace2 で学習し AffectNet で微調整，512 次元 | 未検証（7 節） | 書誌と著者のリポジトリ |

## 3. 不一致・要修正の一覧

### 3.1 不一致（Major）

#### M-1 Raudys の実効次元を標本線形判別に帰している（2 節第 2 段落，4.2 節）
- 該当箇所：2 節第 2 段落の第 1 文（R13），4.2 節の式 (3) の直後の文（P2）．
- 本文の主張："The asymptotic expansion of the misclassification rate of the sample linear discriminant contains terms of order $1/N$ that do not depend on the dimension and terms proportional to it [okamoto], and Raudys expressed the second through an effective dimension [raudys]"．
- 出典の実際の記述：Raudys（1998）の抄録 "Small learning-set properties of the Euclidean distance, the Parzen window, the minimum empirical error and the nonlinear single layer perceptron classifiers depend on an "intrinsic dimensionality" of the data, however the Fisher linear discriminant function is sensitive to all dimensions."．2 節 "For GCCM model, however, the generalization error depends on the true dimensionality p, and not on p*"．実効次元 $p^*$ を使う式 (1) はユークリッド距離分類器の式である．
- 問題：本文は「標本線形判別の次元比例の項を Raudys が実効次元で表した」と読める．Raudys は Fisher の判別についてその逆を述べている．$q$ は本稿の抄録にも出る中心の量であり，Raudys を読んだ査読者はすぐ気づく．
- 求める対応：例 "Raudys showed that for the Euclidean distance classifier and the single-layer perceptron this term depends on an effective dimension rather than on $p$, whereas for the Fisher discriminant it depends on $p$" と書き，本稿の罰則つきロジスティック回帰の $q$ をどちらに近いものとして読むのかを 1 文足す．4.2 節の式 (3) の出典の文も同じ内容に揃える．

#### M-2 比較相手の記述と「原典の値」（6.1 節，6.3 節）
- 該当箇所：6.1 節 第 1 段落と第 3 段落，6.3 節 第 1 段落．
- 本文の主張："self-training on the student's own confident predictions in the form of Lee [lee2013pseudolabel]"，"the *fixed* version uses the values of the original papers"，6.3 節 "With the hyperparameters of the original papers, entropy minimization, co-regularization, and prior adjustment change the UAR ... and co-regularization, and entropy minimization for the audio student, reduce it to chance level"．
- 出典の実際の記述：
  - Lee（2013）は確信度の閾値で選ばない．§2.4 "We just pick up the class which has maximum predicted probability for each unlabeled sample."．ラベル無しの損失の重み $\alpha(t)$ はエポックに応じた段階関数（式 (16)）で，値は "with αf = 3, T1 = 100, T2 = 600 without pre-training, T1 = 200, T2 = 800 with DAE"．閾値に当たる操作は本文に無い（監査者が PDF 全体を検索して確かめた）．
  - Grandvalet & Bengio（2004）は固定値を示さない．§4.1 "All hyper-parameters (weight-decay for all logistic regression models plus the λ parameter (6) for minimum entropy) are tuned by ten-fold cross-validation."
  - Sindhwani et al.（2005）も固定値を示さない．"We avoided the model selection issue due to the small size of the labeled set and chose best parameters over a small range of values."
  - Lopez-Paz et al.（2016）は人工データで "with λ = T = 1"，MNIST では $T$ と $\lambda$ を振っており，単一の既定値は無い．
  - Hinton et al.（2015）の温度は実験ごとに違う（MNIST で "a temperature of 20"，音声で "[1, 2, 5, 10]" から選択）．
  - Blum & Mitchell（1998）の値 "p = 1, n = 3, k = 30 and u = 75" は正例・負例の 2 クラスの手続きの値である．
  - 「教師の確信度の閾値による硬い擬似ラベル」と「開始点の出力の再均衡」には引用が無く，原典の値がそもそも定まらない．
- なぜ Major か：6.3 節は「原典のハイパーパラメータ」でエントロピー最小化と共正則化が偶然の水準まで落ちると報告している．原典が値を固定していない手法について，この結果を原典の設定に帰すことはできない．
- 求める対応：各比較相手の fixed 版で使った値と，その値の出どころ（原典の実験の値・著者の既定値・本稿の決め方）を補足の表に示す．原典が値を与えていない手法は「原典の値」ではなく「本稿が決めた既定値」と書く．Lee の形を名乗るなら「ラベル無しの全件に argmax のラベルを付け，重みを段階的に上げる」と書き，"confident" を外す（閾値を使ったなら Lee の形とは書かない）．Blum & Mitchell の 2 クラスの手続きを 4 クラスにどう移したかを書く．閾値つきの擬似ラベルに典拠があるならそれを引く．

#### M-3 交差モダリティ蒸留の先行研究の区別が xMUDA に当てはまらない（2 節第 5〜6 段落）
- 該当箇所：2 節第 5 段落の最後の文（R36），第 6 段落の最後の文（新しさの文）．
- 本文の主張："In these studies the teacher is trained on the target corpus or a third corpus, or the student uses all modalities at inference; the teachers of our basic method are trained on the source corpora alone ..."．第 6 段落 "Specific to our combination are a teacher trained on sources alone that sees modalities the student lacks, ..."．
- 出典の実際の記述：xMUDA（Jaritz et al., CVPR 2020）の教師に当たる側（他方の流れの主ヘッド，または xMUDA Fusion の融合ヘッド）は，ラベルをソースからだけ受け取る："the segmentation loss Lseg on source and the cross-modal loss LxM on source and target"（式 4）．4.4 節の拡張 "In xMUDA Fusion, we add two uni-modal outputs P2D→fuse and P3D→fuse that are used to mimic the fusion output Pfuse."．表 1 は "We report the result for each network stream (2D and 3D) as well as the ensembling result" と単一モダリティの流れの結果も報告する．前提は "we here assume all modalities are available at train and test time on both source and target domains"（2 節）．後継の BFtD（Wu et al., ACM MM 2023，pp. 490–498，doi 10.1145/3581783.3612013）と FtD++（Wu et al., IEEE TCSVT 35(9):9030–9045, 2025，doi 10.1109/TCSVT.2025.3545971）も，ソースだけにラベルがある設定で融合の出力と単一モダリティの出力を一致させる．BFtD は抄録と FtD++ による説明で，FtD++ は arXiv:2410.19446v1 の該当節で確かめた．
- 問題：xMUDA の教師は「ターゲットのコーパス」でも「第三のコーパス」でも学習しておらず，ラベルはソースだけから来る．xMUDA Fusion と BFtD は「他モダリティを見る（融合の）出力が単一モダリティのヘッドを教える」ことを，ソースだけのラベルで既にしている．本稿の新しさの主張に最も近いこの先例を，本文の区別の文は誤って分類している．
- 求める対応：区別の文を xMUDA について正確に書き直す．例 "xMUDA and its successors train the cross-modal or fused teacher jointly with the student on labeled source and unlabeled target data and assume all modalities at test time"．BFtD と FtD++ を引く．そのうえで本稿との差を次の形で書く：(i) 教師はソースだけで学習して固定し，ターゲットのデータを教師の学習に使わない，(ii) 弱いラベルで分類器そのもの（別の模倣ヘッドではない）を学習する（xMUDA は "uncouple the mimicry objective from the main segmentation objective"，3.1 節），(iii) 配備する生徒は自分のモダリティだけを使う，(iv) 重みと切片の別々の錨と $K$ ごとの選択．R35 の "fused outputs train unimodal heads [jaritz]" は xMUDA Fusion（4.4 節）と特定する（m-10）．

#### M-4 IEMOCAP の台本の記述が原典と違う（3.1 節）
- 該当箇所：3.1 節 第 1 段落の第 2 文．
- 本文の主張："all ten speakers read the same lines in the scripted sessions"．
- 出典の実際の記述（Busso et al. 2008，SAIL にある版の本文）："Three scripts were selected"，"these plays were selected so that they each consisted of a female and a male role"，"Since the selected scripts have a female and a male role, an actor and an actress were recorded in each of the five sessions"．同じ台詞を読むのは同じ役（同じ性別）の 5 人ずつであり，10 人全員ではない．
- 設定の台帳の原則に従い Major とする．群を分ける判断そのもの（文字の分類器が感情ではなく台本を覚える）は，訂正後の事実でも成り立つので，直すのは 1 文で済む．
- 求める対応：例 "the five sessions perform the same three scripts, so each scripted line is read by five speakers" と直し，この文にも busso2008iemocap を付ける．即興と台本を別のドメインとして扱った先例（feng2023fewshot，dong2026test，Li et al., Interspeech 2021）を引く（5 節の項目 2）．

### 3.2 要修正（Minor）

各項目は「該当箇所 / 本文の主張 / 出典の実際の記述 / 求める対応」の順に書く．

- **m-1 序論第 2 段落（I2：bendavid2010theory，blitzer2007learning，desilva2023value）**
  - 本文："Theory of learning from several domains shows that pooling a source helps when target labels are scarce and hurts when they are plentiful, and it bounds the target error by the source error, a divergence, and the size of the target sample"．
  - 出典：Ben-David et al.（2010）Sec. 6 は上界を最小にする $\alpha^*$ から "if there are enough target data (specifically, if m_T ≥ D² = d/A²) then no source data are needed, and in fact using any source data will yield suboptimal performance." と述べる．Blitzer et al.（2007）Sec. 5 "This is reflected in our bound as a phase transition in the value of the optimal α"．上界には両ドメインで最良の結合仮説の誤差 $\lambda$ の項もある．De Silva et al. は "the upper bound proposed by Ben-David et al. (2010) ... is significantly vacuous and does not predict the non-monotonic trends we have identified" と書き，自身の結果を OOD の量についての非単調性として示し，"as the number of target samples increases, the non-monotonicity is less pronounced or even completely absent" と述べる．
  - 対応：「示す」を「上界を最小にする重みが…ことを示唆する」に直し，上界の項に $\lambda$ を加える．De Silva は「1 次元の判別で，少ない OOD は得になり多い OOD は損になりうることを解析で示した」と別に書く．
- **m-2 2 節第 2 段落と 3.2 節（R16，S5：kumar2022lpft）**
  - 本文：2 節 "linear probing beats fine-tuning under distribution shift"．3.2 節 "with a few labels per emotion fine-tuning can distort the pretrained features [kumar2022lpft]"．
  - 出典：抄録 "fine-tuning can achieve worse accuracy than linear probing out-of-distribution (OOD) when the pretrained features are good and the distribution shift is large"．Sec. 4.1 "linear-probing does better on 8 out of 10 OOD datasets ... except CIFAR-10.1 and ImageNetV2"，同 "Our training datasets vary in size from 20K examples to over a million examples, so linear probing does not appear to perform better than fine-tuning simply because of a small training set."．少数例については理論の注記 Sec. 3.2.2 "Span of Training data (S). Theorem 3.3 lower bounds the error outside the span of the training data. If the training dataset is very small, then even the support of the ID distribution Pid may not be spanned by the training data, and the ID error can be large. Indeed, even in the ID setting Kornblith et al. (2019) show that linear probing can do better than fine-tuning if we have very few training examples, but fine-tuning does better on all 11 of their datasets once we have more than just 30 examples per class."
  - 判定：2 節の文は条件が落ちている．3.2 節の文は，原典の実験ではなく Sec. 3.2.2 の理論の注記（訓練データの張る空間の外では特徴が変わらない）で支えられる．原典の実験は小標本を原因から外しているので，引用の根拠が注記であることが読者に分からない．補助は当初「不支持に近い」と報告したが，監査者が Sec. 3.2.2 を読み直して一部支持とした．
  - 対応：2 節は "can" と 2 つの条件を戻す．3.2 節は「訓練点の張る空間の外では微調整が特徴を変えないので，ラベルが少ないと分布内でも誤差が大きくなりうる（Kumar et al. の Sec. 3.2.2）」のように根拠の箇所が分かる書き方にするか，Kornblith et al.（2019）を直接引く（Kornblith は本監査では確かめていない）．本稿の LoRA の確かめ（6.4 節）も根拠に添える．
- **m-3 序論第 2 段落と 2 節第 1 段落（I5，R8：ahn2025multitask）**
  - 本文："cross-corpus methods that use several modalities, including the face, remain few [ahn]"．
  - 出典：抄録 "Multimodal approaches for cross-corpus scenario is substantially limited as well."．Ahn らのモダリティは音声と文字で，顔については述べていない．
  - 補足：顔を含むコーパス間のマルチモーダル研究として，MERBench（Lian et al., IEEE TPAMI 48:5793–5810, 2026，doi 10.1109/TPAMI.2026.3653457．arXiv:2401.03429 の 5.4 節で，IEMOCAP の 4 クラス・MELD・MER-MULTI の共通 4 ラベルについて，視覚・音響・言語の特徴と融合の交差コーパス結果を報告）と，Ryumina et al.（Pattern Recognition Letters 190:192–200, 2025，doi 10.1016/j.patrec.2025.02.024．音声・映像・文字，MOSEI・MELD・IEMOCAP・AFEW で leave-one-corpus-out）がある．どちらもターゲットのラベル数を変えていないので，2 節第 1 段落の "none varies $K$ over many corpus pairs with three modalities including the face" は保たれる．
  - 対応：この 2 本を引き，「少ない」の根拠を ahn だけにしない．
- **m-4 2 節第 5 段落（R34：pendyala2025leveraging）**
  - 本文："audio-visual or video teachers [pendyala, hajavi] ... train speech students"．
  - 出典：教師は 2 つの単一モダリティのモデルである．"We developed the speech teacher model by fine-tuning the pre-trained HuBERT for the SER task, utilizing labeled speech samples from the same dataset used to train the student model."，"Our video teacher is an S2D model trained in [14], using video samples from the FERV39k corpus"．融合した音声映像の教師ではない．
  - 対応："speech and video teachers [pendyala] or video teachers [hajavi]" と書く．台帳では「一部不一致」だが，本文の区別（教師はターゲットか第三のコーパスで学習）は Pendyala にも当てはまり，主張の結論は変わらないので Minor とした．
- **m-5 2 節第 2 段落（R15：rosenfeld2022domainadjusted）**
  - 本文："domain-adjusted regression bounds the excess risk through the error of the target mean"．
  - 出典：Theorem 5.6 の上界は，対立的に選んだテスト環境の平均のノルムの上界 $\rho$ で決まる（"sup Re′(β̂) = (1 + ρ²)(…)"，条件 "‖b_e′‖ ≤ ρ"）．証明の概略は "We decompose the excess risk into error on the the target mean β*ᵀb_e′ and error on the residuals"．
  - 対応："error of the target mean" は平均の推定誤差と読まれるので，「テスト環境の平均の大きさと共分散の調整誤差で抑える」と書く．出版版は無く，NeurIPS 2022 DistShift ワークショップ版がある（OpenReview Ypo0AckYW8）．
- **m-6 2 節第 2 段落（R17：kini2021labelimbalanced）**
  - 本文："additive logit corrections lose their effect as the regularization vanishes"．
  - 出典：Theorem 1 は "Assume ... the data are linearly separable" のもとでの結果である．
  - 対応：「線形分離可能なデータで」を加える（4.2 節の (A2) は条件を書けている）．
- **m-7 2 節第 3 段落（R20：lescao2021data）**
  - 本文："The value of pretraining or of a source has been expressed as an effective amount of target data [hernandez, lescao]"．
  - 出典：Le Scao & Rush の抄録 "Results show that prompting is often worth 100s of data points on average"．データ点数で表したのはプロンプトの価値である．
  - 対応："the value of pretraining [hernandez] and of a prompt [lescao]" と分ける．Hanneke & Kpotufe（2019）の Remark 2 は "effective sample size contributed by P" と書くので，ソースの価値の例としても引ける．
- **m-8 2 節第 4 段落（R25：tiomoko2021deciphering）**
  - 本文："The bias of pooled linear classifiers has been solved in high-dimensional limits for centered multi-task models with task-specific intercepts"．
  - 出典：抄録 "studying the tractable least-square support vector machine multi-task learning (LS-SVM MTL) method ... We prove (i) that the standard MTL LS-SVM algorithm is in general strongly biased"．プールはその極限である（前身の arXiv:2009.01591 "as λ → ∞, both source and target tasks are considered together as one task"）．原典の "bias" は ±1 のラベルの選び方による負の転移と判定の閾値のずれを指す．
  - 対応："for the LS-SVM multi-task model, which contains pooling as a limit" と範囲を書く．中心化と課題ごとの切片は前身の版で確かめた（ICLR 版の本文は未確認，7 節）．
- **m-9 3.6 節と 2 節第 6 段落（S7，R44：gulrajani2021domainbed）**
  - 本文："selected with pseudo-targets, as in leave-one-domain-out selection"．
  - 出典：Sec. 3.1 "Given d_tr training domains, we train d_tr models with equal hyperparameters, each holding one of the training domains out. We evaluate each model on its held-out domain, and average the accuracies"．外したドメインにラベルを与える手続きは無い．
  - 対応：本稿は $c$ と同じコーパスの群も外し，擬似ターゲット $c$ に $K$ 本のラベルを与えるので，"a corpus-level variant of leave-one-domain-out selection in which the held-out group also receives $K$ labels" と書く．原典の Sec. 4 "model selection with a training domain validation set outperforms leave-one-domain-out cross-validation across multiple datasets and algorithms" は 6.4 節の選択の損失の議論で触れるとよい．
- **m-10 2 節第 5 段落（R35：jaritz2020xmuda）**
  - 本文："fused outputs train unimodal heads [jaritz]"．
  - 出典：これに当たるのは 4.4 節の xMUDA Fusion である．本体の xMUDA は "the objective of the 2D mimicry head is to estimate the main 3D output and vice versa"（図 2）という相互模倣である．
  - 対応：xMUDA Fusion と特定する（M-3 と一緒に直す）．
- **m-11 2 節第 6 段落と 8 節（R40，R41，L1：berthelot2022adamatch，veilleux2021realistic）**
  - 本文："Balancing pseudo-labels across classes is common [..., berthelot]"，"class imbalance in the unlabeled set can make such balancing harmful [veilleux]"，8 節 "stronger skew can make it harmful [veilleux]"．
  - 出典：AdaMatch の分布の整列は擬似ラベルのクラス比をソースのラベル分布に合わせる（3 節 "when the target label distribution is unknown, we approximate it using the only available distribution – the source label distribution"，"aligning the target pseudo-labels to match the source label distribution helps significantly"）．均等化と同じになるのはソースが均衡のときだけである．Veilleux et al. が示したのは転導的な少数ショット推論に組み込んだクラス均衡の制約の害（"Such a hard constraint is detrimental to the performance in more realistic scenarios where the class distributions of the query sets could be arbitrary"）であり，擬似ラベルで生徒を学習する場面ではない．
  - 対応：AdaMatch は「分布の整列」として別に書くか外す．Veilleux を引く 2 箇所は「転導的推論での結果からの類推」と分かる書き方にする．AdaMatch 自身の "the more the source label distribution differs from the target distribution, the more incorrect the approximation will be, which may cause the model performance to degrade" も 8 節の根拠に使える．
- **m-12 補足 S2（X5：okamoto1963asymptotic，raudys1998intrinsic）**
  - 本文："for the sample linear discriminant with $p$ features and squared Mahalanobis distance $\Delta^2$, the first-order term of the expansion of the misclassification rate gives $q = 2(p-1)/\Delta^2$"．
  - 出典：6.3 のとおり，この値は平均の推定に由来する項だけから出る．標本線形判別では共分散の推定による項 $a_3/n$ も 1 次に入り，$q$ に $(p-1)/2$ 程度を加える（Shutoh 2010 の再掲による監査者の計算）．Raudys も Fisher の判別に共分散の因子 $T_\Sigma$ を掛ける．
  - 対応：「共分散が既知のとき（平均の推定による項だけを取ると）」と条件を書くか，共分散の項を加える．Okamoto（1968）の訂正を引く．
- **m-13 2 節第 3 段落と 4.5 節（R23，P5：raudys1998intrinsic）**
  - 本文："the classical relation between the cosine of an estimated direction and the sample size [raudys, dobriban]"．
  - 出典：Raudys（1998）に推定方向の余弦の記述は無い．Dobriban & Wager は古典的な出典として "Efron (1975) studies the angle Γ in detail for low-dimensional asymptotics" を挙げる．
  - 対応：Efron（1975）を加える（5 節の項目 4）．
- **m-14 4.5 節と補足 S2（式 (5) の導出）**
  - 本文："Combining two independent estimates with optimal weights adds their sample sizes"．
  - 出典との関係：引いた文献の関係は標本の揺らぎによる方向の誤差についてのものである．ソースの方向の系統的なずれを同じ等方的な雑音として逆分散で重ねるのは，本稿の仮定である．
  - 対応：その仮定（ソースのずれが白色化した空間で等方的で，ターゲットの推定誤差と独立）を明記する．
- **m-15 2 節第 7 段落（R45，R46：charikar2024quantifying，dong2025discrepancies，xue2025representations）**
  - 本文："The gain of a student trained on a weak teacher's labels has been bounded from below by their misfit [charikar] and related to the disagreement between their representations [dong, xue]. We use these results for a student anchored to a start: the teacher's UAR advantage over the start sets the scale of the gain"．
  - 出典：Charikar et al. の定理は二乗損失の回帰で強いモデルが凸な関数集合から学ぶ場合の結果で（"for a convex set of functions Fs"，"dP(f,g) = E(f(x)−g(x))^2"），結論で "Our work also does not address classification tasks" と書く．Dong et al. は抄録で "in the ridgeless regression setting"，Xue et al. は本文で "While Thm. 3.8 is based on linear regression with MSE loss" と書く．
  - 対応：これらの条件を書き，UAR で測る分類への当てはめは期待であると書く．5.3 節は "we expect, without a theorem" と書いているので，2 節の "We use these results" もそれに揃える．
- **m-16 3.2 節と補足 S1（S3，X3：radford2021clip）**
  - 本文：補足 "the image encoder of CLIP ViT-L/14 ... maps the same crops ... to 1024 dimensions"．
  - 出典：arXiv 版の表で ViT-L/14 の埋め込みの次元は 768，ViT の幅は 1024 である．Hugging Face の openai/clip-vit-large-patch14 も vision hidden_size 1024，projection_dim 768 である（補助の確認による）．
  - 対応：射影前の出力を使ったと書く（射影後なら 768 に直す）．
- **m-17 3.1 節と 8 節（S1：コーパスの記述）**
  - MELD のラベルは "Joy" である（"Ekman's six universal emotions (Joy, Sadness, Fear, Anger, Surprise, and Disgust) ... extended with ... Neutral"）．happy との対応を書く．
  - 8 節 "six English corpora"：Dhall et al.（2012）に AFEW の言語の記述は無い．IEMOCAP・CMU-MOSEI・CREMA-D も原典に明示は無く，英語と明記するのは RAVDESS（"actors needed to have English as their first language"）だけである．言語の根拠を示すか，書き方を弱める．
  - CMU-MOSEI は "annotated on a [0,3] Likert scale for presence of emotion ... by 3 crowdsourced judges" であり，3 人の評点のまとめ方は原典に無い．閾値 0.5 と評点のまとめ方が本稿の規則であることを書く．
- **m-18 6.1 節（C5：blum1998combining）**
  - 出典："select the p examples it most confidently labels as positive, and the n examples it most confidently labels negative"（2 クラス，naive Bayes）．
  - 対応：4 クラスへの移し方を書く（M-2 と一緒に直す）．
- **m-19 書誌の細部**
  - scholkopf2001generalized：PDF の脚注 "COLT/EuroCOLT 2001, LNAI 2111, pp. 416–426" に合わせ，巻 2111 を書き足す．
  - du2021improving：arXiv だけである．同じグループの査読版の別論文（Du et al., "On Uni-Modal Feature Learning in Supervised Multi-Modal Learning", ICML 2023, PMLR 202:8632–8656．補助の確認による）が UMT を含むので，差し替えを検討する．
  - zaidi2024enhancing：内容は題名の違う arXiv の前身版（2306.13804v3）で確かめた．数値を引く場合は OJCS 版で確かめる．
  - he2025m4ser：少数ショットの表（Table IX）は arXiv:2509.18706v1 で確かめた．出版版で同じかを確かめる．
- **m-20 2 節第 5 段落（R36 のうち albanie2018emotion）**
  - 出典：Albanie et al. の教師は第三のコーパス（FERplus）で学習し，蒸留はターゲットではない VoxCeleb の上で行い，評価コーパスでは affine 写像を 10 分割交差検証で学ぶ．
  - 対応：本稿との差は「教師の学習データ」よりも「教師がラベルを付けるのがターゲットのラベル無しのプールかどうか」にある．M-3 の書き直しでこの差も書くとよい．
- **m-21 序論第 2 段落（新しさの文 N2）**
  - 本文："Neither line says ... what a second modality can add when the deployed model sees only one"．
  - 出典との関係：本文 2 節第 5 段落で引く albanie2018emotion，srinivasan2022representation，pendyala2025leveraging，hajavi2024audio は，配備するモデルが音声だけのときに他モダリティが何を足すかを測っている．序論で引く lopezpaz2016unifying も同じ問いの枠組みである．
  - 対応：「コーパスをまたぎ，ターゲットのラベルが少ない場合に」と範囲を限る．


## 4. 新しさを言う文の監査

本文が「先行研究に無い」「〜とは違う」と述べる文を挙げ，近い先例を探した．「確かめた水準」は先例の文献を読んだ範囲である．先例が見つからなかった項目でも，検索した範囲の外に先例が無いとは言えない．

| 番号 | 箇所と文 | 見つけた近い先例（書誌・確かめた水準） | 判定 |
|---|---|---|---|
| N1 | 序論の貢献 1："A gain map of cross-corpus pooling with three modalities and their fusions, including the face, over all ordered pairs of seven groups and label budgets from one label per emotion to the full pool" | Schuller et al. 2010（音響だけ，6 コーパスのコーパス間評価，ターゲットのラベル無し．抄録）．MERBench（Lian et al., IEEE TPAMI 48:5793–5810, 2026．視覚・音響・言語と融合のコーパス間評価，ターゲットのラベル無し．arXiv:2401.03429 の 5.4 節）．Ryumina et al., Pattern Recognition Letters 190:192–200, 2025（音声・映像・文字，MOSEI・MELD・IEMOCAP・AFEW で leave-one-corpus-out．研究室の告知の頁と Crossref の書誌だけ）．Gideon et al. 2021（ターゲットのラベル 0〜3200 発話，音声だけ，2〜3 コーパス．本文）．He et al. 2025（クラスあたり 5〜60，IEMOCAP↔MELD，音声＋文字．本文） | 主張は保たれる．3 モダリティと融合を，全ての順序対とラベル数の範囲で測った先例は見つからなかった．ただし MERBench と Ryumina et al. 2025 は顔を含むコーパス間のマルチモーダル評価なので，関連研究で引く（m-3） |
| N2 | 序論第 2 段落："Neither line says which part of a classifier the few target labels repair, how much a source supplies in their place, or what a second modality can add when the deployed model sees only one" | 第 1・第 2 の点：台帳の 13 本はどれも分類器のどの部分をラベルが直すかを測っていない（2 節）．第 3 の点：Albanie et al. 2018，Srinivasan et al. 2022，Pendyala et al. 2025，Hajavi & Etemad 2024 は，音声だけを配備するときに他モダリティが足すものを測っている（いずれも本文） | 第 1・第 2 の点は保たれる．第 3 の点は範囲を限らないと言い過ぎになる（m-21） |
| N3 | 2 節第 1 段落："None of these studies measures which part of the classifier the target labels change, and none varies $K$ over many corpus pairs with three modalities including the face" | 台帳の文献のうち $K$ を変えるのは Gideon，Feng，Chen & Yu，He，Zaidi，Xu だが，どれも 1〜2 モダリティで，コーパスの対は少ない | 保たれる |
| N4 | 序論の貢献 2：閾値と方向への分解 | 分解そのものの先例は見つからなかった．関連：コーパスごとの正規化（Schuller 2010，引用済み），領域ごとの統計の取り直し（AdaBN：Li et al., Pattern Recognition 80:109–117, 2018，doi 10.1016/j.patcog.2018.03.005．書誌だけ），事前確率の調整（Saerens 2002，引用済み），ドメイン調整回帰（Rosenfeld 2022，引用済み），転移 LDA で目標領域のクラス数の偏りによる切片の偏りを扱う Zhang et al.（arXiv:2607.06936，2026．抄録："We also characterize the intercept bias induced by unbalanced target-domain class sample sizes"）．検索語は "few target labels correct classifier bias intercept rather than weight direction"，"threshold direction decomposition domain shift linear probe" など | 保たれる（検索の範囲で）．Zhang et al. 2026 は切片と方向の両方を扱う転移 LDA なので関連研究で触れるとよい |
| N5 | 序論の貢献 3："A conversion of the value of a source into target labels, $n = q e^2/(1-e^2)$" | Hernandez et al. 2021（事前学習から「移った」実効データ量．抄録・Fig. 1），Hanneke & Kpotufe 2019 の Remark 2（"effective sample size contributed by P"．本文），Zhang et al. 2026（arXiv:2607.06936．ソースとターゲットの方向を重みつきで合わせる転移 LDA で，誤り率の極限と最適な転移の重みを導く．抄録），Sun, Zhang & Watkins（arXiv:2510.10866．二方向の汎化からデータの類似度を測り，標準的な線形モデルで決定境界の余弦と結びつける．抄録） | 式 (5) と同じ式は見つからなかった．主張は保たれる．Zhang et al. 2026 と Sun et al. は「方向を合わせる重み」と「決定境界の余弦」を扱うので，"Value of a source" の段落で引いて差を書くとよい（5 節の項目 7） |
| N6 | 序論の貢献 4 と 2 節第 5〜6 段落："the teachers of our basic method are trained on the source corpora alone and never refit on the $K$ labels, and our student uses only its own modalities at test time"，"Specific to our combination are ..." | xMUDA Fusion（Jaritz et al. 2020，本文），BFtD（Wu et al., ACM MM 2023．抄録と後継論文の説明），FtD++（Wu et al., TCSVT 2025．arXiv 版の該当節）：ソースだけにラベルがある設定で，他モダリティ・融合の出力が単一モダリティのヘッドを教える．Hatano et al. 2024（モダリティ別の教師をソースとターゲットで事前学習して固定し，ラベル無しのターゲットで RGB の生徒に蒸留．本文の該当節）．Phoo & Hariharan 2021（固定した教師が全てのラベル無しデータに柔らかいラベルを付ける．該当節）．感情認識：Albanie 2018，Li et al. 2021（MCED，Interspeech 2021．文字の教師を第三のコーパスで学習し，映画の音声と文字の組で音声の生徒へ蒸留．本文の冒頭），Pendyala 2025 | 組み合わせとしては保たれる．感情認識で「ソースのコーパスだけで学習した他モダリティの教師がターゲットのコーパスのラベル無しのプールにラベルを付け，単一モダリティの生徒を作る」先例は見つからなかった（検索語は補助の 6 本と監査者の "cross-corpus speech emotion recognition multimodal teacher pseudo-labels unlabeled target corpus distillation audio-only student domain adaptation"）．ただし区別の文の xMUDA の扱いは誤り（M-3） |
| N7 | 2 節第 1 段落："Gideon et al. leave open when to generalize and when to specialize, and Feng and Chaspari the effect of source properties and of combining sources; our analysis addresses both for linear heads" | 原文の確認は 2 節の台帳のとおり（Gideon："it is important to formulate exactly when to generalize and when to specialize"．Feng："it would be beneficial to explore the extent to which the combination of multiple source datasets can benefit few-shot learning"） | 保たれる |
| N8 | 2 節第 2 段落："Finite-sample analyses of transfer with few target samples exist for regression [ha] and one-dimensional discriminants [desilva]; we measure the corresponding quantities for multiclass logistic regression" | Ha & Chen（arXiv:2507.14661v2）は "Extending the finite-sample analysis to logistic, cross-entropy, or other losses is left for future work." と書く（補助の確認） | 保たれる |
| N9 | 2 節第 4 段落：式 (4) は "extends the one-dimensional threshold shift of De Silva et al. with a weight path" | De Silva の $E[\hat c]=m\Delta/(n+m)$ への帰着は 6.4 で確かめた．群影響の一歩近似（Koh 2019）自体は重みの変化も含む一般の式なので，新しさは「ターゲットの切片を持たないプールに加えた点へ当てはめ，閾値の補正を 2 つの経路に分けた」ことにある | 保たれる．"extends" の中身（どこが新しいか）を上の形で 1 句足すとよい |
| N10 | 序論の最後："To our knowledge, these four combinations of setting and measurement have not been reported before" | N1，N4，N5，N6 のとおり | 「組み合わせ」として保たれる．N6 の書き直し（M-3）を済ませてから残す |


## 5. 引用漏れの一覧

事実として断定しているのに引用が無い記述と，引くべき先例を挙げる．「書誌の確認」は本監査で確かめた範囲である．

| 番号 | 箇所 | 記述 | 引くべき文献の候補（書誌の確認） |
|---|---|---|---|
| 1 | 序論第 3 段落，6.4 節，補足 S7 | 低ランク適応（"low-rank adaptation"，"low-rank adapters"）を使った | Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", arXiv:2106.09685（arXiv の記録で確認．ICLR 2022 での採録は本監査では確かめていない） |
| 2 | 3.1 節 | IEMOCAP の即興と台本を別の群にする | 同じ分け方の先例：feng2023fewshot（ソースに IEMOCAP の台本，ターゲットに即興．本文で確認），dong2026test（課題 2 "Acted to natural emotion adaptation" を IEMOCAP で行う．本文で確認），Li et al., "Acted vs. Improvised: Domain Adaptation for Elicitation Approaches in Audio-Visual Emotion Recognition", Interspeech 2021, pp. 3395–3399, doi 10.21437/Interspeech.2021-666（`references.bib` に記録があり，補助が抄録で内容を確認） |
| 3 | 序論第 2 段落，2 節第 1 段落 | 顔を含むコーパス間のマルチモーダル研究は少ない | MERBench（Lian et al., IEEE TPAMI 48:5793–5810, 2026, doi 10.1109/TPAMI.2026.3653457．Crossref で書誌，arXiv:2401.03429 で内容を確認），Ryumina et al., "Multi-corpus emotion recognition method based on cross-modal gated attention fusion", Pattern Recognition Letters 190:192–200, 2025, doi 10.1016/j.patrec.2025.02.024（Crossref で書誌，研究室の告知の頁で内容を確認） |
| 4 | 2 節第 3 段落，4.5 節，補足 S2 | 推定方向の余弦と標本サイズの古典的な関係 | Efron, "The Efficiency of Logistic Regression Compared to Normal Discriminant Analysis", JASA 70(352):892–898, 1975, doi 10.1080/01621459.1975.10480319（Crossref で書誌．内容は Dobriban & Wager の記述による） |
| 5 | 2 節第 2 段落，補足 S2 | Okamoto の展開の係数 | Okamoto, "Correction to 'An Asymptotic Expansion for the Distribution of the Linear Discriminant Function'", Ann. Math. Statist. 39(4):1358–1359, 1968, doi 10.1214/aoms/1177698265（Crossref で書誌） |
| 6 | 3.5 節，5.1 節 | 縮小した線形判別（$\Delta$ の基準），$((1-\rho_s)\Sigma_s+\rho_s\tau_s I)^{-1}\tilde M_s$ の閉じた形の判別方向 | Friedman, "Regularized Discriminant Analysis", JASA 84(405):165–175, 1989, doi 10.1080/01621459.1989.10478752．Ledoit & Wolf, "A well-conditioned estimator for large-dimensional covariance matrices", J. Multivariate Anal. 88(2):365–411, 2004, doi 10.1016/S0047-259X(03)00096-4（$\tau_s$ を固有値の平均とする縮小の目標はこの形）．どちらも Crossref で書誌だけ確認 |
| 7 | 2 節第 3 段落 | ソースの価値をターゲットの標本数で表す先行研究 | Zhang, Fan, Luo & Hu, "Transfer Learning for Linear Discriminant Analysis with a Shared Classification Signal", arXiv:2607.06936, 2026（抄録で確認）．Sun, Zhang & Watkins, "Quantifying Data Similarity Using Cross Learning", arXiv:2510.10866v3（抄録で確認）．Hanneke & Kpotufe の Remark 2（引用済み）をこの文脈でも引く |
| 8 | 5.2 節 | ラベル無しの選び方 7 種（"margin, herding, a k-center coreset, and typicality clustering"） | herding：Welling, "Herding dynamical weights to learn", ICML 2009, pp. 1121–1128, doi 10.1145/1553374.1553517（Crossref で書誌）．k-center coreset：Sener & Savarese, "Active Learning for Convolutional Neural Networks: A Core-Set Approach", arXiv:1708.00489（arXiv の記録の注記 "ICLR 2018 Paper"）．typicality：hacohen2022active（引用済み．この箇所でも引く）．margin とほかの 3 種は出典を本監査で確かめていない |
| 9 | 2 節第 5〜6 段落 | 交差モダリティの教師なし DA で融合の出力が単一モダリティのヘッドを教える後継研究 | Wu et al., ACM MM 2023, pp. 490–498, doi 10.1145/3581783.3612013（抄録）．Wu et al., IEEE TCSVT 35(9):9030–9045, 2025, doi 10.1109/TCSVT.2025.3545971（arXiv 版の該当節）．M-3 |
| 10 | 4.3 節，5.1 節 | ソースとターゲットをそれぞれのラベル無しの平均で中心化する | コーパスごとの正規化：schuller2010crosscorpus（関連研究で引用済み．この箇所でも引く）．領域ごとに特徴の統計を取り直す AdaBN：Li et al., Pattern Recognition 80:109–117, 2018, doi 10.1016/j.patcog.2018.03.005（Crossref で書誌だけ） |
| 11 | 6.1 節 | 比較相手の「教師の確信度の閾値による硬い擬似ラベル」と「開始点の出力の再均衡」 | 引用が無く，fixed 版の値の出どころも書かれていない（M-2）．典拠があれば引き，無ければ本稿の定義だと書く．本監査では候補を確かめていない |
| 12 | 7 節 | "Entropy minimization relies on classes that form separate clusters in the unlabeled data" | grandvalet2004semisupervised（引用済み）をこの箇所でも引くか，クラスタ仮定の出典を足す（本監査では出典を確かめていない） |
| 13 | 6.2 節 | Wilcoxon の符号順位検定と Holm の補正 | 標準的な手続きであり必須ではない．引くなら Holm（1979）．本監査では書誌を確かめていない |


## 6. 式と命題の出典

本文の式番号は PDF に合わせた（(1) 2 つの雲の正答率，(2) 閾値のコスト，(3) 方向のコスト，(4) $\lambda$，(5) ソースの価値 $n$，(6) 生徒の目的関数）．

### 6.1 式 (1) $a(d,o)=\tfrac12[\Phi(d/2+o)+\Phi(d/2-o)]$（R14）

- 出典の該当箇所：De Silva et al.（ICML 2023）の付録 A.3 の式 (7) は，1 次元・単位分散・等しい事前確率の 2 群で，閾値 $\hat c$ の誤り率を "$\tfrac12\{1+\Phi(\hat c-\mu)-\Phi(\hat c+\mu)\}$" と書く．Dobriban & Wager（Ann. Statist. 2018）の Theorem 3.2 は，切片 $c$ を持つ正則化判別の誤り率の極限を "$\pi_-\Phi(-\Theta_-)+\pi_+\Phi(-\Theta_+)$" と書く．どちらも式 (1) と同じ形である．
- 仮定：両文献とも共通の分散（共分散）を持つガウスの 2 群である．本文の仮定 (A1) と合う．
- 判定：一致．確かめた水準：両文献とも本文．

### 6.2 式 (2) 閾値のコスト（引用なし）

- 補足 S2 の導出を検算した．$K$ 本ずつから推定した 2 つの平均の中点の分散は $\sigma^2/(2K)$，$\partial^2 a/\partial o^2|_{o=0}=-(d/2)\phi(d/2)$，$(d/4)\phi(d/2)$ の最大は $d=2$ で $\phi(1)/2$ であり，式 (2) は正しい．
- 補足：Okamoto（1963）の展開を Shutoh（2010，広島統計研究グループ技術報告 10-04）の再掲から読み取り，$N_1=N_2=N$，$v=-\Delta/2$ で平均の推定による 1 次の項を計算すると $\phi(\Delta/2)[\Delta/(8N)+(p-1)/(2N\Delta)]$ になる．次元によらない部分 $\phi(\Delta/2)\Delta/(8N)$ は式 (2) の $(d/4)\phi(d/2)/(2K)$ と一致する．これは「次元によらない $1/N$ の項が閾値に当たる」という 2 節第 2 段落の読みを支える．ただしこの計算は再掲された式の読み取りに基づく監査者の計算であり，Okamoto の原典では確かめていない（7 節）．

### 6.3 式 (3) $d(K)=\Delta/\sqrt{1+q/K}$ と補足の $q=2(p-1)/\Delta^2$（P2，R12，R13，X5）

- Raudys（1998）の式 (1) は，ユークリッド距離分類器（最近平均分類器）の期待誤り率を $\Phi\{-(\delta^*/2)/\sqrt{T^*}\}$，$T^*=1+2p^*/(\delta^{*2}N)$ と書き，$p^*$ を "an effective dimensionality" と呼ぶ．これは式 (3) の形そのもの（$q=2p^*/\delta^{*2}$）である．一方 Raudys は Fisher の線形判別について，抄録で "the Fisher linear discriminant function is sensitive to all dimensions"，本文 2 節で "For GCCM model, however, the generalization error depends on the true dimensionality p, and not on p*" と書き，さらに共分散の推定による因子 $T_\Sigma$ を掛ける．したがって「標本線形判別の展開の次元比例の項を Raudys が実効次元で表した」という R13 の書き方は，Raudys が Fisher の判別について述べたことと逆になる（3 節の M-1）．
- Okamoto（1963）の展開（Shutoh 2010 の再掲による）で，平均の推定に由来する $1/N_1$，$1/N_2$ の項からは，上の計算のとおり $q=2(p-1)/\Delta^2$ が出る．ところが標本線形判別は共分散も推定するので，自由度 $n$ の項 $a_3/n$ も 1 次に入る．$a_3=4v^3+4\Delta v^2+(6p-6+\Delta^2)v+2(p-1)\Delta$ は $v=-\Delta/2$ で $-(p-1)\Delta$ となり，誤り率に $(p-1)\Delta\phi(\Delta/2)/(4n)$ を足す．$n\approx 2K$ なら $q$ に $(p-1)/2$ が加わる．補足 S2 の "for the sample linear discriminant with $p$ features ..., the first-order term of the expansion of the misclassification rate gives $q = 2(p-1)/\Delta^2$" は，共分散を既知とした場合（平均の推定の項だけ）に限られる（3 節の m-12）．
- Okamoto（1963）には訂正がある：M. Okamoto, "Correction to 'An Asymptotic Expansion for the Distribution of the Linear Discriminant Function'," Ann. Math. Statist. 39(4):1358–1359, 1968, doi 10.1214/aoms/1177698265（Crossref で書誌を確認．Shutoh 2010 も "with correction, Okamoto (1968)" と書く）．展開の係数を根拠に使うなら訂正も引く．
- 本文の学習器は罰則つき多クラスロジスティック回帰であり，$q$ は学習曲線への当てはめで決めている．したがって上の不正確さは測った数値には影響しない．影響するのは「$q$ がどの理論量に当たるか」という説明だけである．

### 6.4 式 (4) $\lambda=\rho/(1+\rho)$（P3，P4，R28，R29）

- Koh et al.（NeurIPS 2019）は群の除去の効果を "the one-step Newton approximation (Pregibon et al., 1981)" で特徴づけ，脚注で "In the statistics literature, influence typically refers to the effect of adding weight, so the sign is flipped." と書く．本文は「加えた点について書いた」と明記しており，使い方は合う．
- De Silva et al. の付録 A.3 は，$n$ 個のターゲットと $m$ 個の OOD をまとめた 1 次元の Fisher 判別で "$\mu_h=E[\hat c]=m\Delta/(n+m)$" を与える．式 (4) の $\rho$ を切片の項だけにすると $1-\lambda=N_s\kappa_s/(N_s\kappa_s+N_t\kappa_t)$ で，$\kappa_s=\kappa_t$ なら $m/(n+m)$ に一致する．P4 は正確である．
- Pregibon（1981）の本文は取得できず，一歩近似の中身は Koh らの記述と抄録で確かめた（7 節）．
- 判定：一致．

### 6.5 式 (5) $n=q\,e^2/(1-e^2)$（P5，R23）

- Dobriban & Wager の Corollary 3.3 は，$\Sigma=I$ で "We need to pay a price $\alpha/\sqrt{\alpha^2+\gamma}$ for estimating $\mu_{\pm1}$" と書く（$\gamma=p/n$，$\alpha^2=E\|\delta\|^2$）．$\cos^2\theta=1/(1+\gamma/\alpha^2)$ であり，$n=2K$，$\alpha=\Delta/2$ とおくと $\tan^2\theta=2p/(K\Delta^2)$，すなわち本文の $\tan^2\theta\approx q/K$ と同じ形になる．R23・P5 の「古典的な関係」は Dobriban & Wager で支えられる．
- Raudys（1998）には推定方向の余弦の記述は無く，誤り率の式だけがある．余弦と標本サイズの関係の古典的な出典は，Dobriban & Wager 自身が "Efron (1975) studies the angle Γ in detail for low-dimensional asymptotics ... the sampling distribution of n(1 − Γ) converges to a (scaled) $\chi^2_{p-1}$ distribution" と挙げる Efron（1975）である（3 節の m-13）．
- 仮定の対応：Dobriban & Wager と Efron の関係は「標本の揺らぎによる方向の誤差」についてのものである．本文の導出（補足 S2）はソースの方向のずれ（標本の揺らぎではなく系統的なずれ）を同じ雑音比 $(1-e^2)/e^2$ として逆分散で重ねている．これは引用した文献の結果ではなく本文の仮定（ソースのずれが白色化した空間で等方的で，ターゲットの推定誤差と独立）であり，そう明記するのがよい（3 節の m-14）．

### 6.6 命題 1（表現定理，B1）

- Schölkopf et al.（COLT 2001）の抄録は "Wahba's classical representer theorem states that the solutions of certain risk minimization problems involving an empirical risk term and a quadratic regularizer can be written as expansions in terms of the training examples. We generalize the theorem to a larger class of regularizers and empirical risk terms" と書く．
- 本文は $V=W-W_0$ を変数に取り直す．損失は $V\mathbf z_i+W_0\mathbf z_i+b$ を通してだけ $V$ に依存し，罰則は $\|V\|_M^2$ なので，$U=VM^{1/2}$ について表現定理を使えば $V$ の各行は $M^{-1}\mathbf z_i$ の張る空間に入る．切片には別の罰則があるが，$V$ の最適性条件には影響しない．使い方は合う．
- 命題 1 は式 (6)（弱いラベルの項を含まない）についての主張である．5.3 節で弱いラベルの項を足すと張る点にプールの $N_u$ 点が加わる．本文の議論の順序はこれと矛盾しない．
- 確かめた水準：Schölkopf et al. は抄録と冒頭 2 頁．Kimeldorf & Wahba（1971）は書誌だけ（7 節）．

### 6.7 命題 2（ラベルを使わない計画，B2，R47）

- Chaloner（1984）の抄録は，線形モデルのベイズ最適計画が "minimizing the quantity $\mathrm{tr}\,\psi(R + XX^T)^{-1}$" に帰着すると書く．基準は計画 $X$ と事前精度 $R$ だけで決まり，観測した応答によらない．本文の「ガウス線形モデルでは事後共分散が観測ラベルによらないので最良の計画もよらない」と一致する．
- 本文は多クラスロジスティック回帰をガウス尤度で近似したうえでこれを使うと明記し（命題 2 の仮定），近似を実験で確かめている．非線形モデルでは最適計画がパラメータに依存するという一般論と矛盾しない書き方である．
- 確かめた水準：Chaloner（1984）と Chaloner & Verdinelli（1995）は抄録．

### 6.8 仮定 (A2) と Kini et al.（P1，R17）

- Kini et al. の Theorem 1 は線形分離可能なデータで，ノルム制約 $R\to\infty$ のとき解の向きが SVM 解に収束することを示し，直後に "the theorem makes clear that ω± and ι± become ineffective in the TPT as they all result in the same SVM solutions" と書く．Remark 1 は消えゆく正則化と結びつける．
- (A2) は "with a vanishing penalty on separable data" と条件を書いており一致する．R17（関連研究）は「分離可能」の条件を落としている（3 節の m-6）．


## 7. 未検証の一覧と理由

### 7.1 主張を判定していない文献（2 本）

| 引用キー | 確かめたこと | 確かめられなかったこと・理由 |
|---|---|---|
| kimeldorf1971results | Crossref の書誌（J. Math. Anal. Appl. 33(1):82–95, 1971）．Schölkopf et al. 2001 の抄録が "Wahba's classical representer theorem" と呼ぶこと | 本文と抄録を入手できなかった．命題 1 の根拠は Schölkopf et al. 2001 の抄録で判定した |
| ryumina2022facial | Crossref の書誌（Neurocomputing 514:435–450, 2022）．著者の GitHub の README（"the backbone emotion recognition model, which is based on the VGGFace2 ... ResNet50 model"，"the backbone model was fine-tuned on the largest facial expression dataset AffectNet"）とコード（512 次元の 'features' 層）．補助の確認による | ScienceDirect が 403 を返し，OpenAlex にも抄録が無く，論文の本文と抄録を読めなかった．3.2 節と補足 S1 の記述が論文の内容と合うかは未検証である |

### 7.2 抄録または二次資料の水準で判定した文献（14 本）

schuller2010crosscorpus，ahn2025multitask，wiesenfarth2020quantification，vapnik2009new，gupta2016cross，dong2025discrepancies，chaloner1984optimal，chaloner1995bayesian，scholkopf2001generalized（抄録と冒頭 2 頁），okamoto1963asymptotic，pregibon1981logistic，cao2018vggface2，mollahosseini2019affectnet，liu1989lbfgs（抄録の抜粋）．

- okamoto1963asymptotic と pregibon1981logistic：Project Euclid がボット対策で PDF を返さなかった．Okamoto は抄録（OpenAlex）と Shutoh（2010）の再掲で，Pregibon は抄録と Koh et al. 2019 の記述で判定した．6.2・6.3 の係数の計算は，pdftotext で崩れた再掲の式を監査者が読み取ったものであり，著者は Okamoto（1963）と訂正（1968）の原典で確かめる必要がある．
- schuller2010crosscorpus と ahn2025multitask：全文は購読が必要で取得できなかった．Schuller の 6 コーパスの名前，ラベル，評価指標は未確認である．
- vapnik2009new：ScienceDirect が 403．R30 の文言は lopezpaz2016unifying の 4.2 節で支えられる．

### 7.3 出版版との照合をしていないもの

- he2025m4ser：少数ショットの表を arXiv v1 で確認．IEEE TASLPRO 版は未確認．
- zaidi2024enhancing：題名の違う arXiv 版（2306.13804v3）で確認．OJCS 版は未確認．
- tiomoko2021deciphering：ICLR 版は抄録だけ．モデルの式は前身の arXiv:2009.01591 で確認．
- kini2021labelimbalanced：多クラスへの拡張がある補遺は未読．
- kuzborskij2017fast：MLJ の組版ではなく arXiv:1412.1619v3 で確認．
- busso2008iemocap（SAIL 版），dhall2012afew（著者の草稿），saerens2002adjusting（機関リポジトリの著者版）：出版社の組版ではない．
- BFtD（ACM MM 2023）は本文を読めず，抄録と後継論文の説明だけ．FtD++ は arXiv 版だけ．
- Ryumina et al. 2025（Pattern Recognition Letters）は Crossref の書誌と研究室の告知の頁だけで，論文の本文と抄録は読めていない．

### 7.4 その他

- 新しさの監査（4 節）の「先例が見つからなかった」は，挙げた検索語と，補助が使った 6 本の検索語の範囲での結果である．
- CMU-MOSEI の 3 人の評点のまとめ方，AFEW の言語は，引いた原典に記述が無い．
- RoBERTa の ICLR 2020 への投稿の採否，LoRA の ICLR 2022 での採録，Holm（1979）の書誌は確かめていない．
- 監査には補助を 3 つ使い，書誌と主張の一部（理論の 17 本，蒸留と弱いラベルの 19 本，比較相手・コーパス・エンコーダの 22 本）を分担させた．Major の根拠の引用文（Lee 2013，Grandvalet & Bengio 2004，Sindhwani et al. 2005，Busso et al. 2008，xMUDA）と，Kumar et al. 2022，Charikar et al. 2024，AdaMatch，Veilleux et al. 2021 の引用文は，監査者が補助の取得した PDF のテキストで原文を確かめ直した．Kumar et al. はこの読み直しで判定を「不支持に近い」から「一部支持」に改めた（m-2）．そのほかの補助の報告による引用文（理論の 17 本の大半，コーパスとエンコーダの記述など）は，監査者は読み直していない．


## 付録 A. 引用の棚卸し

本文（`main.tex`，`sections/*.tex`）と補足資料（`supplementary.tex`）の `\cite` は 95 箇所（本文 85，補足 10），引用キーは 81 本である（本文 78 本，補足 11 本．補足だけのものは `yolov8face`，`cao2018vggface2`，`mollahosseini2019affectnet` の 3 本）．`supp/tables` は補足の PDF で確かめ，引用は無い．
番号は本文の出現順に振った（I：序論，R：関連研究，S：設定，P：第 I 部，B：第 II 部，C：比較，L：限界，X：補足）．

| 番号 | 箇所 | 引用キー | 本文がその引用に負わせている主張 |
|---|---|---|---|
| I1 | 1 節 第 1 段落 | schuller2010crosscorpus, gideon2021addog, feng2023fewshot | 新しいコーパスに持ち込んだ感情認識器はコーパスをまたぐと精度が落ちる |
| I2 | 1 節 第 2 段落 | bendavid2010theory, blitzer2007learning, desilva2023value | 複数ドメインからの学習の理論は，ソースのプールがターゲットのラベルが少ないと得，多いと損になることを示し，ターゲットの誤差をソースの誤差・ダイバージェンス・ターゲット標本の大きさで抑える |
| I3 | 1 節 第 2 段落 | feng2023fewshot, he2025m4ser, zaidi2024enhancing | 少数のターゲットラベルでの感情認識の方法は，音声または音声＋文字のモデルを少数のラベル付き発話で適応させる |
| I4 | 1 節 第 2 段落 | lee2013pseudolabel, hinton2015distilling, lopezpaz2016unifying | 半教師あり学習と蒸留はラベル無しデータまたは強い教師を使う |
| I5 | 1 節 第 2 段落 | ahn2025multitask | 顔を含む複数モダリティを使うクロスコーパスの方法は少ない |
| R1 | 2 節 第 1 段落 | schuller2010crosscorpus | クロスコーパス評価は音声感情認識の大きな低下を示し，コーパスごとの正規化を動機づけた |
| R2 | 同 | feng2023fewshot | 自発音声の少数ショット学習（シャム網） |
| R3 | 同 | gideon2021addog | ターゲットのラベルの量を増やして評価した敵対的ドメイン汎化 |
| R4 | 同 | chen2023prefinetuning | ソースのコーパスでの事前微調整 |
| R5 | 同 | xu2024crosstask | 凍結した音声特徴での能動学習 |
| R6 | 同 | he2025m4ser | 音声＋文字のモデルをクラスあたり 5〜60 ラベルで適応 |
| R7 | 同 | zaidi2024enhancing | 言語をまたいでコーパスあたり少数のラベルで適応 |
| R8 | 同 | ahn2025multitask | クロスコーパスのマルチモーダル手法は少ない |
| R9 | 同 | gideon2021addog | いつ汎化しいつ特化するかを未解決として残した |
| R10 | 同 | feng2023fewshot | ソースの性質とソースの組み合わせの効果を未解決として残した |
| R11 | 同 | dong2026test | 音声感情認識のテスト時適応では，エントロピー最小化と擬似ラベルがほとんどの設定で失敗する |
| R12 | 2 節 第 2 段落 | okamoto1963asymptotic | 標本線形判別関数の誤判別率の漸近展開は，次元によらない $1/N$ の項と次元に比例する項を含む |
| R13 | 同 | raudys1998intrinsic | Raudys は後者を実効次元で表した |
| R14 | 同 | desilva2023value, dobriban2018highdimensional | 式 (1)（2 つのガウスの群に対する線形規則の正答率）は標準的である |
| R15 | 同 | rosenfeld2022domainadjusted | 凍結特徴＋線形ヘッドで，ドメイン調整回帰は超過リスクをターゲット平均の誤差で抑える |
| R16 | 同 | kumar2022lpft | 分布シフトの下で線形プロービングが微調整に勝つ |
| R17 | 同 | kini2021labelimbalanced | 正則化が消えると，ロジットへの加法的補正は効果を失う |
| R18 | 同 | ha2026labeled | 少数のターゲット標本での転移の有限標本解析が回帰について存在する |
| R19 | 同 | desilva2023value | 同じく 1 次元の判別について存在する |
| R20 | 2 節 第 3 段落 | hernandez2021scaling, lescao2021data | 事前学習やソースの価値はターゲットデータの実効量で表されてきた |
| R21 | 同 | wiesenfarth2020quantification | 事前分布の実効標本サイズとして表されてきた |
| R22 | 同 | hanneke2019value | 収束率の指数として表されてきた |
| R23 | 同 | raudys1998intrinsic, dobriban2018highdimensional | 推定した方向の余弦と標本サイズの古典的な関係（式 (5) はこれを逆に解いた） |
| R24 | 2 節 第 4 段落 | bendavid2010theory, blitzer2007learning | 複数ドメインからの学習の上界では，ドメイン間の距離は罰則としてしか働かない |
| R25 | 同 | tiomoko2021deciphering | プールした線形分類器の偏りは，課題ごとの切片を持つ中心化したマルチタスクのモデルについて高次元極限で解かれている |
| R26 | 同 | yang2025precise | 切片の無いプールした最小二乗について高次元極限で解かれている |
| R27 | 同 | zhu2026domain | ドメインの情報を明示的に加えるとプールに勝ちうる |
| R28 | 同 | pregibon1981logistic, koh2019accuracy | ニュートン法の 1 歩は点の群の影響を近似する |
| R29 | 同 | desilva2023value | 式 (4) は De Silva らの 1 次元の閾値のずれを拡張したもの |
| R30 | 2 節 第 5 段落 | vapnik2009new, lopezpaz2016unifying | ラベルを共有し入力モダリティが違うとき，余分なモダリティは特権情報である |
| R31 | 同 | gupta2016cross | クロスモーダル蒸留は対になったモダリティの間で教師信号を移す |
| R32 | 同 | albanie2018emotion | 感情認識で，顔の教師が音声の生徒を学習させる |
| R33 | 同 | srinivasan2022representation | 音声＋文字の融合教師が音声の生徒を学習させる |
| R34 | 同 | pendyala2025leveraging, hajavi2024audio | 音声映像または映像の教師が音声の生徒を学習させる |
| R35 | 同 | du2021improving, xue2021multimodal, rakib2025g2d, jaritz2020xmuda | 単一モダリティの教師が融合の生徒を改善し，融合の出力が単一モダリティのヘッドを学習させる |
| R36 | 同（引用なし） | R32〜R35 の全文献 | これらの研究では教師はターゲットのコーパスか第三のコーパスで学習され，または生徒は推論時に全モダリティを使う |
| R37 | 2 節 第 6 段落 | phoo2021selftraining | 固定した教師が，クラスの違う課題をまたぐ表現学習のためにラベル無しデータ全部に柔らかいラベルを付けた |
| R38 | 同 | zanella2024boosting | 固定の重みでの転導的な少数ショット分類 |
| R39 | 同 | hatano2024multimodal | モダリティ別の教師をドメインをまたいで単一モダリティに蒸留した |
| R40 | 同 | huang2022unsupervised, xie2020selftraining, hu2021leveraging, berthelot2022adamatch | クラス間で擬似ラベルを均すことはよく行われる |
| R41 | 同 | veilleux2021realistic | ラベル無し集合のクラスの偏りは，その均しを有害にしうる |
| R42 | 同 | silvarodriguez2024closer | ゼロショット分類器への 2 次の錨 |
| R43 | 同 | kuzborskij2017fast, li2018explicit | 仮説転移は重みを事前の解に縛る |
| R44 | 同 | gulrajani2021domainbed | 1 ドメイン抜きの検証はターゲットのラベル無しでハイパーパラメータを選ぶ |
| R45 | 2 節 第 7 段落 | charikar2024quantifying | 弱い教師のラベルで学んだ生徒の得は，両者の不一致で下から抑えられる |
| R46 | 同 | dong2025discrepancies, xue2025representations | 得は両者の表現の不一致と関係づけられる |
| R47 | 同 | chaloner1984optimal, chaloner1995bayesian | ガウス線形モデルでは最適計画は観測された応答によらない |
| R48 | 同 | hacohen2022active | 予算が小さいとき，無作為または典型的な選び方に勝つのは難しい |
| S1 | 3.1 節 | busso2008iemocap, zadeh2018mosei, poria2019meld, dhall2012afew, livingstone2018ravdess, cao2014cremad | 使った 6 コーパスの出典 |
| S2 | 3.2 節 | chen2022wavlm, hsu2021hubert, liu2019roberta, he2023debertav3 | 音声と文字のエンコーダ |
| S3 | 3.2 節 | ryumina2022facial, radford2021clip | 表情認識用に学習した ResNet-50，CLIP ViT-L/14 の画像エンコーダ |
| S4 | 3.2 節 | liu1989lbfgs | L-BFGS |
| S5 | 3.2 節 | kumar2022lpft | クラスあたり少数のラベルでは，微調整は事前学習の特徴を歪めうる |
| S6 | 3.4 節 | gideon2021addog | 先行するクロスコーパスの研究も，この場合（ターゲットのラベルが少ない場合）ハイパーパラメータを固定する |
| S7 | 3.6 節 | gulrajani2021domainbed | 擬似ターゲットでの選択は 1 ドメイン抜きの選択と同じ考え方である |
| P1 | 4.2 節 (A2) | kini2021labelimbalanced | 分離可能なデータで罰則が消えると，ロジットの加法的補正は効果を失う |
| P2 | 4.2 節 式 (3) | okamoto1963asymptotic, raudys1998intrinsic | $d(K)=\Delta/\sqrt{1+q/K}$ は線形判別の誤判別率の漸近展開の次元比例の項から得られる形である |
| P3 | 4.4 節 式 (4) | koh2019accuracy, pregibon1981logistic | 式 (4) は点の集合の群影響を，加えた点について書いたもの |
| P4 | 4.4 節 式 (4) | desilva2023value | 切片の経路は，$n$ 個のターゲットと $m$ 個のソースからプールした 1 次元判別の閾値のずれ $m/(n+m)$ に帰着する |
| P5 | 4.5 節 式 (5) | raudys1998intrinsic, dobriban2018highdimensional | 式 (5) は推定方向の余弦と標本サイズの古典的な関係を逆に解いたもの |
| B1 | 5.2 節 命題 1 | kimeldorf1971results, scholkopf2001generalized | 命題 1 は表現定理を $W-W_0$ に当てはめて従う |
| B2 | 5.2 節 命題 2 の後 | chaloner1984optimal, chaloner1995bayesian | ガウス線形モデルでは事後共分散が観測ラベルによらないので，最良の計画も観測ラベルによらない |
| B3 | 5.4 節 | lopezpaz2016unifying, vapnik2009new | ソースで学習した融合分類器は別の方向を知る．特権情報を使う学習と同じ |
| C1 | 6.1 節 | lee2013pseudolabel | 生徒自身の確信度の高い予測による自己学習（Lee の形） |
| C2 | 6.1 節 | grandvalet2004semisupervised | エントロピー最小化 |
| C3 | 6.1 節 | hinton2015distilling | 温度つきの蒸留 |
| C4 | 6.1 節 | lopezpaz2016unifying | 一般化蒸留 |
| C5 | 6.1 節 | blum1998combining | 共訓練 |
| C6 | 6.1 節 | sindhwani2005coregularization | 共正則化 |
| C7 | 6.1 節 | saerens2002adjusting | EM による事前確率の調整 |
| L1 | 8 節 | veilleux2021realistic | 偏りが強いと列の正規化は有害になりうる |
| X1 | 補足 S1 | yolov8face | YOLOv8n-face で顔を検出 |
| X2 | 補足 S1 | ryumina2022facial, cao2018vggface2, mollahosseini2019affectnet | Ryumina らの ResNet-50 は VGGFace2 で学習し AffectNet で微調整したもの．512 次元 |
| X3 | 補足 S1 | radford2021clip | CLIP ViT-L/14 の画像エンコーダは 1024 次元を出す |
| X4 | 補足 S1 | chen2022wavlm, hsu2021hubert, liu2019roberta, he2023debertav3 | 最終層の平均（1024 次元，768 次元） |
| X5 | 補足 S2 | okamoto1963asymptotic, raudys1998intrinsic | 標本線形判別では，誤判別率の展開の 1 次の項から $q=2(p-1)/\Delta^2$ が得られる |
