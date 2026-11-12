**検証の結果（2026-11-12，`verify_emotion.md`）**：Knyazev 2017，Savchenko 2023・2026，ECAN の数は原文と一致．CAST は本文未確認のまま，公開コードには切片を当て直して誤りを分ける処理は無い．

# 誤りを「切片側」と「向き側」に分けた研究の有無：顔・映像の感情認識（FER，動画の感情，AU）

調べた日：2026-11-11．担当：顔・映像．ほかの 3 体（言葉，一般の機械学習，声と融合）と並行して調べた．

## 要約（判定）

1. **完全一致は見つからなかった．**顔・映像の感情認識の実データで，分類器の誤りを「切片を当て直せば取り戻せる分」と「切片をどう当て直しても残る分」に分け，情報源（ターゲットの少数のラベル，ほかのデータセット，ターゲットのラベル無しデータ，ほかのモダリティの予測）ごとにどちらが減ったかを比べた研究は，約 45 本の検索語と 20 本あまりの本文の確認では見つからなかった．
2. **部分一致（切片だけを当て直した効果を 1 つの条件で測ったもの）は 3 本ある．**Knyazev ほか 2017（AFEW．テストのクラスの割合で点数を重みづけると正解率が 54.98 → 60.03），Savchenko 2026（Aff-Wild2．クラスごとの切片だけを検証集合で当て直すと顔の F1 が 38.68 → 41.40，声の F1 が 30.50 → 32.33），Savchenko 2023（AU の閾値だけを AU ごとに当て直すと F1 が 0.508 → 0.537）．どれも切片の側の効果だけを測り，残る誤り（向きの側）は測っていない．情報源は 1 つで，Knyazev はテストのラベルの割合を使った．
3. **分け方が近い研究（ラベルの割合のずれと条件つきのずれを手法の部品として分けたもの）は 2 本ある．**Li & Deng の ECAN（割合で重みづけた MMD と条件つき MMD の切り離し実験．85.24 / 83.56 / 86.49）と，Wang & Liu 2025（CAST．クラスの割合が疑似ラベルの学習に与える悪影響を「クラスごとの表現の分布」と「予測確率」の両方から分析したと抄録に書いてある．本文は読めていない）．どちらも，切片と向きに分けて誤りを測ってはいない．
4. **顔の研究で，人ごとの特徴の平均（中央値）を引く処理は古くから使われている**（Baltrušaitis 2015，OpenFace 2.0，Haber 2016）．重みを固定した線形分類器でこの処理をすると切片を当て直すのと同じ計算になるが，そう書いた論文も，その分を測った論文も見つからなかった．
5. **顔に特化した理論は見つからなかった．**関係する理論は一般の機械学習の側（ラベルシフトの補正，表現と分類器を分けて学ぶ方法，BN の統計の入れ替え）にあり，担当外として名前だけを挙げる．

---

## 1. 判定の基準

- 切片側の誤り（a）：出力の偏り，判定の閾値，クラスの事前確率（割合）のずれ．重みを固定したまま切片（バイアス項・閾値・クラスの割合）を当て直せば取り戻せる分．
- 向き側の誤り（b）：判別の向き，特徴の重み，表現のずれ．切片をどう当て直しても残る分．
- 情報源：① ターゲットの少数のラベル，② ほかのデータセットのラベル付きデータ，③ ターゲットのラベル無しデータ（正規化，平均を引く，テスト時適応，ドメイン適応，自己学習，ラベルシフトの補正），④ ほかのモダリティの予測（クロスモーダル蒸留，疑似ラベル）．
- 水準：◎ 原文の該当箇所を確かめた，○ 抄録か信頼できる要約，△ 書誌か検索の断片だけ．

## 2. 研究ごとの表

### 2.1 部分一致（切片の側の効果だけを測った）

| 研究 | 設定 | 測った量 | (a)(b) を分けたか | 情報源 | 水準 | URL |
|---|---|---|---|---|---|---|
| Knyazev, Shvetsov, Ilyina, Zuev, "Convolutional neural networks pretrained on large face recognition datasets for emotion classification from video," EmotiW 2017（arXiv 1711.04598） | AFEW 7 クラス．顔認識で学んだ CNN を FER2013 で微調整したものの集団と声の特徴．検証集合はクラスがほぼ均等で，テスト集合は Happy・Neutral・Angry が多い（表 5：テスト 653 本のうち Neutral 193，Happy 144，Angry 98） | 点数に「テストのクラスの頻度の平方根」を掛けた前後の正解率．検証 56.40 → 48.30，テスト 54.98 → 60.03（表 4） | (a) だけ．点数にクラスごとの定数を掛けるのは，対数の点数ではクラスごとの切片を足すのと同じである．残る誤りは測っていない | テストのクラスの割合（提出で返ってきた混同行列から求めた）．①〜④ のどれでもなく，オラクルの割合である | ◎（2.6 節，表 4・5） | https://arxiv.org/abs/1711.04598 |
| Savchenko, "HSEmotion team at ABAW-10"（arXiv 2603.12693）と同著者のポスター "Confidence-Gated Calibration for Video-based FER in-the-Wild" | Aff-Wild2 の表情 8 クラス．AffectNet で学んだ EmotiEffNet の埋め込みを固定し，Aff-Wild2 の学習集合で MLP を学ぶ．Generalized Logit Adjustment で最終層のクラスごとの切片 b* を，検証集合の F1 が最大になるよう [−2, 2] の格子で座標ごとに探す．初期値はクラスの事前確率 | 切片の当て直しの前後の F1／正解率．顔 38.68/47.59 → 41.40/51.62，焦点損失つきの顔 38.99/50.22 → 40.18/51.90，声（wav2vec 2.0）30.50/40.08 → 32.33/42.15（表 1） | (a) だけ．著者は効果を「クラスの不均衡を和らげた」と説明し，表現の変化は論じていない | ①（ターゲットのラベル．ただし少数ではなく検証集合全体）．同じ検証集合で切片を選び，その集合で点数を報告している．コーパス内 | ◎（3.1 節，表 1）．ポスター ◎ | https://arxiv.org/abs/2603.12693 ，https://affective-behavior-analysis-in-the-wild.github.io/10th/img/21.pdf |
| Savchenko, "EmotiEffNet Facial Features in Uni-task Emotion Recognition in Video at ABAW-5"（arXiv 2303.09162） | Aff-Wild2 の AU 検出．EmotiEffNet の特徴に MLP または LightAutoML | 閾値 0.5 と，AU ごとに検証集合で F1 を最大にする閾値の比較．MLP 0.508 → 0.537，LightAutoML 0.397 → 0.477（表 3） | (a) だけ（閾値は 2 値分類の切片にあたる） | ①（検証集合全体）．同じ検証集合で選んで報告．コーパス内 | ◎（表 3 と本文） | https://arxiv.org/abs/2303.09162 |

### 2.2 部分一致（弱い）：割合のずれと条件つきのずれを手法の部品として分けた，または定性的に述べた

| 研究 | 設定 | 測った量 | (a)(b) を分けたか | 情報源 | 水準 | URL |
|---|---|---|---|---|---|---|
| Li & Deng, "A Deeper Look at Facial Expression Dataset Bias"（ECAN．arXiv 1904.11150，IEEE TAC 2022） | 7 表情のコーパス間．ResNet 系．ソースの各クラスに α_l = P^t(l)/P^s(l)（ターゲットの割合は疑似ラベルから推定）を掛けた MMD と，クラス条件つき MMD | 切り離し実験（表 VIII，RAF-DB 2.0 → CK+ の 1 組だけ，分散なし）：重みづけだけ 85.24，条件つきだけ 83.56，両方 86.49．V 節 3) で CK+ の anger が sadness に，fear が surprise に流れる誤りを，ソースで多いクラスへの偏り（class distribution bias）に帰す | 分けていない．α は MMD の中にだけ入り，分類の損失には入らない．どちらの部品も表現を動かすので，切片の側と向きの側の分け方にはならない．測ったのは正解率だけ | ③ | ◎（ar5iv 本文の式 7〜12，表 VIII．手元の記録でも確認済み） | https://ar5iv.labs.arxiv.org/html/1904.11150 |
| Wang & Liu, "Unsupervised Cross-Domain Facial Expression Recognition via Class Adaptive Self-Training"（CAST．IEEE TAC 16(3):1618–1630, 2025．DOI 10.1109/TAFFC.2025.3529947） | 顔のコーパス間の教師なし DA と自己学習 | 抄録：クラスの割合が疑似ラベルの学習に与える悪影響を「クラスごとの表現の分布」と「予測確率」の両面から分析し，両方を合わせる手を提案した | 抄録の言い方は (b) と (a) に近い．ただし本文を読めていないので，誤りを分けて測ったかは確かめていない | ③ | ○（Semantic Scholar の抄録．本文は IEEE の有料ページで開けなかった） | https://doi.org/10.1109/TAFFC.2025.3529947 |
| Agarwal, Tsaftaris, Sevilla-Lara, McDonagh, "Why Do Vision Language Models Struggle To Recognize Human Emotions?"（arXiv 2604.15280，ECCV 2026 と記載） | VLM と MAE-DFER．MAFW（11 クラス）・DFEW（7 クラス）の均等なテスト集合 | 混同の非対称（まれなクラスが多いクラスへ流れ，逆は少ない）から「頻度の高いクラスの事前確率に寄る」と論じる．均等に並べた微調整で改善 | 定性的に (a) を論じただけで，量として分けていない．後処理の補正（logit adjustment など）は試していないと本文 2.2 節に書く | なし（微調整の学習データの並べ方） | ◎（HTML 本文．付録 S5 以降は読み切れていない） | https://arxiv.org/abs/2604.15280 |

### 2.3 該当しない（切片の側の処理を使うが効果を分けて測っていない，または正解率だけ）

| 研究 | 設定 | 測った量 | (a)(b) を分けたか | 水準 | URL |
|---|---|---|---|---|---|
| Baltrušaitis, Mahmoud, Robinson, "Cross-dataset learning and person-specific normalisation for automatic Action Unit detection," FG 2015 | AU の有無を線形 SVM（liblinear），HOG＋形状の 1606 次元．人ごとに動画の特徴の中央値を引く（ラベル無し．③ を人ごとに行う）．SEMAINE・DISFA・BP4D を合併（②） | F1．中央値を引くと選んだ AU の平均 0.33 → 0.46（表 I．効果が正の AU だけを載せたと明記）．合併で全体の平均 0.40〜0.44 → 0.49（表 II） | 分けていない．中央値を引いた特徴で学び直すので重みも変わる．重みを固定した線形分類器なら特徴から定数を引くことは切片の当て直しと同じ計算になるが，論文にその記述は無い | ◎（IV-E 節，表 I〜III） | https://www.cl.cam.ac.uk/~mmam3/pub/FG2015.pdf |
| Baltrušaitis, Zadeh, Lim, Morency, "OpenFace 2.0," FG 2018 | AU 強度．人ごとの特徴の中央値を引き，さらに人ごとに予測の下位 n パーセンタイル（n は検証集合で決める）を全予測から引く | この「予測の補正」の効果を単独で報告していない | 出力の切片だけを人ごとに当て直す処理そのものだが，測っていない | ◎（III 節の AU の段落） | https://par.nsf.gov/servlets/purl/10099458 |
| Haber, Voss, Fazel, Winograd, Wall, "A Practical Approach to Real-Time Neutral Feature Subtraction for Facial Expression Recognition," WACV 2016 | HOG・SIFT．人ごとの中立の顔の特徴を引く．中立のラベルを使うオラクルと，ラベル無しで推定する方法 | ラベル無しの推定がオラクルの得（約 8%）の約 3 分の 2 を取り戻す．テスト時のクラスの偏りに強い作り | オラクルとの比べはあるが，切片と向きには分けていない | ○（抄録） | https://mlanthology.org/wacv/2016/haber2016wacv-practical |
| Ertugrul, Cohn, Jeni, Zhang, Yin, Ji, "Crossing Domains for AU Coding," IEEE TBIOM 2020（FG 2019 版あり） | AU のコーパス間（EB+，GFT → DISFA，UNBC）．CNN と線形 SVM | S 値，AUC，F1，NA．閾値はソースの 5 分割で決め，ターゲットにはそのまま使う．AU の出現率（base rate）の違いは低下の理由の候補として定性的に挙げるだけ | 分けていない．ターゲットで閾値を当て直す実験は無い | ◎（PMC 本文） | https://pmc.ncbi.nlm.nih.gov/articles/PMC7202467 |
| Turnbull ほか, "Evaluating Test-Time Adaptation for FER Under Natural Cross-Dataset Distribution Shifts"（arXiv 2603.19994） | ViT-L/16，AffectNet・RAF-DB・FERPlus と 2 ソースの合併，TTA 8 手法 | 正解率，MMD の近さ．クラスの割合，クラスごとの正解率，混同行列は無い | 分けていない．T3A（分類器をプロトタイプに置き換える）も正解率だけ | ◎（HTML 本文） | https://arxiv.org/abs/2603.19994 |
| Suresh ほか, "Critically examining the Domain Generalizability of FER models"（arXiv 2106.15453） | 12 データセットの DG | 正解率と F1．クラスの不均衡はデータの性質として定性的に述べる | 分けていない | ◎（本文の該当語を確認） | https://arxiv.org/abs/2106.15453 |
| Sreenivas ほか, "Improved Cross-Dataset FER by Handling Data Imbalance and Feature Confusion"（DIFC），ECCV Workshops 2022 | ソースの不均衡を LDAM の余白で，ターゲットの紛らわしいクラスを余白の追加で扱う | 平均正解率 CE 60.85 → 不均衡の処理 62.76 → 両方 63.67（MCD が土台）．紛らわしいクラスとそうでないクラスの正解率を分けて報告 | 分けていない．余白は学習全体（表現と分類器の両方）を変える | ○（著者の解説ページ） | https://manogna-s.github.io/difc/ |
| Albanie, Nagrani, Vedaldi, Zisserman, ACM MM 2018（arXiv 1808.05561） | 顔の教師 → 声の生徒（④），ラベル無しの VoxCeleb | 教師のラベルが中立に偏ると報告し，均等な抜き出しを試したが有意な効果なし | 割合の側を 1 回試しただけで，誤りは分けていない | ◎（手元の記録 `docs/related-work.md`） | https://arxiv.org/abs/1808.05561 |
| Xue ほか, "The Modality Focusing Hypothesis," ICLR 2023 | RAVDESS などのクロスモーダル蒸留 | 教師の情報を「モダリティに共通の成分」と「固有の成分」に分ける | 別の分け方である．切片と向きの分け方ではない | ◎（手元の記録） | https://arxiv.org/abs/2206.06487 |
| Kang ほか, "Decoupling Representation and Classifier for Long-Tailed Recognition," ICLR 2020 を FER に当てた研究 | — | 検索したが FER への適用（分類器だけの学び直し，τ 正規化）は見つからなかった | — | △（検索の結果のみ） | https://arxiv.org/abs/1910.09217 |
| Neo & Chen, "FER-C: Benchmarking Out-of-Distribution Soft Calibration for FER"（arXiv 2312.11542） | FER の分布外での較正 | ECE などの較正の指標 | 較正の良し悪しで，切片と向きの分け方ではない | ○（検索の要約） | https://arxiv.org/abs/2312.11542 |
| "Memory Integrity of CNNs for Cross-Dataset FER"（arXiv 1905.12082） | ソースで学んだ CNN を，全結合層だけ，最後の畳み込み層＋全結合層，学び直し，合併の 4 通りで適応 | 正解率と忘れの度合い．合併が最良と結論 | 分けていない（最終層だけの学び直しは比べるが，切片だけの当て直しは無い） | ○（抄録） | https://arxiv.org/abs/1905.12082 |
| Chu, De la Torre, Cohn, Selective Transfer Machine（CVPR 2013，TPAMI 2017），Zen/Sangineto ほか（ACM MM 2014，IEEE TMM 2016） | 人ごとの個別化．STM はラベル無しのテストの人に合わせて学習例を重みづけ，Zen らは人ごとの分類器のパラメータをラベル無しの分布から回帰で予測 | 正解率・F1 | 分けていない | ○（抄録） | https://openaccess.thecvf.com/content_cvpr_2013/html/Chu_Selective_Transfer_Machine_2013_CVPR_paper.html |
| SFDA-PFT（ICLR 2026，arXiv 2508.09202） | 分類器を固定し，特徴を変換する小さな網を中立のターゲット画像で適応 | 正解率 | 分けていない（分類器を固定して特徴の側を直す構成だが，測っていない） | ○（検索の要約） | https://arxiv.org/abs/2508.09202 |
| Jeni, Cohn, De la Torre, "Facing Imbalanced Data," ACII 2013．Hinduja & Canavan（arXiv 2010.07982，Pattern Recognition Letters 2024） | AU の評価指標とクラスの偏り | ROC AUC 以外の指標は偏りで値が動く．F1-binary は出現率と閾値に敏感 | 指標の性質の研究である．閾値に依らない AUC と閾値に依る F1 を並べれば (a) の分を見積もれるが，情報源ごとの比べには使っていない | ○（抄録と二次資料） | https://www.ri.cmu.edu/?p=119675 ，https://arxiv.org/abs/2010.07982 |

### 2.4 関係する理論（顔に特化したものは無い．一般の機械学習の担当の範囲）

- Saerens ほか 2002 の EM による事前確率の推定，Lipton ほか 2018 の BBSE，Alexandari ほか 2020（較正してから最尤推定），Sipka ほか WACV 2022 の総説：ラベルシフト（クラスの割合だけがずれる）の下では，出力の切片を当て直せば最適な分類器に戻ると示す．○（検索の要約）．https://arxiv.org/abs/1802.03916 ，https://arxiv.org/abs/1901.06852 ，https://openaccess.thecvf.com/content/WACV2022/papers/Sipka_The_Hitchhikers_Guide_to_Prior-Shift_Adaptation_WACV_2022_paper.pdf
- Li ほか 2016（AdaBN）：「ラベルに関する知識は各層の重みに，ドメインに関する知識は BN の統計にある」という仮説を置く．③ の正規化が直すものについての仮説として近い．FER に AdaBN を当てて分析した研究は見つからなかった．○．https://arxiv.org/abs/1603.04779
- Kang ほか 2020：表現の学習と分類器の学習を分け，分類器だけを当て直すと長い裾のデータで大きく上がると示す．FER での適用は見つからなかった．○．

## 3. 検索語の記録

すべて英語．WebSearch で行い，必要な論文は WebFetch か PDF の文字起こしで本文を読んだ．

| # | 検索語 | 見つかったもの | 無かったもの |
|---|---|---|---|
| 1 | facial expression recognition cross-dataset label distribution shift class prior correction | Turnbull 2026，ECAN，MDPI の動的なラベル重み，AAAI の開集合 FER | 事前確率の補正（EM，logit adjustment）を FER で使った研究 |
| 2 | cross-domain facial expression recognition class-conditional shift versus label shift analysis | ECAN，LA-CMFER，MDPI 2023 | ラベルシフトと条件つきシフトを分けて量った FER の研究 |
| 3 | action unit detection cross-dataset person-specific output calibration normalisation | Baltrušaitis 2015，DeepFN | — |
| 4 | cross-domain action unit detection base rates threshold domains learning approaches measures | Ertugrul 2019/2020，Bilalpur 2026 | ターゲットで閾値を当て直す実験 |
| 5 | IEEE Transactions on Affective Computing 2025 cross-domain facial expression recognition class distribution mismatch pseudo-label Wang Liu | 半教師あり FER の分布の不一致（IEEE Access 2023） | CAST の本文 |
| 6 | logit adjustment facial expression recognition imbalanced cross-domain | DIFC（ECCVW 2022），Zhang NeurIPS 2023（コーパス内の不均衡） | logit adjustment をコーパス間の FER に使った研究 |
| 7 | test-time adaptation facial expression recognition label shift batch normalization statistics analysis | TempT，FIESTA，TTA-CaP，Turnbull | FER の TTA でラベルシフトや BN の統計を分析した研究 |
| 8 | facial expression recognition classifier recalibration last layer retraining domain shift | FIESTA，FER-C，Savchenko ABAW-10 のポスター | FER で最終層だけを学び直して分析した研究 |
| 9 | ABAW action unit detection threshold optimization per-AU validation set F1 improvement ablation | EmotiEffNet ABAW-5（閾値だけの効果），AUD-TGN，ABAW-5 の多モーダル | コーパス間での閾値だけの効果 |
| 10 | HSEmotion team ABAW 10 competition facial expression recognition arXiv class bias offsets frozen embeddings | Savchenko ABAW-10（arXiv 2603.12693） | — |
| 11 | "label shift" facial expression recognition prior estimation EM Saerens target class proportions | 一般のラベルシフトの論文（MLLS，Sipka WACV 2022） | FER への適用 |
| 12 | facial expression dataset bias analysis class prior versus appearance decomposition cross-dataset accuracy drop | ECAN，Face2Exp，Deep FER の総説 | 事前確率と見た目の分解 |
| 13 | cross-corpus facial expression recognition bias term adaptation only linear classifier intercept | Suresh 2021，STM，ECAN | 切片だけを適応する FER の研究 |
| 14 | FER calibration benchmark out-of-distribution expected calibration error temperature scaling facial expression | FER-C，MaxEnt 損失，2609.17130 | — |
| 15 | few-shot target labels cross-dataset facial expression recognition fine-tune only classifier layer versus full network | Memory Integrity（1905.12082），Ciubotaru 2019 | 少数のターゲットラベルで最終層と切片を比べた研究 |
| 16 | cross-modal distillation audio-visual emotion what is transferred analysis teacher student bias class prior | MFH（Xue 2023），C²KD，LiSER，XA-CMD | 教師の事前確率の偏りが生徒に移る量を測った研究 |
| 17 | EmotiW AFEW audio-video emotion recognition class prior adjustment test distribution bias correction ablation | Knyazev 2017（AFEW で割合の補正），Kollias 1809.04359 | 検証集合だけで割合を補正したきれいな切り離し実験 |
| 18 | dynamic facial expression recognition cross-dataset DFEW FERV39k MAFW label distribution shift analysis | FERV39k，S4D，DFEW の不均衡の記述 | 動画 FER のデータセット間のラベルの割合のずれの分析 |
| 19 | CLIP linear probe facial expression few-shot bias prior zero-shot calibration class prior emotion | CLIPER，EmoCapCLIP，CLAP，LP++ | CLIP の線形プローブに事前確率の補正を当てた FER の研究 |
| 20 | person-specific normalization facial expression recognition subtract neutral mean features unlabeled target analysis what it corrects | Haber 2016，Jeni の個人の平均形状，DeepFN | 平均を引く処理が直すものを切片と向きで分析した研究 |
| 21 | Haber "A Practical Approach to Real-Time Neutral Feature Subtraction for Facial Expression Recognition" pdf | 書誌と抄録 | 本文 |
| 22 | OpenFace 2.0 action unit person-specific prediction correction shift lowest percentile neutral calibration | OpenFace 2.0 の本文 | 予測の補正の効果の数値 |
| 23 | cross-domain AU detection decision threshold recalibration on target domain recovers performance base rate | Ertugrul，Bilalpur，顔以外の 2608.16852 | AU で閾値を当て直した効果 |
| 24 | facial action unit detection skew imbalanced data performance metrics recommendations Jeni Cohn De la Torre | Jeni 2013，Girard 2015 | — |
| 25 | Improved cross-dataset facial expression recognition handling data imbalance feature confusion ECCV workshop 2022 class imbalance module ablation | DIFC，POCAN | — |
| 26 | unsupervised domain adaptation facial expression recognition label proportion mismatch target prior estimation imbalanced target | GAN を使う UDA，CluP，AU で導く UDA | ターゲットの事前確率を推定する FER の UDA |
| 27 | source-free domain adaptation facial expression recognition cluster-level pseudo-labelling class prior analysis | CluP（BMVC 2022） | 事前確率の分析 |
| 28 | AVEC RECOLA valence arousal post-processing bias correction scaling time shift gain ablation video features | 2209.09068，1906.10623（前処理の効果） | 映像特徴で偏りの補正だけの効果を分けた表（連続値の回帰でコーパス内） |
| 29 | 10.1109/TAFFC.2025.3529947 | 無関係の DOI のみ（Crossref と Semantic Scholar で題名と抄録を取得） | — |
| 30 | "Unsupervised Cross-Domain Facial Expression Recognition via Class Adaptive Self-Training" | DBLP の書誌，CWCST（TIP 2023） | 本文 |
| 31 | CAST class-adaptive self-training cross-domain facial expression recognition implicit class margins imbalanced class distributions predicted probabilities | 同上 | 本文 |
| 32 | facial expression teacher pseudo-labels biased toward neutral class distillation speech student class rebalancing effect | IJCAI 2025 の不均衡な半教師あり FER，LiSER | 顔の教師の中立への偏りが声の生徒に移る量 |
| 33 | audio-visual emotion recognition knowledge distillation analysis logit bias teacher prior mismatch modality gap per-class | Aslam CVPRW 2023，BioKD，2511.15741 | 蒸留での教師と生徒の事前確率のずれの分析 |
| 34 | personalized facial action unit detection adapting only the bias offset of classifier per subject few labels | STM，Co-Incrementation 2023 | 人ごとに切片だけを当て直す AU 検出 |
| 35 | transductive parameter transfer personalized facial expression analysis hyperplane parameters predicted from unlabeled data | Sangineto 2014，Zen 2016 | 切片と重みを分けた分析 |
| 36 | decoupling representation and classifier long-tailed facial expression recognition classifier re-training tau-normalization | Kang 2020 | FER への適用 |
| 37 | vision-language model zero-shot facial expression recognition prediction bias toward a class calibration debiasing logits | "Reading Smiles"（2506.19079），2405.19100 | CLIP 型の FER で出力の切片を補正した研究 |
| 38 | multimodal large language model emotion recognition video label bias neutral calibration contextual calibration prior correction | 2602.20624（中立への偏り），2604.27345（言葉．担当外），2604.15280 | 動画 MLLM での文脈較正 |
| 39 | AdaBN batch normalization statistics re-estimation cross-dataset facial expression recognition target domain | AdaBN，TempT | FER への AdaBN の適用と分析 |
| 40 | cross-corpus audiovisual emotion recognition corpus-level feature normalization video features analysis per-corpus z-normalization | 声のコーパス正規化（Schuller 2010 ほか） | 映像特徴のコーパスごとの正規化の分析 |
| 41 | action unit detection domain adaptation few labeled target frames fine-tune final layer versus threshold adjustment comparison | 1909.12158（メタ学習），2503.08974 | 最終層の学び直しと閾値の当て直しの比べ |
| 42 | frozen face embeddings linear classifier cross-dataset expression recognition what transfers features versus classifier head analysis | 2108.03427，1812.06387，SFDA-PFT | 固定した特徴と分類器のどちらが移るかを分けた研究 |
| 43 | black box shift estimation BBSE emotion recognition video label shift experiment | BBSE，Garg 2020，ACL Findings 2024 の多モーダル DA | 感情・動画での BBSE の実験 |
| 44 | pain detection facial expression cross-dataset decision threshold operating point recalibration target AUC preserved F1 drops | Othman 2019（BioVid と X-ITE） | 閾値を当て直した効果 |
| 45 | long-tailed dynamic facial expression recognition logit adjustment post-hoc DFEW MAFW balanced softmax | MICACL（2509.04344） | 動画 FER での事後の logit adjustment |
| 46 | "prior shift" OR "class prior shift" facial expression OR "emotion recognition" adaptation unlabeled test | LA-CMFER，FIESTA，Face2Exp | "prior shift" を FER で使った研究 |
| 47 | facial expression recognition domain shift accuracy recovered by retraining only the softmax layer on target features frozen backbone analysis | SFDA-PFT，FaceNet2ExpNet，Memory Integrity | 最終層だけの学び直しで取り戻せる分を測った FER の研究 |
| 48 | Hinduja Canavan impact of action unit occurrence distribution on detection F1 class imbalance threshold | Hinduja 2020，2024 | — |

本文を読んだもの（◎）：Knyazev 2017，Savchenko 2026（arXiv HTML とポスター），Savchenko 2023，Baltrušaitis 2015，OpenFace 2.0，Ertugrul 2020，Turnbull 2026，Suresh 2021，ECAN，Agarwal 2026．

## 4. 見つからなかったこと

- 顔・映像の感情認識で，同じ分類器について「切片だけを当て直した点数」と「重みも学び直した点数」の差を，情報源ごとに比べた研究は，今回の検索では見つからなかった．
- コーパス間（データセット間）の FER で，オラクルの事前確率や閾値の補正で取り戻せる分を報告した研究は Knyazev 2017（AFEW のテストの割合）の 1 本だけだった．Savchenko の 2 本はコーパス内（Aff-Wild2）である．
- ターゲットの少数のラベル（クラスあたり K 件）が，顔の分類器の切片と向きのどちらを直すかを調べた研究は見つからなかった．
- テスト時適応（TENT，SHOT，T3A など）や BN の統計の入れ替えが，FER の分類器の切片と向きのどちらを直したかを分析した研究は見つからなかった．Turnbull 2026 は正解率だけを報告する．
- 人ごとの特徴の中央値（中立の顔）を引く処理は顔の研究で広く使われるが，重みを固定した線形分類器では切片の当て直しと同じ計算になることを書いた論文も，その分を測った論文も見つからなかった．
- 顔の教師の予測（④）が，生徒の切片と向きのどちらを動かしたかを測った研究は見つからなかった．Albanie 2018 は教師の中立への偏りを報告し，均等な抜き出しで効果が無かったと書くだけである．
- logit adjustment，BBSE，EM による事前確率の推定を，コーパス間の FER や動画の感情認識に当てて分析した研究は見つからなかった．
- 顔に特化して，誤りを切片と向きに分ける理論は見つからなかった．
- 読めなかった本文：CAST（Wang & Liu 2025），Haber 2016，Zen 2016，DIFC の本文（解説ページは読んだ），Agarwal 2026 の付録 S5 以降．CAST は抄録の言い方が最も近いので，本文を入手して確かめる価値がある．

## 5. 狙うべき差分（顔の欄から言えること）

- 先行研究の部分一致は，切片の側の効果（クラスの割合の補正，閾値の当て直し）を 1 つの条件で測ったものに限られる．切片を当て直しても残る誤り（向きの側）を同じ物差しで測り，情報源ごとに比べた研究は無い．MER-DA が同じ分類器について「切片だけを当て直した UAR」と「元の UAR」を情報源ごとに出せるなら，顔の欄ではこれが新しい．
- 部分一致の 3 本のうち 2 本は，切片を選んだ検証集合で点数を報告している．Knyazev はテストのラベルの割合を使った．切片を選ぶデータと評価のデータを分けて測ることは，顔の欄では先行研究に無い手続きである．
- 顔の研究で使われてきた「人ごとに特徴の平均を引く」処理が，固定した線形分類器では切片の当て直しにあたることを示し，その分を測ることは，顔の欄で書かれていない．これは確かめた事実ではなく，線形分類器の式からの推測である（特徴から定数ベクトル m を引くと，w·(x − m) + b = w·x + (b − w·m) となり，切片だけが変わる）．
