# 提案手法の部品（モダリティをまたぐ蒸留，特権モダリティ，クラス均衡した弱ラベル，出発点への L2 罰則つきの少数ショット学習，仮のターゲットでの設定の選択）の先例（2018〜2026 年．2026-10-27 の調べ）

この調べは，依頼文の「既知」の文献と，`docs/related-work.md` 1.8〜1.12 節に既に載っている文献（Srinivasan 2022，LiSER 2025，UMT，MKE，SLA，CBST，CReST，FlexMatch，DebiasPL，KgCoOp など）を既知として扱い，見落としだけを探した．調べ方は WebSearch，WebFetch（arXiv の抄録・HTML 本文，ar5iv 本文），OpenAlex API（題名・書誌・抄録）である．Semantic Scholar API は 429（回数制限）で使えず，OpenAlex も途中から 429 になったので，Gupta 2016 と Lopez-Paz 2016 の被引用の網羅的なたどりはできなかった（各節の「Gaps」に書く）．

確かめた水準の記号：◎＝本文（HTML か全文）を読んだ，○＝抄録（arXiv・学会の頁・OpenAlex の抄録）を読んだ，△＝検索結果の抜き書きか書誌の記録だけ，背＝今回は確かめていない背景知識（論文に引くなら原典を開くこと）．

## Q1. ソースで学んだ（ターゲットで学び直さない）多モーダルの先生から，ラベルの少ない新しいドメインの単モーダルの生徒へ蒸留する研究はあるか（部品 4 の骨格）

### Takeaway
感情認識の外では，ほぼ同じ骨格の先例が 3 系統見つかった．(a) 行動認識の MM-CDFSL（ECCV 2024）は，モダリティごとの先生をソースのラベルとターゲットのラベル無しデータで作って凍結し，ターゲットのラベル無しデータで RGB の生徒に特徴を蒸留してから，K-shot の支持集合で頭を学ぶ．(b) 3 次元の分割の xMUDA／xMoSSDA は，2 つのモダリティの予測をターゲットのラベル無しデータ上で互いにまねさせ，ターゲットに少数のラベルがある設定（半教師ありの DA）も含む．(c) 少数ショット転移の STARTUP は，ソースだけで学んだ先生（学び直さない）の確率を温度 1 でターゲットのラベル無しデータ全体に付けて生徒を学ぶ．感情認識で「ソースのコーパスだけで学んだ融合の分類器を，ターゲットのコーパスのラベル無し発話に当てて，単モーダルの生徒の弱いラベルにする」研究は，今回も見つからなかった．

### Cited Findings

| 文献 | 何をするか | 本研究（部品 4）との違い | 水準 | 出典 |
|---|---|---|---|---|
| Hatano, Hachiuma, Fujii, Saito, "Multimodal Cross-Domain Few-Shot Learning for Egocentric Action Recognition" (MM-CDFSL), ECCV 2024, arXiv:2405.19917 | RGB・光学流・手の姿勢のモダリティごとに VideoMAE の先生を作る．先生はソースのラベル（分類の損失）とソース・ターゲットのラベル無しデータ（再構成）で学び，「蒸留の段では先生の重みを凍結する」．生徒（RGB）はターゲットのラベル無しの RGB だけで，各モダリティの先生の特徴を L2 で回帰する．最後に N-way K-shot の支持集合で分類の頭を学ぶ | 骨格（多モーダルの先生 → 単モーダルの生徒，ターゲットのラベル無しデータで蒸留，少数ショットで仕上げ）が同じ．違い：先生は融合ではなくモダリティごと，先生はターゲットのラベル無しデータで再構成を学んでいる，蒸留は確率ではなく特徴，クラス均衡なし，出発点への罰則なし，ソースとターゲットでクラスが違う（少数ショットの新しいクラス）．釣り合いの重みは EPIC-Kitchens の 5-shot で掃いた値を報告に使っている（別の検証は無い） | ◎（arXiv HTML 本文．掲載は ECVA の頁で確認） | [arXiv HTML](https://arxiv.org/html/2405.19917v1)，[ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04830.pdf) |
| Guo, Jiao, Guo, Song, "Multimodal Helps Unimodal: Frequency-Aware Distillation for Cross-domain Few-shot Egocentric Action Recognition," Information Fusion, 2026 | 題名から，MM-CDFSL と同じ設定（多モーダル → 単モーダル，ドメインをまたぐ少数ショット）の続き | 中身は未確認 | △（OpenAlex の書誌のみ．抄録なし） | [DOI](https://doi.org/10.1016/j.inffus.2026.104636) |
| Jaritz, Vu, de Charette, Wirbel, Pérez, "xMUDA: Cross-Modal Unsupervised Domain Adaptation for 3D Semantic Segmentation," CVPR 2020；"Cross-modal Learning for Domain Adaptation in 3D Semantic Segmentation," TPAMI 2022（arXiv:2101.07253） | 2D 画像と 3D 点群の 2 つの頭の予測を，ターゲットのラベル無しデータ上で互いにまねさせる（mutual mimicking）．分割の損失とまねの損失を別の頭に分け，強いモダリティが弱いモダリティの誤りを取り込まないようにする．雑誌版は「半教師ありの DA の設定」（ターゲットに少数のラベル）を加えた | モダリティをまたいでターゲット上で予測を合わせる DA の代表．違い：2 つのモダリティを同時に学ぶ（凍結したソースの先生ではない），深層網，予測の一致であって融合の先生ではない．xMoSSDA という名前とターゲットのラベルの量は検索の要約でしか確かめていない | ○（CVPR 版は OpenAlex の抄録，雑誌版は arXiv の抄録と comments 欄の「TPAMI 2022」）．xMoSSDA は △ | [CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Jaritz_xMUDA_Cross-Modal_Unsupervised_Domain_Adaptation_for_3D_Semantic_Segmentation_CVPR_2020_paper.html)，[arXiv](https://arxiv.org/abs/2101.07253) |
| xMUDA の続き：Cross-Domain and Cross-Modal Knowledge Distillation in Domain Adaptation for 3D Semantic Segmentation (ACM MM 2022)；Cross-modal Unsupervised Domain Adaptation for 3D Semantic Segmentation via Bidirectional Fusion-then-Distillation (ACM MM 2023)；Fusion-Then-Distillation (IEEE TCSVT 2025) | 題名から，融合した予測を各モダリティに蒸留し直す（fusion-then-distillation）形が DA で使われている | 部品 5（融合と単モーダルの往復）に近い可能性がある．中身は未確認 | △（OpenAlex の題名のみ） | [MM 2022](https://doi.org/10.1145/3503161.3547990)，[MM 2023](https://doi.org/10.1145/3581783.3612013)，[TCSVT](https://doi.org/10.1109/tcsvt.2025.3545971) |
| Phoo & Hariharan, "Self-training for Few-shot Transfer Across Extreme Task Differences" (STARTUP), arXiv:2010.07734（ICLR 2021 は要確認） | 先生はソース（miniImageNet か ImageNet）のラベルだけで学び，ターゲットでは学び直さない．先生の確率を「温度 1」でターゲットのラベル無し画像全体に付け，生徒をソースの交差エントロピー＋先生への KL＋SimCLR の和（重みなし）で学ぶ．評価は表現を凍結し，支持集合で線形分類器を学ぶ | 「学び直さないソースの先生の確率を，ターゲットのラベル無しデータ全体の弱いラベルにする」点が部品 4 と同じ．違い：ソースとターゲットでクラスが違うので，弱いラベルは表現を学ぶためだけに使い，分類器はラベルだけで学ぶ．単モーダル．クラス均衡なし．総重量の選択なし（学習率だけをラベル無しデータの 10% の検証損失で選ぶ） | ◎（ar5iv 本文） | [ar5iv](https://ar5iv.labs.arxiv.org/html/2010.07734) |
| Sarafianos, Vrigkas, Kakadiaris, "Adaptive SVM+: Learning with Privileged Information for Domain Adaptation," ICCV Workshops 2017 | A-SVM（ソースの分類器への罰則で適応する SVM）と SVM+（学習時だけ使える特権情報）を 1 つの目的関数に組む．Animals with Attributes と INTERACT | 「仮説転移（線形）＋特権情報＋DA」の組み合わせの先例（2017 年なので期間の境目）．違い：特権情報は SVM+ の補正関数として使い，蒸留の弱いラベルではない．ターゲットのラベル無しデータを使うかは抄録に無い | ○（OpenAlex の抄録） | [DOI](https://doi.org/10.1109/iccvw.2017.313) |
| Li, Sun, Yan, "Domain adaptive twin support vector machine learning using privileged information," Neurocomputing 2021 | 題名から，特権情報つきの DA を双子 SVM で行う | 中身は未確認 | △（書誌のみ） | [DOI](https://doi.org/10.1016/j.neucom.2021.10.069) |
| Nair ほか, "Let Me Show You How It's Done – Cross-modal Knowledge Distillation as Pretext Task for Semantic Segmentation," CVPRW (EarthVision) 2024 | 光学画像のラベルで学んだ先生が，位置を合わせた PolSAR 画像に疑似ラベルを付け，SAR の生徒を学ぶ（対になったラベル無しデータだけを使う） | リモートセンシングでの「別モダリティの先生の疑似ラベル → 単モーダルの生徒」．DA ではなく事前学習として扱う | △（検索の要約） | [CVF PDF](https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/papers/Nair_Let_Me_Show_You_How_Its_Done_-_Cross-modal_Knowledge_CVPRW_2024_paper.pdf) |
| Pande ほか, "An Adversarial Approach to Discriminative Modality Distillation for Remote Sensing Image Classification," ICCVW 2019 | 多モーダル（高分光など）の先生から，欠けるモダリティの特徴を生徒に再現させる（特権情報） | ラベル無しのターゲットのドメインは扱わない | △（検索の要約） | [CVF](https://openaccess.thecvf.com/content_ICCVW_2019/html/CROMOL/Pande_An_Adversarial_Approach_to_Discriminative_Modality_Distillation_for_Remote_Sensing_ICCVW_2019_paper.html) |
| JDCNet: Confidence-Gated Privileged-Modality Distillation for Cost-Preserving X-ray Inference, arXiv:2603.29167 (2026) | CT を学習時だけ使い，X 線だけで推論する分類器に蒸留する（医療） | 新しい施設や少数ラベルの設定は検索の要約に無い | △ | [arXiv](https://arxiv.org/pdf/2603.29167) |
| PGUDA: Pressure-Guided Unsupervised Domain Adaptation with Cross-Modal Knowledge Distillation for sEMG-Based Gesture Recognition, arXiv:2606.31349 (2026) | 題名から，圧力センサーを先生にした筋電の UDA | 装着型センサーでの同じ骨格の候補．中身は未確認 | △（書誌のみ） | [DOI](https://doi.org/10.48550/arxiv.2606.31349) |
| Unsupervised domain adaptation for lip reading based on cross-modal knowledge distillation, EURASIP JASM 2021 | 題名から，音声の先生による読唇の UDA | 中身は未確認 | △（書誌のみ） | [DOI](https://doi.org/10.1186/s13636-021-00232-5) |

- 医療画像で「cross-modality domain adaptation」と呼ばれる研究（例：MR から CT への適応．IEEE TMI 2023 の [Structure-Aware Framework ... via Frequency and Spatial Knowledge Distillation](https://doi.org/10.1109/tmi.2023.3318006)）は，ソースとターゲットが別のモダリティである設定で，本研究の「同じ発話に複数のモダリティがある」設定とは違う（△，題名のみ）．検索語が重なるので，関連研究で混同しない．
- 背（今回は確かめていない）：Garcia, Morerio, Murino の modality distillation（ECCV 2018，TPAMI 2019），Vu ほか DADA（ICCV 2019．深度をソースの特権情報にした DA），Lee ほか SPIGAN（ICLR 2019），Afouras, Chung, Zisserman "ASR is all you need"（ICASSP 2020．音声認識の先生の疑似ラベルで読唇の生徒を学ぶ），Munro & Damen MM-SADA（CVPR 2020）．

### Inferences
- 「学び直さない先生の確率を，ターゲットのラベル無しデータ全体に付ける」こと自体は STARTUP にあり，「多モーダルの先生 → 単モーダルの生徒をドメインをまたぐ少数ショットで」は MM-CDFSL にある．したがって部品 4 の新しさを「モダリティをまたぐ蒸留を少数ラベルの新しいドメインで初めて行った」と書くことはできない．
- 残る違いは，(i) 先生が融合（音声＋顔＋言葉）の線形分類器で，ターゲットのデータを一切使わずに作られていること（MM-CDFSL の先生はターゲットのラベル無しデータで再構成を学ぶ），(ii) 感情のラベル集合がソースとターゲットで同じなので，弱いラベルを分類器そのものの学習に使えること（STARTUP は表現にしか使えない），(iii) 凍結した特徴の上の線形の生徒なので，弱いラベルの効き方を解析できること，の 3 点である．
- xMUDA は「ターゲット上でモダリティ間の予測を合わせる」ので，比較相手としては本研究の共正則化（Sindhwani 型）と同じ系統に入る．関連研究では xMUDA を共正則化の深層版として共正則化と並べて引くのが妥当である．

### Gaps
- Gupta 2016 と Lopez-Paz 2016 を引く 2022〜2026 年の論文のうち few-shot・domain adaptation を含むものを，被引用のたどりで網羅することは，Semantic Scholar と OpenAlex の回数制限で完了しなかった．
- Guo ほか 2026（Information Fusion）と xMUDA の続き（fusion-then-distillation の 3 本）は書誌しか取れていない．部品 5 に近い可能性があるので，引用の前に本文を読むこと．

## Q2. 少数のターゲットラベル＋出発点への L2 罰則（仮説転移・L2-SP・線形プローブ）＋疑似ラベル／弱ラベルを組み合わせた研究はあるか（部品 2＋4）

### Takeaway
視覚と言語の模型（CLIP）の少数ショット適応で，この組み合わせに近い先例が見つかった．CLAP（CVPR 2024）は，テキストから作ったゼロショットの分類器を出発点にした線形プローブを，出発点への L2 罰則つきで少数ラベルで学ぶ（罰則の強さはクラスごと）．TransCLIP（NeurIPS 2024）は，ゼロショットの予測（テキスト側の先生．学び直さない）への KL 罰則を，ラベル無しのテスト標本全体にかけ，少数ラベルの項と一緒に解く．ただし両者とも，出発点と先生は同じ模型のテキスト側であり，ソースのドメインで学んだ分類器ではない．感情認識では，少数のターゲットラベル＋ラベル無しの発話の硬い・柔らかい疑似ラベルの組み合わせが Agarla ほか（ESWA）にある．

### Cited Findings

| 文献 | 何をするか | 本研究との違い | 水準 | 出典 |
|---|---|---|---|---|
| Silva-Rodríguez, Hajimiri, Ben Ayed, Dolz, "A Closer Look at the Few-Shot Adaptation of Large Vision-Language Models" (CLAP), CVPR 2024, arXiv:2312.12730 | 線形プローブの重みを CLIP のゼロショットの原型（テキストの埋め込み）で初期化し，支持集合の交差エントロピー＋Σ_c λ_c‖t_c − w_c‖² を最小にする（式 7）．λ_c は拡張ラグランジュ法で支持集合だけから決め，初期値はそのクラスの支持標本のゼロショット確率の平均．16-shot の ImageNet で学び ImageNetV2・Sketch・A・R で測ると，ゼロショットより ResNet-50 で +2.30，ViT-B/16 で +2.87．罰則なしの初期化だけ（ZS-LP）はゼロショットを下回る | 部品 2 と同じ「出発点への L2 罰則つきの線形の少数ショット学習」で，出発点は別のモダリティ（テキスト）から来る．違い：罰則の強さをクラスごとに分ける（本研究は重みと切片で分ける），ラベル無しデータは使わない，出発点はソースのドメインで学んだ分類器ではない．切片は記述が無い | ◎（arXiv HTML 本文の前半 10 万字） | [arXiv HTML](https://arxiv.org/html/2312.12730)，[DOI](https://doi.org/10.1109/cvpr52733.2024.02235) |
| Zanella, Gérin, Ben Ayed, "Boosting Vision-Language Models with Transduction" (TransCLIP), NeurIPS 2024, arXiv:2406.01837 | ラベル無しのテスト標本全体の割り当て z について，混合ガウスの尤度＋ラプラシアンの項＋ゼロショットの予測 ŷ への KL(z‖ŷ) を最小にする．少数ショットでは支持標本の項（重み γ/|S|）を足す．模型は学び直さない．λ はゼロショットで 1，少数ショットで 0.5 に固定し，全データセットで同じ | 部品 4 の「学び直さない別モダリティの先生の確率を，ラベル無しのターゲット全体に罰則として付け，少数ラベルと一緒に解く」に最も近い．違い：分類器は線形ではなく混合ガウスの割り当て，クラスの総量の制約は明示されない（「balanced mixture」とだけ書く），KL の重みは固定でラベル予算ごとに選ばない，ドメインのずれは扱わない | ◎（arXiv HTML 本文の前半 10 万字） | [arXiv HTML](https://arxiv.org/html/2406.01837)，[DOI](https://doi.org/10.52202/079017-1988) |
| Menghini, Delworth, Bach, "Enhancing CLIP with CLIP: Exploring Pseudolabeling for Limited-Label Prompt Tuning," NeurIPS 2023 | ゼロショットの疑似ラベルを使えば，半教師あり・転導的ゼロショット・教師なしの 3 つが同じ損失の最適化として書けると述べる．疑似ラベルを繰り返し直すプロンプト学習で，半教師ありで 19.5 点上がる．通常の疑似ラベルは質の良いクラスへの偏りを強めるが，プロンプト学習ではクラスごとの正解率が均等になる | 「少数ラベル＋学び直さない先生の疑似ラベル」の統一の見方の先例．違い：プロンプトの学習，ドメインのずれ・出発点への罰則なし | ○（OpenAlex の抄録） | [DOI](https://doi.org/10.52202/075280-2665) |
| Agarla, Bianco, Celona, Napoletano, "Semi-supervised cross-lingual speech emotion recognition," Expert Systems with Applications（OpenAlex は 2023，検索の要約は 2024），arXiv:2207.06767 | 新しい言語（ターゲット）にラベルが少ししかないとき，Transformer をラベル無しの発話の疑似ラベル（硬い版と柔らかい版）で適応する．話者を分けた評価，5 言語 | 感情認識での「少数のターゲットラベル＋疑似ラベル」の直接の先例．違い：深層網の微調整，先生は同じモダリティ（音声），出発点への罰則とクラス均衡は抄録に無い．ラベルの本数は抄録に無い | ○（OpenAlex と arXiv の抄録） | [DOI](https://doi.org/10.1016/j.eswa.2023.121368)，[arXiv](https://arxiv.org/abs/2207.06767) |
| Berthelot ほか, "AdaMatch: A Unified Approach to Semi-Supervised Learning and Domain Adaptation," arXiv:2106.04732（ICLR 2022 は要確認） | UDA・半教師あり学習・半教師ありの DA を 1 つの方法で扱う．DomainNet でターゲットのラベルを 1 クラス 1 本足すと +6.1，5 本で +13.6．τ=0.9 と uratio=3 を全実験で固定 | 「ソースのラベル＋ターゲットの K 本＋ターゲットのラベル無しの疑似ラベル」の標準形．出発点への罰則はない（ソースとターゲットを一緒に学ぶ） | ◎（ar5iv 本文の前半 10 万字） | [ar5iv](https://ar5iv.labs.arxiv.org/html/2106.04732) |
| Zanella & Ben Ayed, "Low-Rank Few-Shot Adaptation of Vision-Language Models" (CLIP-LoRA), CVPRW 2024 | 全データセット・全ショット数で同じハイパーパラメータを使って，既存のプロンプト・アダプタ法を上回る | 設定を固定する方針の例（Q4 にも関係） | ○（OpenAlex の抄録） | [DOI](https://doi.org/10.1109/cvprw63382.2024.00166) |

- 背：MME（Saito ほか ICCV 2019），DECOTA（Yang ほか ICCV 2021．UDA と半教師ありの 2 つの分類器の共訓練），Motiian ほか FADA（NeurIPS 2017），LP-FT（Kumar ほか ICLR 2022），SHOT++（Liang ほか TPAMI 2021）は今回確かめていない．

### Inferences
- CLAP は「別のモダリティから来た分類器を出発点にし，L2 罰則で寄せる線形の少数ショット学習」をそのまま持つ．本研究の部品 2 の出発点（ソースのコーパスで学んだ分類器）とは出所が違うが，査読者が「CLAP と同じ」と指摘しうるので，関連研究に入れる．CLAP は罰則の強さをクラスごとに分け，本研究は重みと切片で分けるので，両者は別の軸である．
- TransCLIP は「先生の予測への KL をラベル無しの全標本にかけ，少数ラベルと同時に解く」ので，本研究の弱いラベル（総重量 m_eff）を損失として見たものと同じ形である．違いは，TransCLIP の重みが固定（λ=0.5）で，本研究は K ごとに仮のターゲットで m_eff を選ぶ点と，本研究の先生がソースのドメインで学んだ融合の分類器である点である．
- 以上から，「出発点への L2 ＋ K 本 ＋ 別モダリティの先生の弱いラベル」の 3 つを同時に持つ研究は，CLIP 系の 2 本（CLAP は罰則と出発点，TransCLIP は先生の弱いラベルと少数ラベル）でほぼ覆われる．1.12 節の判定「三つの同時使用は見つからなかった」は，CLIP 系を含めると「部分的」に下げるのが安全である．

### Gaps
- CLAP と TransCLIP は本文の後半（10 万字以降）を読んでいない．付録に切片や他の設定の扱いがあるかは未確認である．
- Agarla ほかの疑似ラベルの作り方，ラベルの本数，掲載年（2023 か 2024 か）は本文で確かめていない．
- CLIP 系の転導的な少数ショット（ZLaP，EM-Dirichlet など）は探していない．

## Q3. クラス均衡した疑似ラベル（列正規化，クラスごとの総質量をそろえる，総重量を縛る）の先例

### Takeaway
「クラスごとの総量をそろえる」操作は，既知の ReMixMatch・SeLa・SLA・SHOT のほかに，転導的な少数ショット学習（PT-MAP の Sinkhorn の列の制約「各クラス q 本」，TIM の情報最大化），CLIP の疑似ラベル（UPL の「各クラスで確信度の上位 K 本」），Noisy Student（クラスごとに 13 万枚にそろえる），AdaMatch の分布整合，DARP の凸最適化にある．一方で，Veilleux ほか（NeurIPS 2021）は，転導的な少数ショットの方法が「ラベル無しの集合はクラスが均等」という仮定に頼っており，クラスの比率が偏ると帰納的な方法より悪くなることがあると示した．本研究の列正規化（一様な総量に合わせる）はこの警告の対象になる．

### Cited Findings

| 文献 | 何をするか | 本研究（列正規化＋m_eff）との違い | 水準 | 出典 |
|---|---|---|---|---|
| Hu, Gripon, Pateux, "Leveraging the Feature Distribution in Transfer-based Few-Shot Learning" (PT-MAP), ICANN 2021 (LNCS)，arXiv:2006.03806 | ラベル無しの標本とクラスの中心の間の最適輸送を Sinkhorn で解く．行の制約で各標本の質量を 1，列の制約で各クラスの質量を q にする．「多くの研究は各クラスにちょうど q 本あるという情報を使う」と書く．クラスが不均衡なときの実験もあり，比率がおおよそ分かれば帰納的な方法を上回る | 「クラスごとの総量をそろえた柔らかい割り当て」の先例．本研究の列正規化は，Sinkhorn の列の制約だけを 1 回かけて行を正規化しない形にあたる．違い：先生は無く，割り当てはクラスの中心からの距離で作る | ◎（ar5iv 本文）．掲載先は OpenAlex の書誌 | [ar5iv](https://ar5iv.labs.arxiv.org/html/2006.03806)，[DOI](https://doi.org/10.1007/978-3-030-86340-1_39) |
| Huang, Chu, Wei, "Unsupervised Prompt Learning for Vision-Language Models" (UPL), arXiv:2204.03649 | CLIP のゼロショットの予測で疑似ラベルを作り，閾値ではなく「各クラスで確信度の上位 K 本（K=16）」を選ぶ．理由は「CLIP は下流の課題でクラスごとに偏った好みを示し，疑似ラベルが不均衡になる」．CLIP の重みは凍結 | 「学び直さない先生の出力がクラスに偏るので，クラスごとに同じ量を採る」という動機が本研究と同じ．違い：硬いラベルで本数をそろえる（本研究は柔らかいラベルの総量をそろえる），総重量の選択なし | ◎（ar5iv 本文） | [ar5iv](https://ar5iv.labs.arxiv.org/html/2204.03649) |
| Xie ほか, "Self-training with Noisy Student improves ImageNet classification," arXiv:1911.04252（CVPR 2020 は要確認） | 先生の確信度の低い画像を捨て，クラスごとに 13 万枚にそろえる（足りないクラスは複製し，多いクラスは確信度の上位だけ残す）．疑似ラベルは柔らかい版が既定 | クラスごとの量をそろえる操作の大規模な先例．違い：本数でそろえる，先生は同じモダリティ | ◎（ar5iv 本文） | [ar5iv](https://ar5iv.labs.arxiv.org/html/1911.04252) |
| Berthelot ほか, AdaMatch, arXiv:2106.04732 | ターゲットの疑似ラベルに E[ソースの予測]/E[ターゲットの予測] を掛け，行を正規化し直す（ReMixMatch の分布整合を，ソースの予測の平均を目標にして DA に移したもの）．ターゲットのラベル分布が分かればそれを目標にできる | 列平均で割る点は同じ．違い：行を正規化し直す，目標はソースの予測の平均（本研究は一様），逐次的 | ◎（ar5iv 本文） | [ar5iv](https://ar5iv.labs.arxiv.org/html/2106.04732) |
| Kim, Hur, Park, Yang ほか, "Distribution Aligning Refinery of Pseudo-label for Imbalanced Semi-supervised Learning" (DARP), arXiv:2007.08844（NeurIPS 2020 は要確認） | 偏った模型の疑似ラベルを，目標のクラス分布に合うように凸最適化で柔らかく直す | 「柔らかいラベルのクラスごとの総量を目標に合わせる」最適化の先例．本研究の列正規化はその 1 段の近似と見なせる | ○（OpenAlex の抄録） | [arXiv](https://doi.org/10.48550/arxiv.2007.08844) |
| Boudiaf ほか, "Transductive Information Maximization For Few-Shot Learning" (TIM), NeurIPS 2020, arXiv:2008.11297 | ベースのクラスで学んだ特徴抽出器を固定し，支持集合の教師あり損失＋問い合わせ標本の予測とその特徴の相互情報量の最大化で解く | 相互情報量の周辺エントロピーの項がクラスの予測を均等に寄せる（抄録に明記は無く，これは推論）．凍結した特徴の上で少数ラベル＋ラベル無しを使う点は本研究と同じ | ○（arXiv の抄録） | [arXiv](https://arxiv.org/abs/2008.11297) |
| Veilleux, Boudiaf, Piantanida, Ben Ayed, "Realistic Evaluation of Transductive Few-Shot Learning," NeurIPS 2021, arXiv:2204.11181 | 転導的な少数ショットのベンチマークは「推論時に完全にクラスが均衡した課題」を使うと指摘する．クラスの比率をディリクレ分布で引くと，最新の転導的な方法の成績が大きく落ち，帰納的な方法を下回る場合がある．α ダイバージェンスで一般化した相互情報量の損失（α-TIM）を提案 | **本研究の列正規化（一様な総量）への直接の警告**．ターゲットのコーパスのクラスの比率が偏っていると，一様にそろえる操作は損になりうる | ○（arXiv の抄録） | [arXiv](https://arxiv.org/abs/2204.11181) |
| Menghini, Delworth, Bach, NeurIPS 2023 | 通常の疑似ラベルは質の良いクラスへの偏りを強める，と述べる | 均衡が要る理由の根拠 | ○ | [DOI](https://doi.org/10.52202/075280-2665) |

### Inferences
- 「学び直さない先生の出力はクラスに偏るので，クラスごとの量をそろえる」という動機と操作は UPL・Noisy Student・PT-MAP にある．本研究の列正規化を新しい操作として書くことはできない．「柔らかいラベルの列の総和で 1 回だけ割り，行を正規化せず，総重量 m_eff を別に縛る」という形そのものは今回も見つからなかったが，これは Sinkhorn（PT-MAP，SeLa）の 1 回目の列のスケーリングと DARP の近似として説明できるので，手法の新しさの根拠にはならない．
- Veilleux ほかの結果から，査読者は「ターゲットのクラスの比率が一様でないときに列正規化は害にならないか」と問うはずである．本研究の 7 群はクラスの比率が群ごとに違う（`data/README.md` で確かめること）ので，比率の偏りと列正規化の得の関係を示す表を用意しておくのが安全である．

### Gaps
- 感情認識のドメイン適応でクラス均衡した疑似ラベルを使った論文は，1.12 節と同じく今回も見つからなかった（探したのは英語の論文だけ）．
- PT-MAP と UPL の掲載先（会議か雑誌か）は，PT-MAP は OpenAlex の ICANN 2021 の記録，UPL は arXiv の頁だけで確かめた．

## Q4. leave-one-source-out（仮のターゲット）で，蒸留や半教師あり学習のハイパーパラメータをラベル予算ごとに選ぶ先例はあるか（部品 3）

### Takeaway
ラベル予算 K ごとに，仮のターゲット（ソースのコーパスを 1 つずつ抜いたもの）で蒸留の重みや先生を選ぶ研究は，今回も見つからなかった．見つかったのは，(a) 設定を 1 つに固定して全データセット・全ショット数に使う流儀（AdaMatch，CLIP-LoRA，CLAP，TransCLIP），(b) あるデータセットで決めた設定を別のデータセットに使うと最大 21% 落ちるという CLAP の報告，(c) 半教師ありの方法はラベルの量によって効き方が変わるという Oliver ほか（NeurIPS 2018）の指摘，である．(b) は本研究の仮のターゲットでの選択への反証の材料になりうる．

### Cited Findings

| 文献 | 何を述べるか | 本研究との関係 | 水準 | 出典 |
|---|---|---|---|---|
| Silva-Rodríguez ほか, CLAP, CVPR 2024 | 既存のアダプタは大きな検証集合での格子探索に頼り，それは非現実的と述べる．あるデータセットで調整した設定を別のデータセットに使うと，最大 21% 落ちる（図 1）．CLAP は検証データを使わず 1 つの設定を全データセット・全ショット数・全骨格に使う．ショット数が増えると利点は小さくなる（限界の節） | 「別のデータで選んだ設定は移らない」という反証の材料．本研究は仮のターゲットで選んだ値とオラクルの差を K ごとに示す必要がある | ◎（arXiv HTML 本文の前半） | [arXiv HTML](https://arxiv.org/html/2312.12730) |
| Zanella & Ben Ayed, CLIP-LoRA, CVPRW 2024 | 同じ設定を全データセット・全ショット数で使って既存法を上回る | 設定を K ごとに変えない流儀の例 | ○ | [DOI](https://doi.org/10.1109/cvprw63382.2024.00166) |
| Berthelot ほか, AdaMatch | τ=0.9，uratio=3 を全実験で固定．τ は UDA の 1 組で {0.8, 0.9, 0.95} から選んだ | 固定の設定で SSL・UDA・SSDA を扱う流儀の例 | ◎ | [ar5iv](https://ar5iv.labs.arxiv.org/html/2106.04732) |
| Zanella, Gérin, Ben Ayed, TransCLIP | KL の重み λ をゼロショットで 1，少数ショットで 0.5 と，ラベルの有無で変えて固定 | 「ラベル予算によって先生への重みを変える」ことの最も粗い形（0 本と少数の 2 段階） | ◎ | [arXiv HTML](https://arxiv.org/html/2406.01837) |
| Oliver, Odena, Raffel, Cubuk, Goodfellow, "Realistic Evaluation of Deep Semi-Supervised Learning Algorithms," NeurIPS 2018 | ラベル無しデータを使わない基準が過小に報告されている，方法によってラベル・ラベル無しの量への敏感さが違う，ラベル無しデータに別のクラスが混ざると大きく落ちる | 「最適な設定はラベル予算で変わる」ことの間接の根拠 | ○（arXiv の抄録） | [arXiv](https://arxiv.org/abs/1804.09170) |
| Hatano ほか, MM-CDFSL, ECCV 2024 | 釣り合いの重みを報告に使うベンチマーク（EPIC-Kitchens の 5-shot）で掃いて決めている | 少数ショットの DA で設定の選び方が明示されない例 | ◎ | [arXiv HTML](https://arxiv.org/html/2405.19917v1) |
| Phoo & Hariharan, STARTUP | ラベル無しデータの 10% を検証に取り，学習率を検証損失で選ぶ．SimCLR の温度は検証集合が無いので 1 に固定 | 選び方の例 | ◎ | [ar5iv](https://ar5iv.labs.arxiv.org/html/2010.07734) |
| "Rethinking Unsupervised Domain Adaptation for Semantic Segmentation," arXiv:2207.00067 | 1・2・5・10・50・100 本のラベル付きの検証集合で選んだ設定を，最適な設定と比べ，何本あれば足りるかを調べる | K 本での選択の当たりを本数の関数で測った先例の候補．本研究の E28-8 に近い | △（検索の要約．著者は要確認） | [arXiv](https://arxiv.org/pdf/2207.00067) |
| Distributionally Robust Classification for Multi-source Unsupervised Domain Adaptation, arXiv:2601.21315 (2026) | 設定はターゲットの小さな検証集合（クラスあたり 10 本）で選び，ターゲットのラベルを使わない leave-one-domain-out の交差検証の結果を補足として示す | 複数ソースで LODO を選び方として併記する例 | △（検索の要約） | [arXiv](https://arxiv.org/pdf/2601.21315) |
| Mahajan ほか（著者は要確認）, "Domain Generalization using Causal Matching," arXiv:2006.07500 | 見えないドメインのデータを見てから設定を決める慣行は DG の前提に反すると述べ，ソースだけで検証集合を作る | 仮のターゲットの選択をソースだけで閉じる根拠 | △（検索の要約） | [arXiv](https://arxiv.org/pdf/2006.07500) |
| Ericsson ほか, "Better Practices for Domain Adaptation," arXiv:2309.03879；Saito ほか, "Tune it the Right Way: Unsupervised Validation of Domain Adaptation via Soft Neighborhood Density," ICCV 2021 | DA の検証の手順の整理と，ラベル無しの検証指標 | 選び方の関連研究 | △（検索の要約．著者・掲載先は要確認） | [ar5iv](https://ar5iv.labs.arxiv.org/html/2309.03879)，[DeepAI](https://deepai.org/publication/tune-it-the-right-way-unsupervised-validation-of-domain-adaptation-via-soft-neighborhood-density) |

### Inferences
- 選び方の手続き（ソースのドメインを 1 つ抜いて評価する）は DomainBed の leave-one-domain-out と同じで，新しくない．選ぶ対象（先生の種類，重みづけ，m_eff）と，選ぶ軸（K ごと）を変えた研究は見つからなかったが，これは適用の範囲の違いであって手法の新しさにはならない．
- CLAP の「設定は別のデータセットに移すと最大 21% 落ちる」は，本研究の仮のターゲットでの選択がうまくいかない可能性の根拠として査読者が使いうる．本研究は「仮のターゲットで選んだ値」「K 本の交差検証」「オラクル」の差を K ごとに示し（R7），仮のターゲットの選択がオラクルにどれだけ近いかを数値で書く必要がある．

### Gaps
- 半教師あり学習・蒸留で「ラベル予算ごとに設定を選び直す」ことを明示的に調べた研究は，Oliver 2018 の抄録以外に見つからなかった．本文（ラベル数を変えた実験の節）は未読である．
- SER のクロスコーパスで，leave-one-corpus-out を評価ではなく設定の選択に使った研究は探しきれていない．

## Q5. 感情認識での半教師ありの DA・少数ショットの DA（2022〜2026）で，標準の比較相手になっている手法は何か

### Takeaway
感情認識の半教師ありの DA には，画像の DA の MME・DomainBed のような共通の比較の枠は見つからなかった．見つかった論文は，疑似ラベル（Agarla ほか），敵対的な整列（Latif ほか TAC 2022），PEFT の 2 段の適応（Lashkarashvili ほか ICASSP 2024）を個別に提案している．テスト時の適応（ターゲットのラベル無しだけで推論時に直す）では，Dong, Jia, Dang（ICASSP 2026）が 11 手法を SER の 3 つの課題で比べ，「エントロピー最小化と疑似ラベルは概して失敗する」と報告した．これは本研究の比較相手（自己訓練，エントロピー最小化）の結果を解釈するときに引ける．

### Cited Findings

| 文献 | 何をするか | 比較相手としての意味 | 水準 | 出典 |
|---|---|---|---|---|
| Dong, Jia, Dang, "Test-Time Adaptation for Speech Emotion Recognition," ICASSP 2026, arXiv:2601.16240 | SER で初めての TTA の系統的な評価．11 手法・3 課題．逆伝播を使わない方法が最も有望で，エントロピー最小化と疑似ラベルは「概して失敗する」．理由は，1 つの確かな正解を仮定することが感情の表出のあいまいさと合わないため．どの手法もすべての設定で最良ではない | 本研究でエントロピー最小化・自己訓練が弱いなら，この報告と合う．手法名とコーパスは抄録に無い | ○（arXiv の抄録） | [arXiv](https://arxiv.org/abs/2601.16240) |
| EMO-TTA: Improving Test-Time Adaptation of Audio-Language Models for Speech Emotion Recognition, arXiv:2509.25495 | 音声言語模型（CLAP）のための，学習を要しない TTA．クラスごとの統計を逐次更新する．6 つのクロスコーパスで改善と二次資料が述べる | 著者は資料によって食い違う（要確認） | △ | [arXiv](https://arxiv.org/pdf/2509.25495) |
| Agarla, Bianco, Celona, Napoletano, ESWA 2023/2024 | 少数のターゲットラベル＋疑似ラベル（硬い・柔らかい）での言語をまたぐ SER．既存法より UA が平均 40% 上がると報告 | SER の少数ラベル＋疑似ラベルの比較相手の候補 | ○ | [DOI](https://doi.org/10.1016/j.eswa.2023.121368) |
| Zhang, Jia, Cao, Ru, "Cross-corpus speech emotion recognition using semi-supervised domain adaptation network," Speech Communication 2024 | 題名から，クロスコーパス SER の半教師ありの DA の網 | 中身（ターゲットのラベルの量，比較相手）は未確認 | △（書誌のみ） | [DOI](https://doi.org/10.1016/j.specom.2024.103185) |
| Latif, Rana, Khalifa, Jurdak, "Self Supervised Adversarial Domain Adaptation for Cross-Corpus and Cross-Language Speech Emotion Recognition," IEEE TAC 2022 | 自己教師ありの事前学習と敵対的な DA．クロスコーパスとクロス言語で成績が大きく落ちることを出発点にする | TAC に載った SER の DA の代表．敵対的な整列の系統 | ○（抄録の冒頭のみ） | [DOI](https://doi.org/10.1109/taffc.2022.3167013) |
| Lashkarashvili, Wu, Sun, Woodland, "Parameter Efficient Finetuning for Speech Emotion Recognition and Domain Adaptation," ICASSP 2024 | PEFT の組み合わせが全体の微調整を上回る．演技のデータで学んだ模型を自然な感情に合わせる 2 段の適応 | 微調整する設定で，本研究（凍結したエンコーダ）とは違う | ○ | [DOI](https://doi.org/10.1109/icassp48485.2024.10446272) |
| MER2024 の大会論文（arXiv:2408.10500 は Emotion-LLaMA で 2 万件のラベル無しにラベルを付けて学習に足す；arXiv:2409.05007 は音声・映像・言葉を融合し確信度 0.9 以上の疑似ラベルを残す） | 多モーダルの模型の疑似ラベルで学習集合を増やす | 同じコーパスの中の半教師あり．ドメインをまたがない | △ | [2408.10500](https://arxiv.org/pdf/2408.10500)，[2409.05007](https://arxiv.org/pdf/2409.05007) |

- 背：画像の半教師ありの DA の標準は MME（Saito ほか 2019），APE，CDAC，AdaMatch（Q2），テスト時の適応の標準は Tent（Wang ほか ICLR 2021），SAR，T3A で，本研究の比較相手のうちエントロピー最小化は Tent の線形版にあたる．これらは今回の調べでは本文を確かめていない．

### Inferences
- 感情認識には共通の比較の枠が無いので，本研究が比較相手を「半教師あり学習の古典（自己訓練，エントロピー最小化，共訓練，共正則化，Saerens の EM），蒸留（温度付き，一般化蒸留），能動 DA（CLUE，TQS）」から選び，すべてを同じ線形の生徒と同じ選び方で比べた，と書くのは筋が通る．
- AdaMatch は「ソースのラベル＋ターゲットの K 本＋ターゲットのラベル無し」の設定の代表で，分布整合（Q3）も持つので，比較相手に足すか，足さない理由（深層網のデータ拡張が前提で，凍結した特徴の上の線形分類器では弱い拡張と強い拡張を作れない）を書くのがよい．

### Gaps
- Dong ほか 2026 の 11 手法の名前とコーパスは本文を読んでいない．
- 顔表情のクロスドメイン（例：Chen ほか TPAMI 2021 の統一ベンチマーク）の比較相手は調べていない．

## Q6. 学習時だけ使えるモダリティ（特権モダリティ・LUPI）を使う感情認識の研究（2020〜2026）で，見落としていたものはあるか

### Takeaway
1.12 節の表に無いものとして，PriCon（ICMI 2025．実験室から実環境への移し替えを目的にした特権情報の事前学習），映像から脳波への蒸留（Pattern Recognition 2022），映像から音声への知識の移し替えと半教師あり学習の組み合わせ（Knowledge-Based Systems 2021．書誌は要確認）が見つかった．このうちドメインのずれを目的に明記するのは PriCon だけで，それもコーパスをまたぐ評価かは抄録から分からない．Hajavi & Etemad 2023 は掲載誌が IEEE Transactions on Artificial Intelligence であることが分かった（1.12 節では「未確認」）．

### Cited Findings

| 文献 | 何をするか | 本研究との関係 | 水準 | 出典 |
|---|---|---|---|---|
| Pinitas, Makantasis, Yannakakis, "Privileged Contrastive Pretraining for Multimodal Affect Modelling" (PriCon), ICMI 2025 | 教師ありの対照学習で事前学習した模型を，LUPI の枠の先生にする．目的は実験室（in-vitro）から実環境（in-vivo）への移し替え．RECOLA と AGAIN で LUPI と端から端まで学ぶ模型を上回り，全モダリティを使う模型に並ぶ場合が多い | 感情での「特権情報＋ドメインのずれ」を目的に書いた先例．違い：深層網，ターゲットの少数ラベル・弱いラベルの記述は無い | ○（OpenAlex の抄録） | [DOI](https://doi.org/10.1145/3716553.3750766) |
| Zhang, Tang, Guan, "Visual-to-EEG cross-modal knowledge distillation for continuous emotion recognition," Pattern Recognition 2022 | 映像の先生の暗黙知で脳波の連続感情の予測を良くする | 別モダリティの先生 → 単モーダルの生徒（感情，連続値）．ドメインのずれは扱わない | ○（抄録の冒頭） | [DOI](https://doi.org/10.1016/j.patcog.2022.108833) |
| Zhang, Chen, Chen, Li ほか, "Combining cross-modal knowledge transfer and semi-supervised learning for speech emotion recognition," Knowledge-Based Systems 2021 | 映像から音声への知識の移し替えと半教師あり学習を組み，映像の手がかりで疑似ラベルの雑音を減らす．一致の正則化と 2 つの融合の仕方（重みつき，consistent & random）．CH-SIMS と IEMOCAP | 「別モダリティで疑似ラベルを良くして音声の生徒を学ぶ」感情認識の先例．コーパスの中か，コーパスをまたぐかは不明 | △（検索の要約．巻と論文番号は資料によって 229 巻 107340 と「346」で食い違い，要確認．本文の PDF は 403） | [datalearner の記録](https://www.datalearner.com/academic/journal-papers/0950-7051/volumes-and-issues/346/paper-detail/10089)，[著者の頁の PDF](https://users.monash.edu/~yli/assets/pdf/2021-KBS-SpeechEmotionRecognition.pdf) |
| Mitra, Chien, Kowtha, Cheng ほか, "Speech Emotion: Investigating Model Representations, Multi-Task Learning and Knowledge Distillation," Interspeech 2022 | 言葉の情報が感情価（valence）の推定を良くすることを出発点にした，音声の模型の表現・多課題学習・蒸留の調べ | 言葉 → 音声の蒸留の候補．蒸留の向きと設定は抄録の冒頭からは確かめられない | △（抄録の冒頭のみ） | [DOI](https://doi.org/10.21437/interspeech.2022-957) |
| Hajavi & Etemad, "Audio Representation Learning by Distilling Video as Privileged Information," IEEE Transactions on Artificial Intelligence 2023 | 映像を特権情報にした音声の表現の蒸留（1.12 節で既知） | 掲載誌が分かった（1.12 節の「掲載誌は未確認」を直せる） | ○（OpenAlex の書誌と抄録） | [DOI](https://doi.org/10.1109/tai.2023.3243596) |
| Aslam ほか, "Distilling Privileged Multimodal Information for Expression Recognition using Optimal Transport" (PKDOT), FG 2024 | 1.12 節の PKDOT（arXiv:2401.15489）の掲載先が IEEE FG 2024 と分かった | 書誌の補い | ○（OpenAlex の書誌と抄録） | [DOI](https://doi.org/10.1109/fg59268.2024.10582047) |
| MGCP（ENIAC．ブラジルの会議） | 音声と言葉の模型を組み，ターゲットのラベル無しで音声の感情認識を可能にする | 言葉の先生による音声の疑似ラベルの候補．詳細不明 | △ | [SBC](https://sol.sbc.org.br/index.php/eniac/article/download/25746/25562) |

### Inferences
- 感情認識の特権モダリティの研究は，ほぼすべてがコーパスの中か第三のコーパス経由で，ソースのコーパスで学んだ多モーダルの先生をターゲットのコーパスに当てる形は無い，という 1.12 節の判定は今回も変わらない．PriCon は「ドメインのずれのために特権情報を使う」目的を持つので，関連研究で 1 文触れる価値がある．
- KBS 2021 は「別モダリティの手がかりで音声の疑似ラベルを良くする半教師あり学習」で，本研究の部品 4 の動機に近い．書誌が食い違っているので，引くなら出版社の頁で確かめる．

### Gaps
- KBS 2021 と Mitra 2022 の本文は読めていない．
- 脳波・生理信号の感情認識（DEAP，SEED）での特権情報とドメインのずれ（被験者をまたぐ）の研究は，DISD-Net（IEEE TMM 2025，[DOI](https://doi.org/10.1109/tmm.2025.3535344)．題名のみ △）以外は探していない．

## Q7. 「単モーダルの先生の積を参照にして融合の生徒を学ぶ」「往復の二段構え」に近いものはあるか（部品 5）

### Takeaway
単モーダルの先生から多モーダルの生徒へ，という向きは G²D（ICCV 2025）が新たに見つかり，既知の UMT・UME・MKE・SUMMER と同じ系統である．融合の模型と単モーダルの模型の間で双方向に蒸留するものは，DAMR（Electronics 2026，MELD・IEMOCAP）と，xMUDA 系の fusion-then-distillation（題名のみ）があった．単モーダルの予測の「積」を蒸留の参照にする研究は，今回も見つからなかった．

### Cited Findings

| 文献 | 何をするか | 本研究（部品 5）との関係 | 水準 | 出典 |
|---|---|---|---|---|
| Rakib & Bagavathi, "G²D: Boosting Multimodal Learning with Gradient-Guided Distillation," ICCV 2025, arXiv:2506.21514 | 単モーダルと多モーダルの目的を合わせた損失で多モーダルの模型を学び，モダリティが順に学習を主導するようにして，強いモダリティが弱いモダリティを押しのけるのを防ぐ | 単モーダルの先生 → 多モーダルの生徒．積は使わない．ドメインのずれ・少数ラベルなし | ○（arXiv の抄録） | [arXiv](https://arxiv.org/abs/2506.21514) |
| G²D の関連研究の節（検索の要約） | UMT は単モーダルの先生の出力で生徒を直接教え，UME は先生の logit をまとめる．どちらを使うかは経験的な調整が要る | logit の和（＝確率の積に比例）を予測に使う UME と，本研究の「積を参照にする」の違いを書く材料 | △ | [arXiv PDF](https://arxiv.org/pdf/2506.21514) |
| Liu, Gu, Zhang, Wang, "Dynamic Mutual Refinement for Multimodal Speech Emotion Recognition: A Bidirectional Distillation and Inverse Curriculum Learning Approach" (DAMR), Electronics 2026 | 多モーダルの模型が単モーダルの模型の正しく分類する標本を誤る（逆も）という「分類のあいまいさ」を減らすため，EMA で能力を測って重みを変える相互蒸留を多モーダルと単モーダルの模型の間で行う．MELD・IEMOCAP で基準より 5.54% 上がる | 感情認識での融合と単モーダルの往復の蒸留の先例．違い：同時に学ぶ相互蒸留（二段構えではない），同じコーパスの中，積の参照なし | ○（OpenAlex の抄録） | [DOI](https://doi.org/10.3390/electronics15204690) |
| Cross-modal UDA for 3D Semantic Segmentation via Bidirectional Fusion-then-Distillation (ACM MM 2023)；Fusion-Then-Distillation (IEEE TCSVT 2025) | 題名から，融合した予測を各モダリティへ蒸留し，双方向に回す DA | ドメインをまたぐ往復の候補．中身は未確認 | △（題名のみ） | [MM 2023](https://doi.org/10.1145/3581783.3612013)，[TCSVT](https://doi.org/10.1109/tcsvt.2025.3545971) |
| MST-Distill: Mixture of Specialized Teachers for Cross-Modal Knowledge Distillation, arXiv:2507.07015 | 専門化した先生の混合でモダリティをまたぐ蒸留を行う | 先生の組み合わせ方の例（積ではない） | △ | [arXiv HTML](https://arxiv.org/html/2507.07015v1) |
| Jaritz ほか, xMUDA | 2 つのモダリティの頭が互いの予測をまねる（ターゲット上） | 往復の同時版．二段構えではない | ○ | [arXiv](https://arxiv.org/abs/2101.07253) |

- 背：Zhang ほか Deep Mutual Learning（CVPR 2018），Guo ほか KDCL（CVPR 2020．仲間の集団を先生にする），Anil ほか codistillation（ICLR 2018）は，往復・相互の蒸留の一般の原型である．今回は確かめていない．

### Inferences
- 「単モーダルの先生が融合を助ける」と「融合と単モーダルの往復」はどちらも既にある（UMT・MKE・G²D と DAMR・xMUDA 系）．部品 5 で残る違いは，(i) 参照が単モーダルの生徒の出力の積（確率の積を正規化したもの）であること，(ii) 学習が同時の相互蒸留ではなく，先に単モーダル，次に融合という二段構えであること，(iii) 単モーダルの生徒もターゲットの K 本と弱いラベルで学んでいること，の 3 点である．
- 単モーダルの確率の積を正規化したものは，logit の和を softmax に通したものに厳密に等しい．したがって UME（logit をまとめる単モーダルの集団）を予測器として使うことと，本研究の「積を参照にして融合の生徒を学ぶ」ことの違いは，積を最終の予測にせず，融合の線形の生徒に写し取る点にある．この違いが成績に効くかどうかは，「積をそのまま予測に使う」比較を表に入れて示す必要がある（これは推論で，先行研究の主張ではない）．

### Gaps
- fusion-then-distillation の 3 本（ACM MM 2022・2023，TCSVT 2025）の本文を読んでいない．部品 5 に最も近い可能性があるので，引用の前に必ず確かめる．
- 「product of experts」を蒸留の目標に使う研究は，検索語を変えても見つからなかった．英語以外の論文は探していない．

## Q8. 部品 (1)（ソースとターゲットをそれぞれ中心化する）に見落とした先例はあるか

### Takeaway
転導的な少数ショット学習に，本研究の部品 (1) と同じ操作が 2 つある．PT-MAP の「trans-mean-sub」は，ラベル付きの標本からはラベル付きの平均を，ラベル無しの標本からはラベル無しの平均を，別々に引く．BD-CSPN（ECCV 2020）は，問い合わせ（ラベル無し）の特徴に「支持集合の平均 − 問い合わせの平均」を足す（cross-class bias の除去）．どちらもドメインのずれではなく，同じドメインの中のラベル付き集合とラベル無し集合のずれを対象にしている．

### Cited Findings

| 文献 | 何をするか | 本研究との違い | 水準 | 出典 |
|---|---|---|---|---|
| Hu, Gripon, Pateux, PT-MAP, ICANN 2021 | 転導的な設定で「trans-mean-sub」：ラベル付きの平均をラベル付きの標本から，ラベル無しの平均をラベル無しの標本から引き，単位球に射影する（べき変換の後） | 集合ごとに平均を引く操作は同じ．対象は同じドメインの支持集合と問い合わせ集合 | ◎（ar5iv 本文） | [ar5iv](https://ar5iv.labs.arxiv.org/html/2006.03806) |
| Liu, Song, Qin, "Prototype Rectification for Few-Shot Learning" (BD-CSPN), ECCV 2020, arXiv:1911.10713 | 正規化した問い合わせの特徴に ξ ＝（支持集合の平均）−（問い合わせの平均）を足す（式 8）．さらに確信度の上位 Z 本の問い合わせに疑似ラベルを付けて原型を直す | 平均の差だけ動かす操作は本研究の中心化と同じ効果．疑似ラベルで原型を直す部分は部品 4 の素朴な版 | ◎（ar5iv 本文） | [ar5iv](https://ar5iv.labs.arxiv.org/html/1911.10713) |

### Inferences
- 中心化は，1.8.1 節の AdaBN・コーパス正規化・Chen & Bühlmann 2021 に，転導的な少数ショットの PT-MAP・BD-CSPN を加えて，操作として既知と書く．BD-CSPN は中心化と疑似ラベルによる原型の修正を同じ論文で組んでいるので，「中心化＋弱いラベル」の組み合わせも少数ショットの文脈では既にある．

### Gaps
- 音声の話者正規化・コーパス正規化と転導的な少数ショットの中心化を並べて論じた文献は探していない．

## Q9. 判定：部品 (1)〜(5) と組み合わせ全体の新規性，論文に書いてよいこと・書いてはいけないこと

### Takeaway
5 つの部品はどれも操作として先例がある．1.12 節の判定と比べると，CLIP 系の少数ショット適応（CLAP，TransCLIP，UPL）と，ドメインをまたぐ少数ショットの多モーダル蒸留（MM-CDFSL）を見落としていたので，部品 2＋4 の組み合わせの判定を「見つからず」から「部分的」に下げる．組み合わせ全体（ソースのコーパスだけで学んだ融合の線形分類器を学び直さずにターゲットの全発話に当て，列正規化した弱いラベルの総量を K ごとに仮のターゲットで選び，出発点への罰則つきの線形の単モーダルの生徒を K 本で学び，さらに単モーダルの積を参照に融合の生徒を学ぶ）は見つからなかった．

### Cited Findings
- 部品 (1) 中心化：**あり**．AdaBN・コーパス正規化・Chen & Bühlmann 2021（1.8.1 節）に加え，PT-MAP の trans-mean-sub（[ar5iv](https://ar5iv.labs.arxiv.org/html/2006.03806)）と BD-CSPN の平均の差による移動（[ar5iv](https://ar5iv.labs.arxiv.org/html/1911.10713)）．
- 部品 (2) 出発点への L2 罰則（重みと切片で強さを分ける）：**操作はあり，重みと切片で分ける点は見つからず**．仮説転移の族（1.8.2 節）に加え，CLAP は別モダリティ（テキスト）から来た分類器への L2 罰則をクラスごとの強さで使う（[arXiv HTML](https://arxiv.org/html/2312.12730)）．
- 部品 (3) 仮のターゲットで K ごとに選ぶ：**手続きはあり，K ごとに蒸留の設定を選ぶ点は見つからず**．DomainBed の leave-one-domain-out（1.8.3 節）が手続きの先例．CLAP は設定を別のデータセットに移すと最大 21% 落ちると報告する（[arXiv HTML](https://arxiv.org/html/2312.12730)）．
- 部品 (4) 学び直さないソースの融合の先生の弱いラベル（列正規化，m_eff）：**部分的**．学び直さない先生の確率をターゲットのラベル無し全体に付けるのは STARTUP（[ar5iv](https://ar5iv.labs.arxiv.org/html/2010.07734)），別モダリティの先生の予測への KL と少数ラベルを同時に解くのは TransCLIP（[arXiv HTML](https://arxiv.org/html/2406.01837)），多モーダルの先生 → 単モーダルの生徒をドメインをまたぐ少数ショットで行うのは MM-CDFSL（[arXiv HTML](https://arxiv.org/html/2405.19917v1)），クラスごとの量をそろえるのは UPL・Noisy Student・PT-MAP・DARP・AdaMatch（Q3）．
- 部品 (5) 単モーダルの積を参照にした融合の生徒（二段構え）：**積を参照にする点は見つからず，往復は部分的**．単モーダル → 融合は UMT・MKE・G²D（[arXiv](https://arxiv.org/abs/2506.21514)），融合と単モーダルの双方向の蒸留は DAMR（[DOI](https://doi.org/10.3390/electronics15204690)）と xMUDA 系．
- 組み合わせ全体：**見つからず**（探した範囲：本ノートの Q1〜Q8 と 1.12 節の検索語．英語の論文だけ）．

### Inferences

**論文に「新しさ」として書いてよいこと**
- コーパスをまたぐマルチモーダル感情認識で，凍結したエンコーダの上の線形分類器という解析しやすい形に部品を組み，「ソースのコーパスだけで学んだ融合の分類器を，ターゲットのデータで学び直さずに弱いラベルの元にする」構成を，K = 0〜50 の全域で測ったこと（MM-CDFSL の先生はターゲットのラベル無しデータを使い，STARTUP の弱いラベルは表現にしか使えず，TransCLIP の先生はドメインのずれを扱わない，という違いを並べて書く）．
- 弱いラベルの総重量 m_eff と先生の種類を，ラベル予算 K ごとに仮のターゲットで選ぶと，最適な値が K でどう動くかを示したこと（TransCLIP は 0 本と少数の 2 段階で固定値を変えるだけ）．これは手法の新しさではなく，測った結果として書く．
- 弱いラベルの得がどの条件（K，群，モダリティの組）で生じ，どの条件で消えるかを，仕組みの量（相補性 J など）で説明したこと．
- 単モーダルの生徒の出力の積を，最終の予測ではなく融合の生徒の参照に使う二段構え（ただし fusion-then-distillation の 3 本を読んで違いを確かめてから書く）．

**論文に書いてはいけないこと**
- 「モダリティをまたぐ蒸留を，ドメインをまたぐ少数ショットの設定で初めて行った」（MM-CDFSL，xMUDA／xMoSSDA）．
- 「学び直さない先生の柔らかいラベルを，ターゲットのラベル無しデータ全体に付ける新しい方法」（STARTUP，TransCLIP，Menghini 2023）．
- 「別のモダリティから来た分類器を出発点にした L2 罰則つきの線形プローブは新しい」（CLAP）．
- 「先生の出力のクラスの偏りを，クラスごとの量をそろえて直すのは新しい」（UPL，Noisy Student，PT-MAP，DARP，AdaMatch）．列正規化は Sinkhorn の列の制約の 1 回分として既知の操作の簡略形と書く．
- 「クラスを均衡させる仮定は安全」（Veilleux ほか 2021 が，ラベル無しの集合のクラスの比率が偏ると転導的な方法が帰納的な方法を下回ることを示した）．ターゲットのクラスの比率と列正規化の得の関係を示すまで，この仮定の安全さを主張しない．
- 「ソースとターゲットを別々に中心化する操作は新しい」（PT-MAP の trans-mean-sub，BD-CSPN，AdaBN）．
- 「仮のターゲットでの選択は新しい選び方」（DomainBed の leave-one-domain-out）．また，CLAP の「設定は別のデータセットへ移すと最大 21% 落ちる」に答えずに，仮のターゲットで選んだ設定が移ると主張すること．
- 「融合と単モーダルの間の双方向の蒸留は新しい」（DAMR，xMUDA 系）．
- 「エントロピー最小化や自己訓練が感情認識で効かないことを示した」を新しい発見として書くこと（Dong ほか ICASSP 2026 が TTA で報告済み）．

### Gaps
- 判定は英語の論文の検索だけに基づく．Gupta 2016・Lopez-Paz 2016 の被引用のたどりは回数制限で完了していない．
- 本文を読んでいない候補のうち，判定を変えうるのは fusion-then-distillation の 3 本（部品 5），Guo ほか 2026 の Information Fusion 論文（部品 4），KBS 2021（部品 4，感情認識）である．引用の前にこの 3 つを最初に確かめる．
