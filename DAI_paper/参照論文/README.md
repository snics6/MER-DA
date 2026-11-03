# 参照論文（DAI 論文のために読む論文の PDF）

2026-11-03 に作成．選んだ理由と各論文の要点は `research_notes/先行研究_研究目標の新規性_2026-11-03/report.md` と `docs/related-work.md`（1.12・1.15・2.2 節）にある．
PDF は git に入れない（`.gitignore` で除外．合計 36 MB）．消えたら下の取得元から落とし直す．
PDF はすべて arXiv か学会の公開版である．出版社の最終版と頁や文言が違うことがあるので，引用の頁は最終版で確かめる．

## 問い 1：別モダリティからの蒸留はなぜ効くか

5 本の要点（本文と付録を全部読んで書いた）：`要点_問い1_別モダリティ蒸留.md`

| ファイル | 書誌 | 取得元 |
|---|---|---|
| `2023_Xue_ModalityFocusingHypothesis_ICLR.pdf` | Xue, Gao, Ren, Zhao. The Modality Focusing Hypothesis: Towards Understanding Crossmodal Knowledge Distillation. ICLR 2023 | arXiv 2206.06487 |
| `2016_LopezPaz_UnifyingDistillationPrivilegedInformation_ICLR.pdf` | Lopez-Paz, Bottou, Schölkopf, Vapnik. Unifying Distillation and Privileged Information. ICLR 2016 | arXiv 1511.03643 |
| `2018_Albanie_CrossModalTransferSER_ACMMM.pdf` | Albanie, Nagrani, Vedaldi, Zisserman. Emotion Recognition in Speech using Cross-Modal Transfer in the Wild. ACM MM 2018 | arXiv 1808.05561 |
| `2022_Srinivasan_CrossModalConditionalTeacherStudentSER_ICASSP.pdf` | Srinivasan, Huang, Kirchhoff. Representation Learning through Cross-Modal Conditional Teacher-Student Training for Speech Emotion Recognition. ICASSP 2022 | arXiv 2112.00158 |
| `2020_Jaritz_xMUDA_CVPR.pdf` | Jaritz, Vu, de Charette, Wirbel, Pérez. xMUDA: Cross-Modal Unsupervised Domain Adaptation for 3D Semantic Segmentation. CVPR 2020 | arXiv 1911.12676 |

## 問い 2：どのソースのどのサンプルを，どの感情に使うか

5 本の要点（本文を全部読んで書いた）：`要点_問い2_ソースの選び方.md`

| ファイル | 書誌 | 取得元 |
|---|---|---|
| `2018_Guo_MultiSourceDAMixtureOfExperts_EMNLP.pdf` | Guo, Shah, Barzilay. Multi-Source Domain Adaptation with Mixture of Experts. EMNLP 2018 | arXiv 1809.02256 |
| `2021_Mansour_MultipleSourceAdaptationLimitedTargetLabels_AISTATS.pdf` | Mansour, Mohri, Ro, Suresh, Wu. A Theory of Multiple-Source Adaptation with Limited Target Labeled Data. AISTATS 2021 | arXiv 2007.09762 |
| `2026_Marin_ClassWiseAdaptiveModalityFusion_arXiv.pdf` | Marín, Marí, Haro, Redondo. Multimodal Emotion Recognition in Conversations via Class-Wise Adaptive Modality Fusion and Affective Geometry. arXiv 2026 | arXiv 2609.09924 |
| `2019_Li_Deng_FacialExpressionDatasetBias_ECAN_arXiv.pdf` | Li, Deng. A Deeper Look at Facial Expression Dataset Bias（ECAN）. arXiv 2019 | arXiv 1904.11150 |
| `2022_Ahn_Lee_Shin_CorpusWiseWeightsSER_Interspeech.pdf` | Ahn, Lee, Shin. Multi-Corpus Speech Emotion Recognition for Unseen Corpus Using Corpus-Wise Weights in Classification Loss. Interspeech 2022 | ISCA Archive |

## 問い 3：同じ設定の研究（少数ラベル，感情，コーパス間）

| ファイル | 書誌 | 取得元 |
|---|---|---|
| `2023_Feng_Chaspari_FewShotSERTransfer_TAC.pdf` | Feng, Chaspari. Few-shot Learning in Emotion Recognition of Spontaneous Speech Using a Siamese Neural Network with Adaptive Sample Pair Formation. IEEE TAC | arXiv 2109.02915 |
| `2021_Li_Kim_Kuo_Narayanan_AVEmotionDomainAdaptation_Interspeech.pdf` | Li, Kim, Kuo, Narayanan. Acted vs. Improvised: Domain Adaptation for Elicitation Approaches in Audio-Visual Emotion Recognition. Interspeech 2021 | arXiv 2104.01978 |
| `2025_He_M4SER_TASLP.pdf` | He, Shi, Hu, Mi, Li, Toda. M4SER. IEEE Transactions on Audio, Speech and Language Processing 33:4055–4070, 2025 | arXiv 2509.18706 |
| `2019_Saito_MME_SemiSupervisedDA_ICCV.pdf` | Saito, Kim, Sclaroff, Darrell, Saenko. Semi-supervised Domain Adaptation via Minimax Entropy. ICCV 2019 | arXiv 1904.06487 |

## 取得できなかった論文

- Schuller, Zhang, Weninger, Rigoll. Selecting Training Data for Cross-Corpus Speech Emotion Recognition: Prototypicality vs. Generalization. 2011．TUM のリポジトリ（mediatum.ub.tum.de/doc/1107310/109056.pdf）がスクリプトからの取得を拒む．ブラウザで開けば落とせる可能性がある．
- Ahn, Rana, Busso, Rajapakse. IEEE TAC 16(3):1581–1591, 2025．公開版を探していない．
- 中核の理論の Ben-David ほか 2010 は，リポジトリの最上位に `A theory of learning from different domains.pdf` として置いてある．
