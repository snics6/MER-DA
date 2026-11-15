# 参考文献のキーの一覧（2026-10-27）

本文の執筆者は，この表で `references.bib` のキーを引く．書誌の出所と確かめていない項目は，`references.bib` の各項目の直前のコメントに書いてある．

- **水準**：◎ 本文を読んだ（多くは arXiv 版．掲載版との差は未確認），○ 抄録・出版社の頁・会議の頁で確かめた，△ 書誌だけ確かめた．
- **引く位置**：「段落 n」は `../../reports/TAC 論文 先行研究の精読と穴.md` の「関連研究の節」の段落 1〜7．「表：〇〇」は同じ報告の「主張ごとの置き場所」の表の行．「1.11」は `../../docs/related-work.md` 1.11 節の比べる相手（E61 の標準手法）．
- **下書きの仮キー**：`sections/*.tex` が `\cite{CITE:...}` で引いている仮のキー．執筆者は仮のキーを左端のキーに置き換える．
- 下書きの仮キーのうち `CITE:kuzborskij2017` と `CITE:li2018l2sp` には対応する項目が無い．この 2 本は今回の対象（報告の段落 1〜7 と表，1.11 節の 9 本）に入っていなかった．使うなら書誌を確かめてから足す．
- 3 節用の 21 本（busso2008iemocap ほか）は表に入れていない．ただし関連研究でも引く 5 本（feng2023fewshot，gideon2021addog，kumar2022lpft，gulrajani2021domainbed，bendavid2010theory）は「既存」と書いて載せた．

## 段落 1：少数のターゲットラベルを使うコーパス間の感情認識

| キー | 文献 | 掲載先 | 水準 | 引く位置 | 下書きの仮キー |
|---|---|---|---|---|---|
| he2025m4ser | He ほか 2025，M4SER | IEEE Trans. Audio, Speech and Language Processing 33 | ◎ | 段落 1，表：設定 | CITE:m4ser2025 |
| zaidi2024enhancing | Zaidi, Latif, Qadir 2024，MDAT（掲載版の題は Enhancing Cross-Language Multimodal Emotion Recognition…） | IEEE Open J. Computer Society 5 | ◎ | 段落 1，表：設定 | CITE:mdat2024 |
| feng2023fewshot | Feng & Chaspari 2023（既存） | IEEE TAC 14(2) | 既存（△） | 段落 1，表：部品 1・設定 | CITE:feng2023 |
| gideon2021addog | Gideon ほか 2021，ADDoG（既存） | IEEE TAC 12(4) | 既存（△） | 段落 1，表：部品 3・設定 | CITE:gideon2021addog |
| chen2023prefinetuning | Chen & Yu 2023，Pre-Finetuning for Few-Shot Emotional Speech Recognition | Interspeech 2023 | ◎ | 段落 1，表：設定 | CITE:chen2023 |
| li2021acted | Li, Kim, Kuo, Narayanan 2021，Acted vs. Improvised | Interspeech 2021 | ◎ | 段落 1，表：設定 | CITE:li2021 |
| ahn2025multitask | Ahn, Rana, Busso, Rajapakse 2025，Multitask Transformer for Cross-Corpus SER | IEEE TAC 16(3) | ○ | 段落 1，表：設定 | CITE:ahn2025 |
| xu2024crosstask | Xu, Jiang, Wu 2024，CTIAL | IEEE TAC 15(3) | ◎ | 段落 1・7，表：命題 4 | CITE:ctial2024 |
| dong2026test | Dong, Jia, Dang 2026，Test Time Adaptation for SER | ICASSP 2026 | ◎ | 段落 1，表：設定（書いてはいけない言い方） | CITE:dong2026 |
| zhang2026voxemo | Zhang, Chou, Narayanan, Hain 2026，VoxEmo | arXiv:2603.08936 | ◎（一部） | 関連研究の節の「初めて」の段落（書いてはいけない言い方） | — |
| schuller2010crosscorpus | Schuller ほか 2010，Cross-Corpus Acoustic Emotion Recognition | IEEE TAC 1(2) | △ | 表：部品 1 | CITE:schuller2010 |

## 段落 2：線形の頭の誤差の分解と少数標本の理論

| キー | 文献 | 掲載先 | 水準 | 引く位置 | 下書きの仮キー |
|---|---|---|---|---|---|
| okamoto1963asymptotic | Okamoto 1963，An Asymptotic Expansion for the Distribution of the LDF | Ann. Math. Statist. 34(4) | △ | 段落 2，表：主張 A | CITE:okamoto1963 |
| shutoh2010asymptotic | Shutoh 2010，Okamoto の展開の再掲を含む技術報告 | 広島統計研究グループ 技術報告 10-04 | ◎（1〜2 節） | 表：主張 A（Okamoto の式を読んだ出所） | — |
| raudys1991small | Raudys & Jain 1991，Small sample size effects… | IEEE TPAMI 13(3) | ○ | 段落 2 | — |
| raudys1998intrinsic | Raudys 1998，Intrinsic dimensionality and small sample properties of classifiers | Kybernetika 34(4) | ◎ | 段落 2・3，表：主張 A・主張 B | CITE:raudys1998 |
| dobriban2018highdimensional | Dobriban & Wager 2018，High-dimensional asymptotics of prediction | Ann. Statist. 46(1) | ◎（arXiv 版） | 段落 2・3，表：主張 A・主張 B | CITE:dobriban2018 |
| desilva2023value | De Silva ほか 2023，The Value of Out-of-Distribution Data | ICML 2023（PMLR 202） | ◎ | 段落 2・4，表：主張 A・主張 C | CITE:desilva2023 |
| rosenfeld2022domainadjusted | Rosenfeld, Ravikumar, Risteski 2022，DARE | arXiv:2202.06856v2 | ◎ | 段落 2，表：主張 A・部品 1 | CITE:dare2022 |
| kumar2022lpft | Kumar ほか 2022，LP-FT（既存） | ICLR 2022 | 既存（○） | 段落 2，表：部品 2 | CITE:kumar2022lpft |
| kini2021labelimbalanced | Kini ほか 2021，Label-Imbalanced and Group-Sensitive Classification | NeurIPS 2021 | ◎ | 段落 2（罰則が有限である条件を書く位置） | CITE:kini2021 |
| ha2026labeled | Ha & Chen 2026，When few labeled target data suffice | arXiv:2507.14661v2 | ◎ | 段落 2・3，表：主張 B・部品 2 | CITE:hachen2026 |
| zhang2026transfer | Zhang, Fan, Luo, Hu 2026，Transfer Learning for LDA with a Shared Classification Signal | arXiv:2607.06936 | ◎（前半） | 段落 2・3，表：主張 A・主張 B | — |
| luthra2026directional | Luthra, Salunkhe, Galanti 2026，Directional Neural Collapse | arXiv:2603.03530 | ◎ | 段落 2，表：主張 A | — |

## 段落 3：ソースの価値の換算

| キー | 文献 | 掲載先 | 水準 | 引く位置 | 下書きの仮キー |
|---|---|---|---|---|---|
| hernandez2021scaling | Hernandez ほか 2021，Scaling Laws for Transfer | arXiv:2102.01293 | ◎ | 段落 3，表：主張 B | CITE:hernandez2021 |
| lescao2021data | Le Scao & Rush 2021，How Many Data Points is a Prompt Worth? | NAACL-HLT 2021 | ◎ | 段落 3，表：主張 B | CITE:lescao2021 |
| wiesenfarth2020quantification | Wiesenfarth & Calderazzo 2020，effective current sample size | Biometrics 76(1) | ○ | 段落 3，表：主張 B | CITE:wiesenfarth2020 |
| hanneke2019value | Hanneke & Kpotufe 2019，On the Value of Target Data in Transfer Learning | NeurIPS 2019 | ◎（arXiv 版） | 段落 3，表：主張 B | CITE:hanneke2019 |

段落 3 で引く Ha & Chen，Raudys 1998，Dobriban & Wager，Zhang 2026 は段落 2 の表にある．

## 段落 4：合併と複数ソース

| キー | 文献 | 掲載先 | 水準 | 引く位置 | 下書きの仮キー |
|---|---|---|---|---|---|
| bendavid2010theory | Ben-David ほか 2010（既存） | Machine Learning 79 | 既存（○） | 段落 4，表：主張 C | CITE:bendavid2010 |
| blitzer2007learning | Blitzer ほか 2007，Learning Bounds for Domain Adaptation | NIPS 2007 | ◎ | 段落 4，表：主張 C | CITE:blitzer2007 |
| pregibon1981logistic | Pregibon 1981，Logistic Regression Diagnostics | Ann. Statist. 9(4) | △ | 段落 4，表：主張 C | CITE:pregibon1981 |
| koh2019accuracy | Koh ほか 2019，On the Accuracy of Influence Functions for Measuring Group Effects | NeurIPS 2019 | ◎ | 段落 4，表：主張 C | CITE:koh2019 |
| tiomoko2021deciphering | Tiomoko, Tiomoko Ali, Couillet 2021，Deciphering and Optimizing Multi-Task Learning | ICLR 2021 | ○（ICLR 版） | 段落 4，表：主張 C・部品 1 | CITE:tiomoko2021 |
| tiomoko2020large | Tiomoko ほか 2020，同じ研究の長い版（題が違う） | arXiv:2009.01591 | ◎ | 段落 4（長い版の定理番号を引くとき） | — |
| yang2025precise | Yang ほか 2025，Precise High-Dimensional Asymptotics for Quantifying Heterogeneous Transfers | JMLR 26(113) | ◎（一部） | 段落 4，表：主張 C | CITE:yang2025 |
| chen2021domain | Chen & Bühlmann 2021，Domain adaptation under structural causal models | JMLR 22(261) | ◎（一部） | 段落 4，表：主張 C・部品 1 | — |
| zhu2026domain | Zhu ほか，Domain Generalization Under Posterior Drift（報告では Zhu 2025．arXiv v2 が 2026 年なのでキーは 2026） | arXiv:2510.04441v2 | ◎ | 段落 4，表：主張 C | CITE:zhu2025 |
| mansour2021theory | Mansour ほか 2021，A Theory of Multiple-Source Adaptation with Limited Target Labeled Data | AISTATS 2021（PMLR 130） | ◎（arXiv 版） | 段落 4 | — |
| saraomannelli2025biasinducing | Sarao Mannelli ほか 2025，Bias-inducing geometries | Phys. Rev. E 112(2) | ◎（一部） | 段落 4 | — |

段落 4 で引く De Silva 2023 は段落 2 の表にある．

## 段落 5：特権情報とモダリティ間の蒸留

| キー | 文献 | 掲載先 | 水準 | 引く位置 | 下書きの仮キー |
|---|---|---|---|---|---|
| vapnik2009new | Vapnik & Vashist 2009，Learning using privileged information | Neural Networks 22(5–6) | △ | 段落 5 | CITE:vapnik2009 |
| lopezpaz2016unifying | Lopez-Paz ほか 2016，Unifying distillation and privileged information | ICLR 2016 | ◎ | 段落 5，表：相補性・部品 4 | CITE:lopezpaz2016 |
| gupta2016cross | Gupta, Hoffman, Malik 2016，Cross Modal Distillation for Supervision Transfer | CVPR 2016 | ◎ | 段落 5 | CITE:gupta2016 |
| albanie2018emotion | Albanie ほか 2018，Emotion Recognition in Speech using Cross-Modal Transfer in the Wild | ACM MM 2018 | ◎ | 段落 5，表：部品 4 | CITE:albanie2018 |
| srinivasan2022representation | Srinivasan, Huang, Kirchhoff 2022，cross-modal conditional teacher-student training for SER | ICASSP 2022 | ◎ | 段落 5 | CITE:srinivasan2022 |
| hajavi2024audio | Hajavi & Etemad 2024，Distilling Video as Privileged Information | IEEE Trans. Artificial Intelligence 5(1) | ◎ | 段落 5 | CITE:hajavi2024 |
| pendyala2025leveraging | Pendyala, Morgado, Sethares 2025，LiSER | Interspeech 2025 | ◎ | 段落 5 | CITE:liser2025 |
| xue2021multimodal | Xue ほか 2021，MKE（Multimodal Knowledge Expansion） | ICCV 2021 | ◎ | 段落 5，表：部品 5 | CITE:xue2021 |
| du2021improving | Du ほか 2021，UMT（Uni-Modal Teachers） | arXiv:2106.11059 | ◎ | 段落 5，表：部品 2・部品 5 | CITE:du2021 |
| rakib2025g2d | Rakib & Bagavathi 2025，G²D | ICCV 2025 | ◎ | 段落 5，表：部品 2・部品 5 | CITE:g2d2025 |
| pinitas2025privileged | Pinitas ほか 2025，PriCon | ICMI 2025 | ◎ | 段落 5 | — |
| jaritz2020xmuda | Jaritz ほか 2020，xMUDA | CVPR 2020 | ◎ | 段落 5，表：部品 5 | CITE:xmuda2020 |
| wu2023crossmodal | Wu ほか 2023，BFtD（Bidirectional Fusion-then-Distillation） | ACM MM 2023 | △ | 段落 5，表：部品 5 | — |
| wu2025fusionthendistillation | Wu ほか 2025，FtD++ | IEEE TCSVT 35(9) | ○（要約） | 段落 5，表：部品 5 | — |
| liu2026dynamic | Liu ほか 2026，DAMR | Electronics 15(20) | ○ | 段落 5，表：部品 5 | — |
| huang2021makes | Huang ほか 2021，What Makes Multi-Modal Learning Better than Single (Provably) | NeurIPS 2021 | ◎ | 段落 5，表：相補性 | — |

## 段落 6：学び直さない先生の弱いラベル，クラス均衡，出発点への罰則，設定の選択

| キー | 文献 | 掲載先 | 水準 | 引く位置 | 下書きの仮キー |
|---|---|---|---|---|---|
| phoo2021selftraining | Phoo & Hariharan 2021，STARTUP | ICLR 2021 | ◎ | 段落 6，表：部品 4 | CITE:startup2021 |
| zanella2024boosting | Zanella, Gérin, Ben Ayed 2024，TransCLIP | NeurIPS 2024 | ◎ | 段落 6，表：部品 3・部品 4 | CITE:transclip2024 |
| hatano2024multimodal | Hatano ほか 2024，MM-CDFSL | ECCV 2024（LNCS） | ◎ | 段落 6，表：部品 3・部品 4 | CITE:mmcdfsl2024 |
| huang2022unsupervised | Huang, Chu, Wei 2022，UPL | arXiv:2204.03649v2 | ◎ | 段落 6，表：部品 4 | CITE:upl2022 |
| xie2020selftraining | Xie ほか 2020，Noisy Student | CVPR 2020 | ◎ | 段落 6，表：部品 4 | CITE:noisystudent2020 |
| lichtenstein2020tafssl | Lichtenstein ほか 2020，TAFSSL（trans-mean-sub の出所） | ECCV 2020（LNCS） | △ | 段落 6，表：部品 1 | — |
| hu2021leveraging | Hu, Gripon, Pateux 2021，PT-MAP | ICANN 2021（LNCS） | ◎ | 段落 6，表：部品 1・部品 4 | CITE:ptmap2021 |
| liu2020prototype | Liu, Song, Qin 2020，BD-CSPN | ECCV 2020（LNCS） | ◎ | 段落 6，表：部品 1 | — |
| berthelot2022adamatch | Berthelot ほか 2022，AdaMatch | ICLR 2022 | ◎ | 段落 6，表：部品 4 | CITE:adamatch2022 |
| veilleux2021realistic | Veilleux ほか 2021，Realistic Evaluation of Transductive Few-Shot Learning | NeurIPS 2021 | ◎ | 段落 6，表：部品 4 | CITE:veilleux2021 |
| silvarodriguez2024closer | Silva-Rodríguez ほか 2024，CLAP | CVPR 2024 | ◎ | 段落 6，表：部品 2・部品 3 | CITE:clap2024 |
| gulrajani2021domainbed | Gulrajani & Lopez-Paz 2021，DomainBed（既存） | ICLR 2021 | 既存（○） | 段落 6，表：部品 3 | CITE:gulrajani2021domainbed |

## 段落 7：弱いラベルの得の理論と能動学習

| キー | 文献 | 掲載先 | 水準 | 引く位置 | 下書きの仮キー |
|---|---|---|---|---|---|
| charikar2024quantifying | Charikar ほか 2024，Quantifying the Gain in Weak-to-Strong Generalization | NeurIPS 2024 | ◎ | 段落 7，表：相補性 | CITE:charikar2024 |
| dong2025discrepancies | Dong ほか 2025，Discrepancies are Virtue | ICML 2025（PMLR 267） | ◎（arXiv 版） | 段落 7，表：相補性 | CITE:dong2025 |
| xue2025representations | Xue, Li, Mirzasoleiman 2025，Representations Shape Weak-to-Strong Generalization | ICML 2025（PMLR 267） | ◎ | 段落 7，表：命題 1・相補性 | CITE:xue2025 |
| kumar2020understanding | Kumar, Ma, Liang 2020，Understanding Self-Training for Gradual DA | ICML 2020（PMLR 119） | ◎（arXiv 版） | 段落 7，表：相補性・部品 2 | CITE:kumar2020 |
| hacohen2022active | Hacohen, Dekel, Weinshall 2022，Active Learning on a Budget | ICML 2022（PMLR 162） | ◎ | 段落 7，表：命題 1・命題 4 | CITE:hacohen2022 |
| chaloner1984optimal | Chaloner 1984，Optimal Bayesian Experimental Design for Linear Models | Ann. Statist. 12(1) | ◎（技術報告） | 段落 7，表：命題 4・部品 2 | CITE:chaloner1984 |
| chaloner1995bayesian | Chaloner & Verdinelli 1995，Bayesian Experimental Design: A Review | Statistical Science 10(3) | ◎（著者版） | 段落 7，表：命題 4 | CITE:chaloner1995 |
| sloman2022characterizing | Sloman ほか 2022，robustness of Bayesian adaptive designs to active learning bias | arXiv:2205.13698v2 | ◎（一部） | 段落 7，表：命題 4 | — |
| kimeldorf1971results | Kimeldorf & Wahba 1971，Some results on Tchebycheffian spline functions | J. Math. Anal. Appl. 33(1) | ○ | 段落 7，表：命題 1 | CITE:kimeldorf1971 |
| scholkopf2001generalized | Schölkopf, Herbrich, Smola 2001，A Generalized Representer Theorem | COLT 2001（LNCS） | ○ | 段落 7，表：命題 1 | CITE:scholkopf2001 |

段落 7 で引く CTIAL は段落 1 の表（xu2024crosstask）にある．

## 比べる相手（related-work.md 1.11 節．E61 の標準手法）

| キー | 文献 | 掲載先 | 水準 | 引く位置 | 下書きの仮キー |
|---|---|---|---|---|---|
| hinton2015distilling | Hinton, Vinyals, Dean 2015，Distilling the Knowledge in a Neural Network | arXiv:1503.02531 | ○ | 1.11 | CITE:hinton2015 |
| grandvalet2004semisupervised | Grandvalet & Bengio 2004，Semi-supervised Learning by Entropy Minimization | NIPS 2004 | ○ | 1.11 | CITE:grandvalet2004 |
| saerens2002adjusting | Saerens, Latinne, Decaestecker 2002，Adjusting the Outputs of a Classifier to New a Priori Probabilities | Neural Computation 14(1) | ○ | 1.11 | CITE:saerens2002 |
| blum1998combining | Blum & Mitchell 1998，co-training | COLT 1998 | △ | 1.11 | CITE:blum1998 |
| sindhwani2005coregularization | Sindhwani, Niyogi, Belkin 2005，A Co-Regularization Approach to Semi-supervised Learning with Multiple Views | ICML 2005 Workshop on Learning with Multiple Views | ○ | 1.11 | CITE:sindhwani2005 |
| lee2013pseudolabel | Lee 2013，Pseudo-Label | ICML 2013 Workshop on Challenges in Representation Learning | △ | 1.11 | CITE:lee2013 |
| prabhu2021active | Prabhu ほか 2021，CLUE | ICCV 2021 | ○ | 1.11 | — |
| fu2021transferable | Fu, Cao, Wang, Long 2021，TQS | CVPR 2021 | ○ | 1.11 | — |
