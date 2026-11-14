# 引用の照合：7・8・9 節（`DAI/paper/ja/07_本番の評価.md`，`08_3条件.md`，`09_中間層.md`）

- 照合役：paper-citations（2026-11-15）．
- 対象の版：下書きは作業の途中の 01:58 に書き換わった．02:02 に読み直した版を対象にした．書き換えで 9 節から Pasad ほか 2023 と Yang ほか 2024 の引用が消えたので，その 2 本は 9 節の照合から外した．Pasad は 8 節に残っている．
- 照らした記録：
  - `DAI/paper/notes/引用の候補_2026-11-14.md`（以下「候補」）
  - `research_notes/有名な論文_2026-11-07/半教師ありDA.md`（F-SSDA）
  - `research_notes/先行研究_層の重みをラベル無しで選ぶ_2026-11-09/report.md`（LW）
  - `research_notes/疑似ラベル 情報の持ち込み方 先行研究/1_ラベル無しデータの構造.md`
  - `research_notes/TAC 論文の先行研究と書き方/method_components_precedents.md`
  - `research_notes/先行研究_ソースの発話の選び方_2026-11-14/report.md`
  - `DAI/checks/DAI-14/compare.md`・`review_affect.md`
  - `DAI/docs/実験1_設計.md`・`実験2_設計.md`
- 記録に書誌が無いものは，一般の語の検索で書誌だけを確かめた（検索した語は末尾に書いた）．

## 判定

**直してから通す．**
- 本文が文献に言わせている内容は，ほとんど記録と合う．
- 直す点は次の 3 件である．
  - C8-5：Pasad ほか 2023 の言い過ぎ．
  - C8-1：Wei ほか 2021 が候補の「削る案」に入っている．確認事項_07-09.md の 6 は「削る案には入っていない」と書いており，記録と食い違う．
  - C7-10：xMUDA の bib の頁が違う．
- `[@要追加]` の 7 本のうち，6 本は記録か検索で書誌を確かめた．Zhou ほか 2004 は記録に書誌が無く，検索で確かめた．bib に足す項目は下の表に書いた．

## 7 節

| 番号 | 原稿の文 | 文献 | 記録の場所 | 判定 | 直し方 |
|---|---|---|---|---|---|
| C7-1 | 形 (b) は Hinton ほか の蒸留の損失 | hinton2015distilling | 候補 #21（F-KD ◎） | 合う | bib は arXiv 1503.02531，NIPS 2014 DL Workshop で記録と一致 |
| C7-2 | τ = 2 は Albanie ほか の値 | albanie2018 | 候補 #17，実験1_設計.md 122 行（「教師と生徒の両方に温度 2．3 節と 5 節．PDF で確認」） | 合う | bib は ACM MM 2018，292–301，DOI で記録と一致 |
| C7-3 | 一致した発話だけの疑似ラベル | shi2024audioguided | 候補 #23（AlbB ◎．D1 は「原文は読んでいない」） | 合う | 本文は手法の出典としてだけ引いている．題は記録に無いので検索で確かめた（"Audio-Guided Fusion Techniques for Multimodal Emotion Analysis"，Shi & Gao，arXiv 2409.05007，MRAC '24 で発表）．bib に掲載先 MRAC '24（ACM MM 2024 のワークショップ）を足すかを決める．付録で τ = 0.9 を書くなら，原文で確かめてから書く |
| C7-4 | 自分の予測と混ぜる方法 | zhang2021combining | 候補 #22 | 合う | bib は KBS 229:107340 で記録と一致．付録で α = 0.2 としきい値 0.9 を書くなら，記録の「元の表 5 で確かめる」を先に済ませる |
| C7-5 | 柔らかい共訓練（Blum & Mitchell の考えを著者らが変形したもの） | blum1998 | 候補 #25，104 行（「著者らの変形と書く」） | 合う | 記録の指示どおりに書いている |
| C7-6 | 共正則化 | sindhwani2005coreg，sindhwani2008 | 候補 #26（「本文未読」．RW:655 は「引く前に本文を確かめる」） | 合う（確認水準が低い） | 本文は手法の名前の出典としてだけ引いており，細部は言わせていない．検索の結果，2005 年の題は bib と一致し，頁は 74–79 である（ICML 2005 Workshop on Learning with Multiple Views）．bib に pages = {74--79} を足す |
| C7-7 | 自己学習 | lee2013pseudolabel | 候補 #24（F-SSDA ◎） | 合う | — |
| C7-8 | エントロピー最小化 | grandvalet2004 | 候補 #30（F-SSDA ◎．「NIPS 17, pp. 529–536, 2005（RW は NIPS 2004 と書く）」） | 合う | bib に pages = {529--536} を足す．年は 2004（会議の年）か 2005（刊行の年）のどちらかに決め，参考文献の全体で統一する |
| C7-9 | CORAL，DANN，MME | sun2016coral，ganin2016dann，saito2019mme | 候補 #27〜#29 | 合う | 「凍結した特徴の上の線形の写像と分類器に損失の項を入れた版」は，候補 #29 の「分類器の側に限った変種であると断る」に合う |
| C7-10 | xMUDA 型の相互の模倣 | jaritz2020xmuda | 候補 #20（Q1 全文） | 合う（本文）．bib が違う | bib の pages = {12602--12611} は，CVF の公開版の 12605–12614 と違う．pages を直す |
| C7-11 | IM | [@要追加: Liang ほか 2020; Boudiaf ほか 2020] | 実験2_設計.md 41 行「SHOT の IM，Boudiaf ほか 2020 の TIM」．SHOT は F-SSDA 409・770 行（◎）．TIM は method_components_precedents.md 83 行 | 記録あり | bib に足す：Liang J, Hu D, Feng J. Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation. ICML 2020, PMLR 119:6028–6039．Boudiaf M, Ziko I, Rony J, Dolz J, Piantanida P, Ben Ayed I. Transductive Information Maximization for Few-Shot Learning. NeurIPS 2020（arXiv 2008.11297．著者の並びは記録に無いので，arXiv で確かめてから足す） |
| C7-12 | T3A | [@要追加: Iwasawa & Matsuo 2021] | 先行研究_ソースの発話の選び方 report 71 行（NeurIPS 2021．題なし） | 記録あり（題は検索で確かめた） | bib に足す：Iwasawa Y, Matsuo Y. Test-Time Classifier Adjustment Module for Model-Agnostic Domain Generalization. NeurIPS 2021 |
| C7-13 | ラベル伝播 | [@要追加: Zhou ほか 2004; Iscen ほか 2019] | Iscen は 1_ラベル無しデータの構造.md 40〜42 行（◎．CVPR 2019，arXiv 1904.04717）．Zhou は同 325 行で「原文を読み直していない」とあり，書誌は記録に無い | Iscen は記録あり．Zhou は記録が無い（検索で書誌を確かめた） | bib に足す：Iscen A, Tolias G, Avrithis Y, Chum O. Label Propagation for Deep Semi-supervised Learning. CVPR 2019．Zhou D, Bousquet O, Lal TN, Weston J, Schölkopf B. Learning with Local and Global Consistency. NIPS 16, pp. 321–328（2003 年の会議．刊行は 2004 年）．本文は手法の出典として引くだけなので，原文を読んでいなくてもこの使い方でよい |

## 8 節

| 番号 | 原稿の文 | 文献 | 記録の場所 | 判定 | 直し方 |
|---|---|---|---|---|---|
| C8-1 | 条件 2 は，疑似ラベルを付けるモデルがターゲットで十分に正確であることを仮定する自己学習の理論に対応する | [@要追加: Wei ほか 2021] | 候補 #34（◎．「それなりに正確なことを仮定する」）．F-SSDA 649 行（仮定 4.1：各クラスで疑似ラベルの誤りの割合が 1/3 未満．あわせて expansion の仮定） | 内容は合う．**候補の削る案に入っている** | 候補 283 行の「削る 12 本」に #34 がある（理由：条件 2 は TelME と Xue で書ける）．確認事項_07-09.md の 6 の「削る案には入っていない」は記録と違う．残すなら削る案から外し，bib に足す：Wei C, Shen K, Chen Y, Ma T. Theoretical Analysis of Self-Training with Deep Networks on Unlabeled Data. ICLR 2021（arXiv 2010.03622）．外すなら，この文の前半を消して TelME の文から始める |
| C8-2 | 感情認識で弱い非言語のモデルの出力を学習の目標にすると学ぶ側のモデルが悪くなるという報告 | yun2024telme | 候補 #36（AlbC ◎．引用文 "treating a weak non-verbal model as the teacher impairs student performance"） | 合う | 言い換えは引用文の意味を保っている．正式な題は記録に無い（候補 232 行）．bib の題 "TelME: Teacher-leading Multimodal Fusion Network for Emotion Recognition in Conversation" は ACL Anthology 2024.naacl-long.5 の題と一致する |
| C8-3 | 標本が少ないと融合が最良の単独のモダリティを下回りうることは Huang ほか が示した | huang2021 | 候補 #35（「本文を読んだ．付録は未読」．IEMOCAP で学習の標本を減らすと 3 モダリティが言葉だけに負ける） | 合う | この内容は理論の結果ではなく IEMOCAP の実験の結果である．「示した」を「IEMOCAP の実験で報告した」にすると，記録の範囲に合う |
| C8-4 | 疑似ラベルを付けるモデルの正解率が高くても…モダリティに共通の決め手の特徴の割合が成否を決めるという Xue ほか の仮説…人工データでだけ動かした | xue2023mfh | 候補 #33（Q1 全文．表 1，人工データ） | 合う | — |
| C8-5 | 自己教師ありの音声モデルの中間の層ほど単語の情報が多いという報告 | [@要追加: Pasad ほか 2023] | 候補に無い．DAI-14 の compare.md 113 行（「WavLM・HuBERT の単語の内容は中間の層で最も多い．要旨で確かめた．本文は読んでいない」）と review_affect.md 44 行（「語の情報がどの層に多いかはモデルによって違う」） | 言い過ぎ | 「ほど」は層が中間に近いほど単語の情報が増えるという単調な関係を言わせている．記録と要旨（"phonetic and word information peaks in different layers depending on the model"）が言うのは，情報が最も多い層がモデルごとに違い，WavLM・HuBERT では中間の層だということだけである．「WavLM や HuBERT では単語の情報が中間の層で最も多いという報告」に直す．bib に足す：Pasad A, Shi B, Livescu K. Comparative Layer-wise Analysis of Self-supervised Speech Models. ICASSP 2023，DOI 10.1109/ICASSP49357.2023.10096149．記録の間で年が食い違う（実験6_下調べ_設計.md は「Pasad ほか 2024」と書く．arXiv 2307.00162 の雑誌版を指すとみられる）．どちらを引くかを決める．候補の表に 1 行足す |
| C8-6 | 微調整した音声モデルが言葉の感情の手がかりを拾うという報告 | wagner2023 | 候補 #60（「本文（問いと答えの形式の節）を読んだ」．音声の特徴は言葉の情報を暗黙に含む） | 合う | 巻・頁は記録に無い．bib の 45(9):10745–10759 は確かめていない．DOI 10.1109/TPAMI.2023.3263585 で確かめる |
| C8-7 | Albanie ほか は顔の分類器の判断を声のモデルへラベル無しで移したが，声のモデルそのものを学習しており，設定が違う | albanie2018 | 候補 #17，DAI-14 の review_affect_v2.md 44 行 | 合う | — |

## 9 節

| 番号 | 原稿の文 | 文献 | 記録の場所 | 判定 | 直し方 |
|---|---|---|---|---|---|
| C9-1 | 層の重みつき和は SUPERB が…定めた方式 | yang2021superb | 候補 #37（◎） | 合う | — |
| C9-2 | WavLM の原論文も感情認識の評価にこの方式を使う | chen2022wavlm | 候補 #57（◎），LW 表（表 I の感情認識 70.62．SUPERB の方式） | 合う | — |
| C9-3 | IEMOCAP では，最終層だけを使う場合の 58.5% に対して全層の重みつき和で 67.2% になったという報告がある | pepino2021 | 候補 #38（◎），LW 46 行（wav2vec 2.0 base を凍結．重みつき和の後に全結合 2 層．セッションを外す 5 分割．感情ごとの再現率の平均） | 合う | 数は合う．モデルが書かれていないので，この節の WavLM の値と読み違えられうる．「wav2vec 2.0 base を凍結した特徴で」と足す．bib の題 "Emotion Recognition from Speech Using wav2vec 2.0 Embeddings" は Interspeech 2021 の題と一致する．候補 232 行の「題は記録に無い」は解消した |

## 位置づけに要る文献の欠け

- 無し．7〜9 節は MER の中の位置づけを Albanie 2018，TelME，Pepino，SUPERB で書けている．

## 検索した語（書誌の確認だけ．内容の主張には使っていない）

- "Iwasawa Matsuo Test-Time Classifier Adjustment Module for Model-Agnostic Domain Generalization NeurIPS 2021"
- "Zhou Bousquet Lal Weston Schölkopf Learning with Local and Global Consistency NIPS"
- "Pasad Shi Livescu Comparative layer-wise analysis of self-supervised speech models ICASSP 2023"
- "arXiv 2409.05007 Audio-Guided Fusion Techniques for Multimodal Emotion Analysis Shi Gao MRAC"
- "xMUDA Cross-Modal Unsupervised Domain Adaptation 3D Semantic Segmentation CVPR 2020 pages"
- "Sindhwani Niyogi Belkin A Co-Regularization Approach to Semi-supervised Learning with Multiple Views ICML 2005 workshop"
