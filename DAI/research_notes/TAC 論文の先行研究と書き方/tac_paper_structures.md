# TAC（IEEE Transactions on Affective Computing）2021〜2026 年の近い論文 13 本の構成と書き方

凡例：確かめた水準は ◎（本文を読んだ．arXiv 版を含む），○（抄録を読んだ．目次は見ていないものは「抄録のみ」と書く），△（検索結果の抜き書きだけ）．
本文の語数・図表数・参考文献数・式の数は，arXiv 版の PDF を pdftotext で文字にして数えた概数である．2 段組みの文字化で図表の文字が本文に混ざるため，節ごとの語数の比は ±数ポイントの誤差を含む．掲載版の頁数は OpenAlex に記録された TAC の頁番号（参考文献と著者紹介を含む）から求めた．
作業日は 2026-10-27．IEEE Xplore の本文と目次には到達できなかった（curl は空の応答，WebFetch も空のページを返した）．

## 1. 対象論文の書誌と確かめた水準

### Takeaway
依頼の 9 本と追加候補 5 本の計 14 本の書誌を OpenAlex と arXiv で確かめた．TAC 掲載は 13 本で，MERBench は TPAMI 掲載のため除外した．本文を読めたのは 8 本（すべて arXiv 版），残る 5 本は抄録だけを読んだ．

### Cited Findings
| 記号 | 題名 | 著者 | 巻(号):頁，年（早期公開日） | DOI | 種類 | 水準 |
|---|---|---|---|---|---|---|
| P1 | Few-Shot Learning in Emotion Recognition of Spontaneous Speech Using a Siamese Neural Network With Adaptive Sample Pair Formation | Kexin Feng, Theodora Chaspari | 14(2):1627–1633, 2023（2021-09-03） | 10.1109/TAFFC.2021.3109485 | 手法（少数ショット，音声，演技→自発） | ◎ arXiv 2109.02915 |
| P2 | Improving Cross-Corpus Speech Emotion Recognition with Adversarial Discriminative Domain Generalization (ADDoG) | John Gideon, Melvin G. McInnis, Emily Mower Provost | 12(4):1055–1068, 2021（2019-05-14） | 10.1109/TAFFC.2019.2916092 | 手法（コーパス間，音声） | ◎ arXiv 1903.12094 |
| P3 | Multitask Transformer for Cross-Corpus Speech Emotion Recognition | Chung-Soo Ahn, Rajib Kumar Rana, Carlos Busso, Jagath Chandana Rajapakse | 16(3):1581–1591, 2025（2025-01-07） | 10.1109/TAFFC.2025.3526592 | 手法（コーパス間，音声＋言葉の判定の融合） | ○ 抄録のみ |
| P4 | Phonetically-Anchored Domain Adaptation for Cross-Lingual Speech Emotion Recognition | Shreya G. Upadhyay, Luz Martinez-Lucas, William F. Katz, Carlos Busso, Chi-Chun Lee | 16(3):1631–1645, 2025（2025-01-15） | 10.1109/TAFFC.2025.3530105 | 分析＋手法（言語間，音声） | ○ 抄録のみ |
| P5 | Video-Based Cross-Domain Emotion Recognition via Sample-Graph Relations Self-Distillation | Dongliang Chen, Guihua Wen, Pengcheng Wen, Huihui Li, Pei Yang, Chuyun Chen, Bao Wang | 17(3):3154–3169, 2026（2026-04-02） | 10.1109/TAFFC.2026.3680134 | 手法（顔の動画，ドメイン間，ターゲットのラベルなし） | ○ 抄録のみ |
| P6 | How many raters do we need? Analyses of uncertainty in estimating ambiguity-aware emotion labels | Jingyao Wu, Ting Dang, Vidhyasaharan Sethu, Eliathamby Ambikairajah | early access, pp.1–15, 2025（正確な日付は要確認） | 10.1109/TAFFC.2025.3616071 | 分析（注釈者の人数） | ○ 抄録のみ |
| P7 | Rethinking Emotion Annotations in the Era of Large Language Models | Minxue Niu, Yara El-Tawil, Amrit Romana, Emily Mower Provost | 16(4):2668–2679, 2025（2025-06-30） | 10.1109/TAFFC.2025.3584775 | 分析＋手法（注釈，テキスト） | ◎ arXiv 2412.07906 |
| P8 | Why Pre-trained Models Fail: Feature Entanglement in Multimodal Depression Detection | Xiangyu Zhang, Beena Ahmed, Julien Epps | early access, pp.1–17, 2026 | 10.1109/TAFFC.2026.3711426 | 分析＋手法（抑うつ，音声＋言葉） | ◎ arXiv 2503.06620v2 |
| （除外） | MERBench: A Unified Evaluation Benchmark for Multimodal Emotion Recognition | Zheng Lian ほか | **TPAMI** 48(5):5793–5810, 2026 | 10.1109/TPAMI.2026.3653457 | TAC ではない | ○ |
| P10 | Self Supervised Adversarial Domain Adaptation for Cross-Corpus and Cross-Language Speech Emotion Recognition | Siddique Latif, Rajib Rana, Sara Khalifa, Raja Jurdak, Björn W. Schuller | 14(3):1912–1926, 2023（2022-04-13） | 10.1109/TAFFC.2022.3167013 | 手法（コーパス間・言語間，音声） | ◎ arXiv 2204.08625 |
| P11 | EmoNet: A Transfer Learning Framework for Multi-Corpus Speech Emotion Recognition | Maurice Gerczuk, Shahin Amiriparian, Sandra Ottl, Björn W. Schuller | 14(2):1472–1487, 2023（2021-12-14） | 10.1109/TAFFC.2021.3135152 | 手法（26 コーパスの転移，音声） | ◎ arXiv 2103.08310 |
| P12 | Versatile Audio-Visual Learning for Emotion Recognition | Lucas Goncalves, Seong-Gyun Leem, Wei-Cheng Lin, Berrak Sisman ほか（全著者は要確認） | 16(1):306–318, 2025（2024-07-25） | 10.1109/TAFFC.2024.3433386 | 手法（音声＋顔，コーパス内） | ◎ arXiv 2305.07216 |
| P13 | Benchmarking Cross-Domain Audio-Visual Deception Detection | Xiaobao Guo, Zitong Yu, Nithish Muthuchamy Selvaraj, Bingquan Shen ほか | 17(3):2625–2642, 2026（2026-06-02） | 10.1109/TAFFC.2026.3699375 | ベンチマーク＋手法（嘘の検出，音声＋顔，ドメイン間） | ◎ arXiv 2405.06995 |
| P14 | Adversarial Domain Generalized Transformer for Cross-Corpus Speech Emotion Recognition | Yuan Gao, Longbiao Wang, Jiaxing Liu, Jianwu Dang, Shogo Okada | 15(2):697–708, 2024（2023-06-29） | 10.1109/TAFFC.2023.3290795 | 手法（コーパス間，音声） | ○ 抄録のみ |

- P1〜P8，P10〜P14 の巻号・頁・早期公開日・著者は OpenAlex の記録による — [OpenAlex P1](https://api.openalex.org/works/doi:10.1109/taffc.2021.3109485)，[P2](https://api.openalex.org/works/doi:10.1109/taffc.2019.2916092)，[P3](https://api.openalex.org/works/doi:10.1109/taffc.2025.3526592)，[P4](https://api.openalex.org/works/doi:10.1109/taffc.2025.3530105)，[P5](https://api.openalex.org/works/doi:10.1109/taffc.2026.3680134)，[P6](https://api.openalex.org/works/doi:10.1109/taffc.2025.3616071)，[P7](https://api.openalex.org/works/doi:10.1109/taffc.2025.3584775)，[P8](https://api.openalex.org/works/doi:10.1109/taffc.2026.3711426)，[P10](https://api.openalex.org/works/doi:10.1109/taffc.2022.3167013)，[P11](https://api.openalex.org/works/doi:10.1109/taffc.2021.3135152)，[P12](https://api.openalex.org/works/doi:10.1109/taffc.2024.3433386)，[P13](https://api.openalex.org/works/doi:10.1109/taffc.2026.3699375)，[P14](https://api.openalex.org/works/doi:10.1109/taffc.2023.3290795)
- MERBench は OpenAlex で IEEE TPAMI 48(5):5793–5810（2026-01-12）と記録されており，TAC ではない — [OpenAlex MERBench](https://api.openalex.org/works/doi:10.1109/tpami.2026.3653457)，[arXiv 2401.03429](https://arxiv.org/abs/2401.03429)
- P2 の早期公開は 2019-05-14 で，号への収録は 2021 年の 12(4) である．2019 年に書かれた論文が 2021 年の号に載った形である — [OpenAlex P2](https://api.openalex.org/works/doi:10.1109/taffc.2019.2916092)
- P8 の著者名は OpenAlex では「X T Zhang」と記録されているが，arXiv 版では Xiangyu Zhang である — [arXiv 2503.06620](https://arxiv.org/abs/2503.06620)
- P3 には，同じ著者のうち 3 人（Ahn，Rajapakse，Rana）による arXiv 2310.04703 "Integrating Contrastive Learning into a Multitask Transformer Model for Effective Domain Adaptation" があるが，Busso が著者に入っておらず題名も違うため，P3 の本文としては扱わない — [arXiv 2310.04703](https://arxiv.org/abs/2310.04703)
- P4 の前身は ICASSP 2023 の "Phonetic Anchor-Based Transfer Learning to Facilitate Unsupervised Cross-Lingual Speech Emotion Recognition" である．検索結果によると TAC 版は NSF PAR で 2026-07-01 から公開予定と記されていたが，NSF PAR への接続が拒否されたため本文は確かめていない — [OpenAlex ICASSP 2023 版](https://doi.org/10.1109/ICASSP49357.2023.10095250)，[NSF PAR（検索結果）](https://par.nsf.gov/biblio/10441289)
- P5，P6 には arXiv 版や著者版が見つからなかった．OpenAlex の記録では P3〜P6 はすべて open access ではない（oa_status: closed） — [OpenAlex P5](https://api.openalex.org/works/doi:10.1109/taffc.2026.3680134)，[OpenAlex P6](https://api.openalex.org/works/doi:10.1109/taffc.2025.3616071)
- P13 は感情認識ではなく嘘の検出の論文であるが，TAC 掲載で，音声と顔のドメイン間の評価を主題にしているため，マルチモーダルのドメイン間評価の書き方の例として入れた — [arXiv 2405.06995](https://arxiv.org/abs/2405.06995)

### Inferences
- arXiv 版は掲載版と文言・図表・参考文献数が違うことがある（P1 は arXiv で参考文献 51 本，OpenAlex で 49 本．P7 は arXiv で 76 本，OpenAlex で 64 本）．構成の傾向を見る目的には足りるが，頁数は掲載版の頁番号で比べるのがよい．
- P12 の arXiv 版は TAC の書式ではない別の書式で組まれているため，頁数と見出しの書式は掲載版と違う．

### Gaps
- P3〜P6，P14 の節の見出し，図表数，統計検定，限界の書き方は確かめられなかった（IEEE Xplore に到達できず，著者版も見つからない）．
- P12，P13 の全著者は OpenAlex の先頭 4 人までしか確かめていない．
- 依頼の 10 番目の候補のうち「理論や統計モデルを前面に置いた TAC の論文」は，検索した範囲では見つからなかった（9 節参照）．

## 2. 節の見出しと順序，長さ（頁数，参考文献，図表）

### Takeaway
13 本の掲載版の頁数は 7〜18 頁で，中央値は 15 頁，平均は約 13.9 頁である．本文を読んだ 8 本はおおむね「序論 → 関連研究 → データ／方法 → 実験 → 結果 → 議論・結論」の順に並び，独立した Limitations の節を持つのは P7 の 1 本だけである．関連研究は本文の約 7〜22%，実験と結果の節は約 25〜67% を占める．

### Cited Findings
掲載版の頁数（OpenAlex の頁番号から計算．参考文献と著者紹介を含む）：P1 7 頁，P2 14 頁，P3 11 頁，P4 15 頁，P5 16 頁，P6 15 頁，P7 12 頁，P8 17 頁，P10 15 頁，P11 16 頁，P12 13 頁，P13 18 頁，P14 12 頁 — 1 節の OpenAlex の記録

| 記号 | 節の見出しと順序（arXiv 版） | 本文語数（参考文献の前まで） | 参考文献 | 図 | 表 | 節ごとの語数の比（概数） | 出典 |
|---|---|---|---|---|---|---|---|
| P1 | 1 Introduction / 2 Previous Work / 3 Data Description and Pre-Processing / 4 Proposed Methodology（4.1 SNN，4.2 MeL，4.3 MeL-S，4.4 MeL-S-ASPF，4.5 Experimental Design） / 5 Results / 6 Discussion & Conclusions | 約 5,200 | 51 | 5 | 1 | 序論 13%，先行研究 8%，データ 7%，方法＋実験計画 40%，結果 18%，議論・結論 13% | [arXiv 2109.02915](https://arxiv.org/abs/2109.02915) |
| P2 | 1 Introduction / 2 Related Works（domain adaptation，generalization など） / 3 Datasets / 4 Classification Models / 5 Experimental Design（5.1〜5.4 が実験 1〜4） / 6 Results / 7 Discussion and Conclusion | 約 10,800 | 56 | 8 | 2 | 序論 11%，関連研究 20%，データ 11%，モデル 26%，実験計画 10%，結果 15%，議論・結論 8% | [arXiv 1903.12094](https://arxiv.org/abs/1903.12094) |
| P7 | 1 Introduction / 2 Related Work / 3 Datasets / 4 GPT-4's Emotion Annotation Capability / 5 Feasibility of GPT-4 Aided Emotion Annotation / 6 Discussion / 7 Limitations and Future Work / 8 Conclusion / 9 Acknowledgments | 約 7,500 | 76 | 4 | 3 | 序論 12%，関連研究 13%，データ 3%，分析（4 節）33%，手法の検討（5 節）29%，議論 4%，限界 4%，結論 1% | [arXiv 2412.07906](https://arxiv.org/abs/2412.07906) |
| P8 | I Introduction / II Related Work / III Baseline System / IV Current Limitations and Discussion / V Information Separation Framework for …（見出しの後半は抽出できず） / VI Evaluating the Information Separation … / VII Large Language Model Results and Analysis / VIII Conclusion | 約 11,000 | 77 | 9 | 9 | 序論 6%，関連研究 9%，基準の系 17%，既存の限界の分析 9%，枠組み 11%，評価 24%，LLM の結果 24% | [arXiv 2503.06620](https://arxiv.org/abs/2503.06620) |
| P10 | 1 Introduction / 2 Related Work / 3 Proposed Model / 4 Experimental Setup / 5 Experiments and Evaluations / 6 Conclusions and Future Work | 約 8,500 | 92 | 4 | 8 | 序論 12%，関連研究 22%，モデル 15%，実験設定 15%，実験 31%，結論 4% | [arXiv 2204.08625](https://arxiv.org/abs/2204.08625) |
| P11 | 1 Introduction / 2 Related Works / 3 EmoSet – Collection of Speech Emotion Recognition Corpora / 4 Baselines / 5 Deep Learning Architectures / 6 Cross-Corpus Strategies / 7 Evaluation / 8 Conclusion and Outlook | 約 12,600 | 110 | 6 | 4 | 序論 6%，関連研究 18%，コーパス 13%，基準 4%，構造 17%，転移の方針 4%，評価 33%，結論 5% | [arXiv 2103.08310](https://arxiv.org/abs/2103.08310) |
| P12 | 1 Introduction / 2 Background / 3 Proposed Approach（3.5 Complexity Analysis を含む） / 4 Experimental Settings / 5 Results（5.3 Ablation Analysis，5.4 Shared Embedding Analysis を含む） / 6 Conclusions | 約 9,800 | 68 | 4 | 3 | 抽出できず（別書式のため） | [arXiv 2305.07216](https://arxiv.org/abs/2305.07216) |
| P13 | 1 Introduction / 2 Related Work / 3 Methodology（3.1〜3.3） / 4 Experiments（4.1 Databases and Metrics，4.2 Implementation Details，4.3 Cross-domain Testing with Unimodal Features，4.4 Domain-Simultaneous with Gradient Reversal，4.5 Cross-domain Testing with Multimodal Fusion，4.6 Network Details，4.7 Discussion） / 5 Conclusion | 約 14,600 | 58 | 4 | 11 前後 | 序論 9%，関連研究 7%，方法 16%，実験 67%，結論 1% | [arXiv 2405.06995](https://arxiv.org/abs/2405.06995) |

- P7 の掲載版は 12 頁で，本文は約 7,500 語である．P2 は 14 頁で約 10,800 語，P10 は 15 頁で約 8,500 語である — 上の表の出典と 1 節の OpenAlex の記録
- P7 は Limitations and Future Work を独立の 7 節（約 300 語）として，Discussion（6 節）と Conclusion（8 節）の間に置いている — [arXiv 2412.07906](https://arxiv.org/abs/2412.07906)
- IEEE Computer Society の著者向けの案内では，頁数の上限は要旨・参考文献・著者紹介を含めて数え，上限は雑誌ごとに異なると書かれている（検索結果の抜き書き，△） — [IEEE Computer Society Author Resources](https://www.computer.org/publications/author-resources/authors)
- Computer Society の MOPC 方針（2018 年の文書）では，Transactions の通常論文は 12 頁を超えた分に 1 頁 200 ドルの超過料金がかかり，投稿は 18 頁を超えてはならないと書かれている．TAC に今も同じ値が当てはまるかは確かめていない（検索結果の抜き書き，△） — [MOPC policy (PDF)](https://ieeecs-media.computer.org/assets/pdf/MOPC_policy.pdf)

### Inferences
- 掲載版の頁数の分布（中央値 15 頁）と，Computer Society の「12 頁を超えると超過料金」という方針（TAC に当てはまるかは未確認）を合わせると，多くの著者が超過料金を払って 13〜18 頁にしている可能性がある．12 頁に収める本研究の論文は TAC では短めの部類であり，P7（12 頁）と P14（12 頁），P3（11 頁）が同じ長さの例である．
- 12 頁の掲載版の本文はおよそ 7,500 語（P7）であり，本研究の論文も本文を 7,500〜8,000 語，図表を合わせて 10 枚前後に収める見込みで計画するのがよい．
- P1 は 7 頁で，ほかより極端に短い．TAC の短い論文の枠（short paper）の可能性があるが，種別は確かめていない．

### Gaps
- TAC 固有の頁数の上限（通常論文と短い論文）と超過料金の現行の値は，著者向け案内のページ（computer.org の TAC の頁）が WebFetch で空になったため確かめられなかった．
- P3〜P6，P14 の見出しと図表数は不明．

## 3. 要旨の書き方（語数，数値の有無，強調する点）

### Takeaway
要旨は 173〜260 語で，多くは 200 語前後である．性能の数値を書いた要旨は 13 本中 2 本（P11 の検定結果つきの件数と，P6 の「5 人」）だけで，ほかは「既存の問題 → 本研究の手法または問い → 改善した・分かった」を数値なしで書いている．分析型の要旨は「何を調べ，何が分かり，実務に何を勧めるか」の順に書く．

### Cited Findings
| 記号 | 語数 | 性能の数値 | 強調する点と流れ | 出典 |
|---|---|---|---|---|
| P1 | 198 | なし | 実生活の録音はラベルが少ない → 演技の音声のラベルを使う少数ショット学習を提案 → 4 データセットで実現可能性と，微調整・敵対的学習より良いことを示す → 精神保健の評価への基盤 | [arXiv 2109.02915](https://arxiv.org/abs/2109.02915) |
| P2 | 200 | なし（「significantly improved」とだけ書く） | 既存の敵対的手法は収束しにくい → ADDoG と MADDoG を提案 → 収束が安定し，ターゲットのラベルなしで有意に改善 | [arXiv 1903.12094](https://arxiv.org/abs/1903.12094) |
| P3 | 206 | なし | コーパス間の精度が低い → 対照学習と情報最大化を補助課題にする多課題 Transformer を提案 → 言葉を第 2 のモダリティにして判定を融合 → IEMOCAP，MSP-IMPROV，EMO-DB で改善 | [OpenAlex P3](https://api.openalex.org/works/doi:10.1109/taffc.2025.3526592) |
| P4 | 173 | なし | 既存研究は言語間の共通点を見落とす → 本研究は 2 部構成（第 1 部：感情ごとに共通する母音を調べる，第 2 部：その母音を錨にして適応する） → 基準を上回る | [OpenAlex P4](https://api.openalex.org/works/doi:10.1109/taffc.2025.3530105) |
| P5 | 215 | なし | 動画の表情認識のドメイン間の問題 → SRS を提案（構成要素を順に説明） → 最先端の手法を上回る | [OpenAlex P5](https://api.openalex.org/works/doi:10.1109/taffc.2026.3680134) |
| P6 | 260 | あり（「5 人の注釈者で足りる可能性」） | 既存手法は評定から分布を推定できると仮定している → 注釈者の人数が推定の不確かさに与える影響を調べる → 人工の評定と MSP-Conversation で分析 → 5 人で足りる可能性 → 「実務への助言」 | [OpenAlex P6](https://api.openalex.org/works/doi:10.1109/taffc.2025.3616071) |
| P7 | 200 | なし | 人の注釈は高価で設計に左右される → GPT-4 の注釈を分析 → 人の評価では GPT-4 が高く評価されるが知覚は人と違う → 注釈の工程に組み込む 2 つの方法を検討 → 新しい注釈の実務を提案 | [arXiv 2412.07906](https://arxiv.org/abs/2412.07906) |
| P8 | 187 | なし | 事前学習モデルはデータ増強なしでは成績が低い → LLM の系を作って調べる → 「系統的な分析で，内容と音声の高次の情報が混ざることが原因の 1 つと分かった」 → 分離の枠組みを提案して改善 | [arXiv 2503.06620](https://arxiv.org/abs/2503.06620) |
| P10 | 213 | なし（「5 つの公開データセット，3 言語」と規模だけ書く） | 言語間で性能が落ちる → ADDi と sADDi を提案 → 5 データセット・3 言語で評価し最先端を上回る | [arXiv 2204.08625](https://arxiv.org/abs/2204.08625) |
| P11 | 251 | あり（84,181 録音，26 コーパス，65 時間超，26 中 21 で改善，McNemar 検定で 10 データセットが p < 0.05 で有意，有意に悪化は 2 コーパス，パラメータ数は単独モデルの 3.5 倍） | 大きな集成コーパスを作る → 残差アダプタで多コーパスを学習 → 検定つきで件数を報告 → コードを公開 | [arXiv 2103.08310](https://arxiv.org/abs/2103.08310) |
| P12 | 198 | なし | 音声だけ，または映像だけしかない場合にも動く系が要る → VAVL を提案（構成要素を列挙） → 3 コーパスで有意に上回る | [arXiv 2305.07216](https://arxiv.org/abs/2305.07216) |
| P13 | 199 | なし | 既存の音声・映像の嘘検出の汎化は未検討 → 初のドメイン間ベンチマーク → 2 つの評価手順，3 つのサンプリング方針，MM-IDGM，Attention-Mixer | [arXiv 2405.06995](https://arxiv.org/abs/2405.06995) |
| P14 | 186 | なし | 未知のデータセットで性能が落ちる → 「ドメイン汎化の影響を分析し」ADoGT を提案 → 4 データセットで評価し，構造と融合の比較実験（ablation）も示す | [OpenAlex P14](https://api.openalex.org/works/doi:10.1109/taffc.2023.3290795) |

- Computer Society の一般方針では，通常論文の要旨は 100〜200 語，短い論文は 50 語とされている（検索結果の抜き書き，△．TAC 固有の値かは未確認） — [IEEE Computer Society Author Resources](https://www.computer.org/publications/author-resources/authors)

### Inferences
- 上の方針（200 語まで）に対し，P6（260 語），P11（251 語），P5（215 語），P10（213 語），P3（206 語）は超えている．TAC では 200 語を厳密な上限として運用していない可能性がある．ただし arXiv 版と掲載版で語数が違う可能性がある．
- 分析型の P6 は，結論を数値（5 人）で 1 文に書き，続けて「実務への助言」と言い切っている．本研究の分析の主張（たとえば少数ラベルが主に切片を直すこと，ラベル 1 本の価値）も，要旨に具体的な数値で 1 文ずつ書くと，TAC の分析型の要旨の型に合う．
- 手法型の要旨は数値を書かないものが多い．本研究の第 2 部（弱いラベルの蒸留）は，UAR の改善幅を 1 つだけ書けば，数値なしの多数派よりも主張がはっきりする．

### Gaps
- 掲載版の要旨と arXiv 版の要旨の違いは，P3〜P6，P14 以外は確かめていない（P3〜P6，P14 は OpenAlex に記録された掲載版の要旨を使った）．

## 4. 序論の貢献の書き方

### Takeaway
手法型は貢献を番号か黒丸で 3〜5 個並べる（P1 は 3 個を 1 段落の中に番号で，P10 は 3 個の番号つき，P13 は 5 個の黒丸）．分析型は貢献の一覧の代わりに「研究の問い」（P7 は 2 個，P8 は 3 個と「中心の仮説」1 個）を立てて，各問いに答える節を対応させる．

### Cited Findings
- P1 は「The contributions of this paper are: (1) … (2) … and (3) …」と 1 段落に 3 個を書き，3 個目を「演技の音声から自発の音声へ知識を移せるかの調査」という分析の貢献にしている — [arXiv 2109.02915](https://arxiv.org/abs/2109.02915)
- P2 の arXiv 版には「contribution」という語の一覧が見当たらず，序論の文章の中で ADDoG と MADDoG を導入している — [arXiv 1903.12094](https://arxiv.org/abs/1903.12094)
- P7 は序論で「In this work, we focus on two Research Questions」と書き，問い 1（GPT-4 は感情認識でどれだけできるか）を 4 節，問い 2（GPT-4 は人の注釈を助けられるか）を 5 節に対応させ，「To the best of our knowledge, this is the first study to propose and evaluate the pre-filtering and post-filtering methods」と新しさを 1 文で述べる — [arXiv 2412.07906](https://arxiv.org/abs/2412.07906)
- P8 は序論で 3 つの「primary research questions」（(1) SSL モデルはなぜ抑うつ検出で成績が低いか，(2) 音声と言葉を組み合わせる根本の難しさは何か，(3) データ増強に頼らない系をどう作るか）と「core hypothesis」（内容と音声の高次の特徴が事前学習モデルの表現の中で混ざり，判定境界が曖昧になる）を書く．「contribution」という語は本文に出てこない — [arXiv 2503.06620](https://arxiv.org/abs/2503.06620)
- P10 は「We summarise the contributions of this paper below」として 1)〜3) の 3 個を並べ，3 個目に「5 つのデータセットでの包括的な評価」と「ADDoG と比べてソースのラベル付きデータを 15〜20% 減らせる」を入れる — [arXiv 2204.08625](https://arxiv.org/abs/2204.08625)
- P12 は「Our main contribution in this study is the proposal of a versatile audio-visual learning (VAVL) model」と主な貢献を 1 個に絞り，構成要素を文章で説明する — [arXiv 2305.07216](https://arxiv.org/abs/2305.07216)
- P13 は「our main contributions include:」として，太字の見出しつきの黒丸 5 個（ベンチマーク，2 つの評価手順の比較，3 つのドメインのサンプリング方針，MM-IDGM，Attention-Mixer）を並べ，続けて各節の内容を 1 文ずつ予告する — [arXiv 2405.06995](https://arxiv.org/abs/2405.06995)
- P4 は要旨で「Our approach is structured in two main parts. First, we investigate … Second, we utilize …」と 2 部構成を宣言している（抄録のみ） — [OpenAlex P4](https://api.openalex.org/works/doi:10.1109/taffc.2025.3530105)

### Inferences
- 本研究は分析（第 1 部）と手法（第 2 部）の 2 部構成なので，P7 と P8 の「研究の問い → 各問いに答える節」の形と，P4 の「2 部構成の宣言」の形を組み合わせるのが合う．問いを 2〜3 個立て，その後に番号つきの貢献を 3〜4 個並べる（分析の発見，理論の式，手法，評価の規模）と，手法型の読者にも分析型の読者にも読みやすい．
- P10 の「ソースのラベルを 15〜20% 減らせる」のように，貢献の文に数値を 1 つ入れる書き方は，本研究の「ラベル 1 本の価値」の主張と相性がよい．

### Gaps
- P3〜P6，P14 の貢献の書き方は本文未読のため不明．P11 の貢献の書き方は抽出した文字から見つけられなかった．

## 5. データセットと評価手順，指標，反復と統計検定

### Takeaway
コーパス間の論文はどれも話者（被験者）を分けた分割を明記し，指標は UAR が標準である（P1，P2，P10，P11）．反復は 5〜50 回で，比べる方法の間で分割をそろえる．モデル同士の比較に統計検定を使ったのは本文を読んだ 8 本中 3 本（P2 は分散分析，P11 は McNemar 検定，P12 は両側 t 検定）で，多重比較の補正，効果量，信頼区間を書いた論文は 8 本の中に 1 本もなかった．

### Cited Findings
| 記号 | データセット（数） | 評価手順 | 指標 | 反復 | 統計検定 | 出典 |
|---|---|---|---|---|---|---|
| P1 | ソース 4（IEMOCAP の台本，RAVDESS，CREMA-D，eNTERFACE'05）→ ターゲット 1（IEMOCAP の即興）．3 感情（怒り・喜び・悲しみ） | ソースだけで学習したもの（out-of-domain）と，ターゲットの leave-one-subject-out（in-domain）を基準にし，ターゲットのラベルを「話者ごと・感情ごとに 1〜10 本」与えて残りで評価 | UAR（でたらめなら 33.3%） | ラベルの選び方を 10 回変えて平均 | なし（「significantly improves」とは書くが検定は書かない） | [arXiv 2109.02915](https://arxiv.org/abs/2109.02915) |
| P2 | 3（IEMOCAP，MSP-Improv，PRIORI Emotion．PRIORI は双極性障害の患者の電話の録音） | 実験 1：IEMOCAP ↔ MSP-Improv でターゲットのラベルなし．実験 2：ターゲットのラベルを 0，200，400，800 … 3,200 発話と増やす．実験 3・4：実環境の PRIORI を含める．価値（valence）を 3 段階に分けて分類．テスト話者ごとに UAR を出す | UAR（でたらめなら 0.33） | 10 回（実験 1 は 50 回）．分割は方法の間でそろえる | 「反復 × 話者」の UAR の行列に R で分散分析（p = 0.05）．図では点，表では太字と星で有意を示す | [arXiv 1903.12094](https://arxiv.org/abs/1903.12094) |
| P7 | 3（ISEAR，SemEval 2018 Task 1，GoEmotions．すべて英語のテキスト） | GPT-4 の zero-shot の判定を人の評価で比べる（59 人，2,948 件）．ISEAR は 60/20/20 に無作為分割 | 一致度，人の好み，作業負荷（NASA-TLX），所要時間 | — | 対応のある t 検定と独立の t 検定（* p < 0.1，** p < 0.05）．特徴の選択にも独立の t 検定 | [arXiv 2412.07906](https://arxiv.org/abs/2412.07906) |
| P8 | 2（DAIC-WOZ：面接 189 件，CMDC：中国語） | DAIC-WOZ は開発用の分割で報告．ランダムな探索で選んだハイパーパラメータの平均と最大の F1 を報告 | F1 | 平均（標準偏差）で報告 | 特徴の分析（音響の目印の 2 つ組の長さ）に Mann-Whitney の U 検定．モデル同士の比較の検定は見当たらない | [arXiv 2503.06620](https://arxiv.org/abs/2503.06620) |
| P10 | 5（IEMOCAP，MSP-IMPROV，RECOLA，EMODB，FAU-AIBO．3 言語） | コーパス間とコーパス・言語間．先行研究と比べるときはターゲットの 30% を開発用，70% をテスト用にし，話者を分ける | UAR | 10 回．平均と標準偏差 | なし | [arXiv 2204.08625](https://arxiv.org/abs/2204.08625) |
| P11 | 26 コーパス（84,181 録音，65 時間超） | 各コーパスに話者を分けた学習・開発・テストの分割を作る（無ければ手で作る）．単一課題の転移と全コーパスの多課題学習 | UAR | 記述なし（抽出した範囲） | McNemar 検定（p < 0.05）で基準との誤りの割合の差を検定し，「+」「−」で示す．「不均衡なデータでは McNemar の有意と UAR の上昇が一致しない」と注記 | [arXiv 2103.08310](https://arxiv.org/abs/2103.08310) |
| P12 | 3（CREMA-D，MSP-IMPROV，CMU-MOSEI） | コーパス内．学習・開発・テストを約 70/15/15 に話者を分けて分割 | F1-Macro，F1-Micro，CCC，MAE | 分割か乱数の種を変えて 5 回．平均を報告 | 両側 t 検定（p < 0.05），表に * | [arXiv 2305.07216](https://arxiv.org/abs/2305.07216) |
| P13 | 6（Real-life Trial，Bag of Lies，Box of Lies，MU3D，DOLOS，MDPE） | ドメイン内は被験者を分けた 5 分割の交差検証．ドメイン間は「1 つ → 1 つ」と「複数 → 1 つ」の 2 手順．Box of Lies は品質の懸念から学習に使わない | 正解率（ほかの指標は未確認） | 5 分割の平均 | 見当たらない | [arXiv 2405.06995](https://arxiv.org/abs/2405.06995) |
| P3 | 3（IEMOCAP，MSP-IMPROV，EMO-DB） | コーパス間．ターゲットのラベルなしのデータを補助課題に使う | 未確認 | 未確認 | 未確認 | [OpenAlex P3](https://api.openalex.org/works/doi:10.1109/taffc.2025.3526592) |
| P4 | 2（MSP-Podcast，BIIC-Podcast．英語と台湾の中国語） | 言語間でターゲットのラベルなし | 未確認（ICASSP 版は UAR 58.64%） | 未確認 | 未確認 | [OpenAlex P4](https://api.openalex.org/works/doi:10.1109/taffc.2025.3530105) |
| P6 | 人工の評定と MSP-Conversation | 注釈者の人数を変えて分布の推定を比べる | Belief Mismatch Coefficient（BMC） | 未確認 | 未確認 | [OpenAlex P6](https://api.openalex.org/works/doi:10.1109/taffc.2025.3616071) |
| P14 | 4 つの感情データセット | コーパス間（ドメイン汎化） | 未確認 | 未確認 | 未確認 | [OpenAlex P14](https://api.openalex.org/works/doi:10.1109/taffc.2023.3290795) |

- 本文を読んだ 8 本の文字を「Bonferroni，Holm，Benjamini，false discovery，confidence interval，effect size，Cohen，standard error」で検索すると，当たったのは参考文献の中の人名・地名（Stockholm，Baron-Cohen など）だけで，本文には 1 件もなかった — 8 本の arXiv 版（[P1](https://arxiv.org/abs/2109.02915)，[P2](https://arxiv.org/abs/1903.12094)，[P7](https://arxiv.org/abs/2412.07906)，[P8](https://arxiv.org/abs/2503.06620)，[P10](https://arxiv.org/abs/2204.08625)，[P11](https://arxiv.org/abs/2103.08310)，[P12](https://arxiv.org/abs/2305.07216)，[P13](https://arxiv.org/abs/2405.06995)）
- P1 のラベル数 K は「話者ごと・感情ごと」に数えている（1 本から始めて 2，…，10 本と増やす）．クラスごとに数える本研究とは数え方が違う — [arXiv 2109.02915](https://arxiv.org/abs/2109.02915)
- P1 は，ラベル 1 本のときに FNN の微調整で負の転移（ソースを使うと悪くなる）が多くの組で起きたと報告する — [arXiv 2109.02915](https://arxiv.org/abs/2109.02915)
- P2 は実験 2 で「ターゲットのラベルが増えると ADDoG の改善幅が小さくなる」と報告し，「ラベルが十分にあると表現の汎化の効果は減る」と解釈している — [arXiv 1903.12094](https://arxiv.org/abs/1903.12094)

### Inferences
- 「ターゲットのラベルの本数を横軸にして成績の曲線を描く」評価は P1（1〜10 本）と P2（0〜3,200 発話）に前例がある．本研究の K の軸は TAC の読者になじみのある形である．ただし P1 は話者ごとに数えるので，本研究では「K はクラスあたり．K=25 なら 4 クラス × 25 = 100 本」を最初に定義し，P1 との数え方の違いを関連研究か設定の節で 1 文書くと誤読を防げる．
- 統計検定は TAC で必須ではないが，比べる方法の間で分割をそろえ，話者ごと・反復ごとの点数の行列に検定をかけた P2 の書き方が，本研究の「組ごと・反復ごとの対応のある比較」に最も近い前例である．多重比較の補正と効果量を書けば，読んだ 8 本の水準より厳密になる．
- P11 の「不均衡なデータでは McNemar の有意と UAR の上昇が一致しない」という注記は，UAR を主指標にする本研究が検定の種類を選ぶときの注意として使える．

### Gaps
- P3〜P6，P14 の評価手順・反復・検定は本文未読のため不明．
- TAC の査読者が統計検定を求める頻度は，掲載論文から推測するしかなく，直接の資料は見つからなかった．

## 6. 比較相手とハイパーパラメータの選び方

### Takeaway
コーパス間の論文は，ターゲットのラベルが無いと検証用のデータで設定を選べないことを理由に，ハイパーパラメータを先行研究の値に固定し，全実験で同じ値を使うと明記する（P2，P10）．比較相手は 2〜7 個程度で，ソースだけで学習した基準，微調整，敵対的なドメイン適応（ADDA，DANN），単一モダリティの基準が典型である．

### Cited Findings
- P2 は「It is difficult to validate multiple sets of hyperparameters when conducting cross-dataset experiments, due to the lack of labelled data in the target domain. For this reason, we select hyperparameters based on those found to be commonly selected in prior work and keep them constant for all experiments.」と書く．テスト時の予測は検証の成績が最も高いエポックのものを使う — [arXiv 1903.12094](https://arxiv.org/abs/1903.12094)
- P2 の比較相手は，ソースのラベルだけを使う CNN と，ADDoG，MADDoG の 3 つである．DANN と ADDA は関連研究で扱い，ADDA は音声感情で成功例がない理由を説明している — [arXiv 1903.12094](https://arxiv.org/abs/1903.12094)
- P10 は「Due to the unavailability of target data in the experiments, it is difficult to validate all the hyperparameters of the network for cross-dataset SER. Therefore, we select the parameters commonly used in prior studies」と P2 とほぼ同じ理由づけで固定し，P2 を引用する．比較相手として CNN（基準），GAN，DANN，DBN，CNN-LSTM，オートエンコーダ型のモデルの 6 つを実装し，さらに先行研究が報告した数値と比べる — [arXiv 2204.08625](https://arxiv.org/abs/2204.08625)
- P1 の比較相手は FNN の微調整と ADDA の 2 つで，学習率 0.0005，250 エポックなどの値を固定で書く — [arXiv 2109.02915](https://arxiv.org/abs/2109.02915)
- P11 の比較相手は 2 つの基準（eGeMAPS 特徴を使う基準と DeepSpectrum）と，ResNet の単独学習・従来の転移である — [arXiv 2103.08310](https://arxiv.org/abs/2103.08310)
- P12 の比較相手は音声＋映像の手法 5 つ（UAVM，AuxFormer，SFAV，TLSTM，MulT）と，単一モダリティの基準 2 つ（Uni.(A)，Uni.(V)）である — [arXiv 2305.07216](https://arxiv.org/abs/2305.07216)
- P8 は開発用の分割でランダムな探索をしてハイパーパラメータを選び，平均と最大の F1 を報告する．比較の一部（ablation）では Tree-structured Parzen Estimator（TPE）で調整する — [arXiv 2503.06620](https://arxiv.org/abs/2503.06620)

### Inferences
- 本研究は合併の正則化の強さ C を「ソースだけの C」に固定した地図を土台にしている．P2 と P10 が「ターゲットのラベルが無いので先行研究の値に固定する」と書いて TAC で採録されている前例があるので，本研究も同じ理由（少数のターゲットのラベルで C を選ぶと，選ぶこと自体にラベルを使い，テストの情報も漏れやすい）を設定の節で 2〜3 文で書けば，TAC の読者に通じる．
- P8 の「開発用の分割での最大値」を報告する書き方は，テストの成績で設定を選ぶのに近く，本研究の「実用値とオラクル値を分けて報告する」方針のほうが厳密である．この違いは本研究の評価手順の強みとして書ける．

### Gaps
- P3〜P6，P14 の比較相手とハイパーパラメータの選び方は不明．

## 7. 理論・導出・数式の扱い

### Takeaway
本文を読んだ 8 本には，定理・補題・命題として述べた理論は 1 つもなかった．式は損失関数・サンプリングの規則・アルゴリズムの説明に限られ，番号つきの式は 2〜20 個程度である．理論寄りの書き方に最も近いのは，P8 の「仮説 → 系統的な分析 → 枠組み」と，P6 の「人工データと実データで推定の不確かさを測る」分析（抄録のみ）である．

### Cited Findings
- 8 本の本文を「theorem，lemma，proposition，we assume，assumption」で検索すると，P2 と P7 で「assumption」が関連研究の説明の中に 1〜2 回出るだけで，定理・補題・命題は 0 件だった — [P2](https://arxiv.org/abs/1903.12094)，[P7](https://arxiv.org/abs/2412.07906)
- 番号つきの式の最大の番号（概数）は，P1 が 4，P2 が 3，P7 が 2 前後，P8 が 22，P10 が 12，P11 が 4，P12 が 4，P13 が 7 である．P2 と P13 はアルゴリズムの枠を本文に載せる — 8 本の arXiv 版
- P8 は序論で「core hypothesis」を述べ，III 節で基準の系，IV 節で既存モデルの限界の分析，V 節で情報分離の枠組み，VI〜VII 節で評価という順に書く — [arXiv 2503.06620](https://arxiv.org/abs/2503.06620)
- P12 は 3.5 節「Complexity Analysis」でパラメータ数と推論の遅れ（RTX 3090 で音声のみ約 0.0035 秒など）を報告する — [arXiv 2305.07216](https://arxiv.org/abs/2305.07216)
- P6 は Belief Mismatch Coefficient を使い，人工の評定と MSP-Conversation の実際の評定で，注釈者の人数と分布の推定の不確かさの関係を調べる（抄録のみ） — [OpenAlex P6](https://api.openalex.org/works/doi:10.1109/taffc.2025.3616071)
- TAC の論文で汎化の上界や定理を前面に出したものを Web で検索したが，見つかったのは TAC 以外の arXiv の論文（グラフのドメイン適応の目標リスクの上界を述べる arXiv 2603.26840 など）だけだった — [WebSearch 結果に出た arXiv 2603.26840](https://arxiv.org/abs/2603.26840)

### Inferences
- 読んだ範囲では，TAC の近い論文は理論を定理の形で前面に出していない．本研究が「ニュートン法の近似」と「ラベルの価値の式」を前面に置くと，TAC では珍しい書き方になる．珍しいこと自体は新しさにもなるが，読者の多くは感情認識の応用研究者なので，式は (1) 仮定を番号つきで日常語で書き，(2) 式から予言を導き，(3) 予言を実測と照らし合わせる，という「検証できるモデル」として書くのが安全である．P8 の「仮説 → 分析 → 枠組み」の流れはこの書き方に近い．
- 導出の細部（ニュートン法の 1 歩の近似の誤差など）は補足資料に回し，本文には結果の式と仮定と検証だけを残すと，12 頁に収めやすい．

### Gaps
- TAC 2021〜2026 年に定理つきの理論を前面に置いた論文があるかは，検索した範囲では確かめられなかった（OpenAlex が途中で 429 Too Many Requests を返し，追加の検索ができなかった）．見つからなかったことは「存在しない」ことを意味しない．
- P6 が数式モデル（分布の推定の理論）をどの程度本文に書いているかは本文未読のため不明．

## 8. 限界と倫理の書き方

### Takeaway
限界を独立の節にしたのは本文を読んだ 8 本中 P7 の 1 本だけで，約 300 語である．P1 は議論・結論の節の中で「First … Second …」と 2 点を書き，P11 は結論の中で 1 文だけ書く．残りの 5 本には自分の研究の限界をまとめた箇所が見当たらない．倫理は P13 が段落を 1 つ設け，P2 と P7 は参加者の同意を 1 文で書く．

### Cited Findings
- P7 の 7 節「Limitations and Future Work」（約 300 語）は，(1) 人のラベルを正解として評価に使うことが常に信頼できるわけではないのに代わりの評価法がない，(2) 分類の課題に限った（次元のラベルは扱っていない），(3) 前処理と後処理のフィルタは予備的な実証である，などを順に書く — [arXiv 2412.07906](https://arxiv.org/abs/2412.07906)
- P1 は 6 節「Discussion & Conclusions」の中で「Our study depicts the following limitations. First, … Second, …」と書き，(1) ソースもターゲットも実験室で録音されていてずれが小さくなりうる，(2) 英語だけである，の 2 点と，それぞれの今後の課題（EmotiW などの実環境のデータ，言語間の転移）を対にして書く — [arXiv 2109.02915](https://arxiv.org/abs/2109.02915)
- P11 は結論の中で「One limitation of the work presented herein is that the base architecture of the ResNet has not been extensively optimised」と 1 文だけ書く — [arXiv 2103.08310](https://arxiv.org/abs/2103.08310)
- P8 の IV 節「Current Limitations and Discussion」は既存の事前学習モデルの限界の分析であり，自分の研究の限界の節ではない — [arXiv 2503.06620](https://arxiv.org/abs/2503.06620)
- P13 は 4.7 節の末尾に「Ethical Consideration」の段落を置き，嘘の検出の AI はプライバシー，公平性，心理的な害の防止を重視し，同意と責任あるデータの使用を確保すべきだと書く — [arXiv 2405.06995](https://arxiv.org/abs/2405.06995)
- P2 は PRIORI のうち手作業の注釈に同意した 12 人だけを使ったと書き，P7 は人の評価の参加者が研究の目的を知らされて同意したと書く — [arXiv 1903.12094](https://arxiv.org/abs/1903.12094)，[arXiv 2412.07906](https://arxiv.org/abs/2412.07906)

### Inferences
- TAC では限界の節は必須ではないが，分析型の P7 は独立の節を置いている．本研究の分析の主張は「凍結したエンコーダ＋線形分類器」「4 感情」「7 群」という条件に依存するので，P7 の形で 300 語前後の独立の節を置き，条件と外れる場合（微調整したエンコーダ，次元のラベル，他言語）を具体的に書くのがよい．
- 感情認識は TAC で倫理が問われやすい分野であり，P13 のように 1 段落で，公開データだけを使うこと・各データの利用条件・顔の映像を扱うことの注意を書いておくと，査読での指摘を先回りできる．

### Gaps
- P3〜P6，P14 の限界と倫理の書き方は不明．
- TAC が倫理の記述を投稿規定で求めているかは確かめられなかった．

## 9. 補足資料・付録・コードの公開

### Takeaway
本文を読んだ 8 本の arXiv 版には，補足資料（supplementary material）への言及も付録（Appendix）もなかった．代わりに 3 本（P1，P7，P11）がコードの GitHub の場所を本文に書いている．

### Cited Findings
- 8 本の文字を「supplementa」「appendix」で検索すると 0 件だった — 8 本の arXiv 版（[P1](https://arxiv.org/abs/2109.02915)，[P2](https://arxiv.org/abs/1903.12094)，[P7](https://arxiv.org/abs/2412.07906)，[P8](https://arxiv.org/abs/2503.06620)，[P10](https://arxiv.org/abs/2204.08625)，[P11](https://arxiv.org/abs/2103.08310)，[P12](https://arxiv.org/abs/2305.07216)，[P13](https://arxiv.org/abs/2405.06995)）
- コードの場所：P1 は github.com/HUBBS-Lab-TAMU/pytorch-MeL，P7 は github.com/chailab-umich/GPT-4-Emotion-Annotation，P11 は github.com/EIHW/EmoNet（P11 は要旨でもコードの公開を述べる） — [P1](https://arxiv.org/abs/2109.02915)，[P7](https://arxiv.org/abs/2412.07906)，[P11](https://arxiv.org/abs/2103.08310)
- Computer Society の別の Transactions（Transactions on Computers）の案内では，補足資料は本文と別のファイルで出し，頁数の上限はないとされている．TAC に同じ規定があるかは確かめていない（検索結果の抜き書き，△） — [Submitting a Paper to IEEE Transactions on Computers](https://computer.org/web/tc/author.)

### Inferences
- 近い論文は補足資料をほとんど使っていない（少なくとも arXiv 版からは見えない）．本研究が補足資料を使うなら，本文だけで主張が完結し，補足資料には「導出の細部」「全組の表」「2 つ目のエンコーダでの再現」だけを置く，という分け方を本文に 1 文で書くと，読者は本文だけで判断できる．
- 掲載版の IEEE Xplore に補足資料が付いているかは確かめていないので，この推論は arXiv 版に限った観察である．

### Gaps
- IEEE Xplore の掲載版の補足資料の有無（「Supplemental Items」の欄）は確かめられなかった．

## 10. 分析型の論文の貢献の言い表し方

### Takeaway
TAC の分析型の論文は，貢献を「発見（findings）」と「実務への助言・指針（practical advice，practices）」の 2 つで言い表し，発見の後に小さな手法（P7 の前後のフィルタ，P8 の情報分離の枠組み，P4 の母音の錨）を足して「分析から導いた手法」で締める形が多い．

### Cited Findings
- P6 は「Our results suggest that using 5 raters may be sufficient … Our study provides insights and practical advice for collecting emotion labels」と，数値つきの発見と実務への助言で締め，さらに「分析の方法は主観的な量を扱うほかの分野にも使える」と方法の一般性を述べる — [OpenAlex P6](https://api.openalex.org/works/doi:10.1109/taffc.2025.3616071)
- P7 は要旨を「Together, our findings highlight opportunities for new emotion labeling practices」で締め，序論で「These findings advocate for thoughtful consideration of annotation design choices」と書き，議論の節で「In the first part of our study (Section 4) … In the second part, we further explored ways to integrate GPT-4 into annotation processes」と 2 部構成を振り返る — [arXiv 2412.07906](https://arxiv.org/abs/2412.07906)
- P8 は「Through systematic analysis, we discover that the poor performance of pre-trained models stems from the conflation of high-level information」と「discover」を使って発見を述べ，「providing new insights for developing more effective multi-modal depression detection systems」で締める — [arXiv 2503.06620](https://arxiv.org/abs/2503.06620)
- P4 は第 1 部で「感情ごとに言語をまたいで共通する母音」を調べ，第 2 部でその母音を錨にした適応を作る．結論は「言語の適応には音声学的な類似が重要である」という発見の形で書く（抄録のみ） — [OpenAlex P4](https://api.openalex.org/works/doi:10.1109/taffc.2025.3530105)
- P14 は手法の論文だが，要旨で「we analyze the impact of the domain generalization method on cross-corpus SER」と分析を先に述べる — [OpenAlex P14](https://api.openalex.org/works/doi:10.1109/taffc.2023.3290795)

### Inferences
- 本研究の第 1 部の貢献は「少数ラベルが主に何を直すか（切片か向きか）の発見」と「いつソースを混ぜるべきかの指針」，第 2 部は「その発見から導いた弱いラベルの蒸留」と書くと，P4・P7・P8 の「分析 → 分析から導いた手法」の型にそのまま乗る．
- P6 のように，発見を数値つきの 1 文（「K が○本以下ではソースを混ぜると UAR が平均○ポイント上がる」など）にまとめ，実務の指針と言い切る書き方が，TAC の分析型で採録された例に合う．

### Gaps
- 分析だけで手法を出さない TAC の論文（P6 が候補）の本文の構成は，本文未読のため確かめられなかった．

## 11. 総合：TAC で好まれる書き方と，本研究の論文（第 1 部 分析，第 2 部 手法，12 頁）への当てはめ

### Takeaway
TAC の近い論文は「序論 → 関連研究 → データと設定 → 方法 → 実験 → 議論・結論」の順で，掲載版は中央値 15 頁，関連研究は本文の 1〜2 割，実験と結果は 3〜7 割を占める．分析の後に手法を置く 2 部構成は P4，P7，P8 に前例があり，本研究の構成はこの型に合う．一方，定理の形の理論を前面に置いた近い論文は見つからず，本研究の理論は「仮定 → 予言 → 実測での確認」の形で書くのがよい．

### Cited Findings
- 2 部構成の前例：P4 は要旨で「two main parts」と宣言する — [OpenAlex P4](https://api.openalex.org/works/doi:10.1109/taffc.2025.3530105)．P7 は 4 節（分析）と 5 節（手法の検討）に分け，議論で「In the first part … In the second part …」と振り返る — [arXiv 2412.07906](https://arxiv.org/abs/2412.07906)．P8 は既存の限界の分析（IV 節）の後に枠組み（V 節）を置く — [arXiv 2503.06620](https://arxiv.org/abs/2503.06620)
- 評価手順の正当化：P2 と P10 はターゲットのラベルが無いことを理由にハイパーパラメータを固定し — [P2](https://arxiv.org/abs/1903.12094)，[P10](https://arxiv.org/abs/2204.08625)，P1 はドメイン内（leave-one-subject-out）とドメイン外を上下の基準として「ソースとターゲットのもともとの差を知るため」と理由を書き — [P1](https://arxiv.org/abs/2109.02915)，P13 は 2 つの評価手順が「ドメインのずれの大きさが違う場合を比べられる」と理由を書く — [P13](https://arxiv.org/abs/2405.06995)．P11 は話者を分けた分割が無いコーパスには手で作ったと書く — [P11](https://arxiv.org/abs/2103.08310)
- モダリティ別と融合の並べ方：P12 の表 1 は，コーパスごとに「Audio-Visual／Acoustic／Visual」の列を並べ，行に VAVL，ほかの音声＋映像の手法，単一モダリティの基準（Uni.(A)，Uni.(V)）を置き，5 回の平均 ± 標準偏差と有意の * をつける — [arXiv 2305.07216](https://arxiv.org/abs/2305.07216)．P13 は 4.3 節（単一モダリティの特徴でのドメイン間の評価）→ 4.5 節（融合でのドメイン間の評価）と節を分け，ドメイン内の表でも「単一モダリティの中では音声のメル・スペクトログラムが最も高く（Real-life Trial で 75.78%），融合で一貫して上がる」の順に書く — [arXiv 2405.06995](https://arxiv.org/abs/2405.06995)．P3 は音声と言葉の多課題 Transformer を別々に作り，推論時に判定を融合する（抄録のみ） — [OpenAlex P3](https://api.openalex.org/works/doi:10.1109/taffc.2025.3526592)
- 分析型・理論寄りの受け入れ：分析型の P6（15 頁），P7（12 頁），P8（17 頁）がいずれも 2025〜2026 年に TAC に採録されている — 1 節の OpenAlex の記録．8 本の本文に定理・補題・命題は無い — 7 節
- 長さ：掲載版 13 本は 7〜18 頁，中央値 15 頁．12 頁の P7 の本文は約 7,500 語，図 4 枚，表 3 枚，参考文献は arXiv 版で 76 本（OpenAlex では 64 本） — 2 節

### Inferences
本研究の論文（12 頁）には次の構成を推す．理由は，P7（12 頁，分析 → 手法の 2 部構成）の長さと節の比を土台にし，P2・P10 の評価手順の正当化と P12 のモダリティ別の表の形を取り込めば，TAC で採録された前例の組み合わせだけで本研究の主張を全部載せられるからである．頁の配分は本文約 7,500〜8,000 語を前提にした目安である．

1. **Introduction（約 1.2 頁）**：少数ラベルの設定（K はクラスあたり，凍結したエンコーダ＋線形分類器）を最初の頁で定義する．研究の問いを 2〜3 個（例：少数のラベルは分類器の何を直すか，ソースを混ぜると得をするのはいつか，その仕組みから手法を作れるか）立て，番号つきの貢献を 3〜4 個並べる（P7・P8 の問いの形と，P10・P13 の番号つきの貢献の形の組み合わせ）．
2. **Related Work（約 1 頁，本文の 1〜1.5 割）**：コーパス間の音声・マルチモーダル感情認識のドメイン適応（P2，P10，P14，P3），少数ラベル（P1），擬似ラベル・蒸留（P5 の自己蒸留を含む），DA の理論（Ben-David 2010）．P1 の「話者ごと・感情ごと」の K と本研究の「クラスごと」の K の違いをここで 1 文書く．
3. **Problem Setting, Data, and Protocol（約 1.5 頁）**：7 群，話者の分離，4 感情，UAR，全てのソースとターゲットの組，反復の回数．C をソースだけの値に固定する理由を P2・P10 の前例を引いて 2〜3 文で書く．実用値とオラクル値を分けて報告することもここで宣言する．
4. **Part I: What Do a Few Target Labels Correct?（約 3 頁）**：仮定を番号つきで書き，ニュートン法の近似とラベルの価値の式から予言を導き，実測（切片と向き）で確かめる．導出の細部は補足資料に回す．発見は P6 の形で数値つきの 1 文にまとめる．
5. **Part II: Source-Classifier Outputs as Weak Labels（約 1.5 頁）**：第 1 部の式から蒸留の手法を導く（P8 の「分析 → 枠組み」の流れ）．
6. **Experiments（約 2 頁）**：比較相手は，ターゲットのみ，合併，ソースで学んだ分類器の微調整，代表的なドメイン適応（ADDA・DANN 型）または自己学習の 4〜5 個．モダリティ別の表は P12 の形で，列を A，V，T，AV，VT，AT，AVT の順に並べ，行に方法を置く．組ごと・反復ごとの対応のある検定と多重比較の補正を書く（読んだ 8 本には補正の例が無く，ここが手堅さの差になる）．
7. **Discussion（約 0.5 頁）**：いつソースを混ぜるべきか，何本ラベルを付けるべきかの指針を，P6・P7 の「実務への助言」の形で書く．
8. **Limitations and Ethics（約 0.3 頁，300 語前後）**：P7 の形の独立の節．凍結したエンコーダ・4 感情・7 群という条件，微調整したエンコーダや次元のラベルに当てはまるかが未確認であること，公開データだけを使うことと顔の映像の扱い（P13 の倫理の段落の形）．
9. **Conclusion（約 0.2 頁）**．
10. 参考文献と著者紹介（約 1.5 頁．参考文献は 50〜60 本が近い論文の値）．補足資料には導出の細部，全組の表，2 つ目のエンコーダでの再現を置く．

要旨は 200 語以内とし，第 1 部の発見を数値で 1 文，第 2 部の改善幅を数値で 1 文書く（数値を書く要旨は TAC では少数派だが，分析型の P6 と P11 が書いている）．

### Gaps
- TAC 固有の頁数の上限・超過料金・補足資料の規定は確かめられなかった．投稿前に TAC の著者向け案内（computer.org の TAC の頁）を直接確かめる必要がある．
- 本文を読めなかった 5 本（P3〜P6，P14）が上の傾向に当てはまるかは不明である．とくに分析型の P6 の構成（理論の扱い，限界の書き方）は，本研究に最も近い可能性があるのに本文を確かめられていない．
- 「理論を前面に置いた論文が TAC の査読でどう評価されるか」を直接示す資料（査読の方針や編集者の発言）は見つからなかった．
