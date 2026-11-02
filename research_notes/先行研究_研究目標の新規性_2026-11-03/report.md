# 先行研究との差分：「このターゲットのこの感情の対には，このモダリティのこのソースの方向が使える」を解析で示し，メタ学習で使う機構を作ることは新しいか（2026-11-03）

担当：prior-work-diff（調査役の実行環境が報告ファイルの作成を拒んだため，本体のセッションが調査役の本文をそのまま保存した）．検証の水準の記号：◎ 原文（本文か HTML 全文）で該当箇所を確かめた，○ 抄録か信頼できる二次資料で確かめた，△ 検索の断片だけ．2026-11-03 の文献調査（`research_notes/研究提案_2026-11-03/調査_文献.md`）と `docs/related-work.md` で既に調べた文献は，要点を引くだけにして調べ直していない．数値はすべて，記号の水準で確かめた出所から引いた．

## 0. 要約（10 行）

1. **判定：研究目標の達成物そのものは，今回の検索では先行研究に無い．**「ソース × 感情の対 × モダリティ」を 1 つの表に並べて，どのソースのどのモダリティの方向がどの感情の対で使えるかを測った研究は，音声・顔・言葉のどれでも，コーパス間の感情認識では見つからなかった．
2. ただし部品は 1 つずつ先例がある．(a) ソースのサンプルやドメインに重みを付ける手法（Hoffman–Mohri–Zhang 2018，Guo–Shah–Barzilay 2018，Cao ほか 2019，Yang ほか 2020），(b) 感情認識でのソースの選択（Schuller ほか 2011 の典型性による選択，Ahn–Lee–Shin 2022 のコーパスごとの重み），(c) クラスごとのずれ合わせ（Zhou & Chen 2019，Li & Deng の ECAN），(d) 感情ごとにモダリティの重みを変える融合（Marín ほか 2026），(e) 複数ソースを 1 つずつ仮のターゲットにして「どのソースを使うか」をメタ学習する手法（Guo–Shah–Barzilay 2018 が最も近い）．
3. 最も近い 5 本：Guo, Shah, Barzilay 2018（EMNLP．ソースを 1 つずつ仮のターゲットにして，発話ごとのソースの重みを学ぶ），Marín ほか 2026（arXiv．感情ごとにモダリティの重みを学ぶ融合），Schuller, Zhang, Weninger, Rigoll 2011（音声感情認識で学習データを選ぶ），Li & Deng（ECAN．顔の表情のコーパス間で，クラスごとの比率でソースに重みを付ける），Mansour, Mohri, Ro, Suresh, Wu 2021（AISTATS．少数のターゲットラベルで混合を選ぶ理論と，対ごとのずれで重みを決めることへの反論）．
4. **狙うべき差分（優先順）**：(i) ソース × 感情の対 × モダリティの「方向が使える度合い」の表を，話者を半分に分けて選ぶ側と測る側を分けた上で出す（設計済み．保存済みの出力から計算できる）．先行研究の「クラスごとの重み」は，ラベルの割合の比（ECAN）か学習で決めた注意（Bi-ATEN）で，対ごとの地図として報告したものは無い．(ii) 得を「位置」と「方向」に分けた上で，K 本・同じモダリティのラベル無し・他モダリティの疑似ラベル・ソースの各資源が，どの感情の対の何を直すかを対応づける．先行研究のクラスごとの DA はこの分解を持たない．(iii) メタ学習で学ぶものを「候補からの選択」でなく「測った量（対ごと・モダリティごとの方向の使える度合い）から設定を出す規則」にし，仮のターゲットでの選択がターゲットの K 本での選択に対してどこで勝ち負けするかを Mansour 2021 の理論と照らして書く．
5. 注意：Mansour 2021 は「ソースとターゲットの対ごとのずれで重みを決める方法は最適でないことがある」「少数のターゲットラベルで候補から選ぶ方が保証が良い（余分の誤差は √(ソース数/ラベル数)）」を示している．対ごとにソースを選ぶ設計は，この反論に答える形（半分に分けた検証，K 本での選択との比較）で書く必要がある．
6. 見つからなかったこと：感情ごと・対ごとの転移しやすさの指標，感情認識でのメタ学習で「何を転移するか（どのソース・どのモダリティをどの感情に）」を決めたもの，Shapley 値によるモダリティの価値づけを感情で感情ごとに出したもの，顔と言葉でのコーパス間の感情ごとの混同の体系的な報告．

## 1. 用語（この報告の中での意味）

- **ドメイン適応（DA）**：学習に使うデータ（ソース）と使う場面のデータ（ターゲット）の分布が違うときに，ターゲットで良く働く分類器を作る問題．ターゲットのラベルを使わないものを「教師なし DA」，少しだけ使うものを「半教師あり DA」と呼ぶ．
- **重要度の重み（importance weighting）**：ソースの各サンプルに「ターゲットにどれだけ似ているか」の重みを掛けて学ぶこと．
- **疑似ラベル**：ラベル無しのサンプルに，別のモデルが付けた予測．
- **leave-one-domain-out（1 つ抜き）**：複数のソースのうち 1 つを外して仮のターゲットにし，残りで学んで外した 1 つで測ること．
- **メタ学習**：多くの「課題」（ここではソースのコーパスを 1 つずつ仮のターゲットにしたもの）で学習の仕方そのものを学ぶこと．
- **Shapley 値**：複数の部品（サンプル，モダリティ，ソース）が一緒に出した成果を，各部品の「平均の貢献」として分配する計算．
- **位置の誤り／方向の誤り**：`DAI_paper/inventory.md` の語．分類器の誤りを，感情ごとの点数の底上げ（切片）のずれで失う分と，重みの向きのずれで失う分に分けたもの．

## 2. 問いごとの表

### A. ソースのサンプル単位・クラス単位の選択と重みづけ

分け方の列：**選ぶ単位**（ドメイン／サンプル／クラス），**ターゲットの何を使うか**（無し／ラベル無し／少数ラベル）．

| 論文 | 会場/年 | 何をしたか | 測った・主張したこと | 単位／ターゲット | 水準 | URL |
|---|---|---|---|---|---|---|
| Hoffman, Mohri, Zhang, "Algorithms and Theory for Multiple-Source Adaptation" | NeurIPS 2018 | ソースごとの予測器を，入力 x ごとにソースの密度で重みづけて合わせる（分布重みづけ結合）．ターゲットはソースの混合と仮定 | 「固定の凸結合はごく単純な問題でも失敗する」．ターゲットのデータを一切使わずに，どんな混合のターゲットでも損失 ≤ ε + δ．実験は感情極性（本・DVD・家電・台所），数字画像，Office | サンプルごと（密度）／無し | ◎ | https://ar5iv.arxiv.org/html/1805.08727 |
| Mansour, Mohri, Rostamizadeh, "Domain Adaptation with Multiple Sources" | NIPS 2008 | 上の原型．ソースの仮説の標準的な凸結合は「非常に悪くなりうる」，分布で重みづけた結合は保証を持つ | 理論のみ | ドメインごと／無し | ○ | https://papers.nips.cc/paper/3550-domain-adaptation-with-multiple-sources |
| Mansour, Mohri, Ro, Suresh, Wu, "A Theory of Multiple-Source Adaptation with Limited Target Labeled Data" | AISTATS 2021 | 混合比の候補ごとにモデルを作り，少数のターゲットラベル m₀ 本で選ぶ（LMSA） | 余分の誤差 O(√(ソース数/m₀))．**対ごとのずれで重みを決める方法（Konstantinov & Lampert 2019）は最適でないことがある**．m₀ が小さいとずれの推定が弱い | ドメインごと／少数ラベル | ◎（既読．`docs/related-work.md` 2.9） | https://arxiv.org/abs/2007.09762 |
| Konstantinov & Lampert, "Robust Learning from Untrusted Sources" | ICML 2019 | 少量の信頼できるターゲットの参照データでソースごとのずれを推定して重みを決める．凍結特徴＋線形ロジスティック回帰 | 本研究と同じ学習器．ただし上の論文に批判された | ドメインごと／少数ラベル | ◎（既読） | https://arxiv.org/abs/1901.10310 |
| Crammer, Kearns, Wortman, "Learning from Multiple Sources" | JMLR 2008 | ラベルの付け方だけが違うソースを近い順に何個まで混ぜるかを上界で決める | 理論と人工データ | ドメインごと／無し | ◎（既読） | https://www.jmlr.org/papers/v9/crammer08a.html |
| Xu ほか, "Deep Cocktail Network" (DCTN) | CVPR 2018 | ターゲットの各サンプルに対し，ソースごとのドメイン判別器の当惑度からソースの分類器の重みを決める．ソース間でクラスの集合が違う「カテゴリのずれ」を扱う | 3 つの画像ベンチマーク（名前は抄録に無い） | サンプルごと／ラベル無し | ○ | https://arxiv.org/abs/1803.00830 |
| Yang, Balaji, Lim, Shrivastava, "Curriculum Manager for Source Selection" (CMSS) | ECCV 2020 | ソースのサンプルごとに「今どれを合わせるか」のカリキュラムを敵対的に学ぶ．ドメインのラベルが要らない | 4 つの画像ベンチマークで既存法を上回る | サンプルごと／ラベル無し | ○ | https://arxiv.org/abs/2007.01261 |
| Cao, You, Long, Wang, Yang, "Learning to Transfer Examples for Partial DA" (ETN) | CVPR 2019 | ソースの各サンプルの「転移しやすさ」をターゲットとの似方から数値化し，分類器と判別器の両方の重みにする | Office-31，Office-Home，ImageNet-1K，Caltech-256 | サンプルごと／ラベル無し | ○ | https://arxiv.org/abs/1903.12230 |
| Li, Li, Li, Zhu, Lu, "Agile Multi-Source-Free DA" (Bi-ATEN) | arXiv 2024 | 凍結したソースのモデル群を，ターゲットのサンプルごとの注意で重みづける．図ではクラスごとの重みの偏りも示す | ターゲットはラベル無しのみ．**クラスごとの重みは「サンプルごとの重みをクラスで平均した図」で，数値は無い**．「異なる転移しやすさを要するクラスに動的に適応する」と記述 | サンプルごと（クラスは可視化のみ）／ラベル無し | ◎ | https://arxiv.org/html/2403.05062v1 |
| Li & Deng, "A Deeper Look at Facial Expression Dataset Bias" (ECAN) | arXiv 1904.11150（ICPR 2018 の拡張） | 顔の表情のコーパス間で，ソースの各クラスに比率 α_l = P^t(l)/P^s(l)（ターゲットの比率は疑似ラベルから）を掛け，クラス条件つき MMD で合わせる | 7 表情の域内平均 63.83% に対しコーパス間 39.40%（深層特徴）．RAF-DB 2.0 → CK+ で anger が sadness に，fear が surprise に流れる | クラスごと（比率）／ラベル無し | ◎ | https://ar5iv.labs.arxiv.org/html/1904.11150 |
| Tachet des Combes, Zhao, Wang, Gordon（GLS） | NeurIPS 2020 | クラスごとの比率でソースを重みづけてから条件つき分布を合わせる | 理論と画像 | クラスごと（比率）／ラベル無し | ◎（既読） | https://arxiv.org/abs/2003.04475 |
| Ren, Zeng, Yang, Urtasun, "Learning to Reweight Examples" | ICML 2018 | きれいな検証集合の損失が減る向きに，ミニバッチの各サンプルの重みをメタ勾配で決める | 雑音ラベル・不均衡 | サンプルごと／少数ラベル（検証集合） | ○ | https://proceedings.mlr.press/v80/ren18a.html |
| Shu ほか, "Meta-Weight-Net" | NeurIPS 2019 | 損失 → 重みの関数そのものを小さなメタ集合で学ぶ | 同上 | サンプルごと／少数ラベル | ○ | https://arxiv.org/abs/1902.07379 |
| Ruder & Plank, "Learning to select data for transfer learning with Bayesian Optimization" | EMNLP 2017 | ドメインの近さの尺度の重みをベイズ最適化で学び，ソースのサンプルを選ぶ | 感情極性・品詞・構文解析で既存の近さの尺度を上回る | サンプルごと／ラベル無し（近さ） | ○ | https://aclanthology.org/D17-1038/ |
| Fan, Tian, Qin, Bian, Liu, "Learning What Data to Learn" | arXiv 2017（ICLR 2017 ワークショップ） | ミニバッチの中でどのサンプルを捨てるかを強化学習で決める（Neural Data Filter） | 収束の速さ．転移の設定ではない | サンプルごと／— | ○ | https://arxiv.org/abs/1702.08635 |
| Parvez & Chang, "Evaluating the Values of Sources in Transfer Learning" (SEAL-Shap) | NAACL 2021 | ソース（ドメイン・言語）の価値を Shapley 値で測る | 「どのソースも等しく役立つわけではない」「価値は直感的な近さと一致」．単位はソース全体 | ドメインごと／（抄録に記載なし） | ○ | https://arxiv.org/abs/2104.12567 |
| Ghorbani & Zou, "Data Shapley" | ICML 2019 | 学習データの各点の価値を Shapley 値で測る．DA の実験では価値の高いソースのデータは少数派の群に集中 | 顔画像の性別で 84.1% → 91.5%（検索の二次資料） | サンプルごと／少数ラベル（検証集合） | ○ | https://arxiv.org/abs/1904.02868 |
| Nguyen ほか LEEP（ICML 2020），You ほか LogME（ICML 2021），Huang ほか TransRate（ICML 2022） | — | 転移しやすさの指標．LEEP はソースの分類器の予測とターゲットのラベル，LogME はベイズ線形モデルの証拠，TransRate は特徴とラベルの相互情報量（符号化率で近似） | **どれもソースとターゲットの対に 1 つの点数**で，クラスごとの点数は出さない | ドメインごと／少数ラベル | ○（LEEP・LogME は既読．TransRate は抄録） | https://proceedings.mlr.press/v162/huang22d.html |
| **感情認識**：Schuller, Zhang, Weninger, Rigoll, "Selecting Training Data for Cross-Corpus SER: Prototypicality vs. Generalization" | Afeka-AVIOS 2011 | 大きな音響特徴空間の距離で「典型性」を定義し，典型的な学習サンプルだけを選ぶ | 覚醒の 2 値で最大 7.5% の改善．データベースの典型性とコーパス間の正解率の相関 .571．**感情価では効果なし**．選ぶ単位はサンプルとデータベースで，感情のクラスごとではない（抄録と二次資料による．本文は取得できず） | サンプル・ドメインごと／無し | ○ | https://mediatum.ub.tum.de/doc/1107310/109056.pdf |
| **感情認識**：Hassan, Damper, Niranjan, "On acoustic emotion recognition: compensating for covariate shift" | IEEE TASLP 2013 | KMM・uLSIF・KLIEP の 3 つの重要度の重みを SVM に入れる．FAU Aibo の 2 つの学校を 2 つのドメインとみなす | 重みはサンプルごと，ラベル無しの共変量のずれだけを扱う．数値は本文を開けず未確認 | サンプルごと／ラベル無し | ○ | https://eprints.soton.ac.uk/350948 |
| **感情認識**：Ahn, Lee, Shin, "Multi-Corpus SER for Unseen Corpus Using Corpus-Wise Weights in Classification Loss" | Interspeech 2022 | CREMA-D・IEMOCAP・MSP-IMPROV・MSP-Podcast の 3 つで学び 1 つで測る（4 感情）．各学習コーパスの UA からコーパスごとの重みを毎エポック更新し，学びにくいコーパスに重みを足す | 平均 UA：交差エントロピーだけ 43.1 → コーパス重み 45.5 → ＋疑似ラベル 46.0．**ターゲットの情報を一切使わない**．クラスごとではない | ドメインごと／無し | ◎ | https://www.isca-archive.org/interspeech_2022/ahn22_interspeech.html |
| **感情認識**：Huang, Lin, Lee, "MI-FUSE" | arXiv 2025-09 | ソースで学んだ SER 分類器と API だけの音声 LLM の 2 つの教師の予測を，相互情報量に基づく不確かさで重みづけて，ラベル無しのターゲットで生徒を学ぶ | 最良の相手を 3.9% 上回る（指標は抄録に無い）．重みがクラスごとかサンプルごとかは抄録に無い | 教師ごと／ラベル無し | ○ | https://arxiv.org/abs/2509.20706 |
| 補：CASR（クラス別の最適輸送による重み），UMDA（クラス単位の重み），CAMSDA-RMM（医用画像） | 2022〜2025 | 多ソース DA でクラス単位の重みを扱う | 検索の断片のみ | クラスごと／ラベル無し | △ | https://arxiv.org/pdf/2011.02594 |

**A の読み**：ソースのサンプルやドメインに重みを付ける方法は一般の DA に多数あり，重みの根拠は「ターゲットとの近さ（密度・判別器・距離）」か「検証集合の損失」のどちらかである．感情のクラス（対）ごとに「どのソースの方向が使えるか」で重みを決めた研究は無い．ECAN と GLS の「クラスごとの重み」はラベルの割合の比であり，方向の良し悪しではない．感情認識では，Schuller 2011 がサンプルの典型性で選び，Ahn 2022 がコーパスの学びにくさで重みを付けるが，どちらもターゲットの情報を使わず，感情ごとに分けない．

### B. 感情の対ごと・クラスごとの転移可能性の解析と，クラスごとの DA

| 論文 | 会場/年 | 何をしたか | 測った・主張したこと | 水準 | URL |
|---|---|---|---|---|---|
| Milner, Jalal, Ng, Hain | ASRU 2019 | 演技・誘発・自然の 4 コーパスで感情ごとのコーパス間 UA/WA（音声） | happy を除く全感情で IEMOCAP・MOSEI 学習が良い（既読．`調査_文献.md` Q3） | ◎ | https://arxiv.org/abs/2207.02104 |
| Ismail | arXiv 2025 | IEMOCAP＋CREMA-D → RAVDESS（音声） | happy → anger，sad → neutral に流れる（既読） | ○ | https://arxiv.org/abs/2512.23435 |
| Shamsi & Tahon | arXiv 2022 | 次元ラベルからカテゴリへ | Angry の Neutral に対する位置がコーパス間で逆（既読） | ◎ | https://arxiv.org/html/2210.07642 |
| Zhou & Chen, "Transferable Positive/Negative SER via Class-wise Adversarial DA" (CADA) | ICASSP 2019 | Aibo（自然・子ども）→ EMODB（演技・大人）の正負 2 値．ソースとターゲットの「同じクラス」どうしを混同させる 4 クラスの判別器（ソース正負・ターゲット正負）で，クラスごとに合わせる．GeMAPS 62 次元＋MLP | ターゲットのラベル 2〜12 本（合計か 1 クラスあたりかは本文に明記なし）．UA：微調整 55→62，CADA 58→64（Aibo-Ohm）．**どのクラスが転移しやすいかの分析は無く，クラスを別々に重みづけてもいない**．多クラスは今後の課題 | ◎ | https://ar5iv.arxiv.org/html/1810.12782 |
| Ye ほか, Emo-DNA | ACM MM 2023 | 感情に関わる特徴とコーパス固有の特徴を分け，確信度の高い疑似ラベルのターゲットでクラス単位の整列 | クラスごとの転移しやすさは測らない | ○ | https://arxiv.org/abs/2308.02190 |
| MDAR | Entropy 2022 | 周辺・クラス条件つき・粗いクラス単位の分布を同時に合わせる | EmoDB・eNTERFACE・CASIA | ○ | https://www.mdpi.com/1099-4300/24/9/1250/htm |
| Li & Deng（ECAN） | 上記 | 顔の表情のコーパス間で，クラスごとの比率で重みづけ＋クラス条件つき MMD | RAF-DB 2.0 → CK+ で anger → sadness，fear → surprise．bias の分析は happiness と sadness を代表に選ぶが，どちらがコーパス間で強いかの文は無い（表の値は読み取れず） | ◎ | https://ar5iv.labs.arxiv.org/html/1904.11150 |
| Bostan & Klinger, "An Analysis of Annotated Corpora for Emotion Classification in Text" | COLING 2018 | 言葉の感情コーパス 14 本を同じ形式に統一し，コーパス内（10 分割）とコーパス間（1 つで学び他で測る，全部で学び 1 つで測る）を MaxEnt＋BOW で測る | コーパス内では感情ごとの F1 を表にする（例：ISEAR は joy 67，sadness 69，fear 69，anger 47，disgust 59）．**コーパス間は micro 平均の F1 だけで，感情ごとの転移は出していない**．「DailyDialogs は自分で学ぶより Blogs で学んだ分類器が最も良い」「全部で学ぶと良くなるとは限らない」．近さの尺度と性能の相関 r = 0.32 | ◎ | https://aclanthology.org/C18-1179.pdf |
| Oulu 大学の授業プロジェクト（CREMA-D） | 2026 | 域内で感情ごとに音声と顔のどちらが強いか | 顔：happy・disgust，音声：sad・neutral・angry（既読） | ○ | https://github.com/Leili-M/Multimodal-Emotion-Disagreement |
| Gaya-Morey ほか | IEEE TAC 17(3) | 顔の 24 データセットの移りやすさの類似度（既読） | 感情ごとではない | ○ | （`docs/related-work.md` 1.13） |
| Pándy ほか GBC（Bhattacharyya による転移しやすさ） | arXiv 2021 | ターゲットの各クラスをソースの特徴空間のガウスで表し，クラスの対ごとの重なりを足す | クラスの対ごとの重なりを使うが，報告はデータセットに 1 つの点数．**クラスごとの点数としては報告しない** | ○ | https://arxiv.org/abs/2111.12780 |

**B の読み**：音声では「喜び ↔ 怒り」「悲しみ ↔ 平静」の混同が複数の報告で一致する（既知）．顔では Li & Deng の 1 例（anger → sadness，fear → surprise）があるだけで，コーパス間の感情ごとの混同を体系的に並べた研究は見つからなかった．言葉では Bostan & Klinger がコーパス内の感情ごとの F1 を出すが，コーパス間は平均だけである．クラスごとの DA（CADA，Emo-DNA，MDAR，ECAN）は「クラスごとに合わせる」手法であって，「どのクラスがどのソースから転移するか」は測らない．転移しやすさの指標（LEEP，LogME，TransRate，GBC）はどれもデータセットに 1 つの点数で，クラスごとには出さない．

### C. モダリティの対ごと・クラスごとの重みづけ

| 論文 | 会場/年 | 何をしたか | 測った・主張したこと | 水準 | URL |
|---|---|---|---|---|---|
| Marín, Marí, Haro, Redondo, "MER in Conversations via Class-Wise Adaptive Modality Fusion and Affective Geometry" | arXiv 2609.09924（2026-09） | 会話の感情認識（MELD・IEMOCAP）で，各モダリティの表現と単モダリティの予測から「感情クラス c に対するモダリティ m の情報量」を 2 層 MLP で推し，クラスごとにモダリティ間 softmax で重みを作り，ロジットを融合する．**コーパス内・全ラベル・DA 無し** | 重みつき F1：softmax ゲート 75.76 → クラスごと 75.93（MELD），73.86 → 74.11（IEMOCAP）．モダリティごとだがクラスに依らない変種は 73.47 / 73.17 で，「モダリティに 1 つの点数では感情に依る情報量を表せない」．得は少数クラスに集中し，正負の感情での規則性は無し．**学んだクラスごとの重みの分析は今後の課題**と明記 | ◎ | https://arxiv.org/abs/2609.09924 |
| Tellamekala ほか, COLD Fusion | IEEE TPAMI（arXiv 2206.05833） | 各モダリティの偶然の不確かさを潜在分布から推し，融合の重みにする（較正と順序の制約） | AVEC 2019 CES，CMU-MOSEI，IEMOCAP．サンプルごとの重みで，感情クラスごとではない | ○ | https://arxiv.org/abs/2206.05833 |
| Wei ほか, "Enhancing Multimodal Cooperation via Sample-level Modality Valuation" | CVPR 2024 | 各サンプルについて，モダリティの Shapley 値（効用＝正解なら使ったモダリティ数，不正解なら 0）で貢献を測り，貢献の低いモダリティを再標本化する | Kinetics-Sounds，UCF-101，MM-Debiased（**感情データセットは無い**）．「同じクラスの中でも支配的なモダリティはサンプルで違う」．**クラスごとの分析は無い** | ◎ | https://ar5iv.labs.arxiv.org/html/2309.06255 |
| Liu ほか, MODfinity | CVPR 2025 | マルチモーダルの教師なし DA で，ターゲットの各サンプルに「教師のモダリティ」と疑似ラベルと確信度を付け（ラベルを独立したモダリティとみなす親和度で測る），上位 α% だけを使ってモダリティ間で蒸留する | 3 つのマルチモーダルのデータセット（名前は未確認．感情ではない）．サンプルごとにどのモダリティを信じるかをラベル無しで決める点が本研究の決め事 D10 に近い | ◎ | https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_MODfinity_Unsupervised_Domain_Adaptation_with_Multimodal_Information_Flow_Intertwining_CVPR_2025_paper.pdf |
| Zhang ほか, Amanda | Findings of ACL 2024 | 感情の多モーダル教師なし DA で，モダリティごとに整列の重みを代理損失で自動配分（既読） | 配分はモダリティ単位で，感情ごとではない | ◎ | https://aclanthology.org/2024.findings-acl.859.pdf |
| Jaritz ほか xMUDA（CVPR 2020），Munro & Damen MM-SADA（CVPR 2020），Zhang ほか audio-adaptive（CVPR 2022） | — | モダリティ間で予測をまねる・疑似ラベルを交換する多モーダル DA（既読） | モダリティの信頼はサンプル単位か全体．感情ではない | ○ | https://arxiv.org/abs/2203.14240 |
| Li ほか, "Towards Multimodal DG with Few Labels" | CVPR 2026 | 融合と単モダリティの予測が一致し確信度が高いサンプルだけ疑似ラベルに使う（既読） | 一般課題．規則は固定の閾値 | ○ | https://arxiv.org/abs/2602.22917 |
| SHAP による解釈（VISTANet，EmoMM ほか） | 2022〜2026 | モダリティの寄与を Shapley 値や SHAP で可視化 | 検索の範囲では感情ごとの表は無い | △ | https://arxiv.org/pdf/2208.11450 |

**C の読み**：「感情ごとにモダリティの重みを変える」は Marín 2026 が最も近く，コーパス内で学んだ重みが小さい得（+0.17，+0.25）を出すが，重みの中身は分析していない．サンプルごとにモダリティを選ぶ多モーダル DA（MODfinity，xMUDA 系）と Shapley 値によるモダリティの価値づけ（Wei 2024）は感情以外にある．**コーパス間の感情認識で，感情の対ごとにどのモダリティの方向が使えるかを測ったものは無い**．

### D. メタ学習で「何を使うか」を決める研究

| 論文 | 会場/年 | 何をしたか | 測った・主張したこと | 水準 | URL |
|---|---|---|---|---|---|
| **Guo, Shah, Barzilay, "Multi-Source DA with Mixture of Experts"** | EMNLP 2018 | ターゲットの各サンプルについて，ソースごとの専門家（分類器）の重みを，サンプルとソースの「点と集合の距離」（学習する低ランクのマハラノビス距離）から決める．**その距離は，ソースを 1 つずつ仮のターゲット（meta-target）にして残りをソースにする meta-training で学ぶ**．ターゲットはラベル無しのみ（MMD の敵対項でだけ使う） | Amazon の感情極性（Chen12 の平均：最良の単一ソース 82.25，合併 83.18，MoE 84.34），品詞（88.15 / 88.76 / 90.22）．「the metric is learned in an unsupervised fashion using meta-training」．重みはサンプルごとで，クラスごとではない．モダリティ無し | ◎ | https://ar5iv.labs.arxiv.org/html/1809.02256 |
| Li, Xu, Wang, Zhao, Kankanhalli, "GradMix" | arXiv 2020 | 複数ソースの勾配に**層ごと・ソースごとの重み**を付け，ターゲットの少数ラベル（検証集合）の損失が最も減るように毎ステップ解く（SLSQP）．ミニバッチの重要度と疑似ラベルも | 数字（SVHN＋MNIST 0–4 → MNIST 5–9，検証は 1 クラス 2〜5 本），行動認識．重みはサンプルやクラスではなく層とソースの単位 | ◎ | https://arxiv.org/abs/2002.03264 |
| Guo, Pasunuru, Bansal, "DistanceNet-Bandits" | AAAI 2020 | 多腕バンディットで学習中に使うソースドメインを切り替える | 感情極性．報酬の中身は抄録に無い | ○ | https://arxiv.org/abs/2001.04362 |
| Wei, Zhang, Huang, Yang, "Transfer Learning via Learning to Transfer" (L2T) | ICML 2018 | 過去の転移の経験から「何を・どう転移するか」を反射関数として学び，新しい対で最適化する | 安定性と汎化の上界．選ぶ対象は転移の仕方 | ○ | https://proceedings.mlr.press/v80/wei18a.html |
| Li, Yang, Song, Hospedales, MLDG | AAAI 2018 | ソースを meta-train と meta-test に分けてずれを模倣して学ぶ（既読） | 選択でなく初期値の学習 | ○ | https://arxiv.org/abs/1710.03463 |
| Li & Hospedales, "Online Meta-Learning for Multi-Source and Semi-Supervised DA" | ECCV 2020 | 既存の DA 手法（DANN・MCD・MME）の初期条件をメタ学習 | DomainNet ほか．ソースの重みを学ぶかは抄録に無い | ○ | https://arxiv.org/abs/2004.04398 |
| Gulrajani & Lopez-Paz, DomainBed | ICLR 2021 | 1 つ抜き交差検証で設定を選ぶ（既読） | 本研究の「仮のターゲットで候補から選ぶ」の直接の先行 | ◎ | https://arxiv.org/abs/2007.01434 |
| Qiu, Zhu, Zhou, "Meta Self-Learning for MSDA" | ICCV W 2021 | 多ソースの自己学習とメタ学習（既読） | 文字認識 | ○ | https://arxiv.org/abs/2108.10840 |
| **感情認識**：Gandhi, Tsouvalas, Meratnia, DA-MLDG | PerCom W 2023 | 10 コーパス・7 つの場面・5 感情で，ソースを meta-train と meta-test に分け，**meta-test の損失をドメイン間のエネルギースコアの差で重みづける** | 「コーパス間の汎化はむらがある」．数値は公開ページに無い．決めるのはドメインの重みで，クラスやモダリティではない | ○（GitHub の説明） | https://github.com/Nousphera/DA-MLDG |
| **感情認識**：Chopra, Mathur, Sawhney, Shah, "Meta-Learning for Low-Resource SER" | ICASSP 2021 | 言語を課題にして MAML．ペルシア語・イタリア語・ウルドゥー語がターゲット | 多課題・転移学習を上回る（抄録）．「何を転移するか」は決めない | ○ | https://rc.signalprocessingsociety.org/conferences/icassp-2021/spsicassp21vid0668 |
| **感情認識**：Fujioka ほか（ICASSP 2020），Meta-PerSER（Interspeech 2025），Ahn–Lee–Shin（SPL 2021），DTCML（CMC 2025，言葉） | — | 域内のラベルの曖昧さ，聴取者ごとの適応，少数ショット＋DANN，言葉の少数ショットのメタ学習（既読か抄録） | どれもソースやモダリティの選択を学ばない | ○ | https://www.sciopen.com/article/10.32604/cmc.2024.059115 |

**D の読み**：「ソースを 1 つずつ仮のターゲットにして，どのソースを使うかを学ぶ」は Guo–Shah–Barzilay 2018 にある（サンプル単位，言葉，ラベル無しターゲット）．「少数のターゲットラベルでソースの重みを決める」は GradMix（層とソースの単位）にある．感情認識のメタ学習（DA-MLDG，Chopra 2021）は初期値かドメインの重みを学ぶだけで，感情のクラスやモダリティの単位で「何を転移するか」を決めたものは無い．

### E. DA の理論で，クラスごと・ソースごとの価値

「ラベル何本ぶんか」の換算の先例は `research_notes/ラベル換算の価値で重みを決める先例/report.md`（Jain–Montanari–Sasoglu 2024，CLIP，catalytic prior，power prior ほか）で調べ済みなので，ここにはクラスごと・ソースごとの分解に関わるものだけを足す．

| 論文 | 会場/年 | 何をしたか | 測った・主張したこと | 水準 | URL |
|---|---|---|---|---|---|
| Hanneke & Kpotufe, "On the Value of Target Data in Transfer Learning" | NeurIPS 2019 | ソース n_P 本・ターゲット n_Q 本のミニマックス率．転移指数 ρ でソースの実効標本数 ñ_P を定義（既読） | 「ソースかターゲットの弱い方を無視する」のが最適率．付録 A.2 に複数ソースからの選択．**クラスごとの分解は無い** | ◎ | https://arxiv.org/abs/2002.04747 |
| Mansour ほか 2021（上記） | AISTATS 2021 | 少数ラベルでの混合の選択の上界と下界 | ソース単位．クラス単位は無い | ◎ | 上記 |
| Maity, Dutta, Terhorst, Sun, Banerjee, "A linear adjustment-based approach to posterior drift in transfer learning" | Biometrika 111(1):31–50, 2024 | 入力の分布は同じでラベルの付き方（事後確率）だけが違う「事後のずれ」を，ソースの回帰関数の線形の調整でモデル化し，少数のターゲットで学ぶ．2 値分類のミニマックス率 | ラベルの付き方のずれが「線形の調整」で済むという仮定が，本研究の「位置の誤りは安く直る」と同じ向き．**多クラスでクラスごとの調整量を分けた記述は抄録に無い** | ○ | https://arxiv.org/abs/2111.10841 |
| Cai & Wei（Ann. Statist. 2021），Reeve–Cannings–Samworth（2021），Kpotufe & Martinet（2021） | — | 事後のずれ・共変量のずれの下でのミニマックス率（既読） | クラスごとの価値は出さない | ◎ | （`docs/related-work.md` 付録） |
| Li, Han, Yang, Tong, "Quantifying Cross-Domain Knowledge Distillation in the Presence of Domain Shift" | ICML 2026 | 線形回帰で，教師の予測と真のラベルを混ぜる最適な重みを閉じた形で出す（既読．1.16 節） | 教師の価値をラベル本数で読むと N₂ に依らない（本調べの推論）．クラス無し | ◎ | https://proceedings.mlr.press/v306/ |
| Schoch, Xu, Ji, "CS-Shapley: Class-wise Shapley Values for Data Valuation in Classification" | NeurIPS 2022 | 各学習サンプルの価値を「自分のクラスへの貢献」と「他のクラスへの貢献」に分けて Shapley 値で測る | 自分のクラスを害するサンプルは雑音や誤ラベル．**同じ分布の中の話で，ソースとターゲットの違いは無い** | ○ | https://ar5iv.labs.arxiv.org/html/2211.06800 |
| Just ほか, "LAVA: Data Valuation without Pre-Specified Learning Algorithms" | ICLR 2023 | 学習集合と検証集合のクラスごとの Wasserstein 距離の感度で各サンプルの価値を出す | 学習器に依らない．検証集合はラベル付き | ○ | https://ar5iv.labs.arxiv.org/html/2305.00054 |
| Parvez & Chang, SEAL-Shap（上記） | NAACL 2021 | ソース単位の Shapley 値 | クラス無し | ○ | 上記 |
| Morita, Thall, Müller, "Determining the effective sample size of a parametric prior" | Biometrics 64:595–602, 2008 | 事前分布を「観測何件ぶんか」に換算する定義 | ベイズ統計の換算の原型（power prior は既調査） | ○ | https://ideas.repec.org/a/bla/biomet/v64y2008i2p595-602.html |
| Ben-David, Lu, Luu, Pál, "Impossibility Theorems for DA" | AISTATS 2010 | ターゲットのラベル無しでは，重みづけ直しの手法は「学習データから区別できない 2 つの課題」で片方を悪化させる（既読） | ラベル無しでソースの価値を決めることの限界 | ◎ | https://proceedings.mlr.press/v9/david10a.html |

**E の読み**：ソースの価値をターゲットのラベル本数に換算する考えはある（Hanneke & Kpotufe の実効標本数，Jain 2024，CLIP，power prior）．サンプルやソースの価値を Shapley 値で測る研究もある（Data Shapley，CS-Shapley，LAVA，SEAL-Shap）．**ソースのコーパス × 感情の対 × モダリティの単位で価値を分解した理論も測定も見つからなかった．**CS-Shapley は「クラスごと」だが同じ分布の中の話である．

## 3. 最も近い先行研究との 1 本ずつの比較

研究目標が達成された形を「達成物」と呼ぶ：ソース（6 コーパス 7 群）× 感情の対（6 対）× モダリティ（音声・顔・言葉・融合）で「ソースだけで学んだ分類器の方向がターゲットの対をどれだけ分けるか」を測り，対ごとに最良のソース・モダリティが変わるかと選ぶ余地を話者の半分で確かめ，その量から設定を出す規則をソースのコーパスを仮のターゲットにして学ぶ．

### 3.1 Guo, Shah, Barzilay 2018（EMNLP）◎

| 項目 | 先行研究 | 達成物 |
|---|---|---|
| 設定 | 言葉（感情極性・品詞），ソース 3〜4 ドメイン，ターゲットはラベル無し，深層の符号化器 | 音声・顔・言葉の凍結特徴の上の線形分類器，ソース 6〜7 群，ターゲットに感情ごとに K 本 |
| 測る量 | ターゲットのサンプルとソースの「点と集合の距離」 | ソースの方向が対を分ける度合い（感情の対ごと，モダリティごと） |
| 決める単位 | サンプルごとにソースの重み | 感情の対ごと・モダリティごとにソースの方向 |
| メタ学習 | ソースを 1 つずつ仮のターゲットにして距離を学ぶ | 同じ骨組み（仮のターゲット）で，対ごとの量から設定を出す規則を学ぶ |
| 同じ点 | 「ソースを 1 つずつ仮のターゲットにして，どのソースを使うかを学ぶ」 | — |
| 違う点 | クラスごとでない．モダリティが無い．少数ラベルを使わない．位置と方向の分解が無い | — |

### 3.2 Marín, Marí, Haro, Redondo 2026（arXiv）◎

| 項目 | 先行研究 | 達成物 |
|---|---|---|
| 設定 | MELD・IEMOCAP のコーパス内，全ラベル，Transformer の融合 | コーパス間，K 本 |
| 測る量 | 感情クラス c に対するモダリティ m の情報量（学習で推す） | 感情の対ごとのモダリティの方向の使える度合い（ソースの分類器から直接測る） |
| 主張 | モダリティに 1 つの点数では感情に依る情報量を表せない（73.47 → 75.93） | 対ごとにモダリティを分ける余地があるかを半分で測る |
| 同じ点 | 「感情ごとにモダリティの重みを変える」という発想 | — |
| 違う点 | DA 無し，ソース無し，学んだ重みの分析は今後の課題と明記（本研究の表はその分析にあたる） | — |

### 3.3 Schuller, Zhang, Weninger, Rigoll 2011 ○

| 項目 | 先行研究 | 達成物 |
|---|---|---|
| 設定 | 音声のみ，覚醒・感情価の 2 値，複数コーパス，ターゲットの情報なし | 3 モダリティ，4 感情の 6 対，K 本 |
| 測る量 | 学習サンプルの典型性（特徴空間の距離） | 方向の使える度合い |
| 主張 | 典型的なサンプルを選ぶと覚醒で最大 7.5% 改善，感情価は効かない | 対ごとに選ぶ余地があるか |
| 同じ点 | 感情認識でソースの学習データを選ぶ | — |
| 違う点 | 感情のクラスごとでない．ターゲットの情報を使わない．選ぶ側と測る側を分けていない | — |

### 3.4 Li & Deng（ECAN）◎

| 項目 | 先行研究 | 達成物 |
|---|---|---|
| 設定 | 顔の表情，7 表情，コーパス間，ターゲットはラベル無し，深層の微調整 | 3 モダリティ，4 感情，K 本，凍結特徴 |
| 測る量 | クラスごとのラベルの比率 P^t(l)/P^s(l)（疑似ラベルから）と条件つき MMD | 方向の使える度合い（対ごと） |
| 主張 | コーパス間で 63.83% → 39.40%．anger → sadness，fear → surprise | 対ごと・ソースごとに違う |
| 同じ点 | ソースにクラスごとの重みを付ける | — |
| 違う点 | 重みは割合の比（本研究の決め事 D9「釣り合い直し」に相当）で，方向の良し悪しではない．ソースは 1 つ | — |

### 3.5 Mansour, Mohri, Ro, Suresh, Wu 2021（AISTATS）◎

| 項目 | 先行研究 | 達成物 |
|---|---|---|
| 設定 | 複数ソース＋ターゲットの少数ラベル m₀ 本，一般の仮説集合 | 同じ設定（ソース 6〜7，K 本） |
| 測る量 | 混合比の候補ごとのモデルの，少数ラベルでの損失 | 対ごとの方向の使える度合いと，仮のターゲットでの選択 |
| 主張 | 少数ラベルで選ぶ方法の余分の誤差は O(√(ソース数/m₀))．対ごとのずれで重みを決めると最適でないことがある | 対ごとに選ぶ余地が δ 以上かを数える |
| 同じ点 | 「候補から少数ラベルで選ぶ」の保証を使える | — |
| 違う点 | クラスごとでない．本研究は選択をターゲットの K 本でなく仮のターゲットで行うので，この理論の保証は直接は当てはまらない．**論文では，仮のターゲットでの選択と K 本での選択を並べ，どちらが勝つかを K ごとに示す必要がある** | — |

## 4. 狙うべき差分

`DAI_paper/inventory.md` で既に測れている，または保存済みの出力から出せるものを先に置く．

1. **ソース × 感情の対 × モダリティの表（設計済み．保存済みの出力から CPU で出せる）**．先行研究のクラスごとの重みは「割合の比」（ECAN，GLS）か「学習した注意の可視化」（Bi-ATEN）で，「どのソースのどのモダリティの方向がどの対で使えるか」を測って表にしたものは無い．選ぶ側と測る側を話者の半分で分ける手続きは，Schuller 2011 にも Bi-ATEN にも無い．この表は Marín 2026 が「今後の課題」とした「学んだクラスごとの重みの分析」を，学習を介さずソースの分類器から直接出すことにあたる．
2. **対ごとにソースを選ぶ余地の判定**．Mansour 2021 の「対ごとのずれで重みを決めるのは最適でないことがある」に対し，本研究は「6 対の平均で最良の 1 つ」と比べて対ごとの選択に δ 以上の余地があるかを数える．余地が無ければ「全ソースを使う今の決め方を変えない」と書け，これも先行研究に無い否定の結果になる（感情認識ではソースの選択がコーパス単位でも発話単位でも「全部使う」に勝てなかった既存の記録と合わせて書く）．
3. **位置と方向に分けた上での資源と対の対応**．棚卸しの行 17（方向の誤りは感情の対ごとに測れ，音声では怒りを含む 3 対で崩れる），行 18（位置を直す割合は対に依らない），行 22（文献：音声は喜び ↔ 怒り，悲しみ ↔ 平静で混ざる）を，「K 本は位置を直す，他モダリティの疑似ラベルは方向を直す」と組み合わせ，**どの資源がどの対の何を直すか**を書く．クラスごとの DA（CADA，Emo-DNA，MDAR，ECAN）はこの分解を持たない．
4. **メタ学習で学ぶものを「規則」にする**．Guo 2018 はサンプルとソースの距離を学び，GradMix は層とソースの重みを学ぶ．本研究は「対ごと・モダリティごとの方向の使える度合い」という測った量から設定（疑似ラベルのモデルの組み方，対ごとの配分，モダリティの重み）を出す規則を，ソースを仮のターゲットにして学ぶ．規則が「候補からの選択」に勝つかは棚卸しの行 23 の取りこぼし（テスト最良 +6.3 に対し選ばれた候補 +3.9）で動機づけられる．ただし線形の分類器では 6 対の方向が 4 本の重みで縛られるので，対ごとに別のソースを使えるかは別に示す（設計の限界の節に既に書いてある）．
5. **仮のターゲットでの選択と K 本での選択の比較を K ごとに書く**．Mansour 2021 の保証は K 本で選ぶ側にある．CLAP（既読．1.13 節）は「別のデータセットで調整した設定は最大 21% 落ちる」と報告している．本研究は K が小さいと K 本で選ぶ方法の損が大きい（既存の記録）ので，「K が小さいときは仮のターゲット，大きいときは K 本」という境目を示せる．
6. **ソースの価値をクラスの対ごとに分ける**（後回し）．「1 ソース ≈ 感情ごとに約 10 本」（棚卸しの行 3）を対ごと・モダリティごとに分けると，CS-Shapley や SEAL-Shap が持たない「ソース × 対 × モダリティ」の価値の分解になる．Maity 2024 の「事後のずれは線形の調整で済む」は，位置の誤りが安く直ることの理論の側の引用先になりうる（抄録のみ．引く前に本文を確かめる）．

## 5. 見つからなかったこと

- コーパス間の感情認識（音声・顔・言葉のどれでも）で，「どのソースの方向がどの感情の対で使えるか」をソース × 対 × モダリティで表にした研究は，今回の検索では見つからなかった．
- 感情のクラス（対）ごとにソースのサンプルやコーパスを選ぶ手法は，感情認識では見つからなかった（Schuller 2011 はサンプルとデータベースの単位，Ahn 2022 はコーパスの単位）．一般の DA でも，クラスごとの重みは割合の比（ECAN，GLS，UMDA）か学習した注意（Bi-ATEN）で，「クラスごとに別のソースを使う」手法は見つからなかった．
- 感情ごと・対ごとの転移しやすさの指標は見つからなかった．LEEP・LogME・TransRate・GBC はどれもデータセットに 1 つの点数である．
- 感情認識で，メタ学習によって「どのソース・どのモダリティをどの感情に使うか」を決めた研究は見つからなかった（DA-MLDG はドメインの重み，Chopra 2021 は言語を課題にした MAML）．
- Shapley 値によるモダリティの価値づけ（Wei 2024）を感情のデータで感情ごとに出したものは見つからなかった．
- 顔と言葉で，コーパス間の感情ごとの混同を体系的に並べた研究は見つからなかった（顔は Li & Deng の 1 例，言葉は Bostan & Klinger のコーパス内の表だけ）．
- ソース × クラス × モダリティの単位でソースの価値をラベル本数に換算した理論は見つからなかった．

## 6. 調査の限界

- Schuller 2011（典型性）の本文は TUM のリポジトリが bot 対策で開けず，抄録と二次資料だけで判断した．Hassan 2013 も抄録の水準である．論文で引く前に本文を確かめる．
- Semantic Scholar の API は回数制限で使えなかった．dblp・OpenReview・IEEE Xplore は直接検索していない．
- 検索は英語のみで，中国語・日本語の論文誌は探していない．2026 年の arXiv は検索結果に出た範囲（Marín 2026，Bi-ATEN 2024，MODfinity 2025，MI-FUSE 2025）に限る．
- MODfinity のデータセット名，DA-MLDG の数値，DistanceNet-Bandits の報酬の中身は確かめていない．
- 読み込んだ本文の一時ファイルは，このセッションの scratchpad（`pw/`．Marín 2026，GradMix，Bostan 2018，MODfinity，Ahn 2022，Parlak 2014）にあり，セッションが終わると消える．
