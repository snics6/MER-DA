**検証の結果（2026-11-12，`verify_emotion.md`）**：Reif & Schwartz 2024 の「ラベルの数が同じ k=16」は誤り（text.md の冒頭の注を見ること）．Myung の頁は印刷の頁で 110〜112．Haider の数・作者・公開日・テストのラベルで閾値を選ぶことはコードで確かめた．CAST は本文未確認のまま，公開コードには切片を当て直して誤りを分ける処理は無い．

# 先行研究の調べ直し 第 2 回：誤りを切片側と向き側に分けた研究（言葉・声・顔の続き）（2026-11-12）

第 1 回（`text.md`，`speech_fusion.md`，`face.md`）で検索の上限のために使えなかった検索語と，読めなかった本文だけを調べた．問いと判定の段階（完全一致，部分一致，関係，該当しない）と水準の記号（◎ 本文の該当箇所を読んだ，○ 抄録か信頼できる要約，△ 検索の断片か書誌だけ）は第 1 回と同じである．

## 0. 要約（第 1 回の判定は変わるか）

1. **第 1 回の判定（言葉・声・顔のどれにも完全一致は無い）は変わらない．**今回の 16 回の検索と 9 本の本文の確認でも，4 つの情報源（ターゲットの少数ラベル，ほかのデータセット，ターゲットのラベル無しデータ，ほかのモダリティの予測）のどれが切片側と向き側のどちらを減らしたかを感情の実データで測った研究は無かった．
2. **言葉の部分一致が 3 本増えた．**最も近いのは Haider の公開の解析（2026 年 8 月．GitHub と Hugging Face の公開ページで，査読は無い）である．凍結した DistilBERT の特徴に線形の分類器を載せ，感情極性の 4 コーパスの間で性能の落ちを「閾値を当て直せば戻る分（placement）」と「戻らない分（ranking）」に分けた．線形の分類器では落ち 5.87 点のうち 1.76 点（29.9%）が閾値で戻る分だった．情報源は「ほかのデータセットで学ぶ」の 1 つだけである．
3. 残りの 2 本は次のとおり．Myung（2026．ToM4AI ワークショップ）は GoEmotions から SemEval-2018 E-c への感情の転移で，感情ごとの閾値だけを当て直すと macro-F1 が 0.14 から 0.54 に上がった（切片側だけを測った）．Xian ほか（ICLR 2022）は言語をまたぐ感情極性の転移で，「クラスごとの特徴のずれ」と「クラスの割合のずれ」を別々に測り，どちらも成績と相関することを示した（ずれを 2 つに分けたが，誤りは分けていない）．
4. Reif & Schwartz 2024 の図の値を読み取った．Llama-2 7B で同じ 16 件のラベルを使う場合，切片だけを当て直す方法（LOOC）で取り戻せたのは，モデル全体を学ぶ方法（LoRA）の得のおよそ 5〜7 割だった．本文の「ほぼ埋まる」は図の値より強い言い方である．感情極性の課題だけの結果は本文にも付録にも無い．
5. 声では切片だけを当て直す研究は今回も見つからなかった．Abdelwahab & Busso 2015 と Busso ほか 2013 の本文を読み，どちらも誤りを分けていないことを確かめた．顔の CAST（Wang & Liu 2025）は arXiv 版が無く，IEEE の頁も開けなかったため，今回も判定できなかった．
6. 狙うべき差分は第 1 回と同じで，同じ分け方で情報源ごとの得を並べることである．ただし「凍結した特徴の線形の分類器で，ほかのデータセットからの転移の落ちを閾値側と残りに分ける」こと自体には，上の Haider の公開の解析という先例がある．論文ではこの点を新しさとして主張せず，情報源ごとに並べること，4 クラスの感情であること，複数のモダリティであることを新しさとして書く必要がある．

---

## 1. 言葉（text.md の 6.2 節の続き）

### 1.1 研究ごとの表

| 研究 | 設定 | 測った量 | (a)(b) を分けたか | 判定 | 水準 | URL |
|---|---|---|---|---|---|---|
| Haider "undertow"（2026 年 8 月 16 日公開．GitHub の README と Hugging Face の公開ページ．査読なし） | 感情極性の 2 値．IMDB，SST-2，Yelp，Amazon から 6,000 件ずつ．分類器は 2 種類で，TF-IDF＋ロジスティック回帰と，凍結した distilbert-base-uncased の特徴に載せた線形の分類器である．4×4 の組で，あるコーパスで学び別のコーパスで評価する | 落ち＝コーパス内の正解率 − コーパス外の正解率．「オラクルの閾値」（ターゲットで閾値だけを最良の位置に動かしたときの正解率）で落ちを 2 つに分けた．閾値で戻る分を placement，残りを ranking と呼ぶ．12 組の平均で，TF-IDF は落ち 8.30 点＝ranking 6.74＋placement 1.57（placement の割合 18.9%），線形の分類器は落ち 5.87 点＝ranking 4.12＋placement 1.76（29.9%）．最も目立つ例は SST-2 で学んで Yelp で評価した TF-IDF で，正例が 50% のデータに対して正と予測したのは 32.0% だけだった．AUROC は 85.9 で，閾値を動かすだけで正解率が 74.4% から 77.5% になった | **分けた．**問いの (a)(b) と同じ分け方である（2 値なので閾値＝切片）．ただし情報源は「ほかのデータセットで学ぶ」だけで，少数ラベル，ラベル無し，ほかのモダリティの得は分けていない．オラクルの閾値はターゲットのテストのラベルで選ぶ．また，落ちを「ソースのテストでの正解率」から測るので，ranking の分にはターゲットのほうが難しいことによる差も入る（著者は区別していない．この指摘は本報告の推測） | 部分一致（測る量は最も近い．査読なし） | ◎（README の 4 節の表と本文．公開ページの図は JavaScript で読めなかった） | https://github.com/UsmarHaider/undertow ，https://usmar-undertow.static.hf.space/ |
| Myung, "Toward Theory of Mind: BERT Learns and Uses Emotion Geometry in Two Phases"（ToM4AI ワークショップの論文集 arXiv 2603.18786 の 109〜111 頁．2026） | GoEmotions で学んだ BERT を，SemEval-2018 Task 1 E-c に零ショットで移す（共通の 8 感情．多ラベル）．閾値は 3 通り：0.5 固定，全感情で共通の 1 つを選ぶ，感情ごとに選ぶ．閾値は「development split」で選ぶ | 外部（E-c）での BERT の macro-F1：0.5 固定 0.1416，共通の閾値 0.4647，感情ごとの閾値 0.5431．提案の EmotionBERT は 0.1979 → 0.4697 → 0.5846．GoEmotions の中では BERT 0.3144 → 0.4300 → 0.4528 | **切片側だけを測った．**閾値（多ラベルの各感情では切片にあたる）だけを当て直すと，外部での macro-F1 が 0.14 から 0.54 まで上がる．残る誤りは測っていない．外部の評価で閾値を選んだ「development split」が E-c のものか GoEmotions のものかは本文に書かれていない（0.5 固定からの上がり幅の大きさから E-c のものと推測する） | 部分一致（感情の転移で切片側だけ） | ◎（表 1 と本文） | https://arxiv.org/abs/2603.18786 |
| Xian, Ji, Zhao, "Cross-Lingual Transfer with Class-Weighted Language-Invariant Representations"（ICLR 2022） | mBERT．感情極性（MARC．5 段階の星）で英語から 5 言語へ零ショットで移す．クラスの割合を変えて間引いた 500 個のテスト集合で評価した | 2.1 節は「クラスごとの特徴のずれ」（クラスごとの特徴の平均の差の ℓ1 の距離を，クラスについて平均したもの）を，2.2 節は「クラスの割合のずれ」（全変動距離）を別々に測った．図 1 で，どちらのずれも，もう一方をそろえたうえで macro-F1 と強く相関すると述べた．提案の IWDA（ラベル無しのターゲットで割合を推定し，割合で重みづけて特徴をそろえる）と，割合を知っている場合（oracle）を比べた．表 1 の正解率は，英語→日本語で零ショット 37.37，IWDA 42.62，oracle 44.91 | **ずれを 2 つに分けたが，誤りは分けていない．**割合のずれ（(a) の原因）と特徴のずれ（(b) の原因）を別々に測ったが，学習の後に切片だけを当て直す比べは無い．IWDA の割合の補正も表現を学び直す中に入っている | 部分一致（声の Tran 2023 と同じ型） | ◎（2 節，4.1 節，表 1） | https://rxian.github.io/assets/xian2022crosslingual.pdf |
| Reif & Schwartz 2024（NAACL）の図 3a の値（第 1 回の表の続き） | Llama-2 7B の base と chat．279 課題の平均．LOOC は k 件のデモで切片を当て直す．LoRA は別の 16 件で学び，k 件のデモで評価する（k ∈ {0, 8, 16}．LoRA は 7B だけ） | 図 3a から読み取った macro-F1（目盛りから読んだので ±0.005 程度の誤差がある）．k=16 の base：補正なし 0.486，LOOC 0.601，LoRA 0.645．k=16 の chat：0.585，0.632，0.680．k=8 の base：0.466，0.570，0.634．k=8 の chat：0.573，0.612，0.680．LOOC で取り戻せた割合（LOOC の得 ÷ LoRA の得）は，ラベルの数が同じ k=16 で base 約 0.72，chat 約 0.49 である．13B は LoRA が無く，k=16 で補正なし 0.568，LOOC 0.655（base） | 第 1 回の判定（分けていない）は変わらない．本文の「LOOC nearly closes the gap with LoRA-level performance」は，図では 5〜7 割にあたる．感情極性（39 課題）だけの値は，本文にも付録の表 1〜9 にも無い（付録の表はデモの選び方の比べだけ）．実線を base，破線を chat と読んだのは，本文の「CC は instruction-tuned で性能を下げる」と図の線の位置からの推測である | 部分一致（第 1 回と同じ） | ◎（本文と付録の表は pdftotext で読んだ．図は画像で読み取った） | https://arxiv.org/html/2405.02743v1 |
| Opitz, "Schroedinger's Threshold: When the AUC doesn't predict Accuracy"（arXiv 2404.03344） | 生成文の事実性を判定する評価器（TRUE ベンチマーク）．感情は無い | AUC と，ほかのデータで較正して閾値 0.5 で判定した正解率を比べると，順位が大きく入れ替わる | 分けていない．AUC が楽観的であることを示すだけ | 関係（感情ではない） | ○（WebFetch による本文の要約） | https://arxiv.org/html/2404.03344v2 |
| SemEval-2025 Task 11 の参加システム（RSSN，DUT_IR） | 多ラベルの感情検出．感情ごとに閾値を開発用のデータで探す | 閾値の調整で F1 0.7062（RSSN） | 分けていない．コーパス内である | 該当しない | △（検索の断片） | https://aclanthology.org/2025.semeval-1.105 ，https://aclanthology.org/2025.semeval-1.17 |
| 言語をまたぐ感情の転移（ICNLSP 2026） | 多言語の感情の分類 | 抄録の断片に「出力の較正に敏感」とあるだけ | 分けていない（断片の範囲） | 該当しない | △ | https://aclanthology.org/2026.icnlsp-1.24.pdf |

### 1.2 検索語ごとに分かったこと

- **AUROC と正解率の差**：Haider の解析が問いの分け方そのものを使っていた（上の表）．査読つきの論文で同じ分け方を感情極性に当てたものは，今回も見つからなかった．
- **vector scaling（クラスごとの尺度と切片）と matrix scaling（全体の線形変換）の比べ**：文章の分類で，分布のずれの下でこの 2 つを比べた研究は見つからなかった．出てきたのは一般の深層学習の較正の論文（arXiv 1911.11195）だけである（△）．
- **言語をまたぐ転移と割合のずれ**：Xian ほか 2022 が見つかった（上の表）．感情（多クラス）で同じ分析をした研究は無かった．
- **GoEmotions や SemEval E-c に Batch Calibration を当てた研究**：見つからなかった．GoEmotions の ICL の研究（arXiv 2510.06600）は「判定の側」を論じるが，ラベルの偏りの較正も誤りの分け方も無いと本文の要約で確かめた（○）．
- **文章の分類で最終層の切片だけを学び直す研究**：見つからなかった．出てきたのは最終層の重みも学び直す研究（DFR，AFR，TLDR）で，第 1 回の判定と同じである．
- **感情ごとの閾値の調整（多ラベル）**：コーパス間の例は Myung 2026 だけだった．SemEval-2025 の参加システムはコーパス内である．
- **声や顔の予測で言葉の分類器を学ぶ研究**：今回も見つからなかった．出てきたのは向きが逆のもの（言葉の教師から声の生徒へ，顔の教師から声の生徒へ．LiSER，KBS 2021）である．

---

## 2. 声（speech_fusion.md の #59・#60 の続き）

### 2.1 研究ごとの表

| 研究 | 設定 | 測った量 | (a)(b) を分けたか | 判定 | 水準 | URL |
|---|---|---|---|---|---|---|
| Abdelwahab & Busso, "Supervised Domain Adaptation for Emotion Recognition from Speech"（ICASSP 2015） | IEMOCAP または SEMAINE で学んだ線形の SVM を，RECOLA のラベル付きの発話で直す．覚醒度と感情価の高低の 2 値（z 値で ±0.3 の間は捨てる）．直し方は adaptive SVM（元の境界の近くで，新しいラベルを分ける境界を探す）と，逐次の SVM の学習 | 直しに使う話者の数（2，8，22 人．データの約 9%，35%，全部）を振った正解率．覚醒度は 2 人（約 9%）でも有意に上がった．8 人と 22 人の差は有意でない．話者の多様さは効かない | **分けていない．**adaptive SVM は重みと切片を一緒に直す．切片だけを直す比べは無い | 該当しない（第 1 回の △ を ◎ で確かめた） | ◎（2〜5 節） | http://www.lab-msp.com/MSP/publications/Abdelwahab_2015.pdf |
| Busso, Mariooryad, Metallinou, Narayanan, "Iterative Feature Normalization Scheme for Automatic Emotion Detection from Speech"（IEEE TAC 2013） | 平静と感情の 2 値の検出．線形の SVM．話者ごとに，平静と判定した発話の平均と標準偏差で z 正規化する（IFN）．IEMOCAP で学び，管理されていない録音でも評価した | 5.4.1 節：テストが平静 727 本と感情 110 本で偏っている場合，正規化なしの正解率は 18.2%，全体での正規化は 55%．平静の発話で正規化する理想の場合は全体の正規化より 19 点以上高い．5.2 節：正規化の値を全発話から推定して始めると，平静から感情へ判定が変わる発話が多くなると書く | **分けていない．**話者ごとの z 正規化は，固定した線形の分類器では切片だけでなく重みの尺度も変える（式からの推測）．正規化に使う発話の割合が判定を一方のクラスに寄せることを観察したが，その分を量で切り出していない | 関係（第 1 回の ○ を ◎ で確かめた） | ◎（1〜2 節，5.1〜5.4 節） | http://www.lab-msp.com/MSP/publications/Busso_2013_2.pdf |
| Lefter, Baird, Stappen, Schuller, "A Cross-Corpus Speech-Based Analysis of Escalating Negative Interactions"（Frontiers in Computer Science 2022） | オランダ語の 3 コーパス．攻撃性の 3 段階．線形の SVM．コーパス間と合併 | UAR と WAR．合併が効くかどうかは，ラベルの定義の近さで決まると結論した．非役者の UAR 40.4% と WAR 70.5% の差を，クラスの偏りによると説明した | 分けていない．閾値，較正，割合の補正は無い | 関係（コーパス間の失敗の原因としてラベルの定義を挙げた） | ○（WebFetch による本文の要約） | https://www.frontiersin.org/articles/10.3389/fcomp.2022.749804/full |
| Łachut, "Inference-Only Speaker Adaptation Improves Cross-Lingual Speech Emotion Recognition"（PolEval 2025） | HuBERT を多言語の CAMEO で微調整して固定し，ポーランド語の nEMO に移す．推論の時点で話者ごとに埋め込みを k 平均でまとめ，まとまりごとにラベルをそろえる | macro-F1 0.619 → 0.753 | 分けていない | 該当しない | ◎（抄録，2〜4 節） | https://aclanthology.org/2025.poleval-main.12.pdf |

### 2.2 検索語ごとに分かったこと

- **#59（コーパス間の失敗の原因の分析）**：原因として挙がるのは，ラベルの定義，話者，言語，録音の条件である（Lefter 2022，Zhang・Mower Provost・Essl，Ng ほか ASRU 2019．後の 2 本は △）．どの研究も，原因ごとの影響が切片側に出るか向き側に出るかは測っていない．
- **#60（声で切片だけを当て直す手）**：今回も見つからなかった．追加の 3 つの検索語（ラベル無しのターゲットで EM により割合を推定する手と微調整の比べ，少数のラベルで閾値を当て直す手，凍結した特徴での logit adjustment と線形の分類器の学び直しの比べ）でも見つからなかった．SER Evals がロジット調整を評価の手続きに入れていることは第 1 回と同じで，その有無の比べは無い．
- **Busso 2013 と Abdelwahab 2015 の本文**：MSP 研究室の旧ドメイン（lab-msp.com）から読めた．判定は上の表のとおりで，第 1 回の判定は変わらない．

---

## 3. 顔（face.md の CAST）

- **CAST（Wang & Liu, IEEE TAC 16(3):1618–1630, 2025. DOI 10.1109/TAFFC.2025.3529947）の本文は今回も読めなかった．**arXiv の API で題名と著者を検索したが該当は無かった．Semantic Scholar の記録では公開の PDF は無い（openAccessPdf が空）．IEEE Xplore の頁（document 10843182）は取得に失敗した．著者のコードの公開も見つからなかった．
- **抄録だけからの判定（○）**：抄録は「クラスの分布が疑似ラベルの学習に与える悪影響を，クラスごとの表現の分布と予測確率の 2 つの面から分析し，両方をそろえる」と書く．「予測確率」の面は切片側（クラスの割合）に，「クラスごとの表現の分布」の面は向き側に近い．しかし，そろえる手はどちらもネットワークの学習の中に入る（「jointly matching class-level representation distributions and class distributions」）．このため，第 1 回の ECAN と同じく，手法の部品を分けて効果を見た研究である可能性が高い（推測）．切片だけを当て直した場合と残る誤りを量で分けたかは確かめられない．判定は第 1 回と同じ「部分一致（弱い）．本文未確認」のままとする．

---

## 4. 最も近い先行研究（第 2 回で更新）

研究目標が達成されたとき（4 つの情報源の得を切片側と向き側に分けて並べたとき）に最も近いものを，第 1 回の 3 つの報告と合わせて選び直した．

1. **Haider "undertow"（2026．査読なし）**
   - 設定：感情極性の 2 値，4 コーパス．凍結した DistilBERT の特徴に線形の分類器を載せる．ソースのラベルだけで学ぶ．
   - 測った量：コーパス外での落ちを，閾値を最良にすれば戻る分と戻らない分に分けた．
   - 主張：落ちの大部分（線形の分類器で約 70%）は閾値では戻らない．
   - 同じ点：測る量の定義が問いの (a)(b) と同じである．凍結した特徴の線形の分類器であることも同じである．
   - 違う点：情報源が 1 つ（ほかのデータセットで学ぶ）だけで，少数ラベル，ラベル無し，ほかのモダリティの得を分けていない．2 値で，感情の多クラスではない．閾値はテストのラベルで選ぶオラクルである．査読を経ていない．
2. **Reif & Schwartz 2024（NAACL）**
   - 設定：LLM の ICL，279 課題（感情極性 39）．
   - 測った量：同じ 16 件のラベルを，切片だけの当て直し（LOOC）とモデル全体の学習（LoRA）に使ったときの macro-F1．
   - 主張：切片だけで差がほぼ埋まる（図では 5〜7 割）．
   - 同じ点：同じ数の少数ラベルを「切片だけ」と「全体」に使い分けて比べる．
   - 違う点：感情の課題が無い．ほかのデータセットを混ぜる情報源が無い．誤りの量は分けていない．
3. **Myung 2026（ワークショップ）**
   - 設定：GoEmotions から SemEval-2018 E-c への感情の転移（8 感情，多ラベル）．BERT．
   - 測った量：閾値を 0.5，共通の 1 つ，感情ごとに選ぶ場合の macro-F1．
   - 主張：感情ごとの閾値で外部のデータに強くなる．
   - 同じ点：感情の転移で，切片側だけを直したときの得を測った．
   - 違う点：残る誤り（向き側）を測っていない．閾値を選ぶデータの件数は少数ではない．情報源の比べは無い．
4. **Xian ほか 2022（ICLR）と Tran ほか 2023（Interspeech．第 1 回）**
   - 設定：Xian は言語をまたぐ感情極性，Tran は話者ごとの声の感情の回帰．
   - 測った量：ずれを「クラスの割合（ラベル）のずれ」と「特徴のずれ」に分けて別々に測り，成績との相関を出した．
   - 同じ点：切片側の原因と向き側の原因を分けて考える．
   - 違う点：分けたのはずれ（入力とラベルの分布の違い）で，分類器の誤りではない．情報源ごとの比べは無い．

---

## 5. 狙うべき差分（第 2 回で更新）

- **新しさとして書けないこと**：「凍結した特徴の線形の分類器で，ほかのデータセットから移したときの落ちを，閾値で戻る分と戻らない分に分ける」こと自体には，Haider の公開の解析という先例がある（2 値の感情極性．査読なし）．LLM の側にも，切片側の補正で多くが戻るという観察（Zhao 2021，Jiang 2023，Reif & Schwartz 2024）がある．論文でこの分け方を道具として新しいと書くのは避けるべきである．
- **新しさとして書けること**（今回の範囲で先例が無い）：
  - 同じ分け方で，4 つの情報源（少数ラベル，ソースを混ぜる，ラベル無しの中心合わせ，ほかのモダリティの予測）の得を並べること．
  - それを 4 クラスの感情で，言葉・声・顔のモダリティごとに行うこと．
  - 切片を選ぶデータ（ターゲットの K 件）と評価のデータを分けること．Haider と Myung はテストか開発用のデータの全体で閾値を選んでいる．
- **記録で既に出ている量との接点**（`DAI_paper/inventory.md`）：
  - 表の 1 行目「K=5 で，向きを固定して切片だけを K 本で直す手の上がり幅は，ソースと K 本を混ぜて学び直す手の 0.91 倍（音声），0.78 倍（顔）」（2 つの手の比は照合済み）．これは Reif & Schwartz の「切片だけの当て直し ÷ 全体の学習」と同じ型の比である（Reif & Schwartz は k=16 で約 0.49〜0.72）．ただし比べる相手が違う（Reif & Schwartz の LoRA はターゲットの 16 件だけで学び，MER-DA はソースと混ぜて学び直す）．この違いを書いたうえでなら，論文で並べて引ける．
  - 表の 7 行目「言葉は位置の誤りが小さく（1.0 対 方向 9.7）」（検査前）．Haider の線形の分類器でも，落ちの約 7 割が閾値では戻らなかった．どちらも言葉では向き側の誤りが大きいという同じ向きの観察である．ただし定義が違う（MER-DA は感情の対ごとの誤りの 6 対の平均，Haider は 2 値の正解率の落ち）ので，数値は比べられない．
  - 表の 19 行目は，ラベルの決め方の違い（演者の意図か，聞き手の知覚の多数決か）が位置と方向のどちらに出るかを「測っていない」と記録している．声の先行研究（Lefter 2022，Braunschweiler 2021 ほか）はラベルの定義の違いをコーパス間の失敗の主な原因に挙げるが，それが切片側と向き側のどちらに出るかを測った研究は今回も見つからなかった．この問いは先行研究に無い．
- **書き方の注意**：Reif & Schwartz を引くときに「切片の当て直しで差がほぼ埋まる」と要約しないほうがよい．図の値では 5〜7 割である．

---

## 6. 見つからなかったこと

- 4 つの情報源の得を，切片側と向き側に分けて感情の実データで並べた研究（言葉・声・顔のどれでも）．
- 査読つきの論文で，感情極性または感情のコーパス間の転移の落ちを，閾値で戻る分と戻らない分に分けた研究．見つかったのは査読の無い公開の解析（Haider 2026）だけである．
- 文章の分類で，分布のずれの下で vector scaling と matrix scaling を比べた研究．
- GoEmotions や SemEval E-c に Batch Calibration や文脈の較正を当てた研究．
- 声や顔の予測（疑似ラベル）で言葉の分類器を学んだ感情の研究．
- 声の感情で，切片・閾値・クラスの割合だけを少数のラベルまたはラベル無しで当て直し，表現の学び直しと比べた研究．
- コーパス間の声の感情の失敗の原因（ラベルの定義，話者，録音）が，切片側と向き側のどちらに出るかを測った研究．
- CAST の本文（arXiv 版も著者の公開の PDF も無かった）．

---

## 7. 検索語の記録

### 7.1 WebSearch（16 回）

| # | 検索語 | 見つかったもの | 無かったもの |
|---|---|---|---|
| 1 | AUROC stays high while accuracy drops domain shift text classifier threshold miscalibration sentiment | Haider の公開の解析（undertow），Opitz 2024，PredictaBoard | 査読つきの感情極性の閾値の分け方 |
| 2 | vector scaling versus matrix scaling text classification domain shift | 一般の較正の論文（1911.11195），Modular Domain Adaptation | 文章での 2 つの比べ |
| 3 | cross-lingual sentiment emotion transfer label prior shift calibration | Xian ほか 2022（IWDA），ICNLSP 2026，言語をまたぐ感情極性の監査（2602.17469） | 感情の多クラスでの割合のずれの分析 |
| 4 | batch calibration GoEmotions SemEval E-c emotion in-context learning | Li 2023（第 1 回で読んだ），E-ICL，2510.06600，Inoshita 2026 | 感情のデータでの BC |
| 5 | last-layer retraining bias only text classification | TLDR，AFR，DFR | 文章で最終層の切片だけを学び直す研究 |
| 6 | per-class threshold tuning cross-domain multi-label emotion SemEval E-c | Myung 2026（ToM4AI 論文集），SemEval-2025 の参加システム | 閾値で戻る分と残る分の比べ |
| 7 | text student pseudo labels from facial expression or speech emotion | LiSER，KBS 2021（顔→声），Amazon の言葉→声の蒸留，EASUM | 言葉が生徒になる感情の研究 |
| 8 | what causes cross-corpus speech emotion recognition failure analysis label definition speaker recording factors | Lefter 2022，Zhang・Mower Provost・Essl，Milner | 原因ごとに切片側と向き側を分けた研究 |
| 9 | speech emotion few labeled target samples bias-only or intercept-only classifier adaptation | Zhou & Chen 2019，Abdelwahab 2015・2017，Meta-PerSER，Dong 2026 | 声で切片だけを当て直す研究 |
| 10 | Unsupervised Cross-Domain Facial Expression Recognition via Class Adaptive Self-Training arXiv | DBLP の書誌，CWCST | CAST の本文 |
| 11 | Shanmin Wang Qingshan Liu class adaptive self-training facial expression cross-domain pdf | DBLP の書誌 | CAST の本文 |
| 12 | CAST class-adaptive self-training cross-domain FER github code implicit class margins class distribution pseudo labels predicted probabilities | 同名の別の研究（危機のメッセージの分類），CluP，CRST | CAST のコード |
| 13 | speech emotion recognition cross-corpus prior shift correction unlabeled target EM class priors versus fine-tuning representation comparison | CADA，Dong 2026，Tavernor 2025 | EM による割合の補正を声の感情に当てた研究 |
| 14 | speech emotion recognition decision threshold adjustment target corpus few labeled utterances recovers accuracy linear probe frozen self-supervised | SER Evals，Łachut 2025（PolEval） | 少数のラベルで閾値を当て直す声の研究 |
| 15 | "oracle threshold" cross-domain sentiment classification accuracy gap AUC decision boundary misplaced target domain | Haider（再），Raaijmakers & Kraaij 2010，ACAN | 査読つきの同じ分け方 |
| 16 | emotion recognition cross-dataset "logit adjustment" OR "prior correction" few labeled target examples compared with linear probe retraining frozen encoder | SER Evals，脳波の感情の事前学習（2510.22197），EMO-TTA | 割合の補正と線形の分類器の学び直しの比べ |

### 7.2 そのほかの照会（WebSearch 以外）

- arXiv の API：題名 "class adaptive self-training"，著者 "Shanmin Wang"，抄録の語の組（cross-domain facial expression，self-training，class distribution）．CAST は無かった．
- Semantic Scholar の API と Crossref：CAST の抄録と書誌を取得．公開の PDF は無い．
- doi.org → IEEE Xplore（document 10843182）：取得に失敗した．
- 直接の取得：lab-msp.com の Abdelwahab 2015 と Busso 2013（読めた），ojs.aaai.org の Raaijmakers & Kraaij 2010（ボット対策で拒否．第 1 回と同じ），GitHub の undertow の README（読めた），arXiv の Reif & Schwartz の PDF と図の画像（読めた）．

---

## 8. 読んだ原文（水準 ◎ の出どころ）

pdftotext か直接の取得で本文を読んだもの：Haider の README（4 節の表），Myung 2026（arXiv 2603.18786 の 109〜111 頁），Xian ほか 2022（ICLR の PDF．2 節，4.1 節，表 1），Reif & Schwartz 2024（arXiv の PDF の本文と付録の表．図 3a は画像で読み取った），Abdelwahab & Busso 2015（lab-msp.com の PDF），Busso ほか 2013（lab-msp.com の PDF），Łachut 2025（ACL Anthology の PDF の 1〜4 節）．WebFetch の要約で読んだもの（○）：Opitz 2024，Lefter 2022，arXiv 2510.06600．読めなかったもの：CAST，Raaijmakers & Kraaij 2010 の全文，Haider の公開ページの図（JavaScript）．
