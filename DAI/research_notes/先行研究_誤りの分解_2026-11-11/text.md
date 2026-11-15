**検証の結果（2026-11-12，`verify_emotion.md`）**：Reif & Schwartz 2024 の「同じ 16 件のラベル」は誤り．LoRA は 16 件で学んだうえで評価でも k 件のデモを見るので，k=16 の比べでは LoRA が 32 件を使う．ラベルの数を 16 件でそろえると（LoRA の k=0 と LOOC の k=16），LOOC は LoRA と同じか上回る（base 0.601 対 0.548，chat 0.632 対 0.638）．図 3a の実線＝base，破線＝chat は凡例に書いてある．Myung は印刷の頁で 110〜112．Haider の落ちは対角 4 セルの平均と非対角 12 セルの平均の差で，テストのクラスは 50/50．Raaijmakers & Kraaij 2010 は本文を読んで「関係」に下げた．

# 先行研究の調べ直し：誤りを切片側と向き側に分けた研究（言葉と感情極性．LLM の文脈内学習と較正を含む）（2026-11-11）

## 0. 要約（判定）

1. **完全一致は見つからなかった．**言葉の感情認識でも感情極性でも，「切片側の誤り」と「向き側の誤り」を同じ分け方で測り，4 つの情報源（ターゲットの少数ラベル，ほかのデータセット，ターゲットのラベル無しデータ，ほかのモダリティの予測）のどれがどちらを減らすかを比べた研究は無かった．検索は 44 回，原文を読んだ論文は 20 本である．
2. **部分一致は 13 本見つかった．**どれも情報源が 1 つか 2 つで，ほとんどが LLM の文脈内学習の較正の研究である．
3. 最も近いのは Reif & Schwartz 2024（NAACL）である．同じ 16 件のラベルを，切片の当て直し（LOOC）だけに使った場合と，モデル全体の学習（LoRA）に使った場合を比べ，切片の当て直しで性能の差が「ほぼ埋まる」と書いた．279 課題のうち感情極性は 39 課題で，感情の課題は無い．
4. 次に近いのは 3 本である．Estienne ほか 2023 は，ラベル無しで事前確率だけを当て直す場合と，ラベルで尺度と切片を当て直す場合を比べた（SST-2 など）．Jiang ほか 2023 は「順位づけの指標の AUROC は 0.95 を超えるのに F1 は 0.8 に届かない」ことを根拠に，ICL の誤りの主因をクラスの割合のずれとした（SST-2）．Zhao ほか 2021 は，閾値を最良にすれば SST-2 で 94% になると示した．
5. 感情のデータで切片だけの較正と向きを変える学習を並べた研究は 2 本ある．Cho ほか 2025 は tweet_eval_emotion，Li ほか 2023 は Emotion と GoEmotions を使った．どちらも手法の比較で，誤りを 2 つに分けてはいない．
6. 設定が最も近いのは Hosseini & Caragea 2023（Findings of ACL）である．GoEmotions から MELD の言葉への転移で，ソースのみ，ターゲットのみ（K=4/8/16），ラベル無しの使い方を比べた．ただし正解率しか比べていないので，判定は「該当しない」である．
7. 分け方の道具は Ferrer & Ramos 2025（TMLR）にある．誤りを「較正で取り戻せる分」と「取り戻せない分」に分け，SST-2 と IEMOCAP にも当てた．ただし情報源ごとには使っておらず，当て直しには尺度も含む．
8. ほかのモダリティの予測で言葉の分類器を学んだ研究は，今回も見つからなかった（手元の記録と同じ）．
9. 狙うべき差分は，同じ分け方で「ほかのデータセットを混ぜた得」と「少数ラベルの得」と「ラベル無しの中心合わせの得」を並べることである．言葉と感情の分野にこの比べは無い．

---

## 1. 用語と，この報告での読み替え

- **切片側の誤り（問いの (a)）**：分類器の重みは変えず，クラスごとの足し算の項（切片．バイアス項とも呼ぶ）や判定の閾値を当て直せば取り戻せる誤り．
- **向き側の誤り（問いの (b)）**：切片をどう当て直しても残る誤り．判別に使う向きや特徴の重みがずれていることから来る．
- **文脈内学習（ICL）**：LLM の重みを変えず，プロンプトに数件の例（デモ）を並べて分類させるやり方．
- **較正（calibration）**：モデルの出力の確率を，後から変換して直すこと．ICL の較正の研究では，クラスごとに足し算でずらす変換（切片の当て直しに当たる）と，クラスごとに掛け算で尺度を変える変換が使われる．
- **注意（読み替えの限界）**：LLM の較正の研究が「回転」と呼ぶもの（Batch Calibration の論文の 3.3 節）は，クラスの確率に対角の行列を掛けることを指す．特徴の空間で判別の向きを変えることとは別物である．そこで，この報告で問いの (b) に当たるとみなすのは，隠れ層の上で線形の分類器や重心を学び直す場合（Cho ほか 2025）と，モデル自体を学ぶ場合（LoRA など）だけにした．
- **水準**：◎ は原文の該当箇所を読んで確かめた．○ は抄録か信頼できる要約で確かめた．△ は検索の断片か書誌しか見ていない．

---

## 2. 研究ごとの表

### 2.1 部分一致（近い順）

| 研究 | 設定 | 測った量 | (a)(b) を分けたか | 水準 | URL |
|---|---|---|---|---|---|
| Reif & Schwartz 2024（NAACL）"Beyond Performance: Quantifying and Mitigating Label Bias in LLMs" | 10 個の LLM と 279 課題（Super-NaturalInstructions）．感情極性は 39 課題で，感情の分類の区分は無い．同じラベルの使い方を 2 通りで比べる．LOOC は k=8/16 件のデモの予測確率を正解ラベルで束ねて平均し，クラスごとの偏りを引く（切片の当て直し）．LoRA は 16 件で 5 エポック学習する（Llama-2 7B，Mistral 7B）．ラベルを使わない較正の CC と DC も比べた | 性能，RSD（クラスごとの正解率のばらつき），BiasScore（出力の確率の平均と一様分布の差） | **分けていない．**ただし，切片だけを当て直した場合と全体を学んだ場合を，同じラベルの数で比べている．本文には「LOOC nearly closes the gap with LoRA-level performance」とあり，LoRA の後でも RSD は高いまま残り，BiasScore はかえって上がった（"more subtle bias persists"）．CC は性能も偏りも改善しなかった．数値は図にしか無く，読んでいない | ◎（定義と本文の主張．数値は ×） | https://aclanthology.org/2024.naacl-long.378 ，https://arxiv.org/html/2405.02743v1 |
| Estienne, Ferrer, Vera, Piantanida 2023（RANLP 学生の部）"Unsupervised Calibration through Prior Adaptation …" | LLM の出力の事後確率を当て直す．UCPA はラベル無しのデータで事前確率だけを当て直す（切片だけ）．SUCPA はクラスの割合だけを知っている場合で，切片だけを学ぶロジスティック較正（尺度を 1 に固定）と同じ成績になることを確かめた．比べた相手は，ラベル 600 件で尺度と切片を学ぶ affine 較正と，内容の無い入力で較正する CC．データは SST-2，TREC，AGNews，DBPedia | 誤り率と交差エントロピー．ラベル無しのデータの数（10〜600）と，デモの数を振った | **一部だけ分けた．**切片だけの当て直し（ラベル無し，またはラベル有り）と，尺度も変える当て直し（ラベル有り）を同じ表に並べ，「the affine calibration system performs consistently better」と書いた．ただし尺度はクラスの確率への掛け算で，特徴の向きは変えない | ◎ | https://aclanthology.org/2023.ranlp-stud.2.pdf ，https://arxiv.org/abs/2307.06713 |
| Jiang ほか 2023（Findings of EMNLP）"Generative Calibration for In-context Learning" | 12 課題（SST2，SST5，CR，MR，SUBJ，AGNews，DBPedia，TREC，CB，RTE，QQP，SNLI）．12 個の LLM．感情の課題は無い | 3.2 節は SST2 と GPT2-XL の 4-shot で，プロンプトのクラスの比と順番を変えた．AUROC（閾値に依らない順位づけの良さ）は平均 0.95 を超え，ばらつきも小さかった．一方で F1 は 0.8 にほとんど届かなかった．モデルの出力のクラスの割合は，データのクラスの割合からずれていた | **概念として分けた．**誤りの主因を「ラベルの割合のずれ（切片側）」とし，「クラスの条件つきの分布はおおむね合っている（向き側）」とみなした．証拠は AUROC だけで，著者自身が限界の節で「高い AUROC は必要条件で十分条件ではない」と書いた．割合を直した後に残る誤りの割合は報告していない | ◎ | https://aclanthology.org/2023.findings-emnlp.152 ，https://arxiv.org/html/2310.10266v1 |
| Zhao, Wallace, Feng, Klein, Singh 2021（ICML）"Calibrate Before Use" | GPT-3 と GPT-2．感情極性は SST-2 だけ．内容の無い入力（"N/A" など）で偏りを推し，対角の変換で直す（CC） | SST-2 の図 5 では，最良の閾値（Positive の確率 0.68）にすれば検証用のデータで 94% になる．AGNews では，検証用のラベルで最良の対角の変換を選ぶオラクルの較正と比べ，CC は「surprisingly close」だった | **閾値だけで取り戻せる分を 1 例で示した．**誤りの内訳の量は出していない（"the end result of the above three biases is typically a simple shift in the model's output distribution"） | ◎ | https://arxiv.org/abs/2102.09690 |
| Zhou ほか 2024（ICLR）"Batch Calibration" | 13 の言語課題（感情極性は SST-2 だけ）と画像の課題．ラベル無しのバッチの出力の平均を引く（切片の当て直し）．BCL はラベル 128 件/クラスで，補正の強さ 1 つだけを選ぶ | 3.3 節で，各較正が判定の境界をどう動かすかを幾何で整理した．BC は境界を平行にずらし，CC は原点の周りに回す．BCL の得は平均 74.3 → 75.3 で，本文の語は「slight gains, 1% on average」 | **分けていない．**ラベル無しの切片の当て直しにラベルを足しても，得は約 1 点だった．教師ありの学習との比べは無い | ◎ | https://arxiv.org/html/2309.17249v2 |
| Cho ほか 2025（NAACL）"Token-based Decision Criteria Are Suboptimal in In-context Learning"（Hidden Calibration） | 10 データ．**感情の tweet_eval_emotion（TEE）を含む**．6 個の LM．どの手法も 16 件/クラスのラベルを使う．Hidden Calibration は，最後の隠れ層の上でクラスごとの重心を学び，最も近い重心のクラスに決める（向きを学び直す） | macro-F1．GPT2-XL の TEE では，デモ 0/1/4/8 件で BC が 26.39/25.85/28.86/29.23，Hidden Calibration が 43.45/36.17/39.13/35.48（デモの数と列の対応は取り出したときに推定したもので，○） | **分けていない．**3.1 節は，語の確率を較正する方法は「平行移動と限られた回転」しかできないと論じる．付録 A.2 は「向きを適切に回し，原点を移す較正なら近づける」と推測するが，確かめていない．8 節では，BC と組み合わせて試していないと書いた | ◎（主張と設定）．TEE の数値は ○ | https://arxiv.org/html/2406.16535v3 ，https://aclanthology.org/2025.naacl-long.278 |
| Li, Yan, Li, Qian, He, Gui 2023（arXiv 2302.06198．掲載先は △）"Distinguishability Calibration to In-Context Learning" | プロンプト学習の出力の埋め込みを回転と尺度で変換する（TARA）．比べた相手は CC で，ラベルで学ぶ対角の変換．データは **Emotion（Saravia 2018）と GoEmotions**，WOS．K=5/10/50/100 | 重みつき F1．Emotion で CC が 0.337/0.352/0.531/0.706，TARA が 0.348/0.401/0.697/0.783．本文は「improvements are more predominant when K is larger」 | **分けていない．**対角の変換と回転を含む変換を，K を振って並べただけである．ここでの CC はラベルで学ぶ尺度の変換で，切片だけの当て直しではない | ◎ | https://arxiv.org/abs/2302.06198 |
| Olah, Baruah, Bose, Narayanan 2021（arXiv 2110.05021．ICASSP 2022 と思われる）"Cross Domain Emotion Recognition using Few Shot Knowledge Transfer" | GoEmotions（Reddit）で学んだ SBERT の埋め込みと，感情の語の埋め込みのコサイン類似度で判定する．**ターゲット（SemEval のツイート）の開発用データは，感情ごとの閾値を決めるのにだけ使う** | SemEval の F1．最大は 0.49（WordNet の定義＋GoEmotions で学んだ SBERT）で，フルの教師ありの NTUA-SLP は 0.53．図 4 では，閾値を決める開発用データが約 600 件を超えると F1 が伸びなくなった | **分けていない．**少数のラベルを閾値だけに使う方法（切片側だけを直す方法）の先例である | ◎ | https://arxiv.org/abs/2110.05021 |
| Fei, Hou, Chen, Bosselut 2023（ACL）"Mitigating Label Biases for In-context Learning" | 24 の文章分類データ（SST-2，CR，SST5，Tweet hate，Poem sentiment など）．本文を検索した範囲では感情のデータは無い．ICL のラベルの偏りを，語の偏り，文脈の偏り，領域の偏りの 3 つに分けた | 領域の偏りの大きさ＝「でたらめな英単語」と「そのデータの単語」を入れたときの出力の確率の差の半分（式 1）．DC は，そのデータの単語をでたらめに並べた入力で偏りを推して引く | **偏りの出どころを分けた**（切片側の中身を 3 つに分けた）．向き側との分け方は無い．ラベルを使う学習との比べも無い | ◎ | https://aclanthology.org/2023.acl-long.783 |
| Inoshita, Zhou, Kawai, Yada 2026（arXiv 2604.27345）"LLMs Capture Emotion Labels, Not Emotion Uncertainty" | ゼロショットの LLM 4 つで，GoEmotions と EmoBank の人の分布と比べる．後から当てる補正を 3 つ比べた（温度の調整，感情ごとのずらし＝切片，等張回帰）．補正には 2,000 件の 5 分割のラベルを使う | Jensen–Shannon の距離．等張回帰が最もよく縮め，感情ごとのずらしは偏りの強いモデルで効いた | **分けていない．**切片だけの補正と単調な変換を並べただけで，正解率は主な指標ではない | ◎ | https://arxiv.org/html/2604.27345 |
| Chen, Zheng, Awadallah, Ji 2022（ACL Insights）"Pathologies of Pre-trained Language Models in Few-shot Fine-tuning" | BERT と RoBERTa．感情極性（IMDB → Yelp）と，NLI，言い換えの判定．学習データの割合を振った | 予測の偏り PB（予測のクラスの割合とデータのクラスの割合のずれ）と正解率．学習の例が増えると PB は下がる．得の一部は，課題と関係の無い語の特徴から来る | **分けていない．**ラベルが偏りを減らすことは示したが，得のうち偏りが減った分を切り出してはいない | ○（定義と表の位置は本文で見た．数値は読んでいない） | https://aclanthology.org/2022.insights-1.20 |
| Raaijmakers & Kraaij 2010（ICWSM）"Classifier Calibration for Multi-Domain Sentiment Classification" | 既存の単一領域の感情極性の分類器を，新しい領域の少数のラベルで閾値だけ当て直し，投票で合わせる | 抄録の範囲では，閾値を推す基準 3 つの成績と，注釈の手間の兼ね合い | **分けていない**（閾値側だけを直す方法）．全文は取得できなかった（ボット対策で拒否された） | ○ | https://ojs.aaai.org/index.php/ICWSM/article/view/14055 |
| Chochlakis ほか 2024（arXiv 2403.17125．掲載先は △）"The Strong Pull of Prior Knowledge in LLMs and Its Impact on Emotion Recognition" | 感情の ICL（SemEval 2018 E-c，GoEmotions）．事前の知識はゼロショットとでたらめなラベルのデモの予測で表し，最終の予測が正解と事前の知識のどちらに近いかを測る | Jaccard，micro/macro F1 の差（引っ張りの強さ）．大きいモデルほど事前の知識に引っ張られる | **分けていない．**クラスの割合の偏りは切り出していないと本文で確かめた．較正もしていない | ◎ | https://arxiv.org/html/2403.17125v1 |

### 2.2 関係する理論・分け方の枠組み

| 研究 | 内容 | 本件との関係 | 水準 | URL |
|---|---|---|---|---|
| Ferrer & Ramos 2025（TMLR）"Evaluating Posterior Probabilities: Decision Theory, Proper Scoring Rules, and Calibration" | 適正なスコアの期待値を「較正の項」と「判別の項」に分ける．較正の損失は，後から当てる変換（尺度 α と切片 β．Brümmer & van Leeuwen 2006 の向きを保つ変換）の前と後のスコアの差である．表 2 では，SST2 のゼロショットの GPT-2 で正規化交差エントロピーの 46.0%，4-shot で 62.4% が較正の損失だった．IEMOCAP の 4 感情（wav2vec 2.0）では 3.1% だった | 問いの (a)(b) に最も近い分け方の道具である．ただし当て直しに尺度を含み，切片だけの変種は評価していない．情報源ごとには使っていない．2.4 節は「calibration performance appears to be more sensitive to domain mismatch than discrimination performance」と書き，Ferrer ほか 2021（話者認証）を引く | ◎ | https://arxiv.org/abs/2408.02841 |
| Batch Calibration の 3.3 節（Zhou ほか 2024） | 各較正を，判定の境界の平行移動（BC と DC）と原点の周りの回転（CC）に整理した | 平行移動は問いの (a) に当たる．回転はクラスの確率の空間での回転で，特徴の向きとは別物である | ◎ | https://arxiv.org/html/2309.17249v2 |
| Jiang ほか 2023 の 3 節 | ICL を暗黙のベイズ推論とみなし，クラスの条件つきの分布は合うがクラスの割合がずれる，と論じた | 「LLM の誤りは主に切片側にある」という予言である．仮定は理想化した極限（K → ∞）でしか成り立たない | ◎ | https://arxiv.org/html/2310.10266v1 |
| He ほか 2024/2025（Stat 14(1)．arXiv 2410.09411） | ガウス混合を広げたモデルで，2 値の ICL の境界を閉じた形で出し，事前の知識とデモの寄与，少ないクラスの正解率が低いことを示した | デモのクラスの割合が境界をずらすことの理論である．切片と向きに分けた形は抄録では確かめられなかった | ○ | https://arxiv.org/abs/2410.09411 |
| Saerens 2002，Lipton 2018，Alexandari 2020 | ラベルシフト（クラスの割合だけが変わり，クラスの条件つきの分布は変わらないずれ）の補正 | 手元の記録にある（`docs/related-work.md` 59 行）．言葉と感情に当てた実験は今回も見つからなかった．Esuli, Molinari, Sebastiani 2022（ACM TOIS）は，文章の分類で EM による割合の補正を見直し，クラスが少ないときにだけ効くと報告した（検索の断片で見ただけ） | 手元の記録．Esuli は △ | https://iris.cnr.it/retrieve/7ff467d4-8025-49c0-8ca8-2821579fdf71/Esuli_Molinari_Sebastiani_ACM%20TOIS-2022-postprint.pdf |
| DISDE（Cai, Namkoong, Yadlowsky 2023．arXiv 2303.02011） | 性能の落ちを，X の分布の変化と Y|X の変化に分ける | 分け方の軸が違う（切片と向きではない）．言葉の感情に当てた例は見つからなかった | △ | https://arxiv.org/abs/2303.02011 |

### 2.3 該当しない（設定は近いが，正解率などだけを比べた）

| 研究 | 設定 | 該当しない理由 | 水準 | URL |
|---|---|---|---|---|
| Hosseini & Caragea 2023（Findings of ACL）"Semi-Supervised Domain Adaptation for Emotion-Related Tasks" | **GoEmotions（Reddit）→ MELD（ドラマの言葉）の 7 感情**．BERT-base，K=4/8/16/クラス．表 1 の正解率：ソースのみ 24.51，ターゲットのみ 16.69/18.09/19.96，ソースの予測を疑似ラベルにする UDA 21.15，ターゲットの中の自己学習 18.90/21.08/21.75，提案 25.83/27.54/28.85，ターゲットのフルの教師あり 63.02 | 情報源の 3 つ（少数ラベル，ほかのデータセット，ラベル無し）がそろう最も近い設定である．しかし正解率だけで，誤りの内訳は無い | ◎ | https://aclanthology.org/2023.findings-acl.333 |
| Demszky ほか 2020（ACL）GoEmotions の転移の実験 | BASELINE（ターゲットだけで学ぶ），FREEZE（GoEmotions で学んだ BERT の最後の層だけ学び直す），NOFREEZE（全体を学び直す）．ターゲットの件数は 100/200/500/1000/全体の 80%．9 データ | F1 だけである．閾値やクラスの割合の分析は無いと本文で確かめた | ◎ | https://arxiv.org/abs/2005.00547 |
| Bostan & Klinger 2018（COLING） | 感情の 14 コーパスを 1 つの形式にそろえ，コーパス間で予測した | 正解率と F1 だけである（抄録の範囲） | ○ | https://aclanthology.org/C18-1179/ |
| Min ほか 2022（EMNLP）"Rethinking the Role of Demonstrations" | ICL のデモの効き目を，ラベルの空間，入力の分布，形式，入力とラベルの対応に分けた．正解のラベルをでたらめにしても 0〜5 点しか落ちない．感情のデータは無い | 分け方の軸が違う | ◎ | https://arxiv.org/abs/2202.12837 |
| Pan ほか 2023（Findings of ACL）"What In-Context Learning 'Learns' In-Context" | ICL を課題の認識と課題の学習に分けた | 分け方の軸が違う | ○ | https://arxiv.org/abs/2305.09731 |
| Han ほか 2023（ICLR）Prototypical Calibration | ガウス混合で非線形の境界を作る較正 | 抄録の範囲では感情のデータは無い．分けていない | ○ | https://arxiv.org/abs/2205.10183 |
| Abbas ほか 2024（AISTATS）LinC | 約 5 件のラベルで出力の確率の線形の変換を学ぶ | 分けていない | ○ | https://proceedings.mlr.press/v238/abbas24a.html |
| Holtzman ほか 2021 "Surface Form Competition"（PMI_DC） | 答えごとの事前の起こりやすさで割る（切片の当て直しに当たる） | 分けていない．掲載先は arXiv の頁に書かれていない（EMNLP 2021 は △） | ○ | https://arxiv.org/abs/2104.08315 |
| Mosbach ほか 2023（Findings of ACL）少数ショットの微調整と ICL の比較 | NLI と言い換えの判定だけ | 感情極性も感情も無く，切片の分析も無い | ◎（本文を検索した範囲） | https://aclanthology.org/2023.findings-acl.779 |
| Ben Zaken ほか 2022（BitFit），Logan ほか 2022 | 全ての層のバイアス項と頭だけを学ぶ | 中の層のバイアスは表現を変えるので，最終層の切片だけの当て直しとは別物である | ○ | https://arxiv.org/abs/2106.10199 |
| Chen ほか 2020（arXiv 2003.07444）DAN-LPE | 文章の分類の DANN に，ターゲットのクラスの割合の推定を足した | 割合の補正の効果を，表現のずれと分けて測ったかは抄録では分からない | ○ | https://arxiv.org/abs/2003.07444 |
| Moreo & Sebastiani 2022（PLoS ONE）ツイートの感情極性の割合の推定 | クラスの割合を推す方法を 11 データで比べ直した | 分類器の誤りではなく割合の推定の誤りを測る | ○ | https://arxiv.org/abs/2011.08091 |
| Tigges ほか 2024 "Linear Representations of Sentiment in LLMs" | 感情極性は 1 つの向きで表され，作ったデータから SST へ移る | 向きがデータ間で共有されることを示す．誤りの分け方は無い | ○ | https://arxiv.org/abs/2310.15154 |
| Tran ほか 2026（arXiv 2607.05937） | 凍結した LM の特徴で DANN・MMD・対照学習を使う．2 値の感情極性 | 手元の記録にある（`docs/related-work.md` 847〜851 行）．分けていない | 手元の記録 ◎ | https://arxiv.org/abs/2607.05937 |
| "Towards Distribution-shift Robust Text Classification of Emotional Content"（Findings of ACL 2023） | NLI に基づくゼロショットのモデルが，分布がずれると教師ありより良いことが多い | 正解率だけと思われる（検索の断片だけ） | △ | https://aclanthology.org/2023.findings-acl.524 |

---

## 3. 最も近い先行研究 5 本と，MER-DA との同じ点・違う点

研究目標が達成されたとき（4 つの情報源の得を切片側と向き側に分けて並べたとき），最も近いのは次の 5 本である．

1. **Reif & Schwartz 2024**
   - 設定：LLM の ICL．279 課題．16 件のラベル．
   - 測った量：性能と，クラスごとの性能のばらつき．
   - 主張：16 件のラベルを切片の推定だけに使っても，全体を学んだ場合との差がほぼ埋まる．
   - 同じ点：同じ数のラベルを「切片だけ」と「全体」の 2 通りに使って比べている．MER-DA の「少数の例は主に答えの偏りを直す」と同じ向きの観察である．
   - 違う点：感情の課題が無い．ほかのデータセットを混ぜる情報源が無い．ラベル無しの切片の当て直し（CC，DC）は比べたが，そこで減った誤りと 16 件のラベルで減った誤りを同じ物差しで分けてはいない．凍結した特徴の線形の分類器ではなく LLM の出力である．
2. **Estienne ほか 2023**
   - 設定：LLM の出力の事後確率．SST-2 など 4 データ．
   - 測った量：誤り率と交差エントロピー．
   - 主張：ラベル無しで割合だけを当て直しても大きく良くなる．ラベルで尺度も当て直すとさらに良くなる．
   - 同じ点：ラベル無しの切片の当て直しと，ラベル有りの当て直しを並べている（情報源の 2 つを並べている）．
   - 違う点：「尺度」はクラスの確率への掛け算で，特徴の向きは変えない．感情とほかのデータセットが無い．
3. **Jiang ほか 2023（と Zhao ほか 2021）**
   - 設定：ICL．感情極性は SST-2 と SST-5．
   - 測った量：AUROC と F1 の差（Zhao は最良の閾値での正解率）．
   - 主張：ICL の誤りの大部分はクラスの割合のずれで，順位づけはおおむね合っている．
   - 同じ点：閾値に依らない量で「向き側は良く，切片側が悪い」と診断する考え方である．
   - 違う点：情報源ごとに得を分けていない．診断はモデル 1 つとデータ 1 つで，向き側が良いことの証拠は AUROC だけである．
4. **Cho ほか 2025**
   - 設定：ICL．tweet_eval_emotion を含む 10 データ．16 件/クラス．
   - 測った量：macro-F1．
   - 主張：語の確率の較正（平行移動と限られた回転）では足りず，隠れ層の上で向きを学ぶと 20〜50% 良くなる．
   - 同じ点：感情の言葉のデータで，切片側だけの直しと向き側も含む直しを並べている．
   - 違う点：手法の比較で，誤りを分けていない．付録で「回転と原点の移動を合わせれば近づくかもしれない」と推測しただけで，確かめていない．
5. **Hosseini & Caragea 2023**
   - 設定：GoEmotions から MELD の言葉への転移．K=4/8/16/クラス．BERT．
   - 測った量：正解率．
   - 主張：分けて学ぶ共学習が良い．
   - 同じ点：MER-DA と同じ型の比べ（ソースのみ，ターゲットのみ，混ぜる，ラベル無しを使う）を，MELD という同じコーパスで行った．
   - 違う点：誤りを分けていない．エンコーダを微調整している．

分け方の道具としては Ferrer & Ramos 2025 が最も近い．ただし情報源の比べには使われていない．

---

## 4. 狙うべき差分

- **先行研究が測っていない量**：同じ分け方（切片だけを当て直せば取り戻せる分と，残る分）で，「ほかのデータセットを混ぜた得」「ターゲットの少数ラベルの得」「ラベル無しの中心合わせの得」を同じ表に並べること．言葉と感情の分野では，2 つの情報源を並べた例（Reif & Schwartz は少数ラベルの 2 通りの使い方，Estienne はラベル無しとラベル有り）はあるが，ほかのデータセットを混ぜる情報源を同じ分け方に入れた例は無い．
- **手元の記録で既に測れているもの**：`docs/related-work.md` 59 行に，音声の 1 つの設定の K=5 の数値がある．向きを変えずに感情ごとのずらしだけを少数の例で学ぶ操作で +4.9 点，合併の上乗せで +4.7 点，ラベル無しの中心合わせの後はどちらも 1 点未満である．この比べは，上の Reif & Schwartz や Estienne の比べを，ほかのデータセットを混ぜる情報源まで広げたものに当たる．ただし言葉の特徴で同じ比べをした記録があるかは，この調査では確かめていない（`DAI_paper/inventory.md` は読んでいない）．
- **書き方の注意**：言葉の分野の先行研究に対しては，「切片側の直しで大部分が取り戻せる」という観察自体は新しくないと書く必要がある（Zhao 2021，Jiang 2023，Reif & Schwartz 2024，Ferrer & Ramos 2025 の SST2 で 46〜62% が較正の損失）．新しいのは，情報源ごとに分けて並べること，ほかのデータセットを混ぜる得を分けること，感情の 4 クラスでそれを行うことである．

---

## 5. 見つからなかったこと

- 言葉の感情認識で，ほかのデータセットのラベル付きデータを混ぜた得を，切片側と向き側に分けた研究は，今回の検索では見つからなかった．
- 言葉の感情認識で，ターゲットのラベル無しデータの使い方（平均を引く，テスト時適応，ラベルシフトの補正）の得を，少数ラベルの得と同じ物差しで比べた研究は見つからなかった．テスト時適応は，言葉だけの感情の研究そのものが見つからず，音声（Dong 2026）とマルチモーダルの感情極性（ACL 2026 Findings）しか出てこなかった．
- 言葉の分類器を，声や顔の予測（疑似ラベル）で学んだ研究は見つからなかった（手元の記録 `research_notes/先行研究_モダリティ別_2026-11-11/report.md` 436 行と同じ）．Whisper の音声の埋め込みを言葉のモデルに写す研究（arXiv 2311.07014）はあるが，感情の分類ではない（△）．
- ラベルシフト（クラスの割合だけのずれ）と条件つきのずれを，感情の文章のコーパス間で分けて測った研究は見つからなかった．
- 感情の文章の分類で，最終層の切片だけを少数のラベルで学び直す場合と，全体を学び直す場合を比べた研究は見つからなかった．いちばん近いのは，閾値だけを当てる Olah 2021 と，最後の層だけを学び直す GoEmotions の FREEZE である．FREEZE は重みも学び直すので，切片だけの当て直しではない．
- LLM の較正の研究で，感情の多クラスのデータに CC・DC・BC を当てた結果は，Cho ほか 2025（TEE）と Li ほか 2023（Emotion，GoEmotions．ここでの CC は学習つき）にしか無かった．
- Ferrer & Ramos の「較正の損失」を，感情の文章，コーパス間の転移，情報源の比べに当てた研究は見つからなかった．

---

## 6. 検索語の記録

使った検索語は 44 個である．45 個目で，この会話の全ての担当で分け合う検索の上限（200 回）に達し，検索できなくなった．そのため，6.2 の検索語は使っていない．

### 6.1 使った検索語

| # | 検索語 | 見つかったもの | 無かったもの |
|---|---|---|---|
| 1 | contextual calibration few-shot in-context learning majority label bias recency bias | Zhao 2021，Surprise Calibration 2025，BC | 感情のデータ |
| 2 | mitigating label biases for in-context learning domain-context calibration | Fei 2023 | ― |
| 3 | batch calibration rethinking calibration in-context learning prompt engineering linear decision boundary | BC，Cho 2025 への言及 | 教師ありとの比べ |
| 4 | quantifying and mitigating label bias in LLMs calibration few-shot fine-tuning | Reif & Schwartz 2024，NCC（IJCNLP 2025） | 感情の区分 |
| 5 | token-based decision criteria are suboptimal in in-context learning hidden calibration centroid | Cho 2025 | ― |
| 6 | generative calibration in-context learning label shift p(y) changes p(x\|y) | Jiang 2023 | 後続の分析 |
| 7 | prototypical calibration few-shot learning language models gaussian mixture decision boundary | Han 2023 | ― |
| 8 | enhancing in-context learning via linear probe calibration | LinC（Abbas 2024） | ― |
| 9 | unsupervised calibration through prior adaptation text classification large language models | Estienne 2023 | ― |
| 10 | label shift estimation text classification sentiment EM Saerens prior adjustment cross-domain | Alexandari，DAN-LPE，LEIP | 感情極性での EM の比べ |
| 11 | cross-corpus emotion classification text analysis of annotated corpora label distribution transfer | 音声のコーパス間の研究だけ | 文章の研究 |
| 12 | GoEmotions transfer learning experiments other emotion datasets limited training data freeze | Demszky 2020，arXiv 2403.06108 | 閾値の分析 |
| 13 | Bostan Klinger analysis of annotated corpora for emotion classification in text cross-corpus experiments | Bostan & Klinger 2018，Topic Bias 2023 | ― |
| 14 | cross-domain emotion detection text domain adaptation tweets news headlines BERT few target labels | Olah 2021，Hosseini & Caragea 2023，DeepEmotex，Peng 2023（KBS） | 誤りの分け方 |
| 15 | in-context learning emotion recognition LLM label bias towards neutral calibration | Chochlakis 2024，Inoshita 2026，ChatGPT の偏り 2023 | 感情の ICL での CC |
| 16 | logit adjustment class prior emotion classification text imbalanced cross-domain | 最適輸送の重みづけ（SST），SER Evals（音声） | 文章の感情での割合の補正 |
| 17 | semi-supervised domain adaptation emotion text few target labels two classifiers teach each other Findings ACL 2023 | Hosseini & Caragea 2023 | ― |
| 18 | test-time adaptation text classification sentiment distribution shift entropy minimization language model | 言葉のテスト時適応（MEMO 型，STAF，LLM による書き換え），マルチモーダルの感情極性のテスト時適応 | 誤りの分け方 |
| 19 | domain adaptation sentiment classification label shift class imbalance target label distribution differs amazon reviews | DAN-LPE，偏りのあるターゲットでの頑健さの試し | 割合のずれの寄与を切り出した分析 |
| 20 | what in-context learning learns in-context disentangling task recognition and task learning | Pan 2023 | ― |
| 21 | BitFit bias-only fine-tuning sentiment few-shot only bias terms classifier head analysis | BitFit，Logan 2022 | 頭の切片だけの分析 |
| 22 | in-context learning under label shift class prior demonstrations theory Bayes | Surprise Calibration，Bayesian scaling laws，偏ったデモの研究 2025 | 定理 |
| 23 | linear representation of sentiment direction generalizes across datasets probe offset | Tigges 2024，感情の部分空間の論文（2025〜2026） | 切片の項 |
| 24 | audio teacher text student emotion recognition cross-modal distillation pseudo labels text model | LiSER，Whisper → LM（2311.07014），TelME，ACL-MER，PL-Distill | 言葉が蒸留先の感情の研究 |
| 25 | label proportions estimation adversarial domain adaptation text classification BBSE results | DAN-LPE，DATS | ― |
| 26 | cross-domain sentiment classification decision threshold recalibration target domain class prior bias term adaptation | Raaijmakers & Kraaij 2010，Estienne 2023 | 切片の項の分析 |
| 27 | fine-tuning only the classifier bias few target examples text classification domain shift compared with full retraining | 画像の研究だけ | 文章での切片だけと全体の比べ |
| 28 | "label shift" "conditional shift" decomposition performance drop NLP benchmark sentiment analysis | DISDE，DynaSent の一文，NER のずれの研究 | 感情極性での分け方 |
| 29 | emotion recognition text domain shift analysis what transfers class distribution vs feature shift tweets reddit dialogue | 文章の感情の分布のずれの論文（Findings ACL 2023），会話の感情の DANN | 割合と特徴の分け方 |
| 30 | calibration in-context learning emotion classification dataset contextual calibration results emotion tweets GoEmotions LLM prompt | Li 2023（Distinguishability），EICL | 感情のツイートでの CC の数値 |
| 31 | few-shot emotion classification text linear probe frozen sentence embeddings cross-dataset source data mixing | SentiCSE，SemEval 2025 の凍結の探り | ソースを混ぜる比べ |
| 32 | LLM emotion recognition conversation in-context learning demonstration label distribution bias neutral over-prediction MELD IEMOCAP | 段階的な推論で中立を避ける偏り（2026 の前刷り．数値は確かめていない），DialogueLLM | デモのクラスの割合を切り出した研究 |
| 33 | sentiment quantification prior probability shift classifier adjustment tweets experimental re-evaluation | Moreo & Sebastiani 2022，Keith & O'Connor 2018 | ― |
| 34 | "few-shot" "fine-tuning" versus "in-context learning" fair comparison out-of-domain generalization label bias calibration analysis | Mosbach 2023，Pathologies 2022 | ― |
| 35 | multimodal sentiment analysis unimodal label generation text modality bias offset analysis Self-MM | Self-MM，Self-MI | 言葉のずらしの分析 |
| 36 | intercept-only adaptation logistic regression prior shift text classifier new domain few labeled examples per class empirical study | Estienne，Esuli 2022（TOIS），Plugin（2503.02119） | 切片だけの当て直しの実証研究 |
| 37 | Pathologies of Pre-trained Language Models in Few-shot Fine-tuning prediction bias across labels | Chen 2022 | ― |
| 38 | test-time adaptation emotion classification text language model unlabeled target | 音声のテスト時適応（Dong 2026，Emo-TTA），マルチモーダルの感情極性 | 言葉だけの感情 |
| 39 | theoretical analysis why calibration improves in-context learning label bias shift decision boundary theorem | BC，Jiang 2023，He 2024 | 改善の定理 |
| 40 | unsupervised domain adaptation emotion classification text corpora pivot features adversarial label distribution | SemEval 2025 の長さの敵対学習，感情極性の DA の総説 | 感情の割合のずれ |
| 41 | zero-shot emotion classification natural language inference prompts label bias calibration threshold | 閾値を調整する多ラベルの論文 2025，NLI のゼロショット（Findings ACL 2023），Inoshita 2026 | NLI の仮説文の較正 |
| 42 | calibration loss discrimination loss decomposition affine calibration expected cost proper scoring rules Ferrer text classification | Ferrer & Ramos 2025 | ― |
| 43 | refinement loss calibration loss language model classification domain shift which part increases | ずれの下での較正の論文（ECE） | 2 つの項に分けた分析 |
| 44 | "calibration loss" emotion recognition cross-corpus discrimination minimum cross-entropy analysis | 感情の較正の手法（ECE） | コーパス間で 2 つの項に分けた研究 |

### 6.2 上限で使えなかった検索語（次に調べるなら）

- AUROC stays high while accuracy drops domain shift text classifier threshold miscalibration sentiment
- vector scaling versus matrix scaling text classification domain shift
- cross-lingual sentiment emotion transfer label prior shift calibration
- batch calibration GoEmotions / SemEval E-c emotion in-context learning
- last-layer retraining bias only text classification spurious correlation（一般の機械学習の担当と重なる）
- per-class threshold tuning cross-domain multi-label emotion SemEval E-c
- text student pseudo labels from facial expression or speech emotion

---

## 7. 読んだ原文（水準 ◎ の出どころ）

本文を取り出して読んだもの：Zhao 2021（ar5iv），Fei 2023（ACL の PDF），Zhou 2024（arXiv の HTML），Reif & Schwartz 2024（arXiv の HTML），Cho 2025（arXiv の HTML．付録 B の表まで），Jiang 2023（arXiv の HTML），Estienne 2023（arXiv の PDF），Olah 2021（arXiv の PDF），Hosseini & Caragea 2023（ACL の PDF），Demszky 2020（ar5iv），Li 2023（arXiv の PDF），Inoshita 2026（arXiv の HTML），Chochlakis 2024（arXiv の HTML），Min 2022（ar5iv），Mosbach 2023（ACL の PDF．語を検索しただけ），Chen 2022（arXiv の PDF．定義と表の位置），Ferrer & Ramos 2025（arXiv の HTML．表 2 と付録 D）．抄録だけを読んだもの：He 2024，Han 2023，Holtzman 2021，DAN-LPE，Raaijmakers & Kraaij 2010（全文はボット対策で拒否された）．
