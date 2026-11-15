# 先行研究の調査 A：Albanie ほか 2018 を直接引用した後続研究（感情認識の範囲）（2026-11-04）

## 要約（10 行）

1. **判定**：Albanie の枠組みを延ばした後続研究はある．枠組みとは，あるモダリティのモデルがラベル無しの発話に付けた予測を正解にして，別のモダリティのモデルを学ぶことである．ただし研究者の拡張の 4 軸を同時に扱った研究は，引用データベースで洗った約 325 本の中に見つからなかった．4 軸は，クラスあたり K 本のラベル，3 モダリティ，各単一モダリティと融合の両方を良くすること，ソースとターゲットが別コーパスであること，である．
2. **最も近い後続研究は 4 本である**．
   - Zhang ほか 2021（KBS）◎：顔の予測と声自身の予測を混ぜて IEMOCAP の声を学ぶ．EmoVoxCeleb をラベル無しのデータに使う．
   - Tran・Yin・Soleymani 2025（SetPeER，TAC）◎：言葉＋顔で声に，言葉＋声で顔にラベルを付け，別コーパスで評価する．
   - Zhuang ほか 2025（ACL-MER，IJCNLP-AACL）◎：ラベル 100〜1000 本．凍結した HuBERT・RoBERTa・CLIP の特徴の上で融合モデルを学ぶ．
   - He ほか 2020（Image2Audio，CVPRW）◎：別コーパスの顔画像で声を補う半教師ありの DA．
3. **Albanie の手をそのまま別コーパスに当てると弱いことは既に報告されている**．Zhang 2021 では，EmoVoxCeleb で学んだ声のモデルを IEMOCAP に直接当てると，4 クラスの UA は 27.99%（でたらめで 25%）だった．IEMOCAP の全ラベルで微調整しても 54.97% で，ラベルだけで学んだ 59.43% を下回った．
4. **「別モダリティの疑似ラベルの得はラベルが増えると減り，負にもなる」も報告済みである**．ACL-MER の MER2023 では，ラベル 100 本のとき正解率が 0.432 から 0.674 に上がった．1000 本のときは 0.826 から 0.819 に下がった．MELD では全てのラベル数で下がった．
5. **「2 つのモダリティを合わせたラベル付けモデルは，1 つのモダリティより正確」も報告済みである**（SetPeER 表 V，ACL-MER 表 4，Zhang 2021）．
6. **今回の検索で見つからなかったこと（先行研究がやっていないこと）は 4 つある**．
   - (i) 別モダリティの疑似ラベルがどの感情の対の誤りを直すかを，対ごとに分けて測った研究．
   - (ii) ラベル付きのソースのコーパスとクラスあたり K 本のターゲットのラベルを両方持つ設定で，Albanie 型の疑似ラベルを使った研究．
   - (iii) 凍結した特徴の上の線形の分類器を，音声・顔・言葉の各単独と融合の 4 つ全部で学び直して比べた研究．
   - (iv) 話者を分けたうえで，複数のターゲットのコーパスを順に回した研究．
7. **Albanie を引用した論文の約 4 分の 1（82 本）は顔の表情認識の論文である**．これらは Albanie のラベル付けモデル（顔のモデル）の FERPlus での正解率 88.8% を比較表に載せるためだけに引用している．枠組みを延ばした論文は 10 本前後である．

---

## 0. この報告で使う語

- **ラベル付けモデル**：ラベル無しの発話に感情の確率（疑似ラベル）を付けるモデル．Albanie では顔の表情のモデルである．原論文では teacher と呼ぶ．
- **学習側のモデル**：疑似ラベルを正解として学ぶモデル．Albanie では声のモデルである．原論文では student と呼ぶ．
- **蒸留**：ラベル付けモデルが出した確率を正解にして，学習側のモデルを交差エントロピーなどの損失で学ぶこと．
- **単一モダリティ**：音声だけ，顔だけ，言葉だけを入力にすること．
- **融合**：複数のモダリティの特徴をつないで 1 つのモデルに入れること．
- **コーパス間の評価**：学習に使ったコーパス（例：VoxCeleb）とは別のコーパス（例：IEMOCAP）で評価すること．
- **K 本**：ターゲットのコーパスで感情ごとに K 本だけラベルがある設定（MER-DA の語）．
- **UA（UAR）**：感情ごとの正解率を平均した値．MER-DA の主指標と同じである．
- **EmoVoxCeleb**：Albanie が公開したデータ．VoxCeleb の各発話に，顔のモデルの予測が付いている．人が付けたラベルではない．
- **検証の水準**：◎ 原文の該当箇所を読んだ．○ 抄録か信頼できる二次資料で確かめた．△ 検索の断片だけで判断した．

## 1. 調べ方

使った引用データベースと件数は次のとおりである．

- **Semantic Scholar API**（`/paper/arXiv:1808.05561/citations`，引用文脈 `contexts` つき）：320 件．
- **OpenAlex**（`cites:W2886300652` と，同じ論文の別の記録 `cites:W2950785196`）：298 件＋16 件．

2 つを題名で突き合わせた和は約 325 件である．この数には，同じ論文の予稿版と雑誌版の重複が含まれる．
- 題名に感情・情動・表情・感情極性・ストレス・抑うつの語を含むものは 217 件だった．
- そのうち 82 件は顔の表情認識の論文だった．

候補の絞り込みは 2 段で行った．
- 題名か抄録に「蒸留・教師・クロスモーダル・ラベル無し・疑似ラベル・半教師あり・転移・DA・コーパス間」の語を含む感情の論文 75 件の抄録を読んだ．
- Semantic Scholar の引用文脈（Albanie をどう引いたかの 1 文）を 141 件読んだ．

候補のうち，次の 10 本は PDF を入手して本文の該当箇所を読んだ．Zhang 2021，SetPeER 2025，ACL-MER 2025，Image2Audio 2020，EmoBed 2019，Hajavi & Etemad 2023，Liang 2020，Wei 2022，Peri 2021，Khare 2020．

### Albanie がどう引用されているか（引用文脈の読みから）

- **最も多い引用**：顔の表情認識の論文が，SENet（Albanie のラベル付けモデル）の FERPlus での正解率 88.8% を比較表に載せる引用である．例は TransFER（ICCV 2021），RAN（TIP 2020），Dual-Direction Attention Mixed Feature Network（2023）．どれも枠組みとは関係が無い．
- **次に多い引用**：蒸留の総説や別分野（行動認識，物体検出，読唇，人数計測）の論文が，「クロスモーダル蒸留の例」として 1 文で引く引用である．
- **枠組みを使うか延ばした感情認識の論文**：2 節の表の 10 本前後である．
- **Albanie の学習済みの声のモデルを部品として使った応用**：会話エージェントが声の感情を検出する研究（CHI 2021 の拡張要旨と TAC 2022）．
- **EmoVoxCeleb のラベルを正解として使った研究**：Peri ほか（ICASSP 2021），話者照合の Disentangled Speaker and Nuisance Attribute Embedding（2020），表情の合成の DualPathGAN（2021），データセット MEMOS（2020）の自動ラベル．

## 2. 分類表

列の意味は次のとおりである．
- (a) 転移の向き．
- (b) EmoVoxCeleb を使ったか．
- (c) 学習側のモデルを学ぶとき・評価するときに，ターゲット（評価）のコーパスの人のラベルを何本使ったか．「全部」は学習用の分割のラベルを全部使ったことを指す．
- (d) コーパス間の評価か．
- (e) 学習器．

| 論文 | (a) 向き | (b) EmoVox | (c) ターゲットのラベル | (d) コーパス間 | (e) 学習器 | 水準 |
|---|---|---|---|---|---|---|
| Albanie ほか ACM MM 2018（起点） | 顔 → 声 | 作った | 学習は 0 本．評価の線形の写像は 10 分割で全部 | ○（VoxCeleb → RML，eNTERFACE） | 声の CNN をゼロから学ぶ．評価は出力の上の線形 | ◎ |
| Zhang ほか KBS 2021 | 顔（＋声自身の予測）→ 声 | ラベル無しのデータとして使った | 全部（IEMOCAP 8 話者，CH-SIMS） | ラベル無しのデータだけ別コーパス（EmoVoxCeleb → IEMOCAP，iQIYI-VID → CH-SIMS） | wav2vec などの特徴の上の Transformer を学ぶ | ◎ |
| Tran・Yin・Soleymani TAC 2025（SetPeER，EmoCeleb） | 言葉＋顔 → 声，言葉＋声 → 顔 | 使わず，VoxCeleb2 で同じ手順を作り直した | 0 本（表 VI）と，微調整で全部（MSP-Podcast，MSP-Improv） | ○（EmoCeleb → IEMOCAP，MSP-Face） | HuBERT などを微調整 | ◎ |
| Zhuang ほか IJCNLP-AACL 2025（ACL-MER） | 声・顔・言葉の各大規模言語モデル → 3 モダリティの融合モデル | 使わない | 合計 100，200，500，1000 本 | ×（同じコーパス．MER2023，CHERMA，MELD） | 凍結した HuBERT・RoBERTa・CLIP の特徴の上の融合網（注意機構，MMIM，LMF） | ◎ |
| He ほか CVPRW 2020（Image2Audio） | 別コーパスの顔画像 → 声 | 使わない | 4 分の 1（4 分割の 1 つ） | 顔画像は別コーパス（Multi-PIE，CK+，MMI，Oulu-CASIA），声は CREMA-D と RAVDESS | GAN と VAE で顔画像をスペクトログラムに変換し，分類網を学ぶ | ◎ |
| Athanasiadis ほか Neurocomputing 2019（dacssGAN） | 顔 → 声（生成したデータで補う） | 使わない | 4 分の 1（Image2Audio がこの分け方を引用） | ×（同じコーパスの中．2 コーパスで実験） | 条件つき GAN | ○ |
| Han ほか TAC 2019（EmoBed） | 声 ↔ 顔（両方向） | 使わない | 全部（RECOLA，OMG-Emotion） | × | 網をゼロから学ぶ．ラベル付けモデルは使わず，共有の埋め込みを学ぶ | ◎ |
| Hajavi & Etemad IEEE TAI 2023 | 顔 → 声（特権情報として使う） | 使わない（VoxCeleb は話者認識の実験に使用） | 全部（RAVDESS，IEMOCAP） | × | 網を学ぶ | ◎ |
| Liang・Li・Jin ACM MM 2020 | 3 モダリティ間の分布を一致させる（疑似ラベルではない） | 使わない | 全部（IEMOCAP，MELD） | ラベル無しのデータだけ別コーパス（AMI → IEMOCAP） | 網を学ぶ | ◎ |
| Li・Zhao・Jin Interspeech 2021 | 言葉 → 声 | 使わない | 全部と半分 | × | 網を学ぶ | ◎（前回の調査 B） |
| Wei ほか Interspeech 2022 | 顔 → 声（共有の符号化器と領域識別器） | 使わない | 全部 | ×（IEMOCAP を無作為に 80/20 で分割．話者を分けていない） | 網を学ぶ | ◎ |
| Segment-Level CMKT（CCET 2021）と Cross-modal dynamic sentiment annotation（Comp. Elec. Eng. 2023） | 顔 → 声（発話の中の区間ごと） | 不明 | 不明 | 不明 | 網 | ○ / △ |
| Peri ほか ICASSP 2021 | （EmoVoxCeleb の自動ラベルを正解に使う） | 使った | EmoVox の自動ラベルを正解として全部 | × | 網 | ◎ |
| Khare ほか Interspeech 2020 | VoxCeleb2 で事前学習（話者と音声認識の課題）→ MOSEI | 感情の分布を論じる箇所で言及しただけ | 全部 | ○（事前学習のデータが別コーパス） | 網を微調整 | ◎ |
| Aslam ほか CVPRW 2023（Privileged KD） | 声＋顔 → 顔 | 使わない | 全部（RECOLA，Aff-Wild2） | × | 網 | ○ |
| SUMMER（ICME 2025） | 単一モダリティ → 融合 | 使わない | 全部（会話の感情認識） | × | 網 | ○ |
| FedCMD（ACM TIST 2024） | 生理信号 → 顔の映像 | 使わない | 0 本（運転者のラベル無しの映像） | 不明 | 網（連合学習） | ○ |
| CMED（Applied Sciences 2022） | 声のスペクトログラム → 声門電図（声帯の動きを測る信号） | 使わない | 全部 | × | 網 | ○ |

## 3. 枠組みを直接延ばした論文 7 本：何を変えて何が分かったか

### 3.1 Zhang, Chen, Chen, Li, Wu, Li, Zhu. "Combining cross-modal knowledge transfer and semi-supervised learning for speech emotion recognition." Knowledge-Based Systems 229, 107340 (2021) ◎

**Albanie との関係**
- 本文 3.2 節に "we follow Albanie et al. [14]" とある．顔のモデルが各フレームに出した予測を発話ごとに平均し，声の疑似ラベルにしている．
- Albanie の手を，比較の基準 "DCMKT"（直接のクロスモーダル転移）として再実装した．

**変えたこと**
- 顔の疑似ラベルに，声のモデル自身の予測を混ぜた．声の予測は，FixMatch（画像の半教師あり学習の手法）を声に移したもので作る．混ぜ方は 2 通りある．
  - 一致＋無作為：顔と声の予測が一致した発話を全部採る．残りの発話から割合 ε を無作為に選び，顔のラベルを付ける．
  - 重み付き和：声の確率に重み α を，顔の確率に重み 1−α を掛けて足す．
- 学習側のモデルは，ターゲットの全ラベルの交差エントロピーと，疑似ラベルの損失（重み λ=0.1）を同時に最小化して学ぶ．
- 顔のモデルは MobileNetV2 を RAF-DB で学んだものである．
- 声の特徴は，英語では wav2vec（Librispeech で事前学習），中国語では STFT である．その上に Transformer の符号化器を学ぶ．

**データ**
- IEMOCAP（4 感情，5531 発話）．8 話者で学習，1 話者で検証，1 話者で評価する（話者を分けている）．ラベル無しのデータとして EmoVoxCeleb を足した（4 感情に絞った）．
- CH-SIMS（3 つの感情極性）．ラベル無しのデータとして，iQIYI-VID から集めた 14,502 本を足した．

**分かったこと（表 5，IEMOCAP，乱数の種 3 つの平均，UA）**

| 手 | UA |
|---|---|
| ラベルだけ | 59.43% |
| Albanie の手をそのまま（EmoVoxCeleb で学んで IEMOCAP に当てる） | 27.99% |
| それを IEMOCAP の全ラベルで微調整 | 54.97%（ラベルだけより低い） |
| FixMatch | 57.71% |
| π モデル | 61.57% |
| Temporal ensembling | 58.46% |
| 顔と声を混ぜた疑似ラベル（重み付き和，α=0.2） | 62.50% |
| 顔と声を混ぜた疑似ラベル（一致＋無作為，ε=0.5） | 61.20% |

- Albanie の手をそのまま当てたときの低さについて，著者は「IEMOCAP と EmoVoxCeleb の分布の違いと，EmoVoxCeleb の感情の偏り（平静が多い）による」と書いている．

**分かったこと（表 3・表 4，CH-SIMS，UA）**
- ラベルだけ 44.91%．Albanie の手をそのまま当てると 44.94%，微調整すると 45.52%．
- 顔の疑似ラベルを足すと 45.28%．混ぜた疑似ラベル（ε=0.5）では 49.53%．
- ε=0（一致した発話だけを使う）では 37.11% に下がった．著者は「声と違う顔の知識が入らなくなるため」と解釈している．

**ほかの観察**
- CH-SIMS の単一モダリティのラベルを数えると，声と顔の感情が一致しない発話が 51.81% あった（図 1）．

**MER-DA との違い**
- ラベル付きのソースのコーパスが無く，ターゲットのラベルは全部使う．
- 学習側は声だけで，顔と融合の学習側は無い．
- 誤りを感情の対に分けた解析は無く，混同行列を示すだけである．

### 3.2 Tran, Yin, Soleymani. "SetPeER: Set-Based Personalized Emotion Recognition With Weak Supervision." IEEE TAC 2025 ◎

**Albanie との関係**
- 3 節に次の文がある．"we diverge from previous approaches such as the one by Albanie et al. [42], which utilizes a single modality input for cross-modal distillation (from vision to audio). Instead, we perform emotion recognition using two modalities to provide weak supervision."

**変えたこと**
- 3 モダリティ（顔・声・言葉）のうち 2 つでラベルを付け，残りの 1 つの学習側を学ぶ．方向は 2 つある．
  - 言葉＋顔 → 声（EmoCeleb-A）
  - 言葉＋声 → 顔（EmoCeleb-V）
- 言葉は VoxCeleb2 を自動で書き起こして作った．
- 2 つのラベル付けモデルの確率の KL ダイバージェンス（2 つの確率の食い違いの大きさ）が 1 以下の発話だけを残し，2 つの確率の平均を疑似ラベルにする．閾値 1 は，CMU-MOSEI の人のラベルとの一致を見て決めた．
- 声のラベル付けモデルは MSP-Podcast で学んだ WavLM，顔のラベル付けモデルは Aff-Wild2 で学んだモデルである．
- 嫌悪・恐れ・悲しみは数が少なすぎるので外した．悲しみを外している点は，MER-DA の 4 感情と違う．

**分かったこと：ラベル付けの正確さ（表 V，人のラベルとの一致の正解率 %）**

| ラベル付けに使ったモダリティ | CMU-MOSEI | MSP-Face |
|---|---|---|
| 顔 | 37.4 | 29.6 |
| 声 | 52.2 | 34.3 |
| 言葉 | 43.6 | 36.6 |
| 顔＋言葉 | 50.8 | 43.4 |
| 声＋言葉 | 57.2 | 41.8 |
| 人 1 人 | 70.8 | 69.4 |

- 著者は「2 つのモダリティを合わせると，1 つより常に良い」と書いている．

**分かったこと：別コーパスへの転移（表 VI，ターゲットのラベル 0 本，正解率 %）**
- IEMOCAP の声：EmoCeleb で学ぶと 48.1．MSP-Podcast（人のラベル）で学ぶと 53.8，CMU-MOSEI では 39.1，RAVDESS では 31.3．でたらめは 35.5．
- MSP-Face の顔：EmoCeleb で学ぶと 33.5，CMU-MOSEI では 32.3，RAVDESS では 12.9．

**分かったこと：そのほか**
- ラベル付けモデルを YouTube の動画にラベル無しの DA（勾配反転の領域識別器）で合わせると，CMU-MOSEI での成績がかえって下がった（表 III）．声は 52.2 から 46.2 に，顔は 37.4 から 35.8 に下がった．
- 発話の選別に KL と JS のどちらを使っても，MSP-Improv で微調整した後の成績に有意差は無かった（49.4 対 49.2．表 IV）．

**MER-DA との違い**
- 学習側は網の微調整で，凍結特徴＋線形ではない．
- 言葉の学習側と融合の学習側は無い．
- ターゲットに少数のラベル（K 本）がある条件は無く，0 本か全部のどちらかである．

### 3.3 Zhuang, Xiang, Tang, Zhang, Zhou. "Adaptive Collaborative Labeling with MLLMs for Low-Resource Multimodal Emotion Recognition." IJCNLP-AACL 2025 ◎

**Albanie との関係**
- 2 節で "Cross-modal distillation (Albanie et al., 2018) transfers knowledge from a well-resourced modality to a less-resourced one" と引用している．「少なくとも 1 つのモダリティにラベルが要る」ことを，Albanie の手の限界として挙げている．

**変えたこと**
- ラベル付けモデルを，モダリティごとの大規模マルチモーダル言語モデルにした．声は Qwen2-Audio-7B，顔は Qwen2-VL-7B，言葉は Qwen2.5-7B である．
- 各モデルに理由を書いた文と自信度を出させ，自信度で疑似ラベルを選ぶ．
- 学習側は 3 モダリティの融合モデルである．特徴は HuBERT-large，RoBERTa-large，CLIP-large から抽出したもので，MER-DA のエンコーダとほぼ同じ系統である．
- ラベルを合計 100，200，500，1000 本に絞り，学習データの残りをラベル無しにした．

**分かったこと（表 3，MER2023，注意機構の融合モデル，正解率）**

| 手 | 100 本 | 200 本 | 500 本 | 1000 本 |
|---|---|---|---|---|
| ラベルだけ | 0.432 | 0.573 | 0.749 | 0.826 |
| ラベル付けモデルの予測をそのまま疑似ラベルにする（T-PL） | 0.674 | 0.721 | 0.773 | 0.819 |
| 自分の予測を疑似ラベルにする（S-PL） | 0.543 | 0.668 | 0.770 | 0.839 |
| 提案（ACL-MER） | 0.678 | 0.767 | 0.827 | 0.849 |

- T-PL は 1000 本でラベルだけを下回った．
- MELD（付録の表 6）では，T-PL がすべてのラベル数でラベルだけを下回った（100 本で 0.450 から 0.345）．
- 付録の表 7（CHERMA）では，co-training とも比べた．co-training は 1 つのモダリティと残り 2 つで見方を分ける手で，「声｜言葉＋顔」などの組を比べている．
- 表 1：ラベル付けモデルが自信度 5 を付けた発話は，声で全体の 11.3%，顔で 24.3% だった．その発話での正解率は 0.787 と 0.773 だった．

**MER-DA との違い**
- 同じコーパスの中で学習と評価を分けた設定で，ソースのコーパスもコーパス間の評価も無い．
- 評価は融合だけで，単一モダリティの学習側の成績は無い．
- ラベルは合計本数で数え，元の感情の割合を保って抜いた．クラスあたりの本数にはそろえていない．

### 3.4 He, Liu, Fan, You. "Image2Audio: Facilitating Semi-supervised Audio Emotion Recognition with Facial Expression Image." CVPRW 2020 ◎

**Albanie との関係**
- 序論で，Albanie を「顔以外のモダリティはラベル付きのデータが少ない」ことの例として引用している（文献 [1]）．
- 声と顔が対になった動画が要らない点を，Albanie との違いにしている．

**変えたこと**
- 異種の半教師ありの DA にした．ソースはラベル付きの顔画像のコーパス（Multi-PIE，CK+，MMI，Oulu-CASIA を合わせたもの），ターゲットは声（CREMA-D，RAVDESS）である．
- 顔画像を感情ごとにスペクトログラムへ変換する GAN＋VAE を学ぶ．生成したスペクトログラムとラベル無しの声を足して，声の分類網を学ぶ．

**データの分け方**
- Athanasiadis 2019 に従い，各コーパスを 4 分割した．1 つをラベル付き，2 つをラベル無し，1 つを評価に使う．
- 話者を分けたかどうかは本文に書かれていない．

**分かったこと（表 1，正解率）**
- CREMA-D：声のラベルだけでは 30.81%．全部の手を使うと，対の情報なしで 54.53%，対の情報ありで 58.71%．
- RAVDESS：声のラベルだけでは 30.65%．同様に 53.34% と 56.12%．

**MER-DA との違い**
- ソースは顔画像だけで，ソースの声や言葉は無い．
- ラベル付けモデルの疑似ラベルではなく，生成したデータで補う．
- 3 モダリティも融合も無い．

### 3.5 Han, Zhang, Ren, Schuller. "EmoBed: Strengthening Monomodal Emotion Recognition via Training with Crossmodal Emotion Embeddings." IEEE TAC 2019（arXiv 1907.10428）◎

**Albanie との関係**
- 2.1 節に "Our approach differs from the one proposed by Albanie et al. [30] mainly in two key aspects" とある．
  - 1 つめの違い：Albanie は，よく学んだ顔のモデルを必要とする．
  - 2 つめの違い：Albanie の結果について "the performance of the student network falls short of the performance of the teacher network, and cannot compete with a fully supervised network" と書いている．

**変えたこと**
- ラベル付けモデルを使わない．声と顔の網を，共有の部分網とトリプレット損失で同時に学ぶ．推論では片方のモダリティだけを使う．

**分かったこと**
- OMG-Emotion（7 感情，F1，表 3）：声は 36.5 から 41.7 に，顔は 37.9 から 43.9 に上がった．
- RECOLA の回帰でも，声と顔の両方が上がった．例：声の覚醒度の CCC が 0.605 から 0.644 に上がった．

**MER-DA との関係**
- 「声と顔の各単一モダリティを両方とも良くする」を目標にした点が，研究者の拡張の軸の 1 つと重なる．
- ただし 2 モダリティで，同じコーパスの中で，ラベルを全部使っている．

### 3.6 Hajavi, Etemad. "Audio Representation Learning by Distilling Video as Privileged Information." IEEE TAI 2023（arXiv 2302.02845）◎

**Albanie との関係**
- 2 節で，Albanie を「顔の予測を声の唯一の学習信号にする手」として挙げている．
- Albanie 型の確率の蒸留（soft-label distillation）を，比較の基準として再実装した．

**変えたこと**
- 人のラベルがある状態で，顔のモデルの確率ではなく，中間層の埋め込みを声のモデルに写す．

**分かったこと**
- RAVDESS（表 VI）：声のモデルの正解率は，写さないとき 51.0〜57.6%，提案の手で写すと 52.3〜59.9%．
- 本文に次の文がある．"while the baseline models that utilize the video as the only source of supervision for learning audio are successful when the ground-truth labels are not present, they do not improve the performance ... when training is done using both modalities"．
- RAVDESS の図 5 では，Albanie 型の確率の蒸留が成績を下げた．
- IEMOCAP（表 VII）では，顔のモデル（60.8〜65.5%）が声のモデル（68.6〜71.8%）より弱い．この条件で，確率の蒸留は小さい重みのときから成績を下げ，提案の手は成績を変えなかった．

**MER-DA との関係**
- 同じコーパスの中で，次の 2 点が報告されている．
  - ラベルが十分にあるとき，別モダリティの確率を足すと損になりうる．
  - ラベル付けモデルが学習側より弱いと損になる．

### 3.7 Liang, Li, Jin. "Semi-supervised Multi-modal Emotion Recognition with Cross-Modal Distribution Matching." ACM MM 2020（arXiv 2009.02598）◎

**Albanie との関係**
- 序論に "Albanie et al. [1] apply transfer learning to obtain supervision from another labeled modality. However, the improvement is very marginal." とある．
- 疑似ラベルは使わない．代わりに，ラベル無しの発話で 3 モダリティの表現の分布を近づける損失（MMD）を使う．

**データ**
- IEMOCAP（話者を分ける）．ラベル無しのデータとして，別コーパスの AMI（会議の動画）から切り出した 20,000 本を足した．

**分かったこと（表 5，UA）**
- 声＋顔：60.8% から 61.2%．
- 声＋言葉：68.6% から 72.1%．
- 声＋顔＋言葉：72.5% から 74.5%．
- 単一モダリティの学習側は，著者が "needs at least two modalities" と書いているとおり測っていない．

**MER-DA との関係**
- 3 モダリティ，融合，別コーパスのラベル無しのデータ，の組み合わせを持つ．
- ただしターゲットのラベルは全部使い，単一モダリティの成績は無い．

### 3.8 補足（短く）

- **Li, Zhao, Jin. Interspeech 2021** ◎（前回の調査 B）
  - 言葉 → 声の蒸留である．Albanie と "Different from the visual-audio cross-modal distillation [15], we consider using text-audio cross-modal transfer" と対比している．
  - 微調整のラベルが半分でも，蒸留なしで全ラベルを使ったときと同じ水準だった（IEMOCAP の UAR 0.590 対 0.589）．
- **Wei ほか Interspeech 2022** ◎
  - 顔 → 声を，共有の符号化器と勾配反転で写した．IEMOCAP の UA は 71.32% から 73.22% に上がった．
  - 怒りと悲しみの正解率がそれぞれ 2.56 ポイントと 2.6 ポイント上がった．著者は「この 2 つは顔の手がかりがはっきりしているから」と書いている．
  - ただし学習と評価を無作為に 80/20 で分けており，話者を分けていない（3.2 節）．
- **Athanasiadis, Hortal, Asteriadis. Neurocomputing 2019** ○
  - 顔の情報から声の特徴を条件つき GAN で生成し，学習データに足す．
  - 大学のリポジトリの抄録によると，2 つのコーパスで正解率が 49.34% から 50.29%，46.90% から 48.65% に上がった．
  - 同じグループの研究に，ICIP 2020（段階的な GAN），ACIIW 2021，博士論文「Emotion-aware cross-modal domain adaptation in video sequences」（Maastricht 2022）△ がある．

## 4. 研究者の拡張の各軸を扱った後続研究

| 軸 | 扱った後続研究 | 扱い方の限界 |
|---|---|---|
| ターゲットに少数のラベル（クラスあたり K 本） | ACL-MER 2025（合計 100〜1000 本），Image2Audio 2020（4 分の 1），Athanasiadis 2019（4 分の 1），Li 2021（半分） | クラスあたりの本数にそろえた研究は無い．ACL-MER は合計本数で数え，元の割合を保つ．K を 0 から 25 のように細かく変えた研究は無い |
| 3 モダリティ | SetPeER 2025（2 つで残りの 1 つにラベルを付ける），ACL-MER 2025（3 つの融合），Liang 2020（3 つの融合），Khare 2020（ラベル付けではなく事前学習） | SetPeER は言葉を学習側にしない．ACL-MER と Liang は融合だけを評価する |
| 各単一モダリティと融合の両方を良くする | EmoBed 2019（声と顔の単独を両方），SetPeER 2025（声と顔の単独を別々に），Khare 2020（声，声＋顔，声＋言葉，3 つの順に事前学習の効果を示す） | 3 つの単独と融合の 4 つを同じ手で良くし，4 つとも成績を出した研究は見つからなかった．Liang は単独を測れない手だと明記している |
| ソースとターゲットが別コーパス（DA） | Zhang 2021（ラベル無しのデータが別コーパス），SetPeER 2025（EmoCeleb → IEMOCAP・MSP-Face），Image2Audio 2020（顔画像のコーパス → 声のコーパス），Albanie 自身（VoxCeleb → RML・eNTERFACE） | どの研究も「ラベル付きのソースのコーパス群，ターゲットの K 本，ターゲットのラベル無し」の 3 つを同時には持たない．SetPeER のラベル付けモデルは別コーパス（MSP-Podcast，Aff-Wild2）で学んだものだが，ターゲットに合わせて学び直していない |
| 凍結特徴の上の線形の学習側 | Albanie の評価だけ（声の学習側のモデルの 8 次元の出力の上にアフィン写像を学ぶ） | 後続研究はどれも網を学ぶ．ACL-MER は凍結特徴を使うが，その上に注意機構などの網を学ぶ |

## 5. 最も近い 4 本と MER-DA の比較（研究目標が達成されたとして）

| | Zhang 2021 | SetPeER 2025 | ACL-MER 2025 | Image2Audio 2020 | MER-DA |
|---|---|---|---|---|---|
| データ | IEMOCAP，CH-SIMS ＋ 別コーパスのラベル無し | VoxCeleb2 → IEMOCAP，MSP-Face，MSP-Podcast，MSP-Improv | MER2023，CHERMA，MELD（中国語が中心） | 顔画像 4 コーパス → CREMA-D，RAVDESS | 7 群（IEMOCAP 即興・台本，MOSEI，MELD，AFEW，RAVDESS，CREMA-D） |
| ターゲットのラベル | 全部 | 0 本か全部 | 合計 100〜1000 本 | 4 分の 1 | クラスあたり K 本 |
| モダリティ | 声と顔（学習側は声） | 声・顔・言葉（学習側は声か顔） | 声・顔・言葉（学習側は融合） | 顔画像と声（学習側は声） | 声・顔・言葉（学習側は各単独と融合） |
| 学習器 | Transformer を学ぶ | 網を微調整 | 凍結特徴＋融合網 | GAN＋VAE＋分類網 | 凍結特徴＋線形ロジスティック回帰 |
| 測る量 | WA，UA，重み付き F1 | 正解率，F1，ラベル付けの正確さ | 正解率，重み付き F1 | 正解率，IS，FID | UA，誤りの位置と方向（感情の対ごと） |
| 主張 | 顔と声の予測を混ぜた疑似ラベルは，顔だけ・声だけより良い | 2 つのモダリティでラベルを付けると 1 つより正確．EmoCeleb で事前学習すると個人化に効く | モダリティ別の大規模言語モデルの疑似ラベルを自信度で選ぶと，100 本の条件で大きく良くなる | 顔画像から生成したデータで，声の少数ラベルを補える | （研究者が決める） |
| MER-DA と同じ点 | 別モダリティの予測を声の疑似ラベルにし，自分の予測と比べる | 3 モダリティ，別コーパスでの評価，2 つのモダリティでのラベル付け | 少数ラベル，ほぼ同じエンコーダの凍結特徴，ラベル数とともに得が減る | 別コーパスのラベル付きデータ，ターゲットの少数ラベル，ラベル無しを併せ持つ | |
| MER-DA と違う点 | ソースのラベル付きコーパスが無い．全ラベルを使う | 少数ラベルの条件が無い．融合と言葉の学習側が無い | 同じコーパスの中だけ．単独の評価が無い | ソースが顔画像だけ．疑似ラベルではなく生成 | |

## 6. 先行研究がやっていないこと（事実の列挙）

ここに並べるのは，今回の検索で見つからなかったことであり，設計の案ではない．MER-DA の棚卸し（`DAI_paper/inventory.md`）で既に測れているもの，または保存済みの出力から出せるものを先に書く．

1. **別モダリティの疑似ラベルの得を，感情の対ごとに分けて測った後続研究は無い．**
   - 後続研究は，混同行列（Zhang 2021 図 3）か，感情ごとの正解率（Wei 2022 の怒り・悲しみ）までしか示していない．
   - Albanie 自身も 5.3 節で「感情によって，モダリティ間で重なる情報の度合いが違うかもしれない」と書いたが，測っていない．
   - MER-DA の記録では，棚卸しの行 8 と行 17 がこの点に当たる．行 8 は，別モダリティを含むモデルの疑似ラベルが方向の誤りを減らした単位が 81 のうち 38 だったことである．行 17 は，ソースだけで学んだ分類器の方向の誤りを感情の対ごとに測った記録である．
   - ただし「手法がどの対を直したか」はまだ測っていない（行 17 の棚卸しの判定は「不十分」）．
2. **誤りを切片（位置）と重みの向き（方向）に分けた後続研究は無い．**後続研究はどれも正解率・UA・F1 で比べている．
3. **同じモダリティのラベル無しの手と，別モダリティの疑似ラベルを比べた研究はある．**
   - Zhang 2021 は，FixMatch・π モデル・Temporal ensembling と顔の疑似ラベルを比べた．
   - ACL-MER は，自分の予測の疑似ラベル，ラベル付けモデルの疑似ラベル，co-training を比べた．
   - ただし比べた単位は正解率であり，コーパスの組ごと・K ごとに数えた研究は無い．MER-DA の行 5・行 8 の数え上げ（8 手法を 210 単位で，別モダリティを 81 単位で数えたもの）は，この点で先行研究より細かい．
4. **「疑似ラベルの得はラベルが増えると減る」は，後続研究で既に出ている**（ACL-MER の表 3・表 6，Zhang 2021 の微調整，Hajavi & Etemad 2023）．一方，MER-DA の行 14 のような「ラベル何本ぶん」への換算や，K との関係を式で書いた後続研究は見つからなかった．
5. **「ラベル付けモデルは複数のモダリティを合わせた方が正確」も，既に出ている**（SetPeER 表 V，ACL-MER 表 4，Zhang 2021）．
   - SetPeER は，学習側のモダリティを除いた 2 つでラベルを付ける．Zhang 2021 は，学習側のモダリティ自身を含めて付ける．どちらの作り方も先行研究にある．
   - MER-DA の行 9 には「配備する分類器のモダリティを除いた融合は作っていない」とある．この 2 つの作り方を，同じ学習側について並べて比べた研究は見つからなかった．
6. **凍結した特徴の上の線形の学習側で，疑似ラベルの形（柔らかい，温度，硬い，釣り合い）を替えて比べた後続研究は無い．**
   - 近いのは次の 2 つで，どちらも学習側は網である．SetPeER の KL と JS の選別の比較（差なし．網の微調整）と，Albanie の温度 2 の確率とロジットの二乗誤差の比較（網をゼロから学ぶ）．
   - MER-DA の行 10（21 のうち 0 で差なし）は，線形の場合の記録である．
7. **ラベル付きのソースのコーパス群とターゲットの K 本を両方持ち，ターゲットのラベル無しに Albanie 型の疑似ラベルを付けた研究は無い．**Zhang 2021 はラベル無しのデータが別コーパスであること，SetPeER は別コーパスで学んだラベル付けモデル，Image2Audio は顔画像のソース，と，それぞれ一部だけを持つ．
8. **顔の学習側で得が小さい理由を調べた後続研究は無い．**
   - SetPeER の表 VI では，顔（MSP-Face）の成績はどの学習データでも低い（12.9〜33.5%）が，理由の分析は無い．
   - MER-DA の行 15 も「理由が無い」と判定している．
9. **話者を分け，複数のターゲットのコーパスを順に回した後続研究は無い．**
   - Zhang 2021 は IEMOCAP で話者を分けたが，ターゲットは IEMOCAP と CH-SIMS の 2 つだけである．
   - Wei 2022 は話者を分けていない．Image2Audio は分け方を書いていない．

## 7. 見つからなかったこと

- Albanie の枠組みを，クラスあたり K 本のラベルで K を変えて調べた研究は，今回の検索では見つからなかった．
- 言葉を学習側（配備するモダリティ）にして，声や顔の疑似ラベルで学んだ研究は見つからなかった．SetPeER でも ACL-MER でも，言葉はラベル付けの側か融合の一部である．
- 3 つの単一モダリティと融合の 4 つの学習側を，同じ疑似ラベルの手で学び直し，4 つとも成績を出した研究は見つからなかった．
- Albanie の 5.3 節の予想（感情ごとにモダリティ間の重なりが違う）を直接確かめた後続研究は見つからなかった．
- EmoVoxCeleb をラベル付きのソースのコーパスとして，人手のラベルのほかのコーパスと並べて混ぜた研究は見つからなかった．Zhang 2021 はラベル無しのデータとして使い，Peri 2021 は単独で正解に使っている．
- LanSER 2023 と Srinivasan 2022 は，引用データベースで Albanie を引用した論文として登録されていなかった．両者は前回の調査で読んだので，ここでは扱わない．

## 8. 調査の限界

- 件数は，Semantic Scholar が 320 件，OpenAlex が 314 件だった．Google Scholar の引用数（調べていない）より少ない可能性がある．中国語の論文と学位論文の多くは入っていない．
- 抄録が無い論文は，題名と引用文脈だけで判定した．例は，Segment-Level CMKT の 2023 年の続編，SSRN の会話の感情認識の蒸留など．Zhang 2021 も OpenAlex に抄録が無かったが，本文を別に入手して読んだ．
- Elsevier・IEEE などの有料の論文のうち，次の 7 本は本文を読んでいない（水準は ○ か △）．Athanasiadis 2019，Segment-Level CMKT 2021・2023，FedCMD 2024，Aslam 2023，SUMMER 2025，CMED 2022．
- Image2Audio の表 1 の「声のラベルだけ」の 30.81% は，同じコーパスのほかの報告より低い．この比較の基準をどう作ったかは，本文から読み取れなかった．
- ACL-MER の表 3 の値と Zhang 2021 の表 5 の値は，PDF の文字から読んだ．論文に引用する前に，元の表で確かめる．

## 9. 出典（URL）

- Albanie ほか 2018：https://arxiv.org/abs/1808.05561 ◎（`DAI_paper/参照論文/2018_Albanie_CrossModalTransferSER_ACMMM.pdf`）
- 引用一覧：Semantic Scholar API https://api.semanticscholar.org/graph/v1/paper/arXiv:1808.05561/citations ，OpenAlex https://api.openalex.org/works?filter=cites:W2886300652
- Zhang ほか 2021（KBS 229, 107340）：https://doi.org/10.1016/j.knosys.2021.107340 ，本文 https://users.monash.edu/~yli/assets/pdf/2021-KBS-SpeechEmotionRecognition.pdf ◎
- Tran・Yin・Soleymani 2025（SetPeER，TAC）：https://doi.org/10.1109/taffc.2025.3568024 ，本文 https://par.nsf.gov/servlets/purl/10620792 ◎
- Zhuang ほか 2025（ACL-MER）：https://aclanthology.org/2025.ijcnlp-long.152/ ◎
- He ほか 2020（Image2Audio）：https://openaccess.thecvf.com/content_CVPRW_2020/html/w54/He_Image2Audio_Facilitating_Semi-Supervised_Audio_Emotion_Recognition_With_Facial_Expression_Image_CVPRW_2020_paper.html ◎
- He ほか 2020（Classification-aware Semi-supervised DA．Image2Audio と同じ著者・同じ抄録の別の版）：https://openaccess.thecvf.com/content_CVPRW_2020/html/w56/He_Classification-Aware_Semi-Supervised_Domain_Adaptation_CVPRW_2020_paper.html ○
- Athanasiadis ほか 2019（Neurocomputing）：https://doi.org/10.1016/j.neucom.2019.09.106 ，https://cris.maastrichtuniversity.nl/en/publications/9e9e1739-8c37-4eae-9680-748d73af094c ○
- Han ほか 2019（EmoBed）：https://arxiv.org/abs/1907.10428 ◎
- Hajavi & Etemad 2023：https://arxiv.org/abs/2302.02845 ◎
- Liang・Li・Jin 2020：https://arxiv.org/abs/2009.02598 ◎
- Li・Zhao・Jin 2021（Interspeech）：https://doi.org/10.21437/interspeech.2021-785 ◎（前回の調査 B）
- Wei ほか 2022（Interspeech）：https://www.isca-archive.org/interspeech_2022/wei22b_interspeech.html ◎
- Peri ほか 2021：https://arxiv.org/abs/2102.06269 ◎
- Khare ほか 2020：https://arxiv.org/abs/2009.05019 ◎
- Segment-Level CMKT 2021：https://doi.org/10.1109/ccet52649.2021.9544303 ○
- Cross-modal dynamic sentiment annotation 2023：https://doi.org/10.1016/j.compeleceng.2023.108598 △
- Aslam ほか 2023（Privileged KD，CVPRW）：https://doi.org/10.1109/cvprw59228.2023.00336 ○
- SUMMER（ICME 2025）：https://arxiv.org/abs/2503.23721 ○
- FedCMD（ACM TIST 2024）：https://doi.org/10.1145/3650040 ，原稿 https://iris.cnr.it/bitstream/20.500.14243/511509/1/TIST-2023-04-0156_Proof_hi.pdf ○
- CMED（Applied Sciences 2022）：https://doi.org/10.3390/app12094338 ○
- Cross-Modal Knowledge Transfer via Inter-Modal Translation and Alignment for Affect Recognition（2021，CMU-MOSI と RECOLA で単一モダリティを改善）：https://arxiv.org/abs/2108.00809 ○
