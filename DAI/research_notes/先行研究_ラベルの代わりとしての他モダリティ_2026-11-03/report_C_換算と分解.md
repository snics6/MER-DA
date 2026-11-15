# 先行研究の調査 C：ラベル何本ぶんの換算と，位置・方向の分解（2026-11-03）

担当：prior-work-diff（C）．調査役の実行環境がファイルの作成を拒んだため，本体のセッションが返答の本文をそのまま保存した．取得した PDF は調査のセッションの scratchpad（`pdfs/`）にあり，セッションが終わると消える．`results/` は開いていない．

## 要約

1. 「別モダリティの教師は配備する分類器にとってラベルの代わりである」という見方そのものは既知である．de Sa（NIPS 1993）は，別の感覚モダリティの構造を使えば「ラベルを代用できる（substitute for the labels）」と書いた．Blum & Mitchell（COLT 1998）は，条件つき独立のもとでは一方のビューの予測が他方のビューにとって雑音の入ったラベルと同じになることを定理にした．
2. ラベル何本ぶんで測る方法も既知である．Snorkel DryBell（SIGMOD 2019）は，配備時に使えない資源から作った弱いラベルの価値を「手で付けたラベル 8 万本ぶん・1.2 万本ぶん」と読んだ．本物のラベルが増えると価値が減る型も既知である（Reimherr ほか 2021，Hernandez ほか 2021，Zoph ほか 2020，Zhu ほか 2023）．
3. 位置と方向の分け方について．「ラベル無しのデータは事前確率（切片）を直す」は，ラベルシフトの研究の前提として既知である（Saerens ほか 2002，Lipton ほか 2018，Maity ほか 2024）．「別ビューの疑似ラベルは方向を運び，閾値だけがずれる」は，雑音ラベルの理論（Natarajan ほか 2013，Menon ほか 2015）と共同学習の理論（Blum & Mansour 2017）から 1 段で出る．一方，3 つの情報源を同じ分け方で実データで測った研究は見つからなかった．
4. 最も近い先行研究は Blum & Mansour 2017，Menon ほか 2015，Snorkel DryBell 2019，Reimherr ほか 2021，de Sa 1993 の 5 本である．
5. 上澄み（先行研究を超える部分）は 2 つ残る．(i) K 本・同じモダリティのラベル無し・別モダリティの疑似ラベルの 3 つを同じ位置・方向の分け方で実測し，理論が予言する分業が実データで起きることを示した点である（照合済みの記録がある）．(ii) 理論の前提である条件つき独立が崩れた条件でも方向が直る点である（検査前の記録だけで，未検証）．「価値が K とともに減る」は型として既知であり，MER-DA の記録でも使い方によっては減らない（検査前）．これを新しさにしてはいけない．
6. 狙うべき差分は，疑似ラベルのモデルの「ラベル何本ぶん」を，位置の分と方向の分に分けて測ることである．この量は先行研究にも MER-DA の記録にも無く，主張 (a) と (b) を 1 つの量でつなぐ．

## 0. この報告で使う語

- **配備する分類器**：ターゲットで実際に使う分類器．凍結した特徴の上の線形分類器で，入力は 1 つのモダリティか融合である．
- **疑似ラベルのモデル**：ラベル無しのターゲットの発話に予測（確率）を付けるモデル．別モダリティを含む．
- **出発点**：全ソースで学んだ分類器．学び直しの初期値であり，L2 正則化で引き戻す先でもある．
- **位置の誤り／方向の誤り**：分類器の誤りを 2 つに分けたもの．位置の誤りは切片（感情ごとの底上げ）のずれで失う分，方向の誤りは重みの向きのずれで失う分である．
- **ラベル何本ぶん（ラベル換算の価値）**：ある情報源を足したときの成績に，本物のラベルだけで学んだ分類器が何本で届くかを，学習曲線で読んだ値．統計学では実効標本数（effective sample size），Hernandez ほか 2021 は effective data transferred と呼ぶ．
- **条件つき独立**：正解の感情を固定したとき，2 つのモダリティの特徴が互いに独立であること．共同学習（co-training．2 つのビューが互いの予測をラベルにして学ぶ方法）の理論が標準に置く仮定である．
- **クラス条件つきの雑音**：ラベルの誤りの起こり方が正解のクラスだけで決まり，入力には依らないこと．
- **BER（均衡誤り率）**：クラスごとの誤り率の平均．2 クラスでは 1 − UAR に等しい．

確かめた水準は次の 3 段階である．◎ は原文（PDF を pdftotext で文字にしたもの）で該当箇所を読んだもの，○ は抄録か信頼できる二次資料，△ は検索の断片だけのもの．既存のノートで ◎ と記録された文献は「既存 ◎」と書き，本調査では読み直していない．

---

## 1. 判定

| 主張 | 判定 | 根拠（水準） |
|---|---|---|
| 主張 1：配備する分類器が別モダリティの教師から受け取るのは，入力の手がかりではなくラベルの代わりの情報である | **既知**（概念として） | de Sa 1993 ◎，Blum & Mitchell 1998 ◎，Blum & Mansour 2017 ◎，Lopez-Paz ほか 2016（既存 ◎），Snorkel DryBell 2019 ◎ |
| 主張 2：その価値はラベル何本ぶんで測れ，本物のラベルが増えると減る | **測り方も減る型も既知．別モダリティの教師で K を振って測った例は見つからなかった** | DryBell ◎，CLIP（既存 ◎），Hernandez 2021（既存 ◎），Reimherr ほか 2021（既存 ◎），Zoph ほか 2020 ◎，Zhu ほか 2023 ○，Hanneke & Kpotufe 2019 ◎ |
| 主張 3：K 本と同じモダリティのラベル無しは主に位置を直す | **K 本の側は既知（率の式）．ラベル無しの側は前提として既知．実測で方向を直さないと示した例は見つからなかった** | Okamoto 1963・Raudys 1998（既存 ◎），Saerens ほか 2002 ◎，Lipton ほか 2018 ○，Maity ほか 2024 ◎，Batch Calibration 2024 ○，Veilleux ほか 2021 ○ |
| 主張 4：別モダリティの疑似ラベルは方向を直す | **理論の予言としては既知．実データで分けて測った例は見つからなかった** | Blum & Mitchell 1998 ◎，Natarajan ほか 2013 ◎，Menon ほか 2015 ◎，Blum & Mansour 2017 ◎，Manwani & Sastry 2013 ○，Lachenbruch 1966 △ |

### 1.1 主張 1：ラベルの代わり

**de Sa, "Learning Classification with Unlabeled Data", NIPS 6, pp. 112–119, 1993**（◎ 抄録と 3 節．https://papers.neurips.cc/paper_files/paper/1993/file/e0ec453e28e061cc58ac43f91dc2f3f0-Paper.pdf ）

- 抄録の原文："we show that we can substitute for the labels by making use of structure between the pattern distributions to different sensory modalities. We show that minimizing the disagreement between the outputs of networks processing patterns from these different modalities is a sensible approximation to minimizing the number of misclassifications in each modality"
- 3 節の原文："classes that are easily confuseable in one modality may be well separated in another. This should improve the performance of the algorithm as the "labeling" signal for separating the overlapping classes will be more reliable"．音声と読唇については "the speech sounds that are confuseable are not confuseable visually and vice-versa ... the other modality provides more reliable labeling where it is needed most" と書いている．
- 実験は Peterson–Barney の母音のフォルマントで行っている．2 つの「モダリティ」に同じデータセットを割り当てた人工の設定であり，感情認識でも，配備時に片方のモダリティしか使わない設定でもない．
- この論文は主張 1 だけでなく，「別のモダリティは，配備する分類器のモダリティで紛らわしい感情の対を分ける」という対ごとの見方（棚卸しの行 17・22）の先例でもある．棚卸しの行 22 の「別のモダリティが特定の対を解消するかを測った研究は見つかっていない」は，「実データの感情で測った研究は見つかっていない」に限って書く必要がある．

**Blum & Mitchell, "Combining Labeled and Unlabeled Data with Co-Training", COLT 1998**（◎ 4 節の定理 1 と証明．https://www.cs.cmu.edu/~avrim/Papers/cotrain.pdf ）

- 定理 1 の原文："If C2 is learnable in the PAC model with classification noise, and if the conditional independence assumption is satisfied, then (C1, C2) is learnable in the Co-training model from unlabeled data only, given an initial weakly-useful predictor h(x1)."
- 証明の原文："By the conditional independence assumption, for a random example x = (x1, x2), h(x1) is independent of x2 given f(x). Thus, if we use h(x1) as a noisy label of x2, then this is equivalent to (α, β)-classification noise"
- つまり，一方のビュー（モダリティ）で作った予測は，もう一方のビューの分類器にとって，誤りがクラスだけで決まる雑音ラベルと同じになる．条件つき独立のもとで主張 1 を定理にしたものである．

**Lopez-Paz, Bottou, Schölkopf, Vapnik, ICLR 2016**（既存 ◎．`docs/related-work.md` 1.14 節 (g)）は，入力のモダリティだけが違う課題では，他のモダリティを特権情報（学習時にだけ使える情報）とみなせると書いている．配備する分類器の入力は固定なので，教師の情報は学習の目標（ラベル）を通してしか入らない．したがって MER-DA の設定では，主張 1 は定義からほぼ自明であり，新しさにはならない．

### 1.2 主張 2：ラベル何本ぶんと，その K 依存

**測り方の先例**

- **Snorkel DryBell**（Bach ほか，SIGMOD 2019．◎ 抄録，1 節，6 節．https://arxiv.org/abs/1812.00417 ）
  - 配備時に使えない資源（"non-servable organizational resources"）を使うラベル付け関数で弱いラベルを作り，配備できる特徴だけを使う分類器を学ぶ．
  - 6 節の原文："it takes roughly 80K hand-labeled examples to match the predictive accuracy of the weakly supervised classifier. On the product classification task, we find that it takes roughly 12K hand-labeled examples"
  - 配備できる特徴だけのラベル付け関数と比べた除去実験（表 3）では，配備時に使えない資源を足すと F1 が平均 52% 上がる．
  - 配備する分類器が見ない情報を弱いラベルに変え，その価値を手で付けたラベルの本数で読む点で，主張 1・2 の測り方に最も近い．ただし，手で付けたラベルの数を振って価値の変化を測ってはいない．位置と方向への分解も無い．
- **CLIP**（Radford ほか 2021．既存 ◎）：同じ凍結特徴の上のロジスティック回帰の学習曲線で，ゼロショット分類器がクラスあたり何本に当たるかを読んだ（中央値 5.4 本）．測り方は MER-DA と同じである．
- **Hernandez ほか 2021** と **Le Scao & Rush 2021**（どちらも既存 ◎）：事前学習やプロンプトの価値を，学習データの量に換算して読んでいる．

**価値が本物のラベルとともに減ることの先例**

- **Hernandez ほか 2021**（既存 ◎）
  - 1.1 節の原文："pre-training effectively multiplies the fine-tuning dataset, DF, in the low-data regime ... Note that the multiplier goes down as DF increases."
  - 減るのは倍率（換算した量 ÷ 本物の量）である．換算した量そのもの D_T は，微調整データ量 D_F の 0.18 乗で増える．MER-DA の「本数そのものが減る」より弱い主張である．
- **Reimherr, Meng, Nicolae, JRSS-B 83(3):413–437, 2021**（既存 ◎．`新規性と研究価値_2026-10-29/G_第2回.md`・`H_第3回.md`）
  - 事前分布の価値を「同じ誤差に届くのに要る追加の標本数」として，本物の標本数 k の関数で測る．
  - k とともに減ることを，事前分布とデータが食い違っている印として読む．減り方の傾きの下限は −1 である．
  - 本数そのものが減る型は，この論文に既にある．
- **Zoph ほか, "Rethinking Pre-training and Self-training", NeurIPS 2020**（◎ 抄録と 4.1・4.2 節．https://arxiv.org/abs/2006.06882 ）
  - 抄録の原文："stronger data augmentation and more labeled data further diminish the value of pre-training, 2) unlike pre-training, self-training is always helpful ... in both low-data and high-data regimes"
  - COCO のラベルを 20%・50%・100% と増やすと，事前学習の得は消えて損に変わる．ターゲットのラベルで学んだ教師の疑似ラベルによる自己学習は，+1.3〜+3.4 AP の得を保つ．
  - 「ソースで固定した資源の価値は減り，ターゲットのラベルで学び直した教師の価値は減らない」という対比は，棚卸しの行 14 と同じ向きである（行 14 は，ソースだけの融合では減り，学び直して掛けたモデルでは減らないとする未照合の数え上げ）．ただし Zoph らはラベル何本ぶんに換算していない．
- **Zhu ほか, "Weaker Than You Think", ACL 2023, pp. 14229–14253**（○ 抄録．https://aclanthology.org/2023.acl-long.796/ ）：弱教師ありの手法の得は，きれいなラベルがクラスあたり 5 本あれば，それで直接学ぶことでほぼ消える．
- **Hanneke & Kpotufe, "On the Value of Target Data in Transfer Learning", NeurIPS 2019**（◎ 1 節．https://arxiv.org/abs/2002.04747 ）
  - 原文："the best possible rate ... is - up to constants - achievable by an oracle which simply ignores the least informative of the source or target datasets"
  - 最悪の場合の率では，ソースとターゲットの価値は足し算ではなく，大きい方で決まる．ターゲットのラベルが十分に増えると，ソースが足す価値は 0 になる（この読み替えは本調査のもの）．
- **Jain, Montanari, Sasoglu, NeurIPS 2024** と **Li, Han, Yang, Tong, ICML 2026**（既存 ◎）：2 乗損失の模型では，代わりのデータの割合は本物の本数とともに下がる．一方で，ラベル何本ぶんに直した量は本物の本数に依らない．価値が減らない型の先例である．

**MER-DA の記録との照合**（数値は `docs/exploratory-analyses.md` の 2026-10-28 の下調べの節による．検査前で，仮のターゲットだけ）

- 疑似ラベルの損失を K 本の損失と同時に足して学ぶ使い方では，音声の配備する分類器で疑似ラベルの総重みが 2000 のとき，疑似ラベルのモデルが足す本数（中央値）は K = 0, 1, 5, 10, 25 で 29.3, 28.3, 22.8, 13.2, 3.9 だった（棚卸しの行 14 の値）．
- 同じ記録で，疑似ラベルでまず出発点を動かしてから K 本で学び直す使い方では，同じ条件で 28.3 → 27.5 とほぼ一定だった．
- したがって MER-DA の記録の中でも，「本物のラベルが増えると価値が減る」は疑似ラベルの使い方に依る．前者は Reimherr ほかの「資源を強く信じすぎると価値が減る」型に，後者は Jain ほか・Li ほかの「換算した本数が一定」の型に当たる．どちらの型にも先例がある．
- 本調査の計算：前者の音声・総重み 2000 では，25 本で 25.4 本減っており，傾きは約 −1.0 で，Reimherr ほかの正規模型の下限 −1 に当たる．言葉の配備する分類器（総重み 2000）では，同じ記録で 17.6 → −10.9（25 本で 28.5 本の減少，傾き約 −1.14）であり，正規模型の下限より急である．これは中央値どうしの差で，学習曲線の読みの誤差も入る．下限を超えたと書くには照合が要る．

**判定**：主張 2 のうち「ラベル何本ぶんで測る」は，DryBell と CLIP が同じ形で行っている．「本物のラベルが増えると減る」は，Reimherr ほか・Hernandez ほか・Zoph ほか・Zhu ほかに型がある．別モダリティの疑似ラベルのモデルについて，K を 0〜25 で振って本数を測った研究は見つからなかった．ただし MER-DA の記録自体が，価値が減るかどうかは使い方に依ることを示している（検査前）．「別モダリティの教師はラベルの代わりなので，ラベルが増えると価値が減る」という因果の言い方は，記録では支えられない．

### 1.3 主張 3：K 本と同じモダリティのラベル無しは位置を直す

**K 本の側**

- 位置の誤りが K 本で 1/(2K)，方向の誤りが q/K で減るという分け方は，Okamoto 1963 の誤判別率の展開の係数から 1 段の計算で出る（既存 ◎．`TAC 論文 先行研究の精読と穴/g4_theory_intercept_direction.md` 1 節）．原著の文には「閾値」「向き」という解釈は無い．方向の項だけを閉じた形にしたのが Raudys 1998 である（既存 ◎）．
- 少数のターゲットのラベルで低次元の調整だけを学ぶ設計は，**Maity, Dutta, Terhorst, Sun, Banerjee, Biometrika 111(1):31–50, 2024**（◎ 1 節．https://arxiv.org/abs/2111.10841 ）にある．ターゲットの対数オッズを「ソースの対数オッズ ＋ βᵀT(x)」と置き，次のように書く．原文："the unknown transfer parameter β is much easier to learn than the (potentially complex) regression function ηP, so transfer is possible even with few samples from the target domain"．T(x) = 1（切片だけを調整する）場合を，ラベルシフトの特別な場合として挙げている．
- まとめると，K 本が位置を安く直すことは，率の式と設計の前提の両方で既知である．

**ラベル無しの側**

- **Saerens, Latinne, Decaestecker, Neural Computation 14(1):21–41, 2002**（◎ 2.2 節．https://dipot.ulb.ac.be/dspace/bitstream/2013/68391/1/Decaestecker_NeuralComp02.pdf ）
  - 原文："we will assume that ... the within-class densities, p(x|ωi), do not change from the training set to the new data set (only the relative proportion of measurements observed from each class has changed)"
  - この仮定のもとで，ラベル無しのデータから事前確率を EM で推定し，分類器の出力を直す．直すのはクラスごとの対数オッズの定数，つまり位置だけである．
- **Lipton, Wang, Smola, ICML 2018（BBSE）**（○ 抄録．https://arxiv.org/abs/1802.03916 ）と **Alexandari, Kundaje, Shrikumar, ICML 2020**（○ 抄録．https://arxiv.org/abs/1901.06852 ）：同じ仮定（p(x|y) が変わらない）のもとで，ラベル無しのデータからクラスの割合を推定する．
- **Batch Calibration**（Zhou ほか，ICLR 2024．○ 抄録．https://arxiv.org/abs/2309.17249 ）：ラベル無しの入力の束の平均から文脈の偏り（クラスごとの底上げ）を推定して引く．MER-DA の中心化と釣り合い直し（棚卸しの行 4）と同じ働きである．
- **Veilleux, Boudiaf, Piantanida, Ben Ayed, NeurIPS 2021**（○ 抄録．https://proceedings.nips.cc/paper_files/paper/2021/file/4d7a968bb636e25818ff2a3941db08c1-Paper.pdf ）：ラベル無しの問い合わせ集合を使う少数ショットの手法は，クラスの割合が均等でなくなると成績が大きく落ち，ラベル無しを使わない手法を下回ることもある．ラベル無しから得ていた得の多くが，「クラスの割合は均等」という位置の情報に依っていたことを示す結果である．

**ラベル無しが方向を直さないことについて**

- 上の研究はどれも「ラベル無しは位置を直す」を仮定として置いており，方向を直せるかは測っていない．
- 理論では，ラベル無しが方向を直せる場合がある．クラスの雲がよく離れていれば，ラベル無しのデータだけで判別の境界が決まり，ラベルはどちらがどのクラスかを決めるだけで済む（Castelli & Cover 1995・1996．△）．
- 正規分布の 2 群の線形判別では，ラベル無しの観測の効率はクラス間の距離が小さいほど急に下がるとされる（O'Neill, JASA 73:821–826, 1978．Ganesalingam & McLachlan, Biometrika 65(3):658–662, 1978）．どちらも書誌だけの △ で，効率の数値は確かめられなかった．
- したがって MER-DA の「ラベル無しの 8 手法は方向を直さない」（棚卸しの行 5，照合済み）は，感情のクラスが強く重なる条件での実測として読むべきである．「ラベル無しは原理的に方向を直せない」とは書けない．

**判定**：主張 3 のうち K 本の側は既知である．ラベル無しの側は，「位置を直す」が前提として既知であり，「方向を直さない」を多数の手法で測った例は見つからなかった．ただし理論ではクラスがよく離れていれば方向も直せるので，MER-DA の結果は感情の重なりの大きさと合わせて書く必要がある．

### 1.4 主張 4：別モダリティの疑似ラベルは方向を直す

理論の 3 つの部品を組むと，この主張は予言として出る．

- **部品 1（Blum & Mitchell 1998，◎）**：条件つき独立なら，別ビューの予測はクラス条件つきの雑音ラベルと同じである（1.1 節）．
- **部品 2（Natarajan ほか 2013 と Menon ほか 2015，どちらも ◎）**：クラス条件つきの雑音のもとでは，雑音の入ったクラス確率は本物のクラス確率の単調な変換になる．したがって，分類器はそのままで閾値だけを動かせばよい．
  - Menon, van Rooyen, Ong, Williamson, "Learning from Corrupted Binary Labels via Class-Probability Estimation", ICML 2015（http://proceedings.mlr.press/v37/menon15.pdf ）．4 節の原文："It is evident that ηcorr is a strictly monotone increasing transform of η ... this is equivalent to a corrupted classifier thresh(ηcorr, T(α, β, π, πcorr, t)), where the function T ... tells us how to modify the threshold t on corrupted data"（系 6）．抄録の原文："optimising balanced error (BER) as-is on corrupted data equivalently optimises BER on clean data"．5.3 節では，雑音の大きさに依らない閾値を持つ指標は BER だけであることを示す．BER は 2 クラスでは 1 − UAR なので，UAR を主指標にする MER-DA に直接つながる．
  - Natarajan, Dhillon, Ravikumar, Tewari, "Learning with Noisy Labels", NIPS 2013（https://proceedings.neurips.cc/paper_files/paper/2013/file/3871bd64012152bfb53fdf04b401193f-Paper.pdf ）の定理 9：重み α* = (1 − ρ₊₁ + ρ₋₁)/2 を付けた 0-1 損失では，雑音のもとでの危険度が，きれいなデータでの危険度の正の定数倍と定数の和になる．雑音の影響は閾値（重み）の 1 か所に集まる．
- **部品 3（Blum & Mansour, "Efficient Co-Training of Linear Separators under Weak Dependence", COLT 2017, PMLR 65:1–17．◎ 抄録と 1 節．https://par.nsf.gov/servlets/purl/10057811 ）**
  - 抄録の原文："Our algorithm learns from purely unlabeled data, except for a single labeled example to break symmetry of the two classes, and works for any data distribution having an inverse-polynomial margin and with center of mass at the origin."
  - 1 節の原文："this assumption allows one to convert a weak-predictor over one view into a labeler corrupted by random classification noise for the other view"
  - 切片の無い線形分類器を扱い，データの重心が原点にある（中心化してある）と仮定する．2 つのビューのラベル無しデータで向きを決め，ラベルは 1 本だけ使って，2 つのクラスのどちらがどちらかを決める．位置は中心化の仮定で消している．
  - MER-DA の分業（中心化が位置を直し，別ビューが方向を運び，K 本が残りの位置を直す）を，条件つき独立より弱い依存のもとで定理にした形として最も近い．

補足：線形判別での雑音と向き．

- 2 クラスで雑音が一様なら，Fisher の判別の向きは変わらないことが知られている．Manwani & Sastry（IEEE Trans. Cybernetics 43(3), 2013）の結果として，Liu ほか（https://arxiv.org/abs/2201.10242 ）の関連研究の節が次のように要約している．原文："Fisher linear discriminant and linear least squares under squared error loss were noise tolerant under uniform noise, but not under non-uniform noise"（○）．
- Lachenbruch（Technometrics 8:657–662, 1966）は，学習標本が誤ってクラス分けされている場合の線形判別を扱う（△ 書誌だけ）．
- 本調査の計算：2 クラス・共通の共分散・クラス条件つきの雑音を仮定する．雑音の入ったクラスの平均の差は (1 − α − β) 倍に縮むだけで向きは変わらない．プールした共分散は平均の差の向きにだけ膨らむ．そのため Sherman–Morrison の公式から，判別の向き Σ⁻¹Δμ は変わらない．閾値は雑音の非対称さに応じて動く．つまり「別ビューの雑音ラベルは方向を保ち，位置をずらす」は，2 クラスの線形判別なら 1 段の計算で出る．
- 4 クラスでは，雑音の遷移行列が感情の対ごとの平均の差を混ぜるので，対ごとの向きは一般には保たれない（本調査の推論）．

**判定**：主張 4 は，共同学習と雑音ラベルの理論を組み合わせると予言として出る．「別ビューの疑似ラベルは向きを運び，閾値だけがずれ，その閾値は少数のラベルか中心化で直す」という分業は，Blum & Mansour 2017 が中心化の仮定つきで定理として持っている．一方で，実データで分類器の誤りを位置と方向に分け，K 本・同じモダリティのラベル無し・別モダリティの疑似ラベルの 3 つの情報源がそれぞれどちらを直すかを測った研究は見つからなかった．

---

## 2. 最も近い 5 本との比較

| | 設定（データ，ラベルの量，モダリティ，学習器） | 測る量 | 主張 | MER-DA と同じ点 | 違う点 |
|---|---|---|---|---|---|
| **Blum & Mansour 2017**（COLT．◎） | 理論．2 つのビュー，ほぼ全部ラベル無しでラベルは 1 本，切片の無い線形分離器，重心は原点 | PAC の誤り率と計算量 | 弱い依存のもとで，2 ビューのラベル無しデータから向きを多項式時間で学べる．ラベルは 1 本で 2 クラスの入れ替わりを決めるだけ | 別ビューが向きを運び，ラベルは向きにほとんど使わない．中心化で位置を消してから向きを扱う | 実データが無い．切片の無い模型なので位置の誤りを数えない．ドメインのずれが無い．学び直しの出発点が無い．価値をラベルの本数に換算しない |
| **Menon ほか 2015**（ICML．◎）と Natarajan ほか 2013（NIPS．◎） | 理論と小さな実験．2 クラス，雑音の入ったラベルだけ，クラス確率の推定 | BER，AUC，誤分類率と閾値 | 雑音の入ったクラス確率は本物の単調な変換で，閾値だけを直せばよい．BER は雑音を直さずに最適化できる | 雑音ラベル（疑似ラベル）は順序＝向きを運び，閾値＝位置がずれる．UAR（BER）が特別な指標であること | 雑音の出どころ（別モダリティ）を扱わない．4 クラスではない．少数のきれいなラベルとの組み合わせ方と K 依存を扱わない．実データで位置・方向の誤りを測らない |
| **Snorkel DryBell 2019**（SIGMOD．◎） | Google の 3 つの分類課題．配備時に使えない資源のラベル付け関数で弱いラベルを作り，配備できる特徴の分類器（ロジスティック回帰と深層網）を学ぶ | 相対 F1．手で付けたラベルの学習曲線での換算本数 | 弱いラベルは手で付けたラベル 8 万本（話題）・1.2 万本（商品）ぶん．配備時に使えない資源を足すと F1 が平均 52% 上がる | 配備する分類器が見ない情報を弱いラベルに変え，価値を本物のラベルの本数で読む．同じ特徴だけの弱いラベルと比べる | 手で付けたラベルの本数を振って価値の変化を測らない．ドメインのずれと出発点が無い．位置と方向に分けない．感情認識でもマルチモーダルでもない |
| **Reimherr, Meng, Nicolae 2021**（JRSS-B．既存 ◎） | 統計．ベイズの事前分布と尤度．正規模型と一般の模型 | 事前分布の標本数 M(k)（同じ誤差に届くのに要る追加の標本数）を本物の標本数 k の関数として測る | M(k) が減ることは事前分布と尤度が食い違っている印．傾きの下限は −1 | 資源の価値を本物の本数で測り，本物が増えると減ることを測る | 資源は事前分布で，別モダリティの疑似ラベルではない．物差しは母数の 2 乗誤差で，UAR ではない．位置と方向に分けない |
| **de Sa 1993**（NIPS．◎） | 母音のフォルマント（Peterson–Barney）．2 つの「モダリティ」に同じデータを割り当てた人工の設定．ラベル無し．区分線形の分類器 | 正解率 | モダリティ間の不一致を最小にすると，誤分類の最小化の近似になる．紛らわしいクラスがモダリティごとに違うとき良く働く | 別モダリティはラベルの代わり．紛らわしい対がモダリティで違うことが得の源 | ラベル付きのターゲット・ソース・ずれが無い．位置と方向に分けない．本数に換算しない．実際の異なるモダリティではない |

次点は次の 4 つである．

- Saerens ほか 2002：ラベル無しは事前確率＝位置を直す，という前提を置く．
- Maity ほか 2024：少数のラベルで低次元の調整だけを学ぶ．
- Zoph ほか 2020：固定した資源の価値は減り，ラベルで学び直した教師の価値は減らない．
- Okamoto 1963・Raudys 1998：K 本で減る位置の誤りと方向の誤りの率．

---

## 3. 上澄みの判定

MER-DA の記録のうち，C の範囲の先行研究を超えるものを強い順に並べる．期待する結論に合わせず，超えないものも書く．

### 3.1 超える：3 つの情報源を同じ分け方で実測した分業（照合済みの記録がある）

記録の内容は次のとおりである．

- 行 1（照合済み）：K 本は位置を安く直す．K=5 で，方向を固定して切片だけを K 本で直す手の上がり幅は，ソースと K 本を混ぜて学び直す手の 0.91 倍（音声），0.78 倍（顔）である．
- 行 2（照合済み）：K 本は方向を遅く直す．2 感情の離れ方の当てはめの q の中央値は，音声 16.1，顔 1.86，融合 6.82，言葉 47.5〜51.5 である．
- 行 5（照合済み）：同じモダリティのラベル無しの 8 手法では，210 単位のうち方向の誤りを減らした単位が 7（偶然の見込み 5.25），増やした単位が 79 である．
- 行 8（照合済み）：別モダリティを含むモデルの疑似ラベルでは，81 単位のうち方向の誤りを減らした単位が 38，増やした単位が 1（偶然の見込み 2.02）である．

先行研究との差：Blum & Mansour 2017，Menon ほか 2015，Saerens ほか 2002 は，この分業を仮定か定理として持つが，実データで誤りを位置と方向に分けて測ってはいない．Okamoto と Raudys は K 本の率だけを扱い，別の情報源を扱わない．3 つの情報源を同じ分け方で測り，理論が予言する分業がコーパス間の実データで起きることを示した研究は見つからなかった．

書き方：「新しい現象を発見した」とは書かない．「共同学習と雑音ラベルの理論の予言を，その仮定が保証されない実データで，同じ分け方で確かめた」と書く．

### 3.2 超える可能性がある：理論の仮定が崩れた条件でも方向が直る（未検証）

- 理論が方向を保証するのは，別ビューの予測が配備する分類器の入力と条件つき独立のとき（Blum & Mitchell）か，弱い依存のとき（Blum & Mansour）である．
- MER-DA の疑似ラベルのモデルは，多くの場合，配備する分類器自身のモダリティを含む融合である（行 9）．モダリティ間の誤りの相関は 0.77〜1.00（行 12．検査前）で，独立から遠い．
- それでも方向が直る（行 8）なら，理論の仮定より弱い条件で分業が成り立つことになる．これは先行研究に無い．
- ただし今の記録では，「依存が強いほど方向の直りが小さいか」を測っていない．行 9 の「配備する分類器のモダリティを除いた融合は作っていない」も同じ穴である．この 2 つを測るまでは，上澄みとして書けない．行 12 の相関がどのモデルどうしの誤りの相関かも，本文に書く前に記録で確かめる必要がある．

### 3.3 超える可能性がある：感情の対ごとの方向の補い合い（未計算）

- de Sa 1993 は，「紛らわしいクラスがモダリティごとに違うとき，別のモダリティのラベルが効く」ことを，人工の 2 モダリティの母音で示した．
- 実際の音声・顔・言葉の 3 モダリティで，どの感情の対の方向をどのモダリティが直すかを測った研究は見つからなかった．
- MER-DA の記録にあるのは，ソースだけの分類器の方向の誤りが対ごとに違うこと（行 17．検査前）までである．疑似ラベルがどの対の方向を直したかは未計算である（行 17・22．対ごとの分解の設計は検査の直し待ち）．
- 計算が済めば，de Sa の予言を実データで確かめた形になる．その場合は de Sa を必ず先例として引く．

（本体のセッションの注：対ごとの分解は 2026-11-03 に計算済みで，`docs/exploratory-analyses.md` の E83 6 節に記録がある．棚卸しの行 17・22 の「未計算」は更新前の記述である．）

### 3.4 超えない：ラベル何本ぶんと，その K 依存

- 測り方は DryBell と CLIP に，価値が減る型は Reimherr ほか・Hernandez ほか・Zoph ほか・Zhu ほかにある．
- MER-DA の記録では，価値が減るのは「疑似ラベルの損失を K 本と同時に足す」使い方のときである．「疑似ラベルで出発点を動かしてから K 本で学び直す」使い方では，ほぼ一定だった（2026-10-28 の下調べ．検査前，仮のターゲットだけ）．学び直して掛けたモデルでは減らないとする数え上げもある（行 14．未照合）．
- したがって「別モダリティの教師の価値は本物のラベルが増えると減る」は，MER-DA の記録の中でも一般には成り立たない．
- 「同じ資源でも，2 つの使い方で価値の曲線の型が変わる」は，前回の調査（`新規性と研究価値_2026-10-29/G_第2回.md`）で「比べた研究は見つからなかった」と判定済みである．ただし値は検査前で，新しさとしては小さい．

### 3.5 記録にも先行研究にも無い：ラベル何本ぶんを位置の分と方向の分に分ける（狙うべき差分）

- 主張 2（ラベル何本ぶん）と主張 3・4（位置と方向）は，今の記録では別々の量で測られている．疑似ラベルのモデルの換算本数を，位置の誤りの減りで読んだ本数と方向の誤りの減りで読んだ本数に分けた記録は無い．先行研究にも見つからなかった．
- 理論からの見込み（本調査の推論）：2 クラスで条件つき独立なら，雑音率 η の疑似ラベル N 本は，方向については本物のラベル約 N(1 − 2η)² 本ぶんの情報を持つ．平均の差が 1 − 2η 倍に縮むためである．この本数は K に依らない．位置については，疑似ラベルが持つ閾値のずれが残るので，価値は 0 か負になる．使い方によって全体の換算本数が K とともに減るのは，この位置の分の害が，K 本が持つ位置の情報と打ち消し合うためと読める．
- これを測れば，主張 2 と主張 4 を 1 つの量でつなげる．「別モダリティの教師が代わりをするのは方向についてのラベルである」と本数で言えるようになる．
- 保存済みの出力（疑似ラベルを使った配備する分類器の位置と方向の誤り，ターゲットだけの学習曲線）から出せるかは，`results/` を開いていないので確かめていない．出せるかの確認と設計は本体のセッションに任せる．

---

## 4. 論文での書き方の注意

1. **「ラベルの代わり」を新しい見方として書かない．**de Sa 1993 の "substitute for the labels"，Blum & Mitchell 1998 の定理 1，Lopez-Paz ほか 2016 の特権情報の見方，Snorkel DryBell の「配備時に使えない資源から作る弱いラベル」を引く．そのうえで，MER-DA はそれを「どの成分（方向）の代わりか」まで測った，と書く．
2. **「ラベル何本ぶんで測った」を新しい測り方として書かない．**CLIP と DryBell が同じ測り方をしている．
3. **「教師の価値は K とともに減る」を一般の性質として書かない．**記録では，使い方と疑似ラベルのモデルの種類によって変わる（検査前と未照合）．書くなら「疑似ラベルの損失を K 本と同時に足す使い方では」と条件を付け，Reimherr ほか 2021 の型（資源を強く信じすぎると価値が減る）として引く．
4. **「ラベル無しは方向を直せない」と一般化しない．**理論では，クラスがよく離れていれば直せる（Castelli & Cover，O'Neill）．「感情のクラスが強く重なる凍結特徴の上では，8 手法とも方向を直さなかった」と条件つきで書き，対ごとの 2 感情の離れ方の値を並べる．
5. **主張 4 は理論の予言の確かめとして書く．**先に Menon ほか 2015（閾値だけがずれる）と Blum & Mansour 2017（2 ビューで向きを決め，ラベル 1 本でクラスの入れ替わりを決める．中心化の仮定つき）を置く．そのうえで，MER-DA はその仮定（条件つき独立）が保証されない実データで分業を測った，と書く．「仮定が崩れても成り立つ」と言うのは，3.2 節の測定を済ませてからにする．
6. **UAR と BER の関係を使う．**Menon ほか 2015 は，雑音の入ったラベルで BER（2 クラスでは 1 − UAR）を最適化すると，きれいなラベルで最適化したのと同じになることを示した．UAR を主指標にする理論の裏づけとして引ける．ただし 2 クラスの結果であり，4 クラスの UAR にそのまま広がるとは書けない．
7. **対ごとの補い合いは de Sa 1993 を先例に引く．**「別のモダリティが特定の対を解消するかを調べた研究は無い」とは書かない．「実際の 3 モダリティの感情データで，対ごとに方向を測った研究は見つからなかった」と書く．
8. 位置と方向の分け方そのものは，Okamoto 1963 の展開の係数から出る（`docs/related-work.md` 1.14 節 (b)）．「Okamoto が分けた」ではなく「係数から分けられる」と書く．

---

## 5. 見つからなかったことと調査の限界

**今回の検索の範囲で見つからなかったこと**

- 別モダリティの疑似ラベルのモデルの価値を，ターゲットのラベルの本数 K を振って，ラベル何本ぶんで測った研究．
- 分類器の誤りを位置と方向に分け，K 本・同じモダリティのラベル無し・別モダリティの疑似ラベルの 3 つの情報源がそれぞれどちらを直すかを，実データで測った研究．
- 疑似ラベルや教師の換算本数を，位置の分と方向の分に分けた研究．
- 共同学習の理論の仮定（条件つき独立）が崩れた度合いと，方向の直りの大きさの関係を，実データで測った研究．
- ラベル無しの DA の手法が方向の誤りを減らさないことを，多数の手法で測った研究．

**調査の限界**

- O'Neill 1978，Ganesalingam & McLachlan 1978，Lachenbruch 1966，Castelli & Cover 1995・1996 は本文を読めなかった（△）．ラベル無しの観測の効率がクラス間の距離でどう変わるかの数値は確かめていない．論文で引くなら，McLachlan 1992 の教科書の該当章とあわせて原典で確かめる．
- Manwani & Sastry 2013 は，二次資料（Liu ほか 2022 の関連研究の節）でだけ確かめた．
- Blum & Mansour 2017 は抄録と 1 節だけを読んだ．定理の条件（弱い依存の強さ λ と向きの誤差の関係）は読んでいない．3.2 節の測定を設計するなら，この関係を先に読む．
- Natarajan ほか 2013 と Menon ほか 2015 は 2 クラスの結果である．多クラスのクラス条件つきの雑音で向きが保たれる条件は調べていない．
- Snorkel DryBell の図 5（学習曲線）の本体は読んでいない．換算の本数は本文の文だけによる．
- Zhu ほか 2023，Lipton ほか 2018，Alexandari ほか 2020，Batch Calibration，Veilleux ほか 2021 は抄録だけを読んだ．
- dblp と OpenReview は使っていない．IEEE Xplore・ICASSP・Interspeech の会議録は直接検索していない．2025〜2026 年の感情認識の会議論文に同じ測定がある可能性は残る（B の調査役の範囲と重なる）．
- 本調査の計算（2 クラスの線形判別で雑音が向きを保つこと，N(1 − 2η)² の換算，価値の減り方の傾き）は，検算のコードを書いていない．論文に使うなら，式を書き下して数値で確かめる．

---

## 付録：文献の一覧（水準つき）

| 文献 | 水準 | URL |
|---|---|---|
| de Sa, NIPS 6, pp. 112–119, 1993 | ◎ | https://papers.neurips.cc/paper_files/paper/1993/file/e0ec453e28e061cc58ac43f91dc2f3f0-Paper.pdf |
| Blum & Mitchell, COLT 1998 | ◎ | https://www.cs.cmu.edu/~avrim/Papers/cotrain.pdf |
| Blum & Mansour, COLT 2017, PMLR 65:1–17 | ◎（抄録と 1 節） | https://par.nsf.gov/servlets/purl/10057811 |
| Natarajan, Dhillon, Ravikumar, Tewari, NIPS 2013 | ◎（定理 9） | https://proceedings.neurips.cc/paper_files/paper/2013/file/3871bd64012152bfb53fdf04b401193f-Paper.pdf |
| Menon, van Rooyen, Ong, Williamson, ICML 2015 | ◎（抄録，4 節，5.3 節） | http://proceedings.mlr.press/v37/menon15.pdf |
| Bach ほか, Snorkel DryBell, SIGMOD 2019 | ◎（抄録，1 節，6 節） | https://arxiv.org/abs/1812.00417 |
| Saerens, Latinne, Decaestecker, Neural Computation 14(1):21–41, 2002 | ◎（2.2 節） | https://dipot.ulb.ac.be/dspace/bitstream/2013/68391/1/Decaestecker_NeuralComp02.pdf |
| Maity ほか, Biometrika 111(1):31–50, 2024 | ◎（1 節） | https://arxiv.org/abs/2111.10841 |
| Hanneke & Kpotufe, NeurIPS 2019 | ◎（1 節） | https://arxiv.org/abs/2002.04747 |
| Zoph ほか, NeurIPS 2020 | ◎（抄録，4.1・4.2 節） | https://arxiv.org/abs/2006.06882 |
| Zhu ほか, ACL 2023, pp. 14229–14253 | ○ | https://aclanthology.org/2023.acl-long.796/ |
| Lipton, Wang, Smola, ICML 2018 | ○ | https://arxiv.org/abs/1802.03916 |
| Alexandari, Kundaje, Shrikumar, ICML 2020 | ○ | https://arxiv.org/abs/1901.06852 |
| Zhou ほか, Batch Calibration, ICLR 2024 | ○ | https://arxiv.org/abs/2309.17249 |
| Veilleux ほか, NeurIPS 2021, pp. 9290–9302 | ○ | https://proceedings.nips.cc/paper_files/paper/2021/file/4d7a968bb636e25818ff2a3941db08c1-Paper.pdf |
| Manwani & Sastry, IEEE Trans. Cybernetics 43(3), 2013（Liu ほかの要約で確認） | ○ | https://arxiv.org/abs/2201.10242 |
| Lachenbruch, Technometrics 8:657–662, 1966 | △ 書誌だけ | — |
| O'Neill, JASA 73(364):821–826, 1978．Ganesalingam & McLachlan, Biometrika 65(3):658–662, 1978 | △ 書誌だけ | — |
| Reimherr, Meng, Nicolae, JRSS-B 83(3):413–437, 2021 | 既存 ◎ | https://arxiv.org/abs/1406.5958 |
| Hernandez ほか, Scaling Laws for Transfer, 2021 | 既存 ◎ | https://arxiv.org/abs/2102.01293 |
| Radford ほか（CLIP），ICML 2021 | 既存 ◎ | https://arxiv.org/abs/2103.00020 |
| Jain, Montanari, Sasoglu, NeurIPS 2024 | 既存 ◎ | https://arxiv.org/abs/2402.04376 |
| Li, Han, Yang, Tong, ICML 2026, PMLR 306:67268–67304 | 既存 ◎ | `docs/related-work.md` 1.16 節 |
| Okamoto 1963（Shutoh の再掲で読んだ），Raudys 1998 | 既存 ◎ | `research_notes/TAC 論文 先行研究の精読と穴/g4_theory_intercept_direction.md` |
| Lopez-Paz ほか, ICLR 2016 | 既存 ◎ | https://arxiv.org/abs/1511.03643 |
| Castelli & Cover 1995・1996 | △ | `research_notes/ラベル換算の価値で重みを決める先例/report.md` 付録 A |

**検索した主な語**：unlabeled data can only correct the bias not the weight direction／cross-modal pseudo-labels worth labeled examples emotion teacher value decreases／co-training linear classifier other view weight direction threshold class-conditional noise／LDA label noise discriminant direction unchanged／Snorkel DryBell equivalent hand-labeled／LLM labels exchange rate human labels／Lachenbruch 1966／O'Neill 1978／Ganesalingam McLachlan 1978／Maity linear adjustment posterior drift／Hanneke Kpotufe value of target data／de Sa learning classification with unlabeled data／Saerens a priori probabilities／Veilleux transductive few-shot／Batch Calibration．
