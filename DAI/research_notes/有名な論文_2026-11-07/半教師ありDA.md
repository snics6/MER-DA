# 半教師あり DA の有名な論文の参照ノート（2026-11-07）

このノートは，半教師あり DA（semi-supervised domain adaptation，SSDA．ソースのラベル付きデータ，ターゲットのごく少数のラベル，ターゲットのラベル無しデータの 3 つを使って，ターゲットで成績のよい分類器を作る問題）の有名な論文を，その土台になった教師なし DA と半教師あり学習の古典，自己学習の理論とあわせて原典で調べ直した記録である．後のセッションが，論文を引く前の確認や，手法を考えるときの参照に使う．蒸留・MER・メタ学習は別の担当のノートに書かれる．研究の `results/`，`data/`，`logs/` は開いていない．

---

## 0. 調べ方と確認の水準の記号

**調べ方．**arXiv，学会の論文集（PMLR，NeurIPS の論文集，CVF，ACL Anthology），著者の公開 PDF から本文の PDF を取り，scratchpad の新しいディレクトリ（`/tmp/claude-1000/-home-snics-MER-DA/b1f27254-32c7-45f0-ac5a-dee1c5e3ec8a/scratchpad/survey_ssda/`）に置いて `pdftotext` で文字にして読んだ．各論文の抄録，手法の節，主な結果の表，結論の節を読んだ．書誌（題・著者・会場・年・頁・DOI）は Crossref の API，PMLR の巻の目次，NeurIPS の論文集の Metadata.json と BibTeX，JMLR の頁，arXiv の API で確かめた．数値は PDF の文字から読んだもので，表や節の場所を書いた．図だけにある数値は書いていない．

**すでにノートがある論文．**次の 3 本の既存のノートで精読済みの論文は，今回は書誌だけを確かめ直し，要点を縮めて「詳しくは」で参照先を示した．
- `research_notes/半教師あり DA の先例/ssda.md`（MME，APE，CDAC，DECOTA，Donahue 2013，FADA，CCSA，AdaMatch の評価設定）
- `docs/related-work.md` の 1.15 節と 2.6〜2.11 節（Ben-David 2010，Daumé 2007，Blitzer 2007 など）
- `DAI_paper/参照論文/要点_問い3_同じ設定の研究.md`（MME の要点）

**確認の水準の記号．**
- ◎＝本文（PDF）を読んだ．一部の節だけを読んだときは「◎（読んだ節）」と書く
- ○＝抄録か書誌の頁で確かめた
- △＝書誌のみ

**用語（このノートの中での意味）．**
- ソース：ラベル付きのデータがたくさんあるドメイン（データの出どころ）．MER-DA ではターゲット以外の 3 コーパス
- ターゲット：成績を上げたいドメイン．MER-DA では leave-one-corpus-out で外した 1 コーパス
- UDA（unsupervised domain adaptation，教師なし DA）：ターゲットのラベルを 1 本も使わない DA
- SSL（semi-supervised learning，半教師あり学習）：同じ分布の少数のラベル付きデータと多数のラベル無しデータで学ぶ問題
- 教師あり DA（supervised DA）：ソースのラベルとターゲットの少数のラベルだけを使い，ターゲットのラベル無しデータを使わない DA
- 疑似ラベル（pseudo-label）：ラベル無しの標本に，モデルの予測をラベルとして付けたもの
- 自己学習（self-training）：疑似ラベルを付けた標本を学習データに足して学び直す手続き
- S+T：ソースのラベルとターゲットのラベルを合わせて普通に学ぶ比較相手（SSDA の論文の標準の比較相手）

**載せた論文．**論文は 39 本，2 節の見出しは 34 個である（co-training と tri-training，DANN の 2 本，CORAL の 2 本，FADA と CCSA，総説の 2 本の 5 組を 1 つずつの見出しにまとめた）．依頼の目安（20〜30 本）より多い．有名で MER-DA の設定に関わる論文を削らなかったためである．外した論文は 4 節の末尾に書いた．

---

## 1. 分野の流れ

**1998〜2005 年：ラベル無しデータはいつ役に立つか．**半教師あり学習の古典は，ラベル無しデータを足すと成績が上がる条件を問うた．Blum & Mitchell（1998）の co-training は，1 つの標本が 2 つの「見方（view．例えばウェブ頁の本文の語と，その頁を指すリンクの語）」を持ち，どちらの見方でも分類でき，ラベルを条件にすると 2 つの見方が独立であるときに，一方の見方の分類器の予測で他方の学習データを増やせることを示した．Nigam ほか（2000）は単純ベイズの分類器と EM（ラベル無しの標本のラベルを確率で推定し，その推定で模型を学び直す反復）で文書分類を改善したが，生成の模型の仮定が崩れるとラベル無しデータで成績が下がることを示し，ラベル無しデータの重み λ を足した．Grandvalet & Bengio（2005）は，ラベル無しの標本での予測のエントロピー（予測の確率がどれだけ散らばっているか．すべてのクラスに等しい確率なら最大）を小さくする罰則を判別の模型に足す方法を出した．Zhou & Li（2005）の tri-training は，見方の分割を要らなくし，3 つの分類器のうち 2 つの予測が一致した標本を残りの 1 つの学習データに足した．この時期の論文は，ラベル無しデータが役に立つには，クラスの境界がデータの密度の低い所を通るという仮定（Grandvalet & Bengio の語で cluster assumption）や，見方の条件付き独立のような，入力とラベルの関係についての仮定が要ると書いた．

**2007〜2013 年：ドメインのずれの理論とベンチマーク．**Daumé（2007）は，ソースのラベルとターゲットの少数のラベルがあるとき，特徴をソース用・ターゲット用・共通の 3 つに複製するだけの方法が多くの課題で最良になることと，ソースの重みを事前分布にして学ぶ方法（PRIOR）が次点になることを示した．Ben-David ほか（2010）は，ターゲットの誤差をソースの誤差，ドメインのずれ（HΔH-divergence），両方のドメインで同時によい分類器の誤差 λ の和で抑える上界と，ソースとターゲットの誤差を α で混ぜて学ぶときの上界を示し，これが後の DA の手法の動機になった．Saenko ほか（2010）は画像の物体認識でドメインのずれを調べ，Office データセット（amazon・dslr・webcam の 3 ドメイン，31 クラス）と，ターゲットのラベルをクラスあたり 3 本にする評価の手順を作った．Donahue ほか（2013）は，固定した特徴の上の線形の分類器にソースの超平面への罰則とラベル無しの標本の対の制約をかける SSDA を示した．

**2013〜2015 年：深い特徴とドメインの不変な表現．**深い網の特徴が転移しやすいことが分かると，ソースとターゲットの特徴の分布を近づけて学ぶ UDA が主流になった．Long ほか（2015）の DAN は複数の層の特徴の分布の差を MMD（2 つの分布の平均を核関数で写した点どうしの距離）で測って小さくし，Ganin & Lempitsky（2015）と Ganin ほか（2016）の DANN は，特徴からドメインを当てる判別器を置き，勾配反転層（逆伝播のときだけ勾配の符号を反転する層）で特徴抽出器が判別器をだますように学んだ．DANN の著者は，この手法を，Ben-David の上界のドメインのずれの項をドメインの判別器の誤差で近似して小さくする手法と位置づけた．同じ頃，Lee（2013）は深い網で疑似ラベルを使う簡単な方法を出し，それがエントロピー最小化と同じ働きをすると述べた．

**2016〜2018 年：分布合わせの変種と，自己学習の見直し．**CORAL（Sun ほか 2016）は特徴の共分散だけをそろえ，固定した特徴の上でも使えることを示した．ADDA（Tzeng ほか 2017）はソースとターゲットで別々の特徴抽出器を使い，GAN の損失で合わせた．CDAN（Long ほか 2018）は特徴だけでなく予測との外積で条件づけた判別器を使い，クラスごとに分かれた分布を合わせようとした．MCD（Saito ほか 2018）は，ドメインの判別器の代わりに 2 つの分類器のターゲットでの予測の食い違いを使った．CBST（Zou ほか 2018）は，疑似ラベルで学ぶ自己学習がやさしいクラスに偏る問題を，クラスごとの閾値で直した．教師あり DA では FADA と CCSA（Motiian ほか 2017）がクラスあたり 1〜7 本のターゲットのラベルで深い埋め込みを学んだ．半教師あり学習では Mean Teacher（Tarvainen & Valpola 2017）が，一貫性正則化（入力やモデルに小さな乱れを加えても予測が変わらないようにする罰則）の目標を重みの指数移動平均で作り，Oliver ほか（2018）は，ラベル無しの標本に別のクラスが混ざると成績が下がることと，現実的な大きさの検証用データでは手法の差を見分けられないことを示した．

**2019〜2020 年：SSDA の標準の評価設定と，強い半教師あり学習．**MME（Saito ほか 2019）は，DomainNet の 126 クラス・4 ドメイン・7 組で，ターゲットのラベルをクラスあたり 1 本と 3 本にする SSDA の評価設定を定め，UDA の手法にターゲットのラベルを足すだけでは S+T を上回らない場合があることを示した．APE（Kim & Kim 2020）は，少数のラベルの近くの標本だけがソースに揃い残りが揃わない「ドメイン内のずれ」を指摘した．半教師あり学習では MixMatch（Berthelot ほか 2019）と FixMatch（Sohn ほか 2020）が，データ拡張と疑似ラベルの組み合わせでクラスあたり 4 本前後のラベルでも高い正解率を出した．SHOT（Liang ほか 2020）はソースのデータを使わず，ソースの分類器を固定してターゲットの特徴抽出器だけを学ぶ設定を作った．理論では Kumar ほか（2020）と Chen ほか（2020）が，正則化した線形の分類器の自己学習がずれのもとで成功する条件を証明した．

**2021〜2022 年：FixMatch 型の SSDA と，自己学習の理論の一般化．**CDAC（Li ほか 2021），DECOTA（Yang ほか 2021），CLDA（Singh 2021）は，FixMatch 型の疑似ラベルや対照学習（同じクラスの表現を近づけ，違うクラスの表現を遠ざける損失）を SSDA に持ち込み，DomainNet の 3 本の設定（ResNet34）で平均の正解率 75.3〜76.0% に達した（CDAC 76.0，DECOTA 75.6，CLDA 75.3）．AdaMatch（Berthelot ほか 2022）は UDA・SSL・SSDA を 1 つの手法で扱い，疑似ラベルの分布をソースのラベルの分布にそろえる補正を入れた．Tent（Wang ほか 2021）は，学習の後にテストのデータだけでエントロピーを小さくする「テスト時の適応」を示した．理論では Wei ほか（2021）が expansion（各クラスの中の小さな部分集合のまわりの近傍が，その集合よりずっと大きな確率を持つという仮定）のもとで，疑似ラベルの誤りを学習が直せることを示し，Cai ほか（2021）がそれをドメインのずれ（クラスの中の下位の集団の割合が変わるずれ）に広げた．

**2023 年以降．**SLA（Yu & Lin 2023）は，ソースのラベルをターゲットから見た雑音の混じったラベルとみなして書き直す見方を出し，既存の SSDA の手法に足して成績を上げた．基盤モデル（大量のデータで事前学習した大きな網）の凍結した特徴を使う SSDA で，広く引かれる論文は今回の検索では見つからなかった（4 節の末尾）．

---

## 2. 論文ごとの要点

**A. DA の土台**

### Ben-David, Blitzer, Crammer, Kulesza, Pereira, Vaughan 2010（Machine Learning 79(1–2):151–175．arXiv なし）A theory of learning from different domains

**問い．**ソースで学んだ分類器のターゲットでの誤差を，何で抑えられるか．ソースの多くのラベルとターゲットの少しのラベルを，どの割合で混ぜて学べばよいか．

**方法．**ドメインのずれを，仮説の対が食い違う確率のソースとターゲットでの差の最大値（HΔH-divergence）で測る．
```
定理 2：ε_T(h) ≤ ε_S(h) + ½ d̂_HΔH(U_S, U_T) + 4 √((2d log(2m) + log(2/δ)) / m) + λ
         λ = ε_S(h*) + ε_T(h*)（両ドメインの誤差の和が最小の仮説 h* の，その和）
定理 3：ε̂_α(h) = α ε̂_T(h) + (1 − α) ε̂_S(h) を最小にする ĥ について
         ε_T(ĥ) ≤ ε_T(h*_T) + 4 √(α²/β + (1−α)²/(1−β)) √((2d log(2(m+1)) + 2 log(8/δ)) / m)
                  + 2(1 − α) ( ½ d̂_HΔH(U_S, U_T) + 4 √((2d log(2m′) + log(8/δ)) / m′) + λ )
  ε_S, ε_T：ソース・ターゲットでの誤差．d：仮説の集合の VC 次元．β：学習データのうちターゲットの割合
  m：ラベル付きの標本数．m′：各ドメインのラベル無しの標本数（pdftotext の文字では ′ が落ちて m に見える箇所があるので，
  引くときは原典の式で確かめる）
```

**分かったこと．**定理 3 の上界を最小にする α は，ターゲットが十分多ければ 1（ソースを使わない）になり，ソースが少なすぎてもソースを無視する値になる（2 つの相転移）．α=0 なら定理 2 と同じ形，α=1 ならターゲットだけの普通の上界になる（定理 3 の直後の本文）．2 つの相転移は 6 節の議論である（`docs/related-work.md` 2.7 節の記録）．

**前提と限界．**設定は 2 値分類で，ずれの量は VC 次元で抑えている．「We do not yet know whether our divergence measure admits tighter data-dependent bounds」（結論）．λ はラベルが要るので推定しにくい．こちらの読み：多クラスと，λ を少数のラベルでどう扱うかは扱っていない．

**MER-DA との接点**（推測．検定していない）．MER-DA のソースとターゲットの K 本の合併は定理 3 の α 重みの学習にあたる．提案のソースの分類器の平均への L2 の引き戻しは，定理 3 の枠の外（仮説の空間を引き戻しで狭める形）にある．

**確認の水準と出典．**◎（リポジトリの `A theory of learning from different domains.pdf` の定理 2・3 と結論を読み直した）．詳しくは `docs/related-work.md` 2.7 節．https://doi.org/10.1007/s10994-009-5152-4

### Daumé III 2007（ACL 2007, pp. 256–263．arXiv 0907.1815）Frustratingly Easy Domain Adaptation

**問い．**ソースのラベルが多く，ターゲットのラベルが少しあるとき（教師あり DA），簡単な方法でどこまで行けるか．

**方法．**特徴を 3 倍に複製し，共通の部分と各ドメインの部分の重みを同時に学ぶ（AUGMENT）．比べる相手はソースのみ（SRCONLY），ターゲットのみ（TGTONLY），合併（ALL），重み付きの合併（WEIGHT），ソースの重みを事前分布の平均にする学習（PRIOR）など．
```
Φ_s(x) = ⟨x, x, 0⟩（ソースの標本）　　Φ_t(x) = ⟨x, 0, x⟩（ターゲットの標本）
```

**分かったこと．**表 2（誤り率 %）で，多くの課題で AUGMENT が最良，PRIOR が次点である．例：PubMed は SRCONLY 12.02，TGTONLY 4.15，ALL 5.43，PRIOR 3.99，AUGMENT 3.61．ソースのみがターゲットのみより良い課題（Treebank-Chunk の br-*）では AUGMENT が最良でない（br-cm は ALL 3.14，AUGMENT 4.89）．著者は，このときドメインが似ているので特徴の空間を広げる必要が無いと解釈する．

**前提と限界．**著者の記述：系列ラベリング（固有表現・品詞・チャンキング）だけで調べた．ターゲットのラベルは数千文の規模である．こちらの読み：ずれの大きさは測っていない．

**MER-DA との接点**（推測．検定していない）．PRIOR はソースの重みへの L2 の引き戻しと同じ形で，提案の引き戻しの最も古い先例の 1 つである．AUGMENT は凍結した特徴の上の線形の分類器でそのまま作れるので，比べる相手の候補になる．

**確認の水準と出典．**◎（表 2 と 3 節を読み直した）．詳しくは `docs/related-work.md` 2.6 節．https://aclanthology.org/P07-1033.pdf，https://arxiv.org/abs/0907.1815 （arXiv の API で題を確かめた）．

### Saenko, Kulis, Fritz, Darrell 2010（ECCV 2010，LNCS，pp. 213–226．arXiv なし）Adapting Visual Category Models to New Domains

**問い．**撮影の条件（カメラ，解像度，照明，背景）が変わると物体認識はどれだけ落ちるか．ターゲットのラベルが少ないとき，ドメイン間の変換を学んで補えるか．

**方法．**ソースとターゲットの標本の対に，同じクラスなら距離を小さく，違うクラスなら大きくする制約を置き，マハラノビス距離の行列 W を学ぶ（ITML．情報理論に基づく距離学習の定式化を DA に使った）．分類は k 近傍法．
```
d_W(x, y) = (x − y)ᵀ W (x − y)
min_{W ⪰ 0} tr(W) − log det W
s.t. d_W(x_i^A, x_j^B) ≤ u（同じクラスの対），d_W(x_i^A, x_j^B) ≥ ℓ（違うクラスの対）
```

**分かったこと．**Office データセット（amazon・dslr・webcam，31 クラス）を公開した．特徴は SURF の 800 語の bag of words．同じクラスにターゲットのラベルがある設定では，ターゲットのラベルをクラスあたり 3 本，ソースを 8 本（webcam・dslr）か 20 本（amazon）にした（5 節）．表 1 の正解率で，webcam→dslr は knnAB（ソースの標本で近傍を探す）0.14，ITML(A+B) 0.18，提案（symm）0.27．amazon→webcam は knnAB 0.03，提案（asymm）0.48．ターゲットにラベルの無いクラスでも変換で上がる（表 2，webcam→dslr で 0.37 から 0.53）．

**前提と限界．**著者の記述：ラベルで両ドメインの対の制約を作るので，ターゲットに少数のラベルが要る．こちらの読み：深い特徴以前の結果で，数値の水準は今の Office の結果と比べられない．

**MER-DA との接点**（推測．検定していない）．「ターゲットのラベルをクラスあたり数本」という評価の手順の出どころであり，MER-DA の K 本の設定と同じ数え方である．DAN の表 3 などが「classic semi-supervised protocol (Saenko et al., 2010)」として引く．

**確認の水準と出典．**◎（1・3・4・5 節と表 1・2）．https://doi.org/10.1007/978-3-642-15561-1_16 ，PDF：https://people.bu.edu/bkulis/pubs/saenko_eccv_2010.pdf

**B. 半教師あり学習の古典**

### Blum & Mitchell 1998（COLT 1998, pp. 92–100．arXiv なし）Combining Labeled and Unlabeled Data with Co-Training．後続の Zhou & Li 2005（IEEE TKDE 17(11):1529–1541）Tri-Training を含む

**問い．**1 つの標本が 2 つの見方を持つとき，少数のラベルと多数のラベル無しデータで，両方の見方の分類器を良くできるか．

**方法．**見方 1 と見方 2 で別々に分類器を学び，各分類器が最も確信を持つラベル無しの標本に予測のラベルを付けて，共通の学習データに足す（表 1 の算法．実験では正例 1 本・負例 3 本を 1 回に足し，k=30 回繰り返し，候補の池 u=75）．PAC 学習の枠での解析の鍵は，ラベルを条件にした見方の独立である．2 つの見方の確率を掛けて合わせた分類器も作った．
```
条件付き独立：Pr[x₁ = x̂₁ | x₂ = x̂₂] = Pr[x₁ = x̂₁ | f₂(x₂) = f₂(x̂₂)]
2 つの見方の合わせ方（6 節）：P(c_j | x) ∝ P(c_j | x₁) P(c_j | x₂)
```
Zhou & Li の tri-training は，見方の分割を要らなくし，3 つの分類器のうち 2 つの予測が一致したラベル無しの標本を残り 1 つの学習データに足す．雑音の混じったラベルからの学習の結果（Angluin & Laird）を使い，「足した標本の誤り率 × 標本数」が前の回より減るときだけ足す条件を置く．
```
m ≥ 2 / (ε² (1 − 2η)²) · ln(2N/δ)（η：ラベルの雑音率の上限．ε：誤差．N：仮説の数）
足す条件：ẽ_t |L_t| < ẽ_{t−1} |L_{t−1}|（ẽ：他の 2 つが一致したときの誤り率の推定）
```

**分かったこと．**Blum & Mitchell の表 2（ウェブ頁 1,051 件のうち 263 件をテスト，ラベル 12 本，5 回の平均の誤り率 %）：本文の語の分類器は 12.9 から 6.2，リンクの語の分類器は 12.4 から 11.6，合わせた分類器は 11.1 から 5.0 に下がった．すべて負と答えると 22%．Zhou & Li の表 VII（同じウェブ頁の課題）：tri-training の 3 つを合わせた仮説の誤り率は，BP 網で .146 から .109（25.3% 減），J4.8 決定木で .231 から .141．co-training の本文の語の分類器は J4.8 で .152 から .172 に，BP 網で .130 から .154 に悪くなった．

**前提と限界．**Blum & Mitchell：「Our theoretical model is clearly an over-simplification of real-world target functions and distributions」，データは 1 つで「Further experiments are needed」（6・7 節）．結論で，映像と音声を 2 つの見方にする例（テレビ放送で大統領の顔と声を認識する）と，複数のセンサーの例を挙げる．Zhou & Li：「the performance of semi-supervised learning algorithms are usually not stable because the unlabeled examples may often be wrongly labeled」（5 節）．

**MER-DA との接点**（推測．検定していない）．MER-DA の音声・顔・言葉は co-training の見方にあたる．提案の「単独の分類器の積で作る疑似ラベル」は Blum & Mitchell の P(c|x) ∝ P(c|x₁)P(c|x₂) と同じ形で，この積は感情を条件にしたモダリティの独立を仮定すると正当化される（感情のコーパスでこの独立が成り立つかは測っていない）．3 つのモダリティは tri-training の「2 つが一致したら残りの学習データに足す」にそのまま当てはまる．

**確認の水準と出典．**◎（Blum & Mitchell は全文．読んだ PDF は 1 頁目の注記によれば COLT 版の拡張版．Zhou & Li は 2 節・5 節と表 VII）．https://doi.org/10.1145/279943.279962 ，PDF：https://www.cs.cmu.edu/~avrim/Papers/cotrain.pdf ／ https://doi.org/10.1109/TKDE.2005.186 ，PDF：https://www.lamda.nju.edu.cn/publication/tkde05.pdf

### Nigam, McCallum, Thrun, Mitchell 2000（Machine Learning 39(2–3):103–134．arXiv なし）Text Classification from Labeled and Unlabeled Documents using EM

**問い．**少数のラベル付きの文書に多数のラベル無しの文書を足して，文書分類を良くできるか．生成の模型の仮定が崩れるとき，どうすればよいか．

**方法．**単純ベイズの分類器を少数のラベルで学び，ラベル無しの文書のクラスの確率を推定し（E ステップ），その確率で重みづけて模型を学び直す（M ステップ）．ラベル無しの文書の寄与を λ で弱める EM-λ と，1 クラスに複数の混合成分を持たせる拡張を足した．
```
l_c(θ | D; z) = log P(θ) + Σ_{d_i ∈ D_l} Σ_j z_ij log(P(c_j|θ) P(d_i|c_j; θ))
                + λ ( Σ_{d_i ∈ D_u} Σ_j z_ij log(P(c_j|θ) P(d_i|c_j; θ)) )，0 ≤ λ ≤ 1
```
λ はラベル付きの学習データでの leave-one-out の交差検証で選ぶ（5.3.1 節）．

**分かったこと．**20 Newsgroups で，ラベル 40 本（クラスあたり 2 本）の正解率が 27% から 43% に上がった（1 節）．同じ課題で 70% に届くのに，ラベルだけなら 2,000 本，ラベル無し 10,000 本を足せば 600 本で足りた．WebKB では，ラベル 4 本で単純ベイズ 40%，EM 55% だが，ラベル 240 本では単純ベイズ 81%，EM 79% で，EM が下回った（6 節）．

**前提と限界．**著者の記述：「If the intended target concept and model differ from the actual distribution of the data too strongly, then the use of unlabeled data will hurt instead of help performance」（8 節）．ラベルが少ないと，EM はほぼラベル無しデータのクラスタリングになり，クラスタとクラスが対応しないと成績が下がる（5.3.1 節）．

**MER-DA との接点**（推測．検定していない）．EM-λ の λ は，提案の疑似ラベルの損失の重みと同じ役割である．「ラベルが多いと EM が下回る」という結果は，K=25 で疑似ラベルの得が消えるか損になるかを確かめる理由になる．

**確認の水準と出典．**◎（1・5・6・8 節）．https://doi.org/10.1023/A:1007692713085 ，PDF：https://publications.ri.cmu.edu/storage/publications/pub_files/pub1/nigam_k_1999_1/nigam_k_1999_1.pdf

### Grandvalet & Bengio 2005（NIPS 17, pp. 529–536．会議は 2004 年．arXiv なし）Semi-supervised Learning by Entropy Minimization

**問い．**生成の模型を作らずに，判別の模型（P(y|x) だけを学ぶ模型）にラベル無しデータを入れられるか．

**方法．**ラベル無しの標本での予測のエントロピーを小さくする罰則を，対数尤度に足す．λ は 10 分割の交差検証で選ぶ．
```
C(θ, λ) = Σ_{i: ラベル付き} log f_{y_i}(x_i; θ) + λ Σ_{i: ラベル無し} Σ_k f_k(x_i; θ) log f_k(x_i; θ)
  f_k(x; θ)：クラス k の予測確率．第 2 項は −λ × エントロピー
```

**分かったこと．**人工データで，生成の模型が正しいときは混合模型と同程度，模型の仮定が崩れると混合模型より良い．ラベルがクラスタと無関係なデータでは，交差検証で λ がほぼ 0 に選ばれ，ラベルだけの学習と同程度に戻る（4.1 節）．表情認識（7 感情，正面の顔 375 枚，ラベルは 1/10）で，カーネルのロジスティック回帰の誤り率 53.6 ± 1.3% が，エントロピー最小化で 52.0 ± 1.9% になった．近傍法の系の方法（63.1%，63.8%）が弱いのは，同じ人の別の表情が近くに並び，表情ではなく人物を拾うためと著者は説明する（4.2 節）．

**前提と限界．**著者の記述：「Benefitting from unlabeled data requires assumptions of some sort on the relationship between X and Y」（2.1 節）．クラスの重なりが小さいときに役立つという理論に基づく．経験エントロピーは急に変わる関数では真のエントロピーの近似にならないので，滑らかさの罰則が要る（2.3 節）．

**MER-DA との接点**（推測．検定していない）．凍結した特徴の上の線形の softmax 分類器にそのまま足せる，最も単純なラベル無しの損失である．表情の実験で「人物を拾う」と書かれた現象は，MER-DA で話者を分ける理由と同じ種類の交絡である．

**確認の水準と出典．**◎（全文）．https://proceedings.neurips.cc/paper_files/paper/2004/file/96f2b50b5d3613adf9c27049b2a888c7-Paper.pdf ，頁は同じ論文集の Metadata.json で確かめた．

### Lee 2013（ICML 2013 Workshop: Challenges in Representation Learning．arXiv なし）Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks

**問い．**深い網の半教師あり学習を，簡単な手続きでできるか．

**方法．**ラベル無しの標本に，重みを更新するたびに予測の確率が最大のクラスを疑似ラベルとして付け，普通の損失で学ぶ．ラベル無しの損失の重み α(t) を学習の途中で 0 から上げる．
```
L = (1/n) Σ_m Σ_i L(y_i^m, f_i^m) + α(t) (1/n′) Σ_m Σ_i L(y′_i^m, f′_i^m)
α(t) = 0（t < T₁），(t − T₁)/(T₂ − T₁) α_f（T₁ ≤ t < T₂），α_f（T₂ ≤ t）．α_f = 3
  y′：疑似ラベル（予測の確率が最大のクラスを 1 にした符号）
```
著者は，この損失がエントロピー最小化（Grandvalet & Bengio）と同じ働きをすると述べる（3.2 節）．

**分かったこと．**MNIST の誤り率（表 2）：ラベル 100 本で，ドロップアウトつきの網 21.89% が疑似ラベルで 16.15%，自己符号化器の事前学習を足して 10.49%．ラベル 600 本で 8.57% から 5.03%．

**前提と限界．**著者の記述：α(t) が大きすぎるとラベル付きデータの学習まで乱れ，小さすぎるとラベル無しデータの得が無くなる（3 節）．こちらの読み：つまみはラベル付きの検証用データ 1,000 本で決めており（4 節 "For validation set, we picked up 1000 labeled examples"），学習のラベル 100 本より多い．

**MER-DA との接点**（推測．検定していない）．提案の疑似ラベルの損失は，この形に「疑似ラベルを別のモデル（単独の分類器の積）が作る」ことを足したものにあたる．

**確認の水準と出典．**◎（全文）．元の URL（deeplearning.net）は開けなかったので，写し https://storage.googleapis.com/kaggle-forum-message-attachments/8924/pseudo_label_final.pdf を読んだ．会場は PDF の 1 頁目の記載で確かめた．

### Tarvainen & Valpola 2017（NIPS 2017, pp. 1195–1204．arXiv 1703.01780）Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results

**問い．**一貫性正則化の目標（ラベル無しの標本で予測をそろえる先）を，どう作れば良いか．

**方法．**学ぶモデル（論文の語で student）の重み θ の指数移動平均（EMA．過去の値ほど小さい重みで平均する）で，目標を出すモデル（論文の語で teacher）の重み θ′ を作る．2 つのモデルに別々の雑音 η，η′ を入れた予測の二乗誤差を一貫性の損失にする．θ′ は勾配で更新しない．
```
θ′_t = α θ′_{t−1} + (1 − α) θ_t（α = 0.99 → 0.999）
J(θ) = E_{x, η′, η} ‖ f(x, θ′, η′) − f(x, θ, η) ‖²
全体の損失 = ラベル付きの交差エントロピー + w(t) J(θ)
```

**分かったこと．**SVHN のラベル 250 本で誤り率がラベルだけ 27.77 ± 3.18% から 4.35 ± 0.50% に（表 1），CIFAR-10 のラベル 4,000 本で 20.66 ± 0.57% から 12.31 ± 0.28% に（表 2）下がった．ResNet と組み合わせると CIFAR-10 の 4,000 本で 6.28%（抄録）．EMA の減衰率と一貫性の重みは 1 桁程度の範囲を外れると成績が急に下がる（3.4 節）．

**前提と限界．**著者の記述：「The success of consistency regularization depends on the quality of teacher-generated targets」（5 節）．一貫性の損失に二乗誤差を使い，KL ダイバージェンスの方が自然に見えると書く（3.4 節）．

**MER-DA との接点**（推測．検定していない）．線形の分類器でも重みの EMA は安く作れる．一貫性の損失には入力の乱れが要り，凍結した特徴では特徴への雑音やドロップアウトで代わりを作ることになる．

**確認の水準と出典．**◎（2・3・5 節，表 1・2）．https://proceedings.neurips.cc/paper_files/paper/2017/hash/68053af2923e00204c3ca7c6a3150cf7-Abstract.html ，https://arxiv.org/abs/1703.01780

### Berthelot, Carlini, Goodfellow, Papernot, Oliver, Raffel 2019（NeurIPS 2019, pp. 5049–5059．arXiv 1905.02249）MixMatch: A Holistic Approach to Semi-Supervised Learning

**問い．**エントロピー最小化，一貫性正則化，一般の正則化を 1 つにまとめると，どこまで良くなるか．

**方法．**ラベル無しの標本を K 回データ拡張した予測の平均を温度 T で鋭くして推測のラベルにする．ラベル付きと推測のラベル付きの標本を MixUp（2 つの標本と 2 つのラベルを同じ比で線形に混ぜる）で混ぜて学ぶ．
```
Sharpen(p, T)_i = p_i^{1/T} / Σ_j p_j^{1/T}（T = 0.5）
λ ~ Beta(α, α)，λ′ = max(λ, 1 − λ)，x′ = λ′ x₁ + (1 − λ′) x₂，p′ = λ′ p₁ + (1 − λ′) p₂
L = L_X（ラベル付きの交差エントロピー）+ λ_U L_U（推測のラベルとの二乗誤差）
```

**分かったこと．**CIFAR-10 のラベル 250 本で誤り率 11.80%（表 4．抄録は 38% から 11% と書く）．除去実験（表 4）で，鋭くするのをやめる（T=1）と 27.83%，MixUp をやめると 39.11% になった．

**前提と限界．**著者の記述：T と K は全実験で固定し，α と λ_U だけをデータセットごとに変えた（3.5 節）．こちらの読み：K 回のデータ拡張が前提の手法である．

**MER-DA との接点**（推測．検定していない）．MixUp は特徴のベクトルどうしでも作れるので，凍結した特徴の上で使える数少ない部品である．データ拡張の平均は凍結した特徴では作れない．

**確認の水準と出典．**◎（3 節，4.2.3 節，表 4）．https://proceedings.neurips.cc/paper_files/paper/2019/hash/1cd138d0499a68f4bb72bee04bbec2d7-Abstract.html ，https://arxiv.org/abs/1905.02249

### Sohn, Berthelot, Li, Zhang, Carlini, Cubuk, Kurakin, Zhang, Raffel 2020（NeurIPS 2020, pp. 596–608．arXiv 2001.07685）FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence

**問い．**複雑になった半教師あり学習を，簡単な手続きで同じ成績にできるか．

**方法．**弱いデータ拡張の画像の予測の最大確率が閾値 τ 以上なら，そのクラスを疑似ラベルにし，強いデータ拡張（RandAugment など）の画像の予測をその疑似ラベルに合わせる．
```
ℓ_u = (1/(μB)) Σ_b 1(max(q_b) ≥ τ) H(q̂_b, p_m(y | A(u_b)))
  q_b = p_m(y | α(u_b))（弱い拡張の予測），q̂_b = argmax(q_b)，A：強い拡張．τ = 0.95
全体の損失 = ℓ_s + λ_u ℓ_u（λ_u = 1）
```

**分かったこと．**CIFAR-10 でラベル 250 本で正解率 94.93%，40 本（クラスあたり 4 本）で 88.61%（抄録）．クラスあたり 1 本（4.4 節）では，4 通りのラベルの選び方で 48.58〜85.32%（中央値 64.28%）にばらついた．代表的な標本だけを選ぶと中央値 78%，外れた標本だけを選ぶと 10% で学習が進まなかった．閾値は 0.95 が最良で，小さくすると 1.5% 以上下がる（5 節）．

**前提と限界．**著者の記述：重みの減衰と最適化法の選び方が成績を大きく左右し，「the same technique can not always be directly compared across different implementations」（6 節）．こちらの読み：強いデータ拡張が前提である．

**MER-DA との接点**（推測．検定していない）．クラスあたり 1 本の結果が「どの 1 本を選ぶか」で 48〜85% にばらつくことは，MER-DA の K=1 で反復の間のばらつきが大きくなる理由の先例になる．強い拡張の代わりを凍結した特徴の上でどう作るかが，FixMatch 型の手法を比べる相手に入れるときの論点になる．

**確認の水準と出典．**◎（3・4.4・5・6 節）．https://proceedings.neurips.cc/paper_files/paper/2020/hash/06964dce9addb1c5cb5d6e3d9838f733-Abstract.html ，https://arxiv.org/abs/2001.07685

### Oliver, Odena, Raffel, Cubuk, Goodfellow 2018（NeurIPS 2018．arXiv 1804.09170）Realistic Evaluation of Deep Semi-Supervised Learning Algorithms

**問い．**深い網の半教師あり学習の評価は，実際の使い方での問題を見落としていないか．

**方法．**主な手法（Π 模型，Mean Teacher，VAT，疑似ラベル，エントロピー最小化）を同じ実装で作り直し，ラベル無しの標本に別のクラスが混ざる場合，転移学習との比較，検証用データが小さい場合を調べた．検証の誤差の推定の精度は Hoeffding の不等式で見積もる．
```
P(|V̄ − E[V]| < p) > 1 − 2 exp(−2 n p²)（V̄：n 本の検証用データでの誤差の推定）
```

**分かったこと．**ImageNet で事前学習した網を CIFAR-10 の 4,000 本で微調整すると誤り率 12.09% で，同じ網のどの半教師あり学習よりも低かった（4.3 節）．ラベル無しの標本に学習の 6 クラスに無いクラスが混ざると，ラベル無しの標本を使わない場合より悪くなることがある（4.4 節，図 2）．誤差の推定を 95% の確率で 1% 以内にするには検証用データがおよそ 20,000 本要り，学習のラベルの 10% の大きさの検証用データでは手法の差を見分けられなかった（4.6 節，図 5）．

**前提と限界．**著者の記述：「SSL methods which rely on heavy hyperparameter tuning on a large validation set may have limited real-world applicability」（4.6 節）．別のクラスの実験ではつまみを調整し直していない（4.4 節）．

**MER-DA との接点**（推測．検定していない）．MER-DA がつまみをターゲットの K 本ではなくソースのコーパスを仮のターゲットにして選ぶ理由の，SSL の側の根拠になる．ラベル無しの発話に 4 感情以外の発話が混ざる場合には，4.4 節の結果が当てはまる．

**確認の水準と出典．**◎（抄録，3 節の P.1〜P.6，4.3・4.4・4.6 節）．会場は PDF の 1 頁目の記載（32nd Conference on Neural Information Processing Systems）で確かめた．論文集の頁は確かめていない．https://arxiv.org/abs/1804.09170

**C. 教師なし DA**

### Long, Cao, Wang, Jordan 2015（ICML 2015，PMLR 37:97–105．arXiv 1502.02791）Learning Transferable Features with Deep Adaptation Networks（DAN）

**問い．**深い網の上の層の特徴は課題に特化して転移しにくい．上の層の特徴の分布の差を明示的に小さくできるか．

**方法．**AlexNet の conv1〜3 を固定し，conv4〜5 を微調整し，fc6〜fc8 の各層でソースとターゲットの特徴の分布の差を複数の核の MMD（MK-MMD）で測って損失に足す．核の重み β は MMD の検定力が最大になるように二次計画で選ぶ．
```
d²_k(p, q) = ‖ E_p[φ(x_s)] − E_q[φ(x_t)] ‖²_{H_k}（φ：核 k の特徴写像）
min_Θ (1/n_a) Σ_i J(θ(x_i^a), y_i^a) + λ Σ_{ℓ=6}^{8} d²_k(D_s^ℓ, D_t^ℓ)
定理 1：ε_t(θ) ≤ ε_s(θ) + 2 d_k(p, q) + C
```
λ はソースのラベル付きとターゲットのラベル無しからなる検証用データで，ソースの分類器の誤差と 2 標本の判別器の誤差を合わせて選ぶ（4.1 節）．

**分かったこと．**Office-31 の UDA（表 1，6 組の平均の正解率）：微調整だけの CNN 70.1，単層・単一核の DDC 70.6，DAN 72.9．Saenko 2010 の手順の半教師あり（表 3，3 組の平均）：DDC 91.9，DAN 93.1．

**前提と限界．**著者の記述：「it is interesting to study the principled way of deciding the boundary of generality and specificity」（5 節．どの層から先を適応させるかの決め方は未解決）．定理 1 の C は両ドメインで同時によい仮説の誤差を含む．

**MER-DA との接点**（推測．検定していない）．MER-DA ではエンコーダを凍結しているので，特徴を動かして MMD を小さくする部分は作れない．特徴に線形の変換を 1 層足すなら作れる．MER-DA の中心合わせ（平均を引く）は，線形の核の MMD を 0 にする操作にあたる．

**確認の水準と出典．**◎（3 節，4.1 節，表 1・3，5 節）．https://proceedings.mlr.press/v37/long15.html ，https://arxiv.org/abs/1502.02791

### Ganin & Lempitsky 2015（ICML 2015，PMLR 37:1180–1189．arXiv 1409.7495）Unsupervised Domain Adaptation by Backpropagation，および Ganin, Ustinova, Ajakan, Germain, Larochelle, Laviolette, Marchand, Lempitsky 2016（JMLR 17(59):1–35．arXiv 1505.07818）Domain-Adversarial Training of Neural Networks（DANN）

**問い．**ソースのラベルを当てられ，しかもソースかターゲットかを見分けられない特徴を，普通の逆伝播で学べるか．

**方法．**特徴抽出器 G_f の上に，クラスを当てる G_y とドメインを当てる G_d を置く．G_d はドメインの損失を小さくし，G_f はドメインの損失を大きくする．勾配反転層で 1 回の逆伝播にまとめる．Ben-David の上界のドメインのずれを，ドメインの判別器の誤差 ε から作る proxy A-distance で近似する．
```
E(θ_f, θ_y, θ_d) = (1/n) Σ_{ソース} L_y^i(θ_f, θ_y) − λ ( (1/n) Σ_{ソース} L_d^i(θ_f, θ_d) + (1/n′) Σ_{ターゲット} L_d^i(θ_f, θ_d) )
(θ̂_f, θ̂_y) = argmin E，θ̂_d = argmax E
proxy A-distance：d̂_A = 2(1 − 2ε)
λ の予定：λ_p = 2 / (1 + exp(−10 p)) − 1（p：学習の進み 0〜1）
```

**分かったこと．**JMLR 版の表 3（Office，完全な転導的評価）：A→W はソースのみ .642，DAN .685，DANN .730．W→D は .978 から .992．つまみは逆検証（reverse validation．ターゲットに付けた疑似ラベルで逆向きの分類器を学び，ソースの検証用データで測る）で，ターゲットのラベルを使わずに選んだ（5.1.2 節．浅い網の実験）．ICML 版は GTSRB で 1,280 本のラベルを足す半教師ありの実験を 1 つ載せ，「thorough verification of semi-supervised setting is left for future work」と書いた（5 節）．

**前提と限界．**著者の記述：定理の保証を保つには，ドメインの判別器の仮説の集合がクラスの分類器の仮説の集合を含む必要がある（4.2 節）．定理 2 は β ≥ inf [R_DS(η*) + R_DT(η*)] を含み，両ドメインで同時によい分類器が無いと上界は小さくならない（3.3 節）．

**MER-DA との接点**（推測．検定していない）．proxy A-distance はドメインのずれの指標として MER-DA の測定でも使える量である．特徴を動かす部分は凍結したエンコーダでは作れない．逆検証は，ターゲットのラベルを使わずにつまみを選ぶ方法の先例である．

**確認の水準と出典．**◎（JMLR 版の 3〜6 節と表 3．ICML 版は 3 節と 5 節）．https://proceedings.mlr.press/v37/ganin15.html ，https://jmlr.org/papers/v17/15-239.html ，https://arxiv.org/abs/1505.07818

### Sun, Feng, Saenko 2016（AAAI 2016．arXiv 1511.05547）Return of Frustratingly Easy Domain Adaptation（CORAL），および Sun & Saenko 2016（ECCV 2016 Workshops，LNCS, pp. 443–450．arXiv 1607.01719）Deep CORAL

**問い．**ターゲットのラベルが無いとき，Daumé 2007 のように簡単な DA の方法を作れるか．

**方法．**CORAL：ソースの特徴を白色化し（共分散を単位行列にする），ターゲットの共分散で色づけし直す線形変換を，ラベル無しで閉じた形で求める．変換したソースの特徴で線形 SVM を学ぶ．Deep CORAL：網の層の出力の共分散の差を損失に足す．
```
CORAL：C_S = cov(D_S) + I，C_T = cov(D_T) + I，D_S* = D_S C_S^{−1/2} C_T^{1/2}
Deep CORAL：ℓ_CORAL = (1 / 4d²) ‖ C_S − C_T ‖²_F，ℓ = ℓ_class + Σ_i λ_i ℓ_CORAL
```

**分かったこと．**CORAL の表 2（Office-31，AlexNet の fc7 の特徴の上の線形 SVM，6 組の平均）：適応なし 64.4，CORAL 66.9．ソースで微調整した fc7 では 65.5 から 69.4．Amazon レビュー（表 5，4 組の平均）：76.7 から 78.0．SVM の C はソースでの交差検証で選び，他のつまみは無い（4 節）．Deep CORAL の表 1（Office-31，6 組の平均）：CNN 70.1，CORAL 70.4，DAN 71.3，Deep CORAL 72.1．λ は学習の終わりに分類の損失と CORAL の損失がほぼ同じ大きさになるように決めた．

**前提と限界．**著者の記述：疎な bag of words の文書の特徴より，相関の強い深い特徴で得が大きいと書く．深い文書の特徴での CORAL は今後の課題（CORAL 5 節）．こちらの読み：平均と共分散（2 次の統計量）だけをそろえ，クラスごとの分布は扱わない．

**MER-DA との接点**（推測．検定していない）．凍結した特徴の上の線形の分類器で，ターゲットのラベルを使わずに作れるので，MER-DA の中心合わせ（1 次の統計量）を 2 次に広げた比べる相手になる．1024 次元の音声の特徴では共分散の推定にターゲットの発話数が足りるかが論点になる．

**確認の水準と出典．**◎（CORAL は 3・4 節と表 1〜5，Deep CORAL は 3・4 節と表 1）．https://doi.org/10.1609/aaai.v30i1.10306 ，https://arxiv.org/abs/1511.05547 ，https://doi.org/10.1007/978-3-319-49409-8_35 ，https://arxiv.org/abs/1607.01719 ．AAAI 版の頁番号は確かめていない．

### Tzeng, Hoffman, Saenko, Darrell 2017（CVPR 2017, pp. 2962–2971．arXiv 1702.05464）Adversarial Discriminative Domain Adaptation（ADDA）

**問い．**敵対的な DA の手法の設計の選択（生成か判別か，重みを共有するか，どの敵対的な損失か）を整理し，よい組み合わせを示せるか．

**方法．**まずソースのラベルでソースの特徴抽出器 M_s と分類器 C を学ぶ．次に M_s と C を固定し，ターゲット用の特徴抽出器 M_t を M_s で初期化して，ドメインの判別器 D と GAN の損失で競わせる．重みは共有しない．
```
min_D L_advD = − E_{x_s} [log D(M_s(x_s))] − E_{x_t} [log(1 − D(M_t(x_t)))]
min_{M_t} L_advM = − E_{x_t} [log D(M_t(x_t))]
```

**分かったこと．**表 2（数字の UDA の正解率）：MNIST→USPS はソースのみ 0.752，ADDA 0.894．USPS→MNIST は 0.571 から 0.901．SVHN→MNIST は 0.601 から 0.760．表 3（NYUD の RGB→深度，19 クラス）：全体の正解率 0.139 から 0.211．改善しないクラスもあり，適応の前に正解が 0 だったクラスは 3 つとも回復しなかった（5.2 節）．

**前提と限界．**著者の記述：ターゲットのラベル無しだけでは回復しないクラスがある（5.2 節）．こちらの読み：ターゲット用の特徴抽出器を学ぶことが前提である．

**MER-DA との接点**（推測．検定していない）．凍結したエンコーダでは M_t を学べない．「ソースの分類器を固定し，ターゲットの側だけを動かす」という分け方は，SHOT と同じく，MER-DA の提案（特徴を固定して分類器を動かす）と逆の分け方である．

**確認の水準と出典．**◎（3 節，4 節，表 2・3）．https://doi.org/10.1109/CVPR.2017.316 ，https://arxiv.org/abs/1702.05464

### Long, Cao, Wang, Jordan 2018（NeurIPS 2018, pp. 1640–1650．arXiv 1705.10667）Conditional Adversarial Domain Adaptation（CDAN）

**問い．**分類の問題の特徴の分布はクラスごとに山が分かれる（多峰）．特徴だけを合わせる敵対的な DA はクラスの対応を取り違える．分類器の予測で条件づけて合わせられるか．

**方法．**ドメインの判別器 D に，特徴 f と予測 g の外積（多重線形写像）を入れる．次元が大きいときは乱数の射影で近似する．予測のエントロピーが小さい（確信のある）標本の重みを大きくする（CDAN+E）．
```
min_G E(G) − λ E(D, G)，min_D E(D, G)
E(D, G) = − E_{x_s} log D(f_s ⊗ g_s) − E_{x_t} log(1 − D(f_t ⊗ g_t))
CDAN+E の重み：w(H(g)) = 1 + e^{−H(g)}，H(g) = − Σ_c g_c log g_c
```
つまみは重要度重みつきの交差検証（IWCV）で選ぶ（4 節）．

**分かったこと．**表 1（Office-31，ResNet-50，6 組の平均）：ResNet-50 のみ 76.1，DANN 82.2，ADDA 82.9，CDAN 86.6，CDAN+E 87.7．AlexNet では 70.1 から 77.7．エントロピーの重みは，予測が正しい標本でほぼ 1，誤った標本で 1 より十分小さくなった（図 2b）．

**前提と限界．**著者の記述：特徴の表現だけを合わせる方法は合わせ不足になりやすい（5 節）．こちらの読み：予測で条件づけるので，ソースの分類器がターゲットで大きく間違うと，条件づけも間違う．

**MER-DA との接点**（推測．検定していない）．「予測との外積で条件づける」はクラスごとの位置のずれを扱う考え方で，MER-DA の仕組みの仮説のクラスごとのずれの扱いと比べられる．特徴を動かす部分は凍結したエンコーダでは作れない．

**確認の水準と出典．**◎（3 節，4 節，表 1，5 節）．https://proceedings.neurips.cc/paper_files/paper/2018/hash/ab88b15733f543179858600245108dd8-Abstract.html ，https://arxiv.org/abs/1705.10667

### Saito, Watanabe, Ushiku, Harada 2018（CVPR 2018, pp. 3723–3732．arXiv 1712.02560）Maximum Classifier Discrepancy for Unsupervised Domain Adaptation（MCD）

**問い．**ドメインの判別器はクラスの境界を考えないので，境界の近くにあいまいなターゲットの特徴を作る．クラスの境界を考えて分布を合わせられるか．

**方法．**特徴抽出器 G の上に 2 つの分類器 F₁，F₂ を置く．A：ソースのラベルで全体を学ぶ．B：G を固定し，ソースの分類の損失を保ちながら，ターゲットでの 2 つの分類器の予測の食い違いを大きくする．C：F₁，F₂ を固定し，G で食い違いを小さくする（n 回）．食い違いを HΔH-distance の近似と位置づける．
```
d(p₁, p₂) = (1/K) Σ_k |p₁k − p₂k|
B：min_{F₁, F₂} L(X_s, Y_s) − E_{x_t}[d(p₁(y|x_t), p₂(y|x_t))]
C：min_G E_{x_t}[d(p₁(y|x_t), p₂(y|x_t))]
近似：d_HΔH(S, T) ≈ sup_{F₁,F₂} E_{x∼T} I[F₁∘G(x) ≠ F₂∘G(x)]
```

**分かったこと．**表 2（VisDA，ResNet101，12 クラスの平均）：ソースのみ 52.4，MMD 61.1，DANN 57.4，MCD（n=4）71.9．数字の課題では，ターゲットの少数のラベルでつまみを選んだ ATDA が一部の設定で上回ったと著者は書く（4.2 節）．

**前提と限界．**著者の記述：2 つの分類器はソースで正しく分類する前提で，ソースでの食い違いを小さいとみなして近似している（3.4 節）．こちらの読み：特徴抽出器を動かす段 C が中心である．

**MER-DA との接点**（推測．検定していない）．凍結したエンコーダでは段 C を作れない．2 つの分類器の食い違いの考え方は，MER-DA でモダリティごとの分類器がターゲットで食い違う割合を，ずれの量として測ることに使える．

**確認の水準と出典．**◎（3 節，4 節，表 2）．https://doi.org/10.1109/CVPR.2018.00392 ，https://arxiv.org/abs/1712.02560

### Zou, Yu, Kumar, Wang 2018（ECCV 2018，LNCS, pp. 297–313．arXiv 1810.07911）Domain Adaptation for Semantic Segmentation via Class-Balanced Self-Training（CBST）

**問い．**疑似ラベルの自己学習は，転移しやすいクラスに偏り，難しいクラスを無視していく．クラスごとに釣り合った疑似ラベルを作れるか．

**方法．**疑似ラベルの選び方を損失の最小化として書き，クラスごとの閾値 k_c を置く．各クラスについて，そのクラスと予測された画素の確率を並べ，上位 p の位置の確率を exp(−k_c) にする．p は 20% から始めて 1 回ごとに 5% 足し，最大 50% にする．
```
min_{w, ŷ} − Σ_{ソース} y log p(w, I_s) − Σ_{ターゲット} Σ_c [ ŷ^(c) log p(c | w, I_t) + k_c ŷ^(c) ]
解：ŷ^(c)* = 1（c = argmax_c p(c|w, I_t)/exp(−k_c) かつ p(c|w, I_t)/exp(−k_c) > 1），それ以外は 0
```

**分かったこと．**表 3（GTA5→Cityscapes，ResNet-38，19 クラスの mIoU）：ソースのみ 35.4，クラスを区別しない自己学習 41.5，CBST 45.2，空間の事前分布を足して 46.2，複数の尺度で 47.0．

**前提と限界．**著者の記述：やさしいクラスはターゲットで確信度が高くなり，全クラスに共通の閾値では偏る（4.2 節）．空間の事前分布はソースとターゲットで配置が似ているときだけ使える（6 節）．

**MER-DA との接点**（推測．検定していない）．MER-DA の積で作る疑似ラベルが特定の感情に偏る場合，クラスごとの閾値（上位 p を感情ごとに取る）は凍結した特徴の上でもそのまま作れる．UAR はクラスの釣り合いに敏感なので，この偏りは UAR に直接影響する．

**確認の水準と出典．**◎（4 節，表 3，6 節）．https://doi.org/10.1007/978-3-030-01219-9_18 （Crossref の題は "Unsupervised Domain Adaptation for Semantic Segmentation via Class-Balanced Self-training"），https://arxiv.org/abs/1810.07911

### Liang, Hu, Feng 2020（ICML 2020，PMLR 119:6028–6039．arXiv 2002.08546）Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation（SHOT）

**問い．**ソースのデータを渡せず，学習済みのソースのモデルだけがあるとき（ソースフリーの DA），ターゲットのラベル無しデータだけで適応できるか．

**方法．**ソースの分類器 h（論文の語で hypothesis）を固定し，ターゲットの特徴抽出器 g_t だけを学ぶ．損失は，各標本の予測を確信のあるものにする項と，予測の平均をクラスの間で一様にする項（合わせて information maximization，IM）と，クラスの重心に最も近いクラスを疑似ラベルにする自己学習の項である．
```
L_ent = − E_{x_t} Σ_k δ_k(f_t(x_t)) log δ_k(f_t(x_t))
L_div = Σ_k p̂_k log p̂_k = D_KL(p̂, 1/K) − log K，p̂ = E_{x_t}[δ(f_t(x_t))]
L(g_t) = L_ent + L_div − β E Σ_k 1[k = ŷ_t] log δ_k(h_t(g_t(x_t)))（β = 0.3）
```

**分かったこと．**表 4（Office-Home，ResNet-50，12 組の平均）：ソースのモデルのみ 60.2，SHOT-IM 70.5，SHOT 71.8．除去実験（表 6）：L_ent だけでは Office-Home で 55.5 とソースのみ（60.2）を下回り，L_div を足すと 70.5 になった．Lee 2013 の素朴な疑似ラベルは 64.1，重心による疑似ラベルは 68.9．

**前提と限界．**著者の記述：ソースとターゲットでラベルの集合が同じ閉じた設定を基本にし，部分集合（partial-set）と開いた集合（open-set）の設定には手法を変えて評価した．こちらの読み：L_div は「ターゲットの予測は全クラスに一様に散らばる」と仮定しており，ターゲットのクラスの割合が偏るときには誤った方向に引く．

**MER-DA との接点**（推測．検定していない）．「エントロピー最小化だけではソースのみを下回り，多様性の項で直る」という表 6 の結果は，MER-DA の疑似ラベルにクラスの釣り合いの制約が要るかを考える材料になる．SHOT は分類器を固定して特徴を動かし，MER-DA の提案は特徴を固定して分類器を動かすので，役割が逆である．

**確認の水準と出典．**◎（3 節，4 節，表 4・6，5 節）．https://proceedings.mlr.press/v119/liang20a.html ，https://arxiv.org/abs/2002.08546

### Wang, Shelhamer, Liu, Olshausen, Darrell 2021（ICLR 2021．arXiv 2006.10726）Tent: Fully Test-Time Adaptation by Entropy Minimization

**問い．**学習の後，ソースのデータもラベルも無く，テストのデータとモデルの重みだけで適応できるか（完全なテスト時の適応）．

**方法．**テストの予測のエントロピーを小さくする．更新するのは正規化層の統計量（テストのバッチで推定し直す）と，チャネルごとの拡大 γ と平行移動 β（全パラメータの 1% 未満）だけで，他は固定する．バッチごとにオンラインで更新する．
```
H(ŷ) = − Σ_c p(ŷ_c) log p(ŷ_c)
x̄ = (x − μ) / σ，x′ = γ x̄ + β（μ, σ はテストのバッチで推定，γ, β を ∇H で更新）
```

**分かったこと．**表 2（最も強い劣化の誤り率 %）：CIFAR-10-C でソース 40.8，正規化の統計だけ更新 17.3，疑似ラベル 15.7，Tent 14.3．CIFAR-100-C で 67.2，42.6，41.2，37.3．ImageNet-C で誤り率 44.0%（ソース 59.5%）．MNIST→SVHN（ソースのみの誤り率 71.3%）では，Tent で誤り率が 79.8% に上がった（4.2 節）．

**前提と限界．**著者の記述：CIFAR-10.1 と ImageNetV2 の自然なずれでは「Although error is higher on both sets, indicating the presence of shift, tent does not improve generalization」（6 節）．どのパラメータを更新すべきかは「the issue is to identify parameters that are both expressive and reliable」（6 節）．

**MER-DA との接点**（推測．検定していない）．凍結したエンコーダと線形の分類器では，特徴の次元ごとの拡大と平行移動を更新することが Tent にあたる．コーパスの違いは劣化の画像ではなく自然なずれなので，6 節の失敗例に近い可能性がある．

**確認の水準と出典．**◎（3 節，4.1・4.2 節，表 2・3，6 節）．https://arxiv.org/abs/2006.10726 （PDF の頁の頭に "Published as a conference paper at ICLR 2021"）

**D. 教師あり DA と半教師あり DA**

### Donahue, Hoffman, Rodner, Saenko, Darrell 2013（CVPR 2013, pp. 668–675．arXiv なし）Semi-supervised Domain Adaptation with Instance Constraints

**問い．**ターゲットのラベル無しの標本のうち「同じ物体を別の視点から撮った」とわかる対を，線形の分類器の SSDA に使えるか．

**方法．**固定した特徴の上の線形の分類器に，ソースの超平面との角への罰則（PMT-SVM）か，ターゲットからソースへの線形変換（MMDT）を置き，同じ物体の対の予測をそろえる Laplacian の罰則を足す．
```
PMT-SVM の正則化：½‖θ̃‖² + (Γ/2) ‖θ̃‖² sin² α(θ̃, θ)（α：ソースの超平面 θ とのなす角）
ラベル無しの罰則：½ Σ_{j≠j′} β_{jj′} (f̃(x̃_j) − f̃(x̃_{j′}))²
```

**分かったこと．**Office の webcam→dslr（表 2）：ソースのみ 45.80，合併 50.79，DA のみ 54.57，DA＋制約 56.15．ターゲットにラベルの無い 15 クラス（表 3）では合併がソースのみを 1.3 ポイント下回った．

**前提と限界．**ラベル無しの標本に対の情報が要る．多クラスの実験は 1 組だけである．

**MER-DA との接点**（推測．検定していない）．凍結した特徴の上の線形の分類器にソースの重みへの罰則をかける SSDA の先例で，学習器が MER-DA に最も近い．罰則は角で，MER-DA は二乗距離である．

**確認の水準と出典．**◎（既存ノートで全文）．詳しくは `research_notes/半教師あり DA の先例/ssda.md` の 5．https://openaccess.thecvf.com/content_cvpr_2013/html/Donahue_Semi-supervised_Domain_Adaptation_2013_CVPR_paper.html

### Motiian, Jones, Iranmanesh, Doretto 2017（NIPS 2017, pp. 6670–6680．arXiv 1711.02536）Few-Shot Adversarial Domain Adaptation（FADA），および Motiian, Piccirilli, Adjeroh, Doretto 2017（ICCV 2017, pp. 5716–5726．arXiv 1709.10190）Unified Deep Supervised Domain Adaptation and Generalization（CCSA）

**問い．**ターゲットのラベル無しデータを使わず，クラスあたり 1〜8 本のターゲットのラベルとソースのラベルだけで（教師あり DA），深い埋め込みを適応させられるか．

**方法．**FADA：ソースとターゲットの標本の対を 4 群（同じドメインで同じクラス，違うドメインで同じクラス，など）に分け，対の群を当てる判別器をだますように埋め込みを学ぶ．CCSA：同じクラスのソースとターゲットの埋め込みを近づけ，違うクラスをマージン m 以上離す対の損失を足す．
```
CCSA：L = γ L_C + (1 − γ)(L_SA + L_S)
  L_SA = Σ_{同じクラスの対} ½ ‖g(x_s) − g(x_t)‖²，L_S = Σ_{違うクラスの対} ½ max(0, m − ‖g(x_s) − g(x_t)‖)²
```

**分かったこと．**FADA は MNIST→USPS のクラスあたり 1 本で，ターゲットで微調整するだけの 82.3 から 89.1 に上げた（表 1）．Office の 3 本（6 組の平均）は CCSA 85.8，FADA 84.9．CCSA の MNIST-USPS（表 4）で，クラスあたり 7 本の 91.1 はラベル無しを全部使う ADDA の 89.7 を上回る．

**前提と限界．**γ，m の値も選び方も本文に無い．

**MER-DA との接点**（推測．検定していない）．MER-DA の地図（ソースのラベルと K 本の合併）はこの教師あり DA の設定にあたる．

**確認の水準と出典．**◎（既存ノートで全文）．詳しくは `research_notes/半教師あり DA の先例/ssda.md` の 6a・6b．arXiv 番号は arXiv の API で題と注記（Accepted to NIPS 2017／International Conference on Computer Vision ICCV 2017）を確かめた．https://proceedings.neurips.cc/paper_files/paper/2017/hash/21c5bba1dd6aed9ab48c2b34c1a0adde-Abstract.html ，https://openaccess.thecvf.com/content_iccv_2017/html/Motiian_Unified_Deep_Supervised_ICCV_2017_paper.html

### Saito, Kim, Sclaroff, Darrell, Saenko 2019（ICCV 2019, pp. 8049–8057．arXiv 1904.06487）Semi-Supervised Domain Adaptation via Minimax Entropy（MME）

**問い．**SSDA の設定で，UDA の手法にターゲットのラベルを足すだけでは足りないのか．分類器の重みをターゲットに寄せられるか．

**方法．**余弦類似度の分類器の重みベクトルをクラスの代表点（プロトタイプ）とみなす．ラベル無しのターゲットの予測のエントロピー H を，分類器 C の側では大きくし（代表点をターゲットの特徴の方へ動かす），特徴抽出器 F の側では小さくする．
```
p(x) = σ( (1/T) Wᵀ F(x) / ‖F(x)‖ )（T = 0.05）
θ̂_F = argmin_{θ_F} L + λ H，θ̂_C = argmin_{θ_C} L − λ H（λ = 0.1）
```

**分かったこと．**DomainNet（126 クラス，7 組，ResNet34）の平均で，1 本／3 本の S+T 56.9／60.0，ENT（エントロピー最小化だけ）62.6／67.6，MME 66.4／68.9（表 1）．AlexNet では ENT が S+T を下回る（1 本で 29.1 対 40.0）．ラベルを増やすと ENT と MME の差は縮む（図 4）．

**前提と限界．**つまみと早期終了をターゲットのクラスあたり 3 本の検証用ラベルで決める．評価は学習に使ったラベル無しの標本そのもので行う（転導的な評価）．

**MER-DA との接点**（推測．検定していない）．MER-DA の設定は MME が定めた SSDA の設定にあたる．凍結した特徴では分類器の側（エントロピーの最大化）だけを作れる．

**確認の水準と出典．**◎（既存ノートで全文）．詳しくは `research_notes/半教師あり DA の先例/ssda.md` の 1 と `DAI_paper/参照論文/要点_問い3_同じ設定の研究.md` の 4．arXiv 番号は arXiv の API で題と注記（accepted to ICCV2019）を確かめた．https://openaccess.thecvf.com/content_ICCV_2019/html/Saito_Semi-Supervised_Domain_Adaptation_via_Minimax_Entropy_ICCV_2019_paper.html

### Kim & Kim 2020（ECCV 2020，LNCS 12359, pp. 591–607．arXiv 2007.09375）Attract, Perturb, and Explore: Learning a Feature Alignment Network for Semi-supervised Domain Adaptation（APE）

**問い．**ターゲットの少数のラベルは，その近くの標本だけをソースに揃え，残りのターゲットを取り残す（ドメイン内のずれ）．これを縮められるか．

**方法．**ラベル付き（ソースとターゲット）とラベル無しのターゲットの特徴の MMD を小さくし（attraction），代表点をエントロピーが大きくなる向きに動かした摂動に対する一貫性を保ち（perturbation），確信の高いラベル無しの標本に疑似ラベルを付ける（exploration）．
```
min_{p, θ} L_cls + α L_a + β L_e + γ L_p
```

**分かったこと．**DomainNet（ResNet-34）の 1 本／3 本の平均 67.6／71.7（MME 66.4／68.9）．補足資料の 10 本では ENT 74.0 が MME 73.0 を上回り，APE は 76.8．

**前提と限界．**α，β，γ，摂動の大きさ，閾値の値と選び方が本文に無い．

**MER-DA との接点**（推測．検定していない）．SSDA の専用の手法の得がラベルの本数とともに縮む例である．

**確認の水準と出典．**◎（既存ノートで全文）．詳しくは `research_notes/半教師あり DA の先例/ssda.md` の 2．arXiv 番号は arXiv の API で題と注記（Accepted at ECCV 2020）を確かめた．https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590579.pdf

### Li, Li, Shi, Yu 2021（CVPR 2021, pp. 2505–2514．arXiv 2104.09415）Cross-Domain Adaptive Clustering for Semi-Supervised Domain Adaptation（CDAC）

**問い．**MME のエントロピーの項を，ターゲットの標本の対の類似度に基づくクラスタリングの損失に替えると良くなるか．

**方法．**特徴の大きさの上位 5 個の添字が一致する対を同じクラスとみなし，予測の内積をそれに合わせる損失 L_AAC を，分類器では大きく，特徴抽出器では小さくする．FixMatch 型の疑似ラベルと一貫性の損失を足す．
```
θ_G* = argmin L_CE + λ L_AAC + L_PL + L_Con，θ_F* = argmin L_CE − λ L_AAC + L_PL + L_Con
```

**分かったこと．**DomainNet（ResNet34）の 1 本／3 本の平均 73.6／76.0（表 1）．

**前提と限界．**RandAugment のデータ拡張が前提である．λ，ν，τ は値だけ書かれている．

**MER-DA との接点**（推測．検定していない）．凍結した特徴では画像の拡張を前提にする疑似ラベルと一貫性の項を作れない．

**確認の水準と出典．**◎（既存ノートで全文）．詳しくは `research_notes/半教師あり DA の先例/ssda.md` の 3．https://openaccess.thecvf.com/content/CVPR2021/html/Li_Cross-Domain_Adaptive_Clustering_for_Semi-Supervised_Domain_Adaptation_CVPR_2021_paper.html

### Yang, Wang, Gao, Shrivastava, Weinberger, Chao, Lim 2021（ICCV 2021, pp. 8886–8896．arXiv 2007.12684）Deep Co-Training with Task Decomposition for Semi-Supervised Domain Adaptation（DECOTA）

**問い．**ソースのラベルとターゲットのラベルで 1 つの分類器を学ぶと，多いソースのラベルに支配される．問題を分けて扱えるか．

**方法．**ソースのラベルとラベル無しのターゲットで学ぶ UDA の模型 w_g と，ターゲットのラベルとラベル無しのターゲットで学ぶ SSL の模型 w_f を分け，一方の確信度が τ を超えたラベル無しの標本の予測を他方の疑似ラベルにする（co-training）．MixUp を足す．予測は 2 つの模型の平均．

**分かったこと．**DomainNet の 3 本の平均 75.6（MME 68.0，S+T 61.2．DECOTA の表 2）．補足資料の Real→Clipart で 10／20／50 本の S+T 69.1／72.4／77.5，DECOTA 81.8／82.6／85.0 で，差は縮む．50 本で ENT と FixMatch が MME を上回る．動機の文：「A single classifier learned together with both sources of supervision is thus easily dominated by the labeled source data」（1 節）．

**前提と限界．**つまみをターゲットのクラスあたり 3 本の検証用ラベルで選ぶ．評価は転導的である．

**MER-DA との接点**（推測．検定していない）．合併がソースに支配されるという動機は，MER-DA の合併の損の問題意識と重なる．MER-DA では「ソースの分類器の平均」と「K 本の分類器」の 2 つで疑似ラベルを交換する変種を作れる．

**確認の水準と出典．**◎（既存ノートで全文）．詳しくは `research_notes/半教師あり DA の先例/ssda.md` の 4．https://openaccess.thecvf.com/content/ICCV2021/html/Yang_Deep_Co-Training_With_Task_Decomposition_for_Semi-Supervised_Domain_Adaptation_ICCV_2021_paper.html

### Singh 2021（NeurIPS 2021, pp. 5089–5101．arXiv 2107.00085）CLDA: Contrastive Learning for Semi-Supervised Domain Adaptation

**問い．**SSDA には，ソースとラベル無しのターゲットの間のずれ（ドメイン間）と，ラベル付きとラベル無しのターゲットの間のずれ（ドメイン内）がある．両方を対照学習で縮められるか．

**方法．**ソースのクラスの重心（EMA で更新）と，疑似ラベルで作るターゲットのクラスの重心について，同じクラスの重心どうしを近づけ，違うクラスの重心を遠ざける（クラス単位の対照学習）．ラベル無しのターゲットの元の画像と強い拡張の画像の表現をそろえる（標本単位の対照学習，NT-Xent 損失）．
```
L_clu(C_i^t, C_i^s) = − log [ h(C_i^t, C_i^s) / ( h(C_i^t, C_i^s) + Σ_{r=1}^{K} Σ_{q∈{s,t}} 1{r≠i} h(C_i^t, C_r^q) ) ]
  h(u, v) = exp( uᵀv / (‖u‖ ‖v‖ τ) )（τ = 5）
L_tot = L_sup + α L_clu + β L_ins
```

**分かったこと．**DomainNet（表 2）の 1 本／3 本の平均：ResNet34 で CLDA 71.9／75.3，APE 67.6／71.7．AlexNet で CLDA 50.7／54.3，BiAT 45.5／49.4．

**前提と限界．**著者の記述（付録 D）：網の確信度の較正が崩れていると「images being pushed into wrong clusters」．ソースのモデルに偏りがあれば，その偏りもターゲットへ移る．こちらの読み：つまみは Office-Home の Product→Art の検証の成績で決めた（4 節）．

**MER-DA との接点**（推測．検定していない）．クラスの重心をドメイン間でそろえる部分は，凍結した特徴の上でも，特徴に線形の変換を 1 つ足せば作れる．標本単位の部分は強い拡張が要る．

**確認の水準と出典．**◎（3 節，4 節，表 2，付録 D）．https://proceedings.neurips.cc/paper_files/paper/2021/hash/288cd2567953f06e460a33951f55daaf-Abstract.html ，https://arxiv.org/abs/2107.00085

### Berthelot, Roelofs, Sohn, Carlini, Kurakin 2022（ICLR 2022．arXiv 2106.04732）AdaMatch: A Unified Approach to Semi-Supervised Learning and Domain Adaptation

**問い．**UDA，SSL，SSDA を 1 つの手法で扱えるか．FixMatch をドメインのずれに合わせて直せるか．

**方法．**FixMatch に 3 つの部品を足す．(1) ソースとターゲットを一緒に通したときと別々に通したときの出力（logit）を乱数の比で混ぜる（random logit interpolation）．(2) 疑似ラベルの分布をソースのラベルの分布に合わせる（distribution alignment）．(3) 確信度の閾値を，ソースの弱い拡張での最大確率の平均に比例させる（relative confidence threshold）．SSDA ではターゲットのラベル付きをソースのラベル付きのバッチに連結する．
```
Ỹ_TU,w = normalize( Ŷ_TU,w · E[Ŷ_SL,w] / E[Ŷ_TU,w] )
c_τ = (τ / n_SL) Σ_i max_j Ŷ_SL,w^(i,j)，mask(i) = 1[max_j Ỹ_TU,w^(i,j) ≥ c_τ]
L(θ) = L_source(θ) + μ(t) L_target(θ)，μ(t) = ½ − cos(min(π, 2πt/T)) / 2
```

**分かったこと．**除去実験（6 節の表 4，DomainNet の UDA 6 組）：AdaMatch 26.3，分布合わせを外すと 17.7，相対閾値を外すと 23.2，logit の混合を外すと 25.2．SSDA（付録 D.2 の表 6，DomainNet224，ターゲットのラベル 1／5／10／25 本）：AdaMatch 34.8／42.3／46.7／52.8，FixMatch+ 28.3／38.7／44.1／50.8．

**前提と限界．**著者の記述：「which is more important for future progress, making use of unlabeled data more efficiently, or attempting to better model domain shifts?」（7 節）．自己学習の確認の偏り（confirmation bias）が多数派と少数派のクラスの公平さに影響しうる（倫理の節）．ターゲットのラベルの分布が分かるなら E[Ŷ_SL,w] をそれに替えてよいと書く（3 節）．

**MER-DA との接点**（推測．検定していない）．分布合わせは疑似ラベルをソースのラベルの割合に寄せるので，感情の割合がコーパスで大きく違う MER-DA ではターゲットの割合を誤った方向に引く可能性がある．K 本のラベルからターゲットの割合は推定できない（K 本はクラスごとに同数なので）．相対閾値は凍結した特徴でもそのまま作れる．

**確認の水準と出典．**◎（既存ノートで評価設定を精読．今回 3.1 節の式 4〜11，6 節の表 4，7 節，倫理の節を読み直した）．詳しくは `research_notes/半教師あり DA の先例/ssda.md` の 7．https://arxiv.org/abs/2106.04732

### Yu & Lin 2023（CVPR 2023, pp. 24100–24109．arXiv 2302.02335）Semi-Supervised Domain Adaptation with Source Label Adaptation（SLA）

**問い．**既存の SSDA はターゲットをソースに合わせるので，ターゲットの標本を違うクラスのソースに合わせてしまうことがある．逆に，ソースのラベルをターゲットに合わせて直せるか．

**方法．**ソースのラベルを，ターゲットから見た理想のラベルの雑音入りの版とみなす．ターゲットの疑似ラベルで作ったクラスの重心による分類器（Protonet with Pseudo Centers，PPC）の予測と，元のソースのラベルを比 α で混ぜて，ソースのラベルを書き直す．既存の SSDA の手法のソースの損失をこれに替える．
```
c_k = (1/N_k) Σ_i 1{ỹ_i^u = k} f(x_i^u)（疑似ラベルで作る重心）
P_C̃(x)_k = exp(−d(f(x), c_k) · T) / Σ_j exp(−d(f(x), c_j) · T)
ỹ_i^s = (1 − α) y_i^s + α P_C̃(x_i^s)（α = 0.3），L̃_s = (1/|S|) Σ_i H(g(x_i^s), ỹ_i^s)
```

**分かったこと．**表 2（Office-Home，ResNet34）：ソースのラベルを理想のターゲットの模型 g* の予測に置き換えた「ideally-adapted S+T」は，A→C の 1 本／3 本で 82.9／87.4 で，S+T の 52.9／58.1 を大きく上回る（表の見出しは 3-shot と書くが列は 1-shot と 3-shot に分かれる）．表 4（DomainNet，ResNet34，1 本／3 本の平均）：MME 66.4／68.9 が SLA を足して 68.8／70.0，CDAC 73.6／76.0 が 75.0／76.9．

**前提と限界．**著者の記述：「Our work highlights that the usage of source data should be revisited carefully」（5 節）．MCL の報告値は再現できず，SLA を足すと再現値からは上がるが報告値には届かない（4.3 節）．こちらの読み：g* はターゲットの全ラベルで学ぶ理想の模型で，表 2 は上限の目安である．

**MER-DA との接点**（推測．検定していない）．コーパスごとにラベルの付け方（どれだけ強ければ「怒り」と付けるか）がずれるなら，ソースのラベルはターゲットから見て雑音を含む．SLA の書き直しは，凍結した特徴の上の重心の計算だけで作れる．

**確認の水準と出典．**◎（3 節，4 節，表 2・4，5 節）．https://openaccess.thecvf.com/content/CVPR2023/html/Yu_Semi-Supervised_Domain_Adaptation_With_Source_Label_Adaptation_CVPR_2023_paper.html ，https://doi.org/10.1109/CVPR52729.2023.02308 ，https://arxiv.org/abs/2302.02335

**E. 自己学習と疑似ラベルの理論**

### Kumar, Ma, Liang 2020（ICML 2020，PMLR 119:5468–5479．arXiv 2002.11361）Understanding Self-Training for Gradual Domain Adaptation

**問い．**分布が少しずつ変わる（段階的なずれ）とき，ラベル無しデータの自己学習で分類器を追随させられるか．その誤差を抑えられるか．

**方法．**ℓ₂ ノルムを R で抑えた正則化つきの線形の分類器を，ずれた先のラベル無しの標本に付けた硬い疑似ラベル（sign で ±1 にしたもの）で学び直す（ST）．これを中間のドメインの列 S₁, …, S_T に順に繰り返す．
```
ST(θ, S) = argmin_{θ′ ∈ Θ_R} (1/|S|) Σ_{x∈S} ℓ(M_θ′(x), sign(M_θ(x)))
定理 3.2（1 段）：L_r(θ′, Q) ≤ 2/(1 − ρR) · L_r(θ, P) + α* + (4BR + √(2 log(2/δ))) / √n
  ρ：P と Q のずれ（Wasserstein-∞ 距離）．1/R：幾何的なマージン．α*：Q で最良の分類器の損失
系 3.3（T 段）：L_r(θ, P_T) ≤ β^{T+1} ( α₀ + (4BR + √(2 log(2T/δ))) / √n )，β = 2/(1 − ρR)
```

**分かったこと．**表 1（5 回の平均）：回転 MNIST でソースのみ 31.9，ターゲットで直接自己学習 33.0，全部をまとめて自己学習 38.0，段階的な自己学習 87.9．Portraits で 75.3，76.9，78.9，83.8．表 2：正則化を外すと回転 MNIST で 45.8，柔らかい疑似ラベル（確率のまま）にすると 44.1（硬いラベルと正則化で 83.8）．例 3.6：柔らかいラベルの対数損失は今の θ 自身が最小点なので，自己学習しても θ が動かない．例 3.1：中間のドメインが無いと，ずれた先の誤差が 100% になる例がある．

**前提と限界．**著者の記述：定理はクラスの割合が変わらないこと（P(Y) = Q(Y)）と，ずれ ρ がマージンより小さいことを仮定する．Portraits で外挿を延ばすと全手法が弱く「developing and analyzing new techniques for gradual domain adaptation is an exciting avenue for future work」（付録）．全変動距離が小さくても Wasserstein 距離が大きいずれ（画像を 55 度回す）では段階的な自己学習は直接の自己学習と同程度（33.5% と 33.0%．5 節）．

**MER-DA との接点**（推測．検定していない）．正則化つきの線形の分類器の自己学習の理論で，MER-DA の線形の softmax 分類器とソースへの L2 の引き戻しに最も近い枠である．ただしクラスの割合が変わらない仮定は，感情の割合がコーパスで違う MER-DA では成り立たない．「柔らかい疑似ラベルでは動かない」という例 3.6 は，MER-DA の疑似ラベルを確率で与えるか硬いラベルで与えるかの選択に関わる．

**確認の水準と出典．**◎（1〜3 節，5 節，表 1・2，付録の外挿の節）．https://proceedings.mlr.press/v119/kumar20c.html ，https://arxiv.org/abs/2002.11361

### Chen, Wei, Kumar, Ma 2020（NeurIPS 2020, pp. 21061–21071．arXiv 2006.10032）Self-training Avoids Using Spurious Features Under Domain Shift

**問い．**既存の理論はソースとターゲットが近い場合しか扱わない．ずれが大きくても自己学習が成功する場合を証明できるか．

**方法．**線形の分類器 w = (w₁, w₂) で，x₂ はソースではラベルと相関するがターゲットではラベルと独立な見かけの特徴（spurious feature）とする．ソースの分類器から始めて，ターゲットのラベル無しの標本でエントロピー（の近似 exp(−|t|)）を射影つきの勾配法で小さくする．
```
ターゲット：y ~ 一様{±1}，x₁ ~ D_tg,1(·|y)，x₂ ~ N(0, Σ₂)
L(w) = E_{x∼D_tg} ℓ_exp(wᵀx)，ℓ_exp(t) = exp(−|t|)
w₀ = w_s，w_{t+1} = (w_t − η∇L(w_t)) / ‖w_t − η∇L(w_t)‖₂
定理 3.1：分離の仮定 3.1 と「ソースの分類器がそれなりに正確」の仮定 3.2 のもとで，O(log(1/ε)) 回の後 ‖ŵ₂‖₂ ≤ ε
```

**分かったこと．**表 1（ターゲットの正解率）：CelebA 81% から 88%，色つき MNIST の 10 クラス（相関 0.95）82% から 91%．相関 0.97 で最初の正解率が 72% のときは 67% に下がった．著者はこれを「ソースの分類器がそれなりに正確」の仮定の必要性の裏づけとする．例 3.1・3.2：w₁ = 0 などの場合，エントロピー最小化は見かけの特徴への依存をむしろ強める．

**前提と限界．**著者の記述：成功の条件として「good separation between classes and a decently accurate source classifier」（6 節）．見かけの特徴はガウス，本当の特徴は対数凹の分布の混合という仮定である．

**MER-DA との接点**（推測．検定していない）．コーパスに固有の手がかり（録音の条件，話者の偏り，台本の語）は，ソースではラベルと相関しターゲットでは相関しない特徴にあたりうる．定理はソースの分類器がターゲットで「それなりに正確」なことを要求するので，ソースのみの UAR が低い組では自己学習が見かけの特徴を強める側に働く可能性がある．

**確認の水準と出典．**◎（1〜3 節，表 1，6 節）．https://proceedings.neurips.cc/paper_files/paper/2020/hash/f1298750ed09618717f9c10ea8d1d3b0-Abstract.html ，https://arxiv.org/abs/2006.10032

### Wei, Shen, Chen, Ma 2021（ICLR 2021．arXiv 2010.03622）Theoretical Analysis of Self-Training with Deep Networks on Unlabeled Data

**問い．**自己学習の理論は線形の模型に限られていた．深い網でも，疑似ラベルで学ぶと疑似ラベルの誤りが直る理由を説明できるか．

**方法．**2 つの仮定を置く．expansion：各クラスの分布 P_i で，確率が a 以下の集合 V の近傍 N(V)（データ拡張で届く点を含む）は，V の確率の c 倍以上の確率を持つ．separation：違うクラスの近傍はほとんど重ならない（入力の一貫性の罰則 R_B の真の分類器での値が μ 以下）．疑似ラベルとの 0-1 損失と入力の一貫性の罰則を重みづけて足した目的を最小にする．
```
(a, c)-expansion：P_i(V) ≤ a ⇒ P_i(N(V)) ≥ min{c P_i(V), 1}
目的（4.1）：min_G L(G) = (c+1)/(c−1) · L_0-1(G, G_pl) + 2c/(c−1) · R_B(G) − Err(G_pl)
定理 4.3：Err(Ĝ) ≤ 2/(c−1) · Err(G_pl) + 2c/(c−1) · μ
  G_pl：疑似ラベルを作る分類器．仮定 4.1：各クラスで疑似ラベルの誤りの割合 ā < 1/3，c > 3
```

**分かったこと．**仮定のもとで，疑似ラベルで学んだ分類器の誤差は疑似ラベルの誤差の 2/(c−1) 倍まで下がる（c > 3 なら疑似ラベルより良い）．一般化の上界と組み合わせて有限の標本の保証にした（定理 C.2）．実データの数値の表は主な結果ではない．

**前提と限界．**著者の記述：「an interesting question for future work is to extend our assumptions to analyze domain adaptation algorithms based on aligning the source and target」（6 節）．c > 3 の要求は教師なし学習の場合（c > 1）より厳しい（4 節）．

**MER-DA との接点**（推測．検定していない）．凍結した特徴の上では，近傍 N(V) を特徴の空間での小さな球で定めることになる．定理の形を使うと，積の疑似ラベルの誤りの割合と，感情ごとの特徴の近傍の広がり（c）を測れば，疑似ラベルで学ぶ得の上限を見積もれる．

**確認の水準と出典．**◎（1〜4 節，6 節．付録の証明は読んでいない）．https://arxiv.org/abs/2010.03622 （arXiv の注記 "Published at ICLR 2021"），https://iclr.cc/virtual/2021/poster/2547

### Cai, Gao, Lee, Lei 2021（ICML 2021，PMLR 139:1170–1182．arXiv 2102.11203）A Theory of Label Propagation for Subpopulation Shift

**問い．**クラスの中の下位の集団（例：同じ「犬」の中の別の犬種）の割合や中身がソースとターゲットで違うずれ（subpopulation shift）で，ソースで学んだ分類器からラベルをターゲットへ広げられるか．

**方法．**ソースとターゲットの台を互いに素な下位の集団 S_i，T_i の和とし，各集団の中では真のクラスが一定とする．疑似ラベルを作る分類器（論文の語で teacher classifier）g_tc はソースの各集団で真のクラスをマージン γ で多く当てる．入力の一貫性の罰則が μ 以下の分類器の中で，ソースで g_tc と最も一致するものを選ぶ．
```
仮定 1：P_{x∼S_i}[g_tc(x) = y_i] ≥ P_{x∼S_i}[g_tc(x) = k] + γ（k ≠ y_i），P_T[T_i] / P_S[S_i] ≤ r
g = argmin_{g ∈ G} P_{x∼S}[g(x) ≠ g_tc(x)]  s.t. R_B(g) ≤ μ
定理 2.1：½(S + T) が (½, c)-multiplicative expansion を満たすとき，ε_T(g) ≤ max( (c+1)/(c−1), 3 ) · 8rμ/γ
```

**分かったこと．**BREEDS の ENTITY-30（表 1，ターゲットの正解率）：ソースのみ 56.73，DANN 61.03，MDD 63.95，SwAV の特徴の上の FixMatch 72.60．Office-31 で MDD に FixMatch を足すと平均 89.16 から 89.84（表 2）．

**前提と限界．**著者の記述（脚注）：Wei ほかの改訂版でも「Still, a mostly correct pretrained pseudo-labeler is required」．こちらの読み：ターゲットの下位の集団の割合がソースの r 倍を超えないことと，一貫性の罰則を作るデータ拡張を仮定している．

**MER-DA との接点**（推測．検定していない）．コーパスの違いを「同じ感情の中の下位の集団（演技か自然か，話者，場面）の割合の違い」とみなすと，この枠に入る．単独の分類器の積を g_tc とみなすと，γ（ソースでの当たりの余裕）と r（下位の集団の割合の比）が得の上限を決める量になる．

**確認の水準と出典．**◎（1〜2 節，4 節，表 1〜3，5 節．付録の証明は読んでいない）．https://proceedings.mlr.press/v139/cai21b.html ，https://arxiv.org/abs/2102.11203

**F. 総説**

### Wang & Deng 2018（Neurocomputing 312:135–153．arXiv 1802.03601）Deep Visual Domain Adaptation: A Survey，および van Engelen & Hoos 2020（Machine Learning 109(2):373–440）A survey on semi-supervised learning

**問い．**Wang & Deng：深い網の DA の手法を分類して整理する．van Engelen & Hoos：半教師あり学習の手法を，古典から深い網まで分類して整理する．

**方法．**Wang & Deng は DA を，特徴の空間が同じか（homogeneous／heterogeneous），ターゲットのラベルの有無（教師あり・半教師あり・教師なし），ソースとターゲットの距離（1 段の DA か，中間のドメインを経る多段の DA か）で分け，1 段の手法を不一致に基づく手法・敵対的な手法・再構成に基づく手法に分ける．van Engelen & Hoos は半教師ありの分類の手法の新しい分類を提案する（抄録）．

**分かったこと．**Wang & Deng の 2 節の定義：半教師あり DA は「both limited labeled data, D_tl, and redundant unlabeled data, D_tu, in the target domain are available in the training stage」．結論（7 節）で，教師あり DA はラベルが要るため限られ，これまでの研究は教師なし DA に集中していると書く．

**前提と限界．**Wang & Deng は 2018 年の時点の総説で，SSDA の標準の評価設定（MME，2019）より前である．van Engelen & Hoos は抄録と書誌だけを確かめた．

**MER-DA との接点**（推測．検定していない）．用語の定義（教師あり DA，半教師あり DA）を論文で書くときの出典に使える．

**確認の水準と出典．**Wang & Deng は ◎（抄録，2 節の定義，7 節の結論のみ）．van Engelen & Hoos は ○（Crossref の書誌と抄録）．https://doi.org/10.1016/j.neucom.2018.05.083 ，https://arxiv.org/abs/1802.03601 ，https://doi.org/10.1007/s10994-019-05855-6

---

## 3. 分野で標準とされていることと，未解決と書かれていること

### 3.1 標準とされていること

1. **評価の設定．**UDA は Office-31（Saenko 2010 由来），Office-Home，VisDA，DomainNet で，ターゲットのラベル無しの標本を全部学習に使い，同じ標本で評価する（DANN の表 3 の見出し "fully-transductive" protocol）．SSDA は MME の設定（DomainNet の 126 クラス・4 ドメイン・7 組，クラスあたり 1 本と 3 本，Office-Home の 3 本）を APE，CDAC，DECOTA，CLDA，SLA がそのまま使う．比べる相手は S+T，DANN，ADR，CDAN，ENT，MME の値を前の論文から写すことが多い（APE の表 1 は MME の表 1 と全セルで一致．既存ノートの照合）．
2. **つまみの選び方．**SSDA の主な論文はターゲットのクラスあたり 3 本の別の検証用ラベルでつまみを選ぶ（MME，DECOTA）か，1 組の検証の成績で選んで全実験に固定する（MME，DECOTA，CLDA は Office-Home の Product→Art）．UDA ではターゲットのラベルを使わない選び方として逆検証（DANN），重要度重みつきの交差検証（CDAN），ソースのラベルと 2 標本の判別器（DAN）が使われる．半教師あり学習では，学習のラベルより大きな検証用データで選ぶ慣行があり（Lee 2013 は学習 100 本に対し検証 1,000 本），Oliver ほか 2018 がこれを批判した．
3. **ラベル無しの損失の型．**エントロピー最小化（Grandvalet & Bengio 2005），確信度の閾値つきの硬い疑似ラベル（Lee 2013，FixMatch の τ=0.95），一貫性正則化（Mean Teacher，FixMatch），クラスの釣り合いの補正（CBST のクラスごとの閾値，SHOT の多様性の項，AdaMatch の分布合わせ）の 4 つが部品として組み合わされる．
4. **データ拡張への依存．**2019 年以降の強い手法（MixMatch，FixMatch，CDAC，CLDA，AdaMatch）は画像の弱い拡張と強い拡張を前提にする．理論（Wei 2021，Cai 2021）も，一貫性の罰則の近傍をデータ拡張で定める．
5. **ドメインのずれの理論の出発点．**Ben-David ほか 2010 の上界を手法の動機に引く（DAN の定理 1，DANN の 3 節，MCD の 3.4 節，MME の 3.3 節）．

### 3.2 著者たちが未解決と書いていること

- **ずれの量の上界の改良と，複数ソースの組み合わせ（Ben-David 2010）**：「We do not yet know whether our divergence measure admits tighter data-dependent bounds」，「it would be interesting to investigate algorithms that choose a convex combination of multiple sources to minimize the bound」（結論）．
- **見方の仮定の現実性（Blum & Mitchell 1998）**：「Our theoretical model is clearly an over-simplification of real-world target functions and distributions」（7 節）．
- **模型の仮定が崩れたときのラベル無しデータの損（Nigam 2000）**：「the use of unlabeled data will hurt instead of help performance」と書き，ラベル無しデータの利用と模型の不適合の兼ね合いの理論と実験を今後の課題とする（8 節）．
- **疑似ラベルの誤り（Zhou & Li 2005）**：「the unlabeled examples may often be wrongly labeled during the learning process」．誤ったラベルの標本を見つける仕組みを足すことを今後の課題とする（5 節）．
- **SSDA での DANN の検証（Ganin & Lempitsky 2015）**：「thorough verification of semi-supervised setting is left for future work」（5 節）．
- **どの層から先を適応させるか（DAN 2015）**：「it is interesting to study the principled way of deciding the boundary of generality and specificity」（5 節）．
- **目標の質（Mean Teacher 2017）**：「The success of consistency regularization depends on the quality of teacher-generated targets」（5 節）．
- **現実的な検証用データ（Oliver 2018）**：「SSL methods which rely on heavy hyperparameter tuning on a large validation set may have limited real-world applicability」（4.6 節）．ラベル無しの標本に別のクラスが混ざると「can actually hurt performance compared to not using any unlabeled data at all」（4.4 節）．
- **実装の違い（FixMatch 2020）**：「the same technique can not always be directly compared across different implementations」（6 節）．
- **自然なずれでのテスト時の適応（Tent 2021）**：「Although error is higher on both sets, indicating the presence of shift, tent does not improve generalization」．更新するパラメータの選び方：「the issue is to identify parameters that are both expressive and reliable」（6 節）．
- **段階的な DA（Kumar 2020）**：「developing and analyzing new techniques for gradual domain adaptation is an exciting avenue for future work」（付録）．定理はクラスの割合が変わらないことを仮定する．
- **自己学習の理論の DA への拡張（Wei 2021）**：「extend our assumptions to analyze domain adaptation algorithms based on aligning the source and target」（6 節）．Cai 2021 の脚注：「Still, a mostly correct pretrained pseudo-labeler is required」．
- **ラベル無しデータの使い方とずれの模型化のどちらが大事か（AdaMatch 2022）**：「which is more important for future progress, making use of unlabeled data more efficiently, or attempting to better model domain shifts?」（7 節）．
- **較正の崩れ（CLDA 2021）**：「deep neural networks face the problem of miscalibration ... which may result in images being pushed into wrong clusters」（付録 D）．
- **ソースのデータの使い方（SLA 2023）**：「Our work highlights that the usage of source data should be revisited carefully」（5 節）．

### 3.3 MER-DA の設定から見た読み（こちらの推測で，検定していない）

- 2019 年以降の SSDA の強い手法の多くはデータ拡張と特徴抽出器の学習を前提にするので，凍結したエンコーダの上の線形の分類器にはそのまま移せない．そのまま移せる部品は，エントロピー最小化，硬い疑似ラベルと閾値，クラスごとの閾値（CBST），多様性の項（SHOT），相対閾値（AdaMatch），MixUp（特徴どうし），CORAL，ソースの重みへの罰則（Daumé の PRIOR，Donahue の PMT-SVM），重心によるソースのラベルの書き直し（SLA）である．
- 自己学習の理論の 4 本（Kumar 2020，Chen 2020，Wei 2021，Cai 2021）はどれも「疑似ラベルを作る分類器がターゲットでそれなりに正確」なことを仮定する．MER-DA の積の疑似ラベルの正解率と，その感情ごとの内訳を測ることが，どの理論の前提が成り立つかを見る最初の確認になる．
- Kumar 2020 と AdaMatch の分布合わせはクラスの割合が変わらないことを仮定するか，ソースの割合に寄せる．感情の割合がコーパスで違う MER-DA では，この仮定が成り立たない（`docs/related-work.md` 2.7 節の Zhao 2019 の記録と同じ論点）．
- 疑似ラベルを使う手法の得はターゲットのラベルが増えると縮む（APE の補足資料，DECOTA の補足資料，Nigam の WebKB）．MER-DA の K=25 での得の有無は，この流れと並べて書ける．

---

## 4. 出典の一覧

確認の水準つき．URL は本文を読んだ版か書誌を確かめた頁．

| # | 論文 | 会場・書誌 | 水準 | URL |
|---|---|---|---|---|
| 1 | Ben-David ほか 2010 | Machine Learning 79(1–2):151–175 | ◎ | https://doi.org/10.1007/s10994-009-5152-4 |
| 2 | Daumé III 2007 | ACL 2007, pp. 256–263 | ◎ | https://aclanthology.org/P07-1033.pdf |
| 3 | Saenko, Kulis, Fritz, Darrell 2010 | ECCV 2010, LNCS, pp. 213–226 | ◎ | https://doi.org/10.1007/978-3-642-15561-1_16 |
| 4 | Blum & Mitchell 1998 | COLT 1998, pp. 92–100 | ◎ | https://doi.org/10.1145/279943.279962 |
| 5 | Zhou & Li 2005 | IEEE TKDE 17(11):1529–1541 | ◎ | https://doi.org/10.1109/TKDE.2005.186 |
| 6 | Nigam, McCallum, Thrun, Mitchell 2000 | Machine Learning 39(2–3):103–134 | ◎ | https://doi.org/10.1023/A:1007692713085 |
| 7 | Grandvalet & Bengio 2005 | NIPS 17, pp. 529–536 | ◎ | https://proceedings.neurips.cc/paper_files/paper/2004/file/96f2b50b5d3613adf9c27049b2a888c7-Paper.pdf |
| 8 | Lee 2013 | ICML 2013 Workshop WREPL | ◎ | https://storage.googleapis.com/kaggle-forum-message-attachments/8924/pseudo_label_final.pdf |
| 9 | Tarvainen & Valpola 2017 | NIPS 2017, pp. 1195–1204 | ◎ | https://arxiv.org/abs/1703.01780 |
| 10 | Berthelot ほか 2019 MixMatch | NeurIPS 2019, pp. 5049–5059 | ◎ | https://arxiv.org/abs/1905.02249 |
| 11 | Sohn ほか 2020 FixMatch | NeurIPS 2020, pp. 596–608 | ◎ | https://arxiv.org/abs/2001.07685 |
| 12 | Oliver ほか 2018 | NeurIPS 2018 | ◎（一部） | https://arxiv.org/abs/1804.09170 |
| 13 | Long ほか 2015 DAN | ICML 2015, PMLR 37:97–105 | ◎ | https://proceedings.mlr.press/v37/long15.html |
| 14 | Ganin & Lempitsky 2015 | ICML 2015, PMLR 37:1180–1189 | ◎ | https://proceedings.mlr.press/v37/ganin15.html |
| 15 | Ganin ほか 2016 DANN | JMLR 17(59):1–35 | ◎ | https://jmlr.org/papers/v17/15-239.html |
| 16 | Sun, Feng, Saenko 2016 CORAL | AAAI 2016（30(1)） | ◎ | https://doi.org/10.1609/aaai.v30i1.10306 |
| 17 | Sun & Saenko 2016 Deep CORAL | ECCV 2016 Workshops, LNCS, pp. 443–450 | ◎ | https://doi.org/10.1007/978-3-319-49409-8_35 |
| 18 | Tzeng ほか 2017 ADDA | CVPR 2017, pp. 2962–2971 | ◎ | https://doi.org/10.1109/CVPR.2017.316 |
| 19 | Long ほか 2018 CDAN | NeurIPS 2018, pp. 1640–1650 | ◎ | https://arxiv.org/abs/1705.10667 |
| 20 | Saito ほか 2018 MCD | CVPR 2018, pp. 3723–3732 | ◎ | https://doi.org/10.1109/CVPR.2018.00392 |
| 21 | Zou ほか 2018 CBST | ECCV 2018, LNCS, pp. 297–313 | ◎ | https://doi.org/10.1007/978-3-030-01219-9_18 |
| 22 | Liang, Hu, Feng 2020 SHOT | ICML 2020, PMLR 119:6028–6039 | ◎ | https://proceedings.mlr.press/v119/liang20a.html |
| 23 | Wang ほか 2021 Tent | ICLR 2021 | ◎ | https://arxiv.org/abs/2006.10726 |
| 24 | Donahue ほか 2013 | CVPR 2013, pp. 668–675 | ◎（既存ノート） | https://openaccess.thecvf.com/content_cvpr_2013/html/Donahue_Semi-supervised_Domain_Adaptation_2013_CVPR_paper.html |
| 25 | Motiian ほか 2017 FADA | NIPS 2017, pp. 6670–6680 | ◎（既存ノート） | https://proceedings.neurips.cc/paper_files/paper/2017/hash/21c5bba1dd6aed9ab48c2b34c1a0adde-Abstract.html |
| 26 | Motiian ほか 2017 CCSA | ICCV 2017, pp. 5716–5726 | ◎（既存ノート） | https://openaccess.thecvf.com/content_iccv_2017/html/Motiian_Unified_Deep_Supervised_ICCV_2017_paper.html |
| 27 | Saito ほか 2019 MME | ICCV 2019, pp. 8049–8057 | ◎（既存ノート） | https://openaccess.thecvf.com/content_ICCV_2019/html/Saito_Semi-Supervised_Domain_Adaptation_via_Minimax_Entropy_ICCV_2019_paper.html |
| 28 | Kim & Kim 2020 APE | ECCV 2020, LNCS 12359, pp. 591–607 | ◎（既存ノート） | https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590579.pdf |
| 29 | Li ほか 2021 CDAC | CVPR 2021, pp. 2505–2514 | ◎（既存ノート） | https://openaccess.thecvf.com/content/CVPR2021/html/Li_Cross-Domain_Adaptive_Clustering_for_Semi-Supervised_Domain_Adaptation_CVPR_2021_paper.html |
| 30 | Yang ほか 2021 DECOTA | ICCV 2021, pp. 8886–8896 | ◎（既存ノート） | https://openaccess.thecvf.com/content/ICCV2021/html/Yang_Deep_Co-Training_With_Task_Decomposition_for_Semi-Supervised_Domain_Adaptation_ICCV_2021_paper.html |
| 31 | Singh 2021 CLDA | NeurIPS 2021, pp. 5089–5101 | ◎ | https://arxiv.org/abs/2107.00085 |
| 32 | Berthelot ほか 2022 AdaMatch | ICLR 2022 | ◎ | https://arxiv.org/abs/2106.04732 |
| 33 | Yu & Lin 2023 SLA | CVPR 2023, pp. 24100–24109 | ◎ | https://doi.org/10.1109/CVPR52729.2023.02308 |
| 34 | Kumar, Ma, Liang 2020 | ICML 2020, PMLR 119:5468–5479 | ◎ | https://proceedings.mlr.press/v119/kumar20c.html |
| 35 | Chen, Wei, Kumar, Ma 2020 | NeurIPS 2020, pp. 21061–21071 | ◎ | https://arxiv.org/abs/2006.10032 |
| 36 | Wei, Shen, Chen, Ma 2021 | ICLR 2021 | ◎（証明は未読） | https://arxiv.org/abs/2010.03622 |
| 37 | Cai, Gao, Lee, Lei 2021 | ICML 2021, PMLR 139:1170–1182 | ◎（証明は未読） | https://proceedings.mlr.press/v139/cai21b.html |
| 38 | Wang & Deng 2018 | Neurocomputing 312:135–153 | ◎（2・7 節のみ） | https://doi.org/10.1016/j.neucom.2018.05.083 |
| 39 | van Engelen & Hoos 2020 | Machine Learning 109(2):373–440 | ○ | https://doi.org/10.1007/s10994-019-05855-6 |

表の 39 行は論文 1 本ずつである．2 節では 5 組を 1 つの見出しにまとめたので，見出しは 34 個である．

### 外した論文と，確かめられなかったこと

- **BiAT**（Jiang, Wu, Han, Shao, Qi, Li．IJCAI 2020, pp. 934–940．DOI 10.24963/ijcai.2020/130．○）：DomainNet の SSDA の比べる相手として CDAC と CLDA の表に出る（AlexNet の 1 本／3 本の平均 45.5／49.4．CLDA の表 2）．MME 系の他の手法より引用が少ないと判断して見出しを立てなかった．本文は読んでいない．
- **MCL**（Yan, Wu, Li, Qin, Han, Cui．IJCAI 2022．arXiv 2205.04066．○）：SLA の表 4 で DomainNet の 1 本／3 本の平均 74.4／76.5．SLA の著者は MCL の報告値を再現できなかったと書く（SLA の 4.3 節）．本文は読んでいない．
- **UniSSDA**（Zhang, Liu, Cong, Ragab, Foo．CVPR 2024．arXiv 2403.11234．○）：ターゲットのラベル空間がソースと一致しない場合の SSDA．検索の要約では DomainNet-345 で凍結した ViT の基盤モデルを使う実験があるとされたが，本文で確かめていない．
- **基盤モデルの凍結した特徴を使う SSDA**：2023 年以降で広く引かれる論文は，2 回の検索（"semi-supervised domain adaptation CLIP foundation model frozen features"，"semi-supervised domain adaptation foundation models vision-language"）では見つからなかった．見つかったのは画像の分割（SemiDAViL，arXiv 2504.06389）やソースフリー（DIFO，arXiv 2311.16510）の研究で，分類の SSDA ではない．
- 確かめていない書誌：CORAL の AAAI 版の頁番号，CBST と Deep CORAL の LNCS の巻番号，Oliver 2018 の論文集の頁，Tent と Wei 2021 の OpenReview の頁（会場は arXiv の注記と ICLR の頁で確かめた）．
- 読んでいない部分：Wei 2021 と Cai 2021 の付録の証明，各論文の図だけにある数値，DECOTA の表の一部（既存ノートの記録のまま）．
