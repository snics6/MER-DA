# 先行研究：ラベル無しの発話を全体の平均と全体の共分散（モーメント）にだけ使う半教師ありの LDA（実験 10 の下調べの位置づけ．2026-11-08）

報告ファイルは書いていない（この環境の規則で，報告の .md を子が書くことが禁じられている）．以下が全文である．本体が `research_notes/先行研究_モーメントだけを使う半教師ありLDA_2026-11-08/report.md` に保存する．

## 要約

1. 新規性の判定：ラベル無しの標本を全体の平均と全体の共分散の推定にだけ使い，少数のラベルで作るクラスの平均と組み合わせる LDA は既に知られている．Loog 2010・2012・2014 の moment constraint，Normalized LDA（Fan ほか 2008），Shaffer 1991 型の「2 次のモーメントを差し替える最小二乗」がこれに当たる．凍結した埋め込みで，ラベル無しのターゲットの全体の共分散だけを使ってガウスの判別を適応する方法も，話者照合で使われている（Kaldi の PLDA 適応，CORAL+ 2019）．
2. 最も近い先行研究は Krijthe & Loog 2016（S+SSPR．peaking）である．この論文は，ラベル無しの標本を全体の散らばり T の推定にだけ使う分類器 w = T⁻¹(m₁ − m₂) を調べた．得を「共分散を良くした分」と「平均を良くした分」に分けて測り（図 3），クラスが大きく離れているとラベル無しが無限にあっても教師ありより悪いことを示した（図 4）．これが実験 10 の「計量の分と割り当ての分に分ける」測り方の先例になる．
3. 論文の比べ方と食い違う点が 2 つある．(a) MCLDA の誤り率は，2 クラスでは 10 データセットすべてで教師ありの LDA より低かった（Krijthe & Loog 2014 の表 II）．ところが 3 クラス以上では，同じ推定量が 8 データセットのうち 6 で教師ありより悪かった（Loog 2016 の表 5．optical で 0.161 から 0.900）．(b) MER-DA の 2026-10-17 の探索（`exp/explore_metric_room.py`）で，実験 10 の中心の形（K 本の平均の差を，ラベル無しの全体の共分散で縮めた計量で白色化する）を既に測っている．このとき α = 0.5 に当たる強さは，等方（白色化しない形）より大きく悪かった．
4. 狙うべき差分は 4 つある．(1) 割り当ての分（EM が平均と割合を動かす分）の符号を，計量の分と同じ単位・同じ分解で測る．先例の Krijthe & Loog は，平均の分を真のラベルで作った．(2) ラベル付きの本数が次元よりずっと少ない領域（4K ≤ 100 本に対し 2,304 次元）で測る．先行の MCLDA の実験は，どれもラベル付きを 2d 本以上にしている．(3) モダリティをまたぐ共分散のブロックが計量の分にどれだけ寄与するかを測る．(4) 候補の感情の割合が偏っているとき，平均の制約がどれだけ偏りを入れるかを測る．
5. 新しく書けないことは 3 つある．全体の共分散を判別の計量に使えること，ラベル無しの標本で全体のモーメントを推定し直せること，その得を共分散の分と平均の分に分けられることである．

---

## 0. 用語

- **モーメント**：分布の平均（1 次）と共分散（2 次）のこと．この報告の「全体のモーメント」は，ラベルに依らない候補全体の平均と共分散を指す．
- **全体の散らばり T，群内の散らばり W，群間の散らばり B**：T はラベルを無視した全標本の共分散，W は各クラスの中の共分散の平均，B はクラスの平均どうしの共分散である．どれも同じ標本と同じラベルから作れば T = W + B が成り立つ（Fukunaga 1990．Loog 2016 の式 (5) が引く）．
- **moment constraint（モーメントの制約）**：ラベル無しの標本で精度よく推定した全体の平均と全体の共分散に合うように，ラベルを要する推定（クラスの平均と群内の共分散）を直す手続きである（Loog）．
- **USM（updated second moment）**：最小二乗の分類器 w = (XᵀX)⁻¹Xᵀy で，XᵀX だけをラベル付きとラベル無しの全部から推定し直す形である（RSSL の名前．Shaffer 1991 を引く）．
- **peaking**：学習の本数を増やすと，誤り率がいったん上がってから下がる現象である．
- **PLDA**：話者照合で埋め込みの後に置くガウスの判別のモデルである．群間（話者）と群内（録音条件）の共分散を持つ．

確認の水準は次の 3 つである．◎ は本文の該当箇所か実装のコードを読んだもの．○ は要旨か，本文を読んだ二次資料の記述によるもの．△ は検索の断片だけによるもの．

---

## 1. 問いの分け方

| 番号 | 問い | 調べた先 |
|---|---|---|
| Q1 | MCLDA はラベル無しから何を推定し，どう制約するか．正定値でなくなる場合，多クラス，数値，条件はどうか | Loog 2010・2012・2014，Loog & Jensen 2012，Krijthe & Loog 2014，Loog 2016 の表 5，RSSL の実装 |
| Q2 | ラベルに依らない全体の共分散を判別の計量に使う先行研究と，その数値 | Normalized LDA，USM（Shaffer 1991），Krijthe & Loog 2016，Simple CNAPS，Transductive CNAPS，Kaldi の PLDA 適応，CORAL+，GDA（CLIP），Vidaurre 2011 |
| Q3 | 2 次のモーメントだけを使うと悪くならないか（理論） | Azriel ほか 2022，Chakrabortty & Cai 2018，Shaffer 1991，Krijthe & Loog 2016 の図 4（MCPL と射影推定量は前回のノートにあるので重ねない） |
| Q4 | 実験 10 の案との差分．「割り当ての分と計量の分に分ける」先例はあるか | 6〜9 節 |

---

## 2. Q1：MCLDA の定義（論文の主張）

### 2.1 系譜

- **Loog 2010**（ECML PKDD，LNAI 6322，pp. 291–304）○：最近傍平均の分類器に平均の制約 μ̂_t = Σ_k π̂_k μ̂_k を課す．RSSL の実装（`MCNearestMeanClassifier.R`）◎ は，クラスの平均すべてに（全部の標本の平均 − ラベル付きの平均）を足す．
- **Loog 2012**（PSL 2011，LNCS 7081，pp. 32–41．「Semi-supervised linear discriminant analysis using moment constraints」）○：要旨だけを読んだ（Springer Professional の頁）．要旨は，分布の仮定を足さず，ラベルに依る量とラベルに依らない量の関係を使ってラベルに依る推定の揺れを減らすと書き，ラベル無しが増えても「typically does not deteriorate」と書く．本文は有料で読めなかった．
- **Loog & Jensen 2012**（S+SSPR，LNCS 7626，pp. 327–335）○：Krijthe & Loog 2014 の記述によれば，式 (5)(6) の制約を満たすようにラベル付きの尤度を最大にする．目的は非凸で，数値で解く．本文は読んでいない．
- **Loog 2014**（Pattern Recognition Letters 37:24–31，DOI 10.1016/j.patrec.2013.03.004）：本文は読めなかった（ScienceDirect は 403，Unpaywall では閉じた論文）．定義は次の 2 つから取った．

### 2.2 Krijthe & Loog 2014（ICPR，arXiv 1411.4521）III-D 節の定義 ◎

2 クラスで書かれている．

```
制約：  μ_t = π₁μ₁ + π₂μ₂                       (5)
        Σ_t = Σ + Σ_b                           (6)
更新：  μ̂_c^MC = μ̂_c − ( Σ_j π̂_j μ̂_j − μ̂_t )      (7)
        Σ̂^MC   = Θ̂^{1/2} Σ̂_t^{−1/2} Σ̂ Σ̂_t^{−1/2} Θ̂^{1/2}   (8)
```

- 原文の定義では，μ̂_t と Θ̂ は「labeled and unlabeled data」の全部で推定した全体の平均と全体の共分散である．Σ̂_t は「the overal covariance estimated on the labeled data alone」，Σ̂ はラベル付きだけの群内の共分散である．
- 式 (8) の指数は PDF からの文字の取り出しで符号が落ちた．±1/2 の並びは，制約 L Σ̂_t Lᵀ = Θ̂ を満たす L = Θ̂^{1/2} Σ̂_t^{−1/2} と合うように読んだ．
- 原文は (7)(8) を「An ad hoc way to update the parameters」と呼ぶ．Loog & Jensen 2012 の制約つき尤度の版は「more formally」だが，本論文は単純な ad hoc の版を使うと書く．

### 2.3 RSSL の実装（Krijthe の R パッケージ．`MCLinearDiscriminantClassifier.R`）◎

https://github.com/jkrijthe/RSSL/blob/master/R/MCLinearDiscriminantClassifier.R

- 文書の記述では，`method="invariant"` が Loog 2014 の尺度に依らない（scale-invariant）更新，`method="closedform"` が Loog 2012 の尺度に依る更新である．
- invariant の版は次のとおりである．T_all と T_labeled の一般化固有値問題で E と D を求め，Trans = E⁻ᵀ D^{1/2} Eᵀ とする．
  ```
  μ_c ← m_all + Trans (μ_c − m_labeled)
  Σ   ← Trans Σ Transᵀ
  ```
  この Trans は Trans · T_labeled · Transᵀ = T_all を満たす（本調べで式を追って確かめた）．したがってこの版は，ラベル付きの標本を「ラベル付きの散らばりで白色化し，全体の散らばりで色づけし直す」アフィン写像で動かしてから，教師ありの LDA を解くことと同じである．これは CORAL をラベル付きの集合から全体の集合へ当てる形にあたる（本調べの読み）．
- 印刷された式 (7) との食い違い：式 (7) は平均を平行移動するだけだが，実装は平均にも Trans を掛ける．closedform の実装は Σ を T_lab^{−1/2} T_all^{1/2} Σ T_all^{1/2} T_lab^{−1/2} と書いており，式 (8) と転置の位置が違う．どちらが 2012・2014 年の原文どおりかは，原文を読めず確かめられなかった．
- 正定値の扱い：この掛け算の形なら，Σ が正定値で T_labeled が逆にできる限り，Σ^MC も正定値である．一方，T_labeled が特異（ラベル付きの本数 ≤ 次元）だと，一般化固有値問題が退化する．実装は SVD の擬似逆で処理するが，この領域の検査や報告は無い．
- 多クラス：実装は事前確率を `matrix(colMeans(Y),2,1)` と 2 クラスに固定している．

### 2.4 数値と著者の条件

**Krijthe & Loog 2014 の表 II（10 分割の交差検証の誤り率，20 回．ラベル付きは max(2d, 10) 本，残りはラベル無し）◎**

| データ（標本数，次元） | 教師ありの LDA | 全ラベル | MCLDA | EMLDA | 自己学習 | ICLDA |
|---|---|---|---|---|---|---|
| Haberman (305, 4) | 0.37 | 0.25 | 0.36 | 0.47 | 0.36 | 0.37 |
| Ionosphere (351, 33) | 0.21 | 0.15 | 0.18 | 0.57 | 0.20 | 0.18 |
| Parkinsons (195, 20) | 0.27 | 0.15 | 0.22 | 0.41 | 0.26 | 0.23 |
| Pima (768, 9) | 0.34 | 0.23 | 0.32 | 0.37 | 0.35 | 0.31 |
| Sonar (208, 61) | 0.29 | 0.26 | 0.28 | 0.35 | 0.29 | 0.28 |
| SPECT (265, 23) | 0.31 | 0.18 | 0.25 | 0.62 | 0.33 | 0.30 |
| SPECTF (265, 45) | 0.32 | 0.24 | 0.28 | 0.28 | 0.34 | 0.33 |
| Transfusion (748, 4) | 0.34 | 0.23 | 0.32 | 0.52 | 0.37 | 0.33 |
| WDBC (568, 30) | 0.11 | 0.04 | 0.09 | 0.38 | 0.09 | 0.08 |
| BCI (400, 118) | 0.21 | 0.16 | 0.20 | 0.21 | 0.21 | 0.20 |

- MCLDA は 10 データセットすべてで教師ありより誤り率が低い．著者は「overal in terms of error rate MCLDA seems to perform best」と書く．
- 表 III（テストの負の対数尤度）では，ICLDA が最も良いデータが多く，教師ありの LDA は桁で悪い（Ionosphere 199.58 に対し MCLDA 25.93，全ラベル 21.38）．
- どのデータでもラベル付きは 2d 本以上であり，T_labeled は正則である．

**Loog 2016（TPAMI．arXiv 1503.00269）の表 5：同じ ad hoc の MCLDA（θ̂_hoc．参照 [31] = Loog 2014）を多クラスにも当てた結果 ◎**

条件はラベル付き N = 2d + K 本で，次元は PCA で分散の 99.9% を残した．試行は 1,000 回である．テストの誤り率を教師ありと並べる．

| データ（クラス数） | 教師あり ε_sup | MCLDA ε_hoc | MCLDA が教師ありより良かった試行の割合 |
|---|---|---|---|
| banknote (2) | 0.061 | 0.087 | 30.1% |
| climate (2) | 0.150 | 0.117 | 79.9% |
| first-order (6) | 0.666 | 0.626 | 96.8% |
| gas (6) | 0.141 | 0.145 | 44.7% |
| landsat (6) | 0.291 | 0.308 | 29.8% |
| letter (26) | 0.618 | 0.625 | 33.2% |
| low (10) | 0.763 | 0.744 | 74.9% |
| magic (2) | 0.317 | 0.292 | 74.0% |
| miniboone (2) | 0.246 | 0.218 | 73.1% |
| optical (10) | 0.161 | 0.900 | 0.0% |
| pen-based (10) | 0.280 | 0.299 | 24.5% |
| qsar (2) | 0.257 | 0.229 | 86.9% |
| shuttle (7) | 0.134 | 0.822 | 1.6% |
| skin (2) | 0.098 | 0.102 | 40.1% |
| spambase (2) | 0.195 | 0.310 | 51.3% |
| spectf (2) | 0.325 | 0.293 | 71.0% |

- 3 クラス以上の 8 データセットでは，MCLDA が教師ありより良かったのは 2 つ（first-order，low），悪かったのは 6 つである．2 クラスの 8 データセットでは，良かったのが 5 つ，悪かったのが 3 つ（skin は有意差なし）である．
- 著者の記述（6.4 節）：ad hoc の手続きは「does very bad on optical and shuttle (the reason for this remains as yet unclear)」．それでも，テストの誤り率では 16 のうち 7 データセットで最良だった．尤度では，大きな改善を「the averaged class-conditional covariance matrix Σ is much more stably estimated」によると読む．
- 実験 10 の設計の 0 節は「MCLDA は誤り率で最も良かった」と Krijthe & Loog 2014 だけを引いている．多クラスでの表 5 の悪化も並べて書く必要がある．

---

## 3. Q2：ラベルに依らない全体の共分散を判別の計量に使う方法（論文の主張）

### 3.1 LDA と最小二乗の古典の形

- **Fukunaga 1990 の等価性（Loog 2016 の式 (5)，Krijthe & Loog 2016 の式 (3) が引く）◎（引用の記述）**：Krijthe & Loog 2016 は w = T⁻¹(m₁ − m₂) について，「This leads to the same classifier as Equation (2) when n > p」と書く（式 (2) は W⁻¹(m₁ − m₂)）．データが中心化され，事前が等しいときは最小二乗の解と同じになる．前回のノートの Vidaurre & Blankertz 2010 の「same separating hyperplane」も同じ事実である．
- **Normalized LDA（Fan, Lei, Li 2008．FG）○**：要旨によれば，群内の散らばりを全体の散らばりで正規化して LDA を書き直し，ラベル無しの標本で全体の散らばり（data space）の推定を良くする．FRGC v2 と CMU PIE の顔認識で SDA（Cai ほか 2007）より良いと報告する．次元削減の手法で，本文は読んでいない．
- **USM の最小二乗（RSSL の `USMLeastSquaresClassifier.R`）◎**：θ = ((n/N)·X_eᵀX_e + nλI)⁻¹ Xᵀy（X_e はラベル付きとラベル無しの全部）．文書は Shaffer 1991 を引き，「This method seems to work best when the data is first centered」と書く．実験 10 の LDA_T（K 本の平均と A = (1 − α)T + αI）は，one-hot の目標で中心化したこの形と同じ族に入る（本調べの読み．比例の係数と切片の扱いは違う）．

### 3.2 Krijthe & Loog 2016（S+SSPR，LNCS 10029，pp. 299–309．arXiv 1610.05160）◎

https://arxiv.org/abs/1610.05160

- 3 節の式 (5)：半教師ありの最小二乗を w = ((L/(L+U)) X_eᵀX_e)⁻¹ Xᵀy と定める．ラベル無しは XᵀX の推定にだけ使う．Fan ほか 2008 と Shaffer 1991 の更新と同じだと書く．正則化は使わず，擬似逆を使う．
- 図 1（2 つのガウス，50 次元，単位の共分散，平均の距離 4．各クラス 10 本のラベル）：全体の本数が次元に近づくまでは，ラベル無しを足すと誤り率が上がる．その上がり方はラベル付きを足すときより急で，ピークの後の下がり方は遅い．
- 4.2 節と図 3（**分解**）：2 本を足したときの誤り率の変化を，(i) 全体の共分散 T の推定だけを良くした分（Updated Covariance），(ii) ラベルで平均を良くした分（Updated Mean），(iii) 両方に分けて測った．原文は「Adding up the effect of adding unlabeled objects to the effect of having additional labels, we find this approximates the total effect of adding labeled objects very well」と書く．n < p では，共分散の分が誤り率を上げる．原因は逆行列の中の小さな固有値の推定にあると書く（「Fixed rank」の曲線で確かめた）．
- 5 節と図 4（ラベル無しが無限で，真の T = I + (δ²/4p)11ᵀ）：原文「For small distances between the means, the semi-supervised learner generally does increase performance for a larger range of sizes of the labeled set, while for larger distances this is no longer the case and the semi-supervised solution is typically worse than the supervised solution」．次元を変えてもピークの位置が動くだけである．
- 6 節（8 データセット．ラベル付き L = ⌈p/2⌉，PCA で分散の 99% を残す）：ほとんどのデータで，半教師ありの版は基準の教師ありの分類器を上回りそうにない．例外は SPECT と SPECTF である．

### 3.3 少数ショットの学習のタスクの共分散

**Simple CNAPS（Bateni ほか 2020．CVPR．arXiv 1912.03432）◎**

- 式 (3)：Q_k = λ_k Σ_k + (1 − λ_k) Σ^τ + βI，λ_k = |S_k|/(|S_k| + 1)．Σ^τ（all-classes-in-task covariance）は，支えの集合（ラベル付き）の全部から，クラスを区別せずに推定する．問い合わせの集合（ラベル無し）は使わない．
- 表 7（特徴抽出器を適応しない「No Adaptation」．凍結した ResNet18）の全体の平均正解率：Simple CNAPS 59.3，タスクの共分散を外した版（−TR）55.9，二乗ユークリッド 57.8．原文は，凍結した特徴では「the all-classes-task-level covariance estimate as an added regularizer ... becomes crucial」と書く．特徴抽出器を適応すると，差は 72.2 と 72.0 の 0.2 ポイントに縮む．

**Transductive CNAPS（Bateni ほか 2022．WACV．arXiv 2006.12245）◎**

- 式 (8)〜(10)：問い合わせの予測確率を責任度にし，支えの集合（one-hot）と合わせて，クラスの平均と Q_k を soft k-means で推定し直す．β = 1 である．ラベル無しは割り当てを通して平均と共分散の両方に入る．
- 表 3（Meta-Dataset の平均．ドメイン内／ドメイン外）：Simple 73.8／69.7．問い合わせを特徴の適応（タスクの表現）にだけ使う版（FEOT）73.4／70.7．soft k-means の割り当てにだけ使う版（COT）75.4／69.7．両方 75.2／71.8．対数行列式を入れた GMM の版は 68.5／61.7 と低い．
- 4.2 節：「as the class shot increases beyond 10, performance drops compared to Simple CNAPS ... in high-shot classes, transductive updates can act as distractors」．
- FEOT と COT の除去実験は，ラベル無しの得を「ラベルに依らない使い方」と「割り当てを通す使い方」に分けた先例にあたる．ただし特徴抽出器を学習する設定である．

**GDA（Wang ほか 2024．ICLR．arXiv 2402.04087）◎（該当節を検索で読んだ）**：CLIP の凍結した特徴の上で，共通の共分散のガウスの判別を少数ショットのラベル付きから推定し，精度行列は縮小推定で作る．ラベル無しの共分散は使わない．ラベル 0 本の設定では，ゼロショットの予測から始める EM を使う．

TAFSSL（Lichtenstein ほか 2020．問い合わせを含む PCA・ICA の部分空間）は，前回のノート（厄介な成分の除去）に載せたので重ねない．

### 3.4 話者照合：凍結した埋め込みのガウスの判別を，ラベル無しのターゲットの全体の共分散だけで適応する

**Kaldi の PLDA の教師なし適応（`src/ivector/plda.cc` の `PldaUnsupervisedAdaptor::UpdatePlda`）◎**

https://github.com/kaldi-asr/kaldi/blob/master/src/ivector/plda.cc

- 学習データの全体の共分散が単位行列になる空間に，ターゲット（ラベル無し）の全体の共分散を写して固有分解する．固有値 s_i が 1 を超える向きだけで，超えた分 s_i − 1 を群内に 0.3，群間に 0.7 の割合で足す（既定値．`within_covar_scale = 0.3`，`between_covar_scale = 0.7`）．平均はターゲットの平均に置き換える．分散を増やす向きにしか直さないので，正定値は保たれる．

**CORAL+（Lee, Wang, Koshinaka 2019．ICASSP，pp. 5821–5825．arXiv 1812.10260）◎**

- 1 節の原文：「the major challenge is how the adaptation could be performed on the within and between class covariance matrices given that only the total covariance matrix could be estimated directly from the in-domain data」．
- 式 (8)〜(10)：CORAL の変換 A = C_I^{1/2} C_o^{−1/2} で作る擬似ターゲットの群内・群間の共分散と元の共分散を，β と γ で内挿する．同時対角化した空間で max(E − I, 0) を取り，分散が減る向きを除く（正則化）．
- 表 1・2（等誤り率 %，LDA 200 次元）：SRE'16 では，ソースの PLDA 9.94，Kaldi の適応 6.57，CORAL 6.31，CORAL+ 6.30，正則化なし 6.51．SRE'18 では，それぞれ 7.47，6.48，6.42，5.80，6.33．
- 話者照合の検証（2 つの発話が同じ人か）であり，クラスの数は固定でなく，少数ショットのクラスの平均は扱わない．

### 3.5 脳波の分野

- Vidaurre & Blankertz 2010（前回のノートで ◎）は，全体の共分散と全体の平均がラベル無しで推定でき，2 クラスでは同じ超平面になることを書く．偏りの推定にはクラスの事前確率が要るとも書く．
- **Vidaurre ほか 2011**（IEEE TBME 58(3):587–597）○（PubMed の要旨）：クラスに関係しない非定常性に対する LDA の教師なしの適応を，3 種類の手続きで 19 データセットについて比べる．選んだ手続きを 80 人と患者 4 人で試し，オンライン実験も行い，教師ありの適応より有意に良いと書く．3 種類が何をどう更新するか（全体の平均だけか，全体の共分散の逆行列もか）は，本文を読めず確かめていない．

---

## 4. Q3：2 次のモーメントだけを使うと悪くならないか（理論の主張）

前回のノートの MCPL（Loog 2016）と射影推定量（Krijthe & Loog 2017）には重ねない．今回の新しい資料は回帰の側である．

- **Azriel, Brown, Sklar, Berk, Buja, Zhao 2022**（JASA 117(540):2238–2251．arXiv 1612.02391）◎（要旨と 1.1 節）：要旨によれば，E[Y|X] が線形でないとき，ラベル無しも使う推定量は最小二乗を漸近的に支配する．線形なら漸近的に同じになる（「otherwise, they are asymptotically equivalent」）．1.1 節は Buja ほか 2019 を引き，「β does not depend on the distribution of X if and only if E[Y|X] is linear in X」と書く．
- **Chakrabortty & Cai 2018**（Annals of Statistics 46(4)）○（検索の要旨）：半教師ありの線形回帰の推定量（EASE）は，線形のモデルが誤っているときにだけ最小二乗を上回り，正しいときは同じになる．
- **Shaffer 1991**（The American Statistician 45(4):269–273）○（二次資料の要旨）：説明変数のモーメント行列が既知のとき，無条件の不偏性のもとでは Gauss–Markov の定理が成り立たない．回帰係数が 0 に近いときにより正確な別の推定量がある．本文は読んでいない．
- **Krijthe & Loog 2016 の図 4** ◎：真の T を使っても，クラスが離れている（δ が大きい）と教師ありより悪い．
- **まとめ（論文の主張）**：ラベル無しを 2 次のモーメントにだけ使う推定は，割り当てを回さないので，塊の仮定を要らない．しかし，悪くならないことは保証されていない．回帰の理論では，モデルが正しいと得は漸近的に 0 になる．分類の数値では，信号が大きいときと，本数が次元に近いときに悪くなる．悪くならないことを保証した版（MCPL，射影推定量）は，尤度か二乗損失についての保証であり，ラベル付きの本数が次元より多いことを要する（前回のノートの 3.4 節）．

---

## 5. MER-DA の記録との関係（記録の確認．論文ではない）

- **2026-10-17 の探索の下見**（`exp/explore_metric_room.py`，出力 `results/explore_relation/e38pre_summary.csv`．`docs/exploratory-analyses.md` の 4511 行）
  - 方法：ターゲットのテスト以外の発話（ラベル無し）の共分散 Σ（全体の散らばり．群間を含む）から M_ρ = (1 − ρ)Σ/τ + ρI を作る（τ は固有値の平均）．K 本の感情の平均の差 m̂ から向き u = M_ρ⁻¹ m̂ を作り，テストで感情の対ごとの離れ方 d を測った．特徴を標準化すると τ は 1 になるので，ρ は実験 10 の α にほぼ当たる．
  - 数値（反復 0〜9 の平均の d．ρ = 1 は等方）：

    | 種類 | K | ρ=0.5 | ρ=0.7 | ρ=0.9 | ρ=1.0 |
    |---|---|---|---|---|---|
    | WavLM | 5 | 0.44 | 0.55 | 0.76 | 0.81 |
    | WavLM | 25 | 0.90 | 1.05 | 1.29 | 1.13 |
    | HuBERT | 25 | 0.89 | 1.04 | 1.25 | 1.07 |
    | 音声＋映像 | 5 | 0.52 | 0.67 | 0.95 | 1.17 |
    | 音声＋映像 | 25 | 1.16 | 1.39 | 1.71 | 1.37 |
    | 映像 | 25 | 0.83 | 0.94 | 1.03 | 1.04 |

  - 読み：ρ = 0.5（実験 10 の規則の点 α = 0.5 に当たる）は，すべての種類と K で等方より d が小さい．軽い白色化（ρ = 0.9）が勝つのは K = 25 の音声と音声＋映像だけである．Krijthe & Loog 2016 の「本数が少ないと共分散の分が損をし，増えると得になる」と同じ向きである．
  - 実験 10 との違い：7 群の地図の設定（ソースの統計で標準化）であること，言葉と 3 つをつないだ空間が無いこと，量が UAR でなく対ごとの離れ方であること，多クラスの判定でなく対ごとの向きであること．
  - 実験 10 の設計の 0 節は，この下見を引いていない．実験 8 の設計（27 行）が引くのは，同じ日の本番（仮説転移の罰則の計量を変えて音声の K = 5 で +0.6〜+1.0 点）だけである．
- 実験 8・9 の結果の下書きの R5 は，白色化した計量（A = 0.5T + 0.5I）での群間の固有値の和を，感情・グループ・対話について出している．6.1 節の縮みの係数は，この種の固有値から計算できる．

---

## 6. 本調べでの導出（推測の水準．論文の主張ではない）

### 6.1 全体の散らばりを計量にすると，感情の向きの成分が縮む

群内で白色化した座標（W = I）で，真の全体の散らばりを T = I + c·ddᵀ（2 クラス，d は平均の差，c = π₁π₂）とする．K 本の平均の差を d̂ = d + e と書く．

```
T⁻¹ d̂ の d の向きの成分 = (|d| + e_∥) / (1 + c|d|²)
T⁻¹ d̂ の d と直交する成分 = e_⊥        （縮まない）
向きの誤差の tan は，W⁻¹ d̂（理想の計量）の (1 + c|d|²) 倍
```

数値でも確かめた（50 次元，|d| = 3，c = 0.25 で比は 3.25 = 1 + c|d|²）．多クラスでも，T M̃ = M̃(I + ΠM̃ᵀM̃) から T⁻¹M̃ = M̃(I + ΠG)⁻¹（G は群内で白色化した平均のグラム行列）となる．感情の平均が張る空間の中の信号だけが (I + ΠG)⁻¹ で縮み，直交する雑音は縮まない．

- 意味：全体の散らばりは感情の違い自身の分散を含む．そのため，それを計量にすると，K 本の平均の誤差に対して感情の向きの重みが相対的に下がる．得は話者・対話などの感情でない分散の向きを抑える分で，損はこの縮みである．Krijthe & Loog 2016 の図 4（δ が大きいと損）と Shaffer 1991（係数が 0 に近いと得）は，この式と同じ向きである．
- 縮みの係数は，候補の全ラベルでの群間の固有値 λ（全体の散らばりに対する比）から 1/(1 − λ) で計算できる（A の縮小を入れると近似になる）．

### 6.2 群内の共分散の制約を引き算で入れると，正定値でなくなりうる

Σ = A − (1 − α)M̂Π̂M̂ᵀ を U = √(1 − α)·M̂Π̂^{1/2} と書くと，次が成り立つ．

```
Σ が正定値  ⇔  I − UᵀA⁻¹U が正定値  ⇔  λ_max( (1 − α) Π̂^{1/2} M̂ᵀA⁻¹M̂ Π̂^{1/2} ) < 1
```

これも数値で確かめた．K 本の平均には雑音があり，期待値で E[M̂ᵀA⁻¹M̂]_kk = μ_kᵀA⁻¹μ_k + tr(A⁻¹W_k)/K になる．標準化した 2,304 次元で α = 0.5 なら，tr(A⁻¹W) は数百〜千程度と概算できる（未計算）．すると (1 − α)·(1/4)·tr/K は，K = 1 では 1 を大きく超え，K = 25 でも 1 を超えうる．このため，引き算の形の Σ はこの研究の K の範囲で正定値でなくなる見込みが高い．計算の前に，保存した特徴で λ_max を出せば確かめられる．Loog 2014 の掛け算の形（2.3 節）はこの問題を避けるが，T_labeled（K 本で階数 ≤ 4K − 1）が特異なので，縮小を入れないと定義できない．

### 6.3 計量だけの版と，引き算の制約の版の違いは 4 × 4 の行列だけである

Woodbury の式から，引き算の制約の版の向きは A⁻¹M̂Ĉ で，計量だけの版（LDA_T）の向きは A⁻¹M̂ である．張る 3 次元の空間は同じで，組み合わせ（Ĉ）と判定の 2 次の項だけが違う．2 クラスなら向きは同じになる．6.2 節で Ĉ が正定値でなくなる場合，この版は使えない．

### 6.4 平均の制約は，割合の偏りで偏りを入れる

候補の標準化で全体の平均は 0 である．正しい制約は Σ_k π_k^候補 μ_k = 0 で，候補の感情の割合が要る．ラベル付きの割合（等しく 1/4）で 4 つの平均の平均を引くと，各平均を Σ_k (1/4 − π_k^候補) μ_k だけずらす．候補の割合が偏ったコーパス（Neutral が MELD 56.7%，MOSEI 46.8%）では，このずれは 0 でない．等しい事前の判定では，ずれは切片だけを変える（設計の 3 節と同じ）．それでも切片の偏りは UAR を直接動かす．Vidaurre & Blankertz 2010 が偏りの推定に事前確率を要すると書いたのは，この点である．

### 6.5 分解は経路に依る

設計の分解 EM_fromK − LDA_K =（LDA_T − LDA_K）+（EM_fromK − LDA_T）は，計量を先に変え，割り当てを後に変える 1 つの経路である．逆の経路は「K 本だけの計量のまま割り当てを回す版」を要し，この版は実験 8 の保存に無い（実験 8 の EM_μ は最初の M 段の Σ，つまり候補から作った Σ を固定する）．Krijthe & Loog 2016 は 2 つの分を同じ基準から別々に測り，足し合わせが全体と合うことを確かめた．同じ形で書くには，4 つ目の角が要る．

---

## 7. 最も近い先行研究と実験 10 の比較

| | Krijthe & Loog 2016 | Loog 2014（Krijthe & Loog 2014・Loog 2016 の表 5 の形） | Kaldi の PLDA 適応・CORAL+ 2019 | Simple・Transductive CNAPS | 実験 10 |
|---|---|---|---|---|---|
| データ | 人工の 2 ガウス（10〜100 次元），UCI など 8 | UCI 10（2 クラス），UCI 16（2〜26 クラス） | NIST SRE'16・'18 の x-vector | Meta-Dataset（画像 13） | 感情 4 コーパス，4 クラス |
| ラベルの量 | 各クラス 5〜10 本，⌈p/2⌉ | max(2d, 10)，2d + K | ターゲットは 0 本 | 1〜数十ショット | 各クラス 1〜25 本（4K ≪ 2,304） |
| ラベル無しの使い方 | XᵀX（全体の散らばり）だけ | 全体の平均と全体の共分散 | ターゲットの全体の平均と共分散 | タスクの共分散は支えの集合だけ（Simple），問い合わせは割り当て（Transductive） | 全体の散らばり T（と全体の平均） |
| 縮小 | なし（擬似逆） | なし（N ≥ 2d） | 分散を増やす向きだけ | βI | α で I へ |
| モダリティ | 1 | 1 | 1 | 1（画像） | 声・顔・言葉とつないだ空間 |
| 学習器 | 最小二乗 ≈ T⁻¹(m₁ − m₂) | LDA | PLDA（ガウス） | マハラノビスの最近傍 | 共通の共分散のガウスの判別 |
| 測る量 | 誤り率，分解（共分散の分・平均の分） | 誤り率，負の対数尤度 | 等誤り率，MinCost | 正解率，FEOT・COT の除去 | UAR，分解（計量の分・割り当ての分） |
| 主張 | n ≈ p で悪化（peaking），δ が大きいと無限のラベル無しでも損 | 2 クラスでは誤り率で最良．多クラスでは 8 のうち 6 で悪化 | 教師なしの適応で等誤り率が 9.94 から 6.30〜6.57 に下がる | 凍結した特徴ではタスクの共分散で +3.4．割り当ての分は 10 ショットを超えると損 | （未計算） |

**同じ点**

- Krijthe & Loog 2016 と実験 10 は，ラベル無しを全体の散らばりにだけ使う分類器を作り，得を共分散の分と平均の分に分ける．
- Loog 2014 と実験 10 は，全体の平均と全体の共分散で，K 本からの推定を直す．
- Kaldi・CORAL+ と実験 10 は，凍結した埋め込みの上のガウスの判別で，ラベル無しのターゲットの 2 次のモーメントだけを使う．

**違う点**

- 先行のどれも，平均の分を EM の割り当てで作っていない．Krijthe & Loog 2016 は真のラベルで作り，Transductive CNAPS の COT は割り当てと共分散を一緒に動かす．
- 先行の MCLDA の実験は，どれもラベル付きを 2d 本以上にしている．実験 10 は 4K ≤ 100 本に対し 2,304 次元である．
- 先行のどれも，モダリティのブロックを扱わない．
- 先行のどれも，候補のクラスの割合が偏ったときの平均の制約の偏りを，UAR のようにクラスの割合をそろえる指標で測っていない．

---

## 8. 狙うべき差分

### 8.1 新しく書けること（実験 10 の設計の出力か，保存済みの出力で測れるもの）

1. **割り当ての分の符号を，計量の分と同じ単位で測る．**設計の G2（EM_fromK − LDA_T）は，平均の分を EM の割り当てで作る分解である．Krijthe & Loog 2016 の分解（真のラベルで平均の分を作った）を，感情のコーパスで割り当ての側に置き換えたものにあたる．経路に依らない形で書くには，「K 本の計量のまま割り当てを回す版」を足して足し合わせを確かめる（6.5 節）．
2. **ラベル付き ≪ 次元の領域で，縮小つきのモーメントの制約を UAR で測る．**先行の MCLDA の実験はすべて N ≥ 2d で，縮小を使っていない．設計の α の格子 {0.1, 0.5, 0.9} は，この領域で縮小の強さが計量の分の符号を変えるかを示す．MER-DA の下見（5 節）では，0.5 は等方より悪く，0.9 は K = 25 で良かった．
3. **モダリティをまたぐ共分散の寄与を，計量だけの版で測る．**設計の LDA_Tblk（ブロック対角の A）は，割り当てを回さない形で，モダリティ間の 2 次の構造が判別に効くかを測る．先例は見つからなかった．
4. **平均の制約の偏りを測る．**設計の LDA_Tc は等しい割合で平均を合わせる．候補の割合で合わせる版（オラクル）と並べれば，6.4 節の偏りの大きさを UAR で示せる．
5. **縮みの係数と計量の分の得失を並べる．**6.1 節の係数（候補の全ラベルでの群間の固有値から計算）と，LDA_T − LDA_K を，コーパスと空間ごとに並べる．Krijthe & Loog は δ の大小で定性的に述べただけで，実データでこの係数を測っていない．

### 8.2 新しく書けないこと

- ラベル無しの標本で全体の平均と全体の共分散を推定し，ラベルに依る推定を直すこと（Loog 2010・2012・2014，Loog & Jensen 2012，Fan ほか 2008）．
- 全体の散らばりを LDA の計量に使ってよいこと，2 クラスでは同じ超平面になること（Fukunaga，Vidaurre & Blankertz 2010，Krijthe & Loog 2016）．
- ラベル無しで XᵀX だけを推定し直すと，本数が次元に近いときとクラスが離れているときに悪くなりうること（Krijthe & Loog 2016）．
- 得を共分散の分と平均の分に分ける測り方（Krijthe & Loog 2016 の図 3）．ラベル無しの得を「統計だけの使い方」と「割り当ての使い方」に分ける除去実験（Transductive CNAPS の FEOT・COT）．
- 凍結した埋め込みのガウスの判別を，ラベル無しのターゲットの全体の共分散だけで適応すること（Kaldi，CORAL+）．
- モデルが正しいときは，ラベル無しの説明変数の分布は回帰係数について情報を足さないこと（Buja ほか 2019，Azriel ほか 2022）．
- 2 クラスの MCLDA が EM の LDA より頑健であること（Krijthe & Loog 2014）．

### 8.3 設計の文に直しが要る箇所（事実の照合）

| 設計の記述（0 節） | 先行研究と記録 |
|---|---|
| 「MCLDA は誤り率で最も良かった（Ionosphere 0.18，SPECT 0.25）」 | 2 クラスでは正しい．同じ推定量は，多クラスを含む Loog 2016 の表 5 で 16 のうち 9 データセットで教師ありより悪い（optical 0.161 → 0.900，shuttle 0.134 → 0.822．skin は有意差なし）．並べて書く |
| LDA_W は「MCLDA の形．原文で確かめてから定義する」 | PRL 2014 の本文は読めなかった．Krijthe & Loog 2014 の式 (7)(8) と RSSL の実装（2.2・2.3 節）が手がかりになる．どちらも T_labeled が正則であることを前提にし，K 本では特異になる．引き算の形は正定値でなくなる見込みが高い（6.2 節） |
| 位置づけに 2026-10-17 の下見が無い | 同じ形（K 本の平均の差 × ラベル無しの全体の共分散で縮めた計量）を，地図の設定で既に測っている（5 節）．引いて違いを書く |
| 分解 EM_fromK − LDA_K =（LDA_T − LDA_K）+（EM_fromK − LDA_T） | 経路に依る．先例の形にするには 4 つ目の角が要る（6.5 節） |

---

## 9. 見つからなかったこと

- 少数ショットの学習で，問い合わせの集合（ラベル無し）の全体の共分散だけを計量にし，割り当てを回さずに，その得を報告した研究は見つからなかった．Simple CNAPS のタスクの共分散は支えの集合だけから作り，Transductive CNAPS は割り当てを通して共分散を推定し直す．
- ラベル付きの本数が次元より少ない領域で，縮小つきの moment constraint の LDA を評価した研究は見つからなかった．
- 感情認識（音声・顔・言葉）で，moment constraint の LDA か USM の最小二乗を使った研究は見つからなかった．
- ラベル無しの得を「計量の分」と「EM の割り当ての分」に分けて測った研究は見つからなかった．
- 確かめられなかったもの：Loog 2014（PRL）と Loog 2012（PSL）の本文（有料．要旨だけ），Loog & Jensen 2012 の本文，Vidaurre ほか 2011 の本文（3 種類の手続きの中身），Shaffer 1991 の本文，Chakrabortty & Cai 2018 の本文．

---

## 10. 確認の水準の一覧

| 論文・資料 | 水準 | 読んだ箇所 |
|---|---|---|
| Krijthe & Loog 2014（ICPR．arXiv 1411.4521） | ◎ | II・III 節（式 (1)〜(10)），IV 節，表 I〜III，V 節，文献 |
| Loog 2016（TPAMI．arXiv 1503.00269） | ◎ | 2.2 節（式 (4)(5)），5.1・5.2 節，6.4 節，表 2・4・5 |
| RSSL の `MCLinearDiscriminantClassifier.R`・`MCNearestMeanClassifier.R`・`USMLeastSquaresClassifier.R` | ◎（実装） | 全文 |
| Krijthe & Loog 2016（S+SSPR．arXiv 1610.05160） | ◎ | 全文（1〜6 節，図 1〜5，文献） |
| Simple CNAPS（Bateni ほか 2020） | ◎ | 4・5 節，表 7，「No Adaptation」の節 |
| Transductive CNAPS（Bateni ほか 2022） | ◎ | 3 節（式 (8)〜(10)），4.2 節，表 3 |
| Kaldi の `plda.cc`・`plda.h` | ◎（実装） | `PldaUnsupervisedAdaptor` |
| CORAL+（Lee ほか 2019） | ◎ | 1〜5 節，表 1・2 |
| GDA（Wang ほか 2024） | ◎（該当箇所の検索） | 1・3 節の共分散と教師なしの拡張 |
| Azriel ほか 2022 | ◎ | 要旨，1.1 節 |
| Loog 2012（PSL） | ○ | 要旨 |
| Loog & Jensen 2012 | ○ | Krijthe & Loog 2014・Loog 2016 の記述 |
| Loog 2010 | ○ | RSSL の文書と Loog 2016 の記述 |
| Loog 2014（PRL） | ○ | 書誌，Krijthe & Loog 2014 と RSSL の記述（本文は読めず） |
| Fan, Lei, Li 2008 | ○ | 索引の要旨 |
| Vidaurre ほか 2011 | ○ | PubMed の要旨 |
| Chakrabortty & Cai 2018 | ○ | 検索の要旨 |
| Shaffer 1991 | ○ | 二次資料の要旨 |
| Buja ほか 2019 | ○ | Azriel ほか 2022 の記述 |
| MER-DA の 2026-10-17 の下見 | 記録 | `results/explore_relation/e38pre_summary.csv`，`exp/explore_metric_room.py` の冒頭，`docs/exploratory-analyses.md` 4507〜4570 行 |

---

## 11. 文献（URL）

- Azriel, Brown, Sklar, Berk, Buja, Zhao 2022. JASA 117(540). https://arxiv.org/abs/1612.02391
- Bateni, Goyal, Masrani, Wood, Sigal 2020（Simple CNAPS）. CVPR. https://arxiv.org/abs/1912.03432
- Bateni, Barber, van de Meent, Wood 2022（Transductive CNAPS）. WACV. https://arxiv.org/abs/2006.12245
- Chakrabortty & Cai 2018. Annals of Statistics. https://projecteuclid.org/download/pdfview_1/euclid.aos/1530086425
- Fan, Lei, Li 2008. FG. https://doi.org/10.1109/AFGR.2008.4813329
- Kaldi `src/ivector/plda.cc`. https://github.com/kaldi-asr/kaldi/blob/master/src/ivector/plda.cc
- Krijthe & Loog 2014. ICPR, pp. 3762–3767. https://arxiv.org/abs/1411.4521
- Krijthe & Loog 2016. S+SSPR, LNCS 10029, pp. 299–309. https://arxiv.org/abs/1610.05160 ，https://doi.org/10.1007/978-3-319-49055-7_27
- Lee, Wang, Koshinaka 2019（CORAL+）. ICASSP, pp. 5821–5825. https://arxiv.org/abs/1812.10260
- Loog 2012. PSL 2011, LNCS 7081, pp. 32–41. https://doi.org/10.1007/978-3-642-28258-4_4 （要旨：https://www.springerprofessional.de/en/semi-supervised-linear-discriminant-analysis-using-moment-constr/3834660 ）
- Loog 2014. Pattern Recognition Letters 37:24–31. https://doi.org/10.1016/j.patrec.2013.03.004
- Loog 2016（MCPL）. TPAMI 38(3). https://arxiv.org/abs/1503.00269
- RSSL（Krijthe）. https://github.com/jkrijthe/RSSL ，https://rdrr.io/cran/RSSL/man/MCLinearDiscriminantClassifier.html
- Shaffer 1991. The American Statistician 45(4):269–273（書誌：https://acnpsearch.tweb-dev.unibo.it/singlejournalindex/60318910 ）
- Vidaurre, Kawanabe, von Bünau, Blankertz, Müller 2011. IEEE TBME 58(3). https://pubmed.ncbi.nlm.nih.gov/21095857/
- Wang ほか 2024（GDA for CLIP）. ICLR. https://arxiv.org/abs/2402.04087

---

関係するファイル（絶対パス）：
- 実験 10 の設計（直しが要る箇所は 8.3 節）：/home/snics/MER-DA/DAI/docs/実験10_下調べ_設計.md
- 前回までのノート（重ねていない）：/home/snics/MER-DA/research_notes/先行研究_半教師ありのガウス判別_2026-11-08/report.md，/home/snics/MER-DA/research_notes/先行研究_厄介な成分の除去_2026-11-08/report.md
- 2026-10-17 の下見：/home/snics/MER-DA/exp/explore_metric_room.py，/home/snics/MER-DA/results/explore_relation/e38pre_summary.csv
- 6.1・6.2 節の恒等式を数値で確かめた台本：/tmp/claude-1000/-home-snics-MER-DA/b1f27254-32c7-45f0-ac5a-dee1c5e3ec8a/scratchpad/check.py