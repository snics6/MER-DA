# 先行研究：ラベル無しの発話で感情ごとの平均と共通の共分散を推定し直すガウスの判別（DAI-8 の位置づけ．2026-11-08）

## 要約

1. 新規性の判定：凍結した特徴の上で少数のラベルを固定し，ラベル無しの標本の EM で平均と共通の共分散を推定し直す手続きは既知である（Nigam ほか 2000 の EM-λ，Krijthe & Loog 2014 の EMLDA，TransCLIP 2024）．モデルが誤っているとラベル無しの標本が分類を悪くすることも既知である（Cozman・Cohen・Cirelo 2003，Cozman & Cohen 2006）．
2. 最も近い先行研究は 4 本である．TransCLIP（凍結した特徴，共通の共分散の GMM，ゼロショットの予測への KL），Martin ほか 2024（CLIP の予測から始めたガウスの EM は，11 データセットの平均で 58.5% から 40.4%（画像の特徴）に下がる），Krijthe & Loog 2014（EMLDA の誤り率が教師ありの LDA より大きく悪い．Ionosphere で 0.21 から 0.57），Kouw & Loog 2017（ソースの分類器を基準にした DA の LDA．保証は尤度についてだけで，各クラスの標本数が次元を超えることを仮定する）．
3. MER-DA の記録にも近い先例がある．2026-10-11 の探索では，Ledoit–Wolf の共通の共分散を持つ 4 つのガウスの混合を，ソースの事後確率を錨にして EM で推定した．感情の向きの離れ方は WavLM で 1.09 から 0.78 に，10 本のラベルを固定しても 0.85 に下がった．DAI-8 の設計の 0 節は「この形は試していない」と書くが，この記録と食い違う．
4. 狙うべき差分（先行研究に無く，DAI-8 の出力で測れるもの）は 3 つある．(a) ラベル無しの塊が感情と話者・グループのどちらに沿うかを，3 モダリティとつないだ空間で直接測る．(b) モダリティ間の共分散を 0 にした版と全共分散の版を，半教師ありの EM で比べる．(c) 保証の論文が置く「標本数 > 次元」を満たさない次元と本数で，尤度ではなく UAR を測る．
5. 新しく書けないことは，「ラベル無しの標本でガウスの判別が悪くなる」ことと，その原因がモデルの誤りであることである．

---

## 0. 用語

- **生成モデルの分類器**：特徴とラベルの同時の分布 p(x, y) を当てはめ，ベイズの式で p(y | x) を出す分類器．LDA と naive Bayes がこれに当たる．
- **判別モデルの分類器**：p(y | x) だけを当てはめる分類器．ロジスティック回帰（MER-DA の頭）がこれに当たる．
- **LDA（線形判別分析）**：各クラスを，平均 μ_k と全クラスで共通の共分散 Σ を持つガウス分布とみなす生成モデル．判定の向きは Σ⁻¹(μ_k − μ_l) で決まる．
- **EM**：ラベル無しの標本のクラスを確率（責任度 r_ik）で推定する段（E 段）と，その確率で重みづけてパラメータを推定し直す段（M 段）を繰り返す手続き．
- **縮小（shrinkage）**：標本の共分散 S を，扱いやすい行列（単位行列など）との加重平均 (1 − α)S + αI に置き換え，逆行列を安定させる操作．
- **転導的（transductive）**：評価する標本そのもの（ラベル無し）を推定に使う設定．
- **安全な半教師あり学習**：ラベル無しの標本を使った解が，ラベル付きの標本だけの解より悪くならないことを保証する手法の総称．
- **モデルの誤り（misspecification）**：真の分布が，仮定したモデルの族（例：共通の共分散のガウス）に入っていないこと．

確認の水準は次の 3 つである．◎ は本文の該当箇所を読んだもの．○ は要旨，書誌の頁，または本文を読んだ総説の記述によるもの．△ は検索の断片だけによるもの．

---

## 1. 問いの分け方

DAI-8 の試し（`DAI/docs/実験8_下調べ_設計.md`）を，検索できる 5 つの問いに分けた．

| 番号 | 問い | 調べた先 |
|---|---|---|
| Q1 | ラベル無しの標本が生成モデルの分類器を悪くするのは，どの条件のときか．モデルが誤っているときに何が起こるか | Castelli & Cover，O'Neill，Zhang & Oles，Nigam，Cozman ほか，Yang & Priebe，Martin ほか 2024 |
| Q2 | 悪くならないことを保証する半教師ありの LDA・最小二乗は，何を仮定し，何を保証し，数値はどうか | Loog 2016，Krijthe & Loog 2014・2017，Kouw & Loog 2017，Li & Zhou 2015，Mey & Loog の総説 |
| Q3 | 次元が本数より大きいとき，共分散をどう縮めるか．半教師ありの LDA で縮小はどう扱われたか | Ledoit & Wolf 2004，Friedman 1989，Bickel & Levina 2004，Houdouin ほか 2023，Transductive CNAPS |
| Q4 | 感情認識（音声・顔・言葉）で，半教師ありの生成モデルや転導的なガウスの推定を使った研究はあるか | Cohen ほか 2003・2004，Grandvalet & Bengio 2005，Zhang ほか 2011，Zheng ほか 2016，Zong ほか 2016，Vidaurre & Blankertz 2010（脳波の分野の類例） |
| Q5 | 上の 4 つと DAI-8 の差分は何か | 6〜9 節 |

---

## 2. Q1：ラベル無しの標本が生成モデルの分類器を悪くする条件（論文の主張）

### 2.1 モデルが正しいときの古典の結果

**Castelli & Cover 1995（Pattern Recognition Letters 16(1)），1996（IEEE Trans. Inf. Theory 42(6):2102–2117）○**

- 2 つのクラスの密度が既知で混合の割合 η だけが未知のとき，ラベル付きとラベル無しの標本の相対的な価値は，η についての 2 つのフィッシャー情報の比になる（1996 の要旨）．要旨は，ラベル付きの標本は分類規則を作るのに必要であり，ラベル無しの標本より指数的に価値が高いと結論する．
- Mey & Loog の総説の要約（5.1 節，式 (5.1)）：混合が識別可能でラベル無しの標本が無限にあると，残る仕事は塊にラベルを付けることだけになる．このとき誤り率は，ラベル付きの本数 n について指数的にベイズ誤り率へ近づく．
- 仮定：混合の成分とクラスが 1 対 1 に対応する．ラベル無しの標本だけではクラスの名前が決まらず，誤り率は 1/2 になる（Cozman ほか 2003 の脚注 3 が Castelli 1994 を引く）．

**O'Neill 1978（JASA 73(363):821–826）○**

- 要旨：共通の共分散を持つ 2 つのガウス（Fisher の線形判別）は，ラベル無しの観測も使って最尤で推定できる．ラベル無しの観測とラベル付きの観測が持つ情報の比は，統計的に意味のある母集団の離れ方の範囲で，およそ 1/5〜2/3 である．
- 本文は読んでいない．離れ方が小さいときの比の値は確かめていない．

**Efron 1975（JASA 70(352):892–898）○**

- 要旨：データが共通の共分散のガウスに従うとき，ロジスティック回帰の漸近の相対効率は正規判別（LDA）の 1/2〜2/3 である．
- この結果は，モデルが正しければ LDA の方が少ない本数で推定できることを示す．MER-DA の頭（ロジスティック回帰）をガウスの判別に替える動機の根拠になる．

**Zhang & Oles 2000（ICML，pp. 1191–1198）○（本文を読んだ総説の記述による）**

- Mey & Loog の総説の要約（2.1.2 節）：同時の分布を P(X | α)P(Y | X, α) と書く「パラメトリック」な場合は，X の分布が分類のパラメータを共有する生成モデルに当たる．このとき，ラベル付きとラベル無しの両方を使った最尤推定のフィッシャー情報は I_unlab + I_lab に分かれる．P(X)P(Y | X, α) と書く「セミパラメトリック」な場合は判別モデルに当たり，I_unlab = 0 なので，ラベル無しの標本は漸近的に何も足さない．
- Cozman & Cohen 2006 の 4.2 節も，判別モデルの尤度はラベル無しの標本に影響されないと書き，Zhang & Oles を引く．

**Ng & Jordan 2002（NIPS 14，pp. 841–848）○**

- 要旨：判別学習（ロジスティック回帰）は漸近の誤り率が低い．生成モデル（naive Bayes）は自分のより高い漸近の誤り率に速く近づく．このため，学習データの本数によってどちらが勝つかが 2 つの領域に分かれる．

### 2.2 モデルが誤っているとき

**Cozman, Cohen, Cirelo 2003（ICML，pp. 99–106）◎** — https://cdn.aaai.org/ICML/2003/ICML03-016.pdf

- 定理 1（式 (2)）：標本が確率 λ でラベル付きになるとき，最尤推定の極限 θ* は次の量を最大にする．
  ```
  θ* = argmax_θ  λ·E[log p(C, X | θ)] + (1 − λ)·E[log p(X | θ)]
  ```
  つまり半教師ありの最尤は，教師ありの目的と教師なしの目的の凸結合になる．以下，λ = 1 の極限を θ_l，λ = 0 の極限を θ_u と書く．
- 4 節：モデルが正しければ θ_l = θ_u であり，ラベル無しの標本は分散を減らすだけなので必ず分類を良くする．モデルが誤っていて e(θ_u) > e(θ_l)（e は誤り率）なら，ラベル無しの標本を足すほど誤り率は e(θ_u) に近づき，悪くなる．原文：「If the performance obtained with a given set of labeled data is better than the performance with infinitely many unlabeled samples, then at some point the addition of unlabeled data must decrease performance.」
- 4 節のガウスの例：2 次元の 2 クラスで，混合の割合は 3/5 である．片方のクラスの中で 2 つの特徴の相関は ρ = 4/5 だが，naive Bayes はこれを 0 とみなす．最良の線形の境界の誤り率は 0.06975，ラベル付きの極限は 0.07356，ラベル無しの極限は 0.08141 である．
- 図 1：UCI の Adult で naive Bayes に EM でラベル無しの標本を足すと，ラベルが 30 本なら良くなり，300 本と 3,000 本なら悪くなる．
- 5 節：UCI の Shuttle で，ラベル 100 本の naive Bayes の誤り率 18% は，ラベル無しを 43,400 本足すと約 30% になる．同じ条件で，TAN（特徴の間に木の形の依存を許す naive Bayes）は約 19%，構造を探す SSS は 3.7% である．
- 著者の結論（4 節）：「unlabeled data are always beneficial in the absence of modeling errors」．

**Cozman & Cohen 2006（Chapelle・Schölkopf・Zien 編『Semi-Supervised Learning』MIT Press，4 章，pp. 57–72）◎** — 書籍の PDF：https://www.molgen.mpg.de/3659531/MITPress--SemiSupervised-Learning.pdf

- 4.1 節，図 4.1：データが naive Bayes の仮定どおりなら，ラベル無しの標本は多いほど良い（図 4.1(a)）．データが TAN の依存を持つのに naive Bayes を当てると，ラベル無しの標本で悪くなる（図 4.1(b)）．ラベルが少ないと良くなり，多いと悪くなる例もある（図 4.1(c)，特徴 49 個）．
- 4.3 節，定理 4.1：上の定理 1 と同じ式である．
- 4.4 節：同じガウスの例で ρ を −4/5 にすると，ラベル無しの極限の境界が最良の境界と教師ありの境界の間に入り，ラベル無しの標本で良くなる．ρ = −1/5 では，λ = 0.08075 で混ぜると最良の線形の境界に届く．ラベル無しの標本が良くするか悪くするかは，ρ と割合の組で入れ替わる．
- 4.5 節（有限標本）：原文「the inclusion of unlabeled data may lead to a substantial decrease in variance and a decrease in classification error, even as the bias is negatively affected by the unlabeled data」．特徴が多くラベルが少ないと推定の分散が大きいので，ラベル無しの標本を足すのは妥当になる，と書く．
- 4.6 節：原文「If the addition of unlabeled data to an existing pool of labeled data degrades performance, then there is clear indication that modeling assumptions are incorrect.」．性能の低下をモデルの誤りの検出に使えると書く．
- 4.2 節：判別モデルの尤度はラベル無しの標本に影響されないと書き，Zhang & Oles 2000 を引く．

**Nigam, McCallum, Thrun, Mitchell 2000（Machine Learning 39(2–3):103–134）◎（2026-11-07 のノート `research_notes/有名な論文_2026-11-07/半教師ありDA.md` で本文の 1・5・6・8 節を確認した．今回は読み直していない）**

- 5.3 節の EM-λ：ラベル無しの文書の対数尤度に重み λ（0 ≤ λ ≤ 1）を掛ける．λ はラベル付きの学習データの leave-one-out で選ぶ（5.3.1 節）．
  ```
  l_c(θ | D; z) = log P(θ) + Σ_{d∈D_l} Σ_j z_ij log(P(c_j | θ) P(d_i | c_j; θ))
                  + λ·Σ_{d∈D_u} Σ_j z_ij log(P(c_j | θ) P(d_i | c_j; θ))
  ```
- 6 節：WebKB で，ラベル 4 本なら naive Bayes 40% が EM で 55% に上がる．ラベル 240 本なら 81% が 79% に下がり，EM が下回る．
- 8 節の原文：「If the intended target concept and model differ from the actual distribution of the data too strongly, then the use of unlabeled data will hurt instead of help performance」．5.3.1 節：ラベルが少ないと EM はほぼラベル無しの標本のクラスタリングになり，クラスタとクラスが対応しないと悪くなる．

**Yang & Priebe 2011（IEEE TPAMI 33(10):2093–2103）△**

- 書誌と検索の断片だけを確かめた．有限混合のモデルとベイズのプラグインの分類器について，モデルの誤りがいつ・なぜ半教師ありの分類を悪くするかを調べた論文である．PDF（https://www.cis.jhu.edu/~parky/CEP-Publications/YP-PAMI2011.pdf ）の取得は拒否された．定理の内容は確かめていない．

**Martin, Huang, Shakeri, Pesquet, Ben Ayed 2024（CVPR，pp. 28816–28826．EM-Dirichlet）◎** — https://openaccess.thecvf.com/content/CVPR2024/papers/Martin_Transductive_Zero-Shot_and_Few-Shot_CLIP_CVPR_2024_paper.pdf

- 6.1 節：CLIP の凍結した特徴の上で転導的なクラスタリングを比べる．どの手法も初期値は CLIP の予測にそろえる（原文「Initialization is uniform across different clustering techniques, utilizing CLIP's predictions」）．問い合わせの集合は 75 枚で，現れるクラスは 3〜10 種がでたらめに選ばれる（このため割合は偏る）．
- 表 1（11 データセットの平均の正解率）：ゼロショットの CLIP は 58.5 である．画像の特徴の上のガウスの EM は，共分散を単位行列に固定すると 8.3，対角を推定すると 40.4 である．確率のベクトルの上のガウスの EM は，単位行列で 19.3，対角で 49.7 である．ディリクレの EM は 65.8，その硬い割り当ての版は 67.6 である．
- 著者の結論（6.1 節）：「Gaussian-based data-fitting approaches are sub-optimal for simplex clustering」．別の段落では，K-means 型の目的には「a strong, inherent bias towards K-balanced partitions」があると書き，割合の偏った問い合わせの集合のために分け方の複雑さの項（MDL）を足す．
- このガウスの EM には CLIP の予測へ引き戻す項が無く，予測は初期値にだけ入る．

### 2.3 因果の向き

**Schölkopf ほか 2012（ICML）○（Mey & Loog の総説 2.1.3 節の記述による）**

- X が Y の原因なら P(Y | X) は P(X) と独立であり，ラベル無しの標本は役に立たない．Y が X の原因（反因果）なら独立でなくなり，ラベル無しの標本は役に立ちうる．
- 感情（ラベル）が声・顔・言葉（特徴）を生む向きは，反因果の側に当たる．ただしこの枠組みは，P(X) のどの構造が P(Y | X) を決めるかまでは言わない．

### 2.4 Q1 のまとめ（論文の主張）

- モデルが正しければ，ラベル無しの標本は生成モデルの分類器を漸近的に良くする（Castelli & Cover，O'Neill，Zhang & Oles，Cozman ほか）．ただし，ラベル付きの標本の方が価値は高い（Castelli & Cover では指数的に，O'Neill では 1/5〜2/3 の比）．
- モデルが誤っていれば，ラベル無しの標本の比重 1 − λ が大きいほど解は教師なしの極限 θ_u に寄る．θ_u の誤り率が教師ありの極限より高ければ，分類は悪くなる（Cozman ほか 2003 の定理 1）．
- 良くなるか悪くなるかは，ラベルの本数（分散の大きさ）とモデルの誤りの形の組で決まり，事前には分からない（Cozman & Cohen 2006 の 4.4・4.5 節）．
- 原因として著者たちが挙げる候補は，クラスタとクラスが対応しないこと（Nigam 5.3.1 節），特徴の間の依存を無視すること（Cozman の naive Bayes と TAN），外れ値（Cozman & Cohen 4.1 節が Grandvalet & Bengio を引く）である．実データでクラスタが何に対応していたかを測った論文は無い．

---

## 3. Q2：悪くならないことを保証する推定（論文の主張）

### 3.1 半教師ありの LDA

**Loog 2016（IEEE TPAMI 38(3):462–475．MCPL）◎** — https://arxiv.org/abs/1503.00269

- 定義 1（式 (10)(11)）：教師ありの最尤推定 θ̂_sup を基準にし，ラベル無しの標本に最も不利な柔らかいラベル q を付けたときの尤度の差を最大にする．
  ```
  θ̂_semi = argmax_θ  min_q  [ L(θ | X, U, q) − L(θ̂_sup | X, U, q) ]
  ```
  「contrast」は θ̂_sup を基準にすること，「pessimism」は最悪の q を取ることを指す．
- 補題 1（式 (12)）：ラベル付きとラベル無しを合わせた学習データの上で，真のラベルについての対数尤度は L(θ̂_sup) ≤ L(θ̂_semi) ≤ L(θ̂_opt) を満たす（θ̂_opt は全ラベルを使った推定）．
- 式 (13)(14)：q を固定した M 段は π_k，μ_k，Σ の閉じた形で解け，Σ はラベル付きとラベル無しの全部の標本から作る．式 (15)：全体の平均は q に依らない．MCPL の解は μ̂ = Σ_k π̂_k μ̂_k を満たすので，全体の平均がクラスの平均を縛る．
- 定理 1：教師ありの推定が well-posed（Σ̂_sup が正定値）で，ラベル無しの標本が 1 本以上あり，特徴が連続分布に従うなら，学習データの上の尤度の不等号は確率 1 で厳密になる．定理 2：期待値でも厳密になる．
- 5.1 節の実験の条件：16 データセットで，次元を PCA で落とし（分散の 99.9% を残す），ラベル付きの本数を 2d + K（d は次元，K はクラス数）にした．
- 6.2 節：著者は尤度の大きな改善を，共分散を全部の標本で推定できたことによると読む．原文「The immense improvements are probably explained by the fact that the averaged class-conditional covariance matrix Σ is much more stably estimated in case of semi-supervision」．
- 表 4（テストの誤り率．教師あり → MCPL → 全ラベルの順）：optical 0.161 → 0.113 → 0.049，landsat 0.291 → 0.251 → 0.161，shuttle 0.134 → 0.103 → 0.059，spectf 0.325 → 0.325 → 0.260．6.3 節：low と spectf では，尤度が良くなっても誤り率は良くならなかった．5 節の原文「the relation between likelihood and error rate is not necessarily monotonic」．

**Krijthe & Loog 2014（ICPR．Implicitly constrained semi-supervised LDA）◎** — https://arxiv.org/abs/1411.4521

- III 節で 5 つを比べる．教師ありの LDA，自己学習の LDA（SLLDA），EM の LDA（EMLDA），モーメントの制約つきの LDA（MCLDA．Loog の 2012〜2014 の版），陰の制約つきの LDA（ICLDA）である．ICLDA は，ラベル無しの標本に [0, 1] の柔らかいラベルを付けて得られる解の集合の中で，ラベル付きの標本の尤度を最大にする．
- IV-A 節の人工データ（2 次元，ラベル 10 本，ラベル無し 990 本）：クラスの境界がガウスの塊の分かれ目と直交する場合，原文「EMLDA forces the decision boundary to fall between the two gaussian clusters leading to a much worse solution then the supervised LDA」．
- 表 II（10 分割の交差検証の誤り率．ラベル付きは max(2d, 10) 本）：

  | データ | 教師ありの LDA | 全ラベル | MCLDA | EMLDA | SLLDA | ICLDA |
  |---|---|---|---|---|---|---|
  | Ionosphere | 0.21 | 0.15 | 0.18 | 0.57 | 0.20 | 0.18 |
  | SPECT | 0.31 | 0.18 | 0.25 | 0.62 | 0.33 | 0.30 |
  | WDBC | 0.11 | 0.04 | 0.09 | 0.38 | 0.09 | 0.08 |
  | Transfusion | 0.34 | 0.23 | 0.32 | 0.52 | 0.37 | 0.33 |
  | BCI | 0.21 | 0.16 | 0.20 | 0.21 | 0.21 | 0.20 |

  10 データセットのうち，EMLDA が教師ありの LDA より悪くならなかったのは BCI（同じ 0.21）と SPECTF（0.32 → 0.28）だけである．
- 著者の結論（要旨）：制約に基づく方法はモデルの誤りに頑健で，未知の標本の対数尤度で他を上回りうる．IV-B 節：誤り率では MCLDA が最も良い．ICLDA は保守的で，誤り率を下げないデータが多い．

### 3.2 半教師ありの最小二乗と判別モデル

**Krijthe & Loog 2017（Machine Learning 106(7):993–1008．射影推定量）◎** — https://ar5iv.labs.arxiv.org/html/1602.07865

- 式 (4)〜(6)：ラベル無しの標本に [0, 1] のラベルを付けて得られる最小二乗の解の集合 Θ を作り，教師ありの解を，全部の標本で決めた計量で Θ に射影する．
- 定理 1（4.1 節）：ラベル付きとラベル無しを合わせた学習データの上で，真のラベルについての二乗損失は教師ありの解以下になる．4.2 節：L2 の正則化を入れた教師ありの解でも，同じ保証が成り立つ．4.3 節：改善が起こるのは，教師ありの解が Θ の外にあるときだけである．
- 表 1（ラベル d + 5 本）：16 データセットで，射影推定量は教師ありの解を有意に下回らなかった．自己学習は 2 データセットで誤り率を有意に上げた．射影推定量の改善幅は自己学習より小さい．
- 7.1 節の原文：「good properties in terms of this loss do not necessarily translate into good properties in terms of the error rate」．

**Krijthe & Loog 2018（NeurIPS．The pessimistic limits and possibilities of margin-based losses in semi-supervised learning）○（Mey & Loog の総説 2.1.4・6.2.2 節の記述による）**

- 単調に減る損失を使う判別モデルでは，ゆるい条件のもとで，ラベル無しの標本に「教師ありの解より悪くなるラベル付け」が必ず存在する．このため判別モデルでは，決して悪くならない半教師ありの解を作れない．総説は，Loog 2016 の保証が生成モデル（同時の分布）で成り立つのは，尤度が P(X) を含むためだと説明する．

**Li & Zhou 2015（IEEE TPAMI 37(1):175–188．S4VM）○** — ICML 2011 版：https://mlanthology.org/icml/2011/li2011icml-making/

- 低密度の仮定に合う分離面の候補を複数作り，教師ありの SVM に対する最悪の場合の得が最大になる候補を選ぶ．Mey & Loog の総説の定理 16（Li & Zhou 2011 の定理 1）：真のラベルが候補の作るラベル付けの集合に入っていれば，ラベル無しの標本の上の正解率は教師ありの SVM を下回らない．
- 要旨：S3VM は多くの場合に性能を有意に下げるが，S4VM はほとんど下げない．数値の表は確かめていない．

### 3.3 ドメイン適応の LDA

**Kouw & Loog 2017（arXiv 1706.08082．Target contrastive pessimistic risk．後に Pattern Recognition Letters 2021「Robust domain-adaptive discriminant analysis」）◎（arXiv 版）** — https://arxiv.org/abs/1706.08082

- 2.3〜2.4 節：ソースで学んだ分類器のターゲットのリスクを基準にし（contrast），ターゲットのラベル無しの標本に最も不利なラベル q を付けたときのリスクの差を最小にする（pessimism）．ターゲットのラベルは使わない．MER-DA の K = 0 に当たる．
- 4.3 節，定理 1：ターゲットの分布が連続で，各クラスの一意な標本数が特徴の次元を超える（m_k > D）とき，与えられたターゲットの標本の上で，LDA と QDA の経験リスク（負の対数尤度）は，ソースで推定した LDA より厳密に小さい．原文は「not necessarily strictly smaller with respect to new target samples」と注意し，「as long as the same amount of regularization λ is added to both the source θ̂_S and the tcp classifier θ̂_T, the guarantee also holds for a regularized model」と書く．
- 5.3.2 節（心臓病の 4 病院の組）：保証どおりリスクは下がるが，ROC の面積では手法の並びが変わる．保証は尤度についてであり，正解率や AUC についてではない．

### 3.4 Q2 のまとめ（論文の主張）

| 論文 | 保証する量 | どの標本の上で | 仮定 | この研究の条件で成り立つか（8.3 節の推測） |
|---|---|---|---|---|
| Loog 2016 | LDA の対数尤度 | ラベル付き＋ラベル無しの学習データ | 教師ありの Σ̂ が正定値（ラベル付きの本数 > 次元） | ラベル付きは最大 100 本，次元は 512〜2,304 なので満たさない |
| Krijthe & Loog 2017 | 最小二乗の損失 | 同上 | X_eᵀX_e が正定値．L2 正則化つきでもよい | 正則化つきなら形式上は満たす |
| Kouw & Loog 2017 | LDA の負の対数尤度 | 与えたターゲットの標本 | 各クラスの標本数 > 次元．同じ正則化を入れればよい | 正則化なしでは AFEW・IEMOCAP で満たさない |
| Li & Zhou 2015 | 正解率 | ラベル無しの標本 | 真のラベルが候補の集合に入る（低密度の仮定） | SVM の結果で，ガウスの判別には当たらない |

保証は 4 本とも転導的（与えたラベル無しの標本の上の保証）であり，そのうち 3 本は尤度か二乗損失についての保証である．正解率の保証は Li & Zhou の 1 本だけで，低密度の仮定に依る．UAR についての保証を置いた論文は見つからなかった．

---

## 4. Q3：高次元の共分散の縮小と，半教師ありの LDA での扱い（論文の主張）

**Ledoit & Wolf 2004（Journal of Multivariate Analysis 88(2):365–411）○** — 2001 年の作業論文（要旨を読んだ）：http://www.ledoit.net/ole1a.pdf

- 要旨：次元が大きいと，標本の共分散は条件が悪く，逆行列を取れないこともある．提案は，標本の共分散と単位行列の定数倍の凸結合である．標本数と次元が一緒に無限大に向かう漸近のもとで，2 次の損失を最小にする結合の重みを閉じた式で与える．分布は仮定しない．

**Friedman 1989（JASA 84(405):165–175．RDA）○**

- 小標本・高次元の判別分析のための方法である．クラスごとの共分散を共通の共分散へ縮める重み λ と，単位行列の定数倍へ縮める重み γ の 2 つを置き，交差検証で選ぶ（書誌と二次資料による）．

**Bickel & Levina 2004（Bernoulli 10(6):989–1010）○**

- 要旨と二次資料：次元が標本数より速く増えると，標本の共分散の（擬似）逆行列を使う Fisher の線形判別は，最悪の場合の誤り率がでたらめと同じになりうる．共分散を対角とみなす独立の規則（naive Bayes）の方が良い．定理の条件は本文で確かめていない．

**半教師ありの LDA での縮小の扱い**

- Loog 2016（5.1 節）◎：共分散が特異になるのを避けるために，次元を PCA で落とし，ラベル付きを 2d + K 本にした．縮小は使っていない．4.3 節では，中〜高次元では行列式の対数が数値的に無限大になりやすいと注意する．
- Krijthe & Loog 2017（4.2 節）◎ と Kouw & Loog 2017（4.3 節）◎：基準と提案に同じ量の L2 正則化を入れれば保証は残ると書く．縮める強さの選び方は扱わない．
- Martin ほか 2024 の表 1 ◎：ガウスの EM で共分散を単位行列に固定した版（平均 8.3）は，対角を推定する版（40.4）より大きく悪い．共分散の形の選び方で，初期値から離れる幅が大きく変わる．
- TransCLIP（Zanella ほか 2024）◎（2026-11-08 の調べのノート `research_notes/疑似ラベル 情報の持ち込み方 先行研究/1_ラベル無しデータの構造.md` の P12 で本文を確認した．今回は読み直していない）：全クラスで共通の対角の共分散を使う．表 6(a)：共分散を更新しないと，EuroSAT で 65.1 から 51.6 に下がる．
- Houdouin, Ollila, Pascal 2023（arXiv 2303.14989．Regularized EM algorithm）○：標本の少ない GMM では共分散の更新が特異になる．この問題に対し，構造を持つ目標の行列へ縮める罰則つきの尤度を最大にする EM を出す．教師なしのクラスタリングであり，ラベル付きの標本は扱わない．
- Bateni ほか 2022（WACV．Transductive CNAPS）○：Mahalanobis 距離の soft k-means を使い，支えの集合（ラベル付き）から始めて，問い合わせの集合（ラベル無し）でクラスの共分散を推定し直す．共分散は正則化する．特徴抽出器も学習する．
- Wang, Dobriban, Gataric, Samworth 2023（arXiv 2304.09154．Sharp-SSL）○：高次元の半教師ありのガウスの EM を，軸に沿った低次元のでたらめな射影の上で解き，結果を集める．信号を持つ座標が一部だけであることを前提にする．
- Lu & Qiao 2015（arXiv 1509.05438）○：標本の共分散を逆にできない場合のための，ラベル無しの標本も使う疎な Fisher の LDA．

縮小の強さを Ledoit–Wolf の式で決めて EM の各回で使う半教師ありの LDA を，ラベル付きの本数を振って評価した論文は，今回の検索では見つからなかった．

---

## 5. Q4：感情認識での半教師ありの生成モデルと転導的なガウスの推定（論文の主張）

**Cohen, Sebe, Cozman, Cirelo, Huang 2003（CVPR，pp. 595–604）○，Cohen, Cozman, Sebe, Cirelo, Huang 2004（IEEE TPAMI 26(12):1553–1567）○**

- 動画の表情認識をベイジアンネットワークの分類器で行い，ラベル無しの標本を EM で使う．要旨と書誌によれば，naive Bayes と TAN はラベル付きの標本では良いが，ラベル無しの標本を足すと悪くなる．著者は，分類の正解率で構造を探す SSS を提案する．Cozman & Cohen 2006 の 4.6 節がこの結果を引く．
- 本文の数値と，人物を分けた評価かどうかは確かめていない（PDF を取得できなかった）．

**Grandvalet & Bengio 2005（NIPS 17．エントロピー最小化）◎（2026-11-07 のノートで 4.2 節を確認した）**

- 表情認識（7 感情，正面の顔 375 枚，ラベルは 1/10）で，カーネルのロジスティック回帰の誤り率は 53.6 ± 1.3% で，エントロピー最小化を足すと 52.0 ± 1.9% になる．近傍法の系の方法（63.1%，63.8%）が弱い理由を，著者は「同じ人の別の表情が近くに並び，表情ではなく人物を拾うため」と説明する（4.2 節）．人物と表情の近傍の一致は数値で測っていない．

**Zhang, Weninger, Wöllmer, Schuller 2011（ASRU，pp. 523–528）◎** — https://mediatum.ub.tum.de/doc/1107343/687488.pdf

- 6 コーパス（ABC，AVIC，DES，eNTERFACE，SAL，VAM）の leave-one-corpus-out で，2 値の覚醒度と感情価を線形の SVM で認識する．IV-B 節：3 コーパスで学んだモデルで別の 2 コーパスのラベル無しの発話に予測を付け，その予測で学び直す（自己学習の形）．ラベル無しの発話はターゲットではなく，ほかのコーパスのものである．
- 表 VII（UA の平均．覚醒度／感情価）：3 コーパスで 64.7%／55.8%，3 コーパスにラベル無しの 2 コーパスを足すと 65.1%／56.6%，ラベル付きの 5 コーパスで 66.6%／58.1%．原文：「addition of unlabeled training data delivers roughly half of the gain that can be expected from adding labeled training data」．
- 表 VI の本文（コーパスごとの正規化）：覚醒度では z 正規化の 66.6% が中心合わせの 63.7% より良く，感情価では中心合わせの 58.1% が最良である．
- 生成モデルや EM は使っていない．

**Schuller ほか 2010（IEEE TAC 1(2):119–131）○（`docs/related-work.md` の記述による）**：話者ごと・コーパスごとの z 正規化で，コーパス間の成績が上がる．使うのは，ターゲットのラベル無しの発話の 1 次と 2 次の統計だけである．

**Zheng, Zong ほか 2016（IEEE TAC．転導的な転移の部分空間学習）○**：ラベル付きのソースの顔の画像とラベル無しのターゲットの画像で判別の部分空間を学び，ターゲットにラベルを予測してから SVM を学ぶ．データは BU-3DFE と Multi-PIE で，生成モデルではない．

**Zong, Zheng, Zhang, Huang 2016（IEEE SPL 23(5):585–589．DaLSR）○**：ソースとターゲットの不一致を扱う最小二乗の回帰で，コーパス間の音声感情認識を行う．ターゲットはラベル無しである．

**Ahn, Lee, Shin 2021（IEEE SPL 28:1190–1194）○**：ターゲットのラベルを使わずに，few-shot 学習と教師なしの DA を組み合わせてコーパス間の音声感情認識を行う．要旨だけを確認した．

**脳波の分野の類例：Vidaurre & Blankertz 2010（Brain Topography 23(2):194–198）◎** — https://pmc.ncbi.nlm.nih.gov/articles/PMC2874052

- Methods の節：2 クラスの LDA では，クラスの共分散の平均の代わりに，全標本をまとめた共分散を使える．原文は，全体の共分散は「can be estimated without using label information」であり，「yields the same separating hyperplane」と書く．全体の平均についても，「the bias of the separating hyperplane can be estimated (and adapted) in an unsupervised manner」と書く．条件は「to have an estimate of the prior probabilities of the 2 classes」である．
- Level 3 の節：最後の 2 回の試行では，全体の平均だけを試行ごとに教師なしで更新した（更新の係数 0.05）．共分散とクラスの平均は，教師なしでは更新していない．
- 関連して，Vidaurre ほか 2011（IEEE TBME 58(3):587–597）○ は，運動想像の脳波で LDA の教師なしの適応を 19 データセットで比べた．

**Q4 のまとめ**：感情認識でラベル無しの標本を生成モデルの EM に入れた研究として見つかったのは，表情の Cohen ほか 2003・2004 だけであり，そこでも naive Bayes と TAN は悪くなった．音声・顔・言葉の凍結したエンコーダの特徴の上で，ターゲットのラベル無しの発話からガウスの平均と共分散を推定し直し，コーパスをまたいで UAR で評価した研究は見つからなかった．

---

## 6. 最も近い先行研究 4 本と DAI-8 の比較

DAI-8 の試しは次のとおりである．凍結した特徴（WavLM 1,024 次元，ResNet-50 の顔 512，RoBERTa 768，3 つをつないだ 2,304）の上で，感情ごとに K 本（K ∈ {0, 1, 5, 10, 25}）のラベルを固定する．K 本だけで学んだ融合の頭の予測 q を，初期値と E 段の事前（重み β）に使う．ラベル無しの候補の発話の EM（10 回）で，感情ごとの平均，共通の共分散（単位行列へ α で縮める），割合を推定し直す．評価は，4 コーパスの leave-one-corpus-out の仮のターゲットの回の 120 単位で，テストの UAR を測る．

| | TransCLIP（Zanella ほか 2024） | Martin ほか 2024（CVPR） | Krijthe & Loog 2014（ICPR） | Kouw & Loog 2017（arXiv） | DAI-8 |
|---|---|---|---|---|---|
| データ | 画像 11 データセット | 画像 11 データセット | UCI など 10，2 クラス | UCI の標本選択の偏り，心臓病 4 病院 | 感情 4 コーパス，4 クラス |
| ラベルの量 | クラスあたり 0〜16 本 | 0（表 1），1〜16（表 2） | max(2d, 10) 本 | ターゲットは 0 本 | クラスあたり 0〜25 本 |
| ドメインのずれ | 扱わない（事前学習から下流へ） | 扱わない | 扱わない | 扱う（ソースの分類器を基準にする） | 扱う（ほかの 3 コーパスで学んだ出発点） |
| モダリティ | 画像の特徴と文の予測 | 画像の特徴か確率のベクトル | 1 つ | 1 つ | 声・顔・言葉とつないだ空間 |
| 共分散 | 共通の対角 | 単位行列か対角 | 共通の全共分散 | 共通の全共分散（QDA も） | 共通の全共分散を単位行列へ縮める．モダリティ間を 0 にした版も |
| 事前・基準 | ゼロショットの予測への KL | 予測は初期値だけ | 教師ありの LDA を基準にする（MCLDA・ICLDA） | ソースの分類器を基準にする（最悪のラベル） | K 本の頭の予測を E 段の事前に β で入れる |
| クラスの割合 | 釣り合った混合を仮定 | MDL の項で偏りを許す | 推定する | 推定する | E 段で推定し，テストは等しい事前で判定 |
| 測る量 | 正解率 | 正解率 | 誤り率，テストの尤度 | 尤度，二乗損失，AUC | UAR，近傍の感情と話者の一致，候補の上の UAR |
| 主張 | 転導で正解率が上がる．ショットが多いと得が減る（付録 D） | ガウスの EM は初期値の予測より悪い．ディリクレが良い | モデルが合わないと EMLDA は大きく悪くなる．制約の方法は頑健 | 尤度では決して悪くならない（m_k > D のとき） | （未計算） |

**同じ点**

- TransCLIP と DAI-8 は，凍結した特徴の上で，学習済みの分類器の予測を事前に残したまま，共通の共分散のガウスの混合を推定し直す．Martin ほか 2024 と DAI-8 は，どちらも分類器の予測を初期値にしたガウスの EM を使う．
- Krijthe & Loog 2014 の EMLDA と DAI-8 の β = 0 の版は，ラベル付きを固定した素の EM の LDA であり，同じ手続きである．
- Kouw & Loog 2017 と DAI-8 の K = 0 の版は，ソースの分類器から出発し，ターゲットのラベル無しの標本だけで LDA を推定し直す．

**違う点**

- 4 本とも使うモダリティは 1 つで（TransCLIP は画像の特徴と文の予測の 2 つの情報源），モダリティごとの共分散のブロックを扱わない．
- 4 本とも，ガウスの EM が何に沿った塊を作ったかを測っていない．
- 保証のある Kouw & Loog の保証は尤度についてであり，次元より多い標本を仮定する．DAI-8 は UAR を測り，次元の方が標本より多い単位を含む（8.3 節）．
- TransCLIP はクラスの割合が釣り合っていると仮定するが，DAI-8 は割合を推定する．

---

## 7. MER-DA の過去の記録との関係（記録の確認．論文ではない）

- **2026-10-11 の探索**（`docs/exploratory-analyses.md` の 3187〜3241 行，節の題は「ラベル無しのターゲットの構造に，向きの情報はどれだけあるか」．台本は `exp/explore_unlabeled_direction.py`．検査の仕組みを入れる前の記録）
  - 共通の共分散（Ledoit–Wolf）を持つ 4 つのガウスの混合を，ソースだけの分類器の事後確率から始めて EM で 20 回推定した．錨（ソースの事後確率で縛ること）の有無と，各感情 10 本のラベルの固定の有無を比べた．
  - 感情の向きの離れ方（6 対の平均．WavLM）は，ソース 1.09，1 歩 1.00，素の EM 0.80，錨つき EM 0.78，錨つき EM に 10 本を固定すると 0.85 である．ラベル付きの LDA の天井は 1.68，K = 10 の仮説転移は 1.36 である．錨つき EM がソースを上回った組は 2.5%（40 組中 1 組）だった．
  - 記録の結論：ソースの分類器から始めても，EM は感情と無関係な塊（話者や内容の違いと読める）へ流れ，ソースの事後確率で縛っても止まらない．限界の節は，正確に言えるのは「この形のモデルでは情報を取り出せない」ことであり，ラベル無しの構造に情報が無いことの証明ではない，と書く．
  - DAI-8 との違いは 5 つある．(1) 7 群の地図の設定で，1 つのソースから 1 つのターゲットへの組（136 組）を単位にした．(2) 測った量は向きの離れ方で，UAR ではない．(3) 言葉のモダリティと 3 つをつないだ空間が無い．(4) 錨はソースだけの分類器である（DAI-8 は K 本の頭）．(5) K は 10 だけである．
- **2026-10-16 の探索**（同 4202〜4262 行）：ソースごとの LDA の向きを ((1 − ρ)Σ + ρτI)⁻¹ × 感情の平均 で縮めた．ターゲットに移る向きの離れ方は，ρ = 0.001（LDA に近い）と ρ = 1（平均の差だけ）の両端で悪く，ρ = 0.3〜0.7 で最も良かった（WavLM の中央値は ρ = 0.001 で 1.25，ρ = 0.5 で 1.62，ρ = 1 で 1.08）．DAI-8 の α の格子 {0.1, 0.5, 0.9} は，この範囲の中と両側を含む．
- **2026-10-17 の探索**（同 4507〜4570 行）：ラベル無しのターゲットの共分散で仮説転移の罰則の計量を変えても，上積みは音声の K = 5 で +0.6〜+1.0 点に限られ，K ≥ 10 ではほぼ 0 だった．
- **割合の推定**：同 3211 行は，ソースの事後確率から始めた EM による感情の割合の推定が，誤差 0.3〜0.5 で外れた記録を引く．

DAI-8 の設計の 0 節は「平均と共通の共分散を，ラベル無しの発話の EM で推定し直す形は試していない」と書く．これは DAI の 4 コーパスの設定では正しいが，MER-DA 全体では 2026-10-11 の探索がこの形を試している．記録と合わせるには，設計の文を「DAI の設定では試していない．MER-DA の 7 群の地図の設定では 2026-10-11 に向きの離れ方で試し，悪化した」と直す必要がある．

---

## 8. この研究への当てはまり（推測．論文の主張ではない）

### 8.1 Cozman の式で DAI-8 の比重を読む

候補の発話の数は，全体の約 8 割と見なして概算した（設計 2 節と DAI-7 の設計のテストの大きさから）．ラベル付きの本数 4K から，ラベル付きの比重 λ = 4K / 候補の数 を出すと次のようになる．

| コーパス | 候補の数（概算） | K = 1 の λ | K = 25 の λ |
|---|---|---|---|
| AFEW | 約 640 | 0.006 | 0.16 |
| IEMOCAP 即興 | 約 2,300 | 0.002 | 0.04 |
| MELD | 約 8,800 | 0.0005 | 0.011 |
| MOSEI | 約 15,000 | 0.0003 | 0.007 |

Cozman ほか 2003 の定理 1 では，EM の極限は，教師ありの目的と教師なしの目的を λ の重みで混ぜたものを最大にする点である．λ が 0.01 前後なら，β = 0 の EM の行き先はほぼ教師なしの極限 θ_u になると読める．DAI-8 で解を K 本の頭の近くに留める項は，E 段の β·log q だけである．Nigam ほか 2000 の λ（M 段でラベル無しの寄与を弱める重み）にあたるつまみは，設計に無い．TransCLIP の KL の項は，β の項と同じ役割を持つ．

### 8.2 全体の散らばりはラベルに依らないので，EM が向きに足す情報はクラスの平均を通るものだけである（本調べでの導出）

DAI-8 の設計では，特徴を候補の統計で標準化し，S = (XᵀX − Σ_k n_k μ_k μ_kᵀ)/N，Σ = (1 − α)S + αI とする．全体の散らばり T = XᵀX/N は責任度 r に依らない（Loog 2016 の式 (15) と同じ観察であり，Vidaurre & Blankertz 2010 の「without using label information」に当たる）．A = (1 − α)T + αI と置き，感情の平均を並べた d × 4 の行列を M，割合の対角行列を Π とすると，次が成り立つ．

```
Σ = A − (1 − α)·M Π Mᵀ
Σ⁻¹ M = A⁻¹ M · C,     C = [ I − (1 − α)·Π MᵀA⁻¹M ]⁻¹     （Woodbury の式．C は 4 × 4）
```

判定の向き Σ⁻¹(μ_k − μ_l) は，ラベルに依らない A⁻¹ を感情の平均に掛けたものの組み合わせになる．候補の平均が 0 なので Σ_k π_k μ_k = 0 であり，この組み合わせは 3 次元以下の空間に入る．判定の式の 2 次の項 MᵀΣ⁻¹M も G·C（G = MᵀA⁻¹M）なので，判別の全体は A，M，Π だけで決まる．2 クラスなら，向きは A⁻¹(μ_1 − μ_2) と同じ向きになる（Vidaurre & Blankertz の「same separating hyperplane」）．

したがって，EM の各回で共分散を推定し直しても，向きが変わるのは，責任度で重みづけた感情の平均 μ_k と割合 π が動くからだけである．共分散の推定し直しから新しい向きの情報が入るわけではない．A⁻¹ は回によらないので，計算でも 1 回求めれば足りる．2026-10-11 の探索の「1 歩（ソースの事後確率で重みづけた平均と共通の共分散）はソースとほぼ同じ」という記録は，この式では「1 歩目の μ_k はソースの頭の予測で重みづけた平均なので，ソースの頭の情報を越えない」と読める．

この導出は，標準化の後の候補の平均が 0 であることを前提にする．K 本のラベル付きの発話も候補に入っており，各行の重みの和が 1 なので，前提は満たされる．DAI-8 のブロック対角の版では A もブロック対角になり，同じ関係がそのまま成り立つ．

### 8.3 保証の論文の仮定と DAI-8 の次元と本数

- Loog 2016 の定理は，教師ありの Σ̂ が正定値であること（ラベル付きの本数 > 次元）を要し，実験ではラベル付きを 2d + K 本にした．DAI-8 のラベル付きは最大 100 本で次元は 512〜2,304 なので，縮小しなければ教師ありの LDA を定義できない．設計の 3.2 節が「K = 1 で S が 0 になる」と書くのは，この状況である．
- Kouw & Loog 2017 の定理は，各クラスの標本数が次元を超えることを要する．候補の感情ごとの数（全体の約 8 割と見なした概算）は，AFEW 140〜170，IEMOCAP 即興の怒り 約 230，MELD の悲しみ 約 800，MOSEI の怒り 約 1,160 である．AFEW と IEMOCAP 即興はどの空間でも条件を満たさない．MELD は声（1,024）とつないだ空間で満たさず，MOSEI はつないだ空間（2,304）で満たさない．Kouw & Loog は同じ正則化を入れれば保証が残ると書くので，縮小つきの版は形式上は保証の範囲に入りうる．その場合も，保証は尤度についてで，UAR についてではない．
- Bickel & Levina 2004 によれば，次元が候補の数を超える AFEW のつないだ空間（2,304 > 約 640）では，縮小の強さが判別の良し悪しを大きく左右すると読める．DAI-8 の設計は α を 0.5 に固定し，Ledoit–Wolf の推定は使わない（設計 8 節）．2026-10-16 の探索（ρ = 0.3〜0.7 が最良）は，ソースの向きでは α = 0.5 が良い範囲に入ることを示す．ターゲットの EM の中での最良の α は測っていない．

### 8.4 クラスの割合

- Veilleux ほか 2021（NeurIPS）◎ — https://proceedings.nips.cc/paper_files/paper/2021/file/4d7a968bb636e25818ff2a3941db08c1-Paper.pdf ：問い合わせの集合のクラスの割合をディリクレ分布（a = 2·1_K）で偏らせると，割合の釣り合いを強く仮定する転導の手法は成績が下がる．4 節：PT-MAP は 5 クラスのすべての設定で 18〜20% 下がる．表 1：mini-ImageNet の 5 ショットで，PT-MAP 67.1（18.2 下がる），TIM 79.8（4.1 下がる），帰納の Simpleshot 80.1 である．
- DAI-8 は E 段で割合 π を推定し，テストは等しい事前で判定する．割合を推定する形は Saerens ほか 2002 と同じで，TransCLIP の釣り合いの仮定とは違う．MER-DA の記録では割合の EM の推定が外れている（7 節）．このため，E 段の π の誤りが責任度を通して μ_k に入る経路が残ると読める．設計の 3.3 節の「推定した割合と候補の真の割合の差」は，この経路を測る．

### 8.5 モダリティ間のブロック対角の版

- Cozman ほか 2003 と Cozman & Cohen 2006 の主な例は，特徴の間の依存を無視する naive Bayes の誤りによって，ラベル無しの標本で悪くなる例である．DAI-8 の Q4（つないだ空間でモダリティをまたぐ共分散を 0 にする版）は，特徴 1 つずつではなく，声・顔・言葉の 3 つのブロックについて同じ仮定を置く．Cozman の例に従えば，ブロック対角の版の方がモデルの誤りが大きく，ラベル無しの標本で大きく悪化すると予想できる．一方で，ブロック対角にすると推定する共分散の数が減るので，分散の面では有利になる（Cozman & Cohen 4.5 節，Bickel & Levina）．どちらが勝つかは，先行研究からは決まらない．

---

## 9. 狙うべき差分

### 9.1 新しく書けること（先行研究に無く，DAI-8 の出力か保存済みの出力で測れるもの）

1. **ラベル無しの塊が何に沿うかを，感情のコーパスで直接測る．**Nigam（5.3.1 節）は「クラスタとクラスが対応しない」と書き，Grandvalet & Bengio（4.2 節）は「人物を拾う」と書くが，どちらも数値では測っていない．DAI-8 の Q1 は，近傍の感情の一致と話者・グループの一致を，でたらめの水準からの伸びで比べる．これは Cozman のいう「モデルの誤り」の中身を，4 コーパス × 4 つの空間で測ることにあたり，先行研究の推測を実測に置き換える差分になる．
2. **モダリティを 1 つのブロックとみなす独立の仮定を，半教師ありのガウスの EM で検査する．**Cozman・Cohen の naive Bayes 対 TAN は特徴 1 つずつの独立の仮定であり，Cohen ほか 2004 は表情の特徴だけの 1 モダリティである．DAI-8 の Q4 はモダリティのブロックについての版であり，今の疑似ラベル（3 つの頭の積）が置く仮定と同じ形である．この比較は先行研究に見つからなかった．
3. **保証の仮定の外の次元と本数で UAR を測る．**半教師ありの LDA の保証（Loog 2016，Kouw & Loog 2017）は，「標本数 > 次元」と尤度の物差しを前提にする．DAI-8 は次元が候補の数を超える単位（AFEW のつないだ空間）を含み，クラスの割合が偏ったコーパスで UAR を測る．この条件で，ラベル無しの標本がガウスの判別を良くするか悪くするか（設計の R3：EM − K 本だけの LDA）を K の格子で測った研究は見つからなかった．
4. **Cozman の 2 つの極限を同じ単位で並べる．**設計は，K 本だけの LDA（θ_l の有限標本の版），EM（λ がほぼ 0），候補の全ラベルの LDA（天井）を同じ単位で出す．設計の出力の「EM の各回の r の変化」には，解が回ごとに初期値から離れる様子が残る．先行研究の実データの例（Adult，Shuttle，WebKB，UCI）は誤り率の最終値だけを出し，回ごとの経路は出していない．
5. **K 本の頭の予測を事前に残す強さ β と K の組で，ラベル無しの標本の得と損を測る．**TransCLIP は事前の項を外す除去実験をした（文の項を外すと 1 ショットの平均が 73.2 から 50.2 に下がる）が，ドメインのずれのある設定ではない．DAI-8 の β ∈ {0, 1} × K の格子は，ずれのあるコーパスでこの項の役割を測る．

### 9.2 新しく書けないこと

- 生成モデルの分類器がラベル無しの標本で悪くなること．その原因がモデルの誤りであること．ラベルが少ないと良くなり多いと悪くなる場合があること（Cozman ほか 2003，Cozman & Cohen 2006，Nigam ほか 2000）．
- ラベル無しの標本の重みを下げると悪化を抑えられること（Nigam の EM-λ）．
- 教師ありの解を基準にして最悪のラベルを取れば，尤度で悪くならないこと（Loog 2016，Kouw & Loog 2017）．
- 凍結した特徴の上で，分類器の予測を事前に入れて GMM を推定し直すこと（TransCLIP）．
- 高次元で共分散を縮めること（Ledoit & Wolf，Friedman）．
- 分類器の予測から始めたガウスの EM が，その予測より悪くなりうること（Martin ほか 2024 の表 1，Krijthe & Loog 2014 の表 II）．
- ガウスの EM で感情の向きが悪化することは，MER-DA の中でも 2026-10-11 に記録済みである．DAI-8 で悪化が出ても，MER-DA として新しい知見ではない．DAI-8 で新しく言えるのは，UAR での大きさ，言葉とつないだ空間を含めた結果，K 本の頭を事前にした結果，近傍とブロック対角の 2 つの診断である．

---

## 10. 見つからなかったこと

- 音声・顔・言葉の凍結したエンコーダの特徴の上で，ターゲットのラベル無しの発話からガウスの平均と共分散を EM で推定し直し，コーパスをまたいで UAR で評価した研究は，今回の検索では見つからなかった．
- 感情のコーパスで，ラベル無しの塊が感情と話者のどちらに沿うかを数値で測った研究は見つからなかった（Grandvalet & Bengio の説明は数値を伴わない）．
- モダリティのブロックについての条件つき独立を，半教師ありのガウスの EM で全共分散と比べた研究は見つからなかった．
- UAR（クラスごとの再現率の平均）について，半教師ありの推定が悪くならないことを保証した研究は見つからなかった．
- Ledoit–Wolf の縮小を EM の各回で使う半教師ありの LDA を，ラベル付きの本数を振って評価した研究は見つからなかった．
- 確かめられなかったもの：Yang & Priebe 2011 の本文（取得が拒否された），Cohen ほか 2004 の本文の数値，O'Neill 1978 と Efron 1975 の本文，Li & Zhou 2015 の数値の表，Kouw & Loog の Pattern Recognition Letters 版．

---

## 11. 確認の水準の一覧

| 論文 | 水準 | 読んだ箇所 |
|---|---|---|
| Cozman, Cohen, Cirelo 2003 | ◎ | 要旨，3〜5 節，定理 1，図 1・2 |
| Cozman & Cohen 2006 | ◎ | 4.1〜4.7 節，定理 4.1，図 4.1・4.2 |
| Nigam ほか 2000 | ◎（前回のノート） | 1・5・6・8 節 |
| Loog 2016 | ◎ | 要旨，3.2〜6.3 節，定義 1，補題 1，定理 1・2，式 (13)〜(15)，表 2・4 |
| Krijthe & Loog 2014 | ◎ | 要旨，III・IV 節，表 II |
| Krijthe & Loog 2017 | ◎ | 3〜4 節，定理 1，6 節，表 1，7.1 節 |
| Kouw & Loog 2017 | ◎ | 要旨，2.3〜2.4 節，4.2〜4.3 節，定理 1，5.3.2 節 |
| Martin ほか 2024 | ◎ | 5 節の末，6.1・6.2 節，表 1 |
| Veilleux ほか 2021 | ◎ | 4 節，6 節，表 1 |
| Mey & Loog（総説，arXiv 1908.09574） | ◎ | 2.1.2〜2.1.4 節，5.1 節，6.2 節 |
| Zhang ほか 2011 | ◎ | IV-B 節，表 VII，表 VI の本文 |
| Vidaurre & Blankertz 2010 | ◎ | Methods，Level 3 の節 |
| Grandvalet & Bengio 2005 | ◎（前回のノート） | 2.1・4.1・4.2 節 |
| TransCLIP（Zanella ほか 2024） | ◎（前回のノート） | 3 節，4 節，表 1・4・6，付録 D |
| Castelli & Cover 1995・1996 | ○ | 要旨，総説 5.1 節 |
| O'Neill 1978 | ○ | 要旨 |
| Efron 1975 | ○ | 要旨 |
| Zhang & Oles 2000 | ○ | 総説 2.1.2 節 |
| Ng & Jordan 2002 | ○ | 要旨 |
| Schölkopf ほか 2012 | ○ | 総説 2.1.3 節 |
| Krijthe & Loog 2018 | ○ | 総説 2.1.4・6.2.2 節 |
| Li & Zhou 2015 | ○ | 要旨，総説 6.2.1 節 |
| Ledoit & Wolf 2004 | ○ | 作業論文の要旨 |
| Friedman 1989 | ○ | 書誌と二次資料 |
| Bickel & Levina 2004 | ○ | 要旨と二次資料 |
| Houdouin ほか 2023 | ○ | 要旨 |
| Bateni ほか 2022 | ○ | 要旨 |
| Wang ほか 2023（Sharp-SSL） | ○ | 要旨 |
| Lu & Qiao 2015 | ○ | 要旨 |
| Cohen ほか 2003・2004 | ○ | 要旨と書誌 |
| Schuller ほか 2010 | ○ | `docs/related-work.md` の記述 |
| Zheng ほか 2016，Zong ほか 2016，Ahn ほか 2021 | ○ | 要旨と書誌 |
| Vidaurre ほか 2011 | ○ | 要旨 |
| Yang & Priebe 2011 | △ | 書誌と検索の断片 |

---

## 12. 文献（URL）

- Ahn, Lee, Shin 2021. IEEE SPL 28. https://scholar.gist.ac.kr/handle/local/11460
- Bateni, Barber, van de Meent, Wood 2022. WACV. https://openaccess.thecvf.com/content/WACV2022/html/Bateni_Enhancing_Few-Shot_Image_Classification_With_Unlabelled_Examples_WACV_2022_paper.html
- Bickel & Levina 2004. Bernoulli 10(6). https://projecteuclid.org/journals/bernoulli/volume-10/issue-6/Some-theory-for-Fishers-linear-discriminant-function-naive-Bayes-and/10.3150/bj/1106314847.full
- Castelli & Cover 1996. IEEE Trans. Inf. Theory 42(6). https://research.ibm.com/publications/the-relative-value-of-labeled-and-unlabeled-samples-in-pattern-recognition-with-an-unknown-mixing-parameter
- Cohen, Sebe, Cozman, Cirelo, Huang 2003. CVPR. https://mlanthology.org/cvpr/2003/cohen2003cvpr-learning
- Cohen, Cozman, Sebe, Cirelo, Huang 2004. IEEE TPAMI 26(12). https://dare.uva.nl/id/995c95dd-7aa0-46bc-aa1e-6401693c2523
- Cozman, Cohen, Cirelo 2003. ICML. https://cdn.aaai.org/ICML/2003/ICML03-016.pdf
- Cozman & Cohen 2006. Semi-Supervised Learning（MIT Press）4 章．https://www.molgen.mpg.de/3659531/MITPress--SemiSupervised-Learning.pdf
- Efron 1975. JASA 70(352). https://synapsesocial.com/papers/W2090578816
- Friedman 1989. JASA 84(405). https://scilit.com/publications/3f440785bdb447a255254c57cea29a88
- Grandvalet & Bengio 2005. NIPS 17（前回のノート `research_notes/有名な論文_2026-11-07/半教師ありDA.md` を参照）
- Houdouin, Ollila, Pascal 2023. https://arxiv.org/abs/2303.14989
- Kouw & Loog 2017. https://arxiv.org/abs/1706.08082
- Krijthe & Loog 2014. https://arxiv.org/abs/1411.4521
- Krijthe & Loog 2017. https://arxiv.org/abs/1602.07865
- Ledoit & Wolf 2004（作業論文 2001）．http://www.ledoit.net/ole1a.pdf
- Li & Zhou 2011/2015. https://mlanthology.org/icml/2011/li2011icml-making/
- Loog 2016. https://arxiv.org/abs/1503.00269
- Lu & Qiao 2015. https://arxiv.org/abs/1509.05438
- Martin, Huang, Shakeri, Pesquet, Ben Ayed 2024. CVPR. https://openaccess.thecvf.com/content/CVPR2024/papers/Martin_Transductive_Zero-Shot_and_Few-Shot_CLIP_CVPR_2024_paper.pdf
- Mey & Loog 2020（総説）．https://arxiv.org/abs/1908.09574
- Ng & Jordan 2002. https://papers.nips.cc/paper/2001/hash/7b7a53e239400a13bd6be6c91c4f6c4e-Abstract.html
- Nigam, McCallum, Thrun, Mitchell 2000. https://doi.org/10.1023/A:1007692713085
- O'Neill 1978. JASA 73(363). https://www.bohrium.com/en/paper-details/normal-discrimination-with-unclassified-observations/813318537698344962-806
- Veilleux, Boudiaf, Piantanida, Ben Ayed 2021. NeurIPS. https://proceedings.nips.cc/paper_files/paper/2021/file/4d7a968bb636e25818ff2a3941db08c1-Paper.pdf
- Vidaurre & Blankertz 2010. Brain Topography 23(2). https://pmc.ncbi.nlm.nih.gov/articles/PMC2874052
- Vidaurre ほか 2011. IEEE TBME 58(3). https://pure.korea.ac.kr/en/publications/toward-unsupervised-adaptation-of-lda-for-brain-computer-interfac/
- Wang, Dobriban, Gataric, Samworth 2023（Sharp-SSL）．https://arxiv.org/abs/2304.09154
- Yang & Priebe 2011. IEEE TPAMI 33(10). https://dblp.uni-trier.de/rec/journals/pami/YangP11.html
- Zanella, Gérin, Ben Ayed 2024（TransCLIP）．https://arxiv.org/html/2406.01837v1
- Zhang, Weninger, Wöllmer, Schuller 2011. ASRU. https://mediatum.ub.tum.de/doc/1107343/687488.pdf
- Zheng, Zong ほか 2016. IEEE TAC. https://bohrium.dp.tech/paper/arxiv/573f1a930cf2435a6d70b78f
- Zong, Zheng, Zhang, Huang 2016. IEEE SPL 23(5). https://pure.bit.edu.cn/en/publications/cross-corpus-speech-emotion-recognition-based-on-domain-adaptive-
