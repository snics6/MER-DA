# 先行研究の調査 A（理論の側）：別モダリティの疑似ラベルは「ラベルの代わり」として働く，という知見は既に報告されているか（2026-11-03）

担当：prior-work-diff（A）．調査役の実行環境がファイルの作成を拒んだため，本体のセッションが返答の本文をそのまま保存した．作業用の PDF とテキストは調査のセッションの scratchpad（`A_theory_pdf/`，`A_theory_txt/`）にあり，セッションが終わると消える．`results/` は開いていない．

## 要約

1. **判定：この知見の理論の形は，部品ごとに見るとすべて既知である．**新しい定理として書ける部分は見つからなかった．
2. 「教師の出力は生徒にとってラベルの代わり（雑音入りのラベル）である」は，次の箇所に書かれている：Menon ほか 2021 の抄録，Blum & Mitchell 1998 の定理 1 の証明，Nigam & Ghani 2000 の 5.3 節，Dong ほか 2025 の 1 節．
3. 「生徒の理想の出力は，教師の出力を生徒の特徴で条件づけた期待値である」は，次の 2 本に書かれている．
   - Vapnik & Izmailov 2015 の 5.1.4 節（条件つき期待値を回帰で推す）．
   - Yang ほか 2022 の定理 1（特権の特徴を持つ教師から線形回帰の生徒へ，ラベル無しのデータで蒸留する）．
   - 交差エントロピーと softmax の形で明示した文は見つからなかった．ただし，目標について線形な損失の性質から 1 行で出る．
4. 「生徒は教師を自分の仮説集合へ射影したものを学ぶ」は，次の文献にある：Charikar ほか 2024 の定理 1 の証明（2 乗損失），Phuong & Lampert 2019 の定理 1（同じ入力の線形の教師），Dong ほか 2025・Xue ほか 2025・Wu ほか 2026（教師の誤差の受け継ぎ）．
5. 「自分の予測を柔らかい疑似ラベルにしても，線形ロジスティック回帰は動かない」は，Kumar・Ma・Liang 2020 の例 3.6 にそのまま書かれている．「別の見方の予測なら得がある」は，Blum & Mitchell 1998，Nigam & Ghani 2000 の表 6，Lopez-Paz ほか 2016 の実験 2，Yang ほか 2022 の注意 2 にある．
6. **最も近い先行研究**：Yang ほか 2022，Vapnik & Izmailov 2015，Charikar ほか 2024，Kumar ほか 2020，Menon ほか 2021．Yang ほか 2022 は，教師が生徒の特徴と特権の特徴の両方を使う点まで MER-DA と同じ形である．
7. **上澄み（A の範囲で先行研究を超える部分）は薄い．**残るのは次の 3 つで，どれも既知の結果の適用か確認である．
   - (a) 出発点への L2 正則化と K 本の交差エントロピーを含む目的関数でも，疑似ラベルは「Σ_i q_i [1, z_i]」を通してしか入らない，という注意書き．
   - (b) 重みの付け方を替えても差が出ないことの，コーパス間・少数ラベルでの実測．
   - (c) 自分の予測を疑似ラベルにする対照との差の実測．
8. **狙うべき差分**：理論の新しさは主張しない．既知の結果を MER-DA の目的関数に当てはめた「注意書き」として書き，実測（重みの付け方の不変性，自分の予測との対照）は「既知の予言の確認」として示す．重みの付け方の不変性には文献の先例も反例の検証も無いので，導出は「近似の説明」と書き，定理とは書かない．

---

## 0. この報告で使う語

- **配備する分類器**：ターゲットで実際に使う分類器．凍結した特徴の上の線形ソフトマックスである．線形ソフトマックスとは，特徴の 1 次式で感情ごとの点数を出し，softmax で確率にする分類器をいう．文献の語では「生徒」にあたる．
- **疑似ラベルのモデル**：ラベル無しのターゲットの発話に，4 感情の確率 q を付けるモデル．MER-DA で選ばれるのは，配備する分類器自身のモダリティを含む融合のモデルである．文献の語では「教師」にあたる．
- **出発点**：全ソースで学んだ分類器．学び直しの初期値であり，L2 正則化（重みの差の 2 乗和の罰）で引き戻す先でもある．
- **仮のターゲット**：ソースのコーパスを 1 つ外してターゲットに見立てたもの．設定はここで選ぶ．
- **柔らかいラベルの交差エントロピー**：確率の目標 q と予測 p について −Σ_c q_c log p_c を発話で平均した量．目標 q について 1 次式である．
- **条件つき期待値 E[q ∣ z]**：配備する分類器の特徴 z が同じ発話を集めたときの，教師の確率 q の平均．
- **射影**：ある関数を，決められた関数の集まりの中で最も近いものに置き換える操作．ここでの関数の集まりは，z の線形ソフトマックス全体である．
- **共訓練（co-training）**：2 つの見方（例：声と顔）で別々の分類器を作り，片方の予測を他方のラベル無しデータのラベルにする半教師あり学習．
- **自己学習（self-training）**：分類器自身の予測を，自分のラベル無しデータのラベルにする半教師あり学習．
- **特権情報（privileged information）**：学習のときだけ使え，配備のときには使えない入力．MER-DA では，声だけの分類器から見た顔と言葉がこれにあたる．
- **弱から強への一般化（weak-to-strong generalization）**：弱いモデルの予測をラベルにして強いモデルを学ぶと，強いモデルが弱いモデルを上回る現象．

検証の水準は次の 3 段階である．
- **◎**：原文（PDF を文字にしたもの）で該当箇所を読んだ．既存の MER-DA の調査で ◎ と記録されたものは「◎（既存）」と書く．
- **○**：抄録か，信頼できる二次資料．
- **△**：検索結果の断片だけ．

---

## 1. 判定：知見を 6 つの部品に分けた結果

| 部品 | 内容 | 判定 | 最も近い文献と箇所 | 水準 |
|---|---|---|---|---|
| P1 | 教師の出力は，生徒にとってラベルの代わり（雑音入りのラベル）である | **既知** | Menon ほか 2021 の抄録と 2 節，Blum & Mitchell 1998 の定理 1 の証明，Nigam & Ghani 2000 の 5.3 節，Dong ほか 2025 の 1 節，Vapnik & Izmailov 2015 の 3.1 節 | ◎ |
| P2 | 生徒の理想の出力は E[教師の出力 ∣ 生徒の特徴] である | **既知**（回帰と一般の関数では明示されている．交差エントロピーの形は 1 行の帰結） | Vapnik & Izmailov 2015 の 5.1.4 節，Yang ほか 2022 の 5.1 節の定理 1，Xu ほか 2026 の「posterior mean teacher」 | ◎ |
| P3 | 生徒は，教師を自分の仮説集合へ射影したものしか受け取れない．教師の判断のうち，生徒の特徴で（線形に）予測できる部分だけが移る | **既知** | Charikar ほか 2024 の定理 1 の証明，Phuong & Lampert 2019 の定理 1，Dong ほか 2025 の定理 1，Xue ほか 2025 の定理 3.8，Wu ほか 2026 の 4 節，Burns ほか 2023 の付録 E.2 | ◎（Dong・Xue・Wu・Burns は既存） |
| P4 | 教師の確率を生徒の特徴へ最小二乗で写してから学んでも，解は同じになる．疑似ラベルは Σ_i q_i [1, z_i] を通してしか入らないからである | **既知の事実の直接の帰結．明示した文献は見つからなかった** | Patrini ほか 2016 の定理 3（2 値の線形分類器で，損失が「ラベルを含まない項」と「平均作用素 E[y x] の項」に分かれる），最大エントロピー法の積率一致（教科書の事実） | ◎（Patrini は既存）／背（積率一致） |
| P5 | 疑似ラベルの重みの付け方（柔らかい，温度，硬い，釣り合い）を替えても，線形の生徒では差が出にくい | **先例は見つからなかった．反対向きの理論はある** | 反対向き：Harutyunyan ほか 2023，Kumar ほか 2020，Menon ほか 2021．近い実測：Nigam & Ghani 2000 の表 6 | ◎（Kumar，Menon，Nigam & Ghani），○（Harutyunyan） |
| P6 | 自分の予測を疑似ラベルにしても新しい情報は無く得も無いが，別の見方の予測なら得がある | **既知** | Kumar ほか 2020 の例 3.6，Blum & Mitchell 1998 の定理 1，Nigam & Ghani 2000 の表 6，Lopez-Paz ほか 2016 の実験 2 と 4.3 節，Yang ほか 2022 の注意 2，Lang ほか 2024 の抄録 | ◎ |

### 1.1 P1：教師はラベルの代わりである（既知）

**Menon, Rawat, Reddi, Kim, Kumar「Why distillation helps: a statistical perspective」**（arXiv 2005.10419．ICML 2021 では「A statistical perspective on distillation」）◎
- 抄録："Knowledge distillation is a technique for improving the performance of a simple "student" model by replacing its one-hot training labels with a distribution over labels obtained from a complex "teacher" model."
- 2 節の式 (7)：教師がベイズの確率 p*(x) = P(y ∣ x) を出す場合（Bayes-distillation）を扱う．このとき蒸留の経験リスクは，通常の経験リスクと同じく母集団のリスクの不偏推定であり，分散は小さい（補題 1）．
- 命題 3：教師が p* からずれるときの生徒の誤差を，「教師の偏り ‖E p_t(x) − p*(x)‖²」と「教師の分散 V[p_t(x)]」に分ける．
- 同じ節の注："in the limit of infinite student samples, the quality of the distilled objective is wholly determined by how well the teacher probabilities model the Bayes probabilities."
- MER-DA との関係：
  - Menon の p* は，生徒と同じ入力 x の上のベイズの確率である．
  - 別モダリティの教師では，生徒の特徴 z を固定しても教師の確率 q が揺れる．この揺れを命題 3 の「教師の分散」と読み替えると，生徒が受け取る目標は E[q ∣ z]，損は「E[q ∣ z] と P(y ∣ z) のずれ」になる．
  - **この読み替えは本調査の読みで，Menon は書いていない．**Menon の分散は，教師の学習データの揺れによるものである．

**Blum & Mitchell 1998（COLT）の定理 1 の証明** ◎（既存．`research_notes/改善案_2026-11-02/literature.md` 4.8 節）
- 原文："if we use h(x1) as a noisy label of x2, then this is equivalent to (α, β)-classification noise"．

**Nigam & Ghani 2000（CIKM，pp. 86–93）の 5.3 節** ◎
- co-EM は，片方の見方の分類器が全部のラベル無しデータに確率のラベルを付け，もう片方がそれで学ぶ手法である．
- 原文："The essence of their argument is that an initial A classifier can be used to generate a large sample of noisily-labeled data to train a B classifier. The co-EM algorithm does exactly this using one learner to assign labels to all the unlabeled data, from which the second classifier learns from."
- **co-EM は MER-DA の手法と同じ形である．**別の見方のモデルが全部のラベル無しデータに柔らかいラベルを付け，配備する分類器がそれで学ぶ．

**Dong, Li, Li, Lee, Lei 2025（ICML．arXiv 2502.05075）の 1 節** ◎（既存．`I_第4回.md`）
- 原文："the error in weak supervision can be viewed as independent label noise for the student"．

**Vapnik & Izmailov 2015（JMLR 16:2023–2049）** ◎
- 3.1 節：「Oracle Teacher」は，生徒に新しい特徴ではなく，ベイズの決定規則の slack（各学習例がどれだけ境界の内側に入るか）を渡す．これで収束の速さが 1/√ℓ から 1/ℓ に上がる．
- 5 節の冒頭：教師の知識を "Teacher has some function y = f0(x) that distinguishes cancer ... in the pixel space X" と書く．教師が渡したいのは，生徒の入力空間の中の関数である．
- これは，研究者の言う「生徒が既に持つ手がかりをどの感情に結びつけるか」と同じ見方である．

### 1.2 P2：生徒の理想の出力は E[教師 ∣ 生徒の特徴]（既知）

**Vapnik & Izmailov 2015 の 5.1.4 節** ◎
- 原文："Let us call the conditional expectation function φk(x) = ∫ K*(u*k, x*) p(x*|x) dx* the image of frame K*(u*k, x*) in space X. To find m image functions φk(x) ... we solve the following m regression estimation problems"．
- 5.1.5 節：回帰が正確なら，教師の規則の生徒の空間での近似は f(x) = Σ_k β*_k φ_k(x) + b* である．期待値の線形性から，これは「教師の決定関数を，生徒の特徴で条件づけた期待値」そのものになる（本調査の読み）．
- `DAI_paper/overview/DAI.md` 6.5.1 節の考察が既に指摘しているとおり，これは MER-DA の E[q ∣ z] と同じ量である．
- MER-DA との違い：Vapnik & Izmailov は，条件つき期待値を新しい特徴として足し，ラベルで最終の分類器を学ぶ（5.2 節）．
- 線形の配備する分類器に当てると：生徒の特徴の線形回帰で作った新しい特徴は，元の特徴の張る空間の中にある．したがって足しても仮説集合は広がらず，線形の生徒にとって教師は「目標（ラベルの代わり）」としてしか働かない（本調査の帰結）．

**Yang, Sanghavi, Rahmanian, Bakus, Vishwanathan 2022「Toward Understanding Privileged Features Distillation in Learning-to-Rank」**（arXiv 2209.08754．掲載先は NeurIPS 2022 と記録されているが，原典では確かめていない）の 5 節 ◎
- 設定：
  - 通常の特徴 x と観測できない u から，y = xᵀw* + uᵀv* + ε でラベルが作られる．
  - 教師は，学習データで y を [x, z]（z は特権の特徴）に回帰し，ラベル付きとラベル無しの全部の発話に予測を付ける．
  - 生徒は，その予測を x だけに回帰する（α = 0．教師だけから学ぶ）．
- 定理 1：通常の回帰の誤差は O(dx(σ² + ‖v*‖²)/n) である．蒸留では O(dx σ²/n) + O(dx ‖v*‖²/(n + m)) + O(1/(nm)) になる．
- 本文："PFD can explain a proper part of the variance in y by privileged features z. By learning from the teacher's predictions, PFD can therefore reduce the variance of ŵ_pri by exploiting the privileged features and the unlabeled samples."
- 1 節の要約："PFD works because the teacher can explain away the variance arising from the privileged features, thus allowing the student to focus on the part it can predict."
- **MER-DA との対応**：
  - 教師が生徒の特徴と特権の特徴の両方を入力にする点は，MER-DA の融合の疑似ラベルのモデル（配備する分類器のモダリティを含む）と同じ形である．
  - 特権の特徴 u は x と独立なので，教師の予測を x へ回帰すると E[教師 ∣ x] になる．u の部分は，m 本のラベル無しの発話で平均されて消える．
  - これは，研究者の「ラベル無しのデータが十分あれば，生徒の理想の出力は E[教師 ∣ 生徒の特徴]」の回帰版である．

**Xu, Yao, Wang, Liu 2026「Weak-to-Strong Generalization via Bregman Bias–Variance Decomposition」**（ICML 2026，PMLR 306．arXiv 2505.24313）◎（抄録，1・4 節の該当段落）
- 原文："W2SG is more likely when the student effectively approximates the teacher's posterior mean"，"the posterior is defined by conditioning the teacher's distribution on the student model"．
- 条件づけるのは教師の学習の乱数（パラメータ）であり，入力のモダリティではない．
- 交差エントロピーについては "the softmax normalization renders the hypothesis class non-convex" と書く．

**交差エントロピーと softmax の形について**
- 柔らかいラベルの交差エントロピーは目標 q について 1 次式である．したがってラベル無しの発話が十分あれば，目的関数の q を E[q ∣ z] に置き換えても値は変わらない（期待値の塔の性質）．
- この 1 行の論証を，別モダリティの蒸留の文脈で式として書いた文献は，今回の検索では見つからなかった．検索の要約も，これを「一般原理からの推論」として述べるだけだった．
- 内容は自明に近く，査読者が新しさとして認める見込みは低い．

### 1.3 P3：射影としての蒸留（既知）

**Charikar, Pabbaraju, Shiragur 2024「Quantifying the Gain in Weak-to-Strong Generalization」**（NeurIPS 2024．arXiv 2405.15116）の定理 1 と 3 節の証明 ◎
- 強いモデルと弱いモデルは別々の表現 h_s，h_w を持つ．同じ入力の別の写像なので，別モダリティの場合を含む．
- 強いモデルは f_sw = argmin_{f∈F_s} d_P(f∘h_s, f_w∘h_w) で学ぶ．
- 証明の原文："fsw ∘ hs is the projection of fw ∘ hw onto the convex set Vs. We can therefore apply the "Pythagorean theorem" for projections onto a convex set"．
- 仮定：2 乗損失（回帰），F_s が凸（線形の頭は凸），真の関数が V_s に入る（realizability）．
- MER-DA との違い：分類で交差エントロピーを使う．真の事後確率は z の線形ソフトマックスに入らないので realizability が崩れる．出発点への L2 正則化と K 本のラベルがある．

**Mulgund & Pabbaraju 2025**（arXiv 2501.19105）◎（抄録，1 節，系 4.2）
- Charikar の結果をブレグマン距離に広げ，分類に当てはめた．
- ただし生徒を KL の第 1 引数に置く逆向きの KL を使う（原文："the strong model's output is in the first argument in the KL divergence—contrast this to standard supervised classification"）．ロジスティック回帰の層の凸結合も条件である．
- **標準の蒸留（MER-DA の手法）の射影の結果としては引けない．**

**Phuong & Lampert 2019「Towards Understanding Knowledge Distillation」**（ICML 2019，PMLR 97:5142–5151．arXiv 2105.13093）の定理 1 ◎
- 設定は，線形の教師と線形の生徒（同じ入力，2 値），生徒の初期値 0 である．
- 原文："When few data points are available (n < d), the weight vector learned by the student is simply the projection of the teacher's weight vector onto the data span"．
- 射影の先は「データの張る空間」であり，別モダリティの教師ではない．

**Dong ほか 2025 の定理 1・注意 3，Xue, Li, Mirzasoleiman 2025（ICML．arXiv 2502.00620）の定理 3.8，Wu, Zhang, Zhang, Fang 2026（ICML．arXiv 2606.01292）の 4 節** ◎（既存．`I_第4回.md`，`J_第5回.md`）
- 3 本とも，教師の誤差のうち生徒の特徴の部分空間に入る分だけが受け継がれる，と示す．
- Wu の原文："the teacher's error tensor ... is projected onto the student's geometry"．
- Dong は付録 B の証明で，教師の偏りも直交射影 P_s で書く．

**Burns ほか 2023（ICML 2024）の付録 E.2** ◎（既存）
- 生徒が当てはめやすい誤りはそのまま写り，でたらめな誤りは写らない．

**Lang, Sontag, Vijayaraghavan 2024「Theoretical Analysis of Weak-to-Strong Generalization」**（arXiv 2405.16043）◎（抄録と 1 節）
- 原文："weak-to-strong generalization occurs when the strong model is unable to fit the mistakes of the weak teacher without incurring additional error"．
- Blum & Mitchell の共訓練の結果を一般化したと，著者自身が書いている．

**本調査の補足（交差エントロピーの射影の等式．本調査の導出で，文献の該当箇所は確かめていない）**
- 線形ソフトマックスの族は，指数型の条件つきの族である．このため，柔らかいラベルの交差エントロピーの最小解 p* は積率一致 Σ_i q_i [1, z_i] = Σ_i p*_i [1, z_i] を満たす．
- その結果，族の中の任意の r について「平均の KL(q ∥ r) = 平均の KL(q ∥ p*) + 平均の KL(p* ∥ r)」が成り立つ．これは情報幾何の m 射影のピタゴラスの等式である．
- これが「標準の蒸留は，教師を線形ソフトマックスの族へ射影する」の正確な形になる．
- Charikar の不等式の分類版として使うには，真の事後確率が族に入るという仮定が要る．MER-DA ではこの仮定は成り立たない．

### 1.4 P4：最小二乗で写しても解が同じ（帰結としては既知，明示は見つからない）

- **論証**（本調査の導出．母集団の近似を使わない厳密な等式）：
  - MER-DA の目的関数は「K 本の交差エントロピー ＋ 疑似ラベルの柔らかい交差エントロピー（発話ごとの重み 1）＋ 出発点への L2 正則化」である．
  - このうち疑似ラベル q が入るのは −Σ_j q_jᵀ(W z_j + b) の項だけで，この項は Σ_j q_j [1, z_j]ᵀ（4 行 ×（次元 + 1）列の行列）だけで決まる．
  - 切片つきの最小二乗で q を [1, z] に写した q̂ は，正規方程式から Σ_j (q_j − q̂_j)[1, z_j]ᵀ = 0 を満たす．したがってこの行列は変わらず，目的関数はすべての W，b で同じ値になる．
  - K 本の項と L2 の項は q を含まないので，この等式は正則化と K 本があっても厳密に成り立つ．
- **既存の検算**：`research_notes/研究提案_2026-11-04/査読_3.md` 5 節が人工データ 10 種で確かめており，重みの差は最大 2.6×10⁻⁸ だった．検査役が人工データで出した数値で，関門は通していない．
- **先例**：
  - Patrini, Nielsen, Nock, Carioni 2016（ICML．arXiv 1602.02450）の定理 3 ◎（既存）は，2 値の線形分類器で，損失がラベルを含まない項と平均作用素 E[y x] の 1 次の項に分かれることを示す．系 5 で，平均作用素がラベルの十分統計量になる．
  - 多クラスの交差エントロピーへの拡張は直接だが，明示した文献は見つけていない．
  - 多クラスのロジスティック回帰の解が「特徴とラベルの積の平均」だけで決まること（積率一致）は教科書の事実である．原典（Berger, Della Pietra & Della Pietra 1996 など）は開いていない．
- **結論**：「最小二乗で写した疑似ラベルで学んでも，柔らかい疑似ラベルで学んでも，解は同じ」と述べた文献は見つからなかった．既知の事実から 2 行で出るので，論文では「注意（remark）」として書き，定理とは書かない．

### 1.5 P5：重みの付け方を替えても差が出にくい（先例は見つからない）

- **MER-DA の導出の状態**：
  - `DAI_paper/overview/DAI.md` 6.5.1 節の予言 (b) は，温度・硬いラベルのしきい値・感情ごとの釣り合いについてこう書く．これらは目標を E[h(q) ∣ z]（h は発話ごとの変換）に替えるか，発話に重みを付ける操作なので，結果は大きく変わらない．
  - **この導出は「目標が変わる」ところまでは正しい．しかし「大きく変わらない」の大きさを抑える式を持たない．**
  - この命題の先例も，同じ形の不変性を示した理論も，今回の調査では見つからなかった．
- **近い記述と反対向きの理論**：
  - **Kumar, Ma, Liang 2020「Understanding Self-Training for Gradual Domain Adaptation」**（ICML 2020．arXiv 2002.11361）の 3 節 ◎：自分の予測の柔らかいラベルでは更新が起きないので，"we "sharpen" the soft labels to encourage the model to update its parameters" と書く．5 節の実験では柔らかいラベルが硬いラベルより悪かった（"Soft labeling hurts"）．これは自分の予測の場合の話で，別の見方の教師で硬くする必要があるとは言っていない．
  - **Harutyunyan, Rawat, Menon, Kim, Kumar 2023「Supervision Complexity and its Role in Knowledge Distillation」**（ICLR 2023．arXiv 2301.12245）○：教師の予測が生徒の核の上位の固有方向にそろうほど学びやすいことを示す（supervision complexity YᵀK⁻¹Y が小さくなる）．温度を上げると予測がそろう．有限の標本では温度が効く，という向きの理論である．本文は該当語を検索しただけで，通読していない．
  - **Menon ほか 2021 の 3 節** ◎：ラベルの平滑化は偏りを入れる代わりに分散を下げる，と書く．重みの付け方の変更は，偏りと分散の交換として効きうる．
  - **Nigam & Ghani 2000 の表 6** ◎：News 2x2 の誤り率は次のとおりである．ラベルの付け方の差より，見方の分け方の差のほうが大きい．ただし co-training は少数ずつ足す点も co-EM と違うので，柔らかいか硬いかだけの比較ではない．
    - 見方を分ける 2 手法：co-EM（全部のラベル無しデータに確率のラベル）3.3%，co-training（少数ずつ硬いラベル）3.7%．
    - 見方を分けない 2 手法：EM 8.9%，self-training 5.8%．
- **結論**：
  - MER-DA の実測では，重みの付け方を替えて過半のコーパスで差が出た（分類器の種類，K）の組は 21 のうち 0 だった（`DAI_paper/inventory.md` 行 10，照合済み）．
  - 文献には同じ形の報告も，反例の検証も無い．凍結した特徴の上の線形の生徒で，ラベル無しの発話が数百〜数万本ある設定の実測として報告できる．
  - ただし理由は「目標の変化が小さいと見込まれる」という近似の説明であり，命題とは書けない．

### 1.6 P6：自分の予測では得が無く，別の見方なら得がある（既知）

**Kumar ほか 2020 の例 3.6** ◎
- 原文："A simple example shows that this form of self-training may never update the parameters because θ minimizes Lσ,θ: Example 3.6. For all θ ∈ Θ, θ is a minimizer of Lσ,θ ... Note that this is true even on finite data"．
- 線形のロジスティック回帰で，自分の予測を柔らかいラベルにした交差エントロピーの最小解は，自分自身である．
- MER-DA の目的関数への当てはめ（本調査）：
  - 出発点から K 本で学び直した分類器の予測を，同じ正則化の強さで，重み 1 の柔らかい疑似ラベルにする場合を考える．
  - その分類器は，K 本の項と L2 の項の最小解である．同時に，疑似ラベルの項の最小解でもある．したがって解はまったく動かない．
  - 動きうるのは，硬くする変種と釣り合いを取る変種だけである．

**ほかの文献**
- **Blum & Mitchell 1998 の定理 1** ◎（既存）：2 つの見方が感情を与えたとき条件つき独立なら，片方の予測は他方にとって，クラスごとに率の違う裏返しの雑音になる．このため，ラベル無しデータだけから学べる．
- **Nigam & Ghani 2000 の要旨と表 6** ◎：原文 "algorithms explicitly leveraging a natural independent split of the features outperform algorithms that do not"．
- **Lopez-Paz, Bottou, Schölkopf, Vapnik 2016（ICLR）の実験 2 と 4.3 節** ◎（既存．`DAI_paper/参照論文/要点_問い1_別モダリティ蒸留.md`）：教師の情報が生徒の特徴の雑音と独立なら，得は無い．特権情報が p(y ∣ x) について新しい情報を足すときだけ得がある．
- **Yang ほか 2022 の注意 2** ◎：
  - 原文："when z is independent of x, the predictions from the GenD's teacher are not learnable for the student"．
  - 特権の特徴だけで学ぶ教師（GenD）は，生徒の特徴と特権の特徴の両方で学ぶ教師（PFD）より悪い．
  - MER-DA の記録（`DAI_paper/inventory.md` 行 9，検査前）では，声の配備する分類器に対して，K=0 の疑似ラベルの得が顔だけのモデルで +1.3，融合のモデルで +3.7 だった．**これは Yang の GenD と PFD の差と同じ向きで，同じ説明が当てはまる．**その説明とは，他のモダリティだけの教師の判断は，声の特徴から予測しにくい，というものである．
- **Wang & Zhou 2007（ECML，pp. 454–465）** △（検索の断片だけ）：2 つの学習器の違い（diversity）が大きければ，見方が 1 つでも共訓練は成功しうる．違いが 0 の極限が自己学習にあたる．

---

## 2. 最も近い先行研究 5 本の比較

| 文献 | 設定（データ，ラベルの量，モダリティ，学習器） | 測る量 | 主張 | MER-DA と同じ点 | MER-DA と違う点 |
|---|---|---|---|---|---|
| **Yang ほか 2022（PFD）** ◎ | ランキング（公開 3 本と Amazon の記録）．理論は線形回帰．ラベル付き n 本とラベル無し m 本．教師は通常の特徴＋特権の特徴，生徒は通常の特徴だけ | 生徒の重みの推定誤差（定理 1・2），ランキングの指標 | 教師は特権の特徴で説明できる分散を消すので，生徒は予測できる部分に集中できる．特権の特徴が強すぎると教師の予測の分散が増え，生徒が悪くなる（非単調） | 教師が生徒の特徴を含む．ラベル無しのデータに教師の予測を付ける．線形の生徒．「特権の特徴だけの教師は弱い」 | 回帰で 2 乗損失．ドメインのずれが無い．出発点への正則化が無い．重みの付け方の比較が無い．感情の対ごとの分解が無い |
| **Vapnik & Izmailov 2015** ◎ | 特権情報の理論．航空写真の 2 クラスの例（生徒 15 次元，教師 116 次元，学習 64〜192 本） | 誤り率，収束の速さ | 教師の知識は「生徒の空間での条件つき期待値（frame の像）」として回帰で移せる．Oracle Teacher は slack を渡して収束を 1/ℓ にする | 条件つき期待値という同じ量．教師は生徒の入力空間の関数を渡す，という見方 | 条件つき期待値を新しい特徴として足し，ラベルで学び直す．蒸留（教師の確率を目標にする）とは比べていない．線形の生徒では，線形回帰の像は新しい特徴にならない |
| **Charikar ほか 2024** ◎ | 回帰の理論．強いモデルと弱いモデルは別の表現．強いモデルは凸の関数集合の上で，弱いモデルの出力に 2 乗損失で合わせる．実験は合成データと実データ | 強いモデルの誤差，弱いモデルの誤差，両者の食い違い（misfit） | 強いモデルは弱いモデルを凸集合へ射影したもので，真の関数が集合に入るとき，得は食い違い以上になる | 「蒸留は，教師を生徒の仮説集合へ射影すること」 | 2 乗損失．真の関数が生徒の集合に入る仮定．分類と交差エントロピーへの広げ方（Mulgund & Pabbaraju 2025）は逆向きの KL で，MER-DA の手法と違う．MER-DA は出発点に引き戻すので，得は「教師の正解率 − 出発点の正解率」で上から抑えられる（`docs/related-work.md` 1.14 節 (a)） |
| **Kumar ほか 2020** ◎ | 段階的なドメイン適応の自己学習．理論は線形のロジスティック回帰（2 値），実験は回転 MNIST ほか | 誤り率の上界 | 自己学習には正則化と，ラベルを鋭くすることが要る．自分の柔らかいラベルでは更新が起きない（例 3.6） | 線形のロジスティック回帰，ドメインのずれ，正則化．自分の予測では情報が増えないことの厳密な形 | 別の見方の教師を扱わない．ターゲットの少数ラベルが無い．マルチモーダルではない |
| **Menon ほか 2021** ◎ | 蒸留の統計的な見方の理論．教師と生徒は同じ入力．画像の分類と検索 | 蒸留のリスクの分散，偏りと分散の上界（命題 3） | 教師はベイズの確率の推定で，one-hot のラベルを置き換える．ベイズの教師は分散を下げる | 「教師はラベルの代わり」の明示．教師の偏りが生徒の誤差に入る | 教師と生徒の入力が同じ．別モダリティの教師の「ベイズの確率」が，生徒の特徴の上の P(y ∣ z) と違うことは扱わない（1.1 節の読み替えは本調査の読み） |

表に入れなかった文献と，その理由は次のとおりである．
- Dong ほか 2025・Xue ほか 2025・Wu ほか 2026 は，既存の調査（`I_第4回.md`，`J_第5回.md`）で比較済みである．
- Xue ほか 2023（MFH）と Lopez-Paz ほか 2016 は，`要点_問い1_別モダリティ蒸留.md` で全文を読んで比較済みである．
- Blum & Mitchell 1998 と Nigam & Ghani 2000 は，P1・P6 の根拠として 1 節に書いた．

---

## 3. 上澄みの判定：MER-DA の記録のうち，A の範囲の先行研究を超えるもの

期待する結論に合わせずに判定した．**理論として先行研究を超えるものは無い．**残るのは次の 3 つで，どれも既知の結果の適用か，実測による確認である．

### 3.1 出発点への L2 正則化と K 本を含めても成り立つ注意書き（小さい．適用）

- **内容**：MER-DA の目的関数では，疑似ラベルは Σ_j q_j [1, z_j]ᵀ を通してしか入らない．ここから 2 つの帰結が出る．
  - (i) 教師の確率を配備する分類器の特徴へ最小二乗で写してから学んでも，解は厳密に同じになる．
  - (ii) 自分の予測（同じ正則化で学んだもの）を，重み 1 の柔らかい疑似ラベルにしても，解は動かない．
- **先行研究との関係**：
  - (i) は Patrini ほか 2016 の平均作用素の多クラス版から 2 行で出る．明示した文は見つからなかった．
  - (ii) は Kumar ほか 2020 の例 3.6 を，出発点への L2 と K 本を足した目的関数に広げたもので，広げ方は自明である．
  - Vapnik & Izmailov の「知識の移し」を線形の生徒に当てると，線形回帰の像は元の特徴の空間の中にあるので，新しい特徴にならない．教師は目標としてしか働かない．これも 2 行の帰結である．
- **価値**：査読者が「最小二乗で写した教師を使う変種や，Vapnik & Izmailov の方法と比べたか」と問うたときの答えになる．新しさとしては書けない．

### 3.2 重みの付け方を替えても差が出ないことの実測（小さいが文献に無い）

- **記録**：
  - 疑似ラベルの重みの付け方（柔らかい，温度，硬い，釣り合い）を替えても，過半のコーパスで差が出た（分類器の種類，K）の組は 21 のうち 0 だった（`DAI_paper/inventory.md` 行 10，照合済み）．
  - 同じ疑似ラベルのモデルを渡した温度つき蒸留と比べ，感情の対ごとに数えた．手法が向きの損を減らした 72 のうち 63 は，温度つき蒸留でも減っていた．手法と温度つき蒸留の直接の差は，減らした 4・増やした 7・同じ水準 73 だった（`docs/exploratory-analyses.md` 10222 行以降．関門 2 の照合を経た記録）．
- **先行研究との関係**：温度やラベルの鋭さが効くという理論（Harutyunyan 2023，Kumar 2020 の自分の予測の場合）と実測は多い．しかし，凍結した特徴の上の線形の生徒で，別モダリティの教師について「効かない」ことを測った報告は見つからなかった．
- **書き方の制約**：導出は近似の説明であり，「目標 E[h(q) ∣ z] の変化が小さい」こと自体は測っていない．論文には次のように書き，命題とは書かない：「線形の生徒は E[q ∣ z] の線形ソフトマックスへの射影を学ぶので，発話ごとの変換は目標を少ししか動かさないと見込まれ，実測はこの見込みと合う」．

### 3.3 自分の予測を疑似ラベルにする対照との差（既知の予言の確認）

- **記録**：
  - 手続きを同じにし，疑似ラベルのモデルだけを配備する分類器自身の予測に替えた対照を置いた．声の配備する分類器で，この対照が感情の対の向きの損を減らした単位は 0，増やした単位は 2 だった．
  - 手法とこの対照の直接の差は，減らした 75・増やした 2 だった．コーパスと対の塊の数 25 は，偶然の基準 4.0 を超えた（`docs/exploratory-analyses.md` 10220〜10226 行．関門 2 の照合を経た記録．設計に使ったターゲットでの値）．
  - 同じモダリティのラベル無しの 8 手法では，方向の誤りを減らした単位が 210 のうち 7（偶然の見込み 5.25），増やした単位が 79 だった（`DAI_paper/inventory.md` 行 5，照合済み）．
- **先行研究との関係**：Kumar 2020 の例 3.6（自分の柔らかいラベルでは動かない）と，共訓練の文献（Blum & Mitchell，Nigam & Ghani）の予言どおりの結果である．コーパス間・少数ラベル・マルチモーダルの感情認識で，同じ手続きの中で教師だけを替えた対照を置いた報告は見つからなかった．ただし新しさは設定の違いだけである．
- **注意**：
  - 対照の候補の設定は 20 個で，手法の候補（声で 60〜200 個）より少ない（`docs/exploratory-analyses.md` 10145 行）．
  - 重み 1 の柔らかい変種は，理論上まったく動かない．したがって対照の中で動いたのは，釣り合いの変種か硬いラベルの変種のはずである（確かめていない）．

### 3.4 上澄みになりうるが，今は記録が無いもの

- 次の 2 つを同じテストで並べる解析は，設計と台本があるが，計算していない（`docs/exploratory-analyses.md` 10418 行以降）．
  - 疑似ラベルのモデルが出発点を上回る幅のうち，配備する分類器の特徴で表せる分．
  - 実際に受け取った得．
- 同じ種類の量を，Dong 2025 は特徴の共分散の重なりで，Xue 2025 は核の射影で，MFH は共通の手がかりの割合で測っている．しかし，**教師の判断を生徒の特徴へ写した分類器を実際に作り，分類の UAR で測った例は今回の調査では見つからなかった**．
- 計算して検査を通すまでは主張にできない．
- 名前の付け方にも注意が要る．`研究提案_2026-11-04/査読_3.md` は，この量が正解ラベルを教師にしても大きく出ることを人工データで示している．このため，この量を「他のモダリティだけの領域」とは読めない．

---

## 4. 論文での書き方の注意

1. **「ラベルの代わり」の見方は，既知の結果の適用として書く．**引く文献は次のとおりである．
   - Hinton ほか 2015，Menon ほか 2021：教師は one-hot のラベルを置き換える．
   - Blum & Mitchell 1998，Nigam & Ghani 2000：別の見方の予測は雑音入りのラベルである．
   - Vapnik & Izmailov 2015 の 5.1.4 節：条件つき期待値の像．
   - Yang ほか 2022 の定理 1：特権の特徴を含む教師から線形の生徒への蒸留．
2. **「線形の配備する分類器は，E[q ∣ z] の線形ソフトマックスへの射影を学ぶ」は，損失が目標について線形であることからの 1 行の帰結として書く．**
   - Vapnik & Izmailov 2015 の 5.1.4 節と同じ量であることを明記する．
   - Charikar ほか 2024 を引くときは，2 乗損失・凸・realizability の仮定を併記し，交差エントロピーには直接当てはまらないと書く．
   - Mulgund & Pabbaraju 2025 は逆向きの KL なので，「標準の蒸留の射影」として引かない．
3. **「教師の誤差のうち，配備する分類器の特徴で表せる部分だけが移る」は，既知の結果として書き，新しさにしない．**Dong ほか 2025，Xue ほか 2025，Wu ほか 2026，Burns ほか 2023 の付録 E.2 を引く（`I_第4回.md` の判定と同じ）．
4. **最小二乗で写しても解が同じこと，自分の予測では解が動かないことは，「注意」として本文か付録に書く．**前者は Patrini ほか 2016 の定理 3 の多クラス版の帰結，後者は Kumar ほか 2020 の例 3.6 の帰結と書く．
5. **重みの付け方の不変性は実測として書き，説明は近似と断る．**
   - 「命題」「定理」とは書かない．
   - Harutyunyan ほか 2023 のように温度が効くとする理論があることに触れる．
   - そのうえで，違い（凍結した特徴の線形の頭，ラベル無しの発話が多い）を条件として書く．
6. **自分の予測の対照は，「既知の予言（Kumar 2020，共訓練）の確認」として書く．**疑似ラベルのモデルは配備する分類器のモダリティを含むので，「別のモダリティだけの情報に帰せる」とは書かない（`docs/exploratory-analyses.md` 10225 行の規則と同じ）．
7. **教師が配備する分類器のモダリティを含むことは，Yang ほか 2022 の PFD と同じ形であると書く．**
   - 条件つき独立（Blum & Mitchell の仮定）は構成から崩れるので，共訓練の定理を根拠に使わない（`DAI.md` 6.5.3 節と同じ）．
   - 他のモダリティだけのモデルが弱いことは，Yang の注意 2（特権の特徴だけの教師からは生徒が学べない）と同じ向きの結果として引ける．
8. Menon ほか 2021 の命題 3 の「教師の分散」を，「生徒の特徴を固定したときの教師の揺れ」に読み替える議論は本調査の読みである．使うなら「読み替えると」と断る．

---

## 5. 見つからなかったこと，調査の限界

### 5.1 見つからなかったこと

- 別モダリティの蒸留で，交差エントロピーと softmax の生徒について「理想の出力は E[教師の確率 ∣ 生徒の特徴]」を式として明示した文献は，今回の検索では見つからなかった．回帰と一般の関数の形では，Vapnik & Izmailov 2015 と Yang ほか 2022 がある．
- 「教師の確率を生徒の特徴へ最小二乗で写してから学んでも，解は同じ」を明示した文献は見つからなかった．
- 「疑似ラベルの重みの付け方（温度，硬いラベル，釣り合い）を替えても，線形の生徒では差が出にくい」という命題も，その実測の報告も見つからなかった．
- 凍結した特徴の上の線形の生徒で，別モダリティの教師と自分の予測を同じ手続きで比べた理論や実験は，Nigam & Ghani 2000（ナイーブベイズ，文書分類）のほかには見つからなかった．
- 教師の判断を生徒の特徴へ写した分類器を作り，教師の得のうち表せる分を分類の正解率で測った研究は見つからなかった．

### 5.2 調査の限界

- **原文を読んだ範囲**：
  - Menon 2021：抄録，2 節，命題 3 とその後の段落．
  - Charikar 2024：定理 1，3 節の証明．
  - Mulgund & Pabbaraju 2025：抄録，1 節，系 4.2 の周辺．
  - Kumar 2020：3 節の例 3.6，5 節の該当段落．
  - Yang 2022：抄録，1 節，5 節の定理 1・2，注意 2．
  - Lang 2024：抄録，1 節．
  - Somerstep 2025：抄録．
  - Phuong & Lampert 2019：4.1 節の定理 1 の周辺．
  - Xu ほか 2026：抄録，1・3・4 節の断片．
  - Nigam & Ghani 2000：要旨，5 節，表 6．
  - Vapnik & Izmailov 2015：3.1 節，5 節の冒頭，5.1.4・5.1.5・5.2 節．
  - Wang 2019「Everything old is new again」，Ortiz-Jimenez ほか 2023：抄録だけ．
  - Xie・Xu・Sanguinetti 2025：抄録と 3 節の冒頭．
- **抄録か検索の断片だけのもの**：
  - Harutyunyan 2023（○）：本文は該当語を検索しただけ．
  - Wang & Zhou 2007・2010（△）：PDF を取得できなかった．
  - Ando & Zhang 2007（○）：抄録だけ．PDF を取得できなかった．
  - Sridharan & Kakade 2008（○）：掲載先が COLT か NeurIPS かで記録が食い違う．
  - Mobahi ほか 2020（○），Das & Sanghavi 2023（△），積率一致の原典（背）．
- **掲載先の未確認**：Yang 2022 の NeurIPS 2022，Menon 2021 の ICML 2021，Kumar 2020 の ICML 2020 は，原典の頁で確かめていない．引用の前に確かめる．
- **読み直していないもの**：既存の調査で ◎ と判定された Dong 2025，Xue 2025，Wu 2026，Liu・Dong・Lei 2026，Burns 2023，Patrini 2016，Blum & Mitchell 1998，Lopez-Paz 2016，Xue 2023 は読み直していない．
- **検索の範囲**：検索は英語だけである．2026 年の新しい arXiv の論文は，検索に出たものに限る．
- **本調査の導出**：1.3 節の m 射影のピタゴラスの等式と，3.1 節の 2 つの帰結は本調査の導出である．検算は既存の人工データの記録（`査読_3.md`）に頼った．新しい計算はしていない．`results/` の下は開いていない．
- **担当外**：B（マルチモーダル・感情認識・DA の実証）と C（ラベル換算と，位置・方向の分解）の範囲の文献は深追いしていない．

---

## 出典（URL）

- Menon ほか「Why distillation helps: a statistical perspective」 https://arxiv.org/abs/2005.10419
- Charikar, Pabbaraju, Shiragur「Quantifying the Gain in Weak-to-Strong Generalization」 https://arxiv.org/abs/2405.15116
- Mulgund & Pabbaraju「Relating Misfit to Gain in Weak-to-Strong Generalization Beyond the Squared Loss」 https://arxiv.org/abs/2501.19105
- Xu, Yao, Wang, Liu「Weak-to-Strong Generalization via Bregman Bias–Variance Decomposition」 https://arxiv.org/abs/2505.24313
- Kumar, Ma, Liang「Understanding Self-Training for Gradual Domain Adaptation」 https://arxiv.org/abs/2002.11361
- Yang ほか「Toward Understanding Privileged Features Distillation in Learning-to-Rank」 https://arxiv.org/abs/2209.08754
- Lang, Sontag, Vijayaraghavan「Theoretical Analysis of Weak-to-Strong Generalization」 https://arxiv.org/abs/2405.16043
- Somerstep ほか「A Transfer Learning Framework for Weak to Strong Generalization」 https://arxiv.org/abs/2405.16236
- Phuong & Lampert「Towards Understanding Knowledge Distillation」 https://arxiv.org/abs/2105.13093
- Vapnik & Izmailov「Learning Using Privileged Information: Similarity Control and Knowledge Transfer」 https://jmlr.org/papers/v16/vapnik15b.html
- Nigam & Ghani「Analyzing the Effectiveness and Applicability of Co-training」 http://www.kamalnigam.com/papers/cotrain-CIKM00.pdf
- Harutyunyan ほか「Supervision Complexity and its Role in Knowledge Distillation」 https://arxiv.org/abs/2301.12245
- Wang「Everything old is new again: A multi-view learning approach to learning using privileged information and distillation」 https://arxiv.org/abs/1903.03694
- Ortiz-Jimenez ほか「When does Privileged Information Explain Away Label Noise?」 https://proceedings.mlr.press/v202/ortiz-jimenez23a.html
- Xie, Xu, Sanguinetti「Information-Theoretic Criteria for Knowledge Distillation in Multimodal Learning」 https://arxiv.org/abs/2510.13182
- Wang & Zhou「A New Analysis of Co-Training」（ICML 2010） https://mlanthology.org/icml/2010/wang2010icml-new/
- Ando & Zhang「Two-view Feature Generation Model for Semi-supervised Learning」（ICML 2007） https://mlanthology.org/icml/2007/ando2007icml-two/
- Sridharan & Kakade「An Information Theoretic Framework for Multi-View Learning」 https://mlanthology.org/colt/2008/sridharan2008colt-information
- Mobahi, Farajtabar, Bartlett「Self-Distillation Amplifies Regularization in Hilbert Space」 https://arxiv.org/abs/2002.05715
- Dong ほか「Discrepancies are Virtue」 https://arxiv.org/abs/2502.05075（既存の調査で ◎）
- Xue, Li, Mirzasoleiman「Representations Shape Weak-to-Strong Generalization」 https://arxiv.org/abs/2502.00620（既存の調査で ◎）
- Wu ほか「What Makes a Strong Model?」 https://arxiv.org/abs/2606.01292（既存の調査で ◎）
- Patrini ほか「Loss factorization, weakly supervised learning and label noise robustness」 https://arxiv.org/abs/1602.02450（既存の調査で ◎）
- Lopez-Paz ほか「Unifying distillation and privileged information」 https://arxiv.org/abs/1511.03643（既存の調査で ◎）
