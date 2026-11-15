# 合併の切片と向き，複数ソースと少数ターゲットラベルの価値：理論の先行研究 9 本の精読

調べた日：2026-10-27．対象は主張 B（ソースの価値をターゲットのラベルの本数で測る．n = q e²/(1 − e²)）と主張 C（合併で切片のずれが λ = ρ/(1+ρ) の割合で直る．見分けやすいほど合併は暗黙にターゲット専用の切片を学ぶ）の置き場所を決めるための 9 本と，余力の 3 本（Ben-David 2010，Blitzer 2007，Sarao Mannelli 2025）．
確かめた水準：◎ 本文（該当箇所を読んだ），○ 抄録か付随の文書，△ 検索結果の抜き書きだけ．
記号の対応：本研究の o（切片），d（向き），K（クラスあたりのターゲットの本数），ρ（ターゲットの比重），λ（切片の直る割合），n（換算した本数），q（向きの実効次元），e（向きの効率）．
原文の引用は英語のまま短く入れる．PDF は `/tmp/claude-1000/-home-snics-MER-DA/2f47b0d7-4ef9-4797-9837-84fb3de61717/scratchpad/g5_pdfs/` に置いた（作業用．消えてよい）．

前回のノート（`research_notes/TAC 論文の先行研究と書き方/theory_preemption.md`）との食い違いで直すべき点を先に書く．
- **Yang ほか 2025（JMLR）の結論にロジスティック回帰を今後の課題に挙げた文は無い．**その文があるのは arXiv の初版（2010.11750v1，2020-10，題は「Sharp Bias-variance Tradeoffs of Hard Parameter Sharing in High-dimensional Linear Regression」）の 5 節である．JMLR 版の今後の課題は，共変量と模型の両方のずれでの偏りの極限（Remark 7），非ガウス，非等方，ソフトな転移，大きな模型の微調整である．論文で引くなら「arXiv 版の初版で分類への拡張を今後の課題に挙げた」と書くか，引かない．
- **Tiomoko ほか 2021 の偏りの正体は 2 つある．**(1) 判定の閾値 ζ_i = (m_i1 + m_i2)/2 が一般に 0 でないこと，(2) ±1 のラベル（スコア）の選び方が分散を増やして負の転移を起こすこと．さらに重要なのは，Tiomoko らは学習の前に**課題ごとに平均を引き**，課題ごとの切片 b_i を模型に持つので，本研究の主張 C が扱う「合併が作る切片のずれ」は構成上はじめから無い．「同じ現象」とは書けない．

---

## 論文 1：Tiomoko, Tiomoko Ali, Couillet 2021（ICLR）Deciphering and Optimizing Multi-Task Learning: a Random Matrix Approach

### Takeaway
課題ごとに平均を引いた LS-SVM 型の多課題学習を，標本数と次元が比例して大きくなる極限で解析し，合併型の偏りを「判定の閾値が 0 からずれる」と「±1 のラベルが分散を増やして負の転移を起こす」の 2 つに分け，どちらも閉じた形で直した．切片のずれを合併が直す仕組み（主張 C）は，課題ごとの中心化と課題別の切片を置くことで扱いの外に出している．

### Cited Findings
**1. 書誌と読んだ範囲**
- ICLR 2021．著者は Malik Tiomoko, Hafiz Tiomoko Ali, Romain Couillet．OpenReview の id は Cri3xz59ga（○．ML Anthology の頁で題・著者・会議を確かめた．OpenReview 本体はブラウザの確認画面で取得できず，**ICLR 版の本文は未読**） — [ML Anthology](https://mlanthology.org/iclr/2021/tiomoko2021iclr-deciphering)
- 読んだのは arXiv の長い版（2009.01591v1，2020-09-03，53 頁，題は「Large Dimensional Analysis and Improvement of Multi Task Learning」）．要旨，1 節，2.1 節（模型と Remark 3），2.2 節（Assumption 1–2），3 節（Theorem 4，3.2 節の閾値とラベルの最適化，Remark 5–7，Algorithm 1），4 節の Remark 9–10，6.1.1–6.1.2 節，7 節（◎）．未読：4 節の Theorem 8 の導出，5 節（仮説検定と多クラス），6.2 節，付録 — [arXiv 2009.01591](https://arxiv.org/abs/2009.01591)

**2. 設定**
- k 個の課題，各 m クラス．データは「濃縮したランダムベクトル」の混合（ガウス混合を含む）で，クラス j・課題 i の平均 µ_ij と共分散 Σ_ij を持つ（Assumption 1）．極限は n, p → ∞，n/p → c₀，n_ij/n → c_ij（Assumption 2） — arXiv 2.2 節
- 分類器は MTL-LSSVM：W_i = W_0 + V_i（Evgeniou & Pontil 2004 の形），**課題ごとの切片 b_i** を持ち，罰則は tr(W_0ᵀW_0)/(2λ) と tr(V_iᵀV_i)/(2γ_i)，損失は 2 乗（式 (1)）．**学習の前に課題ごとに平均を引く**：X̊_i = X_i(I − 1 1ᵀ/n_i)．理由は "centering the whole dataset induces dependencies across tasks ... negative transfer can still appear" と "class-wise centering has the double deleterious effect of cancelling an important discrimination factor" — arXiv 2.1 節
- 主結果は m = 2 かつ Σ_ij = I_p に限って書かれ，その場合の十分統計量は内積 Δµ_iᵀΔµ_i'（Δµ_i = µ_i1 − µ_i2）と比 c_ij/c_i と超パラメータだけである — arXiv 3.1 節
- ずれの型：課題ごとのクラス平均の差の向き（内積），クラスの比率．課題ごとの平均の位置のずれは中心化で消している．ソースの数：k − 1（任意）．実験は 2 課題と課題を増やす実験

**3. 示したこと**
- Theorem 4：スコア g_i^bin(x) は N(m_ij, σ_i²) に収束し，m = y − Γẙ，σ_i² = ẙᵀΓV_iΓẙ（2k 次元の行列 M, A, Γ で決まる） — arXiv 3.1 節
- 偏り (1)：判定の閾値．"we show that this decision rule is in general biased, not only due to imbalances in the number of available data per class and per task, but also by the data statistics and the MTL hyperparameters (unless Σij = Ip for each i, j)"．最適な閾値は ζ_i = (m_i1 + m_i2)/2（式 (6)），誤り率は Q((m_i1 − m_i2)/(2σ_i))（式 (7)）．クラスと課題の標本数が全部等しいときだけ ζ_i = 0 — arXiv 1 節と 3.2 節
- 偏り (2)：ラベルの選び方．"a fundamental aspect of the (LS)SVM framework is to associate each training data x to a label y ∈ {−1, 1}; we demonstrate that this choice is a dangerous source of negative transfer"．最適なスコア y* は式 (8) で閉じた形．関係の無い課題の最適スコアは 0 に落ち，"not by altering the means mi1, mi2 but by increasing the variance σi²" で害が出る．Δµ_iᵀΔµ_i' < 0 なら符号が反転する — arXiv 3.2 節
- Remark 3（スコアの平行移動不変）：課題内の全スコアに定数を足しても成績は変わらず，閾値だけが動く．Remark 5 で閾値を 0 に戻す操作を与える — arXiv 2.1・3.2 節
- Remark 7：十分統計量は "a single sample (two when i = i′) per task and per class (|Sil| = 1) is sufficient to obtain a consistent estimate for all quantities, so long that p is large"．収束は O(1/n)（通常の中心極限の 2 乗の速さ） — arXiv 3.3 節
- 実験：人工データ（p = 100，c = [0.3, 0.4, 0.1, 0.2]，γ = 1/2，λ = 10）で理論と実測のスコア分布が一致し，"the deviation from 0 of the decision threshold in unbalanced classes" を図 2 で示す．MNIST（ソース各クラス 100，ターゲット各クラス 10，PCA で p = 100，テスト 1,135）で λ の全域で最適ラベルの方が良い（図 3）．Office+Caltech（10 クラス，VGG 特徴）の 12 組の平均は LSSVM 65.60，CDLS 69.37，提案 69.88 — arXiv 6.1 節と 5 節末の表
- 2 乗損失を選ぶ根拠として "quadratic (least-square) cost functions are asymptotically optimal ... and uniformly outperform alternative costs (such as margin-based methods or logistic approaches)"（Mai and Liao 2019 を引く）と書く — arXiv 1 節

**4. 軸との対応**
- A1 ○：LS-SVM（線形）．ロジスティック回帰ではない
- A2 △：誤差は「平均 m_ij と分散 σ_i²」に分かれ，閾値 ζ（切片）と向き（分散を通して）に対応づけられるが，本研究の o・d の分け方とは違う
- A3 ×：ソースをターゲットの本数に換算しない
- A4 △：合併型の偏りを比例の極限で厳密に解いた．ただし課題ごとの中心化と課題別の切片 b_i を持つので，合併が作る切片のずれ（主張 C）は無い．ニュートン法の 1 歩ではなく決定論的な極限
- A5 △：極限は比例だが，Remark 7 で「1 本で十分」と述べ，MNIST でターゲット各クラス 10 本を使う
- A6：クラス平均の向きのずれ・クラス比率．等方（Σ = I）が主結果．ソース数は任意（k − 1）
- A7 ×．A8 ×．A9 △（W_i = W_0 + V_i は共有＋個別の罰則で，ソースの解への錨づけではない）
- A10 ○：MNIST，Office+Caltech，人工データ
- A11 ○：理論の分布と実測の一致．距離の 2 つの働きは無い（関係の無い課題は捨てるのが最適）

**5. 穴**
- 自ら書く限界："imposes by definition that all Vi's be of the same size Rp×m: this severely constrains (i) the data in each task to be of the same dimension p and (ii) the number of classes per task to be the same (m)"．非線形の緩和は "goes beyond the scope of the present article" — arXiv 2.1 節．Σ_ij の推定は "known to be biased, particularly so if nij is not large compared to p"（Remark 10）
- 扱っていない：固定の K の有限標本の式，合併の切片，ソースの価値の換算，ロジスティック回帰，マルチモーダル

### Inferences
- **6. 差の書き方（案）**：「Tiomoko らは課題ごとに平均を引いた上で課題別の切片を持つ LS-SVM を比例の極限で解析し，合併型の偏りを判定の閾値のずれとラベルの符号が招く負の転移に分けて直した．本研究は中心化しない単純な合併を扱い，切片のずれが ρ/(1+ρ) の割合で直る仕組みを，ロジスティック回帰のニュートン法の 1 歩で書く．」
- 本研究の式が Tiomoko らのどの式を書き直したものかというと，直接の対応は無い．対応するのは考え方だけで，Remark 3（スコアの平行移動は閾値だけを動かす）は本研究の「位置のずれは切片だけに出る」と同じ見方である（この対応は本ノートの読み）
- Tiomoko らが「課題ごとの中心化」を負の転移を避ける前処理として選んだことは，本研究の E12 の中心合わせと同じ操作を，理論側が最初から前提にしていることを意味する．本研究の主張 C は「中心化しないとき，合併がどれだけ自力で切片を直すか」を測る点で補い合う

### Gaps
- **7. 注意**：ICLR 版（8 頁＋付録）と arXiv 長版の内容の差は確かめていない．ICLR 版の要旨（ML Anthology）は「ガウス混合」「標準の MTL LS-SVM は強く偏る」「補正」「交差検証無しで超パラメータを選べる」で長版と一致する．論文で引く節番号は ICLR 版で付け直す必要がある
- Theorem 8（一般の Σ と m ≥ 2）と 5 節の多クラス拡張は読んでいない．多クラスで閾値のずれがどう扱われるかは未確認

---

## 論文 2：Yang, Zhang, Wu, Ré, Su 2025（JMLR 26）Precise High-Dimensional Asymptotics for Quantifying Heterogeneous Transfers

### Takeaway
2 つの線形回帰の標本を単純に合併した最小 2 乗（HPS）の偏りと分散を比例の極限で閉じた形にし，ランダム効果の模型でソースを増やすと得から損に変わる相転移を証明した．回帰で切片が無く，ターゲットの標本数は次元より大きい（n₂ ≥ (1+τ)p）ので，少数ショットの分類とは別の体制である．ロジスティック回帰への拡張は arXiv 初版（2020）の今後の課題にだけある．

### Cited Findings
**1. 書誌と読んだ範囲**
- JMLR 26 (2025) 1–88，論文番号 24-0454，2024-03 投稿，2025-04 改訂，2025-06 掲載，編集者 Mladen Kolar（◎．JMLR 版 PDF の冒頭） — [JMLR 24-0454](https://jmlr.org/papers/volume26/24-0454/24-0454.pdf)
- arXiv 2010.11750 は v1（2020-10-22．題「Sharp Bias-variance Tradeoffs of Hard Parameter Sharing in High-dimensional Linear Regression」，著者順は Zhang, Yang, Wu, Su, Ré）から v5（2025-06-09）まである（◎ v1 の冒頭と 5 節，v5 の冒頭） — [arXiv 2010.11750](https://arxiv.org/abs/2010.11750)
- 読んだのは JMLR 版の要旨，1 節（1.2–1.4），2 節，4 節（Theorem 6，Remark 7，Proposition 8，Theorem 9），5.2 節（Proposition 14），6.3–6.4 節，7 節．未読：3 節の Theorem 3 の詳細，5.1 節，付録 A–E（証明）

**2. 設定**
- 線形回帰 Y^(i) = X^(i)β^(i) + ε^(i)，X^(i) = Z^(i)(Σ^(i))^{1/2}，雑音の分散 σ²．**切片は無い**．HPS は合併した最小 2 乗で，分散は Tr(Σ^(2)(X^(1)ᵀX^(1) + X^(2)ᵀX^(2))⁻¹)（式 (1.3)），偏りは Lemma 2 — JMLR 1.2・2.2 節
- 標本数の体制：ρ₁ = n₁/p，ρ₂ = n₂/p で，**1 + τ ≤ ρ₂**（ターゲットだけで OLS が解ける），0 < ρ₁/ρ₂ ≤ τ⁻¹（式 (2.5)） — JMLR 2.1 節
- ずれの型：共変量のずれ（Σ^(1) ≠ Σ^(2)）と模型のずれ（β^(1) ≠ β^(2)）．ランダム効果の模型 β^(i) = β₀ + γ_i，γ_i の各成分の分散は µ²/p．ソースの数は 2 課題が主で，5.2 節で共通の X を持つ多課題に広げる

**3. 示したこと**
- Theorem 6（模型のずれ，Σ 共通，Z がガウス）：L_var = σ² p/(n₁ + n₂ − p)，L_bias = ‖Σ^{1/2}(β^(1) − β^(2))‖² · [n₁²(n₁ + n₂ − p) + p n₁ n₂]/[(n₁ + n₂)²(n₁ + n₂ − p)] — JMLR 4.1 節
- Proposition 8（相転移，n₂ ≥ 3p）：µ² Tr[Σ]/p ≤ σ²p/(2(n₂ − p)) なら任意の n₁ で得，σ²p/(2(n₂ − p)) < µ² Tr[Σ]/p < σ²n₂/(2(n₂ − p)) なら閾値 n₀ があって n₁ ≤ n₀ で得・n₁ > n₀ で損，µ² Tr[Σ]/p ≥ σ²n₂/(2(n₂ − p)) なら任意の n₁ > p で損．"HPS first improves as n1 increases in [0, n0] but turns worse as soon as n1 > n0" — JMLR 4.2 節
- Theorem 9（最小最大の下界）：c · min{n₁²µ²/(n₁ + n₂)² + σ²p/(n₁ + n₂), σ²p/n₂}．ソースを捨てるかどうかを選ぶ「調整した HPS」が定数倍で一致する — JMLR 4.3 節
- Proposition 4（共変量のずれの 2 つの働き）："when n1 < n2, the presence of covariate shifts can be beneficial for HPS. On the other hand, when n1 > n2, the presence of covariate shifts can hurt" — JMLR 1.2・3 節
- SPS（β^(1) = B + z に λ‖z‖² の罰則）："the regularization effect of SPS is similar to downsizing the source task in HPS when the model shift is large" — JMLR 1.2・6.4 節
- 検証は数値実験（p = 100，n₂ = 300，σ = 1/2）で，"Simulations validate the accuracy of the high-dimensional asymptotics for finite dimensions"．JMLR 版に実データは無い．arXiv v1 は 6 つのテキスト分類で段階的にデータを足す手続きを検証していた — JMLR 要旨，arXiv v1 の要旨と 4.2 節

**4. 軸との対応**
- A1 ○：合併した最小 2 乗（線形）．分類ではない
- A2 ×：切片が無く，偏りは β の向きの誤差だけ
- A3 △：相転移の閾値 n₀ はあるが，ソースをターゲットの本数に換算する式は無い．分散の項 σ²p/(n₁ + n₂ − p) は「分散についてはソース 1 本＝ターゲット 1 本」と数えている（本ノートの読み）
- A4 △：合併の推定量の厳密な極限．切片は無い．ニュートン法の 1 歩ではない
- A5 ×：n₂ ≥ (1 + τ)p を仮定するので，クラスあたり数本の体制の外
- A6：共変量のずれと模型のずれ（概念のずれ）．2 課題と，X を共有する多課題
- A7 ×．A8 ×．A9 △（SPS は β の差への L2 罰則で，ソースの解への錨づけに近い）
- A10 ×（JMLR 版は数値実験のみ）．A11 ○：理論と数値実験の一致，n₁ とずれの非単調性（得→損）

**5. 穴**
- 今後の課題（JMLR 版）："Showing the asymptotic bias limit under arbitrary covariate and model shifts is an interesting open problem for future work (See Remarks 7 and 13)"．Remark 7 は非ガウスへの拡張，5 節は非等方の場合，6.4 節は "a thorough understanding of this phenomenon, including soft transfer, for future work" と大きな模型の微調整 — JMLR 1.3・4.1・6 節
- 分類への拡張（arXiv v1 の 5 節のみ）："it would be interesting to extend our results to classification problems. Several recent work has made remarkable progress for logistic regression in the high-dimensional setting (e.g. Sur and Candès (2019)). It is an interesting question to study logistic regression in a multiple-sample setting." — [arXiv 2010.11750v1](https://arxiv.org/abs/2010.11750v1)

### Inferences
- **6. 差の書き方（案）**：「Yang らは 2 つの線形回帰を合併した最小 2 乗の偏りと分散を比例の極限で閉じた形にし，ソースを増やすと得から損に変わる相転移を示した．本研究は分類（L2 罰則つきロジスティック回帰）でターゲットがクラスあたり K 本（4K ≪ p）の体制を扱い，誤差を切片と向きに分けて，合併が切片のずれを直す割合を 1 段の近似で書く．」
- 本研究の式は Yang らのどの式の書き直しでもない．対応するのは「合併の得と損は分散の減少と偏りの増加の差で決まる」という枠（2.3 節）で，本研究の MH2（向きの合い方）と MH3（位置のずれ）がそれぞれ分散と偏りに当たる

### Gaps
- **7. 注意**：前回のノートの「結論でロジスティック回帰を今後の課題に挙げた」は JMLR 版には無い．arXiv v1 にだけある．論文で引くならその旨を書く
- Theorem 3（共変量のずれの分散の極限）の式と 5.1 節（両方のずれ）は読んでいない

---

## 論文 3：Zhu, Deng, Shi, Gangrade, Scott 2025/2026（arXiv 2510.04441）Domain Generalization Under Posterior Drift

### Takeaway
ドメインの情報（メタデータ）を特徴に足した ERM（DI-ERM）が，合併の ERM に勝つ条件をベイズ誤差の差で与えた．P(Y|X) がドメインで変わる（posterior drift）と差は γε/2 以上，どのドメインでも同じ分類器が最適なら差は 0 である．ターゲットのラベルは無く（DG），有限標本の率も線形の仕組みも無い．

### Cited Findings
**1. 書誌と読んだ範囲**
- arXiv 2510.04441．v1（2025-10）の題は「Domain Generalization: A Tale of Two ERMs」，v2（2026-02-04）の題は「Domain Generalization Under Posterior Drift」（◎ 両版の arXiv の頁の題） — [arXiv 2510.04441](https://arxiv.org/abs/2510.04441)；[v1](https://arxiv.org/abs/2510.04441v1)
- NeurIPS 2025 の頁（125209）は v1 の題で，**本会議ではなくワークショップ「Reliable ML from Unreliable Data」のポスター**である（○．頁の要約） — [NeurIPS 2025 の頁](https://neurips.cc/virtual/2025/125209)
- 読んだのは v2 の本文 1–7 節と付録 D（実験の詳細）．未読：付録 A–C（証明）

**2. 設定**
- (X, Y, M, D) の同時分布．D はドメイン（観測しない），M はドメインのメタデータで，P_{XY|D,M} = P_{XY|D}（式 (1)）．3 つのベイズ誤差 R*_pool（x だけ），R*_DI（x と m），R*_full（x と d）．合併 ERM（式 (2)）と DI-ERM（式 (3)，f(x, m)）．分類器の族は理論では制限しない — 3・4 節
- posterior drift の族 Π(γ, ε)：全ての (x, m) で点ごとのマージン γ(x, m) ≥ γ，かつ独立に引いた M, M' で f*_DI(X, M) ≠ f*_DI(X, M') となる確率 ≥ ε（Definition 2） — 5.1 節
- ターゲットのラベルは無い．ソースの数は N ドメイン（実験では注釈者 1,481 人など）

**3. 示したこと**
- Theorem 1：E[γ(X, M)·1{f*_pool(X) ≠ f*_DI(X, M)}] ≤ R*_pool − R*_DI ≤ P(f*_pool ≠ f*_DI)．"domain information is particularly beneficial when f*_DI frequently disagrees with f*_pool in regions of high confidence" — 5 節
- Theorem 2：inf over Π(γ, ε) of (R*_pool − R*_DI) ≥ γε/2 — 5.1 節
- Corollary 1：全ドメインで同じ分類器が最適（UBC）なら R*_pool = R*_DI．Proposition 2：関数族を制限すると UBC でも DI が勝ちうるが，族を大きくすると差は消える — 5.3 節
- Remark 4："In contrast to pessimistic results in domain adaptation, where no method consistently outperforms vanilla ERM under posterior drift [Zhu et al., 2024, Wang et al., 2025], our work presents an optimistic view in domain generalization" — 5.1 節
- 実験：注釈者の不一致データ（Díaz 2018，59,235 文・1,481 注釈者，不一致約 40%）で BERT 微調整の合併 49.1 → DI 90.5．凍結特徴＋線形（scikit-learn のロジスティック回帰）では合併 45.85 → DI 46.42（線形），55.07 → 78.45（MLP）．WILDS-Amazon（レビュアー別）は 72.2 → 73.4．PACS（UBC に近い）では ViT-small で 1–2 点の得，ViT-large で差が消える（CLIP ViT-large は両方 98.6） — 6 節，付録 D，表 2・4・6

**4. 軸との対応**
- A1 △：理論はベイズ誤差（模型に依らない）．実験に凍結特徴＋線形がある
- A2 ×．A3 ×
- A4 △：合併 ERM とドメイン情報つき ERM の比較．ドメインの情報を明示的に足す．率も切片も無い
- A5 ×：ターゲットのラベルは無い（DG）
- A6：posterior drift（P(Y|X) のずれ）．多数のドメイン
- A7 △：文＋注釈者の属性の文字列，画像＋様式の文字列を結合する（2 つの入力だが，感情のマルチモーダルではない）
- A8 ×．A9 ×
- A10 ○：3 つの実データ．A11 ×（数値の予言と実測の照合は無い．「UBC では得が消える」の定性的な確認だけ）

**5. 穴**
- 自ら書く限界："A central practical requirement of DI-ERM is access to domain metadata; when such metadata is unavailable, unlabeled data from the target domain offers a natural alternative."
- 今後の課題："Future work might explore alternative ways of encoding domain information, and a broader range of DG benchmarks. Additionally, it would be interesting to explore other performance measures for DG" — 7 節

### Inferences
- **6. 差の書き方（案）**：「Zhu らは，ドメインの情報を特徴として明示的に足すと合併の ERM に勝つことを，P(Y|X) がドメインで変わる場合のベイズ誤差の差（γε/2 以上）で示した．本研究は特徴を足さない単純な合併を扱い，ソースとターゲットが特徴空間で見分けやすいほど，ロジスティック回帰が暗黙にターゲット専用の切片を学ぶ（直る割合 λ = ρ/(1+ρ)）ことを示す．」
- Zhu らの posterior drift は本研究の「切片のずれ＋向きのずれ」（P(Y|X) が変わる）に当たり，UBC は「ずれ無し」に当たる．本研究の暗黙の切片は，Zhu らの「ドメインの情報を足す」の特別な場合（ドメインの指示変数 1 つを足すのと，見分けやすい方向を切片が兼ねるのが同じ効果）と読める（本ノートの読み）

### Gaps
- **7. 注意**：査読を経た版は確認できていない（ワークショップのポスターと arXiv のみ）．論文で引くなら arXiv として引く
- 付録の証明は未読

---

## 論文 4：Koh, Ang, Teo, Liang 2019（NeurIPS）On the Accuracy of Influence Functions for Measuring Group Effects

### Takeaway
L2 罰則つきの凸な損失で，訓練点の群を取り除いた効果を影響関数（1 次）とニュートン法の 1 歩（2 次）で比べ，同じ点の複製を取り除く場合にニュートン法の 1 歩が影響関数の 1/(1 − n φ'' xᵀH⁻¹x) 倍になること（命題 4）を示した．本研究の λ = ρ/(1+ρ) はこの式を「加える向き」に書き直したものである．DA も切片も扱わない．

### Cited Findings
**1. 書誌と読んだ範囲**
- NeurIPS 2019（33rd，Vancouver），arXiv 1905.13289v2（2019-11-21）（◎ PDF の冒頭） — [arXiv 1905.13289](https://arxiv.org/abs/1905.13289)
- 読んだのは要旨，1 節，2 節，3 節（Table 1），4 節（Proposition 1–4，Corollary 1），6 節，付録 D.1–D.2，付録 E.1–E.3（Assumption 1–2，Proposition 4 の再掲と証明の冒頭）．未読：5 節の事例の詳細，付録 B・C

**2. 設定**
- θ̂(1) = argmin Σ ℓ(x_i, y_i; θ) + (λ/2)‖θ‖²（式 (1)），ℓ は凸で 2 回微分可能．部分集合 w を取り除いた θ̂(1 − w) との差 f(θ̂(1 − w)) − f(θ̂(1)) を，影響関数 I_f(w) = ∇f ᵀ H_{λ,1}⁻¹ g_1(w) と，ニュートン法の 1 歩 Δθ^Nt(w) = H_{λ,1}(1 − w)⁻¹ g_1(w)（取り除いた後の重みで再計算したヘッセ行列）で近似する — 2.1・4 節
- 有限標本．漸近は無い．実験は L2 罰則つきロジスティック回帰（多クラスはソフトマックス）で，Diabetes（n = 20,000，d = 127，λ/n = 2.2×10⁻⁴），Enron，Dogfish，MNIST，CDR，MultiNLI．各データで 1,700 の部分集合（0.25%〜25%） — 3 節，Table 1

**3. 示したこと**
- Proposition 1：|Err_Nt-act(w)| ≤ n‖w‖₁² C_f C_H C_ℓ²/(σ_min + λ)³．2 次の損失では 0 — 4.1 節
- Proposition 2：I_f^Nt(w) − I_f(w) = ∇f ᵀ H^{−1/2} D(w) H^{−1/2} g_1(w) ＋ f の曲率の項，D(w) の固有値は [0, σ_max/λ] — 4.2 節
- **Proposition 4（原文）**："Consider a binary classification setting with y ∈ {−1, +1} and a margin-based model with loss ℓ(x, y; θ) = φ(yθᵀx) for some φ : R → R+. Suppose f(θ) = θᵀx_test and that the subset w comprises ‖w‖₁ identical copies of the training point (x_w, y_w). Then under the assumptions of Proposition 1, the Newton approximation I_f^Nt(w) is related to the influence I_f(w) according to I_f^Nt(w) = I_f(w) / (1 − ‖w‖₁ · φ''(y_w θ̂(1)ᵀx_w) · x_wᵀ H_{λ,1}⁻¹ x_w). This implies the Newton approximation I_f^Nt(w) is bounded between I_f(w) and (1 + σ_max/λ) I_f(w)." — 4.4 節
- **向きについての原文**："the scaling factor d(w) = 1/(1 − ‖w‖₁ · φ''_k x_kᵀ H_{λ,1}⁻¹ x_k) in Proposition 4 can be quite large for some subsets w but not for others. In particular, d(w) is large when there are few remaining points in the direction of the removed points." 一般の部分集合では "it is large when we remove a subset w whose Hessian H_1(w) is almost as large as the full Hessian H_{λ,1} in some direction"（付録 D.2） — 4.4 節，付録 D.2
- 仮定（付録 E）：Assumption 1 は損失が凸で 2 回微分可能，λ > 0，ヘッセ行列が C_H-Lipschitz．Assumption 2 は f が 3 回微分可能で導関数が有界．H_{λ,1} は取り除く前の全データのヘッセ行列（取り除く複製を含む）
- 実験：影響関数と実際の効果の Spearman ρ は 0.89〜0.99，影響関数は過小評価する．ニュートン法の 1 歩は "strikingly small" な誤差．加える向きでは "underestimation turns into overestimation" — 3・6 節
- 一段の近似の出典は Pregibon 1981（Annals of Statistics 9(4):705–724）を引く — 1 節，参考文献

**4. 軸との対応**
- A1 ○：L2 罰則つきロジスティック回帰（線形）
- A2 ×：切片と向きの分解は無い（xᵀH⁻¹x はてこ比）
- A3 ×
- A4 ○（ニュートン法の 1 歩の式の形），×（DA・切片・合併の偏りは無い）．取り除く向きで述べる
- A5 ×：群は訓練データの 0.25%〜25%
- A6 ×：分布のずれは無い．群はデータ源・ラベル関数・作業者
- A7 ×．A8 △（CDR のラベル関数はデータプログラミング＝弱い監督だが，弱いラベルで転移はしない）．A9 ×
- A10 ○．A11 ○（予言と実測の相関・過小評価・反例）

**5. 穴**
- 今後の課題："Why do we observe such striking correlation between predicted and actual effects on real data? ... Do these trends hold for non-convex models like neural networks?" — 6 節
- 扱っていない：ドメイン適応，切片，ソースとターゲットの距離

### Inferences
- **6. 差の書き方（案）**：「Koh らは L2 罰則つきロジスティック回帰で，同じ点の複製を取り除いたときのニュートン法の 1 歩が影響関数の 1/(1 − n φ'' xᵀH⁻¹x) 倍になることを示し，取り除いた点の向きに残りの点が少ないと倍率が大きいと述べた．本研究はこの式を加える向きに書き直し，ターゲットをその平均の 1 点で代表させて，合併が切片のずれを直す割合 λ = ρ/(1+ρ) を得る．ρ をターゲットの割合と見分けやすさに分けること，距離が害と得の両方に働くことは Koh らには無い．」
- 加える向きの導出（本ノートの計算，要確認）：ソースだけの解のヘッセ行列を H_s とし，点 x の複製 n 個を足すと，Sherman–Morrison で (H_s + nφ''xxᵀ)⁻¹x = H_s⁻¹x/(1 + nφ''xᵀH_s⁻¹x)．ρ = nφ''xᵀH_s⁻¹x と置けば，加えた点の損失を 0 にする向きの動きのうち 1 歩で進む割合が ρ/(1+ρ) になる．Koh らの H_{λ,1} は「取り除く前」の行列なので分母が 1 − ρ，本研究は「加える前」なので 1 + ρ になる

### Gaps
- **7. 注意**：Pregibon 1981 の本文は未読（前回と同じ）．付録 D.3（テストの損失）は読んでいない

---

## 論文 5：Mansour, Mohri, Ro, Suresh, Wu 2021（AISTATS）A Theory of Multiple-Source Adaptation with Limited Target Labeled Data

### Takeaway
複数ソースの混合比ごとに学習器を作り，少数（m₀ 本）のターゲットのラベルで選ぶと，余分の誤差が Õ(√(ソース数 p/m₀)) で仮説集合の複雑さに依らないこと（上界と下界）を示した．対ごとのずれで重みを決める方法は最適でないことがある．合併の仕組み・切片・ソースの価値の換算は無い．

### Cited Findings
**1. 書誌と読んだ範囲**
- AISTATS 2021，PMLR 130:2332–2340（△．検索結果の抜き書き） — [PMLR v130](https://proceedings.mlr.press/v130/mansour21a.html)．読んだのは arXiv 2007.09762v2（2020-10-29）（◎） — [arXiv 2007.09762](https://arxiv.org/abs/2007.09762)
- 読んだのは 1–7 節（本文全部）と付録 C.1–C.2．未読：付録 A（関連研究），B，D，E（証明），F（追加の実験）

**2. 設定**
- ソース D₁..D_p から m_k 本ずつ（計 m），ターゲット D₀ から m₀ 本（m ≫ m₀）．仮説集合 H は一般（VC 次元 d），多クラス，損失は M で有界．ラベル込みのずれ disc_H(D, D') = max_h |L_D(h) − L_D'(h)|（Mohri & Muñoz Medina 2012）．仮定はターゲットがソースの凸結合 D_λ に近いこと．歪み s(λ‖m) = Σ λ_k²/m_k — 2・3 節
- 固定の少数ショットではなく，m₀ が小さい体制一般．実験は m₀ = 50〜400（人工），1,280（実データ）

**3. 示したこと**
- Proposition 1（λ 既知）：E(λ) ≤ 4M√(s(λ‖m)/m · (d log(em/d) + log(1/δ))) + 2 disc_H(D₀, D_λ) — 3 節
- Theorem 2（LMSA，ε 被覆 Λ）：L_D₀(h_m) − min_h L_D₀(h) ≤ E(λ*) + 2Mε + 2M√(p log(p/(εδ))/m₀)．"the dominating term of the learning bounds for our algorithms are in Õ(√(p/m0)) ... and do not depend on the complexity of the hypothesis set"．ターゲットだけの O(√(d/m₀)) より良いには p < d が要る — 4・4.1 節
- Theorem 7（下界）：無限のソースと m₀ 本のターゲットでも，ある問題では E[L_D₀(h_A)] ≥ min_h L_D₀(h) + c√(p/m₀) — 4.4 節
- 5 節："this approach can be sub-optimal in various scenarios"．Example 1：D₀ = (D₁ + D₂)/2 で 3 つの対ごとのずれが等しいと，対ごとの重みは λ = (1/3, 1/3, 1/3) になり最適 (0.5, 0.5, 0) から遠い．凸結合とのずれを推定する方法は "the generalization bound varies as Õ(√(d/m0)), which can again be weak or uninformative for small values of m0"（付録 C.2） — 5 節，付録 C.2
- 実験：数字画像 4 ドメイン（各ソース 20,000，ターゲット 1,280，CNN）．ターゲット MNIST-M で sources+target 74.8，Konstantinov & Lampert 84.6，LMSA 87.6，LMSA-MIN-MAX 89.5．人工の回帰（p = 4，d = 100，各ソース 10,000）で m₀ = 50 のときターゲットのみ 13.16，LMSA-MIN-MAX 5.15，λ* 既知 4.47（×1000） — 6 節，Table 1・2

**4. 軸との対応**
- A1 ×：一般の仮説集合．実験は CNN と線形回帰
- A2 ×．A3 △：ソースの価値は「m₀ の項の d が p に置き換わる」形で現れ，本数への換算ではない
- A4 △：混合比 λ で重みづけた合併を扱うが，切片も仕組みも無い
- A5 △：m₀ が小さい体制．クラスあたりの K ではない
- A6：ずれはラベル込みのずれ．ソースは p 個（○）
- A7 ×．A8 ×．A9 ×
- A10 ○（数字画像 4 ドメイン）．A11 ×

**5. 穴**
- 結論："We hope that our analysis will serve as a tool for further theoretical studies of this problem and other related adaptation problems and algorithms." 限界の明記は少ない — 7 節
- 扱っていない：合併の仕組み，切片，線形の頭，ソース 1 つの価値の換算

### Inferences
- **6. 差の書き方（案）**：「Mansour らは，少数のターゲットのラベルで複数ソースの混合比を選ぶと，余分の誤差が √(ソース数/m₀) で仮説集合の複雑さに依らないことを示し，対ごとのずれから重みを決める方法が最適でない例を与えた．本研究の換算した本数は選び方の保証ではなく，ソース 1 つの価値をターゲットのラベルの本数で測る量で，向きの余弦から閉じた形で出す．」
- Mansour らの「対ごとのずれで重みを決めるのは最適でない」は，本研究が e_geom（特徴空間の量）からソースの価値を予言することへの注意になる．本研究は予言を地図（実測）と照合しているので，この注意には実測で答えられる

### Gaps
- **7. 注意**：AISTATS 版（PMLR 130:2332–2340）の頁数は検索結果の抜き書きで，本文は arXiv v2 を読んだ．付録 F の追加実験は未読

---

## 論文 6：Hanneke & Kpotufe 2019（NeurIPS）On the Value of Target Data in Transfer Learning

### Takeaway
ソース n_P 本とターゲット n_Q 本の両方で最小最大の率を初めて与え，率は「ソースかターゲットの情報の少ない方を無視する」オラクルで達成できること，ただしどちらを無視すべきかは事前に分からないことを示した．ソースの実効標本数 ñ_P = n_P^{(2−β_Q)/((2−β_P)ρ)} を率の指数の形で定義する．前提は最良の分類器がソースとターゲットで同じ（RCS）で，線形模型も実験も無い．

### Cited Findings
**1. 書誌と読んだ範囲**
- NeurIPS 2019，pp. 9871–9881（△．検索結果の抜き書き） — [NeurIPS 2019 予稿](https://proceedings.neurips.cc/paper_files/paper/2019/file/b91f4f4d36fa98a94ac5584af95594a0-Paper.pdf)．読んだのは arXiv 2002.04747v1（2020-02-12，付録つき）（◎） — [arXiv 2002.04747](https://arxiv.org/abs/2002.04747)
- 読んだのは要旨，1–6 節（Definition 1–5，Theorem 1–4，Proposition 1–3，Remark 1–4），付録 A.1–A.2（Theorem 5–6）．未読：付録 B–E（証明）

**2. 設定**
- 2 値分類，VC 次元 d_H の固定の仮説集合．(NC) Bernstein の雑音条件（β_P, β_Q），(RCS) E_Q(h*_P) = E_Q(h*_Q) = 0（最良の分類器が同じ．P_{Y|X} ≠ Q_{Y|X} は許す）．転移指数 ρ：C_ρ · E_P(h) ≥ E_Q(h)^ρ（式 (2)）．周辺の転移指数 γ：C_γ · P_X(h ≠ h*_P) ≥ Q_X(h ≠ h*_P)^γ（式 (3)）．ρ, γ は非対称 — 2・3 節
- 有限標本の最小最大の率．線形模型は無い．ソースは 1 つ（付録 A.2 で複数から選ぶ）

**3. 示したこと**
- Theorem 1（下界）：任意の学習器に対して E_Q(ĥ) ≥ c · min{(d_H/n_P)^{1/((2−β_P)ρ)}, (d_H/n_Q)^{1/(2−β_Q)}} となる (P, Q) がある — 4 節
- Theorem 3（上界）：Algorithm 1（S_Q で最適に近い制約の下で R̂_{S_P} を最小化）で E_Q(ĥ) ≤ Õ(min{(d_H/n_P)^{1/((2−β_P)ρ)}, (d_H/n_Q)^{1/(2−β_Q)}})．(RCS) 無しでも成り立つ（Remark 4） — 5 節
- **「弱い方を無視する」の原文**："the best possible rate (matching upper and lower-bounds) in terms of ρ and both sample sizes nP, nQ is - up to constants - achievable by an oracle which simply ignores the least informative of the source or target datasets. In other words, if ĥP and ĥQ denote the ERM on data from P, resp. from Q, one of the two achieves the optimal Q rate over any classifier having access to both P and Q datasets. However, which of ĥP or ĥQ is optimal is not easily decided without prior knowledge: for instance, cross-validating on a holdout target-sample would naively result in a rate of n_Q^{−1/2} which can be far from optimal given large nP." — 1 節
- **実効標本数（Remark 2 の原文）**："we might view (ignoring dH) ñP = nP^{(2−βQ)/(2−βP)ρ} as the effective sample size contributed by P. In fact, the above minimax rate is of order (ñP + nQ)^{−1/(2−βQ)} ... the effective source sample size ñP is smallest for large ρ, but also depends on (2 − βQ)/(2 − βP), i.e., on whether P is noisier than Q." — 5 節
- Proposition 2（RCS の外）：E_Q(ĥ) ≤ min{E_Q(h*_P) + C A_{n_P}^{1/((2−β_P)ρ')}, C A_{n_Q}^{1/(2−β_Q)}}．Proposition 3：役割を入れ替えた Algorithm 1' は "more akin to so-called hypothesis transfer" — 5 節
- Theorem 4：費用の異なる P と Q の標本を，分布のパラメータを知らずに最小最大の費用で集められる．Theorem 5–6（付録 A）：ラベル無しの Q のデータでソースの重み直しと複数ソースからの選択ができる
- 実験は無い："the procedures described in this work remain of a theoretical nature, but yield new insights" — 1 節

**4. 軸との対応**
- A1 ×：一般の VC 族．A2 ×
- A3 ○：ソースの実効標本数 ñ_P（率の指数の形，定数と d_H を無視）
- A4 ×：合併は扱わない（最適な手続きは制約つき ERM）
- A5 △：n_Q は任意（小さくてよい）が率の議論
- A6：RCS（最良の分類器が同じ）．雑音の違い．複数ソースは付録 A.2 で選択
- A7 ×．A8 ×．A9 △（Algorithm 1' は仮説転移に近いが制約の形で，L2 罰則ではない）
- A10 ×．A11 ×

**5. 穴**
- 自ら書く：(RCS) は "it is easily removed, but at the additive (necessary) cost of EQ(h*P)"．最良の分類器が違う場合（ラベルの付け方のずれ）は加法的な項で扱うだけ — 2 節
- 扱っていない：線形の構造，定数込みの換算，合併，実験

### Inferences
- **6. 差の書き方（案）**：「Hanneke & Kpotufe は，ソースの実効標本数を転移指数と雑音条件の指数で定義し，最小最大の率がソースかターゲットの弱い方を無視するのと同じ形になることを示した．本研究の換算した本数 n = q e²/(1 − e²) は同じ役目を，線形の頭の向きの余弦 e と実効次元 q から定数込みで与え，感情コーパスの実測の学習曲線で照合する．」
- 本研究の式は Hanneke らの ñ_P の書き直しではない．対応は「率が (ñ_P + n_Q) の関数になる」という足し算の形と，本研究の「向きの誤差 q/(n + K)」の足し算の形が同じ考え方であること（本ノートの読み）

### Gaps
- **7. 注意**：NeurIPS 版（11 頁）と arXiv v1 の差は確かめていない．付録 B–E の証明は未読．関連研究の節に既にある記述（`docs/related-work.md` 2.8 節）と矛盾は無い

---

## 論文 7：Chen & Bühlmann 2021（JMLR 22(261)）Domain Adaptation under Structural Causal Models

### Takeaway
反因果の線形構造方程式で X に平均のずれ（雑音への介入）があるとき，ソースだけの最小 2 乗の目標誤差は「平均のずれを判別の向きに射影した量」の 2 乗だけ増え，ラベル無しで平均を合わせる DIP はその項を消す代わりに 1 方向を失うこと（定理 1）を母集団の式で示した．Y への介入（ラベルの比率のずれ）があると DIP は壊れる．ターゲットのラベルは手法の選択にだけ使う（20〜60 本）．

### Cited Findings
**1. 書誌と読んだ範囲**
- JMLR 22(261):1–80, 2021（JMLR の id は 20-1227）（○．JMLR の頁の検索結果） — [JMLR v22 20-1227](https://jmlr.org/papers/v22/20-1227.html)．読んだのは arXiv 2010.15764v2（2021-11-24）（◎） — [arXiv 2010.15764](https://arxiv.org/abs/2010.15764)
- 読んだのは要旨，1 節，3.1–3.4 節（設定，OLSTar・OLSSrc・DIP の定義），4.1 節（Assumption 1，Theorem 1，Corollary 2–4），5.4 節（Amazon），6 節．未読：2 節，3.5 節，4.2–4.3 節（CIP・CIRM，Theorem 5），5.1–5.3 節，付録

**2. 設定**
- 線形 SCM：(X; Y) = [[B, b]; [ωᵀ, 0]](X; Y) + g(a, ε)．ソースとターゲットの違いは介入 a^(m) と ã だけ．主結果は平均のずれ g(a, ε) = a + ε．反因果（ω = 0，X は Y の子孫）．回帰（2 乗損失）で，分類は実験のみ．母集団（無限標本）の推定量．ターゲットは共変量だけ観測．ソースは M 個 — 3.2 節
- DIP-mean：ソースで最小 2 乗，制約は E_src[Xᵀβ] = E_tgt[Xᵀβ]（式 (10)）．OLSTar は切片 β₀ つきのターゲットの最小 2 乗（式 (5)）

**3. 示したこと**
- **Theorem 1（原文の式）**：Assumption 1 と Y への介入無し（a_Y^(1) = ã_Y = 0）の下で，R̃(f_OLSTar) = σ²/(1 + σ² bᵀΣ⁻¹b)（式 (16)），R̃(f_OLSSrc^(1)) = σ²/(1 + σ² bᵀΣ⁻¹b) + σ²(bᵀΣ⁻¹(a_X^(1) − ã_X))²/(1 + σ² bᵀΣ⁻¹b)²（式 (17)），R̃(f_DIP^(1)) = R̃(f_DIPOracle^(1)) = σ²/(1 + σ² bᵀΣ^{−1/2} G_DIP Σ^{−1/2} b)（式 (18)）．G_DIP は u^(1) = (a_X^(1) − ã_X)/‖·‖ に直交する階数 d − 1 の射影．"The target population risk of OLSSrc(1) depends on the magnitude of the difference in intervention ‖a_X^(1) − ã_X‖₂, while the target risk of DIP(1) is independent of that magnitude." — 4.1.1 節
- Corollary 2（Σ = (σ²/ρ)I）：R̃(f_DIP) = σ²/(1 + ρ‖b‖² − ρ(u^(1)ᵀb)²)．"the gap between the target population risks of DIP(1) and OLSTar is smaller when the direction of the difference in intervention u(1) becomes less aligned with the vector b"．ずれがガウスのランダムなら "The gap only comes from an order 1/d term" — 4.1.1 節
- Corollary 4：Y への介入があると DIP の誤差に (a_Y^(1) − ã_Y)² の項が加わる — 4.1.2 節末
- Corollary 3：DIP のソースの誤差は目標誤差に等しいので，複数ソースから最良を選べる — 4.1.2 節
- 5.4 節（Amazon レビュー，15 カテゴリ，各 10,000 本，TF-IDF d = 482，L2 罰則つき線形）："the advanced DA methods do not outperform SrcPool except for target environment number 4"．20・40・60 本のターゲットのラベルで手法を選ぶと "the best method based on the small portion of labeled target data is always not worse than SrcPool and can sometimes outperform SrcPool"
- 6 節："It remains an open question what is the minimal amount of labeled target data points in order to guarantee the best DA method selection."

**4. 軸との対応**
- A1 ○：切片つきの線形最小 2 乗（回帰）
- A2 △：OLSSrc の余分の誤差は「平均のずれを b（判別の向き）に射影した量」で，本研究の切片の誤差に当たるが，切片と向きの名前では分けていない．DIP の損は「1 方向を失う」
- A3 ×．A4 ×：ラベル付きターゲットとの合併は無い（SrcPool はソースどうしの合併）
- A5 △：20〜60 本のターゲットのラベルを手法の選択にだけ使う（実験）
- A6 ○：X の平均のずれ，Y への介入，因果・反因果・混合．ソースは M 個
- A7 ×．A8 ×．A9 ×
- A10 ○：模擬，MNIST の変種（未読），Amazon．A11 ○：仮定の必要性を模擬で確かめる

**5. 穴**
- 自ら書く："The theoretical extension to nonlinear SCMs is a challenging future direction."．"our theory does not apply when the matrix B is no longer invariant across source and target environments" — 3.2・6 節
- 扱っていない：有限標本，分類の理論，ターゲットのラベルを学習に使うこと，合併の切片

### Inferences
- **6. 差の書き方（案）**：「Chen & Bühlmann の定理 1 は，反因果の線形模型で X の平均のずれが判別の向きに射影した分だけソースのみの誤差を増やし，ラベル無しで平均を合わせる操作がそれを消すことを母集団の式で示す．本研究の中心合わせはこの操作と同じで，本研究が足すのは，少数ラベルが直す量と中心化が直す量を同じ分割で比べたことと，中心化しない合併が切片を自力でどれだけ直すか（λ）である．」
- 定理 1 の (bᵀΣ⁻¹Δa)² は本研究の「距離の害の働き」（位置のずれ）そのもので，Δa が b に直交すれば害は無い．Chen らに「得の働き」（見分けやすさが合併で切片を直す）は無い．これは本研究の差の根拠になる（本ノートの読み）

### Gaps
- **7. 注意**：4.2–4.3 節（CIP・CIRM・Theorem 5）と 5.1–5.3 節（MNIST の分類の実験）は未読．JMLR 版の頁は検索結果で確かめただけで，本文は arXiv v2

---

## 論文 8：Wiesenfarth & Calderazzo 2020（Biometrics 76(1)）Quantification of prior impact in terms of effective current sample size

### Takeaway
事前分布（過去のデータ）の効果を「今回のデータ何本ぶんか（ECSS）」で測る．ECSS は，対象の事前分布のもとで k − m 本を見た事後分布が，基準の（弱い）事前分布のもとで k 本を見た事後分布と，平均 2 乗誤差などの尺度で同じになる m である．事前分布と今回のデータが食い違うと情報が失われることを捉え，m は負にもなる．本文は入手できず，要旨と R パッケージの文書で確かめた．

### Cited Findings
**1. 書誌と読んだ範囲**
- Biometrics 76(1):326–336, 2020（電子版 2019-09-13），DOI 10.1111/biom.13124，PMID 31364156（◎ 書誌と要旨．Europe PMC の API） — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1111/biom.13124&format=json&resultType=core)
- **本文は未読．**Wiley の全文はこのセッションでは取得できなかった（Europe PMC は Unpaywall 経由の公開 PDF として https://onlinelibrary.wiley.com/doi/pdfdirect/10.1111/biom.13124 を載せている．ブラウザで開けば読める見込み）．arXiv 版は無い（検索で見つからず）
- 定義は著者の R パッケージ ESS（GitHub wiesenfa/ESS）の ecss 関数の文書で確かめた（○） — [rdrr.io の ecss](https://rdrr.io/github/wiesenfa/ESS/man/ecss.html)

**2. 設定**
- ベイズの臨床試験の設計．過去のデータから作った事前分布（頑健な混合事前分布，power prior，commensurate prior）と今回のデータ模型．尺度 D は既定で事後平均の平均 2 乗誤差 D^{θ₀}_MSE(π(θ|y_{1:k})) = E_{y|θ₀}[E_π(θ|y_{1:k}) − θ₀]²．θ₀ は既知と仮定する（事前の設計）か，基準の事前分布のもとでの事後平均で置き換える（感度分析・適応的な設計） — ecss の文書

**3. 示したこと**
- 要旨："the traditional approach identifies the ESS with such a historical sample size. However, this measure is independent of newly observed data, and thus would not capture an actual 'loss of information' induced by the prior in case of prior-data conflict. We build on a recent work to relate prior impact to the number of (virtual) samples from the current data model and introduce the effective current sample size (ECSS) of a prior" — Europe PMC の要旨
- 定義（パッケージの文書）："The ECSS at target sample size k (specified by n.target) is defined as the sample size m which minimizes" |D(prior, k − m) − D(baseline, k)|，つまり ECSS = argmin_m |D^{θ₀}(π(θ|y_{1:(k−m)})) − D^{θ₀}(π_base(θ|y_{1:k}))|．関数は "quantifies the number of current samples to be added or subtracted to the likelihood"（負の値を許す） — rdrr.io の ecss
- 応用：中間解析で実効標本数に応じて募集を調整する適応的な設計．"We argue that the ECSS is the appropriate measure in this case, as the aim is to save current (as opposed to historical) patients from recruitment." 混合事前分布の ESS の合意の無さを解消しうる — 要旨

**4. 軸との対応**
- A1 ×：分類器ではなくパラメトリックなベイズ模型
- A2 ×．A3 ○：事前分布（ソース）の価値を今回（ターゲット）の本数で測る．食い違いで価値が減り，負にもなる
- A4 ×．A5 △：臨床試験の小さな k
- A6 △：事前分布とデータの食い違い（パラメータのずれ）．過去の試験は通常 1 つ（混合事前分布で複数も）
- A7 ×．A8 ×．A9 △：commensurate prior と power prior は過去の推定値に寄せる事前分布で，出発点への L2 罰則に近い
- A10 ×（臨床の例示）．A11 ×

**5. 穴**
- 分類器が無い．尺度が真値 θ₀ に依存する（置き換えが要る）．本文未読のため，頑健な混合事前分布での具体的な式と限界の記述は確かめていない

### Inferences
- **6. 差の書き方（案）**：「ソースや事前分布の価値を『今回のデータ何本ぶんか』で測る尺度は，ベイズの臨床試験の実効標本数（Morita ほか 2008；Wiesenfarth & Calderazzo 2020）にある．Wiesenfarth らの ECSS は事前分布と今回のデータが食い違うと価値が減り負にもなる．本研究の換算した本数は同じ単位を，線形の頭の向きの余弦から閉じた式で与え，食い違いの害（切片のずれ）は別に扱う．」
- 本研究の式は ECSS の書き直しではない．対応は定義の形（「同じ誤差に届く本数の差」）で，本研究は誤差の式 q/(n + K) を陽に持つので argmin を解く必要が無い（本ノートの読み）

### Gaps
- **7. 注意**：本文未読．Morita ほか 2008 の定義も未読（前回と同じ）．論文で引くなら要旨の範囲の主張に限る

---

## 論文 9：Le Scao & Rush 2021（NAACL）How Many Data Points is a Prompt Worth?

### Takeaway
プロンプトつきの微調整と頭つきの微調整の学習曲線を同じ条件で引き，両曲線が重なる正解率の帯で曲線の間の面積を帯の高さで割って「プロンプトは訓練例何本ぶんか（average data advantage）」を出した．MNLI で約 3,500 本，SuperGLUE の多くで数百本，WiC は負．理論は無く，仕組みの解明は今後の課題と書く．

### Cited Findings
**1. 書誌と読んだ範囲**
- NAACL-HLT 2021，pp. 2627–2636（◎ PDF の脚注） — [ACL Anthology 2021.naacl-main.208](https://aclanthology.org/2021.naacl-main.208)
- 全文（1–9 節，付録 A–C）を読んだ

**2. 設定**
- roberta-large（355M）．頭つき（分類ヘッド）とプロンプトつき（PET の pattern と verbalizer）の微調整．SuperGLUE 7 課題と MNLI．訓練本数は 10 本から指数的に全件まで（例：MultiRC は 10, 15, 20, 32, 50, 70, 100, 150, 200, 320, 500, 750, 919）．各点 4 回，計 1,892 回．報告は "the best performance that has been achieved at that amount of data or lower"（累積最大） — 4 節
- ずれは無い．ソースの数：なし（プロンプトが事前知識に当たる）

**3. 示したこと**
- 測り方の原文："we first isolate the y-axis band of the lowest- and highest-accuracy where the two curves match in accuracy. The horizontal line at these points represents the advantage of prompting. We then take the integral in this region, i.e. area between the linearly-interpolated curves, divided by the height of the band. The area has the dimension of a quantity of data points times the metric unit, so dividing by the performance range yields a # of data points advantage." 信頼幅は 4 回ずつの実行から 1 つを除く 16 通りのブートストラップ — 5 節
- 結果（Table 1，P vs H）：MNLI 3506 ± 536，BoolQ 752 ± 46，CB 90 ± 2，COPA 288 ± 242，MultiRC 384 ± 378，RTE 282 ± 34，WiC −424 ± 74，WSC 281 ± 137．"prompting is often worth 100s of data points on average across classification tasks" — 要旨，Table 1
- 無意味な verbalizer の対照（N）：MNLI で P vs N 150，N vs H 3355．"for small data tasks such as CB, the null verbalizer removes much of the benefits of prompting. However, with more training data, the model seems to adapt the verbalizer" — 6 節
- 報告の仕方（最大・平均）と指標の線形性に敏感："This choice does not have a consistent effect for or against prompting" — 6 節，付録 B

**4. 軸との対応**
- A1 ×．A2 ×
- A3 ○：経験的な換算（学習曲線の水平方向の差）
- A4 ×．A5 ○：10 本から
- A6 ×．A7 ×．A8 ×．A9 ×
- A10 ○．A11 ×（理論無し）

**5. 穴**
- 今後の課題："In future work, we hope to study the mechanism and training dynamics of the prompting benefits." — 7 節
- 曲線が漸近で一致する仮定："We assume asymptotically the two curves would match, but are limited by data."（脚注 3）

### Inferences
- **6. 差の書き方（案）**：「『何本ぶんか』という単位は Le Scao & Rush（プロンプトの価値を学習曲線の水平方向の差で測る）と同じである．本研究は曲線の面積ではなく，向きの余弦 e と実効次元 q から閉じた式 n = q e²/(1 − e²) で換算し，その予言を実測の学習曲線（地図）と照合する．」

### Gaps
- **7. 注意**：なし（全文を読んだ）

---

## 余力の 3 本：Ben-David 2010，Blitzer 2007，Sarao Mannelli 2025

### Takeaway
Ben-David 2010 の α* は閉じた形（式 (2)）で，ターゲットが d/A² 本以上ならソース不要という相転移を持つ．Blitzer 2007 の相転移は m_T = C/ζ² で，C = 1600・ζ = 0.715 で 3,130 本．Sarao Mannelli 2025 は集団ごとの先生（ラベル規則と切片）を持つガウス混合で，切片 1 つの学習器の厳密解（レプリカ法）を与えるが，切片が直る割合の閉じた式は無い．

### Cited Findings
- **Ben-David ほか 2010（Theorem 3 と 6 節）**（◎ 手元の PDF）：上界を f(α) = 2B(α²/β + (1−α)²/(1−β)) + 2(1−α)A と書き，D = √d/A として α*(m_T, m_S; D) = 1（m_T ≥ D²），min{1, ν}（m_T ≤ D²），ν = m_T/(m_T + m_S) · (1 + m_S/(D²(m_S + m_T) − m_S m_T))（式 (2)）．"if there are enough target data (specifically, if mT ≥ D² = d/A²) then no source data are needed, and in fact using any source data will yield suboptimal performance"．A = 0 なら α* = β（全例を等重み） — `A theory of learning from different domains.pdf` 6 節
- **Blitzer ほか 2007（4 節）**（◎）：近似した上界 f(α) = √(C/m (α²/β + (1−α)²/(1−β))) + (1−α)ζ(U_S, U_T)（式 (1)）．"The phase transition occurs when mT = C/ζ(US, UT)²"．C = 1600，ζ = 0.715 で 3,130 本 — [NIPS 2007 予稿](https://papers.nips.cc/paper_files/paper/2007/file/42e77b63637ab381e8be5f8318cc28a2-Paper.pdf)
- **Sarao Mannelli, Gerace, Rostamzadeh, Saglietti（arXiv 2205.15935v4，2024-11-29）Bias-inducing geometries: an exactly solvable data model with fairness implications**（◎ 1–2 節と 3 節の冒頭）：Teacher-Mixture 模型．x ~ N(±v/√d, Δ± I) を確率 ρ, 1−ρ で，ラベルは集団ごとの先生 y = sign(W_T^c · x/√d + b_T^c)（先生の切片 b_T^± で集団ごとのラベルの比率を制御，先生どうしの重なり q_T）．学習器は W と**切片 1 つ** b の線形で，凸な損失＋λ‖W‖²/2．n, d → ∞，α = n/d のレプリカ解析で，記述子 {Q, m, R±, δq, b} の不動点．学習器の切片は "The student bias term b implicitly solves the equation ∂_b e(Θ; Δ) = 0"．結果：相対的な代表性・分散・ラベルの比率・規則のずれで偏りが出る．"a positive transfer effect [29] can yet be traced between the two groups when the rules are sufficiently similar"（図 5）．緩和策として集団ごとの 2 つの網を γ‖W⁺ − W⁻‖² で結ぶ（式 (8)，集団ごとの切片 b± を持つ） — [arXiv 2205.15935](https://arxiv.org/abs/2205.15935)
- Sarao Mannelli らの掲載誌：Bologna 大学の研究者の頁が Physical Review E（2025）と載せる（△）．**巻号（112(2):025304）は確かめられなかった**（APS の頁は取得できず）．arXiv の頁に journal-ref は無い — [Bologna CRIS](https://cris.unibo.it/cris/rp/rp293067?type=all)

### Inferences
- Ben-David の α* の形（m_T + m_S の比に，距離 D で決まる補正をかける）は，本研究の ρ（ターゲットの割合 × (1 + 見分けやすさ)）と同じ 2 つの材料（比重と距離）を持つが，距離 A は上界では害の働きしか持たない
- Sarao Mannelli らの模型は「集団ごとの先生の切片 b_T^± と学習器の切片 1 つ」なので，本研究の「合併が暗黙にターゲット専用の切片を学ぶ」を厳密に解ける土台になる．ただし彼らは切片が直る割合も，集団の見分けやすさ（v の大きさ）の得の働きも式にしていない．論文 2 の材料として，彼らの不動点の式に本研究の λ を当てはめて確かめる価値がある

### Gaps
- Sarao Mannelli らの付録 B（レプリカの導出）と図 5 の数値は読んでいない．Phys. Rev. E の巻号は要確認

---

## 9 本 × 軸 A1〜A11 の表

### Takeaway
A3（換算）を持つのは Hanneke（率の指数），Wiesenfarth（ベイズの ECSS），Le Scao（経験的）の 3 本で，いずれも線形の頭の幾何（余弦と実効次元）からは出していない．A4（合併の切片）を閉じた形で持つ論文は無い：Tiomoko は中心化して切片を課題別に持ち，Yang は切片の無い回帰，Koh は取り除く向きの式で DA が無い．A5（クラスあたり数本）を理論で扱う論文は無い．

### Cited Findings
記号：○ 扱う，△ 一部または近い形，× 扱わない．各欄の根拠は上の各論文の 4 項．

| 軸 | 1 Tiomoko 2021 | 2 Yang 2025 | 3 Zhu 2025 | 4 Koh 2019 | 5 Mansour 2021 | 6 Hanneke 2019 | 7 Chen 2021 | 8 Wiesenfarth 2020 | 9 Le Scao 2021 |
|---|---|---|---|---|---|---|---|---|---|
| A1 線形か | ○ LS-SVM | ○ 合併 OLS（回帰） | △ ベイズ誤差の理論．実験に線形プローブ | ○ L2 ロジスティック | × 一般の H（実験 CNN） | × VC 族 | ○ 切片つき OLS（回帰） | × ベイズ模型 | × roberta-large |
| A2 切片と向き | △ 閾値 ζ と分散 σ² | × 切片無し | × | × てこ比のみ | × | × | △ ずれの b への射影 vs 1 方向の損 | × | × |
| A3 換算 | × | △ 相転移の n₀ のみ | × | × | △ d が p に置き換わる | ○ ñ_P（率の指数） | × | ○ ECSS（負も可） | ○ 学習曲線の差 |
| A4 合併の切片・暗黙の切片・近似の形 | △ 厳密な極限．中心化＋課題別切片 | △ 厳密な極限．切片無し | △ 明示的なドメイン特徴 vs 合併 | ○ 1 歩の式（取り除く向き）．DA 無し | △ 混合比の合併．仕組み無し | × | × ターゲットのラベルの合併無し | × | × |
| A5 固定の少数ショット | △ Remark 7，MNIST 10/クラス | × n₂ ≥ (1+τ)p | × DG | × 群 0.25–25% | △ m₀ 小 | △ 率 | △ 20–60 本を選択に | △ 小さな k | ○ 10 本から |
| A6 ずれの型・ソース数 | 平均の向き・比率．k 課題 | 共変量・模型．2 と多課題 | posterior drift．多数 | 無し | ラベル込みのずれ．p 個 | RCS・雑音．1（A.2 で複数） | X の平均・Y 介入・因果．M 個 | 事前とデータの食い違い．1 | 無し |
| A7 マルチモーダル | × | × | △ 文＋属性の結合 | × | × | × | × | × | × |
| A8 弱いラベル | × | × | × | △ ラベル関数の影響 | × | × | × | × | × |
| A9 出発点への L2 | △ W₀ + V_i | △ SPS の λ‖z‖² | × | × | × | △ 制約つき ERM | × | △ commensurate/power prior | × |
| A10 実データ | ○ MNIST，Office+Caltech | × 模擬のみ（JMLR 版） | ○ 3 データ | ○ 6 データ | ○ 数字 4 ドメイン | × | ○ 模擬＋Amazon | × | ○ 8 課題 |
| A11 予言と実測・非単調 | ○ 分布の一致 | ○ 模擬で一致．n₁ で得→損 | × | ○ 相関・過小評価 | × | × | ○ 仮定の必要性 | × | × |

### Inferences
- 主張 C の置き場所：「合併型の線形分類器の偏りを厳密に解いた」のは Tiomoko（LS-SVM，中心化あり），「合併の推定量の偏りと分散を閉じた形にした」のは Yang（回帰，切片無し），「ニュートン法の 1 歩の倍率」は Koh（DA 無し）．3 本のどれも，中心化しない合併のロジスティック回帰で切片が直る割合を与えていない．本研究の式は Koh らの命題 4 の書き直しで，仕組みの解釈（暗黙のターゲット専用の切片，距離の 2 つの働き）は 9 本のどれにも無い
- 主張 B の置き場所：「ターゲットの本数で測る」考え方は Hanneke（率），Wiesenfarth（ベイズ），Le Scao（経験）にある．本研究の差は，線形の頭の向きの余弦という測れる量から閉じた式で出し，実測と照合する点と，その大きさ（1 ソース ≈ クラスあたり 10 本）を感情コーパスで出した点にある

### Gaps
- 表の △ の判定は本ノートの読みを含む．特に Tiomoko の A5（Remark 7 を少数ショットの扱いと読んだ）と Chen の A2（射影の項を切片の誤差と読んだ）は，論文で引用するときは原文の語で書き直す

---

## この群の先行研究に空いている穴

### Takeaway
穴は 5 つある：(1) 中心化しない単純な合併のロジスティック回帰で切片が直る割合の閉じた式，(2) ソースとターゲットの見分けやすさが合併の得に働くこと（先行研究では害か，明示的に足す情報としてだけ扱う），(3) クラスあたり数本（4K ≪ p）の有限標本と比例の極限の間の体制，(4) 線形の頭の幾何（余弦と実効次元）からソースの価値をターゲットの本数に換算する閉じた式，(5) マルチモーダルの感情コーパスの多数の組での予言と実測の照合．

### Cited Findings
- (1) 合併の切片：Tiomoko は課題ごとに平均を引いてから学ぶ（"we choose here to center the data task-wise"）ので合併が作る切片のずれは無い — arXiv 2009.01591 2.1 節．Yang は切片の無い回帰で，分類は arXiv v1 の今後の課題 — arXiv 2010.11750v1 5 節．Koh は取り除く向きで DA を扱わない — arXiv 1905.13289 4.4 節
- (2) 見分けやすさの働き：Chen & Bühlmann では平均のずれは b に沿う分だけ害で，直交すれば無害（Corollary 2）．Zhu ではドメインの情報を明示的に足した場合の得（Theorem 1–2）．Tiomoko では関係の無い課題のスコアを 0 にするのが最適（3.2 節）．Ben-David と Blitzer の上界では距離は害だけ — 各論文の 3 項
- (3) 体制：Yang は n₂ ≥ (1+τ)p（式 (2.5)），Tiomoko は n/p → c₀ の比例の極限，Hanneke と Mansour は率．クラスあたり数本の閉じた式を与える論文は無い
- (4) 換算：Hanneke の ñ_P は "ignoring dH" で定数を無視した率の指数，Wiesenfarth の ECSS はベイズの事後の尺度の argmin，Le Scao は学習曲線の面積．線形の頭の余弦から出す式は無い
- (5) 実データ：9 本の実データは MNIST・Office（Tiomoko），数字 4 ドメイン（Mansour），Amazon（Chen），注釈者・WILDS・PACS（Zhu），SuperGLUE（Le Scao）．感情・マルチモーダル・多数のコーパスの組は無い（A7 は全て ×）

### Inferences
- 論文 1 の関連研究には，主張 C について「合併の偏りの厳密解（Tiomoko 2021；Yang 2025）と一段の近似の式（Pregibon 1981；Koh 2019）は既にあり，本研究はそれらを中心化しない合併のロジスティック回帰に当てはめて，切片の直る割合を『ターゲットの比重』と『見分けやすさ』に分けた」と書く．「合併の偏りを初めて示した」「ニュートン法の式は新しい」とは書かない
- 主張 B については「ソースを今回のデータ何本ぶんかで測る尺度は既にある（Hanneke 2019；Wiesenfarth & Calderazzo 2020；Le Scao & Rush 2021）．本研究は線形の頭の向きの余弦から閉じた式で換算し，感情コーパスで 1 ソースがクラスあたり約 10 本に当たることを実測と照合した」と書く
- 見分けやすさが得に働くという主張は，先行研究では「害」（Ben-David，Blitzer，Chen）か「明示的に足す情報」（Zhu，Daumé 2007）としてだけ扱われている．この対比は論文 1 の新しさの中心に置ける．ただし Sarao Mannelli らの模型で厳密に確かめられるので，論文で「厳密な解析」と書かず「一段の近似と実測」と書く

### Gaps
- 統計物理の多集団の解析（Sarao Mannelli 以外）と，Mignacco ほか 2020・Loureiro ほか 2021 のガウス混合の厳密な漸近に切片の扱いがあるかは，本ノートでは調べていない（前回のノートと同じ）
- Tiomoko の ICLR 版と Wiesenfarth の本文は未読のまま．ICLR 版の節番号と，ECSS の頑健な混合事前分布での式は，論文で引く前に確かめる
