# 曲率で分けた教師の価値の先例（理論の側）

調べた日：2026-10-28．対象は，次の 3 つの主張に近い先行理論である．

- 段 1（ターゲットのラベルを使わない）：θ_Q = argmin［教師の疑似ラベルの交差エントロピー（総重み m）］＋‖θ − a0‖²/(2C0)．
- 段 2（クラスあたり K 本，n = 4K）：θ_B = argmin［ラベル付きの交差エントロピー］＋‖θ − θ_Q‖²/(2C′)．対照 θ_s は同じ段 2 を a0 から始めたもの．
- 実測（相対誤差 1〜2%）：θ_B − θ_s ≈ (P + H_K)⁻¹P(θ_Q − a0)，P = I/C′，H_K は θ_s でのラベル付き損失のヘッセ行列（損失の 2 階微分の行列．曲率）．一般化固有方向（H_K v = λ P v を満たす向き v）ごとに，教師が動かした量の 1/(1 + λ_j) が段 2 の後に残る．
- 導出（2 次近似，ラベル雑音の共分散 ≈ H_K）：得 ≈ ½ Σ_j f_j (e0_j² − eQ_j²)/(1 + λ_j)²．ラベル雑音の項は打ち消し合う．

確かめた水準は次の 3 段で書く．◎ 本文の該当の節・定理を読んだ．○ 抄録・予稿集の頁・書誌データベース（OpenAlex）で確かめた．△ 検索結果の抜き書きだけ．
PDF は arXiv・PMLR・CaltechAUTHORS から取得し，pdftotext で文字にして読んだ．Charikar 2024，Dong 2025，Xue 2025，PPI++，Das & Sanghavi 2023 の中身は，同じ研究ノート群の既存の精読（`research_notes/TAC 論文 先行研究の精読と穴/g6_theory_weak_label_active.md`，`research_notes/ラベル換算の価値で重みを決める先例/report.md`）に頼り，本調査では書誌だけを確かめた．

---

## 問い 1：上の分解，または「教師・疑似ラベルで作った出発点から少数ラベルで学び直すことを固有方向ごとのフィルタで見る」見方は出版されているか

### Takeaway
部品は出版済みである．線形ガウス模型では，事前分布の中心（出発点）の値が，データの行列の固有方向ごとに α/(λ_a + α) の割合で解に残ることを MacKay 1992 が書いている．出発点に L2 で引き戻す転移の期待テスト誤差が「出発点の誤差 × (1 + λ_k/(nα))⁻²」と「出発点に依らない雑音の項」の和になることを Dar, LeJeune, Baraniuk 2024 が等方の線形回帰で示している（Lemma 4.1）．一方で，疑似ラベルで動かした出発点 θ_Q と動かさない出発点 a0 の差をこのフィルタで分解した研究，多クラスのロジスティック回帰の 2 次近似でこの分解を使った研究，実測で確かめた研究は見つからなかった．判定は「部分的な重なり」である．

### Cited Findings

#### 判定の早見表

| 文献 | 模型・損失・体制 | 教師・疑似ラベル → 少数ラベルの学び直し | 固有方向ごとのフィルタ，λ の出どころ | 分類 | 判定 |
|---|---|---|---|---|---|
| Dar, LeJeune, Baraniuk 2024（SIAM J. Math. Data Sci.）[arXiv](https://arxiv.org/abs/2103.05621) | 線形回帰，2 乗損失，n < d も含む．4 節は等方の特徴と等方の事前分布 | 教師なし．ソースの推定値 θ̂ に L2 で引き戻してターゲットの n 本で学ぶ（段 2 と同じ形） | あり．出発点の誤差に (1 + λ_k/(nα))⁻²，λ_k はターゲットの訓練標本の XᵀX の固有値．雑音の項は出発点に依らない | なし | 部分的な重なり（得の式の形がほぼ同じ） |
| MacKay 1992（Neural Computation）[DOI](https://doi.org/10.1162/neco.1992.4.3.415) | 線形ガウス模型（中心 0 の事前分布），証拠（周辺尤度）の枠組み | 教師なし | あり．w_MP,a = γ_a w_ML,a，γ_a = λ_a/(λ_a + α)．λ_a はデータの行列の固有値 | なし（この論文では） | 部分的な重なり（フィルタと「データが決めない向き」） |
| Obst ほか 2021（arXiv のみ）[arXiv](https://arxiv.org/abs/2102.09504) | 線形ガウス模型，固定した計画行列，D < N_T | 教師なし．ソースの最小 2 乗解からターゲットのデータで勾配法を k 回 | あり．ソースの重み (1 − αλ_i)^k，λ_i はターゲットの標本の Σ_T = X_TᵀX_T の固有値 | なし | 部分的な重なり（段 2 のフィルタ） |
| Wu, Zou, Braverman, Gu, Kakade 2022（NeurIPS）[arXiv](https://arxiv.org/abs/2208.01857) | 線形回帰，共変量シフト，オンライン SGD で事前学習 → 微調整，上界と下界 | 教師なし．ソースのラベルで事前学習した出発点からターゲットの N 本で SGD | あり（形が違う）．偏りの項が ∏(I − γ_t H) で縮む．H はターゲットの母集団の共分散．λ_k ≥ 1/(γ_M N_eff) の向きを学ぶ | なし | 部分的な重なり（仕組みの形） |
| Ildiz ほか 2025（ICLR）[arXiv](https://arxiv.org/abs/2410.18837) | リッジ無しの高次元回帰．生徒は代理モデルのラベルだけで学ぶ | 教師あり．真のラベルでの学び直しも出発点も無い | あり（形が違う）．ζ_i = τ/(λ_i + τ)，λ_i は母集団の共分散，τ は n で決まる | なし | 部分的な重なり（n に依る固有方向ごとの教師の価値） |
| Ha & Chen 2025（arXiv のみ）[arXiv](https://arxiv.org/abs/2507.14661) | 線形の構造因果模型，2 乗損失，ミニマックス | 教師なしの DA の推定値から出発し，ラベル付きのターゲットで低次元の部分空間だけを直す | 0/1 の部分空間の分割．部分空間はラベル無しの共変量と介入の構造で決まり，ラベルの曲率では決まらない | なし | 部分的な重なり（段の構造） |
| Li, Han, Yang, Tong 2026（ICML）[PMLR](https://proceedings.mlr.press/v306/li26bd.html) | 線形の教師と生徒，高次元漸近，模型と共変量のずれ | 教師あり．同じラベル付き標本で ξ·教師のラベル ＋ (1 − ξ)·真のラベルを 1 段で学ぶ．出発点は無い | 出発点のフィルタは無い | なし（リッジ回帰） | 違う（問い 3 の話題だけ近い） |
| Mobahi, Farajtabar, Bartlett 2020（NeurIPS）[arXiv](https://arxiv.org/abs/2002.05715) | 核リッジ回帰の自己蒸留を繰り返す | 自分自身が教師．真のラベルでの学び直しは無い | フィルタの形だけ同じ（c/(c + d_k)，d_k は訓練点のグラム行列の固有値） | なし | 違う |
| Phuong & Lampert 2019（ICML）[arXiv](https://arxiv.org/abs/2105.13093) | 線形・深い線形の 2 値分類器の蒸留，勾配流 | 教師あり．真のラベル無し，出発点 0 | 0/1 の射影（データが張る空間） | あり（交差エントロピー） | 違う |
| Dar & Baraniuk 2022（SIAM J. Math. Data Sci.）[arXiv](https://arxiv.org/abs/2006.07002) | 線形回帰，パラメータの一部をソースの値に固定 | 教師なし | なし（座標の部分集合） | なし | 違う |
| Kuzborskij & Orabona 2017（Machine Learning）[arXiv](https://arxiv.org/abs/1412.1619) | 偏った正則化 ‖w − W_src β‖² の ERM，滑らかな損失 | ソースの仮説への引き戻し．教師の疑似ラベル段は無い | なし（汎化の上界） | 損失は一般（滑らかな非負） | 違う |
| Charikar ほか 2024，Dong ほか 2025，Xue ほか 2025，Medvedev ほか 2025，Lang ほか 2024，Somerstep ほか 2025，Liu ほか 2026 | 弱から強への一般化の理論 | 生徒は教師のラベルで学ぶ．出発点＋少数ラベルの学び直しの組は無い | Xue のみ形の近い射影 P = K(K + βI)⁻¹ | 定理は主に 2 乗損失 | 違う |
| Oymak & Gulcu 2021，Wei ほか 2021 | 自己学習の理論 | 疑似ラベルの反復．出発点への引き戻しと少数ラベルの学び直しの組は無い | なし | 線形分類器・深層網 | 違う |

#### 各文献の詳細

**Dar, LeJeune, Baraniuk, "The Common Intuition to Transfer Learning Can Win or Lose: Case Studies for Linear Regression"（◎ 抄録，3 節，4.1 節の Lemma 4.1）**
- 書誌：SIAM Journal on Mathematics of Data Science 6, 454–480（2024），DOI 10.1137/23M1563062（OpenAlex で確認，○）．arXiv 2103.05621v4（2024-05-31）の本文を読んだ — [arXiv](https://arxiv.org/abs/2103.05621)；[DOI](https://doi.org/10.1137/23M1563062)
- 転移の推定量は式 (3.1) の β̂ = argmin_b ‖y − Xb‖² + nα_TL‖H̃b − θ̂‖² である．θ̂ はソースの課題で計算済みの推定値で，H̃ が未知なら H̃ = I_d と置ける．著者は「in many cases this can be conceptually perceived as a fine tuning strategy」と書く — [arXiv](https://arxiv.org/abs/2103.05621)
- Lemma 4.1（H̃ = H が直交行列，Σ_x = I_d，β が等方のガウス分布）：期待テスト誤差は Ē_TL = σ_ε² + E[Σ_k (n²α_TL² C_TL + σ_ε² λ_k(X_ΨᵀX_Ψ)) / (λ_k(X_ΨᵀX_Ψ) + nα_TL)²]．λ_k は d × d の行列 X_ΨᵀX_Ψ（ターゲットの訓練標本から作る）の固有値である．C_TL は「課題間の距離（σ_η²）」と「ソースの解の精度（σ_ξ² とソースのパラメータ数）」を含む — [arXiv](https://arxiv.org/abs/2103.05621)
- 抄録の主張：課題が十分に近ければ，最適に調整した転移は最適に調整したリッジ回帰（独立に学んだターゲットの MMSE 解）に勝つ．最小 ℓ2 ノルム解の二重降下の山を消す — [arXiv](https://arxiv.org/abs/2103.05621)
- 5・6 節は H が一部だけ既知の場合と，H と Σ_x が一般の場合を扱う（6 節の見出しと命題 6.1 の位置だけ確かめ，中身は読んでいない）．

**MacKay, "Bayesian Interpolation"（◎ 4 節の式 (4.9) の周辺と図 6）**
- 書誌：Neural Computation 4(3), 415–447（1992），DOI 10.1162/neco.1992.4.3.415（OpenAlex で確認）．本文は CaltechAUTHORS の PDF で読んだ — [DOI](https://doi.org/10.1162/neco.1992.4.3.415)；[PDF](https://authors.library.caltech.edu/records/r7qgh-q6g10/files/MACnc92a.pdf)
- 図 6 の説明：データの行列 B の固有ベクトルに平行な成分 w1, w2 について，「w1 is a direction in which λ1 is small compared to α, that is, the data have no strong preference about the value of w1; w1 is a poorly measured parameter」，「w2 is well determined by the data」— [PDF](https://authors.library.caltech.edu/records/r7qgh-q6g10/files/MACnc92a.pdf)
- 式 (4.9) の周辺：γ_a = λ_a/(λ_a + α) は「measures the strength of the data in direction a relative to the prior」で，事後分布の最頻値の成分は w_MP,a = γ_a w_ML,a．γ = Σ_a γ_a は「the effective number of parameters that are well determined by the data」．「Poorly determined parameters are determined by the regularizer only」— [PDF](https://authors.library.caltech.edu/records/r7qgh-q6g10/files/MACnc92a.pdf)
- 事前分布の中心は 0 である．中心を出発点 a0 や θ_Q に置いた場合は書いていない．

**Obst, Ghattas, Cugliari, Oppenheim, Claudel, Goude, "Transfer Learning for Linear Regression: a Statistical Test of Gain"（◎ 2 節）**
- 書誌：arXiv 2102.09504v1（2021-02-18，math.ST）．OpenAlex にも arXiv 以外の掲載先は無かった — [arXiv](https://arxiv.org/abs/2102.09504)
- 設定：ガウスの線形模型，計画行列は固定で D < N_ν（低次元）．ソースの最小 2 乗解 β̂_S から，ターゲットのデータで歩幅 α のバッチ勾配法を k 回行う — [arXiv](https://arxiv.org/abs/2102.09504)
- 命題 1：β̂_k = A^k β̂_S + (I − A^k) β̂_T，A = I − αΣ_T，Σ_T = X_TᵀX_T．Σ_T の固有基底では β̃_k^(i) = (1 − αλ_i)^k β̃_S^(i) + (1 − (1 − αλ_i)^k) β̃_T^(i)．著者は「For small eigenvalues of Σ_T the fine-tuning procedure will give a larger weight to the source whereas it is the opposite for larger ones」と書く — [arXiv](https://arxiv.org/abs/2102.09504)
- 命題 2：ターゲットだけの解に対する得は ΔR_k(x) = xᵀH_k x．H_k は σ_T²，σ_S²，B = (β_T − β_S)(β_T − β_S)ᵀ を含む．得の正負は x ごとに決まる — [arXiv](https://arxiv.org/abs/2102.09504)
- 比べる相手はターゲットだけの解で，2 つの出発点の差は比べていない．雑音の項は残る．

**Wu, Zou, Braverman, Gu, Kakade, "The Power and Limitation of Pretraining-Finetuning for Linear Regression under Covariate Shift"（◎ 1〜3 節，定理 3.1）**
- 書誌：NeurIPS 2022，pp. 33041–33053，DOI 10.52202/068431-2394（OpenAlex で確認，○）．arXiv 2208.01857v1 の本文を読んだ — [arXiv](https://arxiv.org/abs/2208.01857)；[DOI](https://doi.org/10.52202/068431-2394)
- 設定：ソースとターゲットで入力の分布だけが違う線形回帰．ソースの M 本でオンライン SGD の事前学習をし，続けてターゲットの N 本で微調整する．歩幅は幾何級数で減らす — [arXiv](https://arxiv.org/abs/2208.01857)
- 定理 3.1：偏りの誤差の主項は ‖∏_{t=M}^{M+N−1}(I − γ_t H) ∏_{t=0}^{M−1}(I − γ_t G)(w0 − w*)‖²_H．上界は添字の集合 K = {k : λ_k ≥ 1/(γ_M N_eff)}（H の固有値）と J = {j : μ_j ≥ 1/(γ_0 M_eff)}（G の固有値）で最適になる．H はターゲットの母集団の共分散，G はソースの母集団の共分散である — [arXiv](https://arxiv.org/abs/2208.01857)
- 抄録の主張：多くの例で O(N²) 本のソースでの事前学習が N 本のターゲットでの教師あり学習と同じ効果を持つ．少量のターゲットでの微調整が必要なソースの量を大きく減らす — [arXiv](https://arxiv.org/abs/2208.01857)

**Ildiz, Gozeten, Taga, Mondelli, Oymak, "High-dimensional Analysis of Knowledge Distillation: Weak-to-Strong Generalization and Scaling Laws"（◎ 抄録，3 節の定理 1，命題 1，系 1．掲載先は ○）**
- 書誌：ICLR 2025．掲載先は ISTA の研究記録と ML Anthology の検索結果で確かめた（OpenReview の頁は開いていない）— [arXiv](https://arxiv.org/abs/2410.18837)；[ML Anthology](https://mlanthology.org/iclr/2025/ildiz2025iclr-highdimensional)；[ISTA](https://research-explorer.ista.ac.at/record/20033)
- 設定：生徒は代理モデル（教師）が付けたラベルだけで，リッジ無しの最小ノルム補間で学ぶ．(i) 代理モデルが任意の場合と (ii) 分布のずれたデータの ERM の場合 — [arXiv](https://arxiv.org/abs/2410.18837)
- 漸近の危険度は θ1 = Σ_t(Σ_t + τ_t I)⁻¹ を通して書かれ，最適な代理モデルの各座標は ζ_i = τ_t/(λ_i + τ_t) で決まる（系 1）．主成分では真の係数より増幅し，裾では縮め，さらに裾は 0 にするのが最適になる．「as the sample size n decreases, the optimal surrogate provably becomes sparser」— [arXiv](https://arxiv.org/abs/2410.18837)
- 抄録の主張：弱から強の学習は同じデータ量で真のラベルでの学習に勝ちうるが，データ量に対する誤差の縮み方（scaling law）は良くできない — [arXiv](https://arxiv.org/abs/2410.18837)

**Ha & Chen, "When few labeled target data suffice: a theory of semi-supervised domain adaptation via fine-tuning from multiple adaptive starts"（◎ 抄録と式 (16) の周辺）**
- 書誌：arXiv 2507.14661v2（初版 2025-07-19），著者 Wooseok Ha, Yuansi Chen．掲載先は確かめていない — [arXiv](https://arxiv.org/abs/2507.14661)
- FT-DIP(1)（式 (16)）：教師なし DA の推定値 β̂_DIP から出発し，ラベル付きのターゲット n(0) 本の最小 2 乗を，制約 Q̂ᵀΣ̂_X(β − β̂_DIP) = 0 と ‖V̂ᵀΣ̂β‖ ≤ ϱ の下で解く．著者は「unlabeled covariates determine the fine-tuning subspace, while labeled target samples estimate the correction within that subspace」と書く — [arXiv](https://arxiv.org/abs/2507.14661)
- 抄録の主張：ラベル付きのターゲットの必要数は，周囲の次元ではなく介入の次元に比例する — [arXiv](https://arxiv.org/abs/2507.14661)

**Li, Han, Yang, Tong, "Quantifying Cross-Domain Knowledge Distillation in the Presence of Domain Shift"（◎ 抄録，序論，式 (1)，貢献の列挙．定理の中身は読んでいない）**
- 書誌：ICML 2026，PMLR 306:67268–67304 — [PMLR](https://proceedings.mlr.press/v306/li26bd.html)
- 教師はソースで学び，生徒はターゲットの標本ごとに L(ξ) = ξℓ(y_t, y_s) + (1 − ξ)ℓ(y, y_s) を最小化する（ξ は imitation parameter）．生徒はリッジ回帰で β_s = (X_2X_2ᵀ + N_2λ_s I)⁻¹(ξX_2 y_t + (1 − ξ)X_2 y_2) の形になる — [PMLR](https://proceedings.mlr.press/v306/li26bd.html)
- 貢献：ずれが大きくても生徒だけの基準に勝つ ξ ∈ R が存在しうる．その存在は模型と両領域の共分散の幾何で決まる．教師と生徒の次元と標本数の比について「crossed double descent」が出る — [PMLR](https://proceedings.mlr.press/v306/li26bd.html)

**Mobahi, Farajtabar, Bartlett, "Self-Distillation Amplifies Regularization in Hilbert Space"（◎ 2 節，命題 2）**
- 書誌：NeurIPS 2020（PDF の脚注で確認）．arXiv 2002.05715v3 — [arXiv](https://arxiv.org/abs/2002.05715)
- 核リッジ回帰の解 f*(x) = g_xᵀ(cI + G)⁻¹y を使い，回転したラベル z = Vy で訓練誤差が (1/K)Σ_k (z_k c/(c + d_k))² になる（命題 2）．自己蒸留を繰り返すと使える基底関数の数が減り，数回は過学習を減らすが続けると学習不足になる — [arXiv](https://arxiv.org/abs/2002.05715)

**Phuong & Lampert, "Towards Understanding Knowledge Distillation"（◎ 4.1 節）**
- 書誌：ICML 2019（arXiv の注記「ICML'19」，OpenAlex は PMLR 97:5142–5151 と記録）．arXiv 2105.13093 — [arXiv](https://arxiv.org/abs/2105.13093)
- 線形の生徒を教師の柔らかいラベルで勾配流で学ぶと，n ≥ d なら教師の重みをそのまま得る．n < d なら「the weight vector learned by the student is simply the projection of the teacher's weight vector onto the data span」— [arXiv](https://arxiv.org/abs/2105.13093)

**Dar & Baraniuk, "Double Double Descent: On Generalization Errors in Transfer Learning between Linear Regression Tasks"（○ 抄録と 2 節の設定）**
- 書誌：SIAM Journal on Mathematics of Data Science 4, 1447–1472（2022），DOI 10.1137/22M1469559（OpenAlex で確認）— [arXiv](https://arxiv.org/abs/2006.07002)；[DOI](https://doi.org/10.1137/22M1469559)
- ターゲットの解のパラメータの一部をソースで学んだ値に固定する転移を解析し，自由なパラメータ数について 2 次元の二重降下を示す — [arXiv](https://arxiv.org/abs/2006.07002)

**Kuzborskij & Orabona, "Fast Rates by Transferring from Auxiliary Hypotheses"（○ 抄録と式 (3)）**
- 書誌：Machine Learning 106, 171–195，DOI 10.1007/s10994-016-5594-4（OpenAlex で確認．OpenAlex の年は 2016）— [arXiv](https://arxiv.org/abs/1412.1619)；[DOI](https://doi.org/10.1007/s10994-016-5594-4)
- 最小 2 乗の biased regularization：ŵ = argmin Σ(⟨w, x_i⟩ − y_i)² + λ‖w − W_src β‖²（式 (3)）．ソースの仮説の組み合わせが良ければ O(1/m)，悪ければ通常の O(1/√m) の速さを示す — [arXiv](https://arxiv.org/abs/1412.1619)

**弱から強への一般化の理論**
- Charikar, Pabbaraju, Shiragur 2024（NeurIPS）：凸な関数族の生徒が 2 乗損失で弱い教師のラベルに合わせると，生徒の誤差 ≤ 教師の誤差 − misfit．生徒は真のラベルを見ず，正則化も無い — [arXiv](https://arxiv.org/abs/2405.15116)（中身は既存ノート g6 の ◎）
- Dong ほか 2025（ICML）：先生のラベル雑音由来の分散は，先生と生徒の特徴の部分空間が重なる部分ではそのまま移り，食い違う部分では d_s/N 倍に減る．リッジ無し，出発点無し — [arXiv](https://arxiv.org/abs/2502.05075)（既存ノート g6 の ◎）
- Xue, Li, Mirzasoleiman 2025（ICML）：PredGap = ‖P_s(I − P_w)ŷ/√n̂‖² ± o(1)．P はスケールした射影 (K̂/n̂)(K̂/n̂ + (β + γ̃)I)⁻¹ — [PMLR](https://proceedings.mlr.press/v267/xue25d.html)（既存ノート g6 の ◎）
- Medvedev, Lyu, Yu, Arora, Li, Srebro 2025（ICML．掲載先は icml.cc のポスター頁の検索結果で確認，△）：無作為特徴の 2 層網の教師と生徒で，生徒の早期停止が弱から強への一般化を可能にし，その大きさには限界がある — [arXiv](https://arxiv.org/abs/2503.02877)；[icml.cc](https://icml.cc/virtual/2025/poster/45441)
- Lang, Sontag, Vijayaraghavan 2024（NeurIPS，pp. 46837–46880，OpenAlex で確認）：疑似ラベルの修正（pseudolabel correction）と網羅の拡大（coverage expansion）を，データ分布の expansion の性質に基づく上界で説明する — [arXiv](https://arxiv.org/abs/2405.16043)
- Somerstep ほか 2025（ICLR．PDF の頭書きで確認）：弱から強を潜在概念の事前分布の転移として定式化し，素朴な微調整には根本的な限界があり，問題の構造から出る refinement の方法がそれを避けると示す — [arXiv](https://arxiv.org/abs/2405.16236)
- Liu, Dong, Lei 2026（ICLR 2026．掲載先は iclr.cc のポスター頁の検索結果で確認，△）：教師はラベル付きデータで微調整した後に疑似ラベルを付け，強い生徒がそれで学ぶ．少数派の群の割合が両者で一致すれば W2S の得が出て，ずれると (η_u − η_ℓ)² とともに減る — [arXiv](https://arxiv.org/abs/2509.24005)；[iclr.cc](https://iclr.cc/virtual/2026/poster/10011429)

**自己学習の理論**
- Oymak & Gulcu：AISTATS 2021，PMLR 130:3601–3609（OpenAlex，題 "A Theoretical Characterization of Semi-supervised Learning with Self-training for Gaussian Mixture Models"）．arXiv 2006.11006 の抄録は，ガウス混合での自己学習の反復を有限標本で特徴づけ，確信度の低い標本を捨てる利点を示す — [arXiv](https://arxiv.org/abs/2006.11006)；[PMLR](https://proceedings.mlr.press/v130/oymak21a/oymak21a.pdf)
- Wei, Shen, Chen, Ma 2021（ICLR．arXiv の注記で確認）：expansion の仮定の下で，自己学習と入力の一貫性の正則化の母集団の目的関数の最小化点が高い正解率を持つ — [arXiv](https://arxiv.org/abs/2010.03622)

### Inferences
- 線形ガウス模型（2 乗損失）では，実測した式は近似ではなく恒等式になる．段 2 の解は θ = (H + P)⁻¹(Xᵀy/σ² + Pμ)（μ は出発点）なので，出発点を a0 から θ_Q に替えたときの差は (H + P)⁻¹P(θ_Q − a0) にちょうど等しく，ラベル y に依らない．MacKay 1992 の γ_a を中心 μ の事前分布に書き直すと，μ の成分が 1 − γ_a = α/(λ_a + α) = 1/(1 + λ_a/α) の割合で残る．したがって「1/(1 + λ_j) が残る」という部分は，既知の事実を一般化固有方向で書き直したものとして扱うのが安全である．
- Dar, LeJeune, Baraniuk 2024 の Lemma 4.1 の和の項は C_TL/(1 + λ_k/(nα))² + σ²λ_k/(λ_k + nα)² と分けられる．第 1 項が出発点の誤差の (1 + λ)⁻² 倍，第 2 項が出発点に依らない雑音の項である．2 つの出発点で C_TL だけが違えば，差は Σ_k (C_TL^(0) − C_TL^(Q))/(1 + λ_k/(nα))² になり，本研究の得の式の f_j = 1・誤差が等方の場合に当たる．本研究の式は，これを (a) 方向ごとの誤差 e0_j, eQ_j，(b) ターゲットの曲率 f_j，(c) ロジスティック回帰の 2 次近似（H_K を θ_s で取る）に広げたものと位置づけられる．
- 先行研究に無かったのは次の 4 点である．(1) 出発点そのものを疑似ラベルの当てはめ（段 1）で作り，段 1 が動かした量 θ_Q − a0 を段 2 のフィルタで分解したこと．(2) 多クラスのロジスティック回帰での実測（相対誤差 1〜2%）．(3) λ > 1 の成分が K = 25 で教師の出発点だけの価値の約半分を持ち，段 2 で消えるという量的な結果．(4) 教師の得を K = 0 の教師なし DA の得から K > 0 へつなぐ式．ただし (1) は「出発点を何で作るか」を替えただけなので，数式の新しさとして前に出すより，「教師の価値を K の関数として分解する道具」として書くほうが査読で崩れにくい．
- 論文で避けたほうがよい言い方：「少数ラベルでの微調整を固有方向のフィルタとして初めて見た」．MacKay 1992，Obst 2021，Wu 2022，Dar 2024 がある．「出発点の誤差が (1 + λ)⁻² で効き雑音が打ち消すことを初めて示した」．Dar 2024 の Lemma 4.1 が等方の場合に同じ形を持つ．

### Gaps
- Dar, LeJeune, Baraniuk 2024 の 6 節（H と Σ_x が一般の場合）で，方向ごとの出発点の誤差に (1 + λ_k)⁻² の重みが付く形が明示されているかは読んでいない．明示されていれば本研究の得の式（f_j ≠ 1）との重なりがさらに大きくなる．
- 偏った正則化（L2-SP など）や，ソースの事後分布を事前分布に使うベイズ転移で，2 つの事前分布の中心の差を固有方向ごとに比べた解析は，検索では見つからなかった．見つからなかったことは，無いことの証明ではない．
- MacKay の証拠の枠組みを分類に広げた "The evidence framework applied to classification networks"（MacKay 1992）は確かめていない．ロジスティック回帰でのラプラス近似（事後分布を最頻値の周りの 2 次式で近似する方法）の下で同じ γ_a の解釈を述べた箇所があるかは未確認である．
- Evron ほか（COLT 2022，arXiv 2205.09588．書誌は arXiv の journal_ref で確認）は，過剰パラメータの線形回帰で課題を順に学ぶときの忘却を厳密に書く．前の課題の解が次の課題のデータの零空間に残るという，本研究の λ → 0 の極限に当たる記述が本文にあるかは読んでいない — [arXiv](https://arxiv.org/abs/2205.09588)
- Sun & Raskutti（arXiv 2602.13942，事前学習した出発点からの早期停止の理論）と Moniri & Hassani（arXiv 2505.18346，リッジ回帰での教師と生徒の正則化の相互作用）は題と一部の文しか見ていない．

---

## 問い 2：「少数ラベルの下での教師（疑似ラベル）の価値は，ラベルが学べない向きにある」「その向きでは教師が正しいかをラベルで確かめられない」と述べた先例はあるか

### Takeaway
後半の主張（曲率の小さい向きではラベルの尤度がほぼ平らで，データがどちらの値も選ばない）は，MacKay 1992 の「poorly measured parameter」の記述そのものである．前半の主張（出発点や事前学習の寄与は，少数ラベルの曲率が小さい向きにだけ残る）は，Obst 2021（標本の共分散の小さな固有値の向きでソースの重みが大きい），Wu 2022（母集団の共分散の裾で事前学習の偏りが残る），Ha & Chen 2025（ラベルで直さない部分空間では教師なし DA の解を保つ）に形を変えて出ている．2 つを結び，「教師の価値が残る向きは，K 本のラベルで教師と出発点のどちらが正しいかを判定できない向きと一致する」と述べた文献は見つからなかった．判定は「部分的な重なり」である．

### Cited Findings
- MacKay 1992：λ1 が α より小さい向き w1 では「the data have no strong preference about the value of w1」．「Poorly determined parameters are determined by the regularizer only, so they do not reduce χ²_W in this way」— [PDF](https://authors.library.caltech.edu/records/r7qgh-q6g10/files/MACnc92a.pdf)
- Obst ほか 2021：Σ_T（ターゲットの標本 X_TᵀX_T）の固有値が小さい向きでは，微調整した解はソースの係数に大きな重みを置き，固有値が大きい向きではターゲットの係数に置く．得は x ごとに正にも負にもなり，「an uniformly positive transfer may be impossible」— [arXiv](https://arxiv.org/abs/2102.09504)
- Wu ほか 2022：上界は K = {k : λ_k ≥ 1/(γ_M N_eff)} で最適になり，事前学習の出発点の誤差 w0 − w* は ∏(I − γ_t H) で縮む．λ_k が小さい向きでは微調整で縮まず，事前学習の質がそのまま残る — [arXiv](https://arxiv.org/abs/2208.01857)
- Ha & Chen 2025：FT-DIP では，ラベル無しの共変量が微調整の部分空間を決め，ラベル付きのターゲットはその部分空間の中の補正だけを推定する．残りの向きは教師なし DA の解に固定される — [arXiv](https://arxiv.org/abs/2507.14661)
- Ildiz ほか 2025：生徒（ラベルは教師のものだけ）は θ1 = Σ(Σ + τI)⁻¹ を通してしか教師の係数を写せず，n が減ると最適な教師は裾の向きを 0 にする．ここでは「低 λ の向きは生徒が学べないので，教師の情報も届かない」という逆向きの結論になる — [arXiv](https://arxiv.org/abs/2410.18837)
- Phuong & Lampert 2019：n < d の線形の蒸留では，生徒はデータの張る空間の外を学べない（「there is no way for the student to learn anything outside of it」）— [arXiv](https://arxiv.org/abs/2105.13093)

### Inferences
- 「確かめられない」は線形ガウス模型で式にできる．出発点（事前分布の中心）を μ とすると，ラベルの周辺分布は y ~ N(Xμ, σ²I + XC′Xᵀ) である．μ = a0 と μ = θ_Q の対数周辺尤度の差は X(θ_Q − a0) の 2 次式で，一般化固有方向 j の寄与は λ_j/(1 + λ_j) 倍に縮む．λ_j が小さい向きでは，K 本のラベルは a0 と θ_Q を見分けられない．段 2 で残る割合 1/(1 + λ_j) と，見分ける力 λ_j/(1 + λ_j) は足して 1 になる．この導出は本ノートの推論で，文献には見つからなかった．
- この補完関係から，K 本のラベルでの交差検証や検証用の分割で m や教師を選ぶ方法は，教師の価値のうち段 2 の後に残る部分（低 λ の向き）をほとんど見られない，という予言が出る．本研究の「ハイパーパラメータはテストラベルで選ばない」方針の下では，K 本の検証で選んだ m が最適からずれる理由の説明に使える．
- Ildiz 2025 の結論（低 λ の向きでは教師の情報が生徒に届かない）と本研究の結論（低 λ の向きにだけ教師の価値が残る）は逆に見える．違いは段の順序である．Ildiz では教師の情報は n 本の標本の上のラベルとして入るので，標本の曲率が小さい向きには入らない．本研究では教師の情報は段 1 で大量のラベル無しの標本を通して入り，段 2 の K 本は曲率の大きい向きだけを上書きする．論文の関連研究ではこの違いを 1 文で書くとよい．

### Gaps
- 「教師が正しいかを確かめられない向き」をラベルの周辺尤度や Fisher 情報で定義し，蒸留や半教師ありの重み選びに使った文献は見つからなかった．検索語は「pseudo-label then fine-tune theory eigendirections」「weak-to-strong few gold labels theory ridge」「pseudo-label pre-training fine-tuning prior directions not identified」．
- 能動学習のベイズ最適設計（Chaloner & Verdinelli 1995 など）は「ラベルで決まらない向きにラベルを足す」話で，本問いの裏返しに当たる．既存ノート g6 の 6 節に書かれているが，本調査では確かめ直していない．

---

## 問い 3：K に依存する蒸留・模倣・疑似ラベルの最適な重みを，このようなスペクトルのフィルタを通して導いた先例はあるか

### Takeaway
見つからなかった．重みの最適値を標本数の関数として出す研究（Li 2026 の ξ，Jain 2024，PPI++，Dang ほか 2026，Das & Sanghavi 2023）と，最適な教師の形を n に依る固有方向ごとの式で出す研究（Ildiz 2025）はある．しかし「出発点に疑似ラベルを総重み m で積み，K 本で引き戻す」ときの最適な m を，重み f_j/(1 + λ_j(K))² を通して K に結びつけた式は無かった．判定は「違う」（Ildiz 2025 だけ部分的な重なり）である．

### Cited Findings
- Li, Han, Yang, Tong 2026：模倣の重み ξ を ℓ(教師のラベル) と ℓ(真のラベル) の凸結合の係数として入れ，ずれがあっても生徒だけの基準に勝つ ξ ∈ R が存在しうること，その存在が模型と共分散の幾何で決まることを示す．教師のラベルと真のラベルは同じ標本の上に付き，出発点は無い — [PMLR](https://proceedings.mlr.press/v306/li26bd.html)
- Ildiz ほか 2025：最適な代理モデルの各座標は ζ_i = τ_t/(λ_i + τ_t) で決まり，n が減ると疎になる．決めているのは教師の係数の形で，教師のラベルにかける重みではない — [arXiv](https://arxiv.org/abs/2410.18837)
- Jain, Montanari, Sasoglu（arXiv 2402.04376，掲載先は確かめていない）：n 本の真のデータに代理のデータを重みつき ERM で混ぜる．最適な重みを使うことが大事で，誤差は scaling law で近似でき，その式から最適な重みを予言できる — [arXiv](https://arxiv.org/abs/2402.04376)
- Dang, Patil, Rinaldo（arXiv 2602.17565．ICML 2026 のポスター頁が検索に出た，△）：リッジ回帰の自己蒸留で，混ぜる重みに [0, 1] の制約を置かない最適値と，その一発での調整法を出す — [arXiv](https://arxiv.org/abs/2602.17565)；[icml.cc](https://icml.cc/virtual/2026/poster/64532)
- Dang, Patil, Rinaldo（arXiv 2607.15450）：元の訓練データが無く，教師の予測と新しいラベル無しの共変量だけがある場合の自己蒸留を，リッジ回帰の比例漸近で解析する．題は線形回帰とロジスティック回帰を扱うと書く — [arXiv](https://arxiv.org/abs/2607.15450)
- Lecoiu, Mukherjee, Sur（arXiv 2605.17778）：自己蒸留をスペクトル縮小推定量（固有値ごとに縮める推定量）の族として扱い，スパイクが s 本の共分散では s 段の自己蒸留がその族の中で最適になる — [arXiv](https://arxiv.org/abs/2605.17778)
- PPI++（Angelopoulos, Duchi, Zrnic，arXiv 2311.01453）と Das & Sanghavi 2023（arXiv 2301.13304）は既存ノート「ラベル換算の価値で重みを決める先例」で本文を確かめた．そこでの要約では PPI++ の係数は λ* ∝ 1/(1 + n/N) で n とともに下がり，Das & Sanghavi は線形回帰で模倣の重みの最適値を出す．いずれも固有方向ごとのフィルタは使わない — [PPI++](https://arxiv.org/abs/2311.01453)；[Das & Sanghavi](https://arxiv.org/abs/2301.13304)
- Dar, LeJeune, Baraniuk 2024 の定理 4.2 は段 2 の罰則の強さ α_TL の最適値を出す．疑似ラベルの重み m に当たる量は無い — [arXiv](https://arxiv.org/abs/2103.05621)

### Inferences
- 本研究の (iii)「m は K を重み f_j/(1 + λ_j(K))² を通してだけ受け取る」には条件が 2 つ要る．(a) 一般化固有ベクトル v_j も K 本の標本で変わるので，e0_j，eQ_j(m)，f_j も K に依る．「重みを通してだけ」と言えるのは，基底を固定して見たとき（または K を増やしても基底がほぼ変わらないとき）に限る．(b) f_j = v_jᵀ F v_j と書くのは，ターゲットの曲率 F の非対角の成分を無視したときである．論文ではこの 2 つを仮定として書く必要がある．
- K = 0 では λ_j = 0 なので重みは f_j で，最適な m は教師なし DA の最適値に一致する．K が増えると λ_j が大きい向きの重みが (1 + λ_j)⁻² で減るので，最適な m は「低曲率の向きで疑似ラベルが誤差を減らす量」だけで決まるようになる．この予言は先行研究に無く，実験で確かめられる形をしている．
- 先行研究の重みの式（Li 2026 の ξ，PPI++ の λ，Jain 2024 の重み）は 1 つの数で教師とラベルの比を決める．本研究の式は，同じ 1 つの数 m が向きごとに違う重みで評価されることを示す点で違う．比べるときは「重みの最適値が n とともに下がる」こと自体は既知（PPI++，Jain 2024）と書き，新しさを「下がり方を一般化固有方向のフィルタで書く」ことに置く．

### Gaps
- Li 2026 の定理 3.1〜3.7 が ξ の最適値を標本数の関数として閉じた形で出しているかは読んでいない．出していれば，本研究の (iii) と比べる式が 1 本増える．
- Dang ほか 2607.15450 のロジスティック回帰の部分の結果（何を示したか）は抄録の冒頭しか読んでいない．
- 段 1 の総重み m と段 2 の罰則 C′ を同時に最適化した解析は見つからなかった．

### BibTeX（部分的な重なり以上の文献）

```bibtex
@article{dar2024common,
  author  = {Dar, Yehuda and LeJeune, Daniel and Baraniuk, Richard G.},
  title   = {The Common Intuition to Transfer Learning Can Win or Lose: Case Studies for Linear Regression},
  journal = {SIAM Journal on Mathematics of Data Science},
  volume  = {6},
  pages   = {454--480},
  year    = {2024},
  doi     = {10.1137/23M1563062},
  note    = {arXiv:2103.05621}
}

@article{mackay1992bayesian,
  author  = {MacKay, David J. C.},
  title   = {Bayesian Interpolation},
  journal = {Neural Computation},
  volume  = {4},
  number  = {3},
  pages   = {415--447},
  year    = {1992},
  doi     = {10.1162/neco.1992.4.3.415}
}

@misc{obst2021transfer,
  author        = {Obst, David and Ghattas, Badih and Cugliari, Jairo and Oppenheim, Georges and Claudel, Sandra and Goude, Yannig},
  title         = {Transfer Learning for Linear Regression: a Statistical Test of Gain},
  year          = {2021},
  eprint        = {2102.09504},
  archivePrefix = {arXiv},
  primaryClass  = {math.ST}
}

@inproceedings{wu2022power,
  author    = {Wu, Jingfeng and Zou, Difan and Braverman, Vladimir and Gu, Quanquan and Kakade, Sham M.},
  title     = {The Power and Limitation of Pretraining-Finetuning for Linear Regression under Covariate Shift},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {35},
  pages     = {33041--33053},
  year      = {2022},
  note      = {arXiv:2208.01857; pages from OpenAlex}
}

@inproceedings{ildiz2025highdimensional,
  author    = {Ildiz, M. Emrullah and Gozeten, Halil Alperen and Taga, Ege Onur and Mondelli, Marco and Oymak, Samet},
  title     = {High-dimensional Analysis of Knowledge Distillation: Weak-to-Strong Generalization and Scaling Laws},
  booktitle = {International Conference on Learning Representations},
  year      = {2025},
  note      = {arXiv:2410.18837; venue confirmed via ISTA record and ML Anthology, OpenReview page not opened}
}

@misc{ha2025few,
  author        = {Ha, Wooseok and Chen, Yuansi},
  title         = {When few labeled target data suffice: a theory of semi-supervised domain adaptation via fine-tuning from multiple adaptive starts},
  year          = {2025},
  eprint        = {2507.14661},
  archivePrefix = {arXiv}
}
```
