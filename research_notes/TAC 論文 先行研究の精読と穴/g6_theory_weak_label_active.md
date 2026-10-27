# G6：弱いラベル・能動学習・マルチモーダルの理論 9 本の精読（本研究の手法側の命題の置き場所）

調べた日：2026-10-27．各論文の本文（arXiv・PMLR の PDF をダウンロードし pdftotext で読んだ）を読み，「どの仮定で，何を証明し，どこで止まっているか」を本研究の軸 A1〜A11 に沿って記録する．
確かめた水準は ◎ 本文（該当の節・定理を読んだ），○ 抄録か予稿集・出版社の頁，△ 検索結果の抜き書きだけ，の 3 段で書く．書誌は確かめた範囲だけを書き，確かめられなかったものは「要確認」と書く．
記号の対応：本研究の W_0（ソースで学んだ出発点），E_0 = W* − W_0（出発点の向きの誤り），K（クラスあたりのターゲットのラベル数），J（先生と出発点の相補性）を各論文の記号に対応づけて書く．読み手（本ノート）の推論は「推論」と明記する．

本研究の手法側の命題（本ノートが置き場所を定める対象）は，(命題 1) W − W_0 の各行はラベルを付けた発話が張る高々 4K 次元の空間に入る（表現定理），(命題 4) ガウスの事前分布 W ~ N(W_0, C′I) の下では事後共分散がラベルの値を含まないので，E_0 の向きの知識が無い限りラベル無しの設計はランダムと変わらない，(弱いラベルの上限) ソースの分類器（先生）の出力を弱いラベルにしたときの得は J（先生が正しく出発点が誤る割合 − その逆）で上限が決まる，の 3 つである．

---

## 1. Charikar, Pabbaraju, Shiragur 2024 — Quantifying the Gain in Weak-to-Strong Generalization

### 1.1 書誌と読んだ範囲
- NeurIPS 2024（PDF の脚注「38th Conference on Neural Information Processing Systems (NeurIPS 2024)」で確認）．arXiv 2405.15116v2（2024-10-23）— [arXiv](https://arxiv.org/abs/2405.15116)（◎）
- 読んだ範囲：1〜6 節（序論，関連研究と準備，結果（定理 1・2），証明の方針（4 節），実験（5.1〜5.3），結論），付録 A（定理 2 の証明の冒頭），付録 C（NLP の実験の設定），付録 E（分類への拡張の議論）．読まなかった範囲：付録 B（ESOL・FreeSolv の表），付録 D（非凸な F_s の実験，図の見出しだけ読んだ），付録 F（実装）．

### 1.2 設定
- 模型：データ x ∈ R^d，真の表現 h*: R^d → R^{d*}，真の課題 f*∘h*（実数値の回帰）．弱いモデルは表現 h_w と任意の関数 f_w で f_w∘h_w を出力する．強いモデルは固定の表現 h_s の上で，凸な関数の集合 F_s から f_sw を選ぶ．損失は 2 乗距離 d_P(f, g) = E_{x∼P}(f(x) − g(x))² — [arXiv 2405.15116](https://arxiv.org/abs/2405.15116)
- 先生と生徒の関係：生徒（強いモデル）は真のラベルを一切見ず，先生（弱いモデル）が付けたラベル f_w∘h_w(x) だけに最小 2 乗で合わせる．正則化は入れない（「our work does not assume that the student model has any access to ground-truth labels, and also does not incorporate a regularization term in the objective」）— [arXiv 2405.15116](https://arxiv.org/abs/2405.15116)
- 標本数の体制：定理 1 は母集団（無限標本），定理 2 は弱いラベル付きの n 標本．
- 本研究の記号との対応：先生 = ソースで学んだ分類器（別モダリティの先生なら h_w が別モダリティの特徴），生徒 = ターゲットのモダリティの凍結特徴 h_s の上の線形の頭 W．本研究の出発点 W_0 に当たるものは無い（生徒は 0 から学ぶ）．

### 1.3 示したこと
- 定理 1（実現可能な場合）：F_s が凸で，ある f_s ∈ F_s について f_s∘h_s = f*∘h* が成り立つなら，d_P(f_sw∘h_s, f*∘h*) ≤ d_P(f_w∘h_w, f*∘h*) − d_P(f_sw∘h_s, f_w∘h_w)．日常語では「生徒の真の誤差は，先生の真の誤差から，生徒が先生のラベルに合わせきれなかった量（misfit）を引いた値以下」，つまり得 ≥ misfit．証明は凸集合への射影のピタゴラスの定理（Hazan 2016, Thm 2.1）による — [arXiv 2405.15116](https://arxiv.org/abs/2405.15116)
- 定理 2（実現不能・有限標本）：最良の f_s でも距離 ε が残り，弱いラベル付きの n 標本で経験損失を最小化する場合，同じ不等式に O(√ε) + O((C_{F_s}/n)^{1/4}) + O((log(1/δ)/n)^{1/4}) の誤差項が付く（値域が有界という仮定の下）— [arXiv 2405.15116](https://arxiv.org/abs/2405.15116)
- 仮定の列挙：(a) F_s が凸（最終層の線形の頭なら満たす），(b) 2 乗損失，(c) 生徒は F_s 上の厳密な最小化を行う，(d) 定理 1 は実現可能性，(e) 定理 2 は有界な値域．f* と f_w には何も仮定しない．
- 実験：人工データ（h* は 5 層 MLP R^8→R^16，F_s は R^16→R の線形関数，P = N(0, 500²I)，弱 2 層・強 8 層，M = 100 課題，N_f = 2000）で「the gain is more or less exactly the misfit」．MolBERT（強）対 2 層 2 ヘッドの小さな transformer（弱，隠れ次元 8〜96，事前学習 2 epoch）で ESOL・FreeSolv・Lipop の回帰．弱いモデルの真の誤差 − misfit が最小のものを選ぶと真の誤差も最小（Lipop で隠れ次元 96：0.8969 ± 0.0327 と 1.0713 ± 0.0489）．NLP は Feedback Prize（作文の 6 評価軸の回帰，bert-base-uncased 対小型 BERT）— [arXiv 2405.15116](https://arxiv.org/abs/2405.15116)
- 5.3 節：弱と強を入れ替えると図が鏡像になる．表現学習の標本が少ない（T = 5, N_r = 250）と単純な方が良い表現を学ぶので役割が逆転する．「weak と strong は表現の質で決めるべき」— [arXiv 2405.15116](https://arxiv.org/abs/2405.15116)

### 1.4 軸との対応
A1 ○（凸な F_s，実験は凍結した表現の上の線形の頭）／A2 ×／A3 ×（n は誤差項に入るだけ）／A4 ×（出発点も罰則も無い）／A5 ×（生徒は真のラベルを見ない）／A6 ×（P は共通）／A7 ×／A8 ○（下限：得 ≥ misfit．決めるのは生徒が先生に合わせきれない 2 乗距離）／A9 ×／A10 ○（人工・分子 3 組・NLP 2 組）／A11 ×

### 1.5 穴
- 扱っていないこと：分類（「Our work also does not address classification tasks (see also Appendix E)」），非凸な F_s（「Our theorems notably do not apply when the set from which the strong model learns the finetuning task is not convex」），正則化，真のラベルとの併用 — [arXiv 2405.15116](https://arxiv.org/abs/2405.15116)
- 付録 E の自認：分類では「additional regularization terms in the objective (along with other error terms) seem to be necessary」．分類では誤りの種類（random errors vs correlated errors）で得が変わりうるが，回帰では「any misfit is only ever “in the correct direction”」なので区別が要らない — [arXiv 2405.15116](https://arxiv.org/abs/2405.15116)
- 今後の課題（原文）：「it would be interesting to see if similar results could be obtained for more general loss functions」；弱いモデルの ensembling；misfit の大きい弱いモデルを作る問い — [arXiv 2405.15116](https://arxiv.org/abs/2405.15116)

### 1.6 本研究との差の書き方（案）
- 「Charikar ら [NeurIPS 2024] は，凸な関数族（線形の頭を含む）の生徒が 2 乗損失で弱いラベルに合わせるとき，生徒の得が misfit 以上であることを示した（下限）．本研究は分類（UAR）で，ソースの出発点 W_0 に L2 で錨づけた生徒を扱い，得の上限を先生と出発点の 0/1 の食い違い J で書く．Charikar らが付録 E で未解決とした『分類では誤りの種類で得が変わる』点を，J が先生と出発点の正誤の組み合わせで数える．」
- 推論：本研究の錨づけは生徒の集合を W_0 の周りの球 {W : ‖W − W_0‖ ≤ r} に縮めることに当たる．この集合も凸なので射影のピタゴラスの不等式自体は成り立つが，定理 1 の実現可能性（f* がその球に入る）は E_0 が球より大きいと破れ，定理 2 の ε（球から f* までの距離）が残る．つまり Charikar らの枠組みに錨づけを入れると「得 ≥ misfit − O(√ε)」の形になり，ε は E_0 の直せない分に対応する．この対応づけは本ノートの推論で，論文で示されていない．

### 1.7 注意
- 本研究の「得」は UAR の差で，Charikar らの得は 2 乗距離の差である．同じ語で呼ばないほうがよい．
- 付録 B・D は未読．D（非凸）の図 7 の見出しだけ見た．

---

## 2. Dong, Li, Li, Lee, Lei 2025 — Discrepancies are Virtue: Weak-to-Strong Generalization through Lens of Intrinsic Dimension

### 2.1 書誌と読んだ範囲
- ICML 2025，PMLR 267:14079–14113（PMLR 第 267 巻の目次で確認）— [PMLR v267](https://proceedings.mlr.press/v267/)（○）．読んだ本文は arXiv 2502.05075v6（2026-04-19）— [arXiv](https://arxiv.org/abs/2502.05075)（◎）
- 読んだ範囲：1〜5 節（序論と関連研究，問題設定，主結果 3.1〜3.2，実験 4.1〜4.2，限界と今後）と付録 E.2・E.3（分類の実験の設定）．読まなかった範囲：付録 A（関連研究の詳細），B（証明），C（リッジ回帰への拡張，定理 3），D（正準角），E.1．

### 2.2 設定
- 模型：y = f*(x) + z，z ∼ N(0, σ²)，f*(x) = φ*(x)ᵀθ*．弱・強の特徴 φ_w, φ_s は平均 0 の劣ガウスで，共分散 Σ_w, Σ_s は低ランク（内在次元 d_w = rank Σ_w，d_s = rank Σ_s ≪ d）．近似誤差 ρ_s < ρ_w，ρ_s + ρ_w ≪ σ²（分散優位）．先生と生徒の相関は相関次元 d_{s∧w} = ‖V_sᵀV_w‖_F² ∈ [0, min(d_s, d_w)] — [arXiv 2502.05075](https://arxiv.org/abs/2502.05075)
- 先生と生徒：先生は n 本のラベル付き標本でリッジ無し回帰，生徒は N 本のラベル無し標本に先生が付けた疑似ラベルでリッジ無し回帰（正則化 α → 0，過剰パラメータなら勾配法の暗黙の最小ノルム解に当たる．Remark 2）．参照として，強い特徴を n 本のラベルで直接学ぶ強い基準 f_s と，n + N 本の真のラベルで学ぶ天井 f_c — [arXiv 2502.05075](https://arxiv.org/abs/2502.05075)
- 標本数：n > d_w + 1，N > d_s（Assumption 2）．
- 本研究との対応：先生 = 別モダリティ（別の特徴部分空間 V_w）で学んだ分類器，生徒 = ターゲットのモダリティの特徴 V_s の上の線形の頭．出発点 W_0 は無い．強い基準 f_s は「K 本のラベルだけで学ぶ」に相当するが，n > d_w + 1 が要るので少数ショットではない．

### 2.3 示したこと
- 定理 1（ガウスの特徴，厳密な分散）：Var(f_w2s) = σ²/(n − d_w − 1) · [d_{s∧w} + (d_s/N)(d_w − d_{s∧w})]，Bias(f_w2s) ≤ Bias(f_w) + ρ_s ≤ ρ_w(1 + d_w/(n − d_w − 1)) + ρ_s．日常語では「先生のラベル雑音由来の分散は，先生と生徒の特徴が重なる部分（d_{s∧w}）ではそのまま生徒に移り，食い違う部分（d_w − d_{s∧w}）では d_s/N 倍に減る」．偏り（先生の表現の限界 ρ_w）は生徒に引き継がれ，上限しか示されていない — [arXiv 2502.05075](https://arxiv.org/abs/2502.05075)
- 命題 1・系 1：Var(f_w) = σ² d_w/(n − d_w − 1)，Var(f_s) = σ² d_s/(n − d_s − 1)，Var(f_c) = σ² d_s/(N + n)．
- 命題 2：PGR（先生から天井までの差のうち生徒が埋めた割合）と OPR（強い基準の誤差 ÷ 生徒の誤差）の下限．n = d_w + q + 1 として，PGR の下限は q が最小のとき最大（系 2），OPR の下限は n について単峰．d_{s∧w} ≪ min(d_s, d_w) か，d_{s∧w} = 0 で N ≫ max(d_w, d_s) のとき W2S が良い — [arXiv 2502.05075](https://arxiv.org/abs/2502.05075)
- 仮定の列挙：平均 0 の（主結果では同時ガウスの）特徴，厳密な低ランク，リッジ無し，2 乗損失，ラベル雑音がガウス，ρ_s + ρ_w ≪ σ²，n > d_w + 1．
- 実験：人工（d = 20000，d* = 300，d_s = 100，d_w = 200，d_{s∧w} ∈ {10, 50, 90}）で定理の予言（破線）と実測が一致．図 3 は「When the variance is low (not dominant, e.g. σ² = 0.0004 ≈ ρ_s + ρ_w), f_w2s can fail to outperform f_w」．UTKFace の年齢回帰（23,708 枚，強 CLIP ViT-B/32（d_s = 443），弱 ResNet18〜152（d_w = 194, 150, 615, 589），n = 1600，N = 10000，リッジ 10⁻⁶）で d_{s∧w}/d_w が小さいほど PGR・OPR が高く，ラベルに雑音を足すほど相対的な W2S が良くなる．分類は ColoredMNIST（DINOv2-s14 対 ResNet，交差エントロピー，線形プローブ）と SST-2（Electra-base 対 BERT-Tiny〜Medium，全層の微調整，Burns らの confidence-weighted loss）— [arXiv 2502.05075](https://arxiv.org/abs/2502.05075)

### 2.4 軸との対応
A1 ○（凍結特徴の上の線形プローブ．核の体制で全層の微調整にも読み替える Remark 1）／A2 ×（偏りと分散の分解で，切片と向きではない．特徴は平均 0 で切片が無い）／A3 △（n, N が分散の式に陽に入るが，先生の価値をラベル本数に換算する式は無い）／A4 ×（出発点無し．強い基準 f_s は別の学習器で，疑似ラベルと真のラベルの併用は無い）／A5 △（n 本の真のラベルで学ぶ基準との比較 OPR があるが n > d_w + 1 が要る）／A6 ×／A7 ×（序論の「ロゴと設計」の例は別の情報源の特徴だが，モダリティとは言っていない）／A8 ○（分散の厳密な式．得は食い違う部分空間 V_w \ V_s での分散の減少から来る．PGR・OPR は下限）／A9 ×（リッジは 0 に向かう）／A10 ○（人工，UTKFace，ColoredMNIST，SST-2）／A11 ×

### 2.5 穴
- 自認する限界：「a limitation of our analysis is the quantification of the advantage of W2S in bias (see Footnote 6)」．「the mechanism of W2S may differ as the balance between variance and bias shifts」．偏り優位の体制（Ildiz ら 2024；Wu & Sahai 2024）との統一は今後の課題 — [arXiv 2502.05075](https://arxiv.org/abs/2502.05075)
- 扱っていないこと：ドメインのずれ，切片，出発点からの学び直し，少数の真のラベルと疑似ラベルの併用，分類の理論（分類は実験だけ）．

### 2.6 本研究との差の書き方（案）
- 「Dong ら [ICML 2025] は，先生と生徒の特徴の部分空間が食い違うほど，先生のラベル雑音由来の分散が生徒で減ることを示した（分散優位の体制）．本研究の先生は別モダリティで，その誤りはラベル雑音ではなく表現の限界（Dong らの ρ_w に当たる偏り）が主である．Dong らの定理 1 では偏りは生徒に引き継がれる（Bias(f_w2s) ≤ Bias(f_w) + ρ_s）ので，本研究の『得は先生の質（J）で頭打ち』はこの枠組みの偏り優位の側に置ける．」
- 推論：本研究の「同じモダリティでは J ≈ +0.6，別のモダリティでは +7〜9」は，Dong らの d_{s∧w}/d_w（同じモダリティなら大きく，別モダリティなら小さい）と向きが合う．ただし Dong らの量は特徴の部分空間の重なりで，本研究の J は予測の正誤の重なりである．

### 2.7 注意
- 定理 1 の厳密な等式は同時ガウスの特徴に限る．劣ガウスでは定数倍の上限（定理 2）．
- 付録 C の定理 3（リッジ回帰）は未読．正則化を 0 以外に取るときの主張はそこにある．

---

## 3. Xue, Li, Mirzasoleiman 2025 — Representations Shape Weak-to-Strong Generalization: Theoretical Insights and Empirical Predictions

### 3.1 書誌と読んだ範囲
- ICML 2025，PMLR 267:70120–70159（PMLR 第 267 巻の目次と PDF の脚注で確認）— [PMLR v267 xue25d](https://proceedings.mlr.press/v267/xue25d.html)（◎）．arXiv 版は 2502.00620（ダウンロードしたが版は照合していない）．
- 読んだ範囲：1〜6 節（序論，関連研究，3 節の設定と定理 3.8，4 節の benign overfitting（定理 4.1，例 4.2，系 4.3），5 節の指標と実験，結論）．読まなかった範囲：付録 A〜E（証明，例の詳細，実験の詳細，付録 E の表現の定義の議論）．

### 3.2 設定
- 模型：固定の表現 h の上の線形関数 f（「focus on the case where f is a linear functions」）．先生は ñ 本の真のラベルで L2 正則化 β_w のリッジ回帰，生徒は n̂ 本の別の標本に先生が付けたラベルで β_s のリッジ回帰，天井は同じ n̂ 本の真のラベルで学ぶ．損失は 2 乗，誤差 Err(g) = E(g(x) − y)²．主対象は PredGap = E[(f_w2s(h_s(x)) − f_sc(h_s(x)))²]（生徒と天井の予測の差）で，正則化が小さく表現の条件が良ければ Err_w2s ≈ PredGap + Err_sc（Thm B.3）— [PMLR v267 xue25d](https://proceedings.mlr.press/v267/xue25d.html)
- 表現の仮定（定義 3.3，(δ, γ̂, γ̃)-decomposability）：表現を，標本で良く集中する主成分の部分（部分空間 V）と，残り（V⊥）に分け，残りの核が δ の範囲で等方（δ-isotropy），標本間の内積が小さく，母集団の共分散が小さいことを要求する．例 3.4（有界で内在次元が低い表現）と例 3.5（スパイク共分散の劣ガウス）が条件を満たす — [PMLR v267 xue25d](https://proceedings.mlr.press/v267/xue25d.html)
- 本研究との対応：先生 = h_w の上の線形の頭，生徒 = h_s の上の線形の頭．出発点 W_0 は無い．正則化は 0 に向かうリッジ．

### 3.3 示したこと
- 定理 3.8（主結果）：δ_w ≤ β_w = O(1)，δ_s ≤ β_s = O(1) の下で PredGap = ‖P_s (I − P_w) ŷ/√n̂‖² ± o(1)．ここで P_w = (K̂(Π_{V_w}h_w)/n̂)(K̂/n̂ + (β_w + γ̃_w)I)⁻¹，P_s も同様で，主成分の核の張る空間へのスケールした射影．日常語では「先生の主成分では表せず（I − P_w），生徒の主成分では表せる（P_s）ラベルの成分の大きさが，生徒が天井に届かない分をきっかり決める」— [PMLR v267 xue25d](https://proceedings.mlr.press/v267/xue25d.html)
- 定理 4.1（benign overfitting の一般条件）：δ_s = o(γ̂_s)（強い表現の非主成分が高度に等方）で天井がほぼ完全なら，生徒は訓練データで先生の誤りをほぼ完全に写しながら Err_w2s ≤ Err_w − Δ + o(1)．例 4.2 では η = 0.6 で Err_w ≈ 0.4，Err_w2s ≈ (1 − η)² = 0.16．「only errors within the span of the strong model’s principal kernel are overfitted harmfully」— [PMLR v267 xue25d](https://proceedings.mlr.press/v267/xue25d.html)
- 系 5.1・5.2：PredGap ≤ C‖P_s(I − P_w)‖²_op + o(1)（C はラベルの分散）など，ラベル無しで計算できる上限．
- 実験：Exp I（MolBERT 対 150 個の小型 transformer，ESOL・FreeSolv・Lipop，線形回帰），Exp II（ETHICS の Justice・Commonsense，NV-Embed-v2 対 22 個の埋め込みモデル，交差エントロピーの線形分類），Exp III（Qwen-7B 対 28 個の小型 LLM，SciQ・Amazon Polarity・Cosmos QA，全層の微調整，活性化マップを表現の代わりに使う）．指標 ‖P_s(I − P_w)‖_op と Err_w2s の Spearman 相関は図の表示で 0.71〜0.94（例：Exp II の Justice 0.923，Commonsense 0.711）．Cosmos QA で次元 8000 以下のモデルに限ると，表現の次元は Err_w2s と相関せず（−0.065，−0.048）指標は 0.905 — [PMLR v267 xue25d](https://proceedings.mlr.press/v267/xue25d.html)

### 3.4 軸との対応
A1 ○（固定の表現の上の線形 f．実験 III は全層）／A2 ×（理論に切片が無い．実験では定数 1 を付け足す）／A3 ×／A4 ×／A5 ×／A6 ×／A7 ×／A8 ○（厳密な等式．得 = Err_w − Err_sc − PredGap で，先生の誤りのうち生徒の主成分で表せる分だけが引き継がれる．系 5.1・5.2 は上限）／A9 ×（0 へのリッジ）／A10 ○（8 データ組，150 小型 transformer，52 LLM）／A11 ×

### 3.5 穴
- 扱っていないこと：分類の理論（実験 II・III は分類だが定理は 2 乗損失．「While Thm. 3.8 is based on linear regression with MSE loss, Exps. II and III demonstrate that the correlation also holds for classification tasks」），出発点，ドメインのずれ，真のラベルとの併用．
- 自認：全層の微調整では表現の定義が発見的（「This heuristic serves primarily as a proof of concept」，「More principled definitions of representations ... could be explored in future work」）— [PMLR v267 xue25d](https://proceedings.mlr.press/v267/xue25d.html)
- Charikar らへの評価（原文）：「assumes high-quality representations in the strong model and does not address the role of the weak model's representations」．

### 3.6 本研究との差の書き方（案）
- 「Xue ら [ICML 2025] は，先生の主成分で表せず生徒の主成分で表せるラベル成分の大きさが，生徒が天井に届かない分を決めることを示し，ラベル無しの指標 ‖P_s(I − P_w)‖_op を提案した．比べる相手は先生の表現と生徒の表現である．本研究の J は先生の予測と出発点 W_0 の予測の正誤を数え，生徒の表現ではなく出発点と比べる．Xue らの等式は 2 乗損失で，本研究は UAR である．」
- 推論：Xue らの核の表現 P_w, P_s は n̂ × n̂ の行列で，学んだ関数が疑似ラベルを付けた n̂ 点の核の張る空間に入ること（表現定理）を使っている．本研究の命題 1 は同じ事実を W − W_0 に当てはめたもので，Xue らの枠組みでは n̂ が本研究の 4K に当たる．

### 3.7 注意
- Spearman の値は図の表示（SC=…）から読んだ．図の (a)(b)(c) と数値の対応は PDF の文字抽出で崩れうるので，論文で引用するときは原図で確かめる．
- 付録 B.3（Err_w2s ≈ PredGap + Err_sc の条件）は未読．

---

## 4. Kumar, Ma, Liang 2020 — Understanding Self-Training for Gradual Domain Adaptation

### 4.1 書誌と読んだ範囲
- ICML 2020，PMLR 119:5468–5479（検索結果の ML Anthology の BibTeX による．○）— [ICML 2020 virtual](https://icml.cc/virtual/2020/poster/6815)．読んだ本文は arXiv 2002.11361v1（2020-02-26）— [arXiv](https://arxiv.org/abs/2002.11361)（◎）．ICML 版は実験に Cover Type が足されている（△，検索結果）．
- 読んだ範囲：1〜6 節（設定，マージンの設定の理論 3.1〜3.5，ガウスの設定の理論 4.1〜4.2，実験 5.1〜5.3，関連研究）．読まなかった範囲：付録 A〜C（証明と実験の詳細）．

### 4.2 設定
- 模型：2 値分類，線形モデル Θ_R = {(w, b) : ‖w‖₂ ≤ R}，ramp 損失 r(m) = min(max(1 − m, 0), 1)．ドメイン P_0（ソース，ラベル有り n_0 本），P_1, …, P_T（ラベル無し n 本ずつ）．ずれはクラス条件付き分布の W∞（Wasserstein-∞）距離で ρ(P_t, P_{t+1}) ≤ ρ < 1/R（幾何マージン 1/R より小さい）．α*-分離（各ドメインに ramp 損失 α* 以下の線形分類器がある），ラベルの比率は不変，E‖X‖² ≤ B² — [arXiv 2002.11361](https://arxiv.org/abs/2002.11361)
- 先生と生徒：先生は直前の分類器 θ，生徒は θ が付けた硬い疑似ラベル sign(M_θ(x)) に合わせて学ぶ同じ線形モデル（自己訓練）．真のラベルはソースだけ．
- ガウスの設定（4 節）：P_t(X | Y = y) = N(yμ_t, σ_t²I)，‖μ_{t+1} − μ_t‖ ≤ B/4，‖μ_t‖ ≥ B，ラベル無しデータは無限．w_t = argmin U(w, P_t) s.t. ‖w‖ ≤ 1，‖w − w_{t−1}‖ ≤ 1/2．
- 本研究との対応：先生 = 出発点 W_0（ソースの分類器）そのもの．本研究の弱いラベルは先生がターゲットの全発話に付けたラベルで，これは Kumar らの「ターゲットに直接自己訓練」（T = 1）に当たる．ガウスの設定の制約 ‖w − w_{t−1}‖ ≤ 1/2 は前の分類器への錨づけ（罰則ではなく球の制約）である．

### 4.3 示したこと
- 定理 3.2（1 段）：ρ(P, Q) = ρ < 1/R，ラベル比率が同じ，有界なら，Q のラベル無し n 本で自己訓練した θ′ は，確率 1 − δ で L_r(θ′, Q) ≤ 2/(1 − ρR) · L_r(θ, P) + α* + (4BR + 2√(log(2/δ)))/√n（式 (12)．係数の正確な形は原文を参照）．日常語では「前の分類器の誤差が小さく，ずれがマージンより小さければ，疑似ラベルで学び直しても誤差は定数倍（2/(1 − ρR)）以上には増えない」．系 3.3：T 段で β^{T+1}(α_0 + O(1/√n))，β = 2/(1 − ρR)．例 3.4 でこの指数的な増え方が達成されうる（厳密）— [arXiv 2002.11361](https://arxiv.org/abs/2002.11361)
- 例 3.5〜3.7（必要な要素）：正則化が無いと「there is no incentive for the model to change when self-training」（元のモデルをスケールしたものが最適解）．軟らかいラベルの対数損失では θ 自身が最小化子なので更新されない（例 3.6）．ヒンジ損失ではずれの後の誤差を抑えられない（例 3.7）— [arXiv 2002.11361](https://arxiv.org/abs/2002.11361)
- 命題 3.8：ずれが無いときは T 段で損失が α_0(T + 1) まで（線形に）増えうる．付録 A では自己訓練で α が 2α になる例がある．
- 定理 4.1（ガウスの設定）：‖w_0 − w*(μ_0)‖ ≤ 1/4 なら w_T = w*(μ_T)（ベイズ最適）を回復する．条件は「良い出発点，小さいずれ，無限のラベル無しデータ」— [arXiv 2002.11361](https://arxiv.org/abs/2002.11361)
- 実験：表 1（Source／Target ST／All ST／Gradual ST）：ガウス 47.7／49.6／92.5／98.8，回転 MNIST 31.9／33.0／38.0／87.9，Portraits 75.3／76.9／78.9／83.8．正則化を外すと Portraits 84→77，回転 MNIST 88→46（序論の記述）．ずれを「回転した画像の割合を増やす」形にすると W∞ が大きく，段階的な自己訓練 33.5 ± 1.5 と直接 33.0 ± 2.2 で差が無い — [arXiv 2002.11361](https://arxiv.org/abs/2002.11361)

### 4.4 軸との対応
A1 ○（理論は線形モデル．実験は MNIST・Portraits で CNN，ガウスでロジスティック回帰）／A2 ×（ガウスの設定は平均 0 に動かして切片を消す）／A3 ×／A4 △（出発点はソースの分類器 θ_0．ガウスの設定の ‖w − w_{t−1}‖ ≤ 1/2 は前の分類器への球の制約）／A5 ×（ターゲットのラベル無し）／A6 ○（段階的なずれ，W∞）／A7 ×／A8 ○（疑似ラベルの後の誤差の上限．得の量ではなく「悪化しない条件」．悪化は指数的になりうることも示す）／A9 △（A4 と同じ．L2 罰則ではなく球の制約で，罰則の強さに当たる量は 1/2 に固定）／A10 ○（ガウス，回転 MNIST，Portraits）／A11 ×

### 4.5 穴
- 扱っていないこと：ターゲットの少数ラベルとの併用，別モダリティの先生，得の量の定量（上限は「先生の誤差の定数倍」で，先生より良くなるかは言わない），自己訓練が制約付き最小化に収束するか（「we leave this optimization analysis to future work」）— [arXiv 2002.11361](https://arxiv.org/abs/2002.11361)
- 自認：「if we want sub-exponential bounds we either need to make additional assumptions on the data distributions, or devise alternative algorithms」．

### 4.6 本研究との差の書き方（案）
- 「Kumar ら [ICML 2020] は，ソースの線形分類器を出発点に，ずれがマージンより小さいターゲットで疑似ラベルを使って学び直すと誤差が定数倍以上に増えないことを示した（ラベルは使わない）．本研究の弱いラベルは，先生が別モダリティである点と，K 本の真のラベルと併用する点が異なる．Kumar らの『正則化が無いと自己訓練は動かない』は，本研究で先生が出発点と同じ（同じモダリティ）だと得が小さい（J ≈ +0.6）ことの先例である．」
- 推論：同じモダリティの先生が出発点 W_0 と同じなら，Kumar らの例 3.5・3.6 のとおり疑似ラベルは W_0 を動かす動機を与えない．本研究で同じモダリティの J が小さいのは，先生と出発点がほぼ同じ分類器だからだと読める．

### 4.7 注意
- 定理 3.2 の第 3 項の係数は pdftotext の抽出で崩れており，上の式は概形である．引用するなら原文の式 (12)（arXiv v1 の 7 頁）を見る．
- ICML 版（PMLR 119）の本文は読んでいない．arXiv v1 と実験が一部違う（△）．

---

## 5. Hacohen, Dekel, Weinshall 2022 — Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets

### 5.1 書誌と読んだ範囲
- ICML 2022，PMLR 162（PDF の脚注で確認）．arXiv 2202.02794v4（2022-06-16）— [arXiv](https://arxiv.org/abs/2202.02794)（◎）
- 読んだ範囲：1〜5 節（序論，理論 2.1〜2.4，手法 3，実験 4.1〜4.3，まとめ），付録 C.1（定理 4 の証明）．読まなかった範囲：付録 A（関連研究），B（定理 2・3 の証明），C.2（1-NN），D〜G（深層モデルの誤差曲線，実装，追加の結果）．

### 5.2 設定
- 模型（理論）：入力空間を 2 つの領域 R_1, R_2 に分け，それぞれを独立な学習器が学ぶ混合．領域 R_1 の確率 p．誤差の期待値 E_D(m) = p E(m_1) + (1 − p) E(αm_2) で，R_1 ≺ R_2（R_1 の方が少ない例で学べる，α < p/(1 − p)）．偏った抽出 m_1 = pm + Δ — [arXiv 2202.02794](https://arxiv.org/abs/2202.02794)
- 線形分類器の具体化（2.4.1）：最小 2 乗の分離器 ŵ = yXᵀ(XXᵀ)⁻¹ = μ̂Σ̂⁻¹（μ̂ は {y_i x_i} の標本平均，Σ̂ は標本共分散）．仮定は ‖x_i‖ ≤ β，XXᵀ の最小固有値が Λ_1 以上，マージン δ で分離可能，共分散がフルランク．
- 先生と生徒：無い．能動学習で L_0 = ∅ から B 本を選ぶ．
- 本研究との対応：予算 m = K × クラス数．本研究の命題 4 の「ラベル無しの選び方」は Hacohen らの TypiClust（ラベルを見ずに密度と多様性で選ぶ）を含む．出発点 W_0 は無い（教師あり学習は 0 から，または自己教師の埋め込みの上の線形分類器）．

### 5.3 示したこと
- 定理 1（閾値検定）：E′(pm)/E′(α(1 − p)m) が α(1 − p)/p より大きければ R_1 を，小さければ R_2 を多めに取ると誤差が減る．定理 2：E が適切（単調減少，E(0) = c_0，E → 0）で，極限が存在し，−log E(x) ∈ ω(log x) なら相転移（undulating）がある．系 1：E(x) ≤ ke^{−νx} なら成り立つ．定理 3・系 2：指数関数なら転移は 1 回（SP-undulating）— [arXiv 2202.02794](https://arxiv.org/abs/2202.02794)
- 定理 4（線形分類器）：上の仮定の下で E[0-1 損失] ≤ ke^{−νm}．証明は μ̂ と Σ̂ の集中（Hoeffding と行列 Bernstein）と，補題 6「Σ̂⁺Σ̂ は訓練集合 {x_i} の張る部分空間への射影」による．定理 5：2 つの線形分類器の混合は undulating — [arXiv 2202.02794](https://arxiv.org/abs/2202.02794)
- 1-NN（2.4.2）：最も確率の高い点の集合を R_1 とすると R_1 ≺ 補集合．発見的な指針は「密度の最大化」と「多様性」．
- 手法 TypiClust：自己教師の埋め込みで K = 20 近傍の平均距離の逆数を典型度とし，B 個に集団化して各集団の最も典型的な点を選ぶ．
- 実験：予算は 1 クラスあたり 1〜10 本（B = M または 5M）．CIFAR-10/100，TinyImageNet，ImageNet-50/100/200．教師ありでは他の能動学習の基準は「on par with random selection or worse」，TypiClust は大きく上回る．半教師あり（FlexMatch）で CIFAR-10 の 10 本から 93.2%（ランダムより 39.4 点高い）．削除実験：全ラベルで学んだ oracle のマージンで選んでも「even a reliable measure of uncertainty leads to poor performance in the low-budget regime, even worse than the baseline uncertainty-based methods」（|L_0| = 10，B = 10，100 モデルの平均）— [arXiv 2202.02794](https://arxiv.org/abs/2202.02794)

### 5.4 軸との対応
A1 △（理論の具体化は最小 2 乗の線形分類器．実験は深層モデルと自己教師の埋め込みの上の線形分類器）／A2 ×／A3 ×／A4 ×／A5 ○（1 クラスあたり 1〜10 本）／A6 ×／A7 ×／A8 ×（半教師ありを使うが理論は無い）／A9 ×／A10 ○（6 データ組，複数の枠組み）／A11 ○（少数予算ではランダムが強く典型的な点が良いことを，混合の模型と実験で示す．予言と実測の照合は図 1a と 1b の定性的な一致にとどまる）

### 5.5 穴
- 扱っていないこと：出発点のある転移（能動学習は 0 からか自己教師の埋め込みから），向きと切片の区別，ラベル無しの選び方がどの誤差（位置か向きか）を減らすかの理論．理論の模型は「領域ごとの独立な学習器」で，線形分類器の向きを学ぶ過程とは別の抽象化である．
- 自認：「Determining the range of training sizes with “low budget” characteristics is a challenging problem, which we leave for future work」— [arXiv 2202.02794](https://arxiv.org/abs/2202.02794)
- 定理 4 は分離可能（実現可能）な場合に限る．

### 5.6 本研究との差の書き方（案）
- 「Hacohen ら [ICML 2022] は，予算が少ないときは典型的な点を選ぶ方が良く，不確かさで選ぶ方法はランダムに劣ることを，領域ごとに独立な学習器の混合の模型と画像の実験で示した．本研究は強い出発点 W_0 の上で 9 通りのラベル無しの選び方を比べ，どれも向きの誤り E_0 を捕らえる割合がランダムの 0.5〜1.2 倍であることを示す．Hacohen らの模型は向きと切片を区別しないので，本研究の『ラベル無しの設計が直せるのは位置で向きではない』という整理は Hacohen らに無い．」
- 推論：Hacohen らの「典型的な点が良い」は，本研究の言葉では切片（位置）の推定の分散を下げる選び方と読める（前回のノートと同じ推論．本文に向きと切片の区別は無い）．また oracle のマージンで選んでも少数予算で悪いという削除実験は，ラベルを知っていても少数予算では不確かな点を選ぶのが良くないことを示しており，本研究の命題 4（ラベルを見ない設計の限界）より強い主張である．ただし Hacohen らの oracle は深層モデルを 0 から学ぶ設定で，本研究の出発点のある線形の頭とは違う．

### 5.7 注意
- 付録 D.2（深層モデルの誤差曲線が指数で抑えられる実証）は未読．
- 93.2% は FlexMatch との組み合わせの値で，TypiClust 単独の値ではない．

---

## 6. Chaloner 1984 と Chaloner & Verdinelli 1995 — ガウスの線形模型のベイズ最適設計

### 6.1 書誌と読んだ範囲
- Chaloner, K. (1984). Optimal Bayesian experimental design for linear models. Annals of Statistics 12(1):283–300. DOI 10.1214/aos/1176346407（Project Euclid の記録．検索結果で確認，△．Project Euclid は本文のダウンロードを拒否した）— [Project Euclid](https://projecteuclid.org/journals/annals-of-statistics/volume-12/issue-1/Optimal-Bayesian-Experimental-Design-for-Linear-Models/10.1214/aos/1176346407.short)．本文として読んだのは同じ題名のミネソタ大学の技術報告 No. 424（1983 年 8 月，34 頁）— [UMN Conservancy](https://conservancy.umn.edu/items/119079a4-908c-4f3a-af84-b9fb633221df)（◎ 要約，1 節，2 節の冒頭）．技術報告と 1984 年の論文の本文が同じかは要確認．
- Chaloner, K. and Verdinelli, I. (1995). Bayesian experimental design: A review. Statistical Science 10(3):273–304. DOI 10.1214/ss/1177009939（検索結果，○；JSTOR の安定 URL 2246015 の走査版の冒頭も見たがフォントが崩れて読めない）— [Project Euclid](https://projecteuclid.org/euclid.ss/1177009939)．本文として読んだのはアイオワ大学の頁にある著者版（69 頁）— [uiowa](https://homepage.cs.uiowa.edu/~gwoodwor/AdvancedDesign/Chaloner%20Verdinelli.pdf)（◎ 1.3〜1.5 節，2.1〜2.2 節，2.5 節，5.5 節）．節の番号は著者版のもので，雑誌版と同じかは要確認．

### 6.2 設定
- Chaloner 1983/84：y = Xᵀθ + ε，ε ∼ N(0, τ⁻¹I)．θ | τ の事前分布は正規で精度行列 τR（R は既知の正定値行列），τ は既知でも未知（ガンマ）でもよい．事後分布は正規で，平均 (R + XXᵀ)⁻¹(Xy + Rθ_0)，精度 τ(R + XXᵀ)．cᵀθ の推定の preposterior risk は cᵀ(R + XXᵀ)⁻¹c E(τ⁻¹)．関心のある c に確率測度 μ を置き，Ψ = E_μ(ccᵀ) として tr Ψ(R + XXᵀ)⁻¹ を最小にする設計（Ψ-optimality，Bayes A-optimality，Duncan & DeGroot 1976）— [UMN TR 424](https://conservancy.umn.edu/items/119079a4-908c-4f3a-af84-b9fb633221df)
- C&V 1995 の 2.1 節：y | θ, σ² ∼ N(Xθ, σ²I)，θ | σ² ∼ N(θ_0, σ²R⁻¹)．事後分布は N((nM(η) + R)⁻¹(Xᵀy + Rθ_0), σ²(nM(η) + R)⁻¹)．「D(η) is a function of the design η and of the prior precision matrix σ⁻²R」— [uiowa](https://homepage.cs.uiowa.edu/~gwoodwor/AdvancedDesign/Chaloner%20Verdinelli.pdf)
- 本研究との対応：θ_0 = W_0 の 1 行，R = I/C′（等方な事前分布），X = ラベルを付けた発話の特徴，n = 4K．事後平均は W_0 に L2 で錨づけた最小 2 乗解そのものである．

### 6.3 示したこと
- 事後共分散 σ²(nM(η) + R)⁻¹ は設計 η と事前精度 R だけの関数で，観測 y を含まない（2.1 節）．Shannon 情報の期待利得（U_1）は det(nM + R) の最大化（Bayes D-optimality），2 乗損失の期待効用（U_2）は tr A(nM + R)⁻¹ の最小化（Bayes A-optimality）に帰着する（2.2 節）— [uiowa](https://homepage.cs.uiowa.edu/~gwoodwor/AdvancedDesign/Chaloner%20Verdinelli.pdf)
- 5.5 節（逐次設計）：「In most linear design problems, however, both Bayesian and non-Bayesian, the optimal sequential procedure is the fixed, non-sequential procedure. There is nothing to be gained by designing sequentially. This is easily seen when the error variance σ² is known: the posterior utility depends on the design η, but does not depend on the data y.」σ² が未知で共役事前分布なら A-optimality については同じ（2.5 節：積分が「does not depend on y」）．「For nonlinear problems the posterior utility clearly depends on the data y ... and there should be a gain from choosing design points sequentially.」— [uiowa](https://homepage.cs.uiowa.edu/~gwoodwor/AdvancedDesign/Chaloner%20Verdinelli.pdf)
- 事前分布とデータの換算（2.2 節）：固定の設計 X_0ᵀX_0 を足す非ベイズの問題（det(nM + X_0ᵀX_0) の最大化）は「algebraically identical to Bayesian D-optimality」．Chaloner 1983 の 1 節も，X_0X_0ᵀ がフルランクなら cᵀ(X_0X_0ᵀ + XXᵀ)⁻¹c の最小化が cᵀ(R + XXᵀ)⁻¹c の最小化と同じ問題だと述べる（Silvey 1969）．つまり事前精度 R は過去の観測 X_0 と同じ役割を持つ — [uiowa](https://homepage.cs.uiowa.edu/~gwoodwor/AdvancedDesign/Chaloner%20Verdinelli.pdf)；[UMN TR 424](https://conservancy.umn.edu/items/119079a4-908c-4f3a-af84-b9fb633221df)
- 標本数への依存：「A characteristic of optimal Bayesian design measures is the dependence on the sample size n, since D(η) = n⁻¹(M(η) + n⁻¹R)⁻¹」．n が大きければ非ベイズの設計と差が無く，「if n is small the prior distribution will have more of an effect on the posterior distribution and on the design」— [uiowa](https://homepage.cs.uiowa.edu/~gwoodwor/AdvancedDesign/Chaloner%20Verdinelli.pdf)
- Chaloner 1983/84 の寄与（要約と 1 節末）：Ψ-optimality への一般同値定理，最適設計の台の点の数の上界，c-optimality の幾何的解釈と Elfving の定理の対応物，整数設計への丸め，多項式回帰の予測の最適設計 — [UMN TR 424](https://conservancy.umn.edu/items/119079a4-908c-4f3a-af84-b9fb633221df)

### 6.4 軸との対応
A1 ○（正規線形模型）／A2 ×（切片は x の座標の 1 つとして扱う．x = (1, x, …, x^p)ᵀ）／A3 △（事前精度 R が過去の観測 X_0 と代数的に同じ役割を持つ．ソースの価値をラベル本数で測る式は無い）／A4 △（事前平均 θ_0 への収縮は出発点からの学び直しに当たるが，転移とは呼ばない）／A5 △（n が小さいと事前分布が設計に効く．少数ショットの枠組みではない）／A6 ×／A7 ×／A8 ×／A9 ○（事後平均 = θ_0 への L2 錨づけの最小 2 乗解．ただし当時の語は「事前分布」）／A10 ×（例題のみ）／A11 ○（最適設計の理論．予言と実測の照合は無い）

### 6.5 穴
- 扱っていないこと：分類（ロジスティック回帰は C&V の 4〜5 節の「非線形」の側で，期待効用が閉じた形にならず近似が要る．設計は θ の事前分布に依存する），「等方な事前分布ならどの設計もランダムと同じ」という主張（下の推論を見よ），実データでの検証．
- 自認（C&V 1.1 節）：「applications to actual experiments still lag behind the theory」，「there are no true “case studies” that we know of」— [uiowa](https://homepage.cs.uiowa.edu/~gwoodwor/AdvancedDesign/Chaloner%20Verdinelli.pdf)

### 6.6 本研究との差の書き方（案）
- 「ガウスの線形模型では事後共分散が設計と事前精度だけで決まり観測を含まないこと，したがって逐次設計に得が無いことは古典的である（Chaloner 1984；Chaloner & Verdinelli 1995, §2.1, §5.5）．本研究の命題 4 はこの事実を，ソースの出発点 W_0 を事前平均，等方な精度 I/C′ を事前精度とする転移の設定に当てはめ，『ラベル無しの選び方が向きの誤り E_0 を捕らえる割合は等方な事前分布の下でランダムと変わらない』と言い直したものである．」
- 推論（重要）：古典の理論が言うのは「設計は y に依らない」であって「設計はどれも同じ」ではない．A-optimal な設計は tr(nM + R)⁻¹ を通して事後分散を下げるので，散らばりの大きい点を選ぶ設計はランダムより事後分散を小さくできる．本研究の「ランダムと同じ」は，E_0 が等方なら選んだ点の張る空間（階数 4K）への E_0 の射影の期待割合が 4K/q で設計に依らない，という向きについての主張に限って成り立つ（本ノートの推論．原文にこの形の記述は無い）．論文では「事後分散が設計に依らない」とは書かず，「E_0 の向きの情報を設計が持てない」と書く．
- 推論：本研究の主分類器はロジスティック回帰なので，命題 4 はガウスの尤度での近似である．C&V はロジスティック回帰を非線形の側に置き，設計が事前分布に依存すると述べているので，論文では「ガウスの線形模型の事実をロジスティック回帰の近似として使う」と断る．

### 6.7 注意
- 1984 年の雑誌版の本文は未読．読んだのは 1983 年の技術報告で，Project Euclid の抄録（trace 表式の最小化，多項式回帰の予測の設計）と要約が一致するが，同一の本文かは要確認．
- C&V の著者版は 69 頁で雑誌版（32 頁）と節の番号が同じかは要確認．

---

## 7. Sloman, Oppenheimer, Broomell, Shalizi 2022 — Characterizing the robustness of Bayesian adaptive experimental designs to active learning bias

### 7.1 書誌と読んだ範囲
- arXiv 2205.13698v2（2022-11-28）— [arXiv](https://arxiv.org/abs/2205.13698)（◎）．arXiv のコメントに「Submitted to NeurIPS 2022」とある（△，検索結果）．掲載先（査読誌・会議）は確認できなかった（要確認）．
- 読んだ範囲：1〜4 節（能動学習の偏り，目的，記法，ベイズ適応設計），6 節の冒頭と 6.1 節（ベイズ線形回帰），8 節（議論と今後）．読まなかった範囲：5 節（誤特定の度合い D_model の定義），6.2〜6.4（結果 1〜3），7 節（非線形の選好学習），付録．

### 7.2 設定
- 模型（6.1 節）：y ∼ N(βφ_k(x)ᵀ, σ²)，φ_k は次数 k の多項式展開，σ² は固定，β ∼ N(M_0, S_0)．観測ごとに S_t = (S_{t−1}⁻¹ + φ_k(x)ᵀφ_k(x)/σ²)⁻¹，M_t = S_t(S_{t−1}⁻¹M_{t−1} + φ_k(x)ᵀy/σ²)．効用（式 12）は次の観測の予測分散の対数で，最大化は φ_k(x)ᵀS_tφ_k(x) の最大化（Bayes D-optimality）と同値 — [arXiv 2205.13698](https://arxiv.org/abs/2205.13698)
- 真の模型 f と仮説の模型族 m(x, θ) を分け，f がどの θ でも表せないとき誤特定と呼ぶ．能動学習の偏り（ALB）は，適応的に選んだ x で学んだ推定量のリスクがリスク最小化子 θ* より大きいこと（n → ∞ の近似）．
- 本研究との対応：M_0 = W_0，S_0 = C′I．φ_k(x) = ラベルを付けた発話の特徴．

### 7.3 示したこと
- 6.1 節（主張，証明は式 (10) の観察）：「It turns out that the optimal design does not depend on the history of observed y values y_t. This can be seen by noticing that both φ_k(x) and S_t are independent of y_{t−1} for all t (see Equation 10). In other words, the experimenter does not need to actually run the experiment to construct an optimal sequence of samples (Krause & Guestrin, 2007; MacKay, 1992). In fact, identifying the optimal design after each sequential observation yields (asymptotically) the batch D-optimal design (Wynn, 1970).」帰結として「the distribution of designs will not depend on the extent of model misspecification, or indeed, whether the model is misspecified at all」— [arXiv 2205.13698](https://arxiv.org/abs/2205.13698)
- 設計は仮定した σ² には依存する（S_t を通して）．
- 結果 1〜3（6.2〜6.4 節，模擬実験）：ALB は誤特定があるときに起き，誤特定の度合いが大きいほど ALB が大きく，仮定する観測雑音を大きくすると ALB が小さくなる．7 節で非線形の選好学習でも再現（要旨と 6 節冒頭の記述から．本文未読）— [arXiv 2205.13698](https://arxiv.org/abs/2205.13698)
- 仮定の列挙：ガウスの尤度と事前分布，σ² 既知，多項式の特徴，D-optimal の効用．

### 7.4 軸との対応
A1 ○（6 節はベイズ線形回帰）／A2 ×／A3 ×／A4 ×／A5 ×／A6 △（誤特定と標本の偏りを共変量シフトと見なす議論はあるがドメイン転移ではない）／A7 ×／A8 ×／A9 △（事前分布 N(M_0, S_0) で M_0 への収縮）／A10 △（模擬実験のみ）／A11 ○（能動学習の偏り．線形・ガウスでは設計が y に依らないことの明示．予言と実測の照合は模擬実験）

### 7.5 穴
- 自認：「Our conclusions in this paper hinge on the results of simulation experiments. In the future, theoretical results should delineate the conditions under which one should expect active learning bias」．線形模型の「idiosyncracies ... (e.g., the equivalence between the batch and adaptive experimental design solutions)」が一般化を制限する — [arXiv 2205.13698](https://arxiv.org/abs/2205.13698)
- 扱っていないこと：分類，転移，少数ショット，向きと切片の区別．

### 7.6 本研究との差の書き方（案）
- 「Sloman ら [arXiv 2022] は，ガウスの事前分布を持つベイズ線形回帰では最適な次の設計点が観測した y の履歴に依らず，したがって設計の分布が模型の誤特定に依らないことを述べ，誤特定があると能動学習に偏りが生じることを模擬実験で示した．本研究はこの『y に依らない』性質を出発点のある転移に当てはめ，9 通りのラベル無しの選び方が向きの誤りを捕らえる割合を実測した．」
- 推論：Sloman らの ALB（偏った設計で学ぶと誤特定の下で推定が偏る）は，本研究で典型的でない点を選ぶ設計が強い出発点の上でランダムを下回る（0.5 倍）ことの説明の候補になる．ただし本研究の実測で誤特定の度合いは測っていないので，説明として書くなら推測と明記する．

### 7.7 注意
- 6.1 節の式 (10)〜(13) の記述は原文のとおりだが，係数の形は pdftotext で崩れているので引用時に原文を見る．
- 掲載先は要確認．

---

## 8. Huang, Du, Xue, Chen, Zhao, Huang 2021 — What Makes Multi-modal Learning Better than Single (Provably)

### 8.1 書誌と読んだ範囲
- NeurIPS 2021（PDF の脚注で確認）．arXiv 2106.04538v2（2021-10-26）— [arXiv](https://arxiv.org/abs/2106.04538)（◎）
- 読んだ範囲：1〜7 節（序論，関連研究，定式化，主結果 4.1〜4.3，実験 5.1〜5.2，議論，結論）．読まなかった範囲：付録 A〜C（証明，例，安定した学習の議論）．

### 8.2 設定
- 模型：x = (x^(1), …, x^(K)) の K モダリティ．真の潜在表現 g*: X → Z と課題写像 h*: Z → Y．モダリティの部分集合 M を使う関数族 G_M（使わないモダリティは ⊥ に置き換える）．N ⊂ M なら G_N ⊂ G_M ⊂ G（Assumption 3）．損失は L-smooth で C で有界（Assumption 1），実現可能（Assumption 2）．ERM で (ĥ_M, ĝ_M) を学ぶ．潜在表現の質 η(g) = inf_h r(h∘g) − r(h*∘g*) — [arXiv 2106.04538](https://arxiv.org/abs/2106.04538)
- 線形の場合（4.3 節）：g(x) = Aᵀx（A は直交列），h(z) = βᵀz（‖β‖ ≤ C_b），y = β*ᵀA*ᵀx + ε，M = [K]，N = [K − 1]，n = d．
- 先生と生徒：無い（全モダリティの同時学習と部分集合の学習の比較）．
- 本研究との対応：本研究の「別モダリティの先生」は，Huang らの枠組みではモダリティ N（先生の側）で学んだ分類器を，モダリティ M \ N（生徒の側）の分類器の学習に使う形で，Huang らの同時学習（融合）とは設定が違う．

### 8.3 示したこと
- 定理 1：r(ĥ_M∘ĝ_M) − r(ĥ_N∘ĝ_N) ≤ γ_S(M, N) + 8L R_m(H∘G_M) + 4C/√m + 2C√(2 ln(2/δ)/m)，γ_S(M, N) = η(ĝ_M) − η(ĝ_N)．日常語では「標本が十分なら，モダリティの組の優劣は学んだ潜在表現の質の差で決まる」— [arXiv 2106.04538](https://arxiv.org/abs/2106.04538)
- 定理 2：η(ĝ_M) ≤ 4L R_m(H∘G_M) + 4L R_m(H∘G) + 6C√(2 ln(2/δ)/m) + L̂(ĥ_M∘ĝ_M, S)（中心化した経験損失）．「原理」：L̂_N − L̂_M ≥ √(C(H∘G_M)/m) − √(C(H∘G_N)/m) ならモダリティを増やす．標本が少ないと複雑さの項が勝ち，多モーダルが単モーダルを下回りうる — [arXiv 2106.04538](https://arxiv.org/abs/2106.04538)
- 命題 1（線形）：上の線形模型で γ_S(M, N) ≤ 0．脚注「Proving that γ_S(M, N) ≤ 0 holds in general is open and will be an interesting future work」— [arXiv 2106.04538](https://arxiv.org/abs/2106.04538)
- 実験（IEMOCAP）：文 100 次元，音声 100 次元，映像 500 次元，6 ラベル（happy, sad, neutral, angry, excited, frustrated），訓練 13200・テスト 3410，符号化は 1 層の線形（隠れ 128），結合して分類．表 1：T 49.93 ± 0.57，TV 51.08 ± 0.66，TA 53.03 ± 0.21，TVA 53.89 ± 0.47．表 3：精度の差 1.15／3.10／0.86／2.81 に対し γ_S が 1.36／3.57／0.19／2.4 で符号が一致．表 2：訓練データの割合を 10⁻⁴〜1 で変えると，少ないとき「TVA ... underperforms the uni-modal」．人工データ（4 モダリティ各 100 次元，重なり w ∈ {0, 0.2, 0.5, 0.8, 1}）では「higher correlation among modalities achieves a lower loss for η」— [arXiv 2106.04538](https://arxiv.org/abs/2106.04538)

### 8.4 軸との対応
A1 △（一般の関数族．命題 1 と IEMOCAP の実験は線形の符号化器）／A2 ×／A3 ×／A4 ×／A5 △（標本の割合を変える実験で，少数標本では多モーダルが劣ると述べる）／A6 ×／A7 ○（多モーダル．相補性の量は潜在表現の質の差 γ_S だけで，モダリティ間の分類器の相補性は定義しない）／A8 ×／A9 ×／A10 ○（IEMOCAP と人工データ）／A11 ×

### 8.5 穴
- 扱っていないこと：先生と生徒（蒸留），ドメインのずれ，少数ショットの転移，分類器レベルの相補性，話者分割（IEMOCAP の分割の詳細は本文に無い．前処理は [35] に従うとだけある）．
- 自認（6 節）：実際の多モーダル学習の性能低下は最適化の問題で，「Our results, on the other hand, mainly focus on the generalization side, which is separated from optimization and assumes that we get the best performance possible in training」— [arXiv 2106.04538](https://arxiv.org/abs/2106.04538)
- 人工データで「重なりが大きいほど η が良い」は，本研究の「別モダリティほど相補性が高く得が大きい」とは問いが違う（Huang らのラベルは全モダリティの和で，重なりが大きいほど少ないモダリティで当てやすい）．

### 8.6 本研究との差の書き方（案）
- 「Huang ら [NeurIPS 2021] は，モダリティを足すと学べる潜在表現の質が上がる（学べる関数族が広がる）ことで多モーダルが単モーダルに勝つことを示し，IEMOCAP で線形の符号化器を使って確かめた．本研究はモダリティを同時に学ぶのではなく，別モダリティで学んだ分類器を弱いラベルの先生として使い，得を先生と出発点の相補性 J で測る．Huang らにはモダリティ間の分類器の相補性を測る量が無い．」

### 8.7 注意
- 表 2 の数値は pdftotext の抽出で列の対応が崩れている．引用するなら原表を見る．
- IEMOCAP の分割が話者で分かれているかは本文で確かめられない．

---

## 9. Lopez-Paz, Bottou, Schölkopf, Vapnik 2016 — Unifying Distillation and Privileged Information（理論の部分）

### 9.1 書誌と読んだ範囲
- ICLR 2016（PDF の脚注「Published as a conference paper at ICLR 2016」で確認）．arXiv 1511.03643v3（2016-02-26）— [arXiv](https://arxiv.org/abs/1511.03643)（◎）
- 読んだ範囲：1〜4 節（序論，蒸留，特権情報，一般化蒸留 4.1〜4.3）と 5 節の人工データの実験 1〜3（偶然読んだ範囲）．読まなかった範囲：5 節の残り（MNIST，CIFAR 等）．実験は G2 が読む．

### 9.2 設定
- 三つ組 {(x_i, x*_i, y_i)}．先生 f_t ∈ F_t を (x*, y) で学び，軟らかいラベル s_i = σ(f_t(x*_i)/T) を作り，生徒 f_s ∈ F_s を (x, y) と (x, s) の混合（λ）で学ぶ．x* = x で |F_s| ≪ |F_t| なら Hinton の蒸留，x* が特権的な記述で |F_t| が小さければ Vapnik の特権情報に帰着する — [arXiv 1511.03643](https://arxiv.org/abs/1511.03643)
- 理論の語り（4.1 節）は「assume」で始まる率の議論で，定理ではない．
- 本研究との対応：本研究の別モダリティの先生は，4.2 節の「if the multiple tasks share the same labels y_i but differ in their input modalities, the input modalities from the source tasks are privileged information」そのものである．ただし本研究の先生はソースで学び，弱いラベルはターゲットのラベル無し発話に付ける（半教師ありの拡張に近い）．

### 9.3 示したこと
- 4.1 節（純粋な蒸留 λ = 1）：生徒が単独で学ぶ率を O(|F_s|_C/√n) + ε_s，先生の率を O(|F_t|_C/n) + ε_t，生徒が先生から学ぶ率を O(|F_s|_C/n^α) + ε_l（1/2 ≤ α ≤ 1）と仮定すると，R(f_s) − R(f) ≤ O((|F_s|_C + |F_t|_C)/n^α) + ε_l + ε_t．先生が役立つ条件は「i) the capacity of the teacher being small, ii) the approximation error of the teacher being smaller than the approximation error of the student, and iii) the coefficient α being greater than 1/2」．「the teacher is most helpful in low data regimes」．「We believe that the “α > 1/2 case” is a general situation, since soft labels ... contain more information than hard labels」— [arXiv 1511.03643](https://arxiv.org/abs/1511.03643)
- 3.1 節（Vapnik & Vashist 2009 の SVM+ との関係）：先生が slack に相当する補正関数を供給すると分離可能な問題の O(n⁻¹) の率に近づく．「a teacher is helpful whenever the capacity of her correcting functions is much smaller than the capacity of the student decision boundary」— [arXiv 1511.03643](https://arxiv.org/abs/1511.03643)
- 4.3 節（因果の見方）：特権情報が x の周辺分布の情報しか足さないなら反因果の問題でしか役立たず，y | x の情報を足すなら因果の問題でも役立つ．
- 人工データの実験 2（読んだ範囲）：特権情報が「きれいな特徴」で生徒の雑音と独立のとき，先生 90 ± 1%，生徒単独 68 ± 1%，蒸留 70 ± 1% で「This improvement is not statistically significant ... there exists no additional information transferable from the teacher to the student」— [arXiv 1511.03643](https://arxiv.org/abs/1511.03643)

### 9.4 軸との対応
A1 △（理論は関数族の容量 |F|_C の一般論．人工実験はロジスティック回帰 d = 50，n = 200）／A2 ×／A3 △（率が n^{−1/2} から n^{−α} に上がるという標本数の議論．換算式は無い）／A4 ×／A5 △（「low data regimes」で最も有効と述べる）／A6 ×（拡張の例として挙げるだけ）／A7 △（別モダリティを特権情報と明示するが量は無い）／A8 ○（蒸留・特権情報の得を率で説明．仮定した率からの上限で，定理ではない）／A9 ×（一般の正則化 Ω(‖f‖)）／A10 ○（人工，MNIST，CIFAR-10 など．G2 が読む）／A11 ×

### 9.5 穴
- 扱っていないこと：得の量を先生と生徒の関係で書くこと（率の次数 α と容量だけ），出発点のある転移，分類の 0/1 の相補性．理論の部分は「assume」で始まり，証明された定理ではない．特権情報の理論としては Pechyony & Vapnik 2010（NeurIPS）を引く — [arXiv 1511.03643](https://arxiv.org/abs/1511.03643)．検索結果によれば Sharoni & Sabato 2023（AISTATS）がその VC 次元の主張に反例を示した（△，未読）— [PMLR v206 sharoni23a](https://proceedings.mlr.press/v206/sharoni23a.html)

### 9.6 本研究との差の書き方（案）
- 「Lopez-Paz ら [ICLR 2016] は，別モダリティで学んだ先生の軟らかいラベルを生徒に写す一般化蒸留を提案し，先生が役立つ条件を学習の率（先生の容量が小さく，近似誤差が生徒より小さく，模倣の率が n^{−1/2} より速い）で説明した．本研究は同じ『別モダリティの先生』の設定で，得の上限を先生と出発点の相補性 J という計れる量で書く．」
- 推論：実験 2（先生の情報が生徒の雑音と独立で得が無い）は，先生が生徒より 22 点良くても得が 2 点という例で，本研究の「得 ≤ J」の J が緩い上限になりうることの具体例として引ける．

### 9.7 注意
- 4.1 節は仮定した率の計算で，本研究の関連研究で「証明した」と書かない．
- Vapnik & Vashist 2009 の書誌：Neural Networks 22(5–6):544–557（検索結果で頁 544–557 は確認，巻・号は要確認）— [LUPI の頁](https://users.sussex.ac.uk/~nq28/lupi/)（△）

---

## 余力の項目：表現定理の原典と特権情報の原典（書誌の確認のみ）
- Kimeldorf, G. and Wahba, G. (1971). Some results on Tchebycheffian spline functions. Journal of Mathematical Analysis and Applications 33(1):82–95．Wahba の公開文献一覧と複数の引用で一致（○，本文未読）— [Wahba publications](https://pages.stat.wisc.edu/~wahba/pubs/bib.html)．表現定理の原典としては同著者の 1970 年の Ann. Math. Statist. の論文（A correspondence between Bayesian estimation on stochastic processes and smoothing by splines）も引かれることがある（△，要確認）．
- Schölkopf, B., Herbrich, R., Smola, A. J. (2001). A generalized representer theorem. COLT/EuroCOLT 2001, LNCS 2111, pp. 416–426, Springer. DOI 10.1007/3-540-44581-1_27（○，MPI の記録）— [MPI 記録](https://is.mpg.de/ei/publications/1838)．主張は，経験損失項と RKHS ノルムの狭義単調な関数の正則化項の和の最小化子が訓練点の核の線形結合で書けること（△，Wikipedia の要約）— [Wikipedia](https://en.wikipedia.org/wiki/Representer_theorem)
- 本研究の命題 1 は，罰則 ‖W − W_0‖² を W − W_0 についての RKHS ノルム（線形核）と見れば，一般化表現定理（Schölkopf 2001）の直接の帰結である．Hacohen 2022 の補題 6（Σ̂⁺Σ̂ が訓練点の張る空間への射影）も同じ事実の別の形である．論文では「新しい定理」と書かず「表現定理から直ちに従う」と書く（前回のノートと同じ判定）．

---

## 表：9 本 × 本研究の軸 A1〜A11

記号：○ 扱う，△ 一部，× 扱わない．各欄の根拠は表の下の軸ごとの一覧にある．

| 論文 | A1 線形・凍結 | A2 切片と向き | A3 本数換算 | A4 合併・出発点 | A5 少数ショット | A6 ずれ | A7 多モーダル | A8 弱いラベル | A9 L2 錨づけ | A10 実データ | A11 能動学習 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 Charikar 2024 | ○ | × | × | × | × | × | × | ○ 下限 | × | ○ | × |
| 2 Dong 2025 | ○ | × | △ | × | △ | × | × | ○ 厳密（分散） | × | ○ | × |
| 3 Xue 2025 | ○ | × | × | × | × | × | × | ○ 厳密（ギャップ） | × | ○ | × |
| 4 Kumar 2020 | ○ | × | × | △ | × | ○ | × | ○ 上限（悪化） | △ | ○ | × |
| 5 Hacohen 2022 | △ | × | × | × | ○ | × | × | × | × | ○ | ○ |
| 6 Chaloner 1984 / C&V 1995 | ○ | × | △ | △ | △ | × | × | × | ○ | × | ○ |
| 7 Sloman 2022 | ○ | × | × | × | × | △ | × | × | △ | △ | ○ |
| 8 Huang 2021 | △ | × | × | × | △ | × | ○ | × | × | ○ | × |
| 9 Lopez-Paz 2016 | △ | × | △ | × | △ | × | △ | ○ 率の議論 | × | ○ | × |

軸ごとの根拠（1 行ずつ）：
- A1：1 凸な F_s（線形の頭）と凍結した h_s．2 凍結特徴の線形プローブ（リッジ無し）．3 固定の h の上の線形 f（核リッジ）．4 ‖w‖ ≤ R の線形モデル（実験は CNN も）．5 理論の具体化は最小 2 乗の線形分類器，実験は深層．6 正規線形模型．7 多項式のベイズ線形回帰．8 一般の関数族，命題 1 と IEMOCAP は線形の符号化器．9 容量の一般論，人工実験はロジスティック回帰．
- A2：どの論文も切片と向きを分けない．2・4 は特徴を平均 0 にして切片を消す．3 は実験で定数 1 を足すだけ．6・7 は切片を x の座標として扱う．
- A3：2 は n, N が分散の式に入るが換算は無い．6 は事前精度 R が過去の観測 X_0 と代数的に同じ（換算の考え方の先例）．9 は率の次数の議論．他は無し．
- A4：4 は出発点 θ_0 からの自己訓練と前の分類器への球の制約．6 は事前平均 θ_0 への収縮．他は出発点無し（1〜3 の生徒は 0 から学ぶ）．
- A5：5 は 1 クラス 1〜10 本の予算．2 は n 本のラベルの基準との比較（n > d_w + 1）．6 は n が小さいと事前分布が効く．8 は標本の割合を変える実験．9 は「low data regimes」で有効と述べる．1・3・4・7 はターゲットの真のラベルを使わない．
- A6：4 だけが扱う（W∞ の段階的なずれ）．7 は誤特定と標本の偏りを共変量シフトとして議論．
- A7：8 が多モーダル（相補性は潜在表現の質の差だけ）．9 は別モダリティを特権情報と明示（量は無い）．他は無し．
- A8：1 得 ≥ misfit（2 乗損失の下限）．2 分散の厳密な式（食い違う部分空間で減る）．3 PredGap の厳密な式（先生の誤りのうち生徒が表せる分が残る）．4 自己訓練後の誤差 ≤ 先生の誤差の定数倍 + α*（悪化しない条件）．9 率の仮定からの上限（定理ではない）．5〜8 は無し．
- A9：6 は事後平均が θ_0 への L2 錨づけの最小 2 乗解．4 は球の制約．7 は事前平均 M_0 への収縮．1〜3 は正則化無しか 0 へのリッジ．
- A10：1 分子 3 組と NLP 2 組．2 UTKFace・ColoredMNIST・SST-2．3 8 データ組・52 LLM．4 ガウス・回転 MNIST・Portraits．5 CIFAR・TinyImageNet・ImageNet の部分集合．6 例題のみ．7 模擬実験．8 IEMOCAP（6 クラス）．9 人工・MNIST・CIFAR 等（G2 が読む）．
- A11：5 少数予算の相転移と TypiClust（照合は定性的）．6 最適設計の古典（設計は y に依らない．照合無し）．7 設計が y の履歴に依らないことと ALB（照合は模擬実験）．他は無し．

---

## この群の先行研究に空いている穴（本研究の命題の置き場所）

### 穴 1：出発点に錨づけた線形の生徒で，弱いラベルと少数の真のラベルを併用する設定が無い
- 1〜3（W2S の理論）の生徒はいずれも 0 から（または 0 へのリッジで）学び，真のラベルを見ない．2 の強い基準 f_s は真のラベル n 本で別に学ぶ学習器で，併用ではない — [arXiv 2405.15116](https://arxiv.org/abs/2405.15116)；[arXiv 2502.05075](https://arxiv.org/abs/2502.05075)；[PMLR v267 xue25d](https://proceedings.mlr.press/v267/xue25d.html)
- 4 は出発点からの自己訓練だが真のラベルを使わず，9 は同じ n 点の真のラベルと軟らかいラベルの混合で，ラベル無しの発話に先生のラベルを付けて足す設定は「半教師ありの拡張」として言及するだけ — [arXiv 2002.11361](https://arxiv.org/abs/2002.11361)；[arXiv 1511.03643](https://arxiv.org/abs/1511.03643)
- 本研究の置き場所：W_0 への L2 錨づけ（6 の事後平均の形）の上に，K 本の真のラベルと先生の弱いラベルを足す．この併用の設定で得の上限を書いた理論はこの群に無い．

### 穴 2：弱いラベルの得の「上限」を，先生と出発点の 0/1 の正誤の組み合わせで書いたものが無い
- 1 は下限（得 ≥ misfit），2 は分散の厳密な式と PGR・OPR の下限，3 は天井との差の厳密な式とラベル無しの上限（表現の射影で書く）．いずれも 2 乗損失で，比べる相手は先生の表現と生徒の表現である．分類での誤りの種類（相関した誤りと無相関の誤り）が得を変えうる点は 1 が付録 E で未解決として挙げる — [arXiv 2405.15116](https://arxiv.org/abs/2405.15116)
- 本研究の置き場所：UAR で，先生の予測と出発点 W_0 の予測の正誤の組み合わせ J で上限を書く．
- 注意（推論）：J = P(先生 ✓, W_0 ✗) − P(先生 ✗, W_0 ✓) は恒等的に P(先生 ✓) − P(W_0 ✓)，つまり先生と出発点の正解率の差に等しい（UAR ならクラスごとに同様）．したがって「得 ≤ J」は「錨づけた生徒は先生の正解率を超えない（弱から強への一般化が起きない）」という主張と同じである．1〜3 はまさに生徒が先生を超える現象（W2S）を扱うので，本研究の上限はその否定の側にある．論文では J を「相補性」と呼ぶより，「先生が直せる割合（第 1 項）」と「先生が壊す割合（第 2 項）」に分けて書き，上限は正解率の差に帰着すると明記した方が，W2S の文献と矛盾しているように読まれない．また本研究の実測（同じモダリティで J ≈ +0.6 に対し得 +1）は上限と整合しないように見えるので，誤差の範囲か，J と得の単位（UAR の点）が同じかを確かめる必要がある．
- W2S の文献との整合（推論）：2 の枠組みでは，先生の誤りのうち生徒が消せるのはラベル雑音由来の分散で，偏り（先生の表現の限界）は引き継がれる．本研究の先生は別モダリティで，その誤りは分散ではなく偏りが主だと考えれば，得が先生の質で頭打ちになる本研究の上限は 2 の「偏り優位では W2S が起きにくい」（図 3）と向きが合う．この対応づけは論文で「先行研究の特別な場合」とは書かず，「設定が偏り優位の側にある」と書く．

### 穴 3：別モダリティの先生の相補性を量で定義した理論が無い
- 8 は融合（同時学習）で，相補性は潜在表現の質の差 γ_S だけ．9 は別モダリティを特権情報と呼ぶが量は無い．2 の相関次元 d_{s∧w} は特徴の部分空間の重なりで，モダリティには言及しない — [arXiv 2106.04538](https://arxiv.org/abs/2106.04538)；[arXiv 1511.03643](https://arxiv.org/abs/1511.03643)；[arXiv 2502.05075](https://arxiv.org/abs/2502.05075)
- 本研究の置き場所：同じモダリティと別モダリティで J を測り（+0.6 と +7〜9），得の差（+1 と +3）と並べる．「相補性が得を決める」こと自体は 1〜3 で既知なので，新しいのは「モダリティ間で J を実測し，得の上限として使った」点に限る．

### 穴 4：ラベル無しの設計が向きを直せないことを，出発点のある線形の頭で言い，実測と照らしたものが無い
- 6・7 は「設計は y に依らない」（ガウスの線形模型）までで，等方な事前分布の下で向きの情報を設計が持てないという形の記述は無く，分類（ロジスティック回帰）は非線形の側で近似が要る — [uiowa](https://homepage.cs.uiowa.edu/~gwoodwor/AdvancedDesign/Chaloner%20Verdinelli.pdf)；[arXiv 2205.13698](https://arxiv.org/abs/2205.13698)
- 5 は少数予算で典型的な点が良いことを示すが，模型は領域ごとの独立な学習器で，向きと切片を区別しない．出発点のある設定でもない — [arXiv 2202.02794](https://arxiv.org/abs/2202.02794)
- 本研究の置き場所：古典の事実を出発点のある転移に当てはめ，9 通りのラベル無しの選び方で E_0 を捕らえる割合（ランダムの 0.5〜1.2 倍）を実測した．ただし「ランダムと同じ」は向きの射影の期待割合についてだけ成り立ち，事後分散（位置）は設計で変わる（穴 4 の推論は 6.6 節）．

### 穴 5：感情認識・クロスコーパスの実データでの検証が無い
- この群で感情認識のデータを使うのは 8 の IEMOCAP（6 クラス，13200 本の訓練）だけで，クロスコーパスの転移でも少数ショットでもない — [arXiv 2106.04538](https://arxiv.org/abs/2106.04538)
- 1〜3 の実データは分子・顔の年齢・NLP，4 は回転 MNIST と Portraits，5 は CIFAR・ImageNet．

### 書き方の注意（この群に共通）
- 「相補性が得を決める」「弱いラベルの理論」「ラベル無しの設計は向きを直せない」を「初めて」と書かない．1〜3，6〜7 に先例がある．
- 1 は下限，本研究は上限なので「同じ結果」と書かない．3 は厳密な等式なので「上限を与えた」とも「下限を与えた」とも書かず，「生徒の表現と先生の表現で書いた」と書く．
- 命題 1 は表現定理の帰結と明記する（Kimeldorf & Wahba 1971；Schölkopf ら 2001）．

---

## 書誌の確認状況と未読の一覧
- 確認した書誌（◎/○）：1 NeurIPS 2024；2 ICML 2025 PMLR 267:14079–14113；3 ICML 2025 PMLR 267:70120–70159；4 ICML 2020 PMLR 119:5468–5479（○）；5 ICML 2022 PMLR 162；8 NeurIPS 2021；9 ICLR 2016．
- 要確認：6 の Chaloner 1984 の雑誌版本文（技術報告 424 と同一か），C&V 1995 の雑誌版の節番号；7 Sloman 2022 の掲載先；Vapnik & Vashist 2009 の巻・号；Kimeldorf & Wahba の 1970 年の論文を表現定理の原典として引くか．
- 未読：1 の付録 B・D；2 の付録 A〜D；3 の付録 A〜E；4 の付録 A〜C と ICML 版本文；5 の付録 A・B・C.2・D〜G；6 の Chaloner 1984 雑誌版，C&V の 3〜4 節・6〜9 節；7 の 5 節・6.2〜6.4・7 節；8 の付録；9 の 5 節の実データの実験．
- 読み方の注意：数式の係数は pdftotext の抽出で崩れている箇所がある（4 の定理 3.2，7 の式 (10)〜(13)，8 の表 2，3 の図の相関値）．論文で引用するときは原文の PDF を見る．
- ダウンロードした PDF と文字化したファイルは /tmp/claude-1000/-home-snics-MER-DA/2f47b0d7-4ef9-4797-9837-84fb3de61717/scratchpad/g6_pdfs/ にある（セッション限り）．
