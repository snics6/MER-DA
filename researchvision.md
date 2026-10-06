# 少数ターゲットラベル下のマルチモーダル感情認識（MER）転移
## 研究方針・理論的裏付け・実験計画（v1.0）

- 作成日：2026-09-29
- 対象期間：2026-09-29 〜 2026-12-28（13週）
- 投稿先・締切（2026-09-29 決定）：測定論文を 2026-10 末、改善論文を 2026-11 末に投稿する。投稿先は結果が良ければ IEEE Transactions on Affective Computing、そうでなければ IEEE Access（中間の候補：Neurocomputing）
- **本文書は下書きである。**研究方針は議論と先行研究をもとにゼロから組み直す（2026-09-29）
- **2026-09-30 に方針を転換した**（0.1 節）。測ったずれとソースの得の関係を調べる実証的な分析を主軸にし、理論からの予言の判定は次に回す

---

## 0. この文書の位置づけと原則

この文書は、今後3ヶ月の研究の「設計図」である。実験を始める前に、何を示したいのか、どの理論のどの量を測るのか、どう判断するのかを固定しておく。

**原則**

1. **問題と手段を分ける。** 問題は1つ（少数ラベルしかない新コーパスへの適応）。DAとメタ学習はその解法候補であり、どちらを採るかはデータで決める。
2. **実験は理論上の量を定量化するものとして設計する。** 各実験に「対応する理論の項」と「事前に書いた仮説」を必ず付ける（第5章・第6章）。
3. **ゼロベースで設計する。** 過去の実験設定は引き継がない。
4. **評価手順とハイパーパラメータ選択規則を実験前に固定する。** ターゲットのテストラベルを選択に使わない。分析用の「オラクル量」と、手法が実際に使える量を明確に区別する。

原則2は、0.1 節の方針転換で弱めた。指標は理論の項と対応づけて選ぶが、分析は探索的でよい。

### 0.1 方針転換（2026-09-30）

**何を変えたか**

- 研究の主軸を、**①の地図と、②で測ったずれと地図の得の関係の実証分析**（`docs/exploratory-analyses.md` の E シリーズ）にする。探索的な分析でよい
- 理論からの予言（6.5.1）の判定と、ずれを消す・足す操作の実験（`docs/experiment-protocol.md` 4 節）は後回しにする
- 測定論文（論文1）の中心の主張を、次の3段にする（3 節）
  1. 少数ラベルのコーパス間 MER では、ソースを混ぜたときの得のばらつきの大部分が、コーパスごとの性質で決まる（E2：K* のばらつきの60%はターゲットで、K=全件の得の48%はソースで決まる。音声、途中の10〜11回）
  2. 残りの組に固有の部分は、測れる距離で説明できる（E2：ターゲットとソースの効果を除くと、入力の距離と得の順位の偏相関が −0.36〜−0.41、K* と −0.47〜−0.61）
  3. その距離を使うと、新しいターゲットに混ぜるソースを選べる（E3・E4：K=5〜25 で、ソースの順位の予想と実際の相関が 0.25〜0.32 から 0.56〜0.58 に上がる。指標の選び方も抜いたターゲットを見ずに決めた値）

**なぜ変えたか**

- 途中の結果で、上の3段の関係が見えた（2026-09-30 の E1〜E4）。これは先行研究（`docs/related-work.md` 3.2 節）が組をそのまま並べて相関をとっているのに対し、コーパスごとの性質を差し引いて初めて見える関係である
- 研究者の判断：「測定・分析論文なので、まずそこを徹底的にやるのが大事。理論からの予想よりは」

**変えないもの**

- 固定する学習器、データの取り決め、実験の手順（`docs/experiment-protocol.md` 1〜3 節）
- 登録済みの予言（6.5・6.5.1・6.5.2）は消さず、書き換えない。判定は後で行う

**次に確かめること**：上の関係が、映像・音声＋映像（反復を10回に増やす）、音声の本番30回、第2エンコーダ（HuBERT）でも出るか。

---

## 1. 問題設定

### 1.1 定式化

- ソースコーパス $S_1,\dots,S_M$：ラベル付き。各 $n_j$ 件。
- ターゲットコーパス $T$：
  - ラベル付き：各クラス $K$ 件（合計 $n_T = KC$、$C$ はクラス数）
  - ラベルなし：任意（使う設定と使わない設定を分ける）
  - テスト：話者独立に固定
- モダリティ $m\in\{a,v,l\}$（音声・映像・テキスト）。各モダリティの表現 $\phi^m$ は凍結済みエンコーダの特徴（キャッシュ済み）。
- 仮説クラス $\mathcal H$：固定表現の上の線形分類器（多クラスロジスティック回帰）。融合は特徴連結（early fusion）を基本とする。
- 目的：ターゲット誤差 $\epsilon_T(h)$ を小さくすること。主評価指標はUAR（クラス平均再現率）。

**なぜ「凍結特徴＋線形ヘッド」から始めるのか**

- DA理論の上界は「固定された仮説クラス」についての主張である。表現を固定し $\mathcal H$ を線形に限れば、理論の各項（ダイバージェンス、$\lambda$、複雑度）と実験量を1対1に対応させられる。
- 線形分類器のVC次元は特徴次元で決まるので、複雑度項の変化も実験で操作できる（PCAで次元を変えるなど）。
- 計算が軽く、ランダムな $K$-shot 抽出を数十回繰り返せる。少数ラベル設定の分散を正しく扱うために必須。
- 深層のfine-tuningへの一般化は第7週で確認する（LP-FTや小規模MLP）。

### 1.2 DAとメタ学習の関係

| 解法族 | 使う情報 | 代表的な理論 |
|---|---|---|
| ソースとターゲットの重み付き混合（半教師ありDA） | ソースのラベル＋ターゲット $K$-shot | Ben-David et al. (2010) Theorem 3 |
| 仮説転移・fine-tuning | ソースで学習したモデル＋ターゲット $K$-shot | Kuzborskij & Orabona (2017)、Kumar et al. (2022) |
| ラベルシフト補正 | ソースモデル＋ターゲットのラベル分布推定 | Lipton et al. (2018)、Alexandari et al. (2020) |
| 多ソースの重み付け | 複数ソース＋ターゲット $K$-shot | Konstantinov & Lampert (2019)、Mansour et al. (2021) |
| メタ学習・表現転移 | 複数ソースをタスクとして使う | Tripuraneni et al. (2020)、Du et al. (2021) |

どれも「ターゲット $K$-shot を与えたときのターゲット誤差」を目的にしている点で同じ問題である。第6章の実験で同じ土俵に並べて比較する。

---

## 2. 問題設定の正当化（3本柱）

### 2.1 理論

- **ターゲットラベルなしでは保証できない。** 共変量シフト＋ラベルなし分布間の距離が小さい、という仮定があっても、ターゲットのラベル付き例がなければDAの成功は保証されない（Ben-David et al., AISTATS 2010）。
- **表現を揃えるだけでは悪化しうる。** ラベル分布がずれていると、表現の分布を揃えるほどソースとターゲットの合計誤差の下界が上がる（Zhao et al., 2019）。一般化ラベルシフトの枠組み（Tachet des Combes et al., 2020）もこの問題への対応である。
- **感情ラベルは「ラベル付け関数」自体がずれる。** アノテータの解釈やラベル定義がコーパスごとに異なるため、$P(Y\mid X)$ がずれる（事後分布ドリフト）。これは上界の $\lambda$ 項に相当し、ターゲットラベルなしには推定も制御もできない。事後分布ドリフト下の転移理論（Cai & Wei, 2021；Reeve et al., 2021）が定式化の候補になる。
- **ターゲットラベルの価値は定量化できる。** ソースとターゲット両方の標本数に対するミニマックスレート（Hanneke & Kpotufe, 2019）が既にある。少数ラベル設定はこの理論の直接の応用先である。

### 2.2 実験（示すべきこと）

1. ラベルなしのまま分布を揃える方法では、コーパス対によって性能が頭打ちになるか、悪化する。
2. ターゲットラベルを $K$ 件足すと改善する。その改善の大きさと、ソースが役立たなくなる $K$ の値は、測定した「ずれ」で説明できる。
3. コーパス間のずれは、入力分布 $P(X)$・ラベル分布 $P(Y)$・条件付き分布 $P(Y\mid X)$ または $P(X\mid Y)$ に分解でき、その内訳はモダリティによって違う。

### 2.3 実用（「MERのDAはどこで要るのか」への答え）

- **想定場面**：コールセンター、臨床面接・メンタルヘルス、車載、対話ロボット、教育、別言語・別文化への展開。
- **変わるもの**：録音・撮影条件、話者層、感情の出させ方（演技／即興／自然）、ラベル体系とアノテータ集団。
- **少数ラベルが現実的な理由**：
  - 導入側が数十〜数百件を自前でアノテーションするのは現実的だが、ゼロ件のまま運用するのは非現実的。性能を検証するだけでもターゲットラベルが要る。
  - 教師なしDAの実験でも、ハイパーパラメータ選択に暗黙にターゲットラベルが使われがち、という批判がある（You et al., ICML 2019）。少数ラベルを明示する方が誠実な設定である。
  - 「何を怒りとみなすか」の定義は導入先が決めるものであり、ラベルでしか伝えられない。

---

## 3. 貢献の型と論文計画

| 型 | 内容 | 位置づけ |
|---|---|---|
| A | 理論に裏付けられた分析：どのコーパス対・モダリティで、何件のターゲットラベルがあればソースは役立つか | 主軸 |
| A+ | Aから導かれる単純な手法（ずれの推定値から混合比 $\alpha$ やモダリティ別の重みを決める） | 論文1 |
| B | 新しいDA／メタ学習手法 | 論文2（Aの知見に依存） |

- **論文1（A+）**：少数ラベルMERにおける「ソースの価値」の定量化と、ずれに応じた混合規則。第8週までにドラフト。
  - **2026-10-01**：先行研究との関係・研究の価値・投稿先で想定される指摘は `docs/paper-positioning.md` にまとめた。論文の新しさと関連研究はそれに合わせて書く
  - **2026-09-30 更新（0.1 節）**：論文1の中心を、(1) 地図、(2) 得のばらつきのコーパスごとの部分と組に固有の部分への分解、(3) 測ったずれによる新しいターゲットのためのソースの選択、の3段にする。ずれに応じた混合規則（A+）は、この3段の後に余力があれば加える
- **論文2（B）**：論文1で見つかった現象（例：アノテーション差が支配的なコーパス対）に特化した手法。第9〜13週。3ヶ月でドラフト完成まで、投稿は延長も可、と正直に見積もる。

---

## 4. 理論候補（実験の裏打ち）

各項目は「主張」→「実験で定量化する量」の順に書く。書誌は末尾の参考文献を参照。

### 4.1 ソースとターゲットの重み付き混合（中核）

**文献**：Ben-David et al. (2010) Theorem 3。

**主張**：ソース誤差とターゲット誤差の重み付き和 $\epsilon_\alpha(h)=\alpha\,\epsilon_T(h)+(1-\alpha)\,\epsilon_S(h)$ を最小化する。ラベル付きデータ $m$ 件のうちターゲットの割合を $\beta$ とすると、定数と高次項を省けば次が成り立つ。

$$
\epsilon_T(\hat h_\alpha)\ \le\ \epsilon_T(h_T^*)\ +\ c_m\sqrt{\frac{\alpha^2}{\beta}+\frac{(1-\alpha)^2}{1-\beta}}\ +\ 2(1-\alpha)\,A,
\qquad A=\tfrac12 d_{\mathcal H\Delta\mathcal H}(\mathcal D_S,\mathcal D_T)+\lambda
$$

ここで $c_m=O\!\left(\sqrt{d\log m/m}\right)$、$d$ は $\mathcal H$ のVC次元である。

**最適な $\alpha$ の性質**（導出は付録A.3）：

- 右辺は $\alpha$ について凸で、$A=0$ なら $\alpha^*=\beta$（全サンプルを等重みで混ぜる）。$A>0$ なら $\alpha^*\ge\beta$ で、$A$ が大きいほど $\alpha^*$ はターゲット寄りになる。
- $\alpha^*=1$（ターゲットだけ使う）になる条件は $c_m/\sqrt\beta\le 2A$、すなわちおおよそ
$$
m_T\ \gtrsim\ \frac{d\log m}{A^2}
$$
である。**ずれ $A$ が大きいほど、少ないターゲットラベルでソースが不要になる**。これが実験で検証する中心的な予測である（第6.5節 H1・H2）。

**定量化する量**：$\alpha^*(K)$ の曲線、ソースが不要になる $K^*$、$A$ の推定値（$\hat d_{\mathcal A}$ と $\hat\lambda$）。

**注意**：原論文は二値分類。多クラスでは、マージンに基づく拡張（Zhang et al., 2019 のMDD）や one-vs-rest での解釈が必要。論文ではこの点を限界として明記する。

### 4.2 ターゲットデータの価値・ミニマックスレート

**文献**：Hanneke & Kpotufe (2019)；Kpotufe & Martinet (2021)；Cai & Wei (2021)；Reeve et al. (2021)。

**主張**：
- ソースとターゲット両方の標本数に対するミニマックスレートが、transfer exponent と呼ばれる新しいずれの概念で決まる（Hanneke & Kpotufe）。分布パラメータが事前に分かっていれば、ミニマックス性能の達成は「ソースかターゲットの片方の標本を無視する」ことに近い。また、標本収集のコストに関する実際の意思決定は、分布パラメータやずれを推定しなくてもミニマックス最適に行える。
- 事後分布ドリフト（入力分布は同じで $P(Y\mid X)$ が変わる）の下で、ソース標本の寄与を正確に特徴づけたミニマックスレートと、適応的な分類器がある（Cai & Wei）。
- ソースとターゲットの関係が入力に依存し、ベイズ決定境界を保たなくてもよい枠組みで、最適レートが標本数の比と転移関係の強さに応じた複数の領域に分かれる（Reeve et al.）。

**定量化する量**：
- 「ソースはターゲットラベル何件分の価値があるか」を表す実効標本数 $K_{\mathrm{eq}}$（付録A.5）。
- $K$ を増やしたときに「ソースを無視する方が良い」領域への移行点。

**MERとの接点**：アノテーション規約の違いは事後分布ドリフトとして定式化できる。論文2の理論的土台の候補。

### 4.3 多ソース＋少数ターゲットラベル

**文献**：Konstantinov & Lampert (2019)；Mansour et al. (2021)；Mohri & Muñoz Medina (2012)。

**主張**：
- 少量の信頼できるターゲットデータを基準に、各ソースとのdiscrepancy（ラベルも使うずれ）を推定し、それに応じてソースを重み付けする手順が、統計的学習理論から導ける（Konstantinov & Lampert）。
- ターゲットのラベル付きデータが少なく、複数ソースのラベル付きデータが多い状況について、モデル選択に基づく手法群が良い保証を持つ。一方、ずれの推定に基づく手法の上界は $\tilde O(\sqrt{d/m_0})$（$m_0$ はターゲットのラベル付き標本数）で変化し、$m_0$ が小さいと弱くなる（Mansour et al.）。
- 時間とともに変化する分布の系列について、ラベルを含めたずれ（discrepancy）に基づく学習上界と、過去の標本への重みをその上界から選ぶアルゴリズム（Mohri & Muñoz Medina）。多ソースの重み付けへの読み替えの参考にする。

**定量化する量**：
- ソースごとの重み、ソース選択の効果。
- $K$ が小さいときに、ずれの推定がどれだけ当てにならないか（推定値の分散）。

### 4.4 ラベルシフト・条件付きシフト

**文献**：Lipton et al. (2018)；Alexandari et al. (2020)；Tachet des Combes et al. (2020)；Garg et al. (2023)；Zhao et al. (2019)。

**主張**：
- ラベルシフト（$P(X\mid Y)$ 不変、$P(Y)$ が変化）ならソースを $w(y)=P_T(y)/P_S(y)$ で重み付けし直せばよい。ラベルなしでも混同行列から $w$ を推定できる（BBSE）。
- 較正を補正した最尤推定（MLLS＋bias-corrected calibration）が、ラベルシフト適応ではなかなか超えられない強いベースラインになる。
- 500以上の分布シフト対のベンチマークで、代表的な13のDA手法がラベル比率のシフト下で従来知られていた以上に広く失敗することが示された。ラベル比率の補正を組み込むメタアルゴリズムで、2〜10ポイント改善する（Garg et al.）。

**定量化する量**：
- ラベル分布のずれ（TV・JS、付録A.4）。
- クラス条件付き分布のずれ（クラスごとのMMD）。
- ラベルシフト補正が効くのは「ラベル分布のずれが大きく、クラス条件付きのずれが小さい」ときだけ、という予測の検証（H4）。

### 4.5 仮説転移・fine-tuning

**文献**：Kuzborskij & Orabona (2017)；Kumar et al. (2022)。

**主張**：
- 補助仮説（ソースで学習したモデル）の重み付き組み合わせがターゲットで良く働くなら $O(1/m)$ の速いレートが得られ、そうでなければ通常のレートに戻る（Kuzborskij & Orabona）。
- 事前学習特徴が良く分布シフトが大きいとき、全層fine-tuningは線形プロービングより分布外精度で劣ることがある。先に線形プロービング、次に全層fine-tuning（LP-FT）が両者の利点を併せ持つ（Kumar et al.）。

**定量化する量**：
- ソースヘッドへの正則化の強さ $\lambda_{\mathrm{HT}}$ と $K$ の関係。
- 「ソースモデルのターゲット性能」と「仮説転移の利得」の相関（H6）。

### 4.6 メタ学習・表現転移

**文献**：Tripuraneni et al. (2020)；Du et al. (2021)；Chen et al. (2019)。

**主張**：
- 共有表現を $t$ 個の訓練タスクで学ぶ標本複雑度は、タスクが多様なら $C(\mathcal H)+t\,C(\mathcal F)$ で表せる（Tripuraneni et al.）。ここだけの記法として、$\mathcal H$ は**共有表現**のクラス、$\mathcal F$ は**タスク固有ヘッド**のクラス、$C(\cdot)$ はその複雑度である（第1章の線形仮説クラス $\mathcal H$ とは別物）。表現を共有しなければ $t\,(C(\mathcal H)+C(\mathcal F))$ 要るところを、表現の複雑度を1回分しか払わずに済む、という主張である。
- 各タスク $n_1$ 件の $N_{\mathrm{task}}$ 個のソースタスクで表現を学べば、ターゲットの過剰リスクは $\tilde O\!\left(C(\Phi)/(n_1N_{\mathrm{task}})+k/n_2\right)$ となる（Du et al.。原論文の記号では $N_{\mathrm{task}}$ は $T$）。$C(\Phi)$ は表現クラスの複雑度、$k$ は表現の次元、$n_2$ はターゲット標本数。
- few-shot分類では、単純な転移学習ベースラインがメタ学習と同等以上になりうる（Chen et al.）。

**定量化する量**：
- ソースコーパス数（タスク数 $N_{\mathrm{task}}$）と、タスクの作り方（コーパス／話者／セッション）による効果。
- タスク多様性の代理指標（ソース間のずれのばらつき）。

**含意**：コーパスが10程度しかないMERでは $N_{\mathrm{task}}$ が小さい。タスクをコーパス以外（話者群など）から作れるかどうかが、メタ学習を選ぶかどうかの分かれ目になる。

### 4.7 マルチモーダル

**文献**：Huang et al. (2021)。

**主張**：共通の潜在空間に符号化する融合の枠組みで、複数モダリティで学ぶ方が、その部分集合のモダリティだけで学ぶより母集団リスクが小さい（Huang et al.）。

**定量化する量**：
- モダリティ別の $A$ と、融合時の $A$。
- クロスコーパスでの融合利得が、同一コーパス内の融合利得より小さいかどうか（探索的）。

**空白**：モダリティごとにずれが異なる状況での、少数ラベル転移の上界は見当たらない。論文2の理論的貢献の候補。

### 4.8 測定ツール

| ツール | 用途 | 文献 |
|---|---|---|
| proxy A-distance | $d_{\mathcal H\Delta\mathcal H}$ の推定（ラベル不要） | Ben-David et al. (2010) |
| MMD＋置換検定 | 分布差の検定とカーネル距離 | Gretton et al. (2012) |
| OTDD | 特徴とラベルの両方を使うデータセット距離 | Alvarez-Melis & Fusi (2020) |
| LogME | 特徴とラベルからの転移性能予測 | You et al. (2021) |
| IWCV・モデル選択 | シフト下でのハイパーパラメータ選択 | Sugiyama et al. (2007)；You et al. (2019) |

### 4.9 MER/SER側の近接研究（応用）

- Ahn et al. (2021)：few-shot学習とDAを組み合わせたクロスコーパスSER。
- Tavernor & Mower Provost (2025, ASRU)：大規模なアノテータ集団で事前学習したモデルから、新しいアノテータに似た既知のアノテータを少数の登録データで特定し、低コストで適応する。**アノテーション差を少数ラベルで吸収する**という点で本研究の問題意識に最も近い。
- 半教師ありDAの上界と手法：Li et al. (2021, LIRR)。周辺分布と条件付き分布の両方を特徴空間で揃えることを上界から導いている。

---

## 5. 実験量と理論の対応表

| 記号 | 実験量 | 定義（付録参照） | 対応する理論の項 | オラクル？ |
|---|---|---|---|---|
| Q1 | $\hat d_{\mathcal A}$ | 線形ドメイン識別器のCV誤差 $\hat\varepsilon$ から $2(1-2\hat\varepsilon)$ | $d_{\mathcal H\Delta\mathcal H}$ | 否（ラベル不要） |
| Q2 | $\widehat{\mathrm{MMD}}^2$ と $p$ 値 | RBFカーネル、置換検定 | 周辺分布の差 | 否 |
| Q3 | $\mathrm{TV}(P_S(Y),P_T(Y))$、JS | ラベル周辺分布の差 | ラベルシフト | 全ラベル使用時はオラクル |
| Q4 | $\hat\lambda$ | $S\cup T$ 全ラベルで同時学習した線形分類器の $\hat\epsilon_S+\hat\epsilon_T$ | $\lambda$ | **オラクル**（分析専用） |
| Q5 | $\hat\Delta_K$ | $\hat\epsilon_T^{(K)}(h_S)-\hat\epsilon_S(h_S)$ | ラベル込みのずれの下界 | 否（$K$-shot で推定） |
| Q6 | クラス条件付きMMD | クラスごとの $\widehat{\mathrm{MMD}}^2$ の平均 | 条件付きシフト（GLS） | 全ラベル使用時はオラクル |
| Q7 | $K_{\mathrm{eq}}$ | ソースの価値をターゲットラベル数で表した値 | ターゲットデータの価値 | 否 |
| Q8 | $K^*$、$\alpha^*(K)$ | ソースが不要になる $K$、最適混合比 | Theorem 3 の $\alpha^*$ | $\alpha^*$ の真値はオラクル |

---

## 6. 最初の実験（明日〜第4週）のプロトコル

### 6.1 データ監査（最初に必ず行う）

- コーパスごとに次の表を作る：クラス別件数、話者数、分割の定義、ラベル写像。
  - 例：4クラス（neutral / happy / sad / angry）への統一。IEMOCAPで happy と excited を統合するかどうか、など。【要確認：キャッシュ済みコーパスの一覧と写像】
  - **ラベルの方針（2026-09-29）**：当面の実験は4ラベル（`label_4`）で行う。IEMOCAP の excited は Happy に含める。元データのラベル（発話一覧の `emotion_raw`。happy と excited を区別できる）での分析は追加で行う可能性がある。
- リーク確認：
  - 話者がテストと $K$-shot プールにまたがらないこと
  - コーパス間で同一素材が重複していないこと
  - 特徴抽出のバージョンとモデルの重みを記録する
- この表からQ3（ラベル分布のずれ）を算出する。

### 6.2 分割とサンプリング

- **テスト**：ターゲットごとに話者独立のテスト集合を固定する。分割は話者（`measure/split.py` の話者群）で行い、コーパスの公式分割（マニフェストの `split` 列）は使わない。公式分割は話者の分離を保証しないためである（`data/README.md` 注意点 4）。
- **$K$-shot プール**：テスト以外の話者から作る。
- **$K\in\{1,2,5,10,20,50,100\}$**（各クラス）。
- **抽出回数 $R=30$**：各 $(T,K)$ でランダムに30回抽出する。**全手法で同じ抽出を使う**（対応のある比較にするため）。
- クラス別に $K$ 件取るので、$K$-shot のラベル分布は一様になる。テストは元のラベル分布のままなので、主指標は事前分布に頑健なUARとし、WA（全体正解率）は副指標とする。

### 6.3 比較する手法

| ID | 手法 | 対応する理論 |
|---|---|---|
| M1 | ターゲットのみ（$K$-shot で線形分類器） | 基準 |
| M2 | ソースのみ | $\epsilon_T(h_S)$ |
| M3 | 単純混合（$\alpha=\beta$） | Theorem 3 で $A=0$ の場合 |
| M4 | $\alpha$ 重み付きERM、$\alpha\in\{0,0.1,\dots,1\}$ | Theorem 3 |
| M5 | 仮説転移：$\min_w \frac1{n_T}\sum\ell(w^\top\phi(x))+\lambda_{\mathrm{HT}}\lVert w-w_S\rVert^2$ | Kuzborskij & Orabona |
| M6 | ラベルシフト補正：ソースを $\hat w(y)=\hat P_T(y)/P_S(y)$ で再重み付け（BBSE／MLLS） | Lipton；Alexandari |
| M7a | 最近傍プロトタイプ（凍結特徴上） | メタ学習の最小形 |
| M7b | エピソード学習した小規模埋め込み（ソースからタスクを作る） | Tripuraneni；Du |
| M8 | 多ソース重み付け（ずれに基づく） | Konstantinov & Lampert |

**ハイパーパラメータ選択規則（事前に固定）**：
- ロジスティック回帰の正則化：ソース側の検証集合で決め、以後固定する。
- $\alpha$ と $\lambda_{\mathrm{HT}}$ は2通りで報告する。
  - **実用値**：$K$-shot 内の leave-one-out または交差検証で選ぶ（$K\ge2$）
  - **オラクル値**：テスト上の最良値（分析専用。手法の性能としては報告しない）

### 6.4 測定量

第5章のQ1〜Q8を、ソース・ターゲットの全順序対 × モダリティ（$a, v, l$, 融合）について計算する。

- Q1：ソースとターゲットから同数ずつ抽出し（上限 例：各2000件）、線形ドメイン識別器を5分割CVで学習する。
- Q2：同じ抽出でRBFカーネル（バンド幅はメディアン法）を使い、置換回数1000で検定する。
- Q4：オラクル。論文では「理論の項がどれだけ性能差を説明するか」の分析にのみ使う。

### 6.5 事前登録する仮説

実験前にこの節を確定し、結果を見てから書き換えない。

- **H1（Theorem 3）**：オラクルの $\alpha^*(K)$ は $K$ について非減少で、$A$ が大きい対ほど大きい。
- **H2（Theorem 3の系）**：ソースが不要になる $K^*$ は $A=\tfrac12\hat d_{\mathcal A}+\hat\lambda$ と負の順位相関を持つ。また特徴次元を大きくすると（PCAの次元を変えて操作）、$K^*$ は大きくなる。ただしPCA次元を変えると $\mathcal H$ 自体が変わるので、$A$ も同時に動く（$\mathcal H$ が大きいほど $d_{\mathcal H\Delta\mathcal H}$ は増え、$\lambda$ は減る）。各次元で $\hat d_{\mathcal A}$ と $\hat\lambda$ を測り直し、「$A$ を揃えたときの次元の効果」と「$A$ の変化を通じた効果」を分けて報告する。
- **H3（不可能性定理・Zhaoらの下界）**：転移利得（最良の $\alpha$ とM1の差）の予測に、Q1（ラベル不要のずれ）だけでは不十分。$\hat\lambda$ またはQ5を加えると予測が改善する（順位相関、または $R^2$ で比較）。
- **H4（ラベルシフト）**：M6が効くのはQ3が大きく、かつQ6が小さい対に限られる。
- **H5（マルチモーダル、探索的）**：$A$ の内訳（$\hat d_{\mathcal A}$ と $\hat\lambda$ の比）はモダリティによって異なる。
- **H6（仮説転移）**：小さい $K$ でのM5の利得は、M2（ソースのみ）のターゲットUARと正に相関する。

### 6.5.1 ずれの4軸についての予言と確かめ方（2026-09-30 登録）

導出は `docs/experiment-protocol.md` 7節。この節の文は、結果を見てから書き換えない。誤りに気づいたときは元の文を残し、日付つきの訂正を節の末尾に書き足す。H1〜H6 の文は変えない（P1-2 は H2 を、P4-4 は H6 を、確かめ方まで具体化したものである）。

**登録の時点**：地図（音声・WavLM、作り方①②③）の本番は30回中9〜10回まで進んでいる。途中結果の図表（`results/map_audio_wavlm/summary/`、9〜10回ぶん）は 2026-09-30 14:17 に作ったが、実装者（Claude）も研究者も開いていない（研究者には 2026-09-30 14:53 に確認）。例外として、実装の動作確認で AFEW←RAVDESS と AFEW←MOSEI の反復1回ずつの UAR を見た。

**共通の決まり**

1. 主判定は音声・第1エンコーダ（WavLM）で行う。映像・言葉・第2エンコーダは、同じ手順で再現を確かめて報告し、主判定は変えない。
2. 手順は `docs/experiment-protocol.md` に従う（R=30、$K\in\{1,5,10,25,50,\text{全件}\}$、話者分割、C は学習データ内の交差検証）。操作の効果は、同じ反復どうしの対の差（操作あり − なし）で測る。
3. 組をまたぐ検定の単位は組（ターゲット × ソース）とする。組は同じコーパスを共有しており互いに独立でないので、p 値は目安とし、効果量（中央値、順位相関）と並べて報告する。
4. 各予言に主判定を1つ置く。検算である P4-1 を除く10個の主判定に Holm 法をかけ、全体の有意水準5%で判定する。結果は「支持」（予言の向きで有意）、「反証」（逆の向きで有意）、「判定できない」のどれかとして、すべて報告する。
5. K* を使う判定（P1-2・P1-3）からは、区間の計算法が決まるまで、IEMOCAP がターゲットの組を除く（`docs/experiment-policy.md` 3節）。除いた組は参考として別に示す。K* を使わない判定は42組すべてで行う。
6. **ずれを測る空間**（P1-2・P1-3・P2-1 の PAD と coverage）は、UAR と K* を一切見ずに次の規則で選ぶ。候補は (a) 両側を合わせて標準化し、PCA で64次元に落とした空間、(b) ソースで学んだ感情の分類器（本研究の固定学習器）が出す4つの対数オッズの空間、の2つである。42組のうち PAD が1.9未満（飽和していない）の組が多い方を選ぶ。同数ならゼロ点（同じ群を話者で2つに割った値）の中央値が小さい方を選び、それも同じなら (a) とする。選んだ空間は、K* との関係を計算する前に `docs/experiment-policy.md` に記録する。
7. 操作 I3（CORAL）は PCA の後に行うので、比べる相手は、同じ PCA 空間で CORAL をかけない合併とする。操作後の K*（K*'）は、「操作後の合併 − 同じ操作を施したターゲットのみ」の差から、K* と同じ定義で求める。

**予言と確かめ方**

以下、$g$ は操作の得（操作あり − なしの UAR の差）、$h$ は害（付け替えあり − なし）、$m$ は組ごとの最悪の差 $\min_K$（合併 − ターゲットのみ）を表す。いずれも30回の対の差の平均である。

| 予言 | 使う実験と量 | 主判定 | 副判定・記述 |
|---|---|---|---|
| **P1-1**：入力のずれを消す操作の得は、K が小さいほど大きく、K* を超えると消える | 合併に I1〜I3 をかけたときの $g(K)$ | I1 について、組ごとに「$K\in\{1,5\}$ の $g$ の平均 − $K\in\{50,\text{全件}\}$ の $g$ の平均」を求め、42組で Wilcoxon 符号付き順位検定（片側、正） | I2・I3 も同じ検定で報告する。K* より大きい K で $g$ の95%区間が0を含む升の割合を記述する |
| **P1-2**：同じモダリティ・エンコーダの組どうしでは $K^*\propto d/A^2$（$\log K^*$ と $\log A$ の傾きが −2） | 地図の K*。$A$ の代わりに PAD/2（共通の決まり6の空間） | K* と $A$ の Spearman 順位相関が負（組の置換による片側検定）。K* が none の組は最小、全件の組は最大の順位とする | K* が有限の組で $\log K^*$ を $\log A$ に回帰した傾きと、組の再抽出による95%区間を示し、−2 を含むかを記述する。限界：$A$ のうち $\lambda$ は測れない |
| **P1-3**：操作で $A$ が $A'$ に下がれば、K* は $(A/A')^2$ 倍に右へ動く | I1 の前後の $A,A'$ と K*、K*' | $A'<A$ の組について、K*' > K* の組と K*' < K* の組の数で符号検定（片側、同じ値の組は除く） | $\log(K^{*\prime}/K^*)$ と $2\log(A/A')$ の対応を記述する（K の格子が粗いので記述にとどめる） |
| **P2-1**：覆いの悪さだけでは、合併がターゲットのみを大きく下回らない（負の転移は主に軸4から来る） | 地図の $m$、$\Delta_{CE}$（軸4）、coverage（軸2．ターゲットがソースに覆われる向き．共通の決まり6の空間） | $\lvert\rho(m,\Delta_{CE})\rvert-\lvert\rho(m,\text{coverage})\rvert$（Spearman）の、組の再抽出による95%区間が0を上回る | 各相関の値と区間を示す。限界：$\Delta_{CE}$ も coverage も入力のずれの影響を受け、軸を完全には分けられない |
| **P2-2**：近傍選択 C1 は、同数のランダム対照とは差が出る。全件より得になるのは $n'c'>nc$ のときだけである | C1 と同数のランダム対照、$n'\in\{0.25n,0.5n\}$、$c$ は `shift.cov_dominance` | $n'=0.5n$、$K\in\{1,5,10\}$ で「UAR(C1) − UAR(対照)」の組ごとの平均を求め、Wilcoxon（片側、正） | 「UAR(C1) − UAR(全件)」の符号と「$n'c'-nc$」の符号の一致率を、二項検定とともに記述する |
| **P3-1**：クラス重みを均等にした学習では、割合の操作（P1・P2）の効果は0に近い | P1・P2 と、同数のランダム対照との差 | P2 について、K ごとに組の平均の90%区間（組の再抽出）が ±0.01（UAR 1点）に収まるかを見る（同等性の検定．6つの K すべてで収まれば支持、どれかで区間が丸ごと外なら反証） | P1 も同じ手順で報告する。P2-oracle を別に記述する |
| **P3-2**：均等化の抜き出し P1 は本数が減るので、K が小さいとわずかに不利になる | UAR(合併＋P1) − UAR(合併、全件) | $K\in\{1,5\}$ の組ごとの平均で Wilcoxon（片側、負） | 大きさを Kish の $n_{\mathrm{eff}}$ の比と並べて記述する |
| **P4-1**：ソースのみの分類器について、$\mathrm{UAR}_S-\mathrm{UAR}_T\le3\Delta_{CE}$ が全組で成り立つ | ソースのみの分類器（選ばれた C）。$S$ 側はソースの交差検証の保留側、$T$ 側はターゲットの全発話で、UAR と混同行列を求める | **検算**：42組すべてで成り立つこと。混同行列から代数的に必ず成り立つので、破れたら実装の誤りとみなす（Holm の対象外） | 上限の余裕 $(\mathrm{UAR}_S-\mathrm{UAR}_T)/(3\Delta_{CE})$ を記述する |
| **P4-2**：一様な付け替え D3s は、ソースが主な情報源の小さい K で害が出て、K とともに消える。閾値の補正 D1 では取り戻せない | ソースのラベルに D3s（$\rho=0.1,0.2,0.4$）をかけた合併の $h(K)$ | $\rho=0.4$ で、組ごとに「$K\in\{1,5\}$ の $h$ の平均 − $K\in\{50,\text{全件}\}$ の $h$ の平均」を求め、Wilcoxon（片側、負．小さい K の方が害が大きい） | 取り戻せないことは、P4-3 の主判定で D3a と比べて確かめる。$\rho$ ごとの $h(K)$ を記述する |
| **P4-3**：特定の感情への付け替え D3a は、$\rho$ とともに害が増え、K* を D3s より強く小さい側へ動かす。D1 で大部分を取り戻せる | D3a は Neutral→Sad（$\rho=0.1,0.2,0.3$）。D1 はソースのみの分類器に、ターゲットの K 本で補正をかける。取り戻し率 = (UAR(付け替え＋D1) − UAR(付け替え)) / (UAR(付け替えなし) − UAR(付け替え))（ソースのみ．分母の95%区間が0を上回る組だけで計算） | $\rho=0.2$ で、取り戻し率（$K\in\{5,10,25\}$ の平均）の「D3a − D3s」を組ごとに求め、Wilcoxon（片側、正） | $\rho$ と $h$ の順位相関、共通の $\rho=0.1,0.2$ での K*（D3a）と K*（D3s）の比較、D3a の取り戻し率の中央値が0.5以上か、を記述する |
| **P4-4**：ソースへ寄せて学ぶ M5 の、小さい K での得は、$R_{\mathrm{src}}$ が小さい組ほど大きい（H6） | M5（$\lambda_{\mathrm{HT}}$ は K 本の中の交差検証で選ぶ）。得 = UAR(M5) − UAR(ターゲットのみ) の $K\in\{1,5\}$ の平均。$R_{\mathrm{src}}$ = 1 − UAR(ソースのみ、ターゲットのテスト) | 得と $R_{\mathrm{src}}$ の Spearman 順位相関が負（組の置換による片側検定） | オラクルの $\lambda_{\mathrm{HT}}$ での値を別に記述する |

### 6.5.2 組に固有のずれについての仮説と，第2エンコーダ（HuBERT）での確認（2026-09-30 登録）

**経緯**：音声（WavLM）の地図の途中結果（10〜11回）で、得と K* をターゲットの効果・ソースの効果・組に固有の部分に分けると、組に固有の部分どうしで、入力の距離とは負に、coverage とは正に相関した（`docs/exploratory-analyses.md` E2）。この方法は結果を見た後に決めたので、確かめた結果として扱わない。そこで、**まだ誰も結果を見ていない第2エンコーダ（HuBERT、`audio_hubert`）の地図で確かめる**ことを、ここに登録する。この節の文は、HuBERT の結果を見てから書き換えない。

**登録の時点**：HuBERT の地図とずれの指標はまだ計算していない（特徴 `audio_hubert_*.npz` だけがある）。

**H7（組に固有のずれ）**：同じターゲット・同じソースの効果を除くと、入力のずれが大きい組ほど、ソースを混ぜたときの小さい K での得が小さく、K* が小さい。ターゲットがソースに覆われている組ほど、得が大きく、K* が大きい。

**データと手順**

1. 地図：`exp/map.py --kind audio_hubert`、R=30、手順は WavLM の地図と同じ（`docs/experiment-protocol.md`、`docs/experiment-policy.md` 2節）
2. 指標：`exp/shift_map.py`（gpu・input・space）を `audio_hubert` で回す。軸1・2の空間は 6.5.1 共通の決まり6の規則で選ぶ
3. 結果の量：$g_{\mathrm{small}}$＝$K\in\{1,5\}$ での「合併 − ターゲットのみ」の UAR の差の平均（30回の平均）、K*（`exp/summarize_map.py` の定義。IEMOCAP がターゲットの組は、区間の計算法が決まるまで除く）
4. 解析：`exp/explore_decomp.py` の偏相関と同じ方法。結果の量と指標をそれぞれ順位に直し、結果の量 = μ + a_t + b_s + β × 指標 を最小二乗法で当てはめ、β を残りの自由度で t 検定する

**主判定（4つ．片側、Holm 法、全体の有意水準5%）**

| 番号 | 指標 | 結果の量 | 予言する β の符号 |
|---|---|---|---|
| H7a | PAD（軸1の主の指標） | $g_{\mathrm{small}}$ | 負 |
| H7b | PAD | K* | 負 |
| H7c | coverage（軸2の主の指標．ターゲットがソースに覆われる向き） | $g_{\mathrm{small}}$ | 正 |
| H7d | coverage | K* | 正 |

結果は 6.5.1 と同じく「支持」「反証」「判定できない」のどれかとして、すべて報告する。

**副判定（記述のみ）**：MMD²、フレシェ距離、density、R_src、Δ_CE についての同じ解析。ばらつきの内訳（ターゲットの効果・ソースの効果・組に固有の部分）。組をそのまま並べた順位相関（E1 の方法）。

**判定の結果（2026-10-06 追記．上の文は変えていない）**：4 つの主判定はすべて「判定できない」だった。偏相関は H7a −0.02、H7b −0.14、H7c +0.31、H7d +0.23 で、向きは 4 つとも予言どおりだったが、Holm 後の p は 0.20〜0.54 だった（`exp/check_h7.py` → `results/check_h7/`）。副判定と WavLM との比較は `docs/exploratory-analyses.md` の「HuBERT での確かめ」に書いた。

### 6.6 統計と再現性

- 各 $(T,K,$手法$)$ について、$R=30$ 回の平均とブートストラップによる95%信頼区間を報告する。
- 手法間の比較は、同じ抽出どうしの対応のある検定（Wilcoxon符号付き順位検定）を使い、多重比較はHolm法で補正する。
- 対レベルの相関（H2・H3）では、コーパス対の数が少ないことに注意する（例：5コーパスなら順序対は20）。順位相関とそのブートストラップ信頼区間を報告し、効果量を中心に述べて過大な主張をしない。
- 乱数シード、設定ファイル、特徴のバージョン、結果の生データ（手法×抽出×$K$）をすべて保存する。

### 6.7 1日目の範囲

1. データ監査の表を作り、Q3を出す。
2. Q1・Q2を、全順序対 × モダリティで計算する。
3. M1〜M4の $K$-shot 曲線を描く（$R=30$、$K\in\{1,2,5,10,20,50\}$、$\alpha$ は11段階）。
4. 図：UAR vs $K$（対数軸）、$\alpha^*(K)$、Q1・Q3のヒートマップ。
5. H1・H2との照合メモ（途中経過で構わない）。

M5〜M8とQ4〜Q8は2〜3週目に回す。

### 6.8 第4週末の判断基準

- **DA路線（論文1をA+として進める）**：M4／M5が小さい $K$（$\le20$）で多くの対においてM1を上回り、かつ $K^*$ が $A$ と対応する場合。
- **メタ学習路線**：M7bが小さい $K$ で一貫してM4／M5を上回る場合。理論はTripuraneni／Duで立てる。
- **「ソースは役立たない」が主な結果の場合**：M1が非常に小さい $K$ から優勢になる場合。これ自体が論文1の結果になる（$\lambda$ による説明を付ける）。論文2はアノテーション差（事後分布ドリフト）のモデル化に向かう。

---

## 7. 13週計画

| 週 | 期間 | 実験 | 理論・文献 | 成果物 |
|---|---|---|---|---|
| W1 | 9/29–10/5 | データ監査、Q1〜Q3、M1〜M4 | Ben-David Thm 3、Zhao 2019 | 1日目の図とメモ |
| W2 | 10/6–10/12 | Q4〜Q6、M5、M6 | Tachet des Combes、Lipton、Alexandari、Garg | 仮説H1〜H4の途中判定 |
| W3 | 10/13–10/19 | M7a／M7b、M8、Q7・Q8 | Hanneke & Kpotufe、Mansour 2021、Konstantinov & Lampert | 全手法の $K$-shot 曲線 |
| W4 | 10/20–10/26 | 結果の集約 | 先行研究調査・1回目（問題設定の正当化） | **方向決定**、関連研究の下書き |
| W5 | 10/27–11/2 | $K^*$・$\alpha^*$ を測定量から予測する分析 | Cai & Wei、Reeve（事後分布ドリフト） | 論文1の中心図 |
| W6 | 11/3–11/9 | 実用的な $\alpha$ 選択規則（A+の手法） | 多クラスへの拡張（MDD） | 手法の定義と評価 |
| W7 | 11/10–11/16 | 頑健性：PCA次元、別エンコーダ、LP-FT／小規模MLP | Kumar 2022、Du 2021 | 付録用の実験 |
| W8 | 11/17–11/23 | 補完実験 | 先行研究調査・2回目（比較対象の確定） | **論文1ドラフト** |
| W9 | 11/24–11/30 | 論文2の手法設計 | 論文1で見つかった支配的なずれに応じて選定 | 設計メモ |
| W10 | 12/1–12/7 | 論文2の実装と実験 | — | 予備結果 |
| W11 | 12/8–12/14 | 論文2の実験 | — | 主結果 |
| W12 | 12/15–12/21 | アブレーションと統計 | — | 図表 |
| W13 | 12/22–12/28 | 執筆とバッファ | — | **論文2ドラフト** |

投稿先と締切が決まったら、この表の論文1・論文2の区切りを締切に合わせて前後させる。

**2026-09-30 の変更（0.1 節）**：W1〜W3 の M5〜M8 と、ずれを消す・足す操作は後回しにする。代わりに、全モダリティと第2エンコーダの地図、ずれの指標、E シリーズの分析（`docs/exploratory-analyses.md`）を先に行う。GPU の順番は `docs/experiment-policy.md` 1 節。

---

## 8. リスクと対策

| リスク | 対策 |
|---|---|
| コーパス対が少なく、相関分析の検出力が低い | モダリティ別に分析する。話者群などのサブドメインで補助的な対を作る。効果量と信頼区間を中心に報告する |
| 線形ヘッドでの結論が深層fine-tuningに一般化しない | W7でLP-FT・小規模MLPで確認し、差があれば限界として明記する |
| 4クラスへの統一で情報が失われる | 限界として明記する。次元ラベル（感情価・覚醒度）は将来課題とする |
| 理論が二値分類向けで、多クラスの設定とずれる | one-vs-restでの解釈と、MDDなど多クラスの上界を参照する |
| メタ学習のタスク数が足りない | タスクを話者・セッション単位で構成する。足りなければDA路線を優先する |
| 3ヶ月で2本は過大 | 論文2はドラフト／ワークショップ水準を許容する。論文1を最優先にする |
| 少数ラベルでのハイパーパラメータ選択が不安定 | 選択規則を事前に固定し、実用値とオラクル値を分けて報告する |

---

## 9. 読む順番（何を実験に持ち帰るか）

1. **Ben-David et al. (2010) Theorem 3**：$\alpha^*$ と $K^*$ の予測（H1・H2）。
2. **Zhao et al. (2019)、Tachet des Combes et al. (2020)**：$\lambda$ とラベルシフトによる失敗の仕組み（H3・H4）。
3. **Hanneke & Kpotufe (2019)**：主定理の主張と transfer exponent の意味（Q7）。
4. **Mansour et al. (2021)、Konstantinov & Lampert (2019)**：多ソース＋少数ラベルの手法と、ずれの推定が弱くなる理由（M8）。
5. **Lipton et al. (2018)、Alexandari et al. (2020)、Garg et al. (2023)**：ラベルシフト補正の強いベースライン（M6）。
6. **Kuzborskij & Orabona (2017)、Kumar et al. (2022)**：仮説転移とLP-FT（M5、W7）。
7. **Du et al. (2021)、Tripuraneni et al. (2020)、Chen et al. (2019)**：メタ学習を選ぶ条件（M7）。
8. **Cai & Wei (2021)、Reeve et al. (2021)**：アノテーション差を事後分布ドリフトとして定式化する土台（論文2）。
9. **Redko et al. (サーベイ)**：辞書的に参照する。

---

## 付録A　指標の定義と導出

実装と，値がモデルやデータの形で動く点の注意は `docs/shift-metrics.md`（実装は `measure/shift.py`）。

### A.1 proxy A-distance

ソースに0、ターゲットに1のドメインラベルを付け、同数ずつ用意する。$\mathcal H$ が補集合で閉じていれば、分類器 $\eta$ のドメイン識別誤り率は

$$
\mathrm{err}(\eta)=\tfrac12P_S(\eta=1)+\tfrac12P_T(\eta=0)=\tfrac12\left[P_S(\eta=1)+1-P_T(\eta=1)\right]
$$

なので、$P_T(\eta=1)-P_S(\eta=1)=1-2\,\mathrm{err}(\eta)$ となる。補集合に閉じていることから、絶対値付きの上限は絶対値なしの上限に一致し、

$$
d_{\mathcal H}=2\sup_\eta\left[P_T(\eta=1)-P_S(\eta=1)\right]=2\left(1-2\min_\eta\mathrm{err}(\eta)\right).
$$

実装では、学習した線形識別器の交差検証誤差 $\hat\varepsilon$ を使い、$\hat d_{\mathcal A}=\max\{0,\,2(1-2\hat\varepsilon)\}$ とする。

### A.2 MMD（最大平均差異）の不偏推定量

$$
\widehat{\mathrm{MMD}}^2_u=\frac{1}{n(n-1)}\sum_{i\ne j}k(x_i,x_j)+\frac{1}{m(m-1)}\sum_{i\ne j}k(y_i,y_j)-\frac{2}{nm}\sum_{i,j}k(x_i,y_j)
$$

これは $\|\mu_S-\mu_T\|_{\mathcal H}^2=\langle\mu_S,\mu_S\rangle-2\langle\mu_S,\mu_T\rangle+\langle\mu_T,\mu_T\rangle$ に $\langle\phi(x),\phi(y)\rangle=k(x,y)$ を代入し、自分自身との対（$i=j$）を除いたものである。

### A.3 最適な混合比 $\alpha^*$ の性質

右辺の $\alpha$ に依存する部分を $f(\alpha)=c_m\sqrt{g(\alpha)}+2(1-\alpha)A$、$g(\alpha)=\frac{\alpha^2}{\beta}+\frac{(1-\alpha)^2}{1-\beta}$ とおく。

1. **凸性**：$\sqrt{g(\alpha)}=\left\lVert\left(\alpha/\sqrt\beta,\ (1-\alpha)/\sqrt{1-\beta}\right)\right\rVert_2$ は、アフィン写像のユークリッドノルムなので凸。したがって $f$ は凸で、最小点は一意に定まる。
2. **微分**：$g'(\alpha)=\frac{2\alpha}{\beta}-\frac{2(1-\alpha)}{1-\beta}$ より
$$
f'(\alpha)=c_m\frac{\alpha/\beta-(1-\alpha)/(1-\beta)}{\sqrt{g(\alpha)}}-2A.
$$
3. **$\alpha=\beta$ での値**：分子は $1-1=0$ なので $f'(\beta)=-2A\le0$。$f$ は凸なので $f'$ は非減少であり、$\alpha^*\ge\beta$。$A=0$ なら $\alpha^*=\beta$。
4. **$\alpha=1$ での値**：$g(1)=1/\beta$、分子は $1/\beta$ なので $f'(1)=c_m/\sqrt\beta-2A$。したがって
$$
\alpha^*=1\iff\frac{c_m}{\sqrt\beta}\le2A.
$$
5. **ターゲット標本数への書き換え**：$c_m=C\sqrt{d\log m/m}$、$m_T=\beta m$ を代入すると $c_m/\sqrt\beta=C\sqrt{d\log m/m_T}$ となる。よって $\alpha^*=1$ の条件は
$$
m_T\ \ge\ \frac{C^2\,d\log m}{4A^2}.
$$
**ずれ $A$ が大きいほど、またVC次元 $d$ が小さいほど、少ないターゲットラベルでソースが不要になる**（H2の根拠）。ただし $A$ は $\mathcal H$ に依存する量なので、$d$ を変えると $A$ も変わる。「$d$ が小さいほど」は $A$ を固定したときの比較として読む。

### A.4 ラベル分布のずれ

- 全変動距離：$\mathrm{TV}(p,q)=\tfrac12\sum_y|p(y)-q(y)|$
- Jensen–Shannonダイバージェンス：$M=\tfrac12(p+q)$ として $\mathrm{JS}(p\|q)=\tfrac12\mathrm{KL}(p\|M)+\tfrac12\mathrm{KL}(q\|M)$。底を2にすると $[0,1]$ の値をとる。

### A.5 実効標本数 $K_{\mathrm{eq}}$ とクロスオーバー $K^*$

- $K_{\mathrm{eq}}$：ターゲットのみ（M1）のUARが、ソースのみ（M2）のUARに到達する最小の $K$。対数軸上で線形補間して求める。「このソースはターゲットラベル $K_{\mathrm{eq}}$ 件分の価値がある」と解釈する。
- $K^*$：M1のUARが、単純混合（M3）以上になる最小の $K$。オラクルの $\alpha^*(K)$ が1に達する $K$ も併記する。

### A.6 UAR（クラス平均再現率）

$$
\mathrm{UAR}=\frac1C\sum_{c=1}^C\frac{\#\{\hat y=c,\ y=c\}}{\#\{y=c\}}
$$

クラス別の再現率を平均するので、テスト集合のクラス事前分布に左右されにくい。

---

## 参考文献

**DA理論の基礎**
- [S. Ben-David, J. Blitzer, K. Crammer, A. Kulesza, F. Pereira, J. W. Vaughan，"A theory of learning from different domains，" Machine Learning，vol. 79，no. 1–2，pp. 151–175，2010](https://doi.org/10.1007/s10994-009-5152-4)
- [S. Ben-David, T. Lu, T. Luu, D. Pál，"Impossibility Theorems for Domain Adaptation，" AISTATS，2010](https://www.cs.toronto.edu/~tl/papers/BenDavidEtAl_aistats2010.pdf)
- [H. Zhao, R. Tachet des Combes, K. Zhang, G. Gordon，"On Learning Invariant Representations for Domain Adaptation，" ICML，pp. 7523–7532，2019](https://arxiv.org/abs/1901.09453)
- [R. Tachet des Combes, H. Zhao, Y.-X. Wang, G. Gordon，"Domain Adaptation with Conditional Distribution Matching and Generalized Label Shift，" NeurIPS，2020](https://arxiv.org/abs/2003.04475)
- [I. Redko, E. Morvant, A. Habrard, M. Sebban, Y. Bennani，"A survey on domain adaptation theory: learning bounds and theoretical guarantees，" arXiv:2004.11829，2020](https://arxiv.org/abs/2004.11829)

**ターゲットデータの価値・ミニマックス**
- [S. Hanneke, S. Kpotufe，"On the Value of Target Data in Transfer Learning，" NeurIPS，pp. 9867–9877，2019](https://arxiv.org/abs/2002.04747)
- [S. Kpotufe, G. Martinet，"Marginal singularity and the benefits of labels in covariate-shift，" The Annals of Statistics，vol. 49，no. 6，pp. 3299–3323，2021](https://arxiv.org/abs/1803.01833)
- [T. T. Cai, H. Wei，"Transfer learning for nonparametric classification: Minimax rate and adaptive classifier，" The Annals of Statistics，vol. 49，no. 1，pp. 100–128，2021](https://arxiv.org/abs/1906.02903)
- [H. W. J. Reeve, T. I. Cannings, R. J. Samworth，"Adaptive transfer learning，" The Annals of Statistics，vol. 49，no. 6，pp. 3618–3649，2021](https://doi.org/10.1214/21-AOS2102)

**多ソース・少数ターゲットラベル**
- [Y. Mansour, M. Mohri, J. Ro, A. T. Suresh, K. Wu，"A Theory of Multiple-Source Adaptation with Limited Target Labeled Data，" AISTATS，PMLR 130，pp. 2332–2340，2021](https://proceedings.mlr.press/v130/mansour21a.html)
- [N. Konstantinov, C. H. Lampert，"Robust Learning from Untrusted Sources，" ICML，PMLR 97，2019](https://arxiv.org/abs/1901.10310)
- [M. Mohri, A. Muñoz Medina，"New Analysis and Algorithm for Learning with Drifting Distributions，" ALT，LNCS 7568，pp. 124–138，2012](https://arxiv.org/abs/1205.4343)
- [B. Li, Y. Wang, S. Zhang, D. Li, K. Keutzer, T. Darrell, H. Zhao，"Learning Invariant Representations and Risks for Semi-supervised Domain Adaptation，" CVPR，pp. 1104–1113，2021](https://arxiv.org/abs/2010.04647)

**ラベルシフト**
- [Z. C. Lipton, Y.-X. Wang, A. Smola，"Detecting and Correcting for Label Shift with Black Box Predictors，" ICML，2018](https://arxiv.org/abs/1802.03916)
- [A. Alexandari, A. Kundaje, A. Shrikumar，"Maximum Likelihood with Bias-Corrected Calibration is Hard-To-Beat at Label Shift Adaptation，" ICML，PMLR 119，pp. 222–232，2020](https://proceedings.mlr.press/v119/alexandari20a.html)
- [S. Garg, N. Erickson, J. Sharpnack, A. Smola, S. Balakrishnan, Z. C. Lipton，"RLSbench: Domain Adaptation Under Relaxed Label Shift，" ICML，PMLR 202，pp. 10879–10928，2023](https://arxiv.org/abs/2302.03020)

**仮説転移・fine-tuning**
- [I. Kuzborskij, F. Orabona，"Fast rates by transferring from auxiliary hypotheses，" Machine Learning，vol. 106，no. 2，pp. 171–195，2017](https://doi.org/10.1007/s10994-016-5594-4)
- [A. Kumar, A. Raghunathan, R. Jones, T. Ma, P. Liang，"Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution，" ICLR，2022](https://arxiv.org/abs/2202.10054)

**メタ学習・表現転移**
- [N. Tripuraneni, M. I. Jordan, C. Jin，"On the Theory of Transfer Learning: The Importance of Task Diversity，" NeurIPS，vol. 33，pp. 7852–7862，2020](https://arxiv.org/abs/2006.11650)
- [S. S. Du, W. Hu, S. M. Kakade, J. D. Lee, Q. Lei，"Few-Shot Learning via Learning the Representation, Provably，" ICLR，2021](https://arxiv.org/abs/2002.09434)
- [W.-Y. Chen, Y.-C. Liu, Z. Kira, Y.-C. F. Wang, J.-B. Huang，"A Closer Look at Few-shot Classification，" ICLR，2019](https://arxiv.org/abs/1904.04232)

**マルチモーダル**
- [Y. Huang, C. Du, Z. Xue, X. Chen, H. Zhao, L. Huang，"What Makes Multi-modal Learning Better than Single (Provably)，" NeurIPS，vol. 34，pp. 10944–10956，2021](https://arxiv.org/abs/2106.04538)

**測定ツール・モデル選択**
- [A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Schölkopf, A. Smola，"A Kernel Two-Sample Test，" JMLR，vol. 13，pp. 723–773，2012](https://jmlr.org/papers/v13/gretton12a.html)
- [D. Alvarez-Melis, N. Fusi，"Geometric Dataset Distances via Optimal Transport，" NeurIPS，vol. 33，pp. 21428–21439，2020](https://arxiv.org/abs/2002.02923)
- [K. You, Y. Liu, J. Wang, M. Long，"LogME: Practical Assessment of Pre-trained Models for Transfer Learning，" ICML，PMLR 139，pp. 12133–12143，2021](https://ise.thss.tsinghua.edu.cn/~mlong/doc/LogME-Practical-Assessment-of-Pre-trained-Models-for-Transfer-Learning-icml21.pdf)
- [K. You, X. Wang, M. Long, M. I. Jordan，"Towards Accurate Model Selection in Deep Unsupervised Domain Adaptation，" ICML，PMLR 97，2019](http://proceedings.mlr.press/v97/you19a.html)
- [M. Sugiyama, M. Krauledat, K.-R. Müller，"Covariate Shift Adaptation by Importance Weighted Cross Validation，" JMLR，vol. 8，pp. 985–1005，2007](https://www.jmlr.org/papers/v8/sugiyama07a.html)

**MER/SER側の近接研究**
- [Y. Ahn, S. J. Lee, J. W. Shin，"Cross-Corpus Speech Emotion Recognition Based on Few-Shot Learning and Domain Adaptation，" IEEE Signal Processing Letters，vol. 28，pp. 1190–1194，2021](https://doi.org/10.1109/LSP.2021.3086395)
- [J. Tavernor, E. Mower Provost，"More Similar than Dissimilar: Modeling Annotators for Cross-Corpus Speech Emotion Recognition，" ASRU，2025](https://arxiv.org/abs/2509.12295)

## 付録B　未決事項（期限つき）

実験に関する未決事項は `docs/experiment-policy.md` 3 節に移した（2026-09-30）．
