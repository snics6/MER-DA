（この報告は prior-work-diff の最後の返答から本体のセッションが保存した．サブエージェントはファイルを書けなかった．2026-11-14）

# 先行研究との差分：信頼度の表を「水準は K 本・相性はソース」で組み立て，戻す強さと相性の使い方を別のコーパスで選ぶ手順（実験 16 の提案の手法）（2026-11-14）

担当は prior-work-diff である．第 1 版の形（感情ごとのモダリティの信頼度の表で答えをまとめること自体）は `research_notes/先行研究_信頼度の表_2026-11-13/report.md` で調べ済みなので，ここでは重ねて調べない．ソースの重みと残り方は `research_notes/先行研究_ソースの重み付き出発点と残り方_2026-11-14/report.md` の結果を引き継ぐ．

## 0. 要約（10 行）

1. 判定：手法の部品は，どれも既知の考え方を当てはめたものである．新しいと言える余地は 2 つある．1 つは部品の組み合わせ方で，どの数をターゲットの K 本で測り，どの数をソースから持ち込むかの分け方である．もう 1 つは，その分け方をクロスコーパスの感情認識で測った記述である．1% の新しさとしては書ける．
2. 分類器 × 感情の重みで答えをまとめる形そのものには，感情認識に先例がある．Kaya ほか 2017 と Kahou ほか 2015 は，AFEW（EmotiW）で，モデル × 感情の重みを検証データの上の乱数探索で決めた．両論文とも，検証データで高く出た成績がテストで下がった例を報告している．
3. 「ラベルが少ないときは推す数を減らし，残りは多いデータから持ち込む」考えにも先例が多い．最も近いのは Guo ほか 2017 の較正の比べである．分類器全体で 1 つの数を使う temperature scaling が，クラスごとの数を持つ vector scaling と，K×K の行列を使う matrix scaling より良かった．ほかに次の 4 つがある．Paun ほか 2018 は，注釈者ごとの表を母集団の値へ寄せる．Simple CNAPS（Bateni ほか 2020）は，クラスの例の数 n から n/(n+1) の比で混ぜる．音声の MAP 適応は，relevance factor を使って n/(n+τ)（τ = 16）で混ぜる．MLLR は回帰クラスの木を使う．
4. 残り方（ソースの外での落ち方）で信頼度を割り引くことに近い先例は 2 つある．Yu ほか 2022 は，複数のソースのドメインで較正の数を測り，新しいドメインでの数を見込む．証拠理論の discounting は情報源の信頼性を割り引く操作で，Mercier ほか 2008 はクラスごとの割引率を持つ版を示した．
5. 戻す強さと相性を使うかを別のコーパスで選ぶ手順には，2 つの先例がある．ドメイン汎化の leave-one-domain-out の選択（Gulrajani & Lopez-Paz 2021）と，縮める強さをメタ学習で決める Chen ほか 2019 である．
6. 見つからなかったことは 3 つある．(a) 信頼度の表の行の平均だけをターゲットで測り，行の中の差をソースから持ち込む分け方．(b) 信頼度をソースどうしの残り方の比で割り引いて，ターゲットでの値を見込む手順．(c) 融合の重みの戻す強さ（ラベル何本ぶん）を，別のコーパスを仮のターゲットにして選ぶ手順．
7. 狙うべき差分は，保存済みの出力から出せる 3 つの記述である．1 つ目は，K 本から 1 つの数（水準）だけを測る版と 12 個すべてを測る版の差で，Guo ほか 2017 の原則がクラスあたり 5 本でも成り立つかを示す．2 つ目は，相性の持ち越し（ソースとターゲットの相性の形の相関）である．3 つ目は，別のコーパスで選ばれた戻す強さの値（`mixsel.csv` に記録済み）を，音声の relevance factor の慣習（16 本ぶん）と比べることである．

---

## 1. 読んだもの・確かめ方・語

### 1.1 読んだもの

- 入力：`DAI/docs/実験16_下調べ_設計.md`（0〜3 節，5〜8 節），`DAI/docs/説明_実験16の案_図解_2026-11-13.tex`（5〜8 節），`research_notes/先行研究_信頼度の表_2026-11-13/report.md`，`research_notes/先行研究_ソースの重み付き出発点と残り方_2026-11-14/report.md`，`DAI_paper/CLAUDE.md`，`DAI_paper/inventory.md`（検索した範囲），`DAI/docs/実験16_下調べ_結果の文の下書き_2026-11-14.md`（見出しと，選んだ格子の点の 1 行だけ）．
- `results/` のファイルは開いていない．計算はしていない．
- 今回 PDF の本文を読んだもの：Kaya ほか 2017，Mercier ほか 2008，Guo ほか 2017，Yu ほか 2022，Paun ほか 2018，Bateni ほか 2020（Simple CNAPS），Lucey & Chen 2003（relevance factor の式の確認），Kittler ほか 1998，Ting & Witten 1999．HTML を道具に読ませたもの：Kahou ほか 2015（EmoNets），Chen ほか 2019，Gulrajani & Lopez-Paz 2021．

### 1.2 検証の水準

- ◎：原文で該当箇所を確かめた．「道具経由」と書いたものは，HTML を道具に読ませて確かめたもので，引用の前に PDF で数値を確かめる．
- ○：抄録か信頼できる二次資料で確かめた．
- △：検索の断片だけで確かめた．
- 過去の 2 つの報告の水準は引き継ぎ，「継承」と書く．

### 1.3 この報告で使う語

- **信頼度の表**：分類器（声・顔・言葉）× 感情（4 つ）の 12 個の数．各数は，その分類器がその感情の発話をほかの感情の発話より高く採点する度合いで，感情の対の AUC から作る d′ で表す．
- **水準**：表の 1 行（1 つの分類器）の 4 つの数の平均．その分類器が全体としてどれだけ当たるかを表す．
- **相性**：行の各数から水準を引いた 4 つの数．その分類器がどの感情を比較的得意とするかを表す．
- **戻す強さ n0**：K 本で測った水準を戻し先の値へ引き寄せる強さを，「ラベル何本ぶん」で表した数．使う水準 =（n × K 本の水準 + n0 × 戻し先）÷（n + n0），n = 4K．
- **残り方**：ソースの分類器を別のソースのコーパスに当てて測った信頼度の合計を，同じソースの中の交差検証で測った信頼度の合計で割った値．
- **縮小（shrinkage）**：少ないデータで測った値を，多いデータで決めた値へ引き寄せて，測定のぶれを減らすこと．
- **部分プーリング（partial pooling）**：グループ（ここでは注釈者や分類器）ごとの値を，全グループの平均へ途中まで寄せて推す階層ベイズの手続き．平均へ全部寄せるのが完全プーリング，寄せないのが非プーリングである．
- **較正**：分類器が出す確率を，実際の正解の割合に合うように写し直すこと．temperature scaling は，全クラスの点を 1 つの数で割る較正である．vector scaling はクラスごとに別の数を掛け，matrix scaling は点に K×K の行列を掛ける．
- **leave-one-domain-out（LODO）**：ドメインを 1 つ外して残りで学び，外したドメインで測る手続き．この報告では「別のコーパスを仮のターゲットにして選ぶ」ことを指す．
- **後段の融合（late fusion）**：モダリティごとの分類器の出力（確率や点）をまとめて答えを出すこと．

---

## 2. 研究目標を検索できる問いに分ける

提案の手法（研究者の 2026-11-14 の決定）は，次の 4 つの決め事からなる．

| 問い | 中身 | 節 |
|---|---|---|
| Q1 | 信頼度の表を水準と相性に分け，水準（3 個）だけをターゲットの K 本で測り，相性（12 個のうち残りの自由度）にはソースで測った値を使う．少数ラベルで測る量と持ち込む量を，パラメータの数で分ける考えに先例はあるか | 3.1 |
| Q2 | ソースの中の信頼度に残り方を掛けて，新しいコーパスでの信頼度を見込む．融合の重みや較正を，ドメインの外での落ち方で割り引く先例はあるか | 3.2 |
| Q3 | K 本の水準を戻し先へ n0 本ぶん戻す縮小と，n0・相性を使うか・戻し先を，別のコーパスを仮のターゲットにして選ぶ手順．融合の重みの縮小の強さをメタ的に決める先例はあるか | 3.3 |
| Q4 | 感情認識で，モダリティごと（または感情ごと）の重みを少数ラベルか検証データで決める後段の融合の代表的な文献．比べる相手（確率の平均，重みつき多数決，naive Bayes combiner，stacking）の出典 | 3.4 |

---

## 3. 問いごとの先行研究

### 3.1 Q1：少数ラベルで測る数と，持ち込む数をパラメータの数で分ける

| 文献 | 何をしたか | 提案の手法との関係 | 水準 | URL |
|---|---|---|---|---|
| **Guo, Pleiss, Sun, Weinberger 2017（ICML．On Calibration of Modern Neural Networks）** | 較正の 3 つの版を比べた．matrix scaling は点に Wz + b を当てる．vector scaling は W を対角に制限する．temperature scaling は「a single scalar parameter T > 0 for all classes」を使う（4.2 節）．結果は「temperature scaling outperforms the vector and matrix Platt scaling variants, which are strictly more general methods」で，「vector scaling recovers essentially the same solution as temperature scaling – the learned vector has nearly constant entries」と書く．著者は「network miscalibration is intrinsically low dimensional」とまとめる．matrix scaling はクラスが数百あると悪い．「Any calibration model with tens of thousands (or more) parameters will overfit to a small validation set, even when applying regularization」（5 節） | **水準と相性の分け方に最も近い原則の先例である．**提案の手法の重みは，分類器 k の感情 i の点 z_k(i) に掛ける数である．分類器ごとに見ると，クラスごとに別の数を掛ける vector scaling と同じ形になる．水準は分類器ごとの 1 つの数で，temperature scaling の 1 つの数に当たる（式から言えることで，文献の主張ではない）．違いは 3 つある．(1) Guo は 1 つのモデルの較正を扱い，提案の手法は複数のモダリティを組み合わせる重みを扱う．(2) Guo はクラスごとの数を捨てるが，提案の手法はクラスごとの差（相性）をソースから持ち込む．(3) Guo の検証データは数千本で，提案の手法の K 本は 20 本である | ◎ | https://arxiv.org/abs/1706.04599 |
| Kull ほか 2019（NeurIPS．Dirichlet calibration） | 対数確率に行列を掛ける多クラスの較正．非対角と切片に罰則（ODIR）を掛ける | 推す数が多い較正を罰則で抑える先例．数を減らすのではなく，値を縮める | ○（継承） | https://arxiv.org/abs/1910.12656 |
| **Paun, Carpenter, Chamberlain, Hovy, Kruschwitz, Poesio 2018（TACL 6．Comparing Bayesian Models of Annotation）** | 注釈者ごと・本当のクラスごとの応答の表（Dawid–Skene の表）を持つ 6 つの模型を比べた．非プーリングの Dawid–Skene では，各注釈者の表に固定の事前を置く．階層の Dawid–Skene（2.3 節）では事前を母集団の分布にし，「This structure provides partial pooling, using information about the population to improve estimates of individuals by regularizing toward the population mean. This is particularly helpful with low count data」と書く．非プーリングの模型について「The higher complexity of unpooled models results in overfitting, which affects their predictive performance」と書く | 判定者 × クラスの表を，データの少ない判定者では母集団の値へ寄せる先例である．提案の手法の分類器 × 感情の表と同じ構造をもつ．違いは寄せ方にある．Paun は表の全部を一様に母集団へ寄せる．提案の手法は，行の平均だけをターゲットのデータで測り，行の中の差は完全にソースの値へ寄せる（完全プーリング）．行の平均と行の中の差で寄せ方を変える分け方は，この論文には無い | ◎ | https://aclanthology.org/Q18-1040/ |
| Kim & Ghahramani 2012（Ghahramani & Kim 2003 の技術報告） | 分類器ごとの表の行にディリクレの事前を置き，検証データを既知の正解として推す | 表の行を縮める先例（継承） | ◎（継承） | https://mlg.eng.cam.ac.uk/zoubin/papers/GhaKim03.pdf |
| **Kuncheva & Rodríguez 2014（KAIS 38）** | 重みつき多数決（分類器ごとに 1 つの数），recall combiner（分類器 × クラスの再現率），naive Bayes combiner（分類器ごとの表の全部）を，入れ子の関係として導いた．パラメータの数を L+c，L(c+1)，L·c²+c と数えた．有限の標本では推定の精度が先に問題になると書く | 「水準だけ」の版は重みつき多数決に，「12 個すべて」の版は recall combiner に近い（継承）．この論文は版を選ぶ問題として扱っており，1 つの版の中で数を分けて別のデータから測ることはしていない | ◎（継承） | https://lucykuncheva.co.uk/papers/lkjrKAIS14.pdf |
| **Bateni, Goyal, Masrani, Wood, Sigal 2020（CVPR．Simple CNAPS）** | 少数例の分類で，クラスごとの共分散 Σ_k とタスク全体の共分散 Σ を，Q_k = λ_k Σ_k + (1 − λ_k) Σ + βI で混ぜる．λ_k = |S_k| / (|S_k| + 1) で，|S_k| はそのクラスのラベル付きの例の数である．「the higher the number of shots, the better the class-within-task covariance estimate gets」と書く．λ を学ぶ版も試したが，「the simple deterministic ratio above produced the best results」 | **ラベルの本数で混ぜる比を決める先例である．**n/(n+1) は「戻し先を 1 本ぶん信じる」ことに当たり，提案の手法で n0 = 1 とした形と同じである．違いは 2 つある．混ぜる対象が特徴の共分散で，融合の重みではない．戻し先はターゲットのタスク全体の値で，ソースの値ではない | ◎ | https://arxiv.org/abs/1912.03432 |
| MLLR の回帰クラスの木（Leggetter & Woodland 1995，Gales 1996 の技術報告） | 話者適応の方法．適応のデータが少ないときは，全ガウス分布に共通の変換 1 つだけを推す．データが増えると分布をクラスに分け，クラスごとの変換を推す．木の各節を，十分なデータがあるところまで下る | 「データの量に応じて推す数を増やす」先例である．共通の変換 1 つは新しい話者のデータで推し，分布どうしの相対的な位置は多数の話者で学んだモデルから持ち込む．この構造は「水準は K 本，相性はソース」と同じ向きである（推測を含む読み） | ○（講義資料と Gales の技術報告の抄録） | https://mi.eng.cam.ac.uk/reports/abstracts/speech/gales_tr263.html |
| Kuhn ほか 2000（IEEE TSAP．eigenvoices） | 多数の話者のモデルから低次元の基底を作る．新しい話者では，基底の係数（K 個．元の次元 D より十分小さい）だけを適応のデータで推す．データが最も少ないときに得が最も大きかった | 「形はソースから持ち込み，少数の数だけをターゲットで推す」古典 | ○（抄録） | https://search.isc.ac/inventory/49/1749674.htm |
| Chen ほか 2019（arXiv 1909.05557．Modular Meta-Learning with Shrinkage．NeurIPS 2020） | ネットワークをモジュールに分け，タスクごとのパラメータを，モジュールごとの中心 φ_m へ分散 σ_m² で縮める．σ_m² は，訓練タスクの検証の分割での予測尤度を最大にして学ぶ（4.1 節）．σ² が小さいモジュールはタスクに依らず再利用され，σ² が大きいモジュールだけが適応する | 「どの部分をターゲットで測り，どの部分を共有の値に固定するか」をデータで決める先例．付録の玩具の解析では，推定値が (N x̄ + φ/σ²)/(N + 1/σ²) になり，n0 = 1/σ² の形である（道具の読み） | ◎（道具経由．3〜5 節） | https://arxiv.org/abs/1909.05557 |
| Evgeniou & Pontil 2004（KDD），Daumé 2007（ACL．Frustratingly easy DA） | タスクの重みを「共有の部分 + タスク固有の部分」に分け，固有の部分を罰則で小さくする．Daumé は特徴を，共有・ソース固有・ターゲット固有の 3 つに複製する | 共有と固有の分解の代表である．罰則で分けるので，どの数をどのデータで測るかは固定しない | ○ | https://dl.acm.org/doi/10.1145/1014052.1014067 ，https://aclanthology.org/P07-1033/ |
| Saerens, Latinne, Decaestecker 2002（Neural Computation），BBSE（Lipton ほか 2018） | ラベルの割合だけがずれると仮定する．ターゲットではクラスの割合（K 個）だけを推し直し，クラスごとの条件付きの分布はソースのままにする | 「少数の数だけをターゲットで推す」別の型．提案の手法の切片の層（ラベル無しで割合をそろえる）は，この系統に当たる | ○（継承） | https://doi.org/10.1162/089976602753284446 |
| Mercier, Quost, Denœux 2008（Information Fusion 9）．classical discounting（Shafer） | 証拠理論の方法．classical discounting は，情報源の信頼性を 1 つの割引率で表す．contextual discounting は，本当のクラスごとに割引率を持つ．割引率のベクトルは，ラベル付きの学習データで決める．割り引いた出力と正解の 0/1 の差の 2 乗の和を最小にする制約つきの最小 2 乗である（5 節） | 「情報源 × クラスの信頼性」で組み合わせる先例で，1 つの数の版（classical）とクラスごとの数の版（contextual）の両方を持つ．ただし，どちらも同じデータで決めている．片方をターゲットで，もう片方をソースで決めることはしていない．実験は 4 つの物体の例だけである | ◎（1・3・5 節） | https://hal.archives-ouvertes.fr/hal-00445473 |

**Q1 の読み**：「ラベルが少ないときは推す数を減らす」原則は，較正（Guo ほか 2017），音声の適応（MLLR，eigenvoices），判定者の表の推定（Paun ほか 2018，Kuncheva & Rodríguez 2014）に共通してある．Guo ほか 2017 の結果は，提案の手法の分け方を支える最も直接の証拠になる．ただし Guo の結果は「学んだクラスごとの数がほぼ一定になった」というもので，クラスごとの差をほかのデータから持ち込むことはしていない．表の出どころを行の平均（水準）と行の中の差（相性）で分け，水準をターゲットの少数ラベルで，相性をソースで決める研究は，見た範囲では見つからなかった．構造が最も近いのは MLLR である．MLLR は，共通の変換を新しい話者のデータで推し，分布の相対的な位置を多数の話者のモデルから持ち込む．

### 3.2 Q2：ソースの外での落ち方で，信頼度や較正を割り引く

| 文献 | 何をしたか | 提案の手法との関係 | 水準 | URL |
|---|---|---|---|---|
| **Yu, Bates, Ma, Jordan 2022（NeurIPS．Robust Calibration with Multi-domain Temperature Scaling）** | 複数のドメインのそれぞれで temperature を測る（「For every domain k, we learn temperature T̂k by applying temperature scaling on validation data」）．次に，特徴の埋め込みから temperature を当てる線形回帰を学び，見ていないドメインの例ごとに temperature を見込む（3 節の手順 1〜3）．ImageNet-C（76 ドメイン），WILDS-RxRx1，GLDv2 で，分布の中と外の両方で既存の方法より較正が良かったと書く | **「ソースのドメインごとに測った数から，新しいドメインでの数を見込む」先例として最も近い．**違いは 3 つある．(1) 見込む量は較正の 1 つの数で，融合の重みではない．(2) Yu は特徴から回帰で見込み，提案の手法は残り方という 1 つの比を掛ける．(3) Yu はターゲットのラベルを使わないが，提案の手法は水準を K 本で測り直す | ◎（3・4 節の該当箇所） | https://arxiv.org/abs/2206.02757 |
| Gong ほか 2021（ICCV．Confidence Calibration for Domain Generalization under Covariate Shift） | 較正を 1 つのドメインではなく複数の較正用のドメインで行い，ターゲットとの分布の差を実質的に縮める．ターゲットのデータは使わない．Office-Home で期待較正誤差が 8.86 点下がったと書く | 複数のソースで較正の数を決め，ドメインの外へ持ち出す先例 | ○（抄録） | https://arxiv.org/abs/2104.00742 |
| Ovadia ほか 2019（NeurIPS．Can You Trust Your Model's Uncertainty?） | 分布がずれると較正が悪くなることを，多くの方法で示した | 「ソースで測った信頼性はターゲットで落ちる」ことの一般の証拠 | ○ | https://arxiv.org/abs/1906.02530 |
| Patil & Parmigiani 2018（PNAS），Miller ほか 2021（accuracy-on-the-line） | Patil は，研究ごとの予測器を，自分以外の研究での成績で重みづける．Miller は，分布の中と外の成績が probit の尺度で直線に乗ることを示した | 残り方の比に近い先例（継承）．Miller の型は「外 ≈ a × 中 + b」で，残り方の比は b = 0 の場合に当たる | ◎（継承） | https://pmc.ncbi.nlm.nih.gov/articles/PMC5856504/ ，https://arxiv.org/abs/2107.04649 |
| Shafer の classical discounting，Mercier ほか 2008 | 情報源の出力を，信頼性の数で弱める | 「中の信頼度 × 残り方」は，モダリティごとに 1 つの割引率で中の信頼度を弱める操作と同じ形である（式から言えること）．ただし割引率は同じドメインのラベルで決めるもので，ドメインをまたぐ落ち方から決めた例は見ていない | ◎（Mercier 2008 の 1・5 節） | https://hal.archives-ouvertes.fr/hal-00445473 |
| Kaya ほか 2017（Image and Vision Computing 65） | EmotiW 2016 の試験で 2 つの版を比べた．2015 年のデータで学んだモデルと融合の重みの表をそのまま使った版は 52.11% で，2016 年のデータで学び直した版は 48.40% だった（表 7） | 融合の重みを別のデータで決めて持ち出した例である．落ち方を測って割り引くことはしていない．差の原因として，著者は 2016 年の学習データで民族の構成が変わったことを挙げる | ◎（5.2 節，表 7） | https://webspace.science.uu.nl/~salah006/imavis_emotiw15_v10.pdf |
| 音声映像の音声認識の流れの重み（Gurban ほか，IBM の Potamianos ほか） | 流れの指数を，取っておいたデータでの誤りを最小にして決める．EPFL の研究は「the choice of a held-out set for training has too big an influence on the results」と書き，その場の信頼性の指標で重みを決める | 取っておいたデータで決めた重みが試験の条件に合わない問題への対処の先例．ドメインの外での落ち方を測る方向ではない | △（検索の断片） | https://infoscience.epfl.ch/record/125044/files/icmi08.pdf |

**Q2 の読み**：「ソースで測った数を，そのまま使わず，見込みを通して新しいドメインに持ち出す」考えは，較正（Yu ほか 2022，Gong ほか 2021）とソースの重み（Patil & Parmigiani 2018）にある．提案の手法の特徴は 2 つある．見込みの道具が「ソースどうしで測った落ち方の比」という 1 つの数であることと，割り引く対象が分類器 × 感情の信頼度であることである．この組み合わせは見つからなかった．なお，提案の手法では水準を K 本で測り直す．このため，残り方が最終の重みに入るのは相性の大きさ（相性 × 残り方）の部分だけである（設計の 3.2 節の式から言えること）．論文では「残り方で信頼度の水準を見込む」と書くより，「残り方で相性の振れ幅を縮める」と書く方が式に合う．

### 3.3 Q3：戻す強さを「ラベル何本ぶん」で表し，別のコーパスで選ぶ

| 文献 | 何をしたか | 提案の手法との関係 | 水準 | URL |
|---|---|---|---|---|
| **Gauvain & Lee 1994（IEEE TSAP），Reynolds, Quatieri, Dunn 2000（DSP 10）．relevance factor** | GMM の MAP 適応（relevance adaptation）では，混合成分 m の適応の重みを α_m = n_m / (τ + n_m) とする．n_m は成分に割り当てたフレームの数の和で，τ が relevance factor である．Lucey & Chen 2003 はこの重みを「a weight used to tune the relative importance of the prior and is calculated via a relevance factor τ」と書き，τ = 16 が良かったとする．データの少ない成分は「remain close to their world model」 | **戻す強さを「データ何個ぶん」で表す形の古典である．**提案の手法の n0 は τ に当たる．違いは 2 つある．戻す対象が特徴の分布の平均で，融合の重みではない．τ は開発データか慣習（16）で決め，別のドメインを仮のターゲットにして選んではいない | ◎（式は Lucey & Chen 2003 の 3 節で確かめた）．Reynolds 2000 と Gauvain & Lee 1994 の本文は ○ | https://www.isca-archive.org/eurospeech_2003/lucey03_eurospeech.pdf ，https://dblp.org/rec/journals/dsp/ReynoldsQD00 |
| Efron & Morris 1975（JASA），James–Stein | グループごとの平均を全グループの平均へ縮める．縮める強さは，グループの間のばらつきから推す（経験ベイズ） | 縮める強さを「ほかのグループのデータ」で決める古典．提案の手法は，強さをほかのコーパスでの成績で選ぶ | ○ | https://doi.org/10.1080/01621459.1975.10479864 |
| Bateni ほか 2020（Simple CNAPS） | 混ぜる比を n/(n+1) に固定した．比を学ぶ版より良かった | 戻す強さを決め打ちにする先例（3.1 節の表） | ◎ | https://arxiv.org/abs/1912.03432 |
| Chen ほか 2019（Modular Meta-Learning with Shrinkage） | 縮める強さ σ² をモジュールごとに，訓練タスクの検証の分割での尤度で学ぶ．Omniglot の少数データの設定で，縮めない版より有意に良かったと書く（5.3 節）．音声合成では，全モジュールを適応する版が「overfit quickly」 | **縮める強さをタスクをまたいでメタ的に決める先例．**違いは 3 つある．決める強さは連続の数で，勾配で学ぶ．強さの単位はラベルの本数ではない．縮める対象は融合の重みではなく，ネットワークのパラメータである | ◎（道具経由） | https://arxiv.org/abs/1909.05557 |
| **Gulrajani & Lopez-Paz 2021（ICLR．In Search of Lost Domain Generalization）** | ドメイン汎化のモデル選択を 3 つ定義した（3.1 節）．学習ドメインの検証，leave-one-domain-out の交差検証（「we train dtr models with equal hyperparameters, each holding one of the training domains out」），試験ドメインの検証（オラクル）である．結果は「model selection with a training domain validation set outperforms leave-one-domain-out cross-validation」で，オラクルは約 2 点上だった | 提案の手法の選び方（評価するコーパスを含まない計算で格子の 1 点を選ぶ）は，LODO の選択に当たる．この論文は LODO の選択が強くないことを示しており，査読で引かれうる | ◎（道具経由．3.1・5.1 節） | https://arxiv.org/abs/2007.01434 |
| Li, Gouk, Hospedales 2022（Finding lost DG） | 理論から，ドメイン汎化の選択には，例ごとの交差検証よりドメインごとの交差検証の方が合うと主張する | 上の結果への反論で，提案の手法の選び方を支える側の文献 | ○（抄録） | https://arxiv.org/abs/2202.00563 |
| Kahou ほか 2015（EmoNets．ICMI 2013 の拡張） | モデル × 感情の重みを検証データの上で探した（3.4 節）．重みを 2 桁に丸めた理由を「to help to avoid overfitting on the validation set」と書く．乱数探索で得た重みは「varied a lot from one run to another」 | 融合の重みの選び方が過適合しやすいことを，感情認識で記録した先例 | ◎（道具経由） | https://arxiv.org/abs/1503.01800 |
| Tommasi ほか 2014，Mansour ほか 2021 | ソースの重みはターゲットの少数ラベルで選ぶ方がほぼ最適であると示した | 提案の手法は，n0 と相性の使い方をターゲットのラベルで選ばない．ターゲットのラベルで選ぶ版を，比べる相手として求められうる（継承） | ◎（継承） | https://arxiv.org/abs/2007.09762 |

**Q3 の読み**：戻す強さをデータの個数で表す形（relevance factor，Simple CNAPS）も，縮める強さをほかのタスクやグループで決める形（経験ベイズ，Chen ほか 2019）も既知である．評価するコーパスを含まない計算で設定を選ぶ手続きは，ドメイン汎化の LODO の選択である．融合の重みの縮小の強さ（ラベル何本ぶん）と相性を使うかを，別のコーパスを仮のターゲットにして選んだ例は見つからなかった．Gulrajani & Lopez-Paz 2021 は，LODO の選択が学習ドメインの検証より弱い場合を報告している．査読でこの点を問われうる．

### 3.4 Q4：感情認識の後段の融合と，比べる相手の出典

#### (a) 感情認識で，モダリティ（モデル）× 感情の重みを検証データで決めたもの

| 文献 | 設定 | 何をしたか | 提案の手法との関係 | 水準 | URL |
|---|---|---|---|---|---|
| **Kaya, Gürpınar, Salah 2017（Image and Vision Computing 65:66–75）** | AFEW（EmotiW 2015・2016），7 感情．学習・開発・試験の分割はコーパスの中 | 2 つの後段の融合を比べた（3.5 節）．1 つ目は，2 つのモデルの点を γ と 1 − γ で混ぜる単純な重みつきの融合で，γ は開発データで決める．2 つ目は「weighted score fusion (WF) for each model and class」で，重み W_ij（モデル i，クラス j．各クラスでモデルの重みの和が 1）を「searched over a pool of randomly generated matrices」．著者は，この方式は各部分の取り違えの表を考慮しており，以前の EmotiW で成功したと書く．EmotiW 2016 では，開発 57.70% の版が試験 46.21%，開発 57.02% の版が試験 52.11% だった（表 7）．音声の重みは Anger と Fear で最も高かった | **提案の手法の「分類器 × 感情の重みで点を足す」形の，感情認識での直接の先例である．**違いは 3 つある．(1) Kaya は重みを開発データ（数百本）の上の乱数探索で決める．提案の手法は 12 個を直接探さず，水準 3 個を K 本で測り，相性をソースから持ち込む．(2) Kaya はコーパスの中で評価し，クロスコーパスではない．(3) Kaya が足すのは点（スコア）で，提案の手法が足すのは中心化した対数確率である | ◎（3.5 節，5.2 節，表 7） | https://webspace.science.uu.nl/~salah006/imavis_emotiw15_v10.pdf |
| **Kahou ほか 2015（EmoNets．arXiv 1503.01800．ICMI 2013 の EmotiW の優勝の拡張）** | AFEW2・AFEW4，7 感情 | 顔・音声・動き・口の領域のモデルの予測を融合した．全モデルの平均は検証 40.15%・試験 37.17%．SVM と MLP の融合は過適合した．モデル × 感情の倍率を 0〜3 の格子で探すと，検証は 43.69% に上がったが試験は 32.69% に下がった．感情ごとに単体の上で重みを乱数探索すると，検証 47.98%・試験 39.42% だった．結論として「it may be better practice to search a moderate space of simple combination models」と書く | 感情ごとのモデルの重みを検証データで決めた最初期の例である．検証と試験の差が大きいこと（格子の探索では 11 点）を記録している．提案の手法が 12 個を K 本から直接決めない理由の先例として引ける | ◎（道具経由） | https://arxiv.org/abs/1503.01800 |
| Busso ほか 2004（ICMI） | 女優 1 人，4 感情 | 取り違えの表から作った重みの融合 0.86，確率の積 0.89 | 継承 | ◎（継承） | https://sail.usc.edu/publications/files/Busso_2004.pdf |
| Lingenfelser, Wagner, André 2011（ICMI） | DaFEx，CALLAS | 16 種の融合を比べ，差は小さかった | 継承 | ◎（継承） | https://opus.bibliothek.uni-augsburg.de/opus4/files/45462/45462.pdf |
| Marín ほか 2026（arXiv 2609.09924） | MELD，IEMOCAP | 発話ごとにモダリティ × 感情の点を MLP で出し，全ラベルで学ぶ | 継承 | ◎（継承） | https://arxiv.org/abs/2609.09924 |
| Wen, Tseng, Chen 2026（arXiv 2609.13857．ReH-FUSE） | IEMOCAP，MELD | 判定の段で，言葉と音声の相対の重みと，モダリティをまたぐ専門家の重みを発話ごとに出す | 重みは発話ごとである．感情ごとの重みを報告しているかは確かめられなかった | △（抄録と検索の断片．検索の要約にあった「重みは感情で変わる」「分布のずれで劣化しうる」の文は，抄録の頁では確かめられなかった） | https://arxiv.org/abs/2609.13857 |
| 「Explicit Modality Weighting With Unimodal Supervision」（IEEE TAC 2026） | 不明 | 前回に続き，本文も抄録も取得できなかった | 未確認 | 未確認 | https://www.computer.org/csdl/journal/ta/2026/03/11589339/2hKbBClYQQE |

#### (b) 比べる相手（既知のまとめ方）の出典

| まとめ方 | 出典 | 確かめた中身 | 水準 | URL |
|---|---|---|---|---|
| 確率の平均（和の規則）と積の規則 | Kittler, Hatef, Duin, Matas 1998（IEEE TPAMI 20(3)） | 積の規則と和の規則を同じ枠から導いた．実験では「the combination rule developed under the most restrictive assumptions—the sum rule—outperforms other classifier combination schemes」で，理由として推定の誤りへの感度の解析を挙げる | ◎（抄録と 2 節の見出しを PDF で確かめた） | https://people.cs.pitt.edu/~chang/231/y14/combineclassifier.pdf |
| 重みつき多数決 | Nitzan & Paroush 1982，Kuncheva & Rodríguez 2014 | 重みは正解率の対数オッズ log(p/(1−p)) | ◎（Kuncheva & Rodríguez，継承）．Nitzan & Paroush は ○ | https://lucykuncheva.co.uk/papers/lkjrKAIS14.pdf |
| naive Bayes combiner | Xu, Krzyzak, Suen 1992，Kuncheva, Bezdek, Duin 2001 | 各分類器の取り違えの表を使って判定を本当のクラスの確率に読み替え，掛け合わせる | ◎（Kuncheva ほか 2001，継承）．Xu ほか 1992 は ○ | https://www.lucykuncheva.co.uk/papers/lkpr01.pdf |
| stacking | Wolpert 1992，Ting & Witten 1999（JAIR 10） | Ting & Witten は，第 1 段のモデルが出すクラスの確率を第 2 段の入力にし，第 2 段に多出力の最小 2 乗の線形回帰を使う．「We find the use of class probabilities to be crucial for the successful application of stacked generalization in classification tasks」．第 2 段の学習データは J 分割の交差検証で作る | ◎（Ting & Witten の 1〜2 節）．Wolpert は ○ | https://arxiv.org/abs/1105.5466 |
| 重みつき対数線形プール（重みつき積） | Heskes 1997 | 誤差が「個々の誤差の重みつき平均 − 専門家どうしの違い」に分かれる | ◎（継承） | https://papers.nips.cc/paper/1997/hash/59f51fd6937412b7e56ded1ea2470c25-Abstract.html |
| 分類器 × クラスの重みつき和 | Kaya ほか 2017（WF），Kahou ほか 2015 | 上の (a) | ◎ | 上と同じ |

**Q4 の読み**：本番で並べる「既知のまとめ方」の候補は 5 つある．確率の平均と等しい重みの積（Kittler 1998），重みつき多数決（Kuncheva & Rodríguez 2014），naive Bayes combiner（Kuncheva ほか 2001），stacking（Ting & Witten 1999）である．これに，**分類器 × 感情の重みを検証データ（ここでは K 本）で探す方法（Kaya ほか 2017 の WF）**を加えるのが自然である．提案の手法の形に最も近い既知の方法がこれだからである．ただし K = 5 では探索に使えるラベルが 20 本しかない．Kahou ほか 2015 の過適合の記録から，この版の成績は低く出ると見込まれる（推測）．

---

## 4. 最も近い先行研究 5 本との比較

「達成物」は，研究目標が達成された形を指す．達成物の手順は次の 4 段である．(1) ソースで測った分類器 × 感情の信頼度を，残り方で縮めて相性を作る．(2) ターゲットの K 本で水準を測り，戻し先へ n0 本ぶん戻す．(3) n0・戻し先・相性を使うかを，別のコーパスを仮のターゲットにして選ぶ．(4) できた表の重みで，声・顔・言葉の点を感情ごとに重みづけて足す．

### 4.1 Kaya, Gürpınar, Salah 2017 ◎

| 項目 | 先行研究 | 達成物 |
|---|---|---|
| データ | AFEW（EmotiW 2015・2016）．コーパスの中 | 4 コーパス（下調べ）・6 コーパス（後続）の leave-one-corpus-out |
| ラベルの量 | 開発データ全部（数百本）で重みを探す | ターゲットは感情ごとに K = 5・10・25 本 |
| モダリティ | 音声と，顔の特徴の複数のモデル | 声・顔・言葉 |
| 学習器 | kernel ELM，PLS．融合は点の重みつき和 | 凍結した特徴の上の線形の分類器．融合は中心化した対数確率の重みつき和 |
| 測る量 | 開発と試験の正解率 | UAR |
| 主張 | モデル × クラスの重みの融合で，EmotiW の最高水準の成績を出した | — |
| 同じ点 | 分類器 × 感情の重みで点を足す．重みを感情ごとに分類器の間で比べる | — |
| 違う点 | 重み（モデルの数 × 7 個）を乱数探索で直接決める．重みの出どころは開発データ 1 つ | 12 個を直接探さない．水準 3 個はターゲットの K 本で，相性はソースで測る．選ぶのは 3 つの数（n0，戻し先，相性を使うか）だけ |

### 4.2 Guo, Pleiss, Sun, Weinberger 2017 ◎

| 項目 | 先行研究 | 達成物 |
|---|---|---|
| データ | 画像と文書の 10 以上のデータ（CIFAR，ImageNet，20 News など） | 上と同じ |
| ラベルの量 | 検証データ（数千〜数万本） | K 本 |
| モダリティ | 無し（1 つのモデル） | 3 つ |
| 学習器 | 深いネットワーク．較正は学習の後に足す | 上と同じ |
| 測る量 | 期待較正誤差 | UAR |
| 主張 | 1 つの数の temperature scaling が，クラスごとの数の vector scaling と行列の matrix scaling より良い．較正のずれは低次元である | — |
| 同じ点 | データが少ないときは推す数を減らす．クラスごとの数より全体の 1 つの数が良い | — |
| 違う点 | 1 つのモデルの較正．クラスごとの差は捨て，ソースから持ち込まない | 3 つのモダリティを組み合わせる重み．クラスごとの差（相性）をソースから持ち込む |

### 4.3 Paun ほか 2018 ◎

| 項目 | 先行研究 | 達成物 |
|---|---|---|
| データ | 注釈のデータ（文の含意，語義，固有表現など） | 上と同じ |
| ラベルの量 | 注釈者ごとに少数〜多数の注釈．正解は推す | K 本 |
| モダリティ | 無し（判定者は注釈者） | 3 つ |
| 学習器 | Dawid–Skene 系の 6 つの模型（HMC で推す） | 線形の分類器と信頼度の表 |
| 測る量 | 正解との一致，予測の尤度 | UAR |
| 主張 | 部分プーリングの模型は過適合を避け，データの少ない注釈者で特に役立つ | — |
| 同じ点 | 判定者 × クラスの表を，データが少ないときは母集団の値へ寄せる | — |
| 違う点 | 表の全部を同じ仕組みで寄せる．母集団は同じデータの中の注釈者 | 行の平均はターゲットで測り，行の中の差は別のコーパス（ソース）の値に固定する |

### 4.4 Yu, Bates, Ma, Jordan 2022 ◎

| 項目 | 先行研究 | 達成物 |
|---|---|---|
| データ | ImageNet-C（76 ドメイン），RxRx1，GLDv2 | 上と同じ |
| ラベルの量 | 各ソースのドメインの検証データ．ターゲットのラベルは使わない | ソースの全ラベルと，ターゲットの K 本 |
| モダリティ | 無し | 3 つ |
| 学習器 | ResNet-50 など．temperature を特徴から線形回帰で見込む | 線形の分類器．信頼度を中 × 残り方で見込む |
| 測る量 | 分布の中と外の期待較正誤差 | UAR |
| 主張 | 複数のドメインの違いを使うと，見ていないドメインでも較正が良くなる | — |
| 同じ点 | ソースのドメインごとに数を測り，見ていないドメインでの数を見込む | — |
| 違う点 | 見込むのは較正の 1 つの数で，見込みは入力の特徴に依る | 見込むのは分類器 × 感情の信頼度で，見込みはソースどうしの落ち方の比で入力に依らない．水準はターゲットで測り直す |

### 4.5 Bateni ほか 2020（Simple CNAPS）と relevance factor（Reynolds ほか 2000）◎・○

| 項目 | 先行研究 | 達成物 |
|---|---|---|
| データ | Meta-Dataset（Simple CNAPS）．話者照合の音声（relevance factor） | 上と同じ |
| ラベルの量 | クラスごとに数本（Simple CNAPS），話者ごとに数秒〜数分（話者照合） | K 本 |
| 学習器 | 共分散の推定（Simple CNAPS），GMM の平均（MAP 適応） | 信頼度の水準 |
| 主張 | ラベルの本数 n から，n/(n+1)（Simple CNAPS）や n/(n+τ)（τ = 16）の比で混ぜる．Simple CNAPS では，学んだ比より固定の比が良かった | — |
| 同じ点 | 戻す強さをデータの個数で表し，個数が増えるほど自分のデータを信じる | — |
| 違う点 | 強さは固定か開発データで決め，別のドメインを仮のターゲットにして選ばない．戻すのは特徴の統計 | 戻すのは融合の重みの水準．強さ（0・20・80・∞ 本ぶん）を別のコーパスで選ぶ |

---

## 5. 論文に書けること・書けないこと

### 5.1 書けないこと（先例があるので，新しさとして主張しない）

- **分類器（モダリティ）× 感情の重みで点を足す形．**感情認識では，Kaya ほか 2017 と Kahou ほか 2015 が AFEW で既に行っている．融合の一般の文献では，Mercier ほか 2008 の contextual discounting と Kuncheva & Rodríguez 2014 の recall combiner がある．Busso ほか 2004 と Marín ほか 2026 も，引き継いだ報告にある．
- **データが少ないときは推す数を減らすこと．**Guo ほか 2017，MLLR の回帰クラスの木，eigenvoices，Kuncheva & Rodríguez 2014 がある．
- **判定者 × クラスの表を，データが少ないときは別の値へ寄せること．**Paun ほか 2018，Kim & Ghahramani 2012 がある．
- **戻す強さをデータの個数で表すこと．**MAP 適応の relevance factor（Gauvain & Lee 1994，Reynolds ほか 2000）と，Simple CNAPS の n/(n+1) がある．
- **縮める強さをほかのタスクやグループで決めること．**経験ベイズ（Efron & Morris 1975）と Chen ほか 2019 がある．
- **評価するドメインを含まない計算で設定を選ぶこと．**ドメイン汎化の leave-one-domain-out の選択（Gulrajani & Lopez-Paz 2021）がある．
- **ソースで測った数を，見込みを通して新しいドメインに持ち出すこと．**Yu ほか 2022，Gong ほか 2021，Patil & Parmigiani 2018 がある．
- **情報源の信頼性を 1 つの数で割り引くこと．**証拠理論の discounting がある．

### 5.2 書けること（今回の検索で見つからなかった組み合わせ．1 つずつの新しさは小さい）

1. **信頼度の表の出どころを，パラメータの数で分ける組み立て．**表を水準（分類器ごとに 1 つ，計 3 個）と相性（行の中の差）に分け，水準をターゲットの K 本で，相性をソースで測る．Guo ほか 2017 は「データが少ないときは 1 つの数だけを推す」ことを示したが，クラスごとの差をほかのデータから持ち込んではいない．Paun ほか 2018 は表の全部を一様に寄せる．行の平均と行の中の差で出どころを分けた融合の重みは見つからなかった．論文では，Guo ほか 2017 と MLLR を引いて「既知の原則を，モダリティ × 感情の融合の重みに当てはめた」と書くのが正確である．
2. **分け方の根拠の記述．**分け方の理由として，2 つの測定を並べる．1 つは，水準（どのモダリティが全体に強いか）がコーパスによって入れ替わることである．ソースで決めた最も強いモダリティがターゲットの正解と合う割合は低かった．もう 1 つは，相性の形がコーパスをまたいで比較的保たれることである．感情認識で，この 2 つを別々に測った研究は見つからなかった．前回の報告の 7 節でも，感情ごとのモダリティの信頼度の順位がコーパスをまたいで保たれるかを調べた研究は見つからなかった．
3. **戻す強さを「ラベル何本ぶん」で表し，別のコーパスで選ぶ手順を，融合の重みに使ったこと．**relevance factor と Simple CNAPS は強さを固定にする．Chen ほか 2019 は強さを学ぶが，単位はラベルの本数ではない．
4. **残り方でソースの相性を縮めること．**ソースどうしの落ち方の比を，分類器 × 感情の信頼度の割引に使った例は見つからなかった．ただし 3.2 節の末に書いたとおり，提案の手法では水準を K 本で測り直すので，残り方が効くのは相性の大きさだけである．論文では，この点を式どおりに書く．

### 5.3 査読で問われうる点

- Kaya ほか 2017 の WF（分類器 × 感情の重みを K 本の上で直接探す版）を，比べる相手に入れるかどうか．形が最も近い既知の方法なので，入れないと「既知の方法と比べていない」と指摘されうる．入れるかどうかは研究者が決める．
- Gulrajani & Lopez-Paz 2021 の「LODO の選択は学習ドメインの検証より弱い」という結果．提案の手法の選び方への問いになりうる．
- Kahou ほか 2015 と Kaya ほか 2017 の，検証と試験の成績の差．感情ごとの重みは過適合しやすいという記録であり，提案の手法が 12 個を直接決めない理由として引ける．同時に，相性をソースから持ち込んでも相性の分の得が出ない可能性への問いにもなる．
- 水準の部分は，分類器ごとの 1 つの数を対数確率に掛けるので，モダリティごとの temperature を K 本で決めることに近い（式から言えること）．査読者が「モダリティごとの温度の較正と何が違うか」と問う可能性がある．違いは 2 つある．数を対数尤度ではなく信頼度（d′）から決めることと，相性をソースから足すことである．

---

## 6. 狙うべき差分（MER-DA の記録で既に測れているもの，保存済みの出力から出せるものを優先）

1. **3 つの版の差を，クラスあたり 5〜25 本での Guo ほか 2017 の原則の確かめとして書く．**3 つの版は，水準だけを K 本で測る版，12 個すべてを K 本で測る版，水準は K 本・相性はソースの版である．設計の比べ（R2 の「K 本だけ」「ソースだけ」「水準は K 本＋相性はソース」と，感情で共通の重みの版）は既に計算されている．Guo ほか 2017 の結果は検証データが数千本の設定で得たもので，クラスあたり 5 本の融合の重みで同じ向きの差が出るかを示した研究は見つからなかった．保存済みの出力（設計の 6 節）から，K = 5・10・25 で版の差がどう変わるかを並べられる．K が増えると 12 個を直接測る版が追いつくかどうかは，MLLR の「データが増えたら推す数を増やす」原則の確かめになる．
2. **分け方の根拠になる 2 つの測定を並べる．**(a) 水準の持ち越し：ソースで決めた最も強いモダリティが，ターゲットの正解と合う割合（説明の PDF の 7.3 節にある値）．(b) 相性の持ち越し：相性の形の相関（R0 の `r0.csv`）．どちらも計算済みである．この 2 つを並べた図は，Paun ほか 2018 の部分プーリングで表のどの部分を寄せるべきかを，データで決める根拠として書ける．
3. **選ばれた戻す強さの値を記述する．**`mixsel.csv` には，コーパスごと・K ごとに選ばれた n0・戻し先・相性を使うかが記録されている．結果の文の下書き 75 行によると，K = 5 では n0 は 0 か 20 で，どのコーパスでも相性を使う点が選ばれた．この値を，relevance factor の慣習（16）と Simple CNAPS（クラスあたり 1 本ぶん）と並べて書ける．「ラベル何本ぶん」という単位で比べられることが，この表し方の利点である．
4. **残り方が感情で共通かの記述．**設計の 5 節の仮定 (b) の `ret_emo.csv` に，モダリティごと・感情ごとの残り方がある．Yu ほか 2022 はドメインごとの数を入力の特徴から見込むが，提案の手法は感情で共通の 1 つの比を使う．感情ごとの残り方のばらつきを並べれば，この単純化がどこまで許されるかを示せる．
5. **比べる相手に Kaya 型の版を足すかを判断する材料．**Kaya 型（分類器 × 感情の重みを K 本の上で直接探す版）は，設計の「K 本だけ」の版（12 個すべてを K 本の交差の推定で測る版）に近いが，同じではない．「K 本だけ」の版は重みを探すのではなく，信頼度を測って重みにするからである．本番の設計で比べる相手を決めるときの材料としてここに書く．新しい案ではなく，比べる相手と出典の対応づけである．

---

## 7. 見つからなかったこと

- 信頼度の表（判定者 × クラス）を行の平均と行の中の差に分け，行の平均をターゲットの少数ラベルで，行の中の差を別のドメインのデータで測った研究は，今回の検索では見つからなかった．
- 融合の重みや較正の数を，ソースどうしで測った落ち方の比（外 ÷ 中）で割り引いて，新しいドメインでの値を見込んだ研究は，今回の検索では見つからなかった．
- 融合の重みの縮小の強さを「ラベル何本ぶん」で表し，別のドメイン（コーパス）を仮のターゲットにして選んだ研究は，今回の検索では見つからなかった．
- 感情認識で，モダリティ × 感情の重みをクロスコーパスの設定で決めた研究は，今回の検索では見つからなかった．Kaya ほか 2017 と Kahou ほか 2015 はコーパスの中の評価で，EmotiW の年をまたいで重みを持ち出した例は Kaya ほか 2017 の 1 例だけだった．
- 感情認識で，モダリティごとの全体の強さ（水準）と感情ごとの得意不得意（相性）を分けて，それぞれのコーパス間の持ち越しを測った研究は，今回の検索では見つからなかった．
- 音声映像の音声認識で，感情（クラス）ごとの流れの重みを取っておいたデータで決めた研究は，今回の検索では確かめられなかった．検索で出てきたのは，状態や信頼性の指標で決める重みだった．

## 8. 調査の限界

- 本文を読めなかった文献は次のとおりである．Reynolds ほか 2000 と Gauvain & Lee 1994（式は Lucey & Chen 2003 で確かめた），Leggetter & Woodland 1995 と Gales 1996（講義資料と抄録で確かめた），Kuhn ほか 2000（抄録），Gong ほか 2021（抄録），Ovadia ほか 2019，Efron & Morris 1975，Evgeniou & Pontil 2004，Daumé 2007，Saerens ほか 2002，Li ほか 2022，Wen ほか 2026（ReH-FUSE），IEEE TAC 2026 の「Explicit Modality Weighting」（未確認のまま）．論文で引く前に本文を確かめる．
- Kahou ほか 2015，Chen ほか 2019，Gulrajani & Lopez-Paz 2021 は，HTML を道具に読ませて確かめた．数値は引用の前に PDF で確かめる．
- 被引用の追跡（Google Scholar，Semantic Scholar）はしていない．EmotiW の 2014〜2019 年の上位の論文には，ほかにもモデル × 感情の重みの融合がある可能性が高い．Kaya ほか 2017 は，この方式が「was shown to be successful on previous EmotiW challenges」と書き，2 本を引いている．
- 検索は英語だけで行った．
- 結果の文の下書きは見出しと 1 行だけを読み，`results/` のファイルは開いていない．6 節の「計算済み」は，設計の保存の契約と下書きの見出しから判断した．
- 今回 PDF から取り出した文字はこのセッションの scratchpad（`/tmp/claude-1000/-home-snics-MER-DA/76085a37-4610-4922-b8ae-ced34f163c39/scratchpad/pw3/`）にあり，セッションが終わると消える．

---

## Sources

- [Guo ほか 2017, On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)
- [Kull ほか 2019, Dirichlet calibration](https://arxiv.org/abs/1910.12656)
- [Paun ほか 2018, Comparing Bayesian Models of Annotation（TACL）](https://aclanthology.org/Q18-1040/)
- [Ghahramani & Kim 2003, Bayesian Classifier Combination](https://mlg.eng.cam.ac.uk/zoubin/papers/GhaKim03.pdf)
- [Kuncheva & Rodríguez 2014](https://lucykuncheva.co.uk/papers/lkjrKAIS14.pdf)
- [Kuncheva, Bezdek, Duin 2001](https://www.lucykuncheva.co.uk/papers/lkpr01.pdf)
- [Bateni ほか 2020, Simple CNAPS](https://arxiv.org/abs/1912.03432)
- [Gales 1996, Regression class trees for MLLR（抄録）](https://mi.eng.cam.ac.uk/reports/abstracts/speech/gales_tr263.html)
- [Edinburgh ASR 講義資料（MLLR の適応）](https://www.inf.ed.ac.uk/teaching/courses/asr/2015-16/asr10-adapt.pdf)
- [Kuhn ほか, eigenvoices（書誌）](https://search.isc.ac/inventory/49/1749674.htm)
- [Chen ほか 2019, Modular Meta-Learning with Shrinkage](https://arxiv.org/abs/1909.05557)
- [Evgeniou & Pontil 2004](https://dl.acm.org/doi/10.1145/1014052.1014067)
- [Daumé 2007, Frustratingly Easy Domain Adaptation](https://aclanthology.org/P07-1033/)
- [Saerens ほか 2002](https://doi.org/10.1162/089976602753284446)
- [Mercier, Quost, Denœux 2008, contextual discounting（HAL）](https://hal.archives-ouvertes.fr/hal-00445473)
- [Mercier ほか 2008 の PDF](https://www.hds.utc.fr/~quostben/dokuwiki/_media/fr/2007_informationfusion.pdf)
- [Yu, Bates, Ma, Jordan 2022, Multi-domain Temperature Scaling](https://arxiv.org/abs/2206.02757)
- [Gong ほか 2021, Confidence Calibration for Domain Generalization](https://arxiv.org/abs/2104.00742)
- [Ovadia ほか 2019](https://arxiv.org/abs/1906.02530)
- [Patil & Parmigiani 2018（PMC）](https://pmc.ncbi.nlm.nih.gov/articles/PMC5856504/)
- [Miller ほか 2021, accuracy-on-the-line](https://arxiv.org/abs/2107.04649)
- [Gurban ほか, AVSR の流れの重み（ICMI 2008）](https://infoscience.epfl.ch/record/125044/files/icmi08.pdf)
- [Lucey & Chen 2003（relevance factor の式）](https://www.isca-archive.org/eurospeech_2003/lucey03_eurospeech.pdf)
- [Reynolds, Quatieri, Dunn 2000（dblp）](https://dblp.org/rec/journals/dsp/ReynoldsQD00)
- [Efron & Morris 1975](https://doi.org/10.1080/01621459.1975.10479864)
- [Gulrajani & Lopez-Paz 2021, In Search of Lost Domain Generalization](https://arxiv.org/abs/2007.01434)
- [Li, Gouk, Hospedales 2022, Finding lost DG](https://arxiv.org/abs/2202.00563)
- [Mansour ほか 2021](https://arxiv.org/abs/2007.09762)
- [Kaya, Gürpınar, Salah 2017（preprint）](https://webspace.science.uu.nl/~salah006/imavis_emotiw15_v10.pdf)
- [Kahou ほか 2015, EmoNets](https://arxiv.org/abs/1503.01800)
- [Busso ほか 2004](https://sail.usc.edu/publications/files/Busso_2004.pdf)
- [Lingenfelser, Wagner, André 2011](https://opus.bibliothek.uni-augsburg.de/opus4/files/45462/45462.pdf)
- [Marín ほか 2026](https://arxiv.org/abs/2609.09924)
- [Wen, Tseng, Chen 2026, ReH-FUSE](https://arxiv.org/abs/2609.13857)
- [IEEE TAC 2026, Explicit Modality Weighting（未確認）](https://www.computer.org/csdl/journal/ta/2026/03/11589339/2hKbBClYQQE)
- [Kittler ほか 1998, On Combining Classifiers](https://people.cs.pitt.edu/~chang/231/y14/combineclassifier.pdf)
- [Ting & Witten 1999, Issues in Stacked Generalization](https://arxiv.org/abs/1105.5466)
- [Heskes 1997](https://papers.nips.cc/paper/1997/hash/59f51fd6937412b7e56ded1ea2470c25-Abstract.html)

---

**読んだ入力**：
- `/home/snics/MER-DA/DAI/docs/実験16_下調べ_設計.md`
- `/home/snics/MER-DA/DAI/docs/説明_実験16の案_図解_2026-11-13.tex`（5〜8 節）
- `/home/snics/MER-DA/research_notes/先行研究_信頼度の表_2026-11-13/report.md`
- `/home/snics/MER-DA/research_notes/先行研究_ソースの重み付き出発点と残り方_2026-11-14/report.md`
- `/home/snics/MER-DA/DAI_paper/CLAUDE.md`，`/home/snics/MER-DA/DAI_paper/inventory.md`（検索した範囲）
- `/home/snics/MER-DA/DAI/docs/実験16_下調べ_結果の文の下書き_2026-11-14.md`（見出しと 75 行だけ）

**保存先（本体が保存する）**：`/home/snics/MER-DA/research_notes/先行研究_水準はK本・相性はソース_2026-11-14/report.md`
