# DA・転移学習の理論が定義する量のうち，少数のターゲットラベルと複数ソースの設定で学習法になっているもの・なっていないもの（2026-10-28 調べ）

確かめた水準の記号：◎ 本文を直接読んだ（手元の PDF か pdftotext で取り出した本文），○ 抄録を読んだか，本文を取得して要約器に質問した（要約器経由の本文は ○ とし，その旨を書く），△ 書誌か検索結果の抜き書きだけ．「既存ノート ◎」は `docs/related-work.md` と `research_notes/TAC 論文 先行研究の精読と穴/` で本文を読み済みの文献で，ここでは学習法の有無の観点だけを足す．

「学習法として定式化されていない」は，下の探索の記録に書いた検索語と場所で探して見つからなかった範囲での判定である．

## 総括：量ごとの判定表

### Takeaway
理論が定義する量のうち，**ターゲットのラベルを使う「ラベル込みのずれ」を上界に入れて重みを最適化する学習法（Best-effort の sbest，Konstantinov & Lampert，Mansour 2021），ソースの重みを上界の最小化で決める学習法（DARN），高次元の閉じた形に推定値を差し込む学習法（Tiomoko 2021，Zhang 2026 の TLDA-E）は既にある**．一方，Ben-David 2010 の λ を K 本で推定して使う学習法，Ben-David 2010 の閉じた形の α* に推定値を差し込む学習法，転移指数 ρ・γ を推定する学習法は見つからなかった．どの量についても，感情認識（音声・顔・言葉）で使った例は見つからなかった（EEG の多ソースの重みづけと，音声の最適輸送には例がある）．

### Cited Findings

#### 判定表

| 量（出典） | (a) 学習法 | (b) 手法名 | (c) 無いときの障害 | (d) 感情認識 | 水準 |
|---|---|---|---|---|---|
| λ：両ドメインで良い共通の分類器の誤差（Ben-David ほか 2010） | 未発見（λ そのものを K 本で推定して重みや選択に使うもの） | 近いもの：標本版 λ̃ = min_η R̂_S(η)+R̂_T(η) を解析に使う（Sicilia 2022），予測器の食い違いを半教師ありで小さくする（LIRR 2021），擬似ラベルで結合誤差を下げる（CDEM 2021），λ を含まない測れる項に置き換える（Tachet des Combes 2020，Best-effort 2023） | 原論文が「λ を正確に推定できるほどターゲットのラベルがあれば適応は要らない」と書く．ターゲットの項を K 本で見積もると過学習で 0 に近づく（本記録の推論は下） | 見つからず | ◎（Ben-David），○（Sicilia，要約器），△（LIRR，CDEM） |
| α*：上界を最小にする混合比（Ben-David 2010 Thm 3，Blitzer 2007） | 閉じた形の α* に推定値を差し込む学習法は未発見．上界を α で最小化する学習法はある | DARN（多ソース，ラベル無しのターゲット），Best-effort の sbest（ターゲットのラベル込み），Konstantinov & Lampert 2019，Jain 2024 の尺度則からの α*，gapBoost | α* の式は λ と VC 次元 d を含み，d/A² が数千本になるほど上界が緩い（Ben-David の例で D² ≈ 3,130） | 見つからず（EEG の多ソースの重みづけはある） | ◎（Ben-David），○（DARN・Best-effort，要約器），既存ノート |
| 転移指数 ρ（Hanneke & Kpotufe 2019），γ（Kpotufe & Martinet 2021） | ρ・γ を推定する学習法は未発見．知らずに率を合わせる（適応する）学習法はある | Hanneke & Kpotufe 2019 の Algorithm 1（制約つき ERM），Cai & Wei 2021，Reeve ほか 2021，木の剪定（Galbraith & Kpotufe 2023） | 推定にはターゲットのラベルが多く要る．モデル選択を伴うと，適応の率はオラクルの率より任意に遅くなりうる（Hanneke, Kpotufe, Mahdaviyeh 2023） | 見つからず | ○（2023 抄録），△，既存ノート |
| ソースの価値（どのソースが役に立つか） | ある（統計の高次元回帰） | Trans-Lasso（Li, Cai, Li 2022），Trans-GLM のソース検出（Tian & Feng），Hanneke & Kpotufe 2022 の順位による集約 | 線形回帰・GLM の係数の差（ℓ1）で定義され，分類のラベルの付け方のずれとは定義が違う | 見つからず | △ |
| discrepancy（Mansour ほか 2009），ラベル込みのずれ（Mohri & Muñoz Medina 2012） | ある | Best-effort の best・sbest（ラベル込みの discrepancy を重み q と同時に最適化），Mansour 2021（混合比を少数ラベルで選ぶ），Konstantinov & Lampert 2019，ドリフトの QP（Mohri & Muñoz Medina 2012），generalized discrepancy（Cortes ほか 2019） | ラベル込みのずれの推定は仮説集合の上の sup で，DC 計画などの非凸の最適化が要る | 見つからず | ○（Best-effort，要約器），△ |
| 多クラスの MDD（Zhang ほか 2019） | ラベル無しの学習法として定式化済み（MDD）．ターゲットの少数ラベルで MDD の λ を扱う学習法は未発見 | MDD | λ は小さいと仮定したまま残る（既存ノート） | MDD を音声の感情認識に使った例は見つからず．音声では最大回帰食い違い（MRD）がある | 既存ノート，△ |
| 高次元の線形の頭の厳密な漸近の式 | ある（回帰・LDA・LS-SVM） | Tiomoko 2021 の最適スコア y* の差し込み，Zhang 2026 の TLDA-E（ŵ* ∝ A⁻¹u の差し込みと切片の補正），WONDER（重みつき平均，重みの和が 1 を超える），Ha & Chen の MASFT（候補を少数のターゲットで選ぶ），Obst 2021（得の検定） | 多クラスのロジスティック回帰と切片・L2 罰則・少数ショット（4K ≪ p）の組で閉じた形を差し込む学習法は見つからず | 見つからず | 既存ノート ◎，○，△ |
| 仮定（共変量シフト，ラベルシフト，ラベルの付け方のずれ）の検定 | ある | BBSE の検定（ラベル無し），DetectShift（X, Y, X\|Y, Y\|X のずれを別々に検定．ターゲットのラベルが要る），Hu & Lei 2024（P(Y\|X) の一致の検定），Podkopaev & Ramdas 2022（害のあるずれの逐次検定） | 不可能性の定理（Ben-David, Lu, Luu, Pál 2010）は，ラベル無しのデータからは区別できない 2 つの課題で重みづけが一方を助け他方を壊す例を示す．したがって P(Y\|X) についての検定にはターゲットのラベルが要る | 音声の個人適応でラベルの分布のずれを KL で測った例はある（検定ではない） | ○（DetectShift），△ |
| 弱から強の misfit（Charikar 2024 ほか） | 部分的にある | Charikar 2024：misfit で使う弱いモデルを選ぶ．Selective W2S（弱いラベルを使うかを分類器で決める）．Ildiz 2025：最適な代理モデル（先生）の形を導く | 定理は 2 乗損失と凸の関数族が前提（Mulgund & Pabbaraju 2025 が他の損失へ広げた） | 見つからず | ○（Charikar 抄録），△ |
| PAC-Bayes の DA の上界（Germain ほか 2013・2016・2020，Sicilia 2022） | ある（ラベル無し，線形分類器） | Germain 2013 の線形分類器の算法，Germain 2016 の比のずれの上界からの算法，2020 の 2 つの算法 | Sicilia 2022 の多クラスの上界は解析にだけ使う．適応性の項はターゲットのラベルが無いと計算できない | 見つからず | ○（Sicilia，要約器），△（Germain） |
| 情報理論の上界（Wang & Mao 2023） | ある（ラベル無し） | 勾配の罰則（GP），ラベル情報の制御（CL） | 条件付き分布のずれ D_KL(P_{Y'\|T'}‖P_{Y\|T}) はラベル無しでは最小化できないと著者が書く | 見つからず | ◎ |
| 最適輸送の上界（Redko ほか 2017，JDOT） | ある | JDOT，重みつき JDOT（多ソース） | — | 音声の感情認識で最適輸送を使った例（SOT，Interspeech 2023）はある．JDOT を感情認識に使った例は見つからず | △ |

#### 判定の根拠のうち，表の全体にかかるもの

- Ben-David ほか 2010 の Theorem 3 は α* を m_T，m_S と D = √d/A の関数として閉じた形で与え，m_T ≥ D² = d/A² ならソースは要らない（α* = 1）．A は推定した H Δ H 距離の半分と標本の項と λ の和である．数値例は d = 1,601，A = 0.715 で D² ≈ 3,130 — [Ben-David ほか 2010（手元の PDF，◎）](https://link.springer.com/article/10.1007/s10994-009-5152-4)
- 同論文 7.2 節は λ について "in order to estimate λ, we must estimate εT(h*) independently of the source data. If we had enough target data to do this accurately, we would not need to adapt a source classifier in the first place. For our sentiment task, however, λ is small enough to be a negligible term in the bound. Thus we ignore it here." と書き，実験では λ を落とした — [Ben-David ほか 2010（◎）](https://link.springer.com/article/10.1007/s10994-009-5152-4)
- DA の理論の総説は Redko, Morvant, Habrard, Sebban, Bennani の arXiv 2004.11829（v6，2022-07．書籍 Advances in Domain Adaptation Theory の縮約版で，章に発散の上界，PAC-Bayes，頑健性，安定性と仮説転移，不可能性の結果がある）と，Kouw & Loog（IEEE TPAMI 43(3):766–785．ラベル無しの DA を標本の重みづけ・特徴の変換・推定への組み込みに分ける）である．どちらも本文は読んでいない — [Redko ほか（△）](https://arxiv.org/abs/2004.11829)，[Kouw & Loog（△）](https://arxiv.org/abs/1901.05335)

#### 探索の記録（検索語と場所）

- 場所：一般のウェブ検索（WebSearch．米国の検索エンジン），arXiv の抄録の頁，ar5iv の本文，PMLR，JMLR，NeurIPS の予稿の頁，ISCA の予稿の頁，Springer（認証の頁に飛ばされて取得できず），CVF の PDF（403 で取得できず）．
- 検索語（この順で使った）：「ideal joint error λ domain adaptation estimate with few target labels algorithm」「Best-effort adaptation Awasthi Cortes Mohri discrepancy labeled target」「Domain Aggregation Networks multi-source domain adaptation Wen Greiner Schuurmans optimal weights bound」「Hanneke Kpotufe Mahdaviyeh limits of model selection under transfer learning COLT 2023」「Obst Ghattas transfer learning linear regression statistical test of gain」「unified framework dataset shift diagnostics DetectShift Polo Izbicki」「Adaptation based on generalized discrepancy Cortes Mohri Muñoz Medina JMLR 2019 labeled target」「PAC-Bayesian domain adaptation bound algorithm DALC Germain Habrard Laviolette Morvant」「information-theoretic analysis of unsupervised domain adaptation Wang Mao ICLR 2023 algorithm regularization」「Trans-GLM transferable source detection Tian Feng high-dimensional GLM transfer learning」「weak-to-strong generalization theory-inspired algorithm misfit selection 2025」「speech emotion recognition margin disparity discrepancy cross-corpus」「semi-supervised domain adaptation learning invariant representations and risks LIRR joint error bound」「adaptive to unknown transfer exponent covariate shift estimator 2024 2025 arXiv」「Ildiz Gozeten Mondelli Oymak high-dimensional analysis knowledge distillation weak-to-strong optimal surrogate」「test covariate shift assumption conditional distribution P(Y|X) equal two-sample test conformal Hu Lei」「multi-source domain adaptation emotion recognition source weighting discrepancy EEG speech facial」「Sheng Dobriban one-shot weighted distributed ridge regression optimal weights high dimensions WONDER」「PAC-Bayesian domain adaptation bounds multiclass learners Sicilia UAI 2022 target labels estimate」「Chuang Torralba Jegelka estimating generalization under distribution shifts domain-invariant representations proxy」「Redko Morvant Habrard Sebban Bennani survey domain adaptation theory open problems λ term」「optimal transport domain adaptation speech emotion recognition cross-corpus JDOT Wasserstein」「transfer learning high-dimensional linear regression Trans-Lasso Q-aggregation informative sources Li Cai Li JRSSB」「Podkopaev Ramdas tracking risk deployed model detecting harmful distribution shifts sequential test labels」「gapBoost transfer learning minimizing performance gap between domains Wang Mendez Cai Eaton NeurIPS 2019」「Mohri Muñoz Medina 2012 new analysis and algorithm learning with drifting distributions Y-discrepancy ALT」「Ben-David Lu Luu Pál impossibility theorems for domain adaptation AISTATS 2010 covariate shift support」「emotion recognition domain adaptation H-divergence generalization bound Ben-David theorem guided cross-corpus speech facial」「estimating the transfer exponent from samples source target classification Kpotufe follow-up」「Hanneke Kpotufe no-free-lunch theorem multitask learning Annals of Statistics adaptive multi-source rates」「speech emotion recognition label shift class prior adaptation cross-corpus EM prior correction」「optimal combination α source target error Ben-David bound plug-in estimated divergence choose α instance weighting experiment」「Kouw Loog "A review of domain adaptation without target labels" IEEE TPAMI 2021」．
- 本文を取得した文献：Ben-David 2010（手元の PDF を pdftotext），Wang & Mao 2023（arXiv の PDF を pdftotext．作業用ディレクトリ `scratchpad/dl_wangmao/`），DARN・Best-effort・Sicilia 2022（ar5iv の本文を要約器で．Best-effort と Sicilia は前半の約 10 万字だけ）．
- 探さなかった語：「semi-supervised MDD」「few-shot λ estimation」「Mai & Liao の最適な損失」「CGMT の多クラスのロジスティック回帰」「Kpotufe & Martinet の後続の能動学習」．どれも本記録の判定を変えうるので Gaps に書く．

### Inferences
- 表の「定式化済み」の多くは，λ を推定する代わりに「ラベル込みのずれ」（ある仮説がソースとターゲットでどれだけ違う損失を出すか）をターゲットの少数ラベルで推定し，それを上界に入れて最適化する形をとる．λ そのものを推定対象にした学習法が見つからないのは，λ が「最良の共通の分類器」という未知の仮説に依存し，ラベル込みのずれのほうが推定量として扱いやすいからだと考える．
- 感情認識で使われた例が無いことは，MER-DA の設定（凍結特徴の線形の頭，7 群，K 本）に移すこと自体が新しい組み合わせになることを意味する．ただし手法の新しさとしては「既存の学習法を移した」と書くことになる．

### Gaps
- Best-effort（arXiv 2305.05816）の掲載先は arXiv の頁に無い．付録 B.3（ラベル込みの discrepancy の推定）と付録 C は読めていない．
- WONDER が未知の量（信号の強さ・雑音の分散）をデータから推定して重みを計算するかは抄録に書かれておらず，確かめていない．
- Cortes, Mohri, Muñoz Medina 2019（generalized discrepancy）がターゲットのラベル付きの少数標本を使うかは抄録に無く，確かめていない．
- 「semi-supervised MDD」「多クラスのロジスティック回帰の比例極限の差し込み（CGMT，Mai & Liao）」は探していない．

## λ（ラベルの付け方のずれ）を K 本で推定または最小化する学習法はあるか．MDD などでの扱い

### Takeaway
λ そのものを少数のターゲットラベルで推定し，混合比やソースの選択に使う学習法は見つからなかった．既存の手法は (i) λ が小さいと仮定する（Ben-David 2010 の実験，MDD），(ii) 標本版の λ̃ をターゲットのラベルで計算して解析にだけ使う（Sicilia 2022），(iii) λ に近い項（予測器の食い違い）を半教師ありで小さくする（LIRR 2021）か擬似ラベルで小さくする（CDEM），(iv) λ を含まない測れる項に置き換える（Tachet des Combes 2020，Best-effort 2023）のどれかである．

### Cited Findings
- Ben-David 2010 は λ の推定には εT(h*) をソースと独立に推定する必要があり，それができるほどターゲットのラベルがあれば適応は要らないと書き，感想分類の実験では λ を無視した — [Ben-David ほか 2010（◎）](https://link.springer.com/article/10.1007/s10994-009-5152-4)
- MDD（Zhang, Liu, Long, Jordan 2019）は多クラスのマージン損失で上界を作るが，λ に当たる項は残り，小さいと仮定する — 既存ノート `docs/related-work.md` 2.7 節（◎），[arXiv 1904.05801](https://arxiv.org/abs/1904.05801)
- Sicilia, Atwell, Alikhani, Hwang（UAI 2022，PMLR 180:1824–1834）は Theorem 2 で標本に依存する適応性の項 λ̃_{S,T} = min_{η∈H} R_S(η) + R_T(η) を使い，母集団の λ は実際には推定しにくいと対比する．λ̃ はターゲットのラベルで学んだ η の訓練誤差で上から見積もり，"we assume access to target labels for purpose of studying our assumptions" と書く．上界は学習の目的関数ではなく，DANN の標本の複雑さの解析に使う — [Sicilia ほか 2022（○，本文前半を要約器で）](https://arxiv.org/abs/2207.05685)
- Wen, Greiner, Schuurmans（ICML 2020，DARN）の上界には λ が無く，最も近い η_{H,i}（H が真の模型にどれだけ合うか）は「H を固定すると制御できない」として算法では落とす — [DARN（○，本文を要約器で）](https://arxiv.org/abs/1909.05352)
- Li ほか（CVPR 2021，LIRR）は半教師あり DA で，ターゲットの誤差をターゲットの経験誤差，ソースの経験誤差，特徴の分布のずれ，最適なソースとターゲットの予測器の食い違いに分ける上界を示し，後ろの 2 項を表現と危険度の不変性として同時に学ぶ．ここでの記述は後続論文の要約による — [LIRR（△，検索結果の要約．CVF の PDF は 403 で取得できず）](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Learning_Invariant_Representations_and_Risks_for_Semi-Supervised_Domain_Adaptation_CVPR_2021_paper.html)，[要約した後続論文 arXiv 2509.13792（△）](https://arxiv.org/pdf/2509.13792)
- 結合誤差を擬似ラベルで下げる手法の例として Cross-domain error minimization（arXiv 2106.15057）がある．検索結果の要約では，λ* はターゲットのラベルが無いので普通は計算できないとし，擬似ラベルで第 3 項を下げるのが有望とされる — [arXiv 2106.15057（△）](https://arxiv.org/pdf/2106.15057)，[arXiv 2304.02064（△）](https://arxiv.org/pdf/2304.02064)
- Chuang, Torralba, Jegelka（ICML 2020，PMLR 119:1984–1994）はドメイン不変な予測器の集まりを未知のターゲットのラベルの代わりにして，ラベル無しでターゲットの誤差を推定し，その誤差は代理の予測器のターゲットの危険度で抑えられるとした — [Chuang ほか 2020（△）](https://arxiv.org/abs/2007.03511)
- Tachet des Combes ほか 2020 の分解（ラベル比率のずれ × ソースの釣り合い誤差 + 2(k−1) × クラスごとの予測の分布のずれ Δ_CE）は未知の最適分類器を含まず，「ターゲットに少数ラベルがあれば各項を推定できる」と既存ノートに書いた — 既存ノート `docs/related-work.md` 2.7 節（◎）
- Wang & Mao（ICLR 2023）は "In UDA, since Y′ is completely unavailable to the algorithm A, it is impossible to minimize the misalignment of conditional distributions, i.e. D_KL(P_{Y′|T′}‖P_{Y|T})" と書き，擬似ラベルが誤ると D_KL(P_{Y′|T′}‖Q_{Ŷ′|T′}) が発散して上界を大きくしうると示した — [Wang & Mao 2023（◎，pdftotext）](https://arxiv.org/abs/2210.00706)

### Inferences
- MER-DA の線形の頭で λ̃ を K 本から訓練誤差で見積もると，ターゲットの 4K 点（K ≤ 50 なら 200 点以下）は 768〜1024 次元の空間で一般の位置にあれば線形分離できるので（Cover の計数定理の帰結．本記録の推論），min_η の中のターゲットの項はほぼ 0 になり，λ̃ は「ターゲットの K 本を当てる制約の下でのソースの最小誤差」に近くなる．このため λ̃ を使うには罰則つきの η か，話者で分けた交差検証で外の誤差を見積もることが要る．
- MER-DA の既知の結果（K 本は切片を 1/(2K)，向きを q/K の速さで直す．ラベル無しの中心化で平均のずれが消える．ラベル無しでは向きを直せない）は，λ を「中心化と切片で消える部分」と「向きのずれの部分」に分けて推定できる可能性を示す．後者が Wang & Mao の「ラベル無しでは最小化できない条件付きのずれ」に対応すると読める．

### Gaps
- 「λ を K 本で推定し，その推定値で混合比やソースを決める」学習法が本当に無いかは，「few-shot ideal joint error estimation」「semi-supervised DA λ estimate」の語で探し切っていない．
- LIRR の上界の正確な形と，最適予測器の食い違いを K 本でどう推定するかは原典で確かめていない．
- MDD を少数のターゲットラベルで使う半教師あり版は探していない．

## Ben-David 2010 Theorem 3・Blitzer 2007 の上界を最小にする混合比 α* を，未知の項を推定して直接使う学習法はあるか

### Takeaway
Ben-David 2010 の閉じた形の α*（式 (2)）に H Δ H 距離と λ の推定値を差し込んで重みを決める学習法は見つからなかった．上界そのものを重みについて最小化する学習法は，ラベル無しのターゲットで多ソースの重みを決める DARN（ICML 2020）と，ターゲットのラベルでラベル込みの discrepancy を推定し標本ごとの重み q と仮説を同時に最適化する Best-effort の sbest（arXiv 2023）が定式化済みである．

### Cited Findings
- Ben-David 2010 の α* は m_T ≥ D² で 1，それ以外で min{1, ν}．A = 0（ずれ無し）なら α* = β（全標本を一様に重みづける）．7 節の感想分類で予言を実験と比べたが，λ は落とし，算法としては提案していない — [Ben-David ほか 2010（◎）](https://link.springer.com/article/10.1007/s10994-009-5152-4)
- DARN の上界は ℒ_T(h, f_T) ≤ Σ_i α_i(ℒ_Ŝi(h, f_Si) + disc(T, S_i) + 2ℜ_m(H_Si) + η_{H,i}) + ‖α‖₂ M_S √(log(1/δ)/2m)．disc は 0-1 損失で d_A 距離，回帰で 2 次モーメント行列の差の最大固有値になり，ラベル無しの標本から推定する（分類はドメイン分類器で 2(1 − min ε̂)）．Rademacher の項と η の項は「落としても実験上大きな差がない」として算法から外す — [DARN（○，本文を要約器で）](https://arxiv.org/abs/1909.05352)
- DARN の重みは α* = [z − ν*1]₊/‖[z − ν*1]₊‖₁，z = −g/τ，g_i = ℒ_Ŝi + disc(T, S_i) で，ν* は閉じた形が無く二分探索で求める．温度 τ が疎らさを決める．ターゲットのラベルは使わず，「少数のターゲットのラベルを追加のソースとして入れられる」と述べるが試していない．数字 4 種の平均は DARN 83.38 対 MDAN 81.11，感想分類 4 種は DARN 83.60 対 MDMN 83.61 — [DARN（○，要約器）](https://arxiv.org/abs/1909.05352)，[PMLR v119](https://proceedings.mlr.press/v119/wen20b.html)
- Awasthi, Cortes, Mohri "Best-Effort Adaptation"（arXiv 2305.05816，2023-05-10，v1 のみ）は，ソース m 本（多い）とターゲット n 本（中程度）のラベル付き標本で，ラベル込みの discrepancy dis(P, Q)（式 1）に基づく重みづけの上界を，重みについて一様に示した．算法 best は h と q ∈ [0,1]^{m+n} について Σ q_i[ℓ(h(x_i), y_i) + d_i] + dis(q, p⁰) + λ∞‖q‖∞‖h‖² + λ1‖q − p⁰‖₁ + λ2‖q‖²₂（d_i = dis(P, Q)·1{i はソース}，p⁰ はターゲットの一様分布）を最小にする．sbest は dis(q, p⁰) を ‖q − p⁰‖₁ で上から抑えた簡略版で，実験はすべて sbest．discrepancy の推定は DC 計画で行い，罰則の強さはターゲットのラベル付きデータで交差検証する — [Best-Effort Adaptation（○，本文の前半を要約器で）](https://arxiv.org/abs/2305.05816)
- Best-effort の実験は，人工の線形 2 値分類（ロジスティック損失，ソースに雑音 η = 10%，m = 1000），CIFAR-10 の微調整（ソース 6 クラス，ターゲットは frog・horse・ship・truck，ResNet-18），Civil Comments（BERT-small）．比較相手は「P だけで学ぶ」と gapBoost で，CIFAR-10 の最後の層で sbest 89.62±0.32，P だけ 88.61±0.43，gapBoost 87.1．ソースは 1 つだけを扱う — [Best-Effort Adaptation（○，要約器）](https://ar5iv.labs.arxiv.org/html/2305.05816)
- Wang, Mendez, Cai, Eaton（NeurIPS 2019）はソースとターゲットの性能の差（performance gap）を算法依存の正則化として上界に入れ，boosting の転移学習 gapBoost を導いた．JMLR 24（2023，論文番号 22-0099）で表現学習版 gapMTNN に広げた — [NeurIPS 2019（△）](https://papers.nips.cc/paper/2019/hash/c66dd00e5fc44ba8de89d7713fedcd50-Abstract.html)，[JMLR 22-0099（△）](https://jmlr.org/papers/v24/22-0099.html)
- 既存の調査にあるもの：Konstantinov & Lampert 2019（ラベル込みのずれを信頼できる少数のデータで推定し，上界から重みを決める），Mansour ほか 2021（混合比ごとの学習器を m₀ 本のターゲットのラベルで選ぶ．余分の誤差は Õ(√(p/m₀))），Jain, Montanari, Sasoglu 2024（学習曲線の尺度則から α* = R_or(n)/(R_su + R_or(n)) を予言），Crammer, Kearns, Wortman 2005（上界を最小にするソースの部分集合を選ぶ）— `docs/related-work.md` 2.9 節，`research_notes/ラベル換算の価値で重みを決める先例/report.md` 3.6 節（既存ノート）

### Inferences
- Ben-David 2010 の α* を差し込みで使わない理由は，上界が緩く（d/A² が数千本），λ の推定が要るためだと読める．DARN と Best-effort は，閉じた形の α* を使わずに上界の項（経験誤差 + ずれ）を重みと同時に数値最適化する方向をとった．
- MER-DA の「ソースの価値 n = q e²/(1−e²) 本」を使えば，重みの規則 α = n/(n + K) 型（Jain 2024 の式と同じ形）が閉じた形で書ける．これは Ben-David の α* の式とは別物で，λ と VC 次元を含まない．この比較は本記録の推論である．

### Gaps
- Blitzer 2007 の上界から α を予言する実験が，その後の論文で学習法として使われた例は探し切っていない．
- Best-effort の掲載先と，多ソースへの拡張の有無は確かめていない．

## 転移指数（ρ，γ）とソースの価値を推定し，それに合わせて学習を変える手法はあるか（Hanneke, Kpotufe, Mahdaviyeh 2023 など後続を含む）

### Takeaway
ρ や γ を標本から推定する学習法は見つからなかった．後続の研究は「ρ を知らなくてもオラクルに近い率を出す適応の手続き」を作る方向と，「適応には限界がある」ことを示す方向に進んだ．Hanneke, Kpotufe, Mahdaviyeh（COLT 2023）は，モデル選択を伴うと分布を知らない手続きの率がオラクルの率より任意に遅くなりうることを示した．ソースが役に立つかを判定する手続きは高次元の回帰・GLM（Trans-Lasso，Trans-GLM）にある．

### Cited Findings
- Hanneke, Kpotufe, Mahdaviyeh "Limits of Model Selection under Transfer Learning"（COLT 2023，PMLR 195:5781–5812，arXiv 2305.00152 v4 は 2023-10-12）：仮説集合を選ぶ場合，ソースとターゲットの距離が選んだ集合に依存する新しい複雑さの項が現れ，分布を知らない手続きの率は距離を知るオラクルの率より任意に遅くなりうる．一方，模型の次元だけを知る適切な学習器で適応の上界も与える — [arXiv 2305.00152（○，抄録）](https://arxiv.org/abs/2305.00152)，[PMLR](https://proceedings.mlr.press/v195/hanneke23c.html)
- Hanneke & Kpotufe "A No-Free-Lunch Theorem for MultiTask Learning"（Annals of Statistics 50(6):3119–3143，2022）：すべての課題で最適な分類器が同じでも，分布の情報なしに課題数 N とともに率が良くなる適応の算法は存在しない．課題をターゲットへの距離で順位づけできれば，順位に基づく単純な手続きでほぼ最適に集約でき，最適な集約は同じ分類器を持つ課題でも捨てることがある — [Annals of Statistics（△）](https://projecteuclid.org/journals/annals-of-statistics/volume-50/issue-6/A-no-free-lunch-theorem-for-multitask-learning/10.1214/22-AOS2189.full)，[arXiv 2006.15785](https://arxiv.org/abs/2006.15785)
- Hanneke & Kpotufe 2019 は，ターゲットのラベルを何本集めるかの判断を，分布の母数もずれも推定せずに最小最大で最適に行えると述べる．Algorithm 1（ソースでほぼ最適という制約の下でターゲットの経験誤差を最小化）が適応の手続きである — [NeurIPS 2019（△，検索結果）](https://proceedings.neurips.cc/paper_files/paper/2019/file/b91f4f4d36fa98a94ac5584af95594a0-Paper.pdf)，既存ノート `g5_theory_pooling_multisource.md`（◎）
- Galbraith & Kpotufe（arXiv 2305.04335，2023）は転移指数を緩めた aggregate transfer exponent を使い，分類木の剪定に ICI（交差区間の手続き）を当てて，未知のずれの程度に適応する率を示した — [arXiv 2305.04335（△）](https://arxiv.org/abs/2305.04335)
- Suk & Kpotufe（arXiv 2007.08584）は共変量シフトが未知の文脈つきバンディットで，問題の母数に自動で適応する算法を示した — [arXiv 2007.08584（△）](https://arxiv.org/pdf/2007.08584)
- Cai & Wei 2021 の適応的な分類器と Reeve, Cannings, Samworth 2021 の適応的な転移は既存ノートにある（どちらも ρ に当たる量を推定せずに適応する）— `docs/related-work.md` 2.8 節（◎）
- Li, Cai, Li（JRSSB 84(1):149–173，2022，doi:10.1111/rssb.12479）：役に立つソースが既知なら推定量と予測量の最適性を示し，未知なら非情報的なソースに頑健なデータ駆動の手続き Trans-Lasso を提案した — [IDEAS（△）](https://ideas.repec.org/a/bla/jorssb/v84y2022i1p149-173.html)，[arXiv 2006.10593](https://arxiv.org/abs/2006.10593)
- Tian & Feng（arXiv 2105.14328，Trans-GLM）：役に立つソースが未知のとき，各ソースを 1 つずつ足した得をターゲットだけの基準と比べ，有意な得があるソースを転移可能とする「算法に依らない」検出法を示し，一致性を証明した．R パッケージ glmtrans — [arXiv 2105.14328（△）](https://arxiv.org/pdf/2105.14328)，[glmtrans の説明書（△）](https://ytstat.r-universe.dev/glmtrans/doc/glmtrans-demo.html)

### Inferences
- 転移指数は「ソースで誤差 ε のときターゲットで ε^{1/ρ}」という率の指数で，推定には ε を何段階かで測る必要がある．K ≤ 50 本のターゲットでこれを推定する学習法は，指数という量の性質上作りにくい．MER-DA の「ソースの価値 = ラベル n 本」は，指数ではなく 1 つの数で価値を表すので，有限の K で推定する対象として扱いやすい．
- Hanneke ほか 2023 の不可能性は，MER-DA で C（罰則）やエンコーダを選ぶ場合にも当てはまる可能性がある．MER-DA が C をソースだけで固定した地図を土台にしていることは，この不可能性を避ける設計と読める（本記録の推論）．

### Gaps
- Hanneke ほか 2023 の本文（どの距離の量か，上界の形）は読んでいない．
- 「転移指数の推定」を題にした 2024〜2026 年の論文は，検索では見つからなかった．引用索引（2002.04747 を引く論文の一覧）での確認はしていない．
- Trans-GLM の掲載先（JASA と記憶するが未確認）．

## discrepancy・Y-discrepancy・多クラスの MDD で，ターゲットの少数のラベルを使う学習法

### Takeaway
ラベル込みの discrepancy（ある仮説がソースとターゲットで出す損失の差の上限）をターゲットのラベルで推定して使う学習法は定式化済みである（Best-effort の best・sbest，Konstantinov & Lampert 2019，Mansour ほか 2021，Mohri & Muñoz Medina 2012 のドリフトの QP）．多クラスの MDD はラベル無しの学習法で，ターゲットの少数ラベルで λ を扱う版は見つからなかった．

### Cited Findings
- Best-effort はラベル込みの discrepancy dis(P, Q)（式 1，"computing the labeled discrepancy assumes access to labels from both" distributions）とラベル無しの dis̄（式 2），仮説を部分集合 H₁ × H₂ に制限した局所 discrepancy（式 3）を定義する．ラベル無しの DA では基準の仮説 h₀ を使ってラベル込みのずれをラベル無しの項で上から抑え（補題 6・7），h₀ はターゲットのラベルがあればそれで選べる — [Best-Effort Adaptation（○，要約器）](https://ar5iv.labs.arxiv.org/html/2305.05816)
- Mohri & Muñoz Medina（ALT 2012，doi:10.1007/978-3-642-34106-9_13）は分布がドリフトする一括学習を discrepancy で解析し，ドリフトの PAC と追跡の 2 つの場面で上界を示し，単純な QP になる算法を提案した．後続の研究は，この上界がドリフトの大きさの事前知識か，各分布からの複数の標本を要すると指摘する — [arXiv 1205.4343（△）](https://arxiv.org/pdf/1205.4343)，[後続 AISTATS 2023（△）](https://proceedings.mlr.press/v206/awasthi23b/awasthi23b.pdf)
- Cortes, Mohri, Muñoz Medina "Adaptation Based on Generalized Discrepancy"（JMLR 20:1–30，2019）は，固定の重みづけではなく，学ぶ仮説に応じて訓練の損失の重みを変える算法を，discrepancy 最小化（DM）より緩い generalized discrepancy から導き，凸最適化で解けることを示した．ターゲットのラベル付きの標本を使うかは抄録に無い — [arXiv 1405.1503（○，抄録）](https://arxiv.org/abs/1405.1503)，[JMLR](https://www.jmlr.org/papers/v20/15-192.html)
- Mansour ほか 2021 と Konstantinov & Lampert 2019 はラベル込みのずれ（Mohri & Muñoz Medina 2012 の定義）を使い，前者は少数のターゲットのラベルで混合比を選び，後者は信頼できる少数のデータでずれを推定して重みを決める — 既存ノート `docs/related-work.md` 2.9 節・付録（◎）
- 音声の感情認識で MDD を使った例は検索で見つからなかった．最も近いのは Chao ほか（Interspeech 2019）の最大回帰食い違い（MRD）ネットワークで，IEMOCAP と MSP-IMPROV の間の valence の回帰で，ソースのみより一致相関係数が 10% と 5% 上がったとする — [ISCA Archive（△）](https://www.isca-archive.org/interspeech_2019/chao19b_interspeech.html)

### Inferences
- MER-DA の線形の頭でラベル込みの discrepancy を K 本で推定すると，仮説集合全体の sup はターゲットの 4K 点を当てる仮説で大きくなり，推定が不安定になる．Best-effort の局所 discrepancy（仮説を部分集合に制限する）を「出発点の重み W₀ の周りの球 ‖W − W₀‖ ≤ r」に当てると，MER-DA の仮説転移（出発点への L2 罰則）と同じ仮説の範囲で discrepancy を測ることになる．この対応づけは本記録の推論である．

### Gaps
- 「semi-supervised MDD」「few-shot MDD」の語では探していない．
- Mohri & Muñoz Medina 2012 で Y-discrepancy という名前が使われているかは確かめていない（検索結果では確認できなかった）．

## 高次元の線形の頭の転移の厳密な漸近の式から，混ぜる重み・罰則・切片を閉じた形で決める学習法はあるか

### Takeaway
回帰・LDA・LS-SVM では定式化済みである．Tiomoko ほか 2021 は最適なスコア y* を閉じた形で出し，その統計量を課題・クラスあたり 1 本から一致推定できるとした．Zhang ほか 2026 は最適な重み w* ∝ A_p⁻¹u_p と切片の補正を差し込み推定量で実装した（TLDA-E）．WONDER は分散したリッジの重みつき平均の最適な重みを使う．多クラスのロジスティック回帰に切片と L2 罰則を付けた頭で，漸近の式から重み・罰則・切片を決める学習法は見つからなかった．

### Cited Findings
- Tiomoko, Tiomoko Ali, Couillet（ICLR 2021）は多課題の LS-SVM を比例の極限で解き，判定の閾値の偏りと ±1 のラベルの選び方が負の転移の源になることを示し，最適なスコア y* を式 (8) の閉じた形で与えた．Remark 7 は "a single sample (two when i = i′) per task and per class (|Sil| = 1) is sufficient to obtain a consistent estimate for all quantities, so long that p is large" と書く．MNIST（ソース各クラス 100，ターゲット各クラス 10，PCA で p = 100）で検証した．学習の前に課題ごとに中心化する — 既存ノート `g5_theory_pooling_multisource.md` 論文 1（◎ 長版）
- Zhang, Fan, Luo, Hu（arXiv 2607.06936，2026-07）は 2 クラスの転移 LDA で，誤差の極限（Theorem 2），最適な切片の補正（Proposition 1：ターゲットのクラス比の不均衡の項 w_KΔ_K を消す），最適な重み w* ∝ A_p⁻¹u_p（Corollary 1）を示し，差し込み推定量 ŵ* の一致性（Theorem 3）を証明した．ADHD-200（ターゲットの学習標本 28〜115）と PPMI で検証した — 既存ノート `g4_theory_intercept_direction.md` 10a（◎ 前半）
- Yang, Zhang, Wu, Ré, Su（JMLR 26，2025）は合併の最小 2 乗の偏りと分散を閉じた形にし，ソースを足すと得から損に変わる相転移を示した．最小最大の下界と定数倍で一致するのは「ソースを捨てるかを選ぶ調整した HPS」で，閉じた形の重みの差し込みではない．ターゲットの標本数は n₂ ≥ (1+τ)p を仮定する — 既存ノート `g5_theory_pooling_multisource.md` 論文 2（◎ 一部）
- Dobriban & Sheng "WONDER"（JMLR 21(66):1–52，2020．短い版は ICML 2020 の Sheng & Dobriban）：高次元のランダム効果の模型で，各機械のリッジ推定量の重みつき和の最適な重みは，リッジの縮小の偏りのため和が 1 を超え，単純な平均は最適でない．算法 WONDER を Million Song Dataset で試し，計算時間を 100 倍以上減らしてほぼ同じ精度を出した — [arXiv 1903.09321（○，抄録）](https://arxiv.org/abs/1903.09321)，[JMLR](https://jmlr.org/papers/v21/19-277.html)
- Obst ほか（arXiv 2102.09504，2021）は線形回帰のパラメータ転移で，新しい入力での転移の質が問題の母数から作った固有基底での表現に依ると示し，微調整した模型がターゲットだけの模型より 2 乗の予測危険度が小さいかを予言する検定を作った．検索結果の要約では，得は未知の母数に依るので F 分布に基づく検定を導き，W·β̂_S + (I − W)·β̂_T の形の推定量一般に使えるとされる．人工データと電力需要のデータで検証した — [arXiv 2102.09504（○，抄録．F 検定の部分は △）](https://arxiv.org/abs/2102.09504)
- Ha & Chen（arXiv 2507.14661v2）の MASFT は，ずれの型ごとの候補を作り，少数のターゲットの検証データで選ぶ（追加の費用は候補数の対数）．線形・2 乗損失で，分類は今後の課題 — 既存ノート `docs/related-work.md` 付録（◎）
- Ildiz, Gozeten, Taga, Mondelli, Oymak（ICLR 2025，arXiv 2410.18837）はリッジ無しの高次元回帰で，代理のモデル（先生）が任意の場合と，分布のずれた別データで ERM した場合の生徒の危険度を非漸近で抑え，最適な代理モデルの形を導いた．弱い特徴をデータに応じて捨てることが得になる場合と損になる場合を示した — [arXiv 2410.18837（△）](https://arxiv.org/abs/2410.18837v2)

### Inferences
- Tiomoko の Remark 7（p が大きければクラスあたり 1 本で十分）は，MER-DA の p = 768〜1024，K = 1〜50 の体制で差し込み推定量が成り立つ根拠になりうる．ただし Tiomoko は LS-SVM で課題ごとに中心化し，MER-DA は中心化しない合併も扱うので，そのまま移せるかは確かめが要る．
- Zhang 2026 の切片の補正（ターゲットのクラス平均の中点で切片を決める）は，MER-DA の「K 本は切片を 1/(2K) で直す」と同じ操作の 2 クラス LDA 版と読める．

### Gaps
- 多クラスのロジスティック回帰の比例極限（CGMT など）で，ソースの重みや出発点への罰則の強さを閉じた形で決める学習法は探していない．
- WONDER が未知の母数を推定するか，Ildiz 2025 の最適な代理モデルを推定して使う算法を提案しているかは確かめていない．

## 不可能性の定理が示す条件や，共変量シフト・ラベルシフトなどの仮定を少数のラベルで検定する方法はあるか

### Takeaway
検定の方法はある．ラベル無しで済むのはラベルシフトの検出（BBSE）で，P(Y|X) の一致（共変量シフトの仮定）や Y|X・X|Y のずれを検定するにはターゲットのラベルが要る（DetectShift，Hu & Lei 2024）．ラベルが後から届く場面で害のあるずれを逐次に検定する方法（Podkopaev & Ramdas 2022），転移の得を検定する方法（Obst 2021，Trans-GLM の検出）もある．不可能性の定理（Ben-David, Lu, Luu, Pál 2010）は，ラベル無しのデータだけでは区別できない 2 つの課題があることを示すので，P(Y|X) についての条件はラベル無しでは検定できない．

### Cited Findings
- Ben-David, Lu, Luu, Pál "Impossibility Theorems for Domain Adaptation"（AISTATS 2010，PMLR 9:129–136）：ラベル付きのソースとラベル無しのターゲットを使う不可知の PAC の模型で，訓練の入力からは区別できないのに，重みづけが一方を大きく助け他方を大きく損なう課題の組がある．共変量シフトの仮定は多くの DA の中心だが，重みづけの方法は分散を抑えるのに非現実的な標本数を要するとする — [PMLR（△）](https://proceedings.mlr.press/v9/david10a.html)，[著者版 PDF（△）](https://www.cs.toronto.edu/~tl/papers/BenDavidEtAl_aistats2010.pdf)
- Polo, Izbicki, Lacerda, Ibieta-Jimenez, Vicente "A unified framework for dataset shift diagnostics"（Information Sciences 2023: 119612，arXiv 2205.08340）：DetectShift は (X, Y)，X，Y，X|Y，Y|X の分布のずれを，誤警報を制御した仮説検定で検定し，同じ性質の検定統計量でずれの大きさも測る．ターゲットのラベル付き標本が限られる場面で，ソースとターゲットの両方を使う予測器の適応や学び直しの判断に役立つとする．Python のパッケージ detectshift がある — [arXiv 2205.08340（○，抄録）](https://arxiv.org/abs/2205.08340)
- Hu & Lei（JASA 119(546):1136–1154，2024，arXiv 2010.07147）は，2 つの母集団で Y|X の条件付き分布が同じかを，重みつきの共形予測と順位和の統計量で検定する．帰無仮説の下で U 統計量の期待値は 1/2．後続の研究は，条件付き密度比と周辺密度比の両方の推定が要り，次元が高いと不安定になると指摘する — [arXiv 2010.07147（△）](https://arxiv.org/pdf/2010.07147)，[IDEAS（△）](https://ideas.repec.org/a/taf/jnlasa/v119y2024i546p1136-1154.html)，[後続 arXiv 2402.00164（△）](https://arxiv.org/pdf/2402.00164)
- Podkopaev & Ramdas（ICLR 2022，arXiv 2110.06177）は，ターゲットの危険度がソースの危険度から許容幅を超えて上がったかを，時間一様な信頼系列で逐次に検定する．予測の後に（一部の）真のラベルが届く場面で使える — [arXiv 2110.06177（△）](https://arxiv.org/pdf/2110.06177)
- Lipton, Wang, Smola 2018（BBSE）は，ソースの混同行列とターゲットの予測分布からラベルシフトを検出・補正する — 既存ノート `docs/related-work.md` 2.11 節（◎）
- 音声の感情の個人適応（arXiv 2309.02418）は，話者ごとのラベルの分布と訓練集合のラベルの分布の KL をラベルのずれとして測り，arousal では特徴のずれと性能の相関 −0.714 に対しラベルのずれと性能の相関は −0.502 だった．検定ではなく診断である — [arXiv 2309.02418（△）](https://arxiv.org/pdf/2309.02418)

### Inferences
- MER-DA で K 本を使って P(Y|X) のずれを検定すると，ターゲットのラベル付き標本は 4K 本しかなく，しかも同じ話者の発話が相関するので，検定の力は小さく，話者をまとまりとして並べ替える検定が要る．DetectShift も Hu & Lei も話者のまとまりを扱うとは書かれていない（抄録の範囲）．
- 不可能性の定理と Wang & Mao の記述，MER-DA の「ラベル無しの情報では向きを直せない」は，同じ事実（ラベルの付け方のずれはラベル無しでは見えない）を別の言葉で述べていると読める．

### Gaps
- DetectShift が分類で Y|X のずれを検定するときに必要なターゲットのラベル数は確かめていない．
- 感情認識で共変量シフトやラベルシフトの仮定を検定した例は見つからなかった．

## 弱から強への一般化の理論（Charikar 2024 の misfit，Dong 2025，Xue 2025）を学習法にした例

### Takeaway
Charikar ほか 2024 は，抄録で「misfit の大きさから，どの弱いモデルで強いモデルを学ばせるかを選べる」と述べており，理論から出た選び方の規則がある．後続には，弱いラベルを使うかを分類器で決める Selective W2S（arXiv 2511.14166）や，最適な代理モデルの形を導く Ildiz 2025 がある．Dong 2025 と Xue 2025 を学習法にした例は見つけていない．

### Cited Findings
- Charikar, Pabbaraju, Shiragur（NeurIPS 2024，arXiv 2405.15116 v2 は 2024-10-23 のカメラ版）：強いモデルの得は，強いモデルが弱いモデルのラベルに対して出す misfit の誤差で定量化できる．抄録は，この理論が得の大きさを予言し，misfit に基づいて強いモデルを学ばせる弱いモデルを選ぶ指針になると述べ，これを "curious algorithmic insights" と呼ぶ — [arXiv 2405.15116（○，抄録）](https://arxiv.org/abs/2405.15116)
- Mulgund & Pabbaraju（ICML 2025，arXiv 2501.19105）は misfit と得の関係を 2 乗損失の外（Bregman ダイバージェンスと分類）に広げ，k 個の強いモデルの凸結合を使い，余分の誤差の項は k が大きいと消える — [ML Anthology（△）](https://mlanthology.org/icml/2025/mulgund2025icml-relating/)
- Bregman の偏りと分散の分解による解析（arXiv 2505.24313）は，生徒と先生の母集団の危険度の差が両者の期待 misfit で表されるとし，生徒の仮説集合が凸であることを要しない — [arXiv 2505.24313（△）](https://arxiv.org/pdf/2505.24313)
- Lang, Huang, Li "Selective Weak-to-Strong Generalization"（arXiv 2511.14166）は，弱いラベルが要るかを予測する 2 値分類器を学び，要らないときは弱いラベルを使わない枠組みを示した．misfit の理論を引くが，分類器が misfit を基準にするかは検索結果では分からない — [arXiv 2511.14166（△）](https://arxiv.org/pdf/2511.14166)
- Dong 2025（先生と生徒の部分空間の食い違いで分散が減る）と Xue 2025（ギャップの厳密な式）は既存ノートで理論として読んだ — `g6_theory_weak_label_active.md`（◎）

### Inferences
- MER-DA の既存の報告では，純相補性 J が先生と出発点の正解率の差に恒等的に等しく，E59 で J ≈ +0.6 に対して得 +1.1 だった．Charikar の「misfit で先生を選ぶ」規則を MER-DA で使うには，この食い違い（得が上限とみなした量を超える）が 2 乗損失と凸の前提から外れていることによるのかを先に確かめる必要がある．
- misfit はラベル無しのターゲットの発話だけで計算できるので，K 本を検証に使わずに先生や総重量を選ぶ規則になりうる．

### Gaps
- Selective W2S の基準，Ildiz 2025 が最適な代理モデルを推定する算法を出しているかは確かめていない．
- 弱から強の理論を感情認識に使った例は見つからなかった（探した語は「weak-to-strong」と感情認識の組み合わせを含まないので，探し切ってはいない）．

## 上のほかに，量は定義されているが学習法になっていない DA の理論（PAC-Bayes，情報理論，最適輸送など）

### Takeaway
PAC-Bayes（Germain ほか）と情報理論（Wang & Mao 2023）と最適輸送（JDOT 系）の上界は，ラベル無しのターゲットを使う学習法として定式化済みである．どの系統でも，ターゲットのラベルが要る項（PAC-Bayes の適応性の項，情報理論の条件付き分布のずれ）は学習法の中で扱われていない．ターゲットの少数ラベルでこれらの項を推定する学習法は見つからなかった．

### Cited Findings
- Germain, Habrard, Laviolette, Morvant（ICML 2013，PMLR 28(3):738–746）は不一致の平均による擬距離を使った初の PAC-Bayes の DA の上界を示し，確率的な Gibbs 分類器について直接最適化できる上界を線形分類器に特化した．ICML 2016 は比で表したずれがソースの誤差とターゲットの投票者の不一致の釣り合いを決める上界を示し，線形分類器の学習法を実データで評価した．Neurocomputing 2020（arXiv 1707.05712）は両者をまとめて 2 つの学習法を導いた — [PMLR v28（△）](https://proceedings.mlr.press/v28/germain13.html)，[HAL 2016（△）](https://hal.archives-ouvertes.fr/hal-01307045)，[arXiv 1707.05712（△）](https://ar5iv.arxiv.org/html/1707.05712)
- Sicilia ほか 2022 は多クラスの PAC-Bayes の DA の上界と，多クラスのずれの近似法を示した．平坦さの項 ρ はターゲットのラベルが要るので統制された研究の場面の外では計算できず，実用のために小さいと仮定する — [Sicilia ほか 2022（○，要約器）](https://arxiv.org/abs/2207.05685)
- Wang & Mao 2023 は算法に依存する情報理論の上界から 2 つの技法を導いた．(1) 勾配の罰則（GP）：各段の勾配のノルムを抑えると，最終の重み W_T と初期値 W₀ の距離も抑えられ，算法が届く仮説の範囲が縮む．(2) ラベル情報の制御（CL）：本当のソースのラベルを見ない補助の分類器 f_w̃ をターゲット（とソース）の擬似ラベルで学び，‖W − W̃‖² を正則化に足す．RotatedMNIST と Digits で検証し，モデル選択はターゲットの検証集合（オラクル）で行った — [Wang & Mao 2023（◎，pdftotext）](https://arxiv.org/abs/2210.00706)
- 最適輸送の DA の理論（Redko, Habrard, Sebban 2017，arXiv 1610.04420）がある．JDOT はラベルの情報を輸送の費用に入れる変種で，多ソースの重みつき JDOT（MSDA-WJDOT）は構音障害の音声で基準より命令の誤り率を 16% 減らしたとされる — [arXiv 1610.04420（△）](https://arxiv.org/pdf/1610.04420)，[構音障害の音声 arXiv 2104.02535（△）](https://arxiv.org/pdf/2104.02535)
- 音声の感情認識では SOT（Interspeech 2023）が最適輸送でドメイン間の分布の距離を減らし，感情のクラスが大きく重なると素朴な最適輸送が負の輸送を起こすので自己教師ありの部品を足した — [ISCA Archive（△）](https://www.isca-archive.org/interspeech_2023/zhang23y_interspeech.html)
- EEG の感情認識では多ソースの重みづけが多い：DAMSDAN（arXiv 2510.17475．ソース間の分布の違いとターゲットとの関連の度合いをモデル化），CWMAN-EEG（多くの手法がソースを単純に合併するか等しく寄与させると批判し，条件つきの重みを付ける），MS-DCDA（SEED で平均 90.84%，SEED-IV で 78.49%）— [DAMSDAN（△）](https://arxiv.org/html/2510.17475v1)，[CWMAN-EEG（△）](https://www.researchsquare.com/article/rs-10250843)，[MS-DCDA（△）](https://arxiv.org/abs/2408.10235v1)
- 音声の感情認識のテスト時適応 Emo-TTA（arXiv 2509.25495）は，音声と言語のモデルの予測を事前分布として，クラスごとの統計量（事前確率・平均・共有の共分散）を EM で逐次に更新する — [arXiv 2509.25495（△）](https://arxiv.org/pdf/2509.25495)

### Inferences
- Wang & Mao の GP は「最終の重みと初期値の距離を抑える」ことで仮説の範囲を縮める技法で，MER-DA の出発点への L2 罰則（仮説転移）と同じ働きを，情報理論の上界から理由づけている．MER-DA の L2 罰則の理論的な根拠として引ける可能性がある（本記録の推論）．
- Wang & Mao が「ラベル無しでは最小化できない」とした条件付き分布のずれは，ターゲットの K 本があれば推定の対象になる．これを K 本で推定して罰則に入れる学習法は見つからなかった．

### Gaps
- Germain 2016 の算法の名前（DALC と記憶するが検索では確認できなかった）と，2013 の算法の名前（PBDA と記憶するが未確認）．
- JDOT を感情認識に使った例は見つからなかった．

## MER-DA の設定で学習法にするなら何が要るか（候補ごとに）

### Takeaway
候補は 8 つで，どれも凍結特徴の上の多クラスのロジスティック回帰と K 本・話者分割・UAR に合わせる作業が要る．既存の学習法を移す候補（局所のラベル込みの discrepancy，差し込みの閉じた形，得の検定）は作りやすく，λ や条件付きのずれを K 本で推定する候補は「4K 点は線形分離できてしまう」ことへの対処が共通の障害になる．以下はすべて本記録の推論で，文献の裏づけは上の各節にある．

### Cited Findings
- 候補の土台にした先行の学習法と事実は上の各節に出典つきで書いた（Ben-David 2010 の α*，Best-effort の sbest と局所 discrepancy，DARN，Tiomoko 2021 の Remark 7，Zhang 2026 の TLDA-E，Trans-GLM の検出，DetectShift，Charikar 2024，Wang & Mao 2023）．

### Inferences
1. **λ の K 本推定（切片の分と向きの分に分ける）**：ソースと K 本を合併して出発点の重み W₀ の周りで罰則つきに当て，話者で分けた交差検証でターゲットの外の誤差を測り，λ̃ を「中心化と切片の補正で消える分」と「残る分」に分ける．要るのは話者の数（K 本が少数の話者に偏ると交差検証が崩れる）と，4K 点が線形分離できる体制で過学習しない罰則の選び方である．
2. **Ben-David の α* の差し込み → MER-DA 版の重み**：Ben-David の式は仮説集合の複雑さ d（線形の頭では特徴の次元 768〜1024 に比例する大きさ）と λ を含み，D² = d/A² が数千本になりうるので，K ≤ 50（ターゲットは 4K ≤ 200 本）では m_T ≥ D² を満たさず，式はソースを捨てる α* = 1 をほぼ返さない．代わりに MER-DA の換算 n = q e²/(1−e²) から α = n/(n + K) を作るなら，要るのは e と q をソースとラベル無しのターゲットと K 本から推定する方法と，その推定の偏りの補正である．
3. **ラベル込みの局所 discrepancy を出発点の周りで測る（Best-effort の sbest を線形の頭に移す）**：仮説を ‖W − W₀‖ ≤ r に制限した dis(P̂, Q̂) を K 本で計算し，ソースの各コーパスの重みと罰則を同時に決める．要るのは多クラスのロジスティック損失での discrepancy の計算（Best-effort は DC 計画），7 群の多ソースへの拡張（Best-effort は 1 ソース），r と罰則の選び方（Best-effort はターゲットのラベルで交差検証するが，MER-DA では K 本しかない）である．
4. **ソースの価値 n の推定と，Trans-GLM 型のソースの検出**：ソースを 1 つずつ足したときの得をターゲットだけの基準と比べ，有意なものだけ使う．要るのは K 本での得の推定の分散の見積もり（話者で並べ替える検定）と，7 群で何度も検定する分の補正である．転移指数 ρ の推定は K 本では作りにくいので，価値を「ラベル何本ぶん」の 1 つの数で表す MER-DA の量のほうが推定の対象に向く．
5. **高次元の閉じた形の差し込み（Tiomoko，Zhang 2026）を多クラスのロジスティック回帰に移す**：コーパスごと・クラスごとの平均と共通の共分散から最適な重みと切片を差し込みで計算する．Tiomoko の Remark 7 は p が大きければクラスあたり 1 本で統計量が一致推定できるとするので，p = 768〜1024 の MER-DA の体制に合う．要るのは LDA・LS-SVM の式とロジスティック回帰＋L2 罰則の解の対応づけ（比例極限の CGMT などの結果）と，凍結特徴の発話平均がガウス模型からどれだけ外れるかの確認である．
6. **仮定の検定（DetectShift，Hu & Lei，BBSE）**：BBSE はラベル無しでラベルシフトを検定できるので，MER-DA の 7 群でそのまま使える．Y|X のずれの検定（DetectShift，Hu & Lei）は K 本を使うが，話者のまとまりを扱う並べ替えに作り直す必要がある．検定の結果で「中心化だけ」「切片も直す」「向きも直す」のどれを使うかを決める学習法にするなら，K ごとの検定の力の見積もりが要る．
7. **misfit で先生と重みを選ぶ（Charikar 2024）**：ラベル無しのターゲットの発話で生徒と先生の misfit を測り，先生（融合か単モダリティか）と総重量を選ぶ．K 本を検証に使わずに済む．要るのは，2 乗損失と凸の前提をロジスティック回帰と UAR に当てたときに misfit と得の関係が保たれるかの確認で，E59 で得が J を超えた事実と整合するかを先に調べる必要がある．
8. **情報理論の条件付きのずれ D_KL(P_{Y'|T'}‖P_{Y|T}) を K 本で推定して罰則にする（Wang & Mao 2023 の欠けた項）**：K 本でターゲットの P(Y|X) を推定し，ソースの頭との KL を罰則に入れる．要るのは K 本でのクラス事後確率の推定の安定化（出発点の頭からの差分として推定する）で，結果として仮説転移の L2 罰則と同じ形に戻るかを確かめる必要がある．Wang & Mao の勾配の罰則（初期値との距離を抑える）は MER-DA の L2 罰則の理由づけとしても使える．

### Gaps
- 候補 2 と 4 の e と q を K 本とラベル無しのデータから推定する方法が，MER-DA の既存の文書（`docs/methods.md`）で定まっているかは本記録では確かめていない．
- 候補 3 と 5 の計算量（7 群 × K の格子 × 反復）の見積もりはしていない．
- 候補 7 の前提の確認には，既存の報告（`reports/TAC 論文 先行研究の精読と穴.md`）の注意 (a) の扱いを先に決める必要がある．
