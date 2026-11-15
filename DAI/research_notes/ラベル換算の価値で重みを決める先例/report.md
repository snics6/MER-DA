# ラベル換算の価値で重みを決める先例（調査メモ）

調査日：2026-10-28．調査者：Claude（先行研究の調査担当）．書き込んだのは本ファイルだけである．

**調べた形**：ターゲットに感情ごとに K 本しかラベルが無いコーパス間の感情認識で，凍結したエンコーダの特徴の上に線形の生徒を学ぶ．資源は (1) ターゲットの K 本のラベル，(2) ソースのコーパスのラベル付きデータ，(3) ソースだけで学んだ別モダリティの先生がターゲットのラベル無しの発話すべてに付けた弱いラベル（確率），の 3 種類である．各資源の価値を，ターゲットのみで学んだ学習曲線の上で「ラベル何本ぶんか」（ラベル換算の価値）として読み，生徒の目的関数で各資源にかける重みをこの本数から決める（例：先生の重み : K 本のラベルの重み ＝ 先生の本数 : K）．さらに，K が増えると先生の重みがどう下がるべきかを予言する．

**確かめた水準の記号**：
- ◎ 本文を読んだ．PDF を pdftotext で文字にして読んだもの（Jain 2024，CLIP 2021，catalytic prior 2020，Li 2017）と，HTML 版を WebFetch で読んだもの（PPI++，DeSSL，二重頑健な自己学習，SLaM，xMUDA，FtD++）がある．後者は要約器を通した読みなので，引いた語句が原文と細部で違う可能性がある
- ○ 抄録（出版社・arXiv・PMLR・学会の頁）だけ
- △ 検索結果の抜き書きだけ
- 既存のノート `research_notes/TAC 論文 先行研究の精読と穴/` の g1〜g6 で ◎ と記録された文献は，その記録を引いた（本調査では読み直していない）．その場合は「g2 ◎」のように書く

---

## 1. 結論

1. 上の形をそのまま行った先例は，感情認識（音声・顔・言葉・マルチモーダル）でも一般の機械学習でも見つからなかった（検索した語は付録 B）．
2. 最も近いのは Jain, Montanari, Sasoglu（NeurIPS 2024）である．少数の本物のデータ n 本に「代わりのデータ」（別のデータセットや，別のモデルが付けた擬似ラベル）を重みつきで混ぜ，本物だけ・代わりだけで学んだ 2 本の学習曲線から重みを予言し，漸近の式 α* = R_or(n)/(R_su + R_or(n)) を示した．代わりのデータの価値を本物の本数に直すと，この式は「代わり : 本物 ＝ 代わりの本数 : n」になり，研究者の規則と同じ形である（この書き直しは本調査の計算で，論文は本数で書いていない）．
3. 測り方は CLIP（Radford ほか 2021）が同じ形で行った．同じ凍結特徴の上のロジスティック回帰の学習曲線で，ラベルを使わないゼロショット分類器が「クラスあたり何本」に当たるかを読んでいる．ただし本数を重みには使わず，ゼロショット分類器と少数ラベルの組み合わせ方を今後の課題としている．先生が付けたラベルの総重みを「データ何点ぶんか」で表す考えは，catalytic prior（Huang ほか 2020）と power prior の系統にある．
4. 残る違いは次の組み合わせである．換算の本数をそのまま重みの規則に使う．3 つの資源を同じ単位（感情ごとのラベル本数）で並べる．ソースだけで学んだ別モダリティの先生を，コーパス間・感情ごとの K 本の設定で使う．K に対する重みの変化を予言して，格子で選んだ重みと照合する．
5. K に対する重みの予言は，先例の中で 2 つの型に分かれる．2 乗誤差で偏りが 2 乗で効く模型（Jain の式，ベイズの擬似観測）では，先生の割合 n_T/(n_T + K) だけが下がり，ラベル何本ぶんかに直した重みの絶対量は K に依らず一定である．Ben-David ほか 2010 の上界（ずれの罰が重みに比例する）では，ターゲットが m_T ≥ d/A² 本あるとソースの重みは 0 になる．研究者の E59 で K とともに選ばれる総重みが小さくなり K=25 でほぼ 0 になったことは，後者の型と向きが合う（4.3 節．この対応づけは本調査の読み）．別モダリティの先生の弱いラベルについて，絶対量が K とともに下がることを予言した先例は見当たらない．

---

## 2. 最も近い先例（上位 6 本）

| 順 | 文献 | 何をしているか | 上の形と同じ点 | 違う点 | 水準 |
|---|---|---|---|---|---|
| 1 | Jain, Montanari, Sasoglu, "Scaling laws for learning with real and surrogate data", NeurIPS 2024（Advances in Neural Information Processing Systems 37，doi:10.52202/079017-3499，arXiv 2402.04376） | 本物 n 本に (1−α)/n，代わりのデータ m 本に α/m の重みをかけた経験損失を最小にする．本物だけ・代わりだけの学習曲線を当てはめた尺度則（式 4）を α について最小にして，重みと足すべき m を予言する．低次元の漸近で α* = R_or(n)/(R_su(m) + R_or(n))（論文の付録 B の Corollary B.2．R_or(n) = Tr(H⁻¹K)/n） | 重みを格子探索ではなく学習曲線から決める．代わりのデータの価値を，本物のテストでの過剰誤差という本物だけの学習曲線と同じ物差しで測る．α* を本数に直すと n_su : n になり，n が増えると下がる（Remark B.2「when the distribution shift is large, and the sample size n is large enough, we expect small α to be optimal」）．実験にロジスティック回帰がある．代わりのデータの 1 つは，CIFAR10 の 1 万枚で学んだモデルが CIFAR100 の画像に付けた擬似ラベル | 「何本ぶんか」という言い方も effective sample size の語も使わない．代わりのデータは 1 種類で，複数の資源を同じ単位で並べない．代わりのデータはターゲットとは別の入力で，ターゲットのラベル無しの入力に付けた弱いラベルではない．別モダリティの先生は無い．Corollary B.2 は n, m が大きい漸近の主張で，K ≤ 10 の少数では保証が無い．尺度則の指数 β は当てはめる．感情認識ではない | ◎ |
| 2 | Radford ほか, "Learning Transferable Visual Models From Natural Language Supervision"（CLIP），ICML 2021，PMLR 139:8748–8763（arXiv 2103.00020） | 同じ CLIP の特徴の上のロジスティック回帰を 1・2・4・8・16 本／クラスと全ラベルで学び，その学習曲線を対数線形で内挿して，ゼロショット分類器の性能が何本／クラスに当たるかをデータセットごとに出した（中央値 5.4，平均 20.8，最大 184）．「this estimates the effective data efficiency of zero-shot transfer」 | 凍結特徴の上の線形分類器の学習曲線で，ラベルを使わない分類器の価値をクラスあたりの本数に換算する．研究者の測り方と同じ | 本数を重みに使わない．「use CLIP's zero-shot classifier as a prior for the weights of the few-shot classifier」として L2 罰則を試したが，超母数の最適化が大きすぎる罰則を選び，少数ショットの分類器がゼロショットそのものになったと書く．組み合わせ方を今後の課題とする．先生は言葉から作った分類器で，ソースのコーパスで学んだ別モダリティの分類器ではない．ソースのラベル付きデータとドメインのずれは扱わない | ◎（3.1 節，図 7） |
| 3 | Huang, Stein, Rubin, Kou, "Catalytic prior distributions with application to generalized linear models", PNAS 117(22):12004–12010, 2020, doi:10.1073/pnas.1920913117 | 簡単なモデル（切片だけ，主効果だけなど）を観測データに当てはめ，その予測分布から合成の応答を作り，合成の共変量 M 点に付ける．その尤度を τ/M 乗して本物の尤度に足す．τ は「can be interpreted as the effective prior sample size」．τ は予測の危険度の推定（パラメトリック・ブートストラップか Stein の不偏推定）の最小化か，τ に置く超事前分布で決める．GLM（ロジスティック回帰）に適用 | 先生（簡単なモデル）が付けたラベルの総重みを「データ何点ぶんか」の単位で表し，本物の n 点と同じ尤度に足す．τ を固定すれば，合成データの割合 τ/(τ + n) は n とともに下がる（この割合は本調査の読み） | 先生は同じ観測データで学んだ簡単なモデルで，ソースでも別モダリティでもない．τ を先生の価値の測定（学習曲線）から決めず，予測の危険度の推定で選ぶ．目的は高次元の GLM の推定を安定させること | ◎ |
| 4 | Li, Yang, Song, Cao, Luo, Li, "Learning from Noisy Labels with Distillation", ICCV 2017, doi:10.1109/ICCV.2017.211（arXiv 1703.02391） | 少数のきれいなラベルで学んだ補助モデル（先生）の出力 s と，大量の雑音のあるラベル y を ŷ = λy + (1−λ)s で混ぜて学ぶ．命題 1：2 乗距離の危険度 R について λ = R_s/(R_s + R_y) で最小 R_y R_s/(R_s + R_y)．実験では λ = mAP_Dc/(mAP_D + mAP_Dc)（開発用データの mAP の比） | 先生の項の重みを，先生と他のラベルの質の比から閉じた式で決める（逆分散の重み）．格子探索をしない | 混ぜるのは同じ標本の上の 2 種類のラベルで，資源の総量（本数）の比ではない．ラベルが増えたときの重みの変化を扱わない．先生は同じドメインのきれいなデータで学ぶ | ◎ |
| 5 | Angelopoulos, Duchi, Zrnic, "PPI++: Efficient Prediction-Powered Inference"（arXiv 2311.01453．掲載先は arXiv の頁に無い）と Schmutz, Humbert, Mattei, "Don't fear the unlabelled: safe semi-supervised learning via debiasing", ICLR 2023（arXiv 2203.07512） | PPI++：ラベル付き n 本の損失 L_n(θ) + λ(L̃^f_N(θ) − L^f_n(θ))．推定量の分散を最小にする λ を閉じた式で推定する（平均なら λ* = Cov(Y, f)/((1 + n/N)Var f)．M 推定は Proposition 2）．DeSSL：同じ形を半教師あり学習の危険度に入れ，定理 3.1 で λ_opt = (n_u/n)·Cov(L, H)/Var(H)（n = n_l + n_u） | 先生の予測の項の重みを，先生の質（ラベルとの共分散）とデータの数から理論で決める．ラベル付きの数が増えると係数 λ が下がる．PPI++ はロジスティック回帰を扱う | 予測の項はラベル付きデータでの差を引いて偏りを消す形で，先生の偏りを重みで抑える形ではない．PPI++ の目的は母数の推定と信頼区間で，分類器の正解率ではない．DeSSL は「the estimation of Cov(L, H) with few labels led to unsatisfactory results」と書き，実験では λ を振る．どちらも重みをラベル換算の本数で表さない | ◎（HTML．要約器を通した読み） |
| 6 | power prior・commensurate prior・MAP prior の系統：Ibrahim & Chen, Statistical Science 15(1):46–60, 2000（doi:10.1214/ss/1009212673），Hobbs, Carlin, Mandrekar, Sargent, Biometrics 67(3):1047–1056, 2011，Neuenschwander ほか, Clinical Trials 7(1):5–18, 2010 | 過去のデータの尤度を a0 乗して事前分布にし，過去のデータを実効の標本数ぶんだけ借りる．a0 は過去と現在のデータの整合の度合いから決める．MAP prior は過去の対照群を「何人ぶんの患者を節約できるか」で表す | 別の情報源を「何人ぶんか」で数え，その数を目的関数（尤度）の重みにする．現在のデータが増えれば借りた分の割合は下がる | 別の情報源はラベル付きの過去データで，先生の弱いラベルではない．a0 を学習曲線での価値の測定から決めない．臨床試験の母数の推定で，分類器の学習ではない | △ |

---

## 3. 部分的な先例の一覧（観点ごと）

### 3.1 感情認識（音声・顔・言葉・マルチモーダル）

- 別モダリティの先生の重みを，先生の価値（ラベル換算の本数）から決める感情認識の研究は見つからなかった（付録 B の検索）．
- 既存ノート g2（◎）の別モダリティ蒸留 10 本と余力分では，蒸留の重みは固定値か検証データでの選択である．Albanie, Nagrani, Vedaldi, Zisserman 2018（ACM MM 2018，doi:10.1145/3240508.3240578．温度 T = 2．蒸留は第三のコーパスのラベル無し動画で行う），Srinivasan, Huang, Kirchhoff 2022（arXiv 2112.00158．会議名は未確認．λ = 30，「relatively insensitive in the range 3-50」），Hajavi & Etemad（IEEE TAI．α = 0.5），LiSER（Pendyala, Morgado, Sethares 2025．Interspeech 2025 の ISCA Archive の頁が検索結果に出た（△）．λ ∈ {0.1, 0.5, 1, 5, 10} を検証データで選ぶ．標本ごとに先生の最大確率で重みづけする版もある），PriCon（Pinitas, Makantasis, Yannakakis, ICMI 2025．α ∈ {0.25, 0.5, 0.75, 1.0} を検証データで選び，ほかの窓に流用）．g2 の結論は「蒸留の重みをラベルの本数ごとに選ぶ研究は無い」である．
- 価値をラベルの数で述べた記述として，LiSER の「ラベルを半分にしても蒸留ありが全ラベルの蒸留なしを上回る」がある（g2 の記録）．本数への換算と重みへの利用は無い．
- 既存ノート g1（◎）の応用研究（M4SER，MDAT，CTIAL，Li, Kim, Kuo, Narayanan 2021 ほか）にも，先生や擬似ラベルの重みを価値から決める例は無い．Li, Kim, Kuo, Narayanan 2021 はソースのモデルのクラスごとの平均出力を τ = 2 のソフトラベルにし，重みは固定である．
- 本調査の検索で新たに出た Aslam ほか 2023（CVPRW．音声を特権情報とした音声映像の先生から映像の生徒へ）と多先生の特権蒸留（arXiv 2408.09035．確信度の高い先生を選ぶ）は △ の水準で，重みの決め方は検索結果に無い．

### 3.2 蒸留や擬似ラベルの損失の重みを，先生の誤り・確信度・雑音から理論で決める研究

| 文献 | 何をしているか | 上の形との違い | 水準 |
|---|---|---|---|
| Li ほか, ICCV 2017（2 節の 4） | λ = R_s/(R_s + R_y)．実験では mAP の比 | 同じ標本の上の 2 種類のラベルの混合比．本数の比でない | ◎ |
| Kontonis, Iliopoulos, Trinh, Baykal, Menghani, Vee, "SLaM: Student-Label Mixing for Distillation with Unlabeled Examples", NeurIPS 2023（arXiv 2302.03806） | 先生のラベルが正しい確率 α(x) を，先生の余裕（margin）から検証データ上の isotonic 回帰で推定し，生徒の予測と先生のラベルを α(x) で混ぜる．線形の場合に収束の保証 | 標本ごとの混合比．ラベル付きと先生ラベル付きの間の重みは置かない（1/\|A∪B\| で平均） | ◎（前半） |
| Iliopoulos, Kontonis, Baykal, Menghani, Trinh, Vee, "Weighted Distillation with Unlabeled Examples", NeurIPS 2022（arXiv 2210.06711） | 先生のラベルの雑音を打ち消す標本ごとの重み（debiasing）．母数なし | 標本ごとの重みで，資源の総量の重みではない | ○ |
| Borup & Andersen, "Even your Teacher Needs Guidance: Ground-Truth Targets Dampen Regularization Imposed by Self-Distillation", NeurIPS 2021（arXiv 2102.13088） | カーネル回帰の反復自己蒸留で，本当のラベルと先生の出力の重みの最適値を各段で閉じた式で与え，深層学習でも格子探索より安く推定する | 先生は同じデータで学ぶ（自己蒸留）．ラベル換算の本数は使わない | ○ |
| Das & Sanghavi, "Understanding Self-Distillation in the Presence of Label Noise", ICML 2023, PMLR 202:7102–7140 | 線形回帰で模倣の重み ξ の最適値を求め，雑音が大きいと ξ > 1．ロジスティック回帰でも生徒が先生を上回る範囲を示す | 自己蒸留．重みは雑音の量で決まり，資源の本数ではない | ○ |
| Pareek, Du, Oh, "Understanding the Gains from Repeated Self-Distillation", NeurIPS 2024（arXiv 2407.04600） | 線形回帰の多段の自己蒸留で，段ごとの ξ を最適に取ると過剰危険度が最大 d 倍減る | 自己蒸留 | △ |
| Dang, Patil, Rinaldo, "Optimal Unconstrained Self-Distillation in Ridge Regression: Strict Improvements, Precise Asymptotics, and One-Shot Tuning", ICML 2026, PMLR 306:22834–22895（arXiv 2602.17565） | リッジ回帰の自己蒸留で混合の重み ξ を実数全体で最適化し，ξ* を危険度の微分で表し，格子探索なしの一度きりの推定法を出す | 自己蒸留．重みは危険度の微分で決まる | ○ |
| Dang, Patil, Rinaldo, "Prediction-Only Distillation in Linear and Logistic Regression"（arXiv 2607.15450．NeurIPS 2026 のポスター） | 元のラベルが無いとき，新しいラベル無しの入力に先生が付けた擬似ラベルで生徒を学び，先生と生徒の予測を最適な重みで混ぜる．重みはラベル無しだけでは決まらず，少数の較正用ラベルで一度で推定できる．ロジスティック回帰にも広げる | 混ぜるのは予測で，目的関数の項の重みではない | ○ |
| Ji & Zhu, "Knowledge Distillation in Wide Neural Networks: Risk Bound, Data Efficiency and Imperfect Teacher", NeurIPS 2020, 33:20823–20833 | 線形化した広い網で，完全な先生なら soft label の比率が高いほど良く，不完全な先生なら hard label が先生の誤りを直す | 最適な比率の式とラベル数への依存は抄録に無い（未確認） | ○ |
| Menon, Rawat, Reddi, Kim, Kumar, "A statistical perspective on distillation", ICML 2021, PMLR 139:7632–7642 | Bayes の確率を出す先生は生徒の目的関数の分散を下げる．近似の先生では偏りと分散の釣り合い | 重みの式ではなく先生の良さの基準 | ○ |
| Dao, Kamath, Syrgkanis, Mackey, "Knowledge Distillation as Semiparametric Inference", ICLR 2021 | 先生の確率を局外母数の差し込み推定とみなし，cross-fitting と損失の補正で先生の過学習・過少学習の影響を減らす | 重みではなく偏りの補正 | ○ |
| Zhou ほか, "Rethinking Soft Labels for Knowledge Distillation: A Bias–Variance Tradeoff Perspective", ICLR 2021（arXiv 2102.00650） | 標本ごとの偏りと分散の釣り合いから weighted soft labels | 標本ごとの重み | △ |
| Fang, Chen, Zhong, Ma, "Bayesian Knowledge Distillation", ICML 2024, PMLR 235:12935–12956 | 先生の予測確率から生徒の母数の事前分布を作り，蒸留の損失の最小化を事後分布の最頻値の推定とみなす | 事前分布の強さの決め方は抄録に無い（未確認） | ○ |

共通の違い：重みは標本ごとか，同じ標本の上の 2 種類の目標の混合比である．先生は同じデータか同じドメインで学ぶ．資源の総量を「ラベル何本ぶんか」で表して重みにする形は無い．

### 3.3 ラベルが増えるにつれて蒸留や擬似ラベルの重みを下げる仕組み

- **係数が n の関数として下がることを理論で示すもの**：PPI++（λ* ∝ 1/(1 + n/N)，◎），DeSSL（λ_opt ∝ n_u/(n_l + n_u)，◎），Jain 2024（α* = R_or(n)/(R_su + R_or(n))．Remark B.2，◎），Ben-David ほか 2010 の最適な α（既知．m_T ≥ d/A² でソースの重みが 0 になり，絶対量も下がる型．4.3 節）．
- **ベイズの事前分布**：catalytic prior の τ，power prior の a0·n0 は，絶対の重みを固定すれば割合 τ/(τ + n) が n とともに下がる（本調査の読み）．
- **学習の進み具合に合わせる段階づけ**：Zhu, Ding, Jacobson, Wu, Zhan, Jordan, Jiao, "Doubly-Robust Self-Training", NeurIPS 2023（arXiv 2306.00265，◎）は α_t = t/T（エポックで増やす）．重み自体は 1/(m + n) と 1/n に固定し，先生の正解率から決めない．ラベルの数に合わせる段階づけは見つからなかった．
- **経験的に「蒸留の得は少数データで大きい」**：Lanzillotta, Sarnthein, Kur, Hofmann, He, "Revisiting Knowledge Distillation: The Hidden Role of Dataset Size", arXiv 2510.15516（○．「the effect of distillation is not only preserved but amplified in low-data regimes」），Lopez-Paz ほか 2016（g2 ◎．「the teacher is most helpful in low data regimes」）．重みの式は無い．
- **価値そのものがデータの数で変わる**：Reimherr, Meng, Nicolae, "Prior sample size extensions for assessing prior impact and prior-likelihood discordance", JRSSB 83(3):413–437, 2021, doi:10.1111/rssb.12414（△）は，事前分布の標本数を尤度の標本数の関数として扱い，事前分布と尤度が食い違うときは負の値も許す．先生の換算の本数 n_T 自体が K で変わる場合の言い方の先例になる．
- **話者照合の MAP 適応**：Reynolds, Quatieri, Dunn, "Speaker verification using adapted Gaussian mixture models", Digital Signal Processing 10:19–41, 2000（書誌は △）．記憶では，適応の係数が n/(n + r)（r は relevance factor）で，事前のモデル（UBM）を r 本ぶんのデータとして扱い，データが増えるほど事前の重みを下げる形である．原典で式を確かめられなかった（未確認）．

### 3.4 「ラベルの無いデータ・事前学習・別のデータはラベル何本ぶんか」を測る研究（既知の 4 本より近いもの）

- **CLIP（Radford ほか 2021，◎）**：凍結特徴の上のロジスティック回帰の学習曲線を内挿し，ゼロショット分類器を何本／クラスに換算した．研究者の測り方に最も近い（2 節の 2）．
- **指示・合成データ・弱教師ありの換算**：Puri ほか, Findings of EACL 2023（△．指示 1 つは約 200 標本ぶん），Mikami ほか, ECML PKDD 2022, pp. 477–492（△．合成データでの事前学習の効果を尺度則で表す），Zhu, Shen, Mosbach, Stephan, Klakow, "Weaker Than You Think: A Critical Look at Weakly Supervised Learning", ACL 2023, pp. 14229–14253（△．弱教師ありの得は，きれいなラベルがクラスあたり 5 本あればそれで直接学ぶことでほぼ消え，約 30 本で頭打ち）．
- **推定での換算**：二重頑健な自己学習（◎．「The effective sample size is m+n, compared with effective sample size n」），Boyeau ほか, "AutoEval Done Right", ICML 2025（△．人のラベルの実効の標本数を最大 50% 増やす），Chaganty, Mussmann, Liang, ACL 2018（△．自動指標を制御変量にして人の評価の費用を 7〜13% 減らす）．
- **古典の換算**：Castelli & Cover, IEEE Trans. Inf. Theory 42(6):2102–2117, 1996（△．密度が既知で混合比だけ未知なら，ラベル付きとラベル無しの価値の比は Fisher 情報の比．密度も未知ならラベル付きは指数的に価値が高い），Sinha & Belkin, NIPS 2007, pp. 1361–1368（△），Robinson, Jegelka, Sra, ICML 2020（△．弱いラベルで事前学習すると強いラベル n 本での率が O(1/√n) から O(1/n) に上がりうる），Zhu, UAI 2020（△），Göpfert ほか, COLT 2019（△）．
- **既知（確かめ直していない）**：Hernandez ほか 2021（effective data transferred），Le Scao & Rush 2021，Wiesenfarth & Calderazzo 2020（Biometrics 76(1):326–336，doi:10.1111/biom.13124 は △ で確認），Hanneke & Kpotufe 2019．
- **名前の衝突に注意**：Cui, Jia, Lin, Song, Belongie, "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019, pp. 9268–9277（△）の「effective number of samples」は，クラスごとの標本数から損失のクラス重みを決める別の概念である．

共通の違い：これらは価値を本数で測るところで止まり，その本数を目的関数の重みに使わない（例外は 3.5 節のベイズの系統と Jain 2024）．

### 3.5 power prior・commensurate prior と，その機械学習・転移学習への応用

- **統計の系統**：Ibrahim & Chen 2000（△），Hobbs ほか 2011（△．過去と現在のデータの整合で借りる量を決める．「effective historical sample size」の語は Hobbs の博士論文の抄録で見た），Neuenschwander ほか 2010（△．MAP prior を何人ぶんの患者かで表す），Morita, Thall, Müller, "Determining the effective sample size of a parametric prior", Biometrics 64(2):595–602, 2008, doi:10.1111/j.1541-0420.2007.00888.x（△），Reimherr ほか 2021（△），Wiesenfarth & Calderazzo 2020（既知）．
- **事前の情報を擬似データとして足す系統（data augmentation prior）**：Bedrick, Christensen, Johnson, JASA 91(436):1450–1460, 1996（△．回帰面の選んだ点での平均応答に事前分布を置く．data augmentation prior はその特殊な場合），Greenland & Christensen, Statistics in Medicine 20:2421–2428, 2001（△），Discacciati, Orsini, Greenland, Stata Journal 15(3):712–736, 2015（△．事前分布を，それを生むのに要るデータ（prior-data records）として足す）．
- **機械学習寄りの応用**：catalytic prior（Huang ほか 2020，◎．2 節の 3），Polson & Sokolov, "Synthetic Priors", arXiv 2603.00347, 2026（○．リッジ回帰・catalytic prior・prediction-powered inference を 1 つの informativeness parameter でつなぐ．その決め方は抄録に無い），Suder, Xu, Dunson, "Bayesian Transfer Learning", arXiv 2312.13484（○．総説），Bretherton, Bon, Warne, Mengersen, Drovandi, "A Principled Approach to Bayesian Transfer Learning", arXiv 2502.19796（○．power prior の転移の母数を，ターゲットでの leave-one-out 交差検証で比べ，transfer sequential Monte Carlo で選ぶ），Fang ほか 2024（BKD，○）．
- **違い**：重みは整合の度合い，交差検証，危険度の推定で決める．学習曲線で測ったラベル換算の価値を a0 や τ にそのまま使う機械学習の研究は見つからなかった．

### 3.6 ソースとターゲットのデータを混ぜる比重を理論で決める研究（Ben-David 2010・Blitzer 2007 より近いもの）

- **Jain ほか 2024（◎）**：2 節の 1．学習曲線から α を予言し，漸近で α* = R_or(n)/(R_su + R_or(n))．
- **Chen, Owen, Shi, "Data enriched linear regression", Electronic Journal of Statistics 9(1):1078–1112, 2015, doi:10.1214/15-EJS1027（△）**：小さい高品質のデータと大きい偏りうるデータに別々の回帰を当て，予測の差に罰則をかける．罰則の強さ（大きいデータの重みに当たる）は差し込み（plug-in）か AICc で決める．係数が 5 個以上・誤差の自由度が 10 以上なら，偏りの大きさに依らず小さいデータだけを使うのは非許容（Stein 型）．
- **Kuzborskij & Orabona, "Stability and Hypothesis Transfer Learning", ICML 2013, PMLR 28:942–950（△）**：ソースの仮説に寄せる正則化（biased regularization）の最小 2 乗で，転移の母数を leave-one-out で選ぶ．
- **Hashimoto, "Model Performance Scaling with Multiple Data Sources", ICML 2021, PMLR 139:4107–4116（△）**：各ソースの量を変えたときの損失を予測する尺度則（有理関数の近似）を，少数の学習から当てはめる．複数のソースを扱う点は Jain より研究者の 3 資源に近い．混合比の最適化まで行うかは未確認．
- **Crammer, Kearns, Wortman, "Learning from Data of Variable Quality", NIPS 2005（○．PDF を要約器で読んだ）**：雑音の率が違う複数のソースから，上界を最小にする部分集合（先頭からいくつのソースを使うか）を選ぶ．重みづけではなく選択．
- **既知**：Ben-David ほか 2010，Blitzer ほか 2007，Hanneke & Kpotufe 2019，Reeve, Cannings, Samworth 2021（`docs/related-work.md` に既載）．

### 3.7 モダリティをまたぐ蒸留で，先生の項の重みを理論や価値の推定から決めるもの

- **見つからなかった．**確かめた範囲の重みの決め方は次のとおりである．
- Lopez-Paz, Bottou, Schölkopf, Vapnik, "Unifying distillation and privileged information", ICLR 2016（g2 ◎）：λ と T を掃くだけで，限界として「the lack of a proper weighting between labeled and unlabeled data」を挙げる．
- Jaritz ほか, "xMUDA: Cross-Modal Unsupervised Domain Adaptation for 3D Semantic Segmentation"（arXiv 1911.12676，◎ 前半）：モダリティ間の損失の重み λ_s・λ_t と擬似ラベルの重み λ_PL は超母数で，アブレーションでは λ_s = 1.0 に固定して λ_t を振る．値の選び方の記述は見つけられなかった．
- Wu, Xing, Zhang, Xie, Qu, "Fusion-then-Distillation"（FtD++，arXiv 2410.19446，◎）：5 つの項の λ は固定値（λ5 = 1.0 は「an empirical value」，λ1・λ2 は先行研究のまま，λ3〜λ5 は感度の掃引）．擬似ラベルの損失には 2 つの頭の予測の KL による点ごとの重み exp(−KL) を掛ける．
- Xue, Gao, Ren, Zhao, "The Modality Focusing Hypothesis: Towards Understanding Crossmodal Knowledge Distillation", ICLR 2023（○）：モダリティ間の蒸留が効くかは，先生がモダリティに共通する決定的な特徴にどれだけ注目するかで決まるという仮説．重みの規則は出さない．
- Vapnik & Izmailov, "Learning Using Privileged Information: Similarity Control and Knowledge Transfer", JMLR 16:2023–2049, 2015（△）：特権情報で学習を速める 2 つの仕組み（類似度の制御と知識の移し替え）．率の主張は未確認．
- 感情認識の別モダリティ蒸留は 3.1 節のとおり固定値か検証データでの選択である．

---

## 4. 論文で「新しい」と書くなら，どこまで書けるか

### 4.1 書いてよい言い方（案）

1. 「我々の知る限り，感情認識で，別モダリティの先生の弱いラベルの重みを，ターゲットのみの学習曲線で測った先生のラベル換算の価値から決め，K に対する重みの変化の予言を確かめた研究は無い」．範囲を感情認識に限り，「我々の知る限り」を付ける．
2. 「3 つの資源（感情ごとの K 本，ソースのコーパス，ソースだけで学んだ別モダリティの先生）の価値を，同じ凍結特徴の上の線形の頭の学習曲線で，感情ごとのラベル本数という同じ単位に換算して並べた」．測り方は CLIP の effective data efficiency of zero-shot transfer（Radford 2021）と effective data transferred（Hernandez 2021）に従うと明記する．
3. 「重みの規則 n_T : K は，Jain ほか 2024 の最適な重み（Corollary B.2）を，学習曲線が 1/n で下がる場合に本数で書き直したものに当たる．ベイズの言い方では，先生を n_T 本ぶんの擬似観測として足すこと（catalytic prior，power prior）に当たる」．規則は既知の理論から導いたものとして書き，新しさは，この設定で換算の本数を測って規則に使い，格子で選んだ重みと照合した点に置く．
4. 4.3 節の「絶対量が K とともに下がる」現象を説明できた場合に限り，その説明を新しい点として書ける．

### 4.2 書いてはいけない言い方

1. 「ラベル換算の価値（実効の標本数）で損失の重みを決める新しい原理を提案する」．Jain 2024 の α*，Li 2017 の λ = R_s/(R_s + R_y)，catalytic prior の τ（実効の事前標本数），power prior の a0·n0 が同じ考えである．
2. 「資源の価値をラベルの本数で測ったのは初めて」．CLIP 2021，Hernandez 2021，Le Scao & Rush 2021，Puri 2023，Castelli & Cover 1996 がある．
3. 「ラベルが増えると先生の重みを下げるべきことを初めて示した」．PPI++，DeSSL，Jain 2024 が係数の低下を式で示しており，ベイズの事後分布でも同じことが起きる．Ben-David ほか 2010 はターゲットが d/A² 本を超えるとソースの重みを 0 にすべきことを上界から示している．
4. 「格子探索なしで蒸留の重みを決める初めての方法」．Li 2017，Borup & Andersen 2021，Dang ほか 2026，SLaM 2023，PPI++ がある．
5. 仮定を書かずに「最適な重み」と書くこと．Jain の Corollary B.2 は，n と m が大きい漸近，誤差が足し算になる形，β = 1 の下での主張で，K ≤ 10 の少数ショットでは保証されない．論文では「近似の規則」と書き，実測で確かめた範囲を示す．
6. 「effective number」の語を定義せずに使うこと．Cui ほか 2019 のクラス重みと混同される．
7. 「融合から単モーダルへの蒸留は新しい」（既存の判定どおり．xMUDA，BFtD，FtD++ がある）．

### 4.3 数理の注意（本調査の計算．論文に書く前に確かめる）

- 「重みが下がる」には 2 つの意味がある．相対の係数（Jain の α，PPI++ の λ）が下がることと，ラベル何本ぶんかに直した先生の重みの絶対量が下がることである．2 乗誤差で偏りが 2 乗で効く型の先例（Jain，ベイズの擬似観測，PPI++，DeSSL）が予言するのは前者だけである．後者は Ben-David の上界の型にある（下の項）．
- Jain の Corollary B.2（β = 1）では，代わりのデータの総重みを本物何本ぶんかに直すと n·α*/(1 − α*) = Tr(H⁻¹K)/R_su(m) = n_su になり，n に依らず一定である．n とともに下がるのは割合 α* = n_su/(n_su + n) だけである．
- 1 次元のガウスの平均の例：本物 n 本の平均（分散 σ²/n）に，偏り b をもつ先生の値を n_0 本ぶんの重みで足すと，平均 2 乗誤差は (nσ² + n_0²b²)/(n + n_0)² で，これを最小にする n_0 は σ²/b² である．n に依らない．このとき誤差は σ²/(n + σ²/b²) になり，先生はちょうど σ²/b² 本ぶんの価値をもつ．先生だけの誤差 b² を本物だけの曲線 σ²/n で読んだ本数も σ²/b² なので，この例では「K = 0 で測った先生の本数をそのまま重みにする」規則が最適になる（数値で確かめた：σ² = 1，b² = 0.04 で n = 1〜100 のすべてで n_0* = 25）．
- PPI++ と DeSSL では，係数 λ は n とともに下がるが，ラベル付き 1 本を 1 とする単位に直した量 λ·n は n とともに増える（λ* ∝ 1/(1 + n/N) なら λ*·n ∝ nN/(N + n)）．
- Ben-David ほか 2010 の 6 節（リポジトリ直下の `A theory of learning from different domains.pdf` を pdftotext で読んだ．◎）は別の型である．上界を f(α) = 2B√(α²/β + (1−α)²/(1−β)) + 2(1−α)A と書き（α はターゲットの重み，A はソースとターゲットのずれ，B ≈ √(d/m) は複雑さの項．pdftotext では根号の記号が落ちるので，Theorem 3 の形と「B ≈ √(d/m)」の記述から根号つきと読んだ），最適な α* は m_T ≥ D² = d/A² のとき 1 になる．原文は「if there are enough target data (specifically, if mT ≥ D² = d/A²) then no source data are needed, and in fact using any source data will yield suboptimal performance」．ずれの罰 2(1−α)A がソースの重みに比例する（2 乗ではない）ので，ターゲットが増えるとソースの重みの絶対量も 0 まで下がる．
- 研究者の E59（`docs/exploratory-analyses.md`）では「本当のラベルが増えるほど弱いラベルの総重みが小さく選ばれ，K=25 ではほぼ使わない」．m_eff は弱いラベルの総重みをラベル何本ぶんかで表した絶対量なので，この結果は 2 乗誤差の型（絶対量は一定か増える）とは違い，Ben-David の上界の型（罰が重みに比例し，ターゲットが十分あると重みが 0）と向きが合う．UAR や 0/1 の誤りでは先生の誤りが重みに比例して効く可能性があり，そうなら後者の型で説明できる（本調査の推測．確かめていない）．
- 一方で「K = 0 で測った先生の本数 n_T を重みにし，割合 n_T/(n_T + K) で下げる」規則は 2 乗誤差の型に属し，絶対量を一定に保つ．研究者の予言をどちらの型で書くかを先に決める必要がある．絶対量が下がることを本数の言葉で書くなら，先生の換算の本数 n_T そのものが K とともに減る（たとえば先生の情報が K 本のラベルや出発点 W_0 の情報と重なる）と置くことになる．この形を扱った蒸留の先例は見当たらない．言い方の手がかりは Reimherr ほか 2021（事前分布の標本数を尤度の標本数の関数として扱い，負の値も許す）である．
- 学習曲線の指数が 1 でない場合（Jain の β ≠ 1）は，本数の比の形がずれる．研究者の学習曲線が 1/K で下がるかを先に確かめる必要がある．

---

## 5. 未確認の一覧

- Jain 2024：NeurIPS 版の頁番号（DOI 10.52202/079017-3499 は確認）．3 節の定理（ガウス列，ノンパラメトリック，高次元リッジ）での α* の形は読んでいない．
- PPI++：掲載先（arXiv の頁に記載なし）．本文の 10 万字以降（後半の実験と付録）は読んでいない．「effective sample size」の語が後半にあるか．
- PPI（Angelopoulos, Bates, Fannjiang, Jordan, Zrnic, Science 382:669–674, 2023）：DOI と本文．
- DeSSL：本文の 10 万字以降．定理 3.1 の式は要約器を通した読みで，原文の記号で確かめていない．
- SLaM：本文の 10 万字以降．
- Li 2017：頁番号．
- CLIP：図 7 のデータセットの数．
- Ji & Zhu 2020：最適な比率の式とラベル数への依存．
- Borup & Andersen 2021：閉じた式の形．
- Das & Sanghavi 2023，Pareek ほか 2024，Dang ほか 2026 の 2 本：本文．
- Zhou ほか 2021（ICLR）：重みの式と書誌の著者名．
- Fang ほか 2024（BKD）：事前分布の強さの決め方．
- xMUDA：λ_t・λ_PL の値と，CVPR 2020 の書誌．
- FtD++：IEEE TCSVT 35(9):9030–9045, 2025 の書誌（検索結果だけ）．
- Vapnik & Izmailov 2015：率の主張（O(1/√n) から O(1/n)）．
- Reynolds, Quatieri, Dunn 2000：MAP 適応の式（relevance factor r）．Gauvain & Lee 1994（IEEE Trans. Speech Audio Process. 2(2):291–298）の事前分布の数の扱い．
- O'Neill 1978（JASA，"Normal discrimination with unclassified observations"）と Ganesalingam & McLachlan 1978（Biometrika 65(3):658–662）の，線形判別でのラベル無しの観測の相対効率．O'Neill は書誌も確認できなかった．
- Schmidli ほか 2014（Biometrics 70:1023–1032）：書誌と本文．
- Hobbs ほか 2011：本文で「effective historical sample size」を使うか．
- Crammer, Kearns, Wortman 2008（JMLR 9）"Learning from multiple sources"：本文．
- Corduneanu & Jaakkola 2002（UAI，pp. 111–118）：重みの記号と定義．
- Takahashi（JMLR 27，arXiv 2205.07739）：擬似ラベルの係数の扱い．
- Shwartz-Ziv ほか 2022（NeurIPS，"Pre-Train Your Loss"）：事前分布の強さの選び方．
- Hashimoto 2021：混合比の最適化まで行うか．
- Puri ほか 2023：頁（1042–1057 と 1012–1027 の 2 説）．
- 二重頑健な自己学習・Pareek ほか 2024：NeurIPS 版の頁番号．
- Lanzillotta ほか 2025：査読つきの掲載先．
- Polson & Sokolov 2026：informativeness parameter の決め方．
- Aslam ほか 2023（CVPRW）と多先生の特権蒸留（arXiv 2408.09035）：重みの決め方．

---

## 付録 B：検索した語（主なもの）

- 感情認識：「cross-corpus speech emotion recognition few-shot target labels cross-modal teacher soft labels unlabeled target weight」「"effective sample size" OR "label-equivalent" emotion recognition transfer」「speech emotion recognition distillation from facial expression teacher unlabeled target corpus few labeled target samples weighting」「multimodal emotion recognition cross-corpus few-shot pseudo-labels from another modality teacher trained on source corpus」「emotion recognition "learning using privileged information" few labeled cross-corpus distillation weight」
- 重みの導出：「pseudo-label loss weight derived from effective sample size of teacher」「inverse-variance weighting teacher soft labels and ground-truth labels knowledge distillation」「knowledge distillation teacher as Dirichlet prior pseudo-counts equivalent sample size」「Bayesian knowledge distillation teacher predictions as prior prior sample size」「semi-supervised learning unlabeled loss weight set by equivalent number of labeled examples」「optimal distillation weight decreases as number of labeled samples increases theory」「self-training theory optimal weight of pseudo-labeled loss」「weak supervision combining weak labels with small set of gold labels optimal weight」
- 換算：「exchange rate between labeled and unlabeled data」「how many labeled examples is a weak label worth」「LLM-generated labels worth how many human labels」「few-shot CLIP zero-shot prior regularization strength set by effective number of shots」「weighted ERM real and synthetic data effective sample size」
- ベイズ・転移：「catalytic prior」「conditional means prior」「power prior transfer learning machine learning」「effective historical sample size commensurate prior」「Morita Thall Müller」「Neuenschwander MAP prior」「Reimherr Meng Nicolae」「data augmentation priors Greenland」「Kuzborskij Orabona hypothesis transfer」「Data enriched linear regression」「Hashimoto multiple data sources」「Scaling laws for learning with real and surrogate data」
- モダリティ間：「xMUDA」「fusion-then-distillation loss weight」「modality focusing hypothesis」「Vapnik Izmailov LUPI」

---

## 付録 A：文献ごとの記録（調べた順．作業中に書いたもの）

- [catalytic prior] Huang, Stein, Rubin, Kou 2020, PNAS 117(22):12004–12010, doi:10.1073/pnas.1920913117．本文 PDF を pdftotext で読んだ．簡単なモデル（切片だけ・主効果だけなど）を観測データに当てはめ，その予測分布から合成の応答 Y* を作り，合成の共変量（観測 X の再標本など）M 点に付けて，尤度を τ/M 乗して足す．「τ … can be interpreted as the effective prior sample size」．τ の決め方は (1) 予測の危険度の推定（パラメトリック・ブートストラップか Stein の不偏推定）の最小化，(2) τ に超事前分布を置く，の 2 通り．ブートストラップの予備推定 β̂(τ0) に使う初期値は線形回帰で τ0 = 1，それ以外で τ0 = p/4（最終の τ ではない）．先生（簡単なモデル）は同じデータで学ぶ．ソースや別モダリティではない．τ を先生の測ったラベル換算の価値から決めてはいない．
- [PPI++] Angelopoulos, Duchi, Zrnic, "PPI++: Efficient Prediction-Powered Inference", arXiv 2311.01453（v2 2024-03-26．掲載先は arXiv の頁に無い）．本文（HTML）を読んだ．目的関数 L_n(θ) + λ(L̃^f_N(θ) − L^f_n(θ))．最適な λ は分散を最小にする閉じた式（平均なら Cov(Y,f)/((1+n/N)Var f)．M 推定は Proposition 2）を差し込みで推定（"power tuning"）．ロジスティック回帰の例と実験あり．「effective sample size」の語は本文前半（10 万字まで）に無い．目的は母数の推定と信頼区間で，分類器の正解率ではない．予測の項はラベル付きの n 本での差を引いて偏りを消す形．
- [DeSSL] Schmutz, Humbert, Mattei, "Don't fear the unlabelled: safe semi-supervised learning via debiasing", ICLR 2023（arXiv 2203.07512）．本文（ar5iv）前半を読んだ．ラベル無しの項 λ/n_u ΣH − ラベル付きの項 λ/n_l ΣH で偏りを消す．定理 3.1 で分散を最小にする λ_opt = (n_u/n)·Cov(L,H)/Var(H)．ただし「the estimation of Cov(L, H) with few labels led to unsatisfactory results」で，実験は λ を振るか固定．ラベル換算の語は無い．
- [DR self-training] Zhu, Ding, Jacobson, Wu, Zhan, Jordan, Jiao, "Doubly-Robust Self-Training", NeurIPS 2023（arXiv 2306.00265）．本文（ar5iv）を読んだ．擬似ラベルの全体の項 − ラベル付きの上の擬似ラベルの項 + 本当のラベルの項．重みは 1/(m+n) と 1/n で固定．「The effective sample size is m+n, compared with effective sample size n」（予測が正確な場合）．先生の正解率から重みを決めない．実用では α_t = t/T のエポックの段階づけ．
- [Li 2017] Li, Yang, Song, Cao, Luo, Li, "Learning from Noisy Labels with Distillation", ICCV 2017（doi:10.1109/ICCV.2017.211．arXiv 1703.02391）．本文 PDF を pdftotext で読んだ．少数のきれいなデータ D_c で学んだ補助モデル（先生）の出力 s と雑音のあるラベル y を ŷ = λy + (1−λ)s で混ぜる．命題 1：2 乗距離の危険度 R で λ = R_s/(R_s + R_y) のとき最小 R_y R_s/(R_s+R_y)（逆分散の重み）．実験では λ = mAP_Dc/(mAP_D + mAP_Dc)（開発用データの mAP の比）．重みを先生の質から閉じた式で決める先例．ただし換算の単位はラベル本数ではなく危険度，ラベルの増え方による変化は扱わない．先生は同じドメインのきれいなデータで学ぶ．
- [SLaM] Kontonis, Iliopoulos, Trinh, Baykal, Menghani, Vee, "SLaM: Student-Label Mixing for Distillation with Unlabeled Examples", NeurIPS 2023（arXiv 2302.03806）．本文（ar5iv）前半を読んだ．先生のラベルが正しい確率 α(x) を，先生の余裕（margin）から検証データ上の isotonic 回帰で推定し，生徒の予測と先生のラベルを α(x) で混ぜる．ラベル付きと先生ラベル付きの間の重みは無い（1/|A∪B| で平均）．ラベル換算の語は無い．
- [Dang 2026a] Dang, Patil, Rinaldo, "Optimal Unconstrained Self-Distillation in Ridge Regression: Strict Improvements, Precise Asymptotics, and One-Shot Tuning", ICML 2026, PMLR 306:22834–22895（arXiv 2602.17565）．抄録（PMLR の頁）だけ．リッジ回帰の自己蒸留で混合の重み ξ を実数全体で最適化し，ξ* を危険度の微分で表し，格子探索なしの一度きりの推定法を出す．先生は同じデータで学ぶ．
- [Dang 2026b] Dang, Patil, Rinaldo, "Prediction-Only Distillation in Linear and Logistic Regression"（arXiv 2607.15450．NeurIPS 2026 のポスターの題は "Prediction-only distillation with optimal mixing in ridge-regularized linear and logistic regression"）．NeurIPS の頁の抄録だけ．元のラベルを使えないとき，新しいラベル無しの入力に先生が付けた擬似ラベルで生徒を学び，先生と生徒の予測を最適な重みで混ぜる．最適な重みはラベル無しだけでは決まらず，少数のラベル付きの較正データで一度で推定できる．ロジスティック回帰にも広げる．
- [Ji & Zhu 2020] Guangda Ji, Zhanxing Zhu, "Knowledge Distillation in Wide Neural Networks: Risk Bound, Data Efficiency and Imperfect Teacher", NeurIPS 2020（33:20823–20833．arXiv 2010.10090）．抄録だけ．「For a perfect teacher, a high ratio of teacher's soft labels can be beneficial」「hard labels can correct teacher's wrong prediction」．最適な比率の式とラベル数への依存は抄録に無い（未確認）．
- [Borup & Andersen 2021] Kenneth Borup, Lars N. Andersen, "Even your Teacher Needs Guidance: Ground-Truth Targets Dampen Regularization Imposed by Self-Distillation", NeurIPS 2021（arXiv 2102.13088）．抄録だけ．カーネル回帰の反復自己蒸留で，本当のラベルとの重みの最適値を各段で閉じた式で与え，深層学習でも格子探索より安く推定する．
- [Das & Sanghavi 2023] Rudrajit Das, Sujay Sanghavi, "Understanding Self-Distillation in the Presence of Label Noise", ICML 2023, PMLR 202:7102–7140（arXiv 2301.13304）．抄録と検索結果だけ．線形回帰で ξ の最適値を求め，ラベルの雑音が大きいと ξ > 1．ロジスティック回帰でも生徒が先生を上回る雑音の範囲を出す．
- [Jain 2024 ★最有力] Ayush Jain, Andrea Montanari, Eren Sasoglu, "Scaling laws for learning with real and surrogate data", NeurIPS 2024（arXiv 2402.04376v3, 2024-12-03）．本文 PDF を pdftotext で読んだ（1 節，式 (4)〜(6)，付録 B の Remark B.2・Corollary B.2，実験の設定）．重みつき ERM：本物の n 本に (1−α)/n，代わりのデータ（surrogate）m 本に α/m．代わりのデータは別のデータセット（Rotten Tomatoes，Goodreads），別の集団（TCGA の男性），**CIFAR10 の 1 万枚で学んだモデルの予測を CIFAR100 の画像に付けた擬似ラベル**など．実用の手順は「本物だけの学習曲線と代わりだけの学習曲線を当てはめ，式 (4) の尺度則を α について最小化する」（α を格子で選ぶ代わり．検証データでの選択も併記）．付録 B の Corollary B.2（低次元の漸近）で α* = R_or(n)/(R_su(m) + R_or(n))，R_or(n) = Tr(H⁻¹K)/n．**R_su(m) を本物の何本ぶんか（n_su）に直すと α*:(1−α*) = n_su : n になり，研究者の規則「先生 : K = 先生の本数 : K」と同じ形**（この書き直しは本調査の計算．論文は本数に換算していない．「effective sample size」の語も無い）．Remark B.2：ずれが大きく n が大きいと小さい α が最適．感情認識ではない，別モダリティの先生ではない，ラベル無しのターゲットの発話に付けた弱いラベルではない（CIFAR100 は別の画像）．
- [CLIP 2021 ★測り方の最有力] Radford ほか, "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021（arXiv 2103.00020）．本文 PDF を pdftotext で読んだ（3.1 節の few-shot との比較と図 7）．「estimates for the number of labeled examples per class that a logistic regression classifier on the same feature space requires to match the performance of zero-shot CLIP ... this estimates the effective data efficiency of zero-shot transfer」．1・2・4・8・16 本と全ラベルの学習曲線を対数線形で内挿して読む．中央値 5.4 本／クラス，平均 20.8，最大 184．**同じ凍結特徴の上の線形分類器の学習曲線で，ラベルを使わない先生の価値をクラスあたりの本数に換算する，という測り方そのもの．**一方で重みには使っていない：「use CLIP's zero-shot classifier as a prior for the weights of the few-shot classifier ... adding an L2 penalty towards the generated weights ... hyperparameter optimization would often select for such a large value of this regularizer that the resulting few-shot classifier was "just" the zero-shot classifier. Research into better methods of combining ... is a promising direction for future work.」
- [Power prior] Ibrahim & Chen, "Power prior distributions for regression models", Statistical Science 15(1):46–60, 2000, doi:10.1214/ss/1009212673（検索結果の抜き書き）．過去のデータの尤度を a0 ∈ [0,1] 乗して事前分布にする．GLM などに適用．
- [Commensurate prior] Hobbs, Carlin, Mandrekar, Sargent, "Hierarchical commensurate and power prior models for adaptive incorporation of historical information in clinical trials", Biometrics 67(3):1047–1056, 2011（検索結果の抜き書き）．過去と現在のデータの整合（commensurability）の度合いで借りる量を決める．「effective historical sample size」の語は Hobbs の博士論文の抄録で確認（2011 年の論文本文は未確認）．
- [MAP prior] Neuenschwander, Capkun-Niggli, Branson, Spiegelhalter, "Summarizing historical information on controls in clinical trials", Clinical Trials 7(1):5–18, 2010（二次資料の抜き書きだけ）．過去の対照群を変量効果のメタ解析でまとめ，予測分布を事前分布にする．事前分布の実効標本数（何人ぶんの患者を節約できるか）で表す．Schmidli ほか 2014（Biometrics 70:1023–1032．未確認）が頑健化．
- [Prior ESS] Morita, Thall, Müller, "Determining the effective sample size of a parametric prior", Biometrics 64(2):595–602, 2008, doi:10.1111/j.1541-0420.2007.00888.x（抄録の抜き書き）．
- [Prior sample size] Reimherr, Meng, Nicolae, "Prior sample size extensions for assessing prior impact and prior-likelihood discordance", JRSSB 83(3):413–437, 2021, doi:10.1111/rssb.12414（抄録の抜き書き．arXiv 1406.5958）．事前分布の標本数を，共役でない場合に広げ，**尤度の標本数の関数として扱い，事前分布と尤度が食い違うときは負の値も許す**．
- [Castelli & Cover 1996] V. Castelli, T. M. Cover, "The relative value of labeled and unlabeled samples in pattern recognition with an unknown mixing parameter", IEEE Trans. Inf. Theory 42(6):2102–2117, 1996（抄録の抜き書き）．密度が既知で混合比だけ未知なら，ラベル付きとラベル無しの相対的な価値は混合比についての Fisher 情報の比．密度も未知なら，ラベル付きは指数的に価値が高い．「ラベル無しの 1 点はラベル付き何点ぶんか」の古典．
- [Sinha & Belkin 2007] K. Sinha, M. Belkin, "The value of labeled and unlabeled examples when the model is imperfect", NIPS 2007（20:1361–1368．抄録の抜き書き）．
- [Kuzborskij & Orabona 2013] I. Kuzborskij, F. Orabona, "Stability and Hypothesis Transfer Learning", ICML 2013, PMLR 28:942–950（抄録の抜き書き）．ソースの仮説への偏った正則化（biased regularization）の最小 2 乗．転移の母数は leave-one-out で選ぶ．
- [Chen, Owen, Shi 2015] A. Chen, A. B. Owen, M. Shi, "Data enriched linear regression", Electronic Journal of Statistics 9(1):1078–1112, 2015, doi:10.1214/15-EJS1027（抄録の抜き書き）．小さい高品質のデータと大きい偏りうるデータに別々の回帰を当て，予測の差に罰則．罰則の強さは差し込み（plug-in）か AICc．係数 5 以上・誤差の自由度 10 以上なら，偏りの大きさによらず小さいデータだけを使うのは非許容（Stein 型）．
- [Hashimoto 2021] T. Hashimoto, "Model Performance Scaling with Multiple Data Sources", ICML 2021, PMLR 139:4107–4116（抄録の抜き書き）．データの構成（各ソースの量）を変えても損失を予測できる尺度則（有理関数の近似）を少数の学習から当てはめる．
- [Lanzillotta 2025] G. Lanzillotta, F. Sarnthein, G. Kur, T. Hofmann, B. He, "Revisiting Knowledge Distillation: The Hidden Role of Dataset Size", arXiv 2510.15516（査読つきの掲載先は頁に無い．抄録だけ）．蒸留の効果は少数データで増幅される（"data efficiency of distillation"）．重みの式は抄録に無い．
- [Iliopoulos 2022] F. Iliopoulos, V. Kontonis, C. Baykal, G. Menghani, K. Trinh, E. Vee, "Weighted Distillation with Unlabeled Examples", NeurIPS 2022（35．arXiv 2210.06711．抄録の抜き書き）．先生のラベルの雑音を打ち消す「debiasing」の標本ごとの重み（重要度の重みづけ），母数なし．
- [Menon 2021] A. K. Menon, A. S. Rawat, S. J. Reddi, S. Kim, S. Kumar, "A statistical perspective on distillation", ICML 2021, PMLR 139:7632–7642（抄録の抜き書き）．Bayes の確率を出す先生は生徒の目的関数の分散を下げる．近似の先生では偏りと分散の釣り合い．
- [Dao 2021] T. Dao, G. M. Kamath, V. Syrgkanis, L. Mackey, "Knowledge Distillation as Semiparametric Inference", ICLR 2021（抄録の抜き書き）．先生の確率を局外母数の差し込み推定とみなし，cross-fitting と損失の補正で先生の過学習・過少学習の影響を減らす．
- [Zhou 2021] H. Zhou ほか, "Rethinking Soft Labels for Knowledge Distillation: A Bias–Variance Tradeoff Perspective", ICLR 2021（arXiv 2102.00650．検索結果の抜き書き）．標本ごとの偏りと分散の釣り合いから weighted soft labels．
- [xMUDA] M. Jaritz, T.-H. Vu, R. de Charette, É. Wirbel, P. Pérez, "xMUDA: Cross-Modal Unsupervised Domain Adaptation for 3D Semantic Segmentation"（arXiv 1911.12676．CVPR 2020 は記憶で未確認）．本文（ar5iv）を読んだ範囲では，モダリティ間の損失の重み λ_s・λ_t と擬似ラベルの重み λ_PL は超母数で，アブレーションで λ_s = 1.0 に固定して λ_t を振る．値の選び方の記述は見つけられなかった．
- [MFH] Z. Xue, Z. Gao, S. Ren, H. Zhao, "The Modality Focusing Hypothesis: Towards Understanding Crossmodal Knowledge Distillation", ICLR 2023（arXiv 2206.06487．抄録の抜き書き）．モダリティ間の蒸留が効くかは，先生がモダリティに共通する決定的な特徴にどれだけ注目するかで決まる，という仮説．重みの決め方ではない．
- [CLAP 系の補足] 感情認識の別モダリティ蒸留 10 本の重みは，既存ノート g2（◎）で「固定値か検証集合で選ぶ．ラベルの本数ごとに選ぶ研究は無い」と確認済み（Albanie T=2，Srinivasan λ=30，UMT λ=50，Hajavi α=0.5，LiSER λ を検証で，PriCon α を検証で）．
- [FtD++] Y. Wu, M. Xing, Y. Zhang, Y. Xie, Y. Qu, "Fusion-then-Distillation: Toward Cross-modal Positive Distillation for Domain Adaptive 3D Semantic Segmentation"（arXiv 2410.19446v1．掲載先は検索結果で IEEE TCSVT 35(9):9030–9045, 2025 と出たが原典で未確認）．本文（HTML）を読んだ．全体の損失 L_seg + λ1〜λ5 の 5 項．λ は固定値（λ5 = 1.0 は「an empirical value」，λ1・λ2 は xMoSSDA に従い固定，λ3・λ4・λ5 は感度の掃引で選ぶ）．擬似ラベルの損失には 2 つの頭の予測の KL による点ごとの重み exp(−KL) を掛ける．先生の項の重みを価値の推定から決める記述は無い．
- [Polson & Sokolov 2026] N. Polson, V. Sokolov, "Synthetic Priors", arXiv 2603.00347（2026-02-27．抄録だけ）．Bedrick 1996 の条件付き平均の事前分布（専門家が選んだ点での平均応答に Beta 事前分布）を Pólya-Gamma で扱い，リッジ回帰・catalytic prior・prediction-powered inference を 1 つの「informativeness parameter」でつなぐ．その母数の決め方は抄録に無い．
- [Bedrick 1996] E. J. Bedrick, R. Christensen, W. Johnson, "A New Perspective on Priors for Generalized Linear Models", JASA 91(436):1450–1460, 1996（検索結果の抜き書き）．回帰面の選んだ点での条件付き平均に事前分布を置く．data augmentation prior はその特殊な場合．
- [Greenland] S. Greenland, R. Christensen, "Data augmentation priors for Bayesian and semi-Bayes analyses of conditional-logistic and proportional-hazards regression", Statistics in Medicine 20:2421–2428, 2001（検索結果の抜き書き．題名は原典で未確認）．A. Discacciati, N. Orsini, S. Greenland, Stata Journal 15(3):712–736, 2015（検索結果の抜き書き）：事前分布をそれを生むのに要るデータ（prior-data records）として足す．
- [Fang 2024 BKD] L. Fang, Y. Chen, W. Zhong, P. Ma, "Bayesian Knowledge Distillation: A Bayesian Perspective of Distillation with Uncertainty Quantification", ICML 2024, PMLR 235:12935–12956（抄録だけ）．先生の予測確率から生徒の母数の事前分布を作り，蒸留の損失の最小化を事後分布の最頻値の推定と同じとみなす．事前分布の強さの決め方は抄録に無い．多先生版（Fang ほか, arXiv 2605.27967, 2026．抄録だけ）はエントロピーで先生ごとの重みを変える．
- [CLIP 以外の換算] Puri, Mishra, Parmar, Baral, "How Many Data Samples is an Additional Instruction Worth?", Findings of EACL 2023（検索結果の抜き書き．頁は 1042–1057 と 1012–1027 の 2 説）：指示 1 つは約 200 標本ぶん．Mikami ほか, "A Scaling Law for Syn2real Transfer: How Much Is Your Pre-training Effective?", ECML PKDD 2022, pp. 477–492, doi:10.1007/978-3-031-26409-2_29（検索結果の抜き書き）．D. Zhu, X. Shen, M. Mosbach, A. Stephan, D. Klakow, "Weaker Than You Think: A Critical Look at Weakly Supervised Learning", ACL 2023, pp. 14229–14253（検索結果の抜き書き）：弱教師ありの得は，きれいなラベルがクラスあたり 5 本あるとそれで直接学ぶことでほぼ消え，約 30 本で頭打ち．
- [PPI 系の換算] P. Boyeau ほか, "AutoEval Done Right: Using Synthetic Data for Model Evaluation", ICML 2025（arXiv 2403.07008．検索結果の抜き書き）：PPI で「effective human-labeled sample size」を最大 50% 増やす．A. Chaganty, S. Mussmann, P. Liang, "The price of debiasing automatic metrics in natural language evaluation", ACL 2018（P18-1060．検索結果の抜き書き）：制御変量で人の評価の費用を 7〜13% 減らす．
- [率の換算] Robinson, Jegelka, Sra, "Strength from Weakness: Fast Learning Using Weak Supervision", ICML 2020（arXiv 2002.08483．検索結果の抜き書き）：弱いラベルで事前学習すると強いラベル n 本での率が O(1/√n) から O(1/n) に上がりうる．Zhu, "Semi-Supervised Learning: the Case When Unlabeled Data is Equally Useful", UAI 2020, PMLR 124（検索結果の抜き書き）．Göpfert ほか, "When can unlabeled data improve the learning rate?", COLT 2019（検索結果の抜き書き）．
- [名前の衝突] Y. Cui, M. Jia, T.-Y. Lin, Y. Song, S. Belongie, "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019, pp. 9268–9277（検索結果の抜き書き）．クラスごとの標本数から損失のクラス重みを決める．資源の価値の換算とは別の概念．
