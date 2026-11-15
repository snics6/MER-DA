# 先行研究の著者が未解決と書いた問いの一覧（定理の手順 0 段）

作成：2026-10-28．目的は，MER-DA の定理の出発点を，先行研究の著者が自分で「未解決」「今後の課題」「扱っていない」と書いた問いに置くための材料をそろえることである．本ファイルは定理も手法も提案しない．MER-DA の新しさも判定しない．

項目は MER-DA の設定に近い順に並べた．近さは，(1) その文献の設定が MER-DA の設定（クラスあたり K 本のターゲットのラベル，凍結した特徴の上の切片つき多クラスのロジスティック回帰，ソースで学んだアンカーへの L2 正則化つきの学び直し，ソースだけで学んだ別のモダリティの教師モデルの疑似ラベル，コーパス間のずれ）と共有する要素の数と，(2) 著者が挙げた問いが MER-DA の設定のどこに直接当たるか，で私が決めた．順位は判断であり，計算した値ではない．

---

## 0. 記号と書き方

| 記号 | 意味 |
|---|---|
| ◎ | 本文を読んで引用の文を確かめた（arXiv・PMLR などの PDF を pdftotext で文字にし，引用の文を検索して前後を読んだ．引用は原文のまま） |
| ○ | 抄録か序論だけを読んだ |
| △ | 書誌か検索結果の抜き書きだけ |

- 「引用」は著者の原文である．節の番号は PDF の見出しによる．番号の無い節は節の名前で書いた
- 「未解決の問い」は私の言い換えである．著者の文に無いことを足した所は「推測」と書いた
- 「後続」は，2025〜2026 年の文献を中心に，その問いを解いた文献があるかを調べた結果である．調べ方が項目ごとに違うので，調べた範囲を毎回書いた
- 「MER-DA との関係」の E 番号は `docs/results.md` 3.6 節と `docs/exploratory-analyses.md` の実験の番号である

---

## 1. 検索した場所と語，検索の限界

### 1.1 読んだ文献の取り方

- arXiv の PDF を curl で取り，pdftotext で文字にして，"future work"，"open question"，"open problem"，"remains open"，"we leave"，"left for"，"beyond the scope"，"does not address"，"limitation"，"conjecture"，"it would be interesting"，"unclear"，"unexplored" で検索し，当たった文の前後を読んだ．取った PDF は 56 本である（本一覧に載せなかった文献を含む）
- arXiv に無いものは出版元の PDF を使った：Li ほか（ICML 2026．PMLR の GitHub の assets），Tenzer ほか（AISTATS 2022．Weizmann 研究所の著者の頁），Zwieback ほか（Nonlinear Processes in Geophysics 2012．Copernicus），Gruber の博士論文（TU Wien の repositum），Ben-David ほか 2010（MER-DA の根にある PDF），Yang ほか（JMLR 2025．jmlr.org）
- TransCLIP（arXiv 2406.01837）は PDF の文字が取れなかったので，arXiv の HTML 版から結論の文を要約器（WebFetch）で抜き出した
- 既存のノート（`docs/related-work.md` 1.12〜1.18 節，`research_notes/` の各フォルダ，`reports/` の 3 本，`paper/reviews/round1/citation-audit.md`）に記録のある引用は，上の方法でもう一度原文で確かめてから載せた

### 1.2 後続の文献を探した語（WebSearch．標準と拡張）

- "semi-supervised domain adaptation fine-tuning finite-sample theory logistic regression cross-entropy few labeled target 2026"（拡張）
- "weak-to-strong generalization theory classification logistic regression cross-entropy misfit gain forward KL"
- "Relating Misfit to Gain in Weak-to-Strong Generalization Beyond the Squared Loss"
- "Quantifying Cross-Domain Knowledge Distillation in the Presence of Domain Shift"
- "theory knowledge distillation domain shift logistic regression soft labels teacher trained on source high-dimensional 2025 2026"
- "value of out-of-distribution data non-monotonic follow-up optimal weighting without knowing OOD samples"
- "more data can hurt / non-monotonic out-of-distribution source samples target error theory 2024 2025"
- "Scaling laws for learning with real and surrogate data"
- "semi-supervised unsupervised ensemble regression few labeled samples estimate regressor accuracy without ground truth spectral"
- "Crowdsourcing Regression: A Spectral Approach"，Zwieback 2012，Gruber 2016 の書誌
- "What Makes Multi-modal Learning Better than Single follow-up theory"
- "multimodal distillation domain adaptation adaptive modality weights per target domain few-shot MM-CDFSL follow-up"
- "last layer retraining few target samples theory domain shift linear probe sample complexity"
- "weak-to-strong generalization student initialized from pretrained head regularized toward pretrained model"
- "A Theory of Label Propagation for Subpopulation Shift"
- "arXiv 2026 theory pseudo-labels unlabeled target plus few labeled target samples logistic regression anchor regularization semi-supervised domain adaptation provable"（拡張）

### 1.3 検索の限界

- **引用した側からの網羅的な追跡はしていない．**Google Scholar の「被引用」，Semantic Scholar，OpenReview，dblp は使っていない．arXiv の API で語を組み合わせた検索を試みたが，応答が 429（回数の上限）で止まり使えなかった．このため「後続で解かれたか」は WebSearch の結果と，取った PDF の関連研究の節で見えた範囲に限る．2025〜2026 年に同じ問いを解いた文献があっても，見落としている可能性がある
- 後続を調べた深さは項目ごとに違う．上位 13 項目は 1 回以上の検索をした．下位の項目には「調べていない」と書いたものがある
- Zwieback ほか 2012 の 7 節（結論）は，PDF の文字の層が二重になっていて読めなかった．抄録，1 節，5.3 節だけを読んだ
- Gruber ほか 2016（JGR の extended collocation の論文），Gruber ほか 2017（TGRS），2019（ESSD），Yilmaz ほか 2012，Pierdicca ほか 2015・2017 は読んでいない．collocation の項目は Zwieback 2012 と Gruber の博士論文（2016）だけで書いた
- 著者が未解決の問いを書いていなかったので載せなかった文献：Rothenberg 1971（識別の古典．開いた問いの文を確かめていない．本文は未読），Hanneke & Kpotufe 2019（結論の節が無く，未解決の問いの文が見つからなかった），Kpotufe & Martinet（arXiv 1803.01833．「発散と γ の関係は不明」の 1 文だけ），Mansour ほか 2021（結論は一般的な期待だけ），Awasthi・Cortes・Mohri（Best-effort．結論は一般的な期待だけ），MME（Saito ほか 2019．結論に今後の課題が無い），Menon ほか 2020（教師モデルと生徒モデルの標本が別という仮定の注意だけ．MER-DA はこの仮定を満たす），Lang ほか 2024（問いが対照学習の事前学習に関するもので，凍結した特徴の MER-DA に当たらない），Chen ほか 2020（"explore other settings" の 1 文だけ）
- 中国語圏・日本語の論文誌と，IEEE Xplore・ACM DL の本文検索はしていない

---

## 2. 一覧（MER-DA の設定に近い順）

### 1. Ha & Chen：少数のターゲットのラベルによる半教師ありの DA の有限標本の理論を，ロジスティック損失・交差エントロピーに広げること

- **文献**：Wooseok Ha, Yuansi Chen, "When few labeled target data suffice: a theory of semi-supervised domain adaptation via fine-tuning from multiple adaptive starts," arXiv:2507.14661v2（2026-06-30）．掲載先は本文に無い．水準 ◎
- **引用**
  - 2.1 節："The squared loss is used to simplify theoretical discussions, although several motivating examples below involve binary or multiclass classification problems, such as image classification. Extending the finite-sample analysis to logistic, cross-entropy, or other losses is left for future work."
  - 7 節："Looking ahead, extending our analysis beyond anticausal linear SCMs to mixed causal-anticausal or nonlinear SCMs is a promising direction. ... Integrating ideas from causal representation learning to recover latent structure and adapt in representation space remains an open challenge."
- **未解決の問い**：教師なしの DA の解を出発点にして，低い次元の部分空間だけを少数のターゲットのラベルで直すと，超過リスクが d/n から r/n（r は介入の次元）に下がる．この結果は二乗損失でしか示されていない．ロジスティック損失や交差エントロピーで同じ形の有限標本の保証が成り立つかは未解決である．表現の空間で適応する場合も未解決である
- **設定と仮定**：反因果の線形構造因果模型（ラベル Y が共変量 X を生む）．予測器は切片の無い線形，損失は二乗．ずれは 3 型（共通の隠れ因子による共分散のずれ，回帰係数の疎な変化，ラベルの現れ方 b の変化）．ターゲットのラベル n(0) ≥ r log(1/δ)，ソースとラベル無しのターゲットは d 以上．候補の中からの選択（MASFT）には別の検証用のターゲットのラベル n_val を使う
- **後続**：拡張検索で，有限標本・半教師ありの DA・ロジスティック損失をそろえた 2025〜2026 年の理論は見つからなかった．近いものは Kostin ほか（arXiv 2606.12680，2026-06．少数のターゲットのラベルの教師ありの DA．線形回帰のまま．本一覧の 11 番）と，"A Unified Analysis of Generalization and Sample Complexity for Semi-Supervised Domain Adaptation"（arXiv 2507.22632．深層網の特徴の整合の標本複雑度で，出発点からの微調整もロジスティック損失の解析も無い．抄録と序論で確かめた，○）である
- **MER-DA との関係**：当たる設定は，K 本，切片，多クラス，ロジスティック損失，アンカーつきの学び直し，コーパス間である．Ha & Chen の「直す部分の次元 r」は，MER-DA で切片だけを直す場合（r に当たる数が小さい）と決定境界の向きも直す場合に分けて読める（この対応は `docs/related-work.md` 付録の読みで，原文には切片が無い）．データでは，E28（切片が自由な仮説転移で揺れの分散が 1/(2K)）と E69（K 本の損失の 2 階微分の固有方向で学び直しの動きを分けた）の数値が，ロジスティック損失での「直る部分と K」の関係を照らす実測になる

### 2. Li, Han, Yang, Tong：ずれのもとでの蒸留の超過リスクの式を，線形回帰の外に広げること

- **文献**：Xiangchao Li, Xiao Han, Qing Yang, Xin Tong, "Quantifying Cross-Domain Knowledge Distillation in the Presence of Domain Shift," ICML 2026（PMLR 306）．水準 ◎（1 節の動機の段落と 6 節を今回読んだ．1〜3 節の内容は `docs/related-work.md` 1.16 節で読んだもの）
- **引用**
  - 6 節："While our current analysis focuses on linear models and observes a novel double descent phenomenon, future work should extend these theoretical frameworks to nonlinear architectures to further generalize these findings and provide a rigorous characterization of their risk landscapes."
  - 1 節（Ildiz ほか 2025 の限界として）："Moreover, their results are restricted to the setting ξ = 1 (i.e., pure teacher supervision), leaving the trade-off between distillation and learning from observed student data unexplored."
- **未解決の問い**：ソースで学んだ教師モデルの予測と真のラベルを ξ : (1 − ξ) で混ぜる最適な ξ を閉じた形で出したのは，線形回帰と二乗損失の場合だけである．非線形の模型（著者の言う範囲）での式は未解決である．分類やロジスティック損失は本文で扱っていない（本文の検索で "logistic" の語は関連研究にしか出ない）
- **設定と仮定**：教師モデルはソースの N1 本のリッジ回帰．生徒モデルはターゲットの N2 本の同じ入力で，教師モデルの予測と真のラベルの混合を目標にリッジ回帰を学ぶ．正則化の中心は 0．ずれは共変量のずれとモデルのずれ．次元と標本数が同じ速さで大きくなる極限
- **後続**：ICML 2026（2026 年 7 月）の論文で，後続は検索で見つからなかった．同じ方向のより前の研究に Ildiz ほか（ICLR 2025，arXiv 2410.18837．リッジ無し回帰で，代理のモデルの予測で目標のモデルを学ぶ．ずれを含む）がある．Ildiz ほかの結論も "a non-asymptotic characterization ... remains an open problem" と書く（◎）
- **MER-DA との関係**：当たる設定は，別のモダリティの教師モデル，ラベル無しの発話全体への疑似ラベル，コーパス間，ロジスティック損失，アンカーつきの学び直しである．Li ほかの ξ は，MER-DA の疑似ラベルの総重み m と K 本の重みの比に当たる（違い：Li ほかは教師モデルの予測をラベルのある N2 本の上でだけ使い，教師モデルと生徒モデルが同じ入力を見る）．本文に最適な ξ を推定する手続きは無い（これは `docs/related-work.md` 1.16 節の私たちの読みで，著者の文ではない）．データでは，E59 の追加（m を K ごとに仮のターゲットで選ぶと K=10 以上の損が消えた）と E65・E66（教師モデルの価値をラベルの本数で読んだ値が，同時に学ぶ形では K とともに 3〜6 分の 1 に減り，アンカーを動かしてから学び直す形では音声と顔でほぼ一定）が，最適な重みが K とともにどう動くかの実測になる

### 3. Lopez-Paz ほか：一般化蒸留で，ラベルのある標本と教師モデルの軟らかいラベルだけの標本の重みをどう決めるか

- **文献**：David Lopez-Paz, Léon Bottou, Bernhard Schölkopf, Vladimir Vapnik, "Unifying Distillation and Privileged Information," ICLR 2016, arXiv:1511.03643v3．水準 ◎
- **引用**
  - 5 節の実験 6（半教師あり）："We believe that the drops in performance for some distillation temperatures are due to the lack of a proper weighting between labeled and unlabeled data in (4)."
  - 4.1 節：理論の部分は "we will place some assumptions about how the student, teacher, and true function interplay" で始まり，学習の速さを仮定として置く（定理ではない）
  - 4.2 節（MER-DA の設定の根拠）："if the multiple tasks share the same labels yi but differ in their input modalities, the input modalities from the source tasks are privileged information."
- **未解決の問い**：教師モデルが全部の標本に軟らかいラベルを付け，生徒モデルが少数の硬いラベルと一緒に学ぶとき，2 種類の標本の重みの決め方が無い．なぜ蒸留が効くかの説明も，率を仮定として置いた形にとどまる
- **設定と仮定**：教師モデルは特権情報（学習時だけある入力）で学ぶ．生徒モデルは通常の入力だけを使う．損失は (1 − λ) × 真のラベルの損失 ＋ λ × 温度つきの軟らかいラベルの損失．理論の部分は推定誤差の率 O(|F|_C/n^α) を仮定する
- **後続**：Li ほか（本一覧の 2 番）は混合の重み ξ を線形回帰で閉じた形にしたが，教師モデルの予測を使うのはラベルのある標本の上だけである．Jain・Montanari・Sasoglu（本一覧の 9 番）は重みつきの経験リスク最小化でソースのラベル付きデータの重みを扱うが，軟らかいラベルではない．調べた範囲で，ラベル無しの標本の軟らかいラベルとラベルのある標本の重みを分類の損失で決めた理論は見つからなかった
- **MER-DA との関係**：当たる設定は，別のモダリティの教師モデル（4.2 節の文がそのまま当てはまる），ラベル無しの発話全体への疑似ラベル，K 本である．MER-DA は重み m を仮のターゲットで K ごとに選んでいる．データでは，E59 の追加（m を K ごとに選ぶと K=25 ではほぼ蒸留しない値が選ばれる）と E61（一般化蒸留を同じ生徒モデル・同じ K で比べた．B4 は不支持）が，重みと K の関係の実測になる

### 4. Hatano ほか（MM-CDFSL）：ドメインをまたぐ少数ショットで，モダリティごとの蒸留の重みをターゲットに合わせて変えること

- **文献**：Masashi Hatano, Ryo Hachiuma, Ryo Fujii, Hideo Saito, "Multimodal Cross-Domain Few-Shot Learning for Egocentric Action Recognition," ECCV 2024, arXiv:2405.19917．水準 ◎（結論の段落）
- **引用**（結論の "Limitations and future work"）："our approach applies constant loss weights for feature distillation losses during the multimodal distillation process, regardless of the specific target dataset. This strategy does not account for the varying significance of modalities based on the target domain ... Dynamically adjusting distillation weights according to the modality's relevance in the target domain is crucial for achieving more targeted and efficient training outcomes. We will leave this for our future efforts."
- **未解決の問い**：複数のモダリティの教師モデルから 1 つのモダリティの生徒モデルへ蒸留するとき，どのモダリティの教師モデルをどれだけ重く使うかを，ターゲットのドメインごとに決める方法が無い
- **設定と仮定**：映像（RGB）・オプティカルフロー・手の関節の教師モデルから RGB の生徒モデルへ特徴を蒸留する．ターゲットのラベル無しの映像で教師モデルを適応させる．評価はクラスあたり 1 本と 5 本の少数ショット．深層網，損失の重みは定数
- **後続**：Guo, Jiao, Guo, Song, "Multimodal Helps Unimodal: Frequency-Aware Distillation for Cross-domain Few-shot Egocentric Action Recognition"（Information Fusion 137:104636，2026）が同じ設定の続きと推定されるが，本文も抄録も取れなかった（△．`research_notes/TAC 論文 先行研究の精読と穴/g2_crossmodal_distillation_er.md`）．検索で見つかった "Cross-Modal Distillation For Widely Differing Modalities"（arXiv 2507.16296）は標本ごとの重みで，ターゲットごとのモダリティの重みではない（検索結果の要約だけ，△）．解いた文献は見つからなかった
- **MER-DA との関係**：当たる設定は，別のモダリティの教師モデル，コーパス間，K 本である．MER-DA は教師モデルの種類と m を仮のターゲットで K ごとに選んでおり，この問いに答える形の 1 つになっている．データでは，E60（AV・VT・AT・AVT の世界で，生徒モデルごとに選ばれた教師モデルと得）と E65（モダリティごとの教師モデルの価値をラベルの本数で読んだ値）が，ターゲットのコーパスごとに最適なモダリティの重みが違うかを示す実測になる

### 5. Charikar・Pabbaraju・Shiragur（と Xue・Li・Mirzasoleiman）：弱から強への一般化の得を分類で量にすること，誤りの種類が得を変えるか

- **文献**
  - Moses Charikar, Chirag Pabbaraju, Kirankumar Shiragur, "Quantifying the Gain in Weak-to-Strong Generalization," NeurIPS 2024, arXiv:2405.15116v2．水準 ◎（結論と付録 E）
  - Yihao Xue, Jiping Li, Baharan Mirzasoleiman, "Representations Shape Weak-to-Strong Generalization: Theoretical Insights and Empirical Predictions," ICML 2025（PMLR 267）, arXiv:2502.00620．水準 ◎（5 節の該当の文）
- **引用**
  - Charikar ほか，結論："Our theorems notably do not apply when the set from which the strong model learns the finetuning task is not convex. ... Our work also does not address classification tasks (see also Appendix E), and it would be interesting to see if similar results could be obtained for more general loss functions."
  - Charikar ほか，付録 E："the performance gain in classification ought also be (mathematically) characterized, at least to some extent, by the misfit between the weak and strong model; however, additional regularization terms in the objective (along with other error terms) seem to be necessary." 続けて "do different forms of disagreement between the strong and weak model (e.g., random errors vs correlated errors), for the same weak model accuracy, lead to differing weak-to-strong generalization?"
  - Xue ほか，5 節："While Thm. 3.8 is based on linear regression with MSE loss, Exps. II and III demonstrate that the correlation also holds for classification tasks using CE finetuning loss"
- **未解決の問い**：教師モデルの疑似ラベルで学んだ生徒モデルが教師モデルをどれだけ上回るかを，教師モデルと生徒モデルの食い違い（misfit）で書く式は，二乗損失の回帰で凸な関数の集合のときだけ証明されている．分類では，正則化の項を足す必要がありそうだと著者が書き，さらに教師モデルの正解率が同じでも誤りの入り方（でたらめな誤りか，相関した誤りか）で得が変わるかを問いとして残している
- **設定と仮定**：生徒モデルは強い表現の上の凸な関数の集合から，教師モデルのラベルへの二乗誤差を最小にする．実現可能性（真の関数が生徒モデルの集合に入る）を仮定する．真のラベルは使わない．正則化の中心は無い（0）
- **後続**（部分的に解かれた）
  - Mulgund & Pabbaraju, "Relating Misfit to Gain in Weak-to-Strong Generalization Beyond the Squared Loss," ICML 2025（PMLR 267）, arXiv:2501.19105．水準 ◎（抄録，1 節，6 節）．Bregman ダイバージェンス（交差エントロピーを含む）に広げた．ただし保証の付く学び方は "the strong model's output is in the first argument in the KL divergence—contrast this to standard supervised classification with the cross-entropy loss, where instead, the supervisor's labels are in the first argument"（1 節）であり，通常の交差エントロピーとは KL の向きが逆である．凸でない場合は k 個のロジスティック回帰の頭の凸結合で扱い，k とともに消える誤差の項が残る．6 節は "Can the proof of Theorem 4.3 be improved to tighten the error term?" を未解決として挙げる
  - Xu, Yao, Wang, Liu, "Weak-to-Strong Generalization via Bregman Bias–Variance Decomposition," arXiv:2505.24313v3（2026-08）．水準 ○（抄録と序論）．生徒モデルの関数の集合が凸という仮定を外し，交差エントロピーでは生徒モデルの予測のエントロピーを下げると得が出やすいこと，逆向きの交差エントロピーは教師モデルの予測の不確かさに影響されにくいことを示したと抄録に書く
  - 調べた範囲（上の 2 本の抄録と序論）では，(a) 真のラベル K 本との併用，(b) 0 でないアンカーへの正則化，(c) 教師モデルと生徒モデルの間のドメインのずれ，(d) 付録 E の「誤りの種類」の問い，を扱った文は見つからなかった．2 本の本文は付録まで読んでいないので，(d) を扱っている可能性は残る
- **MER-DA との関係**：当たる設定は，ロジスティック損失，多クラス，別のモダリティの教師モデル，K 本，アンカーつきの学び直しである．MER-DA の生徒モデルの損失は通常の向きの交差エントロピー（教師モデルの確率が第 1 引数）であり，Mulgund & Pabbaraju の保証の向きとは逆である．`docs/related-work.md` 1.14 節 (a) が示したとおり，教師モデルが直す割合と壊す割合の差は教師モデルとアンカーの正解率の差に等しい．データでは，E60・E61 の教師モデルとアンカーの予測の正誤の組と，E66 の対照 p（アンカー自身の予測を疑似ラベルにしたもの）と教師モデル pF の差が，誤りの入り方と得の関係を測る材料になる

### 6. Medvedev ほか：弱から強への一般化の限界の定理は原点から学ぶ生徒モデルにだけ当てはまり，学習済みの重みに寄せる生徒モデルは対象外であること

- **文献**：Marko Medvedev, Kaifeng Lyu, Dingli Yu, Sanjeev Arora, Zhiyuan Li, Nathan Srebro, "Weak-to-Strong Generalization Even in Random Feature Networks, Provably," arXiv:2503.02877v3（2025-11）．掲載先は本文で確かめていない．水準 ◎（定理 4.3 の後の注意の段落）
- **引用**："The main caveat here is that the limitation applies only with gradient flow initialized at the origin, with convex hypothesis classes containing the origin or with convex regularizers minimized at the origin. ... In contrast, consider a student that fine-tunes a pre-trained top layer, starting from the pre-trained initialization (or alternatively, explicitly regularizes to be close to the pre-trained model). Such a student is not shrink optimal and is not subject to the limitation of Theorem 4.3."
- **未解決の問い**：生徒モデルが教師モデルをどこまで上回れるかの上限（定理 4.3）は，正則化の中心が原点のときにしか成り立たない．学習済みの重み（MER-DA のアンカー）に寄せる正則化をかけた生徒モデルについて，教師モデルとの差の上限と下限は示されていない（この文は限界の注意であり，著者は今後の課題とは書いていない．問いとして読むのは私の読みである）
- **設定と仮定**：ランダム特徴の網（2 層の 1 層目を固定）．教師モデルは少ない特徴，生徒モデルは多い特徴．二乗損失，勾配流の早期終了．目標の関数は正規化されている
- **後続**：Yao ほか, "On the Blessing of Pre-training in Weak-to-Strong Generalization," arXiv:2605.05710（2026-05）．水準 ○（抄録と Limitations）．事前学習を「スペクトル初期化」として模型にし，初期化が「効く領域」に入ると弱から強への一般化が起きることを単一指標の模型で示した．初期化の扱いであり，学習済みの重みへの L2 正則化は扱っていない．Limitations は "our analysis naturally relies on modeling unsupervised pre-training as a spectral initialization step (PCA) under spiked Gaussian data assumptions" と書く．アンカーへの正則化の場合を解いた文献は見つからなかった
- **MER-DA との関係**：当たる設定は，アンカーつきの学び直し，別のモダリティの教師モデル，K 本である．MER-DA の生徒モデルはまさに「学習済みの重みに寄せる」側にある．データでは，E66 の 2 つの形（A：疑似ラベルと K 本を同時に学ぶ，B：疑似ラベルでアンカーを動かしてから K 本で学び直す）と，E63（LoRA で微調整した版ではアンカーへの罰則が +0.4 しか効かない）が，アンカーの有無と強さで教師モデルとの差がどう変わるかの実測になる

### 7. Dong ほか：弱から強への一般化を，偏りが主の体制と分散が主の体制で 1 つの見方にまとめること

- **文献**：Yijun Dong, Yicheng Li, Yunai Li, Jason D. Lee, Qi Lei, "Discrepancies are Virtue: Weak-to-Strong Generalization through Lens of Intrinsic Dimension," ICML 2025（PMLR 267）, arXiv:2502.05075v6．水準 ◎（"Limitations and future directions" の節と脚注 6）
- **引用**："In contrast, in the bias-dominated regime, the lower approximation error of the strong student is generally brought by the larger "capacity" of the strong model corresponding to a higher intrinsic dimension (Ildiz et al., 2024; Wu & Sahai, 2024). This calls for future studies on unified views and transitions between the two regimes ... Toward this goal, a limitation of our analysis is the quantification of the advantage of W2S in bias (see Footnote 6), which could be a promising next step." 脚注 6："Quantifying the advantage of W2S in bias requires further assumptions on the downstream task D(f*) and the covariance matrices Σw, Σs ... could be an interesting future direction."
- **未解決の問い**：教師モデルの誤りが分散（ラベルの雑音）由来なら生徒モデルは消せる．誤りが偏り（教師モデルの表現の限界）由来のとき，生徒モデルがどれだけ得をするかは量になっていない．2 つの体制の間の移り変わりも未解決である
- **設定と仮定**：教師モデルは n 本の真のラベルでリッジ無し回帰，生徒モデルは N 本のラベル無しの標本に教師モデルが付けたラベルでリッジ無し回帰．特徴は低い内在次元を持つ．近似誤差がラベルの雑音より十分小さい（分散が主の体制）．回帰で切片は無い．ドメインのずれは無い
- **後続**：Xu・Yao ほか（5 番の後続．Bregman の偏りと分散の分解）が統一に近い方向だが，抄録だけでは偏りの体制の得を量にしたかは分からない（○）．解いた文献は確かめられなかった
- **MER-DA との関係**：当たる設定は，別のモダリティの教師モデル，K 本，アンカーつきの学び直しである．別のモダリティの教師モデルの誤りは，生徒モデルのモダリティでは表せない情報の欠けに由来するので偏りが主だと考えられる（推測）．データでは，E66 の「教師モデルだけが持つ情報の分」（pF − p）が K によらずほぼ一定（音声・m=2000 で 3.2〜3.6 ポイント）であることと，E69 の固有方向ごとの分け方が，偏りが主の体制で得がどう決まるかの実測になる

### 8. De Silva ほか：ソースの本数に対して誤差が増減する現象の理論と，ソースとターゲットの重みを手元のデータだけで決めること

- **文献**：Ashwin De Silva, Rahul Ramesh, Carey E. Priebe, Pratik Chaudhari, Joshua T. Vogelstein, "The Value of Out-of-Distribution Data," ICML 2023（PMLR 202）, arXiv:2208.10967v5．水準 ◎
- **引用**
  - 2 節の Remark 2："Current tools in learning theory ... are fundamentally about understanding generalization when the number of samples is asymptotically large—whether they be from the target or OOD. In future work, we hope to formally characterize this non-monotonic trend in generalization error by building new learning-theoretic tools."
  - 3.2 節（Limitations of the proof-of-concept solution）："this procedure is dependent on two critical ideal conditions: (1) We must know which samples in the dataset are OOD, and (2) We must have a held out dataset of target samples to tune the weight α. ... we hope that it would serve as a proof-of-concept solution that motivates future research into accurately identifying OOD samples within datasets, designing ways of determining the optimal weights, and developing better procedures for exploiting OOD samples"
- **未解決の問い**：ソースを足すとターゲットの誤差が初めは下がり後で上がる現象が，どの条件で起きるかを一般の形で述べた定理は無い（論文は存在を示す例だけ）．重み α で直せるが，その α をターゲットの別の検証データ無しに決める方法が無い
- **設定と仮定**：理論は 1 次元・2 クラスの Fisher 線形判別．ソースはターゲットを Δ だけ平行移動した分布．罰則は無い．実験は CNN をゼロから学ぶ
- **後続**：検索（2 回）では，重みを検証データ無しで決める理論も，非単調の条件を一般の形で述べた理論も見つからなかった
- **MER-DA との関係**：当たる設定は，K 本，切片（理論の閾値），コーパス間である．MER-DA は条件 (1)（どれがソースかは分かる）を満たし，条件 (2) の代わりに仮のターゲットで選ぶ．De Silva ほかの合併の閾値の式（μ_h = mΔ/(n+m)）は MER-DA の E15 の λ の最も近い先例である（`docs/related-work.md` 1.13 節）．データでは，地図（K が小さいと合併が得，大きいと損．K* の位置），E15・E17（合併が位置のずれを直す割合 λ），E26（全ソースを混ぜると最良の 1 つより 0.5〜1.3 点悪い）が，非単調が起きる条件の実測になる

### 9. Jain・Montanari・Sasoglu：代わりのデータとの混合の重みの理論を分類の損失に広げること，本物と代わりのデータで尺度則の指数が違う場合

- **文献**：Ayush Jain, Andrea Montanari, Eren Sasoglu, "Scaling laws for learning with real and surrogate data," NeurIPS 2024, arXiv:2402.04376v3．水準 ◎
- **引用**
  - 1 節の結果の要約："The above conclusion holds under the condition that α can be tuned (nearly) optimally."
  - 実験の節（Gaussian mixture の段落）："Although our theoretical results do not apply directly to classification loss, we believe that our qualitative conclusions generalize."
  - Discussion 節："In practice, we observe often two somewhat different exponents βor ≠ βsu. In these cases, we set β = βor, and this appears to work reasonably well. However, we can imagine cases in which the difference between βor and βsu is significant enough (4) will stop being accurate."
- **未解決の問い**：本物のデータと代わりのデータ（ソース）を重み α で混ぜたときの誤差の尺度則と最適な α は，二乗損失（ガウスの系列模型，ノンパラメトリック回帰，高次元のリッジ回帰）と低次元の漸近でしか導かれていない．分類の損失での理論と，2 種類のデータで学習曲線の指数が違うときの式は未解決である
- **設定と仮定**：重みつきの経験リスク最小化 (1 − α)×本物の損失 ＋ α×代わりの損失．代わりのデータはラベル付き．ベイズ誤差が 0 に届くと仮定（Discussion で外す方向を述べる）
- **後続**：検索では，分類の損失での後続の理論は見つからなかった
- **MER-DA との関係**：当たる設定は，K 本，コーパス間，ロジスティック損失である．MER-DA の「ソースの価値をラベルの本数に換算する」読み方は，この尺度則と同じ種類の量を扱う．データでは，E32（1 ソースの仮説転移の価値は感情ごとに約 10 本で K=5〜25 でほぼ一定，合併の価値は K とともに減り K=25 で負）と E65（教師モデルの価値の本数）が，ソースや教師モデルで学習曲線の形が本物のラベルと違うかの実測になる

### 10. Chen & Bühlmann：DA の手法を正しく選ぶのに要るターゲットのラベルの最小の数

- **文献**：Yuansi Chen, Peter Bühlmann, "Domain adaptation under structural causal models," JMLR 22(261), 2021, arXiv:2010.15764v2．水準 ◎（結論の段落と 3 節の 3 つの問い）
- **引用**
  - 結論："The assessment of the goodness of a model or algorithm is difficult in general but it can be done when having access to a small fraction of labeled target data. We show empirically that a small fraction of labeled target data substantially helps to select the best DA method. It remains an open question what is the minimal amount of labeled target data points in order to guarantee the best DA method selection."
  - 結論："The theoretical extension to nonlinear SCMs is a challenging future direction."
  - 3 節："The general solution to the third question remains open."（3 つ目の問いは "If the prediction direction is a mix of causal and anticausal, is domain adaptation still beneficial?"）
- **未解決の問い**：候補の DA の手法の中から最良のものを選ぶことを保証するには，ターゲットのラベルが何本あればよいかが分かっていない
- **設定と仮定**：線形の構造因果模型．DIP・CIP・CIRM などの線形回帰の DA の手法．ソースの環境は複数．ターゲットのラベルは少数を選択だけに使う
- **後続**（部分的に解かれた．どちらも線形回帰）
  - Ha & Chen（本一覧の 1 番）の Corollary 1（◎）：候補のうち少なくとも 1 つが正しい型のとき，検証用のターゲットのラベル n_val ≥ c′ log(M/δ) で選び，選ぶ損は √(log(M/δ)/n_val) の大きさで済む
  - Kostin ほか（本一覧の 11 番）の Theorem 3.3（◎）：候補どうしのターゲットのリスクの差 Δ に対して n_Q ≲ 1/Δ なら，どの学習法も候補の集まりを使って Δ より良くはなれない
  - 分類の損失で同じ問いに答えた文献は見つからなかった
- **MER-DA との関係**：当たる設定は，K 本，コーパス間，ロジスティック損失である．MER-DA は設定の値を K 本ではなく仮のターゲット（ソースのコーパス）で選んでいるので，この問いの「ターゲットのラベルを使わずに選ぶ」側の答えを実測で持つ．データでは，E24（K 本で試す選び方は平均的な良さとともによく選べた），E39・E59 の追加（仮のターゲットで K ごとに選ぶ）とそのオラクルとの差が，選択に要る本数の実測になる

### 11. Kostin ほか：部分的な因果の知識による少数ラベルの DA を線形回帰の外に広げること，つまみの要らない手順，候補の数への依存

- **文献**：Julia Kostin, Kasra Jalaldoust, Elias Bareinboim, Samory Kpotufe, Fanny Yang, "How Useful is Causal Invariance for Domain Adaptation in Finite-Sample Settings?," arXiv:2606.12680v1（2026-06-10）．掲載先は無い（プレプリント）．水準 ◎（抄録，3 節の Theorem 3.3 の後，付録 B）
- **引用**
  - 付録 B："our theoretical results are, so far, limited to the linear regression setting ... An exciting direction for future work is thus to compute such confidence sets for other parametric classes"
  - 付録 B の 3："it is an interesting open question whether the same guarantee can be established for a procedure based on an end-to-end loss function. Further, Algorithm 1 requires as input two constants C1, C2 which have to be fixed a priori or estimated from (limited) target data. It remains an important open question, especially for practitioners, whether a hyperparameter-free adaptation method for supervised adaptation under partial causal knowledge could be developed"
  - 3 節："our results leave open the question whether the log |H′acc| dependence on the number of accepted candidates is necessary or can be further reduced"
- **未解決の問い**：ソースで学んだ候補の予測器を少数のターゲットのラベルでまとめる手順の保証は線形回帰だけである．分類への拡張，少数のターゲットのラベルからつまみを推さずに済む手順，候補の数の対数の依存が要るかどうかが未解決である
- **設定と仮定**：線形回帰．候補の予測器は特徴の部分集合ごとにソースで学ぶ．ターゲットのラベルは n_Q 本．雑音と関数の集合は有界．手順は「弱い信頼集合」で候補を絞ってからまとめる
- **後続**：2026 年 6 月の論文で，後続は検索で見つからなかった
- **MER-DA との関係**：当たる設定は，K 本，コーパス間，複数の候補（複数のソース・複数のモダリティの教師モデル）である．データでは，E26（全ソースの合併は最良の 1 つより 0.5〜1.3 点悪く，良い 2 つは最良の 1 つを上回る）と E71（アンカーと複数の教師モデルの解を候補として平均する．仮のターゲットでは本当の誤差も計算できる設計）が，候補のまとめ方と K の関係の実測になる

### 12. Tenzer ほか：ラベル無しで回帰器の誤差を推して合わせる方法を，少数のラベルがある場合に広げること，誤差の相関を許すこと

- **文献**：Yaniv Tenzer, Omer Dror, Boaz Nadler, Erhan Bilal, Yuval Kluger, "Crowdsourcing Regression: A Spectral Approach," AISTATS 2022（PMLR 151:5225–5242）．水準 ◎（結論の段落）
- **引用**："Our work raises several directions for future research. One is to extend our approach to a semi-supervised setting, in which there is also a limited amount of labeled data. It is also interesting to theoretically understand the benefits of labeled versus unlabeled data for ensemble learning. Another direction is to explore other relaxations of the uncorrelated error assumption."
- **未解決の問い**：複数の回帰器の予測だけから各回帰器の二乗誤差を推し，それで重みを付けて合わせる方法は，ラベルが 1 本も無い場合のものである．少数のラベルがあるときにどう組み込むか，ラベル付きとラベル無しのデータがアンサンブルにどれだけ役立つかの理論，回帰器どうしの誤差が相関する場合の扱い，が未解決である
- **設定と仮定**：回帰器の予測の誤差は互いに無相関．応答の 1 次と 2 次のモーメントは既知．ラベル無しの標本で予測の共分散を測り，スペクトルの方法で誤差を推す
- **後続**：検索で，半教師ありへの拡張は見つからなかった
- **MER-DA との関係**：当たる設定は，K 本，別のモダリティの教師モデル（複数の見方），アンカーつきの学び直しである．E72・E73 の机上の導出（食い違いから誤差を推す重みを K に合わせる）は，この問いと同じ設定にある（`docs/related-work.md` 1.18 節が Tenzer ほかを先例として記録済み）．データでは，E71（見方の食い違いから推した誤差と，仮のターゲットで計算した本当の誤差を比べる設計）が，ラベルの本数と推した重みの良さの関係の実測になる

### 13. Zwieback ほか と Gruber：collocation で決まらない誤差の共分散の部分と，誤差の相関の推定

- **文献**
  - S. Zwieback, K. Scipal, W. Dorigo, W. Wagner, "Structural and statistical properties of the collocation technique for error characterization," Nonlinear Processes in Geophysics 19:69–80, 2012．水準 ◎（抄録，1 節，5.3 節．7 節は PDF の文字が重なって読めなかった）
  - Alexander Gruber, "Estimation of error structures in remotely sensed soil moisture data sets," 博士論文，TU Wien, 2016．水準 ◎（1.2 節）
- **引用**
  - Zwieback ほか，抄録："It turns out that only parts of the covariance structure can be resolved by the collocation technique, thus emphasizing the necessity of expert knowledge for the correct validation of geophysical products."
  - Zwieback ほか，5.3 節："the relative merits and drawbacks of the different estimators of the calibration constants remain an open topic of research."
  - Gruber，1.2 節："the subsequent simultaneous use of data sets with correlated errors in any statistically rigorous data assimilation or merging scheme requires an accurate parameterization of their error correlation magnitude ... However, no method is currently available to provide such estimates. While the estimation of error cross-correlations remains an unresolved issue, ..."
- **未解決の問い**：3 つ以上のデータの対ごとの差から各データの誤差の分散を推すとき，誤差の共分散の一部はデータだけでは決まらない．誤差が相関するデータを合わせるには相関の大きさが要るが，その推し方が（Gruber の extended collocation 以前は）無かった．較正の定数の推し方の良し悪しも未解決である
- **設定と仮定**：各データは真の値の線形変換に誤差を足したもの．誤差は真の値と無相関で，基本形では互いにも無相関
- **後続**：Gruber の博士論文の 3 本目の論文（Gruber ほか 2016，JGR．本文は読んでいない，△）が extended collocation で一部の相関を推した．ただし条件は "Each member of the data set pairs with assumed non-zero error cross correlation must also be a member of at least one data set triplet with fully independent errors"（博士論文 1.2.3 節，◎）である．`docs/related-work.md` 1.18 節の Pierdicca ほか（quadruple collocation）と Tenzer ほか 2022 も 4 つ以上での相関の検出を扱う
- **MER-DA との関係**：当たる設定は，別のモダリティの教師モデル（複数の見方），アンカー，K 本である．MER-DA では見方がアンカーと教師モデルが作ったパラメータのベクトルであり，地球科学の観測の時系列とは対象が違う．データでは，E71 の AVT の世界（見方が 4 つ）で見方どうしの誤差の相関を記述で調べる手順（E71 の判定の分岐）が，相関がどれだけあり，どの部分が決まらないかの実測になる

### 14. Rosenfeld ほか（DARE）：少数のターゲットのデータで最後の線形層を学び直せば足りるという推測，ラベル無しの標本をテスト時にだけ使う適応

- **文献**：Elan Rosenfeld, Pradeep Ravikumar, Andrej Risteski, "Domain-Adjusted Regression or: ERM May Already Learn Features Sufficient for Out-of-Distribution Generalization," arXiv:2202.06856v2（2022-10）．掲載先は本文に無い．水準 ◎（2 節，3 節，付録 A）
- **引用**
  - 2 節："We conjecture that this phenomenon occurs more broadly ... This observation also implies that in settings where we have access to a small amount of data from the target distribution, simply retraining the last linear layer on this data will suffice for excellent performance. Two immediate questions are in which other settings this holds and if, instead of using more elaborate approaches such as complex regularization, the remaining gap could be closed using simpler methods."
  - 3 節："We believe JIT-UDA presents a promising direction for future theoretical research: it is much weaker—and arguably more realistic—to assume access to unlabeled samples only at test-time."
  - 付録 A（アンカー回帰との関係）："However, this still requires us to estimate the target mean, so it is unclear if or how this objective could be applied to the task of classification."
- **未解決の問い**：凍結した特徴の上の最後の線形層を少数のターゲットのデータで学び直せば足りるかは推測のままで，ターゲットの本数で誤差がどう減るかの式は無い．テスト時のラベル無しの標本で平均と共分散を合わせる適応の理論も今後の課題とされている
- **設定と仮定**：各ドメインの特徴は共通の不変な部分とドメインごとの平均・共分散のずれを持つ（仮定 5.1）．DARE はドメインごとに白色化してから線形の予測器を学ぶ．制約つきのロジスティック回帰の閉じた解はガウスの場合だけ証明する
- **後続**：検索で，最後の線形層の学び直しの少数ラベルでの誤差の理論は見つからなかった．経験的な研究（Kirichenko ほか 2022 の DFR など）は多いが，検索結果の要約によれば大量のターゲットのデータを前提にする（△）
- **MER-DA との関係**：当たる設定は，K 本，切片，コーパス間である．MER-DA の中心化（ラベル無しのターゲットの平均を引く）は JIT-UDA の平均の調整に当たる．データでは，E12・E25・E27（中心化だけで K=1 の最良の手になり，176 組中 9 組で損）と E28（切片が自由な仮説転移で揺れの分散が 1/(2K)）が，最後の線形層の学び直しの誤差を K の関数として測った実測になる

### 15. Xue・Payani・Yang・Mirzasoleiman（MixPro）：凍結した埋め込みの上の少数ショットの線形分類器の理論が二乗損失だけであること，少数のターゲットでのつまみの選び方

- **文献**：Yihao Xue, Ali Payani, Yu Yang, Baharan Mirzasoleiman, "Few-shot Adaptation to Distribution Shifts By Mixing Source and Target Embeddings," arXiv:2305.14521v3（2024-05）．掲載先は本文で確かめていない．水準 ◎（4〜6 節）
- **引用**
  - 6 節："Prior work that uses this approach does admit that selecting hyperparameters in a realistic manner remains an open problem that they have not addressed yet."（大きな検証用のターゲットのデータでつまみを選ぶ先行研究，Kirichenko ほか 2022 など，について）
  - Theorem 5.1 の仮定："Assuming that the noise is sufficiently small, σ = o(1), and l(., .) is the MSE loss"
- **未解決の問い**：少数のターゲットのデータだけでつまみを現実的に選ぶ方法が，最後の層の学び直しの先行研究では扱われていない（MixPro は 2 分割の交差検証で試した）．MixPro の理論は二乗損失だけである
- **設定と仮定**：凍結した事前学習の埋め込み．ソースの埋め込みと少数のターゲットの埋め込みを線形に混ぜた大きなデータで線形分類器を学ぶ．クラスあたり 2〜16 本
- **後続**：調べていない
- **MER-DA との関係**：当たる設定は，K 本，凍結した特徴の上の線形の分類器，コーパス間である．MER-DA は仮のターゲットでつまみを選ぶので，この問いに別の答え方をしている．データでは，E39・E40（5 つの値を仮のターゲットで選ぶ）と E59 の追加の，選んだ値とオラクルの値の差が実測になる

### 16. Yang ほか：ソフトな転移（パラメータどうしを L2 で結ぶ学び方）の理解，任意の共変量とモデルのずれのもとでの偏りの極限

- **文献**：Fan Yang, Hongyang R. Zhang, Sen Wu, Christopher Ré, Weijie J. Su, "Precise High-Dimensional Asymptotics for Quantifying Heterogeneous Transfers," JMLR 26, 2025．水準 ◎（JMLR 版の 1 節，Remark 7，6 節）．arXiv の初版 2010.11750v1 も取った
- **引用**
  - 1 節："Showing the asymptotic bias limit under arbitrary covariate and model shifts is an interesting open problem for future work (See Remarks 7 and 13 for further discussions)."
  - 6 節の図 6 の説明文："We leave a thorough understanding of this phenomenon, including soft transfer, for future work."
  - 6.4 節："It would be interesting to further study the regularization effect of soft transfer in large models, which would also require developing tools for analyzing over-parameterized models (and perhaps a separate data model). These are left for future work."
- **未解決の問い**：2 つの線形回帰の標本を合併したときの偏りと分散の正確な極限は，共分散が等しいか等方の場合しか得られていない．パラメータを L2 で結ぶソフトな転移（SPS）が負の転移を抑える仕組みの理解も今後の課題である．分類への拡張は arXiv 初版にだけ今後の課題として書かれている（`research_notes/TAC 論文 先行研究の精読と穴/g5_theory_pooling_multisource.md` の確認による）
- **設定と仮定**：線形回帰，切片無し，ガウスの共変量，比例の極限．ターゲットの標本数は次元より大きい
- **後続**：2025 年以降は調べていない．等方の場合のアンカーつきのリッジは Dar ほか 2024（本一覧の 17 番）が先に扱った
- **MER-DA との関係**：当たる設定は，アンカーつきの学び直し（ソフトな転移の一種），合併，コーパス間である．データでは，E25（仮説転移はふつうの合併を K=5 で 1.0〜2.7 点上回る）と E17（C を固定した地図）が，合併とソフトな転移の差の実測になる

### 17. Dar・LeJeune・Baraniuk：ソースのパラメータに寄せる L2 の転移の，混ぜた正則化・他の課題の関係の模型・最小最大の最適性

- **文献**：Yehuda Dar, Daniel LeJeune, Richard G. Baraniuk, "The Common Intuition to Transfer Learning Can Win or Lose: Case Studies for Linear Regression," SIAM Journal on Mathematics of Data Science 6:454–480, 2024, arXiv:2103.05621v4．水準 ◎（7 節）
- **引用**（7 節）："Future extensions may study other optimization formulations such as hybrid regularizers that merge the ideas of ridge regression and parameter transfer, and additional task relation models and their utilization in the transfer learning process. Moreover, future work may use other analysis tools and optimality definitions (e.g., minimax optimality) to further understand the performance of the transfer learning methods that we proposed in this paper."
- **未解決の問い**：ソースの推定値に L2 で寄せてターゲットで学ぶ線形回帰の誤差は，等方の特徴とある課題の関係の模型で得られた．0 への L2 とソースへの L2 を混ぜた正則化，別の課題の関係の模型，最小最大の意味での最適性が未解決である
- **設定と仮定**：線形回帰，過剰パラメータ化を含む．ソースの課題の解に寄せる罰則．等方の特徴と事前分布（6 節で一般の共分散）
- **後続**：調べていない
- **MER-DA との関係**：当たる設定は，アンカーつきの学び直し，K 本である．MER-DA は疑似ラベルでアンカーを動かしてから K 本で学び直す形（E66 の B）を持ち，「どの点に寄せるか」を変える版にあたる．データでは，E67〜E70（学び直しで中心の値が固有方向ごとに残る割合と得の式）が実測と導出の材料になる

### 18. Zanella ほか（TransCLIP）：教師モデル（テキスト側）の確率が信頼できないときに，クラスごとの重みで頑健にすること

- **文献**：Maxime Zanella, Benoît Gérin, Ismail Ben Ayed, "Boosting Vision-Language Models with Transduction," NeurIPS 2024, arXiv:2406.01837．水準 ◎（arXiv の HTML 版の 5 節．PDF の文字は取れず，要約器で文を抜き出した）
- **引用**（5 節）："TransCLIP's text-guided KL divergence term appears as a key factor in its success" ... "future works may consider making it more resilient (e.g., through adaptive class-wise weighting factor) when text prompts are less reliable."
- **未解決の問い**：ラベル無しの全標本に，学び直さない教師モデル（テキストのゼロショット予測）の確率への KL をかける方法で，教師モデルの確率が信頼できないクラスがあるときの扱いが無い
- **設定と仮定**：CLIP の凍結した特徴．転導的な推論（ラベル無しのテスト標本全体を使う）．クラスの混合比は均等と仮定する．少数ショットは 0〜16 本
- **後続**：調べていない
- **MER-DA との関係**：当たる設定は，ラベル無しの発話全体への疑似ラベル，K 本，多クラス，学び直さない教師モデルである．データでは，E59 の釣り合い直し（A の K=0 で +0.85）と，E60 の教師モデルの確率の感情ごとの正解率が，クラスごとに教師モデルの信頼度が違うかの実測になる

### 19. Berthelot ほか（AdaMatch）と Veilleux ほか：疑似ラベルのクラスの比率の仮定がずれたときの損

- **文献**
  - David Berthelot, Rebecca Roelofs, Kihyuk Sohn, Nicholas Carlini, Alex Kurakin, "AdaMatch: A Unified Approach to Semi-Supervised Learning and Domain Adaptation," ICLR 2022, arXiv:2106.04732．水準 ◎（3 節と 7 節）
  - Olivier Veilleux, Malik Boudiaf, Pablo Piantanida, Ismail Ben Ayed, "Realistic Evaluation of Transductive Few-Shot Learning," NeurIPS 2021, arXiv:2204.11181．水準 ◎（結論）
- **引用**
  - AdaMatch，3 節："when the target label distribution is unknown, we approximate it using the only available distribution – the source label distribution. A limitation of this approach is that the more the source label distribution differs from the target distribution, the more incorrect the approximation will be, which may cause the model performance to degrade."
  - AdaMatch，7 節："This leaves open an important question: which is more important for future progress, making use of unlabeled data more efficiently, or attempting to better model domain shifts?"
  - Veilleux ほか，結論："Future works could study the mixed effect of imbalance on both support and query sets."
- **未解決の問い**：疑似ラベルのクラスの比率をソースの比率か一様にそろえる操作は，ターゲットの本当の比率が違うと損をする．その損の大きさと，比率を推す方法は扱われていない．少数ショットで支持集合（ラベルあり）と問い合わせ集合（ラベル無し）の両方が偏るときの影響も未解決である
- **設定と仮定**：AdaMatch は深層網をゼロから学び，弱い拡張の予測を分布整合してから硬い疑似ラベルにする．Veilleux ほかは転導的な少数ショット（クラスは別），問い合わせ集合の比率をディリクレ分布で偏らせる
- **後続**：調べていない
- **MER-DA との関係**：当たる設定は，ラベル無しの発話全体への疑似ラベル，多クラス，コーパス間（コーパスごとに感情の比率が違う）である．MER-DA の釣り合い直しは目標を一様にする．データでは，E59 の釣り合い直しの得（A の K=0 で +0.85）を，`data/metadata/` から計算できるターゲットのコーパスごとの感情の比率と並べると，比率の仮定のずれと損の関係の実測になる

### 20. Jaritz ほか（xMUDA）：弱いモダリティからの負の転移と，まねの損失の重みの釣り合い

- **文献**
  - Maximilian Jaritz, Tuan-Hung Vu, Raoul de Charette, Émilie Wirbel, Patrick Pérez, "xMUDA: Cross-Modal Unsupervised Domain Adaptation for 3D Semantic Segmentation," CVPR 2020, arXiv:1911.12676．水準 ◎（3 節と 5.3 節）
  - 同著者ら, "Cross-modal Learning for Domain Adaptation in 3D Semantic Segmentation," arXiv:2101.07253v2（2022．TPAMI 版と推定，掲載誌は本文で確かめていない）．水準 ◎（3 節と結論）
- **引用**
  - TPAMI 版 3 節："The risk is that a negative transfer from the weak modality might degrade the performance of the strong one. This is why, in practice, one needs to down-weight the mimicry loss w.r.t. the segmentation loss to boost performance. However, this is a serious limitation, because down-weighting the mimicry loss also decreases its adaptation effect."
  - CVPR 版："we conjecture that best performance can be achieved if private information is also used."；"We conjecture that LxM is a beneficial auxiliary loss and can help to regularize training and prevent overfitting."
- **未解決の問い**：モダリティの間で予測をまねさせる損失は，弱いモダリティから強いモダリティへの負の転移を起こしうる．重みを下げると適応の効果も下がる．著者は頭を 2 つに分ける構造で回避したが，いつ負の転移が起きるかの説明はなく，推測（conjecture）が 2 つ残る
- **設定と仮定**：2D 画像と 3D 点群の 2 本の深層網を同時に学ぶ．ターゲットはラベル無し（UDA）か一部ラベルあり（半教師あり版）．損失はモダリティ間の KL
- **後続**：調べていない
- **MER-DA との関係**：当たる設定は，別のモダリティの教師モデル，コーパス間，K 本である．MER-DA の教師モデルは固定で，まねの向きは教師モデルから生徒モデルへの 1 方向である点が違う．データでは，E60 の M3（13 の生徒モデルのうち 4 つで K ≥ 10 に 0.1 超の損．最大は AT の音声の K=25 で −1.8）が，弱いモダリティからの負の転移の実測になる

### 21. Kumar ほか（LP-FT）：明示的な正則化の効果

- **文献**：Ananya Kumar, Aditi Raghunathan, Robbie Jones, Tengyu Ma, Percy Liang, "Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution," ICLR 2022, arXiv:2202.10054．水準 ◎（6 節，脚注 7）
- **引用**
  - 6 節："There are several open questions and extensions such as dealing with non-linear activations, different layerwise learning rates, and the effect of explicit regularization."
  - 脚注 7："Intuitively this cancelation is very "unlikely" to happen, and we hope future work can capture this intuition."
- **未解決の問い**：線形の頭を先に学んでから全体を微調整すると分布外で良いことを 2 層の線形網で示したが，学習済みの重みへの明示的な正則化（L2-SP など）を入れたときの効果の理論が無い
- **設定と仮定**：2 層の線形網，二乗損失，雑音なし．ID と OOD で真の線形の頭は共通．切片は無い
- **後続**：調べていない
- **MER-DA との関係**：当たる設定は，アンカーつきの学び直し（明示的な正則化），凍結した特徴の上の線形の頭である．データでは，E63（LoRA の微調整では凍結の E39 を 7.3 下回り，アンカーへの罰則は +0.4）が，特徴を動かす場合と動かさない場合の差の実測になる

### 22. Kini ほか：有限の L2 罰則のもとでの切片の補正の理論，ロジスティック損失での勾配流

- **文献**：Ganesh Ramachandra Kini, Orestis Paraskevas, Samet Oymak, Christos Thrampoulidis, "Label-Imbalanced and Group-Sensitive Classification under Overparameterization," NeurIPS 2021, arXiv:2103.01550v3．水準 ◎（5.1 節，6 節，付録 D.3）
- **引用**
  - 6 節："The current theory is limited to settings with fixed features."
  - 5.1 節："we treat it as our validation set and use it to tune our hyperparameters. More sophisticated tuning strategies (perhaps using bi-level optimization) are deferred to future work."
  - 付録 D.3："For simplicity, we consider a VS-type adjusted exponential loss ℓ(t)=e^{−t}, rather than logistic loss ... one would expect that insights also apply to logistic loss."
- **未解決の問い**：クラスの不均衡のもとで，加法的な補正（切片の調整）が効くかどうかの理論は正則化が 0 に向かう極限のもので，有限の罰則のもとでの理論が無い．勾配流の結果は指数損失で，ロジスティック損失は予想にとどまる．つまみをテストの集合で選んでおり，現実的な選び方は今後の課題である
- **設定と仮定**：固定した特徴，過剰パラメータ化，比例の極限．切片の無い模型（Theorem 1・3・5）
- **後続**：調べていない
- **MER-DA との関係**：当たる設定は，切片（MER-DA は切片に別の罰則 C_b を持つ），多クラス，ロジスティック損失である．データでは，E28（切片の縛りは K=1 でだけ効き +0.6〜+2.2 点）が，有限の罰則で切片の補正がいつ効くかの実測になる

### 23. Tiomoko・Tiomoko Ali・Couillet：課題ごとに次元とクラスの数が違う多課題の線形分類器

- **文献**：Malik Tiomoko, Hafiz Tiomoko Ali, Romain Couillet, "Deciphering and Optimizing Multi-Task Learning: a Random Matrix Approach," ICLR 2021．読んだのは arXiv:2009.01591v1（題は "Large Dimensional Analysis and Improvement of Multi Task Learning"．ICLR 版とは題が違う）．水準 ◎（2 節．引用の文は arXiv 版の文）
- **引用**："the linear common-hyperplane condition Wi = W0 + Vi, imposes by definition that all Vi's be of the same size Rp×m: this severely constrains (i) the data in each task to be of the same dimension p and (ii) the number of classes per task to be the same (m). Further linear or even non-linear relaxation schemes for Wi of the type Wi = Vi + fi(W0) for some operator fi could be envisioned to relax this constraint. This however goes beyond the scope of the present article"
- **未解決の問い**：共通の超平面 W0 に課題ごとの差を足す多課題の学び方の高次元の理論は，課題どうしで特徴の次元とクラスの数が同じ場合に限られる．次元が違う課題をつなぐ形は範囲外とされている
- **設定と仮定**：ガウスの混合，LS-SVM，課題ごとに中心化して課題ごとの切片を持つ，比例の極限
- **後続**：調べていない
- **MER-DA との関係**：当たる設定は，別のモダリティの教師モデル（教師モデルの特徴と生徒モデルの特徴の次元が違う），コーパス間，多クラスである．MER-DA は次元の違うモダリティを，パラメータではなく予測確率（疑似ラベル）でつないでいる．データでは，E60 の教師モデルの融合の特徴と生徒モデルの単一のモダリティの特徴の組が，次元の違う課題をつなぐ場合の実測になる

### 24. Zhang・Fan・Luo・Hu：転移の線形判別で，関係の弱いソースを自動で外すこと

- **文献**：Yonghan Zhang, Yimeng Fan, Wenya Luo, Jiang Hu, "Transfer Learning for Linear Discriminant Analysis with a Shared Classification Signal," arXiv:2607.06936v1（2026-07-08）．水準 ◎（5 節）
- **引用**（5 節）："There are several possible directions for future work. First, the present model focuses on linear discriminant rules, while nonlinear classification methods may be more suitable for some complex data. Second, it would be interesting to study how to select useful source domains automatically when some source domains are weakly related to the target domain. Finally, future work may consider more general types of domain heterogeneity beyond the spiked covariance models"
- **未解決の問い**：ソースのうちターゲットと関係の弱いものを自動で選び外す方法が無い．線形判別の外と，スパイクのある共分散の模型の外のずれも未解決である
- **設定と仮定**：線形判別（LDA），共通の判別の信号，スパイクのある共分散のずれ
- **後続**：2026 年 7 月の論文で，調べていない
- **MER-DA との関係**：当たる設定は，コーパス間，多クラスである．データでは，E24（式でソースを選ぶと既存の選び方に勝たない）と E55（ソースの発話ごとの信頼度で下位を外すと K=1 で A +1.9，V +3.9 ポイント）が，ソースを選ぶ・外す操作の実測になる

### 25. Kuzborskij & Orabona：多数のソースの仮説から少数のラベルで関係のあるものを選ぶこと

- **文献**：Ilja Kuzborskij, Francesco Orabona, "Fast Rates by Transferring from Auxiliary Hypotheses," Machine Learning 106:171–195, 2017, arXiv:1412.1619v3．水準 ◎（結論）
- **引用**（結論）："Our conclusions suggest the key importance of a source hypothesis selection procedure. Indeed, when an algorithm is provided with enormous pool of source hypotheses, how to select relevant ones on the basis of only few labeled examples? ... Thus, it remains unclear if having few good sources from hundreds is a reasonable assumption."
- **未解決の問い**：多数のソースの仮説が与えられたとき，少数のラベルだけで関係のある仮説を選ぶ方法と，その前提（数百の中に少数の良いソースがある）が妥当かが未解決である
- **設定と仮定**：正則化つきの経験リスク最小化，滑らかな非負の損失（ロジスティック損失を含む），強凸の正則化．ソースの仮説の組み合わせに寄せる
- **後続**：調べていない
- **MER-DA との関係**：当たる設定は，K 本，アンカーつきの学び直し，ロジスティック損失，コーパス間である．データでは，E24・E26（良い 2 つのソースはテストの値で選ぶと最良の 1 つを上回るが，K 本で試す選び方との差）が，少数のラベルでソースを選ぶ難しさの実測になる

### 26. Hanneke・Kpotufe・Mahdaviyeh：転移のもとでのモデルの選択で，適応的な率と神託の率の差

- **文献**：Steve Hanneke, Samory Kpotufe, Yasaman Mahdaviyeh, "Limits of Model Selection under Transfer Learning," COLT 2023（謝辞による）, arXiv:2305.00152v4．水準 ◎（Remark 1 と結論）
- **引用**
  - Remark 1："It remains unclear from our analysis whether and when we could achieve an adaptive model-transfer rate in terms of βQ,i*Q rather than βQ as defined above."
  - 結論："the gap between optimal adaptive rates and oracle minimax rates can be arbitrary ... However this leaves open the possibility of smaller or more controlled gaps under, e.g., further structural assumptions on the model hierarchy."
- **未解決の問い**：ソースとターゲットの距離を知らずに仮説の集合の大きさを選ぶと，距離を知る場合の率との差は任意に大きくなりうる．模型の階層に構造を仮定すれば差が小さくなるかは未解決である
- **設定と仮定**：分類，入れ子の仮説の集合，ソースからターゲットへの転移の指数，雑音の条件（Bernstein 型）
- **後続**：調べていない
- **MER-DA との関係**：当たる設定は，K 本，コーパス間である．MER-DA の正則化の強さ C の選び方はモデルの選択の一種である．データでは，E17（合併の C をソースのみの C に固定した地図）と交差検証の地図の差，E39 の 5 つの値の選択が，選び方による差の実測になる

### 27. Ben-David ほか：データに依存するずれの上界，複数のソースの凸結合を上界の最小化で選ぶ算法

- **文献**：Shai Ben-David, John Blitzer, Koby Crammer, Alex Kulesza, Fernando Pereira, Jennifer Wortman Vaughan, "A theory of learning from different domains," Machine Learning 79:151–175, 2010．水準 ◎（9 節．MER-DA の根の PDF）
- **引用**（9 節）："There are two interesting open problems that deserve future exploration. First, our bounds on the divergence between source and target distribution are in terms of VC dimension. We do not yet know whether our divergence measure admits tighter data-dependent bounds ... or if there are other, more appropriate divergence measures which do. Second, it would be interesting to investigate algorithms that choose a convex combination of multiple sources to minimize the bound in Theorem 5 as possible approaches to adaptation from multiple sources."
- **未解決の問い**：ずれの量の上界を VC 次元ではなくデータに依存する形で書けるか，複数のソースの混合比を上界の最小化で選ぶ算法，の 2 つ
- **設定と仮定**：2 値分類，0-1 損失，H Δ H ダイバージェンス，ソースとターゲットの経験誤差の凸結合の最小化
- **後続**（部分的に解かれた）：データに依存するずれの量として discrepancy（Mansour, Mohri, Rostamizadeh 2009．本文は読んでいない），上界の最小化で複数のソースの重みを決める DARN（Wen, Greiner, Schuurmans, ICML 2020．`research_notes/MER DA 未定式化理論の実現可能性/da_theory.md` で要約器による本文の確認，○）．ターゲットのラベルを使う版に Mansour ほか 2021（AISTATS），Konstantinov & Lampert 2019（ICML）がある
- **MER-DA との関係**：当たる設定は，K 本，コーパス間である．データでは，E1・E18（距離は「ソースのみ − ターゲットのみ」を下げ，「合併 − ソースのみ」を上げ，得では打ち消し合う）と E26 が，上界の量と実際の得の関係の実測になる

### 28. Huang ほか と Du ほか（UMT）：多モダリティが単一モダリティより良いことの一般の証明，単一モダリティの教師モデルで融合の生徒モデルを学ぶことの推測

- **文献**
  - Yu Huang, Chenzhuang Du, Zihui Xue, Xuanyao Chen, Hang Zhao, Longbo Huang, "What Makes Multi-modal Learning Better than Single (Provably)," NeurIPS 2021, arXiv:2106.04538v2．水準 ◎（脚注と 6・7 節）
  - Chenzhuang Du ほか, "Improving Multi-Modal Learning with Uni-Modal Teachers," arXiv:2106.11059（2021．査読つきの掲載先は見つかっていない）．水準 ◎（3 節の Conjecture 1 と結論）
- **引用**
  - Huang ほか，脚注："Proving that γS(M, N) ≤ 0 holds in general is open and will be an interesting future work."
  - Huang ほか，6 節："Understanding why multi-modal fails in practice is a very interesting direction and worth further investigation in future research."
  - Du ほか，Conjecture 1："By carefully picking the λ parameter, the encoders trained by the UMT method can at least learn all the features learned by the corresponding teacher encoders. ... We leave the proof of this conjecture as a future direction."
- **未解決の問い**：モダリティを増やすと潜在表現の質が下がらないこと（γ_S ≤ 0）の一般の証明が無い．実際の多モダリティの学習がなぜ失敗するかも未解決である．単一モダリティの教師モデルの特徴を蒸留した融合の網が，教師モデルの特徴を全部学べるかは推測のままである
- **設定と仮定**：Huang ほかは共通の潜在空間へのエンコーダと潜在空間から出力への写像，汎化の側だけを扱い最適化は理想とする．Du ほかは深層網の特徴の蒸留
- **後続**：Huang ほかの後続は検索で見つからなかった．Du ほかは調べていない
- **MER-DA との関係**：当たる設定は，別のモダリティの教師モデル，凍結した特徴の上の線形の頭である．データでは，E60-2（モダリティごとの生徒モデルを組み直して融合の生徒モデルに蒸留する．融合の生徒モデルは AV +1.2／+0.6，VT +2.6／+0.9，AVT +3.0／+0.8．K=0／5）と E62（第 2 のエンコーダでの再現）が，融合が単一モダリティを上回る条件の実測になる

### 29. Kumar・Ma・Liang と Wei ほか：自己訓練の理論で，最適化の解析，指数的に増える誤差の上界，整合に基づく DA への拡張

- **文献**
  - Ananya Kumar, Tengyu Ma, Percy Liang, "Understanding Self-Training for Gradual Domain Adaptation," ICML 2020, arXiv:2002.11361．水準 ◎
  - Colin Wei, Kendrick Shen, Yining Chen, Tengyu Ma, "Theoretical Analysis of Self-Training with Deep Networks on Unlabeled Data," ICLR 2021, arXiv:2010.03622v5．水準 ◎（6 節）
- **引用**
  - Kumar ほか（式 (19) の直後）："we do not show that self-training actually converges to the constrained minimum of U in Equation (19) and prior work only shows that self-training descends on U —we leave this optimization analysis to future work."
  - Kumar ほか（Example 3.4 の直後）："This suggests that if we want sub-exponential bounds we either need to make additional assumptions on the data distributions, or devise alternative algorithms to achieve better bounds (which we believe is unlikely)."
  - Wei ほか，6 節："an interesting question for future work is to extend our assumptions to analyze domain adaptation algorithms based on aligning the source and target (Hoffman et al., 2018)."
- **未解決の問い**：正則化つきの自己訓練が制約つきの最小化の解に収束するかの最適化の解析が無い．段階的な自己訓練の誤差の上界は段の数に対して指数的で，これを避ける条件が未解決である．拡張の仮定による自己訓練の解析を，ソースとターゲットを整合させる DA の算法に広げることも未解決である
- **設定と仮定**：Kumar ほかは線形の分類器，ずれは段階的，ラベルの比率は変わらない，マージン損失と正則化．Wei ほかは「拡張」の仮定（近い標本は同じラベル），入力の一貫性の正則化，深層網
- **後続**（部分的に解かれた）：Wang, Li, Zhao, "Understanding Gradual Domain Adaptation: Improved Analysis, Optimal Path and Beyond," ICML 2022, arXiv:2204.08200（◎）は "our bound only depends on T linearly" と，段の数への依存を線形にした（Kumar ほかの「別の仮定か算法が要る」に，追加の仮定で答えた形）．ただし同論文も "It also remains an open question on how to efficiently construct the optimal intermediate domains when only unlabeled data are available from the target domain" と書く．Cai, Gao, Lee, Lei, "A Theory of Label Propagation for Subpopulation Shift," ICML 2021, arXiv:2102.11203（○，検索結果の要約）が Wei ほかの拡張の仮定を部分集団のずれの DA に広げた
- **MER-DA との関係**：当たる設定は，ラベル無しの発話全体への疑似ラベル，コーパス間，アンカーつきの学び直しである．MER-DA の疑似ラベルは自分の予測ではなく別のモダリティの教師モデルのもので，反復もしない点が違う．データでは，E61（自己訓練，硬い疑似ラベル，エントロピー最小化を同じ生徒モデル・同じ K で比べた．エントロピー最小化の固定版は E62 で 4 生徒とも損 −2.6〜−16.2）が，自己訓練の型の手法が損をする条件の実測になる

### 30. Hernandez ほか：転移の尺度則を教師ありの設定とゼロショットに広げること，限られたデータでの最適な混合比の予測

- **文献**：Danny Hernandez, Jared Kaplan, Tom Henighan, Sam McCandlish, "Scaling Laws for Transfer," arXiv:2102.01293v1（2021）．水準 ◎（5 節と 6.7 節）
- **引用**
  - 5 節 7)："Equations 3.1 and 3.2 don't handle zero-shot case unless we use an approximation (i.e. fine-tuning on 1 character) for the zero-shot case."
  - 6.7 節 4："Transfer scaling laws in supervised and reinforcement learning settings."
  - 6.7 節 6："A method to cheaply predict the ideal pre-training ratio for a pair of datasets A and B to maximize performance on a target distribution C, for which we have limited data."
- **未解決の問い**：事前学習が「微調整のデータ何本ぶん」かを表す転移の尺度則は，教師なしの言語モデルでしか測られていない．教師ありの設定，ゼロショット（ターゲットのデータが 0 本）の扱い，限られたターゲットのデータで最適な混合比を予測する方法が未解決である
- **設定と仮定**：Transformer の言語モデル，自然言語で事前学習してコードで微調整，損失で測る（正解率ではない）
- **後続**（部分的）：Jain・Montanari・Sasoglu 2024（本一覧の 9 番）が，ラベル付きの本物と代わりのデータの混合で尺度則と最適な重みを扱った
- **MER-DA との関係**：当たる設定は，K 本（K=0 を含む），コーパス間である．MER-DA の「ソースの価値・教師モデルの価値をラベルの本数で読む」（E32・E65）は，教師ありの分類で K=0 を含めて同じ種類の量を測った実測である

---

## 3. 問いの型ごとの項目番号

同じ型の問いを複数の著者が挙げている．型の名前は私が付けた．

| 問いの型 | 項目番号 |
|---|---|
| 二乗損失の結果をロジスティック損失・交差エントロピー・分類に広げる | 1，2，5，9，11，15，16（arXiv 初版），22 |
| 疑似ラベル・ソース・代わりのデータとラベルのある標本の重みを決める | 2，3，4，8，9，18，20 |
| 少数のターゲットのラベル（か，ラベル無し）でつまみ・手法・ソースを選ぶ | 8，10，11，15，22，24，25，26，27 |
| 複数の見方の誤差を推して合わせる．誤差の相関・決まらない部分 | 12，13 |
| 0 でない中心（アンカー）への正則化のもとでの理論 | 6，16，17，21 |
| 教師モデルの誤りの種類（偏り・相関）と得 | 5，7 |
| ラベル無しの平均・比率の調整（中心化・釣り合い直し） | 14，19 |
| 多モダリティの学習の理論 | 4，20，23，28 |

---

## 4. 要約（結論として返すもの）

項目は 30 個である（2 本以上の文献をまとめた項目が 6 つあり，原文で引用を確かめた文献は後続を含めて 40 本）．後続の研究で部分的に解かれていたのは 5・6・10・13・27・29・30 番で，どれも MER-DA の設定（多クラスのロジスティック回帰，K 本の真のラベルとの併用，アンカー，別のモダリティの教師モデル，コーパス間）のすべてを満たす形では解かれていなかった．調べた範囲では，30 項目のどれも完全には解かれていない．

上位 5 項目：

1. Ha & Chen（arXiv 2507.14661v2）：少数のターゲットのラベルで教師なしの DA の解から低い次元だけを直す有限標本の理論を，ロジスティック損失・交差エントロピーに広げることは "left for future work"（2.1 節）．後続は見つからない
2. Li ほか（ICML 2026）：ずれのもとで教師モデルの予測と真のラベルを混ぜる最適な重みの式は線形回帰だけで，非線形への拡張は今後の課題（6 節）．分類は扱わず，重みを推す手続きも無い
3. Lopez-Paz ほか（ICLR 2016）：別のモダリティを特権情報とする一般化蒸留で，ラベルのある標本と教師モデルの軟らかいラベルだけの標本の "proper weighting" が無い（5 節）．分類での理論は見つからない
4. Hatano ほか（MM-CDFSL，ECCV 2024）：ドメインをまたぐ少数ショットで，モダリティごとの蒸留の重みをターゲットに合わせて変えることを今後の課題とする（結論）．後続は見つからない
5. Charikar ほか（NeurIPS 2024）：弱から強への一般化の得を分類で量にすること，誤りの種類（でたらめか相関か）で得が変わるかを未解決とする（結論，付録 E）．Mulgund & Pabbaraju（ICML 2025）が Bregman ダイバージェンスに広げたが，保証は KL の向きが通常の交差エントロピーと逆で，真のラベル・アンカー・ずれは扱わない
