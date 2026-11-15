# 検証：一般の機械学習と医療の予測モデルの先行研究の報告（general.md，general_round2.md）（2026-11-12）

要約
1. 原文で確かめた 37 件の主張（1 節 30 件，2 節 7 件）のうち，数値の食い違いは 0 件だった．記述の食い違いは 3 件ある．(1) Johnson 2018 の「他病院のモデル」は 45 病院ではなく，46 病院を病院単位で半分ずつ学習用と評価用に分けた学習用の側で学んでいる．(2) RLSbench の「2〜10 pt」は，学習中の再標本化（RS）と事後の重み直し（RW）を組んだ手順全体の得であり，切片の補正だけの得ではない．(3) Debray 2013 の事例 2 を「どの切片の選び方でも E/O は 1 に近づかなかった」と書いたのは原文より強い（原文は "generally"）．加えて，報告は事例 3（効果のばらつきが弱い場合は切片で直る）を落としている．
2. 条件の書き落としや解釈の注意は 8 件ある（1 節と 2 節の表の「一致（注意）」）．どれも報告の結論を変えない．
3. 前回読めなかった 4 本のうち，Vergouwe 2017 は Utrecht 大学の置き場の出版社版 PDF で本文を読めた．報告の主張（前立腺がんの例で更新が要った理由は切片の違いだけ）は原文と一致するので，水準を ○ から ◎ に上げてよい．Janssen 2008 は PubMed の抄録だけ，Steyerberg 2004 は抄録だけ（無料の本文は見つからない），Kull & Flach 2015 は Bristol 大学の置き場に PDF があるが自動取得が 403 で拒否された．この 3 本は ○ のままにする．
4. 報告の 3 つの判定は原文に照らして妥当である．「完全一致は無い」について，検証の途中で完全一致にあたる研究は見つからなかった（反例を探す検索 5 回の範囲）．見落としの部分一致として Godau ほか 2023（MICCAI．クラスの割合のずれのもとで，ずれの後の割合を使って切片つきの当て直しをする）が 1 本ある．「最も近いのは Johnson 2018」は，4.2 節の 3 点の注意を付ければ妥当である．「切片と向きの分け方は医療で既知」は，Vergouwe 2017 の本文と Janssen 2008 の抄録でさらに強く裏づけられた．

---

## 0. 確かめ方

- arXiv，ACL Anthology，PMLR，著者のサイト，大学の置き場の PDF を取得し，pdftotext で文字にして該当箇所を直接読んだ（Johnson 2018，Debray 2013，DART，AdaMatch，RLSbench，GLA，Albanie 2018，Chen 2020，Joshi 2012，Menon 2015，Nigam & Ghani 2000，Vergouwe 2017，Calibrate Before Use，DISDE，LaBonte 2023，Elkan & Noto 2008，Godau 2023）．
- PMC の論文は Europe PMC の全文 XML（Riley 2016，Guo 2022，Kappen 2012，Martin 2017）か，PMC のページの WebFetch（Strobl 2015，Davis 2019．要約器を通した引用）で読んだ．
- PubMed の抄録（Steyerberg 2004，Janssen 2008）は NCBI の efetch で取得した．
- 判定の語：「一致」は原文の数値と記述が報告と合う．「一致（注意）」は数値は合うが条件か解釈に補足が要る．「食い違い」は原文と報告の記述が違う．「確かめられず」は本文を読めなかった．

---

## 1. 必ず確かめる主張（依頼の 1 の一覧）

| 主張 | 報告の値 | 原文の値 | 判定 | 引用 | URL |
|---|---|---|---|---|---|
| Debray 2013：書誌とデータ | Statistics in Medicine 32: 3158–3180．12 研究，10,014 人，うち 1,897 人が陽性 | 12 研究，10,014 人，1,897 人が深部静脈血栓症．読んだ PDF は校正刷り（"PROOF" の透かし）で頁番号は出版版と違う | 一致 | "the studies contained a total of 10,014 patients of which 1897 ..." | https://doi.org/10.1002/sim.5732 ， https://www.covprecise.org/wp-content/uploads/2020/06/2013-Debray-et-al-A-framework-for-developing-implementing-and-evaluating-clinical-prediction-models-in-an-individual-participant-data-meta-analysis-endnote.pdf |
| Debray 2013：事例 1 の効果のばらつき | τ は 0.03〜0.05 | τ_sex = 0.03，τ_surg = 0.05 | 一致 | "Because the between-study variability ... appears negligible, we considered that assuming homogeneity was sensible" | 同上 |
| Debray 2013：事例 1 の平均の切片 | 12 研究のうち 4 つで E/O > 1.2，3 つで E/O ≤ 0.8 | 同じ | 一致 | "(E/O > 1.2 in 4 of the 12 validation samples) or underestimate (E/O ≤ 0.8 in 3 of the 12 ...)" | 同上 |
| Debray 2013：研究 1 で 1.42 → 1.03 | 発生率が似た研究の切片で 1.42 から 1.03 に直った | 同じ．表 II では，新しい集団の発生率から切片を推定した場合も 1.03 | 一致 | "when study 1 was used as the validation data, the E/O statistic was 1.42 when using the weighted average intercept from random effects model (3) but was 1.03 when using the intercept estimate for the study with the most similar incidence" | 同上 |
| Debray 2013：事例 2 は直らない | 効果が研究間で大きく違う（τ 0.09〜0.41）と，どの切片の選び方でも E/O は 1 に近づかなかった | τ は surg 0.09，calfdif3 0.16，malign 0.35，ddimdich 0.41 で範囲は合う．原文は「全体として 1 に近くない」と書き，表 III には 1 に近い値（1.03 など）も一部ある．新しい集団の発生率から切片を推定した場合の E/O は 10 研究すべてで 1.13〜1.68 だった．事例 2 は 12 研究のうち 10 研究だけを使った | 食い違い（言い過ぎ） | "all strategies for choosing intercepts perform poorly, as they generally give E/O ratios that are not close to 1" | 同上 |
| Debray 2013：AUC | 事例 1 は 0.55〜0.65，事例 2 は 0.73〜0.92 | 同じ | 一致 | "Whereas models from case study 1 achieved an AUC between 0.55 and 0.65 ..., the inclusion of malign, surg, calfdif3, and ddimdich increased this statistic to values between 0.73 and 0.92." | 同上 |
| Debray 2013：事例 3 | 報告に記載なし | 事例 3（効果のばらつきが弱い〜中程度．sex，surg，calfdif3）では，発生率から切片を決める方法で全研究の E/O が 1 に近く，較正曲線も 12 研究のうち少なくとも 9 研究で合った | 報告の書き落とし | "this approach resulted into E/O ratios close to 1 in all validation studies" | 同上 |
| Debray 2013：予測因子の平均と発生率から切片を計算する（式 4・5） | 新しい施設の予測因子の平均（ラベル無しで得られる）と発生率から切片を決める．③で切片側を直す操作の先例 | 式 4 は α = ln(prev/(1−prev))，式 5 は予測因子を新しい集団の平均で中心化する．発生率は文献や専門家から得てもよいと書くが，IECV の実験では新しい集団の個人データ（ラベル）から発生率を求めた | 一致（注意）．「③の先例」とするのは言い過ぎで，実験の中の操作はターゲットのラベルの割合を使っている．報告も「発生率は別に知る必要があり，ラベル無しだけで決まる操作ではない」と 1.2 節で断っているが，要約 4 の書き方は強い | "Estimates of these proportions may be obtainable from (a systematic review of) the medical literature or experts" ／ "the intercept was estimated from the outcome proportion in the IPD for the new population by mean centering" | 同上 |
| Johnson 2018：掲載先 | NeurIPS 2018 の ML4H ワークショップ | 同じ | 一致 | "Machine Learning for Health (ML4H) Workshop at NeurIPS 2018." | https://arxiv.org/abs/1812.02275 |
| Johnson 2018：データとモデル | eICU，46 病院 50,106 件，L2 罰則つきロジスティック回帰，82 特徴 | 同じ | 一致 | "The final cohort contained 50,106 ICU stays across 46 hospitals." ／ "A total of 82 features were extracted." | 同上 |
| Johnson 2018：他病院のモデルの学習データ | 第 1 回 3.1 節「ほかのドメイン（45 病院）のラベル付きデータ」 | 46 病院を病院単位で学習用と評価用に半分ずつ分け，学習用の病院だけで学んだ．評価の病院 73 は評価用の側にある | 食い違い | "We split our cohort evenly into training and test sets hospital-wise." | 同上 |
| Johnson 2018：実験の手順 | 病院 73 で 500 人を評価に残し，200 人刻み，50 回繰り返す | 同じ | 一致 | "we select a single hospital and set aside 500 patients ... sets of 200 patients ... we repeat this procedure 50 times" | 同上 |
| Johnson 2018：当て直し | 1 変数のロジスティック回帰（ずらしと伸縮） | 同じ | 一致 | "Recalibration involves training a univariable logistic regression with hospital mortality as the outcome, effectively shifting and scaling model predictions." | 同上 |
| Johnson 2018：AUROC | 他病院のモデル 0.849 で一定，局所のモデル 0.614 → 2,400 人で 0.796 | 同じ | 一致 | "The AUROC of the transferred model is constant at 0.849, which is expected as the recalibration procedure is a monotonic transform ... The AUROC of the local model is initially poor (0.614), and after 2,400 patients has significantly improved to 0.796, but is still below the transferred model's AUROC." | 同上 |
| Johnson 2018：SMR | 局所のモデルの SMR は「1 を大きく超えた値」から 1 に近づく | 原文は「どちらのモデルの SMR も妥当」で，局所のモデルは最初 SMR > 1 から 1 に収束すると書く．「大きく」は本文に無い．なお原文の SMR は観測 ÷ 予測なので，SMR > 1 は予測が少なすぎることを指すが，原文は "overpredicting" と書いており原文の側に矛盾がある | 一致（注意） | "SMRs for both models are reasonable, with the local model initially overpredicting mortality (SMR > 1.0), and eventually converging to the ideal value (1.0)." | 同上 |
| Johnson 2018：付録 C | 病院ごとの平均と標準偏差で標準化しても目立った差なし | 同じ．ただし付録 C は病院 73 の当て直しの実験ではなく，評価用の全病院での 1 つ目の実験について行った | 一致 | "We found no noticably difference between the two methods." | 同上 |
| Strobl 2015 | 5 コホート，1994〜2010 年，年ごとの局所 73〜1,106 件．当て直しは較正を直し判別をほとんど変えない．改訂と作り直しも AUC をほとんど上げない | 同じ | 一致 | "recalibration often improved calibration but had little impact on discrimination" ／ "the AUC increased with size of the training set for most cohorts, but the gain in AUC was small" | https://pmc.ncbi.nlm.nih.gov/articles/PMC4532612/ |
| Riley 2016：乳がんの例 | 国ごとに基準ハザードを推定し直すと較正の傾きの I² が 98% → 35%，傾きの予測区間 0.93〜1.08 | 同じ．当て直しの前の傾きの予測区間は 0.41〜1.58 | 一致 | "I² fell from 98% without recalibration to 35% with recalibration, and the updated 95% prediction interval for the calibration slope was 0.93 to 1.08" | https://pmc.ncbi.nlm.nih.gov/articles/PMC4916924/ |
| Riley 2016：QRISK2 | C 統計量の平均 0.83，I² 80.9%，予測区間 0.76〜0.88，364 診療所 | 同じ | 一致 | "The summary (average) C statistic was 0.83 ... (I² = 80.9%) ... (0.76 to 0.88)" | 同上 |
| Guo 2022 | ラベル無しの適応（100〜1,500 件）で件数による差なし．AUROC の差 −0.003〜0.050 | 件数による差なしは一致．−0.003〜0.050 は領域汎化（DG）とラベル無しの適応（UDA）を合わせた範囲で，UDA だけの範囲ではない | 一致（注意） | "When compared with ERM[08–16], DG and UDA experiments failed to produce more robust models (range of AUROC difference, −0.003 to 0.050)." ／ "we did not observe meaningful differences across the number of unlabelled samples evaluated" | https://pmc.ncbi.nlm.nih.gov/articles/PMC8854561/ |
| DART：著者と版 | Minguk Jang，Hye Won Chung（KAIST），v2 は 2025-02-05 | 同じ | 一致 | "Minguk Jang, Hye Won Chung ... KAIST" ／ "arXiv:2411.15204v2 [cs.LG] 5 Feb 2025" | https://arxiv.org/abs/2411.15204 |
| DART：表 3 | ρ=1／10／100 で BNAdapt 85.2／79.0／67.0，b だけ 85.4／83.6／80.7，W だけ 85.0／84.0／83.0，両方 85.2／84.7／85.1．W が b より 0.4 pt・2.3 pt よい | 同じ．報告の計算（ρ=100 で b だけが両方の得の 13.7/18.1 = 76%，ρ=10 で 4.6/5.7 = 81%）も検算で合う | 一致 | "using the square matrix W for prediction refinement outperforms using the bias vector b ... showing improvements of 0.4% and 2.3% on CIFAR-10C-LT with ρ = 10 and 100" | 同上 |
| AdaMatch | 分布の合わせ込みを外すと 26.3 → 17.7（6 つの適応の組の平均） | 数値は一致．表 4 は DomainNet（64×64）の教師なしの適応（UDA）6 組の平均．分布の合わせ込みは学習中に疑似ラベルの確率へ「ソースの平均 ÷ ターゲットの平均」をかける操作で，学習の後に出力の切片を直す操作ではない．報告の「(a) に当たる操作の得」は近似の言い方である | 一致（注意） | "w/o distribution alignment 17.7" ／ "we rectify the target unlabeled pseudo-labels by multiplying them by the ratio of the expected value of the weakly augmented source labels ... to the expected value of the target labels" | https://arxiv.org/abs/2106.04732 |
| RLSbench | クラスの割合のずれが大きいと RW などで 2〜10 pt 上がり，ずれが無いと 0.5 pt 未満 | 2〜10 と 0.5 未満は，学習中の再標本化（各エポックで疑似的に釣り合わせる）と事後の重み直しを組んだ手順全体の値．再標本化は重みの学習を変えるので，「(a) に当たる操作の得だけ」とは言えない．第 2 回 4 節の「切片の補正で 2〜18 pt」も同じ理由で不正確 | 食い違い | "(i) pseudo-balance the data at each epoch; and (ii) adjust the final classifier with target label distribution estimate. The meta-algorithm improves existing domain adaptation heuristics under large label proportion shifts, often by 2–10% accuracy points, while conferring minimal effect (<0.5%) when label proportions do not shift." | https://arxiv.org/abs/2302.03020 |
| RLSbench：オラクル | オラクルは早期終了の基準だけ | 付録のオラクルは早期終了の基準だけ | 一致 | "In appendices, we report oracle performance by choosing the early stopping point with target accuracy." | 同上 |
| GLA | 16-shot で推定した偏りを引くと 11 データセットの平均 59.1 → 60.7（+1.6）．微調整したモデルでも末尾のクラスが先頭より 6.3 pt 低い | 同じ（表 5 と 2 節の分解の結果） | 一致 | "We estimate πp at 16 shots ... showcasing an average accuracy gain of 1.6%." ／ "The resulting E2E models are also affected by the bias, with the tail performance being 6.3% lower than the head." | https://arxiv.org/abs/2310.08106 |
| Joshi 2012 | 多ドメイン学習の得の約半分をドメイン ID の特徴（ドメインごとの切片）で再現 | 原文は「多くの場合」約半分と書く．Amazon の FEDA（真のドメイン）は +0.11 で DOM-ID の +0.36 より小さく，「約半分」に当てはまらない例がある．ラベル無しでの切片の当て直しは今後の問いとしてではなく，仮説の動機として 1 文で触れている | 一致（注意） | "We see that in most cases, about half of the improvement seen in MDL is accounted for by a simple baseline of using the domain identifier as a feature" ／ "it suggests that such settings could be amenable to unsupervised adaptation of the bias parameters" | https://aclanthology.org/D12-1119/ |
| Menon 2015 | 雑音のラベルで学んだ確率の AUC と BER は雑音が無いときとほぼ同じ（7.1 節，表 2） | 原文は「全体として同程度」と書き，小さいデータと強い雑音では低下すると書く．例：segment で BER は雑音なし 0.00% に対し (ρ+, ρ−) = (0.2, 0.4) で 3.24% | 一致（注意） | "We see that in general the BER and AUC in the noise-free case ... and in the noisy cases are commensurate. ... For smaller datasets and higher levels of noise, we see a greater degradation in performance." | https://proceedings.mlr.press/v37/menon15.html |
| Nigam & Ghani 2000 | 共訓練は 1 回ごとに各クラス 1 件ずつ（割合が偏るときは経験的な割合どおり）疑似ラベルを足す．News 2x2 で誤り率 34.0% → 3.7% | 同じ．全ラベルを使った naive Bayes は 3.9% | 一致 | "each classifier chooses one unlabeled document per class to add to the labeled set of examples." 脚注 2："If the class frequencies are not even, instead label documents according to the empirical frequencies." ／ "Starting from a baseline error of 34.0%, co-training uses the unlabeled data to reduce error down to 3.7%" | http://www.kamalnigam.com/papers/cotrain-CIKM00.pdf |
| Albanie 2018 | 生徒の予測は中立に偏る．評価はターゲットのラベルで学ぶ行列とバイアスの 1 層．蒸留データの上の ROC AUC は 0.69〜0.71 | 同じ．直接評価しない理由は 2 つで，(1) 生徒の感情の集合と評価の集合が違うこと，(2) 予測が中立に偏ること．報告は (2) だけを理由に書いている．ROC AUC は教師の予測を正解として測った値 | 一致（注意） | "the distribution of predictions made by the student are dominated by the neutral class so the useful signal is primarily encoded through the relative soft weightings of each emotion" ／ "learning a single affine transformation (comprising a matrix multiply and a bias) followed by a softmax" | https://arxiv.org/abs/1808.05561 |
| Chen, Wei, Kumar, Ma 2020 | 線形分類，偽の特徴がガウス・本来の特徴が対数凹の混合．初期の分類器がある程度正しければ，ラベル無しのエントロピー最小化が偽の特徴を使わなくなる | 同じ．抄録だけでなく本文 1 節でも確かめたので ◎ に上げてよい | 一致 | "we prove that entropy minimization on unlabeled target data will avoid using the spurious feature if initialized with a decently accurate source classifier" | https://arxiv.org/abs/2006.10032 |

---

## 2. そのほかの主張（今回確かめた分）

| 主張 | 報告の値 | 原文の値 | 判定 | 引用 | URL |
|---|---|---|---|---|---|
| Davis 2019 | 発生率が上がる場合は切片の補正が選ばれ（局所 1,000 人で 48%，5,000 人で 99.2%），関係が変わる場合は学び直し（96.4〜100%）．最小の標本では学び直しは判別を上げたが較正は上げなかった | 48% は学習 1,000・更新 1,000，99.2% は学習 5,000・更新 5,000 の組の値で，「局所」の 1 つの数ではない．最後の点は「最小の学習標本で，更新の標本が大きいとき」の記述 | 一致（注意） | "refitting with larger updating samples, as recommended, improved discrimination but not calibration" | https://pmc.ncbi.nlm.nih.gov/articles/PMC6857513/ |
| Kappen 2012 | 別の病院 1,389 人で作り，局所 1,847 人で更新し，3,822 人で評価．c は更新後 0.68，元 0.62 | 同じ | 一致 | "The predictive accuracy of the updated model was better (c statistic, 0.68; calibration slope, 1.0) than the original model (c statistic, 0.62; calibration slope, 0.57)." | https://pmc.ncbi.nlm.nih.gov/articles/PMC3426927/ |
| Martin 2017 | 局所 500 件未満では作り直しが過学習し，まとめ直しが判別でも勝った．1,000 件以上で係数の違いが大きいと作り直しが勝った | 同じ趣旨．判別で勝つのは「500 件未満で母集団が似ているとき」 | 一致 | "given training set samples of less than 500 and very similar populations, SR provides consistently higher AUC than redevelopment" | https://pmc.ncbi.nlm.nih.gov/articles/PMC5217317/ |
| Zhao 2021（Calibrate Before Use） | W = diag(p_cf)⁻¹，b = 0．GPT-3 2.7B の AGNews で差は 0-shot 18.5，1-shot 26.6，4-shot 27.8，8-shot 21.9 | 表 1 から 63.2−44.7，59.6−33.0，71.1−43.3，72.7−50.8 で同じ．予測は Wp̂ + b の argmax なので，対数の上でクラスごとの切片を足した場合と同じ予測になる | 一致 | "We then set W = diag(p̂cf)⁻¹ and b to the all-zero vector. To make test predictions, we compute Wp̂ + b and take the argmax." | https://arxiv.org/abs/2102.09690 |
| Cai ほか 2023（DISDE） | DANN は表現の X のずれの項を減らしたが Y|表現 の項を増やし，全体では ERM と同じ | 同じ | 一致 | "For ERM, the ϕ(X) shift terms are largest; for DANN, the Y \| ϕ(X) terms are largest." | https://arxiv.org/abs/2303.02011 |
| LaBonte 2023 | クラスの釣り合わせだけで DFR の最悪群の正解率の 94%（無しで 76%）（△） | 同じ．△ から ◎ に上げてよい．ただし最終層の重みも学び直す手法で，切片だけの操作ではない（報告も「該当しない」に分類している） | 一致 | "this method achieves 94% of DFR worst-group accuracy compared to 76% without class balancing" | https://proceedings.neurips.cc/paper_files/paper/2023/file/265bee74aee86df77e8e36d25e786ab5-Paper-Conference.pdf |
| Elkan & Noto 2008 | 学んだ確率と真の確率は定数倍しか違わない | 同じ．「正例が x に依らず同じ確率でラベルを付けられる」という仮定のもとでの結果 | 一致 | "predicts probabilities that differ by only a constant factor" | https://cseweb.ucsd.edu/~elkan/posonly.pdf |

今回確かめていない主張：Lee 2023，Zhang 2023，FPS，Kirichenko 2023，Izmailov 2022，Rosenfeld 2022，Kang 2020，Burns & Steinhardt 2021，WhyShift，MFH，DebiasPL，Frolic，MASH-Bench，Alexandari 2020，Šipka 2022，Oymak & Gulcu 2021，Berta 2025，Davis 2017，Nestor 2019，Takada 2021，Rockenschaub 2023，LP-FT，BitFit，SimpleShot，Veilleux 2021，Lee 2021，Charoenphakdee 2019，C2KD，MKE，CDMAD・DARP・CReST，Okamoto 1963・Raudys 1998，Blum & Mansour 2017，Maity 2024，Natarajan 2013，Castelli & Cover 1996．どれも依頼の必須の一覧に無く，報告の判定の根拠になる中心の文献ではない．

---

## 3. 前回読めなかった 4 本

| 文献 | 今回の入手先 | 読めたか | 報告の主張と原文 | 水準の判定 |
|---|---|---|---|---|
| Vergouwe ほか 2017（Statistics in Medicine 36: 4529–4539．オンライン版 2016-11-28） | Utrecht 大学の置き場の出版社版 PDF（Semantic Scholar の open access の欄から辿った．前回の Twente 大学の URL は今回も 404） | 本文を読めた | 報告「前立腺がんの例では更新の必要は切片の違いだけから来ていた」は一致．原文は開発集団の陽性 20% に対し更新集団 49% で，切片の補正は 2.58，較正の傾きは 1.01．係数の推定し直しは過学習した（楽観を補正した傾き 0.86）．閉じた検定の手順は切片だけの当て直しを選んだ．引用："The need for updating the prostate cancer model was completely driven by a different model intercept in the update sample (adjustment: 2.58)." 外傷性脳損傷の例は，発生率は違う（24% と 34%）が切片の補正は −0.01 で，傾きが 1.25 だった | ○ → ◎ に上げてよい．URL：https://dspace.library.uu.nl/handle/1874/357924 |
| Janssen ほか 2008（J Clin Epidemiol 61: 76–86） | PubMed の抄録（NCBI efetch）．Semantic Scholar では閉じた扱い，無料の本文は見つからない | 抄録だけ | 報告は「段階として分けた」とだけ書く．抄録は「簡単な当て直しは，より大きく改訂する方法と同程度に較正を直した．判別はどの方法でも上がらなかった」で，報告の扱いと合う．引用："Simple recalibration methods improved the calibration to a similar extent as revision methods ... Discrimination could not be improved by any of the methods." 更新 752 人，評価 283 人 | ○ のまま．URL：https://doi.org/10.1016/j.jclinepi.2007.04.018 |
| Steyerberg ほか 2004（Statistics in Medicine 23: 2567–2586） | PubMed の抄録．Erasmus 大学の置き場（repub.eur.nl/pub/67652）は書誌だけで PDF が無い．Semantic Scholar では閉じた扱い | 抄録だけ | 報告の設定（TIMI-II 3,339 人のモデルを GUSTO-I の 200 人〜10,000 人で更新）と結論（簡素な更新がよく，大きな改訂は大きな標本と縮小を組んだときだけ）は抄録と一致．引用："Parsimonious updating methods were found preferable to more extensive model revisions, which should only be attempted with relatively large validation samples in combination with shrinkage." | ○ のまま．URL：https://doi.org/10.1002/sim.1844 |
| Kull & Flach 2015（ECML PKDD，LNAI 9284: 68–85） | Bristol 大学の置き場に著者の最終稿（2015_ecml_decomposition_cameraready.pdf）がある．自動取得は 403，CORE も 403，Springer は認証が要る | 読めなかった | 4 つの項（調整の損失，調整後の較正の損失，まとめ方の損失，消せない損失）は検索結果の二次資料で確かめた．報告の「調整は出力に定数を足して平均を合わせる操作」は，Brier 得点と対数損失のそれぞれで調整の手順が違う可能性があり，原文で確かめていない | ○ のまま（確かめられず）．URL：https://research-information.bris.ac.uk/en/publications/novel-decompositions-of-proper-scoring-rules-for-classification-s/ |

---

## 4. 報告の判定は妥当か

### 4.1 「完全一致は無い」

妥当である．検証の途中で，①〜④のうち 2 つ以上を「切片の当て直しで取り戻せる分」と「残る分」に分けて実データで比べた研究は見つからなかった．反例を探した検索は 5 回（情報源の比較と切片の当て直しの上限，医療画像のクラスの割合のずれ，切片だけの最終層の学び直し，誤りの切片と向きへの分解，声の感情認識のコーパス間のクラスの割合）である．検索の回数が少ないので，見つからなかったことは「無い」ことの証明ではない．

見落としの部分一致として次の 1 本がある．

- Godau ほか 2023（MICCAI．arXiv 2303.12540）：医療画像の 30 課題で，クラスの割合を人工的にずらし，ずれの後の割合（真の値に小さな乱れを加えたもの）を使って，温度と切片ベクトル b を学ぶ当て直し（"affine scaling"：φ(x)/t + b）の効果を，較正と判定の規則と評価の指標で測った．情報源は「ずれの後の割合を外から知る」1 つだけで，ラベル無しのデータから割合を推定してはいない．分類器の誤りを切片側と向き側に分けてもいない．報告の 1.2 節の③の行に足す候補である．引用："we add a bias term b ∈ R^C to be estimated alongside t ... We refer to this re-calibration approach as 'affine scaling'". URL：https://arxiv.org/abs/2303.12540

### 4.2 「最も近いのは Johnson 2018」

妥当である．①（その病院のラベル 200〜2,400 人）と②（他病院のデータ）を，切片と傾きの当て直しで変わらない判別（AUROC）と，当て直しで変わる較正（SMR）に分けて，同じ評価の患者で比べた研究は，確かめた中では Johnson 2018 だけである．引くときは次の 3 点を直す．

1. 他病院のモデルは 46 病院の半分（学習用の病院）で学んだ．第 1 回 3.1 節の「45 病院」は誤り．
2. 他病院のモデルの当て直しにも，その病院のラベルを使っている．「その病院のラベルが較正を直す」ことは，当て直しの前の SMR を同じ実験で示しておらず，設計から読み取った解釈である（1 つ目の実験では当て直しなしの SMR が病院ごとに 0.560〜1.43 だった）．
3. 局所のモデルの SMR について，原文は「どちらのモデルの SMR も妥当」と書く．「1 を大きく超えた」とは書いていない．

Debray 2013 は「分け方」の点では Johnson 2018 より近い（施設を 1 つずつ外す評価で，切片の選び方だけを変えて較正が直るかを見た）．情報源の比較ではない点で，報告がこれを Johnson 2018 の次に置いたことは妥当である．

### 4.3 「切片と向きの分け方は医療で既知」

妥当であり，今回さらに強く裏づけられた．Vergouwe 2017 の本文は，切片だけの当て直し・切片と傾き・全係数の改訂を順に検定して選ぶ手続きを示し，前立腺がんの例で「更新の必要は切片の違いだけから来ていた」と書く．Janssen 2008 の抄録は「当て直しは改訂と同程度に較正を直し，判別はどの方法でも上がらなかった」と書く．Strobl 2015，Debray 2013，Riley 2016 も同じ区別を実データで示している．論文で「切片と向きに分ける見方を導入した」と書けないという報告の含意（第 1 回 4 節の 1）は正しい．

### 4.4 報告の書き方で直すべき点（まとめ）

1. 第 1 回 3.1 節：「ほかのドメイン（45 病院）」を「46 病院を半分に分けた学習用の病院」に直す．
2. 第 1 回 2 節の RLSbench の行と第 2 回 4 節の表：「2〜10 pt」は再標本化と重み直しを組んだ手順全体の得である．「切片の補正で 2〜18 pt」は DART の値（切片だけでなく W を含む）と混ざっているので，DART の b だけの得（ρ=100 で 13.7 pt）と分けて書く．
3. 第 2 回 1.1 節の Debray 2013：「どの切片の選び方でも」を「全体として」に弱め，事例 3（効果のばらつきが弱ければ発生率からの切片で全研究の E/O が 1 に近い）を足す．
4. 第 2 回 要約 4 と 1.2 節：Debray 2013 の実験の中の発生率は新しい集団のラベルから求めている．「③で切片側を直す操作の先例」は「ターゲットの予測因子の平均（ラベル無し）とクラスの割合（ラベルか外部の情報）で切片を決める操作の先例」と書く．
5. Guo 2022 の −0.003〜0.050 は DG と UDA を合わせた範囲と書く．
6. Vergouwe 2017 と Chen 2020 と LaBonte 2023 の水準を ◎ に上げる．

---

## 5. 検索語の記録（11 回）

| # | 検索語 | 見つかったもの |
|---|---|---|
| 1 | "Validation and updating of predictive logistic regression models: a study on sample size and shrinkage" pdf | 書誌，Erasmus の置き場の記録（PDF なし） |
| 2 | "A closed testing procedure to select an appropriate method for updating prediction models" pdf | Twente の置き場の URL（404），Erasmus の記録 |
| 3 | "Novel decompositions of proper scoring rules for classification" Kull Flach pdf adjustment loss | Bristol の記録，CORE（403） |
| 4 | "Updating methods improved the performance of a clinical prediction model in new patients" pdf | 書誌のみ |
| 5 | Steyerberg Borsboom van Houwelingen Eijkemans Habbema 2004 updating GUSTO-I TIMI-II "recalibration in the large" full text | 無料の本文なし |
| 6 | Kull Flach 2015 "adjustment loss" ... grouping loss | 二次資料（4 つの項の名前） |
| 7 | few target labels versus unlabeled target data versus source data: how much of transfer error is fixed by recalibrating intercept only ... | 完全一致なし．較正の写しの移転の研究など |
| 8 | prevalence shift medical image classification recalibration unlabeled prevalence estimation versus labeled target data ... MICCAI 2023 | Godau 2023 |
| 9 | "bias-only" adaptation last layer few-shot target labels compared to full linear probe retraining ... | 完全一致なし |
| 10 | decompose classifier error under domain shift into "class prior" (bias) component and "decision boundary direction" component ... | 完全一致なし |
| 11 | cross-corpus speech emotion recognition class prior mismatch bias correction few labelled target utterances ... | 完全一致なし |

1〜6 は本文の入手先を探す検索，7〜11 は完全一致の反例を探す検索である．このほかに Semantic Scholar・Unpaywall・PubMed の API で本文の有無と抄録を照会した．

取得した PDF と文字にしたファイルはセッションの一時領域（/tmp/claude-1000/-home-snics-MER-DA/76085a37-4610-4922-b8ae-ced34f163c39/scratchpad/pdf/）にあり，セッションが終わると消える．
