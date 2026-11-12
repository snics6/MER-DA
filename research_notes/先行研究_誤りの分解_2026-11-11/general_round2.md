**検証の結果（2026-11-12，`verify_general.md`）**：4 節の「切片の補正で 2〜18 pt」は RLSbench（手順全体の得）と DART（W を含む）の値が混ざっている．Debray 2013 の「どの切片の選び方でも直らなかった」は原文（"generally give E/O ratios that are not close to 1"）より強い．Debray の発生率は新しい集団のラベルから求めている．

# 誤りの切片側と向き側への分解：一般の機械学習と医療の予測モデルの先行研究（第 2 回，2026-11-12）

担当：第 1 回（`general.md`）が検索の上限で止まった続き．医療の多施設の予測モデル，一般の機械学習の残りの研究，④（ほかのモダリティ・別の見方の予測）の探し直しを調べた．語の定義と水準の記号は第 1 回の 0 節と同じである．

## 要約（第 1 回の判定が変わるか）

1. 第 1 回の判定は変わらない．①〜④のうち 2 つ以上を「切片の当て直しで取り戻せる分」と「残る分」の同じ分け方で実データで比べた研究は，今回の 32 回の検索でも見つからなかった．
2. ①と②を判別と較正に分けて比べた研究として，Johnson ほか 2018 が今も最も近い．掲載先は NeurIPS 2018 の ML4H ワークショップである（arXiv の注記で確かめた）．
3. 分け方が医療統計で既知であることは，今回さらに強く確かめられた．Debray ほか 2013 は，施設ごとに切片だけが違い予測因子の効果が揃っているときは切片の選び方で較正がほぼ直り，効果が施設ごとに違うときはどの切片の選び方でも較正が直らないことを，12 研究の深部静脈血栓症のデータで示した（本文で確認）．
4. Debray 2013 は，新しい施設の予測因子の平均（ラベル無しで得られる）と発生率から切片を決める方法を使っている．これは③で切片側を直す操作の医療での先例である．ただし誤りを情報源ごとに分けて測ってはいない．
5. ④の先例は新たに 1 本見つかった．Albanie ほか 2018 は，顔の教師から声の生徒へ蒸留すると生徒の予測が「中立」に偏ると書き，評価では順位の尺度（ROC AUC）と，ターゲットのラベルで学ぶ「行列とバイアスの 1 層」を使った．偏りの存在の観察だけで，切片側と向き側の量は分けていない．
6. ③は向き側を直さないという観察には医療に先例がある．Guo ほか 2022 では，ラベル無しのドメイン適応（100〜1,500 件）で AUROC はほとんど変わらなかった．
7. 狙うべき差分は第 1 回と同じで，③と④を①②と同じ尺度で測ること，多クラスの UAR で測ること，内訳をラベル数 K の関数として測ることである．研究の記録（`DAI_paper/inventory.md` の項目 1・2・4・5・8）には，この 3 つが既に測られている．

---

## 1. 問い 1：医療の予測モデルの多施設の研究

### 1.1 研究ごとの表

| 研究 | 設定 | 測った量 | 切片側と向き側を分けたか | 情報源 | 水準 | URL |
|---|---|---|---|---|---|---|
| Debray, Moons, Ahmed, Koffijberg, Riley 2013（Statistics in Medicine 32: 3158–3180） | 深部静脈血栓症の診断．12 研究の個人データ（10,014 人，うち 1,897 人が陽性）．ロジスティック回帰．研究を 1 つずつ外して評価する内部外部交差検証（IECV） | 外した研究での E/O（予測した数 ÷ 観測した数），較正の傾き，AUC．切片の選び方（平均の切片，似た研究の切片，発生率から求めた切片，新しいデータで推定し直した切片）ごとに比べた | 分けた．事例 1（予測因子の効果が研究間でほぼ揃う．効果のばらつき τ は 0.03〜0.05）では，平均の切片だと 12 研究のうち 4 つで E/O > 1.2，3 つで E/O ≤ 0.8 だった．発生率が似た研究の切片を使うと，研究 1 で E/O が 1.42 から 1.03 に直った．事例 2（効果が研究間で大きく違う．τ は 0.09〜0.41）では，どの切片の選び方でも E/O は 1 に近づかなかった．事例 2 では AUC が 0.73〜0.92 で，事例 1 の 0.55〜0.65 より高かった（判別は予測因子が決める） | ①（新しい施設のデータで切片を推定し直す）と③に近い操作（新しい施設の予測因子の平均と発生率から切片を計算する．式 4・5） | ◎（PDF 本文 2.2.3 節，4.1〜4.2 節，5 節） | https://doi.org/10.1002/sim.5732 （本文 PDF：https://www.covprecise.org/wp-content/uploads/2020/06/2013-Debray-et-al-A-framework-for-developing-implementing-and-evaluating-clinical-prediction-models-in-an-individual-participant-data-meta-analysis-endnote.pdf ） |
| Riley, Ensor, Snell, Debray, Altman, Moons, Collins 2016（BMJ 353: i3140） | 大きな合併データでの外部検証の総説．QRISK2（364 の診療所），乳がん（8 か国），深部静脈血栓症（12 研究）の例 | 診療所間・国間の C 統計量（AUROC と同じ），E/O，較正の傾きのばらつき | 分けた例を引用．QRISK2 の C 統計量は平均 0.83，診療所間の I²（ばらつきのうち本当の違いによる割合）80.9%，新しい診療所での予測区間 0.76〜0.88．乳がんの例（Snell ほか）では，国ごとに基準ハザード（生存モデルの時間ごとの切片に当たる部分）だけを推定し直すと，較正の傾きの I² が 98% から 35% に下がり，傾きの予測区間が 0.93〜1.08 になった．IECV は方法の説明だけで数値の例は無い | ①（国ごとの当て直し） | ◎（PMC 本文の例 2・5） | https://pmc.ncbi.nlm.nih.gov/articles/PMC4916924/ |
| Johnson, Pollard, Naumann 2018 | 第 1 回の表のとおり | 第 1 回の表のとおり | 第 1 回の表のとおり | ①② | ◎（第 1 回）．掲載先は arXiv の注記 "Machine Learning for Health (ML4H) Workshop at NeurIPS 2018"，報告番号 ML4H/2018/233 で確認 | https://arxiv.org/abs/1812.02275 |
| Strobl, Vickers, Van Calster, Steyerberg, Leach, Thompson, Ankerst 2015（J Biomed Inform） | 前立腺の生検 5 コホート（1994〜2010 年）．年ごとの局所データは 73〜1,106 件 | 既成の計算式をそのまま使う／切片と傾きの当て直し／改訂／作り直し／ランダムフォレスト／ベイズ更新を，毎年の更新で比べた．判別は AUC，較正は Hosmer–Lemeshow 統計量 | 近い．当て直しは較正を大きく直したが判別はほとんど変えなかった（"recalibration often improved calibration but had little impact on discrimination"）．改訂と作り直しも AUC をほとんど上げず，学習データが増えても AUC の上がり幅は小さかった | ①（元の式は別集団で作ったもの） | ◎（PMC 本文） | https://pmc.ncbi.nlm.nih.gov/articles/PMC4532612/ |
| Kappen, Vergouwe, van Klei, van Wolfswinkel, Kalkman, Moons 2012（Med Decis Making） | 術後の吐き気．別の病院（1,389 人）で作ったモデルを局所の 1,847 人で更新し，3,822 人で評価 | 更新したモデル，新しく作ったモデル，元のモデルの c 統計量と較正 | 分けていない．更新は当て直し・係数の改訂・予測因子の追加をまとめて行った．c は更新後 0.68，元 0.62 | ① | ◎（PMC 本文） | https://pmc.ncbi.nlm.nih.gov/articles/PMC3426927/ |
| Martin, Mamas, Peek, Buchan, Sperrin 2017（BMC Med Res Methodol） | シミュレーション．母集団間の違いは予測因子の係数に入れた | 既存の複数のモデルを局所データでまとめ直す方法（stacked regression など）と，局所データだけで作り直す方法を，局所の標本数 150〜10,000 で比べた | 分けていない．当て直し（切片だけ，切片と傾き）は試していない．局所 500 件未満では作り直しが過学習して較正の傾きが 1 を下回り，まとめ直しが判別でも勝った．1,000 件以上で係数の違いが大きいと作り直しが勝った | ①と②に近い（他の母集団のモデル） | ◎（PMC 本文） | https://pmc.ncbi.nlm.nih.gov/articles/PMC5217317/ |
| Guo, Pfohl, Fries ほか 2022（Scientific Reports 12） | MIMIC-IV の ICU，2008〜2019 年を年の群に分けた時間によるずれ．死亡・長期入院・敗血症・人工呼吸．順伝播ネット | 領域汎化（IRM，GroupDRO，CORAL，MMD，敵対学習）とラベル無しのドメイン適応の AUROC と絶対較正誤差 | 分けていない．ラベル無しの適応はターゲットの年の 100・500・1,000・1,500 件を使い，件数による差は無かった．AUROC の差は −0.003〜0.050 で，予測確率の分布のずれも減らなかった．較正の数値は補足表にあり，本文には無い | ③ | ◎（PMC 本文．補足表は未確認） | https://pmc.ncbi.nlm.nih.gov/articles/PMC8854561/ |
| Davis, Lasko, Chen, Siew, Matheny 2017（JAMIA 24: 1052–1061） | 退役軍人病院の急性腎障害．2003 年のデータで 7 つのモデルを作り，その後 9 期間で評価 | 判別と較正の時間変化 | 近い量．判別はすべてのモデルで保たれ，較正は悪化した（予測しすぎ）．予測しすぎは発生率の変化と強く結びつき，予測因子と結果の関係の変化がモデルの種類による較正の違いに対応した | なし（ずれの診断） | ○（抄録） | https://doi.org/10.1093/jamia/ocx030 |
| Nestor ほか 2019（MLHC） | MIMIC-III，2002〜2012 年．2008 年の記録方式の変更 | AUROC の低下と，特徴を臨床の概念にまとめたときの低下 | 分けていない．AUROC だけ（死亡で 0.29 低下，まとめると 0.06） | なし | ○（抄録） | https://arxiv.org/abs/1908.00690 |
| Takada ほか 2021（J Clin Epidemiol 137: 83–91） | 心不全．225 の診療所，871,687 人．IECV | 8 つのモデルの診療所間の C 統計量と O/E のばらつき | 分けていない．非線形の項と交互作用は較正の傾きを少し直したが，O/E の診療所間のばらつきを増やした．判別のばらつきはどのモデルでも同程度 | なし | ○（抄録） | https://doi.org/10.1016/j.jclinepi.2021.03.025 |
| Rockenschaub ほか 2023（arXiv 2303.15354） | MIMIC，eICU，HiRID，AUMCdb（334,812 件） | 単一ソースと複数ソースの学習の AUROC．較正 | 分けていない．新しい病院で AUROC は最大 0.200 下がり，複数ソースは最良の単一ソースと同程度．合併のモデルは危険度を予測しすぎて較正が悪くなったと検索の断片にある | ② | ○（抄録）．較正の記述は △ | https://arxiv.org/abs/2303.15354 |
| Steyerberg ほか 2004，Vergouwe ほか 2017，Janssen ほか 2008 | 第 1 回の表のとおり | 第 1 回の表のとおり | 段階として分けた | ① | ○（今回も出版社の本文を読めなかった．Vergouwe 2017 の大学の置き場の PDF は 404，Janssen 2008 と Steyerberg 2004 は無料の本文が見つからなかった） | https://doi.org/10.1002/sim.1844 ， https://doi.org/10.1002/sim.7179 ， https://doi.org/10.1016/j.jclinepi.2007.04.018 |

### 1.2 医療の研究から言えること

- 「切片が施設ごとに違うだけなら切片の当て直しで較正が直り，判別は変わらない．予測因子の効果が施設ごとに違うと切片では直らない」という区別は，Debray 2013 が 12 研究の実データで示した．論文で切片側と向き側の分け方を使うときは，Debray 2013 と Riley 2016 を Steyerberg 2004・Vergouwe 2017 と並べて引く必要がある．
- Debray 2013 の式 5 は，予測因子を新しい施設の平均で中心化し，切片をその施設の発生率から計算する．予測因子の平均はラベル無しで得られるので，ターゲットの平均を引いて切片を合わせる操作（研究の記録の項目 4）の医療での先例になる．ただし発生率は別に知る必要があり，ラベル無しだけで決まる操作ではない．
- Strobl 2015 と Davis 2017 は，当て直しが較正を直し判別を変えないこと，発生率の変化が予測しすぎ（切片側）に結びつくことを示した．Johnson 2018 と合わせて，①が主に切片側を直すという観察は医療で繰り返し報告されている．
- Guo 2022 は，ラベル無しの適応（③）が判別を上げなかったことを示した．これは研究の記録の項目 5（同じモダリティのラベル無しは向きを直さない）と同じ向きの観察である．ただし Guo 2022 は二値で，切片の当て直しを試していない．
- 新しい施設のラベル数に応じて「切片だけ／切片と傾き／作り直し」を選ぶ研究を，ラベル数を振って実データで比べたものとしては，Davis 2019（第 1 回）と Strobl 2015 が見つかった．Martin 2017 はシミュレーションで，当て直しを含まない．

---

## 2. 問い 2：一般の機械学習の残り

| 研究 | 設定 | 測った量 | 切片側と向き側を分けたか | 情報源 | 水準 | URL |
|---|---|---|---|---|---|---|
| Jang & Chung（KAIST）2024．DART（arXiv 2411.15204．v2 は 2025-02-05．掲載先の記載は無い） | CIFAR-10C-LT．テスト時に BN の統計を入れ替えた分類器（BNAdapt）の出力を，テストのバッチの統計から作る行列 W（K×K）とベクトル b（K）で補正する．W と b を作る小さなネットはソースのデータで学ぶ | 表 3：W と b のどちらを出すか，入力に何を使うかのアブレーション | 近い量．不均衡比 ρ=1／10／100 の正解率は，BNAdapt 85.2／79.0／67.0，b だけ 85.4／83.6／80.7，W だけ 85.0／84.0／83.0，W と b の両方 85.2／84.7／85.1．本文は「不均衡が強いほど W が b より 0.4 pt・2.3 pt よい」と書く．手元の計算では，ρ=100 で b だけが両方の得（18.1 pt）の 76%（13.7 pt），ρ=10 で 81%（4.6／5.7 pt）を取り戻す．W はロジットの行列で，クラス間の順位も変えうるので，研究の記録の「向き」とは同じではない | ③（テストのバッチ） | ◎（PDF 本文の表 3 と 4.2 節）．第 1 回の推定値（b だけ約 80.7，W だけ約 83.0）は正しかった | https://arxiv.org/abs/2411.15204 |
| Berthelot, Roelofs, Sohn, Carlini, Kurakin 2022（ICLR．AdaMatch） | DomainNet の教師なしドメイン適応と半教師ありドメイン適応（ターゲットのラベル数を振る） | 構成要素を 1 つずつ外したときの正解率（6 つの適応の組の平均） | (a) に当たる操作の得だけ．疑似ラベルのクラスの割合をソースの割合に合わせる「分布の合わせ込み」を外すと 26.3 → 17.7．残りの誤りは分けていない | ③ | ◎（PDF 本文の表 4 と 3.1 節） | https://arxiv.org/abs/2106.04732 |
| Kumar, Raghunathan, Jones, Ma, Liang 2022（ICLR．LP-FT） | 理論は過剰パラメータの 2 層線形ネットの回帰（切片の項は無い）．実験は 10 のずれのデータ | 全体の微調整と線形プローブの ID・OOD の正解率 | 分けていない．頭の向きと特徴のゆがみの理論で，切片は扱わない．全体の微調整は ID で平均 2% 高く OOD で 7% 低い | ① | ◎（PDF 本文の 3 節で設定を確認） | https://arxiv.org/abs/2202.10054 |
| Nigam & Ghani 2000（CIKM） | WebKB-Course，News 2x2 など．naive Bayes | 共訓練，EM，共 EM の誤り率 | 分けていない．共訓練は 1 回ごとに各クラス 1 件ずつ（割合が偏るときはラベル付きデータの割合どおりに）疑似ラベルを足す．クラスの割合（切片側）は手順で固定し，各クラスの中の確信度の順位だけを別の見方から取る作りである．News 2x2 で誤り率を 34.0% から 3.7% に下げた | ④（別の見方） | ◎（PDF 本文 3.3 節と脚注 2，4 節） | https://doi.org/10.1145/354756.354805 |
| Ben Zaken, Ravfogel, Goldberg 2022（ACL．BitFit） | BERT の GLUE | バイアス項だけを学ぶ微調整．どのバイアスが大きく動くか | 該当しない．動かすのは注意の query と中間の MLP の内部のバイアスで，出力の切片ではない | ① | ○（抄録と二次資料） | https://arxiv.org/abs/2106.10199 |
| Wang, Chao, Weinberger, van der Maaten 2019（SimpleShot） | miniImageNet など．最近傍法 | 特徴の中心化（基底クラスの平均を引く）と L2 正規化の有無 | 分けていない．中心化に使う平均はターゲットではなく基底クラスのもの．中心化だけではユークリッド距離が変わらず，L2 正規化と組んで初めて効くと本文に書く | なし | ○（抄録と検索の断片） | https://arxiv.org/abs/1911.04623 |
| Veilleux, Boudiaf, Piantanida, Ben Ayed 2021（NeurIPS） | 転導的な少数ショット（ラベル無しのクエリを使う）．クエリのクラスの割合をディリクレ分布で振る | 既存の転導的な手法の正解率 | 分けていない．既存の手法はクエリのクラスが均等という前提（切片側の前提）に頼っており，割合を振ると帰納的な手法より下がる場合がある | ③ | ○（抄録） | https://arxiv.org/abs/2204.11181 |
| Lee, Lei, Saunshi, Zhuo 2021（NeurIPS） | 理論．2 つの見方がラベルと潜在変数のもとで条件付き独立に近いとき，一方から他方を予測する事前課題で得た表現の上の線形の頭は少ないラベルで足りる | 下流の標本数 | 分けていない．切片は扱わない | ④に近い（別の見方） | ○（抄録） | https://arxiv.org/abs/2008.01064 |

---

## 3. 問い 3：④（ほかのモダリティ・別の見方の予測）が切片側と向き側のどちらを直すか

### 3.1 見つかったもの

| 研究 | 設定 | 何を示したか | 水準 | URL |
|---|---|---|---|---|
| Albanie, Nagrani, Vedaldi, Zisserman 2018（ACM MM） | 顔の感情認識の教師から声の生徒へ，ラベル無しの VoxCeleb の動画で蒸留する | 生徒の予測は教師と同じく「中立」に偏る（"the distribution of predictions made by the student are dominated by the neutral class"）．このため生徒の予測を直接は評価せず，生徒の 8 次元の出力からターゲットのラベルへの「行列とバイアスの 1 層」を学んで評価した．蒸留データの上では ROC AUC（平均 0.69〜0.71）で評価した．偏りが切片側のずれであることを観察から認め，切片を含む 1 層をターゲットのラベルで学び直したが，切片だけの学び直しと行列の学び直しを分けてはいない | ◎（arXiv 本文 5 節，5.2 節） | https://arxiv.org/abs/1808.05561 |
| Menon, van Rooyen, Ong, Williamson 2015（ICML） | UCI のデータに，クラスごとに決まった確率でラベルを入れ替える雑音を人工的に入れる | 雑音のラベルで学んだ確率の AUC と，雑音のある割合で閾値を切ったときの均衡誤差（BER）は，雑音が無いときとほぼ同じだった（7.1 節，表 2）．順位は保たれ，閾値だけが雑音の量に応じてずれるという理論を実データ（人工の雑音）で確かめた．雑音は入力に依らない形なので，別のモダリティの予測のように入力ごとに誤り方が変わる疑似ラベルには，そのまま当てはまらない | ◎（PDF 本文 7 節） | https://proceedings.mlr.press/v37/menon15.html |
| Charoenphakdee, Lee, Sugiyama 2019（ICML） | 雑音のラベルからの BER と AUC の最適化 | 対称な損失なら雑音のラベルでの AUC の最適解と雑音の無いラベルでの最適解が一致する．実験は人工の雑音 | ○（抄録と二次資料） | https://arxiv.org/abs/1901.09314 |
| Nigam & Ghani 2000 | 2 節のとおり | 共訓練はクラスの割合をラベル付きデータで固定し，別の見方の確信度の順位だけを使う設計である．切片側を手順で決め，向き側を別の見方から取る分業になっているが，それを測ってはいない | ◎ | https://doi.org/10.1145/354756.354805 |
| C2KD（Huo ほか 2024，CVPR），MKE（Xue ほか 2021，ICCV），MFH（Xue ほか 2023，ICLR） | クロスモーダル蒸留 | C2KD はモダリティ間で正解以外のクラスの順位がずれる（soft label の不整合）ことを問題にした．MKE は単一モダリティの教師の疑似ラベルで学んだ多モダリティの生徒が教師を超えると報告した．どれも生徒の誤りを切片側と向き側に分けていない | ○（抄録と検索の断片） | https://openaccess.thecvf.com/content/CVPR2024/papers/Huo_C2KD_Bridging_the_Modality_Gap_for_Cross-Modal_Knowledge_Distillation_CVPR_2024_paper.pdf ， https://arxiv.org/abs/2103.14431 ， https://arxiv.org/abs/2206.06487 |
| CDMAD（Lee ほか 2024，CVPR），DARP（Kim ほか 2020，NeurIPS），CReST（Wei ほか 2021，CVPR） | クラスの不均衡がある半教師あり学習 | 疑似ラベルのクラスの偏りを直す手法の提案．偏りを直した分の得は測るが，誤りの内訳は測らない | ○（検索の断片） | https://openaccess.thecvf.com/content/CVPR2024/papers/Lee_CDMAD_Class-Distribution-Mismatch-Aware_Debiasing_for_Class-Imbalanced_Semi-Supervised_Learning_CVPR_2024_paper.pdf |

### 3.2 判定

④で学んだ分類器の誤りを，切片の当て直しで取り戻せる分と残る分に分けて実データで測った研究は，今回も見つからなかった．文献にあるのは次の 3 種類である．

- 理論と人工の雑音での確認：雑音のラベルで学んでも順位は保たれ閾値だけがずれる（Menon 2015，Charoenphakdee 2019，Elkan & Noto 2008，Blum & Mansour 2017）．
- 手順での分業：共訓練はクラスの割合をラベル付きデータで決め，別の見方からは各クラスの中の順位だけを取る（Nigam & Ghani 2000，Blum & Mitchell 1998）．
- 観察：顔から声への蒸留で生徒の予測が中立に偏る（Albanie 2018）．

研究の記録の項目 8（配備する分類器が見ないモダリティを含むモデルの疑似ラベルが向きの誤りを 81 単位のうち 38 で減らし，増やしたのは 1）は，上の理論の「疑似ラベルは向きの情報を運ぶ」と同じ向きの実測にあたる．入力ごとに誤り方が変わる疑似ラベル（別のモダリティの予測）で，向きの誤りが減ることを測った例は見つからなかった．

---

## 4. 最も近い先行研究（第 1 回の 4 本に今回の 2 本を足した比較）

| 研究 | データとラベルの量 | モダリティと学習器 | 測る量 | 主張 | MER-DA との同じ点 | 違う点 |
|---|---|---|---|---|---|---|
| Johnson 2018 | eICU の 46 病院，局所 200〜2,400 人 | 表データ，ロジスティック回帰 | AUROC と SMR | 局所のラベルは較正を直し，判別は他病院のモデルに届かない | ①と②の比較，ラベル数を振る，線形 | 二値，1 病院，③④無し，割合に分けない |
| Debray 2013 | 12 研究，10,014 人 | 表データ，ロジスティック回帰 | E/O，較正の傾き，AUC（IECV） | 予測因子の効果が揃えば切片の選び方で較正が直り，揃わなければ直らない | 施設を 1 つずつ外す評価（コーパスを 1 つずつ外すのと同じ形），切片を新しい施設の平均と割合で決める（③に近い） | 二値，ラベル数を振らない，情報源の比較ではない |
| Strobl 2015 | 5 コホート，年ごとの局所 73〜1,106 件 | 表データ，ロジスティック回帰ほか | AUC と Hosmer–Lemeshow | 当て直しは較正を直し判別をほとんど変えない | ①で切片側が直ることの繰り返しの確認 | 二値，③④無し |
| Joshi 2012 | 第 1 回のとおり | 言葉，線形分類器 | 多ドメイン学習の得 | 得の約半分はドメインごとの切片で再現できる | ②の得を切片で分ける | ラベル数を振らない，偏りは人工 |
| RLSbench 2023・DART 2024 | 第 1 回と 2 節のとおり | 画像など | ③の上に切片の補正を足した得 | 割合のずれが大きいと切片の補正で 2〜18 pt 上がる | ③と切片の補正 | ③の手法そのものの内訳は測らない |
| Albanie 2018 | VoxCeleb のラベル無しの動画で蒸留，評価はターゲットのラベルで 1 層を学ぶ | 顔 → 声，CNN | ROC AUC，正解率 | 生徒の予測は中立に偏るが，1 層の写像で使える情報を持つ | ④（別のモダリティの予測），感情認識 | 切片と行列を分けない，ラベル数を振らない |

---

## 5. 狙うべき差分

先行研究が測っていない量で，研究の記録（`DAI_paper/inventory.md`）に既にあるものを優先して挙げる．

1. **③と④を①②と同じ尺度で並べること．**医療の研究は①（局所のラベル）と②（他施設のデータ）を判別と較正で比べたが，③は Guo 2022 の判別だけ，④は扱っていない．記録の項目 4・5（同じモダリティのラベル無しは位置を直し方向を直さない．8 手法の 210 単位で方向の誤りを減らしたのは 7）と項目 8（ほかのモダリティを含むモデルの疑似ラベルは方向を直す．81 単位で 38）は，情報源ごとに切片側と向き側を分けた実測で，文献に対応するものが無い．
2. **多クラスの UAR で切片側と向き側を分けること．**医療の研究は二値で AUROC と較正（E/O，傾き）を使う．記録では感情の対ごとに位置の誤りと方向の誤りを測っている．第 1 回の 4 節の注釈 5（2 クラスに限った AUROC はクラスごとの切片で変わらない）を使えば，医療の判別の尺度と対応づけて書ける．
3. **内訳をラベル数 K の関数として測ること．**Johnson 2018，Strobl 2015，Martin 2017 はラベル数を振ったが，誤りを割合に分けていない．記録の項目 1・2（K 本は切片を安く直す，方向は D/√(1+q/K) に従って遅く直す．音声では K=10 でも方向の誤りの 6〜7 割が残る）は，この割合を K の関数として測った結果である．
4. **④が向き側を直すことを，入力ごとに誤り方が変わる疑似ラベルで実測したこと．**順位が保たれるという理論は，入力に依らない雑音（Menon 2015）での確認しかない．記録の項目 8 はこの点の実測になる．ただし Albanie 2018 が生徒の予測の偏り（切片側）を報告しているので，論文では「④の疑似ラベルで学ぶと切片側のずれが入り，向き側の情報も入る」と両面を書く必要がある．

推測として分けておくこと：Debray 2013 の「効果が揃えば切片で直る」と，記録の項目 7（言葉は位置の誤りが小さく方向の誤りが大きい）や項目 2 のモダリティごとの q の違いとを対応づけられる可能性がある．これは対応づけの案で，確かめていない．

---

## 6. 見つからなかったこと

- ①〜④のうち 2 つ以上を，切片の当て直しで取り戻せる分と残る分の同じ分け方で比べた研究．
- ④で学んだ分類器の誤りを切片側と向き側に分けた実データの研究．Albanie 2018 の観察と，人工の雑音での確認（Menon 2015）までしか見つからなかった．
- 医療の多施設の研究で，ラベル無しのデータ（③）で切片を当て直した分の効果を，局所のラベル（①）による当て直しと比べたもの．Debray 2013 は発生率も使うので純粋な③ではない．
- Johnson 2018 の後続で，局所のラベル数と当て直しの段階を同じ枠で比べた研究．eICU を使った後の研究（Rockenschaub 2023 など）は単一ソースと複数ソースの判別の比較だった．
- Nestor 2019 は AUROC だけで，較正は扱っていなかった（抄録の範囲）．
- 多クラスで「切片の当て直しが取り戻す均衡正解率」を情報源ごとに測った研究．

## 7. 調査の限界

- Steyerberg 2004，Vergouwe 2017，Janssen 2008 は今回も本文を読めず，抄録だけである．Vergouwe 2017 の大学の置き場の PDF は 404 だった．
- Guo 2022 の較正の数値は補足表にあり，読んでいない．
- Davis 2017，Takada 2021，Rockenschaub 2023，BitFit，SimpleShot，Veilleux 2021，Lee 2021，C2KD，MKE は抄録か検索の断片だけである．
- PMC の本文は WebFetch の要約器を通して読んだ．引用した数値と文はその出力による．Debray 2013，DART，AdaMatch，LP-FT，Nigam & Ghani，Menon 2015，Albanie 2018 は PDF を文字にして該当箇所を直接読んだ．
- 取得した PDF はセッションの一時領域にあり，セッションが終わると消える．

## 8. 検索語の記録（第 2 回．32 回）

第 1 回の 66 回目（上限で実行されなかった語）の続きである．著者名は含むが，プロジェクト名や研究者の名前は含めていない．

| # | 検索語（英語） | 見つかったもの | 見つからなかったもの |
|---|---|---|---|
| 1 | internal-external cross-validation clinical prediction model calibration-in-the-large heterogeneity across centres | Takada 2021，Van Calster の群別の較正曲線，BMJ の評価の手引き | 本文 |
| 2 | Debray 2013 "internal-external cross-validation" heterogeneity baseline risk intercept predictor effects individual participant data prediction model | Debray 2013 の書誌と PDF，de Jong 2021 | ― |
| 3 | Riley 2016 external validation clinical prediction models big datasets heterogeneity calibration discrimination across hospitals BMJ | Riley 2016（PMC） | ― |
| 4 | Johnson Pollard Naumann "Generalizability of predictive models for intensive care unit patients" eICU | arXiv の版と注記（ML4H 2018） | 雑誌の版 |
| 5 | Guo 2022 temporal dataset shift clinical domain generalization domain adaptation AUROC calibration Scientific Reports | Guo 2022 | 較正の数値（補足表） |
| 6 | Nestor 2019 feature robustness non-stationary health records MIMIC year-agnostic AUROC calibration | Nestor 2019 | 較正 |
| 7 | Davis 2017 calibration drift regression machine learning acute kidney injury discrimination stable calibration deteriorated JAMIA | Davis 2017 | 本文 |
| 8 | Janssen Moons Kalkman Grobbee Vergouwe 2008 "Updating methods improved the performance of a clinical prediction model in new patients" postoperative nausea recalibration in the large logistic recalibration model revision results | Janssen 2008 の書誌，Kappen 2012（PMC） | Janssen 2008 の本文 |
| 9 | Vergouwe Nieboer Oostenbrink Debray Murray Kattan Koffijberg Moons Steyerberg 2017 closed testing procedure prediction model updating full text PMC | 大学の置き場の PDF の URL（404），多項ロジスティックの版 | 本文 |
| 10 | review statistical updating methods clinical prediction models Su Jaki Hickey 2018 intercept recalibration sample size new setting simulation | Su 2018 の総説，Schnellinger 2021 | 本文 |
| 11 | "Validation and updating of predictive logistic regression models: a study on sample size and shrinkage" Steyerberg Borsboom van Houwelingen Eijkemans Habbema pdf | 書誌 | 無料の本文 |
| 12 | Kumar Raghunathan Jones Ma Liang 2022 fine-tuning can distort pretrained features linear probing then fine-tuning analysis head bias out-of-distribution | LP-FT | 切片の扱い |
| 13 | Nigam Ghani 2000 "Analyzing the effectiveness and applicability of co-training" pdf | 書誌（PDF は著者のサイトから取得） | ― |
| 14 | BitFit bias-only fine-tuning analysis which bias terms change most query bias intermediate MLP "Ben Zaken" | BitFit | 出力の切片の分析 |
| 15 | SimpleShot nearest neighbor few-shot centering L2 normalization base class mean subtraction ablation gains Wang Chao Weinberger van der Maaten | SimpleShot | アブレーションの数値 |
| 16 | co-training pseudo-labels other view errors "class prior" bias versus ranking quality analysis empirical multi-view AUC preserved threshold shifted | 対称な損失の総説（2101.01366），多視点の順位学習 | 共訓練で閾値のずれを実測した研究 |
| 17 | cross-modal distillation student "decision boundary" bias calibration analysis audio-visual teacher pseudo-labels class imbalance student predictions skewed | C2KD，MFH，情報理論の基準（2510.13182） | 生徒の切片と向きの分解 |
| 18 | multimodal emotion recognition cross-modal pseudo-labels unlabeled target logit adjustment class prior correction ablation which errors corrected | HADUA，CMC，自己学習の MER | 誤りの種類のアブレーション |
| 19 | Charoenphakdee Lee Sugiyama symmetric losses learning from corrupted labels AUC balanced error rate experiments ICML 2019 | Charoenphakdee 2019 | 本文 |
| 20 | pseudo-labels from another modality class distribution mismatch student classifier recalibrate bias only few labeled target examples speech emotion text teacher | Agarla 2024，2509.20706，教師の動物園（IJCNLP 2025） | 別モダリティの教師で生徒の切片を K 本で直した研究 |
| 21 | self-training pseudo-labels "class prior" drift "distribution alignment" ablation gain decomposition ReMixMatch FixMatch analysis what pseudo-labeling fixes | CDMAD，CReST，DARP，DC-SSL，AdaMatch | 疑似ラベルの得の内訳 |
| 22 | practice-specific updating risk calculator local data sample size recalibration versus refitting compared global model Strobl Vickers Ankerst prostate cancer | Strobl 2015（PMC） | ― |
| 23 | hospital-specific versus multicenter model local recalibration number of local patients needed discrimination calibration learning curve electronic health records transfer machine learning | Martin 2017，Kappen 2012，農村の病院の較正（medRxiv 2025．本文は 403） | 判別の学習曲線 |
| 24 | Takada Nijman 2021 Journal of Clinical Epidemiology internal-external cross-validation large clustered data between-practice heterogeneity calibration-in-the-large calibration slope discrimination results | Takada 2021 | 本文の数値 |
| 25 | Martin Mamas Peek Buchan Sperrin 2017 clinical prediction in defined populations simulation when and how to aggregate existing models redevelopment local sample size calibration discrimination | Martin 2017（PMC） | ― |
| 26 | Menon van Rooyen Ong Williamson 2015 learning from corrupted binary labels class-probability estimation experiments AUC unaffected threshold adjustment UCI | Menon 2015 の PDF | ― |
| 27 | Lee Lei Saunshi Zhuo 2021 "Predicting what you already know helps" provable self-supervised learning conditional independence linear probe downstream | Lee 2021 | 切片の扱い |
| 28 | Albanie Nagrani Vedaldi Zisserman 2018 emotion recognition in speech cross-modal transfer in the wild teacher face student voice class distribution neutral bias analysis | Albanie 2018 の arXiv | ― |
| 29 | "cross-modal" pseudo-labels student "linear probe" versus "bias" recalibration target labels how much of student error is due to class prior mismatch | MKE，DCASE 2026 の技術報告（ロジット調整） | 生徒の誤りのうち割合のずれによる分 |
| 30 | Veilleux Boudiaf Piantanida Ben Ayed 2021 realistic evaluation transductive few-shot learning class-balance bias Dirichlet query imbalance TIM marginal entropy | Veilleux 2021 | 周辺のエントロピーの項の数値 |
| 31 | knowledge distillation unlabeled target teacher pseudo-labels student post-hoc prior correction balanced accuracy gain "label shift" teacher-student domain adaptation analysis bias versus feature | 雑音のラベルの蒸留（1703.02391），2403.11675，2412.20337 | 生徒の誤りの切片と特徴への分解 |
| 32 | eICU cross-hospital generalizability mortality model transfer AUROC calibration hospital-specific fine-tuning local data amount follow-up study 2022 2023 | Rockenschaub 2023，2512.06511 | Johnson 2018 の形の後続 |

本文の確認に使った WebFetch と PDF の取得：Johnson 2018 の arXiv の書誌，Riley 2016（PMC4916924，2 回），Debray 2013（PDF），Guo 2022（PMC8854561），Nestor 2019（arXiv の書誌），Kappen 2012（PMC3426927），Strobl 2015（PMC4532612），Martin 2017（PMC5217317），DART（arXiv の書誌と PDF），LP-FT（PDF），Nigam & Ghani 2000（PDF），AdaMatch（PDF），Menon 2015（PDF），Albanie 2018（PDF），Rockenschaub 2023（arXiv の書誌）．
