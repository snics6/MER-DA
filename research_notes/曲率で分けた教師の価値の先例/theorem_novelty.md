# 定理の候補 C1〜C3 の先例：食い違いから推した精度で見方を平均する保証，K 本のラベルで決まらない方向，4 つ以上の見方での独立の前提の確かめ

調べた日：2026-10-28．対象は E72（`docs/exploratory-analyses.md` の「E72 の下調べ」）の命題 9・12・13 を言い直した 3 つの主張 C1〜C3 である．`docs/related-work.md` 1.17 節と，同じフォルダの `theory.md`・`methods.md` で調べ済みの文献（MacKay 1992，Obst 2021，Dar ほか 2024，Wu ほか 2022，Ildiz 2025，Kakade & Foster 2007，MetaReg，PSPA/PPI++，Platanios ほか，共同学習の理論，EWC/L2-SP/Fisher merging/OGD/GPM，Grubbs 1948）は調べ直していない．ただし Platanios ほか 2014 は，C1 と C3 の判定に要る式を確かめるために本文を読み直した．

確かめた水準は `theory.md` と同じ 3 段で書く．◎ 本文の該当の節・定理を読んだ．○ 抄録・書誌データベース（Crossref）・要約器（WebFetch）を通して確かめた．△ 検索結果の抜き書きか記憶だけ．PDF は curl で取り，`pdftotext` で文字にして grep した（Gruber の博士論文，Zwieback ほか 2012，Tenzer ほか 2022，Jaffe ほか 2016，Platanios ほか 2014，DECISION）．

**判定の早見表**

| 主張 | 判定 | 最も近い先例 | 重なる内容 | 残る新しさ |
|---|---|---|---|---|
| C1（K に合わせた食い違いの重みと，すべての K での保証） | 部分的な重なり．推定と平均の手順は出版済み．保証の不等式と，学び直しの計量 M_K で重みを K に合わせる点は見つからなかった | 地球科学の triple collocation（TC）による合併（Yilmaz ほか 2012，Gruber ほか 2017・2019），three-cornered hat の相関による偏り（Anthes & Rieckh 2018 の式 A8），機械学習の教師なしの回帰の合併（Tenzer ほか 2022） | 対ごとの食い違いを最小 2 乗で解いて各データの誤差の分散を推し，逆数の重みで平均する．相関を無視した重みは「最良のデータへ重みを寄せるだけで，不確かさを大きくは増やさない」（Gruber ほか 2019 の定性的な主張） | (1) 平均する対象がモデルのパラメータ（頭の重み）で，内積が学び直しの後の超過リスクの計量 M_K であること，(2) 実現した相関 ρ だけで書いた決定的な不等式と，推定の誤差 ε ≤ 3κρ の伝わり方，(3) K = 0（教師なしの DA）と K ≥ 1（半教師ありの DA）を同じ形の保証で扱うこと |
| C2（K に応じて縮む，決められない方向の集合） | 部分的な重なり．仕組み（情報行列が特異な向きのパラメータは決まらない）は古典である | Rothenberg 1971（局所的な識別可能性 ⇔ 情報行列の正則性），Ben-David ほか 2010（ターゲットのラベルが無い DA の不可能性），Zwieback ほか 2012（collocation で見えない誤差の共分散の部分空間） | パラメータを情報行列の零空間の向きにずらしても観測の分布が変わらないこと．ラベル無しでは仮定なしに DA を保証できないこと | 決められない集合が N_K = null(H_K) とちょうど一致し，K が増えると縮み，K = 0 で全空間になること．これを「ラベル無しで決めるどの重みの規則も最良の見方に競合できない」という重みの不可能性の形で書いたもの |
| C3（見方が 4 つ以上なら，一部の組だけに共通の誤差を残差で見つけられる．3 つでは見つけられない） | ほぼ出版済み（スカラーの時系列と回帰の予測の設定で） | Zwieback ほか 2012，Pierdicca ほか 2015・2017，Gruber ほか 2016（地球科学），Tenzer ほか 2022（機械学習），Jaffe ほか 2016（分類器） | 4 つ以上のデータで方程式が未知数より多くなり，誤差の相関の一部を推せる，または相関のある組を自動で特定できる．対ごとの差の共分散で，独立の前提をラベル無しで確かめられる | パラメータのベクトルと計量 M_K への移し替えだけ．論文では新しさとして主張せず，先例として引く |

---

## C1：K に合わせた食い違いの重みと，すべての K で成り立つ保証は出版されているか

### Takeaway
各見方の誤差の大きさを対ごとの食い違いの最小 2 乗で推し（three-cornered hat，N-cornered hat，TC），その逆数で重みを付けて平均する手順は，地球科学で 2012 年から使われており，機械学習でも予測の合併として出版されている（Tenzer ほか 2022）．一方で，平均の誤差を「実現した誤差の相関 ρ」だけで抑える決定的な不等式，推定の誤差 ε ≤ 3κρ を通した最良の見方との比，平均する対象をモデルのパラメータにし内積を学び直しの後のリスクの計量 M_K にして重みを K に合わせる点は，調べた範囲では見つからなかった．判定は「部分的な重なり」である．不等式そのものはコーシー・シュワルツの不等式から数行で出るので，新しさは小さい．

### Cited Findings

#### 食い違いから誤差の大きさを推す部分（出版済み）
- Stoffelen 1998 が TC を導入した．3 つのデータの誤差が無相関という仮定で，各データの誤差の分散と線形の較正係数を推す（書誌 ○．Journal of Geophysical Research: Oceans 103(C4):7755–7766）— [DOI](https://doi.org/10.1029/97jc03180)；方法の説明は [Anthes & Rieckh 2018 の付録 A1](https://amt.copernicus.org/articles/11/4239/2018/)
- three-cornered hat はもともと原子時計の安定度を測る方法で，Gray & Allan 1974 が N 個の時計の場合（N-cornered hat）に広げ，「(N−1)(N−2)/2 個の三つ組をすべて作る」（○．Anthes & Rieckh 2018 の付録 A1 の記述．Gray & Allan の書誌は Crossref で確かめた：28th Annual Symposium on Frequency Control, pp. 243–246）— [Anthes & Rieckh 2018](https://amt.copernicus.org/articles/11/4239/2018/)；[DOI](https://doi.org/10.1109/freq.1974.200027)
- Zwieback, Scipal, Dorigo, Wagner 2012 は，collocation を差の積の「括弧」[i,j;k,l]_n = (y_i − y_j)(y_k − y_l) = (e_i − e_j)(e_k − e_l) で書き直した．真の値は差で消えるので，括弧は誤差だけの関数になる（◎ 4.1 節の式 (7)）．これは C1 の D_st = (x_s − x_t)ᵀM_K(x_s − x_t) と同じ構造であり，違いは内積が時刻 n についての和か，M_K の内積かだけである — [NPG 19:69–80](https://npg.copernicus.org/articles/19/69/2012/)
- Anthes & Rieckh 2018 は，three-cornered hat の推定値に誤差の共分散が入る形を付録 A2 の式 (A8) で書いた：「VARerr(RO) = ½MS(RO−ERA) + ½MS(RO−GFS) − ½MS(GFS−ERA) + COVerr(RO,ERA) + COVerr(RO,GFS) − COVerr(GFS,ERA)」．three-cornered hat は共分散の項を落とすので，推定の偏りはこの共分散の組み合わせになる（○ WebFetch を通して式を確かめた）．C1 の命題 9(c) の恒等式 Ê − E = −2A⁺c は，V = 3 ではこの式と同じである — [AMT 11:4239–4260](https://amt.copernicus.org/articles/11/4239/2018/)
- Platanios, Blum, Mitchell 2014 は，分類器の一致率 a_{i,j} から誤り率を推す．独立な誤りをする 3 つの関数では「only 3 unknown variables ... and we have 3 equations」で，誤り率を一致率で直接解く（式 (7)(8)）．独立を仮定しない場合は「the set of error rates that minimize the dependence among errors」を選ぶ（◎ 3.1.1・3.1.2 節）．これは three-cornered hat の分類版である — [UAI 2014 の PDF](http://auai.org/uai2014/proceedings/individuals/313.pdf)
- Tenzer, Dror, Nadler, Bilal, Kluger 2022（AISTATS）は，m 個の回帰の予測だけ（ラベル無し）から各予測の平均 2 乗誤差を推し，合併した予測を作る（U-PCR）．誤差が互いに無相関なら予測の共分散の非対角が低ランクの成分と一致し，未知数は組 × 予測の設計行列 B1（各行が 1 つの組で，その組の 2 つの列に 1）を通して解かれる（◎ 4 節の式 (8)〜(10)）．この B1 は C1 の最小 2 乗の設計行列 A と同じ形である — [PMLR 151](https://proceedings.mlr.press/v151/tenzer22a.html)

#### 推した誤差の逆数で重みを付けて平均する部分（出版済み）
- Yilmaz, Crow, Anderson, Hain 2012 は，モデル・熱赤外・マイクロ波の土壌水分を最小 2 乗の枠組みで合併し，各入力の不確かさを TC で推した．利用者が決める変数の要らない「objective」な方法と書く（書誌 ○．Water Resources Research 48(11)．中身は △ 検索結果の抄録の抜き書き）— [DOI](https://doi.org/10.1029/2011wr011682)
- Gruber, Dorigo, Crow, Wagner 2017 は，能動と受動のマイクロ波の衛星の土壌水分を，TC で推した誤差の分散から決めた重みで平均する．TC が信頼できない格子では，相関の有意確率（p 値）で能動のみ・受動のみ・重み無しの平均を選ぶ（書誌 ○．IEEE TGRS 55(12):6780–6792．中身は △ 抄録の抜き書き）— [DOI](https://doi.org/10.1109/tgrs.2017.2734070)
- Gruber, Scanlon, van der Schalie, Wagner, Dorigo 2019（ESA CCI の土壌水分の合併の方法）は，最適な推定値を「the weighted average of the individual measurements with the weights being derived from their error variances and covariances」とし，重みの行列を誤差の共分散行列の逆行列 P = C⁻¹ とする（5 節の式 (2)）．v3 では誤差の共分散行列の非対角を 0 とおいた．6.3 節はこれが「may lead to biases in the estimated weights」と認めたうえで，「it cannot lead to a substantial uncertainty increase」と書き，理由を「error correlations only pull the weights further towards the best product」とする（○ WebFetch を通して該当の文を確かめた）．合併の結果が最良の入力より誤差が小さいことの保証は書いていない — [ESSD 11:717–739](https://essd.copernicus.org/articles/11/717/2019/)
- 降水量や蒸発散量の製品でも，three-cornered hat で推した誤差の分散・共分散の行列の逆行列で重みを付けて合併する研究が複数ある（△ 検索結果の抜き書き．個別の論文は確かめていない）— [HESS 22:4513（2018）](https://hess.copernicus.org/articles/22/4513/2018/hess-22-4513-2018.pdf)；[Water 16:2603（2024）](https://www.mdpi.com/2073-4441/16/18/2603)
- Tenzer ほか 2022 は，ラベル無しで推した精度から線形の合併の重みを作る（式 (15)）．多くの回帰の問題で予測の平均と中央値より良いと報告する．重みは誤差の分散の逆数ではなく，スペクトルの方法で推した「予測と応答の共分散」ρ から作る．理論は摂動の解析で，推定の安定性 ∥ρ̂ − ρ∥ = O(∥L̂ − L∥, |ĝ2 − g2|) を示す（◎ 1 節と 4 節）— [PMLR 151](https://proceedings.mlr.press/v151/tenzer22a.html)．前身は Dror, Nadler, Bilal, Kluger の arXiv 1703.02965（○ 抄録）— [arXiv](https://arxiv.org/abs/1703.02965)
- 分類器では，Parisi, Strino, Nadler, Kluger 2014（PNAS 111(4):1253–1258．書誌 ○）がラベル無しで分類器を順位付けして合併する．Jaffe, Fetaya, Nadler, Jiang, Kluger 2016（AISTATS）は条件付き独立が崩れる場合を扱う（後述 C3）— [Parisi の DOI](https://doi.org/10.1073/pnas.1219097111)；[Jaffe の PDF](https://proceedings.mlr.press/v51/jaffe16.pdf)

#### ラベル無しの重みで「最良の 1 つ」に負けない保証（仮定が違う先例）
- Ahmed, Raychaudhuri, Paul, Oymak, Roy-Chowdhury 2021（DECISION．CVPR 2021, pp. 10098–10107．書誌 ○）は，ソースのデータを使わずに複数のソースのモデルを混ぜる重みを，ラベル無しのターゲットで教師なしの損失を最小にして学ぶ．Lemma 1 は，損失が凸で「the target distribution is exactly equal to the mixture of source distributions」なら，分布の重みで混ぜた予測が L(Q_T, θ_T) ≤ min_j L(Q_T, θ_S^j) を満たすと示す（◎ 3 節の Lemma 1）．保証の仮定は「ターゲットがソースの混合」であり，誤差の相関ではない．重みは標本ごとで，K 本のラベルでの学び直しは扱わない — [arXiv 2104.01845](https://arxiv.org/abs/2104.01845)；[DOI](https://doi.org/10.1109/cvpr46437.2021.00997)
- Gruber ほか 2019 の「相関は重みを最良の製品へ寄せるだけ」という文（上記）は，C1 の命題 9 の定性的な版である．不等式では書いていない — [ESSD](https://essd.copernicus.org/articles/11/717/2019/)

#### 重みを K に合わせる部分（ラベルを使う先例だけ）
- Tommasi, Orabona, Caputo 2010（CVPR, pp. 3081–3088．書誌 ○）と 2014（IEEE TPAMI 36(5):928–941，DOI 10.1109/TPAMI.2013.197．書誌 ○）は，少数のターゲットの例で新しい物体を学ぶとき，どのソースのモデルからどれだけ移すかを「minimal leave-one-out error on the available training set」になる凸最適化で選ぶ（○ 抄録）．重みはラベル付きの少数の例で決まるので，K に応じて変わるが，ラベルを使う — [Crossref の DOI（2010）](https://doi.org/10.1109/cvpr.2010.5540064)；[IRIS の PDF（2014）](https://iris.polito.it/retrieve/e384c430-7565-d4b2-e053-9f05fe0a1d67/29230c_400ee65bc2cd44bbb5351d78df78c33c.pdf)；[ML Anthology](https://mlanthology.org/cvpr/2010/tommasi2010cvpr-safety)
- Kuzborskij & Orabona 2017 の偏った正則化 ‖w − W_src β‖² も，ソースの重み β をラベル付きのターゲットで決める（`theory.md` で確かめ済み）— [arXiv](https://arxiv.org/abs/1412.1619)

#### モデルのパラメータの平均（ラベル無し）
- Choshen, Venezian, Slonim, Katz 2022 は，複数の微調整済みモデルの重みを平均して，次の微調整の出発点にすると事前学習モデルより良いと報告する（○ 抄録）．重みは平均で，誤差の推定や保証は無い — [arXiv 2204.03044](https://arxiv.org/abs/2204.03044)
- RegMean（Jin ほか ICLR 2023）は，学習データを使わずに内積の行列だけで合併する（△ 検索結果の抜き書き）．AdaMerging，TIES のような合併法も含めて，「食い違いから推した誤差で重みを付け，誤差の相関で保証する」合併の論文は見つからなかった — [OpenReview の PDF](https://openreview.net/pdf?id=FCnohuR6AnM)

### Inferences
- **推定と平均の手順は先例どおりである．**D_st = E_s + E_t の最小 2 乗は N-cornered hat そのもので，逆数の重みの平均は TC による合併そのものである．論文では Grubbs 1948，Gray & Allan 1974，Zwieback ほか 2012，Gruber ほか 2017・2019，Tenzer ほか 2022 を引き，「この推定量と重みは既知であり，本研究はそれをパラメータの空間に移す」と書くのが正直である．
- **新しいと言える部分は 3 つである．**(1) 平均する対象が，教師モデルの疑似ラベルで当てはめた頭のパラメータ（ベクトル）で，内積が学び直しの後の超過リスクを測る M_K = (I + H_K)⁻¹F(I + H_K)⁻¹ であること．先例の内積は時刻や標本についての和であり，パラメータの方向についての和ではない．(2) 推定の誤差（Anthes & Rieckh の式 (A8) の偏り）を ε ≤ 3κρ で抑え，平均の誤差を ((1 + ε)/(1 − ε))²(1 + (V − 1)ρ)H で抑える，確率の仮定の無い不等式．(3) K = 0 では M_0 = F，K ≥ 1 では M_K と計量を替えるだけで，教師なしの DA と半教師ありの DA を同じ形で扱うこと．ラベル無しのまま重みが K に応じて変わる先例は見つからなかった（K に応じて重みを変える Tommasi ほかはラベルを使う）．
- **不等式は短く，新しさは小さい．**1 + (V − 1)ρ は，誤差の分散が等しく相関が一定の V 個を平均したときの分散の膨らみ方（標本調査の design effect と同じ形）を，分散が違い相関の上限が ρ の場合に広げたものである．論文では「命題」として置き，主定理にしない方が査読で突かれにくい．design effect の一次の出典（Kish 1965）は本調査では確かめていない．
- **「同じ重みの平均にも負けない」の書き方に注意が要る．**E72 の命題 9(a) が比べているのは，同じ重みの平均の対角の項 (1/V²)ΣE_t であり，同じ重みの平均の誤差そのものではない．誤差の相関が負なら，同じ重みの平均の誤差は対角の項より小さくなりうる．|ρ_st| ≤ ρ なら同じ重みの平均の誤差は (1 − (V − 1)ρ)(1/V²)ΣE_t 以上なので，比は (1 + (V − 1)ρ)/(1 − (V − 1)ρ) 倍まで広がる．論文ではこの係数で書く．
- **「すべての K で」の保証の強さは K で変わる．**ρ_K は M_K の内積での相関で，K が増えると M_K が少数の方向に集まり有効な次元が下がるので，命題 10 により実現した相関は大きくなりやすい．保証の形はすべての K で同じだが，保証の強さは K が大きいほど弱くなりうる．「K に一様な定数」とは書かない．
- **Gruber ほか 2019 の定性的な主張は，C1 の命題 9 の弱い版である．**同論文は「相関は最良の製品へ重みを寄せるだけ」と書くが，一部の組だけが誤差を共有すると three-cornered hat の推定値はその組を小さく推し，共有した誤差を薄められない（E72 の命題 12）．本研究の不等式は，この場合に保証が弱まることを ρ の大きさで示すので，先例の主張を条件つきで正す材料になる．

### Gaps
- Gruber ほか 2017（TGRS）と Yilmaz ほか 2012（WRR）の本文は読んでいない．合併の誤差の分散の式や，最良の入力との比較の定理が本文にある可能性は残る．論文で引く前に本文を確かめる
- 降水量・蒸発散量の three-cornered hat による合併（HESS 2018，Water 2024 ほか）は検索結果の要約しか見ていない．相関を含む共分散行列の逆行列で重みを付ける版（一般化 three-cornered hat）が，相関の上限で合併の誤差を抑えているかは確かめていない
- 予測の合併の文献（Bates & Granger 1969，Operational Research Quarterly 20(4):451–468．書誌 ○ [DOI](https://doi.org/10.2307/3008764)．推した重みが同じ重みに負けやすい「forecast combination puzzle」の Smith & Wallis 2009，Oxford Bulletin of Economics and Statistics 71(3):331–355．書誌 ○ [DOI](https://doi.org/10.1111/j.1468-0084.2008.00541.x)）は，書誌だけ確かめ，中身は記憶である．推定の誤差 ε を通した ((1 + ε)/(1 − ε))² の形の結果がこの分野にある可能性は残る
- Kuzborskij & Orabona の多ソースの続編（2018〜2026 年）で，ソースの重みをラベル無しで決める論文は検索では見つからなかった．dblp と OpenReview は bot 対策で使えず，網羅はしていない
- 三角形の不等式に近い「Ranking LLMs without ground truth」（三つ組で互いを評価する）などの最近の LLM の評価の論文は，検索結果に題が出ただけで確かめていない

---

## C2：K 本のラベルで決まらない方向の集合が N_K = null(H_K) で，K とともに縮むという主張は出版されているか

### Takeaway
仕組みそのものは古典である．情報行列が特異な向きにはパラメータが局所的に決まらないことは Rothenberg 1971 が一般の模型で示し，ターゲットのラベルが無い DA が仮定なしには保証できないことは Ben-David ほか 2010 が示した．collocation の側では，差しか使わない方法には誤差の共分散の M 次元の見えない部分空間があり，「独立で雑音の無い 5 つのセンサ」と「完全に相関した 5 つのセンサ」を仮定なしには区別できないことを Zwieback ほか 2012 が書いた．一方で，決められない集合が K 本のラベルの曲率の零空間 null(H_K) とちょうど一致して K とともに縮むこと，それを「ラベル無しで決める重みの規則は最良の見方に競合できない」という重みの不可能性として書いた文献は見つからなかった．判定は「部分的な重なり」である．

### Cited Findings
- Rothenberg 1971（Econometrica 39(3):577–591．書誌 ○）の抄録は，弱い正則条件のもとで「local identifiability of the unknown parameter vector is equivalent to nonsingularity of the information matrix」とする（○ 抄録．本文は JSTOR で読めなかった）— [DOI](https://doi.org/10.2307/1913267)；[Econometric Society の頁](https://www.econometricsociety.org/publications/econometrica/1971/05/01/identification-parametric-models)
- Ben-David, Lu, Luu, Pál 2010（AISTATS，PMLR 9:129–136．○ 抄録）は，(i) ラベル無しの分布の近さ，(ii) 両方の領域で誤りの小さい分類器の存在，などの仮定を調べ，「Our negative results hold with respect to any domain adaptation learning algorithm, as long as it does not have access to target labeled examples」と書く．ターゲットのラベルが少しある場合は扱っていない（抄録の範囲）— [PMLR](https://proceedings.mlr.press/v9/david10a.html)
- Zwieback ほか 2012 の 4.3 節は，差の括弧で表せる誤差の共分散の情報が M(M−1)/2 次元しかなく，残りの M 次元の部分空間 K_M は「invisible to the collocation technique」と示す．例として「5 independent sensors ... none of them has any measurement noise」と「5 sensors with equal error variance that are all perfectly correlated」を挙げ，「there is no way to tell the difference between the two cases without additional assumptions」と書く．そのうえで「In practice it will be easier to postulate that certain covariances vanish」とし，標準の TC の 3 つの仮定は「exactly those that all correlations are 0」だと述べる（◎ 4.3 節）— [NPG 19:69–80](https://npg.copernicus.org/articles/19/69/2012/)
- Ha & Chen 2025（arXiv 2507.14661．`theory.md` で確かめ済み）は，ラベル無しの共変量が微調整の部分空間を決め，ラベル付きのターゲットがその部分空間の中の補正を推すと書き，必要なラベルの数が介入の次元に比例すると示す．ラベルの数に応じて決まる方向が増える構造は C2 に近いが，部分空間は介入の構造で決まり，K 本の曲率の零空間ではない — [arXiv](https://arxiv.org/abs/2507.14661)
- Zhang & Oles 2000（ICML 2000 "A probability analysis on the value of unlabeled data for classification problems"）は，ラベル無しのデータが分類の模型のパラメータの推定にどれだけ役立つかを Fisher 情報で調べたとされる（△ 書誌は HKUST の研究者頁で確かめた．中身は読んでいない）— [HKUST](https://researchportal.hkust.edu.hk/en/publications/a-probability-analysis-on-the-value-of-unlabeled-data-for-classif/)

### Inferences
- **C2 は Rothenberg 1971 の系として書くのが正直である．**線形の softmax の頭では，K 本のラベルの条件付き尤度の Fisher 情報は，ラベルの値に依らず特徴と確率だけで決まり，2 次近似の H_K と一致する．その零空間 N_K は「K 本のロジットがクラスに共通の定数だけ変わる向き」である．Rothenberg の結果から，K 本のラベルの分布は N_K の向きの θ* を決めない．見方 x_t，教師モデルの出力，特徴はそもそも θ* を含まないので，観測のすべてが N_K の向きで変わらない．
- **新しい部分は，不可能性を「重みの規則」の言葉で書いたことと，K に応じた集合の形である．**(1) N_K の向きに θ* をずらすと E_t = e_tᵀM_K e_t が変わり最適な重みが変わるので，ラベル無しで決めるどの重みの規則も 2 つの場合の両方では最適になれない．(2) N_K はラベル付きの集合が入れ子なら K とともに縮み，K = 0 では全空間になる．(3) 「ターゲットのラベルが無い DA の不可能性」（Ben-David ほか 2010）を，ラベルが K 本ある場合のパラメータの版に広げている．この 3 点は調べた範囲に先例が無かった．
- **集合の定義を正確にする必要がある．**u ∈ N_K なら (I + H_K)⁻¹u = u なので uᵀM_K u = uᵀFu になる．F u = 0 の向き（ターゲットの母集団の全体でロジットがクラスに共通の定数だけ変わる向き．softmax の冗長な向き）では，θ* をずらしてもリスクも最適な重みも変わらない．したがって「最適な重みを変える」と言える集合は N_K のうち F の零空間に入らない部分であり，「決められない集合がちょうど N_K」と書くときは softmax の冗長な向きを除いた商空間で書く．
- **「入れ子なら縮む」の条件を書く．**K 本の標本と K′ > K 本の標本が別々に抜かれると，null(H_{K′}) ⊂ null(H_K) は一般には成り立たない．成り立つのはラベル付きの集合が入れ子のときである．入れ子でなくても，次元は確率 1 で K とともに増えない（特徴が一般の位置にある場合）と書ける見込みだが，本調査では確かめていない．
- **Zwieback ほか 2012 の例は，C2 の「見方の側」の版である．**θ* の側の不可能性（C2）とは別に，誤差の共分散の側にも差だけでは見えない部分空間がある．共通の誤差（E72 の命題 11）は Zwieback の Σ A_m にあたる．論文で命題 11 と 13 を書くときは，この 2 つを並べて引くと位置づけがはっきりする．

### Gaps
- Rothenberg 1971 の本文（定理の正確な正則条件）は読んでいない．論文で引く前に定理 1 の条件を確かめる
- 半教師ありの DA の理論で，ターゲットのラベルの数とともに「決められない集合」が縮むと書いた文献（Hanneke & Kpotufe 2019，Kpotufe & Martinet 2021 などは速さの結果）は，本調査では確かめていない．2025〜2026 年の論文は網羅していない
- 線形回帰の「推定可能な関数は計画行列の行空間に入るもの」という古典（Bose，Searle の線形模型の教科書）は一次の資料を確かめていない．論文で使うなら Rothenberg で足りる
- Zhang & Oles 2000 と Seeger の技術報告（判別型の模型ではラベル無しのデータがパラメータの情報を持たないという議論）は中身を確かめていない

---

## C3：見方が 4 つ以上なら独立の前提の崩れをラベル無しで一部確かめられ，3 つでは確かめられないという主張は出版されているか

### Takeaway
中身はほぼ出版済みである．地球科学では，4 つ以上のデータで collocation の方程式が未知数より多くなり，最小 2 乗で解けること（Pierdicca ほか 2015），誤差の相関の一部を推せること（Zwieback ほか 2012，Gruber ほか 2016），相関のある組を自動で特定できること（Pierdicca ほか 2017）が出版されている．機械学習では，Tenzer ほか 2022 が回帰の予測の対ごとの差の共分散 Cov(Δ_ij, Δ_kl) = S_ik − S_il − S_jk + S_jl（4 つの異なる添字が要る）で，独立の前提をラベル無しで確かめられると書き，相関のある組が少なければ一意に復元できる条件を定理にしている．3 つのデータでは方程式と未知数が同じ数で確かめられないことも，TC と Platanios ほか 2014 の数え方から直接出る．残る違いは，対象がパラメータのベクトルで内積が M_K であることだけである．論文ではこれを新しさとして主張せず，先例を引いて使うのがよい．

### Cited Findings
- Zwieback ほか 2012 は，誤差の共分散行列のうち差で見える部分が M(M−1)/2 次元で，その基底が [i,j;i,j]（各組の差の 2 乗）であると示す（◎ 4.3.1 節と 4.3 節の結び）．6.3 節は「the possibility of estimating cross-covariances in quadruple collocation, i.e. when M = 4」を扱う（◎ 5・6 節の前置き．6.3 節の結果は 2 段組の文字化けで読めていない）— [NPG 19:69–80](https://npg.copernicus.org/articles/19/69/2012/)
- Pierdicca, Fascetti, Pulvirenti, Crapolicchio, Muñoz-Sabater 2015（IEEE GRSL 12(8):1595–1599．書誌 ○）について，Gruber の博士論文は「extend TC analysis with a fourth data set and to solve this quadruple collocation (QC) problem as an overdetermined system of three possible triplets in a least-squares sense. This minimizes the uncertainty of the individual error estimates, but still requires uncorrelated errors between all four data sets」と要約する（◎ Gruber の博士論文の 4.1 節）— [DOI](https://doi.org/10.1109/lgrs.2015.2414654)；[Gruber の博士論文（TU Wien）](https://repositum.tuwien.at/handle/20.500.12708/2438)
- Gruber, Su, Crow, Zwieback, Dorigo, Wagner 2016（JGR Atmospheres 121(3):1208–1219．書誌 ○）は，TC を任意の数のデータ（extended collocation，EC）に広げ，いくつかの組で誤差の相関が 0 という仮定を外し，その相関を推す．推せる相関の数は「limited by the overall number of data sets used and by their underlying error cross-correlation structure: Each member of the data set pairs with assumed non-zero error cross correlation must also be a member of at least one data set triplet with fully independent errors. Remaining degrees of freedom are used to solve the collocation system of equations in a least-squares sense」（◎ 博士論文の 1.2.3 節と 4.2.4 節．4.2.4 節は「A⊺A has to have full rank」と書く）．実データでは，AMSR-E の 2 つの製品の間と，予想に反して ASCAT と AMSR-E の間に 0 でない相関を見つけた — [DOI](https://doi.org/10.1002/2015jd024027)；[博士論文の PDF](https://repositum.tuwien.at/bitstream/20.500.12708/2438/2/Gruber%20Alexander%20-%202016%20-%20Estimation%20of%20error%20structures%20in%20remotely%20sensed%20soil...pdf)
- Pierdicca, Fascetti, Pulvirenti, Crapolicchio 2017（IEEE JSTARS 10(10):4522–4530．書誌 ○）の抄録は，extended quadruple collocation（E-QC）を「in order to consider the possibility of a cross correlation between product errors, identifying automatically the couple of error cross-correlated systems」と書き，SMOS・ASCAT・SMAP・ERA-Interim・ISMN に当てて衛星の製品の間の誤差の相関を見つけたとする（△ 検索結果に出た抄録の抜き書き．Sapienza の IRIS の頁）．4 つのデータで「どの組が相関しているか」をラベル（真の値）無しで特定するので，C3 に最も近い — [DOI](https://doi.org/10.1109/jstars.2017.2714025)；[IRIS](https://iris.uniroma1.it/handle/11573/1090658)
- Tenzer ほか 2022 の 5 節は，「directly estimating S_{i,j} = E[h_i(X)h_j(X)] is impossible. Nonetheless ... it is possible to assess the validity of this assumption in an unsupervised manner. The idea is to look at the pairwise differences between experts」と書き，Δ_{i,j} = f_i(X) − f_j(X) と ρ_{i,j,k,l} = Cov(Δ_{i,j}, Δ_{k,l}) = S_ik − S_il − S_jk + S_jl（1 ≤ i ≠ j ≠ k ≠ l ≤ m）を定義する．誤差の相関がまばらなら，ほとんどの四つ組で 0 になる（◎ 5 節）．定理 2 は，m ≥ 5 なら式 (9)（相関の行列 S の非 0 の数を最小にする分解）が一意に解ける必要十分条件を ∥vec(S)∥₀ < (m − 1)/2 とする（◎ 4 節と付録の Lemma 3）— [PMLR 151](https://proceedings.mlr.press/v151/tenzer22a.html)
- Jaffe, Fetaya, Nadler, Jiang, Kluger 2016（AISTATS，JMLR W&CP 51）は，分類器の条件付き独立の「strong violations」をラベル無しで見つけ（「how to detect, in an unsupervised manner, strong violations of conditional independence」），強く依存する分類器の組を共分散行列の低ランクの構造から見つける（◎ 抄録と 1 節）— [PMLR の PDF](https://proceedings.mlr.press/v51/jaffe16.pdf)
- Platanios ほか 2014 は，独立な誤りをする 3 つの関数では未知数 3 つに方程式 3 つで，一致率から誤り率がちょうど決まると書く（◎ 3.1.1 節）．方程式が余らないので，独立の前提を一致率だけでは確かめられない — [UAI 2014 の PDF](http://auai.org/uai2014/proceedings/individuals/313.pdf)
- 因子分析では，1 因子の模型で指標が 4 つ以上あると，消える四つ組の差（tetrad）という確かめられる制約が出る．Bollen & Ting 1993 がこれを検定にした（書誌 ○．Sociological Methodology 23:147–．中身は △ 記憶）— [DOI](https://doi.org/10.2307/271009)

### Inferences
- **数え方は TC・1 因子の模型と同じである．**V 個の見方では，差で見える誤差のグラム行列の情報は V(V−1)/2 個（Zwieback ほか 2012）で，独立の模型の未知数は V 個なので，余る方程式の数は V(V−3)/2 である．V = 3 で 0，V = 4 で 2，V = 5 で 5 になる．これは 1 因子の模型の自由度 p(p−3)/2 と同じ式である．E72 の命題 12 の「V = 3 では残差が常に 0，V ≥ 4 で一部の組だけの共通の誤差が残差に出る」は，この数え方の言い直しである．
- **対ごとの差の 2 乗 D_st だけで，ラベル無しの情報はすべて使っている．**Zwieback ほか 2012 によれば，[i,j;i,j] の括弧（D_st にあたる）が見える部分空間の基底になる．Tenzer ほか 2022 の四つ組の共分散 ρ_ijkl も，D_st の線形結合で書ける（ρ_ijkl = ½(D_il + D_jk − D_ik − D_jl)）．したがって E72 の最小 2 乗の残差は，四つ組の検査と同じ情報を持ち，それより多くも少なくもない．
- **残差で見つけられない相関の形がある．**C_st = b_s + b_t の形（設計行列 A の列の空間に入る相関．共通の誤差は b_t がすべて等しい場合）は Ê に吸収され，残差に出ない．これは Zwieback の見えない部分空間 K_M にあたる．「一部確かめられる（partly testable）」という C3 の書き方は正しく，論文でもこの限界を書く．
- **C3 を新しさとして主張しない方がよい．**本研究の版の違いは，見方がモデルのパラメータで，内積が M_K であることだけである．論文では「Zwieback ほか 2012，Gruber ほか 2016，Pierdicca ほか 2017，Tenzer ほか 2022 と同じ数え方で」と書いて使う．E71 の AV の世界（V = 3）で確かめられないことは，この先例の数え方で説明できる．
- **相関のある組を推す方法も先例から借りられる．**V ≥ 4 なら，Gruber ほか 2016 の EC（相関のある組の各要素が独立な三つ組に 1 つ以上入れば推せる）や Tenzer ほか 2022 の定理 2（m ≥ 5，相関のある組が (m − 1)/2 未満）をそのまま使える．E72 の「前提が崩れる場合の扱い 2」（残差が大きい区間で B の中心に戻す）より踏み込んだ扱いとして，相関を推して重みに入れる案がありうる．

### Gaps
- Pierdicca ほか 2015・2017 の本文は読んでいない．E-QC が「どの組か」を特定する手順（残差の比較か，組ごとの当てはめの比較か）と，4 つで特定できる相関の組の数の条件は確かめていない
- Zwieback ほか 2012 の 6.3 節（M = 4 での共分散の推定）の結果は，PDF の 2 段組の文字化けで読めていない．`pdftotext -layout` で読み直す必要がある
- 計時の分野で，相関のある N-cornered hat の版（Premoli & Tavella 1993，IEEE TIM 42(1):7–13．書誌 ○ [DOI](https://doi.org/10.1109/19.206671)．Torcaso ほか，Galindo & Palacio）が，N ≥ 4 での最小 2 乗の残差を独立の検定に使っているかは確かめていない．Premoli & Tavella は相関を含む共分散行列を正定値の制約で決める方法とされる（△ 記憶）
- TC をモデルのパラメータのベクトルに当てた例は見つからなかった．機械学習での使い方は，予測（Platanios ほか 2014，Tenzer ほか 2022）か，機械学習で作った地球科学の製品の評価に限られていた（後者は個別に確かめていない）

### BibTeX（部分的な重なり以上の文献）

```bibtex
@article{stoffelen1998toward,
  author  = {Stoffelen, Ad},
  title   = {Toward the true near-surface wind speed: Error modeling and calibration using triple collocation},
  journal = {Journal of Geophysical Research: Oceans},
  volume  = {103},
  number  = {C4},
  pages   = {7755--7766},
  year    = {1998},
  doi     = {10.1029/97JC03180}
}

@inproceedings{gray1974method,
  author    = {Gray, J. E. and Allan, D. W.},
  title     = {A Method for Estimating the Frequency Stability of an Individual Oscillator},
  booktitle = {28th Annual Symposium on Frequency Control},
  pages     = {243--246},
  year      = {1974},
  doi       = {10.1109/FREQ.1974.200027}
}

@article{zwieback2012structural,
  author  = {Zwieback, S. and Scipal, K. and Dorigo, W. and Wagner, W.},
  title   = {Structural and statistical properties of the collocation technique for error characterization},
  journal = {Nonlinear Processes in Geophysics},
  volume  = {19},
  number  = {1},
  pages   = {69--80},
  year    = {2012},
  doi     = {10.5194/npg-19-69-2012}
}

@article{yilmaz2012objective,
  author  = {Yilmaz, M. T. and Crow, W. T. and Anderson, M. C. and Hain, C.},
  title   = {An objective methodology for merging satellite- and model-based soil moisture products},
  journal = {Water Resources Research},
  volume  = {48},
  number  = {11},
  year    = {2012},
  doi     = {10.1029/2011WR011682},
  note    = {Article number not verified}
}

@article{pierdicca2015quadruple,
  author  = {Pierdicca, Nazzareno and Fascetti, Fabio and Pulvirenti, Luca and Crapolicchio, Raffaele and Mu{\~n}oz-Sabater, Joaquin},
  title   = {Quadruple Collocation Analysis for Soil Moisture Product Assessment},
  journal = {IEEE Geoscience and Remote Sensing Letters},
  volume  = {12},
  number  = {8},
  pages   = {1595--1599},
  year    = {2015},
  doi     = {10.1109/LGRS.2015.2414654}
}

@article{gruber2016estimating,
  author  = {Gruber, A. and Su, C.-H. and Crow, W. T. and Zwieback, S. and Dorigo, W. A. and Wagner, W.},
  title   = {Estimating error cross-correlations in soil moisture data sets using extended collocation analysis},
  journal = {Journal of Geophysical Research: Atmospheres},
  volume  = {121},
  number  = {3},
  pages   = {1208--1219},
  year    = {2016},
  doi     = {10.1002/2015JD024027}
}

@article{pierdicca2017error,
  author  = {Pierdicca, Nazzareno and Fascetti, Fabio and Pulvirenti, Luca and Crapolicchio, Raffaele},
  title   = {Error Characterization of Soil Moisture Satellite Products: Retrieving Error Cross-Correlation Through Extended Quadruple Collocation},
  journal = {IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  volume  = {10},
  number  = {10},
  pages   = {4522--4530},
  year    = {2017},
  doi     = {10.1109/JSTARS.2017.2714025}
}

@article{gruber2017triple,
  author  = {Gruber, Alexander and Dorigo, Wouter Arnoud and Crow, Wade and Wagner, Wolfgang},
  title   = {Triple Collocation-Based Merging of Satellite Soil Moisture Retrievals},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  volume  = {55},
  number  = {12},
  pages   = {6780--6792},
  year    = {2017},
  doi     = {10.1109/TGRS.2017.2734070}
}

@article{anthes2018estimating,
  author  = {Anthes, Richard and Rieckh, Therese},
  title   = {Estimating observation and model error variances using multiple data sets},
  journal = {Atmospheric Measurement Techniques},
  volume  = {11},
  pages   = {4239--4260},
  year    = {2018},
  doi     = {10.5194/amt-11-4239-2018}
}

@article{gruber2019evolution,
  author  = {Gruber, Alexander and Scanlon, Tracy and van der Schalie, Robin and Wagner, Wolfgang and Dorigo, Wouter},
  title   = {Evolution of the {ESA CCI} Soil Moisture climate data records and their underlying merging methodology},
  journal = {Earth System Science Data},
  volume  = {11},
  pages   = {717--739},
  year    = {2019},
  doi     = {10.5194/essd-11-717-2019}
}

@inproceedings{platanios2014estimating,
  author    = {Platanios, Emmanouil Antonios and Blum, Avrim and Mitchell, Tom},
  title     = {Estimating Accuracy from Unlabeled Data},
  booktitle = {Proceedings of the Thirtieth Conference on Uncertainty in Artificial Intelligence (UAI)},
  year      = {2014},
  note      = {Pages not verified; PDF at auai.org/uai2014/proceedings/individuals/313.pdf}
}

@inproceedings{jaffe2016unsupervised,
  author    = {Jaffe, Ariel and Fetaya, Ethan and Nadler, Boaz and Jiang, Tingting and Kluger, Yuval},
  title     = {Unsupervised Ensemble Learning with Dependent Classifiers},
  booktitle = {Proceedings of the 19th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  series    = {JMLR Workshop and Conference Proceedings},
  volume    = {51},
  year      = {2016},
  note      = {Pages not verified}
}

@inproceedings{tenzer2022crowdsourcing,
  author    = {Tenzer, Yaniv and Dror, Omer and Nadler, Boaz and Bilal, Erhan and Kluger, Yuval},
  title     = {Crowdsourcing Regression: A Spectral Approach},
  booktitle = {Proceedings of the 25th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {151},
  year      = {2022},
  note      = {Pages not verified}
}

@inproceedings{ahmed2021unsupervised,
  author    = {Ahmed, Sk Miraj and Raychaudhuri, Dripta S. and Paul, Sujoy and Oymak, Samet and Roy-Chowdhury, Amit K.},
  title     = {Unsupervised Multi-source Domain Adaptation Without Access to Source Data},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {10098--10107},
  year      = {2021},
  doi       = {10.1109/CVPR46437.2021.00997}
}

@article{tommasi2014learning,
  author  = {Tommasi, Tatiana and Orabona, Francesco and Caputo, Barbara},
  title   = {Learning Categories From Few Examples With Multi Model Knowledge Transfer},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume  = {36},
  number  = {5},
  pages   = {928--941},
  year    = {2014},
  doi     = {10.1109/TPAMI.2013.197}
}

@article{rothenberg1971identification,
  author  = {Rothenberg, Thomas J.},
  title   = {Identification in Parametric Models},
  journal = {Econometrica},
  volume  = {39},
  number  = {3},
  pages   = {577--591},
  year    = {1971},
  doi     = {10.2307/1913267}
}

@inproceedings{bendavid2010impossibility,
  author    = {Ben-David, Shai and Lu, Tyler and Luu, Teresa and P{\'a}l, D{\'a}vid},
  title     = {Impossibility Theorems for Domain Adaptation},
  booktitle = {Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics (AISTATS)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {9},
  pages     = {129--136},
  year      = {2010}
}
```
