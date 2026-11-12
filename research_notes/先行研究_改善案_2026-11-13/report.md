# 先行研究との差分：改善案の候補 1（疑似ラベルの取り違えを K 本で直してから渡す）（2026-11-13）

## 要約

- **判定：手法としての新しさは主張できない．**取り違えの表を少数の信頼できるラベルで推し，雑音のあるラベルを補正する手順は，Gold Loss Correction（GLC，Hendrycks ほか 2018）そのものである．候補 1 の表の推し方（本当の感情ごとに疑似ラベルの確率を平均する）は GLC の推し方と同じ式で，実験 12 の下調べで測った「行列 A」とも同じ量である．
- 最も近い先行研究は次の 5 本である．GLC（Hendrycks ほか 2018），ラベル無しの例での蒸留で教師の誤りを小さな検証用の正解で推して補正する系統（Iliopoulos ほか 2022，SLaM：Kontonis ほか 2023），教師と生徒が同じラベルを使うときの交差の当てはめ（Dao ほか 2021），ラベルシフトの BBSE（Lipton ほか 2018），顔から声への感情の蒸留（Albanie ほか 2018）．
- **候補 1 が書いた「ベイズの規則で置き換える」補正は，GLC の仮定の下では誤りを打ち消さない（導出．4 節）．**この置き換えは表を 2 回掛けることになり，感情ごとに強さの違うラベル平滑化（正解の確率を周りの感情へ広げる操作）に当たる．取り違えを打ち消すのは，GLC の前向き補正か，表の逆行列を掛ける後ろ向き補正である．
- 狙うべき差分は 4 つある．(1) クラスあたり 1〜25 本という GLC が試していない少なさで，補正の得が K と表の特異値の小ささでどう決まるか．(2) 生徒の特徴でまねできない分には補正が届かないという予言を，3 つの条件の 3 つ目と結びつけて確かめること．(3) 感情の対ごとに，どの取り違えが直ったか．(4) 補正の得と平滑化の得を対照で分けること．(1) と (3) は保存済みの出力から出せる．
- 論文の最後の節には「分析から導いた既知の補正（GLC）を当て，分析の予言（効くのは声だけ，MOSEI では推定が崩れる）を確かめた」という形で書くことを推す（5 節に文案）．

---

## 0. 読んだもの・確かめ方

- 入力：`research_notes/改善案_2026-11-13/候補.md`，`review_ml.md`，`過去の試み.md`，`DAI/docs/主張_採用した説明_2026-11-13.md`，`DAI_paper/CLAUDE.md`，`DAI_paper/inventory.md`，`docs/related-work.md`（BBSE，Lipton，Garg の節），`DAI/docs/実験12_下調べ_結果の文の下書き_2026-11-09.md` の 7 節．
- 過去の文献調査：`research_notes/疑似ラベル 情報の持ち込み方 先行研究/3_少数のラベルで疑似ラベルを直す.md`（2026-11-08．GLC，Patrini，Veit，BBSE，L2RW，MW-Net，MLC，MPL を原文で確かめ済み），同 `5_外部の知識と弱い情報源.md`（Snorkel 系），`research_notes/有名な論文_2026-11-07/蒸留.md`（C2KD，Albanie），`research_notes/先行研究_ラベルの代わりとしての他モダリティ_2026-11-03/report_B_実証.md`．
- 今回新たに PDF を取得して本文を読んだもの：Dao ほか 2021，Kontonis ほか 2023（SLaM），Iliopoulos ほか 2022，Diamant ほか 2024（DCPL），Lukasik ほか 2020．GLC の 3 節と 4.4 節は今回も本文で読み直した．
- 検証の水準：◎ 原文で該当箇所を確かめた，○ 抄録か信頼できる二次資料，△ 検索の断片だけ．
- 計算はしていない．保存済みの出力はファイルの中の項目の名前だけを確かめ，値は見ていない．

---

## 1. 研究目標を検索できる問いに分ける

候補 1 は「ほかのモダリティを含むモデルの疑似ラベルが，本当の感情をどの感情と取り違えるか（4×4 の表）をターゲットの K 本のラベルで推し，その取り違えを差し引いてから疑似ラベルとして渡し，線形の分類器を学び直す」手法である．これを次の 5 つの問いに分けて調べた．

| 問い | 中身 |
|---|---|
| Q1 | 少数の信頼できるラベルで雑音の遷移行列（本当のクラス i が観測ラベル j になる確率の表．以下「取り違えの表」）を推し，雑音のあるラベルや疑似ラベルを補正する手法はあるか．補正の式は何か |
| Q2 | 教師の出力（疑似ラベル）を雑音のあるラベルとみなし，少数の正解で教師の誤り方を推してから生徒に渡す蒸留はあるか．教師が信頼ラベルで学んでいる場合はどう扱うか |
| Q3 | ラベル無しの多数の判断と少数の正解から取り違えの表を推す系統（Dawid–Skene に正解を足す系統，Confident Learning，弱い教師）は，候補 1 と何が違うか |
| Q4 | クロスモーダル蒸留やマルチモーダルの疑似ラベル（感情認識を含む）で，教師の取り違えを補正してから渡す研究はあるか |
| Q5 | 領域適応で，ソースのモデルの混同行列をターゲットのラベルで推して直す研究はあるか．ラベルシフトの BBSE とは何が違うか |

---

## 2. 系統ごとの先行研究と，候補 1 との差分

### 2.1 少数の信頼ラベルで取り違えの表を推して補正する系統（Q1）

**Hendrycks, Mazeika, Wilson, Gimpel 2018．Gold Loss Correction（GLC）．NeurIPS 2018 ◎**
- 手順（3 節）：雑音のあるデータだけで分類器 p̂(ỹ|x) を学ぶ．信頼できるデータのうち本当のクラスが i の例の集合を Aᵢ として，次の式で表を推す．
  ```
  Ĉᵢⱼ = (1/|Aᵢ|) Σ_{x∈Aᵢ} p̂(ỹ = j | x)      （≈ p(ỹ = j | y = i)）
  ```
  そのうえで最終の分類器 g を，信頼データでは普通の交差エントロピー，雑音のあるデータでは ℓ(Ĉᵀ g(x), ỹ)（前向き補正）で学ぶ．
- 仮定："Assuming conditional independence of ỹ and y given x"（3 節）．入力 x が決まれば，観測ラベルは本当のラベルに依らないという仮定である．
- 弱い分類器の出力を雑音のあるラベルにした実験（4.4 節）がある．著者はこの設定について "This setting exactly satisfies the conditional independence assumption on label noise used for our Ĉ estimate, because the weak classifier does not take the true label as input" と書いている．
- 信頼データの量：最小は CIFAR-100 のクラスあたり 25 本（Table 1）と，Clothing1M の 100 枚／14 クラス（約 7 本／クラス．Figure 3 の図だけで数値の表は無い）である．
- 候補 1 との関係：
  - **候補 1 の表の推し方は GLC の Ĉ と同じ式である．**候補 1 では疑似ラベルの確率 q(x) がそのまま p̂(ỹ|x) に当たり，GLC の 4.4 節の「弱い分類器の出力」と同じ構造になる．実験 12 の下調べの「行列 A（本当の感情ごとの出所の平均の確率）」もこれと同じ量である．
  - **違いの 1 つ目は補正の掛け方である．**GLC は生徒の出力に Ĉᵀ を掛けて疑似ラベルに合わせる（前向き補正）．候補 1 は疑似ラベルの側を「疑似ラベルが j のとき本当が i である確率」で置き換える．4 節で示すとおり，この置き換えは GLC の仮定の下では誤りを打ち消さない．
  - **違いの 2 つ目は信頼データの使い方である．**GLC は p̂(ỹ|x) を雑音のあるデータだけで学び，信頼データを使わない．候補 1 の疑似ラベルのモデルは K 本で学んでいるので，同じ分離を作るには K 本を分けて作り直す必要がある（検査役の [M1]）．
  - **違いの 3 つ目は量である．**GLC はクラスあたり 25 本以上でしか数値の表を出していない．候補 1 は K = 1〜25 である．
- URL：https://arxiv.org/abs/1802.05300

**Patrini, Rozza, Krishna Menon, Nock, Qu 2017．前向き補正と後ろ向き補正．CVPR 2017 ◎（過去の調査）**
- 後ろ向き補正は損失に T⁻¹ を掛けて不偏にする（Theorem 1）．前向き補正は予測に Tᵀ を掛けて雑音のあるラベルに合わせる（Theorem 2）．どちらも T が正則であることを仮定する．
- きれいなラベルがあれば T を直接作れることに，1 文で触れている（4.3 節）："If we know T, for example by cleaning manually a subset of training data, we can train with ℓ← or ℓ→."
- 著者が書いた限界：クラスあたり 500 枚でも，ラベル無しで T を推すと CIFAR-100 で補正の得の大半が消えた（6 節）．
- 候補 1 との関係：候補 1 の補正の式を「後ろ向き」か「前向き」に置き換えれば，この論文の型に入る．K = 5 の表の 1 成分の標準誤差は約 0.19 で（検査役の [M2]），MOSEI の表の最小の特異値 0.04〜0.08（実験 12 の 7 節）より大きい．このため，後ろ向き補正（逆行列を掛ける）は MOSEI では大きく揺れると推測する．
- URL：https://arxiv.org/abs/1609.03683

**Lukasik, Bhojanapalli, Menon, Kumar 2020．ラベル平滑化は雑音を和らげるか．ICML 2020 ◎**
- 表 1 で，ラベル平滑化と後ろ向き補正を「単位行列と全要素 1 の行列の組み合わせ」として並べた．平滑化は (1 − α)·I + (α/L)·J のように足す形，後ろ向き補正は I から J を引く形である．著者の文："Both methods combine an identity matrix with an all-ones matrix; the striking difference, however, is that this combination is via addition in one, but subtraction in the other."（3.4 節）
- 雑音のあるデータから蒸留するときは，教師を平滑化すると生徒が良くなると報告した："when distilling models from noisy data, label smoothing of the teacher is beneficial"（抄録）．
- 候補 1 との関係：4 節で示すとおり，候補 1 の置き換えは「足す側」（平滑化の側）に入る．得が出ても，取り違えを打ち消した分か平滑化の分かを分けられない．これは検査役の [M4] を文献の側から裏づける．
- URL：https://arxiv.org/abs/2003.02819

**Veit ほか 2017（ラベルの洗浄網．CVPR 2017）◎（過去の調査）**：同じ例に雑音のあるラベルと人のラベルの両方がある約 4 万枚の集合で，「雑音のあるラベルと特徴から正しいラベルへの差」を学ぶ．人のラベルが無ければ，雑音のあるラベルを変えない形である．候補 1 は表の 12 個の数だけを推すので，学ぶ量が桁違いに少ない．https://arxiv.org/abs/1701.01619

**Kye ほか（FasTEN．arXiv 2111.14932）○**：小さなきれいなデータで遷移行列を学習の各回に推し直しながら，雑音のあるラベルを付け直す．抄録だけを読んだ．https://arxiv.org/abs/2111.14932

**Yang ほか 2022（ベイズ最適ラベルからの遷移行列．ICML 2022）○**：本当のラベルからではなくベイズ最適のラベルから雑音のあるラベルへの遷移を例ごとに網で推し，前向き補正に使う．候補 1 が使う「疑似ラベルが j のとき本当が i」という向きの表とは別のものである．https://proceedings.mlr.press/v162/yang22p.html

### 2.2 教師の誤りを少数の正解で推してから生徒に渡す蒸留（Q2）

**Iliopoulos, Kontonis, Baykal, Menghani, Trinh, Vee 2022．Weighted Distillation with Unlabeled Examples．NeurIPS 2022（掲載先は SLaM の文献表で確かめた）◎**
- 設定：教師は少数のラベル付きデータで学び，ラベル無しの多数のデータに疑似ラベルを付ける．生徒はその疑似ラベルで学ぶ．これに加えて，小さなきれいな検証用データ Sv を置く（2.1 節）．
- 手法：Sv を使い，「教師の確信度と生徒の確信度」から「教師が誤る確率」と「損失のゆがみ」を推す写像を学ぶ．その推定で例ごとに損失の重みを付け，期待値の上できれいなラベルで学んだのと同じにする．
- 仮定："The only assumption we need to make about the validation set Sv is that it is not in the train set of the teacher model."（Remark 2.1）
- 候補 1 との関係：「少数の正解で教師の誤り方を推し，生徒の損失を直す」という骨組みは同じである．違いは 2 つある．Iliopoulos ほかは誤り方を「例ごとの正誤の確率」として推し，候補 1 は「感情の対ごとの取り違えの表」として推す．また，Iliopoulos ほかは検証データを教師の学習から外すことを必須にしている．
- URL：https://arxiv.org/abs/2210.06711

**Kontonis, Iliopoulos, Trinh, Baykal, Menghani, Vee 2023．SLaM: Student-Label Mixing．NeurIPS 2023 ◎**
- 設定は上と同じである．教師の誤りを「確率 α(x) で正しく，誤るときは教師の上位 k(x) 個のうち正解以外から一様に選ぶ」と模型化する（Definition 3.2）．
- α(x) は，教師が学びに使っていない小さな検証用データ（例：500 本）の上で，教師の確信度の差（1 位と 2 位の確率の差）から正誤への単調な回帰で推す（3 節）："we use a small validation dataset V with correct labels that the teacher has not seen during training."
- 生徒の予測の側に雑音を混ぜて，教師の疑似ラベルに合わせる（式 1・2．前向きの補正の一種）．
- 遷移行列を推す系統について，著者は次のように書いた："forward methods typically attempt to estimate a noise transition matrix ... which can be rather problematic when dealing with general, instance specific noise like in the case of distillation with unlabeled examples."（2 節）
- 候補 1 との関係：SLaM の著者は，ラベル無しの例での蒸留では教師の雑音が例ごとに違うので，クラスごとの遷移行列は合わないと述べている．候補 1 はクラスごとの表を使うので，限界の節にこの指摘を引き，「クラス条件つきの雑音の仮定」を書く必要がある．
- URL：https://arxiv.org/abs/2302.03806，https://proceedings.neurips.cc/paper_files/paper/2023/hash/d56b84c063265da949fe0feb815dcce8-Abstract-Conference.html

**Dao, Kamath, Syrgkanis, Mackey 2021．Knowledge Distillation as Semiparametric Inference．ICLR 2021（arXiv の版に掲載先の記載が無く，掲載先は確かめていない）◎**
- 教師と生徒が同じラベル付きデータを使うと，教師の過学習が生徒に移る．これを交差の当てはめで抑える（4.2 節）．交差の当てはめとは，データを B 個に分け，ある分割の点での教師の予測を，その分割を除いて学んだ教師で作る方法である．原文："the nuisance estimates (γ̂⁽ᵗ⁾, p̂⁽ᵗ⁾) that are evaluated on the data points in fold t when fitting the student in step 3, are estimated only using data points outside of Pt."
- 損失の補正は，各点の本当のラベル y を使って教師の偏りを 1 次で打ち消す（式 1）．
  ```
  ℓ_ortho(z; f(x), p̂(x)) = ℓ(z; f(x), p̂(x)) + ⟨y − p̂(x), E[∇_π ℓ(z; f(x), p̂(x)) | x]⟩
  ```
- 候補 1 との関係：検査役の [M1]（K 本の二重使用）の直し方は，この論文の交差の当てはめと同じ考え方である．ただし，この論文の損失の補正は，蒸留に使うすべての点に本当のラベルがあることを前提にする．候補 1 の候補（ラベル無しの発話）には本当のラベルが無いので，この補正は直接は使えない．
- URL：https://arxiv.org/abs/2104.09732

**Wen, Lai, Qian（Preparing Lessons．arXiv 1911.07471）○**：教師の誤った柔らかい目標を，その例の正解ラベルを見て直す（Probability Shift）．正解ラベルのある例にしか使えないので，候補にラベルが無い候補 1 の設定とは違う．https://arxiv.org/abs/1911.07471

### 2.3 ラベル無しの判断と少数の正解から表を推す系統（Q3）

**Dawid–Skene に正解を足す系統 ○〜△**
- Tang & Lease 2011（SIGIR の Crowdsourcing for IR の研究会）は，Dawid–Skene（注釈者ごとの混同行列を持つ模型を EM で推す方法）の EM の更新に，正解が分かっている例の本当のラベルを直接入れた．原文は読めず，Atarashi ほか（AAAI 2018）の要約と Carpenter の TREC 2011 の報告で確かめた（△）．
- Carpenter（TREC 2011）は，Dawid–Skene に既知の正解の種を足した半教師ありの模型を使った（△）．
- Shi ほか（SSME．arXiv 2501.11866）は，ラベル付きとラベル無しのデータから，複数の分類器の予測と本当のラベルの同時分布を半教師ありの混合模型で推し，分類器を評価する（○）．
- 候補 1 との関係：MER-DA では，声・顔・言葉の 3 つの判断から Dawid–Skene で混同の表を推す版を全ソース設定で試した．その事後確率は融合の分類器を下回った（`過去の試み.md` 1.1 節の表 6 行目）．Dawid–Skene に正解を足す系統は「複数の判断者のそれぞれの表」を推す．候補 1 は「掛け合わせた 1 つの疑似ラベルの表」を推す．候補 1 は，判断者が 1 人の Dawid–Skene に正解の K 本を足し，EM を回さずに正解だけで表を推したものに当たる．
- URL：https://bigml.cs.tsinghua.edu.cn/%7Ejun/pub/selective-crowds.pdf（Atarashi ほか．Tang & Lease の要約を含む），https://trec.nist.gov/pubs/trec20/papers/LingPipe.crowdsourcing.pdf，https://arxiv.org/abs/2501.11866

**Northcutt, Jiang, Chuang 2021．Confident Learning．JAIR 2021 ○**
- クラス条件つきの雑音を仮定する．モデルの予測確率と与えられたラベルから「与えられたラベルと本当のラベルの同時分布」を数え（confident joint），誤ったラベルの例を見つけて除く．
- 候補 1 との関係：Confident Learning は正解の集合を使わずに表を推し，その表を例を除くために使う．候補 1 は正解の K 本で表を推し，疑似ラベルを付け直すために使う．目的と使う情報の両方が違う．
- URL：https://arxiv.org/abs/1911.00068

**弱い教師（data programming，Snorkel）◎（過去の調査）と少数の正解 ○〜△**
- Snorkel 系のラベルモデルは，ラベル無しで情報源の正確さと相関を推す（`5_外部の知識と弱い情報源.md`）．
- 少数の正解をラベルモデルと一緒に使うと F1 で 2〜5 点上がるという報告がある（arXiv 2206.09496．検索の断片だけ．△）．
- Prediction-Powered Inference（Angelopoulos ほか 2023）は，少数の正解でモデルの予測の偏りを測り，平均や回帰係数の信頼区間を正しくする（○）．学習ではなく推定の手法である．
- 候補 1 との関係：「少数の正解で，モデルが付けたラベルの系統的な偏りを測って直す」という考えは，弱い教師と統計的推定の両方で使われている．
- URL：https://arxiv.org/abs/2206.09496，https://arxiv.org/abs/2301.09633

### 2.4 クロスモーダル蒸留・マルチモーダルの疑似ラベル（Q4）

**Albanie, Nagrani, Vedaldi, Zisserman 2018．顔から声への感情の蒸留．ACM MM 2018 ◎（過去の調査）**
- 顔の教師の出力を雑音のあるラベルとして扱い，声の生徒を学ぶ．教師と生徒の感情ごとの混同行列を図 5 に示し，"the degree to which there is a redundant emotional signal across modalities may differ across emotions" と書いた．
- 候補 1 との関係：教師の感情ごとの取り違えを測って示したが，補正には使っていない．ターゲットの正解は評価用の線形の写しにだけ使った．
- URL：https://arxiv.org/abs/1808.05561

**Huo ほか 2024．C2KD．CVPR 2024 ◎（過去の調査）**
- 教師と生徒の柔らかいラベルで「正解以外のクラスの順位」を比べ，順位相関が閾値以下の例を蒸留から外す．教師も生徒と双方向に更新する．
- 候補 1 との関係：C2KD は教師の誤りを例ごとに選んで外す方向で扱い，正解ラベルで表を推して直すことはしない．C2KD の設定は，全サンプルにラベルがある教師ありの設定である．
- URL：https://openaccess.thecvf.com/content/CVPR2024/html/Huo_C2KD_Bridging_the_Modality_Gap_for_Cross-Modal_Knowledge_Distillation_CVPR_2024_paper.html

**感情認識の半教師ありの疑似ラベル（MER2023・MER2024 の SEMI の部門）○**：クラスごとに確信度の上位 k 本を選ぶ方法，確信度 0.9 以上だけを使う方法，合議とラベルの修正を組み合わせる方法がある．検索の範囲では，混同行列で取り違えを直すものは無かった．https://web3.arxiv.org/pdf/2409.04447，https://doi.org/10.1145/3607865.3613182

**LanSER（Gong ほか 2023．Interspeech）◎（過去の調査）と Li ほか（arXiv 2409.16937）○**：言葉から作った弱いラベルで声のモデルを学ぶ．Li ほかは，音の側と大規模言語モデルの側の 2 つの疑似ラベルが一致するかで信頼度を分ける．どちらも少数の正解で表を推して直すことはしない．https://arxiv.org/abs/2309.03978，https://arxiv.org/abs/2409.16937

**Cheng ほか 2022（Joint-Modal Label Denoising．ECCV 2022）○**：音と映像の動画の解析で，あるモダリティに現れない事象のラベルを，モダリティ内とモダリティ間の損失の順位で見つけて外す．遷移行列は使わない．https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/785_ECCV_2022_paper.php

### 2.5 領域適応で混同行列を推して直す研究（Q5）

**Lipton, Wang, Smola 2018．BBSE．ICML 2018 ◎（過去の調査）**
- ラベルシフト（ラベルの割合だけが変わり，各クラスの見た目は変わらない）を仮定する．ソースの取っておいたラベル付きデータで予測器の混同行列 C を推し，ターゲットでの予測の平均 μ から割合の比 w = C⁻¹μ を推す．誤差の上界は C の最小の固有値の 2 乗に反比例する（Theorem 3）．
- 候補 1 との違い：
  - BBSE が直すのは感情の割合（クラスごとの倍率）である．候補 1 が直すのは，どの感情をどの感情と取り違えるか（表の非対角）である．
  - BBSE は混同行列をソースのラベルで推す．候補 1 はターゲットの K 本で推す．
  - MER-DA の提案は疑似ラベルの感情ごとの総量をそろえる（`soft_rows`）ので，割合の補正は打ち消される．K 本は感情ごとに同数なので，K 本から割合は推せない（検査役の [M3]・[M7]）．
  - 共通点：BBSE の誤差の上界の形（最小の固有値が小さいほど推定が揺れる）は，候補 1 の表の推定にもそのまま当てはまると推測する．
- Azizzadenesheli ほか 2019（RLLS．ICLR 2019）○：標本が少ないと BBSE の推定が大きく崩れるので，正則化を足した．https://tensorlab.cms.caltech.edu/users/anima/pubs/RLLS_ICLR.pdf
- URL：https://arxiv.org/abs/1802.03916

**Diamant ほか 2024．DCPL（ソースのデータを使わない領域適応で疑似ラベルの取り違えを解く）．ECCV 2024 ◎**
- ターゲットの疑似ラベルを雑音のあるラベルとみなし，雑音の遷移行列 T̂ を網と一緒に学ぶ．損失は「T̂ · p_θ(y|x) と疑似ラベルの交差エントロピー ＋ λ·tr(T̂)」である（式 4）．ターゲットのラベルは使わない．
- 候補 1 との違い：DCPL は表をラベル無しで推し，対角の和への罰則で一意にする．候補 1 は K 本の正解で表を推す．
- URL：https://arxiv.org/abs/2401.01650

**Guo ほか 2021．MetaCorrection．CVPR 2021 ○**：意味分割の領域適応で，疑似ラベルの雑音を遷移行列で表し，ソースのデータをメタの集合にしたメタ学習で推す．ターゲットのラベルは使わない．https://openaccess.thecvf.com/content/CVPR2021/html/Guo_MetaCorrection_Domain-Aware_Meta_Loss_Correction_for_Unsupervised_Domain_Adaptation_in_CVPR_2021_paper.html

**Yi ほか 2023．ソースのデータを使わない領域適応と雑音のあるラベルの学習．ICLR 2023 ○**：ソースのモデルの疑似ラベルの雑音は普通のラベル雑音と分布が違い，雑音のあるラベルの標準の手法は効きにくいと述べた．学習の初期の性質を使う正則化で対処する．https://arxiv.org/abs/2301.13381

**Yu & Lin 2023．Source Label Adaptation（半教師ありの領域適応）．CVPR 2023 ○**：ターゲットの少数ラベル（クラスあたり 1 本か 3 本）の設定で，混同行列を「ソースとターゲットを混ぜて学ぶとターゲットがずれる」ことの診断に使い，ソースのラベルをターゲットの特徴空間に合わせて付け直す．ターゲットの K 本で教師の取り違えの表を推して疑似ラベルを直すことはしない．https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_Semi-Supervised_Domain_Adaptation_With_Source_Label_Adaptation_CVPR_2023_paper.pdf

---

## 3. 最も近い先行研究 5 本と候補 1 の比較

| | GLC（Hendrycks ほか 2018） | SLaM（2023）と Iliopoulos ほか 2022 | Dao ほか 2021 | BBSE（Lipton ほか 2018） | Albanie ほか 2018 | 候補 1 |
|---|---|---|---|---|---|---|
| データ | CIFAR-10・100，SVHN，Clothing1M | CIFAR，ImageNet，CelebA など | 表形式 5 つ，CIFAR-10 | MNIST，CIFAR-10 | VoxCeleb（ラベル無し）．評価は外部の感情コーパス | AFEW，IEMOCAP 即興，MELD，MOSEI の LOCO |
| ラベルの量 | 信頼データは全体の 1〜25%（数値の表の最小はクラスあたり 25 本） | 検証用 500〜2000 本 | 全点にラベル | ソースの取っておいたデータ | ターゲットの感情ラベル無し | ターゲットに感情ごとに K = 1〜25 本 |
| 雑音のあるラベルの出し手 | 人工の雑音か弱い分類器 | 少数のラベルで学んだ教師 | 同じデータで学んだ教師 | ソースで学んだ予測器 | 顔の教師 | 声・顔・言葉の分類器の確率の積（K 本を使って学び直したもの） |
| モダリティ | 1 つ | 1 つ | 1 つ | 1 つ | 顔 → 声 | 3 つの積 → 声・顔・言葉の各分類器と融合 |
| 学ぶ側 | 深い網 | 深い網 | 決定木の森，ResNet | 任意（重みを付けて学び直す） | 声の深い網 | 凍結した特徴の上の線形の分類器（ソースの平均への L2 の引き戻しつき） |
| 測る量 | 誤り率と，雑音の強さに対する曲線下面積 | 正解率 | AUC，正解率 | 割合の推定誤差，正解率 | 外部コーパスでの感情の正解率と混同行列 | UAR．K 本だけとの差，温度つき蒸留との差 |
| 誤り方の推し方 | 信頼データでクラスごとに p̂(ỹ|x) を平均 | 確信度から正誤の確率を単調な回帰で推す | 表を推さない（各点のラベルで 1 次の補正） | 取っておいたデータの混同行列 | 推さない（測って示すだけ） | K 本でクラスごとに疑似ラベルを平均（GLC と同じ式）して縮小 |
| 補正の掛け方 | 前向き補正（生徒の出力に Ĉᵀ） | 生徒の予測に雑音を混ぜる（前向きの一種） | 直交化した損失 | 割合の比で重みを付ける | なし | 疑似ラベルを「疑似が j のとき本当が i の確率」で置き換える |
| 信頼データと教師の学習 | 分ける（教師は雑音のあるデータだけで学ぶ） | 分ける（検証データを教師の学習に入れない） | 交差の当てはめで分ける | 分ける | 該当しない | 分けていない（K 本で教師を学ぶ）．抜いて作り直す必要がある |
| 主張 | 少数の信頼データで推した表で補正すると，強い雑音でも誤りが大きく減る | 教師の雑音の模型を小さな検証データで推すと蒸留が良くなる | 交差の当てはめは教師の過学習の害を，損失の補正は学び足りない教師の害を減らす | 任意の予測器で割合のずれを検出して直せる | 顔の教師から声の生徒へ感情の知識を移せる．移り方は感情で違う | （未定） |

- **同じ点**：少数の正解で「雑音のあるラベルの出し手の誤り方」を推し，それで生徒の学習を直す骨組みは，GLC，Iliopoulos ほか，SLaM と同じである．表の推し方は GLC と同じ式である．
- **違う点（設定）**：(a) 雑音のあるラベルの出し手が 3 つのモダリティの分類器の積であり，学ぶ側はその一部のモダリティしか見ない．(b) 正解はクラスあたり 1〜25 本で，その正解が出し手の学習にも入っている．(c) 学ぶ側は凍結した特徴の上の線形の分類器である．(d) コーパスをまたぐ転移である．
- **違う点（式）**：候補 1 の置き換えの式は，先行研究の前向き補正と後ろ向き補正のどちらでもない．4 節のとおり，この式は誤りを打ち消さない．

---

## 4. 候補 1 の補正の式について（こちらの導出．計算での検算はしていない）

記号：本当の感情を y，疑似ラベルの確率を q(x)（4 つの感情の確率），取り違えの表を Cᵢⱼ = P(疑似 = j | 本当 = i) とする．

1. **GLC の仮定**（x が決まれば疑似ラベルは本当のラベルに依らず，誤り方はクラスだけで決まる）の下では，疑似ラベルは本当の事後確率 p(y|x) に表を 1 回掛けたものになる．
   ```
   q(x) = Cᵀ p(y | x)
   ```
2. このとき誤りを打ち消すのは，後ろ向き補正 p(y|x) = C⁻ᵀ q(x) か，生徒の出力 g に Cᵀ を掛けて q に合わせる前向き補正（GLC）である．
3. **候補 1 の置き換え**は，割合を一様とおくと Rⱼᵢ = P(本当 = i | 疑似 = j) = Cᵢⱼ / Σₖ Cₖⱼ を使って次の式になる．
   ```
   q′(x) = Rᵀ q(x) = Rᵀ Cᵀ p(y | x)
   ```
   Rᵀ も Cᵀ も各列の和が 1 の行列なので，積も同じ性質を持つ．**表を 2 回掛けることになり，p(y|x) には戻らない．**置き換えた疑似ラベルは，混ぜ合わせが 2 回入るので，元の疑似ラベルより発話の間で互いに似た分布になる．C が単位行列（取り違えが無い）のときだけ R も単位行列になり，変化が無い．
4. この置き換えが正しいのは，学ぶ側が硬い疑似ラベル ỹ だけを見て入力 x を見ないときである（判断者が 1 人の Dawid–Skene の事後確率と同じ）．MER-DA の学ぶ側は特徴 x を見るので，この条件に当たらない．
5. Lukasik ほか 2020 の表 1 の言い方では，候補 1 の置き換えは「単位行列に何かを足す」平滑化の側にあり，「引く」後ろ向き補正の側には無い．ただし感情の対ごとに強さが違い，取り違えが非対称なら最大の感情が入れ替わりうる．したがって候補 1 は温度つき蒸留とは違うが，「感情ごとに強さの違う平滑化」とは区別できない．検査役の [M4] の心配は，式の上でも当たる．
6. **線形の生徒が受け取れる分**：ソフトマックスの交差エントロピーは目標について線形なので，学ぶ側の最適解は「自分の特徴から予測できる目標の部分」にだけ合う．目標に定数の行列 M を掛ける補正（後ろ向き補正，候補 1 の置き換えのどちらも）では，学ぶ側は「自分の特徴から予測できる疑似ラベルの部分」に M を掛けたものを学ぶ．ここから次の予言が出る（推測）．**3 つの条件の 3 つ目（疑似ラベルの判断を自分の特徴でまねできる）が欠ける分類器では，どの補正でも得はほぼ 0 になる．**採用した主張では，3 つ目がそろったのは声だけだった．
7. **MOSEI での推定の崩れ**：MOSEI の表 A（出所 `pair`）の最小の特異値は 0.04〜0.08，条件数は 13.6〜26.1 である（実験 12 の 7 節）．K = 5 の 1 成分の標準誤差は約 0.19 である（検査役の [M2]）．後ろ向き補正は逆行列を掛けるので，MOSEI では推定の誤差が大きく増えると推測する．一方，4×4 の読み出しで直せる上限が大きいのも MOSEI である（K = 0 で声 +2.8，言葉 +4.0，顔 +5.0．実験 12 の 7 節．全ラベルを使った上限）．**直せる分が大きいコーパスほど，K 本では表を推しにくい．**この関係は BBSE の誤差の上界（最小の固有値の 2 乗に反比例）と同じ形である．

この節の結論：候補 1 を実行するなら，補正の式は GLC の前向き補正（または縮小つきの後ろ向き補正）に替えるのがよい．置き換えの式をそのまま使うなら，それは「感情ごとの平滑化」として扱い，取り違えを打ち消す手法とは書けない．

---

## 5. 新しさの判定と，論文に書ける形

### 5.1 判定

**候補 1 は「既知の手法を新しい設定に当てた」以上の新しさを，手法としては主張できない．**理由は 3 つある．
1. 表の推し方は GLC と同じ式である．補正の式を GLC の前向き補正に替えれば手順全体が GLC になり，替えなければ 4 節のとおり補正として誤りを打ち消さない．
2. 「少数の正解で教師の誤り方を推し，ラベル無しの例での蒸留を直す」という骨組みは，Iliopoulos ほか 2022 と SLaM（2023）が既に出している．
3. 「信頼データを教師の学習にも使うときは分割して作り直す」ことは，BBSE の分割と Dao ほか 2021 の交差の当てはめで既知である．

設定の新しさ（3 つのモダリティの積を出し手にして 1 つのモダリティの線形の分類器に渡す，クラスあたり 1〜25 本，コーパスをまたぐ）は書ける．ただし，それは「既知の補正を新しい設定に当てた」の範囲である．

### 5.2 論文の最後の節としての書き方（推す形）

「分析から導いた既知の補正を当て，分析の予言を確かめる」という形で書くことを推す．得が出なくても，予言が当たったか外れたかとして報告できる．

節の文案（英語の本文の下書き．数は計算の後に入れる）：

> Section X: Testing the analysis with a known correction. The analysis in Sections A–C predicts that the pseudo-labels help a classifier only when (i) there is room to improve, (ii) the pseudo-labels are more accurate than the classifier, and (iii) the classifier can reproduce the pseudo-labels from its own features. Condition (ii) suggests a direct intervention: if the multimodal pseudo-labeler systematically confuses some emotion pairs, correcting this confusion should raise its accuracy. We estimate the pseudo-labeler's class-conditional confusion matrix from the K labeled target utterances per class, using the estimator of Gold Loss Correction (Hendrycks et al., 2018) with cross-fitting (Dao et al., 2021) because the same K utterances are used to build the pseudo-labeler, and apply forward correction when retraining each unimodal classifier. We do not claim the correction as a new method. Instead, the analysis yields three predictions made before the computation: (P1) the correction raises UAR only for classifiers that satisfy condition (iii), i.e., the speech classifier in our corpora; (P2) the gain cannot exceed the full-label 4×4 read-out bound measured in Section B; (P3) on corpora whose confusion matrix is close to singular (MOSEI), the K-shot estimate is too noisy for the correction to help at K ≤ 5. A control that estimates the matrix after randomly reassigning the K labels across classes separates the effect of the label information from that of smoothing the targets (Lukasik et al., 2020).

注：検査役の [M4] の対照 (i) は「K 本のラベルを感情の中で入れ替える」と書かれているが，感情の中で入れ替えても感情ごとの平均は変わらないので，推す表は元と同じになり対照にならない．文案のとおり「感情の間で入れ替える」か，対照 (ii)（同じ平均のエントロピーになるよう対称に混ぜる）に直す必要がある．設計の段で決めること．

### 5.3 書けること・書けないこと

| 書けること | 書けないこと |
|---|---|
| 補正の手順は GLC であり，交差の当てはめは Dao ほか 2021 の型である | 新しい補正の手法を提案した |
| 3 つの条件から，どの分類器で補正が効くかを計算の前に予言した | 補正で普通の蒸留より良くなった（計算の前には書けない．検査役の見込みは声で 0.2〜1.3 ポイント程度が上限の目安） |
| 3 つのモダリティの積を出し手にし，1 つのモダリティの線形の分類器に渡す設定で，クラスあたり 1〜25 本の正解で表を推した | クラス条件つきの雑音の仮定が成り立つ（SLaM の著者は，ラベル無しの例での蒸留では成り立ちにくいと書いている） |
| 候補 1 の元の置き換えの式は感情ごとの平滑化に当たる（4 節の導出を検算した後なら） | 温度つき蒸留と違う効果が出た（[M4] の対照の結果が出るまでは書けない） |

---

## 6. 狙うべき差分（MER-DA の記録で測れるものを優先）

1. **K と表の特異値で補正の得が決まるか（クラスあたり 1〜25 本）．**GLC の数値の表はクラスあたり 25 本以上だけで，それより少ない場合は図だけである．MER-DA には，全ラベルで推した表 A の最小の特異値と条件数，全ラベルの 4×4 の読み出しの上限が，コーパス × モダリティ × K ごとにある（実験 12 の下調べ．`results/dai12_probe/cells_nl.csv`）．単位ごとの `results/dai12_probe/rows/*.npz` には，候補の上の疑似ラベルの出所の確率（`pool_teacher_K{K}_{m}_{出所}`），候補の本当の感情（`pool_y`），K 本の位置（`kpos_K{K}`）が入っている．**学び直しをせずに，「K 本で推した表で補正した疑似ラベルの当たり」と「全ラベルで推した表で補正した当たり」の差を，K と特異値ごとに数えられる．**ただしこの出所は `pair` で，提案が使う「学び直した積」ではない．学び直した 3 つの分類器の候補の上の確率は，実験 1 の評価の回の出力（`results/dai1/eval/units/*_p2.npz` の `p2_K{K}_ret_{A,V,T}_pool`）にあり，積を作れる．評価のターゲットのラベルを使うと評価の回の解析になるので，検査役の 6 節の 4 点目に従い，まず仮のターゲットで行う．
2. **まねできる分と補正の得の関係（3 つの条件の 3 つ目）．**4 節の 6 の予言「まねできない分類器では補正の得はほぼ 0」は，実験 14 のまねの一致の量と組み合わせて確かめられる．GLC，SLaM，C2KD は，生徒が教師の判断をどれだけ再現できるかと補正の得を結びつけていない．
3. **感情の対ごとに，どの取り違えが直ったか．**Albanie ほか 2018 は教師と生徒の感情ごとの混同行列を示したが，補正はしていない．MER-DA には全ソース設定での感情の対ごとの向きの損の分解（声で減った対の 7 割が喜びを含む 3 対）がある．表の非対角のうち，どの対の取り違えが補正で動いたかを，保存済みの確率から数えられる．
4. **補正の得と平滑化の得を分けること．**Lukasik ほか 2020 は 1 つのモダリティで平滑化と補正を比べた．3 つのモダリティの積を出し手にした設定での切り分けは，今回の検索では見つからなかった．検査役の [M4] の対照を入れれば測れるが，新しい計算（学び直し）が要る．

優先の順：1 と 3 は保存済みの出力から学び直しなしで出せる（評価の回のラベルの扱いに注意）．2 は実験 14 の量と結合すれば出せる．4 は新しい計算が要る．

---

## 7. 今回の検索で見つからなかったこと

- 3 つ以上のモダリティの分類器の積（や融合）を出し手にし，その取り違えの表を少数の正解で推して補正してから，1 つのモダリティの生徒に渡す研究は見つからなかった．
- 感情認識（声・顔・言葉）で，GLC や前向き・後ろ向き補正を疑似ラベルや弱いラベルに当てた研究は見つからなかった（"gold loss correction" と感情・マルチモーダルの組み合わせで検索して 0 件）．
- クロスモーダル蒸留で，ターゲットの少数の正解を使って教師の取り違えを補正する研究は見つからなかった．C2KD，Albanie ほか，LanSER は補正をしない．
- 半教師ありの領域適応で，ソースのモデル（または疑似ラベルのモデル）の混同行列をターゲットの K 本で推して疑似ラベルを直す研究は見つからなかった．疑似ラベルの遷移行列を推す領域適応の研究（DCPL，MetaCorrection）は，ターゲットのラベルを使わない．
- 信頼データが教師の学習にも入っている（分けられない）ときに，GLC の表の推定がどれだけ偏るかを測った研究は見つからなかった．Iliopoulos ほかと SLaM は分けることを前提にしている．
- 候補 1 の「ベイズの規則での置き換え」と同じ式を，入力を見る生徒の学習に使って誤りを打ち消すと主張した研究は見つからなかった．4 節の導出（この式は表を 2 回掛ける平滑化になる）を述べた文献も見つからなかった．ただしこの導出は Lukasik ほか 2020 の表 1 の議論から直接出るので，新しい知見として書く価値は小さいと推測する．
- 調べていないもの：Tang & Lease 2011 と Carpenter の原文，FasTEN と Yang ほか 2022 の本文，Iliopoulos ほか 2022 と Dao ほか 2021 の掲載先の公式の頁．

---

## 8. 文献一覧

| 文献 | 掲載先 | 水準 | URL |
|---|---|---|---|
| Hendrycks, Mazeika, Wilson, Gimpel. Using Trusted Data to Train Deep Networks on Labels Corrupted by Severe Noise | NeurIPS 2018 | ◎ | https://arxiv.org/abs/1802.05300 |
| Patrini, Rozza, Krishna Menon, Nock, Qu. Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach | CVPR 2017 | ◎（過去の調査） | https://arxiv.org/abs/1609.03683 |
| Lukasik, Bhojanapalli, Menon, Kumar. Does label smoothing mitigate label noise? | ICML 2020 | ◎ | https://arxiv.org/abs/2003.02819 |
| Veit ほか. Learning From Noisy Large-Scale Datasets With Minimal Supervision | CVPR 2017 | ◎（過去の調査） | https://arxiv.org/abs/1701.01619 |
| Iliopoulos, Kontonis, Baykal, Menghani, Trinh, Vee. Weighted Distillation with Unlabeled Examples | NeurIPS 2022（文献表で確認） | ◎ | https://arxiv.org/abs/2210.06711 |
| Kontonis, Iliopoulos, Trinh, Baykal, Menghani, Vee. SLaM: Student-Label Mixing for Distillation with Unlabeled Examples | NeurIPS 2023 | ◎ | https://arxiv.org/abs/2302.03806 |
| Dao, Kamath, Syrgkanis, Mackey. Knowledge Distillation as Semiparametric Inference | ICLR 2021（未確認） | ◎ | https://arxiv.org/abs/2104.09732 |
| Wen, Lai, Qian. Preparing Lessons: Improve Knowledge Distillation with Better Supervision | （未確認） | ○ | https://arxiv.org/abs/1911.07471 |
| Kye ほか. FasTEN | arXiv 2111.14932 | ○ | https://arxiv.org/abs/2111.14932 |
| Yang ほか. Estimating Instance-dependent Bayes-label Transition Matrix using a Deep Neural Network | ICML 2022 | ○ | https://proceedings.mlr.press/v162/yang22p.html |
| Northcutt, Jiang, Chuang. Confident Learning | JAIR 2021 | ○ | https://arxiv.org/abs/1911.00068 |
| Tang & Lease. Semi-Supervised Consensus Labeling for Crowdsourcing | SIGIR CIR Workshop 2011 | △ | https://bigml.cs.tsinghua.edu.cn/%7Ejun/pub/selective-crowds.pdf（要約を含む論文） |
| Carpenter. TREC 2011 Crowdsourcing Track | TREC 2011 | △ | https://trec.nist.gov/pubs/trec20/papers/LingPipe.crowdsourcing.pdf |
| Shi ほか. Evaluating multiple models using labeled and unlabeled data（SSME） | arXiv 2501.11866 | ○ | https://arxiv.org/abs/2501.11866 |
| Angelopoulos ほか. Prediction-Powered Inference | Science 2023（未確認） | ○ | https://arxiv.org/abs/2301.09633 |
| Albanie, Nagrani, Vedaldi, Zisserman. Emotion Recognition in Speech using Cross-Modal Transfer in the Wild | ACM MM 2018 | ◎（過去の調査） | https://arxiv.org/abs/1808.05561 |
| Huo ほか. C2KD | CVPR 2024 | ◎（過去の調査） | https://openaccess.thecvf.com/content/CVPR2024/html/Huo_C2KD_Bridging_the_Modality_Gap_for_Cross-Modal_Knowledge_Distillation_CVPR_2024_paper.html |
| Gong ほか. LanSER | Interspeech 2023 | ◎（過去の調査） | https://arxiv.org/abs/2309.03978 |
| Li ほか. Semi-Supervised Cognitive State Classification from Speech with Multi-View Pseudo-Labeling | arXiv 2409.16937 | ○ | https://arxiv.org/abs/2409.16937 |
| Cheng ほか. Joint-Modal Label Denoising | ECCV 2022 | ○ | https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/785_ECCV_2022_paper.php |
| Lipton, Wang, Smola. BBSE | ICML 2018 | ◎（過去の調査） | https://arxiv.org/abs/1802.03916 |
| Azizzadenesheli ほか. RLLS | ICLR 2019 | ○ | https://tensorlab.cms.caltech.edu/users/anima/pubs/RLLS_ICLR.pdf |
| Diamant ほか. De-Confusing Pseudo-Labels in Source-Free Domain Adaptation（DCPL） | ECCV 2024 | ◎ | https://arxiv.org/abs/2401.01650 |
| Guo ほか. MetaCorrection | CVPR 2021 | ○ | https://openaccess.thecvf.com/content/CVPR2021/html/Guo_MetaCorrection_Domain-Aware_Meta_Loss_Correction_for_Unsupervised_Domain_Adaptation_in_CVPR_2021_paper.html |
| Yi ほか. When Source-Free Domain Adaptation Meets Learning with Noisy Labels | ICLR 2023 | ○ | https://arxiv.org/abs/2301.13381 |
| Yu & Lin. Semi-Supervised Domain Adaptation With Source Label Adaptation | CVPR 2023 | ○ | https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_Semi-Supervised_Domain_Adaptation_With_Source_Label_Adaptation_CVPR_2023_paper.pdf |
| MER2023 SEMI の参加報告（クラスごとの上位 k の疑似ラベルなど） | ACM MM 2023 の併設の課題 | ○ | https://doi.org/10.1145/3607865.3613182 |
| 弱い教師と少数のラベルの統合 | arXiv 2206.09496 | △ | https://arxiv.org/abs/2206.09496 |
