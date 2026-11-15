# 査読報告（機械学習・統計の立場）

## 1. 論文の要約

新しいコーパス（ターゲット）に感情あたり K 件（0〜50）のラベルしかないとき，凍結した音声・顔・テキスト特徴の上の多項ロジスティック回帰に他コーパスのラベル付きデータを加えると何が改善するかを，6 コーパス 7 群の全順序対で測る論文である．
各 2 感情の判定を閾値のずれ o と分離度 d に分ける等分散ガウスのモデルを使い，K≤5 の単一モダリティの音声と顔では，ラベルが合併の分類器に加える改善の 0.54〜0.83 が閾値の修正から来ること，方向の誤差は q/K（音声で q=16.1〜22.2）でしか減らないこと，1 つのソースは方向について感情あたり 8.5〜11.8 ラベルに相当することを報告する．
後半では，ソースだけで学習した別モダリティの分類器（融合した始点など）がターゲットの未ラベル発話全体にクラス均衡の弱ラベルを付け，始点に繋留した単一モダリティの線形の生徒を学習する手法を提案する．
同じ擬ターゲット選択で調整した 9 手法と比べ，基本版は 22 セル中 19 で，教師を K ラベルで学習し直す拡張版は 11 の生徒すべてで，最強の比較手法を平均 0.8〜3.0 UAR 点上回ると主張する．

## 2. 推薦

**Major Revision**

実験の設計は丁寧である．話者を分けた分割，K について入れ子でソースに依らない抽出，ターゲットのラベルを使わない擬ターゲット選択，オラクルとの差の報告がそろっており，閾値と方向の分解は感情認識のコーパス間転移に新しい見方を与える．
しかし次の 3 点が解決されないと結論を支持できない．(1) 区間と検定が独立でない反復を独立とみなしている．手法の優位の主張は 4〜6 ターゲットの平均に基づき，拡張版は評価と同じターゲットの結果を見て設計されている．(2) 要旨の中心の数値（閾値の割合，q，n）は，近似の成り立つ範囲の外で導いた式と 1 母数の当てはめに依存しており，当てはまりの診断が示されていない．(3) 比較の差は教師と候補の数の違いを含み，部品の寄与が分かれておらず，主要な格子と部品が再実装できるほど書かれていない．
いずれも追加の解析と記述の修正で対応できる範囲であるため，Reject ではなく Major Revision とする．

## 3. 強み

- 全手法がテストの話者と K ラベルの抽出を反復ごとに共有し，差を反復の中で対にして比べている（3 節冒頭，3.3 節）．抽出は K について入れ子で，ソースに依らない．IEMOCAP の 2 群の間ではテストの話者をソースからも除いている．
- ハイパーパラメータを，ターゲットのラベルを使わない擬ターゲット（leave-one-domain-out）で選び，テスト話者で選んだオラクルとの差（最大 2.5 点）を報告している（3.6 節，6.4 節）．比較手法も同じ手順で調整し，「最強の比較手法」はセルごとにテストの平均で選んでいるので，比較は提案手法に不利な側に寄っている．
- 利得の地図が 3 モダリティとその融合（顔を含む），7 群の全順序対，K=1 から全プールまでを覆い，第 2 エンコーダでも繰り返している（4.1 節，図 2）．感情認識のコーパス間転移でこの範囲を測った報告は少ない．
- 閾値の説明を，モデルに依らない操作（中心化，切片だけの学び直し）でも確かめている（4.3 節，図 S2）．
- 失敗も報告している．基本版が K≥10 で始点を下回ること，選択の損失，IEMOCAP の fold の重複，原論文の値で固定したエントロピー最小化がチャンスレベルまで崩れること（6.3 節，図 S8）は，この分野の実務者に役立つ負の結果である．
- 限界の節が，対象の範囲（凍結特徴と線形の頭部，4 感情，英語のコーパス，CMU-MOSEI の話者の不完全な分離）を具体的に書いている（8 節）．

## 4. 指摘

### Major

### [Major-1] 区間と検定は反復の間の依存を無視しており，それに基づいて手法の優位を「11 の生徒すべて」に一般化している
- 該当箇所: sections/3-setting.tex:60, 98–100（3.3 節，3.6 節第 2 段落）／sections/4-part1-what-few-labels-fix.tex:10, 18（4.1 節第 1 段落，図 2 の説明文）／sections/6-comparison.tex:22–25（6.2 節第 2 段落）／表 3（6-comparison.tex:27–53）／main.tex:35（要旨）／sections/9-conclusion.tex:118／補足の表 S3
- 現状:
  - 3.3 節: "For each target group, we split the speakers into five folds, and iteration $r$ tests on fold $r \bmod 5$"
  - 3.6 節: "Intervals are 95% $t$ intervals of the paired differences over iterations ... The ten IEMOCAP speakers yield only five (improvised) or seven (scripted) distinct test folds"
  - 4.1 節: "its 95% interval, treating the thirty iterations as independent, lies above zero for 40 of the 42 ordered pairs"．図 2 の説明文: "(too narrow for IEMOCAP targets, whose iterations share test folds)"
  - 6.2 節: "We test each margin over target and iteration pairs with a two-sided Wilcoxon signed-rank test ... this leaves 3 and 7 significant students ... With four to six targets per configuration, these tests describe the variation within our corpora, not generalization to other corpora."
  - 要旨: "an extended version beats them for all eleven students, second encoders included, by 0.8 to 3.0 UAR points"．結論: "exceeds the strongest competitor ... for every student at $K=0$ and $K=5$"
- 問題:
  1. 30 反復は同じソースのデータ（IEMOCAP の対以外では全反復で同一）を使い，テストの話者も反復の間で重なる．3.3 節の文どおり fold が r mod 5 で固定なら，どのターゲットでも 30 反復は 5 つのテスト集合を 6 回ずつ使う．その場合，図 2 の説明文が IEMOCAP にだけ認めている「区間が狭すぎる」問題は全ターゲットに及ぶ．fold を反復ごとに引き直しているなら，その手順が書かれていない（IEMOCAP で 7 つの異なる fold が生じる理由も分からない）．どちらの場合も，反復を独立とみなした t 区間は分散を小さく見積もる（学習とテストを引き直す反復の間の相関は Nadeau and Bengio, Machine Learning 52, 2003 が扱っている）．図 2b の「区間が 0 より上の対の割合」はこの区間に依存し，対・K の 6 段・構成にわたる多数の区間の多重性も考慮していない．
  2. Wilcoxon 検定は「ターゲット×反復」を単位にする．同じターゲットの 10 反復は始点，ソース，未ラベルのプールの大部分を共有するので，有効な標本数はターゲットの数（4〜6）に近い．fold を平均すると K=0 で有意な生徒は 11 中 3 に減る．ターゲットを単位にすると，4 ターゲットでは両側の符号検定も Wilcoxon 検定も p 値の最小値が 0.125 であり，有意になりえない．
  3. 表 S3 の「Pos.」列によると，AVT のテキストの生徒では ours-ext が最強の比較手法を上回るのは K=0 でも K=5 でも 4 ターゲット中 2 であり，AV の融合の生徒の K=5 では 6 ターゲット中 4 である．要旨と結論の "for all eleven students"／"for every student" はターゲットの平均についての記述であり，個々のターゲットでの優位は示されていない．本文はこの列に触れていない．
  4. 表 3，図 6，表 2，図 4a には区間も標準誤差もない．
- 求める対応:
  - fold の作り方（話者の fold への割り当てを反復ごとに引き直すか，どの単位で引き直すか）を正確に書く．
  - 区間を，依存を考慮した方法で計算し直す．例：ターゲットとテスト fold を変量効果とする混合モデル，話者を単位にした階層ブートストラップ，補正した resampled t．図 2b の数え上げもこの区間でやり直す．
  - 表 3 に，ターゲットを単位にした効果（ターゲットごとの差の平均と範囲，上回ったターゲットの数）を加え，表 S3 の「Pos.」列の内容を本文で述べる．
  - 要旨と結論の文を，「4〜6 ターゲットの平均で上回る．ターゲットと反復を単位とした検定で有意なのは，fold を平均した後に K=0 で 11 中 3，K=5 で 11 中 7」のように，証拠の範囲に合わせて書き換える．
- 著者のデータが必要か: 要

### [Major-2] 拡張版の手法，教師の候補，比較に入れる生徒が，評価と同じターゲットの結果を見て決められている
- 該当箇所: sections/5-part2-mechanism-to-method.tex:28（5.1 節第 3 段落），:84–85（5.4 節第 1 段落），:90–91（5.4 節第 2 段落）／sections/6-comparison.tex:12（6.1 節）／sections/9-conclusion.tex:119
- 現状:
  - 5.4 節: "In a preliminary check over two iterations in AV, the fused start exceeds the audio start by 7.9 UAR points on the pool"．J は 5.3 節で "the UAR of the teacher minus that of the start on the pool" と定義され，プールのラベルを使う．
  - 5.4 節: "At $K \ge 10$, 4 of the 13 students fall below the five-value start ... The extended family replaces the candidate teachers ... it removes these losses for the AVT audio and AV fused students evaluated in Section 6 (the VT text and AT audio students were not run with it)."
  - 5.1 節: "the A, A (HuBERT), V, and AV students keep $\gamma$ and $\rho_s$ and reselect $(\rho_M, C', C_b)$ on pseudo-targets from a wider grid"
  - 6.1 節: "The AVT configuration evaluates the audio, face, text, and fused students on 4 targets, the AV, VT, and AT configurations the fused student on 6, 4, and 4 targets"
- 問題:
  1. 教師の候補を決めた予備の確認は，評価に使う AV のターゲットのプールのラベルで J を測っている．テストの話者は含まないが，評価対象のコーパスのラベルが手法の設計に入っている．
  2. 拡張版は，基本版が同じターゲットの K≥10 で始点を下回った結果を見てから足されたと読める．要旨の中心の主張である拡張版の優位は，設計に使ったのと同じターゲットと（おそらく同じ）10 反復で測られている．4〜6 ターゲットで 0.8〜3.0 点の差を論じるとき，設計の段階での選択は結果を楽観的にする．
  3. 5.4 節で弱ラベルの効果を示した 13 の生徒のうち，表 3 に入るのは AVT の 4 生徒と AV・VT・AT の融合の生徒だけである．AV の音声と顔，VT のテキストと顔，AT のテキストと音声の 6 つの単一モダリティの生徒は比較にない．AV は 6 ターゲットを持ち，ターゲットが最も多い構成である．除いた理由が書かれていない．
  4. 格子を "a wider grid" に広げた時期と理由が書かれていない．
- 求める対応:
  - 評価のターゲットの結果を見てから決めた事項（教師の候補，拡張版の部品と重み付けの選択肢，格子の拡張，比較に入れる生徒）を列挙する．
  - 拡張版を，設計に使っていないデータで確かめる．例：手法を凍結した後，比較に使っていない反復（テストの fold が異なるもの）で表 3 を作り直す．第 2 エンコーダの結果を手法の凍結後に得たのであれば，その順序を明記し，確認のための結果として示す．
  - AV・VT・AT の単一モダリティの生徒を比較に加えるか，除いた理由を書く．
  - どれもできない場合は，拡張版の結果を探索的な結果として要旨と結論で位置づける．
- 著者のデータが必要か: 要

### [Major-3] 閾値と方向の分解は，4 クラスの切片では同時に取り戻せない閾値の損失を数えており，UAR との対応も示されていない
- 該当箇所: sections/3-setting.tex:79–89（3.5 節）／sections/4-part1-what-few-labels-fix.tex:25–36（4.2 節），:62, 69–76（4.3 節）／表 2／main.tex:35（要旨）
- 現状:
  - 3.5 節: "the *threshold offset* $o$ ... can be set to zero by the intercepts alone"，"the loss of a classifier splits into a *threshold loss* $a(d,0)-a(d,o)$, which moving the threshold to the midpoint recovers, and a *direction loss* $a(\Delta,0)-a(d,0)$ ... we average both over the six pairs, in points of pairwise accuracy."
  - 4.3 節: "the accuracy predicted from the measured $d$ and $o$ correlates with the measured accuracy at 0.986–0.996, with a mean absolute error of 0.7–1.1 points."
  - 4.3 節: "Refitting only the intercepts of the source-only classifier on $K=5$ labels per emotion gains 4.9 points for WavLM audio, close to the 5.4 points that pooling gains with the same labels."
  - 要旨: "0.54 to 0.83 of what the labels add to a pooled classifier comes from correcting thresholds"
- 問題:
  1. 4 クラスでは切片の差 $b_k-b_l$ の自由度は 3 である．6 対それぞれの閾値のずれを 0 にする切片の差 $c_{kl}$ が $c_{kl}+c_{lm}=c_{km}$ を満たすのは特別な場合に限り，一般には 6 対を同時に 0 にできない．"can be set to zero by the intercepts alone" は対ごとには正しいが，6 対の平均としての閾値の損失は，切片で同時に取り戻せる量の上限である．閾値の割合 0.54〜0.83 は，切片で直せる割合を大きく見積もっている可能性がある．
  2. 分解は対の正解率（2 感情に限った再現率の平均）で行う．主指標の UAR は 4 クラスの argmax による判定で決まり，対の正解率の平均とは別の量である．要旨の文は UAR の改善の割合と読める．原稿の中で，UAR で測った量と分解の量が食い違っている．K=5 の WavLM 音声では，UAR で測った切片だけの学び直しの改善が合併の改善 5.4 点のうち 4.9 点（0.91）に達するのに対し，分解による閾値の割合は 0.62 である（表 2）．反復の数（3 と 10），平均と中央値，対の集合が違うことを考えても，この差は説明が要る．
  3. 改善を閾値の部分と方向の部分に分ける順序が書かれていない．閾値の損失の係数 $(d/4)\phi(d/2)$ は d に依存するので，「方向を先に変える」場合と「閾値を先に変える」場合とで割合が変わる．
  4. 2 群のモデルの平均絶対誤差 0.7〜1.1 点は，テキストと融合の構成の閾値の損失（表 2 の 0.7〜1.2 点）と同じ大きさである．これらの構成の閾値の割合は，モデルの誤差と区別できない．相関 0.986〜0.996 は正解率が対の間で大きく違うことから高くなる量で，当てはまりの根拠としては平均絶対誤差の方が重要である．
  5. 基準の Δ は，ターゲットの話者の半分で学習した縮小 LDA を残りの半分で測った値である（3.5 節）．AFEW（795 発話）や RAVDESS（672 発話）では Δ 自体が小さく見積もられ，分類器の d が Δ を超えて方向の損失が負になる対が生じうる．その頻度と扱い，縮小の強さが書かれていない．
  6. 式 (2) の閾値の費用 $(d/4)\phi(d/2)/(2K)$ を，測った閾値の損失と K ごとに比べた結果がない．また $\mathbb{E}[o^2]=1/(2K)$ は K 個のプールの標本の抽出のばらつきだけを数えており，プールの話者とテストの話者の間の平均のずれを含まない．
- 求める対応:
  - 切片だけを（プールの全ラベルまたはテストのラベルで）最適化したときの UAR と対の正解率を測り，6 対で同時に取り戻せる閾値の損失を示す．分解による閾値の損失との差を報告する．
  - 閾値の割合を UAR でも示し（例：切片だけの学び直しの改善÷全体の改善），上の 0.91 と 0.62 の違いを説明する．
  - 分解の順序を明記し，両方の順序の割合（またはその平均）を示す．
  - 方向の損失が負になる対の数と扱い，縮小 LDA の縮小の強さを書く．
  - 式 (2) の予言と測った閾値の損失を K ごとに比べる図を補足に加える．
  - 要旨の文を「対の正解率で測った改善のうち」と書き換える．
- 著者のデータが必要か: 要

### [Major-4] q/K の式とソースの価値 n の式は前提の成り立たない範囲で導かれており，予言と測定の照らし方も予言を支えるのに足りない
- 該当箇所: sections/4-part1-what-few-labels-fix.tex:38–45（4.2 節），:126–127（4.4 節第 2 段落），:133–146（4.5 節）／supplementary.tex:57–60, 62–65, 93（S2 節，図 S3 の説明文）／表 2／図 4b／main.tex:35（要旨）／sections/1-introduction.tex:28–29（貢献 2・3）
- 現状:
  - 4.2 節: "the form obtained from the dimension-proportional term of the asymptotic expansion of the misclassification rate of the linear discriminant ... Fitted to the separations of target-only classifiers along the label budget, $q$ is 16.1 for WavLM audio, 22.2 for HuBERT audio, 6.8 for AV, and 1.9 for the face"
  - 補足 S2 節: "for the sample linear discriminant with $p$ features and squared Mahalanobis distance $\Delta^2$, the first-order term ... gives $q = 2(p-1)/\Delta^2$ ... With an $\ell_2$ penalty the effective dimension is smaller than $p$, and we fit $q$ to target-only learning curves."
  - 4.5 節: "$n$ does not depend on $K$"，"For a single audio source, $n$ rises only from 8.5 to 11.8 labels per emotion while $K$ grows from 5 to 25"，"Starting from all sources pooled, $n$ is 20.2 and 28.9 labels per emotion for WavLM audio at $K=5$ and $K=10$."
  - 4.5 節: "Across audio pairs, the rank correlation between the predicted and the measured $n$ is 0.77–0.90"．4.4 節: "over ordered pairs and $K \in \{1, 5, 10\}$, the rank correlation between the predicted and the measured $\lambda$ is 0.84–0.91"
  - 図 S3 の説明文: "Points beyond the learning curve are in the top band, and points with $n \le 0$ in the bottom band."
- 問題:
  1. Okamoto と Raudys の展開は，次元 p を固定して標本数を大きくする近似（K≫p）である．本論文の K は 1〜50，特徴の次元は音声 1024，顔 512，テキスト 768（補足 S1 節）で，K≪p の範囲にある．補足の式 $q=2(p-1)/\Delta^2$ で p=1024 のとき q=16.1 を得るには Δ≈11.3 が要り，これは対の正解率 Φ(5.6)≈100% に当たるので，観測された正解率と合わない．したがって q は導出された「有効次元」ではなく，学習曲線に当てはめた 1 つの自由な係数である．さらに target-only の C は K ごとに交差検証で選ばれる（3.4 節）ので，当てはめは標本数の効果と正則化の変化を混ぜている．
  2. 当てはまりの診断がない．図 4b は式の曲線だけを描き，測った d(K) を描いていない．q の区間，ターゲットごとの q のばらつき，残差，ほかの関数形（例：K の冪）との比較がない．表 2 の q がターゲットの中央値なのか全体への当てはめなのかも書かれていない（図 4b の説明文は中央値）．テキストを含む構成で q が「–」になっている理由もない．当てはめが失敗したのであれば，それは式 (3) の適用範囲を示す結果である．
  3. 式 (5) は n が K に依存しないと予言する．原稿の数値では，単一ソースの n は K=5 から 25 で 8.5 から 11.8 へ約 39% 増え，全ソースの n は K=5 から 10 で 20.2 から 28.9 へ約 43% 増える．"rises only" は予言からのずれを小さく見せている．
  4. n の測り方は，仮説転移の UAR を target-only の学習曲線に置き K を引く．UAR には閾値の効果も入るので，この n は方向だけの価値ではない．図 S3 には曲線の外（上の帯）と n≤0（下の帯）の点が少なくないが，順位相関と比（0.79〜0.87 倍）の計算でこの打ち切られた点をどう扱ったかが書かれていない．学習曲線は K∈{1,5,10,25,50,全プール} でしか測っておらず，その間の補間の方法も書かれていない．
  5. 順位相関は対と K を混ぜて計算している．λ の予言は $N_t$（K に比例）を含むので，K を混ぜた順位相関は K の違いだけでも高くなる．n の予言と測定はどちらも同じターゲットの q と学習曲線を使うので，ターゲットの違いだけでも相関が生じる．水準については，λ の予言は平均を過大に見積もり（4.4 節），全ソースの n の比は 0.65〜1.15 と幅が広い．
- 求める対応:
  - 式 (3) を「K≫p の古典的な展開と同じ形を持つ経験式」と位置づけ直し，要旨の "a rate set by an effective dimension" と貢献 2 の書き方を弱める．
  - ターゲットごとに測った d(K) と当てはめた曲線を補足に示し，q の区間（反復または話者のブートストラップ），残差，ほかの関数形との比較を加える．テキストの構成で q を出さない理由を書く．
  - λ と n の順位相関を，K ごとに，またターゲットの中で計算して示す．
  - n の打ち切りと補間の扱いを書く．"$n$ does not depend on $K$" を，観測された K への依存を報告する文に改める．
- 著者のデータが必要か: 要

### [Major-5] 比較の差は教師と候補の数の違いを含んでおり，提案の部品の寄与が分かれていない．凍結特徴で標準的な分布の合わせ込みも比較にない
- 該当箇所: sections/6-comparison.tex:8–11（6.1 節）／sections/5-part2-mechanism-to-method.tex:67–68, 83–85, 91（5.3・5.4 節）／sections/2-related-work.tex:61–62, 68（2 節）／表 3／補足の表 S1・S2
- 現状:
  - 6.1 節: "Competitors that need a teacher receive the fused start (distillation with a temperature and hard pseudo-labels for single-modality students), the fused student refit on the $K$ labels (generalized distillation), or the product of the single-modality starts (distillation for fused students); none receives a two-stage reference, so the margins include the choice of teacher."
  - 基本版は融合の始点，単一モダリティの始点の積，一致で重みを付けた始点から教師を選び，融合の生徒では 2 段階の参照も候補にする．拡張版は K ラベルで学び直した融合の生徒と 2 段階の参照を教師にし，重み付けを 3 通り（balanced，unbalanced，hard above a threshold）から選ぶ（5.4 節）．
  - 2 節: "Specific to our combination are a teacher trained on sources alone that sees modalities the student lacks, weak labels that train the classifier itself ..., separate anchors for weights and intercepts, and the selection of the teacher and the weak-label weight for each $K$."
- 問題:
  1. ours-ext と比較手法の差は，学習の仕方の違いと，教師の違い（2 段階の参照の有無，教師の候補の数）を同時に含む．ours-ext と同じ教師（K ラベルで学び直した融合の生徒）を受け取る一般化蒸留（GD）は，K=5 で AVT の音声とテキスト，第 2 エンコーダの音声とテキストの最強の比較手法である（表 3）．ours-ext と GD の違いは，列の正規化（クラスの均衡），2 段階の参照，重み付けの選択肢，格子の広さである．どの部品が差を生んでいるかを示す除去実験がない．5.3 節は同じモダリティの教師について列の正規化の寄与（K=0 の音声で 0.9 点）を示すが，別モダリティの教師については示していない．提案の新しさを部品の組み合わせとして述べている（2 節）以上，各部品の寄与が要る．
  2. ours-ext は教師，重み付け，m の組から擬ターゲットで選ぶ．比較手法が選ぶ候補の数は書かれていない．擬ターゲットとターゲットが似ている場合，候補の多い手法が有利になる．
  3. Part I は閾値の損失の大部分がコーパスの平均のずれであると示し，中心化を手法の部品にしている．中心化を 2 次の統計に広げる CORAL や subspace alignment のような，凍結特徴での教師なしの分布の合わせ込みは計算が軽いが，比較にない．関連研究で挙げた xMUDA［41］のように融合の出力で単一モダリティの頭部を学習する教師なしの領域適応も，比較手法にない．
- 求める対応:
  - 教師を使う比較手法（KD，HPL，GD）に ours-ext と同じ教師（2 段階の参照を含む）を与えた版を加える．
  - 同じ教師のもとで，列の正規化，2 段階の参照，重み付けの選択，m の選択を 1 つずつ外した除去実験の表を加える（少なくとも AVT の 4 生徒の K=0 と K=5）．
  - 各手法が擬ターゲットで選ぶ候補の数を表にする．
  - 中心化した特徴に CORAL などの 2 次の合わせ込みを加えた始点を比較に入れるか，入れない理由を書く．
- 著者のデータが必要か: 要

### [Major-6] 提案手法と比較手法を再実装するのに要る設定が書かれていない
- 該当箇所: sections/3-setting.tex:53, 88／sections/4-part1-what-few-labels-fix.tex:77, 143／sections/5-part2-mechanism-to-method.tex:19, 25, 28, 68, 75, 83, 85, 91／sections/6-comparison.tex:9–10／図 5 の説明文／supplementary.tex:113–114, 157
- 現状と欠けている項目:
  - 弱ラベルの重み m と教師の "small grid"（5.3 節）：値がない．
  - 拡張版の "hard labels above $\tau$"（図 5）の τ：値がない．
  - 比較手法の tuned 版の格子（閾値，温度，未ラベルの損失の重み）と，fixed 版の "the values of the original papers"（6.1 節）：値がない．co-training の 2 つの変種の中身もない．
  - "A belief that combines the per-source classifiers, each weighted by its confusion pattern on pseudo-targets"（5.3 節），"the fused start down-weighted where these starts disagree"（5.4 節），2 段階の参照で予測を組み合わせる方法：式がない．
  - 計量 M（"a metric $M$ built from the scatter of the unlabeled target pool with mixing weight $\rho_M$"）：式がなく，$\rho_M=1$ が何を意味するか分からない．
  - 3.2 節の標準化の平均と分散を，どのデータ（ソースだけ，ターゲットのプールを含む，コーパスごと）で計算するか：書かれていない．コーパスごとに標準化すれば中心化と同じ働きをするので，Part I の閾値の結果に直接関わる．
  - 合併の学習でのクラスの重みを，ソースとターゲットをまとめて数えるか：書かれていない．
  - Δ を測る縮小 LDA の縮小の強さ（3.5 節）：書かれていない．
  - 2 層パーセプトロン（4.3 節）の幅，学習の設定，正則化：書かれていない．
  - n を測る学習曲線の補間（4.5 節）：書かれていない．
  - T，VT，AT，AVT の生徒で $(\rho_M, C', C_b)$ を選び直したか（5.1 節は A，A (HuBERT)，V，AV だけを述べる）：書かれていない．
  - LoRA の rank，学習率，エポック数，適用した層（補足 S7 節）：書かれていない．
  - コードと特徴の公開の予定：書かれていない．
- 問題: これらがなければ，提案手法の数値も比較手法の数値も第三者が再現できない．格子の大きさは Major-5 の 2 の公平さを判断するのにも要る．
- 求める対応: 上の項目を補足資料に表としてまとめ，コードの公開の予定を書く．
- 著者のデータが必要か: 不要（記述の追加）

### Minor

### [Minor-1] AFEW と CMU-MOSEI の話者の定義が不十分で，話者の分離の不完全さの影響が測られていない
- 該当箇所: sections/3-setting.tex:16–17（3.1 節第 2 段落）／表 1
- 現状: "CMU-MOSEI has no speaker annotation; its channel identity is known for 35% of the videos, and the video identifier stands in for the speaker elsewhere, so its speaker separation is incomplete."．表 1 の AFEW の話者の群は 282．
- 問題: AFEW は映画のクリップで話者の注釈を持たない．282 の話者の群をどう作ったかが書かれていない．CMU-MOSEI では 65% の動画で動画 ID を話者の代わりにするので，同じ YouTube の話者がプールとテストに入りうる．これは target-only と生徒の UAR を高くし，CMU-MOSEI をターゲットにした利得や n を変えうるが，影響の大きさが測られていない．
- 求める対応: AFEW の話者の群の作り方を書く．CMU-MOSEI をターゲットにした主要な結果を，チャンネルが分かる動画だけで測り直すか，影響の向きを議論する．
- 著者のデータが必要か: 要

### [Minor-2] IEMOCAP-script をターゲットにしたテキストの結果には，同じ台詞による重複が残る
- 該当箇所: sections/3-setting.tex:10（3.1 節第 1 段落）
- 現状: "The two IEMOCAP session types form separate groups because all ten speakers read the same lines in the scripted sessions, which lets a text classifier recognize the script instead of the emotion."
- 問題: 群を分けても，IEMOCAP-script の中ではプール（K ラベルと未ラベル）の話者とテストの話者が同じ台詞を読む．この群をターゲットにすると，K ラベルやプールで学習するテキストの分類器は台詞を覚えて正解できる．テキストを含む構成のターゲットは 4 つしかなく，その 1 つがこの群である．図 2 の T の利得（大きな K で負）と，表 3 のテキストと融合の生徒の数値に影響しうる．
- 求める対応: 台詞で分けた分割にするか，IEMOCAP-script を除いたテキストの結果を補足に示す．
- 著者のデータが必要か: 要

### [Minor-3] 要旨・序論・本文・補足のあいだで数値が食い違う
- 該当箇所と現状:
  1. sections/1-introduction.tex:29（貢献 3）は "20.2–30.5 for all audio sources pooled" と書くが，4.5 節（4-part1-what-few-labels-fix.tex:145）は "$n$ is 20.2 and 28.9 labels per emotion for WavLM audio at $K=5$ and $K=10$" と書き，30.5 は本文のどこにもない．
  2. main.tex:35（要旨）の "one source corpus supplies 8.5 to 11.8 labels per emotion of direction" は，本文（4-part1-what-few-labels-fix.tex:144）では K=5 から 25 にかけての変化である．要旨はソースや設定の間の幅と読ませる．
  3. sections/6-comparison.tex:19 は "exceeds the strongest competitor in 19 of the 22 combinations" と書き，上回らないセルとして AVT の音声とテキストの K=5，AT の融合の K=5 だけを挙げる．しかし表 3 の AVT の融合の K=5 は最強の比較手法も基本版も始点との差が 0.8 で，表 S2 でも 54.6 と 54.6 で同じである．第 2 エンコーダの音声の K=5 も 2.7 と 2.8 である．表示の桁で同じセルを「上回る」に数えている．
  4. sections/6-comparison.tex:24 は fold を平均した後について "this leaves 3 and 7 significant students (... at $K=5$ all but the AVT face and text students and the AV and AT fused students)" と書く．表 S3 では K=5 の AT の融合の p は <0.001，第 2 エンコーダの顔の p は 0.144 である．本文の列挙どおりなら，fold を平均した後に第 2 エンコーダの顔が有意になり，AT の融合が有意でなくなったことになり，"leaves" と合わない．
  5. 4-part1-what-few-labels-fix.tex:66 は "The median over pairs of the threshold's share of the loss is 0.10–0.27 across configurations" と書く．表 2 の平均の損失から計算すると，顔は 2.5/(2.5+2.4)≈0.51，AV は 2.5/(2.5+6.4)≈0.28 である．中央値と平均がこれほど離れるなら分布が大きく偏っている．:75 の "This share is much larger than the threshold's share of the loss" が顔で成り立つかもこの値に依存する．
  6. 5-part2-mechanism-to-method.tex:89 は "These gains exceed those from a same-modality teacher for audio and text students but not for face students" と書くが，AV の顔の生徒は K=0 で別モダリティの教師から 2.4 点，同じモダリティの教師から 1.0 点を得ている（:88，:76）．顔の生徒が下回るのは VT と AVT だけである．
- 問題: 読者が要旨と本文のどちらの数値を信じればよいか分からない．3 と 4 は検定と数え上げの結論に関わる．
- 求める対応: 1 は 30.5 の出所（HuBERT か）を本文に書く．2 は要旨に「K=5〜25 で」と書く．3 は差に許容幅（例：0.1 点）を設けて数え直すか，区間で判定する．4 は fold 平均後の検定の表を補足に加え，列挙を確かめる．5 は表 2 に構成ごとの損失の割合の中央値を加え，平均との違いを説明する．6 は「VT と AVT の顔の生徒では」に直す．
- 著者のデータが必要か: 要（3〜5），不要（1，2，6）

### [Minor-4] 「二十ラベルで目標の方向に近づく」という言い方は式 (3) と合わない
- 該当箇所: sections/4-part1-what-few-labels-fix.tex:45（4.2 節）
- 現状: "so for audio a direction comparable to the target's best needs on the order of twenty labels per emotion."
- 問題: 式 (3) で K=q のとき $d=\Delta/\sqrt{2}\approx0.71\Delta$ であり，分離度は目標より 29% 小さい．$d=0.95\Delta$ には $K\approx9q$，音声で感情あたり 150〜200 ラベルが要る．
- 求める対応: 「comparable」の水準を書き，数値を直す．5 節冒頭（5-part2-mechanism-to-method.tex:4）の "audio needs on the order of $q$ labels per emotion to approach the target's best direction" も同じように直す．
- 著者のデータが必要か: 不要

### [Minor-5] 加法モデルの説明率は，母数の数を考えると評価できない
- 該当箇所: sections/4-part1-what-few-labels-fix.tex:13（4.1 節）
- 現状: "an additive model with one effect per target and one per source explains 46–89% of the variance of the gains and of the largest budget at which a source still helps"
- 問題: A の 42 セルに対して母数は 13（ターゲット 7＋ソース 7−1），V と AV の 30 セルに対して 11 である．利得がでたらめでも説明率の期待値はおよそ 12/41≈0.29 と 10/29≈0.34 になる．下限の 46% はこの値から大きく離れていない．どの K の利得を使ったかも書かれていない．"the largest budget at which a source still helps" は 6 段の順序の値であり，分散で説明率を測る扱いが書かれていない．
- 求める対応: 自由度で調整した説明率か，ターゲットとソースの名前を並べ替えた基準との比較を示す．使った K と，順序の値の扱いを書く．
- 著者のデータが必要か: 要

### [Minor-6] 利得の地図と中心化の比較の結論が，C の決め方に依存する
- 該当箇所: sections/3-setting.tex:72–74（3.4 節第 2 段落）／supplementary.tex:86, 103, 107（図 S2・S4）／sections/4-part1-what-few-labels-fix.tex:70–71
- 現状: 合併の分類器は source-only の C を使う．"($C=1$ when a fold lacks an emotion)"．図 S2 の説明文: "Without centering the pooled classifier takes the $C$ of the source-only classifier; with centering, $C$ is selected by cross-validation."
- 問題:
  1. K=1 では 4 ラベルを 3 fold に分けるので，どの分割でも感情の欠ける fold が生じ，target-only の C は常に 1 になると読める．K=1 の利得（音声で +10.5 点）はこの固定した C に依存する．
  2. 図 S4 から読むと，交差検証で C を選ぶと顔の K=5 の利得が負（実線で約 −0.7）になり，本文の規則（点線）では正である．利得が負に変わる K は C の規則で変わる．
  3. 4.3 節の "After centering, the two gains shrink to 0.3 and 0.6 points" は，中心化の前後で合併の C の規則が違う比較である．また図 S2 の顔では，中心化と合併を組み合わせた改善（約 0.9 点）が中心化だけ（約 2.5 点）を下回るが，説明がない．
- 求める対応: 利得が負に変わる K を両方の C の規則で表に示す．図 S2 の比較を同じ C の規則でやり直し，顔の結果を説明する．
- 著者のデータが必要か: 要

### [Minor-7] 解析ごとに反復の数が 3，5，10，30 と違い，どの反復を使ったかが書かれていない
- 該当箇所: sections/4-part1-what-few-labels-fix.tex:69（中心化，3 反復），:85（図 4，10 反復），:126（λ，10 反復）／supplementary.tex:103（テキストの構成の図 S4，5 反復），:157（微調整，3 反復）／sections/6-comparison.tex:12（比較，10 反復と擬ターゲットの 5 反復）
- 問題: 反復の数を変えた理由と，どの反復（例：最初の 10）を使ったかが書かれていない．擬ターゲットの選択は「最初の 5 反復」を使う（3.6 節）．比較の 10 反復がそれと重なるかどうかも分からない．3 反復の結果（中心化，微調整）は区間を付けずに 4.3 節の主張の根拠にしている．
- 求める対応: 各解析で使った反復を明記し，3 反復の結果には区間を付けるか，予備的な結果として扱う．
- 著者のデータが必要か: 不要（記述），要（区間）

### [Minor-8] 命題 2 は前提をほぼ言い換えたものであり，その検証の読み方が測定より強い
- 該当箇所: sections/5-part2-mechanism-to-method.tex:47–59（5.2 節）／補足（命題 1・2 の証明がない）
- 現状: "If $E_0$ has no preferred direction under this prior, the expected fraction of $\|E_0\|^2$ that the span of any $4K$ points captures is the same for every choice of points made without their labels."．検証: "the label-free rules capture 0.5–1.2 times as much"．結び: "The direction error left by a good start is thus invisible to label-free selection"
- 問題:
  1. E0 が事前分布のもとで等方的なら，固定した部分空間が捕らえる割合の期待値はその次元だけで決まる．命題 2 はこの前提の言い換えに近く，証明も補足にない（補足 S2 節の導出は式 (2)〜(5) だけである）．4K 点が一次独立であるという条件も書かれていない．
  2. 測定では label-free の規則は 0.5〜1.2 倍で，1 より小さい側にも大きく外れる．等しさを予言する命題に対し，両側にずれが出ている．"invisible to label-free selection" は等方性の確認ではなく，「ランダムを大きく上回る規則がなかった」という観測である．
  3. UAR の差の基準 "beats random selection by one point"（最大 0.74 点）に区間がない．捕捉率の絶対値と，5.3〜12.2% を捕らえた部分空間の次元が書かれていない．次元が違えば捕捉率は比べられない．図 S5 の "Random (pool order)" が基準のランダムとどう違うかも分からない．
  4. 検証は K=5 だけであるが，要旨の "Neither few labels nor label-free selection repairs a strong start's direction" は K を限っていない．
- 求める対応: 命題 2 を観察（remark）として書き直すか，証明を補足に加える．捕捉率の絶対値と部分空間の次元，UAR の差の区間を示す．要旨に K=5 での結果であることを書く．
- 著者のデータが必要か: 要

### [Minor-9] J による上限は経験的な目安であるのに，関連研究では理論の帰結として述べている
- 該当箇所: sections/2-related-work.tex:73（2 節最終段落）／sections/5-part2-mechanism-to-method.tex:71（5.3 節）
- 現状: 2 節: "We use these results for a student anchored to a start: the teacher's UAR advantage over the start sets the scale of the gain"．5.3 節: "we expect, without a theorem, that the part of its gain that comes from the teacher's information stays below $J$."
- 問題: 引用した弱から強への汎化の研究［54］–［56］は，生徒が教師を上回る条件を扱っている．J が利得の尺度を決めるという文は，これらの結果から導かれていない．J はターゲットのプールのラベルを使って測る量でもある．
- 求める対応: 2 節の文を「経験的な目安として J を使う」に改める．
- 著者のデータが必要か: 不要

### [Minor-10] 擬ターゲットの始点は，実際のターゲットの始点より少ないソースで作られる
- 該当箇所: sections/3-setting.tex:94（3.6 節）／sections/6-comparison.tex:77（6.4 節）
- 現状: "each source group $c$ from another corpus serves in turn as the target of the method built from the groups outside the corpora of $T$ and $c$"
- 問題: テキストを含む構成の群は IEMOCAP の 2 群，CMU-MOSEI，MELD の 4 つである．例えば T=MELD，c=IEMOCAP impro のとき，T と c のコーパスの外に残るのは CMU-MOSEI だけである．擬ターゲットの始点は 1〜2 群のソースから作られ，実際のターゲットの始点（2〜3 群）より弱い．繋留の強さ C′ や重み m の最適値は始点の質に依存するので，テキストの生徒で大きい選択の損失（2.5 点と 2.2 点）の一因になりうる．
- 求める対応: 構成ごとに擬ターゲットの始点のソースの数を示し，選択の損失との関係を議論する．
- 著者のデータが必要か: 不要（記述），一部要

### [Minor-11] 「微調整より強い」という序論の主張は，確認の規模と設定の記述に比べて強い
- 該当箇所: sections/1-introduction.tex:12（1 節第 3 段落）／supplementary.tex:154–161（S7 節，図 S9）
- 現状: "a setting simple enough to analyze and, in a check with low-rank adaptation of the audio encoder on four targets at $K \le 5$, stronger than fine-tuning."
- 問題: 確認は 4 ターゲット×3 反復で，LoRA の設定と調整の手順が書かれていない．図 S9 から読むと，K=0 の微調整した始点（約 40%）は凍結した始点（約 47%）を大きく下回る．ソースでの微調整そのものが十分に調整されていない可能性があり，この結果から凍結特徴の方が強いとは言い切れない．
- 求める対応: LoRA の設定と調整の手順を書くか，序論の文を「この確認の設定では」に限る．
- 著者のデータが必要か: 不要（記述）

### [Minor-12] 測定を示さずに述べた説明が 2 つある
- 該当箇所: sections/4-part1-what-few-labels-fix.tex:128（4.4 節）／sections/7-discussion.tex:89（7 節第 3 段落）
- 現状: 4.4 節: "The weight path is consistent with the observation that the distance between corpora does not predict the gain"．7 節: "Entropy minimization relies on classes that form separate clusters in the unlabeled data, which holds on the improvised IEMOCAP sessions in AVT and AT but not on MELD, CMU-MOSEI, or the same target in VT and AV"
- 問題: 4.4 節の "the observation" に当たる数値も，使ったコーパス間の距離の定義も原稿にない．7 節では，クラスタがどれだけ分かれているかを測った量がなく，同じ IEMOCAP impro のターゲットで AVT・AT と VT・AV の結果が分かれる理由も示されていない．
- 求める対応: 距離と利得の相関を数値で示す．クラスの分かれ方の指標（例：未ラベルのプールでの真のクラスのシルエット値）を構成とターゲットごとに示すか，推測として書く．
- 著者のデータが必要か: 要

### [Minor-13] 「閾値が先」の主張は合併の学習でしか測っていない
- 該当箇所: 題名（main.tex:25）／sections/4-part1-what-few-labels-fix.tex:74–77（4.3 節）
- 現状: 閾値の割合は "the improvement of the pooled classifier over the source-only classifier" で測る．
- 問題: 合併の学習では，ターゲットの 4K ラベルは学習の重みのごく一部であり，式 (4) のとおり主に切片を動かす．Part II で使う仮説転移と繋留した生徒は，ラベルを別の仕方で使う．題名の "What Few Target Labels Fix First" を学習の仕方に依らない結論として読ませるなら，ほかの学習の仕方でも閾値の割合を示す必要がある．2 層パーセプトロンの確認も合併の学習の中にとどまる．
- 求める対応: 仮説転移と繋留した生徒での閾値の割合を加えるか，主張の範囲を合併の学習に限る．
- 著者のデータが必要か: 要

### [Minor-14] 図表が自足していない箇所がある
- 該当箇所と現状:
  1. 図 4b（4-part1-what-few-labels-fix.tex:85）と図 1（1-introduction.tex:21）は，閾値の $\mathbb{E}[o^2]=1/(2K)$（σ² を単位にした分散）と方向の $\tan^2\theta=q/K$（雑音と信号の比）を同じ軸 "Error of the estimate from $K$ labels" に描く．別の量であり，正解率に直すとそれぞれ $(d/4)\phi(d/2)$ と $(\Delta/4)\phi(\Delta/2)$ が掛かる．同じ軸で比べる意味を説明文に書くか，正解率の損失に直して描く．
  2. 図 6 と図 S6 の凡例の "AV world"，"AVT world" は本文で定義されていない．
  3. 表 2 の説明文 "(for A (HuBERT) the losses include all pairs)"：1 行だけ対の集合が違う．そろえるか理由を書く．
  4. 図 2a，図 3b，図 4a，図 6 に誤差の棒がなく，n（対の数，反復の数）も一部の図にしか書かれていない．
  5. 補足の PDF では，S3，S4，S6，S8 節の見出しの下に本文がなく，図表が別の節に流れ，参考文献が 2〜4 頁の途中に入る．図表を対応する節に置く．
- 求める対応: 上のとおり直す．
- 著者のデータが必要か: 不要

### [Minor-15] 中心化はプールのクラスの割合に依存するが，その影響が議論されていない
- 該当箇所: sections/5-part2-mechanism-to-method.tex:17（5.1 節）／sections/8-limitations-ethics.tex:107（8 節）／表 1
- 現状: "the target with the mean of its unlabeled pool, which removes most of the threshold loss"．限界の節は列の正規化についてだけ "we did not analyze how its gain depends on the class ratios" と書く．
- 問題: プールの平均はクラスの割合で変わる（表 1 の MELD は neutral が約 57%，CREMA-D はほぼ均等）．ソースとターゲットでクラスの割合が違うと，中心化はラベルの分布の違いを特徴のずれとして取り除き，閾値を別の方向にずらしうる．閾値の結果と手法の両方がこの操作に依存する．
- 求める対応: ターゲットのクラスの割合と中心化の利得の関係を示すか，限界の節に加える．
- 著者のデータが必要か: 要

### Optional

### [Optional-1] q，e，J を擬ターゲットで推定できるかを小さく試す
- 該当箇所: sections/9-conclusion.tex:119（9 節）
- 現状: "Because $q$, $e$, and the teacher's UAR advantage $J$ were measured with target labels, estimating them on pseudo-targets ... is the next step."
- 提案: 3 つの量はいずれもターゲットのラベルで測っているので，Part I の解析は実務での判断（ラベルを足すか，ソースを足すか，モダリティを足すか）にまだ使えない．擬ターゲットで推定した q，e，J と，ターゲットで測った値との順位相関だけでも示せば，TAC の読者にとっての意義がはっきりする．
- 著者のデータが必要か: 要

### [Optional-2] 記号と量の一覧表を本文か補足に置く
- 該当箇所: 3.5 節から 5.4 節まで全体
- 提案: Δ，d，o，q，e，n，λ，ρ，J，C，C′，C_b，ρ_M，γ，ρ_s，m と，それぞれを測るデータ（プール，テスト，話者の半分）と反復の数を 1 つの表にまとめると，読者が数値の出所を追える．本文は多くの量と実験を短い文に詰めており，現状では読み通すのが難しい．
- 著者のデータが必要か: 不要

### [Optional-3] コーパスの間で話者が重なっていないかを確かめる
- 該当箇所: sections/3-setting.tex:63（3.3 節）
- 現状: "the test speakers are also removed from the source, which matters only between the two IEMOCAP groups."
- 提案: AFEW は映画のクリップ，MELD はテレビドラマ Friends から作られており，同じ俳優が両方に現れる可能性がある．影響は小さいと考えられるが，"matters only between the two IEMOCAP groups" と書くなら確認した方法を添える．
- 著者のデータが必要か: 要

### [Optional-4] 一般化蒸留と xMUDA との違いを方法の節で表にする
- 該当箇所: sections/2-related-work.tex:58–68／sections/5-part2-mechanism-to-method.tex:79–91
- 提案: 提案手法は，教師がソースだけで学習されること，弱ラベルの列を正規化すること，重みと切片に別の繋留を置くこと，K ごとに教師と m を選ぶことで一般化蒸留［11］や xMUDA［41］と異なる．どの点が違うかを方法の節で 1 つの表にし，Major-5 の除去実験の行と対応させると，新しさの主張と証拠の対応が読み取りやすくなる．
- 著者のデータが必要か: 不要
