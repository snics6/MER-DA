**検証の結果（2026-11-12，`verify_emotion.md`）**：Amanda（Findings ACL 2024）で外したのは表現をそろえる項（CA）だけで，比べは「そのまま移す → LA だけ → LA＋CA」の入れ子の 3 段．「LA と CA を別々に外した」は誤り．EMO-TTA の w_y・b_y の式は論文の式 12 にある．

# 声（SER）と融合（MER）で，誤りを切片側と向き側に分けて情報源ごとに測った研究はあるか（2026-11-11 のウェブ調べ直し）

## 要約（判定）

- **完全一致は見つからなかった．**声と融合の感情認識の実データで，分類器の誤りを切片側（出力の偏り・閾値・クラスの割合）と向き側（重みの向き・表現）に分け，4 つの情報源のどれがどちらを減らしたかを測った研究は，58 回の検索と 15 本の本文確認の範囲では無い．
- **部分一致は 4 本．**最も近いのは Tran, Yin, Soleymani（Interspeech 2023）である．話者ごとに「特徴のずれ」と「ラベルのずれ」を別々に測って成績との相関を出し，出力の平均だけをずらす補正（切片の当て直しに当たる）と表現を変える学習を別の手として比べた．ただし回帰（感情の強さを連続値で当てる課題）であり，分類器の誤りを (a)(b) に分けた量は測っていない．
- 残りの部分一致 3 本は次のとおり．Amanda（ACL Findings 2024．3 モダリティ）は，予測のクラスの割合を合わせる項と表現を合わせる項を外す実験をした．Naini ほか（ICASSP 2025）は，ラベルの割合だけを合わせたときの得を，ターゲットのラベルを使う場合と疑似ラベルを使う場合とで比べた．EMO-TTA（2025）は，クラスの平均の更新と共分散の更新を外す実験をした．どれも情報源は 1〜2 個で，比べたのは正解率か CCC（予測と正解の一致の度合い）の合計だけである．
- 多くの研究は，「コーパスが変わると分類器が偏る」「決定の境界がずれる」と書く（Shah 2015，Albanie 2018 など）．しかし，その偏りの量を分けて測った研究は無い．SER Evals（2024）は，テストの時点でクラスの割合の違いに合わせて出力を当て直す（ロジット調整）手続きを評価に組み込んでいるのに，その補正の有無で成績を比べていない．
- 声と融合の分野に限ると，(a)(b) の分け方を予言する理論は見つからなかった．
- 狙うべき差分は「同じ物差し（切片側／向き側）で 4 つの情報源を並べること」である．MER-DA の記録（`DAI_paper/inventory.md`）では，この物差しで 4 つの情報源の量が既に出ている．
- **限界**：この会話の全担当で共有する検索の上限（200 回）に達したため，最後の 2 つの検索語は実行できなかった（6 節）．MSP 研究室のサーバー（msp.utdallas.edu）と Springer と CVF の本文は，名前解決の失敗か 403（アクセス拒否）で読めなかった．

---

## 1. 言葉の定義と記号

- **切片側の誤り (a)**：重みはそのままで，切片（感情ごとの点数の底上げ），閾値，クラスの割合を当て直せば取り戻せる誤り．
- **向き側の誤り (b)**：切片をどう当て直しても残る誤り．重みの向きや表現のずれから来る．
- **情報源**：① ターゲットの少数のラベル，② ほかのコーパスのラベル付きデータを混ぜる，③ ターゲットのラベル無しデータ（正規化，平均を引く，TTA，DA，自己学習，ラベルシフトの補正），④ ほかのモダリティの予測（蒸留，疑似ラベル）．
- **TTA（テスト時適応）**：テストの時点で，ラベル無しのテストデータだけを使ってモデルを直す手法．
- **DA（ドメイン適応）**：学習に使ったデータ（ソース）と違う性質のデータ（ターゲット）に合わせてモデルを直す手法．
- **ラベルシフト**：クラスの割合だけが変わり，各クラスの中の特徴の分布は変わらないずれ．
- **疑似ラベル**：モデルの予測を正解の代わりに使うラベル．
- **蒸留**：別のモデル（教師）の出力を目標にして，モデル（生徒）を学ばせる手法．
- **CCC**：連続値の予測と正解の一致の度合い．相関（ピアソンの相関係数）と，平均と広がりのずれを掛け合わせた量である．
- **検証の水準**：◎ 本文の該当箇所を読んだ，○ 抄録か信頼できる二次資料で読んだ，△ 検索の断片だけ．
- **判定**：完全一致，部分一致，関係（問いの一部に関わる），該当しない（正解率・UAR などの合計だけを比べた）．

---

## 2. 研究ごとの表

### 2.1 部分一致（4 本）

| 研究 | 設定 | 情報源 | 測った量 | (a)(b) を分けたか | 水準 | URL |
|---|---|---|---|---|---|---|
| Tran, Yin, Soleymani. Interspeech 2023「Personalized Adaptation with Pre-trained Speech Encoders for Continuous Emotion Recognition」 | 声．MSP-Podcast．覚醒度と感情価の回帰．HuBERT-large を微調整 | ③（ターゲットの話者のラベル無し音声で事前学習を続ける）と②（似た学習話者のラベルの平均と標準偏差） | 話者ごとに「特徴のずれ」と「ラベルのずれ」を KL（2 つの分布の違いの量）で測り，話者ごとの CCC との相関を出した．覚醒度で −0.714 と −0.502，2 つのずれの相関は 0.285．補正は 2 つの手を別に比べた．1 つは表現を変える手（話者の埋め込みを足して事前学習を続ける）．もう 1 つは出力だけを直す手で，予測の平均だけをずらす（μ），広がりだけを変える（σ），両方を変える（μ, σ）の 3 通りを試した．覚醒度の話者平均の CCC は，補正なし 0.531，平均だけ 0.528，広がりだけ 0.541 だった．著者は「平均をずらすと CCC が下がることが多い」と書いた | 部分的に分けた．「ラベルのずれ」と「特徴のずれ」を別々に測り，出力の平均だけをずらす手（切片に当たる）を表現を変える手と分けて比べた．ただし回帰であり，分類器の誤りの分解ではない．情報源ごとに (a)(b) のどちらを減らしたかは測っていない | ◎（2〜5 節と表 2・3） | https://arxiv.org/abs/2309.02418 |
| Amanda（Findings of ACL 2024．Adaptively modality-balanced domain adaptation for MER） | 声＋顔＋言葉．IEMOCAP，MELD，MOSEI，MSP-IMPROV の 7 組．4 クラス．ラベル無しの DA．指標は重み付き F1 | ③ | 2 つの項を外して比べた．1 つは予測したターゲットのクラスの割合をソースの割合に合わせる項（ソースとターゲットでラベルの割合が同じと仮定する）．もう 1 つは表現を合わせる項（CORAL．特徴の共分散を合わせる手法）．図 2 で「表現を合わせる項を外した版」が「そのまま移した版」より大きく上がることを示した．数値は図だけで，表には無い | 部分的．割合を合わせる項と表現を合わせる項の得を分けて見せた．しかし割合を合わせる項もネットワーク全体を学び直すので，切片だけの補正ではない．誤りの量は測っていない | ◎（3.1 節 C，4.2 節，表 1，図 2） | https://aclanthology.org/2024.findings-acl.859.pdf |
| Naini, Robinson, Richerson, Busso. ICASSP 2025「Domain-Specific Adaptation in SER Using Emotional Distribution Alignment」 | 声．MSP-Podcast から VAM，RECOLA，WHiSER へ．WavLM-large を微調整してから凍結し，その上の分類器を学ぶ．覚醒度と感情価の回帰 | ①（ターゲットのラベルで分布を推定）と③（疑似ラベルで推定．反復版あり） | ソースを間引いて，ターゲットのラベルの分布に合わせてから学び直した．間引かない場合と比べた CCC の上がり幅は，ラベルで推定した場合が覚醒度で約 3.0〜4.3%，感情価で約 3.2〜21%．疑似ラベルで推定した場合はそれより少し小さかった | 分けていない．ラベルの割合（(a) に近い量）だけを合わせたときの得は測った．しかし分類器全体を学び直すので，切片だけの効果ではない．向き側は測っていない | ◎（3 節と 5 節） | http://www.lab-msp.com/MSP/publications/Naini_2025.pdf |
| EMO-TTA（arXiv 2509.25495, 2025） | 声．CLAP（音声と文章を対応づけるモデル）の埋め込みの上で，感情ごとにガウス分布（共分散は全クラス共通）を置く．テストの発話 1 本ずつで EM（推定を繰り返す手続き）を回し，平均，共分散，クラスの割合を更新する．IEMOCAP，CREMA-D，RAVDESS ほか | ③（TTA）と④に近いもの（文章側の試作の点（プロトタイプ）を初期値に使う） | 更新を外す実験をした．平均の更新を外すと平均で −1.65 点，共分散の更新を外すと −6.41 点（正解率）．クラスの割合の更新は外していない | 分けていない．共分散が共通のガウス判別では，平均を更新すると重みと切片の両方が動く（推測）ので，平均と共分散の分け方は (a)(b) の分け方と一致しない．混同行列もクラスごとの成績も無い | ◎（2.3〜2.4 節，表 3） | https://arxiv.org/html/2509.25495v1 |

### 2.2 関係（問いの一部に関わるが，分けていない）

| 研究 | 設定 | 情報源 | 関わる点 | (a)(b) | 水準 | URL |
|---|---|---|---|---|---|---|
| SER Evals（Osman, Kaplan, Nadeem. arXiv 2408.07851, 2024） | 声．多言語の多数のコーパス．凍結した基盤モデル＋MLP．コーパス外の評価 | 評価の時点での割合の補正（ターゲットの割合を使う） | 「訓練とテストの分布の違いに基づいてロジットを調整する」とあり，(a) の補正を評価の手続きに入れている．補正の有無で成績を比べた表は無い．式も無い | 分けていない | ◎（3.4・4.2〜4.4 節） | https://arxiv.org/html/2408.07851v1 |
| Shah, Chakrabarti, Spanias. EURASIP JASMP 2015「Within and cross-corpus SER using latent topic model-based features」 | 声．IEMOCAP↔SEMAINE．覚醒度と感情価の 2 値 | ①（ターゲットのラベル付きの発話で重みを推定し，それを手本にソースで学び直す） | 「アノテーションの違いが分類器を偏らせ，学習コーパスで学んだ決定の境界がテストコーパスでは最適でなくなる」と書く．補正は 2 つ．ソースの発話を選ぶ手と，重みを手本の重みに近づける正則化である | 分けていない．「偏り」は p(y|特徴) のずれ全体を指し，切片と向きを区別していない | ◎（4.4 節，13〜14 ページ） | https://sensip.engineering.asu.edu/wp-content/uploads/sites/68/2020/03/Within-and-cross-corpus-speech-emotion-recognition-using-latent-topic-model-based-features.pdf |
| Albanie, Nagrani, Vedaldi, Zisserman. ACM MM 2018 | 顔の教師から声の生徒へ蒸留．VoxCeleb．RML，eNTERFACE で評価 | ④ | 生徒の予測は平静に偏ると書く（数値なし）．外部のコーパスで評価するときは，行列と切片からなる変換をターゲットのラベルで学ぶ | 分けていない | ◎（ar5iv の 3.1・5 節） | https://ar5iv.labs.arxiv.org/html/1808.05561 |
| Busso, Metallinou, Narayanan（ICASSP 2011）と Busso ほか（IEEE TAC 2013）．IFN（繰り返して平静の発話を選び，それで正規化する手法） | 声．IEMOCAP ほか | ③（話者ごとの正規化） | 話者の全発話で正規化すると，話者の感情の割合が偏っている場合に正規化そのものが偏る．そのため平静と判定した発話だけで正規化の値を推定する．この手法は，正規化がクラスの割合の違いと話者のずれを一緒に消してしまう問題を扱っている | 分けていない．成績の合計だけを報告 | ○（抄録，ポスター，特許の記述．MSP のサーバーは読めなかった） | https://msp.utdallas.edu/publications/Busso_2013_2.pdf |
| Braunschweiler ほか（Toshiba．ASRU 2021 と arXiv 2201.03511） | 声．4 クラス．複数コーパス | ② | 音響のずれを拡張で埋めても効果は小さく，ラベルを一部入れ替える実験から，ずれは音響より深いところ（ラベルの付け方）にあると述べる | 分けていない | ○（検索の抜き書き．arXiv の版は読めた範囲が 1 ページで不完全） | https://arxiv.org/abs/2201.03511 |
| Huo ほか．C2KD（CVPR 2024） | 声と顔ほか．教師と生徒が違うモダリティの蒸留 | ④ | 「モダリティの不均衡」と「教師と生徒の柔らかいラベル（確率の分布）のずれ」が蒸留の効かない理由だと実験で示した．蒸留の対象を正解以外のクラスに限る | 分けていない．分けているのは正解クラスと他のクラスで，切片と向きではない | ○（抄録．CVF の本文は 403） | https://openaccess.thecvf.com/content/CVPR2024/html/Huo_C2KD_Bridging_the_Modality_Gap_for_Cross-Modal_Knowledge_Distillation_CVPR_2024_paper.html |
| Xue ほか．Modality Focusing Hypothesis（ICLR 2023） | モダリティを問わない理論と 6 つのデータ | ④ | モダリティ間の蒸留の得は，教師が持つ「生徒のモダリティからも見える決め手の特徴」の割合で決まるという仮説 | 予言しない．(a)(b) の区別は無い | ○（抄録と二次資料） | https://arxiv.org/abs/2206.06487 |
| Chou ほか．Interspeech 2023（感情ごとの温度で確率を較正する） | 声．MSP-Podcast．コーパス内 | 較正（確率の当て直し） | 感情ごとに温度（確率の尖り具合を変える係数）を学び，予測の確率と実際の正解率のずれを 15〜20% 減らした．コーパスはまたいでいない | 分けていない | ○（抄録） | https://www.isca-archive.org/interspeech_2023/chou23_interspeech.html |
| Emmanouilidou, Gamper. JASA 157, 2025 | 声．wav2vec，VGGish，WavLM，CLAP．複数コーパスでの学習 | ② | 大きな事前学習モデルではクラスの表現がコーパスごとに分かれると述べる（表現側の問題としての記述） | 分けていない | ○（抄録） | https://www.microsoft.com/en-us/research/?p=1155873 |
| Breaking Neutral Bias（Interspeech 2026，lu26c） | 声．音声と言葉を扱う大規模モデル | なし（零ショット） | 大規模な音声言語モデルが平静に偏ると述べる | 分けていない | △（検索の断片） | https://papers.cool/venue/lu26c@interspeech_2026@ISCA |

### 2.3 該当しない（合計の成績だけ．本文で確かめたもの）

| 研究 | 設定 | 情報源 | 確かめた点 | 水準 | URL |
|---|---|---|---|---|---|
| Dong ほか．ICASSP 2026「Test-Time Adaptation for SER」 | 声．wav2vec2．11 の TTA 手法 | ③ | 正解率と macro-F1 だけ．クラスごとの成績も混同行列もラベルシフトの分析も無い．LAME は出力の確率だけを変え，T3A は分類器の試作の点を作り直すが，その効果の内訳は測っていない | ◎ | https://arxiv.org/html/2601.16240 |
| Schuller ほか．IEEE TAC 2010（ACII 2015 の拡張要旨） | 声．6 コーパス | ③（話者・コーパス・両方の正規化，正規化なし） | UAR と WAR，最良からの距離だけ | ◎ | https://infoscience.epfl.ch/record/210631/files/Schuller_ACII2015_2015.pdf |
| Pepino, Riera, Ferrer. Interspeech 2021 | 声．凍結 wav2vec2．IEMOCAP，RAVDESS | ③（話者ごとの正規化） | 話者ごとの正規化は全体の正規化より良い．理由は「話者の情報の一部を除き，感情に集中させる」という説明だけ | ◎ | https://arxiv.org/abs/2104.03502 |
| Gat ほか．ICASSP 2022 | 声．HuBERT | ③（敵対的に話者の情報を消す） | 話者を当てる線形分類器の正解率が 60.7% から 45.9% に下がった．表現の側の確認だけで，感情の分類器の切片は見ていない | ◎ | https://ar5iv.labs.arxiv.org/html/2202.01252 |
| Milner ほか（ASRU 2019．arXiv 2207.02104） | 声．7 感情．多コーパス | ①（ターゲットで微調整），② | 感情ごとの UA・WA の表はある．微調整で UA が平均 +8.9．何を直したかの分析は無い | ◎ | https://arxiv.org/html/2207.02104 |
| Neumann, Vu. ICASSP 2018 | 声．英語↔フランス語 | ①（ターゲットの 500 本で微調整） | UAR だけ | ◎ | https://arxiv.org/abs/1803.00357 |
| Xi ほか．APSIPA 2019（Speaker to Emotion） | 声．話者認識で事前学習した ResNet | ① | 最上段の全結合層だけを学び直す手，全体の微調整，アダプタを比べた．正解率だけ | ◎（ざっと） | https://apsipa.org/proceedings/2019/pdfs/93.pdf |
| Hassan, Damper, Niranjan. IEEE TASLP 2013 | 声．FAU Aibo | ③（重要度重み付け．共変量シフトとして扱う） | ラベルシフトは扱わない | ○ | https://eprints.soton.ac.uk/350948 |
| Kaya, Karpov. Neurocomputing 2018 | 声．5 言語 | ③（正規化の多段の組み合わせ） | 成績の合計だけ | ○ | https://research-portal.uu.nl/en/publications/efficient-and-effective-strategies-for-cross-corpus-acoustic-emot/ |
| Abdelwahab, Busso. ICASSP 2015 | 声．SVM の教師ありの DA | ① | ラベル付きの話者を増やしても大きく上がらない．本文は読めなかった | △ | https://msp.utdallas.edu/publications/Abdelwahab_2015.pdf |
| Zhou, Chen. ICASSP 2019（クラスごとの敵対的 DA） | 声．EMO-DB↔Aibo．正負の 2 値 | ① | クラスごとに分布を合わせる．成績の合計だけ | ○ | https://arxiv.org/abs/1810.12782 |
| Tavernor, Mower Provost. arXiv 2509.12295, 2025 | 声．評定者ごとの分類器の頭（出力層）を少数の発話で選ぶ | ① | 評定者 1 人あたり 15〜20 本で全データに近い成績．内訳の分析は無い | ○ | https://arxiv.org/abs/2509.12295 |
| Ryumina ほか．Pattern Recognition Letters 190, 2025 | 融合（声＋顔＋言葉）．コーパスを 1 つ抜く評価 | ② | UAR だけ | ○ | https://hci.nw.ru/en/news/61 |
| Lashkarashvili ほか．ICASSP 2024 | 声．LoRA などの少数パラメータの微調整 | ① | 正解率だけ．切片だけを学ぶ手（BitFit）は試していない | ○（記録の◎を引き継ぐ） | https://arxiv.org/abs/2402.11747 |

手元の記録で既に「×」と判定されていた声と融合の研究（Chen & Yu 2023，Feng & Chaspari，Ahn 2021，CTIAL，EmoBox，Zhang 2021 KBS，LanSER，LiSER，Hajavi，M4SER，MDAT，Li 2021，MKE，UMT，G²D，DAMR，MERBench，TelME，MST-Distill，ACL-MER，MER-SEMI）は今回読み直していない．今回の検索で，これらの判定を覆す記述は出てこなかった．

---

## 3. 最も近い先行研究 5 本と，MER-DA の問いとの違い

### 3.1 Tran, Yin, Soleymani（Interspeech 2023）

- **同じ点**：成績が落ちる原因を「ラベルのずれ」と「特徴のずれ」の 2 つに分け，それぞれを別々に測った．出力の平均だけをずらす補正（切片の当て直しに当たる）と表現を変える学習を，別の手として比べた．
- **違う点**：
  - 課題が回帰（覚醒度と感情価）であり，4 クラスの分類ではない．
  - 2 つのずれは入力とラベルの分布の違い（KL）で測っており，分類器の誤りを切片側と向き側に分けた量ではない．
  - 単位は話者で，コーパス間ではない．
  - 情報源はラベル無しの音声（③）と学習話者のラベルの統計（②に近い）の 2 つで，少数のラベル（①）とほかのモダリティ（④）は無い．
  - 平均をずらすと成績が下がることが多かったと報告している．MER-DA の記録では，ターゲットの平均を引く手は 176 組の 95% で損をしない．ただし，平均を引く対象が違う（Tran らは出力，MER-DA は特徴）ので，そのまま比べられない．

### 3.2 Amanda（Findings of ACL 2024）

- **同じ点**：3 モダリティの MER で，「予測のクラスの割合を合わせる項」と「表現を合わせる項」を別々に外して得を見た．IEMOCAP，MELD，MOSEI を含み，4 クラスで，MER-DA のコーパスと重なる．
- **違う点**：
  - 割合を合わせる項もネットワーク全体を学び直すので，切片だけの効果として切り出していない．
  - 割合が同じという仮定を置き，その仮定のもとでターゲットの予測の割合をソースに合わせている．
  - 結果は図だけで，誤りの量は無い．
  - 情報源は③だけである．

### 3.3 Naini ほか（ICASSP 2025）

- **同じ点**：凍結した特徴の上で，ラベルの割合（(a) に近い量）だけを合わせたときの得を測った．割合をターゲットのラベル（①）で推定する場合と疑似ラベルで推定する場合を比べた．
- **違う点**：
  - 回帰である．
  - 割合を合わせる手段がソースの間引きで，分類器全体を学び直す．
  - 向き側は測っていない．
  - ほかのモダリティは使っていない．

### 3.4 EMO-TTA（2025）

- **同じ点**：凍結した埋め込みの上で，ラベル無しのデータ（③）から感情ごとの平均・共分散・割合を推定し直す．その更新を外す実験で得の内訳を見た．
- **違う点**：
  - 外したのは平均と共分散で，割合の更新は外していない．
  - 共分散が共通のガウス判別では，平均の更新が重みと切片の両方を動かす（推測）．このため，この内訳は切片側と向き側の分け方と一致しない．
  - 指標は正解率だけである．

### 3.5 Shah ほか（EURASIP JASMP 2015）

- **同じ点**：コーパス間の劣化を「アノテーションの違いによる分類器の偏り」とみなし，少数のターゲットのラベル（①）で決定の境界を直した．
- **違う点**：
  - 「偏り」は p(y|特徴) のずれ全体を指し，切片と向きを区別していない．
  - 直す手（重みを手本に近づける正則化）は重みと切片を一緒に動かす．
  - 分類は 2 値である．

---

## 4. 狙うべき差分（記録で既に測れているもの）

- **先行研究が測っていない量**：1 つの分類器の誤りのうち，切片を当て直せば消える分（a）と残る分（b）．さらに，情報源ごとに a と b のどちらが何点減ったか．先行研究は，切片側の補正（ロジット調整，割合の整合，出力の平均のずらし）か表現側の補正（正規化，DA，蒸留）のどちらか一方を手法として入れ，その効果を合計の成績で示しただけである．
- **先行研究が置いていない設定**：4 つの情報源（少数のラベル，ソースを混ぜる，ラベル無しのデータ，ほかのモダリティの予測）を，同じ分類器の同じ物差しで並べる設定．声と融合の分野では，今回の範囲で 1 本も無い．
- **MER-DA の記録で既に出ている量**（`DAI_paper/inventory.md`．一部は検査前）：
  - 少数のラベルは切片を安く直す．K=5 で，向きを固定して切片だけを直す手の上がり幅は，ソースと混ぜて学び直す手の 0.91 倍（声）である．
  - 少数のラベルは向きも直すが遅い．声では K=10 でも向き側の誤りの 6〜7 割が残る．
  - ソースを K 本と混ぜて学び直すと，K が大きいときに損をする（声，K=25 で −1.69）．
  - 同じモダリティのラベル無しデータは位置を直す（平均を引く手は 176 組の 95% で損をしない）．しかし向きは直さない．ラベル無しの DA・半教師あり学習の 8 手法で，向き側の誤りを減らした単位は 210 のうち 7（偶然の見込みは 5.25），増やした単位は 79 である．
  - 配備する分類器が見ないモダリティを含むモデルの疑似ラベルは向きを直す．81 単位のうち減らした単位は 38，増やした単位は 1 である．
  - 先行研究の部分一致 4 本は，どれもこの表の 1 行か 2 行にしか相当しない．
- **先行研究の結果との接点**：
  - Tran ほかの「出力の平均だけをずらすと下がることが多い」という結果は，MER-DA の「ラベル無しで平均を引く手は損をしない」と方向が逆に見える．ただし対象（出力か特徴か），課題（回帰か分類か），単位（話者かコーパスか）が違う．論文で引くなら，この違いを明記する必要がある．
  - Amanda の「割合を合わせる項だけで大きく上がる」という結果は，「ラベル無しは位置を直す」と同じ向きの観察として引ける．

---

## 5. 検索語の記録

すべて英語で検索した．「見つかった」は判定に使った研究，「無かった」はその検索語で狙ったが出てこなかったもの．

| # | 検索語 | 見つかったもの | 無かったもの |
|---|---|---|---|
| 1 | cross-corpus speech emotion recognition bias calibration analysis class prior | ADDoG，Emo-DNA，DSNet，Shah 2015 | コーパス間でクラスの割合の補正を分析した研究 |
| 2 | speaker normalization speech emotion recognition what does it correct bias versus representation analysis | Gat 2022，話者正規化の特許 | 正規化が偏りを直すか表現を直すかを分けた研究 |
| 3 | label shift speech emotion recognition prior adaptation EM Saerens cross-corpus | Zhou & Chen 2019，Emo-DNA，Dong 2026 | SER に EM の割合補正を当てた研究 |
| 4 | threshold adjustment cross-corpus emotion recognition decision threshold recalibration target data | Emo-DNA の疑似ラベルの閾値，Palotti 2023（MOSEI のラベルを 2 値にする閾値），ABAW の GLA | ターゲットで決定の閾値を当て直す研究 |
| 5 | "cross-corpus" emotion recognition "neutral" bias confusion matrix over-predicts analysis corpus mismatch | THAI-SER，Braunschweiler，Schuller | コーパス間で平静に偏る量を測った研究 |
| 6 | "prior probability" shift speech emotion recognition class distribution mismatch training test corpora | EMO-TTA，IS25 チャレンジ | 割合のずれとして SER を扱った研究 |
| 7 | BitFit bias-only fine-tuning speech emotion recognition cross-corpus adaptation | Lashkarashvili 2024，PEFT-SER | SER で切片だけを学ぶ微調整 |
| 8 | audio language model zero-shot speech emotion recognition calibration label bias neutral prior correction | Breaking Neutral Bias（IS2026），EMO-TTA，Chou 2023 | 音声言語モデルの感情の割合補正 |
| 9 | test-time adaptation speech emotion recognition batch normalization statistics label shift analysis per-class | Dong 2026，EMO-TTA | SER の TTA でクラスごとの分析 |
| 10 | cross-corpus speech emotion recognition "last layer" OR "classifier only" retraining few target samples versus full fine-tuning | Tavernor 2025，次元の SER の部分微調整 | 最終層だけと全体の比較（コーパス間） |
| 11 | Busso iterative feature normalization emotion detection speaker variability neutral reference model | Busso 2011・2013 の IFN | — |
| 12 | Abdelwahab Busso supervised domain adaptation emotion recognition from speech few labeled target samples | Abdelwahab 2015・2017 | 本文（サーバーに届かない） |
| 13 | "speech emotion recognition" "logit adjustment" OR "prior correction" domain shift unlabeled target | SER Evals | ラベル無しのターゲットで割合を推定してロジットを直す SER の研究 |
| 14 | multimodal emotion recognition cross-dataset domain shift label distribution analysis audio visual text | Amanda，DGDA，ACL 2025 の部分的なずれ | MER のコーパス間でラベルの割合のずれを分析した研究 |
| 15 | cross-modal distillation emotion recognition what is transferred analysis teacher student audio visual | LiSER，Albanie，TelME，Aslam 2023，CDGKD | 何が伝わったかの体系的な比較 |
| 16 | linear probe self-supervised speech emotion few-shot cross-corpus frozen wav2vec2 logistic regression analysis | Naini 2024，Chen & Yu 2023，APSIPA 2025 | 凍結特徴＋ロジスティック回帰の少数ラベルのコーパス間の研究 |
| 17 | Neumann Vu cross-lingual multilingual speech emotion recognition English French adaptation small amount target data layers | Neumann & Vu 2018 | — |
| 18 | Gerczuk EmoNet multi-corpus speech emotion residual adapters transfer learning linear head comparison | EmoNet | — |
| 19 | Naini Busso domain adaptation speech emotion matching emotional label distribution target resampling arXiv | Naini 2023，Abdelwahab 2018 | （Naini 2025 は #34 で見つけた） |
| 20 | speech emotion recognition uncertainty calibration out-of-domain corpus ECE temperature scaling cross-corpus | Chou 2023，Jia 2025，2406.18065 | コーパス外での較正の研究 |
| 21 | modality focusing hypothesis cross-modal knowledge distillation when does it work modality-general decisive features | Xue 2023 | — |
| 22 | speech emotion recognition fairness post-processing group-specific threshold gender bias correction analysis | 公平性の研究群 | SER で群ごとの閾値を後から直す研究 |
| 23 | Schuller cross-corpus acoustic emotion recognition variances and strategies speaker normalization corpus normalization results | Schuller 2010・2015 | — |
| 24 | Kaya Karpov efficient effective strategies cross-corpus acoustic emotion recognition normalization | Kaya & Karpov 2018 | — |
| 25 | Hassan Damper Niranjan acoustic emotion recognition compensating for covariate shift importance weighting | Hassan 2013 | — |
| 26 | cross-corpus emotion recognition "chance level" "mostly predicts" OR "majority of predictions" one class analysis per-class recall degradation | 中国語の診療の小さな評価，Schuller | コーパス間で 1 クラスに偏る現象を測った研究 |
| 27 | "Within and cross-corpus speech emotion recognition using latent topic model-based features" instance selection weight regularization | Shah 2015（ASU の PDF） | — |
| 28 | personalized speech emotion recognition speaker-specific bias term adaptation unlabeled speaker data | Tran 2023，Sridhar & Busso | 話者ごとの切片を学ぶ SER |
| 29 | few-shot speech emotion recognition new corpus calibrate classifier bias prototypes analysis error sources | Meta-PerSER，F-MAML，Chen & Yu | 少数ラベルの SER で誤りの源を分析した研究 |
| 30 | semi-supervised speech emotion recognition pseudo-label class imbalance distribution alignment bias toward majority class analysis | Agarla 2023，Semi-FedSER，MER2024-SEMI の参加者 | 疑似ラベルが多数のクラスに偏る量を測った SER の研究 |
| 31 | Zhou Chen transferable positive negative speech emotion class-wise adversarial domain adaptation label distribution | Zhou & Chen 2019 | — |
| 32 | emotion recognition in conversation cross-dataset IEMOCAP MELD label distribution shift prior correction transfer | DGDA，SCoPE（感情の移り変わりで，分布のずれではない） | IEMOCAP↔MELD の割合の補正 |
| 33 | "emotion recognition" adapting "output layer" OR "softmax layer" only target corpus versus fine-tuning entire network cross-corpus comparison | Xi 2019（APSIPA），Lashkarashvili 2024 | 出力層だけと全体の比較の数値 |
| 34 | "speech emotion" "label shift" OR "prior shift" OR "target shift" adaptation 2024 2025（詳しい検索） | Naini 2025，Tran 2023，EMO-TTA | — |
| 35 | cross-modal knowledge distillation emotion student gains explained class prior soft labels dark knowledge analysis face to voice | Albanie，C2KD | 顔から声への蒸留で，生徒が教師の割合を受け継ぐ量を測った研究 |
| 36 | C2KD bridging modality gap cross-modal knowledge distillation CVPR 2024 analysis logits mismatch | C2KD | — |
| 37 | multimodal emotion recognition domain adaptation which modality shifts more analysis per-modality degradation cross-corpus audio visual | Amanda，Boomda，NLPCC 2023 | モダリティごとの劣化を量で比べた研究 |
| 38 | decomposing domain shift error emotion recognition covariate shift versus label shift contribution measured | Hassan，一般の ML（sparse joint shift，2303.02011） | 感情認識で 2 種類のずれの寄与を測った研究 |
| 39 | "emotion" classifier "bias term" OR "intercept" recalibration domain adaptation unlabeled target mean shift frozen embeddings | EMO-TTA | 凍結した埋め込みで切片だけを当て直す感情の研究 |
| 40 | cross-corpus speech emotion recognition feature normalization per corpus mean subtraction self-supervised embeddings wav2vec HuBERT | Pastor 2022・2023，Wagner 2023，Emmanouilidou 2025 | SSL（自己教師あり学習）の特徴でコーパスごとに平均を引く研究 |
| 41 | "Exploring cross-dataset generalization of speech emotion recognition models" | Emmanouilidou & Gamper 2025，Ibrahim 2024 | — |
| 42 | multimodal large language model emotion recognition prediction bias toward neutral class calibration debiasing zero-shot MER | 表情の VLM（画像と言葉を扱うモデル）の長い裾の偏りの論文，MLLM と MER のサーベイ（文脈の較正の紹介） | MER で平静への偏りを較正した研究 |
| 43 | CLAP zero-shot speech emotion recognition prompt calibration class prior bias improvement | EMO-TTA，ZS-Fuse，CLEP-DG，ParaCLAP | CLAP の零ショットで割合を補正した研究 |
| 44 | Ryumina 2025 multi-corpus audio-visual emotion recognition cross-corpus RAVDESS CREMA-D MELD AFEW generalization | Ryumina 2025（PRL） | — |
| 45 | Saerens Latinne Decaestecker prior adjustment applied emotion recognition audio unlabeled test priors | Saerens 2002，Esuli 2020 の再検討 | 感情・音声への適用 |
| 46 | few-shot speech emotion recognition prototypical network cross-corpus distribution calibration Tukey analysis | Ion 2024，Meta-PerSER | 少数ラベルの分布の較正を SER に当てた研究 |
| 47 | multimodal sentiment analysis cross-dataset MOSI MOSEI transfer threshold shift label distribution analysis | CASP，GD-Adapt，Senti-iFusion | 感情極性のコーパス間で閾値のずれを測った研究 |
| 48 | disentangling sources of cross-corpus degradation speech emotion recognition annotation mismatch versus acoustic mismatch controlled study | Braunschweiler 2021，Upadhyay 2024 | アノテーションと音響のずれを交差させた実験 |
| 49 | speech emotion recognition embeddings corpus identity linearly decodable dataset bias direction removal cross-corpus | SISER，話者の情報の層ごとの研究 | コーパスの向きを消して転移を測った研究 |
| 50 | "More Similar than Dissimilar" modeling annotators cross-corpus speech emotion recognition enrollment | Tavernor 2025 | — |
| 51 | "Domain adaptation for speech emotion recognition by sharing priors between related source and target classes" | Mao 2016 の書誌だけ | 抄録と本文 |
| 52 | speech emotion recognition "unsupervised" adaptation "mean" "variance" normalization target corpus batch statistics AdaBN analysis what it fixes | Cai 2020（DANN と BN） | SER で AdaBN が何を直すかの分析 |
| 53 | speech emotion recognition domain shift analysis: how much performance recovered by recalibrating classifier bias or class priors versus retraining features（詳しい検索） | Xi 2019，Marczewski | 切片・割合の当て直しで取り戻せる量 |
| 54 | multimodal emotion recognition cross-modal pseudo-label teacher student gain analysis which errors corrected per-class bias distribution（詳しい検索） | UF-AMA，SUMMER，条件つき教師と生徒 | 疑似ラベルの得の内訳 |
| 55 | multimodal emotion recognition late fusion calibration modality-specific bias logits domain shift reliability weighting analysis | CMC 2025，信頼度で重み付けする融合の研究群 | ずれのもとでのモダリティごとの切片の偏り |
| 56 | "emotion" "label shift" "black box shift estimation" OR "BBSE" OR "MLLS" audio OR speech OR multimodal | Lipton 2018，Garg 2020（一般） | 感情への適用 |
| 57 | cross-corpus arousal valence prediction CCC versus Pearson correlation gap mean offset scale bias post-hoc rescaling | CCC＝相関×ずれの補正係数という関係，Chou の敵対的手法 | 平均と広がりを後から直してコーパス間の CCC の差が縮むかを測った研究 |
| 58 | Wagner dawn of the transformer era speech emotion recognition robustness cross-corpus IEMOCAP MOSI CCC drop analysis bias | Wagner TPAMI 2023 | コーパス間の CCC の低下の内訳（本文を grep して無いことを確かめた） |
| 59 | what causes cross-corpus speech emotion recognition failure analysis study label definition speaker recording factors Interspeech 2024 2025 | **実行できず（上限）** | — |
| 60 | "speech emotion" few labeled target samples "bias" adaptation "only the bias" OR "bias-only" OR "intercept only" classifier | **実行できず（上限）** | — |

---

## 6. 見つからなかったこと

次の研究は，今回の検索では見つからなかった．

1. 声・融合の感情認識で，1 つの分類器の誤りを「切片を当て直せば消える分」と「残る分」に分けて量で示した研究．
2. 少数のターゲットのラベルを切片（または閾値，クラスの割合）だけに使う手と，全体を学び直す手を，コーパス間の SER・MER で比べた研究．Albanie 2018 は行列と切片の両方を学び，Xi 2019 は最上段の層を丸ごと学ぶ．切片だけを学ぶ手は無かった．SER で BitFit（切片だけを学ぶ微調整）を試した研究も無かった．
3. 話者正規化・コーパス正規化・平均を引く手が「位置（切片）を直すのか，向きを直すのか」を分けて測った研究．Schuller 2010，Pepino 2021，Kaya 2018，Gat 2022 は，どれも合計の成績か，話者を当てる正解率しか見ていない．
4. SER・MER で，EM による割合の推定（Saerens 2002），BBSE，MLLS（ラベルシフトを推定する手法）を当てて，その得を測った研究．EMO-TTA は割合を更新するが，その寄与を外していない．SER Evals はロジット調整を使うが，有無の比較が無い．
5. ほかのモダリティの予測（蒸留，疑似ラベル）が，生徒の切片側と向き側のどちらを直したかを測った研究．Albanie は「生徒は平静に偏る」と書くだけで，量を出していない．C2KD が分けたのは正解クラスと他のクラスで，切片と向きではない．
6. 3 モダリティの MER で，コーパス間のクラスの割合のずれを測り，その補正の得を表現の補正と並べて数値で示した研究．Amanda は図で示しただけで，同じ割合を仮定している．
7. 声と融合の分野に固有の，(a)(b) の分け方を予言する理論．

---

## 7. 限界

- この会話の全担当で共有する検索の上限（200 回）に達し，表の #59 と #60 の検索語は実行できなかった．#60 は「SER で切片だけを当て直す手」を直接狙った検索語である．上限が戻ったら実行し直すべきである．
- MSP 研究室のサーバー（msp.utdallas.edu）は名前解決に失敗し，Busso 2013（IFN）と Abdelwahab 2015 の本文を読めなかった．Springer は認証のリダイレクトで，C2KD（CVF）は 403 で読めなかった．Shah 2015 は ASU の写しで読んだ．
- 「共通の共分散のガウス判別で平均を更新すると重みと切片の両方が動く」は，式から導いた推測であり，EMO-TTA の論文には書かれていない．
- 手元の記録で既に「×」と判定された研究は，読み直していない．
