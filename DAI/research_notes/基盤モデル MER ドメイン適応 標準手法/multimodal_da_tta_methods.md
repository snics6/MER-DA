# 事前学習エンコーダを使うマルチモーダル感情認識・感情極性分析のドメイン適応（2022〜2026）：UDA・TTA・ソースなし・少数ラベル

調査日：2026-11-08．対象は，音声・映像・言葉のうち 2 つ以上を使う感情認識（MER）と感情極性分析（MSA）で，ソースのコーパスからターゲットのコーパスへ適応する論文である．

**読んだ水準の記号**
- ◎：論文の PDF を自分でテキストにして，手法と実験の節を直接読んだ（Amanda，READ，M2CAN）
- ○：arXiv の HTML の全文を取得したが，取得ツールが要約した内容で読んだ．数値は要約を経ているので，論文に引く前に PDF の表で確かめる必要がある（CASP，SuMi，Boomda，M4SER，Dong ほか 2026，MERBench，MER2024，MMDG-Bench）
- △：抄録か検索の抜粋だけ（MIDG，TTSD-FAR，EMO-TTA，CLAIP-Emo，GD-Adapt，PTA，MuSACo，S2LIF）

**依頼文の訂正**：READ（Yang ほか）は ICCV 2023 ではなく **ICLR 2024** の論文である．PDF の 1 行目に "Published as a conference paper at ICLR 2024" とある — [READ, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/f25d75fc760aec0a6174f9f5d9da59b8-Paper-Conference.pdf)

**本研究の設定（比較のため）**：凍結した WavLM（音声），ResNet-50 の顔の特徴，RoBERTa（言葉）．モダリティごとの線形 softmax 分類器と融合の分類器．ターゲットにはクラスあたり K 件のラベル付き発話とラベル無し発話がある．ソースは AFEW・IEMOCAP・MOSEI・MELD のうち 2〜3 コーパス．4 クラス．主指標は UAR．

---

## 問い 1：マルチモーダル感情・感情極性の教師なし・半教師ありドメイン適応（整列の損失，自己訓練，モダリティ固有と共有の扱い）

### Takeaway
実コーパス間の 3 モダリティ MER で教師なし DA を行った論文は少なく，見つけた中心は Amanda（ACL Findings 2024）1 本である．Amanda と同じグループの Boomda（2025）は，モダリティごとに 2 次の統計量を合わせる CORAL 型の損失を使い，モダリティごとの整列の重みを自動で調整する．敵対的学習（DANN 系）は提案手法の中身より比較相手として現れることが多い．

### Cited Findings

**Amanda（Zhang, Sun, Hong, Li．ACL Findings 2024）◎**
- 設定：教師なしのマルチモーダル DA．ターゲットのラベルは一切使わない．仮定は「特徴の分布はドメインで変わるが，ラベルの分布は変わらない（p(Y^t)=p(Y^s)）」 — [Amanda](https://aclanthology.org/2024.findings-acl.859.pdf)
- 特徴：言葉は BERT-base，音声は wav2vec2 で，1・7・12 番目の Transformer ブロックの出力を連結した 2304 次元の系列を使う．映像は 64 フレームを等間隔に取り，S3FD で顔を切り出し，APViT に通して 64×768 の系列にする．事前学習モデルで特徴を抽出してから，その上に映像は 1 層の LSTM，音声と言葉は TextCNN を学習する（エンコーダは特徴抽出に使うだけで，学習で更新していない） — [Amanda](https://aclanthology.org/2024.findings-acl.859.pdf)
- 損失は 3 項 L = L_IB + α1·L_LA + α2·h(L_CA)：
  - L_IB：モダリティごとの情報ボトルネック．表現をガウスと仮定し，log|Σ| で圧縮し，ソースのラベルの交差エントロピーで予測を保つ — [Amanda](https://aclanthology.org/2024.findings-acl.859.pdf)
  - L_LA（ラベルの整列）：ターゲットの予測の平均（融合の出力）と，ソースのラベルの平均との交差エントロピー．ターゲットの発話ごとの疑似ラベルは使わない — [Amanda](https://aclanthology.org/2024.findings-acl.859.pdf)
  - L_CA（モダリティごとの表現の整列）：ソースとターゲットの表現の分散（2 次の統計量）を合わせる．CORAL と同じ考え方で，敵対的学習を避けるためと著者は書く — [Amanda](https://aclanthology.org/2024.findings-acl.859.pdf)
- モダリティの扱い：音声・映像・言葉の 3 つに，3 つを合わせた「補助のモダリティ（融合）」を加えた M+1=4 個のそれぞれに分類器と整列の損失を置く．4 個の整列の損失を p ノルムまたは log-sum-exp の代理関数 h でまとめると，損失が大きいモダリティの重みが自動で大きくなる（p=∞ なら最大の損失だけが勾配に効く） — [Amanda](https://aclanthology.org/2024.findings-acl.859.pdf)
- データ：IEMOCAP・MSP-IMPROV・MELD・CMU-MOSEI の 4 コーパスから neutral・happy・sad・angry の 4 クラスを取る（Zhao ほか 2021 に従う）．1 ソース → 1 ターゲットの 7 組．指標は重み付き F1，3 回の平均 — [Amanda](https://aclanthology.org/2024.findings-acl.859.pdf)
- 数値（表 1，重み付き F1）：直接転移（ソースのみ）の 7 組の平均 49.20 に対し，Amanda A-N(p=2) は 56.73（+7.5 点）．組ごとの差は +3.7（MSP→IEMOCAP 59.95→63.61）から +14.0（MOSEI→IEMOCAP 46.29→60.25）．比較相手で最良の DALN の平均は 54.24，DANN は 50.73（平均は表 1 の値から本ノートで計算） — [Amanda](https://aclanthology.org/2024.findings-acl.859.pdf)
- α2=0.1 を比較に使ったと書くが，ターゲットのラベル無しでどう選んだかは本文に書かれていない — [Amanda](https://aclanthology.org/2024.findings-acl.859.pdf)

**Boomda（Sun, Zhang, Hong, Zhu, Gao．arXiv 2511.08152，2025-11．掲載先の記載なし）○**
- Amanda と同じ著者を含むグループの続編．音声は WavLM＋TextCNN，映像は RAF-DB で学んだ APViT＋1 層 LSTM，言葉は BERT-base＋TextCNN．事前学習モデルの最後の 3 層は学習し，残りは凍結 — [Boomda](https://arxiv.org/abs/2511.08152)
- 損失：モダリティごとの情報ボトルネック，ターゲットへの疑似ラベル（3 つのモダリティの投票，M_v=3），モダリティごとの CORAL．モダリティの重みは MGDA 型の多目的最適化（パレート停留点）を対角の 2 次問題に簡略化した閉じた式で決める — [Boomda](https://arxiv.org/abs/2511.08152)
- ドメインのずれは**人工的**：IEMOCAP と MSP-IMPROV をそれぞれ半分に分け，ターゲット側の音声に白色雑音，映像の明るさを 20% に下げてガウス雑音，言葉の 40% をマスクする．コーパス間の転移ではない — [Boomda](https://arxiv.org/abs/2511.08152)
- 数値（重み付き F1）：IEMOCAP で直接転移 36.75 → 最良の比較相手 CDAN 46.95 → Boomda 48.73．MSP-IMPROV で 33.53 → 40.07 → 41.50．除去実験では CORAL だけで 45.55，疑似ラベルだけで 45.09（IEMOCAP） — [Boomda](https://arxiv.org/abs/2511.08152)

**M2CAN（Multi-source Multi-modal Domain Adaptation for Visual-textual Sentiment Classification．取得した PDF は匿名の投稿版で，掲載先と年は確認できない）◎**
- 画像と文の感情極性で，**複数ソース**の教師なし DA．ドメインは Yelp・Twitter・MVSA の 3 つで，2 つをソース，1 つをターゲットにする．指標は正解率 — [M2CAN](https://warushrush.github.io/papers/M2CAN.pdf)
- 整列は 3 種類：(1) ドメインの中で画像と文の表現を近づける対照学習（cross-modal contrastive），(2) モダリティごとにドメイン間の差を MMD で測って縮める整列（著者は cross-domain contrastive alignment と呼ぶ），(3) 融合した表現に対するドメインの敵対的学習 — [M2CAN](https://warushrush.github.io/papers/M2CAN.pdf)
- エンコーダは ResNet-50 と BERT-base で，学習率 2e-5 で**微調整する** — [M2CAN](https://warushrush.github.io/papers/M2CAN.pdf)
- 数値（表 1，3 ターゲットの平均正解率）：ソースを合併したソースのみ 57.9，比較相手の MM-SADA 60.3〜60.8，xMUDA 60.2〜60.4，MDAN 61.3，M2CAN 64.0，ターゲットで学習した上限 67.4 — [M2CAN](https://warushrush.github.io/papers/M2CAN.pdf)

**MuSACo（Zeeshan ほか．WACV 2026．arXiv 2508.12522）△**
- 被験者 1 人を 1 ソースドメインとみなす複数ソースの DA で，ターゲットの被験者に関係の深いソースの被験者を選ぶ．優勢なモダリティから疑似ラベルを作ってクラスを区別する学習に使い，確信度の低いターゲットの標本にはクラスを使わない損失を当てる．データは BioVid・StressID・BAH（痛み・ストレス・ためらい）で，コーパス間の転移ではない — [MuSACo（arXiv）](https://arxiv.org/abs/2508.12522)，[WACV 2026](https://wacv.thecvf.com/virtual/2026/poster/587)

**古い先例（2022 年より前．文脈としてだけ記す）**
- Amanda は先行の多モダリティ DA として，RGB とオプティカルフローの DA（Munro & Damen 2020，MM-SADA）などを挙げ，それらはモダリティどうしの分布の差が近い場合で，感情のように異質なモダリティには専用の設計が要ると述べる — [Amanda](https://aclanthology.org/2024.findings-acl.859.pdf)
- M2CAN は単一ソースのマルチモーダル DA の比較相手として MMAN（Qi ほか 2018），MM-SADA（CVPR 2020），xMUDA（CVPR 2020）を使う — [M2CAN](https://warushrush.github.io/papers/M2CAN.pdf)

**半教師あり（ターゲットに少数ラベル）の DA**
- 3 モダリティの MER で，ターゲットの少数ラベルとラベル無しデータを両方使う DA の論文は，今回の検索では見つからなかった（問い 3 の M4SER は少数ラベルを使うが，ラベル無しのターゲットは使わない）．MER2024 の MER-SEMI は 115,595 件中ラベル付き約 1,169 件という半教師ありの課題を設けたが，主催者のベースラインはラベル無しデータを学習に使っていない — [MER2024](https://arxiv.org/abs/2404.17113)

### Inferences
- 実コーパス間の 3 モダリティ MER の教師なし DA で整列の損失の中身まで確かめられたのは Amanda 1 本で，その延長の Boomda も CORAL 型を使う．この系統では「モダリティごとの CORAL ＋ 整列の損失のモダリティ間の重み付け」が使われており，敵対的学習（DANN・CDAN・MADA・DALN）は比較相手として置かれている．
- 最適輸送（OT）とプロトタイプによる整列を MER・MSA の DA の中心に使った 2022〜2026 年の論文は，今回の検索では見つからなかった．
- 疑似ラベルの作り方は，Boomda がモダリティの投票，MuSACo が優勢なモダリティ，CASP（問い 2）が予測の時間的な安定さで標本を選ぶ．どれも「別のモダリティや別の時点の予測が一致する標本だけを使う」という選び方である．

### Gaps
- 半教師あり DA（ターゲットに K 件のラベル＋ラベル無し）を 3 モダリティの MER で行った論文は見つからなかった．音声だけの SER には半教師ありの DA がある（例：SDAN）が，本調査の範囲外として読んでいない．
- M2CAN の掲載先と年は，取得した PDF が匿名版のため確認できなかった．
- MuSACo の手法の細部（どのエンコーダを凍結するか，数値）は抄録の抜粋しか読んでいない．

---

## 問い 2：マルチモーダル感情・感情極性のテスト時適応（何を更新するか，損失，評価に使う感情データ）

### Takeaway
マルチモーダル TTA の基準になる READ（ICLR 2024）は，融合層の注意の射影（Q・K・V）を更新し，確信度が閾値 e^{-1} を超える予測だけを強める損失を使うが，評価は行動認識と音響イベント（Kinetics50-C・VGGSound-C）で，感情データを使っていない．感情極性で TTA を評価した論文（CASP・SuMi・GD-Adapt）は CMU-MOSI・CMU-MOSEI・CH-SIMS の間の転移を使い，多くは正規化層だけを更新する．感情カテゴリの MER で 3 モダリティの TTA を評価した論文は，抄録しか読めなかった TTSD-FAR（MELD・DFEW・BAH）だけである．

### Cited Findings

**READ（Yang, Li, Zhang, Hu, Peng．ICLR 2024）◎**
- 更新するもの：従来の TTA が正規化層を更新するのに対し，READ はモダリティをまたぐ注意の融合層の Q・K・V の射影の重みと偏りだけをテスト時に更新する（図では残りを "Frozen" と表示） — [READ](https://proceedings.iclr.cc/paper_files/paper/2024/file/f25d75fc760aec0a6174f9f5d9da59b8-Paper-Conference.pdf)
- 損失：確信度 p_i（softmax の最大値）について L_ra = (1/B) Σ p_i log(e^γ / p_i)．閾値 γ は全設定で e^{-1} に固定．p_i>γ の予測は最適化に寄与し，p_i<γ の予測の影響は消える．これに予測のクラスの偏りを防ぐ負のエントロピーの項 L_bal を足す — [READ](https://proceedings.iclr.cc/paper_files/paper/2024/file/f25d75fc760aec0a6174f9f5d9da59b8-Paper-Conference.pdf)
- 骨格：Web 規模の音声・映像で事前学習した CAV-MAE を Kinetics50 と VGGSound の学習データで微調整してソースのモデルにする．Adam，学習率 1e-4，ミニバッチ 64，1 エポックのオンライン更新 — [READ](https://proceedings.iclr.cc/paper_files/paper/2024/file/f25d75fc760aec0a6174f9f5d9da59b8-Paper-Conference.pdf)
- データ：Kinetics50-C と VGGSound-C（映像 15 種・音声 6 種の破損，強さ 5 段階）．感情・感情極性のデータは使っていない — [READ](https://proceedings.iclr.cc/paper_files/paper/2024/file/f25d75fc760aec0a6174f9f5d9da59b8-Paper-Conference.pdf)
- 比較相手：Tent，MMT（Shin ほか 2022），EATA，SAR — [READ](https://proceedings.iclr.cc/paper_files/paper/2024/file/f25d75fc760aec0a6174f9f5d9da59b8-Paper-Conference.pdf)
- 数値（表 1，Kinetics50-C の映像の破損 15 種の平均正解率）：注意の融合のソースのモデル 59.9 → READ 62.5（+2.6）．同じ表で Tent 59.4，EATA 60.1，SAR 59.8 で，ソースとほぼ変わらない — [READ](https://proceedings.iclr.cc/paper_files/paper/2024/file/f25d75fc760aec0a6174f9f5d9da59b8-Paper-Conference.pdf)

**CASP（Guo, Jin, Xu, Lin, Wu．"Bridging the Gap for Test-Time Multimodal Sentiment Analysis"．AAAI の論文集に掲載．arXiv 2412.07121）○**
- 感情極性の回帰（−3〜+3）の TTA．ソースのデータは適応のときに使わず，ターゲットのラベルも使わない — [CASP（arXiv）](https://arxiv.org/abs/2412.07121)，[AAAI](https://ojs.aaai.org/index.php/AAAI/article/download/33867/36022)
- 特徴：言葉は BERT-base（SIMS は中国語の BERT-base），音声は LibROSA，映像は OpenFace 2.0．骨格は Transformer の符号化器で，早期融合と後期融合の 2 種類．エンコーダの凍結の有無は書かれていない — [CASP](https://arxiv.org/abs/2412.07121)
- 更新するもの：著者は Tent に従い "the linear and lower-dimensional feature modulation parameters (i.e. normalization layers)" を更新すると書く — [CASP](https://arxiv.org/abs/2412.07121)
- 損失：段 1 は，1 つのモダリティをランダムに落とした入力を拡張とみなし，元の入力との表現を NT-Xent 型の対照損失で近づける（15 エポック）．段 2 は，段 1 の途中の複数の保存点の予測の変動が小さい上位 5% の標本を選び，保存点の予測の平均を疑似ラベルにして自己訓練する — [CASP](https://arxiv.org/abs/2412.07121)
- データ：MOSEI→SIMS，MOSI→SIMS，MOSI→MOSEI，SIMS→MOSI，SIMS→MOSEI の 5 組．MOSEI→MOSI は MOSEI が MOSI を含むので除く．指標は 2 値の正解率・F1・MAE，5 シードの平均 — [CASP](https://arxiv.org/abs/2412.07121)
- 比較相手：ソースのみ，自己訓練，正規化層の更新（Tent 型），TTA-IQA の group contrastive，READ の reliable fusion．Tent・SHOT・SAR・EATA・CoTTA そのものは実験に入れていない — [CASP](https://arxiv.org/abs/2412.07121)
- 数値（表 1，2 値の正解率）：後期融合で 5 組の平均がソースのみ 50.79 → CASP 56.35（+5.6）．組ごとの差は +2.0（SIMS→MOSEI）から +12.1（MOSI→SIMS 39.17→51.27）．早期融合では MOSEI→SIMS で 45.95→63.89（+17.9）（平均は表の値から本ノートで計算） — [CASP](https://arxiv.org/abs/2412.07121)

**SuMi（Guo, Jin．"Smoothing the Shift"．arXiv 2503.02616．取得したページに掲載先の記載はない．検索の要約は ICLR 2025 とするが未確認）○**
- 更新するもの：正規化層のアフィンの係数だけ — [SuMi](https://arxiv.org/abs/2503.02616)
- 損失：エントロピー最小化に標本の重み 1/exp(Ent−Ent0) を掛ける．四分位範囲で表現の外れ値を除く標本選択，融合の予測のエントロピーが小さく単独モダリティのエントロピーの和が大きい標本を選ぶ条件，単独モダリティの予測と「残りのモダリティと融合の予測の平均」との KL を縮める相互情報の共有の損失 — [SuMi](https://arxiv.org/abs/2503.02616)
- 骨格は READ と同じ CAV-MAE．主な評価は Kinetics50-C と VGGSound-C で，付録で CMU-MOSI（3 モダリティ）と，実際のずれとして MOSI→SIMS と SIMS→MOSI を使う（2 値分類として交差エントロピーで扱う）．感情カテゴリのデータは使わない — [SuMi](https://arxiv.org/abs/2503.02616)
- 数値：Kinetics50-C（弱い分布外）でソース 60.2 → SuMi 63.9，VGGSound-C（強い分布外）で 15.6 → 19.7．Tent は VGGSound-C の強い分布外で 5.1 まで下がる — [SuMi](https://arxiv.org/abs/2503.02616)

**GD-Adapt（Zhang ほか．ACL Findings 2026）△**
- 生成の拡散モデルと判別の回帰モデルをベイズの蒸留で組み合わせる MSA の TTA．MOSI→SIMS で MAE 0.6872→0.5673，2 値の正解率 +5.81 点で 57.33%．SIMS→MOSI で適応なしより +11.18 点．5 つのドメイン間の設定で評価．更新するパラメータと損失は抄録に無い — [GD-Adapt](https://aclanthology.org/2026.findings-acl.350/)

**TTSD-FAR（Aslam, Koerich, Pedersoli, Etemad, Granger．arXiv 2608.18386，2026-08）△**
- 大規模な動画言語モデル（LVLM）で，テスト時にモダリティが 0〜50% 欠ける状況をドメインのずれとみなす．完全なモダリティで学んだ凍結の教師が低ランクの生徒を自己蒸留で導き，Fisher 情報の安定性を見て生徒を教師へ戻す．データは MELD・DFEW・BAH．エントロピーに基づく適応は長い適応で劣化すると抄録は述べる．数値は抄録に無い — [TTSD-FAR](https://arxiv.org/abs/2608.18386)

**Partition-Then-Adapt（Wang, Lyu, Ding．NeurIPS 2025）△**
- 複数モダリティが同時にずれる TTA．テストのバッチを信頼できる部分とできない部分に分けて重みを付けた**エントロピー損失**を使い，注意に導かれたモダリティ間の整列で正則化する．評価は Kinetics50-MC・VGGSound-MC で感情データは無い — [PTA](https://neurips.cc/virtual/2025/poster/117876)

**音声だけの SER の TTA（範囲の外だが，感情データでの TTA の結果として記す）**
- Dong, Jia, Dang（arXiv 2601.16240，2026-01）○：wav2vec 2.0 base をソースで微調整し，11 の TTA 手法を比べる．エントロピー最小化 6 種（Tent・SAR・EATA・SUTA・DSUTA・CEA．正規化層などを更新），疑似ラベル 2 種（CoTTA・AWMC．EMA の教師），逆伝播なし 3 種（T3A はクラスのプロトタイプの再計算，LAME は予測だけを補正，FOA は CMA-ES でプロンプトを最適化）．IEMOCAP↔RAVDESS の 4 クラスのコーパス間で，RAVDESS をターゲットにするとソース 37.8% → T3A 43.8%（正解率），IEMOCAP をターゲットにすると 50.0 → 50.2．エントロピー最小化の平均は多くの設定でソース以下（例：IEMOCAP の台本→即興で 49.7 対 51.3） — [Dong ほか 2026](https://arxiv.org/abs/2601.16240)
- EMO-TTA（Shi ほか．ICASSP 2026）△：音声言語モデルの予測を事前分布にして，クラスごとの統計量を EM で逐次更新する，重みを変えない TTA．6 つの分布外の SER データで評価 — [EMO-TTA](https://arxiv.org/abs/2509.25495)

### Inferences
- マルチモーダル TTA で更新する部分は，正規化層（CASP・SuMi・比較相手の Tent・EATA・SAR）が多数派で，READ の注意の融合層の更新が 2 番目の型である．重みを変えずに予測やクラスの統計を直す型（T3A・LAME・EMO-TTA）は，音声だけの SER で現れる．
- 損失は，エントロピー最小化に確信度で標本を選ぶか重みを付ける型（READ・SuMi・PTA）が主流で，回帰の MSA では対照学習と疑似ラベルの自己訓練（CASP）や拡散モデルとの蒸留（GD-Adapt）に置き換わる．
- 感情データで TTA を評価した結果では，エントロピー最小化がソースのみを下回る報告が複数ある（Dong ほか 2026 の SER，SuMi の表の Tent）．本研究の線形 softmax の分類器は正規化層も注意の層も持たないので，これらの TTA を本研究にそのまま当てはめることはできない．当てはまるのは分類器の重みか特徴の統計量を直す型（T3A，クラスの統計の更新）である．

### Gaps
- 感情カテゴリの MER（IEMOCAP・MELD など）で 3 モダリティの TTA を評価し，数値まで確かめられた論文は見つからなかった．TTSD-FAR は抄録だけで，更新する部分と数値を確かめていない．
- GD-Adapt と PTA の更新するパラメータと損失の式は本文を読んでいない．
- CASP の掲載年は，AAAI の論文集（ojs.aaai.org）にあることと arXiv v2 が 2025-02 であることから AAAI 2025 と推定したが，巻の記載は確かめていない．

---

## 問い 3：マルチモーダル感情の少数ラベル（K-shot）のターゲット適応（プロンプト調整，アダプタ，LoRA，凍結した特徴の線形分類，プロトタイプ・クラス平均の分類器）

### Takeaway
ソースのコーパスで学んだ MER のモデルをターゲットのクラスあたり K 件で適応する論文として見つけたのは M4SER（2025）で，音声と ASR の書き起こしの 2 モダリティを使い，エンコーダを微調整する．凍結した基盤モデルに LoRA やプロンプトを足す感情の論文（CLAIP-Emo・PE-CLIP など）は同じコーパスの中での学習で，コーパス間の少数ラベル適応は評価していない．凍結した特徴の上の線形分類やクラス平均の分類器でコーパス間の K-shot 適応を測ったマルチモーダルの論文は見つからなかった．

### Cited Findings
- M4SER（He, Shi, Hu, Mi, Li, Toda．arXiv 2509.18706，2025-09）○：音声は HuBERT（hubert-base-ls960），言葉は BERT（bert-base-uncased）で，どちらも学習中に微調整する．言葉は Whisper（medium.en と large-v2）の書き起こし．ドメイン適応のモジュールは無く，敵対的な部品はモダリティの識別器でドメインの識別器ではない — [M4SER](https://arxiv.org/abs/2509.18706)
- M4SER の手順：ソースのコーパス全体で学び，ターゲットの 30% を検証，70% をテストに回す．少数ラベルはターゲットの検証の部分からクラスあたり 5・10・20・60 件を取る．モデル全体を更新するか分類器だけかは書かれていない．4 クラス（MELD の驚き・恐れ・嫌悪は捨てる） — [M4SER](https://arxiv.org/abs/2509.18706)
- M4SER の数値（MELD→IEMOCAP の UA）：0 件 62.2，5 件 64.4，10 件 67.7，20 件 68.7，60 件 75.9．IEMOCAP→MELD の正解率は 0 件 57.3，60 件 62.1．ターゲットで全部のラベルを使った上限は報告されていない — [M4SER](https://arxiv.org/abs/2509.18706)
- CLAIP-Emo（Chen ほか．arXiv 2509.14527）△：凍結した CLIP（映像）と CLAP（音声）に LoRA を足し，全パラメータの 4.0% 以下（ViT-L/14 で約 8M）だけを学ぶ．DFEW で WAR 80.14%，MAFW で 61.18%．コーパス間の評価は無い — [CLAIP-Emo](https://arxiv.org/abs/2509.14527)
- MERBench（Lian ほか．arXiv 2401.03429）○：主な実験では "we freeze the weights of pre-trained models and only optimize the shallow classifiers" と書く．微調整との比較（表 XVI）は結果が割れ，VideoMAE は CH-SIMS で +6.43，CLIP-base は MER-MULTI で −3.05．著者は感情認識のように標本が少ない課題では微調整より事前学習を勧める — [MERBench](https://arxiv.org/abs/2401.03429)
- 検索で見つかった凍結した CLIP のプロンプト調整（PE-CLIP，EmoVCLIP）は映像だけか同じコーパスの中での学習で，コーパス間の少数ラベル適応ではない（検索の要約だけで本文は読んでいない） — [検索結果の一例：CLIP-AUTT](https://arxiv.org/html/2603.27999)

### Inferences
- コーパス間の K-shot 適応を MER で測る論文は少なく，見つけた M4SER はエンコーダを微調整するので，「凍結したエンコーダ＋線形分類器＋K 件」という本研究の設定と同じ条件の比較相手は見つからなかった．
- MERBench の「凍結したエンコーダの上で浅い分類器を学ぶ」という手順は，本研究の土台の手順と同じ型である．ただし MERBench はコーパス間の評価で適応をしない．

### Gaps
- 凍結した特徴の線形分類やクラス平均（プロトタイプ）の分類器で，マルチモーダルのコーパス間 K-shot 適応を測った論文は見つからなかった．音声だけの SER の少数ラベル適応（例：Ahn ほか 2021，IEEE SPL）は引用で見たが本文を読んでいない．
- 画像と文の少数ラベルの感情極性（Twitter の MVSA などでのプロンプト学習）は音声を含まないので調べていない．

---

## 問い 4：これらの DA の論文で使われる基盤エンコーダと凍結の有無

### Takeaway
音声は wav2vec2・WavLM・HuBERT，言葉は BERT-base，顔は APViT や OpenFace，音声と映像の同時のモデルは CAV-MAE が使われる．凍結の有無は論文で割れる．Amanda と MERBench は特徴を凍結したエンコーダで抽出し，Boomda は最後の 3 層だけを学び，M2CAN・M4SER はエンコーダを微調整する．MSA の TTA の系統（CASP・MMDG-Bench）は BERT＋LibROSA＋OpenFace の古い特徴の組を使う．

### Cited Findings
- Amanda：BERT-base・wav2vec2（1・7・12 層の連結）・APViT（S3FD で顔を切り出す）．特徴を抽出し，上に LSTM・TextCNN を学ぶ — [Amanda](https://aclanthology.org/2024.findings-acl.859.pdf)
- Boomda：WavLM・APViT（RAF-DB）・BERT-base．最後の 3 層は学習し，残りは凍結 — [Boomda](https://arxiv.org/abs/2511.08152)
- M2CAN：ResNet-50・BERT-base を微調整（学習率 2e-5） — [M2CAN](https://warushrush.github.io/papers/M2CAN.pdf)
- M4SER：HuBERT base・BERT base を微調整．Whisper で書き起こし — [M4SER](https://arxiv.org/abs/2509.18706)
- CASP：BERT-base（中国語は中国語の BERT），LibROSA，OpenFace 2.0．凍結の有無の記載なし — [CASP](https://arxiv.org/abs/2412.07121)
- MMDG-Bench：768 次元の BERT・LibROSA・OpenFace 2.0 を Transformer で融合．凍結の有無の記載なし — [MMDG-Bench](https://arxiv.org/abs/2605.06643)
- READ・SuMi：CAV-MAE を源のデータで微調整し，テスト時は注意の層か正規化層だけを更新 — [READ](https://proceedings.iclr.cc/paper_files/paper/2024/file/f25d75fc760aec0a6174f9f5d9da59b8-Paper-Conference.pdf)，[SuMi](https://arxiv.org/abs/2503.02616)
- MERBench：映像は ResNet・VideoMAE・CLIP・DINOv2・EVA-02 など，音声は wav2vec 2.0・WavLM・Whisper・HuBERT など，言葉は RoBERTa-large・DeBERTa-large・Llama2-13B・Baichuan-13B など．コーパス間の表の多モダリティの行は CLIP-large・HuBERT-large・Baichuan-13B．エンコーダは凍結 — [MERBench](https://arxiv.org/abs/2401.03429)
- CLAIP-Emo：CLIP・CLAP を凍結して LoRA — [CLAIP-Emo](https://arxiv.org/abs/2509.14527)
- TTSD-FAR：LVLM（モデル名は抄録に無い）を教師として凍結し，低ランクの生徒を更新 — [TTSD-FAR](https://arxiv.org/abs/2608.18386)

### Inferences
- 確かめた論文のうち抄録だけの TTSD-FAR を除く 10 本では，エンコーダを完全に凍結して特徴を抽出するのは Amanda・MERBench・CLAIP-Emo（LoRA を足す）の 3 本，一部を学ぶのが Boomda の 1 本，微調整が M2CAN・M4SER・READ・SuMi（ソースで微調整）の 4 本，記載なしが CASP・MMDG-Bench の 2 本である．WavLM と RoBERTa を DA に使ったのは，見つけた中では Boomda（WavLM）と MERBench（両方．ただし DA はしない）だけである．
- LLM は，MERBench で言葉のエンコーダとして，TTSD-FAR で LVLM として現れるが，DA の損失を LLM の表現に当てた論文は見つからなかった．

### Gaps
- CASP と MMDG-Bench の BERT が凍結か微調整かは確認できなかった．
- ResNet-50 の顔の特徴（本研究の映像の特徴）を DA に使った MER の論文は見つからなかった（M2CAN の ResNet-50 は一般の画像の特徴である）．

---

## 問い 5：これらの論文で標準の比較相手

### Takeaway
教師なし DA の MER では「直接転移（ソースのみ）」と DANN・CDAN・MADA・DALN の敵対的な系統が標準の比較相手で，MMD と CORAL は比較表に入っていないことが多い．MSA と一般のマルチモーダルの TTA では，ソースのみ・Tent・EATA・SAR・READ が標準で，SHOT はどの比較表にも入っていなかった．

### Cited Findings
- Amanda の比較相手：直接転移，DANN，CDAN，CDAN+E，MADA，DALN．単独モダリティ向けの手法はモダリティごとに DA を当てて比べる — [Amanda](https://aclanthology.org/2024.findings-acl.859.pdf)
- Boomda の比較相手：直接転移，DANN，DADA，MADA，DALN，PCL，CDAN．MMD は関連研究にだけ現れる．Amanda は参考文献にあるが表 1 に無い — [Boomda](https://arxiv.org/abs/2511.08152)
- M2CAN の比較相手：ソースのみ（合併・最良の単一ソース），MMAN，MM-SADA，xMUDA，MDAN，ターゲットで学んだ上限 — [M2CAN](https://warushrush.github.io/papers/M2CAN.pdf)
- READ の比較相手：Tent，MMT，EATA，SAR — [READ](https://proceedings.iclr.cc/paper_files/paper/2024/file/f25d75fc760aec0a6174f9f5d9da59b8-Paper-Conference.pdf)
- SuMi の比較相手：Tent，EATA，SAR，SoTTA，CEMA，DeYO，READ — [SuMi](https://arxiv.org/abs/2503.02616)
- CASP の比較相手：ソースのみ，自己訓練，正規化層の更新，group contrastive，READ の reliable fusion — [CASP](https://arxiv.org/abs/2412.07121)
- MMDG-Bench の比較相手：ERM，RNA-Net，SimMMDG，MOOSA，CMRF，NEL，JAT，MBCD，GMP，上限．DANN・CORAL・MMD・IRM は無い — [MMDG-Bench](https://arxiv.org/abs/2605.06643)
- Dong ほか 2026（SER）：Tent，SAR，EATA，SUTA，DSUTA，CEA，CoTTA，AWMC，T3A，LAME，FOA．SHOT と BN の統計の更新は無い — [Dong ほか 2026](https://arxiv.org/abs/2601.16240)

### Inferences
- 比較相手の数を数えると，確かめた教師なし DA の 3 本（Amanda・Boomda・M2CAN）のうち，ソースのみは 3 本，DANN は 2 本，CDAN と MADA と DALN は 2 本，MM-SADA と xMUDA は 1 本（M2CAN）で，MMD・CORAL の単独の手法を比較表に入れたのは 0 本である．TTA の 4 本（READ・SuMi・CASP・Dong）では，Tent は 3 本（READ・SuMi・Dong．CASP も Tent 型の正規化層の更新を比較相手に入れる），EATA と SAR は 3 本（READ・SuMi・Dong），READ は 2 本（SuMi・CASP），SHOT は 0 本である．
- 本研究の比較相手に「ソースのみ」と「合併」（M2CAN の source-combined にあたる）を置くことは，この分野の慣習に合う．

### Gaps
- 半教師あり DA（MME・APE など）を比較相手に入れたマルチモーダルの感情の論文は見つからなかった．

---

## 問い 6：標準のデータ・ドメインの分け方・指標

### Takeaway
感情極性では CMU-MOSI・CMU-MOSEI・CH-SIMS の間の転移が標準で（MOSEI→MOSI は除く），指標は 2 値の正解率・F1・MAE である．感情カテゴリでは IEMOCAP・MSP-IMPROV・MELD・CMU-MOSEI から neutral・happy・sad・angry の 4 クラスを取り，1 ソース → 1 ターゲットで評価し，指標は重み付き F1 が多い．UAR を主指標にした 3 モダリティの DA の論文は見つからず，AFEW を DA に使った論文も見つからなかった．

### Cited Findings
- Amanda：IEMOCAP・MSP-IMPROV・MELD・CMU-MOSEI の 4 クラス，7 組，重み付き F1 — [Amanda](https://aclanthology.org/2024.findings-acl.859.pdf)
- Boomda：IEMOCAP と MSP-IMPROV を半分に分け，ターゲット側を人工的に劣化させる．重み付き F1 — [Boomda](https://arxiv.org/abs/2511.08152)
- CASP：MOSEI→SIMS，MOSI→SIMS，MOSI→MOSEI，SIMS→MOSI，SIMS→MOSEI．2 値の正解率・F1・MAE — [CASP](https://arxiv.org/abs/2412.07121)
- MMDG-Bench：MOSI・MOSEI・SIMS をそれぞれ 1 ドメインとし，複数ソースの MOSI+MOSEI→SIMS と MOSI+SIMS→MOSEI．MAE・F1・2 値の正解率 — [MMDG-Bench](https://arxiv.org/abs/2605.06643)
- MIDG（Li, Luo, Hu．arXiv 2512.07430）△：MSA の領域汎化．取得した抄録は 3 つのデータとしか書かない．検索の抜粋は MOSI→MOSEI，MOSI→SIMS，SIMS→MOSEI などを使い，SIMS→MOSEI で CASP より約 5.5% 高いと述べる（未確認） — [MIDG](https://arxiv.org/abs/2512.07430)
- MERBench：IEMOCAP（4 クラス）・MELD・MER-MULTI は 4 ラベル（happiness・sadness・anger・neutral）を共有させてコーパス間で評価し，MOSI・MOSEI・SIMS・SIMS v2 は感情極性の範囲を正規化してから評価する．指標は重み付き F1（WAF）．多モダリティの融合の利得はコーパスの中より小さい（例：MELD→IEMOCAP で単独の最良 46.62 → 多モダリティ 48.86，IEMOCAP の中では 69.67 → 78.20） — [MERBench](https://arxiv.org/abs/2401.03429)
- M4SER：IEMOCAP↔MELD の 4 クラス．MELD→IEMOCAP は WA と UA，IEMOCAP→MELD は正解率と重み付き F1 — [M4SER](https://arxiv.org/abs/2509.18706)
- 一般のマルチモーダル TTA（READ・SuMi・PTA）：Kinetics50-C・VGGSound-C（または -MC）で正解率 — [READ](https://proceedings.iclr.cc/paper_files/paper/2024/file/f25d75fc760aec0a6174f9f5d9da59b8-Paper-Conference.pdf)，[PTA](https://neurips.cc/virtual/2025/poster/117876)

### Inferences
- 「4 クラス・IEMOCAP/MSP-IMPROV/MELD/MOSEI の間の 1 対 1 の転移・重み付き F1」が MER の DA の評価の型で，本研究の「4 クラス・複数ソース・UAR・AFEW を含む」は，クラスの数以外がこの型と違う．重み付き F1 はクラスの比率がずれると UAR と違う順位を出しうるので，本研究の UAR の数値をこれらの論文の数値と直接比べることはできない．
- MSA の系統は 2 値・回帰なので，4 クラスの本研究とは課題そのものが違う．

### Gaps
- MERBench のコーパス間の表で，感情極性の値をどう対応させたかは取得した範囲に書かれていなかった．
- MSP-IMPROV・CREMA-D・RAVDESS・AFEW を 3 モダリティの DA に使った論文は見つからなかった．

---

## 問い 7：何が「標準の手法」か，それぞれの選択がどれだけ多いか（数えた結果と，論文ごとの本研究との違い）

### Takeaway
マルチモーダルの感情・感情極性の DA には，この分野で共有された 1 つの標準の手法は無い．確かめた論文の範囲では，教師なし DA は「モダリティごとの CORAL 型の整列＋モダリティ間の重み付け」（Amanda・Boomda），TTA は「正規化層の更新＋確信度で選ぶエントロピー最小化」（Tent・EATA・SAR を土台にした READ・SuMi）が最も多い型である．MMDG-Bench は，MSA の領域汎化の専用手法が ERM（ソースを合併して普通に学ぶ）を平均で 1 点未満しか上回らないと報告している．

### Cited Findings
- MMDG-Bench（Dong ほか．arXiv 2605.06643，2026）○：MSA の複数ソースの 2 組の平均の 2 値の正解率は ERM 65.63，最良の MOOSA 66.60（+0.97），SimMMDG 64.68・MBCD 62.34・GMP 62.35 は ERM を下回る．ターゲットで学んだ上限は 75.22．著者は ERM が約半数の設定で専用手法と同等か上回ると書く — [MMDG-Bench](https://arxiv.org/abs/2605.06643)
- Amanda は，単独モダリティ向けの DA をモダリティごとに当てる比較相手より Amanda が高いことから，マルチモーダルの DA には専用の設計が要ると結論する — [Amanda](https://aclanthology.org/2024.findings-acl.859.pdf)
- CASP は，比較した TTA の中で CASP だけがソースのモデルより一貫して良くなったと報告する — [CASP](https://arxiv.org/abs/2412.07121)
- MERBench は，コーパス間では多モダリティの融合の利得が小さく，融合がターゲット（学習に使ったコーパス）に過学習しうると書く — [MERBench](https://arxiv.org/abs/2401.03429)

**数えた結果（本ノートで確かめた 2022〜2026 年の論文．音声だけの SER の 2 本を除く 16 本）**

| 区分 | 論文 | 本数 |
|---|---|---|
| 実コーパス間の教師なし DA（感情・感情極性） | Amanda，M2CAN（画像と文） | 2 |
| 人工的なずれでの教師なし DA | Boomda | 1 |
| 被験者間の複数ソース DA | MuSACo | 1 |
| 感情極性のデータで評価した TTA | CASP，SuMi（付録），GD-Adapt | 3 |
| 感情カテゴリのデータで評価した TTA | TTSD-FAR（抄録だけ） | 1 |
| 感情データを使わないマルチモーダル TTA | READ，PTA | 2 |
| 感情極性の領域汎化（ターゲットのデータを使わない） | MIDG，MMDG-Bench，S2LIF（検索の抜粋だけ） | 3 |
| コーパス間の少数ラベル適応 | M4SER（音声＋書き起こし） | 1 |
| 適応せずにコーパス間を評価 | MERBench | 1 |
| 凍結の基盤モデル＋LoRA（同じコーパスの中） | CLAIP-Emo | 1 |

- 整列の損失（教師なし DA の 3 本：Amanda・Boomda・M2CAN）：CORAL 型 2，MMD 1，敵対的 1（M2CAN の融合の表現），対照学習 1（M2CAN），ラベルの分布の整列 1（Amanda），最適輸送 0，プロトタイプ 0．
- 自己訓練・疑似ラベル：Boomda（モダリティの投票），MuSACo（優勢なモダリティ），CASP（予測の安定さで上位 5% を選ぶ）の 3 本．
- TTA で更新する部分（READ・SuMi・CASP・PTA・TTSD-FAR）：正規化層 2（SuMi・CASP），注意の融合層 1（READ），低ランクの生徒 1（TTSD-FAR，抄録），記載を確かめていない 1（PTA）．
- 複数ソースを明示的に扱う論文：M2CAN，MuSACo，MMDG-Bench の 3 本．ほかは 1 ソース → 1 ターゲットである．

**論文ごとの設定と本研究との違い**

| 論文 | エンコーダと凍結 | 学習する部分 | ターゲットのデータ | ソース | 課題・指標 | 本研究との主な違い |
|---|---|---|---|---|---|---|
| Amanda（ACL Findings 2024） | BERT・wav2vec2・APViT，凍結で特徴抽出 | LSTM・TextCNN と分類器 | ラベル無しのみ | 1 コーパス | 4 クラス・重み付き F1 | ターゲットのラベル 0 件．非線形の頭．1 ソース．指標が重み付き F1 |
| Boomda（arXiv 2025） | WavLM・APViT・BERT，最後の 3 層は学習 | エンコーダの一部と頭 | ラベル無しのみ | 同じコーパスの半分 | 4 クラス・重み付き F1 | ずれが人工的．コーパス間ではない |
| M2CAN（掲載先未確認） | ResNet-50・BERT を微調整 | 全体 | ラベル無しのみ | 2 ドメイン | 画像と文の極性・正解率 | 音声が無い．微調整．ラベル 0 件 |
| READ（ICLR 2024） | CAV-MAE をソースで微調整 | 注意の Q・K・V | ラベル無し（オンライン） | ソースのデータは使わない | 行動・音響イベント | 感情ではない．ソースのデータなし |
| CASP（AAAI） | BERT・LibROSA・OpenFace | 正規化層 | ラベル無し | ソースのデータは使わない | 極性の回帰・2 値の正解率・MAE | 回帰．ラベル 0 件．ソースのデータを使わない |
| SuMi（arXiv 2025） | CAV-MAE | 正規化層 | ラベル無し（オンライン） | 同上 | 主に行動．付録で MOSI・SIMS | 同上 |
| M4SER（arXiv 2025） | HuBERT・BERT を微調整，Whisper の書き起こし | 全体（少数ラベルで何を更新するかは記載なし） | クラスあたり 5〜60 件．ラベル無しは使わない | 1 コーパス | 4 クラス・UA ほか | 微調整．映像が無い．1 ソース．ラベル無しを使わない．K-shot の点は最も近い |
| MERBench（arXiv 2024） | 多数のエンコーダを凍結 | 浅い分類器 | 使わない（適応なし） | 1 コーパス | 4 クラス・重み付き F1 | 凍結＋浅い頭は同じ．適応をしない |
| MMDG-Bench（arXiv 2026） | BERT・LibROSA・OpenFace | 全体（凍結の記載なし） | 使わない | 2 コーパス | 極性・2 値の正解率・MAE | 領域汎化．極性の課題 |
| TTSD-FAR（arXiv 2026） | LVLM（教師を凍結） | 低ランクの生徒 | ラベル無し（欠けたモダリティ） | 使わない | MELD・DFEW・BAH | ずれがモダリティの欠落．LVLM |

### Inferences
- 本研究の設定（凍結したエンコーダ，線形 softmax，ターゲットに K 件＋ラベル無し，2〜3 ソース，UAR）と同じ条件をすべて満たす論文は見つからなかった．最も近いのは，凍結の点で MERBench と Amanda，K-shot の点で M4SER，複数ソースの点で M2CAN と MMDG-Bench である．
- MMDG-Bench の「専用手法は ERM を平均で 1 点未満しか上回らない」という結果と，MERBench の「コーパス間では融合の利得が小さい」という結果は，本研究がソースのみ・合併を比較相手として必ず置く理由として引ける．ただしどちらも本研究とは課題（極性）か手順（適応なし）が違う．
- Amanda の報告する利得（7 組の平均で +7.5 点）は，ハイパーパラメータ α2 の選び方が書かれていないので，ターゲットのラベルで選んだ可能性を除けない．

### Gaps
- 数えた本数は本ノートで確かめた論文の範囲に限られ，網羅的な数え上げではない．特に IEEE TAC・TMM・TASLP と ACM MM の論文は，検索で原文に届かなかった．
- AFEW をソースかターゲットに使ったマルチモーダルの DA の論文は見つからなかった．
- S2LIF（arXiv 2409.04473）と AdaPGC（arXiv 2604.19093）は検索の抜粋しか見ていない．
