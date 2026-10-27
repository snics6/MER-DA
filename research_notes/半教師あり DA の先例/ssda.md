# 半教師あり DA（SSDA）と教師あり DA の代表 7 本の精読と AdaMatch の評価設定（2026-10-28）

この記録は，論文の関連研究で半教師あり DA（semi-supervised domain adaptation，SSDA．ソースのラベル付きデータ，ターゲットの少数のラベル，ターゲットのラベル無しデータを使って，ターゲットで成績のよい分類器を作る問題）の代表論文を引くために，原典で書誌と中身を確かめたものである．対象は SSDA の 5 本（MME，APE，CDAC，DECOTA，Donahue 2013），ターゲットのラベル無しを使わない教師あり DA の 2 本（FADA，CCSA），引用済みの AdaMatch の SSDA の評価設定である．論文（`paper/`）と `docs/` は編集していない．

**調べ方．**CVF の公開版（openaccess.thecvf.com）から MME・CDAC・DECOTA・Donahue 2013・CCSA の本文 PDF と補足資料を，ECVA（ecva.net）から APE の本文と補足資料を，NeurIPS の論文集（proceedings.neurips.cc）から FADA の本文を，arXiv から AdaMatch（2106.04732v2）の PDF を curl で取り，scratchpad の新しいディレクトリに置いて `pdftotext` で文字にして読んだ．書誌は CVF・ECVA・NeurIPS の頁と Crossref の API で確かめた．LNCS の巻番号は CiNii の書誌で確かめた．表の数値は PDF の文字から読んだ．図の数値は読んでいない．

**確かめた水準の記号．**◎＝本文（PDF 全文）を読んだ，○＝抄録か書誌の頁で確かめた，△＝書誌のみ．原文の引用は英語のまま短く．

**比べる相手（本研究の設定）．**凍結したエンコーダの特徴の上の線形の頭（多クラスのロジスティック回帰）だけを学ぶ．ターゲットのラベルはクラスあたり K 本（K=0〜50）．ラベル無しのターゲットの候補を使い，候補からテスト話者を除く（`docs/methods.md` の手法の節：「ターゲットはテスト話者以外のラベル無し発話の平均を」）．話者を分ける．つまみ（正則化の強さなど）はソースのコーパスを仮のターゲットにして選ぶ（同：「メタ検証は『ソース S をターゲット T′（T 以外）に使ったときの平均が最良の設定』」）．課題は音声・顔・言葉の感情認識で，4 感情，指標は UAR．仮説転移の段では，ソースの分類器の重みへの L2 罰則 ‖W − W_s‖² をかける（同）．

## Q1. 7 本の本文と AdaMatch の評価設定の記録

### Takeaway
7 本すべての本文を読んだ（◎）．依頼者の記憶と比べて，題・著者・会議はすべて一致した．違ったのは次の点である：(1) 頁が IEEE 版と CVF の公開版で違うものがある（MME は IEEE 版 8049–8057・CVF 版 8050–8058，DECOTA は 8886–8896・8906–8916，CCSA は 5716–5726・5715–5725）．(2) DECOTA の題の "Co-Training" は PDF と CVF の頁の表記で，Crossref は "Co-training" である．(3) FADA の会議は当時の名称が NIPS 2017 で，論文集は "Advances in Neural Information Processing Systems 30" である．第 3 著者は PDF で Seyed Mehdi Iranmanesh，論文集の頁で Seyed Iranmanesh と表記が分かれる．(4) AdaMatch の references.bib のキーは `adamatch` ではなく `berthelot2022adamatch` である．
MME の仕組みについての依頼者の記憶は本文の式 3 と一致した．分類器 C は重みベクトルをクラスのプロトタイプ（代表点）として持ち，ラベル無しのターゲットでの予測のエントロピーを C は最大化し，特徴抽出器 F は最小化する（θ̂_C = argmin L − λH，θ̂_F = argmin L + λH）．
SSDA の 5 本のうち MME・APE・CDAC・DECOTA の 4 本は，骨格網（AlexNet，VGG，ResNet-34）を微調整し，DomainNet の 126 クラス・4 ドメイン・7 組，Office-Home，Office で，ターゲットのラベルをクラスあたり 1 本と 3 本にした同じ評価設定を使う．MME・CDAC・DECOTA は ImageNet の事前学習を明記し，APE は書いていない．MME と DECOTA は，学習にラベル無しで使ったターゲットの標本そのもので評価すると明記し，APE と CDAC は MME の設定に合わせると書く．MME と DECOTA はつまみをターゲットのクラスあたり 3 本の検証用ラベルで選ぶ．固定した特徴の上の線形の分類器を学ぶのは Donahue 2013 だけである．

### Cited Findings
以下，論文ごとに 1 書誌と読んだ範囲，2 設定，3 方法が変えるもの・骨格網・つまみの選び方，4 示したこと，5 本研究との違い，6 注意，の順で書く．

#### 1. MME — Saito, Kim, Sclaroff, Darrell, Saenko, "Semi-Supervised Domain Adaptation via Minimax Entropy"

**1. 書誌と読んだ範囲**
- 書誌：Kuniaki Saito, Donghyun Kim, Stan Sclaroff, Trevor Darrell, Kate Saenko（Boston University，UC Berkeley）．2019 IEEE/CVF International Conference on Computer Vision (ICCV)，Seoul, Korea (South)，2019-10-27〜11-02．IEEE 版の頁 8049–8057，DOI 10.1109/ICCV.2019.00814（Crossref）．CVF の公開版の頁 8050–8058（CVF の頁の表記と PDF の頁番号）．月は CVF の BibTeX で October．
- 読んだ範囲：◎ 本文全節（1 Introduction，2 Related Work，3 Minimax Entropy Domain Adaptation（3.1 SSDA の定式化と模型，3.2 Training Objectives，3.3 Theoretical Insights），4 Experiments（4.1 Setup，4.2 Results，4.3 Analysis），5 Conclusion）と補足資料（1 Datasets，2 Implementation Detail，2.1 Baseline Implementation，3 Additional Results Analysis）．補足資料の表 3〜7 の数値と図 4 の数値は読んでいない．
- 出典：[CVF の頁](https://openaccess.thecvf.com/content_ICCV_2019/html/Saito_Semi-Supervised_Domain_Adaptation_via_Minimax_Entropy_ICCV_2019_paper.html)，[本文 PDF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Saito_Semi-Supervised_Domain_Adaptation_via_Minimax_Entropy_ICCV_2019_paper.pdf)，[補足資料](https://openaccess.thecvf.com/content_ICCV_2019/supplemental/Saito_Semi-Supervised_Domain_Adaptation_ICCV_2019_supplemental.pdf)，[Crossref](https://api.crossref.org/works?query.bibliographic=Saito+Kim+Sclaroff+Darrell+Saenko+Semi-Supervised+Domain+Adaptation+via+Minimax+Entropy)．

**2. 設定**
- ソースのラベル：ソースの全標本をラベル付きで学習に使う（"All examples of the source are used for training"）．
- ターゲットのラベル：クラスあたり 1 本（one-shot）か 3 本（three-shot）をランダムに選ぶ．これとは別に，クラスあたり 3 本をターゲットの検証用に取る．4.3 節の図 4 では，クラスあたり 0〜20 本（DomainNet の 126 クラスで合計 2,520 本まで）に振る．表 3 は 0 本（UDA）の結果である．
- ラベル無しのターゲット：残りのターゲットの標本をラベル無しで学習に使い，同じ標本のラベルで正解率を測る（"Our goal is to train the model on Ds, Dt, and Du and evaluate on Du"）．評価は転導的（学習に使ったラベル無しの標本で評価する設定）である．
- ベンチマーク：DomainNet の 4 ドメイン（Real，Clipart，Painting，Sketch）と 126 クラスを選び，ターゲットが Real でない 7 組を作る（ラベルの雑音が多いドメインとクラスを除いたと説明）．Office-Home（4 ドメイン，65 クラス，12 組），Office（31 クラス．Amazon をターゲットにした DSLR→Amazon と Webcam→Amazon の 2 組）．
- 指標：正解率．主表は 1 回の学習の値（補足資料 2 節 "The reported performance in the main paper is obtained by one-time training"）．補足資料に 3 回の平均と標準偏差（ResNet，DomainNet の Sketch→Painting）と，別のラベルの選び方での結果がある．

**3. 方法が変えるもの・骨格網・つまみの選び方**
- 模型（3.1 節）：特徴抽出器 F の出力を ℓ2 正規化し，分類器 C は重み行列 W = [w_1, …, w_K] と温度 T で p(x) = σ((1/T) Wᵀ F(x)/‖F(x)‖) を出す（余弦類似度の分類器．Chen ほか 2018 と Ranjan ほか 2017 の形）．各 w_i をクラス i のプロトタイプとみなす（"the weight vectors can be regarded as estimated prototypes for each class"）．W は乱数で初期化し，T = 0.05 は先行研究に従って固定する．
- 目的（3.2 節，式 1〜3）：L はラベル付きのソースとラベル付きのターゲットの交差エントロピー，H はラベル無しのターゲットでの予測のエントロピーである．θ̂_F = argmin_{θ_F} L + λH，θ̂_C = argmin_{θ_C} L − λH．分類器 C は H を最大化して重みベクトルをターゲットの特徴の方へ動かし（"we move the weight vectors towards target by maximizing the entropy of unlabeled target examples"），特徴抽出器 F は H を最小化してラベル無しの特徴をプロトタイプのまわりに集める．勾配反転層で 1 回の逆伝播にまとめる．
- 変えるもの：分類器の重み（プロトタイプの位置）と特徴の両方．
- 理論（3.3 節）：Ben-David ほか（NIPS 2007）の ε_t(h) ≤ ε_s(h) + d_H(p, q) + C_0 を引き，「予測のエントロピーが閾値 γ 以上なら 1」という領域の判定器を仮説に置くと，ソースでエントロピーが γ 以上の割合がターゲットでの割合以下という仮定のもとで，d_H がターゲットでエントロピーが γ 以上の割合の 2 倍の上限で抑えられる，と述べる．エントロピーの最大化を領域の差の測定，最小化を差の縮小と解釈する．定理と証明の形ではなく，仮定を置いた説明である．図 6c で A 距離（SVM の領域判別器で測る）が S+T より小さくなることを示す．
- 骨格網：ImageNet で事前学習した AlexNet，VGG16，ResNet34 の最後の線形層を外して F にする（ResNet34 は全結合層を 2 つ足す）．網全体を微調整する．補足資料：初期学習率は全結合層 0.01，畳み込み層と BN 層 0.001，momentum 0.9，学習率の減衰は Ganin & Lempitsky の方式．ミニバッチはラベル付き（ソースとターゲットが半分ずつ）s 本とラベル無しのターゲット 2s 本（AlexNet で s = 32，VGG・ResNet で 24）．左右反転とランダムな切り出しで拡張する．
- つまみの選び方：λ = 0.1 を DomainNet の Real→Clipart のターゲットの検証用ラベル（クラスあたり 3 本）での成績で決め，全実験で固定する（4.1 節 "This is decided by the validation performance on Real to Clipart experiments"，補足資料の図 2）．検証用ラベルは早期終了と学習の予定にも使う（"The validation examples are used for early stopping, choosing the hyper-parameter λ, and training scheduling"）．**ターゲットの検証用ラベルを使う．**

**4. 示したこと**
- DomainNet（表 1）：ResNet で 7 組の平均の 1-shot／3-shot が S+T 56.9／60.0，DANN 58.4／60.7，ADR 57.6／60.4，CDAN 62.5／66.5，ENT 62.6／67.6，MME 66.4／68.9．VGG で S+T 50.5／54.3，MME 59.2／62.1．AlexNet で S+T 40.0／43.4，ENT 29.1／39.8，MME 44.2／48.2．S+T はソースとターゲットのラベルだけで学ぶ模型，ENT はエントロピー最小化だけを網全体にかける模型である．
- UDA の手法（DANN，ADR，CDAN）にラベル付きのターゲットを足しても，S+T を上回らない場合がある（AlexNet の 3-shot で DANN 42.4，ADR 42.7，CDAN 41.0 に対し S+T 43.4）．ENT は AlexNet と VGG で S+T を大きく下回る．
- Office-Home と Office（表 2，VGG）：MME の 1-shot／3-shot は Office-Home 62.7／67.6，Office 73.4／77.0．
- ラベルの本数（図 4，0〜20 本）：ENT はクラスあたり 5 本以上で成績が上がり，本数を増やすと MME との差が縮む（"As we add more labeled examples, the performance gap between ENT and ours is reduced"）．UDA の設定（表 3）でも MME だけがすべての組でソースのみの模型を上回る．
- 分類器の形（表 4，AlexNet）：普通の線形層では Real→Clipart の 1-shot／3-shot が S+T 41.4／44.3，MME 44.9／47.7．余弦類似度の分類器では S+T 43.3／47.1，MME 48.9／55.6．普通の線形層でも MME は S+T を 3.4〜3.5 ポイント上回る．
- 補足資料：VAT（仮想敵対的訓練）は S+T を上回らない．ラベル付きとラベル無しを別々に流して BN の統計を分ける（Separate BN）かどうかで ENT の成績は大きく変わり，MME は変わりにくい．

**5. 本研究との違い（差の書き方の案）**
MME（Saito ほか，ICCV 2019）は ImageNet で事前学習した骨格網を微調整し，余弦類似度の分類器の重みをラベル無しのターゲットのエントロピーの最大化でターゲットへ動かし，特徴抽出器ではエントロピーを最小化する．つまみはターゲットのクラスあたり 3 本の検証用ラベルで選ぶ．本研究は凍結した特徴の上の線形の頭だけを学ぶので，特徴を動かす項を持たず，ターゲットのラベルを K 本より多く使わずにつまみをソースのコーパスで選ぶ点で異なる．
- 補足：MME の評価は学習に使ったラベル無しの標本で行い，本研究はラベル無しの候補からテスト話者を除く．MME は画像の物体認識（126／65／31 クラス）で単一モーダルである．

**6. 注意**
- 表 1 の見出しに "P: Clipart" とあるが，本文の定義では P は Painting である（原文の誤記）．
- MME は自分の理論の節で Ben-David ほかの境界を引く．本研究の中核理論（Ben-David ほか 2010）と同じ系統の境界を使うので，関連研究で並べるときは「MME は境界の d_H の項を手法の動機にし，本研究は境界の各項を測る」のように役割の違いを書ける（これは私の推論で，本研究の主張文は `paper/CLAIM.md` に合わせて決める）．

#### 2. APE — Kim & Kim, "Attract, Perturb, and Explore: Learning a Feature Alignment Network for Semi-supervised Domain Adaptation"

**1. 書誌と読んだ範囲**
- 書誌：Taekyung Kim, Changick Kim（KAIST）．Computer Vision – ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XIV，Lecture Notes in Computer Science 12359，Springer, Cham，pp. 591–607，DOI 10.1007/978-3-030-58568-6_35（Crossref．オンライン公開 2020-11-13）．編者は Andrea Vedaldi, Horst Bischof, Thomas Brox, Jan-Michael Frahm（Crossref の書籍の記録）．LNCS の巻番号 12359 は CiNii の書誌（Part 14，ISBN 9783030585679 が Crossref の ISBN と一致）で確かめた．ECVA の PDF のファイル名 123590579 の先頭も 12359 である．
- 読んだ範囲：◎ 本文全節（1 Introduction，2 Related Work（UDA，SSL，SSDA），3 Intra-domain Discrepancy，4 Proposed Method（4.1 定式化，4.2 Spherical Feature Space with Prototypes，4.3 Attraction，4.4 Perturbation，4.5 Exploration，4.6 Objective），5 Experiments（5.1 設定，5.2 結果，5.3 分析），6 Conclusion）と補足資料（5-shot・10-shot の表）．図 4 の数値は読んでいない．
- 出典：[ECVA の一覧](https://www.ecva.net/papers.php)，[本文 PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590579.pdf)，[補足資料](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590579-supp.pdf)，[Crossref（章）](https://api.crossref.org/works/10.1007/978-3-030-58568-6_35)，[Crossref（書籍）](https://api.crossref.org/works/10.1007/978-3-030-58568-6)，[CiNii](https://ci.nii.ac.jp/ncid/BC05543737)．

**2. 設定**
- ソースのラベル付き D_s，ターゲットのラベル付き D_t，ターゲットのラベル無し D_u．
- ターゲットのラベル：MME と比べるため同じ設定にそろえ，クラスあたり 1 本と 3 本（"For a fair comparison with the state-of-the-art SSDA method [26]"）．5.3 節の図 4 で Real→Clipart の 0〜20 本，補足資料で DomainNet の全 7 組の 5 本と 10 本．5 本と 10 本はターゲットの全標本の 1.8〜6.7%，3.6〜13.5% にあたる（補足資料）．
- ベンチマーク：DomainNet（4 ドメイン，126 クラス，7 組），Office-Home（12 組，表 2 は 3-shot だけ），Office（Webcam→Amazon，DSLR→Amazon，3-shot，AlexNet だけ）．
- 評価：MME と同じ設定にそろえると書く．評価用の標本の定義は本文に明記されない．

**3. 方法が変えるもの・骨格網・つまみの選び方**
- 分類器：Chen ほか（ICLR 2019）の類似度に基づくプロトタイプ分類器．正規化した特徴 f_θ(x) と学習するプロトタイプ p_k の内積を温度 T で割って softmax をとる（式 1）．
- 目的（式 6）：min_{p, θ} L_cls + αL_a + βL_e + γL_p．
  - attraction（L_a，式 3）：ラベル付き（ソース∪ターゲット）の特徴の分布とラベル無しのターゲットの特徴の分布の MMD（複数の幅のガウス RBF 核）を最小化する．著者は，ラベル付きのターゲットとその近傍だけがソースに揃い，残りが揃わずに残る「領域内のずれ（intra-domain discrepancy）」をこの項で縮めると説明する．
  - perturbation（L_p，式 4）：プロトタイプをエントロピーが大きくなる向きに動かし，その向きに大きさが ε 未満の摂動 r_x を求め，元の入力と摂動後の予測の KL を小さくする．ラベル無しとラベル付きのターゲットの両方にかける．
  - exploration（L_e，式 5）：要素ごとのエントロピーが閾値より小さいラベル無しのターゲットだけを選び，最も近いプロトタイプのクラスを疑似ラベルにして交差エントロピーで学ぶ．
- 変えるもの：特徴（MMD と摂動の正則化）とプロトタイプ（分類器の重み）の両方．MME と違い，敵対的な最小最大ではなく，埋め込み θ とプロトタイプ p について和を最小化する．
- 骨格網：AlexNet と ResNet-34．埋め込み f_θ の全パラメータを学ぶ（微調整）．ImageNet の事前学習は本文に書かれていない（MME と同じ設定にそろえると書くだけ）．SGD，初期学習率 0.01，momentum 0.9，weight decay 0.0005．ミニバッチはラベル付きのソースとラベル付きのターゲットを同数，ラベル無しのターゲットをその 2 倍（AlexNet 32，ResNet-34 24）．
- つまみの選び方：α，β，γ，摂動の大きさ，エントロピーの閾値の値も選び方も，本文と補足資料に書かれていない．

**4. 示したこと**
- 領域内のずれ（3 節，図 2）：ターゲットのラベルがクラスあたり 3 本でも，t-SNE で多くのターゲットの標本がソースに揃い，誤って予測される標本はソースから離れたまま残る（DomainNet の Real→Sketch，AlexNet）．
- DomainNet（表 1）：ResNet-34 で 1-shot／3-shot の平均が APE 67.6／71.7（MME 66.4／68.9）．AlexNet で 44.6／48.9（MME 44.2／48.2）．
- Office-Home 3-shot（表 2）：平均が ResNet で APE 74.0（MME 73.1），AlexNet で 55.6（MME 55.2）．Office 3-shot（表 3，AlexNet）：68.3（MME 67.6）．
- ラベルの本数（5.3 節，図 4）：ENT もラベルが増えると大きく伸び，クラスあたり 20 本（AlexNet）と 5 本（ResNet-34）を超えると SSDA の最良の手法を上回る（"it even outperformed the state-of-the-art SSDA methods when more than twenty and five target labels are given per class for the AlexNet and ResNet-34 backbone networks, respectively"）．著者は，ラベル付きのターゲットの割合が増えると SSDA が SSL（同じ分布の半教師あり学習）に近づくためと解釈する．
- 補足資料（ResNet-34，DomainNet 7 組の平均）：5-shot で S+T 63.3，DANN 61.4，CDAN 64.6，ENT 71.1，MME 71.2，APE 73.6．10-shot で S+T 67.3，DANN 64.6，CDAN 65.1，ENT 74.0，MME 73.0，APE 76.8．10 本では ENT が MME を上回る．DANN は 5 本と 10 本で S+T を下回る．
- 5.2 節：DANN と ADR は 1-shot では S+T をわずかに上回るが，ラベルの本数が増えると効果が小さくなるか負の転移になる（"these methods become less effective or even cause negative transfer as the number of the labeled target samples increases"）．
- 除去実験（表 4，AlexNet，3-shot）：S+T 43.4，DANN 42.4，MMD 43.8，VAT 42.5 に対し，項の組み合わせで 46.1〜48.3，3 項すべてで 48.9．

**5. 本研究との違い（差の書き方の案）**
APE（Kim & Kim，ECCV 2020）は，ターゲットの少数のラベルがその近傍だけをソースに揃えて領域内のずれを生むと指摘し，MMD・摂動の正則化・疑似ラベルで骨格網の特徴とプロトタイプを同時に学ぶ．本研究は特徴を凍結して線形の頭だけを学ぶ点で異なる．
- 補足：APE の補足資料は，クラスあたり 10 本でエントロピー最小化だけの ENT が MME を上回ることを示す．本研究で K を 50 まで振る設計と関わる観察である．

**6. 注意**
- 表 1 の S+T，DANN，ADR，CDAN，ENT，MME の値は，MME の論文の表 1 と全セルで一致する（私が照合した）．APE は比較相手を再実行せず MME の値を載せたとみられるが，本文に明記はない．
- 表 4 の見出しは「DomainNet の Real to Sketch」と書くが，表には 7 組と平均が並ぶ（原文の不整合）．チェック印が PDF の文字で取れず，どの行がどの項の組み合わせかは読み取れなかった．

#### 3. CDAC — Li, Li, Shi, Yu, "Cross-Domain Adaptive Clustering for Semi-Supervised Domain Adaptation"

**1. 書誌と読んだ範囲**
- 書誌：Jichang Li（The University of Hong Kong），Guanbin Li（Sun Yat-sen University），Yemin Shi（Deepwise AI Lab），Yizhou Yu（The University of Hong Kong）．2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)，Nashville, TN, USA，2021-06-20〜25．pp. 2505–2514（CVF の頁と Crossref で一致），DOI 10.1109/CVPR46437.2021.00253．arXiv:2104.09415（CVF の頁のリンク．arXiv の頁は開いていない）．コードは github.com/lijichang/CVPR2021-SSDA（本文の記載）．
- 読んだ範囲：◎ 本文全節（1 Introduction，2 Related Work，3 Methodology（3.1 SSDA，3.2 Adversarial Adaptive Clustering，3.3 Pseudo Labeling，3.4 Overall Loss），4 Experiments（4.1 Setups，4.2 Comparisons，4.3 Analysis），5 Conclusions）と補足資料（5-shot・10-shot の表，Cluster Core Distance の補足）．
- 出典：[CVF の頁](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Cross-Domain_Adaptive_Clustering_for_Semi-Supervised_Domain_Adaptation_CVPR_2021_paper.html)，[本文 PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Cross-Domain_Adaptive_Clustering_for_Semi-Supervised_Domain_Adaptation_CVPR_2021_paper.pdf)，[補足資料](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Li_Cross-Domain_Adaptive_Clustering_CVPR_2021_supplemental.pdf)，[Crossref](https://api.crossref.org/works?query.bibliographic=Li+Li+Shi+Yu+Cross-Domain+Adaptive+Clustering+for+Semi-Supervised+Domain+Adaptation)．

**2. 設定**
- ソースのラベル付き S，ターゲットのラベル無し U，ターゲットのラベル付き L（"only contains one shot or few shots per class"）．
- ターゲットのラベル：クラスあたり 1 本と 3 本．補足資料で 5 本と 10 本（DomainNet，ResNet34）．
- ベンチマーク：DomainNet（MME と同じ 4 ドメイン，126 クラス，7 組），Office-Home（3-shot，12 組），Office（AlexNet，W→A と D→A，1-shot と 3-shot）．組・本数・標本の選び方は既存の SSDA 研究（MME，UODA，APE）に合わせる（"the settings of our benchmark datasets refer to the existing SSDA approaches ... including adaptation scenarios of each dataset, the number of labeled target data ..., sample selection strategies, etc."）．
- 評価：ターゲットのテストデータ（"perform well on test data from the target domain"）．MME の設定に合わせるので，評価はラベル無しの集合で行うとみられるが，本文に定義はない．

**3. 方法が変えるもの・骨格網・つまみの選び方**
- 模型：特徴抽出器 G（AlexNet か ResNet34，ImageNet で事前学習）と分類器 F（バイアスの無い線形層と正規化層．線形層は乱数で初期化）．MME と同じ形である．
- 敵対的適応クラスタリング損失 L_AAC（3.2 節，式 3〜4）：ミニバッチ内のラベル無しのターゲットの対ごとに，特徴の大きさの上位 k = 5 個の添字の集合が一致すれば s_ij = 1，それ以外は 0 とする．予測の内積 p_iᵀ p′_j（p′_j は拡張した画像の予測）を s_ij に合わせる二値交差エントロピーである．
- 式 5：θ_G* = argmin L_CE + λL_AAC，θ_F* = argmin L_CE − λL_AAC．勾配反転層を使う．特徴抽出器は L_AAC を最小化してターゲットの特徴をクラスタにまとめ，分類器は L_AAC を最大化してソースに偏った表現を減らしクラスタ単位の整合を促す，と著者は説明する．MME の条件付きエントロピーを L_AAC に置き換えた形である（"we replace the conditional entropy loss with our adversarial adaptive clustering loss"）．
- 疑似ラベル L_PL（3.3 節，式 6）：FixMatch の手順．弱い変換の画像の予測の最大確率が τ 以上なら，その硬いラベルを別の変換 x″ の予測の交差エントロピーの目標にする．
- 一貫性 L_Con：2 つの変換の予測をそろえる．重みは係数 ν の ramp-up．データ拡張は RandAugment．
- 全体（式 7）：θ_G* = argmin L_CE + λL_AAC + L_PL + L_Con，θ_F* = argmin L_CE − λL_AAC + L_PL + L_Con．
- 変えるもの：特徴と分類器の両方（L_AAC について敵対的な最小最大）．
- 骨格網：微調整する．最適化法・学習率・ミニバッチの大きさは MME と同じ．各反復でまずラベル付きデータの交差エントロピーだけで学び，次にラベル無しの損失を足す．
- つまみの選び方：λ = 1.0，ν = 30.0，τ = 0.95 と書くだけで，選び方は書かれていない．

**4. 示したこと**
- DomainNet（表 1）：ResNet34 で 1-shot／3-shot の平均が CDAC 73.6／76.0，APE 67.6／71.7，UODA 68.5／71.2，MME 66.4／68.9，S+T 56.9／60.0．AlexNet で CDAC 52.1／56.2，BiAT 45.5／49.4，APE 44.6／48.9．
- Office-Home 3-shot（表 2）：平均が ResNet34 で CDAC 74.2，APE 74.0，MME 73.1，ENT 71.9，S+T 66.2，DANN 63.5．Office（表 3，AlexNet）：1-shot／3-shot の平均が CDAC 63.1／70.0，MME 56.5／67.6，S+T 50.2／61.8．
- 除去実験（表 4，ResNet34，DomainNet 3-shot の平均）：交差エントロピーだけ（S+T と同じ）60.0，4 つの損失すべて 76.0．ソースのラベルだけの UDA の設定でも 58.6 から 73.0 に上がる．
- 疑似ラベル（図 3 左）：正しい疑似ラベルの数は，各エポックの学習標本全体の最大 59.9%（Real→Sketch，3-shot）と 63.8%（Painting→Clipart，1-shot）．
- 補足資料（5-shot と 10-shot，ResNet34，DomainNet）：先行手法の最良を平均で 3.3 ポイントと 3.0 ポイント上回る．

**5. 本研究との違い（差の書き方の案）**
CDAC（Li ほか，CVPR 2021）は MME の枠組みのエントロピーを，特徴の上位の添字で作った対の類似度に基づくクラスタリング損失に置き換え，FixMatch 型の疑似ラベルと RandAugment を加えて骨格網を微調整する．本研究は凍結した特徴の上の線形の頭を学ぶので，画像のデータ拡張を前提にする疑似ラベルと一貫性の項は作れない．
- 補足：AdaMatch を相手にしない理由として `paper/CLAIM.md` が挙げる「凍結した特徴の上では弱い・強いデータ拡張を作れない」が，CDAC にも同じく当てはまる．

**6. 注意**
- 本文 4.3 節は「L_AAC と L_PL を足すと平均が 25.3% 上がる」と書くが，表 4 の値（60.0 から 75.3）との対応は相対値か絶対値か本文から決められなかった．引くときは表 4 の値を使う．

#### 4. DECOTA — Yang, Wang, Gao, Shrivastava, Weinberger, Chao, Lim, "Deep Co-Training with Task Decomposition for Semi-Supervised Domain Adaptation"

**1. 書誌と読んだ範囲**
- 書誌：Luyu Yang（University of Maryland），Yan Wang（Cornell University），Mingfei Gao（Salesforce Research），Abhinav Shrivastava（University of Maryland），Kilian Q. Weinberger（Cornell University），Wei-Lun Chao（Ohio State University），Ser-Nam Lim（Facebook AI）．2021 IEEE/CVF International Conference on Computer Vision (ICCV)，Montreal, QC, Canada，2021-10-10〜17．IEEE 版の頁 8886–8896，DOI 10.1109/ICCV48922.2021.00878（Crossref）．CVF の公開版の頁 8906–8916．arXiv:2007.12684（CVF の頁のリンク．arXiv の頁は開いていない）．CDAC の参考文献 [42] は同じ arXiv 番号を "MiCo: Mixup co-training for semi-supervised domain adaptation"（2020）として引く．
- 題の表記：PDF と CVF の頁は "Deep Co-Training with Task Decomposition ..."（CVF の頁は "With" も大文字），Crossref は "Deep Co-training with Task Decomposition ..." である．
- 読んだ範囲：◎ 本文全節（1 Introduction，2 Related Work，3 Deep Co-training with Task Decomposition（3.1 概要と MixUp，3.2 co-training の条件，3.3 他の co-training との比較），4 Experiments，5 Conclusion）と補足資料（A 実験設定，B.1〜B.12）．表 4，表 7，表 8，表 10〜12 の数値は PDF の文字で一部しか取れず，読んでいない．
- 出典：[CVF の頁](https://openaccess.thecvf.com/content/ICCV2021/html/Yang_Deep_Co-Training_With_Task_Decomposition_for_Semi-Supervised_Domain_Adaptation_ICCV_2021_paper.html)，[本文 PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Yang_Deep_Co-Training_With_Task_Decomposition_for_Semi-Supervised_Domain_Adaptation_ICCV_2021_paper.pdf)，[補足資料](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Yang_Deep_Co-Training_With_ICCV_2021_supplemental.pdf)，[Crossref](https://api.crossref.org/works?query.bibliographic=Yang+Wang+Gao+Shrivastava+Weinberger+Chao+Lim+Deep+Co-Training+Task+Decomposition)．

**2. 設定**
- ソースのラベル付き D_S，ターゲットのラベル付き D_T，ターゲットのラベル無し D_U．例：DomainNet の Real→Clipart の 3-shot で |D_S| = 70,358，|D_U| = 18,325，|D_T| = 378（126 クラス × 3 本）．
- ターゲットのラベル：MME に従いクラスあたり 1 本と 3 本．主表は 3-shot，1-shot は補足資料．補足資料 B.4 で 10・20・50 本（DomainNet，ResNet-34．ターゲットから標本を追加でランダムに選ぶ）．
- ラベル無しのターゲット：D_U で学び，その後 D_U の正解を明かして評価する（"We train with DS, DT, and unlabeled DU. We then reveal the true label of DU for evaluation."）．評価は転導的である．
- ベンチマーク：DomainNet（4 ドメイン，126 クラス，7 組．ResNet-34），Office-Home（12 組．VGG-16．補足資料で ResNet-34），Office-31（補足資料，Webcam→Amazon と DSLR→Amazon，ResNet-34）．

**3. 方法が変えるもの・骨格網・つまみの選び方**
- 問題の分け方：SSDA を 2 つの部分問題に分ける．w_g は D_S と D_U で学ぶ UDA の模型，w_f は D_T と D_U で学ぶ SSL の模型である．各反復で，一方の模型の最大確率が τ を超えたラベル無しの標本にその予測を疑似ラベルとして付け，もう一方の模型の学習に渡す（式 1．co-training）．
- MixUp：疑似ラベル付きの標本をラベル付きの標本と Beta(α, α) の比で混ぜて学ぶ（U(f) と T，U(g) と S の組み合わせ．式 2〜3）．
- 予測：2 つの模型の出力確率の平均．
- 変えるもの：網全体（特徴と分類器）を 2 本学ぶ．特徴の分布を揃える項と敵対的学習は無い（"D E C OTA requires no adversarial training"）．
- 理論との対応（3.2 節）：co-training が成り立つ ε-expandability（Balcan ほか 2005）の緩めた条件を，「ちょうど一方の模型だけが確信を持つ標本の数」で実測する．Real→Clipart の 3-shot で学習の終わりにもその割合が 14% あり，ε は最初の 500 反復の後おおよそ 2 で推移する（図 3a）．
- 骨格網：ImageNet で事前学習した ResNet-34（DomainNet）と VGG-16（Office-Home）の最後の線形層を K 方向の余弦分類器に替え，温度 0.05 で学ぶ．w_f は D_S で微調整した模型で，w_g は D_S で微調整した後に D_T でも微調整した模型で初期化する（本文の記述のまま．2 つの模型を最初から違わせるため）．SGD，momentum 0.9，初期学習率 0.001，各ミニバッチ 24 本，DomainNet 5 万反復，Office-Home 1 万反復．
- つまみの選び方：τ = 0.5，α = 1.0．補足資料 A：「MME に従い，ターゲットのクラスあたり 3 本の別のラベル付き標本で選ぶ．DomainNet の 3-shot の Real→Clipart だけで選び，他の実験では固定する」（"We follow [51] to select these hyper-parameters using three other labeled examples per class in the target domain"）．**ターゲットの検証用ラベルを使う．**10・20・50 本の表では SSL の比較相手も DA の比較相手と同じように調整したと書く．

**4. 示したこと**
- DomainNet 3-shot（表 2，ResNet-34）：7 組の平均が DECOTA 75.6，APE 71.7，ELP 71.6，UODA 71.2，MME 68.0，ENT 63.9，S+T 61.2，DANN 60.5．1-shot（補足資料の表 5）：DECOTA 73.9，ELP 69.0，UODA 68.5，APE 67.6，MME 65.3，S+T 58.1．
- Office-Home 3-shot（表 3，VGG-16）：平均が DECOTA 69.3，UODA 68.9，ELP 68.0，MME 67.6，APE 66.5，ENT 64.8，DANN 63.9，S+T 62.9．
- 部分問題に分けず 1 つの模型で疑似ラベルと MixUp を使う MiST より，DomainNet で 1 ポイント，Office-Home で 5 ポイント高い（3-shot）．MixUp を外した S+T+pseudo-U は MiST より DomainNet の 3-shot の平均で 9 ポイント低い．
- ラベルを増やすと（補足資料の表 9，DomainNet，ResNet-34．例は Real→Clipart）：10／20／50 本で S+T 69.1／72.4／77.5，ENT 77.9／80.0／83.0，MME 77.0／78.5／80.9，FixMatch 76.6／79.5／82.3，DECOTA 81.8／82.6／85.0．DECOTA と S+T の差は 12.7，10.2，7.5 ポイントと縮む（私の計算）．50 本で ENT と FixMatch が MME を上回る．
- 動機（1 節）："A single classifier learned together with both sources of supervision is thus easily dominated by the labeled source data and is unable to take advantage of the additional labeled target data."
- w_g はソースのテスト標本で，ソースだけで学んだ模型と同程度の正解率を保つ（表 4e，補足資料 B.8）．
- 閾値の感度（補足資料 B.9）：τ ≤ 0.7 で差は小さく，τ ≥ 0.9 で大きく下がる．

**5. 本研究との違い（差の書き方の案）**
DECOTA（Yang ほか，ICCV 2021）は，ソースとターゲットのラベルを 1 つの分類器に合併するとソースのラベルに支配されると述べ，ソースのラベルで学ぶ模型とターゲットのラベルで学ぶ模型を分けて疑似ラベルを交換し，2 つの骨格網を微調整する．本研究は合併の得と損を凍結した特徴の上の線形の頭で測り，つまみをターゲットの検証用ラベル無しで選ぶ点で異なる．
- 補足：「合併するとソースに支配される」という DECOTA の動機は，本研究がソースの合併で損が出る条件を調べる問題意識と重なる．DECOTA はこの動機を手法で扱い，損の大きさや条件は測っていない．

**6. 注意**
- DECOTA の表の S+T と MME の値は MME の論文の値と違う（DomainNet 3-shot の平均で，MME の論文は S+T 60.0，MME 68.9，DECOTA の表 2 は 61.2，68.0）．比較相手を再実行したとみられるが，本文に明記はない．APE の値（71.7）は APE の論文と一致する．

#### 5. Donahue, Hoffman, Rodner, Saenko, Darrell, "Semi-supervised Domain Adaptation with Instance Constraints"

**1. 書誌と読んだ範囲**
- 書誌：Jeff Donahue（UC Berkeley EECS・ICSI），Judy Hoffman（同），Erik Rodner（UC Berkeley ICSI，University of Jena），Kate Saenko（University of Massachusetts Lowell），Trevor Darrell（UC Berkeley EECS・ICSI）．2013 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)，Portland, OR, USA，2013-06-23〜28．pp. 668–675（PDF の頁番号，CVF の頁，Crossref で一致），DOI 10.1109/CVPR.2013.92（PDF の 1 頁目にも記載）．
- 読んだ範囲：◎ 本文全節（1 Introduction，2 Related Work，3 Domain Adaptation with Instance Constraints（3.1 類似度のグラフ，3.2 DA の模型，3.3 最適化），4 Multi-category adaptation，5 Video domain adaptation，6 Experiments（6.1 多クラス分類，6.2 動画の物体検出），7 Conclusion）．補足資料は CVF の頁に無い．
- 出典：[CVF の頁](https://openaccess.thecvf.com/content_cvpr_2013/html/Donahue_Semi-supervised_Domain_Adaptation_2013_CVPR_paper.html)，[本文 PDF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Donahue_Semi-supervised_Domain_Adaptation_2013_CVPR_paper.pdf)，[Crossref](https://api.crossref.org/works?query.bibliographic=Donahue+Hoffman+Rodner+Saenko+Darrell+Semi-supervised+Domain+Adaptation+with+Instance+Constraints)．

**2. 設定**
- 多クラス分類（6.1 節）：Office（Saenko ほか 2010）の webcam をソース（795 枚，31 クラスすべてにクラスあたり 8 本のラベル），dslr をターゲット（498 枚．ラベルは最初の 16 クラスにクラスあたり 1 本，合計 16 本．残り 15 クラスはターゲットのラベル無し）とする．ラベル無しのターゲットには「同じ物体を別の視点から撮った」というインスタンス制約が付く．特徴は 800 次元の SURF の bag of words を PCA で次元を落としたもの．10 通りの分割の平均．
- 物体検出（6.2 節）：PASCAL VOC 2007 で学んだ人物検出器（DPM の 1 成分，部品なし）を VisInt の動画に適応する．ターゲットのラベル付きフレームは N_f = 10，20，30，40（1 本の動画から 10 フレーム），ラベル無しは 5N_f フレームで，追跡で得た「同じ物体」の情報を持つ．指標は AP．
- ラベル無しのターゲットの使い方：一般のラベル無し標本としてではなく，同じ物体どうしの対の関係（インスタンス制約）を持つ標本として使う．
- K の単位：多クラスではクラスあたり 1 本（半分のクラスだけ）．

**3. 方法が変えるもの・骨格網・つまみの選び方**
- 学習器：固定した特徴の上の線形の分類器 f̃(x̃) = θ̃ᵀx̃（3.1 節 "we focus on linear models due to their efficiency at test time"）．
- 一般形（式 1）：正則化 R(θ, θ̃, A) ＋ C・ソースの損失 ＋ C̃・ターゲットの損失．これにラベル無しのターゲットの Laplacian 正則化 r = ½ Σ_{j≠j′} β_{jj′} (f̃(x̃_j) − f̃(x̃_{j′}))²（式 2）を足す．β_{jj′} はインスタンス制約の重みで，同じ物体の対に正の値を置く（動画ではフレームの距離で減らす）．Laplacian SVM（Melacci & Belkin 2011）を DA に広げたものと位置づける．
- 2 つの DA の型に入れる：
  - PMT-SVM（Aytar & Zisserman 2011）：学習済みのソースの超平面 θ を固定し，ターゲットの超平面 θ̃ を ½‖θ̃‖² + (Γ/2)‖θ̃‖² sin² α(θ̃, θ) で正則化する（式 3．α は 2 つの超平面のなす角，θᵀθ̃ ≥ 0 の制約つき）．ターゲットの分類器の重みの向きをソースの重みの向きに近づける罰則である．損失はヒンジ．検出の実験で使う．
  - MMDT（Hoffman ほか 2013）：ターゲットからソースへの線形変換 A とソースの超平面 θ を同時に学び，θ̃ = Aᵀθ とする．A と単位行列の差に ‖A − I‖²_F の罰則をかける（式 4）．1 対他でクラスごとの超平面を学び，A を全クラスで共有するので，ターゲットのラベルが無いクラスの分類器にも同じ A がかかる．多クラスの実験で使う．
- 変えるもの：分類器の重み（PMT-SVM）か，特徴の線形変換と分類器（MMDT）．深い骨格網は無い．
- つまみの選び方：多クラスは「da only（Laplacian 項なし）の比較相手の交差検証」で決める（"Algorithm hyperparameters for all methods were set by cross-validation of the da only baseline"）．検出は (C̃, Γ, β) を「取り置いたデータでの交差検証」で決める．交差検証にどのドメインのどのラベルを使ったかは本文に書かれていない．

**4. 示したこと**
- 多クラス（表 2，ターゲットの全 31 クラスのテスト）：svm_S 45.80 ± 2.2，svm_S∪T 50.79 ± 2.4，da only 54.57 ± 2.2，da + lap-sim 56.15 ± 2.7．
- ターゲットのラベルが無い 15 クラスだけ（表 3）：35.46 ± 1.0，34.15 ± 0.9，35.79 ± 1.4，39.89 ± 1.3．DA だけではラベルの無いクラスはほとんど上がらず，インスタンス制約を足すと 4.1 ポイント上がる．ソースとターゲットのラベルの合併（svm_S∪T）は，ラベルの無いクラスでソースだけ（svm_S）を 1.3 ポイント下回る．
- 検出（表 5，AP）：N_f = 10 で dpm_S 0.3220，dpm_S∪T 0.3502，da only 0.1121，da + lap-sim 0.3317．N_f = 20 で 0.3220，0.3473，0.3530，0.3913．N_f = 30 と 40 では da only と da + lap-sim がほぼ同じ（0.4306 と 0.4303，0.4538 と 0.4584）．ラベル付きのフレームが増えると制約の得が小さくなる（著者も記述）．

**5. 本研究との違い（差の書き方の案）**
Donahue ほか（CVPR 2013）は固定した特徴の上の線形の分類器で SSDA を扱い，ソースの超平面への角の罰則（PMT-SVM）か特徴の線形変換（MMDT）に，ラベル無しのターゲットの「同じ物体」の対に予測をそろえる Laplacian 正則化を加える．本研究も固定した特徴の上の線形の頭を学ぶが，ラベル無しの候補に対の制約は無く，ターゲットのラベルを全クラスにクラスあたり K 本（0〜50）与え，多クラスのロジスティック回帰で学ぶ点で異なる．
- 補足：PMT-SVM の「ターゲットの重みとソースの重みのなす角への罰則」は，本研究の仮説転移の罰則 ‖W − W_s‖² と同じ系統の正則化である（角か二乗距離か，切片を扱うかが違う）．Donahue ほかの多クラスの実験は webcam→dslr の 1 組だけで，単一モーダルである．

**6. 注意**
- 2013 年の論文は，ターゲットのラベルの有無で「supervised」（ラベル付きのターゲットだけ）と「semi-supervised」（ラベル無しのターゲットも使う）を分ける（1 節 "Supervised approaches do this by utilizing a few labeled examples in the target domain ..., while semi-supervised methods also take into account the (typically much more abundant) unlabeled target samples"）．FADA と CCSA の「教師あり DA」の用語はこの区分と一致する．

#### 6a. FADA — Motiian, Jones, Iranmanesh, Doretto, "Few-Shot Adversarial Domain Adaptation"

**1. 書誌と読んだ範囲**
- 書誌：Saeid Motiian, Quinn Jones, Seyed Mehdi Iranmanesh, Gianfranco Doretto（West Virginia University）．Advances in Neural Information Processing Systems 30（NIPS 2017，Long Beach, CA, USA，2017-12-04〜09）．pp. 6670–6680（論文集の Metadata.json の page_first・page_last．論文集の BibTeX の pages は空）．出版社は Curran Associates, Inc.（論文集の BibTeX）．第 3 著者は PDF で "Seyed Mehdi Iranmanesh"，論文集の頁と BibTeX で "Seyed Iranmanesh"．CDAC の参考文献 [25] も pp. 6670–6680 で引く．
- 読んだ範囲：◎ 本文全節（1 Introduction，2 Related work，3 Few-shot adversarial domain adaptation（3.1 Handling Scarce Target Data），4 Experiments（4.1 MNIST-USPS-SVHN，4.2 Office），5 Conclusions）．論文集の頁に補足資料は無い．
- 出典：[論文集の頁](https://proceedings.neurips.cc/paper_files/paper/2017/hash/21c5bba1dd6aed9ab48c2b34c1a0adde-Abstract.html)，[本文 PDF](https://proceedings.neurips.cc/paper_files/paper/2017/file/21c5bba1dd6aed9ab48c2b34c1a0adde-Paper.pdf)，[Metadata.json](https://proceedings.neurips.cc/paper_files/paper/2017/file/21c5bba1dd6aed9ab48c2b34c1a0adde-Metadata.json)，[会期](https://neurips.cc/Conferences/2017)．

**2. 設定**
- 教師あり DA（SDA）：ソースのラベル付き D_s とターゲットのラベル付き D_t（クラスあたりごく少数．1 本の場合や，ラベルの無いクラスがある場合も想定する）だけを使う．**ラベル無しのターゲットは使わない**（"UDA methods use all target samples in their training stage, while we only use very few labeled target samples per category in training"）．
- 数字（4.1 節）：MNIST，USPS，SVHN の 6 組（M→U，U→M，S→M，M→S，S→U，U→S）．ターゲットからクラスあたり n 本を選ぶ．表 1 は n = 1〜7（本文は "n ranging from 1 to 4" と書き，表と食い違う）．10 回の反復の平均．画像は 16×16 に縮める．
- Office（4.2 節）：Tzeng ほか（ICCV 2015）の設定と分割をそのまま使う．31 クラス，5 通りの分割．ソースは Amazon でクラスあたり 20 本，Webcam と DSLR で 8 本．ターゲットはクラスあたり 3 本，残りのターゲットでテスト．A→W，A→D，W→A，D→A の 4 組（表 2 は 6 組を載せる）．

**3. 方法が変えるもの・骨格網・つまみの選び方**
- 標本の対を 4 群に分ける：G1 はソース同士で同じクラス，G2 はソースとターゲットで同じクラス，G3 はソース同士で違うクラス，G4 はソースとターゲットで違うクラス．最も小さい G2 の全対を使い，他の群は G2 と同数を一様に抜く．
- 領域とクラスの判別器 DCD（全結合 2 層，4 出力）を，埋め込み g を止めて学ぶ（式 3）．次に DCD を止めて g と予測 h を更新する（式 5）：γ ×（G2 を G1 と，G4 を G3 と取り違えさせる損失）＋ ソースの分類損失 ＋ ターゲットの分類損失．これを交互に繰り返す（算法 1）．
- 変えるもの：特徴（埋め込み g．重み共有 g_s = g_t）の側．分類器 h はソースとターゲットのラベルで同時に学ぶ．
- 骨格網：数字は LeNet 型の小さな CNN（畳み込み 2 層，全結合 120・84）をソースで学んでから更新する（事前学習なし，Adam）．Office は VGG-16 の畳み込み層（ImageNet で事前学習）に全結合層 2 つ（1024，128）を足し，全結合層をソースで初期化してから網を更新する．
- つまみの選び方：γ の値も選び方も本文に書かれていない．

**4. 示したこと**
- 数字（表 1）：n = 1 で M→U は FT（ソースで学んだ模型をターゲットで微調整）82.3，CCSA 85.0，FADA 89.1．U→M は 72.6，78.4，81.1．S→M は FT 65.5，FADA 72.8．M→S は 29.7，37.7．S→U は 69.4，78.3．U→S は 19.9，27.5．n を増やすと上がり，M→U の n = 7 で 94.4．
- 著者の記述：クラスあたり 1 本（合計 10 本）で，ラベル無しのターゲットを数千本使う UDA の最良の値に近い値が出る．
- Office（表 2，3-shot の 6 組の平均）：ソースだけ 68.2，UDA の 3 手法 70.6〜73.6，Tzeng 2015 82.2，Koniusz 82.6，CCSA 85.8，FADA 84.9．Office では FADA は CCSA を平均で 0.9 ポイント下回る．
- 重み共有を外すと U→M（n = 5）で 84.1 に下がる（共有した場合の表 1 の値は 91.1）．

**5. 本研究との違い（差の書き方の案）**
FADA（Motiian ほか，NIPS 2017）はラベル無しのターゲットを使わない教師あり DA で，ソースとターゲットの標本の対を 4 群に分けた判別器を欺くように深い埋め込みを学び，ターゲットのラベルがクラスあたり 1〜7 本の範囲を調べる．本研究の地図（ソースのラベルと K 本の合併）はこの教師あり DA の設定にあたるが，本研究は特徴を凍結して線形の頭だけを学ぶ点で異なる．

#### 6b. CCSA — Motiian, Piccirilli, Adjeroh, Doretto, "Unified Deep Supervised Domain Adaptation and Generalization"

**1. 書誌と読んだ範囲**
- 書誌：Saeid Motiian, Marco Piccirilli, Donald A. Adjeroh, Gianfranco Doretto（West Virginia University）．2017 IEEE International Conference on Computer Vision (ICCV)，Venice（Crossref の開催地の表記），2017-10-22〜29．IEEE 版の頁 5716–5726，DOI 10.1109/ICCV.2017.609（Crossref）．CVF の公開版の頁 5715–5725．
- 読んだ範囲：◎ 本文全節（1 Introduction，2 Related work，3 Supervised DA with Scarce Target Data（3.1 Deep SDA，3.2 Handling Scarce Target Data），4 Extension to Domain Generalization，5 Experiments（5.1 DA，5.2〜5.4 DG），6 Conclusions）．
- 出典：[CVF の頁](https://openaccess.thecvf.com/content_iccv_2017/html/Motiian_Unified_Deep_Supervised_ICCV_2017_paper.html)，[本文 PDF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Motiian_Unified_Deep_Supervised_ICCV_2017_paper.pdf)，[Crossref](https://api.crossref.org/works?query.bibliographic=Motiian+Piccirilli+Adjeroh+Doretto+Unified+Deep+Supervised+Domain+Adaptation+and+Generalization)．

**2. 設定**
- 教師あり DA（ラベル無しのターゲットは使わない）と領域汎化（DG．ターゲットのデータを学習に使わない）．
- Office（5.1.1 節）の 3 つの実験：(1) Tzeng 2015 の設定（31 クラス，ターゲットはクラスあたり 3 本，5 分割，6 組，VGG-16 を ImageNet の事前学習から微調整）．(2) 15 クラスだけにターゲットのラベルがある設定で，ラベルの無い 16 クラスで評価する（原文 "only 10 target labeled samples of 15 classes"．10 本がクラスあたりか合計かは原文から決められない）．(3) Office の元の分割で 10 クラス．DeCaF-fc6 の特徴か 800 次元の SURF の特徴を入力にし，全結合 2 層（1024・128，または 512・32）の埋め込みと全結合 1 層の予測を学ぶ．
- MNIST-USPS（5.1.2 節）：M→U と U→M，クラスあたり n = 1〜8 本，10 回の反復．
- DG：VLCS（DeCaF-fc6 の特徴，全結合 2 層），回転 MNIST．

**3. 方法が変えるもの・骨格網・つまみの選び方**
- 損失（式 5）：L_CCSA = L_C（分類）＋ L_SA（意味の整合．同じクラスのソースとターゲットの埋め込みを近づける）＋ L_S（分離．違うクラスのソースとターゲットの埋め込みをマージン m 以上離す）．分布の距離を点の対の距離の平均で置き換え（式 7〜8），L_SA ＋ L_S は contrastive loss（Hadsell ほか 2006）と同じ形になる（式 9〜10）．分類の項を γ，整合と分離の項を 1 − γ で重みづける．
- 予測 h はソースだけで学び，ターゲットの少数ラベルで微調整してよい（式 6）．
- 変えるもの：埋め込み（特徴）の側．
- 骨格網：(1)(2) は VGG-16 の全重みを分割ごとに微調整する．(3) と DG の VLCS は固定した特徴の上の全結合網を学ぶ．
- つまみの選び方：γ と m の値も選び方も本文に書かれていない．

**4. 示したこと**
- Office 31 クラス 3-shot（表 1，6 組の平均）：ソースだけ 68.2，Tzeng 2015 82.21，Koniusz 82.68，CCSA 85.8．
- 15 クラスだけラベルがある設定（表 2）：ソースだけ 62.26，Tzeng 2015 66.4，CCSA 67.83．
- 10 クラス，固定した特徴（表 3）：DeCaF-fc6 でソースだけ 85.4，RTML 90.55，CCSA 95.4．SURF でソースだけ 30.6，RTML 49.8，CCSA 63.2．
- MNIST-USPS（表 4，M→U と U→M の平均）：ソースだけ 62.0，CCSA-1 81.7，CCSA-2 85.5，CCSA-3 87.9，CCSA-4 88.7，CCSA-5 90.1，CCSA-6 91.3，CCSA-7 91.1，CCSA-8 91.4．ラベル無しのターゲットを全部使う UDA の ADDA は 89.7，CoGAN は 90.1．
- 除去実験（図 4d，M→U）：分類＋整合と分類＋分離はどちらも微調整を上回り，3 項すべてが最も高い（数値は図のみ）．
- 重み共有を外すと M→U（n = 4）で 88.6 に下がる．

**5. 本研究との違い（差の書き方の案）**
CCSA（Motiian ほか，ICCV 2017）は，ラベル無しのターゲットを使わない教師あり DA で，同じクラスのソースとターゲットの埋め込みを近づけ違うクラスを離す対の損失で埋め込みを学ぶ．固定した特徴（DeCaF-fc6，SURF）を入力にする実験もあるが，学習器は全結合 2 層の埋め込みと予測層の網で，本研究の線形の頭とは異なる．

**6. 注意**
- 表 2 の見出しは「全 31 クラスでテスト」と書き，本文は「ラベルの無い 16 クラスで精度を計算」と書く（原文の不整合）．

#### 7. AdaMatch の SSDA の評価設定（Berthelot ほか，ICLR 2022．references.bib の `berthelot2022adamatch`）

**1. 書誌と読んだ範囲**
- 書誌は references.bib の `berthelot2022adamatch` のとおり（水準 ◎．本文は g3 の精読で読み済み）．今回は arXiv:2106.04732v2 の PDF（50 頁．各頁に "Published as a conference paper at ICLR 2022"）の抄録，1 節，3.3 節，4 節，5 節の冒頭，5.3 節，6 節の冒頭，付録 D.2 と付録 E の表の見出しを読み直した（◎ 該当節）．
- 出典：[arXiv PDF](https://arxiv.org/pdf/2106.04732v2)．

**2. 設定（SSDA に限る）**
- ターゲットのラベルの本数：1，5，10，25．抄録は "providing AdaMatch with just one labeled example per class from the target domain (i.e., the SSDA setting), we increase the target accuracy by an additional 6.1%, and with 5 labeled examples, by 13.6%"，1 節は "With just one label per class on the target dataset ... achieving a gain of 6.1% over UDA and 13.6% with 5 labels" と書く．付録 E の表の見出しは "num_target_labels=1"〜"=25" で，「クラスあたり」とは書かれていない．5.3 節は "we randomly sample a subset of the target dataset and treat it as labeled" と書く．抄録と 1 節の書き方から 1 本はクラスあたりで，5 本以上もクラスあたりと読むのが自然だが，表の注記では確かめられない（g3 の注意と同じ結論）．
- ベンチマーク：DigitFive（5 ドメイン，20 組，32×32），DomainNet（6 ドメイン，345 クラス，30 組，64×64 と 224×224）．DomainNet は MME 以来の 126 クラス・4 ドメイン・7 組の部分集合ではなく全体を使う．5 つの手法を全組と本数で回し，UDA・SSL・SSDA を合わせて 2,420 の実験．
- 学習：事前学習なし（"unless otherwise stated, we train models from scratch"）．WRN-28-2（32×32），WRN-34-2（64×64），ResNetV2-101（224×224）．
- SSDA での使い方（3.3 節）：ラベル付きのターゲットをソースのラベル付きのバッチに連結する（"feeding the concatenated batch X_STL = {X_SL, X_TL} in place of X_SL"）．このため uratio（ラベル無しとラベル付きの比）が実質半分になると注記する．
- つまみの選び方（4 節）：全課題で同じ値を使う．確信度の閾値は 0.8，0.9，0.95 の 3 つを 64×64 の UDA の clipart→infograph で試して最良を選ぶ（何の成績で選んだかは書かれていない）．uratio は 8 GPU に載る最大．学習率・学習率の減衰・weight decay は 64×64 のソースのラベル付きデータだけで 3 つの候補から粗く選ぶ．
- 評価：各組で最後の 10 個のチェックポイントの正解率の中央値をとり，全組で平均する．
- 結果（付録 D.2 の表 6）：DomainNet224 で 1／5／10／25 本の AdaMatch 34.8／42.3／46.7／52.8，FixMatch+ 28.3／38.7／44.1／50.8，BaselineBN 25.3／33.8／39.3／47.6．DomainNet64 で AdaMatch 30.4／36.5／40.6／46.4．DigitFive で AdaMatch 97.9／98.1／97.3／97.6．

**3. 本研究との違い（評価設定）**
AdaMatch の SSDA は，ターゲットのラベルが 1〜25 本（クラスあたりと読める），網をゼロから学び，つまみを全課題で固定する．本研究の K = 0〜50 の一部と重なる．AdaMatch の DomainNet は MME 系の 126 クラスの部分集合と違うので，MME・APE・CDAC・DECOTA の数値とは直接比べられない．

### Inferences（Q1）
- MME・APE・CDAC・DECOTA は同じ評価設定（DomainNet の 126 クラス・7 組，クラスあたり 1 本と 3 本，骨格網の微調整，MME と DECOTA が明記する転導的な評価）を共有し，値の比較もこの設定の中で行う．本研究とこの 4 本との差は，学習器（微調整した深い網か，凍結した特徴の上の線形の頭か），つまみの選び方（ターゲットの検証用ラベルか，ソースのコーパスを仮のターゲットにするか），評価用の標本の分け方（ラベル無しの集合で評価するか，テスト話者を除くか）の 3 点にまとめられる．
- APE の補足資料（10 本で ENT が MME を上回る）と DECOTA の補足資料（Real→Clipart の 50 本で ENT と FixMatch が MME を上回り，DECOTA と S+T の差が 12.7 から 7.5 ポイントに縮む）は，SSDA の専用の手法の得がラベルの本数とともに小さくなることを示す．本研究で K を 50 まで振る結果と並べて書ける．
- 教師あり DA の FADA と CCSA は，ラベル無しのターゲットを使わずに「ソースのラベル＋ターゲットの少数ラベル」で学ぶ点で，本研究の地図（合併）と同じ設定にあたる．

### Gaps（Q1）
- APE の α，β，γ，摂動の大きさ，エントロピーの閾値と ImageNet の事前学習の有無，CDAC の λ・ν・τ の選び方，FADA の γ，CCSA の γ と m，Donahue 2013 の交差検証に使ったデータは，本文に書かれていなかった．公開コードは見ていない．
- 図の数値（MME の図 4，APE の図 4，CDAC の図 3，DECOTA の図 3・7・8）は読んでいない．DECOTA の表 4，7，8，10〜12 は PDF の文字で数値が取れなかった．
- CDAC と DECOTA の arXiv 版と，MME の arXiv 版は開いていない．

## Q2. 8 本の設定の一覧

### Takeaway
固定した特徴の上で線形の分類器を学ぶのは Donahue 2013 だけである．CCSA は固定した特徴の上で小さな全結合網を学ぶ実験を 1 つ持つ．つまみをターゲットの検証用ラベルで選ぶと明記するのは MME と DECOTA で，AdaMatch はつまみを全課題で固定し，Donahue 2013 は交差検証で選ぶ．残りの 4 本（APE，CDAC，FADA，CCSA）は選び方を書いていない．

### Cited Findings

| 論文 | ターゲットのラベル（クラスあたり） | ラベル無しのターゲット | ベンチマーク | 学習器と骨格網 | 変えるもの | つまみの選び方 | 評価 |
|---|---|---|---|---|---|---|---|
| MME（ICCV 2019） | 1，3（図 4 で 0〜20） | エントロピーの最小最大 | DomainNet 126 クラス 7 組，Office-Home，Office | AlexNet・VGG16・ResNet34（ImageNet）を微調整，余弦分類器 | 分類器の重み（プロトタイプ）と特徴 | ターゲットの検証用ラベル（クラスあたり 3 本） | ラベル無しの集合（転導的） |
| APE（ECCV 2020） | 1，3（図 4 で 0〜20，補足で 5・10） | MMD，摂動の正則化，疑似ラベル | 同上 | AlexNet・ResNet-34 を微調整（事前学習の記載なし），プロトタイプ分類器 | 特徴とプロトタイプ | 記載なし | MME と同じ設定と記述 |
| CDAC（CVPR 2021） | 1，3（補足で 5・10） | 対の類似度のクラスタリング損失（最小最大），FixMatch 型の疑似ラベル，一貫性 | 同上 | AlexNet・ResNet34（ImageNet）を微調整，正規化した線形分類器，RandAugment | 特徴と分類器 | 値だけ記載 | ターゲットのテスト（定義は既存研究に従う） |
| DECOTA（ICCV 2021） | 1，3（補足で 10・20・50） | 2 つの模型の疑似ラベルの交換と MixUp | DomainNet 126 クラス 7 組，Office-Home，Office-31 | ResNet-34・VGG-16（ImageNet）を 2 本微調整，余弦分類器 | 網全体（2 本） | ターゲットの検証用ラベル（クラスあたり 3 本） | ラベル無しの集合（転導的） |
| Donahue（CVPR 2013） | 1（31 クラスのうち 16 クラスだけ） | 同じ物体の対に予測をそろえる Laplacian 正則化 | Office（webcam→dslr），PASCAL→VisInt の検出 | 固定した特徴（SURF＋PCA，DPM）の上の線形 SVM | 分類器の重み（PMT-SVM）か，特徴の線形変換と分類器（MMDT） | 交差検証 | ターゲットのテスト（10 分割） |
| FADA（NIPS 2017） | 1〜7（数字），3（Office） | 使わない | MNIST・USPS・SVHN，Office | 小さな CNN（ゼロから），VGG-16（ImageNet）を更新 | 特徴（埋め込み） | 記載なし | ターゲットのテスト |
| CCSA（ICCV 2017） | 1〜8（数字），3（Office），15 クラスだけにラベル | 使わない | MNIST・USPS，Office，VLCS（DG） | VGG-16 を微調整，固定した特徴の上の全結合網 | 特徴（埋め込み） | 記載なし | ターゲットのテスト |
| AdaMatch（ICLR 2022） | 1，5，10，25（クラスあたりと読める） | FixMatch 型の疑似ラベル，分布整合，相対閾値 | DigitFive 20 組，DomainNet 345 クラス 30 組 | WRN・ResNetV2-101 をゼロから | 網全体 | 全課題で固定（閾値は UDA の 1 組で選ぶ） | 最後の 10 チェックポイントの中央値 |
| 本研究 | 0〜50 | 候補（テスト話者を除く）に先生の見込みを弱いラベルとして使う | 感情の 7 群（音声・顔・言葉） | 凍結したエンコーダの特徴の上の多クラスロジスティック回帰 | 線形の頭の重みと切片 | ソースのコーパスを仮のターゲットにする | 話者を分けたテスト，UAR |

### Inferences（Q2）
- 本研究の行と比べると，学習器が一致するのは Donahue 2013 だけで，つまみの選び方が一致する論文は無い．つまみをターゲットのラベルで選ぶ慣行（MME，DECOTA）は，ターゲットのラベルが K 本しかない前提と両立しない．本研究がソースのコーパスを仮のターゲットにする理由として，この 2 本を引ける．

### Gaps（Q2）
- 本研究の行の「候補に先生の見込みを弱いラベルとして使う」は `paper/CLAIM.md` の E59〜E61 の記述から私が要約したもので，論文に書く語は CLAIM.md に合わせる必要がある．

## Q3. 本研究の設定に最も近いもの

### Takeaway
最も近いのは Donahue 2013 と MME の 2 本である．Donahue 2013 は学習器が近い（固定した特徴の上の線形の分類器で，ソースの分類器の重みとのなす角に罰則をかける SSDA）．MME は問題設定が近く，SSDA の標準の評価設定（ソースのラベル＋クラスあたり 1・3 本＋ラベル無しのターゲット）を定め，ラベル無しのターゲットで分類器の重みを動かし，Ben-David ほかの境界を動機に引く．

### Cited Findings
- Donahue 2013：「we focus on linear models due to their efficiency at test time」（3.1 節）．PMT-SVM の正則化 ½‖θ̃‖² + (Γ/2)‖θ̃‖² sin² α(θ̃, θ)（式 3）．ラベル無しのターゲットの Laplacian 正則化（式 2）．
- MME：θ̂_C = argmin L − λH，θ̂_F = argmin L + λH（式 3）．Ben-David ほかの境界との対応（3.3 節）．λ をターゲットの検証用ラベルで選ぶ（4.1 節）．
- 次点は CCSA の 3 つ目の Office の実験（DeCaF-fc6 と SURF の固定した特徴の上で学ぶ）と DECOTA（合併がソースに支配されるという動機）である．

### Inferences
- Donahue 2013 を近い先例として引くときの差は，ラベル無しの使い方（対の制約か，先生の見込みを弱いラベルにするか），ターゲットのラベルの与え方（半分のクラスに 1 本か，全クラスに K 本か），罰則の形（角か二乗距離か）である．
- MME を引くときの差は，学習器（微調整した深い網か，凍結した特徴の上の線形の頭か）とつまみの選び方（ターゲットの検証用ラベルか，仮のターゲットか）である．
- 両方とも画像の単一モーダルで，マルチモーダルの感情認識や話者の分離は扱わない．

### Gaps
- 本研究の新しさの書き方は `docs/related-work.md` と `paper/CLAIM.md` に合わせて決める．このノートは先行研究の側の事実だけを記録した．
