# 検証：誤りの分解の先行研究の報告（face.md，text.md，speech_fusion.md，emotion_round2.md）（2026-11-12）

1. 原文と照らした主張は 30 本の文献の約 70 項目である．食い違いは 7 件（重大 1，中 1，軽微 5）で，残りは原文と一致した．
2. 重大な食い違いは Reif & Schwartz 2024 の「同じ 16 件のラベル」である．LoRA は 16 件で学んだうえで，評価でも k 件のデモを文脈に入れる．k=16 の比べでは LoRA 側が 16＋16 件のラベルを見ている．ラベルの数がそろうのは「LoRA の k=0（16 件で学ぶだけ）」と「LOOC の k=16」の組で，図 3a ではこの組の値は base で 0.548 対 0.601，chat で 0.638 対 0.632 である．どちらでも LOOC は LoRA と同じか上回る．したがって「図では 5〜7 割」は k をそろえた場合に限った読みであり，emotion_round2 の「ほぼ埋まると要約しないほうがよい」という書き方の注意は根拠が弱い．
3. 中程度の食い違いは Amanda（Findings of ACL 2024）である．報告は「割合を合わせる項と表現を合わせる項を別々に外した」と書いた．原文で外したのは表現を合わせる項（CA）だけで，割合を合わせる項（LA）を単独で外した版は無い．図 2 は 7 組のうち 3 組だけを描いている．
4. 指定された文献はすべて本文を読めた．Haider の undertow は README，コード（metrics.py，analysis.py），コミットの日付，Hugging Face の記録で確かめた．閾値はターゲットのテストのラベルで選ぶことをコードで確かめた（`oracle_threshold_accuracy(y_true, margin)` を評価のコーパスの test に当てる）．
5. 前回読めなかった Raaijmakers & Kraaij 2010 は TNO の機関リポジトリから全文を読めた．CAST（Wang & Liu 2025）の本文は，IEEE（購読が要る）でも，Semantic Scholar・OpenAlex（公開版なし）・ResearchGate（403）でも読めなかった．代わりに筆頭著者の公開コード（github.com/smwanghhh/CAST）を読んだ．コードでの「予測確率」の扱いは，疑似ラベルを選ぶクラスごとの確信度の閾値である．分類器の切片を当て直して戻る誤りと戻らない誤りを分ける処理は無い．
6. 4 つの報告の判定（言葉・声・顔のどれにも完全一致は無い）は，原文に照らして妥当である．感情の実データで複数の情報源を切片側と向き側に分けて比べた研究（反例）は，検証の途中の検索でも見つからなかった．新しく見つかった関係する研究（声の公開の解析．査読なし）は 4 節に書いた．

---

## 1. 主張ごとの表

判定の記号：一致，食い違い（原文ではどうか），確かめられず（理由）．「報告」の欄の略号は E2＝emotion_round2.md，T＝text.md，S＝speech_fusion.md，F＝face.md である．

### 1.1 Haider "undertow"（E2 の最も近い研究）

| 主張 | 報告の値 | 原文の値 | 判定 | 引用 | URL |
|---|---|---|---|---|---|
| 分類器 | 凍結した distilbert-base-uncased の特徴＋線形の分類器，TF-IDF＋ロジスティック回帰 | 同じ．平均プーリング，L-BFGS のロジスティック回帰，C は学習側のコーパスの検証で選ぶ | 一致 | "on mean-pooled features from a frozen `distilbert-base-uncased`. The encoder is never updated; only the linear head is fitted." | https://github.com/UsmarHaider/undertow |
| コーパス | IMDB，SST-2，Yelp，Amazon，6,000 件ずつ | 学習 6,000，検証 2,000，テスト 4,000（SST-2 のテストは 856）．どれもクラスが 50/50 になるよう作った | 一致 | "Each corpus is cut to 6,000 training / 2,000 validation / 4,000 test rows, class-balanced by construction" | 同上 |
| 落ちの内訳（線形） | 5.87＝ranking 4.12＋placement 1.76（29.9%） | 同じ | 一致 | "Linear probe \| 5.87 pp \| 4.12 pp \| 1.76 pp \| 29.9%" | 同上 |
| 落ちの内訳（TF-IDF） | 8.30＝6.74＋1.57（18.9%） | 同じ | 一致 | "TF-IDF + LR \| 8.30 pp \| 6.74 pp \| 1.57 pp \| 18.9%" | 同上 |
| 落ちの計算 | 「12 組の平均で」 | total＝対角 4 セルの平均正解率 − 非対角 12 セルの平均正解率．placement＝12 セルの平均（オラクル − 実際），ranking＝対角の平均 − 12 セルのオラクルの平均 | 食い違い（軽微）．落ちは 12 組の差の平均ではなく，4 セルの平均と 12 セルの平均の差である．E2 の「ranking にはターゲットの難しさも入る」という指摘は定義から正しい | `"ranking_drop": float(id_acc - ood_oracle)`，`"placement_drop": float(ood_oracle - ood_acc)`（metrics.py の shift_decomposition） | https://github.com/UsmarHaider/undertow/blob/main/src/undertow/metrics.py |
| 閾値の選び方 | テストのラベルで選ぶオラクル | 評価のコーパスの test のラベルで全ての切れ目を試す | 一致 | "This is the accuracy a model *would* get if someone handed it labelled data from the new domain purely to re-place its boundary."．pipeline.py で `y = labels_of(eval_domain, "test")` | 同上（metrics.py，pipeline.py） |
| 最も目立つ例 | SST-2→Yelp の TF-IDF，正と予測 32.0%，AUROC 85.9，74.4%→77.5% | 同じ | 一致 | "predicts "positive" for only **32.0%** of a set that is 50% positive. Its AUROC there is a respectable 85.9 … moving it alone lifts accuracy from 74.4% to 77.5%." | 同上 |
| 作者・公開日・査読 | Haider，2026 年 8 月 16 日，査読なし | GitHub の利用者 UsmarHaider（README の末尾 "Built by Usmar Haider"），コミットは 2026-08-16，Hugging Face の Space（usmar/undertow）の lastModified は 2026-08-16．論文や査読の記載は無い | 一致 | "Built by [Usmar Haider](https://github.com/UsmarHaider)." | 同上，https://huggingface.co/api/spaces/usmar/undertow |
| 補足（報告に無い点） | — | テストはクラスが 50/50 にそろえてあるので，placement はクラスの割合のずれから来る誤りではない．README は「少数のラベルで直る」と書くが，少数のラベルで閾値を選ぶ実験は無い | 補足 | "A handful of labelled target examples fixes it."（実験は無い） | 同上 |

### 1.2 Reif & Schwartz 2024（NAACL）（T，E2）

| 主張 | 報告の値 | 原文の値 | 判定 | 引用 | URL |
|---|---|---|---|---|---|
| 課題の数と区分 | 279 課題，感情極性 39，感情の区分なし | 同じ．付録の表 2 に "Sentiment Analysis 39"，感情（emotion）の語は本文と付録に無い | 一致 | "Sentiment Analysis 39 37748" | https://arxiv.org/abs/2405.02743 |
| LoRA の設定 | T：「同じ 16 件のラベルを LOOC と LoRA に使い分けた」．E2：「LoRA は別の 16 件で学び，k 件のデモで評価する」「ラベルの数が同じ k=16」 | LoRA は 16 件の held-out の例で 5 エポック学び，評価でも同じ文脈 C（k 件のデモ）を使う．LoRA の 16 件とデモが同じ例か別の例かは書かれていない（64 件の候補から両方を取る） | 食い違い（重大）．k=16 では LoRA は 16 件で学んだうえで 16 件のデモも見る．「同じ 16 件」「ラベルの数が同じ」とは言えない．E2 の「別の 16 件」も原文には書かれていない | "training adapters on 16 held-out training examples for 5 epochs. Importantly, we use the same context C during both fine-tuning and evaluation." | 同上（3.3 節） |
| LOOC の中身 | k 件のデモの予測確率をラベルごとに平均して偏りを引く | 1 件ずつ抜いた文脈で各デモの確率を出し，同じラベルの中で平均してから全ラベルで平均し，CC と同じく W＝diag(p̂)⁻¹ で直す．対数確率ではクラスごとの切片を足すことにあたる | 一致 | "We first average p^i's associated with the same label ℓ … and then set p̂ as the mean of these intra-label averages" | 同上（5 節） |
| 本文の言い方 | "LOOC nearly closes the gap with LoRA-level performance" | 同じ | 一致 | "Further, LOOC nearly closes the gap with LoRA-level performance while improving upon it in both bias metrics" | 同上 |
| 図 3a の値（7B，k=16） | base：0.486，0.601，0.645．chat：0.585，0.632，0.680 | 画像から読み取った値：base 0.486，0.601，0.645．chat 0.584，0.632，0.681 | 一致（±0.005） | 図 3a（7B の段） | 同上（6 頁の図 3） |
| 取り戻せた割合 | k=16 で base 約 0.72，chat 約 0.49 | (0.601−0.486)/(0.645−0.486)＝0.72，(0.632−0.584)/(0.681−0.584)＝0.49 | 一致（ただし k をそろえた場合の比） | 同上 | 同上 |
| ラベルの数をそろえた比べ（報告に無い） | — | LoRA の k=0（16 件で学ぶだけ）：base 約 0.548，chat 約 0.638．LOOC の k=16（16 件のデモだけ）：base 0.601，chat 0.632 | 食い違い（重大）の続き．ラベル 16 件どうしで比べると LOOC は LoRA と同じか上回る．E2 の「ほぼ埋まると要約しないほうがよい．図の値では 5〜7 割」は，比べ方を 1 つに決めたときだけ成り立つ．引くときはどちらの比べ方かを書く必要がある | 図 3a | 同上 |
| 実線と破線 | E2：実線＝base，破線＝chat は「推測」 | 図 3c の凡例に "Model Type: — Base，-- Instruct" とある | 食い違い（軽微）．推測ではなく凡例に書いてある．読みは正しい | 図 3 の凡例 | 同上 |
| LoRA の後の偏り | RSD は高いまま，BiasScore は上がる，CC は性能も偏りも直さない | 同じ | 一致 | "BiasScore noticeably increases after LoRA fine-tuning, indicating that more subtle bias persists."，"CC improves neither performance nor bias" | 同上（4.3 節） |

### 1.3 Myung 2026（ToM4AI の論文集）（E2）

| 主張 | 報告の値 | 原文の値 | 判定 | 引用 | URL |
|---|---|---|---|---|---|
| 頁 | 109〜111 頁 | 目次では 110 頁から．本文は印刷の頁で 110〜111，参考文献が 112（PDF の 113〜115 枚目） | 食い違い（軽微） | "Toward Theory of Mind: BERT Learns and Uses Emotion Geometry in Two Phases … 110" | https://arxiv.org/abs/2603.18786 |
| 設定 | GoEmotions で学んだ BERT を SemEval-2018 E-c へ零ショットで移す．共通の 8 感情 | 同じ．BERT は微調整する（凍結ではない） | 一致 | "For zero-shot transfer evaluation, we retain only the eight emotion labels shared with GoEmotions" | 同上 |
| 外部での BERT の macro-F1 | 0.1416 → 0.4647 → 0.5431 | 0.5 固定 0.1416，共通の閾値 0.4647，感情ごとの閾値 0.5431 | 一致 | 表 1 "BERT (External) 0.5 Default 0.1402 0.1416 0.0856 / Global tuned 0.5042 0.4647 0.3434 / Per-class tuned 0.5970 0.5431 0.4370" | 同上 |
| EmotionBERT と内部の値 | 0.1979 → 0.4697 → 0.5846，内部の BERT 0.3144 → 0.4300 → 0.4528 | 同じ | 一致 | 表 1 | 同上 |
| 閾値を選ぶデータ | 「development split」で選ぶ．どのデータかは書かれていない | 同じ | 一致 | "with thresholds tuned on the development split." | 同上 |

### 1.4 Xian ほか（ICLR 2022）（E2）

| 主張 | 報告の値 | 原文の値 | 判定 | 引用 | URL |
|---|---|---|---|---|---|
| クラスごとの特徴のずれ | クラスごとの特徴の平均の差の ℓ1 距離をクラスで平均 | 同じ（式 1，2） | 一致 | "we use the ℓ1-distance between the feature means as the discrepancy measure" | https://rxian.github.io/assets/xian2022crosslingual.pdf |
| クラスの割合のずれ | 全変動距離 | 同じ | 一致 | "we compute the total variation between p^Y_S and p^Y_T" | 同上（2.2 節） |
| 相関 | どちらのずれも，もう一方をそろえたうえで macro-F1 と強く相関 | 同じ | 一致 | "observe strong correlations between performance and either of the two factors when controlling the other." | 同上 |
| 評価の集合 | 割合を変えて間引いた 500 個 | 500 個（各 2,500 件） | 一致 | "we perform our evaluations on 500 smaller datasets sub-sampled from MARC with various class priors (each contains 2,500 test examples)" | 同上 |
| 表 1（英語→日本語） | 37.37，42.62，44.91 | 零ショット 37.37，IWDA 42.62，IWDA (oracle) 44.91 | 一致 | 表 1 "Zero-shot … 37.37"，"IWDA … 42.62"，"IWDA (oracle) … 44.91" | 同上 |
| 割合の補正の入り方 | 表現を学び直す中に入る | 推定した重みで重みづけた特徴をそろえる目的関数の中に入る | 一致 | "The alignment objective minimizes source loss and the distributional discrepancy between w-weighted source features and target features." | 同上（3 節） |

### 1.5 Tran, Yin, Soleymani 2023（Interspeech）（S）

| 主張 | 報告の値 | 原文の値 | 判定 | 引用 | URL |
|---|---|---|---|---|---|
| 掲載先 | Interspeech 2023 | Interspeech 2023，636〜640 頁（ISCA の記録） | 一致 | ISCA Archive の tran23c | https://www.isca-archive.org/interspeech_2023/tran23c_interspeech.html |
| 相関 | 覚醒度で −0.714（特徴のずれ），−0.502（ラベルのずれ），2 つのずれの相関 0.285 | 同じ | 一致 | "the PCC between the feature shift and the regression performance is −0.714 while the PCC between the label shift and performance is −0.502 … the correlation between the feature shift and label shift is 0.285" | https://arxiv.org/abs/2309.02418 |
| 出力の補正 | 覚醒度の A-CCC：補正なし 0.531，平均だけ 0.528，広がりだけ 0.541 | PAPT-FT 0.531，+µ shift 0.528，+σ shift 0.541，+(µ,σ) 0.540 | 一致（「補正なし」は PAPT-FT のこと） | 表 2 | 同上 |
| 平均をずらすと下がる | 「平均をずらすと CCC が下がることが多い」 | 同じ | 一致 | "while µ shifting often reduces both A-CCC and O-CCC." | 同上 |

### 1.6 Amanda（Findings of ACL 2024）（S）

| 主張 | 報告の値 | 原文の値 | 判定 | 引用 | URL |
|---|---|---|---|---|---|
| 設定 | 3 モダリティ，IEMOCAP・MELD・MOSEI・MSP-IMPROV の 7 組，4 クラス，重みつき F1 | 同じ | 一致 | "we select samples in the four classes— neutral, happy, sad and angry"，"weighted F1 score as model performance metric" | https://aclanthology.org/2024.findings-acl.859.pdf |
| 割合を合わせる項 | ターゲットの予測の割合をソースの割合に合わせる．割合が同じと仮定 | 同じ（式 6〜8） | 一致 | "As assumption 2) states, the label distributions of the target and source domains remain the same." | 同上（3.1 節 C） |
| 部品を外した比べ | 「2 つの項を外して比べた」「別々に外して得を見た」 | 外したのは CA（表現を合わせる項）だけである．比べたのは D.T.（そのまま移す），A w/o CA（LA だけ），Amanda（LA＋CA）の入れ子の 3 段で，LA を単独で外した版は無い．図 2 は IEMOCAP→MSP，MOSEI→IEMOCAP，MELD→IEMOCAP の 3 組だけ | 食い違い（中）．「LA だけで D.T. より大きく上がる」という読みは正しい | "Comparing Amanda without correlation alignment (A. w/o CA) to direct transfer (D.T.), it is clear that label alignment enhances the knowledge transferring capability of the model significantly." | 同上（4.2 節，図 2） |
| 数値は図だけ | 表には無い | 表 1 は D.T. と各手法の値だけで，部品を外した値は図 2 にしか無い | 一致 | 表 1，図 2 | 同上 |

### 1.7 Naini ほか 2025（ICASSP）（S）

| 主張 | 報告の値 | 原文の値 | 判定 | 引用 | URL |
|---|---|---|---|---|---|
| モデル | WavLM-large を微調整してから凍結し，その上を学ぶ | 同じ | 一致 | "After fine-tuning, we freeze the model's parameters" | http://www.lab-msp.com/MSP/publications/Naini_2025.pdf |
| 上がり幅（ラベルで推定） | 覚醒度 約 3.0〜4.3%，感情価 約 3.2〜21% | 覚醒度 VAM 4.3%，RECOLA 3.5%，WHiSER 3.0%．感情価 VAM 21%，RECOLA 3.2%，WHiSER 6.3% | 一致 | "The approximate gains were ∼ 4.3% for VAM, ∼ 3.5% for RECOLA, and ∼ 3.0% for WHiSER."，"an improvement of ∼ 21% for the VAM corpus" | 同上 |
| 疑似ラベル | ラベルで推定した場合より少し小さい | 同じ | 一致 | "However, the gains were slightly smaller than the results for Case 2." | 同上 |

### 1.8 EMO-TTA（arXiv 2509.25495）（S）

| 主張 | 報告の値 | 原文の値 | 判定 | 引用 | URL |
|---|---|---|---|---|---|
| 平均の更新を外す | −1.65 点 | 表 3 の 3 データ（IEMOCAP，CREMA-D，RAVDESS）の平均で 40.05 → 38.40 | 一致（6 データではなく表 3 の 3 データの平均である） | 表 3 "w/out Mean Vectors Update 42.33 43.37 29.51 38.40" | https://arxiv.org/abs/2509.25495 |
| 共分散の更新を外す | −6.41 点 | 40.05 → 33.64．Σ を単位行列に固定した版 | 一致（「更新を外す」は Σ＝I に固定したこと） | "the covariance matrix Σ is fixed as an identity matrix during evaluation" | 同上（4 節） |
| 割合の更新 | 外していない | 表 3 に割合の更新を外した行は無い | 一致 | 表 3 | 同上 |
| 平均を更新すると重みと切片が動く | 推測で，論文には書かれていない | 論文の式 12 に w_y＝Σ⁻¹µ_y，b_y＝log P(y) − ½µ_yᵀΣ⁻¹µ_y がある．両方が µ に依ることは式から読める | 食い違い（軽微）．式は論文にある．「平均の更新は (a)(b) の分け方と一致しない」という結論は論文には無く，報告の推論である | "wy = Σ−1 µy , by = log P (y) − 1/2 µy⊤ Σ−1 µy from the generative model." | 同上 |

### 1.9 顔の研究（F）

| 主張 | 報告の値 | 原文の値 | 判定 | 引用 | URL |
|---|---|---|---|---|---|
| Knyazev 2017 | テストの頻度の平方根で重みづけ，検証 56.40→48.30，テスト 54.98→60.03．テストは Neutral 193，Happy 144，Angry 98（653 本） | 同じ．テストの頻度は提出ごとに返る混同行列から求めた | 一致 | "weighting scores of emotions according to the square root of the observed test set frequencies"，表 4 "Ensemble + A 56.40 54.98 / + class weights 48.30 60.03" | https://arxiv.org/abs/1711.04598 |
| Savchenko 2026（ABAW-10） | GLA で切片を検証集合の F1 で [−2, 2] の格子を座標ごとに探す．顔 38.68/47.59→41.40/51.62，焦点損失つき 38.99/50.22→40.18/51.90，声 30.50/40.08→32.33/42.15 | 同じ．初期値はクラスの割合 | 一致 | "look for biases b∗ … that maximize F1 score … perform the coordinate search in a grid [−2; 2]"，表 1 | https://arxiv.org/abs/2603.12693 |
| Savchenko 2023（ABAW-5） | AU の閾値．MLP 0.508→0.537，LightAutoML 0.397→0.477（表 3） | 同じ（EmotiEffNet の行，表 3） | 一致 | 表 3 "MLP, threshold 0.5 0.508 / MLP, best thresholds 0.537" | https://arxiv.org/abs/2303.09162 |
| ECAN（Li & Deng） | 表 VIII（RAF-DB 2.0→CK+）：重みづけ 85.24，条件つき 83.56，両方 86.49．anger→sadness，fear→surprise をソースの多いクラスへの偏りに帰す | 同じ | 一致 | 表 VIII，"samples with anger and fear labels in CK+ tend to be misclassified into sadness and surprise respectively, which are two dominant categories in the source dataset" | https://arxiv.org/abs/1904.11150 |
| Baltrušaitis 2015 | 中央値を引くと選んだ AU の平均 0.33→0.46，合併で 0.49 | 表 I の Static 平均 0.33，Dynamic 0.46．効果が正の AU だけを載せたと明記．表 II の Combined 0.49 | 一致 | "We are only displaying detection results where using a dynamic model had a significant positive effect" | https://www.cl.cam.ac.uk/~mmam3/pub/FG2015.pdf |
| Albanie 2018 | 均等な抜き出しは有意な効果なし．生徒は平静に偏る．外部では行列と切片の変換をラベルで学ぶ | 同じ | 一致 | "we found that it did not have a significant effect on the quality of the learned student network"，"the distribution of predictions made by the student are dominated by the neutral class"，"fit a single affine transformation (linear transformation plus bias)" | https://arxiv.org/abs/1808.05561 |

### 1.10 言葉の研究（T）

| 主張 | 報告の値 | 原文の値 | 判定 | 引用 | URL |
|---|---|---|---|---|---|
| Cho 2025（NAACL） | 16 件/クラス．GPT2-XL の TEE：BC 26.39/25.85/28.86/29.23，Hidden Calibration 43.45/36.17/39.13/35.48（デモ 0/1/4/8．○） | 付録の表 8 で k=0,1,4,8 の行がこの値である | 一致（○ を ◎ に上げてよい） | 表 8 "k=0 … Bat.C … 26.39 … Hidd.C … 43.45"，"We set m = 16\|Y\| training examples (16 examples per class)" | https://arxiv.org/abs/2406.16535 |
| Ferrer & Ramos 2025 | TMLR．SST2 で 46.0%（0-shot），62.4%（4-shot），IEMOCAP で 3.1% | 表 2 の RCL：SST2 GPT2-0sh 46.0，GPT2-4sh 62.4，IEMOCAP W2V2 3.1．掲載は TMLR 2025（MLAnthology の記録．arXiv は v1 のみ） | 一致 | 表 2，"calibration performance appears to be more sensitive to domain mismatch than discrimination performance" | https://arxiv.org/abs/2408.02841 ，https://mlanthology.org/tmlr/2025/ferrer2025tmlr-evaluating/ |
| Hosseini & Caragea 2023 | GoEmotions→MELD，7 感情，K=4/8/16．ソースのみ 24.51，ターゲットのみ 16.69/18.09/19.96，UDA 21.15，SSL 18.90/21.08/21.75，提案 25.83/27.54/28.85，フル 63.02 | 同じ（表 1 の emotion の列．6 基本感情＋neutral） | 一致 | 表 1 | https://aclanthology.org/2023.findings-acl.333 |
| Jiang 2023 | AUROC は平均 0.95 超，F1 は 0.8 にほとんど届かない．高い AUROC は必要条件で十分条件ではない | 同じ | 一致 | "the average AUROC performance typically exceeds 0.95, while the average F1 performance hardly reaches 0.8."，"AUROC is high is just a necessary condition … but not the sufficient condition" | https://aclanthology.org/2023.findings-emnlp.152 |
| Zhao 2021 | 最良の閾値 0.68 で 94%，CC はオラクルに "surprisingly close"，"simple shift" | 同じ | 一致 | "(p(Positive) = 0.68 in this case), the classifier would be highly accurate (94% on the validation set)." | https://arxiv.org/abs/2102.09690 |
| Estienne 2023 | SUCPA と α＝1 のロジスティック較正が同じ成績，affine が一貫して良い | 同じ | 一致 | "SUCPA and linear logistic calibration with α = 1 give identical performance."，"the affine calibration system performs consistently better than our proposed approaches" | https://arxiv.org/abs/2307.06713 |
| Batch Calibration | BCL で 74.3→75.3，"1% on average" | 同じ | 一致 | "further slight gains, 1% on average" | https://arxiv.org/abs/2309.17249 |
| Li 2023（TARA） | Emotion で CC 0.337/0.352/0.531/0.706，TARA 0.348/0.401/0.697/0.783 | 同じ | 一致 | 表の "Context Calibration 0.337 0.352 0.531 0.706"，"TARA 0.348 0.401 0.697 0.783" | https://arxiv.org/abs/2302.06198 |
| Fei 2023 | 24 データ，感情なし | 24 データ．emotion の語は本文に無い | 一致 | "We conduct experiments on 24 text classification datasets" | https://aclanthology.org/2023.acl-long.783 |
| Inoshita 2026 | 2,000 件の 5 分割，等張回帰が最良 | 同じ | 一致 | "Using 5-fold cross-validation on the 2,000-text core set" | https://arxiv.org/abs/2604.27345 |

### 1.11 声の研究（S，E2）

| 主張 | 報告の値 | 原文の値 | 判定 | 引用 | URL |
|---|---|---|---|---|---|
| Busso ほか 2013 | 平静 727・感情 110 で，正規化なし 18.2%，全体の正規化 55%，理想の正規化は「19 点以上」高い | 18.2%，55% は同じ．理想の正規化は "more than 19% higher" | 一致（「19 点」か「19%」かは原文からは決まらない．軽微） | "(18.2% accuracy). Similarly, the global normalization does not yield satisfactory performance (55% accuracy)." | http://www.lab-msp.com/MSP/publications/Busso_2013_2.pdf |
| Abdelwahab & Busso 2015 | z 値 ±0.3 で 2 値，約 9% と 35% の話者で適応 | 同じ | 一致 | "values are less than zl = 0.3 … higher than zh = 0.3" | http://www.lab-msp.com/MSP/publications/Abdelwahab_2015.pdf |
| Łachut 2025 | macro-F1 0.619→0.753 | 同じ | 一致 | "increasing the Macro F1 score from 0.619 to 0.753" | https://aclanthology.org/2025.poleval-main.12.pdf |
| SER Evals | ロジット調整を評価に入れるが有無の比べは無い | 3.4 節にロジット調整がある．有無の比べの表は見当たらない | 一致 | "We employ logit adjustment[9] to account for varying class …" | https://arxiv.org/abs/2408.07851 |

### 1.12 個別には確かめなかった主張

時間の都合で，次の主張は原文を開き直していない：Olah 2021 の図 4（約 600 件で伸びが止まる．本文の該当行は見たが図は見ていない），Chen 2022，Chochlakis 2024，Demszky 2020，Opitz 2024，Lefter 2022，Agarwal 2026，Turnbull 2026，Ertugrul 2020，OpenFace 2.0，Haber 2016，Dong 2026，Milner，Pepino，Gat，Shah 2015 の数値（Shah は「決定の境界が最適でなくなる」の文だけ確かめた）．これらは判定（完全一致は無い）を左右する研究ではない．

---

## 2. 前回読めなかった文献

### 2.1 Raaijmakers & Kraaij 2010（ICWSM）

- **入手先**：TNO の機関リポジトリの PDF（https://publications.tno.nl/publication/34606958/ThtYlH/raaijmakers-2010-classifier.pdf ）．ojs.aaai.org はボット対策の画面（Anubis）で今回も読めなかった．
- **中身**：Blitzer の Amazon の 4 領域（各 2,000 件，正負 50/50）．SVM の出力に下と上の 2 つの閾値を置き，その間を「分類しない（OOD）」とする．閾値はターゲットの 100 件のラベルで選ぶ（"Effectively only 100 data points were used for calibration."）．閾値を選ぶ基準は正解率，F 値，yield の 3 つ．領域を合わせて学ぶ比べ（表 2：leave-domain-out と use-all）も同じ論文にある．
- **報告（T）の判定との照合**：T は「閾値だけを当て直し，投票で合わせる．分けていない」とした．本文と一致する．ただし閾値は 1 本の切片ではなく，分類しない領域を作る 2 本である．正解率の上がりの多くは難しい例を分類しないことから来る（表 4 の正解率基準では，領域の組ごとに 33〜53% を OOD にする）．較正した分類器は 1,600 件ではなく 1,000 件で学んでいる．このため「切片だけを当て直せば戻る誤り」を測った研究にはあたらない．T の「部分一致」の欄から「関係」に下げるのが原文に合う．

### 2.2 CAST（Wang & Liu, IEEE TAC 16(3):1618–1630, 2025）

- **本文**：読めなかった．Semantic Scholar（openAccessPdf が空），OpenAlex（is_oa: false，リポジトリの全文なし），ResearchGate（403），IEEE Xplore（購読が要る）．Crossref で巻号と頁（16(3)，1618–1630，2025 年 7 月）を確かめた．
- **代わりに読んだもの**：筆頭著者 Shanmin Wang の公開コード（https://github.com/smwanghhh/CAST ．利用者のプロフィールは "Shanmin Wang"．2023-11 作成）の `train.py`．別の利用者による写し（tiantongtong1-oss/CAST．2026-09 作成．論文の著者ではない）もあるが，判定には使っていない．
- **コードでの「予測確率」の扱い**：`annotate_target` は，予測したクラスごとに確信度の平均を求め，それに係数を掛けた値をクラスごとの閾値とし，閾値を超えた例だけを疑似ラベルとして使う．これは疑似ラベルを選ぶ閾値であり，分類器の切片を当て直す処理ではない．学習の損失は，ソースと疑似ラベルの分類の損失，affinity の損失，最終層の重みを直交に近づける損失の和である．
- **判定**：CAST の「クラスごとの表現の分布」と「予測確率」の分析は，コードから見る限り，「切片を当て直せば戻る誤り／戻らない誤り」の分け方にはあたらない．どちらの手も表現と分類器を一緒に学び直す．ただし論文の本文（分析の節）は読めていないので，本文の中に切片だけを当て直す比べがあるかは確かめられない．F と E2 の「部分一致（弱い）．本文未確認」はそのままでよい．

---

## 3. 4 つの報告の判定の妥当さ

- **「3 つの感情の領域とも完全一致は無い」**：原文で確かめた範囲では妥当である．最も近い Haider は情報源が 1 つ（ほかのコーパスで学ぶ）で，2 値の感情極性で，閾値はテストのラベルで選ぶ．Reif & Schwartz は感情の課題を含まず，誤りを分けていない．Tran・Xian・Amanda はずれや部品の得を分けたが，分類器の誤りを切片側と向き側に分けていない．
- **直すべき点**：
  1. Reif & Schwartz を引くとき（T の 1 節・3 節，E2 の 0 節 4・1.1 表・4 節・5 節）：「同じ 16 件」「ラベルの数が同じ k=16」を消し，「LoRA は 16 件で学んだうえで評価でも k 件のデモを使う」と書く．「5〜7 割」はデモの数 k をそろえた比べの値であり，ラベル 16 件どうしの比べ（LoRA の k=0 と LOOC の k=16）では LOOC が同じか上回る，と両方を書く．`DAI_paper/inventory.md` の比（0.91 倍，0.78 倍）と並べる記述（E2 の 5 節）も，どちらの比べ方に対応するかを書き直す必要がある．
  2. Amanda（S の 0 節・2.1 表・3.2 節）：「割合を合わせる項と表現を合わせる項を別々に外した」を「表現を合わせる項だけを外し，D.T.→LA→LA＋CA の 3 段で比べた（図 2 は 7 組のうち 3 組）」に直す．
  3. 軽微な点：Myung の頁（110〜112），Haider の落ちの計算（4 セルの平均と 12 セルの平均の差），EMO-TTA の式 12（w と b の式は論文にある），Reif の凡例（推測ではない），Raaijmakers & Kraaij の判定（部分一致から関係へ）．

---

## 4. 検証の途中で見つけた関係する研究（反例ではない）

| 研究 | 中身 | 反例にあたるか | URL |
|---|---|---|---|
| prashant290605 "Source-validation invariance in cross-corpus speech emotion recognition: an affine adaptation audit"（GitHub．Speech Communication に向けた準備中の原稿．査読なし．2026-04 作成） | RAVDESS↔CREMA-D の声の感情．凍結した自己教師ありの特徴に，そろえ方 6 通り（なし，z 正規化，平均をずらす，CORAL，MK-MMD の 2 種）と分類器 5 種．ずれをラベルの割合・共変量・クラス条件つきに分けて測り，割合のずれは小さい（KL 0.0139〜0.0336）と書く．抄録では，平均をずらすとターゲットの macro-F1 が平均 0.1342 と 0.1969 上がると書く | あたらない．情報源はラベル無しのターゲットだけで，分けたのはずれであり，分類器の誤りではない．Tran・Xian と同じ型である．README と原稿（main.tex，methods.tex）の該当箇所だけを読み，数値の再計算はしていない | https://github.com/prashant290605/Speech-Emotion-Recognition |

この研究の「源の特徴を一定のベクトルでずらすと，RBF-SVM の源の検証の成績は変わらず，ターゲットの判定だけが変わる」という主張は，F の 5 節の「平均を引く処理は線形の分類器では切片の当て直しにあたる」という推論と同じ型の観察である．顔ではなく声の研究なので，F の「顔でこれを書いた論文は無い」という記述とは食い違わない．

---

## 5. 確かめ方の記録

- PDF は `/tmp/claude-1000/-home-snics-MER-DA/76085a37-4610-4922-b8ae-ced34f163c39/scratchpad/dl/` に保存し，pdftotext で文字にして該当箇所を読んだ．Reif & Schwartz の図 3a は 6 頁を 200 dpi で画像にし，目盛り（0.4，0.5，0.6）の位置から値を読み取った（±0.005 程度の誤差）．
- Haider の閾値の選び方は README だけでなく `src/undertow/metrics.py`（oracle_threshold_accuracy，shift_decomposition，evaluate_logits），`analysis.py`（method_summary），`pipeline.py`（評価のコーパスの test のラベルを読む）で確かめた．
- 検索（WebSearch）は 7 回：CAST の本文 2 回，反例の探索 3 回，Tran と Ferrer の掲載先 2 回．API の照会は Semantic Scholar，OpenAlex，Crossref，GitHub，Hugging Face．
