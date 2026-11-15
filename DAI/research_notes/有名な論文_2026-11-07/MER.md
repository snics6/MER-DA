# マルチモーダル感情認識（MER）の有名な論文：原典で確かめた参照ノート（2026-11-07）

このノートは，MER の分野で広く引かれる論文 38 本を原典で確かめ，後のセッションが引用と設計の判断に使えるように要点をまとめたものである．
対象はデータセット，総説，融合の手法，会話の感情認識，モダリティの釣り合い，欠けたモダリティ，自己教師ありの音声エンコーダ，感情の表し方，コーパス間の感情認識である．
半教師ありの DA，蒸留，メタ学習はほかの担当のノートにあり，ここでは重なる論文（Albanie 2018，UMT）を短く書くだけにした．

---

## 0. 調べ方と確認の水準の記号

### 調べ方

- arXiv の抄録の頁，ACL Anthology の頁と BibTeX，Crossref の API，PMLR と NeurIPS の論文集の頁で，題・著者・会場・巻号・頁・年を確かめた．
- 本文は PDF を scratchpad に落として `pdftotext` で文字にし，抄録と手法の節と結果の表を読んだ．PDF が無い論文は，著者や機関の公開版（受理原稿），PubMed Central の本文，後継の論文で代えた．代えた場合は各論文の項に書いた．
- 数値は原典で見たものだけを書き，表や節の場所を添えた．PDF から取り出した文字を読んだので，表の列の対応は文字の並びから読み取っている．論文に引用する前に PDF の表で確かめること．
- 研究の `results/`，`logs/`，`data/cache/` は開いていない．
- `docs/related-work.md` と `DAI_paper/参照論文/要点_問い1・問い3` にすでに精読の記録がある論文（Schuller 2010，Albanie 2018 など）は，書誌を確かめ直して要点を数行に縮め，元のノートを指した．

### 確認の水準

| 記号 | 意味 |
|---|---|
| ◎ | 本文を読んだ（括弧の中に読んだ範囲を書く）．「道具経由」は，ウェブの頁を取得の道具に要約させて読んだもの |
| ○ | 抄録か書誌の頁で確かめた |
| △ | 書誌のみ，または二次資料（ほかの論文の記述）だけで確かめた |

### このノートで使う語（初出の定義をここにまとめる）

- **UA / UAR**（unweighted accuracy / unweighted average recall）：感情ごとの再現率（その感情の発話のうち正しく当てた割合）を，感情の数で単純に平均したもの．でたらめに当てると「1 ÷ 感情の数」になる．MER-DA の主指標である．
- **WA**（weighted accuracy）：音声の感情認識の慣習では，全発話に対する正解率（多い感情ほど重く数える）を指す．ただし MOSEI の系の論文は別の定義の WA を使う（3.2 節）．
- **F1**：適合率と再現率の調和平均．**重み付き F1**（weighted F1）は，感情ごとの F1 を，その感情の発話の数で重み付けして平均したもの．
- **Fleiss の κ**：3 人以上の評価者のラベルの一致度から，偶然に一致する分を差し引いた値．1 が完全な一致，0 が偶然と同じ．
- **話者独立**（speaker-independent）：同じ話者の発話が学習側と評価側の両方に入らない分け方．
- **早期融合**：モダリティの特徴をつないでから 1 つのモデルで予測する．**後期融合**：モダリティごとに予測を出してから合わせる．
- **自己教師あり学習**：ラベルの無いデータだけを使い，入力の一部を隠して当てるなどの課題で表現を学ぶ方法．
- **凍結**：事前に学んだエンコーダの重みを変えずに，その出力（特徴）だけを使うこと．
- **MSA**（multimodal sentiment analysis）：動画の話者の意見の肯定・否定の度合い（感情価に近い）を当てる課題．**ERC**（emotion recognition in conversation）：会話の中の各発話の感情を当てる課題．

---

## 1. 分野の流れ

**2008〜2014 年：演技と誘発で収録したコーパスの時代．**IEMOCAP（Busso 2008）は，俳優 10 人が台本と即興で対話する場面を，音声・顔・手の動きと一緒に収録し，感情のカテゴリと次元（快・不快，興奮の度合い，支配の度合い）を付けた．即興の発話のほうが評価者の一致率が高かった（83.1% と 66.9%）．同じころ Schuller ほか（2010）は，音声の感情認識をコーパスをまたいで評価すると成績が大きく下がり，演技のコーパスから自然な発話のコーパスへはほとんど移らないことを示した．AFEW（Dhall 2012，EmotiW 2013）は映画の場面を字幕で探して集め，実験室の外に近い条件の顔と声のデータを作った．CREMA-D（Cao 2014）は 91 人の俳優の発話を，音声だけ・顔だけ・両方の 3 通りで人に判定させ，人の正解率がモダリティで違う（40.9%，58.2%，63.6%）ことを数で示した．

**2016〜2018 年：YouTube の意見動画と融合の手法の時代．**CMU の Morency らは，YouTube の意見動画に感情価の強さ（−3〜+3）を付けた CMU-MOSI（Zadeh 2016）を作り，言葉・顔・声の 3 つをどう組み合わせるかを研究の中心に置いた．TFN（Zadeh 2017）は 3 つの表現の外積で 1 次・2 次・3 次の組み合わせを全部作り，LMF（Liu 2018）はその重みを低い階数に分解して計算を軽くした．MFN（Zadeh 2018）は時間方向の記憶で各時刻のモダリティ間の関係を追い，CMU-MOSEI（Zadeh 2018）は話者約 1,000 人・23,453 文に広げて 6 つの感情の強さも付けた．この系の LMF と MulT は IEMOCAP の 4 感情をそれぞれ 2 値（ある／ない）で評価しており，4 クラスの単一ラベルを UA で評価する音声の感情認識の系とは評価の仕方が分かれた（3.2 節）．

**2017〜2019 年：分類の整理と，時間のずれと会話の文脈．**Poria ほか（2017）の総説は融合を特徴の段・判断の段・混合・モデルの段の 4 つに分け，Baltrušaitis ほか（2019）の総説はマルチモーダル機械学習の課題を表現・変換・対応づけ・融合・共同学習の 5 つに整理した．MulT（Tsai 2019）は，モダリティ間の注意で，単語の時刻にそろえていない系列どうしを直接結んだ．MELD（Poria 2019）はテレビドラマ Friends の多人数の会話に 7 感情を付け，DialogueRNN（Majumder 2019）と DialogueGCN（Ghosal 2019）は，話者ごとの状態と前後の発話の文脈を使うと会話の感情認識が良くなることを示した．ただしこの系の手法の多くは言葉だけで評価され，DialogueRNN では 3 モダリティを足しても言葉だけより F1 が 0.2 点しか上がらなかった．

**2020〜2022 年：事前学習した言語モデルを中心に置く融合と，表現の分け方．**MAG-BERT（Rahman 2020）は BERT の内部の単語の表現を，声と顔から作ったずらしの量で動かした．MISA（Hazarika 2020）は各モダリティの表現を「モダリティに共通の部分」と「固有の部分」に分けて学ばせた．CH-SIMS（Yu 2020）は 1 つの動画に，モダリティごとのラベルと全体のラベルを別々に付け，両者が食い違うことを示した．Self-MM（Yu 2021）はモダリティごとのラベルを自動で作り，UniMSE（Hu 2022）は MSA と ERC のラベルを 1 つの形式にそろえて 4 つのデータを一緒に学習した．MMGCN（Hu 2021）は会話の各発話のモダリティごとの節点をつないだグラフで融合し，MMIN（Zhao 2021）は一部のモダリティが欠けた場合に欠けた側の表現を予測した．

**2020〜2023 年：融合の学習がうまくいかない理由の研究．**Wang ほか（2020）は，動画の分類で，複数モダリティを一緒に学んだモデルが最良の単モダリティのモデルに負けることを示し，その原因を，融合の網は容量が大きく過学習しやすいことと，過学習と汎化の速さがモダリティで違うことに求めた．OGM-GE（Peng 2022）は強いモダリティの勾配を弱めて釣り合いを取り，Huang ほか（2022）は，後期融合の網を勾配降下で学ぶとモダリティが競合し，一部のモダリティしか学ばれないことを理論で示した．UMT/UME（Du 2023）は，単モダリティの特徴が学び足りないことが汎化を悪くすると論じ，連結の線形分類器と単モダリティの予測の平均のどちらを使うかを決める手順を出した．この系の研究の多くはエンコーダを一緒に学ぶ設定を扱う．

**2020〜2022 年：自己教師ありの音声エンコーダと，凍結した特徴での評価．**wav2vec 2.0（Baevski 2020），HuBERT（Hsu 2021），WavLM（Chen 2022）は，ラベルの無い大量の音声から表現を学ぶ．SUPERB（Yang 2021）はエンコーダを凍結し，各層の重み付き和の上に軽い予測器を載せて比べる評価の型を決め，感情認識（IEMOCAP の 4 クラス）もその 1 つに入れた．凍結した特徴の上に浅い分類器を学ぶ評価は，のちの EmoBox や MERBench（`docs/related-work.md` 2.3 節）にも引き継がれた．自然な発話の大きなコーパスとしては MSP-Podcast（Lotfian & Busso 2019，最終版の記述は Busso ほか 2025）が作られた．

**2022 年以降：総説の整理と大規模言語モデル．**Liang, Zadeh, Morency（2024）は課題を表現・対応づけ・推論・生成・転移・定量化の 6 つに整理し直し，各モダリティの役立ち方と危うさを数で表す方法を未解決に挙げた．その後の MER は，大規模言語モデルによる自由記述の感情理解へ進んでいる（AffectGPT，MER2024，MER と LLM の総説．`docs/related-work.md` 2.4 節）．

---

## 2. 論文ごとの要点

各論文に 6 項目を書く．「MER-DA との接点」はこちらの推測であり，MER-DA の中で検定していない．

### 2.1 データセット

### Busso ほか 2008（Language Resources and Evaluation 42(4):335–359．arXiv なし）IEMOCAP: Interactive emotional dyadic motion capture database

- **問い**：感情は言葉と身ぶりの両方で表れるので，音声と顔と手の動きを一緒に調べられる感情の対話のコーパスを作る．
- **方法（収録とラベル）**：
  - 俳優 10 人（女性 5 人，男性 5 人）が 2 人 1 組で 5 セッションを収録した．台本の劇を演じる部分と，場面の設定だけを与えて即興で話す部分がある．1 人の顔に 53 個，手首・頭・手を含め計 61 個のマーカーを付け，モーションキャプチャで記録した．全体で約 12 時間である（3・4 節）．
  - 発話（turn）は 10,039 個（台本 5,255，即興 4,784），平均 4.5 秒である（5.1 節）．
  - カテゴリは anger，sadness，happiness，disgust，fear，surprise，frustration，excited，neutral（と other）．6 人の評価者のうち 3 人が各発話を，動画と前の発話を見ながら評価した．複数のラベルを選んでよい．最多票のラベルが 1 つに決まる場合にそれを代表ラベルにした（5.2 節）．
  - 次元は valence（快・不快），activation（興奮の度合い），dominance（支配の度合い）を，絵の尺度（SAM）で 1〜5 の整数で 2 人が付けた．
- **分かったこと**：
  - 代表ラベルが決まった発話は 74.6%（台本 66.9%，即興 83.1%）である（5.2 節）．
  - Fleiss の κ は全発話で 0.27，代表ラベルが決まった発話で 0.40 である．happiness と excited をまとめ，fear・disgust・surprise を other にすると，0.35 と 0.48 に上がる（5.2 節，表 V）．
  - 代表ラベルを正解としたときの人の判定の平均の正解率は 72% である．neutral・anger・disgust は frustration と，happiness は excitement と混同された（表 IV）．
- **前提と限界（著者）**：俳優による収録である．著者は，即興の場面は本物に近い感情を引き出すための工夫だと位置づけている．カテゴリの数を増やすと一致率が下がり，減らすと記述が粗くなるという兼ね合いを書いている．
- **MER-DA との接点**（推測．未検定）：
  - 即興と台本で一致率が 16 点違うので，MER-DA が IEMOCAP を即興と台本の 2 群に分けることの根拠の 1 つになる．
  - MER-DA は hap と exc を Happy にまとめる（`data/README.md` 3 節）．著者が κ を計算し直すときも同じ 2 つをまとめている．
  - κ = 0.27 は，テストの正解ラベルそのものが揺れていることを示す．UAR の上限の議論に使える．
- **確認**：◎（著者の研究室が公開する 2008 年 9 月の原稿版．抄録・1〜5 節）．[DOI](https://doi.org/10.1007/s10579-008-9076-6)，[PDF](https://sail.usc.edu/publications/files/BussoLRE2008.pdf)

### Zadeh, Zellers, Pincus, Morency 2016（IEEE Intelligent Systems 31(6):82–88．arXiv 1606.06259）MOSI: Multimodal Corpus of Sentiment Intensity and Subjectivity Analysis in Online Opinion Videos

雑誌版の題は「Multimodal Sentiment Intensity Analysis in Videos: Facial Gestures and Verbal Messages」で，arXiv 版と違う．

- **問い**：YouTube の意見動画で，意見の区切りごとに感情価の強さを付けたデータを作り，言葉と顔の動きの関係を調べる．
- **方法**：
  - 93 本の動画，話者 89 人（女性 41，男性 48）．3,702 区間のうち主観的な意見の区間が 2,199 個，客観的な区間が 1,503 個．意見の区間は平均 4.2 秒，平均 12 語（表 1）．
  - 感情価の強さは −3（strongly negative）〜 +3（strongly positive）の 7 段階で，Amazon Mechanical Turk の作業者 5 人が付け，平均を最終の値にした（3.3 節）．
- **分かったこと**：
  - 作業者間の一致は Krippendorff の α（一致度の指標．1 が完全な一致）で 0.77（感情価），主観・客観の区切りは 0.68 である．
  - 人の予測は平均絶対誤差 0.61，相関 0.83．言葉と顔の動きを組にした辞書のモデルは 1.10 と 0.53 である（表 2）．
- **前提と限界（著者）**：顔の動きは笑顔・しかめ面・うなずき・首振りの 4 つの 2 値の特徴だけで調べた．
- **MER-DA との接点**（推測．未検定）：MER-DA は MOSI を使わない．ただし TFN・MISA・Self-MM などの融合の手法の多くはこのデータで比べられており，その結果は感情のカテゴリではなく感情価の回帰についてのものである．TFN は同じデータの話者を「93 distinct speakers」と書いており，原典の 89 人と食い違う．
- **確認**：◎（arXiv 版の本文）．[arXiv](https://arxiv.org/abs/1606.06259)，[DOI](https://doi.org/10.1109/MIS.2016.94)

### Zadeh, Liang, Poria, Cambria, Morency 2018（ACL 2018, pp. 2236–2246．arXiv なし）Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph

- **問い**：話者と話題の幅が狭いデータでは汎化を調べられないので，大きな 3 モダリティのデータを作る．あわせて，どのモダリティの組み合わせが使われているかを読み取れる融合の手法を作る．
- **方法（データ）**：
  - YouTube の独白の動画 3,228 本から 23,453 文．話者は約 1,000 人，話題は 250 種，男女比は 57% 対 43%，計 65 時間 53 分（1・3.1 節，表 1・2）．
  - 話者を人手で見分けるのは費用が大きいので，チャンネルを話者の代わりにし，1 チャンネルから最大 10 本までとした（3.1 節）．
  - 感情価を −3〜+3，Ekman の 6 感情（happiness，sadness，anger，fear，disgust，surprise）の有無を 0〜3（0: no evidence，3: highly）で，Mechanical Turk の作業者 3 人が付けた（3.2 節）．
  - 最も多い感情は happiness（12,000 件超），最も少ないのは fear（約 1,900 件）である（図 2）．
- **方法（DFG）**：言葉 l，映像 v，音声 a の 1 つずつ・2 つずつ・3 つ全部の組み合わせを頂点にし，含まれる関係にある頂点を辺でつなぐ．各辺に重み（論文の語で efficacy）を付ける．

  ```
  頂点 {l}, {v}, {a}, {l,v}, {l,a}, {v,a}, {l,v,a}
  vi ⊂ vj のときだけ辺 eij を張り，vi に αij を掛けてから Dj に渡す
  αij = sigmoid(…)   ∈ (0, 1)   ← 入力ごとに変わる重み
  ```

  MFN の融合の部品を DFG に替えたものを Graph-MFN と呼ぶ．
- **分かったこと**：MOSEI の感情価の 2 値の正解率は Graph-MFN で 76.9，F1 は 77.0．感情は 6 つの感情ごとに WA と F1 を報告し，happy の WA は 66.3 である（表 3）．0〜3 の強さを評価のためにどう 2 値にしたかは，本文では確かめられなかった．
- **前提と限界（著者）**：感情の評価では，偏りがあるので Tong ほか 2017 の weighted accuracy を使うと書いている．式は本文に無い．
- **MER-DA との接点**（推測．未検定）：
  - 原典が話者をチャンネルで代えているので，MER-DA の MOSEI の話者分離（チャンネルが分かるのは 35%．`data/README.md` 3 節）の不完全さは原典の時点から続く問題である．
  - MER-DA の MOSEI の Neutral は「どの感情の強さも 0.5 未満」という MER-DA 側の規則で作ったもので，原典の注釈者は Neutral を付けていない．原典の評価は 6 感情の有無の 2 値で，4 クラスの単一ラベルではない．
- **確認**：◎（本文．表 3 まで）．[ACL Anthology](https://aclanthology.org/P18-1208/)

### Poria, Hazarika, Majumder, Naik, Cambria, Mihalcea 2019（ACL 2019, pp. 527–536．arXiv 1810.02508）MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations

- **問い**：文字だけの会話の感情データ EmotionLines に音声と映像を足し，多人数の会話で文脈とモダリティを使う感情認識を調べられるようにする．
- **方法**：
  - Friends の会話 1,433 個（学習 1,039，開発 114，評価 280），発話 13,708 個（9,989，1,109，2,610）．感情は Ekman の 6 つと neutral の 7 つ（表 8）．
  - 元の EmotionLines は文字だけを見て付けたラベルなので，3 人が動画も見て付け直し，多数決にした．3 人とも違うラベルを付けた 89 発話（11 会話）は会話ごと落とした（3・3.1 節）．
- **分かったこと**：
  - Fleiss の κ は 0.43 で，EmotionLines の 0.34 より高い．著者は，動画を見せたことが一致を上げたと解釈している（3 節）．
  - 発話の約 42% が 5 語未満である（1 節）．
  - 基準のモデル（DialogueRNN）の感情の重み付き F1 は，言葉 57.03，音声 41.79，言葉＋音声 60.25 である（表 11）．disgust・fear・sadness の成績が特に低く，著者は学習の本数の少なさを原因に挙げている．
- **前提と限界（著者）**：基準のモデルは映像を使っていない．原文："Identifying face of the speaker in a video where multiple other persons are present is very challenging."（5 節）．分割は会話の単位で，話者で分けたとは書いていない．
- **MER-DA との接点**（推測．未検定）：
  - 原典の著者自身が，画面に複数の人がいると話者の顔を特定しにくいと書いている．MER-DA は面積が最大の顔を選ぶ（`data/README.md` 4 節）ので，MELD の顔の特徴には話者でない人の顔が混ざりうる．
  - 原典の分割では Friends の主要な人物が学習と評価の両方に現れうる．MER-DA が話者名で分けているのは原典の分割より厳しい条件である．
- **確認**：◎（arXiv 版の本文）．[ACL Anthology](https://aclanthology.org/P19-1050/)，[arXiv](https://arxiv.org/abs/1810.02508)

### Dhall, Goecke, Lucey, Gedeon 2012（IEEE MultiMedia 19(3):34–41）Collecting Large, Richly Annotated Facial-Expression Databases from Movies；Dhall, Goecke, Joshi, Wagner, Gedeon 2013（ICMI 2013, pp. 509–516）Emotion Recognition In The Wild Challenge 2013（AFEW と EmotiW）

- **問い**：実験室で撮った表情のデータは実際の条件を表さないので，映画から実際に近い条件の表情と声のデータを集める．
- **方法（EmotiW 2013 の記述）**：
  - 映画の字幕（聴覚障害者向けの字幕を含む）から，感情を表す語を正規表現で探し，その字幕が出ている時間の場面を切り出す．人が見て，話者の情報と表情のラベルを付ける（データベースの作り方の節，図 1）．
  - AFEW 3.0 は 1,832 本の場面，長さ 300〜5,400 ミリ秒，注釈者 2 人，ラベルは Angry・Disgust・Fear・Happy・Neutral・Sad・Surprise の 7 つ（表 2）．映画は 75 本．
  - EmotiW 2013 はそこから 1,088 本を使い，Train 380，Val 396，Test 312 本に分けた．原文："The data is subject independent and the sets contains clips from different movies."（データの分割の節）．各集合の人数は Train 99 人，Val 126 人，Test 90 人（表 3）．
- **分かったこと**：基準のモデルの 7 クラスの正解率は，音声 Val 19.95・Test 22.44，映像 Val 27.27・Test 22.75，音声と映像の特徴の連結 Val 22.22・Test 27.56 である（表 6）．
- **前提と限界（著者）**：映画は演出された演技なので，データは部分的に自然（Spontaneous (Partial)）と位置づけている（表 1）．顔の検出は照明・隠れ・横顔で誤りが多いと書いている．
- **後の版**：EmotiW 2017 の AFEW は Train 773，Val 383，Test 653 本と，参加者の論文が書いている（△．例：[arXiv 1709.07200](https://arxiv.org/abs/1709.07200)）．MER-DA の発話一覧の 1,156 行は 773 と 383 の和に一致する（推測．`features/build_manifest_afew.py` の元データの版は確かめていない）．
- **MER-DA との接点**（推測．未検定）：
  - 場面を字幕の感情の語で探して集めているので，AFEW のラベルは言葉の手がかりと結びつきやすい作り方をしている．AFEW の言葉を Whisper の書き起こしで作った MER-DA の言葉の特徴が AFEW でどれだけ当たるかに，この作り方が関わりうる．
  - 原典の分割は人物と映画で分けているので，MER-DA の話者分割（人物名，一部は作品名）と考え方が同じである．
- **確認**：EmotiW 2013 は ◎（本文 6 頁）．Dhall 2012 は ○（書誌と機関の頁の抄録．本文は 2037 年まで公開されない）．[Dhall 2012 DOI](https://doi.org/10.1109/MMUL.2012.26)，[ANU の頁](https://openresearch-repository.anu.edu.au/items/a63944bd-dfad-4020-85db-039c5d72a3e8)，[EmotiW 2013 DOI](https://doi.org/10.1145/2522848.2531739)，[EmotiW 2013 PDF](https://openresearch-repository.anu.edu.au/bitstreams/2af070d2-4c1d-4882-9557-531182584179/download)

### Cao, Cooper, Keutmann, Gur, Nenkova, Verma 2014（IEEE Transactions on Affective Computing 5(4):377–390）CREMA-D: Crowd-Sourced Emotional Multimodal Actors Dataset

- **問い**：多様な俳優の感情の演技を，声だけ・顔だけ・両方の 3 通りで多くの人に判定させ，モダリティごとの伝わり方を調べられるデータを作る．
- **方法**：
  - 俳優 91 人（男性 48，女性 43，20〜74 歳）が，感情の中立な 12 文を，happy・sad・anger・fear・disgust・neutral の 6 つで演じた．1 文目だけは low・medium・high の 3 段階の強さで演じた（2.1 節）．
  - 7,442 本．予定の 7,462 本のうち 20 本は取り出せなかった．
  - 2,443 人の評価者（クラウドソーシング）が，音声だけ・映像だけ・音声と映像の 3 条件でカテゴリと強さを付けた．目標は 1 本あたり 10 件で，95% 超の本数に 8 件以上の評価がある（2.2・3.1 節，表 6）．
- **分かったこと**：
  - 人が意図した感情を当てた割合は，音声だけ 40.9%，映像だけ 58.2%，音声と映像 63.6% である（抄録，5.1 節）．
  - 当てやすさは neutral，happy，anger，disgust，fear，sad の順である（5.1 節）．著者は，disgust と fear は両方のモダリティがそろうとよく当たり，anger と happy は 1 つのモダリティでも当たると結論している（抄録）．
- **前提と限界（著者）**：俳優の演技であり，演出家が場面を与えて感情を引き出した．
- **MER-DA との接点**（推測．未検定）：人の判定でもモダリティごとの伝わり方が感情によって違う．MER-DA で別モダリティの役立ち方が感情の組ごとに違うという観察と同じ向きの，人の側の証拠として引ける．12 文しかないので，MER-DA は言葉の特徴を作らない（`data/README.md` 2 節）．
- **確認**：◎道具経由（PubMed Central の本文 HTML を取得の道具で読み，数値を節と表の番号つきで取り出した）．[DOI](https://doi.org/10.1109/TAFFC.2014.2336244)，[PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4313618/)

### Livingstone, Russo 2018（PLoS ONE 13(5):e0196391）The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)

- **問い**：感情の研究に使える，妥当性と信頼性を人の評価で確かめた，発話と歌の感情の映像と音声のデータを作る．
- **方法**：
  - 北米英語の俳優 24 人（男女同数）．2 つの中立な文（"Kids are talking by the door"，"Dogs are sitting by the door"）を，発話では neutral・calm・happy・sad・angry・fearful・surprise・disgust の 8 つで，歌では 6 つで表現した．neutral 以外は normal と strong の 2 段階の強さがある（Methods の節）．
  - 映像と音声・映像だけ・音声だけの 3 形式で，計 7,356 ファイル（発話 4,320，歌 3,036）．
  - 247 人が各ファイルを 10 回ずつ評価し，別の 72 人が再評価の信頼性を測った．
- **分かったこと**：
  - 発話の正答の割合（表 2）：映像と音声は normal 0.77・strong 0.83，映像だけは 0.70・0.75，音声だけは 0.58・0.67．Fleiss の κ は映像と音声の normal で 0.62，音声だけの normal で 0.41．
  - 発話全体の正答の割合の平均は 0.72 である．
- **前提と限界（著者）**：calm と neutral は知覚上ほとんど区別されないので，どちらの答えも互いに正解として扱った（Methods の評価の手順）．
- **MER-DA との接点**（推測．未検定）：MER-DA は calm を Neutral に混ぜない（`data/README.md` 3 節）．原典の評価が両者を区別していないことは，calm を除く判断の根拠になる．文が 2 つしかないので言葉の特徴は作れない．
- **確認**：◎（序論・Methods・表 2）．[DOI](https://doi.org/10.1371/journal.pone.0196391)

### Lotfian, Busso 2019（IEEE Transactions on Affective Computing 10(4):471–483）Building Naturalistic Emotionally Balanced Speech Corpus by Retrieving Emotional Speech from Existing Podcast Recordings；Busso ほか 2025（arXiv 2509.09791）The MSP-Podcast Corpus

- **問い**：演技でない自然な会話の音声で，感情の偏りが少なく，話者の多い大きなコーパスを作る．
- **方法（2025 年の最終版の記述）**：
  - 再利用を許す使用許諾のポッドキャストを話者交代ごとに切り，2.75〜11 秒の発話を選ぶ．機械学習のモデルで感情のこもっていそうな発話を探してから人に注釈させ，感情の偏りを減らす（III 節）．
  - 1 発話に 5 人以上が，主な感情 1 つ（8 種：happiness，anger，sadness，neutral，surprise，fear，disgust，contempt），副次的な感情（16 種），感情価・覚醒度・支配性を付ける．
  - 最終版 2.0 は 409 時間，話者 3,641 人（女性 2,043，男性 1,598）．
- **分かったこと**：
  - neutral の割合は 28% で，探して選ぶ手順が偏りを減らしたと著者は書く（IV 節）．
  - 主な感情の Fleiss の κ は全体で 0.411 である（IV 節，表 III）．
  - 分割：Test 1 は約 17.2%（465 人），Test 2 は探す手順を使わずに集めた発話（neutral が約 45.8%，既知の話者 112 人），Test 3 は 3,200 発話（428 人）でラベルを公開しない．開発用は 704 人，学習用は残りの 2,220 人と話者が不明の発話（V-A 節）．
- **前提と限界（著者）**：原文："The partitions aim to be speaker-independent, although some unknown speakers in the training set may overlap with those in the development or test partitions."（V-A 節）．
- **MER-DA との接点**（推測．未検定）：話者が分からない発話が残るために話者分離が完全でない点は，MER-DA の MOSEI と同じ種類の問題である．MER-DA は使っていないが，自然な会話の音声のソースの候補として最もよく引かれるコーパスである（Chen & Yu 2023 では最も役立つソースだった．`docs/related-work.md` 2.2 節）．
- **確認**：2019 年版は ○（Crossref の書誌．本文は取得できなかった）．2025 年版は ◎（抄録・I〜V 節の要所）．[2019 DOI](https://doi.org/10.1109/TAFFC.2017.2736999)，[2025 arXiv](https://arxiv.org/abs/2509.09791)

### Yu, Xu, Meng, Zhu, Ma, Wu, Zou, Yang 2020（ACL 2020, pp. 3718–3727．arXiv なし）CH-SIMS: A Chinese Multimodal Sentiment Analysis Dataset with Fine-grained Annotation of Modality

- **問い**：これまでの MSA のデータは動画全体に 1 つのラベルしか持たないが，そのラベルは各モダリティ単独の感情を表すとは限らない．モダリティごとのラベルを持つデータを作る．
- **方法**：
  - 中国語の映画・ドラマ・バラエティ番組の 60 本から 2,281 区間．話者 474 人，平均 3.67 秒，平均 15 語．画面に話者以外の顔が映らない区間だけを選んだ（3.1 節，表 1）．
  - 1 区間に，言葉だけ・音声だけ・映像だけの 3 つのラベルと，全体のラベルを付けた．注釈者は 5 人で，各人が −1・0・1 を付け，平均を取る（3.2 節）．混ざらないように，単モダリティのラベルを先に付けた．
- **分かったこと**：モダリティのラベルと全体のラベルの食い違いを混同行列で示した（図 2）．全体のラベルで学ぶより，モダリティのラベルも使う多課題学習のほうが良かった（表 2）．
- **前提と限界（著者）**：原文："the unified annotations do not always reflect the independent sentiment of single modalities"（抄録）．
- **MER-DA との接点**（推測．未検定）：MER-DA の各単モダリティの分類器は，全体を見て付けたラベルで学んでいる．CH-SIMS の結果は，単モダリティから見たラベルの付け方のずれがコーパスの中にもあることを示す．MER-DA の「ラベルの付け方のずれ」の一部はモダリティごとに違う可能性がある．
- **確認**：◎（本文の 1〜4 節）．[ACL Anthology](https://aclanthology.org/2020.acl-main.343/)

### 2.2 総説

### Baltrušaitis, Ahuja, Morency 2019（IEEE TPAMI 41(2):423–443．arXiv 1705.09406）Multimodal Machine Learning: A Survey and Taxonomy

- **問い**：応用ごとに散らばったマルチモーダル機械学習の研究を，共通の分類で整理する．
- **方法（分類）**：早期・後期の融合という分け方を超えて，5 つの課題に分ける．表現（representation），変換（translation），対応づけ（alignment），融合（fusion），共同学習（co-learning：あるモダリティの知識で別のモダリティの学習を助ける）．
  融合はさらにモデルを問わない型（早期・後期・混合）と，モデルに組み込む型（カーネル法・グラフィカルモデル・ニューラルネット）に分ける（6 節）．

  ```
  早期融合：  ŷ = f([x_a ; x_v ; x_t])            （特徴をつないで 1 つの予測器）
  後期融合：  ŷ = g(f_a(x_a), f_v(x_v), f_t(x_t))  （モダリティごとの予測を平均・投票・学んだ g で合わせる）
  ```
- **分かったこと**：後期融合は，モダリティごとに別のモデルを使え，一部のモダリティが欠けても予測でき，対になっていないデータでも学べる．その代わり低い水準のモダリティ間の関係を無視する（6.1 節）．
- **前提と限界（著者）**：融合に残る課題として 3 つを挙げる（6 節末）．原文："2) it is difficult to build models that exploit supplementary and not only complementary information; 3) each modality might exhibit different types and different levels of noise at different points in time."
- **MER-DA との接点**（推測．未検定）：MER-DA の融合は早期融合で，予測器は線形の softmax である．線形の予測器では，連結した特徴の点数はモダリティごとの点数の和になる．

  ```
  W [x_a ; x_v ; x_t] + b = W_a x_a + W_v x_v + W_t x_t + b
  ```

  このため，MER-DA の連結の分類器は，3 つの単モダリティの線形の点数を足す形（確率では掛け合わせて正規化する形）と同じ関数の族であり，モダリティ間の掛け算の項（TFN の 2 次・3 次の項）を持たない．これは式の上の事実であり，MER-DA の結果との関係は調べていない．
- **確認**：◎（抄録・1 節・6 節・7 節冒頭）．[DOI](https://doi.org/10.1109/TPAMI.2018.2798607)，[arXiv](https://arxiv.org/abs/1705.09406)

### Poria, Cambria, Bajpai, Hussain 2017（Information Fusion 37:98–125）A review of affective computing: From unimodal analysis to multimodal fusion

- **問い**：感情計算（感情を機械で認識・解釈する研究）を，単モダリティの解析から融合まで 1 本で概観する．
- **方法**：音声・映像・言葉の 3 つに絞る（関連文献の約 90% がこの 3 つを扱うため）．融合を特徴の段の融合，判断の段の融合，混合，モデルの段の融合に分け，判断の段の融合をさらに規則型・分類器型・推定型に分ける（5 節）．約 100 本を調べた．
- **分かったこと（著者のまとめ）**：調べた研究の 90% 超が映像を音声より強いと報告していた．2010 年以降は言葉を使う研究が増え，言葉を足すと成績が上がるという報告が多い（7 節）．
- **前提と限界（著者）**：原文："the amount of trust that should be placed in acted corpora is debatable"，"in the case of emotions, the number of emotional dimensions to use is unclear"（7 節）．今後の課題として，自然な表出の動画，複数のデータでの検証と汎化，会話の中の感情（話者どうしの影響）を挙げる．
- **MER-DA との接点**（推測．未検定）：演技のコーパスを信頼してよいかという問いは，MER-DA の「演技の録音か自然な会話か」の軸（`docs/related-work.md` 2.3 節の Milner の項）の古い出典になる．「複数のデータでの検証」を課題に挙げた総説として引ける．
- **確認**：◎（スターリング大学の公開する受理原稿．抄録・5 節の分類・7 節）．[DOI](https://doi.org/10.1016/j.inffus.2017.02.003)，[受理原稿](https://dspace.stir.ac.uk/handle/1893/25490)

### Liang, Zadeh, Morency 2024（ACM Computing Surveys 56(10):1–42．arXiv 2209.03430）Foundations & Trends in Multimodal Machine Learning: Principles, Challenges, and Open Questions

- **問い**：Baltrušaitis 2019 以降の研究を含めて，マルチモーダル機械学習の原理と課題を整理し直す．
- **方法（分類）**：原理として，モダリティが異なること（heterogeneity），つながり（connections），相互作用（interactions）の 3 つを置く．課題を表現・対応づけ・推論・生成・転移（transference）・定量化（quantification）の 6 つに分ける．相互作用の情報は，共通の情報（redundancy），1 つのモダリティだけにある情報（uniqueness），組み合わせて初めて出る情報（synergy）に分けて論じる（2・8 節）．
- **分かったこと（著者のまとめ）**：学習の過程について，複数モダリティの網は容量が大きいので過学習しやすく，モダリティによって過学習と汎化の速さが違うと Wang 2020 を引いて書く（8.3 節）．
- **前提と限界（著者）**：原文："How can we formally quantify the utility and risks of each input modality, while balancing these tradeoffs for reliable real-world usage?"（8.3 節）．
- **MER-DA との接点**（推測．未検定）：「各入力モダリティの役立ち方と危うさを数で表す」は，MER-DA の問い（どのモダリティをソースから持ち込むと得か）と同じ形の未解決の問いとして引ける．共通・固有・組み合わせの 3 分類は，Xue 2023 の共通の手がかりの割合（`要点_問い1`）と同じ語で書ける．
- **確認**：◎（抄録・1〜2 節・8 節）．[DOI](https://doi.org/10.1145/3656580)，[arXiv](https://arxiv.org/abs/2209.03430)

### 2.3 融合の手法

### Zadeh, Chen, Poria, Cambria, Morency 2017（EMNLP 2017, pp. 1103–1114．arXiv 1707.07250）Tensor Fusion Network for Multimodal Sentiment Analysis（TFN）

- **問い**：言葉・顔・声の間の 1 つずつ・2 つずつ・3 つの関係を，学習の中ではっきり表す．
- **方法**：各モダリティの埋め込みに 1 を付け足し，3 つの外積を取る．1 を足すことで，外積の中に単モダリティの項と 2 つずつの項も残る（4.2 節）．

  ```
  z_m ∈ R^d   （m = 言葉 l，映像 v，音声 a の埋め込み）
  z^m = [z_m ; 1]
  Z = z^l ⊗ z^v ⊗ z^a   ∈ R^{(d_l+1)×(d_v+1)×(d_a+1)}
  ⊗ は外積．Z の中に z_l, z_v, z_a, z_l⊗z_v, z_l⊗z_a, z_v⊗z_a, z_l⊗z_v⊗z_a がすべて入る
  Z を平らにして全結合の網で感情価を予測する
  ```
- **分かったこと**：MOSI で 2 値の正解率 77.1，5 クラス 42.0，平均絶対誤差 0.87，相関 0.70（表 1）．単モダリティでは言葉が最も強い（2 値 74.8，映像 66.8，音声 65.1）．特徴をつなぐだけの早期融合は 75.2 だった（表 2）．
- **前提と限界（著者）**：5 分割の交差検証で，原文 "no speaker is shared between train and test sets"（5.4 節）．外積の次元は掛け算で増える．
- **MER-DA との接点**（推測．未検定）：TFN の項のうち単モダリティの項だけを残すと，MER-DA の線形の連結の分類器と同じ形になる（2.2 節の Baltrušaitis の項の式）．K が小さい MER-DA で 2 次・3 次の項を足すと，パラメータが掛け算で増える．
- **確認**：◎（本文）．[ACL Anthology](https://aclanthology.org/D17-1115/)，[arXiv](https://arxiv.org/abs/1707.07250)

### Liu, Shen, Lakshminarasimhan, Liang, Zadeh, Morency 2018（ACL 2018, pp. 2247–2256．arXiv 1806.00064）Efficient Low-rank Multimodal Fusion with Modality-Specific Factors（LMF）

- **問い**：TFN の外積の重みはモダリティの数に対して指数的に大きくなるので，同じ形の融合を少ない計算でする．
- **方法**：外積に掛ける重みの塊（テンソル）を，モダリティごとの低い階数 r の因子の和に分ける．すると外積を作らずに計算できる（3.2 節，式 6）．

  ```
  W ≈ Σ_{i=1..r} ⊗_{m=1..M} w_m^(i)
  h = Λ_{m=1..M} [ Σ_{i=1..r} w_m^(i) · z^m ]
  Λ は要素ごとの積．z^m = [z_m ; 1]
  ```
- **分かったこと**：IEMOCAP の F1 は Happy 85.8，Sad 85.9，Angry 89.0，Neutral 71.7 で，TFN（83.6，82.8，84.2，65.4）より高い（表 2）．パラメータは TFN の約 1/11，学習の速さは 1 秒あたり 1,134.82 件（TFN 340.74 件）である（5.3 節，表 3）．
- **前提と限界（著者）**：分割は話者独立で，IEMOCAP は 6,373/1,775/1,807 区間（表 1）．F1 は感情ごとの値である．
- **MER-DA との接点**（推測．未検定）：IEMOCAP を 4 感情それぞれの F1 で報告する CMU の系の評価で，MER-DA の 4 クラスの UAR とは比べられない（3.2 節）．
- **確認**：◎（本文）．[ACL Anthology](https://aclanthology.org/P18-1209/)，[arXiv](https://arxiv.org/abs/1806.00064)

### Zadeh, Liang, Mazumder, Poria, Cambria, Morency 2018（AAAI 2018, 32(1)．arXiv 1802.00927）Memory Fusion Network for Multi-view Sequential Learning（MFN）

- **問い**：時系列の各時刻で，モダリティ内の変化とモダリティ間の関係の両方を追う．
- **方法**：モダリティごとの LSTM（時系列を読む再帰の網）の記憶を，前の時刻と今の時刻の 2 つ並べて注意（Delta-memory Attention）をかけ，その結果で共有の記憶 u を更新する（3 節）．

  ```
  c[t−1,t] = [c_{t−1} ; c_t]            （全モダリティの LSTM の記憶を 2 時刻ぶん並べる）
  ĉ[t−1,t] = c[t−1,t] ⊙ a[t−1,t]        （注意の重み a を要素ごとに掛ける）
  u_t = γ1_t ⊙ u_{t−1} + γ2_t ⊙ tanh(û_t)  （γ1 は残す門，γ2 は更新する門，û_t は更新の案）
  ```
- **分かったこと**：IEMOCAP の 9 感情で MA(9) 36.5，F1 34.9（表 2）．
- **前提と限界（著者）**：分割は話者独立（表 1）．
- **MER-DA との接点**（推測．未検定）：時系列のモデルなので，発話ごとに平均した特徴だけを使う MER-DA とは入力の形が違う．MOSEI の論文の Graph-MFN はこの網の融合部分を DFG に替えたものである．
- **確認**：◎（本文の 3 節と表）．[DOI](https://doi.org/10.1609/aaai.v32i1.12021)，[arXiv](https://arxiv.org/abs/1802.00927)

### Tsai, Bai, Liang, Kolter, Morency, Salakhutdinov 2019（ACL 2019, pp. 6558–6569．arXiv 1906.00295）Multimodal Transformer for Unaligned Multimodal Language Sequences（MulT）

- **問い**：言葉・音声・映像の系列は時間の刻みが違い，単語の時刻にそろえる前処理が要る．そろえずに直接融合する．
- **方法**：あるモダリティ β の系列を，別のモダリティ α の系列の問いに答える形で取り込む注意（crossmodal attention）を，6 つの向きの組で重ねる（3 節）．

  ```
  Q_α = X_α W_Q,  K_β = X_β W_K,  V_β = X_β W_V
  Y_α = CM_{β→α}(X_α, X_β) = softmax(Q_α K_βᵀ / √d_k) V_β
  Y_α の長さは X_α と同じ．i 行目は，α の i 番目の時刻から見た β の重み付き要約
  ```
- **分かったこと**：単語の時刻にそろえた IEMOCAP で，2 値の正解率と F1 は Happy 90.7・88.6，Sad 86.7・86.0，Angry 87.4・87.0，Neutral 72.4・70.7（表 3）．そろえない条件でも既存の方法を上回った．
- **前提と限界（著者）**：IEMOCAP は 4 感情を選び，複数ラベルの課題として扱い，感情ごとの 2 値の正解率と F1 を報告する（4.1 節）．sad の結果だけは既存の方法を上回らなかった．
- **MER-DA との接点**（推測．未検定）：MER-DA は発話ごとに平均した特徴を使うので，時刻のそろえ方の問題は起きない．IEMOCAP で 90% 前後の数値は「感情ごとの 2 値」であり，4 クラスの UAR とは別物である（3.2 節）．
- **確認**：◎（本文の 3〜5 節）．[ACL Anthology](https://aclanthology.org/P19-1656/)，[arXiv](https://arxiv.org/abs/1906.00295)

### Rahman, Hasan, Lee, Zadeh, Mao, Morency, Hoque 2020（ACL 2020, pp. 2359–2369．arXiv 1908.05787）Integrating Multimodal Information in Large Pretrained Transformers（MAG-BERT）

- **問い**：BERT や XLNet は言葉しか受け取れないので，微調整のときに声と顔を入れる部品を足す．
- **方法**：各単語の表現 Z_i を，声 A_i と顔 V_i から作ったずらしの量 H_i だけ動かす．ずらしの大きさは Z_i の大きさに比例する上限で抑える（4 節，式 1〜5）．

  ```
  g_i^v = R(W_gv [Z_i ; V_i] + b_v),   g_i^a = R(W_ga [Z_i ; A_i] + b_a)   （R は非線形の活性化）
  H_i = g_i^a · (W_a A_i) + g_i^v · (W_v V_i) + b_H
  Z̄_i = Z_i + α H_i,   α = min( β ‖Z_i‖₂ / ‖H_i‖₂ , 1 )
  ```
- **分かったこと**：MOSI で MAG-BERT の 2 値の正解率 84.2/86.1，F1 84.1/86.0，平均絶対誤差 0.712，相関 0.796．MAG-XLNet は 84.7/86.7（表 1）．
- **前提と限界（著者）**：2 値の正解率を 2 通りの定義で併記している（「/」の左は Zadeh 2018 の定義，右は Tsai 2019 の定義）（5 節，表 1）．
- **MER-DA との接点**（推測．未検定）：言葉の表現を主にし，声と顔はその補正として足す設計である．MER-DA の多くの組で言葉が最も強い単モダリティであれば，同じ考え方（言葉の分類器に他モダリティのずらしを足す）を線形で書ける．Self-MM の論文が同じ条件で再現すると 82.54/84.3 であり（Self-MM 表 1），報告値と約 2 点違う．
- **確認**：◎（本文）．[ACL Anthology](https://aclanthology.org/2020.acl-main.214/)，[arXiv](https://arxiv.org/abs/1908.05787)

### Hazarika, Zimmermann, Poria 2020（ACM Multimedia 2020, pp. 1122–1131．arXiv 2005.03545）MISA: Modality-Invariant and -Specific Representations for Multimodal Sentiment Analysis

- **問い**：融合の前に，各モダリティの表現を「モダリティに共通の部分」と「そのモダリティに固有の部分」に分けて学べば，融合が楽になるか．
- **方法**：各モダリティの発話の表現 u_m を，共通の部分 h_m^c と固有の部分 h_m^p に写す．4 つの損失を足して学ぶ（3.5 節）．

  ```
  L = L_task + α L_sim + β L_diff + γ L_recon
  L_sim ：モダリティ間の共通の部分の分布の差（CMD：分布の平均と高次のモーメントの差の和）
  L_diff：Σ_m ‖ H_m^cᵀ H_m^p ‖_F²  ＋ 固有の部分どうしの同様の項（共通と固有を直交に近づける）
  L_recon：共通と固有の和から元の表現を復元する誤差
  ```
- **分かったこと**：2 値の正解率は MOSI 81.8/83.4，MOSEI 83.6/85.5（表 1・2．「/」の左は negative/non-negative，右は negative/positive の定義）．
- **前提と限界（著者）**：今後の課題として，原文 "we plan to analyze MISA in other dimensions of affect, such as emotions"（6 節）．
- **MER-DA との接点**（推測．未検定）：MISA が近づけるのは同じ発話の中のモダリティ間の分布であり，コーパス間の分布ではない．MER-DA のコーパス間のずれの指標（`measure/shift.py`）とは近づける対象が違う．Self-MM の論文の再現値は MOSI 80.79/82.1 で，報告値より約 1 点低い（Self-MM 表 1）．
- **確認**：◎（本文）．[DOI](https://doi.org/10.1145/3394171.3413678)，[arXiv](https://arxiv.org/abs/2005.03545)

### Yu, Xu, Yuan, Wu 2021（AAAI 2021, 35(12):10790–10797．arXiv 2102.04830）Learning Modality-Specific Representations with Self-Supervised Multi-Task Learning for Multimodal Sentiment Analysis（Self-MM）

- **問い**：CH-SIMS のようなモダリティごとのラベルは付けるのに費用がかかるので，全体のラベルとモダリティの表現から自動で作る．
- **方法**：学習中に，全体の表現 F_m と各モダリティの表現 F_s について，肯定のクラスの中心 C_p と否定のクラスの中心 C_n からの距離を測る．相対的な距離の差の分だけ，全体のラベルをずらしてモダリティのラベルにする（3.3 節，式 3〜8）．ラベルは運動量のように前のエポックの値と混ぜて更新する．

  ```
  D_ip = ‖F_i − C_ip‖₂ / √d_i,   D_in = ‖F_i − C_in‖₂ / √d_i     （i ∈ {m, t, a, v}）
  α_i = (D_in − D_ip) / (D_ip + ε)                                （肯定の中心への相対的な近さ）
  y_s = y_m + (α_s − α_m)/2 + y_m (α_s − α_m) / (2 α_m)          （s ∈ {t, a, v}．式 8 の整理）
  ```
- **分かったこと**：同じ条件で再現した比較で，MOSI の 2 値の正解率 84.00/85.98，MOSEI 82.81/85.17（表 1）．CH-SIMS では，人が付けたモダリティのラベルを使う版とほぼ同じ成績だった（表 3）．
- **前提と限界（著者）**：モダリティのラベルの元は全体のラベルであり，その差分を表現の距離から推定する．
- **MER-DA との接点**（推測．未検定）：「クラスの中心からの相対的な距離で，モダリティごとにラベルを補正する」は，ほかのモダリティの予測を使って弱いラベルを作る MER-DA の手法の部品と並べて書ける先例である．Self-MM はコーパス内で全ラベルを使う設定である．
- **確認**：◎（本文）．[DOI](https://doi.org/10.1609/aaai.v35i12.17289)，[arXiv](https://arxiv.org/abs/2102.04830)

### Hu, Lin, Zhao, Lu, Wu, Li 2022（EMNLP 2022, pp. 7837–7851．arXiv 2211.11256）UniMSE: Towards Unified Multimodal Sentiment Analysis and Emotion Recognition

- **問い**：MSA（感情価の強さ）と ERC（感情のカテゴリ）は別々に研究されてきたが，両者の知識を 1 つのモデルで共有する．
- **方法**：
  - 各標本のラベルを，極性・強さ・感情のカテゴリの 3 つ組（universal label）にそろえる．欠けている部分は，同じ極性を持つ別のデータの標本のうち，文の意味が最も近いもの（SimCSE の文の埋め込みの類似度）から借りる（3.2 節）．
  - T5（文を入力して文を出力する事前学習モデル）の符号器の後ろの 3 層に音声と映像を差し込み，モダリティ間の対照学習の損失を足す．
  - MOSI・MOSEI・MELD・IEMOCAP の学習用の分割を全部まとめて 1 つのモデルを学ぶ（4.4 節）．
- **分かったこと**：MELD の正解率 65.09，重み付き F1 65.51．IEMOCAP の正解率 70.56，重み付き F1 70.66．MOSI の 2 値の正解率 85.85/86.9，MOSEI 85.86/87.50（表 2）．著者は，後年の論文が一部のデータと指標でしか評価しないことを問題に挙げている（4.5 節）．
- **前提と限界（著者）**：原文 "the generation of universal labels only considers textual modality, without considering acoustic and visual modalities"（Limitations）．
- **MER-DA との接点**（推測．未検定）：融合の文献の中で，複数の感情コーパスをまとめて学ぶ最も有名な例である．ただし評価は各コーパスの中の分割で行い，あるコーパスで学んで別のコーパスで評価するコーパス間の評価はしていない．ラベルの付け方の違いを，文の類似度でラベルを借りることで埋めている．
- **確認**：◎（本文）．[ACL Anthology](https://aclanthology.org/2022.emnlp-main.534/)，[arXiv](https://arxiv.org/abs/2211.11256)

### 2.4 会話の感情認識（ERC）

### Majumder, Poria, Hazarika, Mihalcea, Gelbukh, Cambria 2019（AAAI 2019, 33(01):6818–6825．arXiv 1811.00405）DialogueRNN: An Attentive RNN for Emotion Detection in Conversations

- **問い**：会話の感情は，話者が誰か，前の発話の文脈，前の感情で決まる．これらを話者ごとに追う．
- **方法**：3 つの GRU（再帰の網）を使う．会話全体の状態 g，話者ごとの状態 q，感情の表現 e である（Our Model の節）．

  ```
  g_t = GRU_G(g_{t−1}, u_t ⊕ q_{s(u_t),t−1})          （u_t は t 番目の発話，s(u_t) はその話者）
  c_t = 過去の g への注意で作った文脈
  q_{s(u_t),t} = GRU_P(q_{s(u_t),t−1}, u_t ⊕ c_t)
  e_t = GRU_E(e_{t−1}, q_{s(u_t),t})  → softmax で感情
  ```
- **分かったこと**：IEMOCAP の 6 クラス（happy, sad, neutral, angry, excited, frustrated）で，双方向で注意を使う版の重み付き平均の正解率 63.40，F1 62.75（表 2）．3 モダリティを連結した入力では F1 62.9 で，言葉だけ（62.7）と 0.2 点しか違わない（表 3）．
- **前提と限界（著者）**：主に言葉で評価した．分割はセッション 1〜4 の 8 人を学習，セッション 5 を評価（5,810 と 1,623 発話）で，話者を共有しない（Experimental Setting の節，表 1）．
- **MER-DA との接点**（推測．未検定）：会話の文脈を使うと IEMOCAP と MELD の成績が上がるが，MER-DA は発話を 1 つずつ扱う．論文で「文脈を使わない」ことを限界として書く根拠になる．
- **確認**：◎（本文）．[DOI](https://doi.org/10.1609/aaai.v33i01.33016818)，[arXiv](https://arxiv.org/abs/1811.00405)

### Ghosal, Majumder, Poria, Chhaya, Gelbukh 2019（EMNLP-IJCNLP 2019, pp. 154–164．arXiv 1908.11540）DialogueGCN: A Graph Convolutional Neural Network for Emotion Recognition in Conversation

- **問い**：再帰の網は長い会話で遠い発話の情報を伝えにくいので，話者間の依存をグラフで表す．
- **方法**：発話を節点にし，前後 10 発話の窓の中の節点を辺でつなぐ．辺の種類（関係）を「どの話者からどの話者へ」と「前か後か」で分け，話者が M 人なら最大 2M² 種類にする．関係ごとに別の重みで近傍の情報を集める（3 節）．

  ```
  h_i = σ( Σ_{r∈R} Σ_{j∈N_i^r} (α_ij / c_{i,r}) W_r g_j + α_ii W_0 g_i )
  r は関係の種類，N_i^r は関係 r で i につながる節点，α は辺の重み
  ```
- **分かったこと**：IEMOCAP の 6 クラスで正解率 65.25，重み付き平均の F1 64.18．MELD の重み付き F1 は 58.10（5 節，表 3・4）．
- **前提と限界（著者）**：言葉だけで評価した．原文 "Multimodal emotion recognition is outside the scope of this paper, and is left as future work."（4.1 節）．MELD は発話が短く感情の語を含みにくいので難しいと書いている．
- **MER-DA との接点**（推測．未検定）：MELD の重み付き F1 は言葉だけでも 58 前後であり，MER-DA の MELD の数値（4 クラスの UAR）とは課題も指標も違う．
- **確認**：◎（本文）．[ACL Anthology](https://aclanthology.org/D19-1015/)，[arXiv](https://arxiv.org/abs/1908.11540)

### Hu, Liu, Zhao, Jin 2021（ACL-IJCNLP 2021, pp. 5666–5675．arXiv 2107.06779）MMGCN: Multimodal Fusion via Deep Graph Convolution Network for Emotion Recognition in Conversation

- **問い**：会話の感情認識の多くは言葉だけか単純な連結なので，モダリティ間と発話間の関係を 1 つのグラフで扱う．
- **方法**：各発話に音声・映像・言葉の 3 つの節点を作る．同じモダリティの節点は会話の中ですべてつなぎ，同じ発話の異なるモダリティの節点もつなぐ．話者の埋め込みを足す．辺の重みは角度の類似度で決め，異なるモダリティの辺には係数 γ を掛ける（3 節，式 4・5）．

  ```
  同じモダリティの辺：  A_ij = 1 − arccos(sim(n_i, n_j)) / π
  異なるモダリティの辺：A_ij = γ (1 − arccos(sim(n_i, n_j)) / π)
  ```
- **分かったこと**：重み付き F1（表 3）は次のとおりである．

  | 入力 | IEMOCAP | MELD |
  |---|---|---|
  | 音声 | 54.66 | 42.63 |
  | 映像 | 33.86 | 33.27 |
  | 言葉 | 62.35 | 57.72 |
  | 音声＋言葉 | 65.70 | 58.02 |
  | 映像＋言葉 | 62.89 | 57.92 |
  | 3 つ | 66.22 | 58.65 |

- **前提と限界（著者）**：IEMOCAP の 6 クラスと MELD の 7 クラスで評価した．
- **MER-DA との接点**（推測．未検定）：会話の感情認識では，映像が最も弱く，MELD では 3 つを使っても言葉だけより 0.93 点しか上がらない．MER-DA の MELD で別モダリティを足す得が小さい場合の，同じ向きの先行の数値として引ける．
- **確認**：◎（本文）．[ACL Anthology](https://aclanthology.org/2021.acl-long.440/)，[arXiv](https://arxiv.org/abs/2107.06779)

### 2.5 モダリティの釣り合い（融合の学習で一部のモダリティしか学ばれない問題）

### Wang, Tran, Feiszli 2020（CVPR 2020, pp. 12692–12702．arXiv 1905.12681）What Makes Training Multi-Modal Classification Networks Hard?

- **問い**：複数モダリティの網は単モダリティの網より多くの情報を受け取るのに，実際には最良の単モダリティの網に負けることが多い．その原因と対処を調べる．
- **方法**：過学習と汎化の比 OGR を定義する．学習の損失 L^T と，検証の損失で近似した本当の損失 L* の差を過学習 O とする（2.2 節）．モダリティごとの頭（予測器）を付け，各頭の損失の重み w_k を，OGR を小さくするように決める（Gradient-Blending．命題 1）．

  ```
  O_N = L*_N − L^T_N
  OGR = ΔO_{N,n} / ΔG_{N,n} = (O_{N+n} − O_N) / (L*_N − L*_{N+n})
  命題 1（重みの和が 1 の下で OGR² を最小にする重み）：
     w_k* = (1/Z) · ⟨∇L*, v_k⟩ / (2 σ_k²),   σ_k² = E[⟨∇L^T − ∇L*, v_k⟩²]
  損失 = Σ_k w_k L_k   （L_k はモダリティ k の頭と，融合の頭の損失）
  ```
- **分かったこと**：Kinetics の音声＋RGB で，素朴な一緒の学習 71.7，RGB だけ 72.6，Gradient-Blending（学習前に重みを決める版）74.7，（学習中に更新する版）75.8（動画単位の上位 1 位の正解率．表 2）．事前学習・早期終了・dropout では最良の単モダリティを大きく超えなかった（図 1）．
- **前提と限界（著者）**：原文 "different modalities overfit and generalize at different rates, so training them jointly with a single optimization strategy is sub-optimal"（抄録）．重みの推定には検証用のデータが要る．
- **MER-DA との接点**（推測．未検定）：MER-DA の連結の線形分類器は 2,304 次元（1024＋512＋768）で，単モダリティより係数が多い．K が小さいときに連結が最良の単モダリティに負けるなら，その原因の 1 つの候補は容量の差による過学習である（Huang 2021 の理論とも同じ向き．`docs/related-work.md` 2.12 節）．
- **確認**：◎（抄録・1〜3 節）．[DOI](https://doi.org/10.1109/CVPR42600.2020.01271)，[arXiv](https://arxiv.org/abs/1905.12681)

### Peng, Wei, Deng, Wang, Hu 2022（CVPR 2022, pp. 8228–8237．arXiv 2203.15332）Balanced Multimodal Learning via On-the-fly Gradient Modulation（OGM-GE）

- **問い**：融合のモデルが単モダリティより良くても，各モダリティの表現が学び足りないまま終わる．強いモダリティが学習を独占する問題を，勾配の大きさで直す．
- **方法**：各モダリティの部分だけで正解のクラスに付ける確率の和を比べ，比が 1 より大きい（強い）モダリティの勾配を縮める．縮めた分の汎化の低下を補うため，勾配にガウス雑音を足す（GE）（3.2〜3.3 節）．

  ```
  s_i^u = softmax(W^u φ^u(x_i^u) + b/2)_{y_i}          （モダリティ u だけでの正解のクラスの確率）
  ρ_t^v = Σ_{i∈B_t} s_i^v / Σ_{i∈B_t} s_i^a,   ρ_t^a = 1/ρ_t^v
  k_t^u = 1 − tanh(α · ρ_t^u)  （ρ_t^u > 1 のとき），それ以外は 1
  θ_{t+1}^u = θ_t^u − η · k_t^u · g̃(θ_t^u)
  ```
- **分かったこと**：CREMA-D の正解率は，音声だけ 52.5，映像だけ 41.9，連結 51.7，連結＋OGM-GE 61.9（表 1）．音声だけのモデルが素朴な融合の全手法を上回っていた．
- **前提と限界（著者）**：原文 "even equipped with OGM-GE, the uni-modal performance in multimodal model still do not surpass the best uni-modal model"（5 節）．CREMA-D は無作為に学習・検証 6,698 本と評価 744 本に分けており，話者で分けていない（4.1 節）．
- **MER-DA との接点**（推測．未検定）：OGM-GE が扱うのはエンコーダを一緒に学ぶときの不釣り合いなので，凍結した特徴の上の線形分類器にはそのまま当てはまらない．CREMA-D の数値は話者が学習と評価の両方に入る分割のものであり，MER-DA の CREMA-D の UAR とは比べられない．
- **確認**：◎（本文の 3〜5 節）．[DOI](https://doi.org/10.1109/CVPR52688.2022.00806)，[arXiv](https://arxiv.org/abs/2203.15332)

### Huang, Lin, Zhou, Yang, Huang 2022（ICML 2022, PMLR 162．arXiv 2203.12221）Modality Competition: What Makes Joint Training of Multi-modal Network Fail in Deep Learning? (Provably)

- **問い**：最良の単モダリティの網が一緒に学んだ融合の網に勝つという Wang 2020 の観察を，理論で説明する．
- **方法**：各モダリティが疎な符号化の模型から生まれ，1 つのモダリティだけでは課題を十分に解けない（insufficient）データ分布を置く．各モダリティの符号化器は 1 層の滑らかな ReLU の網で，出力を足し合わせる後期融合を勾配降下で学ぶ（2・4 節）．
- **分かったこと**：
  - 定理 5.1：単モダリティで学ぶと，各網はそのモダリティの特徴を学ぶ．
  - 定理 5.2：一緒に学ぶと，ランダムな初期値との相関が大きいモダリティが「勝ち」，符号化器は一部のモダリティの特徴しか学ばない（modality competition）．
  - 系 5.3：その結果，一緒に学んだ網の試験の誤差は最良の単モダリティの網より大きくなりうる．
- **前提と限界（著者）**：符号化器を学ぶ後期融合の 1 層の網についての結果である．著者は，各モダリティの良い特徴を取り出せる事前学習が融合を良くする理由の説明にもなると書く（7 節）．
- **MER-DA との接点**（推測．未検定）：MER-DA は符号化器を凍結しているので，この理論の「符号化器が一部のモダリティしか学ばない」仕組みは起きない．MER-DA で融合が単モダリティに負ける場合は，別の説明（容量と標本数の兼ね合い，ラベルの付け方のずれ）を探す必要がある．
- **確認**：◎（抄録・1 節・定理の主張・7 節．証明は読んでいない）．[PMLR](https://proceedings.mlr.press/v162/huang22e.html)，[arXiv](https://arxiv.org/abs/2203.12221)

### Du, Teng, Li, Liu, Yuan, Wang, Yuan, Zhao 2023（ICML 2023, PMLR 202．arXiv 2305.01233）On Uni-Modal Feature Learning in Supervised Multi-Modal Learning（UMT/UME）

詳しくは `research_notes/先行研究_Albanieの後続研究_2026-11-04/report_C_3モダリティと融合.md` の Q3 と 3.4 節．

- **問い**：一緒に学ぶ後期融合のモデルは，単モダリティだけで学べる特徴（uni-modal features）を学び足りない（著者の語で Modality Laziness）．それが汎化を悪くすることを示し，対処を選ぶ指針を出す．
- **方法**：2 つの方法から選ぶ．UME（Uni-Modal Ensemble）は単モダリティのモデルを別々に学んで予測を平均する．UMT（Uni-Modal Teacher）は，単独で学んだ単モダリティのモデル（蒸留元のモデル．論文の語で teacher）の特徴を，融合のモデルの各モダリティの部分へ特徴の段で蒸留する．選び方は，凍結した単モダリティの符号化器の上で連結の線形分類器を学び，それが単モダリティの予測の平均より良ければ UMT，そうでなければ UME である（3〜4 節）．
- **分かったこと**：凍結した符号化器の上の連結の線形分類器でも，分類器の中の各モダリティの部分は単独で学んだ分類器より弱くなる（UCF101 の RGB で 68.2 と 77.1．付録 A.7.1．既存のノートによる）．
- **前提と限界（著者）**：2 モダリティ，全ラベル，同じ分布の評価である．
- **MER-DA との接点**（推測．未検定）：「連結の線形分類器」と「単モダリティの予測の平均」を比べる手順は，MER-DA の設定（凍結した特徴の上の線形分類器）でそのまま実行できる．K ごと・ターゲットごとにどちらが良いかを測ることは，MER-DA の融合の作り方を決める材料になる．
- **確認**：◎（抄録・3〜4 節．付録の数値は既存のノートによる）．[PMLR](https://proceedings.mlr.press/v202/du23e.html)，[arXiv](https://arxiv.org/abs/2305.01233)

### 2.6 欠けたモダリティ

### Zhao, Li, Jin 2021（ACL-IJCNLP 2021, pp. 2608–2618．arXiv なし）Missing Modality Imagination Network for Emotion Recognition with Uncertain Missing Modalities（MMIN）

- **問い**：実際の利用ではどのモダリティが欠けるか分からない．欠け方ごとに別のモデルを学ばずに，1 つのモデルで対応する．
- **方法**：使えるモダリティの表現から，欠けたモダリティの表現を予測する（論文の語で imagination）．予測には残差を重ねた自己符号化器（Cascade Residual Autoencoder）を使い，欠けた側から使える側へ戻す予測も学ぶ（cycle consistency）．学習では 6 通りの欠け方（{a}，{v}，{t}，{a,v}，{a,t}，{v,t}）を作る（3 節）．
- **分かったこと**：IEMOCAP の 4 クラスで，全モダリティがそろう条件の WA 0.7650，UA 0.7812（表 5）．全モダリティで学んだ基準のモデルを音声だけで評価すると WA 0.4190，UA 0.4719 に下がる．MMIN では音声だけで WA 0.5658，UA 0.5900 である（表 5）．
- **前提と限界（著者）**：IEMOCAP は 10 分割の話者独立の交差検証で，4 セッションを学習に，残る 1 セッションを話者で検証と評価に分けた．指標は WA と UA．MSP-IMPROV は偏りがあるので F1 にした（4.5 節）．
- **MER-DA との接点**（推測．未検定）：全モダリティで学んだモデルは，評価のときに 1 つのモダリティしか無いと大きく下がる．MER-DA でコーパスによってモダリティが欠ける（RAVDESS に映像が無い，CREMA-D と RAVDESS に言葉が無い）ことを，融合の群を限るのではなく欠けたまま扱う場合の比較相手になる．
- **確認**：◎（本文の 1・3・4 節）．[ACL Anthology](https://aclanthology.org/2021.acl-long.203/)

### 2.7 自己教師ありの音声エンコーダ

### Baevski, Zhou, Mohamed, Auli 2020（NeurIPS 2020．arXiv 2006.11477）wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations

- **問い**：ラベルの無い音声だけで表現を学び，少しの書き起こしで微調整して音声認識を良くする．
- **方法**：畳み込みの符号化器が波形を潜在表現にし，その一部の時刻を隠して Transformer に入れる．隠した時刻の正しい量子化表現（離散の符号帳から選んだ表現）を，同じ発話のほかの時刻から取った紛らわしい候補 K 個の中から当てる（3 節）．

  ```
  L = L_m + α L_d       （L_d は符号帳の項目を均等に使わせる損失）
  L_m = − log [ exp(sim(c_t, q_t)/κ) / Σ_{q̃∈Q_t} exp(sim(c_t, q̃)/κ) ]
  c_t は隠した時刻 t の文脈表現，q_t は正しい量子化表現，Q_t は q_t と紛らわしい候補 K 個
  ```
- **分かったこと**：LibriSpeech の全ラベルで単語誤り率 1.8/3.3（clean/other）．ラベル 10 分とラベル無し 5.3 万時間で 4.8/8.2（抄録）．
- **前提と限界（著者）**：事前学習は朗読の音声（LibriSpeech，LibriVox）である．
- **MER-DA との接点**（推測．未検定）：MER-DA の音声のエンコーダ（WavLM，HuBERT）はこの枠組みの後継である．
- **確認**：◎（抄録・1〜3 節）．[NeurIPS](https://proceedings.neurips.cc/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html)，[arXiv](https://arxiv.org/abs/2006.11477)

### Hsu, Bolte, Tsai, Lakhotia, Salakhutdinov, Mohamed 2021（IEEE/ACM TASLP 29:3451–3460．arXiv 2106.07447）HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units

- **問い**：音声には単位の辞書も区切りも無いので，隠した部分を当てる学習の「当てる的」をどう作るか．
- **方法**：まず MFCC（音声の古典的な特徴）を k-means で 100 個のクラスタに分け，その番号を各時刻の的にする（論文は "k-means teacher" と呼ぶ）．隠した時刻の番号を当てる損失で学び，学んだ表現をまた k-means にかけて的を作り直す（II 節）．

  ```
  L = α L_m + (1 − α) L_u
  L_m = Σ_{t∈M} log p_f(z_t | X̃, t)    （M は隠した時刻，z_t はクラスタ番号，X̃ は隠した入力）
  L_u は隠していない時刻についての同じ形の項．α = 1 は隠した時刻だけで損失を取る
  ```
- **分かったこと**：LibriSpeech 960 時間か Libri-Light 6 万時間で事前学習し，微調整のラベルが 10 分〜960 時間のすべてで wav2vec 2.0 と同等かそれ以上．10 億パラメータの版は dev-other と test-other で最大 19% と 13% の相対的な誤り率の低下（抄録）．
- **前提と限界（著者）**：著者は，的のラベルが正しいことより，一貫していることが重要だと書く（I 節）．
- **MER-DA との接点**（推測．未検定）：MER-DA の第 2 の音声エンコーダ `facebook/hubert-large-ll60k` は Libri-Light 6 万時間で学んだ Large である（`data/README.md` 4 節）．朗読の音声だけで学んでいる．
- **確認**：◎（抄録・I〜II 節）．[DOI](https://doi.org/10.1109/TASLP.2021.3122291)，[arXiv](https://arxiv.org/abs/2106.07447)

### Chen ほか 2022（IEEE Journal of Selected Topics in Signal Processing 16(6):1505–1518．arXiv 2110.13900）WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing

- **問い**：音声認識以外の課題（話者・感情・分離など）にも使える汎用の音声表現を作る．
- **方法**：HuBERT の隠した部分の予測に，雑音や別の話者の声を混ぜた入力から元の話者の的を当てる課題（masked speech denoising）を足す．Transformer に門付きの相対位置の項を加える．事前学習の音声を 6 万時間から 9.4 万時間に増やした（Libri-Light 6 万，GigaSpeech 1 万〔朗読・ポッドキャスト・YouTube〕，VoxPopuli の英語 2.4 万〔欧州議会〕）（I 節，IV-C 節）．
- **分かったこと**：SUPERB の感情認識（IEMOCAP の 4 クラスの正解率）で WavLM Base 65.94，Base+ 68.65，Large 70.62．同じ表で HuBERT Base 64.92，wav2vec 2.0 Base 63.43（表 I）．
- **前提と限界（著者）**：SUPERB の方針に従って，各層の出力の重み付き和を下流の課題に使っている．
- **MER-DA との接点**（推測．未検定）：
  - MER-DA の第 1 の音声エンコーダ `microsoft/wavlm-large` は，YouTube やポッドキャストを含む音声で事前学習している．MOSEI（YouTube）の音声と事前学習のデータの性質が近い可能性がある．
  - MER-DA は最終層の出力を時間で平均している（`data/README.md` 4 節）．SUPERB と WavLM の論文は層の重み付き和を使っており，最終層が感情に最も良いとは限らないと書く（SUPERB 3 節）．MER-DA の音声の単独の成績が層の選び方で変わるかは測っていない．
- **確認**：◎（I 節・IV-C 節・表 I）．[DOI](https://doi.org/10.1109/JSTSP.2022.3188113)，[arXiv](https://arxiv.org/abs/2110.13900)

### Yang ほか 2021（Interspeech 2021, pp. 1194–1198．arXiv 2105.01051）SUPERB: Speech processing Universal PERformance Benchmark

- **問い**：自己教師ありの音声表現を，多くの課題で同じ条件で比べる．
- **方法**：エンコーダを凍結し，各層の出力の重み付き和を表現にする．感情認識（ER）は，時間方向の平均の上に線形の層を 1 つ置き，交差エントロピーで学ぶ（3 節）．IEMOCAP の 4 クラス（neutral，happy，sad，angry）で，原文 "cross-validates on five folds of the standard splits"．指標は正解率（2 節）．
- **分かったこと**：ER の正解率は，FBANK（フィルタバンクの特徴）35.39，wav2vec 2.0 Base 63.43・Large 65.64，HuBERT Base 64.92・Large 67.62（表 2）．
- **前提と限界（著者）**：原文 "the last-layer representation is not always the best"（3 節）．ER の 4 クラスへの絞り方は "drop the unbalance emotion classes" とだけ書き，happy と excited をまとめるかは本文に無い．
- **MER-DA との接点**（推測．未検定）：凍結した特徴と線形の頭で感情を評価する型は MER-DA と同じである．SUPERB の正解率は WA に当たり，UAR ではない．
- **確認**：◎（本文）．[DOI](https://doi.org/10.21437/Interspeech.2021-1775)，[arXiv](https://arxiv.org/abs/2105.01051)

### 2.8 顔と言葉のエンコーダ（書誌だけ）

### Liu ほか 2019（arXiv 1907.11692）RoBERTa: A Robustly Optimized BERT Pretraining Approach

- **問い・方法・分かったこと**：BERT の事前学習の設定（学習の長さ，データの量など）を一つずつ確かめ直し，BERT は学び足りていなかったと結論した．GLUE・RACE・SQuAD で当時の最高の成績（抄録）．
- **MER-DA との接点**：MER-DA の第 1 の言葉のエンコーダ `roberta-base`（768 次元，最終層を単語方向に平均）の出典．査読付きの会場の版は無く，arXiv で引く．
- **確認**：○（arXiv の抄録の頁）．[arXiv](https://arxiv.org/abs/1907.11692)

### He, Zhang, Ren, Sun 2016（CVPR 2016, pp. 770–778．arXiv 1512.03385）Deep Residual Learning for Image Recognition

- **問い・方法・分かったこと**：層の入力を基準にした残差を学ぶ形にすると，深い網を学びやすくなる．152 層の網の組み合わせで ImageNet の試験の誤り率 3.57%（抄録）．
- **MER-DA との接点**：MER-DA の第 1 の顔のエンコーダは ResNet-50 の構造で，重みは Ryumina らの静的な表情のモデル（VGGFace2 で学び AffectNet で微調整）である（`data/README.md` 4 節）．構造の出典としてだけ引く．
- **確認**：○（Crossref と arXiv の抄録の頁）．[DOI](https://doi.org/10.1109/CVPR.2016.90)，[arXiv](https://arxiv.org/abs/1512.03385)

### 2.9 感情の表し方

### Ekman 1992（Cognition and Emotion 6(3–4):169–200）An Argument for Basic Emotions

- **要点**：感情は生きるうえでの基本的な課題への対処として進化した，という立場を取る．各感情には固有の表出（signal），生理，先行する出来事があり，すべての感情に共通する性質として，速い立ち上がり，短い持続，意図しない発生，自動的な評価，反応のまとまりを挙げる（抄録）．著者は，ある顔の表情が文化をまたいで共通に見えるという自身の 20 年前の発見から出発したと書く（序論）．
- **MER-DA との接点**：MOSEI の 6 感情と MELD の 6 感情はこの系譜の「Ekman の基本感情」を使う．MER-DA の 4 クラスはその部分集合に neutral を足したものである．
- **確認**：○（雑誌の版の抄録と序論の 1 頁．PDF は画像なので OCR せずに画面で読んだ）．[DOI](https://doi.org/10.1080/02699939208411068)

### Russell 1980（Journal of Personality and Social Psychology 39(6):1161–1178）A Circumplex Model of Affect

- **要点**：感情の概念は独立な多くの次元ではなく，円の上に並ぶ．快（0°），興奮（45°），覚醒（90°），苦痛（135°），不快（180°），抑うつ（225°），眠気（270°），くつろぎ（315°）の順である．感情を表す 28 語を 4 通りの方法（円の並べ方，似ている度合いの多次元尺度法，快・不快と覚醒の 1 次元の尺度，343 人の自己報告の主成分分析）で配置して支持を得た（抄録）．
- **MER-DA との接点**：IEMOCAP・MSP-Podcast の次元のラベル（感情価・覚醒度）の理論の出典．MER-DA の 4 クラスでは，Happy と Angry は覚醒が近く感情価が違い，Sad と Neutral は覚醒が低い側にある，という読みの出典に使える（MER-DA の結果との対応は調べていない）．
- **確認**：○（PDF の抄録）．[DOI](https://doi.org/10.1037/h0077714)

### 2.10 コーパス間の感情認識

### Schuller, Vlasenko, Eyben, Wöllmer, Stuhlsatz, Wendemuth, Rigoll 2010（IEEE Transactions on Affective Computing 1(2):119–131）Cross-Corpus Acoustic Emotion Recognition: Variances and Strategies

詳しくは `docs/related-work.md` 2.3 節と付録の読書メモ．

- **要点**：6 つの音声感情コーパスで学習とテストの組を総当たりにし，コーパス間で成績が大きく下がることと，話者ごとの正規化が最も良いことを示した．演技・固定文のコーパスどうしはある程度移るが，自然発話へはほぼ偶然の水準である．
- **MER-DA との接点**（推測．未検定）：「話者独立で測る」「コーパス間で大きく下がる」の古典の根拠．中心合わせ（MH3）の最も近い先例．
- **確認**：○（今回は Crossref で書誌を確かめた．既存のノートは TAC の本文ではなく同じ内容の著者の拡張要旨を読んでいる）．[DOI](https://doi.org/10.1109/T-AFFC.2010.8)

### Albanie, Nagrani, Vedaldi, Zisserman 2018（ACM Multimedia 2018, pp. 292–301．arXiv 1808.05561）Emotion Recognition in Speech using Cross-Modal Transfer in the Wild

詳しくは `DAI_paper/参照論文/要点_問い1_別モダリティ蒸留.md` の 3 節．

- **要点**：顔の表情のモデル（蒸留元のモデル．論文の語で teacher）の予測を，ラベルの無い VoxCeleb の動画で音声のモデル（蒸留先．論文の語で student）に蒸留した．外部のコーパスでの音声のモデルの正解率は RML 49.7，eNTERFACE 34.3（表 5）．著者は，モダリティ間で共通の感情の信号の量は感情によって違うかもしれないと考察した（5.3 節）．
- **MER-DA との接点**（推測．未検定）：MER-DA の DAI の研究の土台の論文（`CLAUDE.md`）．蒸留の側の詳細は蒸留の担当のノートにある．
- **確認**：◎（既存のノートで本文を精読）．今回は Crossref で書誌を確かめた．[DOI](https://doi.org/10.1145/3240508.3240578)，[arXiv](https://arxiv.org/abs/1808.05561)

ほかのコーパス間の代表（Schuller 2011，Milner 2019，Ryumina 2025，EmoBox 2024，Wagner 2023，MERBench 2024）は `docs/related-work.md` 2.3 節に精読の記録がある．

---

## 3. 分野で標準とされていることと，未解決と書かれていること

### 3.1 標準とされていること（原典で見た記述と数値）

1. **話者独立で評価する．**TFN（5.4 節），LMF（表 1），MFN（表 1），DialogueRNN（Experimental Setting の節），MMIN（4.5 節）は話者独立の分割を明記する．MSP-Podcast の著者は，原文 "It is essential to ensure that data splits for train, validation, and test are speaker-independent"（IV-D 節）と書く．ただし例外がある（3.2 節の表）．
2. **言葉が最も強い単モダリティであることが多い．**MOSI の TFN（言葉 74.8，映像 66.8，音声 65.1．表 2），MELD（言葉 57.03，音声 41.79．表 11），MMGCN（IEMOCAP と MELD の両方で言葉が最高，映像が最低．表 3）．Poria 2017 は 2017 年までの文献で映像が音声より強いという報告が 90% 超と書いており，言葉を含めるかどうかで比べる相手が変わる．
3. **融合の得は最良の単モダリティに対して小さいことがある．**DialogueRNN（言葉 62.7 → 3 つ 62.9），MMGCN の MELD（言葉 57.72 → 3 つ 58.65），OGM-GE の CREMA-D（音声だけ 52.5 が素朴な融合の 51.7 を上回る），Wang 2020 の Kinetics（RGB 72.6 が素朴な融合 71.7 を上回る）．
4. **凍結した自己教師ありの特徴と軽い頭で比べる．**SUPERB が型を決め，WavLM の論文もこの型で比べる．層の重み付き和を使うのが SUPERB の方針である．
5. **IEMOCAP は 4 クラスに絞ることが多い．**ただし絞り方（happy と excited をまとめるか）は論文ごとに書き方が違い，SUPERB は書いていない．

### 3.2 評価の仕方の違い（データセットと論文ごと）

同じデータでも，感情の数，分割，指標が論文の系統で違う．数値を比べる前にこの表で条件をそろえる．

| データ（論文） | 感情とラベルの形 | 分割（話者を分けるか） | 指標 | 出典の場所 |
|---|---|---|---|---|
| IEMOCAP（SUPERB 2021，音声の自己教師あり） | 4 クラス（neutral, happy, sad, angry），単一ラベル | "five folds of the standard splits"．セッション単位かは本文に書かれていない | 正解率（WA に当たる） | 2 節 |
| IEMOCAP（MMIN 2021） | 4 クラス（Xu 2019，Liang 2020 に従う） | 10 分割，話者独立（4 セッションで学習，残り 1 セッションを話者で検証と評価に分ける） | WA と UA | 4.5 節 |
| IEMOCAP（LMF 2018，MulT 2019：CMU の系） | 4 感情をそれぞれ「ある／ない」の 2 値（MulT は複数ラベルと書く） | 話者独立（LMF 表 1：6,373/1,775/1,807 区間） | 感情ごとの 2 値の正解率と F1 | LMF 表 1・2，MulT 4.1 節・表 3 |
| IEMOCAP（MFN 2018） | 9 感情 | 話者独立 | MA(9)，F1 | 表 1・2 |
| IEMOCAP（DialogueRNN・DialogueGCN・MMGCN：会話） | 6 クラス（happy, sad, neutral, angry, excited, frustrated） | セッション 1〜4 を学習，5 を評価（5,810 と 1,623 発話） | 重み付き平均の正解率と F1 | DialogueRNN 表 1・2 |
| IEMOCAP（UniMSE 2022） | 感情のカテゴリ | 5,354/528/1,650 発話 | 正解率と重み付き F1 | 表 1・2 |
| MOSI（TFN 2017） | 感情価 −3〜+3 | 5 分割の交差検証，話者独立 | 2 値・5 クラスの正解率，F1，平均絶対誤差，相関 | 5.4 節 |
| MOSI・MOSEI（MAG-BERT，MISA，Self-MM，UniMSE） | 感情価 −3〜+3 | 公式の分割（MOSI 1,284/229/686） | 2 値の正解率を 2 通りの定義で併記（negative/non-negative と negative/positive） | MISA 4.2 節，MAG-BERT 5 節 |
| MOSEI の感情（Graph-MFN 2018） | 6 感情それぞれ（感情ごとに評価．0〜3 の強さの 2 値化の仕方は本文で確かめられなかった） | 本文で確かめられなかった | WA（Tong 2017 による．式は本文に無い）と F1 | 表 3 |
| MELD（Poria 2019） | 7 感情，単一ラベル | 会話の単位の公式の分割（9,989/1,109/2,610）．話者で分けたとは書いていない | 重み付き F1 | 表 8・11 |
| CREMA-D（OGM-GE 2022） | 6 感情 | 無作為に 6,698/744．話者を分けていない | 正解率，mAP | 4.1 節 |
| RAVDESS（Xue 2023．既存のノート） | 8 感情 | 画像と音声の組を無作為に分けている．話者で分けていない | 正解率 | `要点_問い1` 1 節 |
| AFEW（EmotiW 2013） | 7 表情 | 人物と映画で分けた Train/Val/Test | 正解率 | データの分割の節，表 6 |
| MSP-Podcast（Busso 2025） | 8 つの主な感情と 3 次元 | 話者独立を目指す．話者不明の発話は重なりうる | （論文の基準は未読） | V-A 節 |
| MER-DA | 4 クラス（Neutral, Happy, Sad, Angry） | 話者を分ける．1 コーパスをターゲットにする LOCO | UAR | `CLAUDE.md` |

**指標の名前の食い違い**：

- 音声の感情認識の慣習では，WA は全体の正解率，UA は感情ごとの再現率の平均である．MMIN はこの意味で使う．
- MOSEI の系の WA は Tong ほか 2017 の定義で，感情ごとの 2 値の評価に使う．名前が同じでも上の WA とは別の量である．
- Milner ほか 2019 は UA と WA の呼び方が通常と逆である（`docs/related-work.md` 2.3 節）．
- MOSI・MOSEI の 2 値の正解率は 2 通りの定義があり，「/」で併記する論文と，一方の定義の値だけを書く論文がある（MISA 4.2 節，MAG-BERT 表 1）．
- 報告値と再現値の差：Self-MM の論文が同じ条件で再現すると，MAG-BERT の MOSI の 2 値の正解率は 84.2/86.1 から 82.54/84.3 に，MISA は 81.8/83.4 から 80.79/82.1 になった（Self-MM 表 1）．論文の間の 1〜2 点の差は，手法の差ではなく実装と条件の差で生じうる．

### 3.3 著者たちが未解決と書いていること

| 論文 | 原文 | 平易な言い換え |
|---|---|---|
| Baltrušaitis 2019（6 節末） | "it is difficult to build models that exploit supplementary and not only complementary information" | 互いに補う情報だけでなく，同じ内容を重ねて持つ情報も使えるモデルを作るのは難しい |
| Baltrušaitis 2019（6 節末） | "each modality might exhibit different types and different levels of noise at different points in time" | 雑音の種類と大きさはモダリティごと，時刻ごとに違う |
| Liang 2024（8.3 節） | "How can we formally quantify the utility and risks of each input modality, while balancing these tradeoffs for reliable real-world usage?" | 各モダリティの役立ち方と危うさを数で表す方法は無い |
| Poria 2017（7 節） | "the amount of trust that should be placed in acted corpora is debatable" | 演技のコーパスをどこまで信頼してよいかは決着していない |
| Poria 2017（7 節） | "in the case of emotions, the number of emotional dimensions to use is unclear" | 感情を何種類で表すべきかは決まっていない |
| Wang 2020（抄録） | "different modalities overfit and generalize at different rates" | モダリティによって過学習と汎化の速さが違う |
| Peng 2022（5 節） | "even equipped with OGM-GE, the uni-modal performance in multimodal model still do not surpass the best uni-modal model" | 釣り合いを取っても，融合の中の各モダリティは単独で学んだモデルに届かない |
| MELD 2019（5 節） | "Identifying face of the speaker in a video where multiple other persons are present is very challenging." | 複数の人が映る動画で話者の顔を特定するのは難しい |
| DialogueGCN 2019（4.1 節） | "Multimodal emotion recognition is outside the scope of this paper, and is left as future work." | 会話の感情認識の手法を 3 モダリティで評価していない |
| UniMSE 2022（Limitations） | "the generation of universal labels only considers textual modality" | 異なるデータのラベルをそろえるときに言葉しか見ていない |
| CH-SIMS 2020（抄録） | "the unified annotations do not always reflect the independent sentiment of single modalities" | 全体のラベルは各モダリティ単独の感情と一致しない |
| MSP-Podcast 2025（V-A 節） | "some unknown speakers in the training set may overlap with those in the development or test partitions" | 話者の分からない発話が学習と評価の両方に入りうる |
| Albanie 2018（5.3 節．既存のノート） | "the degree to which there is a redundant emotional signal across modalities may differ across emotions" | モダリティ間で共通の感情の信号の量は感情によって違うかもしれない |

**このノートの範囲で見た限りのこと**（推測．網羅的な検索はしていない）：上の 38 本のうち，コーパスをまたいで学習と評価を分けたのは Schuller 2010 と Albanie 2018 だけで，複数のコーパスで学んだ UniMSE も評価は各コーパスの中で行っている．融合の手法の論文と釣り合いの論文は，コーパス間のずれのもとでの融合の得を扱っていない．

---

## 4. 出典の一覧

データセット

- Busso, C., Bulut, M., Lee, C.-C., Kazemzadeh, A., Mower, E., Kim, S., Chang, J. N., Lee, S., Narayanan, S. S. (2008). IEMOCAP: interactive emotional dyadic motion capture database. Language Resources and Evaluation 42(4):335–359. https://doi.org/10.1007/s10579-008-9076-6 ◎
- Zadeh, A., Zellers, R., Pincus, E., Morency, L.-P. (2016). Multimodal sentiment intensity analysis in videos: Facial gestures and verbal messages. IEEE Intelligent Systems 31(6):82–88. https://doi.org/10.1109/MIS.2016.94 ，arXiv 1606.06259 ◎
- Zadeh, A., Liang, P. P., Poria, S., Cambria, E., Morency, L.-P. (2018). Multimodal language analysis in the wild: CMU-MOSEI dataset and interpretable dynamic fusion graph. ACL 2018, pp. 2236–2246. https://aclanthology.org/P18-1208/ ◎
- Poria, S., Hazarika, D., Majumder, N., Naik, G., Cambria, E., Mihalcea, R. (2019). MELD: A multimodal multi-party dataset for emotion recognition in conversations. ACL 2019, pp. 527–536. https://aclanthology.org/P19-1050/ ，arXiv 1810.02508 ◎
- Dhall, A., Goecke, R., Lucey, S., Gedeon, T. (2012). Collecting large, richly annotated facial-expression databases from movies. IEEE MultiMedia 19(3):34–41. https://doi.org/10.1109/MMUL.2012.26 ○
- Dhall, A., Goecke, R., Joshi, J., Wagner, M., Gedeon, T. (2013). Emotion recognition in the wild challenge 2013. ICMI 2013, pp. 509–516. https://doi.org/10.1145/2522848.2531739 ◎
- Cao, H., Cooper, D. G., Keutmann, M. K., Gur, R. C., Nenkova, A., Verma, R. (2014). CREMA-D: Crowd-sourced emotional multimodal actors dataset. IEEE Transactions on Affective Computing 5(4):377–390. https://doi.org/10.1109/TAFFC.2014.2336244 ◎（道具経由）
- Livingstone, S. R., Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). PLoS ONE 13(5):e0196391. https://doi.org/10.1371/journal.pone.0196391 ◎
- Lotfian, R., Busso, C. (2019). Building naturalistic emotionally balanced speech corpus by retrieving emotional speech from existing podcast recordings. IEEE Transactions on Affective Computing 10(4):471–483. https://doi.org/10.1109/TAFFC.2017.2736999 ○
- Busso, C., Lotfian, R., Sridhar, K., Salman, A. N., Lin, W.-C., Goncalves, L., Parthasarathy, S., Naini, A. R., Leem, S.-G., Martinez-Lucas, L., Chou, H.-C., Mote, P. (2025). The MSP-Podcast corpus. arXiv 2509.09791. ◎
- Yu, W., Xu, H., Meng, F., Zhu, Y., Ma, Y., Wu, J., Zou, J., Yang, K. (2020). CH-SIMS: A Chinese multimodal sentiment analysis dataset with fine-grained annotation of modality. ACL 2020, pp. 3718–3727. https://aclanthology.org/2020.acl-main.343/ ◎

総説

- Baltrušaitis, T., Ahuja, C., Morency, L.-P. (2019). Multimodal machine learning: A survey and taxonomy. IEEE TPAMI 41(2):423–443. https://doi.org/10.1109/TPAMI.2018.2798607 ，arXiv 1705.09406 ◎
- Poria, S., Cambria, E., Bajpai, R., Hussain, A. (2017). A review of affective computing: From unimodal analysis to multimodal fusion. Information Fusion 37:98–125. https://doi.org/10.1016/j.inffus.2017.02.003 ◎（受理原稿）
- Liang, P. P., Zadeh, A., Morency, L.-P. (2024). Foundations & trends in multimodal machine learning: Principles, challenges, and open questions. ACM Computing Surveys 56(10):1–42. https://doi.org/10.1145/3656580 ，arXiv 2209.03430 ◎
- （載せなかったが書誌を確かめたもの）Soleymani, M., Garcia, D., Jou, B., Schuller, B., Chang, S.-F., Pantic, M. (2017). A survey of multimodal sentiment analysis. Image and Vision Computing 65:3–14. https://doi.org/10.1016/j.imavis.2017.08.003 △

融合の手法

- Zadeh, A., Chen, M., Poria, S., Cambria, E., Morency, L.-P. (2017). Tensor fusion network for multimodal sentiment analysis. EMNLP 2017, pp. 1103–1114. https://aclanthology.org/D17-1115/ ◎
- Liu, Z., Shen, Y., Lakshminarasimhan, V. B., Liang, P. P., Zadeh, A., Morency, L.-P. (2018). Efficient low-rank multimodal fusion with modality-specific factors. ACL 2018, pp. 2247–2256. https://aclanthology.org/P18-1209/ ◎
- Zadeh, A., Liang, P. P., Mazumder, N., Poria, S., Cambria, E., Morency, L.-P. (2018). Memory fusion network for multi-view sequential learning. AAAI 2018, 32(1). https://doi.org/10.1609/aaai.v32i1.12021 ◎
- Tsai, Y.-H. H., Bai, S., Liang, P. P., Kolter, J. Z., Morency, L.-P., Salakhutdinov, R. (2019). Multimodal transformer for unaligned multimodal language sequences. ACL 2019, pp. 6558–6569. https://aclanthology.org/P19-1656/ ◎
- Rahman, W., Hasan, M. K., Lee, S., Zadeh, A., Mao, C., Morency, L.-P., Hoque, E. (2020). Integrating multimodal information in large pretrained transformers. ACL 2020, pp. 2359–2369. https://aclanthology.org/2020.acl-main.214/ ◎
- Hazarika, D., Zimmermann, R., Poria, S. (2020). MISA: Modality-invariant and -specific representations for multimodal sentiment analysis. ACM MM 2020, pp. 1122–1131. https://doi.org/10.1145/3394171.3413678 ◎
- Yu, W., Xu, H., Yuan, Z., Wu, J. (2021). Learning modality-specific representations with self-supervised multi-task learning for multimodal sentiment analysis. AAAI 2021, 35(12):10790–10797. https://doi.org/10.1609/aaai.v35i12.17289 ◎
- Hu, G., Lin, T.-E., Zhao, Y., Lu, G., Wu, Y., Li, Y. (2022). UniMSE: Towards unified multimodal sentiment analysis and emotion recognition. EMNLP 2022, pp. 7837–7851. https://aclanthology.org/2022.emnlp-main.534/ ◎

会話の感情認識

- Majumder, N., Poria, S., Hazarika, D., Mihalcea, R., Gelbukh, A., Cambria, E. (2019). DialogueRNN: An attentive RNN for emotion detection in conversations. AAAI 2019, 33(01):6818–6825. https://doi.org/10.1609/aaai.v33i01.33016818 ◎
- Ghosal, D., Majumder, N., Poria, S., Chhaya, N., Gelbukh, A. (2019). DialogueGCN: A graph convolutional neural network for emotion recognition in conversation. EMNLP-IJCNLP 2019, pp. 154–164. https://aclanthology.org/D19-1015/ ◎
- Hu, J., Liu, Y., Zhao, J., Jin, Q. (2021). MMGCN: Multimodal fusion via deep graph convolution network for emotion recognition in conversation. ACL-IJCNLP 2021, pp. 5666–5675. https://aclanthology.org/2021.acl-long.440/ ◎

モダリティの釣り合い

- Wang, W., Tran, D., Feiszli, M. (2020). What makes training multi-modal classification networks hard? CVPR 2020, pp. 12692–12702. https://doi.org/10.1109/CVPR42600.2020.01271 ◎
- Peng, X., Wei, Y., Deng, A., Wang, D., Hu, D. (2022). Balanced multimodal learning via on-the-fly gradient modulation. CVPR 2022, pp. 8228–8237. https://doi.org/10.1109/CVPR52688.2022.00806 ◎
- Huang, Y., Lin, J., Zhou, C., Yang, H., Huang, L. (2022). Modality competition: What makes joint training of multi-modal network fail in deep learning? (Provably). ICML 2022, PMLR 162. https://proceedings.mlr.press/v162/huang22e.html ◎（証明は未読）
- Du, C., Teng, J., Li, T., Liu, Y., Yuan, T., Wang, Y., Yuan, Y., Zhao, H. (2023). On uni-modal feature learning in supervised multi-modal learning. ICML 2023, PMLR 202. https://proceedings.mlr.press/v202/du23e.html ◎

欠けたモダリティ

- Zhao, J., Li, R., Jin, Q. (2021). Missing modality imagination network for emotion recognition with uncertain missing modalities. ACL-IJCNLP 2021, pp. 2608–2618. https://aclanthology.org/2021.acl-long.203/ ◎

自己教師ありの音声エンコーダ

- Baevski, A., Zhou, H., Mohamed, A., Auli, M. (2020). wav2vec 2.0: A framework for self-supervised learning of speech representations. NeurIPS 2020. https://proceedings.neurips.cc/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html ◎
- Hsu, W.-N., Bolte, B., Tsai, Y.-H. H., Lakhotia, K., Salakhutdinov, R., Mohamed, A. (2021). HuBERT: Self-supervised speech representation learning by masked prediction of hidden units. IEEE/ACM TASLP 29:3451–3460. https://doi.org/10.1109/TASLP.2021.3122291 ◎
- Chen, S., Wang, C., Chen, Z., Wu, Y., Liu, S., Chen, Z., Li, J., Kanda, N., Yoshioka, T., Xiao, X., Wu, J., Zhou, L., Ren, S., Qian, Y., Qian, Y., Wu, J., Zeng, M., Yu, X., Wei, F. (2022). WavLM: Large-scale self-supervised pre-training for full stack speech processing. IEEE JSTSP 16(6):1505–1518. https://doi.org/10.1109/JSTSP.2022.3188113 ◎
- Yang, S.-w., Chi, P.-H., Chuang, Y.-S., ほか (2021). SUPERB: Speech processing Universal PERformance Benchmark. Interspeech 2021, pp. 1194–1198. https://doi.org/10.21437/Interspeech.2021-1775 ◎

顔と言葉のエンコーダ

- Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., Stoyanov, V. (2019). RoBERTa: A robustly optimized BERT pretraining approach. arXiv 1907.11692. ○
- He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep residual learning for image recognition. CVPR 2016, pp. 770–778. https://doi.org/10.1109/CVPR.2016.90 ○

感情の表し方

- Ekman, P. (1992). An argument for basic emotions. Cognition and Emotion 6(3–4):169–200. https://doi.org/10.1080/02699939208411068 ○
- Russell, J. A. (1980). A circumplex model of affect. Journal of Personality and Social Psychology 39(6):1161–1178. https://doi.org/10.1037/h0077714 ○

コーパス間の感情認識

- Schuller, B., Vlasenko, B., Eyben, F., Wöllmer, M., Stuhlsatz, A., Wendemuth, A., Rigoll, G. (2010). Cross-corpus acoustic emotion recognition: Variances and strategies. IEEE Transactions on Affective Computing 1(2):119–131. https://doi.org/10.1109/T-AFFC.2010.8 ○（既存のノートは同内容の拡張要旨を精読）
- Albanie, S., Nagrani, A., Vedaldi, A., Zisserman, A. (2018). Emotion recognition in speech using cross-modal transfer in the wild. ACM MM 2018, pp. 292–301. https://doi.org/10.1145/3240508.3240578 ◎（既存のノート）

### 確かめられなかったこと・載せなかった論文

- Dhall 2012（AFEW の原論文）の本文は，機関の公開版が 2037 年まで閲覧できない．AFEW の数値は EmotiW 2013 から書いた．
- Lotfian & Busso 2019 の本文は取得できず，2025 年の最終版の論文で代えた．
- AFEW の EmotiW 2017 の分割（773/383/653）は参加者の論文の記述だけで確かめた（△）．MER-DA の発話一覧がこの版から作られたかは確かめていない．
- MOSEI の感情の評価に使う WA の式（Tong ほか 2017）は読んでいない．
- 候補のうち載せなかったもの：Han 2021（MMIM），Fan 2023（PMR），Ma 2021（SMIL．AAAI 2021 であることだけ arXiv の頁で確かめた），Soleymani 2017（書誌のみ）．合計の本数を抑えるために外した．
