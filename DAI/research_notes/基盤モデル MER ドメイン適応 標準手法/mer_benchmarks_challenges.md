# 基盤モデルの特徴を使う MER のベンチマークと競技会：コーパス間・ラベル無しデータの使い方と標準手法（2026-11-08 調査）

読み方の注意：
- 「本文」は原論文の本文（arXiv の HTML 版か PDF）を読んだもの．「要旨のみ」「検索結果の抜粋のみ」はそう書く．
- 「前回の精読」は，このリポジトリの `docs/related-work.md` 2.3 節に，以前のセッションで原論文を読んで書いた記録を使ったもの．今回は原文を読み直していない．
- WAF は weighted average F-score（クラスごとの F1 をクラスの標本数で重み付けした平均）．UAR はクラスごとの再現率の単純平均．
- 本研究の設定（比較用）：凍結したエンコーダ（音声 WavLM，顔 ResNet-50，言葉 RoBERTa），モダリティごとの線形ソフトマックス分類器と融合した分類器，ターゲットはクラスあたり K 件のラベルとラベル無しの発話，ソースは 2〜3 個の別コーパス（AFEW・IEMOCAP・MOSEI・MELD），4 クラス（neutral・happy・sad・angry），指標は UAR．

---

## MERBench（Lian ほか 2024）：比べたエンコーダ，コーパス間の評価，凍結と微調整・融合についての結論

### Takeaway
MERBench は，多数の事前学習エンコーダから凍結のまま発話単位の特徴を取り出し，浅い分類器と注意による融合で比べる型を標準として定めた．コーパス間の評価は，4 クラス（喜・悲・怒・中立）に揃えたコーパスの組と感情価のコーパスの組で「ソースで学習し，ターゲットで試す」形で行い，融合の上積みはコーパス間ではコーパス内より小さいこと，微調整は事前学習ほどの得にならないことを結論にしている．

### Cited Findings
- **書誌**：arXiv 2401.03429（v3 は 2024-04-21）．本文に掲載先の記載は無い — [MERBench arXiv HTML](https://arxiv.org/html/2401.03429)．MERTools の README は IEEE Xplore の文書 11347465 に結びつけている — [MERTools](https://github.com/zeroQiaoba/MERTools)．このリポジトリの記録では IEEE TPAMI 48(5), 2026 の掲載（`docs/related-work.md` 1.13 節の訂正）．
- **比べた視覚エンコーダ（本文）**：ResNet-MSCeleb，ResNet-ImageNet，VideoMAE-base/large，EmoNet，SENet-FER2013，ResNet-FER2013，MANet-RAFDB，DINOv2-large，CLIP-base，CLIP-large，EVA-02-base．顔は OpenFace で切り出して位置を揃え，フレームか区間ごとの特徴を平均して動画 1 本の特徴にする — [MERBench](https://arxiv.org/html/2401.03429)
- **比べた音声エンコーダ（本文）**：eGeMAPS，IS09，VGGish，data2vec-base，wav2vec-large，wav2vec 2.0-base/large，WavLM-base/large，Whisper-base/large，HuBERT-base/large．音声は 16 kHz モノラルに変換 — [MERBench](https://arxiv.org/html/2401.03429)
- **比べた言葉のエンコーダ（本文）**：OPT-13B，ALBERT-small，Llama-13B，Vicuna-13B，XLNet-base，StableLM-7B，Sentence-BERT，DeBERTa-large，Llama2-13B，PERT-base，MOSS-7B，ELECTRA-base，Falcon-7B，LERT-base，ChatGLM2-6B，MacBERT-base，RoBERTa-large，BLOOM-7B，Baichuan-13B．MER2023 の字幕は WeNet の音声認識と EasyOCR の結果を人手で統合した — [MERBench](https://arxiv.org/html/2401.03429)
- **チェックポイント（付録の表 XVIII）**：chinese-hubert-large と chinese-wav2vec2-large は TencentGameMate，WavLM-large は microsoft/wavlm-large，Whisper-large は openai/whisper-large-v2，CLIP-large は openai/clip-vit-large-patch14，DINOv2-large は facebook/dinov2-large，Baichuan-13B は baichuan-inc/Baichuan-13B-Base — [MERBench](https://arxiv.org/html/2401.03429)
- **特徴の取り出し方（公開コード）**：音声の特徴抽出の台本は `layer_ids = [-4, -3, -2, -1]` で最後の 4 層の隠れ状態を足し合わせ（平均ではなく和），発話単位（UTTERANCE）では時間方向に平均する．`model.eval()` と `torch.no_grad()` の下で推論だけを行い，エンコーダは更新しない．Whisper は最終層だけを使う — [extract_audio_huggingface.py](https://raw.githubusercontent.com/zeroQiaoba/MERTools/master/MERBench/feature_extraction/audio/extract_audio_huggingface.py)．README は `--feature_level` に UTTERANCE と FRAME を選べると書き，基準の命令は `--model='attention'` を渡す — [MERTools/MERBench](https://github.com/zeroQiaoba/MERTools/tree/master/MERBench)
- **比べた融合（本文）**：発話単位の融合は TFN，LMF，MISA，MMIM，注意（Attention）．系列単位の融合は MFN，GMFN，MCTN，MFM，MulT．著者は「注意の仕組みが全体で最良」と書き，注意を既定の融合にした．中程度の特徴の組での平均は Attention 74.91，TFN 74.82，MulT 74.59，MFN 74.37．高性能の特徴の組では Attention 78.12，TFN 77.69，MulT 77.87 — [MERBench](https://arxiv.org/html/2401.03429)
- **特徴を増やす効果（表 X）**：モダリティごとに上位 k 個の特徴を足すと，平均は Top1 78.12 から Top6 79.92 に上がる．著者は，成績の悪い特徴を足すと下がる場合もあると書いている — [MERBench](https://arxiv.org/html/2401.03429)
- **評価の手順（本文）**：離散感情は WAF，感情価は MSE．MER2023 の Train&Val では 5 分割交差検証で，6 回の平均．IEMOCAP は公式の分割が無いので leave-one-session-out の 5 分割．ほかの公開コーパスは既定の分割 — [MERBench](https://arxiv.org/html/2401.03429)．話者独立の分割を明記しているのは IEMOCAP の leave-one-session-out だけである（同）．
- **コーパス間の設定（本文）**：IEMOCAP(4 クラス)，MELD，MER-MULTI を共通の 4 ラベル（happiness, sadness, anger, neutral）に揃え，CMU-MOSI・CMU-MOSEI・CH-SIMS・CH-SIMS v2 は感情価の連続値を正規化して使う．ソースで学習し，ターゲットで WAF を測る — [MERBench](https://arxiv.org/html/2401.03429)．ターゲットのラベルは使わない（前回の精読．`docs/related-work.md` 2.3 節）．
- **コーパス間の結果（表 XI．単モダリティの最良 → マルチモーダルの最良，WAF）**：MER-MULTI→IEMOCAP(4) 50.69→59.58（+8.89），MELD→IEMOCAP(4) 46.62→48.86（+2.24），CMU-MOSEI→CH-SIMS 75.57→81.64（+6.07），CMU-MOSI→CH-SIMS 75.03→76.55（+1.52），CH-SIMS→CMU-MOSI 70.15→70.86（+0.71），CH-SIMS v2→CMU-MOSI 74.22→73.85（−0.37） — [MERBench](https://arxiv.org/html/2401.03429)
- **コーパス間についての著者の結論**：「コーパス内で良い特徴は概ねコーパス間でも良い結果になる」．融合はどちらの設定でも役立つが，コーパス間では効用が比較的小さく，融合がターゲットのデータセットに過学習させるためかもしれない，と書く．GPT-4V（表 XII）は MER-MULTI 63.50，CMU-MOSI 80.43，CH-SIMS 81.24 で，一部ではコーパス間の結果に近いか上回る．コーパス間を本当に解くには課題をまたぐ知識の統合と，説明できる推論でラベルの一貫性を高めることが要る，と結論する — [MERBench](https://arxiv.org/html/2401.03429)
- **ドメインの近いデータでの事前学習（表 VI）**：VideoMAE-base を MER-MULTI で試すと，Kinetics-400 で事前学習したもの 46.72，MER-SEMI のラベル無し動画で事前学習したもの 60.86，VoxCeleb2 で事前学習したもの 61.76．CH-SIMS では 68.95，81.37，78.33．VideoMAE-large の MER-MULTI では 53.26，59.89，58.35．著者は，良い視覚エンコーダは弱教師ありか自己教師ありで，人の映った大量の動画で学ぶべきだと書く — [MERBench](https://arxiv.org/html/2401.03429)
- **凍結と微調整（V-G 節，表 XVI・XVII）**：VideoMAE-base，CLIP-base，data2vec-base，HuBERT-base，ALBERT-small，MacBERT-base について，凍結した特徴＋浅い分類器と，エンコーダごと微調整した場合を比べた．例：VideoMAE-base の MER-MULTI 46.72→51.20（+4.48），CH-SIMS 68.95→75.38（+6.43），CLIP-base の MER-MULTI 60.12→57.07（−3.05），data2vec-base の CH-SIMS v2 59.87→66.93（+7.06），MacBERT-base はほとんど 1 点未満の変化．学習するパラメータは VideoMAE-base で 0.48M から 65.11M に増え，標本 1 つあたりの学習時間は 0.03 ms から 49.83 ms に増える — [MERBench](https://arxiv.org/html/2401.03429)
- **凍結と微調整についての著者の結論**：微調整の得失は特徴とデータセットの組で変わり，エンコーダがデータセットに合わないときは役立つが，感情の小さいデータセットでは過学習する．微調整は計算が高く，得は事前学習より小さいので，標本の少ない感情認識では微調整より事前学習（特に下流の課題と相性のよいデータでの事前学習）に力を入れるべきだ，と書く — [MERBench](https://arxiv.org/html/2401.03429)
- **そのほかの結論**：言葉のエンコーダは言語に敏感で，翻訳すると感情の情報が失われやすい．音声のエンコーダも言語の情報を暗に含むので，表情豊かな多言語の音声で学ぶことを勧める．データセットによっては感情が主に 1 つのモダリティで伝わるので，融合の研究に向かない — [MERBench](https://arxiv.org/html/2401.03429)

### Inferences
- MERBench が定めた型は「凍結した事前学習エンコーダ → 最後の数層の隠れ状態を足して時間平均した発話単位の特徴 → モダリティごとの全結合の写像 → 注意の重みで足す融合 → 交差エントロピー」である．本研究の「凍結特徴＋線形分類器」はこの型から，全結合の隠れ層と注意の重みを取り除いたものに当たる．
- 本研究との違い：MERBench のコーパス間はソース 1 つ・ターゲットのラベル無しで，指標は WAF である．本研究はソース 2〜3 個を混ぜ，ターゲットに K 件のラベルを使い，指標は UAR である．MERBench は AFEW と MOSEI の 4 クラス版をコーパス間に使っていない（MOSEI は感情価の組だけ）．
- 「融合の上積みはコーパス間で縮む」は，MER-MULTI→IEMOCAP の +8.89 から CH-SIMS v2→CMU-MOSI の −0.37 まで組によって大きく違い，平均的な主張として読むべきである．

### Gaps
- コーパス間でのモデルの選び方（ターゲットのデータで早期終了や超パラメータを選んだか）は，今回読んだ範囲の本文に書かれていなかった．
- コーパス間の全組の表（付録の表 XIX・XX）の個々の値は今回書き写していない．
- 映像と言葉の特徴抽出の台本でどの層を使うかは確かめていない（音声の台本だけ読んだ）．

---

## MER2023・MER2024・MER2025（と MER2026）：基準の手法と上位の解法は，ラベル無しデータとエンコーダをどう使ったか

### Takeaway
3 回の競技会の基準の手法は，どれも凍結した事前学習エンコーダ（音声は中国語の HuBERT-large，映像は CLIP-large か表情で学んだ ResNet/MANet，言葉は Baichuan-13B か MacBERT/RoBERTa）の特徴を注意で融合するだけで，ラベル無しデータを使わない．上位の解法は同じ特徴の上で，(1) 確信度の高い疑似ラベルを足す自己学習（閾値 0.9，クラス別の閾値 0.99/0.85，クラスごとの上位 k 件，2 つのモデルか MLLM との一致で絞る），(2) ラベル無しの動画での自己教師ありの事前学習や音声と映像の対照学習による位置合わせ，(3) GPT-4・Gemini などの出力を言葉の入力に足すこと，(4) アンサンブル，を組み合わせる．融合した系で疑似ラベルが上げた幅は，報告のある例で WAF +1〜+2 点である．どの回も 1 つのコーパスの中の半教師あり学習で，コーパス間の移転は扱っていない．

### Cited Findings

**MER2023（ACM MM 2023．Lian ほか．本文：arXiv v1 の PDF と v2 の HTML）**
- 3 部門：MER-MULTI（離散感情と感情価），MER-NOISE（テスト動画に雑音を加える），MER-SEMI（大量のラベル無し標本で半教師あり学習） — [MER2023 arXiv](https://arxiv.org/abs/2304.08981)
- 規模（表 1）：Train&Val 3,373 件，MER-MULTI 411 件，MER-NOISE 412 件，MER-SEMI はラベル付き 834 件とラベル無し 73,148 件．離散感情は neutral, anger, happiness, sadness, worry, surprise の 6 つ．指標は離散感情が WAF，感情価が MSE，合成の指標は metric_e − 0.25 × metric_v — [MER2023 v2 HTML](https://arxiv.org/html/2304.08981v2)
- **MER-SEMI のテスト標本は，事前学習した感情分類器の最大ソフトマックス確率で選んだ確信度の高い標本だけに注釈した**．クラスの偏りがあるので，固定の閾値ではなく，クラスごとに確信度の上位 η の割合を選ぶ．著者は MER-SEMI の成績が他の 2 部門より高い理由を「確信度で選んだので感情がはっきりした標本になり，分類器が当てやすい」と書く — [MER2023 v1 PDF](https://arxiv.org/pdf/2304.08981v1)
- 基準の特徴（v1 の 3 節）：音声は openSMILE の IS09・IS10・eGeMAPS，VGGish，wav2vec，wav2vec 2.0，HuBERT（TencentGameMate の中国語版）．言葉は WeNet で書き起こした文に BERT，RoBERTa，XLNet，GPT-2，DeBERTa，MacBERT，ALBERT，ELECTRA．映像は OpenFace で顔を切り出して揃え，ResNet・SENet・MANet を ImageNet・MS-Celeb-1M・FER-2013・RAF-DB で学んだモデルのフレーム特徴を平均する — [MER2023 v1 PDF](https://arxiv.org/pdf/2304.08981v1)
- 基準のモデル（v1 の 3.7 節）：モダリティごとに ReLU つきの全結合で隠れ表現を作り，3 つを並べてソフトマックスの注意の重みで足す（z = hα）．離散感情の交差エントロピーと感情価の MSE を同時に学ぶ．隠れ次元は {128, 256}，学習率は {1e-3, 1e-4, 1e-5}，Adam，最大 100 エポック，dropout 0.3，Train&Val の 5 分割交差検証で超パラメータを選び，3 回の平均を報告 — [MER2023 v1 PDF](https://arxiv.org/pdf/2304.08981v1)
- 基準で融合に選んだ特徴：音声は HuBERT-large と HuBERT-base，言葉は MacBERT-large・RoBERTa-large・MacBERT-base，映像は ResNet-FER2013 と MANet-RAFDB．著者は「融合は常に単モダリティより良い」と書く．HuBERT-large の MER-SEMI の単モダリティ WAF は 85.11 — [MER2023 v1 PDF](https://arxiv.org/pdf/2304.08981v1)
- 基準の成績：MER-MULTI の合成 0.56（v1 要旨では F1 77.57%，MSE 0.82），MER-NOISE 0.41（F1 69.82%，MSE 1.12），MER-SEMI の WAF は本文で 86.40%，表 4 で 0.8675，v1 の要旨で 86.75% と記述が食い違う．基準はラベル無しデータを使わない — [MER2023 v1 PDF](https://arxiv.org/pdf/2304.08981v1)；[MER2023 v2 HTML](https://arxiv.org/html/2304.08981v2)
- 76 チームが登録．sense-dl-lab が 3 部門すべてで 1 位（MER-MULTI 0.7005，MER-NOISE 0.6846，MER-SEMI 0.8911）．概要論文は sense-dl-lab の手法を書いていない — [MER2023 v2 HTML](https://arxiv.org/html/2304.08981v2)
- Expression MAE の解法（MER-SEMI 2 位，F1 88.55%）：ラベル付きが少ないので，分類器が高い確信度で 1 つのクラスに当てたラベル無し動画に疑似ラベルを付けた．**検索結果に出た ACM DL の要旨の抜粋だけを読んだ**（本文は取得できず）．チームの資料には HuBERT-large と，偏りのあるクラスだけを閾値で疑似ラベルとして足す手順があるという検索結果の要約があったが，これも原文を確かめていない — [AMiner の記録](https://aminer.org/pub/6544b60a939a5f4082e056f7/semi-supervised-multimodal-emotion-recognition-with-expression-mae)
- Chen ほか（MM 2023．class-balanced pseudo-labeling）：クラスごとに確信度の上位 k 件の疑似ラベルを選ぶ．単モダリティと融合の分類器の予測が一致した疑似ラベルは確証バイアスが小さいとする．MER2023-SEMI のテストで重み付き F1 88.53%．**要旨のみ**（k の値は確かめられなかった） — [NWPU の記録](https://pure.nwpu.edu.cn/en/publications/semi-supervised-multimodal-emotion-recognition-with-class-balance/)

**MER2024（ACM MM 2024 の MRAC ワークショップ．Lian ほか．本文：arXiv HTML）**
- 3 部門：MER-SEMI（ラベル付き 1,169 件），MER-NOISE（1,170 件．音声に MUSAN の音声雑音を SNR 5〜10 dB で足し，画像は縮小率 r ∈ {1, 2, 4} でぼかす），MER-OV（開いた語彙．表 1 は 322 件，本文は 332 件で食い違う）．Train&Val は MER2023 を合わせた 5,030 件．全体は 115,595 件で，参加者は全部を予測し，その一部だけが採点される．MER-SEMI と MER-NOISE のラベルは worried, happy, neutral, angry, surprised, sad で，順位は WAF — [MER2024 arXiv HTML](https://arxiv.org/html/2404.17113)
- 基準のエンコーダ（表 2）：映像は VideoMAE-base/large（Kinetics-400），CLIP-base/large，EVA-02-base，DINOv2-large，VoxCeleb2 か MER2023 で事前学習した VideoMAE-base．音声は emotion2vec，Whisper-base/large，wav2vec 2.0-base/large（WenetSpeech），chinese-hubert-base/large，eGeMAPS，VGGish．言葉は PERT，LERT，XLNet，MacBERT，RoBERTa，ELECTRA，ChatGLM2-6B，MOSS-7B，BLOOM-7B，Baichuan-13B — [MER2024](https://arxiv.org/html/2404.17113)
- 基準の融合と学習：各モダリティを h_m = ReLU(f_m W_m + b_m) で写し，3 つを並べてソフトマックスの注意の重みで足す．d ∈ {64, 128, 256}，学習率 ∈ {1e-3, 1e-4}，dropout ∈ {0.2, 0.3, 0.4, 0.5}，Adam，50 通りの超パラメータを無作為に試して 5 分割交差検証で選ぶ — [MER2024](https://arxiv.org/html/2404.17113)
- 基準はラベル無しデータで学習しない（疑似ラベルの記述も無い）．ただし表で，ラベル無しの同じ領域の動画での事前学習の効果を示している：VideoMAE-base の平均 WAF は元の事前学習 49.59 → VoxCeleb2 59.50 → MER2023 60.05．自己教師ありの CLIP-large 63.32 は教師ありの SENet-FER2013 54.87 より高く，HuBERT-large 76.55 は eGeMAPS 32.96 より高い — [MER2024](https://arxiv.org/html/2404.17113)
- 基準の成績（WAF．Train&Val / MER-SEMI / MER-NOISE / 平均）：映像の最良 CLIP-large 66.66 / 63.27 / 60.04 / 63.32，音声の最良 HuBERT-large 73.02 / 83.42 / 73.21 / 76.55，言葉の最良 Baichuan-13B 54.86 / 56.63 / 52.01 / 54.50．3 モダリティの融合は平均 WAF の最良が Top6 の 81.97，MER-SEMI の最良が Top1 の 86.73 — [MER2024](https://arxiv.org/html/2404.17113)
- 採録された MER-SEMI の論文は 5 本（Qi ほか，Ge ほか，Shi & Gao，Zhao ほか，Fan ほか）．順位表は CodaLab にあり，ウェブサイトには載っていない — [MER2024 ウェブサイト](https://zeroqiaoba.github.io/MER2024-website/)

**MER2024-SEMI の上位の解法（どれも本文を読んだ．順位は各論文の自己申告）**
- **Qi ほか（1 位を自己申告，テスト 90.15%）**：音声は Chinese-HuBERT-large の最後の 4 層を足したフレーム特徴．映像は OpenFace で揃えた顔の CLIP 特徴と，生のフレームを入れる EmoVCLIP．EmoVCLIP は CLIP の重みを凍結し，画像と言葉のエンコーダの最初の 12 層に学習できる接頭のプロンプトを 4 個ずつ足すプロンプト学習で，時間平均したフレームの埋め込みを 6 つの感情ラベルの文の埋め込みと比べる．言葉は Baichuan-13B-Base に，GPT-4 が 6 感情を確からしい順に並べた出力を連結して入れる．融合は連結と基準の注意．モダリティの dropout（確率 0.3 で埋め込みを 0 にする）．ラベル無しデータは Chen ほか 2023 に従う自己学習で，各回でクラスごとに上位 k=10 件の疑似ラベルを足し，最大 Nr=10 回繰り返す（確信度の固定の閾値は書かれていない）．基準 86.32 に対し，自己学習つきで EmoVCLIP なし 88.68，EmoVCLIP ありで dropout 0 が 89.52，0.15 が 89.87，0.3 が 90.15，0.5 が 89.08．自己学習を外した行は無い．指標の名前は要旨で正解率，結論で weighted F1，実験の節で WAF と揺れる — [Qi ほか 2024](https://arxiv.org/html/2409.07078)
- **Shi & Gao（3 位，89.83%）**：音声 HuBERT-large，映像 CLIP-ViT-large を 5,030 件のラベル付きで微調整（フレームの埋め込みを「the person is <感情>」の文の埋め込みに合わせる交差エントロピー），言葉は GPT-4 が付けたラベルを文に連結して Baichuan-13B に入れる．融合は HuBERT の特徴が映像と言葉を導く Audio-Guided Transformer と，ほかとの類似度が低いモダリティを 0 にする適応融合．疑似ラベルはソフトマックスの確信度 p > 0.9 で，基準の HuBERT-large と提案モデルの両方で確信度の高い予測の共通部分を足し，繰り返し学習する（回数と足した件数は書かれていない）．表 2：基準 87.61 → 疑似ラベル 88.64 → ＋投票 89.02，提案モデル 88.28 → 89.26 → 89.83 — [Shi & Gao 2024](https://arxiv.org/html/2409.05007v1)
- **Zhao ほか（4 位，88.90%）**：音声は Chinese-HuBERT-large の 16〜21 層にボトルネックのアダプタ（128 次元に落として ReLU，戻す）を入れる部分的な微調整で，層の重みは 18 層を 1.0，ほかを 0.0 で初期化して学習する．損失は交差エントロピーと，特徴を隠して復元する MSE の和で，Train&Val で学習する．映像は CLIP-large の顔特徴を平均し，ラベル無しの映像と音声の組で対称な対照損失を使って，映像の MLP だけを音声の空間に合わせる（CLIP と微調整済み HuBERT は凍結）．言葉は Train&Val の書き起こしとラベル無しデータの字幕から作る Baichuan2 の特徴．融合は基準の注意．疑似ラベルは使わない．表 2：HuBERT-large 83.49 → アダプタつき 84.88，CLIP-large 60.95 → 対照学習で合わせて 64.59（+3.64），HuBERT＋CLIP 86.78 → 合わせた CLIP で 87.01，3 モダリティで 88.90 — [Zhao ほか 2024](https://arxiv.org/html/2409.05015v2)
- **Fan ほか（6 位，88.25%）**：特徴は CLIP-Large，Chinese-HuBERT-Large，Baichuan2-13B-Base．その上にモダリティ固有の Transformer のエンコーダと，モダリティに共通な線形のエンコーダを置き，「使えるラベル無しデータ全部」でモダリティ内（雑音を足した版との対比）とモダリティ間（1 対 1，1 対残り 2 つの連結）の対照学習で事前学習する．自己学習は 2 段で，初期モデルと基準モデルの予測の共通部分を，クラス別の確信度の閾値（happy・neutral・angry・sad は 0.99，worried と surprise は 0.85）で絞り，計 1,398 件（neutral 170，angry 122，happy 206，sad 429，worried 321，surprise 150）を足し，学習率を 1e-4 から 5e-5 に下げて学び直す．少数のクラスを無作為に重複させて 850 件（2 段目は 1,000 件）に揃える．最後は単モダリティ 3 つと融合の分類器の重み付きソフト投票（A 0.7，V 0.5，T 0.4，融合 0.7）．アブレーション（MER-SEMI のテスト WAF）：全部 88.25，疑似ラベルなし 86.35，対照の事前学習なし 85.86，重複なし 86.07，基準 86.73．表 1 と表 2 の値が一致しない箇所がある — [Fan ほか 2024](https://arxiv.org/html/2409.04447v1)
- **Cheng ほか（SZTU-CMU．MER-NOISE 1 位，WAF 85.30%）**：Emotion-LLaMA でラベル無しの 20,000 件に疑似ラベルを付け，5,030 件の Train&Val に足した．確信度での絞り込みは書かれていない．特徴抽出のエンコーダは凍結（図 2 の説明）で，音声は Chinese-HuBERT，映像は CLIP-large，言葉はプロンプトを付けた Qwen1.5-32B と Baichuan-13B．VideoMAE は MER2024 のラベル無し標本で事前学習した．融合は注意と畳み込みの枝を足す Conv-Attention．疑似ラベルを使う割合を 20% から 100% にすると MER-NOISE の WAF は 79.88 から 83.87．疑似ラベルを使わない行は無い — [Cheng ほか 2024](https://arxiv.org/html/2408.10500)

**MER2025（ACM MM 2025 の MRAC25．Lian ほか．本文：arXiv HTML）**
- 4 部門：MER-SEMI（ラベル付き Train&Val 7,369 件，ラベル無し 124,802 件，テストはラベル無しの中から人手で注釈した 2,026 件．ラベルは neutral, happy, angry, sad, surprise, worry．順位は WAF），MER-FG（細かい感情．OV-MERD 332 件と MER-Caption+ 31,327 件で学び，テスト 1,200 件），MER-DES（説明文を人手で採点），MER-PR（MDPE の 3,728 件で Big Five の RMSE） — [MER2025 arXiv HTML](https://arxiv.org/html/2504.19423v1)
- MER-SEMI の基準：映像は ResNet-FER2013，DINOv2-large，SENet-FER2013，MANet-RAFDB，CLIP-base/large，音声は WavLM-base，wav2vec 2.0-base/large，Whisper-large，HuBERT-base/large，言葉は MacBERT-base/large，BLOOM-7B，RoBERTa-base/large．融合は MERBench に従う単純な注意で，著者はもっと複雑な融合は過学習しうると書く．50 回の超パラメータ探索の後に最良の設定で 6 回．基準はラベル無しデータを使わない．テストの WAF / ACC は単モダリティ最良の HuBERT-large 72.27 / 72.90，音声＋映像の Top1 77.10 / 77.54，3 モダリティの Top1 78.63 / 78.77 — [MER2025](https://arxiv.org/html/2504.19423v1)
- 概要論文は，ラベル無しデータがラベル付きデータと同じ領域から来ること，Train&Val の 5 分割交差検証の値がテストの値と強く相関することを書いている．MER2025 の参加チームの結果は載っていない（過去の参加数は MER2023 で 76，MER2024 で 94） — [MER2025](https://arxiv.org/html/2504.19423v1)；[MER2025 ウェブサイト](https://zeroqiaoba.github.io/MER2025-website/)

**MER2025-SEMI の解法（本文を読んだ）**
- **Zhu ほか（2 位を自己申告，F1 0.8772）**：映像は CLIP，FG-CLIP，InternVL（全体のフレームと顔の切り出しの 2 系統），音声は中国語 HuBERT・中国語 wav2vec2・Qwen2-Audio，言葉は Baichuan2・中国語 RoBERTa・BLOOM．さらに Gemini-2.5-Flash に音声つきの動画を見せて，推論の文・感情ごとの確信度・ラベル・頼ったモダリティを JSON で出させ，それを Baichuan2-7B で符号化して新しい入力の枝にする．表情の動作単位（AU）35 次元の時系列も足す．融合は枝ごとの専門家をルーターで重み付けする MoE．エンコーダは凍結と読める記述がある．ラベル無しデータは，ラベル付きで学んだ基準モデルと Gemini の予測が一致した標本だけに疑似ラベルを付け（数値の閾値は無い），約 20,000 件で事前学習してからラベル付きデータで微調整する．表 3：6 特徴で事前学習なし 0.8430 → 事前学習の印つき 0.8547 → 投票も 0.8655 → 並べ替えも 0.8772（表の印と構成要素の対応が読み取りにくい） — [Zhu ほか 2025](https://arxiv.org/html/2508.06036v1)
- **ECMF（順位の記載なし，WAF 87.49%）**：音声は HuBERT-Large の 16〜21 層，言葉は GPT-4 のキーワードと疑似ラベル・Qwen-Omni の手がかりを足した文を Chinese-RoBERTa-wwm-ext-large で，映像は全体のフレームと OpenFace の顔の CLIP-ViT-Large．融合はモダリティごとのトークンと 2 層の自己注意．訓練データのラベルを，モダリティごとの弱い分類器と Qwen-Omni の多数決で見直し，すべての情報源が元のラベルと食い違う標本を人手で確かめた．ラベル無しの 20,000 件に触れているが使い方は書かれていない．アブレーション（テスト WAF）：自分で再実行した基準 76.80 → ラベルの見直し 78.67 → … → 正規化 84.40 → RoBERTa 85.30 → GPT-4 のラベル 86.08 → … → HuBERT の層の選択 87.14 → アンサンブル 87.49．比較に引く公式の基準は 78.63 — [ECMF 2025](https://arxiv.org/html/2508.05991v1)

**MER2026（arXiv 2604.19417．本文）**
- MER-SEMI は無くなり，MER-Cross（話し手ではなく対話相手の感情．学習 9,395 件，テスト 574 件，6 感情，WAF），MER-FG，MER-Prefer（感情の説明の好みの 2 択），MER-PS（脳波と fNIRS）の 4 部門になった．MER-Cross の「Cross」は人をまたぐ意味で，コーパス間ではない．MER-Cross の基準は融合の Top-1 が 57.44，Top-2 が 55.24 で，映像の単独の最良 CLIP-large 58.88 を下回った．概要論文はコーパス間やドメインのずれを論じていない — [MER2026 arXiv HTML](https://arxiv.org/html/2604.19417)

### Inferences
- 確かめた MER2024・MER2025 の上位の解法 7 本（Qi，Shi & Gao，Zhao，Fan，Cheng，Zhu，ECMF）で，7 本すべてが中国語の HuBERT-large を音声に，7 本すべてが CLIP の ViT-L 系を映像に使っていた．言葉は 6 本が Baichuan 系（Baichuan-13B か Baichuan2），1 本（ECMF）が中国語 RoBERTa-large を使った．4 本（Qi，Shi & Gao，Zhu，ECMF）は GPT-4・Gemini・Qwen-Omni の出力を言葉の入力か疑似ラベルの出所に使った．
- 同じ 7 本で，ラベル無しデータを疑似ラベル（自己学習）に使ったのは 5 本（Qi，Shi & Gao，Fan，Cheng，Zhu）．疑似ラベルの絞り方は（1 本が複数に当たる），確信度 0.9（1 本．Shi & Gao），クラス別の閾値 0.99/0.85（1 本．Fan），クラスごとの上位 k=10 件を最大 10 回（1 本．Qi），2 つのモデルの一致か MLLM との一致（3 本．Shi & Gao，Fan，Zhu），絞らない（1 本．Cheng）．ラベル無しの動画を自己教師ありか対照学習の事前学習に使ったのは 3 本（Fan の対照学習，Zhao の音声と映像の対照による位置合わせ，Cheng の VideoMAE の事前学習）．ECMF はラベル無しデータの使い方を書いていない．FixMatch のような弱い拡張と強い拡張の一致を課す損失（consistency loss）は，確かめた解法のどれにも無かった．
- ラベル無しデータが上げた幅：融合した系で疑似ラベルの有無を比べられる 2 本では，Shi & Gao が +1.03（87.61→88.64）と +0.98（88.28→89.26），Fan が +1.90（86.35→88.25）．Zhu の事前学習の段は +1.17（0.8430→0.8547．表の読み取りに不確かさあり）．ラベル無しの動画での事前学習は映像の単独では大きく効く（MERBench の VideoMAE-base で +14.14，MER2024 で +10.46，Zhao の CLIP の位置合わせで +3.64）が，融合した系では Zhao で +0.23（86.78→87.01）だった．音声が強い競技会のデータでは，映像の改善は融合の後に小さくなる．
- エンコーダの扱い：基準はすべて凍結．上位の解法 7 本のうち，少なくとも 1 つのエンコーダを動かしたのは 4 本（Shi & Gao は CLIP の全体の微調整，Qi は凍結した CLIP へのプロンプト学習，Zhao は HuBERT へのアダプタ，Cheng は VideoMAE のドメインでの事前学習）．残り 3 本は凍結か記載なしである．LoRA を使った解法は確かめた中に無かった．
- アンサンブルか投票を最後に使ったのは 4 本（Shi & Gao，Fan，Zhu，ECMF）．
- 本研究との違い：競技会はすべて 1 つのコーパス（中国語の映画・ドラマの切り抜き）の中の半教師あり学習で，ラベル付きは 5,030〜7,369 件（クラスあたり数百件以上），6 クラス，指標は WAF で，ほかのコーパスのラベル付きデータは使わない．本研究はクラスあたり K 件という桁違いに少ないラベルと，別コーパスのソースを使う．競技会の上位の工夫（MLLM の出力を入力に足す，アンサンブル，エンコーダの部分的な微調整）は，本研究の「凍結特徴＋線形」の設定の外にある．
- MER2023 の MER-SEMI のテスト標本は，分類器が確信を持てた標本だけを注釈して作った．これは疑似ラベルで確信度の高い標本を選ぶ操作と同じ向きの偏りを持つので，MER-SEMI での疑似ラベルの得は，確信度の低い標本を含む普通のテストより大きく出る可能性がある（論文は検証していない）．MER2024・MER2025 のテスト標本の選び方が同じかは確かめていない．

### Gaps
- MER2023 の全部門 1 位の sense-dl-lab の手法は概要論文に書かれておらず，チームの論文も見つからなかった．
- MER2024-SEMI の 2 位と，MER2025-SEMI の 1 位のチームと手法は見つからなかった（順位表は CodaLab にあり，ウェブサイトや概要論文に載っていない）．Qi ほかの 1 位と Zhu ほかの 2 位は各論文の自己申告で，公式の順位表とは照合していない．
- Chen ほか 2023 と Expression MAE は要旨（後者は検索結果の抜粋）だけで，k の値，MAE の事前学習のデータ量・マスク率，閾値は確かめられなかった．
- 上位の解法の多くは，エンコーダを凍結したかどうかを明記していない（Fan，ECMF，Shi & Gao の HuBERT）．

---

## 2022〜2026 年のコーパス間のマルチモーダル感情・感情極性の研究：プロトコル（LOCO，共通クラスへの写像，話者独立，指標）

### Takeaway
基盤モデルの特徴でコーパス間を測る研究は，共通の 4 クラス（neutral・happy・sad・angry）にラベルを揃え，1 つのソースで学んで別のターゲットで試すか，複数コーパスで学んで残り 1 つで試す（leave-one-corpus-out，LOCO）のが典型である．ターゲットのラベルは使わない研究が多く，ターゲットのラベル無しデータを使う研究（テスト時適応，教師なし DA）は 2024〜2025 年に増えたが，ターゲットにクラスあたり K 件のラベルを与えたのは確かめた中で M4SER（音声と書き起こし）1 本だけだった．指標は研究ごとに WAF・UAR・UA/WA・重み付き F1・正解率がばらばらで，話者独立の記述はほとんど無い．

### Cited Findings
- **Ryumina ほか 2025（Pattern Recognition Letters 190）**：IEMOCAP・MOSEI・MELD・AFEW の 4 コーパスで，単一コーパスの学習 → 他コーパスでの試験，複数コーパスの学習と LOCO を比べた．コーパス間は 4 クラスの UAR（ほかに Acc，F1，wAcc，wF1，mF1）．融合は文脈を使わないゲートつきの注意 — [プロジェクトのページ](https://smil-spcras.github.io/MER/)；[ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167865525000662)．エンコーダは映像が AffectNet で学んだ ResNet-50（研究のコーパスでは微調整しない），音声は感情で学んだ wav2vec2 を各コーパスで微調整，言葉は感情で学んだ RoBERTa を各コーパスで微調整．MOSEI はラベルが 1 つの標本だけ使い，ラベルの無い標本を Neutral にした．ターゲットのラベルは使わない．分割は各コーパスの固定の分割で，話者独立かの記述は無い．単一コーパスで学んだ他コーパスの UAR は 15.5〜47.3%（例：テスト IEMOCAP で自コーパス 71.3，MOSEI から 38.2，MELD から 42.3，AFEW から 47.3），複数コーパスの学習で平均 +1.7〜+6.6 点（前回の精読．`docs/related-work.md` 2.3 節）．
- **Ryumina, Markitantov, Karpov 2023（Mathematics 11(16):3519）**：RAMAS と CMU-MOSEI で単一コーパスと複数コーパスの学習を比べ，音声と映像の時間モデルから区間単位の中間特徴を作って融合する．CMU-MOSEI の感情認識で重み付き正解率 78.1%（従来 59.9%）．**検索結果の要約だけで，本文は読んでいない** — [MDPI](https://www.mdpi.com/2227-7390/11/16/3519)
- **M4SER（He, Shi, Hu, Mi, Li, Toda．arXiv 2509.18706，2025-09）**：音声と Whisper の書き起こし（IEMOCAP は whisper-medium.en で WER 20.48%，MELD は whisper-large-v2 で WER 37.87%）．HuBERT-base（hubert-base-ls960）と BERT-base-uncased を微調整する．コーパス間は IEMOCAP↔MELD で，MELD を IEMOCAP と共通の 4 ラベル（Neutral, Happy, Angry, Sad）に絞り，Surprise・Fear・Disgust を捨てる．ソースは全部を学習に使い，ターゲットの 30% を検証と調整，70% をテストにする．**少数ショットの適応として，ターゲットの検証集合からクラスあたり K = 5，10，20，60 件のラベル付き標本を取る**（適応の仕組みとラベル無しのターゲットを使うかは本文の読んだ範囲に無い）．指標は IEMOCAP が WA・UA，MELD が ACC・W-F1．ゼロショット：基準（HuBERT＋BERT）IE→ME 53.0 / 52.4，ME→IE 58.9 / 57.4，M4SER 57.3 / 55.5 と 64.0 / 62.2．K 件の結果（M4SER）：IE→ME の ACC / W-F1 は K=5 で 57.9 / 56.0，K=60 で 62.1 / 61.5，ME→IE の WA / UA は K=5 で 66.4 / 64.4，K=60 で 74.0 / 75.9 — [M4SER](https://arxiv.org/html/2509.18706)
- **Guo ほか（arXiv 2412.07121，2025-02 の v2）**：テスト時適応の感情極性分析．CMU-MOSI・CMU-MOSEI・CH-SIMS の間で 5 通り（MOSEI→SIMS，MOSI→SIMS，MOSI→MOSEI，SIMS→MOSI，SIMS→MOSEI．MOSEI は MOSI を含むので MOSEI→MOSI は除く）．特徴は言葉が BERT-base（CH-SIMS は中国語 BERT-base），音声は LibROSA，映像は OpenFace 2.0 で，音声と映像は基盤モデルではない．ラベル無しのターゲットでは，1 つのモダリティを 0 にした版との対照損失（NT-Xent の変形）を 15 エポック，3 エポックごとの予測の変化が小さい上位約 5% に疑似ラベルを付けて 5 エポックの自己学習．指標は 2 値の正解率・F1・MAE，5 シードの平均．例：Late 融合で MOSI→SIMS の ACC 39.17 → 51.27，Early 融合で MOSEI→SIMS 45.95 → 63.89．比較した自己学習・正規化層の更新（TENT 型）などの既存手法は一部の設定で悪化した — [Guo ほか 2025](https://arxiv.org/html/2412.07121)
- **Boomda（Sun ほか．arXiv 2511.08152，2025-11）**：IEMOCAP と MSP-IMPROV のそれぞれを無作為に半分に分け，片方をソース，もう片方に雑音を加えてターゲットにする（音声は SNR 1.0 の白色雑音，映像は明るさを 20% にして SNR 0.5 のガウス雑音，言葉は各発話の 40% の単語を隠す）．別のコーパスをターゲットにはしていない．4 クラス（neutral, happy, sad, angry）．音声は WavLM＋TextCNN，映像は RAF-DB で学んだ APViT＋1 層の LSTM，言葉は BERT-base＋TextCNN で，事前学習した骨格は最後の 3 層だけを学習する．ターゲットのラベルは使わず，モダリティ間の投票で 3 票以上一致した標本に疑似ラベルを付ける．指標は重み付き F1．モダリティの組み合わせの平均で，ソースのみ IEMOCAP 36.75 / MSP-IMPROV 33.53，最良の既存手法 CDAN 46.95 / 40.07，Boomda 48.73 / 41.50 — [Boomda](https://arxiv.org/html/2511.08152v1)
- **MERIT（Zhang ほか．ICML 2025）**：凍結した Qwen-32B に軽い LoRA のアダプタを足し，テスト時に強化学習（多数決で確かめた報酬）で適応する．CMU-MOSEI と IEMOCAP で評価．**ポスターのページだけを読んだ**．学習とテストのデータの分け方（コーパス間か）と数値はページに無い — [ICML 2025 のページ](https://icml.cc/virtual/2025/50449)
- **EmoBox（Ma, Chen ほか．Interspeech 2024．音声のみ）**：コーパス間は IEMOCAP・MELD・RAVDESS・SAVEE を 4 クラス（怒・喜・中立・悲）に揃え，ターゲットのラベル無しで試す．テスト集合は各 240 件（各感情 60）で，emotion2vec の予測と元のラベルが一致した標本だけを残す．特徴は凍結した事前学習モデルの最終層＋小さな頭．コーパス間の平均正解率は Whisper large v3 52.0%，WavLM large 44.5%，HuBERT large 38.3%（前回の精読．`docs/related-work.md` 2.3 節） — [EmoBox](https://arxiv.org/abs/2406.07162)
- **Wagner ほか 2023（IEEE TPAMI．音声のみ）**：wav2vec2・HuBERT 系を MSP-Podcast で微調整し，IEMOCAP と MOSI で次元感情（CCC）のコーパス間を測る．最良のモデルでもコーパス間の低下は覚醒度 −11%，支配 −21%，感情価 −30%．凍結より微調整が常に良い（コーパス内で CCC +0.03〜+0.10）（前回の精読．`docs/related-work.md` 2.3 節） — [Wagner ほか](https://arxiv.org/abs/2203.07378)
- **Phukan ほか 2025（APSIPA ASC．音声のみ）**：凍結した事前学習音声モデル 8 種で，CREMA-D・RAVDESS・EmoDB など 5 コーパスのコーパス間．パラ言語用の TRILLsson が最良（前回は要旨と結果だけを読んだ．`docs/related-work.md` 2.3 節） — [Phukan ほか](https://arxiv.org/abs/2509.16182)
- MERBench のコーパス間の設定と結果は 1 つ目の問いに書いた — [MERBench](https://arxiv.org/html/2401.03429)

### Inferences
- 確かめたコーパス間の研究 9 本（MERBench，Ryumina 2025，M4SER，Guo 2025，Boomda，MERIT，EmoBox，Wagner 2023，Phukan 2025．Ryumina 2023 は要約のみなので数えない）のうち，ラベルを共通の 4 クラス（neutral・happy・sad・angry）に揃えたのは 5 本（MERBench，Ryumina 2025，M4SER，Boomda，EmoBox）．感情極性か次元感情で揃えたのは 3 本（MERBench の一部，Guo，Wagner）．
- 評価の形：1 つのソース → 1 つのターゲットの組ごとの評価が 7 本（MERBench，Ryumina 2025，M4SER，Guo，EmoBox，Wagner，Phukan），LOCO が 1 本（Ryumina 2025．組ごとの評価と両方を行う），同じコーパスを分けて雑音を足したターゲットが 1 本（Boomda），分け方が不明が 1 本（MERIT）．
- ターゲットの情報の使い方：使わない（ゼロショット）が 5 本（MERBench，Ryumina 2025，EmoBox，Wagner，Phukan），ラベル無しのターゲットを使う（テスト時適応・教師なし DA）が 3 本（Guo，Boomda，MERIT），クラスあたり K 件のラベル付きのターゲットを使うのが 1 本（M4SER．K = 5，10，20，60）．ターゲットの K 件のラベルとラベル無しを同時に使う研究と，複数のソースを混ぜてから K 件で適応する研究は，確かめた中に無かった．
- 指標：UAR を主に使うのは Ryumina 2025 だけで，M4SER は IEMOCAP 側で UA（UAR と同じ量）を出す．WAF・重み付き F1 は MERBench，Boomda，M4SER の MELD 側．正解率は EmoBox．したがって研究をまたいで数値を比べることはほとんどできない．
- エンコーダ：凍結のままコーパス間を測ったのは MERBench，EmoBox，Phukan の 3 本．音声か言葉のエンコーダを微調整したのは Ryumina 2025，M4SER，Wagner の 3 本，最後の 3 層だけ学習したのが Boomda．映像に顔の表情で学んだ CNN を凍結で使う点は Ryumina 2025（ResNet-50，AffectNet）が本研究と同じである．
- 話者独立：コーパスが違えば話者は重ならないが，ターゲットの中で K 件を取った話者とテストの話者を分けたかを書いた研究は無かった（M4SER も記載なし）．
- 本研究に最も近いのは，コーパスとクラスと指標が一致する Ryumina 2025（ただし音声と言葉を微調整し，ターゲットのラベルを使わない）と，クラスあたり K 件の設定が一致する M4SER（ただし音声と書き起こしだけ，ソース 1 つ，エンコーダを微調整，K 件の使い方が不明）である．

### Gaps
- MSP-IMPROV・CREMA-D を別コーパスとしてマルチモーダルのコーパス間を測った 2022〜2026 年の研究は，今回の検索では見つからなかった（Boomda は同じコーパスの中で雑音を足した形）．
- M4SER の K 件の使い方（微調整か，ラベル無しのターゲットも使うか）は，本文の最後の約 7,000 字を読まなかったので確かめていない．
- MERIT の数値とコーパス間かどうかは，OpenReview の本文を読んでいないので分からない．
- Ryumina 2023 は本文を読んでいない．

---

## 全体として「標準の手法」は何か，各選択はどれだけ多いか

### Takeaway
2023〜2026 年の基盤モデルの特徴を使う MER の事実上の標準は，「凍結した事前学習エンコーダから発話単位の特徴を取り出し（音声は HuBERT-large/WavLM の最後の数層の和か平均を時間平均，映像は OpenFace で揃えた顔の CLIP ViT-L/14 か表情で学んだ CNN の特徴を時間平均，言葉は RoBERTa 系か 7〜13B の LLM の特徴），モダリティごとに全結合で写して注意の重みで足し，交差エントロピーで学ぶ」形である．ラベル無しデータは主に確信度の高い疑似ラベル（2 つのモデルか MLLM との一致で絞る）と，ラベル無しの同じ領域の動画での自己教師ありの事前学習に使う．コーパス間は 4 クラスに揃えてソースで学び，ターゲットで試す．

### Cited Findings
- 基準の型（凍結した特徴，発話単位，全結合＋注意の融合，5 分割交差検証，超パラメータの無作為探索）は MERBench と MER2023〜MER2025 の基準に共通する — [MERBench](https://arxiv.org/html/2401.03429)；[MER2023 v1](https://arxiv.org/pdf/2304.08981v1)；[MER2024](https://arxiv.org/html/2404.17113)；[MER2025](https://arxiv.org/html/2504.19423v1)
- 音声の特徴は最後の 4 層の隠れ状態の和を時間平均し，推論だけでエンコーダを更新しない — [MERTools の台本](https://raw.githubusercontent.com/zeroQiaoba/MERTools/master/MERBench/feature_extraction/audio/extract_audio_huggingface.py)
- 3 回の競技会の基準で，単モダリティの最良は毎回 HuBERT-large（MER2023 の MER-SEMI 85.11，MER2024 の平均 76.55，MER2025 のテスト 72.27） — [MER2023 v1](https://arxiv.org/pdf/2304.08981v1)；[MER2024](https://arxiv.org/html/2404.17113)；[MER2025](https://arxiv.org/html/2504.19423v1)
- 映像の最良は MER2023 では表情で学んだ CNN（ResNet-FER2013，MANet-RAFDB を融合に採用），MER2024 以降は CLIP-large（MER2024 の平均 63.32，MER2026 の MER-Cross 58.88） — [MER2023 v1](https://arxiv.org/pdf/2304.08981v1)；[MER2024](https://arxiv.org/html/2404.17113)；[MER2026](https://arxiv.org/html/2604.19417)
- 上位の解法の具体的な選択は 2 つ目の問いの各項に書いた — [Qi](https://arxiv.org/html/2409.07078)；[Shi & Gao](https://arxiv.org/html/2409.05007v1)；[Zhao](https://arxiv.org/html/2409.05015v2)；[Fan](https://arxiv.org/html/2409.04447v1)；[Cheng](https://arxiv.org/html/2408.10500)；[Zhu](https://arxiv.org/html/2508.06036v1)；[ECMF](https://arxiv.org/html/2508.05991v1)

### Inferences
数えた範囲：基準 4 本（MERBench，MER2023〜2025 の基準．MER2026 は MER-SEMI が無いので除く），競技会の上位の解法 7 本（本文を読んだもの），コーパス間の研究 9 本（3 つ目の問い）．

| 選択 | 数え方 | 数 |
|---|---|---|
| 音声に HuBERT-large（中国語版を含む） | 上位の解法 7 本 | 7 / 7 |
| 映像に CLIP の ViT-L 系 | 上位の解法 7 本 | 7 / 7 |
| 言葉に Baichuan 系の LLM | 上位の解法 7 本 | 6 / 7（残り 1 本は中国語 RoBERTa-large） |
| GPT-4・Gemini・Qwen-Omni などの出力を入力か疑似ラベルに使う | 上位の解法 7 本 | 4 / 7 |
| エンコーダを凍結（基準） | 基準 4 本 | 4 / 4 |
| 少なくとも 1 つのエンコーダを動かす（全体の微調整，プロンプト学習，アダプタ，領域での事前学習） | 上位の解法 7 本 | 4 / 7（LoRA は 0） |
| 注意による融合（基準の注意か，その変形の Transformer・MoE を含む） | 上位の解法 7 本 | 6 / 7（Fan ほかは融合の分類器の形が書かれていない） |
| 疑似ラベル（自己学習） | 上位の解法 7 本 | 5 / 7（Qi，Shi & Gao，Fan，Cheng，Zhu） |
| ラベル無しの動画での自己教師あり・対照の事前学習 | 上位の解法 7 本 | 3 / 7 |
| 一致性の損失（弱い拡張と強い拡張の予測を揃える） | 上位の解法 7 本 | 0 / 7 |
| アンサンブル・投票 | 上位の解法 7 本 | 4 / 7 |
| ラベルを共通の 4 クラスに揃える | コーパス間 9 本 | 5 / 9 |
| ターゲットのラベルを使わない | コーパス間 9 本 | 8 / 9（K 件を使うのは M4SER だけ） |
| ターゲットのラベル無しデータを使う | コーパス間 9 本 | 3 / 9 |
| LOCO | コーパス間 9 本 | 1 / 9 |
| 主指標が UAR（UA を含む） | コーパス間 9 本 | 2 / 9（Ryumina 2025，M4SER の IEMOCAP 側） |
| 凍結のエンコーダでコーパス間を測る | コーパス間 9 本 | 3 / 9 |

- 本研究の設定（凍結した WavLM・ResNet-50 の顔・RoBERTa，線形分類器，4 クラス，UAR）は，MERBench の凍結特徴の型から分類器を単純にしたもので，コーパス間の研究の中では Ryumina 2025 のコーパスとクラスと指標に一致する．本研究のように「ターゲットのクラスあたり K 件のラベル＋ターゲットのラベル無し＋複数のソースコーパス」を同時に扱う研究は，今回確かめた範囲に無かった．競技会の半教師あり学習は同じ領域のラベル無しデータ，コーパス間の研究はラベル無しか少数ラベルのどちらか一方で，2 つの流れは分かれている．
- 競技会で標準になった「確信度で絞った疑似ラベル」は，確信度の高い標本を足す操作なので，ソースとターゲットの間でラベルの付け方がずれている場合の効果は，競技会の結果からは分からない（競技会のラベル無しデータはラベル付きと同じ領域で，テストもラベル無しの集合から取る）．
- 数値の比較の注意：競技会は WAF で 6 クラス，本研究は UAR で 4 クラスなので，競技会の +1〜+2 点の得を本研究の得と直接並べられない．

### Gaps
- 数えたのは今回本文を読めた論文だけで，MER の競技会の参加チームの全論文（MER2024 の MER-SEMI だけで採録 5 本，うち Ge ほか 1 本は未読）や，2022〜2026 年のコーパス間の研究の網羅的な一覧ではない．割合は「確かめた範囲で」の値として扱う．
- 英語のコーパス（IEMOCAP・MELD・MOSEI・AFEW）で，MER の競技会と同じ「凍結特徴＋疑似ラベル」の型をコーパス間に当てた研究は見つからなかった．
