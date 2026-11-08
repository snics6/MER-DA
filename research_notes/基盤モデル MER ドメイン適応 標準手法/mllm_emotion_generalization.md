# MLLM と大規模基盤モデルによる感情認識：コーパスとドメインをまたぐ汎化（2023–2026）

調べた日：2026-11-08．読み方の印は次のとおりである．◎ は arXiv の HTML 版または PDF の本文を読んだもの（HTML は WebFetch で取得し，要約用のモデルに表と本文から値を抜き出させた．100,000 字を超える論文は前半だけを読んだものがあり，その場合は項目ごとに書く）．○ はこのリポジトリの既存のノート（`research_notes/疑似ラベル 情報の持ち込み方 先行研究/5_外部の知識と弱い情報源.md`）が原典の本文を読んで書いた値を引き写したもの．△ は検索結果の抄録だけで内容を知ったもの．

本研究の設定（比較のための前提）：凍結したエンコーダ（音声 WavLM，顔 ResNet-50，言葉 RoBERTa）の特徴の上で，モダリティごとの線形 softmax 分類器と融合の分類器を学ぶ．ターゲットのコーパスにはクラスあたり K 本のラベルとラベル無しの発話がある．ソースは AFEW・IEMOCAP・MOSEI・MELD のうち 2〜3 コーパスで，ラベルは 4 クラス，主指標は UAR である．

## 問い 1：感情に特化した MLLM と音声の LLM は，どのエンコーダと LLM を使い，どの部分を学習するか

### Takeaway
感情に特化した MLLM の多くは，凍結した事前学習エンコーダ（音声は HuBERT か Whisper，映像は CLIP・EVA 系の ViT か MAE 系）の出力を，学習する射影層または Q-Former で 7B 級の LLM（LLaMA2・Vicuna か Qwen 系）の入力に写し，LLM には LoRA だけを足して学ぶ．学習データは，ラベル無しの動画に別の MLLM と GPT が書いた感情の記述文で作る例が多い．

### Cited Findings

**Emotion-LLaMA（Cheng ほか，NeurIPS 2024）◎**
- エンコーダ：音声は HuBERT（4.2 節で「HuBERT-Chinese large」），映像は 3 種で，顔の系列に MAE，時間方向に VideoMAE，表情の頂点のフレーム（448×448）に EVA を使う．文章は LLaMA の tokenizer に通すだけで，別の文章エンコーダは無い．各特徴は線形層で 4096 次元に写し，文の token とつなぐ — [arXiv 2406.11161](https://arxiv.org/html/2406.11161)
- LLM：「LLaMA2-chat (7B) equipped with LoRA」．LoRA は Wq と Wv に r=64，α=16 で入れる．エンコーダは凍結し（「we froze the visual and audio backbones, focusing on training the linear projection layer」），学習するパラメータは「only 34 million, representing a mere 0.495%」— [arXiv 2406.11161](https://arxiv.org/html/2406.11161)
- 学習：段 1 は MERR の粗い記述 28,618 本で音声と映像の token を語の埋め込みに揃え，段 2 は細かい記述 4,487 本に MER2023 と DFEW を加えて指示学習する（4.1 節「on the MER2023 and DFEW datasets」）．4 枚の A100 で 300,000 ステップ，約 20 時間 — [arXiv 2406.11161](https://arxiv.org/html/2406.11161)
- 記述文の作り方：OpenFace の Action Unit の組み合わせから規則で疑似ラベルを付け（「they were not relied upon as ground truth」），Qwen-Audio（音声の手がかり）と MiniGPT-v2（映像の手がかり）の記述と字幕を LLaMA-3 にまとめさせて粗い記述を作る．細かい記述は 4 人の専門家の投票で選ぶ．元の動画の集まりには「over 70,000 unannotated samples」が含まれる — [arXiv 2406.11161](https://arxiv.org/html/2406.11161)

**AffectGPT（Lian ほか，ICML 2025．arXiv 2501.16566）◎**
- エンコーダと LLM：音声は HuBERT-L，映像は CLIP ViT-L が既定（付録 B）．EVA_CLIP と DINOv2 も比べ，CLIP ViT がわずかに良い．LLM は Qwen2.5 が既定で，ほかの LLM との差は小さいと報告する（読んだ範囲では大きさを確認できなかった）— [arXiv 2501.16566](https://arxiv.org/html/2501.16566)
- 設計の特徴：LLM に入れる前に音声と映像の特徴を融合する「pre-fusion」を持つ．Q-Former 型と，各モダリティを平均してから注意の重みで混ぜる型があり，報告の主の値（平均 74.77）は後者である — [arXiv 2501.16566](https://arxiv.org/html/2501.16566)
- 学習する部分：LLM の LoRA（既定の rank 16），射影層，pre-fusion の枝だけを学び，LLM とエンコーダは凍結する．AdamW，学習率 1e-5，最大 60 エポック — [arXiv 2501.16566](https://arxiv.org/html/2501.16566)
- 学習データ：MER-Caption 115,595 本（粗いラベル）と，それを絞った MER-Caption+ 31,327 本．元の動画は MER2024 のラベル無しの部分である（「This paper does not involve the collection of new data」）．注釈は「model-led human-assisted」で，SALMONN が音声の手がかり，Chat-UniVi が映像の手がかりを書き，GPT-3.5（gpt-3.5-turbo-16k-0613）が文章と合わせる．GPT-3.5 を公開の LLM に替えると成績が下がる．絞り込みでは，人がラベルを付けたデータで学んだ分類器の投票と記述から取り出したラベルが食い違う標本を捨てる — [arXiv 2501.16566](https://arxiv.org/html/2501.16566)

**EmoVIT（Xie ほか，CVPR 2024．画像の感情）◎**
- InstructBLIP を土台にし，「fine-tune exclusively the Q-Former module, whilst keeping both the image encoder and the language model frozen」．画像エンコーダの名前は本文に無い．LLM の比較（EmoSet の正解率）は Vicuna-7B 83.36，Vicuna-13B 82.21，FlanT5-XL 80.98 — [arXiv 2404.16670](https://arxiv.org/html/2404.16670)
- 指示データ：BLIP2 のキャプションと感情の属性（明るさ，色，場面，物体，表情，動作）を GPT-4 に渡して作る．元は EmoSet の 51,200 枚（EmoSet 全体の 50% 未満．GPT の API の費用が理由）— [arXiv 2404.16670](https://arxiv.org/html/2404.16670)

**EmoLLM（Yang, Ye, Du．武漢大学．arXiv 2406.16442v2．掲載先の記載なし）◎**
- 映像は CLIP ViT-L/14，音声は Whisper-base と MLP．LLM は 4.2 節で LLaMA2-7B，5.1 節で Vicuna-v1.5（7B）と書かれ，本文の中で食い違う．LoRA で学習すると書くが，どの部分を学ぶかは書かれていない — [arXiv 2406.16442](https://arxiv.org/html/2406.16442)
- 学習データ EmoBench：8 つのデータ（Emotic，CAER-S，MELD，Emotion6，MintRec，SMILE，MMHS，MMSD）から約 287k の指示．質問の型は GPT-4 が書き，答えは元のデータのラベルである — [arXiv 2406.16442](https://arxiv.org/html/2406.16442)
- 注意：同名に近い「EmoLLMs」（文章だけの感情分析の LLM 群）は別の研究で，今回は読んでいない（問い 1 の Gaps）．

**Emotion-Qwen（arXiv 2505.06685）◎**
- 映像は CLIP ViT（変種の記載なし）と，DeepFace で顔の表情の確信度が高いフレームを選ぶ前処理．音声は使わない（今後の課題と書く）．LLM は Qwen2.5-7B．射影層は 2 つの専門家の MLP（感情用と一般用）を注意のゲートで混ぜる「Hybrid Compressor」（約 95.74M パラメータ）— [arXiv 2505.06685](https://arxiv.org/html/2505.06685)
- 学習：事前学習の段 1・2 では視覚エンコーダとゲートと専門家を学び，段 3 で LLM を解いて一般の指示データ 1.9M で学ぶ．その後の指示学習では視覚エンコーダと Hybrid Compressor を凍結し，データごとに 1 つの LoRA（rank 64）を学ぶ．LoRA 1 つや全体の微調整より複数の LoRA が良い（EmoSet で 85.49，82.27，81.57）— [arXiv 2505.06685](https://arxiv.org/html/2505.06685)
- 学習データ VER：MAFW の 8,034 本と MER2024 の 36,357 本．Qwen-VL-Max と GPT-4 が下書きし，DeepSeek でまとめ，12 人が 0〜5 点で採点して 3 点以上を残す — [arXiv 2505.06685](https://arxiv.org/html/2505.06685)

**R1-Omni（arXiv 2503.05379）◎**
- HumanOmni-0.5B を，EMER の 232 本と著者の注釈 348 本の計 580 本で SFT してから，MAFW と DFEW の学習データ 15,306 本で RLVR（GRPO．報酬は正解なら 1 の正解報酬と形式の報酬の和）で学ぶ．推論の文の教師は使わない．別のエンコーダ名は本文に無い — [arXiv 2503.05379](https://arxiv.org/html/2503.05379)

**音声の LLM：Qwen-Audio，Qwen2-Audio，SALMONN，SECap ◎**
- Qwen-Audio：音声エンコーダは Whisper-large-v2 で初期化（約 640M），LLM は Qwen-7B．多タスクの事前学習では「we freeze the weights of LLM and only optimize the audio encoder」，Qwen-Audio-Chat の SFT では逆に「we fix the weights of the audio encoder and only optimize the LLM」．表 1 に感情認識（ER）が「<<1k」時間あるが，どのコーパスかは書かれていない — [arXiv 2311.07919](https://arxiv.org/html/2311.07919)
- Qwen2-Audio：音声エンコーダは Whisper-large-v3 で初期化，LLM は Qwen-7B（全体 8.2B）．事前学習の階層タグを自然文のプロンプトに替え，SFT と DPO を加える．後の段でどの部分を学ぶかは読んだ範囲で書かれていない — [arXiv 2407.10759](https://arxiv.org/html/2407.10759)
- SALMONN：Whisper-Large-v2 のエンコーダと BEATs を凍結し，窓ごとの Q-Former（1 query，約 0.33 秒の窓）と LoRA（rank 8）だけを学ぶ（約 33M，全体の「∼0.24%」）．LLM は Vicuna-13B．指示学習の 12 タスクに感情認識が入り，IEMOCAP の Session 1〜4（5 時間，約 4K 本）を使う — [arXiv 2310.13289](https://arxiv.org/html/2310.13289)
- SECap（AAAI 2024．音声の感情の説明文を生成）：音声は WenetSpeech で事前学習した HuBERT-large を凍結し，BERT-base で初期化した Q-Former と射影層を学び（約 103M），中国語向けに調整した LLaMA は凍結する．学習データは社内の EMOSpeech（話者 7 人，41.6 時間，30,526 文，中国語，1 発話に人の説明文 3〜5 本）．話の内容と音声の埋め込みの相互情報量を減らす損失と，説明文との対照学習を足す — [arXiv 2312.10381](https://arxiv.org/html/2312.10381)

### Inferences
- 読んだ 10 本（Emotion-LLaMA，AffectGPT，EmoVIT，EmoLLM，Emotion-Qwen，R1-Omni，Qwen-Audio，Qwen2-Audio，SALMONN，SECap）での数え上げ：
  - 音声エンコーダ：HuBERT が 3 本（Emotion-LLaMA，AffectGPT，SECap），Whisper が 4 本（EmoLLM，Qwen-Audio，Qwen2-Audio，SALMONN），音声なしが 2 本（EmoVIT，Emotion-Qwen），不明が 1 本（R1-Omni）．WavLM を MLLM の音声エンコーダに使う例は見つからなかった（WavLM-large は Qwen-Audio・Qwen2-Audio の比較相手として出るだけ）．
  - 映像エンコーダ：CLIP 系の ViT が 3 本（AffectGPT，EmoLLM，Emotion-Qwen），EVA と MAE・VideoMAE の組が 1 本（Emotion-LLaMA），名前の記載なしが 2 本（EmoVIT，R1-Omni），映像なしが 4 本（音声の LLM）．ResNet-50 の顔特徴を使う MLLM は見つからなかった．
  - LLM：LLaMA 系（LLaMA2，Vicuna，中国語 LLaMA）が 5 本，Qwen 系が 4 本（AffectGPT，Emotion-Qwen，Qwen-Audio，Qwen2-Audio），不明が 1 本．大きさは 7B が中心で，SALMONN だけが 13B，R1-Omni は 0.5B．
  - 学習する部分：「エンコーダ凍結＋射影層または Q-Former＋LoRA」が 4 本（Emotion-LLaMA，AffectGPT，SALMONN，EmoLLM．EmoLLM は部分の記載なし），「Q-Former か射影層だけ（LLM も凍結）」が 2 本（EmoVIT，SECap），「エンコーダか LLM の全体を段ごとに学ぶ」が 2〜3 本（Qwen-Audio，Emotion-Qwen の事前学習，Qwen2-Audio は不明），強化学習が 1 本（R1-Omni）．
- 学習データの作り方：MLLM と GPT に記述を書かせた指示データで学ぶものが 5 本（Emotion-LLaMA，AffectGPT，EmoVIT，EmoLLM の質問文と EmoPrompt，Emotion-Qwen）．R1-Omni の初期の SFT データ EMER の作り方は今回確かめていない．SECap は人が書いた説明文で学ぶ．5 本のうち Emotion-LLaMA・AffectGPT・Emotion-Qwen の 3 本は MER2023 か MER2024 の動画（ラベル無しの部分を含む）を素材にしている．
- 本研究との違い：これらはすべて，エンコーダの上に 7B 級の生成モデルを置き，数万〜数十万本の記述文で学ぶ．本研究は凍結特徴の上の線形分類器で，学習データはターゲットの K 本とソースの 2〜3 コーパスのラベルである．本研究の音声（WavLM）と顔（ResNet-50）のエンコーダは，読んだ MLLM のどれとも一致しない．言葉の RoBERTa は ACL-MER の学生モデル（問い 4）と一致する．

### Gaps
- AffectGPT の Qwen2.5 の大きさ（7B か）と，学習に使った MER2024 のラベル無しの部分と評価に使った MER2024 の部分の重なりは，読んだ範囲（本文の前半 100,000 字）で確認できなかった．
- EmoVIT の画像エンコーダは本文に名前が無い（InstructBLIP の既定の構成は本文で確かめていない）．
- R1-Omni（HumanOmni-0.5B）のエンコーダと，RLVR でどのパラメータを更新するかは本文に書かれていない．
- 文章だけの「EmoLLMs」（Liu ほか 2024）は読んでいない．名前が似ているので，報告で引くときはどちらかを書き分ける必要がある．
- Qwen-Audio の感情認識の事前学習データ（「<<1k」時間）がどのコーパスかは書かれておらず，IEMOCAP や MELD を含むかは確かめられない．

## 問い 2：感情に特化した MLLM はコーパスをまたいでどう評価され，教師ありの専門モデルと比べてどこまで汎化すると報告されているか

### Takeaway
感情に特化した MLLM の評価の主な場は MER2023・MER2024（中国語の映画とドラマ）と DFEW で，学習データの素材と同じ動画の集まりから評価データを取る例が多い．学習に使っていないコーパスでの評価（R1-Omni の RAVDESS，EmoVIT の held-out，AffectGPT の MELD・IEMOCAP など）は少なく，指標も UAR ではなく「出力にラベルが含まれるか」の hit rate や集合の重なりの F を使うことが多い．

### Cited Findings

**Emotion-LLaMA ◎**
- MER2023（F1）：音声・映像・文章で 0.9036 で表の最高．音声・映像だけでは 0.8905 で，VAT（0.8911）をわずかに下回る — [arXiv 2406.11161](https://arxiv.org/html/2406.11161)
- DFEW（表 2）：zero-shot の行が UAR 45.59，WAR 59.37 で，ほかの MLLM より高い（「showcasing its strong generalization ability」）．微調整の行は UAR 64.21，WAR 77.06 で，教師ありの専門モデル S2D（61.82 / 76.03），VideoMAE（63.60 / 74.60），MAE-DFER（63.41 / 74.43）を上回る．zero-shot で disgust の再現率は 0.00 で，著者は「safety constraints on the term 'disgust' within large language models」によると推測する — [arXiv 2406.11161](https://arxiv.org/html/2406.11161)
- zero-shot の行で DFEW を学習から外したかは，本文から確定できない．4.1 節は MER2023 と DFEW で指示学習したと書き，表 2 は zero-shot の行と微調整の行を並べる．後半の本文（100,000〜200,000 字）にも DFEW を外したと明記した文は無かった — [arXiv 2406.11161](https://arxiv.org/html/2406.11161)
- EMER（100 本．ChatGPT で採点）：手がかりの重なり 7.83，ラベルの重なり 6.25 で，次点の Valley（7.24 / 5.77）を上回る — [arXiv 2406.11161](https://arxiv.org/html/2406.11161)
- MER2024：MER-NOISE で WAF 84.52% で 1 位（2 位と 3 位との差は 1.47 と 1.65 ポイント），MER-OV で平均 66.10（GPT-4V は 57.58）— [arXiv 2406.11161](https://arxiv.org/html/2406.11161)
- 読んだ範囲で，MELD・IEMOCAP・CMU-MOSEI の結果は無い — [arXiv 2406.11161](https://arxiv.org/html/2406.11161)

**AffectGPT と MER-UniBench ◎**
- MER-UniBench の構成：細かい感情は OV-MERD+（532 本．集合の Precision・Recall・F を感情の輪 K 種で平均），基本感情は MER2023（MER-MULTI，411 本），MER2024（MER-SEMI，1,169 本），IEMOCAP（Session 5，1,241 本），MELD（test，2,610 本）で，指標は「hit rate」（基本感情のラベルが出力の中に現れるか）．感情の極性は CMU-MOSI（686），CMU-MOSEI（4,659），CH-SIMS（457），CH-SIMS v2（1,034）で WAF — [arXiv 2501.16566](https://arxiv.org/html/2501.16566)
- 全データの平均（表 2）：AffectGPT 74.77，Emotion-LLaMA 64.17，mPLUG-Owl 62.45，Chat-UniVi 57.94，SALMONN 57.89，Qwen-Audio 49.26．著者は「over a 9% performance improvement compared to existing MLLMs」と書く．比較相手は公開の重みの MLLM で，GPT-4o と教師ありの分類器との比較は読んだ範囲に無い — [arXiv 2501.16566](https://arxiv.org/html/2501.16566)
- 著者は評価を zero-shot とは明記していない．学習データの素材は MER2024 のラベル無しの部分であり，評価にも MER2024 の部分を使う — [arXiv 2501.16566](https://arxiv.org/html/2501.16566)
- 著者の限界の記述：「MER-Caption+ may contain inaccurate descriptions due to the use of an automatic annotation strategy without manual checks」，評価は 1 人が映る動画に限る，「increasing the input modality does not always improve performance」— [arXiv 2501.16566](https://arxiv.org/html/2501.16566)

**OV-MER（自由語彙の感情認識．Lian ほか，ICML 2025．arXiv 2410.01495）◎**
- 課題：決まったラベル集合の外の感情も予測させる．OV-MERD は MER2023 の一部を付け直したもので，236 種の感情，1 標本あたり多くは 2〜4 ラベル．映像の手がかりは GPT-4V（3 フレーム），音声の手がかりは SALMONN が下書きし，人が 2 回直す．GPT-3.5 が記述からラベルを取り出し，人が確かめる — [arXiv 2410.01495](https://arxiv.org/html/2410.01495)
- 指標：予測と正解のラベルを同義語の群に写してから集合の Precision・Recall・F を取る．群への写し方は GPT-3.5（既定．API の版で結果が変わりうると著者が書く）か感情の輪で，感情の輪の得点は GPT の得点と相関が最大 0.94 — [arXiv 2410.01495](https://arxiv.org/html/2410.01495)
- 英語の F（表 2）：無作為 17.42，Qwen-Audio 38.13，SECap 45.72，SALMONN 47.96，Video-LLaMA 44.73，Chat-UniVi 53.08，GPT-4V 55.51，PandaGPT（音声＋映像）45.89．人が直した手がかりを使う上限は，文章 46.00，映像 60.55，音声 65.35，全部 80.05．著者は MLLM の成績を「remains unsatisfactory」と書き，音声の LLM は「struggle to capture emotion-related paralinguistic features」と書く — [arXiv 2410.01495](https://arxiv.org/html/2410.01495)
- 判別型（one-hot の分類器）との比較は付録 U にあると本文が書くが，読んだ範囲（前半 100,000 字）に入っていない — [arXiv 2410.01495](https://arxiv.org/html/2410.01495)

**EmoVIT ◎**
- EmoSet で学び，別の 7 データで評価する（正解率．EmoVIT / BLIP2 / InstructBLIP / LLaVA）：WebEmo 21.12 / 20.10 / 12.80 / 12.55，FI 68.09 / 57.72 / 37.97 / 56.04，Emotion6 57.81 / 50.00 / 46.11 / 49.44，ArtPhoto 44.90 / 36.25 / 26.25 / 36.25，EmotionROI 53.87 / 50.51 / 46.13 / 46.46．EmoSet 自体は 83.36．著者はこの held-out を「does not constitute a strict zero-shot evaluation in the conventional sense」と書く — [arXiv 2404.16670](https://arxiv.org/html/2404.16670)
- held-out の評価で文脈内の例を足しても成績は上がらなかった — [arXiv 2404.16670](https://arxiv.org/html/2404.16670)

**EmoLLM ◎**
- 評価はすべて EmoBench の test（30K）で，学習と同じ手順で作ったデータである．Emo-C（閉じたラベル集合の感情）で EmoLLM 64.06，OneLLM 51.16，Chat-UniVi 47.62，Vicuna の zero-shot 29.21．画像だけの部分集合（表 3）では EmoLLM 75.03，GPT-4V 57.90，Gemini 1.0 と 1.5 は同じ値 45.47 が並ぶ．コーパスをまたぐ評価は無い — [arXiv 2406.16442](https://arxiv.org/html/2406.16442)

**Emotion-Qwen ◎**
- 表 2 の「zero-shot」の行（事前学習の後，VER・DFEW での指示学習の前）：MER2024 SEMI / NOISE 82.85 / 77.53，DFEW は WAR 77.19・UAR 38.60（本文は 77.19 を UAR と呼び，表と食い違う）．微調整の後：MER2024 85.47 / 79.67，DFEW WAR 78.31・UAR 62.11（Emotion-LLaMA 77.06 / 64.21，MMA-DFER 77.51 / 67.01）— [arXiv 2505.06685](https://arxiv.org/html/2505.06685)
- 指示学習の後に一般の画像言語の成績が下がる：MMBench 87.3→85.3，VQAv2 84.8→79.2，POPE 83.2→79.4 — [arXiv 2505.06685](https://arxiv.org/html/2505.06685)

**R1-Omni ◎**
- 学習に使っていない RAVDESS（OOD）で，SFT だけの模型が UAR 29.33・WAR 30.75，RLVR の後が 43.00・44.69．学習に使った DFEW では 60.23→65.83（UAR），MAFW では 50.44→57.68（UAR）．評価は自由語彙の手順である — [arXiv 2503.05379](https://arxiv.org/html/2503.05379)
- 著者が挙げる失敗：字幕の読み違い，動画に根拠の無い推論（「fabricated reasoning leads the model to incorrectly predict the emotion as surprise」），声の調子より映像に頼る — [arXiv 2503.05379](https://arxiv.org/html/2503.05379)

**音声の LLM の感情の評価 ◎**
- Qwen-Audio：MELD の test で正解率 0.557（WavLM-large 0.542）．「The evaluation datasets are rigorously excluded from the training data」と書くが，感情認識の学習データのコーパス名は無い — [arXiv 2311.07919](https://arxiv.org/html/2311.07919)
- Qwen2-Audio：MELD で 0.553 で，Qwen-Audio（0.557）をわずかに下回る — [arXiv 2407.10759](https://arxiv.org/html/2407.10759)
- SALMONN：IEMOCAP の Session 1〜4 で学び，Session 5 で正解率 0.63（activation tuning なし），0.69（あり）．論文の参照値は 0.81．学習したタスクであり zero-shot ではない — [arXiv 2310.13289](https://arxiv.org/html/2310.13289)
- SECap：説明文の指標（BLEU，METEOR，ROUGE-L，CIDEr など）と人の評価（MOS 3.77）だけで，感情の分類の正解率とコーパス間の評価は無い — [arXiv 2312.10381](https://arxiv.org/html/2312.10381)

**MER の挑戦（challenge）の論文 ◎**
- MER2024：MER-SEMI と MER-NOISE は 6 クラス（worried，happy，neutral，angry，surprised，sad）で WAF で順位を決める．ラベル付きの学習・検証は 5,030 本，ラベル無しを含む全体は 115,595 本．基準の手法は MLLM ではなく，凍結した特徴（映像 CLIP-large 63.32，音声 HuBERT-large 76.55，言葉 Baichuan-13B 54.50．単一モダリティの平均 WAF）の注意による融合で，3 モダリティで平均 81.97．MER-OV の基準は MLLM（GPT-4V 56.69，Chat-UniVi 53.09，Qwen-Audio 40.23，無作為 19.13）で，「there is still a significant gap between MLLMs and ground truth」— [arXiv 2404.17113](https://arxiv.org/html/2404.17113)
- MER2025：MER-SEMI は 7,369 本のラベル付きと 124,802 本のラベル無し．基準は単一モダリティ特徴の注意融合（HuBERT-large だけで test WAF 72.27，3 モダリティで 78.63）．細かい感情の MER-FG では zero-shot の MLLM の最良が mPLUG-Owl の 36.80，MER-Caption+ で学んだ AffectGPT が 46.86（OV-MERD で学ぶと 38.98）— [arXiv 2504.19423](https://arxiv.org/html/2504.19423)

### Inferences
- 学習に使っていないコーパスで評価した例は，読んだ感情特化のモデル 7 本（Emotion-LLaMA，AffectGPT，EmoVIT，EmoLLM，Emotion-Qwen，R1-Omni，SECap）のうち，はっきりしているのが 2 本（EmoVIT の held-out 7 データ，R1-Omni の RAVDESS），学習データとの関係がはっきりしないものが 3 本（Emotion-LLaMA の DFEW zero-shot，AffectGPT の MELD・IEMOCAP・MOSEI など，Emotion-Qwen の「zero-shot」行），学習と同じ手順で作ったデータでの評価だけが 2 本（EmoLLM，SECap）である．汎用の音声 LLM の SALMONN は，IEMOCAP の別の Session で評価する（同じコーパスの話者違い）．
- 教師ありの専門モデルとの比較は，ターゲットのデータで微調整したときに限って MLLM が並ぶか上回る（Emotion-LLaMA の DFEW 微調整 UAR 64.21 対 MAE-DFER 63.41）．学習に使っていないデータでは値が下がる（R1-Omni の RAVDESS の UAR は 43.00．Emotion-LLaMA の DFEW zero-shot の UAR 45.59 は微調整の 64.21 から 18.6 ポイント低い）．RAVDESS の教師ありの値は R1-Omni の本文に無く，差は出せない．
- 指標が本研究と違う．AffectGPT の hit rate は「出力のどこかに正解のラベルがあれば正解」なので，複数のラベルを出すほど得点が上がりうる．OV-MER は GPT-3.5 で同義語をまとめてから集合の重なりを取る．どちらも 4 クラスの UAR とは直接比べられない．UAR を報告するのは DFEW・MAFW・RAVDESS の動画の表情の評価（Emotion-LLaMA，Emotion-Qwen，R1-Omni）である．
- 本研究の 4 コーパスとの重なり：MELD と IEMOCAP は AffectGPT の評価と，SALMONN（IEMOCAP の学習），Qwen-Audio・Qwen2-Audio（MELD の評価）に出る．CMU-MOSEI は AffectGPT では感情ではなく極性（WAF）の評価で，AFEW は読んだ MLLM の評価に出なかった（GPT-4V with Emotion の SFEW 2.0 と EmoCLIP の AFEW は問い 3 と問い 4）．

### Gaps
- Emotion-LLaMA の DFEW の zero-shot の行が，DFEW を学習から外したモデルの値かは，本文の 2 か所が食い違い，確定できなかった．GitHub（ZebangCheng/Emotion-LLaMA）の README は読んでいない．
- AffectGPT と MER-UniBench の評価が，学習データに含まれないコーパスでの zero-shot といえるか（MELD・IEMOCAP・MOSEI が MER-Caption に入っていないか）は，読んだ範囲で明記が無い．
- OV-MER の付録 U（判別型の手法との比較）は読んでいない．
- 4 クラスの UAR で，MLLM を IEMOCAP・MELD・MOSEI・AFEW に zero-shot で当てた値を報告した感情特化の MLLM の論文は見つからなかった．

## 問い 3：汎用の MLLM（GPT-4V・GPT-4o，Gemini，Qwen-VL・Qwen2-Audio，LLaVA など）は感情のベンチマークで zero-shot と few-shot にどこまで届き，どう失敗するか

### Takeaway
汎用の MLLM の zero-shot は，読んだ比較のすべてで教師ありの最良を下回り，差は 1〜27 ポイントである（SFEW 2.0 の顔で 57.24 対 58.50，OmniVox の MELD の音声で 51.3 対 52.0，CREMA-D で 61.9 対 71.0，MER-MULTI の融合で 65.39 対 82.94，OmniVox の IEMOCAP の文章で 44.2 対 71.4）．報告された失敗は，neutral への偏り，出力の形式の崩れ，文章への依存，声の変動を読めないこと，安全審査による拒否，ラベル集合の外の語，学習データとの重なり（汚染）である．文脈内の例（few-shot）は，VILA では EMOTIC と HECO の両方で zero-shot より上がり，LLaVA の 2-shot，EmoVIT の held-out，推論の文を付けた例では上がらないか下がる．

### Cited Findings

**GPT-4V with Emotion（Lian ほか，arXiv 2312.04293）○**
- 顔の静止画（正解率）：SFEW 2.0 で GPT-4V 57.24（教師ありの最良 FaceCaps 58.50），RAF-DB 75.81（POSTER++ 92.21），AffectNet 42.77（63.77）．動画の表情（WAR）：DFEW 54.80（S2D 76.16），FERV39k 38.72，RAVDESS 34.31．微表情は無作為を下回る．MER-MULTI の融合 65.39（教師ありの MulT 82.94）— [arXiv 2312.04293](https://arxiv.org/abs/2312.04293)
- 失敗：人や暴力を含む画像で安全審査により回答を拒む．音声を入力できない（メルスペクトログラムの画像にしても答えられない）．10 回の繰り返しで予測が揺れる — [arXiv 2312.04293](https://arxiv.org/abs/2312.04293)

**EmoBench-M（Hu ほか，arXiv 2502.04424．GitHub の記載では MM 2026 採択）◎**
- 13 の場面で，基礎の感情認識は RAVDESS の歌（6 クラス）と話（8 クラス），CMU-MOSI（3 クラス），CMU-MOSEI（強度の 3 クラス），FMSA-SC．会話の場面は MER2023（6 クラス），CH-SIMS・CH-SIMS v2，MC-EIU，MELD（7 クラス）．各 250〜500 本で，すべて zero-shot，指標は正解率 — [arXiv 2502.04424](https://arxiv.org/html/2502.04424v3)
- 基礎の感情認識の平均：Gemini-3.0-Pro 69.3（RAVDESS の話 72.1），Gemini-2.0-Flash 61.4，公開モデルの最良は Qwen2-Audio-Instruct 59.9（RAVDESS の話 71.7），Emotion-LLaMA 36.9．評価したモデルに GPT-5.2 は入るが GPT-4o は入らない．人と教師ありの最良の値は示されず，「markedly inferior to human benchmarks」とだけ書く — [arXiv 2502.04424](https://arxiv.org/html/2502.04424v3)
- 推論（Thinking）の版は元のモデルを一貫しては上回らず，著者は感情の課題が推論より知覚に頼ると読む．感情特化の Emotion-LLaMA と AffectGPT について「emotion-specific priors alone are insufficient」と結論する — [arXiv 2502.04424](https://arxiv.org/html/2502.04424v3)

**OmniVox（arXiv 2503.21480）◎**
- Gemini-2.0-Flash・Flash-Lite，GPT-4o-Audio，Phi-4-Multimodal を，IEMOCAP（6 クラス）と MELD（7 クラス）で zero-shot に評価する．指標は重みつき F1 — [arXiv 2503.21480](https://arxiv.org/html/2503.21480v1)
- 音声だけの最良：IEMOCAP 55.9（GPT-4o-Audio，前の 12 発話の文脈つき），MELD 51.3（文脈なし）．微調整した音声モデルは IEMOCAP 48.1〜58.8，MELD 46.6〜52.0 で，最良の Shou ほかには両方で届かない．文章だけでは IEMOCAP 44.2，MELD 62.8 で，微調整の文章モデル（71.4〜72.6，67.6〜69.2）より低い — [arXiv 2503.21480](https://arxiv.org/html/2503.21480v1)
- 失敗：MELD で sadness の 64%，joy の 42% を neutral と答える．IEMOCAP で angry の 61% を frustrated と答える．音量や高さの平均は読めるが変動は読めない（IEMOCAP の音量の変動 F1 0.33，高さの変動 0.27）．文脈は IEMOCAP では効き，MELD の GPT-4o-Audio では 51.3→47.7 に下がる — [arXiv 2503.21480](https://arxiv.org/html/2503.21480v1)

**VoxEmo（Zhang ほか，arXiv 2603.08936，2026-03）◎**
- 35 コーパス・15 言語で Qwen2-Audio-7B-Instruct と Audio Flamingo 3 を zero-shot に評価する．プロンプトは直接，書き起こし付き，音の説明付き，両方，両方と推論の 5 種．分割は EmoBox に従い話者を分ける — [arXiv 2603.08936](https://arxiv.org/html/2603.08936)
- 抄録：「zero-shot speech LLMs trail supervised baselines in hard-label accuracy」．例：CREMA-D で Qwen2-Audio の直接のプロンプトが Macro-F1 61.9，教師ありの参照 71.0．LoRA（r=8）で SFT すると Qwen2-Audio は zero-shot の最良より平均 23.7 ポイント上がり，30 データ中 10 で EmoBox の参照を上回る — [arXiv 2603.08936](https://arxiv.org/html/2603.08936)
- 失敗：音の説明付きのプロンプトで Qwen2-Audio の出力の解析失敗が最大 93.2%．書き起こしを足すと演技のデータ 28 本中 22 本（Qwen2-Audio），25 本（AF3）で Macro-F1 が下がり，CREMA-D では最大 46.7 ポイント下がる．自然な発話の確率が「over-concentrated (over-confident)」．IEMOCAP→MELD では，IEMOCAP に MELD の disgust・fear・surprise が無いため 3.4 ポイント下がる．「AF3’s released training data explicitly includes SER-relevant corpora such as IEMOCAP and MELD」で，Qwen2-Audio の学習データは公開されていない — [arXiv 2603.08936](https://arxiv.org/html/2603.08936)

**文脈の中の感情（Large Vision-Language Models as Emotion Recognizers in Context Awareness，arXiv 2407.11300）◎**
- LLaVA-7B と VILA-8B を EMOTIC（26 ラベル）と HECO（8 ラベル）で比べる．few-shot は学習データから 200 本の候補を置き，人物の切り出しと場面の ViT の類似度で例を選ぶ — [arXiv 2407.11300](https://arxiv.org/html/2407.11300)
- EMOTIC（micro / macro F1）：学習した EMOT-Net 45.98 / 21.39，VILA zero-shot 30.84 / 21.49，VILA の SFT 41.94 / 19.76，VILA 10-shot 43.11 / 26.70．HECO（正解率）：EMOT-Net 38.80，zero-shot 35.56〜35.69，SFT 57.16〜60.82，VILA 10-shot 47.55，GPT-4V の推論の文を例に付けた 8-shot は 39.80 — [arXiv 2407.11300](https://arxiv.org/html/2407.11300)
- 失敗：LLaVA は複数の画像を扱えない．10 例を超えると下がる．推論の文を付けると HECO の正解率が下がる．VILA は EMOTIC の 26 ラベルに無い「Discomfort」を答えた — [arXiv 2407.11300](https://arxiv.org/html/2407.11300)

**その他**
- EmoLLM の表 3 で，GPT-4V は画像の閉じたラベル集合の感情で 57.90（EmoBench で学んだ EmoLLM は 75.03）— [arXiv 2406.16442](https://arxiv.org/html/2406.16442)
- OV-MER では GPT-4V が映像だけで F 55.51 で，評価した MLLM の最良（問い 2）— [arXiv 2410.01495](https://arxiv.org/html/2410.01495)
- GPT-4o と Gemini を NimStim の顔で比べた研究（GPT-4o 86%，Gemini 2.0 84%）と，GPT-4V を視覚の感情計算に当てた研究（arXiv 2403.05916）は，検索結果の要約だけで本文を読んでいない △ — [arXiv 2403.05916](https://arxiv.org/abs/2403.05916)

### Inferences
- 汎用の MLLM の zero-shot は，ラベル集合の写し方とプロンプトで値が大きく動く（VoxEmo の解析失敗 93.2%，OmniVox の文脈の有無で 3.6 ポイント）．zero-shot の値を比較相手に使うなら，プロンプト・ラベルの写し方・解析できない出力の扱いを固定して報告する必要がある．
- neutral への偏り（OmniVox），文章への依存（VoxEmo），声の変動を読めないこと（OmniVox，R1-Omni の「audio underuse」，OV-MER の「paralinguistic」）は，音声の情報を LLM が使い切れていないという同じ方向を示す．本研究の音声の識別器（凍結 WavLM の線形分類器）が持つ情報と，汎用 MLLM の音声の答えが持つ情報は，この点で違うと予想する（推測であり，確かめた研究は見ていない）．
- 学習データの汚染：IEMOCAP と MELD は AF3 の学習データに明記され，SALMONN の指示学習にも IEMOCAP が入る．Qwen2-Audio は非公開である．本研究のコーパスのうち 2 つで，音声の LLM の zero-shot の値は「見たことの無いコーパス」の値とはいえない可能性がある．
- 本研究との違い：これらの評価は，ラベルを使わない（zero-shot）か数例（few-shot）の推論だけで，ソースのコーパスのラベルとターゲットの K 本を両方使う設定ではない．クラス数は 6〜8 が多く，指標は正解率・重みつき F1・Macro-F1 で，UAR（VoxEmo の UA を除く）は少ない．

### Gaps
- GPT-4o・Gemini を IEMOCAP・MELD・MOSEI・AFEW の 4 クラスに zero-shot で当て，UAR を報告した研究は見つからなかった．
- 音声と映像の MLLM に K 本の例を文脈内に入れる few-shot を，MER のコーパスで系統的に測った研究は見つからなかった（画像の EMOTIC・HECO だけ）．
- MME-Emotion などほかの 2025 年のベンチマークは読んでいない．

## 問い 4：MLLM や LLM は，ドメイン適応や半教師ありの流れの中で道具として使われているか（疑似ラベル，記述文の生成，蒸留の教師，ラベル集合の写像）

### Takeaway
使われている．読んだ範囲では，(a) ラベル無しのターゲットに MLLM か LLM で疑似ラベルを付けて小さいモデルに学ばせる例が 5 本，(b) MLLM と GPT に感情の記述文を書かせて MLLM の学習データを作る例が 6 本と，クラスの記述文を書かせて類似度で分類する例が 1 本（EmoCLIP），(c) LLM の出力の語や分布を小さいモデルに蒸留する例が 3 本，(d) GPT・感情の輪・別名の表でラベル集合を写す例が 5 本ある．ソースのラベル付きコーパスとターゲットのラベル無しデータを両方使うドメイン適応で MLLM を教師にしたのは MI-Fuse の 1 本だけで，それも音声だけ・ターゲットのラベルなしの設定である．

### Cited Findings

**(a) 疑似ラベル**
- MI-Fuse（arXiv 2509.20706）◎：ソースで学んだ分類器（ソースのデータは適応のときには使えない）と，API でしか使えない音声言語モデル Gemini 2.5 Flash（温度 0.6）を教師にして，ラベル無しのターゲットの音声に疑似ラベルを付ける．Gemini には自然文で感情の確率分布を答えさせ 5 回の平均を取り，分類器は MC dropout 8 回の平均を取る．各教師の分布を相互情報量 MI で exp(−MI) の重みにして混ぜ，学生を学ぶ．分類器の教師は学生の EMA（0.999）．学生と分類器は WavLM base+ の層の重みつき和と線形 2 層．ラベルは 4 クラス（happy，sad，angry，neutral）．MSP-Podcast・MSP-IMPROV・IEMOCAP の 6 方向の平均 UAR（論文の語では unweighted accuracy）は，MI-Fuse 58.38，Gemini を教師にした source-free UDA 54.48，Gemini の zero-shot 53.69，ソースだけ 49.64，SHOT 50.62，NRC 51.53．学生は 6 方向のうち 5 方向で Gemini の zero-shot を上回り，IEMOCAP→MSP-Podcast だけ下回る（59.85 対 61.44）．限界として，共有の固定ラベル集合を仮定すること，API の費用と遅延を挙げる — [arXiv 2509.20706](https://arxiv.org/html/2509.20706)
- ACL-MER（Zhuang ほか，IJCNLP-AACL 2025）◎：音声は Qwen2-Audio-7B-Instruct，映像は Qwen2-VL-7B-Instruct，文章は Qwen2.5-7B-Instruct をモダリティごとの教師にし，思考の連鎖のプロンプトで感情と 1〜5 の自己申告の確信度を出させる．学生は凍結した HuBERT-large・RoBERTa-large・CLIP-large の特徴の上の融合モデル（注意，MMIM，LMF）．確信度で学生の確率を直し，確信度の高い標本をクラスごとに最大 200 本ずつラベル付きの集合に移して繰り返す．MER2023（6 クラス）と CHERMA（7 クラス）で，ラベル付きを 100・200・500・1000 本に減らし，ラベル無しは MER2023 で 73,148 本，CHERMA で 15,918 本．指標は重みつき F1 と正解率 — [ACL Anthology 2025.ijcnlp-long.152](https://aclanthology.org/2025.ijcnlp-long.152)
  - 教師の確信度と正しさ（表 1，MER2023 の test）：確信度 5 の標本は Qwen2-Audio で全体の 11.3%，正解率 0.7872．確信度 4 以上に広げると 98.5%，0.5124．Qwen2-VL は 24.3%・0.7734 と 85.2%・0.3653 — [ACL Anthology](https://aclanthology.org/2025.ijcnlp-long.152)
  - 主の表（表 3，MER2023，注意の融合，重みつき F1）：ラベル 100 本で，疑似ラベルなし 0.4042，教師だけの疑似ラベル 0.6626，学生だけの疑似ラベル 0.5320，ACL-MER 0.6753．ラベル 1000 本では，疑似ラベルなし 0.8213，教師だけ 0.8094，ACL-MER 0.8416．ラベルが 1000 本あると，MLLM の疑似ラベルをそのまま使うと疑似ラベルなしより下がる — [ACL Anthology](https://aclanthology.org/2025.ijcnlp-long.152)
  - 3 つのモダリティ別の教師を 1 つの Qwen2.5-Omni-3B に替えると下がる（CHERMA，ラベル 500 本で 0.6328 対 0.6216）— [ACL Anthology](https://aclanthology.org/2025.ijcnlp-long.152)
- SZTU-CMU の MER2024 の解法（Cheng ほか，arXiv 2408.10500）◎：Emotion-LLaMA でラベル無しの 20,000 本に疑似ラベルを付け，5,030 本のラベル付きと合わせて Conv-Attention の融合（Chinese-HuBERT，CLIP，VideoMAE，Qwen1.5-32B，Baichuan-13B の特徴）を学ぶ．疑似ラベルの割合を 20% にすると MER-NOISE の WAF 79.88%，100% で 83.87%．確信度による選別は書かれていない — [arXiv 2408.10500](https://arxiv.org/html/2408.10500)
  - 食い違い：既存のノート（`research_notes/先行研究_Albanieの後続研究_2026-11-04/report_B_少数ラベルとコーパス間DA.md`）は「疑似ラベルを 100% 使うのは最適ではないと書く」とする．今回読んだ本文では 100% が最高で，著者は「while using 100% of pseudo-labeled data can enhance performance, it may also introduce some noise」「The optimal ratio may depend on the quality of the pseudo-labels」と注意を書くだけである — [arXiv 2408.10500](https://arxiv.org/html/2408.10500)
- LanSER（Gong ほか，Interspeech 2023）◎：People's Speech（4,321,002 発話）と Condensed Movies（1,030,711 発話）を Whisper-small で書き起こし，MNLI で微調整した RoBERTa の含意判定で 43 種のラベル（BRAVE-43 を改変）から 1 つを選んで弱いラベルにする．メルスペクトログラムの ResNet-50 を弱いラベルで事前学習し，IEMOCAP（4 クラス，話者を分けた 6:2:2）と CREMA-D（6 クラス）で微調整する．指標は unweighted accuracy — [arXiv 2309.03978](https://arxiv.org/pdf/2309.03978)
  - IEMOCAP：学習データの 10% で，事前学習なし 38.5%，LanSER（Condensed Movies）50.0%．100% で 47.6% と 54.5%．無作為のラベルで事前学習すると 47.6%．書き起こしに含意判定を直接当てると 47.5% — [arXiv 2309.03978](https://arxiv.org/pdf/2309.03978)
  - CREMA-D（12 文を 6 感情で読む演技のデータ）：含意判定だけでは 16.1% で多数派の 17.1% より低いが，LanSER は 100% で 58.7%（事前学習なし 57.2%）．著者はこれを，文章由来のラベルからでも韻律の表現を学べた証拠とする — [arXiv 2309.03978](https://arxiv.org/pdf/2309.03978)
- Feng & Narayanan（arXiv 2309.08108）○：Whisper の書き起こしを LLM（Flan-T5 XXL，LLaMa2，Falcon）に読ませて IEMOCAP・MELD などに感情ラベルを付け，音声感情認識を学ぶ．IEMOCAP の 4 クラスで LLM のラベルの UAR 43.68〜51.60%，真のラベル 71.78%．MELD では複数 LLM の多数決 53.90%，真のラベル 54.32%．著者は「LLMs struggle to provide correct emotion labels for SER training, leading to a 10-20% decrease」と書く — [arXiv 2309.08108](https://arxiv.org/abs/2309.08108)

**(b) 記述文の生成（学習データを作る道具）**
- Emotion-LLaMA の MERR は Qwen-Audio・MiniGPT-v2・LLaMA-3，AffectGPT の MER-Caption は SALMONN・Chat-UniVi・GPT-3.5，OV-MERD は GPT-4V・SALMONN・GPT-3.5，EmoVIT は BLIP2・GPT-4，EmoLLM の EmoPrompt は GPT-4V，Emotion-Qwen の VER は Qwen-VL-Max・GPT-4・DeepSeek で書かれる — [arXiv 2406.11161](https://arxiv.org/html/2406.11161)；[arXiv 2501.16566](https://arxiv.org/html/2501.16566)；[arXiv 2410.01495](https://arxiv.org/html/2410.01495)；[arXiv 2404.16670](https://arxiv.org/html/2404.16670)；[arXiv 2406.16442](https://arxiv.org/html/2406.16442)；[arXiv 2505.06685](https://arxiv.org/html/2505.06685)
- EmoCLIP（arXiv 2310.16640）○：ChatGPT に書かせて人が直したクラスの記述文と動画の埋め込みの類似度で分類する．AFEW の 7 クラス（検証セット）で，CLIP の zero-shot UAR 23.05，MAFW の記述で学んだ EmoCLIP の zero-shot 36.13 — [arXiv 2310.16640](https://arxiv.org/abs/2310.16640)

**(c) 蒸留の教師**
- Contrastive Distillation of Emotion Knowledge from LLMs（arXiv 2505.18040）◎：GPT-4（gpt-4-1106-preview）が GoEmotions の約 43.4k 文に自由な感情の語（2,173 種）を付け，BERT の文エンコーダと BERT のラベルエンコーダを対照学習で揃える（GPT-4 の「10,000 times smaller」）．人のラベルは使わない．推論のときにラベル集合を各データの集合に差し替えて zero-shot で評価する．Macro-F1 は SemEval-2018 で学生 0.480・GPT-4 0.486・BERT の微調整 0.563，ISEAR で 0.479・0.728・0.677．文章だけ — [arXiv 2505.18040](https://arxiv.org/html/2505.18040v1)
- MI-Fuse と ACL-MER も，大きいモデルの出力を小さい学生（WavLM base+，凍結特徴の融合モデル）に移す点で蒸留にあたる（上の (a)）．

**(d) ラベル集合の写像**
- MER2024 の MER-OV と OV-MER は GPT-3.5 で同義語をまとめてから得点を取る — [arXiv 2404.17113](https://arxiv.org/html/2404.17113)；[arXiv 2410.01495](https://arxiv.org/html/2410.01495)
- MER2025 の MER-FG は感情の輪でラベルを群にまとめる — [arXiv 2504.19423](https://arxiv.org/html/2504.19423)
- VoxEmo は出力のラベルを別名の表で正規化し，写せない出力は誤りとして数える — [arXiv 2603.08936](https://arxiv.org/html/2603.08936)
- Contrastive Distillation は推論のときにラベル集合を差し替える — [arXiv 2505.18040](https://arxiv.org/html/2505.18040v1)

### Inferences
- 本研究の設定に最も近いのは ACL-MER と MI-Fuse である．
  - ACL-MER は，凍結した HuBERT・RoBERTa・CLIP の特徴の上の小さい融合モデル，少ないラベル（100〜1000 本），多数のラベル無しという点で本研究と同じ形をしている．違いは，ソースのコーパスを使わない単一コーパスの半教師あり学習であること，中国語のコーパス（MER2023，CHERMA），6〜7 クラス，重みつき F1 と正解率，ラベルがクラスあたり K 本ではなく元の分布に従う標本であることである．100 本は本研究の K=25（4 クラスで 100 本）と本数が同じだが，クラスの数と配分が違う．
  - MI-Fuse は，4 クラス，UAR，IEMOCAP を含む，ソースからターゲットへの適応という点で本研究と同じ形をしている．違いは，ターゲットのラベルを 1 本も使わないこと，ソースのデータを適応のときに使わないこと（source-free），音声だけであること，学生が WavLM を微調整することである．
- ACL-MER の表 3 は，MLLM の疑似ラベルの価値がターゲットのラベルの数で変わることを示す．ラベル 100 本では教師だけの疑似ラベルで重みつき F1 が 0.40 から 0.66 に上がり，1000 本では 0.82 から 0.81 に下がる．本研究の K の軸（ラベルが増えると外の情報の価値が減る）と同じ向きの観察である．
- 確信度で疑似ラベルを絞る方法は 2 本（MI-Fuse の相互情報量，ACL-MER の自己申告の確信度）で，SZTU-CMU は絞らずに割合だけを変える．MLLM の自己申告の確信度は，確信度 5 で正解率 0.79 でも全体の 11% しか拾えない（ACL-MER の表 1）．
- LLM の文章だけのラベル（LanSER，Feng & Narayanan）は IEMOCAP で真のラベルより大きく低い（LanSER の含意判定 47.5%，Feng & Narayanan の 43.68〜51.60% 対 71.78%）．MELD では真のラベルに近い（53.90% 対 54.32%）．文章の情報が強いコーパスで LLM のラベルの価値が大きいという点は，本研究のコーパスごとの言葉の強さの違いと照らせる．
- ソースのラベル付きコーパス 2〜3 本と，ターゲットの K 本とラベル無しを同時に使い，そこに MLLM を教師として加えた研究は見つからなかった．

### Gaps
- ACL-MER の教師の zero-shot の全体の正解率（確信度で絞らない値）と，付録 D・E は読んでいない．
- MI-Fuse がターゲットの少数ラベルを加えた場合の結果は無い（設定に無い）．
- 「LLM でコーパス間のラベル集合を写して，ソースのラベルをターゲットのラベル集合に合わせる」研究（例えば 7 クラスの MELD を 4 クラスに写す手順を LLM に任せる研究）は見つからなかった．読んだ論文はすべて評価の採点のための写像である．
- Santoso ほか（ICASSP 2024）の LLM による音声の感情の注釈は，既存のノートと同じく本文を読んでいない．

## 問い 5：これらの研究で「標準の手法」は何で，各選択はどれくらい多いか

### Takeaway
感情特化の MLLM の標準の作り方は，「凍結した HuBERT か Whisper の音声エンコーダと CLIP 系 ViT の映像エンコーダ ＋ 学習する射影層か Q-Former ＋ LoRA を足した 7B 級の LLM（LLaMA 系か Qwen 系）を，MLLM と GPT が書いた記述文で指示学習する」である．評価の標準は MER2023・MER2024・DFEW で，学習に使っていないコーパスで UAR を測る評価は少ない．ドメイン適応の中での MLLM の標準の使い方はまだ定まっておらず，疑似ラベルの教師として使い，確信度で絞って小さいモデルに学ばせる形が 2025 年に出てきた段階である．

### Cited Findings
- 「凍結エンコーダ＋射影層か Q-Former＋LoRA」：Emotion-LLaMA（LoRA r=64，34M）— [arXiv 2406.11161](https://arxiv.org/html/2406.11161)；AffectGPT（LoRA r=16，射影層，pre-fusion）— [arXiv 2501.16566](https://arxiv.org/html/2501.16566)；SALMONN（LoRA r=8，Q-Former，33M）— [arXiv 2310.13289](https://arxiv.org/html/2310.13289)；EmoLLM（LoRA．部分の記載なし）— [arXiv 2406.16442](https://arxiv.org/html/2406.16442)
- 「Q-Former か射影層だけ」：EmoVIT — [arXiv 2404.16670](https://arxiv.org/html/2404.16670)；SECap — [arXiv 2312.10381](https://arxiv.org/html/2312.10381)
- 「段ごとにエンコーダか LLM の全体を学ぶ」：Qwen-Audio — [arXiv 2311.07919](https://arxiv.org/html/2311.07919)；Emotion-Qwen（事前学習の段 3 で LLM を解き，指示学習は複数 LoRA）— [arXiv 2505.06685](https://arxiv.org/html/2505.06685)
- 「強化学習（RLVR）」：R1-Omni — [arXiv 2503.05379](https://arxiv.org/html/2503.05379)
- MER の挑戦の基準の手法は 2024 年も 2025 年も MLLM ではなく，凍結した特徴（HuBERT-large，CLIP-large，Baichuan-13B など）の注意による融合である．MLLM は自由語彙と細かい感情の部門の基準にだけ使う — [arXiv 2404.17113](https://arxiv.org/html/2404.17113)；[arXiv 2504.19423](https://arxiv.org/html/2504.19423)
- MLLM を疑似ラベルの教師にした研究：MI-Fuse — [arXiv 2509.20706](https://arxiv.org/html/2509.20706)；ACL-MER — [ACL Anthology](https://aclanthology.org/2025.ijcnlp-long.152)；SZTU-CMU — [arXiv 2408.10500](https://arxiv.org/html/2408.10500)．文章の LLM のラベル：LanSER — [arXiv 2309.03978](https://arxiv.org/pdf/2309.03978)；Feng & Narayanan — [arXiv 2309.08108](https://arxiv.org/abs/2309.08108)

### Inferences

**数え上げ（読んだ論文の範囲での数．母集団の割合ではない）**

| 選択 | 本数 | 該当 |
|---|---|---|
| 音声エンコーダが HuBERT | 3 / 8（音声を使う 8 本中） | Emotion-LLaMA，AffectGPT，SECap |
| 音声エンコーダが Whisper | 4 / 8 | EmoLLM，Qwen-Audio，Qwen2-Audio，SALMONN |
| 映像エンコーダが CLIP 系 ViT | 3 / 4（映像を使いエンコーダの名前のある 4 本中） | AffectGPT，EmoLLM，Emotion-Qwen（残る 1 本は EVA と MAE・VideoMAE の Emotion-LLaMA） |
| LLM が LLaMA 系 | 5 / 9（名前のある 9 本中） | Emotion-LLaMA，EmoLLM，EmoVIT，SALMONN，SECap |
| LLM が Qwen 系 | 4 / 9 | AffectGPT，Emotion-Qwen，Qwen-Audio，Qwen2-Audio |
| LLM の大きさが 7B 級 | 7 / 10 | SALMONN（13B），R1-Omni（0.5B），AffectGPT（大きさ未確認）以外 |
| エンコーダ凍結＋射影か Q-Former＋LoRA | 4 / 10 | 上の Cited Findings |
| 学習データを MLLM・GPT の記述で作る | 5 / 7（感情特化 7 本中．SECap は人の説明文，R1-Omni は未確認） | Emotion-LLaMA，AffectGPT，EmoVIT，EmoLLM，Emotion-Qwen |
| 学習に使っていないコーパスでの評価が明確 | 2 / 7 | EmoVIT，R1-Omni |
| 主の評価に UAR を含む | 3 / 7 | Emotion-LLaMA，Emotion-Qwen，R1-Omni（いずれも DFEW・MAFW・RAVDESS の表情） |
| DA・半教師ありで MLLM か LLM を疑似ラベルの教師にする | 5 本 | MI-Fuse，ACL-MER，SZTU-CMU，LanSER，Feng & Narayanan |
| うち確信度か不確かさで絞る | 2 / 5 | MI-Fuse，ACL-MER |
| うちソースのコーパスのラベルも使う | 1 / 5 | MI-Fuse（ソースで学んだ分類器だけ） |

**論文ごとの本研究との違い**

| 論文 | 本研究との主な違い |
|---|---|
| Emotion-LLaMA | 7B の生成モデルを LoRA で学ぶ．中国語の MER2023 と DFEW で学び評価する．ソースとターゲットの区別も K 本の設定も無い．指標は F1・UAR/WAR・ChatGPT の採点 |
| AffectGPT | 115,595 本の記述文で学ぶ．評価は hit rate（出力にラベルがあるか）と集合の F で，UAR ではない．MELD・IEMOCAP・MOSEI を含むが，ターゲットのラベルで適応しない |
| OV-MER | 決まったラベル集合を捨てる課題で，4 クラスの分類と目的が違う |
| EmoVIT・EmoLLM・Emotion-Qwen | 画像か動画の視覚が中心で音声が無いか弱い．評価は同じ手順で作ったデータか，近いデータ |
| R1-Omni | 0.5B の全体を強化学習で学ぶ．OOD は RAVDESS の 1 本だけ |
| Qwen-Audio・Qwen2-Audio・SALMONN | 汎用の音声 LLM．感情は多タスクの 1 つで，MELD か IEMOCAP の 1 つの値だけ報告する |
| SECap | 中国語の社内データで説明文を生成する．分類もコーパス間の評価も無い |
| GPT-4V with Emotion・EmoBench-M・OmniVox・VoxEmo | ラベルを使わない推論の評価．ソースのラベルもターゲットの K 本も使わない．クラス数 6〜8 が多い |
| LVLM の文脈の感情（2407.11300） | 画像の few-shot の文脈内学習で，K 本の例をプロンプトに入れる点だけが本研究の K 本に近い．学習はしない |
| MI-Fuse | 4 クラス・UAR・IEMOCAP を含む点が近い．ターゲットのラベル 0 本，ソースのデータを使わない，音声だけ，WavLM を微調整する点が違う |
| ACL-MER | 凍結の HuBERT・RoBERTa・CLIP の特徴，少数ラベル＋ラベル無しの点が近い．ソースのコーパスが無い単一コーパス，中国語，6〜7 クラス，重みつき F1 の点が違う |
| SZTU-CMU（MER2024） | 単一コーパス（MER2024）の半教師あり．ラベル付きは 5,030 本で，少数ラベルではない |
| LanSER・Feng & Narayanan | 書き起こしの文章だけから付けたラベルで音声モデルを学ぶ．LanSER はターゲット外の大量の音声で事前学習してから微調整する |
| Contrastive Distillation | 文章だけ．ラベル集合を差し替える zero-shot の評価 |

- まとめると，「MLLM を使って新しいコーパスに汎化する」と書く論文の多くは，(1) 記述文で指示学習した生成モデルを，学習データと近いコーパスで zero-shot か微調整で評価するか，(2) 汎用 MLLM の zero-shot を測るかのどちらかである．本研究のように，ターゲットの K 本のラベルとソースの複数コーパスのラベルの組み合わせで UAR の得と損を測る設計は，読んだ範囲には無かった．MLLM の出力を本研究の設定に持ち込む先例は，疑似ラベルの教師として小さいモデルに移す 2 本（MI-Fuse，ACL-MER）に限られる．

### Gaps
- 数え上げは今回読んだ 20 本ほどに限り，2023〜2026 年の論文全体の割合ではない．MER2024・MER2025 の上位チームの解法論文を網羅して数えると，「MLLM の疑似ラベル」や「LLM の文章特徴」の割合が変わる可能性がある．
- 2026 年の感情特化の MLLM（AffectGPT-R1 や Emotion-LLaMA の後継など）は検索で確かめていない．
