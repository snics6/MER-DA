# MER-DA で新しい学習法を作るときに使える材料と制約（リポジトリの棚卸し．2026-10-28）

調べた範囲は `/home/snics/MER-DA` の文書とコードと発話一覧であり，ウェブは使っていない．生データ `/mnt/d/Databases/` は `data/README.md` 4 行目が「読むだけ」と定めた MER-DA の入力なので，評定の票があるかを確かめるために一覧と先頭の数行だけを読んだ．出典は「ファイル:行」で書く．行番号の無い出典（発話一覧の列の集計）は，2026-10-28 に CSV の列を数えた値である．

---

## 1. データ：7 群のラベルの出所，話者，録音の種類，モダリティ，件数，感情の割合

### Takeaway
7 群のラベルは，演技者への指示（RAVDESS・CREMA-D のファイル名，AFEW のフォルダ名，IEMOCAP 台本の場面設定）と，評定者の判断（IEMOCAP の多数決，MELD の多数決，MOSEI の強さのしきい値）が混ざっている．解析に使うのは硬い 4 クラスのラベル 1 つだけである．IEMOCAP の評定者ごとのラベルと CREMA-D の知覚の票は，生データにはあるが発話一覧には取り込んでいない．CREMA-D と RAVDESS のラベルは，意図した感情（ファイル名）である．

### Cited Findings

**群とモダリティ**
- 群は IEMOCAP-impro / IEMOCAP-script / MOSEI / MELD / AFEW / RAVDESS / CREMA-D の 7 つ．映像は 6 群（RAVDESS は特徴を作っていない），音声は 7 群，言葉は 4 群（AFEW は転写が無く，RAVDESS と CREMA-D は読む文が 2 種類と 12 種類）で使える．連結は，音声＋映像が 6 群，言葉を含む組が 4 群である — [data/README.md:63-80](/home/snics/MER-DA/data/README.md)
- 転写はすべて人が書いたもので，音声認識の出力ではない — [data/README.md:202-210](/home/snics/MER-DA/data/README.md)

**測定に入る件数と感情の数（音声 WavLM，`load_group` の後）**
- 発話数（WavLM）：IEMOCAP-impro 2,943，IEMOCAP-script 2,588，MOSEI 18,919，MELD 11,343，AFEW 795，RAVDESS 672，CREMA-D 4,900．映像（ResNet-50）では顔を検出できなかった発話が抜ける（例：MOSEI 18,634） — [data/README.md:86-94](/home/snics/MER-DA/data/README.md)
- 話者群と感情の数（Neutral／Happy／Sad／Angry）：IEMOCAP-impro 10 群・1,099／947／608／289，IEMOCAP-script 10・609／689／476／814，MOSEI 2,626・8,847／6,790／1,830／1,452，MELD 283・6,430／2,305／1,002／1,606，AFEW 282・207／213／178／197，RAVDESS 24・96／192／192／192，CREMA-D 91・1,087／1,271／1,271／1,271 — [data/README.md:99-107](/home/snics/MER-DA/data/README.md)
- MOSEI の話者群 2,626 の多くは人物ではなく動画 ID である．チャンネルが分かる動画は 3,225 本中 1,122 本（35%）だけで，話者の分離は不完全である — [data/README.md:109](/home/snics/MER-DA/data/README.md)，[data/README.md:216-220](/home/snics/MER-DA/data/README.md)．発話一覧の `speaker_source` は `youtube_channel` 6,722 行，`video_id_fallback` 16,134 行（全 22,856 行）である — `data/metadata/mosei_utterance_manifest.csv` の列の集計
- 発話一覧の全行と使う行：IEMOCAP 10,039／5,531，MOSEI 22,856／18,919，MELD 13,708／11,352，AFEW 1,156／795，RAVDESS 1,440／672，CREMA-D 7,442／4,900．発話一覧は 4 クラスに入らない感情の行も持つが，特徴は `label_4` のある行だけから作る — [data/README.md:140-154](/home/snics/MER-DA/data/README.md)

**ラベルの出所**
- IEMOCAP：注釈ファイルの 1 行目の代表ラベル（多数決）を使う．neu→Neutral，hap と exc→Happy，sad→Sad，ang→Angry．fru（苛立ち）は Angry に入れない．xxx（注釈者が一致しなかったもの）・sur・fea・dis・oth は採らない — [features/build_manifest_iemocap.py:14-30](/home/snics/MER-DA/features/build_manifest_iemocap.py)，[data/README.md:188](/home/snics/MER-DA/data/README.md)
- IEMOCAP の `emotion_raw` の内訳（全 10,039 行）：neu 1,708，xxx 2,507，fru 1,849，ang 1,103，sad 1,084，hap 595，exc 1,041，sur 107，fea 40，oth 3，dis 2 — `data/metadata/iemocap_utterance_manifest.csv` の列の集計
- IEMOCAP の生の注釈ファイルには，評定者ごとの感情のラベル（`C-E2`，`C-E3`，`C-E4`，`C-F1` など．1 人が複数の感情を付ける行もある）と，評定者ごとの価数・覚醒度・支配度（`A-E3`，`A-E4`，`A-F1`）がある — `/mnt/d/Databases/IEMOCAP_full_release/Session1/dialog/EmoEvaluation/Ses01F_impro01.txt` の 1〜16 行目（この 1 ファイルの先頭 2 発話だけを見た）
- MOSEI：CMU 配布の 6 感情の強さ（0〜3）のうち 0.5 以上の感情が 1 つならその感情，2 つ以上なら `multi_label`（4 クラスに入れない），0 個なら `neutral` とする．**MOSEI の Neutral は消去法で決まり，注釈者が付けた名前ではない** — [data/README.md:189](/home/snics/MER-DA/data/README.md)，[data/README.md:195-196](/home/snics/MER-DA/data/README.md)，[features/build_manifest_mosei.py:8-34](/home/snics/MER-DA/features/build_manifest_mosei.py)．`multi_label` は 2,930 行である — `data/metadata/mosei_utterance_manifest.csv` の列の集計
- MELD：公式 CSV の `Emotion` 列をそのまま使う（注釈者の多数決で決まっている）．surprise・disgust・fear は採らない — [features/build_manifest_meld.py:12-18](/home/snics/MER-DA/features/build_manifest_meld.py)．公式 CSV の列は `Sr No., Utterance, Speaker, Emotion, Sentiment, Dialogue_ID, Utterance_ID, Season, Episode, StartTime, EndTime` で，評定者ごとの列は無い — `/mnt/d/Databases/MELD/Labels/dev_sent_emo.csv` の 1 行目
- AFEW：公式のフォルダ名（感情別）がラベルである — [data/README.md:191](/home/snics/MER-DA/data/README.md)，[features/build_manifest_afew.py:10-11](/home/snics/MER-DA/features/build_manifest_afew.py)
- RAVDESS：ファイル名の 3 番目の数字から付ける（01→Neutral，03→Happy，04→Sad，05→Angry．02 calm は Neutral に混ぜない）．CREMA-D：ファイル名の 3 番目の項目から付ける（NEU・HAP・SAD・ANG．DIS・FEA は空）．**どちらも演技者に指示した感情であり，知覚された感情ではない** — [data/README.md:192-193](/home/snics/MER-DA/data/README.md)，[features/build_manifest_extra.py:12-45](/home/snics/MER-DA/features/build_manifest_extra.py)
- CREMA-D の生データには，評定者の知覚の票（`tabulatedVotes.csv`：感情ごとの票 A・D・F・H・N・S，`numResponses`，`agreement`，`emoVote`，感情ごとの強さの平均）と，評定の生の記録（`finishedResponses.csv`，`finishedEmoResponses.csv`，`summaryTable.csv`）がある．例として，意図が `1001_IEO_HAP_LO` の発話は 9 票のうち 6 票が N で，`emoVote` は N，`agreement` は 0.667 である — `/mnt/d/Databases/CREMA-D_repo/processedResults/tabulatedVotes.csv` の 1〜3 行目
- methods.md も「発話一覧には注釈者ごとのラベルの列が無い．作れるのは IEMOCAP（注釈ファイル）と CREMA-D（評価者の票）の見込み（未確認）」と書いている．注釈者の一致率の参考値として「Larsen 2026（IEMOCAP で全員一致は約 20%）」を挙げる — [docs/methods.md:379-382](/home/snics/MER-DA/docs/methods.md)
- ラベルの定義がコーパスの間で違うこと（MOSEI の Neutral は消去法，IEMOCAP と MELD は注釈者の判定，AFEW はフォルダ）は，論文で限界として書くと決めてある — [data/README.md:196](/home/snics/MER-DA/data/README.md)，[data/README.md:473](/home/snics/MER-DA/data/README.md)

**話者と性別**
- IEMOCAP は 10 人（`Ses01_F`〜`Ses05_M`）で，`utterance_speaker` は F 4,800 行，M 5,239 行（全 10,039 行）である — `data/metadata/iemocap_utterance_manifest.csv` の列の集計．`marker_side` はマーカーを付けた人の性別であり，発話者ではない — [features/build_manifest_iemocap.py:44-53](/home/snics/MER-DA/features/build_manifest_iemocap.py)
- RAVDESS は俳優 24 人で，番号が奇数なら男性，偶数なら女性 — [features/build_manifest_extra.py:25-34](/home/snics/MER-DA/features/build_manifest_extra.py)
- AFEW の発話一覧は `actor_name`，`actor_age`，`gender`（Male 704，Female 452．全 1,156 行），`pose`（Frontal 742，Non-Frontal 414），`movie_title`（95 作品）を持つ — `data/metadata/afew_utterance_manifest.csv` の列の集計．話者は公式 XML の `NameOfActor`（俳優名）で，対応がつかない 4 本だけが作品名である — [data/README.md:234](/home/snics/MER-DA/data/README.md)
- CREMA-D の発話一覧に性別の列は無い．生データの `VideoDemographics.csv` に `ActorID, Age, Sex, Race, Ethnicity` がある — `/mnt/d/Databases/CREMA-D_repo/VideoDemographics.csv` の 1〜3 行目
- MELD の話者は登場人物の名前（304 種）で，上位 6 人が 82.6% を占める．合同発話（`Ross_and_Rachel` など）は落とす — [features/build_manifest_meld.py:20-26](/home/snics/MER-DA/features/build_manifest_meld.py)，[data/README.md:46-48](/home/snics/MER-DA/data/README.md)

**録音の種類**
- RAVDESS と CREMA-D は，俳優が決まった文を感情だけ変えて読むコーパスであり，検出力を上げるために足した — [features/build_manifest_extra.py:3-10](/home/snics/MER-DA/features/build_manifest_extra.py)
- IEMOCAP の台本の場面では 10 人全員が同じ台詞を読むので，即興と台本を別の群にする — [data/README.md:49-50](/home/snics/MER-DA/data/README.md)
- MOSEI は YouTube の動画，MELD は Friends，AFEW は映画の切り抜きである — [features/build_manifest_mosei.py:38](/home/snics/MER-DA/features/build_manifest_mosei.py)，[features/build_manifest_meld.py:22-23](/home/snics/MER-DA/features/build_manifest_meld.py)，[features/build_manifest_afew.py:22-31](/home/snics/MER-DA/features/build_manifest_afew.py)
- 地図の得のばらつきを分けた E8 では，音声 7 群の K=5 で「演技の録音か，自然な会話か」を分ける 1 本の軸が見つかった — [docs/results.md:160](/home/snics/MER-DA/docs/results.md)，[docs/results.md:209](/home/snics/MER-DA/docs/results.md)

**エンコーダ**
- 映像の第 1 は Ryumina らの ResNet-50（VGGFace2 で学習し AffectNet で微調整したもの．512 次元），第 2 は CLIP ViT-L/14（1024 次元）．音声は WavLM-large と HuBERT-large（1024 次元），言葉は RoBERTa-base と DeBERTa-v3-base（768 次元）．どれも本研究の 7 群では微調整していない — [data/README.md:245-254](/home/snics/MER-DA/data/README.md)，[data/README.md:402-404](/home/snics/MER-DA/data/README.md)
- 特徴は時間方向（言葉では単語方向）の平均だけで，長さは別の列（映像 `n_segments`，音声 `duration_sec`，言葉 `n_token`）に記録する — [data/README.md:263-265](/home/snics/MER-DA/data/README.md)

### Inferences
- 感情の割合は群によって大きく違う．上の数から計算すると，Neutral の割合は MELD 56.7%（6,430／11,343），MOSEI 46.8%（8,847／18,919），IEMOCAP-impro 37.3%，CREMA-D 22.2%，RAVDESS 14.3% であり，Angry は IEMOCAP-impro で 9.8%（289／2,943），IEMOCAP-script で 31.5%（814／2,588）である．RAVDESS と CREMA-D は感情がほぼ均等になるよう設計されている．
- 「意図したラベル」と「知覚されたラベル」の違いを扱う候補は，CREMA-D の票（生データにある）を使えば 1 群の中で比べられる．IEMOCAP では評定者ごとのラベルから評定の割れ（soft label）を作れる．ほかの 4 群（MOSEI，MELD，AFEW，RAVDESS）では，リポジトリ内に評定者ごとの情報が見当たらない．
- MOSEI の `int_*` の値は 0.333，0.667，0.1667 のような刻みをとる（列の集計）．複数の評定の平均である可能性が高いが，評定者の人数はリポジトリに書かれていない．
- 性別で分けた解析は，IEMOCAP（`utterance_speaker`），AFEW（`gender`），RAVDESS（番号の奇偶），CREMA-D（生データの `VideoDemographics.csv`）でできる．MELD と MOSEI では性別の列が無い．

### Gaps
- 各コーパスの評定者の人数と一致度（κ など）は，リポジトリの文書に書かれていない．IEMOCAP については methods.md が他文献の「全員一致は約 20%」を引くだけである．
- 言語は文書に明記されていない（転写の出所の表 [data/README.md:202-210](/home/snics/MER-DA/data/README.md) は言語を書いていない）．すべて英語のコーパスかどうかを，リポジトリ内では確かめられなかった．
- RAVDESS の知覚の評定（妥当性確認の評定）が生データにあるかは確かめていない．
- IEMOCAP の評定者の人数が発話によって違うかは，1 ファイルの先頭しか見ていないので分からない．

---

## 2. 測れる量：指標の定義の場所と，計算済みの結果の場所

### Takeaway
ずれの指標（PAD，MMD，フレシェ距離，coverage・density，共分散の支配 c，感情の割合の差，R_src，Δ_CE，BER）は `measure/shift.py` に，仕組みの量（d，o，λ，e，q，n，E_0，J，dprime）は `docs/exploratory-analyses.md` の各 E 節と `exp/` の台本にある．ずれの指標は 8 種類すべてで `results/shift_<kind>/` に計算済みであり，仕組みの量は `results/explore_relation/` のスナップショットと `results/check_mh/` にある．

### Cited Findings

**ずれの指標（4 つの軸）と実装**
- 共通の決まり：入力の指標は両側を同じ本数（既定 300 本）に抜き出し，同じ群を話者で 2 つに割った「ゼロ点」を必ず併記する．共分散を使う指標は PCA で 64 次元に落とす．coverage の近傍数は 5 — [measure/shift.py:1-46](/home/snics/MER-DA/measure/shift.py)
- 軸 1・2 を測る空間は logit（ソースの感情分類器の 4 つの点数）である．8 種類すべてで規則により logit が選ばれた — [docs/researchvision.md:482](/home/snics/MER-DA/docs/researchvision.md)，[docs/methods.md:194](/home/snics/MER-DA/docs/methods.md)
- 軸 1（入力の違い）：PAD（主）`shift.pad`（measure/shift.py:93），MMD の不偏推定 `mmd2_unbiased`（127），フレシェ距離 `frechet`（147） — [docs/methods.md:243-288](/home/snics/MER-DA/docs/methods.md)
- 軸 2（覆い方．向きがある）：coverage と density `density_coverage`（159），偶然の期待値 `expected_coverage`（178），共分散の支配 c `cov_dominance`（185）．転移指数 γ は保留 — [docs/methods.md:289-318](/home/snics/MER-DA/docs/methods.md)
- 軸 3（感情の割合）：割合の差 `label_l1`（275），ラベル無しのターゲットで割合を推定する MLLS＋BCTS `mlls_prior`・`fit_bcts`（280-316） — [docs/methods.md:320-340](/home/snics/MER-DA/docs/methods.md)．MLLS の推定はコーパス間では誤差が大きい（音声で ‖q − p_t‖₁ が 0.34〜1.71） — [docs/results.md:553](/home/snics/MER-DA/docs/results.md)
- 軸 4（ラベルの付け方）：ソースのみの誤り R_src と混同行列 `confusion`（326），`uar`（333），BER `ber`（337），Δ_CE `delta_ce`（342）．OTDD は実装していない．閾値のずれは測るより消す操作（D1）として使う．注釈者の一致率は発話一覧に列が無いので測っていない．accuracy-on-the-line は未実装 — [docs/methods.md:342-387](/home/snics/MER-DA/docs/methods.md)
- ずれを消す・足す操作（`measure/ablate.py`）：感情の割合をそろえる抜き出し（`balanced_subset`，`prior_matched_subset`），標準化（`standardize_with`，`standardize_by_speaker`），CORAL（`coral`），ターゲットに近いソースの抜き出し（`near_target_subset`），ラベルの付け替え（`flip_labels`，`flip_labels_pair`），感情ごとの切片の当てはめ（`fit_class_bias`） — [measure/ablate.py:26-153](/home/snics/MER-DA/measure/ablate.py)
- ずれの指標の結果：`results/shift_<kind>/` に 8 種類すべて済み．`labeling.csv`（軸 4），`label.csv`（軸 3），`input.csv`（軸 1・2），`zero.csv`（ゼロ点），`space_choice.json`，`logits/` — [docs/results.md:545-553](/home/snics/MER-DA/docs/results.md)

**仕組みの量**
- d と o：4 感情の分類を 2 つの感情を直線で分ける 6 つの問題とみなし，感情の対の正解率を ½[Φ(d/2+o)+Φ(d/2−o)] で表す．d は向きの良さ（識別力），o は位置のずれ（判断基準のずれ） — [docs/researchvision.md:166-177](/home/snics/MER-DA/docs/researchvision.md)．定義は E11（[docs/exploratory-analyses.md:835](/home/snics/MER-DA/docs/exploratory-analyses.md)），ソースのみの分類器の d と o を感情の対ごとに測るのは E13（[docs/exploratory-analyses.md:1007](/home/snics/MER-DA/docs/exploratory-analyses.md)）．E13 の値は `results/check_mh/e13.csv` にもある — [docs/results.md:592](/home/snics/MER-DA/docs/results.md)
- dprime：2 つの感情の重心を結ぶ線の上で，2 つの雲の中心が標準偏差の何倍離れているか — [exp/explore_geometry.py:14](/home/snics/MER-DA/exp/explore_geometry.py)．MH5（組の相性は dprime の差で決まる）に使う — [docs/researchvision.md:177](/home/snics/MER-DA/docs/researchvision.md)，[exp/check_mh.py:139-171](/home/snics/MER-DA/exp/check_mh.py)
- E9・E10 の量（β，shift_emo，eff_lda，own_lda）は量の定義が本筋，得との関係は端に仕分けた — [docs/results.md:210](/home/snics/MER-DA/docs/results.md)
- 向きの効率 e と学びにくさ q：MH2 は追い越す K を K_x ≈ q_t e²/(1−e²) とする — [docs/researchvision.md:173](/home/snics/MER-DA/docs/researchvision.md)．e は e_geom（E10 の eff_lda）と e_emp（地図のソースのみの UAR から）の 2 通りで測った — [docs/exploratory-analyses.md:893](/home/snics/MER-DA/docs/exploratory-analyses.md)．q は q_dir（向きだけの雑音）と q_t（UAR の学習曲線）の 2 通りで，q_dir の中央値は W7 16.1，Hu7 22.2，V 1.9，AV 6.8 — [docs/exploratory-analyses.md:3128-3158](/home/snics/MER-DA/docs/exploratory-analyses.md)
- λ：合併が位置のずれを直す割合 λ = ρ/(1+ρ)．ρ はターゲットの割合 ×（1 + 見分けやすさ）に比例する．測った λ との順位相関は 0.84〜0.91 — [docs/results.md:217](/home/snics/MER-DA/docs/results.md)，[docs/exploratory-analyses.md:1533](/home/snics/MER-DA/docs/exploratory-analyses.md)
- ソースの価値 n = q e²/(1−e²)（ターゲットのラベル何本ぶんか）．1 ソースの仮説転移で感情ごとに約 10 本（音声），組ごとの予言と学習曲線の順位相関 0.77〜0.90 — [docs/results.md:236](/home/snics/MER-DA/docs/results.md)，[docs/exploratory-analyses.md:3589](/home/snics/MER-DA/docs/exploratory-analyses.md)．結果は `snapshots/20261016-0954_e32/` — [docs/results.md:650](/home/snics/MER-DA/docs/results.md)
- E_0 = W* − W_0（出発点の向きの誤り．W* は候補全体のラベルで学んだ解）と，ラベルを付けた発話が張る空間が E_0 を捕らえる割合 r — [docs/results.md:262](/home/snics/MER-DA/docs/results.md)，[docs/exploratory-analyses.md:6345-6365](/home/snics/MER-DA/docs/exploratory-analyses.md)．結果は `results/explore_relation/e58_targets.csv`・`e58_tests.csv`・`e58_corr.csv` — [docs/exploratory-analyses.md:6345](/home/snics/MER-DA/docs/exploratory-analyses.md)
- J：候補の上の「先生の UAR − 出発点の UAR」（先生が直す割合 − 壊す割合）．弱いラベルの得の目安 — [paper/CLAIM.md:39](/home/snics/MER-DA/paper/CLAIM.md)
- 先生の本数 n_mod = K'(u_先生あり) − K'(u_none)（学習曲線で読む） — [docs/exploratory-analyses.md:6981](/home/snics/MER-DA/docs/exploratory-analyses.md)．結果は `results/explore_relation/e65_teacher_value.csv`・`e65_summary.csv`・`e65_selection.csv` — [docs/exploratory-analyses.md:6985](/home/snics/MER-DA/docs/exploratory-analyses.md)
- K*（合併 − ターゲットのみの 95% 区間が 0 を上回る最大の K），K_x，K_eq，α* の定義は付録 A — [docs/researchvision.md:685-755](/home/snics/MER-DA/docs/researchvision.md)，[docs/researchvision.md:462-464](/home/snics/MER-DA/docs/researchvision.md)

**結果の場所**
- 地図：`results/map_<kind>/<ターゲット>.csv`（30 回：audio_wavlm，audio_hubert，video，fuse-audio-video．5 回：言葉を含む 4 種類）．列は `method`（tgt_only／src_only／merged），`K`，`rep`，`C`，`uar`，C の候補ごとの交差検証とテストの UAR など — [docs/results.md:494-535](/home/snics/MER-DA/docs/results.md)
- C を固定した地図（論文と発表の本文の土台）：`results/fixc_map/<kind>/` — [docs/results.md:504](/home/snics/MER-DA/docs/results.md)，[docs/researchvision.md:459-460](/home/snics/MER-DA/docs/researchvision.md)
- 学び直し（E12）：`results/mech_<kind>/`，HuBERT での MH の判定：`results/check_mh/` — [docs/results.md:577-596](/home/snics/MER-DA/docs/results.md)
- E シリーズ：`results/explore_relation/snapshots/<日時>/`（直下は台本を流すたびに上書きされる） — [docs/results.md:565-575](/home/snics/MER-DA/docs/results.md)
- 手法の比較：`results/baselines_<世界>/`，`results/multimodal_distill_<世界>/`，`results/finetune_check/`，検定と効果量は `results/stats_compare/` — [docs/exploratory-analyses.md:6920](/home/snics/MER-DA/docs/exploratory-analyses.md)，[docs/exploratory-analyses.md:6967](/home/snics/MER-DA/docs/exploratory-analyses.md)
- 論文の数値の台帳は `paper/results/numbers.json` — [paper/CLAIM.md:5](/home/snics/MER-DA/paper/CLAIM.md)
- `results/` の直下には 257 項目，`results/explore_relation/` の直下には 286 項目がある（2026-10-28 に `ls` で数えた）

### Inferences
- 新しい学習法の候補が「ずれの量に応じて重みや罰則を変える」形なら，PAD・MMD・coverage などは全組で計算済みなので，追加の計算なしに説明変数として使える．ただし E1 と H7 で，ラベル無しの距離は利得を予言できなかった（4 節）．
- 候補が「d と o を別々に直す」形なら，E13・E29 の d と o の測り方と，E58 の E_0 の測り方をそのまま評価の道具に使える．

### Gaps
- ラベルの揺れ（注釈者の一致率）を測る量は未実装である（[docs/methods.md:379-382](/home/snics/MER-DA/docs/methods.md)）．
- OTDD と accuracy-on-the-line は実装していない．

---

## 3. 既存の学習法と否定された方向

### Takeaway
今の手法は「凍結した特徴＋多クラスのロジスティック回帰」の上に，中心化，全ソースの出発点への L2 罰則つきの仮説転移，仮のターゲットで選ぶ 5 つの値（E39），先生の出力をクラス均衡した弱いラベルとして候補全体に付ける蒸留（E59・E60・oursx）を積んだものである．否定された方向は，ラベル無しのターゲットから向きの情報を取る試み（EM，ラベル拡散，能動学習，白色化した設計），メタ学習で多くの数を学ぶ試み（MAML，E35），エントロピー最小化，LoRA の微調整，「先生の本数は K によらず一定」とする重みの規則である．

### Cited Findings

**固定する学習器**
- 特徴は微調整していないエンコーダの出力の発話平均，融合は次元ごとに標準化して連結する早期融合，分類器は多クラスのロジスティック回帰（L2．切片に罰則なし．`class_weight="balanced"` 相当）を GPU（float64 の L-BFGS，勾配の最大絶対値 1e-6）で解く — [docs/methods.md:29-36](/home/snics/MER-DA/docs/methods.md)，[docs/researchvision.md:471](/home/snics/MER-DA/docs/researchvision.md)

**手法の積み上げ**
- P1〜P4（E31）：P1〜P3 はソースのみの分類器を下書きにした仮説転移（P1 は中心合わせなし・切片自由，P2 は帰納の中心合わせ・切片自由，P3 は帰納の中心合わせ・切片を C_b で縛る）．罰則 C′ ∈ {0.001, 0.01, 0.1, 1}，C_b ∈ {2, 8, 自由} の 24 設定，既定 C′=0.01，C_b=8．P4 は同じコーパスの群を除く全ソースを混ぜた分類器を下書きにする — [docs/exploratory-analyses.md:3380-3387](/home/snics/MER-DA/docs/exploratory-analyses.md)
- E31 の結果：P4 の K=5 の UAR は W7 53.6，Hu7 52.1，AV 58.1，V 47.1 で，主な 4 種類の K=1〜50 のすべてで最終候補の中で最も高い．全件では転移の得が消える — [docs/results.md:235](/home/snics/MER-DA/docs/results.md)
- 等しい比重の出発点 D1（E32 の 5 節）：ソースごとの分類器を等しい重みで足した下書きからの仮説転移は，P4 を K=5 で W7 +2.03，Hu7 +1.64 ポイント上回った — [docs/results.md:236](/home/snics/MER-DA/docs/results.md)
- E39（5 つの値を選ぶ）：下書き・基準・向きに対応する 5 つの値（γ，ρ_s，C_b，C′，ρ_M）を，ソースコーパスを仮のターゲットにして選ぶ．D1 を K=5 で W7 +1.00，Hu7 +1.62，AV +1.21，V +1.64 ポイント上回り，P4 からは +3.0〜+4.0．得の大部分は罰則の計量 ρ_M から来る — [docs/results.md:243](/home/snics/MER-DA/docs/results.md)
- E59（見込みの蒸留）：ソースの票の見込みを候補全体の弱いラベルにして E39 の出発点から学び直す．K ごとに総重みを選ぶと，E39 との差は A で K=0／1／5 に +1.41／+0.79／+0.88，V で +1.01／+1.16／+0.33．内訳は釣り合い直し（A +0.9）と見込みの情報（+0.3〜+0.5） — [docs/results.md:263-264](/home/snics/MER-DA/docs/results.md)
- E60（モダリティをまたぐ蒸留）：各モダリティの生徒に融合の出発点などを弱いラベルにする．9 つの（世界, モダリティ）すべてで K=0・5 に正（AV の音声 +3.4／+2.2，AVT の言葉 +3.4／+3.0）．K ≥ 10 では 13 生徒中 4 つで損 — [docs/results.md:265](/home/snics/MER-DA/docs/results.md)
- 基本版と拡張版（oursx）の先生の候補：基本版は pF，poe，agree（単独の生徒），poe0，ref，refdirect（融合の生徒）．拡張版は K 本で学び直した融合の生徒 pFK と ref′，重みづけ（釣り合わせる／釣り合わせない／しきい値 0.5・0.7・0.9 の硬いラベル）を K ごとに選ぶ — [paper/CLAIM.md:15-29](/home/snics/MER-DA/paper/CLAIM.md)
- E61（比べた相手）：自己訓練，エントロピー最小化，温度付きの蒸留，一般化蒸留，硬い疑似ラベル，共訓練，共正則化，Saerens の EM，出力の釣り合い直し，CLUE・TQS を，同じ生徒・同じ K・同じ仮のターゲットの選び方で測った．調整版・固定版・オラクルを併記 — [docs/results.md:266](/home/snics/MER-DA/docs/results.md)，[docs/exploratory-analyses.md:6612-6618](/home/snics/MER-DA/docs/exploratory-analyses.md)．K ≥ 5 では ours の得が消えるが，oursx は AVT +2.1〜2.4，VT +1.7〜2.2，AT +1.3〜1.6，AV +0.5〜1.0 を保つ — [docs/exploratory-analyses.md:6763](/home/snics/MER-DA/docs/exploratory-analyses.md)．基本版は最も強い相手を 22 セル中 19 セルで上回る — [paper/CLAIM.md:28](/home/snics/MER-DA/paper/CLAIM.md)
- E62：第 2 エンコーダ（HuBERT，CLIP，DeBERTa）の AVT で結論が再現した（S1〜S4 支持） — [docs/results.md:267](/home/snics/MER-DA/docs/results.md)
- E64：ターゲット × 反復の対を単位にしたウィルコクソン検定と効果量．ターゲットが 4 つ（AV は 6 つ）しかないので，ターゲットの符号検定では全部正でも両側 p は 0.125 にしかならない — [docs/exploratory-analyses.md:6912-6973](/home/snics/MER-DA/docs/exploratory-analyses.md)
- E65：別のモダリティの先生の本数は K とともに 3〜6 分の 1 に減る（音声・m=2000 で 29.3 → 3.9 本）．「本数は一定なので m は K によらない 1 つの値でよい」という読みは K=25 で外れ，ラベル換算の価値で重みを決める最も単純な形は支持されない — [docs/exploratory-analyses.md:6986-7006](/home/snics/MER-DA/docs/exploratory-analyses.md)
- E66（下調べ．仮のターゲットだけ）：先生の弱いラベルで K=0 の出発点を動かし，K ≥ 1 では弱いラベルを外して，動かした出発点を罰則の中心にして K 本だけで学び直す．弱いラベルの項は「罰則の中心を W0 から W_Q へ動かす」ことと「保つ強さを M/C′ から m H_Q + M/C′ へ増やす」ことを同じ m で同時に行う，という 2 次の展開を根拠にした．予言 P1〜P5 は計算の前に書いた — [docs/exploratory-analyses.md:7009-7055](/home/snics/MER-DA/docs/exploratory-analyses.md)．判定はログにだけある：P1 は音声・言葉・顔で支持（音声 +3.79／+2.99／+2.46／+1.51，K=1／5／10／25），P2 は音声（比 0.97）と顔で支持・言葉で不支持（比 0.49），P3・P4 は 3 種類で支持，P5 は不支持 — `logs/e66_analysis.log` の末尾 15 行（2026-10-28 03:13 の結果ファイル `results/explore_relation/e66_*.csv` と同時）．この判定は文書にまだ書かれていない

**否定された方向**
- E58：ラベル無しの選び方 9 通りが張る空間は E_0 をランダムの 0.5〜1.2 倍しか捕らえず，条件 1 の教える選び方でも 1.4〜2 倍．ソース間の向きの散らばりと仮のターゲットの直し向きは E_0 の 1 割前後しか含まない．登録した決まりにより「E39 の残りの誤りは，ラベル無しの情報では向きが分からないターゲット固有の誤り」と結論した — [docs/exploratory-analyses.md:6379-6403](/home/snics/MER-DA/docs/exploratory-analyses.md)．結果を見た後の確かめで，直しのランク切り，近い 10 本への広げ，上位主成分への制限，主成分の事前分散，逐次の選び方はどれも効かず，蒸留だけが得を出した — [docs/exploratory-analyses.md:6405-6411](/home/snics/MER-DA/docs/exploratory-analyses.md)
- E28-10：ソースの事後確率から始めた EM（錨つきも錨なしも）は向きを 25〜30% 悪くし，10 本のラベルを固定しても仮説転移の K=10 に遠く及ばない．「向きを直す情報源はラベルであり，ラベル無しのターゲットは位置（中心合わせ）にだけ使える」．限界として，試した塊のモデルは共通の共分散の正規分布の混合 1 つだけである — [docs/exploratory-analyses.md:3213-3240](/home/snics/MER-DA/docs/exploratory-analyses.md)
- E38：ラベル無しのターゲットの散らばりで罰則の計量を作り，白色化の度合いを K の関数としてメタ学習しても，P4 からの差は音声の K=5 で +0.59〜+0.95，K ≥ 10 でほぼ 0，映像で損 — [docs/results.md:242](/home/snics/MER-DA/docs/results.md)
- E35：適応後の見込みを損失にして全ソースの下書きを学ぶメタ学習は，等しい重みの下書きを 4 種類すべてで下回った．効いたのはコーパスに等しい重みを付けることだけ — [docs/results.md:239](/home/snics/MER-DA/docs/results.md)
- O7（MAML による線形の頭の初期値）と O6（最近傍セントロイド）は P3 より 9〜13 ポイント低い（音声，反復 10 回．P3 − O7 は W7 +11.23，Hu7 +13.17） — [docs/results.md:235](/home/snics/MER-DA/docs/results.md)，[docs/exploratory-analyses.md:3502](/home/snics/MER-DA/docs/exploratory-analyses.md)．5 つの値なら，数千個の数を学んだ E35・MAML と違い，学習に使ったコーパスに合わせすぎない — [docs/exploratory-analyses.md:4634](/home/snics/MER-DA/docs/exploratory-analyses.md)
- E25：CORAL は全種類で最も悪い — [docs/results.md:229](/home/snics/MER-DA/docs/results.md)
- E27：ラベル無しの TV で中心合わせの門は作れない（常に中心合わせをすれば 95% の組で損しない） — [docs/results.md:231](/home/snics/MER-DA/docs/results.md)
- 感情の割合の補正は E28-5 で効果が無く，K 本での候補の選択は E28-8 で当てにならないので，手法に入れないと決めた — [docs/researchvision.md:439](/home/snics/MER-DA/docs/researchvision.md)
- 条件 2（ラベルを付ける前に選ぶ）：既存の能動学習（迷い，散らばり，典型，迷い＋散らばり）は K=5 でランダムを −3.0〜+0.8 しか動かさない．E39 の生徒では，どのラベル無しの選び方も K=5 でランダムを 1 ポイント以上は上回らない — [docs/researchvision.md:124-129](/home/snics/MER-DA/docs/researchvision.md)，[docs/results.md:250-254](/home/snics/MER-DA/docs/results.md)
- E42：近い 10 本で見込みをならすラベル拡散は，音声ではほぼ変わらない（K=5 のラベルの後で −0.90，−1.22） — [docs/results.md:246](/home/snics/MER-DA/docs/results.md)
- E56：E39 の出発点の上では，信頼度の低いソースの発話を外す得はほとんど残らない — [docs/results.md:260](/home/snics/MER-DA/docs/results.md)
- E57（条件 3：ウォームスタートの能動学習）は研究者の判断で中止 — [docs/results.md:261](/home/snics/MER-DA/docs/results.md)
- E61：エントロピー最小化の固定版は 4 世界の平均ですべて損（AVT −2.9，VT −10.3，AT −7.2，AV −14.3，K=0）．共正則化・Saerens の固定版も −10〜−33 — [docs/exploratory-analyses.md:6766](/home/snics/MER-DA/docs/exploratory-analyses.md)
- E63：WavLM の LoRA（r=8）微調整は，K=5 で凍結の E39 を 7.3 下回り，出発点への罰則は +0.4 しか効かない．弱いラベルの得は微調整の上でも K=0 +5.8，K=5 +4.4 残るが，微調整版は凍結版を 5.4〜6.2 下回る．中心化を足しても差の 3 分の 1 しか埋まらない — [docs/exploratory-analyses.md:6880-6910](/home/snics/MER-DA/docs/exploratory-analyses.md)
- E14：話者ごとの位置のずれ（MH6）は支持されない — [docs/results.md:214](/home/snics/MER-DA/docs/results.md)．E1：距離と得の相関はほぼ 0 — [docs/results.md:205](/home/snics/MER-DA/docs/results.md)
- E30：2 層のパーセプトロンでも o の直しの振る舞いは線形と同じだが，K=5 の得に占める o の割合は 0.37〜0.55 に下がる — [docs/results.md:234](/home/snics/MER-DA/docs/results.md)

### Inferences
- 「ラベル無しのターゲットの構造から向き（d）を学ぶ」型の候補（クラスタの仮定，多様体の仮定，自己訓練，エントロピー最小化）は，E28-10・E42・E58・E61 の 4 つの実験の結果と逆向きなので，採るなら，これまで試していないモデルの形（部分空間，非線形の塊）を明示する必要がある．E28-10 は試したのが 1 つのモデルだけだと限界に書いている．
- 「メタ学習で多くの数を学ぶ」型（MAML，損失の重みのメタ学習）は O7・E35 で負けている．仮のターゲットで少数の値を選ぶ形（E39）が今の作法である．
- 得が出ている経路は，(1) ラベル無しの中心化による o の直し，(2) ソースを等しく束ねた出発点と罰則の計量による d の事前分布，(3) 別のモダリティの先生の弱いラベルである．新しい候補を評価するときは，この 3 つのどれを置き換えるのか，どれに足すのかを決めると，比べる相手（E39，oursx）が決まる．
- E66 のログの判定（P1 支持，P5 不支持）は，「先生で出発点を動かしてから K 本で学ぶ」形が仮のターゲットでは得を K=25 まで保つことを示す．評価に使うターゲットでの確かめはまだである．

### Gaps
- E66 の結果は文書（`docs/exploratory-analyses.md`）にまだ書かれていない．`logs/e66_analysis.log` の判定表だけを見たので，その解釈と次の手（7049-7051 行の分岐）は研究者の確認が要る．
- E28-5（感情の割合の差）の詳しい数値は，results.md の行が長く，冒頭だけを読んだ．

---

## 4. 計算資源と所要時間の目安

### Takeaway
研究室の RTX A5000（24 GB，WSL2）と RunPod の H100 80GB（日本）・RTX 4090（ルーマニア）を使ってきた．地図 1 回は約 46 分，主な手法の比較は 1 ターゲット数時間，E62 は H100 で 5 時間 10 分，E63 は A5000 で 21 時間（運用の失敗を直した後なら約 3 時間）である．特徴を作り直すと GPU で丸 1 日かかる．

### Cited Findings
- 研究室の GPU は RTX A5000 24 GB — [docs/exploratory-analyses.md:6837](/home/snics/MER-DA/docs/exploratory-analyses.md)
- RunPod：日本のデータセンター（AP-JP-1）の H100 80GB HBM3 とネットワークボリューム（50 GB）を使う．ルーマニアの RTX 4090 への上りは毎秒約 0.2 MB で遅い．RTX 4090 でも H100 でも結果は研究室PCと差 0.0 だったが，herding のような貪欲な選び方は GPU が違うとターゲットの平均で最大 0.5 点ずれた — `/home/snics/.claude/projects/-home-snics-MER-DA/memory/runpod-setup.md`（MER-DA の作業メモ）
- 地図：音声の地図は 1 回（7 ターゲット）約 46 分，30 回で約 23 時間．GPU の仕事と並行して CPU の重い処理を流すときは並列数を 8 に絞る（取り合うと学習が約 2.5 倍遅くなった） — [docs/researchvision.md:472-474](/home/snics/MER-DA/docs/researchvision.md)
- 言葉系の C を固定した地図の 30 回は GPU 約 1 時間，交差検証の地図の延長は約 67 時間かかるので行わない — [docs/researchvision.md:475](/home/snics/MER-DA/docs/researchvision.md)，[docs/researchvision.md:502](/home/snics/MER-DA/docs/researchvision.md)
- 事前登録した操作の予言 P1〜P4 を登録どおり回すと WavLM だけで約 70 時間，O2（特徴の水増し）を全部流すと約 40 時間，操作の実験の本番は約 400 時間 — [docs/researchvision.md:441-442](/home/snics/MER-DA/docs/researchvision.md)，[docs/researchvision.md:505](/home/snics/MER-DA/docs/researchvision.md)
- E58：`run/e58.sh` を 2026-10-21 22:15〜22:36 に流した（約 21 分．A5000） — [docs/exploratory-analyses.md:6347](/home/snics/MER-DA/docs/exploratory-analyses.md)
- E61：1 ターゲット 2.6〜6.1 時間（A5000） — [docs/exploratory-analyses.md:6667](/home/snics/MER-DA/docs/exploratory-analyses.md)
- E62：H100 1 枚で 5 時間 10 分（4 ターゲット，反復 0〜9） — [docs/results.md:267](/home/snics/MER-DA/docs/results.md)．計画時の見込みは H100 で約 12 時間（第 1 の世界は A5000 で段 3 が約 17 時間） — [docs/exploratory-analyses.md:6795](/home/snics/MER-DA/docs/exploratory-analyses.md)
- E63：A5000 で 21 時間．PyTorch の空きブロックが GPU の上限に触って評価が 5〜10 倍遅くなっていたためで，直した後なら 3 時間ほど．`expandable_segments` は WSL2 で動かない．E63c は約 6 時間．LoRA の 1 歩（16 本 × 10 秒）は 0.35〜0.5 秒，GPU 4.7 GB — [docs/exploratory-analyses.md:6841](/home/snics/MER-DA/docs/exploratory-analyses.md)，[docs/exploratory-analyses.md:6893-6896](/home/snics/MER-DA/docs/exploratory-analyses.md)
- E39 は RunPod の H100 2 台，広げた候補（576 通り）は H100 4 台で流した．E39 の試走（1 ターゲット，反復 0・1）は H100 で 298 秒 — [docs/results.md:243](/home/snics/MER-DA/docs/results.md)，[docs/exploratory-analyses.md:4598](/home/snics/MER-DA/docs/exploratory-analyses.md)，[docs/exploratory-analyses.md:4664](/home/snics/MER-DA/docs/exploratory-analyses.md)
- E28-10 は CPU で 4 種類並列に約 3 時間，`check_mh.py` は CPU で約 7 分，E64 は CPU で 17 秒，E65 は 6 秒，E66 の試走（AV の AFEW の反復 0）は 158 秒 — [docs/exploratory-analyses.md:3215](/home/snics/MER-DA/docs/exploratory-analyses.md)，[docs/results.md:593](/home/snics/MER-DA/docs/results.md)，[docs/exploratory-analyses.md:6918](/home/snics/MER-DA/docs/exploratory-analyses.md)，[docs/exploratory-analyses.md:6979](/home/snics/MER-DA/docs/exploratory-analyses.md)，[docs/exploratory-analyses.md:7055](/home/snics/MER-DA/docs/exploratory-analyses.md)
- E31 段 3 の言葉系の延長の 1 段は約 9 時間 — [docs/exploratory-analyses.md:4937](/home/snics/MER-DA/docs/exploratory-analyses.md)
- 特徴の作り直し：第 1・第 2 のエンコーダの抽出がそれぞれ半日（GPU），全部で丸 1 日 — [data/README.md:433-438](/home/snics/MER-DA/data/README.md)，[data/README.md:460](/home/snics/MER-DA/data/README.md)．E63 の微調整のために AVT 4 群の波形を float16 で `data/cache/wave_<群>.npy` に保存した（56.6 時間ぶん，約 6.5 GB） — [docs/exploratory-analyses.md:6840](/home/snics/MER-DA/docs/exploratory-analyses.md)

### Inferences
- 凍結した特徴の上の線形の生徒なら，1 つの手法の 4 ターゲット × 反復 0〜9 は A5000 で数時間から十数時間，H100 で数時間に収まる．エンコーダを動かす候補（微調整，時間方向の特徴）は，E63 と同じく 1 日規模になり，さらに特徴の作り直し（丸 1 日）が要る場合がある．
- 仮のターゲットだけで判定する下調べ（E65・E66）は数秒から数分で済むので，候補を絞る段に向く．

### Gaps
- E59・E60 の所要時間は今回読んだ範囲に書かれていなかった．

---

## 5. 規則と制約

### Takeaway
データの取り決め 1〜9（とくに話者の分離，4 クラスと UAR，微調整しないエンコーダ），ずれの指標と操作を `measure/` だけに通す規則，テストのラベルで設定を選ばない規則，長時間の処理を確認なしに起動しない規則，事前登録の文を書き換えない規則がある．論文の主張契約（`paper/CLAIM.md`）は研究者の承認なしに変えられず，「主張しないこと」の表が新しさの書き方を縛る．微調整の例外は E63 の確かめだけである．

### Cited Findings

**データの取り決めと進め方**
- 取り決め 1〜9：ラベルは発話一覧から貼り直す，`sample_id` で並べ直す，`ok` が偽の行と NaN・Inf の行を落とす，映像の `X_std` を使わない，話者を必ず分ける，IEMOCAP の即興と台本を別の群にする，K はクラスあたりの本数，ラベルは 4 つで主指標は UAR，エンコーダは微調整していないものだけ — [CLAUDE.md:41-49](/home/snics/MER-DA/CLAUDE.md)．理由（`X_std` はセグメント数の代理で，セグメント数だけで 6 群を UAR 0.2822 で当てられる，など）は [data/README.md:26-55](/home/snics/MER-DA/data/README.md)
- 公式の分割（`split` 列）は話者の分離を保証しないので使わない — [data/README.md:45](/home/snics/MER-DA/data/README.md)
- MER-DA は自己完結させ，`~/MER` などを参照しない — [CLAUDE.md:10](/home/snics/MER-DA/CLAUDE.md)
- ずれの指標は `measure/shift.py`，ずれを消す操作は `measure/ablate.py` を必ず通す．例外は学び直し（E12）の `exp/mech.py` — [CLAUDE.md:85](/home/snics/MER-DA/CLAUDE.md)，[docs/researchvision.md:481](/home/snics/MER-DA/docs/researchvision.md)
- 長時間の処理は勝手に起動せず，コマンド・所要見込み・上書きするファイルを示して確認を取る．tmux の `mer-da` で回す — [CLAUDE.md:86-88](/home/snics/MER-DA/CLAUDE.md)
- 数値の主張はログか出力ファイルを示してから述べる．事前登録した仮説は結果を見てから書き換えない．ハイパーパラメータはテストラベルで選ばず，実用値とオラクル値を分けて報告する — [CLAUDE.md:90-92](/home/snics/MER-DA/CLAUDE.md)
- 設定はソースコーパスを仮のターゲットにして選ぶ（E36・E39・E40・E44・E59・E60）．E61 では相手にも同じ作法を与える — [docs/results.md:478](/home/snics/MER-DA/docs/results.md)
- 手法の比較の決まり：候補の設定ごとにテストの点数と K 本の交差検証の成績を保存し，メタ検証・K 本の交差検証・オラクルの 3 規則を全手法に同じく適用する．統計の単位は組，主な比較を限定して Holm — [docs/researchvision.md:440](/home/snics/MER-DA/docs/researchvision.md)

**地図と評価の決定事項（6 節）**
- テストはターゲットの話者を 5 袋に分けた r mod 5 番の袋（約 20%）．K 本は K をまたいで入れ子に抜き，ソースに依らず同じ K 本にする．反復は 30 回（言葉を含む 4 種類は主な実験で 30 回にそろえた） — [docs/researchvision.md:453-455](/home/snics/MER-DA/docs/researchvision.md)，[docs/researchvision.md:502](/home/snics/MER-DA/docs/researchvision.md)
- 論文と発表資料の本文は，合併の C をソースのみの C に固定した地図だけで書く — [docs/researchvision.md:459-460](/home/snics/MER-DA/docs/researchvision.md)
- IEMOCAP がターゲットの K* の区間は，テスト話者の組（即興 5，台本 7）を単位にする — [docs/researchvision.md:464](/home/snics/MER-DA/docs/researchvision.md)
- 論文は 1・2 を統合して IEEE TAC に出し，数理を前面に置く．主張の前半は単独の音声と顔，K ≤ 5 に狭めた．論文は `paper/` で書き，足りない結果は `paper/requests.md` に起票する — [docs/researchvision.md:432-434](/home/snics/MER-DA/docs/researchvision.md)
- 新しいコーパスでの確かめ（A3）と，操作の予言 P1〜P4 の判定（A4）は行わない — [docs/researchvision.md:441](/home/snics/MER-DA/docs/researchvision.md)
- 言葉の特徴は正規化の段階 L1 のまま使う — [docs/researchvision.md:437](/home/snics/MER-DA/docs/researchvision.md)

**未決事項（7 節）**
- 統合論文の主張契約と章立て（第 3 版を暫定承認），RAVDESS の映像を入れるか（保留），音声の最後の確認に使うエンコーダ（未検討），操作の実験の計算量（後回し），比較実験の計算量（未検討）などが並ぶ — [docs/researchvision.md:495-510](/home/snics/MER-DA/docs/researchvision.md)

**事前登録（8 節）**
- 8 節の仮説と判定の手順の文は，結果を見た後に書き換えない．H1〜H6 と P1〜P4 は判定していない．H7 は 4 つとも判定できない．MH は MH2・MH4・MH5 が支持，反証 0．R1〜R11 は反証 0 — [docs/researchvision.md:514-524](/home/snics/MER-DA/docs/researchvision.md)
- 各 E 節の予言は「計算の前に書いた」と節の題に記し，結果を見た後に足した版は予言を持たない探索として区別する（例：E66 の試走の後に足した 2 つの版） — [docs/exploratory-analyses.md:7055](/home/snics/MER-DA/docs/exploratory-analyses.md)

**エンコーダを微調整しない規則と例外**
- 微調整したエンコーダを使うと，データのずれとエンコーダの癖を分けて測れなくなる — [data/README.md:54-55](/home/snics/MER-DA/data/README.md)．映像の第 1 エンコーダは AffectNet で微調整済みだが，本研究の 7 群では微調整していない — [data/README.md:404](/home/snics/MER-DA/data/README.md)
- 例外の E63 は「凍結した特徴＋線形の頭という設定に対する指摘に 1 つの実測で答える」限界への備えで，AVT の 4 群の音声の生徒だけを LoRA で微調整した — [docs/exploratory-analyses.md:6833-6840](/home/snics/MER-DA/docs/exploratory-analyses.md)．E63 の実行は研究者の決定（2026-10-24） — [docs/researchvision.md:510](/home/snics/MER-DA/docs/researchvision.md)

**論文の主張契約（paper/CLAIM.md）**
- 査読役も改稿役も CLAIM.md を編集しない．変更には研究者の承認が要る — [paper/CLAIM.md:3](/home/snics/MER-DA/paper/CLAIM.md)
- 主張しないこと（抜粋）：「切片と向きに分けたのは初めて」「新しい式を導いた」「中心化は新しい手法」（主張 A），「実効標本数という新しい概念」「ラベル無しで価値が分かる」（主張 B），「ニュートン法の式は新しい」（主張 C），「能動学習は役に立たないと証明」「ロジスティック回帰で成り立つ」（命題 1・4），「モダリティをまたぐ蒸留をドメインをまたぐ少数ショットで初めて」「融合から単モーダルへの蒸留は新しい」「クラス均衡の操作は新しい」「仮のターゲットでの選択は新しい選び方」（手法の部品），「マルチモーダル × 少数ラベル × コーパス間は初めて」（設定），「エントロピー最小化は常に損」「クラス均衡の仮定は安全」（比較の結果），4 感情・7 コーパス・凍結特徴＋線形の生徒・話者分離の外への一般化（一般化） — [paper/CLAIM.md:70-81](/home/snics/MER-DA/paper/CLAIM.md)
- 反証条件 7 の前半（仮のターゲットで選んだ設定がオラクルを 1 ポイント以上下回る）は観測された（K=0 で 11 生徒中 5）．後半（感情の比率が偏ったターゲットで列正規化の得が負になるか）は未判定 — [paper/CLAIM.md:60](/home/snics/MER-DA/paper/CLAIM.md)

**特徴抽出の状態**
- 4 ラベルの外の感情の特徴抽出は中止して保留中であり，再開するときは「足りない分だけ追加」の方式と merge の上書きの落とし穴に注意する — `/home/snics/.claude/projects/-home-snics-MER-DA/memory/MEMORY.md` の 2 行目（MER-DA の作業メモの索引）
- `run/build_data.sh manifest` はラベルの正本（発話一覧）を書き換えるので，走らせる前に git で状態を確かめる — [data/README.md:430-431](/home/snics/MER-DA/data/README.md)

### Inferences
- 7 群・4 クラス・UAR・話者の分離・凍結したエンコーダという枠を変える候補（5 クラス以上，次元のラベル，微調整）は，取り決め 7〜9 と論文の主張の範囲の外に出る．取り入れるなら，研究者の決定（6 節の更新）が先に要る．
- 発話一覧に列を足して評定の票や強さを使う候補は，`load_manifest()` が 11 列だけを読む作り（[data/README.md:164](/home/snics/MER-DA/data/README.md)）なので，`features/common.py` と `measure/data.py` の変更と，発話一覧の作り直しと監査が要る．
- 新しさの主張は，CLAIM.md の「主張しないこと」に抵触しない書き方にする必要がある．とくに中心化，クラス均衡，仮のターゲットでの選択，融合から単モーダルへの蒸留は，部品としては既知と扱われている．

### Gaps
- `paper/CLAUDE.md` と `paper/requests.md` の中身は今回読んでいない．

---

## 6. 未使用の材料：データにあるが解析に使っていない列と情報

### Takeaway
発話一覧には，解析に使っていない列として，IEMOCAP の価数・覚醒度・支配度，hap と exc の区別，MOSEI の 6 感情の強さと sentiment，MELD の sentiment と対話の文脈，AFEW の俳優の年齢・性別・顔の向き，RAVDESS と CREMA-D の感情の強さがある．生データには，IEMOCAP の評定者ごとのラベルと CREMA-D の知覚の票・俳優の属性がある．4 クラスに入らない感情の行（fru，calm，fear など）は発話一覧にあるが，特徴を作っていない．

### Cited Findings
- `load_manifest()` が読むのは 11 列（`sample_id, corpus, split, speaker_id, label_4, dialog_type, marker_side, path_video, path_audio, transcript, exclude`）だけである．読まない列として，全コーパスの `emotion_raw`（IEMOCAP の hap と exc の区別を含む），RAVDESS と CREMA-D の `emotion_intensity`，MOSEI と AFEW の `speaker_source` が残してある — [data/README.md:164-182](/home/snics/MER-DA/data/README.md)
- IEMOCAP の発話一覧は `valence`，`arousal`，`dominance`（評定の平均．値は 21〜22 種類），`utterance_speaker`（性別），`session`，`dialog_id`，`start_sec`，`end_sec` を持つ — `data/metadata/iemocap_utterance_manifest.csv` の列の集計．注釈ファイルの行は `[開始 - 終了] 発話名 感情 [価数, 覚醒度, 支配度]` の形 — [features/build_manifest_iemocap.py:39-42](/home/snics/MER-DA/features/build_manifest_iemocap.py)
- MOSEI の発話一覧は `int_happy`，`int_sad`，`int_anger`，`int_surprise`，`int_disgust`，`int_fear`（0〜3 の強さ）と `sentiment` を持つ（22,850 行に値がある） — `data/metadata/mosei_utterance_manifest.csv` の列の集計
- MELD の発話一覧は `sentiment_raw`（neutral 6,436，positive 3,088，negative 4,184），`dialogue_id`（1,423 対話），`utterance_id`，`season`，`episode`，`start_time`，`end_time` を持つ — `data/metadata/meld_utterance_manifest.csv` の列の集計
- AFEW の発話一覧は `actor_age`，`gender`，`pose`，`movie_title` を持つ — `data/metadata/afew_utterance_manifest.csv` の列の集計
- RAVDESS の `emotion_intensity` は normal 768 行，strong 672 行．CREMA-D の `emotion_intensity` は XX（指定なし）6,077 行，HI・LO・MD が 455 行ずつ — `data/metadata/ravdess_utterance_manifest.csv`，`data/metadata/crema-d_utterance_manifest.csv` の列の集計．CREMA-D の強さの記号は XX 指定なし／LO／MD／HI — [features/build_manifest_extra.py:41-45](/home/snics/MER-DA/features/build_manifest_extra.py)
- 4 クラスの外の行：IEMOCAP の fru 1,849・xxx 2,507・sur 107 など，MOSEI の multi_label 2,930・disgust 564・fear 278・surprise 159，MELD の surprise 1,636・disgust 361・fear 358，AFEW の Disgust 114・Fear 127・Surprise 120，RAVDESS の calm・fearful・disgust・surprised 各 192，CREMA-D の disgust・fear 各 1,271 — 各発話一覧の `emotion_raw` の集計．特徴の抽出と読み込みは 4 クラスの行だけを扱う — [data/README.md:153-154](/home/snics/MER-DA/data/README.md)
- 生データの評定の情報：IEMOCAP の注釈ファイルの評定者ごとの行（`C-E2:` など）— `/mnt/d/Databases/IEMOCAP_full_release/Session1/dialog/EmoEvaluation/Ses01F_impro01.txt`，CREMA-D の `tabulatedVotes.csv`・`finishedResponses.csv`・`finishedEmoResponses.csv`・`summaryTable.csv` と `VideoDemographics.csv` — `/mnt/d/Databases/CREMA-D_repo/` の一覧
- 操作 D2（ソースを注釈者の一致が高い発話に絞る）は methods.md に定義されているが，注釈者ごとのラベルが発話一覧に無いので回していない — [docs/methods.md:501](/home/snics/MER-DA/docs/methods.md)
- 長さの情報：映像の `n_segments`・`n_faces_frames`，音声の `duration_sec`・`truncated`，言葉の `n_token` が npz にある．長さは `X_std` の形で混ざるとコーパスを当てる手がかりになる — [data/README.md:36-41](/home/snics/MER-DA/data/README.md)，[data/README.md:361-365](/home/snics/MER-DA/data/README.md)
- 顔の座標 `faces_<コーパス>.csv.gz` が残っているので，別の表情モデルで同じ画素から特徴を作り直せる — [data/README.md:461](/home/snics/MER-DA/data/README.md)
- RAVDESS の映像の生データ（mp4 2,880 本）は取得済みだが，特徴は無い（入れるかは保留） — [data/README.md:467-469](/home/snics/MER-DA/data/README.md)，[docs/researchvision.md:500](/home/snics/MER-DA/docs/researchvision.md)
- 言葉の正規化 L2〜L4 は実装済みだが，解析は L1 だけを使う．L1 のままでは記号の表記の違い（IEMOCAP は ASCII だけ，MOSEI・MELD は Unicode を含む）が残り，表記の 16 次元だけでコーパスを均衡精度 0.7047 で見分けられる — [data/README.md:294-348](/home/snics/MER-DA/data/README.md)
- MOSEI のチャンネル対応表は 1,122 行（`video_id, channel_id`） — [data/README.md:216-219](/home/snics/MER-DA/data/README.md)
- 第 2 エンコーダの CLIP と DeBERTa の特徴は，2026-10-08 の時点では「作っただけ」だったが，E62 で使った — [data/README.md:254](/home/snics/MER-DA/data/README.md)，[docs/results.md:267](/home/snics/MER-DA/docs/results.md)
- 決定事項には「元のラベル（excited を Happy にまとめない形）での分析は追加で行う可能性がある」とある — [docs/researchvision.md:452](/home/snics/MER-DA/docs/researchvision.md)

### Inferences
- soft label を使う候補は，MOSEI（6 感情の強さ）と IEMOCAP（評定者ごとのラベルと価数・覚醒度・支配度）で作れる．CREMA-D では「意図したラベル」と「知覚の票」の両方があるので，ラベルの付け方のずれ（軸 4）をコーパスの中で直接測れる．ほかの群には同じ情報が無いので，7 群すべてにそろった soft label は作れない．
- 感情の強さ（RAVDESS の normal／strong，CREMA-D の LO／MD／HI，MOSEI の 0〜3）を条件にした解析は，特徴をすでに持つ 4 クラスの行の中でできる．ただし CREMA-D の強さの指定は 1 つの文（IEO）の 455 行ずつに限られ，4 クラスの行に何本入るかは数えていない．
- 4 クラスの外の感情（fru，calm など）を使う候補は，特徴の抽出（保留中）を再開する必要がある．
- MELD の対話の文脈（`dialogue_id`，`utterance_id`）を使う候補は，発話ごとの平均特徴しか無いので，前後の発話の特徴を組み合わせる形なら追加の抽出なしに作れる．
- 時間方向の構造を使う候補は，特徴が平均だけなので，抽出のやり直し（半日〜丸 1 日）が要る．

### Gaps
- MOSEI の評定者の人数と，`int_*` が何人の評定の平均かは，リポジトリ内で確かめられなかった．
- CREMA-D の知覚の票が，本研究の 4,900 発話のすべてに対応しているかは確かめていない．
- RAVDESS に知覚の評定のファイルがあるかは確かめていない．
- 言語・文化の情報（話者の母語，地域）は，CREMA-D の `Race`・`Ethnicity` 以外にリポジトリ内で見つからなかった．
