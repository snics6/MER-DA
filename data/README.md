# data/ — データと特徴

この文書は，本研究で使う発話一覧（マニフェスト）・特徴・重みについて，中身，読み込みの規則，作り方，作り直し方をまとめた正本である．
`data/` にあるものは，生データ `/mnt/d/Databases/`（読むだけ）から `features/` のコードが作った成果物であり，生データそのものは含まない．
データの扱いについて他の文書と食い違ったら，この文書が優先する（`CLAUDE.md`）．直すときはここで作り直し，git で差分を見る．

| 場所 | 中身 | 大きさ | git |
|---|---|---|---|
| `data/metadata/*_utterance_manifest.csv` | 発話一覧 6 本．**ラベルと話者の正本**（3 節） | 15 MB（下の対応表を含む） | 入れる |
| `data/metadata/mosei_channel_map.csv` | MOSEI の動画 ID と YouTube チャンネル ID の対応表（3 節） | 42 KB | 入れる |
| `data/metadata/afew_transcript_whisper.csv` | AFEW の Whisper large-v3 の書き起こし（3 節） | 168 KB | 入れる |
| `data/cache/*.npz` | 特徴 30 本（4 節） | 1.1 GB | 入れない |
| `data/cache/faces_*.csv.gz` | 顔検出の結果 5 本（4 節）．**消さない** | 17.6 MB | 入れる |
| `data/cache/parts/` | 抽出の途中保存．抽出中にだけ生じ，統合済みの npz があれば要らない | 1.1 GB 規模 | 入れない |
| `data/weights.sha256` | 重み 2 本の sha256（5 節） | — | 入れる |
| `weights/` | 顔検出器と表情モデルの重み（5 節） | 105 MB | 入れない |

---

## 1. データの取り決め

**特徴は必ず `measure/data.py` の `load_group(群, エンコーダ)` を通して読む．`np.load` した `X` をそのまま使ってはいけない．**
`load_group` は下の 1〜6 を処理し，成功した行だけを `sample_id` の順に並べた表（`sample_id`・`speaker_id`・`speaker_group`・`label_4` など）と特徴行列を返す．
複数モダリティを連結した特徴は `load_fused`，単一モダリティと連結の両方は `load_any` で読む．
番号は `CLAUDE.md` の「データの取り決め」と同じである．

1. **ラベルは `data/metadata/*.csv` から貼り直す．npz の `label_4` は使わない．**
   npz の `label_4` には抽出した時点のマニフェストの値が保存されており，マニフェストを直しても古い値のまま残る．
   実際，`video_IEMOCAP.npz` の `label_4` は現行のマニフェストと 5,531 件中 1,767 件で食い違う（2026-10-08 に照合した．他の 27 本は全件一致した．2026-11-04 に足した AFEW の言葉 2 本も全件一致した）．
   前身の研究ではマニフェスト自体が壊れており，IEMOCAP の 57% に誤ったラベルが付いていた（3 節「前身のマニフェストにあった誤り」）．
   `load_group` はマニフェストと `sample_id` で突き合わせ，ラベルと話者を取り直す．
2. **`sample_id` で並べ直す．行番号で標本を選ばない．**
   npz 30 本のうち 29 本は `load_manifest()` が返す行の順に並んでいる．`video_IEMOCAP.npz` だけは古いマニフェスト（9,169 行）から作ったので，行数も順序も現行のマニフェスト（5,531 行）と違う（現行の 5,531 件はすべて含まれている）．
   抽出をやり直せば順序が変わりうるので，K-shot の抽出などで行番号に頼ると結果を再現できない．`load_group` は最後に `sample_id` の昇順に並べ直す．
3. **`ok` が偽の行と，特徴に NaN・Inf を含む行は落とす．**
   抽出に失敗した行の特徴は NaN で埋めてあり，理由は `fail_reason` に入っている（件数は 2 節）．
4. **映像の `X_std` は使わない．**
   `X_std` はセグメント間の標準偏差であり，セグメントが 1 つしかない発話では必ずゼロになる．
   セグメントが 1 つになるのは顔を検出できたフレームが 20 枚未満の発話であり（4 節），その割合はコーパスによって 5 倍違う（成功した行のうち CREMA-D 91.3%・AFEW 89.8%・MELD 64.4%・IEMOCAP 43.6%・MOSEI 17.9%．現行の `video_*.npz` の `n_segments` で数えた）．
   このため `X_std` は発話の長さの代理変数になっており，連結するとコーパス間の距離の一部が動画の長さの違いで説明されてしまう．
   `measure/data.py` の記録では，セグメント数という整数 1 つだけで 6 群を UAR 0.2822 で当てられる（でたらめなら 0.1667）．
   `load_group` は既定（`with_std=False`）で `X_mean` だけを返す．
5. **話者は必ず分ける．**同じ話者が，学習に使う集合と評価に使う集合の両方に現れてはならない．
   分けないと，分類器は「この人は誰か」を当てるだけで高い精度を出せてしまう．
   分割は `load_group` が付ける `speaker_group` 列を単位にして，`measure/split.py`（`speaker_folds` など）で行う．
   コーパスの公式分割（`split` 列）は話者の分離を保証しないので，学習と評価を分けるのに使わない．
   `speaker_group` は `speaker_id` を `measure/split.py` の `speaker_groups` で補正した列である．
   いまの補正は，MELD の合同発話（`Ross_and_Rachel` のように `_and_` を含む話者 8 種・8 発話）を落とすことだけである．`Ross` を評価に回しても `Ross_and_Rachel` が学習に残ると，Ross の声と顔が学習に漏れるからである．
   `A_Waiter` と `The_Waiter` のような一般名詞の役名（`measure/split.py` の記録で 118 発話）は，同じ役名の別の端役と見なして残す．
   **DAI の実験（`DAI/`．2026-11-04〜）だけは，塊を作る前にさらに次を直す**（感情認識の査読者の指摘 A1．`DAI/exp/dai_data.py` の `AFEW_ALIAS`・`MELD_ALIAS`・`MELD_MULTI`）．
   `measure/split.py` は変えない（前の実験の分割を保つため）ので，前の実験の分割にはこの別名が残っている．
   - AFEW：`AFEW_Bradely_Cooper`（公式の注釈の綴りの誤り．Hangover の 5 発話）を `AFEW_Bradley_Cooper` にまとめる．
   - MELD：同じ俳優の別名 `Joey/Drake`・`Dr._Drake_Remoray`（Joey が劇中で演じる役）→`Joey`，`Rachel/actress`→`Rachel`，`Phoebe/Waitress`・`Ursula`（Phoebe と同じ俳優が演じる双子の姉）→`Phoebe`，`Phoebe_Sr.`→`Phoebe_Sr`，`Jessica_Lockhart`（Cecilia が劇中で演じる役）→`Cecilia`，`Helena`（Chandler の父の舞台名）→`Mr._Bing` をまとめる．`Dr._Stryker_Ramoray` は書き起こしから別の俳優と判断してまとめない（2026-11-04）．
     誰の声か決まらない `All`（38 発話），`Both`（1），`The_Guys`（2）は `_and_` と同じく発話ごと落とす（MELD の 3 モダリティの集合は 11,308 → 11,267 発話）．
     複数の人の声のラベルのうち AFEW の `AFEW_Group`（2 発話）と MELD の `Her-Friends`（1 発話）は残す．どちらも関係する人物と同じ分割にあり，漏れは生まない（2026-11-04 に数えた）．
6. **IEMOCAP は即興（`IEMOCAP-impro`）と台本（`IEMOCAP-script`）を別の群として扱う．**
   台本の場面では 10 人全員が同じ台詞を読むので，言葉の分類器は台詞を覚えるだけで当たってしまう．`load_group` はマニフェストの `dialog_type` で絞り込む．
7. **K はクラスあたりの本数で数える．**K=25 なら 4 クラス × 25 = 100 本である．
8. **ラベルは 4 つ（0=Neutral 1=Happy 2=Sad 3=Angry）で，主指標は UAR である．**
   UAR はクラスごとの再現率を平均した値で，でたらめに答えると 25% になる．
9. **エンコーダは，本研究のコーパスで微調整していないものだけを使う．**
   微調整したエンコーダを使うと，データのずれとエンコーダの癖を分けて測れなくなる（4 節「エンコーダ」）．

---

## 2. 群と件数

### 群とモダリティ

測定の単位は群であり，次の 7 つを使う．

```
IEMOCAP-impro / IEMOCAP-script / MOSEI / MELD / AFEW / RAVDESS / CREMA-D
```

| 群 | 映像 | 音声 | 言葉 |
|---|:---:|:---:|:---:|
| IEMOCAP-impro | ○ | ○ | ○ |
| IEMOCAP-script | ○ | ○ | ○ |
| MOSEI | ○ | ○ | ○ |
| MELD | ○ | ○ | ○ |
| AFEW | ○ | ○ | ○（Whisper の書き起こし．2026-11-04〜） |
| RAVDESS | ×（特徴を作っていない．7 節） | ○ | ×（読む文が 2 種類） |
| CREMA-D | ○ | ○ | ×（読む文が 12 種類） |

映像は 6 群，音声は 7 群，言葉は 5 群で使える．複数モダリティを連結するときは，すべてのモダリティがそろう群と発話だけを使う（音声と映像の連結は 6 群，言葉を含む連結は 5 群）．
言葉は 2026-11-04 に AFEW を足すまで 4 群だった．それより前に計算した言葉を含む結果は 4 群のものである．
どの群でどのモダリティを使えるかは `measure/data.py` の `has_modality` と `groups_for` が決める．

### 測定に入る件数

`load_group` を通した後の行数である（2026-10-08 に数えた）．

| 群 | 映像 ResNet-50 | 映像 CLIP | 音声 WavLM | 音声 HuBERT | 言葉 RoBERTa | 言葉 DeBERTa |
|---|---:|---:|---:|---:|---:|---:|
| IEMOCAP-impro | 2,943 | 2,943 | 2,943 | 2,943 | 2,943 | 2,943 |
| IEMOCAP-script | 2,549 | 2,549 | 2,588 | 2,588 | 2,588 | 2,588 |
| MOSEI | 18,634 | 18,634 | 18,919 | 18,919 | 18,919 | 18,919 |
| MELD | 11,308 | 11,308 | 11,343 | 11,343 | 11,344 | 11,344 |
| AFEW | 795 | 795 | 795 | 795 | 790 | 790 |
| RAVDESS | — | — | 672 | 672 | — | — |
| CREMA-D | 4,899 | 4,899 | 4,900 | 4,900 | — | — |

クラス別の件数と話者群の数は次のとおりである（音声 WavLM）．
ほかのエンコーダのクラス別の件数は，上の表の行数の差の分だけこれと違う（映像では顔を検出できなかった発話が抜け，MELD の言葉では音声の抽出に失敗した 1 発話が加わり，AFEW の言葉では Whisper が声を拾えなかった 5 発話が抜ける）．

| 群 | 話者群 | Neutral | Happy | Sad | Angry |
|---|---:|---:|---:|---:|---:|
| IEMOCAP-impro | 10 | 1,099 | 947 | 608 | 289 |
| IEMOCAP-script | 10 | 609 | 689 | 476 | 814 |
| MOSEI | 2,626 | 8,847 | 6,790 | 1,830 | 1,452 |
| MELD | 283 | 6,430 | 2,305 | 1,002 | 1,606 |
| AFEW | 282 | 207 | 213 | 178 | 197 |
| RAVDESS | 24 | 96 | 192 | 192 | 192 |
| CREMA-D | 91 | 1,087 | 1,271 | 1,271 | 1,271 |

MOSEI の話者群 2,626 の多くは人物ではなく動画 ID である（3 節「MOSEI のチャンネル対応表」）．

### npz ごとの行数と成功した行数

| ファイル | 行 | `ok` が真 | 失敗の理由（`fail_reason`） |
|---|---:|---:|---|
| `video_IEMOCAP` | 9,169 | 9,092 | 顔が 1 つも検出されない 77（古いマニフェストの行を含む．現行の 5,531 件では失敗 39） |
| `video_MOSEI` | 18,919 | 18,634 | 顔が 1 つも検出されない 285 |
| `video_MELD` | 11,352 | 11,316 | 顔が 1 つも検出されない 35，動画を開けない 1 |
| `video_AFEW` | 795 | 795 | |
| `video_CREMA-D` | 4,900 | 4,899 | 顔が 1 つも検出されない 1 |
| `video_clip_IEMOCAP` | 5,531 | 5,492 | 第 1 の抽出で顔が取れていない 39 |
| `video_clip_MOSEI` | 18,919 | 18,634 | 同 285 |
| `video_clip_MELD` | 11,352 | 11,316 | 同 36 |
| `video_clip_AFEW` | 795 | 795 | |
| `video_clip_CREMA-D` | 4,900 | 4,899 | 同 1 |
| `audio_wavlm_*`・`audio_hubert_*` | IEMOCAP 5,531・MOSEI 18,919・MELD 11,352・AFEW 795・RAVDESS 672・CREMA-D 4,900 | 行数と同じ（MELD だけ 11,351） | MELD の 1 件が「音声を取り出せない」 |
| `text_roberta_*`・`text_deberta_*` | IEMOCAP 5,531・MOSEI 18,919・MELD 11,352・AFEW 795 | 行数と同じ（AFEW だけ 790） | AFEW の 5 件が「文字起こしが無い」（3 節「AFEW の書き起こし」） |

`video_*` 5 本，`video_clip_*` 5 本，`audio_*` 12 本，`text_*` 8 本の計 30 本である（`text_*_AFEW` の 2 本は 2026-11-04 に足した）．
`video_IEMOCAP` 以外の 29 本の行数は，3 節の「使う行」と一致する．

---

## 3. 発話一覧（マニフェスト）

`data/metadata/` には，発話一覧 6 本と，MOSEI のチャンネル対応表 1 本と，AFEW の Whisper の書き起こし 1 本がある．
発話一覧は 1 行が 1 発話で，動画と音声の場所，話者，ラベル，転写を持つ．
6 本とも本リポジトリのコードが生データから作ったものであり，前身の研究が作った CSV は使っていない．
`features/common.py` の `load_manifest()` が発話一覧を読み，特徴の抽出はその行順で行う．

| ファイル | 全行 | **使う行** | 生成コード |
|---|---:|---:|---|
| `iemocap_utterance_manifest.csv` | 10,039 | **5,531** | `features/build_manifest_iemocap.py` |
| `mosei_utterance_manifest.csv` | 22,856 | 18,919 | `features/build_manifest_mosei.py` |
| `meld_utterance_manifest.csv` | 13,708 | 11,352 | `features/build_manifest_meld.py` |
| `afew_utterance_manifest.csv` | 1,156 | 795 | `features/build_manifest_afew.py` |
| `ravdess_utterance_manifest.csv` | 1,440 | 672 | `features/build_manifest_extra.py` |
| `crema-d_utterance_manifest.csv` | 7,442 | 4,900 | 同上 |

「使う行」は `load_manifest()` が返す行数であり，`exclude` が `false` で，`label_4` があり，映像か音声のどちらかのファイルがある行を数えた．
**論文の件数はこの母数で数える．**測定に入る件数は，ここから取り決め 3・5・6 で落とした後の値になる（2 節）．
MELD は `label_4` のある行が 11,353 あるが，`val_dia110_utt7` の 1 行は映像も音声も無いので落ちる．

発話一覧は 4 クラスに入らない感情の行も持つ（元データのラベルを残すため．RAVDESS と CREMA-D は 2026-09-29 にこの形へ作り直した）．
`label_4` は 4 クラスに当てはまる行にだけ付いており，`load_manifest()` は `label_4` が空の行を落とすので，特徴の抽出と読み込みは 4 クラスの行だけを扱う．

```
./run/build_data.sh manifest    # 発話一覧 6 本を作り直して監査する（6 節）
```

監査は `measure/audit_manifest.py` が生データと照合して行う（`sample_id` の一意性，ラベルの値，媒体ファイルの実在と重複，話者，文字化け，生データとの件数の一致）．

### 列

`load_manifest()` が読むのは次の 11 列である．無い列は空で埋められる．

| 列 | 中身 |
|---|---|
| `sample_id` | 発話の名前．コーパスの中で一意．**特徴との突き合わせはこの列で行う** |
| `corpus` | `IEMOCAP` / `MOSEI` / `MELD` / `AFEW` / `RAVDESS` / `CREMA-D` |
| `split` | 元のコーパスの区分．**本研究は話者で分けるので，この列で学習と評価を分けない**（取り決め 5） |
| `speaker_id` | 話者．**話者分割はこの列から作る `speaker_group` で行う**（取り決め 5） |
| `label_4` | 0=Neutral 1=Happy 2=Sad 3=Angry．空の行は落とされる |
| `dialog_type` | IEMOCAP だけ `impro` / `script`．**必ず分ける**（取り決め 6） |
| `marker_side` | IEMOCAP だけ．モーションキャプチャのマーカーを付けた人の性別．発話者が画面の左右どちらに映るかを決めるのに使う（`common.speaker_side`） |
| `path_video` | 映像ファイル．無ければ空（RAVDESS は全 1,440 行が空．7 節） |
| `path_audio` | 音声ファイル．IEMOCAP・RAVDESS・CREMA-D は wav を持ち，MOSEI・MELD・AFEW は空．空のときは `path_video` から ffmpeg で取り出す |
| `transcript` | 書き起こした文．IEMOCAP・MOSEI・MELD は人が書いたもの，AFEW は Whisper large-v3 の出力（下の「転写の出所」） |
| `exclude` | `false` 以外の行は落とされる（いまは全行 `false`） |

`load_manifest()` が読まない列も残してある．元の感情は全コーパスで `emotion_raw` に入っており，IEMOCAP の happy と excited はこの列で区別できる．
RAVDESS と CREMA-D は感情の強さを `emotion_intensity` に持つ．
MOSEI は話者の出所を `speaker_source`（`youtube_channel` か `video_id_fallback`），AFEW は `speaker_source`（`actor` か `film`）に持つ．
AFEW は書き起こしの出所を `transcript_source`（`whisper-large-v3`）に持つ．

### ラベルの決め方

| コーパス | 決め方 |
|---|---|
| IEMOCAP | 注釈ファイルの代表ラベル（多数決）から．`label_4` は neu → Neutral，hap と exc → Happy，sad → Sad，ang → Angry．**fru（苛立ち）は Angry に入れない**．`emotion_raw` では hap と exc を分けて持つ |
| MOSEI | CMU 配布の注釈（6 感情の強さ，0〜3）から．強さ 0.5 以上の感情が 1 つならその感情，2 つ以上なら `multi_label`（4 クラスに入れない），0 個なら `neutral` とする．**`neutral` と `multi_label` は注釈者が付けた名前ではなく，この規則で決めたもの**である（下の注意） |
| MELD | 公式 CSV の感情ラベルから |
| AFEW | 公式のフォルダ名（感情別）から |
| RAVDESS | ファイル名の 3 番目の数字から．01→Neutral，03→Happy，04→Sad，05→Angry．**02（calm）は Neutral に混ぜない**．06〜08 は `label_4` を空にする |
| CREMA-D | ファイル名の 3 番目の項目から．NEU・HAP・SAD・ANG に `label_4` を付け，DIS・FEA は空にする |

MOSEI のしきい値 0.5 は，0.0・0.5・1.0・1.5・2.0 を総当たりして前身の内訳と突き合わせ，0.5 のときだけ一致したので採用した（`features/build_manifest_mosei.py`）．
**MOSEI の Neutral は，どの感情の強さも 0.5 未満という消去法で決まる．**IEMOCAP と MELD の Neutral は注釈者が中立と判定したもの，AFEW の Neutral は「Neutral」フォルダの動画であり，ラベルの定義そのものがコーパスの間で違う．本研究が測るのはコーパス間の分布のずれなので，この違いは測定結果に直接影響する．論文では限界として書く．

RAVDESS の `sample_id` は，ファイル名の 7 つの数字から先頭（種類：01 映像+音声／02 映像のみ／03 音声のみ）を除いた 6 つ組である．同じ発話の音声ファイルと映像ファイルが同じ `sample_id` を持つようにするためである．

### 転写の出所

IEMOCAP・MOSEI・MELD の転写は人が書いたものである．AFEW には人の転写が無いので，音声認識（Whisper large-v3）の出力を使う（2026-11-04 から）．

| コーパス | 出所 |
|---|---|
| IEMOCAP | 公式配布の `transcriptions/` |
| MELD | 公式 CSV の `Utterance` 列 |
| MOSEI | CMU 配布の `label.csv` の `text` 列 |
| AFEW | `afew_transcript_whisper.csv`（Whisper large-v3．下の節） |
| RAVDESS・CREMA-D | 読む文が固定（2 種類・12 種類）なので記録はするが，言葉の特徴は作らない |

MELD の転写の 28.1% には文字化けがある（4 節「文字化けの修復」）．

### MOSEI のチャンネル対応表 `mosei_channel_map.csv`

MOSEI の生データには話者の注釈が無い．
前身の研究は YouTube から各動画のチャンネル ID を取得して話者の代わりにしていたが，その取得処理は再現できない（動画が消えている可能性もある）．
そこで取得結果を `video_id` と `channel_id` の 2 列，1,122 行の表として切り出して保存した．`features/build_manifest_mosei.py` はこの表を読むだけで，作り直さない．
チャンネルが分かるのは 3,225 本中 1,122 本（35%）だけであり，残りの動画は動画 ID を話者の代わりにする．
**このため MOSEI の話者分離は不完全である**（同じ人物の別の動画が，学習と評価に分かれて入りうる）．論文では限界として書く．
MOSEI の原論文は話者を約 1,000 のチャンネルで数える（Zadeh ほか 2018，3.1 節）が，3 モダリティの集合は 2,614 の塊（チャンネルの塊 561，動画 ID の塊 2,053）に割れ，発話の 68.5% が動画 ID の塊にある．
同じ定型の文が別の動画 ID の塊にまたがる例がある（5 語以上の書き起こしが分割をまたいで一致するのは 36 文・91 発話．感情認識の査読者の a1，2026-11-04）．チャンネルが分かる発話だけをテストにしても，同じチャンネルの別の動画が動画 ID の塊として学習側に残りうるので，漏れは 0 にならない．

### AFEW の書き起こし `afew_transcript_whisper.csv`

AFEW には人の書き起こしが無いので，`features/build_transcript_afew.py` が Whisper large-v3 で書き起こした（2026-11-04．研究者の指示．`docs/researchvision.md` 6.1 節）．
Ryumina ほか（2025，Pattern Recognition Letters 190）が公開した MER のリポジトリには，AFEW を Whisper で書き起こす台本（`get_text_AFEW.py`）がある．論文本文で AFEW の書き起こしをどう書いているかは，まだ原文で確かめていない（2026-11-04 の感情認識の査読想定の指摘 A5．`DAI/checks/計画_2026-11-04/review_affect.md`）．

- 対象は発話一覧の全 1,156 行（4 クラスに入らない行を含む）である．
- 音声は音声の特徴と同じく ffmpeg で 16kHz モノラルにし（`build_audio.decode`），正規化しない．
- モデルは `openai/whisper-large-v3` で，版はコミット `06f233fe06e710322aca913c1bc4249a0d71fce1` である（表の `asr_model` 列）．計算は fp32 である．
- 言語は英語に固定し，ビーム幅 5，サンプリングなしで解いた．同じ入力からは同じ文が出る．
- GPU（RTX A5000）で 1,437 秒かかった．

表の列は `sample_id`・`transcript_asr`（Whisper の出力そのまま）・`duration_sec`・`compression_ratio`・`asr_model` の 5 つである．
**この表を正本にし，git に入れる．**`build_manifest_afew.py` がこの表を `sample_id` で突き合わせて発話一覧の `transcript` 列を埋め，`transcript_source` 列に `whisper-large-v3` と書く．
Whisper を流し直すと GPU の丸めで文がまれに変わりうるので，`run/build_data.sh` は書き起こしを作り直さない．

**書き起こしの誤りは測らず，出力をそのまま使う**（研究者の指示）．例外は 1 つだけで，英字も数字も含まない出力（点の並びと音符の記号）は声が無い印とみなして空にする．空の発話は言葉の特徴を作らない．
4 クラスの 795 本についての内訳は次のとおりである．

| 項目 | 本数 |
|---|---:|
| 書き起こしに英字か数字がある（言葉の特徴を作る） | 790 |
| 英字も数字も含まないので空にした（点の並び 4，音符の記号 1） | 5 |
| 同じ句のくり返しの疑い（圧縮率が 2.4 を超える．Whisper 自身が失敗とみなす基準） | 33 |
| 「Thank you.」だけの文（くり返しを含む．声の無い区間で Whisper がよく作る文） | 6 |
| 単語数の中央値・最大 | 6・200 |

くり返しの例は，長さ 0.96 秒の発話に「I'm sorry.」が 50 回並ぶものである（くり返しの疑いの 33 本はどれも 4.4 秒以下）．言葉の特徴は単語方向の平均なので，同じ文のくり返しは 1 回ぶんの文とほぼ同じ特徴になる．
書き起こしの記号は ASCII の `'` だけで，Unicode の引用符・ダッシュ・三点リーダは無い（4 節「記号の表記の違い」）．

### 前身のマニフェストにあった誤り

本研究がマニフェストを生データから作り直した理由である．`measure/audit_manifest.py` は，これらの誤りを生データとの照合で検出する．

| コーパス | 誤り | 規模 |
|---|---|---|
| IEMOCAP | 台本のラベルが芝居の番号で付いていた（script01 は中身に関係なく全部 ang） | 5,255 件＝57% |
| IEMOCAP | fru（苛立ち）を Angry に入れていた | 1,849 件＝Angry の 63% |
| MELD | `sample_id` に分割名が無く重複していた | 2,576 件 |
| AFEW | 話者が作品名だった（公式の XML に俳優名があるのに） | のべ 373 本＝32% が学習と評価の両方に現れる |
| MOSEI | 誤りなし（強度 0.5 以上で立つ，という規則を総当たりで特定して再現した） | — |

AFEW の話者は，いまは公式注釈の `NameOfActor`（俳優名）である．XML と対応がつかない 4 本だけは作品名を話者にしている．

---

## 4. 特徴

### エンコーダ

**どのエンコーダも本研究のコーパスで微調整していない**（取り決め 9）．
各モダリティの解析は第 1 エンコーダで行い，結論が別のエンコーダでも成り立つかを第 2 エンコーダで最後に確かめる．

| モダリティ | | モデル | 次元 | ファイル名 | 2026-10-08 の使用状況 |
|---|---|---|---:|---|---|
| 映像 | 第 1 | ResNet-50（Ryumina らの静的表情モデル．VGGFace2 で学習し，AffectNet で微調整したもの．5 節） | 512 | `video_<コーパス>.npz` | 映像の解析に使う |
| 映像 | 第 2 | `openai/clip-vit-large-patch14` の視覚エンコーダ（`pooler_output`） | 1024 | `video_clip_<コーパス>.npz` | 特徴を作っただけ |
| 音声 | 第 1 | `microsoft/wavlm-large` | 1024 | `audio_wavlm_<コーパス>.npz` | 音声の解析に使う |
| 音声 | 第 2 | `facebook/hubert-large-ll60k` | 1024 | `audio_hubert_<コーパス>.npz` | 地図（`results/map_audio_hubert/`）とずれの指標（`results/shift_audio_hubert/`）まで計算した |
| 言葉 | 第 1 | `roberta-base` | 768 | `text_roberta_<コーパス>.npz` | 言葉の解析に使う |
| 言葉 | 第 2 | `microsoft/deberta-v3-base` | 768 | `text_deberta_<コーパス>.npz` | 特徴を作っただけ |

第 2 エンコーダで地図とずれの指標まで計算してあるのは音声の HuBERT だけであり，CLIP と DeBERTa は特徴を作っただけである．

音声と言葉の第 2 エンコーダは，別のコードではなく `--model` の切り替えで作る．
使えるモデルは `features/common.py` の `AUDIO_MODELS`・`TEXT_MODELS` に限られ，綴りが違えば実行前に止まる．出力のファイル名が短い名前（`wavlm`・`hubert` など）で分かれるので，第 1 と第 2 の特徴は混ざらない．
映像の第 2 エンコーダだけは別のコード（`features/build_video_clip.py`）であり，第 1 が保存した顔の矩形（`faces_<コーパス>.csv.gz`）を読み直して，ResNet-50 に渡したのと同じ画素を CLIP に渡す．
CLIP を選んだ理由は 2 つある．MERBench（Lian et al.）が同じ条件で比べた視覚エンコーダ 12 種のうち CLIP-large が平均 WAF 58.90 で最も高かったこと，そして CLIP は表情ラベルで学習していないので，AffectNet の表情ラベルで学習した第 1 の ResNet-50 と学習のしかたが独立していることである．

### 3 つのモダリティに共通する作り方

- エンコーダの重みは更新しない．計算は fp32 で行う．
- 時間方向（言葉では単語方向）は平均だけを特徴にし，長さの情報は別の列に記録する（映像は `n_segments`，音声は `duration_sec`・`truncated`，言葉は `n_token`・`truncated`）．
- 行は `load_manifest()` が返す行と同じ順に並べる．抽出は 2,000 行ごとに `data/cache/parts/` へ途中保存し，`--merge` で 1 本の npz にまとめる．

### 映像（第 1：`features/build_video.py`）

1. 動画から `int(fps × 5 / 25)` 枚に 1 枚のフレームを取り出す（23.98fps なら 4 枚，25fps と 29.97fps なら 5 枚，30fps なら 6 枚に 1 枚．どれも 1 秒あたり 5〜6 枚になる）．1 発話で 400 枚を超えたら，等間隔に 400 枚へ間引く（現行で MOSEI 6 件・MELD 1 件が該当する）．
2. IEMOCAP の動画は 2 人が左右に並ぶので，発話者が映る半分だけを使う．左にはマーカーを付けた人，右にはその相手が映っており，`marker_side` と `sample_id` 末尾の性別（`F000` / `M001` の先頭文字）が同じなら発話者は左にいる（`common.speaker_side`）．
3. YOLOv8n-face（確信度のしきい値 0.25）で顔を検出し，フレームごとに面積が最も大きい顔を 1 つ選ぶ．検出した顔は選ばなかったものも含めて，フレーム番号・矩形・確信度・選んだかどうかを `faces_<コーパス>.csv.gz` に保存する．顔が 1 つも無いフレームは使わない．
4. 選んだ顔を 224×224 に整えて ResNet-50 に通し，1 枚あたり 512 次元にする．
5. 顔を検出できたフレームの並びを 20 枚ずつのセグメントに切り，切る位置を 10 枚ずつずらす（コードでは `WINDOW 4 × NEED_FRAMES 5` と `STEP 2 × NEED_FRAMES 5`．前身が使った Ryumina らの実装と同じ値）．最後のセグメントが 10 枚に満たなければ捨てる．各セグメントの 512 次元を平均する．顔のフレームが 20 枚未満の発話はセグメントが 1 つになる．
6. セグメント間の平均を `X_mean`，標準偏差を `X_std` として保存する．解析に使うのは `X_mean` だけである（取り決め 4）．

### 映像（第 2：`features/build_video_clip.py`）

`faces_<コーパス>.csv.gz` から選んだ顔の矩形を読み，同じフレームから同じ矩形を切り出して CLIP ViT-L/14 の視覚エンコーダに通す（`pooler_output`，1 枚あたり 1024 次元）．
画像と文を対応づけるための射影を通した後の表現（768 次元）は，言語にそろえた空間なので使わない．
セグメントの切り方は第 1 と同じ関数（`build_video.summarize`）を使う．第 1 の抽出で顔が取れなかった発話は `ok` が偽になる．

### 音声（`features/build_audio.py`）

1. `path_audio` がある行（IEMOCAP・RAVDESS・CREMA-D）はその wav を，無い行（MOSEI・MELD・AFEW）は `path_video` を，ffmpeg で 16kHz モノラルの波形にする．
2. 先頭 60 秒で切り，切ったら `truncated` を真にする（現行で MOSEI 8 件・MELD 1 件）．元の長さは `duration_sec` に記録する．10 ミリ秒未満の波形は失敗とする．
3. 波形ごとに平均 0・標準偏差 1 に正規化し，WavLM-large（第 2 は HuBERT-large）の最終層の出力を，詰め物の部分を除いて時間方向に平均する（1024 次元）．
4. **全層の特徴（2026-11-09，DAI-13 の下調べ用）**：`features/build_audio_layers.py` が，同じ前処理で WavLM-large の 25 個の隠れ状態（入力 1 ＋ 24 層）のそれぞれを時間方向に平均し，`data/cache/audio_wavlm_layers_<コーパス>.npz` の `X_layers`（n × 25 × 1024，float32）に保存する．添字 24 は `audio_wavlm_<コーパス>.npz` の `X` と同じもの（試走 24 発話で最大の差 1.66e−5）．`load_group` は読まない（DAI-13 の台本が `sample_id` で合わせて読む）．既存の特徴のファイルは書き換えない．作るのは AFEW・IEMOCAP・MELD・MOSEI の 4 コーパス．

### 言葉（`features/build_text.py`）

1. `transcript` を正規化の段階（下）に従って整える．**解析にはすべて L1 を使う．**
2. 256 トークンを上限に切り（現行で切られたのは MOSEI の 5 件．`truncated`），RoBERTa-base（第 2 は DeBERTa-v3-base）の最終層の出力を，詰め物の部分を除いて単語方向に平均する（768 次元）．`n_token` には切った後のトークン数（開始記号と終了記号を含む）を記録する．
3. 作るのは転写のある IEMOCAP・MOSEI・MELD・AFEW である（`common.CORPORA_TEXT`．AFEW は 2026-11-04 に足した）．RAVDESS と CREMA-D は読む文が決まっていて感情の手がかりにならないので作らない．転写が空の発話は特徴を作らず，`ok` を偽にする．

### 言葉の正規化の段階

`build_text.py --level N` で L0〜L4 を選ぶ．既定は L1 である．
出力のファイル名は，L1 なら `text_<モデル>_<コーパス>.npz`，それ以外なら `text_<モデル>_L<N>_<コーパス>.npz` になる．
`data/cache/` にあるのは L1 だけであり，`measure/data.py` も L1 だけを読む．

| 段階 | やること |
|---|---|
| L0 | 何もしない |
| **L1** | 文字化けの修復のみ（**既定．解析はすべてこれを使う**） |
| L2 | L1 ＋ 引用符・ダッシュ・三点リーダを ASCII にそろえる（`common.normalize_punct`） |
| L3 | L2 ＋ `[LAUGHTER]` のような注記を削除する |
| L4 | L3 ＋ 文末に句点を補う |

感嘆符と疑問符は L4 でも消さない．感情の手がかりそのものだからである．

**言葉の特徴は L1 のままにする（2026-10-08 決定．`docs/researchvision.md` の『決定事項』）．**
理由は 2 つある．言葉が使えるのは 4 群だけなので，仕組みの仮説（MH1〜MH5）の主な検定を言葉では行えない．また，L2 にするには，言葉の特徴の抽出と，言葉を含む 4 種類（言葉，音声＋言葉，映像＋言葉，3 つの連結）の地図とずれの指標を作り直す必要がある．
代わりに，下の「記号の表記の違い」を論文で限界として書く．

### 文字化けの修復（L1）

MELD の転写には，Windows-1252 の記号が壊れて C1 制御文字（0x80〜0x9F）として混入している．
該当するのは `load_manifest()` が返す 11,352 行のうち 3,191 行（28.1%）であり，全 13,708 行では 3,741 行（27.3%）である．IEMOCAP と MOSEI には 1 行も無い（2026-10-08 に数え直した）．

| 壊れた文字 | 延べ回数（11,352 行） | 本来の文字 |
|---|---:|---|
| `\x92` | 4,192 | ’ 右引用符（`don't` の `'`） |
| `\x85` | 157 | … 三点リーダ |
| `\x97` | 134 | — 長ダッシュ |
| `\x91` | 101 | ‘ 左引用符 |
| `\x93` `\x94` `\x96` | 7・5・2 | “ ” – |

修復しないと，MELD にしか現れないトークン列ができる．RoBERTa は byte 単位の BPE なので処理自体はできるが，`company\x92s` は `company / Â / Ĵ / s` に分かれ，修復後の `company’s` は MOSEI と同じ `company / âĢ / Ļ / s` に分かれる．
DeBERTa-v3 では，修復しないと fast と slow のトークナイザで結果が食い違った（2026-09-20 に MELD 3,000 件で測り，847 件＝28.2%）．

修復は `common.fix_mojibake` が行う．
npz の `n_mojibake_fixed` は発話ごとの値ではない．`build_text.py` は 2,000 行の途中保存ごとに修復した行数を数え，その数をその 2,000 行の全行に同じ値として書いている．
MELD の 6 つの塊の値（396・518・527・569・571・610）を足すと 3,191 になる．発話ごとに知りたいときは `transcript` から数え直す．

### 記号の表記の違い（L1 では残る）

`load_manifest()` が返す行について，記号の延べ数を 2026-10-08 に数えた．

| コーパス | 行 | ASCII `'` | 壊れた `\x92` | `’` | `‘` | `“ ”` | `— –` | `…` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IEMOCAP | 5,531 | 4,194 | 0 | 0 | 0 | 0 | 0 | 0 |
| MOSEI | 18,919 | 10,178 | 0 | 1,186 | 22 | 263 | 155 | 17 |
| MELD | 11,352 | 2,397 | 4,192 | 64 | 0 | 0 | 0 | 5 |
| AFEW（Whisper．2026-11-04 に数えた） | 795 | 797 | 0 | 0 | 0 | 0 | 0 | 0 |

L1 で `\x92` を ’ に直すと，MELD も Unicode の記号を使うコーパスになる．
したがって L1 の後も，ASCII の記号だけを使う IEMOCAP・AFEW と，Unicode の記号を含む MOSEI・MELD の差が残る．
この差は言葉の特徴でコーパスを見分ける手がかりになり，感情表現の違いと区別できない．
`features/common.py` の `normalize_text` の説明には，意味を使わない表記の 16 次元だけでコーパスを均衡精度 0.7047（でたらめなら 0.333）で見分けられたという記録がある．
L2 はこの差を消すが，上の決定により解析には使わない．

### npz の中身

| キー | 形 | 持つファイル | 説明 |
|---|---|---|---|
| `X` | (n, 次元) float32 | 音声・言葉 | 特徴．失敗した行は NaN |
| `X_mean`・`X_std` | (n, 次元) float32 | 映像（第 1・第 2） | セグメント間の平均と標準偏差．**`X_std` は使わない**（取り決め 4） |
| `ok` | (n,) bool | すべて | 抽出に成功したか．**偽の行は使わない**（取り決め 3） |
| `sample_id` | (n,) str | すべて | 発話の識別子．**突き合わせは必ずこれで行う**（取り決め 2） |
| `label_4` | (n,) int8 | すべて | 抽出した時点の値．**使わない**（取り決め 1） |
| `speaker_id`・`dialog_type`・`corpus`・`split` | (n,) str | すべて | 抽出した時点の値．マニフェストから取り直す |
| `fail_reason` | (n,) str | すべて | 失敗した理由 |
| `n_segments`・`n_faces_frames` | (n,) int32 | 映像（第 1・第 2） | セグメント数と，顔を検出できたフレーム数 |
| `n_frames_sampled`・`n_frames_total`・`frame_step`・`fps`・`width`・`height` | (n,) | 映像（第 1） | 取り出したフレーム数（400 枚への間引き後）・動画の総フレーム数・間引き幅・フレームレート・画面の大きさ |
| `side`・`marker_side` | (n,) str | 映像（第 1） | IEMOCAP で使った画面の半分（`left` / `right`）とその判定に使った値 |
| `duration_sec`・`truncated` | (n,) float32・bool | 音声 | 元の長さ（秒）と，60 秒で切ったか |
| `n_token`・`truncated` | (n,) int32・bool | 言葉 | 切った後のトークン数と，256 トークンで切ったか |
| `n_mojibake_fixed` | (n,) int32 | 言葉 | 2,000 行の塊ごとの修復行数（発話ごとの値ではない） |

### 数値の約束

| 項目 | 値 | 理由 |
|---|---|---|
| 音声の長さの上限 | 60 秒 | 20 秒で切ると 462 件が切られ，うち 437 件が MOSEI なので，処理がコーパスに偏る．60 秒なら 9 件（全 42,169 件の 0.02%）で済む．MELD に 305 秒の発話があるので上限自体は要る（現行の `audio_wavlm_*.npz` の `duration_sec` で数えた） |
| 計算の精度 | fp32 | fp16 では音声の特徴が平均 1.9% ずれた（値の平均の大きさ 0.0480 に対し平均差 0.00090）．言葉では 0.12% ずれた．速度の利得は無かった（`features/build_audio.py`・`build_text.py` の記録） |
| 顔検出のしきい値 | 0.25 | 前身は 0.01 で誤検出していた．検出した顔はすべて座標を保存するので，後から変えられる |
| フレームの間引き | `int(fps × 5 / 25)` 枚に 1 枚，上限 400 枚 | 間引き幅は前身と同じ．上限は，長い発話でメモリが足りなくなるのを防ぐため |
| セグメント | 顔のフレーム 20 枚，10 枚ずつずらす | 前身と同じ |
| 言葉の長さの上限 | 256 トークン | 超えるのは MOSEI の 5 件だけ |

### 前身から変えた点

特徴は前身のキャッシュを引き継がず，生データから作り直した（2026-09-20 に方針を決めた）．主な変更は次のとおりである．各コードの先頭に，変えた点とその根拠を書いてある．**実装を変えたら，この文書とコードの先頭の説明を両方直す．**

1. **顔検出の結果（フレーム番号と矩形）を保存する．**別の表情モデルに差し替えても同じ切り出しを再現でき，「エンコーダ 1 種では結果がエンコーダの癖と区別できない」という査読の指摘に答えられる．作り直した最大の理由であり，CLIP の特徴はこれで作った．
2. **セグメント数を記録し，平均と標準偏差を分けて保存する．**前身は両者を 1024 次元に連結しており，標準偏差の 512 次元が発話の短さを表していた（取り決め 4）．
3. **IEMOCAP で発話者が映る半分を正しく選び，複数の顔を平均しない．**前身はパスを `'.'` で区切っていたため判定が常に偽になり，常に右半分を見ていた．また検出した顔をすべて平均していたので，相手や背景の人の顔が混ざっていた．
4. **音声と言葉の計算を fp32 にし，音声の上限を 20 秒から 60 秒にし，MELD の文字化けを修復した．**AFEW の言葉は Whisper の書き起こしから作り，書き起こしが空の発話では作らない（前身は AFEW の空文字から開始記号と終了記号だけの平均を作っていた）．
5. **RAVDESS と CREMA-D を足した．**前身では 2 つとも音声しか無かった．CREMA-D は映像の特徴も作ったが，RAVDESS の映像の特徴はまだ無い（7 節）．

---

## 5. 重み

`weights/` は git の対象外である．消えたら下の配布元から取り直し，リポジトリ直下で `sha256sum -c data/weights.sha256` を実行して同じファイルか確かめる（2026-10-08 に 2 本とも一致を確かめた）．

| ファイル | サイズ（バイト） | sha256 |
|---|---:|---|
| `FER_static_ResNet50_AffectNet.pt` | 98,562,934 | `8274190b5be4355bd2f07b59f593fcdb294f9d7c563bfa9ac9e5ea06c10692d2` |
| `yolov8n-face.pt` | 6,389,512 | `d545bf1add5aa736a4febac4f4f9245a6d596cd0fe70d5d57989fe0cb9e626ca` |

手元の 2 本は，Ryumina らの MER リポジトリ（README に下記の Google Drive へのリンクがある）に同梱されていた `src/weights/` から取った（2026-01 取得）．

**表情モデル `FER_static_ResNet50_AffectNet.pt`**

- 中身：Ryumina らの EMO-AffectNetModel の静的モデル．ResNet-50 を VGGFace2 で学習し，AffectNet で微調整したもの（AffectNet の検証集合で正解率 66.4%）．本研究の 7 群では微調整していない．
- 配布元：
  - https://github.com/ElenaRyumina/EMO-AffectNetModel
  - https://huggingface.co/ElenaRyumina/face_emotion_recognition （同名のファイル．サイズは一致するが，sha256 は照合していない）
  - https://drive.google.com/drive/folders/1NTVQatpihMwe5im_LZqAAouzAQWVjRwx （Ryumina らの MER リポジトリの README が示す置き場．中身は照合していない）
- 文献：Ryumina, Dresvyanskiy, Karpov (2022), *In search of a robust facial expressions recognition model: A large-scale visual cross-corpus study*, Neurocomputing

**顔検出器 `yolov8n-face.pt`**

- 中身：YOLOv8n を顔検出（顔枠と 5 点）用に WIDER FACE で学習したもの．姿勢推定版 `yolov8n-pose.pt` から作られている．
- 配布元：https://github.com/derronqi/yolov8-face （ライセンス GPL-3.0）
- ファイルに埋め込まれた学習設定（ファイル中の文字列から読んだもの．ほかに記録は無い）：

  | 項目 | 値 |
  |---|---|
  | データ設定 | `/ssd2t/derron/ultralytics/ultralytics/datasets/widerface.yaml` |
  | ベースモデル | `yolov8n-pose.pt` |
  | ultralytics の版 | 8.0.66 |
  | 作成日時 | 2023-04-07T07:07:26 |

Hugging Face のモデル（WavLM・HuBERT・RoBERTa・DeBERTa・CLIP）は，初回の実行時に自動で取得される．コードは取得する版（revision）を固定していない．`transformers` の版は `requirements.txt` で固定している．

---

## 6. 作り直し方

生データ `/mnt/d/Databases/`（読むだけ）と GPU が要る．台本は `run/build_data.sh` であり，ログは `logs/build_data.log` に追記される．
**既存の `data/` を上書きする．**特に `manifest` はラベルの正本を書き換えるので，走らせる前に git で `data/metadata/` の状態を確かめ，終わったら差分を見る．

```
./run/build_data.sh manifest    発話一覧 6 本を作り直して監査する（数分）
./run/build_data.sh check       話者の診断と話者分割の検査（数分）
./run/build_data.sh feat1       映像 第 1・音声 第 1・言葉 第 1（半日．GPU）
./run/build_data.sh feat2       映像 第 2・音声 第 2・言葉 第 2（半日．GPU）
./run/build_data.sh             全部（manifest → check → feat1 → feat2）
```

`feat2` の映像（CLIP）は `feat1` が保存した顔の座標を読むので，順序を入れ替えない．
抽出は途中保存（`data/cache/parts/`）を見て済んだ塊を飛ばすので，止まっても同じコマンドを流し直せば続きから再開する．

台本が呼ぶスクリプトは次のとおりである．

```
features/build_manifest_*.py    発話一覧（5 本のスクリプトで 6 コーパス．RAVDESS と CREMA-D は build_manifest_extra.py）
measure/audit_manifest.py       発話一覧の監査（生データと照合）
measure/check_speakers.py       話者の診断（results/speaker_audit.csv）
measure/split.py                話者分割の検査（results/split_summary.csv）
features/build_video.py         映像 第 1（顔検出 → ResNet-50）
features/build_video_clip.py    映像 第 2（保存した顔座標 → CLIP）
features/build_audio.py         音声（--model で第 1／第 2 を切り替える）
features/build_transcript_afew.py  AFEW の書き起こし（Whisper large-v3）．台本からは呼ばない．出力は git に入っている
features/build_text.py          言葉（--model で第 1／第 2，--level で正規化の段階を切り替える）
```

RAVDESS の映像の生データは `features/download_ravdess.sh` で Zenodo から取る（台本には入っていない．7 節）．
Python 環境は `.venv/` の 1 つだけであり，作り方は `requirements.txt` の冒頭にある．

**git に入っていないもの**：特徴 `data/cache/*.npz`（1.1 GB）と重み `weights/` は除外している．npz が消えたら `run/build_data.sh` で作り直すしかない（生データと GPU で丸 1 日かかる）．
顔の座標 `data/cache/faces_*.csv.gz` と発話一覧は git に入っている．**顔の座標は消さない．**表情モデルを差し替えても同じ画素を切り出せる唯一の手段だからである．

---

## 7. 既知の限界と注意点

- **RAVDESS の映像の特徴は無い．**生データは取得済みである（`/mnt/d/Databases/RAVDESS/video_speech/Actor_01`〜`Actor_24`，mp4 2,880 本．`features/download_ravdess.sh` で取った．Zenodo の API 形式の URL は 403 を返したが，`/records/…/files/…?download=1` 形式の URL で取れた）．
  しかし発話一覧の RAVDESS 1,440 行はすべて `path_video` が空であり，`video_RAVDESS.npz` と `video_clip_RAVDESS.npz` は無い．
  映像の特徴を足すかは保留している（`docs/researchvision.md` の『未決事項』）．足すと映像の群が 6 から 7 に増え，映像と音声＋映像の地図とずれの指標を作り直す必要がある．
- **言葉の特徴は 5 群（IEMOCAP の 2 群・MOSEI・MELD・AFEW）にある．**RAVDESS と CREMA-D は読む文が 2 種類・12 種類しかないので作らない．
- **AFEW の言葉だけは機械の書き起こし（Whisper large-v3）から作る．**ほかの 3 コーパスは人の書き起こしである．AFEW をターゲットやソースにした言葉の得と損には，コーパスの違いと書き起こしの作り方の違い（認識の誤り，同じ句のくり返し 33 本など）の両方が入り，分けられない．書き起こしの誤りは測っていない（3 節「AFEW の書き起こし」）．論文では限界として書く．
- **言葉を含む融合特徴の標準化の統計に，2026-11-04 から AFEW が入った．**`measure/data.py` の `_fuse_stats` は，その種類の特徴がある全群をまとめて平均と標準偏差を作る．AFEW の 790 発話が加わったので，既存の 4 群の融合特徴は列ごとに一定の倍率（0.94〜1.006）とずらし（絶対値 0.04 以下）だけ変わった（MELD と IEMOCAP 即興で，AVT・AT・第 2 エンコーダの AVT を確かめた）．分類器は学習データで，距離を使う指標の多くは 2 標本を合わせて標準化し直すので，結果は計算の丸めの範囲でしか変わらない．生の特徴のまま測るずれの指標（`measure/shift.py` の `space="raw"`）だけは，計算し直すと 2026-11-03 までの値から少し変わる．
- **`video_IEMOCAP.npz` は古いマニフェスト（9,169 行）から作った．**行順と `label_4` が現行のマニフェストと違うが，取り決め 1・2 を守れば結果に影響しない．`video_clip_IEMOCAP.npz` は現行のマニフェストから作った．
- **MOSEI の話者分離は不完全である．**チャンネルが分かる動画は 35% だけで，残りは動画 ID を話者の代わりにしている（3 節）．
- **MOSEI の Neutral は消去法で決まる．**ほかのコーパスの Neutral と定義が違う（3 節「ラベルの決め方」）．
- **AFEW の 4 本は話者が作品名である．**公式の XML と対応がつかなかった（3 節）．
- **言葉の記号の表記がコーパスで違う．**言葉の特徴は L1 のままなので，IEMOCAP（ASCII の記号だけ）と MOSEI・MELD（Unicode の記号を含む）の差が残っている．論文では限界として書く（4 節「記号の表記の違い」）．
- **Hugging Face のモデルの版を固定していない．**取り直すと，配布元の更新によって特徴が変わる可能性がある（5 節）．
