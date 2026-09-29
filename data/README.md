# data/ — 特徴キャッシュとマニフェスト

**`~/MER/data/` を 2026-09-29 に複製したものである．**ファイル名・配置・中身は MER と
同一なので，MER の読み込みコード（`measure/data.py`）がそのまま動く．

元をたどれば生データ `/mnt/d/Databases/`（読むだけ）から `~/MER/features/build_*.py` が
作ったものである．**ここにあるのはすべてその成果物であり，生データは含まない．**

---

## 中身

| 場所 | 中身 | 大きさ |
|---|---|---|
| `metadata/*.csv` | 発話の一覧 7 本．**ラベルと話者の正本** | 15 MB |
| `cache/*.npz` | 特徴 28 本 | 1.1 GB |
| `cache/faces_*.csv.gz` | 顔検出の結果（フレーム番号と矩形座標）5 本 | 17.6 MB |

MER の `cache/parts/`（抽出の途中経過 159 本・1.1 GB）は複製していない．統合済みなので
完成 npz があれば要らない．**顔の座標は消さないこと．**表情モデルを差し替えても
まったく同じ画素を切り出せる唯一の手段であり，第 2 エンコーダはこれで作られている．

## エンコーダ

**どれも 6 コーパスで微調整していない．**微調整済みを使うと「データのずれ」と
「エンコーダの癖」が分離できなくなる（MER `CLAUDE.md` 取り決め 1）．

| モダリティ | | モデル | 次元 | ファイル名 |
|---|---|---|---:|---|
| 映像 | 第 1 | ResNet-50（AffectNet 事前学習） | 512 | `video_<コーパス>.npz` |
| 映像 | 第 2 | `openai/clip-vit-large-patch14` の視覚側 | 1024 | `video_clip_<コーパス>.npz` |
| 音声 | 第 1 | `microsoft/wavlm-large` | 1024 | `audio_wavlm_<コーパス>.npz` |
| 音声 | 第 2 | `facebook/hubert-large-ll60k` | 1024 | `audio_hubert_<コーパス>.npz` |
| 言葉 | 第 1 | `roberta-base` | 768 | `text_roberta_<コーパス>.npz` |
| 言葉 | 第 2 | `microsoft/deberta-v3-base` | 768 | `text_deberta_<コーパス>.npz` |

映像は顔を検出して切り出し，4 フレーム窓・2 フレーム送りのセグメントに分けて通し，
セグメント方向の平均（`X_mean`）と標準偏差（`X_std`）を別々に保存してある．
音声は 60 秒で打ち切り，計算は fp32．言葉は文字化けの修復のみを行った L1 が既定である
（MELD の転写の 28.1% に Windows-1252 の壊れた文字が混入していたため）．

## 件数（行数／`ok` が真の行／成功率）

```
video_IEMOCAP         9169   9092  99.2%      audio_wavlm_IEMOCAP   5531   5531 100.0%
video_MOSEI          18919  18634  98.5%      audio_wavlm_MOSEI    18919  18919 100.0%
video_MELD           11352  11316  99.7%      audio_wavlm_MELD     11352  11351 100.0%
video_AFEW             795    795 100.0%      audio_wavlm_AFEW       795    795 100.0%
video_CREMA-D         4900   4899 100.0%      audio_wavlm_CREMA-D   4900   4900 100.0%
                                              audio_wavlm_RAVDESS    672    672 100.0%
video_clip_IEMOCAP    5531   5492  99.3%
video_clip_MOSEI     18919  18634  98.5%      text_roberta_IEMOCAP  5531   5531 100.0%
video_clip_MELD      11352  11316  99.7%      text_roberta_MOSEI   18919  18919 100.0%
video_clip_AFEW        795    795 100.0%      text_roberta_MELD    11352  11352 100.0%
video_clip_CREMA-D    4900   4899 100.0%
```

第 2 エンコーダ（`audio_hubert_*`・`text_deberta_*`）の行数は第 1 と同じである．

**無いもの**

- RAVDESS の映像（Zenodo が 403 を返し取得できなかった）
- AFEW・RAVDESS・CREMA-D の言葉（転写が無い，または読む文が 2 種類・12 種類しかない）

## npz の中身

| キー | 形 | 説明 |
|---|---|---|
| `X` または `X_mean`・`X_std` | (n, 次元) float32 | 特徴．映像だけ平均と標準偏差に分かれる |
| `ok` | (n,) bool | 抽出に成功したか．**偽の行は使わない** |
| `sample_id` | (n,) str | 発話の識別子．**突き合わせは必ずこれで行う** |
| `label_4` | (n,) int8 | 0=Neutral 1=Happy 2=Sad 3=Angry．**焼き付きなので使わない**（下記 1） |
| `speaker_id`・`dialog_type`・`corpus`・`split` | (n,) str | 同上，マニフェストから取り直す |
| `fail_reason` | (n,) str | 失敗した理由 |
| `n_segments`・`n_faces_frames`・`fps` ほか | (n,) int32 | 映像の補助情報 |
| `duration_sec`・`truncated` | (n,) | 音声の補助情報 |

---

## 使い方

**`np.load` して `X` をそのまま使ってはいけない．**次の 5 つを必ず通す．
MER では `measure/data.py` の `load_group(群, エンコーダ)` が一手に引き受けている．

1. **ラベルをマニフェストから貼り直す**（下記 1）
2. `ok` が偽の行と，特徴に NaN・Inf のある行を落とす
3. IEMOCAP は**即興と台本に分ける**（下記 5）
4. 話者グループの列を付け，**学習側と評価側でそのグループを分ける**
5. **`sample_id` で並べ直す**（下記 2）

群は次の 7 つで数える．

```
IEMOCAP-impro / IEMOCAP-script / MOSEI / MELD / AFEW / RAVDESS / CREMA-D
```

---

## 注意点

1. **ラベルの正本は `metadata/*.csv` だけである．**npz の `label_4` は抽出した時点の
   ものが焼き付いており，マニフェストを直しても古いまま残る．MER の前身は
   ここが崩れて IEMOCAP の 57% に誤ったラベルが付いていた．**必ず貼り直す．**

2. **npz の行順は抽出したときのファイル走査順であり，モダリティごとに違う．**
   行番号で標本を選ぶと再現しない．**`sample_id` で並べ直してから使う．**
   実際 `video_IEMOCAP.npz` は古いマニフェスト（9,169 行）で作られており，
   音声・言葉（5,531 行）と行数からして違う．

3. **`X_std` は使わない．**セグメントが 1 つしかない発話では必ずゼロになり，
   その割合はコーパスで 5 倍違う（AFEW 89.8%・MOSEI 17.9%）．
   つまりこの次元は**発話の長さの代理変数**であって表情ではない．
   連結すると，コーパス間の距離の一部が「動画の長さの違い」で説明されてしまう．

4. **話者は必ず分ける．**同じ話者が学習側と評価側の両方に現れてはならない．
   分けないと「この人は誰か」を当てているだけで高い精度が出る．

5. **IEMOCAP は即興と台本を分ける．**台本では 10 人全員が同じ台詞を読むので，
   言葉の分類器が台詞を覚えるだけで当たってしまう．

6. **k はクラスあたりの本数で数える．**k=25 なら 4 クラス × 25 ＝ 100 本．

7. **ラベルは 4 つ，指標は均衡精度（UAR）．**クラスごとの正解率を平均したもので，
   でたらめに答えれば 25% になる．

8. **正本は `~/MER/data/` 側である．**マニフェストを直したり特徴を作り直したりしたら，
   MER 側で行い，ここへ複製し直す．**片方だけ直すとラベルの正本が 2 つになる．**

---

## 作り直し方

MER 側で行う．生データ `/mnt/d/Databases/` と GPU が要る．

```
~/MER/features/build_manifest_iemocap.py    発話の一覧（コーパスごとに 5 本）
~/MER/features/build_video.py               映像 第 1（顔検出 → ResNet-50）
~/MER/features/build_video_clip.py          映像 第 2（保存した顔座標 → CLIP）
~/MER/features/build_audio.py               音声（--model で第 1／第 2 を切替）
~/MER/features/build_text.py                言葉（--model で第 1／第 2 を切替）
```

作り方の正本は `~/MER/docs/feature-spec.md`．**重みは `~/MER/weights/` にある**
（`FER_static_ResNet50_AffectNet.pt`・`yolov8n-face.pt`．git 対象外）．
