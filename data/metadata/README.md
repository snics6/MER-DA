# 発話一覧（マニフェスト）

1 行が 1 発話で，動画・音声の場所，話者，ラベルが入っている。
`features/common.py` の `load_manifest()` がここを読み，**返す行順が以降すべての
出力の行順になる**（映像・音声・言葉の npz はすべて同じ順序で並ぶ）。

---

## ファイル

**6 つとも本リポジトリのコードで生成する。**前身が作った CSV はもう使っていない
（2026-09-21 に全部作り直した）。

| ファイル | 全行 | **使う行** | 生成コード |
|---|---:|---:|---|
| `iemocap_utterance_manifest.csv` | 10,039 | **5,531** | `features/build_manifest_iemocap.py` |
| `mosei_utterance_manifest.csv` | 22,856 | 18,919 | `features/build_manifest_mosei.py` |
| `meld_utterance_manifest.csv` | 13,708 | 11,352 | `features/build_manifest_meld.py` |
| `afew_utterance_manifest.csv` | 1,156 | 795 | `features/build_manifest_afew.py` |
| `ravdess_utterance_manifest.csv` | 672 | 672 | `features/build_manifest_extra.py` |
| `crema-d_utterance_manifest.csv` | 4,900 | 4,900 | 同上 |

**「使う行」は `load_manifest()` が返す行数**である（4 クラスに入り，映像か音声がある行）。
**`docs/paper-notes.md` の数値はこの母数で数える。**

    ./run/pipeline.sh manifest      # 6 つとも作り直して監査する

### 前身のマニフェストにあった誤り（作り直した理由）

| コーパス | 誤り | 規模 |
|---|---|---|
| **IEMOCAP** | 台本のラベルが**芝居の番号**で付いていた（script01 は中身に関係なく全部 ang） | **5,255 件＝57%** |
| **IEMOCAP** | fru（苛立ち）を Angry に潰していた | Angry の 63% |
| **MELD** | `sample_id` に分割名が無く重複 | **2,576 件** |
| **AFEW** | 話者が**作品名**だった（公式 XML に俳優名があるのに） | のべ 373 本＝32% が漏れ |
| MOSEI | 誤りなし（強度 0.5 以上で立つ，という規則を総当たりで特定し再現した） | — |

詳細は `docs/paper-notes.md`。監査は `measure/audit_manifest.py`（生データと照合する）。

### `mosei_channel_map.csv`

MOSEI の話者は生データに無い。YouTube のチャンネル ID を `video_id` に対応づけた表で，
**取得の処理は再現できない**（動画が消えている可能性もある）ので切り出して保存してある。
**分かるのは 3,225 本中 1,122 本（35%）だけ**で，残りは動画 ID を話者の代わりにする。
**MOSEI の話者分離は不完全である**（`docs/paper-notes.md` に限界として書く）。

---

## 列

`load_manifest()` が読むのは次の 11 列。無い列は空で埋められる。

| 列 | 中身 |
|---|---|
| `sample_id` | 発話の名前。コーパス内で一意 |
| `corpus` | `IEMOCAP` / `MOSEI` / `MELD` / `AFEW` / `RAVDESS` / `CREMA-D` |
| `split` | 元のコーパスの区分。**本研究は話者で切るので，この列で学習と評価を分けない** |
| `speaker_id` | 話者。**話者分割はこの列で行う**（`CLAUDE.md` 取り決め 2） |
| `label_4` | 0=中立 1=喜び 2=悲しみ 3=怒り。**空の行は落とされる** |
| `dialog_type` | IEMOCAP のみ `impro` / `script`。**必ず分ける**（取り決め 5） |
| `marker_side` | IEMOCAP のみ。マーカーを付けた人の性別。**発話者が画面の左右どちらかを決めるのに使う**（`common.speaker_side`） |
| `path_video` | 映像ファイル。**無ければ空**（RAVDESS はいま全件空） |
| `path_audio` | 音声ファイル。**無ければ空**。空のときは `path_video` から ffmpeg で取り出す |
| `transcript` | 人が書き起こした文。AFEW は全件空 |
| `exclude` | `false` 以外の行は落とされる |

`label_4` が空，または映像も音声も無い行は，読み込み時に落とされる。

---

## ラベルの決め方

| コーパス | 決め方 |
|---|---|
| IEMOCAP | 注釈から。**台本（script）のラベルは注釈ではなく芝居の番号で決まっている**ので，即興と混ぜられない |
| MOSEI | CMU 配布の注釈から |
| MELD | 公式 CSV の感情ラベルから |
| AFEW | 公式のフォルダ名（感情別）から |
| RAVDESS | ファイル名の 3 番目の数字。01→中立 03→喜び 04→悲しみ 05→怒り。**02（calm）は中立に混ぜない**。06〜08 は落とす |
| CREMA-D | ファイル名の 3 番目。NEU/HAP/SAD/ANG を採用。DIS・FEA は落とす |

---

## 文字起こしの出所と前処理

**すべて人が書いたもの**で，音声認識の出力ではない。

| コーパス | 出所 |
|---|---|
| IEMOCAP | 公式配布の `transcriptions/` |
| MELD | 公式 CSV の `Utterance` 列 |
| MOSEI | CMU 配布の `label.csv` の `text` 列 |
| AFEW | **無い**（795 件すべて空） |
| RAVDESS・CREMA-D | 読む文が固定（2 種・12 種）なので記録はするが**測らない** |

**MELD の転写の 28.1% に文字化けがある。**修復の実装と実測値は
[../../docs/feature-spec.md](../../docs/feature-spec.md)「言葉の前処理」を見ること。

---

## 前身の README について

前身が書いた同名の文書は [../../inherited/metadata-README-from-predecessor.md](../../inherited/metadata-README-from-predecessor.md)
に移した。**読むだけ。**本リポジトリに存在しない場所（`scripts/preprocessing/`・
`D:/CrossCorpusMER/features/`・`docs/data/`）を正本として指しているので，
**そこに書かれた手順をそのまま実行しない。**
