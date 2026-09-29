# 重みファイルの出所

`weights/` は git 対象外である．消えたら下の配布元から取り直し，
`sha256sum -c docs/weights.sha256` で同じものか確かめる（リポジトリ直下で実行）．

| ファイル | サイズ（バイト） | sha256 |
|---|---:|---|
| `FER_static_ResNet50_AffectNet.pt` | 98,562,934 | `8274190b5be4355bd2f07b59f593fcdb294f9d7c563bfa9ac9e5ea06c10692d2` |
| `yolov8n-face.pt` | 6,389,512 | `d545bf1add5aa736a4febac4f4f9245a6d596cd0fe70d5d57989fe0cb9e626ca` |

手元の 2 本は，Ryumina らの MER リポジトリ（README に下記 Google Drive へのリンクあり）に
同梱されていた `src/weights/` から来ている（2026-01 取得）．

## 表情モデル `FER_static_ResNet50_AffectNet.pt`

- 何か：Ryumina らの EMO-AffectNetModel の静的モデル．ResNet-50 を VGGFace2 で学習し，
  **AffectNet で微調整したもの**（AffectNet 検証集合で正解率 66.4%）．本研究の 7 群では微調整していない
- 配布元：
  - https://github.com/ElenaRyumina/EMO-AffectNetModel
  - https://huggingface.co/ElenaRyumina/face_emotion_recognition （同名ファイル．**サイズは一致，sha256 は未照合**）
  - https://drive.google.com/drive/folders/1NTVQatpihMwe5im_LZqAAouzAQWVjRwx （Ryumina らの MER リポジトリの README が示す置き場．**中身は未照合**）
- 文献：Ryumina, Dresvyanskiy, Karpov (2022), *In search of a robust facial expressions recognition model: A large-scale visual cross-corpus study*, Neurocomputing

## 顔検出 `yolov8n-face.pt`

- 何か：YOLOv8n を顔検出（顔枠＋5 点）用に WIDER FACE で学習したもの．姿勢推定版 `yolov8n-pose.pt` から作られている
- 配布元：https://github.com/derronqi/yolov8-face （ライセンス GPL-3.0）
- ファイルに埋め込まれた学習設定（中身の文字列から読んだもの．他に記録は無い）：

  | 項目 | 値 |
  |---|---|
  | データ設定 | `/ssd2t/derron/ultralytics/ultralytics/datasets/widerface.yaml` |
  | ベースモデル | `yolov8n-pose.pt` |
  | ultralytics の版 | 8.0.66 |
  | 作成日時 | 2023-04-07T07:07:26 |
