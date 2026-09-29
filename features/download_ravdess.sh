#!/usr/bin/env bash
# RAVDESS の映像（俳優 24 人・各 0.5〜0.6 GB・計 13.4 GB）を Zenodo から取る。
#
# **1 本ずつ取ること。**前身では 25 本を一気に叩いて 403（"unusual traffic from
# your network"）でブロックされた。1 本あたり数分かかるので全体で 2〜3 時間。
#
# 途中で止まっても，同じコマンドで続きから再開する（展開済みの俳優は飛ばす）。
# ライセンス: CC BY-NC-SA 4.0（非商用）。Zenodo record 1188976。
set -uo pipefail

DEST=/mnt/d/Databases/RAVDESS
REC=1188976
mkdir -p "$DEST/video_speech"
cd "$DEST" || exit 1

for i in $(seq -w 1 24); do
  f="Video_Speech_Actor_${i}.zip"
  d="video_speech/Actor_${i}"
  if [ -d "$d" ]; then
    echo "[$(date +%F' '%T)] Actor_${i} 展開済み．飛ばす"
    continue
  fi
  if [ ! -f "$f" ]; then
    echo "[$(date +%F' '%T)] $f 取得開始"
    if ! curl -sSL --fail -A "Mozilla/5.0" --retry 3 --retry-delay 60 --max-time 1800 \
         -o "${f}.part" "https://zenodo.org/api/records/${REC}/files/${f}/content"; then
      echo "[$(date +%F' '%T)] 取得失敗: $f（次へ進む）"
      rm -f "${f}.part"
      sleep 60
      continue
    fi
    mv "${f}.part" "$f"
    echo "[$(date +%F' '%T)] $f 取得完了 $(du -h "$f" | cut -f1)"
  fi
  unzip -q -o "$f" -d video_speech && echo "[$(date +%F' '%T)] Actor_${i} 展開完了"
  sleep 10
done

echo "[$(date +%F' '%T)] 全件終了"
echo "俳優ディレクトリ: $(ls -d video_speech/Actor_* 2>/dev/null | wc -l) / 24"
echo "動画ファイル: $(find video_speech -name '*.mp4' | wc -l) 本"
