#!/usr/bin/env bash
# 実験 19 の下調べ：3 つの腕の出力を研究室の PC に戻した後に，検算（eq の腕と実験 1・16・17 の保存）と集計を流す．
# 終わると logs/dai19_post.done（失敗は logs/dai19_post.failed）．
set -euo pipefail
cd /home/snics/MER-DA
mkdir -p logs
trap 'touch logs/dai19_post.failed' ERR
rm -f logs/dai19_post.done logs/dai19_post.failed
[ -n "${SHA19:-}" ] || { echo "SHA19 が渡されていない"; touch logs/dai19_post.failed; exit 1; }
[ "$(sha256sum DAI/exp/dai19_probe.py | cut -d' ' -f1)" = "$SHA19" ] || { echo "dai19_probe.py の sha256 が違う"; touch logs/dai19_post.failed; exit 1; }
PY=.venv/bin/python
# 実験 1・16・17 の置き場が書き換わらないことを，前後の sha256 で確かめる
snap() { find results/dai1/source results/dai1/pseudo results/dai16_probe results/dai17_probe results/dai16_probe_k10 results/dai17_probe_k10 DAI/frozen -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1; }
BEFORE=$(snap)
for STAGE in verify summary; do
  echo "=== dai19 $STAGE $(date '+%F %T')"
  $PY DAI/exp/dai19_probe.py --stage "$STAGE" 2>&1 | tee -a "logs/dai19_${STAGE}.log"
done
AFTER=$(snap)
[ "$BEFORE" = "$AFTER" ] || { echo "実験 1・16・17 の置き場が書き換わった"; touch logs/dai19_post.failed; exit 1; }
date '+%F %T' > logs/dai19_post.done
echo "EXIT=0"
