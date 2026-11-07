#!/usr/bin/env bash
# DAI-7 の下調べの正の計算（`DAI/docs/実験7_下調べ_設計.md` 第 8 版，台本 `DAI/exp/dai7_probe_v2.py`）．
# 関門（gate-probe・gate-code・dai-review-ml・dai-review-affect）を通した版．研究者の確認 2026-11-07「どうぞ」．
# GPU は使わない（gate-code の試走と同じ CPU の設定）．集計の段は判定の表を画面に出さず，ファイルにだけ書く．
# 終わったら logs/dai7_probe_v2.done ができる（区切りはこのファイルで待つ）．
# 使い方：tmux の mer-da で  bash DAI/run/dai7_probe_v2.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=results/dai7_probe_v2
LOG=logs/dai7_probe_v2.log
if [ -e "$OUT" ]; then echo "$OUT は既にある（上書きしない）"; exit 1; fi
mkdir -p logs
export CUDA_VISIBLE_DEVICES=""
echo "開始 $(date '+%F %T') コミット $(git rev-parse --short HEAD) python $(./.venv/bin/python -c 'import sys,scipy,pandas;print(sys.version.split()[0],"scipy",scipy.__version__,"pandas",pandas.__version__)')" | tee -a "$LOG"
./.venv/bin/python DAI/exp/dai7_probe_v2.py --out "$OUT" 2>&1 | tee -a "$LOG"
./.venv/bin/python DAI/exp/dai7_probe_v2.py --out "$OUT" --summarize 2>&1 | tee -a "$LOG"
echo "終了 $(date '+%F %T')" | tee -a "$LOG"
touch logs/dai7_probe_v2.done
