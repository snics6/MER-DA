#!/usr/bin/env bash
# DAI-13 の下調べの本計算（設計 DAI/docs/実験13_下調べ_設計.md．台本 DAI/exp/dai13_probe.py）
# 単位の段（種 r = 0 の 60 単位）→ 集計の段 を続けて流す．終わると logs/dai13_probe.done を作る（失敗は logs/dai13_probe.failed）．
# 検査を通した台本の sha256 を DAI13_SHA で必ず渡す．
set -euo pipefail
cd /home/snics/MER-DA
mkdir -p logs
trap 'touch logs/dai13_probe.failed' ERR
OUT=results/dai13_probe
if [ -e "$OUT" ]; then
  echo "$OUT は既にある（本計算は空の出力先に書く）"
  touch logs/dai13_probe.failed
  exit 1
fi
if [ -z "${DAI13_SHA:-}" ]; then
  echo "DAI13_SHA（検査を通した台本の sha256）が無い"
  touch logs/dai13_probe.failed
  exit 1
fi
ACTUAL_SHA=$(sha256sum DAI/exp/dai13_probe.py | cut -d' ' -f1)
if [ "$ACTUAL_SHA" != "$DAI13_SHA" ]; then
  echo "台本の sha256 が検査を通した版と違う：$ACTUAL_SHA"
  touch logs/dai13_probe.failed
  exit 1
fi
if [ ! -e logs/dai13_feat.done ]; then
  echo "全層の特徴の取り出しが終わっていない（logs/dai13_feat.done が無い）"
  touch logs/dai13_probe.failed
  exit 1
fi
if [ ! -e DAI/checks/DAI-12/code_post.md ]; then
  echo "DAI-12 の計算の後の点検（DAI/checks/DAI-12/code_post.md）が無い"
  touch logs/dai13_probe.failed
  exit 1
fi
VERDICT=$(awk '/^## *判定/{f=1; next} f && NF {print; exit}' DAI/checks/DAI-12/code_post.md)
if ! echo "$VERDICT" | grep -q "通す"; then
  echo "DAI-12 の計算の後の点検の判定が「通す」でない（見出し「## 判定」の次の行：$VERDICT）"
  touch logs/dai13_probe.failed
  exit 1
fi
if [ ! -e results/dai12_probe/provenance.json ]; then
  echo "DAI-12 の provenance.json が無い（集計の段が終わっていない）"
  touch logs/dai13_probe.failed
  exit 1
fi
if [ ! -s "DAI/docs/実験13_下調べ_設計.md" ]; then
  echo "設計の文書が無い"
  touch logs/dai13_probe.failed
  exit 1
fi
echo "設計の sha256: $(sha256sum "DAI/docs/実験13_下調べ_設計.md" | cut -d' ' -f1)"
N12=$(ls results/dai12_probe/rows/*.npz 2>/dev/null | wc -l)
if [ "$N12" != "120" ]; then
  echo "DAI-12 の保存が $N12 単位（120 のはず）"
  touch logs/dai13_probe.failed
  exit 1
fi
export CUDA_VISIBLE_DEVICES=0
echo "開始 $(date '+%F %T')"
.venv/bin/python DAI/exp/dai13_probe.py --out "$OUT" --stage units > logs/dai13_probe_units.log 2>&1
echo "単位の段を終えた $(date '+%F %T')"
.venv/bin/python DAI/exp/dai13_probe.py --out "$OUT" --stage summarize > logs/dai13_probe_sum.log 2>&1
echo "集計の段を終えた $(date '+%F %T')"
touch logs/dai13_probe.done
