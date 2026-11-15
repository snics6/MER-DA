#!/bin/bash
# E82（模擬査読 S4-3：P4 の entmin・IM・自己学習の C′ を E39 の値の 3 倍・10 倍に広げて選び直す）を RunPod の H100 で回す．
# 設計は docs/exploratory-analyses.md の E82，台本は exp/s43_penalty.py，実装の記録は checks/E82/impl.md．
# 引数は流れの番号 1・2（2 本を並べて流す）．終わると logs/e82h_<番号>.done を書く．失敗は logs/e82h_fail.log に 1 行ずつ書く（E77 の run/e77h.sh と同じ形）．
# 配分（H100 の見積もり．checks/E82/impl.md の 10 節）：流れ 1 ＝ AVT（約 2.2 時間）＋ WavLM（1.8），流れ 2 ＝ AV（2.2）＋ 映像（0.9）＋ RoBERTa（0.8）．
# 各実行で，最初に IEMOCAP-impro だけを流し，--check で E77 の none・鍵の重なり・GPU の型と照らしてから残りのターゲットを流す
# （照合が落ちたらその実行の残りを流さない）．最後に全ターゲットを照らす．
#
# 準備：run/e82_setup.sh <コミット> <git bundle> を Pod A で 1 回だけ流す（絶対の場所と set -eu だけで書いた．関門 gate-code の再検査の指摘）．
#   別の作業木 /workspace/MER-DA-e82 を作り，.venv・results を /workspace/MER-DA のものへのリンクに，data/cache の npz をリンクにする．
# GPU の型の照合は E77 の記録を全部数えるので，流す前に results/e77_<実行> の provenance の gpu が 1 種類であることを確かめる．
# 流し方：tmux で 2 本（例 tmux new -d -s e82h1 'cd /workspace/MER-DA-e82 && bash run/e82h.sh 1'，e82h2 も同じ）．
# E77 の評価の出力（results/e77_<実行>/IEMOCAP-impro.csv）が無い実行は流さない（照合ができない）．GPU の割り当ては 1 本 0.2（E77 と同じ GPU で
# 同時に流すときは，E77 の流れの --gpu-fraction との和が 1 を超えないようにする）．試しの出力を results/e82_* に書かない（--allow-dirty の記録が残ると読み手が止まる）．
set -u
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
F="--ks 0 5 --copy-pseudo --nc-seeds 1 2 --gpu-fraction 0.2"
DRY=${E82H_DRY:-0}
FAIL=logs/e82h_fail.log
mkdir -p logs
step() { local log=$1; shift; if [ "$DRY" = 1 ]; then echo "+ $* >> $log 2>&1"; return 0; fi; "$@" >> "$log" 2>&1; }   # DRY は表示だけ
note() { local log=$1; shift; if [ "$DRY" = 1 ]; then echo "（DRY）$log に書く：$*"; else echo "$*" >> "$log"; fi; }
fail() { if [ "$DRY" = 1 ]; then echo "（DRY）$FAIL に書く：失敗 $*"; else echo "失敗 $(date '+%F %T') $*" >> $FAIL; fi; }

precheck() {
  # 作業木の準備ができているか（DRY では表示だけ）
  local bad=""
  [ -L results ] || bad="$bad results が共有のリンクでない;"
  [ -x "$PY" ] || bad="$bad $PY が無い;"
  ls data/cache/*.npz > /dev/null 2>&1 || bad="$bad data/cache に特徴の npz が無い;"
  [ -z "$(git status --porcelain --untracked-files=no -- exp measure)" ] || bad="$bad exp/・measure/ に変更がある;"
  [ "$(pwd -P)" != "/workspace/MER-DA" ] || bad="$bad E77 の作業木で流そうとしている;"
  if [ -n "$bad" ]; then
    if [ "$DRY" = 1 ]; then echo "（DRY）準備の確かめ：$bad"; else fail "準備：$bad"; exit 1; fi
  else
    echo "準備の確かめ：ok（$(git rev-parse --short HEAD)）"
  fi
}

run() {
  # run <名前> --kind|--world <実行>
  local name=$1; shift
  local what=$2 log=logs/e82h_${name}.log e77=results/e77_$2
  if [ ! -f "$e77/IEMOCAP-impro.csv" ]; then
    fail "$name：E77 の出力 $e77/IEMOCAP-impro.csv が無いので流さない"
    [ "$DRY" = 1 ] || return
  fi
  note $log "開始 $(date '+%F %T') $name $what"
  step $log $PY exp/s43_penalty.py "$@" --targets IEMOCAP-impro $F || { fail "$name IEMOCAP-impro $* $?"; return; }
  step $log $PY exp/s43_penalty.py "$@" --check "$e77" || { fail "$name 照合（IEMOCAP-impro）$*：E77 の none・鍵・GPU の型と食い違う（$log と results/e82_$what/check.csv）．残りを流さない"; return; }
  step $log $PY exp/s43_penalty.py "$@" $F || { fail "$name 残りのターゲット $* $?"; return; }   # 終わった IEMOCAP-impro は同じコミットなら飛ばす（check_resume）
  step $log $PY exp/s43_penalty.py "$@" --check "$e77" || fail "$name 照合（全ターゲット）$*（$log と results/e82_$what/check.csv）"
  note $log "終了 $(date '+%F %T') $name"
}

precheck
case ${1:-} in
  1) run avt --world fuse-audio-video-text; run audio --kind audio_wavlm ;;
  2) run av --world fuse-audio-video; run video --kind video; run text --kind text_roberta ;;
  *) echo "引数は流れの番号 1 か 2"; exit 2 ;;
esac
if [ "$DRY" = 1 ]; then echo "（DRY）logs/e82h_$1.done を書く"; else date "+終了 %F %T" > logs/e82h_$1.done; fi
