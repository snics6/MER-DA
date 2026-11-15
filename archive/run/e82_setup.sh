#!/bin/bash
# E82 の本番の前に，RunPod の Pod A で別の作業木を 1 回だけ作る（関門 gate-code の再検査の指摘で，run/e82h.sh の冒頭の手順から分けた）．
# 絶対の場所だけを使い，どの段で失敗しても止まる．E77 が動いている /workspace/MER-DA の作業木と HEAD には触らない
# （コミットは fetch でオブジェクトだけを入れ，FETCH_HEAD から作業木を作る）．
# 使い方：bash run/e82_setup.sh <コミット> <研究室から送った git bundle の絶対の場所>
set -eu
C=${1:?コミットを渡す}
B=${2:?git bundle の場所を渡す}
SRC=/workspace/MER-DA
W=/workspace/MER-DA-e82
[ "$W" != "$SRC" ]
[ -d "$SRC/.git" ] || { echo "$SRC が git の作業木でない"; exit 1; }
[ ! -e "$W" ] || { echo "$W が既にある．中身を確かめてから手で消す"; exit 1; }
git -C "$SRC" fetch "$B" main
git -C "$SRC" cat-file -e "$C^{commit}"
git -C "$SRC" worktree add --detach "$W" "$C"
[ "$(git -C "$W" rev-parse HEAD)" = "$(git -C "$SRC" rev-parse "$C")" ]
ln -s "$SRC/.venv" "$W/.venv"
rm -r -- "$W/results"
ln -s "$SRC/results" "$W/results"
mkdir -p "$W/data/cache" "$W/logs"
for f in "$SRC"/data/cache/*.npz; do ln -s "$f" "$W/data/cache/"; done
git -C "$W" status --porcelain --untracked-files=no -- exp measure | { ! grep -q .; }
echo "作業木 $W を $(git -C "$W" rev-parse --short HEAD) で作った．E77 の作業木の HEAD は $(git -C "$SRC" rev-parse --short HEAD) のまま"
cd "$W"
E82H_DRY=1 bash run/e82h.sh 1
E82H_DRY=1 bash run/e82h.sh 2
