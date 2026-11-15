# 整理の後の台本の再現の確かめ：論文の後半（実験 1〜3，11〜18，20，21）（2026-11-15）

判定：直す

整理の後の台本は，流した全部の単位で動き，保存してある出力と同じ値を出した．研究室の PC で計算した実験（3 の validate，11〜18，20，21）は，比べた数の全部が完全に一致した．RunPod の H200 で計算した実験（1・2 と 3 の評価の段）では，GPU とライブラリの版の違いによる小さいずれが出た．ただし，実験 2 の xMUDA 型の学習率 0.01 の点だけは，判定が最大 28.2 ポイントずれた．

「直す」にした理由は，評価の段の守りにある．実験 1 の守り（`frozen/eval_gate.json` と `dai1.check_eval_gate`）は写しで通る．実験 2 と実験 3 の評価の段は，整理で台本の sha256 が変わったために，台本自身の照合で止まる．このことは `docs/再現の手順.md` に書かれていない．あわせて，実験 21 の固定のファイルの照合が 1 本ぶん黙って省かれる．値が誤っている箇所は見つからなかった．

## 条件

- 写し：`…/scratchpad/standalone/DAI`．コミットは `8e703b4`（`405b810` の書き出し）．流している間，作業木は汚れていない（`git status --porcelain` は 0 行）．
- 研究室の PC：RTX A5000．torch 2.11.0+cu126，numpy 2.5.2，`MERDA_GPU_FRACTION=0.6`．GPU の計算は 1 本ずつ流した．
- 保存してある出力：`/home/snics/MER-DA/results/` を読むだけにした．入力は写しの `results/` に**写して**置いた（リンクは使っていない）．
- 保存の GPU（各置き場の `provenance_log*.jsonl` から読んだ）：
  - 実験 1 の source・pseudo・eval と実験 2 の pseudo，実験 3 の評価の段：NVIDIA H200，torch 2.8.0+cu128
  - 実験 1 の select，実験 3 の validate・detect，実験 11〜21：NVIDIA RTX A5000，torch 2.11.0+cu126
- 流した単位：主に `AFEW_IEMOCAP_f0_r0`．評価の回は `AFEW_ALL_f0_r0`．実験 18・21 は CREMA-D の f0・r0 の単位．
- 一致の物差し：
  - 「最大差」は，同じ鍵の行（csv）または同じ名前の配列（npz）の要素ごとの絶対差の最大である．
  - 「不一致」は差が 0 でない要素の数である．
  - 判定の配列（int）は，一致しない要素の数を全体の要素の数と並べて書く．
  - 記録の列（commit，sec，gpu，host，gpu_max_mb）は比べていない．
- 検算の台本とログ：`checks/整理_2026-11-15/code_run_repro_late/`
  - 比べる道具：`cmp.py`，`cmp_rows.py`，`cmp_tree.py`
  - 1 単位だけを流す呼び出し：`drv_*.py`
  - 流した順：`queue1〜5.sh`
  - ログ：`logs/`，`selftest_logs/`

`drv_*.py` は，台本の main が 1 単位だけの実行を受け付けない段で使った．台本の中の関数（`run_pass`，`stage_validate`，`eval_unit`，`run_unit`，`stage_units`，`loop_units`）をそのまま呼び，計算は変えていない．飛ばした守りは次の 3 つである．

- 選ぶ段の json のコミットの照合（実験 11 の units）
- 全単位で選び直す処理を，保存の `frozen_sel.json` の選択に替えたこと（実験 17・18 の生徒の段）
- 本番の置き場だけに書き，絞らないという守り（実験 20 の評価の回）

## 1．selftest

12 本のすべてが通った（ログは `selftest_logs/`）．

| 台本 | 結果 | 所要 |
|---|---|---|
| `dai_solver.py --selftest` | 通った | 4 秒 |
| `dai1.py --stage selftest` | 通った | 6 秒 |
| `dai11_probe.py` | selftest ok | 3 秒 |
| `dai12_probe.py` | selftest ok | 14 秒 |
| `dai13_probe.py` | 自己検査 OK | 14 秒 |
| `dai14.py` | 12 項目すべて通った | 4 秒 |
| `dai15.py` | 7 項目すべて通った | 3 秒 |
| `dai16_probe.py` | 通った（版 204，差の定義 288） | 3 秒 |
| `dai17_probe.py` | 通った（版 145，差の定義 195，格子 17） | 4 秒 |
| `dai18_probe.py` | 通った | 4 秒 |
| `dai20_eval.py` | 通った | 4 秒 |
| `dai21_eval5.py` | 通った | 4 秒 |

`dai2.py`，`dai2_grad.py`，`dai3.py` には selftest の段が無い．

## 2．実験ごとの再現

### 研究室の PC と違う GPU で保存した実験（H200）

| 実験 | 流したもの | 比べたもの | 結果（最大差，不一致） | 所要 |
|---|---|---|---|---|
| 1 source | `dai1.py --stage source --targets AFEW IEMOCAP`：14 の識別器 | `dai1/source/{AFEW,IEMOCAP}_*.npz`，`summary.csv` | 選んだ C は 14 本とも同じ．選んだ C の交差検証の UAR は差 0．交差検証の 840 マスのうち 8 マスがずれ，最大 0.19 ポイント（すべて C ≥ 0.1 の弱い正則化）．選んだ W は最大 4.3e-6，b は最大 2.4e-4．全部の C の W・b（`W_allC`・`b_allC`）は最大 0.037 | 約 4 分 |
| 1 pseudo 段 1 | `run_pass` で `AFEW_IEMOCAP_f0_r0` | `dai1/pseudo/units/…_p1.{npz,csv}` | 735 行のうち UAR は 2 行が 0.087 ポイントずれた（どちらも oracle，C′ = 1.0）．判定は 532,168 要素のうち 2 が不一致 | 63 秒（読み込みを除く） |
| 1 pseudo 段 2 | 同じ単位．引き戻しは保存の `AFEW_tether_w.json` | `…_p2.*` | 385 行で UAR と uar_pool の差は 0．npz の最大差は 1.2e-4（確率），1e-6 を超える要素は 2 | 25 秒 |
| 1 pseudo 段 3 | 同じ単位．引き戻しと指数は保存の値 | `…_p3.*` | 3,686 行のうち UAR は 3,195 行にあり，25 行がずれた．最大 0.45 ポイント．内訳は coreg 15，joint 2，joint_scratch 3，kd 2，rolesep 3．agree・mix・cotrain・tt は差 0．判定は 2,382,329 要素のうち 69（0.003%）が不一致 | 522 秒 |
| 2 線形（dai2.py） | `--pseudo-c IEMOCAP --students A`，S+T の交差検証（`st_cv`）も計算し直した | `dai2/pseudo/units/…_lin.*` の生徒 A の 576 行，`st_cv/MELD+MOSEI_A.json` | UAR は 576 行で差 0．判定は 329,681 要素で差 0．n_iter は 39 行で違い，最大 49 回．W_dist は最大 2.0e-5．st_cv は選んだ C が同じで，交差検証の 60 マスのうち 4 マスが最大 0.0085 ポイントずれた | 118 秒 |
| 2 勾配法（dai2_grad.py） | `--parts xmuda coral` | `…_xmuda.*`，`…_coral.*` | CORAL：92 行で UAR は差 0．npz の最大差は 1.1e-5．<br>xMUDA 型：学習率 0.001 の 195 行は UAR が差 0．学習率 0.01 の 180 行のうち 114 行がずれた．反復 200 では最大 5.7 ポイント，反復 1000 では 90 行すべてがずれ，最大 28.2 ポイント，組ごとの平均は 2.1〜9.8 ポイント．kl_last の差は最大 43．判定は 274,125 要素のうち 25,997（9.5%）が不一致 | 146 秒 |
| 3 評価の段 | `drv_dai3_eval.py` で `eval_unit` を `AFEW_f0_r0` に当てた．固定のファイルは保存の `dai3/frozen.json` | `dai3/eval/units/AFEW_f0_r0.pkl` | 数 86,735 個のうち 290 個がずれ，最大 3.1e-8．数でない 2,735 個はすべて一致 | 20 秒 |

xMUDA 型の欄の補足：`dis_ctl` は写しの出力ではすべて NaN になった．map の部分を流していないためで，台本の仕様どおりである（`dai2_grad.py:60` に「無ければ NaN」とある）．比べる数からは外した．

### 研究室の PC で保存した実験（A5000）

| 実験 | 流したもの | 比べたもの | 結果（最大差，不一致） | 所要 |
|---|---|---|---|---|
| 3 validate | `drv_dai3_validate.py`：`stage_validate` を 1 単位に絞った | `dai3/validate/check1_units.csv`（168 行），`check2_units.csv`（12 行） | UAR の列は差 0．diff は 4.4e-16．grad_max は 5.3e-23 | 8 秒 |
| 11 段 A | `--stage enc2k --units AFEW_IEMOCAP_f0_r0` | `rowsA/…_enc2k.csv`（228 行） | 差 0 | 18 秒 |
| 11 段 B | `drv_dai11_units.py`．選ぶ段の json は保存のもの | `rows/….npz`（1,637,044 要素），`…_req.csv`（3,780） | 差 0 | 16 秒 |
| 12 | `--stage units --units AFEW_IEMOCAP_f0_r0` | `rows/….npz`（41,396,330 要素），`_lab.csv`，`_nl.csv` | 差 0 | 165 秒 |
| 13 | 同上 | `rows/….npz`（7,890,665 要素），`_lab`・`_layers`・`_lw.csv` | 数は差 0．文字の違いは commit と script_sha256 の 2 つだけで，今の台本の値が入った | 101 秒 |
| 14 | `--stage units --targets AFEW --folds 0 --seeds 0` | `units/AFEW_f0_r0.{npz,csv}` | 差 0．台本の中の照合（実験 3 の repro_units.csv と比べる）も 18 行で最大 7.1e-15 | 45 秒 |
| 15 | pseudo：AFEW の f0・r0 の 3 単位．eval：`AFEW_AFEW_f0_r0` | `pseudo/units/*`，`eval/units/*` | 4 単位とも npz と csv の差は 0 | 100 秒＋25 秒 |
| 16 | `--stage units`：AFEW の f0・r0 の 3 単位 | `units/*` | 差 0 | 292 秒 |
| 17 units | 同じ 3 単位 | `units/*`，`srcdb/*`（12 本） | 差 0 | 183 秒 |
| 17 reweight | `--folds 0 --seeds 0`：3 組 | `reweight/*` の種 0 の行（380 行），npz の共通の鍵 | uar と hub_mean は 1 要素で 3.6e-15 の差．ほかは差 0．保存だけにある鍵 392 は種 1 の鍵である | 119 秒 |
| 17 students | `drv_dai17_students.py`．格子の点は保存の `frozen_sel.json` | `students/*`（3 単位） | 差 0．提案との一致は 1.0000 | 165 秒 |
| 18 source | `--stage source --out-dir <新しい置き場>` | `source/*`（4 本と summary），`splits_new.csv` | 差 0 | 84 秒 |
| 18 base・units | `--targets CREMA-D --folds 0 --seeds 0`：5 単位ずつ | `base/*`，`units/*` | 差 0．違うのは記録の列の gpu_max_mb だけ | 35 秒＋164 秒 |
| 18 students | `drv_dai18_students.py`．格子の点は保存の `frozen_sel.json` | `students/*`（5 単位） | 差 0 | 287 秒 |
| 20 | `drv_dai20_eval.py`：評価の回の `AFEW_ALL_f0_r0`．入力の実験 1 の評価の単位は保存の写し | `dai20_eval/units/AFEW_ALL_f0_r0.*` | npz（1,007,012 要素）と csv（91,560）の差は 0 | 55 秒 |
| 21 check3 | `--stage check3` | 台本の中の照合（実験 16 の保存の 2 単位） | 4 行とも agree_min 1.0，38 版のすべてが一致．「check3：通った」 | 9 秒 |
| 21 check18 | `--stage check18` | 台本の中の照合（実験 18 の保存） | K = 5・10 は 39 版のすべてが一致．K = 25 は agree_min 0.999739（M_F6_mix5）．保存のログ `logs/dai21_check18.log` の表と数がすべて同じ．「check18：通った」 | 68 秒 |
| 21 students4 | `--targets AFEW --folds 0 --seeds 0` | `students4/AFEW_ALL_f0_r0.*` | 差 0 | 10 秒 |
| 21 cremad | `--folds 0 --seeds 0` | `cremad/CREMA-D_ALL_f0_r0.*` | 差 0 | 72 秒 |

GPU で計算した時間の合計は約 70 分である．流さなかった段は次のとおりである．

- 実験 2 の map（MME・DANN）：保存では 1 単位あたり約 1,400 秒かかった（H200 で 10 本を並べたとき）．
- 実験 2 の線形の生徒 V・T・AVT：生徒 A の 1 本だけを流した．
- 実験 3 の detect：4 つのターゲットの仮のターゲットの回の全部（7.7 GB）を写す必要がある．
- 各実験の集計の段と，実験 1 の評価の回の全部．

## 3．評価の段の守り

- **実験 1（通る）**
  - 写しでは，`dai1.check_eval_gate` が `dai1_frozen.json`・`dai2_frozen.json`・`dai3_frozen.json` の 3 つで通った．eval_gate.json の sha256 は `92378fa0…` である．
  - `results/dai1/select/frozen.json` と `frozen/dai1_frozen.json` の sha256 は一致した（`6926fb3e…`）．`frozen/dai1_frozen.json` はコミット済みである．
  - 作業木の側の HEAD（`89ef337`）でも，読むだけで流すと 3 つとも通った（eval_gate.json は `a6ba9623…`）．
- **実験 2（止まる）**
  - `dai2.check_eval_frozen` は「台本 dai2.py が選ぶ段のときから変わっている（m72）」で止まる．
  - `frozen/dai2_frozen.json` の `_meta.script_sha256` と今の台本を比べた．一致したのは dai_solver.py だけで，dai2.py・dai2_grad.py・dai2_select.py・dai1.py・dai_data.py は一致しない．
  - 照合するのは自分の 3 本（dai2.py，dai2_grad.py，dai2_select.py）だけなので，この 3 本で止まる．
- **実験 3（止まる）**
  - `stage_eval` は，固定のファイルの `_meta.script_sha256` と `script_shas()` が一致しないと止まる（`exp/dai3.py:1877-1878`）．
  - 一致しないのは dai3.py，dai1_stats.py，dai1.py である（dai_solver.py は一致）．
  - 2 節の実験 3 の評価の段の値は，この照合を通さずに `eval_unit` を直接呼んで出した．

## 4．結果の記録（provenance）

台本の main を通して流した置き場では，`provenance_log.jsonl` の最後の行に，写しのコミット `8e703b4f…`，`dirty: false`，今の台本の sha256 が書かれた．対象は dai1_srccheck，dai2_try，dai2_lin_try，dai11_try，dai12_try，dai13_try，dai14_try，dai15_try，dai16_try，dai17_try，dai18_src，dai18_try，dai21_try である．

台本の sha256 は `sha256sum exp/…` と照らした．照らした例は次のとおりである．

- dai1.py：`f98cb51c…`
- dai13_probe.py：`dfbe4967…`
- dai12_probe.py：`8450951f…`
- dai21_eval5.py：`cb7a5bb3…`

実験 13 の単位の npz の中の `script_sha256` も今の値になった．

`drv_*.py` で関数を直接呼んだ段は，main を通らないので provenance を書かない．ただし単位の csv と npz の commit の列には写しのコミットが入った．実験 21 の check18 は，台本の仕様で provenance を書かずに戻る．

## 5．直すべき点

1. **`exp/dai2.py:181-183`（`check_eval_frozen`）と `docs/再現の手順.md` の 4 節**
   - 何がおかしいか：整理で dai2.py・dai2_grad.py・dai2_select.py の中身（置き場の行）が変わった．このため，実験 2 の評価の段は m72 の照合で必ず止まる．`run/dai_eval.sh 2 <T>` は流せない．手順の 4 節は eval_gate.json を作り直したことだけを書いていて，この停止を書いていない．原因は整理である．
   - どう直すか：`frozen/eval_gate.json` の `_meta` と同じ形の対応表を 1 本足す．例は `frozen/reorg_sha_map.json`（コミット済み）で，各台本の整理前の sha256 と整理後の sha256 を並べ，整理で置き場の行だけを変えたことを書く．`check_eval_frozen` は，固定のファイルの sha256 が対応表の整理前の値で，今の台本が整理後の値のときに通す．固定のファイルは書き換えない．手順の 4 節にもこの対応表を書く．

2. **`exp/dai3.py:1877-1878`（`stage_eval`）と同じ 4 節**
   - 何がおかしいか：固定のファイル `frozen/dai3_frozen.json`（= `results/dai3/frozen.json`）の `script_sha256` と今の dai3.py・dai1_stats.py・dai1.py が一致しない．このため，実験 3 の評価の段は「detect から流し直す」で止まる．detect を流し直すと固定のファイルが変わる．原因は整理である．
   - どう直すか：1 と同じ対応表を `stage_eval` の照合にも使う．手順の 4 節に，実験 3 の評価の段もこの照合を通ることを書く．

3. **`exp/dai21_eval5.py:332-334`（`load_frozen`）と `frozen/dai21_frozen.json:226`**
   - 何がおかしいか：固定のファイルの `source_sha256` に整理の前の鍵 `"DAI/frozen/dai20_frozen.json"` が残っている．`load_frozen` は `if (ROOT / p).exists() and …` で，見つからないファイルの照合を黙って飛ばす．このため，整理の後は実験 20 の固定のファイルの照合が行われない．今の `frozen/dai20_frozen.json` の sha256 は記録の値 `faa2391d…` と同じなので，値への影響は無い．原因は整理である．
   - どう直すか：`load_frozen` で鍵の頭の `DAI/` を外してから読む．そのうえで，見つからないファイルは止まるようにする（固定のファイルは書き換えない）．

4. **`docs/再現の手順.md` の 7 節（記録の追加）**
   - 何がおかしいか：H200 で保存した単位を研究室の PC で流すと，ずれは小さいものから大きいものまである．
     - 実験 1 の段 3：UAR が最大 0.45 ポイント
     - 実験 2 の xMUDA 型の学習率 0.01：最大 28.2 ポイント，判定の 9.5% が不一致
     - 保存のときの torch は 2.8.0+cu128 で，今の `.venv` は 2.11.0+cu126 である
     7 節は「GPU の種類が違うと値が小さくずれうる」とだけ書いていて，大きさと版の違いを書いていない．原因は元からのものである．
   - どう直すか：7 節にこの検査の数（上の 2 節の表）と torch の版の違いを書く．あわせて，xMUDA 型の学習率 0.01 の点は計算機を替えると大きく動くことを書く．

## 6．動かない台本・段

- **整理による（既知，作業木の側の `7748bc0` で直した）**：`run/dai1_pseudo.sh:7`，`run/dai2_pseudo.sh:7`，`run/dai_eval.sh:19` の `cd "$(dirname "$0")/../.."` は，DAI の外を指す．写しで `PY=python3 bash run/dai1_pseudo.sh DUMMY` を流すと，`…/standalone/exp/dai1.py` が無いと言って止まり，`logs/dai1_DUMMY.failed` を DAI の外に書いた（書かれた `logs/` は消した）．協調役の連絡のとおり，5 節には入れない．
- **整理による**：実験 2 と実験 3 の評価の段の守り（5 節の 1・2）．
- **元から**（直さなくてよい．記録だけにする）：
  - `dai16_probe.py:341` の `RuntimeWarning: Mean of empty slice`．保存のログ `/home/snics/MER-DA/logs/dai16_probe_units.log` にも 18 回出ている．実験 20 の単位でも同じ警告が出た．
  - 3 本の起動の台本は `PY=${PY:-python}` を既定にしている．手順の 1 節の「起動の台本は `.venv/bin/python` を呼ぶ」と合わない（読んだだけ）．
  - `run/dai_eval.sh:37` は `nvidia-smi` を呼ぶ．この WSL のシェルでは `nvidia-smi` が PATH に無く（`/usr/lib/wsl/lib/nvidia-smi` にある），入力の記録の GPU の行が「command not found」になる．入力の記録の誤りの行を探す grep（`No such file|cannot|error|Traceback`）はこの文を拾わないので，止まらずに GPU の行が欠ける（読んだだけ．評価の回は流していない）．
- **検査の側の手違い**：実験 12・13 の 1 回目は，私が実験 11 の `enc2/source` を写すときに置き場を誤ったので止まった．写し直して流し直し，2 節の値を出した．台本の不備ではない．
