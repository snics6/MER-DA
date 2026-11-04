判定：通す（pseudo の起動に進んでよい．直してから進む・止めるに当たる指摘は無い．記録の指摘 5 件）
理由：前回の 8 点は申告どおりに直っており，評価の段だけで動く新しい経路（引き戻しの 14 点）を 1 単位で実際に流して実験 1 の p3t と対にでき，守りも偽の置き場で止まる・通ることを確かめた．
仮のターゲットの成績の値はこの報告に書かない．

# 関門 1（差分の再検査）：実装と出力（DAI-4，2026-11-05，コミット de19f28．前回 b018ae0）

検査の方法：`git diff b018ae0 HEAD -- DAI/exp/dai4.py DAI/exp/dai4_stats.py`（dai4.py +38/−，dai4_stats.py +60/−，ほかの DAI/exp の変更は探索用の別ファイル）を読み，下の 8 項目を流すか読むかして確かめた．出力は `DAI/checks/DAI-4/code_run_v2/` の下．`results/` には書いていない．GPU は MERDA_GPU_FRACTION=0.2，別の計算が動いている中で流した（nvidia-smi で使用 3.4 GB・利用率 26〜90%）．

## A．前回の指摘 8 点と差分の対応

| 前回の指摘（申告の番号） | 台本の箇所 | 合っているか |
|---|---|---|
| (1) R6・R7：cons の選んだ点を 14 点で解く | dai4.py 195〜199 行（`tether_sel`，`p4t_{s}_K{K}`，method = cons_tether），run_unit の grid に `tether|s|K|(ρ, γ)` を追加，main 571・578 行の `tsel`，dai4_stats.py 103〜113 行（`tc|`・`tj|` のセル），462〜470 行の desc（read = tether_curve，A = cons_tether，B = joint_tether） | 合う．下の「小さく流した結果」の 1 |
| (2) S3・R：stats の守り | dai4_stats.py `guard_eval` 258〜266 行 | 合う．下の 2 |
| (3) S3：joint 側の並びの照合と sha256 | dai4_stats.py `pairing` 280〜297 行，evaluate 556〜559 行 | 合う．下の 3．sha256 は読んだだけ |
| (4) R：pseudo_commit と pseudo_unit_sha256 | dai4.py `check_eval_gate4` 492〜501 行 | 合う．下の 4 |
| (5) I1：selftest の合否 | dai4.py 243〜245 行 | 合格側を流して通った．止まる側は読んだだけ（記録 3） |
| (6) I9：pseudo の計算機名 | dai4.py `select_stage`（gpus の集め，1 種でなければ停止，`_meta.pseudo_gpu`） | 合う．下の 5 |
| (7) S：正解の感情に置かれた重みの割合から K 本を除く | dai4_stats.py 515〜519 行 | 合う．下の 6 |
| (8) docstring の問の数 | dai4.py 44 行（1,140，評価は 1,420 = 1,140 + 280） | 合う．台本に「1,200」の記述は残っていない |

設計の文書は 4 節の守り 6・8・9・10（学習の経路の文，pseudo 後の照合，stats の守り，対の照合と sha256），113 行（引き戻しの感度 280 問），25 行（W の差 1e-4・b の差 1e-3），125 行（1,140 問 + 280 問）が台本と一致する．前回の記録 I3（設計の文）と I8（未追跡ファイル）も直っている（下の 7）．

## A．小さく流した結果

### 1．評価の段の引き戻しの 14 点（実験 1 の p3t と対にできるか）
- 実行：`tools/run_eval_unit.py AFEW 0 0`．評価の段と同じ `dai4.run_unit(tag="eval", tether_sel=…)` を，実験 1 の評価の回の保存（`results/dai1/eval/units/AFEW_ALL_f0_r0_p1・p2・p3`）を読んで 1 単位だけ流した．固定のファイルは偽（選んだ点を (生徒, K) ごとに変えたもの）．出力は `out_e/`．所要 396 秒（別の 2 つの計算と同時，pool 632）．収束 240/240（cons）．例外・警告なし（ログ `out_e.log`）．
- 行：cons_tether 280 行（20 の (生徒, K) × 14），収束 280/280，勾配の最大 1.0e-6，NaN 0，finite が偽の問 0．`_weights` の variant = cons_tether_base が 20 行（(生徒, K) ごとに 1 行）で，stats の `zero`・`wc` の集計は variant で絞るので混ざらない．
- 保存：`p4t_{s}_K{K}_pred/_W/_b/_items` が 80 キー．p4t の問の名前（method・rho・gamma・Cw・Cb）の ρ・γ は固定のファイルで選んだ点と全部一致．14 点の並びは 20 の (生徒, K) すべてで `dai1.TETHER` の順．実験 1 の p3t の prop_sel の 14 点も同じ順で，不一致 0/20．
- 対にできる確認（`tools/verify_v2.py`）：既定の引き戻し（fz1 の tether）の点で，p4t の判定は p4 の cons の判定と，p3t の prop_sel の判定は p3 の joint の判定と，20 の (生徒, K) すべてで 100% 一致（min 1.0）．引き戻しの値の割り当てが問の順とずれていれば一致しないので，14 点の値の割り当てが正しいことの確認になる．
- stats の経路：`collect(extras=True, check_rows=True)` を eval の保存に当てて，tc セル 280 と tj セル 280（20 の (生徒, K) × 14）が作れる．cons の行との不一致 0/20，塊と件数の不一致 0．`effect_stats` に tc と tj を渡して動く（1 単位なので折り畳みを 1 つ落とすときに「Mean of empty slice」の警告が出る．本番は 5 折で出ない）．
- K = 0 でも p3t・p4t とも 14 点そろう．
- 評価の段の main 経路（`tsel` の lambda）は固定のファイルが無いので流さず，読んだだけ．キー `"{s}|{K}"` は `select_stage` が書く cons のキー（接尾辞なし）と一致する．

### 2．stats の守り（`guard_eval`）
`tools/try_guard_eval.py`．scratch の git clone に評価のコミット E を作り，偽の評価の保存（コミット = E）で流した．

| 場合 | 結果 |
|---|---|
| HEAD = E | 通る |
| E の後に文書と `dai4_stats.py` を変えたコミットを積んだ（HEAD ≠ E） | 通る |
| `dai4_stats.py` に未コミットの変更 | 通る |
| `dai4.py` に未コミットの変更 | 止まる |
| `dai4.py` をコミットで変更 | 止まる |
| `dai1_frozen.json` をコミットで変更 | 止まる |
| 単位のコミットが HEAD の別の枝（祖先でない） | 止まる |
| 単位のコミットが 2 つ | 止まる |

評価の後に結果と台帳をコミットして push しても，stats を流し直せる．stats の台本に解析を足しても止まらない．評価の単位のコミットが短縮形でないことは `dai1.git_commit()`（`rev-parse HEAD`）で確認した．

### 3．pairing の joint 側の照合
`tools/try_pairing.py`．本物の入力（out_e と実験 1 の保存）で `joint_uar_mismatch` = 0（一致）．p3 の AVT・K = 5 の判定の列だけテストの並びを入れ替えた p3 の保存（`in_perm/`）で `joint_uar_mismatch` = 1 になり，`SystemExit` で止まる．sha256 の照合（evaluate 556〜559 行）は `eval_gate_dai4.json` の `input_sha256` と stats の 600 ファイル（200 単位 × p1・p2・p3．キーは `{T}_ALL_f{f}_r{r}_p{1,2,3}.npz`）が同じ形で作られることを読んで確かめた（流していない）．

### 4．check_eval_gate4 の pseudo の照合
`tools/try_gate2.py`（前回の `try_gate.py` に固定のファイルと pseudo の置き場の偽を足した）．

| 場合 | 結果 |
|---|---|
| pseudo_commit = HEAD，pseudo_unit_sha256 一致 | 通る |
| pseudo_commit = b018ae0（`dai4.py` に差） | 止まる（「pseudo_commit と HEAD の間で … に差がある」） |
| pseudo の保存の sha256 が違う | 止まる |
| pseudo の保存が無い | 止まる |
| 固定のファイルに T が無い | KeyError（文は無い．記録 4） |

### 5．select の計算機名
`tools/try_select_gpu.py`．4 T × 30 単位の偽の置き場で，全単位 1 種なら通り `_meta.pseudo_gpu` が書かれる．1 単位だけ別の計算機にすると「pseudo の全単位の計算機名が 1 種でない」で止まる．

### 6．K 本を除いた重みの割合
`tools/try_wfrac.py`．AFEW の評価の 1 単位の K = 0・5・25 で，新しい `wfrac` を numpy で作り直した値と比べて一致（差 6 桁まで一致）．K = 0 では旧版と同じで，K = 5・25 では旧版（全 pool）と違う値になる（K 本を除いたため）．

### 7．前回の記録の扱い
- I3：設計 4 節の守り 6 が「学習の経路（G と fit）がラベルを読まない．行の uar は fit の後にテストのラベルで計算する．候補のラベルは stats の後処理でだけ読む」に直っている．台本の動作と合う．
- I8：`git status` で DAI/exp・DAI/splits・measure・features・data/metadata に未追跡・未コミットのファイルは無い（`dai1.git_commit()` の dirty の対象）．`explore_fusion_grid.py` などはコミット済み．今の作業木で pseudo は起動できる．

### 8．pseudo の経路（`--selftest-g1` 付き）
`MERDA_GPU_FRACTION=0.2 ./.venv/bin/python DAI/exp/dai4.py --stage pseudo --targets AFEW --cs MELD --folds 1 --seeds 0 --allow-dirty --selftest-g1 --out-dir …/out_p`．AFEW_MELD_f1_r0 を 416 秒で完了（1,520 問．別の 2 つの計算と同時）．収束 320/320（cons），例外・警告なし．`_g1check` 80 点で判定の一致の最小 1.0，W の差 1.8e-5 以下，b の差 8.0e-5 以下（設計の 1e-4・1e-3 の内側）．行のコミットは de19f28，dirty = False．cons・対照の行に NaN 0．pseudo では `tether_sel = None` で，格子の名前の末尾に `tether|…` が付かない（評価の回だけ）．

## A．本番の規模の見積もり
- 評価の回の増分は 280 問/単位（1,140 問の 25%）．前回の見積もり（評価 200 単位，A5000 換算 7〜12 時間）に +25% で 9〜15 時間．今回の 396 秒は別の 2 つの計算と同時で，単独の時間ではない．
- ディスク：評価の 1 単位（pool 632）の npz は 23.5 MB で，p4t の 80 キーを含む．320 単位で約 6〜8 GB（pool の大きい IEMOCAP・MELD・MOSEI ほど大きい．空きは前回 548 GB）．
- stats：desc に 14 点 × 20 セル × 4 T = 1,120 回の `effect_stats` が加わる．前回の約 2,400 回に足すと約 3,500 回で，0.6〜1.6 秒として約 1.5 時間．
- GPU メモリ：0.2 × 24 GB の設定で 2 本の流しが並行して例外なし．評価ごとに空きブロックを返す処理は前回と同じ．

## B．予想される指摘と，出力から答えられるか（今回の差分の分）

| 表・図 | 予想される指摘 | もとにした項目か例 | (a)(b)(c) | (b) なら足すもの | 足す費用 |
|---|---|---|---|---|---|
| desc の引き戻しの曲線 | 引き戻しの強さを替えると cons − joint の結論は変わるか | tac-round1-status の「1 通りでしか計算していない」，R | (a) desc の read = tether_curve（14 点）から出る．none の 14 点は実験 1 の p1，partner の 14 点は p3t の partner_sel から作れる | — | — |
| 同上 | 引き戻しを替えたときに cons の ρ・γ を選び直したら | R | (c) 選び直さない．実験 1 の joint の p3t も同じ扱い（`dai1.pass3` の注記「選び直さない」）．設計 113 行に「選んだ点」と書いてあるので主張を限れる | 設計 8 節に 1 文（範囲の外と理由） | — |
| 全体 | 性別，話者を絞る，部品を外す，同じ教師の相手，第 2 エンコーダ | 前回の B の表 | 前回の判定のまま（(a) と (c)） | — | — |

前回の B の (b) は引き戻しの感度の 1 行だけで，今回の差分で出力に入った．ほかに (b) は無い．

## 指摘

### [S3・記録 1] pairing は p3t の並びを確かめない
- 重さ：記録する
- 該当箇所：dai4_stats.py 280〜297 行
- 問題：`joint_uar_mismatch` は p3 の joint の判定の並びだけを確かめる．p3t の prop_sel の判定は同じ `Unit` の同じテスト順で作られるが，stats の照合の対象外．今回は 1 単位で p3t の既定の引き戻しの点と p3 の joint の判定が 100% 一致することを確かめた（20/20）．
- 計算の後で直せるか：直せる（stats の台本は評価の後に変えてよい）
- 求める対応：pairing に「p3t の既定の引き戻しの点の判定 = p3 の joint の判定」の一致の数を足す（評価の後でよい）．

### [S3・記録 2] evaluate の sha256 の照合が集計の最後にあり，設計 4 節の守り 7 の文が p1・p3 のまま
- 重さ：記録する
- 該当箇所：dai4_stats.py 556〜559 行，設計 4 節 守り 7（「p1・p3 の保存の sha256」）
- 問題：照合は約 1.5 時間の集計の後に置かれており，食い違うと全部を流した後に止まる．`check_eval_gate4` が評価の直前に同じ照合をするので誤った数値は出ないが，集計の時間を無駄にする．台本は p1・p2・p3 を書くが，設計の守り 7 は p1・p3 と書く（守り 10 は p1・p2・p3）．
- 計算の後で直せるか：直せる（stats は評価の後でも変えられる）
- 求める対応：照合を `evaluate` の冒頭に移す．守り 7 の文に p2 を足す．

### [I1・記録 3] selftest の合否が点ごとの最小で，テストが少ない単位では 1 件の食い違いで止まる
- 重さ：記録する
- 該当箇所：dai4.py 243〜245 行
- 問題：`ag = min(pred_agree)` が 0.999 を下回ると止まる．テストの発話が 158 本の単位では，1 点で 1 件の判定が違うだけで 0.9937 になり止まる．前回の GPU の揺れは 180,120 点中 1 点なので，起きる確率は低い．止まる側は流していない（読んだだけ）．selftest は試走だけで本番に影響しない．
- 計算の後で直せるか：直せる
- 求める対応：設計 9 節の「99.9% 以上」が点ごとの最小か全点の合算かを 1 文で決める．本番には必要ない．

### [記録 4] `check_eval_gate4` で固定のファイルに T が無いとき KeyError で，文が出ない
- 重さ：記録する
- 該当箇所：dai4.py 492〜494 行（`fzj[T]["pseudo_commit"]`）
- 問題：select の出力に T が欠けると KeyError で止まる（止まるので誤った数値は出ない）．
- 求める対応：なし．任意で `fzj.get(T)` の検査文を足す（ただし dai4.py を変えると pseudo の後は変えられないので，pseudo の前に足すなら今．足さなくてよい）．

### [R・記録 5] 評価の回だけの経路は pseudo で動かないため，pseudo の後に不備が見つかると pseudo をやり直すことになる
- 重さ：記録する（今回の検査で 1 単位を流して確認済み．申し送り）
- 該当箇所：dai4.py の評価の段だけで動く経路（`tether_sel`，`check_eval_gate4`）と，守り 8（pseudo_commit と HEAD の間で dai4.py・dai1.py・dai_solver.py・dai_data.py に差があれば評価の段が止まる）
- 問題：評価の段だけで動く経路は pseudo の run では動かない．pseudo の後に dai4.py を直すと守り 8 が評価の段を止めるので，pseudo のやり直し（約 5〜7 時間）になる．今回の検査で p4t の経路を 1 単位で動かして確認した．残る未確認は，評価の段の main の分岐（`sel_all`，`tsel` の lambda．読んだだけ）と，実物の固定のファイルでの `check_eval_gate4` の通過．
- 求める対応：pseudo の起動の前に，dai4.py をこれ以上変えないと決める．変える場合は再検査してから起動する．

## 目録に無い種類の不備
- 提案：`docs/methods.md` 9.3 節 R に「評価の段だけで動く経路は，pseudo の前に評価の保存を読む形の小さい流しで動かしておく．pseudo の後に台本を変えると守りが評価の段を止めるため」を足す．
- 提案：9.3 節 S に「別の保存と対にする条件を検査する台本は，偽の置き場で『通る』と『止まる』の両方を流して確かめる」を足す（今回の guard_eval・pairing・check_eval_gate4・select の計算機名で実施）．

## 計算の後の点検（関門 2 の前）への申し送り
- 欠けと重複：eval は 4 T × 50 単位 × 1,420 問（本番 240 + cons_self 180 + own・conf・rand 各 240 + cons_tether 280．識別子は (method, student, K, rho, gamma, Cw, Cb)）．単位の行数の見込みは `out_e/eval/units/AFEW_ALL_f0_r0_p4.csv` の method の内訳（cons 240，cons_self 180，cons_own・cons_conf・cons_rand 各 240，cons_tether 280，`_weights` 1,160，`_conv` 115，`_marks` 20，`_marks_ctrl` 75）と同じはず．
- 書き出されているべき出力：`p4t_*` の 80 キー，`p4_K*_Q`，`ret_*_am`，`cluster_test`，stats の pairing.csv（`joint_uar_mismatch` 列を含む），desc.csv（read = tether_curve の行：4 T × 4 生徒 × 5 K × 14 = 1,120 行），teacher_check.csv，sensitivity.csv，marks_variants.csv，eval_input_sha256.json．
- 結果のコミット番号が de19f28 以降の dai4.py と同じ内容か（`git diff de19f28 <結果のコミット> -- DAI/exp/dai4.py DAI/exp/dai1.py DAI/exp/dai_solver.py DAI/exp/dai_data.py` が空）．
- ログに例外・収束しない警告・メモリ不足からのやり直しが無いか．

ファイル（すべて `DAI/checks/DAI-4/code_run_v2/` の下）：`tools/run_eval_unit.py`（評価の段の solve を 1 単位），`tools/verify_v2.py`（p4t の確認と collect），`tools/try_guard_eval.py`（stats の守り），`tools/try_pairing.py`，`tools/try_gate2.py`，`tools/try_select_gpu.py`，`tools/try_wfrac.py`，出力 `out_p/`・`out_e/`・`out_p.log`・`out_e.log`・`gate_test/`・`fake_sel/`・`in_perm/`．scratch の git clone は `/tmp/claude-1000/-home-snics-MER-DA/b1f27254-32c7-45f0-ac5a-dee1c5e3ec8a/scratchpad/clone`．
