（検査役は Sonnet のサブエージェント．報告の .md を書けなかったので，本体が記録から取り出して保存した．2026-11-05）

判定：直してから進む
理由：最新版（コミット b018ae0）の計算は，独立に作り直した教師・印・重み・判定と一致した．ただし評価の回の後に学び直しになる欠け（引き戻しの強さを替えたときの cons − joint）が 1 つ，評価の後の再集計を塞ぐ守りが 1 つ，対の照合の抜けが 1 つあり，pseudo を起動する前に直す必要がある（pseudo の後は台本を変えない規則のため）．
（このファイルは書いていない．harness の指示でファイルを書かず，全文を返答に入れた．保存先は DAI/checks/DAI-4/code.md．）

# 関門 1：実装と出力（DAI-4，2026-11-05，コミット b018ae0．設計第 3 版 c7c9010）

検査の経過：最初に e4e3f47 を検査し，途中で最新版 b018ae0 に替えた．e4e3f47 で見つけた点が b018ae0 で直ったかは「e4e3f47 の指摘の照合」に書く．仮のターゲットの成績の値はこの報告に書かない．

## A．設計第 3 版と台本 b018ae0 の対応表

| 設計の項目 | 台本の箇所 | 合っているか |
|---|---|---|
| 教師 Q：段 2 の float32 の W・b から float64 で確率の積の softmax．K = 0 も ret | dai4.py `mark_and_teacher` | 合う．numpy で独立に作り直した判定と 0 件不一致（2 単位） |
| 印：生徒以外の ret の argmax の一致．AVT は 3 つ | dai4.py `solve_unit`，`others_of` | 合う．独立に作り直して 0 件不一致 |
| 重み G = gk + soft_rows(diag(m) Q, ρn, balance=True)，K 本を含む全候補 | dai4.py `build_specs` | 合う．G の列の合計と有効な本数を式 m_i Q_ik ρn/(4 S_k) から作り直し，保存した行と最大差 1.4e-10 |
| 保存：S_k，有効な本数，印の無い発話の重みの割合，重み 0 の感情の数 | `build_specs` の `_weights` 行（S_k，colsum，neff，unmarked_wfrac，n_zero_weight_classes） | 合う．S_k と割合も作り直しと一致（最大差 6.6e-12 と 6.0e-15） |
| 感情ごとの印の割合（K 本を除く） | `_marks` 行（marked_cls，n_cls，frac_cls） | 合う．n_cls の和が n_cand に一致 |
| 教師の一致（pseudo は行，eval は stats） | `_teacher_check`（pseudo のみ），stats の teacher_check.csv | 合う．実験 1 の `_teacher_stats` と 5/5 一致（2 単位とも差 7e-15 以下） |
| 対照 4 種：cons_self，cons_own，cons_conf，cons_rand（conf と rand は感情ごとに本番と同じ数） | dai4.py `solve_unit` | 合う．独立に作り直して 0 件不一致，感情ごとの印の数が本番と同じ |
| 格子：ρ 4 × γ 3，対照も全点（pseudo も eval も） | `ctrl_pts`（main），`grid_points` | 合う．1 単位の問の数は本番 240 + 対照 900 = 1,140（cons_self の AVT を省くため）．設計 6 節の「1,200 問」は 1,140 の誤り |
| 選ぶ規則：実験 1 の `pick` と同じ，対照ごとに点を選び，cons_rho25 も固定 | `pick4`，`select_stage` | 合う．`pick4` と `dai1_select.pick` を乱数の 500 組で比べて 0 件不一致（同点の規則が働いた組は 284） |
| 保存の契約：判定 int8，W・b float32，印，Q，問の名前，計算機名，コミット，sha256 | `run_unit`，`solve_unit` | 合う．固定のファイルの sha256 は pseudo では空（設計どおり），eval で入る |
| 再開の照合（コミット，固定のファイル，格子，有限） | `run_unit` 冒頭 | 合う（読んだだけ） |
| 画面に成績を出さない | 各段の print | 合う．出るのは問の数，収束，秒，食い違いの数，単位数 |
| select：eval 出力があれば止まる，単位 30，コミット 1 つ，γ = 1 の混入を止める，pseudo の sha256 を記録 | `select_stage` | 合う．偽の置き場（4 T × 30 単位）で流し，γ = 1 の行の混入で停止，29 単位で停止，eval 出力ありで停止，pseudo_unit_sha256 が 30 件入ることを確認 |
| detect：c を 1 つ除き cons と joint の両方を選び直す，joint は p3 から | `expected_effects` | 合う（下の「小さく流した結果」の条件つき）．選び直しで joint の主の版が one 以外になる場合の対処は無いが，12 通りの選び直しで全 K が one だった |
| 評価の守り 1〜7 | `check_eval_gate4`，main の detect と H200 の検査 | 1〜5 は合う（偽の固定ファイルと偽の承認で 9 通り流して確認）．6 は構造として担保（ラベルの並べ替えで確認）．7 は stats 側（指摘 3） |
| stats：族 6，Holm，desc（none，kd_one，partner，agree，cotrain，coreg，cons_g1，cons0，joint_default，joint_a1，cons_rho25，γ の曲線，取りこぼし，対照 2 通り），marks，sensitivity，pairing，teacher_check | dai4_stats.py `evaluate` | 設計の一覧はそろう．追加の抜けは指摘 3・4 |

設計にあって台本に無いもの：なし（指摘 3・4 を除く）．台本にあって設計に無いもの：`--allow-other-gpu`（設計 6 節に試走用と書くとよい）．計算しない組：設計の表の世界・生徒・ターゲット・手法の組はすべて計算される．

## A．小さく流した結果

環境：RTX A5000，MERDA_GPU_FRACTION=0.3，`./.venv/bin/python`．出力は DAI/checks/DAI-4/code_run/ の下（results/ には書いていない）．最新版は DAI/exp/dai4.py を直接呼び，`--allow-dirty --selftest-g1` で pseudo を流した（`git_commit()` は DAI/exp の未追跡ファイル `explore_fusion_grid.py` も数えるため dirty になる）．

| 単位（T，c，f，r） | 問 | 収束 | γ = 1 の検算 | 教師の照合 | 警告・例外 |
|---|---|---|---|---|---|
| AFEW，MELD，f1，r0（pool 5,878） | 1,520 | 1,520/1,520 | 80/80 点で判定の一致 100%，W の差 1.8e-5 以下，b の差 8.0e-5 以下 | 5/5 | なし |
| IEMOCAP，MOSEI，f4，r1（pool 14,908） | 1,520 | 1,520/1,520 | 80/80 点で判定の一致 100%，W の差 4.5e-5 以下，b の差 3.9e-6 以下 | 5/5 | なし |

確かめた内容（`tools/verify2.py`，保存した npz と csv を独立に作り直した）：
- 保存した判定から作り直した UAR と行の uar の最大差は 7e-15（2 単位 × 1,520 問）．同じ識別子の重複行 0．NaN・Inf 0．勾配の最大 1.0e-6．stalled 0．
- 教師，印の元の判定（`ret_*_am`），Q，印（本番・self・own・conf・rand）はすべて 0 件不一致．
- 退化の検算：γ = 1 は実験 1 の joint（a = 1，teacher = one，同じ ρ）と判定が 100% 一致．ただし W と b の差は設計 1 節の「学習器の許容 1e-6」より大きい（W は 4.5e-5，b は 9e-4 まで）．同じ設定を 2 回流した差（旧版で W 2.6e-5，b 9.2e-4）と同じ桁で，GPU の計算の揺れである．
- 学習の経路がラベルを読まない（守り 6）：最新版で，K 本の外の候補のラベルとテストのラベルを並べ替えて同じ単位を解いた（`tools/run_permuted2.py`）．印，教師，Q，ret の判定は 0 件の変化．W は 3.5e-5，b は 9.8e-5 以内で，判定は 2,398,560 点中 2,398,557 点が一致（上の揺れと同じ桁）．テストの UAR の行だけが変わった．
- 選ぶ段：4 T × 30 単位 × 115 セル（cons 20 + cons_rho25 20 + self 15 + own・conf・rand 60）．
- detect：偽の置き場で全 4 T を最後まで流した．`collect` を 1 単位の複製に差し替えたため，経路・キー・frozen への書き込み（4 T すべてに detect の出力）の確認だけで，値は検査していない．
- 評価の守り `check_eval_gate4`：偽の固定ファイルと偽の承認の報告で 9 通り流した（`tools/try_gate.py`）．固定ファイル無し，診断の表 csv が required に無い，査読が条件つき OK，承認に古い版，input_sha256 無し，入力の sha256 違い，入力 1 ファイル欠け，評価の保存がコミットより前，の 8 通りがそれぞれの文で停止し，正常は通った．評価の回の入力は現在 4 T × 50 単位 × p1・p2・p3 がローカルにそろっている（1,200 ファイル）．H200 の検査（main の 536 行目）と detect の存在の検査は読んだだけ．
- 評価の集計の経路：eval と同じ `collect(extras=True)` を pseudo の 1 単位で流し，設計 2 節の比べの相手 21 種（partner，agree，cotrain，coreg，cons_g1，joint_a1，cons_rho25，対照の自分の点と cons の点 8 種，格子 16 点，joint の格子 12 点）がすべて作れること，cons との塊と件数の不一致が 0 であることを確認した．

e4e3f47 の指摘の照合：e4e3f47 では，印の割合の分母に K 本が入る，S_k と印の無い重みの割合が保存に無い，Q が保存に無い，eval の実行が候補のラベルを読む，対照が cons の点でしか読めない，守り 1・4・5 が未実装，対の照合が先頭で止まるだけ，の点があった．b018ae0 ではこれらはすべて直っている．

## A．本番の規模の見積もり
- 1 単位の実測（1,140 問，A5000）：pool 632 で 128〜136 秒，pool 5,878 で 171 秒．pool 14,908 は 3 本並行で 471 秒（selftest で 1,520 問，単独の値ではない）．設計 6 節の 85 秒は 1,140 問では合わず，pool の大きさで 2 倍ほど変わる．
- 本番の見積もり（A5000 換算）：pseudo 120 単位は約 5〜7 時間，eval 200 単位は約 7〜12 時間（設計は約 3 時間と約 5 時間）．H200 では測っていない．
- ディスク：1 単位の npz は 25〜27 MB（selftest 込み 1,520 問）．本番は 1,140 問で約 19〜20 MB と見積もり，320 単位で約 6.3 GB．空きは 548 GB．
- GPU メモリ：上限 0.3 × 24 GB = 7.4 GB の設定で 3 本並行しても例外なし．評価ごとに空きブロックを返す処理は `solve_unit`（K ごと）と `run_unit` にある．
- stats の所要：`effect_stats` は 1 回 0.6〜1.6 秒．呼び出しは約 2,400 回で約 1 時間の見積もり．

## B．予想される指摘と，出力から答えられるか

| 表・図 | 予想される指摘 | もとにした項目か例 | (a)(b)(c) | (b) なら足すもの | 足す費用 |
|---|---|---|---|---|---|
| families6（cons − joint 12 セル） | この差の区間を話者単位で作り直せ | ML round1，R | (a) `effect_stats`（塊の jackknife か塊の引き直し）が話者単位 | — | — |
| 同上 | 引き戻しの強さ (C′, C_b) を替えたら結論は変わるか（C の別の規則に当たる） | tac-round1-status の「結果を左右する決め方を 1 通りでしか計算していない」，R | (b)．実験 1 の評価の保存には joint の 14 点の引き戻しの感度（p3t，28 問/(生徒, K)）があるが，cons 側は 1 点だけ | eval の `solve_unit` に，cons の選んだ (ρ, γ) の点を 14 の引き戻しで解く問を足す（生徒 4 × K 5 × 14 = 280 問）．あるいは 3 点に絞る（60 問）．足さないなら設計 8 節に「cons の引き戻しの感度は範囲の外」と理由を書き，結論を「実験 1 の固定の引き戻しのもと」に限る | 14 点で 1 単位 +25%（eval で A5000 換算 +2〜3 時間），3 点で +5%．ディスク +6 MB/単位 |
| 同上 | 性別ごとの UAR を出せ | 感情認識 round1 | (a) AFEW（manifest の gender）と IEMOCAP（utterance_speaker の M/F）は，保存した判定 + `sample_test` + manifest から出せる．MELD・MOSEI は性別の注釈が無く (c) | MELD・MOSEI を設計 8 節に範囲外と書く | — |
| 同上 | 話者を絞って測り直せ（1 話者除外，上位話者除外） | R | (a) 判定と `cluster_test` から（jackknife が既に 1 塊除外） | — | — |
| desc | 感情ごとの再現率と混同行列 | 感情認識 round1 | (a) int8 の判定とラベルから．stats の既定の出力には生徒の感情ごとの再現率が無く，別の事後の台本が要る | 事後の台本を書く（後述の指摘 3 の守りの扱いが要る） | — |
| desc | 同じ教師を渡した相手との差 | ML round1，R6 | (a) partner（セルごとに相手の名前と版），kd_one，agree，cotrain，coreg を比べている．agree・kd は teacher = one を受け取る | — | — |
| desc | 部品を外したらどうなるか（γ = 1，印の定義，K 本の重み） | R6 | (a) cons_g1，対照 4 種，cons_rho25 がある．感情ごとの総量をそろえる操作（balance）を外した版は無い | 設計 8 節に「balance は joint と共通で，外した版は範囲外」と書く | — |
| marks | 重みが正解の感情に置かれた割合，有効な本数，印の有無ごとの直す・壊す | 設計 2 節，定義 D2 | (a) Q（保存）と p2 の none[K] の確率から．stats が既に出す | — | — |
| marks | 教師の信頼度と正解率の関係（較正） | ML round1 | (a) Q を保存済み．`cons_conf` の根拠として出せる | — | — |
| marks | 単位ごとの直す・壊すと UAR の差の関係（単位をまたぐ相関） | R | (a) ただし stats は単位の平均しか書かない．単位ごとの値は保存（印，Q，判定）から事後の台本で作れる | 事後の台本 | — |
| sensitivity | IEMOCAP の筋書き・MELD の顔の競合・AFEW の書き起こしの疑いを除いたら | L12，R7 | (a) 台本にある（部分集合 4 種，筋書き対 3 種）．IEMOCAP は即興だけ（sample_id が impro）なので「台本を除く」は当たらない | — | — |
| 全体 | 第 2 のエンコーダ | R7，ML M1 の補足 | (c) 設計 8 節に理由つきで書いてある | — | — |
| 全体 | MOSEI の話者の代用（チャンネル）が不完全 | 感情認識 round1 | (c) 設計 8 節の限界に書いてある | — | — |
| 全体 | 反復ごとの値だけでなく，単位ごとの判定が出せるか | tac-round1-status | (a) 全点の判定を保存 | — | — |

足したときの費用の見積もりは上の表のとおり．(b) は引き戻しの感度の 1 行だけである．

## 指摘

### [R6・R7] cons の引き戻しの強さの感度が出力に無い
- 重さ：止める（足すか，設計 8 節に範囲外と書くまで pseudo を起動しない）
- 該当箇所：dai4.py `solve_unit`（`tether[(s, max(K, 1))]` の 1 点だけ，184 行付近），設計 8 節（引き戻しの感度への言及なし），実験 1 の dai1.py 510〜518 行（joint と相手には 14 点の感度がある）
- 問題：実験 1 は評価の保存に joint と相手の引き戻しの感度（p3t）を持つ．cons は固定の 1 点だけで，結論（cons − joint）が引き戻しの強さに依るかを評価の後に出せない．これは過去の査読で「結果を左右する決め方を 1 通りでしか計算していない」と指摘された型で，評価の後に求められると解き直しになる．
- 計算の後で直せるか：直せない（eval の再計算が要る．eval は 1 回だけ流す規則）
- 求める対応：(i) eval の `solve_unit` で，cons の選んだ点（固定のファイル）を `dai1.TETHER` の 14 点で解く問を足す（保存は `p4t_{s}_K{K}_pred/_W/_b/_items`．実験 1 の joint の p3t と同じ並びで対にできる）．または (ii) 設計 8 節に範囲外と理由を書き，結論を固定の引き戻しのもとに限る．(i) か (ii) を pseudo の前に決める（台本は pseudo の後に変えない規則）．

### [S3・R] 評価の後の stats 再実行が，コミットの移動で塞がる
- 重さ：直してから進む
- 該当箇所：dai4_stats.py 250〜251 行（`if commit != ucommit: sys.exit(...)`）
- 問題：stats は今のコミットが評価の単位のコミットと同じであることを求める．評価の後に結果や台帳をコミットして push すると HEAD が動き，stats を流し直せない（評価のコミットを checkout し直す必要がある）．さらに stats の台本に解析を足すと，その時点で HEAD が動くため，その台本では再集計できない．査読で解析を足したいときに止まる．
- 計算の後で直せるか：直せる（ただし台本を変えるため pseudo より前に直す）
- 求める対応：250 行を「評価の単位のコミットが HEAD の祖先で，`git diff --quiet <ucommit> HEAD -- DAI/exp/dai4.py DAI/exp/dai1.py DAI/exp/dai_solver.py DAI/exp/dai_data.py DAI/frozen/dai4_frozen.json DAI/frozen/dai1_frozen.json` が空」に替える．stats の台本の変更は許す（結果の csv に書くコミットは今のコミット）．

### [S3] 対の照合が joint 側の並びを確かめておらず，入力の sha256 を評価時の記録と照合しない
- 重さ：直してから進む
- 該当箇所：dai4_stats.py 255〜275 行（`pairing`：`ok_t`・`ok_k` は p1 の値と p4 が p1 から写した値の比べで同じ配列どうし），111〜116 行（`check_rows` は cons の行だけ），519〜526 行（stats が書く sha256 は p1・p3 だけで，p2 が無く，eval_gate_dai4.json の `input_sha256` と照合しない）
- 問題：(1) p3 の保存にはテストの `sample_id` も K 本も入っていない（確認済み）．cons − joint の対は，p3 の判定が p1 の `sample_test` の順に並ぶという前提に依る．`ok_t` と `ok_k` は p4 が p1 から写した値を p1 と比べるだけで，この前提を確かめない．塊の名前の照合（`ok_c`）だけが実質の検査である．p1 の順で解釈した joint の p3 の判定から作った UAR が p3 の行の uar と一致すること（差 0.0）を 1 単位で確かめた．テストの順を入れ替えると 33 ポイント違う．この照合を stats に入れれば並びの食い違いを検出できる．(2) stats が書く sha256 は評価の後に取る．評価の後に p1・p3 が流し直されても検出できず，p2 は記録すらない（cons は p2 に依る）．
- 計算の後で直せるか：直せる（stats は保存から作る）が，台本を変えるため pseudo より前に直す
- 求める対応：`collect` の `check_rows` を joint にも広げ，選んだ点の p3 の判定から作った UAR と p3 の csv の行の uar を比べ，不一致の数を pairing.csv に足す．evaluate の sha256 の対象を p1・p2・p3 にし，eval_gate_dai4.json の `input_sha256` と照合して食い違えば止める．

### [R] 評価の段が，pseudo の台本・共通の部品が変わっていないことを照合しない
- 重さ：直してから進む
- 該当箇所：dai4.py `check_eval_gate4`（467〜481 行付近），`select_stage`（`pseudo_commit` を書くだけ，403 行）
- 問題：設計 0 節は「pseudo の結果を見た後は変えない」と書くが，eval の守りは pseudo のときの dai4.py・dai1.py・dai_solver.py と今の版が同じかを見ない．pseudo の単位の sha256（`pseudo_unit_sha256`）も select の後に変わっていないかを eval と detect が照合しない．
- 計算の後で直せるか：直せる
- 求める対応：`check_eval_gate4` に `git diff --quiet <pseudo_commit> HEAD -- DAI/exp/dai4.py(solve 部分は分けられないので全体) DAI/exp/dai1.py DAI/exp/dai_solver.py DAI/exp/dai_data.py` と，frozen の `pseudo_unit_sha256` と現在の pseudo の npz の sha256 の照合を足す．dai4.py の守りの部分だけを後で変えたいなら，守りを別のファイルに分けて pseudo の後でも変えられる範囲を明示する．

### [I1] `--selftest-g1` に合否が無く，許容の書き方が実測と合わない
- 重さ：記録する
- 該当箇所：dai4.py `_g1check` の行（229〜240 行付近），設計 1 節（「学習器の許容 1e-6 の範囲」），9 節（99.9% 以上）
- 問題：判定の一致 99.9% 以上は設計 9 節の基準だが台本は数を行に残すだけで止めない．設計 1 節の 1e-6 は W の差の実測（最大 4.5e-5）より小さい．
- 計算の後で直せるか：直せる
- 求める対応：自己検査の最後に `pred_agree.min() < 0.999` で止める．設計 1 節の文を「判定の一致 99.9% 以上，W の差 1e-4 以下，b の差 1e-3 以下（GPU の計算の揺れ）」に直す．

### [I9] 計算機の検査が eval だけで，pseudo は計算機を問わない．同じ設定の再計算は実行間で判定が揺れる
- 重さ：記録する
- 該当箇所：dai4.py 536 行（H200 の検査は eval のみ），設計 6 節（pseudo も H200 と書く）
- 問題：pseudo を A5000 で流し，eval を H200 で流すと機械がそろわない．select は単位ごとの計算機名を照合しない．同じ設定の再計算で判定が 1/180,120 点変わった（GPU の揺れ）．再開は保存を読むので影響は出ない．
- 計算の後で直せるか：直せる
- 求める対応：select で pseudo の全単位の計算機名が 1 種であることを `frozen.json` に書く．設計 6 節に「判定は GPU の揺れで 1e-5 点ほど変わりうる」と書く．

### [I3・記録] 設計 4 節の守り 6 の文が「ラベルを読まない」より広い
- 重さ：記録する
- 該当箇所：設計 4 節の守り 6，dai1.py `solve_items`（`S.uar_batch(u.ytest, Z)`）
- 問題：学習の経路（G と `fit`）はラベルを読まない（並べ替えの検査で確認）が，行の uar は `solve_items` が fit の後にテストのラベルで計算する．設計の「ラベルは stats の後処理でだけ読む」は文字どおりには成り立たない．
- 求める対応：設計の文を「学習の経路（G，fit）はラベルを読まない．行の uar は fit の後にテストのラベルで計算する（実験 1 と同じ）」に直す．

### [I8・記録] 作業木の未追跡ファイルが eval と select を止める
- 重さ：記録する
- 該当箇所：dai1.py `git_commit`（`--untracked-files=all` で DAI/exp を数える）
- 問題：今 DAI/exp に未追跡の `explore_fusion_grid.py` があり，select・detect・eval・stats はそれだけで止まる．
- 求める対応：起動の前に DAI/exp の未追跡ファイルをコミットするか移す．

### [S] marks の「正解の感情に置かれた重みの割合」が K 本を含む
- 重さ：記録する
- 該当箇所：dai4_stats.py 482〜485 行（`wfrac`）
- 問題：定義の関門の D2 は「K 本を除く候補のラベルで数える」と書く．実装は全 pool で数え，K 本の発話に置かれた疑似ラベルの重みも入る．差は K の小さい設定で小さい．
- 求める対応：設計の文に「K 本を含む全 pool」と書くか，`nk` で絞る．

### e4e3f47 の指摘のうち，直っていることを確認したもの
- `_weights` の S_k と印の無い重みの割合（作り直しと一致），Q の保存，印の割合の分母の K 本除外，eval で候補のラベルを読まない（`check_teacher` は pseudo のみ．並べ替えの検査で確認），対照の評価の回の格子の全点，per-emotion の conf・rand，守り 1・2・4・5，H200，select の pseudo_unit_sha256，対の照合の表と停止，`cons_g1`・partner・agree・cotrain・coreg・joint_a1・取りこぼし（joint も）・感情ごとの再現率の比べ．

## 計算の後の点検（関門 2 の前）への申し送り
- 欠けと重複：eval は 4 T × 50 単位 × 1,140 問（本番 240 + cons_self 180 + own・conf・rand 各 240）．単位ごとの問の識別子は (method, student, K, rho, gamma)．重複は 0 のはず．
- 書き出されているべき出力：p4_K*_Q，`ret_*_am`，`cluster_test`，`_weights` の S_k と unmarked_wfrac，stats/ の pairing.csv・teacher_check.csv・sensitivity.csv・marks_variants.csv，eval_input_sha256.json．
- 結果のコミット番号が b018ae0 と同じか（今回は台本を変えるので，この点は「直した版」のコミットと比べる）．
- 今後直す台本は，上の指摘のうち 止める と 直してから進む のものを入れた版を差分で再検査する．

## 目録に無い種類の不備
- 提案：9.3 節 R に「評価の後の再集計を塞ぐ守り」を足す．集計の台本の守りが「今のコミット = 評価のコミット」を求めると，査読で解析を足せない（本件の stats 250 行）．守りは「評価の単位のコミットが祖先で，結果を決める台本・共通部品・固定のファイルが変わっていない」に書く．
- 提案：9.3 節 S に「別の保存と対にする条件の検査は，片側の並びが保存の列から検証できるか確かめる」を足す．p3 にテストの id が無く，`sample_test` の一致の assert が自分自身との比べだった（本件の pairing）．
- 旧版（e4e3f47）の同じ並べ替えの検査で，cons_conf と cons_rand の印が実行の間で一致しなかった．原因を特定できていないが，最新版では並べ替えても一致する．最新版の出力の同一性は，pseudo の後に `select` の前で 2 単位を流し直して印を比べると確かめられる．

ファイル（すべて DAI/checks/DAI-4/code_run/ の下）：tools/verify2.py（保存の作り直しと照合），tools/run_permuted2.py（ラベルの並べ替え），tools/try_gate.py（評価の守り），tools/try_select_detect2.py（select と detect の偽の置き場），tools/try_collect_extras.py（stats の collect），tools/try_pair.py（p3 の並びの照合），tools/try_pick.py（pick4 と pick の比べ），tools/time_stats.py，出力 out2/a，out2/b，out2/perm，out2/fake_all，gate_test/．旧版（e4e3f47）の検査の出力は out/ と src/．
