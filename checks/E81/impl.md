# E81 の台本の実装の記録（`exp/g3_split.py`・`exp/analyze_e81.py`）

2026-11-02．台本の担当（サブエージェント）．評価の値（UAR・向きの損など）は画面に出しておらず，この記録にも書かない．書くのは差・数・所要だけである．
設計の正本は `docs/exploratory-analyses.md` の E81（4・5 節は関門 1 の後に書き直した版．コミット 4a52140）．

## 1. `exp/g3_split.py`（GPU．単位を作り直して解く）

E77（`compare_dai.py`）・E78（`compare_dai2.py`）・E80 段 c（`e80_stage_c.py`）・段 a（`e80_teacher_j.py`）の関数を import して呼ぶ．この 4 本と `analyze_e79.py` は書き換えていない．

| 部品 | 行 | すること |
|---|---|---|
| `block_labels` | 120 | 塊（IEMOCAP は `session`，MELD は `split|dialogue_id`）を発話一覧から引く．`analyze_e79.cluster_labels` と同じ作り方．欠けがあれば止まる |
| `G3Ctx` | 133 | `compare_b1.Ctx` を継いで，テストと今の K 本の並びはそのまま，学習側＝テスト以外からテストと同じ塊の発話を除いたもの（`pool[r]`），K 本＝今の感情ごとの並び（`map.kshot_order`，種 r）から除いた発話を抜いた先頭 K 本，中心＝新しい学習側の平均．`exclude=False`（`--no-exclude`）なら今の分割と同じ．除くものが無い反復では今の候補・K 本と同じことを assert する |
| `check_counts` | 169 | 照合 (d)：反復ごとのテストの本数，今の候補，除いた本数，新しい候補，最小の感情の本数，今の K=5・25 のうち除かれた本数，テストの塊の数が `checks/E81/g3_counts_all.csv` と同じか．違えば止まる |
| `read_chosen` | 191 | E79 の `select/chosen.csv` の `--variant`（既定 incl）・`--selectors`（既定 oursx・kd・groupB）・K の行を読む．none を選んだ行は none（足す行は無い），候補なし（`__none__`）は記録する．選び直さない |
| `read_chosen78` | 231 | E78 が評価で種 1・2 を解いた設定（E78 の `_chosen.csv`）を，chosen.csv の excl の dann・mme・mapctl の行から作る（E79 は両者が同じことを確かめて止める）．DANN・MME の束の中身を本番にそろえるために使う |
| `check_kept` | 249 | 照合 (e)：oursx・kd が選んだ教師が `--kept` で残した教師か（台本の行が無ければ即興の行．E79 の `load_kept` と同じ） |
| `unit_pass1x` | 271 | `e80_teacher_j.unit_pass1` の写し．段 c と同じ float32 の配列 u に加えて，E77 の `world_unit` と同じ float64 の教師（pF・pFK・ref・ref0・poe0・ownK）と none の W・b を返す |
| `agg_units`・`check_agg` | 305・332 | agg_T（poolC・mlp）：T の仮のターゲットの全単位（分割は今のまま）の生徒の出発点・ラベル・人物だけを作り，`e80_stage_c.fit_agg` を種 crc32(world\|T\|eval) で呼ぶ．照合 (c)：`--e80c-dir` の `<T>_pool.csv`（`*eval*` の `final` の行）と λ・選び方・単位の数・行の数・種・メンバーが一致し，MLP の損失の相対の差が 1e−9 以下か．違えば止まる．ファイルが無ければ `not_checked` と記録する |
| `Teachers` | 361 | 波 e77 の教師は E77 と同じ float64 の配列，波 e80c の教師は `e80_stage_c.teacher_array` の float32 を float64 にしたもの．波と教師の組が E77・段 c の教師の集合と合わなければ止まる |
| `solve_student` | 413 | 1 つの生徒の選んだ設定だけを解く：dist・kd・kdh（`ES.rows_of`・`temp_rows`・`CD.kdh_solve`），self・entmin・IM（E77 の `run_ab` と同じ全格子の束で解いて選んだ行だけ残す），t3a，lp，dann・mme・mapctl（`run_map_sel`） |
| `map_chunks`・`run_map_sel` | 517・536 | E78 の `run_map` と同じ束（(学習率, 反復の回数, 種) ごとに 30 問，次元が 1,024 を超えれば 12 問．種 0 は全候補，種 1・2 は chosen78）を作り，選んだ設定を含む束だけを解く．TF32 はこの間だけ真（E78 と同じ） |
| `coral_chosen` | 575 | `compare_dai2.run_coral` の写し．選んだ (r, ρ_M) だけを，元と同じ束（K × C_b の群と C′ の全部）で解き，選んだ param の行だけを足す |
| `run_target` | 658 | 単位ごとの流れ：照合 (a)(b)(d) → agg_T → 生徒（`make_students`）→ CORAL → `unit_pass1x` と none の行 → 選んだ設定 → 保存 |
| `main` | 828 | 引数，読み込んだコードの sha256 と dirty の判定（未コミットなら止まる．`--allow-dirty` は試しだけ），provenance |

照合 (a)：AVT・AV の IEMOCAP 台本の反復 3・8 は除く発話が 0 で，候補が今と同じであることを確かめる（`run_target` の冒頭）．照合 (b)：`label_sets_of` が通らなければ止まる．

引数：`--world`，`--targets`（既定は 3 群），`--reps`（0〜9），`--ks`（既定 0 5．行を残す K），`--batch-ks`（既定 0 5 25．束に入れる K．本番の E77・E78・E80c の `--ks 0 5 25` と束の中身をそろえる），`--chosen`，`--selectors`，`--variant`，`--kept`，`--e80c-dir`（既定 `results/e80c_<世界>`），`--counts`，`--pseudo-reps`，`--nc-seeds`（既定 1 2），`--e78-methods`（既定 dann mme mapctl），`--out`（既定 `results/e81_<世界>/`），`--gpu-fraction`（既定 0.3），`--no-exclude`，`--check`（分割と chosen.csv の照合だけ．解かない），`--allow-incomplete`・`--allow-dirty`（試しだけ）．

保存（E77 と同じ形）：`<T>.csv`（run, target, rep, row_id, student, method, param, meff, K, uar, wave, selectors, pnorm, seed, split, n_pool, commit），`_logp.npy`（float32 の対数の確率），`_pred.npy`，`_te.npy`，`_pool.npy`，`_excl.npy`，`_kshot.npz`（K 1・5・10・25 の行番号），`_teachers.npz`（使った教師の配列．鍵は「波__教師__K<K>」），全行の W・b，`<T>_units.csv`（学習側の大きさと所要），`_plan.csv`，`_pool.csv`（agg_T），`_conv.csv`・`_convp.csv`・`_diag.csv`（DANN・MME の束の大きさを含む），`_missing.csv`，`_samples.csv`，`_settings.json`，provenance．

## 2. `exp/analyze_e81.py`（CPU．書き直した 4 節の読み方）

| 部品 | 行 | すること |
|---|---|---|
| `SrcStore` | 126 | 選んだ鍵（oursx・kd・groupB と none）の点の値と複製を，E79 の `stage_measure`・`measure_target` と同じ手順で作る：点の値（種を反復ごとに平均して反復で平均），人物と反復の 2 段のブートストラップ（`boot_draws`），G_eff < 20 なら人物の jackknife，塊（MELD の対話，IEMOCAP のセッション）のブートストラップ（`tag="cluster"`）と塊が 10 未満なら塊の jackknife．全反復の版と，IEMOCAP 台本の反復 3・8 を足さない版を同じ抽出で同時に作る．E81 の出力（保存した候補で D を作る）と，E77・E78・E80c の本番の出力（候補＝テスト以外．E79 と同じ）のどちらも読める．`a_auc`・`e_model`（模型に依る単位の印の材料）も E79 と同じ式で行ごとに作る |
| `E79Store`・`CurStore` | 331・362 | 今の分割の主は E79 の `measure/`（`analyze_e79.Store`）．台本の反復 3・8 を除いた版は本番の出力から作り直した `SrcStore` |
| `parts`・`intervals`・`compare` | 382・405・441 | E79 の `stage_judge` の `compare` と同じ区間：主（百分位．G_eff < 20 は jackknife の t 区間との和集合），`cluster_boot`（主 ∪ 塊の百分位 ∪ 塊の jackknife の t）．ΔΔ は同じ抽出ごとの差で同じ区間を作る |
| `transition` | 459 | Δ_cur と Δ_G3 の判定の移り：unchanged，both_undet（両方で判定できない．「同じ」に数えない），adjacent（隣の区分），reversal（上回った ↔ 下回った），missing |
| `model_dep` | 474 | E79 と同じ「模型に依る」の印（P5-3 と向きの損の行．分割ごと） |
| `combos`・`keys_of` | 483・499 | E79 の `summarize` の数え上げ（母数は世界の全コーパス）と `statements_p5` で文の鍵を作る．P5-3u は E79 に文の行が無いので，P5-1 と同じ 3 つの区分の鍵を作る（下の 5 節の決めごと） |
| `key_2corpus` | 523 | 記述：IEMOCAP と MELD の 2 コーパスだけの鍵 |
| `main` | 541 | 照合，単位の表，移りの表，主の読み方（E79 の `judge/units.csv` の IEMOCAP・MELD を差し替えて鍵を作り直す），感度の版，規則 (i)・(ii)，同じ単位の照合，provenance |

主の読み方：E79 の `judge/units.csv` の主の行（sensitivity ＝ main）の P5-1・P5-2・P5-3・P5-3u（K 0・5）のうち IEMOCAP・MELD の判定だけを G3 の判定に替え，MOSEI・AFEW・CREMA-D は E79 の判定のまま `statements_p5` を当てる．今の分割の鍵は E79 の判定のまま同じ関数で作り，E79 の `judge/statements.csv` の鍵と一致することを照合する（`cur_keys_vs_e79_statements`）．今の分割の IEMOCAP・MELD の単位を E79 の measure から作り直した値と判定が E79 の `units.csv` と同じことも照合する（`cur_units_vs_e79_units`）．
感度の版：区間を `cluster_boot` に替えた版，IEMOCAP 台本の反復 3・8 を除いた版（今の分割は本番の出力から作り直す．`--e77` などが要る），IEMOCAP を即興だけにした版．どれも同じ規則で鍵を作り直す（`statements_sens.csv`）．K ごとの数は `transitions.csv` の K の列と `rule.json` の `ii_K0`・`ii_K5`．
規則 (ii)：P5-3u（oursx − none の UAR）の単位のうち選択が定義から 0 にする単位（両方の腕が同じ設定）を除き，ΔΔ の判定が「同じ水準」の数が過半で，Δ_cur と Δ_G3 の判定の向きの逆転が 0 のとき真．
変わった鍵ごとに，判定の変わった単位の数と，そのうち ΔΔ の区間が 0 を外れた数を `statements_main.csv` に出す（4 節の雑音の見込み）．`units_dd.csv` に反復の数，話者の数，塊の数，有効な複製の数，選んだ m_eff の種類（n か固定か），「検出できる ΔΔ はおよそ 1.5δ 以上」の注，位置の損の行の E75 の注を付けた．
出力：`units_dd.csv`，`transitions.csv`，`statements_main.csv`，`statements_sens.csv`，`statements_2corpus.csv`，`rule.json`，`rows_g3_<世界>.csv.gz`・`rows_cur_<世界>.csv.gz`，`checks.csv`，`identity_same_split.csv`，`provenance.json`．画面には行の数とファイルの場所だけを出す．

## 3. 試し（lab の A5000，`--gpu-fraction 0.15`，GPU はほかの仕事と共有）

試し用の入力は scratchpad の `e81_test/inputs/` にある．`TEST_ONLY_chosen_av.csv`・`TEST_ONLY_chosen_avt.csv` は E77・E78・E80c の格子にある設定から手で作った偽の chosen.csv（E79 の出力ではない），`TEST_ONLY_kept_av.csv` は偽の kept.csv である．AVT の試しは本物の `results/e80/kept.csv` と照合が通る設定にした．
3.1〜3.4 の GPU の試しは，台本と台帳のコミットの前に流した（関門 1 の leak の指摘．台帳の「E81（台本の担当の試走）」の行）．その後に AVT の IEMOCAP 即興の反復 0 も 1 単位解いた（3.4）．

### 3.1 分割の照合（`--check`，CPU）
AVT・AV の 3 群 × 反復 0〜9 の 60 単位で，学習側の大きさの 8 つの数が `g3_counts_all.csv` と全部一致し，台本の反復 3・8 で除く発話が 0 であること（照合 (a)），`label_sets_of` が通ること（照合 (b)）を確かめた．AV の IEMOCAP 即興の反復 0 では，G3 の候補 ⊂ 今の候補，G3 の候補 ∪ 除いた発話 ＝ 今の候補，テストのセッション（1・5）が候補に残らないこと，K 本 1・5・10・25 が入れ子で全部が候補の中にあり，今の K 本のうち除かれなかったもの（K=5 で 13 本＝20 − 7）が全部残ることを確かめた．未コミットのコードでは止まること（`--allow-dirty` なし）も確かめた．

### 3.2 今の分割と同じ単位の一致（`--no-exclude`，AV の IEMOCAP 即興の反復 0）
同じ単位を `compare_dai.py`（`--pseudo-targets none --ks 0 5 25`）と `compare_dai2.py`（`--methods coral dann --ks 0 5 25`）で新しく解き，同じ鍵の行の対数の確率を比べた（`compare_identity.py`）．35 行のうち比べた 24 行：

| 手法 | 行 | 対数の確率の差の最大 | 予測の一致の最小 | UAR の差 |
|---|---|---|---|---|
| none（E77・E78 の両方） | 6・6 | 0 | 1 | 0 |
| entmin（種 0〜2） | 3 | 0 | 1 | 0 |
| IM（種 0〜2） | 3 | 0 | 1 | 0 |
| self・t3a・kdh | 1・1・1 | 0 | 1 | 0 |
| CORAL | 2 | 0 | 1 | 0 |
| DANN（種 0） | 1 | 0 | 1 | 0 |
| dist（波 e77） | 2 | 2.3e−5 | 1 | 0 |
| kd | 2 | 4.8e−7 | 1 | 0 |
| lp | 2 | 1.9e−6 | 1 | 0 |

E77 の教師の配列（pFK・ref）4 つもビット単位で同じだった．凸の解き（dist・kd・lp）だけ，束の中身（選んだ 1 問か全格子か）で L-BFGS の止まる点がずれる．self・entmin・IM を E77 と同じ全格子の束で解き，DANN を E78 と同じ 30 問の束で解くようにした後の値である（1 問だけで解いた初めの版では entmin の差は 1.3e−3，DANN は予測の一致 0.935 だった）．
DANN・MME の種 1・2 と MME（参照の側で回していない）は比べていない．

### 3.3 E80 段 c の道（同じ単位．プロセスの中で比べた）
`e80_stage_c.py` を丸ごと流す試しは，MOSEI の仮のターゲットの単位で GPU の 0.15（3.6 GB）を超えて落ちた．代わりに `test_e80c_path.py` で，(1) `unit_pass1x` の u が `unit_pass1` の u と全部の配列でビット単位で同じ，(2) `agg_units` の出発点・ラベル・人物が `unit_pass1` の単位（AFEW・CREMA-D の反復 0）とビット単位で同じ，(3) 同じ agg で，段 c の `solve_unit` の行と `solve_student` の波 e80c の行（10 行．dist・kd・kdh，poolC・mlp・ds・agree・poe0・ref0 の教師）の対数の確率の差の最大が 2.4e−7，予測の一致が 1，を確かめた．照合 (c) は，`<T>_pool.csv` を同じ値で置いたときに ok，損失・λ・種のどれかを変えたときに止まることを CPU で確かめた（本物の段 c の `_pool.csv` との照合は lab では試せない）．

### 3.4 G3 の分割で解いた単位と所要
| 世界・群・反復 | 候補（今 → G3） | 行 | 1 単位の所要 | agg_T | GPU の最大 |
|---|---|---|---|---|---|
| AV・IEMOCAP 即興・0（今の分割） | 2,327 | 35 | 52 秒 | 206 秒（20 単位） | 2.4 GB |
| AV・IEMOCAP 即興・0（G3） | 2,327 → 1,691 | 35 | 48 秒 | 93 秒 | 2.3 GB |
| AV・MELD・0（G3） | 9,193 → 5,980 | 35 | 99 秒 | 98 秒（25 単位） | 2.8 GB |
| AVT・IEMOCAP 即興・0（G3） | 2,327 → 1,691 | 41 | 83 秒 | 51 秒（10 単位） | 2.7 GB |

行の数はどれも期待の数と同じだった．試しの設定は 1 つの（生徒, K）に 3〜5 個の手法を置いたので，本番（oursx・kd・群 B の 3 つ）より行が多い．AVT の融合の生徒の CORAL（r = 128）は 0.15 の上限で SVD が落ちたので，AVT の試しでは IM に替えた（E78 の本番は 0.22 で同じ計算を流している）．agg_T の所要は GPU の混み具合で 51〜288 秒と揺れた（大半は `pool_info` と λ の選び方）．

### 3.5 解析の台本の試し（書き直す前の版で済ませたもの）
- 本物の `analyze_e79.py --stage measure` を，3.2 の出力を波ごとに E77・E78・E80c の形に分けた偽の場所（`fake79/`，δ は試し用の 1 点）に流し，同じ単位を `analyze_e81.py` で E79 の道と自前の道の両方から読んだ．点の値と区間の端の差の最大は 3.6e−15，ΔΔ は 0，判定の違いは 0 だった（自前のブートストラップが E79 の measure と同じ手順であることの確かめ）．
- 学習側が変わらない単位の照合の道（`SAME_SPLIT` を即興の反復 0 に替えて流した）で，点の値の差の最大 3.6e−15，対数の確率の差 0 だった．

### 3.6 書き直した版の試し
（この節はコミットの後に流した試しで埋める）

## 4. 所要の見込み

A5000（共有，0.15）の試しから：AV の IEMOCAP の 1 単位は約 50 秒，MELD は約 100 秒，AVT の IEMOCAP は約 80 秒（MELD は約 150〜200 秒と見込む），agg_T は（世界, T）ごとに約 1〜5 分．本番の 60 単位（2 世界 × 3 群 × 10 反復）は A5000 でおよそ 1.5〜2.5 時間に当たる．H100 は A5000 の約 3.5 倍速い（`checks/E80/code.md`）ので，本番は 1 世界ずつ順に流しておよそ 30〜60 分と見込む．群 B が DANN・MME（反復 1,000 回の 30 問の束を種 0 で解く）や CORAL（r = 128，AVT の融合の生徒）を多く選ぶと伸びる．解析（CPU）は 1,000 回の複製で数分〜十数分．

## 5. 設計から変えたこと・決めたこと（理由）

1. 保存は float16 の確率ではなく float32 の対数の確率（E77 と同じ形）にした．float16 では 6e−8 より小さい確率が 0 になり，向きの損・位置の損に使う対数の比が無限になるからである．
2. 束の中身を本番にそろえた（`--batch-ks 0 5 25`，self・entmin・IM は E77 の全格子の束，DANN・MME・mapctl は E78 の 30 問の束，CORAL は選んだ (r, ρ_M) の元の束）．DANN は束の中身で値が種の違いほど変わる（E78 の再検査 I5-2．3.2 節の初めの版でも予測の一致が 0.935 だった）ので，選んだ設定だけを解くと今の分割の値と比べられなくなる．K=25 の行は解くが保存しない．
3. DANN・MME の種 1・2 の束は E78 の `_chosen.csv` で決まるので，chosen.csv の excl の行（E79 が E78 と同じことを確かめた選択）から作る．行が無ければ選んだ設定だけを足し，provenance の notes に書く．
4. agg_T のための仮のターゲットの単位は，段 c の `pseudo_pass_a` を丸ごと回さず，生徒の出発点・ラベル・人物だけを作る（同じ値になることを 3.3 で確かめた）．`fit_agg` は段 c の関数そのもの．
5. 解析の P5-3u の文の鍵：E79 の `statements_p5` は P5-3u の鍵を作らない（P5-3 の受け皿としてだけ使う）．4 節 (i) は P5-3u の鍵も比べると書くので，P5-1・P5-2 と同じ 3 つの区分（none が oursx と同じ水準か上回った／oursx が none を上回った／判定できない）の鍵を作った．本体が別の鍵を望むなら `keys_of` の最後の繰り返しを替える．
6. (i) で，両方の分割で「判定できない（受け皿の行）」の鍵も同じに数える（4 節の (i) の文のまま）．その数を `rule.json` の `i.n_both_undetermined` と `statements_main.csv` の列に出す．(ii) の「向きの逆転」は，P5-3u の単位の Δ_cur と Δ_G3 の判定が上回った ↔ 下回ったに移ったことと読んだ．
7. 台本の反復 3・8 を除いた版の今の分割は，E79 の measure が反復ごとの複製を残さないので，E77・E78・E80c の本番の出力から E79 と同じ手順で作り直す（`--e77`・`--e78`・`--e80c`）．全反復の版が E79 の measure と同じ値になることを照合する（`cur_store_vs_e79_measure`，許容 1e−6）．
8. D（向きの損の基準の向き）は各分割の候補のラベルで作る．前向きの「none − 手法」では打ち消し合うので Δ と ΔΔ に入らない．逆の順序の量は分割の間で比べない．

## 6. 試せなかったこと

- 本物の E79 の出力（chosen.csv・delta.csv・kept_used.csv・measure・judge）と E77・E78・E80c の本番の出力が lab に無いので，本番の照合（照合 (c) の段 c の `_pool.csv`，学習側が変わらない単位の本番との対数の確率，`cur_keys_vs_e79_statements` など）は RunPod で初めて当たる．
- MME，DANN の種 1・2，AVT の融合の生徒の CORAL は参照と比べていない（3.2・3.4）．
- 1 単位ずつしか解いていないので，10 反復の流れ（再開，agg_T の使い回し，IEMOCAP 台本）は通していない．

## 7. RunPod での流し方（本体が gate-code の検査の後に）

```
# 前提：E77・E78・E80c が終わり，E79 の select・measure・judge が済んでいる．2 本の台本をコミットしてある
./.venv/bin/python exp/g3_split.py --world fuse-audio-video-text --check          # 分割と chosen.csv・kept の照合（解かない）
./.venv/bin/python exp/g3_split.py --world fuse-audio-video --check
./.venv/bin/python exp/g3_split.py --world fuse-audio-video-text --gpu-fraction 0.3
./.venv/bin/python exp/g3_split.py --world fuse-audio-video --gpu-fraction 0.3
./.venv/bin/python exp/analyze_e81.py --e81 fuse-audio-video-text=results/e81_fuse-audio-video-text fuse-audio-video=results/e81_fuse-audio-video \
    --e79 results/e79 --e77 fuse-audio-video-text=results/e77_fuse-audio-video-text fuse-audio-video=results/e77_fuse-audio-video \
    --e78 fuse-audio-video-text=results/e78_fuse-audio-video-text fuse-audio-video=results/e78_fuse-audio-video \
    --e80c fuse-audio-video-text=results/e80c_fuse-audio-video-text fuse-audio-video=results/e80c_fuse-audio-video
```
`g3_split.py` は既定で `results/e80c_<世界>/<T>_pool.csv` と照合し（照合 (c)），`results/e80/kept.csv` と `checks/E81/g3_counts_all.csv` を読む．
