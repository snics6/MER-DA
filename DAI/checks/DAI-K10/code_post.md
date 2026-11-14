# 計算の後の点検：DAI-16・17 の下調べの K = 10 の追補（2026-11-14，追補のコミット 0c14e4e）

## 判定

通す．DAI-16（120 単位）と DAI-17（単位 120，組 60，生徒 120，`srcdb` 24）の出力は欠け 0・重複 0・設計に無い非有限 0 で，各段のコミットは 1 つ，dirty は偽だった．台本の sha256 は検査を通した値（`dai16_probe.py` 62863fa8…，`dai17_probe.py` 1a99e29b…）で，K は 10 だけだった．`dai1_frozen_sha256` と入力 32 本の sha256 は K = 5・25 の回と一致し，`scripts` で違うのは DAI-16・17 の台本だけだった．K に依らない量は K = 5・25 の回と差 0 で一致した．抜き取りの作り直しと数え直しの不一致も 0 だった．直すものは無い．結果の文の書き方にかかわる 4 点を「記録する」に書いた．

- 読んだもの：追補 `DAI/docs/実験16・17_下調べ_K10_追補.md`（第 2 版，0c14e4e），起動の台本 `DAI/run/dai_k10.sh`，計算の前の検査 `DAI/checks/DAI-K10/code_v2.md`，元の回の計算の後の点検 `DAI/checks/DAI-16/code_post.md`・`DAI/checks/DAI-17/code_post.md`，`results/dai16_probe_k10/`・`results/dai17_probe_k10/` の全体，ログ 8 本と `logs/dai_k10.done`．突き合わせのために `results/dai16_probe/`・`results/dai17_probe/` と入力（`results/dai1/source/*.npz`，`DAI/splits/*.csv`）を読んだ．
- GPU は使っていない（検算はすべて `CUDA_VISIBLE_DEVICES=""` で流した）．計算はやり直していない．`results/` のファイルは読むだけにした．`results/dai1/eval/` は開いていない．
- 画面に出したのは，数・一致・不一致・差の最大・検算の量だけである．UAR，差の符号，判定の語の内訳，選んだ点は見ていない．
- 検算の台本と出力は `DAI/checks/DAI-K10/code_run_post/` に置いた．元の回の点検の台本を写して置き場と K を替えたもの（`d16_units.py`・`d16_summary.py`・`d17_inventory.py`・`d17_summary.py`），K に依らない量の突き合わせ（`cmp_kindep.py`），mixsel の同点の確かめ（`d16_ties.py`），依存の表の作り直し（`dep_k10.py`）である．出力は同じ名前の `.out`．

## 1．コミット，dirty，台本と入力の sha256，K

| 項目 | 期待 | 実際 |
|---|---|---|
| DAI-16 の段のコミット | 段ごとに 1 つ，dirty 偽 | units 0c14e4e（07:43），summary dff7b95（08:38）．どちらも dirty 偽．単位の csv の `commit` は 1 種類（0c14e4e），`dirty` は全行で偽．`dai1_commits` は 1 種類（896136c） |
| DAI-17 の段のコミット | 段ごとに 1 つ，dirty 偽 | units・reweight・students・summary の 4 段とも dff7b95，dirty 偽．units・reweight・students の csv の `commit` は 3 段とも dff7b95 だけ，`dirty` は偽だけ．`dai1_commits` は T ごとに 1 種類．`frozen_sel.json` の `units_commit` は dff7b95 だけ |
| posthoc のコミット | | `dai16_posthoc.py` 10:20:56 dff7b95 dirty 偽，`dai17_posthoc.py` 10:21:16 dff7b95 dirty 偽．`dai16_posthoc` の `units_commit` は 0c14e4e，`dai17_posthoc` の `units_commit` は dff7b95 |
| 0c14e4e → dff7b95 の差分 | 台本に触れない | `DAI/checks/DAI-18/` の報告と台本の写し，DAI-18 の設計，`docs/seen-data.md` の 1 行だけ（`git diff --stat`）．`DAI/exp`・`DAI/run`・`measure`・`features`・追補には差分が無い（記録 1） |
| `dai16_probe.py` の sha256 | 62863fa8741277762357e9cc9401bde131c4059e3b1f89ebd3d38daffd390e34 | DAI-16 の 2 段と DAI-17 の 4 段の `scripts["dai16_probe.py"]` がこの値．今の作業木も同じ |
| `dai17_probe.py` の sha256 | 1a99e29b40274867a4377263b6a1e31128b1aa1fcbc16cdcc0edcb542f1ee558 | DAI-17 の 4 段の `this_script_sha256` と `scripts["dai17_probe.py"]` がこの値．今の作業木も同じ |
| posthoc の台本の sha256 | `dai16_posthoc.py` 243a96a9…，`dai17_posthoc.py` 4301835e…（`code_v2.md`） | `posthoc_provenance.json` の `script_sha256` がどちらもこの値 |
| 共通の部品の sha256 | 段の間で同じ | 各回の `provenance_log.jsonl` の全行で `scripts`・`inputs_sha256`・`dai1_frozen_sha256`・`design_sha256`・`KS` はそれぞれ 1 種類．今の作業木の台本と部品の sha256 も記録と一致 |
| K | 10 だけ | `provenance.json` の `KS` は両方とも [10]．単位の csv の K は DAI-16 が 10 だけ，DAI-17 は 10 と −1（K に依らない `_nn` の行），reweight は 10・0・0ob・−1，students は 10 だけ．集計の K は 10 と K に依らない行（DAI-16 の `reliability.csv` の K = 0 の 48 行，DAI-17 の reweight の K = 0・0ob の行）だけ．`frozen_sel.json` の K は 10 だけ．規則の主の行（`main`）の K は 4 つの数えの表で 10 だけ |
| 終わりの印 | | `logs/dai_k10.done`（10:21:18）あり，`logs/dai_k10.failed` なし．`results/dai16_probe_k10/units.done`，`results/dai17_probe_k10/{units,reweight,students}.done` あり |
| 既存の置き場 | 書き換わらない | `dai_k10.done` があるので起動の台本の前後の sha256 の照合は一致した．加えて `results/dai16_probe`・`results/dai17_probe` の下に 07:40 より新しいファイルは 0 個 |

### K = 5・25 の回との `provenance.json` の照合（追補 2 節の「表を作る前の確かめ」）

| 項目 | DAI-16 | DAI-17 |
|---|---|---|
| `dai1_frozen_sha256` | 一致（6926fb3e…） | 一致（6926fb3e…） |
| `inputs_sha256`（32 本） | 鍵の集合が同じで，値の不一致 0．今の `results/dai1/source/*.npz` 28 本と `DAI/splits/*.csv` 4 本の sha256 とも一致 | 同じ |
| `scripts` | 8 本のうち違うのは `dai16_probe.py`（d10ca941 → 62863fa8）だけ | 9 本のうち違うのは `dai16_probe.py`（d10ca941 → 62863fa8）と `dai17_probe.py`（edfcc3de → 1a99e29b）だけ |
| `design_sha256` | 一致（7b5b1e7d…） | 一致（6e4b087b…） |
| ほかの定数 | `stage`・`argv`・`commit`・`time`・`KS` のほかに違う項目 0 | 同じ（`this_script_sha256` を除く） |

## 2．単位・組の欠けと重複，中身

| 出力 | 期待 | 実際 |
|---|---|---|
| DAI-16 `units/` | 120 単位（T 4 × c 3 × 分割 5 × 種 2） | npz 120・csv 120．欠け 0，余り 0，ほかのファイル 0 |
| DAI-16 単位の csv の行 | 1 単位 257 行（K 1 × {版 204，見積もり 37，見込みで選ぶ版 5，`sel_best`，`_est_prod`，参照 8，`_check`}） | 全単位 257 行（計 30,840）．(K, 版) の集合が期待と違う単位 0．(T, c, 分割, 種, K, 版) の重複 0．ファイル名と列の食い違い 0 |
| DAI-17 `units/` | 120 単位 | npz 120・csv 120．欠け 0，余り 0．csv は全ファイル 796 行（版 150 ＋ `_nn` 590 ＋ `_single` 55 ＋ `_check` 1）．同じ識別子の重複 0 |
| DAI-17 `reweight/` | 60 組 | npz 60・csv 60．欠け 0，余り 0．csv は全ファイル 380 行．重複 0．格子 14 点の欠けは (組, K, モダリティ, 重み, 種) のすべてで 0 |
| DAI-17 `students/` | 120 単位 | npz 120・csv 120．欠け 0，余り 0．csv は全ファイル 298 行．重複 0．疑似ラベルは全単位で 6 つ，(単位, K, 生徒, 疑似ラベル) 2,880 組で格子 12 点の欠け 0，実験 1 の設定の点は各組にちょうど 1 つ |
| DAI-17 `srcdb/` | 24 | 24．欠け 0．長さの食い違い・非有限 0 |
| 単位のソース | T と c 以外の 2 コーパス | 外れ 0（DAI-16・17 とも） |

- **npz の鍵**：DAI-16 は元の点検と同じ期待の鍵の一覧（K を 10 にしたもの）と比べて欠け 0・余り 0（120 単位）．DAI-17 は units・students・reweight の 3 つで欠け・余りのあるファイル 0．さらに K = 5・25 の回の同じ名前の npz と比べ，K を名前に含む鍵の集合は「K5 の鍵の K を 10 に替えた集合」とすべてのファイルで同じだった（DAI-16 units 120，DAI-17 units 120・reweight 60・students 120）．形が違う鍵は K 本の数に連動する鍵（`pred_pool_*`，`kpos`，`kfold`，`pcf*`）だけで，違いはどれもちょうど 20 本（K = 10 の 40 本と K = 5 の 20 本の差）だった．
- **長さ・範囲**：DAI-16 は判定の長さ・確率の形の食い違い 0，判定の値の 0〜3 の外 0，確率の 0〜1 の外 0，行の和と 1 の差の最大 5.2e-8．DAI-17 は長さの食い違い 0（units・students・reweight），近い発話の番号が表の外 0，`hub_*` と A の単位の候補の側の近い発話の数えの不一致 0，students の `sid_test`・`y_test` は A の単位と一致．
- **K 本**：DAI-16 で `kpos_K10` が 40 本で重複なく，感情ごとに 10 本でない単位 0．K = 5 の回の `kpos_K5` が `kpos_K10` に含まれ，`kpos_K10` が K = 25 の回の `kpos_K25` に含まれる．この入れ子が崩れる単位は 0（追補 0 節の「入れ子」）．
- **話者と発話**：DAI-16 でテストと候補の sample_id・塊（話者）の重なり 0，sample_id の重複 0．同じ (c, 分割, 種) の 3 つの T でテスト・候補・K 本が違う組は 40 組中 0．
- **NaN・Inf**：DAI-16 の単位の csv の Inf 0，版・参照・見積もり（4,440 行）・見込みで選ぶ版（720 行）・`_check`（120 行，83 列）の NaN 0．npz の非有限は感情の対ごとの AUC の 4 つ（`AUC_M_K` 1,440，`AUC_M_orc` 1,440，`SR_S_homeauc` 2,880，`SR_S_awayauc` 2,880）だけで，すべて 4 × 4 の表の対角（対角の外の NaN 0）．`AUC_M_*` は K が 1 つなので元の回の半分，`SR_S_*` は K に依らないので元の回と同じ数である．DAI-17 は units・students・reweight の npz に非有限を含む配列 0，csv の `uar_*`・`hit`・`sd_pool` などの非有限 0．

## 3．`summary/` と `posthoc/` の行数，欠け，重複

DAI-16（集計のログ：単位 120，差の行 2,396，数えの行 118，選んだ格子の行 20）．

| ファイル | 行数 | 設計と台本から出る数 | 重複 | NaN（設計どおりのもの） |
|---|---|---|---|---|
| `mixsel.csv` | 20 | c 4 × K 1 × 選ぶ版 5 | 0 | `lam` 16（MS 以外） |
| `mixsel_units.csv` | 600 | 120 × K 1 × 5 | 0 | `est_gain` 360（`S_F6`・`S_F3`・`MS_F6` の mixsel） |
| `levels.csv` | 912 | c 4 × K 1 × 228 版 | 0 | `uar_kb`・`uar_raw`・`uar_pool`・`uar_ob` 各 32（見込みで選ぶ版 8 × 4．層は adj だけ） |
| `effects.csv` | 2,396 | 差の定義 288 の層の数の和 599 × c 4 × K 1 | 0 | `df` 599（MOSEI の全行） |
| `criteria.csv` | 118 | 差の行がある (K, 層) を数えた数 | 0 | 無し |
| `estimate.csv` | 28 | c 4 × K 1 × 7 版 | 0 | 無し |
| `sel_best_used.csv` | 19 | (c, K) 4 組 × 使った版 | 0 | 無し |
| `per_emotion.csv` | 40 | c 4 × K 1 × 10 版 | 0 | 無し |
| `reliability.csv` | 72 | c 4 × {K に依らない出どころ 4 ＋ K 1 × 2} × モダリティ 3 | 0 | 無し |
| `reliability_src.csv` | 48 | (home 4 ＋ away 12) × 3 | 0 | 無し |
| `r0.csv` | 32 | c 4 × K 1 × 組 2 × 出どころ 4 | 0 | 無し |
| `r0_count.csv` | 8 | K 1 × 組 2 × 出どころ 4 | 0 | 無し |
| `ret_emo.csv` | 48 | c 4 × 3 × 感情 4 | 0 | 無し（`n_finite` は全行 30） |
| `assumptions.csv` | 4 | c 4 × K 1 | 0 | 無し |
| `posthoc/additive_*` | units 1,920，levels 52，effects 64，corr 16，clip 16 | K = 5・25 の回の行数のちょうど半分 | — | — |

- `effects.csv`：(c, K, a, b, 層) の欠け 0．`n_fr` 10，`n_folds` 5．`est`・`lo`・`hi`・`se`・`p` の NaN 0，`lo > hi` 0，点推定が区間の外 0．自由度は AFEW 13，IEMOCAP 3，MELD 7．方法は AFEW・IEMOCAP・MELD が `jk+seed`，MOSEI が `boot_cl+seed`・`boot_cl_shared+seed`（元の回と同じ）．`folds_dropped` が空でない行は 1,170 で，元の回の記録 1 と同じ扱いである．
- `levels.csv`：(c, K) ごとの版はどれも 228，`n_fr` 10，`uar_adj` の NaN 0，0〜100 の外 0．
- `posthoc_provenance.json`（DAI-16）の `agree_saved_min` は 8 版すべて 1.0．

DAI-17（集計のログ：A 差の行 1,680・数えの行 75，B 組 60・差の行 168・数えの行 42，C 単位 120・差の行 272・数えの行 68）．

| ファイル | 期待 | 実際 |
|---|---|---|
| `sel.csv` | 4 × 1 × 11 = 44 | 44 |
| `levels.csv` | 4 × 1 × 161 = 644 | 644．欠け 0，余り 0，重複 0，`n_fr` 10 |
| `effects.csv` | 4 × 1 × 420 = 1,680 | 1,680．重複 0 |
| `criteria.csv` | 層の数 × K の和 = 75 | 75．`main` 21 行，`proceed` 3 行 |
| `per_emotion.csv` | 4 × 1 × 9 × 4 = 144 | 144．`n_units` 30 |
| `neighbors.csv`・`_by_source`・`_rules`・`_count` | 2,440・240・360・90 | 同じ |
| `single.csv` | 4 × 1 × 58 = 232 | 232 |
| `reweight_sel.csv` | 4 × 1 × 3 × 4 = 48 | 48 |
| `reweight_levels.csv` | 4 × (K0・0ob の 24 ＋ K10 の 16 ＋ 選んだ点 12) = 208 | 208．`n_fr` 10 |
| `reweight_effects.csv` | 4 × (18 ＋ 21 ＋ 3) = 168 | 168．重複 0 |
| `reweight_criteria.csv` | 42 | 42．`main` 9 行 |
| `reweight_weights.csv` | 36 | 36 |
| `students_sel.csv` | 4 × 1 × 4 × 6 = 96 | 96 |
| `students_levels.csv` | 4 × 1 × 4 × 13 = 208 | 208．`n_fr` 10，重複 0 |
| `students_per_emotion.csv` | 832 | 832．`n_units` 30 |
| `students_effects.csv` | 4 × 1 × 4 × 17 = 272 | 272．重複 0 |
| `students_criteria.csv` | 1 × 4 × 17 = 68 | 68．`main` 12 行 |
| `students_conditions.csv` | 96 | 96 |
| `posthoc/students_cond_sel.csv` | 96 | 96（元の回 192 の半分） |
| `posthoc/dependency_count.csv`・`_detail.csv` | 20・60 | 20・60 |

- 区間が NaN の行は 0（`effects.csv` 0 / 1,680，`reweight_effects.csv` 0 / 168，`students_effects.csv` 0 / 272）．点推定・`se` の NaN 0，下端 > 上端 0，点推定が区間の外 0，`n_folds` 5，`n_fr` 10．
- 残る NaN は構造によるものだけである：`sel.csv` の 28 個（連動する版 8 行の `k`・`n_tied`・`n_cands`，`N_eq_sel` 4 行の `lam`）と，MOSEI の行の `df`（`effects.csv` 420，`reweight_effects.csv` 42，`students_effects.csv` 68．元の回の記録 2 と同じ）．
- 4 つの数えの表の `main` の行は，どれも `n_corpora` が 4 だった．

## 4．検算の量とログ

| 量 | 許し | K = 10 の回 |
|---|---|---|
| DAI-16 `convergence.json`：作り直しの積の差の最大 `teacher_dev_max` | 1e-4 | 7.55e-5（声 7.0e-5，顔 5.9e-5，言葉 4.3e-5）．単位の csv の `_check` から作り直した値と一致 |
| 同：勾配の最大・止まり・`adj_resid_max` | 1e-5・0・`ADJ_TOL` 1e-6 | 1.0e-6・0・1.0e-8 |
| DAI-17 `convergence.json`：`teacher_dev_max`・勾配・止まり・`adj_resid_max`・GPU | 1e-4・1e-5・0・—・6 割（約 14.7 GB） | 5.73e-5・1.0e-6・0・1.0e-8・5,524 MB |
| `reweight_checks.json`：`none_agree_min`・`start_dev_max`・`base_dev_max`・勾配・止まり・GPU | 0.995・1e-3・1e-3・1e-5・0・同上 | 0.99965・7.95e-5・3.58e-5・1.0e-6・0・6,527 MB |
| `students_checks.json`：`prop_agree_min`・`teacher_agree_min`・勾配・止まり・GPU | 0.995・0.995・1e-5・0・同上 | 0.99863・1.0・1.0e-6・0・4,071 MB．1 未満は 480 行のうち 3 行（`MOSEI_IEMOCAP_f0_r1` AVT，`IEMOCAP_MELD_f2_r0` AVT，`MELD_MOSEI_f2_r0` T．記録 3） |

- 所要：DAI-16 の単位の段 3,272 秒（約 55 分．追補 3 節の見込みは約 1 時間 10 分），DAI-17 の A の段 2,125 秒．保存の大きさは DAI-16 `units/` 394 MB，DAI-17 `units/` 2.1 GB・`students/` 437 MB・`reweight/` 66 MB・`srcdb/` 17 MB．
- ログ：`dai16_k10_units.log` は 156 行で，単位の行 120（すべて「行 257」）と警告 18 件（36 行）．警告は `RuntimeWarning: Mean of empty slice` で，`dai16_probe.py:258` が 12 件，`:341` が 6 件である．どちらも感情の対ごとの AUC（`SR_S_homeauc`・`SR_S_awayauc`）の対角が NaN であるために出る．元の回の 257・340 行と同じ箇所で，追補の変更で行の番号が 1 つずれた．`dai17_k10_units.log` 120 行（すべて「行 796」），`_reweight.log` 60 行（「行 380」），`_students.log` 120 行（「行 298」），`_summary.log` 3 行，posthoc の 2 本は各 1 行．「済み（飛ばす）」の行は 0（途中で止まって流し直した跡は無い）．例外，エラー，収束しない警告，メモリ不足の行はどのログにも無い．

## 5．抜き取りの作り直しと数え直し

| 対象 | やり方 | 結果 |
|---|---|---|
| DAI-16 `levels.csv` 5 行（MELD `M_F6_orc` adj，AFEW `M_F3_K` kb，IEMOCAP `none_AVT`，MOSEI `prod` 候補，AFEW `M_F6_mixsel` ob） | 単位の npz の判定とラベルから，3 つの T の正解の数を平均 → 感情ごとの再現率の平均 → 種 → 分割（集計の関数を使わない） | 差の最大 7.1e-15 |
| DAI-16 `effects.csv` 4 行の点推定（adj，ob，候補，kb） | 同上の差 | 差の最大 3.6e-15．全 2,396 行の点推定は `levels.csv` の 2 行の差と一致（差の最大 3.0e-14）．判定の語の作り直しの不一致 0 |
| DAI-16 `mixsel.csv` 20 行 | 単位の csv から T = c の 30 単位の adj の平均の最大（同点は並びで先）を選び直す | 選んだ点・n0・β・戻し先・λ・平均の不一致 0．選ぶ単位に c が入る単位 0．最大で同点になった候補は，どの組でも 120 単位の UAR がすべて同じ点（中身が同じ点）どうしだった（中身の違う点の間の同点 0．件数は選んだ点を示すので書かない） |
| DAI-16 選んだ点を借りる版 | `levels.csv` の `*_mixsel` と借りる版の行を，選んだ格子の点の行と比べる | 5 つの層で不一致 0．`mixsel_units.csv` の `chosen` は `mixsel.csv` と全行一致，`uar_adj` は単位の csv と差 0 |
| DAI-16 `criteria.csv`・`r0.csv`・`r0_count.csv` | 差の表と単位の `_check` から数え直す．順位の量は Tukey の当てはめを自分で書いて npz から作る | 不一致 0（118 行，32 行，8 行）．順位の量の差の最大 1.1e-16．`main` 49 行，`proceed` 15 行 |
| DAI-17 `levels.csv` 8 行（5 つの層，`prod`・選ぶ版・連動する版・格子の点の版・`MN_F2K_sel`，4 コーパス） | 単位の npz の判定から独立に数え直す | 差の最大 7.1e-15．どの行も 10 (分割, 種) × 3 T |
| DAI-17 `effects.csv` 6 行 | 点推定は独立に，区間は共通の関数に同じ乱数の鍵 `row_rng("A", c, K, a, b, 層)` を渡して作り直す | 点推定の差の最大 5.6e-17，区間の端の差の最大 1.1e-16 |
| DAI-17 判定の列（全行） | `lo`・`hi`・`est` から作り直す | 不一致 0 / 1,680，0 / 168，0 / 272 |
| DAI-17 `sel.csv`・`frozen_sel.json` | 単位の csv の `uar_adj` で T = c の最大を選び直す | 不一致 0 / 44，`n_tied` の不一致 0，`at_min_lam`・`at_max_lam` の不一致 0，`frozen_sel.json` との不一致 0．選ぶ単位に c が入る単位 0 |
| DAI-17 `students_sel.csv`・`reweight_sel.csv` | 格子の行の `uar` で T = c の最大を選び直す | 不一致 0 / 96（`n_tied` も 0），0 / 48．各点の平均に入る単位はどれも 30 |
| DAI-17 `students_levels.csv` の `_sel` 6 行 | 選んだ格子の点の判定から独立に数え直す | 差の最大 7.1e-15．選んだ点の確率の鍵の欠け 0 |
| DAI-17 数えの表 3 つ | 差の表の判定から数え直す | 不一致 0 / 75，0 / 42，0 / 68 |
| DAI-17 依存の表（設計 3.9 節．`dai17_posthoc.py` の出力） | `dep_k10.py` で独立に作り直し，`posthoc/dependency_detail.csv`・`dependency_count.csv` と突き合わせる | 片方だけの行 0，`changed`・元の点・選び直した点・使った単位の数の不一致 0，`n_changed` の不一致 0．30 単位で選び直すと `frozen_sel.json` と同じ点になる組 60 / 60．使った単位はどの組も 20．変わった組は 22 / 60（記録 2） |

## 6．K に依らない量の一致（追補 2 節．基準 1e-9 以内）

K = 10 の回と K = 5・25 の回の同じ名前のファイルを突き合わせた（`cmp_kindep.py`）．置き場ごとのファイルの名前の集合はすべて同じだった．

| 置き場 | 突き合わせたもの | 差の最大 | 不一致 |
|---|---|---|---|
| DAI-16 `units/` | npz の K を名前に含まない鍵 3,840 個（ラベル・塊・sample_id，ソースの分類器の確率 `p_*_src_*`，信頼度の表 `SR_*` など） | 0 | 0 |
| DAI-16 `summary/` | `reliability.csv` の K に依らない 48 行（`M_home`・`M_ext`・`M_away`・`F8`），`reliability_src.csv` 48 行，`ret_emo.csv` 48 行 | 0 | 0 |
| DAI-17 `units/` | npz の K を名前に含まない鍵 9,960 個（近い発話の番号と値 `nn_*`・`nnsim_*`・`nncos_*`，近いソースの発話の答えの確率 `p_*_knn_*`，ソースの分類器の確率など），csv の `_nn` の行 70,800 行（K = −1） | 0 | 0 |
| DAI-17 `srcdb/` | 24 ファイルの全鍵 336 個 | 0 | 0 |
| DAI-17 `reweight/` | npz の K に依らない鍵 4,020 個（K = 0・0ob の判定と確率，出発点の W・b，`hub_*` など），csv の K に依らない行 1,800 行 | 0 | 0 |
| DAI-17 `students/` | npz の K に依らない鍵 360 個（`y_test`・`cl_test`・`sid_test`） | 0 | 0 |
| DAI-17 `summary/` | `neighbors.csv` 2,440 行，`neighbors_by_source.csv` 240 行，`neighbors_rules.csv` 360 行，`neighbors_count.csv` 90 行，`reweight_weights.csv` 36 行，reweight の K = 0・0ob の行 186 行（levels 96，effects 72，criteria 18） | 0 | 0 |
| DAI-17 `posthoc/` | `overlap.csv` 24 行，`same_actor.csv` 20 行，`same_actor_units.csv` 600 行，`by_source_bal.csv` 120 行，`speakers_top20.csv` 160 行 | 0 | 0 |

- K に依存する行と鍵については，識別子の集合が K = 5 の行と同じかを見た．単位・組・集計・posthoc の全ファイルで同じだった．ただし `sel_best_used.csv` は行が「使われた版」ごとなので，行数は K で違う（設計どおり）．
- 近い発話の記述（D1〜D3，`neighbors*.csv`，posthoc の重なり・同じ俳優の対・ソースごとの当たる割合・話者の数）は差 0 で一致した．追補 2 節のとおり，結果の文ではこれらを K = 10 の新しい結果として数えず，一致の検算にだけ使える．

## 指摘

指摘なし（直すもの）．

## 記録する（数値は変わらない．読み方と書き方に関わる）

1. **[I9] DAI-16 の単位の段と集計の段でコミットが違う．**単位の段は 0c14e4e，集計の段と DAI-17 の 4 段と posthoc は dff7b95 で流れた．計算の途中で別の作業（DAI-18 の設計）が 3 回コミットした（`cac0bb4` 08:01，`b3e5ed4` 08:12，`dff7b95` 08:12）．差分は `DAI/checks/DAI-18/`，DAI-18 の設計の文書，`docs/seen-data.md` の 1 行だけで，`DAI/exp`・`DAI/run`・`measure`・`features`・追補は変わっていない．台本と部品の sha256 はすべての段で同じである．`dai16_probe.py` の集計の段の「単位のコミットが 1 つか」の確かめ（単位の csv を見る）は通っている．数値には影響しない．結果の文の冒頭の申告に，両方のコミット番号と「差分は台本に触れない」ことを書く．なお，`DAI/exp/dai18_probe.py` は 10:23 に 84f65be でコミットされた．`dai16_posthoc.py`（10:20:56．未追跡のファイルも dirty に数える `dai1.git_commit`）は dirty を偽と記録したので，その時点で `DAI/exp` に未追跡のファイルは無かった．`dai17_posthoc.py`（10:21:16）の dirty の確かめは未追跡のファイルを数えない（`dai17_posthoc.py:167`）が，posthoc の 2 本は `dai18_probe.py` を読み込まないので，結果には影響しない．
2. **[S6・L3] 依存の表（設計 3.9 節）で，K = 10 では 60 組のうち 22 組の c′ の点が，c を仮のターゲットにした単位を除くと変わる．**評価するコーパスごとには AFEW 9，IEMOCAP 9，MELD 1，MOSEI 3，疑似ラベルごとには `MN_eq_sel` 5，`MN_F2K_sel` 6，`MN_F6K_sel` 5，`MS_eq_sel` 4，`MS_eq_tied` 2 である．R5 の 3 つの疑似ラベルの 12 の (コーパス, 疑似ラベル) のうち 9 つに変わった c′ があり，変わった c′ が無いのは (MELD, `MN_eq_sel`)，(MELD, `MN_F2K_sel`)，(MOSEI, `MN_eq_sel`) の 3 つである．K = 5 では 12 のすべてに変わった c′ があった（`DAI/checks/DAI-17/code_post.md` 指摘 1）．設計 3.9 節の規則（変わった組のあるコーパスで R5 が「上回る」になったら結果の文にそのことを書く）を，K = 10 ではこの 9 つに当てる．追補 2 節のとおり，この数の K の間の違いを「K による違い」として書かない．依存を除いた R5 は保存の出力からは出せない（元の回の記録と同じ）．
3. **[設計 6 節・追補 3 節] 検算の量の最大と最小は K = 5・25 の回と違う値である．**DAI-16 の `teacher_dev_max` は 7.55e-5（K = 5・25 の回は 7.0e-5），DAI-17 の `prop_agree_min` は 0.99863（同 0.99946．1 未満の 3 行は 4 節の表に挙げた 3 単位），`none_agree_min` は 0.99965（同 0.99973）．どれも許しの中である．結果の文の検算の量は，この報告の 4 節の K = 10 の値を書く．
4. **[区間・自由度] DAI-16 の `folds_dropped`（1,170 行）と MOSEI の行の `df` の NaN は，元の回の記録 1・2 と同じ現れ方である．**DAI-16 の記録 1（AFEW の jackknife の行は分割 0，IEMOCAP の行は 5 つの分割のすべてで種の項が除かれる）と DAI-17 の指摘 2（MOSEI は引き直しで自由度なし）の書き方を，K = 10 の表にもそのまま当てる．追補 2 節は K = 10 で標準誤差と自由度を並べるとしているので，MOSEI の自由度の欄には「引き直し（自由度なし）」と書く．

当てた項目：計算の後の点検の 5 項目（格子のそろい・重複，NaN・Inf・あり得ない値，関門 1 で決めた出力の書き出し，コミットと sha256，ログ），追補 2 節の「表を作る前の確かめ」と「K に依らない量の一致」，`code_v2.md` の I9（各段の sha256 とコミットの照合，既存の置き場の書き換え），S6・L3（記録 2）．

## 目録に無い種類の不備

無し．ただし記録 1 の型（計算の途中で別の作業がコミットし，段ごとにコミットが変わる）は今後も起きうる．`docs/methods.md` 9.3 節の I9 に「長い計算の途中のコミットで段ごとのコミットが変わったときは，差分が台本・部品・入力に触れないことを計算の後の点検で確かめ，結果の文に両方のコミットを書く」を足す案を出す．
