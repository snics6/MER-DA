# 計算の後の点検：DAI-7 の下調べの正の計算（2026-11-08，結果のコミット 1c9f7b2，検査した台本 d0502d6）

## 判定
出力は約束どおりで，直しは要らない．設計 5 節の出力はすべてあり，余りのファイルは無い．どの表も行の数が設計の格子から数えた期待の数と一致し，鍵の重複と欠けは 0 である．NaN はすべて設計で決まる場所にだけある．保存した `rows/` と `fits/` から集計の出力（cells，uar_cells，diffs の点推定と区間，criteria）を台本の関数を使わずに作り直すと，不一致は 0 である．criteria.csv の文の列 6 本は，規則から作り直した文と 4 コーパスのすべてで一致した．単位の段と集計の段は同じコミット 1c9f7b2 で流れ，どちらも dirty は偽である．台本 `DAI/exp/dai7_probe_v2.py` は d0502d6 と 1c9f7b2 と今の作業木で同じファイルである．ログの PerformanceWarning は結果の値を変えていない．記録の項目が 3 つある（下の「指摘」）．どれも結果の数値に関わらない．

点検の範囲と決まり：
- `results/dai7_probe_v2/` は読むだけにした．点検の後に，その下でログ `logs/dai7_probe_v2.log`（01:11:27）より新しいファイルは 0 個，`logs/dai7_probe_v2.done` より新しいファイルも 0 個である．
- GPU は使っていない（すべての検査を `CUDA_VISIBLE_DEVICES=""` で流した）．Python は `.venv/bin/python -I` である．
- 検査の台本とログは `DAI/checks/DAI-7/code_run_post/` に置いた（`chk_struct`，`chk_nan`，`chk_rebuild`，`chk_rows_fits`，`chk_criteria`，`chk_reproduce`，`chk_tether_log` の `.py` と `.log`）．
- この報告とログには，判定の値，ΔD・φ・L・G・S・UAR の値，判定の当否と区分の分布，criteria.csv の文の列の中身と長さを書いていない．差は桁だけを書いた．
- **申告**：`chk_nan.py` の 1 回目の実行で，criteria.csv の H4 の K ごとの文字列の列（`H4_largest_by_K` など 4 列）の列ごとの NaN の数が画面に出た．この数は「前提 (iv) を満たす K が 1 つも無いコーパスの数」を表す．台本を直して流し直し，保存したログ `chk_nan.log` にはこの数も criteria の NaN の合計も残していない．この報告にも書かない．

## 1．出力の一覧と provenance（`chk_struct.log`）

| 項目 | 結果 |
|---|---|
| 上の階の 9 個（rows/，fits/，cells.csv，uar_cells.csv，diffs.csv，criteria.csv，tether_weakening.csv，tether_weakening_units.csv，provenance.json） | 9 個ともある．余りは 0 |
| `rows/` | 240 個（120 単位 × units と uar）．欠け 0，余り 0 |
| `fits/` | 120 個．欠け 0，余り 0 |
| 入力の単位（`results/dai1/pseudo/units/*_p3.npz`） | 120 個で，期待の 120 単位（評価のターゲット T 4 × 仮のターゲット c 3 × 分割 5 × 種 2）と同じ集合 |
| provenance の runs | 1 回だけ（単位の段を途中で流し直していない）．commit 1c9f7b2，dirty 偽，argv は `--out results/dai7_probe_v2` だけ，gpu は cpu |
| ソフトの版 | Python 3.12.3，torch 2.11.0+cu126，numpy 2.5.2，pandas 3.0.6，scipy 1.18.1，CUDA 12.6．ログの 1 行目（起動の台本が書いた scipy と pandas の版）と同じ |
| 定数 | n_null 200，n_null_pool 50，n_null_h 50，n_boot 1000，min_n 15，min_cell 3，min_half 5，min_share 0.3，floor_d 0.02，floor_l 0.5，null_ratio 0.1，h_max 0.5，r_share 0.25，r_share_hi 0.5，r_half 0.5，neff_jk 20，part_order，vkind，eps_d，eps_lda．設計の 3〜4 節の値と同じ |
| 乱数の種 | unit_seed（単位の名前から作る式），perm_rng（共通の乱数の作り方），boot_seed 0 |
| 入力の sha256 | p1〜p3 の 360 個がすべて今のファイルと一致する．固定のファイルの sha256 は今のファイルとも，`git show 1c9f7b2:DAI/frozen/dai1_frozen.json` とも一致する．360 個の入力の commit はすべて固定のファイルの `pseudo_commit` と同じ |
| 台本の sha256 | provenance に鍵は無い（[記録 1]）．`git show 1c9f7b2:DAI/exp/dai7_probe_v2.py`，`git show d0502d6:…`，今の作業木の 3 つが同じ sha256（先頭 9bd8b51a1c021f44）．`git diff d0502d6 1c9f7b2` は起動の台本 `DAI/run/dai7_probe_v2.sh` を足しただけで，ほかのファイルを変えていない．1c9f7b2 の後のコミット 02d6eaa は `docs/seen-data.md` だけを変えた |
| 単位の段と集計の段 | rows/ と fits/ の commit はすべて 1c9f7b2，dirty はすべて偽．集計の出力 6 本の commit と summarize_commit はどちらも 1c9f7b2，summarize_dirty は偽，boot_seed 0，n_boot 1000．2 つの段は同じ起動の台本の中で続けて `.venv` の同じ Python で流れた |
| 所要 | 単位の段の合計 64.7 分（23:54:42 開始，最後の fits が 00:59），集計の段は約 12 分（01:11:27 終了） |

## 2．行の数，鍵の重複，欠け（`chk_struct.log`）

期待の数は，台本と別に，設計 3.2 節の組（3 つのモダリティを持つ組 13，`tt_single_X` 3 は 2 つのモダリティだけ，引き戻しの格子の none_tw 14 は K ≥ 1 だけ），対 3（と all），K 5，単位 120 から数えた．

| ファイル | 行 | 期待 | 鍵の重複 | 欠け・余り |
|---|---|---|---|---|
| `rows/*_units.csv`（合わせて） | 45,360 | 45,360（単位ごとに 378） | 0 | 0・0 |
| `rows/*_uar.csv`（合わせて） | 7,800 | 7,800（単位ごとに 65） | 0 | 0・0 |
| `cells.csv` | 2,056 | 2,056 | 0 | 0・0 |
| `uar_cells.csv` | 260 | 260 | 0 | 0・0 |
| `diffs.csv` | 3,105 | 3,105 | 0 | 0・0 |
| `criteria.csv` | 4 | 4（コーパス 4） | — | — |
| `tether_weakening.csv` | 48 | 48（c 4 × K 4 × モダリティ 3） | 0 | 0 |
| `tether_weakening_units.csv` | 1,440 | 1,440（単位 120 × K 4 × モダリティ 3） | 0 | 0 |

- `cells.csv` の `n_units` は，すべてのセルで 30 である（3 つの T × 5 分割 × 2 種）．
- `diffs.csv` の期待の鍵（c，K，variant，pair，quantity，感情）は，設計 4 節と 5 節から数えた．main は (c, K) ごとに 93 行，jk は実効の塊の数が 20 未満の AFEW・IEMOCAP・MELD で 3 行，変種は 1 つにつき 16 行（AFEW は cent・keep・sub_asr_ok，IEMOCAP は cent・keep・scen・sub_marker・sub_nomarker・spk，MELD は cent・keep・sub_face_single，MOSEI は cent・keep・chan）である．実効の塊の数を fits の塊から数え直すと 14.99，4.91，8.59，783.22 で，設計 3.3 節の値と同じである．
- `fits/*.npz`：120 個のすべてで，配列の鍵の集合が設計 5 節から数えた期待の集合（点数 z，判定，prod・blk・full・各モダリティだけの判別分析の判定，候補の上の点数を 9 組，教師の点数，K 本の位置，テストと候補の sample_id・感情・塊・グループ，交差の当てはめの半分，変種の印，IEMOCAP の筋書き）と同じである．点数の形はすべて（テストの本数，9），点数・候補の上の点数・教師の点数に有限でない値は 0，fits の commit はすべて 1c9f7b2．
- 範囲：UAR が 0〜100 の外のセル 0，再現率が 0〜1 の外 0，d_e が 0〜1 の外 0，φ が −1〜1 の外 0．Inf はどの表でも 0．
- 区間の有効な回：jk 以外の区間のある行は，すべて 1,000 回が有効である．jk の行の回の数は塊の数（AFEW 51，IEMOCAP 5，MELD 272）と同じで，自由度は 13，3，7 である．

### NaN の出どころ（`chk_nan.log`）

すべての NaN が設計で決まる場所にあり，予定に無い NaN は 0 である．

| ファイル | NaN のある場所 | 設計の根拠 |
|---|---|---|
| rows/*_units.csv | (a) none_tw の行の，作らない列（候補の上の D，中心化，教師を除いた D，変種）．作る列（D，φ，d_e，d0_e，本数）に NaN は 0．(b) その c に無い変種の列（例：AFEW の行の scen と chan）．区間を作らない組の行では cent 以外の変種の列．(c) IEMOCAP 以外の `n_dropped_scen` | 3.2 節（none_tw はテストの D と φ だけ），3.3 節（変種はコーパスごと，区間を作る組だけ） |
| 同上 | 全部のテストでは，感情の本数が 15 本未満の (行, 感情) は 0 で，D・φ が NaN の行も 0．部分集合の版で使える感情の数は IEMOCAP の sub_nomarker だけが 3 か 4 で，ほかはすべて 4．変種の D が NaN の行は 0 | 3.3 節の部分集合の版の感情（第 7 版．b5） |
| rows/*_uar.csv，uar_cells.csv，tether の 2 本 | 0 | — |
| cells.csv | none_tw の行の D_pool・D_cent・D_partial_t（4 コーパス × 224），ほかのコーパスの変種の列，区間を作らない組の cent 以外の変種の列 | 上と同じ |
| diffs.csv | lo・hi は G_joint（区間を作らない記述）と dDt_joint_a1_none（記述だけ）の 40 行だけ．est の NaN は 0．d_null_diff は ΔD 以外の量と感情ごとの行だけ（感情ごとの行を除く ΔD の行で NaN は 0）．emotion・n_test_5folds は感情ごとの 160 行以外，G_A〜G_max は G の 60 行以外，n_eff・df は jk の 45 行以外 | 3.3・3.5・4 節と 5 節の列の説明 |
| criteria.csv | MOSEI 以外の `chan_n_K_robust`・`chan_share_n_K_robust`（6），MOSEI の `jk_*`（6．MOSEI は実効の塊の数が 783 で jackknife を付けない），H4 の K ごとの文字列の列（前提 (iv) を満たす K が無いときに空の文字列になり，CSV で NaN として読まれる．NaN の場所が「前提 (iv) を満たす K の数が 0」と一致しない数は 0）．文の列 H1_wording・H4_wording の NaN は 0 | 4 節の MOSEI の漏れ，3.3 節の jackknife，4 節の H4 |

## 3．作り直しの照合（`chk_rebuild.log`，`chk_rows_fits.log`，`chk_reproduce.log`，`chk_tether_log.log`）

台本の関数は呼ばずに書き直した．区間は，台本の (塊, 感情) ごとの和の作り方ではなく，発話ごとの重みで重みつきの共分散・UAR・φ を作る別の書き方で作り直した．

| 照合 | 比べた数 | 不一致 | 最大の差の桁 |
|---|---|---|---|
| cells.csv の量の列 26 本の全部を rows/*_units.csv の平均から（`D_e*` は d_e − d0_e で作り直した） | 53,456 | 0 | < 1e-16 |
| uar_cells.csv の量の列 51 本の全部を rows/*_uar.csv の平均から | 13,260 | 0 | < 1e-13 |
| rows/*_uar.csv の L_dep = full − blk，L_cal = blk − prod | 15,600 | 0 | < 1e-13 |
| diffs.csv の 3,105 行の est と，G の行の G_A〜G_max（cells と uar_cells から） | 3,525 | 0 | < 1e-12 |
| diffs.csv の d_null_diff（ΔD の行） | 1,725 | 0 | < 1e-15 |
| 恒等式 S = Scal + S1b + dLdep + dLcal と S1 = Scal + S1b（joint_a1 と ρ = 1．各 20 セル） | 80 | 0 | < 1e-15 |
| diffs.csv の n_test_5folds（fits の本数から） | 160 | 0 | 0 |
| diffs.csv の区間（lo と hi）：main を IEMOCAP K=5，AFEW K=10，MELD K=1，MOSEI K=25 で 7 量ずつ（ΔD 3 つ，D(none)，Δφ，ΔL_dep，G），変種を IEMOCAP の cent K=25・scen K=10・spk K=1，AFEW の keep K=5，MOSEI の chan K=10，MELD の sub_face_single K=25 で ΔD 3 つずつ | 46 区間（92 端） | 0 | < 1e-15 |
| 上の 46 区間の有効な回の数 | 46 | 0 | — |
| jackknife と t の区間（AFEW K=1，IEMOCAP K=5，MELD K=25 の 3 量ずつ）と，回の数と自由度 | 9 区間（18 端） | 0 | < 1e-16 |
| rows/*_uar.csv の UAR 70,200 個と再現率 280,800 個を fits の判定とテストの感情から | 351,000 | 0 | < 1e-14 |
| rows/*_units.csv の d_e を fits の点数から（fits の点数は float32 なので許す差を 1e-4 にした） | 93,600 | 0 | < 1e-7 |
| rows/*_units.csv の D = d_e − d0_e の平均と φ（fits の判定から） | 46,800 | 0 | < 1e-15 |
| tether_weakening_units.csv（p3 の引き戻し，固定のファイルの ρ，fits の候補の本数から）と tether_weakening.csv（その平均） | 8,928 | 0 | — |

- **保存済みの値の再現**：実験 1 の仮のターゲットの回が `results/dai1/pseudo/units/*_p1.csv`・`*_p3.csv` に保存したテストの UAR と，この計算が保存した W・b を特徴に当て直して出した UAR を突き合わせた．none（K ≥ 1，選んだ引き戻し）1,440 個，oracle 360 個，joint_a1 1,800 個，joint_a1[ρ = 1] 1,800 個，tt_ownret 1,800 個のすべてで差は 0 である．特徴の読み込みと W・b の取り出しが実験 1 と同じであることを示す．
- **共通の乱数と同じ問の確かめ**：joint と joint_a1 の点数は 600 個の (単位, K) のすべてで同じで（設計 3.2 節の「固定のファイルの実際」），その d⁰ も 1,800 個の (単位, K, 対) のすべてで同じである．ret と tt[fused_AVT]（同じ最適化の問）の点数の相対の差の最大は 1e-4 未満である．

## 4．criteria.csv の作り直し（`chk_criteria.log`）

設計 4 節の規則（H1，H1(joint)，H1′，H1_AV，Δφ，共有の比べ，H2，H3，前提 (iv)，H4，jackknife で合否が変わる K，MOSEI の漏れと塊の水準の確かめ，ρ が 1 でないセルの数，全体の数と IEMOCAP 即興を除いた数）を diffs.csv・cells.csv・uar_cells.csv・tether_weakening_units.csv から台本と別に書き直した．H2 の順位相関は scipy の `spearmanr` で作った（台本は pandas）．

- 列：criteria.csv のメタを除いた 104 列と，作り直した 104 列は同じ集合である．
- 数と真偽の列（98 列 × 4 コーパス）：一致 392，不一致 0．
- 文の列：`H1_wording` 4 / 4，`H4_wording` 4 / 4，`H4_largest_by_K` 4 / 4，`H4_share_by_K` 4 / 4，`H4_rho1_largest_by_K` 4 / 4，`H4_rho1_share_by_K` 4 / 4 が一致した．

## 5．ログ（`chk_tether_log.log`）

- ログは 130 行で，単位の行が 120 本，開始と終了の行がある．例外，Traceback，収束しない警告，NaN・Inf の警告，メモリ不足，やり直しの行は無い．警告らしい行は 123 行目の PerformanceWarning の 1 つ（とその続きの 124 行目）だけである．
- **PerformanceWarning は結果に影響しない**．警告は台本 712 行 `U[f"D_e{e}"] = U[f"d_e{e}"] - U[f"d0_e{e}"]` が，多くの列を持つ DataFrame に列を 1 本ずつ足すときに pandas が出す，速さについての警告である．足す値は d_e − d0_e の引き算で，警告は値を変えない．実際に，`D_e*` を別の書き方（`pd.concat` で列をまとめて作る）で作り直した cells.csv の `D_e0`〜`D_e3` と，それを使う diffs.csv の感情ごとの 160 行の est は，上の 3 節の照合で不一致 0 である．計算の前の試走（`code_v5.md`）にも同じ警告が 1 つあった．

## 指摘

### [記録 1] provenance に台本の sha256 と集計の段の環境が書かれていない
- 重さ：記録する
- 該当箇所：`DAI/exp/dai7_probe_v2.py` 1087〜1094 行（provenance は単位の段だけが書く），959〜962 行（集計の段は CSV の列に commit と dirty だけを書く）．
- 問題：provenance の鍵に台本の sha256 が無い．集計の段は provenance を書かず，集計の段の Python とライブラリの版は記録に無い．今回は，commit 1c9f7b2 で dirty が偽，1c9f7b2 と d0502d6 と今の作業木の台本が同じ sha256，2 つの段が同じ起動の台本の中で同じ `.venv` で続けて流れ，起動の台本がログの 1 行目に scipy と pandas の版を書いたので，台本と環境は特定できる．
- 計算の後で直せるか：直せる（記録を足すだけ）．
- 求める対応：今回の結果には要らない．次の台本では，provenance に `script_sha256` を足し，集計の段も版と commit を provenance に追記する．

### [記録 2] 特徴のファイル（`data/cache/*.npz`）の sha256 が provenance に無い
- 重さ：記録する
- 該当箇所：`DAI/exp/dai7_probe_v2.py` 344 行（`dai1.Unit` が特徴を読む），1098 行（sha256 は p1〜p3 だけ）．
- 問題：点数は保存した W・b を特徴に当てて作り直すので，特徴が実験 1 の時点と違えば点数も違う．provenance からはこれを確かめられない．今回は，実験 1 が保存したテストの UAR 7,200 個をこの計算の UAR が差 0 で再現したので（3 節），読み込んだ特徴は実験 1 と同じとみなせる．
- 計算の後で直せるか：直せる（上の再現の照合で代えた）．
- 求める対応：今回の結果には要らない．次の台本では，読んだ特徴のファイルの sha256 を provenance に書く．

### [記録 3] 作業木の dirty の確かめの範囲に `DAI/run` と `DAI/frozen` が入っていない
- 重さ：記録する
- 該当箇所：`DAI/exp/dai1.py` 72〜73 行（`git status --porcelain` を `DAI/exp`，`DAI/splits`，`measure`，`features`，`data/metadata` に限る）．
- 問題：起動の台本と固定のファイルに未コミットの変更があっても dirty は偽になる．今回は，起動の台本はコミット 1c9f7b2（計算の起動と同じ時刻）にあり，その後に変わっていない．固定のファイルは sha256 が `git show 1c9f7b2:DAI/frozen/dai1_frozen.json` と一致する．今の `git status` でも 7 つの場所のどれにも変更は無い．
- 計算の後で直せるか：直せる（今回は上の照合で確かめた）．
- 求める対応：今回の結果には要らない．`dai1.git_commit` の範囲に `DAI/run` と `DAI/frozen` を足すかは，次に `dai1.py` を直すときに決める．

当てた項目：9.3 節の I（I1-a の抜き直しの外し方，I7 の共通の乱数，I9 の集計の段のコミット，I11 の計画の外の NaN・Inf，I12 の dirty）と，計算の後の点検の 5 項目（格子のそろい，NaN・Inf・あり得ない値，足すと決めた出力，コミット番号，ログ）．
