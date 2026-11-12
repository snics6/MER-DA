# 計算の後の点検：実装と出力（DAI-15，2026-11-13，コミット 14e99f0．台本は a5a6a6f と同じ中身）

点検した出力：`results/dai15/`（`pseudo/units` 120 単位，`eval/units` 200 単位，`summary/`，`provenance.json`，`provenance_log.jsonl`，`*.done`），`DAI/frozen/dai15_frozen.json`，ログ `logs/dai15_{pseudo,select,eval,summary}.log` と `logs/dai15_run.sh`．
検算の台本は `DAI/checks/DAI-15/code_run_post/` の 3 本（`check_units.py`，`check_npz.py`，`check_summary.py`）である．どれも CPU で流し，`results/` には書いていない．
設計 2 節の規則に従い，この報告には予言の判定，UAR とその差，`select_means*.csv` と固定のファイルの `pseudo_uar_*` の値を写さない．件数，一致・不一致，検算の量だけを書く．

## 判定
通す．単位は仮のターゲット 120・評価 200 がすべてそろい重複は 0，全単位が 1 つのコミットの汚れていない作業木で作られ，4 つの段の台本と設計の sha256 がすべて a5a6a6f の版と一致し，effects の 1,400 行の点推定・per_emotion の 2,304 行・hypotheses の 50 欄・criteria の 210 行を出力から独立に作り直して食い違いは 0 だった．

## 1．単位の欠けと重複，コミット

| 確かめたこと | 仮のターゲットの回（pseudo） | 評価の回（eval） |
|---|---|---|
| 期待する単位（`dai3.pseudo_units`／分割 5 × `n_seeds_eval`） | 120 | 200 |
| csv・npz の数 | 120・120 | 200・200 |
| 欠け・余分・ほかのファイル | 0・0・0 | 0・0・0 |
| ファイル名と T・c・fold・seed の列の食い違い | 0 | 0 |
| 同じ（単位，K，option，生徒）の重複した行 | 0 | 0 |
| 単位あたりの行の数 | すべて 140（16,800 行．固定のファイルの `n_rows` 16,800 と一致） | すべて 140（28,000 行） |
| (K, option) ごとの行の数 = 単位の数 × 期待（19 種の option） | 一致 | 一致 |
| 単位の `commit` | すべて 56afd51 | すべて 14e99f0 |
| 単位の `dirty` | すべて False | すべて False |
| `dai1_commits`（読んだ実験 1 の保存のコミット）の種類 | 1 | 1 |
| 単位の所要の合計 | 2,699 秒（約 45 分） | 4,652 秒（約 78 分） |

56afd51 は a5a6a6f の後に関門の報告 `DAI/checks/DAI-15/code_v6.md` だけを足したコミットで，14e99f0 は固定のファイル `DAI/frozen/dai15_frozen.json` だけを足したコミットである（`git show --stat`）．固定のファイルの `pseudo_commit` は 56afd51 で，仮のターゲットの単位のコミットと一致した．

## 2．行の中身：有限でない値と検算の量

| 確かめたこと | pseudo | eval | 許し・設計 |
|---|---|---|---|
| 版の行（I，0.0，0.5，1.0，naive，oracle，prop，none，partner，kd_one）の uar の有限でない数／0〜100 の外 | 0 / 9,600・0 | 0 / 16,000・0 | — |
| 補正の問の勾配の最大，止まり，目的関数の値の有限でない数 | 1.00e-06，0，0 | 1.00e-06，0，0 | 勾配 10⁻⁶ 以下で止まり 0（仮定 A6） |
| 表の行（`_table`）の有限でない値 | 0 | 0 | — |
| 作り直した疑似ラベルと保存の積の差の最大（teacher_dev） | 7.37e-05 | 7.42e-05 | 10⁻⁴（設計 3 節） |
| 疑似ラベルのモデルの作り直しの勾配の最大，止まり | 1.00e-06，0 | 1.00e-06，0 | — |
| I と提案の重みの相対の差（value）の最大 | 2.32e-04 | 2.04e-04 | 10⁻²（設計 3 節） |
| 同じ差を出発点からの移動で割った値（value_moved）の最大 | 2.90e-04 | 1.82e-04 | 記録 |
| I と提案の判定の入れ替わり（n_flip）：合計，入れ替わりのある（単位，K，生徒）の数，最大 | 2，2 / 960，1 | 6，6 / 1,600，1 | 記録 |
| 割合をそろえる切片の残りの最大 | 1.11e-16 | 1.11e-16 | 10⁻⁴（ADJ_TOL） |
| 補正しない単位（skip，d̄ ≤ 0.25） | K = 5 で 1（MELD），K = 25 で 0 | K = 5 で 2（MOSEI），K = 25 で 0 | 件数を添える（設計 5 節 H3，7 節 A9） |
| 教師の判定し直しの特異（singular） | 0 / 240 | 0 / 400 | 件数を添える（設計 3 節 43 行．前回の [記録-h]） |
| π′ が 0〜1 に入らない（pi_ok が偽）の行 | 504 / 5,760 | 844 / 9,600 | 版ごとの件数を書く（仮定 A8） |
| π′ が有限でない行 | 0 | 0 | — |

npz の点検（`check_npz.py`）：

| 確かめたこと | pseudo | eval |
|---|---|---|
| 鍵の組の種類，鍵の数 | 1，492 | 1，492 |
| 浮動小数の配列の有限でない要素 | 0 | 0 |
| 確率の行の和と 1 の差の最大 | 5.22e-08 | 5.22e-08 |
| 確率の argmax と保存の判定（pred）のずれ | 0 行 | 0 行 |
| csv の uar と npz の判定から作り直した UAR の差の最大 | 7.11e-15 | 7.11e-15 |
| 同じ（T，c，分割）の単位の間でテストの y・塊・sample_id が違う数 | 0（60 組） | 0（20 組） |
| 実験 1 の保存（段 3 の csv）の提案（joint，one，選んだ ρ と a）の UAR との差の最大 | 0（960 組） | 0（1,600 組） |
| 実験 1 の保存の kd_one（kd，one，選んだ温度と ρ）の UAR との差の最大 | 0（960 組） | 0（1,600 組） |
| 実験 1 の固定のファイルの主の版 | 4 コーパスとも one | 同左 |

鍵の数 492 は，設計 6 節の保存（K 2 × 生徒 4 × [判定と確率 16 版 × 2，W・b 6 版 × 2，切片のずれ 6，候補の上の平均 7] ＋ K 2 × [表 3，w，qcf，ykpos，重みに合わせた表 6，教師の確率と判定 2，teacher_w，teacher_rj0] ＋ candcount，y，cl，sid）の数え 456 ＋ 32 ＋ 4 と一致する．

## 3．provenance と，選ぶ段の後の変更（I9・I12．前回からの持ち越し）

`provenance_log.jsonl` は 4 行（pseudo，select，eval，summary の順）で，引数はすべて `--stage <段>` だけ（本番の置き場で対象を絞っていない）である．

| 段 | 時刻 | commit | dirty | 台本 7 本の sha256 = 今のファイル | 設計の sha256 = 今のファイル | 固定のファイルの sha256 | 実験 1 の固定のファイルの sha256 |
|---|---|---|---|---|---|---|---|
| pseudo | 02:18:34 | 56afd51 | False | 一致 | 一致 | （まだ無い） | 6926fb3e… |
| select | 03:03:48 | 56afd51 | False | 一致 | 一致 | （書く前） | 6926fb3e… |
| eval | 03:03:50 | 14e99f0 | False | 一致 | 一致 | 今のファイルと一致 | 6926fb3e… |
| summary | 04:21:50 | 14e99f0 | False | 一致 | 一致 | 今のファイルと一致 | 6926fb3e… |

- 台本 7 本（`dai15.py`，`dai14.py`，`dai3.py`，`dai1.py`，`dai1_stats.py`，`dai_solver.py`，`dai_data.py`）と設計 `DAI/docs/実験15_設計.md` の sha256 は，a5a6a6f の版（`git show a5a6a6f:<path>`）とすべて一致した．`dai15.py` は f6fb459b…，設計は 3a1e126c… で，関門 1 第 6 回で検査した版と同じである．
- `git diff --stat 56afd51 HEAD -- DAI/exp measure features DAI/splits data/metadata DAI/docs/実験15_設計.md` は空で，`a5a6a6f` からの差分も空だった．作業木の同じ範囲と `DAI/frozen` の `git status --porcelain --untracked-files=all` も空である．したがって選ぶ段の後に，台本，共通の部品，`measure/`，設計は変わっていない（設計 2 節の確かめ）．
- 固定のファイルは 03:03:48 に書かれ，14e99f0 で 03:03:49 にコミットされ，評価の段は 03:03:50 に始まった．14e99f0 の中の固定のファイルの sha256 は今のファイルと一致した．`select_means.csv`・`select_means_by_c.csv` の更新時刻は select の段の時刻（03:03:48）のままである．
- `provenance.json` は `provenance_log.jsonl` の最後の行（summary）と同じ中身だった．
- 定数（KS，LAMBDAS，NF，TOL，MAX_ITER，SEL_OPTS，D_MIN，LPI_MIN_SHARE，CHECK_TEACHER，CHECK_W，ADJ_TOL，ADJ_ITERS）は台本の値と同じに記録されている．

## 4．summary の行の数と，出力からの作り直し

effects.csv：

| 確かめたこと | 結果 |
|---|---|
| 行の数 | 1,416（差 768，版の水準 576，記録 lpi 64，教師 8．設計 4 節の数えと一致） |
| （T，K，student，contrast）の重複 | 0 |
| commit，dirty | すべて 14e99f0，False |
| 記録 lpi の行：作った／作らない | 48 / 16（作らないのは MELD 4，MOSEI 12） |
| 記録以外の 1,352 行と作った記録の 48 行の effect・lo・hi の有限でない数 | 0 |
| 作った記録の行の raw_effect・raw_lo・raw_hi の有限でない数，作らない行で raw_* が空でない数 | 0，0 |
| lo ≤ effect ≤ hi を満たさない行 | 0 |
| 判定の語が「上回る・下回る・判定できない・作らない」以外の行，水準の行で判定が空でない数 | 0，0 |
| 区間の作り | jk+seed 1,058 行，boot_cl_shared+seed 342 行（MOSEI），作らない 16 行（計 1,416） |
| 記録以外の行の n_units | すべて 50 |
| 塊の数（n_clusters） | AFEW 51，IEMOCAP 5，MELD 272，MOSEI 2,614（コーパスの中で全行同じ） |
| selected の列が固定のファイルの選んだ版と違う行 | 0 |

作り直し（`check_summary.py`）：

- **点推定（全行）**：dai14 を使わずに，npz の判定から単位ごとの UAR（現れた感情の再現率の平均）を作り，分割ごとに種で平均してから分割で平均した．差 768 行，水準 576 行，教師 8 行，作った記録 48 行（effect と raw_effect）の計 1,400 行で，effects.csv との差の最大は 7.1e-15 だった．記録 lpi の 64 行では，π′ = (C′ᵀ)⁻¹(1/4, …) を逆行列で独立に作り，使った単位の数，使った分割の数，作る・作らないの判定の食い違いが 0 だった．
- **区間（抜き取り）**：4 コーパスのそれぞれで，sel − prop（K = 5，声），sel_adj − prop（K = 5，融合），oracle_adj − λ0_adj（K = 25，顔），λ0_adj − I_adj（K = 5，声），水準 prop（K = 5，言葉），教師の判定し直し（K = 25），λ0_lpi − I（K = 5，声），sel_lpi − prop（K = 5，融合）を，`dai14.combo_stats` と `dai15.row_rng` で npz から作り直した．作らない行（MELD 1，MOSEI 2）を除く 29 行で，effect・lo・hi（記録の行は raw_* も）の差の最大は 7.1e-15，判定の食い違いは 0 だった（水準の行は effects.csv の判定が空なので照合の対象から外した）．MOSEI の 6 行は引き直し 2,000 回の区間で，行ごとの乱数で同じ値が出た．
- **per_emotion.csv**：2,304 行（4 × 2 × 4 × 18 版 × 4 感情）すべてを npz から独立に作り直し，recall・pred_share・true_share の差の最大は 8.3e-17，NaN は 0 だった．
- **hypotheses.csv**：8 行（K 2 × H1・H2・H3・C1）の 50 欄（n，hit，corpora，off，raw_only，adj_only，rejudge，rejudge_any，H3 の 3 つの判定と n_skip_units，C1 の base）を，effects.csv の判定と固定のファイルから設計 5 節と 3 節 43 行の規則で独立に作り直した．食い違いは 0 だった．n_skip_units は評価の単位の csv の skip から数えた値と一致した．
- **criteria.csv**：210 行（K 2 ×［生徒 4 × 差 26 ＋ 教師 1］）の上回る・下回るの数を effects.csv から作り直し，食い違いは 0 だった．

## 5．ログ

- `logs/dai15.finished` は OK，`logs/dai15.fail` は無い．pseudo のログは 120 行，eval のログは 200 行で，すべて「行 140，勾配の最大 ≤ 1.0e-06，止まり 0」だった．「済み（飛ばす）」の行は 0 で，単位の作り直しや途中からのやり直しは無い．例外，収束しない警告，メモリ不足の跡は無い．
- summary のログに `dai14.py:94: RuntimeWarning: Mean of empty slice` が 1 回ある（Python は同じ箇所の警告を 1 回だけ表示する）．出どころを抜き取りで確かめた．警告は IEMOCAP の行だけで出た（8 行で計 878 回．AFEW・MELD・MOSEI は 0 回）．IEMOCAP は単位ごとのテストの塊が 1 つ（5 つの塊が 5 つの分割に 1 つずつ）なので，塊を全分割から同時に除く jackknife で塊を 1 つ除くと，その塊をテストにした分割の単位が空になり，`uar_present` が空の平均を取る．その値は NaN になり，`theta` はその分割を除いて平均する．これは設計 4 節の jackknife の作り（実験 14 と同じ部品）どおりで，effects.csv の IEMOCAP の行の effect・lo・hi はすべて有限だった．数値の誤りではない．

## 指摘

### [記録-k] summary のログの「Mean of empty slice」の警告は IEMOCAP の jackknife の作りから出る
- 重さ：記録する
- 該当箇所：`DAI/exp/dai14.py:94`（`u = 100.0 * np.nanmean(rec, axis=-1)`），`dai14.py:175`（塊を除いた値）
- 問題：IEMOCAP では 1 つの単位のテストが 1 つの塊でできているので，jackknife で塊を除くと単位が空になり警告が出る．値は NaN として扱われ，その分割を除いた平均が jackknife の値になる．抜き取りの 8 行で警告は 878 回，区間の作り直しは保存の値と一致した（差の最大 7.1e-15）．
- 計算の後で直せるか：直す必要は無い．
- 求める対応：結果の文の再現の節か付録で，IEMOCAP の区間は「1 つのセッションの組を全分割から同時に除く jackknife（塊 5）」であり，ログの警告はその作りから出ると書く．

### [記録-l] provenance に，読んだ実験 1 の保存（単位の npz）の sha256 が無い
- 重さ：記録する
- 該当箇所：`DAI/exp/dai15.py:649–655`（`prov` に `dai1_frozen_sha256` はあるが，単位の入力の sha256 は無い），単位の csv の `dai1_commits` 列
- 問題：DAI の取り決め（`DAI/CLAUDE.md`）は provenance に入力の sha256 も残すとしている．DAI-15 は実験 1 の固定のファイルの sha256 と，単位ごとの実験 1 のコミット（`dai1_commits`．全 320 単位で 1 種類）だけを残す．今回は提案と kd_one の UAR が実験 1 の保存と 2,560 組で完全に一致したので，読んだ入力が実験 1 の保存と同じであることは出力から確かめられた．
- 計算の後で直せるか：直せる（今の `results/dai1/` の npz の sha256 を後から記録できる．ただし計算の時点の値であることは上の一致からしか言えない）．
- 求める対応：結果の文の provenance の説明に，入力の同一性は `dai1_commits` と実験 1 の保存の UAR との一致（差 0）で確かめたと書く．

### 前回からの持ち越しの解消
- [記録-h]（特異で置き換えた単位の数）：単位の csv の `singular` を数えた．教師の判定し直しの特異は pseudo 0 / 240，eval 0 / 400 で，置き換えた単位は無い．結果の文に「特異で置き換えた単位は 0」と添える．
- [I9・I12]（eval と summary が select の時と同じ台本か）：3 節のとおり，4 段の台本と設計の sha256 が同じで，56afd51 から HEAD までの `DAI/exp`・`measure`・`features`・`DAI/splits`・`data/metadata`・設計の差分は空だった．解消した．
- [記録-i]（MOSEI の raw_* と主の行の区間は別の引き直し）と [記録-j]（provenance の `rng_seed` は行ごとの種の作り方を書かない）は，結果の文で扱う記録として残る．

当てた項目：計算の後の点検の 5 項目（格子のそろい，有限でない値とあり得ない値，足すと決めた出力の書き出し，コミット番号の照合，ログ），I9，I12．

## この点検で画面に出したもの

- 出したのは，件数，行の数，一致・不一致，検算の量（作り直しの差，重みの差，入れ替わりの数，切片の残り，勾配），コミットと sha256，時刻，所要，ファイルの大きさである．
- 誤って画面に出したものが 2 つある．(1) 評価の単位 `AFEW_AFEW_f0_r0.csv` の先頭の約 400 字を表示し，K = 5 の `_table` の行の dbar_cf，teacher_dev，teacher_grad，teacher_stalled，uar_teacher_full，uar_teacher_fold0・fold1 の値を見た．これは疑似ラベルのモデルの候補の上の UAR と表の対角の平均で，補正の版や提案の UAR，その差，予言の判定ではない．この報告には写していない．(2) 固定のファイルの選んだ版の種類の数（32 セルの選択に現れる候補の種類の数）を 1 つの数として表示した．セルごとの選択と `pseudo_uar_*` は表示していない．
- `select_means*.csv` は開いていない．hypotheses.csv と effects.csv の判定は，作り直した値との一致・不一致だけを表示した．

---

判定：通す
