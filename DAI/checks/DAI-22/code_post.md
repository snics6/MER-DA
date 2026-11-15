# 計算の後の点検（DAI-22，2026-11-15，結果のコミット 59f9e1d，台本 540bba2 の版）

## 判定
判定：通す．200 単位の格子は欠けも重複も無くそろっている．単位の csv のコミットは 59f9e1d の 1 つで，dirty は全行で偽である．provenance の sha256 は台本と設計に一致し，照合が外れた 7 行はすべて設計 6 節の規則どおり same（3 行）か near_tie（4 行）として通った．要約の表に意図しない NaN・Inf は無く，設計 13 節の 5 項目はすべて保存した npz と表から学び直さずに出せた．

点検に使った台本とログ：`checks/DAI-22/code_run_post/post_check.py`・`post_check.log`，`post_check_npz.py`・`post_check_npz.log`．どちらも `results/dai22/` を読むだけで，書き換えていない．画面と報告に出したのは行の数，照合の差，予測の違いの本数，収束，NaN の数，照合の最大差だけで，UAR・Δ・P・一致の割合・κ の値は出していない．

## 1. 来歴（コミット，dirty，sha256）

- 作業木 `/home/snics/MER-DA-wt22` の HEAD は 59f9e1d である．`git diff 540bba2 59f9e1d` で `exp`・`measure`・`features` に差は無く，設計の差は 6 節の 1 行（code_v5 の [記録 3]：「上限に届いたら」を「上限を超えたら」に直したもの）だけである．
- `provenance_log.jsonl` は 2 行（units 21:52:37，summary 23:07:08）で，どちらも commit 59f9e1d，dirty 偽である．1 回目の本番の記録は残っていない（出力とともに消したため）．`provenance.json` は summary の行と同じ中身である．
- 台本 7 本（dai22・dai14・dai3・dai1・dai1_stats・dai_solver・dai_data）の sha256 は，provenance の値，59f9e1d の版，作業木のファイル，今の本体の作業木のファイルの 4 つで一致した．dai22.py は a63c06b8…で，540bba2 の版と同じであり，code_v5 で検査した版である．
- 設計の sha256 fad8165c…は，59f9e1d の版，作業木，今の本体のファイルと一致した．
- 入力の sha256：実験 3 の `repro_units.csv`（fd2f740a…），実験 14 の `summary/conditions.csv`（ca618aa3…）と `provenance.json`（339a9658…）は，今のファイルと一致した．実験 1 の固定のファイルは 6926fb3e…（provenance の値）である．
- 単位の csv の `dai1_commits` は ba98974 の 1 つ，`target` は全行 prop_one である．
- 作業木の `git status` には `DAI/results/` の追跡ファイルが「消えた」と出る．これは `results` を本体へのリンクに替えたためで，台本の dirty の判定（`exp`・`measure`・`features` だけを見る）には入らない．

## 2. 格子と重複

| 点検 | 結果 |
|---|---|
| 単位のファイル | npz 200，csv 200．見込みの 4 コーパス × 5 分割 × 10 種との差は，欠け 0，余り 0（npz と csv の両方）．ほかの拡張子のファイル 0 |
| 単位の csv の行 | 800（見込み 200 × K 2 つ × 読み出し 2 つ）．単位ごとに 4 行．(T, K, 読み出し) ごとに 50 行 |
| 識別子（T, fold, seed, K, readout）の重複 | 0 |
| ファイル名と中身の識別子 | 全部一致 |
| ログ `dai22_units.log` | 207 行＝200 単位の行＋照合が外れた 7 行の説明．「済み（飛ばす）」の行は 0 で，200 単位すべてをこの回で計算した．`warn`・`error`・`traceback`・`memory` の語は units と summary のログに 0 回 |
| 終わりの印 | `units.done`（23:07:06），`summary.done`（23:09:38），`logs/dai22.finished`（OK）．`logs/dai22.fail` と `check_fail_*.csv` は無い |
| `check_saved_20261115-230706.csv` | 400 行（R0 の行），重複 0，`abs_diff` は単位の csv と一致 |
| 所要 | 単位の `sec` の合計 74.1 分（設計の見込み 1.5 時間以内） |
| ディスク | `results/dai22/` は 87 MB（見込み約 90 MB） |

## 3. 収束

- 全 800 行の勾配の最大は 9.998 × 10⁻⁷（基準 10⁻⁶ の内側）．止まり 0．コーパスごとの最大も 9.99 × 10⁻⁷ 前後で，4 つとも 10⁻⁶ 未満である．
- R1 の内側の勾配の最大は 9.999 × 10⁻⁷，内側の止まりの最大は 0．
- 単位の csv の数の列に Inf は 0．R0 の行の NaN は R1 専用の列（C0，C1，inner_*，pca_*，dim_in）と，解き直した行にだけ値が入る列（393 行が NaN＝400 − 7）だけである．
- `convergence.json`：`n_rows` 800，`n_units` 200，`grad_ok`・`inner_grad_ok`・`cat_ok` はすべて真，`C_chosen_counts` の合計 800（200 単位 × K 2 つ × 組 2 つ）．

## 4. 実験 3 の保存との照合が外れた 7 行

照合が 0.001 を超えて外れたのは 400 行のうち 7 行で，超えない 393 行の差の最大は 1.4 × 10⁻¹⁴ である．`abs_diff` の単位は，2 組の交差の予測の J_prop に対する UAR 型の一致の百分率の差である（`dai14.collect`）．

| 単位 | K | 候補の数 | 保存との差 | resolve_kind | 10⁻⁸ の違い | 10⁻¹⁰ の違い | 厳しい 2 解どうし | 本番の解との UAR の差（点） | 勾配 10⁻⁸／10⁻¹⁰ | 止まり 10⁻⁸／10⁻¹⁰ | 反復 10⁻⁸／10⁻¹⁰ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MELD f2 r0 | 5 | 5,946 | 0.0161 | near_tie | 1 | 1 | 0 | −0.0073 | 9.5e-9／9.1e-11 | 0／0 | 243／442 |
| MELD f3 r7 | 5 | 5,777 | 0.0203 | same | 0 | 0 | 0 | 0 | 9.4e-9／9.0e-11 | 0／0 | 272／381 |
| MELD f3 r8 | 5 | 5,777 | 0.0158 | near_tie | 1 | 1 | 0 | 0.0075 | 9.9e-9／7.5e-11 | 0／0 | 308／466 |
| MOSEI f0 r1 | 25 | 14,907 | 0.0073 | same | 0 | 0 | 0 | 0 | 1.0e-8／**2.9e-10** | 0／**1** | 243／319 |
| MOSEI f0 r6 | 25 | 14,907 | 0.0072 | same | 0 | 0 | 0 | 0 | 9.2e-9／9.8e-11 | 0／0 | 266／362 |
| MOSEI f1 r3 | 25 | 14,907 | 0.0076 | near_tie | 2 | 2 | 0 | 0.0093 | 1.0e-8／9.9e-11 | 0／0 | 246／332 |
| MOSEI f4 r5 | 25 | 14,908 | 0.0054 | near_tie | 1 | 1 | 0 | 0.0046 | 1.0e-8／9.5e-11 | 0／0 | 266／366 |

- **種類の作り直し**：csv の列から same と near_tie の条件（厳しい 2 解どうしの違い 0，違いが候補の 0.1% 以下，保存との差 × 候補 ÷ 100 ≤ max(2, 0.001 × 候補)）を計算し直すと，7 行とも csv の `resolve_kind` と同じ種類になった．許す本数は MELD 5.8〜5.9 本，MOSEI 14.9 本で，実際の違いは最大 2 本である．保存との差を本数に換えると 0.80〜1.17 本で，上限（MELD 5.8〜5.9，MOSEI 14.9）の内側である．
- **npz との突き合わせ**：npz の `K<K>__R0_tight` の鍵は 7 個で，外れた行の数と同じである．各行で `K<K>__R0_tight` と `K<K>__R0` の違いの本数は csv の 10⁻⁸ の列と一致した．npz から作り直した UAR の差と csv の `uar_diff_tight_vs_unit` の差は 10⁻¹⁶ の桁である．解き直していない行で `resolve_kind` が埋まった行は 0．
- **コーパスごとの上限**（max(4, 0.25 × 100) = 25 行）：通した行は AFEW 0，IEMOCAP 0，MELD 3，MOSEI 4 で，どのコーパスも上限の内側である．
- **near_tie で予測が違った本数の最大**は 2 本（MOSEI f1 r3），`r0_uar_diff_tight_abs_max` は 0.0093 点である．
- **基準に届かずに止まった解き直し**：MOSEI f0 r1（same）の 10⁻¹⁰ の解き直しが 1 問止まり，その勾配は 2.9 × 10⁻¹⁰ で 10⁻¹⁰ に届いていない．ほかの 6 行の 10⁻⁸・10⁻¹⁰ の解き直しはすべて基準に届いた．same 3 行のうち止まりを含む行は 1 行，near_tie 4 行のうちは 0 行である（下の [記録 2]）．
- MOSEI の 3 行（f0 r1，f0 r6，f1 r3）は 1 回目の本番で外れた 3 行と同じ単位で，差も 0.0073，0.0072，0.0076 と同じである．MOSEI f4 r5 と MELD の 3 行は 1 回目では流れる前に止まった単位である．
- 設計 6 節（第 5 版の記録 6）の「境目の入れ替わりは候補が 1,000 本以上の単位でだけ働く」のとおり，AFEW に外れた行は無く，near_tie は MELD（約 5,800〜5,950 本）と MOSEI で起きた．

## 5. そのほかの照合の記録（単位の csv）

- J_none・J_prop の UAR と実験 1 の `p2.csv`・`p3.csv` の差の最大：none 7.1 × 10⁻¹⁵，prop（p2）7.1 × 10⁻¹⁵，prop（p3）7.1 × 10⁻¹⁵（許し 10⁻⁶）．
- 融合の特徴と 3 つをつないだ特徴の差の最大 5.3 × 10⁻¹⁵（許し 10⁻¹⁰）．
- つないだ線形の判定と J_prop が違う発話は 400 行すべてで 0 本．同点でも切られてもいない発話の違い（`n_linear_diff_nontie_unclamped`）の最大 0，同点の発話の最大 1 本（1 行だけ．その発話でも判定は一致），確率が切られた発話 0．
- 要約の段の `uar_pseudo` と実験 14 の `conditions.csv` の照合を作り直すと，24 行で最大差 0（許し 10⁻⁹）．

## 6. 要約の表

| ファイル | 点検 | 結果 |
|---|---|---|
| `summary/conditions.csv` | 行の数 | 210 ＝（AFEW，IEMOCAP，MOSEI の all ＋ MELD の all と nonshared）の 5 つ × K 2 つ × 量 21 個．(T, K, subset) ごとに 21 行 |
| | 量の名前 | 21 個（`uar_none`，`uar_pseudo`，`delta`，`uar_imitation_R0/R1`，`imitable_R0/R1`，`uar_start`，`start_minus_none`，`R0_minus_start`，`R0_minus_pseudo`，`R1_minus_pseudo`，`delta_minus_delta_A/V/T`，`start_minus_none_A/V/T`，`R0_minus_start_A/V/T`）．設計 3 節と 5 節の量がすべてある |
| | 重複 | 0 |
| | NaN | `v_split`・`v_split_indep` は 210 行すべて NaN（分け方の揺れを足さない設計 4 節どおり．`split_vals` を渡していない）．`df` は 42 行が NaN で，MOSEI の 2 K × 21 量の全部（区間の作りが塊の重みの引き直し `boot_cl_shared+seed` で，自由度を使わない）．ほかの列に NaN は無い |
| | Inf，区間の順序 | Inf 0．`lo ≤ effect ≤ hi` を破る行 0 |
| | 区間の作り | AFEW・IEMOCAP・MELD は `jk+seed`，MOSEI は `boot_cl_shared+seed`（設計 4 節どおり） |
| | コミット | 59f9e1d，dirty 偽 |
| `summary/assumptions.csv` | 行の数 | 141 ＝ C_edge 8，pca 4，n_per_dim 8，nonshared_size 1，linear_identity 8，kappa_pseudo_none 8，kappa_pseudo_none_A/V/T 各 8，agree の 10 種（R0_tgt，R0_start，R1_tgt，R1_start，R0_tgt_A/V/T，R0_start_A/V/T）各 8 |
| | NaN，Inf | 種類ごとに使う列の NaN 0，Inf 0 |
| `summary/convergence.json` | 鍵 | 設計 6 節と 8 節の記録（照合の差の最大 0.0203，外れた行 7，通した行 7，near_tie 4，near_tie の違いの最大 2，UAR の差の最大 0.0093，収束の 3 つの真偽，C′ の数え，所要）がそろう |

## 7. 単位の npz の中身

- 200 個すべてで，設計 10 節の鍵（y，cl，sid，half，nonshared，marker，sex，pca，start と，K ごとの nk，none，tgt，maxp，clamped，R0，prob R0，R1，prob R1，inner）がそろい，`K<K>__R0_tight` は外れた 7 行の単位にだけある．
- 浮動の配列の NaN 0，Inf 0．発話の数と配列の行の数の食い違い 0．交差の予測の確率の和が 1 から 10⁻⁴ 以上ずれる発話 0．ラベルと判定が 0〜3 の外にある単位 0．`nonshared` の数は csv の `n_nonshared` と全単位で一致．

## 8. 設計 13 節の「計算の後に保存から出すもの」

5 項目とも学び直さずに出せることを，実際に計算して確かめた（値は出していない）．

| 13 節の項目 | 使うもの | 確かめたこと |
|---|---|---|
| 出発点と J_prop の一致（4 つの行） | 融合：dai22 の npz の `start`・`tgt__K<K>`・`nk__K<K>`．声・顔・言葉：実験 14 の npz の `start__<s>`・`tgt__K<K>__prop_one__true` | 200 単位 × K 2 つ × 4 行で有限に出た．候補の並びは要約の段で実験 14 と照合済み |
| 本当の感情ごとの読み出しと J_prop の不一致の割合 | dai22 の npz の `y`・`K<K>__R0`・`K<K>__R1`・`tgt__K<K>`・`nk__K<K>` | 単位 × K ごとに（読み出し 2 × 感情 4）が有限に出た |
| MELD の共有を除いた候補（K 本を除く）の塊の実効の数 | MELD の npz の `cl`・`nonshared`・`nk__K<K>` | 50 単位 × K 2 つで出た．範囲 6.1〜8.5（全部の候補の塊の実効の数 `g_eff_of("MELD")` は 8.6．どちらも 20 未満で，区間の作りの分かれ目の同じ側にある） |
| 声・顔・言葉の行の `start_minus_none_<s>` ＋ `R0_minus_start_<s>` と実験 14 の `imitable_R0` の照合 | 両方の `conditions.csv`（subset = all） | 24 組（4 コーパス × K 2 つ × 3 行）で最大差 1.8 × 10⁻¹⁵ |
| `student` の列の読み替え | `conditions.csv` | `student` は全行 AVT で，`_A`・`_V`・`_T` で終わる量は声・顔・言葉の行の量として読み替える必要がある |

あわせて，`delta_minus_delta_<s>` が「融合の Δ − 実験 14 の Δ_s」と一致することも 24 組で確かめた（最大差 7.1 × 10⁻¹⁵）．

## 指摘

### [記録 1] 単位のログは near_tie の行も「解き直しで同じ予測」と数える
- 重さ：記録する
- 該当箇所：`exp/dai22.py:494`，`logs/dai22_units.log`
- 問題：code_v5 の [記録 1] のとおり，ログの「解き直しで同じ予測 1」は near_tie の行も含む．本番のログでも，予測が 1 本違う MELD f2 r0（near_tie）が「解き直しで同じ予測 1」と出ている．
- 計算の後で直せるか：直せる
- 求める対応：結果の文と限界の節では，ログではなく単位の csv の `resolve_kind` で数える（same 3 行：MELD 1，MOSEI 2．near_tie 4 行：MELD 2，MOSEI 2）．

### [記録 2] same の 1 行で 10⁻¹⁰ の解き直しが基準に届かずに止まった
- 重さ：記録する
- 該当箇所：`results/dai22/units/MOSEI_f0_r1.csv`（K = 25 の R0 の行．`stalled_tol1e-10 = 1`，`grad_tol1e-10 = 2.9e-10`）
- 問題：この行の same の判定は，10⁻⁸ の解（基準に届いた）と，10⁻¹⁰ に届かずに止まった解の両方が単位の予測と同じだったことに拠る．10⁻⁸ の解は基準に届いているので，same の判定そのものは 10⁻⁸ の解だけでも成り立つ．
- 計算の後で直せるか：直せる（csv の列で分かる）
- 求める対応：設計 6 節の規則どおり，限界の節で「same 3 行・near_tie 4 行，そのうち解き直しが基準に届かずに止まった行は same の 1 行（10⁻¹⁰ の解き直し）」と分けて書く．

### [記録 3] 結果の集計と記録が git に入っていない
- 重さ：記録する
- 該当箇所：`results/dai22/`（`git status --ignored` で `!!`）
- 問題：DAI の取り決め（`DAI/CLAUDE.md` の置き場の表）は，結果の集計と記録を `git add -f` で入れるとしている．点検の時点で `summary/`・`provenance.json`・`provenance_log.jsonl`・`check_saved_*.csv` はどれも git に入っていない．値には影響しない．
- 計算の後で直せるか：直せる
- 求める対応：結果の文を書く前に，`tools/collect_results.py` で `results/MANIFEST.tsv` を更新し，集計と記録の 4 種を `git add -f` で入れる（単位の npz は取り決めに従う）．

### [記録 4] conditions.csv の NaN の列は設計どおりで，表を作るときに読み違えない
- 重さ：記録する
- 該当箇所：`results/dai22/summary/conditions.csv` の `v_split`・`v_split_indep`（全行），`df`（MOSEI の 42 行）
- 問題：分け方の揺れを足さない設計（4 節）と MOSEI の引き直しの区間（自由度を使わない）から来る NaN で，欠けではない．
- 計算の後で直せるか：直せる
- 求める対応：表を作る道具が NaN を欠けとして扱って止まらないようにし，区間は `lo`・`hi` を使う（実験 14 の表と同じ列）．

## 目録に無い種類の不備

なし．code_v5 で挙げた 2 案（止まった記録のファイルを通して値が見える，検査の途中で台本が替わる）は，この回では起きなかった（`check_fail_*.csv` は書かれず，台本は検査した 540bba2 の版のまま流れた）．
