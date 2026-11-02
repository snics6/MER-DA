# 関門 1 の指摘の直し（E83，2026-11-03）

対象：`checks/E83/probe.md`（gate-probe．判定「直してから通す」）と `checks/E83/code.md`（gate-code．判定「止める」）の指摘を，設計（`docs/exploratory-analyses.md` の E83 1〜4 節）と台本（`exp/analyze_pairs_e83.py`）に入れた．設計の 5 節（計算と解析の記録）と E84 の節は触っていない．評価のターゲットの値を出す (A)(B) は流していない．動作の確認は，(C) を `--stages c --targets MELD --boot 20 --smoke` で流したもの（24 秒，最大 RSS 0.73 GB，警告なし，出力は scratchpad に置いて消した）と，(B) の経路の関数を人工データ（results/ を読まない）で動かした検算である．`python -m py_compile` は通る．

行番号は直した後の版のもの（設計は `docs/exploratory-analyses.md`，台本は `exp/analyze_pairs_e83.py`）．

## gate-code（止める）

| 番号 | 指摘 | どこをどう直したか |
|---|---|---|
| (1) | (B) の手法の行が ours（除去実験）を引く | 台本：`chosen_setting(sel, run, T, student, K, selector)`（226〜239 行）に一般化し，主の比較は `COMPARISONS` の先頭 `("oursx", "none-oursx")`（54 行）で `selector == "oursx"`，`variant == "incl"` を引く．`chosen_ours` は oursx を呼ぶ薄い包み（241 行）．docstring（1〜14 行）を oursx に直した．設計：10080 行を「selector = oursx，variant = incl（…E79 の Claim 5 の P5-3 が `pick(run, T, st, K, "oursx", "incl")` で引く鍵と同じ）」に直した．台本に `"ours"` の文字列は残っていない |
| (2) | Δe の 100 倍 | 台本 479 行：`res["des"][r] = (xo.acc_model − xo.acc_actual) − (xn.acc_model − xn.acc_actual)`（100 を掛けない．注「すでにポイント（I7）」）．設計 10086 行に「pairs の acc はすでにポイントなので 100 は掛けない」と書いた |

## gate-code（直してから進む）

| 番号 | 指摘 | どこをどう直したか |
|---|---|---|
| (3) | (A) の融合の実行の二重読み | 台本 281 行：`for mdir, runs in ((e79/measure, RUNS), (e79_p4/measure, KINDS))` で，e79_p4 からは KINDS だけを読む．`a_none_pairs_rep.csv` に `source` 列（どの場所から読んだか）を足し，(run, target, student, K, rep, pair) の重複があれば止まる（303〜304 行）．設計 10073 行に書いた |
| (4) | (C) と P3c に融合の f が無い | 台本 631 行：`for st in mods + [run]` で `none__<実行>__<K>` を f に足した（g は other_m のすべて，pF，poe0，ref，ref0）．P3c（816〜826 行）は role を限らず，scope = all（主）・uni・fused・nomd の 4 行を出す．(C) の試走で f は 5 つ（音声・顔・言葉・AVT 融合・AV 融合），融合の f の g は AVT 7 つ・AV 6 つになった．設計 10092 行と 10108 行に書いた |

## gate-code（記録）

| 番号 | 指摘 | どこをどう直したか |
|---|---|---|
| (5) | rev_order と iemocap_impro_only の感度の行 | 台本：`dir_diff_rev`（178 行．100·{[A(D,o_none) − A(d_none,o_none)] − [A(D,o_手法) − A(d_手法,o_手法)]}．E79 の `dir_rev` の none − 手法と同じ符号），`pair_d_o_from_feats`（142 行．抽出した重みから d と o を出す．E79 の `boot_rep` と同じ式）．反復ごとに `revs`・`boot_rev`・`jk_rev` を作り（480・500 行），D は固定．`specs`（436〜438 行）で main・rev_order（metric = dir_rev）・iemocap_impro_only（AVT だけ．Ts から IEMOCAP-script を除く．乱数は主と同じコーパスの話者で引いたもの）の 3 つの集め方を `aggregate_unit`（509 行）に渡す．逆の順序の点も対数の確率から計算し直して 1e−6 で照合する（489〜492 行）．模型の印は dir だけ（E79 と同じ）．判定の数の表を `b_judgments_by_pair_rev_order.csv`・`_iemocap_impro_only.csv` に出す（784 行）．設計 10087 行に書いた |
| (6) | abl_own（と kd）の行，文の規則 (3) | 台本 54 行 `COMPARISONS` に `("kd", "none-kd")`，`("abl_own", "none-abl_own")` を足し，同じ `pair_diffs_target`・`aggregate_unit` で対ごとに出す（`comparison` 列）．none を選んだ群は差 0 の複製（E79 と同じ．463〜470 行），全群 none なら same_by_definition．表は `b_judgments_by_pair_kd.csv`・`_abl_own.csv`（786〜787 行）．P3a・P3c の判定は `MAIN_COMP`（none-oursx）だけで行う．設計 10081 行（補助の比較）と 10111 行の文の規則 (3)（「手法が減らした対と，候補の上で g が補う対が一致する」まで．KD・abl_own で同じ対が減るならほかのモダリティの情報には帰せない），10122 行に書いた |
| (7) | 単位ごとの複製の保存 | 台本 546 行：`boot_store["<comparison>|<run>|<f>|<K>|<corpus>|<sensitivity>"] = b.astype(float32)`（(B, 6)．main・rev_order・iemocap_impro_only）．317〜319 行で `b_boot_units.npz` と索引 `b_boot_units_index.csv` に書く．設計 10083 行に書いた |
| (8) | provenance に読んだ npz の大きさと時刻 | 台本 68〜80 行 `note_input(prov, path, with_sha)`：すべての入力について bytes と mtime（ISO）を書き，CSV（chosen.csv，<T>.csv，_samples.csv，pairs_*.csv.gz）は sha256 も書く．npy・npz（logp，rowids，te，_ps_，E80 の units）は大きさと時刻だけ．started・finished も書く（860・867 行）．dirty の扱いはそのまま（本番の前にコミットする） |
| (9) | IEMOCAP の 2 群で選択が違う | 台本 424〜428 行：群ごとに `chosen_setting` を引き，集合の大きさが 2 以上なら `chosen_differs_in_corpus` を checks.csv に書き，主の比較（none-oursx）なら SystemExit．kd・abl_own は記録して群ごとに続ける（none の群は差 0）．設計 10080・10081 行に書いた |
| (10) | 設計の数の誤記 | 設計 10079 行：「AVT の 4 つの f × K 3 × コーパス 3 ＝ 36 単位，AV の 3 つの f × K 3 × コーパス 5 ＝ 45 単位．計 81 単位 × 6 対 ＝ 486」．10091 行：「（T，c）の組は AVT 8 組（IEMOCAP-impro 2，MELD 3，MOSEI 3），AV 24 組．各 5 反復」．10110 行：融合の偶然の見込み 3.6（24 単位） |
| (11) | 判定の数の表の列の固定 | 台本 657〜666 行 `judgment_table`：`reindex(columns=JUDG_COLS, fill_value=0)`（JUDG_COLS = pos, neg, same, undet, missing, same_by_definition．56 行）．空の表でも同じ列を出す（人工データで確認） |
| (12) | agree は pF と同じ判定 | 台本 605〜609 行：preds から agree を外した（注を 608 行に）．docstring 12 行と設計 10092 行に「pF に行ごとの正の重みを掛けたもので，2 感情の判定も 4 クラスの argmax も pF と同じになるので出さない」と書いた |
| (13) | 1e−6 の照合が NaN の食い違いを見逃す | 台本 183〜189 行 `max_abs_or_nan_mismatch`：片方だけ NaN の対の数を数え，1 つでもあれば止まる（492 行）．人工データで，d をずらした場合と NaN にした場合の両方で止まることを確かめた．設計 10082 行に書いた |

## gate-probe（直してから進む）

| 番号 | 指摘 | どこをどう直したか |
|---|---|---|
| (14) | P3a の数の条件を塊で数える | 台本 712〜719 行：（コーパス，対）に畳み（`pos_any`），`n_blocks`・`n_pos_blocks`・`chance_pos_blocks = n_blocks × 0.025`・`n_pos_corpora`・`pos_corpora`（`pos_pairs` と同じ形）を出す．条件 `cond_count_blocks`（塊の数 > 塊の偶然の見込み）と `cond_two_corpora`（2 つ以上のコーパス）を `P3a_holds` に入れた（728〜752 行）．参考に `chance_pos_blocks_anyof`（塊ごとの 1 − 0.975^m の和）も出す．設計 10100〜10102 行（(i)(ii)），10110 行（偶然の見込み (c)） |
| (15) | P3a の土台を条件つきに替え，0.5 と並べる．見出しに E13 の断り | 台本 722〜723 行：`elig`（none の向きの損 ≥ δ_d,対，手法が none を選ばなかった，判定が missing でない（単位，対））のうち選んだ対の割合 `base_share_cond`（`n_eligible_cond` も）．`P3a_holds` は条件つきの土台，`P3a_holds_base_fixed` は 0.5（顔は 1/6）で判定する（751〜752 行）．設計 10100 行の見出しに「E13 は同じコーパスの none の損がどの対にあるかの記録なので，P3a は確認の予言ではない」，10103 行に (iii)，10118 行にも書いた |
| (16) | 割合の偶然の分布 | 台本 668〜688 行 `perm_share_null`：コーパスごとに 6 対の名前の並べ替えを 1 つ引き，その塊の全行に当てる（判定は固定）．1,000 回（`--perm`．smoke では 100）．種は `[PERM_SEED, crc32(族), 印の有無]` で固定．95% 点 `perm_q95` と `perm_p` を表に出し，`cond_share_gt_perm_q95` を `P3a_holds` に入れた．分布は `b_p3a_perm.npz` に残す．設計 10104 行 (iv) |
| (17) | 9.1 の区分：本番 | 設計 10116 行：「本番として扱う．(A)(B) は…(B) の結果は論文の主張に入れる．関門 1 は 4 体（gate-stats・gate-leak・gate-compare・gate-define）と gate-code．gate-probe の報告は下調べとしての先行の検査」．`docs/seen-data.md` 26 行の E83 の行を「本番（…関門 1 は 5 体．gate-probe の報告は下調べとしての先行の検査…）」に直した．台本 docstring 3 行にも書いた |

## gate-probe（記録）

| 番号 | 指摘 | どこをどう直したか |
|---|---|---|
| (18) | 喜び–怒りの帰属 | 台本 46〜48 行：`P3A_SIDE_PAIRS`（平静–怒り，悲しみ–怒り），`P3B_SIDE_PAIRS`（平静–悲しみ），`HAPPY_ANGER`．表に `share_p3a_side_among_pos`・`share_p3b_side_among_pos`・`n_pos_happy_anger`（745〜747 行）．P3a の (iii) の集合は登録どおり怒りを含む 3 対のまま．設計 10107 行に帰属の規則（喜び–怒りはどちらの側にも数えず，減った対として名前だけを書く） |
| (19) | 音声の f の主の版，(A) の d_AUC と模型の誤差の列 | 台本 734 行 `is_main_version`（音声・言葉・融合は印を含む版，顔は除く版）．(A) は `a_rows_from`（257〜272 行）で `none_dirloss_auc = 100·[A(D,0) − A(d_AUC,0)]` と `none_model_err = acc_model − acc_actual` を反復ごとに出し，`a_none_pairs.csv`・`a_summary_K0.csv` にも載せる．設計 10074 行と 10106 行 |
| (20) | none を選んだ単位を除いた検定の数，P3c の除外 | 台本 706〜709 行：`n_units_nonone`・`n_tests_nonone`・`chance_pos_nonone`．P3c（816〜826 行）：`chosen == "none"` と J が NaN の行を土台と割合の両方から除き，`n_excluded_none_chosen`・`n_excluded_J_nan` を書く．設計 10108・10110 行 |
| (21) | P3c の J の読み方 | 設計 10121 行：「P3c の J は評価のコーパス自身のラベル（同じ話者．同じ反復のテストの発話は含まないが，ほかの反復のテストの発話は含む）で作るので，手法の選び方に使える量の予言としては読まない．J は g を作った候補の上の値であり，ref・poe0 は候補に合わせた分だけ g に有利に出うる」 |
| (22) | (A) の反復ごとの行とターゲットの範囲 | 台本：反復ごとの行を `a_none_pairs_rep.csv`（300 行），`a_none_pairs.csv` に `A_D0_rep_med/min/max`・`none_dirloss_rep_med/min/max`（305〜310 行），`a_summary_K0.csv`（と全 K の `a_summary_by_K.csv`）にターゲットの中央値・最小・最大・四分位（764〜775 行）．設計 10075 行 |
| (23) | δ のコーパスの中央値を先に取った列，NaN・Inf の確認 | 台本 323〜367 行 `delta_pairs`：対数の確率の NaN/Inf（342 行），A(d,0) の NaN/Inf（351 行），反復が 2 つ以上ある (T,c) が無い場合（355 行），δ_d,対 と `delta_d_pair_corpus` の NaN/Inf（365〜366 行）で止まる．`delta_d_pair_corpus`（E79 の `delta_d_corpus` と同じ作り）と `pairs_by_corpus` を `b_delta_pairs.csv` に出す．設計 10084 行 |

## 直せなかったもの・入れなかったもの

- `docs/methods.md` 9.3 節の目録への追加（gate-probe の L8・S8・D12 案，gate-code の I1・I14 の追記案）は入れていない．今回の依頼の範囲は設計と台本で，9.3 節には E84 の関門で同じ番号（S8・L8・D12）がすでに使われているため，番号を振り直す判断は本体に任せる（案：S11「テストの発話を共有する単位は塊で数える」，L10「保存済みの評価のターゲットの値を新しい切り口で判定し直す解析は本番」，D14「カテゴリの偏りは条件つきの土台と塊の中の並べ替えで比べる」，I1 の追記「設計の中の同じ項目の書き方の食い違いを照合する」，I14 の追記「上流の保存の列の単位（割合かポイントか）を読む」）．
- (A)(B) の評価のターゲットでの試走はしていない（本番の前にコミットが要る）．(B) の経路は人工データで，主・逆の順序・即興だけの 3 つの集め方，重み 1 の複製が点に戻ること，none の群が差 0 になること，d をずらすと止まること，NaN の食い違いで止まること，表の列の固定，塊の数え方，条件つきの土台，並べ替え，P3c の除外を確かめた．E79 の保存した複製との照合（gate-code の boot の検算）は，再検査で `checks/E83/code_run/check_e83.py` を今の台本に合わせて流し直す必要がある（`chosen_ours` の引数と `pair_diffs_target` の形が変わった．`pair_feats`・`pair_d_from_feats`・`boot_draws`・`load_labels_speakers`（prov が要る）は残している）．
- gate-probe の (14) の偶然の見込みは指摘どおり「塊の数 × 0.025」を判定に使い，塊ごとの 1 − 0.975^m の和は参考の列にとどめた．

## 本番の規模の見込み

gate-code の見積もり（(A)(B) 56 秒，(C) 29 秒）に対し，(B) は比較が 3 つ（oursx・kd・abl_own）と感度の集め方が 3 つに増えたので，(A)(B) はおよそ 3 倍（2〜3 分）．(C) は融合の f の分で 1.4 倍（試走 24 秒）．GPU は使わない．

## git diff --stat（直した後．コミットはしていない）

```
 docs/exploratory-analyses.md |  52 ++--
 docs/seen-data.md            |   2 +-
 exp/analyze_pairs_e83.py     | 691 +++++++++++++++++++++++++++++--------------
 3 files changed, 507 insertions(+), 238 deletions(-)
```
